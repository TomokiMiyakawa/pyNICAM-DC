#!/usr/bin/env python3
"""
gather_pack_unpack_spike.py  --  comm-replace-plan_v4.txt  §1 DE-RISK spike.

Proves the GATHER-based halo pack/unpack (the FESOM-jax pattern) is
  (A) bit-identical to the current SCATTER-based pack/unpack in
      mod_comm._comm_data_transfer_shardmap  (numpy V9 unit -- runs anywhere), and
  (B) compiles fast under jax.shard_map + ragged_all_to_all, where the scatter
      version blows up XLA's SPMD partitioner (the inc3 root cause, §3).

Part A (default): a self-contained numpy simulation of ALL ranks of a toy
ragged halo exchange with REALISTIC 5D fancy index maps (gi / ikv / si), the
local copies (r2r / p2r / r2p) and singular, plus var + var_pl sources/targets
-- i.e. every index-map shape the real _build_comm_plan_device produces. It runs
the full pack -> ragged -> unpack pipeline twice (scatter vs gather) and asserts
the two agree EXACTLY, for every rank. This validates the perm/mask construction
we will port into _build_ragged_layout.

Part B (--jax, needs GPU + MPI, launched by the .pbs): the same toy under a
whole-step jax.shard_map with a real lax.ragged_all_to_all, AOT-lowered and
compiled both ways, printing per-rank COMPILED + wall time.

  numpy check :  python gather_pack_unpack_spike.py            # login node OK
  gpu compile :  mpirun -np 4 python gather_pack_unpack_spike.py --jax [--scatter]

The perm/mask construction here is the reference implementation for §1(a).
"""
import argparse
import numpy as np


# ============================================================================
#  TOY PLAN  --  mirrors _build_comm_plan_device's op dicts + _build_ragged_layout
# ============================================================================
# var    : (I, J, K, L, V)  region field with halo   (flattened for pack/unpack)
# var_pl : (Ipl, K, Lpl, V) pole field
# A send op:  dict(src='var'|'var_pl', n, gi=(idx-arrays into src), ikv=perm[n])
# A recv op:  dict(tgt='var'|'var_pl', n, si=(idx-arrays into tgt), ikv=perm[n])
# ragged sizes S[i][j] = #cells rank i sends rank j  (must match j's recv from i).

DIMS_VAR = (18, 18, 40, 1, 6)      # ~gl05 region field
DIMS_PL  = (6, 40, 2, 6)           # small pole field


def _rng(*key):
    # deterministic per-key RNG so sender & receiver of a pair agree on sizes
    h = 0
    for k in key:
        h = (h * 1000003 + int(k)) & 0x7FFFFFFF
    return np.random.default_rng(h)


def _rand_multi_index(rng, shape, n):
    return tuple(rng.integers(0, s, size=n, dtype=np.int64) for s in shape)


def _rand_unique_multi_index(rng, shape, n):
    """n DISTINCT cells of `shape` (halo targets don't collide within one op)."""
    size = int(np.prod(shape))
    n = min(n, size)
    flat = rng.choice(size, size=n, replace=False)
    return tuple(np.asarray(a, dtype=np.int64) for a in np.unravel_index(flat, shape))


def build_global_plan(nproc, ops_per_pair=2, pole_frac=0.25, seed=0):
    """Build the complete cross-rank plan once (we simulate all ranks in-process).

    Returns plan[(i,j)] = list of ops, each with sender-side (src/gi/ikv) and
    receiver-side (tgt/si/ikv) index maps + size n. Sizes are pair-deterministic
    so S is consistent from both the send and recv side.
    """
    plan = {}
    for i in range(nproc):
        for j in range(nproc):
            if i == j:
                continue
            prng = _rng(seed, 777, i, j)                 # shared: op count + sizes
            nops = int(prng.integers(1, ops_per_pair + 1))
            ops = []
            for o in range(nops):
                n = int(prng.integers(200, 900))         # ~ edge*K*V block
                use_pl_src = prng.random() < pole_frac
                use_pl_tgt = prng.random() < pole_frac
                srng = _rng(seed, 111, i, j, o)          # sender-private maps
                rrng = _rng(seed, 222, i, j, o)          # receiver-private maps
                if use_pl_src:
                    gi = _rand_multi_index(srng, DIMS_PL, n); src = 'var_pl'
                else:
                    gi = _rand_multi_index(srng, DIMS_VAR, n); src = 'var'
                if use_pl_tgt:
                    si = _rand_unique_multi_index(rrng, DIMS_PL, n); tgt = 'var_pl'
                else:
                    si = _rand_unique_multi_index(rrng, DIMS_VAR, n); tgt = 'var'
                n = len(si[0])                           # unique-index may shrink n
                gi = tuple(a[:n] for a in gi)
                ikv_s = srng.permutation(n).astype(np.int64)
                ikv_r = rrng.permutation(n).astype(np.int64)
                # some ops fill only a SUBSET of their block (partial ikv coverage):
                # sb = zeros(n).at[ikv].set(...) leaves the uncovered rows ZERO. The
                # gather path must reproduce that zero-fill (send_sel stays 0 there).
                if srng.random() < 0.3 and n > 4:
                    k = int(srng.integers(1, n))         # cover only k of n rows
                    ikv_s = ikv_s[:k]; gi = tuple(a[:k] for a in gi)
                ops.append(dict(n=n, src=src, tgt=tgt, gi=gi, si=si,
                                ikv_s=ikv_s, ikv_r=ikv_r))
            plan[(i, j)] = ops
    return plan


def rank_sends_recvs(plan, nproc, me):
    """Per-rank sends/recvs in the canonical (dst/src, tag) order used by
    _build_ragged_layout (here: destination/source rank order == tag order)."""
    sends = []
    for d in range(nproc):
        if d == me:
            continue
        for x in plan[(me, d)]:
            sends.append(dict(dst=d, src=x['src'], n=x['n'], gi=x['gi'], ikv=x['ikv_s']))
    recvs = []
    for s in range(nproc):
        if s == me:
            continue
        for x in plan[(s, me)]:
            recvs.append(dict(src_rank=s, tgt=x['tgt'], n=x['n'], si=x['si'], ikv=x['ikv_r']))
    return sends, recvs


def build_S(plan, nproc):
    S = np.zeros((nproc, nproc), dtype=np.int64)
    for (i, j), ops in plan.items():
        S[i, j] = sum(o['n'] for o in ops)
    return S


def ragged_layout(sends, recvs, S, me, nproc):
    """The offsets _build_ragged_layout computes, plus the pack/unpack row lists."""
    send_sizes = S[me, :]
    recv_sizes = S[:, me]
    input_offsets = np.zeros(nproc, np.int64)
    input_offsets[1:] = np.cumsum(send_sizes)[:-1]
    recv_off = np.zeros(nproc, np.int64)
    recv_off[1:] = np.cumsum(recv_sizes)[:-1]
    rag_pack = []
    for d in range(nproc):
        off = int(input_offsets[d])
        for x in [s for s in sends if s['dst'] == d]:
            rag_pack.append((x, off)); off += x['n']
    rag_unpack = []
    for s in range(nproc):
        off = int(recv_off[s])
        for x in [r for r in recvs if r['src_rank'] == s]:
            rag_unpack.append((x, off)); off += x['n']
    operand_rows = int(send_sizes.sum())
    output_rows = int(recv_sizes.sum())
    return dict(rag_pack=rag_pack, rag_unpack=rag_unpack,
                operand_rows=operand_rows, output_rows=output_rows,
                recv_off=recv_off, input_offsets=input_offsets)


def build_local_copies(rng, has_pl):
    """A representative copy_r2r + singular (index remaps within var). Kept as
    scatters in BOTH pipelines (they are the FEW ops §1 may fold later)."""
    nc = 40
    src = _rand_multi_index(rng, DIMS_VAR, nc)
    dst = _rand_unique_multi_index(rng, DIMS_VAR, nc)
    nc = len(dst[0]); src = tuple(a[:nc] for a in src)
    copy_r2r = (src, dst)
    ns = 12
    ssrc = _rand_multi_index(rng, DIMS_VAR, ns)
    sdst = _rand_unique_multi_index(rng, DIMS_VAR, ns)
    ns = len(sdst[0]); ssrc = tuple(a[:ns] for a in ssrc)
    singular = (ssrc, sdst)
    return copy_r2r, singular


# ============================================================================
#  SCATTER reference  (numpy mirror of _comm_data_transfer_shardmap)
# ============================================================================
def scatter_pack(var, var_pl, rag, mo, dtype):
    operand = np.zeros(mo, dtype)
    for (s, off) in rag['rag_pack']:
        srcarr = var if s['src'] == 'var' else var_pl
        sb = np.zeros(s['n'], dtype)
        sb[s['ikv']] = srcarr[s['gi']]
        operand[off:off + s['n']] = sb
    return operand


def scatter_unpack(var, var_pl, recvd, rag, copy_r2r, singular):
    var = var.copy(); var_pl = var_pl.copy()
    recv_arrs = {id(r): recvd[off:off + r['n']] for (r, off) in rag['rag_unpack']}
    # local copy (r2r): within var
    src, dst = copy_r2r
    var[dst] = var[src]
    # recv scatters
    for (r, _off) in rag['rag_unpack']:
        vals = recv_arrs[id(r)][r['ikv']]
        if r['tgt'] == 'var':
            var[r['si']] = vals
        else:
            var_pl[r['si']] = vals
    # singular (last)
    ssrc, sdst = singular
    var[sdst] = var[ssrc]
    return var, var_pl


# ============================================================================
#  GATHER rewrite  (the FESOM pattern -- the §1 reference implementation)
# ============================================================================
def _copy_perm(src, dst, size):
    """A within-array copy `arr[dst] = arr[src]` as a gather perm/mask over flat `arr`:
    perm[flat_dst]=flat_src, mask[flat_dst]=True -> arr = where(mask, arr[perm], arr)."""
    perm = np.arange(size, dtype=np.int64)          # identity default (self-gather = no-op)
    mask = np.zeros(size, bool)
    flat_dst = np.ravel_multi_index(dst, DIMS_VAR)
    perm[flat_dst] = np.ravel_multi_index(src, DIMS_VAR)
    mask[flat_dst] = True
    return perm, mask


def build_gather_perms(rag, mo, var_size, pl_size, copy_r2r=None, singular=None):
    """Precompute, ONCE at setup, the flat perm/mask arrays so pack/unpack become
    pure gathers + a masked select -- NO scatters at all (fancy-indexed .at[].set()
    blows up XLA's GPU SPMD partitioner under shard_map; the diagnostic spike proved
    even a ~40-cell copy scatter hangs the compile). Inverse of the scatter index maps.

    send_perm_var/pl : int[mo]   flat src index feeding each operand row
    send_sel         : int8[mo]  0=pad(->0), 1=var, 2=var_pl
    recv_perm_var    : int[var_size]  flat recvd index for each var cell
    recv_mask_var    : bool[var_size] True where a recv writes the cell
    recv_perm_pl / recv_mask_pl : same for var_pl
    cr2r_perm/mask, sing_perm/mask : the local copy_r2r + singular folded to gathers
    """
    send_perm_var = np.zeros(mo, np.int64)
    send_perm_pl = np.zeros(mo, np.int64)
    send_sel = np.zeros(mo, np.int8)
    for (s, off) in rag['rag_pack']:
        rows = off + s['ikv']                       # operand rows this op fills
        if s['src'] == 'var':
            send_perm_var[rows] = np.ravel_multi_index(s['gi'], DIMS_VAR)
            send_sel[rows] = 1
        else:
            send_perm_pl[rows] = np.ravel_multi_index(s['gi'], DIMS_PL)
            send_sel[rows] = 2

    recv_perm_var = np.zeros(var_size, np.int64)
    recv_mask_var = np.zeros(var_size, bool)
    recv_perm_pl = np.zeros(pl_size, np.int64)
    recv_mask_pl = np.zeros(pl_size, bool)
    for (r, off) in rag['rag_unpack']:
        src_rows = off + r['ikv']                   # recvd rows read by this op
        if r['tgt'] == 'var':
            flat_dst = np.ravel_multi_index(r['si'], DIMS_VAR)
            recv_perm_var[flat_dst] = src_rows
            recv_mask_var[flat_dst] = True
        else:
            flat_dst = np.ravel_multi_index(r['si'], DIMS_PL)
            recv_perm_pl[flat_dst] = src_rows
            recv_mask_pl[flat_dst] = True

    out = dict(send_perm_var=send_perm_var, send_perm_pl=send_perm_pl, send_sel=send_sel,
               recv_perm_var=recv_perm_var, recv_mask_var=recv_mask_var,
               recv_perm_pl=recv_perm_pl, recv_mask_pl=recv_mask_pl)
    if copy_r2r is not None:
        out['cr2r_perm'], out['cr2r_mask'] = _copy_perm(copy_r2r[0], copy_r2r[1], var_size)
    if singular is not None:
        out['sing_perm'], out['sing_mask'] = _copy_perm(singular[0], singular[1], var_size)
    return out


def gather_pack(var, var_pl, g):
    vf = var.reshape(-1); pf = var_pl.reshape(-1)
    ov = vf[g['send_perm_var']]
    op = pf[g['send_perm_pl']]
    operand = np.where(g['send_sel'] == 1, ov,
                       np.where(g['send_sel'] == 2, op, 0)).astype(var.dtype)
    return operand


def gather_unpack(var, var_pl, recvd, g):
    # ZERO scatters -- copies + recv + singular are all gather+where over flat arrays,
    # applied in the SAME order as the scatter reference (copy_r2r -> recv -> singular).
    vf = var.reshape(-1).copy()
    vf = np.where(g['cr2r_mask'], vf[g['cr2r_perm']], vf)          # copy_r2r (before recv)
    vf = np.where(g['recv_mask_var'], recvd[g['recv_perm_var']], vf)   # recv into var
    pf = var_pl.reshape(-1).copy()
    pf = np.where(g['recv_mask_pl'], recvd[g['recv_perm_pl']], pf)     # recv into var_pl
    vf = np.where(g['sing_mask'], vf[g['sing_perm']], vf)          # singular (after recv)
    return vf.reshape(DIMS_VAR), pf.reshape(DIMS_PL)


# ============================================================================
#  PART A  --  numpy equivalence over ALL ranks
# ============================================================================
def numpy_check(nproc=4, seed=0, verbose=True):
    dtype = np.float64
    plan = build_global_plan(nproc, seed=seed)
    S = build_S(plan, nproc)
    mo = int(S.sum(axis=1).max())          # global max operand rows (uniform shard)
    mout = int(S.sum(axis=0).max())        # global max output rows

    # per-rank state
    vars = {r: _rng(seed, 9, r).standard_normal(DIMS_VAR).astype(dtype) for r in range(nproc)}
    var_pls = {r: _rng(seed, 8, r).standard_normal(DIMS_PL).astype(dtype) for r in range(nproc)}
    layouts, copies, perms, operands_sc, operands_ga = {}, {}, {}, {}, {}

    # (1) pack on every rank, scatter vs gather
    for me in range(nproc):
        sends, recvs = rank_sends_recvs(plan, nproc, me)
        rag = ragged_layout(sends, recvs, S, me, nproc)
        layouts[me] = rag
        crng = _rng(seed, 5, me)
        cr2r, sing = build_local_copies(crng, has_pl=True)
        copies[me] = (cr2r, sing)
        g = build_gather_perms(rag, mo, int(np.prod(DIMS_VAR)), int(np.prod(DIMS_PL)),
                               copy_r2r=cr2r, singular=sing)
        perms[me] = g
        o_sc = scatter_pack(vars[me], var_pls[me], rag, mo, dtype)
        o_ga = gather_pack(vars[me], var_pls[me], g)
        operands_sc[me] = o_sc
        operands_ga[me] = o_ga
        assert np.array_equal(o_sc, o_ga), f"PACK mismatch rank {me}"

    # (2) ragged exchange (numpy model of lax.ragged_all_to_all) -> recvd per rank
    def do_exchange(operands):
        recvds = {}
        for d in range(nproc):
            recvd = np.zeros(mout, dtype)
            rag = layouts[d]
            for s in range(nproc):
                if s == d:
                    continue
                n = int(S[s, d])
                if n == 0:
                    continue
                in_off = int(layouts[s]['input_offsets'][d])
                out_off = int(rag['recv_off'][s])
                recvd[out_off:out_off + n] = operands[s][in_off:in_off + n]
            recvds[d] = recvd
        return recvds
    recvds_sc = do_exchange(operands_sc)
    recvds_ga = do_exchange(operands_ga)
    assert all(np.array_equal(recvds_sc[d], recvds_ga[d]) for d in range(nproc))

    # (3) unpack on every rank, scatter vs gather -> compare final state
    max_abs = 0.0
    for me in range(nproc):
        rag = layouts[me]; cr2r, sing = copies[me]
        v_sc, vpl_sc = scatter_unpack(vars[me], var_pls[me], recvds_sc[me], rag, cr2r, sing)
        v_ga, vpl_ga = gather_unpack(vars[me], var_pls[me], recvds_ga[me], perms[me])
        d1 = np.max(np.abs(v_sc - v_ga)) if v_sc.size else 0.0
        d2 = np.max(np.abs(vpl_sc - vpl_ga)) if vpl_sc.size else 0.0
        max_abs = max(max_abs, float(d1), float(d2))
        assert np.array_equal(v_sc, v_ga), f"UNPACK var mismatch rank {me} (max|d|={d1})"
        assert np.array_equal(vpl_sc, vpl_ga), f"UNPACK var_pl mismatch rank {me} (max|d|={d2})"
        # sanity: the exchange actually moved data (var changed vs input)
        assert not np.array_equal(v_sc, vars[me]), f"rank {me}: unpack was a no-op?!"

    if verbose:
        print(f"[numpy V9] nproc={nproc} seed={seed}  mo={mo} mout={mout}  "
              f"S.sum={int(S.sum())}  max|scatter-gather|={max_abs:.1e}  -> BIT-IDENTICAL", flush=True)
    return True


# ============================================================================
#  PART B  --  jax.shard_map + ragged_all_to_all compile test (GPU + MPI)
# ============================================================================
def make_step_fns(plan, S, me, nproc, seed, jnp, ragged_exchange, dtype=np.float64):
    """Device-agnostic builder for the two shard_map step bodies + inputs.

    ragged_exchange(operand)->recvd is INJECTED: the real lax.ragged_all_to_all on
    GPU, or a static pad/truncate stub for the CPU unit test (CPU has no ragged
    opcode). Returns everything the compile test / unit test needs. This is the
    exact pack/unpack graph that will land in mod_comm; keeping it in one place
    lets the CPU unit prove it traces before the GPU job."""
    mo = int(S.sum(axis=1).max()); mout = int(S.sum(axis=0).max())
    sends, recvs = rank_sends_recvs(plan, nproc, me)
    rag = ragged_layout(sends, recvs, S, me, nproc)
    cr2r, sing = build_local_copies(_rng(seed, 5, me), has_pl=True)
    g = build_gather_perms(rag, mo, int(np.prod(DIMS_VAR)), int(np.prod(DIMS_PL)),
                           copy_r2r=cr2r, singular=sing)
    var0 = _rng(seed, 9, me).standard_normal(DIMS_VAR).astype(dtype)
    vpl0 = _rng(seed, 8, me).standard_normal(DIMS_PL).astype(dtype)
    jdt = jnp.asarray(np.empty(0, dtype)).dtype

    dev_pack = [(dict(src=s['src'], n=s['n'], ikv=jnp.asarray(s['ikv']),
                      gi=tuple(jnp.asarray(a) for a in s['gi'])), off)
                for (s, off) in rag['rag_pack']]
    dev_unpack = [(dict(tgt=r['tgt'], n=r['n'], ikv=jnp.asarray(r['ikv']),
                        si=tuple(jnp.asarray(a) for a in r['si'])), off)
                  for (r, off) in rag['rag_unpack']]
    # int32 indices: XLA:GPU gather with s64 indices can hit a slow/pathological compile;
    # the model's ragged layout already uses int32. masks stay bool, send_sel stays int8.
    def _dev(k, v):
        if v.dtype == np.bool_ or v.dtype == np.int8:
            return jnp.asarray(v)
        return jnp.asarray(v.astype(np.int32))
    gj = {k: _dev(k, v) for k, v in g.items()}
    d_cr2r = (tuple(jnp.asarray(a) for a in cr2r[0]), tuple(jnp.asarray(a) for a in cr2r[1]))
    d_sing = (tuple(jnp.asarray(a) for a in sing[0]), tuple(jnp.asarray(a) for a in sing[1]))

    def step_scatter(var, var_pl):   # the CURRENT mod_comm pack/unpack (fancy scatters)
        # shard_map keeps the size-1 mapped axis; drop it so var is the 5D LOCAL shard
        # (mirrors the model's step wrapper: "leading process dim already squeezed").
        var = var[0]; var_pl = var_pl[0]
        operand = jnp.zeros(mo, jdt)
        for (s, off) in dev_pack:
            srcarr = var if s['src'] == 'var' else var_pl
            sb = jnp.zeros(s['n'], jdt).at[s['ikv']].set(srcarr[s['gi']])
            operand = operand.at[off:off + s['n']].set(sb)
        recvd = ragged_exchange(operand)
        recv_arrs = {id(r): recvd[off:off + r['n']] for (r, off) in dev_unpack}
        s_, d_ = d_cr2r
        var = var.at[d_].set(var[s_])
        for (r, _o) in dev_unpack:
            vals = recv_arrs[id(r)][r['ikv']]
            if r['tgt'] == 'var':
                var = var.at[r['si']].set(vals)
            else:
                var_pl = var_pl.at[r['si']].set(vals)
        s_, d_ = d_sing
        var = var.at[d_].set(var[s_])
        return var[None], var_pl[None]

    def step_gather(var, var_pl):    # the §1 FESOM-style pack/unpack -- ZERO scatters
        var = var[0]; var_pl = var_pl[0]
        vf = var.reshape(-1); pf = var_pl.reshape(-1)
        ov = vf[gj['send_perm_var']]; op = pf[gj['send_perm_pl']]
        operand = jnp.where(gj['send_sel'] == 1, ov,
                            jnp.where(gj['send_sel'] == 2, op, 0)).astype(jdt)
        recvd = ragged_exchange(operand)
        # unpack: copy_r2r -> recv -> singular, all gather+where over flat arrays (NO .at[].set)
        vf = jnp.where(gj['cr2r_mask'], vf[gj['cr2r_perm']], vf)
        vf = jnp.where(gj['recv_mask_var'], recvd[gj['recv_perm_var']], vf)
        pf = jnp.where(gj['recv_mask_pl'], recvd[gj['recv_perm_pl']], pf)
        vf = jnp.where(gj['sing_mask'], vf[gj['sing_perm']], vf)
        return vf.reshape(DIMS_VAR)[None], pf.reshape(DIMS_PL)[None]

    def step_packonly(var, var_pl):  # BISECT: only the SEND gather; trivial (static) writeback
        var = var[0]; var_pl = var_pl[0]
        vf = var.reshape(-1); pf = var_pl.reshape(-1)
        ov = vf[gj['send_perm_var']]; op = pf[gj['send_perm_pl']]
        operand = jnp.where(gj['send_sel'] == 1, ov,
                            jnp.where(gj['send_sel'] == 2, op, 0)).astype(jdt)
        recvd = ragged_exchange(operand)
        vf = vf.at[:mout].set(recvd[:mout] * 0 + vf[:mout])   # static slice, no fancy gather
        return vf.reshape(DIMS_VAR)[None], var_pl[None]

    def step_recvonly(var, var_pl):  # BISECT: only the UNPACK gathers; trivial (static) pack
        var = var[0]; var_pl = var_pl[0]
        vf = var.reshape(-1); pf = var_pl.reshape(-1)
        operand = vf[:mo].astype(jdt)                          # static slice, no fancy gather
        recvd = ragged_exchange(operand)
        vf = jnp.where(gj['cr2r_mask'], vf[gj['cr2r_perm']], vf)
        vf = jnp.where(gj['recv_mask_var'], recvd[gj['recv_perm_var']], vf)
        pf = jnp.where(gj['recv_mask_pl'], recvd[gj['recv_perm_pl']], pf)
        vf = jnp.where(gj['sing_mask'], vf[gj['sing_perm']], vf)
        return vf.reshape(DIMS_VAR)[None], pf.reshape(DIMS_PL)[None]

    return dict(step_scatter=step_scatter, step_gather=step_gather,
                step_packonly=step_packonly, step_recvonly=step_recvonly,
                var0=var0, vpl0=vpl0, mo=mo, mout=mout, rag=rag)


def _fesom_rowgather_test(jax, jnp, lax, mesh, me, nproc, seed, dtype,
                          use_gather=True, use_ragged=True, idx_as_input=False, tag='ROWGATHER'):
    """FESOM-exact leading-axis (row) gather over [Ncells, K, V]. Mirrors
    fesom_jax/halo.py halo_exchange_ragged: operand=field[send_idx];
    ragged_all_to_all; gathered=recv[recv_gather]; where(mask[...,None,None], ...).

    Disambiguation flags (rowgather hung with BOTH on -- isolate which):
      use_gather=False -> static-slice pack/unpack (NO fancy gather); isolates the multi-dim ragged.
      use_ragged=False -> stub the collective (static pad); isolates the leading-axis gather."""
    import time
    from jax.sharding import NamedSharding, PartitionSpec as P
    NCELL, K, V = 324, 40, 6                 # I*J*L horizontal cells; K,V ride along (=77760 total)
    jdt = jnp.asarray(np.empty(0, dtype)).dtype

    # cell-level send/recv sizes S[d,e] = #cells d ships to e (small; symmetric-ish ring+)
    rng = _rng(seed, 4242)
    S = np.zeros((nproc, nproc), np.int32)
    for d in range(nproc):
        for e in range(nproc):
            if d != e:
                S[d, e] = int(_rng(seed, 4242, min(d, e), max(d, e)).integers(8, 24))
    send_max = int(S.sum(axis=1).max()); recv_max = int(S.sum(axis=0).max())
    Sj = jnp.asarray(S)

    # this rank's leading-axis index maps (short: send_max / NCELL long)
    send_idx = np.zeros(send_max, np.int32)          # local cell ids to gather & ship (tag order)
    pos = 0
    for e in range(nproc):
        n = int(S[me, e])
        send_idx[pos:pos + n] = _rng(seed, 71, me, e).integers(0, NCELL, size=n); pos += n
    recv_off = np.zeros(nproc, np.int64); recv_off[1:] = np.cumsum(S[:, me])[:-1]
    recv_gather = np.zeros(NCELL, np.int32); halo_mask = np.zeros(NCELL, bool)
    halo_cells = _rng(seed, 72, me).choice(NCELL, size=min(int(S[:, me].sum()), NCELL), replace=False)
    for i, c in enumerate(halo_cells):
        recv_gather[c] = i % max(recv_max, 1); halo_mask[c] = True
    send_idx_j = jnp.asarray(send_idx); recv_gather_j = jnp.asarray(recv_gather)
    halo_mask_j = jnp.asarray(halo_mask)

    field0 = _rng(seed, 9, me).standard_normal((NCELL, K, V)).astype(dtype)

    m = min(send_max, recv_max)

    def _core(field, si, rg, hm):                     # field/si/rg/hm are this device's LOCAL arrays
        if use_gather:
            operand = field[si]                        # [send_max, K, V]  LEADING-axis row gather
        else:
            operand = field[:send_max]                # static slice (no fancy gather)
        if use_ragged:
            r = lax.axis_index('p')
            send_sizes = Sj[r, :]; recv_sizes = Sj[:, r]
            in_off = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(send_sizes)[:-1].astype(jnp.int32)])
            out_off = jnp.sum(jnp.where((jnp.arange(nproc) < r)[:, None], Sj, 0), axis=0).astype(jnp.int32)
            recv = lax.ragged_all_to_all(operand, jnp.zeros((recv_max, K, V), jdt),
                                         in_off, send_sizes, out_off, recv_sizes, axis_name='p')
        else:
            recv = jnp.zeros((recv_max, K, V), jdt).at[:m].set(operand[:m])   # stub collective
        if use_gather:
            gathered = recv[rg]                        # [NCELL, K, V]  LEADING-axis row gather
            out = jnp.where(hm[:, None, None], gathered, field)
        else:
            out = field.at[:recv_max].set(recv)       # static slice (no fancy gather)
        return out

    if idx_as_input:   # FESOM structure: index maps are SHARDED shard_map INPUTS, not closures
        def step(field, si, rg, hm):
            return _core(field[0], si[0], rg[0], hm[0])[None]
        wrapped = jax.jit(jax.shard_map(step, mesh=mesh, in_specs=(P('p'),) * 4,
                                        out_specs=P('p'), check_vma=False))
        def _gi(arr):   # global [nproc, *] sharded on axis 0; this rank provides its row
            return jax.make_array_from_single_device_arrays(
                (nproc,) + arr.shape, NamedSharding(mesh, P('p')),
                [jax.device_put(arr[None], jax.local_devices()[0])])
        gf = _gi(field0); gsi = _gi(send_idx); grg = _gi(recv_gather); ghm = _gi(halo_mask)
        args = (gf, gsi, grg, ghm)
    else:              # index maps are captured CONSTANTS inside the body
        def step(field):
            return _core(field[0], send_idx_j, recv_gather_j, halo_mask_j)[None]
        wrapped = jax.jit(jax.shard_map(step, mesh=mesh, in_specs=P('p'),
                                        out_specs=P('p'), check_vma=False))
        gf = jax.make_array_from_single_device_arrays(
            (nproc, NCELL, K, V), NamedSharding(mesh, P('p')),
            [jax.device_put(field0[None], jax.local_devices()[0])])
        args = (gf,)
    if me == 0:
        print(f"[{tag}] nproc={nproc} NCELL={NCELL} K={K} V={V} send_max={send_max} recv_max={recv_max} "
              f"gather={use_gather} ragged={use_ragged} idx_as_input={idx_as_input} -- lowering...", flush=True)
    t0 = time.time(); low = wrapped.lower(*args); t1 = time.time()
    print(f"[{tag}] rank{me} LOWERED in {t1-t0:.1f}s -- compiling...", flush=True)
    comp = low.compile(); t2 = time.time()
    print(f"[{tag}] rank{me} COMPILED in {t2-t1:.1f}s (lower {t1-t0:.1f}s)", flush=True)
    o = comp(*args); o.block_until_ready()
    print(f"[{tag}] rank{me} RAN ok. done.", flush=True)


def _fesom_prim_test(jax, jnp, lax, mesh, me, nproc, seed, dtype, tag='FESOMPRIM'):
    """Run FESOM's ACTUAL halo.halo_exchange_ragged (their code, not our transcription) in a
    1-GPU-per-process Miyabi harness, indices as sharded shard_map inputs (their structure).
    This is 'FESOM on Miyabi' in the only topology our HW allows. Hang => environment, not code."""
    import time, sys
    from jax.sharding import NamedSharding, PartitionSpec as P
    sys.path.insert(0, "/work/gj37/c24028/workforclaude/fesom_jax")
    import fesom_jax.halo as H                          # FESOM's real code
    NCELL, K, V = 324, 40, 6
    jdt = jnp.asarray(np.empty(0, dtype)).dtype
    S = np.zeros((nproc, nproc), np.int32)
    for d in range(nproc):
        for e in range(nproc):
            if d != e:
                S[d, e] = int(_rng(seed, 4242, min(d, e), max(d, e)).integers(8, 24))
    send_max = int(S.sum(axis=1).max()); recv_max = int(S.sum(axis=0).max())
    Sj = jnp.asarray(S)
    send_idx = np.zeros(send_max, np.int32); pos = 0
    for e in range(nproc):
        n = int(S[me, e]); send_idx[pos:pos + n] = _rng(seed, 71, me, e).integers(0, NCELL, size=n); pos += n
    recv_gather = np.zeros(NCELL, np.int32); halo_mask = np.zeros(NCELL, bool)
    hc = _rng(seed, 72, me).choice(NCELL, size=min(int(S[:, me].sum()), NCELL), replace=False)
    for i, c in enumerate(hc):
        recv_gather[c] = i % max(recv_max, 1); halo_mask[c] = True
    field0 = _rng(seed, 9, me).standard_normal((NCELL, K, V)).astype(dtype)

    def body(field, si, rg, hm):
        field, si, rg, hm = field[0], si[0], rg[0], hm[0]
        r = lax.axis_index('p')
        ss = Sj[r, :]; rs = Sj[:, r]
        so = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(ss)[:-1].astype(jnp.int32)])
        oo = jnp.sum(jnp.where((jnp.arange(nproc) < r)[:, None], Sj, 0), axis=0).astype(jnp.int32)
        rd = {"send_idx": si, "send_off": so, "send_sizes": ss,
              "out_off": oo, "recv_sizes": rs, "recv_gather": rg, "halo_mask": hm}
        out = H.halo_exchange_ragged(field, rd, recv_max, axis_name='p')   # FESOM's function
        return out[None]

    wrapped = jax.jit(jax.shard_map(body, mesh=mesh, in_specs=(P('p'),) * 4,
                                    out_specs=P('p'), check_vma=False))
    def _gi(a):
        return jax.make_array_from_single_device_arrays(
            (nproc,) + a.shape, NamedSharding(mesh, P('p')), [jax.device_put(a[None], jax.local_devices()[0])])
    args = (_gi(field0), _gi(send_idx), _gi(recv_gather), _gi(halo_mask))
    if me == 0:
        print(f"[{tag}] FESOM halo_exchange_ragged, idx as sharded inputs; send_max={send_max} "
              f"recv_max={recv_max} -- lowering...", flush=True)
    t0 = time.time(); low = wrapped.lower(*args); t1 = time.time()
    print(f"[{tag}] rank{me} LOWERED in {t1-t0:.1f}s -- compiling...", flush=True)
    comp = low.compile(); t2 = time.time()
    print(f"[{tag}] rank{me} COMPILED in {t2-t1:.1f}s (lower {t1-t0:.1f}s)", flush=True)
    o = comp(*args); o.block_until_ready()
    print(f"[{tag}] rank{me} RAN ok. done.", flush=True)


def jax_compile_test(mode='gather', seed=0):
    import time
    from mpi4py import MPI
    import jax, jax.numpy as jnp
    import jax.lax as lax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    import socket
    comm = MPI.COMM_WORLD
    me = comm.Get_rank(); nproc = comm.Get_size()
    # 1 GPU per rank (jax sees a single local device); build a 1-D mesh over 'p'.
    # Coordinator = rank0 host:free-port over mpi4py, matching mod_backend._init_distributed.
    if me == 0:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0)); coord = f"{socket.gethostname()}:{s.getsockname()[1]}"; s.close()
    else:
        coord = None
    coord = comm.bcast(coord, root=0)
    jax.distributed.initialize(coordinator_address=coord, num_processes=nproc, process_id=me)
    devs = np.array(jax.devices())
    assert devs.size == nproc, f"expected {nproc} global devices, got {devs.size}"
    mesh = Mesh(devs, ('p',))

    dtype = np.float64

    if mode == 'fesomprim':
        # 'FESOM on Miyabi': their ACTUAL halo_exchange_ragged, 1 GPU/proc, idx as sharded inputs.
        _fesom_prim_test(jax, jnp, lax, mesh, me, nproc, seed, dtype, tag='FESOMPRIM')
        return

    if mode in ('rowgather', 'rowgather_nocomm', 'raggedmd', 'rowgather_idxin'):
        # FESOM-EXACT structure: field [Ncells, K, V] (leading axis = horizontal cell; K,V ride
        # along). Disambiguate the rowgather hang: rowgather (gather+ragged, both), rowgather_nocomm
        # (leading gather, stub ragged -> isolates the GATHER), raggedmd (static slice, real
        # multi-dim ragged -> isolates the MULTI-DIM RAGGED), rowgather_idxin (gather+ragged but
        # index maps passed as SHARDED shard_map INPUTS = FESOM structure, not closure constants).
        ug = mode != 'raggedmd'                 # use fancy leading-axis gather
        ur = mode != 'rowgather_nocomm'         # use the real ragged collective
        _fesom_rowgather_test(jax, jnp, lax, mesh, me, nproc, seed, dtype,
                              use_gather=ug, use_ragged=ur,
                              idx_as_input=(mode == 'rowgather_idxin'), tag=mode.upper())
        return

    plan = build_global_plan(nproc, seed=seed)
    S = build_S(plan, nproc)
    mout = int(S.sum(axis=0).max())
    mo = int(S.sum(axis=1).max())
    Sj = jnp.asarray(S.astype(np.int32))
    jdt = jnp.asarray(np.empty(0, dtype)).dtype
    k = min(mo, mout)

    def ragged_real(operand):        # real NCCL ragged over the enclosing shard_map axis 'p'
        r = lax.axis_index('p')
        send_sizes = Sj[r, :]; recv_sizes = Sj[:, r]
        input_offsets = jnp.concatenate(
            [jnp.zeros(1, jnp.int32), jnp.cumsum(send_sizes)[:-1].astype(jnp.int32)])
        output_offsets = jnp.sum(jnp.where((jnp.arange(nproc) < r)[:, None], Sj, 0),
                                 axis=0).astype(jnp.int32)
        return lax.ragged_all_to_all(operand, jnp.zeros(mout, jdt),
                                     input_offsets, send_sizes, output_offsets, recv_sizes,
                                     axis_name='p')

    def ragged_stub(operand):        # NO collective: static pad/truncate (isolates pack/unpack)
        return jnp.zeros(mout, jdt).at[:k].set(operand[:k])

    # mode selects which cost we isolate:
    #   gather   = FESOM gather pack/unpack + REAL ragged   (the candidate)
    #   scatter  = current fancy-scatter pack/unpack + REAL ragged (the inc3 baseline)
    #   nocomm   = gather pack/unpack + STUB ragged  (isolates the gather graph alone)
    #   raggedonly = trivial slice pack/unpack + REAL ragged (isolates the collective alone)
    # nocomm/packonly/recvonly stub the collective (isolate the fancy-index graph); others use real
    ragged_exchange = ragged_stub if mode in ('nocomm', 'packonly', 'recvonly') else ragged_real
    fns = make_step_fns(plan, S, me, nproc, seed, jnp, ragged_exchange, dtype)
    var0, vpl0, rag = fns['var0'], fns['vpl0'], fns['rag']

    if mode == 'raggedonly':
        def step(var, var_pl):       # minimal pack/unpack: no fancy indexing at all
            var = var[0]; var_pl = var_pl[0]
            operand = var.reshape(-1)[:mo].astype(jdt)
            recvd = ragged_real(operand)
            vf = var.reshape(-1).at[:k].set(recvd[:k])
            return vf.reshape(DIMS_VAR)[None], var_pl[None]
    else:
        step = {'scatter': fns['step_scatter'], 'gather': fns['step_gather'],
                'nocomm': fns['step_gather'], 'packonly': fns['step_packonly'],
                'recvonly': fns['step_recvonly']}[mode]

    wrapped = jax.jit(jax.shard_map(step, mesh=mesh, in_specs=(P('p'), P('p')),
                                    out_specs=(P('p'), P('p')), check_vma=False))

    # global-sharded inputs (leading process axis)
    gvar = jax.make_array_from_single_device_arrays(
        (nproc,) + DIMS_VAR, NamedSharding(mesh, P('p')),
        [jax.device_put(var0[None], jax.local_devices()[0])])
    gpl = jax.make_array_from_single_device_arrays(
        (nproc,) + DIMS_PL, NamedSharding(mesh, P('p')),
        [jax.device_put(vpl0[None], jax.local_devices()[0])])

    tag = mode.upper()
    if me == 0:
        print(f"[{tag}] nproc={nproc} mo={mo} mout={mout} "
              f"pack_ops={len(rag['rag_pack'])} unpack_ops={len(rag['rag_unpack'])} "
              f"-- lowering...", flush=True)
    t0 = time.time()
    lowered = wrapped.lower(gvar, gpl)
    t1 = time.time()
    print(f"[{tag}] rank{me} LOWERED in {t1-t0:.1f}s -- compiling...", flush=True)
    compiled = lowered.compile()
    t2 = time.time()
    print(f"[{tag}] rank{me} COMPILED in {t2-t1:.1f}s (lower {t1-t0:.1f}s)", flush=True)
    # actually run once to confirm it executes
    out_v, out_pl = compiled(gvar, gpl)
    out_v.block_until_ready()
    print(f"[{tag}] rank{me} RAN ok. done.", flush=True)


def cpu_trace_test(nproc=4, seed=0):
    """Trace + compile BOTH step bodies on CPU fake-devices with the ragged collective
    STUBBED (CPU has no ragged opcode). Proves the pack/unpack graphs trace correctly
    under shard_map before spending a GPU job -- no MPI, no jax.distributed."""
    import time, jax, jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    assert jax.device_count() >= nproc, (
        f"need {nproc} devices; set XLA_FLAGS=--xla_force_host_platform_device_count={nproc}")
    mesh = Mesh(np.array(jax.devices()[:nproc]), ('p',))
    plan = build_global_plan(nproc, seed=seed)
    S = build_S(plan, nproc)
    mout = int(S.sum(axis=0).max())
    jdt = jnp.asarray(np.empty(0, np.float64)).dtype
    k = min(int(S.sum(axis=1).max()), mout)

    def ragged_stub(operand):        # static pad/truncate -> (mout,), no ragged opcode
        return jnp.zeros(mout, jdt).at[:k].set(operand[:k])

    for me_probe in (0,):            # rank-0's plan is representative for a trace check
        fns = make_step_fns(plan, S, me_probe, nproc, seed, jnp, ragged_stub)
    var0, vpl0 = fns['var0'], fns['vpl0']
    gvar = jax.device_put(np.broadcast_to(var0[None], (nproc,) + DIMS_VAR).copy(),
                          NamedSharding(mesh, P('p')))
    gpl = jax.device_put(np.broadcast_to(vpl0[None], (nproc,) + DIMS_PL).copy(),
                         NamedSharding(mesh, P('p')))
    for name in ('step_gather', 'step_scatter'):
        w = jax.jit(jax.shard_map(fns[name], mesh=mesh, in_specs=(P('p'), P('p')),
                                  out_specs=(P('p'), P('p')), check_vma=False))
        t = time.time(); c = w.lower(gvar, gpl).compile()
        o = c(gvar, gpl); o[0].block_until_ready()
        print(f"[cpu-trace] {name}: traced+compiled+ran (ragged stubbed) in {time.time()-t:.1f}s")
    print("[cpu-trace] BOTH step bodies trace correctly under shard_map -> ready for GPU.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jax', action='store_true', help='run the GPU shard_map compile test')
    ap.add_argument('--cpu', action='store_true', help='CPU trace check (ragged stubbed, no MPI)')
    ap.add_argument('--mode', default='gather',
                    choices=('gather', 'scatter', 'nocomm', 'raggedonly',
                             'packonly', 'recvonly', 'rowgather',
                             'rowgather_nocomm', 'raggedmd',
                             'rowgather_idxin', 'fesomprim'),
                    help='(with --jax) which cost to isolate')
    ap.add_argument('--nproc', type=int, default=4, help='(numpy check) simulated rank count')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.jax:
        jax_compile_test(mode=args.mode, seed=args.seed)
    elif args.cpu:
        cpu_trace_test(nproc=args.nproc, seed=args.seed)
    else:
        # numpy V9 unit -- sweep a few rank counts + seeds
        for nproc in (2, 4, 8, 20):
            for seed in range(3):
                numpy_check(nproc=nproc, seed=seed, verbose=(seed == 0))
        print("[numpy V9] ALL equivalence checks PASSED (gather == scatter, bit-identical).",
              flush=True)


if __name__ == '__main__':
    main()
