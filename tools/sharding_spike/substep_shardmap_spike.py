#!/usr/bin/env python
"""De-risk spike #5 (Option 1): does a REPRESENTATIVE dynamics sub-step compile as ONE
jax.jit(shard_map(...)) on model-shaped, GLOBALLY-SHARDED arrays -- compute + pole + a
ragged_all_to_all halo exchange -- and also in the fused lax.scan form?

Background: spikes #2/#3 proved ragged_all_to_all is correct/device-resident and nests in
lax.scan+jit. The Phase-C1 attempt then showed the model's RESIDENT COMM runs INSIDE a jit
trace, and there is NO traceable way to promote a per-process LOCAL array to a global
sharded array mid-trace (make_array_* are host-only). => "swap only the transport"
(Strategy 2) is impossible; the arrays must be GLOBAL-SHARDED from setup and the whole step
wrapped in shard_map (Option 1 / D1=b). THIS spike de-risks that before the refactor.

What it proves (or breaks) at N=4:
  1. Model-shaped PRG_var-like array (nproc, I,J,K,L,V) + a POLE array (nproc, Ipl,K,Lpl,V)
     can be created GLOBAL-SHARDED (make_array_from_process_local_data, host, once) and fed
     to jit(shard_map).
  2. INSIDE shard_map each device sees its LOCAL shard (I,J,K,L,V) -> existing kernels need
     NO change (printed to confirm).
  3. A representative compute (x/y Laplacian stencil touching neighbours + a pole->region
     mix like p2r) + a ragged_all_to_all halo exchange (axis 'p', pvary'd) + more compute
     all COMPILE and run finite in ONE shard_map.
  4. The SAME body wrapped in a K-step lax.scan (the fused whole-step shape) also compiles.
Success = compiles + runs finite. (Bit-exactness vs alltoall is deferred to the real
implementation; this is a compile/mechanics de-risk, matching the agreed plan.)
"""
import os, socket
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank(); size = comm.Get_size(); host = socket.gethostname()
if rank == 0:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0))
    coord = f"{host}:{s.getsockname()[1]}"; s.close()
else:
    coord = None
coord = comm.bcast(coord, root=0); comm.Barrier()
if rank == 0: print(f"[sub] N={size} coordinator={coord}", flush=True)

import jax
jax.distributed.initialize(coordinator_address=coord, num_processes=size, process_id=rank)
import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from functools import partial
import importlib

_pvary = None
for _p in ("jax.lax", "jax._src.shard_map", "jax._src.core"):
    try:
        _m = importlib.import_module(_p)
        if hasattr(_m, "pvary"): _pvary = getattr(_m, "pvary"); break
    except Exception:
        pass
if rank == 0: print(f"[sub] jax={jax.__version__} pvary={'yes' if _pvary else 'MISSING'}", flush=True)

N = size
# model-representative dims (gl05-ish region incl. halo): I,J horiz; K vertical; L regions; V vars
I = J = 18; K = 40; L = 1; V = 6
IPL = 6; LPL = 1                                   # pole array dims
KSTEP = int(os.environ.get("SUB_K", "4"))
mesh = Mesh(np.array(jax.devices()), ("p",))
sh6 = NamedSharding(mesh, P("p", None, None, None, None, None))
sh5 = NamedSharding(mesh, P("p", None, None, None, None))

# per-process local data -> GLOBAL-SHARDED arrays (host, once; the setup-side of Option 1)
locv  = (np.arange(I*J*K*L*V, dtype=np.float32).reshape(1, I, J, K, L, V) * 1e-4 + rank).astype(np.float32)
locpl = (np.ones((1, IPL, K, LPL, V), np.float32) * (rank + 1)).astype(np.float32)
gvar    = jax.make_array_from_process_local_data(sh6, locv,  (N, I, J, K, L, V))
gvar_pl = jax.make_array_from_process_local_data(sh5, locpl, (N, IPL, K, LPL, V))

# synthetic degree-1 ring halo layout: each rank sends ONE face (J*V cells at K=0,L=0) to r+1.
FACE = J * V
S_np = np.zeros((N, N), np.int32)
for i in range(N):
    S_np[i, (i + 1) % N] = FACE
S = jnp.asarray(S_np)
MAXROWS = int(S_np.sum(axis=1).max())              # = FACE (uniform ring)

def _substep(var6, varpl5):
    """var6:(1,I,J,K,L,V) varpl5:(1,IPL,K,LPL,V) local shards. Returns same shapes."""
    v  = var6[0]                                   # (I,J,K,L,V) -- the LOCAL shape kernels see
    pl = varpl5[0]                                 # (IPL,K,LPL,V)
    # (1) representative compute: x/y Laplacian (touches neighbours) + pole->region mix (p2r-like)
    v = v + 0.1 * (jnp.roll(v, 1, 0) + jnp.roll(v, -1, 0) - 2.0 * v)
    v = v + 0.1 * (jnp.roll(v, 1, 1) + jnp.roll(v, -1, 1) - 2.0 * v)
    pole_mean = jnp.mean(pl, axis=0)               # (K,LPL,V)
    v = v.at[0, :, :, 0, :].add(0.01 * pole_mean[:, 0, :][None, :, :])   # broadcast (K,V)->(J,K,V)? guard below

    # (2) ragged halo exchange (axis 'p', pvary'd) -- pack a face -> ragged -> unpack opposite face
    r   = lax.axis_index("p")
    idx = jnp.arange(N)
    send_sizes = S[r, :]
    recv_sizes = S[:, r]
    input_offsets  = jnp.concatenate([jnp.zeros(1, jnp.int32),
                                      jnp.cumsum(send_sizes)[:-1].astype(jnp.int32)])
    output_offsets = jnp.sum(jnp.where((idx < r)[:, None], S, 0), axis=0).astype(jnp.int32)
    face = v[0, :, 0, 0, :].reshape(FACE)          # (J,V) boundary slab at I=0,K=0,L=0
    operand = jnp.zeros(MAXROWS, v.dtype).at[:FACE].set(face)
    out = jnp.zeros(MAXROWS, v.dtype)
    recvd = lax.ragged_all_to_all(operand, out, input_offsets, send_sizes,
                                  output_offsets, recv_sizes, axis_name="p")
    if _pvary is not None:
        recvd = _pvary(recvd, "p")
    v = v.at[I - 1, :, 0, 0, :].set(recvd[:FACE].reshape(J, V))   # write into opposite face

    # (3) more compute
    v = jnp.tanh(v)
    return v[None], varpl5

# --- S0: representative sub-step as ONE shard_map, jit'd ---
S0 = jax.jit(jax.shard_map(_substep, mesh=mesh,
                           in_specs=(P("p", None, None, None, None, None),
                                     P("p", None, None, None, None)),
                           out_specs=(P("p", None, None, None, None, None),
                                      P("p", None, None, None, None))))

# --- S1: the SAME body wrapped in a K-step lax.scan (the fused whole-step shape) ---
def _scan_step(var6, varpl5):
    def body(carry, _):
        nv, npl = _substep(carry[0], carry[1])
        return (nv, npl), nv[0, 0, 0, 0, 0, 0]
    (fv, fpl), ys = lax.scan(body, (var6, varpl5), xs=None, length=KSTEP)
    return fv, fpl
S1 = jax.jit(jax.shard_map(_scan_step, mesh=mesh,
                           in_specs=(P("p", None, None, None, None, None),
                                     P("p", None, None, None, None)),
                           out_specs=(P("p", None, None, None, None, None),
                                      P("p", None, None, None, None))))

def run(fn, name):
    try:
        v, plout = fn(gvar, gvar_pl)
        loc = np.asarray(v.addressable_shards[0].data)
        finite = bool(np.all(np.isfinite(loc)))
        okall = comm.allreduce(1 if finite else 0, op=MPI.LAND)
        if rank == 0:
            print(f"[sub] {name}: {'COMPILE+RUN OK' if okall else 'NON-FINITE'}  "
                  f"local_shard_shape={loc.shape}  sample={loc.reshape(-1)[0]:.4f}", flush=True)
        return okall
    except Exception as e:
        if rank == 0:
            print(f"[sub] {name}: EXCEPTION {type(e).__name__}: {str(e)[:400]}", flush=True)
        comm.Barrier(); return 0

if rank == 0:
    print(f"[sub] global var shape=({N},{I},{J},{K},{L},{V}) -> each shard sees ({I},{J},{K},{L},{V}); "
          f"pole=({N},{IPL},{K},{LPL},{V}); FACE={FACE} KSTEP={KSTEP}", flush=True)
r0 = run(S0, "S0 substep(compute+pole+ragged) as ONE shard_map"); comm.Barrier()
r1 = run(S1, f"S1 SAME body in lax.scan_{KSTEP} (fused shape)");   comm.Barrier()

if rank == 0:
    verdict = ("OPTION 1 VIABLE: representative sub-step (compute+pole+ragged) compiles "
               "under one shard_map, incl. the fused scan form") if (r0 and r1) else \
              "OPTION 1 BLOCKED HERE -- inspect the exception above"
    print(f"[sub] RESULT S0={'ok' if r0 else 'FAIL'} S1(scan)={'ok' if r1 else 'FAIL'}", flush=True)
    print(f"[sub] VERDICT: {verdict}", flush=True)
    print("[sub] DONE", flush=True)
comm.Barrier()
