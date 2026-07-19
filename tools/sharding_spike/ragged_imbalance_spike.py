#!/usr/bin/env python
"""De-risk spike #4 (R3): isolate ragged_all_to_all's SINGLE-BARRIER cost under NON-uniform
degree + a straggler -- the thing the uniform-degree ring spike (#2) structurally cannot show.

ragged collapses each halo exchange to ONE collective gated by the busiest rank. That single
wait = (a) compute ARRIVAL-SKEW [decomposition-driven; the measured pe40 pole/singular
straggler; ragged does NOT fix] + (b) collective-internal DEGREE imbalance [< dense, non-zero,
unknown]. We measure per-call ragged time across a 2x2 to DECOMPOSE (a) vs (b):

  A  balanced degree,  no straggler   -> baseline
  B  imbalanced degree, no straggler  -> isolates (b) degree imbalance
  C  balanced degree,  + straggler    -> isolates (a) arrival skew
  D  imbalanced + straggler           -> combined (realistic pe64 shape)

Degree distribution mimics the measured gl11-pe64 r2r graph (mean ~5, max 12). Straggler =
the max-degree rank does extra GPU work each iter before the collective (host-side, OUTSIDE
the jitted collective) so it arrives late = everyone waits at the one barrier.
Run at N=20 (so max-degree 12 is meaningful). Feeds comm-replace-plan_v2 R3 / Phase E.
"""
import os, socket, time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank(); size = comm.Get_size(); host = socket.gethostname()
if rank == 0:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0))
    coord = f"{host}:{s.getsockname()[1]}"; s.close()
else:
    coord = None
coord = comm.bcast(coord, root=0); comm.Barrier()
if rank == 0: print(f"[imb] N={size} coordinator={coord}", flush=True)

import jax
jax.distributed.initialize(coordinator_address=coord, num_processes=size, process_id=rank)
import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from functools import partial

N     = size
F     = int(os.environ.get("IMB_F", "512"))       # feature width per row
MSG   = int(os.environ.get("IMB_MSG", "128"))     # rows per edge (message granularity)
K     = int(os.environ.get("IMB_ITERS", "200"))   # timed iterations
STRAG = int(os.environ.get("IMB_STRAG", "40"))    # straggler extra matmuls / iter
mesh  = Mesh(np.array(jax.devices()), ("p",))
shard = NamedSharding(mesh, P("p", None))

# ---- build the two global send-size matrices S[i][j] (ROWS sent i->j), identical on all ranks ----
def ring_degree(deg):
    """out-degree vector -> adjacency: rank i sends MSG rows to (i+1..i+deg[i]) mod N."""
    S = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for k in range(1, deg[i] + 1):
            S[i, (i + k) % N] = MSG
    return S

# balanced: every rank out-degree 5 (uniform)
deg_bal = [5] * N
# imbalanced: mean ~4.6, max 12 (mimics gl11-pe64 r2r: mean 5.2 / max-12 seam outlier)
deg_pattern = [12, 8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 1]
deg_imb = [deg_pattern[i % len(deg_pattern)] for i in range(N)]
S_bal = ring_degree(deg_bal)
S_imb = ring_degree(deg_imb)

straggler_rank = int(np.argmax([r.sum() for r in S_imb]))   # busiest sender = the straggler

def buffers(S):
    max_send = int(S.sum(axis=1).max())     # rows any rank sends
    max_recv = int(S.sum(axis=0).max())     # rows any rank receives
    return max_send, max_recv

def make_exchange(S_np):
    """jit(shard_map(ragged)) with S baked in as a constant; returns fn(operand)->recv buf."""
    max_send, max_recv = buffers(S_np)
    S = jnp.asarray(S_np, dtype=jnp.int32)
    def _body(operand):                      # operand: (max_send, F) this rank's send buffer
        r = lax.axis_index("p")
        idx = jnp.arange(N)
        send_sizes     = S[r, :]                                     # rows to each dest
        recv_sizes     = S[:, r]                                     # rows from each src
        input_offsets  = jnp.concatenate([jnp.zeros(1, jnp.int32),
                                          jnp.cumsum(send_sizes)[:-1].astype(jnp.int32)])
        # output_offsets[d] = rows sent to d by ranks < me  (receiver-side prefix; model form)
        mask           = (idx < r)[:, None]
        output_offsets = jnp.sum(jnp.where(mask, S, 0), axis=0).astype(jnp.int32)
        out = jnp.zeros((max_recv, F), jnp.float32)
        return lax.ragged_all_to_all(operand, out, input_offsets, send_sizes,
                                     output_offsets, recv_sizes, axis_name="p")
    fn = jax.jit(partial(jax.shard_map, mesh=mesh,
                         in_specs=P("p", None), out_specs=P("p", None))(_body))
    # per-rank send operand: every row = (rank+1) so received blocks reveal their source count
    loc = np.full((max_send, F), float(rank + 1), dtype=np.float32)
    operand = jax.make_array_from_process_local_data(shard, loc, (N * max_send, F))
    return fn, operand, max_send, max_recv

# straggler busy-work (host-driven, outside the collective) -> delays arrival on one rank
_bm = jax.device_put(np.random.RandomState(rank).randn(1024, 1024).astype(np.float32)) \
      if False else jnp.ones((1024, 1024), jnp.float32) * (0.001 * (rank + 1))
@jax.jit
def _burn(x):
    for _ in range(4):
        x = jnp.tanh(x @ x) * 1.0001
    return x

def expected_recv_sum(S_np):
    # rank r receives, from each src s, S[s,r] rows each valued (s+1)
    return float(sum(S_np[s, rank] * (s + 1) for s in range(N))) * F

def time_scenario(name, fn, operand, S_np, straggle):
    exp = expected_recv_sum(S_np)
    # correctness sanity (once)
    res = fn(operand); res.block_until_ready()
    got = float(np.asarray(res.addressable_shards[0].data).sum())
    ok = abs(got - exp) < max(1.0, abs(exp) * 1e-5)
    allok = comm.allreduce(1 if ok else 0, op=MPI.LAND)
    # warm
    for _ in range(5):
        fn(operand).block_until_ready()
    comm.Barrier()
    t0 = time.perf_counter()
    for _ in range(K):
        if straggle and rank == straggler_rank:
            b = _bm
            for _ in range(STRAG):
                b = _burn(b)
            b.block_until_ready()
        fn(operand).block_until_ready()
    comm.Barrier()
    dt = (time.perf_counter() - t0) / K * 1e3           # ms/call, this rank
    dt_max = comm.allreduce(dt, op=MPI.MAX)             # slowest rank gates the barrier
    dt_min = comm.allreduce(dt, op=MPI.MIN)
    if rank == 0:
        print(f"[imb] {name:34s} per-call max={dt_max:7.3f} ms  min={dt_min:7.3f} ms  "
              f"skew={dt_max-dt_min:6.3f} ms  corr={'OK' if allok else 'FAIL'}", flush=True)
    return dt_max

fn_bal, op_bal, ms_b, mr_b = make_exchange(S_bal)
fn_imb, op_imb, ms_i, mr_i = make_exchange(S_imb)
if rank == 0:
    print(f"[imb] N={N} F={F} MSG={MSG} ITERS={K} STRAG={STRAG}/iter  straggler_rank={straggler_rank}", flush=True)
    print(f"[imb] balanced  deg=5  send/recv rows max={ms_b}/{mr_b}  (~{ms_b*F*4/1024:.0f} KB send/call)", flush=True)
    print(f"[imb] imbalanced deg max={max(deg_imb)} mean={np.mean(deg_imb):.1f}  send/recv rows max={ms_i}/{mr_i}", flush=True)

A = time_scenario("A balanced,    no straggler", fn_bal, op_bal, S_bal, False); comm.Barrier()
B = time_scenario("B imbalanced,  no straggler", fn_imb, op_imb, S_imb, False); comm.Barrier()
C = time_scenario("C balanced,    + straggler",  fn_bal, op_bal, S_bal, True);  comm.Barrier()
D = time_scenario("D imbalanced,  + straggler",  fn_imb, op_imb, S_imb, True);  comm.Barrier()

if rank == 0:
    print("[imb] ---- DECOMPOSITION (per-call, slowest rank) ----", flush=True)
    print(f"[imb]   baseline A                = {A:7.3f} ms", flush=True)
    print(f"[imb]   (b) degree imbalance  B-A = {B-A:+7.3f} ms   ({(B/A-1)*100:+.1f}%)", flush=True)
    print(f"[imb]   (a) arrival skew      C-A = {C-A:+7.3f} ms   ({(C/A-1)*100:+.1f}%)", flush=True)
    print(f"[imb]   combined              D-A = {D-A:+7.3f} ms   ({(D/A-1)*100:+.1f}%)", flush=True)
    print("[imb] DONE", flush=True)
comm.Barrier()
