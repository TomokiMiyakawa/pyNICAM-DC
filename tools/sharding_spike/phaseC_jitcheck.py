#!/usr/bin/env python
"""Phase C compile-check (R1): does shard_map + ragged_all_to_all compose INSIDE jax.jit,
with a GATHER before and a SCATTER after (the model's _core structure)?

The model's _core is jax.jit'd and does: gather local cells -> collective -> scatter into
the halo. Here we replicate that shape around ragged_all_to_all and wrap the whole thing in
jax.jit, to answer R1 before touching mod_comm. Sparse ring (each rank -> r+1). Also checks
it stays device-resident (the collective is NCCL) and correct.

Tests three nesting levels so we learn exactly where (if anywhere) it breaks:
  L0: shard_map(ragged) alone            (already known good from ragged_spike)
  L1: shard_map(gather->ragged->scatter) (add the pack/unpack around it)
  L2: jax.jit(L1)                        (the real question: shard_map inside jit)
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

import jax
jax.distributed.initialize(coordinator_address=coord, num_processes=size, process_id=rank)
import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from functools import partial

N = size
M = 8                                    # local rows per rank (uniform shard shape)
mesh = Mesh(np.array(jax.devices()), ("p",))
shard = NamedSharding(mesh, P("p", None))

# global (N, M): row-block r = [r*100 + 0..M-1]; local shard on rank r is (1, M) -> flatten to (M,)
local0 = (np.arange(M, dtype=np.float32) + rank * 100.0).reshape(1, M)
g = jax.make_array_from_process_local_data(shard, local0, (N, M))

# ring ragged: each rank sends its WHOLE (padded) row to r+1, size 0 to others.
def _ring_body(operand2d):
    # operand2d: this shard (1, M). ragged leading axis = rows.
    operand = operand2d.reshape(M)                       # (M,) send buffer
    r = lax.axis_index("p")
    idx = jnp.arange(N)
    send_sizes     = jnp.where(idx == (r + 1) % N, M, 0).astype(jnp.int32)
    input_offsets  = jnp.zeros(N, jnp.int32)
    output_offsets = jnp.zeros(N, jnp.int32)
    recv_sizes     = jnp.where(idx == (r - 1) % N, M, 0).astype(jnp.int32)
    out = jnp.zeros(M, jnp.float32)
    res = lax.ragged_all_to_all(operand, out, input_offsets, send_sizes,
                                output_offsets, recv_sizes, axis_name="p")
    return res.reshape(1, M)

def _gather_ring_scatter(operand2d):
    # GATHER: reverse the row (a trivial nontrivial gather), send, then SCATTER: un-reverse.
    gi = jnp.arange(M)[::-1]
    x = operand2d[:, gi]                                  # gather
    res = _ring_body(x)
    si = jnp.arange(M)[::-1]
    out = jnp.zeros_like(operand2d).at[:, si].set(res)    # scatter
    return out

L1 = partial(jax.shard_map, mesh=mesh, in_specs=P("p", None), out_specs=P("p", None))(_gather_ring_scatter)
L2 = jax.jit(L1)

def check(fn, name, inp=None):
    try:
        res = fn(g if inp is None else inp)
        got = np.asarray(res.addressable_shards[0].data).reshape(M)
        # rank r receives rank (r-1)'s row [ (r-1)*100 + 0..M-1 ], gather+scatter cancel -> identity
        exp = np.arange(M, dtype=np.float32) + ((rank - 1) % N) * 100.0
        ok = np.allclose(got, exp)
        allok = comm.allreduce(1 if ok else 0, op=MPI.LAND)
        if rank == 0:
            print(f"[phaseC] {name}: {'ALL-PASS' if allok else 'FAIL'}  (rank0 got[0]={got[0]} exp[0]={exp[0]})", flush=True)
        return allok
    except Exception as e:
        if rank == 0:
            print(f"[phaseC] {name}: EXCEPTION {type(e).__name__}: {str(e)[:300]}", flush=True)
        comm.Barrier()
        return 0

r1 = check(L1, "L1 shard_map(gather->ragged->scatter)")
comm.Barrier()
r2 = check(L2, "L2 jax.jit(shard_map(...))  <-- R1")
comm.Barrier()

# L3: THE MODEL SCENARIO -- can a jitted fn that receives a per-process LOCAL array (like
# the model's jvar) build the operand locally and feed shard_map? If yes, halo-only (D1=a)
# integration is trivial (no global-array wrapping). Try 2 ways to bridge local->global.
# per-process LOCAL (1, M) single-device array (like the model's jvar: addressable only here)
loc = jax.device_put(np.arange(M, dtype=np.float32).reshape(1, M) + rank * 100.0)

def bridge_A(x):
    # declare the per-shard input as a global (N,M) array sharded on 'p', then shard_map.
    x = jax.lax.with_sharding_constraint(x, shard)
    return L1(x)

r3 = check(jax.jit(bridge_A), "L3 jit(local (1,M) -> wsc -> shard_map)  <-- D1=a?", inp=loc)
comm.Barrier()

if rank == 0:
    print(f"[phaseC] RESULT L1={'ok' if r1 else 'FAIL'} L2(jit)={'ok' if r2 else 'FAIL'} "
          f"L3(local-in)={'ok' if r3 else 'FAIL'}", flush=True)
    print("[phaseC] DONE", flush=True)
comm.Barrier()
