#!/usr/bin/env python
"""De-risk spike #2 for the jax.sharding COMM migration: jax.lax.ragged_all_to_all.

Same foundation as ppermute_spike (mpi4py boot -> jax.distributed.initialize -> mesh ->
shard_map), but exchanges via ragged_all_to_all instead of ppermute. Tests the questions
that decide ragged-vs-ppermute (see memory pynicam-comm-architecture / comm-replace-plan):

  1. CORRECTNESS across processes: a sparse ring (each rank sends ONE row to r+1, size 0 to
     all other ranks) -> rank r must receive rank (r-1)'s row.
  2. DEVICE-RESIDENT: repeating K times must NOT grow HtoD/DtoH (nsys memops).
  3. ★ THE KEY QUESTION: does ragged_all_to_all cut MESSAGE COUNT, or only volume? The
     offset/size arrays have length nproc (one entry per dest, most size 0). If NCCL skips
     the size-0 sends it is truly sparse (count ~ degree); if it issues nproc sends (most
     empty) the O(nproc) dispatch survives. Read the nsys SendRecv kernel count per iter.
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
if rank == 0: print(f"[ragged] N={size} coordinator={coord}", flush=True)

import jax
jax.distributed.initialize(coordinator_address=coord, num_processes=size, process_id=rank)
print(f"[ragged] rank={rank} host={host} global={jax.device_count()} local={jax.local_device_count()}",
      flush=True)

import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from functools import partial

N = size
F = 4                                   # feature width of each exchanged row
mesh = Mesh(np.array(jax.devices()), ("p",))
shard = NamedSharding(mesh, P("p", None))

# global (N, F): row r = [r,r,r,r]; local shard on rank r is the single row (1, F)
g = jax.make_array_from_process_local_data(
    shard, np.full((1, F), float(rank), dtype=np.float32), (N, F))

@partial(jax.shard_map, mesh=mesh, in_specs=P("p", None), out_specs=P("p", None))
def ring_raggeda2a(operand):
    r = lax.axis_index("p")
    idx = jnp.arange(N)
    # SEND: my one row (operand[0:1]) goes ONLY to dest (r+1); size 0 to every other dest.
    send_sizes    = (idx == (r + 1) % N).astype(jnp.int32)          # length N (per dest)
    input_offsets = jnp.zeros(N, jnp.int32)                          # my slice is at operand[0]
    output_offsets= jnp.zeros(N, jnp.int32)                          # receiver writes it at its row 0
    # RECV: I receive one row from src (r-1); size 0 from every other src.
    recv_sizes    = (idx == (r - 1) % N).astype(jnp.int32)          # length N (per src)
    out = jnp.zeros((1, F), jnp.float32)
    return lax.ragged_all_to_all(operand, out, input_offsets, send_sizes,
                                 output_offsets, recv_sizes, axis_name="p")

# --- correctness (one shot) ---
res = ring_raggeda2a(g)
got = float(np.asarray(res.addressable_shards[0].data)[0, 0])
expected = float((rank - 1) % N)
ok = abs(got - expected) < 1e-6
all_ok = comm.allreduce(1 if ok else 0, op=MPI.LAND)
print(f"[ragged] rank={rank} got={got} expected={expected} {'OK' if ok else 'FAIL'}", flush=True)
if rank == 0:
    print(f"[ragged] CORRECTNESS: {'ALL-PASS' if all_ok else 'FAIL'}", flush=True)

# --- residency + count stress: K repeats on device arrays ---
K = int(os.environ.get("SPIKE_ITERS", "200"))
comm.Barrier()
if rank == 0:
    print(f"[ragged] --- residency loop: {K} ragged_all_to_all iterations (nsys window) ---", flush=True)
prof = (rank == 0) and os.environ.get("SPIKE_NSYS", "0") != "0"
if prof:
    import ctypes; cudart = ctypes.CDLL("libcudart.so"); cudart.cudaProfilerStart()
x = g
for _ in range(K):
    x = ring_raggeda2a(x)
    x.block_until_ready()
if prof:
    cudart.cudaProfilerStop()
# after K ring steps, my row advanced by -K mod N
final = float(np.asarray(x.addressable_shards[0].data)[0, 0])
exp_final = float((rank - K) % N)
ok2 = abs(final - exp_final) < 1e-6
all_ok2 = comm.allreduce(1 if ok2 else 0, op=MPI.LAND)
if rank == 0:
    print(f"[ragged] LOOP CORRECTNESS after {K}: {'ALL-PASS' if all_ok2 else 'FAIL'}", flush=True)
    print(f"[ragged] DONE", flush=True)
comm.Barrier()
