#!/usr/bin/env python
"""De-risk spike for the jax.sharding COMM migration (see memory pynicam-comm-architecture).

Proves, on this Miyabi/mpirun/IB stack, the FOUNDATION the migration rests on:
  1. mpi4py boots the world (as today) and distributes a coordinator address;
  2. jax.distributed.initialize joins the N single-GPU processes into ONE global mesh;
  3. a neighbor exchange via shard_map + jax.lax.ppermute (-> ncclSend/Recv) is CORRECT
     across processes (ring: rank r receives rank (r-1)%N's value);
  4. it is DEVICE-RESIDENT: repeating the ppermute K times does NOT grow HtoD/DtoH
     (the mpi4jax path host-bounces every call; NCCL registers persistent buffers -> 0).

Run under nsys on rank0 to read memops (KILLER check for #4). Pure standalone; no model.
"""
import os
import socket
import sys

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
host = socket.gethostname()

# --- rank0 picks a free port, everyone learns the coordinator (host:port) ---
if rank == 0:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    coord = f"{host}:{port}"
else:
    coord = None
coord = comm.bcast(coord, root=0)
comm.Barrier()
if rank == 0:
    print(f"[spike] N={size} coordinator={coord}", flush=True)

# --- join the multi-process JAX runtime (the thing the model does NOT do today) ---
import jax  # noqa: E402  (import AFTER MPI so mpi4py owns MPI_Init)

jax.distributed.initialize(
    coordinator_address=coord,
    num_processes=size,
    process_id=rank,
)
print(f"[spike] rank={rank} host={host} "
      f"global_devices={jax.device_count()} local={jax.local_device_count()} "
      f"process_index={jax.process_index()}", flush=True)

import numpy as np                      # noqa: E402
import jax.numpy as jnp                 # noqa: E402
import jax.lax as lax                   # noqa: E402
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding  # noqa: E402
from functools import partial           # noqa: E402

# 1-D mesh over ALL global GPUs, axis 'p' = the rank axis (stand-in for the region-rank axis)
mesh = Mesh(np.array(jax.devices()), ("p",))
shard = NamedSharding(mesh, P("p"))

# global array of one f32 per rank, value == rank; local shard on this process is (rank,)
local_data = np.array([float(rank)], dtype=np.float32)
g = jax.make_array_from_process_local_data(shard, local_data, (size,))

# ring neighbor exchange: value at src is delivered to dst. rank r gets (r-1)%N.
ring_perm = [(i, (i + 1) % size) for i in range(size)]

@partial(jax.shard_map, mesh=mesh, in_specs=P("p"), out_specs=P("p"))
def ring(x):
    return lax.ppermute(x, axis_name="p", perm=ring_perm)

# --- correctness (one shot) ---
out = ring(g)
got = float(np.asarray(out.addressable_shards[0].data)[0])
expected = float((rank - 1) % size)
ok = abs(got - expected) < 1e-6
all_ok = comm.allreduce(1 if ok else 0, op=MPI.LAND)
print(f"[spike] rank={rank} ppermute got={got} expected={expected} {'OK' if ok else 'FAIL'}",
      flush=True)
if rank == 0:
    print(f"[spike] CORRECTNESS: {'ALL-PASS' if all_ok else 'FAIL'}", flush=True)

# --- residency stress: K repeats of the collective on already-device arrays ---
# If NCCL is device-resident, HtoD/DtoH stay ~O(1) (setup) regardless of K.
K = int(os.environ.get("SPIKE_ITERS", "200"))
x = g
comm.Barrier()
if rank == 0:
    print(f"[spike] --- residency loop: {K} ppermute iterations (nsys window) ---", flush=True)
# gate for nsys --capture-range=cudaProfilerApi (only rank0 profiles)
prof = (rank == 0) and os.environ.get("SPIKE_NSYS", "0") != "0"
if prof:
    import ctypes
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaProfilerStart()
for _ in range(K):
    x = ring(x)
    x.block_until_ready()
if prof:
    cudart.cudaProfilerStop()
# after K ring steps, value advanced by -K mod N
final = float(np.asarray(x.addressable_shards[0].data)[0])
exp_final = float((rank - K) % size)
ok2 = abs(final - exp_final) < 1e-6
all_ok2 = comm.allreduce(1 if ok2 else 0, op=MPI.LAND)
if rank == 0:
    print(f"[spike] LOOP CORRECTNESS after {K}: {'ALL-PASS' if all_ok2 else 'FAIL'}", flush=True)
    print(f"[spike] DONE", flush=True)
comm.Barrier()
