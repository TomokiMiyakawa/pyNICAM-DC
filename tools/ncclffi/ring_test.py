#!/usr/bin/env python
"""N1 spike (nccl-ffi-plan_v1.txt): 4-rank ring exchange through our OWN NCCL
communicator via an XLA FFI handler. NO jax.distributed, NO XLA collectives —
each rank is plain single-device jax + mpi4py (the exact Strategy-2 model shape).

  T1 eager single exchange       (value from left neighbor arrives)
  T2 K-iter jitted ring loop     (nsys window: expect nccl kernels, 0 HtoD/DtoH)
  T3 same ring inside lax.scan under ONE jit (the FUSE_TIMELOOP shape)
  T4 ordering probe: TWO independent exchanges (different sizes, opposite
     directions) per scan iteration — detects XLA cross-rank reorder /
     ppermute-style wedge. Deadlock here = plan §5 R1/R2 detector.

Usage: mpirun -np 4 python ring_test.py   (SPIKE_ITERS/SPIKE_NSYS as in ppermute_spike)
"""
import ctypes
import os
import sys

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
N = comm.Get_size()

import jax                      # noqa: E402
import jax.numpy as jnp         # noqa: E402
from jax import lax             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.cdll.LoadLibrary(os.path.join(HERE, "libncclffi.so"))
lib.ncclffi_uid_size.restype = ctypes.c_int
lib.ncclffi_get_uid.argtypes = [ctypes.c_char_p]
lib.ncclffi_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

# Touch the device BEFORE NCCL init so XLA's CUDA (primary) context exists and
# NCCL attaches to the same one (mirrors production: COMM setup after backend).
_ = jnp.zeros(8).block_until_ready()

usz = lib.ncclffi_uid_size()
if rank == 0:
    buf = ctypes.create_string_buffer(usz)
    rc = lib.ncclffi_get_uid(buf)
    assert rc == 0, f"ncclGetUniqueId rc={rc}"
    uid = buf.raw
else:
    uid = None
uid = comm.bcast(uid, root=0)
rc = lib.ncclffi_init(rank, N, uid, 0)   # 1 GPU/node -> device 0
assert rc == 0, f"ncclCommInitRank rc={rc}"
if rank == 0:
    print(f"[ring] jax {jax.__version__} N={N} nccl comm up (uid {usz}B over mpi4py)", flush=True)

jax.ffi.register_ffi_target("nccl_peer_exchange", jax.ffi.pycapsule(lib.PeerExchange),
                            platform="CUDA")


def exchange(v, send_peer, recv_peer):
    return jax.ffi.ffi_call("nccl_peer_exchange",
                            jax.ShapeDtypeStruct(v.shape, v.dtype),
                            has_side_effect=True)(
        v, send_peer=np.int64(send_peer), recv_peer=np.int64(recv_peer))


right = (rank + 1) % N
left = (rank - 1) % N
M = 1 << 20                      # 1M floats = 4 MiB
results = []


def check(tag, ok):
    all_ok = comm.allreduce(bool(ok), op=MPI.LAND)
    if rank == 0:
        print(f"[ring] {tag}: {'PASS' if all_ok else 'FAIL'}", flush=True)
    results.append(all_ok)


# --- T1: eager single exchange (send right, recv from left) ---
x = jnp.full((M,), float(rank + 1), jnp.float32)
y = exchange(x, right, left)
y.block_until_ready()
check("T1 eager", np.all(np.asarray(y) == float(left + 1)))

# --- T2: K-iter jitted ring loop (the nsys memops window) ---
K = int(os.environ.get("SPIKE_ITERS", "200"))
ring_jit = jax.jit(lambda v: exchange(v, right, left))
x = jnp.full((M,), float(rank), jnp.float32)
# warm/compile OUTSIDE the nsys window, but keep its result IN the value chain
# (v2 harness bug: discarding it while still expecting K+1 advances -> T2 FAIL).
x = ring_jit(x)
x.block_until_ready()
comm.Barrier()
prof = (rank == 0) and os.environ.get("SPIKE_NSYS", "0") != "0"
if prof:
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaProfilerStart()
y = x
for _ in range(K):
    y = ring_jit(y)
    y.block_until_ready()
if prof:
    cudart.cudaProfilerStop()
# K+1 exchanges total (warm + K): value advanced by -(K+1) mod N; with K%N==0
# this differs from the start value, so a silent no-op exchange can't pass.
got = np.asarray(y)
exp = float((rank - K - 1) % N)
if rank == 0 and not np.all(got == exp):
    print(f"[ring] T2 debug: got={got[0]} expected={exp}", flush=True)
check(f"T2 jit x{K}", np.all(got == exp))

# --- T3: ring inside lax.scan under ONE jit ---
def scan_ring(v):
    def body(c, _):
        return exchange(c, right, left), ()
    out, _ = lax.scan(body, v, None, length=K)
    return out

y3 = jax.jit(scan_ring)(jnp.full((M,), float(rank), jnp.float32))
y3.block_until_ready()
check(f"T3 jit+scan x{K}", np.all(np.asarray(y3) == float((rank - K) % N)))

# --- T4: ordering probe -- two INDEPENDENT exchanges per scan iter ---
# a: rightward ring, size M; b: leftward ring, size M2 != M. No data dependence
# between them inside an iteration -> if XLA reorders them differently across
# ranks, the count mismatch wedges/errors instead of silently passing.
M2 = (1 << 19) + 3
K4 = int(os.environ.get("SPIKE_ITERS4", "100"))


def scan_two(a, b):
    def body(c, _):
        ca, cb = c
        na = exchange(ca, right, left)
        nb = exchange(cb, left, right)
        return (na, nb), ()
    (fa, fb), _ = lax.scan(body, (a, b), None, length=K4)
    return fa, fb

fa, fb = jax.jit(scan_two)(jnp.full((M,), float(rank), jnp.float32),
                           jnp.full((M2,), float(rank) + 0.5, jnp.float32))
fa.block_until_ready(); fb.block_until_ready()
ok4 = (np.all(np.asarray(fa) == float((rank - K4) % N))
       and np.all(np.asarray(fb) == float((rank + K4) % N) + 0.5))
check(f"T4 2-exchange scan x{K4}", ok4)

verdict = "ALL OK" if all(results) else "SOME FAILED"
print(f"[ring] rank{rank} {verdict}", flush=True)
comm.Barrier()
lib.ncclffi_finalize()
sys.exit(0 if all(results) else 1)
