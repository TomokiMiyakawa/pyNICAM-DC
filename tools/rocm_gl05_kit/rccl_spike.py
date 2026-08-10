#!/usr/bin/env python
"""Standalone exercise of the PRODUCTION `nicam_halo_exchange` FFI handler on RCCL.

Isolates the RCCL wire from the dycore. The model run faults with "Memory access
fault by GPU" a few seconds after `NCCLFFI: comm up`, and a full pe8 model run costs
~5 min of XLA compile per attempt -- this reproduces (or clears) the same handler,
same plan shape, in seconds.

Plan shape mirrors mod_comm's alltoall layout exactly: the send buffer is a dense
(nprocs, chunk) tensor whose row q holds the payload destined for rank q; each partner
contributes one ncclSend of row q and one ncclRecv into row q. Payload is tagged
sender/dest/index so the check is exact rather than statistical:

    st[me][q][k] = me*1000 + q*10 + k   ->   rt[me][p][k] must be p*1000 + me*10 + k

  srun --ntasks=8 ... bind_lumi.sh python rccl_spike.py [--chunk N] [--iters N] [--ring]

--ring restricts partners to the two ring neighbours (the sparse case); default is
all-to-all (every other rank is a partner), which is the denser stress.
"""
import argparse
import ctypes
import os
import sys

import numpy as np
from mpi4py import MPI

ap = argparse.ArgumentParser()
ap.add_argument("--chunk", type=int, default=1024, help="elements per partner row")
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--ring", action="store_true", help="ring neighbours only")
ap.add_argument("--jit", action="store_true", help="run the exchange under jit")
a = ap.parse_args()

comm = MPI.COMM_WORLD
rank, N = comm.Get_rank(), comm.Get_size()

import jax                      # noqa: E402
import jax.numpy as jnp         # noqa: E402

lib_path = os.environ.get(
    "PYNICAM_NCCLFFI_LIB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "ncclffi", "rocm", "libncclffi.so"))
lib = ctypes.cdll.LoadLibrary(lib_path)
lib.ncclffi_uid_size.restype = ctypes.c_int
lib.ncclffi_get_uid.argtypes = [ctypes.c_char_p]
lib.ncclffi_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
_ll = ctypes.POINTER(ctypes.c_longlong)
lib.ncclffi_set_plan.argtypes = [ctypes.c_int, ctypes.c_int] + [_ll] * 5

if rank == 0:
    print(f"[spike] jax {jax.__version__} N={N} lib={lib_path}", flush=True)
    print(f"[spike] devices={jax.devices()} chunk={a.chunk} ring={a.ring} jit={a.jit}",
          flush=True)

# touch the device before comm init so XLA's context exists first (as production does)
jnp.zeros(8).block_until_ready()

usz = lib.ncclffi_uid_size()
uid = None
if rank == 0:
    buf = ctypes.create_string_buffer(usz)
    assert lib.ncclffi_get_uid(buf) == 0
    uid = buf.raw
uid = comm.bcast(uid, root=0)
rc = lib.ncclffi_init(rank, N, uid, 0)
assert rc == 0, f"ncclCommInitRank rc={rc}"
comm.Barrier()
if rank == 0:
    print("[spike] comm up", flush=True)

jax.ffi.register_ffi_target("nicam_halo_exchange", jax.ffi.pycapsule(lib.HaloExchange),
                            platform=os.environ.get("PYNICAM_FFI_PLATFORM", "ROCM"))

CH = a.chunk
peers = ([(rank - 1) % N, (rank + 1) % N] if a.ring
         else [p for p in range(N) if p != rank])
peers = sorted(set(peers))
tabs = [np.asarray(peers, np.int64),
        np.asarray([p * CH for p in peers], np.int64),   # send_off: row p
        np.asarray([CH] * len(peers), np.int64),         # send_cnt
        np.asarray([p * CH for p in peers], np.int64),   # recv_off: row p
        np.asarray([CH] * len(peers), np.int64)]         # recv_cnt
tabs = [np.ascontiguousarray(t) for t in tabs]
ptrs = [t.ctypes.data_as(_ll) for t in tabs]
assert lib.ncclffi_set_plan(0, len(peers), *ptrs) == 0
if rank == 0:
    print(f"[spike] plan: {len(peers)} partners, chunk={CH} elems "
          f"({CH*8} B per pair), buffer {(N, CH)} f64", flush=True)


def exchange(st, tok):
    return jax.ffi.ffi_call(
        "nicam_halo_exchange",
        (jax.ShapeDtypeStruct((N, CH), jnp.float64),
         jax.ShapeDtypeStruct((1,), jnp.float32)),
        has_side_effect=True)(st, tok, plan_id=np.int64(0))


run = jax.jit(exchange) if a.jit else exchange

k = np.arange(CH, dtype=np.float64)
st_np = np.empty((N, CH), np.float64)
for q in range(N):
    st_np[q] = rank * 1000.0 + q * 10.0 + k
st = jnp.asarray(st_np)

bad = 0
for it in range(a.iters):
    tok = jnp.zeros((1,), jnp.float32)
    rt, tok = run(st, tok)
    rt = np.asarray(jax.block_until_ready(rt))
    for p in peers:
        want = p * 1000.0 + rank * 10.0 + k
        if not np.array_equal(rt[p], want):
            bad += 1
            d = np.argmax(rt[p] != want)
            print(f"[spike] rank{rank} iter{it} MISMATCH from peer {p}: "
                  f"rt[{p}][{d}]={rt[p][d]} want {want[d]}", flush=True)
    comm.Barrier()
    if rank == 0:
        print(f"[spike] iter {it} ok", flush=True)

nbad = comm.allreduce(bad, op=MPI.SUM)
if rank == 0:
    print(f"[spike] RESULT: {'PASS' if nbad == 0 else 'FAIL'} "
          f"({nbad} mismatching partner rows over {a.iters} iters)", flush=True)
sys.exit(0 if nbad == 0 else 1)
