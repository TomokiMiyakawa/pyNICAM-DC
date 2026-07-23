"""Direct-NCCL halo transport via XLA FFI (nccl-ffi-plan_v1.txt, increment N2).

Strategy-2: keep 1-proc-1-GPU + local per-rank arrays + the existing mpi4py
world; swap ONLY the wire transport of the on-device COMM core from
mpi4jax.alltoall (dense, host-staged UCX) to grouped ncclSend/ncclRecv through
our OWN ncclComm_t (sparse partner rows, device-resident, GPUDirect). The
communicator is bootstrapped by broadcasting the NCCL unique id over the mpi4py
world -- NO jax.distributed, NO XLA collective runtime (where both the ragged
pe20 thunk crash and the ppermute channel wedge live).

Gate: PYNICAM_COMM_NCCLFFI=1 (default 0; checked in mod_comm, not here).
Library: tools/ncclffi/libncclffi.so (build_ncclffi.sh) or PYNICAM_NCCLFFI_LIB.
De-risked by the N0/N1 spikes (jobs 2431034/2431523): correct under jit and
lax.scan, ncclDevKernel-only nsys window, zero HtoD/DtoH, GPUDirect over IB.
"""
import ctypes
import os

import numpy as np

_lib = None
_inited = False
_plans = {}          # plan_id -> kept-alive numpy tables (ctypes safety)
_next_plan_id = [0]


def _find_lib():
    cand = os.environ.get("PYNICAM_NCCLFFI_LIB")
    if cand:
        if os.path.exists(cand):
            return cand
        raise RuntimeError(f"ncclffi: PYNICAM_NCCLFFI_LIB={cand} not found")
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.normpath(os.path.join(here, "..", "..", "tools", "ncclffi",
                                         "libncclffi.so"))
    if os.path.exists(cand):
        return cand
    raise RuntimeError("ncclffi: libncclffi.so not found -- run "
                       "tools/ncclffi/build_ncclffi.sh or set PYNICAM_NCCLFFI_LIB")


def ensure_comm(comm_world, rank, nprocs):
    """Load the extension, bootstrap the NCCL communicator (uid bcast over the
    mpi4py world -- BLOCKING collective, all ranks must call), register the FFI
    target. Idempotent."""
    global _lib, _inited
    if _inited:
        return
    import jax
    import jax.numpy as jnp

    _lib = ctypes.cdll.LoadLibrary(_find_lib())
    _lib.ncclffi_uid_size.restype = ctypes.c_int
    _lib.ncclffi_get_uid.argtypes = [ctypes.c_char_p]
    _lib.ncclffi_get_uid.restype = ctypes.c_int
    _lib.ncclffi_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                  ctypes.c_int]
    _lib.ncclffi_init.restype = ctypes.c_int
    _ll = ctypes.POINTER(ctypes.c_longlong)
    _lib.ncclffi_set_plan.argtypes = [ctypes.c_int, ctypes.c_int] + [_ll] * 5
    _lib.ncclffi_set_plan.restype = ctypes.c_int

    # Touch the device BEFORE NCCL init so XLA's CUDA (primary) context exists
    # and NCCL attaches to the same one.
    jnp.zeros(8).block_until_ready()

    usz = _lib.ncclffi_uid_size()
    if rank == 0:
        buf = ctypes.create_string_buffer(usz)
        rc = _lib.ncclffi_get_uid(buf)
        if rc != 0:
            raise RuntimeError(f"ncclffi: ncclGetUniqueId rc={rc}")
        uid = buf.raw
    else:
        uid = None
    uid = comm_world.bcast(uid, root=0)
    rc = _lib.ncclffi_init(rank, nprocs, uid, 0)   # 1 GPU/node -> device 0
    if rc != 0:
        raise RuntimeError(f"ncclffi: ncclCommInitRank rc={rc}")

    jax.ffi.register_ffi_target("nicam_halo_exchange",
                                jax.ffi.pycapsule(_lib.HaloExchange),
                                platform="CUDA")
    _inited = True
    if rank == 0:
        print(f"NCCLFFI: comm up nprocs={nprocs} lib={_find_lib()}", flush=True)


def register_plan(peers, send_off, send_cnt, recv_off, recv_cnt):
    """Register per-partner element offset/count tables host-side; returns the
    plan_id to pass as the ffi_call attr. Tables are kept alive on this module."""
    assert _inited, "ncclffi: register_plan before ensure_comm"
    pid = _next_plan_id[0]
    _next_plan_id[0] += 1
    tabs = tuple(np.ascontiguousarray(np.asarray(t, dtype=np.int64))
                 for t in (peers, send_off, send_cnt, recv_off, recv_cnt))
    _plans[pid] = tabs
    ptrs = [t.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)) for t in tabs]
    rc = _lib.ncclffi_set_plan(pid, len(tabs[0]), *ptrs)
    if rc != 0:
        raise RuntimeError(f"ncclffi: set_plan rc={rc}")
    if os.environ.get("PYNICAM_NCCLFFI_VERBOSE", "0") != "0":
        print(f"NCCLFFI: plan {pid} peers={tabs[0].tolist()} "
              f"send_cnt={tabs[2].tolist()} recv_cnt={tabs[4].tolist()}", flush=True)
    return pid
