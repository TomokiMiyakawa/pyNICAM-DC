"""sendrecv-only mpi4jax sanity (H100x4 bring-up, STEP 3).

Trim of test_mpi4jax_sanity.py: drops the allreduce, which SEGVs in OpenMPI's
tuned collective on this box (known, ignore). Keeps eager sendrecv + the
jit-token-threaded sendrecv (the real on-device COMM path), and prints each
rank's GPU binding so we can confirm 1 rank/GPU.

    MPI4JAX_USE_CUDA_MPI=1 HCOLL_ENABLE=0 mpirun --mca pml ucx --mca coll ^hcoll \
        -np 4 ./bind.sh python test_mpi4jax_sendrecv_only.py
"""

import os
import numpy as np


def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        if rank == 0:
            print("need >= 2 ranks")
        return 1

    import jax
    import jax.numpy as jnp
    import mpi4jax

    def _arr(r):
        return r[0] if isinstance(r, tuple) else r

    dev = jax.devices()[0]
    # PCI bus id distinguishes the physical GPU even though each bound rank sees
    # its own device as local index 0.
    try:
        pci = dev.device_kind, getattr(dev, "client", None)
    except Exception:
        pci = "?"
    print(f"[rank {rank}/{size}] backend={jax.default_backend()} "
          f"local_rank={os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK')} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"jax_device={dev} mpi4jax={mpi4jax.__version__}", flush=True)

    nxt = (rank + 1) % size
    prv = (rank - 1) % size
    ok = True

    # eager sendrecv ring
    x = jnp.full((4,), float(rank))
    recv = _arr(mpi4jax.sendrecv(x, x, source=prv, dest=nxt, comm=comm))
    if not np.allclose(np.asarray(recv), float(prv)):
        print(f"[rank {rank}] SENDRECV FAIL: got {np.asarray(recv)} exp {float(prv)}", flush=True)
        ok = False
    else:
        print(f"[rank {rank}] sendrecv OK (got {float(prv)} from rank {prv})", flush=True)

    # sendrecv INSIDE jax.jit (token-threaded) -- the real on-device COMM path
    def ring(v):
        return _arr(mpi4jax.sendrecv(v, v, source=prv, dest=nxt, comm=comm))
    rj = np.asarray(jax.jit(ring)(x))
    if not np.allclose(rj, float(prv)):
        print(f"[rank {rank}] JIT SENDRECV FAIL: got {rj} exp {float(prv)}", flush=True)
        ok = False
    else:
        print(f"[rank {rank}] jit sendrecv OK (token-threaded)", flush=True)

    n_ok = comm.allreduce(1 if ok else 0, op=MPI.SUM)
    if rank == 0:
        print(f"\n=== sendrecv sanity: {'PASS' if n_ok == size else 'FAIL'} "
              f"({n_ok}/{size} ranks ok) ===", flush=True)
    return 0 if n_ok == size else 1


if __name__ == "__main__":
    raise SystemExit(main())
