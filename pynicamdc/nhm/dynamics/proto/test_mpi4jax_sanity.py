"""Standalone mpi4jax sanity check for the GPU on-device COMM work (Phase 0.3 of
GPU_PORTING_PLAN.md).

Run with 2+ ranks on the GPU box:

    mpirun -np 2 python test_mpi4jax_sanity.py

It verifies, on whatever jax backend is active (cpu or gpu):

  1. jax sees a device; prints the platform per rank (cpu / gpu).
  2. mpi4jax.sendrecv ring exchange is numerically correct (eager).
  3. mpi4jax.allreduce(SUM) is correct (eager).
  4. the SAME sendrecv works *inside jax.jit* (token-threaded) -- this is exactly
     the path the real on-device COMM (mpi4jax replacement for
     COMM_data_transfer) will use, so it must work.

If the backend prints "gpu" and these pass, mpi4jax is functional; whether the
exchange is true device-to-device depends on the MPI being CUDA-aware (a perf
property, not tested here -- watch the profiler / nsys later).

Exit code 0 = all ranks passed. Heavy deps (mpi4py / jax / mpi4jax) are imported
inside main() so this file stays import-clean for pytest collection in CI.
"""

import sys  # noqa: F401
import numpy as np


def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 2:
        if rank == 0:
            print("need >= 2 ranks:  mpirun -np 2 python test_mpi4jax_sanity.py")
        return 1

    import jax
    import jax.numpy as jnp
    import mpi4jax

    # mpi4jax >= 0.9 returns the array directly; older versions returned
    # (array, token). Normalise so this test works on both.
    def _arr(r):
        return r[0] if isinstance(r, tuple) else r

    plat = jax.default_backend()
    print(f"[rank {rank}/{size}] jax backend={plat} device={jax.devices()[0]} "
          f"mpi4jax={mpi4jax.__version__}", flush=True)

    nxt = (rank + 1) % size
    prv = (rank - 1) % size
    ok = True

    # 2. eager sendrecv ring: send my value to nxt, receive prv's value
    x = jnp.full((4,), float(rank))
    recv = _arr(mpi4jax.sendrecv(x, x, source=prv, dest=nxt, comm=comm))
    if not np.allclose(np.asarray(recv), float(prv)):
        print(f"[rank {rank}] SENDRECV FAIL: got {np.asarray(recv)} expected {float(prv)}", flush=True)
        ok = False
    else:
        print(f"[rank {rank}] sendrecv OK (received {float(prv)} from rank {prv})", flush=True)

    # 3. eager allreduce(SUM)
    s = _arr(mpi4jax.allreduce(jnp.full((1,), float(rank)), op=MPI.SUM, comm=comm))
    exp_sum = float(sum(range(size)))
    if abs(float(np.asarray(s)[0]) - exp_sum) > 1e-6:
        print(f"[rank {rank}] ALLREDUCE FAIL: {float(np.asarray(s)[0])} != {exp_sum}", flush=True)
        ok = False
    else:
        print(f"[rank {rank}] allreduce OK (sum={exp_sum})", flush=True)

    # 4. sendrecv INSIDE jax.jit (token-threaded) -- the real on-device COMM path
    def ring(v):
        return _arr(mpi4jax.sendrecv(v, v, source=prv, dest=nxt, comm=comm))
    rj = np.asarray(jax.jit(ring)(x))
    if not np.allclose(rj, float(prv)):
        print(f"[rank {rank}] JIT SENDRECV FAIL: got {rj} expected {float(prv)}", flush=True)
        ok = False
    else:
        print(f"[rank {rank}] jit sendrecv OK (token-threaded)", flush=True)

    n_ok = comm.allreduce(1 if ok else 0, op=MPI.SUM)
    if rank == 0:
        print(f"\n=== mpi4jax sanity: {'PASS' if n_ok == size else 'FAIL'} "
              f"({n_ok}/{size} ranks ok) ===")
        print(f"=== jax backend = {plat}  (gpu + CUDA-aware MPI => device-to-device) ===")
    return 0 if n_ok == size else 1


if __name__ == "__main__":
    raise SystemExit(main())
