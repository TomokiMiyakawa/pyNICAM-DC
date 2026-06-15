"""
Fusion ceiling microbenchmark for pyNICAM-DC small-step compute.

Goal: estimate the realistic per-call speedup of fusing a representative
comm-free, compute-bound block into ONE jit graph (device-resident state),
versus the current numpy backend -- to gauge where a fully-JAX CPU build
could land relative to the Fortran90 original (numpy is ~6-7x slower than F90).

NOT a bit-exact extraction. It is a synthetic proxy on the REAL array shapes
(glevel=5, rlevel=1, vlayer=40, 8 ranks -> per-rank (18,18,42,5)) with an
op-mix matched to the small-step core:
  A) horizontal 7-point hexagonal stencil (OPRT divergence/gradient-like; gather/shift)
  B) vertical tridiagonal Thomas solve along k (vi_rhow_solver-like; serial recurrence)
  C) elementwise tendency summation + state update (rhogkin/diff_vh-like)

Three timing scenarios:
  1. numpy
  2. jax-jit, inputs already on device (fused-future: no per-call transfer)
  3. jax-jit, asarray(input)+to_numpy(output) each call (current per-call dispatch)
"""

import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

# ----- real per-rank shapes for glevel=5, rlevel=1, vlayer=40, 8 ranks -----
GALL_1D = 18          # 2^(gl-rl)+2 = 16+2
KALL    = 42          # vlayer + 2
LALL    = 5           # 10*4^rl / nprc = 40/8
SHAPE   = (GALL_1D, GALL_1D, KALL, LALL)
DT      = np.float64(20.0)


def make_inputs(xp, seed=0):
    rng = np.random.default_rng(seed)
    def r(shape, lo=0.1, hi=1.0):
        a = rng.uniform(lo, hi, size=shape).astype(np.float64)
        return xp.asarray(a)
    d = {
        # prognostic-like state
        "rhog":   r(SHAPE),
        "rhogvx": r(SHAPE, -1.0, 1.0),
        "rhogvy": r(SHAPE, -1.0, 1.0),
        "rhogvz": r(SHAPE, -1.0, 1.0),
        "rhogw":  r(SHAPE, -1.0, 1.0),
        "rhoge":  r(SHAPE),
        # tendencies
        "g_rhogvx": r(SHAPE, -0.1, 0.1),
        "g_rhogvy": r(SHAPE, -0.1, 0.1),
        "g_rhogvz": r(SHAPE, -0.1, 0.1),
        # horizontal stencil coefficients: 7 neighbors x 3 directions
        "hcoef":  r((GALL_1D, GALL_1D, LALL, 7, 3), -0.5, 0.5),
        # vertical tridiagonal coefficients (a sub, b diag, c super, made diag-dominant)
        "t_a":    r(SHAPE, -0.2, -0.05),
        "t_c":    r(SHAPE, -0.2, -0.05),
        "t_b":    r(SHAPE,  1.5,  2.0),
    }
    return d


def hstencil(field, hcoef_dir, xp):
    """7-point hexagonal stencil over (i,j) for one direction component.
    field: (i,j,k,l); hcoef_dir: (i,j,l,7) -> out (i,j,k,l), interior only,
    edges left as field (boundary passthrough)."""
    out = xp.zeros_like(field)
    # neighbor shifts: center, +i, +i+j, +j, -i, -i-j, -j
    c = hcoef_dir  # (i,j,l,7)
    acc = (
        c[1:-1, 1:-1, None, :, 0] * field[1:-1, 1:-1, :, :]
        + c[1:-1, 1:-1, None, :, 1] * field[2:,   1:-1, :, :]
        + c[1:-1, 1:-1, None, :, 2] * field[2:,   2:,   :, :]
        + c[1:-1, 1:-1, None, :, 3] * field[1:-1, 2:,   :, :]
        + c[1:-1, 1:-1, None, :, 4] * field[:-2,  1:-1, :, :]
        + c[1:-1, 1:-1, None, :, 5] * field[:-2,  :-2,  :, :]
        + c[1:-1, 1:-1, None, :, 6] * field[1:-1, :-2,  :, :]
    )
    # place interior
    if xp is np:
        out[1:-1, 1:-1, :, :] = acc
        return out
    else:
        return out.at[1:-1, 1:-1, :, :].set(acc)


def thomas_k(a, b, c, d, xp):
    """Solve tridiagonal system along axis=2 (k). a,b,c,d: (i,j,k,l).
    Serial recurrence over KALL levels -> representative of vi_rhow_solver."""
    K = a.shape[2]
    # forward sweep
    cp = [None] * K
    dp = [None] * K
    cp[0] = c[:, :, 0, :] / b[:, :, 0, :]
    dp[0] = d[:, :, 0, :] / b[:, :, 0, :]
    for k in range(1, K):
        m = b[:, :, k, :] - a[:, :, k, :] * cp[k - 1]
        cp[k] = c[:, :, k, :] / m
        dp[k] = (d[:, :, k, :] - a[:, :, k, :] * dp[k - 1]) / m
    # back substitution
    x = [None] * K
    x[K - 1] = dp[K - 1]
    for k in range(K - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return xp.stack(x, axis=2)


def compute_block(d, xp):
    """One small-step-like update. Returns updated (rhogvx, rhogvy, rhogvz, rhoge)."""
    # A) horizontal stencil: pressure-gradient/divergence-like contributions
    dvx = hstencil(d["rhogvx"], d["hcoef"][..., 0], xp)
    dvy = hstencil(d["rhogvy"], d["hcoef"][..., 1], xp)
    dvz = hstencil(d["rhogvz"], d["hcoef"][..., 2], xp)

    # C-part) elementwise tendency summation
    new_vx = d["rhogvx"] + (d["g_rhogvx"] + dvx) * DT
    new_vy = d["rhogvy"] + (d["g_rhogvy"] + dvy) * DT
    new_vz = d["rhogvz"] + (d["g_rhogvz"] + dvz) * DT

    # B) vertical implicit solve for an energy-like variable
    # rhs depends on the just-updated momenta (couples A/C into B)
    rhs = d["rhoge"] + DT * (new_vx * new_vx + new_vy * new_vy + new_vz * new_vz) / d["rhog"]
    new_ge = thomas_k(d["t_a"], d["t_b"], d["t_c"], rhs, xp)

    return new_vx, new_vy, new_vz, new_ge


def time_it(fn, n_iter, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) / n_iter


def main():
    N = 200

    # ---------- numpy ----------
    dn = make_inputs(np)
    np_call = lambda: compute_block(dn, np)
    t_np = time_it(np_call, N)

    # ---------- jax: build jitted block ----------
    block_jit = jax.jit(partial(compute_block, xp=jnp))

    # device-resident inputs (fused-future scenario)
    dj = make_inputs(jnp)
    # warmup + compile timing
    tc0 = time.perf_counter()
    out = block_jit(dj)
    jax.block_until_ready(out)
    t_compile = time.perf_counter() - tc0

    def jax_resident():
        out = block_jit(dj)
        jax.block_until_ready(out)
    t_jax_res = time_it(jax_resident, N)

    # per-call host<->device transfer scenario (current architecture)
    dn_host = make_inputs(np)  # numpy host arrays
    def jax_transfer():
        din = {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in dn_host.items()}
        out = block_jit(din)
        _ = [np.asarray(o) for o in jax.block_until_ready(out)]
    t_jax_xfer = time_it(jax_transfer, N)

    # ---------- report ----------
    us = 1e6
    print(f"array shape (per rank)      : {SHAPE}  ({np.prod(SHAPE):,} elems, "
          f"{np.prod(SHAPE)*8/1e6:.2f} MB/array, float64)")
    print(f"iterations timed            : {N}")
    print()
    print(f"1) numpy                    : {t_np*1e3:8.3f} ms/call   (baseline)")
    print(f"2) jax-jit device-resident  : {t_jax_res*1e3:8.3f} ms/call   "
          f"-> {t_np/t_jax_res:5.2f}x vs numpy")
    print(f"3) jax-jit + host transfer  : {t_jax_xfer*1e3:8.3f} ms/call   "
          f"-> {t_np/t_jax_xfer:5.2f}x vs numpy")
    print()
    print(f"jax first-call compile time : {t_compile*1e3:8.1f} ms (one-time)")
    print()
    # crude correctness sanity: compare numpy vs jax device-resident on same seed
    a = compute_block(make_inputs(np, seed=7), np)
    b = compute_block(make_inputs(jnp, seed=7), jnp)
    maxrel = max(float(np.max(np.abs(np.asarray(bi) - ai) / (np.abs(ai) + 1e-30)))
                 for ai, bi in zip(a, b))
    print(f"numpy vs jax max rel diff   : {maxrel:.3e}  (sanity, same seed)")


if __name__ == "__main__":
    main()
