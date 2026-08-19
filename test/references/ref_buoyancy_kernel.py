"""
Validation harness for kernels/buoyancy.py.

Checks that the pure backend-switchable kernel reproduces, bit-for-bit, the
original in-place src_buoyancy block of mod_src.py (L1044-1088) for:
    (1) xp = numpy        (eager)
    (2) xp = jax.numpy    (eager)
    (3) jax.jit(kernel)   (compiled, xp=jax.numpy)

Run:
    .../envs/jax_nomtl_mpi/bin/python test_buoyancy_kernel.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

# allow running standalone: add repo root (containing the `pynicamdc` package)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.buoyancy import BuoyCfg, compute_buoyancy


# ---------------------------------------------------------------------------
# Reference: exact transcription of mod_src.py src_buoyancy (in-place, numpy)
# ---------------------------------------------------------------------------
def compute_buoyancy_ref(rhog, rhog_pl, C2Wfact, C2Wfact_pl, cfg):
    kmin, kmax, grav = cfg.kmin, cfg.kmax, cfg.GRAV

    buoiw = np.zeros_like(rhog)
    for l in range(rhog.shape[3]):
        for k in range(kmin + 1, kmax + 1):
            buoiw[:, :, k, l] = -grav * (
                C2Wfact[:, :, k, l, 0] * rhog[:, :, k, l] +
                C2Wfact[:, :, k, l, 1] * rhog[:, :, k - 1, l]
            )
        buoiw[:, :, kmin - 1, l] = 0.0
        buoiw[:, :, kmin,     l] = 0.0
        buoiw[:, :, kmax + 1, l] = 0.0

    buoiw_pl = np.zeros_like(rhog_pl)
    if cfg.have_pl:
        buoiw_pl[:, kmin + 1:kmax + 1, :] = -grav * (
            C2Wfact_pl[:, kmin + 1:kmax + 1, :, 0] * rhog_pl[:, kmin + 1:kmax + 1, :] +
            C2Wfact_pl[:, kmin + 1:kmax + 1, :, 1] * rhog_pl[:, kmin:kmax, :]
        )
        buoiw_pl[:, kmin - 1, :] = 0.0
        buoiw_pl[:, kmin,     :] = 0.0
        buoiw_pl[:, kmax + 1, :] = 0.0

    return buoiw, buoiw_pl


# ---------------------------------------------------------------------------
# Random test data (kmin=1, kmax=kall-2 so boundary coverage is complete)
# ---------------------------------------------------------------------------
def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    iall, jall, kall, lall = 5, 4, 10, 3
    gall_pl, lall_pl = 6, 2
    dt = np.float64
    kmin, kmax = 1, kall - 2

    rhog       = rng.uniform(0.5, 1.5, (iall, jall, kall, lall)).astype(dt)
    rhog_pl    = rng.uniform(0.5, 1.5, (gall_pl, kall, lall_pl)).astype(dt)
    C2Wfact    = rng.uniform(0.3, 0.7, (iall, jall, kall, lall, 2)).astype(dt)
    C2Wfact_pl = rng.uniform(0.3, 0.7, (gall_pl, kall, lall_pl, 2)).astype(dt)

    cfg = BuoyCfg(kmin=kmin, kmax=kmax, GRAV=9.80616, have_pl=True)
    return rhog, rhog_pl, C2Wfact, C2Wfact_pl, cfg


def report(name, ref, got, rtol=0.0, atol=0.0):
    """Compare. rtol/atol=0 demands bit-exactness; a small tol tolerates the
    rounding-order changes XLA introduces under jit (fusion / FMA)."""
    names = ["buoiw", "buoiw_pl"]
    print(f"\n[{name}]  (rtol={rtol:g}, atol={atol:g})")
    ok = True
    for nm, r, g in zip(names, ref, got):
        g = np.asarray(g)
        denom = np.maximum(np.abs(r), 1e-300)
        max_abs = np.max(np.abs(g - r))
        max_rel = np.max(np.abs(g - r) / denom)
        exact = np.array_equal(g, r)
        passed = exact or np.allclose(g, r, rtol=rtol, atol=atol)
        ok = ok and passed
        flag = "EXACT" if exact else f"max|d|={max_abs:.3e} max|rel|={max_rel:.3e}"
        print(f"  {nm:9s}: {'OK ' if passed else 'BAD'} {flag}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    rhog, rhog_pl, C2Wfact, C2Wfact_pl, cfg = make_inputs()

    ref = compute_buoyancy_ref(rhog, rhog_pl, C2Wfact, C2Wfact_pl, cfg)

    # (1) numpy backend
    import numpy as xnp
    got_np = compute_buoyancy(rhog, rhog_pl, C2Wfact, C2Wfact_pl, cfg, xnp)
    ok1 = report("numpy backend (eager)", ref, got_np)

    # (2) + (3) jax backend
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    args = (jnp.asarray(rhog), jnp.asarray(rhog_pl),
            jnp.asarray(C2Wfact), jnp.asarray(C2Wfact_pl))

    got_jnp = compute_buoyancy(*args, cfg, jnp)
    ok2 = report("jax.numpy backend (eager)", ref, got_jnp)  # expect bit-exact

    kernel_jit = jax.jit(compute_buoyancy, static_argnames=("cfg", "xp"))
    got_jit = kernel_jit(*args, cfg=cfg, xp=jnp)
    jax.block_until_ready(got_jit)
    # jit fuses ops -> rounding-order differences; require tight relative tol only
    ok3 = report("jax.jit (compiled)", ref, got_jit, rtol=1e-12, atol=1e-12)

    print("\n========================================")
    print(f"numpy : {'PASS' if ok1 else 'FAIL'}")
    print(f"jnp   : {'PASS' if ok2 else 'FAIL'}")
    print(f"jit   : {'PASS' if ok3 else 'FAIL'}")
    print("========================================")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
