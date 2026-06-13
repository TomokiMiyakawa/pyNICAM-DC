"""
Validation harness for proto/diag_kernel.py.

Checks that the pure backend-switchable kernel reproduces, bit-for-bit, the
original in-place block of mod_dynamics.py (L446-506) for:
    (1) xp = numpy        (eager)
    (2) xp = jax.numpy    (eager)
    (3) jax.jit(kernel)   (compiled, xp=jax.numpy)

Run:
    .../envs/jax_nomtl_mpi/bin/python test_diag_kernel.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

# allow running standalone: add repo root (containing the `pynicamdc` package)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.diag import DiagCfg, compute_diagnostics


# ---------------------------------------------------------------------------
# Reference: exact transcription of mod_dynamics.py L446-506 (in-place, numpy)
# ---------------------------------------------------------------------------
def compute_diagnostics_ref(PROG, PROGq, DIAG_in, GSGAM2, C2Wfact, CVW, cfg):
    I_RHOG, I_RHOGVX, I_RHOGVY = cfg.I_RHOG, cfg.I_RHOGVX, cfg.I_RHOGVY
    I_RHOGVZ, I_RHOGW, I_RHOGE = cfg.I_RHOGVZ, cfg.I_RHOGW, cfg.I_RHOGE
    I_pre, I_tem = cfg.I_pre, cfg.I_tem
    I_vx, I_vy, I_vz, I_w = cfg.I_vx, cfg.I_vy, cfg.I_vz, cfg.I_w
    kmin, kmax = cfg.kmin, cfg.kmax
    nmin, nmax, iqv = cfg.nmin, cfg.nmax, cfg.iqv
    Rdry, Rvap, CVdry = cfg.Rdry, cfg.Rvap, cfg.CVdry

    DIAG = DIAG_in.copy()
    rho = np.empty_like(GSGAM2)
    ein = np.empty_like(GSGAM2)
    q = np.empty_like(PROGq)
    cv = np.empty_like(GSGAM2)
    qd = np.empty_like(GSGAM2)

    RHOG    = PROG[:, :, :, :, I_RHOG]
    RHOGVX  = PROG[:, :, :, :, I_RHOGVX]
    RHOGVY  = PROG[:, :, :, :, I_RHOGVY]
    RHOGVZ  = PROG[:, :, :, :, I_RHOGVZ]
    RHOGE   = PROG[:, :, :, :, I_RHOGE]

    rho[:, :, :, :] = RHOG / GSGAM2
    DIAG[:, :, :, :, I_vx] = RHOGVX / RHOG
    DIAG[:, :, :, :, I_vy] = RHOGVY / RHOG
    DIAG[:, :, :, :, I_vz] = RHOGVZ / RHOG
    ein[:, :, :, :] = RHOGE / RHOG

    q[:, :, :, :, :] = PROGq / PROG[:, :, :, :, np.newaxis, I_RHOG]

    cv.fill(0.0)
    qd.fill(1.0)
    q_slice = q[:, :, :, :, nmin:nmax + 1]
    CVW_slice = CVW[nmin:nmax + 1]
    cv += np.sum(q_slice * CVW_slice[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :], axis=4)
    qd -= np.sum(q_slice, axis=4)
    cv += qd * CVdry

    DIAG[:, :, :, :, I_tem] = ein / cv
    DIAG[:, :, :, :, I_pre] = rho * DIAG[:, :, :, :, I_tem] * (qd * Rdry + q[:, :, :, :, iqv] * Rvap)

    numerator = PROG[:, :, kmin + 1:kmax + 1, :, I_RHOGW]
    rhog_k    = PROG[:, :, kmin + 1:kmax + 1, :, I_RHOG]
    rhog_km1  = PROG[:, :, kmin:kmax,         :, I_RHOG]
    fact1 = C2Wfact[:, :, kmin + 1:kmax + 1, :, 0]
    fact2 = C2Wfact[:, :, kmin + 1:kmax + 1, :, 1]
    denominator = fact1 * rhog_k + fact2 * rhog_km1
    DIAG[:, :, kmin + 1:kmax + 1, :, I_w] = numerator / denominator

    return rho, DIAG, ein, q, cv, qd


# ---------------------------------------------------------------------------
# Random test data (positive RHOG so divisions are well-defined)
# ---------------------------------------------------------------------------
def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    iall, jall, kall, lall, nq = 5, 4, 10, 3, 6
    dt = np.float64

    PROG = np.empty((iall, jall, kall, lall, 6), dtype=dt)
    PROG[..., 0] = rng.uniform(0.5, 1.5, (iall, jall, kall, lall))   # RHOG > 0
    PROG[..., 1:] = rng.uniform(-1.0, 1.0, (iall, jall, kall, lall, 5))
    PROGq = rng.uniform(0.0, 0.02, (iall, jall, kall, lall, nq)).astype(dt)
    DIAG_in = rng.uniform(-1.0, 1.0, (iall, jall, kall, lall, 6)).astype(dt)
    GSGAM2 = rng.uniform(0.5, 1.5, (iall, jall, kall, lall)).astype(dt)
    C2Wfact = rng.uniform(0.3, 0.7, (iall, jall, kall, lall, 2)).astype(dt)
    CVW = rng.uniform(700.0, 1900.0, (nq,)).astype(dt)

    cfg = DiagCfg(
        I_RHOG=0, I_RHOGVX=1, I_RHOGVY=2, I_RHOGVZ=3, I_RHOGW=4, I_RHOGE=5,
        I_pre=0, I_tem=1, I_vx=2, I_vy=3, I_vz=4, I_w=5,
        kmin=1, kmax=8, nmin=0, nmax=2, iqv=0,
        Rdry=287.04, Rvap=461.50, CVdry=717.60,
    )
    return PROG, PROGq, DIAG_in, GSGAM2, C2Wfact, CVW, cfg


def report(name, ref, got, rtol=0.0, atol=0.0):
    """Compare. rtol/atol=0 demands bit-exactness; a small tol tolerates the
    rounding-order changes XLA introduces under jit (fusion / FMA)."""
    names = ["rho", "DIAG", "ein", "q", "cv", "qd"]
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
        print(f"  {nm:5s}: {'OK ' if passed else 'BAD'} {flag}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    PROG, PROGq, DIAG_in, GSGAM2, C2Wfact, CVW, cfg = make_inputs()

    ref = compute_diagnostics_ref(PROG, PROGq, DIAG_in, GSGAM2, C2Wfact, CVW, cfg)

    # (1) numpy backend
    import numpy as xnp
    got_np = compute_diagnostics(PROG, PROGq, DIAG_in, GSGAM2, C2Wfact, CVW, cfg, xnp)
    ok1 = report("numpy backend (eager)", ref, got_np)

    # (2) + (3) jax backend
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    args = (jnp.asarray(PROG), jnp.asarray(PROGq), jnp.asarray(DIAG_in),
            jnp.asarray(GSGAM2), jnp.asarray(C2Wfact), jnp.asarray(CVW))

    got_jnp = compute_diagnostics(*args, cfg, jnp)
    ok2 = report("jax.numpy backend (eager)", ref, got_jnp)  # expect bit-exact

    kernel_jit = jax.jit(compute_diagnostics, static_argnames=("cfg", "xp"))
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
