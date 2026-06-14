"""
Validation harness for kernels/virhowsolver.py.

Checks that the pure backend-switchable Thomas-solve kernels reproduce the
in-place reference (exact transcription of mod_vi.py vi_rhow_solver) for:
    (1) xp = numpy        (eager)
    (2) xp = jax.numpy    (eager)
    (3) jax.jit(kernel)   (compiled, xp=jax.numpy; k-loop unrolled in trace)

Run:
    .../envs/jax_nomtl_mpi/bin/python test_virhowsolver_kernel.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.virhowsolver import (
    ViSolverCfg, compute_rhow_solver_reg, compute_rhow_solver_pl,
)


# ---------------------------------------------------------------------------
# References: exact in-place transcription of mod_vi.py vi_rhow_solver.
# ---------------------------------------------------------------------------
def solver_reg_ref(rhogw_in, rhogw0, preg0, rhog0, Srho, Sw, Spre,
                   Mc, Mu, Ml, RGAMH, RGSGAM2, RGAM, RGSGAM2H, GSGAM2H,
                   rdgzh, afact, bfact, dt, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    iall, jall, kall, lall = rhogw_in.shape
    rhogw = rhogw_in.copy()
    Sall = np.zeros_like(rhogw)
    gamma = np.zeros_like(rhogw)
    alpha = np.float64(cfg.alpha)
    CVovRt2 = cfg.CVdry / cfg.Rdry / (dt * dt)
    GRAV = cfg.GRAV
    for l in range(lall):
        for k in range(kmin + 1, kmax + 1):
            Sall[:, :, k, l] = (
                (rhogw0[:, :, k, l] * alpha + dt * Sw[:, :, k, l]) * RGAMH[:, :, k, l]**2
                - (
                    (preg0[:, :, k, l] + dt * Spre[:, :, k, l]) * RGSGAM2[:, :, k, l]
                    - (preg0[:, :, k - 1, l] + dt * Spre[:, :, k - 1, l]) * RGSGAM2[:, :, k - 1, l]
                ) * dt * rdgzh[k]
                - (
                    (rhog0[:, :, k, l] + dt * Srho[:, :, k, l]) * RGAM[:, :, k, l]**2 * afact[k]
                    + (rhog0[:, :, k - 1, l] + dt * Srho[:, :, k - 1, l]) * RGAM[:, :, k - 1, l]**2 * bfact[k]
                ) * dt * GRAV
            ) * CVovRt2
        rhogw[:, :, kmin, l]   *= RGSGAM2H[:, :, kmin, l]
        rhogw[:, :, kmax+1, l] *= RGSGAM2H[:, :, kmax+1, l]
        Sall[:, :, kmin+1, l] -= Ml[:, :, kmin+1, l] * rhogw[:, :, kmin, l]
        Sall[:, :, kmax, l]   -= Mu[:, :, kmax, l]   * rhogw[:, :, kmax+1, l]
        k = kmin + 1
        beta = Mc[:, :, k, l].copy()
        rhogw[:, :, k, l] = Sall[:, :, k, l] / beta
        for k in range(kmin + 2, kmax + 1):
            gamma[:, :, k, l] = Mu[:, :, k - 1, l] / beta
            beta = Mc[:, :, k, l] - Ml[:, :, k, l] * gamma[:, :, k, l]
            rhogw[:, :, k, l] = (Sall[:, :, k, l] - Ml[:, :, k, l] * rhogw[:, :, k - 1, l]) / beta
        for k in range(kmax - 1, kmin, -1):
            rhogw[:, :, k, l]   -= gamma[:, :, k + 1, l] * rhogw[:, :, k + 1, l]
            rhogw[:, :, k + 1, l] *= GSGAM2H[:, :, k + 1, l]
        rhogw[:, :, kmin, l]   *= GSGAM2H[:, :, kmin, l]
        rhogw[:, :, kmin+1, l] *= GSGAM2H[:, :, kmin+1, l]
        rhogw[:, :, kmax+1, l] *= GSGAM2H[:, :, kmax+1, l]
    return rhogw


def solver_pl_ref(rhogw_in, rhogw0, preg0, rhog0, Srho, Sw, Spre,
                  Mc, Mu, Ml, RGAMH, RGSGAM2, RGAM, RGSGAM2H, GSGAM2H,
                  rdgzh, afact, bfact, dt, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    gall, kall, lall = rhogw_in.shape
    rhogw = rhogw_in.copy()
    Sall = np.zeros_like(rhogw)
    gamma = np.zeros_like(rhogw)
    alpha = np.float64(cfg.alpha)
    CVovRt2 = cfg.CVdry / cfg.Rdry / (dt * dt)
    GRAV = cfg.GRAV
    for l in range(lall):
        for k in range(kmin + 1, kmax + 1):
            for g in range(gall):
                Sall[g, k, l] = (
                    (rhogw0[g, k, l] * alpha + dt * Sw[g, k, l]) * RGAMH[g, k, l]**2
                    - (
                        (preg0[g, k, l] + dt * Spre[g, k, l]) * RGSGAM2[g, k, l]
                        - (preg0[g, k - 1, l] + dt * Spre[g, k - 1, l]) * RGSGAM2[g, k - 1, l]
                    ) * dt * rdgzh[k]
                    - (
                        (rhog0[g, k, l] + dt * Srho[g, k, l]) * RGAM[g, k, l]**2 * afact[k]
                        + (rhog0[g, k - 1, l] + dt * Srho[g, k - 1, l]) * RGAM[g, k - 1, l]**2 * bfact[k]
                    ) * dt * GRAV
                ) * CVovRt2
        for g in range(gall):
            rhogw[g, kmin, l]   *= RGSGAM2H[g, kmin, l]
            rhogw[g, kmax+1, l] *= RGSGAM2H[g, kmax+1, l]
            Sall[g, kmin+1, l] -= Ml[g, kmin+1, l] * rhogw[g, kmin, l]
            Sall[g, kmax, l]   -= Mu[g, kmax, l]   * rhogw[g, kmax+1, l]
        k = kmin + 1
        beta = Mc[:, k, l].copy()
        rhogw[:, k, l] = Sall[:, k, l] / beta
        for k in range(kmin + 2, kmax + 1):
            gamma[:, k, l] = Mu[:, k - 1, l] / beta
            beta = Mc[:, k, l] - Ml[:, k, l] * gamma[:, k, l]
            rhogw[:, k, l] = (Sall[:, k, l] - Ml[:, k, l] * rhogw[:, k - 1, l]) / beta
        for k in range(kmax - 1, kmin, -1):
            rhogw[:, k, l]   -= gamma[:, k + 1, l] * rhogw[:, k + 1, l]
            rhogw[:, k + 1, l] *= GSGAM2H[:, k + 1, l]
        rhogw[:, kmin, l]   *= GSGAM2H[:, kmin, l]
        rhogw[:, kmin+1, l] *= GSGAM2H[:, kmin+1, l]
        rhogw[:, kmax+1, l] *= GSGAM2H[:, kmax+1, l]
    return rhogw


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    iall, jall, kall, lall = 8, 8, 12, 3
    gall_pl, lall_pl = 6, 2
    kmin, kmax = 1, kall - 2
    dt = 0.7

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    # Mc dominant so the tridiagonal solve is well conditioned
    def Rc(*shape):
        return rng.uniform(5.0, 7.0, shape).astype(np.float64)

    data = dict(
        rhogw=R(iall, jall, kall, lall), rhogw0=R(iall, jall, kall, lall),
        preg0=R(iall, jall, kall, lall), rhog0=R(iall, jall, kall, lall),
        Srho=R(iall, jall, kall, lall), Sw=R(iall, jall, kall, lall), Spre=R(iall, jall, kall, lall),
        Mc=Rc(iall, jall, kall, lall), Mu=R(iall, jall, kall, lall), Ml=R(iall, jall, kall, lall),
        RGAMH=R(iall, jall, kall, lall), RGSGAM2=R(iall, jall, kall, lall), RGAM=R(iall, jall, kall, lall),
        RGSGAM2H=R(iall, jall, kall, lall), GSGAM2H=R(iall, jall, kall, lall),
        rdgzh=R(kall), afact=R(kall), bfact=R(kall),
        rhogw_pl=R(gall_pl, kall, lall_pl), rhogw0_pl=R(gall_pl, kall, lall_pl),
        preg0_pl=R(gall_pl, kall, lall_pl), rhog0_pl=R(gall_pl, kall, lall_pl),
        Srho_pl=R(gall_pl, kall, lall_pl), Sw_pl=R(gall_pl, kall, lall_pl), Spre_pl=R(gall_pl, kall, lall_pl),
        Mc_pl=Rc(gall_pl, kall, lall_pl), Mu_pl=R(gall_pl, kall, lall_pl), Ml_pl=R(gall_pl, kall, lall_pl),
        RGAMH_pl=R(gall_pl, kall, lall_pl), RGSGAM2_pl=R(gall_pl, kall, lall_pl), RGAM_pl=R(gall_pl, kall, lall_pl),
        RGSGAM2H_pl=R(gall_pl, kall, lall_pl), GSGAM2H_pl=R(gall_pl, kall, lall_pl),
        dt=dt,
    )
    cfg = ViSolverCfg(kmin=kmin, kmax=kmax, have_pl=True,
                      GRAV=9.80616, Rdry=287.04, CVdry=717.6, alpha=1.0)
    return data, cfg


def report(name, ref, got, rtol=0.0, atol=0.0):
    got = np.asarray(got)
    denom = np.maximum(np.abs(ref), 1e-300)
    max_abs = np.max(np.abs(got - ref))
    max_rel = np.max(np.abs(got - ref) / denom)
    exact = np.array_equal(got, ref)
    passed = exact or np.allclose(got, ref, rtol=rtol, atol=atol)
    flag = "EXACT" if exact else f"max|d|={max_abs:.3e} max|rel|={max_rel:.3e}"
    print(f"  {name:22s}: {'OK ' if passed else 'BAD'} {flag}")
    return passed


def main():
    d, cfg = make_inputs()

    ref_r = solver_reg_ref(d["rhogw"], d["rhogw0"], d["preg0"], d["rhog0"], d["Srho"], d["Sw"], d["Spre"],
                           d["Mc"], d["Mu"], d["Ml"], d["RGAMH"], d["RGSGAM2"], d["RGAM"],
                           d["RGSGAM2H"], d["GSGAM2H"], d["rdgzh"], d["afact"], d["bfact"], d["dt"], cfg)
    ref_p = solver_pl_ref(d["rhogw_pl"], d["rhogw0_pl"], d["preg0_pl"], d["rhog0_pl"], d["Srho_pl"], d["Sw_pl"], d["Spre_pl"],
                          d["Mc_pl"], d["Mu_pl"], d["Ml_pl"], d["RGAMH_pl"], d["RGSGAM2_pl"], d["RGAM_pl"],
                          d["RGSGAM2H_pl"], d["GSGAM2H_pl"], d["rdgzh"], d["afact"], d["bfact"], d["dt"], cfg)

    results = []

    def call_reg(c, xp):
        return compute_rhow_solver_reg(c["rhogw"], c["rhogw0"], c["preg0"], c["rhog0"], c["Srho"], c["Sw"], c["Spre"],
                                       c["Mc"], c["Mu"], c["Ml"], c["RGAMH"], c["RGSGAM2"], c["RGAM"],
                                       c["RGSGAM2H"], c["GSGAM2H"], c["rdgzh"], c["afact"], c["bfact"], c["dt"], cfg, xp)

    def call_pl(c, xp):
        return compute_rhow_solver_pl(c["rhogw_pl"], c["rhogw0_pl"], c["preg0_pl"], c["rhog0_pl"], c["Srho_pl"], c["Sw_pl"], c["Spre_pl"],
                                      c["Mc_pl"], c["Mu_pl"], c["Ml_pl"], c["RGAMH_pl"], c["RGSGAM2_pl"], c["RGAM_pl"],
                                      c["RGSGAM2H_pl"], c["GSGAM2H_pl"], c["rdgzh"], c["afact"], c["bfact"], c["dt"], cfg, xp)

    def run_backend(xp, label, rtol=0.0, atol=0.0, conv=lambda a: a):
        c = {k: (conv(v) if hasattr(v, "shape") else v) for k, v in d.items()}
        print(f"\n[{label}]  (rtol={rtol:g}, atol={atol:g})")
        results.append(report("solver reg", ref_r, call_reg(c, xp), rtol, atol))
        results.append(report("solver pl", ref_p, call_pl(c, xp), rtol, atol))

    import numpy as xnp
    run_backend(xnp, "numpy (eager)")

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    run_backend(jnp, "jax.numpy (eager)", conv=lambda a: jnp.asarray(a))

    f_r = jax.jit(compute_rhow_solver_reg, static_argnames=("cfg", "xp"))
    f_p = jax.jit(compute_rhow_solver_pl, static_argnames=("cfg", "xp"))
    cj = {k: (jnp.asarray(v) if hasattr(v, "shape") else v) for k, v in d.items()}
    print(f"\n[jax.jit]  (rtol=1e-11, atol=1e-11)")
    rr = f_r(cj["rhogw"], cj["rhogw0"], cj["preg0"], cj["rhog0"], cj["Srho"], cj["Sw"], cj["Spre"],
             cj["Mc"], cj["Mu"], cj["Ml"], cj["RGAMH"], cj["RGSGAM2"], cj["RGAM"],
             cj["RGSGAM2H"], cj["GSGAM2H"], cj["rdgzh"], cj["afact"], cj["bfact"], cj["dt"], cfg=cfg, xp=jnp)
    rp = f_p(cj["rhogw_pl"], cj["rhogw0_pl"], cj["preg0_pl"], cj["rhog0_pl"], cj["Srho_pl"], cj["Sw_pl"], cj["Spre_pl"],
             cj["Mc_pl"], cj["Mu_pl"], cj["Ml_pl"], cj["RGAMH_pl"], cj["RGSGAM2_pl"], cj["RGAM_pl"],
             cj["RGSGAM2H_pl"], cj["GSGAM2H_pl"], cj["rdgzh"], cj["afact"], cj["bfact"], cj["dt"], cfg=cfg, xp=jnp)
    jax.block_until_ready((rr, rp))
    results.append(report("solver reg", ref_r, rr, 1e-11, 1e-11))
    results.append(report("solver pl", ref_p, rp, 1e-11, 1e-11))

    print("\n========================================")
    print(f"all checks: {'PASS' if all(results) else 'FAIL'}")
    print("========================================")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
