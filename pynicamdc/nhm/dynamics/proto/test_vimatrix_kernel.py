"""
Validation harness for kernels/vimatrix.py.

Checks that the pure backend-switchable kernels reproduce, bit-for-bit, the
COMM-free tri-diagonal matrix assembly of vi_rhow_update_matrix (mod_vi.py),
returning the interior slab k = kmin+1 .. kmax for:
    (1) xp = numpy        (eager)
    (2) xp = jax.numpy    (eager)
    (3) jax.jit(kernel)   (compiled, xp=jax.numpy)

Run:
    .../envs/jax_nomtl_mpi/bin/python test_vimatrix_kernel.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.vimatrix import (
    ViMatrixCfg, compute_rhow_matrix_reg, compute_rhow_matrix_pl,
)


# ---------------------------------------------------------------------------
# References: exact transcription of mod_vi.py vectorized assembly, returning
# only the interior slab (k = kmin+1 .. kmax).
# ---------------------------------------------------------------------------
def matrix_reg_ref(eth, g_tilde, RGSQRTH, RGSGAM2, GAM2H, RGAMH,
                   rdgzh, rdgz, dfact, cfact, dt, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    ks  = slice(kmin + 1, kmax + 1)
    ksp = slice(kmin + 2, kmax + 2)
    ksm = slice(kmin,     kmax)
    GCVovR   = cfg.GRAV * cfg.CVdry / cfg.Rdry
    ACVovRt2 = np.float64(cfg.alpha) * cfg.CVdry / cfg.Rdry / (dt * dt)

    rgdzh   = rdgzh[ks][None, None, :, None]
    rgdz    = rdgz[ks][None, None, :, None]
    rgdzm1  = rdgz[ksm][None, None, :, None]
    dfact_  = dfact[ks][None, None, :, None]
    cfact_  = cfact[ks][None, None, :, None]
    cfactm1 = cfact[ksm][None, None, :, None]
    dfactm1 = dfact[ksm][None, None, :, None]

    RGSQRTH_  = RGSQRTH[:, :, ks, :]
    RGSGAM2_  = RGSGAM2[:, :, ks, :]
    RGSGAM2m1 = RGSGAM2[:, :, ksm, :]
    GAM2H_    = GAM2H[:, :, ks, :]
    eth_      = eth[:, :, ks, :]
    gtilde_   = g_tilde[:, :, ks, :]
    RGAMH_    = RGAMH[:, :, ks, :]

    Mc = (ACVovRt2 / RGSQRTH_ +
          rgdzh * ((RGSGAM2_ * rgdz + RGSGAM2m1 * rgdzm1) * GAM2H_ * eth_ -
                   (dfact_ - cfactm1) * (gtilde_ + GCVovR)))
    Mu = -rgdzh * (RGSGAM2_ * rgdz * GAM2H[:, :, ksp, :] * eth[:, :, ksp, :] +
                   cfact_ * (g_tilde[:, :, ksp, :] +
                             GAM2H[:, :, ksp, :] * RGAMH_ ** 2 * GCVovR))
    Ml = -rgdzh * (RGSGAM2_ * rgdz * GAM2H[:, :, ksm, :] * eth[:, :, ksm, :] -
                   dfactm1 * (g_tilde[:, :, ksm, :] +
                              GAM2H[:, :, ksm, :] * RGAMH_ ** 2 * GCVovR))
    return Mc, Mu, Ml


def matrix_pl_ref(eth_pl, g_tilde_pl, RGSQRTH_pl, RGSGAM2_pl, GAM2H_pl, RGAMH_pl,
                  rdgzh, rdgz, dfact, cfact, dt, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    ks  = slice(kmin + 1, kmax + 1)
    ksp = slice(kmin + 2, kmax + 2)
    ksm = slice(kmin,     kmax)
    GCVovR   = cfg.GRAV * cfg.CVdry / cfg.Rdry
    ACVovRt2 = np.float64(cfg.alpha) * cfg.CVdry / cfg.Rdry / (dt * dt)

    rgdzh   = rdgzh[ks][None, :, None]
    rgdz    = rdgz[ks][None, :, None]
    rgdzm1  = rdgz[ksm][None, :, None]
    dfact_  = dfact[ks][None, :, None]
    cfact_  = cfact[ks][None, :, None]
    cfactm1 = cfact[ksm][None, :, None]
    dfactm1 = dfact[ksm][None, :, None]

    RGSQRTH_  = RGSQRTH_pl[:, ks, :]
    RGSGAM2_  = RGSGAM2_pl[:, ks, :]
    RGSGAM2m1 = RGSGAM2_pl[:, ksm, :]
    GAM2H_    = GAM2H_pl[:, ks, :]
    eth_      = eth_pl[:, ks, :]
    gtilde_   = g_tilde_pl[:, ks, :]
    RGAMH_    = RGAMH_pl[:, ks, :]

    Mc = (ACVovRt2 / RGSQRTH_ +
          rgdzh * ((RGSGAM2_ * rgdz + RGSGAM2m1 * rgdzm1) * GAM2H_ * eth_ -
                   (dfact_ - cfactm1) * (gtilde_ + GCVovR)))
    Mu = -rgdzh * (RGSGAM2_ * rgdz * GAM2H_pl[:, ksp, :] * eth_pl[:, ksp, :] +
                   cfact_ * (g_tilde_pl[:, ksp, :] +
                             GAM2H_pl[:, ksp, :] * RGAMH_ ** 2 * GCVovR))
    Ml = -rgdzh * (RGSGAM2_ * rgdz * GAM2H_pl[:, ksm, :] * eth_pl[:, ksm, :] -
                   dfactm1 * (g_tilde_pl[:, ksm, :] +
                              GAM2H_pl[:, ksm, :] * RGAMH_ ** 2 * GCVovR))
    return Mc, Mu, Ml


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    iall, jall, kall, lall = 8, 8, 10, 3
    gall_pl, lall_pl = 6, 2
    dt = 0.7
    kmin, kmax = 1, kall - 2

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    data = dict(
        eth=R(iall, jall, kall, lall), g_tilde=R(iall, jall, kall, lall),
        RGSQRTH=R(iall, jall, kall, lall), RGSGAM2=R(iall, jall, kall, lall),
        GAM2H=R(iall, jall, kall, lall), RGAMH=R(iall, jall, kall, lall),
        rdgzh=R(kall), rdgz=R(kall), dfact=R(kall), cfact=R(kall),
        eth_pl=R(gall_pl, kall, lall_pl), g_tilde_pl=R(gall_pl, kall, lall_pl),
        RGSQRTH_pl=R(gall_pl, kall, lall_pl), RGSGAM2_pl=R(gall_pl, kall, lall_pl),
        GAM2H_pl=R(gall_pl, kall, lall_pl), RGAMH_pl=R(gall_pl, kall, lall_pl),
        dt=dt,
    )
    cfg = ViMatrixCfg(kmin=kmin, kmax=kmax, have_pl=True,
                      GRAV=9.80616, Rdry=287.04, CVdry=717.6, alpha=1.0)
    return data, cfg


def report(name, refs, gots, names, rtol=0.0, atol=0.0):
    print(f"\n[{name}]  (rtol={rtol:g}, atol={atol:g})")
    ok = True
    for nm, r, g in zip(names, refs, gots):
        g = np.asarray(g)
        denom = np.maximum(np.abs(r), 1e-300)
        max_abs = np.max(np.abs(g - r))
        max_rel = np.max(np.abs(g - r) / denom)
        exact = np.array_equal(g, r)
        passed = exact or np.allclose(g, r, rtol=rtol, atol=atol)
        ok = ok and passed
        flag = "EXACT" if exact else f"max|d|={max_abs:.3e} max|rel|={max_rel:.3e}"
        print(f"  {nm:8s}: {'OK ' if passed else 'BAD'} {flag}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    d, cfg = make_inputs()

    ref_r = matrix_reg_ref(d["eth"], d["g_tilde"], d["RGSQRTH"], d["RGSGAM2"],
                           d["GAM2H"], d["RGAMH"], d["rdgzh"], d["rdgz"],
                           d["dfact"], d["cfact"], d["dt"], cfg)
    ref_p = matrix_pl_ref(d["eth_pl"], d["g_tilde_pl"], d["RGSQRTH_pl"], d["RGSGAM2_pl"],
                          d["GAM2H_pl"], d["RGAMH_pl"], d["rdgzh"], d["rdgz"],
                          d["dfact"], d["cfact"], d["dt"], cfg)

    results = []

    def run_backend(xp, label, rtol=0.0, atol=0.0, conv=lambda a: a):
        c = {k: (conv(v) if hasattr(v, "shape") else v) for k, v in d.items()}
        r = compute_rhow_matrix_reg(c["eth"], c["g_tilde"], c["RGSQRTH"], c["RGSGAM2"],
                                    c["GAM2H"], c["RGAMH"], c["rdgzh"], c["rdgz"],
                                    c["dfact"], c["cfact"], c["dt"], cfg, xp)
        p = compute_rhow_matrix_pl(c["eth_pl"], c["g_tilde_pl"], c["RGSQRTH_pl"], c["RGSGAM2_pl"],
                                   c["GAM2H_pl"], c["RGAMH_pl"], c["rdgzh"], c["rdgz"],
                                   c["dfact"], c["cfact"], c["dt"], cfg, xp)
        results.append(report(f"{label}: reg", ref_r, r, ["Mc", "Mu", "Ml"], rtol, atol))
        results.append(report(f"{label}: pl", ref_p, p, ["Mc_pl", "Mu_pl", "Ml_pl"], rtol, atol))

    import numpy as xnp
    run_backend(xnp, "numpy (eager)")

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    run_backend(jnp, "jax.numpy (eager)", conv=lambda a: jnp.asarray(a))

    f_r = jax.jit(compute_rhow_matrix_reg, static_argnames=("cfg", "xp"))
    f_p = jax.jit(compute_rhow_matrix_pl, static_argnames=("cfg", "xp"))
    j = lambda a: jnp.asarray(a)
    r = f_r(j(d["eth"]), j(d["g_tilde"]), j(d["RGSQRTH"]), j(d["RGSGAM2"]),
            j(d["GAM2H"]), j(d["RGAMH"]), j(d["rdgzh"]), j(d["rdgz"]),
            j(d["dfact"]), j(d["cfact"]), d["dt"], cfg=cfg, xp=jnp)
    p = f_p(j(d["eth_pl"]), j(d["g_tilde_pl"]), j(d["RGSQRTH_pl"]), j(d["RGSGAM2_pl"]),
            j(d["GAM2H_pl"]), j(d["RGAMH_pl"]), j(d["rdgzh"]), j(d["rdgz"]),
            j(d["dfact"]), j(d["cfact"]), d["dt"], cfg=cfg, xp=jnp)
    jax.block_until_ready((r, p))
    results.append(report("jax.jit: reg", ref_r, r, ["Mc", "Mu", "Ml"], 1e-11, 1e-11))
    results.append(report("jax.jit: pl", ref_p, p, ["Mc_pl", "Mu_pl", "Ml_pl"], 1e-11, 1e-11))

    print("\n========================================")
    print(f"all checks: {'PASS' if all(results) else 'FAIL'}")
    print("========================================")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
