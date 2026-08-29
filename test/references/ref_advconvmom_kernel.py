"""
Validation harness for kernels/advconvmom.py.

Checks that the pure backend-switchable kernels reproduce, bit-for-bit, the two
COMM-free sphere blocks of src_advection_convergence_momentum
(mod_src.py velocity-merge `else` branch + tendency `else` branch, plus the
grid-type-independent pole blocks) for:
    (1) xp = numpy        (eager)
    (2) xp = jax.numpy    (eager)
    (3) jax.jit(kernel)   (compiled, xp=jax.numpy)

Run:
    .../envs/jax_nomtl_mpi/bin/python test_advconvmom_kernel.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.advconvmom import (
    AdvMomCfg,
    compute_merged_velocity_reg, compute_merged_velocity_pl,
    compute_momentum_tendency_reg, compute_momentum_tendency_pl,
)

XDIR, YDIR, ZDIR = 0, 1, 2


# ---------------------------------------------------------------------------
# References: exact transcription of mod_src.py (numpy, in-place loops).
# Buffers start as zeros; rows the original leaves stale are never read
# downstream -> zeros is the downstream-equivalent reference.
# ---------------------------------------------------------------------------
def merge_reg_ref(vx, vy, vz, w, cfact, dfact, GRD_x, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1, kminp1, kmaxp1, kmaxp2 = kmin - 1, kmin + 1, kmax + 1, kmax + 2
    rscale = cfg.rscale
    vvx = np.zeros_like(vx); vvy = np.zeros_like(vy); vvz = np.zeros_like(vz)
    cf = cfact[kmin:kmaxp1][None, None, :, None]
    df = dfact[kmin:kmaxp1][None, None, :, None]
    wc = cf * w[:, :, kminp1:kmaxp2, :] + df * w[:, :, kmin:kmaxp1, :]
    gx = GRD_x[:, :, 0, :, XDIR][:, :, None, :].copy()
    gy = GRD_x[:, :, 0, :, YDIR][:, :, None, :].copy()
    gz = GRD_x[:, :, 0, :, ZDIR][:, :, None, :].copy()
    vvx[:, :, kmin:kmaxp1, :] = vx[:, :, kmin:kmaxp1, :] + wc * gx / rscale
    vvy[:, :, kmin:kmaxp1, :] = vy[:, :, kmin:kmaxp1, :] + wc * gy / rscale
    vvz[:, :, kmin:kmaxp1, :] = vz[:, :, kmin:kmaxp1, :] + wc * gz / rscale
    for a in (vvx, vvy, vvz):
        a[:, :, kminm1, :] = 0.0
        a[:, :, kmaxp1, :] = 0.0
    return vvx, vvy, vvz


def merge_pl_ref(vx_pl, vy_pl, vz_pl, w_pl, cfact, dfact, GRD_x_pl, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1, kminp1, kmaxp1, kmaxp2 = kmin - 1, kmin + 1, kmax + 1, kmax + 2
    rscale = cfg.rscale
    vvx = np.zeros_like(vx_pl); vvy = np.zeros_like(vy_pl); vvz = np.zeros_like(vz_pl)
    cf = cfact[kmin:kmaxp1][None, :, None]
    df = dfact[kmin:kmaxp1][None, :, None]
    wc = cf * w_pl[:, kminp1:kmaxp2, :] + df * w_pl[:, kmin:kmaxp1, :]
    gx = GRD_x_pl[:, 0, :, XDIR][:, None, :].copy()
    gy = GRD_x_pl[:, 0, :, YDIR][:, None, :].copy()
    gz = GRD_x_pl[:, 0, :, ZDIR][:, None, :].copy()
    vvx[:, kmin:kmaxp1, :] = vx_pl[:, kmin:kmaxp1, :] + (wc * gx / rscale)
    vvy[:, kmin:kmaxp1, :] = vy_pl[:, kmin:kmaxp1, :] + (wc * gy / rscale)
    vvz[:, kmin:kmaxp1, :] = vz_pl[:, kmin:kmaxp1, :] + (wc * gz / rscale)
    for a in (vvx, vvy, vvz):
        a[:, kminm1, :] = 0.0
        a[:, kmaxp1, :] = 0.0
    return vvx, vvy, vvz


def tend_reg_ref(dvvx, dvvy, dvvz, rhog, vvx, vvy, GRD_x, C2Wfact, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1, kminp1, kmaxp1 = kmin - 1, kmin + 1, kmax + 1
    ohm, alpha, rscale = cfg.ohm, cfg.alpha, cfg.rscale
    dvvx = dvvx.copy(); dvvy = dvvy.copy()
    grhogvx = np.zeros_like(dvvx); grhogvy = np.zeros_like(dvvx)
    grhogvz = np.zeros_like(dvvx); grhogw = np.zeros_like(dvvx)
    dvvx[:, :, kmin:kmaxp1, :] -= -2.0 * rhog[:, :, kmin:kmaxp1, :] * (ohm * vvy[:, :, kmin:kmaxp1, :])
    dvvy[:, :, kmin:kmaxp1, :] -= 2.0 * rhog[:, :, kmin:kmaxp1, :] * (ohm * vvx[:, :, kmin:kmaxp1, :])
    gx = GRD_x[:, :, 0, :, XDIR][:, :, None, :].copy() / rscale
    gy = GRD_x[:, :, 0, :, YDIR][:, :, None, :].copy() / rscale
    gz = GRD_x[:, :, 0, :, ZDIR][:, :, None, :].copy() / rscale
    prd = (dvvx[:, :, kmin:kmaxp1, :] * gx + dvvy[:, :, kmin:kmaxp1, :] * gy +
           dvvz[:, :, kmin:kmaxp1, :] * gz)
    grhogvx[:, :, kmin:kmaxp1, :] = dvvx[:, :, kmin:kmaxp1, :] - prd * gx
    grhogvy[:, :, kmin:kmaxp1, :] = dvvy[:, :, kmin:kmaxp1, :] - prd * gy
    grhogvz[:, :, kmin:kmaxp1, :] = dvvz[:, :, kmin:kmaxp1, :] - prd * gz
    grhogwc = np.zeros_like(dvvx)
    grhogwc[:, :, kmin:kmaxp1, :] = prd * alpha
    f1 = C2Wfact[:, :, kminp1:kmaxp1, :, 0]
    f2 = C2Wfact[:, :, kminp1:kmaxp1, :, 1]
    grhogw[:, :, kminp1:kmaxp1, :] = f1 * grhogwc[:, :, kminp1:kmaxp1, :] + f2 * grhogwc[:, :, kmin:kmax, :]
    for a in (grhogvx, grhogvy, grhogvz):
        a[:, :, kminm1, :] = 0.0
        a[:, :, kmaxp1, :] = 0.0
    grhogw[:, :, kminm1, :] = 0.0
    grhogw[:, :, kmin, :] = 0.0
    grhogw[:, :, kmaxp1, :] = 0.0
    return grhogvx, grhogvy, grhogvz, grhogw


def tend_pl_ref(dvvx, dvvy, dvvz, rhog, vvx, vvy, GRD_x, C2Wfact, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1, kminp1, kmaxp1 = kmin - 1, kmin + 1, kmax + 1
    ohm, alpha, rscale = cfg.ohm, cfg.alpha, cfg.rscale
    dvvx = dvvx.copy(); dvvy = dvvy.copy()
    grhogvx = np.zeros_like(dvvx); grhogvy = np.zeros_like(dvvx)
    grhogvz = np.zeros_like(dvvx); grhogw = np.zeros_like(dvvx)
    dvvx[:, kmin:kmaxp1, :] -= -2.0 * rhog[:, kmin:kmaxp1, :] * (ohm * vvy[:, kmin:kmaxp1, :])
    dvvy[:, kmin:kmaxp1, :] -= 2.0 * rhog[:, kmin:kmaxp1, :] * (ohm * vvx[:, kmin:kmaxp1, :])
    gx = GRD_x[:, 0, :, XDIR][:, None, :].copy() / rscale
    gy = GRD_x[:, 0, :, YDIR][:, None, :].copy() / rscale
    gz = GRD_x[:, 0, :, ZDIR][:, None, :].copy() / rscale
    prd = (dvvx[:, kmin:kmaxp1, :] * gx + dvvy[:, kmin:kmaxp1, :] * gy +
           dvvz[:, kmin:kmaxp1, :] * gz)
    grhogvx[:, kmin:kmaxp1, :] = dvvx[:, kmin:kmaxp1, :] - prd * gx
    grhogvy[:, kmin:kmaxp1, :] = dvvy[:, kmin:kmaxp1, :] - prd * gy
    grhogvz[:, kmin:kmaxp1, :] = dvvz[:, kmin:kmaxp1, :] - prd * gz
    grhogwc = np.zeros_like(dvvx)
    grhogwc[:, kmin:kmaxp1, :] = prd * alpha
    f1 = C2Wfact[:, kminp1:kmaxp1, :, 0]
    f2 = C2Wfact[:, kminp1:kmaxp1, :, 1]
    grhogw[:, kminp1:kmaxp1, :] = f1 * grhogwc[:, kminp1:kmaxp1, :] + f2 * grhogwc[:, kmin:kmax, :]
    for a in (grhogvx, grhogvy, grhogvz):
        a[:, kminm1, :] = 0.0
        a[:, kmaxp1, :] = 0.0
    grhogw[:, kminm1, :] = 0.0
    grhogw[:, kmin, :] = 0.0
    grhogw[:, kmaxp1, :] = 0.0
    return grhogvx, grhogvy, grhogvz, grhogw


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    iall, jall, kall, lall = 8, 8, 10, 3
    gall_pl, lall_pl = 6, 2
    dt = np.float64
    kmin, kmax = 1, kall - 2

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(dt)

    data = dict(
        vx=R(iall, jall, kall, lall), vy=R(iall, jall, kall, lall), vz=R(iall, jall, kall, lall),
        w=R(iall, jall, kall, lall),
        dvvx=R(iall, jall, kall, lall), dvvy=R(iall, jall, kall, lall), dvvz=R(iall, jall, kall, lall),
        rhog=R(iall, jall, kall, lall), vvx=R(iall, jall, kall, lall), vvy=R(iall, jall, kall, lall),
        GRD_x=R(iall, jall, 1, lall, 3), C2Wfact=R(iall, jall, kall, lall, 2),
        cfact=R(kall), dfact=R(kall),
        vx_pl=R(gall_pl, kall, lall_pl), vy_pl=R(gall_pl, kall, lall_pl), vz_pl=R(gall_pl, kall, lall_pl),
        w_pl=R(gall_pl, kall, lall_pl),
        dvvx_pl=R(gall_pl, kall, lall_pl), dvvy_pl=R(gall_pl, kall, lall_pl), dvvz_pl=R(gall_pl, kall, lall_pl),
        rhog_pl=R(gall_pl, kall, lall_pl), vvx_pl=R(gall_pl, kall, lall_pl), vvy_pl=R(gall_pl, kall, lall_pl),
        GRD_x_pl=R(gall_pl, 1, lall_pl, 3), C2Wfact_pl=R(gall_pl, kall, lall_pl, 2),
    )
    cfg = AdvMomCfg(kmin=kmin, kmax=kmax, have_pl=True, XDIR=XDIR, YDIR=YDIR, ZDIR=ZDIR,
                    rscale=6.371e6, ohm=7.292e-5, alpha=1.0)
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
        print(f"  {nm:11s}: {'OK ' if passed else 'BAD'} {flag}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    d, cfg = make_inputs()

    ref_mr = merge_reg_ref(d["vx"], d["vy"], d["vz"], d["w"], d["cfact"], d["dfact"], d["GRD_x"], cfg)
    ref_mp = merge_pl_ref(d["vx_pl"], d["vy_pl"], d["vz_pl"], d["w_pl"], d["cfact"], d["dfact"], d["GRD_x_pl"], cfg)
    ref_tr = tend_reg_ref(d["dvvx"], d["dvvy"], d["dvvz"], d["rhog"], d["vvx"], d["vvy"], d["GRD_x"], d["C2Wfact"], cfg)
    ref_tp = tend_pl_ref(d["dvvx_pl"], d["dvvy_pl"], d["dvvz_pl"], d["rhog_pl"], d["vvx_pl"], d["vvy_pl"], d["GRD_x_pl"], d["C2Wfact_pl"], cfg)

    results = []

    def run_backend(xp, label, rtol=0.0, atol=0.0, conv=lambda a: a):
        c = {k: conv(v) for k, v in d.items()}
        mr = compute_merged_velocity_reg(c["vx"], c["vy"], c["vz"], c["w"], c["cfact"], c["dfact"], c["GRD_x"], cfg, xp)
        mp = compute_merged_velocity_pl(c["vx_pl"], c["vy_pl"], c["vz_pl"], c["w_pl"], c["cfact"], c["dfact"], c["GRD_x_pl"], cfg, xp)
        tr = compute_momentum_tendency_reg(c["dvvx"], c["dvvy"], c["dvvz"], c["rhog"], c["vvx"], c["vvy"], c["GRD_x"], c["C2Wfact"], cfg, xp)
        tp = compute_momentum_tendency_pl(c["dvvx_pl"], c["dvvy_pl"], c["dvvz_pl"], c["rhog_pl"], c["vvx_pl"], c["vvy_pl"], c["GRD_x_pl"], c["C2Wfact_pl"], cfg, xp)
        results.append(report(f"{label}: merge reg", ref_mr, mr, ["vvx", "vvy", "vvz"], rtol, atol))
        results.append(report(f"{label}: merge pl", ref_mp, mp, ["vvx_pl", "vvy_pl", "vvz_pl"], rtol, atol))
        results.append(report(f"{label}: tend reg", ref_tr, tr, ["grhogvx", "grhogvy", "grhogvz", "grhogw"], rtol, atol))
        results.append(report(f"{label}: tend pl", ref_tp, tp, ["grhogvx_pl", "grhogvy_pl", "grhogvz_pl", "grhogw_pl"], rtol, atol))

    import numpy as xnp
    run_backend(xnp, "numpy (eager)")

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    run_backend(jnp, "jax.numpy (eager)", conv=lambda a: jnp.asarray(a))

    # jit each kernel
    f_mr = jax.jit(compute_merged_velocity_reg, static_argnames=("cfg", "xp"))
    f_mp = jax.jit(compute_merged_velocity_pl, static_argnames=("cfg", "xp"))
    f_tr = jax.jit(compute_momentum_tendency_reg, static_argnames=("cfg", "xp"))
    f_tp = jax.jit(compute_momentum_tendency_pl, static_argnames=("cfg", "xp"))
    j = lambda a: jnp.asarray(a)
    mr = f_mr(j(d["vx"]), j(d["vy"]), j(d["vz"]), j(d["w"]), j(d["cfact"]), j(d["dfact"]), j(d["GRD_x"]), cfg=cfg, xp=jnp)
    mp = f_mp(j(d["vx_pl"]), j(d["vy_pl"]), j(d["vz_pl"]), j(d["w_pl"]), j(d["cfact"]), j(d["dfact"]), j(d["GRD_x_pl"]), cfg=cfg, xp=jnp)
    tr = f_tr(j(d["dvvx"]), j(d["dvvy"]), j(d["dvvz"]), j(d["rhog"]), j(d["vvx"]), j(d["vvy"]), j(d["GRD_x"]), j(d["C2Wfact"]), cfg=cfg, xp=jnp)
    tp = f_tp(j(d["dvvx_pl"]), j(d["dvvy_pl"]), j(d["dvvz_pl"]), j(d["rhog_pl"]), j(d["vvx_pl"]), j(d["vvy_pl"]), j(d["GRD_x_pl"]), j(d["C2Wfact_pl"]), cfg=cfg, xp=jnp)
    jax.block_until_ready((mr, mp, tr, tp))
    results.append(report("jax.jit: merge reg", ref_mr, mr, ["vvx", "vvy", "vvz"], 1e-11, 1e-11))
    results.append(report("jax.jit: merge pl", ref_mp, mp, ["vvx_pl", "vvy_pl", "vvz_pl"], 1e-11, 1e-11))
    results.append(report("jax.jit: tend reg", ref_tr, tr, ["grhogvx", "grhogvy", "grhogvz", "grhogw"], 1e-11, 1e-11))
    results.append(report("jax.jit: tend pl", ref_tp, tp, ["grhogvx_pl", "grhogvy_pl", "grhogvz_pl", "grhogw_pl"], 1e-11, 1e-11))

    print("\n========================================")
    print(f"all checks: {'PASS' if all(results) else 'FAIL'}")
    print("========================================")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
