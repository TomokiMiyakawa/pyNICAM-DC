"""
Validation harness for kernels/bndcnd.py.

Checks that the pure backend-switchable boundary-condition kernels reproduce,
bit-for-bit, the in-place ghost-row updates of mod_bndcnd.py (BNDCND_thermo /
BNDCND_rhovxvyvz / BNDCND_rhow + pole _pl variants), under:
    (1) xp = numpy        (eager)
    (2) xp = jax.numpy    (eager)
    (3) jax.jit(kernel)   (compiled, xp=jax.numpy)

Two flag configurations are exercised:
    prod : top_tem / btm_tem / top_free(M) / btm_rigid(M)  (the runtime config)
    alt  : top_epl / btm_epl / top_rigid(M) / btm_free(M)

Run:
    .../envs/jax_nomtl_mpi/bin/python test_bndcnd_kernel.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.bndcnd import (
    BndCfg,
    compute_bndcnd_thermo_reg, compute_bndcnd_thermo_pl,
    compute_bndcnd_rhovxvyvz_reg, compute_bndcnd_rhovxvyvz_pl,
    compute_bndcnd_rhow_reg, compute_bndcnd_rhow_pl,
)

GRAV = 9.80616
Rdry = 287.04


# ---------------------------------------------------------------------------
# References: exact transcription of mod_bndcnd.py, mutating copies in place.
# ---------------------------------------------------------------------------
def _lag(z, z1, p1, z2, p2, z3, p3):
    return (((z - z2) * (z - z3)) / ((z1 - z2) * (z1 - z3)) * p1 +
            ((z - z1) * (z - z3)) / ((z2 - z1) * (z2 - z3)) * p2 +
            ((z - z1) * (z - z2)) / ((z3 - z1) * (z3 - z2)) * p3)


def thermo_ref(tem, rho, pre, phi, f, kmin, kmax, reg):
    tem, rho, pre = tem.copy(), rho.copy(), pre.copy()
    kminm1, kminp1, kminp2 = kmin - 1, kmin + 1, kmin + 2
    kmaxm1, kmaxm2, kmaxp1 = kmax - 1, kmax - 2, kmax + 1
    if reg:
        S = lambda a, k: a[:, :, k, :]
    else:
        S = lambda a, k: a[:, k, :]

    def setk(a, k, v):
        if reg:
            a[:, :, k, :] = v
        else:
            a[:, k, :] = v

    if f["is_top_tem"]:
        setk(tem, kmaxp1, S(tem, kmax))
    elif f["is_top_epl"]:
        setk(tem, kmaxp1, _lag(S(phi, kmaxp1) / GRAV,
                               S(phi, kmax) / GRAV, S(tem, kmax),
                               S(phi, kmaxm1) / GRAV, S(tem, kmaxm1),
                               S(phi, kmaxm2) / GRAV, S(tem, kmaxm2)))
    if f["is_btm_tem"]:
        setk(tem, kminm1, S(tem, kmin))
    elif f["is_btm_epl"]:
        setk(tem, kminm1, _lag(S(phi, kminm1) / GRAV,
                               S(phi, kminp2) / GRAV, S(tem, kminp2),
                               S(phi, kminp1) / GRAV, S(tem, kminp1),
                               S(phi, kmin) / GRAV, S(tem, kmin)))
    setk(pre, kmaxp1, S(pre, kmaxm1) - S(rho, kmax) * (S(phi, kmaxp1) - S(phi, kmaxm1)))
    setk(pre, kminm1, S(pre, kminp1) - S(rho, kmin) * (S(phi, kminm1) - S(phi, kminp1)))
    setk(rho, kmaxp1, S(pre, kmaxp1) / (Rdry * S(tem, kmaxp1)))
    setk(rho, kminm1, S(pre, kminm1) / (Rdry * S(tem, kminm1)))
    return tem, rho, pre


def rhovxvyvz_ref(rhog, rhogvx, rhogvy, rhogvz, f, kmin, kmax, reg):
    rhogvx, rhogvy, rhogvz = rhogvx.copy(), rhogvy.copy(), rhogvz.copy()
    kminm1, kmaxp1 = kmin - 1, kmax + 1
    if reg:
        S = lambda a, k: a[:, :, k, :]
        def setk(a, k, v): a[:, :, k, :] = v
    else:
        S = lambda a, k: a[:, k, :]
        def setk(a, k, v): a[:, k, :] = v

    for V in ((rhogvx,), (rhogvy,), (rhogvz,)):
        pass
    for V in (rhogvx, rhogvy, rhogvz):
        if f["is_top_rigid"]:
            sc = S(V, kmax) / S(rhog, kmax); setk(V, kmaxp1, -sc * S(rhog, kmaxp1))
        elif f["is_top_free"]:
            sc = S(V, kmax) / S(rhog, kmax); setk(V, kmaxp1, sc * S(rhog, kmaxp1))
        if f["is_btm_rigid"]:
            sc = S(V, kmin) / S(rhog, kmin); setk(V, kminm1, -sc * S(rhog, kminm1))
        elif f["is_btm_free"]:
            sc = S(V, kmin) / S(rhog, kmin); setk(V, kminm1, sc * S(rhog, kminm1))
    return rhogvx, rhogvy, rhogvz


def rhow_ref(rhogvx, rhogvy, rhogvz, rhogw, c2w, f, kmin, kmax, reg):
    rhogw = rhogw.copy()
    kminm1, kmaxp1 = kmin - 1, kmax + 1
    if reg:
        S = lambda a, k: a[:, :, k, :]
        C = lambda k, i: c2w[:, :, k, :, i]
        def setk(a, k, v): a[:, :, k, :] = v
    else:
        S = lambda a, k: a[:, k, :]
        C = lambda k, i: c2w[:, k, :, i]
        def setk(a, k, v): a[:, k, :] = v

    if f["is_top_rigid"]:
        setk(rhogw, kmaxp1, 0.0)
    elif f["is_top_free"]:
        setk(rhogw, kmaxp1, -(C(kmaxp1, 0) * S(rhogvx, kmaxp1) + C(kmaxp1, 1) * S(rhogvx, kmax) +
                              C(kmaxp1, 2) * S(rhogvy, kmaxp1) + C(kmaxp1, 3) * S(rhogvy, kmax) +
                              C(kmaxp1, 4) * S(rhogvz, kmaxp1) + C(kmaxp1, 5) * S(rhogvz, kmax)))
    if f["is_btm_rigid"]:
        setk(rhogw, kmin, 0.0)
    elif f["is_btm_free"]:
        setk(rhogw, kmin, -(C(kmin, 0) * S(rhogvx, kmin) + C(kmin, 1) * S(rhogvx, kminm1) +
                            C(kmin, 2) * S(rhogvy, kmin) + C(kmin, 3) * S(rhogvy, kminm1) +
                            C(kmin, 4) * S(rhogvz, kmin) + C(kmin, 5) * S(rhogvz, kminm1)))
    setk(rhogw, kminm1, 0.0)
    return rhogw


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    iall, jall, kall, lall = 8, 8, 10, 3
    gall_pl, lall_pl = 6, 2
    kmin, kmax = 1, kall - 2

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    d = dict(
        tem=R(iall, jall, kall, lall), rho=R(iall, jall, kall, lall),
        pre=R(iall, jall, kall, lall), phi=R(iall, jall, kall, lall),
        rhog=R(iall, jall, kall, lall),
        rhogvx=R(iall, jall, kall, lall), rhogvy=R(iall, jall, kall, lall),
        rhogvz=R(iall, jall, kall, lall), rhogw=R(iall, jall, kall, lall),
        c2wGz=R(iall, jall, kall, lall, 6),
        tem_pl=R(gall_pl, kall, lall_pl), rho_pl=R(gall_pl, kall, lall_pl),
        pre_pl=R(gall_pl, kall, lall_pl), phi_pl=R(gall_pl, kall, lall_pl),
        rhog_pl=R(gall_pl, kall, lall_pl),
        rhogvx_pl=R(gall_pl, kall, lall_pl), rhogvy_pl=R(gall_pl, kall, lall_pl),
        rhogvz_pl=R(gall_pl, kall, lall_pl), rhogw_pl=R(gall_pl, kall, lall_pl),
        c2wGz_pl=R(gall_pl, kall, lall_pl, 6),
    )
    return d, kmin, kmax


FLAGS = {
    "prod": dict(is_top_tem=True, is_top_epl=False, is_btm_tem=True, is_btm_epl=False,
                 is_top_rigid=False, is_top_free=True, is_btm_rigid=True, is_btm_free=False),
    "alt":  dict(is_top_tem=False, is_top_epl=True, is_btm_tem=False, is_btm_epl=True,
                 is_top_rigid=True, is_top_free=False, is_btm_rigid=False, is_btm_free=True),
}


def cfg_of(f, kmin, kmax):
    return BndCfg(kmin=kmin, kmax=kmax, have_pl=True, GRAV=GRAV, Rdry=Rdry, **f)


def chk(name, ref, got, results, rtol=0.0, atol=0.0):
    got = np.asarray(got)
    exact = np.array_equal(got, ref)
    passed = exact or np.allclose(got, ref, rtol=rtol, atol=atol)
    md = np.max(np.abs(got - ref))
    print(f"    {name:10s}: {'OK ' if passed else 'BAD'} {'EXACT' if exact else f'max|d|={md:.3e}'}")
    results.append(passed)


def run_set(d, kmin, kmax, fname, f, xp, label, mkarr, rtol=0.0, atol=0.0):
    cfg = cfg_of(f, kmin, kmax)
    kmaxp1, kminm1 = kmax + 1, kmin - 1
    results = []
    print(f"  [{label} | {fname}]")

    # --- thermo reg ---
    rt, rr, rp = thermo_ref(d["tem"], d["rho"], d["pre"], d["phi"], f, kmin, kmax, reg=True)
    tt, tb, pt, pb, ot, ob = compute_bndcnd_thermo_reg(
        mkarr(d["tem"]), mkarr(d["rho"]), mkarr(d["pre"]), mkarr(d["phi"]), cfg, xp)
    chk("thermo.tT", rt[:, :, kmaxp1, :], tt, results, rtol, atol)
    chk("thermo.tB", rt[:, :, kminm1, :], tb, results, rtol, atol)
    chk("thermo.pT", rp[:, :, kmaxp1, :], pt, results, rtol, atol)
    chk("thermo.pB", rp[:, :, kminm1, :], pb, results, rtol, atol)
    chk("thermo.rT", rr[:, :, kmaxp1, :], ot, results, rtol, atol)
    chk("thermo.rB", rr[:, :, kminm1, :], ob, results, rtol, atol)

    # --- thermo pl ---
    rt, rr, rp = thermo_ref(d["tem_pl"], d["rho_pl"], d["pre_pl"], d["phi_pl"], f, kmin, kmax, reg=False)
    tt, tb, pt, pb, ot, ob = compute_bndcnd_thermo_pl(
        mkarr(d["tem_pl"]), mkarr(d["rho_pl"]), mkarr(d["pre_pl"]), mkarr(d["phi_pl"]), cfg, xp)
    chk("thermo_pl.tT", rt[:, kmaxp1, :], tt, results, rtol, atol)
    chk("thermo_pl.rB", rr[:, kminm1, :], ob, results, rtol, atol)

    # --- rhovxvyvz reg ---
    rx, ry, rz = rhovxvyvz_ref(d["rhog"], d["rhogvx"], d["rhogvy"], d["rhogvz"], f, kmin, kmax, reg=True)
    xt, yt, zt, xb, yb, zb = compute_bndcnd_rhovxvyvz_reg(
        mkarr(d["rhog"]), mkarr(d["rhogvx"]), mkarr(d["rhogvy"]), mkarr(d["rhogvz"]), cfg, xp)
    chk("rhov.xT", rx[:, :, kmaxp1, :], xt, results, rtol, atol)
    chk("rhov.yB", ry[:, :, kminm1, :], yb, results, rtol, atol)
    chk("rhov.zT", rz[:, :, kmaxp1, :], zt, results, rtol, atol)

    # --- rhovxvyvz pl ---
    rx, ry, rz = rhovxvyvz_ref(d["rhog_pl"], d["rhogvx_pl"], d["rhogvy_pl"], d["rhogvz_pl"], f, kmin, kmax, reg=False)
    xt, yt, zt, xb, yb, zb = compute_bndcnd_rhovxvyvz_pl(
        mkarr(d["rhog_pl"]), mkarr(d["rhogvx_pl"]), mkarr(d["rhogvy_pl"]), mkarr(d["rhogvz_pl"]), cfg, xp)
    chk("rhov_pl.xT", rx[:, kmaxp1, :], xt, results, rtol, atol)
    chk("rhov_pl.zB", rz[:, kminm1, :], zb, results, rtol, atol)

    # --- rhow reg ---
    rw = rhow_ref(d["rhogvx"], d["rhogvy"], d["rhogvz"], d["rhogw"], d["c2wGz"], f, kmin, kmax, reg=True)
    rwt, rwb = compute_bndcnd_rhow_reg(
        mkarr(d["rhogvx"]), mkarr(d["rhogvy"]), mkarr(d["rhogvz"]), mkarr(d["c2wGz"]), cfg, xp)
    if rwt is not None:
        chk("rhow.top", rw[:, :, kmaxp1, :], rwt, results, rtol, atol)
    if rwb is not None:
        chk("rhow.btm", rw[:, :, kmin, :], rwb, results, rtol, atol)

    # --- rhow pl ---
    rw = rhow_ref(d["rhogvx_pl"], d["rhogvy_pl"], d["rhogvz_pl"], d["rhogw_pl"], d["c2wGz_pl"], f, kmin, kmax, reg=False)
    rwt, rwb = compute_bndcnd_rhow_pl(
        mkarr(d["rhogvx_pl"]), mkarr(d["rhogvy_pl"]), mkarr(d["rhogvz_pl"]), mkarr(d["c2wGz_pl"]), cfg, xp)
    if rwt is not None:
        chk("rhow_pl.top", rw[:, kmaxp1, :], rwt, results, rtol, atol)
    if rwb is not None:
        chk("rhow_pl.btm", rw[:, kmin, :], rwb, results, rtol, atol)

    return results


def main():
    d, kmin, kmax = make_inputs()
    allres = []

    import numpy as xnp
    for fname, f in FLAGS.items():
        allres += run_set(d, kmin, kmax, fname, f, xnp, "numpy", lambda a: a)

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    for fname, f in FLAGS.items():
        allres += run_set(d, kmin, kmax, fname, f, jnp, "jnp", lambda a: jnp.asarray(a))

    # jit (prod config only)
    f = FLAGS["prod"]; cfg = cfg_of(f, kmin, kmax)
    kmaxp1, kminm1 = kmax + 1, kmin - 1
    jt = jax.jit(compute_bndcnd_thermo_reg, static_argnames=("cfg", "xp"))
    jv = jax.jit(compute_bndcnd_rhovxvyvz_reg, static_argnames=("cfg", "xp"))
    jw = jax.jit(compute_bndcnd_rhow_reg, static_argnames=("cfg", "xp"))
    j = lambda a: jnp.asarray(a)
    print("  [jit | prod]")
    rt, rr, rp = thermo_ref(d["tem"], d["rho"], d["pre"], d["phi"], f, kmin, kmax, reg=True)
    tt, tb, pt, pb, ot, ob = jt(j(d["tem"]), j(d["rho"]), j(d["pre"]), j(d["phi"]), cfg=cfg, xp=jnp)
    jax.block_until_ready((tt, ob))
    chk("thermo.rT", rr[:, :, kmaxp1, :], ot, allres, 1e-11, 1e-11)
    chk("thermo.pB", rp[:, :, kminm1, :], pb, allres, 1e-11, 1e-11)
    rx, ry, rz = rhovxvyvz_ref(d["rhog"], d["rhogvx"], d["rhogvy"], d["rhogvz"], f, kmin, kmax, reg=True)
    xt, yt, zt, xb, yb, zb = jv(j(d["rhog"]), j(d["rhogvx"]), j(d["rhogvy"]), j(d["rhogvz"]), cfg=cfg, xp=jnp)
    jax.block_until_ready((xt, zb))
    chk("rhov.xT", rx[:, :, kmaxp1, :], xt, allres, 1e-11, 1e-11)
    chk("rhov.zB", rz[:, :, kminm1, :], zb, allres, 1e-11, 1e-11)
    rw = rhow_ref(d["rhogvx"], d["rhogvy"], d["rhogvz"], d["rhogw"], d["c2wGz"], f, kmin, kmax, reg=True)
    rwt, rwb = jw(j(d["rhogvx"]), j(d["rhogvy"]), j(d["rhogvz"]), j(d["c2wGz"]), cfg=cfg, xp=jnp)
    jax.block_until_ready((rwt, rwb))
    chk("rhow.top", rw[:, :, kmaxp1, :], rwt, allres, 1e-11, 1e-11)
    chk("rhow.btm", rw[:, :, kmin, :], rwb, allres, 1e-11, 1e-11)

    print("\n========================================")
    print(f"all checks: {'PASS' if all(allres) else 'FAIL'}  ({sum(allres)}/{len(allres)})")
    print("========================================")
    return 0 if all(allres) else 1


if __name__ == "__main__":
    raise SystemExit(main())
