"""
Validation harness for kernels/horizontalflux.py.

Checks that the pure backend-switchable kernel (compute_horizontal_flux)
reproduces, bit-for-bit (numpy) / to round-off (jax), the vectorised in-place
horizontal_flux body of mod_src_tracer.py for:
    (1) xp = numpy        (eager)
    (2) xp = jax.numpy    (eager)
    (3) jax.jit(kernel)   (compiled, xp=jax.numpy)

Run:
    .../envs/jax_nomtl_mpi/bin/python test_horizontalflux_kernel.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.horizontalflux import (  # noqa: E402
    HorizFluxCfg, compute_horizontal_flux,
)

# index conventions for the test
K0 = 0
TI, TJ = 0, 1
AI, AIJ, AJ = 0, 1, 2
W1, W2, W3 = 0, 1, 2
HNX, HNY, HNZ = 0, 1, 2
P_RAREA = 0
XDIR, YDIR, ZDIR = 0, 1, 2
EPS = 1.0e-16


def ref(rho, rhovx, rhovy, rhovz, rho_pl, rhovx_pl, rhovy_pl, rhovz_pl,
        GMTR_t, GMTR_a, GMTR_p, GRD_xr, pntmask,
        GMTR_t_pl, GMTR_a_pl, GMTR_p_pl, GRD_xr_pl, dt, cfg):
    """Transcription of the vectorised-over-k horizontal_flux (regional+pole)."""
    rdt = rho.dtype.type
    iall, jall, kall, lall = rho.shape
    gall_pl, _, lall_pl = rho_pl.shape

    flx_h = np.zeros((iall, jall, kall, lall, 6), dtype=rho.dtype)
    grd_xc = np.zeros((iall, jall, kall, lall, 3, 3), dtype=rho.dtype)
    rhot_TI = np.zeros((iall, jall, kall), dtype=rho.dtype)
    rhot_TJ = np.zeros_like(rhot_TI)
    rhovxt_TI = np.zeros_like(rhot_TI); rhovxt_TJ = np.zeros_like(rhot_TI)
    rhovyt_TI = np.zeros_like(rhot_TI); rhovyt_TJ = np.zeros_like(rhot_TI)
    rhovzt_TI = np.zeros_like(rhot_TI); rhovzt_TJ = np.zeros_like(rhot_TI)

    for l in range(lall):
        isl = slice(0, iall - 1); jsl = slice(0, jall - 1)
        isl_p = slice(1, iall); jsl_p = slice(1, jall)

        def vert(f, T, a, b2, b3):
            return (f[a[0], a[1], :, l] * GMTR_t[isl, jsl, K0, l, T, W1][:, :, None]
                    + f[b2[0], b2[1], :, l] * GMTR_t[isl, jsl, K0, l, T, W2][:, :, None]
                    + f[b3[0], b3[1], :, l] * GMTR_t[isl, jsl, K0, l, T, W3][:, :, None])
        # TI: (i,j),(i+1,j),(i+1,j+1) ; TJ: (i,j),(i+1,j+1),(i,j+1)
        rhot_TI[isl, jsl, :]   = vert(rho,   TI, (isl, jsl), (isl_p, jsl), (isl_p, jsl_p))
        rhovxt_TI[isl, jsl, :] = vert(rhovx, TI, (isl, jsl), (isl_p, jsl), (isl_p, jsl_p))
        rhovyt_TI[isl, jsl, :] = vert(rhovy, TI, (isl, jsl), (isl_p, jsl), (isl_p, jsl_p))
        rhovzt_TI[isl, jsl, :] = vert(rhovz, TI, (isl, jsl), (isl_p, jsl), (isl_p, jsl_p))
        rhot_TJ[isl, jsl, :]   = vert(rho,   TJ, (isl, jsl), (isl_p, jsl_p), (isl, jsl_p))
        rhovxt_TJ[isl, jsl, :] = vert(rhovx, TJ, (isl, jsl), (isl_p, jsl_p), (isl, jsl_p))
        rhovyt_TJ[isl, jsl, :] = vert(rhovy, TJ, (isl, jsl), (isl_p, jsl_p), (isl, jsl_p))
        rhovzt_TJ[isl, jsl, :] = vert(rhovz, TJ, (isl, jsl), (isl_p, jsl_p), (isl, jsl_p))

        m0 = pntmask[K0, l, 0]; m1 = pntmask[K0, l, 1]
        rhot_TI[0, 0, :]   = rhot_TI[0, 0, :]   * m0 + rhot_TJ[1, 0, :]   * m1
        rhovxt_TI[0, 0, :] = rhovxt_TI[0, 0, :] * m0 + rhovxt_TJ[1, 0, :] * m1
        rhovyt_TI[0, 0, :] = rhovyt_TI[0, 0, :] * m0 + rhovyt_TJ[1, 0, :] * m1
        rhovzt_TI[0, 0, :] = rhovzt_TI[0, 0, :] * m0 + rhovzt_TJ[1, 0, :] * m1

        # AI
        isl = slice(0, iall - 1); jsl = slice(1, jall - 1); jslm1 = slice(0, jall - 2)
        rrhoa2 = rdt(1.0) / np.maximum(rhot_TJ[isl, jslm1, :] + rhot_TI[isl, jsl, :], EPS)
        vx2 = rhovxt_TJ[isl, jslm1, :] + rhovxt_TI[isl, jsl, :]
        vy2 = rhovyt_TJ[isl, jslm1, :] + rhovyt_TI[isl, jsl, :]
        vz2 = rhovzt_TJ[isl, jslm1, :] + rhovzt_TI[isl, jsl, :]
        flux = rdt(0.5) * (vx2 * GMTR_a[isl, jsl, K0, l, AI, HNX][:, :, None]
                           + vy2 * GMTR_a[isl, jsl, K0, l, AI, HNY][:, :, None]
                           + vz2 * GMTR_a[isl, jsl, K0, l, AI, HNZ][:, :, None])
        flx_h[isl, jsl, :, l, 0] = flux * GMTR_p[isl, jsl, K0, l, P_RAREA][:, :, None] * dt
        flx_h[1:iall, jsl, :, l, 3] = -flux * GMTR_p[1:iall, jsl, K0, l, P_RAREA][:, :, None] * dt
        grd_xc[isl, jsl, :, l, AI, XDIR] = GRD_xr[isl, jsl, K0, l, AI, XDIR][:, :, None] - vx2 * rrhoa2 * dt * rdt(0.5)
        grd_xc[isl, jsl, :, l, AI, YDIR] = GRD_xr[isl, jsl, K0, l, AI, YDIR][:, :, None] - vy2 * rrhoa2 * dt * rdt(0.5)
        grd_xc[isl, jsl, :, l, AI, ZDIR] = GRD_xr[isl, jsl, K0, l, AI, ZDIR][:, :, None] - vz2 * rrhoa2 * dt * rdt(0.5)

        # AIJ
        isl = slice(0, iall - 1); jsl = slice(0, jall - 1)
        rrhoa2 = rdt(1.0) / np.maximum(rhot_TI[isl, jsl, :] + rhot_TJ[isl, jsl, :], EPS)
        vx2 = rhovxt_TI[isl, jsl, :] + rhovxt_TJ[isl, jsl, :]
        vy2 = rhovyt_TI[isl, jsl, :] + rhovyt_TJ[isl, jsl, :]
        vz2 = rhovzt_TI[isl, jsl, :] + rhovzt_TJ[isl, jsl, :]
        flux = rdt(0.5) * (vx2 * GMTR_a[isl, jsl, K0, l, AIJ, HNX][:, :, None]
                           + vy2 * GMTR_a[isl, jsl, K0, l, AIJ, HNY][:, :, None]
                           + vz2 * GMTR_a[isl, jsl, K0, l, AIJ, HNZ][:, :, None])
        flx_h[isl, jsl, :, l, 1] = flux * GMTR_p[isl, jsl, K0, l, P_RAREA][:, :, None] * dt
        flx_h[1:iall, 1:jall, :, l, 4] = -flux * GMTR_p[1:iall, 1:jall, K0, l, P_RAREA][:, :, None] * dt
        grd_xc[isl, jsl, :, l, AIJ, XDIR] = GRD_xr[isl, jsl, K0, l, AIJ, XDIR][:, :, None] - vx2 * rrhoa2 * dt * rdt(0.5)
        grd_xc[isl, jsl, :, l, AIJ, YDIR] = GRD_xr[isl, jsl, K0, l, AIJ, YDIR][:, :, None] - vy2 * rrhoa2 * dt * rdt(0.5)
        grd_xc[isl, jsl, :, l, AIJ, ZDIR] = GRD_xr[isl, jsl, K0, l, AIJ, ZDIR][:, :, None] - vz2 * rrhoa2 * dt * rdt(0.5)

        # AJ
        isl = slice(1, iall - 1); jsl = slice(0, jall - 1)
        rrhoa2 = rdt(1.0) / np.maximum(rhot_TJ[isl, jsl, :] + rhot_TI[0:iall - 2, jsl, :], EPS)
        vx2 = rhovxt_TJ[isl, jsl, :] + rhovxt_TI[0:iall - 2, jsl, :]
        vy2 = rhovyt_TJ[isl, jsl, :] + rhovyt_TI[0:iall - 2, jsl, :]
        vz2 = rhovzt_TJ[isl, jsl, :] + rhovzt_TI[0:iall - 2, jsl, :]
        flux = rdt(0.5) * (vx2 * GMTR_a[isl, jsl, K0, l, AJ, HNX][:, :, None]
                           + vy2 * GMTR_a[isl, jsl, K0, l, AJ, HNY][:, :, None]
                           + vz2 * GMTR_a[isl, jsl, K0, l, AJ, HNZ][:, :, None])
        flx_h[isl, jsl, :, l, 2] = flux * GMTR_p[isl, jsl, K0, l, P_RAREA][:, :, None] * dt
        flx_h[isl, 1:jall, :, l, 5] = -flux * GMTR_p[isl, 1:jall, K0, l, P_RAREA][:, :, None] * dt
        grd_xc[isl, jsl, :, l, AJ, XDIR] = GRD_xr[isl, jsl, K0, l, AJ, XDIR][:, :, None] - vx2 * rrhoa2 * dt * rdt(0.5)
        grd_xc[isl, jsl, :, l, AJ, YDIR] = GRD_xr[isl, jsl, K0, l, AJ, YDIR][:, :, None] - vy2 * rrhoa2 * dt * rdt(0.5)
        grd_xc[isl, jsl, :, l, AJ, ZDIR] = GRD_xr[isl, jsl, K0, l, AJ, ZDIR][:, :, None] - vz2 * rrhoa2 * dt * rdt(0.5)

        flx_h[1, 1, :, l, 5] *= rdt(pntmask[K0, l, 0])

    # pole
    flx_h_pl = np.zeros((gall_pl, kall, lall_pl), dtype=rho.dtype)
    grd_xc_pl = np.zeros((gall_pl, kall, lall_pl, 3), dtype=rho.dtype)
    rhot_pl = np.zeros((gall_pl, kall), dtype=rho.dtype)
    rhovxt_pl = np.zeros_like(rhot_pl); rhovyt_pl = np.zeros_like(rhot_pl); rhovzt_pl = np.zeros_like(rhot_pl)
    if cfg.have_pl:
        n = cfg.gslf_pl
        for l in range(lall_pl):
            for v in range(cfg.gmin_pl, cfg.gmax_pl + 1):
                ij = v; ijp1 = v + 1
                if ijp1 == cfg.gmax_pl + 1:
                    ijp1 = cfg.gmin_pl
                rhot_pl[v, :]   = rho_pl[n, :, l]   * GMTR_t_pl[ij, K0, l, W1] + rho_pl[ij, :, l]   * GMTR_t_pl[ij, K0, l, W2] + rho_pl[ijp1, :, l]   * GMTR_t_pl[ij, K0, l, W3]
                rhovxt_pl[v, :] = rhovx_pl[n, :, l] * GMTR_t_pl[ij, K0, l, W1] + rhovx_pl[ij, :, l] * GMTR_t_pl[ij, K0, l, W2] + rhovx_pl[ijp1, :, l] * GMTR_t_pl[ij, K0, l, W3]
                rhovyt_pl[v, :] = rhovy_pl[n, :, l] * GMTR_t_pl[ij, K0, l, W1] + rhovy_pl[ij, :, l] * GMTR_t_pl[ij, K0, l, W2] + rhovy_pl[ijp1, :, l] * GMTR_t_pl[ij, K0, l, W3]
                rhovzt_pl[v, :] = rhovz_pl[n, :, l] * GMTR_t_pl[ij, K0, l, W1] + rhovz_pl[ij, :, l] * GMTR_t_pl[ij, K0, l, W2] + rhovz_pl[ijp1, :, l] * GMTR_t_pl[ij, K0, l, W3]
            for v in range(cfg.gmin_pl, cfg.gmax_pl + 1):
                ij = v; ijm1 = v - 1
                if ijm1 == cfg.gmin_pl - 1:
                    ijm1 = cfg.gmax_pl
                rrhoa2 = rdt(1.0) / np.maximum(rhot_pl[ijm1, :] + rhot_pl[ij, :], EPS)
                vx2 = rhovxt_pl[ijm1, :] + rhovxt_pl[ij, :]
                vy2 = rhovyt_pl[ijm1, :] + rhovyt_pl[ij, :]
                vz2 = rhovzt_pl[ijm1, :] + rhovzt_pl[ij, :]
                flux = rdt(0.5) * (vx2 * GMTR_a_pl[ij, K0, l, HNX] + vy2 * GMTR_a_pl[ij, K0, l, HNY] + vz2 * GMTR_a_pl[ij, K0, l, HNZ])
                flx_h_pl[v, :, l] = flux * GMTR_p_pl[n, K0, l, P_RAREA] * dt
                grd_xc_pl[v, :, l, XDIR] = GRD_xr_pl[v, K0, l, XDIR] - vx2 * rrhoa2 * dt * rdt(0.5)
                grd_xc_pl[v, :, l, YDIR] = GRD_xr_pl[v, K0, l, YDIR] - vy2 * rrhoa2 * dt * rdt(0.5)
                grd_xc_pl[v, :, l, ZDIR] = GRD_xr_pl[v, K0, l, ZDIR] - vz2 * rrhoa2 * dt * rdt(0.5)

    return flx_h, grd_xc, flx_h_pl, grd_xc_pl


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    i = j = 8
    kall, lall = 10, 3
    g, lall_pl = 6, 2
    dt = np.float64

    def R(*s):
        return rng.uniform(0.5, 1.5, s).astype(dt)

    arrs = dict(
        rho=R(i, j, kall, lall), rhovx=R(i, j, kall, lall),
        rhovy=R(i, j, kall, lall), rhovz=R(i, j, kall, lall),
        rho_pl=R(g, kall, lall_pl), rhovx_pl=R(g, kall, lall_pl),
        rhovy_pl=R(g, kall, lall_pl), rhovz_pl=R(g, kall, lall_pl),
        GMTR_t=R(i, j, 1, lall, 2, 3), GMTR_a=R(i, j, 1, lall, 3, 3),
        GMTR_p=R(i, j, 1, lall, 1), GRD_xr=R(i, j, 1, lall, 3, 3),
        pntmask=R(1, lall, 2),
        GMTR_t_pl=R(g, 1, lall_pl, 3), GMTR_a_pl=R(g, 1, lall_pl, 3),
        GMTR_p_pl=R(g, 1, lall_pl, 1), GRD_xr_pl=R(g, 1, lall_pl, 3),
    )
    cfg = HorizFluxCfg(iall=i, jall=j, K0=K0, TI=TI, TJ=TJ, AI=AI, AIJ=AIJ, AJ=AJ,
                       W1=W1, W2=W2, W3=W3, HNX=HNX, HNY=HNY, HNZ=HNZ,
                       P_RAREA=P_RAREA, XDIR=XDIR, YDIR=YDIR, ZDIR=ZDIR,
                       have_pl=True, gslf_pl=0, gmin_pl=1, gmax_pl=5, EPS=EPS)
    return arrs, cfg


ARG_ORDER = [
    "rho", "rhovx", "rhovy", "rhovz", "rho_pl", "rhovx_pl", "rhovy_pl", "rhovz_pl",
    "GMTR_t", "GMTR_a", "GMTR_p", "GRD_xr", "pntmask",
    "GMTR_t_pl", "GMTR_a_pl", "GMTR_p_pl", "GRD_xr_pl",
]
OUT = ["flx_h", "grd_xc", "flx_h_pl", "grd_xc_pl"]


def report(name, r, g, rtol=0.0, atol=0.0):
    print(f"\n[{name}]  (rtol={rtol:g}, atol={atol:g})")
    ok = True
    for nm, a, b in zip(OUT, r, g):
        a = np.asarray(a); b = np.asarray(b)
        if a.shape != b.shape:
            print(f"  {nm:10s}: BAD shape {b.shape} != {a.shape}"); ok = False; continue
        exact = np.array_equal(b, a)
        passed = exact or np.allclose(b, a, rtol=rtol, atol=atol)
        ok = ok and passed
        mad = np.max(np.abs(b - a)) if b.size else 0.0
        print(f"  {nm:10s}: {'OK ' if passed else 'BAD'} {'EXACT' if exact else f'max|d|={mad:.3e}'}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    arrs, cfg = make_inputs()
    dt = np.float64(20.0)
    r = ref(*[arrs[k] for k in ARG_ORDER], dt, cfg)

    import numpy as xnp
    got = compute_horizontal_flux(*[arrs[k] for k in ARG_ORDER], dt, cfg=cfg, xp=xnp)
    ok1 = report("numpy backend (eager)", r, got)

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jargs = [jnp.asarray(arrs[k]) for k in ARG_ORDER]
    gj = compute_horizontal_flux(*jargs, dt, cfg=cfg, xp=jnp)
    ok2 = report("jax.numpy backend (eager)", r, gj)

    kjit = jax.jit(compute_horizontal_flux, static_argnames=("cfg", "xp"))
    gjit = kjit(*jargs, dt, cfg=cfg, xp=jnp)
    jax.block_until_ready(gjit)
    ok3 = report("jax.jit (compiled)", r, gjit, rtol=1e-12, atol=1e-12)

    print("\n========================================")
    print(f"all checks: {'PASS' if (ok1 and ok2 and ok3) else 'FAIL'}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
