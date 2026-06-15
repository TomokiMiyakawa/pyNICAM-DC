"""
vi_main device-resident FUSION benchmark -- using the REAL extracted kernels.

This composes the actual pure kernel functions that vi_main is built from
(fluxconv / advconv / virhowsolver / bndcnd-rhow / rhogkin) into ONE jitted
function, on the real per-rank shapes (glevel=5,rlevel=1,vlayer=40 -> (18,18,42,5)),
and measures:

  1. numpy                         (current backend class)
  2. jax-jit, device-resident      (fused future: prognostic state stays on device)
  3. jax-jit, + host<->device      (current per-call dispatch with asarray/to_numpy)

Unlike bench_fusion.py (synthetic op-mix), this runs the REAL model arithmetic:
the real 7-point hexagonal OPRT divergence (gather/shift), the real sequential
Thomas solve, the real rhogkin vertical interpolation, and the real elementwise
energy/prognostic glue -- i.e. the genuine fusion ceiling for vi_main's
comm-free core.

Regional-only (have_pl=False): the pole arrays are <2% of elements and don't
move the per-call perf signal. This is a PERFORMANCE proxy, not a bit-exact
extraction -- constants are random-but-correctly-shaped.
"""

import time
from functools import partial
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pynicamdc.nhm.dynamics.kernels.fluxconv import compute_flux_convergence, FluxConvCfg
from pynicamdc.nhm.dynamics.kernels.advconv import compute_scaled_fluxes, AdvConvCfg
from pynicamdc.nhm.dynamics.kernels.virhowsolver import compute_rhow_solver_reg, ViSolverCfg
from pynicamdc.nhm.dynamics.kernels.bndcnd import compute_bndcnd_rhow_reg, BndCfg
from pynicamdc.nhm.dynamics.kernels.rhogkin import compute_rhogkin_reg, RhogkinCfg

# ---- real per-rank shapes ----
I = J = 18
KALL = 42
L = 5
KMIN, KMAX = 1, KALL - 2          # 1, 40
SHAPE = (I, J, KALL, L)
DT = 20.0
GRAV, RDRY, CVDRY, ALPHA = 9.80616, 287.04, 717.6, 1.0
XDIR, YDIR, ZDIR = 0, 1, 2
I_SRC_DEFAULT, I_SRC_HORIZONTAL = 0, 1

cfgs = {
    "flux": FluxConvCfg(KMIN, KMAX, False, XDIR, YDIR, ZDIR, 0, 4, I_SRC_DEFAULT, I_SRC_HORIZONTAL),
    "adv":  AdvConvCfg(KMIN, KMAX, False, I_SRC_DEFAULT),
    "sol":  ViSolverCfg(KMIN, KMAX, False, GRAV, RDRY, CVDRY, ALPHA),
    "bnd":  BndCfg(KMIN, KMAX, False, False, False, False, False,
                   False, True, True, False, GRAV, RDRY),   # top free, btm rigid
    "kin":  RhogkinCfg(KMIN, KMAX, False, -999.0),
}


def make_const(xp, seed=1):
    rng = np.random.default_rng(seed)
    f = lambda shp, lo=0.5, hi=1.5: xp.asarray(rng.uniform(lo, hi, shp).astype(np.float64))
    return {
        "RGAM":     f(SHAPE), "RGAMH": f(SHAPE), "RGSQRTH": f(SHAPE),
        "RGSGAM2":  f(SHAPE), "RGSGAM2H": f(SHAPE), "GSGAM2H": f(SHAPE),
        "C2WfactGz": f((I, J, KALL, L, 6), -0.5, 0.5),
        "coef_div":  f((I, J, 1, L, 3, 7), -0.5, 0.5),
        "rdgz":  f((KALL,)), "rdgzh": f((KALL,)),
        "afact": f((KALL,), 0.0, 1.0), "bfact": f((KALL,), 0.0, 1.0),
        "C2Wfact": f((I, J, KALL, L, 2), 0.0, 1.0),
        "W2Cfact": f((I, J, KALL, L, 2), 0.0, 1.0),
        "PHI": f(SHAPE, 0.0, 1e4),
    }


def make_prog(xp, seed=2):
    rng = np.random.default_rng(seed)
    f = lambda lo=-1.0, hi=1.0: xp.asarray(rng.uniform(lo, hi, SHAPE).astype(np.float64))
    return {
        "vx_s1": f(), "vy_s1": f(), "vz_s1": f(),
        "rhog_s0": f(0.5, 1.5), "vx_s0": f(), "vy_s0": f(), "vz_s0": f(),
        "rhogw_s0": f(), "rhoge_s0": f(0.5, 1.5), "preg_s0": f(0.5, 1.5),
        "rhog0": f(0.5, 1.5), "vx0": f(), "vy0": f(), "vz0": f(), "rhogw0": f(),
        "eth0": f(0.5, 1.5),
        "grhog": f(-0.1, 0.1), "grhogw": f(-0.1, 0.1),
        "grhoge": f(-0.1, 0.1), "grhogetot": f(-0.1, 0.1),
        # tridiagonal matrix coefs (diag-dominant for stability)
        "Mc": f(2.0, 3.0), "Mu": f(-0.5, -0.1), "Ml": f(-0.5, -0.1),
    }


def _build_rhogw_bc(vx, vy, vz, c2w, xp):
    """rhogw_split1 = 0 everywhere, then BNDCND_rhow sets kmin & kmax+1 rows."""
    rw_top, rw_btm = compute_bndcnd_rhow_reg(vx, vy, vz, c2w, cfgs["bnd"], xp)
    row0   = xp.zeros((I, J, 1, L), dtype=vx.dtype)              # k=0 (kmin-1)
    rkmin  = (rw_btm if rw_btm is not None else xp.zeros((I, J, L), vx.dtype))[:, :, None, :]
    inter  = xp.zeros((I, J, KMAX - KMIN, L), dtype=vx.dtype)    # k=kmin+1..kmax
    rkmaxp = (rw_top if rw_top is not None else xp.zeros((I, J, L), vx.dtype))[:, :, None, :]
    return xp.concatenate([row0, rkmin, inter, rkmaxp], axis=2)


def _flux_conv(vx, vy, vz, w, C, fluxtype, xp):
    z = xp.zeros((1,), dtype=vx.dtype)
    grhog, _ = compute_flux_convergence(
        vx, vy, vz, w, vx, vy, vz, w,
        C["RGAM"], C["RGAMH"], C["RGSQRTH"], C["C2WfactGz"], C["coef_div"], C["rdgz"],
        C["RGAM"], C["RGAMH"], C["RGSQRTH"], C["C2WfactGz"], C["coef_div"],
        fluxtype, cfgs["flux"], xp)
    return grhog


def _adv_conv(vx, vy, vz, w, scl, C, fluxtype, xp):
    sx, sy, sz, sw, *_ = compute_scaled_fluxes(
        vx, vy, vz, w, scl, vx, vy, vz, w, scl,
        C["afact"], C["bfact"], fluxtype, cfgs["adv"], xp)
    return _flux_conv(sx, sy, sz, sw, C, fluxtype, xp)


def _rhogkin(rhog, vx, vy, vz, w, C, xp):
    return compute_rhogkin_reg(rhog, vx, vy, vz, w, C["C2Wfact"], C["W2Cfact"], cfgs["kin"], xp)


def vi_main_core(P, C, xp):
    """Single fused composition of vi_main's comm-free regional core."""
    # 1) split source terms (TIME_split branch -> horizontal flux/adv convergence)
    drhog  = _flux_conv(P["vx_s1"], P["vy_s1"], P["vz_s1"], P["rhogw_s0"], C, I_SRC_HORIZONTAL, xp)
    drhoge = _adv_conv(P["vx_s1"], P["vy_s1"], P["vz_s1"], P["rhogw_s0"], P["eth0"], C, I_SRC_HORIZONTAL, xp)

    # 2) grhog1/grhoge1/gpre
    grhog1  = P["grhog"]  + drhog
    grhoge1 = P["grhoge"] + drhoge
    gpre    = grhoge1 * RDRY / CVDRY

    # 3) rhogw_split1 boundary init
    rhogw_s1 = _build_rhogw_bc(P["vx_s1"], P["vy_s1"], P["vz_s1"], C["C2WfactGz"], xp)

    # 4) vertical-implicit Thomas solve
    rhogw_s1 = compute_rhow_solver_reg(
        rhogw_s1, P["rhogw_s0"], P["preg_s0"], P["rhog_s0"],
        grhog1, P["grhogw"], gpre, P["Mc"], P["Mu"], P["Ml"],
        C["RGAMH"], C["RGSGAM2"], C["RGAM"], C["RGSGAM2H"], C["GSGAM2H"],
        C["rdgzh"], C["afact"], C["bfact"], DT, cfgs["sol"], xp)

    # 5) rhog_split1 via default flux convergence
    drhog2 = _flux_conv(P["vx_s1"], P["vy_s1"], P["vz_s1"], rhogw_s1, C, I_SRC_DEFAULT, xp)
    rhog_s1 = P["rhog_s0"] + (P["grhog"] + drhog2) * DT

    # 6) three rhogkin evaluations
    kin0 = _rhogkin(P["rhog0"], P["vx0"], P["vy0"], P["vz0"], P["rhogw0"], C, xp)
    rhog1   = P["rhog0"] + P["rhog_s0"]
    vx1     = P["vx0"] + P["vx_s0"]; vy1 = P["vy0"] + P["vy_s0"]; vz1 = P["vz0"] + P["vz_s0"]
    rhogw1  = P["rhogw0"] + P["rhogw_s0"]
    kin10 = _rhogkin(rhog1, vx1, vy1, vz1, rhogw1, C, xp)
    rhog1b  = P["rhog0"] + rhog_s1
    vx1b = P["vx0"] + P["vx_s1"]; vy1b = P["vy0"] + P["vy_s1"]; vz1b = P["vz0"] + P["vz_s1"]
    rhogw1b = P["rhogw0"] + rhogw_s1
    kin11 = _rhogkin(rhog1b, vx1b, vy1b, vz1b, rhogw1b, C, xp)

    # 7) energy correction
    ethtot0 = P["eth0"] + kin0 / P["rhog0"] + C["PHI"]
    drhogetot = _adv_conv(vx1b, vy1b, vz1b, rhogw1b, ethtot0, C, I_SRC_DEFAULT, xp)
    rhoge_s1 = (P["rhoge_s0"] + (P["grhogetot"] + drhogetot) * DT
                + (kin10 - kin11) + (P["rhog_s0"] - rhog_s1) * C["PHI"])

    return rhog_s1, rhogw_s1, rhoge_s1


def time_it(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    N = 100
    Cn, Pn = make_const(np), make_prog(np)
    np_call = lambda: vi_main_core(Pn, Cn, np)
    t_np = time_it(np_call, N)

    core_jit = jax.jit(partial(vi_main_core, xp=jnp))
    Cj, Pj = make_const(jnp), make_prog(jnp)
    tc0 = time.perf_counter()
    jax.block_until_ready(core_jit(Pj, Cj))
    t_comp = time.perf_counter() - tc0

    def jres():
        jax.block_until_ready(core_jit(Pj, Cj))
    t_res = time_it(jres, N)

    def jxfer():
        Pin = {k: jnp.asarray(v) for k, v in Pn.items()}
        Cin = {k: jnp.asarray(v) for k, v in Cn.items()}
        out = core_jit(Pin, Cin)
        [np.asarray(o) for o in jax.block_until_ready(out)]
    t_xfer = time_it(jxfer, N)

    print(f"shape (per rank)            : {SHAPE}  ({np.prod(SHAPE):,} elems, float64)")
    print(f"iterations timed            : {N}")
    print()
    print(f"1) numpy                    : {t_np*1e3:8.3f} ms/call   (baseline)")
    print(f"2) jax-jit device-resident  : {t_res*1e3:8.3f} ms/call   -> {t_np/t_res:5.2f}x")
    print(f"3) jax-jit + host transfer  : {t_xfer*1e3:8.3f} ms/call   -> {t_np/t_xfer:5.2f}x")
    print()
    print(f"jax first-call compile time : {t_comp*1e3:8.1f} ms (one-time)")
    print()
    a = vi_main_core(make_prog(np, 5), make_const(np, 5), np)
    b = vi_main_core(make_prog(jnp, 5), make_const(jnp, 5), jnp)
    mr = max(float(np.max(np.abs(np.asarray(bi) - ai) / (np.abs(ai) + 1e-30))) for ai, bi in zip(a, b))
    print(f"numpy vs jax max rel diff   : {mr:.3e}  (sanity)")


if __name__ == "__main__":
    main()
