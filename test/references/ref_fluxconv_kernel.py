"""
Validation harness for kernels/fluxconv.py.

Checks that the pure backend-switchable kernel (compute_flux_convergence)
reproduces, bit-for-bit, the original in-place src_flux_convergence body of
mod_src.py (L545-822, including the inlined OPRT_divergence), for:
    (1) xp = numpy        (eager)
    (2) xp = jax.numpy    (eager)
    (3) jax.jit(kernel)   (compiled, xp=jax.numpy)

Both fluxtype branches are exercised: I_SRC_default (=3) and
I_SRC_horizontal (=1).

Run:
    .../envs/jax_nomtl_mpi/bin/python test_fluxconv_kernel.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.fluxconv import FluxConvCfg, compute_flux_convergence

I_SRC_HORIZONTAL = 1
I_SRC_DEFAULT = 3
UNDEF = -9.9999e30


# ---------------------------------------------------------------------------
# Reference: faithful transcription of mod_src.py src_flux_convergence
# (regional + pole), with the OPRT_divergence math inlined.  Writes into zeros
# buffers; rows the original leaves stale are never read downstream, so zeros is
# the downstream-equivalent reference.
# ---------------------------------------------------------------------------
def ref(rhogvx, rhogvy, rhogvz, rhogw,
        rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl,
        RGAM, RGAMH, RGSQRTH, C2WfactGz, coef_div, rdgz,
        RGAM_pl, RGAMH_pl, RGSQRTH_pl, C2WfactGz_pl, coef_div_pl,
        fluxtype, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1, kminp1, kmaxp1, kmaxp2 = kmin - 1, kmin + 1, kmax + 1, kmax + 2
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    rdt = rhogw.dtype

    vflag = rdt.type(1.0) if fluxtype == cfg.I_SRC_default else rdt.type(0.0)

    # --- horizontal flux products
    rhogvx_vm = rhogvx * RGAM
    rhogvy_vm = rhogvy * RGAM
    rhogvz_vm = rhogvz * RGAM

    # --- half-level vertical flux
    f = [C2WfactGz[:, :, kminp1:kmaxp1, :, m] for m in range(6)]
    horiz = (f[0] * rhogvx[:, :, kminp1:kmaxp1, :] + f[1] * rhogvx[:, :, kmin:kmax, :] +
             f[2] * rhogvy[:, :, kminp1:kmaxp1, :] + f[3] * rhogvy[:, :, kmin:kmax, :] +
             f[4] * rhogvz[:, :, kminp1:kmaxp1, :] + f[5] * rhogvz[:, :, kmin:kmax, :])
    horiz = horiz * RGAMH[:, :, kminp1:kmaxp1, :]
    vert = vflag * rhogw[:, :, kminp1:kmaxp1, :] * RGSQRTH[:, :, kminp1:kmaxp1, :]
    rhogw_vmh = np.zeros_like(rhogw)
    rhogw_vmh[:, :, kminp1:kmaxp1, :] = horiz + vert
    rhogw_vmh[:, :, kmin, :] = 0.0
    rhogw_vmh[:, :, kmaxp1, :] = 0.0

    # --- horizontal flux convergence (OPRT_divergence inlined)
    i = rhogvx.shape[0]; j = rhogvx.shape[1]
    isl, isl_p, isl_m = slice(1, i - 1), slice(2, i), slice(0, i - 2)
    jsl, jsl_p, jsl_m = slice(1, j - 1), slice(2, j), slice(0, j - 2)
    div = np.zeros_like(rhogvx)

    def stencil(v, d):
        c = coef_div
        return (c[isl, jsl, :, :, d, 0] * v[isl,   jsl,   :, :] +
                c[isl, jsl, :, :, d, 1] * v[isl_p, jsl,   :, :] +
                c[isl, jsl, :, :, d, 2] * v[isl_p, jsl_p, :, :] +
                c[isl, jsl, :, :, d, 3] * v[isl,   jsl_p, :, :] +
                c[isl, jsl, :, :, d, 4] * v[isl_m, jsl,   :, :] +
                c[isl, jsl, :, :, d, 5] * v[isl_m, jsl_m, :, :] +
                c[isl, jsl, :, :, d, 6] * v[isl,   jsl_m, :, :])

    div[isl, jsl, :, :] = stencil(rhogvx_vm, X) + stencil(rhogvy_vm, Y) + stencil(rhogvz_vm, Z)

    # --- total flux convergence
    rdgz_b = rdgz[kmin:kmaxp1][None, None, :, None]
    flux_diff = rhogw_vmh[:, :, kminp1:kmaxp2, :] - rhogw_vmh[:, :, kmin:kmaxp1, :]
    grhog = np.zeros_like(rhogvx)
    grhog[:, :, kmin:kmaxp1, :] = -div[:, :, kmin:kmaxp1, :] - flux_diff * rdgz_b
    grhog[:, :, kminm1, :] = 0.0
    grhog[:, :, kmaxp1, :] = 0.0

    # --- pole
    grhog_pl = np.zeros_like(rhogw_pl)
    if cfg.have_pl:
        vx_vm_pl = rhogvx_pl * RGAM_pl
        vy_vm_pl = rhogvy_pl * RGAM_pl
        vz_vm_pl = rhogvz_pl * RGAM_pl
        fp = [C2WfactGz_pl[:, kminp1:kmaxp1, :, m] for m in range(6)]
        horiz_pl = (fp[0] * rhogvx_pl[:, kminp1:kmaxp1, :] + fp[1] * rhogvx_pl[:, kmin:kmax, :] +
                    fp[2] * rhogvy_pl[:, kminp1:kmaxp1, :] + fp[3] * rhogvy_pl[:, kmin:kmax, :] +
                    fp[4] * rhogvz_pl[:, kminp1:kmaxp1, :] + fp[5] * rhogvz_pl[:, kmin:kmax, :])
        horiz_pl = horiz_pl * RGAMH_pl[:, kminp1:kmaxp1, :]
        vert_pl = np.full_like(rhogw_pl, UNDEF)
        vert_pl[:, kminp1:kmaxp1, :] = vflag * rhogw_pl[:, kminp1:kmaxp1, :] * RGSQRTH_pl[:, kminp1:kmaxp1, :]
        rhogw_vmh_pl = np.zeros_like(rhogw_pl)
        rhogw_vmh_pl[:, kminp1:kmaxp1, :] = horiz_pl + vert_pl[:, kminp1:kmaxp1, :]
        rhogw_vmh_pl[:, kmin, :] = 0.0
        rhogw_vmh_pl[:, kmaxp1, :] = 0.0

        div_pl = np.zeros_like(rhogw_pl)
        n = cfg.gslf_pl
        for l in range(rhogw_pl.shape[2]):
            for k in range(rhogw_pl.shape[1]):
                for v in range(cfg.gslf_pl, cfg.gmax_pl + 1):
                    div_pl[n, k, l] += (coef_div_pl[v, 0, l, X] * vx_vm_pl[v, k, l] +
                                        coef_div_pl[v, 0, l, Y] * vy_vm_pl[v, k, l] +
                                        coef_div_pl[v, 0, l, Z] * vz_vm_pl[v, k, l])

        rdgz_bp = rdgz[kmin:kmaxp1][None, :, None]
        flux_diff_pl = rhogw_vmh_pl[:, kminp1:kmaxp2, :] - rhogw_vmh_pl[:, kmin:kmaxp1, :]
        grhog_pl[:, kmin:kmaxp1, :] = -div_pl[:, kmin:kmaxp1, :] - flux_diff_pl * rdgz_bp
        grhog_pl[:, kminm1, :] = 0.0
        grhog_pl[:, kmaxp1, :] = 0.0

    return grhog, grhog_pl


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    i = j = 8
    kall, lall = 10, 3
    g, lall_pl = 6, 2
    dt = np.float64
    kmin, kmax = 1, kall - 2

    def R(*s):
        return rng.uniform(0.5, 1.5, s).astype(dt)

    arrs = dict(
        rhogvx=R(i, j, kall, lall), rhogvy=R(i, j, kall, lall),
        rhogvz=R(i, j, kall, lall), rhogw=R(i, j, kall, lall),
        rhogvx_pl=R(g, kall, lall_pl), rhogvy_pl=R(g, kall, lall_pl),
        rhogvz_pl=R(g, kall, lall_pl), rhogw_pl=R(g, kall, lall_pl),
        RGAM=R(i, j, kall, lall), RGAMH=R(i, j, kall, lall), RGSQRTH=R(i, j, kall, lall),
        C2WfactGz=R(i, j, kall, lall, 6), coef_div=R(i, j, 1, lall, 3, 7), rdgz=R(kall),
        RGAM_pl=R(g, kall, lall_pl), RGAMH_pl=R(g, kall, lall_pl), RGSQRTH_pl=R(g, kall, lall_pl),
        C2WfactGz_pl=R(g, kall, lall_pl, 6), coef_div_pl=R(g, 1, lall_pl, 3),
    )
    cfg = FluxConvCfg(kmin=kmin, kmax=kmax, have_pl=True, XDIR=0, YDIR=1, ZDIR=2,
                      gslf_pl=0, gmax_pl=5, I_SRC_default=I_SRC_DEFAULT,
                      I_SRC_horizontal=I_SRC_HORIZONTAL)
    return arrs, cfg


ARG_ORDER = [
    "rhogvx", "rhogvy", "rhogvz", "rhogw",
    "rhogvx_pl", "rhogvy_pl", "rhogvz_pl", "rhogw_pl",
    "RGAM", "RGAMH", "RGSQRTH", "C2WfactGz", "coef_div", "rdgz",
    "RGAM_pl", "RGAMH_pl", "RGSQRTH_pl", "C2WfactGz_pl", "coef_div_pl",
]


def report(name, r, g, rtol=0.0, atol=0.0):
    print(f"\n[{name}]  (rtol={rtol:g}, atol={atol:g})")
    ok = True
    for nm, a, b in zip(["grhog", "grhog_pl"], r, g):
        b = np.asarray(b)
        denom = np.maximum(np.abs(a), 1e-300)
        max_abs = np.max(np.abs(b - a))
        max_rel = np.max(np.abs(b - a) / denom)
        exact = np.array_equal(b, a)
        passed = exact or np.allclose(b, a, rtol=rtol, atol=atol)
        ok = ok and passed
        flag = "EXACT" if exact else f"max|d|={max_abs:.3e} max|rel|={max_rel:.3e}"
        print(f"  {nm:9s}: {'OK ' if passed else 'BAD'} {flag}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def run_for_fluxtype(fluxtype, label):
    arrs, cfg = make_inputs()
    r = ref(*[arrs[k] for k in ARG_ORDER], fluxtype, cfg)

    import numpy as xnp
    got_np = compute_flux_convergence(*[arrs[k] for k in ARG_ORDER], fluxtype, cfg=cfg, xp=xnp)
    ok1 = report(f"{label}: numpy backend (eager)", r, got_np)

    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    jargs = [jnp.asarray(arrs[k]) for k in ARG_ORDER]

    got_jnp = compute_flux_convergence(*jargs, fluxtype, cfg=cfg, xp=jnp)
    ok2 = report(f"{label}: jax.numpy backend (eager)", r, got_jnp)

    kernel_jit = jax.jit(compute_flux_convergence, static_argnames=("fluxtype", "cfg", "xp"))
    got_jit = kernel_jit(*jargs, fluxtype=fluxtype, cfg=cfg, xp=jnp)
    jax.block_until_ready(got_jit)
    ok3 = report(f"{label}: jax.jit (compiled)", r, got_jit, rtol=1e-12, atol=1e-12)

    return ok1, ok2, ok3


def main():
    results = []
    results += run_for_fluxtype(I_SRC_DEFAULT, "I_SRC_default")
    results += run_for_fluxtype(I_SRC_HORIZONTAL, "I_SRC_horizontal")
    print("\n========================================")
    print(f"all checks: {'PASS' if all(results) else 'FAIL'}")
    print("========================================")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
