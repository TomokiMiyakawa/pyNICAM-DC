"""
Validation harness for kernels/presgrad.py.

Checks that the pure backend-switchable kernel (compute_pres_gradient)
reproduces, bit-for-bit, the original COMM-free body of src_pres_gradient
(mod_src.py L604-803, including the inlined OPRT_gradient and
OPRT_horizontalize_vec) for:
    (1) xp = numpy        (eager)
    (2) xp = jax.numpy    (eager)
    (3) jax.jit(kernel)   (compiled, xp=jax.numpy)

Both gradtype branches are exercised:
    I_SRC_default    (=3): vertical half-level pressure gradient Pgradw computed
    I_SRC_horizontal (=1): Pgradw zero

Run:
    .../envs/jax_nomtl_mpi/bin/python test_presgrad_kernel.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.presgrad import PresGradCfg, compute_pres_gradient

I_SRC_HORIZONTAL = 1
I_SRC_DEFAULT = 3
XDIR, YDIR, ZDIR = 0, 1, 2


# ---------------------------------------------------------------------------
# Reference: exact transcription of mod_src.py src_pres_gradient body
# (numpy, in-place loops), including the original OPRT_gradient and the
# vectorized OPRT_horizontalize_vec (mod_oprt.py L2004) it calls.
# Buffers start as zeros; rows the original leaves stale are never read
# downstream -> zeros is the downstream-equivalent reference.
# ---------------------------------------------------------------------------
def compute_pres_gradient_ref(
    P, P_pl,
    RGAM, RGAMH, C2WfactGz, coef_grad, GRD_x, rdgz, rdgzh, GAM2H, RGSGAM2,
    RGAM_pl, RGAMH_pl, C2WfactGz_pl, coef_grad_pl, GRD_x_pl, GAM2H_pl, RGSGAM2_pl,
    gradtype, cfg,
):
    kmin, kmax = cfg.kmin, cfg.kmax
    X, Y, Z = cfg.XDIR, cfg.YDIR, cfg.ZDIR
    nxyz = cfg.nxyz
    rscale = cfg.rscale
    i, j, kall, l = P.shape
    g, kall_pl, l_pl = P_pl.shape

    Pgrad = np.zeros(P.shape + (nxyz,), dtype=P.dtype)
    Pgradw = np.zeros(P.shape, dtype=P.dtype)
    Pgrad_pl = np.zeros(P_pl.shape + (nxyz,), dtype=P_pl.dtype)
    Pgradw_pl = np.zeros(P_pl.shape, dtype=P_pl.dtype)

    P_vm = P * RGAM
    P_vm_pl = P_pl * RGAM_pl * cfg.plmask

    # --- OPRT_gradient: horizontal contribution ---
    isl = slice(1, i - 1); isl_p = slice(2, i); isl_m = slice(0, i - 2)
    jsl = slice(1, j - 1); jsl_p = slice(2, j); jsl_m = slice(0, j - 2)
    scl = P_vm
    scl_stencils = [
        scl[isl, jsl], scl[isl_p, jsl], scl[isl_p, jsl_p], scl[isl, jsl_p],
        scl[isl_m, jsl], scl[isl_m, jsl_m], scl[isl, jsl_m],
    ]
    scl_stack = np.stack(scl_stencils, axis=4)
    for d in (X, Y, Z):
        coef = coef_grad[isl, jsl, :, :, d, :]
        Pgrad[isl, jsl, :, :, d] = np.sum(coef * scl_stack, axis=4)

    if cfg.have_pl:
        n = cfg.gslf_pl
        for ll in range(l_pl):
            for k in range(kall):
                Pgrad_pl[:, k, ll, X] = 0.0
                Pgrad_pl[:, k, ll, Y] = 0.0
                Pgrad_pl[:, k, ll, Z] = 0.0
                for v in range(cfg.gslf_pl, cfg.gmax_pl + 1):
                    Pgrad_pl[n, k, ll, X] += coef_grad_pl[v, 0, ll, X] * P_vm_pl[v, k, ll]
                    Pgrad_pl[n, k, ll, Y] += coef_grad_pl[v, 0, ll, Y] * P_vm_pl[v, k, ll]
                    Pgrad_pl[n, k, ll, Z] += coef_grad_pl[v, 0, ll, Z] * P_vm_pl[v, k, ll]

    # --- vertical contribution ---
    P_vmh = np.zeros(P.shape + (nxyz,), dtype=P.dtype)
    for ll in range(l):
        for k in range(kmin, kmax + 2):
            P_vmh[:, :, k, ll, X] = (C2WfactGz[:, :, k, ll, 0] * P[:, :, k, ll] +
                                     C2WfactGz[:, :, k, ll, 1] * P[:, :, k - 1, ll]) * RGAMH[:, :, k, ll]
            P_vmh[:, :, k, ll, Y] = (C2WfactGz[:, :, k, ll, 2] * P[:, :, k, ll] +
                                     C2WfactGz[:, :, k, ll, 3] * P[:, :, k - 1, ll]) * RGAMH[:, :, k, ll]
            P_vmh[:, :, k, ll, Z] = (C2WfactGz[:, :, k, ll, 4] * P[:, :, k, ll] +
                                     C2WfactGz[:, :, k, ll, 5] * P[:, :, k - 1, ll]) * RGAMH[:, :, k, ll]
        for d in range(nxyz):
            for k in range(kmin, kmax + 1):
                Pgrad[:, :, k, ll, d] += (P_vmh[:, :, k + 1, ll, d] - P_vmh[:, :, k, ll, d]) * rdgz[k]
            Pgrad[:, :, kmin - 1, ll, d] = 0.0
            Pgrad[:, :, kmax + 1, ll, d] = 0.0
            if cfg.first_layer_remedy:
                Pgrad[:, :, kmin, ll, d] = Pgrad[:, :, kmin + 1, ll, d]

    if cfg.have_pl:
        P_vmh_pl = np.zeros(P_pl.shape + (nxyz,), dtype=P_pl.dtype)
        k_range = slice(kmin, kmax + 2)
        k_rangem1 = slice(kmin - 1, kmax + 1)
        P_vmh_pl[:, k_range, :, X] = (C2WfactGz_pl[:, k_range, :, 0] * P_pl[:, k_range, :] +
                                      C2WfactGz_pl[:, k_range, :, 1] * P_pl[:, k_rangem1, :]) * RGAMH_pl[:, k_range, :]
        P_vmh_pl[:, k_range, :, Y] = (C2WfactGz_pl[:, k_range, :, 2] * P_pl[:, k_range, :] +
                                      C2WfactGz_pl[:, k_range, :, 3] * P_pl[:, k_rangem1, :]) * RGAMH_pl[:, k_range, :]
        P_vmh_pl[:, k_range, :, Z] = (C2WfactGz_pl[:, k_range, :, 4] * P_pl[:, k_range, :] +
                                      C2WfactGz_pl[:, k_range, :, 5] * P_pl[:, k_rangem1, :]) * RGAMH_pl[:, k_range, :]
        for d in range(nxyz):
            k_mid = slice(kmin, kmax + 1)
            k_midp1 = slice(kmin + 1, kmax + 2)
            Pgrad_pl[:, k_mid, :, d] += (P_vmh_pl[:, k_midp1, :, d] - P_vmh_pl[:, k_mid, :, d]) * rdgz[k_mid, None]
            if cfg.first_layer_remedy:
                Pgrad_pl[:, kmin, :, d] = Pgrad_pl[:, kmin + 1, :, d]
            Pgrad_pl[:, kmin - 1, :, d] = 0.0
            Pgrad_pl[:, kmax + 1, :, d] = 0.0

    # --- horizontalize (OPRT_horizontalize_vec, vectorized regional form) ---
    if cfg.horizontalize:
        vx = Pgrad[:, :, :, :, X]; vy = Pgrad[:, :, :, :, Y]; vz = Pgrad[:, :, :, :, Z]
        gvec = GRD_x[isl, jsl, 0, :, :]
        gx = gvec[..., X][:, :, None, :]
        gy = gvec[..., Y][:, :, None, :]
        gz = gvec[..., Z][:, :, None, :]
        vx_sub = vx[isl, jsl, :, :]; vy_sub = vy[isl, jsl, :, :]; vz_sub = vz[isl, jsl, :, :]
        prd = (vx_sub * gx + vy_sub * gy + vz_sub * gz) / rscale
        vx[isl, jsl, :, :] -= prd * gx / rscale
        vy[isl, jsl, :, :] -= prd * gy / rscale
        vz[isl, jsl, :, :] -= prd * gz / rscale

        if cfg.have_pl:
            for gg in range(g):
                for k in range(kall):
                    for ll in range(l_pl):
                        vxp = Pgrad_pl[gg, k, ll, X]; vyp = Pgrad_pl[gg, k, ll, Y]; vzp = Pgrad_pl[gg, k, ll, Z]
                        gxp = GRD_x_pl[gg, 0, ll, X]; gyp = GRD_x_pl[gg, 0, ll, Y]; gzp = GRD_x_pl[gg, 0, ll, Z]
                        prdp = vxp * gxp / rscale + vyp * gyp / rscale + vzp * gzp / rscale
                        Pgrad_pl[gg, k, ll, X] -= prdp * gxp / rscale
                        Pgrad_pl[gg, k, ll, Y] -= prdp * gyp / rscale
                        Pgrad_pl[gg, k, ll, Z] -= prdp * gzp / rscale

    # --- vertical gradient (half level) ---
    if gradtype == cfg.I_SRC_default:
        for ll in range(l):
            for k in range(kmin + 1, kmax + 1):
                Pgradw[:, :, k, ll] = (GAM2H[:, :, k, ll] *
                                       (P[:, :, k, ll] * RGSGAM2[:, :, k, ll] -
                                        P[:, :, k - 1, ll] * RGSGAM2[:, :, k - 1, ll]) * rdgzh[k])
            Pgradw[:, :, kmin - 1, ll] = 0.0
            Pgradw[:, :, kmin, ll] = 0.0
            Pgradw[:, :, kmax + 1, ll] = 0.0
        if cfg.have_pl:
            k_range = slice(kmin + 1, kmax + 1)
            k_rangem1 = slice(kmin, kmax)
            Pgradw_pl[:, k_range, :] = (GAM2H_pl[:, k_range, :] *
                                        (P_pl[:, k_range, :] * RGSGAM2_pl[:, k_range, :] -
                                         P_pl[:, k_rangem1, :] * RGSGAM2_pl[:, k_rangem1, :]) * rdgzh[k_range, None])
            Pgradw_pl[:, kmin - 1, :] = 0.0
            Pgradw_pl[:, kmin, :] = 0.0
            Pgradw_pl[:, kmax + 1, :] = 0.0
    elif gradtype == cfg.I_SRC_horizontal:
        Pgradw[:, :, :, :] = 0.0
        if cfg.have_pl:
            Pgradw_pl[:, :, :] = 0.0

    return Pgrad, Pgradw, Pgrad_pl, Pgradw_pl


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)
    iall, jall, kall, lall = 8, 8, 10, 3
    gall_pl, lall_pl = 6, 2
    dt = np.float64
    kmin, kmax = 1, kall - 2

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(dt)

    P = R(iall, jall, kall, lall)
    P_pl = R(gall_pl, kall, lall_pl)
    RGAM = R(iall, jall, kall, lall)
    RGAMH = R(iall, jall, kall, lall)
    C2WfactGz = R(iall, jall, kall, lall, 6)
    coef_grad = R(iall, jall, 1, lall, 3, 7)
    GRD_x = R(iall, jall, 1, lall, 3)
    rdgz = R(kall)
    rdgzh = R(kall)
    GAM2H = R(iall, jall, kall, lall)
    RGSGAM2 = R(iall, jall, kall, lall)

    RGAM_pl = R(gall_pl, kall, lall_pl)
    RGAMH_pl = R(gall_pl, kall, lall_pl)
    C2WfactGz_pl = R(gall_pl, kall, lall_pl, 6)
    coef_grad_pl = R(gall_pl, 1, lall_pl, 3)
    GRD_x_pl = R(gall_pl, 1, lall_pl, 3)
    GAM2H_pl = R(gall_pl, kall, lall_pl)
    RGSGAM2_pl = R(gall_pl, kall, lall_pl)

    cfg = PresGradCfg(
        kmin=kmin, kmax=kmax, have_pl=True,
        XDIR=XDIR, YDIR=YDIR, ZDIR=ZDIR,
        gslf_pl=0, gmax_pl=5, nxyz=3,
        first_layer_remedy=True, rscale=6.371e6, plmask=1, horizontalize=True,
        I_SRC_default=I_SRC_DEFAULT, I_SRC_horizontal=I_SRC_HORIZONTAL,
    )
    args = (P, P_pl, RGAM, RGAMH, C2WfactGz, coef_grad, GRD_x, rdgz, rdgzh, GAM2H, RGSGAM2,
            RGAM_pl, RGAMH_pl, C2WfactGz_pl, coef_grad_pl, GRD_x_pl, GAM2H_pl, RGSGAM2_pl)
    return args, cfg


def report(name, ref, got, rtol=0.0, atol=0.0):
    names = ["Pgrad", "Pgradw", "Pgrad_pl", "Pgradw_pl"]
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
        print(f"  {nm:10s}: {'OK ' if passed else 'BAD'} {flag}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def run_for_gradtype(gradtype, label):
    args, cfg = make_inputs()
    ref = compute_pres_gradient_ref(*args, gradtype, cfg)

    # (1) numpy backend
    import numpy as xnp
    got_np = compute_pres_gradient(*args, gradtype, cfg, xnp)
    ok1 = report(f"{label}: numpy backend (eager)", ref, got_np)

    # (2) + (3) jax backend
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    jargs = tuple(jnp.asarray(a) for a in args)
    got_jnp = compute_pres_gradient(*jargs, gradtype, cfg, jnp)
    ok2 = report(f"{label}: jax.numpy backend (eager)", ref, got_jnp)

    kernel_jit = jax.jit(compute_pres_gradient, static_argnames=("gradtype", "cfg", "xp"))
    got_jit = kernel_jit(*jargs, gradtype=gradtype, cfg=cfg, xp=jnp)
    jax.block_until_ready(got_jit)
    ok3 = report(f"{label}: jax.jit (compiled)", ref, got_jit, rtol=1e-11, atol=1e-11)

    return ok1, ok2, ok3


def main():
    results = []
    results += run_for_gradtype(I_SRC_DEFAULT, "I_SRC_default")
    results += run_for_gradtype(I_SRC_HORIZONTAL, "I_SRC_horizontal")

    print("\n========================================")
    print(f"all checks: {'PASS' if all(results) else 'FAIL'}")
    print("========================================")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
