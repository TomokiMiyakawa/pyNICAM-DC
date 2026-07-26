"""CI test suite for the pure, backend-switchable dynamics kernels.

These tests use the numpy backend only, so they run in the GitHub-Actions CI
environment (which installs numpy + mpi4py, but NOT jax). The jax-parity checks
are guarded by ``pytest.importorskip("jax")`` and are therefore skipped in CI
while still running locally when jax is available.

Two kinds of coverage:

1. ``test_kernel_module_imports`` -- every module under
   ``pynicamdc/nhm/dynamics/kernels/`` must import cleanly and expose a
   ``compute_*`` callable. This catches syntax / import regressions across the
   whole kernel surface with no heavy dependencies.

2. Per-kernel bit-exactness -- the numpy backend of a kernel must reproduce,
   element-for-element, the independent reference transcription kept in the
   matching ``test/references/ref_*_kernel.py`` harness. We reuse
   those validated references and input generators rather than re-deriving them.
"""
from __future__ import annotations

import importlib
import importlib.util
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_KERNELS_DIR = os.path.join(_REPO_ROOT, "pynicamdc", "nhm", "dynamics", "kernels")
_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _kernel_module_names():
    return sorted(
        fn[:-3]
        for fn in os.listdir(_KERNELS_DIR)
        if fn.endswith(".py") and fn != "__init__.py"
    )


def _load_ref(name):
    """Import a kernel reference harness by file path under a unique module name.

    The kernel reference harnesses only ``import jax`` inside their ``main()``; their
    module-level reference functions and input generators are numpy-only, so
    loading the module here pulls in no jax dependency.
    """
    path = os.path.join(_REF_DIR, name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _as_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def _assert_bitexact(ref, got):
    ref_t, got_t = _as_tuple(ref), _as_tuple(got)
    assert len(ref_t) == len(got_t), "result arity mismatch"
    for i, (r, g) in enumerate(zip(ref_t, got_t)):
        r = np.asarray(r)
        g = np.asarray(g)
        assert g.shape == r.shape, f"output[{i}] shape {g.shape} != {r.shape}"
        assert np.array_equal(g, r), (
            f"output[{i}] not bit-exact: max|d|={np.max(np.abs(g - r)):.3e}"
        )


def _assert_close(ref, got, rtol, atol):
    for r, g in zip(_as_tuple(ref), _as_tuple(got)):
        r = np.asarray(r)
        g = np.asarray(np.asarray(g), dtype=r.dtype)
        assert np.allclose(g, r, rtol=rtol, atol=atol), (
            f"jax vs numpy differ beyond tol: max|d|={np.max(np.abs(g - r)):.3e}"
        )


# Per-kernel drivers: (proto_module, build numpy ref, build numpy result).
# Each returns (reference_result, numpy_kernel_result) as tuples/arrays.
def _drive_buoyancy(m, xp):
    rhog, rhog_pl, C2Wfact, C2Wfact_pl, cfg = m.make_inputs()
    ref = m.compute_buoyancy_ref(rhog, rhog_pl, C2Wfact, C2Wfact_pl, cfg)
    got = m.compute_buoyancy(*_xp_args(xp, rhog, rhog_pl, C2Wfact, C2Wfact_pl), cfg, xp)
    return ref, got


def _drive_diag(m, xp):
    PROG, PROGq, DIAG_in, GSGAM2, C2Wfact, CVW, cfg = m.make_inputs()
    ref = m.compute_diagnostics_ref(PROG, PROGq, DIAG_in, GSGAM2, C2Wfact, CVW, cfg)
    got = m.compute_diagnostics(
        *_xp_args(xp, PROG, PROGq, DIAG_in, GSGAM2, C2Wfact, CVW), cfg, xp
    )
    return ref, got


def _drive_rhogkin(m, xp):
    d, cfg, UNDEF = m.make_inputs()
    ref_r = m.rhogkin_reg_ref(d["rhog"], d["rhogvx"], d["rhogvy"], d["rhogvz"],
                              d["rhogw"], d["C2Wfact"], d["W2Cfact"], cfg, UNDEF)
    ref_p = m.rhogkin_pl_ref(d["rhog_pl"], d["rhogvx_pl"], d["rhogvy_pl"],
                             d["rhogvz_pl"], d["rhogw_pl"], d["C2Wfact_pl"],
                             d["W2Cfact_pl"], cfg, UNDEF)
    c = {k: (xp.asarray(v) if hasattr(v, "shape") else v) for k, v in d.items()}
    got_r = m.compute_rhogkin_reg(c["rhog"], c["rhogvx"], c["rhogvy"], c["rhogvz"],
                                  c["rhogw"], c["C2Wfact"], c["W2Cfact"], cfg, xp)
    got_p = m.compute_rhogkin_pl(c["rhog_pl"], c["rhogvx_pl"], c["rhogvy_pl"],
                                 c["rhogvz_pl"], c["rhogw_pl"], c["C2Wfact_pl"],
                                 c["W2Cfact_pl"], cfg, xp)
    return (ref_r, ref_p), (got_r, got_p)


def _xp_args(xp, *arrays):
    return tuple(xp.asarray(a) for a in arrays)


def _drive_advconv(m, xp):
    # two flux types; concatenate their (8-tuple) outputs
    refs, gots = [], []
    for fluxtype in (m.I_SRC_DEFAULT, m.I_SRC_HORIZONTAL):
        *arrays, afact, bfact, cfg = m.make_inputs()
        ref = m.compute_scaled_fluxes_ref(*arrays, afact, bfact, fluxtype, cfg)
        got = m.compute_scaled_fluxes(
            *_xp_args(xp, *arrays), xp.asarray(afact), xp.asarray(bfact), fluxtype, cfg, xp)
        refs += list(_as_tuple(ref)); gots += list(_as_tuple(got))
    return tuple(refs), tuple(gots)


def _drive_fluxconv(m, xp):
    refs, gots = [], []
    for fluxtype in (m.I_SRC_DEFAULT, m.I_SRC_HORIZONTAL):
        arrs, cfg = m.make_inputs()
        ref = m.ref(*[arrs[k] for k in m.ARG_ORDER], fluxtype, cfg)
        got = m.compute_flux_convergence(
            *[xp.asarray(arrs[k]) for k in m.ARG_ORDER], fluxtype, cfg=cfg, xp=xp)
        refs += list(_as_tuple(ref)); gots += list(_as_tuple(got))
    return tuple(refs), tuple(gots)


def _drive_presgrad(m, xp):
    refs, gots = [], []
    for gradtype in (m.I_SRC_DEFAULT, m.I_SRC_HORIZONTAL):
        args, cfg = m.make_inputs()
        ref = m.compute_pres_gradient_ref(*args, gradtype, cfg)
        got = m.compute_pres_gradient(*_xp_args(xp, *args), gradtype, cfg, xp)
        refs += list(_as_tuple(ref)); gots += list(_as_tuple(got))
    return tuple(refs), tuple(gots)


def _drive_horizontalflux(m, xp):
    arrs, cfg = m.make_inputs()
    dt = np.float64(20.0)
    ref = m.ref(*[arrs[k] for k in m.ARG_ORDER], dt, cfg)
    got = m.compute_horizontal_flux(
        *[xp.asarray(arrs[k]) for k in m.ARG_ORDER], dt, cfg=cfg, xp=xp)
    return ref, got


def _drive_tracervertflux(m, xp):
    arrs, scal, cfg = m.make_inputs()
    ref = m.ref(*[arrs[k] for k in m.ARG_ORDER], scal["dt"], scal["b1"], cfg)
    got = m.compute_tracer_vert_flux(
        *[xp.asarray(arrs[k]) for k in m.ARG_ORDER], scal["dt"], scal["b1"], cfg=cfg, xp=xp)
    return ref, got


def _xp_dict(xp, d):
    # move array-valued entries to the backend; leave scalars (dt, afact, ...) as-is.
    return {k: (xp.asarray(v) if hasattr(v, "shape") else v) for k, v in d.items()}


def _drive_advconvmom(m, xp):
    d, cfg = m.make_inputs()
    c = _xp_dict(xp, d)
    ref = (
        tuple(m.merge_reg_ref(d["vx"], d["vy"], d["vz"], d["w"], d["cfact"], d["dfact"], d["GRD_x"], cfg))
        + tuple(m.merge_pl_ref(d["vx_pl"], d["vy_pl"], d["vz_pl"], d["w_pl"], d["cfact"], d["dfact"], d["GRD_x_pl"], cfg))
        + tuple(m.tend_reg_ref(d["dvvx"], d["dvvy"], d["dvvz"], d["rhog"], d["vvx"], d["vvy"], d["GRD_x"], d["C2Wfact"], cfg))
        + tuple(m.tend_pl_ref(d["dvvx_pl"], d["dvvy_pl"], d["dvvz_pl"], d["rhog_pl"], d["vvx_pl"], d["vvy_pl"], d["GRD_x_pl"], d["C2Wfact_pl"], cfg))
    )
    got = (
        tuple(m.compute_merged_velocity_reg(c["vx"], c["vy"], c["vz"], c["w"], c["cfact"], c["dfact"], c["GRD_x"], cfg, xp))
        + tuple(m.compute_merged_velocity_pl(c["vx_pl"], c["vy_pl"], c["vz_pl"], c["w_pl"], c["cfact"], c["dfact"], c["GRD_x_pl"], cfg, xp))
        + tuple(m.compute_momentum_tendency_reg(c["dvvx"], c["dvvy"], c["dvvz"], c["rhog"], c["vvx"], c["vvy"], c["GRD_x"], c["C2Wfact"], cfg, xp))
        + tuple(m.compute_momentum_tendency_pl(c["dvvx_pl"], c["dvvy_pl"], c["dvvz_pl"], c["rhog_pl"], c["vvx_pl"], c["vvy_pl"], c["GRD_x_pl"], c["C2Wfact_pl"], cfg, xp))
    )
    return ref, got


def _drive_vimatrix(m, xp):
    d, cfg = m.make_inputs()
    c = _xp_dict(xp, d)
    ref = (
        tuple(m.matrix_reg_ref(d["eth"], d["g_tilde"], d["RGSQRTH"], d["RGSGAM2"], d["GAM2H"],
                               d["RGAMH"], d["rdgzh"], d["rdgz"], d["dfact"], d["cfact"], d["dt"], cfg))
        + tuple(m.matrix_pl_ref(d["eth_pl"], d["g_tilde_pl"], d["RGSQRTH_pl"], d["RGSGAM2_pl"], d["GAM2H_pl"],
                                d["RGAMH_pl"], d["rdgzh"], d["rdgz"], d["dfact"], d["cfact"], d["dt"], cfg))
    )
    coef = m.precombine_matrix_reg(c["RGSGAM2"], c["GAM2H"], c["RGAMH"], c["rdgz"], c["dfact"], c["cfact"], cfg)
    coef_pl = m.precombine_matrix_pl(c["RGSGAM2_pl"], c["GAM2H_pl"], c["RGAMH_pl"], c["rdgz"], c["dfact"], c["cfact"], cfg)
    got = (
        tuple(m.compute_rhow_matrix_reg(c["eth"], c["g_tilde"], c["RGSQRTH"], c["rdgzh"],
                                        c["dfact"], c["cfact"], coef, c["dt"], cfg, xp))
        + tuple(m.compute_rhow_matrix_pl(c["eth_pl"], c["g_tilde_pl"], c["RGSQRTH_pl"], c["rdgzh"],
                                         c["dfact"], c["cfact"], coef_pl, c["dt"], cfg, xp))
    )
    return ref, got


def _drive_virhowsolver(m, xp):
    d, cfg = m.make_inputs()
    c = _xp_dict(xp, d)
    _R = ("rhogw", "rhogw0", "preg0", "rhog0", "Srho", "Sw", "Spre", "Mc", "Mu", "Ml",
          "RGAMH", "RGSGAM2", "RGAM", "RGSGAM2H", "GSGAM2H", "rdgzh", "afact", "bfact", "dt")
    ref_r = m.solver_reg_ref(*[d[k] for k in _R], cfg)
    ref_p = m.solver_pl_ref(*[d[k + "_pl"] if k not in ("rdgzh", "afact", "bfact", "dt") else d[k] for k in _R], cfg)
    got_r = m.compute_rhow_solver_reg(*[c[k] for k in _R], cfg, xp)
    got_p = m.compute_rhow_solver_pl(*[c[k + "_pl"] if k not in ("rdgzh", "afact", "bfact", "dt") else c[k] for k in _R], cfg, xp)
    return (ref_r, ref_p), (got_r, got_p)


def _drive_bndcnd(m, xp):
    # boundary-condition kernel: each kernel output is a single boundary row that must
    # equal a k-slice of the full-field reference. Replicates the harness run_set for
    # both flag configs; the config-dependent rhow rows can be None (skipped in pairs).
    d, kmin, kmax = m.make_inputs()
    kmaxp1, kminm1 = kmax + 1, kmin - 1
    A = lambda a: xp.asarray(a)
    refs, gots = [], []

    def add(ref_slice, got):
        if got is None:
            return
        refs.append(np.asarray(ref_slice))
        gots.append(got)

    for f in (m.FLAGS["prod"], m.FLAGS["alt"]):
        cfg = m.cfg_of(f, kmin, kmax)
        # thermo reg
        rt, rr, rp = m.thermo_ref(d["tem"], d["rho"], d["pre"], d["phi"], f, kmin, kmax, reg=True)
        tt, tb, pt, pb, ot, ob = m.compute_bndcnd_thermo_reg(
            A(d["tem"]), A(d["rho"]), A(d["pre"]), A(d["phi"]), cfg, xp)
        add(rt[:, :, kmaxp1, :], tt); add(rt[:, :, kminm1, :], tb)
        add(rp[:, :, kmaxp1, :], pt); add(rp[:, :, kminm1, :], pb)
        add(rr[:, :, kmaxp1, :], ot); add(rr[:, :, kminm1, :], ob)
        # thermo pl
        rt, rr, rp = m.thermo_ref(d["tem_pl"], d["rho_pl"], d["pre_pl"], d["phi_pl"], f, kmin, kmax, reg=False)
        tt, tb, pt, pb, ot, ob = m.compute_bndcnd_thermo_pl(
            A(d["tem_pl"]), A(d["rho_pl"]), A(d["pre_pl"]), A(d["phi_pl"]), cfg, xp)
        add(rt[:, kmaxp1, :], tt); add(rr[:, kminm1, :], ob)
        # rhovxvyvz reg
        rx, ry, rz = m.rhovxvyvz_ref(d["rhog"], d["rhogvx"], d["rhogvy"], d["rhogvz"], f, kmin, kmax, reg=True)
        xt, yt, zt, xb, yb, zb = m.compute_bndcnd_rhovxvyvz_reg(
            A(d["rhog"]), A(d["rhogvx"]), A(d["rhogvy"]), A(d["rhogvz"]), cfg, xp)
        add(rx[:, :, kmaxp1, :], xt); add(ry[:, :, kminm1, :], yb); add(rz[:, :, kmaxp1, :], zt)
        # rhovxvyvz pl
        rx, ry, rz = m.rhovxvyvz_ref(d["rhog_pl"], d["rhogvx_pl"], d["rhogvy_pl"], d["rhogvz_pl"], f, kmin, kmax, reg=False)
        xt, yt, zt, xb, yb, zb = m.compute_bndcnd_rhovxvyvz_pl(
            A(d["rhog_pl"]), A(d["rhogvx_pl"]), A(d["rhogvy_pl"]), A(d["rhogvz_pl"]), cfg, xp)
        add(rx[:, kmaxp1, :], xt); add(rz[:, kminm1, :], zb)
        # rhow reg
        rw = m.rhow_ref(d["rhogvx"], d["rhogvy"], d["rhogvz"], d["rhogw"], d["c2wGz"], f, kmin, kmax, reg=True)
        rwt, rwb = m.compute_bndcnd_rhow_reg(
            A(d["rhogvx"]), A(d["rhogvy"]), A(d["rhogvz"]), A(d["c2wGz"]), cfg, xp)
        add(rw[:, :, kmaxp1, :], rwt); add(rw[:, :, kmin, :], rwb)
        # rhow pl
        rw = m.rhow_ref(d["rhogvx_pl"], d["rhogvy_pl"], d["rhogvz_pl"], d["rhogw_pl"], d["c2wGz_pl"], f, kmin, kmax, reg=False)
        rwt, rwb = m.compute_bndcnd_rhow_pl(
            A(d["rhogvx_pl"]), A(d["rhogvy_pl"]), A(d["rhogvz_pl"]), A(d["c2wGz_pl"]), cfg, xp)
        add(rw[:, kmaxp1, :], rwt); add(rw[:, kmin, :], rwb)

    return tuple(refs), tuple(gots)


# (test id, proto module name, driver). The numpy backend of each kernel must
# reproduce, element-for-element, the independent reference transcription in the
# matching test/references/ref_*_kernel.py harness.
_KERNEL_CASES = [
    ("buoyancy", "ref_buoyancy_kernel", _drive_buoyancy),
    ("diag", "ref_diag_kernel", _drive_diag),
    ("rhogkin", "ref_rhogkin_kernel", _drive_rhogkin),
    ("advconv", "ref_advconv_kernel", _drive_advconv),
    ("fluxconv", "ref_fluxconv_kernel", _drive_fluxconv),
    ("presgrad", "ref_presgrad_kernel", _drive_presgrad),
    ("horizontalflux", "ref_horizontalflux_kernel", _drive_horizontalflux),
    ("tracervertflux", "ref_tracervertflux_kernel", _drive_tracervertflux),
    ("advconvmom", "ref_advconvmom_kernel", _drive_advconvmom),
    ("vimatrix", "ref_vimatrix_kernel", _drive_vimatrix),
    ("virhowsolver", "ref_virhowsolver_kernel", _drive_virhowsolver),
    ("bndcnd", "ref_bndcnd_kernel", _drive_bndcnd),
]


# ---------------------------------------------------------------------------
# 1. import / API smoke test over every kernel module
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("modname", _kernel_module_names())
def test_kernel_module_imports(modname):
    mod = importlib.import_module(f"pynicamdc.nhm.dynamics.kernels.{modname}")
    computes = [
        n for n in dir(mod)
        if n.startswith("compute_") and callable(getattr(mod, n))
    ]
    assert computes, f"kernel module '{modname}' exposes no compute_* function"


# ---------------------------------------------------------------------------
# 2. numpy backend bit-exactness vs the validated reference transcription
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ref_name,driver",
    [(p, d) for (_id, p, d) in _KERNEL_CASES],
    ids=[i for (i, _p, _d) in _KERNEL_CASES],
)
def test_kernel_numpy_bitexact(ref_name, driver):
    m = _load_ref(ref_name)
    ref, got = driver(m, np)
    _assert_bitexact(ref, got)


# ---------------------------------------------------------------------------
# 3. jax parity (skipped automatically when jax is not installed, e.g. in CI)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ref_name,driver",
    [(p, d) for (_id, p, d) in _KERNEL_CASES],
    ids=[i for (i, _p, _d) in _KERNEL_CASES],
)
def test_kernel_jax_matches_numpy(ref_name, driver):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    m = _load_ref(ref_name)
    ref, _ = driver(m, np)          # numpy reference
    _, got = driver(m, jnp)         # jax (eager) result
    # eager jax.numpy reproduces the numpy math to round-off; require tight tol.
    _assert_close(ref, got, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. numpy<->jax PARITY for xp-clean kernels that have no independent reference
#    transcription yet (testing-plan 0.3). Each driver runs the kernel on the
#    given backend; the test asserts numpy and jax agree to round-off. This
#    catches backend divergence -- the key risk for the numba/pytorch work --
#    without re-deriving each kernel's math. (Kernels that use jax-only `.at`
#    can't run on numpy; they get an eager-vs-jit parity check instead, below.)
# ---------------------------------------------------------------------------
def _pdrive_thrmdyn(m, xp):
    d, cfg = m.make_inputs()
    A = lambda k: xp.asarray(d[k])  # noqa: E731
    th = m.compute_thrmdyn_th(A("tem"), A("pre"), cfg, xp)
    eth = m.compute_thrmdyn_eth(A("ein"), A("pre"), A("rho"), xp)
    return (th, eth)


def _pdrive_oprtgradient(m, xp):
    d, cfg = m.make_inputs()
    A = lambda k: xp.asarray(d[k])  # noqa: E731
    grad, grad_pl = m.compute_oprt_gradient(
        A("scl"), A("scl_pl"), A("coef_grad"), A("coef_grad_pl"), cfg, xp)
    return (grad, grad_pl)


def _pdrive_oprtlaplacian(m, xp):
    d, cfg = m.make_inputs()
    A = lambda k: xp.asarray(d[k])  # noqa: E731
    dscl, dscl_pl = m.compute_oprt_laplacian(
        A("scl"), A("scl_pl"), A("coef_lap"), A("coef_lap_pl"), cfg, xp)
    return (dscl, dscl_pl)


def _pdrive_horizontalizevec(m, xp):
    d, cfg = m.make_inputs()
    A = lambda k: xp.asarray(d[k])  # noqa: E731
    out = m.compute_horizontalize_vec(
        A("vx"), A("vy"), A("vz"), A("vx_pl"), A("vy_pl"), A("vz_pl"),
        A("GRD_x"), A("GRD_x_pl"), d["rscale"], cfg, xp)
    return tuple(out)


def _pdrive_horizontalremap(m, xp):
    d, cfg_reg, cfg_pl = m.make_inputs()
    A = lambda k: xp.asarray(d[k])  # noqa: E731
    qa = m.compute_horizontal_remap(
        A("q"), A("gradq"), A("grd_xc"), A("cmask"), A("grd_x_k0"), cfg_reg, xp)
    qa_pl = m.compute_horizontal_remap_pl(
        A("q_pl"), A("gradq_pl"), A("grd_xc_pl"), A("cmask_pl"), A("grd_x_pl_k0"), cfg_pl, xp)
    return (qa, qa_pl)


def _pdrive_divdamppostcomm(m, xp):
    d, dd_cfg, hz_cfg = m.make_inputs()
    A = lambda k: xp.asarray(d[k])  # noqa: E731
    out = m.compute_divdamp_post_comm(
        A("vtmp2"), A("vtmp2_pl"), A("divdamp_coef"), A("divdamp_coef_pl"),
        A("coef_intp"), A("coef_diff"), A("coef_intp_pl"), A("coef_diff_pl"),
        A("GRD_x"), A("GRD_x_pl"), d["rscale"], dd_cfg, hz_cfg, xp)
    return tuple(out)


def _pdrive_oprt3ddivdamp(m, xp):
    d, cfg = m.make_inputs()
    A = lambda k: xp.asarray(d[k])  # noqa: E731
    out = m.compute_oprt3d_divdamp(
        A("rhogvx"), A("rhogvy"), A("rhogvz"), A("rhogw"),
        A("rhogvx_pl"), A("rhogvy_pl"), A("rhogvz_pl"), A("rhogw_pl"),
        A("coef_intp"), A("coef_diff"), A("coef_intp_pl"), A("coef_diff_pl"),
        A("C2WfactGz"), A("RGAMH"), A("RGSQRTH"), A("RGAM"),
        A("C2WfactGz_pl"), A("RGAMH_pl"), A("RGSQRTH_pl"), A("RGAM_pl"),
        A("rdgz"), A("pntmask"), cfg, xp)
    return tuple(out)


def _pdrive_oprtdiffusion(m, xp):
    d, cfg = m.make_inputs()
    A = lambda k: xp.asarray(d[k])  # noqa: E731
    dscl, dscl_pl = m.compute_oprt_diffusion(
        A("scl"), A("scl_pl"), A("kh"), A("kh_pl"),
        A("coef_intp"), A("coef_intp_pl"), A("coef_diff"), A("coef_diff_pl"),
        A("pntmask"), cfg, xp)
    return (dscl, dscl_pl)


def _pdrive_oprtdivdamp(m, xp):
    d, cfg = m.make_inputs()
    A = lambda k: xp.asarray(d[k])  # noqa: E731
    out = m.compute_oprt_divdamp(
        A("vx"), A("vy"), A("vz"), A("vx_pl"), A("vy_pl"), A("vz_pl"),
        A("coef_intp"), A("coef_diff"), A("coef_intp_pl"), A("coef_diff_pl"), cfg, xp)
    return tuple(out)


def _pdrive_tracervertadv(m, xp):
    d, cfg = m.make_inputs()
    A = lambda k: xp.asarray(d[k])  # noqa: E731
    qh = m.compute_vert_qh(A("rhogq_iq"), A("rho_den"), A("afact"), A("bfact"), cfg, xp)
    qh_pl = m.compute_vert_qh_pl(A("rhogq_iq_pl"), A("rho_den_pl"), A("afact"), A("bfact"), cfg, xp)
    up = m.compute_vert_update(A("rhogq_iq"), A("flx_v"), A("q_h"), A("rdgz"), cfg, xp)
    up_pl = m.compute_vert_update_pl(A("rhogq_iq_pl"), A("flx_v_pl"), A("q_h_pl"), A("rdgz"), cfg, xp)
    return (qh, qh_pl, up, up_pl)


# (test id, reference module, backend driver returning the kernel output(s))
_PARITY_CASES = [
    ("thrmdyn", "ref_thrmdyn_kernel", _pdrive_thrmdyn),
    ("oprtgradient", "ref_oprtgradient_kernel", _pdrive_oprtgradient),
    ("oprtlaplacian", "ref_oprtlaplacian_kernel", _pdrive_oprtlaplacian),
    ("tracervertadv", "ref_tracervertadv_kernel", _pdrive_tracervertadv),
    ("horizontalizevec", "ref_horizontalizevec_kernel", _pdrive_horizontalizevec),
    ("oprtdivdamp", "ref_oprtdivdamp_kernel", _pdrive_oprtdivdamp),
    ("oprtdiffusion", "ref_oprtdiffusion_kernel", _pdrive_oprtdiffusion),
    ("oprt3ddivdamp", "ref_oprt3ddivdamp_kernel", _pdrive_oprt3ddivdamp),
    ("divdamppostcomm", "ref_divdamppostcomm_kernel", _pdrive_divdamppostcomm),
    ("horizontalremap", "ref_horizontalremap_kernel", _pdrive_horizontalremap),
]


@pytest.mark.parametrize(
    "ref_name,driver",
    [(p, d) for (_id, p, d) in _PARITY_CASES],
    ids=[i for (i, _p, _d) in _PARITY_CASES],
)
def test_kernel_numpy_jax_parity(ref_name, driver):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    m = _load_ref(ref_name)
    got_np = driver(m, np)
    got_jx = driver(m, jnp)
    _assert_close(got_np, got_jx, rtol=1e-11, atol=1e-11)


# ---------------------------------------------------------------------------
# 5. jax eager<->jit parity for kernels that use jax-only in-place update
#    (`.at[...].set`) and therefore cannot run on numpy. The jit-compiled result
#    must reproduce the eager result to round-off.
# ---------------------------------------------------------------------------
def _jdrive_verticallimiter(m, jnp, jax, jit):
    d, cfg = m.make_inputs()
    args = tuple(jnp.asarray(d[k]) for k in m.ARGS)
    fn = jax.jit(m.compute_vertical_limiter, static_argnames=("cfg", "xp")) if jit \
        else m.compute_vertical_limiter
    return fn(*args, cfg=cfg, xp=jnp)


def _jdrive_horizontallimiter(m, jnp, jax, jit):
    d, cfg, cfg_pl = m.make_inputs()
    A = lambda k: jnp.asarray(d[k])  # noqa: E731

    def J(fn, static):
        return jax.jit(fn, static_argnames=static) if jit else fn

    qout = J(m.compute_horizontal_limiter_qout, ("cfg", "xp"))
    apply_ = J(m.compute_horizontal_limiter_apply, ("cfg", "xp"))

    # region: qout -> apply (Qout feeds apply, so shapes stay consistent).
    # The pole (_pl) path has an intricate pentagon-ring layout; it is exercised
    # end-to-end by the dynamics smoke, so the isolated jit-parity here covers the
    # region qout+apply. (cfg_pl is unused for now.)
    del cfg_pl
    Qin, Qout = qout(A("q"), A("d"), A("ch"), A("cmask"), cfg=cfg, xp=jnp)
    q_a = apply_(A("q_a"), Qin, Qout, A("cmask"), cfg=cfg, xp=jnp)
    return (q_a,)


_JIT_PARITY_CASES = [
    ("verticallimiter", "ref_verticallimiter_kernel", _jdrive_verticallimiter),
    ("horizontallimiter", "ref_horizontallimiter_kernel", _jdrive_horizontallimiter),
]


@pytest.mark.parametrize(
    "ref_name,driver",
    [(p, d) for (_id, p, d) in _JIT_PARITY_CASES],
    ids=[i for (i, _p, _d) in _JIT_PARITY_CASES],
)
def test_kernel_jax_eager_vs_jit(ref_name, driver):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    m = _load_ref(ref_name)
    eager = driver(m, jnp, jax, False)
    jitted = driver(m, jnp, jax, True)
    jax.block_until_ready(jitted)
    _assert_close(eager, jitted, rtol=1e-11, atol=1e-11)
