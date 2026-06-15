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
   matching ``pynicamdc/nhm/dynamics/proto/test_*_kernel.py`` harness. We reuse
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
_PROTO_DIR = os.path.join(_REPO_ROOT, "pynicamdc", "nhm", "dynamics", "proto")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _kernel_module_names():
    return sorted(
        fn[:-3]
        for fn in os.listdir(_KERNELS_DIR)
        if fn.endswith(".py") and fn != "__init__.py"
    )


def _load_proto(name):
    """Import a proto harness by file path under a unique module name.

    The proto harnesses only ``import jax`` inside their ``main()``; their
    module-level reference functions and input generators are numpy-only, so
    loading the module here pulls in no jax dependency.
    """
    path = os.path.join(_PROTO_DIR, name + ".py")
    spec = importlib.util.spec_from_file_location("proto_" + name, path)
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


# (test id, proto module name, driver)
_KERNEL_CASES = [
    ("buoyancy", "test_buoyancy_kernel", _drive_buoyancy),
    ("diag", "test_diag_kernel", _drive_diag),
    ("rhogkin", "test_rhogkin_kernel", _drive_rhogkin),
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
    "proto_name,driver",
    [(p, d) for (_id, p, d) in _KERNEL_CASES],
    ids=[i for (i, _p, _d) in _KERNEL_CASES],
)
def test_kernel_numpy_bitexact(proto_name, driver):
    m = _load_proto(proto_name)
    ref, got = driver(m, np)
    _assert_bitexact(ref, got)


# ---------------------------------------------------------------------------
# 3. jax parity (skipped automatically when jax is not installed, e.g. in CI)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "proto_name,driver",
    [(p, d) for (_id, p, d) in _KERNEL_CASES],
    ids=[i for (i, _p, _d) in _KERNEL_CASES],
)
def test_kernel_jax_matches_numpy(proto_name, driver):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    m = _load_proto(proto_name)
    ref, _ = driver(m, np)          # numpy reference
    _, got = driver(m, jnp)         # jax (eager) result
    # eager jax.numpy reproduces the numpy math to round-off; require tight tol.
    _assert_close(ref, got, rtol=1e-12, atol=1e-12)
