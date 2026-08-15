"""bk.set_at / bk.add_at -- the backend-agnostic element-assignment abstraction
(refactor plan task S / section 2).

numpy spells `a[idx] = x`; jax (immutable arrays) spells `a.at[idx].set(x)`. These
helpers hide that one difference so a single xp-clean kernel can replace the
numpy/jax TWIN methods (BNDCND_all vs BNDCND_all_resident, the _pl pair, the
numfilter *_resident trio, ...). Contract: always rebind -- `a = bk.set_at(a, idx, val)`
-- because numpy mutates in place and returns `a`, while jax returns a NEW array.
"""
import numpy as np
import pytest

from pynicamdc.share.mod_backend import Backend


def _np():
    bk = Backend()
    bk.configure("numpy", "float64")
    return bk


_CASES = [
    ((1, 2), 99.0),
    ((0, slice(None)), -1.0),
    ((2, slice(1, 3)), 5.0),
]


def test_set_at_numpy_matches_plain_assignment():
    bk = _np()
    a = np.arange(12.0).reshape(3, 4)
    ref = a.copy()
    for idx, val in _CASES:
        ref[idx] = val
    for idx, val in _CASES:
        a = bk.set_at(a, idx, val)
    assert np.array_equal(a, ref)


def test_add_at_numpy_matches_plain_augassign():
    bk = _np()
    a = np.ones((4,))
    ref = a.copy()
    ref[1:3] += 2.0
    a = bk.add_at(a, slice(1, 3), 2.0)
    assert np.array_equal(a, ref)


def test_set_at_numpy_is_in_place():
    bk = _np()
    a = np.zeros(3)
    b = bk.set_at(a, 0, 7.0)
    assert b is a and a[0] == 7.0   # numpy branch: same object, mutated


def test_set_at_jax_matches_numpy():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    bkj = Backend(); bkj.configure("jax", "float64")
    bkn = _np()
    a0 = np.arange(12.0).reshape(3, 4)
    an = a0.copy()
    aj = jnp.asarray(a0)
    for idx, val in _CASES:
        an = bkn.set_at(an, idx, val)
        aj = bkj.set_at(aj, idx, val)
    assert np.array_equal(an, np.asarray(aj))


def test_add_at_jax_matches_numpy():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    bkj = Backend(); bkj.configure("jax", "float64")
    bkn = _np()
    an = np.ones((4,)); aj = jnp.ones((4,))
    an = bkn.add_at(an, slice(1, 3), 2.0)
    aj = bkj.add_at(aj, slice(1, 3), 2.0)
    assert np.array_equal(an, np.asarray(aj))


def test_set_at_jax_is_functional():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    bkj = Backend(); bkj.configure("jax", "float64")
    a = jnp.zeros(3)
    b = bkj.set_at(a, 0, 7.0)
    assert float(b[0]) == 7.0 and float(a[0]) == 0.0   # jax branch: original untouched


# --- dispatch is on the VALUE, not on the configured backend ------------------
# The jax backend's host-staged path (PYNICAM_RESIDENT=0) deliberately carries
# numpy arrays; `a.at[...]` does not exist on those. Which spelling works is a
# property of the value in hand.

def test_numpy_array_under_the_jax_backend_takes_the_numpy_branch():
    pytest.importorskip("jax")
    bkj = Backend(); bkj.configure("jax", "float64")
    a = np.zeros(3)
    b = bkj.set_at(a, 0, 7.0)
    assert b is a and a[0] == 7.0
    b = bkj.add_at(a, 0, 1.0)
    assert b is a and a[0] == 8.0


def test_at_base_copies_a_numpy_array_under_the_jax_backend():
    # set_at would mutate it in place, so the caller's array needs the copy
    pytest.importorskip("jax")
    bkj = Backend(); bkj.configure("jax", "float64")
    a = np.zeros(3)
    out = bkj.at_base(a)
    out = bkj.set_at(out, 0, 7.0)
    assert out is not a and a[0] == 0.0 and out[0] == 7.0


def test_a_tracer_takes_the_jax_branch():
    # Inside a jit/scan trace the value is a tracer, not a concrete array. jax
    # registers tracers as jax.Array instances; pin that, since the whole fused
    # stack traces through set_at.
    pytest.importorskip("jax")
    import jax, jax.numpy as jnp
    bkj = Backend(); bkj.configure("jax", "float64")
    seen = {}

    def f(x):
        seen["jax_value"] = bkj.is_jax_value(x)
        return bkj.set_at(x, 0, 7.0)

    out = jax.jit(f)(jnp.zeros(3))
    assert seen["jax_value"] is True
    assert float(out[0]) == 7.0
