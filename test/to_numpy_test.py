"""bk.to_numpy -- the D2H boundary, and its pinned_host fast path.

Large transfers go device -> pinned_host -> numpy, which is ~10x faster than jax's
default asarray on GH200. `pinned_host` is a GPU memory kind: a CPU device
addresses only `unpinned_host` and raises when asked for it. The jax backend also
runs on CPU, so the request has to degrade rather than propagate -- and because it
was only made above the size threshold, a run would survive gl05/gl06 and die at
gl07, which is a bad way to find out.

Setting the threshold to 0 puts every transfer on the pinned path, so this
exercises it with an 8-element array on whatever device is present: on GPU the
real path, on CPU the fallback.
"""
import numpy as np
import pytest

from pynicamdc.share.mod_backend import Backend


def test_pinned_path_returns_correct_values_on_whatever_device_is_present(monkeypatch):
    pytest.importorskip("jax")
    import jax.numpy as jnp
    monkeypatch.setenv("PYNICAM_PINNED_D2H_MB", "0")   # every transfer takes the path
    bk = Backend(); bk.configure("jax", "float64")

    ref = np.arange(8.0)
    out = bk.to_numpy(jnp.asarray(ref))
    assert np.array_equal(out, ref)

    # whichever branch ran, the decision is cached: taken once, not per transfer
    assert (bk._pin_sh is not None) or (bk._pin_ok is False)


def test_the_fallback_announces_itself(capsys):
    # losing the pinned path costs ~10x D2H on GPU and changes no number, so an A/B
    # cannot catch it -- the notice is the only thing that can
    pytest.importorskip("jax")
    import jax.numpy as jnp
    import os
    os.environ["PYNICAM_PINNED_D2H_MB"] = "0"
    try:
        bk = Backend(); bk.configure("jax", "float64")
        bk.to_numpy(jnp.arange(4.0))
    finally:
        os.environ.pop("PYNICAM_PINNED_D2H_MB", None)
    if bk._pin_ok:
        pytest.skip("device has pinned_host; nothing to announce")
    assert "pinned_host D2H unavailable" in capsys.readouterr().err


def test_pinned_probe_is_not_repeated_after_it_fails(monkeypatch):
    # the fallback must be sticky -- probing jax on every large transfer would put an
    # exception in the hot path of a CPU run
    pytest.importorskip("jax")
    import jax.numpy as jnp
    monkeypatch.setenv("PYNICAM_PINNED_D2H_MB", "0")
    bk = Backend(); bk.configure("jax", "float64")
    bk.to_numpy(jnp.arange(4.0))
    if bk._pin_ok:
        pytest.skip("device has pinned_host; nothing to be sticky about")
    calls = []
    monkeypatch.setattr(bk.jax, "local_devices", lambda: calls.append(1) or [])
    bk.to_numpy(jnp.arange(4.0))
    assert calls == []


def test_the_profile_does_not_claim_a_mode_the_device_cannot_do(monkeypatch):
    # The header names the mode at configure time, before the probe has run. On a
    # device with no pinned memory that would file a report labelled "pinned" over
    # transfers that all went through plain asarray.
    pytest.importorskip("jax")
    import jax.numpy as jnp
    monkeypatch.setenv("PYNICAM_PINNED_D2H_MB", "0")
    monkeypatch.setenv("PYNICAM_PROFILE", "xfer")
    import pynicamdc.share.mod_backend as mb
    monkeypatch.setattr(mb, "_PROFILE", None)      # re-read PYNICAM_PROFILE
    bk = mb.Backend(); bk.configure("jax", "float64")
    bk.to_numpy(jnp.arange(4.0))
    if bk._pin_ok:
        pytest.skip("device has pinned_host; the label was right to begin with")
    assert bk._xfer_prof.mode.startswith("asarray"), bk._xfer_prof.mode


def test_numpy_backend_never_takes_the_pinned_path(monkeypatch):
    monkeypatch.setenv("PYNICAM_PINNED_D2H_MB", "0")
    bk = Backend(); bk.configure("numpy", "float64")
    ref = np.arange(8.0)
    assert np.array_equal(bk.to_numpy(ref), ref)
    assert bk._pin_sh is None
