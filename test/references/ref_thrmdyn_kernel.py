"""Parity input generator for the thrmdyn kernels (compute_thrmdyn_th / _eth).

These are pure elementwise thermodynamic maps (th = tem*(PRE00/pre)**RovCP,
eth = ein + pre/rho). Coverage here is numpy<->jax parity (the kernel run on both
backends must agree to round-off) -- no independent transcription; the math is a
two-line elementwise map. Inputs are random arrays of the model's standard region
shape; only shapes/dtypes matter for a backend-parity check.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.thrmdyn import (  # noqa: E402,F401  (re-exported for the driver)
    ThrmdynCfg,
    compute_thrmdyn_th,
    compute_thrmdyn_eth,
)

SHAPE = (8, 8, 10, 3)   # (iall, jall, kall, lall)


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)

    def R(lo, hi):
        return rng.uniform(lo, hi, SHAPE).astype(np.float64)

    d = dict(
        tem=R(200.0, 320.0),   # temperature [K]
        pre=R(1.0e4, 1.0e5),   # pressure [Pa]  (positive -> safe for /pre and **)
        rho=R(0.3, 1.3),       # density (positive -> safe for /rho)
        ein=R(1.0e5, 3.0e5),   # internal energy
    )
    cfg = ThrmdynCfg(RovCP=287.04 / 1004.6, PRE00=1.0e5)
    return d, cfg
