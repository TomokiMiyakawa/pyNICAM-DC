"""Parity input generator for OPRT_gradient (7-point stencil gradient).

numpy<->jax parity only. Inputs are random arrays of the shapes the kernel
indexes: scl (i,j,k,l), coef_grad (i,j,k,l,3,7) over the 7-neighbour stencil and
3 directions, plus the pole spoke arrays. Values needn't be physical for a
backend-parity check -- only shapes/dtypes matter.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.oprtgradient import (  # noqa: E402,F401
    OprtGradientCfg,
    compute_oprt_gradient,
)

IALL, JALL, KALL, LALL = 8, 8, 10, 3
GALL_PL, LALL_PL = 7, 2


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    d = dict(
        scl=R(IALL, JALL, KALL, LALL),
        coef_grad=R(IALL, JALL, KALL, LALL, 3, 7),   # (i,j,k,l, 3 DIR, 7 stencil)
        scl_pl=R(GALL_PL, KALL, LALL_PL),
        coef_grad_pl=R(GALL_PL, KALL, LALL_PL, 3),
    )
    cfg = OprtGradientCfg(have_pl=True, gslf_pl=0, gmax_pl=GALL_PL - 1,
                          k0=0, XDIR=0, YDIR=1, ZDIR=2)
    return d, cfg
