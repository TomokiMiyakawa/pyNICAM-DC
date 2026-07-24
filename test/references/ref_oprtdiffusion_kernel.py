"""Parity input generator for OPRT_diffusion (kh-weighted diffusion operator).

numpy<->jax parity only. Heaviest coef layout: coef_intp (i,j,k,l,3,3,2),
coef_diff (i,j,k,l,3,6), a point-mask, and the pole spoke coefficients. Values
needn't be physical for a backend-parity check -- only shapes/dtypes matter.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.oprtdiffusion import (  # noqa: E402,F401
    OprtDiffusionCfg,
    compute_oprt_diffusion,
)

IALL, JALL, KALL, LALL = 8, 8, 10, 3
GALL_PL, LALL_PL = 7, 2


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    d = dict(
        scl=R(IALL, JALL, KALL, LALL),
        kh=R(IALL, JALL, KALL, LALL),
        coef_intp=R(IALL, JALL, KALL, LALL, 3, 3, 2),
        coef_diff=R(IALL, JALL, KALL, LALL, 3, 6),
        pntmask=rng.integers(0, 2, (1, LALL, 2)).astype(np.float64),
        scl_pl=R(GALL_PL, KALL, LALL_PL),
        kh_pl=R(GALL_PL, KALL, LALL_PL),
        coef_intp_pl=R(GALL_PL, 1, LALL_PL, 3, 3),
        coef_diff_pl=R(GALL_PL, 1, LALL_PL, 3),
    )
    cfg = OprtDiffusionCfg(have_pl=True, gmin=1, gmax=IALL - 2, nxyz=3,
                           gslf_pl=0, gmin_pl=1, gmax_pl=GALL_PL - 1,
                           k0=0, TI=0, TJ=1)
    return d, cfg
