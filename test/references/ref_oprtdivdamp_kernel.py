"""Parity input generator for OPRT_divdamp (2-D divergence-damping operator).

numpy<->jax parity only. coef_intp (i,j,k,l,3,3,2) over 3 dirs x 3 points x TI/TJ;
coef_diff (i,j,k,l,3,6); plus the pole spoke coefficients. Values needn't be
physical for a backend-parity check -- only shapes/dtypes matter.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.oprtdivdamp import (  # noqa: E402,F401
    OprtDivdampCfg,
    compute_oprt_divdamp,
)

IALL, JALL, KALL, LALL = 8, 8, 10, 3
GALL_PL, LALL_PL = 7, 2
GMAX = IALL - 2   # 6: region sub-block max (Rp1 = 1..gmax+1 must fit in iall)


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    d = dict(
        vx=R(IALL, JALL, KALL, LALL), vy=R(IALL, JALL, KALL, LALL), vz=R(IALL, JALL, KALL, LALL),
        coef_intp=R(IALL, JALL, KALL, LALL, 3, 3, 2),
        coef_diff=R(IALL, JALL, KALL, LALL, 3, 6),
        vx_pl=R(GALL_PL, KALL, LALL_PL), vy_pl=R(GALL_PL, KALL, LALL_PL), vz_pl=R(GALL_PL, KALL, LALL_PL),
        coef_intp_pl=R(GALL_PL, 1, LALL_PL, 3, 3),
        coef_diff_pl=R(GALL_PL, 1, LALL_PL, 3),
    )
    cfg = OprtDivdampCfg(have_pl=True, gmax=GMAX, gslf_pl=0, gmin_pl=1, gmax_pl=GALL_PL - 1,
                         k0=0, TI=0, TJ=1, XDIR=0, YDIR=1, ZDIR=2)
    return d, cfg
