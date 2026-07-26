"""Parity input generator for OPRT_3d_divdamp (3-D divergence-damping operator).

numpy<->jax parity only. The heaviest kernel: horizontal divdamp coefs
(coef_intp (i,j,1,l,3,3,2), coef_diff (i,j,1,l,3,6)) plus the vertical geometry
(C2WfactGz (i,j,k,l,6), RGAMH/RGSQRTH/RGAM (i,j,k,l), rdgz (k,)) and the rhogw
component. Values needn't be physical for a backend-parity check.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.oprt3ddivdamp import (  # noqa: E402,F401
    Oprt3DDivdampCfg,
    compute_oprt3d_divdamp,
)

IALL, JALL, KALL, LALL = 8, 8, 10, 3
GALL_PL, LALL_PL = 7, 2


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    d = dict(
        rhogvx=R(IALL, JALL, KALL, LALL), rhogvy=R(IALL, JALL, KALL, LALL),
        rhogvz=R(IALL, JALL, KALL, LALL), rhogw=R(IALL, JALL, KALL, LALL),
        coef_intp=R(IALL, JALL, 1, LALL, 3, 3, 2),
        coef_diff=R(IALL, JALL, 1, LALL, 3, 6),
        C2WfactGz=R(IALL, JALL, KALL, LALL, 6),
        RGAMH=R(IALL, JALL, KALL, LALL), RGSQRTH=R(IALL, JALL, KALL, LALL), RGAM=R(IALL, JALL, KALL, LALL),
        rdgz=R(KALL),
        pntmask=rng.integers(0, 2, (1, LALL, 2)).astype(np.float64),
        rhogvx_pl=R(GALL_PL, KALL, LALL_PL), rhogvy_pl=R(GALL_PL, KALL, LALL_PL),
        rhogvz_pl=R(GALL_PL, KALL, LALL_PL), rhogw_pl=R(GALL_PL, KALL, LALL_PL),
        coef_intp_pl=R(GALL_PL, 1, LALL_PL, 3, 3),
        coef_diff_pl=R(GALL_PL, 1, LALL_PL, 3),
        C2WfactGz_pl=R(GALL_PL, KALL, LALL_PL, 6),
        RGAMH_pl=R(GALL_PL, KALL, LALL_PL), RGSQRTH_pl=R(GALL_PL, KALL, LALL_PL), RGAM_pl=R(GALL_PL, KALL, LALL_PL),
    )
    cfg = Oprt3DDivdampCfg(have_pl=True, kmin=1, kmax=KALL - 2, gmax=IALL - 2,
                           gslf_pl=0, gmin_pl=1, gmax_pl=GALL_PL - 1, k0=0,
                           TI=0, TJ=1, XDIR=0, YDIR=1, ZDIR=2)
    return d, cfg
