"""Parity input generator for the divdamp post-COMM glue kernel.

compute_divdamp_post_comm negates the post-halo vtmp2, runs OPRT_divdamp, scales
by divdamp_coef, and horizontalizes -- so its inputs are the union of the
OPRT_divdamp coefs and the horizontalize geometry. numpy<->jax parity only.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.divdamppostcomm import compute_divdamp_post_comm  # noqa: E402,F401
from pynicamdc.nhm.dynamics.kernels.oprtdivdamp import OprtDivdampCfg  # noqa: E402
from pynicamdc.nhm.dynamics.kernels.horizontalizevec import HorizontalizeVecCfg  # noqa: E402

IALL, JALL, KALL, LALL = 8, 8, 10, 3
GALL_PL, LALL_PL = 7, 2


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    d = dict(
        vtmp2=R(IALL, JALL, KALL, LALL, 3),
        vtmp2_pl=R(GALL_PL, KALL, LALL_PL, 3),
        divdamp_coef=R(IALL, JALL, KALL, LALL),
        divdamp_coef_pl=R(GALL_PL, KALL, LALL_PL),
        coef_intp=R(IALL, JALL, KALL, LALL, 3, 3, 2),
        coef_diff=R(IALL, JALL, KALL, LALL, 3, 6),
        coef_intp_pl=R(GALL_PL, 1, LALL_PL, 3, 3),
        coef_diff_pl=R(GALL_PL, 1, LALL_PL, 3),
        GRD_x=R(IALL, JALL, 1, LALL, 3),
        GRD_x_pl=R(GALL_PL, 1, LALL_PL, 3),
        rscale=1.2,
    )
    dd_cfg = OprtDivdampCfg(have_pl=True, gmax=IALL - 2, gslf_pl=0, gmin_pl=1,
                            gmax_pl=GALL_PL - 1, k0=0, TI=0, TJ=1, XDIR=0, YDIR=1, ZDIR=2)
    hz_cfg = HorizontalizeVecCfg(have_pl=True, XDIR=0, YDIR=1, ZDIR=2)
    return d, dd_cfg, hz_cfg


def make_cfgs(seed=0):
    _, dd_cfg, hz_cfg = make_inputs(seed)
    return dd_cfg, hz_cfg
