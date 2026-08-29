"""Parity input generator for OPRT_horizontalize_vec.

Projects (vx,vy,vz) onto the local horizontal via the cell geometry GRD_x.
numpy<->jax parity only; inputs are random arrays of the model's region/pole
shapes (GRD_x has a singleton k axis indexed at 0).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.horizontalizevec import (  # noqa: E402,F401
    HorizontalizeVecCfg,
    compute_horizontalize_vec,
)

IALL, JALL, KALL, LALL = 8, 8, 10, 3
GALL_PL, LALL_PL = 7, 2


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    d = dict(
        vx=R(IALL, JALL, KALL, LALL), vy=R(IALL, JALL, KALL, LALL), vz=R(IALL, JALL, KALL, LALL),
        vx_pl=R(GALL_PL, KALL, LALL_PL), vy_pl=R(GALL_PL, KALL, LALL_PL), vz_pl=R(GALL_PL, KALL, LALL_PL),
        GRD_x=R(IALL, JALL, 1, LALL, 3),       # cell geometry (singleton k)
        GRD_x_pl=R(GALL_PL, 1, LALL_PL, 3),
        rscale=1.2,                            # scalar
    )
    cfg = HorizontalizeVecCfg(have_pl=True, XDIR=0, YDIR=1, ZDIR=2)
    return d, cfg
