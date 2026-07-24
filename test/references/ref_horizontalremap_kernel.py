"""Parity input generator for the horizontal MIURA remap reconstruction.

Reconstructs the upwind arc-point tracer value from the cell centre + gradient.
Region and pole use separate cfgs. numpy<->jax parity only; shapes per the
kernel's own docstrings (grd_xc region is (i,j,k,l,3(arc),3(dir))).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.horizontalremap import (  # noqa: E402,F401
    RemapCfg,
    RemapCfgPl,
    compute_horizontal_remap,
    compute_horizontal_remap_pl,
)

IALL, JALL, KALL, LALL = 8, 8, 10, 3
GALL_PL, LALL_PL = 7, 2


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    d = dict(
        q=R(IALL, JALL, KALL, LALL),
        gradq=R(IALL, JALL, KALL, LALL, 3),
        grd_xc=R(IALL, JALL, KALL, LALL, 3, 3),   # (arc, dir)
        cmask=R(IALL, JALL, KALL, LALL, 6),
        grd_x_k0=R(IALL, JALL, LALL, 3),
        q_pl=R(GALL_PL, KALL, LALL_PL),
        gradq_pl=R(GALL_PL, KALL, LALL_PL, 3),
        grd_xc_pl=R(GALL_PL, KALL, LALL_PL, 3),
        cmask_pl=R(GALL_PL, KALL, LALL_PL),
        grd_x_pl_k0=R(GALL_PL, LALL_PL, 3),
    )
    cfg_reg = RemapCfg(AI=0, AIJ=1, AJ=2, XDIR=0, YDIR=1, ZDIR=2)
    cfg_pl = RemapCfgPl(n=0, gmin=1, gmax=5, XDIR=0, YDIR=1, ZDIR=2)
    return d, cfg_reg, cfg_pl
