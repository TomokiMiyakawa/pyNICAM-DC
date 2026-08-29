"""Parity input generator for the vi path-2 split/mean update.

The only vertical-implicit-chain kernel that is unit-testable in isolation: it is
COMM-free and has NO solver/division (a fused PROG_split writeback + PROG_mean
accumulation), so synthetic inputs stay finite. (vimain and vipath1 run the full
VI step incl. a tridiagonal solve over ~40-70 arrays -- not synthesizable; they
are covered end-to-end by the dynamics smoke.) numpy<->jax parity + golden.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.vipath2 import (  # noqa: E402,F401
    ViPath2Cfg,
    compute_vi_path2_update,
)

IALL, JALL, KALL, LALL = 8, 8, 10, 3
GALL_PL, LALL_PL = 7, 2
NM = 6   # prognostic components: RHOG, RHOGVX, RHOGVY, RHOGVZ, RHOGW, RHOGE


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    P = dict(
        diff_vh=R(IALL, JALL, KALL, LALL, 3), diff_we=R(IALL, JALL, KALL, LALL, 3),
        PROG_mean=R(IALL, JALL, KALL, LALL, NM),
        diff_vh_pl=R(GALL_PL, KALL, LALL_PL, 3), diff_we_pl=R(GALL_PL, KALL, LALL_PL, 3),
        PROG_mean_pl=R(GALL_PL, KALL, LALL_PL, NM),
        rweight_itr=0.5,   # scalar
    )
    cfg = ViPath2Cfg(have_pl=True, I_RHOG=0, I_RHOGVX=1, I_RHOGVY=2,
                     I_RHOGVZ=3, I_RHOGW=4, I_RHOGE=5)
    return P, cfg
