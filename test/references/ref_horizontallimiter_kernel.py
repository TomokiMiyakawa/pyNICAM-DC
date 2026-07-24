"""Parity input generator for the horizontal Thuburn limiter.

Uses jax-only in-place update (.at) -> jax eager<->jit parity only. The driver
chains qout -> apply (region and pole) so Qin/Qout shapes stay self-consistent.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.horizontallimiter import (  # noqa: E402,F401
    HLimiterCfg,
    HLimiterCfgPl,
    compute_horizontal_limiter_qout,
    compute_horizontal_limiter_apply,
    compute_horizontal_limiter_qout_pl,
    compute_horizontal_limiter_apply_pl,
)

IALL, JALL, KALL, LALL = 8, 8, 10, 3
GALL_PL, LALL_PL = 7, 2


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    d = dict(
        q=R(IALL, JALL, KALL, LALL), d=R(IALL, JALL, KALL, LALL),
        ch=R(IALL, JALL, KALL, LALL, 6), cmask=R(IALL, JALL, KALL, LALL, 6),
        q_a=R(IALL, JALL, KALL, LALL, 6),               # 6 edge arc-points (INOUT)
        q_pl=R(GALL_PL, KALL, LALL_PL), d_pl=R(GALL_PL, KALL, LALL_PL),
        ch_pl=R(GALL_PL, KALL, LALL_PL, 6), cmask_pl=R(GALL_PL, KALL, LALL_PL, 6),
        q_a_pl=R(GALL_PL, KALL, LALL_PL),
    )
    cfg = HLimiterCfg(iall=IALL, jall=JALL, lall=LALL, I_min=0, I_max=1,
                      BIG=1.0e30, EPS=1.0e-16, have_sgp=(False,) * LALL)
    cfg_pl = HLimiterCfgPl(n=0, gmin=1, gmax=5, I_min=0, I_max=1,
                           BIG=1.0e30, EPS=1.0e-16)
    return d, cfg, cfg_pl
