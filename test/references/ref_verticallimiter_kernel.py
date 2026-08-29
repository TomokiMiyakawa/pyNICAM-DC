"""Parity input generator for the vertical Thuburn limiter.

This kernel uses jax-only in-place update (q_h.at[...].set(...)), so it cannot run
on numpy -- coverage is jax eager<->jit parity (the jit-compiled result must match
the eager result to round-off). ck carries the two Courant numbers in its last axis.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.verticallimiter import (  # noqa: E402,F401
    VLimiterCfg,
    compute_vertical_limiter,
)

IALL, JALL, KALL, LALL = 8, 8, 10, 3

# argument order for compute_vertical_limiter(q_h, q, d, ck, cfg, xp)
ARGS = ("q_h", "q", "d", "ck")


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    d = dict(
        q_h=R(IALL, JALL, KALL, LALL),
        q=R(IALL, JALL, KALL, LALL),
        d=R(IALL, JALL, KALL, LALL),
        ck=R(IALL, JALL, KALL, LALL, 2),   # two Courant numbers on the last axis
    )
    cfg = VLimiterCfg(iall=IALL, jall=JALL, kall=KALL, lall=LALL,
                      kmin=1, kmax=KALL - 2, BIG=1.0e30, EPS=1.0e-16)
    return d, cfg
