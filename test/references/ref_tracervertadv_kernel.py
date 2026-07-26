"""Parity input generator for the tracer vertical-advection kernels.

Four pure functions: compute_vert_qh / _pl (build the half-level tracer q_h) and
compute_vert_update / _pl (flux-divergence update). numpy<->jax parity only; each
function is fed independent random inputs of the model's region/pole shapes.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pynicamdc.nhm.dynamics.kernels.tracervertadv import (  # noqa: E402,F401
    TracerVertAdvCfg,
    compute_vert_qh,
    compute_vert_qh_pl,
    compute_vert_update,
    compute_vert_update_pl,
)

IALL, JALL, KALL, LALL = 8, 8, 10, 3
GALL_PL, LALL_PL = 7, 2


def make_inputs(seed=0):
    rng = np.random.default_rng(seed)

    def R(*shape):
        return rng.uniform(0.5, 1.5, shape).astype(np.float64)

    d = dict(
        rhogq_iq=R(IALL, JALL, KALL, LALL),
        rho_den=R(IALL, JALL, KALL, LALL),   # positive -> safe for /rho_den
        flx_v=R(IALL, JALL, KALL, LALL),
        q_h=R(IALL, JALL, KALL, LALL),
        afact=R(KALL),
        bfact=R(KALL),
        rdgz=R(KALL),
        rhogq_iq_pl=R(GALL_PL, KALL, LALL_PL),
        rho_den_pl=R(GALL_PL, KALL, LALL_PL),
        flx_v_pl=R(GALL_PL, KALL, LALL_PL),
        q_h_pl=R(GALL_PL, KALL, LALL_PL),
    )
    cfg = TracerVertAdvCfg(kmin=1, kmax=KALL - 2, have_pl=True)
    return d, cfg
