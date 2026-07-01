"""
Pure / backend-switchable (numpy <-> jax.numpy) kernels for the thermodynamic
diagnostics THRMDYN_th and THRMDYN_eth (host originals in
nhm/share/mod_thrmdyn.py THRMDYN_th / THRMDYN_eth).

Both are pure elementwise maps with NO communication, NO boundary special-cases
and NO I/O -- the lowest-risk shape to port, and the last unported link in the
Pre_Post chain (diag -> BNDCND -> THRMDYN), both of which are already kernels
(kernels/diag.py, kernels/bndcnd.py). Wiring these lets the whole Pre_Post chain
stay device-resident (drop the per-kernel asarray/to_numpy round-trips).

Math (verbatim from the host, same op order):
    th  = tem * (PRE00 / pre) ** RovCP        RovCP = Rdry / CPdry
    eth = ein + pre / rho

Backend notes:
  * numpy backend  -> bit-exact vs the host (identical scalar ops).
  * jax backend    -> `eth` (only +, /) is bit-exact; `th` uses a non-integer
    power (transcendental), so it is machine-precision vs numpy (different libm
    pow), which is the expected/allowed tolerance for GPU compute (cmp_prec).
The host builds th/eth as np.full_like(..., UNDEF) then overwrites every element,
so the returned full arrays are equivalent (no UNDEF survives).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ThrmdynCfg:
    """Static (hashable) scalars: safe to mark static under jax.jit."""
    RovCP: float   # CONST_Rdry / CONST_CPdry
    PRE00: float   # CONST_PRE00


def compute_thrmdyn_th(tem, pre, cfg: ThrmdynCfg, xp):
    """Potential temperature. Pure version of THRMDYN_th.

    Parameters
    ----------
    tem : array   temperature           [IN]
    pre : array   pressure (same shape)  [IN]
    cfg : ThrmdynCfg
    xp  : module  (numpy or jax.numpy)

    Returns
    -------
    th : array    same shape as tem
    """
    # ratio = PRE00 / pre ; ratio **= RovCP ; th = tem * ratio  (host op order)
    ratio = cfg.PRE00 / pre
    ratio = ratio ** cfg.RovCP
    return tem * ratio


def compute_thrmdyn_eth(ein, pre, rho, xp):
    """Enthalpy-like quantity. Pure version of THRMDYN_eth.

    Parameters
    ----------
    ein : array   internal energy        [IN]
    pre : array   pressure               [IN]
    rho : array   density (same shape)    [IN]
    xp  : module  (numpy or jax.numpy)

    Returns
    -------
    eth : array   same shape as ein
    """
    # eth = pre / rho ; eth = ein + eth  (host op order)
    return ein + pre / rho
