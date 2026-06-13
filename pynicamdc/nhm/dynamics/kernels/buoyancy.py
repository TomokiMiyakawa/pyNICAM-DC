"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for src_buoyancy
(mod_src.py L1044-1088).

This block has NO communication and NO I/O, and reads only VMTR_C2Wfact and
CONST_GRAV, so it is a low-risk kernel to extract next after the diagnostic
block (see kernels/diag.py).

The original in-place body computes the vertical buoyancy term

    buoiw[k] = -grav * ( C2Wfact[k,0] * rhog[k] + C2Wfact[k,1] * rhog[k-1] )

on the interior levels k in [kmin+1, kmax], and sets the boundary rows
(kmin-1, kmin, kmax+1) to zero. The `_pl` pole array is handled the same way
when adm.ADM_have_pl.

Bit-identity note
-----------------
For the standard layout kmin == 1 and kmax == kall-2, the set of explicitly
zeroed boundary rows {kmin-1, kmin, kmax+1} is exactly {0, 1, kall-1}, i.e.
all rows outside the interior. This functional version writes zeros to the
*whole* below-interior block [0..kmin] and above-interior block
[kmax+1..kall-1], which is identical under that layout. (This is the same
boundary-coverage assumption the diag kernel already relies on.)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BuoyCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    kmin: int
    kmax: int
    GRAV: float    # CONST_GRAV (python float -> no jax-scalar promotion)
    have_pl: bool  # adm.ADM_have_pl


def compute_buoyancy(rhog, rhog_pl, C2Wfact, C2Wfact_pl, cfg: BuoyCfg, xp):
    """Pure version of mod_src.py src_buoyancy.

    Parameters
    ----------
    rhog       : (i, j, k, l)        density*JxG                 [IN]
    rhog_pl    : (g, k, l)           pole density*JxG            [IN]
    C2Wfact    : (i, j, k, l, 2)     VMTR_C2Wfact                [IN]
    C2Wfact_pl : (g, k, l, 2)        VMTR_C2Wfact_pl             [IN]
    cfg        : BuoyCfg             static indices/scalars      [IN, static]
    xp         : module             numpy or jax.numpy          [IN, static]

    Returns
    -------
    buoiw    : (i, j, k, l)
    buoiw_pl : (g, k, l)   (placeholder == rhog_pl when not have_pl)
    """
    kmin, kmax, grav = cfg.kmin, cfg.kmax, cfg.GRAV

    # --- regional array ---
    i, j, kall, l = rhog.shape
    interior = -grav * (
        C2Wfact[:, :, kmin + 1:kmax + 1, :, 0] * rhog[:, :, kmin + 1:kmax + 1, :] +
        C2Wfact[:, :, kmin + 1:kmax + 1, :, 1] * rhog[:, :, kmin:kmax,         :]
    )
    lower = xp.zeros((i, j, kmin + 1,          l), dtype=rhog.dtype)  # k 0..kmin
    upper = xp.zeros((i, j, kall - (kmax + 1), l), dtype=rhog.dtype)  # k kmax+1..end
    buoiw = xp.concatenate([lower, interior, upper], axis=2)

    # --- pole array ---
    buoiw_pl = rhog_pl  # placeholder; overwritten below when have_pl
    if cfg.have_pl:
        g, kall_pl, l_pl = rhog_pl.shape
        interior_pl = -grav * (
            C2Wfact_pl[:, kmin + 1:kmax + 1, :, 0] * rhog_pl[:, kmin + 1:kmax + 1, :] +
            C2Wfact_pl[:, kmin + 1:kmax + 1, :, 1] * rhog_pl[:, kmin:kmax,         :]
        )
        lower_pl = xp.zeros((g, kmin + 1,             l_pl), dtype=rhog_pl.dtype)
        upper_pl = xp.zeros((g, kall_pl - (kmax + 1), l_pl), dtype=rhog_pl.dtype)
        buoiw_pl = xp.concatenate([lower_pl, interior_pl, upper_pl], axis=1)

    return buoiw, buoiw_pl
