"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for the scaling block
of src_advection_convergence (mod_src.py L469-522).

This is the COMM-free first half of src_advection_convergence: it forms the
scaled fluxes that are then handed to src_flux_convergence (the OPRT part,
extracted separately). It builds, from the prognostic momentum fluxes and a
scalar field scl:

    rhogv{x,y,z}scl = rhogv{x,y,z} * scl            (full k, plain product)
    rhogwscl        = rhogw * (afact*scl[k] + bfact*scl[k-1])   (half level)

The half-level weighting depends on fluxtype:
    I_SRC_default    : weighted half-level w-flux on k = kmin .. kmax+1,
                       zero on the kmin-1 ghost row.
    I_SRC_horizontal : rhogwscl is entirely zero.

Boundary / coverage note
------------------------
Downstream (src_flux_convergence) reads rhogwscl only on k = kmin+1 .. kmax,
so only that interior range affects results. The original in-place code writes
k = kmin .. kmax+1 (and zeros kmin-1), leaving higher rows as stale buffer
contents that are never read. This functional version instead zeros every row
outside the computed interior, which is identical downstream for the standard
layout kmin == 1. The velocity-scaled fluxes are full-array products and are
fully consumed, so they have no boundary subtlety.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AdvConvCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    kmin: int
    kmax: int
    have_pl: bool
    I_SRC_default: int   # fluxtype sentinel for default (h+v) convergence


def compute_scaled_fluxes(
    rhogvx, rhogvy, rhogvz, rhogw, scl,
    rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl, scl_pl,
    afact, bfact, fluxtype, cfg: AdvConvCfg, xp,
):
    """Pure version of mod_src.py src_advection_convergence L469-522.

    Parameters
    ----------
    rhogvx/y/z : (i, j, k, l)     momentum fluxes               [IN]
    rhogw      : (i, j, k, l)     vertical momentum flux        [IN]
    scl        : (i, j, k, l)     scalar field                  [IN]
    *_pl       : pole counterparts (g, k, l)                    [IN]
    afact,bfact: (kall,)          GRD_afact / GRD_bfact         [IN]
    fluxtype   : int              I_SRC_default / I_SRC_horizontal [IN, static]
    cfg        : AdvConvCfg       static indices/scalars        [IN, static]
    xp         : module           numpy or jax.numpy            [IN, static]

    Returns
    -------
    (vxscl, vyscl, vzscl, wscl, vxscl_pl, vyscl_pl, vzscl_pl, wscl_pl)
    The *_pl outputs are placeholders (== inputs) when not have_pl.
    """
    kmin, kmax = cfg.kmin, cfg.kmax
    kmaxp2 = kmax + 2

    # --- plain velocity-scalar products (full k, fully consumed downstream) ---
    vxscl = rhogvx * scl
    vyscl = rhogvy * scl
    vzscl = rhogvz * scl

    # --- half-level vertical flux ---
    if fluxtype == cfg.I_SRC_default:
        af = afact[kmin:kmaxp2][None, None, :, None]
        bf = bfact[kmin:kmaxp2][None, None, :, None]
        wscl_int = rhogw[:, :, kmin:kmaxp2, :] * (
            af * scl[:, :, kmin:kmaxp2, :] +
            bf * scl[:, :, kmin - 1:kmax + 1, :]
        )
        i, j, kall, l = rhogw.shape
        lower = xp.zeros((i, j, kmin,           l), dtype=rhogw.dtype)  # k 0..kmin-1
        upper = xp.zeros((i, j, kall - kmaxp2,  l), dtype=rhogw.dtype)  # k kmax+2..end
        wscl = xp.concatenate([lower, wscl_int, upper], axis=2)
    else:  # I_SRC_horizontal
        wscl = xp.zeros_like(rhogw)

    # --- pole region ---
    vxscl_pl, vyscl_pl, vzscl_pl, wscl_pl = rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl
    if cfg.have_pl:
        vxscl_pl = rhogvx_pl * scl_pl
        vyscl_pl = rhogvy_pl * scl_pl
        vzscl_pl = rhogvz_pl * scl_pl
        if fluxtype == cfg.I_SRC_default:
            af = afact[kmin:kmaxp2][None, :, None]
            bf = bfact[kmin:kmaxp2][None, :, None]
            wscl_pl_int = rhogw_pl[:, kmin:kmaxp2, :] * (
                af * scl_pl[:, kmin:kmaxp2, :] +
                bf * scl_pl[:, kmin - 1:kmax + 1, :]
            )
            g, kall_pl, l_pl = rhogw_pl.shape
            lower_pl = xp.zeros((g, kmin,            l_pl), dtype=rhogw_pl.dtype)
            upper_pl = xp.zeros((g, kall_pl - kmaxp2, l_pl), dtype=rhogw_pl.dtype)
            wscl_pl = xp.concatenate([lower_pl, wscl_pl_int, upper_pl], axis=1)
        else:
            wscl_pl = xp.zeros_like(rhogw_pl)

    return vxscl, vyscl, vzscl, wscl, vxscl_pl, vyscl_pl, vzscl_pl, wscl_pl
