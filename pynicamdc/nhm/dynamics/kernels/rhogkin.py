"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for cnvvar_rhogkin
(mod_cnvvar.py L152-248).

Computes the total kinetic-energy density rhogkin from the prognostic density
rhog, the horizontal momenta (rhogvx, rhogvy, rhogvz) and the vertical momentum
rhogw. Fully data-parallel and COMM-free: every output level depends only on
local inputs at k and k-1 / k+1, so the whole routine collapses to broadcast
expressions with no k-recurrence.

The vertical KE rhogkin_v is defined only on the interior w-levels
k = kmin+1 .. kmax (its kmin and kmax+1 boundary rows are 0). The total combines
the cell-centered horizontal KE with the half-level vertical KE shifted up and
down by one row; we materialize a small "v_block" spanning rows kmin..kmax+1
(zero boundaries) and take v_block[1:] / v_block[:-1] for the up / down shifts.

Returned rhogkin layout (matching the original np.full_like(UNDEF) + writes):
    k in [0, kmin-2]        : UNDEF   (empty in the standard kmin=1 layout)
    k = kmin-1              : 0
    k in [kmin, kmax]       : rhogkin_h + W2Cfact0*v_up + W2Cfact1*v_dn
    k = kmax+1              : 0
    k in [kmax+2, kall-1]   : UNDEF   (empty in the standard kmax=kall-2 layout)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RhogkinCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    kmin: int
    kmax: int
    have_pl: bool
    UNDEF: float


def compute_rhogkin_reg(rhog, rhogvx, rhogvy, rhogvz, rhogw,
                        VMTR_C2Wfact, VMTR_W2Cfact, cfg, xp):
    """Regional total KE. Returns full rhogkin (i,j,kall,l)."""
    kmin, kmax = cfg.kmin, cfg.kmax
    kc  = slice(kmin,     kmax + 1)   # cell rows k = kmin .. kmax
    kvi = slice(kmin + 1, kmax + 1)   # interior w-levels k = kmin+1 .. kmax
    kcm = slice(kmin,     kmax)       # k-1 partner for the half-level denom

    # --- horizontal KE on rows kmin..kmax ---
    kh = 0.5 * (rhogvx[:, :, kc, :] ** 2
                + rhogvy[:, :, kc, :] ** 2
                + rhogvz[:, :, kc, :] ** 2) / rhog[:, :, kc, :]

    # --- vertical KE on interior w-levels kmin+1..kmax ---
    denom = (VMTR_C2Wfact[:, :, kvi, :, 0] * rhog[:, :, kvi, :]
             + VMTR_C2Wfact[:, :, kvi, :, 1] * rhog[:, :, kcm, :])
    v_int = 0.5 * rhogw[:, :, kvi, :] ** 2 / denom

    # v_block spans rows kmin..kmax+1 with zero boundaries at kmin and kmax+1
    z = xp.zeros_like(v_int[:, :, :1, :])
    v_block = xp.concatenate([z, v_int, z], axis=2)
    v_up = v_block[:, :, 1:, :]    # rhogkin_v[kmin+1 .. kmax+1]
    v_dn = v_block[:, :, :-1, :]   # rhogkin_v[kmin   .. kmax]

    total_int = (kh
                 + VMTR_W2Cfact[:, :, kc, :, 0] * v_up
                 + VMTR_W2Cfact[:, :, kc, :, 1] * v_dn)

    # --- assemble the full kall-length column ---
    z_row = xp.zeros_like(total_int[:, :, :1, :])   # boundary 0 rows
    i, j, kall, l = rhog.shape
    n_pre  = kmin - 1
    n_post = kall - kmax - 2
    parts = []
    if n_pre > 0:
        parts.append(xp.full((i, j, n_pre, l), cfg.UNDEF, dtype=rhog.dtype))
    parts.append(z_row)        # k = kmin-1
    parts.append(total_int)    # k = kmin .. kmax
    parts.append(z_row)        # k = kmax+1
    if n_post > 0:
        parts.append(xp.full((i, j, n_post, l), cfg.UNDEF, dtype=rhog.dtype))
    return xp.concatenate(parts, axis=2)


def compute_rhogkin_pl(rhog_pl, rhogvx_pl, rhogvy_pl, rhogvz_pl, rhogw_pl,
                       VMTR_C2Wfact_pl, VMTR_W2Cfact_pl, cfg, xp):
    """Pole total KE. Returns full rhogkin_pl (g,kall,l)."""
    kmin, kmax = cfg.kmin, cfg.kmax
    kc  = slice(kmin,     kmax + 1)
    kvi = slice(kmin + 1, kmax + 1)
    kcm = slice(kmin,     kmax)

    kh = 0.5 * (rhogvx_pl[:, kc, :] ** 2
                + rhogvy_pl[:, kc, :] ** 2
                + rhogvz_pl[:, kc, :] ** 2) / rhog_pl[:, kc, :]

    denom = (VMTR_C2Wfact_pl[:, kvi, :, 0] * rhog_pl[:, kvi, :]
             + VMTR_C2Wfact_pl[:, kvi, :, 1] * rhog_pl[:, kcm, :])
    v_int = 0.5 * rhogw_pl[:, kvi, :] ** 2 / denom

    z = xp.zeros_like(v_int[:, :1, :])
    v_block = xp.concatenate([z, v_int, z], axis=1)
    v_up = v_block[:, 1:, :]
    v_dn = v_block[:, :-1, :]

    total_int = (kh
                 + VMTR_W2Cfact_pl[:, kc, :, 0] * v_up
                 + VMTR_W2Cfact_pl[:, kc, :, 1] * v_dn)

    z_row = xp.zeros_like(total_int[:, :1, :])
    g, kall, l = rhog_pl.shape
    n_pre  = kmin - 1
    n_post = kall - kmax - 2
    parts = []
    if n_pre > 0:
        parts.append(xp.full((g, n_pre, l), cfg.UNDEF, dtype=rhog_pl.dtype))
    parts.append(z_row)
    parts.append(total_int)
    parts.append(z_row)
    if n_post > 0:
        parts.append(xp.full((g, n_post, l), cfg.UNDEF, dtype=rhog_pl.dtype))
    return xp.concatenate(parts, axis=1)
