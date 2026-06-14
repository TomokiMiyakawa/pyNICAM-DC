"""
Pure / backend-switchable (numpy <-> jax.numpy) kernels for the COMM-free
boundary-condition routines of mod_bndcnd.py (Bndc.BNDCND_thermo /
BNDCND_rhovxvyvz / BNDCND_rhow and their pole _pl variants).

Each routine only updates the two ghost rows (k = kmin-1 and k = kmax+1) plus,
for the vertical momentum, the half-levels k = kmin and k = kmax+1. They are
fully data-parallel: every boundary value is an explicit algebraic function of
interior rows, with no k-recurrence. The original code mutates the arrays
in place; here we instead compute and *return* the new boundary slabs (shape
(i,j,l) for regional, (g,l) for pole) and let the caller write them back, which
keeps the kernels purely functional and jit-friendly.

The boundary *type* flags (temperature TEM/EPL, momentum RIGID/FREE at top and
bottom) are static and live in BndCfg, so the Python branches below are resolved
once per cfg under jax.jit.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BndCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    kmin: int
    kmax: int
    have_pl: bool
    is_top_tem: bool
    is_top_epl: bool
    is_btm_tem: bool
    is_btm_epl: bool
    is_top_rigid: bool
    is_top_free: bool
    is_btm_rigid: bool
    is_btm_free: bool
    GRAV: float
    Rdry: float


def _lag_intpl(z, z1, p1, z2, p2, z3, p3):
    """Quadratic Lagrange interpolation/extrapolation to height z."""
    return (
        ((z - z2) * (z - z3)) / ((z1 - z2) * (z1 - z3)) * p1 +
        ((z - z1) * (z - z3)) / ((z2 - z1) * (z2 - z3)) * p2 +
        ((z - z1) * (z - z2)) / ((z3 - z1) * (z3 - z2)) * p3
    )


# ---------------------------------------------------------------------------
# Thermodynamics: tem / pre / rho at the two ghost rows.
# Returns (tem_top, tem_btm, pre_top, pre_btm, rho_top, rho_btm).
# ---------------------------------------------------------------------------
def compute_bndcnd_thermo_reg(tem, rho, pre, phi, cfg, xp):
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1, kminp1, kminp2 = kmin - 1, kmin + 1, kmin + 2
    kmaxm1, kmaxm2, kmaxp1 = kmax - 1, kmax - 2, kmax + 1
    GRAV, Rdry = cfg.GRAV, cfg.Rdry

    # --- top temperature ---
    if cfg.is_top_tem:
        tem_top = tem[:, :, kmax, :]
    elif cfg.is_top_epl:
        tem_top = _lag_intpl(
            phi[:, :, kmaxp1, :] / GRAV,
            phi[:, :, kmax,   :] / GRAV, tem[:, :, kmax,   :],
            phi[:, :, kmaxm1, :] / GRAV, tem[:, :, kmaxm1, :],
            phi[:, :, kmaxm2, :] / GRAV, tem[:, :, kmaxm2, :],
        )
    else:
        tem_top = tem[:, :, kmaxp1, :]

    # --- bottom temperature ---
    if cfg.is_btm_tem:
        tem_btm = tem[:, :, kmin, :]
    elif cfg.is_btm_epl:
        tem_btm = _lag_intpl(
            phi[:, :, kminm1, :] / GRAV,
            phi[:, :, kminp2, :] / GRAV, tem[:, :, kminp2, :],
            phi[:, :, kminp1, :] / GRAV, tem[:, :, kminp1, :],
            phi[:, :, kmin,   :] / GRAV, tem[:, :, kmin,   :],
        )
    else:
        tem_btm = tem[:, :, kminm1, :]

    # --- pressure (hydrostatic) ---
    pre_top = pre[:, :, kmaxm1, :] - rho[:, :, kmax, :] * (
        phi[:, :, kmaxp1, :] - phi[:, :, kmaxm1, :])
    pre_btm = pre[:, :, kminp1, :] - rho[:, :, kmin, :] * (
        phi[:, :, kminm1, :] - phi[:, :, kminp1, :])

    # --- density (equation of state, uses the new tem/pre boundaries) ---
    rho_top = pre_top / (Rdry * tem_top)
    rho_btm = pre_btm / (Rdry * tem_btm)

    return tem_top, tem_btm, pre_top, pre_btm, rho_top, rho_btm


def compute_bndcnd_thermo_pl(tem, rho, pre, phi, cfg, xp):
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1, kminp1, kminp2 = kmin - 1, kmin + 1, kmin + 2
    kmaxm1, kmaxm2, kmaxp1 = kmax - 1, kmax - 2, kmax + 1
    GRAV, Rdry = cfg.GRAV, cfg.Rdry

    if cfg.is_top_tem:
        tem_top = tem[:, kmax, :]
    elif cfg.is_top_epl:
        tem_top = _lag_intpl(
            phi[:, kmaxp1, :] / GRAV,
            phi[:, kmax,   :] / GRAV, tem[:, kmax,   :],
            phi[:, kmaxm1, :] / GRAV, tem[:, kmaxm1, :],
            phi[:, kmaxm2, :] / GRAV, tem[:, kmaxm2, :],
        )
    else:
        tem_top = tem[:, kmaxp1, :]

    if cfg.is_btm_tem:
        tem_btm = tem[:, kmin, :]
    elif cfg.is_btm_epl:
        tem_btm = _lag_intpl(
            phi[:, kminm1, :] / GRAV,
            phi[:, kminp2, :] / GRAV, tem[:, kminp2, :],
            phi[:, kminp1, :] / GRAV, tem[:, kminp1, :],
            phi[:, kmin,   :] / GRAV, tem[:, kmin,   :],
        )
    else:
        tem_btm = tem[:, kminm1, :]

    pre_top = pre[:, kmaxm1, :] - rho[:, kmax, :] * (
        phi[:, kmaxp1, :] - phi[:, kmaxm1, :])
    pre_btm = pre[:, kminp1, :] - rho[:, kmin, :] * (
        phi[:, kminm1, :] - phi[:, kminp1, :])

    rho_top = pre_top / (Rdry * tem_top)
    rho_btm = pre_btm / (Rdry * tem_btm)

    return tem_top, tem_btm, pre_top, pre_btm, rho_top, rho_btm


# ---------------------------------------------------------------------------
# Horizontal momentum (rhogvx/vy/vz) at the two ghost rows.
# Returns (vx_top, vy_top, vz_top, vx_btm, vy_btm, vz_btm).
# ---------------------------------------------------------------------------
def compute_bndcnd_rhovxvyvz_reg(rhog, rhogvx, rhogvy, rhogvz, cfg, xp):
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1, kmaxp1 = kmin - 1, kmax + 1

    # --- top (k = kmax+1) ---
    if cfg.is_top_rigid:
        sgn_t = -1.0
    elif cfg.is_top_free:
        sgn_t = 1.0
    else:
        sgn_t = None
    if sgn_t is not None:
        rg_t = rhog[:, :, kmaxp1, :]
        vx_top = sgn_t * (rhogvx[:, :, kmax, :] / rhog[:, :, kmax, :]) * rg_t
        vy_top = sgn_t * (rhogvy[:, :, kmax, :] / rhog[:, :, kmax, :]) * rg_t
        vz_top = sgn_t * (rhogvz[:, :, kmax, :] / rhog[:, :, kmax, :]) * rg_t
    else:
        vx_top = rhogvx[:, :, kmaxp1, :]
        vy_top = rhogvy[:, :, kmaxp1, :]
        vz_top = rhogvz[:, :, kmaxp1, :]

    # --- bottom (k = kmin-1) ---
    if cfg.is_btm_rigid:
        sgn_b = -1.0
    elif cfg.is_btm_free:
        sgn_b = 1.0
    else:
        sgn_b = None
    if sgn_b is not None:
        rg_b = rhog[:, :, kminm1, :]
        vx_btm = sgn_b * (rhogvx[:, :, kmin, :] / rhog[:, :, kmin, :]) * rg_b
        vy_btm = sgn_b * (rhogvy[:, :, kmin, :] / rhog[:, :, kmin, :]) * rg_b
        vz_btm = sgn_b * (rhogvz[:, :, kmin, :] / rhog[:, :, kmin, :]) * rg_b
    else:
        vx_btm = rhogvx[:, :, kminm1, :]
        vy_btm = rhogvy[:, :, kminm1, :]
        vz_btm = rhogvz[:, :, kminm1, :]

    return vx_top, vy_top, vz_top, vx_btm, vy_btm, vz_btm


def compute_bndcnd_rhovxvyvz_pl(rhog, rhogvx, rhogvy, rhogvz, cfg, xp):
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1, kmaxp1 = kmin - 1, kmax + 1

    if cfg.is_top_rigid:
        sgn_t = -1.0
    elif cfg.is_top_free:
        sgn_t = 1.0
    else:
        sgn_t = None
    if sgn_t is not None:
        rg_t = rhog[:, kmaxp1, :]
        vx_top = sgn_t * (rhogvx[:, kmax, :] / rhog[:, kmax, :]) * rg_t
        vy_top = sgn_t * (rhogvy[:, kmax, :] / rhog[:, kmax, :]) * rg_t
        vz_top = sgn_t * (rhogvz[:, kmax, :] / rhog[:, kmax, :]) * rg_t
    else:
        vx_top = rhogvx[:, kmaxp1, :]
        vy_top = rhogvy[:, kmaxp1, :]
        vz_top = rhogvz[:, kmaxp1, :]

    if cfg.is_btm_rigid:
        sgn_b = -1.0
    elif cfg.is_btm_free:
        sgn_b = 1.0
    else:
        sgn_b = None
    if sgn_b is not None:
        rg_b = rhog[:, kminm1, :]
        vx_btm = sgn_b * (rhogvx[:, kmin, :] / rhog[:, kmin, :]) * rg_b
        vy_btm = sgn_b * (rhogvy[:, kmin, :] / rhog[:, kmin, :]) * rg_b
        vz_btm = sgn_b * (rhogvz[:, kmin, :] / rhog[:, kmin, :]) * rg_b
    else:
        vx_btm = rhogvx[:, kminm1, :]
        vy_btm = rhogvy[:, kminm1, :]
        vz_btm = rhogvz[:, kminm1, :]

    return vx_top, vy_top, vz_top, vx_btm, vy_btm, vz_btm


# ---------------------------------------------------------------------------
# Vertical momentum (rhogw) at k = kmax+1 (top) and k = kmin (bottom).
# c2wfact here is VMTR_C2WfactGz (6 components). The k = kmin-1 ghost is always
# zero (handled by the caller). Returns (rw_top, rw_btm).
# ---------------------------------------------------------------------------
def compute_bndcnd_rhow_reg(rhogvx, rhogvy, rhogvz, c2wfact, cfg, xp):
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1, kmaxp1 = kmin - 1, kmax + 1

    if cfg.is_top_rigid:
        rw_top = xp.zeros_like(rhogvx[:, :, kmaxp1, :])
    elif cfg.is_top_free:
        rw_top = -(
            c2wfact[:, :, kmaxp1, :, 0] * rhogvx[:, :, kmaxp1, :] +
            c2wfact[:, :, kmaxp1, :, 1] * rhogvx[:, :, kmax,   :] +
            c2wfact[:, :, kmaxp1, :, 2] * rhogvy[:, :, kmaxp1, :] +
            c2wfact[:, :, kmaxp1, :, 3] * rhogvy[:, :, kmax,   :] +
            c2wfact[:, :, kmaxp1, :, 4] * rhogvz[:, :, kmaxp1, :] +
            c2wfact[:, :, kmaxp1, :, 5] * rhogvz[:, :, kmax,   :]
        )
    else:
        rw_top = None

    if cfg.is_btm_rigid:
        rw_btm = xp.zeros_like(rhogvx[:, :, kmin, :])
    elif cfg.is_btm_free:
        rw_btm = -(
            c2wfact[:, :, kmin, :, 0] * rhogvx[:, :, kmin,   :] +
            c2wfact[:, :, kmin, :, 1] * rhogvx[:, :, kminm1, :] +
            c2wfact[:, :, kmin, :, 2] * rhogvy[:, :, kmin,   :] +
            c2wfact[:, :, kmin, :, 3] * rhogvy[:, :, kminm1, :] +
            c2wfact[:, :, kmin, :, 4] * rhogvz[:, :, kmin,   :] +
            c2wfact[:, :, kmin, :, 5] * rhogvz[:, :, kminm1, :]
        )
    else:
        rw_btm = None

    return rw_top, rw_btm


def compute_bndcnd_rhow_pl(rhogvx, rhogvy, rhogvz, c2wfact, cfg, xp):
    kmin, kmax = cfg.kmin, cfg.kmax
    kminm1, kmaxp1 = kmin - 1, kmax + 1

    if cfg.is_top_rigid:
        rw_top = xp.zeros_like(rhogvx[:, kmaxp1, :])
    elif cfg.is_top_free:
        rw_top = -(
            c2wfact[:, kmaxp1, :, 0] * rhogvx[:, kmaxp1, :] +
            c2wfact[:, kmaxp1, :, 1] * rhogvx[:, kmax,   :] +
            c2wfact[:, kmaxp1, :, 2] * rhogvy[:, kmaxp1, :] +
            c2wfact[:, kmaxp1, :, 3] * rhogvy[:, kmax,   :] +
            c2wfact[:, kmaxp1, :, 4] * rhogvz[:, kmaxp1, :] +
            c2wfact[:, kmaxp1, :, 5] * rhogvz[:, kmax,   :]
        )
    else:
        rw_top = None

    if cfg.is_btm_rigid:
        rw_btm = xp.zeros_like(rhogvx[:, kmin, :])
    elif cfg.is_btm_free:
        rw_btm = -(
            c2wfact[:, kmin, :, 0] * rhogvx[:, kmin,   :] +
            c2wfact[:, kmin, :, 1] * rhogvx[:, kminm1, :] +
            c2wfact[:, kmin, :, 2] * rhogvy[:, kmin,   :] +
            c2wfact[:, kmin, :, 3] * rhogvy[:, kminm1, :] +
            c2wfact[:, kmin, :, 4] * rhogvz[:, kmin,   :] +
            c2wfact[:, kmin, :, 5] * rhogvz[:, kminm1, :]
        )
    else:
        rw_btm = None

    return rw_top, rw_btm
