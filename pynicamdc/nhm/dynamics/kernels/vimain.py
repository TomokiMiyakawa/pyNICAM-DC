"""
Pure / backend-switchable (numpy <-> jax.numpy) FUSED kernel for the whole
COMM-free core of mod_vi.py vi_main (L793-1323).

vi_main is comm-free: it composes five already-extracted pure kernels --
src_flux_convergence (fluxconv), src_advection_convergence (advconv +
fluxconv), the vertical-implicit Thomas solve (virhowsolver), the rhogw
boundary condition (bndcnd), and the kinetic-energy diagnostic (rhogkin) --
plus elementwise energy / prognostic glue.

The live model currently dispatches each of those kernels separately, paying an
asarray/to_numpy host<->device round-trip and a fresh jit launch per call (~7
per vi_main). This module instead composes them into ONE pure function so that,
under jax.jit, XLA fuses the entire sequence into a single graph (no host
round-trips between sub-kernels, full cross-kernel fusion). Under numpy it is
arithmetically identical to the per-kernel path (bit-for-bit), since it calls
the exact same sub-kernels with the same constants in the same order.

Inputs / outputs are passed as dicts of arrays (jax pytrees); the static config
bundles the five sub-cfgs plus the TIME_split flag and Rdry/CVdry.

Returned dict (matching vi_main's in-place writes):
    rhog_split1, rhogw_split1, rhoge_split1   (regional)
    *_pl                                       (pole; present only if have_pl)
"""

from __future__ import annotations
from dataclasses import dataclass

from pynicamdc.nhm.dynamics.kernels.fluxconv import compute_flux_convergence, FluxConvCfg
from pynicamdc.nhm.dynamics.kernels.advconv import compute_scaled_fluxes, AdvConvCfg
from pynicamdc.nhm.dynamics.kernels.virhowsolver import (
    compute_rhow_solver_reg, compute_rhow_solver_pl, ViSolverCfg,
)
from pynicamdc.nhm.dynamics.kernels.bndcnd import (
    compute_bndcnd_rhow_reg, compute_bndcnd_rhow_pl, BndCfg,
)
from pynicamdc.nhm.dynamics.kernels.rhogkin import (
    compute_rhogkin_reg, compute_rhogkin_pl, RhogkinCfg,
)


@dataclass(frozen=True)
class VimainCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit.

    Bundles the five sub-kernel cfgs (each itself a frozen dataclass, hence
    hashable) plus the scalars / flags vi_main's glue needs.
    """
    kmin: int
    kmax: int
    have_pl: bool
    TIME_split: bool
    Rdry: float
    CVdry: float
    I_SRC_default: int
    I_SRC_horizontal: int
    flux: FluxConvCfg
    adv: AdvConvCfg
    sol: ViSolverCfg
    bnd: BndCfg
    kin: RhogkinCfg


# ---------------------------------------------------------------------------
# composable helpers (mirror mod_src.src_flux_convergence /
# src_advection_convergence, both reg+pl in one call)
# ---------------------------------------------------------------------------
def _flux_conv(C, vx, vy, vz, w, vx_pl, vy_pl, vz_pl, w_pl, fluxtype, cfg, xp):
    return compute_flux_convergence(
        vx, vy, vz, w, vx_pl, vy_pl, vz_pl, w_pl,
        C["RGAM"], C["RGAMH"], C["RGSQRTH"], C["C2WfactGz"], C["coef_div"], C["rdgz"],
        C["RGAM_pl"], C["RGAMH_pl"], C["RGSQRTH_pl"], C["C2WfactGz_pl"], C["coef_div_pl"],
        fluxtype, cfg.flux, xp,
    )


def _adv_conv(C, vx, vy, vz, w, scl, vx_pl, vy_pl, vz_pl, w_pl, scl_pl,
              fluxtype, cfg, xp):
    (sx, sy, sz, sw,
     sx_pl, sy_pl, sz_pl, sw_pl) = compute_scaled_fluxes(
        vx, vy, vz, w, scl, vx_pl, vy_pl, vz_pl, w_pl, scl_pl,
        C["afact"], C["bfact"], fluxtype, cfg.adv, xp,
    )
    return _flux_conv(C, sx, sy, sz, sw, sx_pl, sy_pl, sz_pl, sw_pl,
                      fluxtype, cfg, xp)


def _build_rhogw_bc_reg(vx, vy, vz, c2w, cfg, xp):
    """rhogw_split1 = 0 everywhere, then BNDCND_rhow sets k=kmin and k=kmax+1.

    Reproduces mod_bndcnd.BNDCND_rhow: row kmin-1 stays 0, row kmin <- rw_btm
    (0 when rigid), interior kmin+1..kmax stay 0, row kmax+1 <- rw_top, rows
    above kmax+1 (none in the standard layout) stay 0.
    """
    kmin, kmax = cfg.kmin, cfg.kmax
    i, j, kall, l = vx.shape
    rw_top, rw_btm = compute_bndcnd_rhow_reg(vx, vy, vz, c2w, cfg.bnd, xp)
    zrow = xp.zeros((i, j, 1, l), dtype=vx.dtype)
    rkmin  = (rw_btm if rw_btm is not None
              else xp.zeros((i, j, l), dtype=vx.dtype))[:, :, None, :]
    rkmaxp = (rw_top if rw_top is not None
              else xp.zeros((i, j, l), dtype=vx.dtype))[:, :, None, :]
    parts = []
    if kmin > 0:                                   # rows 0 .. kmin-1
        parts.append(xp.zeros((i, j, kmin, l), dtype=vx.dtype))
    parts.append(rkmin)                            # row kmin
    parts.append(xp.zeros((i, j, kmax - kmin, l), dtype=vx.dtype))  # kmin+1..kmax
    parts.append(rkmaxp)                           # row kmax+1
    n_post = kall - kmax - 2
    if n_post > 0:                                 # rows kmax+2 .. end
        parts.append(xp.zeros((i, j, n_post, l), dtype=vx.dtype))
    return xp.concatenate(parts, axis=2)


def _build_rhogw_bc_pl(vx, vy, vz, c2w, cfg, xp):
    kmin, kmax = cfg.kmin, cfg.kmax
    g, kall, l = vx.shape
    rw_top, rw_btm = compute_bndcnd_rhow_pl(vx, vy, vz, c2w, cfg.bnd, xp)
    rkmin  = (rw_btm if rw_btm is not None
              else xp.zeros((g, l), dtype=vx.dtype))[:, None, :]
    rkmaxp = (rw_top if rw_top is not None
              else xp.zeros((g, l), dtype=vx.dtype))[:, None, :]
    parts = []
    if kmin > 0:
        parts.append(xp.zeros((g, kmin, l), dtype=vx.dtype))
    parts.append(rkmin)
    parts.append(xp.zeros((g, kmax - kmin, l), dtype=vx.dtype))
    parts.append(rkmaxp)
    n_post = kall - kmax - 2
    if n_post > 0:
        parts.append(xp.zeros((g, n_post, l), dtype=vx.dtype))
    return xp.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# the fused vi_main core
# ---------------------------------------------------------------------------
def compute_vi_main(P, C, dt, cfg: VimainCfg, xp):
    """Single fused composition of vi_main's comm-free core (reg + pole).

    P : dict of prognostic / source arrays (see mod_vi.vi_main wiring).
    C : dict of metric constants (see mod_vi.vi_main wiring).
    dt: small-step timestep (traced scalar / python float).
    Returns dict with rhog_split1, rhogw_split1, rhoge_split1 (+ _pl if have_pl).
    """
    Rdry, CVdry = cfg.Rdry, cfg.CVdry
    have_pl = cfg.have_pl
    Ih, Id = cfg.I_SRC_horizontal, cfg.I_SRC_default

    # --- 1) split source terms (TIME_split -> horizontal h/adv convergence) ---
    if cfg.TIME_split:
        drhog, drhog_pl = _flux_conv(
            C, P["rhogvx_s1"], P["rhogvy_s1"], P["rhogvz_s1"], P["rhogw_s0"],
            P["rhogvx_s1_pl"], P["rhogvy_s1_pl"], P["rhogvz_s1_pl"], P["rhogw_s0_pl"],
            Ih, cfg, xp)
        drhoge, drhoge_pl = _adv_conv(
            C, P["rhogvx_s1"], P["rhogvy_s1"], P["rhogvz_s1"], P["rhogw_s0"], P["eth0"],
            P["rhogvx_s1_pl"], P["rhogvy_s1_pl"], P["rhogvz_s1_pl"], P["rhogw_s0_pl"], P["eth0_pl"],
            Ih, cfg, xp)
    else:
        drhog   = xp.zeros_like(P["rhogw_s0"])
        drhoge  = xp.zeros_like(P["rhogw_s0"])
        drhog_pl  = xp.zeros_like(P["rhogw_s0_pl"])
        drhoge_pl = xp.zeros_like(P["rhogw_s0_pl"])

    # --- 2) grhog1 / grhoge1 / gpre ---
    grhog1  = P["grhog"]  + drhog
    grhoge1 = P["grhoge"] + drhoge
    gpre    = grhoge1 * Rdry / CVdry
    if have_pl:
        grhog1_pl  = P["grhog_pl"]  + drhog_pl
        grhoge1_pl = P["grhoge_pl"] + drhoge_pl
        gpre_pl    = grhoge1_pl * Rdry / CVdry

    # --- 3) rhogw_split1 boundary init (zeros + BNDCND_rhow) ---
    rhogw_s1 = _build_rhogw_bc_reg(
        P["rhogvx_s1"], P["rhogvy_s1"], P["rhogvz_s1"], C["C2WfactGz"], cfg, xp)
    if have_pl:
        rhogw_s1_pl = _build_rhogw_bc_pl(
            P["rhogvx_s1_pl"], P["rhogvy_s1_pl"], P["rhogvz_s1_pl"],
            C["C2WfactGz_pl"], cfg, xp)

    # --- 4) vertical-implicit Thomas solve ---
    rhogw_s1 = compute_rhow_solver_reg(
        rhogw_s1, P["rhogw_s0"], P["preg_s0"], P["rhog_s0"],
        grhog1, P["grhogw"], gpre, P["Mc"], P["Mu"], P["Ml"],
        C["RGAMH"], C["RGSGAM2"], C["RGAM"], C["RGSGAM2H"], C["GSGAM2H"],
        C["rdgzh"], C["afact"], C["bfact"], dt, cfg.sol, xp)
    if have_pl:
        rhogw_s1_pl = compute_rhow_solver_pl(
            rhogw_s1_pl, P["rhogw_s0_pl"], P["preg_s0_pl"], P["rhog_s0_pl"],
            grhog1_pl, P["grhogw_pl"], gpre_pl, P["Mc_pl"], P["Mu_pl"], P["Ml_pl"],
            C["RGAMH_pl"], C["RGSGAM2_pl"], C["RGAM_pl"], C["RGSGAM2H_pl"], C["GSGAM2H_pl"],
            C["rdgzh"], C["afact"], C["bfact"], dt, cfg.sol, xp)

    # --- 5) rhog_split1 via default flux convergence (uses solved rhogw_s1) ---
    drhog2, drhog2_pl = _flux_conv(
        C, P["rhogvx_s1"], P["rhogvy_s1"], P["rhogvz_s1"], rhogw_s1,
        P["rhogvx_s1_pl"], P["rhogvy_s1_pl"], P["rhogvz_s1_pl"], rhogw_s1_pl if have_pl else P["rhogw_s0_pl"],
        Id, cfg, xp)
    rhog_s1 = P["rhog_s0"] + (P["grhog"] + drhog2) * dt
    if have_pl:
        rhog_s1_pl = P["rhog_s0_pl"] + (P["grhog_pl"] + drhog2_pl) * dt

    # --- 6) three rhogkin evaluations ---
    kin0 = compute_rhogkin_reg(
        P["rhog0"], P["rhogvx0"], P["rhogvy0"], P["rhogvz0"], P["rhogw0"],
        C["C2Wfact"], C["W2Cfact"], cfg.kin, xp)
    # previous + split(t=n)
    rhog1   = P["rhog0"]   + P["rhog_s0"]
    rhogvx1 = P["rhogvx0"] + P["rhogvx_s0"]
    rhogvy1 = P["rhogvy0"] + P["rhogvy_s0"]
    rhogvz1 = P["rhogvz0"] + P["rhogvz_s0"]
    rhogw1  = P["rhogw0"]  + P["rhogw_s0"]
    kin10 = compute_rhogkin_reg(
        rhog1, rhogvx1, rhogvy1, rhogvz1, rhogw1,
        C["C2Wfact"], C["W2Cfact"], cfg.kin, xp)
    # previous + split(t=n+1)
    rhog1b   = P["rhog0"]   + rhog_s1
    rhogvx1b = P["rhogvx0"] + P["rhogvx_s1"]
    rhogvy1b = P["rhogvy0"] + P["rhogvy_s1"]
    rhogvz1b = P["rhogvz0"] + P["rhogvz_s1"]
    rhogw1b  = P["rhogw0"]  + rhogw_s1
    kin11 = compute_rhogkin_reg(
        rhog1b, rhogvx1b, rhogvy1b, rhogvz1b, rhogw1b,
        C["C2Wfact"], C["W2Cfact"], cfg.kin, xp)

    if have_pl:
        kin0_pl = compute_rhogkin_pl(
            P["rhog0_pl"], P["rhogvx0_pl"], P["rhogvy0_pl"], P["rhogvz0_pl"], P["rhogw0_pl"],
            C["C2Wfact_pl"], C["W2Cfact_pl"], cfg.kin, xp)
        rhog1_pl   = P["rhog0_pl"]   + P["rhog_s0_pl"]
        rhogvx1_pl = P["rhogvx0_pl"] + P["rhogvx_s0_pl"]
        rhogvy1_pl = P["rhogvy0_pl"] + P["rhogvy_s0_pl"]
        rhogvz1_pl = P["rhogvz0_pl"] + P["rhogvz_s0_pl"]
        rhogw1_pl  = P["rhogw0_pl"]  + P["rhogw_s0_pl"]
        kin10_pl = compute_rhogkin_pl(
            rhog1_pl, rhogvx1_pl, rhogvy1_pl, rhogvz1_pl, rhogw1_pl,
            C["C2Wfact_pl"], C["W2Cfact_pl"], cfg.kin, xp)
        rhog1b_pl   = P["rhog0_pl"]   + rhog_s1_pl
        rhogvx1b_pl = P["rhogvx0_pl"] + P["rhogvx_s1_pl"]
        rhogvy1b_pl = P["rhogvy0_pl"] + P["rhogvy_s1_pl"]
        rhogvz1b_pl = P["rhogvz0_pl"] + P["rhogvz_s1_pl"]
        rhogw1b_pl  = P["rhogw0_pl"]  + rhogw_s1_pl
        kin11_pl = compute_rhogkin_pl(
            rhog1b_pl, rhogvx1b_pl, rhogvy1b_pl, rhogvz1b_pl, rhogw1b_pl,
            C["C2Wfact_pl"], C["W2Cfact_pl"], cfg.kin, xp)

    # --- 7) energy correction (Satoh 2002) ---
    ethtot0 = P["eth0"] + kin0 / P["rhog0"] + C["PHI"]
    drhogetot, drhogetot_pl = _adv_conv(
        C, rhogvx1b, rhogvy1b, rhogvz1b, rhogw1b, ethtot0,
        (P["rhogvx0_pl"] + P["rhogvx_s1_pl"]) if have_pl else P["rhogvx_s1_pl"],
        (P["rhogvy0_pl"] + P["rhogvy_s1_pl"]) if have_pl else P["rhogvy_s1_pl"],
        (P["rhogvz0_pl"] + P["rhogvz_s1_pl"]) if have_pl else P["rhogvz_s1_pl"],
        (P["rhogw0_pl"]  + rhogw_s1_pl)        if have_pl else P["rhogw_s0_pl"],
        (P["eth0_pl"] + kin0_pl / P["rhog0_pl"] + C["PHI_pl"]) if have_pl else P["eth0_pl"],
        Id, cfg, xp)
    rhoge_s1 = (P["rhoge_s0"] + (P["grhogetot"] + drhogetot) * dt
                + (kin10 - kin11)
                + (P["rhog_s0"] - rhog_s1) * C["PHI"])

    out = {"rhog_split1": rhog_s1, "rhogw_split1": rhogw_s1, "rhoge_split1": rhoge_s1}

    if have_pl:
        rhoge_s1_pl = (P["rhoge_s0_pl"] + (P["grhogetot_pl"] + drhogetot_pl) * dt
                       + (kin10_pl - kin11_pl)
                       + (P["rhog_s0_pl"] - rhog_s1_pl) * C["PHI_pl"])
        out["rhog_split1_pl"]  = rhog_s1_pl
        out["rhogw_split1_pl"] = rhogw_s1_pl
        out["rhoge_split1_pl"] = rhoge_s1_pl

    return out
