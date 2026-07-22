"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for the COMM-free
tri-diagonal matrix assembly of vi_rhow_update_matrix (mod_vi.py L686-842).

It builds the three diagonals Mc / Mu / Ml of the vertical-implicit rho*w
system on the interior k-rows k = kmin+1 .. kmax:

    A_o = VMTR_RGSGAM2 ; A_i = VMTR_GAM2H * eth ; B = g_tilde
    C_o = (1/VMTR_RGAMH^2) * (CVdry/Rdry*GRAV) ; C_i = VMTR_RGAMH^2
      (so C_o * C_i = GAM2H * RGAMH^2 * GCVovR in the original layout)
    D   = CVdry/Rdry/(dt*dt)/VMTR_RGSQRTH

    Mc(k) = alpha*D(k)
          + rdgzh(k) * ( ( rdgz(k)*A_o(k) + rdgz(k-1)*A_o(k-1) )*A_i(k)
                       - 0.5*(dfact(k)-cfact(k-1))*( B(k)+C_o(k)*C_i(k) ) )
    Mu(k) = - rdgzh(k) * ( rdgz(k)*A_o(k)*A_i(k+1)
                       + 0.5*cfact(k)*( B(k+1)+C_o(k)*C_i(k+1) ) )
    Ml(k) = - rdgzh(k) * ( rdgz(k)*A_o(k)*A_i(k-1)
                       - 0.5*dfact(k-1)*( B(k-1)+C_o(k)*C_i(k-1) ) )

The 0.5 * dfact/cfact factors of the "Original concept" are already folded into
GRD_dfact / GRD_cfact in this port (see mod_vi.py L751-754 which pass them raw),
so they appear here without the explicit 0.5 -- matching the vectorized original.

Only the interior slab (k = kmin+1 .. kmax) is returned; the caller writes it
back into the persistent Mc/Mu/Ml buffers, whose other rows are never read by
vi_rhow_solver (they stay at their CONST_UNDEF init, exactly as in the original).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ViMatrixCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    kmin: int
    kmax: int
    have_pl: bool
    GRAV: float
    Rdry: float
    CVdry: float
    alpha: float


# ---------------------------------------------------------------------------
# PRE-COMBINED geometry coefficients (COMM-sharding fix, plan v5.8)
# ---------------------------------------------------------------------------
# The matrix assembly forms pure-geometry-const sub-products
# (RGSGAM2*rgdz*GAM2H, GAM2H*RGAMH^2*GCVovR, ...) that XLA/algsimp const-folds
# when the geometry consts are HLO CONSTANTS (default alltoall path) but CANNOT
# fold when they are shard_map INPUTS (the ragged devconst-buffer path), leaving
# a different-but-valid runtime re-association that the ill-conditioned HEVI
# tridiagonal amplifies (~1e-3 on near-zero RHOGW; see comm-replace-plan_v5.txt
# v5.8). Pre-folding these products HOST-SIDE at setup (in source order) and
# feeding the COMBINED coefficient as an atomic leaf removes the const x const
# from the traced graph, so both paths compute one C*eth. Every fold here is
# CONTIGUOUS -> BIT-EXACT to the original expression (verified numpy 0.0).
#
# Returned dict (interior [ks] slabs, matching the compute_* consumers):
#   Ca  = (RGSGAM2[k]*rgdz[k] + RGSGAM2[k-1]*rgdz[k-1]) * GAM2H[k]   (Mc A_o*A_i)
#   Cb  = dfact[k] - cfact[k-1]                                       (Mc dfact/cfact)
#   Cu1 = RGSGAM2[k]*rgdz[k] * GAM2H[k+1]                             (Mu A_o*A_i(k+1))
#   Cu2 = GAM2H[k+1] * RGAMH[k]^2 * GCVovR                            (Mu C_o*C_i(k+1))
#   Cl1 = RGSGAM2[k]*rgdz[k] * GAM2H[k-1]                             (Ml A_o*A_i(k-1))
#   Cl2 = GAM2H[k-1] * RGAMH[k]^2 * GCVovR                            (Ml C_o*C_i(k-1))
def precombine_matrix_reg(RGSGAM2, GAM2H, RGAMH, rdgz, dfact, cfact, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    ks  = slice(kmin + 1, kmax + 1)
    ksp = slice(kmin + 2, kmax + 2)
    ksm = slice(kmin,     kmax)
    GCVovR = cfg.GRAV * cfg.CVdry / cfg.Rdry
    rgdz_   = rdgz[ks][None, None, :, None]
    rgdzm1  = rdgz[ksm][None, None, :, None]
    RGSGAM2_  = RGSGAM2[:, :, ks, :]
    RGSGAM2m1 = RGSGAM2[:, :, ksm, :]
    RGAMH_    = RGAMH[:, :, ks, :]
    GAM2H_    = GAM2H[:, :, ks, :]
    return {
        "Ca":  (RGSGAM2_ * rgdz_ + RGSGAM2m1 * rgdzm1) * GAM2H_,
        "Cb":  (dfact[ks] - cfact[ksm])[None, None, :, None],
        "Cu1": RGSGAM2_ * rgdz_ * GAM2H[:, :, ksp, :],
        "Cu2": GAM2H[:, :, ksp, :] * RGAMH_ ** 2 * GCVovR,
        "Cl1": RGSGAM2_ * rgdz_ * GAM2H[:, :, ksm, :],
        "Cl2": GAM2H[:, :, ksm, :] * RGAMH_ ** 2 * GCVovR,
    }


def precombine_matrix_pl(RGSGAM2_pl, GAM2H_pl, RGAMH_pl, rdgz, dfact, cfact, cfg):
    kmin, kmax = cfg.kmin, cfg.kmax
    ks  = slice(kmin + 1, kmax + 1)
    ksp = slice(kmin + 2, kmax + 2)
    ksm = slice(kmin,     kmax)
    GCVovR = cfg.GRAV * cfg.CVdry / cfg.Rdry
    rgdz_   = rdgz[ks][None, :, None]
    rgdzm1  = rdgz[ksm][None, :, None]
    RGSGAM2_  = RGSGAM2_pl[:, ks, :]
    RGSGAM2m1 = RGSGAM2_pl[:, ksm, :]
    RGAMH_    = RGAMH_pl[:, ks, :]
    GAM2H_    = GAM2H_pl[:, ks, :]
    return {
        "Ca":  (RGSGAM2_ * rgdz_ + RGSGAM2m1 * rgdzm1) * GAM2H_,
        "Cb":  (dfact[ks] - cfact[ksm])[None, :, None],
        "Cu1": RGSGAM2_ * rgdz_ * GAM2H_pl[:, ksp, :],
        "Cu2": GAM2H_pl[:, ksp, :] * RGAMH_ ** 2 * GCVovR,
        "Cl1": RGSGAM2_ * rgdz_ * GAM2H_pl[:, ksm, :],
        "Cl2": GAM2H_pl[:, ksm, :] * RGAMH_ ** 2 * GCVovR,
    }


def compute_rhow_matrix_reg(eth, g_tilde, RGSQRTH, rdgzh, dfact, cfact,
                            coef, dt, cfg, xp):
    """Regional matrix assembly from PRE-COMBINED geometry coeffs (coef =
    precombine_matrix_reg(...)). Returns Mc, Mu, Ml interior slabs
    (i,j,kmax-kmin,l). Bit-identical to the raw-const assembly."""
    kmin, kmax = cfg.kmin, cfg.kmax
    ks  = slice(kmin + 1, kmax + 1)
    ksp = slice(kmin + 2, kmax + 2)
    ksm = slice(kmin,     kmax)

    GCVovR   = cfg.GRAV * cfg.CVdry / cfg.Rdry
    ACVovRt2 = cfg.alpha * cfg.CVdry / cfg.Rdry / (dt * dt)

    rgdzh_  = rdgzh[ks][None, None, :, None]
    cfact_  = cfact[ks][None, None, :, None]
    dfactm1 = dfact[ksm][None, None, :, None]
    RGSQRTH_ = RGSQRTH[:, :, ks, :]
    eth_,     ethp,     ethm     = eth[:, :, ks, :],     eth[:, :, ksp, :],     eth[:, :, ksm, :]
    gtilde_,  gtildep,  gtildem  = g_tilde[:, :, ks, :], g_tilde[:, :, ksp, :], g_tilde[:, :, ksm, :]
    Ca, Cb = coef["Ca"], coef["Cb"]
    Cu1, Cu2, Cl1, Cl2 = coef["Cu1"], coef["Cu2"], coef["Cl1"], coef["Cl2"]

    Mc = ACVovRt2 / RGSQRTH_ + rgdzh_ * (Ca * eth_ - Cb * (gtilde_ + GCVovR))
    Mu = -rgdzh_ * (Cu1 * ethp + cfact_ * (gtildep + Cu2))
    Ml = -rgdzh_ * (Cl1 * ethm - dfactm1 * (gtildem + Cl2))
    return Mc, Mu, Ml


def compute_rhow_matrix_pl(eth_pl, g_tilde_pl, RGSQRTH_pl, rdgzh, dfact, cfact,
                           coef, dt, cfg, xp):
    """Pole matrix assembly from PRE-COMBINED geometry coeffs (coef =
    precombine_matrix_pl(...)). Returns Mc_pl, Mu_pl, Ml_pl interior slabs
    (g,kmax-kmin,l). Bit-identical to the raw-const assembly."""
    kmin, kmax = cfg.kmin, cfg.kmax
    ks  = slice(kmin + 1, kmax + 1)
    ksp = slice(kmin + 2, kmax + 2)
    ksm = slice(kmin,     kmax)

    GCVovR   = cfg.GRAV * cfg.CVdry / cfg.Rdry
    ACVovRt2 = cfg.alpha * cfg.CVdry / cfg.Rdry / (dt * dt)

    rgdzh_  = rdgzh[ks][None, :, None]
    cfact_  = cfact[ks][None, :, None]
    dfactm1 = dfact[ksm][None, :, None]
    RGSQRTH_ = RGSQRTH_pl[:, ks, :]
    eth_,    ethp,    ethm    = eth_pl[:, ks, :],     eth_pl[:, ksp, :],     eth_pl[:, ksm, :]
    gtilde_, gtildep, gtildem = g_tilde_pl[:, ks, :], g_tilde_pl[:, ksp, :], g_tilde_pl[:, ksm, :]
    Ca, Cb = coef["Ca"], coef["Cb"]
    Cu1, Cu2, Cl1, Cl2 = coef["Cu1"], coef["Cu2"], coef["Cl1"], coef["Cl2"]

    Mc = ACVovRt2 / RGSQRTH_ + rgdzh_ * (Ca * eth_ - Cb * (gtilde_ + GCVovR))
    Mu = -rgdzh_ * (Cu1 * ethp + cfact_ * (gtildep + Cu2))
    Ml = -rgdzh_ * (Cl1 * ethm - dfactm1 * (gtildem + Cl2))
    return Mc, Mu, Ml
