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


def compute_rhow_matrix_reg(eth, g_tilde, RGSQRTH, RGSGAM2, GAM2H, RGAMH,
                            rdgzh, rdgz, dfact, cfact, dt, cfg, xp):
    """Regional matrix assembly. Returns Mc, Mu, Ml interior slabs (i,j,kmax-kmin,l)."""
    kmin, kmax = cfg.kmin, cfg.kmax
    ks  = slice(kmin + 1, kmax + 1)   # k = kmin+1 .. kmax
    ksp = slice(kmin + 2, kmax + 2)   # k+1
    ksm = slice(kmin,     kmax)       # k-1

    GCVovR   = cfg.GRAV * cfg.CVdry / cfg.Rdry
    ACVovRt2 = cfg.alpha * cfg.CVdry / cfg.Rdry / (dt * dt)

    rgdzh   = rdgzh[ks][None, None, :, None]
    rgdz    = rdgz[ks][None, None, :, None]
    rgdzm1  = rdgz[ksm][None, None, :, None]
    dfact_  = dfact[ks][None, None, :, None]
    cfact_  = cfact[ks][None, None, :, None]
    cfactm1 = cfact[ksm][None, None, :, None]
    dfactm1 = dfact[ksm][None, None, :, None]

    RGSQRTH_  = RGSQRTH[:, :, ks, :]
    RGSGAM2_  = RGSGAM2[:, :, ks, :]
    RGSGAM2m1 = RGSGAM2[:, :, ksm, :]
    GAM2H_    = GAM2H[:, :, ks, :]
    eth_      = eth[:, :, ks, :]
    gtilde_   = g_tilde[:, :, ks, :]
    RGAMH_    = RGAMH[:, :, ks, :]

    Mc = (
        ACVovRt2 / RGSQRTH_
        + rgdzh * (
            (RGSGAM2_ * rgdz + RGSGAM2m1 * rgdzm1) * GAM2H_ * eth_
            - (dfact_ - cfactm1) * (gtilde_ + GCVovR)
        )
    )

    Mu = -rgdzh * (
        RGSGAM2_ * rgdz * GAM2H[:, :, ksp, :] * eth[:, :, ksp, :]
        + cfact_ * (
            g_tilde[:, :, ksp, :]
            + GAM2H[:, :, ksp, :] * RGAMH_ ** 2 * GCVovR
        )
    )

    Ml = -rgdzh * (
        RGSGAM2_ * rgdz * GAM2H[:, :, ksm, :] * eth[:, :, ksm, :]
        - dfactm1 * (
            g_tilde[:, :, ksm, :]
            + GAM2H[:, :, ksm, :] * RGAMH_ ** 2 * GCVovR
        )
    )
    return Mc, Mu, Ml


def compute_rhow_matrix_pl(eth_pl, g_tilde_pl, RGSQRTH_pl, RGSGAM2_pl, GAM2H_pl,
                           RGAMH_pl, rdgzh, rdgz, dfact, cfact, dt, cfg, xp):
    """Pole matrix assembly. Returns Mc_pl, Mu_pl, Ml_pl interior slabs (g,kmax-kmin,l)."""
    kmin, kmax = cfg.kmin, cfg.kmax
    ks  = slice(kmin + 1, kmax + 1)
    ksp = slice(kmin + 2, kmax + 2)
    ksm = slice(kmin,     kmax)

    GCVovR   = cfg.GRAV * cfg.CVdry / cfg.Rdry
    ACVovRt2 = cfg.alpha * cfg.CVdry / cfg.Rdry / (dt * dt)

    rgdzh   = rdgzh[ks][None, :, None]
    rgdz    = rdgz[ks][None, :, None]
    rgdzm1  = rdgz[ksm][None, :, None]
    dfact_  = dfact[ks][None, :, None]
    cfact_  = cfact[ks][None, :, None]
    cfactm1 = cfact[ksm][None, :, None]
    dfactm1 = dfact[ksm][None, :, None]

    RGSQRTH_  = RGSQRTH_pl[:, ks, :]
    RGSGAM2_  = RGSGAM2_pl[:, ks, :]
    RGSGAM2m1 = RGSGAM2_pl[:, ksm, :]
    GAM2H_    = GAM2H_pl[:, ks, :]
    eth_      = eth_pl[:, ks, :]
    gtilde_   = g_tilde_pl[:, ks, :]
    RGAMH_    = RGAMH_pl[:, ks, :]

    Mc = (
        ACVovRt2 / RGSQRTH_
        + rgdzh * (
            (RGSGAM2_ * rgdz + RGSGAM2m1 * rgdzm1) * GAM2H_ * eth_
            - (dfact_ - cfactm1) * (gtilde_ + GCVovR)
        )
    )

    Mu = -rgdzh * (
        RGSGAM2_ * rgdz * GAM2H_pl[:, ksp, :] * eth_pl[:, ksp, :]
        + cfact_ * (
            g_tilde_pl[:, ksp, :]
            + GAM2H_pl[:, ksp, :] * RGAMH_ ** 2 * GCVovR
        )
    )

    Ml = -rgdzh * (
        RGSGAM2_ * rgdz * GAM2H_pl[:, ksm, :] * eth_pl[:, ksm, :]
        - dfactm1 * (
            g_tilde_pl[:, ksm, :]
            + GAM2H_pl[:, ksm, :] * RGAMH_ ** 2 * GCVovR
        )
    )
    return Mc, Mu, Ml
