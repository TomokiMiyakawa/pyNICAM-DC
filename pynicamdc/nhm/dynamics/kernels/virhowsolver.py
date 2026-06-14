"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for the COMM-free
tri-diagonal (Thomas) solve of vi_rhow_solver (mod_vi.py L1324-1413).

This solves the vertical-implicit rho*w system column-by-column. Unlike the
data-parallel kernels, the forward / backward sweeps carry a dependency in k
(beta_k needs beta_{k-1}; x_k needs x_{k+1}), so they CANNOT be written as a
single broadcast expression. We therefore loop over k explicitly, carrying the
recurrence state (beta, previous row) in Python and accumulating per-level
results in a list, then stack along the k axis. The horizontal (i,j) and layer
(l) axes are independent and stay fully vectorized.

For numpy this reproduces the original per-element arithmetic order bit-for-bit.
Under jax.jit the k-loop is unrolled into the trace (kmax-kmin ~ a few dozen
steps); if graph size ever matters this inner loop can be swapped for lax.scan
without changing the call site.

Final layout of the returned rhogw (matching the original in-place updates):
    k in [0, kmin-1]        : unchanged (input rhogw)
    k = kmin                : rhogw_in * RGSGAM2H * GSGAM2H
    k in [kmin+1, kmax]     : x_k * GSGAM2H        (Thomas solution, scaled)
    k = kmax+1              : rhogw_in * RGSGAM2H * GSGAM2H
    k in [kmax+2, kall-1]   : unchanged (input rhogw)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ViSolverCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    kmin: int
    kmax: int
    have_pl: bool
    GRAV: float
    Rdry: float
    CVdry: float
    alpha: float


def _thomas(Sall_int, Mc_int, Mu_int, Ml_int, xp):
    """Sequential Thomas solve on the interior axis (axis=-2 carries k).

    Sall_int, Mc/Mu/Ml_int are (..., N, ...) with the k axis already isolated as
    a Python-indexable leading list via the caller. Here they are passed as
    lists along k. Returns list x[0..N-1].
    """
    N = len(Sall_int)
    # Forward sweep
    beta = Mc_int[0]
    u = [Sall_int[0] / beta]
    gamma = [None] * N
    for i in range(1, N):
        g = Mu_int[i - 1] / beta
        beta = Mc_int[i] - Ml_int[i] * g
        u.append((Sall_int[i] - Ml_int[i] * u[i - 1]) / beta)
        gamma[i] = g
    # Backward sweep
    x = [None] * N
    x[N - 1] = u[N - 1]
    for i in range(N - 2, -1, -1):
        x[i] = u[i] - gamma[i + 1] * x[i + 1]
    return x


def compute_rhow_solver_reg(rhogw, rhogw0, preg0, rhog0, Srho, Sw, Spre,
                            Mc, Mu, Ml,
                            RGAMH, RGSGAM2, RGAM, RGSGAM2H, GSGAM2H,
                            rdgzh, afact, bfact, dt, cfg, xp):
    """Regional Thomas solve. Returns new rhogw (i,j,kall,l)."""
    kmin, kmax = cfg.kmin, cfg.kmax
    ks  = slice(kmin + 1, kmax + 1)   # interior k = kmin+1 .. kmax
    ksm = slice(kmin,     kmax)       # k-1
    alpha   = cfg.alpha
    CVovRt2 = cfg.CVdry / cfg.Rdry / (dt * dt)
    GRAV    = cfg.GRAV

    rdgzh_  = rdgzh[ks][None, None, :, None]
    afact_  = afact[ks][None, None, :, None]
    bfact_  = bfact[ks][None, None, :, None]

    # --- source assembly Sall on interior rows (data-parallel in k) ---
    Sall = (
        (rhogw0[:, :, ks, :] * alpha + dt * Sw[:, :, ks, :]) * RGAMH[:, :, ks, :] ** 2
        - (
            (preg0[:, :, ks, :] + dt * Spre[:, :, ks, :]) * RGSGAM2[:, :, ks, :]
            - (preg0[:, :, ksm, :] + dt * Spre[:, :, ksm, :]) * RGSGAM2[:, :, ksm, :]
        ) * dt * rdgzh_
        - (
            (rhog0[:, :, ks, :] + dt * Srho[:, :, ks, :]) * RGAM[:, :, ks, :] ** 2 * afact_
            + (rhog0[:, :, ksm, :] + dt * Srho[:, :, ksm, :]) * RGAM[:, :, ksm, :] ** 2 * bfact_
        ) * dt * GRAV
    ) * CVovRt2

    # --- scaled boundary rho*w (used both in Sall BC and final output) ---
    rw_kmin   = rhogw[:, :, kmin, :]   * RGSGAM2H[:, :, kmin, :]
    rw_kmaxp1 = rhogw[:, :, kmax + 1, :] * RGSGAM2H[:, :, kmax + 1, :]

    N = kmax - kmin   # number of interior rows
    # split Sall / Mc / Mu / Ml into per-level lists along k (local idx 0..N-1)
    Sall_list = [Sall[:, :, i, :] for i in range(N)]
    Mc_list   = [Mc[:, :, kmin + 1 + i, :] for i in range(N)]
    Mu_list   = [Mu[:, :, kmin + 1 + i, :] for i in range(N)]
    Ml_list   = [Ml[:, :, kmin + 1 + i, :] for i in range(N)]

    # apply boundary conditions to the two end interior rows
    Sall_list[0]      = Sall_list[0]      - Ml[:, :, kmin + 1, :] * rw_kmin
    Sall_list[N - 1]  = Sall_list[N - 1]  - Mu[:, :, kmax, :]     * rw_kmaxp1

    x = _thomas(Sall_list, Mc_list, Mu_list, Ml_list, xp)
    x_int = xp.stack(x, axis=2)                       # (i,j,N,l)
    interior = x_int * GSGAM2H[:, :, ks, :]           # scale by GSGAM2H

    kmin_row   = (rw_kmin   * GSGAM2H[:, :, kmin, :])[:, :, None, :]
    kmaxp1_row = (rw_kmaxp1 * GSGAM2H[:, :, kmax + 1, :])[:, :, None, :]

    pre  = rhogw[:, :, 0:kmin, :]            # k = 0 .. kmin-1 (unchanged)
    post = rhogw[:, :, kmax + 2:, :]         # k = kmax+2 .. end (unchanged)
    return xp.concatenate([pre, kmin_row, interior, kmaxp1_row, post], axis=2)


def compute_rhow_solver_pl(rhogw_pl, rhogw0_pl, preg0_pl, rhog0_pl, Srho_pl, Sw_pl, Spre_pl,
                           Mc_pl, Mu_pl, Ml_pl,
                           RGAMH_pl, RGSGAM2_pl, RGAM_pl, RGSGAM2H_pl, GSGAM2H_pl,
                           rdgzh, afact, bfact, dt, cfg, xp):
    """Pole Thomas solve. Returns new rhogw_pl (g,kall,l)."""
    kmin, kmax = cfg.kmin, cfg.kmax
    ks  = slice(kmin + 1, kmax + 1)
    ksm = slice(kmin,     kmax)
    alpha   = cfg.alpha
    CVovRt2 = cfg.CVdry / cfg.Rdry / (dt * dt)
    GRAV    = cfg.GRAV

    rdgzh_  = rdgzh[ks][None, :, None]
    afact_  = afact[ks][None, :, None]
    bfact_  = bfact[ks][None, :, None]

    Sall = (
        (rhogw0_pl[:, ks, :] * alpha + dt * Sw_pl[:, ks, :]) * RGAMH_pl[:, ks, :] ** 2
        - (
            (preg0_pl[:, ks, :] + dt * Spre_pl[:, ks, :]) * RGSGAM2_pl[:, ks, :]
            - (preg0_pl[:, ksm, :] + dt * Spre_pl[:, ksm, :]) * RGSGAM2_pl[:, ksm, :]
        ) * dt * rdgzh_
        - (
            (rhog0_pl[:, ks, :] + dt * Srho_pl[:, ks, :]) * RGAM_pl[:, ks, :] ** 2 * afact_
            + (rhog0_pl[:, ksm, :] + dt * Srho_pl[:, ksm, :]) * RGAM_pl[:, ksm, :] ** 2 * bfact_
        ) * dt * GRAV
    ) * CVovRt2

    rw_kmin   = rhogw_pl[:, kmin, :]   * RGSGAM2H_pl[:, kmin, :]
    rw_kmaxp1 = rhogw_pl[:, kmax + 1, :] * RGSGAM2H_pl[:, kmax + 1, :]

    N = kmax - kmin
    Sall_list = [Sall[:, i, :] for i in range(N)]
    Mc_list   = [Mc_pl[:, kmin + 1 + i, :] for i in range(N)]
    Mu_list   = [Mu_pl[:, kmin + 1 + i, :] for i in range(N)]
    Ml_list   = [Ml_pl[:, kmin + 1 + i, :] for i in range(N)]

    Sall_list[0]     = Sall_list[0]     - Ml_pl[:, kmin + 1, :] * rw_kmin
    Sall_list[N - 1] = Sall_list[N - 1] - Mu_pl[:, kmax, :]     * rw_kmaxp1

    x = _thomas(Sall_list, Mc_list, Mu_list, Ml_list, xp)
    x_int = xp.stack(x, axis=1)                       # (g,N,l)
    interior = x_int * GSGAM2H_pl[:, ks, :]

    kmin_row   = (rw_kmin   * GSGAM2H_pl[:, kmin, :])[:, None, :]
    kmaxp1_row = (rw_kmaxp1 * GSGAM2H_pl[:, kmax + 1, :])[:, None, :]

    pre  = rhogw_pl[:, 0:kmin, :]
    post = rhogw_pl[:, kmax + 2:, :]
    return xp.concatenate([pre, kmin_row, interior, kmaxp1_row, post], axis=1)
