"""
Prototype: pure / backend-switchable (numpy <-> jax.numpy) kernel for the
diagnostic-variable block of dynamics_step (mod_dynamics.py L446-506).

This block has NO communication and NO I/O, so it is the lowest-risk place to
establish the patterns needed for jit:
  * pure function  (inputs -> outputs, no in-place mutation of caller state)
  * one source for both backends via an `xp` (np or jnp) argument
  * static integer indices / scalars passed in a frozen, hashable config
    (so they can be marked static_argnames under jax.jit and used in slicing)

The original in-place block computes, from the prognostic arrays PROG/PROGq:
    rho, DIAG[I_vx,I_vy,I_vz,I_tem,I_pre,I_w], ein, q   (+ work arrays cv, qd)

Index conventions (from nhm/share/mod_runconf.py):
    PROG  last axis : I_RHOG=0, I_RHOGVX=1, I_RHOGVY=2, I_RHOGVZ=3, I_RHOGW=4, I_RHOGE=5
    DIAG  last axis : I_pre=0,  I_tem=1,   I_vx=2,      I_vy=3,      I_vz=4,    I_w=5
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DiagCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    # PROG component indices
    I_RHOG: int
    I_RHOGVX: int
    I_RHOGVY: int
    I_RHOGVZ: int
    I_RHOGW: int
    I_RHOGE: int
    # DIAG component indices
    I_pre: int
    I_tem: int
    I_vx: int
    I_vy: int
    I_vz: int
    I_w: int
    # vertical / tracer ranges
    kmin: int
    kmax: int
    nmin: int   # NQW_STR
    nmax: int   # NQW_END
    iqv: int    # I_QV (index into q / tracer axis)
    # thermodynamic constants
    Rdry: float
    Rvap: float
    CVdry: float


def compute_diagnostics(PROG, PROGq, DIAG, GSGAM2, C2Wfact, CVW, cfg: DiagCfg, xp):
    """Pure version of mod_dynamics.py L446-506.

    Parameters
    ----------
    PROG     : (i, j, k, l, 6)   prognostic variables  [IN]
    PROGq    : (i, j, k, l, nq)  tracer variables       [IN]
    DIAG     : (i, j, k, l, 6)   previous diagnostics   [IN]
                 only the w-boundary rows of DIAG are reused (k<=kmin, k>kmax);
                 every other component is recomputed.
    GSGAM2   : (i, j, k, l)      VMTR_GSGAM2            [IN]
    C2Wfact  : (i, j, k, l, 2)   VMTR_C2Wfact           [IN]
    CVW      : (nq,)             tracer specific heats   [IN]
    cfg      : DiagCfg           static indices/scalars [IN, static]
    xp       : module            numpy or jax.numpy     [IN, static]

    Returns
    -------
    rho  : (i, j, k, l)
    DIAG : (i, j, k, l, 6)       newly assembled diagnostics
    ein  : (i, j, k, l)
    q    : (i, j, k, l, nq)
    cv   : (i, j, k, l)          work array (returned for bit-comparison)
    qd   : (i, j, k, l)          work array (returned for bit-comparison)
    """
    kmin, kmax = cfg.kmin, cfg.kmax

    # --- extract prognostics ---
    RHOG   = PROG[:, :, :, :, cfg.I_RHOG]
    RHOGVX = PROG[:, :, :, :, cfg.I_RHOGVX]
    RHOGVY = PROG[:, :, :, :, cfg.I_RHOGVY]
    RHOGVZ = PROG[:, :, :, :, cfg.I_RHOGVZ]
    RHOGE  = PROG[:, :, :, :, cfg.I_RHOGE]

    # --- density, velocities, internal energy ---
    rho = RHOG / GSGAM2
    vx  = RHOGVX / RHOG
    vy  = RHOGVY / RHOG
    vz  = RHOGVZ / RHOG
    ein = RHOGE / RHOG

    # --- tracer mixing ratios ---
    q = PROGq / RHOG[:, :, :, :, None]

    # --- cv, qd accumulated over the water tracer range [nmin, nmax] ---
    q_slice = q[:, :, :, :, cfg.nmin:cfg.nmax + 1]      # (i,j,k,l,nqr)
    CVW_slice = CVW[cfg.nmin:cfg.nmax + 1]              # (nqr,) -- always float64
    qd = 1.0 - xp.sum(q_slice, axis=4)
    # CVW is a float64 constant array (mod_runconf builds it with np.zeros, i.e.
    # always float64). The authoritative original (the pole block still inline in
    # mod_dynamics.py, L543-554) accumulates cv into a *float32* work buffer:
    #     cv = 0;  cv += np.sum(q(f32) * CVW(f64))   # sum in f64, rounded to f32
    #     cv += qd * CVdry                            # f32 + f32
    # i.e. the tracer sum is computed in float64 and rounded to the working
    # precision *before* the dry-air term is added; the whole thermo chain (tem,
    # pre) is then in the working precision. Reproduce that exactly by rounding
    # the sum to q's dtype here. Bit-identical in float32 to the original in-place
    # block, and a no-op in float64 mode (q is float64 -> astype is identity).
    cv = xp.sum(q_slice * CVW_slice, axis=4).astype(q_slice.dtype) + qd * cfg.CVdry

    # --- temperature, pressure ---
    tem = ein / cv
    pre = rho * tem * (qd * cfg.Rdry + q[:, :, :, :, cfg.iqv] * cfg.Rvap)

    # --- vertical velocity w on interior levels (kmin+1 .. kmax) ---
    rhogw_int = PROG[:, :, kmin + 1:kmax + 1, :, cfg.I_RHOGW]
    rhog_k    = PROG[:, :, kmin + 1:kmax + 1, :, cfg.I_RHOG]
    rhog_km1  = PROG[:, :, kmin:kmax,         :, cfg.I_RHOG]
    f1 = C2Wfact[:, :, kmin + 1:kmax + 1, :, 0]
    f2 = C2Wfact[:, :, kmin + 1:kmax + 1, :, 1]
    w_int = rhogw_int / (f1 * rhog_k + f2 * rhog_km1)

    # boundary w rows are kept from the incoming DIAG (set later by BNDCND).
    # Build the full-k w array functionally (no in-place) via concatenation.
    w_old = DIAG[:, :, :, :, cfg.I_w]
    w_full = xp.concatenate(
        [w_old[:, :, :kmin + 1, :], w_int, w_old[:, :, kmax + 1:, :]],
        axis=2,
    )

    # --- assemble DIAG in the correct component order ---
    comps = [None] * 6
    comps[cfg.I_pre] = pre
    comps[cfg.I_tem] = tem
    comps[cfg.I_vx]  = vx
    comps[cfg.I_vy]  = vy
    comps[cfg.I_vz]  = vz
    comps[cfg.I_w]   = w_full
    DIAG_new = xp.stack(comps, axis=-1)

    return rho, DIAG_new, ein, q, cv, qd
