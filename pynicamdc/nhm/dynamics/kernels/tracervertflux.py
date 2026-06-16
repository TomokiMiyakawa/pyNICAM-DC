"""
Pure / backend-switchable (numpy <-> jax.numpy) kernel for the once-per-call,
tracer-independent vertical quantities of src_tracer_advection's first
fractional vertical step (mod_src_tracer.py L106-353):

  1. half-level vertical mass flux  flx_v   (interior k = kmin+1..kmax)
  2. Courant numbers                ck[...,0], ck[...,1]   (k = kmin..kmax)
  3. density tendency factor        d = b1 * frhog / rhog_in * dt   (all k)
  4. updated density                rhog    (k = kmin..kmax, with the two
     vertical boundary rows copied from rhog_in)

These do NOT depend on the per-tracer (iq) loop or the Thuburn limiter, so they
are a clean, self-contained unit. The original computes the regional parts with
numpy slicing already but runs the pole (`_pl`) parts as triple-nested Python
loops and the rhog update as a Python k-loop; this version is fully vectorised
and routes through `xp`, so under the jax backend the whole block is one
jit-able function.

Boundary / coverage note
------------------------
The original leaves rows it never reads as stale buffer contents (CONST_UNDEF).
This functional version zeros those rows instead, which is downstream-identical
for the standard layout kmin == 1 (verified by full-run bit-exactness):

  flx_v : interior kmin+1..kmax; zero on kmin and kmax+1 (and the unread rest).
  ck    : filled kmin..kmax;     zero on kmin-1, kmax+1 (and the unread rest).
  d     : all k (pure elementwise).
  rhog  : interior kmin..kmax; row kmin-1 = rhog_in[kmin]; row kmax+1 =
          rhog_in[kmax]; the unread rest zeroed.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TracerVertFluxCfg:
    """Static (hashable) parameters: safe to mark static under jax.jit."""
    kmin: int
    kmax: int
    have_pl: bool


def _vert_flux_interior(c2w, vx, vy, vz, rgamh, rgsqrth, rw, ksl, kslm1, half, dt):
    """flx_v interior on a (vx.shape minus the k boundaries) block.

    c2w     : C2WfactGz[..., 6]  sliced on the interior k-rows (ksl)
    vx/vy/vz: rhogv{x,y,z}_mean   (full k; sliced inside)
    rgamh/rgsqrth/rw: VMTR_RGAMH / VMTR_RGSQRTH / rhogw_mean (full k; sliced)
    ksl     : interior k-slice (kmin+1 .. kmax)
    kslm1   : ksl shifted down by one (kmin .. kmax-1)
    """
    horiz = (
        c2w[..., 0] * vx[..., ksl, :] + c2w[..., 1] * vx[..., kslm1, :] +
        c2w[..., 2] * vy[..., ksl, :] + c2w[..., 3] * vy[..., kslm1, :] +
        c2w[..., 4] * vz[..., ksl, :] + c2w[..., 5] * vz[..., kslm1, :]
    ) * rgamh[..., ksl, :]
    vert = rw[..., ksl, :] * rgsqrth[..., ksl, :]
    return (horiz + vert) * half * dt


def compute_tracer_vert_flux(
    rhogvx_mean, rhogvy_mean, rhogvz_mean, rhogw_mean,
    rhogvx_mean_pl, rhogvy_mean_pl, rhogvz_mean_pl, rhogw_mean_pl,
    rhog_in, rhog_in_pl, frhog, frhog_pl,
    C2WfactGz, RGAMH, RGSQRTH, C2WfactGz_pl, RGAMH_pl, RGSQRTH_pl,
    rdgz, dt, b1, cfg: TracerVertFluxCfg, xp,
):
    """Pure version of the tracer-independent part of the 1st vertical step.

    Returns (flx_v, ck, d, rhog, flx_v_pl, ck_pl, d_pl, rhog_pl). The _pl
    outputs are shape-correct zeros when not have_pl.
    """
    kmin, kmax = cfg.kmin, cfg.kmax
    kp1, kmaxp1, kmaxp2 = kmin + 1, kmax + 1, kmax + 2
    dtype = rhog_in.dtype
    half = xp.asarray(0.5, dtype=dtype)

    i, j, kall, l = rhog_in.shape
    ksl = slice(kp1, kmaxp1)        # kmin+1 .. kmax
    kslm1 = slice(kmin, kmax)       # kmin   .. kmax-1
    kc = slice(kmin, kmaxp1)        # kmin   .. kmax
    kcp1 = slice(kp1, kmaxp2)       # kmin+1 .. kmax+1
    rdgz_c = rdgz[kc][None, None, :, None]

    # --- flx_v (interior kmin+1..kmax, zeros elsewhere incl. kmin & kmax+1) ---
    flx_int = _vert_flux_interior(
        C2WfactGz[:, :, ksl, :, :], rhogvx_mean, rhogvy_mean, rhogvz_mean,
        RGAMH, RGSQRTH, rhogw_mean, ksl, kslm1, half, dt,
    )
    z_lo = xp.zeros((i, j, kp1, l), dtype=dtype)            # k = 0 .. kmin
    z_hi = xp.zeros((i, j, kall - kmaxp1, l), dtype=dtype)  # k = kmax+1 .. end
    flx_v = xp.concatenate([z_lo, flx_int, z_hi], axis=2)

    # --- ck (k = kmin..kmax; zero on kmin-1, kmax+1 and the unread rest) ---
    rin_c = rhog_in[:, :, kc, :]
    ck0_int = -flx_v[:, :, kc, :] / rin_c * rdgz_c
    ck1_int = flx_v[:, :, kcp1, :] / rin_c * rdgz_c
    ck_lo = xp.zeros((i, j, kmin, l), dtype=dtype)              # k = 0 .. kmin-1
    ck_hi = xp.zeros((i, j, kall - kmaxp1, l), dtype=dtype)     # k = kmax+1 .. end
    ck0 = xp.concatenate([ck_lo, ck0_int, ck_hi], axis=2)
    ck1 = xp.concatenate([ck_lo, ck1_int, ck_hi], axis=2)
    ck = xp.stack([ck0, ck1], axis=-1)

    # --- d (all k, pure elementwise) ---
    d = b1 * frhog / rhog_in * dt

    # --- rhog (interior kmin..kmax; boundary rows copied from rhog_in) ---
    rhog_int = (
        rhog_in[:, :, kc, :]
        - (flx_v[:, :, kcp1, :] - flx_v[:, :, kc, :]) * rdgz_c
        + b1 * frhog[:, :, kc, :] * dt
    )
    r_lo = xp.zeros((i, j, kmin - 1, l), dtype=dtype)          # k = 0 .. kmin-2
    r_hi = xp.zeros((i, j, kall - kmaxp2, l), dtype=dtype)     # k = kmax+2 .. end
    rhog = xp.concatenate([
        r_lo,
        rhog_in[:, :, kmin:kp1, :],     # row kmin-1 = rhog_in[kmin]
        rhog_int,                        # rows kmin..kmax
        rhog_in[:, :, kmax:kmaxp1, :],  # row kmax+1 = rhog_in[kmax]
        r_hi,
    ], axis=2)

    # --- pole region ---
    flx_v_pl = xp.zeros_like(rhog_in_pl)
    ck_pl = xp.zeros(rhog_in_pl.shape + (2,), dtype=dtype)
    d_pl = xp.zeros_like(rhog_in_pl)
    rhog_pl = xp.zeros_like(rhog_in_pl)
    if cfg.have_pl:
        g, kall_pl, l_pl = rhog_in_pl.shape
        rdgz_cp = rdgz[kc][None, :, None]

        flxp_int = _vert_flux_interior(
            C2WfactGz_pl[:, ksl, :, :], rhogvx_mean_pl, rhogvy_mean_pl, rhogvz_mean_pl,
            RGAMH_pl, RGSQRTH_pl, rhogw_mean_pl, ksl, kslm1, half, dt,
        )
        zp_lo = xp.zeros((g, kp1, l_pl), dtype=dtype)
        zp_hi = xp.zeros((g, kall_pl - kmaxp1, l_pl), dtype=dtype)
        flx_v_pl = xp.concatenate([zp_lo, flxp_int, zp_hi], axis=1)

        rinp_c = rhog_in_pl[:, kc, :]
        ck0p_int = -flx_v_pl[:, kc, :] / rinp_c * rdgz_cp
        ck1p_int = flx_v_pl[:, kcp1, :] / rinp_c * rdgz_cp
        ckp_lo = xp.zeros((g, kmin, l_pl), dtype=dtype)
        ckp_hi = xp.zeros((g, kall_pl - kmaxp1, l_pl), dtype=dtype)
        ck0p = xp.concatenate([ckp_lo, ck0p_int, ckp_hi], axis=1)
        ck1p = xp.concatenate([ckp_lo, ck1p_int, ckp_hi], axis=1)
        ck_pl = xp.stack([ck0p, ck1p], axis=-1)

        d_pl = b1 * frhog_pl / rhog_in_pl * dt

        rhogp_int = (
            rhog_in_pl[:, kc, :]
            - (flx_v_pl[:, kcp1, :] - flx_v_pl[:, kc, :]) * rdgz_cp
            + b1 * frhog_pl[:, kc, :] * dt
        )
        rp_lo = xp.zeros((g, kmin - 1, l_pl), dtype=dtype)
        rp_hi = xp.zeros((g, kall_pl - kmaxp2, l_pl), dtype=dtype)
        rhog_pl = xp.concatenate([
            rp_lo,
            rhog_in_pl[:, kmin:kp1, :],
            rhogp_int,
            rhog_in_pl[:, kmax:kmaxp1, :],
            rp_hi,
        ], axis=1)

    return flx_v, ck, d, rhog, flx_v_pl, ck_pl, d_pl, rhog_pl
