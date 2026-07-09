#
# KESSLER (1969) warm-rain microphysics -- numpy port
#
# Ported from: nicamdc/src/nhm/share/dcmip/kessler.f90  (subroutine KESSLER,
#   Klemp/Skamarock/Park 2015; self-contained, no NICAM module deps).
#
# Column scheme: autoconversion, accretion, saturation adjustment, rain
# evaporation, and rain sedimentation (with per-column CFL sub-cycling),
# operating on (theta, qv, qc, qr, rho, pk) profiles ordered SURFACE -> TOP.
# Used when USE_Kessler is set in FORCING_DCMIP_PARAM (DCMIP2016 1-1/1-2/1-3).
#
# Precision: like the rest of pyNICAM-DC the scheme follows the model working
# precision `wp` (taken from the input dtype), so it runs in float32 in the
# model's fp32 mode and float64 in fp64 mode. Two subtleties reproduce the
# Fortran reference:
#   * The KESSLER dummy arguments are REAL(8) and its 5 working arrays
#     (r, rhalf, velqr, sed, pc) are bare REAL (single precision, since the
#     production nicamdc build carries no -r8/-fdefault-real-8). Those five are
#     therefore cast to `lp` (default float32). In fp64 mode this gives exactly
#     the f64-main / f32-local mix nicamdc uses; in fp32 mode wp==lp==float32
#     so the cast is a no-op. Because each of the five is always combined with a
#     wp array in the expressions that consume it, numpy array-array promotion
#     keeps the intermediates at wp (e.g. f4*f8 -> f8) exactly as Fortran does.
#
import numpy as np


def kessler(theta, qv, qc, qr, rho, pk, dt, z, xp=np, lp=None):
    """Kessler warm-rain microphysics, vectorized over columns.

    Parameters (all SURFACE->TOP along the level axis)
    ----------
    theta, qv, qc, qr : (ncol, nz)  potential temperature [K] and vapor/cloud/
                        rain mixing ratios [gm/gm].  Treated read-only (copied).
                        Their dtype sets the working precision `wp`.
    rho  : (ncol, nz)   dry air density [kg/m^3]
    pk   : (ncol, nz)   Exner function (p/p0)**(R/cp)
    z    : (ncol, nz)   heights of thermodynamic levels [m]
    dt   : float        time step [s]
    xp   : numpy or jax.numpy (numpy-first)
    lp   : dtype for the 5 working arrays r/rhalf/velqr/sed/pc. Defaults to
           float32 (kessler.f90 declares them bare `REAL`); pass wp to force an
           all-working-precision variant.

    Returns
    -------
    theta, qv, qc, qr : (ncol, nz)  updated fields (working precision)
    precl             : (ncol,)     precipitation rate [m_water/s]
    """
    wp = theta.dtype                                  # working precision (f32/f64)
    lp = xp.float32 if lp is None else lp             # local-array precision
    w = lambda x: wp.type(x)                          # wp-typed scalar literal

    # constants (Fortran REAL(8) scalars; here at working precision)
    f2x   = w(17.27)
    f5    = w(237.3) * f2x * w(2500000.0) / w(1003.0)
    xk    = w(0.2875)          # kappa (R/cp)
    psl   = w(1000.0)          # sea-level pressure [mb]
    rhoqr = w(1000.0)          # liquid water density [kg/m^3]

    # writable working-precision copies (INOUT semantics)
    theta = theta.astype(wp).copy()
    qv    = qv.astype(wp).copy()
    qc    = qc.astype(wp).copy()
    qr    = qr.astype(wp).copy()
    rho   = rho.astype(wp)
    pk    = pk.astype(wp)
    z     = z.astype(wp)

    ncol = theta.shape[0]

    # --- level-invariant locals r, rhalf, pc (cast to lp) ---
    r     = (w(0.001) * rho).astype(lp)                          # (ncol,nz)
    rhalf = xp.sqrt(rho[:, 0:1] / rho).astype(lp)               # sqrt(rho(1)/rho(k))
    pc    = (w(3.8) / (pk ** (w(1.0) / xk) * psl)).astype(lp)

    # terminal velocity (KW 2.15), lp
    velqr = (w(36.34) * (qr * r) ** w(0.1364) * rhalf).astype(lp)

    # --- CFL: max stable step per column, then #sub-cycles ---
    dz  = z[:, 1:] - z[:, :-1]                                   # z(k+1)-z(k), k=1..nz-1
    vk  = velqr[:, :-1]                                          # velqr(k), k=1..nz-1
    # Fortran guards the CFL divide with `if (velqr(k)/=0)`; replicate the guard
    # so velqr==0 levels contribute dt (no constraint) without a 0/0 warning.
    vk_safe = xp.where(vk != 0.0, vk, w(1.0))
    cfl = xp.where(vk != 0.0, w(0.8) * dz / vk_safe, w(dt))
    dt_max = xp.minimum(w(dt), cfl.min(axis=1))                 # (ncol,)
    # #sub-cycles: a small count (physically <= a few hundred). int32 is portable
    # to JAX x64-off (fp32 mode); int64 would silently degrade there. Never mixed
    # into float math except via .astype(wp), so its width can't affect results.
    rainsplit = xp.ceil(w(dt) / dt_max).astype(xp.int32)        # (ncol,)
    dt0 = w(dt) / rainsplit.astype(wp)                          # (ncol,)
    dt0c = dt0[:, None]

    precl = xp.zeros(ncol, dtype=wp)
    max_rs = int(rainsplit.max())

    for nt in range(1, max_rs + 1):
        active = (nt <= rainsplit)                              # (ncol,)
        am = active[:, None]

        # precipitation rate (m/s), accumulated per active sub-cycle
        precl = precl + xp.where(active, rho[:, 0] * qr[:, 0] * velqr[:, 0] / rhoqr, w(0.0))

        # sedimentation (upstream differencing), lp
        num = (r[:, 1:] * qr[:, 1:] * velqr[:, 1:]
               - r[:, :-1] * qr[:, :-1] * velqr[:, :-1])
        sed_int = (dt0c * num / (r[:, :-1] * dz)).astype(lp)
        sed_top = (-dt0 * qr[:, -1] * velqr[:, -1]
                   / (w(0.5) * (z[:, -1] - z[:, -2]))).astype(lp)
        sed = xp.concatenate([sed_int, sed_top[:, None]], axis=1)

        # start-of-subcycle state (frozen copy for inactive columns)
        qc0, qr0, qv0, th0 = qc, qr, qv, theta

        # autoconversion + accretion (KW 2.13a,b)
        qrprod = qc0 - (qc0 - dt0c * xp.maximum(w(0.001) * (qc0 - w(0.001)), w(0.0))) \
            / (w(1.0) + dt0c * w(2.2) * qr0 ** w(0.875))
        qc1 = xp.maximum(qc0 - qrprod, w(0.0))
        qr1 = xp.maximum(qr0 + qrprod + sed, w(0.0))

        # saturation vapor mixing ratio (KW 2.11) and condensation production
        pkt = pk * th0
        qvs = pc * xp.exp(f2x * (pkt - w(273.0)) / (pkt - w(36.0)))
        prod = (qv0 - qvs) / (w(1.0) + qvs * f5 / (pkt - w(36.0)) ** 2)

        # rain evaporation (KW 2.14a,b); dim(a,b)=max(a-b,0)
        rqr = r * qr1
        ern = xp.minimum(
            xp.minimum(
                dt0c * (((w(1.6) + w(124.9) * rqr ** w(0.2046)) * rqr ** w(0.525))
                        / (w(2550000.0) * pc / (w(3.8) * qvs) + w(540000.0)))
                * (xp.maximum(qvs - qv0, w(0.0)) / (r * qvs)),
                xp.maximum(-prod - qc1, w(0.0))),
            qr1)

        # saturation adjustment (KW 3.10)
        mpc = xp.maximum(prod, -qc1)
        th_new = th0 + w(2500000.0) / (w(1003.0) * pk) * (mpc - ern)
        qv_new = xp.maximum(qv0 - mpc + ern, w(0.0))
        qc_new = qc1 + mpc
        qr_new = qr1 - ern

        # freeze inactive columns at their start-of-subcycle values
        theta = xp.where(am, th_new, th0)
        qv    = xp.where(am, qv_new, qv0)
        qc    = xp.where(am, qc_new, qc0)
        qr    = xp.where(am, qr_new, qr0)

        # recompute terminal velocity for the next sub-cycle (Fortran: if nt/=rainsplit)
        recompute = (nt < rainsplit)[:, None]
        velqr_new = (w(36.34) * (qr * r) ** w(0.1364) * rhalf).astype(lp)
        velqr = xp.where(recompute, velqr_new, velqr).astype(lp)

    precl = precl / rainsplit.astype(wp)
    return theta, qv, qc, qr, precl
