import numpy as np

# ---------------------------------------------------------------------------
# DCMIP2012 initial-conditions test suite (Ullrich, Jablonowski et al. 2012).
# Ported from nicamdc dcmip_initial_conditions_test_1_2_3_v5.f90.
#
# NOTE: these are the test suite's OWN hardcoded constants (NOT the model
# cnstparam) -- kept verbatim for bit-exactness with the Fortran reference.
# ---------------------------------------------------------------------------
A_DCMIP  = 6371220.0                 # Earth's radius [m]
RD_DCMIP = 287.0                     # dry gas constant [J kg^-1 K^-1]
G_DCMIP  = 9.80616                   # gravity [m s^-2]
CP_DCMIP = 1004.5                    # specific heat cp [J kg^-1 K^-1]
PI_DCMIP = 3.141592653589793238      # pi
P0_DCMIP = 100000.0                  # reference pressure [Pa]


def test3_gravity_wave(lon, lat, z, rdtype=np.float64):
    """DCMIP test 3-1: non-hydrostatic gravity wave (zcoords=1: height given, p diagnosed).

    Vectorized -- lon/lat/z are arrays of a common broadcastable shape.
    Returns (p, u, v, w, t, rho); ps/phis/q are 0 and not returned.
    """
    a   = rdtype(A_DCMIP);  Rd = rdtype(RD_DCMIP); g = rdtype(G_DCMIP)
    cp  = rdtype(CP_DCMIP);  pi = rdtype(PI_DCMIP); p0 = rdtype(P0_DCMIP)

    X       = rdtype(125.0)              # reduced-Earth factor (hardcoded in the test)
    Om      = rdtype(0.0)                # no rotation
    as_     = a / X                      # small-Earth radius
    u0      = rdtype(20.0)
    Teq     = rdtype(300.0)
    peq     = rdtype(100000.0)
    lambdac = rdtype(2.0) * pi / rdtype(3.0)
    d       = rdtype(5000.0)
    phic    = rdtype(0.0)
    delta_theta = rdtype(1.0)
    Lz      = rdtype(20000.0)
    N       = rdtype(0.01)
    N2      = N * N
    bigG    = (g * g) / (N2 * cp)

    two = rdtype(2.0); four = rdtype(4.0); one = rdtype(1.0)

    u = u0 * np.cos(lat)
    v = np.zeros_like(u)
    w = np.zeros_like(u)

    TS = bigG + (Teq - bigG) * np.exp(
        -(u0 * N2 / (four * g * g)) * (u0 + two * Om * as_) * (np.cos(two * lat) - one))

    ps = ( peq * np.exp((u0 / (four * bigG * Rd)) * (u0 + two * Om * as_) * (np.cos(two * lat) - one))
           * (TS / Teq) ** (cp / Rd) )

    height = z
    p = ps * ((bigG / TS) * np.exp(-N2 * height / g) + one - (bigG / TS)) ** (cp / Rd)

    t_mean = bigG * (one - np.exp(N2 * height / g)) + TS * np.exp(N2 * height / g)
    rho = p / (Rd * t_mean)

    sin_tmp = np.sin(lat) * np.sin(phic)
    cos_tmp = np.cos(lat) * np.cos(phic)
    r = as_ * np.arccos(sin_tmp + cos_tmp * np.cos(lon - lambdac))
    s = (d * d) / (d * d + r * r)
    theta_pert = delta_theta * s * np.sin(two * pi * height / Lz)
    t_pert = theta_pert * (p / p0) ** (Rd / cp)
    t = t_mean + t_pert

    # broadcast u/v/w up to the full (p-shaped) array
    u = np.broadcast_to(u, p.shape).astype(rdtype)
    v = np.broadcast_to(v, p.shape).astype(rdtype)
    w = np.broadcast_to(w, p.shape).astype(rdtype)
    return p, u, v, w, t, rho


def test2_steady_state_mountain(z, rdtype=np.float64):
    """DCMIP test 2-x: steady-state atmosphere at rest over orography (zcoords=1 path:
    height given, p diagnosed). The wind and passive tracer are zero; p/t depend only on
    the height z, so the surface/mountain (zs) only affects ps/phis (not returned here).
    Returns (p, u, v, w, t, rho) with u=v=w=0.
    """
    Rd = rdtype(RD_DCMIP); g = rdtype(G_DCMIP); p0 = rdtype(P0_DCMIP)
    T0 = rdtype(300.0); gamma = rdtype(0.0065)
    exponent = g / (Rd * gamma)
    p = p0 * (rdtype(1.0) - gamma / T0 * z) ** exponent
    t = T0 - gamma * z
    rho = p / (Rd * t)
    zero = np.zeros_like(p)
    return p, zero, zero, zero, t, rho


def tropical_cyclone_test(lon, lat, z, rdtype=np.float64):
    """DCMIP2016 tropical cyclone (Reed & Jablonowski), zcoords=1 (height given).
    Vectorized (no FPI needed for the z path); height>ztrop branches via np.where.
    Returns (u, v, t, thetav, ps, rho, q). Constants are the test's own hardcoded set.
    """
    def R(x): return rdtype(x)
    a = R(6371220.0); Rd = R(287.0); g = R(9.80616); cp = R(1004.5)
    Mvap = R(0.608); pi = R(3.14159265358979); p0 = R(100000.0)
    omega = R(7.29212e-5); rp = R(282000.0); dp = R(1115.0); zp = R(7000.0)
    q0 = R(0.021); gamma = R(0.007); Ts0 = R(302.15); p00 = R(101500.0)
    cen_lat = R(10.0); cen_lon = R(180.0); zq1 = R(3000.0); zq2 = R(8000.0)
    exppr = R(1.5); exppz = R(2.0); ztrop = R(15000.0); qtrop = R(1.0e-11)
    constTv = R(0.608); epsilon = R(1.0e-25)
    one = R(1.0); two = R(2.0); half = R(0.5)
    deg2rad = pi / R(180.0)
    exponent = Rd * gamma / g
    T0 = Ts0 * (one + constTv * q0)
    Ttrop = T0 - gamma * ztrop
    ptrop = p00 * (Ttrop / T0) ** (one / exponent)

    clat = cen_lat * deg2rad; clon = cen_lon * deg2rad
    f = two * omega * np.sin(clat)
    gr = a * np.arccos(np.sin(clat) * np.sin(lat) + np.cos(clat) * np.cos(lat) * np.cos(lon - clon))
    ps = p00 - dp * np.exp(-(gr / rp) ** exppr)

    height = z
    above = height > ztrop
    p_above = ptrop * np.exp(-(g * (height - ztrop)) / (Rd * Ttrop))
    p_below = ((p00 - dp * np.exp(-(gr / rp) ** exppr) * np.exp(-(height / zp) ** exppz))
               * ((T0 - gamma * height) / T0) ** (one / exponent))
    p = np.where(above, p_above, p_below)

    d1 = np.sin(clat) * np.cos(lat) - np.cos(clat) * np.sin(lat) * np.cos(lon - clon)
    d2 = np.cos(clat) * np.sin(lon - clon)
    d = np.maximum(epsilon, np.sqrt(d1 ** 2 + d2 ** 2))
    ufac = d1 / d; vfac = d2 / d

    Tz = T0 - gamma * height
    bracket = (exppz * height * Rd * Tz / (g * zp ** exppz)
               + (one - p00 / dp * np.exp((gr / rp) ** exppr) * np.exp((height / zp) ** exppz)))
    inner = (f * gr / two) ** 2 - exppr * (gr / rp) ** exppr * Rd * Tz / bracket
    # above-tropopause cells set wind=0; avoid sqrt of a (masked) negative there
    vmag = -f * gr / two + np.sqrt(np.where(above, one, inner))
    u = np.where(above, R(0.0), ufac * vmag)
    v = np.where(above, R(0.0), vfac * vmag)

    q_below = q0 * np.exp(-height / zq1) * np.exp(-(height / zq2) ** exppz)
    q = np.where(above, qtrop, q_below)

    t_below = Tz / (one + constTv * q) / (one + exppz * Rd * Tz * height
              / (g * zp ** exppz * (one - p00 / dp * np.exp((gr / rp) ** exppr) * np.exp((height / zp) ** exppz))))
    t = np.where(above, Ttrop, t_below)

    thetav = t * (one + constTv * q) * (p0 / p) ** (Rd / cp)
    rho = p / (Rd * t * (one + constTv * q))
    return u, v, t, thetav, ps, rho, q


# ===========================================================================
# DCMIP2016-13 Klemp et al. supercell test (Ullrich, based on Klemp).
# Ported from nicamdc supercell_test.f90. Two stages:
#   Stage 1 -- supercell_init_background(): a grid-INDEPENDENT precompute of a
#              2D (nphi x nz) reference state on a Chebyshev collocation grid,
#              using spectral diff/integration operators (SVD pseudoinverse).
#   Stage 2 -- supercell_test(): pointwise Lagrange-interpolation sampling of
#              that background onto the icosahedral grid (zcoords=1 path only).
# NOTE: supercell_test.f90 uses its OWN hardcoded constants, incl. a coarser
#       pi = 3.14159265358979 (NOT PI_DCMIP) -- kept verbatim for bit-fidelity.
# ===========================================================================
PI_SC = 3.14159265358979             # supercell_test.f90's pi (coarser than PI_DCMIP)

# --- supercell test-case parameters (module scope; verbatim from the Fortran) ---
_SC_NZ   = 30                         # vertical levels in the init collocation grid
_SC_NPHI = 16                         # meridional points in the init collocation grid
_SC_Z1   = 0.0                        # lower sample altitude [m]
_SC_Z2   = 50000.0                    # upper sample altitude [m]
_SC_X    = 120.0                      # Earth reduction factor (enters IC only via pert_rh)


def _sc_lagrangian_coeffs(x, xs):
    """Lagrangian polynomial cardinal coefficients: coeffs[...,i] = prod_{j!=i}
    (xs-x[j])/(x[i]-x[j]). x is the (npts,) node vector; xs a scalar or ndarray
    (vectorized -> returns xs.shape + (npts,)). No 0/0 hazard (the i==j factor is
    skipped, so a sample coinciding with a node yields the cardinal delta)."""
    x = np.asarray(x); npts = x.shape[0]
    xs_a = np.asarray(xs, dtype=x.dtype)
    coeffs = np.empty(xs_a.shape + (npts,), dtype=x.dtype)
    for i in range(npts):
        ci = np.ones(xs_a.shape, dtype=x.dtype)
        for j in range(npts):
            if i == j:
                continue
            ci = ci * (xs_a - x[j]) / (x[i] - x[j])
        coeffs[..., i] = ci
    return coeffs


def _sc_diff_lagrangian_coeffs(x, xs):
    """Coefficients of the derivative of the Lagrangian fit at scalar xs.
    Faithful port of diff_lagrangian_polynomial_coeffs incl. the imatch
    (xs coincides with a node within 1e-14) special case. Scalar xs only
    (Stage-1 use). Returns (npts,)."""
    x = np.asarray(x); npts = x.shape[0]
    coeffs = np.zeros(npts, dtype=x.dtype)
    imatch = -1
    for i in range(npts):
        if abs(xs - x[i]) < 1.0e-14:
            imatch = i
            break
    if imatch != -1:
        for i in range(npts):
            c = 1.0
            csum = 0.0
            for j in range(npts):
                if j == i or j == imatch:
                    continue
                c = c * (xs - x[j]) / (x[i] - x[j])
                csum = csum + 1.0 / (xs - x[j])
            if i != imatch:
                c = c * (1.0 + (xs - x[imatch]) * csum) / (x[i] - x[imatch])
            else:
                c = c * csum
            coeffs[i] = c
    else:
        coeffs = _sc_lagrangian_coeffs(x, xs).copy()
        for i in range(npts):
            diff = 0.0
            for j in range(npts):
                if i == j:
                    continue
                diff = diff + 1.0 / (xs - x[j])
            coeffs[i] = coeffs[i] * diff
    return coeffs


def _sc_pinv_abs(A, tol):
    """Moore-Penrose pseudoinverse via SVD with the Fortran's ABSOLUTE singular-
    value cutoff (|s|<=tol -> 1/s := 0), matching the DGESVD+DSCAL+DGEMM path.
    The pseudoinverse is sign-invariant, so LAPACK-vs-numpy SVD sign/algorithm
    differences don't matter."""
    U, s, Vt = np.linalg.svd(A)
    sinv = np.where(np.abs(s) <= tol, 0.0, 1.0 / np.where(s == 0.0, 1.0, s))
    return (Vt.T * sinv) @ U.T


def _sc_zonal_velocity(z, lat, rdtype=np.float64):
    """Reference zonal velocity (vectorized). z, lat broadcastable."""
    us = rdtype(30.0); uc = rdtype(15.0); zs = rdtype(5000.0); zt = rdtype(1000.0)
    v1 = us * (z / zs) - uc
    v2 = (rdtype(-4.0) / rdtype(5.0) + rdtype(3.0) * z / zs
          - rdtype(5.0) / rdtype(4.0) * (z ** 2) / (zs ** 2)) * us - uc
    v3 = us - uc
    v = np.where(z <= zs - zt, v1, np.where(np.abs(z - zs) <= zt, v2, v3))
    return v * np.cos(lat)


def _sc_equator_theta(z, rdtype=np.float64):
    theta0 = rdtype(300.0); theta_tr = rdtype(343.0); z_tr = rdtype(12000.0)
    T_tr = rdtype(213.0); g = rdtype(G_DCMIP); cp = rdtype(CP_DCMIP)
    below = theta0 + (theta_tr - theta0) * (z / z_tr) ** rdtype(1.25)
    above = theta_tr * np.exp(g / cp / T_tr * (z - z_tr))
    return np.where(z <= z_tr, below, above)


def _sc_equator_relative_humidity(z, rdtype=np.float64):
    z_tr = rdtype(12000.0)
    below = rdtype(1.0) - rdtype(0.75) * (z / z_tr) ** rdtype(1.25)
    return np.where(z <= z_tr, below, rdtype(0.25))


def _sc_saturation_mixing_ratio(p, T, rdtype=np.float64):
    smr = rdtype(380.0) / p * np.exp(rdtype(17.27) * (T - rdtype(273.0)) / (T - rdtype(36.0)))
    return np.minimum(smr, rdtype(0.014))


def _sc_thermal_perturbation(lon, lat, z, rdtype=np.float64):
    """Thermal perturbation dtheta (vectorized). Uses the FULL radius a and the
    X-scaled horizontal halfwidth pert_rh = 10000*X."""
    a = rdtype(A_DCMIP); pi = rdtype(PI_SC); deg2rad = pi / rdtype(180.0)
    pert_dtheta = rdtype(3.0); pert_lonc = rdtype(0.0); pert_latc = rdtype(0.0)
    pert_rh = rdtype(10000.0) * rdtype(_SC_X); pert_zc = rdtype(1500.0); pert_rz = rdtype(1500.0)
    gr = a * np.arccos(np.sin(pert_latc * deg2rad) * np.sin(lat)
                       + (np.cos(pert_latc * deg2rad) * np.cos(lat) * np.cos(lon - pert_lonc * deg2rad)))
    Rtheta = np.sqrt((gr / pert_rh) ** 2 + ((z - pert_zc) / pert_rz) ** 2)
    pert = pert_dtheta * (np.cos(rdtype(0.5) * pi * Rtheta)) ** 2
    return np.where(Rtheta <= rdtype(1.0), pert, rdtype(0.0))


def supercell_init_background(rdtype=np.float64):
    """Stage 1: build the (nphi x nz) supercell background reference state.
    Grid-independent; runs once. Returns (phicoord, zcoord, thetavyz, exneryz, qveq)."""
    def R(x): return rdtype(x)
    Rd = R(RD_DCMIP); g = R(G_DCMIP); cp = R(CP_DCMIP); p0 = R(P0_DCMIP); pi = R(PI_SC)
    nz = _SC_NZ; nphi = _SC_NPHI
    z1 = R(_SC_Z1); z2 = R(_SC_Z2)
    pseq = R(100000.0)

    # Chebyshev nodes in phi (lat 0..pi/4) and z (z1..z2)
    ii = np.arange(nphi, dtype=rdtype)
    phicoord = -np.cos(ii * pi / R(nphi - 1))
    phicoord = R(0.25) * pi * (phicoord + R(1.0))
    kk = np.arange(nz, dtype=rdtype)
    zcoord = -np.cos(kk * pi / R(nz - 1))
    zcoord = z1 + R(0.5) * (z2 - z1) * (zcoord + R(1.0))

    # d/dphi operator (column i = derivative coeffs at phicoord[i]); zero deriv at pole
    ddphi = np.zeros((nphi, nphi), dtype=rdtype)
    for i in range(nphi):
        ddphi[:, i] = _sc_diff_lagrangian_coeffs(phicoord, phicoord[i])
    ddphi[:, nphi - 1] = R(0.0)
    # d/dz operator (column k = derivative coeffs at zcoord[k])
    ddz = np.zeros((nz, nz), dtype=rdtype)
    for k in range(nz):
        ddz[:, k] = _sc_diff_lagrangian_coeffs(zcoord, zcoord[k])

    # Integration operators via pseudoinverse (absolute 1e-12 cutoff)
    intphi = _sc_pinv_abs(ddphi, R(1.0e-12))
    intz = _sc_pinv_abs(ddz, R(1.0e-12))

    phicoordmat = np.repeat(phicoord[:, None], nz, axis=1)   # (nphi,nz)

    # Sampled equatorial ueq**2 and d/dz(ueq**2), replicated across phi
    ueq_col = _sc_zonal_velocity(zcoord, R(0.0), rdtype) ** 2      # (nz,)
    dueq_col = ddz.T @ ueq_col                                     # (nz,): sum_j ddz[j,k]*ueq_col[j]
    ueq2 = np.broadcast_to(ueq_col[None, :], (nphi, nz)).astype(rdtype)
    dueq2 = np.broadcast_to(dueq_col[None, :], (nphi, nz)).astype(rdtype)

    # Equatorial theta / relative humidity
    thetaeq = _sc_equator_theta(zcoord, rdtype)                    # (nz,)
    H = _sc_equator_relative_humidity(zcoord, rdtype)              # (nz,)
    thetavyz = np.zeros((nphi, nz), dtype=rdtype)
    thetavyz[0, :] = thetaeq

    exnereqs = (pseq / p0) ** (Rd / cp)
    exnereq = np.zeros(nz, dtype=rdtype)
    qveq = np.zeros(nz, dtype=rdtype)

    # Iterate on equatorial profile (12 sweeps)
    for _ in range(12):
        rhs0 = -g / cp / thetavyz[0, :]                           # (nz,)
        exnereq = intz.T @ rhs0                                   # exnereq[k]=sum_j intz[j,k]*rhs0[j]
        exnereq = exnereq + (exnereqs - exnereq[0])
        exnereq[0] = exnereqs
        p = p0 * exnereq ** (cp / Rd)
        T = thetaeq * exnereq
        qvs = _sc_saturation_mixing_ratio(p, T, rdtype)
        qveq = qvs * H
        thetavyz[0, :] = thetaeq * (R(1.0) + R(0.61) * qveq)

    # Iterate on remainder of the domain (thermal-wind balance, 12 sweeps)
    for _ in range(12):
        dztheta = thetavyz @ ddz                                  # (nphi,nz): sum_j theta[i,j]*ddz[j,k]
        rhs = np.sin(R(2.0) * phicoordmat) / (R(2.0) * g) * (ueq2 * dztheta - thetavyz * dueq2)
        irhs = intphi.T @ rhs                                     # (nphi,nz): sum_j intphi[j,i]*rhs[j,k]
        irhs = irhs + (thetavyz[0, :] - irhs[0, :])[None, :]      # Dirichlet BC at equator
        irhs[0, :] = thetavyz[0, :]
        thetavyz = irhs

    # Pressure (Exner) through the remainder of the domain
    rhs = -ueq2 * np.sin(phicoordmat) * np.cos(phicoordmat) / cp / thetavyz
    exneryz = intphi.T @ rhs
    exneryz = exneryz + (exnereq - exneryz[0, :])[None, :]
    exneryz[0, :] = exnereq

    return phicoord, zcoord, thetavyz, exneryz, qveq


def supercell_test(lon, lat, z, bg, pert, rdtype=np.float64):
    """Stage 2: sample the supercell background `bg` (= supercell_init_background
    output) at grid points (lon, lat, z), zcoords=1 (height given). Vectorized:
    lon/lat/z are broadcastable arrays. `pert` != 0 includes the thermal bubble.
    Returns (u, v, t, thetav, rho, q). ps is not returned (the model's
    diag_pressure ignores it, exactly as in tc_init)."""
    def R(x): return rdtype(x)
    Rd = R(RD_DCMIP); cp = R(CP_DCMIP); p0 = R(P0_DCMIP)
    phicoord, zcoord, thetavyz, exneryz, qveq = bg

    nh_lat = np.abs(lat)                                          # northern-hemisphere latitude
    fitz = _sc_lagrangian_coeffs(zcoord, z)                       # (...,nz)
    fitphi = _sc_lagrangian_coeffs(phicoord, nh_lat)             # (...,nphi)

    # Background Exner pressure and virtual potential temperature (2D fit -> point)
    exner = np.einsum('...k,...k->...', fitz, np.einsum('...i,ik->...k', fitphi, exneryz))
    p = p0 * exner ** (cp / Rd)
    thetav = np.einsum('...k,...k->...', fitz, np.einsum('...i,ik->...k', fitphi, thetavyz))
    q = np.einsum('...k,k->...', fitz, qveq)                      # qveq is equatorial (z-only)

    rho = p / (Rd * exner * thetav)

    if pert != 0:
        thetav = thetav + _sc_thermal_perturbation(lon, lat, z, rdtype) * (R(1.0) + R(0.61) * q)

    # Updated (final) pressure from density and modified virtual potential temperature
    p = p0 * (rho * Rd * thetav / p0) ** (cp / (cp - Rd))

    u = _sc_zonal_velocity(z, lat, rdtype)
    v = np.zeros_like(u)
    t = thetav / (R(1.0) + R(0.61) * q) * (p / p0) ** (Rd / cp)
    return u, v, t, thetav, rho, q


# ===========================================================================
# DCMIP2016-2x moist baroclinic wave (Ullrich/Melvin/Staniforth/Jablonowski).
# Ported from nicamdc baroclinic_wave_test.f90. Pure-analytic pointwise (zcoords=1
# path only -> the evaluate_z_temperature 100-iter root-find is dead code).
# Uses the module's own coarse pi (= PI_SC = 3.14159265358979, same as supercell).
# ===========================================================================
def _bw_eval_pt(deep, X, lon, lat, z, rdtype=np.float64):
    """evaluate_pressure_temperature: analytic p and (virtual) t. Vectorized."""
    def R(x): return rdtype(x)
    a = R(A_DCMIP); Rd = R(RD_DCMIP); g = R(G_DCMIP); p0 = R(P0_DCMIP); omega = R(7.29212e-5)
    T0E = R(310.0); T0P = R(240.0); B = R(2.0); Kk = R(3.0); lapse = R(0.005)
    aref = a / X
    T0 = R(0.5) * (T0E + T0P)
    constA = R(1.0) / lapse
    constB = (T0 - T0P) / (T0 * T0P)
    constC = R(0.5) * (Kk + R(2.0)) * (T0E - T0P) / (T0E * T0P)
    constH = Rd * T0 / g
    scaledZ = z / (B * constH)
    tau1 = (constA * lapse / T0 * np.exp(lapse * z / T0)
            + constB * (R(1.0) - R(2.0) * scaledZ ** 2) * np.exp(-scaledZ ** 2))
    tau2 = constC * (R(1.0) - R(2.0) * scaledZ ** 2) * np.exp(-scaledZ ** 2)
    inttau1 = (constA * (np.exp(lapse * z / T0) - R(1.0))
               + constB * z * np.exp(-scaledZ ** 2))
    inttau2 = constC * z * np.exp(-scaledZ ** 2)
    rratio = R(1.0) if deep == 0 else (z + aref) / aref
    inttermT = ((rratio * np.cos(lat)) ** Kk
                - Kk / (Kk + R(2.0)) * (rratio * np.cos(lat)) ** (Kk + R(2.0)))
    t = R(1.0) / (rratio ** 2 * (tau1 - tau2 * inttermT))
    p = p0 * np.exp(-g / Rd * (inttau1 - inttau2 * inttermT))
    return p, t


def _bw_exponential(lon, lat, z, rdtype=np.float64):
    def R(x): return rdtype(x)
    pi = R(PI_SC)
    pertup = R(1.0); pertexpr = R(0.1); pertlon = pi / R(9.0)
    pertlat = R(2.0) * pi / R(9.0); pertz = R(15000.0)
    gcr = R(1.0) / pertexpr * np.arccos(np.sin(pertlat) * np.sin(lat)
                                        + np.cos(pertlat) * np.cos(lat) * np.cos(lon - pertlon))
    taper = np.where(z < pertz, R(1.0) - R(3.0) * z ** 2 / pertz ** 2 + R(2.0) * z ** 3 / pertz ** 3, R(0.0))
    val = pertup * taper * np.exp(-gcr ** 2)
    return np.where(gcr < R(1.0), val, R(0.0))


def _bw_streamfunction(lon, lat, z, rdtype=np.float64):
    def R(x): return rdtype(x)
    pi = R(PI_SC)
    pertu0 = R(0.5); pertr = R(1.0) / R(6.0); pertlon = pi / R(9.0)
    pertlat = R(2.0) * pi / R(9.0); pertz = R(15000.0)
    gcr = R(1.0) / pertr * np.arccos(np.sin(pertlat) * np.sin(lat)
                                     + np.cos(pertlat) * np.cos(lat) * np.cos(lon - pertlon))
    taper = np.where(z < pertz, R(1.0) - R(3.0) * z ** 2 / pertz ** 2 + R(2.0) * z ** 3 / pertz ** 3, R(0.0))
    cospert = np.where(gcr < R(1.0), np.cos(R(0.5) * pi * gcr), R(0.0))
    return -pertu0 * pertr * taper * cospert ** 4


def baroclinic_wave_test(deep, moist, pertt, X, lon, lat, z, rdtype=np.float64):
    """Vectorized DCMIP2016 moist baroclinic wave, zcoords=1 (height given).
    deep=0 shallow, X Earth-scaling. pertt: 0=exponential, 1=streamfunction, else none.
    moist: 1 include q. Returns (u, v, t, thetav, ps, rho, q); ps=p0 (constant)."""
    def R(x): return rdtype(x)
    a = R(A_DCMIP); Rd = R(RD_DCMIP); g = R(G_DCMIP); cp = R(CP_DCMIP); p0 = R(P0_DCMIP)
    omega = R(7.29212e-5); Mvap = R(0.608); pi = R(PI_SC)
    T0E = R(310.0); T0P = R(240.0); B = R(2.0); Kk = R(3.0)
    dxeps = R(1.0e-5)
    moistqlat = R(2.0) * pi / R(9.0); moistqp = R(34000.0); moisttr = R(0.1)
    moistqs = R(1.0e-12); moistq0 = R(0.018)
    Xf = R(X)

    p, t = _bw_eval_pt(deep, Xf, lon, lat, z, rdtype)

    aref = a / Xf; omegaref = omega * Xf
    T0 = R(0.5) * (T0E + T0P); constH = Rd * T0 / g
    constC = R(0.5) * (Kk + R(2.0)) * (T0E - T0P) / (T0E * T0P)
    scaledZ = z / (B * constH)
    inttau2 = constC * z * np.exp(-scaledZ ** 2)
    rratio = R(1.0) if deep == 0 else (z + aref) / aref
    inttermU = (rratio * np.cos(lat)) ** (Kk - R(1.0)) - (rratio * np.cos(lat)) ** (Kk + R(1.0))
    bigU = g / aref * Kk * inttau2 * inttermU * t
    rcoslat = aref * np.cos(lat) if deep == 0 else (z + aref) * np.cos(lat)
    omegarcoslat = omegaref * rcoslat
    u = -omegarcoslat + np.sqrt(omegarcoslat ** 2 + rcoslat * bigU)
    v = np.zeros_like(u)

    if pertt == 0:
        u = u + _bw_exponential(lon, lat, z, rdtype)
    elif pertt == 1:
        u = u - R(1.0) / (R(2.0) * dxeps) * (_bw_streamfunction(lon, lat + dxeps, z, rdtype)
                                             - _bw_streamfunction(lon, lat - dxeps, z, rdtype))
        v = v + R(1.0) / (R(2.0) * dxeps * np.cos(lat)) * (_bw_streamfunction(lon + dxeps, lat, z, rdtype)
                                                           - _bw_streamfunction(lon - dxeps, lat, z, rdtype))
    # pertt == -99 (or other): no perturbation

    rho = p / (Rd * t)                                           # density from virtual temperature
    if moist == 1:
        eta = p / p0
        q = np.where(eta > moisttr,
                     moistq0 * np.exp(-(lat / moistqlat) ** 4) * np.exp(-((eta - R(1.0)) * p0 / moistqp) ** 2),
                     moistqs)
        t = t / (R(1.0) + Mvap * q)                             # virtual -> actual temperature
    else:
        q = np.zeros_like(p)
    thetav = t * (R(1.0) + R(0.61) * q) * (p0 / p) ** (Rd / cp)
    u = np.broadcast_to(u, p.shape).astype(rdtype)
    v = np.broadcast_to(v, p.shape).astype(rdtype)
    return u, v, t, thetav, R(p0), rho, q
