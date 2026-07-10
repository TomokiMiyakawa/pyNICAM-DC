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
