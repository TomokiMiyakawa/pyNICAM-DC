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
