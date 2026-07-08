#
# SIMPLE_PHYSICS (DCMIP simple-physics package) -- numpy port
#
# Ported from: nicamdc/src/nhm/share/dcmip/simple_physics_v6.f90
#   module mod_simple_physics / subroutine SIMPLE_PHYSICS (v6, 2016-04-26)
#
# Self-contained column physics (NO NICAM module dependencies in the Fortran):
#   large-scale precipitation + bulk surface fluxes + boundary-layer mixing,
#   time-split in that order, partially-implicit for stability.
#   Model levels are TOP-DOWN (level 0 = uppermost full level).
#
# numpy-first: operates on (pcols, pver) arrays with plain numpy. A JAX/xp
# variant can be layered on later (keep the math backend-agnostic -- no
# in-place scatter that numpy/jnp handle differently; return new arrays).
#
# Fortran interface (simple_physics_v6.f90:98):
#   SUBROUTINE SIMPLE_PHYSICS(pcols, pver, dtime, lat, t, q, u, v,
#                             pmid, pint, pdel, rpdel, ps, precl,
#                             test, RJ2012_precip, TC_PBL_mod, use_HS, MITC_TYPE)
#
import numpy as np


def _constants(use_HS=False):
    """Physical + scheme constants (simple_physics_v6.f90 L234-272).

    Returned as a dict so every helper draws from one place. `C` (the sensible-
    heat/evaporation drag coefficient) is x4 larger under moist Held-Suarez
    (Thacher & Jablonowski 2016 Sec.23).
    """
    rair = 287.0
    rh2o = 461.5
    pi   = 4.0 * np.arctan(1.0)
    eta0 = 0.252
    C = 0.0011 * (4.0 if use_HS else 1.0)
    return dict(
        gravit=9.80616, rair=rair, cpair=1.0045e3, latvap=2.5e6, rh2o=rh2o,
        epsilo=rair / rh2o, zvir=(rh2o / rair) - 1.0,
        a=6371220.0, omega=7.29212e-5, pi=pi,
        C=C, SST_TC=302.15, T0=273.16, e0=610.78, rhow=1000.0,
        Cd0=0.0007, Cd1=0.000065, Cm=0.002, v20=20.0,
        p0=100000.0, pbltop=85000.0, zpbltop=1000.0, pblconst=10000.0,
        T00=288.0, u0=35.0, latw=2.0 * pi / 9.0,
        eta0=eta0, etav=(1.0 - eta0) * 0.5 * pi, q0=0.021, kappa=0.4,
    )


def _large_scale_precip(t, q, pmid, pdel, dtime, C):
    """Large-scale condensation & precipitation (simple_physics_v6.f90 L354-388).

    Reed-Jablonowski (2012) scheme: where a level is supersaturated (q>qsat),
    condense the excess with a partially-implicit adjustment, release latent
    heat into t, remove it from q, and accumulate the column precip rate.

    All (i,k) are independent -> fully vectorized. precl is the vertical
    integral of the condensation (the v3 bug-fix: integrate, don't single-level).

    Args:
      t,q   : (pcols,pver) temperature[K], specific humidity[kg/kg], top-down
      pmid  : (pcols,pver) mid-level pressure[Pa]
      pdel  : (pcols,pver) layer thickness[Pa]
      dtime : timestep[s]
      C     : constants dict from _constants()
    Returns:
      t,q (updated), precl (pcols,) precipitation rate [m/s]
    """
    epsilo = C["epsilo"]; e0 = C["e0"]; latvap = C["latvap"]; rh2o = C["rh2o"]
    T0 = C["T0"]; cpair = C["cpair"]; rair = C["rair"]
    gravit = C["gravit"]; rhow = C["rhow"]

    # saturation specific humidity (Clausius-Clapeyron form used by the scheme)
    qsat = epsilo * e0 / pmid * np.exp(-latvap / rh2o * ((1.0 / t) - 1.0 / T0))

    # condensation rate tmp [1/s * kg/kg], only where supersaturated
    denom = 1.0 + (latvap / cpair) * (epsilo * latvap * qsat / (rair * t * t))
    tmp = np.where(q > qsat, (1.0 / dtime) * (q - qsat) / denom, 0.0)

    # update fields (dtdt = latvap/cpair*tmp, dqdt = -tmp)
    t = t + (latvap / cpair) * tmp * dtime
    q = q - tmp * dtime

    # column precip rate = sum_k tmp*pdel/(gravit*rhow)   [m/s]
    precl = np.sum(tmp * pdel / (gravit * rhow), axis=1)

    return t, q, precl


def simple_physics(
    pcols, pver, dtime, lat,
    t, q, u, v,                       # [INOUT] state at model levels (updated in place / returned)
    pmid, pint, pdel, rpdel, ps,      # [IN] pressure structure (Pa) + surface pressure
    test,                             # [IN] SST setting: 0=const (TC), 1=lat-dependent (moist baroclinic)
    RJ2012_precip=True,               # [IN] Reed-Jablonowski (2012) precip scheme on/off
    TC_PBL_mod=False,                 # [IN] George Bryan PBL mod for TC test
    use_HS=False,                     # [IN] Held-Suarez coupling flag
    MITC_TYPE=1,                      # [IN] SST type (1: default TJ2016)
):
    """DCMIP simple-physics for one block of columns.

    Inputs
      pcols, pver : number of columns, number of model levels
      dtime       : physics timestep [s]
      lat         : latitude [rad], shape (pcols,)
      t,q,u,v     : temperature[K], specific humidity[gm/gm], zonal/meridional wind[m/s],
                    shape (pcols, pver), top-down
      pmid,pint,pdel,rpdel : mid/interface pressure, layer thickness, 1/thickness [Pa]
      ps          : surface pressure [Pa], shape (pcols,)

    Returns
      t,q,u,v (updated), precl (large-scale precip rate [m/s], shape (pcols,))

    Sub-steps (Fortran order -- keep this order for bit-comparability):
      1. Large-scale condensation / RJ2012 precipitation   (simple_physics_v6.f90 ~L150-260)
      2. Surface fluxes (bulk aerodynamic, SST-dependent)   (~L270-380)
      3. Boundary-layer vertical diffusion (implicit)       (~L390-520)
    """
    C = _constants(use_HS)

    # --- 1. Large-scale condensation & precipitation (RJ2012) ---
    if RJ2012_precip:
        t, q, precl = _large_scale_precip(t, q, pmid, pdel, dtime, C)
    else:
        precl = np.zeros(pcols, dtype=t.dtype)

    # --- 2. Surface fluxes  &  3. PBL vertical diffusion  (PENDING) ---
    # Next helpers: _pbl_coeffs -> _surface_flux -> _pbl_diffusion.
    raise NotImplementedError(
        "simple_physics: large-scale precip done; surface flux + PBL pending. "
        "precl is final after step 1 -- validate it with tools/dcmip/test_precip.py."
    )
