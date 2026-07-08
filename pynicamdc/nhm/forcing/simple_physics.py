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


def _hydrostatic_za(t, q, ps, pint, C):
    """Hydrostatic height of the lowest midpoint + interface-height init
    (simple_physics_v6.f90 L282-286).

    Uses the INITIAL t,q (this runs before the precip step in the Fortran).
    Index map: Fortran t(:,pver)->t[:,-1]; pint(:,pver)->pint[:,pver-1]
    (the interface just above the surface); ps == pint(:,pver+1) == pint[:,pver].
    zi(:,pver+1)=0 -> zi[:,pver]=0 (surface height datum).

    Returns:
      za : (pcols,) height of lowest full level [m]
      zi : (pcols,pver+1) interface heights, all zero except set by the Bryan PBL
    """
    rair = C["rair"]; gravit = C["gravit"]; zvir = C["zvir"]
    pcols, pver = t.shape
    dlnpint = np.log(ps) - np.log(pint[:, pver - 1])
    za = rair / gravit * t[:, -1] * (1.0 + zvir * q[:, -1]) * 0.5 * dlnpint
    zi = np.zeros((pcols, pver + 1), dtype=t.dtype)   # zi[:,pver] = 0 (surface)
    return za, zi


def _pbl_coeffs(u, v, t, q, pint, za, zi, TC_PBL_mod, C):
    """Turbulent eddy diffusivities Km,Ke + wind speed and drag Cd
    (simple_physics_v6.f90 L400-439).

    Uses POST-precip t,q but the (still unmodified) lowest-level u,v.
    Km,Ke are interface arrays (pcols,pver+1); only slots 0..pver-1 (Fortran
    1..pver) are filled -- the pver-th slot is never read by the diffusion.

    Two configs:
      RJ2012 (default): pressure-based, fully vectorized over levels.
      Bryan (TC_PBL_mod): height-based; zi has a downward recurrence
        (Fortran k=pver..1) so it stays a k-loop, vectorized over columns.

    Returns: wind (pcols,), Cd (pcols,), Km (pcols,pver+1), Ke (pcols,pver+1)
    """
    rair = C["rair"]; gravit = C["gravit"]; zvir = C["zvir"]
    kappa = C["kappa"]; Cval = C["C"]
    Cd0 = C["Cd0"]; Cd1 = C["Cd1"]; Cm = C["Cm"]; v20 = C["v20"]
    pbltop = C["pbltop"]; zpbltop = C["zpbltop"]; pblconst = C["pblconst"]
    pcols, pver = t.shape

    # wind magnitude at the lowest level (Fortran u(:,pver),v(:,pver))
    wind = np.sqrt(u[:, -1] ** 2 + v[:, -1] ** 2)
    Cd = np.where(wind < v20, Cd0 + Cd1 * wind, Cm)

    Km = np.zeros((pcols, pver + 1), dtype=t.dtype)
    Ke = np.zeros((pcols, pver + 1), dtype=t.dtype)

    if TC_PBL_mod:
        # Bryan: zi(k)=zi(k+1)+... downward. Fortran k=pver..1 -> python kp=pver-1..0.
        # At kp=pver-1, zi[:,kp+1]=zi[:,pver]=0 and pint[:,kp+1]=pint[:,pver]=ps.
        for kp in range(pver - 1, -1, -1):
            dlnpint = np.log(pint[:, kp + 1]) - np.log(pint[:, kp])
            zi[:, kp] = zi[:, kp + 1] + rair / gravit * t[:, kp] * (1.0 + zvir * q[:, kp]) * dlnpint
            below = zi[:, kp] <= zpbltop
            fac = 1.0 - zi[:, kp] / zpbltop
            base = wind * zi[:, kp] * fac * fac
            Km[:, kp] = np.where(below, kappa * np.sqrt(Cd) * base, 0.0)
            Ke[:, kp] = np.where(below, kappa * np.sqrt(Cval) * base, 0.0)
    else:
        # RJ2012: vectorized. Fortran pint(:,1..pver) -> pint[:,0:pver]
        # (excludes the surface interface pint[:,pver]).
        pint_k = pint[:, 0:pver]
        decay = np.where(pint_k >= pbltop, 1.0,
                         np.exp(-(pbltop - pint_k) ** 2 / pblconst ** 2))
        Km[:, 0:pver] = (Cd * wind * za)[:, None] * decay
        Ke[:, 0:pver] = (Cval * wind * za)[:, None] * decay

    return wind, Cd, Km, Ke


def _sst_tsurf(lat, test, use_HS, MITC_TYPE, C):
    """Sea-surface temperature Tsurf(lat) (simple_physics_v6.f90 L296-335).

    Depends only on latitude + constants (no state), so timing is irrelevant.
      use_HS   : moist Held-Suarez, MITC_TYPE 1/2/3 Gaussian-in-lat SST.
      test==0  : constant SST_TC (tropical cyclone).
      test==1  : latitude-dependent SST (moist baroclinic wave).
    Returns Tsurf (pcols,).
    """
    pi = C["pi"]
    if use_HS:
        if MITC_TYPE == 1:
            dphi2 = (26.0 / 180.0 * pi) ** 2
            return 29.0 * np.exp(-0.5 * lat * lat / dphi2) + 271.0
        elif MITC_TYPE == 2:
            dphi2 = (20.0 / 180.0 * pi) ** 2
            dphi0 = (17.0 / 180.0 * pi)
            return 29.0 * np.exp(-0.5 * np.maximum(np.abs(lat) - dphi0, 0.0) ** 2 / dphi2) + 271.0
        elif MITC_TYPE == 3:
            dphi2 = (23.0 / 180.0 * pi) ** 2
            dphi0 = (12.0 / 180.0 * pi)
            return 32.0 * np.exp(-0.5 * np.maximum(np.abs(lat) - dphi0, 0.0) ** 2 / dphi2) + 271.0
        else:
            raise ValueError(f"MITC_TYPE out of range: {MITC_TYPE}")

    if test == 1:   # moist baroclinic wave
        T00 = C["T00"]; u0 = C["u0"]; rair = C["rair"]; etav = C["etav"]
        a = C["a"]; omega = C["omega"]; zvir = C["zvir"]; q0 = C["q0"]; latw = C["latw"]
        sinl = np.sin(lat); cosl = np.cos(lat)
        termA = (-2.0 * sinl**6 * (cosl**2 + 1.0/3.0) + 10.0/63.0) * u0 * (np.cos(etav))**1.5
        termB = (8.0/5.0 * cosl**3 * (sinl**2 + 2.0/3.0) - pi/4.0) * a * omega * 0.5
        pref = pi * u0 / rair * 1.5 * np.sin(etav) * (np.cos(etav))**0.5
        return (T00 + pref * (termA + termB)) / (1.0 + zvir * q0 * np.exp(-(lat/latw)**4))

    # test == 0: tropical cyclone, constant SST
    return np.full_like(lat, C["SST_TC"])


def _surface_flux(u, v, t, q, ps, Tsurf, wind, Cd, za, dtime, C):
    """Implicit bulk surface fluxes at the lowest level (v6 L449-459).

    Reed-Jablonowski (2012) implicit update of u,v,t,q at the bottom level
    (Fortran index pver -> python -1) only; all other levels untouched. wind
    and Cd were computed from the pre-flux lowest-level u,v in _pbl_coeffs.

    Returns u,v,t,q (bottom level updated).
    """
    epsilo = C["epsilo"]; e0 = C["e0"]; latvap = C["latvap"]; rh2o = C["rh2o"]
    T0 = C["T0"]; Cval = C["C"]

    qsats = epsilo * e0 / ps * np.exp(-latvap / rh2o * ((1.0 / Tsurf) - 1.0 / T0))

    denom_m = 1.0 + Cd * wind * dtime / za          # momentum drag
    u[:, -1] = u[:, -1] / denom_m
    v[:, -1] = v[:, -1] / denom_m

    denom_h = 1.0 + Cval * wind * dtime / za        # sensible heat / evaporation
    t[:, -1] = (t[:, -1] + Cval * wind * Tsurf * dtime / za) / denom_h
    q[:, -1] = (q[:, -1] + Cval * wind * qsats * dtime / za) / denom_h

    return u, v, t, q


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

    # --- setup: lowest-level height (INITIAL t,q) + SST ---
    za, zi = _hydrostatic_za(t, q, ps, pint, C)
    Tsurf = _sst_tsurf(lat, test, use_HS, MITC_TYPE, C)

    # --- 1. Large-scale condensation & precipitation (RJ2012) ---
    if RJ2012_precip:
        t, q, precl = _large_scale_precip(t, q, pmid, pdel, dtime, C)
    else:
        precl = np.zeros(pcols, dtype=t.dtype)

    # --- 2. Turbulent diffusivities (post-precip t,q; pre-flux u,v) ---
    wind, Cd, Km, Ke = _pbl_coeffs(u, v, t, q, pint, za, zi, TC_PBL_mod, C)

    # --- 3. Implicit surface fluxes at the lowest level ---
    u, v, t, q = _surface_flux(u, v, t, q, ps, Tsurf, wind, Cd, za, dtime, C)

    # --- 4. PBL vertical diffusion (implicit tridiagonal)  (PENDING) ---
    raise NotImplementedError(
        "simple_physics: precip + PBL coeffs + surface flux done; PBL diffusion "
        "(tridiagonal) pending. bit-exact validation of the whole PBL chain "
        "lands at config D once _pbl_diffusion is in."
    )
