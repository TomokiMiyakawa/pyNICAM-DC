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
    raise NotImplementedError(
        "simple_physics: scaffold only. Port simple_physics_v6.f90 sub-steps "
        "(large-scale precip -> surface flux -> PBL mixing). See module header."
    )
