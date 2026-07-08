#
# KESSLER (1969) warm-rain microphysics -- numpy port (OPTIONAL)
#
# Ported from: nicamdc/src/nhm/share/dcmip/kessler.f90
#   subroutine KESSLER (self-contained, no NICAM module deps)
#
# Column scheme: autoconversion, accretion, evaporation of rain, and
# sedimentation, operating on (theta, qv, qc, qr, rho, pk) profiles.
# Alternative to simple_physics's condensation when USE_Kessler is set
# in FORCING_DCMIP_PARAM. Optional for the minimal DCMIP port.
#
import numpy as np


def kessler(theta, qv, qc, qr, rho, pk, dt, z, nz):
    """Kessler warm-rain microphysics for a single column.

    Fortran signature (kessler.f90): KESSLER(theta, qv, qc, qr, rho, pk, dt, z, nz)
      theta [INOUT] potential temperature [K]
      qv,qc,qr [INOUT] vapor / cloud water / rain mixing ratios [gm/gm]
      rho [IN] dry air density [kg/m^3]
      pk  [IN] Exner function (p/p0)^(R/cp)
      dt  [IN] timestep [s]; z [IN] heights [m]; nz [IN] levels
    Returns updated theta, qv, qc, qr, and rainnc (accumulated rain).
    """
    raise NotImplementedError(
        "kessler: scaffold only. Port kessler.f90 (autoconv/accretion/evap/sediment)."
    )
