#
# mod_af_dcmip -- DCMIP artificial-forcing driver (glue) -- numpy port
#
# Ported from: nicamdc/src/nhm/forcing/mod_af_dcmip.f90
#   subroutines af_dcmip_init, AF_dcmip
#
# This is the GLUE between the icosahedral prognostic state and the
# self-contained column physics (simple_physics / kessler / Terminator).
# It: builds the pressure structure (pmid/pint/pdel/rpdel) the column
# schemes need, converts the 3D wind vector <-> (u,v) via the GMTR_p
# metric (ix..jz), calls the physics, and returns forcing TENDENCIES
# (fvx,fvy,fvz,fe,fq) + precip that forcing_step integrates.
#
import numpy as np
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_prof import prf
from pynicamdc.nhm.forcing.simple_physics import simple_physics
# from pynicamdc.nhm.forcing.kessler import kessler                 # optional
# from pynicamdc.nhm.forcing.terminator import tendency_Terminator  # optional


class AfDcmip:

    _instance = None

    # --- FORCING_DCMIP_PARAM namelist flags (mod_af_dcmip.f90 header) ---
    SET_RJ2012        = False   # Reed-Jablonowski 2012 config
    USE_Kessler       = False   # Kessler microphysics
    USE_SimpleMicrophys = False # simple_physics package
    USE_ToyChemistry  = False   # Terminator toy chemistry
    SM_Tsurf_type     = 0       # simple_physics 'test' (0=const SST TC, 1=lat SST)
    SM_RJ2012_precip  = True
    SM_TC_PBL_mod     = False

    def __init__(self):
        pass

    def AF_dcmip_init(self, fname_in, rcnf, rdtype):
        """Read FORCING_DCMIP_PARAM, set scheme flags, validate tracer config.

        Fortran: af_dcmip_init (mod_af_dcmip.f90:52). Checks CHEM_TYPE/NCHEM_MAX
        for the toy-chemistry case; may delegate to AF_heldsuarez_init(moist).
        """
        raise NotImplementedError(
            "AF_dcmip_init: scaffold only. Parse FORCING_DCMIP_PARAM + set flags."
        )

    def AF_dcmip(
        self,
        lat, lon,                       # [IN] (gall,) latitude/longitude
        alt, alth,                      # [IN] (gall,kall) geopotential height (z, zh) <- GRD_vz
        rho, pre, tem,                  # [IN] (gall,kall) density, pressure, temperature
        vx, vy, vz,                     # [IN] (gall,kall) 3D wind vector components
        q,                              # [IN] (gall,kall,TRC_VMAX) tracers
        ein,                            # [IN] (gall,kall) internal energy
        pre_sfc,                        # [IN] (gall,) surface pressure  <- BNDCND_pre_sfc
        ix, iy, iz, jx, jy, jz,         # [IN] (gall,) GMTR_p metric for vh<->uv
        dt,                             # [IN] timestep
        cnst, rcnf, rdtype,
    ):
        """Evaluate DCMIP forcing tendencies for one region block.

        Fortran: AF_dcmip (mod_af_dcmip.f90:264), called per l from forcing_step.
        Returns: fvx, fvy, fvz, fe, fq, precip  (forcing tendencies + precip).

        Steps:
          1. vh->uv     : u,v = f(vx,vy,vz, ix..jz)           [cnvvar_vh2uv]
          2. pressure   : pmid=pre; pint/pdel/rpdel from pre + pre_sfc
          3. physics    : simple_physics(...) and/or kessler(...) on columns
          4. uv->vh     : back-project du,dv -> fvx,fvy,fvz    [uv2vh-style]
          5. energy/q   : fe from dT (CVdry), fq from dq
        """
        raise NotImplementedError(
            "AF_dcmip: scaffold only. Wire vh<->uv, pressure build, physics call, "
            "tendency back-projection. See method docstring + mod_af_dcmip.f90."
        )


afdcmip = AfDcmip()
