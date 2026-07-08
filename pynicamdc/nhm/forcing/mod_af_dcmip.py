#
# mod_af_dcmip -- DCMIP artificial-forcing driver (glue) -- numpy port
#
# Ported from: nicamdc/src/nhm/forcing/mod_af_dcmip.f90
#   subroutines af_dcmip_init, AF_dcmip
#
# GLUE between the icosahedral prognostic state and the self-contained column
# physics (simple_physics). Per region it:
#   1. flips bottom-up model levels (kmin..kmax) to top-down (1..vlayer) via
#      kk = kmax-k+1,
#   2. projects the 3D wind vector (vx,vy,vz) onto local east/north (u,v) with
#      the GMTR_p metric (ix..jz),
#   3. builds the pressure structure (pmid/pint/pdel/rpdel) simple_physics needs
#      (pint via hydrostatic log-p interpolation; pint top=0, bottom=pre_sfc),
#   4. calls simple_physics,
#   5. back-projects the updated (u,v) to (vx,vy,vz) and forms forcing
#      TENDENCIES fvx/fvy/fvz/fe/fq = (new-old)/dt (+ precip).
#
# This minimal port implements the USE_SimpleMicrophys path (DCMIP RJ2012 /
# DCMIP2016 with simple-physics). USE_Kessler / USE_ToyChemistry /
# USE_HeldSuarez are separate schemes and raise NotImplementedError if enabled.
#
import numpy as np
from pynicamdc.nhm.forcing.simple_physics import simple_physics

# pyNICAM infrastructure (logging / profiling). Optional so the pure-compute
# glue is unit-testable without the MPI stack; no-ops when unavailable.
try:
    from pynicamdc.share.mod_stdio import std
except Exception:
    std = None
try:
    from pynicamdc.share.mod_prof import prf
except Exception:
    prf = None


def _rapstart(name):
    if prf is not None:
        prf.PROF_rapstart(name, 1)


def _rapend(name):
    if prf is not None:
        prf.PROF_rapend(name, 1)


class AfDcmip:

    _instance = None

    def __init__(self):
        # namelist FORCING_DCMIP_PARAM flags (mod_af_dcmip.f90 L39-46 defaults)
        self.USE_Kessler         = False
        self.USE_SimpleMicrophys = False
        self.SM_Latdepend_SST    = False
        self.SM_LargeScaleCond   = False
        self.SM_PBL_Bryan        = False
        self.USE_ToyChemistry    = False
        self.USE_HeldSuarez      = False
        self.SM_MITC_SST_TYPE    = 1

    def AF_dcmip_init(self, params=None):
        """Resolve scheme flags from the FORCING_DCMIP_PARAM presets
        (mod_af_dcmip.f90 af_dcmip_init L52+).

        params: dict of namelist keys (SET_RJ2012, SET_DCMIP2016_11, ...,
        USE_*, SM_*). None -> Fortran defaults (all off). The SET_* presets
        overwrite the USE_*/SM_* flags exactly as the Fortran does.
        """
        p = dict(params or {})

        # direct flags (may be overridden by a SET_* preset below)
        for k in ("USE_Kessler", "USE_SimpleMicrophys", "SM_Latdepend_SST",
                  "SM_LargeScaleCond", "SM_PBL_Bryan", "USE_ToyChemistry",
                  "USE_HeldSuarez"):
            if k in p:
                setattr(self, k, bool(p[k]))
        if "SM_MITC_SST_TYPE" in p:
            self.SM_MITC_SST_TYPE = int(p["SM_MITC_SST_TYPE"])

        dry = bool(p.get("SET_DCMIP2016_DRY", False))
        lsc = bool(p.get("SET_DCMIP2016_LSC", False))
        nosst = bool(p.get("SET_DCMIP2016_NOSST", False))

        if p.get("SET_RJ2012", False):
            self.USE_Kessler = False; self.USE_SimpleMicrophys = True
            self.SM_Latdepend_SST = True; self.SM_LargeScaleCond = True
            self.USE_ToyChemistry = False
        elif p.get("SET_DCMIP2016_11", False) or p.get("SET_DCMIP2016_12", False):
            # 1-1 (moist baroclinic + terminator) / 1-2 (tropical cyclone)
            self.USE_Kessler = True; self.USE_SimpleMicrophys = True
            self.SM_Latdepend_SST = True; self.SM_LargeScaleCond = False
            self.USE_ToyChemistry = p.get("SET_DCMIP2016_11", False)
            if dry:
                self.USE_Kessler = False
            if lsc:
                self.USE_Kessler = False; self.SM_LargeScaleCond = True
            if nosst:
                self.USE_SimpleMicrophys = False; self.SM_Latdepend_SST = False
        elif p.get("SET_DCMIP2016_13", False):
            self.USE_Kessler = True; self.USE_SimpleMicrophys = False
            self.SM_Latdepend_SST = False

        if std is not None and getattr(std, "io_l", False):
            with open(std.fname_log, 'a') as f:
                print("+++ Module[af_dcmip]/Category[nhm forcing]", file=f)
                print(f"*** USE_Kessler={self.USE_Kessler} "
                      f"USE_SimpleMicrophys={self.USE_SimpleMicrophys} "
                      f"SM_Latdepend_SST={self.SM_Latdepend_SST} "
                      f"SM_LargeScaleCond={self.SM_LargeScaleCond} "
                      f"SM_PBL_Bryan={self.SM_PBL_Bryan} "
                      f"USE_HeldSuarez={self.USE_HeldSuarez}", file=f)
        return

    def AF_dcmip(self, lat, lon, alt, alth, rho, pre, tem, vx, vy, vz, q, ein,
                 pre_sfc, ix, iy, iz, jx, jy, jz, dt, cfg):
        """DCMIP forcing tendencies for one region block (mod_af_dcmip.f90 L264+).

        Arrays are packed per region, BOTTOM-UP with halo levels:
          lat,lon,pre_sfc,ix..jz : (ijdim,)
          alt,alth,rho,pre,tem,vx,vy,vz,ein : (ijdim,kdim)
          q : (ijdim,kdim,ntrc)
        cfg supplies model constants/indices:
          kmin,kmax (0-based inclusive), vlayer, I_QV,I_QC,I_QR, CVW (indexable),
          CVdry, RAIN_TYPE ('DRY'|'WARM').

        Returns fvx,fvy,fvz (ijdim,kdim), fe (ijdim,kdim), fq (ijdim,kdim,ntrc),
        precip (ijdim,).  Tendencies are zero outside kmin..kmax.
        """
        _rapstart('__Forcing_dcmip')

        if self.USE_Kessler:
            raise NotImplementedError("AF_dcmip: USE_Kessler path not ported (see kessler.py).")
        if self.USE_ToyChemistry:
            raise NotImplementedError("AF_dcmip: USE_ToyChemistry path not ported (see Terminator).")
        if self.USE_HeldSuarez:
            raise NotImplementedError("AF_dcmip: USE_HeldSuarez overwrite not ported here.")

        ijdim, kdim = tem.shape
        ntrc = q.shape[2]
        kmin, kmax, vlayer = cfg["kmin"], cfg["kmax"], cfg["vlayer"]
        I_QV = cfg["I_QV"]
        rdtype = tem.dtype

        fvx = np.zeros((ijdim, kdim), dtype=rdtype)
        fvy = np.zeros((ijdim, kdim), dtype=rdtype)
        fvz = np.zeros((ijdim, kdim), dtype=rdtype)
        fe = np.zeros((ijdim, kdim), dtype=rdtype)
        fq = np.zeros((ijdim, kdim, ntrc), dtype=rdtype)
        precip = np.zeros(ijdim, dtype=rdtype)

        if not self.USE_SimpleMicrophys:
            _rapend('__Forcing_dcmip')
            return fvx, fvy, fvz, fe, fq, precip

        # --- top-down views of the active column (Fortran kk = kmax-k+1) ---
        # model[:, kmin:kmax+1] is bottom-up; [::-1] on the level axis -> top-down,
        # so column index c=0 is the top (Fortran k=1, kk=kmax).
        sl = slice(kmin, kmax + 1)
        def td(a):                      # bottom-up (:,kmin:kmax+1) -> top-down (:,vlayer)
            return a[:, sl][:, ::-1]

        tem_td = td(tem); pre_td = td(pre); alt_td = td(alt); alth_td = td(alth)
        vx_td = td(vx); vy_td = td(vy); vz_td = td(vz); ein_td = td(ein)
        qv_old_td = td(q[:, :, I_QV])

        # extracted column state for simple_physics
        t_col = tem_td.copy()
        qvv_col = qv_old_td.copy()
        u_col = vx_td * ix[:, None] + vy_td * iy[:, None] + vz_td * iz[:, None]
        v_col = vx_td * jx[:, None] + vy_td * jy[:, None] + vz_td * jz[:, None]
        pmid_col = pre_td.copy()

        # pressure at interfaces (pcols, vlayer+1). Fortran L437-443:
        #   pint(1)=0 ; pint(vlayer+1)=pre_sfc ;
        #   pint(k)=pre(kk)*exp( log(pre(kk+1)/pre(kk)) * (alth(kk+1)-alt(kk))
        #                        / (alt(kk+1)-alt(kk)) )   for k=2..vlayer
        # In top-down indices c (=k-1): pre(kk)->pre_td[c], pre(kk+1)->pre_td[c-1],
        # alth(kk+1)->alth_td[c-1], alt(kk)->alt_td[c], alt(kk+1)->alt_td[c-1].
        pint_col = np.zeros((ijdim, vlayer + 1), dtype=rdtype)
        pint_col[:, 0] = 0.0
        pint_col[:, vlayer] = pre_sfc
        c = slice(1, vlayer)            # python interface index 1..vlayer-1
        cm1 = slice(0, vlayer - 1)      # c-1
        pint_col[:, c] = pre_td[:, c] * np.exp(
            np.log(pre_td[:, cm1] / pre_td[:, c])
            * (alth_td[:, cm1] - alt_td[:, c]) / (alt_td[:, cm1] - alt_td[:, c]))

        pdel_col = pint_col[:, 1:vlayer + 1] - pint_col[:, 0:vlayer]
        rpdel_col = 1.0 / pdel_col
        ps_col = pre_sfc.astype(rdtype).copy()

        test = 1 if self.SM_Latdepend_SST else 0
        t_col, qvv_col, u_col, v_col, precip2 = simple_physics(
            ijdim, vlayer, dt, lat.astype(rdtype),
            t_col, qvv_col, u_col, v_col,
            pmid_col, pint_col, pdel_col, rpdel_col, ps_col,
            test,
            RJ2012_precip=self.SM_LargeScaleCond,
            TC_PBL_mod=self.SM_PBL_Bryan,
            use_HS=self.USE_HeldSuarez,
            MITC_TYPE=self.SM_MITC_SST_TYPE,
        )

        # --- momentum tendency, back-project uv->vh then (new-old)/dt (L472-478) ---
        fvx_td = (u_col * ix[:, None] + v_col * jx[:, None] - vx_td) / dt
        fvy_td = (u_col * iy[:, None] + v_col * jy[:, None] - vy_td) / dt
        fvz_td = (u_col * iz[:, None] + v_col * jz[:, None] - vz_td) / dt
        fvx[:, sl] = fvx_td[:, ::-1]    # top-down -> bottom-up (reverse back)
        fvy[:, sl] = fvy_td[:, ::-1]
        fvz[:, sl] = fvz_td[:, ::-1]

        # --- tracer + energy tendency (L480-511) ---
        CVdry = cfg["CVdry"]; CVW = cfg["CVW"]
        rain = cfg["RAIN_TYPE"]
        qv_new = qvv_col
        if rain == 'DRY':
            qd = 1.0 - qv_new
            cv = qd * CVdry + qv_new * CVW[I_QV]
        elif rain == 'WARM':
            I_QC, I_QR = cfg["I_QC"], cfg["I_QR"]
            qc = td(q[:, :, I_QC]); qr = td(q[:, :, I_QR])   # unchanged by simple_physics
            qd = 1.0 - qv_new - qc - qr
            cv = qd * CVdry + qv_new * CVW[I_QV] + qc * CVW[I_QC] + qr * CVW[I_QR]
        else:
            raise NotImplementedError(f"AF_dcmip: RAIN_TYPE={rain!r} not supported.")

        fq_qv_td = (qv_new - qv_old_td) / dt
        fe_td = (cv * t_col - ein_td) / dt
        fq[:, sl, I_QV] = fq_qv_td[:, ::-1]
        fe[:, sl] = fe_td[:, ::-1]

        precip[:] = precip2

        _rapend('__Forcing_dcmip')
        return fvx, fvy, fvz, fe, fq, precip


afdcmip = AfDcmip()
