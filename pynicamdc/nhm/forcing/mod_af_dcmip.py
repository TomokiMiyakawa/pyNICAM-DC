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
from pynicamdc.nhm.forcing.kessler import kessler

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
                 pre_sfc, ix, iy, iz, jx, jy, jz, dt, cfg, xp=np):
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

        if self.USE_ToyChemistry:
            raise NotImplementedError("AF_dcmip: USE_ToyChemistry path not ported (see Terminator).")
        if self.USE_HeldSuarez:
            raise NotImplementedError("AF_dcmip: USE_HeldSuarez overwrite not ported here.")

        ijdim, kdim = tem.shape
        ntrc = q.shape[2]
        kmin, kmax, vlayer = cfg["kmin"], cfg["kmax"], cfg["vlayer"]
        I_QV = cfg["I_QV"]
        rdtype = tem.dtype

        sl = slice(kmin, kmax + 1)

        # place interior-level (kmin..kmax) results into a full (ijdim,kdim) array,
        # zero outside sl (functional; replaces the old preallocate + slice-scatter).
        def lev(interior):
            ij = interior.shape[0]
            return xp.concatenate([xp.zeros((ij, kmin), dtype=rdtype), interior,
                                   xp.zeros((ij, kdim - kmax - 1), dtype=rdtype)], axis=1)

        fvx = fvy = fvz = xp.zeros((ijdim, kdim), dtype=rdtype)
        fe = xp.zeros((ijdim, kdim), dtype=rdtype)
        precip = xp.zeros(ijdim, dtype=rdtype)
        fq_cols = [xp.zeros((ijdim, kdim), dtype=rdtype) for _ in range(ntrc)]  # per-tracer

        # --- Kessler warm-rain microphysics (mod_af_dcmip.f90 L369-412) ---
        # BOTTOM-UP (kmin..kmax), operating on DRY mixing ratios. Accumulates
        # into fq[QV/QC/QR], fe and precip (SimpleMicrophys adds on top below).
        if self.USE_Kessler:
            I_QC, I_QR = cfg["I_QC"], cfg["I_QR"]
            CVdry = cfg["CVdry"]; CVW = cfg["CVW"]
            PRE00, Rdry, CPdry = cfg["PRE00"], cfg["Rdry"], cfg["CPdry"]

            qv_m = q[:, sl, I_QV]; qc_m = q[:, sl, I_QC]; qr_m = q[:, sl, I_QR]  # wet
            qd = 1.0 - qv_m - qc_m - qr_m                        # dry fraction
            qvk = qv_m / qd; qck = qc_m / qd; qrk = qr_m / qd    # dry mixing ratios
            rhod = rho[:, sl] * qd                               # dry density
            pk = (pre[:, sl] / PRE00) ** (Rdry / CPdry)          # Exner
            theta_k = tem[:, sl] / pk
            zk = alt[:, sl]

            theta_k, qvk, qck, qrk, precl = kessler(
                theta_k, qvk, qck, qrk, rhod, pk, dt, zk, xp=xp)

            qd2 = 1.0 / (1.0 + qvk + qck + qrk)                  # back to wet
            qvk = qvk * qd2; qck = qck * qd2; qrk = qrk * qd2
            cvk = qd2 * CVdry + qvk * CVW[I_QV] + qck * CVW[I_QC] + qrk * CVW[I_QR]

            fq_cols[I_QV] = fq_cols[I_QV] + lev((qvk - qv_m) / dt)
            fq_cols[I_QC] = fq_cols[I_QC] + lev((qck - qc_m) / dt)
            fq_cols[I_QR] = fq_cols[I_QR] + lev((qrk - qr_m) / dt)
            fe = fe + lev((cvk * theta_k * pk - ein[:, sl]) / dt)
            precip = precip + precl

        if not self.USE_SimpleMicrophys:
            _rapend('__Forcing_dcmip')
            return fvx, fvy, fvz, fe, xp.stack(fq_cols, axis=-1), precip

        # --- top-down views of the active column (Fortran kk = kmax-k+1) ---
        # model[:, kmin:kmax+1] is bottom-up; [::-1] on the level axis -> top-down,
        # so column index c=0 is the top (Fortran k=1, kk=kmax).
        def td(a):                      # bottom-up (:,kmin:kmax+1) -> top-down (:,vlayer)
            return a[:, sl][:, ::-1]

        tem_td = td(tem); pre_td = td(pre); alt_td = td(alt); alth_td = td(alth)
        vx_td = td(vx); vy_td = td(vy); vz_td = td(vz); ein_td = td(ein)
        qv_old_td = td(q[:, :, I_QV])

        # extracted column state for simple_physics (now functional -> no defensive copies)
        t_col = tem_td
        qvv_col = qv_old_td
        u_col = vx_td * ix[:, None] + vy_td * iy[:, None] + vz_td * iz[:, None]
        v_col = vx_td * jx[:, None] + vy_td * jy[:, None] + vz_td * jz[:, None]
        pmid_col = pre_td

        # pressure at interfaces (pcols, vlayer+1): slot 0 = 0, slots 1..vlayer-1 the
        # log-interp (Fortran L437-443), slot vlayer = pre_sfc. Built functionally.
        c = slice(1, vlayer)            # python interface index 1..vlayer-1
        cm1 = slice(0, vlayer - 1)      # c-1
        pint_interior = pre_td[:, c] * xp.exp(
            xp.log(pre_td[:, cm1] / pre_td[:, c])
            * (alth_td[:, cm1] - alt_td[:, c]) / (alt_td[:, cm1] - alt_td[:, c]))
        pint_col = xp.concatenate(
            [xp.zeros((ijdim, 1), dtype=rdtype), pint_interior, pre_sfc[:, None]], axis=1)

        pdel_col = pint_col[:, 1:vlayer + 1] - pint_col[:, 0:vlayer]
        rpdel_col = 1.0 / pdel_col
        ps_col = pre_sfc.astype(rdtype)

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
            xp=xp,
        )

        # --- momentum tendency, back-project uv->vh then (new-old)/dt (L472-478) ---
        fvx_td = (u_col * ix[:, None] + v_col * jx[:, None] - vx_td) / dt
        fvy_td = (u_col * iy[:, None] + v_col * jy[:, None] - vy_td) / dt
        fvz_td = (u_col * iz[:, None] + v_col * jz[:, None] - vz_td) / dt
        fvx = lev(fvx_td[:, ::-1])      # top-down -> bottom-up, scatter into (ijdim,kdim)
        fvy = lev(fvy_td[:, ::-1])
        fvz = lev(fvz_td[:, ::-1])

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
        # accumulate on top of any Kessler contribution
        fq_cols[I_QV] = fq_cols[I_QV] + lev(fq_qv_td[:, ::-1])
        fe = fe + lev(fe_td[:, ::-1])
        precip = precip + precip2

        _rapend('__Forcing_dcmip')
        return fvx, fvy, fvz, fe, xp.stack(fq_cols, axis=-1), precip


afdcmip = AfDcmip()
