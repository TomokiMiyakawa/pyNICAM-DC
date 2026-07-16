import toml
import numpy as np
#from mpi4py import MPI
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_prof import prf
from pynicamdc.nhm.forcing.mod_af_dcmip import afdcmip
from pynicamdc.nhm.forcing.mod_af_heldsuarez import afhs


def hs_apply_core(PROG, rho, pre, tem, vx, vy, vz, lat, GSGAM2, dt,
                  cnst, kmin, kmax, rdtype, idx, xp):
    """Pure Held-Suarez forcing apply (NO side effects -> jit-safe). idx =
    (I_RHOGVX, I_RHOGVY, I_RHOGVZ, I_RHOGE). Returns (new_PROG, fvx, fvy, fvz, fe).
    Same float op-order as the eager applier, so numpy/jax/jit all agree."""
    i_vx, i_vy, i_vz, i_e = idx
    fvx, fvy, fvz, fe = afhs.AF_heldsuarez(lat, pre, tem, vx, vy, vz, kmin, kmax, cnst, rdtype, xp=xp)
    z = xp.zeros_like(fvx)
    comps = [z] * PROG.shape[-1]
    comps[i_vx] = dt * fvx * rho * GSGAM2
    comps[i_vy] = dt * fvy * rho * GSGAM2
    comps[i_vz] = dt * fvz * rho * GSGAM2
    comps[i_e]  = dt * fe  * rho * GSGAM2
    return PROG + xp.stack(comps, axis=-1), fvx, fvy, fvz, fe


class Frc:
    
    _instance = None

    nmax_TEND     = 7
    nmax_PROG     = 6
    nmax_v_mean_c = 5

    I_RHOG     = 0  # Density x G^1/2 x gamma^2
    I_RHOGVX   = 1  # Density x G^1/2 x gamma^2 x Horizontal velocity (X-direction)
    I_RHOGVY   = 2  # Density x G^1/2 x gamma^2 x Horizontal velocity (Y-direction)
    I_RHOGVZ   = 3  # Density x G^1/2 x gamma^2 x Horizontal velocity (Z-direction)
    I_RHOGW    = 4  # Density x G^1/2 x gamma^2 x Vertical   velocity
    I_RHOGE    = 5  # Density x G^1/2 x gamma^2 x Internal Energy
    I_RHOGETOT = 6  # Density x G^1/2 x gamma^2 x Total Energy

    # Logical flags
    NEGATIVE_FIXER  = False
    UPDATE_TOT_DENS = True
    
    def __init__(self):
        pass

    def forcing_setup(self, fname_in, rcnf, rdtype):

        self.time = 0.0

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[forcing]/Category[nhm]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'forcing_param' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** forcing_param not found in toml file! Use default.", file=log_file)
                #prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['forcing_param']
            # nicamdc FORCING_PARAM: negative-tracer clamp + total-density update.
            self.NEGATIVE_FIXER  = cnfs.get('NEGATIVE_FIXER',  self.NEGATIVE_FIXER)
            self.UPDATE_TOT_DENS = cnfs.get('UPDATE_TOT_DENS', self.UPDATE_TOT_DENS)

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(f"+++ Artificial forcing type: {rcnf.AF_TYPE.strip()}", file=log_file)


        if rcnf.AF_TYPE == 'NONE':
            # do nothing
            pass

        elif rcnf.AF_TYPE == 'HELD-SUAREZ':
            # Held & Suarez (1994); nicamdc forcing_setup hardcodes moist_case=.false.
            afhs.AF_heldsuarez_init(moist_case=False)

        elif rcnf.AF_TYPE == 'DCMIP':
            # DCMIP artificial forcing: resolve scheme flags from the toml
            # [forcing_dcmip_param] table (SET_*/USE_*/SM_* keys) and hand them
            # to the AF_dcmip driver. The forcing itself runs in forcing_step.
            dcmip_params = cnfs if isinstance(cnfs, dict) else {}
            with open(fname_in, 'r') as _f:
                _all = toml.load(_f)
            if 'forcing_dcmip_param' in _all:
                dcmip_params = _all['forcing_dcmip_param']
            afdcmip.AF_dcmip_init(dcmip_params)

        else:
            print("xxx unsupported forcing type! STOP.")
            prc.prc_mpistop(std.io_l, std.fname_log)

        return
    

    def forcing_update(self,
                       PROG, PROG_pl,
                       cnst, rcnf, grd, tim, trcadv, rdtype,
                       ):
        
        prf.PROF_rapstart('__Forcing',1)

        vx    = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        vx_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        vy    = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        vy_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        vz    = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        vz_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        w     = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        w_pl  = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)

        #--- update velocity
        self.time = self.time + tim.TIME_dtl

        gall_1d= adm.ADM_gall_1d
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        k0 = adm.ADM_K0

        if rcnf.DCTEST_type == 'Traceradvection' and rcnf.DCTEST_case == '1-1':

            trcadv.test11_velocity(self.time,
                                    grd.GRD_LON,
                                    grd.GRD_LAT,
                                    grd.GRD_vz[:,:,:,:,grd.GRD_Z],
                                    grd.GRD_vz[:,:,:,:,grd.GRD_ZH],
                                    vx,
                                    vy,
                                    vz,
                                    w, 
                                    rdtype,
            )

            if adm.ADM_have_pl:      

                trcadv.test11_velocity(self.time,
                                        grd.GRD_LON_pl,
                                        grd.GRD_LAT_pl,
                                        grd.GRD_vz_pl[:,:,:,grd.GRD_Z],
                                        grd.GRD_vz_pl[:,:,:,grd.GRD_ZH],
                                        vx_pl,
                                        vy_pl,
                                        vz_pl,
                                        w_pl, 
                                        rdtype,
                )    

        elif rcnf.DCTEST_type == 'Traceradvection' and rcnf.DCTEST_case == '1-2':

            trcadv.test12_velocity(self.time,
                                    grd.GRD_LON,
                                    grd.GRD_LAT,
                                    grd.GRD_vz[:,:,:,:,grd.GRD_Z],
                                    grd.GRD_vz[:,:,:,:,grd.GRD_ZH],
                                    vx,
                                    vy,
                                    vz,
                                    w,
                                    rdtype,
            )

            if adm.ADM_have_pl:

                trcadv.test12_velocity(self.time,
                                        grd.GRD_LON_pl,
                                        grd.GRD_LAT_pl,
                                        grd.GRD_vz_pl[:,:,:,grd.GRD_Z],
                                        grd.GRD_vz_pl[:,:,:,grd.GRD_ZH],
                                        vx_pl,
                                        vy_pl,
                                        vz_pl,
                                        w_pl,
                                        rdtype,
                )

        PROG[:, :, :, :, self.I_RHOGVX] = vx * PROG[:, :, :, :, self.I_RHOG]
        PROG[:, :, :, :, self.I_RHOGVY] = vy * PROG[:, :, :, :, self.I_RHOG]
        PROG[:, :, :, :, self.I_RHOGVZ] = vz * PROG[:, :, :, :, self.I_RHOG]
        PROG[:, :, :, :, self.I_RHOGW ] = w  * PROG[:, :, :, :, self.I_RHOG]

        if adm.ADM_have_pl:
            PROG_pl[:, :, :, self.I_RHOGVX] = vx_pl * PROG_pl[:, :, :, self.I_RHOG]
            PROG_pl[:, :, :, self.I_RHOGVY] = vy_pl * PROG_pl[:, :, :, self.I_RHOG]
            PROG_pl[:, :, :, self.I_RHOGVZ] = vz_pl * PROG_pl[:, :, :, self.I_RHOG]
            PROG_pl[:, :, :, self.I_RHOGW ] = w_pl  * PROG_pl[:, :, :, self.I_RHOG]

        prf.PROF_rapend  ('__Forcing',1)

        return

    def forcing_step(self, PROG, PROGq, rho, pre, tem, vx, vy, vz, q,
                     vmtr, gmtr, grd, cnst, rcnf, dt, rdtype, xp=np):
        """DCMIP forcing: compute+apply (nicamdc mod_forcing_driver.f90 forcing_step).

        Functional + xp-agnostic. ein=rhoge/rhog, single-layer hydrostatic surface
        pressure, then all (i,j,l) columns are FLATTENED into one block for a single
        AF_dcmip call (bit-exact vs the old per-region loop: AF_dcmip is column-
        independent and Kessler's active-masking makes the rainsplit batch-invariant),
        and the tendencies are applied functionally. RETURNS (PROG, PROGq, precip).
        """
        prf.PROF_rapstart('__Forcing', 1)

        i0, j0 = PROG.shape[0], PROG.shape[1]   # horizontal dims (layout-agnostic)
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        kmin, kmax = adm.ADM_kmin, adm.ADM_kmax
        K0 = adm.ADM_K0
        GRAV = cnst.CONST_GRAV
        ntrc = PROGq.shape[-1]
        Ncol = i0 * j0 * lall

        I_RHOG, I_RHOGVX = self.I_RHOG, self.I_RHOGVX
        I_RHOGVY, I_RHOGVZ = self.I_RHOGVY, self.I_RHOGVZ
        I_RHOGW, I_RHOGE = self.I_RHOGW, self.I_RHOGE

        rhog = PROG[:, :, :, :, I_RHOG]
        ein = PROG[:, :, :, :, I_RHOGE] / rhog                  # (i,j,k,l)

        # geometry / metric views (i,j,l for the horizontal-only fields)
        z = grd.GRD_vz[:, :, :, :, grd.GRD_Z]                   # (i,j,k,l) alt
        zh = grd.GRD_vz[:, :, :, :, grd.GRD_ZH]                 # (i,j,k,l) alth
        z_srf = grd.GRD_zs[:, :, K0, :, grd.GRD_ZSFC]           # (i,j,l)
        lat = grd.GRD_LAT[:, :, K0, :]                          # (i,j,l)
        lon = grd.GRD_LON[:, :, K0, :]
        gp = gmtr.GMTR_p
        ix = gp[:, :, K0, :, gmtr.GMTR_p_IX]; iy = gp[:, :, K0, :, gmtr.GMTR_p_IY]
        iz = gp[:, :, K0, :, gmtr.GMTR_p_IZ]; jx = gp[:, :, K0, :, gmtr.GMTR_p_JX]
        jy = gp[:, :, K0, :, gmtr.GMTR_p_JY]; jz = gp[:, :, K0, :, gmtr.GMTR_p_JZ]

        # surface pressure (nicamdc forcing_step L274-275: single-layer hydrostatic)
        pre_sfc = pre[:, :, kmin, :] + rho[:, :, kmin, :] * GRAV * (z[:, :, kmin, :] - z_srf)

        # tentative negative fixer on the tracers fed to AF_dcmip (nicamdc L279-283)
        if self.NEGATIVE_FIXER:
            q = xp.maximum(q, rdtype(0.0))

        cfg = dict(kmin=kmin, kmax=kmax, vlayer=adm.ADM_vlayer,
                   I_QV=rcnf.I_QV, I_QC=getattr(rcnf, 'I_QC', -1),
                   I_QR=getattr(rcnf, 'I_QR', -1), CVW=rcnf.CVW,
                   CVdry=cnst.CONST_CVdry, RAIN_TYPE=rcnf.RAIN_TYPE,
                   PRE00=cnst.CONST_PRE00, Rdry=cnst.CONST_Rdry,
                   CPdry=cnst.CONST_CPdry,
                   NCHEM_STR=getattr(rcnf, 'NCHEM_STR', -1),
                   NCHEM_END=getattr(rcnf, 'NCHEM_END', -1))

        # --- flatten (i,j,l) columns -> (Ncol, kall); k moved to the end ---
        def colf(a):   # (i,j,k,l) -> (Ncol, kall)
            return a.transpose(0, 1, 3, 2).reshape(Ncol, kall)
        def colt(a):   # (i,j,k,l,ntrc) -> (Ncol, kall, ntrc)
            return a.transpose(0, 1, 3, 2, 4).reshape(Ncol, kall, ntrc)
        def hf(a):     # (i,j,l) -> (Ncol,)
            return a.reshape(Ncol)

        fx, fy, fz, fee, fqq, prc_col = afdcmip.AF_dcmip(
            hf(lat), hf(lon), colf(z), colf(zh), colf(rho), colf(pre), colf(tem),
            colf(vx), colf(vy), colf(vz), colt(q), colf(ein), hf(pre_sfc),
            hf(ix), hf(iy), hf(iz), hf(jx), hf(jy), hf(jz), dt, cfg, xp=xp)

        # --- reshape tendencies back to (i,j,k,l) ---
        def unc(a):    # (Ncol, kall) -> (i,j,k,l)
            return a.reshape(i0, j0, lall, kall).transpose(0, 1, 3, 2)
        fvx = unc(fx); fvy = unc(fy); fvz = unc(fz); fe = unc(fee)
        fq = fqq.reshape(i0, j0, lall, kall, ntrc).transpose(0, 1, 3, 2, 4)
        precip = prc_col.reshape(i0, j0, lall)

        # --- apply tendencies functionally (nicamdc L367-390); fw=0 for DCMIP.
        # Same float op-order as the old in-place +=; RHOG accumulates the (clamped)
        # NQW tracer fluxes in-loop, exactly as nicamdc PROG[RHOG] += frhogq. ---
        GSGAM2 = vmtr.VMTR_GSGAM2
        new_vx = PROG[:, :, :, :, I_RHOGVX] + dt * fvx * rho * GSGAM2
        new_vy = PROG[:, :, :, :, I_RHOGVY] + dt * fvy * rho * GSGAM2
        new_vz = PROG[:, :, :, :, I_RHOGVZ] + dt * fvz * rho * GSGAM2
        new_e  = PROG[:, :, :, :, I_RHOGE] + dt * fe * rho * GSGAM2

        rhog_val = PROG[:, :, :, :, I_RHOG]
        progq_cols = []
        for nq in range(ntrc):
            frhogq = fq[:, :, :, :, nq] * rho * GSGAM2
            if self.NEGATIVE_FIXER:
                tmp = xp.maximum(PROGq[:, :, :, :, nq] + dt * frhogq, rdtype(0.0))
                frhogq = (tmp - PROGq[:, :, :, :, nq]) / dt
                progq_cols.append(tmp)
            else:
                progq_cols.append(PROGq[:, :, :, :, nq] + dt * frhogq)
            if self.UPDATE_TOT_DENS and rcnf.NQW_STR <= nq <= rcnf.NQW_END:
                rhog_val = rhog_val + dt * frhogq
        PROGq = xp.stack(progq_cols, axis=-1)

        slots = [None] * 6
        slots[I_RHOG] = rhog_val
        slots[I_RHOGVX] = new_vx; slots[I_RHOGVY] = new_vy; slots[I_RHOGVZ] = new_vz
        slots[I_RHOGW] = PROG[:, :, :, :, I_RHOGW]        # unchanged (fw=0)
        slots[I_RHOGE] = new_e
        PROG = xp.stack(slots, axis=-1)

        prf.PROF_rapend('__Forcing', 1)
        # pure: return the prognostic + the raw tendencies (caller stashes them for
        # validation/history). No self.* side effects -> the whole thing is jit-safe.
        return PROG, PROGq, precip, fvx, fvy, fvz, fe, fq

    def forcing_step_hs(self, PROG, rho, pre, tem, vx, vy, vz, lat,
                        vmtr, cnst, rcnf, dt, rdtype, xp=np):
        """Held-Suarez forcing: compute the tendencies (Rayleigh friction +
        Newtonian relaxation) and apply them to PROG (nicamdc mod_forcing_driver.f90
        forcing_step, HELD-SUAREZ branch). No tracer forcing (fq=0), no vertical-
        velocity forcing (fw=0). xp = numpy or jax.numpy; RETURNS the updated PROG
        functionally (no in-place mutation, so it is jit/device-safe)."""
        prf.PROF_rapstart('__Forcing', 1)
        kmin, kmax = adm.ADM_kmin, adm.ADM_kmax
        idx = (self.I_RHOGVX, self.I_RHOGVY, self.I_RHOGVZ, self.I_RHOGE)

        PROG, fvx, fvy, fvz, fe = hs_apply_core(
            PROG, rho, pre, tem, vx, vy, vz, lat, vmtr.VMTR_GSGAM2, dt,
            cnst, kmin, kmax, rdtype, idx, xp,
        )
        # stash for validation / history output (nicamdc history_in ml_af_fvx..)
        self.fvx, self.fvy, self.fvz, self.fe = fvx, fvy, fvz, fe

        prf.PROF_rapend('__Forcing', 1)
        return PROG

frc = Frc()