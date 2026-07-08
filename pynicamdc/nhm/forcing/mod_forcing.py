import toml
import numpy as np
#from mpi4py import MPI
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_prof import prf
from pynicamdc.nhm.forcing.mod_af_dcmip import afdcmip

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
            #self.GRD_grid_type = cnfs['GRD_grid_type']

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
            print("sorry, HELD-SUARZ is not implemented yet.")
            #self.AF_heldsuarez_init(moist_case=False)
            prc.prc_mpistop(std.io_l, std.fname_log)

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
            print("this test case is not implemented yet.")
            prc.prc_mpistop(std.io_l, std.fname_log)

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
                     vmtr, gmtr, grd, cnst, rcnf, dt, rdtype):
        """DCMIP forcing: compute+apply (nicamdc mod_forcing_driver.f90
        forcing_step, L253-390). PART A -- pure given inputs.

        Given the prognostic PROG (rhog,rhogvx,rhogvy,rhogvz,rhogw,rhoge) +
        density-weighted tracers PROGq, and the diagnostic (rho,pre,tem,vx,vy,vz,
        q) already produced by the dynamics Pre_Post, this:
          1. ein = rhoge/rhog,
          2. surface pressure (hydrostatic, single-layer -- matches nicamdc
             forcing_step, NOT the 3-level BNDCND_pre_sfc),
          3. per region -> AF_dcmip -> fvx,fvy,fvz,fe,fq,precip,
          4. applies the tendencies to PROG/PROGq
             (rhogvx += dt*fvx*rho*GSGAM2 ...; UPDATE_TOT_DENS on NQW tracers).

        Halo BNDCND_thermo / vh-halo extrapolation are omitted here: they touch
        only the k halo, which AF_dcmip never reads and which receives zero
        tendency, so the interior result is unaffected. In-place on PROG/PROGq.
        Returns precip (i,j,l).

        NOTE (part B, driver hookup): the caller must supply the diag arrays
        (from the dynamics diag machinery), carry an active qv tracer in PROGq,
        and invoke this after dynamics_step. Not exercised by a standalone run.
        """
        prf.PROF_rapstart('__Forcing', 1)

        i0, j0 = PROG.shape[0], PROG.shape[1]   # horizontal dims (layout-agnostic)
        gall = i0 * j0
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        kmin, kmax = adm.ADM_kmin, adm.ADM_kmax
        K0 = adm.ADM_K0
        GRAV = cnst.CONST_GRAV
        ntrc = PROGq.shape[-1]

        I_RHOG, I_RHOGVX = self.I_RHOG, self.I_RHOGVX
        I_RHOGVY, I_RHOGVZ = self.I_RHOGVY, self.I_RHOGVZ
        I_RHOGE = self.I_RHOGE

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

        cfg = dict(kmin=kmin, kmax=kmax, vlayer=adm.ADM_vlayer,
                   I_QV=rcnf.I_QV, I_QC=getattr(rcnf, 'I_QC', -1),
                   I_QR=getattr(rcnf, 'I_QR', -1), CVW=rcnf.CVW,
                   CVdry=cnst.CONST_CVdry, RAIN_TYPE=rcnf.RAIN_TYPE)

        fvx = np.zeros((i0, j0, kall, lall), dtype=rdtype)
        fvy = np.zeros_like(fvx); fvz = np.zeros_like(fvx); fe = np.zeros_like(fvx)
        fq = np.zeros((i0, j0, kall, lall, ntrc), dtype=rdtype)
        precip = np.zeros((i0, j0, lall), dtype=rdtype)

        # --- per region: marshal (i,j)->gall, call AF_dcmip, scatter back ---
        def flat_h(a2d):     # (i,j,l) region l -> (gall,)
            return a2d.reshape(gall)

        for l in range(lall):
            def col(a):      # (i,j,k,l) region l -> (gall, kall)
                return a[:, :, :, l].reshape(gall, kall)

            def col_t(a):    # (i,j,k,l,ntrc) region l -> (gall, kall, ntrc)
                return a[:, :, :, l].reshape(gall, kall, ntrc)

            fx, fy, fz, fee, fqq, prc_l = afdcmip.AF_dcmip(
                lat[:, :, l].reshape(gall), lon[:, :, l].reshape(gall),
                col(z), col(zh), col(rho), col(pre), col(tem),
                col(vx), col(vy), col(vz), col_t(q), col(ein),
                pre_sfc[:, :, l].reshape(gall),
                ix[:, :, l].reshape(gall), iy[:, :, l].reshape(gall),
                iz[:, :, l].reshape(gall), jx[:, :, l].reshape(gall),
                jy[:, :, l].reshape(gall), jz[:, :, l].reshape(gall), dt, cfg)

            fvx[:, :, :, l] = fx.reshape(i0, j0, kall)
            fvy[:, :, :, l] = fy.reshape(i0, j0, kall)
            fvz[:, :, :, l] = fz.reshape(i0, j0, kall)
            fe[:, :, :, l] = fee.reshape(i0, j0, kall)
            fq[:, :, :, l, :] = fqq.reshape(i0, j0, kall, ntrc)
            precip[:, :, l] = prc_l.reshape(i0, j0)

        # --- apply tendencies (nicamdc forcing_step L367-390); fw=0 for DCMIP ---
        GSGAM2 = vmtr.VMTR_GSGAM2
        PROG[:, :, :, :, I_RHOGVX] += dt * fvx * rho * GSGAM2
        PROG[:, :, :, :, I_RHOGVY] += dt * fvy * rho * GSGAM2
        PROG[:, :, :, :, I_RHOGVZ] += dt * fvz * rho * GSGAM2
        PROG[:, :, :, :, I_RHOGE] += dt * fe * rho * GSGAM2

        for nq in range(ntrc):
            frhogq = fq[:, :, :, :, nq] * rho * GSGAM2
            PROGq[:, :, :, :, nq] += dt * frhogq
            if self.UPDATE_TOT_DENS and rcnf.NQW_STR <= nq <= rcnf.NQW_END:
                PROG[:, :, :, :, I_RHOG] += dt * frhogq

        prf.PROF_rapend('__Forcing', 1)
        return precip

frc = Frc()