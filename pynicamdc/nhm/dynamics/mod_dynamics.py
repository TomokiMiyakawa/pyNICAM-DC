import os
import numpy as np
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_prof import prf
from pynicamdc.nhm.dynamics.kernels.diag import DiagCfg, compute_diagnostics

class Dyn:
    
    _instance = None
    
    def __init__(self, adm, cnst, rcnf, rdtype):

        ###Global###

        # Prognostic and tracer variables
        self.PROG        = np.full((adm.ADM_shape + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PROG_pl     = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PROGq       = np.full((adm.ADM_shape + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PROGq_pl    = np.full((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)

        self.PROGq.fill(rdtype(0.0))    # perhaps remove later
        self.PROGq_pl.fill(rdtype(0.0)) # perhaps remove later

        # Tendency of prognostic and tracer variables
        self.g_TEND      = np.full((adm.ADM_shape + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.g_TEND_pl   = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.g_TENDq     = np.full((adm.ADM_shape + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.g_TENDq_pl  = np.full((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)

        # Forcing tendency
        self.f_TEND      = np.full((adm.ADM_shape + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.f_TEND_pl   = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.f_TENDq     = np.full((adm.ADM_shape + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.f_TENDq_pl  = np.full((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)

        # Saved prognostic/tracer variables
        self.PROG00      = np.full((adm.ADM_shape + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PROG00_pl   = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PROGq00     = np.full((adm.ADM_shape + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PROGq00_pl  = np.full((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PROG0       = np.full((adm.ADM_shape + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PROG0_pl    = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)

        # Split prognostic variables
        self.PROG_split     = np.full((adm.ADM_shape + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PROG_split_pl  = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)

        # Mean prognostic variables
        self.PROG_mean      = np.full((adm.ADM_shape + (5,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PROG_mean_pl   = np.full((adm.ADM_shape_pl + (5,)), cnst.CONST_UNDEF, dtype=rdtype)

        # For tracer advection (large step)
        self.f_TENDrho_mean     = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self.f_TENDrho_mean_pl  = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.f_TENDq_mean       = np.full((adm.ADM_shape + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.f_TENDq_mean_pl    = np.full((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PROG_mean_mean     = np.full((adm.ADM_shape + (5,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PROG_mean_mean_pl  = np.full((adm.ADM_shape_pl + (5,)), cnst.CONST_UNDEF, dtype=rdtype)

        # Diagnostic and tracer variables
        self.DIAG     = np.full((adm.ADM_shape + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.DIAG_pl  = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.q        = np.full((adm.ADM_shape + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.q_pl     = np.full((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)

        # Density
        self.rho      = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self.rho_pl   = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        # Internal energy (physical)
        self.ein      = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self.ein_pl   = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        # Enthalpy (physical)
        self.eth      = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self.eth_pl   = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        # Potential temperature (physical)
        self.th       = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self.th_pl    = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        # Density deviation from base state
        self.rhogd    = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self.rhogd_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        # Pressure deviation from base state
        self.pregd    = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self.pregd_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)


        ### Local ###

        # Temporary variables  (use underscores _ )
        self._qd       = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self._qd_pl    = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self._cv      = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self._cv_pl    = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        # work array for the dynamics
        self._numerator_w = np.full((adm.ADM_KSshape), cnst.CONST_UNDEF, dtype=rdtype)
        self._denominator_w = np.full((adm.ADM_KSshape), cnst.CONST_UNDEF, dtype=rdtype)
        self._numerator_pl_w = np.full((adm.ADM_KSshape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self._denominator_pl_w = np.full((adm.ADM_KSshape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        # Static config for the backend-switchable diagnostic-variable kernel
        # (mod_dynamics diagnostics block). Indices/scalars are constant, so this
        # can be marked static under jax.jit. The jitted kernel is built lazily
        # on first dynamics_step call (it needs the backend object msc.bk).
        self._diag_cfg = DiagCfg(
            I_RHOG=rcnf.I_RHOG, I_RHOGVX=rcnf.I_RHOGVX, I_RHOGVY=rcnf.I_RHOGVY,
            I_RHOGVZ=rcnf.I_RHOGVZ, I_RHOGW=rcnf.I_RHOGW, I_RHOGE=rcnf.I_RHOGE,
            I_pre=rcnf.I_pre, I_tem=rcnf.I_tem, I_vx=rcnf.I_vx, I_vy=rcnf.I_vy,
            I_vz=rcnf.I_vz, I_w=rcnf.I_w,
            kmin=adm.ADM_kmin, kmax=adm.ADM_kmax,
            nmin=rcnf.NQW_STR, nmax=rcnf.NQW_END, iqv=rcnf.I_QV,
            Rdry=cnst.CONST_Rdry, Rvap=cnst.CONST_Rvap, CVdry=cnst.CONST_CVdry,
        )
        self._diag_kernel = None  # set lazily in dynamics_step

        return
    

    def dynamics_setup(self, fname_in, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, bndc, bsst, numf, vi, rdtype):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print("+++ Module[dynamics]/Category[nhm]", file=log_file)     
                print(f"+++ Time integration type: {tim.TIME_integ_type.strip()}", file=log_file)

        # Number of large steps (0–4)
        self.num_of_iteration_lstep = 0
        # Number of substeps for each large step (up to 4 stages)
        self.num_of_iteration_sstep = np.zeros(4, dtype=int)

        if tim.TIME_integ_type == 'RK2':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("+++ 2-stage Runge-Kutta", file=log_file)
            self.num_of_iteration_lstep = 2
            self.num_of_iteration_sstep[0] = tim.TIME_sstep_max / 2
            self.num_of_iteration_sstep[1] = tim.TIME_sstep_max

        elif tim.TIME_integ_type == 'RK3':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("+++ 3-stage Runge-Kutta", file=log_file)
            self.num_of_iteration_lstep = 3
            self.num_of_iteration_sstep[0] = tim.TIME_sstep_max / 3  
            self.num_of_iteration_sstep[1] = tim.TIME_sstep_max / 2  
            self.num_of_iteration_sstep[2] = tim.TIME_sstep_max      

        elif tim.TIME_integ_type == 'RK4':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("+++ 4-stage Runge-Kutta", file=log_file)
            self.num_of_iteration_lstep = 4
            self.num_of_iteration_sstep[0] = tim.TIME_sstep_max / 4
            self.num_of_iteration_sstep[1] = tim.TIME_sstep_max / 3
            self.num_of_iteration_sstep[2] = tim.TIME_sstep_max / 2
            self.num_of_iteration_sstep[3] = tim.TIME_sstep_max

        elif tim.TIME_integ_type == 'TRCADV':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("+++ Offline tracer experiment", file=log_file)
            self.num_of_iteration_lstep = 0

            if rcnf.TRC_ADV_TYPE == 'DEFAULT':
                print(f"xxx [dynamics_setup] unsupported advection scheme for TRCADV test! STOP. {rcnf.TRC_ADV_TYPE.strip()}")
                prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            print(f"xxx [dynamics_setup] unsupported integration type! STOP. {tim.TIME_integ_type.strip()}")
            prc.prc_stop(std.io_l, std.fname_log)


        self.trcadv_out_dyndiv = False

        if rcnf.TRC_ADV_LOCATION == 'OUT_DYN_DIV_LOOP':
            if rcnf.TRC_ADV_TYPE == 'MIURA2004':
                self.trcadv_out_dyndiv = True
            else:
                print(f"xxx [dynamics_setup] unsupported TRC_ADV_TYPE for OUT_DYN_DIV_LOOP. STOP. {rcnf.TRC_ADV_TYPE.strip()}")
                prc.prc_mpistop(std.io_l, std.fname_log)

        #self.rweight_dyndiv = rdtype(1.0) / rdtype(rcnf.DYN_DIV_NUM)
        self.rweight_dyndiv = 1.0 / rcnf.DYN_DIV_NUM   # Double precision

        #---< boundary condition module setup >---                                                                         
        bndc.BNDCND_setup(fname_in, rdtype)

        #---< basic state module setup >---                                                                                
        bsst.bsstate_setup(fname_in, cnst, rdtype)

        #---< numerical filter module setup >---                                                                           
        numf.numfilter_setup(fname_in, rcnf, cnst, comm, gtl, grd, gmtr, oprt, vmtr, tim, prgv, tdyn, bndc, bsst, rdtype)

        #---< vertical implicit module setup >---                                                                          
        vi.vi_setup(cnst,rdtype) #(fname_in, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, bndc, bsst, numf, rdtype)

        # skip
        #---< sub-grid scale dynamics module setup >---                                                                    
        #TENTATIVE!     call sgs_setup                                                                                          

        # skip
        #---< nudging module setup >---                                                                                    
        #call NDG_setup

        return
                          
    def dynamics_step(self, msc):
        # better to extract variables from msc and pack it in to xp before entering the function
        # seperate it based on whether it is overwritten or not in the dynamics_step

        adm  = msc.adm
        ppm = msc.ppm
        comm = msc.comm
        cnst = msc.cnst
        grd = msc.grd
        gmtr = msc.gmtr
        oprt = msc.oprt
        vmtr = msc.vmtr
        tim = msc.tim
        rcnf = msc.rcnf
        prgv = msc.prgv
        tdyn = msc.tdyn
        frc = msc.frc
        bndc = msc.bndc
        cnvv = msc.cnvv
        bsst = msc.bsst
        numf = msc.numf
        vi = msc.vi
        src = msc.src
        srctr = msc.srctr
        trcadv = msc.trcadv
        rdtype = msc.bk.ndtype

        # Make views of arrays

        # Prognostic and tracer variables
        PROG        = self.PROG
        PROG_pl     = self.PROG_pl
        PROGq       = self.PROGq
        PROGq_pl    = self.PROGq_pl

        # Tendency of prognostic and tracer variables
        g_TEND      = self.g_TEND
        g_TEND_pl   = self.g_TEND_pl
        g_TENDq     = self.g_TENDq
        g_TENDq_pl  = self.g_TENDq_pl

        # Forcing tendency
        f_TEND      = self.f_TEND
        f_TEND_pl   = self.f_TEND_pl
        f_TENDq     = self.f_TENDq
        f_TENDq_pl  = self.f_TENDq_pl

        # Saved prognostic/tracer variables
        PROG00      = self.PROG00
        PROG00_pl   = self.PROG00_pl
        PROGq00     = self.PROGq00
        PROGq00_pl  = self.PROGq00_pl
        PROG0       = self.PROG0
        PROG0_pl    = self.PROG0_pl

        # Split prognostic variables
        PROG_split     = self.PROG_split
        PROG_split_pl  = self.PROG_split_pl

        # Mean prognostic variables
        PROG_mean      = self.PROG_mean
        PROG_mean_pl   = self.PROG_mean_pl

        # For tracer advection (large step)
        f_TENDrho_mean     = self.f_TENDrho_mean
        f_TENDrho_mean_pl  = self.f_TENDrho_mean_pl
        f_TENDq_mean       = self.f_TENDq_mean
        f_TENDq_mean_pl    = self.f_TENDq_mean_pl
        PROG_mean_mean     = self.PROG_mean_mean
        PROG_mean_mean_pl  = self.PROG_mean_mean_pl

        # Diagnostic and tracer variables
        DIAG     = self.DIAG
        DIAG_pl  = self.DIAG_pl
        q        = self.q
        q_pl     = self.q_pl

        # Density
        rho      = self.rho
        rho_pl   = self.rho_pl

        # Internal energy (physical)
        ein      = self.ein
        ein_pl   = self.ein_pl

        # Enthalpy (physical)
        eth      = self.eth
        eth_pl   = self.eth_pl

        # Potential temperature (physical)
        th       = self.th
        th_pl    = self.th_pl

        # Density deviation from base state
        rhogd    = self.rhogd
        rhogd_pl = self.rhogd_pl

        # Pressure deviation from base state
        pregd    = self.pregd
        pregd_pl = self.pregd_pl

        # Temporary variables
        qd       = self._qd
        qd_pl    = self._qd_pl
        cv       = self._cv
        cv_pl    = self._cv_pl

        #---< work array for the dynamics >---   # these should not be a part of msc, make it local (todo for later)
        numerator = self._numerator_w   
        denominator = self._denominator_w
        numerator_pl = self._numerator_pl_w
        denominator_pl = self._denominator_pl_w

        prf.PROF_rapstart('__Dynamics', 1)
        prf.PROF_rapstart('___Pre_Post', 1)

        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        nmin = rcnf.NQW_STR
        nmax = rcnf.NQW_END

        I_RHOG = rcnf.I_RHOG
        I_RHOGVX = rcnf.I_RHOGVX
        I_RHOGVY = rcnf.I_RHOGVY
        I_RHOGVZ = rcnf.I_RHOGVZ
        I_RHOGW = rcnf.I_RHOGW
        I_RHOGE = rcnf.I_RHOGE

        I_pre = rcnf.I_pre
        I_tem = rcnf.I_tem
        I_vx = rcnf.I_vx
        I_vy = rcnf.I_vy
        I_vz = rcnf.I_vz
        I_w  = rcnf.I_w

        CVW = rcnf.CVW

        iqv = rcnf.I_QV
        itke = rcnf.I_TKE

        rho_bs = bsst.rho_bs
        rho_bs_pl = bsst.rho_bs_pl
        pre_bs = bsst.pre_bs
        pre_bs_pl = bsst.pre_bs_pl

        Rdry  = cnst.CONST_Rdry
        CVdry = cnst.CONST_CVdry
        Rvap  = cnst.CONST_Rvap

        dyn_step_dt = tim.TIME_dtl #DP  # not rdtype(tim.TIME_dtl)
        large_step_dt = tim.TIME_dtl * self.rweight_dyndiv  #DP not rdtype(tim.TIME_dtl) * self.rweight_dyndiv

        msc.bk.set_loop_ctx("PRE")   # in-loop audit: copy-in marshal (boundary)
        prf.PROF_rapstart('____pp_marshal',2)   # decompose Pre_Post (instrument-first)
        PROG[:, :, :, :, :]  = prgv.PRG_var[:, :, :, :, 0:6]
        PROG_pl[:, :, :, :]  = prgv.PRG_var_pl[:, :, :, 0:6]
        PROGq[:, :, :, :, :] = prgv.PRG_var[:, :, :, :, 6:]
        PROGq_pl[:, :, :, :] = prgv.PRG_var_pl[:, :, :, 6:]
        prf.PROF_rapend('____pp_marshal',2)

        prf.PROF_rapend('___Pre_Post', 1)

        # U5-D (RES-CAPSTONE-29): capture the tracer's device rhogq, do the PROGq hyper-
        # viscosity update on device, and drain it ONCE at the step-end prgv marshal --
        # removing the per-ndyn host rhogq drain (@mod_src_tracer:~1201) from the loop
        # body (the U8 lax.scan enabler). Only valid for the tested single-divide path.
        _progqout = (msc.bk.type == "jax"
                     and not self.trcadv_out_dyndiv
                     and rcnf.DYN_DIV_NUM == 1
                     and os.environ.get("PYNICAM_RESIDENT_TRACER_PROGQOUT", "0") != "0")
        _PROGq_out_d = None
        # RES-CAPSTONE-44 (Track B unit 5): pole analog of PROGQOUT -- carry the tracer's
        # device pole rhogq out, do the pole PROGq_pl hyperviscosity update on device, drain
        # once at the marshal (removes the per-nl host PROGq_pl update @1251 + the tracer's
        # pole rhogq drain). Requires _progqout (regular marshal device path) + the device
        # pole vert-adv. Gate PYNICAM_RESIDENT_TRACER_PROGQOUT_PL (default OFF).
        _progqout_pl = (_progqout and adm.ADM_have_pl
                        and os.environ.get("PYNICAM_RESIDENT_TRACER_PROGQOUT_PL", "0") != "0")
        _PROGq_pl_out_d = None
        # RES-CAPSTONE-31 (PROGOUT): the device PROG carry (_prog_carry_d, RES-CP3b-2) is
        # drained to host EVERY nl @~1303 "to keep host valid". Under the resident path the
        # next nl's diag/vi read the DEVICE carry, the tracer reads PROG00 (not PROG), and
        # itke<0 so the TKE fixer (the only other host PROG reader) is off -> host PROG is
        # read only at the step-end marshal. So skip the per-nl drain and marshal the device
        # carry ONCE (analog of PROGQOUT). Same DYN_DIV==1 guard (ndyn>0 would re-read host
        # PROG at the PROG0/PROG00 snapshot). Bit-exact iff host PROG truly unread mid-loop.
        _progout = (msc.bk.type == "jax"
                    and not self.trcadv_out_dyndiv
                    and rcnf.DYN_DIV_NUM == 1
                    and os.environ.get("PYNICAM_RESIDENT_PROGOUT", "0") != "0")

        for ndyn in range(rcnf.DYN_DIV_NUM):

            #--- save the value before tracer advection
            # U1 (RES-CAPSTONE-19): on the tested in-loop MIURA2004 path the regular
            # PROG00 host copy feeds ONLY the tracer's rhog_in (== PROG00[I_RHOG]).
            # Under RKCOPY, snapshot just that one component to device (skip the ~2GB
            # host PROG.copy()) and thread it in as rhog_in_d -> the tracer's two
            # xp.asarray(rhog_in) H2D uploads become no-ops. Pole PROG00_pl stays host
            # (tracer rhog_in_pl). Gate requires the resident tracer-v path and the
            # in-loop tracer (so the host rhog_in fallback sites never execute) and is
            # default OFF. Bit-identical: device snapshot == asarray(PROG00[I_RHOG]).
            _rkcopy = (msc.bk.type == "jax"
                       and not self.trcadv_out_dyndiv
                       and tim.TIME_integ_type != 'TRCADV'
                       and os.environ.get("PYNICAM_RESIDENT_RKCOPY", "0") != "0"
                       and os.environ.get("PYNICAM_RESIDENT_PROG", "0") != "0"
                       and os.environ.get("PYNICAM_RESIDENT_TRACER_V", "1") != "0")
            _PROG00_rhog_d = None
            if (not self.trcadv_out_dyndiv) or (ndyn == 0):

                PROG00_pl = PROG_pl.copy()
                if _rkcopy:
                    _PROG00_rhog_d = msc.bk.xp.asarray(PROG[:, :, :, :, I_RHOG])
                else:
                    PROG00 = PROG.copy()

                if rcnf.TRC_ADV_TYPE == 'DEFAULT':
                    PROGq00 = PROGq.copy()
                    PROGq00_pl = PROGq_pl.copy()
                #endif
            #endif

            #--- save the value before RK loop
            # U1 (RES-CAPSTONE-19) cont'd -- PROG0 removal: under RKCOPY the regular
            # PROG0 host copy feeds ONLY _PROG0_d = asarray(PROG0) (the device
            # PROG_split subtract @~1024; the host PROG_split @~1026 is the non-resident
            # else, and the TRCADV @~472 reader is excluded by the gate). PROG0 == PROG
            # here (pre-RK), so build the device PROG0 snapshot directly and skip the
            # 2nd ~2GB host PROG.copy(). nl-invariant -> serves the CP3a per-nl carry.
            # Pole PROG0_pl stays host (PROG_split_pl). Bit-identical: asarray(PROG)
            # here == asarray(PROG.copy()) == asarray(PROG0).
            PROG0_pl = PROG_pl.copy()
            if not _rkcopy:
                PROG0 = PROG.copy()

            # RES-CP3a: device-resident PROG0 carry. PROG0 is nl-invariant (set once
            # per ndyn, before the RK loop), but the resident PROG_split subtract at
            # nl!=0 re-uploaded it via xp.asarray(PROG0) every iteration (a 340MB H2D
            # at the measured ~11.6 GB/s). Build the device handle ONCE (lazily at the
            # first nl!=0) and reuse it for the remaining iterations. NOTE: the nl==0
            # diag-input _PROG_d is NOT a valid PROG0 substitute -- it diverges from
            # PROG0 (RHOG ~1.2e-2, measured), so we upload PROG0 once rather than carry
            # _PROG_d (relevant intel for CP3b: PROG snapshots across the diag are not
            # bit-equal). Memoized asarray(PROG0) is bit-exact vs the per-nl re-upload.
            # Under RKCOPY the snapshot is pre-built here (PROG0 host copy skipped).
            _PROG0_d = msc.bk.xp.asarray(PROG[:, :, :, :, :]) if _rkcopy else None

            # RES-CP3b-1: device DIAG carry across the RK loop. The diag kernel's DIAG
            # input is re-uploaded via asarray(DIAG) (340MB H2D) every nl, but the diag
            # kernel only reuses its w-boundary rows, and host DIAG is read-only between
            # the Pre_Post drain and the next diag. So carry the post-BNDCND device
            # _DIAG from one nl to the next instead of re-uploading. nl==0 has no carry
            # yet (uses the step-initial host DIAG). Reset per ndyn.
            _DIAG_carry = None

            # RES-CP3b-2: device PROG carry across the RK loop. vi already builds and
            # returns the device PROG (RESIDENT_PROG_DEVOUT); instead of discarding it
            # and re-uploading asarray(PROG) at the next nl's diag, keep the handle,
            # run its halo COMM on-device, and feed it to the next diag. The tracer
            # reads PROG00/PROG_mean (not PROG) and TKE is inactive, so COMM is the
            # only PROG consumer in the carry span. nl==0 has no carry yet. Per ndyn.
            _prog_carry_d = None
            _prog_pl_carry_d = None

            # RES-TP-3b-i: device PROGq carry across the RK loop. In the active
            # MIURA2004 path PROGq is nl-invariant -- it is only written at the last
            # nl (tracer + f_TENDq), after that nl's diag read -- so the diag's
            # asarray(PROGq) is identical every nl. Build it once (lazily at nl==0)
            # and reuse, instead of re-uploading the full vmax-sized array each nl
            # (the [256-512)MB per-nl copy-in for moist runs). nl==0 builds it. Per ndyn.
            _PROGq_carry_d = None


            if tim.TIME_integ_type == 'TRCADV':      # TRC-ADV Test Bifurcation    #comeback later for msc

                prf.PROF_rapstart('__Tracer_Advection', 1)

                f_TEND[:, :, :, :, :] = rdtype(0.0)
                f_TEND_pl[:, :, :, :] = rdtype(0.0)

                # region 11 (rank=2, l=1) i=16 and 17   j= 0 to 17 
                # vs
                # region 30 (rank=6, l=0) i= 0 and i=1  j= 0 to 17

                with open(std.fname_log, 'a') as log_file:
                    if prc.prc_myrank == 2:
                        print("BEFORETRACER: r11, z24  SE inner 15  :", PROGq [15,:,24,1,1],  file=log_file)
                        print("BEFORETRACER: r11, z24  SE inner 16  :", PROGq [16,:,24,1,1],  file=log_file)
                        print("BEFORETRACER: r11, z24  SE edge  17  :", PROGq [17,:,24,1,1],  file=log_file)
                    elif prc.prc_myrank == 6:
                        print("BEFORETRACER: r30, z24  NW inner  2  :", PROGq [ 2,:,24,0,1],  file=log_file)
                        print("BEFORETRACER: r30, z24  NW inner  1  :", PROGq [ 1,:,24,0,1],  file=log_file)
                        print("BEFORETRACER: r30, z24  NW edge   0  :", PROGq [ 0,:,24,0,1],  file=log_file)

                # needed for DCMIP test11
                # print("not tested yet AAA")
                srctr.src_tracer_advection(
                    rcnf.TRC_vmax,                                             # [IN]
                    PROGq      [:,:,:,:,:],        PROGq_pl  [:,:,:,:],        # [INOUT] 
                    PROG0      [:,:,:,:,I_RHOG],   PROG0_pl  [:,:,:,I_RHOG],   # [IN]  
                    PROG       [:,:,:,:,I_RHOG],   PROG_pl   [:,:,:,I_RHOG],   # [IN]  
                    PROG       [:,:,:,:,I_RHOGVX], PROG_pl   [:,:,:,I_RHOGVX], # [IN]  
                    PROG       [:,:,:,:,I_RHOGVY], PROG_pl   [:,:,:,I_RHOGVY], # [IN]  
                    PROG       [:,:,:,:,I_RHOGVZ], PROG_pl   [:,:,:,I_RHOGVZ], # [IN]  
                    PROG       [:,:,:,:,I_RHOGW],  PROG_pl   [:,:,:,I_RHOGW],  # [IN]  
                    f_TEND     [:,:,:,:,I_RHOG],   f_TEND_pl [:,:,:,I_RHOG],   # [IN]  
                    large_step_dt,                                             # [IN]                       
                    rcnf.THUBURN_LIM,                                          # [IN]             
                    None, None,      # [IN] Optional, for setting height dependent choice for vertical and horizontal Thuburn limiter
                    cnst, comm, grd, gmtr, oprt, vmtr, rdtype,
                )


                prf.PROF_rapend('__Tracer_Advection', 1)
                
                #skip for now (not needed for JW test)
                frc.forcing_update( PROG, PROG_pl,  # [INOUT]
                                    cnst, rcnf, grd, tim, trcadv, rdtype,
                                    ) 

            # endif


            #---------------------------------------------------------------------------
            #
            #> Start large time step integration
            #
            #---------------------------------------------------------------------------
            for nl in range(self.num_of_iteration_lstep):
                msc.bk.set_loop_ctx("INLOOP")   # in-loop audit: nl-loop body

                prf.PROF_rapstart('___Pre_Post',1)

        
                # print("in lstep loop, nl = ", nl, "/", self.num_of_iteration_lstep -1) 
                # print("stopping the program AaA")
                # prc.prc_mpifinish(std.io_l, std.fname_log)
                # import sys 
                # sys.exit()

                prf.PROF_rapstart('____pp_log',2)
                with open(std.fname_log, 'a') as log_file:
                    print("lstep starting, iteration number: ", nl, "/", self.num_of_iteration_lstep -1, file=log_file)
                prf.PROF_rapend('____pp_log',2)

                #---< Generate diagnostic values and set the boudary conditions
                # --- Diagnostic variables (backend-switchable pure kernel) ---
                # Computes rho, DIAG[vx,vy,vz,tem,pre,w], ein, q (and work cv, qd)
                # from PROG/PROGq. See kernels/diag.py and proto/test_diag_kernel.py.
                bk = msc.bk
                xp = bk.xp
                if self._diag_kernel is None:
                    self._diag_kernel = bk.maybe_jit(
                        compute_diagnostics, static_argnames=("cfg", "xp"))
                # Read-only metrics/coefs staged device-resident once.
                _diag_dev = bk.device_consts(self, "diag", lambda: {
                    "GSGAM2":  vmtr.VMTR_GSGAM2,
                    "C2Wfact": vmtr.VMTR_C2Wfact,
                    "CVW":     CVW,
                })
                # Pre_Post resident chain: keep rho/DIAG/ein/PROG on device across
                # diag -> BNDCND -> THRMDYN -> perturbations, draining once at the
                # end (drops the per-kernel asarray/to_numpy brackets). JAX-only,
                # gated PYNICAM_RESIDENT_PREPOST (default off). REGULAR path only;
                # the pole block (tiny) stays numpy.
                _resident_prepost = (bk.type == "jax") and os.environ.get("PYNICAM_RESIDENT_PREPOST", "1") != "0"
                # PHASE 2 (U8) -- segment fusion. FUSE_PREPOST: collapse the EAGER
                # BNDCND_all_resident (~20 .at[].set() ops + 3 sub-kernels, run eagerly)
                # into ONE jit graph. Eager device ops each materialise their python
                # scalar constants on device (the 8-byte H2D the nsys baseline found);
                # jit bakes them into the compiled graph AND collapses the per-op
                # dispatch. First call runs eager (warms the bndc device_consts/kernel
                # caches), then builds + caches the jit; bit-exact (jit == eager). Default
                # OFF. Requires RESIDENT_PREPOST (the device BNDCND path).
                _fuse_prepost = _resident_prepost and os.environ.get("PYNICAM_FUSE_PREPOST", "0") != "0"
                # RESIDENT_PROG: keep the Pre_Post device PROG/DIAG (_PROG_d/_DIAG)
                # live past the drain and thread them into downstream phases so each
                # phase slices [...,I_*] as an on-device view instead of a host
                # strided-gather. Requires RESIDENT_PREPOST (source of _PROG_d/_DIAG).
                # Default off; jax-only. Staged: advmom+hdiff first, then vi, tracer.
                _resident_prog = _resident_prepost and os.environ.get("PYNICAM_RESIDENT_PROG", "0") != "0"
                # RESIDENT_DIAG: thread the device-resident DIAG velocity views into
                # vi (removing the strided host-gather asarray(DIAG[...,I_v*]) inside
                # vi_path0). Default ON under RESIDENT_PROG; off-switch for A/B.
                _resident_diag = _resident_prog and os.environ.get("PYNICAM_RESIDENT_DIAG", "1") != "0"
                # RES-CP3a: reuse the nl-invariant device PROG0 across the RK loop
                # (skip the per-nl asarray(PROG0) 340MB re-upload). Default on under
                # RESIDENT_PROG; asarray(PROG0) fallback when off.
                _resident_prog0_carry = _resident_prog and os.environ.get("PYNICAM_RESIDENT_PROG0_CARRY", "1") != "0"
                # RES-CAPSTONE Phase A (g_TEND0): assemble the regular large-step
                # tendency g_TEND on device from the producer device handles (advmom
                # velocity tendencies + hdiff f_TEND) and feed it to vi, removing the
                # ~6.1GB asarray(g_TEND0) re-upload inside vi_path0. Requires
                # RESIDENT_PROG (the producers run their device path only then). The
                # producers stash a handle only when their resident+horizontalized
                # kernel path actually ran, so the assembly below falls back to host
                # asarray(g_TEND0) inside vi whenever either stash is absent. Default
                # on under RESIDENT_PROG; pole (_pl) stays host (tiny) in vi.
                _resident_gtend = _resident_prog and os.environ.get("PYNICAM_RESIDENT_GTEND", "1") != "0"
                # RES-CP3b-1: carry the device _DIAG across the nl boundary so the diag
                # kernel reuses it instead of re-uploading asarray(DIAG). Requires the
                # resident Pre_Post chain (source of the post-BNDCND device _DIAG);
                # asarray fallback otherwise.
                _resident_diag_carry = _resident_prepost and os.environ.get("PYNICAM_RESIDENT_DIAG_CARRY", "1") != "0"
                # RES-CP3b-2: carry the device PROG across the nl boundary (vi device-out
                # + on-device halo COMM) so the diag reuses it instead of re-uploading
                # asarray(PROG). Requires RESIDENT_PROG (vi device-out source); asarray
                # fallback otherwise.
                # TKE GUARD: the carry assumes COMM is the only PROG consumer between
                # vi-out and the next diag. When a turbulence scheme is active (itke>=0)
                # the TKE fixer modifies host PROG[I_RHOGE] after vi (do_tke_correction),
                # which the device carry would miss -> silent divergence. itke<0 (no TKE,
                # e.g. dry/non-turbulent) guarantees do_tke_correction stays False, so
                # only then is the carry safe. Falls back to host COMM + asarray re-upload.
                _resident_prog_carry = _resident_prog and (itke < 0) and \
                    os.environ.get("PYNICAM_RESIDENT_PROG_CARRY", "1") != "0"
                # RES-CAPSTONE Track B unit B (pole PROG carry): run the POLE Pre_Post
                # diag -> BNDCND on device (reshape-reuse compute_diagnostics +
                # BNDCND_all_pl_resident), carry the device pole PROG across the nl
                # boundary (vi already returns _prog_pl_carry_d), and thread the device
                # pole handles into vi so its pole seeds asarray(PROG_pl/PROG_split_pl/
                # PROG_mean_pl/vx_pl..) become device no-ops. The host pole diag block is
                # still drained to host (src_advection/numfilter/THRMDYN pole consumers
                # are un-ported -> future units), but the per-nl pole PROG drain @~1398
                # is skipped and the carry is drained once at the marshal. Requires the
                # regular carry (shares the COMM @~1393 + the itke<0 TKE guard); pole
                # arrays are tiny -> ~0 wall-clock (the U8 enabler, by design). Default
                # OFF; asarray fallback keeps it bit-exact when off. Gate
                # PYNICAM_RESIDENT_PROG_PL.
                _resident_prog_pl = _resident_prog_carry and \
                    os.environ.get("PYNICAM_RESIDENT_PROG_PL", "0") != "0"
                # RES-TP-3b-i: carry the device PROGq across the nl boundary so the diag
                # reuses it instead of re-uploading asarray(PROGq) every nl (the [256-512)
                # MB per-nl copy-in for moist runs). Valid only where PROGq is nl-invariant
                # across the RK loop: the active MIURA2004 path writes PROGq only at the
                # last nl (tracer + f_TENDq), after that nl's diag read; the dead DEFAULT
                # branch updates PROGq every nl (@1049), which a build-once carry would
                # miss. itke<0 (no turbulence; holds for moist non-turbulent runs) keeps
                # the TKE fixer from modifying PROGq mid-span. Falls back to asarray.
                _resident_progq_carry = _resident_prepost and (itke < 0) and \
                    (rcnf.TRC_ADV_TYPE == "MIURA2004") and \
                    os.environ.get("PYNICAM_RESIDENT_PROGQ_CARRY", "1") != "0"
                # U6 SINGLE-DRAIN (full-residency audit) -- two gates over the @~662
                # batch drain (11 arrays {rho,DIAG,ein,q,cv,qd,PROG,th,eth,pregd,rhogd}):
                #  * PYNICAM_DRAIN_SKIP = comma list -- the bisection INSTRUMENT (used to
                #    pin which drained arrays still have live host consumers; Phase D
                #    diverged because static analysis missed eth).
                #  * PYNICAM_SINGLE_DRAIN=1 = the U6 milestone -- skip ALL 11 (remove the
                #    whole batch drain). The regular host chain is fully device-covered:
                #    th/ein/cv/qd/q are dead (no host reader); rho/DIAG/PROG via the nl
                #    carries; eth via the RES-CAPSTONE-16 ethh port; pregd/rhogd via the
                #    resident src P_d/rhog_d. REQUIRES PYNICAM_RESIDENT_ETHH on -- else
                #    host eth still feeds the eth_h interp @mod_vi.py and skipping its
                #    drain diverges (job 2260932: skip='eth' FAIL without the port).
                # Both default OFF = full drain = BIT-EXACT. Self-protected: only honored
                # when the full resident+carry chain is active (else the drains are needed).
                _drain_skip = set()
                _ALL_DRAINS = ("rho", "DIAG", "ein", "q", "cv", "qd",
                               "PROG", "th", "eth", "pregd", "rhogd")
                if (_resident_prog_carry and _resident_diag_carry and _resident_progq_carry
                        and os.environ.get("PYNICAM_RESIDENT_ADVCONVMOM", "1") != "0"
                        and os.environ.get("PYNICAM_HDIFF_RESIDENT_FULL", "1") != "0"
                        and os.environ.get("PYNICAM_RESIDENT_SRCTERM", "1") != "0"
                        and os.environ.get("PYNICAM_RESIDENT_DIAG", "1") != "0"):
                    _drain_skip = set(s for s in
                        os.environ.get("PYNICAM_DRAIN_SKIP", "").split(",") if s)
                    if (os.environ.get("PYNICAM_SINGLE_DRAIN", "0") != "0"
                            and os.environ.get("PYNICAM_RESIDENT_ETHH", "0") != "0"):
                        _drain_skip = set(_ALL_DRAINS)

                prf.PROF_rapstart('____pp_diag',2)
                # RES-CP3b-2: reuse the carried post-COMM device PROG (from the previous
                # nl) as the diag input instead of re-uploading asarray(PROG). nl==0 has
                # no carry yet -> host upload. Bit-identical: host PROG was drained from
                # the same post-COMM device handle and is read-only until here.
                if _resident_prog_carry and _prog_carry_d is not None:
                    _PROG_d = _prog_carry_d
                else:
                    _PROG_d = xp.asarray(PROG)
                # RES-CP3b-1: reuse the carried device _DIAG (post-BNDCND, from the
                # previous nl) as the diag input instead of re-uploading asarray(DIAG).
                # nl==0 has no carry yet -> host upload. Bit-identical: host DIAG was
                # drained from _DIAG_carry and is read-only until here.
                _diag_in = _DIAG_carry if (_resident_diag_carry and _DIAG_carry is not None) else xp.asarray(DIAG)
                # RES-TP-3b-i: reuse the device PROGq carry as the diag input instead of
                # re-uploading asarray(PROGq). Built lazily here at nl==0 (identical to the
                # asarray it replaces) and reused for nl>0 (PROGq is nl-invariant in
                # MIURA2004). Bit-identical: host PROGq is read-only across the carry span.
                if _resident_progq_carry:
                    if _PROGq_carry_d is None:
                        _PROGq_carry_d = xp.asarray(PROGq)
                    _PROGq_in = _PROGq_carry_d
                else:
                    _PROGq_in = xp.asarray(PROGq)
                _rho, _DIAG, _ein, _q, _cv, _qd = self._diag_kernel(
                    _PROG_d, _PROGq_in, _diag_in,
                    _diag_dev["GSGAM2"], _diag_dev["C2Wfact"], _diag_dev["CVW"],
                    cfg=self._diag_cfg, xp=xp,
                )
                if not _resident_prepost:
                    # Write back into the persistent numpy buffers. The local aliases
                    # rho, DIAG, ein, q, cv, qd point to these same arrays, so the
                    # downstream code (BNDCND_all, THRMDYN, etc.) needs no change.
                    rho[:, :, :, :]     = bk.to_numpy(_rho)
                    DIAG[:, :, :, :, :] = bk.to_numpy(_DIAG)
                    ein[:, :, :, :]     = bk.to_numpy(_ein)
                    q[:, :, :, :, :]    = bk.to_numpy(_q)
                    cv[:, :, :, :]      = bk.to_numpy(_cv)
                    qd[:, :, :, :]      = bk.to_numpy(_qd)
                prf.PROF_rapend('____pp_diag',2)

                #DIAG underwent update (msc.dyn.DIAG)

                # Task1
                #print("Task1a done")
                #np.seterr(under='ignore')
                prf.PROF_rapstart('____pp_bndcnd',2)
                _fused_thrmdyn = False
                if _resident_prepost:
                    if _fuse_prepost and getattr(self, "_prepost_jit", None) is not None:
                        # FUSE_PREPOST: ONE jit graph for BNDCND + THRMDYN + perturbations
                        # (scalars baked, dispatch collapsed, pre_bs/rho_bs no longer
                        # re-uploaded). Produces the th/eth/pregd/rhogd device handles the
                        # eager THRMDYN block below would otherwise recompute.
                        (_DIAG, _PROG_d, _rho, _ein,
                         _th_d, _eth_d, _pregd_d, _rhogd_d) = self._prepost_jit(
                            _DIAG, _PROG_d, _rho, _ein)
                        _fused_thrmdyn = True
                    else:
                        _DIAG, _PROG_d, _rho, _ein = bndc.BNDCND_all_resident(
                            msc, _DIAG, _PROG_d, _rho, _ein)
                        if _fuse_prepost:
                            # warm-up done (bndc caches populated by the eager call above);
                            # build + cache the fused BNDCND+THRMDYN jit for subsequent nls.
                            # msc/bndc + the basis state (pre_bs/rho_bs, verified set-once),
                            # GSGAM2 geometry, and the cnst scalars are run-constant ->
                            # captured static (baking the scalars + folding the per-nl
                            # asarray(pre_bs/rho_bs) uploads into one compiled graph); the 4
                            # array args traced. Bit-exact (jit reproduces the eager seq).
                            _msc_c, _bndc_c = msc, bndc
                            _gsg_c    = _diag_dev["GSGAM2"]
                            _pre_bs_d = xp.asarray(pre_bs)
                            _rho_bs_d = xp.asarray(rho_bs)
                            _RovCP_c  = cnst.CONST_Rdry / cnst.CONST_CPdry
                            _PRE00_c  = cnst.CONST_PRE00
                            _Ipre, _Item = I_pre, I_tem
                            def _prepost_fn(_D, _P, _r, _e):
                                _D, _P, _r, _e = _bndc_c.BNDCND_all_resident(_msc_c, _D, _P, _r, _e)
                                _pre = _D[:, :, :, :, _Ipre]
                                _tem = _D[:, :, :, :, _Item]
                                _th  = _tem * (_PRE00_c / _pre) ** _RovCP_c    # THRMDYN_th
                                _eth = _e + _pre / _r                          # THRMDYN_eth
                                _pregd = (_pre - _pre_bs_d) * _gsg_c           # perturbation
                                _rhogd = (_r - _rho_bs_d) * _gsg_c             # perturbation
                                return _D, _P, _r, _e, _th, _eth, _pregd, _rhogd
                            self._prepost_jit = bk.jax.jit(_prepost_fn)
                else:
                    bndc.BNDCND_all(msc)
                prf.PROF_rapend('____pp_bndcnd',2)
                prf.PROF_rapstart('____pp_thrmdyn',2)

                    # prc.prc_mpifinish(std.io_l, std.fname_log)
                    # print("stopping the program AAAA")
                    # import sys 
                    # sys.exit()

                #call BNDCND_all

                if _resident_prepost:
                    # THRMDYN + perturbations inline on device, then ONE drain of the
                    # whole regular chain (rho/DIAG/ein/PROG/q/cv/qd/th/eth/pregd/rhogd).
                    if not _fused_thrmdyn:
                        # eager path (FUSE_PREPOST off, or nl==0 warm-up before the jit is
                        # cached). When fused, the jit above already produced these handles.
                        _pre_d = _DIAG[:, :, :, :, I_pre]
                        _tem_d = _DIAG[:, :, :, :, I_tem]
                        _RovCP = cnst.CONST_Rdry / cnst.CONST_CPdry
                        _th_d  = _tem_d * (cnst.CONST_PRE00 / _pre_d) ** _RovCP   # THRMDYN_th
                        _eth_d = _ein + _pre_d / _rho                            # THRMDYN_eth
                        _gsg_d = _diag_dev["GSGAM2"]
                        _pregd_d = (_pre_d - xp.asarray(pre_bs)) * _gsg_d
                        _rhogd_d = (_rho - xp.asarray(rho_bs)) * _gsg_d
                    if "rho"   not in _drain_skip: rho[:, :, :, :]     = bk.to_numpy(_rho)
                    if "DIAG"  not in _drain_skip: DIAG[:, :, :, :, :] = bk.to_numpy(_DIAG)
                    if "ein"   not in _drain_skip: ein[:, :, :, :]     = bk.to_numpy(_ein)
                    if "q"     not in _drain_skip: q[:, :, :, :, :]    = bk.to_numpy(_q)
                    if "cv"    not in _drain_skip: cv[:, :, :, :]      = bk.to_numpy(_cv)
                    if "qd"    not in _drain_skip: qd[:, :, :, :]      = bk.to_numpy(_qd)
                    if "PROG"  not in _drain_skip: PROG[:, :, :, :, :] = bk.to_numpy(_PROG_d)
                    if "th"    not in _drain_skip: th  = bk.to_numpy(_th_d)
                    if "eth"   not in _drain_skip: eth = bk.to_numpy(_eth_d)
                    if "pregd" not in _drain_skip: pregd[:, :, :, :] = bk.to_numpy(_pregd_d)
                    if "rhogd" not in _drain_skip: rhogd[:, :, :, :] = bk.to_numpy(_rhogd_d)
                    # RES-CP3b-1: stash the post-BNDCND device _DIAG for the next nl's
                    # diag input (skips its asarray(DIAG) re-upload). Host DIAG above is
                    # the drain of this same handle and is read-only until then.
                    if _resident_diag_carry:
                        _DIAG_carry = _DIAG
                else:
                    # Task2
                    th = tdyn.THRMDYN_th(
                            DIAG[:, :, :, :, I_tem],
                            DIAG[:, :, :, :, I_pre],
                            cnst,
                    )
                    # Task3
                    eth = tdyn.THRMDYN_eth(
                            ein,
                            DIAG[:, :, :, :, I_pre],
                            rho,
                            cnst,
                    )
                    # perturbations ( pre, rho with metrics )
                    pregd[:, :, :, :] = (DIAG[:, :, :, :, I_pre] - pre_bs) * vmtr.VMTR_GSGAM2
                    rhogd[:, :, :, :] = (rho                  - rho_bs) * vmtr.VMTR_GSGAM2


                # with open(std.fname_log, 'a') as log_file:
                #     print("vmtr.VMTR_GSGAM2_pl", vmtr.VMTR_GSGAM2_pl, file=log_file)
                #     print("PROG_pl[:, :, :, I_RHOG]", PROG_pl[:, :, :, I_RHOG], file=log_file)
                          
                # Track B unit B: device pole PROG + post-BNDCND DIAG handles for vi
                # (threaded into vi_small_step so its pole asarray seeds become no-ops).
                # None when the gate is off / no pole -> vi falls back to asarray (bit-
                # exact). Reset every nl.
                _PROG_pl_d = None
                _DIAG_pl_dev = None
                # RES-CAPSTONE-61/62: device pole THRMDYN handles (None unless the fused
                # pole jit produced them). _thrmdyn_pl_done gates the host THRMDYN skip;
                # _pregd_pl_d/_rhogd_pl_d are threaded into vi's pole src terms (RC-62) so
                # their drains die. Defined here (before have_pl) so they're in scope at
                # the vi call on no-pole ranks too.
                _thrmdyn_pl_done = False
                _eth_pl_d = _pregd_pl_d = _rhogd_pl_d = None
                # RES-CAPSTONE-62: mirror vi's RESIDENT_SRCTERM gate so the pole pregd/rhogd
                # drain-skip and the device-handle thread to vi are gated identically (no
                # half-on combo where vi reads a stale host pregd_pl/rhogd_pl).
                _resident_srcterm_pl = (os.environ.get("PYNICAM_RESIDENT_SRCTERM", "1") != "0")
                _thread_thrmdyn_pl = False   # set True when the device pole pregd/rhogd are threaded to vi
                # RES-CAPSTONE-63: eth_pl drain is removable only when ALL its consumers go
                # device -- vi eth_h_pl interp (needs RESIDENT_ETHH) + src_advection pole
                # (RESIDENT_SRCTERM) + pole matrix + _eth0_pl_d. Mirror RESIDENT_ETHH.
                _resident_ethh_pl = (os.environ.get("PYNICAM_RESIDENT_ETHH", "0") != "0")
                _thread_eth_pl = False       # set True when the device pole eth is threaded to vi
                if adm.ADM_have_pl:

                    if _resident_prog_pl:
                        # === Track B unit B: POLE Pre_Post diag + BNDCND on device ===
                        # step 1: device pole PROG carry (vi's _prog_pl_carry_d from the
                        # previous nl; nl==0 has no carry -> host upload). This is the
                        # PRE-BNDCND pole PROG; the device diag+BNDCND below mirror the
                        # host block 1:1 (a carry-only port diverges precisely because the
                        # carry is pre-BNDCND, so diag+BNDCND must run on it on-device).
                        if _prog_pl_carry_d is not None:
                            _PROG_pl_d = _prog_pl_carry_d
                        else:
                            _PROG_pl_d = xp.asarray(PROG_pl)
                        # step 3: reshape-reuse compute_diagnostics on the pole. The pole
                        # is (g,k,l); reshape -> (g,1,k,l[,c]) so k lands on axis 2 (the
                        # regular layout) and the SAME jitted kernel applies (bit-exact --
                        # the kernel is axis-positional). The only pole tweak is the
                        # GSGAM2 divisor (plmask dummy-pole guard, matches host @~845),
                        # passed as the GSGAM2 arg.
                        _dgp = bk.device_consts(self, "diag_pl", lambda: {
                            "GSGAM2":  (vmtr.VMTR_GSGAM2_pl - rdtype(ppm.plmask - 1))[:, None, :, :],
                            "C2Wfact": vmtr.VMTR_C2Wfact_pl[:, None, :, :, :],
                            "CVW":     CVW,
                        })
                        # FUSE_PREPOST (Phase 2): collapse the pole diag -> squeeze ->
                        # BNDCND -> THRMDYN+perturbations into ONE jit graph (the pole
                        # analog of the regular _prepost_jit). Bakes the pole diag/BNDCND/
                        # THRMDYN scalar constants + the GSGAM2 plmask guard and collapses
                        # the per-op dispatch. RES-CAPSTONE-61: the graph now also produces
                        # the device pole th/eth/pregd/rhogd, drained below (the host
                        # THRMDYN block is skipped) -- moves that host compute to device.
                        # The un-ported vi/src pole consumers still read the host drains;
                        # threading the device handles into vi is the follow-on unit.
                        if _fuse_prepost and getattr(self, "_prepost_pl_jit", None) is not None:
                            (_DIAG_pl, _PROG_pl_d, _rho_pl, _ein_pl,
                             _q_pl, _cv_pl, _qd_pl,
                             _th_pl_d, _eth_pl_d, _pregd_pl_d, _rhogd_pl_d) = self._prepost_pl_jit(
                                _PROG_pl_d, xp.asarray(PROGq_pl), xp.asarray(DIAG_pl))
                            _thrmdyn_pl_done = True
                        else:
                            _rho_pl, _DIAG_pl, _ein_pl, _q_pl, _cv_pl, _qd_pl = self._diag_kernel(
                                _PROG_pl_d[:, None, :, :, :], xp.asarray(PROGq_pl)[:, None, :, :, :],
                                xp.asarray(DIAG_pl)[:, None, :, :, :],
                                _dgp["GSGAM2"], _dgp["C2Wfact"], _dgp["CVW"],
                                cfg=self._diag_cfg, xp=xp,
                            )
                            # squeeze the dummy j-axis (axis 1) back out
                            _rho_pl  = _rho_pl[:, 0, :, :]
                            _DIAG_pl = _DIAG_pl[:, 0, :, :, :]
                            _ein_pl  = _ein_pl[:, 0, :, :]
                            _q_pl    = _q_pl[:, 0, :, :, :]
                            _cv_pl   = _cv_pl[:, 0, :, :]
                            _qd_pl   = _qd_pl[:, 0, :, :]
                            # step 2: device pole BNDCND (boundary ghost rows); identical
                            # return tuple to BNDCND_all_resident, pole shapes.
                            _DIAG_pl, _PROG_pl_d, _rho_pl, _ein_pl = bndc.BNDCND_all_pl_resident(
                                msc, _DIAG_pl, _PROG_pl_d, _rho_pl, _ein_pl)
                            if _fuse_prepost:
                                # warm-up done (kernel/bndc caches warm); build + cache the
                                # fused pole jit. msc/bndc + the pole device consts (_dgp,
                                # run-constant geometry) + cfg captured static; the 3 pole
                                # arrays traced. Bit-exact (jit reproduces the eager seq).
                                _msc_c, _bndc_c = msc, bndc
                                _dgp_c = _dgp
                                _cfg_c = self._diag_cfg
                                _dk_c  = self._diag_kernel
                                # RES-CAPSTONE-61: pole THRMDYN + perturbations baked into
                                # the same graph. Pole basis state (pre_bs_pl/rho_bs_pl,
                                # verified set-once in mod_bsstate) + VMTR_GSGAM2_pl + the
                                # cnst scalars are run-constant -> captured static.
                                _Ipre, _Item = I_pre, I_tem
                                _PRE00_c  = cnst.CONST_PRE00
                                _RovCP_c  = cnst.CONST_Rdry / cnst.CONST_CPdry
                                _pre_bs_pl_c = xp.asarray(pre_bs_pl)
                                _rho_bs_pl_c = xp.asarray(rho_bs_pl)
                                _gsg_pl_c    = xp.asarray(vmtr.VMTR_GSGAM2_pl)
                                def _prepost_pl_fn(_P, _Pq, _D):
                                    _r, _DI, _e, _qq, _cvv, _qdd = _dk_c(
                                        _P[:, None, :, :, :], _Pq[:, None, :, :, :],
                                        _D[:, None, :, :, :],
                                        _dgp_c["GSGAM2"], _dgp_c["C2Wfact"], _dgp_c["CVW"],
                                        cfg=_cfg_c, xp=xp,
                                    )
                                    _r = _r[:, 0, :, :]; _DI = _DI[:, 0, :, :, :]; _e = _e[:, 0, :, :]
                                    _qq = _qq[:, 0, :, :, :]; _cvv = _cvv[:, 0, :, :]; _qdd = _qdd[:, 0, :, :]
                                    _DI, _P, _r, _e = _bndc_c.BNDCND_all_pl_resident(_msc_c, _DI, _P, _r, _e)
                                    # THRMDYN th/eth + perturbations (pole), mirror of the
                                    # host tdyn block + regular _prepost_jit.
                                    _pre = _DI[:, :, :, _Ipre]
                                    _tem = _DI[:, :, :, _Item]
                                    _th    = _tem * (_PRE00_c / _pre) ** _RovCP_c   # THRMDYN_th
                                    _ethpl = _e + _pre / _r                         # THRMDYN_eth
                                    _pregd = (_pre - _pre_bs_pl_c) * _gsg_pl_c       # perturbation
                                    _rhogd = (_r - _rho_bs_pl_c) * _gsg_pl_c         # perturbation
                                    return (_DI, _P, _r, _e, _qq, _cvv, _qdd,
                                            _th, _ethpl, _pregd, _rhogd)
                                self._prepost_pl_jit = bk.jax.jit(_prepost_pl_fn)
                        _DIAG_pl_dev = _DIAG_pl   # post-BNDCND device velocity views for vi
                        # drain to host for the un-ported pole consumers (THRMDYN pole,
                        # pregd/rhogd pole, src_advection, numfilter). Tiny pole arrays;
                        # bit-exact (the device path mirrors the host block exactly).
                        # rho_pl is rebound (matches the host @~845 fresh-array semantics).
                        rho_pl = bk.to_numpy(_rho_pl)
                        DIAG_pl[:, :, :, :] = bk.to_numpy(_DIAG_pl)
                        ein_pl[:, :, :]     = bk.to_numpy(_ein_pl)
                        q_pl[:, :, :, :]    = bk.to_numpy(_q_pl)
                        cv_pl[:, :, :]      = bk.to_numpy(_cv_pl)
                        qd_pl[:, :, :]      = bk.to_numpy(_qd_pl)
                        PROG_pl[:, :, :, :] = bk.to_numpy(_PROG_pl_d)
                        # RES-CAPSTONE-71 (audit campaign): poison-classify the 7 per-nl pole
                        # diag drains @958-964 (D2H, the largest remaining cluster). NaN each
                        # host pole array AFTER the drain; PASS vs gold => no downstream pole
                        # consumer reads it (device handle already threaded by RC-38..65) =>
                        # drain removable; FAIL => live, needs a consumer device-thread.
                        # Diagnostic scaffolding, default-empty = bit-exact. Tags:
                        # rhopl/diagpl/einpl/qpl/cvpl/qdpl/progpl (comma-list).
                        _pdp = os.environ.get("PYNICAM_PL_DIAG_POISON", "")
                        if _pdp:
                            # rho_pl is rebound to a read-only to_numpy view -> rebind to a
                            # fresh nan array (can't [:]=). The other 6 are pre-allocated
                            # writable host arrays -> in-place nan is fine.
                            if "rhopl"  in _pdp: rho_pl = np.full(rho_pl.shape, np.nan, dtype=rho_pl.dtype)
                            if "diagpl" in _pdp: DIAG_pl[:] = np.nan
                            if "einpl"  in _pdp: ein_pl[:]  = np.nan
                            if "qpl"    in _pdp: q_pl[:]    = np.nan
                            if "cvpl"   in _pdp: cv_pl[:]   = np.nan
                            if "qdpl"   in _pdp: qd_pl[:]   = np.nan
                            if "progpl" in _pdp: PROG_pl[:] = np.nan
                        # RES-CAPSTONE-61: drain the fused device pole THRMDYN outputs to
                        # host (the host THRMDYN block below is then skipped). The un-ported
                        # vi/src pole consumers still read these host arrays; threading the
                        # device handles into vi is the follow-on. th_pl/eth_pl rebound to
                        # match the host fresh-array semantics; pregd_pl/rhogd_pl in place.
                        if _thrmdyn_pl_done:
                            _thread_thrmdyn_pl = _resident_srcterm_pl
                            _thread_eth_pl = _resident_srcterm_pl and _resident_ethh_pl
                            # RES-CAPSTONE-62/63: th_pl is dead (no host reader on the tested
                            # path, like regular th under SINGLE_DRAIN) -> never drained.
                            # eth_pl/pregd_pl/rhogd_pl: drain ONLY when their consumers won't
                            # use the device handle (gate off) -> host stays valid for the
                            # asarray fallback. When threaded, the drains die.
                            if not _thread_eth_pl:
                                eth_pl = bk.to_numpy(_eth_pl_d)
                            if not _thread_thrmdyn_pl:
                                pregd_pl[:, :, :] = bk.to_numpy(_pregd_pl_d)
                                rhogd_pl[:, :, :] = bk.to_numpy(_rhogd_pl_d)
                    else:
                        #rho_pl = PROG_pl[:, :, :, I_RHOG]   / vmtr.VMTR_GSGAM2_pl
                        rho_pl = PROG_pl[:, :, :, I_RHOG]   / (vmtr.VMTR_GSGAM2_pl - rdtype(ppm.plmask - 1))  #Divide by value if plmask is 1, divide by value + 1 if plmask is 0 (value allowed to be 0 for dummy poles)
                        DIAG_pl[:, :, :, I_vx] = PROG_pl[:, :, :, I_RHOGVX] / PROG_pl[:, :, :, I_RHOG]
                        DIAG_pl[:, :, :, I_vy] = PROG_pl[:, :, :, I_RHOGVY] / PROG_pl[:, :, :, I_RHOG]
                        DIAG_pl[:, :, :, I_vz] = PROG_pl[:, :, :, I_RHOGVZ] / PROG_pl[:, :, :, I_RHOG]
                        ein_pl[:, :, :] = PROG_pl[:, :, :, I_RHOGE]  / PROG_pl[:, :, :, I_RHOG]

                        # Tracer mass mixing ratios
                        q_pl[:, :, :, :] = PROGq_pl / PROG_pl[:, :, :, np.newaxis, I_RHOG]

                        # Specific heat capacity and dry air fraction
                        cv_pl.fill(rdtype(0.0))
                        qd_pl.fill(rdtype(1.0))

                        q_slice_pl = q_pl[:, :, :, nmin:nmax+1]
                        CVW_slice = CVW[nmin:nmax+1]

                        cv_pl += np.sum(q_slice_pl * CVW_slice[np.newaxis, np.newaxis, np.newaxis, :], axis=3)
                        qd_pl -= np.sum(q_slice_pl, axis=3)
                        cv_pl += qd_pl * CVdry

                        # Temperature and pressure
                        DIAG_pl[:, :, :, I_tem] = ein_pl / cv_pl
                        DIAG_pl[:, :, :, I_pre] = rho_pl * DIAG_pl[:, :, :, I_tem] * (
                            qd_pl * Rdry + q_pl[:, :, :, iqv] * Rvap
                        )

                        numerator_pl   = PROG_pl[:, kmin+1:kmax+1, :, I_RHOGW]
                        rhog_k_pl      = PROG_pl[:, kmin+1:kmax+1, :, I_RHOG]
                        rhog_km1_pl    = PROG_pl[:, kmin:kmax,     :, I_RHOG]
                        fact1_pl       = vmtr.VMTR_C2Wfact_pl[:, kmin+1:kmax+1, :, 0]
                        fact2_pl       = vmtr.VMTR_C2Wfact_pl[:, kmin+1:kmax+1, :, 1]
                        denominator_pl = fact1_pl * rhog_k_pl + fact2_pl * rhog_km1_pl

                        DIAG_pl[:, kmin+1:kmax+1, :, I_w] = numerator_pl / denominator_pl

                        # Task1b
                        #print("Task1b done")
                        #np.seterr(under='ignore')
                        bndc.BNDCND_all_pl(
                            adm.ADM_kmin,
                            adm.ADM_kmax,
                            adm.ADM_gall_pl,
                            adm.ADM_kall,
                            adm.ADM_lall_pl,
                            rho_pl [:, :, :],                # [INOUT] view with additional dimension may stay after the BNDCND_all call. Squeeze it back later explicitly.
                            DIAG_pl[:, :, :, I_vx],          # [INOUT]
                            DIAG_pl[:, :, :, I_vy],          # [INOUT]
                            DIAG_pl[:, :, :, I_vz],          # [INOUT]
                            DIAG_pl[:, :, :, I_w],           # [INOUT]
                            ein_pl [:, :, :],                # [INOUT]
                            DIAG_pl[:, :, :, I_tem],         # [INOUT]%
                            DIAG_pl[:, :, :, I_pre],         # [INOUT]
                            PROG_pl[:, :, :, I_RHOG],        # [INOUT]
                            PROG_pl[:, :, :, I_RHOGVX],      # [INOUT]
                            PROG_pl[:, :, :, I_RHOGVY],      # [INOUT]
                            PROG_pl[:, :, :, I_RHOGVZ],      # [INOUT]
                            PROG_pl[:, :, :, I_RHOGW],       # [INOUT]
                            PROG_pl[:, :, :, I_RHOGE],       # [INOUT]
                            vmtr.VMTR_GSGAM2_pl,    # [IN]
                            vmtr.VMTR_PHI_pl,    # [IN]
                            vmtr.VMTR_C2Wfact_pl, # [IN]
                            vmtr.VMTR_C2WfactGz_pl, # [IN]
                            cnst,
                            rdtype,
                        )
                        #np.seterr(under='raise')

                    # Task2 -- THRMDYN th/eth + perturbations (host; runs under both the
                    # device and host pole diag paths, reading the host DIAG_pl/rho_pl/
                    # ein_pl that each path leaves valid). RES-CAPSTONE-61: skipped when the
                    # fused pole jit already produced + drained the device th/eth/pregd/rhogd.
                    if not _thrmdyn_pl_done:
                        th_pl = tdyn.THRMDYN_th(
                            DIAG_pl[:, :, :, I_tem],
                            DIAG_pl[:, :, :, I_pre],
                            cnst,
                        )
                        # Task3
                        eth_pl = tdyn.THRMDYN_eth(
                            ein_pl [:, :, :],
                            DIAG_pl[:, :, :, I_pre],
                            rho_pl [:, :, :],
                            cnst,
                        )

                        # perturbations ( pre, rho with metrics )
                        pregd_pl[:, :, :] = (DIAG_pl[:, :, :, I_pre] - pre_bs_pl) * vmtr.VMTR_GSGAM2_pl
                        rhogd_pl[:, :, :] = (rho_pl - rho_bs_pl) * vmtr.VMTR_GSGAM2_pl

                else:

                    PROG_pl [:, :, :, :] = rdtype(0.0)
                    DIAG_pl [:, :, :, :] = rdtype(0.0)
                    rho_pl  [:, :, :]    = rdtype(0.0)
                    q_pl    [:, :, :, :] = rdtype(0.0)
                    th_pl   [:, :, :]    = rdtype(0.0)
                    eth_pl  [:, :, :]    = rdtype(0.0)
                    pregd_pl[:, :, :]    = rdtype(0.0)
                    rhogd_pl[:, :, :]    = rdtype(0.0)

                prf.PROF_rapend('____pp_thrmdyn',2)
                prf.PROF_rapend('___Pre_Post',1)
                #------------------------------------------------------------------------
                #> LARGE step
                #------------------------------------------------------------------------
                prf.PROF_rapstart('___Large_step', 1)

                #--- calculation of advection tendency including Coriolis force
                # AUDIT seg-bisect: restore host PROG from the device carry before advmom
                # (vprg_nl0 NaN'd it). PASS => the live host-PROG reader is at/after advmom.
                if nl > 0 and _prog_carry_d is not None and "rstr_advmom" in os.environ.get("PYNICAM_REG_POISON", ""):
                    PROG[:, :, :, :, :] = msc.bk.to_numpy(_prog_carry_d)
                # Task 4
                #print("Task4 done but not tested yet")
                #np.seterr(under='ignore')
                src.src_advection_convergence_momentum(
                        DIAG  [:,:,:,:,I_vx],     DIAG_pl  [:,:,:,I_vx],     # [IN]
                        DIAG  [:,:,:,:,I_vy],     DIAG_pl  [:,:,:,I_vy],     # [IN]
                        DIAG  [:,:,:,:,I_vz],     DIAG_pl  [:,:,:,I_vz],     # [IN]
                        DIAG  [:,:,:,:,I_w],      DIAG_pl  [:,:,:,I_w],      # [IN]
                        PROG  [:,:,:,:,I_RHOG],   PROG_pl  [:,:,:,I_RHOG],   # [IN]
                        PROG  [:,:,:,:,I_RHOGVX], PROG_pl  [:,:,:,I_RHOGVX], # [IN]
                        PROG  [:,:,:,:,I_RHOGVY], PROG_pl  [:,:,:,I_RHOGVY], # [IN]
                        PROG  [:,:,:,:,I_RHOGVZ], PROG_pl  [:,:,:,I_RHOGVZ], # [IN]
                        PROG  [:,:,:,:,I_RHOGW],  PROG_pl  [:,:,:,I_RHOGW],  # [IN]
                        g_TEND[:,:,:,:,I_RHOGVX], g_TEND_pl[:,:,:,I_RHOGVX], # [OUT]   # pl 2,0  sign reversed
                        g_TEND[:,:,:,:,I_RHOGVY], g_TEND_pl[:,:,:,I_RHOGVY], # [OUT]   # pl 2,0  sign of #5 reversed
                        g_TEND[:,:,:,:,I_RHOGVZ], g_TEND_pl[:,:,:,I_RHOGVZ], # [OUT]   # pl 2,0  sign of #5 reversed, others off
                        g_TEND[:,:,:,:,I_RHOGW],  g_TEND_pl[:,:,:,I_RHOGW],  # [OUT]   # pl 2,0  sign of #5 reversed, others off
                        rcnf, cnst, grd, oprt, vmtr, rdtype,
                        prog_d=(_PROG_d if _resident_prog else None),
                        diag_d=(_DIAG   if _resident_prog else None),
                        stash_device=_resident_gtend,
                )
                #np.seterr(under='raise')


                g_TEND[:, :, :, :, I_RHOG]  = rdtype(0.0)
                g_TEND[:, :, :, :, I_RHOGE] = rdtype(0.0)

                # Zero out specific components of g_TEND_pl
                g_TEND_pl[:, :, :, I_RHOG]  = rdtype(0.0)
                g_TEND_pl[:, :, :, I_RHOGE] = rdtype(0.0)


                #---< numerical diffusion term
                if rcnf.NDIFF_LOCATION == 'IN_LARGE_STEP':

                    print("xxx [dynamics_step] NDIFF_LOCATION = IN_LARGE_STEP is not implemented! STOP.")
                    prc.prc_mpistop(std.io_l, std.fname_log)

                    if nl == 0: # only first step
                        #------ numerical diffusion

                        # Task skip
                        #call numfilter_hdiffusion

                        if numf.NUMFILTER_DOverticaldiff : # numerical diffusion (vertical)
                            # Task skip
                            #    call numfilter_vdiffusion
                            pass

                        if numf.NUMFILTER_DOrayleigh :  # rayleigh damping
                            # Task skip
                            #    call numfilter_vdiffusion
                            pass

                elif rcnf.NDIFF_LOCATION == 'IN_LARGE_STEP2':        

                    #------ numerical diffusion

                    # Task 5
#                    print("Task5")
                    #"Task5 done but not tested yet"
                    # with open(std.fname_log, 'a') as log_file:  
                    #     print("g_TEND check (6,5,2,0,:)", g_TEND[6, 5, 2, 0, :], file=log_file) 
                    #     print("going into numfilter_hdiffusion IN_LARGE_STEP2", file=log_file)
                    #np.seterr(under='ignore')
                    # AUDIT seg-bisect: restore host PROG from the carry before hdiff.
                    # PASS => reader is at/after hdiff (advmom did NOT read host PROG).
                    if nl > 0 and _prog_carry_d is not None and "rstr_hdiff" in os.environ.get("PYNICAM_REG_POISON", ""):
                        PROG[:, :, :, :, :] = msc.bk.to_numpy(_prog_carry_d)
                    numf.numfilter_hdiffusion(
                        PROG   [:,:,:,:,I_RHOG], PROG_pl   [:,:,:,I_RHOG], # [IN]
                        rho,                     rho_pl,                   # [IN]
                        DIAG   [:,:,:,:,I_vx],   DIAG_pl   [:,:,:,I_vx],   # [IN]
                        DIAG   [:,:,:,:,I_vy],   DIAG_pl   [:,:,:,I_vy],   # [IN]
                        DIAG   [:,:,:,:,I_vz],   DIAG_pl   [:,:,:,I_vz],   # [IN]
                        DIAG   [:,:,:,:,I_w],    DIAG_pl   [:,:,:,I_w],    # [IN]
                        DIAG   [:,:,:,:,I_tem],  DIAG_pl   [:,:,:,I_tem],  # [IN]
                        q,                       q_pl,                     # [IN]
                        f_TEND [:,:,:,:,:],      f_TEND_pl [:,:,:,:],      # [OUT]     #you
                        f_TENDq[:,:,:,:,:],      f_TENDq_pl[:,:,:,:],      # [OUT]
                        cnst, comm, grd, oprt, vmtr, tim, rcnf, bsst, rdtype,
                        prog_d=(_PROG_d if _resident_prog else None),
                        diag_d=(_DIAG   if _resident_prog else None),
                        rho_d =(_rho    if _resident_prog else None),
                        stash_device=_resident_gtend,
                    )
                    #np.seterr(under='raise')

                    if numf.NUMFILTER_DOverticaldiff : # numerical diffusion (vertical)
                        print("xxx [dynamics_step] NUMFILTER_DOverticaldiff is not implemented! STOP.")
                        prc.prc_mpistop(std.io_l, std.fname_log)
                        # Task skip
                        #    call numfilter_vdiffusion
                        pass

                    if numf.NUMFILTER_DOrayleigh :  # rayleigh damping
                        print("xxx [dynamics_step] NUMFILTER_DOrayleigh is not implemented! STOP.")
                        prc.prc_mpistop(std.io_l, std.fname_log)
                        # Task skip
                        #    call numfilter_vdiffusion
                        pass

                #endif

                # Skip NUDGING for now
                #
                # if ndg.FLAG_NUDGING:
                #   if ( nl == 1 ) then
                #      call NDG_update_reference( TIME_CTIME )
                #   endif
                #   if ( nl == num_of_iteration_lstep ) then
                #      ndg_TEND_out = .true.
                #   else
                #      ndg_TEND_out = .false.
                #   endif
                #   call NDG_apply_uvtp
                #   endif
                    

                g_TEND[:, :, :, :, 0:6] += f_TEND[:, :, :, :, 0:6]

                # with open(std.fname_log, 'a') as log_file:
                #     print("g_TEND afteradded (6,5,37,0,:)", g_TEND[6, 5, 37, 0, :], file=log_file)

                g_TEND_pl += f_TEND_pl

                # RES-CAPSTONE Phase A (g_TEND0): assemble the regular g_TEND on device
                # from the producer device handles so vi reuses it instead of
                # re-uploading asarray(g_TEND0) (~6.1GB). Component order = I_RHOG..
                # I_RHOGE (0..5), matching vi's _g0[...,I_*] indexing. Bit-exact: the
                # device handles are the exact sources of the host grhog*/f_TEND
                # drained above, device f64 add == host f64 add, and the host
                # g_TEND[RHOG/RHOGE]=0 then += f_TEND reduces to just the f_TEND
                # component (x + 0.0 == x). Falls back (g_tend_d=None -> vi asarray)
                # whenever a producer did not take its device+horizontalized path.
                _g_TEND_d = None
                if _resident_gtend:
                    _adv = getattr(src,  "_gtend_adv_d", None)
                    _ft  = getattr(numf, "_ftend_d",     None)
                    if _adv is not None and _ft is not None:
                        _avx, _avy, _avz, _aw = _adv
                        _ftvx, _ftvy, _ftvz, _ftw, _fte, _ftrho = _ft
                        _g_TEND_d = xp.stack([
                            _ftrho,        # I_RHOG   = 0  (host: 0 += f_TEND[RHOG])
                            _avx + _ftvx,  # I_RHOGVX = 1
                            _avy + _ftvy,  # I_RHOGVY = 2
                            _avz + _ftvz,  # I_RHOGVZ = 3
                            _aw  + _ftw,   # I_RHOGW  = 4
                            _fte,          # I_RHOGE  = 5  (host: 0 += f_TEND[RHOGE])
                        ], axis=-1)

                # RES-CAPSTONE-38 (Track B): assemble the POLE g_TEND on device from the
                # device pole producer stashes (advmom _gtend_adv_pl_d + hdiff _ftend_pl_d),
                # exact pole analog of _g_TEND_d above -> vi reuses it instead of
                # asarray(g_TEND0_pl) @mod_vi:752. Bit-exact (device f64 add == host:
                # advmom writes g_TEND_pl[VX..W], zero RHOG/RHOGE, += f_TEND_pl). Gate
                # PYNICAM_RESIDENT_GTEND_PL (default OFF); falls back to host asarray.
                _g_TEND_pl_d = None
                if (_resident_gtend and adm.ADM_have_pl
                        and os.environ.get("PYNICAM_RESIDENT_GTEND_PL", "0") != "0"):
                    _adv_pl = getattr(src,  "_gtend_adv_pl_d", None)
                    _ft_pl  = getattr(numf, "_ftend_pl_d",     None)
                    if _adv_pl is not None and _ft_pl is not None:
                        _avxp, _avyp, _avzp, _awp = _adv_pl
                        _ftvxp, _ftvyp, _ftvzp, _ftwp, _ftep, _ftrhop = _ft_pl
                        _g_TEND_pl_d = xp.stack([
                            _ftrhop,         # I_RHOG   = 0
                            _avxp + _ftvxp,  # I_RHOGVX = 1
                            _avyp + _ftvyp,  # I_RHOGVY = 2
                            _avzp + _ftvzp,  # I_RHOGVZ = 3
                            _awp  + _ftwp,   # I_RHOGW  = 4
                            _ftep,           # I_RHOGE  = 5
                        ], axis=-1)


                prf.PROF_rapend('___Large_step',1)
                #------------------------------------------------------------------------
                #> SMALL step
                #------------------------------------------------------------------------
                prf.PROF_rapstart('___Small_step',1)

                # RESIDENT_PROG Stage 2b 2.2: build the regular PROG_split on device
                # (PROG0 device view minus the resident device PROG _PROG_d) and feed
                # it to vi as prog_split_d, eliminating the 340MB host subtract + the
                # asarray re-upload inside vi. Bit-exact (device f64 sub == host f64
                # sub; _PROG_d == asarray(PROG) here). Pole (_pl) stays host (tiny).
                # host PROG_split is not read after vi, so it is left untouched under
                # resident. _PROG_split_d only referenced when _resident_prog.
                _PROG_split_d = None
                # Track B unit B: device pole PROG_split (PROG0_pl - post-BNDCND device
                # pole PROG); threaded into vi so its asarray(PROG_split_pl) seed becomes
                # a no-op. None when the gate is off / no pole -> vi asarray fallback.
                _PROG_split_pl_d = None
                if nl != 0:
                    # Update split values
                    if _resident_prog:
                        # RES-CP3a: build the device PROG0 once (first nl!=0) and reuse
                        # it for the rest, instead of a fresh 340MB asarray(PROG0)
                        # re-upload each nl. Bit-identical: PROG0 is nl-invariant, so
                        # the memoized handle == a fresh asarray(PROG0).
                        # U1: under RKCOPY, _PROG0_d is pre-built from the device
                        # snapshot at top-of-ndyn (host PROG0 copy skipped). Else build
                        # it lazily from host PROG0 (CP3a carry-memoized).
                        if not _rkcopy and not (_resident_prog0_carry and _PROG0_d is not None):
                            _PROG0_d = xp.asarray(PROG0)
                        _PROG_split_d = _PROG0_d[:, :, :, :, 0:6] - _PROG_d[:, :, :, :, 0:6]
                    else:
                        PROG_split[:, :, :, :, 0:6] = PROG0[:, :, :, :, 0:6] - PROG[:, :, :, :, 0:6]
                    if _resident_prog_pl and _PROG_pl_d is not None:
                        # device pole split from the post-BNDCND device pole PROG (pole
                        # PROG0_pl stays host). Bit-identical to the host subtract below.
                        _PROG_split_pl_d = xp.asarray(PROG0_pl) - _PROG_pl_d
                    PROG_split_pl[:, :, :, :] = PROG0_pl[:, :, :, :] - PROG_pl[:, :, :, :]
                else:
                    # Zero out split values
                    if _resident_prog:
                        _PROG_split_d = xp.zeros_like(_PROG_d)
                    else:
                        PROG_split[:, :, :, :, 0:6] = rdtype(0.0)
                    if _resident_prog_pl and _PROG_pl_d is not None:
                        _PROG_split_pl_d = xp.zeros_like(_PROG_pl_d)
                    PROG_split_pl[:, :, :, :] = rdtype(0.0)
                #endif
            
                #------ Core routine for small step
                #------    1. By this subroutine, prognostic variables ( rho,.., rhoge ) are calculated through
                #------    2. grho, grhogvx, ..., and  grhoge has the large step
                #------       tendencies initially, however, they are re-used in this subroutine.
                #------

                if tim.TIME_split:   # check closely !!!
                    small_step_ite = self.num_of_iteration_sstep[nl]
                    small_step_dt = tim.TIME_dts * self.rweight_dyndiv   #DP
                else:
                    small_step_ite = 1
                    small_step_dt = large_step_dt / (self.num_of_iteration_lstep - nl)
                #endif

                # AUDIT seg-bisect: restore host PROG from the carry before vi. PASS =>
                # reader is vi (advmom+hdiff did NOT read host PROG).
                if nl > 0 and _prog_carry_d is not None and "rstr_vi" in os.environ.get("PYNICAM_REG_POISON", ""):
                    PROG[:, :, :, :, :] = msc.bk.to_numpy(_prog_carry_d)
                # Task 6
#               print("Task6")
                #np.seterr(under='ignore')
                _vi_ret = vi.vi_small_step(
                           PROG      [:,:,:,:,:],    PROG_pl      [:,:,:,:],    #   [INOUT] prognostic variables      #
                           DIAG      [:,:,:,:,I_vx], DIAG_pl      [:,:,:,I_vx], #   [IN] diagnostic value
                           DIAG      [:,:,:,:,I_vy], DIAG_pl      [:,:,:,I_vy], #   [IN]
                           DIAG      [:,:,:,:,I_vz], DIAG_pl      [:,:,:,I_vz], #   [IN]
                           eth,                      eth_pl,                    #   [IN]
                           rhogd,                    rhogd_pl,                  #   [IN]
                           pregd,                    pregd_pl,                  #   [IN]
                           g_TEND,                   g_TEND_pl,                 #   [IN] large step TEND
                           PROG_split[:,:,:,:,:],    PROG_split_pl[:,:,:,:],    #   [INOUT] split value               #
                           PROG_mean [:,:,:,:,:],    PROG_mean_pl[:,:,:,:],     #   [OUT] mean value                  #
                           small_step_ite,                                      #   [IN]
                           small_step_dt,                                       #   [IN]
                           cnst, comm, grd, oprt, vmtr, tim, rcnf, bndc, cnvv, numf, src, rdtype,
                           prog_d=(_PROG_d if _resident_prog else None),
                           prog_split_d=(_PROG_split_d if _resident_prog else None),
                           vx_d=(_DIAG[:,:,:,:,I_vx] if _resident_diag else None),
                           vy_d=(_DIAG[:,:,:,:,I_vy] if _resident_diag else None),
                           vz_d=(_DIAG[:,:,:,:,I_vz] if _resident_diag else None),
                           # RES-CAPSTONE Phase A: reuse the Pre_Post device enthalpy
                           # (_eth_d @~633, drained to host eth @~645, read-only until
                           # here) instead of vi re-uploading asarray(eth). Bit-identical.
                           eth_d=(_eth_d if _resident_prepost else None),
                           # RES-CAPSTONE Phase A: device-assembled regular g_TEND
                           # (None when the producers fell back to host) -> vi skips
                           # the ~6.1GB asarray(g_TEND0) re-upload. Pole stays host.
                           g_tend_d=_g_TEND_d,
                           g_tend_pl_d=_g_TEND_pl_d,   # RES-CAPSTONE-38 (Track B): device pole g_TEND
                           # RES-CAPSTONE Phase B: Pre_Post device pregd/rhogd
                           # (_pregd_d/_rhogd_d @~645-646, drained to host @~656-657,
                           # read-only until here) -> vi's vp0 src_pres_gradient /
                           # src_buoyancy skip asarray(pregd)/asarray(rhogd). Bit-exact.
                           preg_d=(_pregd_d if _resident_prepost else None),
                           rhog_d=(_rhogd_d if _resident_prepost else None),
                           # RES-CAPSTONE-62: device POLE pregd/rhogd from the fused pole
                           # THRMDYN (_pregd_pl_d/_rhogd_pl_d) -> vi's pole src_pres_gradient
                           # /src_buoyancy skip asarray(pregd_pl)/asarray(rhogd_pl) AND the
                           # mod_dynamics drains die. None (no fused pole) -> asarray fallback.
                           preg_pl_d=(_pregd_pl_d if _thread_thrmdyn_pl else None),
                           rhog_pl_d=(_rhogd_pl_d if _thread_thrmdyn_pl else None),
                           eth_pl_d=(_eth_pl_d if _thread_eth_pl else None),  # RES-CAPSTONE-63 (pole eth)
                           # RES-CAPSTONE Track B unit B: device POLE PROG (post-BNDCND)
                           # + PROG_split + velocity views from the device pole diag
                           # block, so vi's pole asarray(PROG_pl/PROG_split_pl/PROG_mean_pl
                           # /vx_pl..) seeds become device no-ops. None -> asarray fallback.
                           prog_pl_d=(_PROG_pl_d if _resident_prog_pl else None),
                           prog_split_pl_d=(_PROG_split_pl_d if _resident_prog_pl else None),
                           vx_pl_d=(_DIAG_pl_dev[:,:,:,I_vx] if (_resident_prog_pl and _DIAG_pl_dev is not None) else None),
                           vy_pl_d=(_DIAG_pl_dev[:,:,:,I_vy] if (_resident_prog_pl and _DIAG_pl_dev is not None) else None),
                           vz_pl_d=(_DIAG_pl_dev[:,:,:,I_vz] if (_resident_prog_pl and _DIAG_pl_dev is not None) else None),
                )
                # RES-CP3b-2: capture vi's returned device PROG (regular + pole) for the
                # cross-nl carry. vi returns the tuple only on its device-out path
                # (RESIDENT_PROG_DEVOUT); None otherwise -> carry stays disabled.
                # RES-CAPSTONE-35: under PYNICAM_RESIDENT_PROGMEAN_OUT vi returns a
                # 4-tuple (adds the device PROG_mean regular+pole, already on-device
                # COMM'd) so the tracer reads its mean mass flux from device handles.
                _pm_carry_d = _pm_pl_carry_d = None
                if _vi_ret is None:
                    _prog_carry_d = _prog_pl_carry_d = None
                elif len(_vi_ret) == 4:
                    _prog_carry_d, _prog_pl_carry_d, _pm_carry_d, _pm_pl_carry_d = _vi_ret
                else:
                    _prog_carry_d, _prog_pl_carry_d = _vi_ret
                if not _resident_prog_carry:
                    _prog_carry_d = _prog_pl_carry_d = None
                # RESIDENCY-AUDIT nl-bisect for the LIVE regular host-PROG reader (vprgreg
                # FAIL). NaN host PROG after vi at a chosen nl; whichever tag FAILs localizes
                # the reader's timing (nl0 -> read at nl>0 despite the device carry; last ->
                # read by marshal/cross-step). Default empty = bit-exact.
                _rpnl = os.environ.get("PYNICAM_REG_POISON", "")
                if ("vprg_nl0" in _rpnl and nl == 0) or \
                   ("vprg_last" in _rpnl and nl == self.num_of_iteration_lstep - 1) or \
                   ("vprg_mid" in _rpnl and 0 < nl < self.num_of_iteration_lstep - 1):
                    PROG[:] = np.nan
                #np.seterr(under='raise')
                #print("out of vi_small_step")
                #prc.prc_mpistop(std.io_l, std.fname_log)


                prf.PROF_rapend('___Small_step',1)
                #------------------------------------------------------------------------
                #>  Tracer advection (in the large step)
                #------------------------------------------------------------------------
                prf.PROF_rapstart('___Tracer_Advection',1)

                do_tke_correction = False

                if not self.trcadv_out_dyndiv:  # calc here or not

                    with open(std.fname_log, 'a') as log_file:     
                        print("WOW1", file=log_file)   # came here

                    if rcnf.TRC_ADV_TYPE == "MIURA2004":

                        with open(std.fname_log, 'a') as log_file:     
                            print("WOW2", file=log_file)    # came here

                        if nl == self.num_of_iteration_lstep-1:  # 

                            with open(std.fname_log, 'a') as log_file:     
                                print("WOW3", file=log_file)   # should come here at last iteration step ()


                            # with open (std.fname_log, 'a') as log_file:
                            #     print("partially tested, do not trust the tracer scheme just yet", file=log_file)                            
                            # RES-CAPSTONE-36: thread the device f_TEND[I_RHOG] (= the hdiff
                            # stash _ftrho = numf._ftend_d[5]) into the tracer as frhog_d, so
                            # its 4 asarray(frhog) H2D uploads no-op. Bit-exact: host frhog ==
                            # to_numpy(_ftrho). Gate PYNICAM_RESIDENT_TRACER_FRHOG (default OFF).
                            _frhog_dev = None
                            if (_resident_gtend
                                    and os.environ.get("PYNICAM_RESIDENT_TRACER_FRHOG", "0") != "0"):
                                _fts = getattr(numf, "_ftend_d", None)
                                if _fts is not None:
                                    _frhog_dev = _fts[5]   # _ftrho = device f_TEND[I_RHOG]
                            # RES-CAPSTONE-39 (Track B unit 2): device POLE frhog (= the hdiff
                            # pole stash _ftend_pl_d[5] == f_TEND_pl[I_RHOG]) -> the tracer's
                            # pole TVF asarray(frhog_pl) @241 no-ops. Pole analog of RC-36.
                            _frhog_pl_dev = None
                            if (_resident_gtend
                                    and os.environ.get("PYNICAM_RESIDENT_TRACER_FRHOG_PL", "0") != "0"):
                                _ftspl = getattr(numf, "_ftend_pl_d", None)
                                if _ftspl is not None:
                                    _frhog_pl_dev = _ftspl[5]   # device f_TEND_pl[I_RHOG]
                            _trc_ret = srctr.src_tracer_advection(
                                rcnf.TRC_vmax,                                                  # [IN]
                                PROGq       [:,:,:,:,:],        PROGq_pl      [:,:,:,:],        # [INOUT]    brakes at 0 0 6 1 et al. @rank0 in SP at step 14
                                (None if _rkcopy else PROG00[:,:,:,:,I_RHOG]),   PROG00_pl     [:,:,:,I_RHOG],   # [IN]  (U1: rhog_in via rhog_in_d under RKCOPY)
                                PROG_mean   [:,:,:,:,I_RHOG],   PROG_mean_pl  [:,:,:,I_RHOG],   # [IN]
                                PROG_mean   [:,:,:,:,I_RHOGVX], PROG_mean_pl  [:,:,:,I_RHOGVX], # [IN]  
                                PROG_mean   [:,:,:,:,I_RHOGVY], PROG_mean_pl  [:,:,:,I_RHOGVY], # [IN]  
                                PROG_mean   [:,:,:,:,I_RHOGVZ], PROG_mean_pl  [:,:,:,I_RHOGVZ], # [IN]  
                                PROG_mean   [:,:,:,:,I_RHOGW],  PROG_mean_pl  [:,:,:,I_RHOGW],  # [IN]  
                                f_TEND      [:,:,:,:,I_RHOG],   f_TEND_pl     [:,:,:,I_RHOG],   # [IN]  
                                large_step_dt,                                                  # [IN]                       
                                rcnf.THUBURN_LIM,                                               # [IN]             
                                None, None,              # [IN] Optional, for setting height dependent choice for vertical and horizontal Thuburn limiter
                                cnst, comm, grd, gmtr, oprt, vmtr, rdtype,
                                rhog_in_d=_PROG00_rhog_d,   # U1 (RES-CAPSTONE-19): device PROG00[I_RHOG] snapshot
                                skip_drain=_progqout,       # U5-D.2: drain _rhogq_d at the marshal instead
                                skip_drain_pl=_progqout_pl, # RES-CAPSTONE-44: device pole PROGq marshal
                                frhog_d=_frhog_dev,         # RES-CAPSTONE-36: device f_TEND[I_RHOG]
                                frhog_pl_d=_frhog_pl_dev,   # RES-CAPSTONE-39: device f_TEND_pl[I_RHOG]
                                # RES-CAPSTONE-35: device PROG_mean slices (regular only;
                                # pole stays host). None unless vi returned them.
                                rhog_mean_d=(_pm_carry_d[:,:,:,:,I_RHOG]   if _pm_carry_d is not None else None),
                                rhogvx_mean_d=(_pm_carry_d[:,:,:,:,I_RHOGVX] if _pm_carry_d is not None else None),
                                rhogvy_mean_d=(_pm_carry_d[:,:,:,:,I_RHOGVY] if _pm_carry_d is not None else None),
                                rhogvz_mean_d=(_pm_carry_d[:,:,:,:,I_RHOGVZ] if _pm_carry_d is not None else None),
                                rhogw_mean_d=(_pm_carry_d[:,:,:,:,I_RHOGW]  if _pm_carry_d is not None else None),
                            )
                            # RES-CAPSTONE-44: tracer returns (rhogq_d, rhogq_pl_d) under
                            # skip_drain_pl, else just rhogq_d.
                            if isinstance(_trc_ret, tuple):
                                _trc_rhogq_d, _trc_rhogq_pl_d = _trc_ret
                            else:
                                _trc_rhogq_d, _trc_rhogq_pl_d = _trc_ret, None


                            # RC-72: on MIURA2004 (our tracer scheme) numfilter sets
                            # tendency_q / f_TENDq[_pl] identically 0 (the q-hyperdiff block
                            # is skipped -> mod_numfilter:1470). So
                            #   _trc_rhogq_d + dt * asarray(f_TENDq) == _trc_rhogq_d  (exact)
                            # -> skip the asarray(f_TENDq[_pl]) H2D entirely (regular + pole).
                            # Gate PYNICAM_RESIDENT_FTENDQ (default OFF); asarray fallback keeps
                            # the non-MIURA (nonzero f_TENDq) path bit-exact.
                            _ftendq_zero = (bk.type == "jax"
                                            and os.environ.get("PYNICAM_RESIDENT_FTENDQ", "0") != "0"
                                            and rcnf.TRC_ADV_TYPE == "MIURA2004")
                            if _progqout and _trc_rhogq_d is not None:
                                # U5-D: device PROGq = device advected rhogq + dt*f_TENDq
                                # (== the host update; drained once at the marshal). U5-D.2:
                                # the host PROGq update below is skipped (host rhogq is stale
                                # -- the tracer drain was skip_drain'd -- and only the marshal
                                # reads regular PROGq under _progqout). Pole PROGq_pl stays host.
                                if _ftendq_zero:
                                    _PROGq_out_d = _trc_rhogq_d
                                else:
                                    _PROGq_out_d = _trc_rhogq_d + large_step_dt * msc.bk.xp.asarray(f_TENDq)
                            else:
                                PROGq[:, :, :, :, :] += large_step_dt * f_TENDq

                            if adm.ADM_have_pl:
                                if _progqout_pl and _trc_rhogq_pl_d is not None:
                                    # RES-CAPSTONE-44: device pole PROGq = device advected
                                    # pole rhogq + dt*f_TENDq_pl (== the host update; drained
                                    # once at the marshal). Host PROGq_pl update skipped.
                                    if _ftendq_zero:
                                        _PROGq_pl_out_d = _trc_rhogq_pl_d
                                    else:
                                        _PROGq_pl_out_d = _trc_rhogq_pl_d + large_step_dt * msc.bk.xp.asarray(f_TENDq_pl)
                                else:
                                    PROGq_pl[:, :, :, :] += large_step_dt * f_TENDq_pl

                            # [comment] H.Tomita: I don't recommend adding the hyperviscosity term because of numerical instability in this case.
                            if itke >= 0:
                                do_tke_correction = True

                        #endif

                    elif rcnf.TRC_ADV_TYPE == 'DEFAULT':

                        with open(std.fname_log, 'a') as log_file:     
                            print("WOW4, not tested", file=log_file)

                        for nq in range(rcnf.TRC_vmax):

                            with open(std.fname_log, 'a') as log_file:     
                                print("WOW5, not tested", file=log_file)

                            # Task skip for now, not used for ICOMEX_JW
                            #call src_advection_convergence
                            pass

                        #end tracer LOOP

                        step_coeff = self.num_of_iteration_sstep[nl] * small_step_dt

                        # Update PROGq for all interior points
                        PROGq += step_coeff * (g_TENDq + f_TENDq)

                        PROGq[:, :, kmin-1, :, :] = rdtype(0.0)
                        PROGq[:, :, kmax+1, :, :] = rdtype(0.0)

                        if adm.ADM_have_pl:
                            PROGq_pl[:, :, :, :] = PROGq00_pl + step_coeff * (g_TENDq_pl + f_TENDq_pl)
                            PROGq_pl[:, kmin-1, :, :] = rdtype(0.0)
                            PROGq_pl[:, kmax+1, :, :] = rdtype(0.0)

                        # Set TKE correction flag if needed
                        if itke >= 0:
                            do_tke_correction = True

                    #endif

                    # TKE fixer
                    if do_tke_correction:


                        with open(std.fname_log, 'a') as log_file:     
                            print("WOW6, not tested", file=log_file)

                        # Compute correction term (clip negative TKE values to zero)
                        TKEG_corr = np.maximum(-PROGq[:, :, :, :, itke], rdtype(0.0))

                        # Apply correction to RHOGE and TKE
                        PROG[:, :, :, :, I_RHOGE] -= TKEG_corr
                        PROGq[:, :, :, :, itke]   += TKEG_corr

                        # Polar region
                        if adm.ADM_have_pl:
                            TKEG_corr_pl = np.maximum(-PROGq_pl[:, :, :, itke], rdtype(0.0))

                            PROG_pl[:, :, :, I_RHOGE] -= TKEG_corr_pl
                            PROGq_pl[:, :, :, itke]  += TKEG_corr_pl
                        #endif
                    #endif

                else:

                    with open(std.fname_log, 'a') as log_file:     
                        print("WOW7, not tested", file=log_file)

                    #--- calculation of mean ( mean mass flux and tendency )
                    if nl == self.num_of_iteration_lstep-1:

                        with open(std.fname_log, 'a') as log_file:     
                                print("WOW8, not tested", file=log_file)

                        if ndyn == 1:

                            with open(std.fname_log, 'a') as log_file:     
                                print("WOW9, not tested", file=log_file)

                            PROG_mean_mean[:, :, :, :, 0:5] = self.rweight_dyndiv * PROG_mean[:, :, :, :, 0:5]
                            f_TENDrho_mean[:, :, :, :] = self.rweight_dyndiv * f_TEND[:, :, :, :, I_RHOG]
                            f_TENDq_mean[:, :, :, :, :] = self.rweight_dyndiv * f_TENDq


                            PROG_mean_mean_pl[:, :, :, :] = self.rweight_dyndiv * PROG_mean_pl
                            f_TENDrho_mean_pl[:, :, :]    = self.rweight_dyndiv * f_TEND_pl[:, :, :, I_RHOG]
                            f_TENDq_mean_pl[:, :, :, :]   = self.rweight_dyndiv * f_TENDq_pl

                        else:

                            with open(std.fname_log, 'a') as log_file:     
                                print("WOW10, not tested", file=log_file)

                            PROG_mean_mean[:, :, :, :, 0:5] += self.rweight_dyndiv * PROG_mean[:, :, :, :, 0:5]
                            f_TENDrho_mean[:, :, :, :] += self.rweight_dyndiv * f_TEND[:, :, :, :, I_RHOG]
                            f_TENDq_mean[:, :, :, :, :] += self.rweight_dyndiv * f_TENDq

                            PROG_mean_mean_pl[:, :, :, :] += self.rweight_dyndiv * PROG_mean_pl
                            f_TENDrho_mean_pl[:, :, :]    += self.rweight_dyndiv * f_TEND_pl[:, :, :, I_RHOG]
                            f_TENDq_mean_pl[:, :, :, :]   += self.rweight_dyndiv * f_TENDq_pl

                        #endif     
                    #endif
                #endif

                prf.PROF_rapend('___Tracer_Advection',1)

                prf.PROF_rapstart('___Pre_Post',1)

                #------ Update
                if nl != self.num_of_iteration_lstep-1:   # ayashii
                    prf.PROF_rapstart('____pp_comm',2)
                    if _resident_prog_carry and _prog_carry_d is not None:
                        # RES-CP3b-2: run PROG's halo COMM on-device (auto-routed for
                        # jax arrays, returns the updated handle) so the next diag reuses
                        # it instead of re-uploading asarray(PROG). On-device COMM is
                        # bit-exact vs host COMM (pure data movement). Also drain to host
                        # (cheap pinned D2H) so host PROG stays valid/consistent.
                        _prog_carry_d, _prog_pl_carry_d = comm.COMM_data_transfer(
                            _prog_carry_d, _prog_pl_carry_d)
                        if not _progout:   # RES-CAPSTONE-31: drained once at the marshal instead
                            PROG[:, :, :, :, :] = bk.to_numpy(_prog_carry_d)
                        if adm.ADM_have_pl and not _resident_prog_pl:   # Track B unit B: pole drained once at the marshal instead
                            PROG_pl[:, :, :, :] = bk.to_numpy(_prog_pl_carry_d)
                    else:
                        comm.COMM_data_transfer( PROG, PROG_pl )
                    prf.PROF_rapend('____pp_comm',2)
                    prf.PROF_rapstart('____pp_log',2)
                    with open(std.fname_log, 'a') as log_file:
                        print("WOW11", file=log_file)      #came here
                    prf.PROF_rapend('____pp_log',2)
                #endif

                prf.PROF_rapend  ('___Pre_Post',1)

            #end nl loop --- large step    <for nl in range(self.num_of_iteration_lstep):>
            msc.bk.set_loop_ctx("POST")   # in-loop audit: per-ndyn copy-out marshal (boundary)



            # prc.prc_mpifinish(std.io_l, std.fname_log)
            # print("stopping the program AAA")
            # import sys 
            # sys.exit()

            #---------------------------------------------------------------------------
            #>  Tracer advection (out of the large step)
            #---------------------------------------------------------------------------

            # Run on the FINAL dyn-divide iteration. The guard was previously
            # `ndyn == rcnf.DYN_DIV_NUM`, which can never hold inside
            # range(DYN_DIV_NUM) -> the block was dead, so under
            # TRC_ADV_LOCATION='OUT_DYN_DIV_LOOP' tracers were NEVER advected
            # (the in-loop path #2 above is skipped when trcadv_out_dyndiv).
            # Corrected to DYN_DIV_NUM-1 so it actually fires on the last divide.
            if self.trcadv_out_dyndiv and ndyn == rcnf.DYN_DIV_NUM - 1:

                # Guard: this out-of-dyn-div-loop tracer path is opt-in
                # (TRC_ADV_LOCATION='OUT_DYN_DIV_LOOP') and NOT yet validated.
                # Warn once per run so results from it are not silently trusted.
                # The default in-loop path (trcadv_out_dyndiv=False) is unaffected
                # by this fix, so all current/tested configs stay bit-exact.
                if not getattr(self, "_trcadv_outloop_warned", False):
                    self._trcadv_outloop_warned = True
                    with open(std.fname_log, 'a') as log_file:
                        print("*** [dynamics_step] WARNING: out-of-dyn-div-loop tracer "
                              "advection (TRC_ADV_LOCATION=OUT_DYN_DIV_LOOP) is ENABLED "
                              "but NOT YET VALIDATED -- tracer results are unverified.",
                              file=log_file)

                prf.PROF_rapstart('___Tracer_Advection',1)
                srctr.src_tracer_advection(
                    rcnf.TRC_vmax,                                                       # [IN]
                    PROGq         [:,:,:,:,:],        PROGq_pl         [:,:,:,:],        # [INOUT] 
                    PROG00        [:,:,:,:,I_RHOG],   PROG00_pl        [:,:,:,I_RHOG],   # [IN]  
                    PROG_mean_mean[:,:,:,:,I_RHOG],   PROG_mean_mean_pl[:,:,:,I_RHOG],   # [IN]  
                    PROG_mean_mean[:,:,:,:,I_RHOGVX], PROG_mean_mean_pl[:,:,:,I_RHOGVX], # [IN]  
                    PROG_mean_mean[:,:,:,:,I_RHOGVY], PROG_mean_mean_pl[:,:,:,I_RHOGVY], # [IN]  
                    PROG_mean_mean[:,:,:,:,I_RHOGVZ], PROG_mean_mean_pl[:,:,:,I_RHOGVZ], # [IN]  
                    PROG_mean_mean[:,:,:,:,I_RHOGW],  PROG_mean_mean_pl[:,:,:,I_RHOGW],  # [IN]  
                    f_TENDrho_mean[:,:,:,:],          f_TENDrho_mean_pl[:,:,:],          # [IN]  
                    large_step_dt,                                                       # [IN]                       
                    rcnf.THUBURN_LIM,                                                    # [IN]             
                    None, None,                                                          # [IN] Optional, for setting height dependent choice for vertical and horizontal Thuburn limiter
                    cnst, comm, grd, gmtr, oprt, vmtr, rdtype,
                )




                PROGq[:, :, :, :, :] += dyn_step_dt * f_TENDq_mean  # update rhogq by viscosity

                if adm.ADM_have_pl:
                    PROGq_pl[:, :, :, :] += dyn_step_dt * f_TENDq_mean_pl
                #endif

                TKEG_corr = np.maximum(-PROGq[:, :, :, :, itke], rdtype(0.0))
                PROG[:, :, :, :, I_RHOGE] -= TKEG_corr
                PROGq[:, :, :, :, itke]   += TKEG_corr

                if adm.ADM_have_pl:
                    TKEG_corr_pl = np.maximum(-PROGq_pl[:, :, :, itke], rdtype(0.0))
                    PROG_pl[:, :, :, I_RHOGE] -= TKEG_corr_pl
                    PROGq_pl[:, :, :, itke]  += TKEG_corr_pl
                #endif

                prf.PROF_rapend('___Tracer_Advection',1)

            #endif

        #enddo --- divided step for dynamics

        prf.PROF_rapstart('___Pre_Post',1)

        prf.PROF_rapstart('____pp_marshal',2)
        # RES-CAPSTONE-31 (PROGOUT): drain the device PROG carry once here (the per-nl
        # @~1303 host drain was skipped). Pole PROG_pl stays host.
        if _progout and _prog_carry_d is not None:
            prgv.PRG_var[:, :, :, :, 0:6] = msc.bk.to_numpy(_prog_carry_d)
        else:
            prgv.PRG_var[:, :, :, :, 0:6] = PROG[:, :, :, :, :]
        # Track B unit B: drain the device pole PROG carry once here (the per-nl pole
        # drain @~1477 was skipped under the gate). Else host PROG_pl.
        if _resident_prog_pl and _prog_pl_carry_d is not None:
            prgv.PRG_var_pl[:, :, :, 0:6] = msc.bk.to_numpy(_prog_pl_carry_d)
        else:
            prgv.PRG_var_pl[:, :, :, 0:6] = PROG_pl[:, :, :, :]
        # U5-D: drain the device PROGq (advected + dt*f_TENDq) once here, instead of the
        # tracer's per-ndyn host rhogq drain + the host @~1158 update. Pole stays host.
        if _progqout and _PROGq_out_d is not None:
            prgv.PRG_var[:, :, :, :, 6:]  = msc.bk.to_numpy(_PROGq_out_d)
        else:
            prgv.PRG_var[:, :, :, :, 6:]  = PROGq[:, :, :, :, :]
        # RES-CAPSTONE-44: drain the device pole PROGq once here (the per-nl host PROGq_pl
        # update @~1251 was skipped). Else host.
        if _progqout_pl and _PROGq_pl_out_d is not None:
            prgv.PRG_var_pl[:, :, :, 6:]  = msc.bk.to_numpy(_PROGq_pl_out_d)
        else:
            prgv.PRG_var_pl[:, :, :, 6:]  = PROGq_pl[:, :, :, :]
        prf.PROF_rapend('____pp_marshal',2)

        prf.PROF_rapstart('____pp_log',2)
        with open(std.fname_log, 'a') as log_file:
            kc= 5
            lc= 0
            print(f"pre_comm: prgv.PRG_var_pl [1, {kc}, {lc}, :]", prgv.PRG_var_pl [1, kc, lc, :], file=log_file)
            print(f"pre_comm: prgv.PRG_var_pl [2, {kc}, {lc}, :]", prgv.PRG_var_pl [2, kc, lc, :], file=log_file)
        prf.PROF_rapend('____pp_log',2)

        prf.PROF_rapstart('____pp_comm',2)
        comm.COMM_data_transfer(prgv.PRG_var, prgv.PRG_var_pl)
        #This comm is done in prgvar_set in the original code. Is it really necessary? # results change very slightly.
        prf.PROF_rapend('____pp_comm',2)


        prf.PROF_rapstart('____pp_log',2)
        with open(std.fname_log, 'a') as log_file:
            #ic = 6
            #jc = 5

            kc= 5
            lc= 0
            print(" ",file=log_file)
            print("ENDOF_largestep",file=log_file)
            print(f"prgv.PRG_var[:,  2, {kc}, {lc}, 5]", file=log_file)   
            print(prgv.PRG_var[:,  2, kc, lc, 5], file=log_file)   # RHOGE  rank 2 has region 10 (l=0)
            print(f"prgv.PRG_var[:, 16, {kc}, {lc}, 5]", file=log_file)
            print(prgv.PRG_var[:, 16, kc, lc, 5], file=log_file)   # RHOGE  rank 2 has region 10 (l=0)  i=0 is close to pole

            # pentagon check
            # print(prgv.PRG_var[0, 0, kc, :, 5], file=log_file) 

            # pole check   
            # #if adm.ADM_have_pl:
            print(f"prgv.PRG_var_pl [0, {kc}, {lc}, :]", prgv.PRG_var_pl [0, kc, lc, :], file=log_file)   
            print(f"prgv.PRG_var_pl [1, {kc}, {lc}, :]", prgv.PRG_var_pl [1, kc, lc, :], file=log_file)
            print(f"prgv.PRG_var_pl [2, {kc}, {lc}, :]", prgv.PRG_var_pl [2, kc, lc, :], file=log_file)

            print(" ",file=log_file)
        prf.PROF_rapend('____pp_log',2)


        prf.PROF_rapend  ('___Pre_Post',1)

        #
        #  Niwa [TM]
        #


        prf.PROF_rapend('__Dynamics', 1)

        return
        #print("dynamics_step")
        #return

