import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf
from mod_forcing import frc

class Dyn:
    
    _instance = None
    
    def __init__(self, cnst, rcnf, rdtype):

        # work array for the dynamics
        self._numerator_w = np.full((adm.ADM_KSshape), cnst.CONST_UNDEF, dtype=rdtype)
        self._denominator_w = np.full((adm.ADM_KSshape), cnst.CONST_UNDEF, dtype=rdtype)
        self._numerator_pl_w = np.full((adm.ADM_KSshape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self._denominator_pl_w = np.full((adm.ADM_KSshape_pl), cnst.CONST_UNDEF, dtype=rdtype)

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

        # Temporary variables
        self.qd       = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self.qd_pl    = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.cv       = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self.cv_pl    = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        # # work array for the dynamics
        # self._numerator_w = np.empty((adm.ADM_KSshape), cnst.CONST_UNDEF, dtype=rdtype)
        # self._denominator_w = np.empty((adm.ADM_KSshape), dtype=rdtype)
        # self._numerator_pl_w = np.empty((adm.ADM_KSshape_pl), dtype=rdtype)
        # self._denominator_pl_w = np.empty((adm.ADM_KSshape_pl), dtype=rdtype)

        # # Prognostic and tracer variables
        # self.PROG        = np.empty((adm.ADM_shape + (6,)), dtype=rdtype)
        # self.PROG_pl     = np.empty((adm.ADM_shape_pl + (6,)), dtype=rdtype)
        # self.PROGq       = np.empty((adm.ADM_shape + (rcnf.TRC_vmax,)), dtype=rdtype)
        # self.PROGq_pl    = np.empty((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), dtype=rdtype)

        # # Tendency of prognostic and tracer variables
        # self.g_TEND      = np.empty((adm.ADM_shape + (6,)), dtype=rdtype)
        # self.g_TEND_pl   = np.empty((adm.ADM_shape_pl + (6,)), dtype=rdtype)
        # self.g_TENDq     = np.empty((adm.ADM_shape + (rcnf.TRC_vmax,)), dtype=rdtype)
        # self.g_TENDq_pl  = np.empty((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), dtype=rdtype)

        # # Forcing tendency
        # self.f_TEND      = np.empty((adm.ADM_shape + (6,)), dtype=rdtype)
        # self.f_TEND_pl   = np.empty((adm.ADM_shape_pl + (6,)), dtype=rdtype)
        # self.f_TENDq     = np.empty((adm.ADM_shape + (rcnf.TRC_vmax,)), dtype=rdtype)
        # self.f_TENDq_pl  = np.empty((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), dtype=rdtype)

        # # Saved prognostic/tracer variables
        # self.PROG00      = np.empty((adm.ADM_shape + (6,)), dtype=rdtype)
        # self.PROG00_pl   = np.empty((adm.ADM_shape_pl + (6,)), dtype=rdtype)
        # self.PROGq00     = np.empty((adm.ADM_shape + (rcnf.TRC_vmax,)), dtype=rdtype)
        # self.PROGq00_pl  = np.empty((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), dtype=rdtype)
        # self.PROG0       = np.empty((adm.ADM_shape + (6,)), dtype=rdtype)
        # self.PROG0_pl    = np.empty((adm.ADM_shape_pl + (6,)), dtype=rdtype)

        # # Split prognostic variables
        # self.PROG_split     = np.empty((adm.ADM_shape + (6,)), dtype=rdtype)
        # self.PROG_split_pl  = np.empty((adm.ADM_shape_pl + (6,)), dtype=rdtype)

        # # Mean prognostic variables
        # self.PROG_mean      = np.empty((adm.ADM_shape + (5,)), dtype=rdtype)
        # self.PROG_mean_pl   = np.empty((adm.ADM_shape_pl + (5,)), dtype=rdtype)

        # # For tracer advection (large step)
        # self.f_TENDrho_mean     = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.f_TENDrho_mean_pl  = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # self.f_TENDq_mean       = np.empty((adm.ADM_shape + (rcnf.TRC_vmax,)), dtype=rdtype)
        # self.f_TENDq_mean_pl    = np.empty((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), dtype=rdtype)
        # self.PROG_mean_mean     = np.empty((adm.ADM_shape + (5,)), dtype=rdtype)
        # self.PROG_mean_mean_pl  = np.empty((adm.ADM_shape_pl + (5,)), dtype=rdtype)

        # # Diagnostic and tracer variables
        # self.DIAG     = np.empty((adm.ADM_shape + (6,)), dtype=rdtype)
        # self.DIAG_pl  = np.empty((adm.ADM_shape_pl + (6,)), dtype=rdtype)
        # self.q        = np.empty((adm.ADM_shape + (rcnf.TRC_vmax,)), dtype=rdtype)
        # self.q_pl     = np.empty((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), dtype=rdtype)

        # # Density
        # self.rho      = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.rho_pl   = np.empty((adm.ADM_shape_pl), dtype=rdtype)

        # # Internal energy (physical)
        # self.ein      = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.ein_pl   = np.empty((adm.ADM_shape_pl), dtype=rdtype)

        # # Enthalpy (physical)
        # self.eth      = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.eth_pl   = np.empty((adm.ADM_shape_pl), dtype=rdtype)

        # # Potential temperature (physical)
        # self.th       = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.th_pl    = np.empty((adm.ADM_shape_pl), dtype=rdtype)

        # # Density deviation from base state
        # self.rhogd    = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.rhogd_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)

        # # Pressure deviation from base state
        # self.pregd    = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.pregd_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)

        # # Temporary variables
        # self.qd       = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.qd_pl    = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # self.cv       = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.cv_pl    = np.empty((adm.ADM_shape_pl), dtype=rdtype)

        return
    

#    def dynamics_setup(self, fname_in, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, bndc, bsst, numf, vi, rdtype):
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
        #numf.numfilter_setup(fname_in, rcnf, cnst, comm, gtl, grd, gmtr, oprt, vmtr, tim, prgv, tdyn, frc, bndc, bsst, rdtype)
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
                          
    #def dynamics_step(self, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, bndc, cnvv, bsst, numf, vi, src, srctr, trcadv, rdtype):
    def dynamics_step(self, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, bndc, cnvv, bsst, numf, vi, src, srctr, trcadv, rdtype):

        # Make views of arrays

        #---< work array for the dynamics >---
        numerator = self._numerator_w   
        denominator = self._denominator_w
        numerator_pl = self._numerator_pl_w
        denominator_pl = self._denominator_pl_w

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
        qd       = self.qd
        qd_pl    = self.qd_pl
        cv       = self.cv
        cv_pl    = self.cv_pl

        prf.PROF_rapstart('__Dynamics', 1)
        prf.PROF_rapstart('___Pre_Post', 1)

        gall = adm.ADM_gall
        #gall_1d = adm.ADM_gall_1d
        kall = adm.ADM_kall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        nall = rcnf.TRC_vmax
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

        PROG[:, :, :, :, :]  = prgv.PRG_var[:, :, :, :, 0:6]
        PROG_pl[:, :, :, :]  = prgv.PRG_var_pl[:, :, :, 0:6]
        PROGq[:, :, :, :, :] = prgv.PRG_var[:, :, :, :, 6:]
        PROGq_pl[:, :, :, :] = prgv.PRG_var_pl[:, :, :, 6:]

        prf.PROF_rapend('___Pre_Post', 1)

        for ndyn in range(rcnf.DYN_DIV_NUM):

            #--- save the value before tracer advection
            if (not self.trcadv_out_dyndiv) or (ndyn == 0):

                PROG00 = PROG.copy()
                PROG00_pl = PROG_pl.copy()

                if rcnf.TRC_ADV_TYPE == 'DEFAULT':
                    PROGq00 = PROGq.copy()
                    PROGq00_pl = PROGq_pl.copy()
                #endif
            #endif

            #--- save the value before RK loop
            PROG0 = PROG.copy()
            PROG0_pl = PROG_pl.copy()


            if tim.TIME_integ_type == 'TRCADV':      # TRC-ADV Test Bifurcation

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
                        #print("BEFORETRACER: R     :", PROGq [6,5,3,1,2:],  file=log_file)
                        #print("BEFORETRACER: R k0-4:", PROGq [6,5,:4,1,2],  file=log_file)
                        #print("BEFORETRACER: R 0071:", PROGq [0,0, 7,1,2:],  file=log_file)
                        #print("BEFORETRACER: Pole 0:", PROGq_pl[0,10,0,2:], file=log_file)
                        #print("BEFORETRACER: Pole 1:", PROGq_pl[1,10,0,2:], file=log_file)
                        #print("BEFORETRACER: Pole 2:", PROGq_pl[2,10,0,2:], file=log_file)
                    #print("BEFORETRACER: R     :", PROGq  [6,5,3,1,2:],  file=log_file)
                    #print("BEFORETRACER: R     :", PROGq [6,5,10,0,2:],  file=log_file)
                    # print("BEFORETRACER: R     :", PROGq  [6,5,3,1,2:],  file=log_file)
                    # print("BEFORETRACER: R k0-4:", PROGq  [6,5,:4,1,2],  file=log_file)
                    # print("BEFORETRACER: R 0071:", PROGq [0,0, 7,1,2:],  file=log_file)
                    # print("BEFORETRACER: Pole 0:", PROGq_pl[0,10,0,2:], file=log_file)
                    # print("BEFORETRACER: Pole 1:", PROGq_pl[1,10,0,2:], file=log_file)
                    # print("BEFORETRACER: Pole 2:", PROGq_pl[2,10,0,2:], file=log_file)

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

                with open(std.fname_log, 'a') as log_file:
                    if prc.prc_myrank == 2:
                        print("AFTERTRACER: r11, z24  SE inner 15  :", PROGq [15,:,24,1,1],  file=log_file)
                        print("AFTERTRACER: r11, z24  SE inner 16  :", PROGq [16,:,24,1,1],  file=log_file)
                        print("AFTERTRACER: r11, z24  SE edge  17  :", PROGq [17,:,24,1,1],  file=log_file)
                    elif prc.prc_myrank == 6:
                        print("AFTERTRACER: r30, z24  NW inner  2  :", PROGq [2,:,24,0,1],  file=log_file)
                        print("AFTERTRACER: r30, z24  NW inner  1  :", PROGq [1,:,24,0,1],  file=log_file)
                        print("AFTERTRACER: r30, z24  NW edge   0  :", PROGq [0,:,24,0,1],  file=log_file)

                # with open(std.fname_log,'a') as log_file:
                #     #print("AFTERTRACER: R     :", PROGq [6,5,10,0,2:],  file=log_file)
                #     print("AFTERTRACER:  R     :", PROGq [6,5,3,1,2:],  file=log_file)
                #     print("AFTERTRACER:  R k0-4:", PROGq [6,5,:4,1,2],  file=log_file)
                #     print("AFTERTRACER:  R 0071:", PROGq [0,0, 7,1,2:],  file=log_file)
                #     print("AFTERTRACER:  Pole 0:", PROGq_pl[0,10,0,2:], file=log_file)
                #     print("AFTERTRACER:  Pole 1:", PROGq_pl[1,10,0,2:], file=log_file)
                #     print("AFTERTRACER:  Pole 2:", PROGq_pl[2,10,0,2:], file=log_file)

                prf.PROF_rapend('__Tracer_Advection', 1)
                
                #skip for now (not needed for JW test)
                frc.forcing_update( PROG, PROG_pl,  # [INOUT]
                                    cnst, rcnf, grd, tim, trcadv, rdtype,
                                    ) 

                # with open(std.fname_log, 'a') as log_file:
                #     #print("AFTERFUPDATE: R     :", PROG [6,5,10,0,:], file=log_file)
                #     print("AFTERFUPDATE: R     :", PROG [6,5,3,1,:], file=log_file)
                #     print("AFTERFUPDATE: R 0071:", PROG [0,0, 7,1,:], file=log_file)
                #     print("AFTERFUPDATE: Pole 0:", PROG_pl[0,10,0,:], file=log_file)
                #     print("AFTERFUPDATE: Pole 1:", PROG_pl[1,10,0,:], file=log_file)
                #     print("AFTERFUPDATE: Pole 2:", PROG_pl[2,10,0,:], file=log_file)

            # endif


            #---------------------------------------------------------------------------
            #
            #> Start large time step integration
            #
            #---------------------------------------------------------------------------
            for nl in range(self.num_of_iteration_lstep):

                prf.PROF_rapstart('___Pre_Post',1)

        
                # print("in lstep loop, nl = ", nl, "/", self.num_of_iteration_lstep -1) 
                # print("stopping the program AaA")
                # prc.prc_mpifinish(std.io_l, std.fname_log)
                # import sys 
                # sys.exit()

                with open(std.fname_log, 'a') as log_file:
                    print("lstep starting, iteration number: ", nl, "/", self.num_of_iteration_lstep -1, file=log_file)

                #---< Generate diagnostic values and set the boudary conditions
                
                # Extract variables
                RHOG    = PROG[:, :, :, :, I_RHOG]
                RHOGVX  = PROG[:, :, :, :, I_RHOGVX]
                RHOGVY  = PROG[:, :, :, :, I_RHOGVY]
                RHOGVZ  = PROG[:, :, :, :, I_RHOGVZ]
                RHOGE   = PROG[:, :, :, :, I_RHOGE]

                rho[:, :, :, :] = RHOG / vmtr.VMTR_GSGAM2
                DIAG[:, :, :, :, I_vx] = RHOGVX / RHOG      # zero devide encountered in 2nd or 3rd loop?

                #np.seterr(under='ignore')
                DIAG[:, :, :, :, I_vy] = RHOGVY / RHOG
                DIAG[:, :, :, :, I_vz] = RHOGVZ / RHOG
                ein[:, :, :, :] = RHOGE / RHOG

                q[:, :, :, :, :] = PROGq / PROG[:, :, :, :, np.newaxis, I_RHOG]
                #np.seterr(under='raise')
                # with open (std.fname_log, 'a') as log_file:
                #     print("ZEROsearch",file=log_file) 
                #     print(RHOG[16, 0, 41, 0], RHOG[16, 0, 40, 0],file=log_file)
                #     print(RHOG[17, 0, 41, 0], RHOG[17, 0, 40, 0],file=log_file)

                # Preallocated arrays: cv, qd, q, ein, rho, DIAG all have shape (i, j, k, l [, nq])
                # q has shape: (i, j, k, l, nq)

                # Reset cv and qd
                cv.fill(rdtype(0.0))
                qd.fill(rdtype(1.0))

                # Slice tracers from nmin to nmax
                q_slice = q[:, :, :, :, nmin:nmax+1]                # shape: (i, j, k, l, nq_range)
                CVW_slice = CVW[nmin:nmax+1]                        # shape: (nq_range,)

                # Accumulate cv and qd over tracer range
                cv += np.sum(q_slice * CVW_slice[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :], axis=4)
                qd -= np.sum(q_slice, axis=4)

                # Add dry-air contribution to cv
                cv += qd * CVdry

                # Compute temperature

                # mask_zero = cv == 0
                # zero_indices = np.argwhere(mask_zero)
                # with open(std.fname_log, 'a') as log_file:
                #     if zero_indices.size > 0:
                #         print(f"Zero division risk at {len(zero_indices)} locations:", file=log_file)
                #         for idx in zero_indices:
                #             print(f"cv is zero at index {tuple(idx)}", file=log_file)
                #     else:
                #         print("No zero values found in cv.", file=log_file)
                #     print("CVvalueF:", cv[0,0,6,1], cv[0,0,7,1], cv[1,1,6,1], cv[1,1,7,1], file=log_file)
                #     print("qd, CVdry:", qd[0,0,6,1], qd[0,0,7,1], qd[1,1,6,1], qd[1,1,7,1], CVdry, file=log_file)

                    # huge values in qd found. (e+19 to e+23)
                    # print("q[0,0,6,1,:]  ", q[0, 0, 6, 1, :], file=log_file)
                    # print("q[0,0,7,1,:]  ", q[0, 0, 7, 1, :], file=log_file)
                    # print("q[1,1,6,1,:]  ", q[1, 1, 6, 1, :], file=log_file)
                    # print("q[1,1,7,1,:]  ", q[1, 1, 7, 1, :], file=log_file)

                    # print("PROGq[0,0,6,1,:]  ", PROGq[0, 0, 6, 1, :], file=log_file)
                    # print("PROGq[0,0,7,1,:]  ", PROGq[0, 0, 7, 1, :], file=log_file)
                    # print("PROGq[1,1,6,1,:]  ", PROGq[1, 1, 6, 1, :], file=log_file)
                    # print("PROGq[1,1,7,1,:]  ", PROGq[1, 1, 7, 1, :], file=log_file)

                    # print("prgv.PRG_var[0,0,6,1,:6]  ", prgv.PRG_var[0, 0, 6, 1, :6], file=log_file)
                    # print("prgv.PRG_var[0,0,7,1,:6]  ", prgv.PRG_var[0, 0, 7, 1, :6], file=log_file)
                    # print("prgv.PRG_var[1,1,6,1,:6]  ", prgv.PRG_var[1, 1, 6, 1, :6], file=log_file)
                    # print("prgv.PRG_var[1,1,7,1,:6]  ", prgv.PRG_var[1, 1, 7, 1, :6], file=log_file)
                    # print("prgv.PRG_var[0,0,6,1,6:]  ", prgv.PRG_var[0, 0, 6, 1, 6:], file=log_file)
                    # print("prgv.PRG_var[0,0,7,1,6:]  ", prgv.PRG_var[0, 0, 7, 1, 6:], file=log_file)
                    # print("prgv.PRG_var[1,1,6,1,6:]  ", prgv.PRG_var[1, 1, 6, 1, 6:], file=log_file)
                    # print("prgv.PRG_var[1,1,7,1,6:]  ", prgv.PRG_var[1, 1, 7, 1, 6:], file=log_file)

                    #prgv.PRG_var[:, :, :, :, 6:]                    # print("PROGq[0,0,6,1,:]  ", PROGq[0, 0, 6, 1, :], file=log_file)
                    # print("PROGq[0,0,7,1,:]  ", PROGq[0, 0, 7, 1, :], file=log_file)
                    # print("PROGq[1,1,6,1,:]  ", PROGq[1, 1, 6, 1, :], file=log_file)
                    # print("PROGq[1,1,7,1,:]  ", PROGq[1, 1, 7, 1, :], file=log_file)

                DIAG[:, :, :, :, I_tem] = ein / cv     ###### zero devide encountered in 2nd lstep, after src_tracer implemented  JJJ   # zero devide in 14th step, for SP $$$

                # Compute pressure
                DIAG[:, :, :, :, I_pre] = rho * DIAG[:, :, :, :, I_tem] * (qd * Rdry + q[:, :, :, :, iqv] * Rvap)

                numerator[:, :, :, :] = PROG[:, :, kmin+1:kmax+1, :, I_RHOGW]
                rhog_k   = PROG[:, :, kmin+1:kmax+1, :, I_RHOG]
                rhog_km1 = PROG[:, :, kmin:kmax,     :, I_RHOG]
                fact1 = vmtr.VMTR_C2Wfact[:, :, kmin+1:kmax+1, :, 0]
                fact2 = vmtr.VMTR_C2Wfact[:, :, kmin+1:kmax+1, :, 1]
                denominator[:, :, :, :] = fact1 * rhog_k + fact2 * rhog_km1
                DIAG[:, :, kmin+1:kmax+1, :, I_w] = numerator / denominator

                # Task1
                #print("Task1a done")
                #np.seterr(under='ignore')
                bndc.BNDCND_all(
                    adm.ADM_gall_1d, 
                    adm.ADM_gall_1d, 
                    adm.ADM_kall, 
                    adm.ADM_lall,
                    rho,                        
                    DIAG[:, :, :, :, I_vx],     
                    DIAG[:, :, :, :, I_vy],     
                    DIAG[:, :, :, :, I_vz],     
                    DIAG[:, :, :, :, I_w],      
                    ein,
                    DIAG[:, :, :, :, I_tem], 
                    DIAG[:, :, :, :, I_pre],
                    PROG[:, :, :, :, I_RHOG],
                    PROG[:, :, :, :, I_RHOGVX],
                    PROG[:, :, :, :, I_RHOGVY],
                    PROG[:, :, :, :, I_RHOGVZ],
                    PROG[:, :, :, :, I_RHOGW], 
                    PROG[:, :, :, :, I_RHOGE],  
                    vmtr.VMTR_GSGAM2, 
                    vmtr.VMTR_PHI, 
                    vmtr.VMTR_C2Wfact, 
                    vmtr.VMTR_C2WfactGz,
                    cnst,
                    rdtype,
                )
                #np.seterr(under='raise')

                # if nl == 2:
                #     print("in lstep loop, nl = ", nl)
                #     ic= 6
                #     jc= 5
                #     kc= 41
                #     lc= 0
                #     with open (std.fname_log, 'a') as log_file:
                #         print("in 2nd lstep loop, nl = ", nl, file=log_file)
                #         print(f"DIAG[{ic}, {jc}, {kc}, {lc}, I_vx]",     DIAG[ic, jc, kc, lc, I_vx], file=log_file)
                #         print(f"DIAG[{ic}, {jc}, {kc}, {lc}, I_vy]",     DIAG[ic, jc, kc, lc, I_vy], file=log_file)
                #         print(f"DIAG[{ic}, {jc}, {kc}, {lc}, I_vz]",     DIAG[ic, jc, kc, lc, I_vz], file=log_file)
                #         print(f"DIAG[{ic}, {jc}, {kc}, {lc}, I_w]",      DIAG[ic, jc, kc, lc, I_w], file=log_file)
                #         print(f"DIAG[{ic}, {jc}, {kc}, {lc}, I_tem]",    DIAG[ic, jc, kc, lc, I_tem], file=log_file)
                #         print(f"DIAG[{ic}, {jc}, {kc}, {lc}, I_pre]",    DIAG[ic, jc, kc, lc, I_pre], file=log_file)
                #         print(f"PROG[{ic}, {jc}, {kc}, {lc}, I_RHOG]",   PROG[ic, jc, kc, lc, I_RHOG], file=log_file)
                #         print(f"PROG[{ic}, {jc}, {kc}, {lc}, I_RHOGVX]", PROG[ic, jc, kc, lc, I_RHOGVX], file=log_file)
                #         print(f"PROG[{ic}, {jc}, {kc}, {lc}, I_RHOGVY]", PROG[ic, jc, kc, lc, I_RHOGVY], file=log_file)
                #         print(f"PROG[{ic}, {jc}, {kc}, {lc}, I_RHOGVZ]", PROG[ic, jc, kc, lc, I_RHOGVZ], file=log_file)
                #         print(f"PROG[{ic}, {jc}, {kc}, {lc}, I_RHOGW]",  PROG[ic, jc, kc, lc, I_RHOGW], file=log_file)
                #         print(f"PROG[{ic}, {jc}, {kc}, {lc}, I_RHOGE]",  PROG[ic, jc, kc, lc, I_RHOGE], file=log_file)


                    # prc.prc_mpifinish(std.io_l, std.fname_log)
                    # print("stopping the program AAAA")
                    # import sys 
                    # sys.exit()


                #call BNDCND_all

                # Task2
                #print("Task2a done but not tested yet")
                th = tdyn.THRMDYN_th( 
                        DIAG[:, :, :, :, I_tem], 
                        DIAG[:, :, :, :, I_pre],
                        cnst,
                )
                
                # Task3
                #print("Task3a done but not tested yet")
                eth = tdyn.THRMDYN_eth(
                        ein,
                        DIAG[:, :, :, :, I_pre],
                        rho,
                        cnst,
                )


                # perturbations ( pre, rho with metrics )
                pregd[:, :, :, :] = (DIAG[:, :, :, :, I_pre] - pre_bs) * vmtr.VMTR_GSGAM2
                rhogd[:, :, :, :] = (rho                  - rho_bs) * vmtr.VMTR_GSGAM2



                # if prc.prc_myrank == 0:
                #         print("I am in dynamics_step  0-0")
                #         print(grd.GRD_x[6, 5, 0, 0, grd.GRD_XDIR])#, file=log_file)
                #         print(grd.GRD_x[6, 5, 0, 0, grd.GRD_YDIR])#, file=log_file)
                #         print(grd.GRD_x[6, 5, 0, 0, grd.GRD_ZDIR])#, file=log_file)
                #         #prc.prc_mpistop(std.io_l, std.fname_log)

                if adm.ADM_have_pl:

                    rho_pl = PROG_pl[:, :, :, I_RHOG]   / vmtr.VMTR_GSGAM2_pl
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

                    # with open(std.fname_log, 'a') as log_file:
                    #     print("before BNDCND_all fore POLE", file=log_file)
                    #     print("rho_pl[:, 41, 0]", rho_pl[:, 41, 0], file=log_file)
                    #     print("DIAG_pl[:, 41, 0, I_vx]", DIAG_pl[:, 41, 0, I_vx], file=log_file)
                    #     print("DIAG_pl[:, 41, 0, I_vy]", DIAG_pl[:, 41, 0, I_vy], file=log_file)
                    #     print("DIAG_pl[:, 41, 0, I_vz]", DIAG_pl[:, 41, 0, I_vz], file=log_file)
                    #     print("DIAG_pl[:, 41, 0, I_w]", DIAG_pl[:, 41, 0, I_w], file=log_file)
                    #     print("ein_pl[:, 41, 0]", ein_pl[:, 41, 0], file=log_file)
                    #     print("DIAG_pl[:, 41, 0, I_tem]", DIAG_pl[:, 41, 0, I_tem], file=log_file)
                    #     print("DIAG_pl[:, 41, 0, I_pre]", DIAG_pl[:, 41, 0, I_pre], file=log_file)
                    #     print("PROG_pl[:, 41, 0, I_RHOG]", PROG_pl[:, 41, 0, I_RHOG], file=log_file)
                    #     print("PROG_pl[:, 41, 0, I_RHOGVX]", PROG_pl[:, 41, 0, I_RHOGVX], file=log_file)
                    #     print("PROG_pl[:, 41, 0, I_RHOGVY]", PROG_pl[:, 41, 0, I_RHOGVY], file=log_file)
                    #     print("PROG_pl[:, 41, 0, I_RHOGVZ]", PROG_pl[:, 41, 0, I_RHOGVZ], file=log_file)
                    #     print("PROG_pl[:, 41, 0, I_RHOGW]", PROG_pl[:, 41, 0, I_RHOGW], file=log_file)
                    #     print("PROG_pl[:, 41, 0, I_RHOGE]", PROG_pl[:, 41, 0, I_RHOGE], file=log_file)
                    #     print("vmtr.VMTR_GSGAM2_pl[:, 41, 0]", vmtr.VMTR_GSGAM2_pl[:, 41, 0], file=log_file)
                    #     print("vmtr.VMTR_PHI_pl[:, 41, 0]", vmtr.VMTR_PHI_pl[:, 41, 0], file=log_file)
                    #     print("vmtr.VMTR_C2Wfact_pl[:, 41, 0, :]", vmtr.VMTR_C2Wfact_pl[:, 41, 0, :], file=log_file)
                    #     print("vmtr.VMTR_C2WfactGz_pl[:, 41, 0, :]", vmtr.VMTR_C2WfactGz_pl[:, 41, 0, :], file=log_file)

                    # Task1b
                    #print("Task1b done")
                    #np.seterr(under='ignore')
                    bndc.BNDCND_all_pl(
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
                    # changed to using func_pl, because np.newaxis sometimes cause issues when using func
                    # probably giving a dummy dimension for poles in the entire code would be better

                    # with open(std.fname_log, 'a') as log_file:
                    #     print("after BNDCND_all fore POLE", file=log_file)
                    #     print("rho_pl[:, 41, 0]", rho_pl[:, 41, 0], file=log_file)
                    #     print("DIAG_pl[:, 41, 0, I_vx]", DIAG_pl[:, 41, 0, I_vx], file=log_file)
                    #     print("DIAG_pl[:, 41, 0, I_vy]", DIAG_pl[:, 41, 0, I_vy], file=log_file)
                    #     print("DIAG_pl[:, 41, 0, I_vz]", DIAG_pl[:, 41, 0, I_vz], file=log_file)
                    #     print("DIAG_pl[:, 41, 0, I_w]", DIAG_pl[:, 41, 0, I_w], file=log_file)
                    #     print("ein_pl[:, 41, 0]", ein_pl[:, 41, 0], file=log_file)
                    #     print("DIAG_pl[:, 41, 0, I_tem]", DIAG_pl[:, 41, 0, I_tem], file=log_file)
                    #     print("DIAG_pl[:, 41, 0, I_pre]", DIAG_pl[:, 41, 0, I_pre], file=log_file)
                    #     print("PROG_pl[:, 41, 0, I_RHOG]", PROG_pl[:, 41, 0, I_RHOG], file=log_file)
                    #     print("PROG_pl[:, 41, 0, I_RHOGVX]", PROG_pl[:, 41, 0, I_RHOGVX], file=log_file)
                    #     print("PROG_pl[:, 41, 0, I_RHOGVY]", PROG_pl[:, 41, 0, I_RHOGVY], file=log_file)
                    #     print("PROG_pl[:, 41, 0, I_RHOGVZ]", PROG_pl[:, 41, 0, I_RHOGVZ], file=log_file)
                    #     print("PROG_pl[:, 41, 0, I_RHOGW]", PROG_pl[:, 41, 0, I_RHOGW], file=log_file)
                    #     print("PROG_pl[:, 41, 0, I_RHOGE]", PROG_pl[:, 41, 0, I_RHOGE], file=log_file)
                      
                    # Assign modified slices back to the original arrays (not needed for read-only views)
                    # Note: This triggers a copy operation. I think the effect is minimal because this is only for the poles.
                    #       However, it may be better to have a size 1 dummy dimension for poles throughout the entire code.
                    #       Then the expand/squeeze can be avoided, keeping the code cleaner. Consider this in the future.
                    #           Or, this is completely unnecessary. Seems to be working without it.
            

                    # Task2
                    # #print("Task2b done but not tested yet")
                    # th_pl = tdyn.THRMDYN_th(
                    #     adm.ADM_gall_pl, 
                    #     1, 
                    #     adm.ADM_kall, 
                    #     adm.ADM_lall_pl, 
                    #     DIAG_pl[:, np.newaxis, :, :, I_tem], 
                    #     DIAG_pl[:, np.newaxis, :, :, I_pre],
                    #     cnst,
                    # )
                    # th_pl = np.squeeze(th_pl, axis=1) # removing dummy dimension

                    # This function should work without newaxis
                    th_pl = tdyn.THRMDYN_th(
                        DIAG_pl[:, :, :, I_tem], 
                        DIAG_pl[:, :, :, I_pre],
                        cnst,
                    )
                    
                    
                    # Task3
                    #print("Task3b done but not tested yet")
                    # eth_pl = tdyn.THRMDYN_eth(
                    #     adm.ADM_gall_pl, 
                    #     1, 
                    #     adm.ADM_kall, 
                    #     adm.ADM_lall_pl, 
                    #     ein_pl [:, np.newaxis, :, :],  
                    #     DIAG_pl[:, np.newaxis, :, :, I_pre],
                    #     rho_pl [:, np.newaxis, :, :], 
                    #     cnst,
                    # )
                    # eth_pl = np.squeeze(eth_pl, axis=1) # removing dummy dimension

                    # This function should work without newaxis
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

                prf.PROF_rapend('___Pre_Post',1)
                #------------------------------------------------------------------------
                #> LARGE step
                #------------------------------------------------------------------------
                prf.PROF_rapstart('___Large_step', 1)

                # if prc.prc_myrank == 0:
                #     print("I am in dynamics_step  0-0-1")
                #     print(grd.GRD_x[6, 5, 0, 0, grd.GRD_XDIR])#, file=log_file)
                #     print(grd.GRD_x[6, 5, 0, 0, grd.GRD_YDIR])#, file=log_file)
                #     print(grd.GRD_x[6, 5, 0, 0, grd.GRD_ZDIR])#, file=log_file)
                #     #prc.prc_mpistop(std.io_l, std.fname_log)


                #--- calculation of advection tendency including Coriolis force
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
                )
                #np.seterr(under='raise')

                # if prc.prc_myrank == 0:
                #         print("I am in dynamics_step  0-1")
                #         print(grd.GRD_x[6, 5, 0, 0, grd.GRD_XDIR])#, file=log_file)
                #         print(grd.GRD_x[6, 5, 0, 0, grd.GRD_YDIR])#, file=log_file)
                #         print(grd.GRD_x[6, 5, 0, 0, grd.GRD_ZDIR])#, file=log_file)
                #         prc.prc_mpistop(std.io_l, std.fname_log)


                # with open(std.fname_log, 'a') as log_file:  
                #     print("g_TEND 1st (6,5,2,0)", g_TEND[6, 5, 2, 0, I_RHOGVX:I_RHOGE], file=log_file) 
                #     print("g_TEND 1st (5,6,2,0)", g_TEND[5, 6, 2, 0, I_RHOGVX:I_RHOGE], file=log_file) 



                g_TEND[:, :, :, :, I_RHOG]  = rdtype(0.0)
                g_TEND[:, :, :, :, I_RHOGE] = rdtype(0.0)

                # Zero out specific components of g_TEND_pl
                g_TEND_pl[:, :, :, I_RHOG]  = rdtype(0.0)
                g_TEND_pl[:, :, :, I_RHOGE] = rdtype(0.0)


                # with open(std.fname_log, 'a') as log_file:
                #     kc= 41
                #     print(f"DIAG_pl[0, {kc}, 0, :]", DIAG_pl[0, kc, 0, :], file=log_file)
                #     print(f"DIAG_pl[1, {kc}, 0, :]", DIAG_pl[1, kc, 0, :], file=log_file)
                #     print(f"DIAG_pl[2, {kc}, 0, :]", DIAG_pl[2, kc, 0, :], file=log_file)
                #     print(f"DIAG_pl[3, {kc}, 0, :]", DIAG_pl[3, kc, 0, :], file=log_file)
                #     print(f"DIAG_pl[4, {kc}, 0, :]", DIAG_pl[4, kc, 0, :], file=log_file)
                #     print(f"DIAG_pl[5, {kc}, 0, :]", DIAG_pl[5, kc, 0, :], file=log_file)

                #     print(f"PROG_pl[0, {kc}, 0, :]", PROG_pl[0, kc, 0, :], file=log_file)
                #     print(f"PROG_pl[1, {kc}, 0, :]", PROG_pl[1, kc, 0, :], file=log_file)
                #     print(f"PROG_pl[2, {kc}, 0, :]", PROG_pl[2, kc, 0, :], file=log_file)
                #     print(f"PROG_pl[3, {kc}, 0, :]", PROG_pl[3, kc, 0, :], file=log_file)
                #     print(f"PROG_pl[4, {kc}, 0, :]", PROG_pl[4, kc, 0, :], file=log_file)
                #     print(f"PROG_pl[5, {kc}, 0, :]", PROG_pl[5, kc, 0, :], file=log_file)    # DAIG_pl and PROG_pl ALL good at 2,0
                #     #print("g_TEND     (0,{kc}, 0,:)", g_TEND   [0, kc}, 0, :], file=log_file)       
                #     print(f"g_TEND_pl(0, {kc}, 0, :)", g_TEND_pl[0, kc, 0, :], file=log_file) 
                #     print(f"g_TEND_pl(1, {kc}, 0, :)", g_TEND_pl[1, kc, 0, :], file=log_file) 
                #     print(f"g_TEND_pl(2, {kc}, 0, :)", g_TEND_pl[2, kc, 0, :], file=log_file) 
                #     print(f"g_TEND_pl(3, {kc}, 0, :)", g_TEND_pl[3, kc, 0, :], file=log_file) 
                #     print(f"g_TEND_pl(4, {kc}, 0, :)", g_TEND_pl[4, kc, 0, :], file=log_file) 
                #     print(f"g_TEND_pl(5, {kc}, 0, :)", g_TEND_pl[5, kc, 0, :], file=log_file)  # sign reversed  --> coriolis bug fixed, now good.

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

                    # if prc.prc_myrank == 0:
                    #         print("I am in dynamics_step  1")
                    #         print(grd.GRD_x[6, 5, 0, 0, grd.GRD_XDIR])#, file=log_file)
                    #         print(grd.GRD_x[6, 5, 0, 0, grd.GRD_YDIR])#, file=log_file)
                    #         print(grd.GRD_x[6, 5, 0, 0, grd.GRD_ZDIR])#, file=log_file)
                    #         prc.prc_mpistop(std.io_l, std.fname_log)


                    #------ numerical diffusion

                    # if prc.prc_myrank == 0:
                    #     print("I am in dynamics step")
                    #     print(grd.GRD_x[6, 5, 0, 0, grd.GRD_XDIR])#, file=log_file)
                    #     print(grd.GRD_x[6, 5, 0, 0, grd.GRD_YDIR])#, file=log_file)
                    #     print(grd.GRD_x[6, 5, 0, 0, grd.GRD_ZDIR])#, file=log_file)
                    #     prc.prc_mpistop(std.io_l, std.fname_log)

                    # Task 5
#                    print("Task5")
                    #"Task5 done but not tested yet"
                    # with open(std.fname_log, 'a') as log_file:  
                    #     print("g_TEND check (6,5,2,0,:)", g_TEND[6, 5, 2, 0, :], file=log_file) 
                    #     print("going into numfilter_hdiffusion IN_LARGE_STEP2", file=log_file)
                    #np.seterr(under='ignore')
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
                    )
                    #np.seterr(under='raise')
                    # with open(std.fname_log, 'a') as log_file:  
                    #     print("f_TEND  numf (6,5,37,0,:)", f_TEND[6, 5, 37, 0, :], file=log_file) 
                    #     print("f_TENDq numf (6,5,37,0,:)", f_TENDq[6, 5, 37, 0, :],file=log_file) 

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

                # with open(std.fname_log, 'a') as log_file:  
                #     print("g_TEND_pl beforeadded (0,2,0,:)", g_TEND_pl[0, 2, 0, :], file=log_file) 
                #     print("g_TEND_pl beforeadded (1,2,0,:)", g_TEND_pl[1, 2, 0, :], file=log_file) 
                #     print("g_TEND_pl beforeadded (2,2,0,:)", g_TEND_pl[2, 2, 0, :], file=log_file) 
                #     print("g_TEND_pl beforeadded (3,2,0,:)", g_TEND_pl[3, 2, 0, :], file=log_file) 
                #     print("g_TEND_pl beforeadded (4,2,0,:)", g_TEND_pl[4, 2, 0, :], file=log_file) 
                #     print("g_TEND_pl beforeadded (5,2,0,:)", g_TEND_pl[5, 2, 0, :], file=log_file) 

                #     print("f_TEND_pl beforeadded (0,2,0,:)", f_TEND_pl[0, 2, 0, :], file=log_file) 
                #     print("f_TEND_pl beforeadded (1,2,0,:)", f_TEND_pl[1, 2, 0, :], file=log_file) 
                #     print("f_TEND_pl beforeadded (2,2,0,:)", f_TEND_pl[2, 2, 0, :], file=log_file) 
                #     print("f_TEND_pl beforeadded (3,2,0,:)", f_TEND_pl[3, 2, 0, :], file=log_file) 
                #     print("f_TEND_pl beforeadded (4,2,0,:)", f_TEND_pl[4, 2, 0, :], file=log_file) 
                #     print("f_TEND_pl beforeadded (5,2,0,:)", f_TEND_pl[5, 2, 0, :], file=log_file) 


                # with open(std.fname_log, 'a') as log_file:  
                #     print("g_TEND beforeadded (6,5,37,0,:)", g_TEND[6, 5, 37, 0, :], file=log_file) 
                #     print("f_TEND beforeadded (6,5,37,0,:)", f_TEND[6, 5, 37, 0, :], file=log_file) 
                    

                g_TEND[:, :, :, :, 0:6] += f_TEND[:, :, :, :, 0:6]

                # with open(std.fname_log, 'a') as log_file:  
                #     print("g_TEND afteradded (6,5,37,0,:)", g_TEND[6, 5, 37, 0, :], file=log_file) 

                g_TEND_pl += f_TEND_pl


                # with open(std.fname_log, 'a') as log_file:
                #     ic = 6
                #     jc = 5
                #     kc= 37
                #     lc= 1
                #     print("BEFOREsmallstep", file=log_file)

                #     print(f"DIAG[{ic}, {jc}, {kc}, {lc}, :]", DIAG[ic, jc, kc, lc, :], file=log_file)
                #     print(f"PROG[{ic}, {jc}, {kc}, {lc}, :]", PROG[ic, jc, kc, lc, :], file=log_file)
                #     print(f"g_TEND[{ic}, {jc}, {kc}, {lc}, :]", g_TEND[ic, jc, kc, lc, :], file=log_file)    # component 5 (6th) is quite different bt f and p when SP
                    
                #     if adm.ADM_have_pl:
                #         print(f"DIAG_pl[0, {kc}, {lc}, :]", DIAG_pl[0, kc, lc, :], file=log_file)
                #         print(f"DIAG_pl[1, {kc}, {lc}, :]", DIAG_pl[1, kc, lc, :], file=log_file)
                #         print(f"DIAG_pl[2, {kc}, {lc}, :]", DIAG_pl[2, kc, lc, :], file=log_file)
                #         print(f"DIAG_pl[3, {kc}, {lc}, :]", DIAG_pl[3, kc, lc, :], file=log_file)
                #         print(f"DIAG_pl[4, {kc}, {lc}, :]", DIAG_pl[4, kc, lc, :], file=log_file)
                #         print(f"DIAG_pl[5, {kc}, {lc}, :]", DIAG_pl[5, kc, lc, :], file=log_file)

                #         print(f"PROG_pl[0, {kc}, {lc}, :]", PROG_pl[0, kc, lc, :], file=log_file)
                #         print(f"PROG_pl[1, {kc}, {lc}, :]", PROG_pl[1, kc, lc, :], file=log_file)
                #         print(f"PROG_pl[2, {kc}, {lc}, :]", PROG_pl[2, kc, lc, :], file=log_file)
                #         print(f"PROG_pl[3, {kc}, {lc}, :]", PROG_pl[3, kc, lc, :], file=log_file)
                #         print(f"PROG_pl[4, {kc}, {lc}, :]", PROG_pl[4, kc, lc, :], file=log_file)
                #         print(f"PROG_pl[5, {kc}, {lc}, :]", PROG_pl[5, kc, lc, :], file=log_file)   

                #         print(f"g_TEND_pl(0, {kc}, {lc}, :)", g_TEND_pl[0, kc, lc, :], file=log_file) 
                #         print(f"g_TEND_pl(1, {kc}, {lc}, :)", g_TEND_pl[1, kc, lc, :], file=log_file) 
                #         print(f"g_TEND_pl(2, {kc}, {lc}, :)", g_TEND_pl[2, kc, lc, :], file=log_file) 
                #         print(f"g_TEND_pl(3, {kc}, {lc}, :)", g_TEND_pl[3, kc, lc, :], file=log_file) 
                #         print(f"g_TEND_pl(4, {kc}, {lc}, :)", g_TEND_pl[4, kc, lc, :], file=log_file) 
                #         print(f"g_TEND_pl(5, {kc}, {lc}, :)", g_TEND_pl[5, kc, lc, :], file=log_file)  


                prf.PROF_rapend('___Large_step',1)
                #------------------------------------------------------------------------
                #> SMALL step
                #------------------------------------------------------------------------
                prf.PROF_rapstart('___Small_step',1)

                if nl != 0:    
                    # Update split values
                    PROG_split[:, :, :, :, 0:6] = PROG0[:, :, :, :, 0:6] - PROG[:, :, :, :, 0:6]
                    PROG_split_pl[:, :, :, :] = PROG0_pl[:, :, :, :] - PROG_pl[:, :, :, :]
                else:
                    # Zero out split values
                    PROG_split[:, :, :, :, 0:6] = rdtype(0.0)
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

                # Task 6
#               print("Task6")
                #np.seterr(under='ignore')
                vi.vi_small_step(
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
                ) 
                #np.seterr(under='raise')
                #print("out of vi_small_step")
                #prc.prc_mpistop(std.io_l, std.fname_log)

                # with open(std.fname_log, 'a') as log_file:
                #     ic = 6
                #     jc = 5
                #     kc= 37
                #     lc= 1
                #     print("AFTERsmallstep", file=log_file)

                #     print(f"PROG[{ic}, {jc}, {kc}, {lc}, :]", PROG[ic, jc, kc, lc, :], file=log_file)    
                #     print(f"PROG_split[{ic}, {jc}, {kc}, {lc}, :]", PROG_split[ic, jc, kc, lc, :], file=log_file)
                #     print(f"PROG_mean [{ic}, {jc}, {kc}, {lc}, :]", PROG_mean [ic, jc, kc, lc, :], file=log_file)

                #     if adm.ADM_have_pl:
                #         print(f"PROG_pl[0, {kc}, {lc}, :]", PROG_pl[0, kc, lc, :], file=log_file)   
                #         print(f"PROG_pl[1, {kc}, {lc}, :]", PROG_pl[1, kc, lc, :], file=log_file)
                #         print(f"PROG_pl[2, {kc}, {lc}, :]", PROG_pl[2, kc, lc, :], file=log_file)
                #         print(f"PROG_pl[3, {kc}, {lc}, :]", PROG_pl[3, kc, lc, :], file=log_file)
                #         print(f"PROG_pl[4, {kc}, {lc}, :]", PROG_pl[4, kc, lc, :], file=log_file)
                #         print(f"PROG_pl[5, {kc}, {lc}, :]", PROG_pl[5, kc, lc, :], file=log_file)   
                        
                #         print(f"PROG_split_pl[0, {kc}, {lc}, :]", PROG_split_pl[0, kc, lc, :], file=log_file)
                #         print(f"PROG_split_pl[1, {kc}, {lc}, :]", PROG_split_pl[1, kc, lc, :], file=log_file)
                #         print(f"PROG_split_pl[2, {kc}, {lc}, :]", PROG_split_pl[2, kc, lc, :], file=log_file)
                #         print(f"PROG_split_pl[3, {kc}, {lc}, :]", PROG_split_pl[3, kc, lc, :], file=log_file)
                #         print(f"PROG_split_pl[4, {kc}, {lc}, :]", PROG_split_pl[4, kc, lc, :], file=log_file)
                #         print(f"PROG_split_pl[5, {kc}, {lc}, :]", PROG_split_pl[5, kc, lc, :], file=log_file)   
                        
                #         print(f"PROG_mean_pl[0, {kc}, {lc}, :]", PROG_mean_pl[0, kc, lc, :], file=log_file)
                #         print(f"PROG_mean_pl[1, {kc}, {lc}, :]", PROG_mean_pl[1, kc, lc, :], file=log_file)
                #         print(f"PROG_mean_pl[2, {kc}, {lc}, :]", PROG_mean_pl[2, kc, lc, :], file=log_file)
                #         print(f"PROG_mean_pl[3, {kc}, {lc}, :]", PROG_mean_pl[3, kc, lc, :], file=log_file)
                #         print(f"PROG_mean_pl[4, {kc}, {lc}, :]", PROG_mean_pl[4, kc, lc, :], file=log_file)
                #         print(f"PROG_mean_pl[5, {kc}, {lc}, :]", PROG_mean_pl[5, kc, lc, :], file=log_file)   

                

                # with open (std.fname_log, 'a') as log_file:
                #     print("ZEZE in lstep loop, nl = ", nl, file= log_file)
                #     for l in range(lall):
                #         if l == 0:
                #             print("l = ", l, file= log_file)
                #             print("RHOG", RHOG[17, :, 10, l], file= log_file) 
                #             #print("PROG I_RHOG", PROG[17, :, 10, l, I_RHOG], file= log_file)
                #             print("RHOG", RHOG[16, 1:17, 10, l], file= log_file) 
                #             # print("PROG I_RHOG", PROG[16, 1:17, 10, l, I_RHOG], file= log_file)
                #             print("RHOG", RHOG[0, :, 10, l+1], file= log_file)   # already corrupted here! region 1
                #             print("RHOG", RHOG[1, :, 10, l+1], file= log_file)   # already corrupted here! region 1
                #             print("RHOG", RHOG[10, :, 10, l+1], file= log_file)  # already corrupted here! region 1


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


                            # with open(std.fname_log, 'a') as log_file:
                            #     print("WWW0:PROG [0,0,6,1,:]  ", PROG[0, 0, 6, 1, :], file=log_file)
                            #     print("     PROG [0,0,7,1,:]  ", PROG[0, 0, 7, 1, :], file=log_file)
                            #     print("     PROG [1,1,6,1,:]  ", PROG[1, 1, 6, 1, :], file=log_file)
                            #     print("     PROG [1,1,7,1,:]  ", PROG[1, 1, 7, 1, :], file=log_file)
                            #     print("WWW0:PROGq[0,0,6,1,:]  ", PROGq[0, 0, 6, 1, :], file=log_file)
                            #     print("     PROGq[0,0,7,1,:]  ", PROGq[0, 0, 7, 1, :], file=log_file)
                            #     print("     PROGq[1,1,6,1,:]  ", PROGq[1, 1, 6, 1, :], file=log_file)
                            #     print("     PROGq[1,1,7,1,:]  ", PROGq[1, 1, 7, 1, :], file=log_file)

                            # with open (std.fname_log, 'a') as log_file:
                            #     print("partially tested, do not trust the tracer scheme just yet", file=log_file)                            
                            srctr.src_tracer_advection(
                                rcnf.TRC_vmax,                                                  # [IN]
                                PROGq       [:,:,:,:,:],        PROGq_pl      [:,:,:,:],        # [INOUT]    brakes at 0 0 6 1 et al. @rank0 in SP at step 14   
                                PROG00      [:,:,:,:,I_RHOG],   PROG00_pl     [:,:,:,I_RHOG],   # [IN]  
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
                            )                            
                
                            # with open(std.fname_log, 'a') as log_file:
                            #     print("WWW1:PROG [0,0,6,1,:]  ", PROG[0, 0, 6, 1, :], file=log_file)
                            #     print("     PROG [0,0,7,1,:]  ", PROG[0, 0, 7, 1, :], file=log_file)
                            #     print("     PROG [1,1,6,1,:]  ", PROG[1, 1, 6, 1, :], file=log_file)
                            #     print("     PROG [1,1,7,1,:]  ", PROG[1, 1, 7, 1, :], file=log_file)
                            #     print("WWW1:PROGq[0,0,6,1,:]  ", PROGq[0, 0, 6, 1, :], file=log_file)
                            #     print("     PROGq[0,0,7,1,:]  ", PROGq[0, 0, 7, 1, :], file=log_file)
                            #     print("     PROGq[1,1,6,1,:]  ", PROGq[1, 1, 6, 1, :], file=log_file)
                            #     print("     PROGq[1,1,7,1,:]  ", PROGq[1, 1, 7, 1, :], file=log_file)
                            #     print("     PROGq[1,1,5,1,:]  ", PROGq[1, 1, 5, 1, :], file=log_file)
                            #     print("     PROGq[1,1,8,1,:]  ", PROGq[1, 1, 8, 1, :], file=log_file)


                            PROGq[:, :, :, :, :] += large_step_dt * f_TENDq

                            if adm.ADM_have_pl:
                                PROGq_pl[:, :, :, :] += large_step_dt * f_TENDq_pl

                            # with open(std.fname_log, 'a') as log_file:
                            #     print("WWW2:PROG [0,0,6,1,:]  ", PROG[0, 0, 6, 1, :], file=log_file)
                            #     print("     PROG [0,0,7,1,:]  ", PROG[0, 0, 7, 1, :], file=log_file)
                            #     print("     PROG [1,1,6,1,:]  ", PROG[1, 1, 6, 1, :], file=log_file)
                            #     print("     PROG [1,1,7,1,:]  ", PROG[1, 1, 7, 1, :], file=log_file)
                            #     print("WWW2:PROGq[0,0,6,1,:]  ", PROGq[0, 0, 6, 1, :], file=log_file)
                            #     print("     PROGq[0,0,7,1,:]  ", PROGq[0, 0, 7, 1, :], file=log_file)
                            #     print("     PROGq[1,1,6,1,:]  ", PROGq[1, 1, 6, 1, :], file=log_file)
                            #     print("     PROGq[1,1,7,1,:]  ", PROGq[1, 1, 7, 1, :], file=log_file)

                                # for l in range(adm.ADM_lall):
                                #     #for k in range(adm.ADM_kall):
                                #     for j in range(adm.ADM_gall_1d):
                                #     for i in range(adm.ADM_gall_1d):
                                #         if PROGq[i, j, k, l, 0] > rdtype(0.0):
                                #             print(i, j ,k, l,PROGq[i, j, k, l,0]) 

                            # with open(std.fname_log, 'a') as log_file:
                            #     ic = 6
                            #     jc = 5
                            #     kc= 37
                            #     lc= 1
                            #     print(" ",file=log_file)
                            #     print("AFTERtracer GGG",file=log_file)
                            #     print(f"PROG      [{ic}, {jc}, {kc}, {lc}, :]", PROG[ic, jc, kc, lc, :], file=log_file)  
                            #     print(f"PROG_split[{ic}, {jc}, {kc}, {lc}, :]", PROG_split[ic, jc, kc, lc, :], file=log_file)
                            #     print(f"PROG_mean [{ic}, {jc}, {kc}, {lc}, :]", PROG_mean [ic, jc, kc, lc, :], file=log_file)
                            #     print(f"PROGq     [{ic}, {jc}, {kc}, {lc}, :]", PROGq[ic, jc, kc, lc, :], file=log_file) 
                                #print(f"f_TENDq   [{ic}, {jc}, {kc}, {lc}, :]", f_TENDq[ic, jc, kc, lc, :], file=log_file) 
                                

                                # if adm.ADM_have_pl:
                                #     print(f"PROG_pl [0, {kc}, {lc}, :]", PROG_pl [0, kc, lc, :], file=log_file)   
                                #     print(f"PROG_pl [1, {kc}, {lc}, :]", PROG_pl [1, kc, lc, :], file=log_file)
                                #     print(f"PROG_pl [2, {kc}, {lc}, :]", PROG_pl [2, kc, lc, :], file=log_file)
                                #     print(f"PROG_pl [3, {kc}, {lc}, :]", PROG_pl [3, kc, lc, :], file=log_file)
                                #     print(f"PROG_pl [4, {kc}, {lc}, :]", PROG_pl [4, kc, lc, :], file=log_file)
                                #     print(f"PROG_pl [5, {kc}, {lc}, :]", PROG_pl [5, kc, lc, :], file=log_file)   
                                #     print(f"PROGq_pl[0, {kc}, {lc}, :]", PROGq_pl[0, kc, lc, :], file=log_file)   
                                #     print(f"PROGq_pl[1, {kc}, {lc}, :]", PROGq_pl[1, kc, lc, :], file=log_file)
                                #     print(f"PROGq_pl[2, {kc}, {lc}, :]", PROGq_pl[2, kc, lc, :], file=log_file)
                                #     print(f"PROGq_pl[3, {kc}, {lc}, :]", PROGq_pl[3, kc, lc, :], file=log_file)
                                #     print(f"PROGq_pl[4, {kc}, {lc}, :]", PROGq_pl[4, kc, lc, :], file=log_file)
                                #     print(f"PROGq_pl[5, {kc}, {lc}, :]", PROGq_pl[5, kc, lc, :], file=log_file)
                                #     print(f"f_TENDq_pl[0, {kc}, {lc}, :]", f_TENDq_pl[0, kc, lc, :], file=log_file)   
                                #     print(f"f_TENDq_pl[1, {kc}, {lc}, :]", f_TENDq_pl[1, kc, lc, :], file=log_file)
                                #     print(f"f_TENDq_pl[2, {kc}, {lc}, :]", f_TENDq_pl[2, kc, lc, :], file=log_file)
                                #     print(f"f_TENDq_pl[3, {kc}, {lc}, :]", f_TENDq_pl[3, kc, lc, :], file=log_file)
                                #     print(f"f_TENDq_pl[4, {kc}, {lc}, :]", f_TENDq_pl[4, kc, lc, :], file=log_file)
                                #     print(f"f_TENDq_pl[5, {kc}, {lc}, :]", f_TENDq_pl[5, kc, lc, :], file=log_file)
                                # print(" ",file=log_file)
                                
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


                # with open (std.fname_log, 'a') as log_file:
                #     print("ZDZD in lstep loop, nl = ", nl, file= log_file)
                #     for l in range(lall):
                #         if l == 0:
                #             print("l = ", l, file= log_file)
                #             print("RHOG", RHOG[17, :, 10, l], file= log_file) 
                #             #print("PROG I_RHOG", PROG[17, :, 10, l, I_RHOG], file= log_file)
                #             print("RHOG", RHOG[16, 1:17, 10, l], file= log_file) 
                #             # print("PROG I_RHOG", PROG[16, 1:17, 10, l, I_RHOG], file= log_file)
                #             print("RHOG", RHOG[0, :, 10, l+1], file= log_file)   # already corrupted before data_transfer! region 1
                #             print("RHOG", RHOG[1, :, 10, l+1], file= log_file)   # already corrupted before data_transfer! region 1

                #------ Update
                if nl != self.num_of_iteration_lstep-1:   # ayashii
                    comm.COMM_data_transfer( PROG, PROG_pl )
                    with open(std.fname_log, 'a') as log_file:     
                        print("WOW11", file=log_file)      #came here 
                #endif

                prf.PROF_rapend  ('___Pre_Post',1)

                # with open (std.fname_log, 'a') as log_file:
                #     print("ZCZC in lstep loop, nl = ", nl, file= log_file)
                #     for l in range(lall):
                #         if l == 0:
                #             print("l = ", l, file= log_file)
                #             print("RHOG", RHOG[17, :, 10, l], file= log_file) 
                #             #print("PROG I_RHOG", PROG[17, :, 10, l, I_RHOG], file= log_file)
                #             print("RHOG", RHOG[16, 1:17, 10, l], file= log_file) 
                #             # print("PROG I_RHOG", PROG[16, 1:17, 10, l, I_RHOG], file= log_file)

            #end nl loop --- large step    <for nl in range(self.num_of_iteration_lstep):>



            # prc.prc_mpifinish(std.io_l, std.fname_log)
            # print("stopping the program AAA")
            # import sys 
            # sys.exit()

            #---------------------------------------------------------------------------
            #>  Tracer advection (out of the large step)
            #---------------------------------------------------------------------------

            if self.trcadv_out_dyndiv and ndyn == rcnf.DYN_DIV_NUM:

                with open(std.fname_log, 'a') as log_file:     
                    print("WOW12", file=log_file)
                
                prf.PROF_rapstart('___Tracer_Advection',1)
                print("not tested, do not trust the tracer scheme yet")
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

        # with open(std.fname_log, 'a') as log_file:
        #     print("BB:PROG [0,0,6,1,:]  ", PROG[0, 0, 6, 1, :], file=log_file)
        #     print("   PROG [0,0,7,1,:]  ", PROG[0, 0, 7, 1, :], file=log_file)
        #     print("   PROG [1,1,6,1,:]  ", PROG[1, 1, 6, 1, :], file=log_file)
        #     print("   PROG [1,1,7,1,:]  ", PROG[1, 1, 7, 1, :], file=log_file)
        #     print("BB:PROGq[0,0,6,1,:]  ", PROGq[0, 0, 6, 1, :], file=log_file)
        #     print("   PROGq[0,0,7,1,:]  ", PROGq[0, 0, 7, 1, :], file=log_file)
        #     print("   PROGq[1,1,6,1,:]  ", PROGq[1, 1, 6, 1, :], file=log_file)
        #     print("   PROGq[1,1,7,1,:]  ", PROGq[1, 1, 7, 1, :], file=log_file)
            # print("prgv.PRG_var[0,0,6,1,5:]  ", prgv.PRG_var[0, 0, 6, 1, 5:], file=log_file)
            # print("prgv.PRG_var[0,0,7,1,5:]  ", prgv.PRG_var[0, 0, 7, 1, 5:], file=log_file)
            # print("prgv.PRG_var[1,1,6,1,5:]  ", prgv.PRG_var[1, 1, 6, 1, 5:], file=log_file)
            # print("prgv.PRG_var[1,1,7,1,5:]  ", prgv.PRG_var[1, 1, 7, 1, 5:], file=log_file)
            # print("prgv.PRG_var[0,0,6,1,6:]  ", prgv.PRG_var[0, 0, 6, 1, 6:], file=log_file)
            # print("prgv.PRG_var[0,0,7,1,6:]  ", prgv.PRG_var[0, 0, 7, 1, 6:], file=log_file)
            # print("prgv.PRG_var[1,1,6,1,6:]  ", prgv.PRG_var[1, 1, 6, 1, 6:], file=log_file)
            # print("prgv.PRG_var[1,1,7,1,6:]  ", prgv.PRG_var[1, 1, 7, 1, 6:], file=log_file)


        prgv.PRG_var[:, :, :, :, 0:6] = PROG[:, :, :, :, :] 
        prgv.PRG_var_pl[:, :, :, 0:6] = PROG_pl[:, :, :, :]  
        prgv.PRG_var[:, :, :, :, 6:]  = PROGq[:, :, :, :, :]  
        prgv.PRG_var_pl[:, :, :, 6:]  = PROGq_pl[:, :, :, :] 

        with open(std.fname_log, 'a') as log_file:
            #ic = 6
            #jc = 5

            kc= 5
            lc= 0
            print(f"pre_comm: prgv.PRG_var_pl [1, {kc}, {lc}, :]", prgv.PRG_var_pl [1, kc, lc, :], file=log_file)
            print(f"pre_comm: prgv.PRG_var_pl [2, {kc}, {lc}, :]", prgv.PRG_var_pl [2, kc, lc, :], file=log_file)

        comm.COMM_data_transfer(prgv.PRG_var, prgv.PRG_var_pl)
        #This comm is done in prgvar_set in the original code. Is it really necessary? # results change very slightly.


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
            print(prgv.PRG_var[0, 0, kc, :, 5], file=log_file) 

            # pole check   
            # #if adm.ADM_have_pl:
            print(f"prgv.PRG_var_pl [0, {kc}, {lc}, :]", prgv.PRG_var_pl [0, kc, lc, :], file=log_file)   
            print(f"prgv.PRG_var_pl [1, {kc}, {lc}, :]", prgv.PRG_var_pl [1, kc, lc, :], file=log_file)
            print(f"prgv.PRG_var_pl [2, {kc}, {lc}, :]", prgv.PRG_var_pl [2, kc, lc, :], file=log_file)

            print(" ",file=log_file)


        # call prgvar_set( PROG(:,:,:,I_RHOG),   PROG_pl(:,:,:,I_RHOG),   & ! [IN]
        #              PROG(:,:,:,I_RHOGVX), PROG_pl(:,:,:,I_RHOGVX), & ! [IN]
        #              PROG(:,:,:,I_RHOGVY), PROG_pl(:,:,:,I_RHOGVY), & ! [IN]
        #              PROG(:,:,:,I_RHOGVZ), PROG_pl(:,:,:,I_RHOGVZ), & ! [IN]
        #              PROG(:,:,:,I_RHOGW),  PROG_pl(:,:,:,I_RHOGW),  & ! [IN]
        #              PROG(:,:,:,I_RHOGE),  PROG_pl(:,:,:,I_RHOGE),  & ! [IN]
        #              PROGq(:,:,:,:),       PROGq_pl(:,:,:,:)        ) ! [IN]

        prf.PROF_rapend  ('___Pre_Post',1)

        #
        #  Niwa [TM]
        #

        # with open(std.fname_log, 'a') as log_file:
        #     ic = 6
        #     jc = 5
        #     kc= 3
        #     lc= 1
        #     print(" ",file=log_file)
        #     print("ENDOF_largestep",file=log_file)
        #     print(f"PROG      [{ic}, {jc}, {kc}, {lc}, :]", PROG[ic, jc, kc, lc, :], file=log_file)  
        #     print(f"PROG_split[{ic}, {jc}, {kc}, {lc}, :]", PROG_split[ic, jc, kc, lc, :], file=log_file)
        #     print(f"PROG_mean [{ic}, {jc}, {kc}, {lc}, :]", PROG_mean [ic, jc, kc, lc, :], file=log_file)
        #     print(f"PROGq     [{ic}, {jc}, {kc}, {lc}, :]", PROGq[ic, jc, kc, lc, :], file=log_file) 
        #     if adm.ADM_have_pl:
        #         print(f"PROG_pl [0, {kc}, {lc}, :]", PROG_pl [0, kc, lc, :], file=log_file)   
        #         print(f"PROG_pl [1, {kc}, {lc}, :]", PROG_pl [1, kc, lc, :], file=log_file)
        #         print(f"PROG_pl [2, {kc}, {lc}, :]", PROG_pl [2, kc, lc, :], file=log_file)
        #     print(" ",file=log_file)

        prf.PROF_rapend('__Dynamics', 1)

        return
        #print("dynamics_step")
        #return

