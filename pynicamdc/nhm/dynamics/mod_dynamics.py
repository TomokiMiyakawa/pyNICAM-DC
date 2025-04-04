import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf

class Dyn:
    
    _instance = None
    
    def __init__(self, rcnf, rdtype):

        # work array for the dynamics
        self._numerator_w = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kmax - adm.ADM_kmin, adm.ADM_lall), dtype=rdtype)
        self._denominator_w = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kmax - adm.ADM_kmin, adm.ADM_lall), dtype=rdtype)
        self._numerator_pl_w = np.empty((adm.ADM_gall_pl, adm.ADM_kmax - adm.ADM_kmin, adm.ADM_lall), dtype=rdtype)
        self._denominator_pl_w = np.empty((adm.ADM_gall_pl, adm.ADM_kmax - adm.ADM_kmin, adm.ADM_lall), dtype=rdtype)

        # Prognostic and tracer variables
        self.PROG        = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall, 6), dtype=rdtype)
        self.PROG_pl     = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, 6), dtype=rdtype)
        self.PROGq       = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall,    rcnf.TRC_vmax), dtype=rdtype)
        self.PROGq_pl    = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, rcnf.TRC_vmax), dtype=rdtype)

        # Tendency of prognostic and tracer variables
        self.g_TEND      = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall, 6), dtype=rdtype)
        self.g_TEND_pl   = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, 6), dtype=rdtype)
        self.g_TENDq     = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall,    rcnf.TRC_vmax), dtype=rdtype)
        self.g_TENDq_pl  = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, rcnf.TRC_vmax), dtype=rdtype)

        # Forcing tendency
        self.f_TEND      = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall, 6), dtype=rdtype)
        self.f_TEND_pl   = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, 6), dtype=rdtype)
        self.f_TENDq     = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall,    rcnf.TRC_vmax), dtype=rdtype)
        self.f_TENDq_pl  = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, rcnf.TRC_vmax), dtype=rdtype)

        # Saved prognostic/tracer variables
        self.PROG00      = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall, 6), dtype=rdtype)
        self.PROG00_pl   = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, 6), dtype=rdtype)
        self.PROGq00     = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall,    rcnf.TRC_vmax), dtype=rdtype)
        self.PROGq00_pl  = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, rcnf.TRC_vmax), dtype=rdtype)
        self.PROG0       = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall, 6), dtype=rdtype)
        self.PROG0_pl    = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, 6), dtype=rdtype)

        # Split prognostic variables
        self.PROG_split     = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall, 6), dtype=rdtype)
        self.PROG_split_pl  = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, 6), dtype=rdtype)

        # Mean prognostic variables
        self.PROG_mean      = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall, 5), dtype=rdtype)
        self.PROG_mean_pl   = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, 5), dtype=rdtype)

        # For tracer advection (large step)
        self.f_TENDrho_mean     = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall), dtype=rdtype)
        self.f_TENDrho_mean_pl  = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)
        self.f_TENDq_mean       = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall,    rcnf.TRC_vmax), dtype=rdtype)
        self.f_TENDq_mean_pl    = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, rcnf.TRC_vmax), dtype=rdtype)
        self.PROG_mean_mean     = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall, 5), dtype=rdtype)
        self.PROG_mean_mean_pl  = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, 5), dtype=rdtype)

        # Diagnostic and tracer variables
        self.DIAG     = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall, 6), dtype=rdtype)
        self.DIAG_pl  = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, 6), dtype=rdtype)
        self.q        = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,    rcnf.TRC_vmax), dtype=rdtype)
        self.q_pl     = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, rcnf.TRC_vmax), dtype=rdtype)

        # Density
        self.rho      = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall), dtype=rdtype)
        self.rho_pl   = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)

        # Internal energy (physical)
        self.ein      = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall), dtype=rdtype)
        self.ein_pl   = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)

        # Enthalpy (physical)
        self.eth      = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall), dtype=rdtype)
        self.eth_pl   = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)

        # Potential temperature (physical)
        self.th       = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall), dtype=rdtype)
        self.th_pl    = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)

        # Density deviation from base state
        self.rhogd    = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall), dtype=rdtype)
        self.rhogd_pl = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)

        # Pressure deviation from base state
        self.pregd    = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall), dtype=rdtype)
        self.pregd_pl = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)

        # Temporary variables
        self.qd       = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall), dtype=rdtype)
        self.qd_pl    = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)
        self.cv       = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall), dtype=rdtype)
        self.cv_pl    = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)

        return
    

    def dynamics_setup(self, fname_in, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, bndc, bsst, numf, vi, rdtype):

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
            prc.prc_mpistop(std.io_l, std.fname_log)


        self.trcadv_out_dyndiv = False

        if rcnf.TRC_ADV_LOCATION == 'OUT_DYN_DIV_LOOP':
            if rcnf.TRC_ADV_TYPE == 'MIURA2004':
                self.trcadv_out_dyndiv = True
            else:
                print(f"xxx [dynamics_setup] unsupported TRC_ADV_TYPE for OUT_DYN_DIV_LOOP. STOP. {rcnf.TRC_ADV_TYPE.strip()}")
                prc.prc_mpistop(std.io_l, std.fname_log)

        self.rweight_dyndiv = rdtype(1.0) / rdtype(rcnf.DYN_DIV_NUM)

        #---< boundary condition module setup >---                                                                         
        bndc.BNDCND_setup(fname_in, rdtype)

        #---< basic state module setup >---                                                                                
        bsst.bsstate_setup(fname_in, cnst, rdtype)

        #---< numerical filter module setup >---                                                                           
        numf.numfilter_setup(fname_in, rcnf, cnst, comm, gtl, grd, gmtr, oprt, vmtr, tim, prgv, tdyn, frc, bndc, bsst, rdtype)


        #---< vertical implicit module setup >---                                                                          
        vi.vi_setup(rdtype) #(fname_in, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, bndc, bsst, numf, rdtype)

        # skip
        #---< sub-grid scale dynamics module setup >---                                                                    
        #TENTATIVE!     call sgs_setup                                                                                          

        # skip
        #---< nudging module setup >---                                                                                    
        #call NDG_setup

        return
                          
    def dynamics_step(self, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, bndc, bsst, numf, vi, src, rdtype):

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
        kall = adm.ADM_kdall
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

        dyn_step_dt = rdtype(tim.TIME_dtl)
        large_step_dt = rdtype(tim.TIME_dtl) * self.rweight_dyndiv

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

                f_TEND[:, :, :, :, :] = 0.0
                f_TEND_pl[:, :, :, :] = 0.0

                # skip for now (not needed for JW test)
                #call src_tracer_advection

                prf.PROF_rapend('__Tracer_Advection', 1)
                
                #skip for now (not needed for JW test)
                #call forcing_update( PROG(:,:,:,:), PROG_pl(:,:,:,:) ) ! [INOUT]

            # endif


            #---------------------------------------------------------------------------
            #
            #> Start large time step integration
            #
            #---------------------------------------------------------------------------
            for nl in range(self.num_of_iteration_lstep):

                prf.PROF_rapstart('___Pre_Post',1)

                #---< Generate diagnostic values and set the boudary conditions
                
                # Extract variables
                RHOG    = PROG[:, :, :, :, I_RHOG]
                RHOGVX  = PROG[:, :, :, :, I_RHOGVX]
                RHOGVY  = PROG[:, :, :, :, I_RHOGVY]
                RHOGVZ  = PROG[:, :, :, :, I_RHOGVZ]
                RHOGE   = PROG[:, :, :, :, I_RHOGE]

                rho[:, :, :, :] = RHOG / vmtr.VMTR_GSGAM2
                DIAG[:, :, :, :, I_vx] = RHOGVX / RHOG
                DIAG[:, :, :, :, I_vy] = RHOGVY / RHOG
                DIAG[:, :, :, :, I_vz] = RHOGVZ / RHOG
                ein[:, :, :, :] = RHOGE / RHOG

                q[:, :, :, :, :] = PROGq / PROG[:, :, :, :, np.newaxis, I_RHOG]

                # with open (std.fname_log, 'a') as log_file:
                #     print("ZEROsearch",file=log_file) 
                #     print(RHOG[16, 0, 41, 0], RHOG[16, 0, 40, 0],file=log_file)
                #     print(RHOG[17, 0, 41, 0], RHOG[17, 0, 40, 0],file=log_file)

                # Preallocated arrays: cv, qd, q, ein, rho, DIAG all have shape (i, j, k, l [, nq])
                # q has shape: (i, j, k, l, nq)

                # Reset cv and qd
                cv.fill(0.0)
                qd.fill(1.0)

                # Slice tracers from nmin to nmax
                q_slice = q[:, :, :, :, nmin:nmax+1]                # shape: (i, j, k, l, nq_range)
                CVW_slice = CVW[nmin:nmax+1]                        # shape: (nq_range,)

                # Accumulate cv and qd over tracer range
                cv += np.sum(q_slice * CVW_slice[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :], axis=4)
                qd -= np.sum(q_slice, axis=4)

                # Add dry-air contribution to cv
                cv += qd * CVdry

                # Compute temperature
                DIAG[:, :, :, :, I_tem] = ein / cv

                # Compute pressure
                DIAG[:, :, :, :, I_pre] = rho * DIAG[:, :, :, :, I_tem] * (qd * Rdry + q[:, :, :, :, iqv] * Rvap)


                numerator[:, :, :, :] = PROG[:, :, kmin+1:kmax+1, :, I_RHOGW]
                rhog_k   = PROG[:, :, kmin+1:kmax+1, :, I_RHOG]
                rhog_km1 = PROG[:, :, kmin:kmax,     :, I_RHOG]
                fact1 = vmtr.VMTR_C2Wfact[:, :, kmin+1:kmax+1, 0, :]
                fact2 = vmtr.VMTR_C2Wfact[:, :, kmin+1:kmax+1, 1, :]
                denominator[:, :, :, :] = fact1 * rhog_k + fact2 * rhog_km1
                DIAG[:, :, kmin+1:kmax+1, :, I_w] = numerator / denominator

                # Task1
                #print("Task1a done")
                bndc.BNDCND_all(
                    adm.ADM_gall_1d, 
                    adm.ADM_gall_1d, 
                    adm.ADM_kdall, 
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

                #call BNDCND_all

                # Task2
                #print("Task2a done but not tested yet")
                th = tdyn.THRMDYN_th(
                        adm.ADM_gall_1d, 
                        adm.ADM_gall_1d, 
                        adm.ADM_kdall, 
                        adm.ADM_lall, 
                        DIAG[:, :, :, :, I_tem], 
                        DIAG[:, :, :, :, I_pre],
                        cnst,
                )
                
                # Task3
                #print("Task3a done but not tested yet")
                eth = tdyn.THRMDYN_eth(
                        adm.ADM_gall_1d, 
                        adm.ADM_gall_1d, 
                        adm.ADM_kdall, 
                        adm.ADM_lall, 
                        ein,
                        DIAG[:, :, :, :, I_pre],
                        rho,
                        cnst,
                )


                # perturbations ( pre, rho with metrics )
                pregd[:, :, :, :] = (DIAG[:, :, :, :, I_pre] - pre_bs) * vmtr.VMTR_GSGAM2
                rhogd[:, :, :, :] = (rho                  - rho_bs) * vmtr.VMTR_GSGAM2


                if adm.ADM_have_pl:

                    rho_pl = PROG_pl[:, :, :, I_RHOG]   / vmtr.VMTR_GSGAM2_pl
                    DIAG_pl[:, :, :, I_vx] = PROG_pl[:, :, :, I_RHOGVX] / PROG_pl[:, :, :, I_RHOG]
                    DIAG_pl[:, :, :, I_vy] = PROG_pl[:, :, :, I_RHOGVY] / PROG_pl[:, :, :, I_RHOG]
                    DIAG_pl[:, :, :, I_vz] = PROG_pl[:, :, :, I_RHOGVZ] / PROG_pl[:, :, :, I_RHOG]
                    ein_pl[:, :, :] = PROG_pl[:, :, :, I_RHOGE]  / PROG_pl[:, :, :, I_RHOG]

                    # Tracer mass mixing ratios
                    q_pl[:, :, :, :] = PROGq_pl / PROG_pl[:, :, :, np.newaxis, I_RHOG]

                    # Specific heat capacity and dry air fraction
                    cv_pl.fill(0.0)
                    qd_pl.fill(1.0)

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
                    fact1_pl       = vmtr.VMTR_C2Wfact_pl[:, kmin+1:kmax+1, 0, :]
                    fact2_pl       = vmtr.VMTR_C2Wfact_pl[:, kmin+1:kmax+1, 1, :]
                    denominator_pl = fact1_pl * rhog_k_pl + fact2_pl * rhog_km1_pl

                    DIAG_pl[:, kmin+1:kmax+1, :, I_w] = numerator_pl / denominator_pl

                    # Task1b
                    #print("Task1b done")
                    bndc.BNDCND_all(
                        adm.ADM_gall_pl, 
                        1, 
                        adm.ADM_kdall, 
                        adm.ADM_lall_pl,
                        rho_pl [:, np.newaxis, :, :],                # [INOUT] view with additional dimension may stay after the BNDCND_all call. Squeeze it back later explicitly.
                        DIAG_pl[:, np.newaxis, :, :, I_vx],          # [INOUT]
                        DIAG_pl[:, np.newaxis, :, :, I_vy],          # [INOUT]
                        DIAG_pl[:, np.newaxis, :, :, I_vz],          # [INOUT]
                        DIAG_pl[:, np.newaxis, :, :, I_w],           # [INOUT]
                        ein_pl [:, np.newaxis, :, :],                # [INOUT]
                        DIAG_pl[:, np.newaxis, :, :, I_tem],         # [INOUT]%
                        DIAG_pl[:, np.newaxis, :, :, I_pre],         # [INOUT]
                        PROG_pl[:, np.newaxis, :, :, I_RHOG],        # [INOUT]
                        PROG_pl[:, np.newaxis, :, :, I_RHOGVX],      # [INOUT]
                        PROG_pl[:, np.newaxis, :, :, I_RHOGVY],      # [INOUT]
                        PROG_pl[:, np.newaxis, :, :, I_RHOGVZ],      # [INOUT]
                        PROG_pl[:, np.newaxis, :, :, I_RHOGW],       # [INOUT]
                        PROG_pl[:, np.newaxis, :, :, I_RHOGE],       # [INOUT]
                        vmtr.VMTR_GSGAM2_pl   [:, np.newaxis, :, :],    # [IN] view with additional dimension is temporaly, i.e., shape does not change after BNDCND_all call
                        vmtr.VMTR_PHI_pl      [:, np.newaxis, :, :],    # [IN]
                        vmtr.VMTR_C2Wfact_pl  [:, np.newaxis, :, :, :], # [IN]
                        vmtr.VMTR_C2WfactGz_pl[:, np.newaxis, :, :, :], # [IN]
                        cnst,
                        rdtype,
                    )

                    # Assign modified slices back to the original arrays (not needed for read-only views)
                    # Note: This triggers a copy operation. I think the effect is minimal because this is only for the poles.
                    #       However, it may be better to have a size 1 dummy dimension for poles throughout the entire code.
                    #       Then the expand/squeeze can be avoided, keeping the code cleaner. Consider this in the future.
                    #           Or, this is completely unnecessary. Seems to be working without it.
            
                    #print("DIAG_pl shape before squeeze:", DIAG_pl.shape)
                    #print("DIAG_pl I_vx slice shape before squeeze:", DIAG_pl[:, :, :, I_vx].shape)
                    # DIAG_pl[:, :, :, I_vx] = DIAG_pl[:, :, :, :, I_vx].squeeze(axis=1)
                    # ein_pl = ein_pl.squeeze(axis=1)

                    # Task2
                    #print("Task2b done but not tested yet")
                    th = tdyn.THRMDYN_th(
                        adm.ADM_gall_pl, 
                        1, 
                        adm.ADM_kdall, 
                        adm.ADM_lall_pl, 
                        DIAG_pl[:, np.newaxis, :, :, I_tem], 
                        DIAG_pl[:, np.newaxis, :, :, I_pre],
                        cnst,
                    )
                    
                    # Task3
                    #print("Task3b done but not tested yet")
                    eth = tdyn.THRMDYN_eth(
                        adm.ADM_gall_pl, 
                        1, 
                        adm.ADM_kdall, 
                        adm.ADM_lall_pl, 
                        ein_pl [:, np.newaxis, :, :],  
                        DIAG_pl[:, np.newaxis, :, :, I_pre],
                        rho_pl [:, np.newaxis, :, :], 
                        cnst,
                    )

                    # perturbations ( pre, rho with metrics )
                    pregd_pl[:, :, :] = (DIAG_pl[:, :, :, I_pre] - pre_bs_pl) * vmtr.VMTR_GSGAM2_pl
                    rhogd_pl[:, :, :] = (rho_pl - rho_bs_pl) * vmtr.VMTR_GSGAM2_pl

                else:

                    PROG_pl [:, :, :, :] = 0.0
                    DIAG_pl [:, :, :, :] = 0.0
                    rho_pl  [:, :, :]    = 0.0
                    q_pl    [:, :, :, :] = 0.0
                    th_pl   [:, :, :]    = 0.0
                    eth_pl  [:, :, :]    = 0.0
                    pregd_pl[:, :, :]    = 0.0
                    rhogd_pl[:, :, :]    = 0.0

                prf.PROF_rapend('___Pre_Post',1)
                #------------------------------------------------------------------------
                #> LARGE step
                #------------------------------------------------------------------------
                prf.PROF_rapstart('__Large_step', 1)

                #--- calculation of advection tendency including Coriolis force
                # Task 4
                #print("Task4 done but not tested yet")
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
                        g_TEND[:,:,:,:,I_RHOGVX], g_TEND_pl[:,:,:,I_RHOGVX], # [OUT]
                        g_TEND[:,:,:,:,I_RHOGVY], g_TEND_pl[:,:,:,I_RHOGVY], # [OUT]
                        g_TEND[:,:,:,:,I_RHOGVZ], g_TEND_pl[:,:,:,I_RHOGVZ], # [OUT]
                        g_TEND[:,:,:,:,I_RHOGW],  g_TEND_pl[:,:,:,I_RHOGW],  # [OUT]
                        rcnf, cnst, grd, oprt, vmtr, rdtype,
                )


                g_TEND[:, :, :, :, I_RHOG]  = 0.0
                g_TEND[:, :, :, :, I_RHOGE] = 0.0

                # Zero out specific components of g_TEND_pl
                g_TEND_pl[:, :, :, I_RHOG]  = 0.0
                g_TEND_pl[:, :, :, I_RHOGE] = 0.0

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
                    numf.numfilter_hdiffusion(
                        PROG   [:,:,:,:,I_RHOG], PROG_pl   [:,:,:,I_RHOG], # [IN]
                        rho    [:,:,:,:],        rho_pl    [:,:,:],        # [IN]
                        DIAG   [:,:,:,:,I_vx],   DIAG_pl   [:,:,:,I_vx],   # [IN]
                        DIAG   [:,:,:,:,I_vy],   DIAG_pl   [:,:,:,I_vy],   # [IN]
                        DIAG   [:,:,:,:,I_vz],   DIAG_pl   [:,:,:,I_vz],   # [IN]
                        DIAG   [:,:,:,:,I_w],    DIAG_pl   [:,:,:,I_w],    # [IN]
                        DIAG   [:,:,:,:,I_tem],  DIAG_pl   [:,:,:,I_tem],  # [IN]
                        q      [:,:,:,:,:],      q_pl      [:,:,:,:],      # [IN]
                        f_TEND [:,:,:,:,:],      f_TEND_pl [:,:,:,:],      # [OUT]
                        f_TENDq[:,:,:,:,:],      f_TENDq_pl[:,:,:,:],      # [OUT]
                        cnst, comm, grd, oprt, vmtr, tim, rcnf, bsst, rdtype,
                    )


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

                g_TEND_pl += f_TEND_pl


                prf.PROF_rapend('__Large_step',1)
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
                    PROG_split[:, :, :, :, 0:6] = 0.0
                    PROG_split_pl[:, :, :, :] = 0.0
                #endif
            
                #------ Core routine for small step
                #------    1. By this subroutine, prognostic variables ( rho,.., rhoge ) are calculated through
                #------    2. grho, grhogvx, ..., and  grhoge has the large step
                #------       tendencies initially, however, they are re-used in this subroutine.
                #------

                if tim.TIME_split:   # check closely !!!
                    small_step_ite = self.num_of_iteration_sstep[nl]
                    small_step_dt = tim.TIME_dts * self.rweight_dyndiv
                else:
                    small_step_ite = 1
                    small_step_dt = large_step_dt / (self.num_of_iteration_lstep - nl)
                #endif

                # Task 6
#               print("Task6")
                # vi.vi_small_step(
                #            PROG      [:,:,:,:,:],    PROG_pl      [:,:,:,:],    #   [INOUT] prognostic variables
                #            DIAG      [:,:,:,:,I_vx], DIAG_pl      [:,:,:,I_vx], #   [IN] diagnostic value
                #            DIAG      [:,:,:,:,I_vy], DIAG_pl      [:,:,:,I_vy], #   [IN]
                #            DIAG      [:,:,:,:,I_vz], DIAG_pl      [:,:,:,I_vz], #   [IN]
                #            eth       [:,:,:,:],      eth_pl       [:,:,:],      #   [IN]
                #            rhogd     [:,:,:,:],      rhogd_pl     [:,:,:],      #   [IN]
                #            pregd     [:,:,:,:],      pregd_pl     [:,:,:],      #   [IN]
                #            g_TEND    [:,:,:,:,:],    g_TEND_pl    [:,:,:,:],    #   [IN] large step TEND
                #            PROG_split[:,:,:,:,:],    PROG_split_pl[:,:,:,:],    #   [INOUT] split value
                #            PROG_mean [:,:,:,:,:],    PROG_mean_pl[:,:,:,:],     #   [OUT] mean value
                #            small_step_ite,                                      #   [IN]
                #            small_step_dt,                                       #   [IN]
                #            cnst, comm, grd, oprt, vmtr, tim, rcnf, bndc, numf, src, rdtype,
                # ) 
                
                prf.PROF_rapend('___Small_step',1)
                #------------------------------------------------------------------------
                #>  Tracer advection (in the large step)
                #------------------------------------------------------------------------
                prf.PROF_rapstart('___Tracer_Advection',1)

                do_tke_correction = False

                if not self.trcadv_out_dyndiv:  # calc here or not

                    if rcnf.TRC_ADV_TYPE == "MIURA2004":

                        if nl == self.num_of_iteration_lstep:

                            # Task skip for now
                            #call src_tracer_advection
                            pass 

                
                            PROGq[:, :, :, :, :] += large_step_dt * f_TENDq

                            if adm.ADM_have_pl:
                                PROGq_pl[:, :, :, :] += large_step_dt * f_TENDq_pl

                            # [comment] H.Tomita: I don't recommend adding the hyperviscosity term because of numerical instability in this case.
                            if itke >= 0:
                                do_tke_correction = True

                        #endif

                    elif rcnf.TRC_ADV_TYPE == 'DEFAULT':

                        for nq in range(rcnf.TRC_vmax):

                            # Task skip for now, not used for ICOMEX_JW
                            #call src_advection_convergence
                            pass

                        #end tracer LOOP

                        step_coeff = self.num_of_iteration_sstep[nl] * small_step_dt

                        # Update PROGq for all interior points
                        PROGq += step_coeff * (g_TENDq + f_TENDq)

                        PROGq[:, :, kmin-1, :, :] = 0.0
                        PROGq[:, :, kmax+1, :, :] = 0.0

                        if adm.ADM_have_pl:
                            PROGq_pl[:, :, :, :] = PROGq00_pl + step_coeff * (g_TENDq_pl + f_TENDq_pl)
                            PROGq_pl[:, kmin-1, :, :] = 0.0
                            PROGq_pl[:, kmax+1, :, :] = 0.0

                        # Set TKE correction flag if needed
                        if itke >= 0:
                            do_tke_correction = True

                    #endif

                    # TKE fixer
                    if do_tke_correction:

                        # Compute correction term (clip negative TKE values to zero)
                        TKEG_corr = np.maximum(-PROGq[:, :, :, :, itke], 0.0)

                        # Apply correction to RHOGE and TKE
                        PROG[:, :, :, :, I_RHOGE] -= TKEG_corr
                        PROGq[:, :, :, :, itke]   += TKEG_corr

                        # Polar region
                        if adm.ADM_have_pl:
                            TKEG_corr_pl = np.maximum(-PROGq_pl[:, :, :, itke], 0.0)

                            PROG_pl[:, :, :, I_RHOGE] -= TKEG_corr_pl
                            PROGq_pl[:, :, :, itke]  += TKEG_corr_pl
                        #endif
                    #endif

                else:

                    #--- calculation of mean ( mean mass flux and tendency )
                    if nl == self.num_of_iteration_lstep:

                        if ndyn == 1:

                            PROG_mean_mean[:, :, :, :, 0:5] = self.rweight_dyndiv * PROG_mean[:, :, :, :, 0:5]
                            f_TENDrho_mean[:, :, :, :] = self.rweight_dyndiv * f_TEND[:, :, :, :, I_RHOG]
                            f_TENDq_mean[:, :, :, :, :] = self.rweight_dyndiv * f_TENDq


                            PROG_mean_mean_pl[:, :, :, :] = self.rweight_dyndiv * PROG_mean_pl
                            f_TENDrho_mean_pl[:, :, :]    = self.rweight_dyndiv * f_TEND_pl[:, :, :, I_RHOG]
                            f_TENDq_mean_pl[:, :, :, :]   = self.rweight_dyndiv * f_TENDq_pl

                        else:

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
                if nl != self.num_of_iteration_lstep:
                    comm.COMM_data_transfer( PROG, PROG_pl )
                #endif

                prf.PROF_rapend  ('___Pre_Post',1)

            #enddo --- large step

            #---------------------------------------------------------------------------
            #>  Tracer advection (out of the large step)
            #---------------------------------------------------------------------------

            if self.trcadv_out_dyndiv and ndyn == rcnf.DYN_DIV_NUM:
                
                prf.PROF_rapstart('___Tracer_Advection',1)

                # Task 1
                print("Task1")
                # call src_tracer_advection

                PROGq[:, :, :, :, :] += dyn_step_dt * f_TENDq_mean  # update rhogq by viscosity

                if adm.ADM_have_pl:
                    PROGq_pl[:, :, :, :] += dyn_step_dt * f_TENDq_mean_pl
                #endif

                TKEG_corr = np.maximum(-PROGq[:, :, :, :, itke], 0.0)
                PROG[:, :, :, :, I_RHOGE] -= TKEG_corr
                PROGq[:, :, :, :, itke]   += TKEG_corr

                if adm.ADM_have_pl:
                    TKEG_corr_pl = np.maximum(-PROGq_pl[:, :, :, itke], 0.0)
                    PROG_pl[:, :, :, I_RHOGE] -= TKEG_corr_pl
                    PROGq_pl[:, :, :, itke]  += TKEG_corr_pl
                #endif

                prf.PROF_rapend('___Tracer_Advection',1)

            #endif

        #enddo --- divided step for dynamics

        prf.PROF_rapstart('___Pre_Post',1)

        prgv.PRG_var[:, :, :, :, 0:6] = PROG[:, :, :, :, :] 
        prgv.PRG_var_pl[:, :, :, 0:6] = PROG_pl[:, :, :, :]  
        prgv.PRG_var[:, :, :, :, 6:]  = PROGq[:, :, :, :, :]  
        prgv.PRG_var_pl[:, :, :, 6:]  = PROGq_pl[:, :, :, :] 

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

        prf.PROF_rapend('__Dynamics', 1)

        return
        #print("dynamics_step")
        #return

