import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf


class Dyn:
    
    _instance = None
    
    def __init__(self, rcnf, rdtype):

        # work array for the dynamics
        self._numerator_w = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kmax - adm.ADM_kmin, adm.ADM_lall), dtype=rdtype)
        self._denominator_w = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kmax - adm.ADM_kmin, adm.ADM_lall), dtype=rdtype)

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
        num_of_iteration_sstep = np.zeros(4, dtype=int)

        if tim.TIME_integ_type == 'RK2':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("+++ 2-stage Runge-Kutta", file=log_file)
            self.num_of_iteration_lstep = 2
            num_of_iteration_sstep[0] = tim.TIME_sstep_max / 2
            num_of_iteration_sstep[1] = tim.TIME_sstep_max

        elif tim.TIME_integ_type == 'RK3':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("+++ 3-stage Runge-Kutta", file=log_file)
            self.num_of_iteration_lstep = 3
            num_of_iteration_sstep[0] = tim.TIME_sstep_max / 3
            num_of_iteration_sstep[1] = tim.TIME_sstep_max / 2
            num_of_iteration_sstep[2] = tim.TIME_sstep_max

        elif tim.TIME_integ_type == 'RK4':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("+++ 4-stage Runge-Kutta", file=log_file)
            self.num_of_iteration_lstep = 4
            num_of_iteration_sstep[0] = tim.TIME_sstep_max / 4
            num_of_iteration_sstep[1] = tim.TIME_sstep_max / 3
            num_of_iteration_sstep[2] = tim.TIME_sstep_max / 2
            num_of_iteration_sstep[3] = tim.TIME_sstep_max

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
                          
    def dynamics_step(self, prf, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, bndc, bsst, numf, vi, rdtype):

        # Make views of arrays

        #---< work array for the dynamics >---
        numerator = self._numerator_w   
        denominator = self._denominator_w

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

            #--- save the value before RK loop
                PROG0 = PROG.copy()
                PROG0_pl = PROG_pl.copy()


        if tim.TIME_integ_type == 'TRCADV':      # TRC-ADV Test Bifurcation

            prf.PROF_rapstart('__Tracer_Advection', 1)

            f_TEND[:, :, :, :, :] = 0.0
            f_TEND_pl[:, :, :, :] = 0.0

            # Task1
            #call src_tracer_advection

            prf.PROF_rapend('__Tracer_Advection', 1)
            
            # Task2
            #call forcing_update( PROG(:,:,:,:), PROG_pl(:,:,:,:) ) ! [INOUT]

        # endif


        #---------------------------------------------------------------------------
        #
        #> Start large time step integration
        #
        #---------------------------------------------------------------------------
        for nl in range(self.num_of_iteration_lstep):

            prf.PROF_rapstart('___Pre_Post',1)

            # Extract variables
            RHOG    = PROG[:, :, :, :, I_RHOG]
            RHOGVX  = PROG[:, :, :, :, I_RHOGVX]
            RHOGVY  = PROG[:, :, :, :, I_RHOGVY]
            RHOGVZ  = PROG[:, :, :, :, I_RHOGVZ]
            RHOGE   = PROG[:, :, :, :, I_RHOGE]

            # Compute rho
            rho[:, :, :, :] = RHOG / vmtr.VMTR_GSGAM2

            # Compute velocity diagnostics
            DIAG[:, :, :, :, I_vx] = RHOGVX / RHOG
            DIAG[:, :, :, :, I_vy] = RHOGVY / RHOG
            DIAG[:, :, :, :, I_vz] = RHOGVZ / RHOG

            # Compute internal energy
            ein[:, :, :, :] = RHOGE / RHOG

            q[:, :, :, :, :] = PROGq / PROG[:, :, :, :, np.newaxis, I_RHOG]


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

            # Task3
            #call BNDCND_all

            # Task4
            #call THRMDYN_th 

            # Task5
            #call THRMDYN_eth

            pregd[:, :, :, :] = (DIAG[:, :, :, :, I_pre] - pre_bs) * vmtr.VMTR_GSGAM2
            rhogd[:, :, :, :] = (rho                  - rho_bs) * vmtr.VMTR_GSGAM2



            prf.PROF_rapend('___Pre_Post',1)


        prf.PROF_rapend('__Dynamics', 1)

        return
        #print("dynamics_step")
        #return

