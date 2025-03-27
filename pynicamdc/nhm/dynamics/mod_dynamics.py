import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf


class Dyn:
    
    _instance = None
    
    def __init__(self):
        pass

    def dynamics_setup(self, comm, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, rdtype):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("")
                print("+++ Module[dynamics]/Category[nhm]", file=log_file)     
                print(f"+++ Time integration type: {tim.TIME_integ_type.strip()}", file=log_file)

        # Number of large steps (0–4)
        #num_of_iteration_lstep = 0
        # Number of substeps for each large step (up to 4 stages)
        num_of_iteration_sstep = np.zeros(4, dtype=int)


        if tim.TIME_integ_type == 'RK2':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("+++ 2-stage Runge-Kutta", file=log_file)
            num_of_iteration_lstep = 2
            num_of_iteration_sstep[0] = tim.TIME_sstep_max / 2
            num_of_iteration_sstep[1] = tim.TIME_sstep_max

        elif tim.TIME_integ_type == 'RK3':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("+++ 3-stage Runge-Kutta", file=log_file)
            num_of_iteration_lstep = 3
            num_of_iteration_sstep[0] = tim.TIME_sstep_max / 3
            num_of_iteration_sstep[1] = tim.TIME_sstep_max / 2
            num_of_iteration_sstep[2] = tim.TIME_sstep_max

        elif tim.TIME_integ_type == 'RK4':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("+++ 4-stage Runge-Kutta", file=log_file)
            num_of_iteration_lstep = 4
            num_of_iteration_sstep[0] = tim.TIME_sstep_max / 4
            num_of_iteration_sstep[1] = tim.TIME_sstep_max / 3
            num_of_iteration_sstep[2] = tim.TIME_sstep_max / 2
            num_of_iteration_sstep[3] = tim.TIME_sstep_max

        elif tim.TIME_integ_type == 'TRCADV':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("+++ Offline tracer experiment", file=log_file)
            num_of_iteration_lstep = 0

            if rcnf.TRC_ADV_TYPE == 'DEFAULT':
                print(f"xxx [dynamics_setup] unsupported advection scheme for TRCADV test! STOP. {rcnf.TRC_ADV_TYPE.strip()}")
                prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            print(f"xxx [dynamics_setup] unsupported integration type! STOP. {tim.TIME_integ_type.strip()}")
            prc.prc_mpistop(std.io_l, std.fname_log)


        trcadv_out_dyndiv = False

        if rcnf.TRC_ADV_LOCATION == 'OUT_DYN_DIV_LOOP':
            if rcnf.TRC_ADV_TYPE == 'MIURA2004':
                trcadv_out_dyndiv = True
            else:
                print(f"xxx [dynamics_setup] unsupported TRC_ADV_TYPE for OUT_DYN_DIV_LOOP. STOP. {rcnf.TRC_ADV_TYPE.strip()}")
                prc.prc_mpistop(std.io_l, std.fname_log)

        rweight_dyndiv = np.float64(1.0) / np.float64(rcnf.DYN_DIV_NUM)

    #---< boundary condition module setup >---                                                                         
    #call BNDCND_setup

    #---< basic state module setup >---                                                                                
    #call bsstate_setup

    #---< numerical filter module setup >---                                                                           
    #call numfilter_setup

    #---< vertical implicit module setup >---                                                                          
    #call vi_setup


    # skip
    #---< sub-grid scale dynamics module setup >---                                                                    
    #TENTATIVE!     call sgs_setup                                                                                          

    # skip
    #---< nudging module setup >---                                                                                    
    #call NDG_setup


        return
