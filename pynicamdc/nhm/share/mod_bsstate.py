import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf


class Bsst:
    
    _instance = None
    
    ref_type  = 'NOBASE' 
    ref_fname = 'ref.dat'
    sounding_fname = ''

    def __init__(self):
        pass

    def bsstate_setup(self, fname_in, cnst, rdtype):

        pre_ref = np.zeros(adm.ADM_kall, dtype=rdtype)  # Reference pressure
        tem_ref = np.zeros(adm.ADM_kall, dtype=rdtype)  # Reference temperature
        qv_ref  = np.zeros(adm.ADM_kall, dtype=rdtype)  # Water vapor
        rho_ref = np.zeros(adm.ADM_kall, dtype=rdtype)  # Density
        th_ref  = np.zeros(adm.ADM_kall, dtype=rdtype)  # Potential temperature (dry)

        pre_sfc    = cnst.CONST_Pstd
        tem_sfc    = cnst.CONST_Tstd
        BV_freq    = rdtype(0.0)
        lapse_rate = rdtype(0.0)
        Z_tem      = rdtype(0.0)

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[basic state]/Category[nhm share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'bsstateparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** bsstateparam not found in toml file! Use default.", file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['bsstateparam']
            #self.GRD_grid_type = cnfs['GRD_grid_type']
            ref_type = cnfs['ref_type']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)


        self.rho_bs    = np.zeros((adm.ADM_shape), dtype=rdtype)
        self.rho_bs_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)

        self.pre_bs    = np.zeros((adm.ADM_shape), dtype=rdtype)
        self.pre_bs_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)

        self.tem_bs    = np.zeros((adm.ADM_shape), dtype=rdtype)
        self.tem_bs_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)


        if ref_type == 'NOBASE':
            # Do nothing
            pass

        elif ref_type == 'INPUT':
            print("Sorry, INPUT is not implemented yet.")
            prc.prc_mpistop(std.io_l, std.fname_log)
            # bsstate_input_ref(
                # ref_fname,   # [IN]
                # pre_ref,     # [OUT]
                # tem_ref,     # [OUT]
                # qv_ref       # [OUT]
            # )

        else:
            print("Sorry, not ready yet.")
            prc.prc_mpistop(std.io_l, std.fname_log)
            # bsstate_generate(
            #     sounding_fname,  # [IN]
            #     pre_ref,         # [OUT]
            #     tem_ref,         # [OUT]
            #     qv_ref           # [OUT]
            # )

            # bsstate_output_ref(
            #     ref_fname,   # [IN]
            #     pre_ref,     # [IN]
            #     tem_ref,     # [IN]
            #     qv_ref       # [IN]
            # )


        if ref_type != 'NOBASE':
        # Set 3-D basic state
            print("Sorry, not ready yet.")
            prc.prc_mpistop(std.io_l, std.fname_log)
            # set_basicstate(pre_ref, tem_ref, qv_ref)

            # if IO_L:
            #     print("-------------------------------------------------------")
            #     print("Level   Density  Pressure     Temp. Pot. Tem.        qv")

            # for k in range(ADM_kall, 0, -1):  # Fortran 1-based descending loop
            #     k_idx = k - 1  # Adjust to 0-based index for Python

            #     th_ref[k_idx] = tem_ref[k_idx] * (PRE00 / pre_ref[k_idx]) ** (Rdry / CPdry)
            #     rho_ref[k_idx] = pre_ref[k_idx] / tem_ref[k_idx] / (
            #         (rdtype(1.0) - qv_ref[k_idx]) * Rdry + qv_ref[k_idx] * Rvap
            #     )

            #     if k == ADM_kmax and IO_L:
            #         print("-------------------------------------------------------")

            #     if IO_L:
            #         print(f"{k:4d}{rho_ref[k_idx]:12.4f}{pre_ref[k_idx]:10.2f}"
            #             f"{tem_ref[k_idx]:10.2f}{th_ref[k_idx]:10.2f}{qv_ref[k_idx

        return
    