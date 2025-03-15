import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf


class Prgv:
    
    _instance = None
    
    # --- Public Variables ---
    PRG_var = None  # Equivalent to allocatable array PRG_var(:,:,:,:)
    DIAG_var = None  # Equivalent to allocatable array DIAG_var(:,:,:,:)

    restart_input_basename = ""  
    restart_output_basename = ""

    # --- Private Variables ---
    PRG_var_pl = None  # Equivalent to private allocatable array PRG_var_pl(:,:,:,:)
    DIAG_var_pl = None  # Equivalent to private allocatable array DIAG_var_pl(:,:,:,:)

    TRC_vmax_input = 0  # Number of input tracer variables

    layername = ""       # Equivalent to character(len=H_SHORT)
    input_io_mode = "ADVANCED"
    output_io_mode = "ADVANCED"
    allow_missingq = False  # Equivalent to logical variable

    restart_ref_basename = ""
    ref_io_mode = "ADVANCED"
    verification = False  # Equivalent to logical variable


    def __init__(self):
        pass

    def prgvar_setup(self, fname_in, rcnf):

        input_basename    = ''
        output_basename   = 'restart'
        ref_basename      = 'reference'
        restart_layername = ''

        TRC_vmax_input = rcnf.TRC_vmax

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[prgvar]/Category[nhm share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'restartparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** restartparam not found in toml file! Use default.", file=log_file)
                #prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['restartparam']
            input_io_mode     = cnfs['input_io_mode']
            output_io_mode    = cnfs['output_io_mode']
            output_basename   = cnfs['output_basename']
            restart_layername = cnfs['restart_layername']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        self.restart_input_basename  = input_basename
        self.restart_output_basename = output_basename
        self.restart_ref_basename    = ref_basename
        self.layername               = restart_layername


        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print(f"*** io_mode for restart, input : {self.input_io_mode.strip()}", file=log_file)
                
        valid_input_modes = {"POH5", "ADVANCED", "IDEAL", "IDEAL_TRACER"}
        if input_io_mode not in valid_input_modes:
            print("xxx [prgvar] Invalid input_io_mode. STOP.")
            prc.prc_mpistop(std.io_l, std.fname_log)

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(f"*** io_mode for restart, output: {output_io_mode.strip()}", file=log_file)

        valid_output_modes = {"POH5", "ADVANCED"}
        if output_io_mode not in valid_output_modes:
            print("xxx [prgvar] Invalid output_io_mode. STOP")
            prc.prc_mpistop(std.io_l, std.fname_log)

        if self.allow_missingq:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("*** Allow missing tracer in restart file.", file=log_file)
                    print("*** Value will be set to zero for missing tracer.", file=log_file)
            
        self.PRG_var = np.zeros((adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, rcnf.PRG_vmax), dtype=np.float64)
        self.PRG_var_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kall, adm.ADM_lall_pl, rcnf.PRG_vmax), dtype=np.float64)

        self.DIAG_var = np.zeros((adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, rcnf.DIAG_vmax), dtype=np.float64)
        self.DIAG_var_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kall, adm.ADM_lall_pl, rcnf.DIAG_vmax), dtype=np.float64)

        return
    
    def restart_input(self, fname_in, cnst, rcnf, ide, rdtype):

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("\n*** read restart/initial data", file=log_file)

        if self.input_io_mode == "ADVANCED":
            print("ADVANCED not implemented yet")
            prc.prc_mpistop(std.io_l, std.fname_log)
            ## Read diagnostic variables
            #for nq in range(DIAG_vmax0):
            #    FIO_input(rcnf.DIAG_var[:, :, :, nq], basename, rcnf.DIAG_name[nq],
            #              layername, 1, adm.ADM_kall, 1)

            ## Read tracer variables
            #for nq in range(1, TRC_vmax_input + 1):
            #    FIO_input(rcnf.DIAG_var[:, :, :, DIAG_vmax0 + nq - 1], basename, rcnf.TRC_name[nq - 1],
            #              layername, 1, adm.ADM_kall, 1, allow_missingq=allow_missingq)

        elif self.input_io_mode == "POH5":
            print("POH5 not implemented yet")
            prc.prc_mpistop(std.io_l, std.fname_log)
            # Read diagnostic variables
            #for nq in range(1, DIAG_vmax0 + 1):
            #    HIO_input(rcnf.DIAG_var[:, :, :, nq - 1], basename, rcnf.DIAG_name[nq - 1],
            #              layername, 1, adm.ADM_kall, 1)

            ## Read tracer variables
            #for nq in range(1, TRC_vmax_input + 1):
            #    HIO_input(rcnf.DIAG_var[:, :, :, DIAG_vmax0 + nq - 1], basename, rcnf.TRC_name[nq - 1],
            #              layername, 1, adm.ADM_kall, 1, allow_missingq=allow_missingq)

        elif self.input_io_mode == "IDEAL":
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("*** make ideal initials", file=log_file) 
        
            self.DIAGvar = ide.dycore_input(fname_in, cnst, rcnf, rdtype)



        elif self.input_io_mode == "IDEAL_TRACER":
            print("IDEAL_TRACER not implemented yet")
            prc.prc_mpistop(std.io_l, std.fname_log)
            ## Read diagnostic variables
            #for nq in range(1, DIAG_vmax0 + 1):
            #    FIO_input(rcnf.DIAG_var[:, :, :, nq - 1], basename, rcnf.DIAG_name[nq - 1],
            #          layername, 1, adm.ADM_kall, 1)

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("*** make ideal initials for tracer", file=log_file)

        # Call tracer_input for tracer initialization
        ide.tracer_input(rcnf.DIAG_var[:, :, :,rcnf.DIAG_vmax0:rcnf.DIAG_vmax0 + rcnf.TRC_vmax])
