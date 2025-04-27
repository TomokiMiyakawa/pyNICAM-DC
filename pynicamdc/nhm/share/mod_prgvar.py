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

    def prgvar_setup(self, fname_in, rcnf, cnst, rdtype):

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
            input_basename    = cnfs['input_basename']
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
        self.input_io_mode           = input_io_mode
        self.output_io_mode          = output_io_mode

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print(f"*** io_mode for restart, input : {self.input_io_mode.strip()}", file=log_file)
                
        valid_input_modes = {"json", "POH5", "ADVANCED", "IDEAL", "IDEAL_TRACER"}
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
            # 
        self.PRG_var = np.full((adm.ADM_shape + (rcnf.PRG_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.PRG_var_pl = np.full((adm.ADM_shape_pl + (rcnf.PRG_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)

        self.DIAG_var = np.full((adm.ADM_shape + (rcnf.DIAG_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.DIAG_var_pl = np.full((adm.ADM_shape_pl + (rcnf.PRG_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)

        return
    
    def restart_input(self, fname_in, comm, gtl, cnst, rcnf, grd, vmtr, cnvv, tdyn, idi, rdtype):

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("\n*** read restart/initial data", file=log_file)

        if self.input_io_mode == "ADVANCED":
            print("ADVANCED input not implemented yet")
            prc.prc_mpistop(std.io_l, std.fname_log)
            ## Read diagnostic variables
            #for nq in range(DIAG_vmax0):
            #    FIO_input(rcnf.DIAG_var[:, :, :, nq], basename, rcnf.DIAG_name[nq],
            #              layername, 1, adm.ADM_kall, 1)

            ## Read tracer variables
            #for nq in range(1, TRC_vmax_input + 1):
            #    FIO_input(rcnf.DIAG_var[:, :, :, DIAG_vmax0 + nq - 1], basename, rcnf.TRC_name[nq - 1],
            #              layername, 1, adm.ADM_kall, 1, allow_missingq=allow_missingq)

        elif self.input_io_mode == "json":
            with open(std.fname_log, 'a') as log_file:
                    print("*** reading json file", file=log_file)

            import json
            fullname = self.restart_input_basename+str(prc.prc_myrank).zfill(8)+".json"
            #print(f"fullname: {fullname}")
            with open(fullname, "r") as json_file:
                loaded_data = json.load(json_file)

            #np.seterr(under='ignore')
            cnt=0
            for varname, var_data in loaded_data["Variables"].items():
                variable_array = np.array(var_data["Data"])
                #print(f"{varname}: {variable_array.shape}")
                for i in range(adm.ADM_gall_1d):
                    for j in range(adm.ADM_gall_1d):
                        #ij = i * adm.ADM_gall_1d + j
                        ij = j * adm.ADM_gall_1d + i
                        self.DIAG_var[i,j,:,:,cnt] = variable_array[ij,:,:].astype(rdtype)
                        #if i==1 and 
                        # if j==17:# and prc.prc_myrank==0:
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("HALLO", cnt, i, self.DIAG_var[i,j-1:j+1,0,2,cnt],file=log_file)
                                
#                            print("HALLO", self.DIAG_var[i,j,0,2,cnt])#variable_array[ij,0,2])
                cnt += 1 
 
            #np.seterr(under='raise')
            #print("DIAG_vmax ", rcnf.DIAG_vmax, cnt)

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
                    print("*** IDEAL initials is slow and untested", file=log_file)
                    print("*** make ideal initials", file=log_file) 
        
            self.DIAGvar = idi.dycore_input(fname_in, cnst, rcnf, grd, idi, rdtype)

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
            idi.tracer_input(self.DIAG_var[:, :, :, rcnf.DIAG_vmax0:rcnf.DIAG_vmax0 + rcnf.TRC_vmax])

        ####compare input data here with the original code!!!!
        ###  and recommendef after COMM_var as well once checked green here.

        # prc.PRC_MPIbarrier()

        # with open(std.fname_log, 'a') as log_file:
        #     print("QQQ", self.DIAG_var[14, 4, 39, 4, rcnf.I_vx], file=log_file)

        comm.COMM_var(self.DIAG_var, self.DIAG_var_pl)

        # with open(std.fname_log, 'a') as log_file:
        #     print("QQQq", self.DIAG_var[14, 4, 39, 4, rcnf.I_vx], file=log_file)
        
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("\n====== Data Range Check: Diagnostic Variables ======", file=log_file)

                for nq in range(rcnf.DIAG_vmax0):
                    #print("nq=", nq)
                    val_max = gtl.GTL_max(self.DIAG_var[:,:,:,:, nq], self.DIAG_var_pl[:,:,:, nq], 
                                        adm.ADM_kall, adm.ADM_kmin, adm.ADM_kmax, cnst, comm, rdtype
                                        )
                    val_min = gtl.GTL_min(self.DIAG_var[:,:,:,:, nq], self.DIAG_var_pl[:,:,:, nq], 
                                        adm.ADM_kall, adm.ADM_kmin, adm.ADM_kmax, cnst, comm, rdtype
                                        )
                    print(f"--- {rcnf.DIAG_name[nq]:16}: max={val_max:24.17E}, min={val_min:24.17E}", file=log_file)

                #print("TRC_vmax", rcnf.TRC_vmax)

                for nq in range(rcnf.TRC_vmax):  # Fortran 1-based index → Python 0-based range
                    val_max = gtl.GTL_max(self.DIAG_var[:,:,:,:, rcnf.DIAG_vmax0 + nq],  
                                            self.DIAG_var_pl[:,:,:, rcnf.DIAG_vmax0 + nq],
                                            adm.ADM_kall, adm.ADM_kmin, adm.ADM_kmax, cnst, comm, rdtype
                                            )
                    # val_min = gtl.GTL_min(self.DIAG_var[:,:,:,:, rcnf.DIAG_vmax0 + nq],  
                    #                         self.DIAG_var_pl[:,:,:, rcnf.DIAG_vmax0 + nq],
                    #                         adm.ADM_kall, adm.ADM_kmin, adm.ADM_kmax, cnst, comm, rdtype
                    #                         )
                    
                    nonzero = val_max > rdtype(0.0)  # Direct boolean conversion
                    val_min = gtl.GTL_min(self.DIAG_var[:,:,:,:, rcnf.DIAG_vmax0 + nq],
                                            self.DIAG_var_pl[:,:,:, rcnf.DIAG_vmax0 + nq],
                                            adm.ADM_kall, adm.ADM_kmin, adm.ADM_kmax, cnst, comm, rdtype, nonzero
                                            )
                    print(f"--- {rcnf.TRC_name[nq]:16}: max={val_max:24.17E}, min={val_min:24.17E}", file=log_file)

        #np.seterr(under='ignore')
        self.PRG_var, self.PRG_var_pl = cnvv.cnvvar_diag2prg(self.DIAG_var, self.DIAG_var_pl, cnst, vmtr, rcnf, tdyn, rdtype)
        #np.seterr(under='raise')

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("\n====== Data Range Check: Prognostic Variables ======", file=log_file)
 

        for nq in range(rcnf.PRG_vmax0):
            val_max = gtl.GTL_max(
                self.PRG_var[:, :, :, :, nq],
                self.PRG_var_pl[:, :, :, nq],
                adm.ADM_kall, adm.ADM_kmin, adm.ADM_kmax,
                cnst, comm, rdtype
            )
            val_min = gtl.GTL_min(
                self.PRG_var[:, :, :, :, nq],
                self.PRG_var_pl[:, :, :, nq],
                adm.ADM_kall, adm.ADM_kmin, adm.ADM_kmax,
                cnst, comm, rdtype,
                #nonzero
            )

            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"--- {rcnf.PRG_name[nq]:<16}: max={val_max:24.17e}, min={val_min:24.17e}", file=log_file)
            
            # if nq ==1 or nq ==2 or nq==3 or nq ==4:
            #     for i in range(adm.ADM_gall_1d):
            #         for j in range(adm.ADM_gall_1d):
            #             for k in range(adm.ADM_kall):
            #                 for l in range(adm.ADM_lall):
            #                     # if self.PRG_var[i, j, k, l, nq] == val_max:
            #                     #     with open(std.fname_log, 'a') as log_file:
            #                     #         print(rcnf.PRG_name[nq],file=log_file)
            #                     #         print(f"MMMAX {rcnf.PRG_name[nq]}:, {i}, {j}, {k}, {l}, {self.PRG_var[i, j, k, l, nq]}", file=log_file)
            #                     if self.PRG_var[i, j, k, l, nq] == val_min:
            #                         with open(std.fname_log, 'a') as log_file:
            #                             print(rcnf.PRG_name[nq],file=log_file)
            #                             print(f"MMMIN {rcnf.PRG_name[nq]}:, {i}, {j}, {k}, {l}, {self.PRG_var[i, j, k, l, nq]}", file=log_file)

        for nq in range(rcnf.TRC_vmax):
            idx = rcnf.PRG_vmax0 + nq
            val_max = gtl.GTL_max(
                self.PRG_var[:, :, :, :, idx],
                self.PRG_var_pl[:, :, :, idx],
                adm.ADM_kall, adm.ADM_kmin, adm.ADM_kmax,
                cnst, comm, rdtype
            )

            nonzero = val_max > rdtype(0.0)

            val_min = gtl.GTL_min(
                self.PRG_var[:, :, :, :, idx],
                self.PRG_var_pl[:, :, :, idx],
                adm.ADM_kall, adm.ADM_kmin, adm.ADM_kmax,
                cnst, comm, rdtype,
                nonzero
            )

            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"--- rhog * {rcnf.TRC_name[nq]:<16}: max={val_max:24.17e}, min={val_min:24.17e}", file=log_file)

        return
    
