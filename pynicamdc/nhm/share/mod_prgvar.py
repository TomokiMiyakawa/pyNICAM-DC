import toml
import numpy as np
#from mpi4py import MPI
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.nhm.share import mod_fio as fio
#from mod_prof import prf


class Prgv:
    
    
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
            self.allow_missingq = cnfs.get('allow_missingq', self.allow_missingq)
            TRC_vmax_input      = cnfs.get('TRC_vmax_input', TRC_vmax_input)

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
        self.TRC_vmax_input          = TRC_vmax_input

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print(f"*** io_mode for restart, input : {self.input_io_mode.strip()}", file=log_file)
                
        valid_input_modes = {"json", "npz", "POH5", "ADVANCED", "IDEAL", "IDEAL_TRACER"}
        if input_io_mode not in valid_input_modes:
            print("xxx [prgvar] Invalid input_io_mode. STOP.")
            prc.prc_mpistop(std.io_l, std.fname_log)

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(f"*** io_mode for restart, output: {output_io_mode.strip()}", file=log_file)

        valid_output_modes = {"POH5", "ADVANCED", "npz"}
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
    
    # nicamdc restart_output metadata (mod_prgvar.f90): DIAG_vmax0 dynamics fields.
    _DLABEL = ['Pressure', 'Temperature', 'H-Velocity(XDIR)',
               'H-Velocity(YDIR)', 'H-Velocity(ZDIR)', 'V-Velocity']
    _DUNIT  = ['Pa', 'K', 'm/s', 'm/s', 'm/s', 'm/s']

    def _advanced_pack(self, slot):
        # inverse of _advanced_unpack: DIAG_var[i,j,k,l,slot] -> (ij,k,l).
        gall = adm.ADM_gall_1d * adm.ADM_gall_1d
        arr = self.DIAG_var[:, :, :, :, slot].transpose(1, 0, 2, 3)   # (j,i,k,l)
        return arr.reshape(gall, adm.ADM_kall, adm.ADM_lall)          # (ij,k,l)

    def restart_output(self, basename, rcnf, rdtype, ctime=0):
        # Write the current DIAG_var as a restart file, the inverse of restart_input.
        # ADVANCED -> native NICAM fio (byte-compatible with nicamdc); npz -> a numpy
        # archive keyed by variable name (what the npz input path reads). basename
        # carries the trailing '.pe'. (nicamdc converts PRG->DIAG first; here DIAG_var
        # is written directly, so the caller must ensure it is current.)
        names = ([rcnf.DIAG_name[nq] for nq in range(rcnf.DIAG_vmax0)]
                 + [rcnf.TRC_name[nq] for nq in range(rcnf.TRC_vmax)])

        if self.output_io_mode == "npz":
            # (ij,k,l) per var, keyed by name -> <basename><rank8>.npz
            path = basename + str(prc.prc_myrank).zfill(8) + ".npz"
            data = {name: self._advanced_pack(slot).astype(rdtype)
                    for slot, name in enumerate(names)}
            np.savez(path, **data)
        elif self.output_io_mode == "ADVANCED":
            path = basename + str(prc.prc_myrank).zfill(6)
            datatype = fio.RDTYPE2FIO[np.dtype(rdtype)]
            rgnid = [int(adm.RGNMNG_lp2r[l, adm.ADM_prc_me]) for l in range(adm.ADM_lall)]
            meta = dict(header='INITIAL/RESTART_data_of_prognostic_variables', note='',
                        fmode=0, endian=2, topo=0, glevel=adm.ADM_glevel, rlevel=adm.ADM_rlevel,
                        num_of_rgn=adm.ADM_lall, rgnid=rgnid)
            descs = self._DLABEL + [rcnf.WLABEL[nq] for nq in range(rcnf.TRC_vmax)]
            units = self._DUNIT + ['kg/kg'] * rcnf.TRC_vmax
            items = [dict(varname=name, description=descs[slot], unit=units[slot],
                          layername=self.layername, datatype=datatype, num_layer=adm.ADM_kall,
                          step=1, time_start=int(ctime), time_end=int(ctime),
                          data=self._advanced_pack(slot))
                     for slot, name in enumerate(names)]
            fio.fio_write(path, meta, items)
        else:
            print(f"xxx [prgvar] restart_output supports ADVANCED/npz (got {self.output_io_mode}).")
            prc.prc_mpistop(std.io_l, std.fname_log)
            return

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(f"*** wrote {self.output_io_mode} restart file: {path}", file=log_file)

    def _advanced_unpack(self, variable_array, slot, rdtype):
        # flat (ij,k,l) fio array -> DIAG_var[i,j,k,l,slot]. ij = j*g1d + i, so
        # reshape(g1d,g1d,...) gives (j,i,k,l); transpose(1,0,2,3) -> (i,j,k,l).
        # Same unpack as the json/npz path (verified bit-identical).
        g1d = adm.ADM_gall_1d
        arr = np.asarray(variable_array).reshape(g1d, g1d, *variable_array.shape[1:])
        self.DIAG_var[:, :, :, :, slot] = arr.transpose(1, 0, 2, 3).astype(rdtype)

    def restart_input(self, fname_in, comm, gtl, cnst, rcnf, grd, vmtr, cnvv, tdyn, idi, rdtype):

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("\n*** read restart/initial data", file=log_file)

        if self.input_io_mode == "ADVANCED":
            # native NICAM fio (PaNDa) binary restart: <basename>.pe is the prefix
            # (already carries the trailing '.pe' as in the json/IDEAL convention),
            # the 6-digit rank is appended.
            base = self.restart_input_basename + str(prc.prc_myrank).zfill(6)
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"*** reading ADVANCED (fio) restart file: {base}", file=log_file)
            _meta, _vars = fio.fio_read(base)

            # Read diagnostic variables (by name), then tracers (allow_missingq -> 0).
            for nq in range(rcnf.DIAG_vmax0):
                self._advanced_unpack(_vars[rcnf.DIAG_name[nq]], nq, rdtype)
            for nq in range(self.TRC_vmax_input):
                name = rcnf.TRC_name[nq]
                slot = rcnf.DIAG_vmax0 + nq
                if name in _vars:
                    self._advanced_unpack(_vars[name], slot, rdtype)
                elif self.allow_missingq:
                    if std.io_l:
                        with open(std.fname_log, 'a') as log_file:
                            print(f"*** missing tracer '{name}' in restart -> set to 0", file=log_file)
                    self.DIAG_var[:, :, :, :, slot] = rdtype(0.0)
                else:
                    print(f"xxx [prgvar] tracer '{name}' not found in restart file. STOP.")
                    prc.prc_mpistop(std.io_l, std.fname_log)

        elif self.input_io_mode in ("json", "npz"):
            with open(std.fname_log, 'a') as log_file:
                    print(f"*** reading {self.input_io_mode} restart file", file=log_file)

            base = self.restart_input_basename + str(prc.prc_myrank).zfill(8)
            if self.input_io_mode == "json":
                import json
                with open(base + ".json", "r") as json_file:
                    loaded_data = json.load(json_file)
                # (varname, Data-array) in file order
                items = [(k, np.array(v["Data"])) for k, v in loaded_data["Variables"].items()]
            else:  # "npz": arrays keyed by varname (tools/restart2json.py --format npz)
                nz = np.load(base + ".npz")
                items = [(k, nz[k]) for k in nz.files]

            # Unpack flat (ij, k, l) restart arrays into DIAG_var[i, j, k, l, var].
            # The original per-(i,j) loop is exactly a reshape+transpose: the flat
            # index is ij = j*ADM_gall_1d + i (j outer), so reshape(g1d, g1d, ...)
            # gives axes (j, i, k, l) and transpose(1,0,2,3) -> (i, j, k, l).
            # Verified bit-identical to the loop (gl05/gl07, all ranks, f32/f64).
            g1d = adm.ADM_gall_1d
            for cnt, (varname, variable_array) in enumerate(items):
                arr = np.asarray(variable_array).reshape(g1d, g1d, *variable_array.shape[1:])
                self.DIAG_var[:, :, :, :, cnt] = arr.transpose(1, 0, 2, 3).astype(rdtype)
 
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
        
            self.DIAG_var = idi.dycore_input(fname_in, cnst, rcnf, grd, idi, rdtype)

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

        # env-gated initial-condition dump (validation vs nicamdc). PYNICAM_IC_DUMP=<path>.
        import os as _os
        _ic_dump = _os.environ.get("PYNICAM_IC_DUMP", "")
        if _ic_dump:
            np.savez(f"{_ic_dump}_rank{prc.prc_myrank}.npz", DIAG_var=np.asarray(self.DIAG_var))

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
    
