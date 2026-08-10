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

    # Restart variable set (mod_prgvar_restart.f90). The prognostic form
    # (rhog, rhog{vx,vy,vz,w,e}, rhog<q>) is the DEFAULT and is bit-exact on a
    # round trip (pure array load). The diagnostic form (pre, tem, vx.., q) is
    # the optional/fallback physical format, reconstructed through cnvvar and
    # therefore NOT bit-exact. On input, prognostics are auto-detected (rhog &
    # rhoge present) unless input_diagnostics forces the diagnostic read.
    input_prognostics  = True    # .false. -> forces input_diagnostics
    output_prognostics = True    # .false. -> forces output_diagnostics
    input_diagnostics  = False   # force reading pre/tem instead of auto-detect
    output_diagnostics = False   # also write pre/tem

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
            self.input_prognostics  = cnfs.get('input_prognostics',  self.input_prognostics)
            self.output_prognostics = cnfs.get('output_prognostics', self.output_prognostics)
            self.input_diagnostics  = cnfs.get('input_diagnostics',  self.input_diagnostics)
            self.output_diagnostics = cnfs.get('output_diagnostics', self.output_diagnostics)

        # prgvar_restart_setup: prognostics=false forces the diagnostic form.
        if not self.input_prognostics:
            self.input_diagnostics = True
        if not self.output_prognostics:
            self.output_diagnostics = True

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
    
    # nicamdc restart metadata (mod_prgvar_restart.f90). Diagnostic (physical) set:
    _DLABEL = ['Pressure', 'Temperature', 'H-Velocity(XDIR)',
               'H-Velocity(YDIR)', 'H-Velocity(ZDIR)', 'V-Velocity']
    _DUNIT  = ['Pa', 'K', 'm/s', 'm/s', 'm/s', 'm/s']
    # Prognostic (conservative) set -- PRG_name order [rhog, rhogvx, rhogvy, rhogvz, rhogw, rhoge]:
    _PLABEL = ['Density * G^1/2',
               'Density * G^1/2 * H-Velocity(XDIR)', 'Density * G^1/2 * H-Velocity(YDIR)',
               'Density * G^1/2 * H-Velocity(ZDIR)', 'Density * G^1/2 * V-Velocity',
               'Density * G^1/2 * Energy']
    _PUNIT  = ['kg/m3', 'kg/m3*m/s', 'kg/m3*m/s', 'kg/m3*m/s', 'kg/m3*m/s', 'kg/m3*J/kg']

    def _pack5d(self, arr5d, slot):
        # (i,j,k,l) slot -> flat (ij,k,l), ij = j*g1d+i (j outer). Inverse of _unpack5d.
        gall = adm.ADM_gall_1d * adm.ADM_gall_1d
        arr = arr5d[:, :, :, :, slot].transpose(1, 0, 2, 3)   # (j,i,k,l)
        return arr.reshape(gall, adm.ADM_kall, adm.ADM_lall)   # (ij,k,l)

    def _advanced_pack(self, slot):
        # back-compat wrapper: pack a DIAG_var slot.
        return self._pack5d(self.DIAG_var, slot)

    def _restart_output_items(self, rcnf):
        # (varname, description, unit, source_array, slot) for every variable to write,
        # honoring output_prognostics / output_diagnostics. Prognostic tracer key is
        # 'rhog'+TRC_name (NICAM convention); diagnostic tracer key is TRC_name.
        items = []
        if self.output_prognostics:
            for nq in range(rcnf.PRG_vmax0):
                items.append((rcnf.PRG_name[nq], self._PLABEL[nq], self._PUNIT[nq],
                              self.PRG_var, nq))
            for nq in range(rcnf.TRC_vmax):
                items.append(('rhog' + rcnf.TRC_name[nq],
                              'Density * G^1/2 * ' + rcnf.WLABEL[nq], 'kg/m3',
                              self.PRG_var, rcnf.PRG_vmax0 + nq))
        if self.output_diagnostics:
            for nq in range(rcnf.DIAG_vmax0):
                items.append((rcnf.DIAG_name[nq], self._DLABEL[nq], self._DUNIT[nq],
                              self.DIAG_var, nq))
            for nq in range(rcnf.TRC_vmax):
                items.append((rcnf.TRC_name[nq], rcnf.WLABEL[nq], 'kg/kg',
                              self.DIAG_var, rcnf.DIAG_vmax0 + nq))
        return items

    def restart_output(self, basename, rcnf, rdtype, ctime=0):
        # Write a restart file (inverse of restart_input). output_prognostics (default)
        # writes the conservative PRG_var directly -> BIT-EXACT round trip;
        # output_diagnostics additionally writes the physical DIAG_var (reconstructed,
        # not bit-exact). ADVANCED -> native NICAM fio (byte-compatible with nicamdc);
        # npz -> numpy archive keyed by variable name. basename carries the trailing
        # '.pe'. The caller must ensure the written source arrays are current.
        items = self._restart_output_items(rcnf)

        if self.output_io_mode == "npz":
            # (ij,k,l) per var, keyed by name -> <basename><rank8>.npz
            path = basename + str(prc.prc_myrank).zfill(8) + ".npz"
            data = {name: self._pack5d(src, slot).astype(rdtype)
                    for (name, _desc, _unit, src, slot) in items}
            np.savez(path, **data)
        elif self.output_io_mode == "ADVANCED":
            path = basename + str(prc.prc_myrank).zfill(6)
            datatype = fio.RDTYPE2FIO[np.dtype(rdtype)]
            rgnid = [int(adm.RGNMNG_lp2r[l, adm.ADM_prc_me]) for l in range(adm.ADM_lall)]
            meta = dict(header='INITIAL/RESTART_data_of_prognostic_variables', note='',
                        fmode=0, endian=2, topo=0, glevel=adm.ADM_glevel, rlevel=adm.ADM_rlevel,
                        num_of_rgn=adm.ADM_lall, rgnid=rgnid)
            fio_items = [dict(varname=name, description=desc, unit=unit,
                              layername=self.layername, datatype=datatype, num_layer=adm.ADM_kall,
                              step=1, time_start=int(ctime), time_end=int(ctime),
                              data=self._pack5d(src, slot))
                         for (name, desc, unit, src, slot) in items]
            fio.fio_write(path, meta, fio_items)
        else:
            print(f"xxx [prgvar] restart_output supports ADVANCED/npz (got {self.output_io_mode}).")
            prc.prc_mpistop(std.io_l, std.fname_log)
            return

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(f"*** wrote {self.output_io_mode} restart file: {path} "
                      f"(prognostics={self.output_prognostics}, diagnostics={self.output_diagnostics})",
                      file=log_file)

    def _unpack5d(self, dest5d, variable_array, slot, rdtype):
        # flat (ij,k,l) fio array -> dest5d[i,j,k,l,slot]. ij = j*g1d + i, so
        # reshape(g1d,g1d,...) gives (j,i,k,l); transpose(1,0,2,3) -> (i,j,k,l).
        # Same unpack as the json/npz path (verified bit-identical).
        g1d = adm.ADM_gall_1d
        variable_array = np.asarray(variable_array)
        arr = variable_array.reshape(g1d, g1d, *variable_array.shape[1:])
        dest5d[:, :, :, :, slot] = arr.transpose(1, 0, 2, 3).astype(rdtype)

    def _advanced_unpack(self, variable_array, slot, rdtype):
        # back-compat wrapper: unpack into DIAG_var.
        self._unpack5d(self.DIAG_var, variable_array, slot, rdtype)

    def _load_restart_raw(self):
        # {varname: flat (ij,k[,l]) array} for the current file-backed input_io_mode,
        # or None for modes without a file (IDEAL/POH5). Used for prognostic-vs-
        # diagnostic auto-detection and the prognostic (bit-exact) read. The
        # diagnostic read path keeps its own inline loaders (byte-identical to the
        # validated version), so this does not disturb it.
        if self.input_io_mode == "ADVANCED":
            base = self.restart_input_basename + str(prc.prc_myrank).zfill(6)
            _meta, _vars = fio.fio_read(base)
            return dict(_vars)
        elif self.input_io_mode == "npz":
            base = self.restart_input_basename + str(prc.prc_myrank).zfill(8)
            nz = np.load(base + ".npz")
            return {k: nz[k] for k in nz.files}
        elif self.input_io_mode == "json":
            import json
            base = self.restart_input_basename + str(prc.prc_myrank).zfill(8)
            with open(base + ".json", "r") as json_file:
                loaded = json.load(json_file)
            return {k: np.array(v["Data"]) for k, v in loaded["Variables"].items()}
        return None

    def _read_prognostic_restart(self, comm, gtl, cnst, rcnf, vmtr, cnvv, tdyn, rdtype, raw):
        # Bit-exact prognostic restart: load PRG_var directly from the file (rhog,
        # rhog{vx,vy,vz,w,e}, rhog<q>), COMM the halos/poles, and reconstruct DIAG_var
        # via prg2diag (NICAM does the same). The model integrates PRG_var directly,
        # so the read is a pure array load -> bit-for-bit restart.
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("*** restart: prognostic variables detected -> reading "
                      "PRG_var directly (bit-exact)", file=log_file)

        for nq in range(rcnf.PRG_vmax0):
            self._unpack5d(self.PRG_var, raw[rcnf.PRG_name[nq]], nq, rdtype)

        for nq in range(self.TRC_vmax_input):
            key = 'rhog' + rcnf.TRC_name[nq]
            slot = rcnf.PRG_vmax0 + nq
            if key in raw:
                self._unpack5d(self.PRG_var, raw[key], slot, rdtype)
            elif self.allow_missingq:
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print(f"*** missing prognostic tracer '{key}' -> set to 0", file=log_file)
                self.PRG_var[:, :, :, :, slot] = rdtype(0.0)
            else:
                print(f"xxx [prgvar] prognostic tracer '{key}' not found in restart file. STOP.")
                prc.prc_mpistop(std.io_l, std.fname_log)

        comm.COMM_var(self.PRG_var, self.PRG_var_pl)
        self.DIAG_var, self.DIAG_var_pl = cnvv.cnvvar_prg2diag(
            self.PRG_var, self.PRG_var_pl, cnst, vmtr, rcnf, tdyn, rdtype)

        # range check (collectives called on all ranks; printed on log ranks)
        for nq in range(rcnf.PRG_vmax0):
            val_max = gtl.GTL_max(self.PRG_var[:, :, :, :, nq], self.PRG_var_pl[:, :, :, nq],
                                  adm.ADM_kall, adm.ADM_kmin, adm.ADM_kmax, cnst, comm, rdtype)
            val_min = gtl.GTL_min(self.PRG_var[:, :, :, :, nq], self.PRG_var_pl[:, :, :, nq],
                                  adm.ADM_kall, adm.ADM_kmin, adm.ADM_kmax, cnst, comm, rdtype)
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"--- {rcnf.PRG_name[nq]:<16}: max={val_max:24.17e}, min={val_min:24.17e}", file=log_file)

    def restart_input(self, fname_in, comm, gtl, cnst, rcnf, grd, vmtr, cnvv, tdyn, idi, rdtype):

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("\n*** read restart/initial data", file=log_file)

        # Prognostic (bit-exact) restart, auto-detected for file-backed modes unless
        # input_diagnostics forces the physical read. Self-contained early return so
        # the validated diagnostic/IDEAL flow below is untouched. Falls through to
        # that flow when the file has no rhog/rhoge (e.g. an externally-generated IC).
        if self.input_io_mode in ("ADVANCED", "npz", "json") and not self.input_diagnostics:
            raw = self._load_restart_raw()
            if raw is not None and ('rhog' in raw) and ('rhoge' in raw):
                self._read_prognostic_restart(comm, gtl, cnst, rcnf, vmtr, cnvv, tdyn, rdtype, raw)
                import os as _os
                _ic_dump = _os.environ.get("PYNICAM_IC_DUMP", "")
                if _ic_dump:
                    np.savez(f"{_ic_dump}_rank{prc.prc_myrank}.npz", DIAG_var=np.asarray(self.DIAG_var))
                return
            elif std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("*** restart: no rhog/rhoge in file -> diagnostic read", file=log_file)

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
    
