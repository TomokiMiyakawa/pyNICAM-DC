import toml
import numpy as np
#from mpi4py import MPI
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
#from mod_prof import prf
import dask.array as da
import zarr
#from zarr import DirectoryStore
#from zarr.storage import DirectoryStore
import xarray as xr
import threading
import queue


class Io:
    
    
    def __init__(self):
        pass

    def _make_compressor(self):
        # Build the zarr/numcodecs compressor from the [ioparam] settings; None disables
        # compression. Applied at store creation, so every region write inherits it.
        name = str(getattr(self, 'PRGout_compressor', 'lz4')).lower()
        if name in ('none', 'off', ''):
            return None
        from numcodecs import Blosc
        shuf = {'noshuffle': Blosc.NOSHUFFLE, 'shuffle': Blosc.SHUFFLE,
                'bitshuffle': Blosc.BITSHUFFLE}.get(
            str(getattr(self, 'PRGout_shuffle', 'shuffle')).lower(), Blosc.SHUFFLE)
        return Blosc(cname=name, clevel=int(getattr(self, 'PRGout_clevel', 1)), shuffle=shuf)

    # ---- output write (synchronous, or async on a background thread) --------------
    def _ensure_writer(self):
        # Lazily start the single background writer thread + its bounded queue. maxsize=2
        # gives one in-flight write + one queued; a fuller queue blocks the producer
        # (backpressure -> bounded memory, degrades to ~sync if writes can't keep up).
        if getattr(self, "_writer_thread", None) is not None:
            return
        self._write_q = queue.Queue(maxsize=2)
        self._writer_err = None

        def _worker():
            while True:
                item = self._write_q.get()
                try:
                    if item is None:
                        return
                    builder, region = item
                    if self._writer_err is None:
                        # builder() does the (strided) per-variable slicing off the main
                        # thread, then the compress+disk write -- all overlapping compute.
                        xr.Dataset(builder()).to_zarr(self.PRGout_name, mode="r+", region=region)
                except Exception as e:      # remember, surface on the main thread
                    self._writer_err = e
                finally:
                    self._write_q.task_done()

        self._writer_thread = threading.Thread(target=_worker, name="zarr-writer", daemon=True)
        self._writer_thread.start()

    def _enqueue(self, builder, region):
        # Hand a deferred write (builder + region) to the background writer. builder() is
        # a closure over already-taken CONTIGUOUS snapshots, so the model's live buffers
        # are free to mutate while the write is in flight. Blocks if the queue is full.
        self._ensure_writer()
        if self._writer_err is not None:                 # a prior async write failed
            raise self._writer_err
        self._write_q.put((builder, region))

    def IO_finalize(self):
        # Drain + join the background writer (call once after the main loop). No-op if
        # async was never used. Re-raises any deferred write error.
        t = getattr(self, "_writer_thread", None)
        if t is not None:
            self._write_q.put(None)
            self._writer_thread.join()
            self._writer_thread = None
        if getattr(self, "_writer_err", None) is not None:
            err = self._writer_err
            self._writer_err = None
            raise err

    def IO_setup(self, fname_in, tim, grd, rcnf, rdtype):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[io]/Category[common share]", file=log_file)        
                #print(f"*** input toml file is ", fname_in, file=log_file)
                print(f"currently only for quick output of prognostic variables", file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'ioparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** ioparam not found in toml file. using default.", file=log_file)
                #prc.prc_mpistop(std.io_l, std.fname_log)
                self.PRGout_name = "deftestout.zarr"
                self.PRGout_interval = 72
            self.PRGout_prognostics = True
            self.PRGout_tracers = False
            self.PRGout_diagnostics = False
            self.PRGout_step0 = False
            self._diag_items_want = None
            self.PRGout_interval_2d = self.PRGout_interval
            self.PRGout_compressor = 'lz4'
            self.PRGout_clevel = 1
            self.PRGout_shuffle = 'noshuffle'
            self.PRGout_async = False

        else:
            cnfs = cnfs['ioparam']
            self.PRGout_name = cnfs['PRGout_name']
            self.PRGout_interval = cnfs['PRGout_interval']
            # Write the base prognostics (RHOG,RHOGVX..RHOGE); default on. Turn off to
            # output only the derived diagnostics (ml_/sl_).
            self.PRGout_prognostics = bool(cnfs.get('PRGout_prognostics', True))
            # Append tracer fields (qv, passive...) to the output when enabled.
            self.PRGout_tracers = bool(cnfs.get('PRGout_tracers', False))
            # Append derived history diagnostics (ml_u/v/w/th/thv/omg/...) when enabled.
            self.PRGout_diagnostics = bool(cnfs.get('PRGout_diagnostics', False))
            # Also emit a snapshot at step 0 (the initial condition), like nicamdc doout_step0.
            self.PRGout_step0 = bool(cnfs.get('PRGout_step0', False))
            # Optional per-variable selection (nicamdc-style item list). When given, only
            # these diagnostics are computed+written; omit for the full set.
            _items = cnfs.get('PRGout_diag_items', None)
            self._diag_items_want = set(_items) if _items else None
            # Separate output interval for the 2D (single-/pressure-level sl_) diagnostics;
            # defaults to PRGout_interval (3D fields = prognostics + ml_ share PRGout_interval).
            self.PRGout_interval_2d = cnfs.get('PRGout_interval_2d', self.PRGout_interval)
            # zarr compression for the output store. Fast default (Blosc-lz4, level 1,
            # NO shuffle) -- measured on gl08 fp32: the prognostic data is near-
            # incompressible (shuffle+lz4 shrinks it only ~10% but costs ~24% more write
            # time), so noshuffle is the better speed/size point. PRGout_compressor:
            # 'lz4'|'zstd'|'blosclz'|'lz4hc'|'zlib'|'none'; PRGout_clevel: 0-9;
            # PRGout_shuffle: 'shuffle'|'noshuffle'|'bitshuffle'.
            self.PRGout_compressor = cnfs.get('PRGout_compressor', 'lz4')
            self.PRGout_clevel = int(cnfs.get('PRGout_clevel', 1))
            self.PRGout_shuffle = str(cnfs.get('PRGout_shuffle', 'noshuffle'))
            # Async output: hand the zarr write to a background thread so it overlaps the
            # next compute step (hidden behind compute). Default off (synchronous).
            self.PRGout_async = bool(cnfs.get('PRGout_async', False))

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        # Number of output snapshots. Output fires at large-step n = 1, 1+interval,
        # 1+2*interval, ... (n in [1, lstep_max); see driver-dc.py), so the count is
        # ceil((lstep_max-1)/interval). This is robust for ANY interval (incl. 1) and
        # matches the internal write counter self._it used in IO_PRGstep. (The old
        # int(lstep_max/interval) + it=TIME_cstep/interval mis-indexed for small
        # intervals -> zarr region-write "changing dimension size" errors.)
        lstep = tim.TIME_lstep_max
        step0 = 1 if getattr(self, "PRGout_step0", False) else 0

        def _nslots(iv):
            return max(1, (max(0, lstep - 1) + iv - 1) // iv) + step0

        # 3D group (prognostics + ml_) on the "time" axis; 2D group (sl_) on "time2d".
        nt = _nslots(self.PRGout_interval)
        nt2d = _nslots(self.PRGout_interval_2d)
        self._nt = nt; self._it = 0            # 3D write counter
        self._nt_2d = nt2d; self._it_2d = 0    # 2D write counter
        ni = adm.ADM_shape[0]
        nj = adm.ADM_shape[1]
        nk = adm.ADM_shape[2]
        nl = adm.ADM_shape[3]
        nr = nl * prc.prc_nprocs
        nxyz=3
        myrank = prc.prc_myrank
        shape = (nt, ni, nj, nk, nr)   # for all regions 

        #store_path = self.PRGout_name
        #zarr_store = DirectoryStore(store_path)
        #xr.DataArray(da.empty(shape, chunks=shape, dtype=rdtype), dims=["time", "i", "j", "k", "r"])

        # --- Output variable set: single source of truth, reused in IO_PRGstep ---
        # Base prognostics (RHOG..RHOGE) are always written. Tracers (qv,
        # passive...) are appended only when PRGout_tracers is enabled in the
        # toml ioparam, so e.g. a 6-var run and an 11-var tracer run share the
        # same code and differ only by config.
        out_names = []
        out_idx   = []
        if self.PRGout_prognostics:
            out_names += ["RHOG", "RHOGVX", "RHOGVY", "RHOGVZ", "RHOGW", "RHOGE"]
            out_idx   += [rcnf.I_RHOG, rcnf.I_RHOGVX, rcnf.I_RHOGVY,
                          rcnf.I_RHOGVZ, rcnf.I_RHOGW, rcnf.I_RHOGE]
        if self.PRGout_tracers:
            for v in range(rcnf.TRC_vmax):
                out_names.append(str(rcnf.TRC_name[v]))
                out_idx.append(rcnf.PRG_vmax0 + v)   # tracers follow the 6 base vars
        self._out_names = out_names
        self._out_idx   = out_idx

        # Derived history diagnostics (computed each output step from the prognostic;
        # not indexed into PRG_var). Model-level (ml_) share the (time,i,j,k,r) layout;
        # pressure-level slices (sl_) are single-level (time,i,j,r).
        self._diag_names = []      # 3D model-level diagnostics
        self._diag_names_2d = []   # 2D pressure-level slices
        if self.PRGout_diagnostics:
            self._diag_names = ['ml_u', 'ml_v', 'ml_w', 'ml_th', 'ml_thv',
                                'ml_omg', 'ml_pres', 'ml_tem', 'ml_rho', 'ml_hgt',
                                'ml_mse', 'ml_rh', 'ml_rha', 'ml_rhi',
                                'ml_du', 'ml_dv', 'ml_dw', 'ml_dtem', 'ml_dq',
                                'ml_ucos', 'ml_vcos', 'ml_th_prime']
            self._diag_names_2d = (['sl_ps', 'sl_pw', 'sl_lwp', 'sl_iwp']
                                   + [f'sl_{f}{lev}' for lev in ('850', '500', '250', '100')
                                      for f in ('u', 'v', 'w', 't')])
            # DCMIP Terminator chemistry column means (only produced for AF_TYPE=DCMIP)
            if getattr(rcnf, 'AF_TYPE', '') == 'DCMIP' and getattr(rcnf, 'NCHEM_STR', -1) >= 0:
                self._diag_names_2d += ['sl_cl', 'sl_cl2', 'sl_cly']
            # restrict to the requested subset (PRGout_diag_items), if given
            _w = self._diag_items_want
            if _w is not None:
                self._diag_names = [n for n in self._diag_names if n in _w]
                self._diag_names_2d = [n for n in self._diag_names_2d if n in _w]
        shape2d = (nt2d, ni, nj, nr)

        ds = xr.Dataset({
            **{nm: (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype))
               for nm in out_names + self._diag_names},
            **{nm: (["time2d", "i", "j", "r"], da.empty(shape2d, chunks=shape2d, dtype=rdtype))
               for nm in self._diag_names_2d},
        }, coords={
            "time": (("time",), np.arange(nt)),
            "time2d": (("time2d",), np.arange(nt2d)),
            "GRD_x": (["i", "j", "r", "xyz"], da.empty((ni,nj,nr,nxyz), chunks=(ni,nj,nr,nxyz), dtype=rdtype)),
            #"lat": (["i", "j", "r"], da.empty((ni,nj,nr), chunks=(ni,nj,nr), dtype=rdtype)),
        }, attrs={
            "title": "fancy simulation",
            "config": """some
            longer config
            """,
            "history": "derived from ...",
        })

        chunks = [
            {"time": 1, "i": ni, "j": nj, "k": nk, "r": nl},
            {"time2d": 1, "i": ni, "j": nj, "r": nl},   # 2D sl_ diagnostics (time2d,i,j,r)
            {"time": nt},
            {"time2d": nt2d},
            {"i": ni, "j": nj, "r": nl},
            {"i": ni, "j": nj, "r": nl, "xyz": 3},
        ]
        chunks = {tuple(sorted(c)): c for c in chunks}

        _comp = self._make_compressor()
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(f"*** output compressor: {_comp}", file=log_file)
        encoding = {
            name: {
                "chunks": tuple(chunks[tuple(sorted(var.dims))][d] for d in var.dims),
                "compressor": _comp,
            }
            for name, var in ds.variables.items()
        }


        if prc.prc_ismaster:
        #outname = self.PRGout_name
            ds.to_zarr(self.PRGout_name, compute=False, encoding=encoding, consolidated=True)
        #ds.to_zarr(self.PRGout_name, mode='w', compute=False, encoding=encoding)

        prc.PRC_MPIbarrier()

        dsgrd = xr.Dataset({
            "GRD_x": (["i", "j", "r", "xyz"], grd.GRD_x[:,:,0,:,:]),
        #     #"lat"  : (["i", "j", "r"], grd.GRD_lat[:,:,:]),
        # }, coords={
        #     "i": (["i"], np.arange(ni)),
        #     "j": (["j"], np.arange(nj)),
        #     "r": (["r"], np.arange(nr)),
        #     "xyz": (["xyz"], np.arange(3)),
        })

        rs=int(myrank*nl)
        re=int((myrank+1)*nl - 1)

        dsgrd.to_zarr(self.PRGout_name, mode="r+", region={"r": slice(rs, re+1)})

        return

    def IO_PRGstep(self, tim, prgv, rcnf, rdtype, diag=None, write_3d=True, write_2d=True):

        nl = adm.ADM_shape[3]
        myrank = prc.prc_myrank
        rs = int(myrank * nl)
        re = int((myrank + 1) * nl - 1)
        diag_on = getattr(self, "PRGout_diagnostics", False) and diag is not None
        is_async = getattr(self, "PRGout_async", False)
        D3 = ["time", "i", "j", "k", "r"]
        D2 = ["time2d", "i", "j", "r"]

        # --- 3D group: base prognostics + ml_ diagnostics, on the "time" axis ---
        if write_3d and (self._out_names or (diag_on and self._diag_names)):
            it = self._it
            self._it += 1
            if it < self._nt:
                region = {"time": slice(it, it + 1), "r": slice(rs, re + 1)}
                names_idx = list(zip(self._out_names, self._out_idx))
                dnames = self._diag_names if diag_on else []
                if is_async:
                    # ONE contiguous snapshot of PRG_var (fast memcpy); the strided per-
                    # variable slicing is deferred into the builder -> writer thread.
                    prog = np.array(prgv.PRG_var) if names_idx else None
                    dsnap = {nm: np.array(np.asarray(diag[nm])) for nm in dnames}

                    def _b3(prog=prog, dsnap=dsnap, names_idx=names_idx):
                        d = {nm: (D3, prog[None, :, :, :, :, ix]) for nm, ix in names_idx}
                        d.update({nm: (D3, a[None, ...]) for nm, a in dsnap.items()})
                        return d
                    self._enqueue(_b3, region)
                else:
                    data = {nm: (D3, prgv.PRG_var[None, :, :, :, :, ix]) for nm, ix in names_idx}
                    for nm in dnames:
                        data[nm] = (D3, np.asarray(diag[nm])[None, ...])
                    xr.Dataset(data).to_zarr(self.PRGout_name, mode="r+", region=region)

        # --- 2D group: sl_ diagnostics, on the "time2d" axis ---
        if write_2d and diag_on and self._diag_names_2d:
            it2 = self._it_2d
            self._it_2d += 1
            if it2 < self._nt_2d:
                region = {"time2d": slice(it2, it2 + 1), "r": slice(rs, re + 1)}
                if is_async:
                    dsnap = {nm: np.array(np.asarray(diag[nm])) for nm in self._diag_names_2d}

                    def _b2(dsnap=dsnap):
                        return {nm: (D2, a[None, ...]) for nm, a in dsnap.items()}
                    self._enqueue(_b2, region)
                else:
                    data2d = {nm: (D2, np.asarray(diag[nm])[None, ...]) for nm in self._diag_names_2d}
                    xr.Dataset(data2d).to_zarr(self.PRGout_name, mode="r+", region=region)

        return