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


class Io:
    
    _instance = None
    
    def __init__(self):
        pass

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
            self.PRGout_tracers = False
            self.PRGout_diagnostics = False
            self.PRGout_step0 = False
            self._diag_items_want = None

        else:
            cnfs = cnfs['ioparam']
            self.PRGout_name = cnfs['PRGout_name']
            self.PRGout_interval = cnfs['PRGout_interval']
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
        interval = self.PRGout_interval
        nt = max(1, (max(0, lstep - 1) + interval - 1) // interval)
        if getattr(self, "PRGout_step0", False):
            nt += 1                       # extra leading slot for the step-0 (IC) snapshot
        self._nt = nt
        self._it = 0
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
        out_names = ["RHOG", "RHOGVX", "RHOGVY", "RHOGVZ", "RHOGW", "RHOGE"]
        out_idx   = [rcnf.I_RHOG, rcnf.I_RHOGVX, rcnf.I_RHOGVY,
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
        shape2d = (nt, ni, nj, nr)

        ds = xr.Dataset({
            **{nm: (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype))
               for nm in out_names + self._diag_names},
            **{nm: (["time", "i", "j", "r"], da.empty(shape2d, chunks=shape2d, dtype=rdtype))
               for nm in self._diag_names_2d},
        }, coords={
            "time": (("time",), np.arange(nt)),
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
            {"time": 1, "i": ni, "j": nj, "r": nl},   # 2D pressure-level slices (time,i,j,r)
            {"time": nt},
            {"i": ni, "j": nj, "r": nl},
            {"i": ni, "j": nj, "r": nl, "xyz": 3},
        ]
        chunks = {tuple(sorted(c)): c for c in chunks}

        encoding = {
            name: {
                "chunks": tuple(chunks[tuple(sorted(var.dims))][d] for d in var.dims),
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

    def IO_PRGstep(self, tim, prgv, rcnf, rdtype, diag=None):

        data = {
            nm: (["time", "i", "j", "k", "r"], prgv.PRG_var[None, :, :, :, :, idx])
            for nm, idx in zip(self._out_names, self._out_idx)
        }
        # derived history diagnostics (computed by dyn.history_vars_step, passed in)
        if getattr(self, "PRGout_diagnostics", False) and diag is not None:
            for nm in self._diag_names:      # model-level (i,j,k,l) -> (time,i,j,k,r)
                data[nm] = (["time", "i", "j", "k", "r"], np.asarray(diag[nm])[None, ...])
            for nm in self._diag_names_2d:   # pressure-level slice (i,j,l) -> (time,i,j,r)
                data[nm] = (["time", "i", "j", "r"], np.asarray(diag[nm])[None, ...])
        dsregion = xr.Dataset(data)

        nl = adm.ADM_shape[3]
        #nr = nl * prc.prc_nprocs
        #nxyz=3
        myrank = prc.prc_myrank
        rs=int(myrank*nl)
        re=int((myrank+1)*nl - 1)
        # write to the next snapshot slot (internal counter, robust for any interval /
        # chunked time loop). Skip if the schema slots are exhausted (defensive).
        it = self._it
        self._it += 1
        if it >= self._nt:
            return
        dsregion.to_zarr(self.PRGout_name, mode="r+", region={"time": slice(it, it+1), "r": slice(rs, re+1)})

        return