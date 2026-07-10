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

        else:
            cnfs = cnfs['ioparam']
            self.PRGout_name = cnfs['PRGout_name']
            self.PRGout_interval = cnfs['PRGout_interval']
            # Append tracer fields (qv, passive...) to the output when enabled.
            self.PRGout_tracers = bool(cnfs.get('PRGout_tracers', False))
            # Append derived history diagnostics (ml_u/v/w/th/thv/omg/...) when enabled.
            self.PRGout_diagnostics = bool(cnfs.get('PRGout_diagnostics', False))

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        nt = int(tim.TIME_lstep_max / self.PRGout_interval)
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
                                'ml_omg', 'ml_pres', 'ml_tem', 'ml_rho', 'ml_hgt']
            self._diag_names_2d = [f'sl_{f}{lev}' for lev in ('850', '500', '250', '100')
                                   for f in ('u', 'v', 'w', 't')]
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
        it=int(tim.TIME_cstep/self.PRGout_interval)
        dsregion.to_zarr(self.PRGout_name, mode="r+", region={"time": slice(it, it+1), "r": slice(rs, re+1)})

        return