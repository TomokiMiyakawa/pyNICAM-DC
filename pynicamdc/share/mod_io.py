import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
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

    def IO_setup(self, fname_in, tim, grd, rdtype):

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

        else:
            cnfs = cnfs['ioparam']
            self.PRGout_name = cnfs['PRGout_name']
            self.PRGout_interval = cnfs['PRGout_interval']

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

        ds = xr.Dataset({
            "RHOG"  : (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype)), # {"long_name": "..."}),
            "RHOGVX": (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype)),
            "RHOGVY": (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype)),
            "RHOGVZ": (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype)),
            "RHOGW" : (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype)),
            "RHOGE" : (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype)),
            # "qv"       : (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype)), 
            # "passive00": (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype)),
            # "passive01": (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype)),
            # "passive02": (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype)),
            # "passive03": (["time", "i", "j", "k", "r"], da.empty(shape, chunks=shape, dtype=rdtype)),
            #"RHOG": xr.DataArray(da.empty(shape, chunks=shape, dtype=rdtype), dims=["time", "i", "j", "k", "r"]),  # the same
            # more variables here
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

    def IO_PRGstep(self, tim, prgv, rcnf, rdtype):

        dsregion = xr.Dataset({
            "RHOG"   : (["time", "i", "j", "k", "r"], prgv.PRG_var[None,:,:,:,:,rcnf.I_RHOG  ]),
            "RHOGVX" : (["time", "i", "j", "k", "r"], prgv.PRG_var[None,:,:,:,:,rcnf.I_RHOGVX]),
            "RHOGVY" : (["time", "i", "j", "k", "r"], prgv.PRG_var[None,:,:,:,:,rcnf.I_RHOGVY]),
            "RHOGVZ" : (["time", "i", "j", "k", "r"], prgv.PRG_var[None,:,:,:,:,rcnf.I_RHOGVZ]),
            "RHOGW"  : (["time", "i", "j", "k", "r"], prgv.PRG_var[None,:,:,:,:,rcnf.I_RHOGW ]),
            "RHOGE"  : (["time", "i", "j", "k", "r"], prgv.PRG_var[None,:,:,:,:,rcnf.I_RHOGE ]),
            #"qv"        : (["time", "i", "j", "k", "r"], prgv.PRG_var[None,:,:,:,:, 6]),
            #"passive00" : (["time", "i", "j", "k", "r"], prgv.PRG_var[None,:,:,:,:, 7]),
            #"passive01" : (["time", "i", "j", "k", "r"], prgv.PRG_var[None,:,:,:,:, 8]),
            #"passive02" : (["time", "i", "j", "k", "r"], prgv.PRG_var[None,:,:,:,:, 9]),
            #"passive03" : (["time", "i", "j", "k", "r"], prgv.PRG_var[None,:,:,:,:,10]),
        })

        nl = adm.ADM_shape[3]
        #nr = nl * prc.prc_nprocs
        #nxyz=3
        myrank = prc.prc_myrank
        rs=int(myrank*nl)
        re=int((myrank+1)*nl - 1)
        it=int(tim.TIME_cstep/self.PRGout_interval)
        dsregion.to_zarr(self.PRGout_name, mode="r+", region={"time": slice(it, it+1), "r": slice(rs, re+1)})

        return