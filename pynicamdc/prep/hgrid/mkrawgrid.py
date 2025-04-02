import numpy as np
import toml
import zarr
import sys
import os
#from concurrent.futures import ThreadPoolExecutor
script_dir = os.path.dirname(os.path.abspath(__file__))
share_module_dir = os.path.join(script_dir, "../../share")  
sys.path.insert(0, share_module_dir)

# Global instances are instantiated in the modules when first called
# They will be singleton
from mod_process import prc 
from mod_adm import adm
from mod_prof import prf
from mod_stdio import std
from mod_vector import vect

# These classes are instantiated in this main program after the toml file is read and the Mkhgrid class is instantiated
from mod_precision import Precision
from mod_const import Const
from mod_comm import Comm
from mod_mkgrd import Mkgrd
from mod_gtl import Gtl
#from mod_prof import Prof

class Mkhgrid:
    def __init__(self,fname_in):

        # Load configurations from TOML file
        #cnfs = toml.load('prep.toml')['precision_sd']
        #lsingle = cnfs['lsingle']
        cnfs = toml.load(fname_in)['admparam']
        self.glevel = cnfs['glevel']
        self.rlevel = cnfs['rlevel']
        #self.vlayer = cnfs['vlayer']
        self.rgnmngfname = cnfs['rgnmngfname']

#  main program start

# read configuration file (toml) and instantiate Mkhgrid class
intoml = '../../case/config/mkrawgrid.toml'
main  = Mkhgrid(intoml)

# Conditionally select executor type for parallel file writing based on configuration
cnfs = toml.load(intoml)['admparam']
executor_type = cnfs.get("executor_type", "thread")
if executor_type == "process":
    from concurrent.futures import ProcessPoolExecutor as Executor
else:
    from concurrent.futures import ThreadPoolExecutor as Executor

# instantiate classes
mkg = Mkgrd(intoml)
pre  = Precision(mkg.mkgrd_precision_single)  #True if single precision, False if double precision
cnst = Const(mkg.mkgrd_precision_single)
#prf  = Prof()  
gtl = Gtl()

#print("RP:", repr(pre.RP))
#print("RP_PREC:", pre.RP_PREC)
#r = pre.rdtype(1.234567890123456789012)
#print("r:", r)

comm = Comm(pre.rdtype)


# ---< MPI start >---
#comm_world=mkg.prc.prc_mpistart()
comm_world = prc.prc_mpistart()
#rank = comm_world.Get_rank()
if prc.prc_myrank == 0:
    is_master = True
else:
    is_master = False
 
#size = comm_world.Get_size()
#name =prc.prc_local_comm_world() 
##name = MPI.Get_processor_name() 
print(f"Hello, world! from rank {prc.prc_myrank} out of {prc.prc_nprocs}")

#---< STDIO setuppyNICAM-DC', intoml)
#std=Stdio()
std.io_setup('pyNICAM-DC', intoml)
print("io_setup done")
#---< Logfile setup >---
std.io_log_setup(prc.prc_myrank, is_master)
print("io_log_setup done")

prf.PROF_setup(intoml, pre.rdtype)
print("PROF_setup done")

prf.PROF_setprefx("INIT")
prf.PROF_rapstart("Initialize", 0)

cnst.CONST_setup(intoml)
print("CONST_setup done")

adm.ADM_setup(intoml)
print("ADM_setup done")

print("hio and fio skip")
#  !---< I/O module setup >---
#  call FIO_setup
#  call HIO_setup

#print("COMM_setup start")
comm.COMM_setup(intoml)
print("COMM_setup done")

#  call MKGRD_setup
mkg.mkgrd_setup(pre.rdtype)
print("mkgrd_setup done")

prf.PROF_rapend("Initialize", 0)

prf.PROF_setprefx("MAIN")
prf.PROF_rapstart("Main_MKGRD", 0)

prf.PROF_rapstart("MKGRD_standard", 0)
mkg.mkgrd_standard(pre.rdtype,cnst,comm)
prf.PROF_rapend("MKGRD_standard", 0)
print("mkgrd_standard done")

prf.PROF_rapstart("MKGRD_spring", 0)
mkg.mkgrd_spring(pre.rdtype,cnst,comm,gtl)
prf.PROF_rapend("MKGRD_spring", 0)
print("mkgrd_spring done")

# Add additional performance profiling for file I/O
prf.PROF_rapstart("Write_zarr_files", 0)

p = prc.prc_myrank
def write_region(l):
    region = adm.RGNMNG_lp2r[l, p]
    outfile = "../../case/prepdata/" + mkg.mkgrd_out_basename + ".zarr" + f"{region:08d}"
    zarr_store = zarr.open(outfile, mode="w", shape=mkg.GRD_x[:,:,0,l,:].shape, dtype=pre.rdtype)
    zarr_store[:,:,:] = mkg.GRD_x[:,:,0,l,:]
    zarr_store.attrs["units"] = "xyz Cartesian coordinate unit globe"
    zarr_store.attrs["description"] = "raw grid data"
    zarr_store.attrs["glevel"] = adm.ADM_glevel
    zarr_store.attrs["rlevel"] = adm.ADM_rlevel
    zarr_store.attrs["region"] = f"{region:08d}"
    zarr_store.attrs["cnfs"] = mkg.cnfs

with Executor() as executor:
    list(executor.map(write_region, range(mkg.GRD_x.shape[3])))

prf.PROF_rapend("Write_zarr_files", 0)

prf.PROF_rapend("Main_MKGRD", 0)
prf.PROF_rapreport()

prc.prc_mpifinish(std.io_l, std.fname_log)

print("peacefully done")

#  end program mkhgrid

