import numpy as np
import toml

# Global instances are instantiated in the modules when first called
# They will be singleton
from mod_process import prc 
from mod_adm import adm
from mod_stdio import std

# These classes are instantiated in this main program after the toml file is read and the Mkhgrid class is instantiated
#from mod_stdio import Stdio
from mod_precision import Precision
from mod_const import Const
from mod_comm import Comm
from mod_mkgrd import Mkgrd
#from process import Comm
#from grd import Grd
#from gmtr import Gmtr
#from mpi4py import MPI

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
intoml = 'prep.toml'
main  = Mkhgrid(intoml)

# instantiate classes
mkg = Mkgrd(intoml)
#print(mkg.mkgrd_out_basename, 'ho')
pre  = Precision(mkg.mkgrd_precision_single)
cnst = Const(mkg.mkgrd_precision_single)

print("RP:", repr(pre.RP))
print("RP_PREC:", pre.RP_PREC)
r = pre.rdtype(1.234567890123456789012)
print("r:", r)

#std  = Stdio()
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

mkg.mkgrd_standard(pre.rdtype,cnst,comm)
print("mkgrd_standard done")
#  call MKGRD_standard

mkg.mkgrd_spring(pre.rdtype,cnst,comm)
print("mkgrd_spring (not) done")
#  call MKGRD_spring

#  call GRD_output_hgrid( basename      = MKGRD_OUT_BASENAME, & ! [IN]
#                         output_vertex = .false.,            & ! [IN]
#                         io_mode       = MKGRD_OUT_io_mode   ) ! [IN]

#  !--- finalize all process
#prc.prc_mpifinish(std.io_l, std.fname_log)
##  call PRC_MPIfinish

print("peacefully done")
#  end program mkhgrid
