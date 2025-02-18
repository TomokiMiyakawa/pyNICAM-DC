import numpy as np
import toml

# Global instances are instantiated in the modules when first called
# They will be singleton
from mod_process import prc 
from mod_adm import adm

# These classes are instantiated in this main program after the toml file is read and the Mkhgrid class is instantiated
from mod_stdio import Stdio
from mod_precision import Precision
from mod_const import Const
from mod_comm import Comm
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

        cnfs = toml.load(fname_in)['param_mkgrd']
        self.mkgrd_dospring = cnfs['mkgrd_dospring']
        self.mkgrd_doprerotate = cnfs['mkgrd_doprerotate']
        self.mkgrd_dostretch = cnfs['mkgrd_dostretch']
        self.mkgrd_doshrink = cnfs['mkgrd_doshrink']
        self.mkgrd_dorotate = cnfs['mkgrd_dorotate']
        self.mkgrd_in_basename = cnfs['mkgrd_in_basename']
        self.mkgrd_in_io_mode = cnfs['mkgrd_in_io_mode']
        self.mkgrd_out_basename = cnfs['mkgrd_out_basename']
        self.mkgrd_out_io_mode = cnfs['mkgrd_out_io_mode']

        self.mkgrd_spring_beta = cnfs['mkgrd_spring_beta']
        self.mkgrd_prerotation_tilt = cnfs['mkgrd_prerotation_tilt'] 
        self.mkgrd_stretch_alpha = cnfs['mkgrd_stretch_alpha'] 
        self.mkgrd_shrink_level = cnfs['mkgrd_shrink_level'] 
        self.mkgrd_rotation_lon = cnfs['mkgrd_rotation_lon']
        self.mkgrd_rotation_lat = cnfs['mkgrd_rotation_lat']
        self.mkgrd_precision_single = cnfs['mkgrd_precision_single']


#  main program start

# read configuration file (toml) and instantiate Mkhgrid class
intoml = 'prep.toml'
mkg  = Mkhgrid(intoml)

# instantiate classes
pre  = Precision(mkg.mkgrd_precision_single)
cnst = Const(mkg.mkgrd_precision_single)
std  = Stdio()
comm = Comm()

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

cnst.CONST_setup(std.io_l, std.io_nml, std.fname_log, intoml)
print("CONST_setup done")

adm.ADM_setup(std.io_l, std.io_nml, std.fname_log, intoml)
print("ADM_setup done")

print("hio and fio skip")
#  !---< I/O module setup >---
#  call FIO_setup
#  call HIO_setup

print("COMM_setup start")
comm.COMM_setup(std.io_l, std.io_nml, std.fname_log, intoml)
print("COMM_setup (not) done")

#  call MKGRD_setup

#  call MKGRD_standard

#  call MKGRD_spring

#  call GRD_output_hgrid( basename      = MKGRD_OUT_BASENAME, & ! [IN]
#                         output_vertex = .false.,            & ! [IN]
#                         io_mode       = MKGRD_OUT_io_mode   ) ! [IN]

#  !--- finalize all process
#prc.prc_mpifinish(std.io_l, std.fname_log)
##  call PRC_MPIfinish

print("peacefully done")
#  end program mkhgrid
