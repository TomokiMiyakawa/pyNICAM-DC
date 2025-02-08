import numpy as np
import toml
from adm import Adm
from stdio import Stdio
#from process import Process
from process import prc 
from precision import Precision
from const import Const
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
adm  = Adm()
cnst = Const(mkg.mkgrd_precision_single)
#prc  = Process()   # instantiated in process.py
std  = Stdio()
        
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
#---< Logfile setup >---
std.io_log_setup(prc.prc_myrank, is_master)

cnst.CONST_setup(std.io_l, std.io_nml, std.fname_log, intoml)

adm.ADM_setup(std.io_l, std.io_nml, std.fname_log, intoml)

#  call ADM_setup
#
#  !---< I/O module setup >---
#  call FIO_setup
#  call HIO_setup
#  !---< comm module setup >---
#  call COMM_setup
#  !---< mkgrid module setup >---
#  call MKGRD_setup

