import numpy as np
import toml
from mod_stdio import stdio
#from process import Process
from mod_process import prc 
from mod_precision import Precision
from mod_const import Const
from mod_adm import Adm
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

        #self.prec =Precision(self.mkgrd_precision_single)
        #self.ad = Adm()
        #self.cnst = Const(self.mkgrd_precision_single)
        ###self.prc = Process()
        #self.std=Stdio()
        
intoml='prep.toml'

mkg=Mkhgrid(intoml)

prec =Precision(mkg.mkgrd_precision_single)
print('instantiated class Precision in mkhgrid as prec (imported from mod_precision.py) with single precision = ', mkg.mkgrd_precision_single)
cnst = Const(mkg.mkgrd_precision_single)
print('instantiated class Const in mkhgrid as cnst (imported from mod_const.py) with single precision = ', mkg.mkgrd_precision_single)

# ---< MPI start >---
#comm_world=mkg.prc.prc_mpistart()
comm_world=prc.prc_mpistart()
rank = comm_world.Get_rank()
if rank == 0:
    is_master = True
else:
    is_master = False
 
#size = comm_world.Get_size()
prc.nprocs=comm_world.Get_size()
#prc.nprocs=self.comm.Get_size()
#name =prc.prc_local_comm_world() 
##name = MPI.Get_processor_name() 
#print(f"Hello, world! from rank {rank} out of {size}")
print(f"Hello, world! from rank {rank} out of {prc.nprocs}")

adm=Adm(comm_world,np)

#---< STDIO setuppyNICAM-DC', intoml)
#std=Stdio()
stdio.io_setup('pyNICAM-DC', intoml)
#---< Logfile setup >---
stdio.io_log_setup(rank, is_master)

cnst.CONST_setup(stdio.io_l, stdio.io_nml, stdio.fname_log, intoml)

#mkg.ad.ADM_setup(mkg.std.io_l, mkg.std.io_nml, mkg.std.fname_log, intoml)

adm.ADM_setup(stdio.io_l, stdio.io_nml, stdio.fname_log, intoml)

#
#  !---< I/O module setup >---
#  call FIO_setup
#  call HIO_setup
#  !---< comm module setup >---
#  call COMM_setup
#  !---< mkgrid module setup >---
#  call MKGRD_setup
