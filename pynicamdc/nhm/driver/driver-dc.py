import numpy as np
import toml
import zarr
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
share_module_dir = os.path.join(script_dir, "../../share")  
nhmdyn_module_dir = os.path.join(script_dir, "../dynamics") 
nhmfrc_module_dir = os.path.join(script_dir, "../forcing")   
nhmshare_module_dir = os.path.join(script_dir, "../share")  
sys.path.insert(0, share_module_dir)
sys.path.insert(0, nhmdyn_module_dir)
sys.path.insert(0, nhmfrc_module_dir)
sys.path.insert(0, nhmshare_module_dir)

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
from mod_gtl import Gtl
#from mod_prof import Prof
from mod_grd import Grd
class Driver_dc:
    def __init__(self,fname_in):

        # Load configurations from TOML file
        #cnfs = toml.load('prep.toml')['precision_sd']
        #lsingle = cnfs['lsingle']
        cnfs = toml.load(fname_in)['admparam']
        self.glevel = cnfs['glevel']
        self.rlevel = cnfs['rlevel']
        #self.vlayer = cnfs['vlayer']
        self.rgnmngfname = cnfs['rgnmngfname']
        self.precision_single = cnfs['precision_single']

#  main program start

# read configuration file (toml) and instantiate Driver_dc class
intoml = '../../case/config/nhm_driver.toml'
main  = Driver_dc(intoml)   

# instantiate classes
pre  = Precision(main.precision_single)  #True if single precision, False if double precision
cnst = Const(main.precision_single)
#prf  = Prof()
gtl = Gtl() 
grd = Grd()

comm = Comm(pre.rdtype)

# ---< MPI start >---
comm_world = prc.prc_mpistart()
if prc.prc_myrank == 0:
    is_master = True
else:
    is_master = False

#---< STDIO setup >---
std.io_setup('pyNICAM-DC', intoml)
#print("io_setup done")

#---< Logfile setup >---
std.io_log_setup(prc.prc_myrank, is_master)
#print("io_log_setup done")

#---< profiler module setup >---
prf.PROF_setup(intoml, pre.rdtype)
#print("PROF_setup done")

prf.PROF_setprefx("INIT")
prf.PROF_rapstart("Initialize", 0)

#---< cnst module setup >---
cnst.CONST_setup(intoml)
#print("CONST_setup done")

# skip calendar module setup
#---< calendar module setup >---
#  call CALENDAR_setup

# skip random module setup
#---< radom module setup >---
#  call RANDOM_setup

#---< admin module setup >---
adm.ADM_setup(intoml)
#print("ADM_setup done")

#print("hio and fio skip")
#  !---< I/O module setup >---
#  call FIO_setup
#  call HIO_setup

#print("COMM_setup start")
comm.COMM_setup(intoml)
#print("COMM_setup done")

#---< grid module setup >---
grd.GRD_setup(intoml, cnst)
print("GRD_setup (not) done")
#  call GRD_setup

#---< geometrics module setup >---
#  call GMTR_setup

#---< operator module setup >---
#  call OPRT_setup

#---< vertical metrics module setup >---
#  call VMTR_setup

#---< time module setup >---
#  call TIME_setup

#---< external data module setup >---
#  call extdata_setup


#---< nhm_runconf module setup >---
#  call runconf_setup

#---< saturation module setup >---
#  call saturation_setup

#---< prognostic variable module setup >---
#  call prgvar_setup
#  call restart_input( restart_input_basename )

#---< dynamics module setup >---
#  call dynamics_setup

#---< forcing module setup >---
#  call forcing_setup

#---< energy&mass budget module setup >---
#  call embudget_setup

#---< history module setup >---
#  call history_setup

#---< history variable module setup >---
#  call history_vars_setup

prf.PROF_rapend("Initialize", 0)


prf.PROF_setprefx("MAIN")
prf.PROF_rapstart("Main_Loop", 0)

#--- history output at initial time
#  if ( HIST_output_step0 ) then
#     TIME_CSTEP = TIME_CSTEP - 1
#     TIME_CTIME = TIME_CTIME - TIME_DTL
#     call history_vars
#     call TIME_advance
#     call history_out
#  else
#     call TIME_report
#  endif

TIME_LSTEP_MAX = 3  

for n in range(TIME_LSTEP_MAX):

    prf.PROF_rapstart("_Atmos", 1)
    #     call dynamics_step
    #     call forcing_step
    prf.PROF_rapend("_Atmos", 1)

    prf.PROF_rapstart("_History", 1)
    #     call history_vars
    #     call TIME_advance

    #--- budget monitor
    #     call embudget_monitor
    #     call history_out

    if ( n == TIME_LSTEP_MAX -1 ):
        pass
#        call restart_output( restart_output_basename )
#        ??? no need to be inside the loop...

    prf.PROF_rapend("_History", 1)


prf.PROF_rapend("Main_Loop", 0)
prf.PROF_rapreport()

prc.prc_mpifinish(std.io_l, std.fname_log)

print("peacefully done")
