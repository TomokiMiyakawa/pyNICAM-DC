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
from mod_io_param import iop
from mod_stdio import std
from mod_vector import vect
from mod_calendar import cldr
from mod_chemvar import chem
from mod_saturation import satr

# These classes are instantiated in this main program after the toml file is read and the Mkhgrid class is instantiated
from mod_precision import Precision
from mod_const import Const
from mod_comm import Comm
from mod_gtl import Gtl
#from mod_prof import Prof
from mod_grd import Grd
from mod_vmtr import Vmtr
from mod_gmtr import Gmtr
from mod_oprt import Oprt
from mod_time import Tim
from mod_runconf import Rcnf
from mod_prgvar import Prgv
from mod_cnvvar import Cnvv
from mod_thrmdyn import Tdyn
from mod_ideal_init import Idi
from mod_forcing import Frc
from mod_dynamics import Dyn
from mod_bndcnd import Bndc
from mod_bsstate import Bsst
from mod_numfilter import Numf
from mod_vi import Vi

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

print()
print("driver_dc.py start")

# read configuration file (toml) and instantiate Driver_dc class
intoml = '../../case/config/nhm_driver.toml'
main  = Driver_dc(intoml)   

# instantiate classes
pre  = Precision(main.precision_single)  #True if single precision, False if double precision
comm = Comm(pre.rdtype)
cnst = Const(main.precision_single)
#prf  = Prof()
gtl = Gtl() 
grd = Grd()
vmtr = Vmtr()
gmtr = Gmtr()
oprt = Oprt()
tim = Tim()
rcnf = Rcnf()
prgv = Prgv()
cnvv = Cnvv()
tdyn = Tdyn()
idi = Idi()
frc = Frc()
#dyn = Dyn() #rcnf, pre.rdtype)
bndc = Bndc()
bsst = Bsst()
numf = Numf()
vi   = Vi()

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

#---< calendar module setup >---
cldr.CALENDAR_setup(intoml)

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
grd.GRD_setup(intoml, cnst, comm, pre.rdtype)
#print("GRD_setup done") slight suspicion on the pl communication, where the original code may have a bug?

#---< geometrics module setup >---
gmtr.GMTR_setup(intoml, cnst, comm, grd, vect, pre.rdtype)
#print("GMTR_setup done")

#---< operator module setup >---
oprt.OPRT_setup(intoml, gmtr, pre.rdtype)
#print("OPRT_setup done")

#---< vertical metrics module setup >---
vmtr.VMTR_setup(intoml, cnst, comm, grd, gmtr, oprt, pre.rdtype)
#print("VMTR_setup done")  

#---< time module setup >---
tim.TIME_setup(intoml)
#print("TIME_setup done")

#==========================================

#---< external data module setup >---
#skip
#  call extdata_setup

#---< nhm_runconf module setup >---
rcnf.RUNCONF_setup(intoml,cnst)
#print("RUNCONF_setup done")

#---< saturation module setup >---
satr.SATURATION_setup(intoml,cnst)
#print("SATURATION_setup done")

#---< prognostic variable module setup >---
prgv.prgvar_setup(intoml, rcnf, pre.rdtype)
#print("prgvar_setup done")
prgv.restart_input(intoml, comm, gtl, cnst, rcnf, grd, vmtr, cnvv, tdyn, idi, pre.rdtype) #prgv.restart_input_basename)
#print("restart_input done")

#============================================

dyn = Dyn(rcnf, pre.rdtype)

#---< dynamics module setup >---
dyn.dynamics_setup(intoml, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, bndc, bsst, numf, vi, pre.rdtype)
#print("dynamics_setup done")

#---< forcing module setup >---
frc.forcing_setup(intoml, rcnf, pre.rdtype)
#print("forcing_setup done")

#=================================================

#---< energy&mass budget module setup >---
#  call embudget_setup
#skip

#---< history module setup >---
#  call history_setup
#skip

#---< history variable module setup >---
#  call history_vars_setup
#skip

prf.PROF_rapend("Initialize", 0)
print("Initialization complete")

#skip
#--- history output at initial time
#  if ( HIST_output_step0 ) then
#     TIME_CSTEP = TIME_CSTEP - 1
#     TIME_CTIME = TIME_CTIME - TIME_DTL
#     call history_vars
#     call TIME_advance
#     call history_out
#  else
#     call TIME_report
tim.TIME_report(cldr)
#  endif

# overriding lstep_max for testing
lstep_max = tim.TIME_lstep_max 
#lstep_max = 3 

print("starting Main_Loop")
prf.PROF_setprefx("MAIN")
prf.PROF_rapstart("Main_Loop", 0)

# print("hohoa", lstep_max)
# prc.prc_mpistop(std.io_l, std.fname_log)
# import sys
# sys.exit()

#lstep_max=0

for n in range(lstep_max):

    prf.PROF_rapstart("_Atmos", 1)

    dyn.dynamics_step(prf, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, bndc, bsst, numf, vi, pre.rdtype)
    #print("dynamics_se
    #     call dynamics_step
    #     call forcing_step

    prf.PROF_rapend("_Atmos", 1)

    #prf.PROF_rapstart("_History", 1)
    #skip
    #     call history_vars

    tim.TIME_advance(cldr)

    #skip
    #--- budget monitor
    #     call embudget_monitor
    #     call history_out

    #print("running step ", n)   

    if ( n == lstep_max - 1 ):
        print("last step, start finalizing")
        pass
#        call restart_output( restart_output_basename )
#        ??? no need to be inside the loop...?

    #prf.PROF_rapend("_History", 1)


prf.PROF_rapend("Main_Loop", 0)
prf.PROF_rapreport()

#print("hoho I am rank ", prc.prc_myrank)
#prc.prc_mpifinish(std.io_l, std.fname_log)
#import sys
#sys.exit()

prc.prc_mpifinish(std.io_l, std.fname_log)

print("peacefully done:  rank ", prc.prc_myrank)


