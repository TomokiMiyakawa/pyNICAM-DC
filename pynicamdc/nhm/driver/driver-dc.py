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

# Global instants are instantiated in the modules when first called
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

# These classes are instantiated in this main program after the toml file is read
# Also singleton
from mod_precision import Precision
from mod_const import Const
from mod_comm import Comm
from mod_gtl import Gtl
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
from mod_src import Src
from mod_src_tracer import Srctr

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

# set numpy to raise exceptions on floating point errors
np.seterr(all='raise')


# read configuration file (toml) and instantiate Driver_dc class
intoml = '../../case/config/nhm_driver.toml'
main  = Driver_dc(intoml)   

# instantiate classes
pre  = Precision(main.precision_single)  #True if single precision (not ready yet), False if double precision
comm = Comm(pre.rdtype)
cnst = Const(main.precision_single)
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

#---< Logfile setup >---
std.io_log_setup(prc.prc_myrank, is_master)

#---< profiler module setup >---
prf.PROF_setup(intoml, pre.rdtype)

prf.PROF_setprefx("INIT")
prf.PROF_rapstart("Initialize", 0)

#---< cnst module setup >---
cnst.CONST_setup(intoml)

#---< calendar module setup >---
cldr.CALENDAR_setup(intoml)

# skip random module setup
#---< radom module setup >---
#  call RANDOM_setup

#---< admin module setup >---
adm.ADM_setup(intoml)

#print("hio and fio skip")
#  !---< I/O module setup >---
#  call FIO_setup
#  call HIO_setup

#print("COMM_setup start")
comm.COMM_setup(intoml)

#---< grid module setup >---
grd.GRD_setup(intoml, cnst, comm, pre.rdtype)
#print("GRD_setup done") slight suspicion on the pl communication, where the original code may have a bug?

#---< geometrics module setup >---
gmtr.GMTR_setup(intoml, cnst, comm, grd, vect, pre.rdtype)

#---< operator module setup >---
oprt.OPRT_setup(intoml, gmtr, pre.rdtype)

#---< vertical metrics module setup >---
vmtr.VMTR_setup(intoml, cnst, comm, grd, gmtr, oprt, pre.rdtype)

#---< time module setup >---
tim.TIME_setup(intoml)

#==========================================

#---< external data module setup >---
#skip
#  call extdata_setup

#---< nhm_runconf module setup >---
rcnf.RUNCONF_setup(intoml,cnst)

#---< saturation module setup >---
satr.SATURATION_setup(intoml,cnst)

#---< prognostic variable module setup >---
prgv.prgvar_setup(intoml, rcnf, pre.rdtype)
prgv.restart_input(intoml, comm, gtl, cnst, rcnf, grd, vmtr, cnvv, tdyn, idi, pre.rdtype) 

#============================================

# instantiate Dynamics and Source classes
dyn = Dyn(cnst, rcnf, pre.rdtype)
src   = Src(cnst, pre.rdtype)
srctr   = Srctr(cnst, pre.rdtype)

#---< dynamics module setup >---
dyn.dynamics_setup(intoml, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, bndc, bsst, numf, vi, pre.rdtype)

#---< forcing module setup >---
frc.forcing_setup(intoml, rcnf, pre.rdtype)

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

lstep_max = tim.TIME_lstep_max 
##overriding lstep_max for testing
#lstep_max = 1

print("starting Main_Loop")
prf.PROF_setprefx("MAIN")
prf.PROF_rapstart("Main_Loop", 0)

for n in range(lstep_max):

    prf.PROF_rapstart("_Atmos", 1)

    dyn.dynamics_step(comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, bndc, cnvv, bsst, numf, vi, src, srctr, pre.rdtype)

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

prc.prc_mpifinish(std.io_l, std.fname_log)

print("peacefully done:  rank ", prc.prc_myrank)
