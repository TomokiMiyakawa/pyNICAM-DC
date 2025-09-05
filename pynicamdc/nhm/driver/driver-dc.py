import numpy as np
import toml
#import zarr
#from zarr.storage import DirectoryStore   #use Zarr v2.15 for this, not the newer Zarr v3.x
import sys
import os

# os.environ["JAX_PLATFORM_NAME"] = "cpu"  # must be BEFORE jax import
# import jax
# jax.config.update("jax_enable_x64", True)
# print("Available platforms:", jax.devices())
# print("JAX default platform:", jax.default_backend())

# Global instants are instantiated in the modules when first called
# They will be singleton
#[Section1]
from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_ppmask import ppm
from pynicamdc.share.mod_prof import prf
from pynicamdc.share.mod_io_param import iop
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_vector import vect
from pynicamdc.share.mod_calendar import cldr
from pynicamdc.nhm.share.mod_chemvar import chem
from pynicamdc.nhm.share.mod_saturation import satr
from pynicamdc.nhm.forcing.mod_forcing import frc

# These classes are instantiated in this main program after the toml file is read
# Also singleton
#[Section2]
from pynicamdc.share.mod_precision import Precision
from pynicamdc.share.mod_const import Const
from pynicamdc.share.mod_comm import Comm
from pynicamdc.share.mod_gtl import Gtl
from pynicamdc.share.mod_grd import Grd
from pynicamdc.share.mod_vmtr import Vmtr
from pynicamdc.share.mod_gmtr import Gmtr
from pynicamdc.share.mod_oprt import Oprt
from pynicamdc.share.mod_time import Tim
from pynicamdc.nhm.share.mod_runconf import Rcnf
from pynicamdc.nhm.share.mod_prgvar import Prgv
from pynicamdc.nhm.share.mod_cnvvar import Cnvv
from pynicamdc.nhm.share.mod_thrmdyn import Tdyn
from pynicamdc.nhm.share.mod_ideal_init import Idi
#from mod_forcing import Frc > moved to Section1  check later if this is also better for JAX/GPU
from pynicamdc.nhm.dynamics.mod_dynamics import Dyn
from pynicamdc.nhm.share.mod_bndcnd import Bndc
from pynicamdc.nhm.share.mod_bsstate import Bsst
from pynicamdc.nhm.dynamics.mod_numfilter import Numf
from pynicamdc.nhm.dynamics.mod_vi import Vi
from pynicamdc.nhm.dynamics.mod_src import Src
from pynicamdc.nhm.dynamics.mod_src_tracer import Srctr
from pynicamdc.nhm.forcing.mod_af_trcadv import Trcadv
from pynicamdc.share.mod_io import Io
from pynicamdc.nhm.share.mod_statecontainer import NumpyStateContainer

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

print("driver_dc.py start")

# set numpy to raise exceptions on floating point errors
#np.seterr(all='raise')
#np.seterr(under='ignore')

# read configuration file (toml) and instantiate Driver_dc class
intoml = '../../case/config/nhm_driver.toml'
main  = Driver_dc(intoml)   

# instantiate classes
pre  = Precision(main.precision_single)  #True if single precision, False if double precision

cnst = Const(pre.rdtype)
comm = Comm()
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
#frc = Frc()
bndc = Bndc()
bsst = Bsst()
numf = Numf()
vi   = Vi()
io  = Io()

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
cnst.CONST_setup(pre.rdtype, intoml)

#---< calendar module setup >---
cldr.CALENDAR_setup(pre.rdtype, intoml)

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

# For pentagon handling
ppm.PNT_setup()

#---< grid module setup >---
grd.GRD_setup(intoml, cnst, comm, pre.rdtype)
#print("GRD_setup done") slight suspicion on the pl communication, where the original code may have a bug?

#---< geometrics module setup >---
gmtr.GMTR_setup(intoml, cnst, comm, grd, vect, pre.rdtype)

#---< operator module setup >---
#oprt.OPRT_setup(intoml, cnst, gmtr, pre.rdtype, pre.jdtype)
oprt.OPRT_setup(intoml, cnst, gmtr, pre.rdtype)

#---< vertical metrics module setup >---
vmtr.VMTR_setup(intoml, cnst, comm, grd, gmtr, oprt, pre.rdtype)

#---< time module setup >---
#tim.TIME_setup(intoml, pre.rdtype)
tim.TIME_setup(intoml, np.float64)  # use double precision for time

#==========================================

#---< external data module setup >---
#skip
#  call extdata_setup

#---< nhm_runconf module setup >---
rcnf.RUNCONF_setup(intoml,cnst)

#---< saturation module setup >---
satr.SATURATION_setup(intoml,cnst,pre.rdtype)

#---< prognostic variable module setup >---
prgv.prgvar_setup(intoml, rcnf, cnst, pre.rdtype)
prgv.restart_input(intoml, comm, gtl, cnst, rcnf, grd, vmtr, cnvv, tdyn, idi, pre.rdtype) 

#============================================

# instantiate Dynamics and Source classes
dyn = Dyn(cnst, rcnf, pre.rdtype)
src   = Src(cnst, pre.rdtype)
srctr   = Srctr(cnst, pre.rdtype)
trcadv = Trcadv(pre.rdtype)

#---< dynamics module setup >---
#dyn.dynamics_setup(intoml, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, frc, bndc, bsst, numf, vi, pre.rdtype)
dyn.dynamics_setup(intoml, comm, gtl, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, bndc, bsst, numf, vi, pre.rdtype)            

#---< forcing module setup >---
frc.forcing_setup(intoml, rcnf, pre.rdtype)

#---< io module setup >---
io.IO_setup(intoml, tim, grd, pre.rdtype)

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
#tim.TIME_report(cldr, pre.rdtype)
tim.TIME_report(cldr, np.float64)
#  endif

lstep_max = tim.TIME_lstep_max 

##overriding lstep_max for testing
#lstep_max = 3


# Combine everything needed for integration into state container object(s) here, and pass it into (? or should just import?) dynstep/physstep.
# This to simplify the main loops and also maintain consistency with the orginal fortran code structure for the time being.
# Make both numpy and jnp objects (and dictionary?)
# Example:
#    numpystate.variable.prog.rhog, numpystate.variable.prog.rhoge, numpystate.variable.diag.u, numpystate.variable.diag.t
#    jnpstate.variable.prog.rhog, jnpstate.variable.prog.rhoge, jnpstate.variable.diag.u, jnpstate.variable.diag.t
#    numpystate.settings.glevel ?  numpystate.static.grid ?

nsc = NumpyStateContainer()   # instantiate
                              # nsc should contain ( configuration settings, static data, and variable data)
                              # nsc should not contain information unrelated to the model state/operation, (std, prf)  
                              # communication and process (comm, prc) is up for consideration

                              ####!!!!  Once things are loaded inside the container, they should be accessed through the container interface only. !!!!####
                              ####  Do not touch the original variables directly or through other interfaces/aliases.  ####

#nsc.load(comm, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn,  #frc, bndc, cnvv, bsst, numf, vi, src, srctr, trcadv, pre.rdtype) 
nsc.load(adm, comm, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, bndc, cnvv, bsst, numf, vi, src, srctr, trcadv, pre) 

### On 2nd thought, seperating config, static, variables might be needless
#nsc.load_config(rcnf, adm)   # this loads the configuration settings into the container, like nsc.rcnf, nsc.adm
#nsc.load_static()            # this loads static array data into the container (e.g. grids, coef)
#nsc.load_variable()          # this loads variable data into the container (surely prognostic variables. what about diagnostic variables?)
                             # (it makes sense to have diagnostic variables here if they are needed for calculating their tendencies for output) ... make it optional?
                             # load prgv.DIAG_var for the time being.

print("starting Main_Loop")
prf.PROF_setprefx("MAIN")
prf.PROF_rapstart("Main_Loop", 0)

for n in range(lstep_max):

    prf.PROF_rapstart("_Atmos", 1)

    #dyn.dynamics_step()  # nsc should be imported in dynamics_step

    dyn.dynamics_step(comm, cnst, grd, gmtr, oprt, 
                      vmtr, tim, rcnf, prgv, tdyn,  #frc,
                      bndc, cnvv, bsst, numf, vi, src, 
                      srctr, trcadv, pre.rdtype)

    # the items passed to dynamics_step/physics_step/surface_step should be packed into nsc/jsc except for things like std, prf


    #phys.physics_step(..., ..., )


    prf.PROF_rapend("_Atmos", 1)

    #prf.PROF_rapstart("_History", 1)
    #skip
    #     call history_vars


    #tim.TIME_advance(cldr, pre.rdtype)
    tim.TIME_advance(cldr, np.float64)

    #skip
    #--- budget monitor
    #     call embudget_monitor
    #     call history_out

    # Output
    if n % io.PRGout_interval == 1:
        io.IO_PRGstep(tim, prgv, rcnf, pre.rdtype)
    # endif

     
    if ( n == lstep_max - 1 ):
        print("last step, start finalizing")
        pass
    #   call restart_output( restart_output_basename )
    #   no need to be inside the loop...?
    #endif

    #prf.PROF_rapend("_History", 1)

prf.PROF_rapend("Main_Loop", 0)
prf.PROF_rapreport()

prc.prc_mpifinish(std.io_l, std.fname_log)

print("peacefully done:  rank ", prc.prc_myrank)
