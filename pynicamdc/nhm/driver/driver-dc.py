import os
import sys
import argparse
import toml


parser = argparse.ArgumentParser()
parser.add_argument(
    "--driver-setting",
    default="./driversettings.toml",
)
args = parser.parse_args()

drv = toml.load(args.driver_setting)["driver"]
backend_name    = drv["backend"]
precision       = drv["precision"]
nhm_driver_cnf = drv["nhm_driver_cnf"]
#backend_name = drv.get("backend", "numpy")
#precision = drv.get("precision", "float64")

import numpy as np # temporary

#import zarr
#from zarr.storage import DirectoryStore   #use Zarr v2.15 for this, not the newer Zarr v3.x


from pynicamdc.share.mod_precision import Precision
from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_prof import prf
from pynicamdc.share.mod_const import Const
from pynicamdc.share.mod_calendar import cldr
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_comm import Comm
from pynicamdc.share.mod_ppmask import ppm
from pynicamdc.share.mod_grd import Grd
from pynicamdc.share.mod_vector import vect
from pynicamdc.share.mod_gtl import Gtl
from pynicamdc.share.mod_gmtr import Gmtr
from pynicamdc.share.mod_oprt import Oprt
from pynicamdc.share.mod_vmtr import Vmtr
from pynicamdc.share.mod_time import Tim
from pynicamdc.nhm.share.mod_runconf import Rcnf
from pynicamdc.nhm.share.mod_saturation import satr
from pynicamdc.nhm.share.mod_prgvar import Prgv
from pynicamdc.nhm.share.mod_cnvvar import Cnvv
from pynicamdc.nhm.share.mod_thrmdyn import Tdyn
from pynicamdc.nhm.share.mod_ideal_init import Idi
from pynicamdc.nhm.dynamics.mod_dynamics import Dyn
from pynicamdc.nhm.share.mod_bndcnd import Bndc
from pynicamdc.nhm.share.mod_bsstate import Bsst
from pynicamdc.nhm.dynamics.mod_numfilter import Numf
from pynicamdc.nhm.dynamics.mod_vi import Vi
from pynicamdc.nhm.dynamics.mod_src import Src
from pynicamdc.nhm.dynamics.mod_src_tracer import Srctr
from pynicamdc.nhm.forcing.mod_af_trcadv import Trcadv
from pynicamdc.nhm.forcing.mod_forcing import frc
from pynicamdc.share.mod_io import Io

from pynicamdc.nhm.share.mod_statecontainer import NumpyStateContainer
nsc = NumpyStateContainer()

# os.environ["JAX_PLATFORM_NAME"] = "cpu"  # must be BEFORE jax import
# import jax
# jax.config.update("jax_enable_x64", True)
# print("Available platforms:", jax.devices())
# print("JAX default platform:", jax.default_backend())

# Global instants are instantiated in the modules when first called
# They will be singleton
#[Section1]
# not used?
#from pynicamdc.nhm.share.mod_chemvar import chem

# These classes are instantiated in this main program after the toml file is read
# Also singleton
#[Section2]
# none left

class Driver_dc:

    def __init__(self,fname_in):

        # Load configurations from TOML file
        cnfs = toml.load(fname_in)['admparam']
        #self.glevel = cnfs['glevel']
        #self.rlevel = cnfs['rlevel']
        #self.vlayer = cnfs['vlayer']
        #self.rgnmngfname = cnfs['rgnmngfname']
        self.precision_single = cnfs['precision_single']

# can set numpy to raise exceptions on floating point errors
#np.seterr(all='raise')
#np.seterr(under='ignore')

# ---<  main program start >---
print("driver_dc.py start")

# ---< Import & Instantiate NumpyStateContainer >---
# This contains the state of the model (configuration/staticdata/variables)
# It should contain all the necessary data for the model to run, and the settings/data should always 
# be referred to as nsc.xxxxxx after loaded, not through other namespaces.

# Exception: ( main, prc, prf, std, iop ) are not loaded in the container
# modules are allowed to import nsc, ( prc, prf, std, iop ), and external libraries



# ---< read configuration file (toml) >---
#intoml = './case/config/nhm_driver.toml'
setattr(nsc, "intoml", nhm_driver_cnf)
#nsc.load("intoml", intoml)

# ---< instantiate Driver_dc class as main >---
main  = Driver_dc(nsc.intoml)   
#nsc.load("main", main)

# ---< set precision >---
pre  = Precision(main.precision_single)  #instantiate, True if single precision, False if double precision
nsc.load("pre", pre)

# ---< MPI start >---
comm_world = prc.prc_mpistart()
if prc.prc_myrank == 0:
    is_master = True
else:
    is_master = False
#nsc.load("prc", prc)

#---< STDIO setup >---
std.io_setup('pyNICAM-DC', nsc.intoml)
#---< Logfile setup >---
std.io_log_setup(prc.prc_myrank, is_master)
#nsc.load("std", std)

#---< profiler module setup >---
prf.PROF_setup(nsc.intoml, nsc.pre.rdtype)
#nsc.load("prf", prf)

#--- start profiling time required for initialization --- 
prf.PROF_setprefx("INIT")
prf.PROF_rapstart("Initialize", 0)


#---< cnst module setup >---
cnst = Const(nsc.pre.rdtype)
cnst.CONST_setup(nsc.pre.rdtype, nsc.intoml)
nsc.load("cnst", cnst)

#---< calendar module setup >---
cldr.CALENDAR_setup(nsc.pre.rdtype, nsc.intoml)
nsc.load("cldr", cldr)

# skip random module setup
#---< radom module setup >---
#  call RANDOM_setup

#---< admin module setup >---
adm.ADM_setup(nsc.intoml)
nsc.load("adm", adm)

#print("hio and fio skip")
#  !---< I/O module setup >---
#  call FIO_setup
#  call HIO_setup

#---< comm module setup >---
#print("COMM_setup start")
comm = Comm()
comm.COMM_setup(nsc.intoml)
nsc.load("comm", comm)

#---< For pole & pentagon handling >---
ppm.PNT_setup()
nsc.load("ppm", ppm)


#---< grid module setup >---
grd = Grd()
grd.GRD_setup(nsc.intoml, nsc.cnst, nsc.comm, nsc.pre.rdtype)
#print("GRD_setup done") slight suspicion on the pl communication, where the original code may have a bug?
nsc.load("grd", grd)

#---< vector operation >---
nsc.load("vect", vect)

#---< GTL operation >---
gtl = Gtl()
nsc.load("gtl", gtl)

#---< geometrics module setup >---
gmtr = Gmtr()
gmtr.GMTR_setup(nsc.intoml, nsc.cnst, nsc.comm, nsc.grd, nsc.vect, nsc.pre.rdtype)
nsc.load("gmtr", gmtr)

#---< operator module setup >---
oprt = Oprt()
oprt.OPRT_setup(nsc.intoml, nsc.cnst, nsc.gmtr, nsc.pre.rdtype)
nsc.load("oprt", oprt)

#---< vertical metrics module setup >---
vmtr = Vmtr()
vmtr.VMTR_setup(nsc.intoml, nsc.cnst, nsc.comm, nsc.grd, nsc.gmtr, nsc.oprt, nsc.pre.rdtype)
nsc.load("vmtr", vmtr)

#---< time module setup >---
tim = Tim()
tim.TIME_setup(nsc.intoml, np.float64)  # use double precision for time (for now)
nsc.load("tim", tim)

#==========================================

#---< external data module setup >---
#skip
#  call extdata_setup

#---< nhm_runconf module setup >---
rcnf = Rcnf()
rcnf.RUNCONF_setup(nsc.intoml,nsc.cnst)
nsc.load("rcnf", rcnf)

#---< saturation module setup >---
satr.SATURATION_setup(nsc.intoml,nsc.cnst,nsc.pre.rdtype)
nsc.load("satr", satr)

#---< prognostic variable module setup >---
prgv = Prgv()
prgv.prgvar_setup(nsc.intoml, nsc.rcnf, nsc.cnst, nsc.pre.rdtype)
nsc.load("prgv", prgv)

cnvv = Cnvv()
nsc.load("cnvv", cnvv)

tdyn = Tdyn()
nsc.load("tdyn", tdyn)

idi = Idi()
nsc.load("idi", idi)

#---< restart input >---
prgv.restart_input(nsc.intoml, nsc.comm, nsc.gtl, nsc.cnst, nsc.rcnf, nsc.grd, nsc.vmtr, nsc.cnvv, nsc.tdyn, nsc.idi, nsc.pre.rdtype)


#============================================

#----- instantiate Dynamics and Source classes
#---< dynamics module setup >---
dyn  = Dyn(nsc.adm, nsc.cnst, nsc.rcnf, nsc.pre.rdtype)
bndc = Bndc()
bsst = Bsst()
numf = Numf()
vi   = Vi()
dyn.dynamics_setup(nsc.intoml, nsc.comm, nsc.gtl, nsc.cnst, nsc.grd, nsc.gmtr, nsc.oprt, nsc.vmtr, nsc.tim, nsc.rcnf, nsc.prgv, nsc.tdyn, bndc, bsst, numf, vi, nsc.pre.rdtype)
# set up of bsst, numf, vi is done within dyn.dynamics_setup
nsc.load("dyn", dyn)
nsc.load("bndc", bndc)
nsc.load("bsst", bsst)
nsc.load("numf", numf)
nsc.load("vi", vi)


src   = Src(nsc.cnst, nsc.pre.rdtype)
nsc.load("src", src)

srctr   = Srctr(nsc.cnst, nsc.pre.rdtype)
nsc.load("srctr", srctr)

trcadv = Trcadv(nsc.pre.rdtype)
nsc.load("trcadv", trcadv)
#-------

#---< forcing module setup >---
frc.forcing_setup(nsc.intoml, nsc.rcnf, nsc.pre.rdtype)
nsc.load("frc", frc)

#---< io module setup >---
io = Io()
io.IO_setup(nsc.intoml, nsc.tim, nsc.grd, nsc.pre.rdtype)
nsc.load("io", io)

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
tim.TIME_report(nsc.cldr, np.float64)
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

#nsc = NumpyStateContainer()   # instantiate
                              # nsc should contain ( configuration settings, static data, and variable data)
                              # nsc should not contain information unrelated to the model state/operation, (std, prf)  
                              # communication and process (comm, prc) is up for consideration

                              ####!!!!  Once things are loaded inside the container, they should be accessed through the container interface only. !!!!####
                              ####  Do not touch the original variables directly or through other interfaces/aliases.  ####

##nsc.load(comm, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn,  #frc, bndc, cnvv, bsst, numf, vi, src, srctr, trcadv, pre.rdtype) 
#nsc.load(adm, comm, cnst, grd, gmtr, oprt, vmtr, tim, rcnf, prgv, tdyn, bndc, cnvv, bsst, numf, vi, src, srctr, trcadv, pre) 

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

    dyn.dynamics_step(nsc)  # nsc should be either passed or imported in dynamics_step

    # dyn.dynamics_step(comm, cnst, grd, gmtr, oprt, 
    #                   vmtr, tim, rcnf, prgv, tdyn,  #frc,
    #                   bndc, cnvv, bsst, numf, vi, src, 
    #                   srctr, trcadv, pre.rdtype)

    # the items passed to dynamics_step/physics_step/surface_step should be packed into nsc/jsc except for things like std, prf


    #phys.physics_step(..., ..., )


    prf.PROF_rapend("_Atmos", 1)

    #prf.PROF_rapstart("_History", 1)
    #skip
    #     call history_vars


    #tim.TIME_advance(cldr, pre.rdtype)
    tim.TIME_advance(nsc.cldr, np.float64)

    #skip
    #--- budget monitor
    #     call embudget_monitor
    #     call history_out

    # Output
    if n % io.PRGout_interval == 1:
        io.IO_PRGstep(nsc.tim, nsc.prgv, nsc.rcnf, nsc.pre.rdtype)
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
