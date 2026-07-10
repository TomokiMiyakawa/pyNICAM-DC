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

from pynicamdc.share.mod_backend import backend as bk

bk.configure(backend_name, precision)

#import numpy as np  
np = bk.np  


#import zarr
#from zarr.storage import DirectoryStore   #use Zarr v2.15 for this, not the newer Zarr v3.x
#from pynicamdc.share.mod_precision import Precision

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

from pynicamdc.nhm.share.mod_statecontainer import StateContainer
msc = StateContainer()   # model state container

# Optional float32 dtype-preservation audit of the pure kernels. Installed here
# (right after imports, before any setup/restart/ideal-init runs) so it patches
# the compute_* references BEFORE any consumer caches a jit-wrapped kernel on
# self at first call. Gated off by default; enable with PYNICAM_DTYPE_AUDIT=1.
if os.environ.get("PYNICAM_DTYPE_AUDIT", "0") != "0":
    import pynicamdc.dtype_audit as _dtype_audit
    _dtype_audit.install()

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
        #self.precision_single = cnfs['precision_single']

# can set numpy to raise exceptions on floating point errors
#np.seterr(all='raise')
#np.seterr(under='ignore')

# ---<  main program start >---
print("driver_dc.py start")

# ---< Import & Instantiate NumpyStateContainer >---
# This contains the state of the model (configuration/staticdata/variables)
# It should contain all the necessary data for the model to run, and the settings/data should always 
# be referred to as msc.xxxxxx after loaded, not through other namespaces.

# Exception: ( main, prc, prf, std, iop ) are not loaded in the container
# modules are allowed to import msc, ( prc, prf, std, iop ), and external libraries

# ---< read configuration file (toml) >---
#intoml = './case/config/nhm_driver.toml'
#setattr(msc, "backend", backend)
msc.load("bk", bk)
setattr(msc, "intoml", nhm_driver_cnf)
#msc.load("intoml", intoml)

# ---< instantiate Driver_dc class as main >---
main  = Driver_dc(msc.intoml)   
#msc.load("main", main)

##### ---< set precision >---  moved to mod_backend
#pre  = Precision(main.precision_single)  #instantiate, True if single precision, False if double precision
#msc.load("pre", pre)

# ---< MPI start >---
comm_world = prc.prc_mpistart()
if prc.prc_myrank == 0:
    is_master = True
else:
    is_master = False
#msc.load("prc", prc)

#---< STDIO setup >---
std.io_setup('pyNICAM-DC', msc.intoml)
#---< Logfile setup >---
std.io_log_setup(prc.prc_myrank, is_master)
#msc.load("std", std)

#---< profiler module setup >---
prf.PROF_setup(msc.intoml, msc.bk.ndtype)
#msc.load("prf", prf)

#--- start profiling time required for initialization --- 
prf.PROF_setprefx("INIT")
prf.PROF_rapstart("Initialize", 0)


#---< cnst module setup >---
cnst = Const(msc.bk.ndtype)
cnst.CONST_setup(msc.bk.ndtype, msc.intoml)
msc.load("cnst", cnst)

#---< calendar module setup >---
cldr.CALENDAR_setup(msc.bk.ndtype, msc.intoml)
msc.load("cldr", cldr)

# skip random module setup
#---< radom module setup >---
#  call RANDOM_setup

#---< admin module setup >---
adm.ADM_setup(msc.intoml)
msc.load("adm", adm)

#print("hio and fio skip")
#  !---< I/O module setup >---
#  call FIO_setup
#  call HIO_setup

#---< comm module setup >---
#print("COMM_setup start")
comm = Comm()
comm.COMM_setup(msc.intoml)
msc.load("comm", comm)

#---< For pole & pentagon handling >---
ppm.PNT_setup()
msc.load("ppm", ppm)


#---< grid module setup >---
grd = Grd()
grd.GRD_setup(msc.intoml, msc.cnst, msc.comm, msc.bk.ndtype)
#print("GRD_setup done") slight suspicion on the pl communication, where the original code may have a bug?
msc.load("grd", grd)

# Grid/vertical-coordinate validation dump (GRD_vz/GRD_zs/gz/gzh). Gated PYNICAM_GRD_DUMP=<path>.
_grd_dump = os.environ.get("PYNICAM_GRD_DUMP", "")
if _grd_dump:
    np.savez(f"{_grd_dump}_rank{prc.prc_myrank}.npz",
             GRD_vz=np.asarray(grd.GRD_vz), GRD_zs=np.asarray(grd.GRD_zs),
             GRD_gz=np.asarray(grd.GRD_gz), GRD_gzh=np.asarray(grd.GRD_gzh))

#---< vector operation >---
msc.load("vect", vect)

#---< GTL operation >---
gtl = Gtl()
msc.load("gtl", gtl)

#---< geometrics module setup >---
gmtr = Gmtr()
gmtr.GMTR_setup(msc.intoml, msc.cnst, msc.comm, msc.grd, msc.vect, msc.bk.ndtype)
msc.load("gmtr", gmtr)

#---< operator module setup >---
oprt = Oprt()
oprt.OPRT_setup(msc.intoml, msc.cnst, msc.gmtr, msc.bk.ndtype)
msc.load("oprt", oprt)

#---< vertical metrics module setup >---
vmtr = Vmtr()
vmtr.VMTR_setup(msc.intoml, msc.cnst, msc.comm, msc.grd, msc.gmtr, msc.oprt, msc.bk.ndtype)
msc.load("vmtr", vmtr)

#---< time module setup >---
tim = Tim()
tim.TIME_setup(msc.intoml, np.float64)  # use double precision for time (for now)
msc.load("tim", tim)

#==========================================

#---< external data module setup >---
#skip
#  call extdata_setup

#---< nhm_runconf module setup >---
rcnf = Rcnf()
rcnf.RUNCONF_setup(msc.intoml,msc.cnst)
msc.load("rcnf", rcnf)

#---< saturation module setup >---
satr.SATURATION_setup(msc.intoml,msc.cnst,msc.bk.ndtype)
msc.load("satr", satr)

#---< prognostic variable module setup >---
prgv = Prgv()
prgv.prgvar_setup(msc.intoml, msc.rcnf, msc.cnst, msc.bk.ndtype)
msc.load("prgv", prgv)

cnvv = Cnvv()
msc.load("cnvv", cnvv)

tdyn = Tdyn()
msc.load("tdyn", tdyn)

idi = Idi()
msc.load("idi", idi)

#---< restart input >---
prgv.restart_input(msc.intoml, msc.comm, msc.gtl, msc.cnst, msc.rcnf, msc.grd, msc.vmtr, msc.cnvv, msc.tdyn, msc.idi, msc.bk.ndtype)


#============================================

#----- instantiate Dynamics and Source classes
#---< dynamics module setup >---
dyn  = Dyn(msc.adm, msc.cnst, msc.rcnf, msc.bk.ndtype)
bndc = Bndc()
bsst = Bsst()
numf = Numf()
vi   = Vi()
dyn.dynamics_setup(msc.intoml, msc.comm, msc.gtl, msc.cnst, msc.grd, msc.gmtr, msc.oprt, msc.vmtr, msc.tim, msc.rcnf, msc.prgv, msc.tdyn, bndc, bsst, numf, vi, msc.bk.ndtype)
# set up of bsst, numf, vi is done within dyn.dynamics_setup
msc.load("dyn", dyn)
msc.load("bndc", bndc)
msc.load("bsst", bsst)
msc.load("numf", numf)
msc.load("vi", vi)

# env-gated basic-state dump (validation vs nicamdc bsstate). PYNICAM_BS_DUMP=<path>.
_bs_dump = os.environ.get("PYNICAM_BS_DUMP", "")
if _bs_dump:
    np.savez(f"{_bs_dump}_rank{prc.prc_myrank}.npz",
             rho_bs=np.asarray(bsst.rho_bs), pre_bs=np.asarray(bsst.pre_bs), tem_bs=np.asarray(bsst.tem_bs),
             rho_bs_pl=np.asarray(bsst.rho_bs_pl), pre_bs_pl=np.asarray(bsst.pre_bs_pl), tem_bs_pl=np.asarray(bsst.tem_bs_pl))


src   = Src(msc.cnst, msc.bk.ndtype)
msc.load("src", src)

srctr   = Srctr(msc.cnst, msc.bk.ndtype)
msc.load("srctr", srctr)

trcadv = Trcadv(msc.bk.ndtype)
msc.load("trcadv", trcadv)
#-------

#---< forcing module setup >---
frc.forcing_setup(msc.intoml, msc.rcnf, msc.bk.ndtype)
msc.load("frc", frc)

#---< io module setup >---
io = Io()
io.IO_setup(msc.intoml, msc.tim, msc.grd, msc.rcnf, msc.bk.ndtype)
msc.load("io", io)

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
tim.TIME_report(msc.cldr, np.float64)
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

#msc = StateContainer()        
                              # msc should contain all data in numpy arrays 
                              # instantiate
                              # msc should contain ( configuration settings, static data, and variable data)
                              # msc should not contain information unrelated to the model state/operation, (std, prf)  
                              # communication and process (comm, prc) is up for consideration

                              ####!!!!  Once things are loaded inside the container, they should be accessed through the container interface only. !!!!####
                              ####  Do not touch the original variables directly or through other interfaces/aliases.  ####



# env-gated history-diagnostics dump at step 0 (validation vs nicamdc history_vars).
# PYNICAM_HVAR_DUMP=<path> -> npz of ml_u/v/w/th/thv/omg/pres/tem/rho/hgt from the IC state.
_hvar_dump = os.environ.get("PYNICAM_HVAR_DUMP", "")
if _hvar_dump:
    _hv = dyn.history_vars_step(msc)
    np.savez(f"{_hvar_dump}_rank{prc.prc_myrank}.npz", **{k: np.asarray(v) for k, v in _hv.items()})

# step-0 (initial condition) snapshot to the zarr (nicamdc doout_step0). Writes the
# leading slot reserved in IO_setup; the main-loop outputs then fill the rest.
if getattr(io, "PRGout_step0", False):
    dyn.sync_prgvar_to_host(msc.prgv, msc)
    _hv0 = dyn.history_vars_step(msc) if io.PRGout_diagnostics else None
    io.IO_PRGstep(msc.tim, msc.prgv, msc.rcnf, msc.bk.ndtype, diag=_hv0)

print("starting Main_Loop")
prf.PROF_setprefx("MAIN")
prf.PROF_rapstart("Main_Loop", 0)

# Opt-in per-step PROF report (PYNICAM_PROF_PERSTEP=1): dumps each timer's
# per-step delta so the JIT-compile-heavy first step is separable from the
# steady steps. Off by default (avoids log bloat on long runs).
_prof_perstep = os.environ.get("PYNICAM_PROF_PERSTEP", "0") != "0"
if _prof_perstep:
    prf.PROF_rapsnap()   # baseline = post-init cumulative (excludes INIT_* from step deltas)

# PROFILE WINDOW (diagnostic, gated): wrap a STEADY step-range in cudaProfilerStart/Stop so
# `nsys profile --capture-range=cudaProfilerApi` captures only those steps (no compile/warmup
# contamination). PYNICAM_NSYS_STEP=<n> = first step to capture; PYNICAM_NSYS_STEP_END=<m>
# = last step (default = same as NSYS_STEP, i.e. a single step). default empty = inert.
_nsys_step = os.environ.get("PYNICAM_NSYS_STEP", "")
_cudart = None
if _nsys_step != "":
    import ctypes as _ct
    for _name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            _cudart = _ct.CDLL(_name); break
        except OSError:
            continue
    _nsys_step = int(_nsys_step)
    _nsys_step_end = int(os.environ.get("PYNICAM_NSYS_STEP_END", _nsys_step))

# STEP C (time-loop fusion, gated PYNICAM_FUSE_TIMELOOP, default off): once the fused stack is
# warm + steady (dyn._step_core built), advance the prognostic device carry in K-step CHUNKS via
# dyn.run_timeloop_chunk (eager K x self._step_core, or a jax.lax.scan over the K steps when
# PYNICAM_TIMELOOP_JIT=1 -- the actual outer-loop fusion). Warm-up steps and any output step run
# through the ordinary per-step dynamics_step; a chunk is trimmed so it never spans an output step.
_fuse_timeloop = os.environ.get("PYNICAM_FUSE_TIMELOOP", "0") != "0"
_tl_warmup = int(os.environ.get("PYNICAM_TIMELOOP_WARMUP", "3"))
_tl_chunk  = int(os.environ.get("PYNICAM_TIMELOOP_CHUNK", "1"))

# DCMIP forcing-tendency validation dump (per-step .npz, per rank). Gated PYNICAM_FRC_DUMP=<path>.
_frc_dump = os.environ.get("PYNICAM_FRC_DUMP", "")

n = 0
while n < lstep_max:
    if _cudart is not None and n == _nsys_step:
        _cudart.cudaProfilerStart()

    # Isolate the very first iteration (carries one-time JIT compilation under
    # jax) so the report shows compile-inclusive step1 separately. Steady-state
    # per-step = (Main_Loop - Main_Loop_step1) / (lstep_max - 1).
    if n == 0:
        prf.PROF_rapstart("Main_Loop_step1", 0)

    # --- fused K-step chunk? (only once warm + steady, never spanning an output step) ---
    _K = 0
    if (_fuse_timeloop and n >= _tl_warmup and getattr(dyn, "_step_core", None) is not None):
        _K = min(_tl_chunk, lstep_max - n)
        for _j in range(_K):
            _m = n + _j
            if _m >= 1 and ((_m - 1) % io.PRGout_interval == 0 or (_m - 1) % io.PRGout_interval_2d == 0):
                _K = _j     # stop the chunk just before an output step (3D or 2D)
                break
    if _K >= 1:
        prf.PROF_rapstart("_Atmos", 1)
        dyn.run_timeloop_chunk(msc, _K)   # (the profiler was started at loop top if n==_nsys_step)
        if _cudart is not None and n == _nsys_step_end:
            _cudart.cudaDeviceSynchronize()   # bound the window: finish the captured chunk's GPU work
            _cudart.cudaProfilerStop()
        prf.PROF_rapend("_Atmos", 1)
        for _j in range(_K):
            tim.TIME_advance(msc.cldr, np.float64)
        if _prof_perstep:
            prf.PROF_rapreport_step(n)
        n += _K
        continue

    # --- ordinary per-step path (warm-up, output steps, or fusion off) ---
    prf.PROF_rapstart("_Atmos", 1)

    dyn.dynamics_step(msc)  # msc should be either passed or imported in dynamics_step

    # Artificial forcing (nicamdc prg_driver-dc.f90: forcing_step follows dynamics_step
    # inside _Atmos). No-op unless AF_TYPE == 'DCMIP'. Re-derives diag from the final
    # prognostic, applies the DCMIP tendencies, writes back + halo/pole COMM.
    dyn.forcing_step(msc)

    # Validation dump: per-step DCMIP forcing tendencies (ml_af_fvx.. + sl_af_prcp) to
    # per-rank .npz, to be compared against the nicamdc golden history. Gated + inert by
    # default. Step index n is 0-based here (nicamdc history frame = n+1).
    if _frc_dump and msc.rcnf.AF_TYPE == 'DCMIP':
        np.savez(f"{_frc_dump}_step{n+1:03d}_rank{prc.prc_myrank}.npz",
                 fvx=frc.fvx, fvy=frc.fvy, fvz=frc.fvz, fe=frc.fe,
                 fq=frc.fq, precip=frc.precip)

    if _cudart is not None and n == _nsys_step_end:
        _cudart.cudaDeviceSynchronize()   # bound the window: finish the captured steps' GPU work
        _cudart.cudaProfilerStop()

    prf.PROF_rapend("_Atmos", 1)

    tim.TIME_advance(msc.cldr, np.float64)

    # Output at large-step n = 1, 1+interval, ... The 3D group (prognostics + ml_) fires on
    # PRGout_interval; the 2D group (sl_) on PRGout_interval_2d (may differ).
    _fire_3d = (n >= 1 and (n - 1) % io.PRGout_interval == 0)
    _fire_2d = (n >= 1 and (n - 1) % io.PRGout_interval_2d == 0)
    if _fire_3d or _fire_2d:
        dyn.sync_prgvar_to_host(msc.prgv, msc)   # PHASE E: materialize host PRG_var from the device stash for output (no-op when the gate is off)
        # derived history diagnostics (only the group(s) being written this step)
        _hv = (dyn.history_vars_step(msc, write_3d=_fire_3d, write_2d=_fire_2d)
               if (io.PRGout_diagnostics or _hvar_dump) else None)
        io.IO_PRGstep(msc.tim, msc.prgv, msc.rcnf, msc.bk.ndtype, diag=_hv,
                      write_3d=_fire_3d, write_2d=_fire_2d)
        if _hvar_dump:
            np.savez(f"{_hvar_dump}_step{n+1:03d}_rank{prc.prc_myrank}.npz",
                     **{k: np.asarray(v) for k, v in _hv.items()})
    # endif

    if ( n == lstep_max - 1 ):
        print("last step, start finalizing")
        pass

    if n == 0:
        prf.PROF_rapend("Main_Loop_step1", 0)

    if _prof_perstep:
        prf.PROF_rapreport_step(n)   # delta since last step -> this step's cost

    n += 1

prf.PROF_rapend("Main_Loop", 0)
prf.PROF_rapreport()

# STEP C validation hook: dump the FINAL prognostic device state to a per-rank .npy so a
# FUSE_TIMELOOP=on run can be compared bit-exact against the FUSE_TIMELOOP=off run (the gl07
# gold only emits one snapshot at n=1 = pre-warm-up, so an end-of-run off-vs-on dump is the
# real check that the K-step scan reproduces the per-step path). Gated PYNICAM_TIMELOOP_DUMP=<path>.
_tl_dump = os.environ.get("PYNICAM_TIMELOOP_DUMP", "")
if _tl_dump:
    dyn.sync_prgvar_to_host(msc.prgv, msc)
    np.save(f"{_tl_dump}_rank{prc.prc_myrank}.npy", np.asarray(msc.prgv.PRG_var))
    print(f"TIMELOOP_DUMP wrote {_tl_dump}_rank{prc.prc_myrank}.npy", flush=True)

# Peak GPU memory report (per rank). Report BOTH metrics so they aren't confused:
#  * peak_pool_bytes    = peak bytes XLA RESERVED from the device (the OOM-relevant footprint,
#                         comparable to nvidia-smi minus the CUDA context ~few hundred MB).
#  * peak_bytes_in_use  = peak LIVE tensor bytes (undercounts the true footprint -- pool reserve,
#                         workspaces and fragmentation are excluded).
# Also dump the full stats dict once (rank0) so the available keys are on record.
# Gated PYNICAM_GPU_MEM_REPORT=1.
if os.environ.get("PYNICAM_GPU_MEM_REPORT", "0") != "0" and msc.bk.type == "jax":
    try:
        for _d in msc.bk.jax.local_devices():
            _ms = _d.memory_stats() or {}
            _pool = _ms.get("peak_pool_bytes", _ms.get("peak_bytes_reserved", 0))
            _inuse = _ms.get("peak_bytes_in_use", _ms.get("bytes_in_use", 0))
            _lim = _ms.get("bytes_limit", 0)
            print(f"GPU_MEM rank{prc.prc_myrank} dev={_d.id} "
                  f"peak_pool={_pool/2**20:.1f}MiB peak_in_use={_inuse/2**20:.1f}MiB "
                  f"limit={_lim/2**20:.1f}MiB", flush=True)
            if prc.prc_myrank == 0:
                print("GPU_MEM_STATS_KEYS rank0: " +
                      ", ".join(f"{k}={v}" for k, v in sorted(_ms.items())), flush=True)
    except Exception as _e:
        print(f"GPU_MEM rank{prc.prc_myrank} unavailable: {_e}", flush=True)

if os.environ.get("PYNICAM_DTYPE_AUDIT", "0") != "0":
    _dtype_audit.report()

prc.prc_mpifinish(std.io_l, std.fname_log)

print("peacefully done:  rank ", prc.prc_myrank)
