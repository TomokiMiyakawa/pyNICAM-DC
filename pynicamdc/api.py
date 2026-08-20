"""Object API for pyNICAM-DC.

    from pynicamdc import pyNICAM

    nicam = pyNICAM("driversettings.toml")
    nicam.initialize()
    nicam = nicam.run(10)        # advances state and time
    nicam.write()
    nicam.finalize()

The objects this builds, and the order it builds them in, are the ones
driver-dc.py has always used -- this module IS that driver with its phases named.
driver-dc.py now calls it, so there is one startup sequence, not two.

`run()` returns `self`, so `nicam = nicam.run(...)` reads as a pipeline while
costing nothing: the prognostic array is allocated once (prgvar_setup) and each
step overwrites it in place (`prgv.PRG_var[...] = ...`). No state is copied,
moved, or rebuilt by the assignment. `nicam.run(...)` with the return value
dropped does exactly the same thing.

IMPORT ORDER IS LOAD-BEARING -- see share/comm_mode.py. backend, precision and
the mpi-vs-serial decision must be fixed BEFORE the first import of mod_process,
which decides mpi-vs-serial once, at import. Hence the split:

  * __init__      reads the settings and configures the backend. Imports NOTHING
                  from the model.
  * initialize()  imports the model, then builds it.

So: never import a model module at the top of this file, and never import one
before constructing the object. __init__ raises if the decision was already made
elsewhere rather than letting a wrong one stand -- a mis-set comm mode turns
`srun -n 64` into 64 independent rank-0 processes.

Two consequences the design accepts:

  * ONE instance per process. adm/prc/std/prf/cldr/satr/frc/embudget are
    module-level singletons, so a second pyNICAM would share their state.
  * backend and precision are fixed at construction.
"""

import copy
import os
import sys
import tempfile

import toml


def _deep_merge(base, over):
    """`base` with `over` applied, recursing into nested tables. Not in place."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class pyNICAM:
    """One model instance: configure -> initialize -> run/write -> finalize."""

    def __init__(self, path_to_config=None, *, backend=None, precision=None, comm=None):
        """Fix backend/precision/comm. Imports no model module (see module docstring).

        `path_to_config` is either

          * a driver-settings toml -- a [driver] table giving backend, precision,
            optional comm, and nhm_driver_cnf (the model config to use), or
          * a model config toml itself (the one with [admparam]), in which case
            backend/precision/comm come from the keyword arguments.

        The keyword arguments override the [driver] table when both are given, so a
        caller can reuse a settings file and switch backend from Python.
        """
        if path_to_config is None:
            raise ValueError("pyNICAM(path_to_config=...) is required")
        if not os.path.exists(path_to_config):
            raise FileNotFoundError(f"config file not found: {path_to_config}")

        cnfs = toml.load(path_to_config)

        if "driver" in cnfs:
            drv = cnfs["driver"]
            self.backend = backend if backend is not None else drv["backend"]
            self.precision = precision if precision is not None else drv["precision"]
            self.comm = comm if comm is not None else drv.get("comm", "auto")
            self.config_path = drv["nhm_driver_cnf"]
            self.settings_path = path_to_config
        elif "admparam" in cnfs:
            # a model config was handed over directly; no [driver] table to read
            self.backend = backend if backend is not None else "numpy"
            self.precision = precision if precision is not None else "float64"
            self.comm = comm if comm is not None else "auto"
            self.config_path = path_to_config
            self.settings_path = None
        else:
            raise ValueError(
                f"{path_to_config} has neither a [driver] table (driver settings) nor "
                "an [admparam] table (model config)")

        # --- the mpi-vs-serial decision, before anything can import mod_process ---
        from pynicamdc.share import comm_mode
        if "pynicamdc.share.mod_process" in sys.modules:
            # Already decided at that import and cannot be revisited. Silence here
            # would mean running the wrong mode; say so unless it happens to agree.
            selected = (comm_mode.SELECTED or "").split()[0]
            if self.comm != "auto" and selected != self.comm:
                raise RuntimeError(
                    f"comm={self.comm!r} was requested but mod_process was already "
                    f"imported and selected {comm_mode.SELECTED!r}. The choice is made "
                    "once, at that import -- construct pyNICAM before importing any "
                    "pynicamdc model module.")
        comm_mode.set_mode(self.comm)

        # --- backend + working precision, likewise before the model imports ---
        from pynicamdc.share.mod_backend import backend as bk
        if bk.type is not None and (bk.type != self.backend or bk.ndtype.__name__ != self.precision):
            raise RuntimeError(
                f"backend is already configured as ({bk.type}, {bk.ndtype.__name__}) and "
                f"cannot be reconfigured to ({self.backend}, {self.precision}) in the same "
                "process -- kernels and device arrays are already bound to it.")
        bk.configure(self.backend, self.precision)
        self.bk = bk
        self.np = bk.np

        self.msc = None
        self._n = 0             # completed steps; == tim.TIME_cstep after initialize
        self._initialized = False
        self._finalized = False
        self._tmpdir = None

    # ------------------------------------------------------------------ helpers

    def _notice(self, msg):
        """Put `msg` in the log and on master's stdout, so it cannot be missed."""
        from pynicamdc.share.mod_process import prc
        from pynicamdc.share.mod_stdio import std
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(msg, file=log_file, flush=True)
        if prc.prc_ismaster:
            print(msg, flush=True)

    def _resolve_config(self, parameters):
        """Path to the model config to hand the setup routines.

        Without `parameters` that is the config as given. With it, the overrides are
        merged in and the result written to a temporary toml -- because the config
        travels as a PATH: ~15 setup routines re-open it themselves, so there is no
        single in-memory dict to patch (a proper config-resolution layer is the
        follow-up; this is the non-invasive form of it).

        Paths inside the config stay relative to the CURRENT DIRECTORY, exactly as
        before -- the temporary file's own location is never used to resolve them.
        """
        if not parameters:
            return self.config_path

        merged = _deep_merge(toml.load(self.config_path), parameters)

        # admparam.rgnmngfname names the file [rgnmngparam] is read from, and in the
        # shipped configs it points back at the config itself. Left alone it would
        # point at the ORIGINAL, so an override of [rgnmngparam] would be read from
        # the un-merged file and silently do nothing. Retarget only that self-
        # reference; a config that genuinely keeps its region table elsewhere is
        # untouched.
        # realpath, not abspath: the shipped layout reaches the config through a
        # `case` symlink, so the two spellings differ until the link is resolved.
        rgn = merged.get("admparam", {}).get("rgnmngfname")
        self_ref = rgn is not None and os.path.realpath(rgn) == os.path.realpath(self.config_path)

        # Every rank writes its own copy: same content, no barrier needed, and no
        # two ranks racing on one path.
        self._tmpdir = tempfile.mkdtemp(prefix="pynicam_cnf_")
        path = os.path.join(self._tmpdir, os.path.basename(self.config_path))
        if self_ref:
            merged["admparam"]["rgnmngfname"] = path
        with open(path, "w") as f:
            toml.dump(merged, f)
        return path

    # --------------------------------------------------------------- properties

    @property
    def rank(self):
        """This process's rank. Safe from construction on -- __init__ has already
        fixed the mpi-vs-serial mode, so importing mod_process here cannot get it
        wrong (which is why callers should ask the instance rather than import prc
        themselves at the top of a script)."""
        from pynicamdc.share.mod_process import prc
        return prc.prc_myrank

    @property
    def nprocs(self):
        """Number of ranks."""
        from pynicamdc.share.mod_process import prc
        return prc.prc_nprocs

    @property
    def step(self):
        """Completed large steps."""
        return self._n

    @property
    def lstep_max(self):
        """Total large steps this configuration runs (timeparam.lstep_max)."""
        return self.msc.tim.TIME_lstep_max

    @property
    def time(self):
        """Model clock [s]."""
        return self.msc.tim.TIME_ctime

    @property
    def chunk(self):
        """The resolved fusion chunk length K (1 unless PYNICAM_TIMELOOP_CHUNK
        raises the cap). A ``run(n)`` whose ``n`` is not a multiple of this leaves
        a tail that runs per-step (correct, just unfused) -- callers that split
        the loop can align on it: ``nicam.run(nicam.chunk * m)``."""
        return self._tl_chunk

    # ------------------------------------------------------------- phase: setup

    def initialize(self, parameters=None):
        """Build the model: grid, metrics, operators, initial/restart state, output.

        `parameters` is a nested dict overlaid on the config file, keyed by its
        tables, e.g. `{"timeparam": {"lstep_max": 24}, "ioparam": {"PRGout_name":
        "run1.zarr"}}`. Keys not present in the file are added; the file is not
        modified. backend/precision/comm are NOT settable here -- they are fixed at
        construction (see the module docstring).
        """
        if self._initialized:
            raise RuntimeError("initialize() has already been called on this instance")

        # ---- model imports. Everything above ran without them, on purpose. ----
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
        from pynicamdc.nhm.share.mod_embudget import embudget
        from pynicamdc.nhm.dynamics.mod_numfilter import Numf
        from pynicamdc.nhm.dynamics.mod_vi import Vi
        from pynicamdc.nhm.dynamics.mod_src import Src
        from pynicamdc.nhm.dynamics.mod_src_tracer import Srctr
        from pynicamdc.nhm.forcing.mod_af_trcadv import Trcadv
        from pynicamdc.nhm.forcing.mod_forcing import frc
        from pynicamdc.share.mod_io import Io
        from pynicamdc.nhm.share.mod_statecontainer import StateContainer

        bk = self.bk
        np = self.np

        # Optional float32 dtype-preservation audit of the pure kernels. Installed
        # here (after imports, before any setup/restart/ideal-init runs) so it
        # patches the compute_* references BEFORE any consumer caches a jit-wrapped
        # kernel on self at first call. Gated off by default (PYNICAM_DTYPE_AUDIT=1).
        self._dtype_audit = None
        if os.environ.get("PYNICAM_DTYPE_AUDIT", "0") != "0":
            import pynicamdc.dtype_audit as _dtype_audit
            _dtype_audit.install()
            self._dtype_audit = _dtype_audit

        # ---< model state container >---
        # Holds configuration/staticdata/variables. Once loaded here, data is read as
        # msc.xxxxxx and never through another namespace or alias.
        # Exception: ( prc, prf, std, iop ) are deliberately NOT in the container.
        msc = StateContainer()
        self.msc = msc

        msc.load("bk", bk)
        setattr(msc, "intoml", self._resolve_config(parameters))

        # ---< MPI start >---
        self.comm_world = prc.prc_mpistart()
        is_master = prc.prc_myrank == 0

        #---< STDIO setup >---
        std.io_setup('pyNICAM-DC', msc.intoml)
        #---< Logfile setup >---
        std.io_log_setup(prc.prc_myrank, is_master)

        #---< profiler module setup >---
        prf.PROF_setup(msc.intoml, msc.bk.ndtype)

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

        #---< admin module setup >---
        adm.ADM_setup(msc.intoml)
        msc.load("adm", adm)

        #---< comm module setup >---
        comm = Comm()
        comm.COMM_setup(msc.intoml)
        msc.load("comm", comm)

        #---< For pole & pentagon handling >---
        ppm.PNT_setup()
        msc.load("ppm", ppm)

        #---< grid module setup >---
        grd = Grd()
        grd.GRD_setup(msc.intoml, msc.cnst, msc.comm, msc.bk.ndtype)
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

        #---< nhm_runconf module setup >---
        rcnf = Rcnf()
        rcnf.RUNCONF_setup(msc.intoml, msc.cnst)
        msc.load("rcnf", rcnf)

        #---< saturation module setup >---
        satr.SATURATION_setup(msc.intoml, msc.cnst, msc.bk.ndtype)
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
        prgv.restart_input(msc.intoml, msc.comm, msc.gtl, msc.cnst, msc.rcnf, msc.grd,
                           msc.vmtr, msc.cnvv, msc.tdyn, msc.idi, msc.bk.ndtype)

        # env-gated ADVANCED (fio) restart write-back, for validation. PYNICAM_RESTART_OUT=<basename.pe>.
        _r_out = os.environ.get("PYNICAM_RESTART_OUT", "")
        if _r_out:
            prgv.restart_output(_r_out, msc.rcnf, msc.bk.ndtype)

        #---< dynamics module setup >---
        dyn = Dyn(msc.adm, msc.cnst, msc.rcnf, msc.bk.ndtype)
        bndc = Bndc()
        bsst = Bsst()
        numf = Numf()
        vi = Vi()
        dyn.dynamics_setup(msc.intoml, msc.comm, msc.gtl, msc.cnst, msc.grd, msc.gmtr,
                           msc.oprt, msc.vmtr, msc.tim, msc.rcnf, msc.prgv, msc.tdyn,
                           bndc, bsst, numf, vi, msc.bk, msc.bk.ndtype, msc)
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
                     rho_bs=np.asarray(bsst.rho_bs), pre_bs=np.asarray(bsst.pre_bs),
                     tem_bs=np.asarray(bsst.tem_bs), rho_bs_pl=np.asarray(bsst.rho_bs_pl),
                     pre_bs_pl=np.asarray(bsst.pre_bs_pl), tem_bs_pl=np.asarray(bsst.tem_bs_pl))

        src = Src(msc.cnst, msc.bk.ndtype)
        msc.load("src", src)

        srctr = Srctr(msc.cnst, msc.bk.ndtype)
        msc.load("srctr", srctr)

        trcadv = Trcadv(msc.bk.ndtype)
        msc.load("trcadv", trcadv)

        #---< forcing module setup >---
        frc.forcing_setup(msc.intoml, msc.rcnf, msc.bk.ndtype)
        msc.load("frc", frc)

        #---< io module setup >---
        io = Io()
        io.IO_setup(msc.intoml, msc.tim, msc.grd, msc.rcnf, msc.bk.ndtype)
        msc.load("io", io)

        #---< energy&mass budget module setup >---
        embudget.embudget_setup(msc.intoml, msc)
        msc.load("embudget", embudget)

        prf.PROF_rapend("Initialize", 0)
        print("Initialization complete")

        tim.TIME_report(msc.cldr, np.float64)

        self._n = 0
        self._initialized = True
        self._resolve_loop_options()

        # env-gated history-diagnostics dump at step 0 (validation vs nicamdc
        # history_vars). PYNICAM_HVAR_DUMP=<path> -> npz of the IC state.
        if self._hvar_dump:
            _hv = dyn.history_vars_step(msc)
            np.savez(f"{self._hvar_dump}_rank{prc.prc_myrank}.npz",
                     **{k: np.asarray(v) for k, v in _hv.items()})

        # step-0 (initial condition) snapshot to the zarr (nicamdc doout_step0). Writes
        # the leading slot reserved in IO_setup; the main-loop outputs then fill the
        # rest. Deliberately NOT routed through write(): the PROF prefix is still INIT
        # here, so write()'s _Out_* timers would land in the wrong group.
        if getattr(io, "PRGout_step0", False):
            dyn.sync_prgvar_to_host(msc.prgv, msc)
            _hv0 = dyn.history_vars_step(msc) if io.PRGout_diagnostics else None
            io.IO_PRGstep(msc.tim, msc.prgv, msc.rcnf, msc.bk.ndtype, diag=_hv0)

        return self

    def _resolve_loop_options(self):
        """Read the run-loop env gates once, at setup, not per step."""
        msc = self.msc

        # Opt-in per-step PROF report (PYNICAM_PROFILE tag perstep): dumps each timer's
        # per-step delta so the JIT-compile-heavy first step is separable from the
        # steady steps. Off by default (avoids log bloat on long runs).
        self._prof_perstep = msc.bk.profile("perstep")

        # PROFILE WINDOW (diagnostic, gated): wrap a STEADY step-range in
        # cudaProfilerStart/Stop so `nsys profile --capture-range=cudaProfilerApi`
        # captures only those steps (no compile/warmup contamination).
        # PYNICAM_NSYS_STEP=<n> = first step to capture; PYNICAM_NSYS_STEP_END=<m> =
        # last step (default = same as NSYS_STEP, i.e. a single step).
        _nsys_step = os.environ.get("PYNICAM_NSYS_STEP", "")
        self._cudart = None
        self._nsys_step = None
        self._nsys_step_end = None
        if _nsys_step != "":
            import ctypes as _ct
            for _name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
                try:
                    self._cudart = _ct.CDLL(_name); break
                except OSError:
                    continue
            self._nsys_step = int(_nsys_step)
            self._nsys_step_end = int(os.environ.get("PYNICAM_NSYS_STEP_END", _nsys_step))

        # STEP C (time-loop fusion, gated PYNICAM_FUSE_TIMELOOP, default off): once the
        # fused stack is warm + steady (dyn._step_core built), advance the prognostic
        # device carry in K-step CHUNKS via dyn.run_timeloop_chunk (eager K x
        # self._step_core, or a jax.lax.scan over the K steps when
        # PYNICAM_TIMELOOP_JIT=1 -- the actual outer-loop fusion). Warm-up steps and any
        # output step run through the ordinary per-step dynamics_step; a chunk is
        # trimmed so it never spans an output step.
        self._fuse_timeloop = os.environ.get("PYNICAM_FUSE_TIMELOOP", "0") != "0"

        # S2(a) (FUSION_SCHEDULE_PLAN): K and warm-up are RESOLVED here, once, from
        # the output schedule. PYNICAM_TIMELOOP_CHUNK is the *cap* (default 1, the
        # measured-neutral default: the resolver returns K=1 unless raised);
        # PYNICAM_TIMELOOP_WARMUP is a development override, else warm-up = K.
        #   K = 1                                  if cap == 1
        #     = max{d <= cap : gcd(intervals) % d == 0}   if cap > 1
        #     = cap                                if no interval fires in the run
        # K | gcd makes every boundary gap tile exactly; warm-up = K puts the phase
        # on a multiple of K, so every boundary lands on a chunk end. Together with
        # the no-short-chunks rule in run(), K is single-valued by construction:
        # one compiled chunk graph for the whole run.
        from pynicamdc.share.output_schedule import resolve_chunk
        _cap = int(os.environ.get("PYNICAM_TIMELOOP_CHUNK", "1"))
        _lstep = msc.tim.TIME_lstep_max
        _cands = [msc.io.PRGout_interval, msc.io.PRGout_interval_2d]
        if msc.embudget.MNT_ON:
            _cands.append(msc.embudget.MNT_INTV)
        _active = [iv for iv in _cands if iv and 0 < iv <= _lstep]
        _K = resolve_chunk(_cap, _active, _lstep)
        self._tl_chunk = _K
        _wu_env = os.environ.get("PYNICAM_TIMELOOP_WARMUP")
        self._tl_warmup = int(_wu_env) if _wu_env is not None else _K
        if self._fuse_timeloop:
            self._notice(
                f"*** FUSE_TIMELOOP schedule: K={_K} (cap={_cap}, active intervals="
                f"{_active or 'none'}), warm-up={self._tl_warmup}"
                + (" [env override]" if _wu_env is not None else " [= K]"))
            if _cap > 1 and _K == 1:
                import math as _math
                self._notice(
                    "*** WARNING: fusion cannot engage beyond K=1 -- no divisor of "
                    f"gcd(intervals)={_math.gcd(*_active)} at or below cap={_cap}. "
                    "Every chunk is one step; consider adjusting the output interval.")

        # S2 GUARD: the fused jit path needs PYNICAM_COMM_NO_BARRIER=1 whenever the
        # config keeps COMM_apply_barrier -- the host PRC_MPIbarrier() fires at TRACE
        # time under jit and desyncs ranks that trace differing COMM counts,
        # deadlocking during compile (hangs silently at ~100% CPU, looking like a
        # slow compile). Refuse the combination loudly instead.
        if (self._fuse_timeloop
                and os.environ.get("PYNICAM_TIMELOOP_JIT", "0") != "0"
                and getattr(msc.comm, "COMM_apply_barrier", False)
                and os.environ.get("PYNICAM_COMM_NO_BARRIER", "0") == "0"):
            self._fuse_timeloop = False
            self._notice(
                "*** WARNING: PYNICAM_FUSE_TIMELOOP disabled -- COMM_apply_barrier is on "
                "and PYNICAM_COMM_NO_BARRIER is not set. The barrier fires at trace time "
                "under jit and DEADLOCKS the compile. Set PYNICAM_COMM_NO_BARRIER=1 (the "
                "barrier is redundant for correctness) or disable COMM_apply_barrier.")

        # SAFETY GUARD: the FUSE_TIMELOOP chunk (dyn.run_timeloop_chunk) advances K steps
        # on the device carry. It applies forcing after each dynamics step (via the
        # shared _forcing_apply_dev core) ONLY when forcing is FUSABLE -- i.e. the
        # resident device forcing path is active, which reduces to bk.resident() (the
        # RESIDENT_PRGVAR device stash). If forcing is active but NOT fusable
        # (non-resident / host forcing), the chunk would silently drop it, so disable
        # FUSE_TIMELOOP and fall back to the per-step path (which calls forcing_step).
        # Loud one-time warning -- never silent.
        _forcing_active = msc.rcnf.AF_TYPE in ('DCMIP', 'HELD-SUAREZ')
        _forcing_fusable = msc.bk.resident()
        if self._fuse_timeloop and _forcing_active and not _forcing_fusable:
            self._fuse_timeloop = False
            self._notice(
                "*** WARNING: PYNICAM_FUSE_TIMELOOP disabled -- AF_TYPE=%s forcing is active "
                "but NOT fusable (needs the resident device forcing path / RESIDENT_PRGVAR); "
                "the chunk would SILENTLY DROP forcing. Running the per-step path instead "
                "(forcing applied, correct)." % msc.rcnf.AF_TYPE)

        # DCMIP forcing-tendency validation dump (per-step .npz, per rank). Gated PYNICAM_FRC_DUMP=<path>.
        self._frc_dump = os.environ.get("PYNICAM_FRC_DUMP", "")
        # per-step history-diagnostics dump. Gated PYNICAM_HVAR_DUMP=<path>.
        self._hvar_dump = os.environ.get("PYNICAM_HVAR_DUMP", "")

    # --------------------------------------------------------------- phase: run

    def run(self, nsteps=None):
        """Advance the model `nsteps` large steps (default: to lstep_max). Returns self.

        The state is advanced IN PLACE -- the return value is this same object, so
        `nicam = nicam.run(n)` and `nicam.run(n)` are equivalent and neither copies
        anything. Successive calls continue from where the last one stopped:
        run(3); run(7) covers the same 10 steps as run(10), with the scheduled
        outputs firing at the same steps either way.

        Steps beyond lstep_max are not run: the output store is sized for lstep_max
        at IO_setup, and the clock is set to end there.
        """
        if not self._initialized:
            raise RuntimeError("run() before initialize()")
        if self._finalized:
            raise RuntimeError("run() after finalize()")

        from pynicamdc.share.mod_process import prc
        from pynicamdc.share.mod_prof import prf
        from pynicamdc.share.output_schedule import prg_output_fires, boundary_fires

        msc = self.msc
        np = self.np
        dyn, io, tim, frc = msc.dyn, msc.io, msc.tim, msc.frc
        lstep_max = tim.TIME_lstep_max

        n_end = lstep_max if nsteps is None else min(self._n + int(nsteps), lstep_max)
        if nsteps is not None and self._n + int(nsteps) > lstep_max:
            self._notice(
                f"*** run({nsteps}) trimmed to {n_end - self._n} step(s): step {self._n} + "
                f"{nsteps} exceeds timeparam.lstep_max = {lstep_max}. Raise lstep_max to "
                "run longer (it also sizes the output store).")
        if n_end <= self._n:
            return self

        first_call = self._n == 0
        if first_call:
            print("starting Main_Loop")
        prf.PROF_setprefx("MAIN")
        prf.PROF_rapstart("Main_Loop", 0)
        if self._prof_perstep and first_call:
            prf.PROF_rapsnap()   # baseline = post-init cumulative (excludes INIT_* from step deltas)

        # Output-step predicates (m = 0-based loop index; TIME_cstep = m+1 after this
        # step's advance). Shared source of truth (share/output_schedule.py) used by
        # BOTH the fusion chunk-trim guard and the per-step output fire, so a fused
        # chunk never spans -- and thus never silently drops -- an output step. Lands
        # output at TIME_cstep = interval, 2*interval, ... (nicamdc mod==0); the step-0
        # snapshot (t=0) is handled separately by PRGout_step0 in initialize().
        def _is_out_3d(m): return prg_output_fires(m + 1, io.PRGout_interval)
        def _is_out_2d(m): return prg_output_fires(m + 1, io.PRGout_interval_2d)

        # BOUNDARY steps (S1, FUSION_SCHEDULE_PLAN): steps after which the host must
        # see the state. The chunk trim uses this -- not the output predicates alone --
        # so a fused chunk never spans the budget monitor either (embudget_monitor is
        # only called on the per-step path; before this, an MNT_INTV step landing
        # inside a chunk was silently skipped). MNT_INTV joins only when MNT_ON.
        _bnd_intervals = [io.PRGout_interval, io.PRGout_interval_2d]
        if msc.embudget.MNT_ON:
            _bnd_intervals.append(msc.embudget.MNT_INTV)
        def _is_boundary(m): return boundary_fires(m + 1, _bnd_intervals)

        while self._n < n_end:
            n = self._n                     # 0-based index of the step about to run
            if self._cudart is not None and n == self._nsys_step:
                self._cudart.cudaProfilerStart()

            # Isolate the very first iteration (carries one-time JIT compilation under
            # jax) so the report shows compile-inclusive step1 separately. Steady-state
            # per-step = (Main_Loop - Main_Loop_step1) / (lstep_max - 1).
            if n == 0:
                prf.PROF_rapstart("Main_Loop_step1", 0)

            # --- fused K-step chunk? (S2: chunks are exactly K, END AT a boundary
            # step, and a tail shorter than K runs per-step instead) ---
            _K = 0
            if (self._fuse_timeloop and n >= self._tl_warmup
                    and getattr(dyn, "_step_core", None) is not None
                    and n_end - n >= self._tl_chunk):     # S2(c): no short chunks
                _K = self._tl_chunk
                for _j in range(_K):
                    if _is_boundary(n + _j):
                        _K = _j + 1   # S2(b): end the chunk AT the boundary step;
                        break         # its output/budget fire below, after the chunk.
                                      # Under the resolver (K | g, warm-up = K) the
                                      # boundary is the chunk's own last step, so
                                      # _K stays K; this trim is the safety net for
                                      # a dev warm-up override or an off-K run(n).
            if _K >= 1:
                prf.PROF_rapstart("_Atmos", 1)
                dyn.run_timeloop_chunk(msc, _K)   # (the profiler was started at loop top if n==_nsys_step)
                if self._cudart is not None and n == self._nsys_step_end:
                    self._cudart.cudaDeviceSynchronize()   # bound the window: finish the captured chunk's GPU work
                    self._cudart.cudaProfilerStop()
                prf.PROF_rapend("_Atmos", 1)
                for _j in range(_K):
                    tim.TIME_advance(msc.cldr, np.float64)
                self._n = n + _K
                # S2(b): host-side consumers for the chunk's LAST step (the only step
                # in the chunk that can be a boundary -- the trim breaks at the first
                # one). Both drain the device state first (embudget_monitor and
                # write() call sync_prgvar_to_host), and the monitor no-ops unless
                # TIME_cstep lands on MNT_INTV -- identical to the per-step path.
                msc.embudget.embudget_monitor(msc)
                _fire_3d = _is_out_3d(self._n - 1)
                _fire_2d = _is_out_2d(self._n - 1)
                if _fire_3d or _fire_2d:
                    self.write(write_3d=_fire_3d, write_2d=_fire_2d)
                if self._prof_perstep:
                    prf.PROF_rapreport_step(n)
                continue

            # --- ordinary per-step path (warm-up, output steps, or fusion off) ---
            prf.PROF_rapstart("_Atmos", 1)

            dyn.dynamics_step(msc)

            # Artificial forcing (nicamdc prg_driver-dc.f90: forcing_step follows
            # dynamics_step inside _Atmos). No-op unless AF_TYPE == 'DCMIP'. Re-derives
            # diag from the final prognostic, applies the DCMIP tendencies, writes back
            # + halo/pole COMM.
            dyn.forcing_step(msc)

            # Validation dump: per-step DCMIP forcing tendencies (ml_af_fvx.. +
            # sl_af_prcp) to per-rank .npz, to be compared against the nicamdc golden
            # history. Gated + inert by default. n is 0-based (nicamdc history frame = n+1).
            if self._frc_dump and msc.rcnf.AF_TYPE == 'DCMIP':
                np.savez(f"{self._frc_dump}_step{n+1:03d}_rank{prc.prc_myrank}.npz",
                         fvx=frc.fvx, fvy=frc.fvy, fvz=frc.fvz, fe=frc.fe,
                         fq=frc.fq, precip=frc.precip)

            if self._cudart is not None and n == self._nsys_step_end:
                self._cudart.cudaDeviceSynchronize()   # bound the window: finish the captured steps' GPU work
                self._cudart.cudaProfilerStop()

            prf.PROF_rapend("_Atmos", 1)

            tim.TIME_advance(msc.cldr, np.float64)
            self._n = n + 1          # this step is complete; tracks TIME_cstep

            # energy & mass budget monitor (nicamdc: after TIME_advance). No-op unless MNT_ON.
            msc.embudget.embudget_monitor(msc)

            # Output fires when TIME_cstep (= n+1 after this step's advance) is a multiple
            # of the interval, i.e. at TIME_cstep = interval, 2*interval, ... (matches
            # nicamdc mod(TIME_CSTEP,interval)==0). The 3D group (prognostics + ml_) uses
            # PRGout_interval; the 2D group (sl_) PRGout_interval_2d.
            _fire_3d = _is_out_3d(n)
            _fire_2d = _is_out_2d(n)
            if _fire_3d or _fire_2d:
                self.write(write_3d=_fire_3d, write_2d=_fire_2d)

            if n == 0:
                prf.PROF_rapend("Main_Loop_step1", 0)

            if self._prof_perstep:
                prf.PROF_rapreport_step(n)   # delta since last step -> this step's cost

        prf.PROF_rapend("Main_Loop", 0)
        return self

    # ------------------------------------------------------------- phase: write

    def write(self, write_3d=True, write_2d=True):
        """Write one output snapshot of the current state to the zarr store.

        Called by run() at the scheduled steps, and callable directly for an
        unscheduled snapshot. On the jax backend this is also the point where the
        device carry is drained to the host arrays -- reading prgv.PRG_var without
        it can give a stale state.

        The store is sized from the output schedule at IO_setup, which is exact for
        scheduled output; an unscheduled snapshot has no slot reserved for it, so the
        time axis grows to take it (in blocks, trimmed to the true length by
        IO_finalize). Growth costs a store-wide metadata rewrite and two barriers, so
        output wanted at a fixed cadence still belongs in PRGout_interval.
        """
        if not self._initialized:
            raise RuntimeError("write() before initialize()")

        from pynicamdc.share.mod_process import prc
        from pynicamdc.share.mod_prof import prf

        msc = self.msc
        np = self.np
        dyn, io = msc.dyn, msc.io

        # Output timing: the three host-side phases are profiled separately (_Out_D2H =
        # device->host drain, _Out_Diag = derived-diagnostic compute, _Out_Write = zarr
        # write). Shown in the PROF report next to _Atmos so output cost is attributable.
        prf.PROF_rapstart("_Out_D2H", 1)
        dyn.sync_prgvar_to_host(msc.prgv, msc)   # materialize host PRG_var from the device stash (no-op when the gate is off)
        dyn.assert_host_prgvar_synced("driver.output")  # host PRG_var must be current before the reads below
        prf.PROF_rapend("_Out_D2H", 1)
        # derived history diagnostics (only the group(s) being written this step)
        prf.PROF_rapstart("_Out_Diag", 1)
        _hv = (dyn.history_vars_step(msc, write_3d=write_3d, write_2d=write_2d)
               if (io.PRGout_diagnostics or self._hvar_dump) else None)
        prf.PROF_rapend("_Out_Diag", 1)
        prf.PROF_rapstart("_Out_Write", 1)
        io.IO_PRGstep(msc.tim, msc.prgv, msc.rcnf, msc.bk.ndtype, diag=_hv,
                      write_3d=write_3d, write_2d=write_2d)
        prf.PROF_rapend("_Out_Write", 1)
        if self._hvar_dump:
            np.savez(f"{self._hvar_dump}_step{self._n:03d}_rank{prc.prc_myrank}.npz",
                     **{k: np.asarray(v) for k, v in _hv.items()})
        return self

    # ---------------------------------------------------------- phase: teardown

    def finalize(self):
        """Drain the output, report timings, run the gated end-of-run dumps, end MPI.

        Ends MPI for the process, so the instance is dead afterwards. Idempotent.
        """
        if not self._initialized:
            raise RuntimeError("finalize() before initialize()")
        if self._finalized:
            return self

        from pynicamdc.share.mod_process import prc
        from pynicamdc.share.mod_stdio import std
        from pynicamdc.share.mod_prof import prf

        msc = self.msc
        np = self.np
        bk = self.bk
        dyn = msc.dyn

        # drain + join the async output writer (no-op unless PRGout_async). The tail here
        # is only the last output(s) still in flight -- everything earlier overlapped the
        # compute loop.
        prf.PROF_rapstart("_Out_Finalize", 0)
        msc.io.IO_finalize()
        prf.PROF_rapend("_Out_Finalize", 0)

        prf.PROF_rapreport()

        # STEP C validation hook: dump the FINAL prognostic device state to a per-rank
        # .npy so a FUSE_TIMELOOP=on run can be compared bit-exact against the
        # FUSE_TIMELOOP=off run (the gl07 gold only emits one snapshot at n=1 =
        # pre-warm-up, so an end-of-run off-vs-on dump is the real check that the K-step
        # scan reproduces the per-step path). Gated PYNICAM_TIMELOOP_DUMP=<path>.
        _tl_dump = os.environ.get("PYNICAM_TIMELOOP_DUMP", "")
        if _tl_dump:
            dyn.sync_prgvar_to_host(msc.prgv, msc)
            np.save(f"{_tl_dump}_rank{prc.prc_myrank}.npy", np.asarray(msc.prgv.PRG_var))
            print(f"TIMELOOP_DUMP wrote {_tl_dump}_rank{prc.prc_myrank}.npy", flush=True)

        # END-OF-RUN restart write (validation of restart reproducibility). Distinct from
        # the startup PYNICAM_RESTART_OUT (which dumps the IC). Syncs the final device
        # PRG_var to host and writes a restart file via prgv.restart_output honoring the
        # config's output_prognostics/output_diagnostics. The default prognostic write is
        # bit-exact on the round trip; a diagnostic write needs DIAG_var refreshed from
        # the current PRG_var first (it is only current at the IC otherwise).
        # PYNICAM_RESTART_OUT_END=<basename.pe>.
        _r_out_end = os.environ.get("PYNICAM_RESTART_OUT_END", "")
        if _r_out_end:
            dyn.sync_prgvar_to_host(msc.prgv, msc)
            if msc.prgv.output_diagnostics:
                msc.prgv.DIAG_var, msc.prgv.DIAG_var_pl = msc.cnvv.cnvvar_prg2diag(
                    msc.prgv.PRG_var, msc.prgv.PRG_var_pl, msc.cnst, msc.vmtr, msc.rcnf,
                    msc.tdyn, msc.bk.ndtype)
            _ctime = int(getattr(msc.tim, "TIME_ctime", 0) or 0)
            msc.prgv.restart_output(_r_out_end, msc.rcnf, msc.bk.ndtype, ctime=_ctime)
            print(f"RESTART_OUT_END wrote {_r_out_end}<rank> (io_mode={msc.prgv.output_io_mode}, "
                  f"prognostics={msc.prgv.output_prognostics}, diagnostics={msc.prgv.output_diagnostics})",
                  flush=True)

        # DEVICE-SIDE CHECKSUM (PYNICAM_DEV_CHECKSUM=1): reductions computed ON the device
        # carry (dyn._prgvar_d), draining ONLY scalars -- sidesteps the multi-rank host
        # array-drain that corrupts the full-array to_numpy. Answers (a) is the device
        # state real (nfin>0, csum finite) and (b) fused-vs-perstep at multi-rank (compare
        # the per-rank scalars across runs).
        if os.environ.get("PYNICAM_DEV_CHECKSUM", "0") != "0":
            _pd = getattr(dyn, "_prgvar_d", None)
            _r = prc.prc_myrank
            if _pd is not None and msc.bk.type == "jax":
                _jnp = msc.bk.xp
                _abs = _jnp.abs(_pd)
                _nfin = int(_jnp.isfinite(_pd).sum())
                _csum = float(_jnp.nansum(_abs))
                _cmax = float(_jnp.nanmax(_abs))
                _csq = float(_jnp.nansum(_pd.astype(_jnp.float64) ** 2))
                _slots = [float(_jnp.nansum(_jnp.abs(_pd[:, :, :, :, s]))) for s in range(_pd.shape[-1])]
                print(f"[DEV_CHECKSUM] rank{_r} nfin={_nfin}/{_pd.size} csum={_csum:.10e} "
                      f"csq={_csq:.10e} cmax={_cmax:.8e}", file=sys.stderr, flush=True)
                print(f"[DEV_CHECKSUM_SLOTS] rank{_r} " + " ".join(f"{v:.8e}" for v in _slots),
                      file=sys.stderr, flush=True)
            else:
                print(f"[DEV_CHECKSUM] rank{_r} NO _prgvar_d (RESIDENT_PRGVAR off / not jax)",
                      file=sys.stderr, flush=True)

        # Peak GPU memory report (per rank). Report BOTH metrics so they aren't confused:
        #  * peak_pool_bytes    = peak bytes XLA RESERVED from the device (the OOM-relevant
        #                         footprint, comparable to nvidia-smi minus the CUDA
        #                         context ~few hundred MB).
        #  * peak_bytes_in_use  = peak LIVE tensor bytes (undercounts the true footprint --
        #                         pool reserve, workspaces and fragmentation are excluded).
        # Also dump the full stats dict once (rank0) so the available keys are on record.
        # Gated by the `mem` tag of PYNICAM_PROFILE.
        if bk.profile("mem") and msc.bk.type == "jax":
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

        if self._dtype_audit is not None:
            self._dtype_audit.report()

        prc.prc_mpifinish(std.io_l, std.fname_log)

        self._finalized = True
        return self

    # ------------------------------------------------------------------ context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._initialized and not self._finalized:
            self.finalize()
        return False
