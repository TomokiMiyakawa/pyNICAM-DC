# The `pyNICAM` object API

```python
from pynicamdc import pyNICAM

nicam = pyNICAM("driversettings.toml")
nicam.initialize(parameters={"timeparam": {"lstep_max": 24}})
nicam = nicam.run(10)          # advances state and time
nicam.write()
nicam.finalize()
```

`driver-dc.py` is this API plus argument parsing, so there is one startup sequence,
not two. Anything the driver could do, a script can.

---

## Phases

| call | what it does |
|---|---|
| `pyNICAM(path, backend=, precision=, comm=)` | fixes backend / precision / comm. Imports no model module. |
| `initialize(parameters=None)` | builds grid, metrics, operators, initial or restart state, output store |
| `run(nsteps=None)` | advances `nsteps` large steps (default: to `lstep_max`); returns `self` |
| `write(write_3d=, write_2d=)` | writes one output snapshot; `run()` calls it on schedule |
| `finalize()` | drains output, reports timings, ends MPI |

`rank`, `nprocs`, `step`, `time` and `lstep_max` are readable at any point after
construction. The model itself is reachable as `nicam.msc` (`msc.prgv.PRG_var`,
`msc.grd`, ...) — the same container the model uses internally.

`with pyNICAM(...) as nicam:` finalizes on exit.

## `run()` returns `self`, and that is not a copy

The prognostic array is allocated once, in `prgvar_setup`, and every step
overwrites it in place (`prgv.PRG_var[:, :, :, :, 0:6] = ...` — note the slice on
the left). `run()` hands back the same object, so

```python
nicam = nicam.run(10)      # identical to nicam.run(10)
```

rebinds a name to what it already pointed at. Nothing is copied, moved, or rebuilt,
and memory does not grow with the number of calls. It also means an earlier
reference is **not** a snapshot of an earlier state — `old = nicam` before the call
sees the advanced state too.

Successive calls continue where the last stopped: `run(3); run(7)` covers the same
ten steps as `run(10)`, with the scheduled outputs firing at the same steps.
Verified bit-exact against a single `run()` (JW gl05rl01, 8 ranks, numpy).

## `initialize(parameters=...)`

`parameters` is a nested dict keyed by the config's tables:

```python
nicam.initialize(parameters={
    "timeparam": {"lstep_max": 24},
    "ioparam":   {"PRGout_name": "run1.zarr", "PRGout_interval": 3},
})
```

Keys absent from the file are added; the file is not modified.

It works by merging into a temporary copy of the config, because the config travels
as a **path**: about fifteen setup routines re-open it themselves (`toml.load` at 38
sites), so there is no single in-memory dict to patch. Two consequences:

- Paths inside the config stay relative to the **current directory**, exactly as
  before. The temporary file's own location is never used to resolve them.
- `admparam.rgnmngfname` names the file `[rgnmngparam]` is read from and, in the
  shipped configs, points back at the config itself. That self-reference is
  retargeted to the merged copy so a `[rgnmngparam]` override is actually seen. A
  config that genuinely keeps its region table in another file is left alone.

backend, precision and comm are **not** settable here — see below.

## Constraints

**One instance per process.** `adm`, `prc`, `std`, `prf`, `cldr`, `satr`, `frc` and
`embudget` are module-level singletons, so a second `pyNICAM` would share their
state with the first. Constructing one is allowed (a re-run with the same settings);
running two models side by side in one process is not.

**backend and precision are fixed at construction.** They bind the kernels and the
device arrays. A conflicting second construction raises rather than silently keeping
the first choice.

**Import order is load-bearing.** `comm_mode` decides mpi-vs-serial once, at the
first import of `mod_process`, so a script must construct `pyNICAM` *before*
importing anything else from `pynicamdc`. Getting this wrong does not raise on its
own — it silently runs the wrong mode, and `srun -n 64` against a serial decision
starts 64 independent rank-0 processes. Two things keep it honest:

- `import pynicamdc` pulls in no submodule at all; `pyNICAM` resolves on first use.
- Construction raises if `mod_process` was already imported with a conflicting mode.

Ask the instance (`nicam.rank`) rather than importing `prc` at the top of a script.

**The output store is sized for the run.** `IO_setup` preallocates
`floor(lstep_max / PRGout_interval)` slots (`+1` with `PRGout_step0`), and
`IO_PRGstep` writes into slot *i*. Scheduled output fits exactly; extra `write()`
calls consume slots from the same pool and are dropped once it is empty. Sizing the
store independently of the step schedule is the follow-up that makes ad-hoc
`write()` fully general.

**On the jax backend, `write()` is also the drain point.** The prognostic lives on
the device between steps; `write()` calls `sync_prgvar_to_host` first. Reading
`msc.prgv.PRG_var` directly after `run()` without that sync can give a stale state.

## Verification

- JW gl05rl01 z40, 8 ranks, numpy, IDEAL initial condition, 12 steps: the driver on
  this API is **bit-exact** against `main` — 48 variables (6 prognostics, 22 `ml_`,
  20 `sl_`) over 5 output slots, `array_equal`.
- The same run driven as `run(3)` × 4 through the API, with the output store
  redirected via `parameters=`, is bit-exact against the single-`run()` driver output.
- `test/api_test.py` guards the import-order contract, the reconfiguration guards,
  the parameters overlay and the phase ordering. It needs no data and no MPI.
- The same case on the jax backend, host-staged (`PYNICAM_RESIDENT=0`), 8 ranks:
  bit-exact against the same baseline. This path needed the `set_at` dispatch fix
  to run at all — the jax backend's device-resident default still needs `mpi4jax`
  for the multi-rank halo exchange, which is absent here, so **the production jax
  path (`RESIDENT=1`) remains un-A/B'd**. Run it where jax + mpi4jax exist.
