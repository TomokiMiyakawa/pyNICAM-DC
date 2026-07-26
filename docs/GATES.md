# PYNICAM_* environment gates — the living ledger

Every gate the code actually reads, its default, and why it exists. Anything
NOT in this table is stale (fix the reference or delete it). Keep this file in
lockstep with `config/production.env`.

Status legend: **prod** = part of the supported production configuration;
**choice** = genuine per-run option; **debug** = diagnostic instrument
(off unless investigating); **tuning** = numeric knob with a safe default.

## Run-mode selection (driversettings.toml, not env)

| key | values | meaning |
|---|---|---|
| `backend` | `numpy` \| `jax` | numerical stack |
| `precision` | `float64` \| `float32` | fp precision (sets jax x64) |
| `comm` | `auto` \| `serial` \| `mpi` | serial = no mpi4py needed (1 rank) |

## Production / choice gates

| gate | default | status | meaning |
|---|---|---|---|
| `PYNICAM_RESIDENT` | 1 | prod (anchor) | device-resident prognostics; master switch of the GPU fast path |
| `PYNICAM_TIMELOOP_JIT` | 1 | choice | jit the per-step body; `=0` = eager-jax, used as a debugging intermediate (ncclffi audits) |
| `PYNICAM_COMM_ALLTOALL` | 1 | choice | on-device alltoall halo vs host sendrecv (sweep compares both) |
| `PYNICAM_FUSE_TIMELOOP` | 0 | choice | lax.scan over the whole driver loop |
| `PYNICAM_COMM_NCCLFFI` | 0 (prod env: 1) | prod | direct NCCL halo exchange via jax FFI |
| `PYNICAM_COMM_NO_BARRIER` | 0 | choice | drop the pre-COMM barrier (required under FUSE_TIMELOOP) |

## Tuning values

| gate | default | meaning |
|---|---|---|
| `PYNICAM_TIMELOOP_CHUNK` | 1 | steps per fused chunk (int) |
| `PYNICAM_TIMELOOP_WARMUP` | 3 | eager warmup steps before entering the fused loop |
| `PYNICAM_PINNED_D2H_MB` | 16 | min transfer size for the pinned-host D2H path |
| `PYNICAM_NCCLFFI_LIB` | (unset) | explicit path to the NCCL FFI shared library |
| `PYNICAM_XFER_PROF_ATTR_MB` | 32 | xfer profiler: attribute call sites for transfers >= this |
| `PYNICAM_XFER_PROF_OUT`, `PYNICAM_H2D_PROF_OUT` | `xfer_prof` / `h2d_prof` | profiler output-file bases |

## Debug / diagnostic instruments (off by default)

| gate | meaning |
|---|---|
| `PYNICAM_TIMELOOP_DUMP` | dump the final state to `<prefix>_rank<r>.npy` (tier2/3 validation uses this) |
| `PYNICAM_IC_DUMP`, `PYNICAM_GRD_DUMP`, `PYNICAM_FRC_DUMP`, `PYNICAM_HVAR_DUMP`, `PYNICAM_BS_DUMP` | stage-specific state dumps |
| `PYNICAM_PROFILE` | comma-separated profiler tags: `xfer`/`h2d` (transfers), `perstep`, `timeloop_timing`, `mem` (GPU memory report), ... |
| `PYNICAM_NSYS_STEP`, `PYNICAM_NSYS_STEP_END` | nsys capture window |
| `PYNICAM_DTYPE_AUDIT` | float32 dtype-preservation audit of pure kernels |
| `PYNICAM_DEV_CHECKSUM` | per-step device checksums |
| `PYNICAM_DRAIN_CANARY` | flags unexpected host drains in the resident span |
| `PYNICAM_COMM_DEGREE`, `PYNICAM_COMM_WARM_LOG` | COMM topology/warmup diagnostics |
| `PYNICAM_NCCLFFI_VERBOSE`, `PYNICAM_NCCLFFI_TRACELOG`, `PYNICAM_NCCLFFI_SELFROW`, `PYNICAM_NCCLFFI_PACKREV` | NCCL-FFI path diagnostics |
| `PYNICAM_RESTART_OUT` | write a restart at run end |

(GPU memory report is not a separate gate -- it is the `mem` tag of `PYNICAM_PROFILE`.)

## Collapsed into code (no longer gates)

2026-07-25 (`29bc08a`), validated bit-neutral on both jax and numpy arms:
`PYNICAM_PINNED_D2H`, `PYNICAM_FAST_COMM` (code hook `self.use_fast_comm`
remains), `PYNICAM_NCCLFFI_TRIM`, `PYNICAM_HDIFF_ONDEVICE_COMM`.

## Not on main (branch-only)

`PYNICAM_CONST_ARGS` (device-consts-as-args mem valve) and its companions
`PYNICAM_DEVCONST_CHECK` / `PYNICAM_DEVCONST_EXEMPT` live only on the `mem-peak`
branch (never merged; see the mem-peak campaign). main's code never reads them,
so they are not gates here -- do not re-add without the branch's code.

## Deleted (closed experiments)

2026-07-25 (user decision): the sharding/ppermute experiment
(`PYNICAM_COMM_SHARDING`, `PYNICAM_COMM_SHARDING_C1`, `PYNICAM_SHARDMAP_*`,
`PYNICAM_PPERMUTE_*`) and the bisection/warmup instruments
(`PYNICAM_BISECT_NONLSCAN`, `PYNICAM_BISECT_NOTRACER`,
`PYNICAM_FORCE_EAGER_WARM`, `PYNICAM_LA_WARM_FORCE_EAGER`).
Recoverable from git history (branch `comm-nccl-ffi` era).
