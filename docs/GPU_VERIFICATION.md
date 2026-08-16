# Verification — `api-layer`

This file was written when everything jax on this branch was owed to Miyabi or
Levante: multi-rank jax needs `mpi4jax`, which would not install on the laptop the
work was done on. It does install (recipe below), so the **device-resident and
fused jax paths have now been A/B'd locally**, including the two cases the CPU
numpy A/B could not reach.

What remains owed to a GPU is what a laptop cannot answer: performance, scale, and
the GPU-specific transports.

---

## Verified — JW gl05rl01 z40, 8 ranks, IDEAL initial condition, 12 steps, float64

| check | result |
|---|---|
| numpy, `main` vs `set-at-value-dispatch` | bit-exact, 48 variables |
| numpy, `main` vs `api-layer` | bit-exact, 48 variables |
| jax host-staged (`RESIDENT=0`), `main+fix` vs `api-layer` | bit-exact, 48 variables |
| **jax device-resident (default), `main` vs `api-layer`** | **bit-exact, 48 variables** |
| **jax fused (`FUSE_TIMELOOP=1`), `main` vs `api-layer`** | **bit-exact, 48 variables** |
| **jax fused vs per-step, `api-layer`** | **bit-exact, 48 variables** |
| **jax fused vs per-step, `main`** | **bit-exact, 48 variables** |
| API as `run(3)` × 4 vs one `run()`, numpy | bit-exact, 48 variables |
| 6 unscheduled `write()` calls with nothing reserved | store ends exactly 6 slots, all finite, all distinct |
| test suite | 74 passed, 1 skipped |

The fused runs were **asserted to engage**, not assumed: `PYNICAM_PROFILE=timeloop_timing`
shows 24 chunk firings (8 ranks × 3), every one at `K=2` — exactly the trim predicted
for `PRGout_interval=3`, warm-up 3, cap 4 (steps 4-5, 7-8, 10-11).

That last group also settles what `docs/FUSION_SCHEDULE_PLAN.md` carried as its main
open question: **the fused path reproduces the per-step path bit-for-bit**, on both
branches. At `K=2` and gl05 — see the plan for what that does and does not license.

## Still owed on GPU

**Performance, all of it.** Step time for the `set_at` change (one `isinstance` per
call at 14 sites, two in the time loop). The cap sweep. Per-step vs fused step time —
the number the whole warm-up choice in `FUSION_SCHEDULE_PLAN.md` turns on, and which
nothing in the repo measures.

**Scale.** Everything above is gl05rl01 at 8 ranks with `K=2`. The fused
bit-exactness that matters for that plan is at production K and resolution.

**The GPU transports.** Locally, jax runs on the CPU backend and `mpi4jax` moves
host buffers. GPU-aware MPI and the NCCL-FFI path (`PYNICAM_COMM_NCCLFFI=1`) are
untouched by any of this — as is the AMD/RCCL variant in `tools/rocm_gl05_kit/`.

**Environment.** `test/set_at_test.py::test_a_tracer_takes_the_jax_branch` pins a
jax implementation detail (tracers are `jax.Array` instances). Confirmed on jax
0.6.0 only; run the suite on whatever jax the cluster has.

---

## Building mpi4jax on macOS — three traps, all silent

All three must be cleared or the install produces something that imports and fails,
or does not import at all.

```bash
# in a venv layered on the conda env, so the env itself stays clean:
$ENV/bin/python -m venv --system-site-packages /path/to/venv
MPICH_CC=/usr/bin/clang /path/to/venv/bin/pip install \
    --no-build-isolation --no-cache-dir mpi4jax
```

1. **conda's `mpicc` calls a compiler that is not there.** It has
   `CC="arm64-apple-darwin20.0.0-clang"` baked in; conda-forge ships the wrapper
   without the compiler. `MPICH_CC` overrides it (`OMPI_CC` for OpenMPI).
2. **pip's build isolation pulls its own jaxlib.** mpi4jax then compiles against
   those headers and registers FFI handlers at an API version the installed jaxlib
   rejects — *at import*, with `handler's API version (0.3) is incompatible with
   the framework's API version (0.1)`. `--no-build-isolation` builds against the
   jaxlib that will actually run it.
3. **pip reuses the wheel from the failed build.** After fixing (2) the install
   looks clean and the same error persists, because nothing was rebuilt.
   `--no-cache-dir` (with `--force-reinstall` if it is already installed).

Confirmed working: mpi4jax 0.9.1, jax/jaxlib 0.6.0, mpich 4.3.0, Python 3.11,
macOS arm64. `pynicamdc/nhm/dynamics/proto/test_mpi4jax_sanity.py` passes at 2
ranks, including its in-graph (`jit sendrecv, token-threaded`) case.

## Running the fused path — three gates, not one

```bash
PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1 PYNICAM_COMM_NO_BARRIER=1 \
PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_WARMUP=3 ...
```

`PYNICAM_COMM_NO_BARRIER=1` is documented in `docs/GATES.md` as a standalone
diagnostic, but with `COMM_apply_barrier = true` in the config it is a
**precondition for fusion**: `PRC_MPIbarrier()` is Python, so under a jit trace it
fires at *trace* time, and ranks that trace a differing number of COMM calls (pole
vs non-pole) desync into a **deadlock during compile** (`mod_comm.py:2011`). Every
fused template under `tools/` sets it; omitting it hangs at ~100% CPU with no
message, three steps in, looking exactly like a slow compile.

## What a compile costs, measured here

Chunk wall time on the CPU backend, 8 ranks, `K=2`:

| | wall |
|---|---|
| first chunk (compile included) | **20.33 s** |
| every later chunk | 1.04 – 1.10 s |

~19×, and 40% of the whole 12-step `Main_Loop` (50.6 s) went into one compile. The
same order as the 16–24 s/rank recorded on GPU in
`tools/jupiter/SCALING-LADDER.md:148`, which is the cost `FUSION_SCHEDULE_PLAN.md`
is about paying once instead of hundreds of times.

## Before trusting any timing measured here

`PROF_setup` reads the profiler's two settings into `self.Prof_rap_level` and
`self.Prof_mpi_barrier` (`mod_prof.py:52-53`), while `PROF_rapstart` / `PROF_rapend`
read `self.PROF_rap_level` and `self.PROF_mpi_barrier` (`mod_prof.py:90,96`).
Different attributes, so `[param_prof]` never reaches the profiler:

- `prof_rap_level = 10` → the level stays at its default of 2. Harmless today, as
  every timer in the model is level 0, 1 or 2, so nothing is dropped — but a
  level-3 timer added later would silently never appear.
- `prof_mpi_barrier = true` → no barrier is taken. Rap times are therefore *not*
  synchronised across ranks: each rank reports its own arrival, and load imbalance
  shows up inside whichever timer happens to contain the next collective rather
  than in the timer that caused it.

Pre-existing on `main`, unrelated to this branch. It does not invalidate the
recorded per-step numbers (a missing barrier makes the timer cheaper, not wrong for
a single rank), but read multi-rank PROF reports with it in mind, and fix it before
using them to attribute imbalance.

**`tools/sweep/make_config.py --output` used to do the opposite of what it says —
fixed by `5518d07` (branch `claude/amazing-shamir-22139a`, off `main`, not yet
merged), so the note below applies to any timing recorded before that commit
lands.** `off` is the default and is documented as "minimise I/O for
clean timing", but it set `PRGout_interval=1` — a zarr write **every step** — while
`on` set it to `lstep_max`, one write at the end. It is a regression from `db224e2`:
under the guard that preceded it (`n % interval == 1`) `interval=1` never fired, so
`off` genuinely disabled writes; `db224e2` corrected the phase to
`(n+1) % interval == 0` and `1` became "every step" without the value being changed
back. `off` now sets `interval = lstep+1`, past `lstep_max` — which is what the
surviving "keep nt>=1" on that line was describing.

**So any timing taken through `run_sweep.sh` before `5518d07` is not comparable with
one taken after**: it passed no `--output`, so every run it drove carried a per-step
zarr write inside the timed `Main_Loop` (measured at gl06 numpy pe04 12 steps: 1062 MB
vs 2 MB written; on a local SSD the wall-time difference was below run-to-run spread —
untested on a shared parallel filesystem or at gl08/gl09, where that volume is the
thing to watch). `timing_hires.pbs` was
never affected — it overrides `PRGout_interval` to 1000 itself, as does every other
script here that sets `FUSE_TIMELOOP`, which is why nobody hit the sharper version of
this: the chunk-trim guard (`driver-dc.py:471-476`) cuts a chunk at every output step,
so `interval=1` would have silently reduced the fused path to per-step.

The golds (`run/golds/gl0N_numpy_gold.zarr`) hold a single frame — but of the state at
`TIME_cstep = 2`, not the final step (they predate `db224e2` too). Reproducing one
needs `--output on --lstep 2`; see `tools/sweep/README.md`, "Validating against the
golds".

`Main_Loop_step1` has a related problem — it wraps only `n == 0`, so the
steady-state formula beside it keeps the rest of the warm-up *and* the first
chunk's compile inside the "steady" average. See `FUSION_SCHEDULE_PLAN.md`, defect 5.

---

## A100 verification — Levante, 2026-08-16 (paid from "Still owed on GPU")

JW gl05rl01 z40, 4 ranks (1/GPU, A100 80GB x4, CUDA-aware OpenMPI 4.1.2),
IDEAL, 12 steps, float64. jax 0.10.2 / python 3.11. Job:
`tools/levante/a100/apilayer_ab.sbatch` (all PORT.md traps honored; zarr
compared with the campaign `cmp_zarr.py --exact`).

| check | result |
|---|---|
| test suite on cluster jax 0.10.2 (login node, CPU) | 76 passed, 4 skipped* |
| jax device-resident per-step, `main` vs `api-layer` — **on GPU** | bit-exact, 9/9 arrays |
| jax fused vs per-step, `api-layer` — **on GPU** | bit-exact, 9/9 arrays |
| NCCL-FFI checksum audit (`PYNICAM_NCCLFFI_CKSUM=1`) | 7344 pairs, 0 mismatches — CLEAN |
| fused asserted to engage | 8 `TIMELOOP_CHUNK` firings, K=4 |

*Skips are benign: three pinned-d2h-fallback tests (device has `pinned_host`;
nothing to fall back from) and one missing tutorial reference file. The
jax-0.6.0-pinned tracer test (`set_at_test.py`) passes unchanged on 0.10.2.

Incidental timing (not a measurement campaign): first fused chunk 20.81 s
(compile), steady chunks 0.185 s = 0.046 s/step — same ~20 s compile-once
cost recorded on the CPU backend above and in `tools/jupiter/SCALING-LADDER.md`.

**Still owed, by decision now owed to JUPITER, not Levante:** the performance
sweep (`set_at` step-time, cap sweep, per-step vs fused timing) and
production-scale K/resolution/multinode.
