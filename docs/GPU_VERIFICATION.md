# Verification — `api-layer`

This file was written when everything jax on this branch was owed to Miyabi or
Levante: multi-rank jax needs `mpi4jax`, which would not install on the laptop the
work was done on. It does install (recipe below), so the **device-resident and
fused jax paths have now been A/B'd locally**, including the two cases the CPU
numpy A/B could not reach.

What remains owed to a GPU is what a laptop cannot answer: performance, scale, and
the GPU-specific transports.

---

## READ FIRST — every multi-rank JUPITER result below 2026-08-20 was computed on a blown-up state

**Found 2026-08-20.** `GRD_gen_plgrid` (`pynicamdc/share/mod_grd.py`) reused one
send buffer for all of its pole-vertex `Isend`s and overwrote it before the
sends completed. Any rank owning more than one pole-adjacent region sent the
*last* region's vertex in every message. OpenMPI copies small messages eagerly
and hid it; ParaStationMPI on JUPITER does not, so on JUPITER `GRD_xt_pl[1..5]`
were wrong, hence `GMTR_*_pl`, `OPRT_coef_*_pl`, `VMTR_*_pl`, and — through
p2r — the pole-corner `GMTR_area`/`VMTR_VOLUME` of the ten pole-adjacent
regions (off by ~1e10). Those regions blew up at **step 1** and the NaN swept
the globe within ~20 steps. Fixed in commit `cc92845`.

What that means for this file:

- Every **"bit-exact A vs B"** row measured on JUPITER with ≥2 ranks before the
  fix (the 2026-08-16 GH200 section; the S2/S3 section's zarr and BUDGET rows)
  compared two NaN-filled states. `array_equal(equal_nan=True)` is satisfied by
  NaN == NaN, so those rows proved nothing about the numerics. They are kept
  below, struck through, as a record.
- **Timing** rows (the fusion-schedule section, the per-step vs fused numbers,
  the K sweeps) were measured on NaN arithmetic. GPU throughput does not depend
  on operand values, so they are probably still representative, but they are
  *unvalidated* until re-measured.
- The laptop (OpenMPI) and Levante (OpenMPI) rows are unaffected by this bug.
  The gl12 pe256 "physically sane" check in `tools/jupiter/SCALING-LADDER.md`
  was real (that decomposition apparently puts each pole-adjacent region on its
  own rank, so the overwrite never happened) — but any rl01 pe4 result on
  JUPITER, on any branch, was NaN.
- How it was missed: every check was A-vs-B bit-exactness; no run with
  `MNT_INTV` small enough to fire, and every zarr inspected either unwritten
  (fill NaN) or compared bytewise. **A verification must include a finiteness
  / physical-sanity check (`tutorial/check_validation.py`, or `BUDGET_*.log`
  with `MNT_INTV=1`) before any bit-exactness claim.**

**Diagnosis trail** (all JUPITER, 2026-08-20): f90 stable on the same case
(jobs 1432550); pyNICAM gl09 and gl05 pe4 blow up at step 1 in fp32/fp64,
numpy/jax, main/api-layer, output on/off (jobs 1432260–1433038); gl05 rl00
pe1 Tier-2 `jw` passes its golden and gl05 rl01 **pe1** is stable; pe4 vs pe1
after one step differs only in the pole-adjacent regions (job 1433116);
`sweep/scripts/comm_probe.py` shows `COMM_data_transfer` correct for every
signature (job 1433208); `sweep/scripts/setup_probe.py` finds the first
differing array: `GRD_xt_pl` (job 1433221).

**After the fix** (gl05 rl01 pe4 fp64 `MNT_INTV=1`, job 1433269): the per-step
budget is identical to the rl01 pe1 reference, numpy and jax agree to 1e-9.
gl09 pe4 fp32 fused (S3 case, job 1433270): finite, residuals ~1e-2 W/m²,
S3 compile counts unchanged. The S2 fused-vs-plain comparison is re-run below
on real data.

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

---

## GH200 verification — JUPITER, 2026-08-16 (first payment on the JUPITER debt)

JW gl09rl01 z40, 4 ranks (1/GPU, GH200 x4, one node), IDEAL, float32,
production fusion config (`FUSE_TIMELOOP=1 JIT=1 CHUNK=4 WARMUP=3`,
`PYNICAM_COMM_NCCLFFI=1 PYNICAM_COMM_NO_BARRIER=1`). Stages/2026 +
ParaStationMPI 5.13.0-1, jax 0.10.2 — the validated xspies campaign stack.
Job 1391145, sbatch `sweep/jupiter_gl09_apilayer_ab.sbatch` (xspies work
area). `api-layer` ran as a git worktree beside `main` (a30c0e3; its
pynicamdc/ is identical to 70ec5d4), both arms sharing the same
`libncclffi.so` and the same allocation.

| check | result |
|---|---|
| fused fp32 final state, `main` vs `api-layer` — **on GPU, NCCL-FFI wire** | ~~bit-exact, 4/4 ranks (equal_nan on the halo/pad NaNs)~~ **INVALID (2026-08-20): both states were all-NaN; see READ FIRST** |
| production perf lstep=43, `main` vs `api-layer`, same allocation | 0.3020/0.3183 vs 0.3023/0.3185 s/step (min/mean) — **+0.1% / +0.06%, below run-to-run spread** |
| historical reference (main, job 1374619) | 0.3028/0.3120 — consistent |

So the object API costs nothing measurable on the production GPU path, and
the refactored driver reproduces `main` bit for bit through the fused
NCCL-FFI pipeline on a second GPU architecture (Hopper, after Levante's
A100/Ampere).

**Still owed on JUPITER:** the `set_at` step-time and cap sweeps, and
production-scale K/resolution/multinode timing on this branch (the main-branch
equivalents are recorded in `tools/jupiter/SCALING-LADDER.md` and the xspies
comparison campaign).

---

## GH200 measurements — the fusion-schedule decisions (JUPITER, 2026-08-19)

The two numbers `FUSION_SCHEDULE_PLAN.md` said must exist before the warm-up and
cap choices, plus the K study they triggered. All on the xspies campaign stack
(Stages/2026, ParaStationMPI 5.13.0-1, jax 0.10.2, 1 rank/GPU), `api-layer` tree,
production fusion env, no history output.

**Per-step vs fused, and the cap sweep (gl09 pe4 fp32, job 1409479).** Per-step
measured by the difference method — `FUSE_TIMELOOP=0` at lstep 3 and lstep 163
pay identical jit compiles, so (Loop₁₆₃ − Loop₃)/160 cancels them exactly:

| path | s/step |
|---|---|
| per-step (`FUSE_TIMELOOP=0`) | **0.3643** |
| fused K=1 / 2 / 4 / 8 / 16 (steady mean) | 0.3091 / 0.3090 / 0.3085 / 0.3079 / 0.3064 |

Fusion is worth 18%; per-step − fused = 0.056 s/step; K is flat (K=16 buys 0.9%
over K=1). → warm-up option **(A)** wins by ~300× over (B) and unconditionally
over (C); see the plan.

**K at scale (gl11 rl05 pe1024 hilbert fp32, 256 nodes, lstep=159).** Two
single-run comparisons gave *opposite* orderings (K=1 beat K=12 by 7% in job
1409610; the reverse in 1409895) — single-run means at this scale carry the
±7% transient-episode noise documented in the xspies campaign. The interleaved
test (job 1409895: K=1/4/12 twice, alternating) shows why: the quiet-state rate
is **identical for every K** (38–40 ms/step), and per-step traces at K=1
resolve the arm-to-arm differences into discrete fabric episodes (a ~1 s
stretch of 60–80 ms steps, isolated 1–2-step spikes) uncorrelated with K.
Floor 33.5 ms/step, reproducing the campaign's 0.0333.

**Conclusion recorded in the plan:** K is performance-neutral on GH200 at both
ends of the regime; default K=1 (schedule trivially consistent, per-step
observability), cap knob retained for slow-dispatch hosts; warmup = K.

**Step-time neutrality at production scale (gl11 rl05 pe1024 hilbert fp32,
K=1, 256 nodes, job 1410187, 2026-08-19).** The host-overhead-worst case: 33 ms
steps, one dispatch per step. Interleaved main→api→main→api (lstep=159):
main 0.0391/0.0414, api-layer 0.0406/0.0388 s/step — tree difference −1.2%,
within-tree repeat spread 4.5–5.7%, floors identical (0.0323–0.0327). The
object API adds nothing measurable even at maximum dispatch rate; the
"production-scale step time" debt is paid at pe1024. (Interleaving matters:
the campaign measured ±7% single-run noise at this scale, and two single-run
K comparisons had previously flipped sign.)

## GH200 verification — the fusion schedule, S2 and S3 (JUPITER, 2026-08-19/20)

The "S2 Verify" and "S3" checks from `FUSION_SCHEDULE_PLAN.md`, run on the same
xspies stack as above. Case: gl09 pe4 fp32, lstep=24, `PRGout_interval=12` and
`MNT_INTV=6` both active, history output **on**, cap `PYNICAM_TIMELOOP_CHUNK=4`
→ the resolver must choose K=3 (the largest divisor of gcd(12, 12, 6) ≤ 4) and
warm-up = K. Harness: `sweep/jupiter_gl09_s2_verify.sbatch`,
`sweep/jupiter_gl09_s3_check.sbatch`.

**S2 — resolver, boundary-aligned chunks, one compiled graph (job 1411545,
commit `2e06340`).** Fused vs `FUSE_TIMELOOP=0`, same case:

| check | result |
|---|---|
| resolver line (rank 0) | `K=3 (cap=4, active intervals=[12, 12, 6]), warm-up=3 [= K]` |
| chunk histogram | every chunk K=3; chunk walls 0.93–1.01 s across both write boundaries (no recompile) |
| `BUDGET_energy.log`, `BUDGET_mass.log` | ~~byte-identical fused vs per-step~~ **INVALID pre-fix: both were NAN from the first MNT line** |
| zarr history | ~~values identical~~ **INVALID pre-fix: slot 1 was 100 % NaN, slot 0 28 % NaN; `equal_nan` made them "identical"**. (The blosc remark stands: `diff -r` on zarr trees is not a data comparison.) |

**S3 — production defaults and countable compiles (jobs 1431497, 1432007,
commit `54cd0c9`).** S3 changes no numerics: the cap default becomes 1, warm-up
follows the resolver, the `tools/` templates that keep `CHUNK=4 WARMUP=3` say
so, and the two scan jits get names so `JAX_LOG_COMPILES` can count them. Same
case as S2, fused arm only, `JAX_LOG_COMPILES=1`:

| compile (per run, 4 ranks) | count | XLA wall / rank |
|---|---|---|
| `jit(_timeloop_chunk_scan)` — the fused K=3 chunk graph | **4** (= 1/rank) | 59.5–64.5 s |
| `jit(_nl_rk_scan)` — the per-step RK graph, compiled by the warm-up steps | **4** (= 1/rank) | 46.6–50.2 s |
| anonymous `jit(<lambda>)` | **0** | — |
| chunk histogram | 29 × K=3 | |
| `BUDGET_*.log` | ~~byte-identical to S2 fused and to per-step~~ **INVALID pre-fix (all NAN)** | |
| zarr history vs `FUSE_TIMELOOP=0` (S2 plain run) | ~~all 9 arrays bit-identical~~ **INVALID pre-fix: NaN == NaN**. Re-done after the fix below. | |

So a fused run compiles exactly two large graphs per rank — the warm-up's
per-step graph and the chunk graph — and nothing recompiles at the output or
budget boundaries. The ~1.6 s "tracing" and the *A large amount of constants
were captured during lowering (4.00GB)* warning that accompany both are the
resident-constant design (see "What a compile costs"), unchanged by S2/S3.

Two things this does not cover: the S3 default path itself (cap=1 → K=1) at
gl09 — K=1 was measured in the cap sweep above and is the per-step-observable
form of the same graph, but a K=1 run with output on has not been diffed
against `FUSE_TIMELOOP=0` here; and the per-step `run(n)` split (a tail shorter
than K runs per-step), which is covered by the CPU unit suite only.

### Re-verified after the pole-vertex fix (`cc92845`, JUPITER, 2026-08-20)

Same S2 case (gl09 rl01 pe4 fp32, lstep 24, `PRGout_interval=12`, `MNT_INTV=6`,
cap 4 → K=3), jobs 1433270 (S3 check) and 1433324 (S2 verify, fused + plain),
this time with a finiteness check first:

| check | result |
|---|---|
| `BUDGET_energy.log` / `BUDGET_mass.log`, fused | **finite at every line**; residuals 1e-3…1e-2 W/m², the same order as f90 on this case (job 1432550) |
| BUDGET fused vs per-step | byte-identical |
| zarr, all 9 arrays | **0 NaN** in either store; fused vs per-step **bit-identical under strict `np.array_equal`** (no `equal_nan`) |
| resolver / chunks | `K=3 (cap=4, active intervals=[12, 12, 6]), warm-up=3 [= K]`; 29 × K=3 |
| compiles (S3 script) | `jit(_timeloop_chunk_scan)` 4, `jit(_nl_rk_scan)` 4, anonymous 0 — unchanged |

So the fusion-schedule claims of S2/S3 now stand on a real state: the fused
path reproduces the per-step path bit for bit at gl09 pe4 with output on and
both output and budget boundaries inside the run.


### `main` vs `api-layer` A/B redone on a finite state (JUPITER, 2026-08-20, job 1433457)

Same protocol as the 2026-08-16 section (`sweep/jupiter_gl09_apilayer_ab.sbatch`,
now with an interior-finiteness gate before the bit-exact test), both trees
carrying the pole-vertex fix (`main` 35eab30, `api-layer` cc92845):

| check | result |
|---|---|
| fused fp32 final state (JIT=1, K=4, lstep 12), interior | **finite on all 4 ranks**, 0 NaN; RHOG 0.002–1.55, \|V\|max 35.0–35.5 m/s (the JW jet) |
| `main` vs `api-layer` | **bit-exact, 4/4 ranks** |
| perf lstep 43, same allocation (min/mean s/step) | main 0.3309/0.3322, api-layer 0.3317/0.3329 → **+0.2 %** |
| vs the pre-fix "historical" 0.3028/0.3120 (job 1374619) | ~7–9 % slower. One run cannot separate this from node-to-node spread (±7 % in the campaign); the pre-fix number was measured on a NaN state. Treat 0.33 s/step as the current gl09 pe4 reference until an interleaved repeat says otherwise. |

**Still to redo on a valid state:** the 2026-08-19 timing numbers (per-step vs fused, the K sweeps, pe1024
step-time neutrality) — expected unchanged, but measured on NaN — and a
sanity pass over every rung of `SCALING-LADDER.md` with `MNT_INTV=1`.

### Why the pre-fix timings were faster: a NaN state runs ~9 % faster on GH200 (job 1433582)

Interleaved on one node (jpbo-019-12), pre-fix tree (`cc92845^`, state NaN from
step 1) vs fixed HEAD, gl09 pe4 fp32 fused K=4, lstep 43, `nvidia-smi` sampled
every 2 s (`sweep/jupiter_gl09_nanperf_interleave.sbatch`):

| arm | state | per-step, fastest half | min |
|---|---|---|---|
| pre a / b | NaN | **0.3153 / 0.3148** | 0.3017 / 0.3014 |
| fix a / b | finite | 0.3436 / 0.3432 | 0.3247 / 0.3267 |

Reproducible to 0.1 %: **real data is 9 % slower than NaN**. GPUs do not slow
down on NaN (no traps, no assists), so the asymmetry is the finite state
being *more expensive*: SM clock dips (1650–1965 MHz) appeared only in the
finite arms (power/thermal), but fix_a held ~1980 MHz and was still 9 % slower,
so clocks are at most part of it. The remaining candidate is the
special-function slow paths (IEEE fp32 divide / sqrt / `pow` / `exp` take an
early exit on NaN operands and the full sequence on real ones); a per-kernel
profile (nsys) would settle it. Consequences:

- every JUPITER timing of an rl01 pe4 case measured before `cc92845` is
  ~9 % optimistic: the 2026-08-19 per-step vs fused (0.3643 / 0.3085) and the
  cap sweep in particular. Ratios are expected to survive; absolute values
  must be re-measured.
- the gl11–gl13 rl05 ladder (`SCALING-LADDER.md`, the xspies campaign) was
  measured on finite states and is unaffected.
- current gl09 pe4 fp32 fused reference: **0.343 s/step** (fastest-half mean),
  0.325 min.

### The rl05 ladder really was unaffected — checked, not assumed (gl11 rl05 pe256, job 1436216)

64 nodes, lstep 12, `MNT_INTV=1`, all-256-rank `check_fleet_sanity.py`, pre-fix
tree (`cc92845^`) and fixed `main` (35eab30) on the same allocation
(`sweep/jupiter_gl11_rl05_polecheck.sbatch`):

| | pre-fix | fixed |
|---|---|---|
| ranks done | 256/256 | 256/256 |
| BUDGET_energy | finite at all 12 steps, residual O(1) W/m² (1e-9 of the total) | steps 0–3 identical to pre-fix to every digit (`main` has no S1/S2, so the fused chunks hide the later MNT lines — a display limitation of `main`, not a numerical one) |
| all-rank sanity | RHOG 2.57e-3..1.55, \|V\|max 35.598 m/s, tracer ≡ 0, 0 problem ranks | identical |

So on rl05 with contiguous region→rank assignment the five pole-adjacent
regions of each hemisphere live on five different ranks (they sit in five
different diamonds, 1024 region ids apart) and the single-buffer overwrite
never happened. The campaign ladder (`SCALING-LADDER.md`, gl11–gl13 rl05) and
its timings stand. Only decompositions that put ≥2 pole-adjacent regions on
one rank were broken — on JUPITER that was every rl01 pe4 case.
