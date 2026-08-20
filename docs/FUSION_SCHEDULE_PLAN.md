# Plan — derive the fusion chunk from the output schedule

`PYNICAM_FUSE_TIMELOOP` advances the prognostic in K-step chunks, each chunk one
`jax.lax.scan` graph compiled once per K. The chunk length is set by hand
(`PYNICAM_TIMELOOP_CHUNK`, 4 everywhere in `tools/`) and then trimmed so it never
spans an output step. That combination works only because every GPU benchmark
disables output — each of `tools/{miyabi,levante,jupiter,ncclffi,rocm_gl05_kit}`
sets `PRGout_interval` to 1000 or 999999. With output on, the trim makes K vary,
and K varying is expensive.

This plan makes K a consequence of the output schedule instead of an independent
knob, and makes it **single-valued by construction** — one compiled chunk graph for
the whole run, in every configuration. Device memory is a hard constraint on this
model, so the goal is one graph, not a cache of graphs.

**Decided 2026-08-19, after the GH200 measurements below: the default is K = 1.**
Chunk length is performance-neutral on GH200 at both ends of the regime (the cap
sweep and the pe1024 interleave, "Choosing the cap"), and at K = 1 every step is a
boundary, so single-valuedness, phase alignment and the no-short-chunks rule hold
trivially, `run(n)` splits anywhere, prime intervals stop being a hazard, and
per-step timing stays observable. The divisor rule below remains the resolver's
behaviour **when the operator raises the cap above 1** — an opt-in for
slow-dispatch hosts or future per-step host work, not the default path.

---

## The regime this targets

At the resolutions this model is built for, the large step is a few seconds and
output every few hundred steps is ordinary. That changes which of the problems
below is worth solving.

Take a simulated day at a 2-second step — 43,200 large steps — writing output every
300:

| | fused | recompiles |
|---|---|---|
| today (cap 4, trim before the boundary) | 99.66% | **286** |
| this plan | 99.99% | **1** |

The unfused step per output interval, problem #2 below, is **negligible here**: one
step in 300. The original motivation for this plan — getting those steps fused —
mostly evaporates at production resolution.

What does not evaporate is #1. Two recompiles per output interval, forever: 286 for
a single simulated day, 142 at interval 600, 192 at interval 450 (where K also
collapses to 1 for part of each interval).

A recompile is **16–24 s per rank**, measured: `tools/jupiter/SCALING-LADDER.md:148`
records `CHUNK=6` against `lstep_max - WARMUP = 40` leaving a ragged `K=4` chunk that
"JIT-compiles a *second* time (~16-24 s/rank), so 2 of 7 chunks per rank are
compile-dominated". At 286 recompiles that is **76–114 minutes of compilation per
simulated day**, against 16–24 s once under this plan.

The same order holds on the CPU backend, measured locally at gl05 pe8 `K=2`: the
first chunk takes 20.33 s wall, every later one 1.04–1.10 s — ~19×, and 40% of the
whole 12-step `Main_Loop`. The cost is not a GPU artefact.

That note is also the same problem found empirically from the other end: it
prescribes that `TIMELOOP_CHUNK` divide `(lstep_max - WARMUP)` exactly. This plan
generalises that rule from the end of the run to **every** host-visible boundary,
and has the resolver apply it instead of the operator.

**So the value of this plan at production resolution is almost entirely
286 → 1.** The throughput arguments are for the low-resolution / short-interval
cases, where they are real but small.

## What is wrong today

**1. K varies, and the scan cache holds one entry.**
`mod_dynamics.py:819-825` rebuilds the jit whenever K differs from the previous
chunk:

```python
_cache = getattr(self, "_timeloop_scan_jit", None)
if _cache is None or _cache[0] != K:
    _fn = jax.jit(lambda _c: jax.lax.scan(_scan_body, _c, xp.arange(K))[0])
    self._timeloop_scan_jit = (K, _fn)
```

A fresh `jax.jit` over a fresh lambda is a function XLA has never seen, so this is
a full recompile of the K-step graph, not a cache lookup. Replaying the trim over
`lstep_max=144, PRGout_interval=12, MNT_INTV=72, cap=4` gives K alternating
4,3,4,3,… — **22 recompiles** of a graph whose whole purpose is to be compiled once.

**2. One step per output interval cannot fuse.** The trim stops the chunk *before*
the output step (`_K = _j`), so that step runs per-step: a floor of `1/interval`
unfused steps no matter how large K is.

**3. The budget monitor is silently skipped.** The trim only knows about output
(`_is_out_3d` / `_is_out_2d`). `embudget_monitor` fires on
`TIME_cstep % MNT_INTV == 0` (`mod_embudget.py:67`) and is not called at all from
the fused branch — so with `MNT_ON` a monitor step that lands inside a chunk is
dropped without a word. Pre-existing on `main`; the driver rewrite carried it over
faithfully.

**4. Warm-up is longer than it needs to be in production.** `PYNICAM_TIMELOOP_WARMUP=3`
separates the JIT-compile-heavy first steps for measurement. Only **one** step is
structurally required: `_step_core` is built inside `dynamics_step`
(`mod_dynamics.py:2752`) from loop-body locals (the frozen `DIAG` seed,
`large_step_dt`), and the driver guards on `dyn._step_core is not None`, so step 0
always runs per-step regardless of the warm-up setting.

**5. The steady-state timing formula includes what it claims to exclude.**
`Main_Loop_step1` wraps only `n == 0` (`api.py:593,659`), and the comment beside it
gives `steady per-step = (Main_Loop - Main_Loop_step1) / (lstep_max - 1)`. With
warm-up 3 and fusion on, that leaves the *other two* per-step warm-up steps **and
the first chunk's scan compile** inside the "steady" average. The bias shrinks as
`lstep_max` grows, so PROF per-step numbers from runs of different length are not
comparable with each other. `tools/sweep/timing_hires.pbs` sidesteps this
independently — `dts[3:]`, drop >5 s, take the median — which is why the sweep
numbers are usable and the PROF ones are not. Independent of this plan, but it is
the instrument any of it would be judged with.

## The rule

A **boundary step** is one after which the host must see the state: the 3D output
fires, the 2D output fires, or the budget monitor fires. A chunk may not run past
one.

Let the **boundary intervals** be the active ones among `PRGout_interval`,
`PRGout_interval_2d` (when diagnostics are on) and `MNT_INTV` (when `MNT_ON`),
dropping any that never fire within `lstep_max`. Every boundary step is a multiple
of `g = gcd(intervals)`, so every gap between boundaries is a multiple of `g`.

```
cap     = PYNICAM_TIMELOOP_CHUNK, default 1
K       = 1                                   if cap == 1 (the default)
        = max{ d <= cap : g mod d == 0 }      if cap > 1 (opt-in)
          (no active interval -> K = cap; there is no boundary to align to)
warmup  = K
chunks  are ALWAYS exactly K steps. Never shorter.
```

At the default cap of 1 all three parts below hold trivially (g mod 1 == 0 for
every schedule, the phase is always aligned, no leftover can exist) and the
resolver degenerates to a constant. The arithmetic matters only for cap > 1:

Three parts, and all three are needed:

- **`K | g`** makes every gap between boundaries tile exactly.
- **`warmup = K`** puts the phase on a multiple of K, so every boundary lands on a
  chunk end rather than mid-chunk.
- **no short chunks** covers what the first two cannot: the tail at `lstep_max`, and
  the tail of a `run(n)` call whose `n` is not a multiple of K. Those leftovers run
  per-step instead of becoming a shorter chunk.

Together these make K **single-valued by construction**: `run_timeloop_chunk` is
only ever called with the one K resolved at setup. One compiled graph, one
allocation of its constants and workspace, for any configuration and any call
pattern. Nothing needs to be cached "just in case" — the earlier draft of this plan
proposed keying the scan cache by K, and that is dropped: holding N compiled graphs
to absorb a K that varies is paying device memory to avoid fixing the schedule.

The cost is bounded: at most `K-1` per-step steps at each leftover, plus the K
warm-up steps.

Replayed over the trim — one compile everywhere, K never varying:

| case (cap 4 unless noted) | K | fused | compiles |
|---|---|---|---|
| interval 12, lstep 144 | 4 | 140/144 | 1 |
| interval 12 + MNT 72, lstep 144 | 4 | 140/144 | 1 |
| interval 12 + 2D interval 8, lstep 144 | 4 | 140/144 | 1 |
| no output, lstep 43 (the benchmark) | 4 | 36/43 | 1 |
| interval 12, lstep 143 (ragged end) | 4 | 136/143 | 1 |
| interval 24, cap 8, lstep 144 | 8 | 136/144 | 1 |
| **interval 13, lstep 143** | **1** | — | 1 |

### When fusion cannot engage

The last row is the case to design for, not to discover in production. A boundary
spacing with no divisor `> 1` at or below the cap — a prime interval, or one
coprime to everything ≤ cap — resolves to `K = 1`, which is a one-step "chunk":
all of fusion's machinery and none of its benefit.

**Say so at setup**, on rank 0 and in the log: that fusion cannot engage with this
output interval, and what to change (an interval divisible by something ≤ cap).
Silently running K=1 would look like fusion is on while delivering nothing.

### Choosing the cap

`cap` is 4 in all 27 production invocations across `tools/`, 6 in
`tools/sweep/timing_hires.pbs`, 20 in the tutorial, 1 in the `JIT=0` diagnostic
runs, and 1 as the documented default in `docs/GATES.md`. **No sweep of K exists**:
nothing in the repository measures one chunk length against another. 4 appears to
be inheritance, not a result.

That mattered little while cap *was* the chunk length. Under this plan it selects K
from the divisors of the boundary spacing, and at production resolution the
spacing is large, so cap alone decides how finely a long interval gets cut:

| cap | K (interval 300) | chunks per output interval |
|---|---|---|
| 4 | 4 | 75 |
| 12 | 12 | 25 |
| 50 | 50 | 6 |
| 300 | 300 | 1 |

**How much that is worth is a smaller question than it looks.** `jax.lax.scan`
defaults to `unroll=1`, so the body is compiled once and looped: the graph for K=4
and the graph for K=400 are the *same graph* with a different trip count. There is
no cross-step optimisation to unlock by raising K, and `_scan_body` returns `None`
as its per-iteration output, so nothing is stacked and device memory does not grow
with K either. All a larger K removes is per-chunk host overhead — one jit
dispatch, one `block_until_ready`, two PROF timers, the idempotent
`_ensure_forcing_caches`, and K cheap `TIME_advance` calls. Order 0.1–1 ms.

Against that, the per-step time recorded for gl11 pe64 z40 is 0.31–0.33 s
(`mod_dynamics.py`, job 2439682), so a K=4 chunk is ~1.25 s of device work and the
overhead is ~0.1% of it. Running 400 steps as 100 chunks of 4 rather than one chunk
of 400 should therefore cost ~0.1 s out of ~125 s.

So cap matters where the per-step time is *small* — low resolution, small grids,
where a millisecond of overhead against a few milliseconds of work is a real
fraction. At the resolutions this plan targets it is a rounding error, and the
recompiles are the whole prize. Sweep cap when convenient, not first.

**Swept 2026-08-19 (JUPITER GH200; see GPU_VERIFICATION.md for the runs).** The
prediction above holds, and more strongly than written:

- gl09 pe4 fp32 (0.31 s steps): K = 1/2/4/8/16 → 0.3091/0.3090/0.3085/0.3079/
  0.3064 s/step. K=16 buys 0.9% over K=1.
- gl11 rl05 pe1024 hilbert fp32 (0.033 s steps — the *small-step* regime where K
  was supposed to matter): an interleaved K=1/4/12 ×2 test shows the quiet-state
  rate **identical for every K** (38–40 ms/step), with arm-to-arm differences
  (±5%) entirely attributable to transient fabric episodes (±7% run-to-run) —
  two single-run orderings flipped sign before the interleave settled it. The
  per-chunk dispatch overhead evidently hides behind the asynchronous device
  queue even at 33 ms steps.

**Consequence — default K = 1, cap becomes an opt-in.** With K measured
performance-neutral on GH200 at both ends of the regime, K=1 is the schedule's
fixed point: every step is a boundary, so the K|g rule, the warm-up phase
alignment, and the no-short-chunks rule are all trivially satisfied, `run(n)`
works for any n, prime output intervals stop being a designed-for hazard, and
per-step timing stays observable (the episode diagnosis above was only possible
at K=1). The resolver machinery in S2 remains correct but becomes the K>1
opt-in path, not the default. Caveats that would reopen the question: hosts
with slower dispatch than Grace (e.g. MPS-oversubscribed Miyabi), and any
future change that adds real host work per step (an io_callback halo
transport, per-step callbacks). Keep `PYNICAM_TIMELOOP_CHUNK` as the cap knob
for those cases; K=4 remains fine for continuity with existing measurements.

### Warm-up: three options

`warmup = K` above is the simplest rule that aligns the phase, but it is not the
only one, and the choice only matters once K is large. All three produce the same
K and the same single compiled graph; they differ in what the first K steps cost
and in what a measurement has to skip.

**(A) warm-up = K.** The rule as stated. The K steps are *not wasted* — they are
real steps that advance the model, just on the per-step path instead of the fused
one, so the cost is `K × (per-step − fused)`, not K whole steps.

**(B) warm-up = 1, first chunk absorbs the phase.** One chunk of a different length
at the start, then the steady K forever. Costs one extra compile at startup and
**no extra resident graph** — the single-entry cache drops the first one when the
steady K arrives. Only one step runs unfused.

**(C) spin and rewind.** Run a few steps to build `_step_core`, restore the
prognostic and the clock, then integrate from step 0 in K-sized chunks. The spun
steps are genuinely discarded, so the cost is whole steps rather than a speed
difference — at K=4 that is *more* compute than (A), and it only pays off once
`K × (per-step − fused) > spun steps`.

| | warm-up | one-off cost | a measurement must skip | extra machinery |
|---|---|---|---|---|
| (A) | K | K × (per-step − fused) | the K warm-up steps **and** the first chunk's compile | none |
| (B) | 1 | one startup compile | 1 step and the first chunk's compile | small |
| (C) | 0 (spun steps discarded) | the spun steps, whole | the first chunk's compile | state save/restore |

(C)'s appeal is not throughput. It is that the procedure states itself in one
sentence, that K's cost stops depending on the warm-up, and that the measured
region becomes uniform from step 0 — one thing to skip, at a fixed place, instead
of two whose position moves with the warm-up setting. That is what `dts[3:]` in the
sweep script is: a magic number coupled to `PYNICAM_TIMELOOP_WARMUP=3`.

Note what (C) does **not** remove: `K | g` still holds, because chunks still have
to tile the gap between boundaries; and the first chunk still carries the scan
compile, so it is still an outlier. Spinning long enough to also warm the scan graph
would need `1 + K` discarded steps, which at large K is worse than (A).

**(C)'s risk is correctness, not cost.** The spin leaves host-side state the steady
path depends on: `_step_core` bakes in the host `DIAG` as a constant
(`mod_dynamics.py:2768`), justified by DIAG being invariant across steps under the
drain-once policy — while the host `DIAG_pl` drain is deliberately *kept during
warm-up* and skipped afterwards (`mod_dynamics.py:1078`). Rewinding the prognostic
while keeping those caches means the frozen seed describes a state that no longer
exists. The code argues DIAG is only a seed for boundary rows that get overwritten
(`mod_dynamics.py:520`); that argument, not the rewind machinery, is what would
have to be established first.

**What decides this.** Two numbers, neither of which exists in the repository:

1. the cap sweep — how much a larger K is worth at all;
2. **per-step vs fused step time** — the whole cost of (A), and the break-even for
   (C). Nothing in the repo measures `FUSE_TIMELOOP=0` against `=1`; the only
   recorded comparison is `JIT=0` vs `JIT=1`, both inside the fused stack.

(2) is the more fundamental and the easier to get: the same case, twice, one
environment variable apart. It should be measured before this choice is made.

**Measured 2026-08-19 (JUPITER GH200, gl09 pe4 fp32; GPU_VERIFICATION.md)** — *re-measured 2026-08-21 on a finite state after the pole-vertex fix: per-step 0.3826, fused 0.3337 (K=4), gain 12.8 %, per-step − fused 0.049 s, K flat to 0.1 %; the conclusions below are unchanged, the absolute numbers below are the superseded NaN-state ones*:

- per-step (`FUSE_TIMELOOP=0`): **0.3643 s/step**, by the difference method —
  lstep 3 and 163 both pay the identical jit compiles, so
  (Loop₁₆₃ − Loop₃)/160 cancels them exactly.
- fused: 0.3085 s/step (K=4) → fusion is worth **18%**, and
  per-step − fused = **0.056 s/step**.
- the cap sweep is flat (see "Choosing the cap").

So the decision closes in favour of **(A) warmup = K**: its one-off cost is
K × 0.056 s ≈ 0.2 s at K=4 — and with K=1 now the recommended default, zero.
(B)'s extra startup compile is ~67 s ≈ 300× that cost; (C)'s break-even
(per-step − fused > per-step) is unreachable, and its DIAG-rewind risk buys
nothing. S1/S2 are identical under all three; the resolver emits warmup = K.

---

## Steps

S1 and S3 change nothing that works today. S2 is the change.

### S1 — one predicate for the boundary steps

`share/output_schedule.py` gains a boundary predicate over a set of intervals, next
to `prg_output_fires`. The chunk trim in `pyNICAM.run` uses it instead of the two
output predicates alone, so `MNT_INTV` becomes a chunk boundary.

Fixes #3. Nothing else changes: on numpy no chunk is ever taken, and on jax a
boundary can only ever shorten a chunk.

*Verify:* numpy A/B bit-exact. jax fused run with `MNT_ON=true` — the budget lines
now appear at every `MNT_INTV`, and match a `FUSE_TIMELOOP=0` run of the same case.

### S2 — resolve K and warm-up, end chunks at the boundary, refuse short chunks

Three halves of one change; none of them delivers anything alone.

**(a) Resolve at setup.** `pyNICAM._resolve_loop_options` computes K and warm-up by
the rule above from `io.PRGout_interval`, `io.PRGout_interval_2d`,
`embudget.MNT_INTV/MNT_ON` and `tim.TIME_lstep_max`. `PYNICAM_TIMELOOP_CHUNK`
becomes the **cap** (default 1, so the resolver returns K=1 unless raised);
`PYNICAM_TIMELOOP_WARMUP` becomes a development override.
Emit the resolved K and warm-up, and the "fusion cannot engage" diagnosis, on rank
0 — they now move when the output interval moves, and a performance change nobody
can attribute is worse than a knob nobody sets.

**(b) End the chunk at the boundary step.** The trim currently stops before it:

```python
if _is_out_3d(n + _j) or _is_out_2d(n + _j):
    _K = _j          # the boundary step itself is left to the per-step path
```

`_K = _j + 1` instead, and the driver fires the output / budget monitor for that
step after the chunk returns. Safe because both consumers drain first: `write()`
calls `sync_prgvar_to_host`, and so does `embudget_monitor`
(`mod_embudget.py:71`); `history_vars_step` reads `msc.prgv.PRG_var` and static
state only, never per-step host forcing state.

**(c) Refuse a chunk shorter than K.** Fewer than K steps left before the end of the
run or the end of this `run(n)` call → run them per-step. This is what makes K
single-valued in the cases (a) and (b) cannot reach.

**Why they are inseparable.** (a) aligns the phase so boundaries fall on chunk ends
— but with (b) missing the trim still cuts one step short of every boundary, so K
alternates exactly as it does today and the alignment buys nothing:

| case (cap 4) | (a) alone | (a)+(b)+(c) |
|---|---|---|
| interval 12, lstep 48 | K = [3,4], 8 compiles | K = [4], **1** |
| interval 12 + MNT 72, lstep 144 | K = [3,4], 24 compiles | K = [4], **1** |
| interval 6, lstep 48 | K = [2,3], 15 compiles | K = [3], **1** |

**(b) is the one place in this plan where a regression is possible**: a boundary
step is now computed inside the fused chunk, so what gets written comes from the
fused path. That is sound exactly to the extent that the fused path is
bit-identical to the per-step path — the premise of the whole fused stack, and what
`PYNICAM_TIMELOOP_JIT=0` exists to check, but not something S1 establishes. Land S2
alone, after S1.

*Verify:* unit test on the resolver (pure integer arithmetic, no model) covering
the table above, including the K=1 diagnosis. numpy A/B bit-exact. On GPU:
`JAX_LOG_COMPILES=1` shows exactly one compile of the scan graph, and the same case
with `FUSE_TIMELOOP=1` vs `=0` must produce identical **zarr output** — not just
the end-of-run dump, since the point of (b) is which path wrote those snapshots.
Assert the chunk engaged (`TIMELOOP_CHUNK` lines under
`PYNICAM_PROFILE=timeloop_timing`), and check the K they report is the resolved one.

Running the fused path at all needs **three** gates, not one:
`PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1 PYNICAM_COMM_NO_BARRIER=1`. The
third is listed in `docs/GATES.md` as a diagnostic, but with `COMM_apply_barrier =
true` it is a precondition — the Python `PRC_MPIbarrier()` fires at *trace* time
under jit and desyncs ranks that trace differing COMM counts, deadlocking during
compile (`mod_comm.py:2011`). Omitting it hangs silently at ~100% CPU, looking like
a slow compile. Worth having the resolver detect and refuse this combination rather
than leaving it to whoever forgets the flag.

### S3 — production defaults

`PYNICAM_TIMELOOP_CHUNK` defaults to **1** (the 2026-08-19 decision) and warm-up
follows S2 rather than the env default of 3. Update the `tools/` templates: they
set `PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_WARMUP=3` explicitly, which after
S2 would pin the values the resolver should be choosing. Benchmark templates that
exist to reproduce historical numbers may keep `CHUNK=4` deliberately — that is
what the cap opt-in is for — but must say so.

---

## What this means from the API

The script does not change:

```python
nicam = pyNICAM("driversettings.toml")
nicam.initialize(parameters={"timeparam": {"lstep_max": 144},
                             "ioparam":   {"PRGout_interval": 12}})
nicam.run()
nicam.finalize()
```

K and the warm-up are resolved inside `initialize()`, so there is one fewer thing
to set. What does become visible is the **`run(n)` split**: `n` that is not a
multiple of K leaves a leftover, and by rule (c) that leftover runs per-step. K
stays single-valued and the compile count stays at 1 — the cost is throughput, not
compilation:

| call pattern (interval 12, lstep 144, K=4) | fused | compiles |
|---|---|---|
| `run()` once | 140/144 | 1 |
| `run(12)` × 12 | 140/144 | 1 |
| `run(10)` × 14 | 112/144 | 1 |
| `run(5)` × 28 | 92/144 | 1 |

So `pyNICAM` should expose the resolved chunk (`nicam.chunk`) for callers that want
to align:

```python
step = nicam.chunk * 3
while nicam.step < nicam.lstep_max:
    nicam = nicam.run(step)
    nicam.write()
```

None of this applies to the numpy backend, which never fuses: split `run()` however
is convenient there.

---

## Verification split

**On CPU (here):** the resolver unit test; numpy A/B for every step (bit-exact —
numpy never fuses); the jax host-staged path (`PYNICAM_RESIDENT=0`) for S1, but note
it does not exercise `_step_core` at all, so it cannot check S2(b).

**Only on GPU (Miyabi / Levante):** everything that matters here. Compile counts
(`JAX_LOG_COMPILES=1`), the S2(b) fused-vs-unfused output comparison, and the actual
step-time gain. See `docs/GPU_VERIFICATION.md`.

**Benchmark comparability:** the existing GPU numbers were all measured with output
disabled, where this plan changes little (K = cap either way, one compile either
way; only the tail treatment differs). Runs *with* output are the ones that improve,
and there is no historical number to compare them against — they have never been
measured.

## Open questions

- ~~**Is the fused path bit-identical to the per-step path in practice?**~~
  **Answered: yes, at small scale.** `FUSE_TIMELOOP=1` vs `=0` produce bit-identical
  zarr output — all 48 variables, on both `main` and `api-layer` — for JW gl05rl01
  z40, 8 ranks, 12 steps, with the chunks asserted to engage (24 firings at `K=2`).
  See `docs/GPU_VERIFICATION.md`. That removes S2(b)'s main risk. It does **not**
  cover production K or resolution, which is where a divergence would more likely
  come from; that check stays owed to a GPU.
- **Should an unscheduled `write()` force a boundary?** A `write()` between `run()`
  calls already lands on a chunk end, because `run()` stops there. A `write()` from
  inside a callback would not — there is no such callback today, so this stays out
  of scope until there is.
- ~~**Which warm-up scheme?**~~ **Answered: (A) warmup = K.** The blocking
  measurement now exists (JUPITER GH200, 2026-08-19): per-step 0.3643 vs fused
  0.3085 s/step, so (A) costs K × 0.056 s once, (B) costs a ~67 s extra compile,
  and (C) can never break even. See "Warm-up: three options" and
  GPU_VERIFICATION.md. The same session's K sweeps also settled the cap
  question: K is performance-neutral on GH200, and **K = 1 is the recommended
  default** (see "Choosing the cap").
- **`DYN_DIV_NUM > 1` and `trcadv_out_dyndiv`** disable `_step_core` entirely
  (`mod_dynamics.py:2755`), so fusion never engages and the resolver's output is
  unused. Worth an early exit and a log line rather than silently computing K.
