# Plan — derive the fusion chunk from the output schedule

`PYNICAM_FUSE_TIMELOOP` advances the prognostic in K-step chunks, each chunk one
`jax.lax.scan` graph compiled once per K. The chunk length is set by hand
(`PYNICAM_TIMELOOP_CHUNK`, 4 everywhere in `tools/`) and then trimmed so it never
spans an output step. That combination works only because every GPU benchmark
disables output — each of `tools/{miyabi,levante,jupiter,ncclffi,rocm_gl05_kit}`
sets `PRGout_interval` to 1000 or 999999. With output on, the trim makes K vary,
and K varying is expensive.

This plan makes K a consequence of the output schedule instead of an independent
knob, so that a production run compiles the chunk graph **once**.

---

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

Replaying the trim over `lstep_max=144, PRGout_interval=12, MNT_INTV=72, cap=4`
gives K alternating 4,3,4,3,… — **22 compiles**, one per chunk pair, of a graph
whose whole purpose is to be compiled once.

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

## The rule

Let the **boundary intervals** be the active ones among `PRGout_interval`,
`PRGout_interval_2d` (when diagnostics are on) and `MNT_INTV` (when `MNT_ON`),
dropping any that never fire within `lstep_max`. Every boundary step is then a
multiple of `g = gcd(intervals)`, so every gap between boundaries is a multiple of
`g`.

```
K      = max{ d <= cap : g mod d == 0 }        cap = PYNICAM_TIMELOOP_CHUNK
warmup = K
```

`K | g` makes every interval tile exactly; `warmup = K` puts the phase on a
multiple of K, after which every boundary lands on a chunk end. K never changes,
so the graph compiles once.

With no active interval (the benchmark configuration) there is nothing to align to
but the end of the run: `K = cap`, `warmup = lstep_max mod K` (or `K` when that
is 0).

Replaying the trim with this rule — every case one compile, every boundary on a
chunk end:

| case (cap = 4) | K | warmup | fused | compiles |
|---|---|---|---|---|
| no output, lstep 43 | 4 | 3 | 40/43 | 1 |
| interval 12, lstep 48 | 4 | 4 | 44/48 | 1 |
| interval 12 + MNT 72, lstep 144 | 4 | 4 | 140/144 | 1 |
| interval 72 + 2D 24, lstep 144 | 4 | 4 | 140/144 | 1 |
| interval 6, lstep 48 | **3** | 3 | 45/48 | 1 |
| interval 10, cap 4, lstep 100 | **2** | 2 | 98/100 | 1 |

The last two are why K must divide `g` rather than just be capped by it: 4 does not
divide 6 or 10, and choosing it there would reintroduce a ragged chunk.

Trading three extra per-step steps (warm-up 1 → 4) for one fewer compile of a
K=4 fused graph is heavily favourable at gl11 scale.

---

## Steps

Each lands separately and is separately verifiable. S1, S2 and S5 do not change
what any currently-working configuration computes; S3 does, and is last.

### S1 — one predicate for "the host must see this step"

`share/output_schedule.py` gains a boundary predicate over a set of intervals,
next to `prg_output_fires`. The chunk trim in `pyNICAM.run` uses it instead of the
two output predicates alone, so `MNT_INTV` becomes a chunk boundary.

Fixes #3. Nothing else changes: on numpy no chunk is ever taken, and on jax a
boundary can only ever shorten a chunk.

*Verify:* numpy A/B bit-exact. jax fused run with `MNT_ON=true` — the budget lines
now appear at every `MNT_INTV`, and match a `FUSE_TIMELOOP=0` run of the same case.

### S2 — derive K and warm-up at setup

`pyNICAM._resolve_loop_options` computes K and warm-up by the rule above from
`io.PRGout_interval`, `io.PRGout_interval_2d`, `embudget.MNT_INTV/MNT_ON` and
`tim.TIME_lstep_max`. `PYNICAM_TIMELOOP_CHUNK` becomes the **cap**;
`PYNICAM_TIMELOOP_WARMUP` becomes an override for development.

**Log the resolved K and warm-up** on rank 0. They now move when the output
interval moves, and a performance change nobody can attribute is worse than a
knob nobody sets.

*Verify:* unit test on the resolver (pure integer arithmetic, no model) covering
the table above. numpy A/B bit-exact. jax fused: `JAX_LOG_COMPILES=1` shows one
compile of the scan graph.

### S3 — end the chunk **at** the boundary step, not before

`_K = _j` becomes `_K = _j + 1`, and after the chunk the driver fires the output
and/or budget monitor for that step.

Fixes #2 — no step has to run unfused for the host's benefit. This is safe because
both consumers drain first: `write()` calls `sync_prgvar_to_host`, and
`embudget_monitor` calls it too (`mod_embudget.py:71`). `history_vars_step` reads
`msc.prgv.PRG_var` and static state only — no per-step host forcing state.

**This step changes which code path produces the state that gets written.** It is
sound exactly to the extent that the fused path is bit-identical to the per-step
path, which is the premise of the whole fused stack (and what
`PYNICAM_TIMELOOP_JIT=0` exists to check) but is not implied by the earlier steps.
Land it separately so a regression here is unambiguous.

*Verify:* same case with `FUSE_TIMELOOP=1` vs `=0`, comparing the **zarr output**,
not just the end-of-run dump — the point of the change is which path wrote those
snapshots. Assert the chunk actually engaged (`TIMELOOP_CHUNK` lines present).

### S4 — production defaults

Warm-up follows S2 rather than the env default of 3. Update `tools/` templates:
they set `PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_WARMUP=3` explicitly, which
after S2 would pin the value the resolver should be choosing.

### S5 — key the scan cache by K (insurance, ~1 line)

`self._timeloop_scan_jit` becomes a dict. After S2 K is constant and this changes
nothing; it bounds the damage to "one compile per distinct K" if a future
configuration produces a ragged chunk anyway.

---

## Verification split

**On CPU (here):** the resolver unit test; numpy A/B for S1/S2/S5 (bit-exact —
numpy never fuses); the jax host-staged path (`PYNICAM_RESIDENT=0`) for S1-S2, but
note it does not exercise `_step_core` at all, so it cannot check S3.

**Only on GPU (Miyabi / Levante):** everything that matters here. Compile counts
(`JAX_LOG_COMPILES=1`), the S3 fused-vs-unfused output comparison, and the actual
step-time gain. See `docs/GPU_VERIFICATION.md`.

**Benchmark comparability:** the existing GPU numbers were all measured with output
disabled, where this plan changes nothing (K = cap either way, one compile either
way). Runs *with* output are the ones that improve, and there is no historical
number to compare them against — they have never been measured.

## Open questions

- **Is the fused path bit-identical to the per-step path in practice?** S3 depends
  on it. There is a validation hook for the end state
  (`PYNICAM_TIMELOOP_DUMP`), but the fused-vs-per-step comparison of *written
  output* has not been run.
- **Should `write()` outside the schedule force a chunk boundary?** An ad-hoc
  `pyNICAM.write()` between `run()` calls already lands on a chunk end, because
  `run()` stops there. A `write()` from inside a callback would not — there is no
  such callback today, so this stays out of scope until there is.
- **`DYN_DIV_NUM > 1` and `trcadv_out_dyndiv`** disable `_step_core` entirely
  (`mod_dynamics.py:2755`), so fusion never engages and the resolver's output is
  unused. Worth an early exit and a log line rather than silently computing K.
