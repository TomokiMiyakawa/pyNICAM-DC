# Owed on GPU — `api-layer`

Two changes on this branch are verified on CPU but **not** on the path production
actually runs: the jax backend with device residency on (`PYNICAM_RESIDENT=1`, the
default). It could not be run on the laptop this work was done on — multi-rank jax
needs `mpi4jax`, which is not installed there — so the checks below are owed to
whoever next has Miyabi or Levante.

Nothing here is expected to fail. Both changes are argued to be no-ops on that
path, and the argument is written out per item. Run them anyway: CODING_POLICY §6
sets the bar at A/B'd, not at argued.

## Environment

`tools/miyabi/env.sh` + `setup_venv.sh`, or the Levante recipe in
`tools/levante/`. Dataset: `tutorial/download_inputs.sh` if `tutorial/case/` is
not populated.

## 1. `set_at`/`add_at`/`at_base` value dispatch — commit `585c948`

Dispatch moved from `bk.type == "jax"` to `bk.is_jax_value(a)`
(`pynicamdc/share/mod_backend.py`).

**Expected: EXACTLY 0.0.** On the resident path every value reaching these three is
a jax array or a tracer, and both are `jax.Array` instances, so the branch taken is
the same one as before. The only behavior that changes is on numpy values, which
that path does not produce.

```bash
# baseline = 79bb734 (main), candidate = 585c948, resident jax, same case
qsub tutorial/run_tier3_gpu.pbs        # or the A/B harness of your choice
python -c "import numpy as np; a=np.load('cand.npy'); b=np.load('base.npy'); \
           print(np.array_equal(a,b), np.max(np.abs(a-b)))"
```

Also confirm the fused path specifically — it is not covered by the default A/B,
and it traces **through** `set_at`, which is where a tracer that failed the
`jax.Array` check would show up:

```bash
PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1 PYNICAM_TIMELOOP_CHUNK=4 \
PYNICAM_TIMELOOP_DUMP=/path/fused ...     # assert the chunk actually engaged
```

`test/set_at_test.py::test_a_tracer_takes_the_jax_branch` pins the tracer property
on whatever jax version is installed; run the suite there too, since the laptop
pinned it against jax 0.6.0 only.

**Performance:** re-measure GPU step time. The numpy branch short-circuits on
`self.jax is None`; the jax branch adds one `isinstance` per call, at 14 call
sites, two of which (`kernels/vimain.py`, `kernels/vipath2.py`) are in the time
loop. Expected to be lost in the noise — but §6 asks for the measurement.

## 2. The `pyNICAM` object API — commit `6250b90`

`driver-dc.py` now runs on `pynicamdc/api.py`. The startup sequence, the loop body
and the teardown moved unchanged; see the commit message for the two deliberate
deletions (both dead).

**Expected: EXACTLY 0.0**, and the same on the fused path, which the CPU A/B did
not exercise (`_step_core` is only built under `PYNICAM_FUSE_TIMELOOP`):

```bash
PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_DUMP=...
```

The loop restructure is where a fused run could diverge from the driver's: the
chunk-trim guard now clamps to the requested step range rather than to `lstep_max`
(`n_end - n` in `pyNICAM.run`). With a single `run()` to `lstep_max` the two are
the same expression — which is the case the driver takes — so a difference here
would mean the trim logic was not transcribed faithfully.

Worth one extra run: **`run()` called in chunks under fusion**, which has no
counterpart in the old driver at all.

```python
nicam = pyNICAM("drv.toml"); nicam.initialize()
for _ in range(4):
    nicam = nicam.run(3)
nicam.finalize()
```

vs one `run()` to 12. Bit-exact on CPU; on the fused device path the chunk
boundaries interact with the warm-up counter (`PYNICAM_TIMELOOP_WARMUP`), so it is
worth seeing rather than assuming.

## What IS verified, on CPU (JW gl05rl01 z40, 8 ranks, IDEAL IC, 12 steps)

| check | result |
|---|---|
| numpy, `main` vs `set-at-value-dispatch` | bit-exact, 48 variables |
| numpy, `main` vs `api-layer` | bit-exact, 48 variables |
| jax `RESIDENT=0`, `main+fix` vs `api-layer` | bit-exact, 48 variables |
| API in `run(3)` × 4 vs one `run()`, numpy | bit-exact, 48 variables |
| test suite | 68 passed, 1 skipped |
