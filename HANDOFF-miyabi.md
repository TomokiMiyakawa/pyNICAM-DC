# Handoff to MIYABI — readability refactor (stages 2+)

Continues the `readability` branch refactor described in `refactor-plan-v2.txt`.
Written on the laptop env after stages 0–1 landed; stages 2+ need MIYABI's full
environment (mpi4jax + GPU) to be **run-verified**, which is why work moves there.

## TL;DR

- **Done & verified:** stage 0 (setup seam) and stage 1 (route static-resolution).
  Both committed on `readability`. Stage 1 confirmed **bit-exact** (numpy A/B).
- **Why hand off:** stages 2–6 touch the **jax RESIDENT path** (`_nl_body`,
  `_prepost_jit`, on-device COMM). That path needs **mpi4jax**, which is absent
  on the laptop env, so it cannot be run there — only on MIYABI (mpi4jax + GPU).
- **Next stage:** stage 2 — move `_prepost_jit` build to setup. The seam and the
  closure analysis are ready (below).

## Where we are (commits on `readability`)

```
db8869a  refactor(dynamics): stage 1 -- resolve the backend route once at setup
77bbf08  refactor(dynamics): stage 0 -- add the dynamics_setup_finalize seam
5f1f940  test: template for the JAX trace/jit semantics behind the prepost move
5beba6a  docs: consolidate the plan into v2 (finalize seam, msc bag split)
```

Working tree is clean except untracked tier2 run sandboxes under `tutorial/runs/`
(gitignored) and the downloaded dataset under `tutorial/case/` (gitignored).

### Stage 0 (commit 77bbf08)
Added `Dyn.dynamics_setup_finalize()` (empty) called at the **end** of
`dynamics_setup` — after the sub-setups, so it is guaranteed to see
`bsst.rho_bs/pre_bs`. NICAM-lineage `dynamics_setup` shape untouched.

### Stage 1 (commit db8869a)
`finalize` now resolves the backend route once:
```python
self._is_jax   = (bk.type == "jax")
self._resident = bk.resident()
```
`bk` is threaded into `dynamics_setup` (explicit arg, after `vi`, before
`rdtype`) and forwarded to `finalize`; the driver passes `msc.bk`. In
`dynamics_step` (+ `forcing_step`, `_forcing_xp`, `sync_prgvar_to_host`) the 88
per-step probes were replaced: `msc.bk.resident()`→`self._resident` (76),
`(msc.)bk.type=="jax"`→`self._is_jax` (16). Safe because both are constant after
`bk.configure` (which runs before `dynamics_setup`) and `resident()` latches its
`PYNICAM_RESIDENT` read on first call.

**Known nit to fix in stage 6:** the mechanical replace left a harmless
copy-paste triple in `_rkcopy`:
`self._resident and self._resident and self._resident` (was `msc.bk.resident()`
×3). Idempotent; collapse to one when splitting.

## The environment reality (why MIYABI)

Laptop env `jax_nomtl_mpi` (python 3.11.12) has jax 0.6.0 but **no mpi4jax**.
Verification tiers and what each actually exercises:

| tier / mode | runs | exercises `_nl_body` / `_prepost_jit`? |
|---|---|---|
| tier1 `run_tier1_pytest.sh` | kernels, RNG, mpi presence | no (never enters `dynamics_step`) |
| tier2 numpy `run_tier2_cpu.sh` | full model, numpy | **no** — takes the INLINE non-resident body |
| jax-CPU non-resident `PYNICAM_RESIDENT=0` + `drv_*_jax.toml` | full model, jax kernels + host COMM | **no** — same INLINE body, `_prepost_jit` gated off |
| jax RESIDENT (default jax) | fusion + on-device COMM | **yes** — but needs **mpi4jax** → fails on laptop |
| tier3 GPU `run_tier3_gpu.pbs` | the real resident/fused path | **yes** (MIYABI) |

**Key structural fact (measured):** the nl-loop forks on
`if _fuse_nlbody (= self._resident):`
- **True (jax resident):** runs `_nl_body` (`mod_dynamics.py` L1159–2069) via
  scan / `_nl_body_jit` / eager-warmup.
- **False (numpy, jax non-resident):** runs a near-duplicate **INLINE** body
  (~L2500–3400) with its own `_prepost_fn` (L~2667) and BNDCND/THRMDYN/src/vi.

So `_nl_body` and `_prepost_jit` (stages 2–5) run **only** on the resident jax
path. numpy/`RESIDENT=0` cannot reach them. Stage 1 was verifiable on the laptop
only because it touched **both** paths.

## The verification recipe (reuse it on MIYABI)

Stage 1 was proven bit-exact by A/B, not by argument. Repeat per stage:

```bash
cd tutorial
export PATH=<env>/bin:$PATH           # the env with mpi4jax + jax
# 1. baseline: run BEFORE your change (or git stash your change), dump out_np
./run_tier2_cpu.sh gw                  # numpy path
#   and the resident jax path (write a jax runner or qsub run_tier3_gpu.pbs)
cp tutorial/runs/gw/out_np_rank0.npy /tmp/gw_baseline.npy
# 2. apply change, re-run same case
# 3. compare bit-for-bit
python -c "import numpy as np; a=np.load('/tmp/gw_stage.npy'); b=np.load('/tmp/gw_baseline.npy'); print(np.array_equal(a,b), np.max(np.abs(a-b)))"
```
`np.array_equal True` / `max abs diff 0.0` = the change altered nothing. Because
stages 2–5 are on the resident path, the A/B **must** be run on the resident jax
path (default jax on MIYABI, or tier3), not just numpy tier2.

Also turn on `JAX_LOG_COMPILES=1` to watch compiles move to setup (plan §10):
the goal is **zero compiles inside the measurement window** after stage 2.

Dataset: `tutorial/download_inputs.sh` (~120 MB, checksummed) if `tutorial/case/`
is not already populated on MIYABI.

## Stage 2 — concrete notes (analysis already done)

Goal: build `_prepost_jit` / `_prepost_pl_jit` in `dynamics_setup_finalize`
instead of lazily inside the nl-loop, and delete the warm-up cascade
(`_fuse_warm_calls`, `_nlbody_steady`, `_step_use_scan` latch, eager warm-up
passes). This is plan §6, "the root knot".

**Closure is setup-safe (measured).** `_prepost_fn` (`mod_dynamics.py` L~1264)
closes over only run-constants:
- `msc`, `bndc` — host objects (bndc IS already a `dynamics_setup` arg; **msc is
  NOT** — thread it in, or pass the few subsystems `BNDCND_all_resident` needs)
- `GSGAM2` = `_diag_dev["GSGAM2"]` (from `vmtr`, via `bk.device_consts`)
- `pre_bs`/`rho_bs` (from `bsst`, ready after `bsstate_setup`)
- cnst scalars `Rdry/CPdry`, `PRE00`; index consts `I_pre`, `I_tem`
- 4 traced array args: `_D, _P, _r, _e` (DIAG, PROG_d, rho, ein)

`BNDCND_all_resident` (`mod_bndcnd.py` L290) pulls only run-constants from msc
(`adm, rcnf, cnst, vmtr`), builds its geometry via `device_consts("bndcnd_geom")`,
and its `_bnd_cfg_*` / `_bnd_kernels`. Its 4 array args are the traced inputs.

**The one real subtlety — cache warming.** `_bnd_kernels_get()` (`mod_bndcnd.py`
L42) lazily `bk.maybe_jit`-**wraps** 6 kernels on first call (wrap ≠ compile).
Today `_prepost_jit` is built only AFTER an eager `BNDCND_all_resident` call
(L1247) that populates these caches, so the first *traced* call of `_prepost_jit`
hits warm caches. If you build `_prepost_jit` at setup and it is first *called*
inside the outer scan trace, `_bnd_kernels_get()` / `device_consts` would first
fire inside that trace. Wrapping inside a trace is not itself illegal (see
`test/prepost_jit_semantics_test.py`), but **verify** it does not leave a
trace-built constant cached for reuse across steps (the "stale baked constant used
silently" hazard the same test pins down). Safe recipe: in `finalize`, warm every
cache eagerly (call `_bnd_kernels_get()`, `_bnd_cfg_thermo/mom`, and build the
`device_consts`) OUTSIDE any trace — e.g. one eager `BNDCND_all_resident` on
correctly-shaped zero arrays — THEN wrap `_prepost_jit`. Shapes are known from
`adm` at setup, so no real state is needed.

**Semantics reference:** `test/prepost_jit_semantics_test.py` (CPU, no GPU) pins
the five facts this rests on: jax.jit wraps ≠ compiles; pre-touching an inner jit
is wasted under fusion; a jit built-in-trace is illegal to use out-of-trace; a
stale baked constant is used *silently* across traces; a setup-built jit closing
setup-constants is reuse-safe. When stage 2 lands, tighten these against the real
`Dyn` object and add the `JAX_LOG_COMPILES` "zero compiles in window" test.

## Progress (done on MIYABI)

- **Stage 1 re-verified** bit-exact on the RESIDENT path (jw/hsshort/jm11 A/B =
  0.0) — resolves open-Q #2 below.
- **Stage 2a done** (`b997606`): `_prepost_jit`/`_prepost_pl_jit` built at setup
  via `_build_prepost_jits` (called from `dynamics_setup_finalize`). msc/bndc/bsst
  threaded in (bndc/bsst are still driver locals at setup — NOT yet on msc). Caches
  warmed eagerly on the `__init__` buffers before wrapping. Numpy path byte-
  identical (early-return). ★ Resident A/B is NOT strictly 0.0 — it sits at the
  **jit-vs-eager fp floor** (jw rel 5.7e-16..5e-13; hsshort/jm11 hit 0.0) because
  prepost is now fused-jit from step0-nl0 where the baseline ran it eager. PROVEN
  benign via a temp `PYNICAM_PROVE_FLOOR` gate (forces eager-first → all 0.0). So
  the resident bar for stages 2+ is "at the fp floor", not 0.0.
- **Stage 2b done** (`3578e1f`) — NARROWER than planned: deleted the dead in-body
  prepost build else-branches (bit-exact, verified 0.0). ★★ The warm-up cascade was
  NOT removed: the eager warm-up also warms the **on-device COMM plan** (topology
  `int64[161280]` index array, lazy in `mod_comm._build_comm_plan_device`) OUTSIDE
  any trace. Removing it → the plan builds inside the `_nl_body_scan` trace →
  `UnexpectedTracerError` (verified). Retiring the cascade needs the COMM plan (+
  other lazy device-const signatures) warmed at SETUP → fold into stage 4.

## Remaining stages (all resident-path → MIYABI)

- **2** move prepost to setup — DONE (2a/2b above). Cascade removal deferred to 4.
- **3** dedup the byte-identical `_prepost_fn` / `_prepost_pl_fn` twins (~416 lines).
- **4** purify `_nl_body`: 17 `nonlocal`s + in-body `msc` → args; introduce the
  ONE immutable state value (plan §7B). The hard one. **★ ALSO warm the on-device
  COMM plan (+ any other lazy device-const) at setup here** (the stage-2b finding),
  so `_nl_scan_jit`/`_nl_body_jit` can move to setup and the warm-up cascade
  (`_fuse_warm_calls`/`_nlbody_steady`/`_step_use_scan` latch) can finally retire.
- **5** delete the ~900-line INLINE copy; route the non-resident path through
  `_nl_body`. `if nl != 0` → uniform `where(nl==0, ...)`. **After this numpy runs
  `_nl_body`, so stages 4–5 become numpy-verifiable in retrospect** (a useful
  cross-check back on any CPU env).
- **6** split `dynamics_step` into named stages; fix the `_rkcopy` triple nit.
- **S** (independent) `bk.set_at` over 157 `.at[]` sites; collapse the numpy/jax
  twin methods (`BNDCND_all` vs `BNDCND_all_resident`, etc.).
- **G** collapse the 118 gates (needs the fused-default decision, plan §15).

## Open questions to resolve on MIYABI

1. **Is the L~2667 `_prepost_fn` (inside the non-resident INLINE body, guarded by
   `_resident_prepost`) reachable at all?** `_fuse_nlbody` and `_resident_prepost`
   are both `= self._resident` at L1153/1062, so the guard looks never-true inside
   the `_fuse_nlbody` False branch — i.e. possibly dead. Confirm against the real
   gate combinations (e.g. `PYNICAM_ONDEVICE_COMM`, `use_ondevice_comm`) before
   relying on it in stages 3/5. Not asserted here — flagged.
2. Confirm stage 1's route resolution stays bit-exact on the **resident** path
   (only the numpy/non-resident A/B ran on the laptop).
3. Decide the fused-default question (plan §15) before the gate collapse (G).
