# Handoff to MIYABI — readability refactor (stages 2+)

> **STATUS — REFACTOR COMPLETE / SUPERSEDED (2026-07-18).** This handoff did its job:
> the work moved to MIYABI and the core refactor is DONE. The **live source of truth is
> `refactor-plan-v4.txt` + the project memory**; everything below is HISTORICAL (the
> stage-0/1/2 analysis and the verification recipe remain accurate reference).
>
> **Final outcome (branch `readability`, HEAD `d200c20`, pushed):** `mod_dynamics.py`
> 3953 → 2729 lines (−31%) with the full GPU stack intact. Landed:
> - **stages 0–2** setup seam, static route-resolve, prepost jits built at setup.
> - **stage 3** stripped the dead device branches from the non-resident "block B".
> - **stage 4** purified `_nl_body`, warmed the on-device COMM plans at setup, and
>   RETIRED the warm-up state machine (`_fuse_warm_calls`/`_nlbody_steady`/
>   `_step_use_scan`) — plan §6's "root knot"; scan engages from step 0.
> - **stage 5** DELETED block B (~900-line duplication): `_nl_body` is now the single
>   dual-path RK body, scan-vs-eager a 1-line driver branch, one shared tracer tail.
> - **stage 6** DROPPED — Fortran `dynamics_step` is one monolithic subroutine (1383
>   lines), so splitting pyNICAM's would break the NICAM mental map.
> - **task J** translated the campaign jargon in the comments to plain language.
> - **task S** added `bk.set_at`/`add_at` and collapsed `mod_bndcnd`'s numpy/jax twins
>   (the one genuine mechanical twin; 768 → 552 lines, backend-agnostic).
>
> **Verification bar that held throughout:** numpy A/B on CPU (EXACTLY 0.0 for
> block-B/numpy-path changes) + resident jax A/B on MIYABI GPU (0.0, or the proven
> jit-vs-eager fp floor rel ≤~5e-13 when a build moved to setup — isolated with a temp
> `PYNICAM_PROVE_FLOOR` gate) + `ast.dump()` equality for comment-only changes.
> Reusable harnesses: `tutorial/stage{4a,4c,4d,4e,4f}_res_ab.pbs`, `stage5{a,b}_np_ab.pbs`,
> `stageS{1,2}_{np,res}_ab.pbs`.
>
> Optional future work (documented, off critical path): **G** gate collapse (needs the
> fused-default decision, plan §15) and **§7B** (thread `dynamics_step` state as one
> immutable value → shrink toward Fortran size + move the jit build to setup).
>
> Both original open questions are RESOLVED — see the end.

Continues the `readability` branch refactor described in `refactor-plan-v4.txt`
(was v3 when this was written). Written on the laptop env after stages 0–1 landed;
stages 2+ need MIYABI's full environment (mpi4jax + GPU) to be **run-verified**,
which is why work moved there.

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

## Stages — final status (all DONE unless noted; see the top banner + plan v4)

- **2** move prepost to setup — DONE (2a `b997606` / 2b `3578e1f`).
- **3** strip block B's dead resident branches — DONE (pt1 `b1c3072`, pt2 `c76ef2a`,
  pt3 `c676947`); block B reduced to the device-free host-numpy reference.
- **4** purify `_nl_body` + warm the COMM plans at setup + retire the warm-up state
  machine — DONE (4a `cc293c7`, 4c `ff43a6e`, 4d `3a4369d`, 4e `1b62947`, 4f `814e4eb`).
  4b (thread the 7 scratch nonlocals) SKIPPED — they're host-fallback scratch, not a
  jit blocker. The scan now engages from step 0; the runtime latch is gone. (Deferred:
  moving the `_nl_scan_jit` BUILD itself into finalize — needs the §7B purification.)
- **5** delete the ~900-line block B; route the non-resident path through `_nl_body` —
  DONE (5a `f413825`, 5b `b996a68`). One dual-path body, one shared tracer tail. This
  made stages 4–5 numpy-verifiable in retrospect (numpy runs `_nl_body` now).
- **6** DROPPED — Fortran `dynamics_step` is monolithic (1383 lines), so splitting
  would break the NICAM mental map. Replaced by task J (jargon → plain comments).
- **J** DONE (`bb88616`) — comment-only, AST-verified bit-exact.
- **S** DONE for `mod_bndcnd` (`fb049ce`/`bd58d63`/`47af3ce`), the one mechanical
  numpy/jax twin. Stopped there — `mod_vi`/`hlimiter`/`numfilter` are jax kernels or
  non-bit-exact device variants, not mechanical twins (a `_resident` suffix != a twin).
- **G** collapse the gates — NOT DONE; needs the fused-default decision (plan §15).
- **§7B** thread `dynamics_step` state as one immutable value — NOT DONE (the hard one;
  would shrink `dynamics_step` toward Fortran size + move the jit build to setup).

## Open questions to resolve on MIYABI

1. **RESOLVED (stage 3).** The block-B (non-resident INLINE) `_prepost_fn` and every
   `if _resident_*:` device branch there were confirmed dead (block B is entered iff
   `self._resident` is False, and each guard ANDs with `self._resident`; `_rkcopy`/
   `_progqout` too). Stage 3 (parts 1/2/3) STRIPPED all of it — block B is now the
   device-free host-numpy reference. Verified numpy A/B **max|d|=0.0** (jm11/jw/gw/
   hsshort, `tutorial/stage3p{2,3}_ab.pbs`).
2. **RESOLVED.** Stage 1 re-verified bit-exact on the resident path (1b: jw/hsshort/
   jm11 A/B = 0.0). Stage 3 also re-verified as a NO-OP on the resident path
   (`tutorial/stage3_res_ab.pbs`, baseline f80609f: jw/hsshort **max|d|=0.0**) —
   confirming the block-B strip and the shared-scope preamble removal don't touch
   the resident path.
3. **STILL OPEN.** Decide the fused-default question (plan §15) before the gate
   collapse (G).
