# pyNICAM-DC — new-user tutorial & test kit

A guided, three-tier path from a fresh checkout to a validated GPU run — covering the
**full nicamdc scientific test suite (15 cases)**. Each tier ends in a clear **PASS/FAIL**.

| tier | what it proves | where | needs |
|---|---|---|---|
| **1** | the code imports and the kernels are correct (numpy = jax bit-exact) | any CPU | numpy env |
| **2** | the model runs all 15 nicamdc test cases on CPU and matches the reference | any CPU | + input dataset |
| **3** | the **GPU (JAX)** path reproduces the CPU science for all 15 cases | Miyabi GPU | + GPU env |

## The 15 scientific test cases
These are the exact nicamdc `test/case` suite, each self-initializing (no restart) at its
**real vertical grid and planet factor**, at `gl05` horizontal resolution (single process):

| short | nicamdc case | init | z | planet |
|---|---|---|---|---|
| `jw` | ICOMEX_JW | Jablonowski dry baroclinic | 40 | ×1 |
| `hs` | ICOMEX_HS | Held-Suarez forced | 40 | ×1 |
| `hsshort` | ICOMEX_HS_short | Held-Suarez (short) | 40 | ×1 |
| `jm11` | DCMIP2016-11 | moist baroclinic wave | 30 | ×1 |
| `jm11dry` | DCMIP2016-11_DRY | moist baroclinic (dry) | 30 | ×1 |
| `jm21` | DCMIP2016-21 | moist baroclinic (2016-21) | 30 | ×1 |
| `tc` | DCMIP2016-12 | tropical cyclone | 30 | ×1 |
| `sc` | DCMIP2016-13 | supercell | 40 | ×120 |
| `tr11` | DCMIP2012-11 | 3D deformational tracer advection | 60 | ×1 |
| `tr12` | DCMIP2012-12 | Hadley-like tracer advection | 120 | ×1 |
| `tr13` | DCMIP2012-13 | tracer advection over orography | 60 | ×1 |
| `mw20` | DCMIP2012-20 | steady-state over mountain | 60 | ×500 |
| `mw21` | DCMIP2012-21 | Schär mountain waves | 60 | ×500 |
| `mw22` | DCMIP2012-22 | Schär steep mountain waves | 60 | ×500 |
| `gw` | DCMIP2012-31 | non-hydrostatic gravity wave | 10 | ×250 |

The manifest is `cases.txt` (drives the tier scripts). Each runs a few steps — enough to exercise
the full solver, quick enough for onboarding.

---

## 0. Setup

### 0a. Get the code + environment
```bash
git clone <pyNICAM-DC repo> && cd pyNICAM-DC
python3 -m venv venv && source venv/bin/activate    # python 3.11 or 3.12
pip install numpy toml xarray "zarr<3" dask mpi4py pytest
pip install jax jaxlib          # optional: enables the jax unit tests + a CPU jax run
```
`mpi4py` needs an MPI library (mpich/openmpi). On **Miyabi** use the prepared `venv-gh200` and
`module load nvidia/25.9` instead. (Full env details: `../INSTALL.md`.)

### 0b. Get the input dataset (grid + 8 vertical grids + 15 goldens, ~120 MB download)
The binary grid/reference data lives **outside** the repo:
```bash
cd tutorial
export PYNICAM_TUTORIAL_INPUTS_URL="<ask the maintainer for the link>"
./download_inputs.sh
```
This populates `case/grid_gl05rl00pe01/` (horizontal grid + the 8 converted vertical grids,
z10→z120) and `case/golden/` (numpy reference snapshots, one per case). The small text configs
(`case/config/*.toml`, `drv/*.toml`, `cases.txt`) ship with the repo.

---

## 1. Tier 1 — unit tests (no dataset needed)
```bash
./run_tier1_pytest.sh
```
The `pytest` suite: kernel bit-exactness (numpy, and numpy-vs-jax if jax is installed), RNG
determinism, mpi4py. jax/full-model tests skip themselves on a minimal env. **Expect:**
`N passed[, M skipped]`, `TIER 1 exit=0 (PASS)`.

## 2. Tier 2 — CPU scientific validation (all 15)
```bash
./run_tier2_cpu.sh            # all 15 cases
./run_tier2_cpu.sh gw         # just one (by short name)
```
Each case runs on numpy and is checked with `check_validation.py`: **physical sanity** (finite,
`RHOG>0`, `RHOGE>0`) + **golden compare** (`rtol=1e-6`). **Expect:** `TIER 2: 15/15 passed -- ALL
PASS`. (On a different CPU/BLAS the golden match sits at ~1e-7, still inside tolerance.)

## 3. Tier 3 — GPU validation (Miyabi, all 15)
```bash
qsub run_tier3_gpu.pbs        # results in pnc_tut.o<jobid>
```
Runs each case on **JAX/GPU** and compares to the CPU numpy golden (**peak-normalized** error
`|Δ|max/|field|max`, `rtol=5e-3`) — proving the GPU reproduces the CPU science to the floating-point
reassociation floor, plus wall-clock timing. **Expect:** `TIER 3: 15/15 passed`. Most cases agree to
~1e-7; the two families that drift more (still well inside tolerance) are physically expected:
- **reduced-planet cases** (`mw*`, `gw`, `sc`) — ~1e-4, from the stiffer dynamics;
- **`hs`** — ~3e-3 in the *forced momentum*: the Held-Suarez Rayleigh friction is an ill-conditioned
  forcing tendency, so it amplifies the CPU-vs-GPU reassociation difference (energy & mass still agree
  to ~1e-7). This is a known property of forcing tendencies, not a bug.

The check uses a **peak-normalized** metric on purpose: near-zero fields (e.g. vertical momentum in
balanced flow) make a per-cell relative error explode meaninglessly; normalizing to each field's own
peak measures error against its actual magnitude.

---

## 4. Visualize — figures & movies per case
```bash
./run_viz.sh <case>          # e.g. ./run_viz.sh gw     (defaults from viz_spec.txt)
./run_viz.sh tc sl_ps        # override the field
```
`run_viz.sh` reruns the case with the physical **diagnostics** on (`ml_*`, `sl_*` — winds,
potential temperature, surface pressure, water paths — not the raw prognostics), then calls
`render_zarr.py`. `viz_spec.txt` picks the field + view that shows each case's **characteristic
feature**, and `render_zarr.py` supports three views:
- **horizontal map** at a level (3D field) or the whole field (2D `sl_*`);
- **lon–height cross-section** (`--cross-section LAT`) — the right view for vertical phenomena.

| case | shows | field | view |
|---|---|---|---|
| jw / jm11dry / jm21 | baroclinic wave onset | `ml_th_prime` | map |
| hs | zonal jets | `ml_u` | map |
| jm11 | precipitable water | `sl_pw` | map |
| tc | cyclone surface low | `sl_ps` | map |
| sc | supercell updraft | `ml_w` | map |
| tr11/12/13 | tracer transport | `passive000` | map |
| mw20/21/22 | mountain lee waves | `ml_w` | **lon–height** |
| gw | gravity wave | `ml_th_prime` | **lon–height** |

Output lands in `viz/<case>/` (PNG per frame + an mp4). `render_zarr.py --list` shows a zarr's fields.

⚠️ **Development-time caveat (physics, not a bug):** a tutorial-length run shows the *initialized
state + onset*. The **fast** cases (`gw` ~1 h, `sc` ~2 h) genuinely develop their signature. The
**slow** cases (`jw`/`jm*`/`tc`) mature over *days* (the JW wave breaks ~day 8), which is far beyond a
few-step run — so their figures show the initial jet/vortex + early growth, not the mature feature.
Raise the case's step count (`./run_viz.sh jw ml_th_prime - 200`) or `lstep_max` to evolve further.

---

## 5. Performance — time-loop fusion & K-step chunks (benchmarking only)

Everything above (goldens, Tier 2/3, viz) runs the **ordinary per-step loop** — every step is a
separate dispatch and every output step is written. That is the correct mode for *producing outputs*.
When you instead want the lowest **seconds/step** on GPU, pyNICAM can fuse the outer time loop so a
whole batch of K steps is advanced on the device as one graph. This is **off by default** and is a
*measurement* tool — do not use it to generate goldens or movies.

**The gates** (JAX backend only; layer on top of the residency stack in `config/production.env`):

| env var | default | what it does |
|---|---|---|
| `PYNICAM_FUSE_TIMELOOP` | `0` (off) | master switch: advance the resident prognostic carry K steps per chunk via `run_timeloop_chunk` |
| `PYNICAM_TIMELOOP_CHUNK` | `1` | **K** — steps per chunk. `K=1` ⇒ no fusion (one step per dispatch) |
| `PYNICAM_TIMELOOP_JIT` | `0` | `1` = lift the K steps into **one `jax.lax.scan`** compiled once per K (the actual fusion — the whole chunk is a single dispatched graph). `0` = call the per-step core K times eagerly (a faithful-extraction check; **no speed win**) |
| `PYNICAM_TIMELOOP_WARMUP` | `3` | first **W** steps run the ordinary per-step path so JIT compiles and the per-step core (`_step_core`) is built + steady; chunking only starts at step ≥ W |
| `PYNICAM_COMM_NO_BARRIER` | (set by the harnesses) | **required** — the fused chunk hides the halo COMM; the barrier would serialize it |

Fusion is **bit-exact** with the per-step path (that is what `PYNICAM_TIMELOOP_JIT=0` proves), and the
driver hard-disables it if a non-fusable physics forcing is active (it would silently drop the
forcing) — so a case with `AF_TYPE≠NONE` falls back to per-step automatically.

### When you need time/step measurement — turn it ON
```bash
# in a PBS job, after `source config/production.env`:
export PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1 \
       PYNICAM_TIMELOOP_CHUNK=20 PYNICAM_TIMELOOP_WARMUP=3 \
       PYNICAM_COMM_NO_BARRIER=1
```
Run a few hundred steps; the per-step wall clock past the warm-up is your fused s/step. `WARMUP=3`
keeps the JIT-compile-heavy first steps out of the measurement. Larger K amortizes dispatch better but
compiles a bigger graph once (and costs more memory) — K = 10–50 is the usual sweet spot.

### When you need outputs, not timing — leave it OFF
Goldens, Tier 2/3 validation, and viz all want **every** step drained, so they run the default per-step
loop (`FUSE_TIMELOOP` unset). Don't enable fusion here — you'd only add compile latency and, if outputs
are frequent, get no fusion anyway (next point).

### Aligning K with the output interval
The driver **auto-trims every chunk so it never spans an output step** — it checks *both*
`PRGout_interval` (3D fields) and `PRGout_interval_2d` (2D fields) and stops the chunk just before the
next output. So **any K is safe for correctness**; the only question is whether the chunk reaches its
full length:
- **Output every step** (`PRGout_interval=1`, as the viz runs use) ⇒ every chunk is trimmed to length
  1 ⇒ **fusion is effectively disabled**. This is why you never benchmark a viz config.
- For real fusion, make outputs **infrequent relative to K**: set `PRGout_interval` (and
  `PRGout_interval_2d`) to a **multiple of K**, or disable output during the measured window. A chunk
  then runs its full K steps between drains. Rule of thumb: **choose K to divide the output interval**
  (e.g. output every 60 steps → `CHUNK=60` for one chunk per drain, or `CHUNK=20` for three).

---

## What the pieces are
- `download_inputs.sh` · `run_tier{1,2,3}` · `check_validation.py` (reusable: `python
  check_validation.py RUN.npy --ref GOLDEN.npy --rtol 1e-6`).
- `cases.txt` — the 15-case manifest (short name | nicamdc case | z | planet | description).
- `case/config/nhm_<short>.toml` — one config per case (native `IDEAL` init, real vgrid + planet).
- `drv/drv_<short>_<backend>.toml` — driver settings.
- **`runs/<case>/`** — a per-run **sandbox**: every run (tier 2/3 or viz) executes in its own
  directory with the shared `case/` inputs symlinked in as `./case`, so runs never clobber each
  other and everything a case produced (the config actually used, `run.log`, the raw `*.zarr` +
  `msg.pe*`, the dumped `out_*_rank0.npy`) lives in one place. Safe to delete; regenerated on the
  next run. Figures still land in `viz/<case>/`.

## Going further
- More steps: raise `lstep_max` in `case/config/nhm_<short>.toml` (and regenerate that golden).
- fp32: set `precision = "float32"` in a `drv_*.toml` (~1.8× faster on GPU, ~½ the memory).
- The vertical grids were converted from nicamdc's `data/grid/vgrid/*.dat` (big-endian Fortran)
  to pyNICAM's JSON; the converter is validated bit-exact against nicamdc's own JSONs.
- Full model background & science: https://nicam.jp/ and `../README.md`.

## Troubleshooting
- **`MPI_Init ... NULL communicator` / silent abort** — a login node without a job context; the
  scripts use `mpirun -np 1`. Prefix hand-run drivers the same way.
- **`ContainsGroupError` / `changing dimension size`** (zarr) — stale output; the scripts `rm -rf
  *.zarr` each run.
- **`ModuleNotFoundError: toml/jax/zarr`** — minimal env; install the step-0a deps (Tier 1 tests
  that need them skip automatically).
- **dataset missing** — run `./download_inputs.sh` (Tiers 2–3 need `case/grid_*` and `case/golden/`).
