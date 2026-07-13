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
Runs each case on **JAX/GPU** and compares to the CPU numpy golden (`rtol=1e-4`) — proving the GPU
reproduces the CPU science to the floating-point reassociation floor (JAX fuses/reorders ops, so
~1e-10…1e-5, not 0), plus wall-clock timing per case. **Expect:** `TIER 3: 15/15 passed`.

---

## What the pieces are
- `download_inputs.sh` · `run_tier{1,2,3}` · `check_validation.py` (reusable: `python
  check_validation.py RUN.npy --ref GOLDEN.npy --rtol 1e-6`).
- `cases.txt` — the 15-case manifest (short name | nicamdc case | z | planet | description).
- `case/config/nhm_<short>.toml` — one config per case (native `IDEAL` init, real vgrid + planet).
- `drv/drv_<short>_<backend>.toml` — driver settings.

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
