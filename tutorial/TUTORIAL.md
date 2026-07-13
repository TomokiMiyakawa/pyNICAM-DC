# pyNICAM-DC — new-user tutorial & test kit

A guided, three-tier path from a fresh checkout to a validated GPU run. Each tier is a
single script; each ends in a clear **PASS/FAIL**.

| tier | what it proves | where | time | needs |
|---|---|---|---|---|
| **1** | the code imports and the kernels are correct (numpy = jax bit-exact) | any CPU | ~30 s | numpy env |
| **2** | the model runs real science on CPU and matches the reference | any CPU | ~2 min | + input dataset |
| **3** | the **GPU (JAX)** path reproduces the CPU science + shows per-step timing | Miyabi GPU | ~5 min | + GPU env |

The two scientific cases (self-initializing — **no restart files needed**):
- **jbw** — Jablonowski dry baroclinic wave — the pure dynamical core.
- **jm** — Jablonowski-Moist — adds moisture + DCMIP forcing/physics.

Both run at `gl05` (≈10k columns), 5 steps — small enough for a laptop, real enough to exercise
the full solver.

---

## 0. Setup

### 0a. Get the code
```bash
git clone <pyNICAM-DC repo>        # or: cd into your existing checkout
cd pyNICAM-DC
```

### 0b. Python environment
Core dependency is just `numpy`; a real run needs the "full env" (see `../INSTALL.md`).

**Minimal CPU env (Tiers 1–2):**
```bash
python3 -m venv venv && source venv/bin/activate    # python 3.11 or 3.12
pip install numpy toml xarray "zarr<3" dask mpi4py pytest
# optional, enables the jax unit tests + a CPU jax run:
pip install jax jaxlib
```
`mpi4py` needs an MPI library (mpich/openmpi) present. On **Miyabi** just use the prepared
`venv-gh200` and `module load nvidia/25.9` instead of building your own.

**GPU env (Tier 3, Miyabi):** use `venv-gh200` (jax+CUDA+mpi4jax already built) — the Tier-3 job
script loads it for you.

### 0c. Get the input dataset (grid + goldens, ~25 MB)
The binary grid and reference data live **outside** the repo. Download them into `tutorial/case/`:
```bash
cd tutorial
export PYNICAM_TUTORIAL_INPUTS_URL="<ask the maintainer for the link>"
./download_inputs.sh
```
This populates `case/grid_gl05rl00pe01/` (the icosahedral grid) and `case/golden/` (numpy
reference snapshots). The small text configs (`case/config/*.toml`, `drv/*.toml`) already ship
with the repo.

---

## 1. Tier 1 — unit tests (no dataset needed)
```bash
cd tutorial
./run_tier1_pytest.sh
```
Runs the `pytest` suite: kernel bit-exactness (numpy, and numpy-vs-jax if jax is installed),
RNG determinism, mpi4py availability. jax and full-model tests **skip themselves** when those
deps are absent, so a minimal numpy env still passes.

**Expect:** `N passed[, M skipped]` and `TIER 1 exit=0 (PASS)`.

---

## 2. Tier 2 — CPU scientific validation
```bash
cd tutorial
./run_tier2_cpu.sh
```
Runs **jbw** and **jm** on the numpy backend (5 steps each) and checks each dump with
`check_validation.py`:
1. **physical sanity** — every value finite, density `RHOG > 0`, energy `RHOGE > 0`;
2. **golden compare** — the final prognostic state vs the bundled numpy golden (`rtol=1e-6`).

**Expect:** for each case `=== PASS ===`, and `TIER 2 ALL PASS`. On the same platform the golden
match is exactly `0.0`; on a different CPU/BLAS expect ~1e-11 (a cross-platform rounding floor,
still well inside the tolerance).

---

## 3. Tier 3 — GPU validation (Miyabi)
```bash
cd tutorial
qsub run_tier3_gpu.pbs
# watch: qstat ; results land in pnc_tut.o<jobid>
```
Runs both cases on the **JAX/GPU** backend and compares each to the **CPU numpy golden**
(`rtol=1e-4`) — i.e. it proves the GPU produces the same science as the CPU reference, up to
the floating-point reassociation floor (JAX fuses/reorders ops, so expect ~1e-10…1e-5, not 0).
It also prints the **wall-clock time** for the 5 steps (which includes the one-time JIT compile).

**Expect:** `=== PASS ===` for each case, `TIER 3 ALL PASS`, and a timing line per case. (Validated:
GPU-vs-CPU agreement ~2e-10 for both cases.)

---

## What the scripts are
- `download_inputs.sh` — fetch + unpack the input dataset.
- `run_tier1_pytest.sh` — the unit suite.
- `run_tier2_cpu.sh` — numpy jbw+jm + validation.
- `run_tier3_gpu.pbs` — Miyabi PBS job: jax jbw+jm + validation + timing.
- `check_validation.py` — the checker (sanity + reference compare; exit 0/1). Reusable:
  `python check_validation.py RUN.npy --ref GOLDEN.npy --rtol 1e-6`.
- `drv/drv_<case>_<backend>.toml` — driver settings (which config, which backend/precision).
- `case/config/nhm_<case>_ideal.toml` — the model configs (native `IDEAL` init, 5 steps).

## Going further
- More steps: raise `lstep_max` in `case/config/nhm_*_ideal.toml`.
- Other ideal cases the model self-initializes (all validated): `Heldsuarez`, `Gravitywave`,
  `Mountainwave`, `Tropical-Cyclone`, `Supercell`, `Traceradvection` — copy a config, change
  `init_type`, and point a `drv_*.toml` at it.
- fp32: set `precision = "float32"` in a `drv_*.toml` (~1.8× faster on GPU, ~½ the memory).
- Full model background & science: https://nicam.jp/ and `../README.md`.

## Troubleshooting
- **`MPI_Init ... on a NULL communicator` / silent abort** — a login node without a job context.
  The scripts use `mpirun -np 1` to avoid this; if you invoke the driver by hand, prefix it too.
- **`ContainsGroupError` / `changing dimension size`** (zarr) — a stale output dir. `rm -rf *.zarr`
  (the scripts already do this each run).
- **`ModuleNotFoundError: toml` (or jax/zarr)** — you're on the minimal env; install the full-env
  deps in step 0b, or ignore for Tier 1 (those tests skip).
- **dataset missing** — run `./download_inputs.sh` (Tiers 2–3 need `case/grid_*` and `case/golden/`).
