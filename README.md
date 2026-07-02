# pyNICAM-DC

**pyNICAM-DC** is a Python translation of the dynamical core of the Non-hydrostatic
ICosahedral Atmospheric Model (**NICAM**). It runs the same icosahedral-grid,
non-hydrostatic dynamics as NICAM-DC, with a backend-switchable design that runs on
**CPU (NumPy)** or **GPU (JAX)** — including on-device MPI halo exchange and an optional
device-resident/kernel-fused fast path for large speedups on GPU.

For NICAM background and science, see https://nicam.jp/.

---

## Features

- **Two backends, one code base** — select `numpy` (reference, CPU) or `jax` (CPU or GPU)
  at run time via the driver settings.
- **GPU acceleration (JAX)** — JIT-compiled kernels, on-device halo communication
  (`mpi4jax`), device residency across the time loop, and `lax.scan` fusion at both the
  RK-substep and whole-time-loop levels.
- **Single or double precision** — `float64` (default, reference) or `float32`
  (~1.8× faster, ~½ the GPU memory).
- **MPI-parallel** — icosahedral region decomposition across ranks.
- **Bit-reproducible reference path** — the default (unoptimized) path is validated
  bit-exact against established golds; all optimizations are opt-in and separately validated.

---

## Installation

Requires Python 3.11/3.12. Core dependency is `numpy`; a full run needs the environment
below (see `INSTALL.md` and `requirements.txt` for details).

```bash
git clone https://github.com/TomokiMiyakawa/pyNICAM-DC.git
cd pyNICAM-DC
pip install -e .
```

> The `-e` (editable) install is intended even for non-developers: the model is run
> in place from the source tree (see Quick start below), and `-e` makes the `pynicamdc`
> package importable from that tree. A plain `pip install .` also works but would run the
> copy in `site-packages` while you sit in the checkout, so local edits wouldn't take effect.

**Runtime dependencies**
- Basic (CPU / NumPy): `mpich` (or another MPI), `mpi4py`, `toml`, `xarray`, `dask`, `zarr` (2.15.x)
- GPU / JAX: `jax` (CUDA build) + `mpi4jax` (CUDA-aware MPI)
- Quicklook / plotting (optional): `matplotlib`, `cartopy`, `scipy`

---

## Quick start (JW baroclinic wave, NumPy, 8 ranks)

```bash
pip install -e .
cd pynicamdc/nhm/driver/
mpiexec -n 8 python3 -u driver-dc.py
```

Output is written to `testout_tmp.zarr` (remove/rename any existing one before re-running).
Quicklook with the notebooks in the repo (e.g. `daskzarr_out-simple.ipynb`).

---

## Choosing a backend / precision

The backend and precision are read from a small `driversettings.toml` passed to the driver:

```toml
[driver]
backend   = "jax"          # "numpy" or "jax"
precision = "float64"      # "float64" or "float32"
nhm_driver_cnf = "./path/to/nhm_driver.toml"   # the model config
```

```bash
mpiexec -n <N> python3 -u pynicamdc/nhm/driver/driver-dc.py --driver-setting driversettings.toml
```

- **`numpy`** — the reference CPU path.
- **`jax` on GPU** — set `JAX_PLATFORMS=cuda`, `MPI4JAX_USE_CUDA_MPI=1`, `HCOLL_ENABLE=0`;
  run 1 rank per GPU.
- **`jax` on CPU** — set `JAX_PLATFORMS=cpu` (useful for testing without a GPU).

### The GPU fast path (device residency + fusion)

The device-resident / fused optimizations are gated by `PYNICAM_*` environment flags,
all **default-OFF** (so the default run reproduces the reference path). Enable the full
validated stack with the bundled preset:

```bash
source config/production.env     # sets the fusion + residency flags + CUDA-aware MPI env
mpiexec -n <N> python3 -u pynicamdc/nhm/driver/driver-dc.py --driver-setting <settings>.toml
```

Use `backend="jax"` (and `precision="float32"` for the fastest path) in your
`driversettings.toml`. To run the same stack on CPU, edit `JAX_PLATFORMS=cpu` in the preset.
See [`docs/MERGE_NOTES.md`](docs/MERGE_NOTES.md) for the layered design and measured
performance.

---

## Test cases

Located under `pynicamdc/test/`. Each has its own `config/` + `prepdata/` + horizontal
grid (all in the repo); the large **restart** data is downloaded separately.

| Case | Test | Setup |
|---|---|---|
| `case1` | small smoke test | fully in-repo |
| `case2` | Jablonowski-Williamson baroclinic wave (RK3) | gl05, z40, 8 ranks |
| `case3` | DCMIP 1-1 tracer advection (TRCADV) | gl05, z60, 8 ranks |

### Fetch the restart data (case2 / case3)

The restart datasets (~49 MB compressed) are hosted outside git. Download + verify + extract:

```bash
bash tools/fetch_testdata.sh
```

(Override the host with `PYNICAM_TESTDATA_URL=... bash tools/fetch_testdata.sh`.)

### Full resolution-sweep dataset (developers)

The large **sweep** dataset used for the resolution-sweep performance + validation runs
(gl05–gl09: horizontal boundary + restart + numpy reference golds, ~10 GB total) is a
separate, heavier download:

```bash
bash tools/fetch_sweepdata.sh            # all glevels gl05..gl09 (~10 GB)
bash tools/fetch_sweepdata.sh 07 08      # a subset (per-glevel tarballs)
```

Per-glevel compressed sizes: gl05 ~33 MB, gl06 ~117 MB, gl07 ~442 MB, gl08 ~1.8 GB,
gl09 ~6.9 GB. It extracts into a sweep root (`PYNICAM_SWEEP_ROOT`, default `./pynicam-sweep`)
laid out as `data/{boundary,restart,mnginfo}` + `run/golds/`, which the run harness points at.
(Override the host with `PYNICAM_SWEEPDATA_URL=...`.)

**To actually run the sweep**, use the harness in [`tools/sweep/`](tools/sweep/) — see
[`tools/sweep/README.md`](tools/sweep/README.md). One command does fetch + assemble a
runnable sweep root:

```bash
bash tools/sweep/setup_sweep.sh          # fetch data + lay out scripts/config/data/golds/code
```

then run the reference (numpy) or GPU fast-path (`source config/production.env`, `BACKEND=jax`)
sweep and validate against the golds with `pynicamdc/nhm/dynamics/proto/cmp_prec.py`.

### Run a case (example: case3)

```bash
cd pynicamdc/test/case3
ln -sfn "$(pwd)" case               # the config uses ./case/... relative paths
cat > driversettings.toml <<'EOF'
[driver]
backend = "numpy"
precision = "float64"
nhm_driver_cnf = "./case/config/nhm_driver.toml"
EOF
mpiexec -n 8 python3 -u ../../nhm/driver/driver-dc.py --driver-setting driversettings.toml
```

---

## Performance (Miyabi-G / NVIDIA GH200, glevel-8, 4 ranks)

Steady per-step wall time of the dynamical core (`__Dynamics`; the one-time JIT-compile
step is excluded), measured this configuration:

| Configuration (glevel-8, `float64` unless noted) | s/step |
|---|---|
| Eager JAX (per-kernel JIT; no residency/fusion) | ~6.5 |
| Full device-resident + fused stack | 0.215 |
| + time-loop fusion (`lax.scan` over the driver loop) | 0.190 (−12%) |
| + `float32` (full stack) | 0.117 |
| + `float32` & time-loop fusion | 0.100 (−15%) |

- **Time-loop fusion** adds −12% (`float64`) / −15% (`float32`) at glevel-8, growing to
  **−26% at glevel-9** (`float32`) — the recovered per-step host/dispatch overhead is
  byte-proportional, so it helps *more* at larger grids.
- **`float32`** roughly halves both step time and GPU memory (glevel-8 peak ≈ 19 GB vs
  ≈ 37 GB for `float64`; glevel-9 `float32` ≈ 73 GB, which is why `float32` is needed to fit
  glevel-9 on 4 ranks).
- All optimizations are **bit-exact** vs the reference path (JW dynamics gold 1.15e-11; DCMIP
  tracer advection ~3e-15) and are **gated default-OFF**.

Numbers are hardware/config specific. Speedup *ratios* depend strongly on the chosen baseline
(the "eager JAX" anchor is itself sensitive to, e.g., the COMM implementation), so absolute
per-step times are reported here rather than a single headline factor. See
[`docs/MERGE_NOTES.md`](docs/MERGE_NOTES.md).

---

## Repository layout

```
pynicamdc/nhm/                    model: driver, dynamics, share modules
pynicamdc/nhm/dynamics/kernels/   backend-switchable JAX kernels
pynicamdc/share/                  backend dispatch (mod_backend), on-device COMM (mod_comm), geometry
pynicamdc/test/                   test cases (case1 in-repo; case2/case3 data distributed separately)
docs/MERGE_NOTES.md               GPU fast-path optimization design + measured performance
```

---

## Credits

pyNICAM-DC is based on [NICAM-DC](http://r-ccs-climate.riken.jp/nicam-dc/), developed by
the Japan Agency for Marine-Earth Science and Technology (JAMSTEC), the Atmosphere and
Ocean Research Institute (AORI) at The University of Tokyo, the National Institute for
Environmental Studies (NIES), and RIKEN / R-CCS.

The pyNICAM-DC Python port is being developed at the Atmosphere and Ocean Research Institute
(AORI), The University of Tokyo, in collaboration with the Niels Bohr Institute, University
of Copenhagen.

## License

pyNICAM-DC is released under the **BSD 2-Clause License** (see [`LICENSE`](LICENSE)),
consistent with the upstream NICAM-DC (also BSD 2-Clause); the original NICAM development
team's copyright notice is retained.
