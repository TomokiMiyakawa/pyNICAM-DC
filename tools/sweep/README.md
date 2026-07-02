# Resolution sweep harness (ICOMEX_JW, rl01, pe04)

The **single source of truth for the resolution-dependent settings** used in the
CPU sweep (and to reproduce on GPU). The settings live in `make_config.py` +
`nhm_driver.template.toml`; this README also tabulates them for quick reference.

## Turnkey setup (assemble a runnable sweep root + fetch data)

`make_config.py` / `run_sweep.sh` expect a sweep-root layout (`scripts/ config/ data/
run/ code/`). `setup_sweep.sh` builds that from this checkout and downloads the datasets
(the heavy developer tier; the ~10 GB gl05–gl09 boundary/restart + numpy golds):

```bash
bash tools/sweep/setup_sweep.sh              # all glevels (~10 GB), root ./pynicam-sweep
bash tools/sweep/setup_sweep.sh 07 08        # a subset
PYNICAM_SWEEP_ROOT=/scratch/sweep bash tools/sweep/setup_sweep.sh 07
```

Then, from the sweep root it prints:

```bash
scripts/run_sweep.sh                                   # numpy reference sweep
source <repo>/config/production.env                    # GPU fast path (fused+resident stack)
BACKEND=jax GLEVELS="7 8" scripts/run_sweep.sh
python <repo>/pynicamdc/nhm/dynamics/proto/cmp_prec.py \
       run/golds/gl07_numpy_gold.zarr run/gl07_jax/testout_tmp.zarr --rtol 1e-9   # validate
```

(Data only: `tools/fetch_sweepdata.sh`. Lite quick-start data, case2/case3: `tools/fetch_testdata.sh`.)

## Resolution-dependent settings (per glevel; rl01, vlayer=40, pe04, test_case=1)

| glevel | gall_1d | dtl (s) | gamma_h = alpha_d | timing lstep |
|--------|---------|---------|-------------------|--------------|
| gl05   | 18      | 1200    | 1.20e16           | 12           |
| gl06   | 34      | 600     | 1.50e15           | 12           |
| gl07   | 66      | 300     | 2.00e14           | 12           |
| gl08   | 130     | 150     | 2.50e13           | 6            |
| gl09   | 258     | 75      | 3.00e12           | 4            |

- **dtl = 1200 / 2^(glevel-5)** (CFL: halve per glevel). A single fixed dtl blows up
  the fine grids -- it MUST scale.
- **gamma_h = alpha_d** (hdiff / divdamp, DIRECT, lap_order=2) are resolution-dependent,
  matched to the f90 ICOMEX_JW namelists. gl05 is the original validated value. These are
  NOT a fixed constant.
- **timing lstep** is tapered so the heavy grids stay affordable; per-step metric excludes
  the first step, so fewer steps is fine. Use the same counts to line up with the CSVs.
- `gall_1d = 2^(glevel-1) + 2` at rl01. Per-region (per-rank, pe04) size is set by
  glevel-rlevel; on GPU prefer the larger grids (they fill the device).

## Files
- `make_config.py` — emits `run/gl0N_<label>/{nhm_driver.toml,driversettings.toml}` for a
  glevel. Encodes dtl + gamma_h/alpha_d (the table above). `--backend numpy|jax`,
  `--lstep N`, `--output on|off`, `--label LABEL` (run-dir/CSV suffix).
  **NOTE: the npz input paths are built relative to the harness root** (boundary/restart/
  vgrid/mnginfo under `data/`). Point them at your rsync'd data dir on the GPU box.
- `nhm_driver.template.toml` — the config template (@GLEVEL@/@DTL@/@GAMMA_H@/@ALPHA_D@/
  @LSTEP@/paths/...).
- `run_sweep.sh` — loop glevels, run `mpirun -np 4 driver-dc.py`, collect timers.
  Env: `GLEVELS BACKEND LSTEP NPROC RUNLABEL` (+ `PYNICAM_*` toggles via the environment).
- `collect_timers.py` — parse the PROF "Computational Time Report" -> CSV (per-step excl.
  step 1). `--backend`, `--label`.
- `compare_f90.py` — join `timers_*.csv` with `f90_reference.csv` -> per-step slowdown table.
- `f90_reference.csv` — f90 NICAM-DC reference (48 steps), the comparison target.
- `miyabi_sweep_fast.pbs` — Miyabi-G (GH200) PBS jobscript: fastest full sweep, jax GPU +
  `config/production.env` optimized stack, pe04. `qsub -v GLEVELS=...,LSTEP=...,SWEEP_ROOT=...`.
- `miyabi_sweep_prof.pbs` — Miyabi-G PBS jobscript: profiled single-glevel run
  (`PYNICAM_PROF_PERSTEP=1` per-step timers + Nsight Systems on rank 0).

## Reproduce on GPU (sketch)
1. rsync the inputs (`data/`) + the CPU golds; point `make_config.py` paths at them.
2. `python make_config.py 7 --backend jax --lstep 12` (etc. per glevel).
3. Run the driver with your GPU launch (CUDA-aware MPI, 1 rank/GPU, on-device COMM env).
4. `collect_timers.py` -> CSV; `compare_f90.py` vs `f90_reference.csv` and the CPU golds
   (machine-precision via `proto/cmp_prec.py`).
