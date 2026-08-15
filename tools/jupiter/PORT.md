# JUPITER (JSC) port kit — pyNICAM-DC

Port of the run environment from Levante / Miyabi to **JUPITER** (Jülich
Supercomputing Centre). Same aarch64 Grace + GH200 architecture as both, so the
model code, the venv recipe and the sm_90 NCCL-FFI build all carry over
unchanged — what differs is the **filesystem layout**, the **module stack**, and
the fact that **only `srun` exists**. This kit mirrors `tools/levante/` and
`tools/miyabi/`: env, venv recipe, job templates with the traps inline, and the
gotchas that cost real debugging time.

Validated 2026-07-28 (account `jureap22`, user `miyakawa1`).

## File map

- `env.sh` — module stack + venv + paths. Source this and nothing else.
- `setup_venv.sh` — venv recipe (two-layer: modules supply numpy/scipy/mpi4py,
  venv supplies the pip-only extras + the jax stack).
- `build_ncclffi_jupiter.sh` — aarch64 FFI lib build (g++, login node OK).
- `tpl_pe4_1node_numpy.sbatch` — T2-CPU: gl05 pe4 numpy functional check.
- `tpl_pe4_1node_jax.sbatch` — T2-GPU: gl05 pe4 plain jax (no fusion/FFI).
- `tpl_pe4_1node_fused.sbatch` — T2-PROD: full production fusion + NCCL-FFI,
  with the order-audit arm and the perf arm in ONE job.

Site-specific absolute paths (`/e/project1/jureap22/miyakawa/...`, account) are
hardcoded in the templates — adjust for a different user/project.

## System map

- **JUPITER booster node** = 4× NVIDIA **GH200 120GB** (97,871 MiB visible),
  4 Grace sockets × 72 cores = 288 cores, ~878 GB. Same SKU as Levante's
  `dolpung`, **including the 680 W GPU power cap** (module limit 900 W, default
  1000 W) — see the Levante T7 finding on power-capped SM clocks; it applies here.
- **Partitions**: `booster` (5688 nodes, default) and `largebooster` (5884).
  Account `jureap22`. `develbooster` is a reservation on `jpbo-101-[01-08]`.
- **Scheduler: Slurm. `mpirun` / `mpiexec` DO NOT EXIST — `srun` only.** The
  tutorial scripts fall back to bare python when `mpirun` is absent, which is
  correct here.
- **Login node is aarch64 with a GH200** (`jpbl-s02-*`) → same arch as compute,
  so the venv and the FFI lib can both be built on login. No x86 contamination
  problem (contrast Levante trap #1) and no `--export=NONE` dance needed.
- **Modules**: `Stages/2026` + `GCC/14.3.0` + `ParaStationMPI/5.13.0-1` +
  `mpi4py/4.1.0` + `SciPy-bundle/2025.07`. Python 3.13.5 from EasyBuild.
  `MPI-settings/CUDA` is the default (CUDA-aware); `MPI-settings/UCX` is the
  plain variant.

## TRAP #1 — `/p` does not exist on compute nodes (THE headline)

JUPITER has **two separate GPFS filesystems** and they are NOT aliases:

| path | login nodes | compute nodes |
|---|---|---|
| `/p/project1`, `/p/scratch` (legacy) | ✅ | ❌ **absent entirely** |
| `/e/project1`, `/e/scratch`, `/e/fscratch`, `/e/data1`, `/e/home` | ✅ | ✅ |

`jureap22` has separate space on each. **Everything a batch job touches —
script, venv, run dir, `--output` path — must live under `/e`.**

**Failure signature**: the job fails after ~10 s with `State=FAILED
ExitCode=0:53`, batch step `CANCELLED`, and **no output file at all**.
slurmstepd cannot `chdir` there, so it never gets far enough to write a log.
This looks exactly like a flaky node and reproduces on every node — it cost
two full debugging cycles. Verify cheaply:

```bash
srun --account=jureap22 --partition=booster --nodes=1 --ntasks=1 \
     --time=00:02:00 bash -c 'ls /e; ls /p'
```

Note `$HOME/work` is a symlink into `/p/project1/...`, so it dangles on compute
nodes even though `$HOME` itself is mounted and writable. Module stacks live on
`/e/software`, which IS mounted — only your own tree needs moving.

A venv cannot simply be copied from `/p` to `/e`: `pyvenv.cfg`, `bin/activate`
and script shebangs bake in absolute paths. Recreate it (`setup_venv.sh`).
Likewise `libncclffi.so` carries an **RPATH** to the venv's NCCL — after any
move, rebuild it or the loader silently falls back to the system
`/lib64/libnccl.so.2`, which must never win.

## TRAP #2 — `PYTHONPATH` is load-bearing here

The environment is deliberately **two-layer**: numpy, scipy, mpi4py and pytest
come from the **EasyBuild modules** and reach `sys.path` **only via
`PYTHONPATH`**; the venv holds just the pip-only extras (zarr, numcodecs,
xarray, dask) plus the jax stack.

So any script doing `export PYTHONPATH="$CODE"` (**assignment**, not extension)
deletes the module site-packages. numpy disappears first and everything
importing it follows — the symptom is a baffling
`ModuleNotFoundError: No module named 'numpy'` (or `pytest`) for a package that
imports fine at the prompt. **Always prepend:**

```bash
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
```

This is JUPITER-specific: Levante and Miyabi pip-install the whole stack INTO
the venv, so the assignment is harmless there — which is why the upstream
scripts carry the assumption. Fixed 2026-07-28 in `tutorial/run_tier1_pytest.sh`,
`run_tier2_cpu.sh`, `run_viz.sh`, `run_tier3_gpu.pbs`, `run_viz_gpu.pbs`,
`test/serial_mode_test.py`, `tools/ncclffi/audit_regression.pbs`,
`tools/miyabi/env.sh`, `tools/sweep/setup_regression.sh` and the
`tools/levante/` templates. **~46 Miyabi-era `.pbs` diagnostics under
`tools/ncclffi/` and `tools/sharding_spike/` still assign** — fix on porting.

Corollary: `pip install <pkg>` can be a **silent no-op** when the module stack
already satisfies the requirement (pytest 8.4.1 does). Check
`pip show <pkg> | grep Location` before believing a package is in the venv.

## Setup ladder (new account)

1. **Clone** `main` somewhere under `/e/project1/<project>/<user>/`.
2. **venv**: `bash tools/jupiter/setup_venv.sh /e/.../venv-jupiter`
   (login node OK — same arch). Two-layer, `--system-site-packages`.
3. **NCCL-FFI lib**: `bash tools/jupiter/build_ncclffi_jupiter.sh`
   (login node OK, compile-only; the script asserts it linked the venv NCCL).
4. **Data**: tutorial (`cd tutorial && ./download_inputs.sh`); multi-rank
   benchmark inputs as per `tools/levante/PORT.md` (IDEAL-init, no restart files).
5. **Run** `source tools/jupiter/env.sh` then submit a template.

## Test ladder — RESULTS (2026-07-28, this stack)

| tier | check | result |
|---|---|---|
| T1 | `tutorial/run_tier1_pytest.sh` | **103 passed, 1 skipped** |
| T2 | `tutorial/run_tier2_cpu.sh jw` vs golden | **worst peak-rel 5.256e-08 PASS** — *identical* to the Levante value, i.e. the fp32-golden floor |
| T2-CPU | gl05 pe4 numpy (job 1085888) | 4/4 `peacefully done`; `MAIN_Main_Loop` **17.357 s** (Miyabi 17.143) |
| T2-GPU | gl05 pe4 plain jax (job 1086056) | 4/4; **0.391 s/step** steady (Miyabi 0.410); 4 distinct GPUs verified |
| T2-PROD | gl05 pe4 fused + NCCL-FFI (job 1086073) | audit **2448 pairs / 0 mismatches PASS**; steady **0.0122 s/step** |

Numerics carry over exactly; performance matches or slightly beats Miyabi.

⚠ The 0.0122 s/step figure comes from a **single steady chunk** at `lstep=12`,
where two JIT compiles (14.4 s for the K=4 shape, 13.3 s for the K=1 remainder)
dominate wall time and `MAIN_Main_Loop` (51.7 s) is therefore meaningless as a
perf metric. For a real number use `lstep≈43` and the TIMELOOP_CHUNK steady
mean/min, as `tools/levante/PORT.md` T4 does.

## Gotchas

1. **`--cpus-per-task=72` + `--cpu-bind=socket` is critical** (one Grace socket
   per rank). Carried over from Levante trap #4: the default is 1 core/rank,
   starving NCCL's IB proxy threads and jax host dispatch — 4.4× on multi-node.
   Invisible on 1 node, so it does NOT show up in the gl05 templates' timings.
2. **`--gpus-per-node=4`, NEVER `--gpus-per-task=1`** (Levante trap #3): the
   latter cgroup-isolates each rank from its peers' devices and every intra-node
   NCCL transport dies with `Cuda failure 101 'invalid device ordinal'`. Grant
   all 4 and mask per-rank with `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` inside the
   task (`SLURM_LOCALID` only exists there). The templates echo a rank→GPU map;
   check it on any new decomposition.
3. **`< /dev/null`** on the srun line (stdin trap).
4. **`peacefully done` goes to stdout**, not `msg.pe*` — tee the srun output and
   grep that.
5. **Python warnings go to STDERR**, so they never reach a tee'd `stdout.log`.
   A `grep -c RuntimeWarning stdout.log` check always reports 0 — read the Slurm
   job log instead.
6. **`PRGout_interval=1000`** for benchmark runs (`--output on` then sed), the
   canonical "no snapshot writes" setting. Do **not** use `--output off`, which
   sets `interval=1`: the driver's guard is `n % interval == 1` and `n % 1` is
   always 0, so it also never writes — but it leaves `testout_tmp.zarr` at its
   **NaN `fill_value`**, which reads exactly like a diverged solution. That
   misdiagnosis cost two jobs on 2026-07-28. Judge numerics from T1/T2 or
   `BUDGET_*.log`, never from an `output=off` zarr.
7. **RuntimeWarnings from `mod_src_tracer` / `mod_thrmdyn` under numpy + IDEAL
   init are benign**: IDEAL starts with identically zero tracers, so the flux
   limiter divides by zero-valued denominators. The same class appears in the
   July Miyabi gl09 numpy logs. The jax path raises none (the arithmetic runs
   inside XLA).
8. **mpi4jax does not autodetect CUDA-aware MPI** and warns "Not using
   CUDA-enabled MPI" even under `MPI-settings/CUDA`. `config/production.env`
   sets `MPI4JAX_USE_CUDA_MPI=1`. Same-job A/B at gl05 pe4 showed **no gain**
   (22.345 → 23.469 s, within DVFS noise) because COMM share is ~3% there;
   revisit at the multi-node rungs.
9. Swapping `MPI-settings/CUDA` → `MPI-settings/UCX` changes nothing for the
   numpy path (bit-identical results) — do not chase MPI transport when
   debugging numerics.

## Differences vs Levante / Miyabi (things you do NOT need here)

- No `--export=NONE` / `SLURM_EXPORT_ENV=ALL` (Levante traps #1/#2): the login
  node is aarch64 with no conda toolchain to leak.
- No OpenMPI rpath dance (Levante trap #5): ParaStationMPI comes from modules.
- No self-built Miniforge (Levante trap #7): the EasyBuild Python 3.13.5 is
  aarch64 already. Because python comes from a module it links
  `libpython3.13.so.1.0`, so the module environment must STAY loaded — the
  Levante build script's `unset LD_LIBRARY_PATH` would break the interpreter.
- No cgroup memsw limit like Miyabi's 120 GB/node.

## Open items

- JAX path beyond gl05 untested here: gl09 pe8/pe20/pe40 multi-node (first
  inter-node IB), and the gl10/gl11 hires rungs. Run the order audit once per
  new `(glevel, pe)` per `tools/ncclffi/ADOPTION.md` invariant #2.
- Real timing sweep (`lstep≈43`) to place JUPITER against the Levante/Miyabi
  T4 tables.
- `dmon` SM-clock sampling to confirm the 680 W cap behaves as on Levante.
- fp32 (`--precision float32`) path unexercised on this stack.
