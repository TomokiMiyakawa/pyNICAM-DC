# Miyabi (U. Tokyo / JCAHPC) port kit — pyNICAM-DC

Miyabi-G is the machine most of this repository's GPU results were produced
on. This kit makes the environment reproducible for a NEW user/account the
same way `tools/levante/` does for DKRZ: venv recipe, runtime env, job
templates with the trap explanations inline, and the hard-won gotchas.

## System map

- **Miyabi-G node** = 1× NVIDIA GH200 (Grace aarch64 72-core + H100-class GPU,
  ~96GB HBM), **120 GB Grace RAM per node enforced by cgroup memsw** (exceed →
  silent SIGKILL, shows as bare `Terminated`). **1 MPI rank per node/GPU** is
  the production layout (no MPS needed).
- **Login node is aarch64 too** → the venv built on login runs on compute
  nodes unchanged. (Miyabi-C is x86 — do NOT run this venv there.)
- Scheduler: **PBS Pro**. Queues: `debug-g` (1–2 nodes, minutes-class, fast
  turnaround), `regular-g` (routes by size into medium-g/large-g). Submit with
  `qsub -q <queue> job.pbs`; every job needs `#PBS -W group_list=<project>`
  (this kit's templates say `gj37` — change to your project).
- Software: `module load nvidia/25.9` (nvhpc: CUDA, HPC-X OpenMPI, nvcc for
  sm_90). Python via `module load python/3.11.15` for the venv build.

## Setup ladder (new account)

1. **Clone** `main`. All templates assume the repo at `$CODE`.
2. **venv**: `bash tools/miyabi/setup_venv.sh /path/to/venv-gh200`
   (login node OK — same arch). Stages: jax[cuda12] → mpi4py built against
   HPC-X `mpicc` (`--no-binary`) → `mpi4jax==0.9.0.post1` → io stack
   (`zarr<3`!) → NCCL-FFI extension (`tools/ncclffi/build_ncclffi.sh`,
   compile-only, sm_90). The frozen reference stack is
   `tools/levante/requirements-gh200.txt` (same jax 0.10.2 line).
3. **Datasets**:
   - Tutorial/validation (gl05, 14 cases + goldens, ~122MB):
     `cd tutorial && ./download_inputs.sh`
   - Benchmark inputs (gl09–gl11, IDEAL init, no restart): built from raw
     panNICAM `grid.rgn` files — see `fromwhale/pynicam-sweep/hires/`
     (`build_hires_inputs.py` + `make_hires_config.py`) or the prebuilt
     tarball listed in `tools/levante/PORT.md` (5.8GB, gl10/gl11).
4. **Smoke**: `qsub -q debug-g tools/miyabi/tpl_1node_smoke.pbs` — jw on the
   full GPU stack (resident + FUSE_TIMELOOP + jit chunk), ~5 min.
5. **Validate**: `qsub -q debug-g tools/miyabi/tpl_tier2_cpu.pbs` — the
   14-case numpy-CPU science suite vs bundled goldens (compute node, NOT
   login — see gotchas).
6. **Benchmark**: `qsub -q regular-g tools/miyabi/tpl_gl11_pe64.pbs` after
   staging hires inputs.

## Run recipe (what every working job does)

```bash
module load nvidia/25.9
source $VENV/bin/activate
# jax's pip CUDA libs must be on LD_LIBRARY_PATH:
NVLIBS=$(python -c "import nvidia,glob,os;b=os.path.dirname(nvidia.__file__);print(':'.join(sorted(glob.glob(b+'/*/lib'))))")
export LD_LIBRARY_PATH="$NVLIBS:$LD_LIBRARY_PATH"
export PYTHONPATH=$CODE
export PYNICAM_NCCLFFI_LIB=$CODE/tools/ncclffi/libncclffi.so
source $CODE/config/production.env        # the validated GPU gate set
export PYNICAM_COMM_NO_BARRIER=1
export PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1 PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_WARMUP=3
unset OMPI_MCA_mca_base_env_list          # module sets a stale env-forward list
mpirun --mca coll ^hcoll \
       -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH -x <each PYNICAM_*/XLA_*/NCCL_* var> \
       -np $PE python $CODE/pynicamdc/nhm/driver/driver-dc.py \
       --driver-setting ./driversettings.toml < /dev/null
```
Or source `tools/miyabi/env.sh` which does the env part.

## Reference numbers (this repo's `main`, fp32 z40, 1 rank/GPU, s/step)

| config | s/step |
|---|---|
| gl09 pe4 | 0.2965 |
| gl09 pe40 | ~0.079 |
| gl11 pe64 | 0.3016 |
| gl11 pe80 z78 | 0.4798 |

Weak scaling gl09-4GPU → gl11-64GPU (27.5M cells/GPU): **98.3%** (NCCL-FFI).
Memory (gl11 pe64): device chunk-module 17.77 GiB/rank (z40), 25.0 (z78);
host MaxRSS ≈ 60 GB/rank (z40), ≈ 86 GB/rank (z78 pe80). **z78 does NOT fit
pe64** (host+device both) — use pe80.

## Gotchas (each one cost a session to learn)

1. **Login-node 14GB watchdog**: heavy python (zarr loads, golden generation,
   rendering) is SIGTERM-killed >14GB with a bare `Terminated`. Run on a
   compute node (debug-g).
2. **mpirun stdin**: always `< /dev/null` — an attached stdin hangs ranks.
3. **`unset OMPI_MCA_mca_base_env_list`** before mpirun, and forward env
   explicitly with `-x` (the module's default forward list is incomplete).
4. **`--mca coll ^hcoll`**: hcoll is broken with this stack.
5. **cgroup memsw kill**: host RSS over ~120GB/node dies silently. At z78
   NEVER enable XLA text dumps (`--xla_dump_to` writes 8000+ files and the
   dump inflates host RSS past the limit — dump at z40 and scale-reason).
6. **GPU pool ceiling**: leave `XLA_CLIENT_MEM_FRACTION` unset (→0.75) so
   NCCL / CUDA-aware-MPI staging keeps headroom; raise only on OOM
   (production.env has the full rationale).
7. **nsys**: fine at z40; crushes z78 (memory). For steady-state numbers use
   `--delay` past compile.
8. **driversettings.toml pointer trap**: the run dir's `driversettings.toml`
   points at the case config — regenerating a config does not retarget an old
   run dir. Always `cd` into the freshly generated run dir.
9. **Compile time**: gl11 first-chunk compile is minutes; benchmark timings
   must skip the first chunk (harnesses here use fastest-N chunk means).
10. **PBS output buffering**: job stdout (`*.o<jobid>`) appears only at job
    end; poll the harness's own log files for live progress.

## File map

- `setup_venv.sh` — staged venv build (jax → mpi4py(no-binary) → mpi4jax →
  io → ncclffi); each stage verified, io failures non-fatal.
- `env.sh` — source after activating the venv: module, NVLIBS, PYTHONPATH,
  NCCLFFI path.
- `tpl_1node_smoke.pbs` — debug-g GPU smoke (jw, full production stack).
- `tpl_tier2_cpu.pbs` — 14-case numpy science validation on a compute node.
- `tpl_gl11_pe64.pbs` — production benchmark template (fp32 z40, LSTEP
  timing, fastest-100 chunk mean).
- Related elsewhere in the repo: `config/production.env` (gate set + memory
  policy), `tools/ncclffi/ADOPTION.md` (COMM transport contract + audit),
  `tutorial/` (3-tier validation kit), `tools/levante/requirements-gh200.txt`
  (frozen package versions).
