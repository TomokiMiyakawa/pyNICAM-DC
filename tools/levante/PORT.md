# Levante (DKRZ) port — handoff from Miyabi (2026-07-24)

Goal: multi-node functional check + performance test of pyNICAM-DC on Levante
GH200 nodes (**4 GPUs/node** — vs Miyabi's 1 GH200/node). Claude Code runs on
the Levante login node; node allocation limits to be confirmed on site.

## Where we come from (Miyabi state)

- Code: `main @ 20fe8e3` (vi-stack campaign merged: V3a+V4 component threading,
  TIMELOOP_JIT=1 default). All reference numbers below are this commit.
- Reference performance (GH200, fp32 z40, FUSE_TIMELOOP + NCCL-FFI,
  1 rank/GPU, mean s/step):
  | config | s/step |
  |---|---|
  | gl05 pe4 | (functional only) |
  | gl09 pe4 | 0.2965 |
  | gl09 pe40 | ~0.079 (pre-refactor sweep value) |
  | gl11 pe64 | 0.3016 |
  | gl11 pe80 z78 | 0.4798 |
  Weak scaling gl09-4GPU vs gl11-64GPU (27.5M cells/GPU) = **98.3%**.
- Same arch as Miyabi (aarch64 Grace + Hopper) → venv recipe and sm_90 build
  carry over; only MPI stack, module names, and the scheduler (Slurm) differ.

## Setup ladder (first Levante session)

1. **Survey**: `module avail` (CUDA / nvhpc / OpenMPI versions), partition
   names, `sinfo`, project/account id, filesystem layout (scratch/work).
2. **Clone**: `git clone git@github.com:TomokiMiyakawa/pyNICAM-DC.git`
   (or https). Use `main`.
3. **venv** (aarch64, mirror of Miyabi's `venv-gh200`): python 3.11,
   `pip install -r tools/levante/requirements-gh200.txt` EXCEPT `mpi4py` /
   `mpi4jax`, which must be built against Levante's system MPI:
   `MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py==4.1.2` then
   `pip install --no-build-isolation mpi4jax==0.9.0.post1`.
   jax is `jax[cuda12]==0.10.2` (pip wheels are fine; NCCL comes via
   `nvidia-nccl-cu12`).
4. **NCCL-FFI lib**: edit `tools/ncclffi/build_ncclffi.sh` (VENV path + module
   load line), run it (login node OK, compile only, `-arch=sm_90`). Export
   `PYNICAM_NCCLFFI_LIB=<repo>/tools/ncclffi/libncclffi.so` in job scripts.
5. **Data**:
   - tutorial dataset (single-rank cases + goldens): `cd tutorial &&
     ./download_inputs.sh` (public host, 122MB).
   - multi-rank benchmark inputs: scp from Miyabi
     `/work/gj37/c24028/workforclaude/levante_inputs_gl05-09.tar.gz` (848MB;
     contains `data/{boundary,mnginfo,vgrid*.json}` for gl05–gl09 ×
     pe04/08/20/40, `scripts/make_config.py`, `config/` templates — the
     `pynicam-sweep` ROOT layout minus restart files; all benchmark configs
     are IDEAL-init so no restart needed). Unpack to some `$SWEEP_ROOT` and
     point the job templates' `ROOT` at it. gl10/gl11 inputs NOT included
     (several GB) — fetch later if the sweet-spot study is extended.

## 4-GPUs-per-node specifics (NEW vs Miyabi)

- **Each rank must see exactly ONE GPU** (every Miyabi harness assumes this
  implicitly — jax takes device 0 of the visible set). On Slurm either use
  `--gpus-per-task=1` with task/GPU binding, or the wrapper in the templates:
  `export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`. VERIFY with a 1-node pe4 run
  that 4 distinct GPUs are busy (`nvidia-smi` on the node / job stats).
- NCCL automatically uses **NVLink intra-node** — the FFI transport should
  get faster halo legs within a node with zero code change.
- **mnginfo locality is now a real experiment axis** (it was noise on Miyabi's
  1 GPU/node): assigning 4 contiguous regions-blocks to the 4 ranks of one
  node can pull a large share of halo traffic onto NVLink. The canonical
  mnginfo generator lives in the sweep scripts; a locality-permuted variant
  is a candidate perf increment AFTER the baseline sweep.
- MPI ranks-per-node=4: watch CPU binding (Grace cores per rank) and NUMA;
  start with the scheduler defaults, only tune if imbalance shows.

## Test ladder (functional -> perf)

T1 single-rank science: `tutorial/` tier1 pytest + a tier2 case (jw) vs
   golden — expect peak-rel ~5e-8 (fp32-golden floor). Proves numerics on the
   new stack (compiler/library versions may differ → small fp drift is the
   thing to check first).
T2 1 node × 4 GPU (gl05 pe4, production env + FFI): adapts
   `tools/ncclffi/vi_v3b_pe4_ab.pbs`'s run_one shape → Slurm. Turn on the
   cksum audit arm once (`PYNICAM_NCCLFFI_CKSUM=1`, see
   `tools/ncclffi/vi_z78_base80.pbs` for the audit parser) — this validates
   pack/wire/unpack on the NEW fabric.
T3 multi-node: gl09 pe8 (2 nodes) then pe20 (5) / pe40 (10) — first
   inter-node IB + NVLink mixed topology. Functional bar: 'peacefully done'
   all ranks + cksum audit clean + (optional) cross-check vs a pe4 run of
   the same case (COMM-count-independent results are bit-comparable on numpy
   only; on GPU compare against Miyabi's z40 numbers for sanity, not bits).
T4 timing sweep: gl09 pe4/8/20/40 z40 fp32, same harness metric
   (TIMELOOP_CHUNK steady mean/min) → table vs the Miyabi reference above.
   Interesting headline: does 4×NVLink/node beat Miyabi's 1 GPU/node fabric
   at equal GPU counts?

## Gotchas carried over from Miyabi (all still apply)

- `mpirun ... < /dev/null` (stdin trap), `PRGout_interval=1000` sed (else the
  fused chunk silently disengages), `PYNICAM_COMM_NO_BARRIER=1` in all fused
  runs, `source config/production.env` for the production gate set.
- z78 runs: NO `PYNICAM_FORCE_EAGER_WARM` (per-chunk re-lowering OOMs);
  TIMELOOP_JIT=1 (now the default).
- Peak-normalized metric for field comparisons (per-cell relative explodes on
  near-zero RHOGW).
- nsys crushes z78 — profile z40 only.
- Perf verdicts ONLY from same-job A/B arms (node-set variance is real).

## Open items to resolve on site

- Account/partition names, node allocation limit, walltime caps.
- MPI flavor + CUDA-awareness (the host COMM fallback path needs plain MPI
  only; the FFI path needs NCCL, which is self-contained via pip).
- Whether `module` names in `build_ncclffi.sh` / job templates need changes.
