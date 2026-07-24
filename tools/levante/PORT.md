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

---

## ON-SITE FINDINGS (2026-07-24, first Levante session)

All resolved values, in the order the open items asked:

- **Partition `dolpung`**: 42 nodes x 4 GH200 120GB (aarch64, RHEL 9.5, driver
  580.159; 4 Grace sockets x 72C, 877GB). Account **`mh1571_gpu`** (needs
  `dolpunguser` group). Walltime cap **8 h** (also the default). MaxNodes
  unlimited. NB the `gpu` partition is x86+A100 -- NOT this one.
- **MPI**: `/opt/mpi/openmpi/5.0.6.1.6` (CUDA-aware, UCX, PMIx 5). No module
  needed; but see the LD_LIBRARY_PATH gotcha below.
- **No CUDA module exists or is needed**: nodes ship the CUDA 12.9 *runtime*
  only (`/usr/local/cuda-12.9/lib64`, no `bin/`). nvcc: the pip
  `nvidia-cuda-nvcc-cu12` wheel ships only ptxas/nvvm, no nvcc driver binary
  -- `build_ncclffi_levante.sh` therefore compiles with **g++** (ncclffi.cu
  has no device code; sm_90 was never material for that TU) against the pip
  wheels' split headers (cuda_runtime + cuda_nvcc + cuda_cccl) with an rpath
  pinned to the venv NCCL (a system `/lib64/libnccl.so.2` exists and must NOT
  win). Network: compute nodes reach pypi directly.
- **Interconnect**: quad-rail NDR200 -- one dedicated 200G HCA per GH200
  (`mlx5_0..3`), NCCL uses NET/IB + GPU Direct RDMA out of the box, rails
  auto-aligned per GPU. Per-GPU IB bandwidth thus matches Miyabi; NVLink
  (NV6 all-to-all) is a pure bonus on top.

### Trap ledger (each cost real debugging time; all are in the templates now)

1. **x86 login-node contamination**: the login node is x86_64 and its profile
   exports a full x86 conda toolchain (CC, CFLAGS=-march=nocona,
   `_CONDA_PYTHON_SYSCONFIGDATA_NAME`, CONDA_PREFIX, ...). srun/sbatch inherit
   it onto the aarch64 nodes; source builds then exec x86 `cc` ("Exec format
   error") or conda writes into the system env. Fix: `#SBATCH --export=NONE`
   in every job; `env -i` re-exec in setup scripts.
2. **`--export=NONE` side effect**: job steps then default to
   `SLURM_EXPORT_ENV=NONE` and srun children lose the PATH/venv ("bash: No
   such file or directory"). Re-export `SLURM_EXPORT_ENV=ALL` inside the job.
3. **GPU binding -- use `--gpus-per-node=4` + CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
   wrapper (the original Miyabi recipe), NEVER `--gpus-per-task=1`**: the
   latter cgroup-isolates each rank from its peers' devices and every
   intra-node NCCL transport (P2P and SHM) dies with `Cuda failure 101
   'invalid device ordinal'` in ncclP2pImportShareableBuffer; only the IB
   path survives. cgroup device limits bite at open(), not readdir, so
   `ls /dev/nvidia*` looks identical in both modes. Verified with the
   standalone reproducer (nccl_min.cc, AllReduce+SendRecv, also fails with
   the system NCCL 2.27 -> not a pip-NCCL bug).
4. **CPU allocation -- `--cpus-per-task=72` (one Grace socket per rank) +
   `--cpu-bind=socket` is CRITICAL**: default is 1 core/rank, which starves
   NCCL's IB proxy threads + jax host dispatch. Same-job A/B (gl09 pe8):
   0.97 -> 0.2185 s/step, a **4.4x** swing. The trap is invisible on 1 node
   (NVLink P2P needs no CPU proxy) and *worsens with node count* -- the
   uncorrected sweep even showed inverted scaling (pe40 slower than pe8).
5. **OpenMPI has no wrapper rpath**: put `$MPI/lib:/opt/prrte/3.0.8/lib:`
   `/opt/pmix/5/lib` on LD_LIBRARY_PATH or mpi4py's import dies on
   libmpi.so.40 and mpirun on libprrte.so.3.
6. **mpi4jax needs `pip install nanobind` first** (build-time dep;
   `--no-build-isolation` means it must pre-exist in the venv).
7. **python 3.11 via own Miniforge-aarch64** into $WORK (system python is 3.9;
   the venv MUST be created on a dolpung node, the login node is x86).
8. **'peacefully done' goes to stdout**, not msg.pe* -- tee srun output and
   grep that (the old `grep msg.pe*` check silently reports 0).
9. **Input bundle is IDEAL-init**: data/restart/ is deliberately absent;
   make_config.py grew `--init ideal|restart` (default ideal) and the config
   template gained `@INPUT_IO_MODE@` for it.

### T1-T3 results (main @ 20fe8e3, this stack)

- T1: pytest 46/46; tier2 jw vs golden worst peak-rel 5.256e-8 (= the
  expected fp32-golden floor). Numerics carry over cleanly.
- T2 gl05 pe4 (1 node x 4 GPU, production+fused+FFI): 4/4 peacefully done,
  cksum audit 4,896 pairs 0 mismatches (NVLink pack/wire/unpack).
- T3: gl09 pe8 (2N) 8/8; pe20 (5N) 20/20 + cksum 113,832 pairs 0 mismatches
  (inter-node IB wire); pe40 (10N) 40/40.
- T4 (gl09 z40 fp32, lstep=43, steady chunks; first sweep invalidated by
  trap #4, this one with socket binding):
  | config | nodes | s/step min / mean | Miyabi ref |
  |---|---|---|---|
  | pe4  | 1  | 0.3262 / 0.3272 | 0.2965 |
  | pe8  | 2  | 0.2124 / 0.2171 | — |
  | pe20 | 5  | 0.0791 / 0.0805 | — |
  | pe40 | 10 | **0.0726** / 0.0764 | ~0.079 |
  **Headline answered: YES** -- at equal GPU count (40), 4xNVLink/node beats
  Miyabi's 1 GPU/node fabric by ~8%. Strong scaling pe4->pe40 is 4.49x on
  Levante vs 3.75x on Miyabi.

### Bonus: A100 (x86, `gpu` partition) cross-arch sweep

Same clone, same inputs, zero model-code changes -- only venv (x86 wheels,
mpi4py vs spack openmpi-4.1.2-mnmady) + libncclffi_x86.so differ. 63 nodes x
4 A100 (59x80GB), NV4 all-to-all, 2 NICs/node, cpus-per-task=32, account
bb1153_gpu, 12h cap. Setup in `workclaude/a100/` (venv on /scratch -- the
/work bb1153 quota was 100% full on 2026-07-24). gl05 pe4 cksum audit clean.
  | gl09 | A100 s/step | GH200 s/step | GH200 advantage |
  |---|---|---|---|
  | pe4  | 0.6822 | 0.3262 | 2.09x |
  | pe8  | 0.3720 | 0.2124 | 1.75x |
  | pe20 | 0.1853 | 0.0791 | 2.34x |
  | pe40 | 0.0867 | 0.0726 | 1.19x |
  A100 strong-scales BETTER (7.87x vs 4.49x, slower GPU -> smaller halo
  fraction) and its pe40 nearly matches Miyabi's GH200 pe40 (~0.079). Two open threads for the next session:
  (a) single-node pe4 is ~10% SLOWER than Miyabi's 4-node pe4 despite NVLink
  (host-side/jax-version difference? worth a profile); (b) pe20->pe40 gains
  only 9% -- the halo-latency floor is near, so the mnginfo NVLink-locality
  experiment (see "4-GPUs-per-node specifics") is the next perf increment.
