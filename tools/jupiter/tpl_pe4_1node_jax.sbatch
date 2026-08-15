#!/bin/bash -l
#SBATCH --job-name=pnc_g5_jax
#SBATCH --partition=booster
#SBATCH --account=jureap22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72     # one full Grace socket per rank. CRITICAL: the default is
                               # 1 core/rank, which starves NCCL's IB proxy threads + jax
                               # host dispatch -- 4.4x measured on Levante (gl09 pe8).
                               # Intra-node NVLink hides it, so a 1-node run looks fine
                               # and the trap only springs on multi-node.
#SBATCH --gpus-per-node=4      # MUST be per-node, NEVER --gpus-per-task=1: the latter
                               # cgroup-isolates each rank from its peers' devices, so
                               # every intra-node NCCL transport dies at
                               # ncclP2pImportShareableBuffer with Cuda failure 101
                               # 'invalid device ordinal' (only the IB path survives).
                               # Grant all 4 in the cgroup, mask with CUDA_VISIBLE_DEVICES.
#SBATCH --time=00:30:00
#SBATCH --output=/e/project1/jureap22/miyakawa/workClaude1/sweep/logs/pnc_g5_jax.%j.out
# T2-GPU: gl05 rl01 pe04, JAX backend, IDEAL init. PLAIN jax -- no NCCL-FFI, no
# FUSE_TIMELOOP -- so it is comparable to the Miyabi pnc4_jax_g5 runs and to the
# numpy template here. For the production path use tpl_pe4_1node_fused.sbatch.
set -uo pipefail

CODE=/e/project1/jureap22/miyakawa/workClaude1/pyNICAM-DC
ROOT=/e/project1/jureap22/miyakawa/workClaude1/sweep
source "$CODE/tools/jupiter/env.sh"

G=5; PE=4; LSTEP=12; LAB=jup_jax
export OMP_NUM_THREADS=1

# --- per-rank single-GPU visibility (4 GPU/node -- REQUIRED) ---
# Must be set INSIDE the task: SLURM_LOCALID only exists there. jax takes device 0 of
# the visible set, so without this every rank piles onto GPU 0.
WRAP="$ROOT/gpu_wrap.sh"
cat > "$WRAP" <<'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
EOF
chmod +x "$WRAP"

python "$ROOT/scripts/make_config.py" "$G" --backend jax --precision float64 \
       --lstep "$LSTEP" --pe "$PE" --label "$LAB" --init ideal
RD="$ROOT/run/gl0${G}_${LAB}"
cd "$RD" || exit 1
rm -rf testout_tmp.zarr msg.pe* out*

# Verify the mapping on any NEW decomposition -- silent all-on-GPU-0 is the failure mode.
echo "### rank->GPU map ###"
srun --ntasks="$PE" --cpus-per-task=72 --cpu-bind=socket "$WRAP" \
     bash -c 'echo "  rank $SLURM_PROCID on $(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"' 2>&1 | sort

srun --ntasks="$PE" --cpus-per-task=72 --cpu-bind=socket "$WRAP" \
     python "$CODE/pynicamdc/nhm/driver/driver-dc.py" \
     --driver-setting ./driversettings.toml < /dev/null | tee stdout.log

echo "done=$(grep -c 'peacefully done' stdout.log 2>/dev/null || true)/$PE"
echo "### TIMER (per-step; exclude MAIN_Main_Loop_step1 for jit warm-up) ###"
grep -E "MAIN_Main_Loop|MAIN___Dynamics" msg.pe00000000 2>/dev/null | head
# Reference (2026-07-28, job 1086056): done=4/4, ranks on GPUs 0..3,
# MAIN_Main_Loop 22.345 s, step1 18.042 s -> 0.391 s/step steady (Miyabi 0.410).
# mpi4jax warns "Not using CUDA-enabled MPI" here; production.env sets
# MPI4JAX_USE_CUDA_MPI=1. Same-job A/B at gl05 pe4 showed no gain (COMM ~3%).
