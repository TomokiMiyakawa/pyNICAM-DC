#!/bin/bash
#SBATCH --job-name=pnc_g9multi
#SBATCH --partition=TODO_GPU_PARTITION
#SBATCH --account=TODO_PROJECT
#SBATCH --nodes=TODO_N_NODES   # pe8->2, pe20->5, pe40->10
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=pnc_g9multi.%j.out
# T3-T4: multi-node gl09 (2-10 nodes, production resident+fused+FFI).
# Adapted from Miyabi tools/ncclffi/vi_v3b_pe4_ab.pbs run_one. Fill the TODOs
# (partition/account/module names/paths) on site.
set -uo pipefail
G=9; PE=TODO_PE; LSTEP=18   # pe in {8,20,40}; 4 ranks/node
CODE=TODO_REPO_PATH                       # pyNICAM-DC clone (main)
VENV=TODO_VENV_PATH
ROOT=TODO_SWEEP_ROOT                      # unpacked levante_inputs_gl05-09.tar.gz
module load TODO_CUDA_MODULE 2>/dev/null || true
source "$VENV/bin/activate"
export PYTHONPATH="$CODE"
export PYNICAM_NCCLFFI_LIB="$CODE/tools/ncclffi/libncclffi.so"

# --- per-rank single-GPU visibility (4 GPU/node -- REQUIRED, see PORT.md) ---
WRAP="$SLURM_SUBMIT_DIR/gpu_wrap.sh"
cat > "$WRAP" <<'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
EOF
chmod +x "$WRAP"

LAB=lev_g9_pe${PE}
python "$ROOT/scripts/make_config.py" "$G" --backend jax --precision float32 \
    --lstep "$LSTEP" --pe "$PE" --label "$LAB" --output on
RD="$ROOT/run/gl0${G}_${LAB}"
sed -i 's/^PRGout_interval=.*/PRGout_interval=1000/' "$RD/nhm_driver.toml"
cd "$RD"; rm -rf out* *.zarr msg.pe*
source "$CODE/config/production.env"
export PYNICAM_COMM_NO_BARRIER=1
export PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1 PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_WARMUP=3
export PYNICAM_TIMELOOP_DUMP="./out" PYNICAM_PROFILE=timeloop_timing
# first run: also set PYNICAM_NCCLFFI_CKSUM=1 once and parse the audit (PORT.md T2)

srun --ntasks="$PE" "$WRAP" python "$CODE/pynicamdc/nhm/driver/driver-dc.py" \
    --driver-setting ./driversettings.toml < /dev/null
echo "done=$(grep -c 'peacefully done' msg.pe* 2>/dev/null || true)"
