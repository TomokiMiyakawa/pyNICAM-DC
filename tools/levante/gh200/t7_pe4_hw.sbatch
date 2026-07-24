#!/bin/bash
#SBATCH --job-name=t7_pe4hw
#SBATCH --partition=dolpung
#SBATCH --account=mh1571_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --output=t7_pe4hw.%j.out
# T7: why is 1-node gl09 pe4 (0.3262) ~10% slower than Miyabi pe4 (0.2965)?
# At pe4 the COMM share is ~3% (ADOPTION.md) and the JIT=1 chunk is one XLA
# call (no host dispatch) -> the gap must be GPU kernel throughput. pyNICAM
# is memory-bound, so: (1) identify the GPU SKU + caps, (2) measure HBM
# triad bandwidth + matmul TFLOPS per GPU, (3) rerun the pe4 baseline with
# nvidia-smi dmon sampling clocks/power/throttle through steady state.
set -uo pipefail
G=9; PE=4; LSTEP=43
CODE=/work/bb1153/b381557/workclaude/pyNICAM-DC
VENV=/work/bb1153/b381557/workclaude/venv-gh200
ROOT=/work/bb1153/b381557/workclaude
export SLURM_EXPORT_ENV=ALL
MPI=/opt/mpi/openmpi/5.0.6.1.6
export PATH="$MPI/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
export LD_LIBRARY_PATH="$MPI/lib:/opt/prrte/3.0.8/lib:/opt/pmix/5/lib"
unset PYTHONHOME
source "$VENV/bin/activate"
export PYTHONPATH="$CODE"
export PYNICAM_NCCLFFI_LIB="$CODE/tools/ncclffi/libncclffi.so"

WRAP="$SLURM_SUBMIT_DIR/gpu_wrap.sh"
cat > "$WRAP" <<'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
EOF
chmod +x "$WRAP"

echo "=== GPU IDENTIFICATION ==="
srun -N1 -n1 nvidia-smi --query-gpu=name,memory.total,power.limit,power.max_limit,clocks.max.sm,clocks.max.mem --format=csv
srun -N1 -n1 nvidia-smi -q -d CLOCK,POWER | grep -E 'Product Name|SM +:|Memory +:|Power Limit|Requested' | head -12

echo "=== MICROBENCH (per GPU) ==="
srun --ntasks="$PE" --cpus-per-task=72 --cpu-bind=socket "$WRAP" python "$SLURM_SUBMIT_DIR/microbench.py"

echo "=== MODEL RUN gl0$G pe$PE z40 fp32 (baseline repro + dmon) ==="
LAB=t7_pe4hw
python "$ROOT/scripts/make_config.py" "$G" --backend jax --precision float32 \
    --lstep "$LSTEP" --pe "$PE" --label "$LAB" --output off
RD="$ROOT/run/gl0${G}_${LAB}"
sed -i 's/^PRGout_interval=.*/PRGout_interval=1000/' "$RD/nhm_driver.toml"
cd "$RD"; rm -rf out* *.zarr msg.pe*
source "$CODE/config/production.env"
export PYNICAM_COMM_NO_BARRIER=1
export PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1 PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_WARMUP=3
export PYNICAM_TIMELOOP_DUMP="./out" PYNICAM_PROFILE=timeloop_timing

# dmon on the node, sampling sm/mem clocks + power + util once a second
srun --overlap -N1 -n1 bash -c 'nvidia-smi dmon -s puc -d 1 -o T' > "$SLURM_SUBMIT_DIR/t7_dmon.$SLURM_JOB_ID.log" 2>&1 &
DMON=$!
sleep 2

srun --ntasks="$PE" --cpus-per-task=72 --cpu-bind=socket "$WRAP" python "$CODE/pynicamdc/nhm/driver/driver-dc.py" \
    --driver-setting ./driversettings.toml < /dev/null | tee stdout.log | grep -E 'TIMELOOP_CHUNK' | tail -5

# mid/post-run throttle check
srun --overlap -N1 -n1 nvidia-smi -q -d PERFORMANCE | grep -E 'Clocks Event Reasons|Idle|SW Power Cap|HW Slowdown|SW Thermal|HW Thermal|Applications Clocks' | head -16
kill $DMON 2>/dev/null || true

echo "done=$(grep -c 'peacefully done' stdout.log 2>/dev/null || true)/$PE"
python - <<'PYEOF'
import re
vals = [float(m.group(1)) for m in re.finditer(r'per_step=([0-9.]+)', open('stdout.log').read())]
steady = [v for v in vals if v < 3*min(vals)]
print(f"MODEL_RESULT n={len(steady)} min={min(steady):.4f} mean={sum(steady)/len(steady):.4f} (Miyabi ref 0.2965, prior Levante 0.3262)")
PYEOF
echo "ALL_DONE"
