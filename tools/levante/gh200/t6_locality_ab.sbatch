#!/bin/bash
#SBATCH --job-name=t6_locab
#SBATCH --partition=dolpung
#SBATCH --account=mh1571_gpu
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72     # one Grace socket per rank -- CRITICAL for multi-node
#SBATCH --gpus-per-node=4      # NOT --gpus-per-task=1 (cgroup breaks NCCL intra-node)
#SBATCH --time=00:45:00
#SBATCH --export=NONE
#SBATCH --output=t6_locab.%j.out
# T6: mnginfo NVLink-locality A/B (PORT.md "4-GPUs-per-node specifics").
# The identity mnginfo + block rank placement already gives max locality
# (each node = one 2x2 diamond, 80/160 halo legs intra-node). The B arm
# scatters each diamond over 4 nodes via srun --distribution=cyclic
# (4/160 legs intra-node) to quantify what NVLink locality is worth at
# gl09 pe40 z40 fp32. Three arms in ONE job (verdict rule): block, cyclic,
# block again (drift check). Baseline ref from T4b: 0.0726 s/step min.
set -uo pipefail
G=9; PE=40; LSTEP=43
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

LAB=t6_locab
python "$ROOT/scripts/make_config.py" "$G" --backend jax --precision float32 \
    --lstep "$LSTEP" --pe "$PE" --label "$LAB" --output off
RD="$ROOT/run/gl0${G}_${LAB}"
sed -i 's/^PRGout_interval=.*/PRGout_interval=1000/' "$RD/nhm_driver.toml"
cd "$RD"
source "$CODE/config/production.env"
export PYNICAM_COMM_NO_BARRIER=1
export PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1 PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_WARMUP=3
export PYNICAM_TIMELOOP_DUMP="./out" PYNICAM_PROFILE=timeloop_timing

run_arm () {  # $1 = arm name, $2 = srun --distribution value
  echo "=== ARM $1 dist=$2 $(date +%T) ==="
  rm -rf out* *.zarr msg.pe*
  # rank->node map proof: localid/node per rank, sorted by rank
  srun --ntasks="$PE" --distribution="$2" bash -c 'echo "MAP rank=$SLURM_PROCID node=$(hostname) localid=$SLURM_LOCALID"' | sort -t= -k2 -n | head -8
  srun --ntasks="$PE" --distribution="$2" --cpus-per-task=72 --cpu-bind=socket \
      "$WRAP" python "$CODE/pynicamdc/nhm/driver/driver-dc.py" \
      --driver-setting ./driversettings.toml < /dev/null | tee "stdout_$1.log" | grep -E 'TIMELOOP_CHUNK|peacefully done' | tail -3
  echo "ARM=$1 done=$(grep -c 'peacefully done' "stdout_$1.log" 2>/dev/null || true)/$PE"
  python - "$1" <<'PYEOF'
import sys, re
vals = [float(m.group(1)) for m in re.finditer(r'per_step=([0-9.]+)', open(f"stdout_{sys.argv[1]}.log").read())]
steady = [v for v in vals if v < 3*min(vals)]  # drop compile-inclusive first chunks
print(f"ARM_RESULT {sys.argv[1]} n={len(steady)} min={min(steady):.4f} mean={sum(steady)/len(steady):.4f}")
PYEOF
}

run_arm block  block
run_arm cyclic cyclic
run_arm block2 block
echo "ALL_ARMS_DONE"
