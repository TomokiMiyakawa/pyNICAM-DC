#!/bin/bash
#SBATCH --job-name=@NAME@
#SBATCH --partition=dolpung
#SBATCH --account=mh1571_gpu
#SBATCH --nodes=@NODES@
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=@NAME@.%j.out
# T5: hires weak-scaling rung (gl10/gl11 rl03). Same recipe as
# tools/levante/tpl_gl09_multinode.sbatch, but configs come from
# hires/make_hires_config.py. All Levante traps (PORT.md ON-SITE FINDINGS) applied.
set -uo pipefail
export SLURM_EXPORT_ENV=ALL
source /work/bb1153/b381557/workclaude/gh200/env.sh
ROOT=/work/bb1153/b381557/workclaude
G=@G@; RL=3; PE=@PE@; LSTEP=43; VLAYER=@VLAYER@; VGRID=@VGRID@

WRAP="$SLURM_SUBMIT_DIR/gpu_wrap.sh"
cat > "$WRAP" <<'EOW'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
EOW
chmod +x "$WRAP"

LAB=@NAME@
python "$ROOT/hires/make_hires_config.py" --gl "$G" --rl "$RL" --pe "$PE" \
    --vlayer "$VLAYER" --vgrid "$VGRID" \
    --backend jax --precision float32 --lstep "$LSTEP" --label "$LAB"
RD="$ROOT/run/gl$(printf %02d $G)rl0${RL}_${LAB}"
sed -i 's/^PRGout_interval=.*/PRGout_interval=1000/' "$RD/nhm_driver.toml"
cd "$RD"; rm -rf out* *.zarr msg.pe*
source "$CODE/config/production.env"
export PYNICAM_COMM_NO_BARRIER=1
export PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1 PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_WARMUP=3
export PYNICAM_TIMELOOP_DUMP="./out" PYNICAM_PROFILE=timeloop_timing
srun --ntasks="$PE" --cpus-per-task=72 --cpu-bind=socket "$WRAP" python \
    "$CODE/pynicamdc/nhm/driver/driver-dc.py" \
    --driver-setting ./driversettings.toml < /dev/null 2>&1 | tee stdout.log
echo "done=$(grep -c 'peacefully done' stdout.log)/$PE"
grep TIMELOOP_CHUNK stdout.log | awk -F'per_step=' '{gsub(/s$/,"",$2);print $2}' | sort -n | head -3
