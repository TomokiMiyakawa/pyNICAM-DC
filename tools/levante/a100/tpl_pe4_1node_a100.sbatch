#!/bin/bash
#SBATCH --job-name=pnc_a100_pe4
#SBATCH --partition=gpu
#SBATCH --account=bb1153_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32     # A100 nodes: 128 cores x 2HT = 256 logical, 8 NUMA;
                               # GPUs live on NUMA 1,3,5,7. 32/rank is plenty for the
                               # NCCL proxies (the GH200 1-core trap, PORT.md #4).
#SBATCH --gpus-per-node=4      # NOT --gpus-per-task=1 (PORT.md trap #3)
#SBATCH --time=00:30:00
#SBATCH --export=NONE          # PORT.md trap #1 (conda toolchain leak)
#SBATCH --output=pnc_a100_pe4.%j.out
# A100 (x86) equivalent of tools/levante/tpl_pe4_1node.sbatch: gl05 pe4 functional
# check, production + fused + FFI. Env/venv differ (a100/env.sh, scratch venv);
# model code and inputs are the same ones the GH200 runs use.
set -uo pipefail
export SLURM_EXPORT_ENV=ALL    # PORT.md trap #2
source /work/bb1153/b381557/workclaude/a100/env.sh
ROOT=/work/bb1153/b381557/workclaude
G=5; PE=4; LSTEP=8

WRAP="$SLURM_SUBMIT_DIR/gpu_wrap_a100.sh"
cat > "$WRAP" <<'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
EOF
chmod +x "$WRAP"

LAB=a100_pe4
python "$ROOT/scripts/make_config.py" "$G" --backend jax --precision float64 \
    --lstep "$LSTEP" --pe "$PE" --label "$LAB" --output on
RD="$ROOT/run/gl0${G}_${LAB}"
sed -i 's/^PRGout_interval=.*/PRGout_interval=1000/' "$RD/nhm_driver.toml"
cd "$RD"; rm -rf out* *.zarr msg.pe*
source "$CODE/config/production.env"
export PYNICAM_COMM_NO_BARRIER=1
export PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1 PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_WARMUP=3
export PYNICAM_TIMELOOP_DUMP="./out" PYNICAM_PROFILE=timeloop_timing
export PYNICAM_NCCLFFI_CKSUM=1   # first A100 run: audit the new fabric (NV4) once

srun --ntasks="$PE" --cpus-per-task=32 "$WRAP" python \
    "$CODE/pynicamdc/nhm/driver/driver-dc.py" \
    --driver-setting ./driversettings.toml < /dev/null 2>&1 | tee stdout.log
echo "done=$(grep -c 'peacefully done' stdout.log)/$PE"
python - <<'PY'
import re
send, recv = {}, {}
for l in open('stdout.log'):
    m = re.search(r'NCCLFFI_CKSUM call=(\d+) plan=(\d+) rank=(\d+) peer=(\d+) send=([0-9a-f]+) recv=([0-9a-f]+)', l)
    if not m: continue
    c, p, r, q = map(int, m.group(1, 2, 3, 4))
    send[(c, r, q)] = m.group(5); recv[(c, q, r)] = m.group(6)
bad = sum(1 for k, cs in send.items() if recv.get(k) != cs)
print(f"pairs={len(send)} MISMATCHES={bad} -> {'AUDIT CLEAN' if bad == 0 and send else 'FAIL'}")
PY
