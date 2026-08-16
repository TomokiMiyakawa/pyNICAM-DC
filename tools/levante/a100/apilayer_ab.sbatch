#!/bin/bash
#SBATCH --job-name=pnc_apilayer_ab
#SBATCH --partition=gpu
#SBATCH --account=bb1153_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --output=pnc_apilayer_ab.%j.out
# api-layer GPU verification (GPU_VERIFICATION.md debts on A100):
#   A: main      jax device-resident per-step   -> zarr
#   B: api-layer jax device-resident per-step   -> zarr   (A vs B exact?)
#   C: api-layer jax fused + NCCL-FFI cksum     -> zarr   (B vs C exact? audit clean?)
set -uo pipefail
export SLURM_EXPORT_ENV=ALL
source /work/bb1153/b381557/workclaude/a100/env.sh
ROOT=/work/bb1153/b381557/workclaude
CODE_MAIN=/scratch/b/b381557/pyNICAM-DC-main
CODE_API=/scratch/b/b381557/pyNICAM-DC
G=5; PE=4; LSTEP=12

WRAP="$SLURM_SUBMIT_DIR/gpu_wrap_ab.sh"
cat > "$WRAP" <<'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
EOF
chmod +x "$WRAP"

run_one() {  # code_dir label fused
  local code=$1 lab=$2 fused=$3
  python "$ROOT/scripts/make_config.py" "$G" --backend jax --precision float64 \
      --lstep "$LSTEP" --pe "$PE" --label "$lab" --output on
  local RD="$ROOT/run/gl0${G}_${lab}"
  ( cd "$RD" && rm -rf out* *.zarr msg.pe*
    source "$code/config/production.env"
    export PYNICAM_COMM_NO_BARRIER=1
    if [ "$fused" = 1 ]; then
      export PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1
      export PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_WARMUP=3
      export PYNICAM_TIMELOOP_DUMP="./out" PYNICAM_PROFILE=timeloop_timing
      export PYNICAM_NCCLFFI_CKSUM=1
    fi
    export PYTHONPATH="$code"
    srun --ntasks="$PE" --cpus-per-task=32 "$WRAP" python \
      "$code/pynicamdc/nhm/driver/driver-dc.py" \
      --driver-setting ./driversettings.toml < /dev/null 2>&1 | tee stdout.log )
  echo "== $lab done=$(grep -c 'peacefully done' "$RD/stdout.log")/$PE"
}

run_one "$CODE_MAIN" ab_main 0
run_one "$CODE_API"  ab_api  0
run_one "$CODE_API"  ab_apifused 1

echo "===== comparisons ====="
Z() { ls -d "$ROOT/run/gl0${G}_$1"/*.zarr 2>/dev/null | head -1; }
for pair in "ab_main ab_api" "ab_api ab_apifused"; do
  set -- $pair
  echo "-- $1 vs $2 (exact):"
  python "$ROOT/scripts/cmp_zarr.py" "$(Z $1)" "$(Z $2)" --exact \
    || python "$ROOT/scripts/cmp_zarr.py" "$(Z $1)" "$(Z $2)" --rtol 1e-10
done
echo "-- NCCLFFI audit (run C):"
python - "$ROOT/run/gl0${G}_ab_apifused/stdout.log" <<'PY'
import re, sys
send, recv = {}, {}
for l in open(sys.argv[1]):
    m = re.search(r'NCCLFFI_CKSUM call=(\d+) plan=(\d+) rank=(\d+) peer=(\d+) send=([0-9a-f]+) recv=([0-9a-f]+)', l)
    if not m: continue
    c, p, r, q = map(int, m.group(1, 2, 3, 4))
    send[(c, r, q)] = m.group(5); recv[(c, q, r)] = m.group(6)
bad = sum(1 for k, cs in send.items() if recv.get(k) != cs)
print(f"pairs={len(send)} MISMATCHES={bad} -> {'AUDIT CLEAN' if bad == 0 and send else 'FAIL'}")
PY
