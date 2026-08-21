#!/bin/bash -l
#SBATCH --job-name=pnc_g5_fused
#SBATCH --partition=booster
#SBATCH --account=jureap22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72     # one Grace socket per rank (4.4x trap on multi-node)
#SBATCH --gpus-per-node=4      # NEVER --gpus-per-task=1 (Cuda failure 101) -- see PORT.md
#SBATCH --time=00:40:00
#SBATCH --output=/e/project1/jureap22/miyakawa/workClaude1/sweep/logs/pnc_g5_fused.%j.out
# T2-PROD: gl05 rl01 pe04 -- FULL PRODUCTION FUSION: device-resident state + kernel
# fusion + NCCL-FFI halo transport (PYNICAM_COMM_NCCLFFI=1, set by production.env).
#
# TWO ARMS IN ONE JOB ("perf verdicts ONLY from same-job A/B arms"):
#   ARM A  order audit -- CKSUM=1 with JIT=0 CHUNK=1 WARMUP=0 so every exchange is
#          individually auditable. tools/ncclffi/ADOPTION.md invariant #2 REQUIRES
#          this after any jax/jaxlib/NCCL change and once per new (glevel,pe)
#          decomposition. NCCL is tag-less, so uniform cross-rank call order is a
#          CORRECTNESS precondition, not a perf detail.
#          PASS = 0 checksum mismatches AND all ranks done.
#   ARM B  production perf -- JIT=1 CHUNK=4 WARMUP=3, cksum off (its per-call
#          printing and CHUNK=1 would distort timing).
set -uo pipefail

CODE=/e/project1/jureap22/miyakawa/workClaude1/pyNICAM-DC
ROOT=/e/project1/jureap22/miyakawa/workClaude1/sweep
source "$CODE/tools/jupiter/env.sh"
DRV="$CODE/pynicamdc/nhm/driver/driver-dc.py"
G=5; PE=4
export OMP_NUM_THREADS=1

# The FFI lib carries an RPATH to the venv's NCCL. After ANY move of the venv or the
# repo it must be rebuilt, or the loader silently falls back to the system
# /lib64/libnccl.so.2 -- which must never win.
[ -f "$PYNICAM_NCCLFFI_LIB" ] || { echo "FATAL: no libncclffi.so -- run tools/jupiter/build_ncclffi_jupiter.sh"; exit 2; }
echo "### ncclffi lib: $PYNICAM_NCCLFFI_LIB"
ldd "$PYNICAM_NCCLFFI_LIB" | grep -E 'nccl|not found'

WRAP="$ROOT/gpu_wrap.sh"
cat > "$WRAP" <<'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
EOF
chmod +x "$WRAP"

run_arm() {   # $1=label  $2=lstep ; gate env comes from the caller's subshell
  local LAB="$1" LSTEP="$2"
  python "$ROOT/scripts/make_config.py" "$G" --backend jax --precision float64 \
      --lstep "$LSTEP" --pe "$PE" --label "$LAB" --output on >/dev/null 2>&1
  local RD="$ROOT/run/gl0${G}_${LAB}"
  # PRGout_interval=1000 = the canonical "no snapshot writes": past lstep_max, so no
  # output step, so the fused chunk is never trimmed (driver-dc.py:471-476). --output on
  # would set interval=lstep and trim the chunk that spans the final step. Kept as a sed
  # because this ROOT may carry an older/site-local make_config; the repo's --output off
  # now sets interval=lstep+1, i.e. exactly this. Either way the zarr is left at its NaN
  # fill_value, which reads exactly like a diverged solution -- never judge numerics from it.
  sed -i 's/^PRGout_interval=.*/PRGout_interval=1000/' "$RD/nhm_driver.toml"
  cd "$RD" || exit 1
  rm -rf out* testout_tmp.zarr *.zarr msg.pe*
  srun --ntasks="$PE" --cpus-per-task=72 --cpu-bind=socket "$WRAP" \
       python "$DRV" --driver-setting ./driversettings.toml < /dev/null \
       > "$ROOT/logs/${LAB}.log" 2>&1
  echo "  ranks done=$(grep -c 'peacefully done' "$ROOT/logs/${LAB}.log")/$PE"
  echo "  NCCLFFI engaged=$(grep -c 'NCCLFFI: comm up' "$ROOT/logs/${LAB}.log")"
}

echo; echo "================ ARM A: NCCL-FFI order audit ================"
( source "$CODE/config/production.env"
  export PYNICAM_COMM_NCCLFFI=1 PYNICAM_NCCLFFI_CKSUM=1 PYNICAM_COMM_NO_BARRIER=1
  export PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=0 PYNICAM_TIMELOOP_CHUNK=1 PYNICAM_TIMELOOP_WARMUP=0
  export PYNICAM_TIMELOOP_DUMP="./out" PYNICAM_PROFILE=timeloop_timing
  run_arm audit_g5pe4 4 )

python3 - "$ROOT/logs/audit_g5pe4.log" "$PE" <<'PY'
import sys, re
log, pe = sys.argv[1], int(sys.argv[2])
done = eng = 0
send = {}; recv = {}
for l in open(log, errors="replace"):
    if 'peacefully done' in l: done += 1
    if 'NCCLFFI: comm up' in l: eng += 1
    m = re.search(r'NCCLFFI_CKSUM call=(\d+) plan=(\d+) rank=(\d+) peer=(\d+) send=([0-9a-f]+) recv=([0-9a-f]+)', l)
    if not m: continue
    c, p, r, q = map(int, m.group(1, 2, 3, 4))
    send[(c, r, q)] = (m.group(5), p); recv[(c, q, r)] = m.group(6)
bad = [(k, cs, p, recv.get(k)) for k, (cs, p) in sorted(send.items()) if recv.get(k) != cs]
print(f"  ranks done={done}/{pe} engaged={eng} pairs={len(send)} MISMATCHES={len(bad)}")
for k, cs, p, cr in bad[:8]:
    print(f"    call={k[0]} plan={p} {k[1]}->{k[2]}: sent={cs} recvd={cr}")
ok = done == pe and eng >= 1 and len(send) > 0 and not bad
print("  ==> AUDIT PASS" if ok else "  ==> AUDIT FAIL (see ncclffi plan v2 S1 prescription)")
PY

echo; echo "================ ARM B: production fused perf ================"
( source "$CODE/config/production.env"
  export PYNICAM_COMM_NCCLFFI=1 PYNICAM_COMM_NO_BARRIER=1
  # CHUNK=4/WARMUP=3 kept DELIBERATELY (S3): the historical-comparability
  # protocol (lstep 43 = 3 + 40; every recorded s/step uses it). Since S2 the
  # resolver default is cap=1 -> K=1, warm-up=K; this opts into the legacy values.
  export PYNICAM_FUSE_TIMELOOP=1 PYNICAM_TIMELOOP_JIT=1 PYNICAM_TIMELOOP_CHUNK=4 PYNICAM_TIMELOOP_WARMUP=3
  export PYNICAM_TIMELOOP_DUMP="./out" PYNICAM_PROFILE=timeloop_timing
  run_arm fused_g5pe4 12 )

# THE perf metric is the STEADY TIMELOOP_CHUNK per_step, NOT MAIN_Main_Loop: with
# FUSE+JIT the Main_Loop timer counts CHUNKS (N = lstep/CHUNK) and at short lstep the
# wall time is dominated by JIT compiles. Raise lstep to ~43 for a real number.
echo "### steady TIMELOOP_CHUNK (ignore the first compiles) ###"
grep "TIMELOOP_CHUNK" "$ROOT/logs/fused_g5pe4.log" 2>/dev/null | tail -12
echo "### MAIN timers (NOT the perf metric -- see above) ###"
grep -E "MAIN_Main_Loop|MAIN___Dynamics" "$ROOT/run/gl05_fused_g5pe4/msg.pe00000000" 2>/dev/null | head
# Reference (2026-07-28, job 1086073): ARM A 2448 pairs / 0 mismatches PASS;
# ARM B steady per_step 0.0122 s (vs 0.391 plain jax). MAIN_Main_Loop read 51.7 s
# there -- that is JIT compile (14.4 s K=4 + 13.3 s K=1), not the step cost.
