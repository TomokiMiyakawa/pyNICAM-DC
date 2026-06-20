#!/usr/bin/env bash
# Toggle decomposition at 1 rank/GPU: attribute the residency A/B effect to each
# gate separately. A=baseline, B1=RESIDENT_VISEG only, B2=ONDEVICE_COMM only,
# B=both. All same config/output -> each must be BIT-EXACT vs A. Reports per-step
# (excl step1) + COMM_xfer so we can see which toggle helps and which hurts.
set -uo pipefail
SW=/lambda/nfs/forpynicamH100x4/sweep
source /lambda/nfs/forpynicamH100x4/venv/bin/activate
DRV=$SW/code/pynicamdc/nhm/driver/driver-dc.py
CMPZ=$SW/code/pynicamdc/nhm/dynamics/proto/cmp_zarr.py
export PYTHONPATH=$SW/code
G=${1:-7}; L=${2:-12}; gp=$(printf "%02d" "$G")
COMMON_ENV=(MPI4JAX_USE_CUDA_MPI=1 HCOLL_ENABLE=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9)
MPI=(mpirun --mca pml ucx --mca coll ^hcoll --mca btl self,vader -np 4 $SW/code/bind.sh)

run () { local label=$1; shift
  python $SW/scripts/make_config.py "$G" --backend jax --lstep "$L" --output on --label "$label" >/dev/null
  local rd=$SW/run/gl${gp}_${label}; rm -rf "$rd/testout_tmp.zarr"
  ( cd "$rd" && env "${COMMON_ENV[@]}" "$@" "${MPI[@]}" python "$DRV" --driver-setting ./driversettings.toml ) > "$rd/run.log" 2>&1
  echo "$rd"; }

echo "===== gl$gp toggle decomposition (lstep=$L, 1 rank/GPU) ====="
echo "-- A  baseline --";                 RA=$(run dcA)
echo "-- B1 RESIDENT_VISEG only --";       RB1=$(run dcB1 PYNICAM_RESIDENT_VISEG=1)
echo "-- B2 ONDEVICE_COMM only --";        RB2=$(run dcB2 PYNICAM_ONDEVICE_COMM=1)
echo "-- B  both --";                      RB=$(run dcB  PYNICAM_RESIDENT_VISEG=1 PYNICAM_ONDEVICE_COMM=1)

echo "-- bit-exact vs A (all must be BIT-EXACT) --"
for tag in B1 B2 B; do rd=$(eval echo \$R$tag)
  v=$(python "$CMPZ" "$RA/testout_tmp.zarr" "$rd/testout_tmp.zarr" 2>/dev/null | grep -oE "BIT-EXACT MATCH|MISMATCH" | head -1)
  echo "   A vs $tag: ${v:-?}"; done

echo "-- timing (per-step excl step1; lstep=$L) --"
python - "$RA" "$RB1" "$RB2" "$RB" "$((L-1))" <<'PY'
import re,sys
paths=dict(zip(["A","B1","B2","B"],sys.argv[1:5])); NS=int(sys.argv[5])
def get(d,n):
    t=open(f"{d}/msg.pe00000000").read(); m=re.search(rf"{n}\s+T=\s*([0-9.]+)\s+N=(\d+)",t); return float(m.group(1)) if m else float('nan')
ps={}
for tag,d in paths.items():
    T=get(d,"MAIN_Main_Loop "); T1=get(d,"MAIN_Main_Loop_step1"); C=get(d,"MAIN_COMM_data_transfer")
    ps[tag]=(T-T1)/NS
    print(f"   {tag:3} per-step={ps[tag]:7.3f}s  COMM_xfer={C:6.2f}s")
a=ps['A']
for tag in ("B1","B2","B"):
    d=ps[tag]; print(f"   {tag} vs A: {d/a:.3f}  ({'faster' if d<a else 'SLOWER'} {abs(1-d/a)*100:.1f}%)")
PY
echo "===== done ====="
