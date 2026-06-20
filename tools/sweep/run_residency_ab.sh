#!/usr/bin/env bash
# Residency A/B at 1 rank/GPU (H100x4). A = baseline (numpy COMM, non-resident);
# B = PYNICAM_RESIDENT_VISEG=1 PYNICAM_ONDEVICE_COMM=1. Same config/output for both
# so cmp_zarr must be BIT-EXACT (pure data movement) and the MAIN_Main_Loop ratio is fair.
set -uo pipefail
SW=/lambda/nfs/forpynicamH100x4/sweep
source /lambda/nfs/forpynicamH100x4/venv/bin/activate
DRV=$SW/code/pynicamdc/nhm/driver/driver-dc.py
CMPZ=$SW/code/pynicamdc/nhm/dynamics/proto/cmp_zarr.py
export PYTHONPATH=$SW/code
G=${1:-5}; L=${2:-12}; gp=$(printf "%02d" "$G")
COMMON_ENV=(MPI4JAX_USE_CUDA_MPI=1 HCOLL_ENABLE=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9)
MPI=(mpirun --mca pml ucx --mca coll ^hcoll --mca btl self,vader -np 4 $SW/code/bind.sh)

run () {  # $1=label  $2..=extra env
  local label=$1; shift
  python $SW/scripts/make_config.py "$G" --backend jax --lstep "$L" --output on --label "$label" >/dev/null
  local rd=$SW/run/gl${gp}_${label}; rm -rf "$rd/testout_tmp.zarr"
  ( cd "$rd" && env "${COMMON_ENV[@]}" "$@" "${MPI[@]}" python "$DRV" --driver-setting ./driversettings.toml ) \
      > "$rd/run.log" 2>&1
  echo "$rd"
}

echo "===== gl$gp residency A/B (lstep=$L, 1 rank/GPU) ====="
echo "-- A: baseline (numpy COMM, non-resident) --"
RA=$(run abA)
echo "-- B: PYNICAM_RESIDENT_VISEG=1 PYNICAM_ONDEVICE_COMM=1 --"
RB=$(run abB PYNICAM_RESIDENT_VISEG=1 PYNICAM_ONDEVICE_COMM=1)

echo "-- bit-exact check (cmp_zarr A vs B; must be BIT-EXACT) --"
python "$CMPZ" "$RA/testout_tmp.zarr" "$RB/testout_tmp.zarr" | tail -4

echo "-- timing (MAIN_Main_Loop from msg.pe00000000, per-step excl step1) --"
NS=$((L-1))
python - "$RA" "$RB" "$NS" <<'PY'
import re,sys
RA,RB,NS=sys.argv[1],sys.argv[2],int(sys.argv[3])
def get(d,name):
    txt=open(f"{d}/msg.pe00000000").read()
    m=re.search(rf"{name}\s+T=\s*([0-9.]+)\s+N=(\d+)",txt); return float(m.group(1)) if m else float('nan')
rows={}
for tag,d in (("A",RA),("B",RB)):
    T=get(d,"MAIN_Main_Loop "); T1=get(d,"MAIN_Main_Loop_step1")
    C=get(d,"MAIN_COMM_data_transfer"); ps=(T-T1)/NS; rows[tag]=(T,T1,ps,C)
    lbl="A (baseline)" if tag=="A" else "B (resident+ondevCOMM)"
    print(f"  {lbl:24} Main_Loop={T:8.2f}s step1={T1:7.2f}s per-step(excl1)={ps:6.3f}s COMM_xfer={C:6.2f}s")
psA,psB=rows['A'][2],rows['B'][2]
print(f"  => per-step ratio B/A = {psB/psA:.3f}  ({'B FASTER' if psB<psA else 'B slower'} by {abs(1-psB/psA)*100:.1f}%)")
PY
echo "===== done ====="
