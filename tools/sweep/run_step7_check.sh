#!/usr/bin/env bash
set -uo pipefail
SW=/lambda/nfs/forpynicamH100x4/sweep
source /lambda/nfs/forpynicamH100x4/venv/bin/activate
DRV=$SW/code/pynicamdc/nhm/driver/driver-dc.py
CMPZ=$SW/code/pynicamdc/nhm/dynamics/proto/cmp_zarr.py
export PYTHONPATH=$SW/code MPI4JAX_USE_CUDA_MPI=1 HCOLL_ENABLE=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
MPI="mpirun --mca pml ucx --mca coll ^hcoll --mca btl self,vader -np 4 $SW/code/bind.sh"
G=7; L=12
run(){ local lbl=$1; shift; python $SW/scripts/make_config.py $G --backend jax --lstep $L --output on --label "$lbl">/dev/null; local rd=$SW/run/gl0${G}_${lbl}; rm -rf "$rd/testout_tmp.zarr"; ( cd "$rd" && env "$@" $MPI python "$DRV" --driver-setting ./driversettings.toml )>"$rd/run.log" 2>&1; }
echo "== s7A FULL=0 =="; run s7A PYNICAM_FUSE_DIVDAMP_FULL=0
echo "== s7B FULL=1 =="; run s7B PYNICAM_FUSE_DIVDAMP_FULL=1
echo "== bit-exact s7A vs s7B =="; python "$CMPZ" $SW/run/gl0${G}_s7A/testout_tmp.zarr $SW/run/gl0${G}_s7B/testout_tmp.zarr | grep -E "RESULT|MISMATCH"
echo "== timing (per-step excl1, divdamp timers) =="
python - $SW/run/gl0${G}_s7A $SW/run/gl0${G}_s7B $((L-1)) <<'PY'
import re,sys
A,B,NS=sys.argv[1],sys.argv[2],int(sys.argv[3])
def g(d,n):
    t=open(f"{d}/msg.pe00000000").read();m=re.search(rf"{n}\s+T=\s*([0-9.]+)\s+N=(\d+)",t);return float(m.group(1)) if m else float('nan')
for tag,d in (("FULL=0",A),("FULL=1",B)):
    T=g(d,"MAIN_Main_Loop ");T1=g(d,"MAIN_Main_Loop_step1");dd=g(d,"____numfilter_divdamp");C=g(d,"MAIN_COMM_data_transfer")
    print(f"  {tag}: per-step={ (T-T1)/NS:.3f}s  divdamp={dd:.2f}s  COMM_xfer={C:.2f}s")
PY
echo "== done =="
