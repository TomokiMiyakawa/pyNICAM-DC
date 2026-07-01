#!/usr/bin/env bash
# gl05-09 jax correctness sweep on the H100x4 box (1 rank/GPU), vs CPU golds.
# lstep per README table (the golds are n=1 snapshots at the final step).
set -uo pipefail
SW=/lambda/nfs/forpynicamH100x4/sweep
source /lambda/nfs/forpynicamH100x4/venv/bin/activate
DRV=$SW/code/pynicamdc/nhm/driver/driver-dc.py
CMP=$SW/code/pynicamdc/nhm/dynamics/proto/cmp_prec.py
export PYTHONPATH=$SW/code
export MPI4JAX_USE_CUDA_MPI=1 HCOLL_ENABLE=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

declare -A LSTEP=( [5]=12 [6]=12 [7]=12 [8]=6 [9]=4 )
GLEVELS="${GLEVELS:-6 7 8 9}"
RESULT=$SW/run/correctness_results.txt
: > "$RESULT"

for g in $GLEVELS; do
  gp=$(printf "%02d" "$g"); L=${LSTEP[$g]}
  echo "================ gl$gp (lstep=$L) ================"
  python $SW/scripts/make_config.py "$g" --backend jax --lstep "$L" --output on --label jax >/dev/null
  rundir=$SW/run/gl${gp}_jax
  rm -rf "$rundir/testout_tmp.zarr"
  ( cd "$rundir" && timeout 3600 mpirun --mca pml ucx --mca coll ^hcoll --mca btl self,vader \
      -np 4 $SW/code/bind.sh python "$DRV" --driver-setting ./driversettings.toml ) \
      > "$rundir/run.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -d "$rundir/testout_tmp.zarr" ]; then
    echo "RUN_FAIL gl$gp rc=$rc"; echo "gl$gp  RUN_FAIL rc=$rc" >> "$RESULT"
    tail -5 "$rundir/run.log"; continue
  fi
  echo "--- cmp_prec gl$gp vs gold ---"
  python "$CMP" "$SW/golds/gl0${g}_numpy_gold.zarr" "$rundir/testout_tmp.zarr" --rtol 1e-10 \
      | tee "$rundir/cmp.log" | tail -3
  verdict=$(grep -oE "^(PASS|FAIL)" "$rundir/cmp.log" | head -1)
  worst=$(grep -oE "worst_rel: [0-9.e+-]+" "$rundir/cmp.log" | head -1)
  echo "gl$gp  ${verdict:-NOVERDICT}  $worst  (lstep=$L)" >> "$RESULT"
  echo "[gl$gp DONE: ${verdict:-?} $worst]"
done
echo "===== SWEEP COMPLETE ====="
cat "$RESULT"
