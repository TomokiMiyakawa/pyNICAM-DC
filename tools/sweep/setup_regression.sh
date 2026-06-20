#!/usr/bin/env bash
# aarch64 same-box bit-exact regression gate for setup-vectorize changes.
#
# WHY same-box: the CPU golds were made on AORI (x86); aarch64 numpy differs at
# ~1e-12 (arch float rounding), so cross-arch can only be machine-precision, NOT
# bit-exact. Setup vectorizations are pure reindex/reshape -> must be bit-exact.
# So we snapshot an aarch64 numpy baseline NOW (pre-change) and, after each
# incoming setup merge, re-run and cmp_zarr bit-exact (max|d| == 0) vs baseline.
#
# Usage:
#   setup_regression.sh baseline   # run gl05/06/07 numpy, save as baselines
#   setup_regression.sh check      # run again, cmp_zarr (bit-exact) vs baselines
set -uo pipefail
export PATH=/usr/mpi/gcc/openmpi-4.1.7rc1/bin:$PATH
SW=${SW:-/lambda/nfs/forpynicamGH200/sweep}   # harness root (override via env)
export PYTHONPATH=$SW/code
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
# hcoll init fails on this box and INTERMITTENTLY deadlocks plain mpirun at
# MPI init -> always disable it (matches the correctness-run recipe).
export HCOLL_ENABLE=0
MPIRUN="mpirun --mca coll ^hcoll"
PY=/lambda/nfs/forpynicamGH200/venv/bin/python
DRV=$SW/code/pynicamdc/nhm/driver/driver-dc.py
CMPZ=$SW/code/pynicamdc/nhm/dynamics/proto/cmp_zarr.py
BDIR=$SW/run/baselines_aarch64_numpy
mode=${1:-check}
mkdir -p "$BDIR"
declare -A LSTEP=( [5]=12 [6]=12 [7]=12 )   # matches the gold / sweep table

for g in 5 6 7; do
  gp=$(printf "%02d" $g); ls=${LSTEP[$g]}
  rd=$SW/run/gl${gp}_numpy
  echo "================ gl$gp (lstep=$ls) [$mode] ================"
  $PY $SW/scripts/make_config.py $g --backend numpy --lstep $ls --output on \
      > /dev/null 2>&1 || { echo "  make_config FAIL"; continue; }
  mkdir -p "$rd"; cp -f $rd/../gl${gp}_numpy/nhm_driver.toml "$rd/" 2>/dev/null
  t0=$(date +%s.%N)
  ( cd "$rd" && rm -rf testout_tmp.zarr msg.pe* tempout* && \
    timeout 1800 $MPIRUN -np 4 $PY $DRV ) > "$rd/run.log" 2>&1
  rc=$?; t1=$(date +%s.%N)
  wall=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
  if [ $rc -ne 0 ]; then echo "  RUN FAIL (rc=$rc); tail:"; tail -5 "$rd/run.log"; continue; fi
  echo "  run OK  wall=${wall}s"
  if [ "$mode" = "baseline" ]; then
    rm -rf "$BDIR/gl${gp}.zarr"; cp -r "$rd/testout_tmp.zarr" "$BDIR/gl${gp}.zarr"
    echo "  saved baseline -> $BDIR/gl${gp}.zarr"
  else
    if [ -d "$BDIR/gl${gp}.zarr" ]; then
      echo "  cmp_zarr (bit-exact) vs baseline:"
      $PY $CMPZ "$BDIR/gl${gp}.zarr" "$rd/testout_tmp.zarr" 2>&1 | tail -12
    else
      echo "  NO baseline for gl$gp -- run 'baseline' first"
    fi
  fi
done
echo "ALL DONE [$mode]"
