#!/usr/bin/env bash
# Run the f90-vs-pyNICAM resolution sweep (gl05..gl09, rl01, pe04) on this machine.
#
# Configure these for your machine (or export them before calling):
#   PYNICAM_PY      : python interpreter (in the env with numpy/jax/mpi4py/toml/xarray/zarr)
#   PYNICAM_MPIRUN  : mpirun/mpiexec launcher
#   PYNICAM_CODE    : path to the pyNICAM-DC source root (the dir that CONTAINS 'pynicamdc/')
#                     -> defaults to the bundled ./code
#
# Usage:
#   scripts/run_sweep.sh                       # gl05..09, numpy, 12 steps
#   GLEVELS="5 6 7" BACKEND=jax scripts/run_sweep.sh
#   LSTEP=6 scripts/run_sweep.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PY="${PYNICAM_PY:-python}"
MPIRUN="${PYNICAM_MPIRUN:-mpirun}"
CODE="${PYNICAM_CODE:-$ROOT/code}"
DRV="$CODE/pynicamdc/nhm/driver/driver-dc.py"

GLEVELS="${GLEVELS:-5 6 7 8 9}"
BACKEND="${BACKEND:-numpy}"
LSTEP="${LSTEP:-12}"
NPROC="${NPROC:-4}"            # pe04
RUNLABEL="${RUNLABEL:-$BACKEND}"   # run-dir/CSV suffix; set e.g. jax_be for the
                                   # best-effort hybrid (PYNICAM_BESTEFFORT=1)

[ -f "$DRV" ] || { echo "ERROR: driver not found: $DRV (set PYNICAM_CODE)"; exit 1; }

echo "ROOT=$ROOT"
echo "PY=$PY  MPIRUN=$MPIRUN  CODE=$CODE"
echo "GLEVELS=[$GLEVELS]  BACKEND=$BACKEND  LSTEP=$LSTEP  NPROC=$NPROC"
echo

export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

for g in $GLEVELS; do
    gp=$(printf "%02d" "$g")
    echo "================ gl$gp ================"
    "$PY" "$ROOT/scripts/make_config.py" "$g" --backend "$BACKEND" --lstep "$LSTEP" --label "$RUNLABEL"
    rundir="$ROOT/run/gl${gp}_${RUNLABEL}"
    rm -rf "$rundir/testout_tmp.zarr"
    ( cd "$rundir" && \
      "$MPIRUN" -np "$NPROC" "$PY" "$DRV" --driver-setting ./driversettings.toml ) \
      2>&1 | tee "$rundir/run.log"
    echo "  log + msg.pe* in $rundir"
    echo
done

echo "Collecting timers ..."
"$PY" "$ROOT/scripts/collect_timers.py" --backend "$BACKEND" --label "$RUNLABEL" $GLEVELS | tee "$ROOT/run/timers_${RUNLABEL}.csv"
echo "Done. Summary: $ROOT/run/timers_${RUNLABEL}.csv"
