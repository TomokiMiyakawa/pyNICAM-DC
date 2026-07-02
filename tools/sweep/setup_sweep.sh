#!/bin/bash
# Assemble a runnable resolution-sweep working dir from this pyNICAM-DC checkout:
# lays out the harness in the scripts/ + config/ + run/ structure make_config.py expects,
# points 'code' at this repo, and fetches the sweep datasets (boundary/restart/golds).
#
#   lite  quick-start data  -> tools/fetch_testdata.sh   (case2/case3, separate)
#   heavy sweep data + harness (this) -> a self-contained sweep root you can run from.
#
# Usage (from the repo root, or anywhere):
#   bash tools/sweep/setup_sweep.sh                    # all glevels gl05..gl09 (~10 GB)
#   bash tools/sweep/setup_sweep.sh 07 08              # subset
#   PYNICAM_SWEEP_ROOT=/scratch/sweep bash tools/sweep/setup_sweep.sh 07
#
# Then run (from the sweep root it prints):
#   scripts/run_sweep.sh                               # numpy reference sweep
#   source <repo>/config/production.env                # GPU fast path (fused+resident stack)
#   BACKEND=jax GLEVELS="7 8" scripts/run_sweep.sh
# and validate against the golds with  <repo>/pynicamdc/nhm/dynamics/proto/cmp_prec.py.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"     # <repo>/tools/sweep
REPO="$(cd "$HERE/../.." && pwd)"          # <repo>
ROOT="${PYNICAM_SWEEP_ROOT:-$(pwd)/pynicam-sweep}"
GLEVELS=("$@")                             # positional glevels; empty -> all (fetch default)

echo "repo:       $REPO"
echo "sweep root: $ROOT"

mkdir -p "$ROOT/scripts" "$ROOT/config" "$ROOT/run"
cp "$HERE"/make_config.py "$HERE"/run_sweep.sh "$HERE"/collect_timers.py "$HERE"/compare_f90.py "$ROOT/scripts/"
cp "$HERE"/nhm_driver.template.toml "$ROOT/config/"
cp "$HERE"/f90_reference.csv "$ROOT/run/" 2>/dev/null || true
ln -sfn "$REPO" "$ROOT/code"               # run_sweep.sh's PYNICAM_CODE default = <root>/code

echo ">>> fetching sweep datasets ..."
PYNICAM_SWEEP_ROOT="$ROOT" bash "$REPO/tools/fetch_sweepdata.sh" "${GLEVELS[@]}"

cat <<EOF

Sweep root ready: $ROOT
  scripts/  config/  data/  run/golds/  code -> $REPO

Run the reference (numpy) sweep:
  cd "$ROOT" && scripts/run_sweep.sh                 # GLEVELS, BACKEND, LSTEP, NPROC overridable

GPU fast path (fused + resident stack; 1 rank/GPU, CUDA-aware MPI):
  source "$REPO/config/production.env"
  cd "$ROOT" && BACKEND=jax GLEVELS="7 8" scripts/run_sweep.sh

Bit-exact validation vs the numpy golds:
  python "$REPO/pynicamdc/nhm/dynamics/proto/cmp_prec.py" \\
         run/golds/gl07_numpy_gold.zarr run/gl07_jax/testout_tmp.zarr --rtol 1e-9
EOF
