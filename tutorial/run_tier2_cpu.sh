#!/bin/bash
# TIER 2 -- scientific validation on CPU (numpy backend), all 15 nicamdc test cases.
# Each is a self-initializing ideal run at its REAL vertical grid + planet factor
# (see cases.txt), checked against the bundled numpy golden (sanity + field match).
# Needs the input dataset -> run ./download_inputs.sh first.
#
# Run one case:   ./run_tier2_cpu.sh gw
# Run all 15:     ./run_tier2_cpu.sh
#
# Each case runs in its OWN sandbox  runs/<case>/  (shared inputs symlinked in as
# ./case), so cases never clobber each other's zarr/msg and everything a case
# produced -- the log (run.log) and dumped state (out_np_rank0.npy) -- is in one place.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$CODE"
cd "$HERE"
[ -f case/grid_gl05rl00pe01/vgrid30_dcmip2016v2.json ] || { echo "ERROR: dataset missing -> ./download_inputs.sh" >&2; exit 2; }
RUN() { if command -v mpirun >/dev/null 2>&1; then mpirun -np 1 "$@"; else "$@"; fi; }
DRIVER="$CODE/pynicamdc/nhm/driver/driver-dc.py"
WANT="${1:-}"
fail=0; npass=0; ntot=0
while IFS='|' read name nicamid z planet desc; do
  [ -z "$name" ] && continue; case "$name" in \#*) continue;; esac
  [ -n "$WANT" ] && [ "$WANT" != "$name" ] && continue
  ntot=$((ntot+1))
  echo; echo "########## TIER 2 [$nicamid] $name (z$z, X$planet) -- $desc ##########"
  rundir="$HERE/runs/${name}"
  rm -rf "$rundir"; mkdir -p "$rundir"; ln -sfn "$HERE/case" "$rundir/case"
  ( cd "$rundir"
    PYNICAM_TIMELOOP_DUMP="./out_np" \
      RUN python "$DRIVER" --driver-setting "$HERE/drv/drv_${name}_np.toml" < /dev/null > run.log 2>&1 )
  if ! grep -q 'peacefully done' "$rundir/run.log"; then
    echo "  RUN FAILED -- see runs/${name}/run.log"; tail -4 "$rundir/run.log"; fail=1; continue
  fi
  if python check_validation.py "$rundir/out_np_rank0.npy" \
       --ref "case/golden/${name}_golden_rank0.npy" --rtol 1e-6 --label "$name vs golden"; then
    npass=$((npass+1))
  else fail=1; fi
done < cases.txt
echo; echo "=== TIER 2: $npass/$ntot passed $([ $fail -eq 0 ] && echo '-- ALL PASS' || echo '-- FAILURES') ==="
exit $fail
