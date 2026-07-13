#!/bin/bash
# TIER 2 -- scientific validation on CPU (numpy backend), all 15 nicamdc test cases.
# Each is a self-initializing ideal run at its REAL vertical grid + planet factor
# (see cases.txt), checked against the bundled numpy golden (sanity + field match).
# Needs the input dataset -> run ./download_inputs.sh first.
#
# Run one case:   ./run_tier2_cpu.sh gw
# Run all 15:     ./run_tier2_cpu.sh
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
  rm -rf ./*.zarr ./msg.pe*
  PYNICAM_TIMELOOP_DUMP="./out_${name}_np" \
    RUN python "$DRIVER" --driver-setting "drv/drv_${name}_np.toml" < /dev/null > "run_${name}_np.log" 2>&1
  if ! grep -q 'peacefully done' "run_${name}_np.log"; then
    echo "  RUN FAILED -- see run_${name}_np.log"; tail -4 "run_${name}_np.log"; fail=1; continue
  fi
  if python check_validation.py "./out_${name}_np_rank0.npy" \
       --ref "case/golden/${name}_golden_rank0.npy" --rtol 1e-6 --label "$name vs golden"; then
    npass=$((npass+1))
  else fail=1; fi
done < cases.txt
echo; echo "=== TIER 2: $npass/$ntot passed $([ $fail -eq 0 ] && echo '-- ALL PASS' || echo '-- FAILURES') ==="
exit $fail
