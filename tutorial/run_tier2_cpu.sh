#!/bin/bash
# TIER 2 -- scientific validation on CPU (numpy backend).
# Runs two self-initializing (no restart) ideal cases at gl05, 5 steps each:
#   jbw = Jablonowski dry baroclinic wave  (pure dynamical core)
#   jm  = Jablonowski-Moist                (adds moisture + DCMIP forcing/physics)
# and checks each against the bundled numpy golden (physical sanity + field match).
# Needs the input dataset -> run ./download_inputs.sh first.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$CODE"
cd "$HERE"

if [ ! -f case/grid_gl05rl00pe01/vgrid30_dcmip2016v2.json ]; then
  echo "ERROR: input dataset missing. Run ./download_inputs.sh first." >&2; exit 2
fi
RUN() { if command -v mpirun >/dev/null 2>&1; then mpirun -np 1 "$@"; else "$@"; fi; }
DRIVER="$CODE/pynicamdc/nhm/driver/driver-dc.py"
fail=0
for c in jbw jm; do
  echo; echo "########## TIER 2: $c (numpy, 5 steps) ##########"
  rm -rf ./*.zarr ./msg.pe*
  PYNICAM_TIMELOOP_DUMP="./out_${c}_np" \
    RUN python "$DRIVER" --driver-setting "drv/drv_${c}_np.toml" > "run_${c}_np.log" 2>&1
  if ! grep -q 'peacefully done' "run_${c}_np.log"; then
    echo "  RUN FAILED -- see run_${c}_np.log"; tail -5 "run_${c}_np.log"; fail=1; continue
  fi
  python check_validation.py "./out_${c}_np_rank0.npy" \
      --ref "case/golden/${c}_golden_rank0.npy" --rtol 1e-6 --label "$c (numpy vs golden)" || fail=1
done
echo; echo "=== TIER 2 $([ $fail -eq 0 ] && echo ALL PASS || echo FAILURES) ==="
exit $fail
