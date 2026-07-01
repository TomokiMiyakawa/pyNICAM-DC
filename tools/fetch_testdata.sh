#!/bin/bash
# Fetch the large restart datasets for pyNICAM-DC test cases (not stored in git).
#
#   case2  = Jablonowski-Williamson baroclinic wave (RK3, gl05 z40, 8 ranks)
#   case3  = DCMIP 1-1 tracer advection            (TRCADV, gl05 z60, 8 ranks)
#
# The configs, prepdata, and horizontal grid ship with the repo; this script
# downloads only the restart data and extracts it into pynicamdc/test/{case2,case3}/.
# Idempotent + checksum-verified. Run from anywhere:  bash tools/fetch_testdata.sh
set -euo pipefail

BASE="${PYNICAM_TESTDATA_URL:-https://filedn.com/l1RCVmoJKNfj5AyQ77icpqJ/testdata_dist}"
TESTDIR="$(cd "$(dirname "$0")/../pynicamdc/test" && pwd)"

# filename  ->  sha256
FILES=(case2_restart.tar.gz case3_restart.tar.gz)
declare -A SHA=(
  [case2_restart.tar.gz]="faea074c8faaefbd52ec8f7df55a1c8d52768d2fa959c8f95234d07224e7bb97"
  [case3_restart.tar.gz]="bf2282f12e9f8fab390e9474a9786a40db5ddfb4381b07a26e59afda69a35b9b"
)

cd "$TESTDIR"
for f in "${FILES[@]}"; do
  echo ">>> $f"
  wget -c -O "$f" "$BASE/$f"
  echo "${SHA[$f]}  $f" | sha256sum -c - || { echo "!! checksum failed for $f"; exit 1; }
  tar xzf "$f"
  rm -f "$f"
done
echo "Done. Restart data extracted under $TESTDIR/{case2,case3}/."
