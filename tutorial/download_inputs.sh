#!/bin/bash
# Download the pyNICAM-DC tutorial input dataset (horizontal grid + vertical grids
# + 14 numpy goldens, ~100 MB) into tutorial/case/.  Hosted OUTSIDE the git repo
# because it is binary grid/reference data; the small text configs ship with the repo.
#
# Override the link if needed:  export PYNICAM_TUTORIAL_INPUTS_URL=<url>  ./download_inputs.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DEFAULT_URL="https://filedn.com/l1RCVmoJKNfj5AyQ77icpqJ/pynicam-tutorial-data_v20260718b/pynicam-tutorial-inputs.tar.gz"
EXPECT_SHA256="a69dd9adb677abd9ba6259dc4f00f81fdcfad42473a566c4eef799a2ae411420"
URL="${PYNICAM_TUTORIAL_INPUTS_URL:-$DEFAULT_URL}"

echo "Downloading tutorial inputs from: $URL"
TARBALL="pynicam-tutorial-inputs.tar.gz"
if command -v curl >/dev/null 2>&1; then curl -fL "$URL" -o "$TARBALL"
elif command -v wget >/dev/null 2>&1; then wget -O "$TARBALL" "$URL"
else echo "ERROR: need curl or wget" >&2; exit 2; fi

# integrity check (skip only if sha256sum is unavailable)
if command -v sha256sum >/dev/null 2>&1; then
  got=$(sha256sum "$TARBALL" | cut -d' ' -f1)
  if [ "$got" != "$EXPECT_SHA256" ]; then
    echo "ERROR: checksum mismatch (got $got, expected $EXPECT_SHA256) -- corrupt/stale download." >&2
    rm -f "$TARBALL"; exit 3
  fi
  echo "checksum OK"
fi

echo "Extracting into $HERE/case/ ..."
tar xzf "$TARBALL"          # unpacks case/grid_gl05rl00pe01/ and case/golden/
rm -f "$TARBALL"
mkdir -p case/output

echo "Done. Contents:"
ls -R case/grid_gl05rl00pe01 case/golden 2>/dev/null | sed 's/^/  /'
echo
echo "You can now run ./run_tier1_pytest.sh, ./run_tier2_cpu.sh, and (on Miyabi) qsub run_tier3_gpu.pbs"
