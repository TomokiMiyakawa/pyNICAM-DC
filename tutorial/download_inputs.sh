#!/bin/bash
# Download the pyNICAM-DC tutorial input dataset (grid + numpy goldens, ~25 MB)
# into tutorial/case/.  The dataset is hosted OUTSIDE the git repo because it is
# binary grid/reference data; the small text configs ship with the repo.
#
# Set the link once (ask the maintainer for the current URL):
#     export PYNICAM_TUTORIAL_INPUTS_URL="https://.../pynicam-tutorial-inputs.tar.gz"
#     ./download_inputs.sh
# or edit the DEFAULT_URL below.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DEFAULT_URL="<PASTE-DOWNLOAD-LINK-HERE>"
URL="${PYNICAM_TUTORIAL_INPUTS_URL:-$DEFAULT_URL}"

if [ "$URL" = "<PASTE-DOWNLOAD-LINK-HERE>" ]; then
  echo "ERROR: no download URL set." >&2
  echo "  export PYNICAM_TUTORIAL_INPUTS_URL=<link>   (or edit DEFAULT_URL in this script)" >&2
  exit 2
fi

echo "Downloading tutorial inputs from: $URL"
TARBALL="pynicam-tutorial-inputs.tar.gz"
if command -v curl >/dev/null 2>&1; then curl -fL "$URL" -o "$TARBALL"
elif command -v wget >/dev/null 2>&1; then wget -O "$TARBALL" "$URL"
else echo "ERROR: need curl or wget" >&2; exit 2; fi

echo "Extracting into $HERE/case/ ..."
tar xzf "$TARBALL"          # unpacks case/grid_gl05rl00pe01/ and case/golden/
rm -f "$TARBALL"
mkdir -p case/output

echo "Done. Contents:"
ls -R case/grid_gl05rl00pe01 case/golden 2>/dev/null | sed 's/^/  /'
echo
echo "You can now run ./run_tier1_pytest.sh, ./run_tier2_cpu.sh, and (on Miyabi) qsub run_tier3_gpu.pbs"
