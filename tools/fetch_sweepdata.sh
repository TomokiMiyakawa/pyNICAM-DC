#!/bin/bash
# Fetch the FULL resolution-sweep datasets for pyNICAM-DC (developer / "heavy" tier).
#
# This is the large dataset used for the resolution-sweep performance + validation runs
# (gl05..gl09: horizontal boundary + restart + numpy reference golds). It is SEPARATE from
# the lite quick-start data (case2/case3), which is fetched by tools/fetch_testdata.sh.
#
#   lite  (new users)   -> tools/fetch_testdata.sh   (~49 MB: case2 JW + case3 DCMIP)
#   heavy (developers)  -> THIS script               (~10 GB: gl05..gl09 sweep + golds)
#
# Downloads per-glevel tarballs (so you can grab a subset), verifies sha256, and extracts
# into a sweep root laid out as the run harness expects:
#     <root>/data/{boundary,restart,mnginfo}/...   <root>/data/vgrid40_stretch_45km.json
#     <root>/run/golds/gl0N_numpy_gold.zarr
#
# Usage:
#   bash tools/fetch_sweepdata.sh                 # all glevels (gl05..gl09), ~10 GB
#   bash tools/fetch_sweepdata.sh 07 08           # only gl07 + gl08 (+ shared)
#   PYNICAM_SWEEP_ROOT=/path/to/sweep bash tools/fetch_sweepdata.sh 07
#   PYNICAM_SWEEPDATA_URL=<host> bash tools/fetch_sweepdata.sh   # override host
#
# Approx per-glevel sizes (compressed): gl05 ~35M, gl06 ~120M, gl07 ~460M, gl08 ~1.8G, gl09 ~7G.
set -euo pipefail

BASE="${PYNICAM_SWEEPDATA_URL:-https://filedn.com/l1RCVmoJKNfj5AyQ77icpqJ/sweep_dist}"
ROOT="${PYNICAM_SWEEP_ROOT:-$(pwd)/pynicam-sweep}"
GLEVELS=("$@"); [ "${#GLEVELS[@]}" -eq 0 ] && GLEVELS=(05 06 07 08 09)

# filename -> sha256  (filled by tools/pack_sweepdata + verified on download)
declare -A SHA=(
  [sweep_shared.tar.gz]="b3a7c32174e6d6c419999368dfe791c3565822ffb75dff92fc551503ffdeb87b"
  [sweep_gl05.tar.gz]="c10100a33171682b2af57892c6e420e0946edd55ac76484f3f21e28cabe09a41"
  [sweep_gl06.tar.gz]="fa70ed6371fe7b6586febfe82a1acaab90d78e4c41a34777e624b39ca0693edf"
  [sweep_gl07.tar.gz]="e56726a29a05ab95a811296b7dfd4346c1876f51e940c619280fb0fa7e746c3d"
  [sweep_gl08.tar.gz]="806c3f2c8daf41afa9777d0af83f5959ae64f7bedca04b18359556db71b50b80"
  [sweep_gl09.tar.gz]="ab0b8d5a64b7d4eeb1d7513e9c94d859f553adcb14f3632fafe8e7ee0010a784"
)

mkdir -p "$ROOT"; cd "$ROOT"
echo "sweep root: $ROOT"

fetch () { # $1 = tarball
  local f="$1"
  echo ">>> $f"
  wget -c -O "$f" "$BASE/$f"
  if [ -n "${SHA[$f]:-}" ] && [[ "${SHA[$f]}" != @@* ]]; then
    echo "${SHA[$f]}  $f" | sha256sum -c - || { echo "!! checksum failed for $f"; exit 1; }
  else
    echo "   (no checksum recorded for $f -- skipping verify)"
  fi
  tar xzf "$f"          # pigz output is standard gzip
  rm -f "$f"
}

fetch sweep_shared.tar.gz
for g in "${GLEVELS[@]}"; do fetch "sweep_gl${g}.tar.gz"; done

echo "Done. Sweep data under $ROOT/data and $ROOT/run/golds."
echo "Point the run harness at it, e.g.:  ROOT=$ROOT  (make_config.py / env_check/*.sh)"
