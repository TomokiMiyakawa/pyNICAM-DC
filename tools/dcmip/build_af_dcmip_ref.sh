#!/usr/bin/env bash
# Build + run the AF_dcmip glue reference driver -> ref_af_dcmip_z*.txt
# (consumed by test_af_dcmip.py). Compiles the UNMODIFIED simple_physics_v6.f90
# with the glue driver (verbatim AF_dcmip transform).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP_SRC="${SP_SRC:-/work/gj37/c24028/workforclaude/nicamdc/src/nhm/share/dcmip/simple_physics_v6.f90}"
SP_FC="${SP_FC:-gfortran}"   # env FC=nvfortran collides; nvfortran rejects the flag
FFLAGS="${FFLAGS:--O2 -ffree-line-length-none}"

cd "$HERE"
echo "compiler: $($SP_FC --version | head -1)"
$SP_FC $FFLAGS -c "$SP_SRC" -o simple_physics_v6.o
$SP_FC $FFLAGS -c af_dcmip_ref.f90 -o af_dcmip_ref.o
$SP_FC $FFLAGS simple_physics_v6.o af_dcmip_ref.o -o af_dcmip_ref.x

PCOLS="${PCOLS:-5}"
PVERS="${PVERS:-30 40 78}"
for pver in $PVERS; do
  out="ref_af_dcmip_z${pver}.txt"
  ./af_dcmip_ref.x "$PCOLS" "$pver" "$out"
  echo "OK -> $HERE/$out"
done
