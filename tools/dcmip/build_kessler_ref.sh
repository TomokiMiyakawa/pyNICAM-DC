#!/usr/bin/env bash
# Build + run the KESSLER column-scheme reference driver -> ref_kessler_z*.txt
# (consumed by test_kessler.py). Compiles the UNMODIFIED kessler.f90.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K_SRC="${K_SRC:-/work/gj37/c24028/workforclaude/nicamdc/src/nhm/share/dcmip/kessler.f90}"
SP_FC="${SP_FC:-gfortran}"
FFLAGS="${FFLAGS:--O2 -ffree-line-length-none}"

cd "$HERE"
echo "compiler: $($SP_FC --version | head -1)"
# faithful golden: bare REAL locals stay f32 (matches the production nicamdc build)
$SP_FC $FFLAGS -c "$K_SRC" -o kessler.o
$SP_FC $FFLAGS -c kessler_ref.f90 -o kessler_ref.o
$SP_FC $FFLAGS kessler.o kessler_ref.o -o kessler_ref.x
# all-f64 golden: -fdefault-real-8 promotes the bare REAL locals to REAL(8), so
# kessler.py with lp=float64 must match this to machine precision (proves the
# f32 locals are the only precision effect and the algorithm is otherwise exact)
$SP_FC $FFLAGS -fdefault-real-8 -c "$K_SRC" -o kessler_f64.o
$SP_FC $FFLAGS kessler_f64.o kessler_ref.o -o kessler_ref_f64.x

PCOLS="${PCOLS:-5}"
PVERS="${PVERS:-30 40 78}"
for pver in $PVERS; do
  ./kessler_ref.x     "$PCOLS" "$pver" "ref_kessler_z${pver}.txt"
  ./kessler_ref_f64.x "$PCOLS" "$pver" "ref_kessler_f64_z${pver}.txt"
  echo "OK -> $HERE/ref_kessler_z${pver}.txt (+ _f64_)"
done
