#!/usr/bin/env bash
# Build + run the forcing_step (part A) reference driver -> ref_forcing_step_z*.txt
# (consumed by test_forcing_step.py). Compiles the UNMODIFIED simple_physics_v6.f90
# with the forcing_step driver (verbatim L253-390 core). SP_FC overridable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP_SRC="${SP_SRC:-/work/gj37/c24028/workforclaude/nicamdc/src/nhm/share/dcmip/simple_physics_v6.f90}"
SP_FC="${SP_FC:-gfortran}"
FFLAGS="${FFLAGS:--O2 -ffree-line-length-none}"
cd "$HERE"
$SP_FC $FFLAGS -c "$SP_SRC" -o simple_physics_v6.o
$SP_FC $FFLAGS forcing_step_ref.f90 simple_physics_v6.o -o forcing_step_ref.x
for pver in ${PVERS:-30 40 78}; do
  ./forcing_step_ref.x "${PCOLS:-5}" "$pver" "ref_forcing_step_z${pver}.txt"
  echo "OK -> $HERE/ref_forcing_step_z${pver}.txt"
done
