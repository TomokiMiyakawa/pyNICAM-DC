#!/usr/bin/env bash
# Build + run the DCMIP simple-physics reference driver, producing
# ref_simple_physics.txt (consumed by test_simple_physics.py).
#
# Compiles the UNMODIFIED nicamdc simple_physics_v6.f90 with the reference
# driver. Points at the nicamdc source tree via SP_SRC (override if moved).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP_SRC="${SP_SRC:-/work/gj37/c24028/workforclaude/nicamdc/src/nhm/share/dcmip/simple_physics_v6.f90}"
# Use its own var (the environment often exports FC=nvfortran via module loads;
# nvfortran rejects -ffree-line-length-none). Default gfortran; override SP_FC=.
SP_FC="${SP_FC:-gfortran}"
FFLAGS="${FFLAGS:--O2 -ffree-line-length-none}"

if [[ ! -f "$SP_SRC" ]]; then
  echo "ERROR: simple_physics_v6.f90 not found at $SP_SRC (set SP_SRC=...)" >&2
  exit 1
fi

cd "$HERE"
echo "compiler: $($SP_FC --version | head -1)"
echo "source  : $SP_SRC"
$SP_FC $FFLAGS -c "$SP_SRC" -o simple_physics_v6.o
$SP_FC $FFLAGS -c simple_physics_ref.f90 -o simple_physics_ref.o
$SP_FC $FFLAGS simple_physics_v6.o simple_physics_ref.o -o simple_physics_ref.x
./simple_physics_ref.x
echo "OK -> $HERE/ref_simple_physics.txt"
