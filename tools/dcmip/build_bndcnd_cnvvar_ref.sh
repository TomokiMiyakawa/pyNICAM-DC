#!/usr/bin/env bash
# Build + run the BNDCND_pre_sfc / cnvvar_vh2uv / cnvvar_uv2vh reference driver
# -> ref_bndcnd_cnvvar.txt (consumed by test_bndcnd_cnvvar.py).
# Self-contained (no external Fortran deps); SP_FC overridable (env FC=nvfortran
# collides -- nvfortran rejects -ffree-line-length-none).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP_FC="${SP_FC:-gfortran}"
FFLAGS="${FFLAGS:--O2 -ffree-line-length-none}"
IJDIM="${IJDIM:-6}"; KDIM="${KDIM:-12}"
cd "$HERE"
$SP_FC $FFLAGS bndcnd_cnvvar_ref.f90 -o bndcnd_cnvvar_ref.x
./bndcnd_cnvvar_ref.x "$IJDIM" "$KDIM" ref_bndcnd_cnvvar.txt
echo "OK -> $HERE/ref_bndcnd_cnvvar.txt"
