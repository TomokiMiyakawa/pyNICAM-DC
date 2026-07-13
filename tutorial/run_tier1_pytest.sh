#!/bin/bash
# TIER 1 -- unit test suite (fast, CPU-only, no input dataset needed).
# Runs the pytest suite: kernel bit-exactness (numpy, + jax if installed), RNG
# determinism, mpi4py availability. jax and full-model (toml) tests importorskip
# themselves when those deps are absent, so this passes on a minimal numpy env.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/.." && pwd)"            # repo root = parent of tutorial/
export PYTHONPATH="$CODE"

echo "=== TIER 1: pytest unit suite (repo: $CODE) ==="
cd "$CODE"
# mpirun -np 1 gives mpi4py a valid singleton context (needed on some clusters'
# login nodes; harmless on a laptop). Falls back to bare pytest if no mpirun.
if command -v mpirun >/dev/null 2>&1; then
  mpirun -np 1 python -m pytest test/ -q
else
  python -m pytest test/ -q
fi
rc=$?
echo "=== TIER 1 exit=$rc ($([ $rc -eq 0 ] && echo PASS || echo FAIL)) ==="
exit $rc
