#!/bin/bash
# Build a jax-ROCm virtualenv for pyNICAM-DC on the AMD box (MI300/gfx942 target).
# Idempotent-ish; re-run to upgrade. Adjust ROCM_MAJOR to match the box's ROCm.
#
#   ROCM_MAJOR=6 VENV=$PWD/venv-rocm bash build_venv_rocm.sh
#
# What the model needs: numpy, toml, jax(rocm), mpi4py (+ optional mpi4jax for the
# non-RCCL comm path), zarr<3 + xarray + dask for output. jax-rocm wheels come from
# the ROCm plugin packages (jax-rocm${M}0-plugin / -pjrt) selected by "jax[rocm]".
set -euo pipefail
VENV="${VENV:-$PWD/venv-rocm}"
ROCM_MAJOR="${ROCM_MAJOR:-6}"           # 6 -> jax-rocm60 wheels; set 7 for ROCm 7.x
PY="${PY:-python3}"

echo "=== creating venv: $VENV (python=$PY, ROCm major=$ROCM_MAJOR) ==="
"$PY" -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel

echo "=== core deps ==="
pip install "numpy>=1.24" "toml>=0.10.2" "zarr<3" xarray dask netCDF4

echo "=== jax (ROCm) ==="
# Preferred: the metapackage that pulls the matching ROCm PJRT plugin.
#   pip install "jax[rocm]"
# If your ROCm needs an explicit index (AMD hosts wheels), use e.g.:
#   pip install jax jaxlib jax-rocm${ROCM_MAJOR}0-plugin jax-rocm${ROCM_MAJOR}0-pjrt \
#       -f https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_MAJOR}.x/
pip install "jax[rocm]" || {
  echo "!!! 'jax[rocm]' failed -- install the ROCm plugin wheels matching your ROCm";
  echo "    see https://github.com/ROCm/jax or repo.radeon.com; then re-run.";
  exit 1;
}

echo "=== mpi4py (against the box MPI; set MPICC if not on PATH) ==="
MPICC="${MPICC:-mpicc}" pip install --no-binary=mpi4py mpi4py

echo "=== mpi4jax (optional: the non-RCCL device-comm path) ==="
pip install mpi4jax || echo "  (mpi4jax optional; RCCL-FFI path does not need it)"

echo; echo "=== verify ==="
python - <<'PY'
import jax
print("jax", jax.__version__)
devs = jax.devices()
print("devices:", devs)
print("device_kind:", getattr(devs[0], "device_kind", "?"))
print("ROCM detected:", "AMD" in str(getattr(devs[0], "device_kind", "")).upper())
PY
echo "=== done. venv: $VENV ==="
