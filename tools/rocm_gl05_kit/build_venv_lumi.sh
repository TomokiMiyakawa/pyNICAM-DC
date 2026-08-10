#!/bin/bash
# LUMI variant of build_venv_rocm.sh.
#
# Why not the generic script: it does `pip install "jax[rocm]"`, and (a) on jax >=0.5
# there is no `rocm` extra -- pip ignores unknown extras and exits 0, silently leaving
# a CPU-only jax (the trap already recorded in PORT.md) -- and (b) LUMI's ROCm is 6.3.4,
# whose plugin wheels are named jax-rocm60-{plugin,pjrt} and top out at 0.5.0. So we pin
# explicitly and then FAIL HARD if the result is not a ROCm device.
#
#   source tools/rocm_gl05_kit/lumi_env.sh && bash tools/rocm_gl05_kit/build_venv_lumi.sh
set -euo pipefail
: "${VENV:?source lumi_env.sh first}"

JAXVER="${JAXVER:-0.5.0}"

echo "=== creating venv: $VENV (python=$(python3 -V 2>&1)) ==="
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel

echo "=== core deps ==="
pip install "numpy>=1.24" "toml>=0.10.2" "zarr<3" xarray dask

echo "=== jax $JAXVER + ROCm 6.x PJRT plugin (explicit, not the phantom [rocm] extra) ==="
pip install "jax==$JAXVER" "jaxlib==$JAXVER" \
            "jax-rocm60-plugin==$JAXVER" "jax-rocm60-pjrt==$JAXVER"

echo "=== mpi4py against Cray MPICH (cc wrapper, not mpicc) ==="
MPICC="cc -shared" pip install --no-binary=mpi4py --no-cache-dir mpi4py

echo
echo "=== verify (CPU-side import only; device check must run on a GPU node) ==="
python - <<'PY'
import jax, jaxlib, mpi4py
print("jax", jax.__version__, "jaxlib", jaxlib.__version__, "mpi4py", mpi4py.__version__)
import importlib.util as u
# The plugin wheel is an importable module; the pjrt wheel is NOT -- it only drops the
# PJRT shared object under jax_plugins/xla_rocm60/. Check each the way it actually ships.
assert u.find_spec("jax_rocm60_plugin"), "jax_rocm60_plugin missing -- jax falls back to CPU"
print("jax_rocm60_plugin", "installed")
import glob, os, sysconfig
so = glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "jax_plugins", "*rocm*", "*.so"))
assert so, "no ROCm PJRT .so under jax_plugins/ -- jax falls back to CPU"
print("jax_rocm60_pjrt  installed:", so[0])
import jax.ffi; print("jax.ffi.include_dir:", jax.ffi.include_dir())
PY
echo "=== done. venv: $VENV  (run check_gpu_lumi.sh on a GPU node next) ==="
