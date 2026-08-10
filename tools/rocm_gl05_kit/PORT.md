# pyNICAM-DC on AMD GPU (ROCm + RCCL) — glevel-5 toolkit

A staged bring-up for running the pyNICAM-DC dynamical core on a rented **AMD GPU
box (MI300/MI325, gfx942)**, replacing the NVIDIA CUDA + NCCL collective path with
**ROCm + RCCL**. Everything here is gl05 (small, fast, fits one GPU or eight).

## TL;DR — what actually changes for AMD

pyNICAM-DC's default multi-GPU halo exchange uses **mpi4jax** (device-resident
alltoall over GPU-aware MPI) — that path has **no NCCL dependency at all**; on AMD it
just needs ROCm-aware MPI. **NCCL is used only on the opt-in `PYNICAM_COMM_NCCLFFI=1`
path**, isolated to one source file: `tools/ncclffi/ncclffi.cu`. Because **RCCL is
ABI-compatible with NCCL** (identical `nccl*` symbols), the port of that file is a
mechanical CUDA→HIP hipify — every `ncclSend/ncclRecv/...` call is unchanged; only
the CUDA runtime calls, the stream type, the headers, and the compiler/link flags
change. The hipified source is `tools/ncclffi/ncclffi_hip.cpp` (already generated).

The only Python change is one NVIDIA-neutral line in `mod_ncclffi.py`: the FFI target
is now registered as `platform="ROCM"` on AMD (auto-detected from the device kind;
override with `PYNICAM_FFI_PLATFORM`).

## Stages

| Stage | Script | GPUs | Comm | Exercises |
|-------|--------|------|------|-----------|
| 1 | `run_1gpu.sh` | 1 | none | jax-ROCm runs the dycore (gl05rl00 pe1) |
| 2a | `run_8gpu_mpi4jax.sh` | 8 | mpi4jax | multi-GPU without RCCL (ROCm-aware MPI) |
| 2b | `run_8gpu_rccl.sh` | 8 | **RCCL** | the RCCL halo exchange (gl05rl01 pe8) |

Do them in order: Stage 1 proves the ROCm jax stack; 2a proves the decomposition +
MPI; 2b swaps the wire to RCCL. 2b should be **bit-identical** to 2a (same plan, only
the transport differs) — that A/B is the RCCL correctness check.

## 0. Prerequisites on the AMD box
- ROCm 6.x (7.x: set `ROCM_MAJOR=7` in the venv build), `hipcc`, `rccl` + `rccl-dev`.
- An MPI with `mpicc` (OpenMPI/MPICH). For Stage 2a, GPU(ROCm)-aware MPI (UCX+ROCm);
  Stage 2b (RCCL) only needs plain MPI for the tiny uid bootstrap bcast.
- `git clone` this repo; everything below is relative to the repo root.
- **Grid inputs**: the 8-GPU grid (`pynicamdc/test/case2/json_gl05rl01pe08/`) ships
  with the clone. The **1-GPU grid is NOT in the repo** — extract the side tarball
  from the repo root before Stage 1:
  ```bash
  tar xzf /path/to/gl05_pe01_grid.tar.gz   # -> tutorial/case/grid_gl05rl00pe01/
  ```
  (711 KB; provided out-of-band. Verify `tutorial/case/grid_gl05rl00pe01/bboundary_GL05RL00.pe00000000.json` exists.)

## 1. Build the jax-ROCm venv
```bash
cd tools/rocm_gl05_kit
ROCM_MAJOR=6 VENV=$PWD/venv-rocm bash build_venv_rocm.sh
```
Confirms `jax.devices()` shows the MI300s and `device_kind` contains "AMD". If the
`jax[rocm]` metapackage doesn't resolve for your ROCm, install the explicit
`jax-rocm${M}0-plugin` / `-pjrt` wheels (see comments in the script / ROCm/jax repo).

## 2. Stage 1 — single-GPU smoke
```bash
VENV=$PWD/venv-rocm bash run_1gpu.sh          # jax-ROCm
```
Expect `peacefully done` and `fin_rank0.npy`. Then the numpy/jax cross-check:
```bash
VENV=$PWD/venv-rocm bash validate.sh          # numpy CPU vs jax-ROCm, rtol 1e-9
```
PASS = jax-ROCm within ~1e-9 of the numpy reference (cross-backend libm/reduction
floor — not zero; same as the ARM-vs-x86 gold floor).

## 3. Build the RCCL FFI lib (for Stage 2b)
```bash
cd ../ncclffi
VENV=../rocm_gl05_kit/venv-rocm ROCM_PATH=/opt/rocm OFFLOAD_ARCH=gfx942 \
    bash build_ncclffi_rocm.sh
# -> tools/ncclffi/rocm/libncclffi.so ; ldd shows librccl + libamdhip64
```

## 4. Stage 2 — multi-GPU
```bash
cd ../rocm_gl05_kit
# 2a: mpi4jax (no RCCL) — get 8-GPU working first
VENV=$PWD/venv-rocm bash run_8gpu_mpi4jax.sh
# 2b: RCCL halo exchange
VENV=$PWD/venv-rocm bash run_8gpu_rccl.sh
# A/B them — expect bit-identical (rtol 0)
source venv-rocm/bin/activate
python validate.py --ref run_8gpu_mpi4jax/fin_rank'*'.npy \
                   --cand run_8gpu_rccl/fin_rank'*'.npy --rtol 0
```
`run_8gpu_rccl.sh` prints `NCCLFFI: comm up nprocs=8 ...` when the RCCL communicator
bootstraps. Per-rank GPU binding is done by `bind_rocm.sh` (HIP_VISIBLE_DEVICES).

## Files
```
tools/ncclffi/ncclffi_hip.cpp        hipified RCCL FFI (nccl* symbols unchanged)
tools/ncclffi/build_ncclffi_rocm.sh  hipcc build -> rocm/libncclffi.so
tools/rocm_gl05_kit/
  build_venv_rocm.sh    jax-ROCm + mpi4py(+mpi4jax) venv
  bind_rocm.sh          per-rank HIP_VISIBLE_DEVICES
  configs/              nhm_{1gpu,8gpu}.toml + drv_*.toml (jax & numpy)
  run_1gpu.sh           Stage 1
  run_8gpu_mpi4jax.sh   Stage 2a
  run_8gpu_rccl.sh      Stage 2b (RCCL)
  validate.py / .sh     numpy-vs-ROCm and RCCL-vs-mpi4jax comparison
```

## Gotchas / notes
- **RCCL header path**: `<rocm>/include/rccl/rccl.h` on recent ROCm; older layouts
  use `<rocm>/include/rccl.h`. The build script checks both and adds `-I$ROCM/include`.
  If include fails, edit the `#include <rccl/rccl.h>` line in `ncclffi_hip.cpp`.
- **Device 0 per rank**: `mod_ncclffi.ncclffi_init(..., 0)` inits ROCm device 0 for
  every rank, relying on `HIP_VISIBLE_DEVICES` masking (bind_rocm.sh). Don't remove
  the bind wrapper or all ranks land on GPU 0.
- **FFI platform**: registered as ROCM via auto-detect; force with
  `PYNICAM_FFI_PLATFORM=ROCM` (or CUDA) if `device_kind` doesn't contain "AMD".
- **mpi4jax on ROCm**: keep `MPI4JAX_USE_CUDA_MPI=0` unless UCX ROCm RMA is verified;
  the host-staged exchange is the safe default. Stage 2b (RCCL) sidesteps this.
- **fp32**: add `precision="float32"` in the drv toml to test half-precision on AMD
  (1.8x faster, ~1e-3 on vertical momentum — a science tradeoff, not a bug).
- **IDEAL init** is used (no restart files) so numpy and jax start from identical ICs
  and Stage-1 validation is apples-to-apples. To test the bit-exact prognostic
  restart on AMD too, point `restartparam` at an npz restart and set
  `PYNICAM_RESTART_OUT_END` (see the restart-reproducibility work).
- This kit was assembled on the NVIDIA/Miyabi side; the ROCm-specific steps are
  **untested on AMD hardware** by construction — that's what the rental box is for.
  The hipify, configs, and Python platform switch are verified; the runtime bring-up
  (wheel resolution, MPI, RCCL link) is the box-side work.
