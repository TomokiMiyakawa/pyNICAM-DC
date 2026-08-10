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
- A python3 with a working `venv`. On stock Ubuntu 24.04 `python3 -m venv` fails
  ("ensurepip is not available") until `apt install python3.12-venv`.
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
Confirms `jax.devices()` shows the MI300s and `device_kind` contains "AMD".

**Check that, don't assume it.** On jax 0.11 there is no `rocm` extra — the extras are
`cuda`, `cuda12`, `cuda13`, `rocm7-local`, `oneapi`, ... pip *ignores an unknown extra
and exits 0*, so the script's `pip install "jax[rocm]"` succeeds while installing
**CPU-only jax**, its `||` fallback never fires, and `jax.devices()` returns
`[CpuDevice(id=0)]`. Observed on ROCm 7.0.2 / jax 0.11.0. Install the real thing:
```bash
pip install "jax[rocm7-local]==0.11.0"   # -> jax-rocm7-plugin + jax-rocm7-pjrt
```
`rocm7-local` links against the system ROCm. Note the wheels are `jax-rocm7-*`, NOT
`jax-rocm70-*`: the `${M}0` naming in the script's fallback comment is a ROCm-6
artifact and 404s on PyPI for 7.x. Verify before moving on:
```bash
python -c "import jax; d=jax.devices(); print(d, d[0].device_kind)"
# want: [RocmDevice(id=0)] AMD Instinct MI300X ...   NOT CpuDevice
```
The script aborts at its mpi4py step on a box without `mpicc` (`set -e`), which is
*before* its own verify block — so a silent CPU-jax venv passes through unnoticed.
On an MI300X VF, `jax.devices()` also emits harmless `rsmi_dev_gpu_metrics_info_get
failed` / "Assuming PCIe Gen4 x16" warnings: PCIe metrics aren't readable under SR-IOV.

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

**Stage 1 needs no MPI — but only with the `nproc==1` guard in `mod_comm.py`.**
`drv_1gpu.toml` sets `comm = "serial"`, so mpi4py is genuinely unused (the
`_SerialComm` stub in `mod_process.py` carries the 1-rank case). mpi4jax, however,
used to be reached anyway: `COMM_data_transfer` dispatches to
`_comm_data_transfer_ondevice` for any jax array with no rank guard, and `_core`
called `mpi4jax.alltoall` unconditionally. At 1 rank that call is degenerate — all
halo traffic is same-rank and lands in the Copy lists, so `a2a_send`/`a2a_recv` are
empty, `a2a_chunk` is 0, and the result is never read — and it could not have worked
regardless, since `prc.comm_world` is the Python stub, not a real `mpi4py.MPI.Comm`.
`mod_comm.py` now imports mpi4jax only when `prc_nprocs > 1` and substitutes the
identity (`rt = st`) at 1 rank. Bit-exact; pe>1 is untouched.

**`validate.py` can report a false FAIL.** Its relative denominator is
`max(|ref|, atol)` with `atol = 1e-12`, so an element that is numerically zero
against its own field scale inflates `max_rel` without meaning anything. Measured on
MI300X: `max_rel = 1.17e-02`, from `ref = 3.80e-13` vs `cand = 3.92e-13` — a 1.17e-14
roundoff on a momentum field whose RMS is 5.47. Every one of the 3078
over-tolerance elements had `|ref| <= 1.77e-4`. Judge by each field's own scale
instead (`max|d| / rms`), which put the whole state at the expected floor:
```
v0 1.8e-15   v1 7.7e-14   v2 7.7e-14   v3 7.7e-12
v4 2.8e-11   v5 3.1e-15   v6 bit-identical      -> worst 2.8e-11
```

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
- This kit was assembled on the NVIDIA/Miyabi side; the ROCm-specific steps were
  **untested on AMD hardware** by construction — that's what the rental box is for.

## Bring-up log — 1×MI300X box (2026-08-10)

Box: Ubuntu 24.04, ROCm 7.0.2 (`hipcc` 7.14), rccl 2.26.6 (NCCL ABI 22606), header at
`include/rccl/rccl.h` (modern layout — no `ncclffi_hip.cpp` edit needed), **one**
MI300X VF (gfx942), no MPI installed.

Verified here:
- **Stage 1 PASSES.** gl05rl00 pe1, 8 steps, `peacefully done`, `fin_rank0.npy`
  written; numpy-CPU vs jax-ROCm at the expected floor (worst 2.8e-11 scale-relative,
  see Stage 1 above). Required the `nproc==1` guard in `mod_comm.py`.
- **RCCL toolchain links and runs**: a probe built with `build_ncclffi_rocm.sh`'s exact
  flags (`hipcc -std=c++17 --offload-arch=gfx942 -I$ROCM/include -lrccl`) compiled,
  resolved `librccl.so.1` + `libamdhip64.so.7`, and returned `ncclGetUniqueId` rc=0.
- jax-ROCm wheel resolution — see the `rocm7-local` trap in step 1.

NOT testable on a 1-GPU box, still open for the 8-GPU droplet: Stages 2a/2b, the
`build_ncclffi_rocm.sh` build itself, the mpi4jax path, `bind_rocm.sh`, and the
2a-vs-2b bit-identical A/B that is the actual RCCL correctness argument.

## Bring-up log — LUMI, 8 GCDs (2026-08-10)

Box: LUMI-G (HPE Cray EX), **4x MI250X = 8 GCDs per node, gfx90a** — not the gfx942 this
kit assumes. ROCm 6.3.4, RCCL 2.21.5, Cray MPICH, Slurm. Use `lumi_env.sh` +
`build_venv_lumi.sh` + `bind_lumi.sh` + `run_8gpu_rccl_lumi.sh`; the generic
`build_venv_rocm.sh` / `bind_rocm.sh` / `run_8gpu_*.sh` are mpirun-shaped and do not
apply here.

**Stage 2a is skipped on LUMI by decision** (mpi4jax is not expected to work), so the
2a-vs-2b bit-identical A/B is not the correctness argument here. The reference is the
**numpy-CPU pe8 leg** instead — cross-backend, so the bar is the ~1e-9 floor, not zero.

Cleared:
- jax-ROCm venv. ROCm 6.x plugin wheels are `jax-rocm60-{plugin,pjrt}` and stop at
  **0.5.0**, so jax/jaxlib pin to 0.5.0. Same `[rocm]`-extra trap as the MI300X box, one
  version earlier. `jax.devices()` -> 8x `RocmDevice`, kind "AMD Instinct MI250X".
- `build_ncclffi_rocm.sh` **builds** (this was untested before): `OFFLOAD_ARCH=gfx90a`,
  warnings only, `ldd` resolves `librccl.so.1` + `libamdhip64.so.6`.
- numpy-CPU pe8 reference: 8/8 `peacefully done`, 8 dumps.
- **The RCCL communicator bootstraps on all 8 ranks** — `NCCLFFI: comm up nprocs=8`,
  RCCL 2.21.5 via the unchanged `nccl*` symbols. The ABI-compatibility premise holds.

**OPEN — Stage 2b does not complete.** Seconds after the comm comes up, rank 0 dies in
the exchange with `Memory access fault by GPU node-4 (Agent handle: ...) Reason: Unknown`
and the step is torn down; 0/8 dumps. Eliminated so far: `NCCL_P2P_DISABLE=1` gives the
*identical* fault, so it is not the P2P/IPC direct-copy path. Not yet separated: whether
this is the RCCL wire at all or a generic jax-ROCm/gfx90a dycore fault — Stage 1 cannot
serve as that control because the 1-GPU grid is not in the repo and its tarball is not on
this box. Next probe is `rccl_spike.py`, which drives the production
`nicam_halo_exchange` handler with a synthetic tagged plan (seconds, vs ~5 min of XLA
compile for a pe8 model run).

Two launcher bugs in the generic kit, found here and fixed in the `lumi_*` counterparts:
- `bind_rocm.sh` sets `ROCR_VISIBLE_DEVICES` **and** `HIP_VISIBLE_DEVICES` to the same
  index. They compose: ROCR masks first and renumbers what survives, so rank N>0 then
  indexes past the end and sees **zero** GPUs. Set exactly one.
- `OMPI_COMM_WORLD_LOCAL_RANK` is unset under `srun`, so every rank takes the `:-0`
  default and lands on GCD 0. Local rank is `SLURM_LOCALID`.
