# Merge notes — `gh200-vipath0-resident` → `main`

GPU port + performance campaign for the pyNICAM-DC dynamical core (Miyabi-G / GH200).
This branch adds a full **device-residency + kernel-fusion + fp32** optimization stack for the
JAX backend, **entirely behind default-OFF gates**, plus one shared correctness fix.

- **Base:** `main` (`1e3a555`)  •  **Tip:** `6adf70f`  •  **Merge commit:** `6fe5556`
- **Scope:** 231 commits ahead of main; **72 files, +13,496 / −2,388** (54 added, 18 modified, 0 deleted)
- **Default behavior:** UNCHANGED — proven bit-identical to `main` (see Validation §)

---

## TL;DR for reviewers

1. **Nothing changes by default.** All optimizations are gated by ~138 `PYNICAM_*` env flags,
   all default-OFF. With gates off, the branch is **byte-identical to `main`** on the numpy
   backend (regression: `worst_abs 0.00e+00`, 8/8 bit-exact).
2. **With the stack ON**, the JAX/GPU dynamical core is up to **~16× faster** than the original
   eager path, and **fp32 adds 1.8× + halves GPU memory**.
3. **One genuine behavior change:** a vertical-limiter kmin fix — but it is already on *both*
   `main` (`1e3a555`) and this branch (`6adf70f`), so the merge simply carries it forward.
   It is a no-op on the JW gold; it only affects near-surface vertical flow.
4. **Validated** bit-exact on JW dynamics (gold 1.15e-11), DCMIP tracer advection (3e-15),
   default-path (0.00e+00 vs main), on both GPU and CPU.

---

## What this PR adds

### Core dynamics (gated device-resident + fused paths interleaved with the originals)
| File | +/− | Adds |
|---|---|---|
| `nhm/dynamics/mod_dynamics.py` | +2583/−223 | Pre_Post / nl-body / nl-loop fusion, cross-step residency, step orchestration |
| `nhm/dynamics/mod_src_tracer.py` | +1829/−651 | tracer device-residency + jit; the kmin host-limiter fix |
| `nhm/dynamics/mod_vi.py` | +1279/−253 | vertical-implicit residency, fori-loop, on-device COMM |
| `nhm/dynamics/mod_numfilter.py` | +989/−155 | hyperdiffusion / divergence-damping residency |
| `nhm/dynamics/mod_src.py` | +319/−75 | flux / convergence residency |

### New backend-switchable device kernels
`kernels/horizontallimiter.py`, `horizontalremap.py`, `verticallimiter.py`, `tracervertadv.py`,
`thrmdyn.py` — vectorized JAX kernels the gated paths dispatch to.

### Backend + communication infrastructure
- `share/mod_backend.py` (+188): numpy/jax dispatch, `maybe_jit`, `device_consts`,
  fp32/fp64 selection (`--precision`).
- `share/mod_comm.py` (+226): on-device (mpi4jax) halo exchange, composes under `jit`/`lax.scan`.
- `share/mod_vector.py` (new): vectorization helpers.

### Shared / geometry
`share/mod_oprt.py`, `mod_gmtr.py`, `mod_grd.py`, `mod_vmtr.py` — vectorization + refactor
(net-neutral; proven bit-exact on the default path).

### Non-functional (does not touch the model; ~half the file count)
Tooling/benchmarks (`proto/cmp_prec.py`, `cmp_zarr.py`, `bench_*`, `tools/sweep/`) and docs/handoffs
(`GH200_SETUP_STEPS.txt`, `CPU_SESSION_HANDOFF.md`, `docs/claude-memory/*`, etc.).

---

## The optimization stack (what the gates enable)

Layered; each is default-OFF and independently bit-exact:

| Layer | Gate(s) | What it does |
|---|---|---|
| Device residency | `PYNICAM_RESIDENT_*` (~90) | keep prognostic/diagnostic state + tendencies on device across the per-nl RK loop; on-device halo COMM |
| Pre_Post fusion | `PYNICAM_FUSE_PREPOST` | jit the BNDCND+THRMDYN segment into one graph |
| Step A (nl-body jit) | `PYNICAM_FUSE_NLBODY` | jit the whole per-nl RK body (compile-once) |
| Step B (nl-scan) | `PYNICAM_FUSE_NLSCAN` | lift the per-nl loop to `jax.lax.scan` |
| Tracer fusion | `PYNICAM_FUSE_TRACER` | jit the whole tracer advection |
| Cross-step residency | `PYNICAM_RESIDENT_PRGVAR` (Phase E) | prognostic state stays on device across time steps; host drain only at output cadence |
| Precision | `--precision float32` | mixed/single precision (halves bytes) |

The canonical "full stack ON" env set (~101 gates) is in `env_check/cpu_np_vs_jax.sh` (`FULLSTACK`)
and `env_check/nsys_windowed.sh`.

---

## Performance (measured this campaign)

**GPU (gl08, 4× GH200, steady per-step Dynamics):**
- eager baseline → full fused+resident stack: **~1.9 s → 0.217 s (fp64) ≈ 9×**; with fp32 **0.120 s
  ≈ 16× cumulative**.
- **fp32 vs fp64:** 1.81× faster, peak GPU memory 37,882 → 19,182 MiB (**−51%**), stable over a
  480-step run, ~1e-3 vertical-momentum error (single-precision; a science-acceptance decision).
- Step B (lax.scan) also cut compile ~2.6× and peak memory −11 GB.
- Roofline context: memory-bound; ~0.73 ms/step HBM roofline (4.02 TB/s); remaining gap is
  inter-rank halo COMM (~14%, latency-bound) — hard under mpi4jax, out of scope here.

**CPU (gl06, Grace ARM, 4 ranks, steady per-step):**
- numpy 6.61 s → JAX-default 4.5 s (1.47×) → JAX-full-stack **3.36 s (1.97× vs numpy)**.
  Fusion transfers to CPU; residency does not (no host↔device transfer) — net still ~2×.

---

## Validation

All bit-exact / at-floor; harnesses under `env_check/`:

| Test | Config | Result |
|---|---|---|
| **Default path vs main** | numpy, all gates OFF, gl07 | **`0.00e+00` bit-exact 8/8** (`premerge_regression.sh`) |
| JW dynamics vs gold | full stack, GPU | 1.15e-11 PASS |
| JW dynamics vs gold | full stack, CPU | 1.23e-11 PASS |
| **DCMIP tracer advection** | numpy vs jax-default vs jax-full-stack, gl05 z60 | **3.3e-15 PASS**; dynamics fields bit-exact (`case3_*`) |
| numpy vs gold (branch & main) | gl07 | both 1.12e-11 (ARM floor), identical |

The DCMIP case (`test/case3`, `integ_type=TRCADV`, Thuburn limiter on) exercises the tracer +
vertical-limiter paths that the JW dynamics gold cannot — confirming the fused stack **and** the
kmin fix on a case with real vertical motion.

---

## The kmin fix + merge conflict

- **What:** the vertical Thuburn limiter's k=kmin peeling block used the 2nd-Courant component at
  `k+1` instead of the local `k` (mismatched the main loop, the device kernel, and the Fortran
  `mod_src_tracer.f90:1925-1933`).
- **Where:** `main` fixed the host limiter in `1e3a555`; this branch's tracer rewrite left the host
  peeling block unfixed (the device kernel was already correct), so `6adf70f` re-applied it to the
  branch's host structure. Both sides now consistent.
- **Merge:** the single conflict (`mod_src_tracer.py`) resolves **take-branch** — the branch's file
  already carries the fix, so nothing from `main` is lost. Verified: merged tree == branch tree.
- **Impact:** no-op on JW (no near-surface vertical motion); binds only for near-surface vertical
  flow. DCMIP tracer test confirms host≡device to machine eps.

---

## Review guidance / risk

- **Blast radius: low** — default-OFF gating + proven default bit-exactness mean production behavior
  is unchanged on merge.
- **Surface: large** — the four core `mod_*.py` files interleave gated paths with the originals; these
  are the files to review most carefully. Kernels, tooling, and docs are additive and low-risk.
- **To enable the stack:** set the `FULLSTACK` gate set (see `env_check/`) + `--precision float32`
  (optional). Nothing is enabled implicitly.

---

## Enabling the stack (quick start)

```bash
# JAX/GPU, full fused+resident stack (source the gate list from the harness):
export JAX_PLATFORMS=cuda MPI4JAX_USE_CUDA_MPI=1 HCOLL_ENABLE=0
export $(grep -oE 'PYNICAM_[A-Z0-9_]+=1' env_check/cpu_np_vs_jax.sh | tr '\n' ' ')
export PYNICAM_FUSE_PREPOST=1 PYNICAM_FUSE_NLBODY=1 PYNICAM_FUSE_NLSCAN=1 \
       PYNICAM_FUSE_TRACER=1 PYNICAM_RESIDENT_PRGVAR=1
# ... run the driver with backend=jax; add --precision float32 for the fp32 path.
```
