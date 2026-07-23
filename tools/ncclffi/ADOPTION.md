# NCCL-FFI halo transport — adoption status (2026-07-23)

Direct-NCCL halo exchange via an XLA FFI custom call, replacing the host-staged
dense `mpi4jax.alltoall`. Gate: `PYNICAM_COMM_NCCLFFI=1` (default **0**).
Plan: `workforclaude/nccl-ffi-plan_v2.txt`. Lessons: `nccl-ffi-lessons_2026-07-23.txt`.

## Evidence (all committed harnesses under tools/ncclffi/)

| check | config | result |
|---|---|---|
| bit-exact A/B vs alltoall | gl05 pe4 (production resident+fused) | EXACTLY 0.0 all fields |
| bit-exact A/B | gl05 pe40 (pole/singular/15-partner) | EXACTLY 0.0 |
| bit-exact A/B | gl09 pe20 (where ragged crashed) | EXACTLY 0.0 |
| order audit (send==recv per call/pair) | pe4 / pe20 / pe40 / pe64 | >240k pairs, 0 mismatches |
| device-resident (nsys) | gl09 pe40 | nccl kernels only, 0 HtoD/DtoH |
| perf | gl09 pe20 fp64 | 0.169 -> 0.150 s/step (-11.5%) |
| **perf headline** | **gl11 pe64 fp32 z40** | **alltoall 0.800-0.921 -> 0.313 s/step (2.6x), jitter ±80% -> ±0.5%** |
| weak scaling (27.5M cells/GPU, 4->64 GPU, FFI-vs-FFI) | gl09-pe4 0.3071 vs gl11-pe64 0.3132 | efficiency 42% -> **98.1%** |
| moist/forced bit-exact | jm11 pe4 (Kessler+SM, 6 tracers, z30) | EXACTLY 0.0 + audit clean |
| gl11 pe80 | fp32 z40 | alltoall 0.818 -> **0.2975** (sweet spot flips back to 80 GPU; 64->80 scaling POSITIVE again) |
| small-scale perf | gl09 pe4 fp32 | 0.3156 -> 0.3071 (-2.7%; COMM share small at pe4) |

## Why it is exactly bit-equal
The dense pack/unpack is byte-identical to the alltoall path; only the wire
transport changes (grouped ncclSend/Recv of partner rows). Pure data movement
=> any nonzero A/B diff is a bug by definition (and one such bug was found and
fixed this way: the thunk-scheduler exchange reorder, see plan v2 §2).

## Correctness contract (MUST-KEEP invariants)
1. **Ordering token** threaded as a real operand/result of every FFI call
   (hard buffer edge). NCCL is tag-less: uniform cross-rank call order is a
   correctness precondition. Never "optimize away" the token; an
   optimization_barrier tie is NOT sufficient (HLO-only, dropped by the GPU
   thunk scheduler).
2. **Order-audit regression** (`audit_regression.pbs`, ~5 min): run after any
   change to COMM paths / dynamics-vi loop structure / jax-jaxlib-NCCL
   versions, and once per new (glevel, pe) decomposition. PASS = 0 mismatches.
3. Never place two transports (NCCL FFI + mpi4jax) in one jit graph.

## How to enable
1. Build once per software stack: `tools/ncclffi/build_ncclffi.sh`
   (links the venv's libnccl.so.2 = the copy jax loads).
2. `export PYNICAM_COMM_NCCLFFI=1` (e.g. in config/production.env).
   Missing/failed lib raises loudly at COMM setup (no silent fallback).
3. First run at a new decomposition: `qsub tools/ncclffi/audit_regression.pbs`
   (with `-v G=..,PE=..`).

## Default-flip checklist (before setting =1 in production.env)
- [x] bit-exact at 3 decompositions + production fused config
- [x] order audit at 4 decompositions incl. the production gl11 pe64
- [x] scale perf win measured (2.65x at the weak-scaling wall)
- [x] moist/forced case A/B at pe4: jm11 (Kessler+SM, 6 tracers, z30)
      EXACTLY 0.0 all vars + audit clean 5376 pairs (a1_jm11_pe4.pbs)
- [x] .so build folded into the standard environment setup (build_venv2.sh STAGE 5)

## Per-step decomposition at gl11 pe64 fp32 (nsys graph-trace, rank0)
| component | alltoall | NCCL-FFI |
|---|---|---|
| dyn-core loop-fusion kernels | 213.8 ms | 213.3 ms (identical) |
| other fusion kernels | 79.8 ms | 79.4 ms (identical) |
| COMM (MPI host / NCCL kernels) | 432.3 ms | 10-19 ms |
| staging memcpy H2D+D2H | 69.2 ms (~20k copies) | ~0 |
| D2D memcpy | 8.5 ms | 8.3 ms |
| = accounted vs clean wall | ~800 ms | ~313 ms (GPU-saturated, no dispatch gap) |

## Remaining headroom (optional, plan v2 §3)
- ~~N2b prefix trim~~ DONE (9b25a6d): wire = exact payload; -10% at gl11 pe64.
  Optional remainder: concatenated sparse pack (removes pack-side zero writes)
- N5: COMM/compute overlap (the 0.347 still contains in-kernel peer waits,
  long tail up to 36 ms)
