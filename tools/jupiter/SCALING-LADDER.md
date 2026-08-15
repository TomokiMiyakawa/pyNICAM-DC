# JUPITER weak-scaling ladder — results and findings (2026-07-29)

JAX/GH200 weak-scaling ladder for pyNICAM-DC on JUPITER (JSC), holding
**655,360 cells/GPU constant at every rung** (`10 * 4^gl / pe`). Companion to
`PORT.md` (the authoritative porting guide and trap ledger) and
`TODO-extreme-scale.md` (open items for rlevel 6 / 20480 GPUs).

Run from `/e/project1/jureap22/miyakawa/workClaude1`. Scripts live in `sweep/`,
which is **not** under version control — see "Unversioned work" at the end.


## 1. Results

| rung | rlevel | nodes | GPUs | job | min / mean s/step | weak-scaling eff. |
|------|--------|-------|------|-----|-------------------|-------------------|
| gl09 pe4    | rl01 | 1   | 4    | 1086307 | 0.3036 / 0.3133 | (baseline) |
| gl11 pe64   | rl03 | 16  | 64   | 1086463 | 0.3554 / 0.3602 | 87.0% |
| gl12 pe256  | rl05 | 64  | 256  | 1087535 | 0.4316 / 0.4444 | 70.5% |
| gl13 pe1024 | rl05 | 256 | 1024 | 1087682 | (pending)       | — |

gl11 pe64 = 0.3602 mean vs Levante 0.3604 and Miyabi 0.3016.

Physics validation — **both new rungs were the first time that resolution had been
run on any machine**, and both were clean on the first attempt:

| rung | smoke job | result |
|------|-----------|--------|
| gl12 pe256  | 1087508 | 256/256 ranks, `nfin=198744000/198744000`, budget step 0 finite |
| gl13 pe1024 | 1087565 | 1024/1024 ranks, all finite, `COMPLETED` in 5:24 |

So the resolution-dependent coefficients hold:

| rung | `dtl` = 1200/2^(g-5) | `gamma_h` = `alpha_d` |
|------|----------------------|------------------------|
| gl12 | 9.375  | 5.859375e9  |
| gl13 | 4.6875 | 7.32421875e8 |

These continue the `/8`-per-level sequence in `sweep/hires/make_hires_config.py`,
whose `HDIFF` table stops at gl11 (4.6875e10) — gl12/gl13 are extrapolations, now
validated. `dtl`, `gamma_h`, `alpha_d` are the ONLY glevel-dependent settings; the
`RK3` / `DIRECT` / `lap_order=2` / `IN_LARGE_STEP2` / `Jablonowski` block is identical
at every rung.


## 2. Regions per rank is the hidden variable

At fixed cells/GPU, the **rlevel/pe combination decides how those cells are split**,
and that — not the cell count — drives halo volume, fused-executable size, and the
weak-scaling loss.

| rung | rlevel | rgn/rank | cells/rgn | `gall_1d` | halo overhead | `rellist_nmax` |
|------|--------|----------|-----------|-----------|---------------|----------------|
| gl09 pe4    | rl01 | 10 | 65,536 | 258 | 1.6% | 10,270 |
| gl11 pe64   | rl03 | 10 | 65,536 | 258 | 1.6% | 10,278 |
| **gl12 pe256** | rl05 | **40** | 16,384 | 130 | **3.1%** | **20,638** |
| gl13 pe1024 | rl05 | 10 | 65,536 | 258 | 1.6% | ~10,320 |

**gl12 pe256 is the only rung that splits its cells into 40 small regions instead of
10 big ones**, giving it 2x the halo relations per GPU. Two consequences, both observed:

1. **Weak-scaling loss.** gl09->gl11 (16x GPUs) held 87%; gl11->gl12 (4x GPUs) held
   81%. The extra 19% is decomposition shape, not a scaling wall.
2. **The fp64 fused audit OOM'd** (see §3).

**Prediction to test with gl13 pe1024** (10 rgn/rank): its per-step cost should recover
toward gl11's rather than continuing to decay, despite 4x more GPUs.

If gl12's rung matters, **gl12 at rl04/pe256** would give 10 rgn/rank at the same
655,360 cells/GPU and should recover much of the 19%. It needs regenerated mnginfo and
boundary data, so it is an experiment, not a rerun.


## 3. Fused-executable constants scale with REGIONS, not cells

gl12 pe256 ARM A (fp64, `FUSE_TIMELOOP=1`, `TIMELOOP_CHUNK=1`, `NCCLFFI_CKSUM=1`,
`TIMELOOP_DUMP`) OOM'd on **all 256 ranks**:

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Failed to load in-memory CUBIN
  CUDA_ERROR_OUT_OF_MEMORY [executable_name='jit__step_fn']
```

preceded by a JAX warning: `A large amount of constants were captured during lowering
(2.95GB total)`. gl09 pe4 and gl11 pe64 pass the *same* fp64 fused audit. The
discriminator is regions per rank (§2) — the fused executable bakes in per-region and
per-halo-relation index arrays, so its constant footprint tracks region and relation
count, not cell count.

**Treat the 2.95 GB constant-capture warning as a leading indicator, not noise.**

Mitigations when a fused audit is needed at high regions/rank: run the audit at
**fp32** (it is a byte-level send-vs-recv checksum compare — fp64 buys nothing), drop
`TIMELOOP_DUMP`, or raise `TIMELOOP_CHUNK` above 1.


## 4. `Comm.Recv_nlim` / `Send_nlim` raised 20 -> 64

`mod_comm.py:68-69`. gl12 pe256 aborted in `_check_commnlim` (`:97`) with rank 76
needing 21 partners; the guard fired correctly rather than corrupting the fixed-size
tables. Symptom is `MPI_Abort(MPI_COMM_WORLD, 0)` plus
`*** [COMM] number of recv (r2r) partner ranks (21) reached the buffer limit nlim=20`
in the rank log, and `ranks done: 0/256` in the sbatch tail.

Memory cost is confined to the r2r tables (`:541-542`,
`6 * rellist_nmax * nlim * 8` B each, ~60 MiB at gl12 with nlim=64); p2r/r2p use
`Send_size_nglobal_pl` and are negligible.

The ceiling is geometric — a quad region has 4 edge + 4 corner neighbours:

```
max partner ranks <= 8 * regions_per_rank ,   regions_per_rank = 10 * 4^rl / pe
```

so **`nlim` is independent of glevel, and scaling ranks UP makes it safer.** Coarse
decompositions are the risk. At rlevel 6 / pe20480 (2 rgn/rank) the hard ceiling is 16,
so 64 needs no further bump.

Measured with `PYNICAM_COMM_DEGREE=1` (`:159-171`, prints the Allreduced global max):

| config | rgn/rank | predicted (mnginfo edge-only) | measured | error |
|--------|----------|-------------------------------|----------|-------|
| rl05 pe256  | 40 | 32 | **r2r=34**, pole=20, total=36 | -6% |
| rl05 pe1024 | 10 | 16 | **r2r=18**, pole=20, total=26 | -12.5% |

**Predicting a new decomposition offline**, without running: parse the mnginfo TOML —
`[RGN_MNG_INFO.*]` gives `PEID` + `MNG_RGNID` (region -> rank), `[RGN_LINK_INFO.*]`
gives `SW`/`NW`/`NE`/`SE` (region -> edge neighbours). Union each rank's regions'
neighbours, map to owners, drop self, take the max. Apply a **1.25x safety factor** —
the edge-only pass misses corner connectivity. A second hop overcounts (38 vs 34 at
pe256), so the truth sits between. Gotcha: `NUM_OF_MNG` appears in both the global
`[PROC_INFO]` block (where it is 1) and per-rank blocks (where it is the real count).


## 5. Harness traps found the hard way

**`AUDIT FAIL` can mean "the run died", not "the ordering is wrong".** The ARM A
parser's `ok` requires `done == pe`, so gl12's post-audit OOM flipped a verdict whose
checksum evidence was perfect: **180,336 pairs, 0 mismatches, 102 distinct calls, all
256 ranks reporting** — broader coverage than gl11's 93,432-pair audit that PASSED.
**Split the two conditions**; ordering correctness and run completion fail for
unrelated reasons.

**The sbatch scripts exit 0 even when `srun` fails**, because the run sits in a
subshell whose status is not propagated. Job 1087502 reported `COMPLETED 0:0` while
achieving `ranks done: 0/256`. **Never trust the Slurm exit code** — read the
`ranks done=` line. Worth fixing before 5120 nodes, where a silent failure is expensive.

**`TIMELOOP_CHUNK` should divide `(lstep_max - WARMUP)` exactly.** 43 - 3 = 40 with
`CHUNK=6` leaves a ragged `K=4` chunk that JIT-compiles a *second* time (~16-24 s/rank),
so 2 of 7 chunks per rank are compile-dominated. gl11's `CHUNK=4` divided evenly and
compiled once. The perf parser's `5*median` filter removes these correctly, but it
means only 30 of 43 steps are actually measured.

**Stale logs outlive a resubmit.** The run logs are overwritten by `>` at job start, so
while a resubmitted job is still PENDING the old log is intact and greps match the
*previous* failure. Move the log aside when resubmitting.


## 6. Verifying a run is physically sane

`tutorial/check_validation.py RUN.npy` reads a `PYNICAM_TIMELOOP_DUMP` per-rank
`(i,j,k,l,nvar)` dump and checks interior (halo/ghost stripped) finiteness, `RHOG > 0`
and `RHOGE > 0`. gl12 ARM B sample ranks:

Beyond positivity — "finite and positive" can still be badly wrong — check
**`|V| = |RHOGV|/RHOG`** (the Jablonowski jet peaks ~35-45 m/s, and 43 steps x 9.375 s
is only 403 s of model time, so the IC should barely have evolved) and the tracer field
(IDEAL init starts at identically zero). `check_fleet_sanity.py` (this directory) does
all of it across every rank of a dump set.

**gl12 pe256 ARM B, all 256 ranks, interior only:**

```
RHOG   range : 2.5709e-03 .. 1.5494e+00
RHOGE  range : 4.8298e+02 .. 2.5161e+05
|V|max       : 35.598 m/s   (JW jet ~35-45 m/s expected)
tracer range : 0.0000e+00 .. 0.0000e+00   (IDEAL init = 0)
ranks with problems: 0
=== ALL RANKS PHYSICALLY SANE ===
```

`|V|max = 35.6 m/s` landing exactly on the expected jet maximum, and the tracer field
identically zero to the bit, are much stronger evidence than positivity alone: they say
the solution is the *right* one, not merely a finite one. **gl12 pe256's timing run is
physically valid** despite ARM A's spurious `AUDIT FAIL`.

Two artifacts that look like results but are not:
- `testout_tmp.zarr` with `PRGout_interval=1000` is unwritten NaN fill, not a solution.
- `BUDGET_*.log` with `MNT_INTV=72` against `lstep_max` 6 or 43 fires at cstep 0 only —
  the analytic IC, finite by construction. Set `MNT_INTV=1` for a real per-step trace.

The strongest available check is unused so far: the dump exists to verify the fused
K-step scan reproduces the per-step path (`driver-dc.py:573`). Rerunning a rung PLAIN
with the same dump path and comparing via `--ref` converts "physically plausible" into
"matches the unfused path".


## 7. NCCL-FFI: sparse on the wire, dense in memory

Worth stating precisely because it is easy to get backwards.
`PYNICAM_COMM_NCCLFFI=1` swaps **only the wire transport** (`:1807-1812`): grouped
`ncclSend`/`ncclRecv` of **partner rows only**, trimmed to the used prefix
(`:1825-1849`). Traffic is `O(degree)`. The true `mpi4jax.alltoall` is only the
non-FFI `else` at `:1893`.

But `st = jnp.zeros((_nproc, a2a_chunk), jdtype)` (`:1865`) sits *above* the fork and
the FFI call declares operand and result as `(_nproc, a2a_chunk)` (`:1884`), so the
dense tensor is materialised on both paths — NCCL simply declines to fill most of it.

**Cost model: wire = O(degree), HBM = O(nproc).** `a2a_chunk` is still unmeasured; see
`TODO-extreme-scale.md` §1-2.


## Unversioned work — at risk

- **`sweep/` is not a git repository.** All ladder scripts, the generated configs, and
  `TODO-extreme-scale.md` (copied here as a sibling file) live there untracked.
- **`tools/jupiter/` is untracked in this repo** — including `PORT.md` and the `env.sh`
  that every script sources. Worth committing deliberately.
