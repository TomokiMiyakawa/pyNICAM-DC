# TODO — extreme-scale readiness (rlevel 6, gl15/gl16, up to 5120 nodes / 20480 GPUs)

Written 2026-07-29. Nothing here is urgent for the gl12/gl13 rungs; it is the list of
things that must be settled *before* committing 5120 nodes. Line numbers are against
`pyNICAM-DC/pynicamdc/share/mod_comm.py` as of this date.

Target config assumed below: **rlevel 6, glevel 15 or 16, pe = 20480** (5120 nodes x 4).
`10 * 4^6 = 40960` regions / 20480 ranks = **2 regions per rank**.


## 1. Instrument `a2a_chunk` (do this first — it gates item 2)

`a2a_chunk` (`:1754`) is the global max per-partner payload, an `MPI.MAX` allreduce over
all ranks. **It is not logged anywhere.** Without it, item 2 cannot be sized.

Add it to the existing `[COMM_DEGREE]` report (`:159-171`, gated on `PYNICAM_COMM_DEGREE=1`).
That block currently runs at the end of `COMM_setup` and prints partner degrees; `a2a_chunk`
is computed later, in the device-plan builder, so it either needs a second print at plan
build time or the value stashed on `self` and printed by an existing hook.

Also worth printing: `_nproc * a2a_chunk * itemsize` — the actual dense-tensor byte count,
which is the number that decides item 2.

Capture it at **gl12 pe256 and gl13 pe1024** so the extrapolation to pe20480 has two points
rather than one. Both rungs are cheap relative to a 5120-node run.

**Do not edit `mod_comm.py` while a job is queued or running** — the job imports it at
launch, so an edit lands mid-flight and silently changes what the run measures. This
already caused one false diagnosis on 2026-07-29 (a stale log, not a stale import, but the
same class of mistake).


## 2. The dense `(nproc, a2a_chunk)` halo buffer — the real extreme-scale risk

**The wire is fine.** With `PYNICAM_COMM_NCCLFFI=1` the transport is grouped
`ncclSend`/`ncclRecv` over **partner rows only**, further trimmed to the used contiguous
prefix (`_sp_len`/`_rp_len`, `:1825-1849`). Traffic is `O(degree)`, not `O(nproc)`. The
true `mpi4jax.alltoall` is only the non-FFI `else` at `:1893`. This was checked directly in
the code — see the comment at `:1807-1812`, which says so explicitly.

**The buffer is not fine.** `:1865` allocates

```python
st = jnp.zeros((_nproc, a2a_chunk), jdtype)
```

*above* the FFI/mpi4jax fork, so it exists on both paths, and the FFI call declares operand
and result as `(_nproc, a2a_chunk)` (`:1884`) with peer offsets `p * a2a_chunk` (`:1846`,
`:1848`). Two dense device tensors are materialized per exchange whatever the transport;
XLA cannot elide them because they are FFI operands of declared shape. NCCL simply declines
to *fill* the non-partner rows (they are left uninitialized by design).

So the cost model splits: **wire = O(degree), HBM = O(nproc).**

Weak-scaling estimate for whether this bites:

| config | rgn/rank | halo pts/rank | partners | per-partner payload |
|---|---|---|---|---|
| gl12 rl05 pe256 | 40 | ~20.6k (`rellist_nmax` 20638) | 34 measured | ~607 pts |
| gl16 rl06 pe20480 | 2 | ~8.2k (2 x 4x1026) | <=16 | ~513 pts |

Per-partner payload is roughly scale-invariant (~15% smaller), so `a2a_chunk` stays about
constant while `_nproc` grows 80x. **The dense tensor therefore grows ~80x from pe256 to
pe20480.** Tens of MB at gl12 is survivable; ~1 GB is not, against 96 GB HBM shared with
the prognostic state.

If item 1 shows this is too big, the fix is a **peer-indexed compact buffer**: allocate
`(len(_peers), a2a_chunk)` instead of `(_nproc, a2a_chunk)` and map `dst -> peer_index`
rather than `dst -> rank_id`. The FFI plan already carries `_peers` and per-peer offsets,
so `register_plan` needs offsets `i * a2a_chunk` for `i, p in enumerate(_peers)` instead of
`p * a2a_chunk`. The pack/unpack loops (`:1871-1874`, `:1896-1897`) index by `dst`/`src`
and would need the same remap. Note this **breaks the "byte-identical to alltoall" property**
the current design leans on, so the non-FFI `mpi4jax.alltoall` path must keep the dense
layout — i.e. the compact layout is FFI-only, and the two paths stop being bit-comparable
by construction. That trade needs a decision, not just an edit.


## 3. `Recv_nlim` / `Send_nlim` — settled, no action needed

Raised 20 -> 64 on 2026-07-29 (`:68-69`) after gl12 pe256 aborted in `_check_commnlim`
(`:97`) with rank 76 needing 21.

**64 is sufficient for rlevel 6 at 20480 ranks — no further bump required.** The ceiling is
geometric: a quad region has 4 edge + 4 corner neighbours, so

```
max partner ranks <= 8 * regions_per_rank ,   regions_per_rank = 10 * 4^rl / pe
```

At 2 regions/rank that is a hard ceiling of **16**, whatever the mnginfo assignment does.
`nlim` is **independent of glevel** — glevel sets region size, rlevel and pe set region
count. Scaling ranks *up* makes `nlim` safer; the risk is coarse decompositions.

Measured / predicted degrees:

| config | rgn/rank | predicted (mnginfo, edge-only) | measured `[COMM_DEGREE]` |
|---|---|---|---|
| rl05 pe256 | 40 | 32 | **r2r=34, pole=20, total=36** |
| rl05 pe1024 | 10 | 16 | not yet run |
| rl06 pe20480 | 2 | <=16 (hard bound) | — |

The offline prediction ran ~6% low (32 vs 34), so apply a safety factor when using it.

**How to predict a new decomposition without running it:** parse the mnginfo TOML —
`[RGN_MNG_INFO.*]` blocks give `PEID` + `MNG_RGNID` (region -> rank), `[RGN_LINK_INFO.*]`
blocks give `SW`/`NW`/`NE`/`SE` (region -> edge neighbours). Union the neighbours of each
rank's regions, map to owners, drop self, take the max. A second hop over that set
approximates the corner neighbours and overcounts (38 vs 34 measured at pe256), so the true
value sits between edge-only and 2-hop. Beware: `NUM_OF_MNG` appears both in the global
`[PROC_INFO]` block and per-rank — anchor the regex on `[PROC_INFO]`.


## 3b. Fused-executable constant capture scales with REGIONS, not cells

Found 2026-07-29 by gl12 pe256 ARM A (job 1087535): fp64 + `FUSE_TIMELOOP=1` +
`TIMELOOP_CHUNK=1` + `NCCLFFI_CKSUM=1` + `TIMELOOP_DUMP` OOM'd on all 256 ranks with
`CUDA_ERROR_OUT_OF_MEMORY [executable_name='jit__step_fn']`, preceded by a JAX
`A large amount of constants were captured during lowering (2.95GB total)` warning.

gl09 pe4 and gl11 pe64 pass the SAME fp64 fused audit. The discriminator is regions per
rank at fixed 655,360 cells/GPU: gl09/gl11/gl13-pe1024 all have **10**, gl12 pe256 has
**40**, and `rellist_nmax` follows (10,270 / 10,278 / ~10,320 vs **20,638**). The fused
executable bakes in per-region and per-halo-relation index arrays, so its constant
footprint tracks region and relation count, NOT cell count.

Implication for rl06 / pe20480: 2 regions per rank is the LOWEST of any configuration run
so far, so constant capture should be the least of the worries there. The scaling hazard
at that end is item 2 (dense buffer, O(nproc)), not this one. Do not conflate them.

Mitigations if a fused fp64 audit is ever needed at high regions/rank: run the audit at
fp32 (it is a byte-level send-vs-recv checksum compare; fp64 buys nothing), drop
`TIMELOOP_DUMP`, or raise `TIMELOOP_CHUNK` above 1.

Also: `TIMELOOP_CHUNK` should divide (lstep_max - WARMUP) exactly. 43 - 3 = 40 with
CHUNK=6 leaves a ragged `K=4` chunk that JIT-compiles a SECOND time (~16-24 s/rank).
gl11's CHUNK=4 divided evenly and compiled once.


## 4. Housekeeping for the next rung

- ~~`jupiter_gl13_pe1024_smoke.sbatch` lacks `PYNICAM_COMM_DEGREE=1` and the tail greps~~
  DONE 2026-07-29, submitted as job 1087565. Both gl12 and gl13 smokes now report degree.
- No rlevel-6 mnginfo exists yet (`kit/mnginfo/` holds only `rl05-prc000256.toml` and
  `rl05-prc001024.toml`). Generating `rl06-prc020480.toml` is a prerequisite for any of the
  above being verifiable offline.
- The `sbatch` scripts exit 0 even when `srun` fails, because the run is inside a subshell
  whose status is not propagated. **Do not trust the Slurm exit code** — job 1087502 showed
  `COMPLETED 0:0` while achieving `ranks done: 0/256`. Read the `ranks done=` line instead.
  Worth fixing properly if these scripts get reused at 5120 nodes, where a silent failure
  is expensive.
