# pyNICAM-DC Coding Policy

Durable rules distilled from the dynamical-core readability refactor. They exist to keep the
model **one readable source that runs bit-exactly on both the numpy reference and the jax/GPU
stack**, and faithful to NICAM's structure. When in doubt, prefer the rule over cleverness.

---

## 1. One source, two backends — never duplicate the science

The axis that matters is **"does the change alter STRUCTURE?"**, not "numpy vs jax". Writing an
algorithm twice (once per backend) duplicates the physics, forces permanent bit-exactness
babysitting, and gives two answers to "what does the model do". Don't.

Classify every divergence:

1. **xp-clean** — same maths, one source, runs on numpy and jax via `xp` (`bk.xp`). **This is the
   target state.** Keep.
2. **Backend-incompatible primitive** (`a[i] = x` numpy vs `a.at[i].set(x)` jax) — not
   acceleration, just incompatibility. **Fix by abstraction, not separation:**
   `a = bk.set_at(a, idx, val)` / `bk.add_at`. Always rebind the result.
3. **True acceleration** (`lax.scan` / `lax.fori_loop` / kernel fusion / device residency) —
   can't be expressed in `xp`, jax-only by nature. **Separate it, and branch only at the loop
   driver** (see §2), never in the kernel body.

- `jit` is **not** the boundary: `bk.maybe_jit` is transparent (numpy no-op, jax jit) and doesn't
  change structure, so per-kernel jit lives on the xp side. An inner `maybe_jit` under an outer
  jit/scan is inlined and fused — the same kernels serve both routes.
- The xp path's contract is the numpy / Python-Array-API surface (jax.numpy mirrors it). Keep it
  close to that surface; what the standard omits stays behind `bk.*` (in-place update, jit/scan,
  restricted fancy indexing in comm/oprt). **Do not call jax-only API on the xp/numpy path.**

**Litmus:** if you're about to write the same maths a second time for the other backend — stop and
abstract the one divergent primitive instead.

---

## 2. Stay faithful to NICAM's structure

Readability comes from removing **duplication and jargon**, not from restructuring faithful code.

- **Keep the Fortran mental map.** Don't split a monolithic method just to lower method count if
  its Fortran equivalent is monolithic (`dynamics_step` mirrors `mod_dynamics.f90` /
  `nhm_driver.f90`). Faithfulness beats method-count — check the Fortran structure before
  "improving" the Python.
- **File layout by concept, one concept → one place:**
  - `nhm/forcing/` — all artificial/idealized forcing, any lineage
  - `nhm/share/dcmip/` — self-contained idealized test physics + initial conditions
  - `nhm/physics/` — real parameterizations only (CP/MP/RD/TB/SF)
- **Data-dependent trip counts** (illegal under jit): shared body, branch only at the loop driver
  (`python for` vs `lax.fori_loop`). Template: `kessler._rain_body`.
- Comments state the **conclusion**, not the campaign/technique that reached it. No codenames.

---

## 3. State: mutable settings vs immutable prognostic value

Split the container by **mutability**:

- **Settings/services** (grid metrics, constants, subsystem objects + methods, run config): built
  at setup, read-only in the loop. **May stay a plain mutable object** — its mutability is never
  exercised in a trace, so it's harmless.
- **Prognostic state** (PROG/DIAG/PROGq + RK snapshots): **one immutable pytree value** threaded as
  `new_state = step(state)`. Never edited in place; each step returns a NEW value and rebinds
  (what `bk.set_at` does per element, for the whole state). Required because jax arrays are
  immutable and `lax.scan` threads the carry as a value, not a mutated attribute.

*"Unchanging content may be mutable; changing content must be immutable."* Resolve backend probes
(`bk.resident()/type/xp`, config-constant after setup) **once at setup** — never probe the backend
in the hot loop.

---

## 4. JIT policy

- **One mechanism: `bk.maybe_jit`.** No raw `jax.jit` in model code.
- **Scope: hot path only** — time loop, shape-stable, pure kernels. NOT setup / IO / diagnostics /
  one-shot code. Don't widen the scope.
- `static_argnames` only for genuinely static, low-cardinality arguments.
- **jax `jit` is lazy:** building the wrap does not compile — the trace/compile fires at the first
  call. To move a compile off the first step, **warm it** (call once) at setup; building the wrap
  at setup alone changes nothing.

---

## 5. Gates (`PYNICAM_*` env vars)

- **KEEP:** the fusion-scope switch (`FUSE_TIMELOOP` + timeloop params), backend/precision,
  diagnostics (`PROFILE` / `*_DUMP` / `DTYPE_AUDIT`), the `RESIDENT` escape hatch.
- **COLLAPSE, don't delete:** an always-on / validated device-vs-host selector folds into the
  `bk.resident()` / array-type master. **Collapsing the GATE ≠ deleting the PATH** — the host /
  reference branch *is* the numpy backend's implementation; it stays. (Deleting it would break
  numpy.) Bit-neutral: a default-off gate is dead, a default-on gate folds to its live arm.
- **DELETE:** one-shot debug instruments.
- A new gate needs a reason (a genuine per-run choice or a diagnostic). Default to no gate.

---

## 6. Verification bar (holds at EVERY commit)

- numpy backend **bit-exact vs the established golds**; tier1 (pytest kernels) + tier2 (CPU golden)
  green. tier3 (GPU/PBS) is the later gate, not the working loop — JAX semantics verify on CPU.
- **A/B every change** against a git-worktree baseline, on **both** backends:
  - host/numpy-path or dead-code change → **EXACTLY 0.0** (`array_equal`).
  - a change that moves a build to setup / fuses from step 0 → the **jit-vs-eager fp floor**
    (per-field rel ≤ ~5e-13). Don't accept it blind: a temporary `PYNICAM_PROVE_FLOOR` gate forcing
    the eager path in the candidate must collapse the A/B to exactly 0.0, isolating the eager→jit
    switch as the sole source.
  - comment / print-only change → **`ast.dump` equality** (byte-identical AST); no A/B needed.
- The **`FUSE_TIMELOOP` / `_step_core` path is not covered by the default A/B** — verify it with a
  dedicated fused run and assert the chunk actually engaged (don't let it pass vacuously).
- **Performance-neutral:** re-measure GPU step-time before merging a large change.
- Small, independently-verified, revertible commits; each holds these invariants.

---

## 7. Naming

Measured convention (`self.` 78%, `_x` 20%, `__x` 0%):

- `self.x` — **real state**: survives across calls, or is read by another method/object.
- local — **scratch. This is the default.** `self.` needs a reason.
- `self._x` — **internal**: caches, jits, bookkeeping — not the module's public contract.
- `__x` (dunder) — **do not use.** These are module-level singletons with no subclasses, so name
  mangling solves a non-problem.

**Test:** does the value survive between calls? yes → `self._x`, no → local.
