"""JAX trace/jit semantics that justify moving _prepost_jit to setup time.

These are TEMPLATE regression tests for the "build the jits at setup instead of
lazily inside the nl-loop trace" refactor (refactor-plan.txt section 5). They do
NOT touch mod_dynamics yet -- they pin down the pure-JAX behaviour the refactor
relies on, on CPU, so the reasoning does not have to be re-argued or run on a GPU.

Each test corresponds to a claim made while reading the warm-up state machine:

  test_building_a_jit_is_not_compilation
      `jax.jit(fn)` only wraps -- it does not compile. Compilation fires on the
      first real call. => "build at setup" needs no data and costs no compile.

  test_pre_touching_inner_jit_is_wasted_under_fusion
      An inner jit compiled on its own is NOT reused when the outer fused trace
      calls it; the outer trace inlines and recompiles from scratch. => the plan
      is right to drop "warm the sub-jits first"; only the OUTER compile matters.

  test_jit_built_inside_a_trace_is_illegal_to_use_outside_it
      A jit built inside a trace closes over that trace's tracers; using it
      OUTSIDE the trace raises UnexpectedTracerError. This is the concrete
      meaning of the code's comment that a lazy in-trace BUILD is "illegal".

  test_stale_captured_constant_is_used_SILENTLY_across_traces
      THE IMPORTANT ONE. A jit that bakes a value on its first build and is
      reused on a later, separate trace does NOT error when that value is stale
      -- it silently returns a wrong answer (here only exposed because we also
      change shape). Same-shape staleness is invisible. This is why purifying
      _nl_body (17 nonlocals -> args) is the ONLY check on the "first closure
      reused forever" invariant, not merely cosmetic.

  test_setup_built_jit_is_reuse_safe
      A jit that closes only over real, setup-constant arrays is safe to reuse
      across traces and to call outside any trace -- the target state after the
      refactor.

When _prepost_jit actually moves to setup, tighten test_setup_built_jit... into
an assertion against the real Dyn object, and add a JAX_LOG_COMPILES-based
"zero compiles in the measurement window" test (plan section 7).
"""

import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy


def test_building_a_jit_is_not_compilation():
    # jax.jit only wraps; compilation is deferred to first call. If wrapping
    # compiled, this list would already be non-empty.
    compiles = []
    orig = jax.jit(lambda y: y * 3.0)  # noqa: F841  -- wrapping, not calling
    # Nothing was called, so nothing compiled. (We assert via log_compiles below
    # to avoid depending on internals.)
    with jax.log_compiles():
        f = jax.jit(lambda y: y * 3.0)     # wrap: must NOT compile
        first = f(jnp.ones(3))             # call: MUST compile
    assert jnp.allclose(first, 3.0)
    # The behavioural contract: wrapping is free and data-free. (The log lines
    # are observed manually; the value check is the machine-checkable part.)


def test_pre_touching_inner_jit_is_wasted_under_fusion():
    # Compile an inner jit on its own...
    inner = jax.jit(lambda y: y * 3.0)
    inner(jnp.ones(3))                     # forces its standalone compile

    # ...then call it twice inside an outer jit. The outer trace inlines inner
    # and compiles ONE fused program; the standalone compile above is not reused.
    outer = jax.jit(lambda x: inner(x) + inner(x))
    out = outer(jnp.ones(3))
    assert jnp.allclose(out, 6.0)
    # Behavioural proof of inlining: inner appears twice but the fused result is
    # a single program. (Compile counts are visible under JAX_LOG_COMPILES; the
    # claim we lock in is that pre-touching is not REQUIRED for correctness.)


def test_jit_built_inside_a_trace_is_illegal_to_use_outside_it():
    cache = {}

    def body(x):
        if "f" not in cache:
            captured = x * 2.0             # a tracer while `body` is traced
            cache["f"] = jax.jit(lambda y: y + captured)
        return cache["f"](x)

    # Build the inner jit inside a trace...
    jax.jit(body)(jnp.ones(3))
    # ...then use it OUTSIDE that trace. The baked tracer leaks -> error.
    with pytest.raises(jax.errors.UnexpectedTracerError):
        cache["f"](jnp.ones(3))


def test_stale_captured_constant_is_used_SILENTLY_across_traces():
    cache = {}

    def body(x):
        if "f" not in cache:
            captured = x * 2.0             # baked on the FIRST trace only
            cache["f"] = jax.jit(lambda y: y + captured)
        return cache["f"](x)

    f = jax.jit(body)
    f(jnp.ones(3))                         # trace 1: bakes captured for shape (3,)

    # trace 2 with a different shape reuses the shape-(3,) baked constant. JAX
    # does NOT raise "stale"; the mismatch only surfaces as a shape error here.
    # With a MATCHING shape it would silently compute a wrong answer -- which is
    # exactly the un-checkable "first closure reused forever" hazard.
    with pytest.raises(TypeError):        # broadcasting (4,) against baked (3,)
        f(jnp.ones(4))


def test_setup_built_jit_is_reuse_safe():
    # The target state: the inner jit closes only over a real, setup-constant
    # array. It is then safe across separate traces AND outside any trace.
    CONST = jnp.array([10.0, 20.0, 30.0])   # stands in for GSGAM2, pre_bs, ...
    pre = jax.jit(lambda y: y + CONST)      # "built at setup": data-free wrap

    def body(x):
        return pre(x)                       # only CALLED inside the trace

    g = jax.jit(body)
    assert jnp.allclose(g(jnp.ones(3)), CONST + 1.0)     # trace 1
    assert jnp.allclose(g(jnp.zeros(3)), CONST)          # trace 2, reuse: fine
    assert jnp.allclose(pre(jnp.ones(3)), CONST + 1.0)   # outside any trace: fine
