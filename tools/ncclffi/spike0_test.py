#!/usr/bin/env python
"""N0 spike test (nccl-ffi-plan_v1.txt): register the trivial CUDA FFI handler and
exercise it exactly the ways the real transport will be used:
  T1 eager ffi_call            (y = 2x)
  T2 under jax.jit
  T3 inside lax.scan under jit (the FUSE_TIMELOOP shape)
  T4 has_side_effect=True under jit+scan (the transport's actual form)
PASS = all four bit-correct. Single process, 1 GPU."""
import ctypes
import os
import sys

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

HERE = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.cdll.LoadLibrary(os.path.join(HERE, "libspike0.so"))
jax.ffi.register_ffi_target("spike0", jax.ffi.pycapsule(lib.Spike0), platform="CUDA")

N = 1_000_003  # deliberately not a multiple of the block size
x_h = np.arange(N, dtype=np.float32) * 0.5
x = jnp.asarray(x_h)
want = x_h * 2.0


def call(v, side_effect=False):
    return jax.ffi.ffi_call("spike0", jax.ShapeDtypeStruct(v.shape, v.dtype),
                            has_side_effect=side_effect)(v)


def check(tag, got, ref):
    ok = np.array_equal(np.asarray(got), ref)
    print(f"[spike0] {tag}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


results = []

# T1 eager
results.append(check("T1 eager", call(x), want))

# T2 jit
results.append(check("T2 jit", jax.jit(call)(x), want))

# T3 scan under jit: y -> 2y -> *0.5 per step, 10 steps => identity
def scan_body(c, _):
    return call(c) * 0.5, ()

y3, _ = jax.jit(lambda v: lax.scan(scan_body, v, None, length=10))(x)
results.append(check("T3 jit+scan", y3, x_h))

# T4 side-effecting form under jit+scan (what the NCCL transport will use)
def scan_body_se(c, _):
    return call(c, side_effect=True) * 0.5, ()

y4, _ = jax.jit(lambda v: lax.scan(scan_body_se, v, None, length=10))(x)
results.append(check("T4 jit+scan+side_effect", y4, x_h))

print(f"[spike0] {'ALL PASS' if all(results) else 'SOME FAILED'}", flush=True)
sys.exit(0 if all(results) else 1)
