"""
Runtime dtype-preservation audit for the ported pure kernels.

Purpose: catch NEP50-style hidden float64 upcasts in float32 runs. We wrap every
`compute_*` entry point in the kernel modules. On each call we look at the
floating-point dtypes of the INPUTS; if the inputs are purely float32 (at least
one float32 float-array and no float64 float-array) but a floating OUTPUT comes
back as float64, that kernel is silently upcasting -> flagged.

Usage (temporary): `import dtype_audit; dtype_audit.install()` near the top of
driver-dc.py, run the model in precision="float32", then read the report written
to /tmp/dtype_audit_<rank>.txt (also printed at exit).

This module patches BOTH the kernel source modules and any already-imported
consumer modules that did `from kernels.X import compute_Y` (which copies the
reference), so install() is order-robust.
"""
from __future__ import annotations
import os
import sys
import functools

import numpy as np

# kernel module dotted paths (under the pynicamdc package, imported as
# "nhm.dynamics.kernels.X" by the running driver) and the compute_* names each
# one exports. We discover names dynamically to avoid drift.
_KERNEL_MODNAMES = [
    "advconv", "advconvmom", "bndcnd", "buoyancy", "diag", "fluxconv",
    "horizontalizevec", "oprt3ddivdamp", "oprtdivdamp", "presgrad",
    "rhogkin", "vimain", "vimatrix", "vipath1", "vipath2", "virhowsolver",
]

# (kernel_func_name) -> set of (count_calls, count_f32_input_calls, count_upcast)
_STATS: dict[str, dict] = {}
_INSTALLED = False
_RANK = 0


def _floats(obj):
    """Yield dtype of every floating ndarray reachable in obj (shallow walk over
    lists/tuples/dicts)."""
    import collections.abc as cabc
    stack = [obj]
    while stack:
        x = stack.pop()
        if x is None:
            continue
        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, np.floating):
                yield x.dtype
            continue
        # jax arrays expose .dtype and aren't ndarray
        dt = getattr(x, "dtype", None)
        if dt is not None and np.issubdtype(np.dtype(dt), np.floating):
            yield np.dtype(dt)
            continue
        if isinstance(x, (list, tuple)):
            stack.extend(x)
        elif isinstance(x, cabc.Mapping):
            stack.extend(x.values())


def _make_wrapper(name, fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        in_dtypes = list(_floats(args)) + list(_floats(kwargs))
        out = fn(*args, **kwargs)
        st = _STATS.setdefault(
            name, {"calls": 0, "f32_calls": 0, "upcast": 0,
                   "f64in": 0})
        st["calls"] += 1
        has_f32 = any(d == np.float32 for d in in_dtypes)
        has_f64_in = any(d == np.float64 for d in in_dtypes)
        if has_f64_in:
            st["f64in"] += 1
        # Real upcast signature: float32 state goes in, a float64 floating array
        # comes out. We flag this even when a float64 *constant* input (e.g. CVW)
        # is present, because in a float32 run every model-state output should
        # stay float32.
        if has_f32:
            st["f32_calls"] += 1
            out_dtypes = list(_floats(out))
            if any(d == np.float64 for d in out_dtypes):
                st["upcast"] += 1
        return out
    wrapped.__wrapped_audit__ = True
    return wrapped


def install():
    global _INSTALLED, _RANK
    if _INSTALLED:
        return
    _INSTALLED = True
    try:
        from mpi4py import MPI
        _RANK = MPI.COMM_WORLD.Get_rank()
    except Exception:
        _RANK = 0

    # locate the loaded kernel modules regardless of the package prefix used.
    targets = {}  # short_name -> module object
    for modname, mod in list(sys.modules.items()):
        if mod is None:
            continue
        short = modname.rsplit(".", 1)[-1]
        if short in _KERNEL_MODNAMES and modname.endswith("kernels." + short):
            targets[short] = mod

    # map original func -> wrapper, so we can repoint consumer modules too.
    repoint = {}
    for short, mod in targets.items():
        for attr in dir(mod):
            if not attr.startswith("compute_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn) or getattr(fn, "__wrapped_audit__", False):
                continue
            w = _make_wrapper(attr, fn)
            repoint[fn] = w
            setattr(mod, attr, w)

    # repoint any consumer module that did `from kernels.X import compute_Y`.
    for modname, mod in list(sys.modules.items()):
        if mod is None or modname.endswith("dtype_audit"):
            continue
        try:
            members = vars(mod)
        except TypeError:
            continue
        for attr, val in list(members.items()):
            if not callable(val):
                continue
            try:
                w = repoint.get(val)
            except TypeError:
                continue  # unhashable attribute value
            if w is not None:
                setattr(mod, attr, w)

    print(f"[dtype_audit] installed; wrapped {len(repoint)} compute_* fns "
          f"across {len(targets)} kernel modules", flush=True)


def report():
    rank = _RANK
    lines = ["=== dtype audit report (rank %d) ===" % rank,
             "%-34s %8s %8s %8s %8s" % (
                 "kernel", "calls", "f32call", "f64in", "UPCAST")]
    flagged = []
    for name in sorted(_STATS):
        st = _STATS[name]
        flag = "  <-- UPCAST" if st["upcast"] else ""
        if st["upcast"]:
            flagged.append(name)
        lines.append("%-34s %8d %8d %8d %8d%s" % (
            name, st["calls"], st["f32_calls"], st["f64in"],
            st["upcast"], flag))
    if flagged:
        lines.append("FLAGGED (float64 output on float32 input): "
                     + ", ".join(flagged))
    else:
        lines.append("OK: no float64 upcast detected on any float32 call")
    txt = "\n".join(lines)
    path = "/tmp/dtype_audit_%d.txt" % rank
    try:
        with open(path, "w") as f:
            f.write(txt + "\n")
    except Exception:
        pass
    print("\n" + txt + "\n", flush=True)
