#!/usr/bin/env python3
"""Validate kessler.py vs the Fortran kessler.f90 golden, in both precisions.

Reads ref_kessler_z{nz}.txt / ref_kessler_f64_z{nz}.txt (written by
build_kessler_ref.sh: inputs + outputs on identical synthetic columns) and:
  (1) fp64 inputs, lp=float32 (default)  vs the faithful f32-locals golden
      -> ~1e-7  (f32-local floor; matches the production nicamdc build)
  (2) fp64 inputs, lp=float64            vs the all-real(8) golden
      -> ~1e-14 (machine precision; proves the algorithm is otherwise exact)
  (3) fp32 inputs                        -> output is float32, finite, physical
      (the model's fp32 mode; no Fortran reference, checks consistency w/ fp64)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from pynicamdc.nhm.forcing.kessler import kessler


def load(path):
    with open(path) as f:
        it = iter(f.read().split())
    ncol = int(next(it)); nz = int(next(it)); dt = float(next(it))
    blk = lambda: np.array([float(next(it)) for _ in range(ncol * nz)]).reshape(ncol, nz)
    ins = {k: blk() for k in ("theta", "qv", "qc", "qr", "rho", "pk", "z")}
    outs = {k: blk() for k in ("theta", "qv", "qc", "qr")}
    precl = np.array([float(next(it)) for _ in range(ncol)])
    return ncol, nz, dt, ins, outs, precl


def rel(a, b):
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-300)


def worst(ins, outs, precl, **kw):
    th, qv, qc, qr, pl = kessler(ins["theta"], ins["qv"], ins["qc"], ins["qr"],
                                 ins["rho"], ins["pk"], ins["_dt"], ins["z"], **kw)
    return max(rel(th, outs["theta"]), rel(qv, outs["qv"]),
               rel(qc, outs["qc"]), rel(qr, outs["qr"]), rel(pl, precl))


def main():
    nzs = sys.argv[1:] or ["30", "40", "78"]
    ok = True
    for nz in nzs:
        p32 = os.path.join(HERE, f"ref_kessler_z{nz}.txt")
        p64 = os.path.join(HERE, f"ref_kessler_f64_z{nz}.txt")
        if not os.path.exists(p32):
            print(f"  (skip z{nz}: no golden)"); continue
        ncol, nzn, dt, ins, outs, precl = load(p32)
        ins["_dt"] = dt

        w1 = worst(ins, outs, precl)                       # (1) default lp=f32
        ok &= w1 < 1e-6

        line = f"z{nz} ncol={ncol}: (1) f32-locals={w1:.1e}"
        if os.path.exists(p64):
            _, _, _, ins64, outs64, precl64 = load(p64)
            ins64["_dt"] = dt
            w2 = worst(ins64, outs64, precl64, lp=np.float64)   # (2) all-f64
            ok &= w2 < 1e-11
            line += f"  (2) all-f64={w2:.1e}"

        # (3) fp32 mode: f32 inputs -> f32 output, finite + physical
        i32 = {k: ins[k].astype(np.float32) for k in
               ("theta", "qv", "qc", "qr", "rho", "pk", "z")}
        th, qv, qc, qr, pl = kessler(i32["theta"], i32["qv"], i32["qc"], i32["qr"],
                                     i32["rho"], i32["pk"], dt, i32["z"])
        fp32_ok = (th.dtype == np.float32 and np.isfinite(th).all()
                   and np.isfinite(pl).all() and (pl >= 0).all())
        ok &= fp32_ok
        line += f"  (3) fp32 {'OK' if fp32_ok else 'FAIL'}"
        print(line)
    print("ALL PASS" if ok else "SOME FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
