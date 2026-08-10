#!/usr/bin/env python3
"""Compare two sets of per-rank PRG_var dumps (fin_rank*.npy) and report the
max abs/rel difference. Used to check jax-ROCm against the numpy CPU reference on
the SAME box (identical IDEAL initial conditions), and to A/B the RCCL vs mpi4jax
comm paths.

  python validate.py --ref 'run_1gpu_numpy/fin_rank*.npy' \
                     --cand 'run_1gpu_jax/fin_rank*.npy' --rtol 1e-9

Cross-backend (numpy-CPU vs jax-ROCm) diffs sit at the libm/reduction-order floor
(~1e-11..1e-9), NOT zero -- same lesson as the ARM-vs-x86 gold floor. RCCL-vs-
mpi4jax (same backend, only the wire differs) SHOULD be bit-identical (rtol 0).
"""
import argparse, glob, os, sys
import numpy as np


def by_rank(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        r = os.path.basename(f).split("rank")[1].split(".npy")[0]
        d[r] = np.load(f)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12)
    a = ap.parse_args()

    R, C = by_rank(a.ref), by_rank(a.cand)
    if not R or not C:
        print(f"FAIL: empty set (ref={list(R)} cand={list(C)})"); sys.exit(2)
    if set(R) != set(C):
        print(f"FAIL: rank mismatch ref={sorted(R)} cand={sorted(C)}"); sys.exit(2)

    worst_abs = worst_rel = 0.0
    nfin_bad = 0
    for r in sorted(R):
        x, y = R[r].astype(np.float64), C[r].astype(np.float64)
        if x.shape != y.shape:
            print(f"FAIL rank{r}: shape {x.shape} vs {y.shape}"); sys.exit(2)
        m = np.isfinite(x) & np.isfinite(y)
        nfin_bad += int((~np.isfinite(y)).sum())
        da = np.abs(x - y)
        dr = da / np.maximum(np.abs(x), a.atol)
        wa, wr = float(np.nanmax(da[m])) if m.any() else np.inf, float(np.nanmax(dr[m])) if m.any() else np.inf
        worst_abs, worst_rel = max(worst_abs, wa), max(worst_rel, wr)
        print(f"  rank{r}: max|d|={wa:.3e}  max_rel={wr:.3e}  cand_nonfinite={(~np.isfinite(y)).sum()}")

    print(f"\nWORST over {len(R)} ranks: max|d|={worst_abs:.3e}  max_rel={worst_rel:.3e}  "
          f"cand_nonfinite={nfin_bad}")
    ok = (worst_rel <= a.rtol) and (nfin_bad == 0)
    print(f"RESULT: {'PASS' if ok else 'FAIL'} (rtol={a.rtol:g})"
          + ("  [BIT-IDENTICAL]" if worst_abs == 0.0 else ""))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
