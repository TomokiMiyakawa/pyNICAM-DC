#!/usr/bin/env python3
"""Compare two sets of per-rank PRG_var dumps (fin_rank*.npy) on the interior.

Peak-normalized per variable (max|a-b| / max|ref| over all ranks), as in
tutorial/check_validation.py. rocm_gl05_kit/validate.py divides per cell with a
1e-12 floor, so a ~1e-13 halo-corner value turns a 1e-14 difference into
max_rel ~1e-2 and reports FAIL on results that agree to 1e-12.

  cmp_ranks.py REF_DIR CAND_DIR [--rtol 1e-9]     # --rtol 0 = bit-identical required
Exit 0 = candidate finite, RHOG > 0, and worst peak-rel <= rtol.
"""
import argparse, glob, os, sys
import numpy as np

NAMES = ["RHOG", "RHOGVX", "RHOGVY", "RHOGVZ", "RHOGW", "RHOGE"]


def load(d):
    return {os.path.basename(f)[len("fin_rank"):-4]: np.load(f)
            for f in sorted(glob.glob(os.path.join(d, "fin_rank*.npy")))}


def interior(a):
    k = a.shape[2] - 2
    return a[1:-1, 1:-1, 1:k + 1, :, :len(NAMES)].astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref"); p.add_argument("cand")
    p.add_argument("--rtol", type=float, default=1e-9)
    a = p.parse_args()
    R, C = load(a.ref), load(a.cand)
    if not R or set(R) != set(C):
        print(f"FAIL: ranks ref={sorted(R)} cand={sorted(C)}"); sys.exit(2)
    nonfinite = sum(int((~np.isfinite(interior(C[r]))).sum()) for r in R)
    rhog_min = min(float(interior(C[r])[..., 0].min()) for r in R)
    peak = np.max([np.abs(interior(R[r])).max(axis=(0, 1, 2, 3)) for r in R], axis=0)
    dmax = np.max([np.abs(interior(C[r]) - interior(R[r])).max(axis=(0, 1, 2, 3)) for r in R], axis=0)
    rel = np.where(peak > 0, dmax / np.where(peak > 0, peak, 1), 0.0)
    bit = all(np.array_equal(R[r], C[r], equal_nan=False) for r in R)
    print(f"  {a.cand} vs {a.ref}  ({len(R)} ranks)")
    print(f"    nonfinite={nonfinite}  RHOG_min={rhog_min:.3e}")
    for n, d, r in zip(NAMES, dmax, rel):
        print(f"    {n:7s} max|d|={d:.3e}  peak-rel={r:.3e}")
    worst = float(rel.max())
    ok = nonfinite == 0 and rhog_min > 0 and worst <= a.rtol
    print(f"    worst peak-rel={worst:.3e}  bit-identical={bit}  -> {'PASS' if ok else 'FAIL'} (rtol={a.rtol:g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
