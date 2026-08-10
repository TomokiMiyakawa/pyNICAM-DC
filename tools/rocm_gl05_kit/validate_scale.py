#!/usr/bin/env python3
"""Scale-relative companion to validate.py.

validate.py judges elementwise with max(|ref|, atol) in the denominator, which lets a
single element that is numerically zero against its OWN field's scale dominate max_rel
and produce a false FAIL -- already measured on MI300X (PORT.md Stage 1: max_rel
1.17e-02 came from ref=3.80e-13 vs cand=3.92e-13 on a field whose RMS is 5.47).

Here each prognostic variable (last axis of the PRG_var dump) is judged against its own
RMS: max|d| / rms(ref), pooled over ranks. That is the number that says whether the halo
exchange is right.

  python validate_scale.py --ref 'run_8cpu_numpy/fin_rank*.npy' \
                           --cand 'run_8gpu_rccl/fin_rank*.npy' --tol 1e-9
"""
import argparse, glob, os, sys
import numpy as np

VNAME = ["rhog", "rhogvx", "rhogvy", "rhogvz", "rhogw", "rhoge", "rhogq"]


def by_rank(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        r = int(os.path.basename(f).split("rank")[1].split(".npy")[0])
        d[r] = np.load(f)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="max allowed max|d|/rms per variable")
    a = ap.parse_args()

    R, C = by_rank(a.ref), by_rank(a.cand)
    if not R or not C or set(R) != set(C):
        print(f"FAIL: rank sets ref={sorted(R)} cand={sorted(C)}"); sys.exit(2)

    nv = next(iter(R.values())).shape[-1]
    num = np.zeros(nv)          # max |d| per variable, over all ranks
    sq = np.zeros(nv); cnt = 0  # pooled sum of ref^2 -> rms
    nonfinite = 0
    for r in sorted(R):
        x, y = R[r].astype(np.float64), C[r].astype(np.float64)
        if x.shape != y.shape:
            print(f"FAIL rank{r}: shape {x.shape} vs {y.shape}"); sys.exit(2)
        nonfinite += int((~np.isfinite(y)).sum())
        f = x.reshape(-1, nv); g = y.reshape(-1, nv)
        num = np.maximum(num, np.abs(f - g).max(axis=0))
        sq += (f ** 2).sum(axis=0); cnt += f.shape[0]
    rms = np.sqrt(sq / cnt)

    print(f"{len(R)} ranks, {nv} prognostic variables, tol={a.tol:g} (max|d|/rms)")
    worst = 0.0
    for v in range(nv):
        name = VNAME[v] if v < len(VNAME) else f"v{v}"
        if rms[v] == 0.0:
            # a field that is identically zero in the reference: bit-equality is the
            # only meaningful test, and a scale-relative ratio is undefined.
            verdict = "bit-identical" if num[v] == 0.0 else f"NONZERO DIFF {num[v]:.3e}"
            print(f"  v{v} {name:<7} rms=0            max|d|={num[v]:.3e}   {verdict}")
            if num[v] != 0.0:
                worst = np.inf
            continue
        rel = num[v] / rms[v]
        worst = max(worst, rel)
        print(f"  v{v} {name:<7} rms={rms[v]:.4e}  max|d|={num[v]:.3e}  "
              f"max|d|/rms={rel:.2e}" + ("   [bit-identical]" if num[v] == 0.0 else ""))

    print(f"\nWORST scale-relative: {worst:.3e}   cand_nonfinite={nonfinite}")
    ok = worst <= a.tol and nonfinite == 0
    print(f"RESULT: {'PASS' if ok else 'FAIL'} (tol={a.tol:g})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
