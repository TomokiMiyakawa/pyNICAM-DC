#!/usr/bin/env python3
"""Tutorial validation checker for pyNICAM-DC.

Given a per-rank prognostic dump (PYNICAM_TIMELOOP_DUMP output, an (i,j,k,l,nvar)
.npy of PRG_var after the run), it:
  1. checks PHYSICAL SANITY on the interior  -- all finite, density RHOG > 0,
     total energy RHOGE > 0 (a broken run shows NaN / negative mass);
  2. optionally COMPARES to a reference dump (the bundled golden, or a numpy run
     when checking the jax backend) and reports the max relative difference,
     PASS if within --rtol.

Usage:
  check_validation.py RUN.npy                       # sanity only
  check_validation.py RUN.npy --ref GOLDEN.npy      # sanity + golden compare
  check_validation.py JAX.npy  --ref NUMPY.npy --rtol 1e-5 --label "jax vs numpy"

Exit code 0 = all checks pass, 1 = a check failed (usable in CI / scripts).
"""
import argparse
import sys
import numpy as np

# PRG_var variable order (first 6 are always present; tracers follow)
NAMES = ["RHOG", "RHOGVX", "RHOGVY", "RHOGVZ", "RHOGW", "RHOGE"]
I_RHOG, I_RHOGE = 0, 5


def _interior(a):
    """Strip the 1-cell halo in i,j and the top/bottom ghost levels in k."""
    kmax = a.shape[2] - 2
    return a[1:-1, 1:-1, 1:kmax + 1, :, :]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--ref", default=None, help="reference dump to compare against")
    p.add_argument("--rtol", type=float, default=1e-6,
                   help="relative tolerance for the reference compare (default 1e-6)")
    p.add_argument("--label", default="run")
    args = p.parse_args()

    a = np.load(args.run)
    interior = _interior(a)
    ok = True

    print(f"=== {args.label}: {args.run}  shape={a.shape} ===")

    # --- 1. physical sanity ---
    finite = np.isfinite(interior).all()
    rhog = interior[..., I_RHOG]
    rhoge = interior[..., I_RHOGE]
    rhog_ok = bool((rhog > 0).all())
    rhoge_ok = bool((rhoge > 0).all())
    print(f"  [sanity] all finite         : {'OK' if finite else 'FAIL'}")
    print(f"  [sanity] RHOG  > 0 (density): {'OK' if rhog_ok else 'FAIL'}  "
          f"(range {rhog.min():.3e} .. {rhog.max():.3e})")
    print(f"  [sanity] RHOGE > 0 (energy) : {'OK' if rhoge_ok else 'FAIL'}  "
          f"(range {rhoge.min():.3e} .. {rhoge.max():.3e})")
    ok = ok and finite and rhog_ok and rhoge_ok

    # --- 2. reference compare ---
    if args.ref:
        b = _interior(np.load(args.ref))
        if b.shape != interior.shape:
            print(f"  [compare] SHAPE MISMATCH {interior.shape} vs {b.shape} -> FAIL")
            sys.exit(1)
        worst = 0.0
        print(f"  [compare] vs {args.ref} (rtol={args.rtol:g}):")
        for v, name in enumerate(NAMES):
            x = interior[..., v].ravel(); y = b[..., v].ravel()
            d = np.abs(x - y)
            m = np.abs(y).max()
            big = np.abs(y) >= 1e-3 * m if m > 0 else np.zeros_like(y, bool)
            rel = float((d[big] / np.abs(y[big])).max()) if big.any() else 0.0
            worst = max(worst, rel)
            print(f"      {name:7s} max|d|={d.max():.3e}  maxrel={rel:.3e}")
        passed = worst <= args.rtol
        print(f"  [compare] worst maxrel = {worst:.3e}  -> {'PASS' if passed else 'FAIL'}")
        ok = ok and passed

    print(f"=== {'PASS' if ok else 'FAIL'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
