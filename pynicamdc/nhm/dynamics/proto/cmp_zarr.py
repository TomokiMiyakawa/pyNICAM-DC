#!/usr/bin/env python
"""Bit-exact comparison of two zarr prognostics dumps.

Usage: python cmp_zarr.py <zarr_a> <zarr_b>

Exit 0 if every shared array is bit-identical; exit 1 otherwise. Used to
validate that on-device COMM (PYNICAM_ONDEVICE_COMM=1) produces prognostics
*bit-identical* to the numpy-COMM path -- COMM is pure data movement, so it
must (GPU rounding does not enter; no arithmetic happens in COMM).
"""
import sys
import numpy as np
import zarr

a_path, b_path = sys.argv[1], sys.argv[2]
A = zarr.open(a_path, mode="r")
B = zarr.open(b_path, mode="r")

keys_a = sorted(k for k in A.array_keys())
keys_b = sorted(k for k in B.array_keys())
print(f"A={a_path}\nB={b_path}")
print(f"arrays A: {keys_a}")
print(f"arrays B: {keys_b}")

shared = [k for k in keys_a if k in keys_b]
ok = True
if keys_a != keys_b:
    print(f"!! array-key sets differ (A-only={set(keys_a)-set(keys_b)}, "
          f"B-only={set(keys_b)-set(keys_a)})")
    ok = False

for k in shared:
    xa = np.asarray(A[k]); xb = np.asarray(B[k])
    if xa.shape != xb.shape:
        print(f"  {k:12s} SHAPE DIFF {xa.shape} vs {xb.shape}")
        ok = False
        continue
    identical = np.array_equal(xa, xb, equal_nan=True)
    if identical:
        print(f"  {k:12s} BIT-IDENTICAL  shape={xa.shape}")
    else:
        diff = np.abs(xa.astype(np.float64) - xb.astype(np.float64))
        nbad = int(np.count_nonzero(xa != xb))
        print(f"  {k:12s} DIFFERS  n_diff={nbad}/{xa.size}  "
              f"max_abs={np.nanmax(diff):.3e}  max_rel="
              f"{np.nanmax(diff/(np.abs(xa.astype(np.float64))+1e-300)):.3e}")
        ok = False

print("\n=== RESULT:", "BIT-EXACT MATCH" if ok else "MISMATCH", "===")
sys.exit(0 if ok else 1)
