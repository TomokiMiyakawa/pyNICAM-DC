#!/usr/bin/env python3
"""
Machine-precision comparison of two pyNICAM-DC output zarr stores (the rel-tol
companion to proto/cmp_zarr.py, which is BIT-EXACT only).

  python cmp_prec.py <ref.zarr> <test.zarr> [--rtol 1e-10] [--exact]

Use which when:
- **cmp_zarr.py (bit-exact):** validate that *data movement* matches exactly --
  e.g. on-device COMM (PYNICAM_ONDEVICE_COMM=1) vs numpy-COMM. COMM moves values
  without arithmetic, so it MUST be bit-identical.
- **cmp_prec.py (this, rel-tol):** validate that *compute* matches to round-off --
  e.g. GPU/jax output vs the deterministic CPU numpy gold. GPU rounding and XLA
  fused op-order differ from CPU, so the result is machine-precision, NOT bit-exact;
  a bit-exact check would (correctly) fail. Use this instead.

Metric: per array, max|x-y| and the error **relative to the field's own scale**
(max|x-y| / max|x|), NOT pointwise |x-y|/|x|. Pointwise relative error blows up
where a field legitimately crosses zero (the momentum components RHOGVX/Y/Z/W),
which is a metric artifact, not a real discrepancy -- field-scale-relative avoids
that false failure. ~1e-11..1e-13 is normal float64 round-off; a real kernel bug
is O(1e-3)+, so the 1e-10 default separates the two with a huge margin.

Exit 0 if every shared array agrees within rtol (or is bit-identical with
--exact), else 1.

Origin: written for the AORI CPU session (gl05-09 numpy<->jax machine-precision
checks + the CPU numpy golds). The CPU golds live on the AORI filesystem
(pynicam-sweep/run/golds/gl0N_numpy_gold.zarr); rsync them to validate GPU runs.
"""
import sys, argparse
import numpy as np
import zarr

def walk(g, prefix=""):
    out = {}
    for k in g.array_keys():
        out[prefix + k] = g[k]
    for k in g.group_keys():
        out.update(walk(g[k], prefix + k + "/"))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("test")
    ap.add_argument("--rtol", type=float, default=1e-10,
                    help="field-scale-relative tolerance (default 1e-10)")
    ap.add_argument("--exact", action="store_true",
                    help="require bit-identical instead (same as proto/cmp_zarr.py)")
    a = ap.parse_args()
    ref = walk(zarr.open(a.ref, mode="r"))
    test = walk(zarr.open(a.test, mode="r"))
    keys = sorted(set(ref) | set(test))
    worst_rel = 0.0; worst_abs = 0.0; bad = []; nbit = 0
    print(f"{'variable':28s}{'shape':>16}{'max_abs':>12}{'max_rel':>12}  status")
    for k in keys:
        if k not in ref or k not in test:
            print(f"{k:28s}  MISSING in {'test' if k in ref else 'ref'}"); bad.append(k); continue
        x = np.asarray(ref[k][:], dtype=np.float64)
        y = np.asarray(test[k][:], dtype=np.float64)
        if x.shape != y.shape:
            print(f"{k:28s}  SHAPE {x.shape} vs {y.shape}"); bad.append(k); continue
        d = np.abs(x - y)
        ma = float(d.max())
        # error relative to the FIELD's own scale (not pointwise) -- pointwise
        # |x-y|/|x| explodes where a field legitimately crosses zero (momentum),
        # which is a metric artifact, not a discrepancy. max|x| is the physical scale.
        sc = float(np.abs(x).max())
        mr = ma / sc if sc > 0 else 0.0
        worst_abs = max(worst_abs, ma); worst_rel = max(worst_rel, mr)
        bitexact = bool((x == y).all()); nbit += bitexact
        ok = bitexact if a.exact else (mr <= a.rtol)
        if not ok: bad.append(k)
        tag = "BIT-EXACT" if bitexact else ("ok" if (mr <= a.rtol) else "FAIL")
        print(f"{k:28s}{str(x.shape):>16}{ma:12.2e}{mr:12.2e}  {tag}")
    print("-"*84)
    print(f"arrays: {len(keys)}  bit-exact: {nbit}  worst_abs: {worst_abs:.2e}  worst_rel: {worst_rel:.2e}")
    crit = ("bit-exact" if a.exact else f"rtol<={a.rtol:g}")
    print(("PASS" if not bad else f"FAIL ({len(bad)} arrays)") + f"  [{crit}]")
    sys.exit(0 if not bad else 1)

if __name__ == "__main__":
    main()
