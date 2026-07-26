#!/usr/bin/env python
"""Geometric validation of a generated boundary npz against a reference grid.

Compares the python mkrawgrid (standard+spring+gravcenter) product with a
reference boundary file (tutorial nicamdc-provenance json, or another npz):

  1. point-to-point |dx| between matching (l, i, j) cells (grids from the same
     algorithm + same spring beta should coincide to ~criteria*lambda, i.e.
     ~1e-4 of the local spacing; NOT bit-exact -- different iteration paths)
  2. spacing statistics (min/mean/max nearest-cell distance) as a
     resolution-quality check that is independent of point pairing.

Usage (from prep/hgrid/):
    python validate_hgrid.py GEN.npz REF.json|REF.npz [--g1d N] [--tol REL]
Default tol: max point offset < 1e-2 of mean spacing.
"""
import argparse
import json
import sys

import numpy as np


def load_boundary(path):
    if path.endswith(".json"):
        d = json.load(open(path))
        return {k: np.asarray(d[k], dtype=np.float64) for k in d}
    return dict(np.load(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen")
    ap.add_argument("ref")
    ap.add_argument("--g1d", type=int, default=None, help="gall_1d (default: infer, 10 regions rl00)")
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="max |dx| tolerance as a fraction of mean spacing")
    args = ap.parse_args()

    gen, ref = load_boundary(args.gen), load_boundary(args.ref)

    npt = gen["grd_x_x"].size
    if args.g1d is None:
        # rl00: 10 regions on 1 rank
        g1d = int(round(np.sqrt(npt / 10)))
    else:
        g1d = args.g1d
    lall = npt // (g1d * g1d)
    print(f"points={npt}  g1d={g1d}  lall={lall}")

    def core(d, keys):  # (lall, j, i) interior cells only, stacked xyz
        a = np.stack([d[k].reshape(lall, g1d, g1d) for k in keys], axis=-1)
        return a[:, 1:-1, 1:-1, :]

    xg = core(gen, ["grd_x_x", "grd_x_y", "grd_x_z"])
    xr = core(ref, ["grd_x_x", "grd_x_y", "grd_x_z"])

    # spacing scale from the reference: distance to the +i neighbour
    sp = np.sqrt(((xr[:, :, 1:, :] - xr[:, :, :-1, :]) ** 2).sum(-1))
    print(f"ref spacing (chord, unit sphere): min={sp.min():.6e} mean={sp.mean():.6e} max={sp.max():.6e}"
          f"  (max/min={sp.max()/sp.min():.3f})")

    ok = True
    for name, keys in [
        ("grd_x ", ["grd_x_x", "grd_x_y", "grd_x_z"]),
        ("xt_TI ", ["grd_xt_ix", "grd_xt_iy", "grd_xt_iz"]),
        ("xt_TJ ", ["grd_xt_jx", "grd_xt_jy", "grd_xt_jz"]),
    ]:
        if keys[0] not in ref:
            print(f"{name}: reference lacks {keys[0]}, skipped")
            continue
        a, b = core(gen, keys), core(ref, keys)
        d = np.sqrt(((a - b) ** 2).sum(-1))
        rel = d.max() / sp.mean()
        tag = "PASS" if rel < args.tol else "FAIL"
        if rel >= args.tol:
            ok = False
        print(f"{tag}  {name} |dx| max={d.max():.3e} mean={d.mean():.3e}"
              f"  max/spacing={rel:.3e}")

    gsp = np.sqrt(((xg[:, :, 1:, :] - xg[:, :, :-1, :]) ** 2).sum(-1))
    print(f"gen spacing: min={gsp.min():.6e} mean={gsp.mean():.6e} max={gsp.max():.6e}"
          f"  (max/min={gsp.max()/gsp.min():.3f})")

    print("OK" if ok else "GEOMETRY MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
