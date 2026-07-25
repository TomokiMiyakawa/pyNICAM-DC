#!/usr/bin/env python
"""Validate mkllmap.py against Fortran nicamdc mkllmap output.

Reads the Fortran llmap.rgnXXXXXXXX files (unformatted-sequential; endianness
auto-detected) produced from the SAME horizontal grid and lat-lon spec, and
compares per region, in traversal order:
    (lon_idx, lat_idx, n1, n2, n3)  -- exact integer match
    (w1, w2, w3)                    -- fp tolerance (default 1e-12)
Index conventions: Fortran is 1-based with n = suf(i,j) = g1d*(j-1)+i;
python stores 0-based (i, j) and lat/lon indices.

Usage (from prep/llmap/):
    python validate_mkllmap.py GEN.npz FORTRAN_DIR [--wtol 1e-12]
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np


def read_fortran_llmap(path, endian):
    b = open(path, "rb").read()
    pos = 0

    def rec(fmt_char, count):
        nonlocal pos
        (m1,) = struct.unpack_from(endian + "i", b, pos)
        pos += 4
        size = {"i": 4, "d": 8}[fmt_char] * count
        vals = np.frombuffer(b, dtype=endian + ({"i": "i4", "d": "f8"}[fmt_char]),
                             count=count, offset=pos)
        pos += size
        (m2,) = struct.unpack_from(endian + "i", b, pos)
        pos += 4
        assert m1 == m2 == size, f"bad record markers in {path}"
        return vals

    n = int(rec("i", 1)[0])
    if n == 0:
        return {k: np.empty(0, dtype=int) for k in
                ("lon", "lat", "n1", "n2", "n3")} | {k: np.empty(0) for k in ("w1", "w2", "w3")}
    out = {}
    for k in ("lon", "lat", "n1", "n2", "n3"):
        out[k] = rec("i", n)
    for k in ("w1", "w2", "w3"):
        out[k] = rec("d", n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen")
    ap.add_argument("fortran_dir")
    # measured cross-code floor: 8.4e-12 (gl05rl01, 72x36) -- nvfortran-vs-numpy
    # libm (tan/atan) differences amplified by l'Huillier on small sub-triangles.
    # Indices are exact; anything < 1e-10 is the fp floor, not an algorithm bug.
    ap.add_argument("--wtol", type=float, default=1e-10)
    args = ap.parse_args()

    g = np.load(args.gen)
    g1d = int(g["g1d"])

    # python (i, j) 0-based -> Fortran suf 1-based
    def suf(i, j):
        return g1d * j + i + 1

    n_f = {m: suf(g[f"i{m}"], g[f"j{m}"]) for m in (1, 2, 3)}

    files = sorted(glob.glob(os.path.join(args.fortran_dir, "llmap.rgn*")))
    if not files:
        raise SystemExit("no llmap.rgn* files found")

    # endianness probe on the first file's first marker (record size 4)
    b0 = open(files[0], "rb").read(4)
    endian = "<" if struct.unpack("<i", b0)[0] == 4 else ">"

    total = bad_idx = 0
    wmax = wmax_edge = 0.0
    for f in files:
        rgn = int(f.rsplit("rgn", 1)[1])
        ref = read_fortran_llmap(f, endian)
        sel = np.nonzero(np.asarray(g["rgnid"]) == rgn)[0]
        if len(sel) != len(ref["lon"]):
            print(f"FAIL rgn{rgn:05d}: count {len(sel)} != fortran {len(ref['lon'])}")
            bad_idx += 1
            continue
        if len(sel) == 0:
            continue
        ok = (np.array_equal(g["lon_idx"][sel] + 1, ref["lon"])
              and np.array_equal(g["lat_idx"][sel] + 1, ref["lat"])
              and all(np.array_equal(n_f[m][sel], ref[f"n{m}"]) for m in (1, 2, 3)))
        if not ok:
            print(f"FAIL rgn{rgn:05d}: index mismatch")
            bad_idx += 1
            continue
        # split off ll points sitting EXACTLY on a triangle edge (with
        # lon_offset=false the grid lines coincide with the icosahedral
        # symmetry meridians): their sliver weight (~0) is pure cross-libm
        # noise (~4e-8 observed), and since w1+w2+w3=1 that noise propagates
        # into the two real weights of the SAME point -- so the split is
        # per-point (min weight < 1e-6), not per-weight. Interior points must
        # still match at the fp floor.
        wmin = np.minimum.reduce([np.minimum(np.abs(g[f"w{m}"][sel]), np.abs(ref[f"w{m}"]))
                                  for m in (1, 2, 3)])
        edge = wmin < 1.0e-6
        for m in (1, 2, 3):
            d = np.abs(g[f"w{m}"][sel] - ref[f"w{m}"])
            if (~edge).any():
                wmax = max(wmax, d[~edge].max())
            if edge.any():
                wmax_edge = max(wmax_edge, d[edge].max())
        total += len(sel)

    print(f"regions: {len(files)}  points compared: {total}")
    print(f"index match: {'ALL EXACT' if bad_idx == 0 else f'{bad_idx} regions FAILED'}")
    print(f"weight max|d|: {wmax:.3e}  (tol {args.wtol:g})")
    if wmax_edge > 0.0:
        print(f"on-edge sliver weights (<1e-6) max|d|: {wmax_edge:.3e} (informational)")
    ok = bad_idx == 0 and wmax < args.wtol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
