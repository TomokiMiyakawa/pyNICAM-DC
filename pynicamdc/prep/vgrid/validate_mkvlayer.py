#!/usr/bin/env python
"""Validate mkvlayer.py bit-exactly against Fortran nicamdc mkvlayer output.

reference/*.dat are unformatted-sequential files produced by the nicamdc
prg_mkvlayer binary (2026-07-25, nvfortran on Miyabi; endianness auto-detected
here) for the three layer types:

    vgrid30_ullrich14_gold.dat  ULLRICH14  n=30  ztop=1.0e4
    vgrid30_even_gold.dat       EVEN       n=30  ztop=1.2e4
    vgrid40_given_gold.dat      GIVEN      n=40  inputzdata/_vgrid_40L_exp.dat

The python port must reproduce z_c and z_h BIT-EXACTLY (same fp64 ops).

PROVENANCE NOTE: the historical production vgrids (e.g. vgrid30_400m_dcmip)
use an older half-level convention (gzh[kmin]=0) that the CURRENT Fortran
mkvlayer EVEN does not produce for any ztop. Do not try to regenerate them
with EVEN; they are frozen data (GIVEN mode reproduces them from their own
half levels).

Run from prep/vgrid/:  python validate_mkvlayer.py
"""
import os
import struct
import sys

import numpy as np

from mkvlayer import mkvlayer

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = [
    ("ULLRICH14", dict(num_of_layer=30, layer_type="ULLRICH14", ztop=1.0e4),
     "vgrid30_ullrich14_gold.dat"),
    ("EVEN", dict(num_of_layer=30, layer_type="EVEN", ztop=1.2e4),
     "vgrid30_even_gold.dat"),
    ("GIVEN", dict(num_of_layer=40, layer_type="GIVEN",
                   infname=os.path.join(HERE, "inputzdata/_vgrid_40L_exp.dat")),
     "vgrid40_given_gold.dat"),
]


def read_dat(path):
    b = open(path, "rb").read()
    for e in ("<", ">"):
        (n,) = struct.unpack(e + "i", b[4:8])
        if 0 < n < 1000:
            kall = n + 2
            zc = np.frombuffer(b[16:16 + kall * 8], dtype=e + "f8")
            zh = np.frombuffer(b[24 + kall * 8:24 + 2 * kall * 8], dtype=e + "f8")
            return n, zc.astype("<f8"), zh.astype("<f8")
    raise ValueError(f"cannot parse {path}")


def main():
    n_fail = 0
    for name, kw, gold in CASES:
        n, zc, zh = read_dat(os.path.join(HERE, "reference", gold))
        m = mkvlayer(outfname=os.devnull, **kw)
        m.generate_layers()
        bit = np.array_equal(m.z_c, zc) and np.array_equal(m.z_h, zh)
        if bit:
            print(f"PASS  {name:10s} n={n}  BIT-EXACT vs {gold}")
        else:
            n_fail += 1
            print(f"FAIL  {name:10s} n={n}  max|dzc|={np.max(np.abs(m.z_c - zc)):.3e}"
                  f" max|dzh|={np.max(np.abs(m.z_h - zh)):.3e}")
    print("ALL PASS" if n_fail == 0 else f"{n_fail} FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
