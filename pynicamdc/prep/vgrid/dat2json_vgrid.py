#!/usr/bin/env python3
"""Convert a nicamdc binary vertical-grid file (data/grid/vgrid/*.dat) to the
pyNICAM-DC JSON format that mod_grd.GRD_input_vgrid reads.

nicamdc .dat layout (big-endian Fortran unformatted sequential), decoded + validated
bit-exact against nicamdc's own reference JSONs:
    [rec: 1 int32]          header  (12 bytes: 4 marker + 4 int + 4 marker)
    [rec: n float64]        gz   (cell-CENTER heights, len n = vlayer+2)  -> data at byte 16
    [rec: n float64]        gzh  (HALF-level heights,   len n)            -> data at byte 24 + n*8
  file size = 28 + 16*n  =>  n = (size - 28) // 16
pyNICAM JSON: {"set1": gz(list), "set2": gzh(list)}  (set1->GRD_gz, set2->GRD_gzh)

Usage:
  dat2json_vgrid.py IN.dat OUT.json                 # convert
  dat2json_vgrid.py IN.dat OUT.json --check REF.json # convert + assert == reference

Related: pyNICAM's own prep tool `pynicamdc/prep/vgrid/mkvlayer.py` GENERATES a vgrid
from a z-layer spec (ULLRICH14/EVEN/GIVEN); this script instead CONVERTS an already-built
nicamdc .dat. Source .dat live in nicamdc*/data/grid/vgrid/ ; readable z-tables in
nicamdc*/data/zaxis/ (Z*=centers, ZS*=centers+ghosts).
"""
import sys
import json
import argparse
import numpy as np


def dat2json(datpath):
    b = open(datpath, "rb").read()
    n = (len(b) - 28) // 16                       # kall = vlayer + 2
    if 28 + 16 * n != len(b):
        raise ValueError(f"{datpath}: size {len(b)} != 28+16*n ({28+16*n}); "
                         "not a standard nicamdc vgrid .dat")
    gz = np.frombuffer(b[16:16 + n * 8], dtype=">f8")
    gzh = np.frombuffer(b[24 + n * 8:24 + 2 * n * 8], dtype=">f8")
    return {"set1": gz.tolist(), "set2": gzh.tolist()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dat")
    ap.add_argument("out")
    ap.add_argument("--check", help="reference JSON to validate the conversion against")
    a = ap.parse_args()
    d, n = dat2json(a.dat)
    json.dump(d, open(a.out, "w"))
    print(f"wrote {a.out}  (z{n-2}, top center = {d['set1'][-2]:.1f} m)")
    if a.check:
        ref = json.load(open(a.check))
        ok1 = np.allclose(d["set1"], ref["set1"], atol=1e-9)
        ok2 = np.allclose(d["set2"], ref["set2"], atol=1e-9)
        print(f"  validate vs {a.check}: set1={'OK' if ok1 else 'FAIL'} "
              f"set2={'OK' if ok2 else 'FAIL'}")
        sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
