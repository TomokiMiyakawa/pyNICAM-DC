#!/usr/bin/env python3
"""
Convert f90 NICAM-DC boundary (HORIZONTAL GRID) binary files into the JSON
format pyNICAM-DC reads with hgrid_io_mode="json", for arbitrary gl / rl / pe.

Generalises testtranspose.ipynb (which was hard-wired to gl05rl01pe04:
`9` datasets and reshape (10, 324)).  Here:

  gall   = (2**(gl - rl) + 2) ** 2        # horizontal points per region
  total regions = 10 * 4**rl
  regions/rank  = total / pe              # also read back from each file header
  num_of_data   = read from the header    # (not assumed to be 9)
  each dataset reshaped to (num_of_rgn, gall)  -> matches reader's [l, ij]

Input  files (f90):  <in_dir>/<in_base><rank:06d>           e.g. boundary_GL05RL01.pe000003
Output files (json): <out_dir>/<out_prefix><rank:08d>.json  e.g. bboundary_GL05RL01.pe00000003.json
(the 8-digit + ".json" suffix is what GRD_input_hgrid builds from hgrid_fname)

Example:
  python boundary2json.py --gl 5 --rl 1 --pe 4 \
      --in-dir ./gl05rl01pe04_org --out-dir ./json
"""
import argparse
import json
import os
import sys
import struct
import numpy as np

# datatype code -> big-endian numpy dtype (matches the f90 fio writer)
DTYPE_MAP = {0: ">f4", 1: ">f8", 2: ">i4", 3: ">i8"}

# fixed fio header field sizes (bytes)
DESC, NOTE = 64, 256                 # file description, file note
ITEM_DESC, ITEM_UNIT, ITEM_LAYER, ITEM_NOTE = 64, 16, 16, 256


def parse_rank_file(path, gall):
    """Parse one f90 boundary file -> (num_of_rgn, num_of_data, {varname: list})."""
    out = {}
    with open(path, "rb") as f:
        f.read(DESC)
        f.read(NOTE)
        # 6 ints; the 6th is num_of_rgn (regions this rank owns)
        *_, num_of_rgn = struct.unpack(">6I", f.read(4 * 6))
        f.read(4 * num_of_rgn)                       # rgnid array
        num_of_data = struct.unpack(">I", f.read(4))[0]

        for _ in range(num_of_data):
            varname = f.read(16).decode(errors="ignore").replace("\x00", "").strip()
            f.read(ITEM_DESC + ITEM_UNIT + ITEM_LAYER + ITEM_NOTE)   # skip metadata
            datasize, datatype, _num_layer, _step = struct.unpack(">Q3I", f.read(8 + 4 * 3))
            f.read(8 * 2)                            # time_start, time_end
            fmt = DTYPE_MAP.get(datatype)
            if fmt is None:
                f.seek(datasize, 1)
                continue
            arr = np.frombuffer(f.read(datasize), dtype=np.dtype(fmt))
            expect = num_of_rgn * gall
            if arr.size != expect:
                raise ValueError(
                    f"{os.path.basename(path)} var '{varname}': {arr.size} elements "
                    f"!= num_of_rgn*gall = {num_of_rgn}*{gall} = {expect}. "
                    f"Check --gl/--rl (gall) and the input file."
                )
            out[varname] = arr.reshape(num_of_rgn, gall)   # keep ndarray (json/npz at write time)
    return num_of_rgn, num_of_data, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gl", type=int, required=True)
    ap.add_argument("--rl", type=int, required=True)
    ap.add_argument("--pe", type=int, required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--in-base", default=None,
                    help="f90 prefix before the 6-digit rank "
                         "(default: boundary_GL<gl>RL<rl>.pe)")
    ap.add_argument("--out-prefix", default=None,
                    help="json prefix before the 8-digit rank "
                         "(default: bboundary_GL<gl>RL<rl>.pe ; must match hgrid_fname)")
    ap.add_argument("--format", choices=("json", "npz"), default="json",
                    help="output format (default json). npz = compact numpy archive "
                         "(needs the reader's npz path; far smaller/faster at high gl).")
    ap.add_argument("--indent", type=int, default=None,
                    help="json indent (default compact; use 4 for readable but larger)")
    ap.add_argument("--warn-mb", type=float, default=200,
                    help="warn (suggest npz) when an estimated JSON file exceeds this MB")
    if len(sys.argv) == 1:                 # no args -> show usage/help
        ap.print_help(sys.stderr)
        return 1
    a = ap.parse_args()

    tag = f"GL{a.gl:02d}RL{a.rl:02d}"
    in_base = a.in_base if a.in_base is not None else f"boundary_{tag}.pe"
    out_prefix = a.out_prefix if a.out_prefix is not None else f"bboundary_{tag}.pe"

    gall_1d = 2 ** (a.gl - a.rl) + 2
    gall = gall_1d * gall_1d
    total_rgn = 10 * 4 ** a.rl
    if total_rgn % a.pe != 0:
        raise SystemExit(f"total regions {total_rgn} not divisible by pe={a.pe}")
    rgn_per_rank = total_rgn // a.pe

    os.makedirs(a.out_dir, exist_ok=True)
    print(f"{tag} pe{a.pe}: gall_1d={gall_1d} gall={gall} "
          f"total_rgn={total_rgn} rgn/rank={rgn_per_rank}")

    for rank in range(a.pe):
        inp = os.path.join(a.in_dir, in_base + str(rank).zfill(6))
        if not os.path.exists(inp):
            raise SystemExit(f"missing input: {inp}")
        nrgn, ndat, d = parse_rank_file(inp, gall)
        if nrgn != rgn_per_rank:
            print(f"  WARN rank {rank}: header num_of_rgn={nrgn} != expected {rgn_per_rank}")
        if a.format == "json":
            est_mb = sum(v.size for v in d.values()) * 20 / 1e6  # ~20 chars per number
            if est_mb > a.warn_mb:
                print(f"  !! WARNING rank {rank}: JSON ~{est_mb:.0f} MB (high resolution). "
                      f"Consider --format npz (much smaller/faster).")
            outp = os.path.join(a.out_dir, out_prefix + str(rank).zfill(8) + ".json")
            with open(outp, "w") as jf:
                json.dump({k: v.tolist() for k, v in d.items()}, jf, indent=a.indent)
        else:
            outp = os.path.join(a.out_dir, out_prefix + str(rank).zfill(8) + ".npz")
            np.savez_compressed(outp, **d)
        print(f"  rank {rank}: {ndat} vars {list(d)[:3]}... {nrgn} regions -> {outp}")


if __name__ == "__main__":
    raise SystemExit(main())
