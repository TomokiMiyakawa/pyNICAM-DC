#!/usr/bin/env python3
"""
Convert f90 NICAM-DC restart binary files into the JSON format pyNICAM-DC reads
with restart input_io_mode="json", for arbitrary gl / rl / pe / vertical levels.

Generalises extract_1d_restart.ipynb (which was hard-wired to
expected_shape=(5,42,324)).  Everything is derived from the file header instead:

  glevel, rlevel, num_of_rgn   <- read from the 6-int header block
  gall   = (2**(glevel-rlevel) + 2) ** 2
  num_of_data                  <- read from header
  kall   = datasize / itemsize / (num_of_rgn * gall)   (per variable)

Per variable the raw stream is reshaped (num_of_rgn, kall, gall) and transposed
to (gall, kall, num_of_rgn) = (ij, layer, region) -- exactly what the reader
mod_prgvar.restart_input expects:  variable_array[ij, k, l].

Input  files (f90):  <in_dir>/<base><rank:06d>            e.g. restart_all_GL05RL01z40.pe000002
Output files (json): <out_dir>/<base><rank:08d>.json      e.g. restart_all_GL05RL01z40.pe00000002.json
(the reader builds restart_input_basename + rank.zfill(8) + ".json")

Usage:
  python restart2json.py --in-dir ./org --out-dir ./json
  # base name + pe are auto-detected from the directory; override with --in-base/--pe.
"""
import argparse
import glob
import json
import os
import re
import sys
import struct
import numpy as np

DTYPE_MAP = {0: ">f4", 1: ">f8", 2: ">i4", 3: ">i8"}
DESC, NOTE = 64, 256
ITEM_DESC, ITEM_UNIT, ITEM_LAYER, ITEM_NOTE = 64, 16, 16, 256


def convert_file(path):
    """Parse one f90 restart file -> the JSON-ready dict (notebook structure)."""
    with open(path, "rb") as f:
        header = f.read(DESC).decode(errors="ignore").strip("\x00").strip()
        note = f.read(NOTE).decode(errors="ignore").strip("\x00").strip()
        fmode, endian, topo, glevel, rlevel, num_of_rgn = struct.unpack(">6I", f.read(24))
        rgnid = struct.unpack(f">{num_of_rgn}I", f.read(4 * num_of_rgn))
        num_of_data = struct.unpack(">I", f.read(4))[0]

        gall = (2 ** (glevel - rlevel) + 2) ** 2

        out = {
            "Header": header, "Note": note, "File Mode": fmode,
            "Endian Type": endian, "Grid Topology": topo,
            "Grid Level": glevel, "Resolution Level": rlevel,
            "Number of Regions": num_of_rgn, "Region IDs": list(rgnid),
            "Number of Data Entries": num_of_data, "Variables": {},
        }

        for _ in range(num_of_data):
            varname = f.read(16).decode(errors="ignore").strip("\x00").strip()
            description = f.read(ITEM_DESC).decode(errors="ignore").strip("\x00").strip()
            unit = f.read(ITEM_UNIT).decode(errors="ignore").strip("\x00").strip()
            layername = f.read(ITEM_LAYER).decode(errors="ignore").strip("\x00").strip()
            _vnote = f.read(ITEM_NOTE)                       # per-var note (unused)
            datasize, datatype, _num_layer, _step = struct.unpack(">Q3I", f.read(8 + 12))
            time_start, time_end = struct.unpack(">QQ", f.read(16))
            raw = f.read(datasize)
            fmt = DTYPE_MAP.get(datatype)
            if fmt is None:
                continue
            arr = np.frombuffer(raw, dtype=np.dtype(fmt))
            per = num_of_rgn * gall
            if arr.size % per != 0:
                raise ValueError(
                    f"{os.path.basename(path)} var '{varname}': {arr.size} elems "
                    f"not divisible by num_of_rgn*gall = {num_of_rgn}*{gall} = {per}")
            kall = arr.size // per
            arr = arr.reshape(num_of_rgn, kall, gall).transpose(2, 1, 0)   # (ij, k, region)
            # NICAM keeps two dummy vertical levels (k=0 below the surface,
            # k=kall-1 above the model top). nicamdc never initialises the top
            # one, so it dumps NaN/garbage there. pyNICAM builds its singular
            # pole arrays directly from the ingested field (before the vertical
            # BC runs), so a garbage top dummy seeds NaN at the pole corners and
            # blows the run up. The dummy levels carry no physical state -- the
            # vertical BC (BNDCND) recomputes them from the physical levels at
            # the start of every step, so the fill VALUE is washed out and does
            # not change the solution (verified bit-exact at k=kmin..kmax across
            # two different fills). Copy the adjacent physical level to keep them
            # finite. Only rewrite non-finite cells so valid data is untouched.
            if kall >= 3:
                arr = np.array(arr)   # frombuffer view is read-only -> own the data
                for kd, ksrc in ((-1, -2), (0, 1)):
                    bad = ~np.isfinite(arr[:, kd, :])
                    if bad.any():
                        arr[:, kd, :] = np.where(bad, arr[:, ksrc, :], arr[:, kd, :])
            out["Variables"][varname] = {
                "Description": description, "Unit": unit, "Layer Name": layername,
                "Time Start": time_start, "Time End": time_end,
                "Data": arr,   # keep ndarray; serialised (json/npz) at write time
            }
        return out, num_of_rgn, num_of_data, glevel, rlevel, kall


def detect(in_dir, in_base):
    """Find restart rank files; return (base, sorted list of (rank, path))."""
    pat = re.compile(r"^(?P<base>.+?)(?P<rank>\d{6})$")
    found = {}
    for p in glob.glob(os.path.join(in_dir, "*")):
        if not os.path.isfile(p):
            continue
        b = os.path.basename(p)
        m = pat.match(b)   # must end in exactly 6 digits (rank); .json/.ipynb won't match

        if not m:
            continue
        base = m.group("base")
        if in_base and base != in_base:
            continue
        found.setdefault(base, []).append((int(m.group("rank")), p))
    if not found:
        raise SystemExit(f"no rank files (…NNNNNN) found in {in_dir}")
    base = max(found, key=lambda k: len(found[k]))     # the largest set
    return base, sorted(found[base])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--in-base", default=None,
                    help="prefix before the 6-digit rank (default: auto-detect)")
    ap.add_argument("--out-prefix", default=None,
                    help="json prefix before the 8-digit rank "
                         "(default: same as in-base; must match restart_input_basename)")
    ap.add_argument("--pe", type=int, default=None,
                    help="expected #ranks (default: all files found)")
    ap.add_argument("--format", choices=("json", "npz"), default="json",
                    help="output format (default json). npz = compact numpy archive "
                         "(needs the reader's npz path; far smaller/faster at high gl).")
    ap.add_argument("--indent", type=int, default=None,
                    help="json indent (default compact; 4 = readable but ~larger)")
    ap.add_argument("--warn-mb", type=float, default=200,
                    help="warn (suggest npz) when an estimated JSON file exceeds this MB")
    if len(sys.argv) == 1:                 # no args -> show usage/help
        ap.print_help(sys.stderr)
        return 1
    a = ap.parse_args()

    base, ranks = detect(a.in_dir, a.in_base)
    out_prefix = a.out_prefix if a.out_prefix is not None else base
    if a.pe is not None and len(ranks) != a.pe:
        print(f"WARN: found {len(ranks)} files but --pe={a.pe}")
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"base='{base}'  ranks={[r for r, _ in ranks]}")

    for rank, path in ranks:
        d, nrgn, ndat, gl, rl, kall = convert_file(path)
        vars_ = d["Variables"]
        if a.format == "json":
            est_mb = sum(v["Data"].size for v in vars_.values()) * 20 / 1e6
            if est_mb > a.warn_mb:
                print(f"  !! WARNING rank {rank}: JSON ~{est_mb:.0f} MB (high resolution). "
                      f"Consider --format npz (much smaller/faster).")
            outp = os.path.join(a.out_dir, out_prefix + str(rank).zfill(8) + ".json")
            d_json = dict(d)
            d_json["Variables"] = {k: {**v, "Data": v["Data"].tolist()}
                                   for k, v in vars_.items()}
            with open(outp, "w") as jf:
                json.dump(d_json, jf, indent=a.indent)
        else:
            outp = os.path.join(a.out_dir, out_prefix + str(rank).zfill(8) + ".npz")
            np.savez_compressed(outp, **{k: v["Data"] for k, v in vars_.items()})
        print(f"  rank {rank}: GL{gl}RL{rl} nrgn={nrgn} kall={kall} vars={ndat} "
              f"{list(vars_)[:4]}... -> {os.path.basename(outp)}")


if __name__ == "__main__":
    raise SystemExit(main())
