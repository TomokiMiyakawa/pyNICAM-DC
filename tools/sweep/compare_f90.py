#!/usr/bin/env python3
"""
Join the pyNICAM-DC sweep timers with the f90 NICAM-DC reference and print a
per-step speed comparison, for every backend that produced a timers CSV.

Metric (agreed methodology):
  f90      per-step = MAIN_Main_Loop / nstep            (f90 steady across steps)
  pyNICAM  per-step = (MAIN_Main_Loop - Main_Loop_step1) / (nstep - 1)
                                                        (exclude 1st step: warmup / jax JIT)
  slowdown = pyNICAM_per_step / f90_per_step

Inputs (default): run/timers_numpy.csv, run/timers_jax.csv, run/f90_reference.csv
Usage:
  python scripts/compare_f90.py
  python scripts/compare_f90.py run/f90_reference.csv run/timers_numpy.csv [more.csv ...]
"""
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(
            (ln for ln in f if not ln.lstrip().startswith("#"))))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main(f90_csv, py_csvs):
    f90 = {r["glevel"]: r for r in read_csv(f90_csv)}
    # backend -> {glevel -> row}
    py = {}
    for p in py_csvs:
        if not os.path.exists(p):
            continue
        for r in read_csv(p):
            py.setdefault(r.get("backend", "numpy"), {})[r["glevel"]] = r

    hdr = ["gl", "backend", "dtl", "f90_loop", "f90/step", "py_loop",
           "py_step1", "nstep", "py/step(x1)", "slowdown"]
    print(f"{hdr[0]:>3} {hdr[1]:>7} {hdr[2]:>6} {hdr[3]:>10} {hdr[4]:>9} "
          f"{hdr[5]:>11} {hdr[6]:>9} {hdr[7]:>5} {hdr[8]:>11} {hdr[9]:>9}")
    for backend in sorted(py):
        for gl in sorted(py[backend]):
            p, f = py[backend][gl], f90.get(gl, {})
            f_ps = fnum(f.get("per_step"))
            p_ps = fnum(p.get("per_step_excl1"))
            slow = (p_ps / f_ps) if (p_ps and f_ps) else None
            print(f"{gl:>3} {backend:>7} {p.get('dtl',''):>6} "
                  f"{f.get('MAIN_Main_Loop','NA'):>10} "
                  f"{f'{f_ps:.3f}' if f_ps else 'NA':>9} "
                  f"{p.get('MAIN_Main_Loop','NA'):>11} "
                  f"{p.get('MAIN_Main_Loop_step1','NA'):>9} "
                  f"{p.get('nstep','NA'):>5} "
                  f"{f'{p_ps:.3f}' if p_ps else 'NA':>11} "
                  f"{f'{slow:.2f}x' if slow else 'NA':>9}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a:
        f90_csv, py_csvs = a[0], a[1:]
    else:
        f90_csv = os.path.join(ROOT, "run", "f90_reference.csv")
        py_csvs = sorted(glob.glob(os.path.join(ROOT, "run", "timers_*.csv")))
    main(f90_csv, py_csvs)
