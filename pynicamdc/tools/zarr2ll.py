#!/usr/bin/env python3
"""Remap pyNICAM-DC zarr output onto a regular lat-lon grid (ico2ll).

The python counterpart of nicamdc fio_ico2ll: applies the 3-point barycentric
remap table built by prep/llmap/mkllmap.py (triangle containment + spherical
area weights, Fortran-validated) to any variable of a pyNICAM zarr store, and
writes a CF-ish netcdf (scipy backend, no netCDF4 dependency).

For quick visualization the KDTree nearest-neighbour path in render_zarr.py
remains the default; use THIS for science-grade fields (the weights are
continuous across cell boundaries and reproduce constants exactly).

Usage:
  zarr2ll.py RUN.zarr [RUN_rank1.zarr ...] --llmap LLMAP.npz --var sl_ps --out ps_ll.nc
  zarr2ll.py RUN.zarr --llmap LLMAP.npz --var RHOG --k 2 --out rho_ll.nc
  zarr2ll.py RUN.zarr --llmap LLMAP.npz --selftest        # remap a constant 1

Multiple zarr stores are concatenated along r in the given order; the
resulting global region order must match the llmap 'rgnid' numbering (true
for the standard contiguous rank->region assignment).
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zarr", nargs="+")
    ap.add_argument("--llmap", required=True, help="npz table from prep/llmap/mkllmap.py")
    ap.add_argument("--var", default=None)
    ap.add_argument("--k", type=int, default=None, help="single level of a 3D field (default: all)")
    ap.add_argument("--time", default="all", help="'all', an index, or 'a:b'")
    ap.add_argument("--out", default=None, help="output netcdf path")
    ap.add_argument("--selftest", action="store_true",
                    help="remap the constant 1 field and report max deviation")
    args = ap.parse_args()

    import xarray as xr

    m = np.load(args.llmap)
    rgn = m["rgnid"]
    jj, ii = m["lat_idx"], m["lon_idx"]
    imax, jmax = int(m["imax"]), int(m["jmax"])
    lat_deg, lon_deg = np.degrees(m["lat"]), np.degrees(m["lon"])

    dss = [xr.open_dataset(p, engine="zarr") for p in args.zarr]
    ds = dss[0] if len(dss) == 1 else xr.concat(dss, dim="r")
    nrgn_needed = int(rgn.max()) + 1
    if ds.sizes["r"] < nrgn_needed:
        sys.exit(f"zarr stores provide r={ds.sizes['r']} regions; llmap needs {nrgn_needed}")

    def remap(F):
        """F: (r, i, j, ...) numpy -> (jmax, imax, ...) lat-lon field."""
        acc = None
        for n in (1, 2, 3):
            g = F[rgn, m[f"i{n}"], m[f"j{n}"]] * \
                m[f"w{n}"].reshape((-1,) + (1,) * (F.ndim - 3))
            acc = g if acc is None else acc + g
        out = np.full((jmax, imax) + F.shape[3:], np.nan, dtype=F.dtype)
        out[jj, ii] = acc
        return out

    if args.selftest:
        ones = np.ones((ds.sizes["r"], ds.sizes["i"], ds.sizes["j"]))
        out = remap(ones)
        print(f"selftest: constant-1 remap  max|out-1| = {np.abs(out - 1).max():.3e}"
              f"   coverage = {np.isfinite(out).sum()}/{imax * jmax}")
        return

    if args.var is None or args.out is None:
        sys.exit("--var and --out are required (or use --selftest)")
    if args.var not in ds.data_vars:
        sys.exit(f"var '{args.var}' not found. available: {list(ds.data_vars)}")

    da = ds[args.var]
    tdim = "time2d" if "time2d" in da.dims else "time"
    nt = ds.sizes[tdim]
    if args.time == "all":
        tsel = list(range(nt))
    elif ":" in args.time:
        a, b = args.time.split(":")
        tsel = list(range(int(a) if a else 0, int(b) if b else nt))
    else:
        tsel = [int(args.time)]

    has_k = "k" in da.dims
    if has_k and args.k is not None:
        da = da.isel(k=args.k)
        has_k = False

    order = ("r", "i", "j") + (("k",) if has_k else ()) + (tdim,)
    F = da.isel({tdim: tsel}).transpose(*order).values

    out = remap(F)  # (jmax, imax[, k], t)

    dims = ("lat", "lon") + (("k",) if has_k else ()) + ("time",)
    coords = {"lat": ("lat", lat_deg, {"units": "degrees_north"}),
              "lon": ("lon", lon_deg, {"units": "degrees_east"}),
              "time": ("time", np.asarray(tsel))}
    if has_k:
        coords["k"] = ("k", np.arange(out.shape[2]))
    da_out = xr.DataArray(out, dims=dims, coords=coords, name=args.var,
                          attrs={"remap": f"3-point barycentric (mkllmap {args.llmap})"})
    da_out.transpose("time", *(("k",) if has_k else ()), "lat", "lon") \
          .to_dataset().to_netcdf(args.out, engine="scipy")
    print(f"wrote {args.out}  ({args.var}, {len(tsel)} step(s), {jmax}x{imax})")


if __name__ == "__main__":
    main()
