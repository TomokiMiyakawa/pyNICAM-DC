#!/usr/bin/env python3
"""Render pyNICAM-DC zarr output to PNG frames + a movie.

Reusable, headless (Agg) version of the daskzarr_out-simple notebook viewer.
Resamples the icosahedral field onto a regular lon/lat grid via a KDTree over the
cell centres (GRD_x), draws one equirectangular world map per time step, saves a
PNG per frame, and stitches the frames into an mp4 (falls back to gif).

Cartopy-free by default (the core needs only numpy/scipy/xarray/matplotlib). If
cartopy is installed, `--coastlines` overlays coastlines.

Examples:
  python render_zarr.py run/gl08_x/testout_tmp.zarr --var RHOGE --k 25
  python render_zarr.py out.zarr --var RHOGVX --k 10 --time 0:4 --fps 4 --movie rhogvx.mp4
  python render_zarr.py out.zarr --list        # just print variables/dims and exit
"""
import argparse
import os
import sys
import numpy as np


def parse_time(spec, nt):
    if spec in (None, "all"):
        return list(range(nt))
    if ":" in spec:                                   # "a:b" python-style half-open
        a, b = spec.split(":")
        a = int(a) if a else 0
        b = int(b) if b else nt
        return list(range(a, b))
    return [int(spec)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zarr", help="path to a *.zarr output dir (must contain GRD_x)")
    ap.add_argument("--var", default="RHOGE", help="variable to plot (default RHOGE)")
    ap.add_argument("--k", type=int, default=None, help="vertical level index (default: middle)")
    ap.add_argument("--time", default="all", help="'all' | int | 'a:b' (default all)")
    ap.add_argument("--outdir", default=None, help="frame output dir (default frames_<var>_k<k>)")
    ap.add_argument("--movie", default=None, help="movie path (default <outdir>/<var>_k<k>.mp4)")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--nx", type=int, default=720, help="lon pixels (default 720)")
    ap.add_argument("--ny", type=int, default=360, help="lat pixels (default 360)")
    ap.add_argument("--cmap", default="coolwarm")
    ap.add_argument("--vmin", type=float, default=None, help="colour min (default 2nd pctile)")
    ap.add_argument("--vmax", type=float, default=None, help="colour max (default 98th pctile)")
    ap.add_argument("--method", choices=("nearest", "linear"), default="nearest",
                    help="resampling: nearest, or linear (k=6 inverse-distance)")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--projection", choices=("mollweide", "flat"), default="mollweide",
                    help="mollweide (equal-area, default) or flat (equirectangular)")
    ap.add_argument("--coastlines", action="store_true", help="overlay coastlines (flat + cartopy)")
    ap.add_argument("--list", action="store_true", help="print variables/dims and exit")
    args = ap.parse_args()

    import xarray as xr
    ds = xr.open_dataset(args.zarr, engine="zarr")

    if args.list:
        print("dims:", dict(ds.sizes))
        print("data_vars:", list(ds.data_vars))
        return
    if args.var not in ds.data_vars:
        sys.exit(f"var '{args.var}' not found. available: {list(ds.data_vars)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    nt = ds.sizes["time"]
    nk = ds.sizes.get("k", 1)
    k = args.k if args.k is not None else nk // 2
    times = [t for t in parse_time(args.time, nt) if 0 <= t < nt]
    if not times:
        sys.exit(f"no valid time steps in '{args.time}' (nt={nt})")

    # strip the 1-cell halo, stack (r,i,j) -> cell, build KDTree over unit-sphere centres
    cell = ds.isel(i=slice(1, -1), j=slice(1, -1)).stack(cell=("r", "i", "j"))
    xyz = cell.GRD_x.transpose("cell", "xyz").values.astype(float)
    xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
    tree = cKDTree(xyz)

    # regular lon/lat pixel centres -> xyz -> tree query weights (frame-independent)
    lon = np.linspace(-180, 180, args.nx, endpoint=False) + 180.0 / args.nx
    lat = np.linspace(-90, 90, args.ny, endpoint=False) + 90.0 / args.ny
    LON, LAT = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    P = np.stack([(np.cos(LAT) * np.cos(LON)).ravel(),
                  (np.cos(LAT) * np.sin(LON)).ravel(),
                  np.sin(LAT).ravel()], axis=-1)
    if args.method == "nearest":
        _, idx = tree.query(P)
        resample = lambda v: v[idx].reshape(args.ny, args.nx)
    else:
        dist, idx = tree.query(P, k=6)
        w = 1.0 / np.maximum(dist, 1e-12)
        w /= w.sum(-1, keepdims=True)
        resample = lambda v: (v[idx] * w).sum(-1).reshape(args.ny, args.nx)

    var = cell[args.var].isel(k=k)                    # dims (time, cell)
    sel = var.isel(time=times).values.astype(float)
    if not np.isfinite(sel).any():
        sys.exit(f"'{args.var}' at k={k} is all-NaN for the selected times -- this run "
                 f"likely had output disabled (perf run). Try a gold or an output-on run, "
                 f"or a different --k/--var.")
    vmin = args.vmin if args.vmin is not None else float(np.nanpercentile(sel, 2))
    vmax = args.vmax if args.vmax is not None else float(np.nanpercentile(sel, 98))

    outdir = args.outdir or f"frames_{args.var}_k{k}"
    os.makedirs(outdir, exist_ok=True)

    frames = []
    for t in times:
        img = resample(var.isel(time=t).values.astype(float))
        if args.projection == "mollweide":
            # matplotlib's native Mollweide axes (equal-area); takes lon/lat in
            # radians and projects internally -> cartopy not required.
            fig, ax = plt.subplots(figsize=(9, 5.0), constrained_layout=True,
                                   subplot_kw={"projection": "mollweide"})
            m = ax.pcolormesh(LON, LAT, img, cmap=args.cmap, vmin=vmin, vmax=vmax,
                              shading="auto", rasterized=True)
            ax.grid(True, linewidth=0.3, color="0.6")
            ax.set_xticklabels([])                      # hide crowded lon labels
            if args.coastlines:
                print("  (coastlines need cartopy; not drawn on the native mollweide)")
        else:                                           # flat / equirectangular
            fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
            m = ax.imshow(img, extent=[-180, 180, -90, 90], origin="lower",
                          aspect="auto", cmap=args.cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
            ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
            if args.coastlines:
                try:
                    import cartopy.feature as cf
                    ax.add_geometries(list(cf.COASTLINE.geometries()), crs=None,
                                      facecolor="none", edgecolor="#333", linewidth=0.5)
                except Exception as e:
                    print(f"  (coastlines skipped: {e})")
        ax.set_title(f"{args.var}  k={k}  time={t}")
        fig.colorbar(m, ax=ax, shrink=0.85, label=args.var)
        fp = os.path.join(outdir, f"{args.var}_k{k}_t{t:04d}.png")
        fig.savefig(fp, dpi=args.dpi)
        plt.close(fig)
        frames.append(fp)
        print("frame:", fp)

    if len(frames) > 1:
        movie = args.movie or os.path.join(outdir, f"{args.var}_k{k}.mp4")
        import imageio.v2 as imageio
        try:
            with imageio.get_writer(movie, fps=args.fps, macro_block_size=None) as wtr:
                for fp in frames:
                    wtr.append_data(imageio.imread(fp))
            print("movie:", movie)
        except Exception as e:
            gif = os.path.splitext(movie)[0] + ".gif"
            imageio.mimsave(gif, [imageio.imread(fp) for fp in frames], fps=args.fps)
            print(f"movie (gif fallback): {gif}   (mp4 failed: {e})")
    else:
        print("only one frame -> no movie")


if __name__ == "__main__":
    main()
