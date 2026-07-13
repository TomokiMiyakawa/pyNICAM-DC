#!/usr/bin/env python3
"""Render pyNICAM-DC zarr output to PNG frames + a movie.

Reusable, headless (Agg). Resamples the icosahedral field onto a regular grid via a
KDTree over the cell centres (GRD_x). Three render modes, auto/opt selected:
  * horizontal map (3D field at level --k)        -- e.g. RHOGE, ml_u, passive000
  * horizontal map (2D field, no k dim)           -- e.g. sl_ps, sl_pw  (auto)
  * lon-HEIGHT cross-section at a latitude (--cross-section LAT, 3D field)
    -- the right view for vertical phenomena (mountain waves, gravity waves).

Cartopy-free by default (numpy/scipy/xarray/matplotlib/imageio). --coastlines needs
cartopy (flat horizontal only).

Examples:
  render_zarr.py out.zarr --var ml_th_prime --k 20              # horizontal 3D
  render_zarr.py out.zarr --var sl_ps                           # horizontal 2D (auto)
  render_zarr.py out.zarr --var ml_w --cross-section 0          # lon-height at equator
  render_zarr.py out.zarr --list
"""
import argparse
import os
import sys
import numpy as np


def parse_time(spec, nt):
    if spec in (None, "all"):
        return list(range(nt))
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a) if a else 0, int(b) if b else nt))
    return [int(spec)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zarr")
    ap.add_argument("--var", default="RHOGE")
    ap.add_argument("--k", type=int, default=None, help="level index for a 3D horizontal map (default middle)")
    ap.add_argument("--cross-section", type=float, default=None, dest="xsec",
                    help="draw a lon-HEIGHT cross-section at this latitude (deg) instead of a map")
    ap.add_argument("--time", default="all")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--movie", default=None)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--nx", type=int, default=720)
    ap.add_argument("--ny", type=int, default=360)
    ap.add_argument("--cmap", default="coolwarm")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--method", choices=("nearest", "linear"), default="nearest")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--projection", choices=("mollweide", "flat"), default="mollweide")
    ap.add_argument("--lon0", type=float, default=0.0)
    ap.add_argument("--coastlines", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    import xarray as xr
    ds = xr.open_dataset(args.zarr, engine="zarr")
    if args.list:
        print("dims:", dict(ds.sizes)); print("data_vars:", list(ds.data_vars)); return
    if args.var not in ds.data_vars:
        sys.exit(f"var '{args.var}' not found. available: {list(ds.data_vars)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree

    has_k = "k" in ds[args.var].dims
    tdim = "time2d" if ("time2d" in ds[args.var].dims) else "time"
    nt = ds.sizes[tdim]
    times = [t for t in parse_time(args.time, nt) if 0 <= t < nt]
    if not times:
        sys.exit(f"no valid time steps in '{args.time}' (n{tdim}={nt})")
    if args.xsec is not None and not has_k:
        sys.exit(f"--cross-section needs a 3D field (with k); '{args.var}' has no k dim")

    # cell centres (halo stripped) on the unit sphere -> KDTree
    cell = ds.isel(i=slice(1, -1), j=slice(1, -1)).stack(cell=("r", "i", "j"))
    xyz = cell.GRD_x.transpose("cell", "xyz").values.astype(float)
    xyz /= np.linalg.norm(xyz, axis=-1, keepdims=True)
    tree = cKDTree(xyz)
    var = cell[args.var]

    def finite_range(sel):
        if not np.isfinite(sel).any():
            sys.exit(f"'{args.var}' is all-NaN for the selected times -- output likely "
                     f"disabled (perf run), or diagnostics not enabled. Try --list.")
        lo = args.vmin if args.vmin is not None else float(np.nanpercentile(sel, 2))
        hi = args.vmax if args.vmax is not None else float(np.nanpercentile(sel, 98))
        return lo, hi

    # ---------------- lon-HEIGHT cross-section ----------------
    if args.xsec is not None:
        nk = ds.sizes["k"]
        lon = np.linspace(-180, 180, args.nx, endpoint=False) + 180.0 / args.nx
        la = np.deg2rad(args.xsec)
        lo_r = np.deg2rad(lon)
        Pline = np.stack([np.cos(la) * np.cos(lo_r), np.cos(la) * np.sin(lo_r),
                          np.sin(la) * np.ones_like(lo_r)], axis=-1)
        _, idx = tree.query(Pline)                         # one cell column per lon
        # height axis: use ml_hgt diagnostic if present, else level index
        hgt = cell["ml_hgt"] if "ml_hgt" in ds.data_vars else None
        vals = var.isel({tdim: times}).transpose(tdim, "k", "cell").values.astype(float)
        vmin, vmax = finite_range(vals)
        outdir = args.outdir or f"xsec_{args.var}_lat{int(args.xsec)}"
        os.makedirs(outdir, exist_ok=True)
        frames = []
        for ti, t in enumerate(times):
            sec = vals[ti][:, idx]                          # (nk, nx)
            if hgt is not None:
                y = hgt.isel({tdim: t}).transpose("k", "cell").values.astype(float)[:, idx].mean(1) / 1e3
                ylabel = "height (km)"
            else:
                y = np.arange(nk); ylabel = "level index k"
            fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
            m = ax.pcolormesh(lon, y, sec, cmap=args.cmap, vmin=vmin, vmax=vmax, shading="auto")
            ax.set_xlabel("longitude"); ax.set_ylabel(ylabel)
            ax.set_title(f"{args.var}  lon-height @ lat={args.xsec:g}  time={t}")
            fig.colorbar(m, ax=ax, shrink=0.85, label=args.var)
            fp = os.path.join(outdir, f"{args.var}_lat{int(args.xsec)}_t{t:04d}.png")
            fig.savefig(fp, dpi=args.dpi); plt.close(fig); frames.append(fp); print("frame:", fp)
        _stitch(frames, args, outdir, f"{args.var}_lat{int(args.xsec)}")
        return

    # ---------------- horizontal map (2D or 3D@k) ----------------
    lon_plot = np.linspace(-180, 180, args.nx, endpoint=False) + 180.0 / args.nx
    lat = np.linspace(-90, 90, args.ny, endpoint=False) + 90.0 / args.ny
    LONp, LAT = np.meshgrid(np.deg2rad(lon_plot), np.deg2rad(lat))
    LONs, _ = np.meshgrid(np.deg2rad(lon_plot + args.lon0), np.deg2rad(lat))
    P = np.stack([(np.cos(LAT) * np.cos(LONs)).ravel(),
                  (np.cos(LAT) * np.sin(LONs)).ravel(), np.sin(LAT).ravel()], axis=-1)
    if args.method == "nearest":
        _, idx = tree.query(P); resample = lambda v: v[idx].reshape(args.ny, args.nx)
    else:
        dist, idx = tree.query(P, k=6); w = 1.0 / np.maximum(dist, 1e-12); w /= w.sum(-1, keepdims=True)
        resample = lambda v: (v[idx] * w).sum(-1).reshape(args.ny, args.nx)

    if has_k:
        k = args.k if args.k is not None else ds.sizes["k"] // 2
        var = var.isel(k=k); ktag = f"_k{k}"; ktitle = f"  k={k}"
    else:
        ktag = ""; ktitle = ""
    sel = var.isel({tdim: times}).values.astype(float)
    vmin, vmax = finite_range(sel)
    outdir = args.outdir or f"frames_{args.var}{ktag}"
    os.makedirs(outdir, exist_ok=True)
    frames = []
    for t in times:
        img = resample(var.isel({tdim: t}).values.astype(float))
        if args.projection == "mollweide":
            fig, ax = plt.subplots(figsize=(9, 5.0), constrained_layout=True,
                                   subplot_kw={"projection": "mollweide"})
            m = ax.pcolormesh(LONp, LAT, img, cmap=args.cmap, vmin=vmin, vmax=vmax,
                              shading="auto", rasterized=True)
            ax.grid(True, linewidth=0.3, color="0.6"); ax.set_xticklabels([])
        else:
            fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
            m = ax.imshow(img, extent=[args.lon0 - 180, args.lon0 + 180, -90, 90],
                          origin="lower", aspect="auto", cmap=args.cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
            if args.coastlines:
                try:
                    import cartopy.feature as cf
                    ax.add_geometries(list(cf.COASTLINE.geometries()), crs=None,
                                      facecolor="none", edgecolor="#333", linewidth=0.5)
                except Exception as e:
                    print(f"  (coastlines skipped: {e})")
        ax.set_title(f"{args.var}{ktitle}  time={t}")
        fig.colorbar(m, ax=ax, shrink=0.85, label=args.var)
        fp = os.path.join(outdir, f"{args.var}{ktag}_t{t:04d}.png")
        fig.savefig(fp, dpi=args.dpi); plt.close(fig); frames.append(fp); print("frame:", fp)
    _stitch(frames, args, outdir, f"{args.var}{ktag}")


def _stitch(frames, args, outdir, stem):
    if len(frames) <= 1:
        print("only one frame -> no movie"); return
    movie = args.movie or os.path.join(outdir, f"{stem}.mp4")
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


if __name__ == "__main__":
    main()
