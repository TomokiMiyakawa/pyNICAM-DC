"""Make the icosahedral -> lat-lon remap table (port of nicamdc mkllmap).

Faithful port of mod_latlon.f90 mkrelmap_ico2ll: for every lat-lon point, find
the icosahedral triangle (TI/TJ of some cell of some region) that contains it
via gnomonic projection onto the triangle plane, and store the 3 vertex cells
+ spherical-area barycentric weights. Lat-lon grids: EQUIDIST or GAUSSIAN
(Newton solver), optional half-cell lon offset.

Standalone: reads the model boundary files (json or npz, all ranks) + mnginfo
toml directly -- no MPI, no model state. Output is a single npz table (see
OUTPUT KEYS below) consumed by tools/zarr2ll.py; indices are 0-based python
(i, j) grid coordinates, NOT Fortran suf().

Exact-pole ll points (|r0 - pole| < 1e-15): handled like the Fortran rgn4pl
block; the pole-carrying regions are detected from the grid data (halo corner
== +-pole). With cell-centered (offset) latitudes the case never occurs.

Run from prep/llmap/:  python mkllmap.py --config my.toml
Config [mkllmap]: hgrid_fname, hgrid_io_mode(json|npz), mnginfo, glevel,
rlevel, imax, jmax, latlon_type(EQUIDIST|GAUSSIAN), lonmin_deg, lonmax_deg,
latmin_deg, latmax_deg, lon_offset, polar_limit_deg, output_fname.
"""
import argparse
import json
import os
import sys

import numpy as np
import toml

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, "../../..")))
from pynicamdc.share.mod_vector import Vect

vect = Vect()

D2R = np.pi / 180.0
EPS_JUDGE = 1.0e-18   # marginal value for inner products
EPS_LATLON = 1.0e-2   # marginal square near grid points [rad]
EPS_VERTEX = 1.0e-15  # marginal value for vertex hit


def load_boundary(prefix, io_mode, nprocs, g1d, rgn_of_rank):
    """Return dict rgnid -> (g1d, g1d, 3) grd_x array ([i, j] indexing)."""
    grids = {}
    for p in range(nprocs):
        fname = f"{prefix}{p:08d}." + ("json" if io_mode == "json" else "npz")
        if io_mode == "json":
            data = {k: np.asarray(v, dtype=np.float64) for k, v in json.load(open(fname)).items()}
        else:
            data = np.load(fname)
        rgns = rgn_of_rank[p]
        x = np.stack([np.asarray(data[k]).reshape(len(rgns), g1d, g1d)
                      for k in ("grd_x_x", "grd_x_y", "grd_x_z")], axis=-1)  # (l, j, i, 3)
        for ll, r in enumerate(rgns):
            grids[r] = x[ll].transpose(1, 0, 2)  # -> [i, j, 3]
    return grids


def xyz2latlon(x):
    """Vectorized VECTR_xyz2latlon for unit-ish vectors, x: (..., 3).

    Pole halo cells: in Fortran the per-point lat/lon field (GMTR_p_ll) is
    computed on interior cells only and halo cells are COMM-filled from the
    OWNER, so the pole halo carries the exact-pole value (lat=+-90, lon=0).
    A locally stored pole coordinate is only pole-ish (|x|,|y| ~ 1e-15), which
    would flip lon to ~180 via arccos and corrupt the triangle lon window --
    so cells that close to the axis are snapped to the owner's value here.
    Non-pole halo cells are bit-identical to the owner's computation (same
    xyz, same formula), so no other COMM emulation is needed.
    """
    z = np.clip(x[..., 2] / np.sqrt((x ** 2).sum(-1)), -1.0, 1.0)
    lat = np.arcsin(z)
    lh = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        lon = np.arccos(np.clip(x[..., 0] / lh, -1.0, 1.0))
    lon = np.where(lh < 1.0e-10, 0.0, np.where(x[..., 1] < 0.0, -lon, lon))
    lat = np.where(lh < 1.0e-10, np.copysign(np.pi / 2, x[..., 2]), lat)
    return lat, lon


def set_equidist(imax, jmax, lonmin, lonmax, latmin, latmax, lon_offset):
    j = np.arange(1, jmax + 1, dtype=np.float64)
    lat = latmin + (latmax - latmin) / jmax * (j - 0.5)
    i = np.arange(1, imax + 1, dtype=np.float64)
    off = 0.5 if lon_offset else 1.0
    lon = lonmin + (lonmax - lonmin) / imax * (i - off)
    return lon, lat


def set_gaussian(imax, jmax, lonmin, lonmax, lon_offset):
    eps = np.finfo(np.float64).eps * 4.0
    mu = np.empty(jmax)
    for j in range(1, jmax + 1):
        mu0 = np.sin(np.pi * (jmax + 1 - 2 * j) / (2 * jmax + 1))
        while True:
            P0 = np.empty(jmax + 1)
            P0[0], P0[1] = 1.0, mu0
            for n in range(1, jmax):
                P0[n + 1] = ((2 * n + 1) * mu0 * P0[n] - n * P0[n - 1]) / (n + 1)
            dP0 = jmax * (P0[jmax - 1] - mu0 * P0[jmax]) / (1 - mu0 * mu0)
            dmu = P0[jmax] / dP0
            mu0 = mu0 - dmu
            if abs(dmu) < eps:
                mu[j - 1] = mu0
                break
    lat = -np.arcsin(mu)
    i = np.arange(1, imax + 1, dtype=np.float64)
    off = 0.5 if lon_offset else 1.0
    lon = lonmin + (lonmax - lonmin) / imax * (i - off)
    return lon, lat


def triangle_area_on_sphere(a, b, c):
    """VECTR_triangle ON_SPHERE (l'Huillier), radius 1 -- the radius cancels
    in the weight ratios. Reuses the model's ported implementation."""
    return vect.VECTR_triangle(a, b, c, "ON_SPHERE", 1.0, None, np.float64)


def main():
    ap = argparse.ArgumentParser(description="icosahedral -> lat-lon remap table (nicamdc mkllmap port)")
    ap.add_argument("--config", default="../../case/config/mkllmap.toml")
    args = ap.parse_args()
    cnf = toml.load(args.config)["mkllmap"]

    glevel, rlevel = cnf["glevel"], cnf["rlevel"]
    imax, jmax = cnf["imax"], cnf["jmax"]
    latlon_type = cnf.get("latlon_type", "EQUIDIST")
    lon_offset = cnf.get("lon_offset", True)
    lonmin, lonmax = cnf.get("lonmin_deg", -180.0) * D2R, cnf.get("lonmax_deg", 180.0) * D2R
    latmin, latmax = cnf.get("latmin_deg", -90.0) * D2R, cnf.get("latmax_deg", 90.0) * D2R
    polar_limit = abs(cnf.get("polar_limit_deg", 89.0)) * D2R

    mng = toml.load(cnf["mnginfo"])
    nprocs = mng["PROC_INFO"]["NUM_OF_PROC"]
    rgn_of_rank = {int(pe["PEID"]): list(pe["MNG_RGNID"]) for pe in mng["RGN_MNG_INFO"].values()}
    nrgn = mng["RGN_INFO"]["NUM_OF_RGN"]

    g1d = 2 ** (glevel - rlevel) + 2
    gmin, gmax = 1, g1d - 2

    grids = load_boundary(cnf["hgrid_fname"], cnf.get("hgrid_io_mode", "npz"),
                          nprocs, g1d, rgn_of_rank)
    assert len(grids) == nrgn, f"regions read {len(grids)} != {nrgn}"

    if latlon_type == "EQUIDIST":
        lon, lat = set_equidist(imax, jmax, lonmin, lonmax, latmin, latmax, lon_offset)
    elif latlon_type == "GAUSSIAN":
        lon, lat = set_gaussian(imax, jmax, lonmin, lonmax, lon_offset)
    else:
        raise SystemExit(f"unknown latlon_type {latlon_type}")

    coslat, sinlat = np.cos(lat), np.sin(lat)
    coslon, sinlon = np.cos(lon), np.sin(lon)
    # all candidate target points on the sphere, (jmax, imax, 3)
    R0 = np.empty((jmax, imax, 3))
    R0[..., 0] = coslat[:, None] * coslon[None, :]
    R0[..., 1] = coslat[:, None] * sinlon[None, :]
    R0[..., 2] = sinlat[:, None]

    out = {k: [] for k in ("rgnid", "t", "lon_idx", "lat_idx",
                           "i1", "j1", "i2", "j2", "i3", "j3", "w1", "w2", "w3")}

    def emit(rgn, t, jj, ii, ijs, w):
        out["rgnid"].append(rgn); out["t"].append(t)
        out["lon_idx"].append(ii); out["lat_idx"].append(jj)
        for n, (pi, pj) in enumerate(ijs, start=1):
            out[f"i{n}"].append(pi); out[f"j{n}"].append(pj)
        for n, wn in enumerate(w, start=1):
            out[f"w{n}"].append(wn)

    two_pi = 2.0 * np.pi
    for rgn in range(nrgn):
        x = grids[rgn]                       # [i, j, 3]
        plat, plon = xyz2latlon(x)           # per-point lat/lon (halo included)

        for jc in range(gmin, gmax + 1):
            for ic in range(gmin, gmax + 1):
                for t in (0, 1):  # TI, TJ
                    if t == 0:
                        ijs = [(ic, jc), (ic + 1, jc), (ic + 1, jc + 1)]
                    else:
                        ijs = [(ic, jc), (ic + 1, jc + 1), (ic, jc + 1)]
                    r1, r2, r3 = (x[pi, pj] for (pi, pj) in ijs)
                    lat123 = np.array([plat[pi, pj] for (pi, pj) in ijs])
                    lon123 = np.array([plon[pi, pj] for (pi, pj) in ijs])

                    latmax_l = lat123.max() + EPS_LATLON
                    latmin_l = lat123.min() - EPS_LATLON
                    if latmin_l > polar_limit:
                        latmax_l = np.pi
                    if latmax_l < -polar_limit:
                        latmin_l = -np.pi

                    lo = lon123.copy()
                    if lo.max() - lo.min() > np.pi:
                        lo = np.where(lo < 0, lo + two_pi, lo)
                    lonmax_l = lo.max() + EPS_LATLON
                    lonmin_l = lo.min() - EPS_LATLON

                    jsel = np.nonzero((lat >= latmin_l) & (lat <= latmax_l))[0]
                    if jsel.size == 0:
                        continue

                    near_pole = (np.abs(lat[jsel]) > polar_limit)
                    lon_ok = (((lon <= lonmax_l) & (lon >= lonmin_l))
                              | ((lon - two_pi <= lonmax_l) & (lon - two_pi >= lonmin_l))
                              | ((lon + two_pi <= lonmax_l) & (lon + two_pi >= lonmin_l)))

                    # candidate (j, i) pairs, j-outer i-inner (Fortran loop order)
                    jj, ii = np.meshgrid(jsel, np.arange(imax), indexing="ij")
                    mask = near_pole[:, None] | lon_ok[None, :]
                    jj, ii = jj[mask], ii[mask]
                    if jj.size == 0:
                        continue

                    r0 = R0[jj, ii]                       # (n, 3)
                    ip = r0 @ r1
                    keep = ip >= 0.0
                    jj, ii, r0 = jj[keep], ii[keep], r0[keep]
                    if jj.size == 0:
                        continue
                    v01 = r1[None, :] - r0                # BEFORE the plane mapping

                    nvec = np.cross(r2 - r1, r3 - r2)
                    nvec = nvec / np.sqrt((nvec ** 2).sum())
                    rn = nvec @ r1
                    rf = r0 @ nvec
                    r0m = r0 * (rn / rf)[:, None]

                    # Fortran VECTR_cross(v, a,b, c,d) = (b-a) x (d-c):
                    # judge12 = nvec . ((r2-r1) x (r1-r0))  etc.
                    judge12 = np.cross(r2 - r1, r1 - r0m) @ nvec
                    judge23 = np.cross(r3 - r2, r2 - r0m) @ nvec
                    judge31 = np.cross(r1 - r3, r3 - r0m) @ nvec
                    inside = (judge12 < EPS_JUDGE) & (judge23 < EPS_JUDGE) & (judge31 < EPS_JUDGE)

                    on_vertex = (~inside) & (t == 0) & (np.abs(v01) < EPS_VERTEX).all(axis=1)

                    for n in np.nonzero(inside)[0]:
                        a1 = triangle_area_on_sphere(r0m[n], r2, r3)
                        a2 = triangle_area_on_sphere(r0m[n], r3, r1)
                        a3 = triangle_area_on_sphere(r0m[n], r1, r2)
                        tot = a1 + a2 + a3
                        if not np.isfinite(tot):
                            raise SystemExit(f"NaN area at rgn={rgn} t={t} (i,j)=({ii[n]},{jj[n]})")
                        emit(rgn, t, jj[n], ii[n], ijs, (a1 / tot, a2 / tot, a3 / tot))
                    for n in np.nonzero(on_vertex)[0]:
                        emit(rgn, 0, jj[n], ii[n],
                             [(ic, jc), (ic + 1, jc), (ic + 1, jc + 1)], (1.0, 0.0, 0.0))

    # exact-pole points (Fortran rgn4pl block); pole-carrying region detected
    # from the grid data: the pole sits in the halo corner after COMM.
    for pole, corner in ((np.array([0.0, 0.0, 1.0]), (gmin, gmax + 1)),
                         (np.array([0.0, 0.0, -1.0]), (gmax + 1, gmin))):
        for rgn in range(nrgn):
            r1 = grids[rgn][corner[0], corner[1]]
            if np.abs(r1 - pole).max() > 1.0e-12:
                continue
            d = np.abs(r1[None, None, :] - R0).max(axis=-1)
            hit = (R0 @ r1 >= 0.0) & (d < EPS_VERTEX)
            for jjj, iii in zip(*np.nonzero(hit)):
                emit(rgn, -1, jjj, iii, [corner, corner, corner], (1.0, 0.0, 0.0))
            break

    npt = len(out["rgnid"])
    print(f"lat-lon points mapped: {npt} / {imax * jmax}")
    if npt != imax * jmax:
        print("WARNING: counted llgrid does not match imax*jmax (duplicates/misses)")

    arrays = {k: np.asarray(v) for k, v in out.items()}
    arrays.update(lat=lat, lon=lon, imax=imax, jmax=jmax,
                  glevel=glevel, rlevel=rlevel, g1d=g1d)
    np.savez(cnf["output_fname"], **arrays)
    print(f"wrote {cnf['output_fname']}")


if __name__ == "__main__":
    main()
