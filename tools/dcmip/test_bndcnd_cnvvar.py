"""Validate BNDCND_pre_sfc + cnvvar_vh2uv + cnvvar_uv2vh against the Fortran golden.

The golden (bndcnd_cnvvar_ref.f90) runs verbatim copies of the reference inner
loops on flat (ij,k) data. Here we place that data into pyNICAM 4D arrays
(i=ijdim, j=1, k=kdim, l=1) with a per-point GMTR_p metric, call the numpy
ports, and compare. The ops are pointwise so the flat<->4D placement is exact.

Run (after building the golden):
    gfortran -O2 -ffree-line-length-none tools/dcmip/bndcnd_cnvvar_ref.f90 \
        -o tools/dcmip/bndcnd_cnvvar_ref.x && \
        tools/dcmip/bndcnd_cnvvar_ref.x 6 12 tools/dcmip/ref_bndcnd_cnvvar.txt
    python3 tools/dcmip/test_bndcnd_cnvvar.py
"""
import os
import sys
import types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

RTOL = 1e-11
ATOL = 1e-12

# inject a minimal mpi4py stub so the pyNICAM share modules import without MPI
import types as _t   # noqa: E402
if "mpi4py" not in sys.modules:
    _mpi = _t.ModuleType("mpi4py")
    _MPI = _t.SimpleNamespace(
        COMM_WORLD=_t.SimpleNamespace(Get_rank=lambda: 0, Get_size=lambda: 1),
        DOUBLE=None, FLOAT=None, INT=None, SUM=None, COMM_NULL=None,
    )
    _mpi.MPI = _MPI
    sys.modules["mpi4py"] = _mpi
    sys.modules["mpi4py.MPI"] = _MPI

import pynicamdc.share.mod_adm as adm_mod   # noqa: E402


def load(path):
    meta, arr = {}, {}
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i].strip(); i += 1
        if not ln or ln.startswith("#"):
            continue
        t = ln.split()
        if t[0] == "META":
            if t[1] == "ijdim":
                meta.update(ijdim=int(t[3]), kdim=int(t[4]))
            elif t[1] == "kmin_1based":
                meta["kmin1"] = int(t[2])
            elif t[1] == "GRAV":
                meta["GRAV"] = float(t[2])
        elif t[0] == "ARRAY":
            cnt = int(t[2])
            arr[t[1]] = np.array([float(lines[i + j]) for j in range(cnt)])
            i += cnt
    return meta, arr


def main():
    path = os.path.join(HERE, "ref_bndcnd_cnvvar.txt")
    if not os.path.exists(path):
        print("ERROR: ref_bndcnd_cnvvar.txt missing; build bndcnd_cnvvar_ref.f90 first.")
        return 2
    meta, A = load(path)
    ij, kd = meta["ijdim"], meta["kdim"]
    kmin = meta["kmin1"] - 1     # -> python index

    def col(name):     # (ij*kd,) -> (ij,1,kd,1)
        return A[name].reshape(ij, kd)[:, None, :, None]

    def hor(name):     # (ij,) -> (ij,1,1)  (i,j,l)
        return A[name].reshape(ij)[:, None, None]

    # --- configure a minimal adm singleton (K0=0, have_pl=False) ---
    adm = adm_mod.adm
    adm.ADM_K0 = 0
    adm.ADM_have_pl = False

    # --- GMTR_p (i,j,1,l,8) from the golden metric vectors ---
    from pynicamdc.share.mod_gmtr import Gmtr
    gmtr = Gmtr()
    gp = np.zeros((ij, 1, 1, 1, gmtr.GMTR_p_nmax))
    for nm, idx in (("IX", gmtr.GMTR_p_IX), ("IY", gmtr.GMTR_p_IY), ("IZ", gmtr.GMTR_p_IZ),
                    ("JX", gmtr.GMTR_p_JX), ("JY", gmtr.GMTR_p_JY), ("JZ", gmtr.GMTR_p_JZ)):
        gp[:, 0, 0, 0, idx] = A[nm]
    gmtr.GMTR_p = gp

    # --- grd stand-in with GRD_LAT (i,j,1,l) ---
    grd = types.SimpleNamespace(GRD_LAT=A["lat"].reshape(ij, 1, 1, 1))
    cnst = types.SimpleNamespace(CONST_GRAV=meta["GRAV"])

    from pynicamdc.nhm.share.mod_bndcnd import Bndc
    from pynicamdc.nhm.share.mod_cnvvar import Cnvv
    bndc, cnvv = Bndc(), Cnvv()

    fails = []

    def chk(name, got, ref_flat, shape2):
        ref = ref_flat.reshape(shape2)
        g = np.asarray(got).reshape(shape2)
        aerr = np.abs(g - ref)
        ok = np.all(aerr <= ATOL + RTOL * np.abs(ref))
        scale = max(np.abs(ref).max(), 1e-300)
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:8s} max|abs|={aerr.max():.3e} "
              f"max|abs|/max|ref|={aerr.max()/scale:.3e}")
        if not ok:
            fails.append(name)

    # ---- BNDCND_pre_sfc ----
    print("BNDCND_pre_sfc")
    rho_srf, pre_srf = bndc.BNDCND_pre_sfc(
        kmin, col("rho"), col("pre"), col("zg"), hor("z_srf"), cnst, np.float64)
    chk("rho_srf", rho_srf, A["rho_srf"], (ij,))
    chk("pre_srf", pre_srf, A["pre_srf"], (ij,))

    # ---- cnvvar_vh2uv (withcos=False) ----
    print("cnvvar_vh2uv")
    zpl = np.zeros((1, kd, 1))
    u, _, v, _ = cnvv.cnvvar_vh2uv(col("vx"), zpl, col("vy"), zpl, col("vz"), zpl,
                                   grd, gmtr, withcos=False)
    chk("u", u, A["u"], (ij, kd))
    chk("v", v, A["v"], (ij, kd))

    # ---- cnvvar_uv2vh (ucos=u, vcos=v; internal coslat=cos(lat)) ----
    print("cnvvar_uv2vh")
    vx2, _, vy2, _, vz2, _ = cnvv.cnvvar_uv2vh(col("u"), zpl, col("v"), zpl, grd, gmtr)
    chk("vx2", vx2, A["vx2"], (ij, kd))
    chk("vy2", vy2, A["vy2"], (ij, kd))
    chk("vz2", vz2, A["vz2"], (ij, kd))

    # round-trip identity: uv2vh(vh2uv(V)) == V (tangential winds)
    print("round-trip vh2uv->uv2vh")
    # note: vh2uv here withcos=False, uv2vh divides by cos(lat) -> feed ucos=u*cos
    ucos = u * np.cos(grd.GRD_LAT[:, :, 0, :])[:, :, None, :]
    vcos = v * np.cos(grd.GRD_LAT[:, :, 0, :])[:, :, None, :]
    rx, _, ry, _, rz, _ = cnvv.cnvvar_uv2vh(ucos, zpl, vcos, zpl, grd, gmtr)
    chk("rt_vx", rx, A["vx"], (ij, kd))
    chk("rt_vy", ry, A["vy"], (ij, kd))
    chk("rt_vz", rz, A["vz"], (ij, kd))

    print("\nRESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
