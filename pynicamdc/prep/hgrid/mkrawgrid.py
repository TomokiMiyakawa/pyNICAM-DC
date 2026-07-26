"""Generate the icosahedral horizontal grid and write the model boundary npz.

Faithful port of nicamdc prg_mkrawgrid + prg_mkhgrid (global standard case):
    MKGRD_standard -> MKGRD_spring -> MKGRD_gravcenter -> boundary npz
(prerotate/stretch/shrink/rotate are reduced-planet options, not ported.)

Run from prep/hgrid/:
    python mkrawgrid.py --comm serial                 # 1 rank, login-node safe
    mpirun -n N python mkrawgrid.py --comm mpi ...    # pe>1 decompositions
The [rgnmngparam] mnginfo toml must match the rank count (see prep/mnginfo).
"""
import sys
import os
import argparse
import toml

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, "../.."))
repo_root = os.path.abspath(os.path.join(script_dir, "../../.."))
sys.path.insert(0, repo_root)
sys.path.insert(0, script_dir)

ap = argparse.ArgumentParser(description="Generate the icosahedral hgrid (standard + spring + gravcenter) and write the model boundary npz")
ap.add_argument("--config", default='../../case/config/mkrawgrid.toml')
ap.add_argument("--comm", default='auto', choices=['auto', 'serial', 'mpi'],
                help="serial: no mpi4py (login-node safe, 1 rank); mpi: required for pe>1")
ap.add_argument("--zarr-raw", action='store_true',
                help="also write the per-region raw-grid zarr (pre-gravcenter)")
ap.add_argument("--spring-loop", action='store_true',
                help="use the original scalar-loop spring solver (slow; bit-compare reference)")
args = ap.parse_args()

# comm + backend must be decided BEFORE the first mod_process / share import
from pynicamdc.share import comm_mode
comm_mode.set_mode(args.comm)

_single = toml.load(args.config)['param_mkgrd']['mkgrd_precision_single']
from pynicamdc.share.mod_backend import backend as bk
bk.configure("numpy", "float32" if _single else "float64")

from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_prof import prf
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_const import Const
from pynicamdc.share.mod_comm import Comm
from pynicamdc.share.mod_gtl import Gtl

from mod_mkgrd import Mkgrd

#  main program start
intoml = args.config

mkg = Mkgrd(intoml)
rdtype = bk.ndtype
cnst = Const(rdtype)
gtl = Gtl()
comm = Comm()

# ---< MPI start >---
comm_world = prc.prc_mpistart()
is_master = (prc.prc_myrank == 0)
print(f"Hello, world! from rank {prc.prc_myrank} out of {prc.prc_nprocs}")

std.io_setup('pyNICAM-DC', intoml)
std.io_log_setup(prc.prc_myrank, is_master)

prf.PROF_setup(intoml, rdtype)
prf.PROF_setprefx("INIT")
prf.PROF_rapstart("Initialize", 0)

cnst.CONST_setup(rdtype, intoml)
adm.ADM_setup(intoml)
comm.COMM_setup(intoml)
mkg.mkgrd_setup(rdtype)

prf.PROF_rapend("Initialize", 0)
prf.PROF_setprefx("MAIN")
prf.PROF_rapstart("Main_MKGRD", 0)

prf.PROF_rapstart("MKGRD_standard", 0)
mkg.mkgrd_standard(rdtype, cnst, comm)
prf.PROF_rapend("MKGRD_standard", 0)
print("mkgrd_standard done")

prf.PROF_rapstart("MKGRD_spring", 0)
mkg.mkgrd_spring(rdtype, cnst, comm, gtl, vectorized=not args.spring_loop)
prf.PROF_rapend("MKGRD_spring", 0)
print("mkgrd_spring done")

if args.zarr_raw:
    import zarr
    p = prc.prc_myrank
    for l in range(mkg.GRD_x.shape[3]):
        region = adm.RGNMNG_lp2r[l, p]
        zname = "../../case/prepdata/" + mkg.mkgrd_out_basename + ".zarr" + f"{region:08d}"
        zarr_store = zarr.open(zname, mode="w", shape=mkg.GRD_x[:, :, 0, l, :].shape, dtype=rdtype)
        zarr_store[:, :, :] = mkg.GRD_x[:, :, 0, l, :]
        zarr_store.attrs["units"] = "xyz Cartesian coordinate unit globe"
        zarr_store.attrs["description"] = "raw grid data"
        zarr_store.attrs["glevel"] = adm.ADM_glevel
        zarr_store.attrs["rlevel"] = adm.ADM_rlevel
        zarr_store.attrs["region"] = f"{region:08d}"
        zarr_store.attrs["cnfs"] = mkg.cnfs

# hgrid finalization (Fortran prg_mkhgrid): gravitational-center pass fills
# GRD_xt (triangle vertices) and recenters GRD_x, then the boundary npz the
# model reads (hgrid_io_mode="npz") is written per rank.
prf.PROF_rapstart("MKGRD_gravcenter", 0)
mkg.mkgrd_gravcenter(rdtype, cnst, comm)
prf.PROF_rapend("MKGRD_gravcenter", 0)
print("mkgrd_gravcenter done")

out_prefix = "../../case/prepdata/" + mkg.mkgrd_out_basename + ".pe"
mkg.mkgrd_output_hgrid_npz(out_prefix, rdtype)

prf.PROF_rapend("Main_MKGRD", 0)
prf.PROF_rapreport()

prc.prc_mpifinish(std.io_l, std.fname_log)

print("peacefully done")
