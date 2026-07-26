"""Tier-1 checks for the prep input-generation tools.

Fast, no dataset needed: mkmnginfo regenerates repo-bundled reference tomls;
mkvlayer reproduces the bundled Fortran-gold .dat bit-exactly. The hgrid chain
(spring iteration) is tier-2 scale and is covered by prep/hgrid/validate_hgrid.py.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MNGINFO_DIR = os.path.join(REPO, "pynicamdc/prep/mnginfo")
VGRID_DIR = os.path.join(REPO, "pynicamdc/prep/vgrid")


def _import_from(d, name):
    sys.path.insert(0, d)
    try:
        return __import__(name)
    finally:
        sys.path.remove(d)


@pytest.mark.parametrize("rl,prc,ref", [
    (0, 1, "tutorial/case/grid_gl05rl00pe01/rl00-prc01.toml"),
    (1, 8, "pynicamdc/test/case1/prepdata/rl01-prc000008.toml"),
])
def test_mkmnginfo_matches_reference(tmp_path, rl, prc, ref):
    toml = pytest.importorskip("toml")
    refpath = os.path.join(REPO, ref)
    if not os.path.exists(refpath):
        pytest.skip(f"reference missing: {ref}")
    mk_mod = _import_from(MNGINFO_DIR, "mkmnginfo")

    out = str(tmp_path / "gen.toml")
    mk = mk_mod.Mkmnginfo(rlevel=rl, prc_num=prc, output_fname=out)
    mk.generate_mngtab(rl, prc, out)

    got, want = toml.load(out), toml.load(refpath)

    def norm(d):
        d = {k: v for k, v in d.items() if k != "Title"}
        d["PROC_INFO"] = {k: v for k, v in d["PROC_INFO"].items() if k != "NUM_OF_MNG"}
        return d

    assert norm(got) == norm(want)


@pytest.mark.parametrize("kw,gold", [
    (dict(num_of_layer=30, layer_type="ULLRICH14", ztop=1.0e4), "vgrid30_ullrich14_gold.dat"),
    (dict(num_of_layer=30, layer_type="EVEN", ztop=1.2e4), "vgrid30_even_gold.dat"),
    (dict(num_of_layer=40, layer_type="GIVEN",
          infname=os.path.join(VGRID_DIR, "inputzdata/_vgrid_40L_exp.dat")), "vgrid40_given_gold.dat"),
])
def test_mkvlayer_bitexact_vs_fortran_gold(kw, gold):
    mkv = _import_from(VGRID_DIR, "mkvlayer")
    val = _import_from(VGRID_DIR, "validate_mkvlayer")

    n, zc, zh = val.read_dat(os.path.join(VGRID_DIR, "reference", gold))
    m = mkv.mkvlayer(outfname=os.devnull, **kw)
    m.generate_layers()
    assert np.array_equal(m.z_c, zc)
    assert np.array_equal(m.z_h, zh)
