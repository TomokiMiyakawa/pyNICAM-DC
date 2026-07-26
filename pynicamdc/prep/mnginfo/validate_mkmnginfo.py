#!/usr/bin/env python
"""Validate mkmnginfo.py against every previously validated mnginfo toml.

For each reference file (named rl<RL>-prc<N>.toml) the table is regenerated
with Mkmnginfo and compared as parsed dicts (key order ignored, Title ignored
-- it only differs in formatting). Any structural difference is a FAIL.

Run from prep/mnginfo/:
    python validate_mkmnginfo.py [ref_dir ...]
Default reference locations cover the repo test cases, the tutorial grid and
the validated sweep set.
"""
import glob
import os
import re
import sys
import tempfile

import toml

from mkmnginfo import Mkmnginfo

DEFAULT_REF_DIRS = [
    "../../test/case1/prepdata",
    "../../test/case2/prepdata",
    "../../test/case3/prepdata",
    "../../../tutorial/case/grid_gl05rl00pe01",
    "/work/gj37/c24028/workforclaude/fromwhale/pynicam-sweep/data/mnginfo",
]

PAT = re.compile(r"rl(\d+)-prc0*(\d+)\.toml$")


def normalize(d):
    """Comparison view: drop Title (formatting-only) and the global
    PROC_INFO.NUM_OF_MNG -- mod_adm reads only NUM_OF_PROC + the per-PE
    entries, and at least one validated legacy file (rl03-prc000064, used by
    the gl11 pe64 sweeps) carries a stale global value of 1."""
    out = {k: v for k, v in d.items() if k != "Title"}
    if "PROC_INFO" in out:
        out["PROC_INFO"] = {k: v for k, v in out["PROC_INFO"].items()
                            if k != "NUM_OF_MNG"}
    return out


def main(ref_dirs):
    refs = []
    for d in ref_dirs:
        refs += sorted(glob.glob(os.path.join(d, "rl*-prc*.toml")))
    if not refs:
        print("no reference tomls found")
        return 1

    seen, n_pass, n_fail = set(), 0, 0
    for ref in refs:
        m = PAT.search(os.path.basename(ref))
        if not m:
            continue
        rl, prc = int(m.group(1)), int(m.group(2))
        if (10 * 4**rl) % prc != 0:
            print(f"SKIP  {ref} (prc does not divide region count)")
            continue

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as tf:
            out = tf.name
        try:
            mk = Mkmnginfo(rlevel=rl, prc_num=prc, output_fname=out)
            mk.generate_mngtab(rl, prc, out)
            got, want = toml.load(out), toml.load(ref)
        finally:
            os.unlink(out)

        ok = normalize(got) == normalize(want)
        tag = "PASS" if ok else "FAIL"
        dup = " (dup config)" if (rl, prc) in seen else ""
        seen.add((rl, prc))
        print(f"{tag}  rl{rl:02d} prc{prc:<3d}  vs {ref}{dup}")
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            for k in set(got) | set(want):
                if k != "Title" and got.get(k) != want.get(k):
                    print(f"      first differing section: {k}")
                    break

    print(f"\n{n_pass} PASS, {n_fail} FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_REF_DIRS))
