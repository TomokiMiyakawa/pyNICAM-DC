"""Loader for the Fortran simple-physics reference dump (ref_simple_physics.txt).

Parses the self-describing text written by simple_physics_ref.f90 into numpy
arrays. 2D arrays were flattened column-major-by-column (for ic: for il:),
so a flat run of nc*nl reshapes C-order to (nc, nl).

Usage:
    ref = load_ref("ref_simple_physics.txt")
    ref.meta            -> {'pcols':5, 'pver':30, 'dtime':1200.0}
    ref.shared['pmid']  -> (pcols, pver) input
    ref.shared['pint']  -> (pcols, pver+1)
    ref.configs['A_rj_noBryan_test0']['t_out'] -> (pcols, pver)
    ref.configs[name]['precl'] -> (pcols,)
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Ref:
    meta: dict = field(default_factory=dict)
    shared: dict = field(default_factory=dict)
    configs: dict = field(default_factory=dict)


def load_ref(path):
    ref = Ref()
    pcols = pver = None
    cur_cfg = None
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].strip()
        i += 1
        if not ln or ln.startswith("#"):
            continue
        tok = ln.split()
        if tok[0] == "META":
            if tok[1] == "pcols":
                pcols, pver = int(tok[3]), int(tok[4])
                ref.meta["pcols"], ref.meta["pver"] = pcols, pver
            elif tok[1] == "dtime":
                ref.meta["dtime"] = float(tok[2])
        elif tok[0] == "CONFIG":
            cur_cfg = tok[1]
            ref.configs[cur_cfg] = {}
        elif tok[0] == "ARRAY":
            name, cnt = tok[1], int(tok[2])
            vals = np.empty(cnt, dtype=np.float64)
            for j in range(cnt):
                vals[j] = float(lines[i].strip())
                i += 1
            arr = _reshape(name, vals, pcols, pver)
            if cur_cfg is None:
                ref.shared[name] = arr
            else:
                ref.configs[cur_cfg][name] = arr
        else:
            raise ValueError(f"unexpected token: {ln!r}")
    return ref


def _reshape(name, vals, pcols, pver):
    n = vals.size
    if pcols and n == pcols:                 # (pcols,)
        return vals
    if pcols and pver and n == pcols * pver:  # (pcols, pver)
        return vals.reshape(pcols, pver)
    if pcols and pver and n == pcols * (pver + 1):  # interface (pcols, pver+1)
        return vals.reshape(pcols, pver + 1)
    return vals


if __name__ == "__main__":
    import sys
    r = load_ref(sys.argv[1] if len(sys.argv) > 1 else "ref_simple_physics.txt")
    print("meta:", r.meta)
    print("shared arrays:", {k: v.shape for k, v in r.shared.items()})
    print("configs:", list(r.configs))
    for c, d in r.configs.items():
        print(f"  {c}: precl[0]={d['precl'][0]:.6e}  "
              f"t_out mean={d['t_out'].mean():.4f}")
