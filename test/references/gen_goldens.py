"""Regenerate the committed numpy goldens for the parity kernels.

Usage:
    PYTHONPATH=<repo-root> python test/references/gen_goldens.py

Each of the xp-clean parity kernels (test_kernels._PARITY_CASES) has no
independent reference transcription, so its numpy output is snapshotted here to
test/references/goldens/<id>.npz. The golden test (test_kernel_numpy_golden)
then guards the numpy output against regression in EVERY CI job (no jax needed).

Regenerate ONLY when a kernel's numpy output legitimately changes, and justify
the change in the commit (a golden diff means the numpy result moved). The check
is tight-tolerance, not bit-exact, so cross-numpy-version ULP noise does not
force regeneration -- only a real O(1e-3)+ change would.
"""
from __future__ import annotations

import os

import numpy as np

import test.test_kernels as tk  # module-level import pulls in no jax

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")


def main():
    os.makedirs(_GOLDEN_DIR, exist_ok=True)
    for kid, ref_name, driver in tk._PARITY_CASES:
        m = tk._load_ref(ref_name)
        out = tk._as_tuple(driver(m, np))
        arrs = {f"out{i}": np.asarray(a) for i, a in enumerate(out)}
        np.savez_compressed(os.path.join(_GOLDEN_DIR, kid + ".npz"), **arrs)
        print(f"  {kid:16s} -> {[tuple(arrs[f'out{i}'].shape) for i in range(len(arrs))]}")
    print(f"wrote {len(tk._PARITY_CASES)} goldens to {_GOLDEN_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())
