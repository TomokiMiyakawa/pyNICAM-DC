"""Hermetic end-to-end dynamics smoke test (numpy backend, no downloaded data).

This is the first CI test that runs the *whole dynamical core*, not just the pure
kernels. It runs the Jablonowski-Williamson baroclinic-wave case for a couple of
RK3 steps on the numpy backend and asserts the prognostic output is physically
sane (finite, positive density/energy).

Why it is hermetic (needs no dataset download):
  * The case is initialised with ``input_io_mode = 'IDEAL'`` (analytic Jablonowski
    init), so it does NOT read the ~169 MB restart file.
  * The only input is the horizontal boundary + vertical grid under
    ``pynicamdc/test/case2/json_gl05rl01pe08/`` (~8 MB), which is committed to git.
  * The driver config ``nhm_driver_jw_ideal_smoke.toml`` (also committed) is the
    tracked ``nhm_driver.toml`` with three lines changed: IDEAL init, lstep_max=2,
    PRGout_interval=2.

The case is decomposed for 8 ranks (the committed grid is pe08), so the run uses
``mpirun -n 8 --oversubscribe`` to stay portable on 2-4 core CI runners.

The test self-skips cleanly when its prerequisites are absent (no MPI launcher,
no zarr<3 / run deps, or when pytest is itself already running inside an MPI rank),
so it is harmless in the minimal unit-test environment and only actually runs in
the dedicated numpy end-to-end CI job.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DRIVER = os.path.join(_REPO_ROOT, "pynicamdc", "nhm", "driver", "driver-dc.py")
_CASE_DIR = os.path.join(_REPO_ROOT, "pynicamdc", "test", "case2")
_SMOKE_CNF = "./case/config/nhm_driver_jw_ideal_smoke.toml"
_NRANKS = 8  # the committed grid is pe08


def _under_mpi() -> bool:
    """True if pytest itself was launched inside an MPI rank (e.g. `mpirun -np 1
    python -m pytest`). Spawning a nested `mpirun` from there would deadlock, so
    we skip instead."""
    return any(
        os.environ.get(v) not in (None, "", "1")
        for v in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK")
    ) or "OMPI_COMM_WORLD_RANK" in os.environ


def _require_prereqs():
    if _under_mpi():
        pytest.skip("running inside an MPI rank; would nest mpirun -- skip")
    for mod in ("toml", "xarray", "dask", "numcodecs"):
        pytest.importorskip(mod, reason=f"{mod} not installed (run env: requirements-run.txt)")
    zarr = pytest.importorskip("zarr", reason="zarr not installed (run env)")
    major = int(zarr.__version__.split(".")[0])
    if major >= 3:
        pytest.skip(f"zarr {zarr.__version__} >= 3; the model needs zarr<3 (see requirements-run.txt)")
    if shutil.which("mpirun") is None and shutil.which("mpiexec") is None:
        pytest.skip("no MPI launcher (mpirun/mpiexec) on PATH")


def _mpi_cmd(nranks):
    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    assert launcher is not None  # guaranteed by _require_prereqs
    cmd = [launcher, "-n", str(nranks)]
    # Open MPI: allow more ranks than cores (CI runners are 2-4 vCPU).
    if launcher.endswith("mpirun"):
        cmd.append("--oversubscribe")
    return cmd


@pytest.mark.slow
def test_jw_ideal_dynamics_smoke_numpy(tmp_path):
    _require_prereqs()

    # Per-run sandbox with the case data symlinked in as ./case (the config uses
    # ./case/... relative paths).
    rundir = tmp_path / "run"
    rundir.mkdir()
    (rundir / "case").symlink_to(_CASE_DIR)
    (rundir / "driversettings.toml").write_text(
        "[driver]\n"
        'backend = "numpy"\n'
        'precision = "float64"\n'
        f'nhm_driver_cnf = "{_SMOKE_CNF}"\n'
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    cmd = _mpi_cmd(_NRANKS) + [
        sys.executable, "-u", _DRIVER,
        "--driver-setting", "driversettings.toml",
    ]
    proc = subprocess.run(
        cmd, cwd=rundir, env=env, capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, (
        f"driver exited {proc.returncode}\n--- stdout tail ---\n"
        f"{proc.stdout[-3000:]}\n--- stderr tail ---\n{proc.stderr[-3000:]}"
    )
    assert "peacefully done" in proc.stdout, (
        f"run did not finish cleanly\n--- stdout tail ---\n{proc.stdout[-3000:]}"
    )

    # Inspect the output store: prognostic density (RHOG) and total energy (RHOGE)
    # must be finite and strictly positive everywhere.
    import zarr
    store = rundir / "testout_tmp.zarr"
    assert store.is_dir(), f"no output store written at {store}"
    g = zarr.open(str(store), mode="r")
    for field in ("RHOG", "RHOGE"):
        arr = np.asarray(g[field])
        assert arr.size > 0, f"{field} is empty"
        assert np.all(np.isfinite(arr)), f"{field} has non-finite values"
        assert np.all(arr > 0.0), f"{field} has non-positive values (min={arr.min():.3e})"
