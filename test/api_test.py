"""pyNICAM object API -- see pynicamdc/api.py.

What is guarded here is the part that cannot be seen from a finished run: the
IMPORT ORDER. backend, precision and the mpi-vs-serial mode must be fixed before
anything imports mod_process, which decides mpi-vs-serial once, at import. A
regression there does not raise -- it silently runs the wrong mode -- so each
check runs in its own subprocess and inspects sys.modules directly.

Full-model equivalence is covered at tier2 level: driver-dc.py runs on this API,
and an API run split into run(3) chunks is bit-identical to one run() to lstep_max.
"""
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG = os.path.join(_REPO, "test", "case2", "config", "nhm_driver.toml")


def _run(snippet):
    env = dict(os.environ, PYTHONPATH=_REPO)
    return subprocess.run([sys.executable, "-c", snippet], env=env,
                          capture_output=True, text=True, timeout=120)


_HEAD = f"import sys\nCONFIG = {_CONFIG!r}\n"


def test_package_import_pulls_in_no_model_module():
    # `import pynicamdc` must stay free of the model (and of the api module): the
    # prep tools import the package, and an eager import here would make the
    # mpi-vs-serial decision behind the caller's back.
    out = _run(_HEAD + r"""
import pynicamdc
loaded = [m for m in sys.modules if m.startswith("pynicamdc.") and m != "pynicamdc._version"]
assert loaded == [], loaded
assert pynicamdc.pyNICAM.__name__ == "pyNICAM"          # resolved lazily, on use
assert "pynicamdc.share.mod_process" not in sys.modules
print("LAZY-IMPORT-OK")
""")
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    assert "LAZY-IMPORT-OK" in out.stdout


def test_construction_configures_before_any_model_import():
    out = _run(_HEAD + r"""
from pynicamdc import pyNICAM
n = pyNICAM(CONFIG, backend="numpy", precision="float32", comm="serial")

# the whole point: the settings are in place and mod_process is still unimported
assert "pynicamdc.share.mod_process" not in sys.modules
from pynicamdc.share import comm_mode
assert comm_mode.REQUESTED == "serial" and comm_mode.SELECTED is None
assert n.bk.type == "numpy" and n.bk.ndtype.__name__ == "float32"

# and the decision the settings asked for is the one taken at that import
assert n.rank == 0 and n.nprocs == 1
assert comm_mode.SELECTED == "serial (requested)"
assert "mpi4py" not in sys.modules
print("IMPORT-ORDER-OK")
""")
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    assert "IMPORT-ORDER-OK" in out.stdout


def test_conflicting_reconfiguration_is_refused():
    # One process, one model: backend/precision are bound to the kernels and the
    # comm mode is bound to an import. Silently keeping the first choice would run
    # a configuration nobody asked for.
    out = _run(_HEAD + r"""
from pynicamdc import pyNICAM
pyNICAM(CONFIG, backend="numpy", precision="float64", comm="serial")
for kw in ({"backend": "jax"}, {"precision": "float32"}):
    try:
        pyNICAM(CONFIG, comm="serial", **kw)
    except RuntimeError as e:
        assert "already configured" in str(e), e
    else:
        raise AssertionError(f"{kw} must be refused")

import pynicamdc.share.mod_process     # comm decision now taken and unrevisitable
try:
    pyNICAM(CONFIG, comm="mpi")
except RuntimeError as e:
    assert "already imported" in str(e), e
else:
    raise AssertionError("a conflicting comm mode must be refused")
print("GUARDS-OK")
""")
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    assert "GUARDS-OK" in out.stdout


def test_parameters_overlay_merges_without_touching_the_config():
    # initialize(parameters=...) reaches the setup routines through a merged copy,
    # because the config travels as a PATH that ~15 modules re-open themselves.
    out = _run(_HEAD + r"""
import os.path, toml
from pynicamdc import pyNICAM
n = pyNICAM(CONFIG, comm="serial")
base = toml.load(CONFIG)
merged = toml.load(n._resolve_config({"timeparam": {"lstep_max": 24},
                                      "ioparam": {"PRGout_name": "run1.zarr"}}))

assert merged["timeparam"]["lstep_max"] == 24
assert merged["ioparam"]["PRGout_name"] == "run1.zarr"
assert merged["timeparam"]["dtl"] == base["timeparam"]["dtl"]                  # sibling keys survive
assert merged["ioparam"]["PRGout_interval"] == base["ioparam"]["PRGout_interval"]
assert toml.load(CONFIG) == base                                               # the file itself is untouched
assert n._resolve_config(None) == CONFIG                                       # no overlay -> no copy

# admparam.rgnmngfname names the file [rgnmngparam] is read from, and in the
# shipped configs it points back at the config itself. Left as it is, the overlay
# would be read from the UN-MERGED file, so that one self-reference is retargeted
# to the merged copy -- and only that one.
assert merged["admparam"]["rgnmngfname"] == base["admparam"]["rgnmngfname"]    # not self-referential here (a CWD-relative spelling)

import tempfile
selfref = os.path.join(tempfile.mkdtemp(), "cnf.toml")
c = toml.load(CONFIG)
c["admparam"]["rgnmngfname"] = selfref
with open(selfref, "w") as f:
    toml.dump(c, f)
n2 = pyNICAM(selfref, comm="serial")
m2 = n2._resolve_config({"rgnmngparam": {"RGNMNG_out_fname": "elsewhere"}})
got = toml.load(m2)["admparam"]["rgnmngfname"]
assert os.path.realpath(got) == os.path.realpath(m2), got
assert toml.load(got)["rgnmngparam"]["RGNMNG_out_fname"] == "elsewhere"        # the override is now reachable
print("PARAMETERS-OK")
""")
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    assert "PARAMETERS-OK" in out.stdout


def test_phases_refuse_to_run_out_of_order():
    out = _run(_HEAD + r"""
from pynicamdc import pyNICAM
n = pyNICAM(CONFIG, comm="serial")
for phase in (n.run, n.write, n.finalize):
    try:
        phase()
    except RuntimeError as e:
        assert "initialize" in str(e), e
    else:
        raise AssertionError(f"{phase.__name__}() before initialize() must raise")
print("PHASE-ORDER-OK")
""")
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    assert "PHASE-ORDER-OK" in out.stdout
