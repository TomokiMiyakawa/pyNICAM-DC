"""comm mode selection (mpi / serial / auto) -- see pynicamdc/share/comm_mode.py.

Full-model equivalence is covered at tier2 level: the gw case run with
comm='serial' (and with mpi4py hard-blocked under 'auto') is bit-identical
to the normal mpirun -np 1 run. These unit tests guard the selection policy
and the serial stub semantics cheaply, each in a subprocess.
"""
import os
import subprocess
import sys

_BLOCK = r"""
import sys
class _BlockMPI:
    def find_spec(self, name, path=None, target=None):
        if name == "mpi4py" or name.startswith("mpi4py."):
            raise ImportError("mpi4py blocked (serial-mode test)")
        return None
sys.meta_path.insert(0, _BlockMPI())
"""

_STUB_CHECKS = r"""
import numpy as np
from pynicamdc.share.mod_process import prc, MPI, mpi_available

assert not mpi_available
assert prc.prc_myrank == 0 and prc.prc_nprocs == 1 and prc.prc_ismaster
assert prc.prc_mpi_alive is False

# collectives degenerate to identity
a = np.arange(6.0); b = np.empty_like(a)
prc.comm_world.Allreduce(a, b, op=MPI.MAX)
assert (a == b).all()
assert prc.comm_world.allreduce(7, op=MPI.MAX) == 7
assert prc.comm_world.bcast({"x": 1}) == {"x": 1}

# self-directed p2p via the mailbox (the mod_grd.GRD_gen_plgrid pattern),
# both post orders
recv = np.zeros(3)
prc.comm_world.Irecv(recv, source=0, tag=5)
prc.comm_world.Isend(np.array([1.0, 2.0, 3.0]), dest=0, tag=5)
MPI.Request.Waitall([None, None])
assert (recv == [1.0, 2.0, 3.0]).all()

prc.comm_world.Isend(np.array([9.0]), dest=0, tag=1)
recv2 = np.zeros(1)
prc.comm_world.Irecv(recv2, source=0, tag=1)
assert recv2[0] == 9.0

# constants / timing surface used by mod_comm
assert MPI.REQUEST_NULL is None
assert MPI.Wtime() > 0.0
print("SERIAL-MODE-OK")
"""


def _run(snippet):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=repo)
    return subprocess.run([sys.executable, "-c", snippet], env=env,
                          capture_output=True, text=True, timeout=120)


def test_auto_falls_back_to_serial_without_mpi4py():
    out = _run(_BLOCK + _STUB_CHECKS + r"""
from pynicamdc.share import comm_mode
assert comm_mode.SELECTED.startswith("serial (auto")
""")
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    assert "SERIAL-MODE-OK" in out.stdout


def test_serial_requested_ignores_installed_mpi4py():
    # no blocker: mpi4py may be importable, but comm='serial' must not touch it
    out = _run(r"""
from pynicamdc.share import comm_mode
comm_mode.set_mode("serial")
import sys
""" + _STUB_CHECKS + r"""
assert "mpi4py" not in sys.modules, "serial mode must not import mpi4py"
assert comm_mode.SELECTED == "serial (requested)"
""")
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    assert "SERIAL-MODE-OK" in out.stdout


def test_mpi_requested_fails_loudly_without_mpi4py():
    out = _run(_BLOCK + r"""
from pynicamdc.share import comm_mode
comm_mode.set_mode("mpi")
try:
    from pynicamdc.share.mod_process import prc
except ImportError as e:
    assert "comm='mpi' was requested" in str(e)
    print("LOUD-FAILURE-OK")
else:
    raise AssertionError("must not silently fall back to serial")
""")
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    assert "LOUD-FAILURE-OK" in out.stdout
