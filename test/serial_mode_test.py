"""Serial (no-MPI) mode: the model must import and behave with mpi4py absent.

Full-model equivalence is covered by running any tier2 case through
tutorial/runs/<case>/ with mpi4py blocked (validated bit-identical to
mpirun -np 1 on the gw case). This unit test guards the import path and the
serial stub semantics cheaply: it hard-blocks mpi4py in a subprocess, imports
mod_process, and exercises the stub surface the model actually uses.
"""
import os
import subprocess
import sys

_SNIPPET = r"""
import sys
class _BlockMPI:
    def find_spec(self, name, path=None, target=None):
        if name == "mpi4py" or name.startswith("mpi4py."):
            raise ImportError("mpi4py blocked (serial-mode test)")
        return None
sys.meta_path.insert(0, _BlockMPI())

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


def test_serial_mode_without_mpi4py():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=repo)
    out = subprocess.run([sys.executable, "-c", _SNIPPET], env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, f"stderr:\n{out.stderr}"
    assert "SERIAL-MODE-OK" in out.stdout
