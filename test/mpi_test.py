"""Smoke test for the MPI runtime the model relies on."""


def test_mpi4py_world_available():
    # CI installs libopenmpi-dev + mpi4py; importing and querying COMM_WORLD
    # verifies the MPI layer is functional even when run on a single rank.
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.Get_size() >= 1
    assert 0 <= comm.Get_rank() < comm.Get_size()
