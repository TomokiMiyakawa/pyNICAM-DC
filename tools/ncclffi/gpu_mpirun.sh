#!/bin/bash
unset OMPI_MCA_mca_base_env_list
X=(--mca coll ^hcoll -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH)
for v in $(compgen -v | grep -E '^(PYNICAM_|MPI4JAX_|XLA_PYTHON_|XLA_FLAGS|HCOLL_ENABLE|JAX_PLATFORMS|NCCL_)'); do X+=(-x "$v"); done
exec mpirun "${X[@]}" "$@"
