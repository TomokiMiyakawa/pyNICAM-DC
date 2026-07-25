"""How pyNICAM decides between MPI and serial execution.

The decision is EXPLICIT and lives in exactly two places a reader must know:

  1. the run's driver-settings toml:      comm = "mpi" | "serial" | "auto"
     (next to `backend` and `precision`; the driver calls set_mode() with it
      BEFORE the first import of mod_process)
  2. mod_process, which reads REQUESTED once at import and acts:
        "serial" -> serial stub, mpi4py is never imported
        "mpi"    -> mpi4py required; ImportError propagates LOUDLY
                    (a broken environment must never silently run serial:
                     `srun -n 64` + missing mpi4py would otherwise start 64
                     independent rank-0 processes)
        "auto"   -> mpi4py if importable, else serial; the choice is logged

Entry points that never read a driver toml (prep tools, pytest) leave the
default "auto" untouched. Production launch scripts should pin comm = "mpi".
"""

REQUESTED = "auto"   # set by the driver before model imports; see docstring
SELECTED = None      # filled in by mod_process: "mpi" or "serial" (+ reason)


def set_mode(mode):
    global REQUESTED
    if mode not in ("mpi", "serial", "auto"):
        raise ValueError(f"comm must be 'mpi', 'serial' or 'auto', got {mode!r}")
    REQUESTED = mode
