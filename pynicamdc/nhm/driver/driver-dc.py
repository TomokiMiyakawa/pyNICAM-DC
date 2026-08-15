#!/usr/bin/env python3
"""Command-line entry point.

    mpiexec -n 8 python3 -u driver-dc.py [--driver-setting ./driversettings.toml]

The startup sequence, time loop and teardown live in pynicamdc/api.py; this is the
CLI around them, and runs the same phases in the same order it always has.

Nothing from the model may be imported at the top of this file: backend, precision
and the mpi-vs-serial decision are fixed when pyNICAM is constructed, and the first
import of mod_process makes the mpi-vs-serial choice permanent. Ask the instance
(nicam.rank) instead of importing prc here.
"""

import argparse

from pynicamdc.api import pyNICAM

parser = argparse.ArgumentParser()
parser.add_argument(
    "--driver-setting",
    default="./driversettings.toml",
)
args = parser.parse_args()

# can set numpy to raise exceptions on floating point errors
#np.seterr(all='raise')
#np.seterr(under='ignore')

# ---<  main program start >---
print("driver_dc.py start")

nicam = pyNICAM(args.driver_setting)
nicam.initialize()
nicam.run()
nicam.finalize()

print("peacefully done:  rank ", nicam.rank)
