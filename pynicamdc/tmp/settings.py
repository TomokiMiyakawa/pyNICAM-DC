from collections import namedtuple

Setting = namedtuple("setting", ("default", "type", "description"))

def optional(type_):
    def wrapped(arg):
        if arg is None:
            return arg

        return type_(arg)

    return wrapped

PI = 3.14159265358979323846264338327950588

SETTINGS = {
    "identifier": Setting("UNNAMED", str, "Identifier of the current simulation"),
    # Model parameters
    "nx": Setting(0, int, "Grid points in zonal (x) direction"),
    "ny": Setting(0, int, "Grid points in meridional (y,j) direction"),
    "nz": Setting(0, int, "Grid points in vertical (z,k) direction"),
    "dt_mom": Setting(0.0, float, "Time step in seconds for momentum"),
    # Physical constants
    "pi": Setting(PI, float, "Pi"),
    }
