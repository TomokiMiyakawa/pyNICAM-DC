#-------------------------------------------------------------------------------
#> Module generic tool
#
# @par Description
#         This module is for the generic subroutine, e.g., global mean.
#
# @author NICAM developers

import numpy as np
from pynicamdc.share.mod_adm import adm

class Gtl:

    _instance = None

    def __init__(self):
        pass


    def GTL_max(self, var, var_pl, kdim, kstart, kend, cnst, comm, rdtype):
        """Compute the global maximum value in the given 3D array."""

        # Vectorized over the interior window (i,j) in [ADM_gmin:ADM_gmax+1],
        # k in [kstart:kend+1], l in [0:ADM_lall]. Equivalent (bit-exact: max
        # only selects existing values) to the former 4-deep Python loop.
        sub = var[adm.ADM_gmin:adm.ADM_gmax + 1,
                  adm.ADM_gmin:adm.ADM_gmax + 1,
                  kstart:kend + 1,
                  0:adm.ADM_lall]
        vmax = np.max(sub) if sub.size else -cnst.CONST_HUGE

        # If ADM_have_pl is True, check additional values
        if adm.ADM_have_pl:
            sub_pl = var_pl[adm.ADM_gslf_pl, kstart:kend + 1, 0:adm.ADM_lall_pl]
            if sub_pl.size:
                vmax = max(vmax, np.max(sub_pl))

        # Perform global max communication across processes
        vmax_g = comm.Comm_Stat_max(vmax)

        return vmax_g

    def GTL_min(self, var, var_pl, kdim, kstart, kend, cnst, comm, rdtype, nonzero=False):
        """Compute the global minimum value in the given 3D array."""

        # Vectorized over the same interior window as GTL_max (bit-exact: min
        # only selects existing values). nonzero=True keeps the smallest
        # strictly-positive value, matching the former `rdtype(0.0) < val`.
        sub = var[adm.ADM_gmin:adm.ADM_gmax + 1,
                  adm.ADM_gmin:adm.ADM_gmax + 1,
                  kstart:kend + 1,
                  0:adm.ADM_lall]
        sub_pl = (var_pl[adm.ADM_gslf_pl, kstart:kend + 1, 0:adm.ADM_lall_pl]
                  if adm.ADM_have_pl else None)

        if nonzero:
            vmin = cnst.CONST_HUGE
            pos = sub[sub > rdtype(0.0)]
            if pos.size:
                vmin = min(vmin, np.min(pos))
            if sub_pl is not None:
                pos_pl = sub_pl[sub_pl > rdtype(0.0)]
                if pos_pl.size:
                    vmin = min(vmin, np.min(pos_pl))

        else:  # If nonzero is False, find the absolute minimum
            vmin = np.min(sub) if sub.size else cnst.CONST_HUGE
            if sub_pl is not None and sub_pl.size:
                vmin = min(vmin, np.min(sub_pl))

        # Perform global min communication across processes
        vmin_g = comm.Comm_Stat_min(vmin)

        return vmin_g
