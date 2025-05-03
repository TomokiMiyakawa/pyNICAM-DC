import numpy as np
from mod_adm import adm

class Ppm:
    
    _instance = None
    
    def __init__(self):
        pass

    def PNT_setup(self):

        self.plmask  = np.zeros((adm.ADM_KNONE, adm.ADM_lall), dtype=np.int32)
        self.pntmask = np.zeros((adm.ADM_KNONE, adm.ADM_lall, 2), dtype=np.int32)

        k0 = adm.ADM_K0

        for l in range(adm.ADM_lall):

            if adm.ADM_have_pl:
                self.plmask = 1
            else:
                self.plmask = 0

            if adm.ADM_have_sgp[l]:
                self.pntmask[k0, l, 0] = 0
                self.pntmask[k0, l, 1] = 1
            else:
                self.pntmask[k0, l, 0] = 1
                self.pntmask[k0, l, 1] = 0

        return

ppm = Ppm()