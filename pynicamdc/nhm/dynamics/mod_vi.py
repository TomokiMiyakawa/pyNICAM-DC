import numpy as np
from mod_adm import adm
from mod_stdio import std
from mod_process import prc

class Vi:
    
    _instance = None
    
    def __init__(self):
        pass


    def vi_setup(self, rdtype):

        self.Mc    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall),    dtype=rdtype)
        self.Mc_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)
        self.Mu    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall),    dtype=rdtype)
        self.Mu_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)
        self.Ml    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall),    dtype=rdtype)
        self.Ml_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)

        return
    
    