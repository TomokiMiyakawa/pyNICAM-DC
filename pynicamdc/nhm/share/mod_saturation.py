import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_io_param import iop

#from mod_prof import prf

class Satr:
    
    _instance = None

    TEM_MIN   = 10.0  #< minimum temperature [K]
    
    SATURATION_ULIMIT_TEMP = 273.15  #_RP !< upper limit temperature
    SATURATION_LLIMIT_TEMP = 233.15  #_RP !< lower limit temperature

    def __init__(self):
        pass

    def SATURATION_setup(self, fname_in, cnst):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[saturation]/Category[nhm share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'saturationparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** grdparam not found in toml file! Use default.", file=log_file)
                #prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['saturationparam']
            self.SATURATION_ULIMIT_TEMP = cnfs['SATURATION_ULIMIT_TEMP']
            self.SATURATION_LLIMIT_TEMP = cnfs['SATURATION_LLIMIT_TEMP']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)


        # Compute RTEM00
        self.RTEM00 = 1.0 / cnst.CONST_TEM00

        if cnst.CONST_THERMODYN_TYPE == "EXACT":
            self.CPovR_liq = (cnst.CONST_CPvap - cnst.CONST_CL) / cnst.CONST_Rvap
            self.CPovR_ice = (cnst.CONST_CPvap - cnst.CONST_CI) / cnst.CONST_Rvap
            self.CVovR_liq = (cnst.CONST_CVvap - cnst.CONST_CL) / cnst.CONST_Rvap
            self.CVovR_ice = (cnst.CONST_CVvap - cnst.CONST_CI) / cnst.CONST_Rvap

        elif cnst.CONST_THERMODYN_TYPE in {"SIMPLE", "SIMPLE2"}:
            self.CPovR_liq = 0.0
            self.CPovR_ice = 0.0
            self.CVovR_liq = 0.0
            self.CVovR_ice = 0.0

        # Compute LovR_liq and LovR_ice
        self.LovR_liq = cnst.CONST_LHV / cnst.CONST_Rvap
        self.LovR_ice = cnst.CONST_LHS / cnst.CONST_Rvap

        # Compute dalphadT_const
        self.dalphadT_const = 1.0 / (self.SATURATION_ULIMIT_TEMP - self.SATURATION_LLIMIT_TEMP)

        # Logging temperature range for ice
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(f"*** Temperature range for ice : {self.SATURATION_LLIMIT_TEMP:7.2f} - {self.SATURATION_ULIMIT_TEMP:7.2f}", file = log_file )

        return
    
satr = Satr()
#print("instantiated satr")