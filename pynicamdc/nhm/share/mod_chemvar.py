import toml
import numpy as np
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf

class Chem:
    
    _instance = None

    CHEM_TRC_vlim = 100  # Upper limit on number of tracers
    CHEM_TRC_vmax = 1  
    
    def __init__(self):
        pass

    def CHEMVAR_setup(self, fname_in):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[chemvar]/Category[nhm share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'chemvarparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** chemvarparam not found in toml file! use default.", file=log_file)

        else:
            cnfs = cnfs['chemvarparam']
            self.CHEM_TRC_vmax = cnfs['CHEM_TRC_vmax']
            #print("CHEM_TRC_vmax = ", self.CHEM_TRC_vmax)

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        self.CHEM_TRC_name = np.full(self.CHEM_TRC_vmax, "", dtype=f"U{std.H_SHORT}")  # H_SHORT for short names
        self.CHEM_TRC_desc = np.full(self.CHEM_TRC_vmax, "", dtype=f"U{std.H_MID}")    # H_MID for descriptions

        # --- Assign tracer names and descriptions ---
        for nq in range(self.CHEM_TRC_vmax):
            self.CHEM_TRC_name[nq] = f"passive{nq:03d}"
            self.CHEM_TRC_desc[nq] = f"passive_tracer_no{nq:03d}"

        return
    
chem = Chem()
#print("instantiated chem")