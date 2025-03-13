import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf

class Tim:
    
    _instance = None
    
    def __init__(self):
        pass

    def TIME_setup(self, fname_in):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[Time]/Category[common share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'timeparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** timeparam not found in toml file! STOP.", file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['timeparam']
            self.integ_type = cnfs['integ_type']
            self.dtl = cnfs['dtl']
            self.lstep_max = cnfs['lstep_max']
            self.start_year = cnfs['start_year']
            self.start_month = cnfs['start_month']
            self.start_day = cnfs['start_day']
            self.start_hour = cnfs['start_hour']
            self.start_min = cnfs['start_min']
            self.start_sec = cnfs['start_sec']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)