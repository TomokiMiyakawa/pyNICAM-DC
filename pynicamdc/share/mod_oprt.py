import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf

class Oprt:
    
    _instance = None
    
    def __init__(self):
        pass

    def OPRT_setup(self, fname_in):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[oprt]/Category[common share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'oprtparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** oprtparam not found in toml file! STOP.", file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['oprtparam']
            self.OPRT_io_mode = cnfs['OPRT_io_mode']
            self.OPRT_fname = cnfs['OPRT_fname']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)