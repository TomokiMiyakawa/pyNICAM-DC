import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc

class Gmtr:

    _instance = None

    GMTR_p_nmax = 8
    GMTR_p_AREA  = 0
    GMTR_p_RAREA = 1
    GMTR_p_IX    = 2
    GMTR_p_IY    = 3
    GMTR_p_IZ    = 4
    GMTR_p_JX    = 5
    GMTR_p_JY    = 6
    GMTR_p_JZ    = 7

    GMTR_t_nmax = 5
    GMTR_t_AREA  = 0
    GMTR_t_RAREA = 1
    GMTR_t_W1    = 2
    GMTR_t_W2    = 3
    GMTR_t_W3    = 4

    # Constants for a (axis-related parameters) - Zero-based
    GMTR_a_nmax = 12
    GMTR_a_nmax_pl = 18

    GMTR_a_HNX  = 0
    GMTR_a_HNY  = 1
    GMTR_a_HNZ  = 2
    GMTR_a_HTX  = 3
    GMTR_a_HTY  = 4
    GMTR_a_HTZ  = 5
    GMTR_a_TNX  = 6
    GMTR_a_TNY  = 7
    GMTR_a_TNZ  = 8
    GMTR_a_TTX  = 9
    GMTR_a_TTY  = 10
    GMTR_a_TTZ  = 11

    GMTR_a_TN2X = 12
    GMTR_a_TN2Y = 13
    GMTR_a_TN2Z = 14
    GMTR_a_TT2X = 15
    GMTR_a_TT2Y = 16
    GMTR_a_TT2Z = 17

    # Public variable
    GMTR_polygon_type = "ON_SPHERE"  # 'ON_SPHERE': triangle fits the sphere, 'ON_PLANE': triangle is treated as 2D

    # These were "Private" variables in the orginal code. 
    GMTR_fname = ""  
    GMTR_io_mode = "ADVANCED"  
    # Probably should be changed to _GMTR_* at some point, like:
    #_GMTR_fname = ""  
    #_GMTR_io_mode = "ADVANCED"  

    def __init__(self):
        pass

    def GMTR_setup(self, fname_in):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[gmtr]/Category[common share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'gmtrparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** gmtrparam not found in toml file! STOP.", file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['gmtrparam']
            self.GMTR_polygon_type = cnfs['GMTR_polygon_type']
            self.GMTR_io_mode = cnfs['GMTR_io_mode']    
            self.GMTR_fname = cnfs['GMTR_fname']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        return