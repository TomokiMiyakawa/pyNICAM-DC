import toml
import numpy as np
#from process import prc

class Comm:

    def __init__(self):
        pass

    def COMM_setup(self, io_l, io_nml, fname_log, fname_in):

        if io_l: 
            with open(fname_log, 'a') as log_file:
                print("+++ Module[comm]/Category[common share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'commparam' not in cnfs:
            with open(fname_log, 'a') as log_file:
                print("*** commparam not found in toml file! STOP.", file=log_file)
                #stop

        else:
            self.COMM_apply_barrier = cnfs['commparam']['COMM_apply_barrier']  
            self.COMM_varmax = cnfs['commparam']['COMM_varmax']  
            debug = cnfs['commparam']['debug']  
            testonly = cnfs['commparam']['testonly']  

        if io_nml: 
            if io_l:
                with open(fname_log, 'a') as log_file: 
                    print(cnfs['rgnmngparam'],file=log_file)
        
        if ( RP == DP ):
            COMM_datatype = MPI_DOUBLE_PRECISION
        elseif( RP == SP ) then
            COMM_datatype = MPI_REAL
        else    
            write(*,*) 'xxx precision is not supportd'
            call PRC_MPIstop
        endif





    def COMM_list_generate(self):
        pass

    def COMM_sortdest(self):
        pass    

    def COMM_sortdest_pl(self):
        pass

    def COMM_sortdest_singular(self):
        pass

