import numpy as np
import toml
from mod_io_param import iop

class Stdio:

    _instance = None

    H_SHORT     = iop.IO_HSHORT       #< Character length (16)
    H_MID       = iop.IO_HMID         #< Character length (64)
    H_LONG      = iop.IO_HLONG        #< Character length (256)


    # Standard, common I/O module 
    #def __init__(self, modelname, fname_in):
    def __init__(self):
        #self.iop = Ioparam()
        pass

#    def io_arg_getfname(self, is_master):

#    def io_setup(self, modelname, fname_in=None):
    def io_setup(self, modelname, fname_in):

#        if fname_in:    
#            fname = fname_in    # ignoring optional read of fname for now       
        self.h_modelname = modelname
        #self.h_libname = modelname  

        # Load configurations from TOML file
        cnfs = toml.load(fname_in)['param_io']
        self.h_source = cnfs['h_source']
        self.h_institute = cnfs['h_institute']
        self.io_log_basename = cnfs['io_log_basename']
        self.io_log_suppress = cnfs['io_log_suppress']  
        self.io_log_allnode = cnfs['io_log_allnode']  
        self.io_log_nml_suppress = cnfs['io_log_nml_suppress']          
        self.io_aggregate = cnfs['io_aggregate']     

        return
         

    def io_make_idstr(self,basename,ext,myrank):

        srank = f"{myrank:08d}"  # Formats 'rank' as a 8-digit string, padded with zeros
        outstr = f"{basename.strip()}.{ext.strip()}{srank.strip()}"
        return outstr

    def io_log_setup(self,myrank,is_master):
        """
        Setup log.
        :param myrank: My rank ID (integer)
        :param is_master: Indicates if this is the master process (boolean)
        """
        self.io_l = not self.io_log_suppress and (is_master or self.io_log_allnode)
        self.io_nml = not self.io_log_nml_suppress and self.io_l

        if self.io_l:
            #if self.io_log_basename == self.IO_STDOUT:
            #    io_fid_log = self.IO_FID_STDOUT
            #else:
            #    io_fid_log = self.io_get_available_fid()

            self.fname_log = self.io_make_idstr(self.io_log_basename, 'pe', myrank)
            #fname_log = "placeholder"
            try:
                with open(self.fname_log, 'w') as log_file:
                    # Writing log content
                    log_file.write('\n')
                    log_file.write('#'*72 + '\n')
                    log_file.write('\n')
                    log_file.write(' NICAM-DC (dynamical core package of NICAM)\n')
                        # ... additional log writing
                    #log_file.write(f'*** Open config file: {fname_in}\n')
                    #log_file.write(f'*** Open log file, FID = {io_fid_log}\n')
                    log_file.write(f'*** Basename of log file = {self.io_log_basename}\n')
                    log_file.write(f'*** Detailed log output = {self.io_nml}\n')
            except IOError:
                print(f'xxx File open error! : {self.fname_log}')
                raise  # Raising the error to stop the execution
        else:
            if is_master:
                print('*** Log report is suppressed.')
                self.fname_log = 'dummy'

        return
    
std = Stdio()
print('instantiated std')

