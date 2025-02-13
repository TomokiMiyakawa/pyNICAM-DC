import os
import sys
import toml
#from mod_io_param import IOParam
from io_param import Ioparam

#class Stdio(IOParam):  # Inherits from IOParam class
class Stdio(Ioparam):  # Inherits from IOParam class
    # ...

    @staticmethod
    def IO_setup(MODELNAME, fname_in=None):
        Stdio.H_MODELNAME = MODELNAME.strip()
        Stdio.H_LIBNAME = MODELNAME.strip()
        Stdio.H_SOURCE = MODELNAME.strip()

        if fname_in is not None:
            try:
                config = toml.load(fname_in)
                # Now 'config' is a dict containing the configuration data. 
                # You can access and manipulate it as needed.
            except:
                print('xxx Failed to read TOML file! STOP.')
                print('xxx filename :', fname_in)
                sys.exit(1)

        # ... Further setup based on the data in the config

    @staticmethod
    def IO_ARG_getfname():
        if len(sys.argv) < 2:
            print('xxx Program needs config file from argument! STOP.')
            sys.exit(1)
        else:
            return sys.argv[1]

    @staticmethod
    def IO_make_idstr(instr, ext, rank, isrgn=False):
        srank = str(rank).zfill(6 if not isrgn else 8)
        return f"{instr}.{ext}{srank}"

    @staticmethod
    def IO_get_available_fid():
        for fid in range(10, 100):  # IO_MINFID, IO_MAXFID
            if not os.path.exists(f'file_{fid}'):
                return fid
        print('xxx Too many open units! STOP.')
        sys.exit(1)

    @staticmethod
    def IO_LOG_setup(myrank, is_master):
        if not Stdio.IO_LOG_SUPPRESS:
            Stdio.IO_L = True if is_master else Stdio.IO_LOG_ALLNODE

        Stdio.IO_NML = False if Stdio.IO_LOG_NML_SUPPRESS else Stdio.IO_L

        if Stdio.IO_L:
            if Stdio.IO_LOG_BASENAME == Stdio.IO_STDOUT:
                Stdio.IO_FID_LOG = Stdio.IO_FID_STDOUT
            else:
                Stdio.IO_FID_LOG = Stdio.IO_get_available_fid()
                fname = Stdio.IO_make_idstr(Stdio.IO_LOG_BASENAME, 'pe', myrank)
                try:
                    open(fname, 'w')
                except IOError:
                    print('xxx File open error! :', fname)
                    sys.exit(1)

            if is_master:
                print('*** Log report is enabled.')
                print('*** Log file basename is', Stdio.IO_LOG_BASENAME)

        else:
            if is_master:
                print('*** Log report is suppressed.')

    @staticmethod
    def IO_CNF_open(fname, is_master):
        try:
            fid = open(fname, 'r')
        except IOError:
            if is_master:
                print('xxx Failed to open config file! STOP.')
                print('xxx filename :', fname)
                sys.exit(1)
        return fid
