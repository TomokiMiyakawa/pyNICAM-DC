import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_calendar import cldr
#from mod_prof import prf

class Tim:
    
    _instance = None
    
    TIME_integ_type = 'UNDEF'
    TIME_split     = True
    TIME_lstep_max = 10
    TIME_dtl       = 5.0   # keep this double pricision
    _TIME_backward_sw = False

    def __init__(self):
        pass

    def TIME_setup(self, fname_in, rdtype):

        integ_type = self.TIME_integ_type
        split = self.TIME_split
        dtl = self.TIME_dtl  #DP
        lstep_max = self.TIME_lstep_max
        sstep_max = -999

        start_date = np.full(6, -999, dtype=int) #[-999] * len(start_date)  
        start_year = 0
        start_month = 1
        start_day = 1
        start_hour = 0
        start_min = 0
        start_sec = 0

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
            integ_type = cnfs['integ_type']
            dtl = cnfs['dtl']
            lstep_max = cnfs['lstep_max']
            start_year = cnfs['start_year']
            start_month = cnfs['start_month']
            start_day = cnfs['start_day']
            start_hour = cnfs['start_hour']
            start_min = cnfs['start_min']
            start_sec = cnfs['start_sec']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        # --- Rewrite
        self.TIME_integ_type = integ_type
        self.TIME_split = split
        self.TIME_dtl = dtl #DP
        self.TIME_lstep_max = lstep_max

        if sstep_max == -999:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"TIME_integ_type is {self.TIME_integ_type}", file=log_file)

            # Set TIME_sstep_max based on TIME_integ_type
            self.TIME_sstep_max = {
                'RK2': 4,
                'RK3': 6,
                'RK4': 12,
                'TRCADV': 0
            }.get(self.TIME_integ_type, None)

            if self.TIME_sstep_max is None:
                print("xxx Invalid TIME_INTEG_TYPE! STOP.")
                exit()

            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"TIME_sstep_max is automatically set to: {self.TIME_sstep_max}", file=log_file)

        else:
            self.TIME_sstep_max = sstep_max

        # Compute self.TIME_dts
        self.TIME_dts = self.TIME_dtl / max(float(self.TIME_sstep_max), 1.0)

            
        if start_date[0] == -999:
            start_date[0] = start_year
        if start_date[1] == -999:
            start_date[1] = start_month
        if start_date[2] == -999:
            start_date[2] = start_day
        if start_date[3] == -999:
            start_date[3] = start_hour
        if start_date[4] == -999:
            start_date[4] = start_min
        if start_date[5] == -999:
            start_date[5] = start_sec

        # Call equivalent function for CALENDAR_yh2ss
        self.TIME_start = cldr.CALENDAR_yh2ss(start_date, rdtype)

        ## Handle optional backward switch
        #self.TIME_backward_sw = backward if 'backward' in locals() else False

        # ---< Large Step Configuration >---

        if self.TIME_lstep_max < 0:
            print("xxx TIME_lstep_max should be positive. STOP.")
            prc.prc_mpistop(std.io_l, std.fname_log)

        if self.TIME_sstep_max < 0:
            print("xxx TIME_sstep_max should be positive. STOP.")
            prc.prc_mpistop(std.io_l, std.fname_log)

        if self.TIME_dtl < 0:
            print("xxx TIME_dtl should be positive. STOP.")
            prc.prc_mpistop(std.io_l, std.fname_log)

        # Compute TIME_END based on backward switch
        if not self._TIME_backward_sw:
            self.TIME_end = self.TIME_start + self.TIME_lstep_max * self.TIME_dtl
        else:
            print("xxx Backward integration is not implemented yet. STOP.")
            prc.prc_mpistop(std.io_l, std.fname_log)
            #self.TIME_END = self.TIME_START - self.TIME_LSTEP_MAX * self.TIME_DTL  # [TM]

        # Initialize time counters
        self.TIME_nstart = 0
        self.TIME_nend = self.TIME_nstart + self.TIME_lstep_max

        self.TIME_ctime = self.TIME_start
        self.TIME_cstep = self.TIME_nstart

        # Convert times to calendar format
        self.HTIME_start = cldr.CALENDAR_ss2cc(self.TIME_start, rdtype)
        self.HTIME_end = cldr.CALENDAR_ss2cc(self.TIME_end, rdtype)
        self.TIME_htime = cldr.CALENDAR_ss2cc(self.TIME_ctime, rdtype)

        # Output debugging information
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("\n====== Time management ======", file=log_file)
                print(f"--- Time integration scheme (large step): {self.TIME_integ_type.strip()}", file=log_file)
                print(f"--- Backward integration?               : {self._TIME_backward_sw}", file=log_file)
                print(f"--- Time interval for large step        : {self.TIME_dtl}", file=log_file)
                print(f"--- Time interval for small step        : {self.TIME_dts}", file=log_file)
                print(f"--- Max steps of large step             : {self.TIME_lstep_max}", file=log_file)
                print(f"--- Max steps of small step             : {self.TIME_sstep_max}", file=log_file)
                print(f"--- Start time (sec)                    : {self.TIME_start}", file=log_file)
                print(f"--- End time   (sec)                    : {self.TIME_end}", file=log_file)
                print(f"--- Start time (date)                   : {self.HTIME_start}", file=log_file)
                print(f"--- End time   (date)                   : {self.HTIME_end}", file=log_file)
                print(f"--- Total integration time              : {self.TIME_end - self.TIME_start}", file=log_file)
                print(f"--- Time step at the start              : {self.TIME_nstart}", file=log_file)
                print(f"--- Time step at the end                : {self.TIME_nend}", file=log_file)

        return
    
    def TIME_report(self, cldr, rdtype):

        #print("TIME_htime: ", self.TIME_htime)
        #print("TIME_ctime: ", self.TIME_ctime)
        #print("TIME_cstep: ", self.TIME_cstep)
        self.TIME_htime = cldr.CALENDAR_ss2cc(self.TIME_ctime, rdtype)

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(f"### TIME = {self.TIME_htime} ( step = {self.TIME_cstep:8d} / {self.TIME_lstep_max:8d} )", file=log_file)

        if prc.prc_ismaster:
            print(f"### TIME = {self.TIME_htime} ( step = {self.TIME_cstep:8d} / {self.TIME_lstep_max:8d} )")

        return
    
    def TIME_advance(self, cldr, rdtype):

        # Time advance
        if not self._TIME_backward_sw:
            self.TIME_ctime += self.TIME_dtl
        else:
            self.TIME_ctime -= self.TIME_dtl  
            print("xxx Backward integration is not implemented yet. STOP.")
            prc.prc_mpistop(std.io_l, std.fname_log)

        self.TIME_cstep += 1

        self.TIME_report(cldr, rdtype)

        return
