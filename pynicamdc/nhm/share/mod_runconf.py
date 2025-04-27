import toml
import numpy as np
from mod_stdio import std
from mod_process import prc
from mod_chemvar import chem

class Rcnf:
    
    _instance = None

    # ---< ComponentSelector >---

    # ---< Dynamics >---
    NON_HYDRO_ALPHA = 1  # Nonhydrostatic/hydrostatic flag
    DYN_DIV_NUM = 1
    TRC_ADV_TYPE = "MIURA2004"
    NDIFF_LOCATION = "IN_LARGE_STEP2"
    TRC_ADV_LOCATION = "IN_DYN_DIV"
    FLAG_NUDGING = False
    THUBURN_LIM = True  # [add] 20130613 R.Yoshida
    CORIOLIS_PARAM = 0.0  # 
    
    # ---< Physics >---
    RAIN_TYPE = "DRY"
    opt_2moment_water = False
    
    CP_TYPE = "NONE"
    MP_TYPE = "NONE"
    RD_TYPE = "NONE"
    SF_TYPE = "DEFAULT"
    ROUGHNESS_SEA_TYPE = "DEFAULT"
    OCEAN_TYPE = "NONE"
    RIVER_TYPE = "NONE"
    LAND_TYPE = "NONE"
    TB_TYPE = "NONE"
    AE_TYPE = "NONE"
    CHEM_TYPE = "NONE"
    GWD_TYPE = "NONE"
    AF_TYPE = "NONE"

    OUT_FILE_TYPE = "DEFAULT"

    # ---< Tracer ID Setting >---
    PRG_vmax = None  # To be set later
    PRG_vmax0 = 6
    I_RHOG     = 0  # Density x G^1/2
    I_RHOGVX   = 1  # Density x G^1/2 x Horizontal velocity (X-direction)
    I_RHOGVY   = 2  # Density x G^1/2 x Horizontal velocity (Y-direction)
    I_RHOGVZ   = 3  # Density x G^1/2 x Horizontal velocity (Z-direction)
    I_RHOGW    = 4  # Density x G^1/2 x Vertical velocity
    I_RHOGE    = 5  # Density x G^1/2 x Energy
    I_RHOGQstr = 6  # Tracers
    I_RHOGQend = -1
    PRG_name = ["rhog", "rhogvx", "rhogvy", "rhogvz", "rhogw", "rhoge"]

    # ---< Diagnostic Variables >---
    DIAG_vmax = None  # To be set later
    DIAG_vmax0 = 6
    I_pre  = 0  # Pressure
    I_tem  = 1  # Temperature
    I_vx   = 2  # Horizontal velocity (X-direction)
    I_vy   = 3  # Horizontal velocity (Y-direction)
    I_vz   = 4  # Horizontal velocity (Z-direction)
    I_w    = 5  # Vertical velocity
    I_qstr = 6  # Tracers
    I_qend = -1
    DIAG_name = ["pre", "tem", "vx", "vy", "vz", "w"]

    TRC_vmax = 0  

    # --- Tracer names and descriptions ---
    TRC_name = []
    WLABEL = []

    # --- Water mass tracers ---
    NQW_MAX = 0
    NQW_STR = -1
    NQW_END = -1
    I_QV = -1
    I_QC = -1
    I_QR = -1
    I_QI = -1
    I_QS = -1
    I_QG = -1

    # --- Water number tracers ---
    NNW_MAX = 0
    NNW_STR = -1
    NNW_END = -1
    I_NC = -1
    I_NR = -1
    I_NI = -1
    I_NS = -1
    I_NG = -1

    # --- Turbulent tracers ---
    NTB_MAX = 0
    I_TKE = -1
    I_QKEp = -1
    I_TSQp = -1
    I_QSQp = -1
    I_COVp = -1

    # --- Chemical (or general-purpose) tracers ---
    NCHEM_MAX = 0
    NCHEM_STR = -1
    NCHEM_END = -1

    # --- Specific heat of water at constant pressure ---
    CVW = []
    CPW = []

    # --- Number of bands for radiation ---
    NRBND = 3
    NRBND_VIS = 0
    NRBND_NIR = 1
    NRBND_IR  = 2

    # --- Direct/Diffuse radiation ---
    NRDIR = 2
    NRDIR_DIRECT  = 0
    NRDIR_DIFFUSE = 1

    # --- Roughness parameters ---
    NTYPE_Z0 = 3
    N_Z0M = 0
    N_Z0H = 1
    N_Z0E = 2

    #Additional from ideal_init
    DCTEST_type = ''
    DCTEST_case = ''

    def __init__(self):
        pass


    def RUNCONF_setup(self, fname_in, cnst):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[runconf]/Category[nhm share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'runconfparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** runconfparam not found in toml file! STOP.", file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)
        else:
            cnfs = cnfs['runconfparam']
            self.NDIFF_LOCATION = cnfs['NDIFF_LOCATION']
            self.THUBURN_LIM = cnfs['THUBURN_LIM']
            self.CHEM_TYPE = cnfs['CHEM_TYPE']
            self.DCTEST_type = cnfs['DCTEST_type']
            self.DCTEST_case = cnfs['DCTEST_case']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        self.RUNCONF_component_setup()

        self.RUNCONF_tracer_setup(fname_in)

        self.RUNCONF_thermodyn_setup(cnst)

        return


    def RUNCONF_component_setup(self):
        if self.THUBURN_LIM:  # [add] 20130613 R.Yoshida
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('Run with "Thuburn Limiter" in MIURA2004 Advection', file=log_file)
        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:  
                    print('### Without "Thuburn Limiter" in MIURA2004 Advection', file=log_file)
        return


    def RUNCONF_tracer_setup(self, fname_in):

        # --- Counting tracer
        self.TRC_vmax = 0

        # --- Mass tracer for Water
        if self.RAIN_TYPE == "DRY":
            self.NQW_MAX = 1                # max   same as fortran
            self.I_QV = self.TRC_vmax       # index  -1 from fortran 
            #print("DRY!!!!")
        elif self.RAIN_TYPE == "CLOUD_PARAM":
            self.NQW_MAX = 2                # max
            self.I_QV = self.TRC_vmax      # index
            self.I_QC = self.TRC_vmax + 1   # index
        elif self.RAIN_TYPE == "WARM":
            self.NQW_MAX = 3                # max
            self.I_QV = self.TRC_vmax      # index
            self.I_QC = self.TRC_vmax + 1   # index
            self.I_QR = self.TRC_vmax + 2   # index
        elif self.RAIN_TYPE == "COLD":
            self.NQW_MAX = 6                # max
            self.I_QV = self.TRC_vmax       # index
            self.I_QC = self.TRC_vmax + 1   # index
            self.I_QR = self.TRC_vmax + 2   # index
            self.I_QI = self.TRC_vmax + 3   # index
            self.I_QS = self.TRC_vmax + 4   # index
            self.I_QG = self.TRC_vmax + 5   # index
        else:
            print("xxx RAIN_TYPE must be set to DRY, CLOUD_PARAM, WARM, or COLD. STOP.")
            self.PRC_MPIstop()

        self.NQW_STR = self.TRC_vmax #+ 1                # index from zero
        self.NQW_END = self.TRC_vmax + self.NQW_MAX      # 
        self.TRC_vmax += self.NQW_MAX           # total number of QW tracers, same as fortran
                                                # index 0 to self.TRC_vmax - 1 in python

        # --- Number tracer for Water
        if self.opt_2moment_water:
            if self.RAIN_TYPE == "DRY":
                self.NNW_MAX = 0
            elif self.RAIN_TYPE == "CLOUD_PARAM":
                self.NNW_MAX = 1
                self.I_NC = self.TRC_vmax    #+ 1
            elif self.RAIN_TYPE == "WARM":
                self.NNW_MAX = 2
                self.I_NC = self.TRC_vmax    #+ 1
                self.I_NR = self.TRC_vmax +1 #+ 2
            elif self.RAIN_TYPE == "COLD":
                self.NNW_MAX = 5
                self.I_NC = self.TRC_vmax     #+ 1
                self.I_NR = self.TRC_vmax + 1 #+ 2
                self.I_NI = self.TRC_vmax + 2 #+ 3
                self.I_NS = self.TRC_vmax + 3 #+ 4
                self.I_NG = self.TRC_vmax + 4 #+ 5

            self.NNW_STR = self.TRC_vmax + min(1, self.NNW_MAX)  # This may be wrong. check later
            self.NNW_END = self.TRC_vmax + self.NNW_MAX
            self.TRC_vmax += self.NNW_MAX

        # --- Number tracer for Water ---
        if self.opt_2moment_water:
            if self.RAIN_TYPE == "DRY":
                self.NNW_MAX = 0
            elif self.RAIN_TYPE == "CLOUD_PARAM":
                self.NNW_MAX = 1
                self.I_NC = self.TRC_vmax     #+ 1
            elif self.RAIN_TYPE == "WARM":
                self.NNW_MAX = 2
                self.I_NC = self.TRC_vmax     #+ 1
                self.I_NR = self.TRC_vmax + 1 #+ 2
            elif self.RAIN_TYPE == "COLD":
                self.NNW_MAX = 5
                self.I_NC = self.TRC_vmax     #+ 1
                self.I_NR = self.TRC_vmax + 1 #+ 2
                self.I_NI = self.TRC_vmax + 2 #+ 3
                self.I_NS = self.TRC_vmax + 3 #+ 4
                self.I_NG = self.TRC_vmax + 4 #+ 5

            self.NNW_STR = self.TRC_vmax + min(1, self.NNW_MAX)  # This may be wrong. check later
            self.NNW_END = self.TRC_vmax + self.NNW_MAX
            self.TRC_vmax += self.NNW_MAX

        # --- Tracer for turbulence ---
        if self.TB_TYPE == "MY2.5":
            self.NTB_MAX = 1
            self.I_TKE = self.TRC_vmax      #+ 1
        elif self.TB_TYPE == "MYNN2.5":
            self.NTB_MAX = 1
            self.I_QKEp = self.TRC_vmax     #+ 1
        elif self.TB_TYPE == "MYNN3":
            self.NTB_MAX = 4
            self.I_QKEp = self.TRC_vmax     #+ 1
            self.I_TSQp = self.TRC_vmax + 1 #+ 2
            self.I_QSQp = self.TRC_vmax + 2 #+ 3
            self.I_COVp = self.TRC_vmax + 3 #+ 4

        self.TRC_vmax += self.NTB_MAX

        # --- Tracer for chemistry ---
        chem.CHEMVAR_setup(fname_in)

        # print("0: TRC_vmax=", self.TRC_vmax)
        # print("self.CHEM_TYPE=", self.CHEM_TYPE)

        if self.CHEM_TYPE == "PASSIVE":
            self.NCHEM_MAX = chem.CHEM_TRC_vmax
            self.NCHEM_STR = self.TRC_vmax + min(0, self.NCHEM_MAX)   # This may be wrong. check later
            #print("TRC_vmax=", self.TRC_vmax)

            self.NCHEM_END = self.TRC_vmax + self.NCHEM_MAX

            self.TRC_vmax += self.NCHEM_MAX
            # print("TRC_vmax=", self.TRC_vmax)
            # print("chem.CHEM_TRC_vmax=", chem.CHEM_TRC_vmax)
            # print("self.NCHEM_MAX=", self.NCHEM_MAX)

            # --- Allocate tracer names and labels ---
        self.TRC_name = [""] * self.TRC_vmax  # [add] H.Yashiro 20110819
        #print("self.TRC_vmax=", self.TRC_vmax)
        self.WLABEL = [""] * self.TRC_vmax  # 08/04/12 [Add] T.Mitsui

        #print("before label!!", self.TRC_vmax)
        #print("self.I_QV", self.I_QV)

        # --- Labeling ---
        for v in range(self.TRC_vmax):
            if v == self.I_QV:
                self.TRC_name[v] = "qv"
                self.WLABEL[v] = "VAPOR"
            elif v == self.I_QC:
                self.TRC_name[v] = "qc"
                self.WLABEL[v] = "CLOUD"
            elif v == self.I_QR:
                self.TRC_name[v] = "qr"
                self.WLABEL[v] = "RAIN"
            elif v == self.I_QI:
                self.TRC_name[v] = "qi"
                self.WLABEL[v] = "ICE"
            elif v == self.I_QS:
                self.TRC_name[v] = "qs"
                self.WLABEL[v] = "SNOW"
            elif v == self.I_QG:
                self.TRC_name[v] = "qg"
                self.WLABEL[v] = "GRAUPEL"

            elif v == self.I_NC:
                self.TRC_name[v] = "nc"
                self.WLABEL[v] = "CLOUD_NUM"
            elif v == self.I_NR:
                self.TRC_name[v] = "nr"
                self.WLABEL[v] = "RAIN_NUM"
            elif v == self.I_NI:
                self.TRC_name[v] = "ni"
                self.WLABEL[v] = "ICE_NUM"
            elif v == self.I_NS:
                self.TRC_name[v] = "ns"
                self.WLABEL[v] = "SNOW_NUM"
            elif v == self.I_NG:
                self.TRC_name[v] = "ng"
                self.WLABEL[v] = "GRAUPEL_NUM"
            elif v == self.NCHEM_STR:
                #print(self.TRC_name[:])
                for i in range(self.NCHEM_MAX):
                    self.TRC_name[v + i] = chem.CHEM_TRC_name[i]
                    self.WLABEL[v + i] = chem.CHEM_TRC_desc[i]
                    # print("self.TRC_name[v + i] = ", self.TRC_name[v + i])
                    # print("self.WLABEL[v + i]  = ", self.WLABEL[v + i])
                    # print(self.TRC_name[:])

        # Update prognostic and diagnostic variables
        self.PRG_vmax = self.PRG_vmax0 + self.TRC_vmax    ####### check these numbers!!!
        self.I_RHOGQend = self.PRG_vmax

        self.DIAG_vmax = self.DIAG_vmax0 + self.TRC_vmax
        self.I_qend = self.DIAG_vmax

        # --- Logging information ---
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print("*** Prognostic Tracers", file=log_file)
                print("|=========================================================|", file=log_file)
                print("|       :varname         :description                     |", file=log_file)
                for v in range(self.TRC_vmax):
                    print(f"|ID={v:4} : {self.TRC_name[v]:16} : {self.WLABEL[v]} |", file=log_file)
                print("|=========================================================|", file=log_file)
                print("", file=log_file)
                print("*** Thermodynamic (water) tracers", file=log_file)
                print(f"---> {self.NQW_MAX} tracers ({self.NQW_STR}-{self.NQW_END})", file=log_file)    

        return
    

    def RUNCONF_thermodyn_setup(self, cnst):

        # 'SIMPLE': standard approximation CVD * T
        # 'EXACT': exact formulation
        #         -> if warm rain
        #            qd*CVD*T + qv*CVV*T + (qc+qr)*CPL*T
        #         -> if cold rain
        #            qd*CVD*T + qv*CVV*T + (qc+qr)*CPL*T
        #            + (qi+qs)*CPI*T

        # --- Allocate memory for heat capacity arrays ---
        size = self.NQW_END - self.NQW_STR + 1  # Compute array size
        self.CVW = np.zeros(size)  # Initialize as zeros
        self.CPW = np.zeros(size)

        # --- Assign values based on thermodynamics type ---
        if cnst.CONST_THERMODYN_TYPE == "SIMPLE":
            for v in range(self.NQW_STR, self.NQW_END + 1):
                idx = v - self.NQW_STR  # Adjust index for zero-based NumPy array
                if v == self.I_QV:  # Vapor
                    self.CVW[idx] = cnst.CONST_CVdry
                    self.CPW[idx] = cnst.CONST_CPdry
                elif v in {self.I_QC, self.I_QR, self.I_QI, self.I_QS, self.I_QG}:  # Cloud, Rain, Ice, Snow, Graupel
                    self.CVW[idx] = cnst.CONST_CVdry
                    self.CPW[idx] = cnst.CONST_CVdry

        elif cnst.CONST_THERMODYN_TYPE == "SIMPLE2":
            for v in range(self.NQW_STR, self.NQW_END + 1):
                idx = v - self.NQW_STR
                if v == self.I_QV:  # Vapor
                    self.CVW[idx] = cnst.CONST_CVvap
                    self.CPW[idx] = cnst.CONST_CPvap
                elif v in {self.I_QC, self.I_QR, self.I_QI, self.I_QS, self.I_QG}:  # Cloud, Rain, Ice, Snow, Graupel
                    self.CVW[idx] = cnst.CONST_CPvap
                    self.CPW[idx] = cnst.CONST_CPvap

        elif cnst.CONST_THERMODYN_TYPE == "EXACT":
            for v in range(self.NQW_STR, self.NQW_END + 1):
                idx = v - self.NQW_STR
                if v == self.I_QV:  # Vapor
                    self.CVW[idx] = cnst.CONST_CVvap
                    self.CPW[idx] = cnst.CONST_CPvap
                elif v in {self.I_QC, self.I_QR}:  # Cloud, Rain
                    self.CVW[idx] = cnst.CONST_CL
                    self.CPW[idx] = cnst.CONST_CL
                elif v in {self.I_QI, self.I_QS, self.I_QG}:  # Ice, Snow, Graupel
                    self.CVW[idx] = cnst.CONST_CI
                    self.CPW[idx] = cnst.CONST_CI

        return