import toml
import numpy as np
#from mod_process import prc
from mod_process import prc
from mod_stdio import std   
class Adm:
    
    _instance = None
    
    # Basic definition & Local region and process
    I_l = 0
    I_prc = 1

    # Region ID and direction
    I_RGNID = 0
    I_DIR = 1

    # Identifiers of directions of region edges
    I_SW = 0
    I_NW = 1
    I_NE = 2
    I_SE = 3

    # Identifiers of directions of region vertices
    I_W = 0
    I_N = 1
    I_E = 2
    I_S = 3

    # Identifier of poles (north pole or south pole)
    I_NPL = 0
    I_SPL = 1

    # Identifier of triangle element (i-axis-side or j-axis side)
    ADM_TI = 0
    ADM_TJ = 1

    # Identifier of arc element (i-axis-side, ij-axis side, or j-axis side)
    ADM_AI = 0
    ADM_AIJ = 1
    ADM_AJ = 2

    # Identifier of 1 variable
    ADM_KNONE = 1  # dim size for single layer
    ADM_K0 = 0  # index for single layer

    # Dimension of the spatial vector
    ADM_nxyz = 3

    # maximum number of vertex linkage, ICO:5, PSP:6, LCP, MLCP:k
    ADM_vlink = 5

    # number of pole region
    ADM_rgn_nmax_pl =  2

    # number of pole region per process
    ADM_lall_pl     =  2

    # index for pole point
    ADM_gslf_pl     =  0  
    # start index of grid around the pole point
    ADM_gmin_pl     =  1      

    # total number of process
    ADM_prc_all = prc.prc_nprocs
    #ADM_prc_me  = prc.prc_myrank
    # master process
    ADM_prc_master = 0

    # maximum number of region per process
    RGNMNG_llim = 2560 

    def __init__(self):
        
        # Edge and vertex name tables
        self.RGNMNG_edgename = np.array(["SW", "NW", "NE", "SE"])  # Edge names
        self.RGNMNG_vertname = np.array(["W ", "N ", "E ", "S "])  # Vertex names

    def ADM_setup(self, fname_in):
        self.ADM_prc_me = prc.prc_myrank

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[adm]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'admparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** cnstparam not found in toml file! STOP.", file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            glevel = cnfs['admparam']['glevel'] 
            rlevel = cnfs['admparam']['rlevel']  
            vlayer = cnfs['admparam']['vlayer']  
            rgnmngfname = cnfs['admparam']['rgnmngfname'] 
            self.ADM_HGRID_SYSTEM = cnfs['admparam']['ADM_HGRID_SYSTEM']  
            debug = cnfs['admparam']['debug']  

            if ( self.ADM_HGRID_SYSTEM == 'ICO' ):
                self.ADM_vlink  = 5
                dmd        = 10
                self.ADM_prc_pl = 0  # process 0 handles pole region

            else:
                with open(std.fname_log, 'a') as log_file:
                    print("xxx [ADM_setup] Not appropriate param for ADM_HGRID_SYSTEM. STOP.", self.ADM_HGRID_SYSTEM, file=log_file)
                    prc.prc_mpistop(std.io_l, std.fname_log)

            self.ADM_gall_pl = self.ADM_vlink + 1
            self.ADM_gmax_pl = self.ADM_vlink  #5 index of the last grid point  (0 to 5), but same as self.ADM_gall_pl in fortran

            self.ADM_glevel = glevel
            self.ADM_rlevel = rlevel
            self.ADM_vlayer = vlayer
            self.ADM_DMD = dmd

            # Calculations
            self.ADM_rgn_nmax = (2 ** self.ADM_rlevel) * (2 ** self.ADM_rlevel) * self.ADM_DMD
            self.ADM_lall = self.ADM_rgn_nmax // prc.prc_nprocs
            #print("hahaha000 ", self.ADM_rgn_nmax, prc.prc_nprocs,self.ADM_lall)
            nmax = 2 ** (self.ADM_glevel - self.ADM_rlevel)
            self.ADM_gall_1d = 1 + nmax + 1
            self.ADM_gmin = 1     #1 + 1
            self.ADM_gmax = nmax  #1 + nmax

            self.ADM_gall = (1 + nmax + 1) * (1 + nmax + 1)
            self.ADM_gall_in = (nmax + 1) * (nmax + 1)

            if self.ADM_vlayer == 1:
                #"this is not tested"
                self.ADM_kdall = 2   # dim size of full level including dummy layers <added by a.kamijo on 2025.04.01>
                self.ADM_kall = 1   # element count
                self.ADM_kmin = 0   # index
                self.ADM_kmax = 0   # index
            else:
                self.ADM_kdall = 1 + self.ADM_vlayer + 1 # dim size of full level including dummy layers
                ####self.ADM_khall = 1 + self.ADM_vlayer     # dim size of half level including ground and TOA
                self.ADM_kall = self.ADM_vlayer          #     size of full level excluding dummy layers
                self.ADM_kmin = 1                # index of 1st layer above the ground in full level and the ground in halflevel (2 in fortran) 
                self.ADM_kdmin = 0               # index of the underground dummy layer (full/half level), and ground (half level)  (1 in fortran)
                self.ADM_kdmax = self.ADM_vlayer + 1  # index of the top dummy/TOA layer (full/half level) 
                self.ADM_kmax = self.ADM_vlayer       # index of the top layer excluding dummy/TOA (full/half level)
    
            # if vlayer is set to 40, full level is from 0 to 41 (42 elements), half level is from 0 to 41 (42 elements).
            # half(0) is the underground level, half(1) is ground level 0m, half(41) is the TOA.

            self.RGNMNG_setup(rgnmngfname)

            self.GLOBAL_extension_rgn = np.empty(self.ADM_lall, dtype=str)
            self.GLOBAL_extension_rgn[:] = ''
            self.ADM_have_sgp = np.full(self.ADM_lall, False, dtype=bool)

            for l in range(self.ADM_lall):  
            # Python's range starts from 0 by default, adjust according to your array indexing
                rgnid = self.RGNMNG_lp2r[l, self.ADM_prc_me]

                # Formatting the string with '.rgn' prefix and the (rgnid-1) value
                #GLOBAL_extension_rgn[l] = f".rgn{rgnid-1:05d}"
                self.GLOBAL_extension_rgn[l] = f".rgn{rgnid:08d}"

                if self.RGNMNG_vert_num[self.I_W, rgnid] == 3:
                    self.ADM_have_sgp[l] = True

            if self.ADM_prc_me == self.ADM_prc_pl:
                self.ADM_have_pl = True
            else:  
                self.ADM_have_pl = False

            self.ADM_l_me = 0

            self.output_info()

        return

    def RGNMNG_setup(self, fname_in=None):
    
        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[rgnmng]", file=log_file)        

        if fname_in is None:
            with open(std.fname_log, 'a') as log_file:
                #if std.io_l: print("*** input toml file is not specified. use default.", file=log_file)
                if std.io_l: print("*** input toml file is not specified. Stop.", file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)
                # maybe should stop here instead of using default
        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print("*** input toml file is ", fname_in, file=log_file)

            with open(fname_in, 'r') as  file:
                cnfs = toml.load(file)

            if 'rgnmngparam' not in cnfs:
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file: 
                        print("*** rgnmngparam not specified in toml file. Stop.", file=log_file)
                        prc.prc_mpistop(std.io_l, std.fname_log)  
                        #print("*** rgnmngparam not specified in toml file. use default.", file=log_file)
                        # maybe should stop here instead of using default 
            else:
                if 'RGNMNG_in_fname' in cnfs['rgnmngparam']:
                    RGNMNG_in_fname = cnfs['rgnmngparam']['RGNMNG_in_fname']                    

                if 'RGNMNG_out_fname' in cnfs['rgnmngparam']:
                    RGNMNG_out_fname = cnfs['rgnmngparam']['RGNMNG_out_fname']      
            
            if self.ADM_lall > self.RGNMNG_llim:
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file: 
                        print('xxx limit exceed! local region:', self.ADM_lall, self.RGNMNG_llim, file=log_file)
                    prc.prc_mpistop(std.io_l, std.fname_log)  #erronius

                if std.io_l:
                    with open(std.fname_log, 'a') as log_file: 
                        print(cnfs['rgnmngparam'],file=log_file)

        self.RGNMNG_edge_tab, self.RGNMNG_lnum, self.RGNMNG_lp2r = self.RGNMNG_input(RGNMNG_in_fname,self.ADM_rgn_nmax,self.ADM_prc_all,self.ADM_lall) #io_l, io_nml, fname_log, RGNMNG_in_fname)

        if self.RGNMNG_lnum[self.ADM_prc_me] != self.ADM_lall:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print('xxx limit exceed! local region:', self.RGNMNG_lnum(self.ADM_prc_me), self.ADM_lall, file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)


        self.RGNMNG_r2lp = np.empty((2, self.ADM_rgn_nmax), dtype=int)
        self.RGNMNG_l2r = np.empty(self.ADM_lall, dtype=int)

        for p in range(self.ADM_prc_all):  
            for l in range(self.ADM_lall): 
                self.RGNMNG_r2lp[self.I_l, self.RGNMNG_lp2r[l, p]] = l  # l already zero-based   
                self.RGNMNG_r2lp[self.I_prc, self.RGNMNG_lp2r[l, p]] = p  # p already zero-based
            
        for l in range(self.ADM_lall):  
            self.RGNMNG_l2r[l] = self.RGNMNG_lp2r[l, self.ADM_prc_me]

        self.RGNMNG_vert_num = np.empty(
            (self.I_S - self.I_W + 1, self.ADM_rgn_nmax), dtype=int
            )   
        
        self.RGNMNG_vert_tab = np.empty(
            (self.I_DIR - self.I_RGNID + 1, self.I_S - self.I_W + 1, self.ADM_rgn_nmax, self.ADM_vlink),
            dtype=int
            )
        
        self.RGNMNG_vert_tab_pl = np.empty(
            (self.I_DIR - self.I_RGNID + 1, self.ADM_rgn_nmax_pl, self.ADM_vlink),
            dtype=int
            )
                
        self.RGNMNG_vertex_walkaround()

        self.RGNMNG_rgn4pl = np.empty((self.I_SPL - self.I_NPL + 1), dtype=int) 
        self.RGNMNG_r2p_pl = np.empty((self.I_SPL - self.I_NPL + 1), dtype=int) 

        for r in range(self.ADM_rgn_nmax):  
            if self.RGNMNG_vert_num[self.I_N, r] == self.ADM_vlink:
                self.RGNMNG_rgn4pl[self.I_NPL] = r
                break  

        for r in range(self.ADM_rgn_nmax):  
            if self.RGNMNG_vert_num[self.I_S, r] == self.ADM_vlink:
                self.RGNMNG_rgn4pl[self.I_SPL] = r
                break  

        self.RGNMNG_r2p_pl[self.I_NPL] = self.ADM_prc_pl
        self.RGNMNG_r2p_pl[self.I_SPL] = self.ADM_prc_pl

        #print("N, rank: ",self.RGNMNG_rgn4pl[self.I_NPL],prc.prc_myrank)
        #print("S, rank: ",self.RGNMNG_rgn4pl[self.I_SPL],prc.prc_myrank)
        #import sys
        #sys.exit(1)

        return


    def RGNMNG_input(self, fname_in, rall, pall, lall):
        """
        Reads TOML input file for region and process details.

        Parameters:
            in_fname (str): Input TOML file name
            rall (int): Total number of regions
            pall (int): Number of regions per process
            lall (int): Limit of local regions that can be handled per process
        Returns:
            edge_tab (np.ndarray): Region link information for 4 edges
            lnum (np.ndarray): Number of local regions per process
            lp2r (np.ndarray): Process-to-region mapping
        """

        # Initialize arrays
        edge_tab = np.full((2, 4, rall), -1, dtype=int)  # Edge connection information
        lnum = np.zeros(pall, dtype=int)  # Number of local regions per process
        lp2r = np.full((lall, pall), -1, dtype=int)  # Local process-to-region mapping

        try:
            data = toml.load(fname_in)
        except FileNotFoundError:
            prc.prc_mpistop(f"TOML file not found: {fname_in}")
            return None
        except toml.TomlDecodeError:
            prc.prc_mpistop(f"Error parsing TOML file: {fname_in}")
            return None

        # Read region number
        num_of_rgn = data["RGN_INFO"]["NUM_OF_RGN"]
        if num_of_rgn != rall:
            prc.prc_mpistop(f"Region count mismatch! Expected: {rall}, Found: {num_of_rgn}")
            return None

        # Read region connectivity
        for key, region in data["RGN_LINK_INFO"].items():
            rgnid = region["RGNID"]   # zero-based index
        
            edge_tab[:, 0, rgnid] = region["SW"]
            edge_tab[:, 1, rgnid] = region["NW"]
            edge_tab[:, 2, rgnid] = region["NE"]
            edge_tab[:, 3, rgnid] = region["SE"]

        # Read process number
        num_of_proc = data["PROC_INFO"]["NUM_OF_PROC"]
    
        if num_of_proc != pall:
            prc.prc_mpistop(f"Process count mismatch! Expected: {pall}, Found: {num_of_proc}")
            return None

        # Read process-region mapping
        for key, RGN_MNG in data["RGN_MNG_INFO"].items():
            peid = RGN_MNG["PEID"]  
            mng_rgnid = RGN_MNG["MNG_RGNID"] 

            lnum[peid] = len(mng_rgnid)
            if lnum[peid] > lall:
                prc.prc_mpistop(f"Too many local regions for Process {peid}: Found {lnum[peid]}, Limit {lall}")
                return None

            lp2r[:lnum[peid], peid] = mng_rgnid

        return edge_tab, lnum, lp2r


    def RGNMNG_vertex_walkaround(self): 

        self.RGNMNG_vert_num.fill(-1)
        self.RGNMNG_vert_tab.fill(-1)
        self.RGNMNG_vert_tab_pl.fill(-1)

        for r in range(self.ADM_rgn_nmax):  
            for d in range(self.I_W, self.I_S + 1):  # Loop from I_W to I_S (inclusive)

                rgnid = r

            # Select case equivalent in Python (match-case in Python 3.10+, else use if-elif)
                if d == self.I_W:
                    dir = self.I_SW
                elif d == self.I_N:
                    dir = self.I_NW
                elif d == self.I_E:
                    dir = self.I_NE
                elif d == self.I_S:
                    dir = self.I_SE

                v = -1

                while True:
                    rgnid_next = self.RGNMNG_edge_tab[self.I_RGNID, dir, rgnid]
                    dir_next = self.RGNMNG_edge_tab[self.I_DIR, dir, rgnid] - 1

                    if dir_next == -1:
                        dir_next = 3

                    v += 1
                    self.RGNMNG_vert_tab[self.I_RGNID, d, r, v] = rgnid
                    self.RGNMNG_vert_tab[self.I_DIR, d, r, v] = dir

                    rgnid = rgnid_next
                    dir = dir_next

                    if rgnid == r:
                        break

                self.RGNMNG_vert_num[d, r] = v+1

        for r in range(self.ADM_rgn_nmax):  
            if self.RGNMNG_vert_num[self.I_N, r] == self.ADM_vlink:
                for v in range(self.ADM_vlink):  
                    self.RGNMNG_vert_tab_pl[self.I_RGNID, self.I_NPL, v] = self.RGNMNG_vert_tab[self.I_RGNID, self.I_N, r, v]
                    self.RGNMNG_vert_tab_pl[self.I_DIR, self.I_NPL, v] = self.RGNMNG_vert_tab[self.I_DIR, self.I_N, r, v]
                break  


        for r in range(self.ADM_rgn_nmax):  
            if self.RGNMNG_vert_num[self.I_S, r] == self.ADM_vlink:
                for v in range(self.ADM_vlink):  
                    self.RGNMNG_vert_tab_pl[self.I_RGNID, self.I_SPL, v] = self.RGNMNG_vert_tab[self.I_RGNID, self.I_S, r, v]
                    self.RGNMNG_vert_tab_pl[self.I_DIR, self.I_SPL, v] = self.RGNMNG_vert_tab[self.I_DIR, self.I_S, r, v]
                break  

        return


    def output_info(self):
                
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("\n====== Process management info. ======", file=log_file)
                print(f"--- Total number of process           : {prc.prc_nprocs}", file=log_file)
                print(f"--- My Process number = (my rank) : {self.ADM_prc_me}", file=log_file)
                print("====== Region/Grid topology info. ======", file=log_file)
                print(f"--- Grid system                      : {self.ADM_HGRID_SYSTEM}", file=log_file)
                print(f"--- #  of diamond                     : {self.ADM_DMD}", file=log_file)
                print("====== Region management info. ======", file=log_file)
                print(f"--- Region level (RL)                 : {self.ADM_rlevel}", file=log_file)
                print(f"--- Total number of region            : {self.ADM_rgn_nmax} ({2**self.ADM_rlevel} x {2**self.ADM_rlevel} x {self.ADM_DMD})", file=log_file)
                print(f"--- #  of region per process          : {self.ADM_lall}", file=log_file)
                print(f"--- ID of region in my process        : {self.RGNMNG_lp2r[:self.ADM_lall, self.ADM_prc_me]}", file=log_file)        
                print(f"--- Region ID, contains north pole    : {self.RGNMNG_rgn4pl[self.I_NPL]}", file=log_file)
                print(f"--- Region ID, contains south pole    : {self.RGNMNG_rgn4pl[self.I_SPL]}", file=log_file)
                print(f"--- Process rank, managing north pole : {self.RGNMNG_r2p_pl[self.I_NPL]}", file=log_file)
                print(f"--- Process rank, managing south pole : {self.RGNMNG_r2p_pl[self.I_SPL]}", file=log_file)
                print("====== Grid management info. ======", file=log_file)
                print(f"--- Grid level (GL)                   : {self.ADM_glevel}", file=log_file)
                print(f"--- Total number of grid (horizontal) : {4**(self.ADM_glevel-self.ADM_rlevel)*self.ADM_rgn_nmax} ({2**(self.ADM_glevel-self.ADM_rlevel)} x {2**(self.ADM_glevel-self.ADM_rlevel)} x {self.ADM_rgn_nmax})", file=log_file)
                print(f"--- Number of vertical layer          : {self.ADM_kmax - self.ADM_kmin + 1}", file=log_file)
        
                debug = True
                if debug:
                    print("", file=log_file) 
                    print("====== Region Management Information ======", file=log_file)
                    print("", file=log_file) 
                    print(f"--- # of region in this node : {self.ADM_lall}", file=log_file)

                    print("--- (l,prc_me) => (rgn)", file=log_file)
                    for l in range(self.ADM_lall):
                        rgnid = self.RGNMNG_l2r[l]
                        print(f"--- ({l},{self.ADM_prc_me}) => ({rgnid})", file=log_file)

                    print("", file=log_file)     
                    print("--- Link information", file=log_file)
                    for l in range(self.ADM_lall):
                        rgnid = self.RGNMNG_l2r[l]

                        print("", file=log_file) 
                        print("--- edge link: (rgn,direction)", file=log_file)
                        for d in range(self.I_SW, self.I_SE + 1):
                            rgnid_next = self.RGNMNG_edge_tab[self.I_RGNID, d, rgnid]
                            dstr = self.RGNMNG_edgename[d]
                            dstr_next = self.RGNMNG_edgename[self.RGNMNG_edge_tab[self.I_DIR, d, rgnid]]
                            print(f"     ({rgnid},{dstr}) -> ({rgnid_next},{dstr_next})", file=log_file)
                            
                        print("--- vertex link: (rgn)", file=log_file)
                        for d in range(self.I_W, self.I_S + 1):
                            dstr = self.RGNMNG_vertname[d]
                            print(f"     ({rgnid},{dstr})", end="", file=log_file)
                            for v in range(1, self.RGNMNG_vert_num[d, rgnid]):
                                dstr = self.RGNMNG_vertname[self.RGNMNG_vert_tab[self.I_DIR, d, rgnid, v]]
                                print(f" -> ({self.RGNMNG_vert_tab[self.I_RGNID, d, rgnid, v]},{dstr})", end="", file=log_file)
                            print(file=log_file)
                        
                    print("--- Pole information (in the global scope)", file=log_file)

                    print(f"--- region, having north pole data : {self.RGNMNG_rgn4pl[self.I_NPL]}", file=log_file)
                    print("--- vertex link: (north pole)", file=log_file)
                    for v in range(1, self.ADM_vlink):
                        rgnid = self.RGNMNG_vert_tab_pl[self.I_RGNID, self.I_NPL, v]
                        dstr = self.RGNMNG_vertname[self.RGNMNG_vert_tab_pl[self.I_DIR, self.I_NPL, v]]
                        print(f" -> ({rgnid},{dstr})", end="", file=log_file)
                    print(file=log_file)
                    print(f"--- process, managing north pole : {self.RGNMNG_r2p_pl[self.I_NPL]}", file=log_file)

                    print(f"--- region, having south pole data : {self.RGNMNG_rgn4pl[self.I_SPL]}", file=log_file)
                    print("--- vertex link: (south pole)", file=log_file)
                    for v in range(1, self.ADM_vlink):
                        rgnid = self.RGNMNG_vert_tab_pl[self.I_RGNID, self.I_SPL, v]
                        dstr = self.RGNMNG_vertname[self.RGNMNG_vert_tab_pl[self.I_DIR, self.I_SPL, v]]
                        print(f" -> ({rgnid},{dstr})", end="", file=log_file)
                    print(file=log_file)
                    print(f"--- process, managing south pole : {self.RGNMNG_r2p_pl[self.I_SPL]}", file=log_file)


adm = Adm()
#print('instantiated adm')




