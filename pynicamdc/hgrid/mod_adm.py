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
    ADM_KNONE = 1

    # Dimension of the spatial vector
    ADM_nxyz = 3

    # maximum number of vertex linkage, ICO:5, PSP:6, LCP, MLCP:k
    ADM_vlink = 5

    # number of pole region
    ADM_rgn_nmax_pl =  2

    # number of pole region per process
    ADM_lall_pl     =  2

    # index for pole point
    ADM_gslf_pl     =  0  #? 
    # start index of grid around the pole point
    ADM_gmin_pl     =  1  #?    

    # total number of process
    ADM_prc_all = prc.prc_nprocs
    #ADM_prc_me  = prc.prc_myrank
    # master process
    ADM_prc_master = 0

    # maximum number of region per process
    RGNMNG_llim = 2560 

    def __init__(self):
        pass

    #def ADM_setup(self, io_l, io_nml, fname_log, fname_in):
    def ADM_setup(self, fname_in):
        #print("hoho0", "self.ADM_prc_me= ", self.ADM_prc_me)
        self.ADM_prc_me = prc.prc_myrank
        #print("hoho00", "self.ADM_prc_me= ", self.ADM_prc_me)
        #ADM_prc_pl = 0  # process 0 handles pole region

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[adm]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'admparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** cnstparam not found in toml file! STOP.", file=log_file)
                #stop

        else:
            glevel = cnfs['admparam']['glevel']  
            rlevel = cnfs['admparam']['rlevel']  
            vlayer = cnfs['admparam']['vlayer']  
            rgnmngfname = cnfs['admparam']['rgnmngfname']  
            ADM_HGRID_SYSTEM = cnfs['admparam']['ADM_HGRID_SYSTEM']  
            #ADM_vlink = cnfs['admparam']['ADM_vlink']  
            #ADM_XTMS_MLCP_S = cnfs['admparam']['ADM_XTMS_MLCP_S']  
            debug = cnfs['admparam']['debug']  

            if ( ADM_HGRID_SYSTEM == 'ICO' ):
                self.ADM_vlink  = 5
                dmd        = 10
                self.ADM_prc_pl = 0  # process 0 handles pole region

            else:
                with open(std.fname_log, 'a') as log_file:
                    print("xxx [ADM_setup] Not appropriate param for ADM_HGRID_SYSTEM. STOP.", ADM_HGRID_SYSTEM, file=log_file)
                    #call PRC_MPIstop


            self.ADM_gall_pl = self.ADM_vlink + 1
            self.ADM_gmax_pl = self.ADM_vlink # ? index of the last grid point

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
                self.ADM_kall = 1
                self.ADM_kmin = 1
                self.ADM_kmax = 1
            else:
                self.ADM_kall = 1 + self.ADM_vlayer + 1
                self.ADM_kmin = 1                # 1 + 1
                self.ADM_kmax = self.ADM_vlayer  # 1 + self.ADM_vlayer

            self.RGNMNG_setup(rgnmngfname)
            #edge_tab, lnum, lp2r = self.RGNMNG_setup(io_l, io_nml, fname_log, rgnmngfname)
            #, rall, pall, lall)

            #    allocate( GLOBAL_extension_rgn(ADM_lall) )
            #    allocate( ADM_have_sgp        (ADM_lall) )
            #GLOBAL_extension_rgn(:) = ''
            #ADM_have_sgp        (:) = .false.
            # Allocate and initialize GLOBAL_extension_rgn with empty strings
            self.GLOBAL_extension_rgn = np.empty(self.ADM_lall, dtype=str)
            self.GLOBAL_extension_rgn[:] = ''
            # Allocate and initialize ADM_have_sgp with False
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

            #self.output_info

        return

    def RGNMNG_setup(self, fname_in=None):
    
        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[rgnmng]", file=log_file)        

        if fname_in is None:
            with open(std.fname_log, 'a') as log_file:
                if std.io_l: print("*** input toml file is not specified. use default.", file=log_file)
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
                        print("*** rgnmngparam not specified in toml file. use default.", file=log_file)
                        # maybe should stop here instead of using default 
            else:
                if 'RGNMNG_in_fname' in cnfs['rgnmngparam']:
                    RGNMNG_in_fname = cnfs['rgnmngparam']['RGNMNG_in_fname']                    

                if 'RGNMNG_out_fname' in cnfs['rgnmngparam']:
                    RGNMNG_out_fname = cnfs['rgnmngparam']['RGNMNG_out_fname']      
               
            #self.ADM_lall = 10000
            if self.ADM_lall > self.RGNMNG_llim:
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file: 
                        print('xxx limit exceed! local region:', self.ADM_lall, self.RGNMNG_llim, file=log_file)
                    prc.prc_mpistop(std.io_l, std.fname_log)  #erronius

            #print(io_nml)
            #print("hahaha001", self.ADM_rgn_nmax, self.ADM_prc_all, self.ADM_lall)
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file: 
                        print(cnfs['rgnmngparam'],file=log_file)

        #print(RGNMNG_in_fname, self.ADM_rgn_nmax, self.ADM_prc_all, self.ADM_lall,'hoho')
        #self.edge_tab, self.lnum, self.lp2r = self.RGNMNG_input
        self.RGNMNG_edge_tab, self.RGNMNG_lnum, self.RGNMNG_lp2r = self.RGNMNG_input(RGNMNG_in_fname,self.ADM_rgn_nmax,self.ADM_prc_all,self.ADM_lall) #io_l, io_nml, fname_log, RGNMNG_in_fname)
        #print(RGNMNG_in_fname,'hoho')

        # RGNMNG_in_fname,        & ! [IN]
        #                          ADM_rgn_nmax,           & ! [IN]
        #                          PRC_nprocs,             & ! [IN]
        #                          RGNMNG_llim,            & ! [IN]
        #                          RGNMNG_edge_tab(:,:,:), & ! [OUT]
        #                          RGNMNG_lnum    (:),     & ! [OUT]
        #                          RGNMNG_lp2r    (:,:)    ) ! [OUT]

        #print(self.ADM_prc_me, self.ADM_prc_all, 'hoho33')
        #print(self.ADM_prc_me, self.RGNMNG_lnum, 'hoho34', self.ADM_lall)

        if self.RGNMNG_lnum[self.ADM_prc_me] != self.ADM_lall:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print('xxx limit exceed! local region:', self.RGNMNG_lnum(self.ADM_prc_me), self.ADM_lall, file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)


        self.RGNMNG_r2lp = np.empty((2, self.ADM_rgn_nmax), dtype=int)
        self.RGNMNG_l2r = np.empty(self.ADM_lall, dtype=int)

        for p in range(self.ADM_prc_all):  # Zero-based indexing (0 to PRC_nprocs-1)
            for l in range(self.ADM_lall):  # Zero-based indexing (0 to ADM_lall-1)
                self.RGNMNG_r2lp[self.I_l, self.RGNMNG_lp2r[l, p]] = l  # l already zero-based
                self.RGNMNG_r2lp[self.I_prc, self.RGNMNG_lp2r[l, p]] = p  # p already zero-based

        for l in range(self.ADM_lall):  # Zero-based indexing (0 to ADM_lall-1)
            self.RGNMNG_l2r[l] = self.RGNMNG_lp2r[l, self.ADM_prc_me]


                        # Allocate arrays (Fortran allocate -> NumPy array initialization)
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
                
                #print("hoho0001", self.ADM_prc_me)
        self.RGNMNG_vertex_walkaround()
                #(
                #    self.ADM_rgn_nmax,              # [IN]
                #    self.ADM_rgn_nmax_pl,           # [IN]
                #    self.ADM_vlink,                 # [IN]
                #    self.RGNMNG_edge_tab,           # [IN] (assumed pre-defined as a NumPy array)
                #    self.RGNMNG_vert_num,           # [OUT]
                #    self.RGNMNG_vert_tab,           # [OUT]
                #    self.RGNMNG_vert_tab_pl         # [OUT]
                #)

        #print(self.RGNMNG_vert_tab[0:2,0:4,0:5,0], "ho0")#print("hoho, end walkaround", self.ADM_prc_me)
        #print(self.RGNMNG_vert_tab[0:2,0:4,0:5,1], "ho1")#print("hoho, end walkaround", self.ADM_prc_me)
        #print(self.RGNMNG_vert_tab[0:2,0:4,0:5,2], "ho2")#print("hoho, end walkaround", self.ADM_prc_me)
        #print(self.RGNMNG_vert_tab[0:2,0:4,0:5,3], "ho3")#print("hoho, end walkaround", self.ADM_prc_me)
        #print(self.RGNMNG_vert_tab[0:2,0:4,0:5,4], "ho4")#print("hoho, end walkaround", self.ADM_prc_me)

                #print(self.RGNMNG_vert_num[0:5,0:6], 'vert_num')

                # Conditional statement
                #if RGNMNG_vert_num[I_W, rgnid] == 3:

        #print("walkaround done")
        #print(self.I_NPL, self.I_SPL)
        self.RGNMNG_rgn4pl = np.empty((self.I_SPL - self.I_NPL + 1), dtype=int) 
        self.RGNMNG_r2p_pl = np.empty((self.I_SPL - self.I_NPL + 1), dtype=int) 
        #print(self.RGNMNG_rgn4pl[1])
        # First loop (Fortran: do r = 1, ADM_rgn_nmax)
        for r in range(self.ADM_rgn_nmax):  # Zero-based indexing
            if self.RGNMNG_vert_num[self.I_N, r] == self.ADM_vlink:
                self.RGNMNG_rgn4pl[self.I_NPL] = r
                break  # Equivalent to Fortran's exit

        # Second loop (Fortran: do r = 1, ADM_rgn_nmax)
        for r in range(self.ADM_rgn_nmax):  # Zero-based indexing
            if self.RGNMNG_vert_num[self.I_S, r] == self.ADM_vlink:
                self.RGNMNG_rgn4pl[self.I_SPL] = r
                break  # Equivalent to Fortran's exit

        #print(self.RGNMNG_rgn4pl[0])
        #print(self.RGNMNG_rgn4pl[1])

    # Assign values after loops
        self.RGNMNG_r2p_pl[self.I_NPL] = self.ADM_prc_pl
        self.RGNMNG_r2p_pl[self.I_SPL] = self.ADM_prc_pl

        #print(self.RGNMNG_r2p_pl[0])
        #print(self.RGNMNG_r2p_pl[1])

        return
    

    def output_info(self):
        pass

    def RGNMNG_input(self, fname_in, rall, pall, lall):
        #import numpy as np
        #import toml
        #from mpi4py import MPI
        #print(fname_in,'hoho')
        #def RGNMNG_input(in_fname, rall, pall, lall):
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
        #prc = Process()

        # Initialize arrays
        edge_tab = np.full((2, 4, rall), -1, dtype=int)  # Edge connection information
        lnum = np.zeros(pall, dtype=int)  # Number of local regions per process
        lp2r = np.full((lall, pall), -1, dtype=int)  # Local process-to-region mapping

        #print(f"*** Reading input management info from TOML file: {fname_in}")

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

            #print(key, rgnid, region, self.edge_tab[:, 0, rgnid], "hohoho100?")
            #print(key, rgnid, self.edge_tab[:, 0, rgnid], "hohoho100?")
        #print(edge_tab, "hohoho100?")

        # Read process number
        num_of_proc = data["PROC_INFO"]["NUM_OF_PROC"]
        #print(num_of_proc, rall, pall, "hohoho?")

        if num_of_proc != pall:
            prc.prc_mpistop(f"Process count mismatch! Expected: {pall}, Found: {num_of_proc}")
            return None

        # Read process-region mapping
        for key, RGN_MNG in data["RGN_MNG_INFO"].items():
            peid = RGN_MNG["PEID"]  # zero-based index
            mng_rgnid = RGN_MNG["MNG_RGNID"]  # zero-based index

            #print(peid, mng_rgnid, len(mng_rgnid), lall, "hohoho200?")
            lnum[peid] = len(mng_rgnid)
            #print('ha', lnum[peid], "hohoho300?")
            if lnum[peid] > lall:
                prc.prc_mpistop(f"Too many local regions for Process {peid}: Found {lnum[peid]}, Limit {lall}")
                return None

            lp2r[:lnum[peid], peid] = mng_rgnid

        return edge_tab, lnum, lp2r


    def RGNMNG_vertex_walkaround(self): 
        #, ADM_rgn_nmax, ADM_rgn_nmax_pl, ADM_vlink, RGNMNG_edge_tab, 
                                 #RGNMNG_vert_num, RGNMNG_vert_tab, RGNMNG_vert_tab_pl):

        #print("hoho, start walkaround", self.ADM_prc_me)
        self.RGNMNG_vert_num.fill(-1)
        self.RGNMNG_vert_tab.fill(-1)
        self.RGNMNG_vert_tab_pl.fill(-1)

        for r in range(self.ADM_rgn_nmax):  # Zero-based: 0 to rall-1
            #print("r", r, self.ADM_rgn_nmax)
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
                    #print("walking around", r, d, rgnid, dir, self.I_RGNID, self.I_DIR)
                    rgnid_next = self.RGNMNG_edge_tab[self.I_RGNID, dir, rgnid]
                    dir_next = self.RGNMNG_edge_tab[self.I_DIR, dir, rgnid] - 1

                    if dir_next == -1:
                        #print('he')
                        dir_next = 3

                    v += 1
                    #print("dims", d, r, v, rgnid, dir)
                    self.RGNMNG_vert_tab[self.I_RGNID, d, r, v] = rgnid
                    self.RGNMNG_vert_tab[self.I_DIR, d, r, v] = dir

                    rgnid = rgnid_next
                    dir = dir_next

                    #print(rgnid, r, 'break?')
                    if rgnid == r:
                        #print(rgnid, r, 'break')
                        break

                self.RGNMNG_vert_num[d, r] = v+1

        #print(self.RGNMNG_vert_num[0:5,0:5], "ho1")#print("hoho, end walkaround", self.ADM_prc_me) 
        #print(self.RGNMNG_vert_num[0:5,0], "ho0")#print("hoho, end walkaround", self.ADM_prc_me) 
        #print(self.RGNMNG_vert_num[0:5,1], "ho1")#print("hoho, end walkaround", self.ADM_prc_me)    
        #print(self.RGNMNG_vert_num[0:5,2], "ho2")#print("hoho, end walkaround", self.ADM_prc_me)
        #print(self.RGNMNG_vert_num[0:5,3], "ho3")#print("hoho, end walkaround", self.ADM_prc_me)
        #print(self.RGNMNG_vert_num[0:5,4], "ho4")#print("hoho, end walkaround", self.ADM_prc_me)

        for r in range(self.ADM_rgn_nmax):  # Zero-based indexing
            if self.RGNMNG_vert_num[self.I_N, r] == self.ADM_vlink:
                for v in range(self.ADM_vlink):  # Zero-based indexing
                    self.RGNMNG_vert_tab_pl[self.I_RGNID, self.I_NPL, v] = self.RGNMNG_vert_tab[self.I_RGNID, self.I_N, r, v]
                    self.RGNMNG_vert_tab_pl[self.I_DIR, self.I_NPL, v] = self.RGNMNG_vert_tab[self.I_DIR, self.I_N, r, v]
                break  # Exit loop when condition is met


        for r in range(self.ADM_rgn_nmax):  # Zero-based indexing
            if self.RGNMNG_vert_num[self.I_S, r] == self.ADM_vlink:
                for v in range(self.ADM_vlink):  # Zero-based indexing
                    self.RGNMNG_vert_tab_pl[self.I_RGNID, self.I_SPL, v] = self.RGNMNG_vert_tab[self.I_RGNID, self.I_S, r, v]
                    self.RGNMNG_vert_tab_pl[self.I_DIR, self.I_SPL, v] = self.RGNMNG_vert_tab[self.I_DIR, self.I_S, r, v]
                break  # Exit loop when condition is met

        return
        # Placeholder for the function's logic
        #pass


adm = Adm()
print('instantiated adm')




