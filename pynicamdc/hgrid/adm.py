import toml
import numpy as np
from process import prc

class Adm:
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

    # master process
    ADM_prc_master = 0

    # maximum number of region per process
    RGNMNG_llim = 2560 

    def __init__(self):
        pass

    def ADM_setup(self, io_l, io_nml, fname_log, fname_in):
        ADM_prc_me = prc.prc_myrank
        #ADM_prc_pl = 0  # process 0 handles pole region

        if io_l: 
            with open(fname_log, 'a') as log_file:
                print("+++ Module[adm]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'admparam' not in cnfs:
            with open(fname_log, 'a') as log_file:
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
                ADM_vlink  = 5
                dmd        = 10
                ADM_prc_pl = 0  # process 0 handles pole region

            else:
                with open(fname_log, 'a') as log_file:
                    print("xxx [ADM_setup] Not appropriate param for ADM_HGRID_SYSTEM. STOP.", ADM_HGRID_SYSTEM, file=log_file)
                    #call PRC_MPIstop


            ADM_gall_pl = ADM_vlink + 1
            ADM_gmax_pl = ADM_vlink # ? index of the last grid point

            self.ADM_glevel = glevel
            self.ADM_rlevel = rlevel
            self.ADM_vlayer = vlayer
            self.ADM_DMD = dmd

            # Calculations
            self.ADM_rgn_nmax = (2 ** self.ADM_rlevel) * (2 ** self.ADM_rlevel) * self.ADM_DMD
            self.ADM_lall = self.ADM_rgn_nmax // prc.prc_nprocs
            print("hahaha000 ", self.ADM_rgn_nmax, prc.prc_nprocs,self.ADM_lall)
            nmax = 2 ** (self.ADM_glevel - self.ADM_rlevel)
            self.ADM_gall_1d = 1 + nmax + 1
            self.ADM_gmin = 1 + 1
            self.ADM_gmax = 1 + nmax

            self.ADM_gall = (1 + nmax + 1) * (1 + nmax + 1)
            self.ADM_gall_in = (nmax + 1) * (nmax + 1)

            if self.ADM_vlayer == 1:
                self.ADM_kall = 1
                self.ADM_kmin = 1
                self.ADM_kmax = 1
            else:
                self.ADM_kall = 1 + self.ADM_vlayer + 1
                self.ADM_kmin = 1 + 1
                self.ADM_kmax = 1 + self.ADM_vlayer

            self.RGNMNG_setup(io_l, io_nml, fname_log, rgnmngfname)
            #edge_tab, lnum, lp2r = self.RGNMNG_setup(io_l, io_nml, fname_log, rgnmngfname)
            #, rall, pall, lall)

            #    allocate( GLOBAL_extension_rgn(ADM_lall) )
            #    allocate( ADM_have_sgp        (ADM_lall) )
            #GLOBAL_extension_rgn(:) = ''
            #ADM_have_sgp        (:) = .false.
            # Allocate and initialize GLOBAL_extension_rgn with empty strings
            GLOBAL_extension_rgn = np.empty(self.ADM_lall, dtype=str)
            GLOBAL_extension_rgn[:] = ''
            # Allocate and initialize ADM_have_sgp with False
            ADM_have_sgp = np.full(self.ADM_lall, False, dtype=bool)

            #do l = 1, ADM_lall
            #rgnid = RGNMNG_lp2r(l,ADM_prc_me)

            #write(GLOBAL_extension_rgn(l),'(A,I5.5)') '.rgn', rgnid-1

            #    if ( RGNMNG_vert_num(I_W,rgnid) == 3 ) then
            #        ADM_have_sgp(l) = .true.
            #    endif
            #enddo

            for l in range(self.ADM_lall):  
            # Python's range starts from 0 by default, adjust according to your array indexing
                rgnid = self.RGNMNG_lp2r[l, ADM_prc_me]

                # Formatting the string with '.rgn' prefix and the (rgnid-1) value
                #GLOBAL_extension_rgn[l] = f".rgn{rgnid-1:05d}"
                GLOBAL_extension_rgn[l] = f".rgn{rgnid:05d}"

                # Conditional statement
                #if RGNMNG_vert_num[I_W, rgnid] == 3:
                if self.RGNMNG_vert_num[self.I_W, rgnid+1] == 3:
                    ADM_have_sgp[l] = True

            if ADM_prc_me == ADM_prc_pl:
                ADM_have_pl = True
            else:  
                ADM_have_pl = False

            ADM_l_me = 0

            self.output_info

        return

    def RGNMNG_setup(self, io_l, io_nml, fname_log, fname_in=None):
    
        if io_l: 
            with open(fname_log, 'a') as log_file:
                print("+++ Module[rgnmng]", file=log_file)        

        if fname_in is None:
            with open(fname_log, 'a') as log_file:
                if io_l: print("*** input toml file is not specified. use default.", file=log_file)
                # maybe should stop here instead of using default
        else:
            if io_l:
                with open(fname_log, 'a') as log_file: 
                    print("*** input toml file is ", fname_in, file=log_file)

            with open(fname_in, 'r') as  file:
                cnfs = toml.load(file)

            if 'rgnmngparam' not in cnfs:
                if io_l:
                    with open(fname_log, 'a') as log_file: 
                        print("*** rgnmngparam not specified in toml file. use default.", file=log_file)
                        # maybe should stop here instead of using default 
            else:
                if 'RGNMNG_in_fname' in cnfs['rgnmngparam']:
                    RGNMNG_in_fname = cnfs['rgnmngparam']['RGNMNG_in_fname']                    

                if 'RGNMNG_out_fname' in cnfs['rgnmngparam']:
                    RGNMNG_out_fname = cnfs['rgnmngparam']['RGNMNG_out_fname']      
               
            #self.ADM_lall = 10000
            if self.ADM_lall > self.RGNMNG_llim:
                if io_l:
                    with open(fname_log, 'a') as log_file: 
                        print('xxx limit exceed! local region:', self.ADM_lall, self.RGNMNG_llim, file=log_file)
                    prc.prc_mpistop(io_l, fname_log)  #erronius

            print(io_nml)
            print("hahaha001", self.ADM_rgn_nmax, self.ADM_prc_all, self.ADM_lall)
            if io_nml: 
                if io_l:
                    with open(fname_log, 'a') as log_file: 
                        print(cnfs['rgnmngparam'],file=log_file)

        print(RGNMNG_in_fname, self.ADM_rgn_nmax, self.ADM_prc_all, self.ADM_lall,'hoho')
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


    def output_info(self):
        pass

    def RGNMNG_input(self, fname_in, rall, pall, lall):
        #import numpy as np
        #import toml
        #from mpi4py import MPI
        print(fname_in,'hoho')
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

        print(f"*** Reading input management info from TOML file: {fname_in}")

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
        print(num_of_proc, rall, pall, "hohoho?")

        if num_of_proc != pall:
            prc.prc_mpistop(f"Process count mismatch! Expected: {pall}, Found: {num_of_proc}")
            return None

        # Read process-region mapping
        for key, RGN_MNG in data["RGN_MNG_INFO"].items():
            peid = RGN_MNG["PEID"]  # zero-based index
            mng_rgnid = RGN_MNG["MNG_RGNID"]  # zero-based index

            print(peid, mng_rgnid, len(mng_rgnid), lall, "hohoho200?")
            lnum[peid] = len(mng_rgnid)
            print('ha', lnum[peid], "hohoho300?")
            if lnum[peid] > lall:
                prc.prc_mpistop(f"Too many local regions for Process {peid}: Found {lnum[peid]}, Limit {lall}")
                return None

            lp2r[:lnum[peid], peid] = mng_rgnid

        return edge_tab, lnum, lp2r



    # You can add methods here as needed.

