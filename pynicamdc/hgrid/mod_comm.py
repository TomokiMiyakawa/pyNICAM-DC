import toml
import numpy as np
#from mod_adm 
from mod_adm import adm
#from mod_stdio import std 

class Comm:

    _instance = None

    #rellist_vindex = 6
    rellist_vindex = 8  #to use i j array instead of ij array
    I_recv_grid_i = 0
    I_recv_grid_j = 1   
    I_recv_rgn    = 2
    I_recv_prc    = 3
    I_send_grid_i = 4
    I_send_grid_j = 5
    I_send_rgn    = 6
    I_send_prc    = 7
    
    info_vindex = 3
    I_size      = 0
    I_prc_from  = 1
    I_prc_to    = 2
    
    list_vindex = 4
    I_grid_from = 0
    I_l_from    = 1
    I_grid_to   = 2
    I_l_to      = 3

    def __init__(self):
        pass   

    #def COMM_setup(self, io_l, io_nml, fname_log, fname_in, adm):
    def COMM_setup(self, io_l, io_nml, fname_log, fname_in):

        #from mod_adm import adm 

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
                    print(cnfs['commparam'],file=log_file)
        
#        if ( RP == DP ):
#            COMM_datatype = MPI_DOUBLE_PRECISION
#        elseif( RP == SP ) then
#            COMM_datatype = MPI_REAL
#        else    
#            write(*,*) 'xxx precision is not supportd'
#            call PRC_MPIstop
#        endif

        #adm = Adm()

        #print("adm.ADM_prc_me= ", adm.ADM_prc_me)     
        #if adm.RGNMNG_r2p_pl[adm.I_NPL] < 0 and adm.RGNMNG_r2p_pl[adm.I_SPL] < 0:
        #    self.COMM_pl = False  # Fortran .false. → Python False

        # Equivalent to Fortran's write(IO_FID_LOG,*) statements
        if io_l:
            with open(fname_log, 'a') as log_file: 
                print("", file=log_file)  # Equivalent to blank write line
                print("====== communication information ======", file=log_file)

        self.COMM_list_generate(io_l, io_nml, fname_log, fname_in)


    def COMM_list_generate(self, io_l, io_nml, fname_log, fname_in):
        print("COMM_list_generate")

        ginner = adm.ADM_gmax - adm.ADM_gmin + 1

        # Allocate rellist (Fortran allocate -> NumPy array initialization)
        self.rellist = np.empty((self.rellist_vindex, adm.ADM_gall * adm.ADM_lall), dtype=int)

        cnt = 0

        for l in range(adm.ADM_lall):  
            rgnid = adm.RGNMNG_l2r[l]
            prc = adm.ADM_prc_me

            #print("rgnid= ", rgnid)
            #print("prc= ", prc)  

            # ---< South West >---
            # NE -> SW halo
            if adm.RGNMNG_edge_tab[adm.I_DIR, adm.I_SW, rgnid] == adm.I_NE:
                rgnid_rmt = adm.RGNMNG_edge_tab[adm.I_RGNID, adm.I_SW, rgnid]
                prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]

                for n in range(ginner):  # Adjust for zero-based indexing
                    #print("n= ", n)

                    cnt += 1

                    #i = adm.ADM_gmin - 1 + n
                    i = adm.ADM_gmin + n
                    j = adm.ADM_gmin - 1
                    #i_rmt = adm.ADM_gmin - 1 + n
                    i_rmt = adm.ADM_gmin + n
                    j_rmt = adm.ADM_gmax

                    self.rellist[self.I_recv_grid_i, cnt] = i  #self.suf(i, j, adm)
                    self.rellist[self.I_recv_grid_j, cnt] = j  
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_grid_i, cnt] = i_rmt #self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_grid_j, cnt] = j_rmt
                    self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                    self.rellist[self.I_send_prc, cnt] = prc_rmt

            # SE -> SW halo (Southern Hemisphere, Edge of diamond)
            if adm.RGNMNG_edge_tab[adm.I_DIR, adm.I_SW, rgnid] == adm.I_SE:
                rgnid_rmt = adm.RGNMNG_edge_tab[adm.I_RGNID, adm.I_SW, rgnid]
                prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]

                for n in range(ginner):  # Adjust for zero-based indexing
                    cnt += 1

                    i = adm.ADM_gmin + n
                    j = adm.ADM_gmin - 1
                    i_rmt = adm.ADM_gmax
                    j_rmt = adm.ADM_gmax - n  # Reverse order

                    self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_grid_j, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_grid_j, cnt] = j_rmt
                    self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                    self.rellist[self.I_send_prc, cnt] = prc_rmt
                
            #---< North West >---
            # SE -> NW 
            if adm.RGNMNG_edge_tab[adm.I_DIR, adm.I_NW, rgnid] == adm.I_SE:
                rgnid_rmt = adm.RGNMNG_edge_tab[adm.I_RGNID, adm.I_NW, rgnid]
                prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]

                for n in range(ginner):  # Adjust for zero-based indexing
                    cnt += 1

                    i = adm.ADM_gmin - 1
                    j = adm.ADM_gmin + n
                    i_rmt = adm.ADM_gmax
                    j_rmt = adm.ADM_gmin + n

                    self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_grid_j, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_grid_j, cnt] = j_rmt
                    self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                    self.rellist[self.I_send_prc, cnt] = prc_rmt

            # NE -> NW  (Northern Hemisphere, Edge of diamond)
            if adm.RGNMNG_edge_tab[adm.I_DIR, adm.I_NW, rgnid] == adm.I_NE:
                rgnid_rmt = adm.RGNMNG_edge_tab[adm.I_RGNID, adm.I_NW, rgnid]
                prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]

                for n in range(ginner):  # Adjust for zero-based indexing
                    cnt += 1

                    i = adm.ADM_gmin - 1
                    j = adm.ADM_gmin + n
                    i_rmt = adm.ADM_gmax - n  # Reverse order
                    j_rmt = adm.ADM_gmax

                    self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_grid_j, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_grid_j, cnt] = j_rmt
                    self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                    self.rellist[self.I_send_prc, cnt] = prc_rmt

            #---< North East >---
            # SW -> NE 
            if adm.RGNMNG_edge_tab[adm.I_DIR, adm.I_NE, rgnid] == adm.I_SW:
                rgnid_rmt = adm.RGNMNG_edge_tab[adm.I_RGNID, adm.I_NE, rgnid]
                prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]

                for n in range(ginner):  # Adjust for zero-based indexing
                    cnt += 1

                    i = adm.ADM_gmin + n
                    j = adm.ADM_gmax + 1
                    i_rmt = adm.ADM_gmin + n
                    j_rmt = adm.ADM_gmin

                    self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_grid_j, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_grid_j, cnt] = j_rmt
                    self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                    self.rellist[self.I_send_prc, cnt] = prc_rmt

            # NW -> NE  (Northern Hemisphere, Edge of diamond)
            if adm.RGNMNG_edge_tab[adm.I_DIR, adm.I_NE, rgnid] == adm.I_NW:
                rgnid_rmt = adm.RGNMNG_edge_tab[adm.I_RGNID, adm.I_NE, rgnid]
                prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]

                for n in range(ginner):  # Adjust for zero-based indexing
                    cnt += 1

                    i = adm.ADM_gmin + 1 + n  # Shift 1 grid !  (1,17) is handled as the north vertex. (2:17,17) is handled here (gl05rl01)
                    j = adm.ADM_gmax + 1
                    i_rmt = adm.ADM_gmin 
                    j_rmt = adm.ADM_gmax - n  # Reverse order

                    self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_grid_j, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_grid_j, cnt] = j_rmt
                    self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                    self.rellist[self.I_send_prc, cnt] = prc_rmt

            #---< South East >---
            # NW -> SE 
            if adm.RGNMNG_edge_tab[adm.I_DIR, adm.I_SE, rgnid] == adm.I_NW:
                rgnid_rmt = adm.RGNMNG_edge_tab[adm.I_RGNID, adm.I_SE, rgnid]
                prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]

                for n in range(ginner):  # Adjust for zero-based indexing
                    cnt += 1

                    i = adm.ADM_gmax + 1
                    j = adm.ADM_gmin + n
                    i_rmt = adm.ADM_gmin
                    j_rmt = adm.ADM_gmin + n

                    self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_grid_j, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_grid_j, cnt] = j_rmt
                    self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                    self.rellist[self.I_send_prc, cnt] = prc_rmt

            # SW -> SE  (Southern Hemisphere, Edge of diamond)
            if adm.RGNMNG_edge_tab[adm.I_DIR, adm.I_SE, rgnid] == adm.I_SW:
                rgnid_rmt = adm.RGNMNG_edge_tab[adm.I_RGNID, adm.I_SE, rgnid]
                prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]

                for n in range(ginner):  # Adjust for zero-based indexing
                    cnt += 1

                    i = adm.ADM_gmax + 1
                    j = adm.ADM_gmin + 1 + n  # Shift 1 grid !!!!!
                    i_rmt = adm.ADM_gmax - n  # Reverse order
                    j_rmt = adm.ADM_gmin

                    self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_grid_j, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_grid_j, cnt] = j_rmt
                    self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                    self.rellist[self.I_send_prc, cnt] = prc_rmt


            #---< Vertex : link to the next next region >---
            # West Vertex
            if adm.RGNMNG_vert_num[adm.I_W, rgnid] == 4:  # 4 regions around the vertex
                rgnid_rmt = adm.RGNMNG_vert_tab[adm.I_RGNID, adm.I_W, rgnid, 2]  # 0 is yourself, 2 is the next next region when 4 regions around the vertex (clockwise)
                prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]

                cnt += 1

                i = adm.ADM_gmin - 1
                j = adm.ADM_gmin - 1

                if adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_W, rgnid, 2] == adm.I_N:
                    i_rmt = adm.ADM_gmin
                    j_rmt = adm.ADM_gmax
                elif adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_W, rgnid, 2] == adm.I_E:
                    i_rmt = adm.ADM_gmax
                    j_rmt = adm.ADM_gmax
                elif adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_W, rgnid, 2] == adm.I_S:
                    i_rmt = adm.ADM_gmax   
                    j_rmt = adm.ADM_gmin

                self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
                self.rellist[self.I_recv_grid_j, cnt] = j
                self.rellist[self.I_recv_rgn, cnt] = rgnid
                self.rellist[self.I_recv_prc, cnt] = prc
                self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                self.rellist[self.I_send_grid_j, cnt] = j_rmt
                self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                self.rellist[self.I_send_prc, cnt] = prc_rmt

            # North Vertex
            if adm.RGNMNG_vert_num[adm.I_N, rgnid] == 4:
                rgnid_rmt = adm.RGNMNG_vert_tab[adm.I_RGNID, adm.I_N, rgnid, 2]
                prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]

                # Known as north pole point (not the north pole of the Earth)
                if adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_N, rgnid, 2] == adm.I_W:   
                    cnt += 1

                    i = adm.ADM_gmin
                    j = adm.ADM_gmax + 1
                    i_rmt = adm.ADM_gmin
                    j_rmt = adm.ADM_gmin

                    self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_grid_j, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_grid_j, cnt] = j_rmt
                    self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                    self.rellist[self.I_send_prc, cnt] = prc_rmt

                    #if(rgnid == 7):
                    #    print("0:cnt= ", cnt) 
                    #    print("i, j, rgnid, prc, i_rmt, j_rmt, rgnid_rmt, prc_rmt= ", i, j, rgnid, prc, i_rmt, j_rmt, rgnid_rmt, prc_rmt)

                # Unused vertex point    ! What for??
                cnt += 1

                i = adm.ADM_gmin - 1
                j = adm.ADM_gmax + 1

                if adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_N, rgnid, 2] == adm.I_W:
                    i_rmt = adm.ADM_gmin
                    j_rmt = adm.ADM_gmin + 1
                elif adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_N, rgnid, 2] == adm.I_N:
                    i_rmt = adm.ADM_gmin
                    j_rmt = adm.ADM_gmax
                elif adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_N, rgnid, 2] == adm.I_E:
                    i_rmt = adm.ADM_gmax
                    j_rmt = adm.ADM_gmax
                elif adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_N, rgnid, 2] == adm.I_S:
                    i_rmt = adm.ADM_gmax
                    j_rmt = adm.ADM_gmin

                self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
                self.rellist[self.I_recv_grid_j, cnt] = j
                self.rellist[self.I_recv_rgn, cnt] = rgnid
                self.rellist[self.I_recv_prc, cnt] = prc
                self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                self.rellist[self.I_send_grid_j, cnt] = j_rmt
                self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                self.rellist[self.I_send_prc, cnt] = prc_rmt

                #if(rgnid == 7):
                #    print("cnt= ", cnt)
                #    print("i, j, rgnid, prc, i_rmt, j_rmt, rgnid_rmt, prc_rmt= ", i, j, rgnid, prc, i_rmt, j_rmt, rgnid_rmt, prc_rmt)

            # East Vertex
            if adm.RGNMNG_vert_num[adm.I_E, rgnid] == 4:
                if adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_E, rgnid, 2] == adm.I_W:
                    rgnid_rmt = adm.RGNMNG_vert_tab[adm.I_RGNID, adm.I_E, rgnid, 2]
                    prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]

                    cnt += 1

                    i = adm.ADM_gmax + 1
                    j = adm.ADM_gmax + 1
                    i_rmt = adm.ADM_gmin
                    j_rmt = adm.ADM_gmin

                    self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_grid_j, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_grid_j, cnt] = j_rmt
                    self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                    self.rellist[self.I_send_prc, cnt] = prc_rmt

            # South Vertex        
            if adm.RGNMNG_vert_num[adm.I_S, rgnid] == 4:
                rgnid_rmt = adm.RGNMNG_vert_tab[adm.I_RGNID, adm.I_S, rgnid, 2]
                prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]

                # Known as south pole point (not the south pole of the Earth)
                if adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_S, rgnid, 2] == adm.I_W:
                    cnt += 1

                    i = adm.ADM_gmax + 1
                    j = adm.ADM_gmin
                    i_rmt = adm.ADM_gmin
                    j_rmt = adm.ADM_gmin

                    self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_grid_j, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_grid_j, cnt] = j_rmt
                    self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
                    self.rellist[self.I_send_prc, cnt] = prc_rmt

            # Unused vertex point   ! Again, what for??
            cnt += 1

            i = adm.ADM_gmax + 1
            j = adm.ADM_gmin - 1

            if adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_S, rgnid, 2] == adm.I_W:
                i_rmt = adm.ADM_gmin + 1
                j_rmt = adm.ADM_gmin
            elif adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_S, rgnid, 2] == adm.I_N:
                i_rmt = adm.ADM_gmin
                j_rmt = adm.ADM_gmax
            elif adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_S, rgnid, 2] == adm.I_E:
                i_rmt = adm.ADM_gmax
                j_rmt = adm.ADM_gmax
            elif adm.RGNMNG_vert_tab[adm.I_DIR, adm.I_S, rgnid, 2] == adm.I_S:
                i_rmt = adm.ADM_gmax
                j_rmt = adm.ADM_gmin

            self.rellist[self.I_recv_grid_i, cnt] = i  # self.suf(i, j, adm)
            self.rellist[self.I_recv_grid_j, cnt] = j
            self.rellist[self.I_recv_rgn, cnt] = rgnid
            self.rellist[self.I_recv_prc, cnt] = prc
            self.rellist[self.I_send_grid_i, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
            self.rellist[self.I_send_grid_j, cnt] = j_rmt
            self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
            self.rellist[self.I_send_prc, cnt] = prc_rmt


        self.rellist_nmax = cnt

        if io_l:
            with open(fname_log, 'a') as log_file:
                print(f'*** rellist_nmax: {self.rellist_nmax}', file=log_file)


        #if debug:
        #    if adm.IO_L:
        #        print('--- Relation Table', file=IO_FID_LOG)
        #        print(f"{'Count':>10} {'|recv_grid':>10} {'|recv_rgn':>10} {'|recv_prc':>10} "
        #              f"{'|send_grid':>10} {'|send_rgn':>10} {'|send_prc':>10}", file=IO_FID_LOG)

        #    for cnt in range(1, self.rellist_nmax + 1):  # Adjust for zero-based indexing in Python
        #        if IO_L:
        #            print(f"{cnt:10} {' '.join(f'{val:10}' for val in self.rellist[:, cnt])}", file=IO_FID_LOG)

        return


    def COMM_sortdest(self):
        pass    

    def COMM_sortdest_pl(self):
        pass

    def COMM_sortdest_singular(self):
        pass

    #def suf(self, i, j, adm):
    #    return adm.ADM_gall_1d * j + i 
