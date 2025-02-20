import toml
import numpy as np
from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc

class Comm:

    _instance = None

    #rellist_vindex = 6
    rellist_vindex = 8  #to use i j array instead of ij array
    I_recv_gridi  = 0
    I_recv_gridj  = 1   
    I_recv_rgn    = 2
    I_recv_prc    = 3
    I_send_gridi  = 4
    I_send_gridj  = 5
    I_send_rgn    = 6
    I_send_prc    = 7
    
    Recv_nlim = 20  # number limit of rank to receive data
    Send_nlim = 20  # number limit of rank to send data

    info_vindex = 3
    I_size      = 0
    I_prc_from  = 1
    I_prc_to    = 2
    
    list_vindex  = 6
    I_gridi_from = 0
    I_gridj_from = 1
    I_l_from     = 2
    I_gridi_to   = 3
    I_gridj_to   = 4
    I_l_to       = 5

    def __init__(self):

        self.Copy_nmax_r2r = 0
        self.Recv_nmax_r2r = 0
        self.Send_nmax_r2r = 0
        Copy_nmax_p2r = 0
        Recv_nmax_p2r = 0
        Send_nmax_p2r = 0
        Copy_nmax_r2p = 0
        Recv_nmax_r2p = 0
        Send_nmax_r2p = 0
        Singular_nmax = 0


    def COMM_setup(self, fname_in):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[comm]/Category[common share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'commparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** commparam not found in toml file! STOP.", file=log_file)
                #stop

        else:
            self.COMM_apply_barrier = cnfs['commparam']['COMM_apply_barrier']  
            self.COMM_varmax = cnfs['commparam']['COMM_varmax']  
            debug = cnfs['commparam']['debug']  
            testonly = cnfs['commparam']['testonly']  

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
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
        if std.io_l:
            with open(std.fname_log, 'a') as log_file: 
                print("", file=log_file)  # Equivalent to blank write line
                print("====== communication information ======", file=log_file)

        self.COMM_list_generate()

        self.COMM_sortdest()


    def COMM_list_generate(self):
        print("COMM_list_generate")

        ginner = adm.ADM_gmax - adm.ADM_gmin + 1

        # Allocate rellist (Fortran allocate -> NumPy array initialization)
        self.rellist = np.empty((self.rellist_vindex, adm.ADM_gall * adm.ADM_lall), dtype=int)

        #cnt = 0
        cnt = -1  # Adjust for zero-based indexing in Python    

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

                    self.rellist[self.I_recv_gridi, cnt] = i  #self.suf(i, j, adm)
                    self.rellist[self.I_recv_gridj, cnt] = j  
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_gridi, cnt] = i_rmt #self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_gridj, cnt] = j_rmt
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

                    self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_gridj, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_gridj, cnt] = j_rmt
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

                    self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_gridj, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_gridj, cnt] = j_rmt
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

                    self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_gridj, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_gridj, cnt] = j_rmt
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

                    self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_gridj, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_gridj, cnt] = j_rmt
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

                    self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_gridj, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_gridj, cnt] = j_rmt
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

                    self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_gridj, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_gridj, cnt] = j_rmt
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

                    self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_gridj, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_gridj, cnt] = j_rmt
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

                self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
                self.rellist[self.I_recv_gridj, cnt] = j
                self.rellist[self.I_recv_rgn, cnt] = rgnid
                self.rellist[self.I_recv_prc, cnt] = prc
                self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                self.rellist[self.I_send_gridj, cnt] = j_rmt
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

                    self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_gridj, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_gridj, cnt] = j_rmt
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

                self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
                self.rellist[self.I_recv_gridj, cnt] = j
                self.rellist[self.I_recv_rgn, cnt] = rgnid
                self.rellist[self.I_recv_prc, cnt] = prc
                self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                self.rellist[self.I_send_gridj, cnt] = j_rmt
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

                    self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_gridj, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_gridj, cnt] = j_rmt
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

                    self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
                    self.rellist[self.I_recv_gridj, cnt] = j
                    self.rellist[self.I_recv_rgn, cnt] = rgnid
                    self.rellist[self.I_recv_prc, cnt] = prc
                    self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
                    self.rellist[self.I_send_gridj, cnt] = j_rmt
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

            self.rellist[self.I_recv_gridi, cnt] = i  # self.suf(i, j, adm)
            self.rellist[self.I_recv_gridj, cnt] = j
            self.rellist[self.I_recv_rgn, cnt] = rgnid
            self.rellist[self.I_recv_prc, cnt] = prc
            self.rellist[self.I_send_gridi, cnt] = i_rmt  # self.suf(i_rmt, j_rmt, adm)
            self.rellist[self.I_send_gridj, cnt] = j_rmt
            self.rellist[self.I_send_rgn, cnt] = rgnid_rmt
            self.rellist[self.I_send_prc, cnt] = prc_rmt


        self.rellist_nmax = cnt + 1  # Adjust for zero-based indexing in Python

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
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


    #Sort data destination for region <-> region
    def COMM_sortdest(self):
        print("COMM_sortdest")

        # Allocate and initialize arrays
        self.Copy_info_r2r = np.full((self.info_vindex,), -1, dtype=int)
        self.Recv_info_r2r = np.full((self.info_vindex, self.Recv_nlim), -1, dtype=int)
        self.Send_info_r2r = np.full((self.info_vindex, self.Send_nlim), -1, dtype=int)

        # Set specific indices to 0
        self.Copy_info_r2r[self.I_size] = 0
        self.Recv_info_r2r[self.I_size, :] = 0
        self.Send_info_r2r[self.I_size, :] = 0

        # Allocate and initialize list arrays
        self.Copy_list_r2r = np.full((self.list_vindex, self.rellist_nmax), -1, dtype=int)
        self.Recv_list_r2r = np.full((self.list_vindex, self.rellist_nmax, self.Recv_nlim), -1, dtype=int)
        self.Send_list_r2r = np.full((self.list_vindex, self.rellist_nmax, self.Send_nlim), -1, dtype=int)

        # Sorting list according to destination
        for cnt in range(self.rellist_nmax):  

            if self.rellist[self.I_recv_prc, cnt] == self.rellist[self.I_send_prc, cnt]:  # No communication
                ipos = self.Copy_info_r2r[self.I_size] # Adjust for zero-based indexing in Python
                self.Copy_info_r2r[self.I_size] += 1

                self.Copy_list_r2r[self.I_gridi_from, ipos] = self.rellist[self.I_send_gridi, cnt]
                self.Copy_list_r2r[self.I_gridj_from, ipos] = self.rellist[self.I_send_gridj, cnt]
                self.Copy_list_r2r[self.I_l_from, ipos] = adm.RGNMNG_r2lp[adm.I_l, self.rellist[self.I_send_rgn, cnt]]
                self.Copy_list_r2r[self.I_gridi_to, ipos] = self.rellist[self.I_recv_gridi, cnt]
                self.Copy_list_r2r[self.I_gridj_to, ipos] = self.rellist[self.I_recv_gridj, cnt]
                self.Copy_list_r2r[self.I_l_to, ipos] = adm.RGNMNG_r2lp[adm.I_l, self.rellist[self.I_recv_rgn, cnt]]

            else:  # Node-to-node communication
                # Search existing rank ID (identify by prc_from)
                irank = -1

                for n in range(self.Recv_nmax_r2r):
                    if self.Recv_info_r2r[self.I_prc_from, n] == self.rellist[self.I_send_prc, cnt]:
                        irank = n
                        break  # Equivalent to Fortran's 'exit'

                if irank < 0:  # Register new rank ID
                    irank = self.Recv_nmax_r2r  # Adjust for zero-based indexing in Python
                    self.Recv_nmax_r2r += 1             
                    print(f"Rank {prc.comm_world.Get_rank()}: Recv_nmax_r2r = {self.Recv_nmax_r2r}")      
                    self.Recv_info_r2r[self.I_prc_from, irank] = self.rellist[self.I_send_prc, cnt]
                    self.Recv_info_r2r[self.I_prc_to, irank] = self.rellist[self.I_recv_prc, cnt]

                ipos = self.Recv_info_r2r[self.I_size, irank]  # Adjust for zero-based indexing in Python
                self.Recv_info_r2r[self.I_size, irank] += 1

                self.Recv_list_r2r[self.I_gridi_from, ipos, irank] = self.rellist[self.I_send_gridi, cnt]
                self.Recv_list_r2r[self.I_gridj_from, ipos, irank] = self.rellist[self.I_send_gridj, cnt]
                self.Recv_list_r2r[self.I_l_from, ipos, irank] = adm.RGNMNG_r2lp[adm.I_l, self.rellist[self.I_send_rgn, cnt]]
                self.Recv_list_r2r[self.I_gridi_to, ipos, irank] = self.rellist[self.I_recv_gridi, cnt]
                self.Recv_list_r2r[self.I_gridj_to, ipos, irank] = self.rellist[self.I_recv_gridj, cnt]
                self.Recv_list_r2r[self.I_l_to, ipos, irank] = adm.RGNMNG_r2lp[adm.I_l, self.rellist[self.I_recv_rgn, cnt]]


        if self.Copy_info_r2r[self.I_size] > 0:
            self.Copy_nmax_r2r = 1
            self.Copy_info_r2r[self.I_prc_from] = adm.ADM_prc_me
            self.Copy_info_r2r[self.I_prc_to] = adm.ADM_prc_me

        # Get maximum number of ranks for communication
        #self.sendbuf1[0] = self.Recv_nmax_r2r  # why is it defined as an array with only one component?
        #self.sendbuf1 = self.Recv_nmax_r2r  # Adjust for zero-based indexing in Python
        #call MPI_Allreduce(sendbuf1, recvbuf1, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD, ierr)

        sendbuf1 = np.array([self.Recv_nmax_r2r], dtype=np.int32)  # Equivalent to sendbuf1(1)
        recvbuf1 = np.zeros(1, dtype=np.int32) 

        # Perform MPI_Allreduce to get the maximum value across all ranks
        prc.comm_world.Allreduce(sendbuf1, recvbuf1, op=MPI.MAX)

        # Store the result
        Recv_nglobal_r2r = recvbuf1[0]

        # Debugging print (optional)
        print(f"Rank {prc.comm_world.Get_rank()}: Recv_nglobal_r2r = {Recv_nglobal_r2r}")
        print("sendbuf1 & recvbuf1 = ", sendbuf1, recvbuf1) 


        # Allocate buffers
        sendbuf_info = np.full(self.info_vindex * Recv_nglobal_r2r, -1, dtype=np.int32)
        recvbuf_info = np.empty(self.info_vindex * Recv_nglobal_r2r * prc.prc_nprocs, dtype=np.int32)

        # Distribute receive request from each rank
        for irank in range(self.Recv_nmax_r2r):  # Adjust for zero-based indexing
            n = irank * self.info_vindex

            sendbuf_info[n + self.I_size] = self.Recv_info_r2r[self.I_size, irank]
            sendbuf_info[n + self.I_prc_from] = self.Recv_info_r2r[self.I_prc_from, irank]
            sendbuf_info[n + self.I_prc_to] = self.Recv_info_r2r[self.I_prc_to, irank]

        # Calculate total size
        totalsize = self.info_vindex * Recv_nglobal_r2r

        # Perform MPI_Allgather if totalsize > 0
        if totalsize > 0:
            prc.comm_world.Allgather(sendbuf_info, recvbuf_info)

        # Final assignment
        Send_size_nglobal = 0

        # Accept receive request to my rank
        for p in range(Recv_nglobal_r2r * prc.prc_nprocs):  # Adjust for zero-based indexing
            n = p * self.info_vindex

            if recvbuf_info[n + self.I_prc_from] == adm.ADM_prc_me:
#                self.Send_nmax_r2r += 1
                irank = self.Send_nmax_r2r
                self.Send_nmax_r2r += 1
                
                self.Send_info_r2r[self.I_size, irank] = recvbuf_info[n + self.I_size]
                self.Send_info_r2r[self.I_prc_from, irank] = recvbuf_info[n + self.I_prc_from]
                self.Send_info_r2r[self.I_prc_to, irank] = recvbuf_info[n + self.I_prc_to]

            Send_size_nglobal = max(Send_size_nglobal, recvbuf_info[n + self.I_size])

        # Print logging information if std.IO_L is enabled
        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print(file=log_file)
                print(f"*** Recv_nmax_r2r(global) = {Recv_nglobal_r2r}", file=log_file)
                print(f"*** Recv_nmax_r2r(local)  = {self.Recv_nmax_r2r}", file=log_file)
                print(f"*** Send_nmax_r2r(local)  = {self.Send_nmax_r2r}", file=log_file)
                print(f"*** Send_size_r2r(global) = {Send_size_nglobal}", file=log_file)
                print(file=log_file)

                print("|---------------------------------------", file=log_file)
                print("|               size  prc_from    prc_to", file=log_file)
                print(f"| Copy_r2r {''.join(f'{val:10}' for val in self.Copy_info_r2r)}", file=log_file)

                for irank in range(self.Recv_nmax_r2r):  
                    print(f"| Recv_r2r {''.join(f'{val:10}' for val in self.Recv_info_r2r[:, irank])}", file=log_file)

                for irank in range(self.Send_nmax_r2r):  
                    print(f"| Send_r2r {''.join(f'{val:10}' for val in self.Send_info_r2r[:, irank])}", file=log_file)


        # Allocate request list
        REQ_list_r2r = np.empty(self.Recv_nmax_r2r + self.Send_nmax_r2r, dtype=MPI.Request)

        # Allocate send and receive buffers
        sendbuf_list = np.full((self.list_vindex, Send_size_nglobal, self.Recv_nmax_r2r), -1, dtype=np.int32)
        recvbuf_list = np.empty((self.list_vindex, Send_size_nglobal, self.Send_nmax_r2r), dtype=np.int32)

        # Initialize request count
        REQ_count = 0

        # Non-blocking receive requests
        for irank in range(self.Send_nmax_r2r):  # Adjust for zero-based indexing
            REQ_count += 1
            totalsize = self.Send_info_r2r[self.I_size, irank] * self.list_vindex
            rank = self.Send_info_r2r[self.I_prc_to, irank] 
            tag = self.Send_info_r2r[self.I_prc_from, irank] 

            # MPI Irecv (Non-blocking receive)
            #REQ_list_r2r[REQ_count - 1] = prc.comm_world.Irecv(recvbuf_list[:, :, irank], source=rank, tag=tag)
            REQ_list_r2r[REQ_count - 1] = prc.comm_world.Irecv(
                np.ascontiguousarray(recvbuf_list[:, :, irank]), source=rank, tag=tag)

        # Copy data and initiate non-blocking sends
        for irank in range(self.Recv_nmax_r2r):  # Adjust for zero-based indexing
            for ipos in range(self.Recv_info_r2r[self.I_size, irank]):
                sendbuf_list[:, ipos, irank] = self.Recv_list_r2r[:, ipos, irank]

            REQ_count += 1
            totalsize = self.Recv_info_r2r[self.I_size, irank] * self.list_vindex
            rank = self.Recv_info_r2r[self.I_prc_from, irank]
            tag = self.Recv_info_r2r[self.I_prc_from, irank] 

            # MPI Isend (Non-blocking send)
            REQ_list_r2r[REQ_count - 1] = prc.comm_world.Isend(
                np.ascontiguousarray(sendbuf_list[:, :, irank]), dest=rank, tag=tag)

        # Wait for all MPI requests to complete
        if self.Recv_nmax_r2r + self.Send_nmax_r2r > 0:
            MPI.Request.Waitall(REQ_list_r2r[:self.Recv_nmax_r2r + self.Send_nmax_r2r])

        # Store received data
        for irank in range(self.Send_nmax_r2r):  
            for ipos in range(self.Send_info_r2r[self.I_size, irank]):
                self.Send_list_r2r[:, ipos, irank] = recvbuf_list[:, ipos, irank]

        #debug section

        # Allocate buffers
#        self.sendbuf_r2r_SP = np.empty((Send_size_nglobal * adm.ADM_kall * self.COMM_varmax, self.Send_nmax_r2r), dtype=np.float32)
#        self.recvbuf_r2r_SP = np.empty((Send_size_nglobal * adm.ADM_kall * self.COMM_varmax, self.Recv_nmax_r2r), dtype=np.float32)
#        self.sendbuf_r2r_DP = np.empty((Send_size_nglobal * adm.ADM_kall * self.COMM_varmax, self.Send_nmax_r2r), dtype=np.float64)
#        self.recvbuf_r2r_DP = np.empty((Send_size_nglobal * adm.ADM_kall * self.COMM_varmax, self.Recv_nmax_r2r), dtype=np.float64)

        return

    def COMM_sortdest_pl(self):
        pass

    def COMM_sortdest_singular(self):
        pass

    #def suf(self, i, j, adm):
    #    return adm.ADM_gall_1d * j + i 
