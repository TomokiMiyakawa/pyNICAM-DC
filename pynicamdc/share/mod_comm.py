import toml
import numpy as np
from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf

class Comm:

    _instance = None

    COMM_pl = True

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

    def __init__(self, rdtype):

        self.rdtype = rdtype    
        self.Copy_nmax_r2r = 0
        self.Recv_nmax_r2r = 0
        self.Send_nmax_r2r = 0
        self.Copy_nmax_p2r = 0
        self.Recv_nmax_p2r = 0
        self.Send_nmax_p2r = 0
        self.Copy_nmax_r2p = 0
        self.Recv_nmax_r2p = 0
        self.Send_nmax_r2p = 0
        self.Singular_nmax = 0


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
                prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            self.COMM_apply_barrier = cnfs['commparam']['COMM_apply_barrier']  
            self.COMM_varmax = cnfs['commparam']['COMM_varmax']  
            #debug = cnfs['commparam']['debug']  
            #testonly = cnfs['commparam']['testonly']  

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
        self.COMM_sortdest_pl()
        self.COMM_sortdest_singular()

        listsize = (self.Recv_nmax_r2r + self.Send_nmax_r2r +
                      self.Recv_nmax_p2r + self.Send_nmax_p2r +
                      self.Recv_nmax_r2p + self.Send_nmax_r2p)
        
        self.REQ_list = np.empty((listsize,), dtype=int)

        #call debugtest for testonly here

        return

    def COMM_list_generate(self):
        #print("COMM_list_generate")

        ginner = adm.ADM_gmax - adm.ADM_gmin + 1

        # Allocate rellist (Fortran allocate -> NumPy array initialization)
        self.rellist = np.empty((self.rellist_vindex, adm.ADM_gall * adm.ADM_lall,), dtype=int)

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

        debug = False
        if debug:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('--- Relation Table', file=log_file)
                    print(f"{'Count':>10} {'|recv_gridi':>10} {'|recv_gridj':>10} {'|recv_rgn':>10} {'|recv_prc':>10} "
                          f"{'|send_gridi':>10} {'|send_gridj':>10} {'|send_rgn':>10} {'|send_prc':>10}", file=log_file)

                    for cnt in range(self.rellist_nmax):  # Adjust for zero-based indexing in Python
                        print(f"{cnt:10} {' '.join(f'{val:10}' for val in self.rellist[:, cnt])}", file=log_file)

        return


    #Sort data destination for region <-> region
    def COMM_sortdest(self):
        #print("COMM_sortdest")

        # Allocate and initialize arrays
        self.Copy_info_r2r = np.full((self.info_vindex,), -1, dtype=int)
        self.Recv_info_r2r = np.full((self.info_vindex, self.Recv_nlim,), -1, dtype=int)
        self.Send_info_r2r = np.full((self.info_vindex, self.Send_nlim,), -1, dtype=int)

        # Set specific indices to 0
        self.Copy_info_r2r[self.I_size] = 0
        self.Recv_info_r2r[self.I_size, :] = 0
        self.Send_info_r2r[self.I_size, :] = 0

        # Allocate and initialize list arrays
        self.Copy_list_r2r = np.full((self.list_vindex, self.rellist_nmax,), -1, dtype=int)
        self.Recv_list_r2r = np.full((self.list_vindex, self.rellist_nmax, self.Recv_nlim,), -1, dtype=int)
        self.Send_list_r2r = np.full((self.list_vindex, self.rellist_nmax, self.Send_nlim,), -1, dtype=int)

        #print(f"self.Send_list_r2r.shape: {self.Send_list_r2r.shape}")
            
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
                    self.Recv_info_r2r[self.I_prc_from, irank] = self.rellist[self.I_send_prc, cnt]
                    self.Recv_info_r2r[self.I_prc_to, irank] = self.rellist[self.I_recv_prc, cnt]

                ipos = self.Recv_info_r2r[self.I_size, irank]  # Adjust for zero-based indexing in Python
                self.Recv_info_r2r[self.I_size, irank] += 1

                self.Recv_list_r2r[self.I_gridi_from, ipos, irank] = self.rellist[self.I_send_gridi, cnt]   
                self.Recv_list_r2r[self.I_gridj_from, ipos, irank] = self.rellist[self.I_send_gridj, cnt]
                self.Recv_list_r2r[self.I_l_from, ipos, irank] = adm.RGNMNG_r2lp[adm.I_l, self.rellist[self.I_send_rgn, cnt]] #####Checkp Looks good

                self.Recv_list_r2r[self.I_gridi_to, ipos, irank] = self.rellist[self.I_recv_gridi, cnt]
                self.Recv_list_r2r[self.I_gridj_to, ipos, irank] = self.rellist[self.I_recv_gridj, cnt]
                self.Recv_list_r2r[self.I_l_to, ipos, irank] = adm.RGNMNG_r2lp[adm.I_l, self.rellist[self.I_recv_rgn, cnt]]


        if self.Copy_info_r2r[self.I_size] > 0:
            self.Copy_nmax_r2r = 1
            self.Copy_info_r2r[self.I_prc_from] = adm.ADM_prc_me
            self.Copy_info_r2r[self.I_prc_to] = adm.ADM_prc_me

        sendbuf1 = np.array([self.Recv_nmax_r2r], dtype=np.int32)  # Equivalent to sendbuf1(1)
        recvbuf1 = np.zeros(1, dtype=np.int32) 

        # Perform MPI_Allreduce to get the maximum value across all ranks
        prc.comm_world.Allreduce(sendbuf1, recvbuf1, op=MPI.MAX)

        # Store the result
        Recv_nglobal_r2r = recvbuf1[0]

        # Allocate buffers
        sendbuf_info = np.full((self.info_vindex * Recv_nglobal_r2r,), -1, dtype=np.int32)
        recvbuf_info = np.empty((self.info_vindex * Recv_nglobal_r2r * prc.prc_nprocs,), dtype=np.int32)

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

        # Accept receive request to my rank
        #print("Recv_nglobal_r2r * prc.prc_nprocs = ", Recv_nglobal_r2r * prc.prc_nprocs)
        #print(Recv_nglobal_r2r, prc.prc_nprocs)
        self.Send_size_nglobal = 0
        for p in range(Recv_nglobal_r2r * prc.prc_nprocs):  # Adjust for zero-based indexing
            n = p * self.info_vindex

            if recvbuf_info[n + self.I_prc_from] == adm.ADM_prc_me:
                irank = self.Send_nmax_r2r
                self.Send_nmax_r2r += 1
                
                self.Send_info_r2r[self.I_size, irank] = recvbuf_info[n + self.I_size]
                self.Send_info_r2r[self.I_prc_from, irank] = recvbuf_info[n + self.I_prc_from]
                self.Send_info_r2r[self.I_prc_to, irank] = recvbuf_info[n + self.I_prc_to]

            self.Send_size_nglobal = max(self.Send_size_nglobal, recvbuf_info[n + self.I_size])

        # Print logging information if std.IO_L is enabled
        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print(file=log_file)
                print(f"*** Recv_nmax_r2r(global) = {Recv_nglobal_r2r}", file=log_file)
                print(f"*** Recv_nmax_r2r(local)  = {self.Recv_nmax_r2r}", file=log_file)
                print(f"*** Send_nmax_r2r(local)  = {self.Send_nmax_r2r}", file=log_file)
                print(f"*** Send_size_r2r(global) = {self.Send_size_nglobal}", file=log_file)
                print(file=log_file)

                print("|---------------------------------------", file=log_file)
                print("|               size  prc_from    prc_to", file=log_file)
                print(f"| Copy_r2r {''.join(f'{val:10}' for val in self.Copy_info_r2r)}", file=log_file)

                for irank in range(self.Recv_nmax_r2r):  
                    print(f"| Recv_r2r {''.join(f'{val:10}' for val in self.Recv_info_r2r[:, irank])}", file=log_file)

                for irank in range(self.Send_nmax_r2r):  
                    print(f"| Send_r2r {''.join(f'{val:10}' for val in self.Send_info_r2r[:, irank])}", file=log_file)


        # Allocate request list
        REQ_list_r2r = np.empty((self.Recv_nmax_r2r + self.Send_nmax_r2r,), dtype=object)
        REQ_list_r2r.fill(MPI.REQUEST_NULL)  # Initialize with NULL requests

        # Allocate send and receive buffers
        sendbuf_list = np.full((self.list_vindex, self.Send_size_nglobal, self.Recv_nmax_r2r,), -1, dtype=np.int32)
        recvbuf_list = np.empty((self.list_vindex, self.Send_size_nglobal, self.Send_nmax_r2r,), dtype=np.int32)

        # Initialize request count
        REQ_count = 0

        # Non-blocking receive requests
        recv_slices = [] 
        REQ_list_r2r = []
        
        for irank in range(self.Send_nmax_r2r):  # Adjust for zero-based indexing
            totalsize = self.Send_info_r2r[self.I_size, irank] * self.list_vindex
            size1 = self.list_vindex
            size2 = self.Send_info_r2r[self.I_size, irank]
            rank = self.Send_info_r2r[self.I_prc_to, irank] 
            tag = self.Send_info_r2r[self.I_prc_from, irank] 

            #print("recieving...")
            #print("size1= ", size1, "size2= ", size2, "source rank=", rank, "tag= ", tag, "   irank= ", irank)   

            recvslice = np.empty((size1,size2,),dtype=np.int32)
            recvslice = np.ascontiguousarray(recvslice)
            recv_slices.append(recvslice) 

            REQ_list_r2r.append(prc.comm_world.Irecv(recv_slices[irank], source=rank, tag=tag))

            REQ_count += 1

        # Copy data and initiate non-blocking sends
        for irank in range(self.Recv_nmax_r2r):  # Adjust for zero-based indexing
            for ipos in range(self.Recv_info_r2r[self.I_size, irank]):
                sendbuf_list[:, ipos, irank] = self.Recv_list_r2r[:, ipos, irank]    ##### Check this line

            totalsize = self.Recv_info_r2r[self.I_size, irank] * self.list_vindex
            size1 = self.list_vindex
            size2 = self.Recv_info_r2r[self.I_size, irank] 
            rank = self.Recv_info_r2r[self.I_prc_from, irank]
            tag  = rank #self.Recv_info_r2r[self.I_prc_from, irank] 

            #print("sending...")
            #print("size1= ", size1, "size2= ", size2, "dest rank=", rank, "tag= ", tag, "   irank= ", irank)  
            sendslice = np.empty((size1,size2,),dtype=np.int32)
            sendslice = np.ascontiguousarray(sendbuf_list[0:size1, 0:size2, irank])

            REQ_list_r2r.append(prc.comm_world.Isend(sendslice, dest=rank, tag=tag))

            REQ_count += 1

        # Wait for all MPI requests to complete
        if self.Recv_nmax_r2r + self.Send_nmax_r2r > 0:
            MPI.Request.Waitall(REQ_list_r2r)
            
        # Store received data
        for irank in range(self.Send_nmax_r2r):  
            size1 = self.list_vindex            
            size2 = self.Send_info_r2r[self.I_size, irank]
            recvbuf_list[0:size1,0:size2,irank]=recv_slices[irank] #(recvslice)
            
            for ipos in range(self.Send_info_r2r[self.I_size, irank]):
                self.Send_list_r2r[:, ipos, irank] = recvbuf_list[:, ipos, irank]
            
        #debug section
        debug = False
        if debug:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(file=log_file)
                    print("--- Copy_list_r2r", file=log_file)
                    print(file=log_file)
                    print(f"{'number':>6} {'|ifrom':>6} {'|jfrom':>6} {'|rfrom':>6} {'|lfrom':>6} {'|pfrom':>6} "
                          f"{'|  ito':>6} {'|  jto':>6} {'|  rto':>6} q{'|  lto':>6} {'|  pto':>6}", file=log_file)

                    for ipos in range(self.Copy_info_r2r[self.I_size]):
                        gi_from = self.Copy_list_r2r[self.I_gridi_from, ipos]
                        gj_from = self.Copy_list_r2r[self.I_gridj_from, ipos]
                        l_from = self.Copy_list_r2r[self.I_l_from, ipos]
                        p_from = self.Copy_info_r2r[self.I_prc_from]
                        r_from = adm.RGNMNG_lp2r[l_from, p_from]

                        gi_to = self.Copy_list_r2r[self.I_gridi_to, ipos]
                        gj_to = self.Copy_list_r2r[self.I_gridj_to, ipos]
                        l_to = self.Copy_list_r2r[self.I_l_to, ipos]
                        p_to = self.Copy_info_r2r[self.I_prc_to]
                        r_to = adm.RGNMNG_lp2r[l_to, p_to]

                        print(f"{ipos:6} {gi_from:6} {gj_from:6} {r_from:6} {l_from:6} {p_from:6} "
                              f"{gi_to:6} {gj_to:6} {r_to:6} {l_to:6} {p_to:6}", file=log_file)

                    print(file=log_file)
                    print("--- Recv_list_r2r", file=log_file)

                    for irank in range(self.Recv_nmax_r2r):
                        print(f"{'number':>6} {'|ifrom':>6} {'|jfrom':>6} {'|rfrom':>6} {'|lfrom':>6} {'|pfrom':>6} "
                              f"{'|  ito':>6} {'|  jto':>6} {'|  rto':>6} {'|  lto':>6} {'|  pto':>6}", file=log_file)

                        for ipos in range(self.Recv_info_r2r[self.I_size, irank]):
                            gi_from = self.Recv_list_r2r[self.I_gridi_from, ipos, irank]
                            gj_from = self.Recv_list_r2r[self.I_gridj_from, ipos, irank]
                            l_from = self.Recv_list_r2r[self.I_l_from, ipos, irank]
                            p_from = self.Recv_info_r2r[self.I_prc_from, irank]
                            r_from = adm.RGNMNG_lp2r[l_from, p_from]

                            gi_to = self.Recv_list_r2r[self.I_gridi_to, ipos, irank]
                            gj_to = self.Recv_list_r2r[self.I_gridj_to, ipos, irank]
                            l_to = self.Recv_list_r2r[self.I_l_to, ipos, irank]
                            p_to = self.Recv_info_r2r[self.I_prc_to, irank]
                            r_to = adm.RGNMNG_lp2r[l_to, p_to]

                            print(f"{ipos:6} {gi_from:6} {gj_from:6} {r_from:6} {l_from:6} {p_from:6} "
                                  f"{gi_to:6} {gj_to:6} {r_to:6} {l_to:6} {p_to:6}", file=log_file)

                    print(file=log_file)
                    print("--- Send_list_r2r", file=log_file)

                    for irank in range(self.Send_nmax_r2r):
                        print(f"{'number':>6} {'|ifrom':>6} {'|jfrom':>6} {'|rfrom':>6} {'|lfrom':>6} {'|pfrom':>6} "
                              f"{'|  ito':>6} {'|  jto':>6} {'|  rto':>6} {'|  lto':>6} {'|  pto':>6}", file=log_file)
                        #print("prc.prc_myrank= ", prc.prc_myrank)
                        #print("irank= ", irank)
                        #print("self.Send_info_r2r[self.I_size, irank]", self.Send_info_r2r[self.I_size, irank])
                        for ipos in range(self.Send_info_r2r[self.I_size, irank]):
                            gi_from = self.Send_list_r2r[self.I_gridi_from, ipos, irank]
                            gj_from = self.Send_list_r2r[self.I_gridj_from, ipos, irank]
                            l_from = self.Send_list_r2r[self.I_l_from, ipos, irank]
                            p_from = self.Send_info_r2r[self.I_prc_from, irank]
                            r_from = adm.RGNMNG_lp2r[l_from, p_from]
                            gi_to = self.Send_list_r2r[self.I_gridi_to, ipos, irank]
                            gj_to = self.Send_list_r2r[self.I_gridj_to, ipos, irank]
                            l_to = self.Send_list_r2r[self.I_l_to, ipos, irank]
                            p_to = self.Send_info_r2r[self.I_prc_to, irank]
                            r_to = adm.RGNMNG_lp2r[l_to, p_to]

                            print(f"{ipos:6} {gi_from:6} {gj_from:6} {r_from:6} {l_from:6} {p_from:6} "
                                  f"{gi_to:6} {gj_to:6} {r_to:6} {l_to:6} {p_to:6}", file=log_file)
        # end of debug section

        # Allocate buffers here to prevent having to reallocate them multiple times in the data_transfer function
        #print("Buffer size info", self.Send_size_nglobal, adm.ADM_kall, self.COMM_varmax, self.Send_nmax_r2r, self.Recv_nmax_r2r) # e.g., 68 1 15 5 5 depending on the rank
        #self.sendbuf_r2r = np.empty((self.Send_size_nglobal * adm.ADM_kall * self.COMM_varmax, self.Send_nmax_r2r), dtype=self.rdtype)
        #!!!self.sendbuf_r2r = np.empty((self.Send_size_nglobal * adm.ADM_kall * self.COMM_varmax,), dtype=self.rdtype)
        #!!!self.recvbuf_r2r = np.empty((self.Send_size_nglobal * adm.ADM_kall * self.COMM_varmax, self.Recv_nmax_r2r), dtype=self.rdtype) 
        #self.recvbuf_r2r = np.empty((self.Send_size_nglobal * adm.ADM_kall * self.COMM_varmax,), dtype=self.rdtype) 
        #!!!self.sendbuf_r2r = np.ascontiguousarray(self.sendbuf_r2r)
        #!!!self.recvbuf_r2r = np.ascontiguousarray(self.recvbuf_r2r)
        ### !!! The contiguous allocation here may cause unneccesary use of memory, slow data transfer, or overhead.
        ### Check and tune later along with the data_transfer procedures inside the loops.
        
        #self.sendbuf_r2r_SP = np.empty((Send_size_nglobal * adm.ADM_kall * self.COMM_varmax, self.Send_nmax_r2r), dtype=np.float32)
        #self.recvbuf_r2r_SP = np.empty((Send_size_nglobal * adm.ADM_kall * self.COMM_varmax, self.Recv_nmax_r2r), dtype=np.float32)
        #self.sendbuf_r2r_DP = np.empty((Send_size_nglobal * adm.ADM_kall * self.COMM_varmax, self.Send_nmax_r2r), dtype=np.float64)
        #self.recvbuf_r2r_DP = np.empty((Send_size_nglobal * adm.ADM_kall * self.COMM_varmax, self.Recv_nmax_r2r), dtype=np.float64)
        #print("HEYHEY!",  np.shape(self.sendbuf_r2r))  # 68*1*15 = 1020  small because size of k is 1 for grid production
        return

    def COMM_sortdest_pl(self):
        #print("COMM_sortdest_pl")

        self.Send_size_nglobal_pl = 10

        #pl_to = -99999

        # Allocate and initialize arrays
        self.Copy_info_p2r = np.full((self.info_vindex,), -1, dtype=int)
        self.Recv_info_p2r = np.full((self.info_vindex, self.Recv_nlim,), -1, dtype=int)
        self.Send_info_p2r = np.full((self.info_vindex, self.Send_nlim,), -1, dtype=int)

        # Set specific indices to 0
        self.Copy_info_p2r[self.I_size] = 0
        self.Recv_info_p2r[self.I_size, :] = 0
        self.Send_info_p2r[self.I_size, :] = 0

        # Allocate and initialize list arrays
        self.Copy_list_p2r = np.full((self.list_vindex, self.Send_size_nglobal_pl,), -1, dtype=int)
        self.Recv_list_p2r = np.full((self.list_vindex, self.Send_size_nglobal_pl, self.Recv_nlim,), -1, dtype=int)
        self.Send_list_p2r = np.full((self.list_vindex, self.Send_size_nglobal_pl, self.Send_nlim,), -1, dtype=int)

        # Allocate and initialize arrays
        self.Copy_info_r2p = np.full((self.info_vindex,), -1, dtype=int)
        self.Recv_info_r2p = np.full((self.info_vindex, self.Recv_nlim,), -1, dtype=int)
        self.Send_info_r2p = np.full((self.info_vindex, self.Send_nlim,), -1, dtype=int)

        # Set specific indices to 0
        self.Copy_info_r2p[self.I_size] = 0
        self.Recv_info_r2p[self.I_size, :] = 0
        self.Send_info_r2p[self.I_size, :] = 0

        # Allocate and initialize list arrays
        self.Copy_list_r2p = np.full((self.list_vindex, self.Send_size_nglobal_pl,), -1, dtype=int)
        self.Recv_list_r2p = np.full((self.list_vindex, self.Send_size_nglobal_pl, self.Recv_nlim,), -1, dtype=int)
        self.Send_list_r2p = np.full((self.list_vindex, self.Send_size_nglobal_pl, self.Send_nlim,), -1, dtype=int)

        #Search in regular region
        #print("adm.ADM_lall= ", adm.ADM_lall)
        for l in range(adm.ADM_lall):
            rgnid = adm.RGNMNG_l2r[l]
            prc = adm.ADM_prc_me
            #print("prc, l, rgnid = ", prc, l, rgnid)

                
            for l_pl in range(adm.I_NPL, adm.I_SPL + 1):
                rgnid_rmt = l_pl      # This is 0 or 1
                prc_rmt = adm.RGNMNG_r2p_pl[rgnid_rmt]    #This is always zero for normal ICO, rank 0 handles North/South poles
                #print("rgnid_rmt, prc_rmt= ", rgnid_rmt, prc_rmt)
                    
                if rgnid_rmt == adm.I_NPL:     # 0
                    check_vert_num = adm.RGNMNG_vert_num[adm.I_N, rgnid]
                    i_from = adm.ADM_gmin
                    j_from = adm.ADM_gmax
                    i_to = adm.ADM_gmin
                    j_to = adm.ADM_gmax + 1
                elif rgnid_rmt == adm.I_SPL:   # 1
                    check_vert_num = adm.RGNMNG_vert_num[adm.I_S, rgnid]
                    i_from = adm.ADM_gmax
                    j_from = adm.ADM_gmin
                    i_to = adm.ADM_gmax + 1
                    j_to = adm.ADM_gmin
                
                #print("check_vert_num, i_from, j_from, i_to, j_to")
                #print(check_vert_num, i_from, j_from, i_to, j_to)

                #print("adm.ADM_vlink= ", adm.ADM_vlink)

                if check_vert_num == adm.ADM_vlink: #search destination in the pole halo
                    #print("check_vert_num", check_vert_num)   # This should be 5 (north or south pole)
                    for v in range(adm.ADM_vlink):    # 0 to 4

                        #print("")
                        #print("rgnid_rmt, v")
                        #print(rgnid_rmt, v)

                        #print(v, rgnid, adm.RGNMNG_vert_tab_pl[adm.I_RGNID, rgnid_rmt, v])
                        #print("")
                        #print("rgnid, adm.RGNMNG_vert_tab_pl[adm.I_RGNID, rgnid_rmt, v]")
                        #print(rgnid, adm.RGNMNG_vert_tab_pl[adm.I_RGNID, rgnid_rmt, v])

                        if rgnid == adm.RGNMNG_vert_tab_pl[adm.I_RGNID, rgnid_rmt, v]:
                            pl_to = v
                            #print("a: pl_to, v= ", pl_to, v)
                            #print("")

                            #print("pl_to, adm.ADM_gmin_pl, adm.ADM_gmax_pl")
                            #print(pl_to, adm.ADM_gmin_pl, adm.ADM_gmax_pl)

                            # 0 is the pole itself, and vertex number0 is renamed as number5, (pl_to can take 6 values)                             
                            if pl_to < adm.ADM_gmin_pl:
                                pl_to = adm.ADM_gmax_pl
                                #print("")
                                #print("pl_to, adm.ADM_gmin_pl")
                                #print(pl_to, adm.ADM_gmin_pl)
                                #print("b: pl_to, v= ", pl_to, v)
                                #print("")
                            break
                        
                    if prc == prc_rmt:  # no communication   
                        # copy region inner -> pole halo
                        ipos = self.Copy_info_r2p[self.I_size]
                        self.Copy_info_r2p[self.I_size] += 1    
                            
                        self.Copy_list_r2p[self.I_gridi_from, ipos] = i_from
                        self.Copy_list_r2p[self.I_gridj_from, ipos] = j_from
                        self.Copy_list_r2p[self.I_l_from, ipos] = l
                        #print("1st pl_to, v, prc = ", pl_to, v, prc)
                        self.Copy_list_r2p[self.I_gridi_to, ipos] = pl_to
                        self.Copy_list_r2p[self.I_gridj_to, ipos] = pl_to
                        self.Copy_list_r2p[self.I_l_to, ipos] = l_pl
                    else:  #node-to-node communication
                        # receive pole center -> region halo
                        irank = -1
                        for n in range(self.Recv_nmax_p2r):
                            if self.Recv_info_p2r[self.I_prc_from, n] == prc_rmt:
                                irank = n
                                break
                            
                        if irank < 0:   # register new rank id   ###########
                            irank = self.Recv_nmax_p2r
                            self.Recv_nmax_p2r += 1
                                
                            self.Recv_info_p2r[self.I_prc_from, irank] = prc_rmt
                            self.Recv_info_p2r[self.I_prc_to, irank] = prc
                            
                        ipos = self.Recv_info_p2r[self.I_size, irank]
                        self.Recv_info_p2r[self.I_size, irank] += 1

                        self.Recv_list_p2r[self.I_gridi_from, ipos, irank] = adm.ADM_gslf_pl
                        self.Recv_list_p2r[self.I_gridj_from, ipos, irank] = adm.ADM_gslf_pl
                        self.Recv_list_p2r[self.I_l_from, ipos, irank] = l_pl
                        self.Recv_list_p2r[self.I_gridi_to, ipos, irank] = i_to #self.adm.suf(i_to, j_to)
                        self.Recv_list_p2r[self.I_gridj_to, ipos, irank] = j_to #self.adm.suf(i_to, j_to)
                        self.Recv_list_p2r[self.I_l_to, ipos, irank] = l
                            
                        # send region inner -> pole halo
                        irank = -1
                        for n in range(self.Send_nmax_r2p):
                            if self.Send_info_r2p[self.I_prc_to, n] == prc_rmt:
                                irank = n
                                break
                            
                        if irank < 0:  ############
                            irank = self.Send_nmax_r2p
                            self.Send_nmax_r2p += 1

                            self.Send_info_r2p[self.I_prc_from, irank] = prc
                            self.Send_info_r2p[self.I_prc_to, irank] = prc_rmt
                            
                        ipos = self.Send_info_r2p[self.I_size, irank]
                        self.Send_info_r2p[self.I_size, irank] += 1

                        self.Send_list_r2p[self.I_gridi_from, ipos, irank] = i_from #self.adm.suf(i_from, j_from)
                        self.Send_list_r2p[self.I_gridj_from, ipos, irank] = j_from #self.adm.suf(i_from, j_from)
                        self.Send_list_r2p[self.I_l_from, ipos, irank] = l
                        #print("2nd pl_to= ", pl_to, v)
                        self.Send_list_r2p[self.I_gridi_to, ipos, irank] = pl_to
                        self.Send_list_r2p[self.I_gridj_to, ipos, irank] = pl_to
                        self.Send_list_r2p[self.I_l_to, ipos, irank] = l_pl

        #Search in pole
        if adm.ADM_have_pl:
            for l_pl in range(adm.I_NPL, adm.I_SPL + 1):
                rgnid = l_pl
                prc = adm.ADM_prc_me
                    
                #for v in range(adm.ADM_vlink + 1, 1, -1):  # 6 5 4 3 2 
                for v in range(adm.ADM_vlink, 0, -1):      # 5 4 3 2 1 
                    vv = v 
                    #if v == adm.ADM_vlink + 1:
                    #    vv = 1                             # 0 4 3 2 1
                    if v == adm.ADM_vlink:
                        vv = 0                             # 0 4 3 2 1
                        
                    rgnid_rmt = adm.RGNMNG_vert_tab_pl[adm.I_RGNID, rgnid, vv]
                    prc_rmt = adm.RGNMNG_r2lp[adm.I_prc, rgnid_rmt]
                        
                    if rgnid == adm.I_NPL:
                        i_from = adm.ADM_gmin
                        j_from = adm.ADM_gmax
                        i_to = adm.ADM_gmin 
                        j_to = adm.ADM_gmax + 1
                    elif rgnid == adm.I_SPL:
                        i_from = adm.ADM_gmax 
                        j_from = adm.ADM_gmin
                        i_to = adm.ADM_gmax + 1
                        j_to = adm.ADM_gmin
                        
                    pl_to = vv 
                    if pl_to < adm.ADM_gmin_pl: 
                        pl_to = adm.ADM_gmax_pl

                    if prc == prc_rmt:
                        ipos = self.Copy_info_p2r[self.I_size]
                        self.Copy_info_p2r[self.I_size] += 1                            

                        self.Copy_list_p2r[self.I_gridi_from, ipos] = adm.ADM_gslf_pl
                        self.Copy_list_p2r[self.I_gridj_from, ipos] = adm.ADM_gslf_pl
                        self.Copy_list_p2r[self.I_l_from, ipos] = l_pl
                        self.Copy_list_p2r[self.I_gridi_to, ipos] = i_to #adm.suf(i_to, j_to)
                        self.Copy_list_p2r[self.I_gridj_to, ipos] = j_to #adm.suf(i_to, j_to)
                        self.Copy_list_p2r[self.I_l_to, ipos] = adm.RGNMNG_r2lp[adm.I_l, rgnid_rmt]
                    else:  #########
                        #irank = next((n for n in range(1, self.Recv_nmax_r2p + 1) if self.Recv_info_r2p[self.I_prc_from, n] == prc_rmt), -1)
                        irank = next((n for n in range(self.Recv_nmax_r2p) if self.Recv_info_r2p[self.I_prc_from, n] == prc_rmt), -1)
                        if irank < 0:
                            irank = self.Recv_nmax_r2p
                            self.Recv_nmax_r2p += 1
                            self.Recv_info_r2p[self.I_prc_from, irank] = prc_rmt
                            self.Recv_info_r2p[self.I_prc_to, irank] = prc
                            
                        ipos = self.Recv_info_r2p[self.I_size, irank]
                        self.Recv_info_r2p[self.I_size, irank] += 1

                        self.Recv_list_r2p[self.I_gridi_from, ipos, irank] = i_from #self.suf(i_from, j_from)
                        self.Recv_list_r2p[self.I_gridj_from, ipos, irank] = j_from #self.suf(i_from, j_from)
                        self.Recv_list_r2p[self.I_l_from, ipos, irank] = adm.RGNMNG_r2lp[adm.I_l, rgnid_rmt]
                        self.Recv_list_r2p[self.I_gridi_to, ipos, irank] = pl_to
                        self.Recv_list_r2p[self.I_gridj_to, ipos, irank] = pl_to
                        self.Recv_list_r2p[self.I_l_to, ipos, irank] = l_pl
                            
                        #irank = next((n for n in range(1, self.Send_nmax_p2r + 1) if self.Send_info_p2r[self.I_prc_to, n] == prc_rmt), -1)
                        irank = next((n for n in range(self.Send_nmax_p2r) if self.Send_info_p2r[self.I_prc_to, n] == prc_rmt), -1)
                        if irank < 0:
                            irank = self.Send_nmax_p2r
                            self.Send_nmax_p2r += 1
                            self.Send_info_p2r[self.I_prc_from, irank] = prc
                            self.Send_info_p2r[self.I_prc_to, irank] = prc_rmt
                            
                        ipos = self.Send_info_p2r[self.I_size, irank]
                        self.Send_info_p2r[self.I_size, irank] += 1

                        self.Send_list_p2r[self.I_gridi_from, ipos, irank] = adm.ADM_gslf_pl
                        self.Send_list_p2r[self.I_gridj_from, ipos, irank] = adm.ADM_gslf_pl
                        self.Send_list_p2r[self.I_l_from, ipos, irank] = l_pl
                        self.Send_list_p2r[self.I_gridi_to, ipos, irank] = i_to
                        self.Send_list_p2r[self.I_gridj_to, ipos, irank] = j_to
                        self.Send_list_p2r[self.I_l_to, ipos, irank] = adm.RGNMNG_r2lp[adm.I_l, rgnid_rmt]
                
            if self.Copy_info_p2r[self.I_size] > 0:
                self.Copy_nmax_p2r = 1
                self.Copy_info_p2r[self.I_prc_from] = adm.ADM_prc_me
                self.Copy_info_p2r[self.I_prc_to] = adm.ADM_prc_me
                
            if self.Copy_info_r2p[self.I_size] > 0:
                self.Copy_nmax_r2p = 1
                self.Copy_info_r2p[self.I_prc_from] = adm.ADM_prc_me
                self.Copy_info_r2p[self.I_prc_to] = adm.ADM_prc_me


        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("\n*** Recv_nmax_p2r(local) =", self.Recv_nmax_p2r, file=log_file)
                print("*** Send_nmax_p2r(local) =", self.Send_nmax_p2r, file=log_file)
                print("|---------------------------------------", file=log_file)
                print("|               size  prc_from    prc_to", file=log_file)
                print("| Copy_p2r", self.Copy_info_p2r[:], file=log_file)
                
                for irank in range(self.Recv_nmax_p2r):
                    print("| Recv_p2r", self.Recv_info_p2r[:, irank], file=log_file)
                for irank in range(self.Send_nmax_p2r):
                    print("| Send_p2r", self.Send_info_p2r[:, irank], file=log_file)
        

        debug = False
        if debug:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("", file=log_file)
                    print("--- Copy_list_p2r", file=log_file)
                    print("", file=log_file)
                    print(f"{'number':>6} {'|rfrom':>6} {'|ifrom':>6} {'|jfrom':>6} {'|lfrom':>6} {'|pfrom':>6} {'|ito':>6} {'|jto':>6} {'|rto':>6} {'|lto':>6} {'|pto':>6}", file=log_file)
                    for ipos in range(self.Copy_info_p2r[self.I_size]):
                        i_from = self.Copy_list_p2r[self.I_gridi_from, ipos]
                        j_from = self.Copy_list_p2r[self.I_gridj_from, ipos]
                        l_from = self.Copy_list_p2r[self.I_l_from, ipos]
                        p_from = self.Copy_info_p2r[self.I_prc_from]
                        r_from = l_from
                        i_to = self.Copy_list_p2r[self.I_gridi_to, ipos]
                        j_to = self.Copy_list_p2r[self.I_gridj_to, ipos]
                        l_to = self.Copy_list_p2r[self.I_l_to, ipos]
                        p_to = self.Copy_info_p2r[self.I_prc_to]
                        #i_to = (g_to - 1) % self.ADM_gall_1d + 1
                        #j_to = (g_to - i_to) // self.ADM_gall_1d + 1
                        r_to = adm.RGNMNG_lp2r[l_to, p_to]
                        print(f"{ipos:6} {r_from:6} {i_from:6} {j_from:6} {l_from:6} {p_from:6} {i_to:6} {j_to:6} {r_to:6} {l_to:6} {p_to:6}", file=log_file)
                        
                    print("", file=log_file)
                    print("--- Recv_list_p2r", file=log_file)
                    print("self.Recv_nmax_p2r", self.Recv_nmax_p2r, file=log_file)
                    for irank in range(self.Recv_nmax_p2r):
                        print(f"{'number':>6} {'|rfrom':>6} {'|ifrom':>6} {'|jfrom':>6} {'|lfrom':>6} {'|pfrom':>6} {'|ito':>6} {'|jto':>6} {'|rto':>6} {'|lto':>6} {'|pto':>6}", file=log_file)
                        for ipos in range(self.Recv_info_p2r[self.I_size, irank]):
                            i_from = self.Recv_list_p2r[self.I_gridi_from, ipos, irank]
                            j_from = self.Recv_list_p2r[self.I_gridj_from, ipos, irank]
                            l_from = self.Recv_list_p2r[self.I_l_from, ipos, irank]
                            p_from = self.Recv_info_p2r[self.I_prc_from, irank]
                            r_from = l_from
                            i_to = self.Recv_list_p2r[self.I_gridi_to, ipos, irank]
                            j_to = self.Recv_list_p2r[self.I_gridj_to, ipos, irank]
                            l_to = self.Recv_list_p2r[self.I_l_to, ipos, irank]
                            p_to = self.Recv_info_p2r[self.I_prc_to, irank]
                            #i_to = (g_to - 1) % self.ADM_gall_1d + 1
                            #j_to = (g_to - i_to) // self.ADM_gall_1d + 1
                            r_to = adm.RGNMNG_lp2r[l_to, p_to]
                            print(f"{ipos:6} {r_from:6} {i_from:6} {j_from:6} {l_from:6} {p_from:6} {i_to:6} {j_to:6} {r_to:6} {l_to:6} {p_to:6}", file=log_file)

                    print("", file=log_file)
                    print("--- Send_list_p2r", file=log_file)
                    for irank in range(self.Send_nmax_p2r):
                        print(f"{'number':>6} {'|rfrom':>6} {'|ifrom':>6} {'|jfrom':>6} {'|lfrom':>6} {'|pfrom':>6} {'|ito':>6} {'|jto':>6} {'|rto':>6} {'|lto':>6} {'|pto':>6}", file=log_file)
                        for ipos in range(self.Send_info_p2r[self.I_size, irank]):
                            i_from = self.Send_list_p2r[self.I_gridi_from, ipos, irank]
                            j_from = self.Send_list_p2r[self.I_gridj_from, ipos, irank]
                            l_from = self.Send_list_p2r[self.I_l_from, ipos, irank]
                            p_from = self.Send_info_p2r[self.I_prc_from, irank]
                            r_from = l_from
                            i_to = self.Send_list_p2r[self.I_gridi_to, ipos, irank]
                            j_to = self.Send_list_p2r[self.I_gridj_to, ipos, irank]
                            l_to = self.Send_list_p2r[self.I_l_to, ipos, irank]
                            p_to = self.Send_info_p2r[self.I_prc_to, irank]
                            #i_to = (g_to - 1) % self.ADM_gall_1d + 1
                            #j_to = (g_to - i_to) // self.ADM_gall_1d + 1
                            r_to = adm.RGNMNG_lp2r[l_to, p_to]
                            print(f"{ipos:6} {r_from:6} {i_from:6} {j_from:6} {l_from:6} {p_from:6} {i_to:6} {j_to:6} {r_to:6} {l_to:6} {p_to:6}", file=log_file)


        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print(f"*** Recv_nmax_r2p(local)  = {self.Recv_nmax_r2p}", file=log_file)
                print(f"*** Send_nmax_r2p(local)  = {self.Send_nmax_r2p}", file=log_file)
                print("", file=log_file)
                print("|---------------------------------------", file=log_file)
                print("|               size  prc_from    prc_to", file=log_file) 
                print(f"| Copy_r2p {' '.join(map(str, self.Copy_info_r2p))}", file=log_file)
                        
                #print(self.Recv_nmax_r2p, "Recv_nmax_r2p", log_file)
                for irank in range(self.Recv_nmax_r2p):
                    print(f"| Recv_r2p {' '.join(map(str, self.Recv_info_r2p[:, irank]))}", file=log_file)
                        
                #print(self.Send_nmax_r2p, "Send_nmax_r2p", log_file)
                for irank in range(self.Send_nmax_r2p):
                    print(f"| Send_r2p {' '.join(map(str, self.Send_info_r2p[:, irank]))}", file=log_file)

        debug = False
        if debug:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("", file=log_file)
                    print("--- Copy_list_r2p", file=log_file)
                    print("", file=log_file)
                    print(f"{'number':>6} {'|ifrom':>6} {'|jfrom':>6} {'|rfrom':>6} {'|lfrom':>6} {'|pfrom':>6} {'|rto':>6} {'|ito':>6} {'|jto':>6} {'|lto':>6} {'|pto':>6}", file=log_file)
                    for ipos in range(self.Copy_info_r2p[self.I_size]):
                        i_from = self.Copy_list_r2p[self.I_gridi_from, ipos]
                        j_from = self.Copy_list_r2p[self.I_gridj_from, ipos]
                        l_from = self.Copy_list_r2p[self.I_l_from, ipos]
                        p_from = self.Copy_info_r2p[self.I_prc_from]
                        #i_from = (g_from - 1) % self.ADM_gall_1d + 1
                        #j_from = (g_from - i_from) // self.ADM_gall_1d + 1
                        r_from = adm.RGNMNG_lp2r[l_from, p_from]
                        i_to = self.Copy_list_r2p[self.I_gridi_to, ipos]
                        j_to = self.Copy_list_r2p[self.I_gridj_to, ipos]
                        l_to = self.Copy_list_r2p[self.I_l_to, ipos]
                        p_to = self.Copy_info_r2p[self.I_prc_to]
                        r_to = l_to
                        print(f"{ipos:6} {i_from:6} {j_from:6} {r_from:6} {l_from:6} {p_from:6} {r_to:6} {i_to:6} {j_to:6} {l_to:6} {p_to:6}", file=log_file)
                        
                    print("", file=log_file)
                    print("--- Recv_list_r2p", file=log_file)
                    for irank in range(self.Recv_nmax_r2p):
                        print(f"{'number':>6} {'|ifrom':>6} {'|jfrom':>6} {'|rfrom':>6} {'|lfrom':>6} {'|pfrom':>6} {'|rto':>6} {'|ito':>6} {'|jto':>6} {'|lto':>6} {'|pto':>6}", file=log_file)
                        for ipos in range(self.Recv_info_r2p[self.I_size, irank]):
                            i_from = self.Recv_list_r2p[self.I_gridi_from, ipos, irank]
                            j_from = self.Recv_list_r2p[self.I_gridj_from, ipos, irank]
                            l_from = self.Recv_list_r2p[self.I_l_from, ipos, irank]
                            p_from = self.Recv_info_r2p[self.I_prc_from, irank]
                            #i_from = (g_from - 1) % self.ADM_gall_1d + 1
                            #j_from = (g_from - i_from) // self.ADM_gall_1d + 1
                            r_from = adm.RGNMNG_lp2r[l_from, p_from]
                            i_to = self.Recv_list_r2p[self.I_gridi_to, ipos, irank]
                            j_to = self.Recv_list_r2p[self.I_gridj_to, ipos, irank]
                            l_to = self.Recv_list_r2p[self.I_l_to, ipos, irank]
                            p_to = self.Recv_info_r2p[self.I_prc_to, irank]
                            r_to = l_to
                            print(f"{ipos:6} {i_from:6} {j_from:6} {r_from:6} {l_from:6} {p_from:6} {r_to:6} {i_to:6} {j_to:6} {l_to:6} {p_to:6}", file=log_file)
                        
                    print("", file=log_file)
                    print("--- Send_list_r2p", file=log_file)
                    for irank in range(self.Send_nmax_r2p):
                        print(f"{'number':>6} {'|ifrom':>6} {'|jfrom':>6} {'|rfrom':>6} {'|lfrom':>6} {'|pfrom':>6} {'|rto':>6} {'|ito':>6} {'|jto':>6} {'|lto':>6} {'|pto':>6}", file=log_file)
                        #print("self.Send_info_r2p[self.I_size, irank]= ",self.Send_info_r2p[self.I_size, irank], file=log_file)
                        for ipos in range(self.Send_info_r2p[self.I_size, irank]):
                            i_from = self.Send_list_r2p[self.I_gridi_from, ipos, irank]
                            j_from = self.Send_list_r2p[self.I_gridj_from, ipos, irank]
                            l_from = self.Send_list_r2p[self.I_l_from, ipos, irank]
                            p_from = self.Send_info_r2p[self.I_prc_from, irank]
                            #i_from = (g_from - 1) % self.ADM_gall_1d + 1
                            #j_from = (g_from - i_from) // self.ADM_gall_1d + 1
                            r_from = adm.RGNMNG_lp2r[l_from, p_from]
                            i_to = self.Send_list_r2p[self.I_gridi_to, ipos, irank]
                            j_to = self.Send_list_r2p[self.I_gridj_to, ipos, irank]
                            l_to = self.Send_list_r2p[self.I_l_to, ipos, irank]
                            p_to = self.Send_info_r2p[self.I_prc_to, irank]
                            r_to = l_to
                            print(f"{ipos:6} {i_from:6} {j_from:6} {r_from:6} {l_from:6} {p_from:6} {r_to:6} {i_to:6} {j_to:6} {l_to:6} {p_to:6}", file=log_file)

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print(f"*** Send_size_p2r,r2p     =  ", self.Send_size_nglobal_pl, file=log_file)
                print("", file=log_file) 


        #print("Buffer size info of p2r", Send_size_nglobal_pl, adm.ADM_kall, self.COMM_varmax, self.Send_nmax_p2r)
        #print("Buffer size info of r2p", Send_size_nglobal_pl, adm.ADM_kall, self.COMM_varmax, self.Send_nmax_r2p)
        #self.sendbuf_r2r = np.empty((Send_size_nglobal * adm.ADM_kall * self.COMM_varmax, self.Send_nmax_r2r), dtype=self.rdtype)
        #self.recvbuf_r2r = np.empty((Send_size_nglobal * adm.ADM_kall * self.COMM_varmax, self.Recv_nmax_r2r), dtype=self.rdtype) 
        ###self.sendbuf_p2r = np.empty((self.Send_size_nglobal_pl * adm.ADM_kall * self.COMM_varmax,), dtype=self.rdtype)
        ###self.recvbuf_p2r = np.empty((self.Send_size_nglobal_pl * adm.ADM_kall * self.COMM_varmax, self.Recv_nmax_p2r,), dtype=self.rdtype) 
        #self.recvbuf_p2r = np.empty((Send_size_nglobal_pl * adm.ADM_kall * self.COMM_varmax), dtype=self.rdtype) 
        ###self.sendbuf_p2r = np.ascontiguousarray(self.sendbuf_p2r)
        ###self.recvbuf_p2r = np.ascontiguousarray(self.recvbuf_p2r)
        
        self.sendbuf_r2p = np.empty((self.Send_size_nglobal_pl * adm.ADM_kall * self.COMM_varmax,), dtype=self.rdtype)
        self.recvbuf_r2p = np.empty((self.Send_size_nglobal_pl * adm.ADM_kall * self.COMM_varmax, self.Recv_nmax_r2p,), dtype=self.rdtype) 
        #self.recvbuf_r2p = np.empty((Send_size_nglobal_pl * adm.ADM_kall * self.COMM_varmax), dtype=self.rdtype) 
        self.sendbuf_r2p = np.ascontiguousarray(self.sendbuf_r2p)
        self.recvbuf_r2p = np.ascontiguousarray(self.recvbuf_r2p)

        return

    def COMM_sortdest_singular(self):
        #print("COMM_sortdest_singular")

        self.Singular_info = np.full(self.info_vindex, -1, dtype=int)
        self.Singular_info[self.I_size] = 0
        self.Singular_list = np.full((self.list_vindex, 4 * adm.ADM_lall), -1, dtype=int)
        
        for l in range(adm.ADM_lall):
            rgnid = adm.RGNMNG_l2r[l]
            
            if adm.RGNMNG_vert_num[adm.I_W, rgnid] == 3:
                ipos = self.Singular_info[self.I_size]
                self.Singular_info[self.I_size] += 1                
                i = adm.ADM_gmin
                j = adm.ADM_gmin - 1
                i_rmt = adm.ADM_gmin - 1
                j_rmt = adm.ADM_gmin - 1
                
                self.Singular_list[self.I_gridi_from, ipos] = i #self.suf(i, j)
                self.Singular_list[self.I_gridj_from, ipos] = j #self.suf(i, j)
                self.Singular_list[self.I_l_from, ipos] = l
                self.Singular_list[self.I_gridi_to, ipos] = i_rmt #_rmt #self.suf(i_rmt, j_rmt)
                self.Singular_list[self.I_gridj_to, ipos] = j_rmt #self.suf(i_rmt, j_rmt)
                self.Singular_list[self.I_l_to, ipos] = l
            
            if adm.RGNMNG_vert_num[adm.I_N, rgnid] != 4:
                ipos = self.Singular_info[self.I_size]
                self.Singular_info[self.I_size] += 1

                i = adm.ADM_gmin
                j = adm.ADM_gmax + 1
                i_rmt = adm.ADM_gmin - 1
                j_rmt = adm.ADM_gmax + 1
                
                self.Singular_list[self.I_gridi_from, ipos] = i #self.suf(i, j)
                self.Singular_list[self.I_gridj_from, ipos] = j
                self.Singular_list[self.I_l_from, ipos] = l
                self.Singular_list[self.I_gridi_to, ipos] = i_rmt #self.suf(i_rmt, j_rmt)
                self.Singular_list[self.I_gridj_to, ipos] = j_rmt
                self.Singular_list[self.I_l_to, ipos] = l
            
            if adm.RGNMNG_vert_num[adm.I_S, rgnid] != 4:
                ipos = self.Singular_info[self.I_size]
                self.Singular_info[self.I_size] += 1
                
                i = adm.ADM_gmax + 1 
                j = adm.ADM_gmin
                i_rmt = adm.ADM_gmax + 1
                j_rmt = adm.ADM_gmin - 1
                
                self.Singular_list[self.I_gridi_from, ipos] = i #self.suf(i, j)
                self.Singular_list[self.I_gridj_from, ipos] = j
                self.Singular_list[self.I_l_from, ipos] = l
                self.Singular_list[self.I_gridi_to, ipos] = i_rmt #self.suf(i_rmt, j_rmt)
                self.Singular_list[self.I_gridj_to, ipos] = j_rmt
                self.Singular_list[self.I_l_to, ipos] = l


        if self.Singular_info[self.I_size] > 0:
            self.Singular_nmax = 1
            self.Singular_info[self.I_prc_from] = adm.ADM_prc_me
            self.Singular_info[self.I_prc_to] = adm.ADM_prc_me
        
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print("|---------------------------------------", file=log_file)
                print("|               size  prc_from    prc_to", file=log_file)
                print(f"| Singular {' '.join(map(str, self.Singular_info))}", file=log_file)
                
                print("", file=log_file)
                print("--- Singular_list", file=log_file)
                print("", file=log_file)
                print(f"{'number':>6} {'|ifrom':>6} {'|jfrom':>6} {'|rfrom':>6} {'|lfrom':>6} {'|pfrom':>6} {'|ito':>6} {'|jto':>6} {'|rto':>6} {'|lto':>6} {'|pto':>6}", file=log_file)
                for ipos in range(self.Singular_info[self.I_size]):
                    i_from = self.Singular_list[self.I_gridi_from, ipos]
                    j_from = self.Singular_list[self.I_gridj_from, ipos]
                    l_from = self.Singular_list[self.I_l_from, ipos]
                    p_from = self.Singular_info[self.I_prc_from]
                    #i_from = (g_from - 1) % self.ADM_gall_1d + 1
                    #j_from = (g_from - i_from) // self.ADM_gall_1d + 1
                    r_from = adm.RGNMNG_lp2r[l_from, p_from]
                    i_to = self.Singular_list[self.I_gridi_to, ipos]
                    j_to = self.Singular_list[self.I_gridj_to, ipos]
                    l_to = self.Singular_list[self.I_l_to, ipos]
                    p_to = self.Singular_info[self.I_prc_to]
                    #i_to = (g_to - 1) % self.ADM_gall_1d + 1
                    #j_to = (g_to - i_to) // self.ADM_gall_1d + 1
                    r_to = adm.RGNMNG_lp2r[l_to, p_to]
                    print(f"{ipos:6} {i_from:6} {j_from:6} {r_from:6} {l_from:6} {p_from:6} {i_to:6} {j_to:6} {r_to:6} {l_to:6} {p_to:6}", file=log_file)
        
        return
    
    def COMM_data_transfer(self, var, var_pl):

        if(self.COMM_apply_barrier): 
            prf.PROF_rapstart('COMM_barrier', 2) 
            prc.PRC_MPIbarrier()
            prf.PROF_rapend('COMM_barrier', 2) 
        #    call PROF_rapstart('COMM_barrier',2)
        #    call MPI_Barrier(PRC_LOCAL_COMM_WORLD,ierr)
        #    call PROF_rapend  ('COMM_barrier',2)
        #endif

        prf.PROF_rapstart('COMM_data_transfer', 2) 

        #call PROF_rapstart('COMM_data_transfer',2)

        # var has the shape of (i, j, k, l, v), all i, j, k data a rank holds (i,e, for all l and v)

        shp = np.shape(var)  # Get the shape of the array
        vdtype = var.dtype  # Get the data type of the array
        kmax = shp[2]  # Equivalent to shp(2) in Fortran (1-based indexing → 0-based)
        vmax = shp[4]  # Equivalent to shp(4) in Fortran

        if kmax * vmax > adm.ADM_kall * self.COMM_varmax:
            print("xxx [COMM_data_transfer] kmax * vmax exceeds ADM_kall * COMM_varmax, stop!")
            print(f"xxx kmax * vmax            = {kmax * vmax}")
            print(f"xxx ADM_kall * COMM_varmax = {adm.ADM_kall * self.COMM_varmax}")
            prc.PRC_MPIstop(std.io_l, std.fname_log)  

        # ---< start communication >---
        # Theres no p2r & r2p communication without calling COMM_sortdest_pl.
        # receive pole   => region
        # receive region => pole
        # receive region => region
        # pack and send pole   => region
        # pack and send region => pole
        # pack and send region => region
        # copy pole   => region
        # copy region => pole
        # copy region => region
        # wait all
        # unpack pole   => region
        # unpack region => pole
        # unpack region => region
        # copy region halo => region halo (singular point)

        REQ_count = 0

        recv_slices = []
        recv_slices_p2r = []
        recv_slices_r2p = []
        REQ_list = []

        nrec = 0
        
        # --- Receive r2r ---
        for irank in range(self.Recv_nmax_r2r):  # Adjust for zero-based indexing

            #if prc.prc_myrank == 1:
            #    break

            rank = self.Recv_info_r2r[self.I_prc_from, irank]   # rank = prc 
            tag = rank
            imax = self.Recv_info_r2r[self.I_size, irank]  ###
            recvbuf1_r2r = np.empty((imax * kmax * vmax,), dtype=vdtype)
            #recvbuf1_r2r = np.empty((self.Send_size_nglobal * adm.ADM_kall * self.COMM_varmax), dtype=vdtype) # 68*1*15=1020 = 8160bytes
            recvbuf1_r2r = np.ascontiguousarray(recvbuf1_r2r)
            recv_slices.append(recvbuf1_r2r)
            REQ_list.append(prc.comm_world.Irecv(recv_slices[irank], source=rank, tag=tag))
            REQ_count += 1

            with open(std.fname_log, 'a') as log_file:
                print("myrank", prc.prc_myrank, "source rank", rank,  "tag", tag, "imax", imax, "recv_slices", recv_slices[irank].shape, file=log_file) 

            #reg = adm.RGNMNG_lp2r[2, prc.prc_myrank]
            #print("reg", reg, prc.prc_myrank)
            #if reg == 7:
            #    print("source", rank) #, self.Send_list_r2r[self.I_l_from, ipos, irank])

        # --- Receive p2r ---
        for irank in range(self.Recv_nmax_p2r):  # Adjust for zero-based indexing
            rank = self.Recv_info_p2r[self.I_prc_from, irank]   # rank = prc
            tag = rank + 1000000  # Adjusted tag
            #recvbuf1_p2r = np.empty((self.Send_size_nglobal_pl * adm.ADM_kall * self.COMM_varmax), dtype=vdtype) 
            recvbuf1_p2r = np.empty((self.Send_size_nglobal_pl * kmax * vmax), dtype=vdtype) 
            recvbuf1_p2r = np.ascontiguousarray(recvbuf1_p2r)
            recv_slices_p2r.append(recvbuf1_p2r)
            REQ_list.append(prc.comm_world.Irecv(recv_slices_p2r[irank], source=rank, tag=tag))
            REQ_count += 1

        # --- Receive r2p ---
        for irank in range(self.Recv_nmax_r2p):  # Adjust for zero-based indexing
            rank = self.Recv_info_r2p[self.I_prc_from, irank]   # rank = prc
            tag = rank + 2000000  # Adjusted tag
            recvbuf1_r2p = np.empty((self.Send_size_nglobal_pl * adm.ADM_kall * self.COMM_varmax), dtype=vdtype) 
            #recvbuf1_r2p = np.empty((self.Send_size_nglobal_pl * kmax * vmax), dtype=vdtype) 
            recvbuf1_r2p = np.ascontiguousarray(recvbuf1_r2p)
            recv_slices_r2p.append(recvbuf1_r2p)
            REQ_list.append(prc.comm_world.Irecv(recv_slices_r2p[irank], source=rank, tag=tag))
            REQ_count += 1

    

        # --- Pack and Send r2r ---
        for irank in range(self.Send_nmax_r2r):  # Adjust for zero-based indexing
            imax = self.Send_info_r2r[self.I_size,irank]
            self.sendbuf_r2r = np.empty((imax * kmax * vmax,), dtype=vdtype)
            self.sendbuf_r2r[:] = -999. 
            
            for v in range(vmax):
                for k in range(kmax):
                    for ipos in range(imax):  # i,j,l are extracted from the list using ipos
                        i_from = self.Send_list_r2r[self.I_gridi_from, ipos, irank]
                        j_from = self.Send_list_r2r[self.I_gridj_from, ipos, irank]
                        l_from = self.Send_list_r2r[self.I_l_from, ipos, irank]
                        ikv = (v * imax * kmax) + (k * imax) + ipos
                        # if v==1:
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("abs transfer send", var[i_from, j_from, k, l_from, 0]**2 + var[i_from, j_from, k, l_from, 1]**2 + var[i_from, j_from, k, l_from, 2]**2, file=log_file)
                        self.sendbuf_r2r[ikv] = var[i_from, j_from, k, l_from, v]
                        
                        # if self.Send_info_r2r[self.I_prc_to, irank]  == 1:
                        #     i_to = self.Send_list_r2r[self.I_gridi_to, ipos, irank]
                        #     j_to = self.Send_list_r2r[self.I_gridj_to, ipos, irank]
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("send to RANK 1 from region:  ", adm.RGNMNG_lp2r[l_from, prc.prc_myrank], i_from, j_from, i_to, j_to, v, file=log_file)
                        #         print(var[i_from, j_from, k, l_from, v], file= log_file)
                        #         print(adm.RGNMNG_lp2r[self.Send_list_r2r[self.I_l_to, ipos, irank], self.Send_info_r2r[self.I_prc_to, irank]], file=log_file)


            rank = self.Send_info_r2r[self.I_prc_to, irank]   # rank = prc (your rank)
            tag = self.Send_info_r2r[self.I_prc_from, irank]   # tag = prc (my rank)
            self.sendbuf_r2r= np.ascontiguousarray(self.sendbuf_r2r)
            REQ_list.append(
                prc.comm_world.Isend(self.sendbuf_r2r, dest=rank, tag=tag)    # 68*15*8bytes=  1020*8bytes = 8160 bytes   
                )   
            with open(std.fname_log, 'a') as log_file:
                print("myrank", prc.prc_myrank, "dest rank", rank,  "tag", tag, "imax", imax, "sendbuf", self.sendbuf_r2r.shape, file=log_file)

            REQ_count += 1

        # --- Pack and Send p2r ---
        for irank in range(self.Send_nmax_p2r):  # Adjust for zero-based indexing
            imax = self.Send_info_p2r[self.I_size, irank]
            #self.sendbuf_p2r = np.empty((self.Send_size_nglobal_pl * adm.ADM_kall * self.COMM_varmax,), dtype=self.rdtype)
            #self.sendbuf_p2r[:] = -999.
            self.sendbuf_p2r = np.empty((self.Send_size_nglobal_pl * kmax * vmax,), dtype=self.rdtype)
            for v in range(vmax):
                for k in range(kmax):
                   for ipos in range(imax):
                        i_from = self.Send_list_p2r[self.I_gridi_from, ipos, irank]
                        l_from = self.Send_list_p2r[self.I_l_from, ipos, irank]
                        ikv = (v * imax * kmax) + (k * imax) + ipos
                        self.sendbuf_p2r[ikv] = var_pl[i_from, k, l_from, v]
                        
                        if i_from == 0 and l_from == 0:
                            with open(std.fname_log, 'a') as log_file:
                                print("sending from pole:  ", i_from, l_from, v, file=log_file)
                                print("towards", self.Send_info_p2r[self.I_prc_to, irank] , self.Send_list_p2r[self.I_l_to, ipos, irank], self.Send_list_p2r[self.I_gridi_to, ipos, irank], self.Send_list_p2r[self.I_gridj_to, ipos, irank], file=log_file)
                                print(var_pl[i_from, k, l_from, v], file= log_file)


            self.sendbuf_p2r = np.ascontiguousarray(self.sendbuf_p2r)
            rank = self.Send_info_p2r[self.I_prc_to, irank]    # rank = prc
            tag = self.Send_info_p2r[self.I_prc_from, irank] + 1000000  # Adjusted tag
            REQ_list.append(
                prc.comm_world.Isend(self.sendbuf_p2r, dest=rank, tag=tag)
            )
            REQ_count += 1

        # --- Pack and Send r2p ---
        for irank in range(self.Send_nmax_r2p):  # Adjust for zero-based indexing
            imax = self.Send_info_r2p[self.I_size, irank]
            for v in range(vmax):
                for k in range(kmax):
                    for ipos in range(imax):
                        i_from = self.Send_list_r2p[self.I_gridi_from, ipos, irank]
                        j_from = self.Send_list_r2p[self.I_gridj_from, ipos, irank]
                        l_from = self.Send_list_r2p[self.I_l_from, ipos, irank]
                        ikv = (v * imax * kmax) + (k * imax) + ipos
                        self.sendbuf_r2p[ikv] = var[i_from, j_from, k, l_from, v]
            self.sendbuf_r2p = np.ascontiguousarray(self.sendbuf_r2p)
            rank = self.Send_info_r2p[self.I_prc_to, irank]   # rank = prc 
            tag = self.Send_info_r2p[self.I_prc_from, irank] + 2000000  # Adjusted tag
            REQ_list.append(
                prc.comm_world.Isend(self.sendbuf_r2p, dest=rank, tag=tag)
            )
            REQ_count += 1

        # --- Copy r2r ---
        for irank in range(self.Copy_nmax_r2r):  # Adjust for zero-based indexing
            imax = self.Copy_info_r2r[self.I_size]
            for v in range(vmax):
                for k in range(kmax):
                    for ipos in range(imax):
                        i_from = self.Copy_list_r2r[self.I_gridi_from, ipos]
                        j_from = self.Copy_list_r2r[self.I_gridj_from, ipos]
                        l_from = self.Copy_list_r2r[self.I_l_from, ipos]
                        i_to = self.Copy_list_r2r[self.I_gridi_to, ipos]
                        j_to = self.Copy_list_r2r[self.I_gridj_to, ipos]
                        l_to = self.Copy_list_r2r[self.I_l_to, ipos]
                        var[i_to, j_to, k, l_to, v] = var[i_from, j_from, k, l_from, v]

        # --- Copy p2r ---
        for irank in range(self.Copy_nmax_p2r):  # Adjust for zero-based indexing
            imax = self.Copy_info_p2r[self.I_size]
            for v in range(vmax):
                for k in range(kmax):
                    for ipos in range(imax):
                        i_from = self.Copy_list_p2r[self.I_gridi_from, ipos]
                        l_from = self.Copy_list_p2r[self.I_l_from, ipos]
                        i_to = self.Copy_list_p2r[self.I_gridi_to, ipos]
                        j_to = self.Copy_list_p2r[self.I_gridj_to, ipos]
                        l_to = self.Copy_list_p2r[self.I_l_to, ipos]
                        var[i_to, j_to, k, l_to, v] = var_pl[i_from, k, l_from, v]

        # --- Copy r2p ---
        for irank in range(self.Copy_nmax_r2p):  # Adjust for zero-based indexing
            imax = self.Copy_info_r2p[self.I_size]
            for v in range(vmax):
                for k in range(kmax):
                    for ipos in range(imax):
                        i_from = self.Copy_list_r2p[self.I_gridi_from, ipos]
                        j_from = self.Copy_list_r2p[self.I_gridj_from, ipos]
                        l_from = self.Copy_list_r2p[self.I_l_from, ipos]
                        i_to = self.Copy_list_r2p[self.I_gridi_to, ipos]
                        l_to = self.Copy_list_r2p[self.I_l_to, ipos]
                        var_pl[i_to, k, l_to, v] = var[i_from, j_from, k, l_from, v]

        # --- Wait for all MPI requests ---

        if REQ_count > 0:
            MPI.Request.Waitall(REQ_list)

            #statuses = [MPI.Status() for _ in REQ_list]  # Create an array of MPI statuses
            #
            #for i, req in enumerate(REQ_list):
            #    if req is not None:
            #        try:
            #            req.Wait(statuses[i])  # Wait for each request individually
            #            error_code = statuses[i].Get_error()
            #            if error_code != MPI.SUCCESS:
            #                print(f"Request {i} failed with MPI_ERROR={error_code}")
            #        except MPI.Exception as e:
            #            print(f"Exception in request {i}: {e}")

        # --- Unpack r2r ---
        for irank in range(self.Recv_nmax_r2r):  # Adjust for zero-based indexing
            imax = self.Recv_info_r2r[self.I_size, irank]
            ###size1 = self.Send_size_nglobal * adm.ADM_kall * self.COMM_varmax
            #print("size1, irank, globalsize, varmax", size1, irank, self.Send_size_nglobal, self.COMM_varmax)
            self.recvbuf_r2r = np.empty((imax * kmax * vmax,self.Recv_nmax_r2r,), dtype=vdtype)
            self.recvbuf_r2r[:,irank] = recv_slices[irank]
            ###self.recvbuf_r2r[0:size1,irank] = recv_slices[irank]
            #print("irank, self.recvbuf_r2r[0:size1,irank]", irank, self.recvbuf_r2r[0:3*68,irank])
            #self.recvbuf_r2r[irank,0:size1] = recv_slices[irank]
            rank = self.Recv_info_r2r[self.I_prc_from, irank]
            for v in range(vmax):
                for k in range(kmax):
                    for ipos in range(imax):
                        i_to = self.Recv_list_r2r[self.I_gridi_to, ipos, irank]
                        j_to = self.Recv_list_r2r[self.I_gridj_to, ipos, irank]
                        l_to = self.Recv_list_r2r[self.I_l_to, ipos, irank]
                        ikv = (v * imax * kmax) + (k * imax) + ipos

                        var[i_to, j_to, k, l_to, v] = self.recvbuf_r2r[ikv,irank]
 
                        # if (i_to==0 or i_to==1) and j_to==17 and l_to==0 and prc.prc_myrank==2 and v==1:  
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("Found in r2r", i_to, j_to, l_to, v, var[i_to, j_to, k, l_to, v], file=log_file)


                        #i_from =  self.Recv_list_r2r[self.I_gridi_from, ipos, irank]
                        #j_from =  self.Recv_list_r2r[self.I_gridj_from, ipos, irank]
                        #l_from =  self.Recv_list_r2r[self.I_l_from, ipos, irank]
                        
                        #if prc.prc_myrank == 1:
                            # with open (std.fname_log, 'a') as log_file:
                            #     print("region_from, i_from, j_from, region_to, i_to, j_to, v", file=log_file)
                            #     print(adm.RGNMNG_lp2r[l_from,rank], i_from, j_from, adm.RGNMNG_lp2r[l_to, prc.prc_myrank], i_to, j_to, v, file=log_file)
                            #     print(var[i_from, j_from, k, l_from, v], file= log_file)

                            # if rank == 6:
                            #     with open(std.fname_log, 'a') as log_file:
                            #         print("recv from rank 6 region", adm.RGNMNG_lp2r[l_from,rank], i_from, j_from, i_to, j_to, v, file=log_file)
                            #         print(var[i_to, j_to, k, l_to, v], file= log_file)

                        #var[i_to, j_to, k, l_to, v] = self.recvbuf_r2r[irank,ikv]
                        # if v==1:
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("abs transfer unpack", var[i_to, j_to, k, l_to, 0]**2 + var[i_to, j_to, k, l_to, 1]**2 + var[i_to, j_to, k, l_to, 2]**2, file=log_file)
                        

        # --- Unpack p2r ---
        for irank in range(self.Recv_nmax_p2r):  # Adjust for zero-based indexing
            imax = self.Recv_info_p2r[self.I_size, irank]
            #size1 = self.Send_size_nglobal_pl * adm.ADM_kall * self.COMM_varmax
            #self.recvbuf_p2r = np.empty((self.Send_size_nglobal_pl * adm.ADM_kall * self.COMM_varmax, self.Recv_nmax_p2r,), dtype=self.rdtype)        
            size1 = self.Send_size_nglobal_pl * kmax * vmax
            self.recvbuf_p2r = np.empty((self.Send_size_nglobal_pl * kmax * vmax, self.Recv_nmax_p2r,), dtype=self.rdtype)        
            self.recvbuf_p2r[0:size1,irank] = recv_slices_p2r[irank]
            for v in range(vmax):
                for k in range(kmax):
                    for ipos in range(imax):
                        i_to = self.Recv_list_p2r[self.I_gridi_to, ipos, irank]
                        j_to = self.Recv_list_p2r[self.I_gridj_to, ipos, irank] 
                        l_to = self.Recv_list_p2r[self.I_l_to, ipos, irank]
                        ikv = (v * imax * kmax) + (k * imax) + ipos
                        var[i_to, j_to, k, l_to, v] = self.recvbuf_p2r[ikv,irank]
                        if (i_to==0 or i_to==1) and j_to==17 and l_to==0 and k==0 and prc.prc_myrank==2: #and v==1:  
                            with open(std.fname_log, 'a') as log_file:
                                i_from = self.Recv_list_p2r[self.I_gridi_from, ipos, irank]
                                j_from = self.Recv_list_p2r[self.I_gridj_from, ipos, irank] 
                                l_from = self.Recv_list_p2r[self.I_l_from, ipos, irank]
                                print("Found in p2r", i_to, j_to, l_to, v, var[i_to, j_to, k, l_to, v], file=log_file)
                                print("SENT from", i_from, j_from, l_from, self.Recv_info_p2r[self.I_prc_from, irank], file=log_file)

        # --- Unpack r2p ---
        for irank in range(self.Recv_nmax_r2p):  # Adjust for zero-based indexing
            imax = self.Recv_info_r2p[self.I_size, irank]
            size1 = self.Send_size_nglobal_pl * adm.ADM_kall * self.COMM_varmax
            self.recvbuf_r2p[0:size1,irank] = recv_slices_r2p[irank]
            for v in range(vmax):
                for k in range(kmax):
                    for ipos in range(imax):
                        i_to = self.Recv_list_r2p[self.I_gridi_to, ipos, irank]
                        l_to = self.Recv_list_r2p[self.I_l_to, ipos, irank]
                        ikv = (v * imax * kmax) + (k * imax) + ipos
                        var_pl[i_to, k, l_to, v] = self.recvbuf_r2p[ikv,irank]
                        # if (i_to==0 or i_to==1) and j_to==17 and l_to==0 and prc.prc_myrank==2 and v==1:  
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("Found in r2p", i_to, j_to, l_to, v, var_pl[i_to, k, l_to, v], file=log_file)



        # --- Singular point (halo to halo) ---
        for irank in range(self.Singular_nmax):  # Adjust for zero-based indexing
            imax = self.Singular_info[self.I_size]
            for v in range(vmax):
                for k in range(kmax):
                    for ipos in range(imax):
                        i_from = self.Singular_list[self.I_gridi_from, ipos]
                        j_from = self.Singular_list[self.I_gridj_from, ipos]
                        l_from = self.Singular_list[self.I_l_from, ipos]
                        i_to = self.Singular_list[self.I_gridi_to, ipos]
                        j_to = self.Singular_list[self.I_gridj_to, ipos]
                        l_to = self.Singular_list[self.I_l_to, ipos]
                        var[i_to, j_to, k, l_to, v] = var[i_from, j_from, k, l_from, v]
                        # if (i_to==0 or i_to==1) and j_to==17 and l_to==0 and prc.prc_myrank==2 and v==1:  
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("Found at singular point", i_to, j_to, l_to, v, var[i_to, j_to, k, l_to, v], file=log_file)
                        #         print("SENT from", i_from, j_from, l_from, var[i_from, j_from, k, l_from, v], file=log_file)

        prf.PROF_rapend('COMM_data_transfer', 2) 


        #prc.PRC_MPIbarrier()
        #print("peace?")
        #prc.prc_mpistop(std.io_l, std.fname_log)

        return

    def Comm_Stat_max(self,localmax,rdtype):
        
        sendbuf = np.array([localmax], dtype=rdtype)  # Single-element send buffer
        recvbuf = np.empty(prc.comm_world.Get_size(), dtype=rdtype)  # Allocate receive buffer

        prc.comm_world.Allgather(sendbuf, recvbuf)
        globalmax = np.max(recvbuf)

        return globalmax
    
    def Comm_Stat_min(self,localmin,rdtype):
        
        sendbuf = np.array([localmin], dtype=rdtype)
        recvbuf = np.empty(prc.comm_world.Get_size(), dtype=rdtype)

        prc.comm_world.Allgather(sendbuf, recvbuf)
        globalmin = np.min(recvbuf)

        return globalmin
    
    def Comm_Stat_sum(self,localsum,rdtype):
        
        if ( self.COMM_pl):
            sendbuf = np.array([localsum], dtype=rdtype)
            recvbuf = np.empty(prc.comm_world.Get_size(), dtype=rdtype)
            prc.comm_world.Allgather(sendbuf, recvbuf)
            globalsum = np.sum(recvbuf)
        else:
            globalsum = localsum

        #globalsum = prc.comm_world.allreduce(localsum, op=MPI.SUM)

        return globalsum

    def COMM_var(self, var, var_pl):
        
        shp = np.shape(var)  # Get the shape of the array
        vdtype = var.dtype  # Get the data type of the array
        kmax = shp[2]   # 42
        vmax = shp[4]   #  7

        with open(std.fname_log, 'a') as log_file:
            print("kmax, vmax", kmax, vmax, file=log_file) 

        #self.sendbuf_h2p = np.empty((kmax * vmax,), dtype=vdtype)
        #self.sendbuf_h2p = np.ascontiguousarray(self.sendbuf_h2p)

        if self.COMM_apply_barrier:
            prf.PROF_rapstart('COMM_barrier', 2) 
            prc.PRC_MPIbarrier()
            prf.PROF_rapend('COMM_barrier', 2)

        prf.PROF_rapstart('COMM_var', 2)

        if self.COMM_pl:

            #print("self.COMM_pl is True")

            REQ_list_NS = np.empty((self.Recv_nmax_p2r + self.Send_nmax_p2r,), dtype=object)
            REQ_list_NS.fill(MPI.REQUEST_NULL)  
            REQ_list_NS = []

            #--- receive p2r-reverse
            for irank in range(self.Send_nmax_p2r): 
                for ipos in range(self.Send_info_p2r[self.I_size, irank]):
                    l_from = self.Send_list_p2r[self.I_l_to, ipos, irank]
                    r_from = adm.RGNMNG_lp2r[l_from, self.Send_info_p2r[self.I_prc_to, irank]] 

                    if (r_from == adm.RGNMNG_rgn4pl[adm.I_NPL]):
                        rank      = self.Send_info_p2r[self.I_prc_to  ,irank]
                        tag       = self.Send_info_p2r[self.I_prc_from,irank] + 1000000
                        recvbuf1_h2p_n = np.empty((kmax * vmax,), dtype=vdtype) 
                        recvbuf1_h2p_n = np.ascontiguousarray(recvbuf1_h2p_n)
                        # if kmax > 0:
                        print("receiving NORTH..., source rank, tag, myrank", rank, tag, prc.prc_myrank)
                        REQ_list_NS.append(prc.comm_world.Irecv(recvbuf1_h2p_n, source=rank, tag=tag))
        
                    if (r_from == adm.RGNMNG_rgn4pl[adm.I_SPL]):
                        rank      = self.Send_info_p2r[self.I_prc_to  ,irank]
                        tag       = self.Send_info_p2r[self.I_prc_from,irank] + 2000000
                        recvbuf1_h2p_s = np.empty((kmax * vmax,), dtype=vdtype) 
                        recvbuf1_h2p_s = np.ascontiguousarray(recvbuf1_h2p_s)
                        # if kmax > 0:
                        #     print("receiving south..., source rank, tag, myrank", rank, tag, prc.prc_myrank)
                        REQ_list_NS.append(prc.comm_world.Irecv(recvbuf1_h2p_s, source=rank, tag=tag))
    
            #--- pack and send p2r-reverse
            for irank in range(self.Recv_nmax_p2r):
                for ipos in range(self.Recv_info_p2r[self.I_size,irank]):
                    i_from = self.Recv_list_p2r[self.I_gridi_to, ipos, irank]
                    j_from = self.Recv_list_p2r[self.I_gridj_to, ipos, irank]
                    l_from = self.Recv_list_p2r[self.I_l_to, ipos, irank]
                    r_from = adm.RGNMNG_lp2r[l_from, self.Recv_info_p2r[self.I_prc_to,irank]]

                    if (r_from == adm.RGNMNG_rgn4pl[adm.I_NPL]):
                        for k in range(kmax):
                            for v in range(vmax):       
                                kk = v * kmax + k
                                self.sendbuf_h2p[kk] = var[i_from,j_from,k,l_from,v]

                        rank = self.Recv_info_p2r[self.I_prc_from,irank]
                        tag  = self.Recv_info_p2r[self.I_prc_from,irank] + 1000000        
                        #if kmax > 0: 
                        with open(std.fname_log, 'a') as log_file:
                            print("SENDING north..., dest rank, tag, myrank", rank, tag, prc.prc_myrank, file=log_file) 
                            print(self.sendbuf_h2p.shape, file=log_file)
                            print(var[i_from,j_from,k,l_from,v], file=log_file)
                        #
                        REQ_list_NS.append(prc.comm_world.Isend(self.sendbuf_h2p, dest=rank, tag=tag))

                    if (r_from == adm.RGNMNG_rgn4pl[adm.I_SPL]):
                        self.sendbuf_h2p = np.empty((kmax * vmax,), dtype=vdtype)
                        #self.sendbuf_h2p = np.ascontiguousarray(self.sendbuf_h2p)
                        for k in range(kmax):
                            for v in range(vmax):       
                                kk = v * kmax + k
                        
                                self.sendbuf_h2p[kk] = var[i_from,j_from,k,l_from,v]
                                #print(f"Send SPL: var[{i_from},{j_from},{k},{l_from},{v}] = {var[i_from,j_from,k,l_from,v]}")

                        rank = self.Recv_info_p2r[self.I_prc_from,irank]
                        tag  = self.Recv_info_p2r[self.I_prc_from,irank] + 2000000  
                        self.sendbuf_h2p= np.ascontiguousarray(self.sendbuf_h2p)    
                        # if kmax > 0: 
                        #     print("sending south..., dest rank, tag, myrank", rank, tag, prc.prc_myrank)
                        #     print(self.sendbuf_h2p.shape)
                        #     #print(self.sendbuf_h2p)
                        REQ_list_NS.append(prc.comm_world.Isend(self.sendbuf_h2p, dest=rank, tag=tag))
    
            #--- copy p2r-reverse
            for irank in range(self.Copy_nmax_p2r):
                for ipos in range(self.Copy_info_p2r[self.I_size]):
                    i_from = self.Copy_list_p2r[self.I_gridi_to, ipos]
                    j_from = self.Copy_list_p2r[self.I_gridj_to, ipos]
                    l_from = self.Copy_list_p2r[self.I_l_to, ipos]
                    r_from = adm.RGNMNG_lp2r[l_from, self.Copy_info_p2r[self.I_prc_to]]
                    i_to   = self.Copy_list_p2r[self.I_gridi_from, ipos]
                    l_to   = self.Copy_list_p2r[self.I_l_from, ipos]
                    r_to   = adm.RGNMNG_lp2r[l_to, self.Copy_info_p2r[self.I_prc_from]]
                    
                    if r_from == adm.RGNMNG_rgn4pl[adm.I_NPL] or r_from == adm.RGNMNG_rgn4pl[adm.I_SPL]:
                        for k in range(kmax):
                            for v in range(vmax):
                                var_pl[i_to,k,l_to,v] = var[i_from,j_from,k,l_from,v]
                                if k < 3:
                                    with open(std.fname_log, 'a') as log_file:
                                         print(f"Copy NPLorSPL: var_pl[{i_to},{k},{l_to},{v}] = {var_pl[i_to,k,l_to,v]}", file=log_file)
                                #     print(f"from: var[{i_from},{j_from},{k},{l_from},{v}] = {var[i_from,j_from,k,l_from,v]}", file=log_file)
                                #     print("from  i, j, l, p, r, kmax:", i_from, j_from, l_from, self.Copy_info_p2r[self.I_prc_to], r_from, kmax, file=log_file)

                                #print(f"Copy NPL: var_pl[{i_to},{k},{l_to},{v}] = {var_pl[i_to,k,l_to,v]}")
                                # invalid value copied to north pole from region 2 i=1, j=17

            #--- wait all
            if len(REQ_list_NS) > 0:
                MPI.Request.Waitall(REQ_list_NS)
                #MPI.Request.Waitall()


            # statuses = [MPI.Status() for _ in REQ_list_NS]  # Create an array of MPI statuses            
            # for i, req in enumerate(REQ_list_NS):
            #     if req is not None:
            #         try:
            #             req.Wait(statuses[i])  # Wait for each request individually
            #             error_code = statuses[i].Get_error()
            #             if error_code != MPI.SUCCESS:
            #                 print(f"Request {i} failed with MPI_ERROR={error_code}")
            #         except MPI.Exception as e:
            #             print(f"Exception in request {i}: {e}")


            #--- unpack p2r-reverse
            if prc.prc_myrank == adm.ADM_prc_pl:
                for irank in range(self.Send_nmax_p2r):
                    for ipos in range(self.Send_info_p2r[self.I_size, irank]):
                        l_from = self.Send_list_p2r[self.I_l_to, ipos, irank]
                        r_from = adm.RGNMNG_lp2r[l_from, self.Send_info_p2r[self.I_prc_to, irank]]
                        ij_to = self.Send_list_p2r[self.I_gridi_from, ipos, irank]
                        l_to = self.Send_list_p2r[self.I_l_from, ipos, irank]

                        if r_from == adm.RGNMNG_rgn4pl[adm.I_NPL]:
                            for k in range(kmax):
                                for v in range(vmax):
                                    kk = v * kmax + k
                                    var_pl[ij_to, k, l_to, v] = recvbuf1_h2p_n[kk]
                                    #print(f"var_pl[{ij_to},{k},{l_to},{v}] = {var_pl[ij_to,k,l_to,v]}")

                        if r_from == adm.RGNMNG_rgn4pl[adm.I_SPL]:
                            for k in range(kmax):
                                for v in range(vmax):
                                    kk = v * kmax + k
                                    var_pl[ij_to, k, l_to, v] = recvbuf1_h2p_s[kk]
                                    #print(f"Recv SPL: var_pl[{ij_to},{k},{l_to},{v}] = {var_pl[ij_to,k,l_to,v]}")
        # if kmax > -100. :
        #     for i in range(18):
        #         for j in range(18):
        #             for k in range(kmax):
        #                 for l in range(5):
        #                     if  var[i,j,k,l,1] < 10. : 
        #                             print(i,j,k,l, var[i,j,k,l,1])          
        #                             with open(std.fname_log, 'a') as log_file:
        #                                 print("DIAGGERS!!!", file=log_file) 
        #                                 print(i,j,k,l,file=log_file)
        if kmax > 5:
            with open(std.fname_log, 'a') as log_file:
                print("DIAGGER1.3!", file=log_file) 
                print("i=0to4, j=17, k=0, l=0, cnt=1", var[0:4,17,0,0,1],file=log_file)
                print("i=0to4, j=17, k=0, l=0, cnt=1", var[0:4,17,19,0,1],file=log_file)
                print("i=0to4, j=17, k=0, l=0, cnt=1", var[0:4,17,20,0,1],file=log_file)
                print("i=0to4, j=17, k=0, l=0, cnt=1", var[0:4,17,21,0,1],file=log_file)
                print("i=0to4, j=17, k=0, l=0, cnt=1", var[0:4,17,40,0,1],file=log_file)
# #                print("i=0to4, j=17, k=1, l=0, cnt=1", var[0:4,17,1,0,1],file=log_file)
# #                print("i=0to4, j=17, k=2, l=0, cnt=1", var[0:4,17,2,0,1],file=log_file)


        self.COMM_data_transfer(var, var_pl)   # invalid value handed from north pole to region 10  i=1, j=17 by p2r
        prf.PROF_rapend('COMM_var', 2)


        if kmax > 5:
            with open(std.fname_log, 'a') as log_file:
                print("DIAGGER1.6!", file=log_file) 
                print("i=0to4, j=17, k=0, l=0, cnt=1", var[0:4,17,0,0,1],file=log_file)
                print("i=0to4, j=17, k=0, l=0, cnt=1", var[0:4,17,19,0,1],file=log_file)
                print("i=0to4, j=17, k=0, l=0, cnt=1", var[0:4,17,20,0,1],file=log_file)
                print("i=0to4, j=17, k=0, l=0, cnt=1", var[0:4,17,21,0,1],file=log_file)
                print("i=0to4, j=17, k=0, l=0, cnt=1", var[0:4,17,40,0,1],file=log_file)
#                print("i=0to4, j=17, k=1, l=0, cnt=1", var[0:4,17,1,0,1],file=log_file)
#                print("i=0to4, j=17, k=2, l=0, cnt=1", var[0:4,17,2,0,1],file=log_file)


        return
    

    #def suf(self, i, j, adm):
    #    return adm.ADM_gall_1d * j + i 
