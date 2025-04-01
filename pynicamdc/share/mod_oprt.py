import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf

class Oprt:
    
    _instance = None
    
    def __init__(self):
        pass

    def OPRT_setup(self, fname_in, gmtr, rdtype):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[oprt]/Category[common share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'oprtparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** oprtparam not found in toml file! Use default.", file=log_file)
                #prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['oprtparam']
            self.OPRT_io_mode = cnfs['OPRT_io_mode']
            self.OPRT_fname = cnfs['OPRT_fname']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        self.OPRT_fname = ""
        self.OPRT_io_mode = "ADVANCED"

        self.OPRT_coef_div     = np.zeros((adm.ADM_gall_1d,   adm.ADM_gall_1d, 7, adm.ADM_nxyz, adm.ADM_lall),    dtype=rdtype)
        self.OPRT_coef_div_pl  = np.zeros((adm.ADM_vlink + 1,                     adm.ADM_nxyz, adm.ADM_lall_pl), dtype=rdtype)

        self.OPRT_coef_rot     = np.zeros((adm.ADM_gall_1d,   adm.ADM_gall_1d, 7, adm.ADM_nxyz, adm.ADM_lall),    dtype=rdtype)
        self.OPRT_coef_rot_pl  = np.zeros((adm.ADM_vlink + 1,   adm.ADM_nxyz, adm.ADM_lall_pl),  dtype=rdtype)

        self.OPRT_coef_grad    = np.zeros((adm.ADM_gall_1d,   adm.ADM_gall_1d, 7, adm.ADM_nxyz, adm.ADM_lall),    dtype=rdtype)
        self.OPRT_coef_grad_pl = np.zeros((adm.ADM_vlink + 1,   adm.ADM_nxyz, adm.ADM_lall_pl),  dtype=rdtype)

        self.OPRT_coef_lap     = np.zeros((adm.ADM_gall_1d,   adm.ADM_gall_1d, 7,               adm.ADM_lall),    dtype=rdtype)
        self.OPRT_coef_lap_pl  = np.zeros((adm.ADM_vlink + 1,                 adm.ADM_lall_pl),  dtype=rdtype)

        self.OPRT_coef_intp    = np.zeros((adm.ADM_gall_1d,   adm.ADM_gall_1d, 3, adm.ADM_nxyz, adm.ADM_TJ - adm.ADM_TI + 1, adm.ADM_lall), dtype=rdtype)
        self.OPRT_coef_intp_pl = np.zeros((adm.ADM_gall_pl,  3, adm.ADM_nxyz,                   adm.ADM_lall_pl), dtype=rdtype)

        self.OPRT_coef_diff    = np.zeros((adm.ADM_gall_1d,   adm.ADM_gall_1d, 6, adm.ADM_nxyz, adm.ADM_lall),    dtype=rdtype)
        self.OPRT_coef_diff_pl = np.zeros((adm.ADM_vlink,       adm.ADM_nxyz, adm.ADM_lall_pl),  dtype=rdtype)

        self.OPRT_divergence_setup(gmtr, rdtype)

        self.OPRT_rotation_setup(gmtr, rdtype)
        
        self.OPRT_gradient_setup(gmtr, rdtype)
        
        self.OPRT_laplacian_setup(gmtr, rdtype)
        
        self.OPRT_diffusion_setup(gmtr, rdtype)


        return
    
    def OPRT_divergence_setup(self, gmtr, rdtype):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("*** setup coefficient of divergence operator", file=log_file)        
        #           1                    18               1
        #gmin = (adm.ADM_gmin - 1) * adm.ADM_gall_1d + adm.ADM_gmin
        #           16                   18               16
        #gmax = (adm.ADM_gmax - 1) * adm.ADM_gall_1d + adm.ADM_gmax
        gmin = adm.ADM_gmin #1
        gmax = adm.ADM_gmax #16
        iall = adm.ADM_gall_1d #18 
        gall = adm.ADM_gall
        nxyz = adm.ADM_nxyz  #3
        lall = adm.ADM_lall
        k0 = adm.ADM_K0
        P_RAREA = gmtr.GMTR_p_RAREA
        AI = adm.ADM_AI
        AJ = adm.ADM_AJ
        AIJ = adm.ADM_AIJ
        TI = adm.ADM_TI
        TJ = adm.ADM_TJ
        W1 = gmtr.GMTR_t_W1    # 2
        W2 = gmtr.GMTR_t_W2    # 3
        W3 = gmtr.GMTR_t_W3    # 4
        HNX = gmtr.GMTR_a_HNX  # 0

        # Initialize arrays to zeros
        # Replace with actual dimensions
        self.OPRT_coef_div[:,:,:,:] = 0.0      #  np.zeros((dim1, dim2, dim3, dim4), dtype=rdtype)
        self.OPRT_coef_div_pl[:,:,:] = 0.0   #np.zeros((dim1, dim2, dim3), dtype=rdtype)
        
        for l in range(lall):
            for d in range(nxyz):
                #hn = d + HNX - 1
                #         0
                hn = d + HNX
                                # 1  to  16 (inner grid points)
                for i in range (gmin, gmax + 1):
                    for j in range(gmin, gmax + 1):
                    #for g in range(gmin, gmax + 1):
                    # ij     = g
                    # ip1j   = g + iall + 1
                    # ip1jp1 = g + iall + 1
                    # ijp1   = g + iall
                    # i-1, j   = g - 1
                    # i-1, jm1 = g - iall - 1
                    # ijm1   = g - iall

                    # ij
                        self.OPRT_coef_div[i, j, 0, d, l] = (
                            + gmtr.GMTR_t[i,   j  , k0, l, TI, W1] * gmtr.GMTR_a[i,   j  , k0, l, AI , hn]
                            + gmtr.GMTR_t[i,   j  , k0, l, TI, W1] * gmtr.GMTR_a[i,   j  , k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i,   j  , k0, l, TJ, W1] * gmtr.GMTR_a[i,   j  , k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i,   j  , k0, l, TJ, W1] * gmtr.GMTR_a[i,   j  , k0, l, AJ , hn]
                            + gmtr.GMTR_t[i-1, j  , k0, l, TI, W2] * gmtr.GMTR_a[i,   j  , k0, l, AJ , hn]
                            - gmtr.GMTR_t[i-1, j  , k0, l, TI, W2] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W3] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i,   j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i,   j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j  , k0, l, AI , hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ip1j
                        self.OPRT_coef_div[i, j, 1, d, l] = (
                            - gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j  , k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j  , k0, l, TI, W2] * gmtr.GMTR_a[i, j  , k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j  , k0, l, TI, W2] * gmtr.GMTR_a[i, j  , k0, l, AIJ, hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                    
                        # ip1jp1
                        self.OPRT_coef_div[i, j, 2, d, l] = (
                            + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ijp1
                        self.OPRT_coef_div[i, j, 3, d, l] = (
                            + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # im1j
                        self.OPRT_coef_div[i, j, 4, d, l] = (
                            + gmtr.GMTR_t[i-1, j  , k0, l, TI, W1] * gmtr.GMTR_a[i,   j  , k0, l, AJ , hn]
                            - gmtr.GMTR_t[i-1, j  , k0, l, TI, W1] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # i-1,  j-1
                        self.OPRT_coef_div[i, j, 5, d, l] = (
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W1] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ijm1
                        self.OPRT_coef_div[i, j, 6, d, l] = (
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W2] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i,   j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i,   j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i,   j  , k0, l, AI , hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                #with open(std.fname_log, 'a') as log_file:
                #    print(adm.ADM_have_sgp[l], 'TR', file=log_file)

                if adm.ADM_have_sgp[l]: 

                    # ij     = gmin
                    i = 1
                    j = 1
                    # ip1j   = gmin + 1
                    # ip1jp1 = gmin + iall + 1
                    # ijp1   = gmin + iall
                    # im1j   = gmin - 1
                    # im1jm1 = gmin - iall - 1
                    # ijm1   = gmin - iall

                    # ij
                    self.OPRT_coef_div[i, j, 0, d, l] = (
                        + gmtr.GMTR_t[i,   j  , k0, l, TI, W1] * gmtr.GMTR_a[i,   j  , k0, l, AI , hn]
                        + gmtr.GMTR_t[i,   j  , k0, l, TI, W1] * gmtr.GMTR_a[i,   j  , k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i,   j  , k0, l, TJ, W1] * gmtr.GMTR_a[i,   j  , k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i,   j  , k0, l, TJ, W1] * gmtr.GMTR_a[i,   j  , k0, l, AJ , hn]
                        + gmtr.GMTR_t[i-1, j  , k0, l, TI, W2] * gmtr.GMTR_a[i,   j  , k0, l, AJ , hn]
                        - gmtr.GMTR_t[i-1, j  , k0, l, TI, W2] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        - gmtr.GMTR_t[i,   j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i,   j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j  , k0, l, AI , hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ip1j
                    self.OPRT_coef_div[i, j, 1, d, l] = (
                        - gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i,   j  , k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j  , k0, l, TI, W2] * gmtr.GMTR_a[i,   j  , k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j  , k0, l, TI, W2] * gmtr.GMTR_a[i,   j  , k0, l, AIJ, hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ip1jp1
                    self.OPRT_coef_div[i, j, 2, d, l] = (
                        + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # i, jp1
                    self.OPRT_coef_div[i, j, 3, d, l] = (
                        + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , hn]
                        + gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , hn]
                        - gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # i-1, j
                    self.OPRT_coef_div[i, j, 4, d, l] = (
                        + gmtr.GMTR_t[i-1, j  , k0, l, TI, W1] * gmtr.GMTR_a[i,   j  , k0, l, AJ , hn]
                        - gmtr.GMTR_t[i-1, j  , k0, l, TI, W1] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # i-1, j-1, 
                    self.OPRT_coef_div[i, j, 5, d, l] = (
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # i, j-1, 
                    self.OPRT_coef_div[i, j, 6, d, l] = (
                        - gmtr.GMTR_t[i, j-1,   k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j-1,   k0, l, TJ, W1] * gmtr.GMTR_a[i,   j  , k0, l, AI , hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]


        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for d in range(adm.ADM_nxyz):
                    #hn = d + HNX - 1
                    hn = d + HNX

                    coef = 0.0
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij   = v
                        ijp1 = v + 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl

                        coef += (
                            gmtr.GMTR_t_pl[ij , k0, l, W1] * gmtr.GMTR_a_pl[ij  , k0, l, hn] +
                            gmtr.GMTR_t_pl[ij , k0, l, W1] * gmtr.GMTR_a_pl[ijp1, k0, l, hn]
                        )

                    self.OPRT_coef_div_pl[0, d, l] = coef * 0.5 * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]
                                        #1              
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                    #for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 2):
                        ij   = v
                        ijp1 = v + 1
                        ijm1 = v - 1

                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl
                        if ijm1 == adm.ADM_gmin_pl - 1:
                            ijm1 = adm.ADM_gmax_pl       #6 or should it be 5?

                        #self.OPRT_coef_div_pl[v - 1, d, l] = (
                        self.OPRT_coef_div_pl[v, d, l] = (
                            + gmtr.GMTR_t_pl[ijm1, k0, l, W3] * gmtr.GMTR_a_pl[ijm1, k0, l, hn]
                            + gmtr.GMTR_t_pl[ijm1, k0, l, W3] * gmtr.GMTR_a_pl[ij  , k0, l, hn]
                            + gmtr.GMTR_t_pl[ij  , k0, l, W2] * gmtr.GMTR_a_pl[ij  , k0, l, hn]
                            + gmtr.GMTR_t_pl[ij  , k0, l, W2] * gmtr.GMTR_a_pl[ijp1, k0, l, hn]
                        ) * 0.5 * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]

        return


    def OPRT_rotation_setup(self, gmtr, rdtype):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("*** setup coefficient of rotation operator", file=log_file)        
        #           1                    18               1
        #gmin = (adm.ADM_gmin - 1) * adm.ADM_gall_1d + adm.ADM_gmin
        #           16                   18               16
        #gmax = (adm.ADM_gmax - 1) * adm.ADM_gall_1d + adm.ADM_gmax
        gmin = adm.ADM_gmin #1
        gmax = adm.ADM_gmax #16
        iall = adm.ADM_gall_1d #18 
        gall = adm.ADM_gall
        nxyz = adm.ADM_nxyz  #3
        lall = adm.ADM_lall
        k0 = adm.ADM_K0
        P_RAREA = gmtr.GMTR_p_RAREA
        AI = adm.ADM_AI
        AJ = adm.ADM_AJ
        AIJ = adm.ADM_AIJ
        TI = adm.ADM_TI
        TJ = adm.ADM_TJ
        W1 = gmtr.GMTR_t_W1    # 2
        W2 = gmtr.GMTR_t_W2    # 3
        W3 = gmtr.GMTR_t_W3    # 4
        HTX = gmtr.GMTR_a_HTX  # 0

        self.OPRT_coef_rot[:,:,:,:] = 0.0      #  np.zeros((dim1, dim2, dim3, dim4), dtype=rdtype)
        self.OPRT_coef_rot_pl[:,:,:] = 0.0   #np.zeros((dim1, dim2, dim3), dtype=rdtype)
        
        for l in range(lall):
            for d in range(nxyz):
                #hn = d + HNX - 1
                #         0
                ht = d + HTX
                                # 1  to  16 (inner grid points)
                for i in range (gmin, gmax + 1):
                    for j in range(gmin, gmax + 1):

                        # ij
                        self.OPRT_coef_rot[i, j, 0, d, l] = (
                            + gmtr.GMTR_t[i,   j,   k0, l, TI, W1] * gmtr.GMTR_a[i,   j,   k0, l, AI , ht]
                            + gmtr.GMTR_t[i,   j,   k0, l, TI, W1] * gmtr.GMTR_a[i,   j,   k0, l, AIJ, ht]
                            + gmtr.GMTR_t[i,   j,   k0, l, TJ, W1] * gmtr.GMTR_a[i,   j,   k0, l, AIJ, ht]
                            + gmtr.GMTR_t[i,   j,   k0, l, TJ, W1] * gmtr.GMTR_a[i,   j,   k0, l, AJ , ht]
                            + gmtr.GMTR_t[i-1, j,   k0, l, TI, W2] * gmtr.GMTR_a[i,   j,   k0, l, AJ , ht]
                            - gmtr.GMTR_t[i-1, j,   k0, l, TI, W2] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W3] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , ht]
                            - gmtr.GMTR_t[i,   j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , ht]
                            + gmtr.GMTR_t[i,   j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j,   k0, l, AI , ht]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ip1j
                        self.OPRT_coef_rot[i, j, 1, d, l] = (
                            - gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j-1, k0, l, AJ , ht]
                            + gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j,   k0, l, AI , ht]
                            + gmtr.GMTR_t[i, j,   k0, l, TI, W2] * gmtr.GMTR_a[i, j,   k0, l, AI , ht]
                            + gmtr.GMTR_t[i, j,   k0, l, TI, W2] * gmtr.GMTR_a[i, j,   k0, l, AIJ, ht]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ip1jp1
                        self.OPRT_coef_rot[i, j, 2, d, l] = (
                            + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AI , ht]
                            + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, ht]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, ht]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , ht]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ijp1
                        self.OPRT_coef_rot[i, j, 3, d, l] = (
                            + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AIJ, ht]
                            + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , ht]
                            + gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , ht]
                            - gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , ht]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # im1j
                        self.OPRT_coef_rot[i, j, 4, d, l] = (
                            + gmtr.GMTR_t[i-1, j,   k0, l, TI, W1] * gmtr.GMTR_a[i,   j,   k0, l, AJ , ht]
                            - gmtr.GMTR_t[i-1, j,   k0, l, TI, W1] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # im1jm1
                        self.OPRT_coef_rot[i, j, 5, d, l] = (
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W1] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , ht]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ijm1
                        self.OPRT_coef_rot[i, j, 6, d, l] = (
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W2] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , ht]
                            - gmtr.GMTR_t[i,   j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , ht]
                            + gmtr.GMTR_t[i,   j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i,   j,   k0, l, AI , ht]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                if adm.ADM_have_sgp[l]: # pentagon
                    # ij     = gmin
                    i = 1
                    j = 1
                    #print("TRTRTRTR, prc, l, reg:", prc.prc_myrank, l, adm.RGNMNG_lp2r[l, prc.prc_myrank])
                    # ij
                    self.OPRT_coef_rot[i, j, 0, d, l] = (
                        + gmtr.GMTR_t[i,   j,   k0, l, TI, W1] * gmtr.GMTR_a[i,   j,   k0, l, AI , ht]
                        + gmtr.GMTR_t[i,   j,   k0, l, TI, W1] * gmtr.GMTR_a[i,   j,   k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i,   j,   k0, l, TJ, W1] * gmtr.GMTR_a[i,   j,   k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i,   j,   k0, l, TJ, W1] * gmtr.GMTR_a[i,   j,   k0, l, AJ , ht]
                        + gmtr.GMTR_t[i-1, j,   k0, l, TI, W2] * gmtr.GMTR_a[i,   j,   k0, l, AJ , ht]
                        - gmtr.GMTR_t[i-1, j,   k0, l, TI, W2] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                        - gmtr.GMTR_t[i,   j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i,   j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j,   k0, l, AI , ht]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ip1j
                    self.OPRT_coef_rot[i, j, 1, d, l] = (
                        - gmtr.GMTR_t[i,  j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i,  j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i,   j,   k0, l, AI , ht]
                        + gmtr.GMTR_t[i,  j  , k0, l, TI, W2] * gmtr.GMTR_a[i,   j,   k0, l, AI , ht]
                        + gmtr.GMTR_t[i,  j  , k0, l, TI, W2] * gmtr.GMTR_a[i,   j,   k0, l, AIJ, ht]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ip1jp1
                    self.OPRT_coef_rot[i, j, 2, d, l] = (
                        + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AI , ht]
                        + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , ht]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ijp1
                    self.OPRT_coef_rot[i, j, 3, d, l] = (
                        + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , ht]
                        + gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , ht]
                        - gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , ht]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # im1j
                    self.OPRT_coef_rot[i, j, 4, d, l] = (
                        + gmtr.GMTR_t[i-1, j,   k0, l, TI, W1] * gmtr.GMTR_a[i,   j,   k0, l, AJ , ht]
                        - gmtr.GMTR_t[i-1, j,   k0, l, TI, W1] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # im1jm1
                    self.OPRT_coef_rot[i, j, 5, d, l] = (
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ijm1
                    self.OPRT_coef_rot[i, j, 6, d, l] = (
                        - gmtr.GMTR_t[i, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i,   j  , k0, l, AI , ht]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for d in range(adm.ADM_nxyz):
                    #hn = d + HNX - 1
                    ht = d + HTX

                    coef = 0.0
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij   = v
                        ijp1 = v + 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl

                        coef += (
                            gmtr.GMTR_t_pl[ij , k0, l, W1] * gmtr.GMTR_a_pl[ij  , k0, l, ht] +
                            gmtr.GMTR_t_pl[ij , k0, l, W1] * gmtr.GMTR_a_pl[ijp1, k0, l, ht]
                        )

                    self.OPRT_coef_rot_pl[0, d, l] = coef * 0.5 * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij   = v
                        ijp1 = v + 1
                        ijm1 = v - 1

                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl
                        if ijm1 == adm.ADM_gmin_pl - 1:
                            ijm1 = adm.ADM_gmax_pl

                        self.OPRT_coef_rot_pl[v, d, l] = (
                            + gmtr.GMTR_t_pl[ijm1, k0, l, W3] * gmtr.GMTR_a_pl[ijm1, k0, l, ht]
                            + gmtr.GMTR_t_pl[ijm1, k0, l, W3] * gmtr.GMTR_a_pl[ij  , k0, l, ht]
                            + gmtr.GMTR_t_pl[ij  , k0, l, W2] * gmtr.GMTR_a_pl[ij  , k0, l, ht]
                            + gmtr.GMTR_t_pl[ij  , k0, l, W2] * gmtr.GMTR_a_pl[ijp1, k0, l, ht]
                        ) * 0.5 * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]

        return


    def OPRT_gradient_setup(self, gmtr, rdtype):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("*** setup coefficient of gradient operator", file=log_file)        
        #           1                    18               1
        #gmin = (adm.ADM_gmin - 1) * adm.ADM_gall_1d + adm.ADM_gmin
        #           16                   18               16
        #gmax = (adm.ADM_gmax - 1) * adm.ADM_gall_1d + adm.ADM_gmax
        gmin = adm.ADM_gmin #1
        gmax = adm.ADM_gmax #16
        iall = adm.ADM_gall_1d #18 
        gall = adm.ADM_gall
        nxyz = adm.ADM_nxyz  #3
        lall = adm.ADM_lall
        k0 = adm.ADM_K0
        P_RAREA = gmtr.GMTR_p_RAREA
        AI = adm.ADM_AI
        AJ = adm.ADM_AJ
        AIJ = adm.ADM_AIJ
        TI = adm.ADM_TI
        TJ = adm.ADM_TJ
        W1 = gmtr.GMTR_t_W1    # 2
        W2 = gmtr.GMTR_t_W2    # 3
        W3 = gmtr.GMTR_t_W3    # 4
        HNX = gmtr.GMTR_a_HNX  # 0

        # Initialize arrays to zeros
        # Replace with actual dimensions
        self.OPRT_coef_grad[:,:,:,:] = 0.0      #  np.zeros((dim1, dim2, dim3, dim4), dtype=rdtype)
        self.OPRT_coef_grad_pl[:,:,:] = 0.0   #np.zeros((dim1, dim2, dim3), dtype=rdtype)
        
        for l in range(lall):
            for d in range(nxyz):
                #hn = d + HNX - 1
                #         0
                hn = d + HNX
                                # 1  to  16 (inner grid points)
                for i in range (gmin, gmax + 1):
                    for j in range(gmin, gmax + 1):

                        # ij
                        self.OPRT_coef_grad[i, j, 0, d, l] = (
                            + gmtr.GMTR_t[i, j, k0, l, TI, W1] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j, k0, l, TI, W1] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W1] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W1] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i-1, j, k0, l, TI, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i-1, j, k0, l, TI, W2] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W3] * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i, j, k0, l, AI , hn]

                            - 2.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            - 2.0 * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                            + 2.0 * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                            + 2.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            + 2.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                            - 2.0 * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ip1j
                        self.OPRT_coef_grad[i, j, 1, d, l] = (
                            - gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j, k0, l, TI, W2] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j, k0, l, TI, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ip1jp1
                        self.OPRT_coef_grad[i, j, 2, d, l] = (
                            + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ijp1
                        self.OPRT_coef_grad[i, j, 3, d, l] = (
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W3] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # im1j
                        self.OPRT_coef_grad[i, j, 4, d, l] = (
                            + gmtr.GMTR_t[i-1, j, k0, l, TI, W1] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i-1, j, k0, l, TI, W1] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # im1jm1
                        self.OPRT_coef_grad[i, j, 5, d, l] = (
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W1] * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ijm1
                        self.OPRT_coef_grad[i, j, 6, d, l] = (
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W2] * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]



                if adm.ADM_have_sgp[l]: # pentagon
                    # ij     = gmin
                    i = 1
                    j = 1

 
                    # i, j
                    self.OPRT_coef_grad[i, j, 0, d, l] = (
                        + gmtr.GMTR_t[i, j, k0, l, TI, W1] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j, k0, l, TI, W1] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W1] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W1] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        + gmtr.GMTR_t[i-1, j, k0, l, TI, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        - gmtr.GMTR_t[i-1, j, k0, l, TI, W2] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        - gmtr.GMTR_t[i, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        - 2.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        - 2.0 * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        + 2.0 * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        + 2.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        - 2.0 * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ip1j
                    self.OPRT_coef_grad[i, j, 1, d, l] = (
                        - gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j, k0, l, TI, W2] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j, k0, l, TI, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ip1jp1
                    self.OPRT_coef_grad[i, j, 2, d, l] = (
                        + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ijp1
                    self.OPRT_coef_grad[i, j, 3, d, l] = (
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W3] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        + gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        - gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # im1j
                    self.OPRT_coef_grad[i, j, 4, d, l] = (
                        + gmtr.GMTR_t[i-1, j, k0, l, TI, W1] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        - gmtr.GMTR_t[i-1, j, k0, l, TI, W1] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # im1jm1
                    self.OPRT_coef_grad[i, j, 5, d, l] = (
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ijm1
                    self.OPRT_coef_grad[i, j, 6, d, l] = (
                        - gmtr.GMTR_t[i, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                    ) * 0.5 * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for d in range(adm.ADM_nxyz):
                    #hn = d + HNX - 1
                    hn = d + HNX

                    coef = 0.0
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij   = v
                        ijp1 = v + 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl

                        coef += 2.0 * (gmtr.GMTR_t_pl[ij, k0, l, W1] - 1.0) * gmtr.GMTR_a_pl[ijp1, k0, l, hn]

                    self.OPRT_coef_grad_pl[0, d, l] = coef * 0.5 * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij   = v
                        ijp1 = v + 1
                        ijm1 = v - 1

                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl
                        if ijm1 == adm.ADM_gmin_pl - 1:
                            ijm1 = adm.ADM_gmax_pl

                        self.OPRT_coef_grad_pl[v, d, l] = (
                            + gmtr.GMTR_t_pl[ijm1, k0, l, W3] * gmtr.GMTR_a_pl[ijm1, k0, l, hn]
                            + gmtr.GMTR_t_pl[ijm1, k0, l, W3] * gmtr.GMTR_a_pl[ij   , k0, l, hn]
                            + gmtr.GMTR_t_pl[ij   , k0, l, W2] * gmtr.GMTR_a_pl[ij   , k0, l, hn]
                            + gmtr.GMTR_t_pl[ij   , k0, l, W2] * gmtr.GMTR_a_pl[ijp1, k0, l, hn]
                        ) * 0.5 * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]

        return


    def OPRT_laplacian_setup(self, gmtr, rdtype):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("*** setup coefficient of laplacian operator", file=log_file)        
        #           1                    18               1
        #gmin = (adm.ADM_gmin - 1) * adm.ADM_gall_1d + adm.ADM_gmin
        #           16                   18               16
        #gmax = (adm.ADM_gmax - 1) * adm.ADM_gall_1d + adm.ADM_gmax
        gmin = adm.ADM_gmin #1
        gmax = adm.ADM_gmax #16
        iall = adm.ADM_gall_1d #18 
        gall = adm.ADM_gall
        nxyz = adm.ADM_nxyz  #3
        lall = adm.ADM_lall
        k0 = adm.ADM_K0
        P_RAREA = gmtr.GMTR_p_RAREA
        T_RAREA = gmtr.GMTR_t_RAREA
        AI = adm.ADM_AI
        AJ = adm.ADM_AJ
        AIJ = adm.ADM_AIJ
        TI = adm.ADM_TI
        TJ = adm.ADM_TJ
        W1 = gmtr.GMTR_t_W1    # 2
        W2 = gmtr.GMTR_t_W2    # 3
        W3 = gmtr.GMTR_t_W3    # 4
        HNX = gmtr.GMTR_a_HNX  # 0
        TNX = gmtr.GMTR_a_TNX  
        TN2X = gmtr.GMTR_a_TN2X  

        self.OPRT_coef_lap[:,:,:,:] = 0.0      #  np.zeros((dim1, dim2, dim3, dim4), dtype=rdtype)
        self.OPRT_coef_lap_pl[:,:] = 0.0   #np.zeros((dim1, dim2, dim3), dtype=rdtype)
        
        for l in range(lall):
            for d in range(nxyz):

                hn = d + HNX
                tn = d + TNX
                                # 1  to  16 (inner grid points)
                for i in range (gmin, gmax + 1):
                    for j in range(gmin, gmax + 1):

                        # coef_lap[i, j, 0, l]
                        self.OPRT_coef_lap[i, j, 0, l] += gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA] * (
                            - 1.0 * gmtr.GMTR_a[i,   j, k0, l, AI,  tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            + 2.0 * gmtr.GMTR_a[i+1, j, k0, l, AJ,  tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            + 1.0 * gmtr.GMTR_a[i,   j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            - 1.0 * gmtr.GMTR_a[i,   j, k0, l, AI,  tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + 2.0 * gmtr.GMTR_a[i+1, j, k0, l, AJ,  tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + 1.0 * gmtr.GMTR_a[i,   j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, 0, l] += gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA] * (
                            - 1.0 * gmtr.GMTR_a[i, j,   k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            - 2.0 * gmtr.GMTR_a[i, j+1, k0, l, AI,  tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + 1.0 * gmtr.GMTR_a[i, j,   k0, l, AJ,  tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            - 1.0 * gmtr.GMTR_a[i, j,   k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            - 2.0 * gmtr.GMTR_a[i, j+1, k0, l, AI,  tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            + 1.0 * gmtr.GMTR_a[i, j,   k0, l, AJ,  tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, 0, l] += gmtr.GMTR_t[i-1, j, k0, l, TI, T_RAREA] * (
                            - 1.0 * gmtr.GMTR_a[i,   j, k0, l, AJ,  tn] * gmtr.GMTR_a[i,   j, k0, l, AJ, hn]
                            - 2.0 * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i,   j, k0, l, AJ, hn]
                            - 1.0 * gmtr.GMTR_a[i-1, j, k0, l, AI,  tn] * gmtr.GMTR_a[i,   j, k0, l, AJ, hn]
                            + 1.0 * gmtr.GMTR_a[i,   j, k0, l, AJ,  tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            + 2.0 * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            + 1.0 * gmtr.GMTR_a[i-1, j, k0, l, AI,  tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        )

                        self.OPRT_coef_lap[i, j, 0, l] += gmtr.GMTR_t[i-1, j-1, k0, l, TJ, T_RAREA] * (
                            -1.0 * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            + 2.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            + 1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            - 1.0 * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            + 2.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            + 1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, 0, l] += gmtr.GMTR_t[i-1, j-1, k0, l, TI, T_RAREA] * (
                            -1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - 2.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            + 1.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - 1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            - 2.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            + 1.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, 0, l] += gmtr.GMTR_t[i, j-1, k0, l, TJ, T_RAREA] * (
                            -1.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            - 2.0 * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            - 1.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            + 1.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            + 2.0 * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            + 1.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        )

                        # coef_lap[i, j, 1, l]
                        self.OPRT_coef_lap[i, j, 1, l] += gmtr.GMTR_t[i, j-1, k0, l, TJ, T_RAREA] * (
                            -1.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            + 2.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            + 1.0 * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            + 1.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            - 2.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            - 1.0 * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        )

                        # coef_lap[i, j, 1, l] (continued)
                        self.OPRT_coef_lap[i, j, 1, l] += gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA] * (
                            -1.0 * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            -2.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            -1.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            -1.0 * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            -2.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            -1.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        )

                        # coef_lap[i, j, 2, l]
                        self.OPRT_coef_lap[i, j, 2, l] += gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA] * (
                            +1.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            +2.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            -1.0 * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            +1.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            +2.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            -1.0 * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, 2, l] += gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA] * (
                            +1.0 * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            -2.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            -1.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            +1.0 * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            -1.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            -2.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        )

                        # coef_lap[i, j, 3, l]
                        self.OPRT_coef_lap[i, j, 3, l] += gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA] * (
                            +1.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            +2.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            +1.0 * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            +1.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            +2.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            +1.0 * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, 3, l] += gmtr.GMTR_t[i-1, j, k0, l, TI, T_RAREA] * (
                            +1.0 * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            +2.0 * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            -1.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            -1.0 * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -2.0 * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            +1.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        )

                        # coef_lap[i, j, 4, l]
                        self.OPRT_coef_lap[i, j, 4, l] += gmtr.GMTR_t[i-1, j, k0, l, TI, T_RAREA] * (
                            -1.0 * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            +2.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            +1.0 * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            +1.0 * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -2.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -1.0 * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        )


                        # coef_lap[i, j, 4, l] (continued)
                        self.OPRT_coef_lap[i, j, 4, l] += gmtr.GMTR_t[i-1, j-1, k0, l, TJ, T_RAREA] * (
                            -1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -2.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -1.0 * gmtr.GMTR_a[i-1, j,   k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            -2.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            -1.0 * gmtr.GMTR_a[i-1, j,   k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        )

                        # coef_lap[i, j, 5, l]
                        self.OPRT_coef_lap[i, j, 5, l] += gmtr.GMTR_t[i-1, j-1, k0, l, TJ, T_RAREA] * (
                            +1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            +2.0 * gmtr.GMTR_a[i-1, j,   k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            +1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            +2.0 * gmtr.GMTR_a[i-1, j,   k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            -1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, 5, l] += gmtr.GMTR_t[i-1, j-1, k0, l, TI, T_RAREA] * (
                            +1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            -2.0 * gmtr.GMTR_a[i, j-1,   k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            -1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            +1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            -2.0 * gmtr.GMTR_a[i, j-1,   k0, l, AJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            -1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                        )

                        # if i == 6 and j == 5 and l== 3 :
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("i = 6, j = 5, l = 3,  v6-0: ", d, file=log_file)
                        #         print(self.OPRT_coef_lap[i, j, 6, l], file=log_file)
                        # coef_lap[i, j, 6, l]
                        self.OPRT_coef_lap[i, j, 6, l] += gmtr.GMTR_t[i-1, j-1, k0, l, TI, T_RAREA] * (
                            +1.0 * gmtr.GMTR_a[i, j-1,   k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            +2.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            +1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            +1.0 * gmtr.GMTR_a[i, j-1,   k0, l, AJ, tn] * gmtr.GMTR_a[i, j-1,   k0, l, AJ, hn]
                            +2.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j-1,   k0, l, AJ, hn]
                            +1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i, j-1,   k0, l, AJ, hn]
                        )

                        # if i == 6 and j == 5 and l== 3 :
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("i = 6, j = 5, l = 3,  v6-1: ", d, file=log_file)
                        #         print(gmtr.GMTR_t[i-1, j-1, k0, l, TI, T_RAREA], file=log_file)
                        #         print(gmtr.GMTR_a[i, j-1, k0, l, AJ, tn], gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn], file=log_file)
                        #         print(gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn], gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn], file=log_file)
                        #         print(gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn], gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn], file=log_file)
                        #         print(gmtr.GMTR_a[i, j-1, k0, l, AJ, tn], gmtr.GMTR_a[i, j-1, k0, l, AJ, hn], file=log_file)
                        #         print(gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn], gmtr.GMTR_a[i, j-1, k0, l, AJ, hn], file=log_file)
                        #         print(gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i, j-1,   k0, l, AJ, hn])
                        #         print("coef lap=", self.OPRT_coef_lap[i, j, 6, l], file=log_file)



                        self.OPRT_coef_lap[i, j, 6, l] += gmtr.GMTR_t[i, j-1, k0, l, TJ, T_RAREA] * (
                            +1.0 * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            +2.0 * gmtr.GMTR_a[i, j,   k0, l, AI, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            -1.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            -1.0 * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j,   k0, l, AI, hn]
                            -2.0 * gmtr.GMTR_a[i, j,   k0, l, AI, tn] * gmtr.GMTR_a[i, j,   k0, l, AI, hn]
                            +1.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j,   k0, l, AI, hn]
                        )

            if adm.ADM_have_sgp[l]: # pentagon
                # ij     = gmin
                i = 1
                j = 1

                self.OPRT_coef_lap[i, j, 0, l] = 0.0
                self.OPRT_coef_lap[i, j, 1, l] = 0.0
                self.OPRT_coef_lap[i, j, 2, l] = 0.0
                self.OPRT_coef_lap[i, j, 3, l] = 0.0
                self.OPRT_coef_lap[i, j, 4, l] = 0.0
                self.OPRT_coef_lap[i, j, 5, l] = 0.0
                self.OPRT_coef_lap[i, j, 6, l] = 0.0

                for d in range(nxyz):
                    hn = d + HNX
                    tn = d + TNX
                
                    # (i, j)
                    self.OPRT_coef_lap[i, j, 0, l] += gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA] * (
                        -1.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +2.0 * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +1.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        -1.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +2.0 * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +1.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                    )

                    self.OPRT_coef_lap[i, j, 0, l] += gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA] * (
                        -1.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -2.0 * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +1.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -1.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -2.0 * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        +1.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                    )

                    self.OPRT_coef_lap[i, j, 0, l] += gmtr.GMTR_t[i-1, j, k0, l, TI, T_RAREA] * (
                        -1.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -2.0 * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -1.0 * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        +1.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        +2.0 * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        +1.0 * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                    )

                    self.OPRT_coef_lap[i, j, 0, l] += gmtr.GMTR_t[i-1, j-1, k0, l, TJ, T_RAREA] * (
                        -1.0 * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        +2.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        +1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        -1.0 * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        +2.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        +1.0 * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                    )

                    self.OPRT_coef_lap[i, j, 0, l] += gmtr.GMTR_t[i, j-1, k0, l, TJ, T_RAREA] * (
                        -1.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        -2.0 * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        -1.0 * gmtr.GMTR_a[i, j,   k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        +1.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +2.0 * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +1.0 * gmtr.GMTR_a[i, j,   k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                    )

                    # ip1j
                    self.OPRT_coef_lap[i, j, 1, l] += gmtr.GMTR_t[i, j-1, k0, l, TJ, T_RAREA] * (
                        +1.0 * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        +2.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        -1.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        -1.0 * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        -2.0 * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +1.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                    )

                    self.OPRT_coef_lap[i, j, 1, l] += gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA] * (
                        -1.0 * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        -2.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        -1.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        -1.0 * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -2.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -1.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                    )

                    # ip1jp1
                    self.OPRT_coef_lap[i, j, 2, l] += gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA] * (
                        +1.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +2.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        -1.0 * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +1.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +2.0 * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -1.0 * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                    )

                    self.OPRT_coef_lap[i, j, 2, l] += gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA] * (
                        +1.0 * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -2.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -1.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +1.0 * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -2.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -1.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                    )

                    # ijp1
                    self.OPRT_coef_lap[i, j, 3, l] += gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA] * (
                        +1.0 * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +2.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +1.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +1.0 * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        +2.0 * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        +1.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                    )

                    self.OPRT_coef_lap[i, j, 3, l] += gmtr.GMTR_t[i-1, j, k0, l, TI, T_RAREA] * (
                        +1.0 * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        +2.0 * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -1.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -1.0 * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        -2.0 * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        +1.0 * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                    )

                    # im1j
                    self.OPRT_coef_lap[i, j, 4, l] += gmtr.GMTR_t[i-1,j,k0,l,TI,T_RAREA] * ( 
                        + 1.0 * gmtr.GMTR_a[i-1,j,k0,l,AIJ,tn] * gmtr.GMTR_a[i,j,k0,l,AJ,hn]
                        + 2.0 * gmtr.GMTR_a[i,j,k0,l,AJ,tn] * gmtr.GMTR_a[i,j,k0,l,AJ,hn]
                        - 1.0 * gmtr.GMTR_a[i-1,j,k0,l,AI,tn] * gmtr.GMTR_a[i,j,k0,l,AJ,hn]
                        - 1.0 * gmtr.GMTR_a[i-1,j,k0,l,AIJ,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn]
                        - 2.0 * gmtr.GMTR_a[i,j,k0,l,AJ,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn]
                        + 1.0 * gmtr.GMTR_a[i-1,j,k0,l,AI,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                    )

                    self.OPRT_coef_lap[i, j, 4, l] += gmtr.GMTR_t[i-1,j-1,k0,l,TJ,T_RAREA] * (
                        - 1.0 * gmtr.GMTR_a[i-1,j-1,k0,l,AJ,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                        - 2.0 * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                        - 1.0 * gmtr.GMTR_a[i-1,j,k0,l,AI,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                        - 1.0 * gmtr.GMTR_a[i-1,j-1,k0,l,AJ,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        - 2.0 * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        - 1.0 * gmtr.GMTR_a[i-1,j,k0,l,AI,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                    )

                    # im1jm1
                    self.OPRT_coef_lap[i, j, 5, l] += gmtr.GMTR_t[i-1,j-1,k0,l,TJ,T_RAREA] * ( 
                        - 1.0 * gmtr.GMTR_a[i-1,j-1,k0,l,AJ,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                        + 2.0 * gmtr.GMTR_a[i-1,j,k0,l,AI,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                        + 1.0 * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                        - 1.0 * gmtr.GMTR_a[i-1,j-1,k0,l,AJ,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        + 2.0 * gmtr.GMTR_a[i-1,j,k0,l,AI,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        + 1.0 * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                    )

                    # ijm1
                    self.OPRT_coef_lap[i, j, 6, l] += gmtr.GMTR_t[i,j-1,k0,l,TJ,T_RAREA] * (
                        + 1.0 * gmtr.GMTR_a[i,j-1,k0,l,AIJ,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        + 2.0 * gmtr.GMTR_a[i,j,k0,l,AI,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        - 1.0 * gmtr.GMTR_a[i,j-1,k0,l,AJ,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        - 1.0 * gmtr.GMTR_a[i,j-1,k0,l,AIJ,tn] * gmtr.GMTR_a[i,j,k0,l,AI,hn] 
                        - 2.0 * gmtr.GMTR_a[i,j,k0,l,AI,tn] * gmtr.GMTR_a[i,j,k0,l,AI,hn] 
                        + 1.0 * gmtr.GMTR_a[i,j-1,k0,l,AJ,tn] * gmtr.GMTR_a[i,j,k0,l,AI,hn] 
                    )

            for i in range(adm.ADM_gall_1d):
                for j in range(adm.ADM_gall_1d):
                    self.OPRT_coef_lap[i, j, 0, l] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / 12.0
                    self.OPRT_coef_lap[i, j, 1, l] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / 12.0
                    self.OPRT_coef_lap[i, j, 2, l] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / 12.0
                    self.OPRT_coef_lap[i, j, 3, l] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / 12.0
                    self.OPRT_coef_lap[i, j, 4, l] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / 12.0
                    self.OPRT_coef_lap[i, j, 5, l] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / 12.0
                    self.OPRT_coef_lap[i, j, 6, l] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / 12.0


        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl  # 0, index for pole point

            for l in range(adm.ADM_lall_pl):
                for d in range(adm.ADM_nxyz):
                    hn  = d + HNX 
                    tn  = d + TNX 
                    tn2 = d + TN2X 

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij = v
                        ijp1 = v + 1
                        ijm1 = v - 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl
                        if ijm1 == adm.ADM_gmin_pl - 1:
                            ijm1 = adm.ADM_gmax_pl

                        # with open(std.fname_log, 'a') as log_file:
                        #     print("coef_lap_pl, v0-0: d and l = ", d, l, file= log_file)
                        #     print(self.OPRT_coef_lap_pl[0, l], file=log_file)

                        self.OPRT_coef_lap_pl[0, l] += gmtr.GMTR_t_pl[ijm1, k0, l, T_RAREA] * (
                            + 1.0 * gmtr.GMTR_a_pl[ijm1, k0, l, tn]  * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            - 2.0 * gmtr.GMTR_a_pl[ijm1, k0, l, tn2] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            - 1.0 * gmtr.GMTR_a_pl[ij,   k0, l, tn]  * gmtr.GMTR_a_pl[ij, k0, l, hn]
                        )

                        self.OPRT_coef_lap_pl[0, l] += gmtr.GMTR_t_pl[ij, k0, l, T_RAREA] * (
                            + 1.0 * gmtr.GMTR_a_pl[ij,   k0, l, tn]  * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            - 2.0 * gmtr.GMTR_a_pl[ij,   k0, l, tn2] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            - 1.0 * gmtr.GMTR_a_pl[ijp1, k0, l, tn]  * gmtr.GMTR_a_pl[ij, k0, l, hn]
                        )

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij = v
                        ijp1 = v + 1
                        ijm1 = v - 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl
                        if ijm1 == adm.ADM_gmin_pl - 1:
                            ijm1 = adm.ADM_gmax_pl

                        self.OPRT_coef_lap_pl[v, l] += gmtr.GMTR_t_pl[ijm1, k0, l, T_RAREA] * (
                            - 2.0 * gmtr.GMTR_a_pl[ijm1, k0, l, tn] * gmtr.GMTR_a_pl[ijm1, k0, l, hn]
                            + 1.0 * gmtr.GMTR_a_pl[ijm1, k0, l, tn2] * gmtr.GMTR_a_pl[ijm1, k0, l, hn]
                            - 1.0 * gmtr.GMTR_a_pl[ij,   k0, l, tn] * gmtr.GMTR_a_pl[ijm1, k0, l, hn]
                            - 2.0 * gmtr.GMTR_a_pl[ijm1, k0, l, tn] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            + 1.0 * gmtr.GMTR_a_pl[ijm1, k0, l, tn2] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            - 1.0 * gmtr.GMTR_a_pl[ij,   k0, l, tn] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                        )

                        self.OPRT_coef_lap_pl[v, l] += gmtr.GMTR_t_pl[ij, k0, l, T_RAREA] * (
                            + 1.0 * gmtr.GMTR_a_pl[ij,   k0, l, tn] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            + 1.0 * gmtr.GMTR_a_pl[ij,   k0, l, tn2] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            + 2.0 * gmtr.GMTR_a_pl[ijp1, k0, l, tn] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            + 1.0 * gmtr.GMTR_a_pl[ij,   k0, l, tn] * gmtr.GMTR_a_pl[ijp1, k0, l, hn]
                            + 1.0 * gmtr.GMTR_a_pl[ij,   k0, l, tn2] * gmtr.GMTR_a_pl[ijp1, k0, l, hn]
                            + 2.0 * gmtr.GMTR_a_pl[ijp1, k0, l, tn] * gmtr.GMTR_a_pl[ijp1, k0, l, hn]
                        )

                for v in range(adm.ADM_gslf_pl, adm.ADM_gmax_pl + 1):
                    self.OPRT_coef_lap_pl[v, l] *= gmtr.GMTR_p_pl[n, k0, l, P_RAREA] / 12.0

        return
    

    def OPRT_diffusion_setup(self, gmtr, rdtype):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("*** setup coefficient of divergence operator", file=log_file)        
        #           1                    18               1
        #gmin = (adm.ADM_gmin - 1) * adm.ADM_gall_1d + adm.ADM_gmin
        #           16                   18               16
        #gmax = (adm.ADM_gmax - 1) * adm.ADM_gall_1d + adm.ADM_gmax
        gmin = adm.ADM_gmin #1
        gmax = adm.ADM_gmax #16
        iall = adm.ADM_gall_1d #18 
        gall = adm.ADM_gall
        nxyz = adm.ADM_nxyz  #3
        lall = adm.ADM_lall
        k0 = adm.ADM_K0
        P_RAREA = gmtr.GMTR_p_RAREA
        T_RAREA = gmtr.GMTR_t_RAREA
        AI = adm.ADM_AI
        AJ = adm.ADM_AJ
        AIJ = adm.ADM_AIJ
        TI = adm.ADM_TI
        TJ = adm.ADM_TJ
        W1 = gmtr.GMTR_t_W1    # 2
        W2 = gmtr.GMTR_t_W2    # 3
        W3 = gmtr.GMTR_t_W3    # 4
        HNX = gmtr.GMTR_a_HNX  # 0
        TNX = gmtr.GMTR_a_TNX
        TN2X = gmtr.GMTR_a_TN2X

        self.OPRT_coef_intp   [:,:,:,:,:,:] = 0.0
        self.OPRT_coef_diff   [:,:,:,:,:]   = 0.0
        self.OPRT_coef_intp_pl[:,:,:,:]     = 0.0
        self.OPRT_coef_diff_pl[:,:,:]       = 0.0

        for l in range(lall):
            for d in range(nxyz):

                tn = d + TNX
                                # 0  to  16 (expanded grid points)
                for i in range (gmin-1, gmax + 1):
                    for j in range(gmin-1, gmax + 1):

                        self.OPRT_coef_intp[i, j, 0, d, TI, l] = (
                            + gmtr.GMTR_a[i, j, k0, l, AIJ, tn] - gmtr.GMTR_a[i, j, k0, l, AI, tn]
                        ) * 0.5 * gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA]

                        self.OPRT_coef_intp[i, j, 1, d, TI, l] = (
                            - gmtr.GMTR_a[i, j, k0, l, AI, tn] - gmtr.GMTR_a[i + 1, j, k0, l, AJ, tn]
                        ) * 0.5 * gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA]

                        self.OPRT_coef_intp[i, j, 2, d, TI, l] = (
                            - gmtr.GMTR_a[i + 1, j, k0, l, AJ, tn] + gmtr.GMTR_a[i, j, k0, l, AIJ, tn]
                        ) * 0.5 * gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA]


                        self.OPRT_coef_intp[i, j, 0, d, TJ, l] = (
                            + gmtr.GMTR_a[i, j, k0, l, AJ, tn] - gmtr.GMTR_a[i, j, k0, l, AIJ, tn]
                        ) * 0.5 * gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA]

                        self.OPRT_coef_intp[i, j, 1, d, TJ, l] = (
                            - gmtr.GMTR_a[i, j, k0, l, AIJ, tn] + gmtr.GMTR_a[i, j + 1, k0, l, AI, tn]
                        ) * 0.5 * gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA]

                        self.OPRT_coef_intp[i, j, 2, d, TJ, l] = (
                            + gmtr.GMTR_a[i, j + 1, k0, l, AI, tn] + gmtr.GMTR_a[i, j, k0, l, AJ, tn]
                        ) * 0.5 * gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA]

        for l in range(lall):
            for d in range(nxyz):

                hn = d + HNX

                                # 1  to  16 (inner grid points)
                for i in range (gmin, gmax + 1):
                    for j in range(gmin, gmax + 1):

                        self.OPRT_coef_diff[i, j, 0, d, l] = (
                            + gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            * 0.5
                            * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                        )

                        self.OPRT_coef_diff[i, j, 1, d, l] = (
                            + gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            * 0.5
                            * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                        )

                        self.OPRT_coef_diff[i, j, 2, d, l] = (
                            - gmtr.GMTR_a[i - 1, j, k0, l, AI, hn]
                            * 0.5
                            * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                        )

                        self.OPRT_coef_diff[i, j, 3, d, l] = (
                            - gmtr.GMTR_a[i - 1, j - 1, k0, l, AIJ, hn]
                            * 0.5
                            * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                        )

                        self.OPRT_coef_diff[i, j, 4, d, l] = (
                            - gmtr.GMTR_a[i, j - 1, k0, l, AJ, hn]
                            * 0.5
                            * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                        )

                        self.OPRT_coef_diff[i, j, 5, d, l] = (
                            + gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            * 0.5
                            * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                        )

                if adm.ADM_have_sgp[l]:
                    #self.OPRT_coef_diff[1, 1, 5, d, l] = 0.0   # this might be correct, overwriting the last (6th) value with zero
                    self.OPRT_coef_diff[1, 1, 4, d, l] = 0.0    # this matches the original code, but could it be a bug?

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for d in range(adm.ADM_nxyz):
                    hn  = d + HNX 
                    tn  = d + TNX 
                    tn2 = d + TN2X

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij   = v
                        ijp1 = v + 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl

                        self.OPRT_coef_intp_pl[v, 0, d, l] = -gmtr.GMTR_a_pl[ijp1, k0, l, tn] + gmtr.GMTR_a_pl[ij, k0, l, tn]
                        self.OPRT_coef_intp_pl[v, 1, d, l] =  gmtr.GMTR_a_pl[ij, k0, l, tn] + gmtr.GMTR_a_pl[ij, k0, l, tn2]
                        self.OPRT_coef_intp_pl[v, 2, d, l] =  gmtr.GMTR_a_pl[ij, k0, l, tn2] - gmtr.GMTR_a_pl[ijp1, k0, l, tn]

                        self.OPRT_coef_intp_pl[v, :, d, l] *= 0.5 * gmtr.GMTR_t_pl[v, k0, l, T_RAREA]

                        self.OPRT_coef_diff_pl[v-1, d, l] = gmtr.GMTR_a_pl[v, k0, l, hn] * 0.5 * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]  # check if v-1 is correct

        return
    
    def OPRT_divergence(self, scl, scl_pl, vx, vx_pl, vy, vy_pl, vz, vz_pl, coef_div, coef_div_pl, grd, rdtype):

        prf.PROF_rapstart('OPRT_divergence', 2)        

        scl = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall), dtype=rdtype)
        scl_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kall, adm.ADM_lall_pl), dtype=rdtype)


        #gall   = adm.ADM_gall
        iall  = adm.ADM_gall_1d
        jall  = adm.ADM_gall_1d
        kall   = adm.ADM_kall
        lall   = adm.ADM_lall


        # --- Scalar divergence calculation
        for l in range(lall):
            for k in range(kall):

                #for g in range(gmin):
                #    scl[g, k, l] = 0.0
                             # 1 to 16   
                for i in range(1, iall -1):
                    for j in range(1, jall -1):
                        # ij     = g
                        # ip1j   = g + 1
                        # ip1jp1 = g + iall + 1
                        # ijp1   = g + iall
                        # im1j   = g - 1
                        # im1jm1 = g - iall - 1
                        # ijm1   = g - iall

                        scl[i, j, k, l] = (
                            coef_div[i, j, 0, grd.GRD_XDIR, l] * vx[i, j, k, l]
                            + coef_div[i, j, 1, grd.GRD_XDIR, l] * vx[i+1, j, k, l]
                            + coef_div[i, j, 2, grd.GRD_XDIR, l] * vx[i+1, j+1, k, l]
                            + coef_div[i, j, 3, grd.GRD_XDIR, l] * vx[i, j+1, k, l]
                            + coef_div[i, j, 4, grd.GRD_XDIR, l] * vx[i-1, j, k, l]
                            + coef_div[i, j, 5, grd.GRD_XDIR, l] * vx[i-1, j-1, k, l]
                            + coef_div[i, j, 6, grd.GRD_XDIR, l] * vx[i, j-1, k, l]
                        )

                for i in range(1, iall -1):
                    for j in range(1, jall -1):
                    # ij     = g
                    # ip1j   = g + 1
                    # ip1jp1 = g + iall + 1
                    # ijp1   = g + iall
                    # im1j   = g - 1
                    # im1jm1 = g - iall - 1
                    # ijm1   = g - iall

                        scl[i, j, k, l] += (
                            coef_div[i, j, 0, grd.GRD_YDIR, l] * vy[i, j, k, l]
                            + coef_div[i, j, 1, grd.GRD_YDIR, l] * vy[i+1, j, k, l]
                            + coef_div[i, j, 2, grd.GRD_YDIR, l] * vy[i+1, j+1, k, l]
                            + coef_div[i, j, 3, grd.GRD_YDIR, l] * vy[i, j+1, k, l]
                            + coef_div[i, j, 4, grd.GRD_YDIR, l] * vy[i-1, j, k, l]
                            + coef_div[i, j, 5, grd.GRD_YDIR, l] * vy[i-1, j-1, k, l]
                            + coef_div[i, j, 6, grd.GRD_YDIR, l] * vy[i, j-1, k, l]
                        )

                for i in range(1, iall -1):
                    for j in range(1, jall -1):
                        # ij     = g
                        # ip1j   = g + 1
                        # ip1jp1 = g + iall + 1
                        # ijp1   = g + iall
                        # im1j   = g - 1
                        # im1jm1 = g - iall - 1
                        # ijm1   = g - iall

                        scl[i, j, k, l] += (
                            coef_div[i, j, 0, grd.GRD_ZDIR, l] * vz[i, j, k, l]
                            + coef_div[i, j, 1, grd.GRD_ZDIR, l] * vz[i+1, j, k, l]
                            + coef_div[i, j, 2, grd.GRD_ZDIR, l] * vz[i+1, j+1, k, l]
                            + coef_div[i, j, 3, grd.GRD_ZDIR, l] * vz[i, j+1, k, l]
                            + coef_div[i, j, 4, grd.GRD_ZDIR, l] * vz[i-1, j, k, l]
                            + coef_div[i, j, 5, grd.GRD_ZDIR, l] * vz[i-1, j-1, k, l]
                            + coef_div[i, j, 6, grd.GRD_ZDIR, l] * vz[i, j-1, k, l]
                        )

                #for g in range(gmax + 1, gall):
                #    scl[i, j, k, l] = 0.0

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for k in range(adm.ADM_kall):
                    #scl_pl[:, k, l] = 0.0
                    for v in range(adm.ADM_gslf_pl, adm.ADM_gmax_pl):
                        scl_pl[n, k, l] += (
                            coef_div_pl[v, grd.GRD_XDIR, l] * vx_pl[v+1, k, l] +
                            coef_div_pl[v, grd.GRD_YDIR, l] * vy_pl[v+1, k, l] +
                            coef_div_pl[v, grd.GRD_ZDIR, l] * vz_pl[v+1, k, l]
                        )
        #else:
        #    scl_pl[:, :, :] = 0.0

        prf.PROF_rapend('OPRT_divergence', 2) 

        return


    def OPRT_gradient(self, grad, grad_pl, scl, scl_pl, coef_grad, coef_grad_pl, grd, rdtype):

        prf.PROF_rapstart('OPRT_gradient', 2)

        iall  = adm.ADM_gall_1d
        jall  = adm.ADM_gall_1d
        kall   = adm.ADM_kall
        lall   = adm.ADM_lall

        grad = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall, adm.ADM_nxyz), dtype=rdtype)
        grad_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kall, adm.ADM_lall_pl, adm.ADM_nxyz), dtype=rdtype)

        for l in range(lall):
            for k in range(kall):

                             # 1 to 16   
                for i in range(1, iall -1):
                    for j in range(1, jall -1):

                        grad[i, j, k, l, grd.GRD_XDIR] = (
                            coef_grad[i, j, 0, grd.GRD_XDIR, l] * scl[i,   j,   k, l] +
                            coef_grad[i, j, 1, grd.GRD_XDIR, l] * scl[i+1, j,   k, l] +
                            coef_grad[i, j, 2, grd.GRD_XDIR, l] * scl[i+1, j+1, k, l] +
                            coef_grad[i, j, 3, grd.GRD_XDIR, l] * scl[i,   j+1, k, l] +
                            coef_grad[i, j, 4, grd.GRD_XDIR, l] * scl[i-1, j,   k, l] +
                            coef_grad[i, j, 5, grd.GRD_XDIR, l] * scl[i-1, j-1, k, l] +
                            coef_grad[i, j, 6, grd.GRD_XDIR, l] * scl[i,   j-1, k, l]
                        )

                for i in range(1, iall -1):
                    for j in range(1, jall -1):

                        grad[i, j, k, l, grd.GRD_YDIR] = (
                            coef_grad[i, j, 0, grd.GRD_YDIR, l] * scl[i,   j,   k, l] +
                            coef_grad[i, j, 1, grd.GRD_YDIR, l] * scl[i+1, j,   k, l] +
                            coef_grad[i, j, 2, grd.GRD_YDIR, l] * scl[i+1, j+1, k, l] +
                            coef_grad[i, j, 3, grd.GRD_YDIR, l] * scl[i,   j+1, k, l] +
                            coef_grad[i, j, 4, grd.GRD_YDIR, l] * scl[i-1, j,   k, l] +
                            coef_grad[i, j, 5, grd.GRD_YDIR, l] * scl[i-1, j-1, k, l] +
                            coef_grad[i, j, 6, grd.GRD_YDIR, l] * scl[i,   j-1, k, l]
                        )

                for i in range(1, iall -1):
                    for j in range(1, jall -1):

                        grad[i, j, k, l, grd.GRD_ZDIR] = (
                            coef_grad[i, j, 0, grd.GRD_ZDIR, l] * scl[i,   j,   k, l] +
                            coef_grad[i, j, 1, grd.GRD_ZDIR, l] * scl[i+1, j,   k, l] +
                            coef_grad[i, j, 2, grd.GRD_ZDIR, l] * scl[i+1, j+1, k, l] +
                            coef_grad[i, j, 3, grd.GRD_ZDIR, l] * scl[i,   j+1, k, l] +
                            coef_grad[i, j, 4, grd.GRD_ZDIR, l] * scl[i-1, j,   k, l] +
                            coef_grad[i, j, 5, grd.GRD_ZDIR, l] * scl[i-1, j-1, k, l] +
                            coef_grad[i, j, 6, grd.GRD_ZDIR, l] * scl[i,   j-1, k, l]
                        )

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for k in range(adm.ADM_kall):
                    # grad_pl[:, k, l, XDIR] = 0.0
                    # grad_pl[:, k, l, YDIR] = 0.0
                    # grad_pl[:, k, l, ZDIR] = 0.0
                    for v in range(adm.ADM_gslf_pl, adm.ADM_gmax_pl):
                        grad_pl[n, k, l, grd.GRD_XDIR] += coef_grad_pl[v, grd.GRD_XDIR, l] * scl_pl[v+1, k, l]
                        grad_pl[n, k, l, grd.GRD_YDIR] += coef_grad_pl[v, grd.GRD_YDIR, l] * scl_pl[v+1, k, l]
                        grad_pl[n, k, l, grd.GRD_ZDIR] += coef_grad_pl[v, grd.GRD_ZDIR, l] * scl_pl[v+1, k, l]
        #else:
        #    grad_pl[:, :, :, :] = 0.0

        prf.PROF_rapend('OPRT_gradient', 2)

        return

    def OPRT_horizontalize_vec(self, vx, vx_pl, vy, vy_pl, vz, vz_pl, grd, rdtype):

        if grd.GRD_grid_type == grd.GRD_grid_type_on_plane:
            return

        prf.PROF_rapstart('OPRT_horizontalize_vec', 2)

        rscale = grd.GRD_rscale
        #gall   = adm.ADM_gall
        iall  = adm.ADM_gall_1d
        jall  = adm.ADM_gall_1d
        kall   = adm.ADM_kall
        lall   = adm.ADM_lall

        # --- Project horizontal wind to tangent plane
        for i in range(iall):   
            for j in range(jall):
                for k in range(kall):
                    for l in range(lall):
                    
                        prd = (
                            vx[i, j, k, l] * grd.GRD_x[i, j, 0, l, grd.GRD_XDIR] / rscale
                            + vy[i, j, k, l] * grd.GRD_x[i, j, 0, l, grd.GRD_YDIR] / rscale
                            + vz[i, j, k, l] * grd.GRD_x[i, j, 0, l, grd.GRD_ZDIR] / rscale
                        )
                        vx[i, j, k, l] -= prd * grd.GRD_x[i, j, 0, l, grd.GRD_XDIR] / rscale
                        vy[i, j, k, l] -= prd * grd.GRD_x[i, j, 0, l, grd.GRD_YDIR] / rscale
                        vz[i, j, k, l] -= prd * grd.GRD_x[i, j, 0, l, grd.GRD_ZDIR] / rscale

        if adm.ADM_have_pl:
            for g in range(adm.ADM_gall_pl):
                for k in range(adm.ADM_kall):
                    for l in range(adm.ADM_lall_pl):
                    
                        prd = (
                            vx_pl[g, k, l] * grd.GRD_x_pl[g, 0, l, grd.GRD_XDIR] / rscale
                            + vy_pl[g, k, l] * grd.GRD_x_pl[g, 0, l, grd.GRD_YDIR] / rscale
                            + vz_pl[g, k, l] * grd.GRD_x_pl[g, 0, l, grd.GRD_ZDIR] / rscale
                        )
                        vx_pl[g, k, l] -= prd * grd.GRD_x_pl[g, 0, l, grd.GRD_XDIR] / rscale
                        vy_pl[g, k, l] -= prd * grd.GRD_x_pl[g, 0, l, grd.GRD_YDIR] / rscale
                        vz_pl[g, k, l] -= prd * grd.GRD_x_pl[g, 0, l, grd.GRD_ZDIR] / rscale
        else:
            vx_pl[:, :, :] = 0.0
            vy_pl[:, :, :] = 0.0
            vz_pl[:, :, :] = 0.0

        prf.PROF_rapend('OPRT_horizontalize_vec', 2)

        return


    def OPRT_laplacian(scl, scl_pl, coef_lap, coef_lap_pl, rdtype):
        
        prf.PROF_rapstart('OPRT_laplacian', 2)

        iall  = adm.ADM_gall_1d
        jall  = adm.ADM_gall_1d
        kall   = adm.ADM_kdall
        lall   = adm.ADM_lall

        scl = np.zeros((iall, jall, kall, lall), dtype=rdtype)
        dscl = np.zeros((iall, jall, kall, lall), dtype=rdtype)
        scl_pl  = np.zeros((adm.ADM_gall_pl, kall, adm.ADM_lall_pl), dtype=rdtype)
        dscl_pl = np.zeros((adm.ADM_gall_pl, kall, adm.ADM_lall_pl), dtype=rdtype)

        dscl[1:iall-1, 1:jall-1, :, :] = (
            coef_lap[1:iall-1, 1:jall-1, 0, :] * scl[1:iall-1, 1:jall-1, :, :] +
            coef_lap[1:iall-1, 1:jall-1, 1, :] * scl[2:iall,   1:jall-1, :, :] +
            coef_lap[1:iall-1, 1:jall-1, 2, :] * scl[2:iall,   2:jall,   :, :] +
            coef_lap[1:iall-1, 1:jall-1, 3, :] * scl[1:iall-1, 2:jall,   :, :] +
            coef_lap[1:iall-1, 1:jall-1, 4, :] * scl[0:iall-2, 1:jall-1, :, :] +
            coef_lap[1:iall-1, 1:jall-1, 5, :] * scl[0:iall-2, 0:jall-2, :, :] +
            coef_lap[1:iall-1, 1:jall-1, 6, :] * scl[1:iall-1, 0:jall-2, :, :]
        )

        # for l in range(lall):
        #     for k in range(kall):
        #         for i in range(1, iall -1):
        #             for j in range(1, jall -1):
        #                 dscl[i, j, k, l] = (
        #                     coef_lap[i, j, 0, l] * scl[i,   j,   k, l] +
        #                     coef_lap[i, j, 1, l] * scl[i+1, j,   k, l] +
        #                     coef_lap[i, j, 2, l] * scl[i+1, j+1, k, l] +
        #                     coef_lap[i, j, 3, l] * scl[i,   j+1, k, l] +
        #                     coef_lap[i, j, 4, l] * scl[i-1, j,   k, l] +
        #                     coef_lap[i, j, 5, l] * scl[i-1, j-1, k, l] +
        #                     coef_lap[i, j, 6, l] * scl[i,   j-1, k, l]
        #                 )

        print('ADM_have_pl', adm.ADM_have_pl, 'ADM_gslf_pl', adm.ADM_gslf_pl, 'ADM_gmax_pl', adm.ADM_gmax_pl, 'ADM_lall_pl', adm.ADM_lall_pl)
        # This needs check around the vertex at pole
        if adm.ADM_have_pl:
            for l in range(adm.ADM_lall_pl):
                for k in range(adm.ADM_kall):
                    for v in range(adm.ADM_gslf_pl, adm.ADM_gall_pl):   # adm.ADM_gall_pl is adm.ADM_gmax_pl + 1 = self.ADM_vlink + 1 = 6
                        dscl_pl[v, k, l] = (
                            coef_lap_pl[v, 0, l] * scl_pl[v,   k, l] +
                            coef_lap_pl[v, 1, l] * scl_pl[v+1, k, l] +
                            coef_lap_pl[v, 2, l] * scl_pl[v+1, k, l] +
                            coef_lap_pl[v, 3, l] * scl_pl[v,   k, l] +
                            coef_lap_pl[v, 4, l] * scl_pl[v-1, k, l] +
                            coef_lap_pl[v, 5, l] * scl_pl[v-1, k, l] +
                            coef_lap_pl[v, 6, l] * scl_pl[v,   k, l]
                        )

        else:
            dscl_pl[:, :, :] = 0.0  

        prf.PROF_rapend('OPRT_laplacian', 2)

        return dscl, dscl_pl