import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc

class Gmtr:

    _instance = None

    GMTR_p_nmax = 8
    GMTR_p_AREA  = 0
    GMTR_p_RAREA = 1
    GMTR_p_IX    = 2
    GMTR_p_IY    = 3
    GMTR_p_IZ    = 4
    GMTR_p_JX    = 5
    GMTR_p_JY    = 6
    GMTR_p_JZ    = 7

    GMTR_t_nmax = 5
    GMTR_t_AREA  = 0
    GMTR_t_RAREA = 1
    GMTR_t_W1    = 2
    GMTR_t_W2    = 3
    GMTR_t_W3    = 4

    # Constants for a (axis-related parameters) - Zero-based
    GMTR_a_nmax = 12
    GMTR_a_nmax_pl = 18

    GMTR_a_HNX  = 0
    GMTR_a_HNY  = 1
    GMTR_a_HNZ  = 2
    GMTR_a_HTX  = 3
    GMTR_a_HTY  = 4
    GMTR_a_HTZ  = 5
    GMTR_a_TNX  = 6
    GMTR_a_TNY  = 7
    GMTR_a_TNZ  = 8
    GMTR_a_TTX  = 9
    GMTR_a_TTY  = 10
    GMTR_a_TTZ  = 11

    GMTR_a_TN2X = 12
    GMTR_a_TN2Y = 13
    GMTR_a_TN2Z = 14
    GMTR_a_TT2X = 15
    GMTR_a_TT2Y = 16
    GMTR_a_TT2Z = 17

    # Public variable
    GMTR_polygon_type = "ON_SPHERE"  # 'ON_SPHERE': triangle fits the sphere, 'ON_PLANE': triangle is treated as 2D

    # These were "Private" variables in the orginal code. 
    GMTR_fname = ""  
    GMTR_io_mode = "ADVANCED"  
    # Probably should be changed to _GMTR_* at some point, like:
    #_GMTR_fname = ""  
    #_GMTR_io_mode = "ADVANCED"  

    def __init__(self):
        pass

    def GMTR_setup(self, fname_in, cnst, comm, grd, vect, rdtype):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[gmtr]/Category[common share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'gmtrparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** gmtrparam not found in toml file! STOP.", file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['gmtrparam']
            self.GMTR_polygon_type = cnfs['GMTR_polygon_type']
            self.GMTR_io_mode = cnfs['GMTR_io_mode']    
            self.GMTR_fname = cnfs['GMTR_fname']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        #k0 = adm.ADM_KNONE + 1  # Zero + 1 so that k0 can be used for allocation of 1 layer

        #self.GMTR_p    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, k0, adm.ADM_lall,    self.GMTR_p_nmax))
        #self.GMTR_p_pl = np.zeros((adm.ADM_gall_pl, k0, adm.ADM_lall_pl, self.GMTR_p_nmax))

        #self.GMTR_t    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_KNONE, adm.ADM_lall, adm.ADM_TJ - adm.ADM_TI + 1, self.GMTR_t_nmax))
        #self.GMTR_t_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_KNONE, adm.ADM_lall_pl, self.GMTR_t_nmax))

        #self.GMTR_a    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_KNONE, adm.ADM_lall, adm.ADM_AJ - adm.ADM_AJ + 1, self.GMTR_a_nmax))
        #self.GMTR_a_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_KNONE, adm.ADM_lall_pl, self.GMTR_a_nmax_pl))

        #self.GMTR_area    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_lall))    # 2D array
        #self.GMTR_area_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_lall_pl)) # 2D array

        # --- Compute geometrical information for cell points ---
        self.GMTR_p_setup(cnst, comm, grd, vect, rdtype)
        #     grd.GRD_x, grd.GRD_x_pl,  # Input: grid coordinates
        #     grd.GRD_xt, grd.GRD_xt_pl,  # Input: transformed grid coordinates
        #     grd.GRD_LON, grd.GRD_LON_pl,  # Input: longitudes
        #     self.GMTR_p, self.GMTR_p_pl,  # Output: geometry data
        #     grd.GRD_rscale  # Input: scale factor
        # )

        # Fill HALO using MPI communication
        comm.COMM_data_transfer(self.GMTR_p, self.GMTR_p_pl)

        # Extract self.GMTR_area for easier use
        self.GMTR_area    = self.GMTR_p[:, adm.ADM_KNONE, :, self.GMTR_p_AREA]
        self.GMTR_area_pl = self.GMTR_p_pl[:, adm.ADM_KNONE, :, self.GMTR_p_AREA]

        # --- Compute geometrical information for cell vertices (triangles) ---
        ##self.GMTR_t = self.GMTR_t_setup(grd.GRD_x, grd.GRD_x_pl, grd.GRD_xt, grd.GRD_xt_pl, self.GMTR_t, self.GMTR_t_pl, grd.GRD_rscale, cnst, rdtype)

        # --- Compute geometrical information for cell arcs ---
        ##self.GMTR_a = self.GMTR_a_setup(grd.GRD_x, grd.GRD_x_pl, grd.GRD_xt, grd.GRD_xt_pl, self.GMTR_a, self.GMTR_a_pl, grd.GRD_rscale, cnst, rdtype)

        # Perform geometry diagnostics
        ##self.GMTR_diagnosis()

        # Output metrics if a filename is provided
        #if self.GMTR_fname:
        #    self.GMTR_output_metrics(self.GMTR_fname)

        return
    
    
    def GMTR_p_setup(self, cnst, comm, grd, vect, rdtype): 
    #, GRD_x, GRD_x_pl, GRD_xt,  GRD_xt_pl, GRD_LON, GRD_LON_pl, GMTR_p,  GMTR_p_pl, GRD_rscale):
                     
        k0 = adm.ADM_KNONE  + 1 # Zero + 1 so that k0 can be used for allocation of 1 layer

        # Define input arrays
        #grd.GRD_x     = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d,    k0, adm.ADM_lall,    adm.ADM_nxyz))
        #grd.GRD_x_pl  = np.zeros((adm.ADM_gall_pl, k0, adm.ADM_lall_pl, adm.ADM_nxyz))
        #grd.GRD_xt    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d,    k0, adm.ADM_lall, adm.ADM_TJ - adm.ADM_TI + 1, adm.ADM_nxyz))
        #grd.GRD_xt_pl = np.zeros((adm.ADM_gall_pl, k0, adm.ADM_lall_pl, adm.ADM_nxyz))

        #grd.GRD_LON    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_lall))
        #grd.GRD_LON_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_lall_pl))

        # Define output arrays
        self.GMTR_p    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d,    k0, adm.ADM_lall,    self.GMTR_p_nmax))
        self.GMTR_p_pl = np.zeros((adm.ADM_gall_pl, k0, adm.ADM_lall_pl, self.GMTR_p_nmax))

        # Define scalar input variable
        #grd.GRD_rscale = 0.0  # Replace 0.0 with an appropriate default value

        # Define working arrays
        wk    = np.zeros((adm.ADM_nxyz, 8, adm.ADM_gall_1d, adm.ADM_gall_1d))  # Fortran (0:7) → Python (8 elements)
        wk_pl = np.zeros((adm.ADM_nxyz, adm.ADM_vlink + 2))  # Fortran (0:ADM_vlink+1) → Python (ADM_vlink + 2 elements)


        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("*** setup metrics for hexagonal/pentagonal mesh", file=log_file)

        # Initialize GMTR arrays
        self.GMTR_p.fill(0.0)
        self.GMTR_p_pl.fill(0.0)

        for l in range(adm.ADM_lall):
            for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                # ij = suf(i, j)
                # ip1j = suf(i + 1, j)
                # ip1jp1 = suf(i + 1, j + 1)
                # ijp1 = suf(i, j + 1)
                # im1j = suf(i - 1, j)
                # im1jm1 = suf(i - 1, j - 1)
                # ijm1 = suf(i, j - 1)

                    # Prepare 1 center and 6 vertices
                    for d in range(adm.ADM_nxyz):
                        wk[d, 0, i, j] = grd.GRD_x[ i,   j,   adm.ADM_KNONE, l,             d]

                        wk[d, 1, i, j] = grd.GRD_xt[i,   j-1, adm.ADM_KNONE, l, adm.ADM_TJ, d]
                        wk[d, 2, i, j] = grd.GRD_xt[i,   j,   adm.ADM_KNONE, l, adm.ADM_TI, d]
                        wk[d, 3, i, j] = grd.GRD_xt[i,   j,   adm.ADM_KNONE, l, adm.ADM_TJ, d]
                        wk[d, 4, i, j] = grd.GRD_xt[i-1, j,   adm.ADM_KNONE, l, adm.ADM_TI, d]
                        wk[d, 5, i, j] = grd.GRD_xt[i-1, j-1, adm.ADM_KNONE, l, adm.ADM_TJ, d]
                        wk[d, 6, i, j] = grd.GRD_xt[i-1, j-1, adm.ADM_KNONE, l, adm.ADM_TI, d]
                        wk[d, 7, i, j] = wk[d, 1, i, j]

            if adm.ADM_have_sgp[l]:  # Pentagon case
                wk[:, 6, adm.ADM_gmin, adm.ADM_gmin] = wk[:, 1, adm.ADM_gmin, adm.ADM_gmin]
                wk[:, 7, adm.ADM_gmin, adm.ADM_gmin] = wk[:, 1, adm.ADM_gmin, adm.ADM_gmin]

            # --- Compute control area ---
            if grd.GRD_grid_type == grd.GRD_grid_type_on_plane:
                for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                    for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                        #ij = suf(i, j)

                        area = 0.0
                        for v in range(1, 7):
                            area += vect.VECTR_triangle_plane(wk[:, 0, i, j], wk[:, v, i, j], wk[:, v + 1, i, j])

                        self.GMTR_p[i, j, adm.ADM_KNONE, l, self.GMTR_p_AREA] = area
                        self.GMTR_p[i, j, adm.ADM_KNONE, l, self.GMTR_p_RAREA] = 1.0 / self.GMTR_p[i, j, adm.ADM_KNONE, l, self.GMTR_p_AREA]
            else:
                for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                    for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                        #ij = suf(i, j)

                        #print("wk[:, :, i, j]", wk[:, :, i, j])

                        wk[:, :, i, j] /= grd.GRD_rscale


                        #print("grd.GRD_rscale", grd.GRD_rscale)

                        area = 0.0
                        for v in range(1, 7):
                            area += vect.VECTR_triangle(wk[:, 0, i, j], wk[:, v, i, j], wk[:, v + 1, i, j], self.GMTR_polygon_type, grd.GRD_rscale, cnst, rdtype)
                            #print("area+", v, area, self.GMTR_polygon_type)
                            #print("wk", wk[:, 0, i, j], wk[:, v, i, j], wk[:, v + 1, i, j])

                        self.GMTR_p[i, j, adm.ADM_KNONE, l, self.GMTR_p_AREA] = area
                        #print("area", area)
                        self.GMTR_p[i, j, adm.ADM_KNONE, l, self.GMTR_p_RAREA] = 1.0 / self.GMTR_p[i, j, adm.ADM_KNONE, l, self.GMTR_p_AREA]

            # --- Compute coefficient between xyz <-> latlon ---
            if grd.GRD_grid_type == grd.GRD_grid_type_on_plane:
                self.GMTR_p[:, adm.ADM_KNONE, l, self.GMTR_p_IX] = 1.0
                self.GMTR_p[:, adm.ADM_KNONE, l, self.GMTR_p_IY] = 0.0
                self.GMTR_p[:, adm.ADM_KNONE, l, self.GMTR_p_IZ] = 0.0
                self.GMTR_p[:, adm.ADM_KNONE, l, self.GMTR_p_JX] = 0.0
                self.GMTR_p[:, adm.ADM_KNONE, l, self.GMTR_p_JY] = 1.0
                self.GMTR_p[:, adm.ADM_KNONE, l, self.GMTR_p_JZ] = 0.0
            else:
                for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                    for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                        #ij = suf(i, j)

                        sin_lambda = np.sin(grd.GRD_LON[i, j, l])
                        cos_lambda = np.cos(grd.GRD_LON[i, j, l])

                        self.GMTR_p[i, j, adm.ADM_KNONE, l, self.GMTR_p_IX] = -sin_lambda
                        self.GMTR_p[i, j, adm.ADM_KNONE, l, self.GMTR_p_IY] = cos_lambda
                        self.GMTR_p[i, j, adm.ADM_KNONE, l, self.GMTR_p_IZ] = 0.0
                        self.GMTR_p[i, j, adm.ADM_KNONE, l, self.GMTR_p_JX] = -(grd.GRD_x[i, j, adm.ADM_KNONE, l, grd.GRD_ZDIR] * cos_lambda) / grd.GRD_rscale
                        self.GMTR_p[i, j, adm.ADM_KNONE, l, self.GMTR_p_JY] = -(grd.GRD_x[i, j, adm.ADM_KNONE, l, grd.GRD_ZDIR] * sin_lambda) / grd.GRD_rscale
                        self.GMTR_p[i, j, adm.ADM_KNONE, l, self.GMTR_p_JZ] = (
                            (grd.GRD_x[i, j, adm.ADM_KNONE, l, grd.GRD_XDIR] * cos_lambda) +
                            (grd.GRD_x[i, j, adm.ADM_KNONE, l, grd.GRD_YDIR] * sin_lambda)
                        ) / grd.GRD_rscale

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl   #  0 (region that holds pole data)

            for l in range(adm.ADM_lall_pl):  # 0 to 1
                # Prepare 1 center and * vertices
                for d in range(adm.ADM_nxyz):  # 3, so 0 to 2
                      # 0to2                   0       0           0to2         
                    wk_pl[d, 0] = grd.GRD_x_pl[n, adm.ADM_KNONE, l, d]
                    for v in range(adm.ADM_vlink):  # (ICO=5)  0to4 
                          # 0to2 0to4             0to4 + 1     0            0to2
                        wk_pl[d, v] = grd.GRD_xt_pl[v + 1, adm.ADM_KNONE, l, d]   # check v or v+1 !!!
                        #0to2   0to4 + 1              0to2
                    wk_pl[d, adm.ADM_vlink + 1] = wk_pl[d, 1]


                #print("wk_pl", wk_pl)
                wk_pl[:, :] /= grd.GRD_rscale
                #print("wk_pl", wk_pl)

                # Compute control area
                area = 0.0
                for v in range(adm.ADM_vlink):  # (ICO=5)
                    area += vect.VECTR_triangle(wk_pl[:, 0], wk_pl[:, v], wk_pl[:, v + 1], self.GMTR_polygon_type, grd.GRD_rscale, cnst, rdtype)   # check v or v+1

                self.GMTR_p_pl[n, adm.ADM_KNONE, l, self.GMTR_p_AREA] = area
                self.GMTR_p_pl[n, adm.ADM_KNONE, l, self.GMTR_p_RAREA] = 1.0 / self.GMTR_p_pl[n, adm.ADM_KNONE, l, self.GMTR_p_AREA]  ###

                # Compute coefficient between xyz <-> latlon
                sin_lambda = np.sin(grd.GRD_LON_pl[n, l])
                cos_lambda = np.cos(grd.GRD_LON_pl[n, l])

                self.GMTR_p_pl[n, adm.ADM_KNONE, l, self.GMTR_p_IX] = -sin_lambda
                self.GMTR_p_pl[n, adm.ADM_KNONE, l, self.GMTR_p_IY] = cos_lambda
                self.GMTR_p_pl[n, adm.ADM_KNONE, l, self.GMTR_p_IZ] = 0.0

                self.GMTR_p_pl[n,adm.ADM_KNONE,l,self.GMTR_p_JX] = \
                    -( grd.GRD_x_pl[n,adm.ADM_KNONE,l,grd.GRD_ZDIR] * cos_lambda ) / grd.GRD_rscale
                self.GMTR_p_pl[n,adm.ADM_KNONE,l,self.GMTR_p_JY] = \
                    -( grd.GRD_x_pl[n,adm.ADM_KNONE,l,grd.GRD_ZDIR] * sin_lambda ) / grd.GRD_rscale
                self.GMTR_p_pl[n,adm.ADM_KNONE,l,self.GMTR_p_JZ] =  \
                    ( grd.GRD_x_pl[n,adm.ADM_KNONE,l,grd.GRD_XDIR] * cos_lambda 
                        + grd.GRD_x_pl[n,adm.ADM_KNONE,l,grd.GRD_YDIR] * sin_lambda ) / grd.GRD_rscale

        return
