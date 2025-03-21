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
        self.GMTR_t_setup(cnst, comm, grd, vect, rdtype)

        # --- Compute geometrical information for cell arcs ---
        self.GMTR_a_setup(cnst, comm, grd, vect, rdtype)

        # Perform geometry diagnostics
        print ("next: GMTR_diagnosis")
        self.GMTR_diagnosis(cnst, comm, grd, vect, rdtype)
        print ("done: GMTR_diagnosis")
        # Output metrics if a filename is provided
        #if self.GMTR_fname:
        #    self.GMTR_output_metrics(self.GMTR_fname)

        return
    
    
    def GMTR_p_setup(self, cnst, comm, grd, vect, rdtype): 
                         
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

    def GMTR_t_setup(self, cnst, comm, grd, vect, rdtype):

        if std.io_l:    
            with open(std.fname_log, 'a') as log_file:
                print('*** setup metrics for triangle mesh', file=log_file)
        
        wk    = np.zeros((adm.ADM_nxyz, 4, adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_TJ - adm.ADM_TI + 1), dtype=rdtype) 
        wk_pl = np.zeros((adm.ADM_nxyz, 4), dtype=rdtype)  

        k0 = adm.ADM_KNONE + 1

        self.GMTR_t = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, k0, adm.ADM_lall, adm.ADM_TJ - adm.ADM_TI + 1, self.GMTR_t_nmax), dtype=rdtype)
        self.GMTR_t_pl = np.zeros((adm.ADM_gall_pl, k0, adm.ADM_lall_pl, self.GMTR_t_nmax), dtype=rdtype)

        # Loop over levels
        for l in range(adm.ADM_lall):
            for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                    # ij = suf(i, j)
                    # ip1j = suf(i + 1, j)
                    # ip1jp1 = suf(i + 1, j + 1)
                    # ijp1 = suf(i, j + 1)

                    # Prepare 1 center and 3 vertices for 2 triangles
                    for d in range(adm.ADM_nxyz):
                        wk[d, 0, i, j, adm.ADM_TI] = grd.GRD_xt[i, j, adm.ADM_KNONE, l, adm.ADM_TI, d]

                        wk[d, 1, i, j, adm.ADM_TI] = grd.GRD_x[i,   j,   adm.ADM_KNONE, l, d]
                        wk[d, 2, i, j, adm.ADM_TI] = grd.GRD_x[i+1, j,   adm.ADM_KNONE, l, d]
                        wk[d, 3, i, j, adm.ADM_TI] = grd.GRD_x[i+1, j+1, adm.ADM_KNONE, l, d]

                        wk[d, 0, i, j, adm.ADM_TJ] = grd.GRD_xt[i,  j, adm.ADM_KNONE, l, adm.ADM_TJ, d]

                        wk[d, 1, i, j, adm.ADM_TJ] = grd.GRD_x[i,   j,   adm.ADM_KNONE, l, d]
                        wk[d, 2, i, j, adm.ADM_TJ] = grd.GRD_x[i+1, j+1, adm.ADM_KNONE, l, d]
                        wk[d, 3, i, j, adm.ADM_TJ] = grd.GRD_x[i,   j+1, adm.ADM_KNONE, l, d]

            # Treat unused triangles
            wk[:, :, adm.ADM_gmax,     adm.ADM_gmin - 1, adm.ADM_TI] = wk[:, :, adm.ADM_gmax,     adm.ADM_gmin - 1, adm.ADM_TJ]
            wk[:, :, adm.ADM_gmin - 1, adm.ADM_gmax,     adm.ADM_TJ] = wk[:, :, adm.ADM_gmin - 1, adm.ADM_gmax,     adm.ADM_TI]

            if adm.ADM_have_sgp[l]:  # Pentagon handling
                wk[:, :, adm.ADM_gmin - 1, adm.ADM_gmin - 1, adm.ADM_TI] = wk[:, :, adm.ADM_gmin, adm.ADM_gmin - 1, adm.ADM_TJ]

            # Compute areas
            if grd.GRD_grid_type == grd.GRD_grid_type_on_plane:
                for t in range(adm.ADM_TI, adm.ADM_TJ + 1):
                    for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                        for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                            #ij = suf(i, j)

                            print("on plane not ready yet")
                            prc.prc_mpistop(std.io_l, std.fname_log)
                            #area1 = vect.VECTR_triangle_plane(wk[:, 0, i, j, t], wk[:, 2, i, j, t], wk[:, 3, i, j, t])
                            #area2 = vect.VECTR_triangle_plane(wk[:, 0, i, j, t], wk[:, 3, i, j, t], wk[:, 1, i, j, t])
                            #area3 = vect.VECTR_triangle_plane(wk[:, 0, i, j, t], wk[:, 1, i, j, t], wk[:, 2, i, j, t])

                            area = area1 + area2 + area3

                            self.GMTR_t[i, j, adm.ADM_KNONE, l, t, self.GMTR_t_AREA] = area
                            self.GMTR_t[i, j, adm.ADM_KNONE, l, t, self.GMTR_t_RAREA] = 1.0 / area

                            self.GMTR_t[i, j, adm.ADM_KNONE, l, t, self.GMTR_t_W1] = area1 / area
                            self.GMTR_t[i, j, adm.ADM_KNONE, l, t, self.GMTR_t_W2] = area2 / area
                            self.GMTR_t[i, j, adm.ADM_KNONE, l, t, self.GMTR_t_W3] = area3 / area
        
            else:
                for t in range(adm.ADM_TI, adm.ADM_TJ + 1):
                    for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                        for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                            #ij = suf(i, j)

                            wk[:, :, i, j, t] /= grd.GRD_rscale

                            area1 = vect.VECTR_triangle(wk[:, 0, i, j, t], wk[:, 2, i, j, t], wk[:, 3, i, j, t], self.GMTR_polygon_type, grd.GRD_rscale, cnst, rdtype)
                            area2 = vect.VECTR_triangle(wk[:, 0, i, j, t], wk[:, 3, i, j, t], wk[:, 1, i, j, t], self.GMTR_polygon_type, grd.GRD_rscale, cnst, rdtype)
                            area3 = vect.VECTR_triangle(wk[:, 0, i, j, t], wk[:, 1, i, j, t], wk[:, 2, i, j, t], self.GMTR_polygon_type, grd.GRD_rscale, cnst, rdtype)

                            area = area1 + area2 + area3

                            self.GMTR_t[i, j, adm.ADM_KNONE, l, t, self.GMTR_t_AREA] = area
                            self.GMTR_t[i, j, adm.ADM_KNONE, l, t, self.GMTR_t_RAREA] = 1.0 / area

                            self.GMTR_t[i, j, adm.ADM_KNONE, l, t, self.GMTR_t_W1] = area1 / area
                            self.GMTR_t[i, j, adm.ADM_KNONE, l, t, self.GMTR_t_W2] = area2 / area
                            self.GMTR_t[i, j, adm.ADM_KNONE, l, t, self.GMTR_t_W3] = area3 / area


        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl
    
            for l in range(adm.ADM_lall_pl):
                for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                    ij = v
                    ijp1 = v + 1
                    if ijp1 == adm.ADM_gmax_pl + 1:
                        ijp1 = adm.ADM_gmin_pl

                    for d in range(adm.ADM_nxyz):
                        wk_pl[d, 0] = grd.GRD_xt_pl[ij, adm.ADM_KNONE, l, d]
                        wk_pl[d, 1] = grd.GRD_x_pl[n, adm.ADM_KNONE, l, d]
                        wk_pl[d, 2] = grd.GRD_x_pl[ij, adm.ADM_KNONE, l, d]
                        wk_pl[d, 3] = grd.GRD_x_pl[ijp1, adm.ADM_KNONE, l, d]

                    wk_pl[:, :] /= grd.GRD_rscale

                    area1 = vect.VECTR_triangle(wk_pl[:, 0], wk_pl[:, 2], wk_pl[:, 3], self.GMTR_polygon_type, grd.GRD_rscale, cnst, rdtype)
                    area2 = vect.VECTR_triangle(wk_pl[:, 0], wk_pl[:, 3], wk_pl[:, 1], self.GMTR_polygon_type, grd.GRD_rscale, cnst, rdtype)
                    area3 = vect.VECTR_triangle(wk_pl[:, 0], wk_pl[:, 1], wk_pl[:, 2], self.GMTR_polygon_type, grd.GRD_rscale, cnst, rdtype)

                    area = area1 + area2 + area3

                    self.GMTR_t_pl[ij, adm.ADM_KNONE, l, self.GMTR_t_AREA] = area
                    self.GMTR_t_pl[ij, adm.ADM_KNONE, l, self.GMTR_t_RAREA] = 1.0 / area

                    self.GMTR_t_pl[ij, adm.ADM_KNONE, l, self.GMTR_t_W1] = area1 / area
                    self.GMTR_t_pl[ij, adm.ADM_KNONE, l, self.GMTR_t_W2] = area2 / area
                    self.GMTR_t_pl[ij, adm.ADM_KNONE, l, self.GMTR_t_W3] = area3 / area

        return


    def GMTR_a_setup(self, cnst, comm, grd, vect, rdtype):
        
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print('*** setup metrics for cell arcs', file=log_file)

        k0 = adm.ADM_KNONE + 1

        self.GMTR_a = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, k0, adm.ADM_lall, adm.ADM_AJ - adm.ADM_AI + 1, self.GMTR_a_nmax,), dtype=rdtype)
        self.GMTR_a_pl = np.zeros((adm.ADM_gall_pl, k0, adm.ADM_lall_pl, self.GMTR_a_nmax_pl,), dtype=rdtype)

        wk = np.zeros((adm.ADM_nxyz, 2, adm.ADM_gall_1d, adm.ADM_gall_1d), dtype=rdtype)
        wk_pl = np.zeros((adm.ADM_nxyz, 2), dtype=rdtype)
        Tvec = np.zeros(3, dtype=rdtype)
        Nvec = np.zeros(3, dtype=rdtype)

        # --- Triangle
        for l in range(adm.ADM_lall):

            #--- AI
            for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 2):
                for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                    #ij = suf(i, j)
                    #ip1j = suf(i + 1, j)

                    for d in range(adm.ADM_nxyz):
                        wk[d, 0, i, j] = grd.GRD_x[i,   j, adm.ADM_KNONE, l, d]
                        wk[d, 1, i, j] = grd.GRD_x[i+1, j, adm.ADM_KNONE, l, d]

                    if std.io_l:
                        with open(std.fname_log, 'a') as log_file:
                            print("i, j, wk", i, j, file=log_file)
                            print(wk[:, 0, i, j], file=log_file)
                            print(wk[:, 1, i, j], file=log_file)

            # Handle arcs of unused triangles
            wk[:, 0, adm.ADM_gmax, adm.ADM_gmin - 1] = grd.GRD_x[adm.ADM_gmax, adm.ADM_gmin - 1, adm.ADM_KNONE, l, :]
            wk[:, 1, adm.ADM_gmax, adm.ADM_gmin - 1] = grd.GRD_x[adm.ADM_gmax, adm.ADM_gmin, adm.ADM_KNONE, l, :]
            wk[:, 0, adm.ADM_gmin - 1, adm.ADM_gmax + 1] = grd.GRD_x[adm.ADM_gmin, adm.ADM_gmax + 1, adm.ADM_KNONE, l, :]
            wk[:, 1, adm.ADM_gmin - 1, adm.ADM_gmax + 1] = grd.GRD_x[adm.ADM_gmin, adm.ADM_gmax, adm.ADM_KNONE, l, :]

            # Handle pentagon case
            if adm.ADM_have_sgp[l]:
                wk[:, 0, adm.ADM_gmin - 1, adm.ADM_gmin - 1] = grd.GRD_x[adm.ADM_gmin, adm.ADM_gmin - 1, adm.ADM_KNONE, l, :]
                wk[:, 1, adm.ADM_gmin - 1, adm.ADM_gmin - 1] = grd.GRD_x[adm.ADM_gmin + 1, adm.ADM_gmin, adm.ADM_KNONE, l, :]

            for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 2):
                for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                    #ij = suf(i, j)
                    Tvec, Nvec = self.GMTR_TNvec(wk[:, 0, i, j], wk[:, 1, i, j], grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale, grd, vect, rdtype)
                    if std.io_l:
                        with open(std.fname_log, 'a') as log_file:
                            print("Tvec", Tvec, file=log_file)
                            print("Nvec", Nvec, file=log_file)
                            print("i,j,grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale", i, j, grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale, file=log_file)
                            print("wk", wk[:, 0, i, j], wk[:, 1, i, j], file=log_file)
                            print("hohoha, I am rank", prc.prc_myrank, file=log_file)
                    #prc.prc_mpistop(std.io_l, std.fname_log)
                    # print("Tvec", Tvec)
                    # print("Nvec", Nvec) 
                    # print("i,j,grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale")
                    # print(i,j,grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale)
                    # print("wk")
                    # print(wk[:, 0, i, j], wk[:, 1, i, j])
                    # print("hohoha, I am rank", prc.prc_myrank )
                    # prc.prc_mpistop(std.io_l, std.fname_log)


                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AI, self.GMTR_a_TNX] = Nvec[0]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AI, self.GMTR_a_TNY] = Nvec[1]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AI, self.GMTR_a_TNZ] = Nvec[2]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AI, self.GMTR_a_TTX] = Tvec[0]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AI, self.GMTR_a_TTY] = Tvec[1]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AI, self.GMTR_a_TTZ] = Tvec[2]

            #--- AIJ
            for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                    #ij = suf(i, j)
                    #ip1jp1 = suf(i + 1, j + 1)

                    for d in range(adm.ADM_nxyz):
                        wk[d, 0, i, j] = grd.GRD_x[i, j, adm.ADM_KNONE, l, d]
                        wk[d, 1, i, j] = grd.GRD_x[i+1, j+1, adm.ADM_KNONE, l, d]

            for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                    #ij = suf(i, j)
                    Tvec, Nvec = self.GMTR_TNvec(wk[:, 0, i, j], wk[:, 1, i, j], grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale, grd, vect, rdtype)
                    
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AIJ, self.GMTR_a_TNX] = Nvec[0]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AIJ, self.GMTR_a_TNY] = Nvec[1]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AIJ, self.GMTR_a_TNZ] = Nvec[2]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AIJ, self.GMTR_a_TTX] = Tvec[0]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AIJ, self.GMTR_a_TTY] = Tvec[1]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AIJ, self.GMTR_a_TTZ] = Tvec[2]

            #--- AJ
            for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 2):
                    #ij = suf(i, j)
                    #ijp1 = suf(i, j + 1)

                    for d in range(adm.ADM_nxyz):
                        wk[d, 0, i, j] = grd.GRD_x[i, j,   adm.ADM_KNONE, l, d]
                        wk[d, 1, i, j] = grd.GRD_x[i, j+1, adm.ADM_KNONE, l, d]

            # Handle arcs of unused triangles
            wk[:, 0, adm.ADM_gmax + 1, adm.ADM_gmin - 1] = grd.GRD_x[adm.ADM_gmax + 1, adm.ADM_gmin, adm.ADM_KNONE, l, :]
            wk[:, 1, adm.ADM_gmax + 1, adm.ADM_gmin - 1] = grd.GRD_x[adm.ADM_gmax, adm.ADM_gmin, adm.ADM_KNONE, l, :]
            wk[:, 0, adm.ADM_gmin - 1, adm.ADM_gmax] = grd.GRD_x[adm.ADM_gmin - 1, adm.ADM_gmax, adm.ADM_KNONE, l, :]
            wk[:, 1, adm.ADM_gmin - 1, adm.ADM_gmax] = grd.GRD_x[adm.ADM_gmin, adm.ADM_gmax, adm.ADM_KNONE, l, :]

            # Compute AJ normal and tangent vectors
            for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 2):
                    #ij = suf(i, j)
                    Tvec, Nvec = self.GMTR_TNvec(wk[:, 0, i, j], wk[:, 1, i, j], grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale, grd, vect, rdtype)
                    
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AJ, self.GMTR_a_TNX] = Nvec[0]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AJ, self.GMTR_a_TNY] = Nvec[1]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AJ, self.GMTR_a_TNZ] = Nvec[2]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AJ, self.GMTR_a_TTX] = Tvec[0]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AJ, self.GMTR_a_TTY] = Tvec[1]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AJ, self.GMTR_a_TTZ] = Tvec[2]

        # --- Hexagon
        for l in range(adm.ADM_lall):

            #---AI
            for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                    #ij = suf(i, j)
                    #ijm1 = suf(i, j - 1)

                    for d in range(adm.ADM_nxyz):
                        wk[d, 0, i, j] = grd.GRD_xt[i, j,   adm.ADM_KNONE, l, adm.ADM_TI, d]
                        wk[d, 1, i, j] = grd.GRD_xt[i, j-1, adm.ADM_KNONE, l, adm.ADM_TJ, d]

            for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                    #ij = suf(i, j)
                    Tvec, Nvec = self.GMTR_TNvec(wk[:, 0, i, j], wk[:, 1, i, j], grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale, grd, vect, rdtype)
                    
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AI, self.GMTR_a_HNX] = Nvec[0]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AI, self.GMTR_a_HNY] = Nvec[1]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AI, self.GMTR_a_HNZ] = Nvec[2]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AI, self.GMTR_a_HTX] = Tvec[0]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AI, self.GMTR_a_HTY] = Tvec[1]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AI, self.GMTR_a_HTZ] = Tvec[2]

            #---AIJ
            for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                    #ij = suf(i, j)
                    for d in range(adm.ADM_nxyz):
                        wk[d, 0, i, j] = grd.GRD_xt[i, j, adm.ADM_KNONE, l, adm.ADM_TJ, d]
                        wk[d, 1, i, j] = grd.GRD_xt[i, j, adm.ADM_KNONE, l, adm.ADM_TI, d]

            # Handle arcs of unused hexagon
            wk[:, 0, adm.ADM_gmax, adm.ADM_gmin - 1] = grd.GRD_xt[adm.ADM_gmax, adm.ADM_gmin - 1, adm.ADM_KNONE, l, adm.ADM_TJ, :]
            wk[:, 1, adm.ADM_gmax, adm.ADM_gmin - 1] = grd.GRD_xt[adm.ADM_gmax, adm.ADM_gmin, adm.ADM_KNONE, l, adm.ADM_TI, :]
            wk[:, 0, adm.ADM_gmin - 1, adm.ADM_gmax] = grd.GRD_xt[adm.ADM_gmin, adm.ADM_gmax, adm.ADM_KNONE, l, adm.ADM_TJ, :]
            wk[:, 1, adm.ADM_gmin - 1, adm.ADM_gmax] = grd.GRD_xt[adm.ADM_gmin - 1, adm.ADM_gmax, adm.ADM_KNONE, l, adm.ADM_TI, :]

            for j in range(adm.ADM_gmin - 1, adm.ADM_gmax):
                for i in range(adm.ADM_gmin - 1, adm.ADM_gmax):
                    #ij = suf(i, j)
                    Tvec, Nvec = self.GMTR_TNvec(wk[:, 0, i, j], wk[:, 1, i, j], grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale, grd, vect, rdtype)
                    
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AIJ, self.GMTR_a_HNX] = Nvec[0]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AIJ, self.GMTR_a_HNY] = Nvec[1]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AIJ, self.GMTR_a_HNZ] = Nvec[2]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AIJ, self.GMTR_a_HTX] = Tvec[0]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AIJ, self.GMTR_a_HTY] = Tvec[1]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AIJ, self.GMTR_a_HTZ] = Tvec[2]

            #---AJ
            for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                    #ij = suf(i, j)
                    #im1j = suf(i - 1, j)

                    for d in range(adm.ADM_nxyz):
                        wk[d, 0, i, j] = grd.GRD_xt[i-1, j, adm.ADM_KNONE, l, adm.ADM_TI, d]
                        wk[d, 1, i, j] = grd.GRD_xt[i,   j, adm.ADM_KNONE, l, adm.ADM_TJ, d]

            # Handle pentagon case
            if adm.ADM_have_sgp[l]:
                wk[:, 0, adm.ADM_gmin, adm.ADM_gmin - 1] = grd.GRD_xt[adm.ADM_gmin, adm.ADM_gmin, adm.ADM_KNONE, l, adm.ADM_TI, :]
                wk[:, 1, adm.ADM_gmin, adm.ADM_gmin - 1] = grd.GRD_xt[adm.ADM_gmin, adm.ADM_gmin - 1, adm.ADM_KNONE, l, adm.ADM_TJ, :]

            for j in range(adm.ADM_gmin - 1, adm.ADM_gmax):
                for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                    #ij = suf(i, j)
                    Tvec, Nvec = self.GMTR_TNvec(wk[:, 0, i, j], wk[:, 1, i, j], grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale, grd, vect, rdtype)
                    
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AJ, self.GMTR_a_HNX] = Nvec[0]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AJ, self.GMTR_a_HNY] = Nvec[1]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AJ, self.GMTR_a_HNZ] = Nvec[2]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AJ, self.GMTR_a_HTX] = Tvec[0]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AJ, self.GMTR_a_HTY] = Tvec[1]
                    self.GMTR_a[i, j, adm.ADM_KNONE, l, adm.ADM_AJ, self.GMTR_a_HTZ] = Tvec[2]

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl
            
            for l in range(adm.ADM_lall_pl):

                #--- Triangle (arc 1)
                for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                    ij = v
                    for d in range(adm.ADM_nxyz):
                        wk_pl[d, 0] = grd.GRD_x_pl[n, adm.ADM_KNONE, l, d]
                        wk_pl[d, 1] = grd.GRD_x_pl[ij, adm.ADM_KNONE, l, d]

                    Tvec, Nvec = self.GMTR_TNvec(wk_pl[:, 0], wk_pl[:, 1], grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale, grd, vect, rdtype)
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_TNX] = Nvec[0]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_TNY] = Nvec[1]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_TNZ] = Nvec[2]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_TTX] = Tvec[0]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_TTY] = Tvec[1]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_TTZ] = Tvec[2]
                
                # Triangle (arc 2)
                for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                    ij = v
                    ijp1 = v + 1 
                    if ijp1 == adm.ADM_gmax_pl + 1:
                        ijp1 = adm.ADM_gmin_pl

                    for d in range(adm.ADM_nxyz):
                        wk_pl[d, 0] = grd.GRD_x_pl[ij, adm.ADM_KNONE, l, d]
                        wk_pl[d, 1] = grd.GRD_x_pl[ijp1, adm.ADM_KNONE, l, d]
                    
                    Tvec, Nvec = self.GMTR_TNvec(wk_pl[:, 0], wk_pl[:, 1], grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale, grd, vect, rdtype)
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_TN2X] = Nvec[0]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_TN2Y] = Nvec[1]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_TN2Z] = Nvec[2]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_TT2X] = Tvec[0]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_TT2Y] = Tvec[1]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_TT2Z] = Tvec[2]
                
                # Hexagon
                for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                    ij = v
                    ijm1 = v - 1 
                    if ijm1 == adm.ADM_gmin_pl - 1:
                        ijm1 = adm.ADM_gmax_pl

                    for d in range(adm.ADM_nxyz):
                        wk_pl[d, 0] = grd.GRD_xt_pl[ijm1, adm.ADM_KNONE, l, d]
                        wk_pl[d, 1] = grd.GRD_xt_pl[ij,   adm.ADM_KNONE, l, d]
                    
                    Tvec, Nvec = self.GMTR_TNvec(wk_pl[:, 0], wk_pl[:, 1], grd.GRD_grid_type, self.GMTR_polygon_type, grd.GRD_rscale, grd, vect, rdtype)
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_HNX] = Nvec[0]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_HNY] = Nvec[1]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_HNZ] = Nvec[2]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_HTX] = Tvec[0]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_HTY] = Tvec[1]
                    self.GMTR_a_pl[ij, adm.ADM_KNONE, l, self.GMTR_a_HTZ] = Tvec[2]

        return
    
    def GMTR_TNvec(self, vFrom, vTo, grid_type, polygon_type, radius, grd, vect, rdtype):

        o = np.zeros(3, dtype=rdtype)  # Origin point
        vT = np.zeros(3, dtype=rdtype)  # tangential vector
        vN = np.zeros(3, dtype=rdtype)  # normal vector

        vT[:] = vTo - vFrom  # Compute tangential vector
    
        if grid_type == grd.GRD_grid_type_on_plane:  # Treat as point on the plane
            vN[0] = -vT[1]
            vN[1] = vT[0]
            vN[2] = 0.0
    
        elif grid_type == grd.GRD_grid_type_on_sphere:  # Treat as point on the sphere
            distance = 0.0

            if polygon_type == "ON_PLANE":  # Length of a line
                distance = vect.VECTR_dot(vFrom, vTo, vFrom, vTo, rdtype)
                distance = np.sqrt(distance)

            elif polygon_type == "ON_SPHERE":  # Length of a geodesic line
                angle = vect.VECTR_angle(vFrom, o, vTo, rdtype)
                distance = angle * radius

            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("000: stopping here?", file=log_file)
                    print("vT", vT, file=log_file)
                    #prc.prc_mpistop(std.io_l, std.fname_log)

            length = vect.VECTR_abs(vT, rdtype)

            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("AAA: length?", length, file=log_file)
                    #prc.prc_mpistop(std.io_l, std.fname_log)

            vT[:] *= distance / length  # Normalize tangential vector
            
            vN[:] = vect.VECTR_cross(o, vFrom, o, vTo, rdtype)  # Compute normal vector
            length = vect.VECTR_abs(vN, rdtype)

            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("BBB: length?", length, file=log_file)
                    #prc.prc_mpistop(std.io_l, std.fname_log)
            vN[:] *= distance / length  # Normalize normal vector
        
        return vT, vN
    
    def GMTR_diagnosis(self, cnst, comm, grd, vect, rdtype):
        
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print('*** Diagnose grid property', file=log_file)
        
        k0 = adm.ADM_KNONE + 1 
        k = adm.ADM_KNONE

        angle = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, k0, adm.ADM_lall), dtype=rdtype)
        angle_pl = np.zeros((adm.ADM_gall_pl, k0, adm.ADM_lall_pl), dtype=rdtype)
        length = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, k0, adm.ADM_lall), dtype=rdtype)
        length_pl = np.zeros((adm.ADM_gall_pl, k0, adm.ADM_lall_pl), dtype=rdtype)
        sqarea = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, k0, adm.ADM_lall), dtype=rdtype)
        sqarea_pl = np.zeros((adm.ADM_gall_pl, k0, adm.ADM_lall_pl), dtype=rdtype)

#        len_arr = np.zeros(6, dtype=rdtype)
#        ang_arr = np.zeros(6, dtype=rdtype)
        len_arr = np.zeros(7, dtype=rdtype)
        ang_arr = np.zeros(7, dtype=rdtype)
        p = np.zeros((adm.ADM_nxyz, 8), dtype=rdtype)  # p was 0 based in Fortran code!!
        nvlenC = 0.0
        nvlenS = 0.0
        nv = np.zeros(3, dtype=rdtype)

        nlen    = 0.0
        len_tot = 0.0

        for l in range(adm.ADM_lall):
            for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):

                    if adm.ADM_have_sgp[l] and i == adm.ADM_gmin and j == adm.ADM_gmin:  # Pentagon case
                        p[:, 0] = grd.GRD_xt[i, j-1, k, l, adm.ADM_TJ, :]
                        p[:, 1] = grd.GRD_xt[i, j, k, l, adm.ADM_TI, :]
                        p[:, 2] = grd.GRD_xt[i, j, k, l, adm.ADM_TJ, :]
                        p[:, 3] = grd.GRD_xt[i-1, j, k, l, adm.ADM_TI, :]
                        p[:, 4] = grd.GRD_xt[i-1, j-1, k, l, adm.ADM_TJ, :]
                        p[:, 5] = p[:, 0]
                        p[:, 6] = p[:, 1]
                        
                        len_arr[:] = 0.0
                        ang_arr[:] = 0.0
                        for m in range(1, 6):  
                            # vector length of Pm->Pm-1, Pm->Pm+1
                            len_arr[m] = np.sqrt(vect.VECTR_dot(p[:, m], p[:, m-1], p[:, m], p[:, m-1], rdtype))
                            len_tot += len_arr[m]   
                            nlen += 1.0
                            # angle of Pm-1->Pm->Pm+1
                            nvlenC = vect.VECTR_dot(p[:, m], p[:, m-1], p[:, m], p[:, m+1], rdtype)
                            nv[:] = vect.VECTR_cross(p[:, m], p[:, m-1], p[:, m], p[:, m+1], rdtype)
                            nvlenS = vect.VECTR_abs(nv, rdtype)
                            ang_arr[m] = np.arctan2(nvlenS, nvlenC)
                        
                        # maximum/minimum ratio of angle between the cell vertexes
                        #angle[i, j, k, l] = np.max(ang_arr[:5]) / np.min(ang_arr[:5]) - 1.0
                        angle[i, j, k, l] = np.max(ang_arr[1:6]) / np.min(ang_arr[1:6]) - 1.0
                        # l_mean: side length of regular pentagon =sqrt(area/1.7204774005)
                        area = self.GMTR_p[i, j, k, l, self.GMTR_p_AREA]
                        l_mean = np.sqrt(4.0 / np.sqrt(25.0 + 10.0 * np.sqrt(5.0)) * area)
 
                        #temp = np.sum((len_arr[:5] - l_mean) ** 2)
                        temp = np.sum((len_arr[1:6] - l_mean) ** 2)
                        # distortion of side length from l_mean
                        length[i, j, k, l] = np.sqrt(temp / 5.0) / l_mean

                    else:  # Hexagon case
                        p[:, 0] = grd.GRD_xt[i, j-1, k, l, adm.ADM_TJ, :]
                        p[:, 1] = grd.GRD_xt[i, j, k, l, adm.ADM_TI, :]
                        p[:, 2] = grd.GRD_xt[i, j, k, l, adm.ADM_TJ, :]
                        p[:, 3] = grd.GRD_xt[i-1, j, k, l, adm.ADM_TI, :]
                        p[:, 4] = grd.GRD_xt[i-1, j-1, k, l, adm.ADM_TJ, :]
                        p[:, 5] = grd.GRD_xt[i-1, j-1, k, l, adm.ADM_TI, :]
                        p[:, 6] = p[:, 0]
                        p[:, 7] = p[:, 1]
                        
                        len_arr[:] = 0.0
                        ang_arr[:] = 0.0
                        for m in range(1, 7):
                            #vector length of Pm->Pm-1, Pm->Pm+1
                            len_arr[m] = np.sqrt(vect.VECTR_dot(p[:, m], p[:, m-1], p[:, m], p[:, m-1], rdtype))
                            len_tot += len_arr[m] 
                            nlen += 1.0
                            # angle of Pm-1->Pm->Pm+1
                            nvlenC = vect.VECTR_dot(p[:, m], p[:, m-1], p[:, m], p[:, m+1], rdtype)
                            nv[:] = vect.VECTR_cross(p[:, m], p[:, m-1], p[:, m], p[:, m+1], rdtype)
                            nvlenS = vect.VECTR_abs(nv, rdtype)
                            ang_arr[m] = np.arctan2(nvlenS, nvlenC)
                        
                        # maximum/minimum ratio of angle between the cell vertexes
                        #angle[i, j, k, l] = np.max(ang_arr[:6]) / np.min(ang_arr[:6]) - 1.0  # divide by 0 error occured here
                        angle[i, j, k, l] = np.max(ang_arr[1:7]) / np.min(ang_arr[1:7]) - 1.0  
                        area = self.GMTR_p[i, j, k, l, self.GMTR_p_AREA]
                        l_mean = np.sqrt(4.0 / np.sqrt(3.0) / 6.0 * area)
                        #temp = np.sum((len_arr[:6] - l_mean) ** 2)
                        temp = np.sum((len_arr[1:7] - l_mean) ** 2)
                        length[i, j, k, l] = np.sqrt(temp / 6.0) / l_mean


        local_area = 0.0
        for l in range(adm.ADM_lall):
            for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                    local_area += self.GMTR_p[i, j, k, l, self.GMTR_p_AREA]

        if adm.ADM_have_pl:
            for l in range(adm.ADM_lall_pl):
                local_area += self.GMTR_p_pl[adm.ADM_gslf_pl, k, l, self.GMTR_p_AREA]

        global_area = comm.Comm_Stat_sum(local_area,rdtype)
        global_grid = 10 * 4**adm.ADM_glevel + 2
        sqarea_avg = np.sqrt(global_area / rdtype(global_grid))

        sqarea[:, :, :, :] = np.sqrt(self.GMTR_p[:, :, :, :, self.GMTR_p_AREA])
        sqarea_pl[:, :, :] = np.sqrt(self.GMTR_p_pl[:, :, :, self.GMTR_p_AREA])

        sqarea_local_max = -1.0e30
        sqarea_local_min = 1.0e30
        length_local_max = -1.0e30
        angle_local_max = -1.0e30

        for l in range(adm.ADM_lall):
            for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                    #ij = suf(i, j)
                    sqarea_local_max = max(sqarea_local_max, sqarea[i, j, k, l])
                    sqarea_local_min = min(sqarea_local_min, sqarea[i, j, k, l])
                    length_local_max = max(length_local_max, length[i, j, k, l])
                    angle_local_max = max(angle_local_max, angle[i, j, k, l])

        if adm.ADM_have_pl:
            for l in range(adm.ADM_lall_pl):
                sqarea_local_max = max(sqarea_local_max, sqarea_pl[adm.ADM_gslf_pl, k, l])
                sqarea_local_min = min(sqarea_local_min, sqarea_pl[adm.ADM_gslf_pl, k, l])
                length_local_max = max(length_local_max, length_pl[adm.ADM_gslf_pl, k, l])
                angle_local_max = max(angle_local_max, angle_pl[adm.ADM_gslf_pl, k, l])


        sqarea_max = comm.Comm_Stat_max(sqarea_local_max,rdtype)
        sqarea_min = comm.Comm_Stat_min(sqarea_local_min,rdtype)
        length_max = comm.Comm_Stat_max(length_local_max,rdtype)
        angle_max = comm.Comm_Stat_max(angle_local_max,rdtype)
        length_avg = len_tot / nlen    

        # Print diagnostic results
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("\n------ Diagnosis result ---", file=log_file)
                print(f"--- ideal  global surface area  = {4.0 * cnst.CONST_PI * cnst.CONST_RADIUS**2 * 1e-6} [km²]", file=log_file)
                print(f"--- actual global surface area  = {global_area * 1e-6} [km²]", file=log_file)
                print(f"--- global total number of grid = {global_grid}", file=log_file)
                print('', file=log_file)
                print(f"--- average grid interval       = {sqarea_avg * 1e-3} [km]", file=log_file)
                print(f"--- max grid interval           = {sqarea_max * 1e-3} [km]", file=log_file)
                print(f"--- min grid interval           = {sqarea_min * 1e-3} [km]", file=log_file)
                print(f"--- ratio max/min grid interval = {sqarea_max / sqarea_min}", file=log_file)
                print(f"--- average length of arc(side) = {length_avg * 1e-3} [km]", file=log_file)
                print('', file=log_file)
                print(f"--- max length distortion       = {length_max * 1e-3} [km]", file=log_file)
                print(f"--- max angle distortion        = {angle_max * 180.0 / cnst.CONST_PI} [deg]", file=log_file)

        return