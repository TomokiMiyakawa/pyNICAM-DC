import toml
import numpy as np
from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_vector import vect
#from mod_prof import prf

class Grd:
    
    _instance = None
    
    # character length

    #++ Public parameters & variables

    #Indentifiers for the directions in the Cartesian coordinate
    GRD_XDIR = 0
    GRD_YDIR = 1
    GRD_ZDIR = 2

    #Indentifiers for the directions in the spherical coordinate
    I_LAT = 0
    I_LON = 1

    GRD_ZSFC = 0
    GRD_ZSD  = 1

    GRD_Z  = 0
    GRD_ZH = 1

    GRD_grid_type_on_sphere = 0
    GRD_grid_type_on_plane = 1


#====== Horizontal Grid ======
#
# Grid points ( X: CELL CENTER )
#           .___.
#          /     \
#         .   p   .
#          \ ___ /
#           '   '
#
# Grid points ( Xt: CELL VERTEX )
#           p___p
#          /     \
#         p       p
#          \ ___ /
#           p   p
#
# Grid points ( Xr: CELL ARC )
#           ._p_.
#          p     p
#         .       .
#          p _ _ p
#           ' p '

    def __init__(self):
        #self._instance = self
        #self._grd = None
        #self._grd = self._grd_setup()
        pass

    def GRD_setup(self, fname_in, cnst, comm, rdtype):
        #self._grd = self._grd_setup()

        kn = adm.ADM_KNONE   # kn (=1) is used as the number of layers in a single layer. 
        k0 = adm.ADM_K0      # k0 (=0) is used as the index of the single layer

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[grd]/Category[common share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'grdparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** grdparam not found in toml file! STOP.", file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['grdparam']
            self.GRD_grid_type = cnfs['GRD_grid_type']
            self.hgrid_io_mode  = cnfs['hgrid_io_mode']
            self.hgrid_fname    = cnfs['hgrid_fname']
            self.vgrid_fname    = cnfs['vgrid_fname']
            self.vgrid_scheme   = cnfs['vgrid_scheme']
            self.h_efold        = cnfs['h_efold']
            self.topo_io_mode   = cnfs['topo_io_mode']
            self.topo_fname     = cnfs['topo_fname']
            self.toposd_fname   = cnfs['toposd_fname']
            self.hflat          = cnfs['hflat']
            self.output_vgrid   = cnfs['output_vgrid']
            self.hgrid_comm_flg = cnfs['hgrid_comm_flg']
            self.triangle_size  = cnfs['triangle_size']

        #    self.COMM_apply_barrier = cnfs['commparam']['COMM_apply_barrier']  
        #    self.COMM_varmax = cnfs['commparam']['COMM_varmax']  
            #debug = cnfs['commparam']['debug']  
            #testonly = cnfs['commparam']['testonly']  

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    #print(cnfs['grdparam'],file=log_file)
                    print(cnfs,file=log_file)


        #---< horizontal grid >---
        self.GRD_x     = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d,    kn, adm.ADM_lall,                              adm.ADM_nxyz), cnst.CONST_UNDEF)
        
        self.GRD_x_pl  = np.full((adm.ADM_gall_pl, kn, adm.ADM_lall_pl,                                               adm.ADM_nxyz), cnst.CONST_UNDEF)
        self.GRD_xt    = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d, kn, adm.ADM_lall,    adm.ADM_TJ - adm.ADM_TI + 1, adm.ADM_nxyz), cnst.CONST_UNDEF)
        self.GRD_xt_pl = np.full((adm.ADM_gall_pl, kn, adm.ADM_lall_pl,                                               adm.ADM_nxyz), cnst.CONST_UNDEF)
        self.GRD_xr    = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d, kn, adm.ADM_lall,    adm.ADM_AJ - adm.ADM_AI + 1, adm.ADM_nxyz), cnst.CONST_UNDEF)
        self.GRD_xr_pl = np.full((adm.ADM_gall_pl,                  kn, adm.ADM_lall_pl,                              adm.ADM_nxyz), cnst.CONST_UNDEF)

        self.GRD_s     = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d, kn, adm.ADM_lall,                                 2), cnst.CONST_UNDEF)
        self.GRD_s_pl  = np.full((adm.ADM_gall_pl,                  kn, adm.ADM_lall_pl,                              2), cnst.CONST_UNDEF)
        self.GRD_st    = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d, kn, adm.ADM_lall,    adm.ADM_TJ - adm.ADM_TI + 1, 2), cnst.CONST_UNDEF)
        self.GRD_st_pl = np.full((adm.ADM_gall_pl,                  kn, adm.ADM_lall_pl,                              2), cnst.CONST_UNDEF)

        self.GRD_LAT   = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d,  adm.ADM_lall),    cnst.CONST_UNDEF)
        self.GRD_LAT_pl= np.full((adm.ADM_gall_pl, adm.ADM_lall_pl),                   cnst.CONST_UNDEF)
        self.GRD_LON   = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d,  adm.ADM_lall),    cnst.CONST_UNDEF)
        self.GRD_LON_pl= np.full((adm.ADM_gall_pl, adm.ADM_lall_pl),                   cnst.CONST_UNDEF)

        
        self.GRD_input_hgrid(self.hgrid_fname, True, self.hgrid_io_mode, comm, rdtype)  

        #print("GRD_x, 0:", self.GRD_x[:,:,0,0,0]**2 + self.GRD_x[:,:,0,0,1]**2 + self.GRD_x[:,:,0,0,2]**2)

        #print("hoho3, adm.ADM_prc_me", adm.ADM_prc_me)
        #prc.prc_mpistop(std.io_l, std.fname_log)

        # Data transfer for self.GRD_x (excluding self.GRD_xt)
        if self.hgrid_comm_flg:
            comm.COMM_data_transfer(self.GRD_x, self.GRD_x_pl)  
        #    print("GRD_x_pl, 0:", self.GRD_x_pl)
        #    print("GRD_x, 1:", self.GRD_x.shape)
  
        #print("hoho_, adm.ADM_prc_me", adm.ADM_prc_me)
        #prc.prc_mpistop(std.io_l, std.fname_log)   
        #    print("GRD_x, 1:", self.GRD_x[:,:,0,0,0]**2 + self.GRD_x[:,:,0,0,1]**2 + self.GRD_x[:,:,0,0,2]**2)

        #print("hoho4, adm.ADM_prc_me", adm.ADM_prc_me)
        #prc.prc_mpistop(std.io_l, std.fname_log)

        # Scaling logic
        if self.GRD_grid_type == self.GRD_grid_type_on_plane:
            self.GRD_scaling(self.triangle_size)  
        else:
            self.GRD_scaling(cnst.CONST_RADIUS)  

        # Calculate latitude/longitude of each grid point
        self.GRD_makelatlon(cnst)  

        # Calculate position of cell arc
        self.GRD_makearc()  

        #---< Surface Height >---
        self.GRD_zs     = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, kn, adm.ADM_lall,    self.GRD_ZSD - self.GRD_ZSFC + 1))
        self.GRD_zs_pl  = np.zeros((adm.ADM_gall_pl,                  kn, adm.ADM_lall_pl, self.GRD_ZSD - self.GRD_ZSFC + 1))

        # Call function to read topographic data (assuming function exists)
        self.GRD_input_topograph(fname_in, self.topo_fname, self.toposd_fname, self.topo_io_mode, cnst, comm)

        # ---< Vertical Coordinate >---
        if adm.ADM_kall != adm.ADM_KNONE :
            self.GRD_gz   = np.zeros(adm.ADM_kdall)
            self.GRD_gzh  = np.zeros(adm.ADM_kdall)
            self.GRD_dgz  = np.zeros(adm.ADM_kdall)
            self.GRD_dgzh = np.zeros(adm.ADM_kdall)
            self.GRD_rdgz = np.zeros(adm.ADM_kdall)
            self.GRD_rdgzh = np.zeros(adm.ADM_kdall)

            self.GRD_afact = np.zeros(adm.ADM_kdall)
            self.GRD_bfact = np.zeros(adm.ADM_kdall)
            self.GRD_cfact = np.zeros(adm.ADM_kdall)
            self.GRD_dfact = np.zeros(adm.ADM_kdall)

            self.GRD_vz    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,    self.GRD_ZH - self.GRD_Z + 1))
            self.GRD_vz_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, self.GRD_ZH - self.GRD_Z + 1))

            self.GRD_input_vgrid(self.vgrid_fname)

            # --- Calculation of grid intervals (cell center) ---
            for k in range(adm.ADM_kmin - 1, adm.ADM_kmax + 1):
                self.GRD_dgz[k] = self.GRD_gzh[k + 1] - self.GRD_gzh[k]
            self.GRD_dgz[adm.ADM_kmax + 1] = self.GRD_dgz[adm.ADM_kmax]

            # --- Calculation of grid intervals (cell wall) ---
            for k in range(adm.ADM_kmin, adm.ADM_kmax + 2):  # +1 in Fortran means +2 in Python due to 0-based indexing
                self.GRD_dgzh[k] = self.GRD_gz[k] - self.GRD_gz[k-1]
            self.GRD_dgzh[adm.ADM_kmin - 1] = self.GRD_dgzh[adm.ADM_kmin]

            # Compute inverse grid spacing
            for k in range(adm.ADM_kdall):
                self.GRD_rdgz[k]  = 1.0 / self.GRD_dgz[k]
                self.GRD_rdgzh[k] = 1.0 / self.GRD_dgzh[k]


            # Compute height top
            self.GRD_htop = self.GRD_gzh[adm.ADM_kmax+1] - self.GRD_gzh[adm.ADM_kmin]

            # Compute vertical interpolation factor   ####### WORKING HERE
            for k in range(adm.ADM_kmin, adm.ADM_kmax + 2):
                self.GRD_afact[k] = (self.GRD_gzh[k] - self.GRD_gz[k - 1]) / (self.GRD_gz[k] - self.GRD_gz[k - 1])

            self.GRD_afact[adm.ADM_kmin - 1] = 1.0

            self.GRD_bfact[:] = 1.0 - self.GRD_afact[:]

            for k in range(adm.ADM_kmin, adm.ADM_kmax + 1):
                self.GRD_cfact[k] = (self.GRD_gz[k] - self.GRD_gzh[k]) / (self.GRD_gzh[k + 1] - self.GRD_gzh[k])

            self.GRD_cfact[adm.ADM_kmin - 1] = 1.0
            self.GRD_cfact[adm.ADM_kmax + 1] = 0.0

            self.GRD_dfact[:] = 1.0 - self.GRD_cfact[:]

            # --- Setup z-coordinate ---
            #nstart = self.suf(adm.ADM_gmin, adm.ADM_gmin)
            #nend   = self.suf(adm.ADM_gmax, adm.ADM_gmax)

            # --- Select Vertical Grid Scheme ---
            if self.vgrid_scheme == "LINEAR":
                print("LINEAR SCHEME not tested yet")  
                prc.prc_mpifinish(std.io_l, std.fname_log)
                import sys
                sys.exit()
                #   Linear transformation of the vertical grid (Gal-Chen & Sommerville, 1975)
                #   gz = H(z-zs)/(H-zs) -> z = (H-zs)/H * gz + zs

                kflat = -1
                if self.hflat > 0.0:  # Default is -999.0 in Fortran
                    for k in range(adm.ADM_kmin + 1, adm.ADM_kmax + 2):  # Adjusted for Python indexing
                        if self.hflat < self.GRD_gzh[k]:
                            kflat = k
                            break

                if kflat == -1:
                    kflat = adm.ADM_kmax + 1
                    htop = self.GRD_htop
                else:
                    htop = self.GRD_gzh[kflat] - self.GRD_gzh[adm.ADM_kmin]


                for l in range(adm.ADM_lall): ####
                    for k in range(adm.ADM_kmin - 1, kflat + 1):
                        for n in range(nstart, nend + 1):
                            self.GRD_vz[n, k, l, self.GRD_Z] = self.GRD_zs[n, k0, l, self.GRD_ZSFC] + \
                                (htop - self.GRD_zs[n, k0, l, self.GRD_ZSFC]) / htop * self.GRD_gz[k]
                            self.GRD_vz[n, k, l, self.GRD_ZH] = self.GRD_zs[n, k0, l, self.GRD_ZSFC] + \
                                (htop - self.GRD_zs[n, k0, l, self.GRD_ZSFC]) / htop * self.GRD_gzh[k]

                if kflat < adm.ADM_kmax + 1:
                    for k in range(kflat + 1, adm.ADM_kmax + 2):
                        for n in range(nstart, nend + 1):
                            self.GRD_vz[n, k, l, self.GRD_Z] = self.GRD_gz[k]
                            self.GRD_vz[n, k, l, self.GRD_ZH] = self.GRD_gzh[k]

                # Handle pole grid points
                if adm.ADM_have_pl:
                    n = adm.ADM_gslf_pl
                    for l in range(1, adm.ADM_lall_pl + 1):

                        for k in range(adm.ADM_kmin - 1, kflat + 1):  # Fortran includes upper bound, Python requires +1
                            self.GRD_vz_pl[n, k, l, self.GRD_Z] = self.GRD_zs_pl[n, k0, l, self.GRD_ZSFC] + \
                                (htop - self.GRD_zs_pl[n, k0, l, self.GRD_ZSFC]) / htop * self.GRD_gz[k]
    
                            self.GRD_vz_pl[n, k, l, self.GRD_ZH] = self.GRD_zs_pl[n, k0, l, self.GRD_ZSFC] + \
                                (htop - self.GRD_zs_pl[n, k0, l, self.GRD_ZSFC]) / htop * self.GRD_gzh[k]

                        # Handle case where kflat < ADM_kmax + 1
                        if kflat < adm.ADM_kmax + 1:
                            for k in range(kflat + 1, adm.ADM_kmax + 2):  # Fortran includes upper bound, Python requires +1
                                self.GRD_vz_pl[n, k, l, self.GRD_Z] = self.GRD_gz[k]
                                self.GRD_vz_pl[n, k, l, self.GRD_ZH] = self.GRD_gzh[k]

            elif self.vgrid_scheme == "HYBRID":
                #   Hybrid transformation : like as Simmons & Buridge(1981)

                for l in range(adm.ADM_lall):                 # ADM_kmin is 2 in f, 1 in p.  ADM_kmax is 41 in f, 40 in p. (when vlayer 40)
                    for k in range(adm.ADM_kmin - 1, adm.ADM_kmax + 2): #0 to 41 (exits at 42) # +2 to match Fortran upper bound behavior
                        #for n in range(nstart, nend + 1):
                        for i in range(adm.ADM_gmin, adm.ADM_gmax+1):  # loop for inner grid points
                            for j in range(adm.ADM_gmin, adm.ADM_gmax+1):  # loop for inner grid points
                                self.GRD_vz[i, j, k, l, self.GRD_Z] = self.GRD_gz[k] + \
                                    self.GRD_zs[i, j, k0, l, self.GRD_ZSFC] * \
                                    np.sinh((self.GRD_htop - self.GRD_gz[k]) / self.h_efold) / \
                                    np.sinh(self.GRD_htop / self.h_efold)

                                self.GRD_vz[i, j, k, l, self.GRD_ZH] = self.GRD_gzh[k] + \
                                    self.GRD_zs[i, j, k0, l, self.GRD_ZSFC] * \
                                    np.sinh((self.GRD_htop - self.GRD_gzh[k]) / self.h_efold) / \
                                    np.sinh(self.GRD_htop / self.h_efold)
                                if i==17 and j==0 and k==40 and l==1:
                                    with open(std.fname_log, 'a') as log_file:
                                        print("i=17, j=0, k=40, l=1, self.GRD_vz[i, j, k, l, self.GRD_Z]: ", self.GRD_vz[i, j, k, l, self.GRD_Z], file=log_file)
                                        print("i=17, j=0, k=40, l=1, self.GRD_vz[i, j, k, l, self.GRD_ZH]: ", self.GRD_vz[i, j, k, l, self.GRD_ZH], file=log_file)
                                        print("self.GRD_gzh[k]: ", self.GRD_gzh[k], file=log_file)
                                        print("self.GRD_zs[i, j, adm.ADM_KNONE, l, self.GRD_ZSFC]: ", self.GRD_zs[i, j, k0, l, self.GRD_ZSFC], file=log_file)
                                        print("self.GRD_htop: ", self.GRD_htop, file=log_file)
                                        print("self.h_efold: ", self.h_efold, file=log_file)
                                    #print("i=3, j=11, k=11, l=0, self.GRD_vz[i, j, k, l, self.GRD_Z]: ", self.GRD_vz[i, j, k, l, self.GRD_Z])
                                        # print("i=3, j=11, k=11, l=0, self.GRD_vz[i, j, k, l, self.GRD_ZH]: ", self.GRD_vz[i, j, k, l, self.GRD_ZH])
                                        # print("self.GRD_gzh[k]: ", self.GRD_gzh[k])
                                        # print("self.GRD_zs[i, j, k0, l, self.GRD_ZSFC]: ", self.GRD_zs[i, j, k0, l, self.GRD_ZSFC])
                                        # print("self.GRD_htop: ", self.GRD_htop)
                                        # print("self.h_efold: ", self.h_efold)
                                    #print("i=3, j=11, k=11, l=0, self.GRD_zs[i, j, k0, l, self.GRD_ZSFC]: ", self.GRD_zs[i, j, k0, l, self.GRD_ZSFC])
                                    #print("i=3, j=11, k=11, l=0, self.GRD_htop: ", self.GRD_htop)
                                    #print("i=3, j=11, k=11, l=0, self.h_efold: ", self.h_efold)
                                    #print("i=3, j=11, k=11, l=0, np.sinh((self.GRD_htop - self.GRD_gz[k]) / self.h_efold): ", np.sinh((self.GRD_htop - self.GRD_gz[k]) / self.h_efold))
                                    #print("i=3, j=11, k=11, l=0, np.sinh(self.GRD_htop / self.h_efold): ", np.sinh(self.GRD_htop / self.h_efold))
                                    #print("i=3, j=11, k=11, l=0, self.GRD_zs[i, j, k0, l, self.GRD_ZSFC]: ", self.GRD_zs[i, j, k0, l, self.GRD_ZSFC])
                                #CHECK!!

                # Handle pole grid points
                if adm.ADM_have_pl:
                    n = adm.ADM_gslf_pl
                    for l in range(adm.ADM_lall_pl):
                        for k in range(adm.ADM_kmin - 1, adm.ADM_kmax + 2): 
                            self.GRD_vz_pl[n, k, l, self.GRD_Z] = self.GRD_gz[k] + \
                                self.GRD_zs_pl[n, k0, l, self.GRD_ZSFC] * \
                                np.sinh((self.GRD_htop - self.GRD_gz[k]) / self.h_efold) / \
                                np.sinh(self.GRD_htop / self.h_efold)

                            self.GRD_vz_pl[n, k, l, self.GRD_ZH] = self.GRD_gzh[k] + \
                                self.GRD_zs_pl[n, k0, l, self.GRD_ZSFC] * \
                                np.sinh((self.GRD_htop - self.GRD_gzh[k]) / self.h_efold) / \
                                np.sinh(self.GRD_htop / self.h_efold)
                            
                            if k==40:
                                with open(std.fname_log, 'a') as log_file:
                                    print("n, k, l, self.GRD_vz_pl[n, k, l, self.GRD_Z]: ", n, k, l, self.GRD_vz_pl[n, k, l, self.GRD_Z], file=log_file)
                                    print("n, k, l, self.GRD_vz_pl[n, k, l, self.GRD_ZH]: ", n, k, l, self.GRD_vz_pl[n, k, l, self.GRD_ZH], file=log_file)
                                    print("self.GRD_gzh[k]: ", self.GRD_gzh[k], file=log_file)
                                    print("self.GRD_zs_pl[n, k0, l, self.GRD_ZSFC]: ", self.GRD_zs_pl[n, k0, l, self.GRD_ZSFC], file=log_file)
                                    print("self.GRD_htop: ", self.GRD_htop, file=log_file)
                                    print("self.h_efold: ", self.h_efold, file=log_file)
                                    ### GRD_zs_pl value differs from original code  ( 112.73991167342389 p vs 0.0 f)
                                

            # fill HALO
            comm.COMM_data_transfer(self.GRD_vz, self.GRD_vz_pl) 

        else:
            self.GRD_gz = np.ones(kn, dtype=np.float64)  # 1.0_RP assumed as float64
            self.GRD_gzh = np.ones(kn, dtype=np.float64)

        #"""Output information about the grid structure"""
        if adm.ADM_kall != adm.ADM_KNONE + 1:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("", file=log_file)
                    print("     |========      Vertical Coordinate [m]      ========|", file=log_file)
                    print("     |                                                   |", file=log_file)
                    print("     |            -GRID CENTER-         -GRID INTERFACE- |", file=log_file)
                    print("     |   k        gz      d(gz)      gzh     d(gzh)    k |", file=log_file)
                    print("     |                                                   |", file=log_file)
            
            # Output for top atmospheric layer (dummy)
            k = adm.ADM_kmax + 1
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"     | {k:3d} {self.GRD_gz[k]:10.1f} {self.GRD_dgz[k]:9.1f}                          | dummy", file=log_file)
                    print(f"     |                         {self.GRD_gzh[k]:10.1f} {self.GRD_dgzh[k]:9.1f} {k:4d} | TOA", file=log_file)

            # Output for kmax layer
            k = adm.ADM_kmax
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"     | {k:3d} {self.GRD_gz[k]:10.1f} {self.GRD_dgz[k]:9.1f}                          | kmax", file=log_file)
                    print(f"     |                         {self.GRD_gzh[k]:10.1f} {self.GRD_dgzh[k]:9.1f} {k:4d} |", file=log_file)

            # Loop through vertical layers in reverse order
            for k in range(adm.ADM_kmax - 1, adm.ADM_kmin, -1):
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print(f"     | {k:3d} {self.GRD_gz[k]:10.1f} {self.GRD_dgz[k]:9.1f}                          |", file=log_file)
                        print(f"     |                         {self.GRD_gzh[k]:10.1f} {self.GRD_dgzh[k]:9.1f} {k:4d} |", file=log_file)
                    
            # Output for kmin layer
            k = adm.ADM_kmin
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"     | {k:3d} {self.GRD_gz[k]:10.1f} {self.GRD_dgz[k]:9.1f}                          | kmin", file=log_file)
                    print(f"     |                         {self.GRD_gzh[k]:10.1f} {self.GRD_dgzh[k]:9.1f} {k:4d} | ground", file=log_file)
                
            # Output for bottom dummy layer
            k = adm.ADM_kmin - 1
            
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"     | {k:3d} {self.GRD_gz[k]:10.1f} {self.GRD_dgz[k]:9.1f}                          | dummy", file=log_file)
                    print(f"     |                         {self.GRD_gzh[k]:10.1f} {self.GRD_dgzh[k]:9.1f} {k:4d} | dummy", file=log_file)
                    print("     |===================================================|", file=log_file)
                    print("", file=log_file)
                    print(f"--- Vertical layer scheme = {self.vgrid_scheme.strip()}", file=log_file)
            
            # Additional information for HYBRID scheme
            if self.vgrid_scheme == 'HYBRID':
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print(f"--- e-folding height = {self.h_efold}", file=log_file)

            # Output vertical grid if required
            if self.output_vgrid and self.PRC_IsMaster:
                self.GRD_output_vgrid('./vgrid_used.dat')

        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("", file=log_file)
                    print("--- vertical layer = 1", file=log_file)


        #print("hoho_no, adm.ADM_prc_me", adm.ADM_prc_me)
        #prc.prc_mpistop(std.io_l, std.fname_log)

        # with open(std.fname_log, 'a') as log_file:
        #     print("GRD input check yahoo", file=log_file)

        # for i in range (0,4): #(adm.ADM_gall_1d):
        #     for j in range (0,4): #(adm.ADM_gall_1d):
        #         for l in range (adm.ADM_lall):
        #             if std.io_l:
        #                 with open(std.fname_log, 'a') as log_file:
        #                     print(f"GRD_x, abs:", self.GRD_x[i,j,0,l,0]**2 + self.GRD_x[i,j,0,l,1]**2 + self.GRD_x[i,j,0,l,2]**2, file=log_file)
        #                     print(f"i, j, l:", i, j, l, "xyz values:", file=log_file)
        #                     print(self.GRD_x[i,j,0,l,0] , self.GRD_x[i,j,0,l,1], self.GRD_x[i,j,0,l,2], file=log_file)

        # for i in range (14,18): #(adm.ADM_gall_1d):
        #     for j in range (14,18): #(adm.ADM_gall_1d):
        #         for l in range (adm.ADM_lall):
        #             if std.io_l:
        #                 with open(std.fname_log, 'a') as log_file:
        #                     print(f"GRD_x, abs:", self.GRD_x[i,j,0,l,0]**2 + self.GRD_x[i,j,0,l,1]**2 + self.GRD_x[i,j,0,l,2]**2, file=log_file)
        #                     print(f"i, j, l:", i, j, l, "xyz values:", file=log_file)
        #                     print(self.GRD_x[i,j,0,l,0] , self.GRD_x[i,j,0,l,1], self.GRD_x[i,j,0,l,2], file=log_file)

                                                  
        return


    def GRD_input_hgrid(self, fname, lrvertex, io_mode, comm, rdtype):

        if io_mode == "json":
            import json
            fullname = fname+str(prc.prc_myrank).zfill(8)+".json"
            with open(fullname, "r") as json_file:
                #loaded_data = json.load(json_file)
                data_arrays = json.load(json_file)

            #print("Datasets in JSON file:", list(data_arrays.keys()))


            grd_x_x= np.array(data_arrays["grd_x_x"]) 
            grd_x_y= np.array(data_arrays["grd_x_y"]) 
            grd_x_z= np.array(data_arrays["grd_x_z"]) 

            for l in range(adm.ADM_lall): # 0 to 4
                for i in range(adm.ADM_gall_1d):      # 0 to 17
                    for j in range(adm.ADM_gall_1d):  # 0 to 17
                        ij=self.suf(i,j)
                        self.GRD_x[i,j,0,l,0] = grd_x_x[l,ij]
                        self.GRD_x[i,j,0,l,1] = grd_x_y[l,ij]
                        self.GRD_x[i,j,0,l,2] = grd_x_z[l,ij]

            if lrvertex:
                grd_xt_ix= np.array(data_arrays["grd_xt_ix"]) 
                grd_xt_jx= np.array(data_arrays["grd_xt_jx"]) 
                grd_xt_iy= np.array(data_arrays["grd_xt_iy"]) 
                grd_xt_jy= np.array(data_arrays["grd_xt_jy"]) 
                grd_xt_iz= np.array(data_arrays["grd_xt_iz"]) 
                grd_xt_jz= np.array(data_arrays["grd_xt_jz"]) 

                for l in range(adm.ADM_lall): # 0 to 4
                    for i in range(adm.ADM_gall_1d):      # 0 to 17
                        for j in range(adm.ADM_gall_1d):  # 0 to 17
                            ij=self.suf(i,j)
                            self.GRD_xt[i,j,0,l,0,0] = grd_xt_ix[l,ij]
                            self.GRD_xt[i,j,0,l,1,0] = grd_xt_jx[l,ij]
                            self.GRD_xt[i,j,0,l,0,1] = grd_xt_iy[l,ij]
                            self.GRD_xt[i,j,0,l,1,1] = grd_xt_jy[l,ij]
                            self.GRD_xt[i,j,0,l,0,2] = grd_xt_iz[l,ij]
                            self.GRD_xt[i,j,0,l,1,2] = grd_xt_jz[l,ij]

        else:
            print("sorry, other data types are under construction")
            prc.prc_mpistop(std.io_l, std.fname_log)

        self.GRD_gen_plgrid(comm,rdtype)

        # with open(std.fname_log, 'a') as log_file:
        #     print("GRD input check", file=log_file)
        #     print("LOOK", file=log_file)
        #     print("GRD_x,    l0:", self.GRD_x[:,:,0,0,0]**2 + self.GRD_x[:,:,0,0,1]**2 + self.GRD_x[:,:,0,0,2]**2, file=log_file)
        #     print("GRD_xt_i, l0:", self.GRD_xt[:,:,0,0,0,0]**2 + self.GRD_xt[:,:,0,0,0,1]**2 + self.GRD_xt[:,:,0,0,0,2]**2, file=log_file)
        #     print("GRD_xt_j, l0:", self.GRD_xt[:,:,0,0,1,0]**2 + self.GRD_xt[:,:,0,0,1,1]**2 + self.GRD_xt[:,:,0,0,1,2]**2, file=log_file)
        #     print("GRD_x, l1:", self.GRD_x[:,:,0,1,0]**2 + self.GRD_x[:,:,0,1,1]**2 + self.GRD_x[:,:,0,1,2]**2, file=log_file)  
        #     print("GRD_x, l2:", self.GRD_x[:,:,0,2,0]**2 + self.GRD_x[:,:,0,2,1]**2 + self.GRD_x[:,:,0,2,2]**2, file=log_file)  
        #     print("GRD_x, l3:", self.GRD_x[:,:,0,3,0]**2 + self.GRD_x[:,:,0,3,1]**2 + self.GRD_x[:,:,0,3,2]**2, file=log_file)  
        #     print("GRD_x, l4:", self.GRD_x[:,:,0,4,0]**2 + self.GRD_x[:,:,0,4,1]**2 + self.GRD_x[:,:,0,4,2]**2, file=log_file)  
        
        #self.GRD_gen_plgrid(comm,rdtype)

        #print("hoho2, adm.ADM_prc_me", adm.ADM_prc_me)
        #prc.prc_mpistop(std.io_l, std.fname_log)

        return


    def suf(self, i, j):
        return adm.ADM_gall_1d * j + i 
    

    def GRD_scaling(self, fact):

        self.GRD_x *= fact
        self.GRD_xt *= fact

        if adm.ADM_have_pl:
            #print("GRD_x_pl, before:", self.GRD_x_pl)
            #print("fact", fact)
            self.GRD_x_pl *= fact
            self.GRD_xt_pl *= fact
            #print("GRD_x_pl, mid:", self.GRD_x_pl)   


        if self.GRD_grid_type == self.GRD_grid_type_on_plane:
            pass  # Do nothing
        else:
            self.GRD_rscale = fact  # Set sphere radius scaling factor

        return
    

    def GRD_makelatlon(self, cnst): # <vectorized by a.kamiijo on 2025.04.01>
        k0 = adm.ADM_K0  # index for single layer
        # Vectorized conversion for cell centers
        x_arr = self.GRD_x[:, :, k0, :, 0]
        y_arr = self.GRD_x[:, :, k0, :, 1]
        z_arr = self.GRD_x[:, :, k0, :, 2]
        lat_arr, lon_arr = vect.VECTR_xyz2latlon_vec(x_arr, y_arr, z_arr, cnst)
        self.GRD_s[:, :, k0, :, 0] = lat_arr
        self.GRD_s[:, :, k0, :, 1] = lon_arr
        self.GRD_LAT = lat_arr
        self.GRD_LON = lon_arr

        # Vectorized conversion for time-dependent grid points
        x_arr0 = self.GRD_xt[:, :, k0, :, 0, 0]
        y_arr0 = self.GRD_xt[:, :, k0, :, 0, 1]
        z_arr0 = self.GRD_xt[:, :, k0, :, 0, 2]
        lat_arr0, lon_arr0 = vect.VECTR_xyz2latlon_vec(x_arr0, y_arr0, z_arr0, cnst)
        self.GRD_st[:, :, k0, :, 0, 0] = lat_arr0
        self.GRD_st[:, :, k0, :, 0, 1] = lon_arr0

        x_arr1 = self.GRD_xt[:, :, k0, :, 1, 0]
        y_arr1 = self.GRD_xt[:, :, k0, :, 1, 1]
        z_arr1 = self.GRD_xt[:, :, k0, :, 1, 2]
        lat_arr1, lon_arr1 = vect.VECTR_xyz2latlon_vec(x_arr1, y_arr1, z_arr1, cnst)
        self.GRD_st[:, :, k0, :, 1, 0] = lat_arr1
        self.GRD_st[:, :, k0, :, 1, 1] = lon_arr1

        if adm.ADM_have_pl:
            x_arr_pl = self.GRD_x_pl[:, k0, :, 0]
            y_arr_pl = self.GRD_x_pl[:, k0, :, 1]
            z_arr_pl = self.GRD_x_pl[:, k0, :, 2]
            lat_arr_pl, lon_arr_pl = vect.VECTR_xyz2latlon_vec(x_arr_pl, y_arr_pl, z_arr_pl, cnst)
            self.GRD_s_pl[:, k0, :, 0] = lat_arr_pl
            self.GRD_s_pl[:, k0, :, 1] = lon_arr_pl
            self.GRD_LAT_pl = lat_arr_pl
            self.GRD_LON_pl = lon_arr_pl

            x_arr_pl_t0 = self.GRD_xt_pl[:, k0, :, 0]
            y_arr_pl_t0 = self.GRD_xt_pl[:, k0, :, 1]
            z_arr_pl_t0 = self.GRD_xt_pl[:, k0, :, 2]
            lat_arr_pl_t0, lon_arr_pl_t0 = vect.VECTR_xyz2latlon_vec(x_arr_pl_t0, y_arr_pl_t0, z_arr_pl_t0, cnst)
            self.GRD_st_pl[:, k0, :, 0] = lat_arr_pl_t0
            self.GRD_st_pl[:, k0, :, 1] = lon_arr_pl_t0
        return
    

    def GRD_makearc(self):
        """
        Calculate the mid-point locations of cell arcs.
        """
        k0 = adm.ADM_K0 # k0 is used as the index of the single layer

        for l in range(self.GRD_xt.shape[3]):  # Loop over layers
            # First loop

                      # (1 in f, 0 in p)   (2 in f, 1 in p)        gl05rl01
            #nstart = self.suf(self.ADM_gmin - 1, self.ADM_gmin)
                        #(17 in f, 16 in p)  (17 in f, 16 in p)    gl05rl01
            #nend = self.suf(self.ADM_gmax, self.ADM_gmax)

            for i in range(adm.ADM_gmax+1):  # 0 to 17  gl05rl01
                for j in range(1, adm.ADM_gmax+1):  # 1 to 17  gl05rl01
            #for n in range(nstart, nend + 1):
            #    ij = n
            #    ijm1 = n - self.ADM_gall_1d

                        self.GRD_xr[i, j, k0, l, 0, 0] = 0.5 * (self.GRD_xt[i, j-1, k0, l, 1, 0] + self.GRD_xt[i, j, k0, l, 0, 0])
                        self.GRD_xr[i, j, k0, l, 0, 1] = 0.5 * (self.GRD_xt[i, j-1, k0, l, 1, 1] + self.GRD_xt[i, j, k0, l, 0, 1])
                        self.GRD_xr[i, j, k0, l, 0, 2] = 0.5 * (self.GRD_xt[i, j-1, k0, l, 1, 2] + self.GRD_xt[i, j, k0, l, 0, 2])

            # Second loop
            #nstart = self.suf(self.ADM_gmin - 1, self.ADM_gmin - 1)
            #nend = self.suf(self.ADM_gmax, self.ADM_gmax)

            #for n in range(nstart, nend + 1):
            #    ij = n

            for i in range(adm.ADM_gmax+1):  # 0 to 17  gl05rl01
                for j in range(adm.ADM_gmax+1):  # 0 to 17  gl05rl01

                    self.GRD_xr[i, j, k0, l, 2, 0] = 0.5 * (self.GRD_xt[i, j, k0, l, 0, 0] + self.GRD_xt[i, j, k0, l, 1, 0])
                    self.GRD_xr[i, j, k0, l, 2, 1] = 0.5 * (self.GRD_xt[i, j, k0, l, 0, 1] + self.GRD_xt[i, j, k0, l, 1, 1])
                    self.GRD_xr[i, j, k0, l, 2, 2] = 0.5 * (self.GRD_xt[i, j, k0, l, 0, 2] + self.GRD_xt[i, j, k0, l, 1, 2])

            # Third loop
            #nstart = self.suf(self.ADM_gmin, self.ADM_gmin - 1)
            #nend = self.suf(self.ADM_gmax, self.ADM_gmax)

            #for n in range(nstart, nend + 1):
            #    ij = n
            #    im1j = n - 1

            for i in range(1, adm.ADM_gmax+1):  # 1 to 17  gl05rl01
                for j in range(adm.ADM_gmax+1):  # 0 to 17  gl05rl01

                    self.GRD_xr[i, j, k0, l, 1, 0] = 0.5 * (self.GRD_xt[i, j, k0, l, 1, 0] + self.GRD_xt[i-1, j, k0, l, 0, 0])
                    self.GRD_xr[i, j, k0, l, 1, 1] = 0.5 * (self.GRD_xt[i, j, k0, l, 1, 1] + self.GRD_xt[i-1, j, k0, l, 0, 1])
                    self.GRD_xr[i, j, k0, l, 1, 2] = 0.5 * (self.GRD_xt[i, j, k0, l, 1, 2] + self.GRD_xt[i-1, j, k0, l, 0, 2])

        if adm.ADM_have_pl:
            for l in range(self.GRD_xr_pl.shape[2]):
                #             (2 in f, 1 in p)     
                for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1): #  1 to 6 in p   gl05rl01
                    ij = v
                    ijm1 = v - 1
                    if ijm1 == adm.ADM_gmin_pl - 1:
                        ijm1 = adm.ADM_gmax_pl  # Wrap around   0 -> 6  gl05rl01

                    self.GRD_xr_pl[v, k0, l, 0] = 0.5 * (self.GRD_xt_pl[ijm1, k0, l, 0] + self.GRD_xt_pl[ij, k0, l, 0])
                    self.GRD_xr_pl[v, k0, l, 1] = 0.5 * (self.GRD_xt_pl[ijm1, k0, l, 1] + self.GRD_xt_pl[ij, k0, l, 1])
                    self.GRD_xr_pl[v, k0, l, 2] = 0.5 * (self.GRD_xt_pl[ijm1, k0, l, 2] + self.GRD_xt_pl[ij, k0, l, 2])

        return
    

    def GRD_input_topograph(self, fname_in, fname, fname_sd, io_mode, cnst, comm):
        
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("*** Topography data input", file=log_file)

        if io_mode == 'json':
            if self.topo_basename != 'NONE':
                print("json file is not supported yet")
        
        elif io_mode == 'IDEAL':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("*** Make ideal topography", file=log_file)

            from mod_ideal_topo import Idt
            idt = Idt()
            #self.IDEAL_topo(self.GRD_s[:, :, :, :, 0],  # Latitude
            #                self.GRD_s[:, :, :, :, 1],  # Longitude
            #                self.GRD_zs[:, :, :, :, 0])  # Topography output
            #self.GRD_zs[:, :, :, :, self.GRD_ZSFC] = 

            #n = adm.ADM_gslf_pl
            #for l in range(adm.ADM_lall_pl):
            # with open(std.fname_log, 'a') as log_file:
            #     print("BEFORE IDEAL_topo, self.GRD_s[1, 17, 0, 2, self.I_LAT]: ", self.GRD_s[1, 17, 0, 2, self.I_LAT], file=log_file)
            #     print("BEFORE IDEAL_topo, self.GRD_s[1, 17, 0, 2, self.I_LON]: ", self.GRD_s[1, 17, 0, 2, self.I_LON], file=log_file)   


            idt.IDEAL_topo(fname_in, 
                           self.GRD_s[:, :, :, :, self.I_LAT], 
                           self.GRD_s[:, :, :, :, self.I_LON], 
                           self.GRD_zs[:, :, :, :, self.GRD_ZSFC], 
                           cnst)
            ###  
            # with open (std.fname_log, 'a') as log_file:
            #     print(self.GRD_s.shape, "LAT, LON, zsZSFC", file=log_file)
            #     print(self.GRD_s[3,11,0,0,self.I_LAT],file= log_file)
            #     print(self.GRD_s[3,11,0,0,self.I_LON],file= log_file)
            #     print(self.GRD_zs[3,11,0,0,self.GRD_ZSFC],file= log_file)

        else:
            print("sorry, other data types are under construction")
            print("or perhaps")
            print("xxx [grd/GRD_input_topograph] Invalid io_mode!")
            prc.prc_mpistop(std.io_l, std.fname_log)

        # if prc.prc_myrank == 0:
        #     with open(std.fname_log, 'a') as log_file:
        #         print("GRD_zs  at r2p point, BEFORE COMM_var", self.GRD_zs[1,17,0,2,self.GRD_ZSFC], file=log_file)


        # n = adm.ADM_gslf_pl
        # for l in range(adm.ADM_lall_pl):
        #     with open(std.fname_log, 'a') as log_file:
        #         print("BEFORE COMM_Var, self.GRD_zs_pl[n, k0, l, self.GRD_ZSFC]: ", l, self.GRD_zs_pl[n, 0, l, self.GRD_ZSFC], file=log_file)

        comm.COMM_var(self.GRD_zs, self.GRD_zs_pl)

#         if prc.prc_myrank == 0:
#             with open(std.fname_log, 'a') as log_file:
#                 print("GRD_zs  at r2p point, AFTER COMM_var", self.GRD_zs[1,17,0,2,self.GRD_ZSFC], file=log_file)
# #            print("GRD_zs  at r2p point, AFTER COMM_var", self.GRD_zs[1,17,0,2,self.GRD_ZSFC])


        # n = adm.ADM_gslf_pl
        # for l in range(adm.ADM_lall_pl):
        #     with open(std.fname_log, 'a') as log_file:
        #         print("AFTER COMM_Var, self.GRD_zs_pl[n, k0, l, self.GRD_ZSFC]: ", l, self.GRD_zs_pl[n, 0, l, self.GRD_ZSFC], file=log_file)

        # with open (std.fname_log, 'a') as log_file:
        #     print("After COMM_var", file=log_file)
        #     print(self.GRD_s.shape, "zsZSFC", file=log_file)
        #     print(self.GRD_zs[3,11,0,0,self.GRD_ZSFC],file= log_file)

        return
    
    def GRD_input_vgrid(self, fname):
        import json

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print('*** Read vertical grid file: ', fname, file=log_file)

        json_file_path = fname
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

            self.GRD_gz  = np.array(data["set1"])
            self.GRD_gzh = np.array(data["set2"])

            #print("self.GRD_gz", self.GRD_gz, self.GRD_gz.shape)
            #print("self.GRD_gzh", self.GRD_gzh, self.GRD_gzh.shape)



        return
    
    def GRD_gen_plgrid(self, comm, rdtype):

        k0 = adm.ADM_K0  # k0 is used as the index of the single layer
    
        prctab = np.zeros(adm.ADM_vlink, dtype=int)
        rgntab = np.zeros(adm.ADM_vlink, dtype=int)

        # Logical (Boolean) array
        send_flag = np.zeros(adm.ADM_vlink, dtype=bool)

        # Real (floating-point) arrays
        #vsend_pl = np.zeros((adm.ADM_nxyz, adm.ADM_vlink), dtype=rdtype)  # Assuming RP = double precision
        vsend_pl = np.zeros((adm.ADM_nxyz), dtype=rdtype)  # Assuming RP = double precision
        #vrecv_pl = np.zeros((adm.ADM_nxyz), dtype=rdtype)  # Assuming RP = double precision


        # --- Find the region that contains the North Pole ---
                            # 40
        for l in range(adm.ADM_rgn_nmax -1, -1, -1):  # Reverse loop from ADM_rgn_nmax-1 to 0
                                                    # 5
            if adm.RGNMNG_vert_num[adm.I_N, l] == adm.ADM_vlink:  # when number of vertices is 5 (NPL or SPL) 
                for n in range(adm.ADM_vlink):  # 0 to 4
                    nn = n + 1
                    if nn > adm.ADM_vlink -1 :   # 5 > 4 when n = 4
                        nn = 0
                    rgntab[n] = adm.RGNMNG_vert_tab[adm.I_RGNID, adm.I_N, l, nn]
                    prctab[n] = adm.RGNMNG_r2lp[adm.I_prc, rgntab[n]]
                    # if std.io_l:
                    #     with open(std.fname_log, 'a') as log_file:
                    #         print("search north", n, nn, rgntab[n], prctab[n], file=log_file)
                break  

        # Initialize send flags
        send_flag[:] = False
        recv_slices = []
        send_requests = []
        recv_requests = []

        # --- Receive data ---
        if adm.ADM_prc_me == adm.RGNMNG_r2p_pl[adm.I_NPL]:
            for n in range(adm.ADM_vlink):
                vrecv_pl = np.empty((adm.ADM_nxyz), dtype=rdtype) 
                vrecv_pl = np.ascontiguousarray(vrecv_pl)
                recv_slices.append(vrecv_pl)
                
                # if std.io_l:
                #     with open(std.fname_log, 'a') as log_file:  
                #         print("receiving north", n, prctab[n], rgntab[n], file=log_file)

                recv_requests.append(prc.comm_world.Irecv(recv_slices[n], source=prctab[n], tag=rgntab[n]))
                #recv_requests.append(req)
                #req = prc.comm_world.Irecv(recv_slices[n], source=prctab[n], tag=rgntab[n])
                #recv_requests.append(req)

        # --- Send grid position from regular region ---
        # if std.io_l:
        #     with open(std.fname_log, 'a') as log_file:
        #         print("rgntab",rgntab[:], file=log_file)
        for n in range(adm.ADM_vlink):
            for l in range(adm.ADM_lall):
                if adm.RGNMNG_lp2r[l, adm.ADM_prc_me] == rgntab[n]:
                    vsend_pl[:] = self.GRD_xt[adm.ADM_gmin, adm.ADM_gmax, k0, l, adm.ADM_TJ, :]
                    #vsend_pl[:] = self.GRD_xt[adm.ADM_gmin, adm.ADM_gmax+1, k0, l, adm.ADM_TJ, :]
                    vsend_pl[:] = np.ascontiguousarray(vsend_pl[:])    

                    #print("sending to NPL: myrank, n, l, vsend_pl ")
                    #print(prc.prc_myrank, n, l, vsend_pl)

                    req = prc.comm_world.Isend(vsend_pl[:], dest=adm.RGNMNG_r2p_pl[adm.I_NPL], tag=rgntab[n])
                    send_requests.append(req)
                    send_flag[n] = True
                    # if std.io_l:
                    #     with open(std.fname_log, 'a') as log_file:
                    #         print("sending north", n, l, adm.RGNMNG_r2p_pl[adm.I_NPL], rgntab[n], file=log_file) 
                    #         print(adm.ADM_gmin, adm.ADM_gmax, adm.ADM_KNONE, l, adm.ADM_TJ, file=log_file)
                    #         #print(adm.ADM_gmin, adm.ADM_gmax+1, adm.ADM_KNONE, l, adm.ADM_TJ, file=log_file)
                    #         print(vsend_pl[:], file=log_file)

        for n in range(adm.ADM_vlink):
            if send_flag[n]:
                MPI.Request.Waitall(send_requests)

        if adm.ADM_prc_me == adm.RGNMNG_r2p_pl[adm.I_NPL]:
            MPI.Request.Waitall(recv_requests)
            for n in range(adm.ADM_vlink):
                self.GRD_xt_pl[n+1, k0, adm.I_NPL, :] = recv_slices[n]    # keeping index 0 open for pole value

                # if std.io_l:
                #     with open(std.fname_log, 'a') as log_file:
                #         print("unpacking north", n, adm.ADM_prc_me, file=log_file)
                #         print(recv_slices[n], file=log_file)


        # --- Find the region that contains the South Pole ---
                            # 40
        for l in range(adm.ADM_rgn_nmax -1, -1, -1):  # Reverse loop from ADM_rgn_nmax-1 to 0
                                                    # 5
            if adm.RGNMNG_vert_num[adm.I_S, l] == adm.ADM_vlink:  # when number of vertices is 5 (NPL or SPL) 
                for n in range(adm.ADM_vlink):  # 0 to 4
                    rgntab[n] = adm.RGNMNG_vert_tab[adm.I_RGNID, adm.I_S, l, n]
                    prctab[n] = adm.RGNMNG_r2lp[adm.I_prc, rgntab[n]]
                    # if std.io_l:
                    #     with open(std.fname_log, 'a') as log_file:
                    #         print("search south", n, rgntab[n], prctab[n], file=log_file)
                break  

        # Initialize send flags
        send_flag[:] = False
        recv_slices = []
        send_requests = []
        recv_requests = []

        # --- Receive data ---
        if adm.ADM_prc_me == adm.RGNMNG_r2p_pl[adm.I_SPL]:
            for n in range(adm.ADM_vlink):
                vrecv_pl = np.empty((adm.ADM_nxyz), dtype=rdtype) 
                vrecv_pl = np.ascontiguousarray(vrecv_pl)
                recv_slices.append(vrecv_pl)

                # if std.io_l:
                #     with open(std.fname_log, 'a') as log_file: 
                #         print("receiving south", n, prctab[n], rgntab[n], file=log_file)
                recv_requests.append(prc.comm_world.Irecv(recv_slices[n], source=prctab[n], tag=rgntab[n]))
                #req = prc.comm_world.Irecv(recv_slices[n], source=prctab[n], tag=rgntab[n])
                #recv_requests.append(req)

        # --- Send grid position from regular region ---
        # if std.io_l:
        #     with open(std.fname_log, 'a') as log_file:
        #         print("rgntab",rgntab[:], file=log_file)
        for n in range(adm.ADM_vlink):
            for l in range(adm.ADM_lall):
                if adm.RGNMNG_lp2r[l, adm.ADM_prc_me] == rgntab[n]:
                    vsend_pl[:] = self.GRD_xt[adm.ADM_gmax, adm.ADM_gmin, k0, l, adm.ADM_TI, :]
                    vsend_pl[:] = np.ascontiguousarray(vsend_pl[:])    
                    req = prc.comm_world.Isend(vsend_pl[:], dest=adm.RGNMNG_r2p_pl[adm.I_SPL], tag=rgntab[n])
                    send_requests.append(req)
                    send_flag[n] = True
                    # if std.io_l:
                    #     with open(std.fname_log, 'a') as log_file:
                    #         print("sending south", n, l, adm.RGNMNG_r2p_pl[adm.I_SPL], rgntab[n], file=log_file) 
                    #         print(adm.ADM_gmax, adm.ADM_gmin, adm.ADM_KNONE, l, adm.ADM_TI, file=log_file)
                    #         print(vsend_pl[:], file=log_file)

        for n in range(adm.ADM_vlink):
            if send_flag[n]:
                MPI.Request.Waitall(send_requests)

        if adm.ADM_prc_me == adm.RGNMNG_r2p_pl[adm.I_SPL]:
            MPI.Request.Waitall(recv_requests)
            # if std.io_l:
            #     with open(std.fname_log, 'a') as log_file:  
            #         print(recv_slices, file=log_file)
            for n in range(adm.ADM_vlink):
                self.GRD_xt_pl[n+1, k0, adm.I_SPL, :] = recv_slices[n]     # keeping index 0 open for pole value

                # if std.io_l:
                #     with open(std.fname_log, 'a') as log_file:
                #         print("unpacking south", n, adm.ADM_prc_me, file=log_file)
                #         print(recv_slices[n], file=log_file)



        comm.COMM_var(self.GRD_x, self.GRD_x_pl)

        # in the above, received data should be unpacked in to index 1 to 5 of self.GRD_xt_pl
        # index 0 of self.GRD_xt_pl will be overwritten by the line below with the pole value from self.GRD_x_pl

        if adm.ADM_prc_me == adm.RGNMNG_r2p_pl[adm.I_NPL] or adm.ADM_prc_me == adm.RGNMNG_r2p_pl[adm.I_SPL]:
            self.GRD_xt_pl[adm.ADM_gslf_pl, :, :, :] = self.GRD_x_pl[adm.ADM_gslf_pl, :, :, :]

        return