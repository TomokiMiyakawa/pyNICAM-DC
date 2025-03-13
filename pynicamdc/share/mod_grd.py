import toml
import numpy as np
#from mpi4py import MPI
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

    def GRD_setup(self, fname_in, cnst, comm):
        #self._grd = self._grd_setup()

        k0 = adm.ADM_KNONE  

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
        self.GRD_x     = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d,    k0, adm.ADM_lall,                              adm.ADM_nxyz), cnst.CONST_UNDEF)
        
        self.GRD_x_pl  = np.full((adm.ADM_gall_pl, k0, adm.ADM_lall_pl,                                               adm.ADM_nxyz), cnst.CONST_UNDEF)
        self.GRD_xt    = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d, k0, adm.ADM_lall,    adm.ADM_TJ - adm.ADM_TI + 1, adm.ADM_nxyz), cnst.CONST_UNDEF)
        self.GRD_xt_pl = np.full((adm.ADM_gall_pl, k0, adm.ADM_lall_pl,                                               adm.ADM_nxyz), cnst.CONST_UNDEF)
        self.GRD_xr    = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d, k0, adm.ADM_lall,    adm.ADM_AJ - adm.ADM_AI + 1, adm.ADM_nxyz), cnst.CONST_UNDEF)
        self.GRD_xr_pl = np.full((adm.ADM_gall_pl,                  k0, adm.ADM_lall_pl,                              adm.ADM_nxyz), cnst.CONST_UNDEF)

        self.GRD_s     = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d, k0, adm.ADM_lall,                                 2), cnst.CONST_UNDEF)
        self.GRD_s_pl  = np.full((adm.ADM_gall_pl,                  k0, adm.ADM_lall_pl,                              2), cnst.CONST_UNDEF)
        self.GRD_st    = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d, k0, adm.ADM_lall,    adm.ADM_TJ - adm.ADM_TI + 1, 2), cnst.CONST_UNDEF)
        self.GRD_st_pl = np.full((adm.ADM_gall_pl,                  k0, adm.ADM_lall_pl,                              2), cnst.CONST_UNDEF)

        self.GRD_LAT   = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d,  adm.ADM_lall),    cnst.CONST_UNDEF)
        self.GRD_LAT_pl= np.full((adm.ADM_gall_pl, adm.ADM_lall_pl),                   cnst.CONST_UNDEF)
        self.GRD_LON   = np.full((adm.ADM_gall_1d, adm.ADM_gall_1d,  adm.ADM_lall),    cnst.CONST_UNDEF)
        self.GRD_LON_pl= np.full((adm.ADM_gall_pl, adm.ADM_lall_pl),                   cnst.CONST_UNDEF)

        
        self.GRD_input_hgrid(self.hgrid_fname, True, self.hgrid_io_mode)  

        # Data transfer for self.GRD_x (excluding self.GRD_xt)
        if self.hgrid_comm_flg:
            comm.COMM_data_transfer(self.GRD_x, self.GRD_x_pl)  

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
        self.GRD_zs     = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, k0, adm.ADM_lall,    self.GRD_ZSD - self.GRD_ZSFC + 1))
        self.GRD_zs_pl  = np.zeros((adm.ADM_gall_pl,                  k0, adm.ADM_lall_pl, self.GRD_ZSD - self.GRD_ZSFC + 1))

        # Call function to read topographic data (assuming function exists)
        self.GRD_input_topograph(fname_in, self.topo_fname, self.toposd_fname, self.topo_io_mode, cnst, comm)
        print("hoho?")
        prc.prc_mpifinish(std.io_l, std.fname_log)
        import sys
        sys.exit()

        # ---< Vertical Coordinate >---
        if adm.ADM_kall != adm.ADM_KNONE:
            self.GRD_gz   = np.zeros(adm.ADM_kall)
            self.GRD_gzh  = np.zeros(adm.ADM_kall)
            self.GRD_dgz  = np.zeros(adm.ADM_kall)
            self.GRD_dgzh = np.zeros(adm.ADM_kall)
            self.GRD_rdgz = np.zeros(adm.ADM_kall)
            self.GRD_rdgzh = np.zeros(adm.ADM_kall)

            self.GRD_afact = np.zeros(adm.ADM_kall)
            self.GRD_bfact = np.zeros(adm.ADM_kall)
            self.GRD_cfact = np.zeros(adm.ADM_kall)
            self.GRD_dfact = np.zeros(adm.ADM_kall)

            self.GRD_vz    = np.zeros((adm.ADM_gall,    adm.ADM_kall, adm.ADM_lall,    self.GRD_Z - self.GRD_ZH + 1))
            self.GRD_vz_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kall, adm.ADM_lall_pl, self.GRD_Z - self.GRD_ZH + 1))

            self.GRD_input_vgrid(self.vgrid_fname)

            # --- Calculation of grid intervals (cell center) ---
            for k in range(adm.ADM_kmin - 1, adm.ADM_kmax):
                self.GRD_dgz[k] = self.GRD_gzh[k + 1] - self.GRD_gzh[k]

            self.GRD_dgz[adm.ADM_kmax + 1] = self.GRD_dgz[adm.ADM_kmax]

            # --- Calculation of grid intervals (cell wall) ---
            for k in range(adm.ADM_kmin, adm.ADM_kmax + 2):  # +1 in Fortran means +2 in Python due to 0-based indexing
                self.GRD_dgzh[k] = self.GRD_gz[k] - self.GRD_gz[k - 1]

            self.GRD_dgzh[adm.ADM_kmin - 1] = self.GRD_dgzh[adm.ADM_kmin]

            # Compute inverse grid spacing
            for k in range(adm.ADM_kall):
                self.GRD_rdgz[k]  = 1.0 / self.GRD_dgz[k]
                self.GRD_rdgzh[k] = 1.0 / self.GRD_dgzh[k]

            # Compute height top
            self.GRD_htop = self.GRD_gzh[adm.ADM_kmax + 1] - self.GRD_gzh[adm.ADM_kmin]

            # Compute vertical interpolation factor
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
            nstart = self.suf(adm.ADM_gmin, adm.ADM_gmin)
            nend   = self.suf(adm.ADM_gmax, adm.ADM_gmax)

            # --- Select Vertical Grid Scheme ---
            if self.vgrid_scheme == "LINEAR":

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


                for l in range(1, adm.ADM_lall + 1):
                    for k in range(adm.ADM_kmin - 1, kflat + 1):
                        for n in range(nstart, nend + 1):
                            self.GRD_vz[n, k, l, self.GRD_Z] = self.GRD_zs[n, adm.ADM_KNONE, l, self.GRD_ZSFC] + \
                                (htop - self.GRD_zs[n, adm.ADM_KNONE, l, self.GRD_ZSFC]) / htop * self.GRD_gz[k]
                            self.GRD_vz[n, k, l, self.GRD_ZH] = self.GRD_zs[n, adm.ADM_KNONE, l, self.GRD_ZSFC] + \
                                (htop - self.GRD_zs[n, adm.ADM_KNONE, l, self.GRD_ZSFC]) / htop * self.GRD_gzh[k]

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
                            self.GRD_vz_pl[n, k, l, self.GRD_Z] = self.GRD_zs_pl[n, adm.ADM_KNONE, l, self.GRD_ZSFC] + \
                                (htop - self.GRD_zs_pl[n, adm.ADM_KNONE, l, self.GRD_ZSFC]) / htop * self.GRD_gz[k]
    
                            self.GRD_vz_pl[n, k, l, self.GRD_ZH] = self.GRD_zs_pl[n, adm.ADM_KNONE, l, self.GRD_ZSFC] + \
                                (htop - self.GRD_zs_pl[n, adm.ADM_KNONE, l, self.GRD_ZSFC]) / htop * self.GRD_gzh[k]

                        # Handle case where kflat < ADM_kmax + 1
                        if kflat < adm.ADM_kmax + 1:
                            for k in range(kflat + 1, adm.ADM_kmax + 2):  # Fortran includes upper bound, Python requires +1
                                self.GRD_vz_pl[n, k, l, self.GRD_Z] = self.GRD_gz[k]
                                self.GRD_vz_pl[n, k, l, self.GRD_ZH] = self.GRD_gzh[k]

            elif self.vgrid_scheme == "HYBRID":
                #   Hybrid transformation : like as Simmons & Buridge(1981)

                for l in range(1, adm.ADM_lall + 1):  # Fortran loops include the upper bound
                    for k in range(adm.ADM_kmin - 1, adm.ADM_kmax + 2):  # +2 to match Fortran upper bound behavior
                        for n in range(nstart, nend + 1):
                            self.GRD_vz[n, k, l, self.GRD_Z] = self.GRD_gz[k] + \
                                self.GRD_zs[n, adm.ADM_KNONE, l, self.GRD_ZSFC] * \
                                np.sinh((self.GRD_htop - self.GRD_gz[k]) / self.h_efold) / np.sinh(self.GRD_htop / self.h_efold)

                            self.GRD_vz[n, k, l, self.GRD_ZH] = self.GRD_gzh[k] + \
                                self.GRD_zs[n, adm.ADM_KNONE, l, self.GRD_ZSFC] * \
                                np.sinh((self.GRD_htop - self.GRD_gzh[k]) / self.h_efold) / np.sinh(self.GRD_htop / self.h_efold)

                # Handle pole grid points
                if adm.ADM_have_pl:
                    n = adm.ADM_gslf_pl
                    for l in range(1, adm.ADM_lall_pl + 1):
                        for k in range(adm.ADM_kmin - 1, adm.ADM_kmax + 2):  # +2 for Python equivalent of Fortran upper bound
                            self.GRD_vz_pl[n, k, l, self.GRD_Z] = self.GRD_gz[k] + \
                                self.GRD_zs_pl[n, adm.ADM_KNONE, l, self.GRD_ZSFC] * \
                                np.sinh((self.GRD_htop - self.GRD_gz[k]) / self.h_efold) / np.sinh(self.GRD_htop / self.h_efold)

                            self.GRD_vz_pl[n, k, l, self.GRD_ZH] = self.GRD_gzh[k] + \
                                self.GRD_zs_pl[n, adm.ADM_KNONE, l, self.GRD_ZSFC] * \
                                np.sinh((self.GRD_htop - self.GRD_gzh[k]) / self.h_efold) / np.sinh(self.GRD_htop / self.h_efold)

            # fill HALO
            self.COMM_data_transfer(self.GRD_vz, self.GRD_vz_pl) 

        else:
            self.GRD_gz = np.ones(adm.ADM_KNONE, dtype=np.float64)  # 1.0_RP assumed as float64
            self.GRD_gzh = np.ones(adm.ADM_KNONE, dtype=np.float64)


        #"""Output information about the grid structure"""
        if adm.ADM_kall != adm.ADM_KNONE:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("", file=log_file)
                    print("     |======      Vertical Coordinate [m]      ======|", file=log_file)
                    print("     |                                               |", file=log_file)
                    print("     |          -GRID CENTER-       -GRID INTERFACE- |", file=log_file)
                    print("     |  k        gz     d(gz)      gzh    d(gzh)   k |", file=log_file)
                    print("     |                                               |", file=log_file)
            
            # Output for top atmospheric layer (dummy)
            k = adm.ADM_kmax + 1
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"     | {k:3d} {self.GRD_gz[k]:10.1f} {self.GRD_dgz[k]:10.1f}                        | dummy", file=log_file)
                    print(f"     |                      {self.GRD_gzh[k]:10.1f} {self.GRD_dgzh[k]:10.1f} {k:4d} | TOA", file=log_file)

            # Output for kmax layer
            k = adm.ADM_kmax
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"     | {k:3d} {self.GRD_gz[k]:10.1f} {self.GRD_dgz[k]:10.1f}                        | kmax", file=log_file)
                    print(f"     |                      {self.GRD_gzh[k]:10.1f} {self.GRD_dgzh[k]:10.1f} {k:4d} |", file=log_file)

            # Loop through vertical layers in reverse order
            for k in range(adm.ADM_kmax - 1, adm.ADM_kmin, -1):
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print(f"     | {k:3d} {self.GRD_gz[k]:10.1f} {self.GRD_dgz[k]:10.1f}                        |", file=log_file)
                        print(f"     |                      {self.GRD_gzh[k]:10.1f} {self.GRD_dgzh[k]:10.1f} {k:4d} |", file=log_file)
                    
            # Output for kmin layer
            k = adm.ADM_kmin
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"     | {k:3d} {self.GRD_gz[k]:10.1f} {self.GRD_dgz[k]:10.1f}                        | kmin", file=log_file)
                    print(f"     |                      {self.GRD_gzh[k]:10.1f} {self.GRD_dgzh[k]:10.1f} {k:4d} | ground", file=log_file)
                
            # Output for bottom dummy layer
            k = adm.ADM_kmin - 1
            
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"     | {k:3d} {self.GRD_gz[k]:10.1f} {self.GRD_dgz[k]:10.1f}                        | dummy", file=log_file)
                    print("     |===============================================|", file=log_file)
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

        return


    def GRD_input_hgrid(self, fname, lrvertex, io_mode):

        self.GRD_x_1d  = np.empty((adm.ADM_gall, adm.ADM_lall, adm.ADM_nxyz,), dtype=np.float64)
        self.GRD_xt_1d  = np.empty((adm.ADM_gall, adm.ADM_lall, 2, adm.ADM_nxyz,), dtype=np.float64)

        if io_mode == "json":
            import json
            fullname = fname+str(prc.prc_myrank).zfill(8)+".json"
            with open(fullname, "r") as json_file:
                loaded_data = json.load(json_file)
                
            loaded_data_arrays = {varname: np.array(data) for varname, data in loaded_data.items()}

            self.GRD_x_1d[:,:,0] = loaded_data_arrays[list(loaded_data_arrays.keys())[0]]    # x
            self.GRD_x_1d[:,:,1] = loaded_data_arrays[list(loaded_data_arrays.keys())[1]]    # y
            self.GRD_x_1d[:,:,2] = loaded_data_arrays[list(loaded_data_arrays.keys())[2]]    # z
            
            if lrvertex:
                self.GRD_xt_1d[:,:,0,0] = loaded_data_arrays[list(loaded_data_arrays.keys())[3]] # ix
                self.GRD_xt_1d[:,:,1,0] = loaded_data_arrays[list(loaded_data_arrays.keys())[4]] # jx
                self.GRD_xt_1d[:,:,0,1] = loaded_data_arrays[list(loaded_data_arrays.keys())[5]] # iy
                self.GRD_xt_1d[:,:,1,1] = loaded_data_arrays[list(loaded_data_arrays.keys())[6]] # jy
                self.GRD_xt_1d[:,:,0,2] = loaded_data_arrays[list(loaded_data_arrays.keys())[7]] # iz
                self.GRD_xt_1d[:,:,1,2] = loaded_data_arrays[list(loaded_data_arrays.keys())[8]] # jz

            for i in range(adm.ADM_gall_1d):
                for j in range(adm.ADM_gall_1d):
                    for l in range(adm.ADM_lall):
                        for t in range(2):
                            for d in range(adm.ADM_nxyz):
                                ij=self.suf(i,j)
                                self.GRD_x[i,j,0,l,d] = self.GRD_x_1d[ij,l,d]
                                if lrvertex:
                                    self.GRD_xt[i,j,0,l,t,d] = self.GRD_xt_1d[ij,l,t,d]

        else:
            print("sorry, other data types are under construction")
            prc.prc_mpistop(std.io_l, std.fname_log)

        return


    def suf(self, i, j):
        return adm.ADM_gall_1d * j + i
    

    def GRD_scaling(self, fact):

        self.GRD_x *= fact
        self.GRD_xt *= fact

        if adm.ADM_have_pl:
            self.GRD_x_pl *= fact
            self.GRD_xt_pl *= fact

        if self.GRD_grid_type == self.GRD_grid_type_on_plane:
            pass  # Do nothing
        else:
            self.GRD_rscale = fact  # Set sphere radius scaling factor

        return
    

    def GRD_makelatlon(self,cnst):
        """
        Convert Cartesian coordinates to latitude and longitude.
        """

        k0 = adm.ADM_KNONE - 1 

        # Loop through each grid point

        for i in range(self.GRD_x.shape[0]):
            for j in range(self.GRD_x.shape[1]):
                for l in range(self.GRD_x.shape[2]):
            
                    # Convert (X, Y, Z) to (LAT, LON)
                    self.GRD_s[i, j, k0, l, 0], self.GRD_s[i, j, k0, l, 1] = vect.VECTR_xyz2latlon(
                        self.GRD_x[i, j, k0, l, 0],  # GRD_XDIR
                        self.GRD_x[i, j, k0, l, 1],  # GRD_YDIR
                        self.GRD_x[i, j, k0, l, 2],  # GRD_ZDIR
                        cnst   
                    )      

                    # Convert time-dependent grid points
                    self.GRD_st[i, j, k0, l, 0, 0], self.GRD_st[i, j, k0, l, 0, 1] = vect.VECTR_xyz2latlon(
                        self.GRD_xt[i, j, k0, l, 0, 0],  
                        self.GRD_xt[i, j, k0, l, 0, 1],  
                        self.GRD_xt[i, j, k0, l, 0, 2],
                        cnst
                    )

                    self.GRD_st[i, j, k0, l, 1, 0], self.GRD_st[i, j, k0, l, 1, 1] = vect.VECTR_xyz2latlon(
                        self.GRD_xt[i, j, k0, l, 1, 0],  
                        self.GRD_xt[i, j, k0, l, 1, 1],  
                        self.GRD_xt[i, j, k0, l, 1, 2],
                        cnst
                    )

                    self.GRD_LAT[i, j, l] = self.GRD_s[i, j, k0, l, 0]
                    self.GRD_LON[i, j, l] = self.GRD_s[i, j, k0, l, 1]

        if adm.ADM_have_pl:
            for ij in range(self.GRD_x_pl.shape[0]):
                for l in range(self.GRD_x_pl.shape[2]):
                        self.GRD_s_pl[ij, k0, l, 0], self.GRD_s_pl[ij, k0, l, 1] = vect.VECTR_xyz2latlon(
                            self.GRD_x_pl[ij, k0, l, 0],  
                            self.GRD_x_pl[ij, k0, l, 1],  
                            self.GRD_x_pl[ij, k0, l, 2],
                            cnst
                        )

                        self.GRD_st_pl[ij, k0, l, 0], self.GRD_st_pl[ij, k0, l, 1] = vect.VECTR_xyz2latlon(
                            self.GRD_xt_pl[ij, k0, l, 0],  
                            self.GRD_xt_pl[ij, k0, l, 1],  
                            self.GRD_xt_pl[ij, k0, l, 2],
                            cnst
                        )

                        self.GRD_LAT_pl[ij, l] = self.GRD_s_pl[ij, k0, l, 0]
                        self.GRD_LON_pl[ij, l] = self.GRD_s_pl[ij, k0, l, 1]

        return
    

    def GRD_makearc(self):
        """
        Calculate the mid-point locations of cell arcs.
        """
        k0 = adm.ADM_KNONE -1  

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
            idt.IDEAL_topo(fname_in, 
                           self.GRD_s[:, :, :, :, self.I_LAT], 
                           self.GRD_s[:, :, :, :, self.I_LON], 
                           self.GRD_zs[:, :, :, :, self.GRD_ZSFC], 
                           cnst)

        else:
            print("sorry, other data types are under construction")
            print("or perhaps")
            print("xxx [grd/GRD_input_topograph] Invalid io_mode!")
            prc.prc_mpistop(std.io_l, std.fname_log)

        comm.COMM_var(self.GRD_zs, self.GRD_zs_pl)
        #call COMM_var( GRD_zs, GRD_zs_pl, ADM_KNONE, 2 )

        return