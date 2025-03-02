import toml
import numpy as np
from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_grd import Grd
#from mod_const import cnst



class Mkgrd:

    _instance = None

    def __init__(self,fname_in):
        self.cnfs = toml.load(fname_in)['param_mkgrd']
        self.mkgrd_dospring = self.cnfs['mkgrd_dospring']
        self.mkgrd_doprerotate = self.cnfs['mkgrd_doprerotate']
        self.mkgrd_dostretch = self.cnfs['mkgrd_dostretch']
        self.mkgrd_doshrink = self.cnfs['mkgrd_doshrink']
        self.mkgrd_dorotate = self.cnfs['mkgrd_dorotate']
        self.mkgrd_in_basename = self.cnfs['mkgrd_in_basename']
        self.mkgrd_in_io_mode = self.cnfs['mkgrd_in_io_mode']
        self.mkgrd_out_basename = self.cnfs['mkgrd_out_basename']
        self.mkgrd_out_io_mode = self.cnfs['mkgrd_out_io_mode']
        self.mkgrd_spring_beta = self.cnfs['mkgrd_spring_beta']
        self.mkgrd_prerotation_tilt = self.cnfs['mkgrd_prerotation_tilt'] 
        self.mkgrd_stretch_alpha = self.cnfs['mkgrd_stretch_alpha'] 
        self.mkgrd_shrink_level = self.cnfs['mkgrd_shrink_level'] 
        self.mkgrd_rotation_lon = self.cnfs['mkgrd_rotation_lon']
        self.mkgrd_rotation_lat = self.cnfs['mkgrd_rotation_lat']
        self.mkgrd_precision_single = self.cnfs['mkgrd_precision_single']
        return

    def mkgrd_setup(self,rdtype):

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print("+++ Program[mkgrd]/Category[prep]", file=log_file)

        
        if std.io_nml:
            with open(std.fname_log, 'a') as log_file:
                print(self.cnfs, file=log_file)

        # Grid arrays
        self.GRD_x = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_KNONE, adm.ADM_lall, adm.ADM_nxyz), dtype=rdtype)
        self.GRD_x_pl = np.empty((adm.ADM_gall_pl, adm.ADM_KNONE, adm.ADM_lall_pl, adm.ADM_nxyz), dtype=rdtype)
        self.GRD_xt = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_KNONE, adm.ADM_lall, adm.ADM_TJ - adm.ADM_TI + 1, adm.ADM_nxyz), dtype=rdtype)
        self.GRD_xt_pl = np.empty((adm.ADM_gall_pl, adm.ADM_KNONE, adm.ADM_lall_pl, adm.ADM_nxyz), dtype=rdtype)

        self.GRD_s = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_KNONE, adm.ADM_lall, 2), dtype=rdtype)
        self.GRD_s_pl = np.empty((adm.ADM_gall_pl, adm.ADM_KNONE, adm.ADM_lall_pl, 2), dtype=rdtype)
        self.GRD_st = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_KNONE, adm.ADM_lall, adm.ADM_TJ - adm.ADM_TI + 1, 2), dtype=rdtype)
        self.GRD_st_pl = np.empty((adm.ADM_gall_pl, adm.ADM_KNONE, adm.ADM_lall_pl, 2), dtype=rdtype)
        
        self.GRD_LAT = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_lall), dtype=rdtype)
        self.GRD_LAT_pl = np.empty((adm.ADM_gall_pl, adm.ADM_lall_pl), dtype=rdtype)
        self.GRD_LON = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_lall), dtype=rdtype)
        self.GRD_LON_pl = np.empty((adm.ADM_gall_pl, adm.ADM_lall_pl), dtype=rdtype)

        return

    def mkgrd_standard(self,rdtype,cnst,comm):
        print("mkgrd_standard started")
        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print(file=log_file)
                print("*** Make standard grid system", file=log_file)
                print(file=log_file)
    
        k = adm.ADM_KNONE - 1   # adm.ADM_KNONE = 1 for allocating 1D arrays, k=0 for reference to the 0th element 

        alpha2 = rdtype(2.0 * cnst.CONST_PI / 5.0)
        phi = rdtype(np.arcsin(np.cos(alpha2) / (1.0 - np.cos(alpha2))))

        #print("alpha2:", alpha2)
        #print("phi:", phi)

        rgn_all_1d = 2 ** adm.ADM_rlevel
        rgn_all = rgn_all_1d * rgn_all_1d
    

        for l in range(adm.ADM_lall):
            rgnid = adm.RGNMNG_l2r[l]

            nmax = 2
            r0 = np.zeros((nmax, nmax, 3), dtype=rdtype)
            r1 = np.zeros((nmax, nmax, 3), dtype=rdtype)

            dmd = (rgnid) // rgn_all 

            #print("dmd:", dmd)
            #print("rgnid:", rgnid)
            #print("rgn_all:", rgn_all)
            #print("rgn_all_1d:", rgn_all_1d)   

            if dmd <= 4:  # Northern Hemisphere  (0-4 are the northern hemisphere)
                rdmd = rdtype(dmd)

                r0[0, 0, Grd.GRD_XDIR] = np.cos(phi) * np.cos(alpha2 * rdmd)
                r0[0, 0, Grd.GRD_YDIR] = np.cos(phi) * np.sin(alpha2 * rdmd)
                r0[0, 0, Grd.GRD_ZDIR] = np.sin(phi)

                r0[1, 0, Grd.GRD_XDIR] = np.cos(-phi) * np.cos(alpha2 * (rdmd + 0.5))
                r0[1, 0, Grd.GRD_YDIR] = np.cos(-phi) * np.sin(alpha2 * (rdmd + 0.5))
                r0[1, 0, Grd.GRD_ZDIR] = np.sin(-phi)

                r0[0, 1, :] = [0.0, 0.0, 1.0]

                r0[1, 1, Grd.GRD_XDIR] = np.cos(phi) * np.cos(alpha2 * (rdmd + 1.0))
                r0[1, 1, Grd.GRD_YDIR] = np.cos(phi) * np.sin(alpha2 * (rdmd + 1.0))
                r0[1, 1, Grd.GRD_ZDIR] = np.sin(phi)

            else:  # Southern Hemisphere
                rdmd = rdtype(dmd - 5)

                r0[0, 0, Grd.GRD_XDIR] = np.cos(-phi) * np.cos(-alpha2 * (rdmd + 0.5))
                r0[0, 0, Grd.GRD_YDIR] = np.cos(-phi) * np.sin(-alpha2 * (rdmd + 0.5))
                r0[0, 0, Grd.GRD_ZDIR] = np.sin(-phi)

                r0[1, 0, :] = [0.0, 0.0, -1.0]

                r0[0, 1, Grd.GRD_XDIR] = np.cos(phi) * np.cos(-alpha2 * rdmd)
                r0[0, 1, Grd.GRD_YDIR] = np.cos(phi) * np.sin(-alpha2 * rdmd)
                r0[0, 1, Grd.GRD_ZDIR] = np.sin(phi)

                r0[1, 1, Grd.GRD_XDIR] = np.cos(-phi) * np.cos(-alpha2 * (rdmd - 0.5))
                r0[1, 1, Grd.GRD_YDIR] = np.cos(-phi) * np.sin(-alpha2 * (rdmd - 0.5))
                r0[1, 1, Grd.GRD_ZDIR] = np.sin(-phi)

            for rl in range(adm.ADM_rlevel):
                nmax_prev = nmax
                nmax = 2 * (nmax - 1) + 1

                #print("r0:", r0)

                r1 = np.zeros((nmax, nmax, 3), dtype=rdtype)
                #print("1st r1:", r1)
                self.decomposition(rdtype,nmax_prev, r0, nmax, r1)

            
                #print("nmax_prev:", nmax_prev)
                #print("nmax:", nmax)
                #print("r1:", r1)

                r0 = np.zeros((nmax, nmax, 3), dtype=rdtype)
                r0[:, :, :] = r1[:, :, :]

            nmax = 2
            g0 = np.zeros((nmax, nmax, 3), dtype=rdtype)
            g1 = np.zeros((nmax, nmax, 3), dtype=rdtype)

            #rgnid_dmd = (rgnid - 1) % rgn_all + 1
            rgnid_dmd = rgnid % rgn_all 
            ir = rgnid_dmd % rgn_all_1d 
            jr = (rgnid_dmd - ir) // rgn_all_1d 
            #print("rgnid_dmd:", rgnid_dmd) 
            #print("ir:", ir)   
            #print("jr:", jr) 
            g0[0, 0, :] = r0[ir, jr, :]
            g0[1, 0, :] = r0[ir + 1, jr, :]
            g0[0, 1, :] = r0[ir, jr + 1, :]
            g0[1, 1, :] = r0[ir + 1, jr + 1, :]

            for gl in range(adm.ADM_rlevel, adm.ADM_glevel):
                nmax_prev = nmax
                nmax = 2 * (nmax - 1) + 1

                g1 = np.zeros((nmax, nmax, 3))
                self.decomposition(rdtype,nmax_prev, g0, nmax, g1)

                g0 = np.zeros((nmax, nmax, 3))
                g0[:, :, :] = g1[:, :, :]

            for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
            #for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
            #    for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                
                    #ij = suf(i, j)
                    #self.GRD_x[ij, k, l, :] = g0[i - 1, j - 1, :]
                    self.GRD_x[i, j, k, l, :] = g0[i - 1, j - 1, :]

                    if False:
                        if std.io_l:
                            with open(std.fname_log, 'a') as log_file:
                                print("", file=log_file)        
                                print("i, j, k, l : ", i, j, k, l, file=log_file)
                                print("self.GRD_x[i, j, k, l, 1]:", self.GRD_x[i, j, k, l, 0], file=log_file)
                                print("self.GRD_x[i, j, k, l, 1]:", self.GRD_x[i, j, k, l, 1], file=log_file)
                                print("self.GRD_x[i, j, k, l, 1]:", self.GRD_x[i, j, k, l, 2], file=log_file)

        ij = adm.ADM_gslf_pl  # zero

        self.GRD_x_pl[ij, k, adm.I_NPL, Grd.GRD_XDIR] = 0.0
        self.GRD_x_pl[ij, k, adm.I_NPL, Grd.GRD_YDIR] = 0.0
        self.GRD_x_pl[ij, k, adm.I_NPL, Grd.GRD_ZDIR] = 1.0

        self.GRD_x_pl[ij, k, adm.I_SPL, Grd.GRD_XDIR] = 0.0
        self.GRD_x_pl[ij, k, adm.I_SPL, Grd.GRD_YDIR] = 0.0
        self.GRD_x_pl[ij, k, adm.I_SPL, Grd.GRD_ZDIR] = -1.0

        comm.COMM_data_transfer(self.GRD_x, self.GRD_x_pl, rdtype)

        return
    


    def mkgrd_spring(self,rdtype,cnst,comm):
        print("mkgrd_spring started")
        return
    

    def decomposition(self,rdtype,n0,g0,n1,g1):
        #print("decomposition started")

        #for i in range(1, n0 + 1):
        #    for j in range(1, n0 + 1):
        for i in range(n0):
            for j in range(n0):
                inew = 2 * i #- 1
                jnew = 2 * j #- 1

                #print("i, j, inew, jnew:", i, j, inew, jnew)

                #g1[inew - 1, jnew - 1, :] = g0[i - 1, j - 1, :]
                g1[inew, jnew, :] = g0[i, j, :]

                if i + 1 < n0 :
                    g1[inew + 1, jnew, :] = g0[i + 1, j, :] + g0[i, j, :]
                if j + 1 < n0 :
                    g1[inew, jnew + 1, :] = g0[i, j + 1, :] + g0[i, j, :]
                if i + 1 < n0 and j + 1 < n0:
                    g1[inew + 1, jnew + 1, :] = g0[i + 1, j + 1, :] + g0[i, j, :]

                #print("g1[:, :, 0]:", g1[:, :, 0])
                #print("g1[:, :, 1]:", g1[:, :, 1])
                #print("g1[:, :, 2]:", g1[:, :, 2])
                ##print("g1[:, :, 3]:", g1[:, :, 3])

        for i in range(n1):
            for j in range(n1):
                #print("i, j: ", i, j)
                r = np.sqrt(
                    g1[i, j, 0] ** 2 +
                    g1[i, j, 1] ** 2 +
                    g1[i, j, 2] ** 2
                )

                g1[i, j, 0] /= r
                g1[i, j, 1] /= r
                g1[i, j, 2] /= r

                #print("g1[i, j, 0]:", g1[i, j, 0])
                #print("g1[i, j, 1]:", g1[i, j, 1])    
                #print("g1[i, j, 2]:", g1[i, j, 2])

        return
    
