import toml
import numpy as np
from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_grd import Grd
from mod_vector import vect
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
        self.GRD_x.fill(-999.0)
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

        rgn_all_1d = 2 ** adm.ADM_rlevel
        rgn_all = rgn_all_1d * rgn_all_1d
    

        for l in range(adm.ADM_lall):
            rgnid = adm.RGNMNG_l2r[l]

            nmax = 2
            r0 = np.zeros((nmax, nmax, 3), dtype=rdtype)
            r1 = np.zeros((nmax, nmax, 3), dtype=rdtype)

            dmd = (rgnid) // rgn_all 

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

                r1 = np.zeros((nmax, nmax, 3), dtype=rdtype)
                self.decomposition(rdtype,nmax_prev, r0, nmax, r1)

                r0 = np.zeros((nmax, nmax, 3), dtype=rdtype)
                r0[:, :, :] = r1[:, :, :]

            nmax = 2
            g0 = np.zeros((nmax, nmax, 3), dtype=rdtype)
            g1 = np.zeros((nmax, nmax, 3), dtype=rdtype)

            rgnid_dmd = rgnid % rgn_all 
            ir = rgnid_dmd % rgn_all_1d 
            jr = (rgnid_dmd - ir) // rgn_all_1d 
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
                    self.GRD_x[i, j, k, l, :] = g0[i - 1, j - 1, :]

        ij = adm.ADM_gslf_pl  # zero

        self.GRD_x_pl[ij, k, adm.I_NPL, Grd.GRD_XDIR] = 0.0
        self.GRD_x_pl[ij, k, adm.I_NPL, Grd.GRD_YDIR] = 0.0
        self.GRD_x_pl[ij, k, adm.I_NPL, Grd.GRD_ZDIR] = 1.0

        self.GRD_x_pl[ij, k, adm.I_SPL, Grd.GRD_XDIR] = 0.0
        self.GRD_x_pl[ij, k, adm.I_SPL, Grd.GRD_YDIR] = 0.0
        self.GRD_x_pl[ij, k, adm.I_SPL, Grd.GRD_ZDIR] = -1.0

        comm.COMM_data_transfer(self.GRD_x, self.GRD_x_pl)

        debug  = False 
        if debug:
            if std.io_l: 
                with open(std.fname_log, 'a') as log_file:
                    for l in range(adm.ADM_lall):
                        for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 2):
                            for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 2):

                                length = np.sqrt(self.GRD_x[i, j, k, l, 0] ** 2 + self.GRD_x[i, j, k, l, 1] ** 2 + self.GRD_x[i, j, k, l, 2] ** 2)
                                if abs(length - 1.0) > 0.1:
                                    print("i, j, k, l, rank, region:  length= ", length, file=log_file)
                                    print(i, j, k, l, adm.ADM_prc_me, adm.RGNMNG_lp2r[l], file=log_file)
                            
                                print("", file=log_file)
                                print(f"i, j, k, l :", i, j, k, l, file=log_file)
                                print(self.GRD_x[i, j, k, l, 0], file=log_file)
                                print(self.GRD_x[i, j, k, l, 1], file=log_file)
                                print(self.GRD_x[i, j, k, l, 2], file=log_file)

        return
    

    def mkgrd_spring(self,rdtype,cnst,comm,gtl):
        print("mkgrd_spring started")

        var_vindex = 8
        I_Rx = 0
        I_Ry = 1
        I_Rz = 2
        I_Wx = 3    
        I_Wy = 4
        I_Wz = 5
        I_Fsum = 6
        I_Ek = 7

        var = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_KNONE, adm.ADM_lall, var_vindex), dtype=rdtype)
        var_pl = np.empty((adm.ADM_gall_pl, adm.ADM_KNONE, adm.ADM_lall_pl, var_vindex), dtype=rdtype)
        var.fill(0.0)
        var_pl.fill(0.0)

        dump_coef = rdtype(1.0)
        dt = rdtype(2.0e-2)
        criteria = rdtype(1.0e-4)

        lambda_ = rdtype(0.0)
        dbar = rdtype(0.0)

        P = np.empty((adm.ADM_nxyz, 7, adm.ADM_gall_1d, adm.ADM_gall_1d,), dtype=rdtype)
        P.fill(0.0)
        F = np.empty((adm.ADM_nxyz, 6, adm.ADM_gall_1d,adm.ADM_gall_1d,), dtype=rdtype)
                #         3(0:2)    6(0:5)   18(0:17)    18(0:17)   gl05rl01
        F.fill(0.0)

        o = np.zeros(3, dtype=rdtype)
        fixed_point = np.empty(3, dtype=rdtype)
        P0Pm = np.empty(3, dtype=rdtype)
        P0PmP0 = np.empty(3, dtype=rdtype)
        Fsum = np.empty(3, dtype=rdtype)
        R0 = np.empty(3, dtype=rdtype)
        W0 = np.empty(3, dtype=rdtype)

        length = rdtype(0.0)
        distance = rdtype(0.0)
        E = rdtype(0.0)

        itelim = 10000001 # adjusting for 0-based indexing
        #itelim = 4 #10000001 # adjusting for 0-based indexing

        if not self.mkgrd_dospring:
            print("not doing mkgrd_spring")
            return

        k0 = adm.ADM_KNONE -1  # 0-based indexing

        lambda_ = rdtype(2.0 * cnst.CONST_PI / (10.0 * 2.0 ** (adm.ADM_glevel - 1)))
        dbar = rdtype(self.mkgrd_spring_beta * lambda_)

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("*** Apply grid modification with spring dynamics", file=log_file)
                print(f"*** spring factor beta  = {self.mkgrd_spring_beta}", file=log_file)
                print(f"*** length lambda       = {lambda_}", file=log_file)
                print(f"*** delta t             = {dt}", file=log_file)
                print(f"*** conversion criteria = {criteria}", file=log_file)
                print(f"*** dumping coefficient = {dump_coef}", file=log_file)
                print("", file=log_file)
                print(f"{'itelation':>16}{'max. Kinetic E':>16}{'max. forcing':>16}", file=log_file)

        var[:, :, :, :, :] = 0.0
        var_pl[:, :, :, :] = 0.0

        var[:, :, :, :, I_Rx:I_Rz + 1] = self.GRD_x[:, :, :, :, Grd.GRD_XDIR:Grd.GRD_ZDIR + 1]
        var_pl[:, :, :, I_Rx:I_Rz + 1] = self.GRD_x_pl[:, :, :, Grd.GRD_XDIR:Grd.GRD_ZDIR + 1]

        print("range  adm_gmin, adm_gmax:" , adm.ADM_gmin, adm.ADM_gmax)  # 1 16 
        # --- Solving spring dynamics ---
        for ite in range(itelim):
        
            for l in range(adm.ADM_lall):
                for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                    for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                        #ij = suf(i, j)
                        #ip1j = suf(i + 1, j)
                        #ip1jp1 = suf(i + 1, j + 1)
                        #ijp1 = suf(i, j + 1)
                        #im1j = suf(i - 1, j)
                        #im1jm1 = suf(i - 1, j - 1)
                        #ijm1 = suf(i, j - 1)

                        P[Grd.GRD_XDIR, 0, i, j] = var[i, j, k0, l, I_Rx]
                        P[Grd.GRD_XDIR, 1, i, j] = var[i+1, j, k0, l, I_Rx]
                        P[Grd.GRD_XDIR, 2, i, j] = var[i+1, j+1, k0, l, I_Rx]
                        P[Grd.GRD_XDIR, 3, i, j] = var[i, j+1, k0, l, I_Rx]
                        P[Grd.GRD_XDIR, 4, i, j] = var[i-1, j, k0, l, I_Rx]
                        P[Grd.GRD_XDIR, 5, i, j] = var[i-1, j-1, k0, l, I_Rx]
                        P[Grd.GRD_XDIR, 6, i, j] = var[i, j-1, k0, l, I_Rx]

                        P[Grd.GRD_YDIR, 0, i, j] = var[i, j, k0, l, I_Ry]
                        P[Grd.GRD_YDIR, 1, i, j] = var[i+1, j, k0, l, I_Ry]
                        P[Grd.GRD_YDIR, 2, i, j] = var[i+1, j+1, k0, l, I_Ry]
                        P[Grd.GRD_YDIR, 3, i, j] = var[i, j+1, k0, l, I_Ry]
                        P[Grd.GRD_YDIR, 4, i, j] = var[i-1, j, k0, l, I_Ry]
                        P[Grd.GRD_YDIR, 5, i, j] = var[i-1, j-1, k0, l, I_Ry]
                        P[Grd.GRD_YDIR, 6, i, j] = var[i, j-1, k0, l, I_Ry]

                        P[Grd.GRD_ZDIR, 0, i, j] = var[i, j, k0, l, I_Rz]
                        P[Grd.GRD_ZDIR, 1, i, j] = var[i+1, j, k0, l, I_Rz]
                        P[Grd.GRD_ZDIR, 2, i, j] = var[i+1, j+1, k0, l, I_Rz]
                        P[Grd.GRD_ZDIR, 3, i, j] = var[i, j+1, k0, l, I_Rz]
                        P[Grd.GRD_ZDIR, 4, i, j] = var[i-1, j, k0, l, I_Rz]
                        P[Grd.GRD_ZDIR, 5, i, j] = var[i-1, j-1, k0, l, I_Rz]
                        P[Grd.GRD_ZDIR, 6, i, j] = var[i, j-1, k0, l, I_Rz]

                if adm.ADM_have_sgp[l]:  # Pentagon case
                    P[:, 6, adm.ADM_gmin, adm.ADM_gmin] = P[:, 1, adm.ADM_gmin, adm.ADM_gmin]

                for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                    for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                        for m in range(1, 7):  # m = 1 to 6
                            P0Pm = vect.VECTR_cross(o, P[:, 0, i, j], o, P[:, m, i, j], rdtype)
                            P0PmP0 = vect.VECTR_cross(o, P0Pm, o, P[:, 0, i, j], rdtype)
                            length = vect.VECTR_abs(P0PmP0, rdtype)
                            distance = vect.VECTR_angle(P[:, 0, i, j], o, P[:, m, i, j], rdtype)
                            F[:, m-1, i, j] = (distance - dbar) * P0PmP0 / length  # this is where error occurs

                if adm.ADM_have_sgp[l]:  # Pentagon case
                    F[:, 5, adm.ADM_gmin, adm.ADM_gmin] = 0.0   # the 6th element (5) is set to 0.0 
                    fixed_point[:]= var[adm.ADM_gmin, adm.ADM_gmin, k0, l, I_Rx:I_Rz + 1]

                for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                    for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                        R0 = var[i, j, k0, l, I_Rx:I_Rz + 1]
                        W0 = var[i, j, k0, l, I_Wx:I_Wz + 1]
                        Fsum = np.sum(F[:, 0:6, i, j], axis=1)  # adding from 0 to 5
                        R0 = R0 + W0 * dt
                        R0 /= vect.VECTR_abs(R0, rdtype)    # div 0 error occurs 
                        W0 = W0 + (Fsum - dump_coef * W0) * dt
                        E = vect.VECTR_dot(o, R0, o, W0, rdtype)
                        W0 = W0 - E * R0
                        var[i, j, k0, l, I_Rx:I_Rz + 1] = R0
                        var[i, j, k0, l, I_Wx:I_Wz + 1] = W0
                        var[i, j, k0, l, I_Fsum] = vect.VECTR_abs(Fsum, rdtype) / lambda_
                        var[i, j, k0, l, I_Ek] = 0.5 * vect.VECTR_dot(o, W0, o, W0, rdtype)

                if adm.ADM_have_sgp[l]:  # Restore fixed point
                    var[adm.ADM_gmin, adm.ADM_gmin, k0, l, :] = 0.0
                    var[adm.ADM_gmin, adm.ADM_gmin, k0, l, I_Rx:I_Rz + 1] = fixed_point[0:3]

            comm.COMM_data_transfer(var, var_pl)

            Fsum_max = gtl.GTL_max(var[:, :, :, :, I_Fsum], var_pl[:, :, :, I_Fsum], 1, 0, 0, cnst, comm, rdtype)
            Ek_max = gtl.GTL_max(var[:, :, :, :, I_Ek], var_pl[:, :, :, I_Ek], 1, 0, 0, cnst, comm, rdtype)

            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("ite, Ek_max, Fsum_max: ", file=log_file)
                    print(f"{ite:16d}{Ek_max:16.8E}{Fsum_max:16.8E}", file=log_file)

            if Fsum_max < criteria:
                break

        self.GRD_x[:, :, :, :, Grd.GRD_XDIR:Grd.GRD_ZDIR + 1] = var[:, :, :, :, I_Rx:I_Rz + 1]
        self.GRD_x_pl[:, :, :, Grd.GRD_XDIR:Grd.GRD_ZDIR + 1] = var_pl[:, :, :, I_Rx:I_Rz + 1]

        comm.COMM_data_transfer(self.GRD_x, self.GRD_x_pl)

        print("mkgrd_spring finished?")

        debug = False
        if debug:
            if std.io_l: 
                with open(std.fname_log, 'a') as log_file:
                    print("springgridcheck", file=log_file)
                    k=adm.ADM_KNONE -1  # zero for vertical
                    for l in range(adm.ADM_lall):
                        for j in range(adm.ADM_gmin - 1, adm.ADM_gmax + 2):
                            for i in range(adm.ADM_gmin - 1, adm.ADM_gmax + 2):

                                length = np.sqrt(self.GRD_x[i, j, k, l, 0] ** 2 + self.GRD_x[i, j, k, l, 1] ** 2 + self.GRD_x[i, j, k, l, 2] ** 2)
                            
                                if True:
                                    if abs(length - 1.0) > 0.1:
                                        #print("ho")    
                                        print("i, j, k, l, rank, region:  length= ", length, file=log_file)
                                        print(i, j, k, l, adm.ADM_prc_me, adm.RGNMNG_lp2r[l], file=log_file)
                                        #print("")
                                    print("", file=log_file)
                                    print(f"i, j, k, l :", i, j, k, l, file=log_file)
                                    print(self.GRD_x[i, j, k, l, 0], file=log_file)
                                    print(self.GRD_x[i, j, k, l, 1], file=log_file)
                                    print(self.GRD_x[i, j, k, l, 2], file=log_file)
                                    print(self.GRD_x[i, j, k, l, 2]**2. + self.GRD_x[i, j, k, l, 1]**2. + self.GRD_x[i, j, k, l, 0]**2., file=log_file)

        return
    

    def decomposition(self,rdtype,n0,g0,n1,g1):

        for i in range(n0):
            for j in range(n0):
                inew = 2 * i #- 1
                jnew = 2 * j #- 1
                g1[inew, jnew, :] = g0[i, j, :]

                if i + 1 < n0 :
                    g1[inew + 1, jnew, :] = g0[i + 1, j, :] + g0[i, j, :]
                if j + 1 < n0 :
                    g1[inew, jnew + 1, :] = g0[i, j + 1, :] + g0[i, j, :]
                if i + 1 < n0 and j + 1 < n0:
                    g1[inew + 1, jnew + 1, :] = g0[i + 1, j + 1, :] + g0[i, j, :]

        for i in range(n1):
            for j in range(n1):
                r = np.sqrt(
                    g1[i, j, 0] ** 2 +
                    g1[i, j, 1] ** 2 +
                    g1[i, j, 2] ** 2
                )

                g1[i, j, 0] /= r
                g1[i, j, 1] /= r
                g1[i, j, 2] /= r

        return
    
