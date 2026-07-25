import toml
import numpy as np
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_prof import prf
from pynicamdc.share.mod_grd import Grd
from pynicamdc.share.mod_vector import vect
#from mod_const import cnst

class Mkgrd:


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
        self.GRD_xt.fill(-999.0)   # deterministic filler for never-computed outer-halo cells
        self.GRD_xt_pl = np.empty((adm.ADM_gall_pl, adm.ADM_KNONE, adm.ADM_lall_pl, adm.ADM_nxyz), dtype=rdtype)
        self.GRD_xt_pl.fill(-999.0)

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
        #print("mkgrd_standard started")
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
    

    def mkgrd_spring(self, rdtype, cnst, comm, gtl, vectorized=True):
        """Spring dynamics smoothing. vectorized=True runs the numpy array
        form (identical formulas and accumulation order to the scalar loops;
        bit-compared at gl03); False keeps the original per-point loops."""
        if vectorized:
            return self.mkgrd_spring_vec(rdtype, cnst, comm, gtl)
        return self.mkgrd_spring_loop(rdtype, cnst, comm, gtl)

    def mkgrd_spring_vec(self, rdtype, cnst, comm, gtl):

        var_vindex = 8
        I_Rx, I_Ry, I_Rz = 0, 1, 2
        I_Wx, I_Wy, I_Wz = 3, 4, 5
        I_Fsum, I_Ek = 6, 7

        var = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_KNONE, adm.ADM_lall, var_vindex), dtype=rdtype)
        var_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_KNONE, adm.ADM_lall_pl, var_vindex), dtype=rdtype)

        dump_coef = rdtype(1.0)
        dt = rdtype(2.0e-2)
        criteria = rdtype(1.0e-4)

        itelim = 10000001

        if not self.mkgrd_dospring:
            print("not doing mkgrd_spring")
            return

        k0 = adm.ADM_KNONE - 1
        gmin, gmax = adm.ADM_gmin, adm.ADM_gmax

        lambda_ = rdtype(2.0 * cnst.CONST_PI / (10.0 * 2.0 ** (adm.ADM_glevel - 1)))
        dbar = rdtype(self.mkgrd_spring_beta * lambda_)

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("*** Apply grid modification with spring dynamics (vectorized)", file=log_file)
                print(f"*** spring factor beta  = {self.mkgrd_spring_beta}", file=log_file)
                print(f"*** length lambda       = {lambda_}", file=log_file)
                print(f"*** delta t             = {dt}", file=log_file)
                print(f"*** conversion criteria = {criteria}", file=log_file)
                print(f"*** dumping coefficient = {dump_coef}", file=log_file)
                print("", file=log_file)
                print(f"{'itelation':>16}{'max. Kinetic E':>16}{'max. forcing':>16}", file=log_file)

        var[:, :, :, :, I_Rx:I_Rz + 1] = self.GRD_x[:, :, :, :, Grd.GRD_XDIR:Grd.GRD_ZDIR + 1]
        var_pl[:, :, :, I_Rx:I_Rz + 1] = self.GRD_x_pl[:, :, :, Grd.GRD_XDIR:Grd.GRD_ZDIR + 1]

        sl = slice(gmin, gmax + 1)
        # neighbour slice pairs (i-offset, j-offset) for m = 1..6
        nb = [(1, 0), (1, 1), (0, 1), (-1, 0), (-1, -1), (0, -1)]

        def cross(u, v):  # (..., 3) x (..., 3), component form == VECTR_cross(o,u,o,v)
            return np.stack([u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1],
                             u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2],
                             u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]], axis=-1)

        def dot(u, v):
            return u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1] + u[..., 2] * v[..., 2]

        def vabs(u):
            return np.sqrt(u[..., 0] * u[..., 0] + u[..., 1] * u[..., 1] + u[..., 2] * u[..., 2])

        for ite in range(itelim):

            for l in range(adm.ADM_lall):

                R = var[:, :, k0, l, I_Rx:I_Rz + 1]
                P0 = R[sl, sl]                                   # (n, n, 3)
                Pm = [R[slice(gmin + di, gmax + 1 + di), slice(gmin + dj, gmax + 1 + dj)]
                      for (di, dj) in nb]

                if adm.ADM_have_sgp[l]:  # pentagon: 6th neighbour == 1st
                    Pm = [np.array(p) if m == 5 else p for m, p in enumerate(Pm)]
                    Pm[5][0, 0, :] = Pm[0][0, 0, :]

                Fsum = None
                for m in range(6):
                    P0Pm = cross(P0, Pm[m])
                    P0PmP0 = cross(P0Pm, P0)
                    length = vabs(P0PmP0)
                    distance = np.arctan2(vabs(P0Pm), dot(P0, Pm[m]))
                    # keep the scalar-loop operation order: (d - dbar) * v / len
                    F = (distance - dbar)[..., None] * P0PmP0 / length[..., None]
                    if adm.ADM_have_sgp[l] and m == 5:
                        F[0, 0, :] = 0.0
                    Fsum = F if Fsum is None else Fsum + F

                W = var[:, :, k0, l, I_Wx:I_Wz + 1][sl, sl]
                if adm.ADM_have_sgp[l]:
                    fixed_point = np.array(var[gmin, gmin, k0, l, I_Rx:I_Rz + 1])

                R0 = P0 + W * dt
                R0 = R0 / vabs(R0)[..., None]
                W0 = W + (Fsum - dump_coef * W) * dt
                E = dot(R0, W0)
                W0 = W0 - E[..., None] * R0

                var[sl, sl, k0, l, I_Rx:I_Rz + 1] = R0
                var[sl, sl, k0, l, I_Wx:I_Wz + 1] = W0
                var[sl, sl, k0, l, I_Fsum] = vabs(Fsum) / lambda_
                var[sl, sl, k0, l, I_Ek] = 0.5 * dot(W0, W0)

                if adm.ADM_have_sgp[l]:  # restore fixed pentagon point
                    var[gmin, gmin, k0, l, :] = 0.0
                    var[gmin, gmin, k0, l, I_Rx:I_Rz + 1] = fixed_point

            comm.COMM_data_transfer(var, var_pl)

            Fsum_max = gtl.GTL_max(var[:, :, :, :, I_Fsum], var_pl[:, :, :, I_Fsum], 1, 0, 0, cnst, comm, rdtype)
            Ek_max = gtl.GTL_max(var[:, :, :, :, I_Ek], var_pl[:, :, :, I_Ek], 1, 0, 0, cnst, comm, rdtype)

            if std.io_l and (ite % 100 == 0 or Fsum_max < criteria):
                with open(std.fname_log, 'a') as log_file:
                    print(f"{ite:16d}{Ek_max:16.8E}{Fsum_max:16.8E}", file=log_file)

            if Fsum_max < criteria:
                break

        self.GRD_x[:, :, :, :, Grd.GRD_XDIR:Grd.GRD_ZDIR + 1] = var[:, :, :, :, I_Rx:I_Rz + 1]
        self.GRD_x_pl[:, :, :, Grd.GRD_XDIR:Grd.GRD_ZDIR + 1] = var_pl[:, :, :, I_Rx:I_Rz + 1]

        comm.COMM_data_transfer(self.GRD_x, self.GRD_x_pl)

        return

    def mkgrd_spring_loop(self,rdtype,cnst,comm,gtl):
        #print("mkgrd_spring started")

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
        #itelim = 10 #10000001 # adjusting for 0-based indexing

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

                prf.PROF_rapstart('mkgrd_spring_1', 2) 

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

                prf.PROF_rapend('mkgrd_spring_1', 2) 
                prf.PROF_rapstart('mkgrd_spring_2', 2) 

                for j in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                    for i in range(adm.ADM_gmin, adm.ADM_gmax + 1):
                        for m in range(1, 7):  # m = 1 to 6

                            prf.PROF_rapstart('mkgrd_spring_loop_cross1', 2)  
                            P0Pm = vect.VECTR_cross(o, P[:, 0, i, j], o, P[:, m, i, j], rdtype)
                            prf.PROF_rapend('mkgrd_spring_loop_cross1', 2)  
                            prf.PROF_rapstart('mkgrd_spring_loop_cross2', 2)  
                            P0PmP0 = vect.VECTR_cross(o, P0Pm, o, P[:, 0, i, j], rdtype)
                            prf.PROF_rapend('mkgrd_spring_loop_cross2', 2)  
                            prf.PROF_rapstart('mkgrd_spring_loop_abs', 2)  
                            #length = np.sqrt(P0PmP0[0] * P0PmP0[0] + P0PmP0[1] * P0PmP0[1] + P0PmP0[2] * P0PmP0[2])
                            length = vect.VECTR_abs(P0PmP0, rdtype)
                            prf.PROF_rapend('mkgrd_spring_loop_abs', 2)  
                            prf.PROF_rapstart('mkgrd_spring_loop_angle', 2)  
                            distance = vect.VECTR_angle(P[:, 0, i, j], o, P[:, m, i, j], rdtype)
                            prf.PROF_rapend('mkgrd_spring_loop_angle', 2)  
                            prf.PROF_rapstart('mkgrd_spring_loop_calF', 2)  
                            F[:, m-1, i, j] = (distance - dbar) * P0PmP0 / length  # this is where error occurs
                            prf.PROF_rapend('mkgrd_spring_loop_calF', 2)  

                prf.PROF_rapend('mkgrd_spring_2', 2) 
                prf.PROF_rapstart('mkgrd_spring_3', 2) 

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

                prf.PROF_rapend('mkgrd_spring_3', 2) 
                

            comm.COMM_data_transfer(var, var_pl)

            prf.PROF_rapstart('mkgrd_spring_4', 2) 

            Fsum_max = gtl.GTL_max(var[:, :, :, :, I_Fsum], var_pl[:, :, :, I_Fsum], 1, 0, 0, cnst, comm, rdtype)
            Ek_max = gtl.GTL_max(var[:, :, :, :, I_Ek], var_pl[:, :, :, I_Ek], 1, 0, 0, cnst, comm, rdtype)

            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("ite, Ek_max, Fsum_max: ", file=log_file)
                    print(f"{ite:16d}{Ek_max:16.8E}{Fsum_max:16.8E}", file=log_file)

            prf.PROF_rapend('mkgrd_spring_4', 2) 

            if Fsum_max < criteria:
                break

        self.GRD_x[:, :, :, :, Grd.GRD_XDIR:Grd.GRD_ZDIR + 1] = var[:, :, :, :, I_Rx:I_Rz + 1]
        self.GRD_x_pl[:, :, :, Grd.GRD_XDIR:Grd.GRD_ZDIR + 1] = var_pl[:, :, :, I_Rx:I_Rz + 1]

        comm.COMM_data_transfer(self.GRD_x, self.GRD_x_pl)

        #print("mkgrd_spring finished?")

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
    

    # --- spherical-triangle gravitational center helpers (MKGRD_gravcenter) ---

    @staticmethod
    def _arc(u, v, eps=None):
        """Great-circle arc contribution cross(u,v)/|cross| * atan2(|cross|, dot).

        u, v: (..., 3). With eps set, degenerate arcs (|cross| < eps) contribute
        zero (the Fortran zerosw guard in MKGRD_vertex2center); without it the
        division reproduces the unguarded Fortran MKGRD_center2vertex.
        """
        c = np.cross(u, v)
        d = np.sum(u * v, axis=-1)
        s = np.sqrt(np.sum(c * c, axis=-1))
        ang = np.arctan2(s, d)
        if eps is None:
            with np.errstate(invalid='ignore', divide='ignore'):
                return c * (ang / s)[..., None]
        zerosw = 0.5 - np.copysign(0.5, np.abs(s) - eps)
        return c * ((1.0 - zerosw) / (s + zerosw) * ang)[..., None]

    @staticmethod
    def _normalize(a):
        return a / np.sqrt(np.sum(a * a, axis=-1))[..., None]

    def mkgrd_center2vertex(self, rdtype, cnst):
        """GRD_x (cell centers) -> GRD_xt (triangle gravitational centers).

        Faithful port of MKGRD_center2vertex (mod_mkgrd.f90). Degenerate halo
        triangles produce NaN exactly where the Fortran overrides discard them.
        """
        k0 = adm.ADM_KNONE - 1
        gmin, gmax = adm.ADM_gmin, adm.ADM_gmax
        TI, TJ = adm.ADM_TI, adm.ADM_TJ
        sl = slice(gmin - 1, gmax + 1)   # i,j = gmin-1 .. gmax
        slp = slice(gmin, gmax + 2)      # +1 neighbours

        for l in range(adm.ADM_lall):
            x = self.GRD_x[:, :, k0, l, :]
            A = x[sl, sl]        # (i  , j  )
            Bi = x[slp, sl]      # (i+1, j  )
            C = x[slp, slp]      # (i+1, j+1)
            Cj = x[sl, slp]      # (i  , j+1)

            with np.errstate(invalid='ignore', divide='ignore'):
                gc_ti = self._arc(A, Bi) + self._arc(Bi, C) + self._arc(C, A)
                gc_tj = self._arc(A, C) + self._arc(C, Cj) + self._arc(Cj, A)
                self.GRD_xt[sl, sl, k0, l, TI, :] = self._normalize(gc_ti)
                self.GRD_xt[sl, sl, k0, l, TJ, :] = self._normalize(gc_tj)

            # unused (degenerate) halo triangles: copy the valid twin
            self.GRD_xt[gmax, gmin - 1, k0, l, TI, :] = self.GRD_xt[gmax, gmin - 1, k0, l, TJ, :]
            self.GRD_xt[gmin - 1, gmax, k0, l, TJ, :] = self.GRD_xt[gmin - 1, gmax, k0, l, TI, :]

            if adm.ADM_have_sgp[l]:  # pentagon
                self.GRD_xt[gmin - 1, gmin - 1, k0, l, TI, :] = self.GRD_xt[gmin, gmin - 1, k0, l, TJ, :]

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl
            for l in range(adm.ADM_lall_pl):
                for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                    vp1 = v + 1 if v + 1 <= adm.ADM_gmax_pl else adm.ADM_gmin_pl
                    w1 = self.GRD_x_pl[n, k0, l, :]
                    w2 = self.GRD_x_pl[v, k0, l, :]
                    w3 = self.GRD_x_pl[vp1, k0, l, :]
                    gc = self._arc(w1, w2) + self._arc(w2, w3) + self._arc(w3, w1)
                    self.GRD_xt_pl[v, k0, l, :] = -self._normalize(gc)

        return

    def mkgrd_vertex2center(self, rdtype, cnst):
        """GRD_xt (triangle centers) -> GRD_x (hexagon gravitational centers).

        Faithful port of MKGRD_vertex2center (zerosw-guarded arcs).
        """
        k0 = adm.ADM_KNONE - 1
        gmin, gmax = adm.ADM_gmin, adm.ADM_gmax
        TI, TJ = adm.ADM_TI, adm.ADM_TJ
        eps = float(cnst.CONST_EPS)
        sl = slice(gmin, gmax + 1)       # i,j = gmin .. gmax
        slm = slice(gmin - 1, gmax)      # -1 neighbours

        for l in range(adm.ADM_lall):
            xt = self.GRD_xt[:, :, k0, l, :, :]
            w1 = xt[sl, slm, TJ]                 # (i  , j-1, TJ)
            w2 = xt[sl, sl, TI]                  # (i  , j  , TI)
            w3 = xt[sl, sl, TJ]                  # (i  , j  , TJ)
            w4 = xt[slm, sl, TI]                 # (i-1, j  , TI)
            w5 = xt[slm, slm, TJ]                # (i-1, j-1, TJ)
            w6 = np.array(xt[slm, slm, TI])      # (i-1, j-1, TI) (copy: pentagon override)
            w7 = w1

            if adm.ADM_have_sgp[l]:  # pentagon: 6th vertex collapses onto the 1st
                w6[0, 0, :] = w1[0, 0, :]

            gc = (self._arc(w1, w2, eps) + self._arc(w2, w3, eps)
                  + self._arc(w3, w4, eps) + self._arc(w4, w5, eps)
                  + self._arc(w5, w6, eps) + self._arc(w6, w7, eps))
            self.GRD_x[sl, sl, k0, l, :] = self._normalize(gc)

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl
            for l in range(adm.ADM_lall_pl):
                wk = [self.GRD_xt_pl[v, k0, l, :]
                      for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1)]
                wk.append(wk[0])
                gc = np.zeros(3)
                for v in range(adm.ADM_vlink):
                    gc = gc + self._arc(wk[v], wk[v + 1])
                self.GRD_x_pl[n, k0, l, :] = -self._normalize(gc)

        return

    def mkgrd_gravcenter(self, rdtype, cnst, comm):
        """MKGRD_gravcenter: center -> vertex -> center + halo exchange."""
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("*** Calc gravitational center", file=log_file)

        self.mkgrd_center2vertex(rdtype, cnst)
        self.mkgrd_vertex2center(rdtype, cnst)

        comm.COMM_data_transfer(self.GRD_x, self.GRD_x_pl)

        return

    def mkgrd_output_hgrid_npz(self, basename, rdtype):
        """Write the boundary npz GRD_input_hgrid reads (hgrid_io_mode='npz').

        One file per rank: <basename><rank:08d>.npz with flat (lall*gall)
        arrays, ij = ADM_gall_1d*j + i (i fastest), regions in local-l order.
        """
        k0 = adm.ADM_KNONE - 1
        TI, TJ = adm.ADM_TI, adm.ADM_TJ

        def flat(a):  # (i, j, lall) -> (lall*gall,) with i fastest, j, then l
            return np.ascontiguousarray(a.transpose(2, 1, 0).reshape(-1), dtype=rdtype)

        data = {
            'grd_x_x': flat(self.GRD_x[:, :, k0, :, 0]),
            'grd_x_y': flat(self.GRD_x[:, :, k0, :, 1]),
            'grd_x_z': flat(self.GRD_x[:, :, k0, :, 2]),
            'grd_xt_ix': flat(self.GRD_xt[:, :, k0, :, TI, 0]),
            'grd_xt_jx': flat(self.GRD_xt[:, :, k0, :, TJ, 0]),
            'grd_xt_iy': flat(self.GRD_xt[:, :, k0, :, TI, 1]),
            'grd_xt_jy': flat(self.GRD_xt[:, :, k0, :, TJ, 1]),
            'grd_xt_iz': flat(self.GRD_xt[:, :, k0, :, TI, 2]),
            'grd_xt_jz': flat(self.GRD_xt[:, :, k0, :, TJ, 2]),
        }
        fname = f"{basename}{prc.prc_myrank:08d}.npz"
        np.savez(fname, **data)
        print(f"wrote {fname}")
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
    
