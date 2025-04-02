import toml
import numpy as np
from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf
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

    def mkgrd_standard(self,rdtype,cnst,comm): # <vectorized by a.kamiijo on 2025.04.02>
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
                self.decomposition_vec(rdtype,nmax_prev, r0, nmax, r1)

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
                self.decomposition_vec(rdtype,nmax_prev, g0, nmax, g1)

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
    

    def mkgrd_spring(self,rdtype,cnst,comm,gtl): # <vectorized by a.kamiijo on 2025.04.02>
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

        # Initialize arrays with optimized memory layout
        # Use 'C' order (row-major) for arrays accessed primarily in the last dimensions
        # Use 'F' order (column-major) for arrays accessed in the first dimensions
        var = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_KNONE, adm.ADM_lall, var_vindex), dtype=rdtype, order='C')
        var_pl = np.empty((adm.ADM_gall_pl, adm.ADM_KNONE, adm.ADM_lall_pl, var_vindex), dtype=rdtype, order='C')
        var.fill(0.0)
        var_pl.fill(0.0)

        dump_coef = rdtype(1.0)
        dt = rdtype(2.0e-2)
        criteria = rdtype(1.0e-4)

        lambda_ = rdtype(0.0)
        dbar = rdtype(0.0)

        # Pre-determine the dimensions for more efficient memory layout
        ni = adm.ADM_gmax - adm.ADM_gmin + 1
        nj = adm.ADM_gmax - adm.ADM_gmin + 1
        
        # P array is accessed with indices [xyz, point, i, j]
        # Use 'F' order for better memory locality when accessing points sequentially
        P = np.empty((adm.ADM_nxyz, 7, adm.ADM_gall_1d, adm.ADM_gall_1d), dtype=rdtype, order='F')
        P.fill(0.0)
        
        # F array is accessed with indices [xyz, m-1, i, j]
        # Use 'F' order for better memory locality when summing over 'm' dimension
        F = np.empty((adm.ADM_nxyz, 6, adm.ADM_gall_1d, adm.ADM_gall_1d), dtype=rdtype, order='F')
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

                # 以前のループによる初期化をスライスとアドバンスドインデックスを使った効率的な操作に置き換え
                # 必要なインデックス範囲を設定
                i_range = np.arange(adm.ADM_gmin, adm.ADM_gmax + 1)
                j_range = np.arange(adm.ADM_gmin, adm.ADM_gmax + 1)
                i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')
                
                # 中心点（P0）を設定 - より効率的な配列操作
                # X座標
                P[Grd.GRD_XDIR, 0, i_grid, j_grid] = var[i_grid, j_grid, k0, l, I_Rx]
                
                # 周囲の点（P1〜P6）を設定 - より効率的な配列操作
                # 隣接点のオフセット
                offsets = [
                    (1, 0),    # P1: (i+1, j)
                    (1, 1),    # P2: (i+1, j+1)
                    (0, 1),    # P3: (i, j+1)
                    (-1, 0),   # P4: (i-1, j)
                    (-1, -1),  # P5: (i-1, j-1)
                    (0, -1)    # P6: (i, j-1)
                ]
                
                # インデックス計算をベクトル化して一度に行い、メモリアクセスパターンを最適化
                for m, (di, dj) in enumerate(offsets, 1):
                    # X, Y, Z座標を一度に設定
                    for xyz in range(adm.ADM_nxyz):
                        # ボーダー条件を考慮した配列インデックス
                        i_shifted = np.clip(i_grid + di, 0, adm.ADM_gall_1d - 1)
                        j_shifted = np.clip(j_grid + dj, 0, adm.ADM_gall_1d - 1)
                        
                        # P[xyz, m, i, j] = var[i+di, j+dj, k0, l, I_Rx+xyz]
                        P[xyz, m, i_grid, j_grid] = var[i_shifted, j_shifted, k0, l, I_Rx + xyz]
                
                # Y座標
                P[Grd.GRD_YDIR, 0, i_grid, j_grid] = var[i_grid, j_grid, k0, l, I_Ry]
                
                # Z座標
                P[Grd.GRD_ZDIR, 0, i_grid, j_grid] = var[i_grid, j_grid, k0, l, I_Rz]

                if adm.ADM_have_sgp[l]:  # Pentagon case
                    P[:, 6, adm.ADM_gmin, adm.ADM_gmin] = P[:, 1, adm.ADM_gmin, adm.ADM_gmin]

                prf.PROF_rapend('mkgrd_spring_1', 2) 
                prf.PROF_rapstart('mkgrd_spring_2', 2) 

                # Vectorized version - eliminating j and i loops
                # Vectorized calculation of interactions between each vertex and its 6 surrounding vertices
                # Create origin point array to match dimensions
                o = np.zeros(3, dtype=rdtype)

                j_range = np.arange(adm.ADM_gmin, adm.ADM_gmax + 1)
                i_range = np.arange(adm.ADM_gmin, adm.ADM_gmax + 1)
                ji_mesh = np.meshgrid(i_range, j_range, indexing='ij')
                i_grid, j_grid = ji_mesh

                for m in range(1, 7):  # m = 1 to 6
                    # Vectorized processing with optimized memory access
                    prf.PROF_rapstart('mkgrd_spring_loop_cross1_vec', 2)  
                    
                    # Calculate VECTR_cross for all combinations of P0 and Pm
                    # Origin o is the same (0,0,0) for all points
                    # Prepare arrays P[:, 0, i, j] and P[:, m, i, j] for all i,j
                    P0 = P[:, 0, i_grid, j_grid]  # shape: (3, ni, nj)
                    Pm = P[:, m, i_grid, j_grid]  # shape: (3, ni, nj)
                    
                    # Rearrange axes for better cache locality
                    P0 = np.transpose(P0, (1, 2, 0))  # shape: (ni, nj, 3)
                    Pm = np.transpose(Pm, (1, 2, 0))  # shape: (ni, nj, 3)
                    
                    # Preallocate arrays to avoid temporary memory allocation
                    o_array = np.zeros_like(P0)  # Zero arrays for origin points
                    P0Pm = np.empty_like(P0)     # Result array for first cross product
                    
                    # Use in-place calculation to eliminate temporary arrays
                    # P0Pm = vect.VECTR_cross_vec(o_array, P0, o_array, Pm, rdtype)
                    vect.VECTR_cross_vec_inplace(o_array, P0, o_array, Pm, P0Pm, rdtype)
                    
                    prf.PROF_rapend('mkgrd_spring_loop_cross1_vec', 2)  
                    prf.PROF_rapstart('mkgrd_spring_loop_cross2_vec', 2)  
                    
                    # Preallocate the second cross product result
                    P0PmP0 = np.empty_like(P0)
                    
                    # P0PmP0 = vect.VECTR_cross_vec(o_array, P0Pm, o_array, P0, rdtype)
                    vect.VECTR_cross_vec_inplace(o_array, P0Pm, o_array, P0, P0PmP0, rdtype)
                    
                    prf.PROF_rapend('mkgrd_spring_loop_cross2_vec', 2)  
                    prf.PROF_rapstart('mkgrd_spring_loop_abs_vec', 2)  
                    
                    # Optimize calculation of vector length
                    # length = vect.VECTR_abs_vec(P0PmP0, rdtype)
                    length = np.sqrt(np.einsum('ijk,ijk->ij', P0PmP0, P0PmP0))
                    
                    prf.PROF_rapend('mkgrd_spring_loop_abs_vec', 2)  
                    prf.PROF_rapstart('mkgrd_spring_loop_angle_vec', 2)  
                    
                    # Optimize calculation of angle
                    # distance = vect.VECTR_angle_vec(P0, o_array, Pm, rdtype)
                    # Calculate the components directly for better performance
                    nvlenC = np.einsum('ijk,ijk->ij', P0 - o_array, Pm - o_array)
                    
                    # Cross product for nvlenS calculation (using VECTR_cross_vec_inplace)
                    nv = np.empty_like(P0)
                    vect.VECTR_cross_vec_inplace(o_array, P0, o_array, Pm, nv, rdtype)
                    
                    # Magnitude of cross product
                    nvlenS = np.sqrt(np.einsum('ijk,ijk->ij', nv, nv))
                    
                    # Calculate angle
                    distance = np.arctan2(nvlenS, nvlenC)
                    
                    prf.PROF_rapend('mkgrd_spring_loop_angle_vec', 2)
                    prf.PROF_rapstart('mkgrd_spring_loop_calF_vec', 2)  
                    
                    # Vectorization of F[:, m-1, i, j] = (distance - dbar) * P0PmP0 / length
                    # distance_expanded = distance[..., np.newaxis]  # (ni, nj, 1)
                    # length_expanded = length[..., np.newaxis]  # (ni, nj, 1)
                    # F_update = (distance_expanded - dbar) * P0PmP0 / length_expanded  # (ni, nj, 3)
                    
                    # Prevent division by zero
                    safe_length = np.where(length > 0, length, 1.0)
                    
                    # Use broadcasting for calculation - more efficient implementation
                    factor = (distance - dbar) / safe_length
                    factor = factor[..., np.newaxis]  # Add dimension for broadcasting
                    
                    # Use broadcasting to directly compute the result - eliminates the need for loops
                    F_update = factor * P0PmP0  # Shape: (ni, nj, 3)
                    
                    # Transpose to match F's layout and assign directly
                    F_update_t = np.transpose(F_update, (2, 0, 1))  # Shape: (3, ni, nj)
                    F[:, m-1, adm.ADM_gmin:adm.ADM_gmax+1, adm.ADM_gmin:adm.ADM_gmax+1] = F_update_t
                    
                    prf.PROF_rapend('mkgrd_spring_loop_calF_vec', 2)  

                prf.PROF_rapend('mkgrd_spring_2', 2) 
                prf.PROF_rapstart('mkgrd_spring_3', 2) 

                if adm.ADM_have_sgp[l]:  # Pentagon case
                    F[:, 5, adm.ADM_gmin, adm.ADM_gmin] = 0.0   # the 6th element (5) is set to 0.0 
                    fixed_point[:]= var[adm.ADM_gmin, adm.ADM_gmin, k0, l, I_Rx:I_Rz + 1]

                # Calculation using vectorization
                prf.PROF_rapstart('mkgrd_spring_3_vec', 2)

                # Create index arrays
                i_indices = np.arange(adm.ADM_gmin, adm.ADM_gmax + 1)
                j_indices = np.arange(adm.ADM_gmin, adm.ADM_gmax + 1)
                i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')

                # Extract R0, W0 - get slices all at once
                R0 = var[i_grid, j_grid, k0, l, I_Rx:I_Rz + 1]  # shape: (ni, nj, 3)
                W0 = var[i_grid, j_grid, k0, l, I_Wx:I_Wz + 1]  # shape: (ni, nj, 3)

                # Calculate Fsum - more efficient method using direct transpose and reshape
                # F has shape (3, 6, ni, nj)
                # Use einsum for better cache efficiency and to avoid temporary arrays
                
                # 最適化されたFsumの計算 - 一時配列の作成を最小限に
                # Fsumはすべての力の合計で、F配列の2番目の次元(m)に沿って合計する
                # F[xyz, m, i, j] → Fsum[i, j, xyz]
                
                # TileサイズをLLCキャッシュサイズに合わせて最適化
                TILE_SIZE = 32
                Fsum = np.zeros((ni, nj, adm.ADM_nxyz), dtype=rdtype)
                
                # タイル処理によるキャッシュ効率の最適化
                for i_start in range(0, ni, TILE_SIZE):
                    i_end = min(i_start + TILE_SIZE, ni)
                    for j_start in range(0, nj, TILE_SIZE):
                        j_end = min(j_start + TILE_SIZE, nj)
                        
                        # タイル内のFsumを最適化された方法で計算
                        # einsum('mij->ijm', F[:, :, i_start:i_end, j_start:j_end])
                        for m in range(6):
                            for xyz in range(adm.ADM_nxyz):
                                i_slice = slice(i_start + adm.ADM_gmin, i_end + adm.ADM_gmin)
                                j_slice = slice(j_start + adm.ADM_gmin, j_end + adm.ADM_gmin)
                                
                                # 直接加算して一時配列を避ける
                                Fsum[i_start:i_end, j_start:j_end, xyz] += F[xyz, m, i_slice, j_slice]
                
                
                # Update R0 and W0 with optimized memory access patterns
                
                # Update R0
                R0 = R0 + W0 * dt

                # Normalize R0 (to length 1) - using vectorized function with optimized memory access
                # R0_length = vect.VECTR_abs_vec(R0, rdtype)
                # Use einsum for better cache efficiency
                R0_length = np.sqrt(np.einsum('ijk,ijk->ij', R0, R0))
                
                # Avoid division by zero
                safe_R0_length = np.where(R0_length > 0, R0_length, 1.0)
                
                # Division using broadcasting - we reshape for better memory alignment
                R0_normalized = R0 / safe_R0_length[..., np.newaxis]

                # Update W0 with optimized memory access patterns
                # タイル処理によるキャッシュ効率の最適化
                W0_new = np.zeros_like(W0)
                
                for i_start in range(0, ni, TILE_SIZE):
                    i_end = min(i_start + TILE_SIZE, ni)
                    for j_start in range(0, nj, TILE_SIZE):
                        j_end = min(j_start + TILE_SIZE, nj)
                        
                        # タイル内のW0を更新
                        tile_W0 = W0[i_start:i_end, j_start:j_end]
                        tile_Fsum = Fsum[i_start:i_end, j_start:j_end]
                        
                        # W0_new = W0 + (Fsum - dump_coef * W0) * dt
                        W0_new[i_start:i_end, j_start:j_end] = tile_W0 + (tile_Fsum - dump_coef * tile_W0) * dt
                
                # Calculate dot product with optimized memory access
                # E = np.sum(R0_normalized * W0_new, axis=2)
                E = np.einsum('ijk,ijk->ij', R0_normalized, W0_new)
                
                # Remove projection component with optimized memory access
                W0_result = W0_new - E[..., np.newaxis] * R0_normalized

                # Return results to var
                # Use advanced indexing to update var in one operation
                var[i_grid, j_grid, k0, l, I_Rx:I_Rz + 1] = R0_normalized
                var[i_grid, j_grid, k0, l, I_Wx:I_Wz + 1] = W0_result
                
                prf.PROF_rapend('mkgrd_spring_3_vec', 2)
                # Vectorize calculation of Fsum magnitude and Ek with optimized memory access
                
                # Fsumの大きさの計算をeinsumで効率化
                # Fsum_mag = vect.VECTR_abs_vec(Fsum, rdtype)
                Fsum_mag = np.sqrt(np.einsum('ijk,ijk->ij', Fsum, Fsum))
                
                # W0_dotの計算をeinsumで効率化
                # W0_dot = np.sum(W0_result * W0_result, axis=2)
                W0_dot = np.einsum('ijk,ijk->ij', W0_result, W0_result)
                
                # メモリアクセスパターンの最適化：タイル化されたアップデート
                for i_start in range(0, ni, TILE_SIZE):
                    i_end = min(i_start + TILE_SIZE, ni)
                    i_slice = slice(i_start + adm.ADM_gmin, i_end + adm.ADM_gmin)
                    
                    for j_start in range(0, nj, TILE_SIZE):
                        j_end = min(j_start + TILE_SIZE, nj)
                        j_slice = slice(j_start + adm.ADM_gmin, j_end + adm.ADM_gmin)
                        
                        # タイル内の値を更新
                        var[i_slice, j_slice, k0, l, I_Fsum] = Fsum_mag[i_start:i_end, j_start:j_end] / lambda_
                        var[i_slice, j_slice, k0, l, I_Ek] = 0.5 * W0_dot[i_start:i_end, j_start:j_end]
                

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
    
    def decomposition_vec(self, rdtype, n0, g0, n1, g1): # <added by a.kamiijo on 2025.04.02>
        """
        Vectorized version of decomposition method
        """
        # Create coordinate matrices for 2D indexing
        i_indices = np.arange(n0)
        j_indices = np.arange(n0)
        i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing='ij')
        
        # Calculate new indices
        inew = 2 * i_grid
        jnew = 2 * j_grid
        
        # Copy direct grid points
        for xyz in range(3):
            g1[inew, jnew, xyz] = g0[i_grid, j_grid, xyz]
        
        # Handle edge cases with masks
        mask_i = i_grid + 1 < n0
        mask_j = j_grid + 1 < n0
        
        # For points to the right
        for xyz in range(3):
            inew_plus = inew + 1
            mask = mask_i
            g1[inew_plus[mask], jnew[mask], xyz] = g0[i_grid[mask] + 1, j_grid[mask], xyz] + g0[i_grid[mask], j_grid[mask], xyz]
        
        # For points below
        for xyz in range(3):
            jnew_plus = jnew + 1
            mask = mask_j
            g1[inew[mask], jnew_plus[mask], xyz] = g0[i_grid[mask], j_grid[mask] + 1, xyz] + g0[i_grid[mask], j_grid[mask], xyz]
        
        # For diagonal points
        for xyz in range(3):
            inew_plus = inew + 1
            jnew_plus = jnew + 1
            mask = np.logical_and(mask_i, mask_j)
            g1[inew_plus[mask], jnew_plus[mask], xyz] = g0[i_grid[mask] + 1, j_grid[mask] + 1, xyz] + g0[i_grid[mask], j_grid[mask], xyz]
        
        # Normalize all points at once
        g1_squared = g1[:n1, :n1, :]**2
        r = np.sqrt(np.sum(g1_squared, axis=2))
        
        # Avoid division by zero
        r = np.where(r > 0, r, 1.0)
        
        # Apply normalization with broadcasting
        g1[:n1, :n1, :] /= r[:, :, np.newaxis]
        
        return
    
