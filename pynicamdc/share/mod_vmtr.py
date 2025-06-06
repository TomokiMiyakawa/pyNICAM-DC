import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf

class Vmtr:
    
    _instance = None
    
    I_a = 0      # index for W2Cfact                                          
    I_b = 1
    
    I_c = 0      # index for C2Wfact                                          
    I_d = 1

    I_a_GZXH = 0 # index for C2WfactGz                                        
    I_b_GZXH = 1
    I_a_GZYH = 2
    I_b_GZYH = 3
    I_a_GZZH = 4
    I_b_GZZH = 5

    VMTR_deep_atmos = False

    def __init__(self):
        pass

    def VMTR_setup(self, fname_in, cnst, comm, grd, gmtr, oprt, rdtype):

        var_max = 6
        JXH     = 0
        JYH     = 1
        JZH     = 2
        JX      = 3
        JY      = 4
        JZ      = 5


        var    = np.full((adm.ADM_shape +    (var_max,)), cnst.CONST_UNDEF, dtype=rdtype)
        var_pl = np.full((adm.ADM_shape_pl + (var_max,)), cnst.CONST_UNDEF, dtype=rdtype)

        # --- G^1/2
        self.GSQRT    = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.GSQRT_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.GSQRTH   = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.GSQRTH_pl= np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        # --- Gamma factor
        self.GAM    = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.GAM_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.GAMH   = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.GAMH_pl= np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        # --- vector G^z at the full level
        self.GZX    = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.GZX_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.GZY    = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.GZY_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.GZZ    = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.GZZ_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        # --- vector G^z at the half level
        self.GZXH    = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.GZXH_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.GZYH    = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.GZYH_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.GZZH    = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.GZZH_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        self.VMTR_GAM2H       = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_GAM2H_pl    = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_GSGAM2      = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_GSGAM2_pl   = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_GSGAM2H     = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_GSGAM2H_pl  = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        self.VMTR_RGSQRTH     = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_RGSQRTH_pl  = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_RGAM        = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_RGAM_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_RGAMH       = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_RGAMH_pl    = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_RGSGAM2     = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_RGSGAM2_pl  = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_RGSGAM2H    = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_RGSGAM2H_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        self.VMTR_W2Cfact     = np.full((adm.ADM_shape    + (2,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_W2Cfact_pl  = np.full((adm.ADM_shape_pl + (2,)), cnst.CONST_UNDEF, dtype=rdtype)  #
        self.VMTR_C2Wfact     = np.full((adm.ADM_shape    + (2,)), cnst.CONST_UNDEF, dtype=rdtype)  
        self.VMTR_C2Wfact_pl  = np.full((adm.ADM_shape_pl + (2,)), cnst.CONST_UNDEF, dtype=rdtype)  #            
        self.VMTR_C2WfactGz   = np.full((adm.ADM_shape    + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_C2WfactGz_pl= np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)  #

        self.VMTR_VOLUME      = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_VOLUME_pl   = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        self.VMTR_PHI         = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        self.VMTR_PHI_pl      = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[vmtr]/Category[common share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'vmtrparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** vmtrparam not found in toml file! Use default.", file=log_file)
                #prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['vmtrparam']
            self.VMTR_deep_atmos = cnfs['VMTR_deep_atmos']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("*** setup metrics for 3-D control volume", file=log_file)        

        # If 1-layer model (shallow water model)
        if adm.ADM_kall == 1:
            for k in range(adm.ADM_kall):
                self.VMTR_VOLUME[:, :, k, :]    = self.GMTR_area[:, :, :]
                self.VMTR_VOLUME_pl[:, k, :] = self.GMTR_area_pl[:, :]
            return

        var   [:, :, :, :, :] = rdtype(0.0)
        var_pl[:, :, :, :] = rdtype(0.0)

        # --- calculation of Jxh, Jyh, and Jzh
                                             #0 to 2 
        oprt.OPRT_gradient( var[:, :, :, :, JXH:JZH+1], var_pl[:, :, :, JXH:JZH+1],  # [OUT]
            grd.GRD_vz[:, :, :, :, grd.GRD_ZH], grd.GRD_vz_pl[:, :, :, grd.GRD_ZH],  # [IN]
            oprt.OPRT_coef_grad, oprt.OPRT_coef_grad_pl, grd, rdtype  # [IN]
        )

        oprt.OPRT_horizontalize_vec(
            var[:, :, :, :, JXH], var_pl[:, :, :, JXH],  # [INOUT]
            var[:, :, :, :, JYH], var_pl[:, :, :, JYH],  # [INOUT]
            var[:, :, :, :, JZH], var_pl[:, :, :, JZH],   # [INOUT]
            grd, rdtype
        )

        # --- calculation of Jx, Jy, and Jz
        oprt.OPRT_gradient(
            var[:, :, :, :, JX:JZ+1], var_pl[:, :, :, JX:JZ+1],  # [OUT]
            grd.GRD_vz[:, :, :, :, grd.GRD_Z], grd.GRD_vz_pl[:, :, :, grd.GRD_Z],  # [IN]
            oprt.OPRT_coef_grad, oprt.OPRT_coef_grad_pl,  # [IN]
            grd, rdtype,  # [IN]
        )

        oprt.OPRT_horizontalize_vec(
            var[:, :, :, :, JX], var_pl[:, :, :, JX],  # [INOUT]
            var[:, :, :, :, JY], var_pl[:, :, :, JY],  # [INOUT]
            var[:, :, :, :, JZ], var_pl[:, :, :, JZ],  # [INOUT]
            grd, rdtype
        )
           

        #--- fill HALO
        comm.COMM_data_transfer(var, var_pl)

        # --- G^1/2 = dz/dgz   ## check GSQRT here
        for l in range(adm.ADM_lall):
            # --- calculation of G^1/2 at full level
            for k in range(adm.ADM_kmin, adm.ADM_kmax + 1):
                for i in range(adm.ADM_gall_1d):
                    for j in range(adm.ADM_gall_1d):
                        self.GSQRT[i, j, k, l] = (
                            grd.GRD_vz[i, j, k + 1, l, grd.GRD_ZH] - grd.GRD_vz[i, j, k, l, grd.GRD_ZH]
                        ) / grd.GRD_dgz[k]
                      

            for i in range(adm.ADM_gall_1d):
                for j in range(adm.ADM_gall_1d):   
                    self.GSQRT[i, j, adm.ADM_kmin - 1, l] = self.GSQRT[i, j, adm.ADM_kmin, l]
                    self.GSQRT[i, j, adm.ADM_kmax + 1, l] = self.GSQRT[i, j, adm.ADM_kmax, l]

            # --- calculation of G^1/2 at half level
            for k in range(adm.ADM_kmin, adm.ADM_kmax + 2):  # +2 since Python end is exclusive
                for i in range(adm.ADM_gall_1d):
                    for j in range(adm.ADM_gall_1d):
                        self.GSQRTH[i, j, k, l] = (
                            grd.GRD_vz[i, j, k, l, grd.GRD_Z] - grd.GRD_vz[i, j, k - 1, l, grd.GRD_Z]
                        ) / grd.GRD_dgzh[k]

            for i in range(adm.ADM_gall_1d):
                for j in range(adm.ADM_gall_1d):
                    self.GSQRTH[i, j, adm.ADM_kmin - 1, l] = self.GSQRTH[i, j, adm.ADM_kmin, l]

        # --- Gamma = (a+z) / a          ##check GAM here
        if self.VMTR_deep_atmos:
            for l in range(adm.ADM_lall):
                for k in range(adm.ADM_kall):
                    for i in range(adm.ADM_gall_1d):
                        for j in range(adm.ADM_gall_1d):
                            self.GAM[i, j, k, l] = rdtype(1.0) + grd.GRD_vz[i, j, k, l, grd.GRD_Z] / grd.GRD_rscale
                            self.GAMH[i, j, k, l] = rdtype(1.0) + grd.GRD_vz[i, j, k, l, grd.GRD_ZH] / grd.GRD_rscale
        else:
            for l in range(adm.ADM_lall):
                for k in range(adm.ADM_kall):
                    for i in range(adm.ADM_gall_1d):
                        for j in range(adm.ADM_gall_1d):
                            self.GAM[i, j, k, l] = rdtype(1.0)
                            self.GAMH[i, j, k, l] = rdtype(1.0)

        for l in range(adm.ADM_lall):
            for k in range(adm.ADM_kall):
                for i in range(adm.ADM_gall_1d):
                    for j in range(adm.ADM_gall_1d):
                        self.VMTR_GAM2H[i, j, k, l] = self.GAMH[i, j, k, l] ** 2
                        self.VMTR_GSGAM2[i, j, k, l] = self.GAM[i, j, k, l] ** 2 * self.GSQRT[i, j, k, l]
                        # if i ==17 and j == 0 and k == 40 and l == 1:
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("i: ", i, "j: ", j, "k: ", k, "l: ", l, file=log_file)
                        #         print("VMTR_GSGAM2: ", self.VMTR_GSGAM2[i, j, k, l], file=log_file)
                        #         print("GAM: ", self.GAM[i, j, k, l], file=log_file)
                        #         print("GSQRT: ", self.GSQRT[i, j, k, l], file=log_file)
                        self.VMTR_GSGAM2H[i, j, k, l] = self.GAMH[i, j, k, l] ** 2 * self.GSQRTH[i, j, k, l]

                        self.VMTR_RGSQRTH[i, j, k, l] = rdtype(1.0) / self.GSQRTH[i, j, k, l]
                        self.VMTR_RGAM[i, j, k, l] = rdtype(1.0) / self.GAM[i, j, k, l]
                        self.VMTR_RGAMH[i, j, k, l] = rdtype(1.0) / self.GAMH[i, j, k, l]
                        self.VMTR_RGSGAM2[i, j, k, l] = rdtype(1.0) / self.VMTR_GSGAM2[i, j, k, l]
                        self.VMTR_RGSGAM2H[i, j, k, l] = rdtype(1.0) / self.VMTR_GSGAM2H[i, j, k, l]
            
        # --- full level <-> half level interpolation factor
        for l in range(adm.ADM_lall):
            for k in range(adm.ADM_kmin, adm.ADM_kmax + 2):
                #if k <= adm.ADM_kmax:
                for i in range(adm.ADM_gall_1d):
                    for j in range(adm.ADM_gall_1d):
                        self.VMTR_C2Wfact[i, j, k, l, self.I_a] = grd.GRD_afact[k] * self.VMTR_RGSGAM2[i, j, k, l] * self.VMTR_GSGAM2H[i, j, k, l]
                        self.VMTR_C2Wfact[i, j, k, l, self.I_b] = grd.GRD_bfact[k] * self.VMTR_RGSGAM2[i, j, k - 1, l] * self.VMTR_GSGAM2H[i, j, k, l]
                #if k == adm.ADM_kmin - 1:
            for i in range(adm.ADM_gall_1d):
                for j in range(adm.ADM_gall_1d):
                    self.VMTR_C2Wfact[i, j, adm.ADM_kmin-1, l, self.I_a] = rdtype(0.0)
                    self.VMTR_C2Wfact[i, j, adm.ADM_kmin-1, l, self.I_b] = rdtype(0.0)

            for k in range(adm.ADM_kmin - 1, adm.ADM_kmax + 1):
                for i in range(adm.ADM_gall_1d):
                    for j in range(adm.ADM_gall_1d):
                        self.VMTR_W2Cfact[i, j, k, l, self.I_c] = grd.GRD_cfact[k] * self.VMTR_GSGAM2[i, j, k, l] * self.VMTR_RGSGAM2H[i, j, k + 1, l]
                        self.VMTR_W2Cfact[i, j, k, l, self.I_d] = grd.GRD_dfact[k] * self.VMTR_GSGAM2[i, j, k, l] * self.VMTR_RGSGAM2H[i, j, k, l]

            for i in range(adm.ADM_gall_1d):
                for j in range(adm.ADM_gall_1d):
                    self.VMTR_W2Cfact[i, j, adm.ADM_kmax + 1, l, self.I_c] = rdtype(0.0)
                    self.VMTR_W2Cfact[i, j, adm.ADM_kmax + 1, l, self.I_d] = rdtype(0.0)

        # --- full level <-> half level interpolation factor with Gz

        #--- Gz(X) = - JX / G^1/2
        #--- Gz(Y) = - JY / G^1/2
        #--- Gz(Z) = - JZ / G^1/2
        for l in range(adm.ADM_lall):
            for k in range(adm.ADM_kall):
                for i in range(adm.ADM_gall_1d):
                    for j in range(adm.ADM_gall_1d):
                        self.GZX[i, j, k, l] = -var[i, j, k, l, JX] / self.GSQRT[i, j, k, l]
                        self.GZY[i, j, k, l] = -var[i, j, k, l, JY] / self.GSQRT[i, j, k, l]
                        self.GZZ[i, j, k, l] = -var[i, j, k, l, JZ] / self.GSQRT[i, j, k, l]
                        self.GZXH[i, j, k, l] = -var[i, j, k, l, JXH] / self.GSQRTH[i, j, k, l]
                        self.GZYH[i, j, k, l] = -var[i, j, k, l, JYH] / self.GSQRTH[i, j, k, l]
                        self.GZZH[i, j, k, l] = -var[i, j, k, l, JZH] / self.GSQRTH[i, j, k, l]

            for k in range(adm.ADM_kmin, adm.ADM_kmax + 2):
                for i in range(adm.ADM_gall_1d):
                    for j in range(adm.ADM_gall_1d):
                        self.VMTR_C2WfactGz[i, j, k, l, self.I_a_GZXH] = grd.GRD_afact[k] * self.VMTR_RGSGAM2[i, j, k, l]     * self.VMTR_GSGAM2H[i, j, k, l] * self.GZXH[i, j, k, l]
                        self.VMTR_C2WfactGz[i, j, k, l, self.I_b_GZXH] = grd.GRD_bfact[k] * self.VMTR_RGSGAM2[i, j, k - 1, l] * self.VMTR_GSGAM2H[i, j, k, l] * self.GZXH[i, j, k, l]
                        self.VMTR_C2WfactGz[i, j, k, l, self.I_a_GZYH] = grd.GRD_afact[k] * self.VMTR_RGSGAM2[i, j, k, l]     * self.VMTR_GSGAM2H[i, j, k, l] * self.GZYH[i, j, k, l]
                        self.VMTR_C2WfactGz[i, j, k, l, self.I_b_GZYH] = grd.GRD_bfact[k] * self.VMTR_RGSGAM2[i, j, k - 1, l] * self.VMTR_GSGAM2H[i, j, k, l] * self.GZYH[i, j, k, l]
                        self.VMTR_C2WfactGz[i, j, k, l, self.I_a_GZZH] = grd.GRD_afact[k] * self.VMTR_RGSGAM2[i, j, k, l]     * self.VMTR_GSGAM2H[i, j, k, l] * self.GZZH[i, j, k, l]
                        self.VMTR_C2WfactGz[i, j, k, l, self.I_b_GZZH] = grd.GRD_bfact[k] * self.VMTR_RGSGAM2[i, j, k - 1, l] * self.VMTR_GSGAM2H[i, j, k, l] * self.GZZH[i, j, k, l]


            for i in range(adm.ADM_gall_1d):
                for j in range(adm.ADM_gall_1d):
                    self.VMTR_C2WfactGz[i, j, adm.ADM_kmin - 1, l, self.I_a_GZXH] = rdtype(0.0)
                    self.VMTR_C2WfactGz[i, j, adm.ADM_kmin - 1, l, self.I_b_GZXH] = rdtype(0.0)
                    self.VMTR_C2WfactGz[i, j, adm.ADM_kmin - 1, l, self.I_a_GZYH] = rdtype(0.0)
                    self.VMTR_C2WfactGz[i, j, adm.ADM_kmin - 1, l, self.I_b_GZYH] = rdtype(0.0)
                    self.VMTR_C2WfactGz[i, j, adm.ADM_kmin - 1, l, self.I_a_GZZH] = rdtype(0.0)
                    self.VMTR_C2WfactGz[i, j, adm.ADM_kmin - 1, l, self.I_b_GZZH] = rdtype(0.0)

    

        for i in range(adm.ADM_gall_1d):
            for j in range(adm.ADM_gall_1d):
                for k in range(adm.ADM_kall):
                    for l in range(adm.ADM_lall):
                        self.VMTR_VOLUME[i, j, k, l] = gmtr.GMTR_area[i, j, l] * self.VMTR_GSGAM2[i, j, k, l] * grd.GRD_dgz[k]
                        self.VMTR_PHI[i, j, k, l] = grd.GRD_vz[i, j, k, l, grd.GRD_Z] * cnst.CONST_GRAV

        if adm.ADM_have_pl:

            # --- G^1/2 = dz/dgz (pole regions)
            for l in range(adm.ADM_lall_pl):
                # Full level
                for k in range(adm.ADM_kmin, adm.ADM_kmax + 1):
                    for g in range(adm.ADM_gall_pl):
                        self.GSQRT_pl[g, k, l] = (
                            grd.GRD_vz_pl[g, k + 1, l, grd.GRD_ZH] - grd.GRD_vz_pl[g, k, l, grd.GRD_ZH]
                        ) / grd.GRD_dgz[k]

                for g in range(adm.ADM_gall_pl):
                    self.GSQRT_pl[g, adm.ADM_kmin - 1, l] = self.GSQRT_pl[g, adm.ADM_kmin, l]
                    self.GSQRT_pl[g, adm.ADM_kmax + 1, l] = self.GSQRT_pl[g, adm.ADM_kmax, l]

                # Half level
                for k in range(adm.ADM_kmin, adm.ADM_kmax + 2):
                    for g in range(adm.ADM_gall_pl):
                        self.GSQRTH_pl[g, k, l] = (
                            grd.GRD_vz_pl[g, k, l, grd.GRD_Z] - grd.GRD_vz_pl[g, k - 1, l, grd.GRD_Z]
                        ) / grd.GRD_dgzh[k]

                for g in range(adm.ADM_gall_pl):
                    self.GSQRTH_pl[g, adm.ADM_kmin - 1, l] = self.GSQRTH_pl[g, adm.ADM_kmin, l]


            # --- Gamma = (a+z) / a (pole regions)
            if self.VMTR_deep_atmos:
                for l in range(adm.ADM_lall_pl):
                    for k in range(adm.ADM_kall):
                        for g in range(adm.ADM_gall_pl):
                            self.GAM_pl[g, k, l] = rdtype(1.0) + grd.GRD_vz_pl[g, k, l, grd.GRD_Z] / grd.GRD_rscale
                            self.GAMH_pl[g, k, l] = rdtype(1.0) + grd.GRD_vz_pl[g, k, l, grd.GRD_ZH] / grd.GRD_rscale
            else:
                for l in range(adm.ADM_lall_pl):
                    for k in range(adm.ADM_kall):
                        for g in range(adm.ADM_gall_pl):
                            self.GAM_pl[g, k, l] = rdtype(1.0)
                            self.GAMH_pl[g, k, l] = rdtype(1.0)

            for l in range(adm.ADM_lall_pl):
                for k in range(adm.ADM_kall):
                    for g in range(adm.ADM_gall_pl):
                        self.VMTR_GAM2H_pl[g, k, l] = self.GAMH_pl[g, k, l] ** 2
                        self.VMTR_GSGAM2_pl[g, k, l] = self.GAM_pl[g, k, l] ** 2 * self.GSQRT_pl[g, k, l]
                        self.VMTR_GSGAM2H_pl[g, k, l] = self.GAMH_pl[g, k, l] ** 2 * self.GSQRTH_pl[g, k, l]

                        self.VMTR_RGSQRTH_pl[g, k, l] = rdtype(1.0) / self.GSQRTH_pl[g, k, l]
                        self.VMTR_RGAM_pl[g, k, l] = rdtype(1.0) / self.GAM_pl[g, k, l]
                        self.VMTR_RGAMH_pl[g, k, l] = rdtype(1.0) / self.GAMH_pl[g, k, l]
                        self.VMTR_RGSGAM2_pl[g, k, l] = rdtype(1.0) / self.VMTR_GSGAM2_pl[g, k, l]
                        self.VMTR_RGSGAM2H_pl[g, k, l] = rdtype(1.0) / self.VMTR_GSGAM2H_pl[g, k, l]

            # --- Full <-> Half level interpolation factors (pole regions)
            for l in range(adm.ADM_lall_pl):
                for k in range(adm.ADM_kmin, adm.ADM_kmax + 2):
                    for g in range(adm.ADM_gall_pl):
                        self.VMTR_C2Wfact_pl[g, k, l, self.I_a] = (
                            grd.GRD_afact[k] * self.VMTR_RGSGAM2_pl[g, k, l] * self.VMTR_GSGAM2H_pl[g, k, l]
                        )
                        self.VMTR_C2Wfact_pl[g, k, l, self.I_b] = (
                            grd.GRD_bfact[k] * self.VMTR_RGSGAM2_pl[g, k - 1, l] * self.VMTR_GSGAM2H_pl[g, k, l]
                        )

                for g in range(adm.ADM_gall_pl):
                    self.VMTR_C2Wfact_pl[g, adm.ADM_kmin - 1, l, self.I_a] = rdtype(0.0)
                    self.VMTR_C2Wfact_pl[g, adm.ADM_kmin - 1, l, self.I_b] = rdtype(0.0)

                for k in range(adm.ADM_kmin - 1, adm.ADM_kmax + 1):
                    for g in range(adm.ADM_gall_pl):
                        self.VMTR_W2Cfact_pl[g, k, l, self.I_c] = (
                            grd.GRD_cfact[k] * self.VMTR_GSGAM2_pl[g, k, l] * self.VMTR_RGSGAM2H_pl[g, k + 1, l]
                        )
                        self.VMTR_W2Cfact_pl[g, k, l, self.I_d] = (
                            grd.GRD_dfact[k] * self.VMTR_GSGAM2_pl[g, k, l] * self.VMTR_RGSGAM2H_pl[g, k, l]
                        )

                for g in range(adm.ADM_gall_pl):
                    self.VMTR_W2Cfact_pl[g, adm.ADM_kmax + 1, l, self.I_c] = rdtype(0.0)
                    self.VMTR_W2Cfact_pl[g, adm.ADM_kmax + 1, l, self.I_d] = rdtype(0.0)

            # --- Gz vector components (pole regions)
            for l in range(adm.ADM_lall_pl):
                for k in range(adm.ADM_kall):
                    for g in range(adm.ADM_gall_pl):
                        self.GZX_pl[g, k, l] =  -var_pl[g, k, l, JX]   / self.GSQRT_pl[g, k, l]
                        self.GZY_pl[g, k, l] =  -var_pl[g, k, l, JY]   / self.GSQRT_pl[g, k, l]
                        self.GZZ_pl[g, k, l] =  -var_pl[g, k, l, JZ]   / self.GSQRT_pl[g, k, l]
                        self.GZXH_pl[g, k, l] = -var_pl[g, k, l, JXH] / self.GSQRTH_pl[g, k, l]
                        self.GZYH_pl[g, k, l] = -var_pl[g, k, l, JYH] / self.GSQRTH_pl[g, k, l]
                        self.GZZH_pl[g, k, l] = -var_pl[g, k, l, JZH] / self.GSQRTH_pl[g, k, l]

                for k in range(adm.ADM_kmin, adm.ADM_kmax + 2):
                    for g in range(adm.ADM_gall_pl):
                        self.VMTR_C2WfactGz_pl[g, k, l, self.I_a_GZXH] = grd.GRD_afact[k] * self.VMTR_RGSGAM2_pl[g, k, l] * self.VMTR_GSGAM2H_pl[g, k, l] * self.GZXH_pl[g, k, l]
                        self.VMTR_C2WfactGz_pl[g, k, l, self.I_b_GZXH] = grd.GRD_bfact[k] * self.VMTR_RGSGAM2_pl[g, k - 1, l] * self.VMTR_GSGAM2H_pl[g, k, l] * self.GZXH_pl[g, k, l]
                        self.VMTR_C2WfactGz_pl[g, k, l, self.I_a_GZYH] = grd.GRD_afact[k] * self.VMTR_RGSGAM2_pl[g, k, l] * self.VMTR_GSGAM2H_pl[g, k, l] * self.GZYH_pl[g, k, l]
                        self.VMTR_C2WfactGz_pl[g, k, l, self.I_b_GZYH] = grd.GRD_bfact[k] * self.VMTR_RGSGAM2_pl[g, k - 1, l] * self.VMTR_GSGAM2H_pl[g, k, l] * self.GZYH_pl[g, k, l]
                        self.VMTR_C2WfactGz_pl[g, k, l, self.I_a_GZZH] = grd.GRD_afact[k] * self.VMTR_RGSGAM2_pl[g, k, l] * self.VMTR_GSGAM2H_pl[g, k, l] * self.GZZH_pl[g, k, l]
                        self.VMTR_C2WfactGz_pl[g, k, l, self.I_b_GZZH] = grd.GRD_bfact[k] * self.VMTR_RGSGAM2_pl[g, k - 1, l] * self.VMTR_GSGAM2H_pl[g, k, l] * self.GZZH_pl[g, k, l]

                for g in range(adm.ADM_gall_pl):
                    self.VMTR_C2WfactGz_pl[g, adm.ADM_kmin - 1, l, self.I_a_GZXH] = rdtype(0.0)
                    self.VMTR_C2WfactGz_pl[g, adm.ADM_kmin - 1, l, self.I_b_GZXH] = rdtype(0.0)
                    self.VMTR_C2WfactGz_pl[g, adm.ADM_kmin - 1, l, self.I_a_GZYH] = rdtype(0.0)
                    self.VMTR_C2WfactGz_pl[g, adm.ADM_kmin - 1, l, self.I_b_GZYH] = rdtype(0.0)
                    self.VMTR_C2WfactGz_pl[g, adm.ADM_kmin - 1, l, self.I_a_GZZH] = rdtype(0.0)
                    self.VMTR_C2WfactGz_pl[g, adm.ADM_kmin - 1, l, self.I_b_GZZH] = rdtype(0.0)

            # --- Volume and geopotential (pole regions)
            for l in range(adm.ADM_lall_pl):
                for k in range(adm.ADM_kall):
                    for g in range(adm.ADM_gall_pl):
                        self.VMTR_VOLUME_pl[g, k, l] = gmtr.GMTR_area_pl[g, l] * self.VMTR_GSGAM2_pl[g, k, l] * grd.GRD_dgz[k]
                        self.VMTR_PHI_pl[g, k, l] = grd.GRD_vz_pl[g, k, l, grd.GRD_Z] * cnst.CONST_GRAV

        else:
                        
            self.VMTR_GAM2H_pl[:, :, :]    = rdtype(0.0)
            self.VMTR_GSGAM2_pl[:, :, :]   = rdtype(0.0)
            self.VMTR_GSGAM2H_pl[:, :, :]  = rdtype(0.0)
            self.VMTR_RGSQRTH_pl[:, :, :]  = rdtype(0.0)
            self.VMTR_RGAM_pl[:, :, :]     = rdtype(0.0)
            self.VMTR_RGAMH_pl[:, :, :]    = rdtype(0.0)
            self.VMTR_RGSGAM2_pl[:, :, :]  = rdtype(0.0)
            self.VMTR_RGSGAM2H_pl[:, :, :] = rdtype(0.0)
            self.VMTR_W2Cfact_pl[:, :, :, :] = rdtype(0.0)
            self.VMTR_C2Wfact_pl[:, :, :, :] = rdtype(0.0)
            self.VMTR_C2WfactGz_pl[:, :, :, :] = rdtype(0.0)
            self.VMTR_VOLUME_pl[:, :, :]   = rdtype(0.0)
            self.VMTR_PHI_pl[:, :, :]      = rdtype(0.0)

        return