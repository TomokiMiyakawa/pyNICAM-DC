import toml
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf
from mod_ppmask import ppm
    
#placeholder location for now
call_jax = True

@jax.jit #(cache=True)
def laplacian(scl, coef_lap, scl_pl, coef_lap_pl, v_idx):
    iall = adm.ADM_gall_1d
    jall = adm.ADM_gall_1d
    out = (
            coef_lap[1:iall-1, 1:jall-1, :, :, 0] * scl[1:iall-1, 1:jall-1, :, :] +
            coef_lap[1:iall-1, 1:jall-1, :, :, 1] * scl[2:iall,   1:jall-1, :, :] +
            coef_lap[1:iall-1, 1:jall-1, :, :, 2] * scl[2:iall,   2:jall,   :, :] +
            coef_lap[1:iall-1, 1:jall-1, :, :, 3] * scl[1:iall-1, 2:jall,   :, :] +
            coef_lap[1:iall-1, 1:jall-1, :, :, 4] * scl[0:iall-2, 1:jall-1, :, :] +
            coef_lap[1:iall-1, 1:jall-1, :, :, 5] * scl[0:iall-2, 0:jall-2, :, :] +
            coef_lap[1:iall-1, 1:jall-1, :, :, 6] * scl[1:iall-1, 0:jall-2, :, :]
        )       
    
    out_pl = jnp.sum(coef_lap_pl[v_idx, :, :] * scl_pl[v_idx, :, :], axis=0)

    return out, out_pl

def laplacian_jax(scl, scl_pl, coef_lap, coef_lap_pl, rdtype):
        
        prf.PROF_rapstart('OPRT_laplacian', 2)

        iall  = adm.ADM_gall_1d
        jall  = adm.ADM_gall_1d
        dscl = np.zeros(adm.ADM_shape, dtype=rdtype)
        dscl_pl = np.zeros(adm.ADM_shape_pl, dtype=rdtype)

        prf.PROF_rapstart('OPRT_jaxprep_laplacian', 2)
        jscl         = jnp.array(scl, dtype=jnp.float64)
        jscl_pl      = jnp.array(scl_pl, dtype=jnp.float64)
        v_idx = jnp.arange(adm.ADM_gslf_pl, adm.ADM_gmax_pl + 1)
        prf.PROF_rapend('OPRT_jaxprep_laplacian', 2)

        # if self.lfirst_lap:
        #     prf.PROF_rapstart('OPRT_jax_laplacian_1st', 2)
        #     jdscl, jdscl_pl = laplacian(jscl, coef_lap, jscl_pl, coef_lap_pl, v_idx)
        #     jdscl.block_until_ready()
        #     jdscl_pl.block_until_ready()
        #     prf.PROF_rapend('OPRT_jax_laplacian_1st', 2)
        #     self.lfirst_lap = False
        # else:
        #     prf.PROF_rapstart('OPRT_jax_laplacian', 2)
        #     jdscl, jdscl_pl = laplacian(jscl, coef_lap, jscl_pl, coef_lap_pl, v_idx)
        #     jdscl.block_until_ready()
        #     jdscl_pl.block_until_ready()
        #     prf.PROF_rapend('OPRT_jax_laplacian', 2)

        prf.PROF_rapstart('OPRT_jax_laplacian', 2)
        jdscl, jdscl_pl = laplacian(jscl, coef_lap, jscl_pl, coef_lap_pl, v_idx)
        jdscl.block_until_ready()
        jdscl_pl.block_until_ready()
        prf.PROF_rapend('OPRT_jax_laplacian', 2)

        prf.PROF_rapstart('OPRT_jaxpost_laplacian', 2)
        dscl[1:iall-1, 1:jall-1, :, :] = np.array(jdscl).astype(rdtype)  
        dscl_pl[adm.ADM_gslf_pl, :, :] = np.array(jdscl_pl).astype(rdtype)
        prf.PROF_rapend('OPRT_jaxpost_laplacian', 2)

        dscl_pl[:, :, :] = dscl_pl * ppm.plmask

        prf.PROF_rapend('OPRT_laplacian', 2)

        return dscl, dscl_pl


def laplacian_np(scl, scl_pl, coef_lap, coef_lap_pl, rdtype):
    prf.PROF_rapstart('OPRT_laplacian', 2)

    iall  = adm.ADM_gall_1d
    jall  = adm.ADM_gall_1d
    kall   = adm.ADM_kall
    lall   = adm.ADM_lall
    k0    = adm.ADM_K0
    dscl = np.zeros(adm.ADM_shape, dtype=rdtype)
    dscl_pl = np.zeros(adm.ADM_shape_pl, dtype=rdtype)

    dscl[1:iall-1, 1:jall-1, :, :] = (
        coef_lap[1:iall-1, 1:jall-1, :, :, 0] * scl[1:iall-1, 1:jall-1, :, :] +
        coef_lap[1:iall-1, 1:jall-1, :, :, 1] * scl[2:iall,   1:jall-1, :, :] +
        coef_lap[1:iall-1, 1:jall-1, :, :, 2] * scl[2:iall,   2:jall,   :, :] +
        coef_lap[1:iall-1, 1:jall-1, :, :, 3] * scl[1:iall-1, 2:jall,   :, :] +
        coef_lap[1:iall-1, 1:jall-1, :, :, 4] * scl[0:iall-2, 1:jall-1, :, :] +
        coef_lap[1:iall-1, 1:jall-1, :, :, 5] * scl[0:iall-2, 0:jall-2, :, :] +
        coef_lap[1:iall-1, 1:jall-1, :, :, 6] * scl[1:iall-1, 0:jall-2, :, :]
    )

    n = adm.ADM_gslf_pl
    dscl_pl[:, :, :] = rdtype(0.0)  # initialize

    v_idx = np.arange(adm.ADM_gslf_pl, adm.ADM_gmax_pl + 1)
    dscl_pl[n, :, :] += np.sum(
        coef_lap_pl[v_idx, :, :] * scl_pl[v_idx, :, :], axis=0
    )

    dscl_pl[:, :, :] = dscl_pl * ppm.plmask

    prf.PROF_rapend('OPRT_laplacian', 2)
    return dscl, dscl_pl
    

class Oprt:
    
    _instance = None
    
    def __init__(self):
        self.lfirst_lap  = True
        pass

    def OPRT_setup_jax(self, fname_in, cnst, gmtr, rdtype, jdtype):
 

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
                                         # gall_1d, gall_1d,   2 additional dims,   lall  (skipping kall)  
        self.OPRT_coef_div     = np.full((adm.ADM_K0shapeXYZ + (7,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.OPRT_coef_div_pl  = np.full((adm.ADM_K0shapeXYZ_pl),     cnst.CONST_UNDEF, dtype=rdtype)
                                                                                       # 5 + 1
        self.OPRT_coef_rot     = np.full((adm.ADM_K0shapeXYZ + (7,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.OPRT_coef_rot_pl  = np.full((adm.ADM_K0shapeXYZ_pl),     cnst.CONST_UNDEF, dtype=rdtype)

        self.OPRT_coef_grad    = np.full((adm.ADM_K0shapeXYZ + (7,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.OPRT_coef_grad_pl = np.full((adm.ADM_K0shapeXYZ_pl),     cnst.CONST_UNDEF, dtype=rdtype)

        self.OPRT_coef_lap     = np.full((adm.ADM_K0shape + (7,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.OPRT_coef_lap_pl  = np.full((adm.ADM_K0shape_pl),     cnst.CONST_UNDEF, dtype=rdtype)

        self.OPRT_coef_intp    = np.full((adm.ADM_K0shapeXYZ + ( 3, adm.ADM_TJ - adm.ADM_TI + 1,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.OPRT_coef_intp_pl = np.full((adm.ADM_K0shapeXYZ_pl + (3,)), cnst.CONST_UNDEF, dtype=rdtype)
                                          # 0 of pole never used (not a problem)

        self.OPRT_coef_diff    = np.full((adm.ADM_K0shapeXYZ + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.OPRT_coef_diff_pl = np.full((adm.ADM_K0shapeXYZ_pl),     cnst.CONST_UNDEF, dtype=rdtype)
                                          # 0 of pole never used, but needed for consistency (6 elements, 1 to 5 used)

        self.OPRT_divergence_setup(gmtr, rdtype)

        self.OPRT_rotation_setup(gmtr, rdtype)
        
        self.OPRT_gradient_setup(gmtr, rdtype)
        
        self.OPRT_laplacian_setup(gmtr, rdtype)
        
        self.OPRT_diffusion_setup(gmtr, rdtype)

        self.OPRT_jcoef_div    = jnp.array(self.OPRT_coef_div,  dtype=jdtype)
        self.OPRT_jcoef_rot    = jnp.array(self.OPRT_coef_rot,  dtype=jdtype)
        self.OPRT_jcoef_grad   = jnp.array(self.OPRT_coef_grad, dtype=jdtype)
        self.OPRT_jcoef_lap    = jnp.array(self.OPRT_coef_lap,  dtype=jdtype)
        self.OPRT_jcoef_diff   = jnp.array(self.OPRT_coef_diff, dtype=jdtype)
        self.OPRT_jcoef_intp   = jnp.array(self.OPRT_coef_intp, dtype=jdtype)
        self.OPRT_jcoef_div_pl  = jnp.array(self.OPRT_coef_div_pl,  dtype=jdtype)
        self.OPRT_jcoef_rot_pl  = jnp.array(self.OPRT_coef_rot_pl,  dtype=jdtype)
        self.OPRT_jcoef_grad_pl = jnp.array(self.OPRT_coef_grad_pl, dtype=jdtype)
        self.OPRT_jcoef_lap_pl  = jnp.array(self.OPRT_coef_lap_pl,  dtype=jdtype)
        self.OPRT_jcoef_diff_pl = jnp.array(self.OPRT_coef_diff_pl, dtype=jdtype)
        self.OPRT_jcoef_intp_pl = jnp.array(self.OPRT_coef_intp_pl, dtype=jdtype)

        self.lfirst_div  = True
        self.lfirst_rot  = True
        self.lfirst_grad = True
        self.lfirst_lap  = True
        self.lfirst_diff = True

        return
    

    def OPRT_setup(self, fname_in, cnst, gmtr, rdtype):

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
                                         # gall_1d, gall_1d,   2 additional dims,   lall  (skipping kall)  
        self.OPRT_coef_div     = np.full((adm.ADM_K0shapeXYZ + (7,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.OPRT_coef_div_pl  = np.full((adm.ADM_K0shapeXYZ_pl),     cnst.CONST_UNDEF, dtype=rdtype)
                                                                                       # 5 + 1
        self.OPRT_coef_rot     = np.full((adm.ADM_K0shapeXYZ + (7,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.OPRT_coef_rot_pl  = np.full((adm.ADM_K0shapeXYZ_pl),     cnst.CONST_UNDEF, dtype=rdtype)

        self.OPRT_coef_grad    = np.full((adm.ADM_K0shapeXYZ + (7,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.OPRT_coef_grad_pl = np.full((adm.ADM_K0shapeXYZ_pl),     cnst.CONST_UNDEF, dtype=rdtype)

        self.OPRT_coef_lap     = np.full((adm.ADM_K0shape + (7,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.OPRT_coef_lap_pl  = np.full((adm.ADM_K0shape_pl),     cnst.CONST_UNDEF, dtype=rdtype)

        self.OPRT_coef_intp    = np.full((adm.ADM_K0shapeXYZ + ( 3, adm.ADM_TJ - adm.ADM_TI + 1,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.OPRT_coef_intp_pl = np.full((adm.ADM_K0shapeXYZ_pl + (3,)), cnst.CONST_UNDEF, dtype=rdtype)
                                          # 0 of pole never used (not a problem)

        self.OPRT_coef_diff    = np.full((adm.ADM_K0shapeXYZ + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        self.OPRT_coef_diff_pl = np.full((adm.ADM_K0shapeXYZ_pl),     cnst.CONST_UNDEF, dtype=rdtype)
                                          # 0 of pole never used, but needed for consistency (6 elements, 1 to 5 used)

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
        self.OPRT_coef_div[:,:,:,:,:,:] = rdtype(0.0)    #  i , j, KNONE, l, xyz, 7
        self.OPRT_coef_div_pl[:,:,:,:]  = rdtype(0.0)    #  ij,    KNONE, l, xyz
        
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
                        self.OPRT_coef_div[i, j, k0, l, d, 0] = (
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
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ip1j
                        self.OPRT_coef_div[i, j, k0, l, d, 1] = (
                            - gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j  , k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j  , k0, l, TI, W2] * gmtr.GMTR_a[i, j  , k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j  , k0, l, TI, W2] * gmtr.GMTR_a[i, j  , k0, l, AIJ, hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                    
                        # ip1jp1
                        self.OPRT_coef_div[i, j, k0, l, d, 2] = (
                            + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ijp1
                        self.OPRT_coef_div[i, j, k0, l, d, 3] = (
                            + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # im1j
                        self.OPRT_coef_div[i, j, k0, l, d, 4] = (
                            + gmtr.GMTR_t[i-1, j  , k0, l, TI, W1] * gmtr.GMTR_a[i,   j  , k0, l, AJ , hn]
                            - gmtr.GMTR_t[i-1, j  , k0, l, TI, W1] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # i-1,  j-1
                        self.OPRT_coef_div[i, j, k0, l, d, 5] = (
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W1] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ijm1
                        self.OPRT_coef_div[i, j, k0, l, d, 6] = (
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W2] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i,   j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i,   j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i,   j  , k0, l, AI , hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

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
                    self.OPRT_coef_div[i, j, k0, l, d, 0] = (
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
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ip1j
                    self.OPRT_coef_div[i, j, k0, l, d, 1] = (
                        - gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i,   j  , k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j  , k0, l, TI, W2] * gmtr.GMTR_a[i,   j  , k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j  , k0, l, TI, W2] * gmtr.GMTR_a[i,   j  , k0, l, AIJ, hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ip1jp1
                    self.OPRT_coef_div[i, j, k0, l, d, 2] = (
                        + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # i, jp1
                    self.OPRT_coef_div[i, j, k0, l, d, 3] = (
                        + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , hn]
                        + gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , hn]
                        - gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # i-1, j
                    self.OPRT_coef_div[i, j, k0, l, d, 4] = (
                        + gmtr.GMTR_t[i-1, j  , k0, l, TI, W1] * gmtr.GMTR_a[i,   j  , k0, l, AJ , hn]
                        - gmtr.GMTR_t[i-1, j  , k0, l, TI, W1] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # i-1, j-1, 
                    self.OPRT_coef_div[i, j, k0, l, d, 5] = (
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j  , k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # i, j-1, 
                    self.OPRT_coef_div[i, j, k0, l, d, 6] = (
                        - gmtr.GMTR_t[i, j-1,   k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j-1,   k0, l, TJ, W1] * gmtr.GMTR_a[i,   j  , k0, l, AI , hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]


        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for d in range(adm.ADM_nxyz):
                    #hn = d + HNX - 1
                    hn = d + HNX

                    coef = rdtype(0.0)
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij   = v
                        ijp1 = v + 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl

                        coef += (
                            gmtr.GMTR_t_pl[ij , k0, l, W1] * gmtr.GMTR_a_pl[ij  , k0, l, hn] +
                            gmtr.GMTR_t_pl[ij , k0, l, W1] * gmtr.GMTR_a_pl[ijp1, k0, l, hn]
                        )

                    self.OPRT_coef_div_pl[0, k0, l, d] = coef * rdtype(0.5) * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]
                                        #1                      # 5 + 1
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):   # 1 to 5
                    #for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 2):
                        ij   = v
                        ijp1 = v + 1
                        ijm1 = v - 1

                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl       #1
                        if ijm1 == adm.ADM_gmin_pl - 1:
                            ijm1 = adm.ADM_gmax_pl       #5    1-5 used,  (0 -> 5, 6 -> 1)

                        #self.OPRT_coef_div_pl[v - 1, d, l] = (
                        self.OPRT_coef_div_pl[v, k0, l, d] = (      # v is from 1 to 5
                            + gmtr.GMTR_t_pl[ijm1, k0, l, W3] * gmtr.GMTR_a_pl[ijm1, k0, l, hn]
                            + gmtr.GMTR_t_pl[ijm1, k0, l, W3] * gmtr.GMTR_a_pl[ij  , k0, l, hn]
                            + gmtr.GMTR_t_pl[ij  , k0, l, W2] * gmtr.GMTR_a_pl[ij  , k0, l, hn]
                            + gmtr.GMTR_t_pl[ij  , k0, l, W2] * gmtr.GMTR_a_pl[ijp1, k0, l, hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]
                    #enddo v
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

        self.OPRT_coef_rot[:,:,:,:,:,:] = rdtype(0.0)      # i,  j,  KNONE, l, xyz, 7  
        self.OPRT_coef_rot_pl[:,:,:,:]  = rdtype(0.0)   # ij,     KNONE, l, xyz
        
        for l in range(lall):
            for d in range(nxyz):
                #hn = d + HNX - 1
                #         0
                ht = d + HTX
                                # 1  to  16 (inner grid points)
                for i in range (gmin, gmax + 1):
                    for j in range(gmin, gmax + 1):

                        # ij
                        self.OPRT_coef_rot[i, j, k0, l, d, 0] = (
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
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ip1j
                        self.OPRT_coef_rot[i, j, k0, l, d, 1] = (
                            - gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j-1, k0, l, AJ , ht]
                            + gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j,   k0, l, AI , ht]
                            + gmtr.GMTR_t[i, j,   k0, l, TI, W2] * gmtr.GMTR_a[i, j,   k0, l, AI , ht]
                            + gmtr.GMTR_t[i, j,   k0, l, TI, W2] * gmtr.GMTR_a[i, j,   k0, l, AIJ, ht]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ip1jp1
                        self.OPRT_coef_rot[i, j, k0, l, d, 2] = (
                            + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AI , ht]
                            + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, ht]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, ht]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , ht]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ijp1
                        self.OPRT_coef_rot[i, j, k0, l, d, 3] = (
                            + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AIJ, ht]
                            + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , ht]
                            + gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , ht]
                            - gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , ht]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # im1j
                        self.OPRT_coef_rot[i, j, k0, l, d, 4] = (
                            + gmtr.GMTR_t[i-1, j,   k0, l, TI, W1] * gmtr.GMTR_a[i,   j,   k0, l, AJ , ht]
                            - gmtr.GMTR_t[i-1, j,   k0, l, TI, W1] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # im1jm1
                        self.OPRT_coef_rot[i, j, k0, l, d, 5] = (
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W1] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , ht]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ijm1
                        self.OPRT_coef_rot[i, j, k0, l, d, 6] = (
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W2] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , ht]
                            - gmtr.GMTR_t[i,   j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i,   j-1, k0, l, AJ , ht]
                            + gmtr.GMTR_t[i,   j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i,   j,   k0, l, AI , ht]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                if adm.ADM_have_sgp[l]: # pentagon
                    # ij     = gmin
                    i = 1
                    j = 1
                    #print("TRTRTRTR, prc, l, reg:", prc.prc_myrank, l, adm.RGNMNG_lp2r[l, prc.prc_myrank])
                    # ij
                    self.OPRT_coef_rot[i, j, k0, l, d, 0] = (
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
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ip1j
                    self.OPRT_coef_rot[i, j, k0, l, d, 1] = (
                        - gmtr.GMTR_t[i,  j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i,  j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i,   j,   k0, l, AI , ht]
                        + gmtr.GMTR_t[i,  j  , k0, l, TI, W2] * gmtr.GMTR_a[i,   j,   k0, l, AI , ht]
                        + gmtr.GMTR_t[i,  j  , k0, l, TI, W2] * gmtr.GMTR_a[i,   j,   k0, l, AIJ, ht]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ip1jp1
                    self.OPRT_coef_rot[i, j, k0, l, d, 2] = (
                        + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AI , ht]
                        + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , ht]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ijp1
                    self.OPRT_coef_rot[i, j, k0, l, d, 3] = (
                        + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i,   j, k0, l, TJ, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , ht]
                        + gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i,   j, k0, l, AJ , ht]
                        - gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , ht]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # im1j
                    self.OPRT_coef_rot[i, j, k0, l, d, 4] = (
                        + gmtr.GMTR_t[i-1, j,   k0, l, TI, W1] * gmtr.GMTR_a[i,   j,   k0, l, AJ , ht]
                        - gmtr.GMTR_t[i-1, j,   k0, l, TI, W1] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # im1jm1
                    self.OPRT_coef_rot[i, j, k0, l, d, 5] = (
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j,   k0, l, AI , ht]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ijm1
                    self.OPRT_coef_rot[i, j, k0, l, d, 6] = (
                        - gmtr.GMTR_t[i, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, ht]
                        + gmtr.GMTR_t[i, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i,   j  , k0, l, AI , ht]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for d in range(adm.ADM_nxyz):
                    #hn = d + HNX - 1
                    ht = d + HTX

                    coef = rdtype(0.0)
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij   = v
                        ijp1 = v + 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl

                        coef += (
                            gmtr.GMTR_t_pl[ij , k0, l, W1] * gmtr.GMTR_a_pl[ij  , k0, l, ht] +
                            gmtr.GMTR_t_pl[ij , k0, l, W1] * gmtr.GMTR_a_pl[ijp1, k0, l, ht]
                        )

                    self.OPRT_coef_rot_pl[0, k0, l, d] = coef * rdtype(0.5) * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij   = v
                        ijp1 = v + 1
                        ijm1 = v - 1

                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl
                        if ijm1 == adm.ADM_gmin_pl - 1:
                            ijm1 = adm.ADM_gmax_pl

                        self.OPRT_coef_rot_pl[v, k0, l, d] = (
                            + gmtr.GMTR_t_pl[ijm1, k0, l, W3] * gmtr.GMTR_a_pl[ijm1, k0, l, ht]
                            + gmtr.GMTR_t_pl[ijm1, k0, l, W3] * gmtr.GMTR_a_pl[ij  , k0, l, ht]
                            + gmtr.GMTR_t_pl[ij  , k0, l, W2] * gmtr.GMTR_a_pl[ij  , k0, l, ht]
                            + gmtr.GMTR_t_pl[ij  , k0, l, W2] * gmtr.GMTR_a_pl[ijp1, k0, l, ht]
                        ) * rdtype(0.5) * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]

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
        self.OPRT_coef_grad[:,:,:,:,:,:] = rdtype(0.0)   #  i , j, KNONE, l, xyz, 7
        self.OPRT_coef_grad_pl[:,:,:,:]  = rdtype(0.0)   #  ij,    KNONE, l, xyz
        
        for l in range(lall):
            for d in range(nxyz):
                #hn = d + HNX - 1
                #         0
                hn = d + HNX
                                # 1  to  16 (inner grid points)
                for i in range (gmin, gmax + 1):
                    for j in range(gmin, gmax + 1):

                        # ij
                        self.OPRT_coef_grad[i, j, k0, l, d, 0] = (
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
                            - rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            - rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                            + rdtype(2.0) * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                            + rdtype(2.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            + rdtype(2.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                            - rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ip1j
                        self.OPRT_coef_grad[i, j, k0, l, d, 1] = (
                            - gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j, k0, l, TI, W2] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j, k0, l, TI, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ip1jp1
                        self.OPRT_coef_grad[i, j, k0, l, d, 2] = (
                            + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                            + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ijp1
                        self.OPRT_coef_grad[i, j, k0, l, d, 3] = (
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + gmtr.GMTR_t[i, j, k0, l, TJ, W3] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # im1j
                        self.OPRT_coef_grad[i, j, k0, l, d, 4] = (
                            + gmtr.GMTR_t[i-1, j, k0, l, TI, W1] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i-1, j, k0, l, TI, W1] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # im1jm1
                        self.OPRT_coef_grad[i, j, k0, l, d, 5] = (
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W1] * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                        # ijm1
                        self.OPRT_coef_grad[i, j, k0, l, d, 6] = (
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - gmtr.GMTR_t[i-1, j-1, k0, l, TI, W2] * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                            - gmtr.GMTR_t[i, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i, j-1, k0, l, AJ , hn]
                            + gmtr.GMTR_t[i, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]



                if adm.ADM_have_sgp[l]: # pentagon
                    # ij     = gmin
                    i = 1
                    j = 1

 
                    # i, j
                    self.OPRT_coef_grad[i, j, k0, l, d, 0] = (
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
                        - rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        - rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        + rdtype(2.0) * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        + rdtype(2.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        - rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ip1j
                    self.OPRT_coef_grad[i, j, k0, l, d, 1] = (
                        - gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j-1, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j, k0, l, TI, W2] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j, k0, l, TI, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ip1jp1
                    self.OPRT_coef_grad[i, j, k0, l, d, 2] = (
                        + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                        + gmtr.GMTR_t[i, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W2] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ijp1
                    self.OPRT_coef_grad[i, j, k0, l, d, 3] = (
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W3] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j, k0, l, TJ, W3] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        + gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        - gmtr.GMTR_t[i-1, j, k0, l, TI, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # im1j
                    self.OPRT_coef_grad[i, j, k0, l, d, 4] = (
                        + gmtr.GMTR_t[i-1, j, k0, l, TI, W1] * gmtr.GMTR_a[i, j, k0, l, AJ , hn]
                        - gmtr.GMTR_t[i-1, j, k0, l, TI, W1] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W3] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # im1jm1
                    self.OPRT_coef_grad[i, j, k0, l, d, 5] = (
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j, k0, l, AI , hn]
                        - gmtr.GMTR_t[i-1, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

                    # ijm1
                    self.OPRT_coef_grad[i, j, k0, l, d, 6] = (
                        - gmtr.GMTR_t[i, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        + gmtr.GMTR_t[i, j-1, k0, l, TJ, W1] * gmtr.GMTR_a[i, j, k0, l, AI , hn]
                    ) * rdtype(0.5) * gmtr.GMTR_p[i, j, k0, l, P_RAREA]

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for d in range(adm.ADM_nxyz):
                    #hn = d + HNX - 1
                    hn = d + HNX

                    coef = rdtype(0.0)
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij   = v
                        ijp1 = v + 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl

                        coef += rdtype(2.0) * (gmtr.GMTR_t_pl[ij, k0, l, W1] - rdtype(1.0)) * gmtr.GMTR_a_pl[ijp1, k0, l, hn]

                    self.OPRT_coef_grad_pl[0, k0, l, d] = coef * rdtype(0.5) * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij   = v
                        ijp1 = v + 1
                        ijm1 = v - 1

                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl
                        if ijm1 == adm.ADM_gmin_pl - 1:
                            ijm1 = adm.ADM_gmax_pl

                        self.OPRT_coef_grad_pl[v, k0, l, d] = (
                            + gmtr.GMTR_t_pl[ijm1, k0, l, W3] * gmtr.GMTR_a_pl[ijm1, k0, l, hn]
                            + gmtr.GMTR_t_pl[ijm1, k0, l, W3] * gmtr.GMTR_a_pl[ij  , k0, l, hn]
                            + gmtr.GMTR_t_pl[ij  , k0, l, W2] * gmtr.GMTR_a_pl[ij  , k0, l, hn]
                            + gmtr.GMTR_t_pl[ij  , k0, l, W2] * gmtr.GMTR_a_pl[ijp1, k0, l, hn]
                        ) * rdtype(0.5) * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]

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

        self.OPRT_coef_lap[:,:,:,:,:] = rdtype(0.0)      #  i, j, KNONE, l, 7
        self.OPRT_coef_lap_pl[:,:,:]  = rdtype(0.0)      #  ij,   KNONE, l
        
        for l in range(lall):
            for d in range(nxyz):

                hn = d + HNX
                tn = d + TNX
                                # 1  to  16 (inner grid points)
                for i in range (gmin, gmax + 1):
                    for j in range(gmin, gmax + 1):

                        # coef_lap[i, j, k0, l, 0]
                        self.OPRT_coef_lap[i, j, k0, l, 0] += gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA] * (
                            - rdtype(1.0) * gmtr.GMTR_a[i,   j, k0, l, AI,  tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            + rdtype(2.0) * gmtr.GMTR_a[i+1, j, k0, l, AJ,  tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i,   j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            - rdtype(1.0) * gmtr.GMTR_a[i,   j, k0, l, AI,  tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + rdtype(2.0) * gmtr.GMTR_a[i+1, j, k0, l, AJ,  tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i,   j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, k0, l, 0] += gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA] * (
                            - rdtype(1.0) * gmtr.GMTR_a[i, j,   k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            - rdtype(2.0) * gmtr.GMTR_a[i, j+1, k0, l, AI,  tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i, j,   k0, l, AJ,  tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            - rdtype(1.0) * gmtr.GMTR_a[i, j,   k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            - rdtype(2.0) * gmtr.GMTR_a[i, j+1, k0, l, AI,  tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i, j,   k0, l, AJ,  tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, k0, l, 0] += gmtr.GMTR_t[i-1, j, k0, l, TI, T_RAREA] * (
                            - rdtype(1.0) * gmtr.GMTR_a[i,   j, k0, l, AJ,  tn] * gmtr.GMTR_a[i,   j, k0, l, AJ, hn]
                            - rdtype(2.0) * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i,   j, k0, l, AJ, hn]
                            - rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AI,  tn] * gmtr.GMTR_a[i,   j, k0, l, AJ, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i,   j, k0, l, AJ,  tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            + rdtype(2.0) * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AI,  tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        )

                        self.OPRT_coef_lap[i, j, k0, l, 0] += gmtr.GMTR_t[i-1, j-1, k0, l, TJ, T_RAREA] * (
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            + rdtype(2.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            - rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            + rdtype(2.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, k0, l, 0] += gmtr.GMTR_t[i-1, j-1, k0, l, TI, T_RAREA] * (
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - rdtype(2.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            - rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            - rdtype(2.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, k0, l, 0] += gmtr.GMTR_t[i, j-1, k0, l, TJ, T_RAREA] * (
                            -rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            - rdtype(2.0) * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            - rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            + rdtype(2.0) * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        )

                        # coef_lap[i, j, k0, l, 1]
                        self.OPRT_coef_lap[i, j, k0, l, 1] += gmtr.GMTR_t[i, j-1, k0, l, TJ, T_RAREA] * (
                            -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            + rdtype(2.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            + rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            - rdtype(2.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            - rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        )

                        # coef_lap[i, j, k0, l, 1] (continued)
                        self.OPRT_coef_lap[i, j, k0, l, 1] += gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA] * (
                            -rdtype(1.0) * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            -rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            -rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        )

                        # coef_lap[i, j, k0, l, 2]
                        self.OPRT_coef_lap[i, j, k0, l, 2] += gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA] * (
                            +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            +rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            +rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, k0, l, 2] += gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA] * (
                            +rdtype(1.0) * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            -rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            -rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        )

                        # coef_lap[i, j, k0, l, 3]
                        self.OPRT_coef_lap[i, j, k0, l, 3] += gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA] * (
                            +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            +rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            +rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, k0, l, 3] += gmtr.GMTR_t[i-1, j, k0, l, TI, T_RAREA] * (
                            +rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            +rdtype(2.0) * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -rdtype(2.0) * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        )

                        # coef_lap[i, j, k0, l, 4]
                        self.OPRT_coef_lap[i, j, k0, l, 4] += gmtr.GMTR_t[i-1, j, k0, l, TI, T_RAREA] * (
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            +rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        )


                        # coef_lap[i, j, k0, l, 4] (continued)
                        self.OPRT_coef_lap[i, j, k0, l, 4] += gmtr.GMTR_t[i-1, j-1, k0, l, TJ, T_RAREA] * (
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -rdtype(2.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j,   k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            -rdtype(2.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j,   k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        )

                        # coef_lap[i, j, k0, l, 5]
                        self.OPRT_coef_lap[i, j, k0, l, 5] += gmtr.GMTR_t[i-1, j-1, k0, l, TJ, T_RAREA] * (
                            +rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            +rdtype(2.0) * gmtr.GMTR_a[i-1, j,   k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            +rdtype(2.0) * gmtr.GMTR_a[i-1, j,   k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        )

                        self.OPRT_coef_lap[i, j, k0, l, 5] += gmtr.GMTR_t[i-1, j-1, k0, l, TI, T_RAREA] * (
                            +rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            -rdtype(2.0) * gmtr.GMTR_a[i, j-1,   k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            -rdtype(2.0) * gmtr.GMTR_a[i, j-1,   k0, l, AJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                        )

                        # if i == 6 and j == 5 and l== 3 :
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("i = 6, j = 5, l = 3,  v6-0: ", d, file=log_file)
                        #         print(self.OPRT_coef_lap[i, j, k0, l, 6], file=log_file)
                        # coef_lap[i, j, k0, l, 6]
                        self.OPRT_coef_lap[i, j, k0, l, 6] += gmtr.GMTR_t[i-1, j-1, k0, l, TI, T_RAREA] * (
                            +rdtype(1.0) * gmtr.GMTR_a[i, j-1,   k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            +rdtype(2.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i, j-1,   k0, l, AJ, tn] * gmtr.GMTR_a[i, j-1,   k0, l, AJ, hn]
                            +rdtype(2.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j-1,   k0, l, AJ, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AI, tn] * gmtr.GMTR_a[i, j-1,   k0, l, AJ, hn]
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
                        #         print("coef lap=", self.OPRT_coef_lap[i, j, k0, l, 6], file=log_file)



                        self.OPRT_coef_lap[i, j, k0, l, 6] += gmtr.GMTR_t[i, j-1, k0, l, TJ, T_RAREA] * (
                            +rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            +rdtype(2.0) * gmtr.GMTR_a[i, j,   k0, l, AI, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j-1, k0, l, AJ, hn]
                            -rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j,   k0, l, AI, hn]
                            -rdtype(2.0) * gmtr.GMTR_a[i, j,   k0, l, AI, tn] * gmtr.GMTR_a[i, j,   k0, l, AI, hn]
                            +rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j,   k0, l, AI, hn]
                        )

            if adm.ADM_have_sgp[l]: # pentagon
                # ij     = gmin
                i = 1
                j = 1

                self.OPRT_coef_lap[i, j, k0, l, 0] = rdtype(0.0)
                self.OPRT_coef_lap[i, j, k0, l, 1] = rdtype(0.0)
                self.OPRT_coef_lap[i, j, k0, l, 2] = rdtype(0.0)
                self.OPRT_coef_lap[i, j, k0, l, 3] = rdtype(0.0)
                self.OPRT_coef_lap[i, j, k0, l, 4] = rdtype(0.0)
                self.OPRT_coef_lap[i, j, k0, l, 5] = rdtype(0.0)
                self.OPRT_coef_lap[i, j, k0, l, 6] = rdtype(0.0)

                for d in range(nxyz):
                    hn = d + HNX
                    tn = d + TNX
                
                    # (i, j)
                    self.OPRT_coef_lap[i, j, k0, l, 0] += gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA] * (
                        -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +rdtype(2.0) * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +rdtype(2.0) * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                    )

                    self.OPRT_coef_lap[i, j, k0, l, 0] += gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA] * (
                        -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -rdtype(2.0) * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -rdtype(2.0) * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                    )

                    self.OPRT_coef_lap[i, j, k0, l, 0] += gmtr.GMTR_t[i-1, j, k0, l, TI, T_RAREA] * (
                        -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -rdtype(2.0) * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        +rdtype(2.0) * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                    )

                    self.OPRT_coef_lap[i, j, k0, l, 0] += gmtr.GMTR_t[i-1, j-1, k0, l, TJ, T_RAREA] * (
                        -rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        +rdtype(2.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        +rdtype(2.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                    )

                    self.OPRT_coef_lap[i, j, k0, l, 0] += gmtr.GMTR_t[i, j-1, k0, l, TJ, T_RAREA] * (
                        -rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        -rdtype(2.0) * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i, j,   k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +rdtype(2.0) * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j,   k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                    )

                    # ip1j
                    self.OPRT_coef_lap[i, j, k0, l, 1] += gmtr.GMTR_t[i, j-1, k0, l, TJ, T_RAREA] * (
                        +rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        +rdtype(2.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j-1, k0, l, AIJ, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i, j-1, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        -rdtype(2.0) * gmtr.GMTR_a[i, j-1, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                    )

                    self.OPRT_coef_lap[i, j, k0, l, 1] += gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA] * (
                        -rdtype(1.0) * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        -rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                    )

                    # ip1jp1
                    self.OPRT_coef_lap[i, j, k0, l, 2] += gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA] * (
                        +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AI, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i+1, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                    )

                    self.OPRT_coef_lap[i, j, k0, l, 2] += gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA] * (
                        +rdtype(1.0) * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                    )

                    # ijp1
                    self.OPRT_coef_lap[i, j, k0, l, 3] += gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA] * (
                        +rdtype(1.0) * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j+1, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        +rdtype(2.0) * gmtr.GMTR_a[i, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                    )

                    self.OPRT_coef_lap[i, j, k0, l, 3] += gmtr.GMTR_t[i-1, j, k0, l, TI, T_RAREA] * (
                        +rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        +rdtype(2.0) * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                        -rdtype(1.0) * gmtr.GMTR_a[i-1, j, k0, l, AIJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        -rdtype(2.0) * gmtr.GMTR_a[i-1, j, k0, l, AI, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                        +rdtype(1.0) * gmtr.GMTR_a[i, j, k0, l, AJ, tn] * gmtr.GMTR_a[i-1, j, k0, l, AI, hn]
                    )

                    # im1j
                    self.OPRT_coef_lap[i, j, k0, l, 4] += gmtr.GMTR_t[i-1,j,k0,l,TI,T_RAREA] * ( 
                        + rdtype(1.0) * gmtr.GMTR_a[i-1,j,k0,l,AIJ,tn] * gmtr.GMTR_a[i,j,k0,l,AJ,hn]
                        + rdtype(2.0) * gmtr.GMTR_a[i,j,k0,l,AJ,tn] * gmtr.GMTR_a[i,j,k0,l,AJ,hn]
                        - rdtype(1.0) * gmtr.GMTR_a[i-1,j,k0,l,AI,tn] * gmtr.GMTR_a[i,j,k0,l,AJ,hn]
                        - rdtype(1.0) * gmtr.GMTR_a[i-1,j,k0,l,AIJ,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn]
                        - rdtype(2.0) * gmtr.GMTR_a[i,j,k0,l,AJ,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn]
                        + rdtype(1.0) * gmtr.GMTR_a[i-1,j,k0,l,AI,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                    )

                    self.OPRT_coef_lap[i, j, k0, l, 4] += gmtr.GMTR_t[i-1,j-1,k0,l,TJ,T_RAREA] * (
                        - rdtype(1.0) * gmtr.GMTR_a[i-1,j-1,k0,l,AJ,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                        - rdtype(2.0) * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                        - rdtype(1.0) * gmtr.GMTR_a[i-1,j,k0,l,AI,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                        - rdtype(1.0) * gmtr.GMTR_a[i-1,j-1,k0,l,AJ,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        - rdtype(2.0) * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        - rdtype(1.0) * gmtr.GMTR_a[i-1,j,k0,l,AI,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                    )

                    # im1jm1
                    self.OPRT_coef_lap[i, j, k0, l, 5] += gmtr.GMTR_t[i-1,j-1,k0,l,TJ,T_RAREA] * ( 
                        - rdtype(1.0) * gmtr.GMTR_a[i-1,j-1,k0,l,AJ,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                        + rdtype(2.0) * gmtr.GMTR_a[i-1,j,k0,l,AI,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                        + rdtype(1.0) * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,tn] * gmtr.GMTR_a[i-1,j,k0,l,AI,hn] 
                        - rdtype(1.0) * gmtr.GMTR_a[i-1,j-1,k0,l,AJ,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        + rdtype(2.0) * gmtr.GMTR_a[i-1,j,k0,l,AI,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        + rdtype(1.0) * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                    )

                    # ijm1
                    self.OPRT_coef_lap[i, j, k0, l, 6] += gmtr.GMTR_t[i,j-1,k0,l,TJ,T_RAREA] * (
                        + rdtype(1.0) * gmtr.GMTR_a[i,j-1,k0,l,AIJ,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        + rdtype(2.0) * gmtr.GMTR_a[i,j,k0,l,AI,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        - rdtype(1.0) * gmtr.GMTR_a[i,j-1,k0,l,AJ,tn] * gmtr.GMTR_a[i-1,j-1,k0,l,AIJ,hn] 
                        - rdtype(1.0) * gmtr.GMTR_a[i,j-1,k0,l,AIJ,tn] * gmtr.GMTR_a[i,j,k0,l,AI,hn] 
                        - rdtype(2.0) * gmtr.GMTR_a[i,j,k0,l,AI,tn] * gmtr.GMTR_a[i,j,k0,l,AI,hn] 
                        + rdtype(1.0) * gmtr.GMTR_a[i,j-1,k0,l,AJ,tn] * gmtr.GMTR_a[i,j,k0,l,AI,hn] 
                    )

            for i in range(adm.ADM_gall_1d):
                for j in range(adm.ADM_gall_1d):
                    self.OPRT_coef_lap[i, j, k0, l, 0] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / rdtype(12.0)
                    self.OPRT_coef_lap[i, j, k0, l, 1] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / rdtype(12.0)
                    self.OPRT_coef_lap[i, j, k0, l, 2] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / rdtype(12.0)
                    self.OPRT_coef_lap[i, j, k0, l, 3] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / rdtype(12.0)
                    self.OPRT_coef_lap[i, j, k0, l, 4] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / rdtype(12.0)
                    self.OPRT_coef_lap[i, j, k0, l, 5] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / rdtype(12.0)
                    self.OPRT_coef_lap[i, j, k0, l, 6] *= gmtr.GMTR_p[i, j, k0, l, P_RAREA] / rdtype(12.0)


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
                        #     print(self.OPRT_coef_lap_pl[0, k0, l], file=log_file)

                        self.OPRT_coef_lap_pl[0, k0, l] += gmtr.GMTR_t_pl[ijm1, k0, l, T_RAREA] * (
                            + rdtype(1.0) * gmtr.GMTR_a_pl[ijm1, k0, l, tn]  * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            - rdtype(2.0) * gmtr.GMTR_a_pl[ijm1, k0, l, tn2] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            - rdtype(1.0) * gmtr.GMTR_a_pl[ij,   k0, l, tn]  * gmtr.GMTR_a_pl[ij, k0, l, hn]
                        )

                        self.OPRT_coef_lap_pl[0, k0, l] += gmtr.GMTR_t_pl[ij, k0, l, T_RAREA] * (
                            + rdtype(1.0) * gmtr.GMTR_a_pl[ij,   k0, l, tn]  * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            - rdtype(2.0) * gmtr.GMTR_a_pl[ij,   k0, l, tn2] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            - rdtype(1.0) * gmtr.GMTR_a_pl[ijp1, k0, l, tn]  * gmtr.GMTR_a_pl[ij, k0, l, hn]
                        )

                        # with open(std.fname_log, 'a') as log_file:
                        #     print("JJBUG: v, d and l = ", v, d, l, file= log_file)
                        #     print(gmtr.GMTR_a_pl[ij, k0, l, tn2], gmtr.GMTR_a_pl[ij, k0, l, hn], file=log_file)

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij = v
                        ijp1 = v + 1
                        ijm1 = v - 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl
                        if ijm1 == adm.ADM_gmin_pl - 1:
                            ijm1 = adm.ADM_gmax_pl

                        self.OPRT_coef_lap_pl[v, k0, l] += gmtr.GMTR_t_pl[ijm1, k0, l, T_RAREA] * (
                            - rdtype(2.0) * gmtr.GMTR_a_pl[ijm1, k0, l, tn] * gmtr.GMTR_a_pl[ijm1, k0, l, hn]
                            + rdtype(1.0) * gmtr.GMTR_a_pl[ijm1, k0, l, tn2] * gmtr.GMTR_a_pl[ijm1, k0, l, hn]
                            - rdtype(1.0) * gmtr.GMTR_a_pl[ij,   k0, l, tn] * gmtr.GMTR_a_pl[ijm1, k0, l, hn]
                            - rdtype(2.0) * gmtr.GMTR_a_pl[ijm1, k0, l, tn] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            + rdtype(1.0) * gmtr.GMTR_a_pl[ijm1, k0, l, tn2] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            - rdtype(1.0) * gmtr.GMTR_a_pl[ij,   k0, l, tn] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                        )

                        self.OPRT_coef_lap_pl[v, k0, l] += gmtr.GMTR_t_pl[ij, k0, l, T_RAREA] * (
                            + rdtype(1.0) * gmtr.GMTR_a_pl[ij,   k0, l, tn] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            + rdtype(1.0) * gmtr.GMTR_a_pl[ij,   k0, l, tn2] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            + rdtype(2.0) * gmtr.GMTR_a_pl[ijp1, k0, l, tn] * gmtr.GMTR_a_pl[ij, k0, l, hn]
                            + rdtype(1.0) * gmtr.GMTR_a_pl[ij,   k0, l, tn] * gmtr.GMTR_a_pl[ijp1, k0, l, hn]
                            + rdtype(1.0) * gmtr.GMTR_a_pl[ij,   k0, l, tn2] * gmtr.GMTR_a_pl[ijp1, k0, l, hn]
                            + rdtype(2.0) * gmtr.GMTR_a_pl[ijp1, k0, l, tn] * gmtr.GMTR_a_pl[ijp1, k0, l, hn]
                        )

                for v in range(adm.ADM_gslf_pl, adm.ADM_gmax_pl + 1):
                    self.OPRT_coef_lap_pl[v, k0, l] *= gmtr.GMTR_p_pl[n, k0, l, P_RAREA] / rdtype(12.0)

        return
    

    def OPRT_diffusion_setup(self, gmtr, rdtype):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("*** setup coefficient of diffusion operator", file=log_file)        
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

        self.OPRT_coef_intp   [:,:,:,:,:,:,:] = rdtype(0.0)  # i, j, KNONE, l, xyz, 3, TIJ
        self.OPRT_coef_diff   [:,:,:,:,:,:]   = rdtype(0.0)  # i, j, KNONE, l, xyz, 6
        self.OPRT_coef_intp_pl[:,:,:,:,:]     = rdtype(0.0)  # ij,   KNONE, l, xyz, 3     [0,:,:,:,:] never used.
        self.OPRT_coef_diff_pl[:,:,:,:]       = rdtype(0.0)  # ij,   KNONE, l, xyz        [0,:,:,:] never used.

        for l in range(lall):
            for d in range(nxyz):

                tn = d + TNX
                                # 0  to  16 (expanded grid points)
                for i in range (gmin-1, gmax + 1):
                    for j in range(gmin-1, gmax + 1):

                        self.OPRT_coef_intp[i, j, k0, l, d, 0, TI] = (
                            + gmtr.GMTR_a[i, j, k0, l, AIJ, tn] - gmtr.GMTR_a[i, j, k0, l, AI, tn]
                        ) * rdtype(0.5) * gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA]

                        self.OPRT_coef_intp[i, j, k0, l, d, 1, TI] = (
                            - gmtr.GMTR_a[i, j, k0, l, AI, tn] - gmtr.GMTR_a[i + 1, j, k0, l, AJ, tn]
                        ) * rdtype(0.5) * gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA]

                        self.OPRT_coef_intp[i, j, k0, l, d, 2, TI] = (
                            - gmtr.GMTR_a[i + 1, j, k0, l, AJ, tn] + gmtr.GMTR_a[i, j, k0, l, AIJ, tn]
                        ) * rdtype(0.5) * gmtr.GMTR_t[i, j, k0, l, TI, T_RAREA]


                        self.OPRT_coef_intp[i, j, k0, l, d, 0, TJ] = (
                            + gmtr.GMTR_a[i, j, k0, l, AJ, tn] - gmtr.GMTR_a[i, j, k0, l, AIJ, tn]
                        ) * rdtype(0.5) * gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA]

                        self.OPRT_coef_intp[i, j, k0, l, d, 1, TJ] = (
                            - gmtr.GMTR_a[i, j, k0, l, AIJ, tn] + gmtr.GMTR_a[i, j + 1, k0, l, AI, tn]
                        ) * rdtype(0.5) * gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA]

                        self.OPRT_coef_intp[i, j, k0, l, d, 2, TJ] = (
                            + gmtr.GMTR_a[i, j + 1, k0, l, AI, tn] + gmtr.GMTR_a[i, j, k0, l, AJ, tn]
                        ) * rdtype(0.5) * gmtr.GMTR_t[i, j, k0, l, TJ, T_RAREA]

        for l in range(lall):
            for d in range(nxyz):

                hn = d + HNX

                                # 1  to  16 (inner grid points)
                for i in range (gmin, gmax + 1):
                    for j in range(gmin, gmax + 1):

                        self.OPRT_coef_diff[i, j, k0, l, d, 0] = (   ##### CCCHHHEEECCCKKK
                            + gmtr.GMTR_a[i, j, k0, l, AIJ, hn]
                            * rdtype(0.5)
                            * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                        )

                        self.OPRT_coef_diff[i, j, k0, l, d, 1] = (
                            + gmtr.GMTR_a[i, j, k0, l, AJ, hn]
                            * rdtype(0.5)
                            * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                        )

                        self.OPRT_coef_diff[i, j, k0, l, d, 2] = (
                            - gmtr.GMTR_a[i - 1, j, k0, l, AI, hn]
                            * rdtype(0.5)
                            * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                        )

                        self.OPRT_coef_diff[i, j, k0, l, d, 3] = (
                            - gmtr.GMTR_a[i - 1, j - 1, k0, l, AIJ, hn]
                            * rdtype(0.5)
                            * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                        )

                        self.OPRT_coef_diff[i, j, k0, l, d, 4] = (
                            - gmtr.GMTR_a[i, j - 1, k0, l, AJ, hn]
                            * rdtype(0.5)
                            * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                        )

                        self.OPRT_coef_diff[i, j, k0, l, d, 5] = (
                            + gmtr.GMTR_a[i, j, k0, l, AI, hn]
                            * rdtype(0.5)
                            * gmtr.GMTR_p[i, j, k0, l, P_RAREA]
                        )

                        # if i == 16 and j == 15 and l == 4:
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print(f"OPRT_coef_diff[{i}, {j}, :, {d}, {l}] = ", self.OPRT_coef_diff[i, j, :, d, l], file=log_file)
                        #         print(f"gmtr.GMTR_a[{i}, {j}, k0, {l} AIJ, hn]", gmtr.GMTR_a[i, j, k0, l, AIJ, hn],file=log_file)
                        #         print(f"gmtr.GMTR_p[{i}, {j}, k0, {l} AIJ, hn]", gmtr.GMTR_p[i, j, k0, l, P_RAREA],file=log_file)

                if adm.ADM_have_sgp[l]:
                    #self.OPRT_coef_diff[1, 1, 5, d, l] = rdtype(0.0)   # this might be correct, overwriting the last (6th) value with zero
                    self.OPRT_coef_diff[1, 1, k0, l, d, 4] = rdtype(0.0)    # this matches the original code, but could it be a bug?

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for d in range(adm.ADM_nxyz):
                    hn  = d + HNX 
                    tn  = d + TNX 
                    tn2 = d + TN2X

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):  # 1 to 5  (2 to 6 in f)
                        ij   = v
                        ijp1 = v + 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl

                        self.OPRT_coef_intp_pl[v, k0, l, d, 0] = -gmtr.GMTR_a_pl[ijp1, k0, l, tn] + gmtr.GMTR_a_pl[ij, k0, l, tn]
                        self.OPRT_coef_intp_pl[v, k0, l, d, 1] =  gmtr.GMTR_a_pl[ij, k0, l, tn] + gmtr.GMTR_a_pl[ij, k0, l, tn2]
                        self.OPRT_coef_intp_pl[v, k0, l, d, 2] =  gmtr.GMTR_a_pl[ij, k0, l, tn2] - gmtr.GMTR_a_pl[ijp1, k0, l, tn]

                        self.OPRT_coef_intp_pl[v, k0, l, d, :] *= rdtype(0.5) * gmtr.GMTR_t_pl[v, k0, l, T_RAREA]

                        self.OPRT_coef_diff_pl[v, k0, l, d] = gmtr.GMTR_a_pl[v, k0, l, hn] * rdtype(0.5) * gmtr.GMTR_p_pl[n, k0, l, P_RAREA]  
                        # Check if v is correct (probably ok. v-1 and v in fortran, but both python and fortran stores coef in 1-5, while GMTR are from 1-5 and 2-6)
                        # This does not give v=0 value which is likely never used (better keep it for consistency).   Tomoki Miyakawa   2025/04/02  

        return
    
    def OPRT_divergence(self, 
            scl, scl_pl,                #[OUT]
            vx, vx_pl,                  #[IN]           
            vy, vy_pl,                  #[IN]     
            vz, vz_pl,                  #[IN]
            coef_div, coef_div_pl,      #[IN]
            grd, rdtype):

        prf.PROF_rapstart('OPRT_divergence', 2)        

        # This should not be done, because it will be detached from the original array handed to the function
        #scl = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall), dtype=rdtype)
        #scl_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kall, adm.ADM_lall_pl), dtype=rdtype)

        scl[:, :, :, :] = rdtype(0.0)
        scl_pl[:, :, :] = rdtype(0.0)

        #gall   = adm.ADM_gall
        iall  = adm.ADM_gall_1d
        jall  = adm.ADM_gall_1d
        kall   = adm.ADM_kall
        lall   = adm.ADM_lall
        k0    = adm.ADM_K0


        # --- Scalar divergence calculation
        isl   = slice(1, iall - 1)
        isl_p = slice(2, iall)       # isl + 1
        isl_m = slice(0, iall - 2)   # isl - 1

        jsl   = slice(1, jall - 1)
        jsl_p = slice(2, jall)       # jsl + 1
        jsl_m = slice(0, jall - 2)   # jsl - 1

        # Define an axis insertion helper for (i, j, 1, l)
        #insert_axis = lambda x: x[:, :, np.newaxis, :]

        scl[isl, jsl, :, :] = (
            coef_div[isl, jsl, :, :, grd.GRD_XDIR, 0] * vx[isl,     jsl,     :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_XDIR, 1] * vx[isl_p,   jsl,     :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_XDIR, 2] * vx[isl_p,   jsl_p,   :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_XDIR, 3] * vx[isl,     jsl_p,   :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_XDIR, 4] * vx[isl_m,   jsl,     :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_XDIR, 5] * vx[isl_m,   jsl_m,   :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_XDIR, 6] * vx[isl,     jsl_m,   :, :]
        )

        scl[isl, jsl, :, :] += (
            coef_div[isl, jsl, :, :, grd.GRD_YDIR, 0] * vy[isl,     jsl,     :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_YDIR, 1] * vy[isl_p,   jsl,     :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_YDIR, 2] * vy[isl_p,   jsl_p,   :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_YDIR, 3] * vy[isl,     jsl_p,   :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_YDIR, 4] * vy[isl_m,   jsl,     :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_YDIR, 5] * vy[isl_m,   jsl_m,   :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_YDIR, 6] * vy[isl,     jsl_m,   :, :]
        )

        scl[isl, jsl, :, :] += (
            coef_div[isl, jsl, :, :, grd.GRD_ZDIR, 0] * vz[isl,     jsl,     :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_ZDIR, 1] * vz[isl_p,   jsl,     :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_ZDIR, 2] * vz[isl_p,   jsl_p,   :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_ZDIR, 3] * vz[isl,     jsl_p,   :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_ZDIR, 4] * vz[isl_m,   jsl,     :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_ZDIR, 5] * vz[isl_m,   jsl_m,   :, :] +
            coef_div[isl, jsl, :, :, grd.GRD_ZDIR, 6] * vz[isl,     jsl_m,   :, :]
        )

        n = adm.ADM_gslf_pl 
        v_range = slice(adm.ADM_gslf_pl, adm.ADM_gmax_pl + 1)
        scl_pl[n,:,:] = np.where(adm.ADM_have_pl,  np.sum(
                (coef_div_pl[v_range, :, :, grd.GRD_XDIR] * vx_pl[v_range, :, :]) +
                (coef_div_pl[v_range, :, :, grd.GRD_YDIR] * vy_pl[v_range, :, :]) +
                (coef_div_pl[v_range, :, :, grd.GRD_ZDIR] * vz_pl[v_range, :, :]), axis=0), 
                rdtype(0.0))

        prf.PROF_rapend('OPRT_divergence', 2) 

        return



    def OPRT_gradient_ij(self, grad, grad_pl, scl, scl_pl, coef_grad, coef_grad_pl, grd, rdtype):

        prf.PROF_rapstart('OPRT_gradient', 2)

        iall  = adm.ADM_gall_1d
        jall  = adm.ADM_gall_1d  #18
        kall   = adm.ADM_kall
        lall   = adm.ADM_lall
        k0    = adm.ADM_K0

        #grad = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall, adm.ADM_nxyz), dtype=rdtype)
        #grad_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kall, adm.ADM_lall_pl, adm.ADM_nxyz), dtype=rdtype)

        for l in range(lall):
            for k in range(kall):

                isl    = slice(1, iall - 1)
                isl_p  = slice(2, iall    )
                isl_m  = slice(0, iall - 2)
                jsl    = slice(1, jall - 1)
                jsl_p  = slice(2, jall    )
                jsl_m  = slice(0, jall - 2)

                # XDIR component
                grad[isl, jsl, k, l, grd.GRD_XDIR] = (
                    coef_grad[isl, jsl, k0, l, grd.GRD_XDIR, 0] * scl[isl   , jsl   , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_XDIR, 1] * scl[isl_p , jsl   , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_XDIR, 2] * scl[isl_p , jsl_p , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_XDIR, 3] * scl[isl   , jsl_p , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_XDIR, 4] * scl[isl_m , jsl   , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_XDIR, 5] * scl[isl_m , jsl_m , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_XDIR, 6] * scl[isl   , jsl_m , k, l]
                )

                # YDIR component
                grad[isl, jsl, k, l, grd.GRD_YDIR] = (
                    coef_grad[isl, jsl, k0, l, grd.GRD_YDIR, 0] * scl[isl   , jsl   , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_YDIR, 1] * scl[isl_p , jsl   , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_YDIR, 2] * scl[isl_p , jsl_p , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_YDIR, 3] * scl[isl   , jsl_p , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_YDIR, 4] * scl[isl_m , jsl   , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_YDIR, 5] * scl[isl_m , jsl_m , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_YDIR, 6] * scl[isl   , jsl_m , k, l]
                )

                # ZDIR component
                grad[isl, jsl, k, l, grd.GRD_ZDIR] = (
                    coef_grad[isl, jsl, k0, l, grd.GRD_ZDIR, 0] * scl[isl   , jsl   , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_ZDIR, 1] * scl[isl_p , jsl   , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_ZDIR, 2] * scl[isl_p , jsl_p , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_ZDIR, 3] * scl[isl   , jsl_p , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_ZDIR, 4] * scl[isl_m , jsl   , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_ZDIR, 5] * scl[isl_m , jsl_m , k, l] +
                    coef_grad[isl, jsl, k0, l, grd.GRD_ZDIR, 6] * scl[isl   , jsl_m , k, l]
                )

                # if k == 41 and l == 0:
                #     with open(std.fname_log, 'a') as log_file:
                #         print("gradelements", file=log_file)

                #         print("coef_grad X", file=log_file)
                #         print(coef_grad[1, 16, :, grd.GRD_XDIR,l], file=log_file)
                #         print("scl", file=log_file)
                #         print(scl[0:3, 16, k, l], file=log_file)
                #         print("grad X", file=log_file)
                #         print(grad[1, 16, k, l, grd.GRD_XDIR], file=log_file)

                #         print("coef_grad Y", file=log_file)
                #         print(coef_grad[1, 16, :, grd.GRD_YDIR,l], file=log_file)
                #         print("scl", file=log_file)
                #         print(scl[0:3, 16, k, l], file=log_file)
                #         print("grad Y", file=log_file)
                #         print(grad[1, 16, k, l, grd.GRD_YDIR], file=log_file)

                #         print("coef_grad Z", file=log_file)
                #         print(coef_grad[1, 16, :, grd.GRD_ZDIR,l], file=log_file)
                #         print("scl", file=log_file)
                #         print(scl[0:3, 16, k, l], file=log_file)
                #         print("grad Z", file=log_file)
                #         print(grad[1, 16, k, l, grd.GRD_ZDIR], file=log_file)


        #with open(std.fname_log, 'a') as log_file:  
        #   print("grad before r2p?", grad[1, 16, 41, 2, grd.GRD_XDIR], file=log_file)

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for k in range(adm.ADM_kall):
                    grad_pl[:, k, l, grd.GRD_XDIR] = rdtype(0.0)
                    grad_pl[:, k, l, grd.GRD_YDIR] = rdtype(0.0)
                    grad_pl[:, k, l, grd.GRD_ZDIR] = rdtype(0.0)
                                        #  0                    5   + 1  (in p)
                    for v in range(adm.ADM_gslf_pl, adm.ADM_gmax_pl + 1):    # 0 to 5  (in p)
                        grad_pl[n, k, l, grd.GRD_XDIR] += coef_grad_pl[v, k0, l, grd.GRD_XDIR] * scl_pl[v, k, l]
                        grad_pl[n, k, l, grd.GRD_YDIR] += coef_grad_pl[v, k0, l, grd.GRD_YDIR] * scl_pl[v, k, l]
                        grad_pl[n, k, l, grd.GRD_ZDIR] += coef_grad_pl[v, k0, l, grd.GRD_ZDIR] * scl_pl[v, k, l]

                    # if k == 41 and l == 0:
                    #     with open(std.fname_log, 'a') as log_file:
                    #         print("grad_pl elements", file=log_file)
                    #         print("coef_grad_pl X", file=log_file)
                    #         print(coef_grad_pl[:,grd.GRD_XDIR,l], file=log_file)
                    #         print("scl_pl", file=log_file)
                    #         print(scl_pl[:, k, l], file=log_file)
                    #         print("grad_pl X", file=log_file)
                    #         print(grad_pl[n, k, l, grd.GRD_XDIR], file=log_file)

                    #         print("coef_grad_pl Y", file=log_file)
                    #         print(coef_grad_pl[:,grd.GRD_YDIR,l], file=log_file)
                    #         print("scl_pl", file=log_file)
                    #         print(scl_pl[:, k, l], file=log_file)
                    #         print("grad_pl Y", file=log_file)
                    #         print(grad_pl[n, k, l, grd.GRD_YDIR], file=log_file)

                    #         print("coef_grad_pl Z", file=log_file)
                    #         print(coef_grad_pl[:,grd.GRD_ZDIR,l], file=log_file)
                    #         print("scl_pl", file=log_file)
                    #         print(scl_pl[:, k, l], file=log_file)
                    #         print("grad_pl Z", file=log_file)
                    #         print(grad_pl[n, k, l, grd.GRD_ZDIR], file=log_file)

        #else:
        #    grad_pl[:, :, :, :] = rdtype(0.0)

        prf.PROF_rapend('OPRT_gradient', 2)

        return


    def OPRT_gradient(self, grad, grad_pl, scl, scl_pl, coef_grad, coef_grad_pl, grd, rdtype):

        prf.PROF_rapstart('OPRT_gradient', 2)

        grad.fill(rdtype(0.0))  ### TTT

        iall  = adm.ADM_gall_1d
        jall  = adm.ADM_gall_1d  #18
        kall   = adm.ADM_kall
        lall   = adm.ADM_lall
        k0    = adm.ADM_K0

        # Define central and shifted slices
        isl    = slice(1, iall - 1)
        isl_p  = slice(2, iall)
        isl_m  = slice(0, iall - 2)
        jsl    = slice(1, jall - 1)
        jsl_p  = slice(2, jall)
        jsl_m  = slice(0, jall - 2)

        # Extract all 7 stencil values for scl
        # Shape of each: (i, j, k, l)
        scl_stencils = [
            scl[isl,   jsl,   :, :],
            scl[isl_p, jsl,   :, :],
            scl[isl_p, jsl_p, :, :],
            scl[isl,   jsl_p, :, :],
            scl[isl_m, jsl,   :, :],
            scl[isl_m, jsl_m, :, :],
            scl[isl,   jsl_m, :, :]
        ]  # List of 7 arrays

        # # Stack to shape: (i, j, 7, k, l)
        # scl_stack = np.stack(scl_stencils, axis=2)

        # Stack to shape: (i, j, k, l, 7)
        scl_stack = np.stack(scl_stencils, axis=4)

        # # Vectorize for each spatial direction
        # for d in [grd.GRD_XDIR, grd.GRD_YDIR, grd.GRD_ZDIR]:
        #     # coef_grad[isl, jsl, :, d, :] → (i, j, 7, l)
        #     coef = coef_grad[isl, jsl, :, d, :]             # (i, j, 7, l)
        #     coef = coef[:, :, :, np.newaxis, :]             # → (i, j, 7, 1, l)
            
        #     # scl_stack already (i, j, 7, k, l)
        #     grad[isl, jsl, :, :, d] = np.sum(coef * scl_stack, axis=2)  # sum over stencil index

        coef = coef_grad[isl, jsl, :, :, grd.GRD_XDIR, :] 
        #coef = coef[:, :, :, np.newaxis, :]  
        grad[isl, jsl, :, :, grd.GRD_XDIR] = np.sum(coef * scl_stack, axis=4) 
        coef = coef_grad[isl, jsl, :, :, grd.GRD_YDIR, :] 
        #coef = coef[:, :, :, np.newaxis, :]  
        grad[isl, jsl, :, :, grd.GRD_YDIR] = np.sum(coef * scl_stack, axis=4) 
        coef = coef_grad[isl, jsl, :, :, grd.GRD_ZDIR, :] 
        #coef = coef[:, :, :, np.newaxis, :]  
        grad[isl, jsl, :, :, grd.GRD_ZDIR] = np.sum(coef * scl_stack, axis=4) 

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for k in range(adm.ADM_kall):
                    grad_pl[:, k, l, grd.GRD_XDIR] = rdtype(0.0)
                    grad_pl[:, k, l, grd.GRD_YDIR] = rdtype(0.0)
                    grad_pl[:, k, l, grd.GRD_ZDIR] = rdtype(0.0)
                                        #  0                    5   + 1  (in p)
                    for v in range(adm.ADM_gslf_pl, adm.ADM_gmax_pl + 1):    # 0 to 5  (in p)
                        grad_pl[n, k, l, grd.GRD_XDIR] += coef_grad_pl[v, k0, l, grd.GRD_XDIR] * scl_pl[v, k, l]
                        grad_pl[n, k, l, grd.GRD_YDIR] += coef_grad_pl[v, k0, l, grd.GRD_YDIR] * scl_pl[v, k, l]
                        grad_pl[n, k, l, grd.GRD_ZDIR] += coef_grad_pl[v, k0, l, grd.GRD_ZDIR] * scl_pl[v, k, l]

        #else:
        #    grad_pl[:, :, :, :] = rdtype(0.0)

        prf.PROF_rapend('OPRT_gradient', 2)

        return

    def OPRT_horizontalize_vec_ij(self, 
            vx, vx_pl,        #[INOUT]
            vy, vy_pl,        #[INOUT]
            vz, vz_pl,        #[INOUT]
            grd, rdtype):

        if grd.GRD_grid_type == grd.GRD_grid_type_on_plane:
            return

        prf.PROF_rapstart('OPRT_horizontalize_vec', 2)

        #with open(std.fname_log, 'a') as log_file:
        #    print("OPRT_horizontalize_vec", file=log_file)

        rscale = grd.GRD_rscale
        #gall   = adm.ADM_gall
        iall  = adm.ADM_gall_1d
        jall  = adm.ADM_gall_1d
        kall   = adm.ADM_kall
        lall   = adm.ADM_lall

        # --- Project horizontal wind to tangent plane
        for l in range(lall):
            for k in range(kall):   

                isl = slice(1, iall - 1)
                jsl = slice(1, jall - 1)

                # prefetch direction components for clarity
                gx = grd.GRD_x[isl, jsl, 0, l, grd.GRD_XDIR]
                gy = grd.GRD_x[isl, jsl, 0, l, grd.GRD_YDIR]
                gz = grd.GRD_x[isl, jsl, 0, l, grd.GRD_ZDIR]

                # project and remove component along grd.GRD_x
                prd = (
                    vx[isl, jsl, k, l] * gx +
                    vy[isl, jsl, k, l] * gy +
                    vz[isl, jsl, k, l] * gz
                ) / rscale

                vx[isl, jsl, k, l] -= prd * gx / rscale
                vy[isl, jsl, k, l] -= prd * gy / rscale
                vz[isl, jsl, k, l] -= prd * gz / rscale

                #for i in range(iall):   
                #    for j in range(jall):

               #        for k in range(kall):                 
                        # if i == 6 and j == 5 and k == 2 and l == 0:
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("OPRT_horizontalize_vec", file=log_file)
                        #         print("vx, vy, vz:", file=log_file)
                        #         print(vx[i, j, k, l], file=log_file)
                        #         print(vy[i, j, k, l], file=log_file)
                        #         print(vz[i, j, k, l], file=log_file)
                        #         print("grd.GRD_x", file=log_file)
                        #         print(grd.GRD_x[i, j, 0, l, grd.GRD_XDIR], file=log_file)
                        #         print(grd.GRD_x[i, j, 0, l, grd.GRD_YDIR], file=log_file)
                        #         print(grd.GRD_x[i, j, 0, l, grd.GRD_ZDIR], file=log_file)
                        #         print("rscale", file=log_file)
                        #         print(rscale, file=log_file)

                        # prd = (
                        #     vx[i, j, k, l] * grd.GRD_x[i, j, 0, l, grd.GRD_XDIR] / rscale
                        #     + vy[i, j, k, l] * grd.GRD_x[i, j, 0, l, grd.GRD_YDIR] / rscale
                        #     + vz[i, j, k, l] * grd.GRD_x[i, j, 0, l, grd.GRD_ZDIR] / rscale
                        # )
                        # vx[i, j, k, l] -= prd * grd.GRD_x[i, j, 0, l, grd.GRD_XDIR] / rscale
                        # vy[i, j, k, l] -= prd * grd.GRD_x[i, j, 0, l, grd.GRD_YDIR] / rscale
                        # vz[i, j, k, l] -= prd * grd.GRD_x[i, j, 0, l, grd.GRD_ZDIR] / rscale

                        # if i == 6 and j == 5 and k == 2 and l == 0:
                        #     with open(std.fname_log, 'a') as log_file:
                        #         print("horizontalized", file=log_file)
                        #         print("vx, vy, vz", file=log_file)
                        #         print(vx[i, j, k, l], file=log_file)
                        #         print(vy[i, j, k, l], file=log_file)
                        #         print(vz[i, j, k, l], file=log_file)
                        #         print("prd", file=log_file)
                        #         print(prd, file=log_file)

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
            vx_pl[:, :, :] = rdtype(0.0)
            vy_pl[:, :, :] = rdtype(0.0)
            vz_pl[:, :, :] = rdtype(0.0)

        prf.PROF_rapend('OPRT_horizontalize_vec', 2)

        return
    
    def OPRT_horizontalize_vec(self, 
            vx, vx_pl,        #[INOUT]
            vy, vy_pl,        #[INOUT]
            vz, vz_pl,        #[INOUT]
            grd, rdtype):

        if grd.GRD_grid_type == grd.GRD_grid_type_on_plane:
            return

        prf.PROF_rapstart('OPRT_horizontalize_vec', 2)

        #with open(std.fname_log, 'a') as log_file:
        #    print("OPRT_horizontalize_vec", file=log_file)

        rscale = grd.GRD_rscale
        #gall   = adm.ADM_gall
        iall  = adm.ADM_gall_1d
        jall  = adm.ADM_gall_1d
        kall   = adm.ADM_kall
        lall   = adm.ADM_lall

        # --- Core slices (exclude halo) ---
        isl = slice(1, iall - 1)
        jsl = slice(1, jall - 1)

        # --- Direction vector: shape (i, j, l, 3)
        gvec = grd.GRD_x[isl, jsl, 0, :, :]   # (i, j, l, 3)
        gx = gvec[..., grd.GRD_XDIR]
        gy = gvec[..., grd.GRD_YDIR]
        gz = gvec[..., grd.GRD_ZDIR]

        # --- Expand direction vector to (i, j, 1, l)
        gx = gx[:, :, np.newaxis, :]
        gy = gy[:, :, np.newaxis, :]
        gz = gz[:, :, np.newaxis, :]

        # --- Fetch velocity: shape (i, j, k, l)
        vx_sub = vx[isl, jsl, :, :]
        vy_sub = vy[isl, jsl, :, :]
        vz_sub = vz[isl, jsl, :, :]

        # --- Project and subtract vector component
        prd = (vx_sub * gx + vy_sub * gy + vz_sub * gz) / rscale  # (i, j, k, l)

        vx[isl, jsl, :, :] -= prd * gx / rscale
        vy[isl, jsl, :, :] -= prd * gy / rscale
        vz[isl, jsl, :, :] -= prd * gz / rscale


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
            vx_pl[:, :, :] = rdtype(0.0)
            vy_pl[:, :, :] = rdtype(0.0)
            vz_pl[:, :, :] = rdtype(0.0)

        prf.PROF_rapend('OPRT_horizontalize_vec', 2)

        return

    def OPRT_laplacian(self, scl, scl_pl, coef_lap, coef_lap_pl, rdtype):
        return laplacian_np(scl, scl_pl, coef_lap, coef_lap_pl, rdtype)
        # if call_jax:
        #     return laplacian_jax (scl, scl_pl, coef_lap, coef_lap_pl, rdtype)
        # else:
        #     return laplacian_np(scl, scl_pl, coef_lap, coef_lap_pl, rdtype)

    
    def OPRT_diffusion(self, 
                       scl, scl_pl,              #[IN]    
                       kh, kh_pl,                #[IN]    
                       coef_intp, coef_intp_pl,  #[IN]    
                       coef_diff, coef_diff_pl,  #[IN]    
                       grd, rdtype):

        prf.PROF_rapstart('OPRT_diffusion', 2)

        XDIR = grd.GRD_XDIR
        YDIR = grd.GRD_YDIR
        ZDIR = grd.GRD_ZDIR

        gmin = adm.ADM_gmin
        gmax = adm.ADM_gmax
        iall  = adm.ADM_gall_1d
        jall  = adm.ADM_gall_1d
        kall   = adm.ADM_kall
        lall   = adm.ADM_lall
        nxyz = adm.ADM_nxyz
        k0   = adm.ADM_K0
        TI = adm.ADM_TI
        TJ = adm.ADM_TJ

        vt = np.empty((adm.ADM_shapeXYZ + (2,)), dtype=rdtype)
        vt_pl = np.empty((adm.ADM_gall_pl, adm.ADM_nxyz,), dtype=rdtype)


        dscl = np.zeros(adm.ADM_shape, dtype=rdtype)
        dscl_pl = np.zeros(adm.ADM_shape_pl, dtype=rdtype)

        # Inner grid extent for i and j
        isl = slice(0, iall - 1)   # i = 0 to iall-2 (for i+1 access)
        jsl = slice(0, jall - 1)   # j = 0 to jall-2 (for j+1 access)

        # --- Extract and expand scalar fields ---
        scl0        = scl[isl,     jsl,     :, :][:, :, :, :, np.newaxis]  # (i,j,k,l,1)
        scl_ip1     = scl[isl.start+1:isl.stop+1, jsl,     :, :][:, :, :, :, np.newaxis]
        scl_ip1jp1  = scl[isl.start+1:isl.stop+1, jsl.start+1:jsl.stop+1, :, :][:, :, :, :, np.newaxis]
        scl_jp1     = scl[isl,     jsl.start+1:jsl.stop+1, :, :][:, :, :, :, np.newaxis]

        # --- Coefficient slicing to match scl domains ---
        coef_TI = coef_intp[isl, jsl, :, :, :, :, TI]  #  (i,j,k,l,d,3)
        coef_TJ = coef_intp[isl, jsl, :, :, :, :, TJ]

        # --- Expand coefficients for broadcasting ---
        c1_TI = coef_TI[:, :, :, :, :, 0]  # (i,j,k,l,d)
        c2_TI = coef_TI[:, :, :, :, :, 1]
        c3_TI = coef_TI[:, :, :, :, :, 2]

        c1_TJ = coef_TJ[:, :, :, :, :, 0]
        c2_TJ = coef_TJ[:, :, :, :, :, 1]
        c3_TJ = coef_TJ[:, :, :, :, :, 2]

        # --- Compute TI direction stencil ---
        term_TI = (
            (+rdtype(2.0) * c1_TI - c2_TI - c3_TI) * scl0 +
            (-rdtype(1.0) * c1_TI + rdtype(2.0) * c2_TI - c3_TI) * scl_ip1 +
            (-rdtype(1.0) * c1_TI - c2_TI + rdtype(2.0) * c3_TI) * scl_ip1jp1
        ) / rdtype(3.0)

        # --- Compute TJ direction stencil ---
        term_TJ = (
            (+rdtype(2.0) * c1_TJ - c2_TJ - c3_TJ) * scl0 +
            (-rdtype(1.0) * c1_TJ + rdtype(2.0) * c2_TJ - c3_TJ) * scl_ip1jp1 +
            (-rdtype(1.0) * c1_TJ - c2_TJ + rdtype(2.0) * c3_TJ) * scl_jp1
        ) / rdtype(3.0)

        # --- Assign to vt ---
        vt[isl, jsl, :, :, :, TI] = term_TI #.transpose(0, 1, 3, 4, 2)
        vt[isl, jsl, :, :, :, TJ] = term_TJ #.transpose(0, 1, 3, 4, 2)

        # with open(std.fname_log, 'a') as log_file:
        #     print("checkPOINT1", file=log_file)
        #     print("k=2: vt[6, 5, 2, 0, XDIR, :],", vt[6,5,2,0,XDIR,:], file=log_file)
        #     print("k=2: vt[6, 5, 2, 0, YDIR, :],", vt[6,5,2,0,YDIR,:], file=log_file)
        #     print("k=2: vt[6, 5, 2, 0, ZDIR, :],", vt[6,5,2,0,ZDIR,:], file=log_file)
        #     print("k=37: vt[6, 5, 37, 0, XDIR, :],", vt[6,5,37,0,XDIR,:], file=log_file)
        #     print("k=37: vt[6, 5, 37, 0, YDIR, :],", vt[6,5,37,0,YDIR,:], file=log_file)
        #     print("k=37: vt[6, 5, 37, 0, ZDIR, :],", vt[6,5,37,0,ZDIR,:], file=log_file)

                # gminm1 = (ADM_gmin-1-1)*ADM_gall_1d + ADM_gmin-1 in the original fortran code
                # ADM_gmin is 2, the begining of the "inner grid"  (1-based)
                # Thus, gminm1 points to the first grid point of the entire grid flattened into a 1D array
                # In this python code, the equivalent to gminm1 is i=0, j=0 or i=gmin-1, j=gmin-1, 
                #                                  and gminm1+1 is i=1, j=0 or i=gmin, j=gmin-1
                #   (gmin = adm.ADM_gmin = 1 in this python code)
                #   When the western vertex is a pentagon, i=1 j=0 is copied into i=0 j=0
                #   [Tomoki Miyakawa 2025/04/02]



        vt[gmin-1, gmin-1, :, :, :, TI] = (vt[gmin-1, gmin-1, :, :, :, TI] * ppm.pntmask[:, :, 0, None] +
                                           vt[gmin,   gmin-1, :, :, :, TJ] * ppm.pntmask[:, :, 1, None]
                                        )    # Watch results carefully

        # for l in range(lall):
        #     if adm.ADM_have_sgp[l]:
        #         vt[gmin-1, gmin-1, :, l, XDIR, TI] = vt[gmin, gmin-1, :, l, XDIR, TJ]
        #         vt[gmin-1, gmin-1, :, l, YDIR, TI] = vt[gmin, gmin-1, :, l, YDIR, TJ]
        #         vt[gmin-1, gmin-1, :, l, ZDIR, TI] = vt[gmin, gmin-1, :, l, ZDIR, TJ]
            #endif
        # end l loop

                # This puts zero for the first i row plus one more grid point in the original flattened array.
                # This python code uses a 2d array, so the edges will be left undefined if we follow this strictly.
                # The entire array is initialized to zero beforehand instead. [Tomoki Miyakawa 2025/04/02]
                #do g = 1, gmin-1
                #    dscl(i,j,k,l) = 0.0_RP

        sl = slice(gmin, gmax + 1)  # shorthand for indexing
        slp1 = slice(gmin+1, gmax + 2)
        slm1 = slice(gmin-1, gmax)

        kh0  = kh[sl,     sl,     :, :]
        kf1  = rdtype(0.5) * (kh0 + kh[slp1, slp1, :, :])
        kf2  = rdtype(0.5) * (kh0 + kh[sl,   slp1, :, :])
        kf3  = rdtype(0.5) * (kh0 + kh[slm1, sl,   :, :])
        kf4  = rdtype(0.5) * (kh0 + kh[slm1, slm1, :, :])
        kf5  = rdtype(0.5) * (kh0 + kh[sl,   slm1, :, :])
        kf6  = rdtype(0.5) * (kh0 + kh[slp1, sl,   :, :])

        for d in range(nxyz):

            cdiff = coef_diff[sl, sl, :, :, d, :]  # shape (i,j,1,l 6)

            vt_ij_ti      = vt[sl,     sl,     :, :, d, TI]
            vt_ij_tj      = vt[sl,     sl,     :, :, d, TJ]
            vt_im1j_ti    = vt[slm1,   sl,     :, :, d, TI]
            vt_im1jm1_tj  = vt[slm1,   slm1,   :, :, d, TJ]
            vt_im1jm1_ti  = vt[slm1,   slm1,   :, :, d, TI]
            vt_ijm1_tj    = vt[sl,     slm1,   :, :, d, TJ]
            #vt_ip1jp1_ti  = vt[sl+1,   sl+1,   k, d, TI]  #unused

            # Calculate each term using broadcasting
            term1 = kf1 * cdiff[:, :, :, :, 0] * (vt_ij_ti + vt_ij_tj)
            term2 = kf2 * cdiff[:, :, :, :, 1] * (vt_ij_tj + vt_im1j_ti)
            term3 = kf3 * cdiff[:, :, :, :, 2] * (vt_im1j_ti + vt_im1jm1_tj)
            term4 = kf4 * cdiff[:, :, :, :, 3] * (vt_im1jm1_tj + vt_im1jm1_ti)
            term5 = kf5 * cdiff[:, :, :, :, 4] * (vt_im1jm1_ti + vt_ijm1_tj)
            term6 = kf6 * cdiff[:, :, :, :, 5] * (vt_ijm1_tj + vt_ij_ti)

            # sum in to dscl for the X component
            dscl[sl, sl, :, :] += term1 + term2 + term3 + term4 + term5 + term6

                # This puts zero for the last i row and one more grid point before it in the original flattened array.
                # do g = gmax+1, gall
                #    dscl(i,j,k,l) = 0.0_RP

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl  

            for l in range(adm.ADM_lall_pl):
                for k in range(kall):
                    # Interpolate vt_pl using 3-point interpolation
                    for d in range(adm.ADM_nxyz):
                        for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):   #1 to 5
                            ij = v
                            ijp1 = adm.ADM_gmin_pl if v + 1 > adm.ADM_gmax_pl else v + 1

                            c = coef_intp_pl[v, k0, l, d, :]
                            vt_pl[ij, d] = (
                                (rdtype(2.0) * c[0] - c[1] - c[2]) * scl_pl[n, k, l] +
                                (-c[0] + rdtype(2.0) * c[1] - c[2]) * scl_pl[ij, k, l] +
                                (-c[0] - c[1] + rdtype(2.0) * c[2]) * scl_pl[ijp1, k, l]
                            ) / rdtype(3.0)
                    # enddo d

                    # Compute dscl_pl at index n (southernmost grid point)
                    dscl_pl[n, k, l] = rdtype(0.0)
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):  #1 to 5
                        ij = v
                        ijm1 = adm.ADM_gmax_pl if v - 1 < adm.ADM_gmin_pl else v - 1

                        kh_avg = rdtype(0.5) * (kh_pl[n, k, l] + kh_pl[ij, k, l])
                        vt_sum = vt_pl[ijm1, :] + vt_pl[ij, :]
                        dscl_pl[n, k, l] += kh_avg * np.sum(coef_diff_pl[v, k0, l, :] * vt_sum)
                    # enddo v

                # enddo k
            #enddo  l
        #endif

        prf.PROF_rapend('OPRT_diffusion',2)

        return dscl, dscl_pl
    

    def OPRT_diffusion_ij(self, scl, scl_pl, kh, kh_pl, coef_intp, coef_intp_pl, coef_diff, coef_diff_pl, grd, rdtype):

        prf.PROF_rapstart('OPRT_diffusion', 2)

        XDIR = grd.GRD_XDIR
        YDIR = grd.GRD_YDIR
        ZDIR = grd.GRD_ZDIR

        gmin = adm.ADM_gmin
        gmax = adm.ADM_gmax
        iall  = adm.ADM_gall_1d
        jall  = adm.ADM_gall_1d
        kall   = adm.ADM_kall
        k0   = adm.ADM_K0
        lall   = adm.ADM_lall
        nxyz = adm.ADM_nxyz
        TI = adm.ADM_TI
        TJ = adm.ADM_TJ

        vt = np.empty((adm.ADM_shapeXYZ + (2,)), dtype=rdtype)
        vt_pl = np.empty((adm.ADM_gall_pl, adm.ADM_nxyz,), dtype=rdtype)


        dscl = np.zeros(adm.ADM_shape, dtype=rdtype)
        dscl_pl = np.zeros(adm.ADM_shape_pl, dtype=rdtype)



        # Loop only over l, k, d — vectorize over i, j
        for l in range(lall):
            for k in range(kall):
                for d in range(nxyz):
                    
                    # Local slices for clarity
                    scl_k_l     = scl[:, :, k, l]
                    scl_ip1     = np.roll(scl_k_l, shift=-1, axis=0)   # i+1
                    scl_ip1jp1  = np.roll(scl_ip1, shift=-1, axis=1)   # i+1, j+1
                    scl_jp1     = np.roll(scl_k_l, shift=-1, axis=1)   # i,   j+1

                    # Coefficients
                    coef_TI = coef_intp[:, :, k0, l, d, :, TI]  # shape: (i, j, 3)
                    c1, c2, c3 = coef_TI[:, :, 0], coef_TI[:, :, 1], coef_TI[:, :, 2]

                    # Compute vt[..., TI]
                    vt[:, :, k, l, d, TI] = (
                        (+rdtype(2.0) * c1 - c2 - c3) * scl_k_l +
                        (-rdtype(1.0) * c1 + rdtype(2.0) * c2 - c3) * scl_ip1 +
                        (-rdtype(1.0) * c1 - c2 + rdtype(2.0) * c3) * scl_ip1jp1
                    ) / rdtype(3.0)

                    # TJ version
                    coef_TJ = coef_intp[:, :, k0, l, d, :, TJ]
                    c1, c2, c3 = coef_TJ[:, :, 0], coef_TJ[:, :, 1], coef_TJ[:, :, 2]

                    vt[:, :, k, l, d, TJ] = (
                        (+rdtype(2.0) * c1 - c2 - c3) * scl_k_l +
                        (-rdtype(1.0) * c1 + rdtype(2.0) * c2 - c3) * scl_ip1jp1 +
                        (-rdtype(1.0) * c1 - c2 + rdtype(2.0) * c3) * scl_jp1
                    ) / rdtype(3.0)
                    
                #enddo  nxyz

                # gminm1 = (ADM_gmin-1-1)*ADM_gall_1d + ADM_gmin-1 in the original fortran code
                # ADM_gmin is 2, the begining of the "inner grid"  (1-based)
                # Thus, gminm1 points to the first grid point of the entire grid flattened into a 1D array
                # In this python code, the equivalent to gminm1 is i=0, j=0 or i=gmin-1, j=gmin-1, 
                #                                  and gminm1+1 is i=1, j=0 or i=gmin, j=gmin-1
                #   (gmin = adm.ADM_gmin = 1 in this python code)
                #   When the western vertex is a pentagon, i=1 j=0 is copied into i=0 j=0
                #   [Tomoki Miyakawa 2025/04/02]


                #vt[gmin-1, gmin-1, k, l, d, TI] = (vt[gmin-1, gmin-1, k, l, d, TI] * ppm.pntmask[k0, l, 0] +
                #                                   vt[gmin,   gmin-1, k, l, d, TJ] * ppm.pntmask[k0, l, 1]                      
                #                      )

                if adm.ADM_have_sgp[l]:
                   vt[gmin-1, gmin-1, k, l, XDIR, TI] = vt[gmin, gmin-1, k, l, XDIR, TJ]
                   vt[gmin-1, gmin-1, k, l, YDIR, TI] = vt[gmin, gmin-1, k, l, YDIR, TJ]
                   vt[gmin-1, gmin-1, k, l, ZDIR, TI] = vt[gmin, gmin-1, k, l, ZDIR, TJ]
                #endif

                # This puts zero for the first i row plus one more grid point in the original flattened array.
                # This python code uses a 2d array, so the edges will be left undefined if we follow this strictly.
                # The entire array is initialized to zero beforehand instead. [Tomoki Miyakawa 2025/04/02]
                #do g = 1, gmin-1
                #    dscl(i,j,k,l) = 0.0_RP
                #enddo

                sl = slice(gmin, gmax + 1)  # shorthand for indexing
                slp1 = slice(gmin+1, gmax + 2)
                slm1 = slice(gmin-1, gmax)

                kh0  = kh[sl,     sl,     k, l]
                kf1  = rdtype(0.5) * (kh0 + kh[slp1, slp1, k, l])
                kf2  = rdtype(0.5) * (kh0 + kh[sl,   slp1, k, l])
                kf3  = rdtype(0.5) * (kh[slm1, sl,   k, l] + kh0)
                kf4  = rdtype(0.5) * (kh[slm1, slm1, k, l] + kh0)
                kf5  = rdtype(0.5) * (kh[sl,   slm1, k, l] + kh0)
                kf6  = rdtype(0.5) * (kh0 + kh[slp1, sl,   k, l])

                for d in range(nxyz):

                    cdiff = coef_diff[sl, sl, k0, l, d, :]  # shape (i,j,6)

                    vt_ij_ti      = vt[sl,     sl,   k, l, d, TI]
                    vt_ij_tj      = vt[sl,     sl,   k, l, d, TJ]
                    vt_im1j_ti    = vt[slm1,   sl,   k, l, d, TI]
                    vt_im1jm1_tj  = vt[slm1,   slm1, k, l, d, TJ]
                    vt_im1jm1_ti  = vt[slm1,   slm1, k, l, d, TI]
                    vt_ijm1_tj    = vt[sl,     slm1, k, l, d, TJ]
                    #vt_ip1jp1_ti  = vt[sl+1,   sl+1,   k, d, TI]  #unused

                    # Calculate each term using broadcasting
                    term1 = kf1 * cdiff[:, :, 0] * (vt_ij_ti + vt_ij_tj)
                    term2 = kf2 * cdiff[:, :, 1] * (vt_ij_tj + vt_im1j_ti)
                    term3 = kf3 * cdiff[:, :, 2] * (vt_im1j_ti + vt_im1jm1_tj)
                    term4 = kf4 * cdiff[:, :, 3] * (vt_im1jm1_tj + vt_im1jm1_ti)
                    term5 = kf5 * cdiff[:, :, 4] * (vt_im1jm1_ti + vt_ijm1_tj)
                    term6 = kf6 * cdiff[:, :, 5] * (vt_ijm1_tj + vt_ij_ti)

                    # sum in to dscl for the X component
                    dscl[sl, sl, k, l] += term1 + term2 + term3 + term4 + term5 + term6

                #enddo  XDIR YDIR ZDIR

                # This puts zero for the last i row and one more grid point before it in the original flattened array.
                # do g = gmax+1, gall
                #    dscl(i,j,k,l) = 0.0_RP
                # enddo

            #enddo k
        #enddo l

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl  

            for l in range(adm.ADM_lall_pl):
                for k in range(adm.ADM_kall):
                    # Interpolate vt_pl using 3-point interpolation
                    for d in range(adm.ADM_nxyz):
                        for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):   #1 to 5
                            ij = v
                            ijp1 = adm.ADM_gmin_pl if v + 1 > adm.ADM_gmax_pl else v + 1

                            c = coef_intp_pl[v, k0, l, d, :]
                            vt_pl[ij, d] = (
                                (rdtype(2.0) * c[0] - c[1] - c[2]) * scl_pl[n, k, l] +
                                (-c[0] + rdtype(2.0) * c[1] - c[2]) * scl_pl[ij, k, l] +
                                (-c[0] - c[1] + rdtype(2.0) * c[2]) * scl_pl[ijp1, k, l]
                            ) / rdtype(3.0)
                    # enddo d

                    # Compute dscl_pl at index n (southernmost grid point)
                    dscl_pl[n, k, l] = rdtype(0.0)
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):  #1 to 5
                        ij = v
                        ijm1 = adm.ADM_gmax_pl if v - 1 < adm.ADM_gmin_pl else v - 1

                        kh_avg = rdtype(0.5) * (kh_pl[n, k, l] + kh_pl[ij, k, l])
                        vt_sum = vt_pl[ijm1, :] + vt_pl[ij, :]
                        dscl_pl[n, k, l] += kh_avg * np.sum(coef_diff_pl[v, k0, l, :] * vt_sum)
                    # enddo v

                # enddo k
            #enddo  l
        #endif

        prf.PROF_rapend('OPRT_diffusion',2)

        return dscl, dscl_pl


    # vales may change if switched to this
    def OPRT_divdamp_ij(self,
        ddivdx,    ddivdx_pl,     #out
        ddivdy,    ddivdy_pl,     #out
        ddivdz,    ddivdz_pl,     #out
        vx,        vx_pl,         #in
        vy,        vy_pl,         #in
        vz,        vz_pl,         #in
        coef_intp, coef_intp_pl,  #in
        coef_diff, coef_diff_pl,  #in
        cnst, grd, rdtype,        
        ):

        #if ij == 2 and k ==2 and l == 0:
        # with open (std.fname_log, 'a') as log_file:
        #     print(f"checking pl: n, ij, ijp1: ij=:, k=2, l=0", file=log_file)
        #     print("vx_pl", vx_pl[:, 2, 0], file=log_file)
        #     print("vy_pl", vy_pl[:, 2, 0], file=log_file)
        #     print("vz_pl", vz_pl[:, 2, 0], file=log_file)
            

        prf.PROF_rapstart('OPRT_divdamp', 2)

        gall_1d = adm.ADM_gall_1d
        gall_pl = adm.ADM_gall_pl
        #gall    = adm.ADM_gall
        kall    = adm.ADM_kall
        lall    = adm.ADM_lall
        lall_pl = adm.ADM_lall_pl
        k0     = adm.ADM_K0

        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax

        TI    = adm.ADM_TI
        TJ    = adm.ADM_TJ

        XDIR = grd.GRD_XDIR
        YDIR = grd.GRD_YDIR
        ZDIR = grd.GRD_ZDIR

        #ddivdx    = np.zeros((gall_1d, gall_1d, kall, lall,), dtype=rdtype)    
        #ddivdy    = np.zeros((gall_1d, gall_1d, kall, lall,), dtype=rdtype)
        #ddivdz    = np.zeros((gall_1d, gall_1d, kall, lall,), dtype=rdtype)
        #ddivdx_pl = np.zeros((gall_pl, kall, lall_pl,), dtype=rdtype)
        #ddivdy_pl = np.zeros((gall_pl, kall, lall_pl,), dtype=rdtype)
        #ddivdz_pl = np.zeros((gall_pl, kall, lall_pl,), dtype=rdtype)
        #sclt_pl   = np.empty((gall_pl, kall, lall_pl,), dtype=rdtype)
        #sclt      = np.empty((gall_1d, gall_1d, kall, 2,), dtype=rdtype)  # TI and TJ
        #sclt_pl   = np.empty((gall_pl,), dtype=rdtype)
        sclt      = np.full((gall_1d, gall_1d, kall, 2,), cnst.CONST_UNDEF, dtype=rdtype)  # TI and TJ
        sclt_pl   = np.full((gall_pl,), cnst.CONST_UNDEF, dtype=rdtype)

        ddivdx_pl[:,:,:] = rdtype(0.0)
        ddivdy_pl[:,:,:] = rdtype(0.0)
        ddivdz_pl[:,:,:] = rdtype(0.0)

        gmin = adm.ADM_gmin # 1
        gmax = adm.ADM_gmax # 16

        for l in range(lall):
            for k in range(kall):

                # Prepare slices
                # i = slice(0, gmax)       #0 to gmax -1 (15)
                # ip1 = slice(1, gmax+1)
                # j = slice(0, gmax)
                # jp1 = slice(1, gmax+1)
                i = slice(0, gmax+1)     # 0 to 16   # perhaps 1, gmax+1 is enough (inner grids)
                ip1 = slice(1, gmax+2)   # 1 to 17
                j = slice(0, gmax+1)     # 0 to 16
                jp1 = slice(1, gmax+2)   # 1 to 17

                # Get coef_intp for TI and TJ
                c = coef_intp  # shorthand

                # with open (std.fname_log, 'a') as log_file:
                #     log_file.write(f"sclt.shape: {sclt.shape}\n")
                #     log_file.write(f"gmax: {gmax}\n")
                # prc.prc_mpistop(std.io_l, std.fname_log)

                # TI direction
#                sclt[:, :, k, TI] = (
                sclt[i, j, k, TI] = (
                    c[i, j, k0, l, XDIR, 0, TI] * vx[i,   j,   k, l] +
                    c[i, j, k0, l, XDIR, 1, TI] * vx[ip1, j,   k, l] +
                    c[i, j, k0, l, XDIR, 2, TI] * vx[ip1, jp1, k, l] +
                    c[i, j, k0, l, YDIR, 0, TI] * vy[i,   j,   k, l] +
                    c[i, j, k0, l, YDIR, 1, TI] * vy[ip1, j,   k, l] +
                    c[i, j, k0, l, YDIR, 2, TI] * vy[ip1, jp1, k, l] +
                    c[i, j, k0, l, ZDIR, 0, TI] * vz[i,   j,   k, l] +
                    c[i, j, k0, l, ZDIR, 1, TI] * vz[ip1, j,   k, l] +
                    c[i, j, k0, l, ZDIR, 2, TI] * vz[ip1, jp1, k, l]
                )

                # TJ direction
                #sclt[:, :, k, TJ] = (
                sclt[i, j, k, TJ] = (
                    c[i, j, k0, l, XDIR, 0, TJ] * vx[i,   j,   k, l] +
                    c[i, j, k0, l, XDIR, 1, TJ] * vx[ip1, jp1, k, l] +
                    c[i, j, k0, l, XDIR, 2, TJ] * vx[i,   jp1, k, l] +
                    c[i, j, k0, l, YDIR, 0, TJ] * vy[i,   j,   k, l] +
                    c[i, j, k0, l, YDIR, 1, TJ] * vy[ip1, jp1, k, l] +
                    c[i, j, k0, l, YDIR, 2, TJ] * vy[i,   jp1, k, l] +
                    c[i, j, k0, l, ZDIR, 0, TJ] * vz[i,   j,   k, l] +
                    c[i, j, k0, l, ZDIR, 1, TJ] * vz[ip1, jp1, k, l] +
                    c[i, j, k0, l, ZDIR, 2, TJ] * vz[i,   jp1, k, l]
                )

                if adm.ADM_have_sgp[l]:
                    sclt[0, 0, k, TI] = sclt[1, 0, k, TJ]
                #endif

                
                #sl = slice(1, gmax + 1)  # equivalent to Fortran 2:gmax  # could go to (1, gmax+2), but probably unnecessary 
                sl = slice(1, gmax + 2)  # equivalent to Fortran 2:gmax  # could go to (1, gmax+2), but probably unnecessary    


                # Precompute shifted slices for reusability
                sl_i   = sl
                #sl_im1 = slice(0, gmax)       # i - 1
                sl_im1 = slice(0, gmax+1)       # i - 1
                sl_j   = sl
                #sl_jm1 = slice(0, gmax)       # j - 1
                sl_jm1 = slice(0, gmax+1)

                # ddivdx
                ddivdx[sl_i, sl_j, k, l] = (
                    coef_diff[sl_i, sl_j, k0, l, XDIR, 0] * (sclt[sl_i, sl_j, k, TI] + sclt[sl_i, sl_j, k, TJ]) +
                    coef_diff[sl_i, sl_j, k0, l, XDIR, 1] * (sclt[sl_i, sl_j, k, TJ] + sclt[sl_im1, sl_j, k, TI]) +
                    coef_diff[sl_i, sl_j, k0, l, XDIR, 2] * (sclt[sl_im1, sl_j, k, TI] + sclt[sl_im1, sl_jm1, k, TJ]) +
                    coef_diff[sl_i, sl_j, k0, l, XDIR, 3] * (sclt[sl_im1, sl_jm1, k, TJ] + sclt[sl_im1, sl_jm1, k, TI]) +
                    coef_diff[sl_i, sl_j, k0, l, XDIR, 4] * (sclt[sl_im1, sl_jm1, k, TI] + sclt[sl_i, sl_jm1, k, TJ]) +
                    coef_diff[sl_i, sl_j, k0, l, XDIR, 5] * (sclt[sl_i, sl_jm1, k, TJ] + sclt[sl_i, sl_j, k, TI])
                )

                # ddivdy
                ddivdy[sl_i, sl_j, k, l] = (
                    coef_diff[sl_i, sl_j, k0, l, YDIR, 0] * (sclt[sl_i, sl_j, k, TI] + sclt[sl_i, sl_j, k, TJ]) +
                    coef_diff[sl_i, sl_j, k0, l, YDIR, 1] * (sclt[sl_i, sl_j, k, TJ] + sclt[sl_im1, sl_j, k, TI]) +
                    coef_diff[sl_i, sl_j, k0, l, YDIR, 2] * (sclt[sl_im1, sl_j, k, TI] + sclt[sl_im1, sl_jm1, k, TJ]) +
                    coef_diff[sl_i, sl_j, k0, l, YDIR, 3] * (sclt[sl_im1, sl_jm1, k, TJ] + sclt[sl_im1, sl_jm1, k, TI]) +
                    coef_diff[sl_i, sl_j, k0, l, YDIR, 4] * (sclt[sl_im1, sl_jm1, k, TI] + sclt[sl_i, sl_jm1, k, TJ]) +
                    coef_diff[sl_i, sl_j, k0, l, YDIR, 5] * (sclt[sl_i, sl_jm1, k, TJ] + sclt[sl_i, sl_j, k, TI])
                )

                # ddivdz
                ddivdz[sl_i, sl_j, k, l] = (
                    coef_diff[sl_i, sl_j, k0, l, ZDIR, 0] * (sclt[sl_i, sl_j, k, TI]     + sclt[sl_i, sl_j, k, TJ]) +
                    coef_diff[sl_i, sl_j, k0, l, ZDIR, 1] * (sclt[sl_i, sl_j, k, TJ]     + sclt[sl_im1, sl_j, k, TI]) +
                    coef_diff[sl_i, sl_j, k0, l, ZDIR, 2] * (sclt[sl_im1, sl_j, k, TI]   + sclt[sl_im1, sl_jm1, k, TJ]) +
                    coef_diff[sl_i, sl_j, k0, l, ZDIR, 3] * (sclt[sl_im1, sl_jm1, k, TJ] + sclt[sl_im1, sl_jm1, k, TI]) +
                    coef_diff[sl_i, sl_j, k0, l, ZDIR, 4] * (sclt[sl_im1, sl_jm1, k, TI] + sclt[sl_i, sl_jm1, k, TJ]) +
                    coef_diff[sl_i, sl_j, k0, l, ZDIR, 5] * (sclt[sl_i, sl_jm1, k, TJ]   + sclt[sl_i, sl_j, k, TI])
                )

            #end  k loop
        #end  l loop

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(lall_pl):
                for k in range(kall):

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij = v
                        ijp1 = v + 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl  # cyclic wrap

                        sclt_pl[ij] = (
                            coef_intp_pl[v, k0, l, XDIR, 0] * vx_pl[n,    k, l] +
                            coef_intp_pl[v, k0, l, XDIR, 1] * vx_pl[ij,   k, l] +
                            coef_intp_pl[v, k0, l, XDIR, 2] * vx_pl[ijp1, k, l] +

                            coef_intp_pl[v, k0, l, YDIR, 0] * vy_pl[n,    k, l] +
                            coef_intp_pl[v, k0, l, YDIR, 1] * vy_pl[ij,   k, l] +
                            coef_intp_pl[v, k0, l, YDIR, 2] * vy_pl[ijp1, k, l] +

                            coef_intp_pl[v, k0, l, ZDIR, 0] * vz_pl[n,    k, l] +
                            coef_intp_pl[v, k0, l, ZDIR, 1] * vz_pl[ij,   k, l] +
                            coef_intp_pl[v, k0, l, ZDIR, 2] * vz_pl[ijp1, k, l]
                        )

                        # if ij == 2 and k ==2 and l == 0:
                        #     with open (std.fname_log, 'a') as log_file:
                        #         print(f"checking vx_pl n, ij, ijp1: ij={ij}, k={k}, l={l}", file=log_file)
                        #         print("coef_intp_pl[v, :, XDIR, l] = ", coef_intp_pl[v, :, XDIR, l], file=log_file)
                        #         print(vx_pl[n, k, l], vx_pl[ij, k, l], vx_pl[ijp1, k, l], file=log_file)
                        #         print(vy_pl[n, k, l], vy_pl[ij, k, l], vy_pl[ijp1, k, l], file=log_file)
                        #         print(vz_pl[n, k, l], vz_pl[ij, k, l], vz_pl[ijp1, k, l], file=log_file)

                    # if k == 2 or k == 10:
                    #     with open (std.fname_log, 'a') as log_file:
                    #         print("l= ", l, "k= ", k, "sclt_pl[:] = ", sclt_pl[:], file=log_file)
                    #         #print("vx_pl[n, k, l] = ", vx_pl[n, k, l], file=log_file)

                    # end loop v

                    # with open (std.fname_log, 'a') as log_file:
                    #     log_file.write(f"coef_diff_pl shape: {coef_diff_pl.shape}\n")
                    #     log_file.write(f"sclt_pl shape: {sclt_pl.shape}\n")
                    #     #log_file.write(f"kimn, kmax: {kmin}, {kmax}\n")
                    #     prc.prc_mpistop(std.io_l, std.fname_log)


                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):   # 1 to 5
                        ij = v
                        ijm1 = v - 1
                        if ijm1 == adm.ADM_gmin_pl - 1:
                            ijm1 = adm.ADM_gmax_pl  # cyclic wrap

                        ddivdx_pl[n, k, l] += coef_diff_pl[v, k0, l, XDIR] * (sclt_pl[ijm1] + sclt_pl[ij])
                        ddivdy_pl[n, k, l] += coef_diff_pl[v, k0, l, YDIR] * (sclt_pl[ijm1] + sclt_pl[ij])
                        ddivdz_pl[n, k, l] += coef_diff_pl[v, k0, l, ZDIR] * (sclt_pl[ijm1] + sclt_pl[ij])
                        #check v ranges of coef_diff_pl and coef_intp_pl, and sclt_pl, vx_pl, vy_pl, vz_pl
                    # end loop v

                # end loop k
            # end loop l
        #endif
        prf.PROF_rapend('OPRT_divdamp', 2)

        return

    def OPRT_divdamp(self,
        ddivdx,    ddivdx_pl,     #out
        ddivdy,    ddivdy_pl,     #out
        ddivdz,    ddivdz_pl,     #out
        vx,        vx_pl,         #in
        vy,        vy_pl,         #in
        vz,        vz_pl,         #in
        coef_intp, coef_intp_pl,  #in
        coef_diff, coef_diff_pl,  #in
        cnst, grd, rdtype,        
        ):

        prf.PROF_rapstart('OPRT_divdamp', 2)

        gall_1d = adm.ADM_gall_1d
        gall_pl = adm.ADM_gall_pl
        kall    = adm.ADM_kall
        lall    = adm.ADM_lall
        lall_pl = adm.ADM_lall_pl
        k0     = adm.ADM_K0

        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax

        TI    = adm.ADM_TI
        TJ    = adm.ADM_TJ

        XDIR = grd.GRD_XDIR
        YDIR = grd.GRD_YDIR
        ZDIR = grd.GRD_ZDIR

        sclt      = np.full((adm.ADM_shape + (2,)), cnst.CONST_UNDEF, dtype=rdtype)  # TI and TJ
        sclt_pl   = np.full((gall_pl,), cnst.CONST_UNDEF, dtype=rdtype)

        ddivdx_pl[:,:,:] = rdtype(0.0)
        ddivdy_pl[:,:,:] = rdtype(0.0)
        ddivdz_pl[:,:,:] = rdtype(0.0)

        gmin = adm.ADM_gmin # 1
        gmax = adm.ADM_gmax # 16


        isl    = slice(0, gmax+1)     # 0 to 16
        isl_p1 = slice(1, gmax+2)     # 1 to 17
        jsl    = slice(0, gmax+1)     # 0 to 16
        jsl_p1 = slice(1, gmax+2)     # 1 to 17

        # shorthand
        c = coef_intp

        # TI direction
        sclt[isl, jsl, :, :, TI] = (
            c[isl, jsl, :, :, XDIR, 0, TI] * vx[isl,     jsl,     :, :] +
            c[isl, jsl, :, :, XDIR, 1, TI] * vx[isl_p1,  jsl,     :, :] +
            c[isl, jsl, :, :, XDIR, 2, TI] * vx[isl_p1,  jsl_p1,  :, :] +
            c[isl, jsl, :, :, YDIR, 0, TI] * vy[isl,     jsl,     :, :] +
            c[isl, jsl, :, :, YDIR, 1, TI] * vy[isl_p1,  jsl,     :, :] +
            c[isl, jsl, :, :, YDIR, 2, TI] * vy[isl_p1,  jsl_p1,  :, :] +
            c[isl, jsl, :, :, ZDIR, 0, TI] * vz[isl,     jsl,     :, :] +
            c[isl, jsl, :, :, ZDIR, 1, TI] * vz[isl_p1,  jsl,     :, :] +
            c[isl, jsl, :, :, ZDIR, 2, TI] * vz[isl_p1,  jsl_p1,  :, :]
        )

        # TJ direction
        sclt[isl, jsl, :, :, TJ] = (
            c[isl, jsl, :, :, XDIR, 0, TJ] * vx[isl,     jsl,     :, :] +
            c[isl, jsl, :, :, XDIR, 1, TJ] * vx[isl_p1,  jsl_p1,  :, :] +
            c[isl, jsl, :, :, XDIR, 2, TJ] * vx[isl,     jsl_p1,  :, :] +
            c[isl, jsl, :, :, YDIR, 0, TJ] * vy[isl,     jsl,     :, :] +
            c[isl, jsl, :, :, YDIR, 1, TJ] * vy[isl_p1,  jsl_p1,  :, :] +
            c[isl, jsl, :, :, YDIR, 2, TJ] * vy[isl,     jsl_p1,  :, :] +
            c[isl, jsl, :, :, ZDIR, 0, TJ] * vz[isl,     jsl,     :, :] +
            c[isl, jsl, :, :, ZDIR, 1, TJ] * vz[isl_p1,  jsl_p1,  :, :] +
            c[isl, jsl, :, :, ZDIR, 2, TJ] * vz[isl,     jsl_p1,  :, :]
        )

        isl    = slice(1, gmax+2)      # inner i (1 to gmax+1)
        isl_m1 = slice(0, gmax+1)      # i - 1
        jsl    = slice(1, gmax+2)      # inner j (1 to gmax+1)
        jsl_m1 = slice(0, gmax+1)      # j - 1

        # ddivdx
        ddivdx[isl, jsl, :, :] = (
            coef_diff[isl, jsl, :, :, XDIR, 0] * (sclt[isl,     jsl,     :, :, TI] + sclt[isl,     jsl,     :, :, TJ]) +
            coef_diff[isl, jsl, :, :, XDIR, 1] * (sclt[isl,     jsl,     :, :, TJ] + sclt[isl_m1,  jsl,     :, :, TI]) +
            coef_diff[isl, jsl, :, :, XDIR, 2] * (sclt[isl_m1,  jsl,     :, :, TI] + sclt[isl_m1,  jsl_m1,  :, :, TJ]) +
            coef_diff[isl, jsl, :, :, XDIR, 3] * (sclt[isl_m1,  jsl_m1,  :, :, TJ] + sclt[isl_m1,  jsl_m1,  :, :, TI]) +
            coef_diff[isl, jsl, :, :, XDIR, 4] * (sclt[isl_m1,  jsl_m1,  :, :, TI] + sclt[isl,     jsl_m1,  :, :, TJ]) +
            coef_diff[isl, jsl, :, :, XDIR, 5] * (sclt[isl,     jsl_m1,  :, :, TJ] + sclt[isl,     jsl,     :, :, TI])
        )

        # ddivdy
        ddivdy[isl, jsl, :, :] = (
            coef_diff[isl, jsl, :, :, YDIR, 0] * (sclt[isl,     jsl,     :, :, TI] + sclt[isl,     jsl,     :, :, TJ]) +
            coef_diff[isl, jsl, :, :, YDIR, 1] * (sclt[isl,     jsl,     :, :, TJ] + sclt[isl_m1,  jsl,     :, :, TI]) +
            coef_diff[isl, jsl, :, :, YDIR, 2] * (sclt[isl_m1,  jsl,     :, :, TI] + sclt[isl_m1,  jsl_m1,  :, :, TJ]) +
            coef_diff[isl, jsl, :, :, YDIR, 3] * (sclt[isl_m1,  jsl_m1,  :, :, TJ] + sclt[isl_m1,  jsl_m1,  :, :, TI]) +
            coef_diff[isl, jsl, :, :, YDIR, 4] * (sclt[isl_m1,  jsl_m1,  :, :, TI] + sclt[isl,     jsl_m1,  :, :, TJ]) +
            coef_diff[isl, jsl, :, :, YDIR, 5] * (sclt[isl,     jsl_m1,  :, :, TJ] + sclt[isl,     jsl,     :, :, TI])
        )

        # ddivdz
        ddivdz[isl, jsl, :, :] = (
            coef_diff[isl, jsl, :, :, ZDIR, 0] * (sclt[isl,     jsl,     :, :, TI] + sclt[isl,     jsl,     :, :, TJ]) +
            coef_diff[isl, jsl, :, :, ZDIR, 1] * (sclt[isl,     jsl,     :, :, TJ] + sclt[isl_m1,  jsl,     :, :, TI]) +
            coef_diff[isl, jsl, :, :, ZDIR, 2] * (sclt[isl_m1,  jsl,     :, :, TI] + sclt[isl_m1,  jsl_m1,  :, :, TJ]) +
            coef_diff[isl, jsl, :, :, ZDIR, 3] * (sclt[isl_m1,  jsl_m1,  :, :, TJ] + sclt[isl_m1,  jsl_m1,  :, :, TI]) +
            coef_diff[isl, jsl, :, :, ZDIR, 4] * (sclt[isl_m1,  jsl_m1,  :, :, TI] + sclt[isl,     jsl_m1,  :, :, TJ]) +
            coef_diff[isl, jsl, :, :, ZDIR, 5] * (sclt[isl,     jsl_m1,  :, :, TJ] + sclt[isl,     jsl,     :, :, TI])
        )


        # for l in range(lall):
        #     if adm.ADM_have_sgp[l]:
        #         sclt[0, 0, :, l, TI] = sclt[1, 0, :, l, TJ]
        #     #endif

        #         sl = slice(1, gmax + 2)  # equivalent to Fortran 2:gmax  # could go to (1, gmax+2), but probably unnecessary    

        #         # Precompute shifted slices for reusability
        #         sl_i   = sl
        #         #sl_im1 = slice(0, gmax)       # i - 1
        #         sl_im1 = slice(0, gmax+1)       # i - 1
        #         sl_j   = sl
        #         #sl_jm1 = slice(0, gmax)       # j - 1
        #         sl_jm1 = slice(0, gmax+1)

        #         # ddivdx
        #         ddivdx[sl_i, sl_j, k, l] = (
        #             coef_diff[sl_i, sl_j, 0, XDIR, l] * (sclt[sl_i, sl_j, k, l, TI] + sclt[sl_i, sl_j, k, l, TJ]) +
        #             coef_diff[sl_i, sl_j, 1, XDIR, l] * (sclt[sl_i, sl_j, k, l, TJ] + sclt[sl_im1, sl_j, k, l, TI]) +
        #             coef_diff[sl_i, sl_j, 2, XDIR, l] * (sclt[sl_im1, sl_j, k, l, TI] + sclt[sl_im1, sl_jm1, k, l, TJ]) +
        #             coef_diff[sl_i, sl_j, 3, XDIR, l] * (sclt[sl_im1, sl_jm1, k, l, TJ] + sclt[sl_im1, sl_jm1, k, l, TI]) +
        #             coef_diff[sl_i, sl_j, 4, XDIR, l] * (sclt[sl_im1, sl_jm1, k, l, TI] + sclt[sl_i, sl_jm1, k, l, TJ]) +
        #             coef_diff[sl_i, sl_j, 5, XDIR, l] * (sclt[sl_i, sl_jm1, k, l, TJ] + sclt[sl_i, sl_j, k, l, TI])
        #         )

        #         # ddivdy
        #         ddivdy[sl_i, sl_j, k, l] = (
        #             coef_diff[sl_i, sl_j, 0, YDIR, l] * (sclt[sl_i, sl_j, k, TI] + sclt[sl_i, sl_j, k, l, TJ]) +
        #             coef_diff[sl_i, sl_j, 1, YDIR, l] * (sclt[sl_i, sl_j, k, TJ] + sclt[sl_im1, sl_j, k, l, TI]) +
        #             coef_diff[sl_i, sl_j, 2, YDIR, l] * (sclt[sl_im1, sl_j, k, TI] + sclt[sl_im1, sl_jm1, k, l, TJ]) +
        #             coef_diff[sl_i, sl_j, 3, YDIR, l] * (sclt[sl_im1, sl_jm1, k, TJ] + sclt[sl_im1, sl_jm1, k, l, TI]) +
        #             coef_diff[sl_i, sl_j, 4, YDIR, l] * (sclt[sl_im1, sl_jm1, k, TI] + sclt[sl_i, sl_jm1, k, l, TJ]) +
        #             coef_diff[sl_i, sl_j, 5, YDIR, l] * (sclt[sl_i, sl_jm1, k, TJ] + sclt[sl_i, sl_j, k, l, TI])
        #         )

        #         # ddivdz
        #         ddivdz[sl_i, sl_j, k, l] = (
        #             coef_diff[sl_i, sl_j, 0, ZDIR, l] * (sclt[sl_i, sl_j, k, TI] + sclt[sl_i, sl_j, k, l, TJ]) +
        #             coef_diff[sl_i, sl_j, 1, ZDIR, l] * (sclt[sl_i, sl_j, k, TJ] + sclt[sl_im1, sl_j, k, l, TI]) +
        #             coef_diff[sl_i, sl_j, 2, ZDIR, l] * (sclt[sl_im1, sl_j, k, TI] + sclt[sl_im1, sl_jm1, k, l, TJ]) +
        #             coef_diff[sl_i, sl_j, 3, ZDIR, l] * (sclt[sl_im1, sl_jm1, k, TJ] + sclt[sl_im1, sl_jm1, k, l, TI]) +
        #             coef_diff[sl_i, sl_j, 4, ZDIR, l] * (sclt[sl_im1, sl_jm1, k, TI] + sclt[sl_i, sl_jm1, k, l, TJ]) +
        #             coef_diff[sl_i, sl_j, 5, ZDIR, l] * (sclt[sl_i, sl_jm1, k, TJ] + sclt[sl_i, sl_j, k, l, TI])
        #         )

            #end  k loop
        #end  l loop

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(lall_pl):
                for k in range(kall):

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij = v
                        ijp1 = v + 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl  # cyclic wrap

                        sclt_pl[ij] = (
                            coef_intp_pl[v, k0, l, XDIR, 0] * vx_pl[n, k, l] + 
                            coef_intp_pl[v, k0, l, XDIR, 1] * vx_pl[ij, k, l] +
                            coef_intp_pl[v, k0, l, XDIR, 2] * vx_pl[ijp1, k, l] +

                            coef_intp_pl[v, k0, l, YDIR, 0] * vy_pl[n, k, l] +
                            coef_intp_pl[v, k0, l, YDIR, 1] * vy_pl[ij, k, l] +
                            coef_intp_pl[v, k0, l, YDIR, 2] * vy_pl[ijp1, k, l] +

                            coef_intp_pl[v, k0, l, ZDIR, 0] * vz_pl[n, k, l] +
                            coef_intp_pl[v, k0, l, ZDIR, 1] * vz_pl[ij, k, l] +
                            coef_intp_pl[v, k0, l, ZDIR, 2] * vz_pl[ijp1, k, l]
                        )

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):   # 1 to 5
                        ij = v
                        ijm1 = v - 1
                        if ijm1 == adm.ADM_gmin_pl - 1:
                            ijm1 = adm.ADM_gmax_pl  # cyclic wrap

                        ddivdx_pl[n, k, l] += coef_diff_pl[v, k0, l, XDIR] * (sclt_pl[ijm1] + sclt_pl[ij])
                        ddivdy_pl[n, k, l] += coef_diff_pl[v, k0, l, YDIR] * (sclt_pl[ijm1] + sclt_pl[ij])
                        ddivdz_pl[n, k, l] += coef_diff_pl[v, k0, l, ZDIR] * (sclt_pl[ijm1] + sclt_pl[ij])
                        #check v ranges of coef_diff_pl and coef_intp_pl, and sclt_pl, vx_pl, vy_pl, vz_pl
                    # end loop v

                # end loop k
            # end loop l
        #endif
        prf.PROF_rapend('OPRT_divdamp', 2)

        return



    #> 3D divergence damping operator
    def OPRT3D_divdamp_ij(self,
        ddivdx,    ddivdx_pl,    
        ddivdy,    ddivdy_pl,    
        ddivdz,    ddivdz_pl,    
        rhogvx,    rhogvx_pl,    
        rhogvy,    rhogvy_pl,    
        rhogvz,    rhogvz_pl,    
        rhogw,     rhogw_pl,     
        coef_intp, coef_intp_pl, 
        coef_diff, coef_diff_pl,
        grd, vmtr, rdtype,        
    ):          
         


        # with open (std.fname_log, 'a') as log_file:
        #     print("rhogvx[16, 15, 38, 4] = ", rhogvx[16, 15, 38, 4], file=log_file)
        #     print("rhogvy[16, 15, 38, 4] = ", rhogvy[16, 15, 38, 4], file=log_file)
        #     print("rhogvz[16, 15, 38, 4] = ", rhogvz[16, 15, 38, 4], file=log_file)
        #     print("rhogw[16, 15, 38, 4] = ", rhogw[16, 15, 38, 4], file=log_file)
        #     print("coef_intp[16, 15, 0, 0, :, 4] = ", coef_intp[16, 15, 0, 0, :, 4], file=log_file)
        #     print("coef_intp[16, 15, 1, 1, :, 4] = ", coef_intp[16, 15, 1, 1, :, 4], file=log_file)
        #     print("coef_intp[16, 15, 2, 2, :, 4] = ", coef_intp[16, 15, 2, 2, :, 4], file=log_file)
        #     print("coef_diff[16, 15, :, 0, 4] = ", coef_diff[16, 15, :, 0, 4], file=log_file)
        #     print("coef_diff[16, 15, :, 1, 4] = ", coef_diff[16, 15, :, 1, 4], file=log_file)
        #     print("coef_diff[16, 15, :, 2, 4] = ", coef_diff[16, 15, :, 2, 4], file=log_file)


        prf.PROF_rapstart('OPRT3D_divdamp', 2)

        gall_1d = adm.ADM_gall_1d
        gall_pl = adm.ADM_gall_pl
        #gall    = adm.ADM_gall
        kall    = adm.ADM_kall
        lall    = adm.ADM_lall
        lall_pl = adm.ADM_lall_pl

        TI    = adm.ADM_TI
        TJ    = adm.ADM_TJ

        ##ddivdx    = np.zeros((gall_1d, gall_1d, kall, lall,), dtype=rdtype)    
        #ddivdy    = np.zeros((gall_1d, gall_1d, kall, lall,), dtype=rdtype)
        #ddivdz    = np.zeros((gall_1d, gall_1d, kall, lall,), dtype=rdtype)
        #ddivdx_pl = np.zeros((gall_pl, kall, lall_pl,), dtype=rdtype)
        #ddivdy_pl = np.zeros((gall_pl, kall, lall_pl,), dtype=rdtype)
        #ddivdz_pl = np.zeros((gall_pl, kall, lall_pl,), dtype=rdtype)
        sclt      = np.empty((gall_1d, gall_1d, kall, 2,), dtype=rdtype)  # TI and TJ
        sclt_pl   = np.empty((gall_pl,), dtype=rdtype)
#        sclt_pl   = np.empty((gall_pl, kall, lall_pl,), dtype=rdtype)

        rhogw_vm   = np.empty((gall_1d, gall_1d, kall, lall,), dtype=rdtype)    
        rhogvx_vm  = np.empty((gall_1d, gall_1d, kall,), dtype=rdtype)    
        rhogvy_vm  = np.empty((gall_1d, gall_1d, kall,), dtype=rdtype)    
        rhogvz_vm  = np.empty((gall_1d, gall_1d, kall,), dtype=rdtype)    
        rhogw_vm_pl  = np.empty((gall_pl, kall, lall_pl,), dtype=rdtype)    
        rhogvx_vm_pl = np.empty((gall_pl,), dtype=rdtype)    
        rhogvy_vm_pl = np.empty((gall_pl,), dtype=rdtype)    
        rhogvz_vm_pl = np.empty((gall_pl,), dtype=rdtype)    

        XDIR = grd.GRD_XDIR
        YDIR = grd.GRD_YDIR
        ZDIR = grd.GRD_ZDIR

        gmin = adm.ADM_gmin # 1
        gmax = adm.ADM_gmax # 16
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        k0   = adm.ADM_K0

        for l in range(lall):
            for k in range(kmin + 1, kmax + 1):
                rhogw_vm[:, :, k, l] = (
                    vmtr.VMTR_C2WfactGz[:, :, k, l, 0] * rhogvx[:, :, k,   l] +
                    vmtr.VMTR_C2WfactGz[:, :, k, l, 1] * rhogvx[:, :, k-1, l] +
                    vmtr.VMTR_C2WfactGz[:, :, k, l, 2] * rhogvy[:, :, k,   l] +
                    vmtr.VMTR_C2WfactGz[:, :, k, l, 3] * rhogvy[:, :, k-1, l] +
                    vmtr.VMTR_C2WfactGz[:, :, k, l, 4] * rhogvz[:, :, k,   l] +
                    vmtr.VMTR_C2WfactGz[:, :, k, l, 5] * rhogvz[:, :, k-1, l]
                ) * vmtr.VMTR_RGAMH[:, :, k, l] + rhogw[:, :, k, l] * vmtr.VMTR_RGSQRTH[:, :, k, l]
                #end loop k

            rhogw_vm[:, :, kmin,   l] = rdtype(0.0)
            rhogw_vm[:, :, kmax+1, l] = rdtype(0.0)

        #end loop  l


        # with open (std.fname_log, 'a') as log_file:
        #     print("U1 rhogw_vm[16, 15, 38, 4] = ", rhogw_vm[16, 15, 38, 4], file=log_file)
           

        for l in range(lall):
            for k in range(kmin, kmax + 1):

                rhogvx_vm[:, :, k] = rhogvx[:, :, k, l] * vmtr.VMTR_RGAM[:, :, k, l]
                rhogvy_vm[:, :, k] = rhogvy[:, :, k, l] * vmtr.VMTR_RGAM[:, :, k, l]
                rhogvz_vm[:, :, k] = rhogvz[:, :, k, l] * vmtr.VMTR_RGAM[:, :, k, l]


                # sl = slice(1, gmax+1)     # corresponds to Fortran indices 2:gmax
                # slp = slice(2, gmax+2)  # sl + 1
                sl = slice(0, gmax+1)     # corresponds to Fortran indices 2:gmax
                slp = slice(1, gmax+2)  # sl + 1

                # TI direction
                sclt_rhogw = (
                    (rhogw_vm[sl, sl, k+1, l] + rhogw_vm[slp, sl, k+1, l] + rhogw_vm[slp, slp, k+1, l]) -
                    (rhogw_vm[sl, sl, k  , l] + rhogw_vm[slp, sl, k  , l] + rhogw_vm[slp, slp, k  , l])
                ) / rdtype(3.0) * grd.GRD_rdgz[k]

                sclt[sl, sl, k, TI] = (
                    coef_intp[sl, sl, k0, l, XDIR, 0, TI] * rhogvx_vm[sl, sl, k] +
                    coef_intp[sl, sl, k0, l, XDIR, 1, TI] * rhogvx_vm[slp, sl, k] +
                    coef_intp[sl, sl, k0, l, XDIR, 2, TI] * rhogvx_vm[slp, slp, k] +

                    coef_intp[sl, sl, k0, l, YDIR, 0, TI] * rhogvy_vm[sl, sl, k] +
                    coef_intp[sl, sl, k0, l, YDIR, 1, TI] * rhogvy_vm[slp, sl, k] +
                    coef_intp[sl, sl, k0, l, YDIR, 2, TI] * rhogvy_vm[slp, slp, k] +

                    coef_intp[sl, sl, k0, l, ZDIR, 0, TI] * rhogvz_vm[sl, sl, k] +
                    coef_intp[sl, sl, k0, l, ZDIR, 1, TI] * rhogvz_vm[slp, sl, k] +
                    coef_intp[sl, sl, k0, l, ZDIR, 2, TI] * rhogvz_vm[slp, slp, k] +
                    sclt_rhogw
                )

                # TJ direction
                sclt_rhogw = (
                    (rhogw_vm[sl, sl, k+1, l] + rhogw_vm[slp, slp, k+1, l] + rhogw_vm[sl, slp, k+1, l]) -
                    (rhogw_vm[sl, sl, k  , l] + rhogw_vm[slp, slp, k  , l] + rhogw_vm[sl, slp, k  , l])
                ) / rdtype(3.0) * grd.GRD_rdgz[k]

                sclt[sl, sl, k, TJ] = (
                    coef_intp[sl, sl, k0, l, XDIR, 0, TJ] * rhogvx_vm[sl,  sl,  k] +
                    coef_intp[sl, sl, k0, l, XDIR, 1, TJ] * rhogvx_vm[slp, slp, k] +
                    coef_intp[sl, sl, k0, l, XDIR, 2, TJ] * rhogvx_vm[sl,  slp, k] +

                    coef_intp[sl, sl, k0, l, YDIR, 0, TJ] * rhogvy_vm[sl,  sl,  k] +
                    coef_intp[sl, sl, k0, l, YDIR, 1, TJ] * rhogvy_vm[slp, slp, k] +
                    coef_intp[sl, sl, k0, l, YDIR, 2, TJ] * rhogvy_vm[sl,  slp, k] +

                    coef_intp[sl, sl, k0, l, ZDIR, 0, TJ] * rhogvz_vm[sl,  sl,  k] +
                    coef_intp[sl, sl, k0, l, ZDIR, 1, TJ] * rhogvz_vm[slp, slp, k] +
                    coef_intp[sl, sl, k0, l, ZDIR, 2, TJ] * rhogvz_vm[sl,  slp, k] +
                    sclt_rhogw
                )

                if adm.ADM_have_sgp[l]:
                    sclt[0, 0, k, TI] = sclt[1, 0, k, TJ]
                #endif

                # Define slices
                sl = slice(1, gmax + 1)    # corresponds to i=1 to gmax (inclusive)
                slm1 = slice(0, gmax)      # i-1 and j-1

                # ddivdx
                ddivdx[sl, sl, k, l] = (
                    coef_diff[sl, sl, k0, l, XDIR, 0] * (sclt[sl, sl, k, TI] + sclt[sl, sl, k, TJ]) +
                    coef_diff[sl, sl, k0, l, XDIR, 1] * (sclt[sl, sl, k, TJ] + sclt[slm1, sl, k, TI]) +
                    coef_diff[sl, sl, k0, l, XDIR, 2] * (sclt[slm1, sl, k, TI] + sclt[slm1, slm1, k, TJ]) +
                    coef_diff[sl, sl, k0, l, XDIR, 3] * (sclt[slm1, slm1, k, TJ] + sclt[slm1, slm1, k, TI]) +
                    coef_diff[sl, sl, k0, l, XDIR, 4] * (sclt[slm1, slm1, k, TI] + sclt[sl, slm1, k, TJ]) +
                    coef_diff[sl, sl, k0, l, XDIR, 5] * (sclt[sl, slm1, k, TJ] + sclt[sl, sl, k, TI])
                )

                # if k == 2 and l == 2:
                #     with open (std.fname_log, 'a') as log_file:
                #         print("PP1 ", file=log_file)
                #         print(f"ddivdx[1, 16, {k}, {l}] = ", ddivdx[1, 16, k, l], file=log_file)
                #         print(f"sclt[0:2, 16, {k}, TI] = ", sclt[0:2, 16, k, TI], file=log_file)
                #         print(f"sclt[0:2, 16, {k}, TJ] = ", sclt[0:2, 16, k, TJ], file=log_file)
                #         print(f"sclt[0:2, 15, {k}, TI] = ", sclt[0:2, 15, k, TI], file=log_file)
                #         print(f"sclt[0:2, 15, {k}, TJ] = ", sclt[0:2, 15, k, TJ], file=log_file)
                #         print(f"coef_diff[1, 16, :, 0, {l}] = ", coef_diff[1, 16, :, 0, l], file=log_file)

                # ddivdy
                ddivdy[sl, sl, k, l] = (
                    coef_diff[sl, sl, k0, l, YDIR, 0] * (sclt[sl, sl, k, TI] + sclt[sl, sl, k, TJ]) +
                    coef_diff[sl, sl, k0, l, YDIR, 1] * (sclt[sl, sl, k, TJ] + sclt[slm1, sl, k, TI]) +
                    coef_diff[sl, sl, k0, l, YDIR, 2] * (sclt[slm1, sl, k, TI] + sclt[slm1, slm1, k, TJ]) +
                    coef_diff[sl, sl, k0, l, YDIR, 3] * (sclt[slm1, slm1, k, TJ] + sclt[slm1, slm1, k, TI]) +
                    coef_diff[sl, sl, k0, l, YDIR, 4] * (sclt[slm1, slm1, k, TI] + sclt[sl, slm1, k, TJ]) +
                    coef_diff[sl, sl, k0, l, YDIR, 5] * (sclt[sl, slm1, k, TJ] + sclt[sl, sl, k, TI])
                )

                # ddivdz
                ddivdz[sl, sl, k, l] = (
                    coef_diff[sl, sl, k0, l, ZDIR, 0] * (sclt[sl, sl, k, TI] + sclt[sl, sl, k, TJ]) +
                    coef_diff[sl, sl, k0, l, ZDIR, 1] * (sclt[sl, sl, k, TJ] + sclt[slm1, sl, k, TI]) +
                    coef_diff[sl, sl, k0, l, ZDIR, 2] * (sclt[slm1, sl, k, TI] + sclt[slm1, slm1, k, TJ]) +
                    coef_diff[sl, sl, k0, l, ZDIR, 3] * (sclt[slm1, slm1, k, TJ] + sclt[slm1, slm1, k, TI]) +
                    coef_diff[sl, sl, k0, l, ZDIR, 4] * (sclt[slm1, slm1, k, TI] + sclt[sl, slm1, k, TJ]) +
                    coef_diff[sl, sl, k0, l, ZDIR, 5] * (sclt[sl, slm1, k, TJ] + sclt[sl, sl, k, TI])
                )
            #end loop k

            # with open (std.fname_log, 'a') as log_file:
            #     print("U2, l= ", l, file=log_file)
            #     print("U2 rhogvx_vm[16, 15, 38] = ", rhogvx_vm[16, 15, 38], file=log_file)
            #     print("U2 rhogvy_vm[16, 15, 38] = ", rhogvy_vm[16, 15, 38], file=log_file)
            #     print("U2 rhogvz_vm[16, 15, 38] = ", rhogvz_vm[16, 15, 38], file=log_file)
            #     print("U2 sclt[16, 15, 38, TI] = ", sclt[16, 15, 38, TI], file=log_file)
            #     print("U2 sclt[16, 15, 38, TJ] = ", sclt[16, 15, 38, TJ], file=log_file)


            ddivdx[:, :, kmin-1, l] = rdtype(0.0)
            ddivdy[:, :, kmin-1, l] = rdtype(0.0)
            ddivdz[:, :, kmin-1, l] = rdtype(0.0)
            ddivdx[:, :, kmax+1, l] = rdtype(0.0)
            ddivdy[:, :, kmax+1, l] = rdtype(0.0)
            ddivdz[:, :, kmax+1, l] = rdtype(0.0)

        #end loop l

        # with open (std.fname_log, 'a') as log_file:
        #     print("R2P ddivdx[ 1, 16, 2, 2] = ", ddivdx[1, 16, 2, 2], file=log_file)
        #     print("    ddivdx[16, 15, 2, 2] = ", ddivdx[16, 15, 2, 2], file=log_file) 
        # #     print("ddivdy[16, 15, 38, 4] = ", ddivdy[16, 15, 38, 4], file=log_file)
        # #     print("ddivdz[16, 15, 38, 4] = ", ddivdz[16, 15, 38, 4], file=log_file)

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(lall_pl):
                for k in range(kmin + 1, kmax + 1):
                    for g in range(gall_pl):
                        rhogw_vm_pl[g, k, l] = (
                            vmtr.VMTR_C2WfactGz_pl[g, k, l, 0] * rhogvx_pl[g, k, l] +
                            vmtr.VMTR_C2WfactGz_pl[g, k, l, 1] * rhogvx_pl[g, k - 1, l] +
                            vmtr.VMTR_C2WfactGz_pl[g, k, l, 2] * rhogvy_pl[g, k, l] +
                            vmtr.VMTR_C2WfactGz_pl[g, k, l, 3] * rhogvy_pl[g, k - 1, l] +
                            vmtr.VMTR_C2WfactGz_pl[g, k, l, 4] * rhogvz_pl[g, k, l] +
                            vmtr.VMTR_C2WfactGz_pl[g, k, l, 5] * rhogvz_pl[g, k - 1, l]
                        ) * vmtr.VMTR_RGAMH_pl[g, k, l] + rhogw_pl[g, k, l] * vmtr.VMTR_RGSQRTH_pl[g, k, l]
                    #end loop g
                #end loop k

                rhogw_vm_pl[:, kmin, l] = rdtype(0.0)
                rhogw_vm_pl[:, kmax + 1, l] = rdtype(0.0)
            #end loop l

            for l in range(lall_pl):
                for k in range(kmin, kmax + 1):

                    # Horizontal velocity times RGAM
                    for v in range(gall_pl):
                        rhogvx_vm_pl[v] = rhogvx_pl[v, k, l] * vmtr.VMTR_RGAM_pl[v, k, l]
                        rhogvy_vm_pl[v] = rhogvy_pl[v, k, l] * vmtr.VMTR_RGAM_pl[v, k, l]
                        rhogvz_vm_pl[v] = rhogvz_pl[v, k, l] * vmtr.VMTR_RGAM_pl[v, k, l]

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij = v
                        ijp1 = adm.ADM_gmin_pl if v + 1 > adm.ADM_gmax_pl else v + 1

                        sclt_rhogw_pl = (
                            (rhogw_vm_pl[n, k+1, l] + rhogw_vm_pl[ij, k+1, l] + rhogw_vm_pl[ijp1, k+1, l]) -
                            (rhogw_vm_pl[n, k  , l] + rhogw_vm_pl[ij, k  , l] + rhogw_vm_pl[ijp1, k  , l])
                        ) / rdtype(3.0) * grd.GRD_rdgz[k]

                        #sclt_rhogw_pl = float(sclt_rhogw_pl) #rdtype(sclt_rhogw_pl)
                        #with open (std.fname_log, 'a') as log_file:
                        #    log_file.write(f"sclt_rhogw_pl shape: {sclt_rhogw_pl.shape}\n")
                        #    log_file.write(f"stopping in oprt3D") #, {rdtype}\n")   
                        #     log_file.write(f"eth_pl shape: {eth_pl.shape}\n")
                        #     log_file.write(f"kimn, kmax: {kmin}, {kmax}\n")
                        #prc.prc_mpistop(std.io_l, std.fname_log)

                        # with open (std.fname_log, 'a') as log_file:
                        #     log_file.write(f"rhogvx_vm_pl shape: {rhogvx_vm_pl.shape}\n")
                        #     log_file.write(f"coef_intp_pl shape: {coef_intp_pl.shape}\n")
                        #     log_file.write(f"stopping in oprt3D")
                        # prc.prc_mpistop(std.io_l, std.fname_log)

                        sclt_pl[ij] = (
                            coef_intp_pl[v, k0, l, XDIR, 0] * rhogvx_vm_pl[n] +
                            coef_intp_pl[v, k0, l, XDIR, 1] * rhogvx_vm_pl[ij] +
                            coef_intp_pl[v, k0, l, XDIR, 2] * rhogvx_vm_pl[ijp1] +
                            coef_intp_pl[v, k0, l, YDIR, 0] * rhogvy_vm_pl[n] +
                            coef_intp_pl[v, k0, l, YDIR, 1] * rhogvy_vm_pl[ij] +
                            coef_intp_pl[v, k0, l, YDIR, 2] * rhogvy_vm_pl[ijp1] +
                            coef_intp_pl[v, k0, l, ZDIR, 0] * rhogvz_vm_pl[n] +
                            coef_intp_pl[v, k0, l, ZDIR, 1] * rhogvz_vm_pl[ij] +
                            coef_intp_pl[v, k0, l, ZDIR, 2] * rhogvz_vm_pl[ijp1] +  
                            sclt_rhogw_pl
                        )

                    ddivdx_pl[n, k, l] = rdtype(0.0)
                    ddivdy_pl[n, k, l] = rdtype(0.0)
                    ddivdz_pl[n, k, l] = rdtype(0.0)

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij = v
                        ijm1 = adm.ADM_gmax_pl if v - 1 < adm.ADM_gmin_pl else v - 1

                        ddivdx_pl[n, k, l] += coef_diff_pl[v, k0, l, XDIR] * (sclt_pl[ijm1] + sclt_pl[ij])
                        ddivdy_pl[n, k, l] += coef_diff_pl[v, k0, l, YDIR] * (sclt_pl[ijm1] + sclt_pl[ij])
                        ddivdz_pl[n, k, l] += coef_diff_pl[v, k0, l, ZDIR] * (sclt_pl[ijm1] + sclt_pl[ij])
                    #end loop v
                #end loop k
            #end loop l
        else:
            ddivdx_pl[:, :, :] = rdtype(0.0)
            ddivdy_pl[:, :, :] = rdtype(0.0)
            ddivdz_pl[:, :, :] = rdtype(0.0)
        #endif

        prf.PROF_rapend('OPRT3D_divdamp', 2)

        return

    #> 3D divergence damping operator
    def OPRT3D_divdamp(self,
        ddivdx,    ddivdx_pl,    
        ddivdy,    ddivdy_pl,    
        ddivdz,    ddivdz_pl,    
        rhogvx,    rhogvx_pl,    
        rhogvy,    rhogvy_pl,    
        rhogvz,    rhogvz_pl,    
        rhogw,     rhogw_pl,     
        coef_intp, coef_intp_pl, 
        coef_diff, coef_diff_pl,
        grd, vmtr, rdtype,        
    ):          

        prf.PROF_rapstart('OPRT3D_divdamp', 2)

        gall_1d = adm.ADM_gall_1d
        gall_pl = adm.ADM_gall_pl
        kall    = adm.ADM_kall
        lall    = adm.ADM_lall
        lall_pl = adm.ADM_lall_pl
        k0   = adm.ADM_K0

        TI    = adm.ADM_TI
        TJ    = adm.ADM_TJ

        sclt      = np.empty((adm.ADM_shape + (2,)), dtype=rdtype)  # TI and TJ
        sclt_pl   = np.empty((gall_pl,), dtype=rdtype)

        rhogw_vm   = np.empty((adm.ADM_shape), dtype=rdtype)    
        rhogvx_vm  = np.empty((adm.ADM_shape), dtype=rdtype)    
        rhogvy_vm  = np.empty((adm.ADM_shape), dtype=rdtype)    
        rhogvz_vm  = np.empty((adm.ADM_shape), dtype=rdtype)        
        rhogw_vm_pl  = np.empty((adm.ADM_shape_pl), dtype=rdtype)    
        rhogvx_vm_pl = np.empty((gall_pl,), dtype=rdtype)    
        rhogvy_vm_pl = np.empty((gall_pl,), dtype=rdtype)    
        rhogvz_vm_pl = np.empty((gall_pl,), dtype=rdtype)    

        XDIR = grd.GRD_XDIR
        YDIR = grd.GRD_YDIR
        ZDIR = grd.GRD_ZDIR

        gmin = adm.ADM_gmin # 1
        gmax = adm.ADM_gmax # 16
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax

        k_slice = slice(kmin + 1, kmax + 1)
        k_m1 = slice(kmin, kmax)

        rhogw_vm[:, :, k_slice, :] = (
            vmtr.VMTR_C2WfactGz[:, :, k_slice, :, 0] * rhogvx[:, :, k_slice, :] +
            vmtr.VMTR_C2WfactGz[:, :, k_slice, :, 1] * rhogvx[:, :, k_m1, :] +
            vmtr.VMTR_C2WfactGz[:, :, k_slice, :, 2] * rhogvy[:, :, k_slice, :] +
            vmtr.VMTR_C2WfactGz[:, :, k_slice, :, 3] * rhogvy[:, :, k_m1, :] +
            vmtr.VMTR_C2WfactGz[:, :, k_slice, :, 4] * rhogvz[:, :, k_slice, :] +
            vmtr.VMTR_C2WfactGz[:, :, k_slice, :, 5] * rhogvz[:, :, k_m1, :]
        ) * vmtr.VMTR_RGAMH[:, :, k_slice, :] + rhogw[:, :, k_slice, :] * vmtr.VMTR_RGSQRTH[:, :, k_slice, :]

        # Set boundary values
        rhogw_vm[:, :, kmin, :] = rdtype(0.0)
        rhogw_vm[:, :, kmax + 1, :] = rdtype(0.0)


        # with open (std.fname_log, 'a') as log_file:
        #     print("U1 rhogw_vm[16, 15, 38, 4] = ", rhogw_vm[16, 15, 38, 4], file=log_file)


        # Slices
        sl   = slice(0, gmax + 1)   # corresponds to Fortran 2:gmax
        slp  = slice(1, gmax + 2)   # sl + 1
        ksl  = slice(kmin, kmax + 1)

        # Compute rhogv*_vm
        rhogvx_vm[:, :, ksl, :] = rhogvx[:, :, ksl, :] * vmtr.VMTR_RGAM[:, :, ksl, :]
        rhogvy_vm[:, :, ksl, :] = rhogvy[:, :, ksl, :] * vmtr.VMTR_RGAM[:, :, ksl, :]
        rhogvz_vm[:, :, ksl, :] = rhogvz[:, :, ksl, :] * vmtr.VMTR_RGAM[:, :, ksl, :]

        # Compute sclt_rhogw for TI
        sclt_rhogw_TI = (
            (rhogw_vm[sl, sl, kmin+1:kmax+2, :] + rhogw_vm[slp, sl, kmin+1:kmax+2, :] + rhogw_vm[slp, slp, kmin+1:kmax+2, :]) -
            (rhogw_vm[sl, sl, kmin:kmax+1, :]   + rhogw_vm[slp, sl, kmin:kmax+1, :]   + rhogw_vm[slp, slp, kmin:kmax+1, :])
        ) / rdtype(3.0) * grd.GRD_rdgz[kmin:kmax+1][np.newaxis, np.newaxis, :, np.newaxis]  # (i,j,k,l)

        # Compute sclt[..., TI]
        sclt[sl, sl, ksl, :, TI] = (
            coef_intp[sl, sl, :, :, XDIR, 0, TI] * rhogvx_vm[sl, sl, ksl, :] +
            coef_intp[sl, sl, :, :, XDIR, 1, TI] * rhogvx_vm[slp, sl, ksl, :] +
            coef_intp[sl, sl, :, :, XDIR, 2, TI] * rhogvx_vm[slp, slp, ksl, :] +

            coef_intp[sl, sl, :, :, YDIR, 0, TI] * rhogvy_vm[sl, sl, ksl, :] +
            coef_intp[sl, sl, :, :, YDIR, 1, TI] * rhogvy_vm[slp, sl, ksl, :] +
            coef_intp[sl, sl, :, :, YDIR, 2, TI] * rhogvy_vm[slp, slp, ksl, :] +

            coef_intp[sl, sl, :, :, ZDIR, 0, TI] * rhogvz_vm[sl, sl, ksl, :] +
            coef_intp[sl, sl, :, :, ZDIR, 1, TI] * rhogvz_vm[slp, sl, ksl, :] +
            coef_intp[sl, sl, :, :, ZDIR, 2, TI] * rhogvz_vm[slp, slp, ksl, :] +
            sclt_rhogw_TI
        )

        # Compute sclt_rhogw for TJ
        sclt_rhogw_TJ = (
            (rhogw_vm[sl, sl, kmin+1:kmax+2, :] + rhogw_vm[slp, slp, kmin+1:kmax+2, :] + rhogw_vm[sl, slp, kmin+1:kmax+2, :]) -
            (rhogw_vm[sl, sl, kmin:kmax+1, :]   + rhogw_vm[slp, slp, kmin:kmax+1, :]   + rhogw_vm[sl, slp, kmin:kmax+1, :])
        ) / rdtype(3.0) * grd.GRD_rdgz[kmin:kmax+1][np.newaxis, np.newaxis, :, np.newaxis]

        # Compute sclt[..., TJ]
        sclt[sl, sl, ksl, :, TJ] = (
            coef_intp[sl, sl, :, :, XDIR, 0, TJ] * rhogvx_vm[sl, sl, ksl, :] +
            coef_intp[sl, sl, :, :, XDIR, 1, TJ] * rhogvx_vm[slp, slp, ksl, :] +
            coef_intp[sl, sl, :, :, XDIR, 2, TJ] * rhogvx_vm[sl, slp, ksl, :] +

            coef_intp[sl, sl, :, :, YDIR, 0, TJ] * rhogvy_vm[sl, sl, ksl, :] +
            coef_intp[sl, sl, :, :, YDIR, 1, TJ] * rhogvy_vm[slp, slp, ksl, :] +
            coef_intp[sl, sl, :, :, YDIR, 2, TJ] * rhogvy_vm[sl, slp, ksl, :] +

            coef_intp[sl, sl, :, :, ZDIR, 0, TJ] * rhogvz_vm[sl, sl, ksl, :] +
            coef_intp[sl, sl, :, :, ZDIR, 1, TJ] * rhogvz_vm[slp, slp, ksl, :] +
            coef_intp[sl, sl, :, :, ZDIR, 2, TJ] * rhogvz_vm[sl, slp, ksl, :] +
            sclt_rhogw_TJ
        )

        sclt[0, 0, :, :, TI] = (  sclt[0, 0, :, :, TI] * ppm.pntmask[:, :, 0]
                                + sclt[1, 0, :, :, TJ] * ppm.pntmask[:, :, 1] 
                                )
        #for l in range(lall):
        #        if adm.ADM_have_sgp[l]:
        #            sclt[0, 0, :, l, TI] = sclt[1, 0, :, l, TJ]
                #endif


        sl = slice(1, gmax + 1)
        slm1 = slice(0, gmax)

        # Precompute relevant sclt sums
        sclt_TI = sclt[..., TI]
        sclt_TJ = sclt[..., TJ]

        sclt_0 = sclt_TI[sl, sl, kmin:kmax+1, :] + sclt_TJ[sl, sl, kmin:kmax+1, :]
        sclt_1 = sclt_TJ[sl, sl, kmin:kmax+1, :] + sclt_TI[slm1, sl, kmin:kmax+1, :]
        sclt_2 = sclt_TI[slm1, sl, kmin:kmax+1, :] + sclt_TJ[slm1, slm1, kmin:kmax+1, :]
        sclt_3 = sclt_TJ[slm1, slm1, kmin:kmax+1, :] + sclt_TI[slm1, slm1, kmin:kmax+1, :]
        sclt_4 = sclt_TI[slm1, slm1, kmin:kmax+1, :] + sclt_TJ[sl, slm1, kmin:kmax+1, :]
        sclt_5 = sclt_TJ[sl, slm1, kmin:kmax+1, :] + sclt_TI[sl, sl, kmin:kmax+1, :]

        for d, tgt in zip([XDIR, YDIR, ZDIR], [ddivdx, ddivdy, ddivdz]):
            coef = coef_diff[sl, sl, :, :, d, :]  # (i, j, k0, l, 6)
            tgt[sl, sl, kmin:kmax+1, :] = (
                coef[:, :, :, :, 0] * sclt_0 +
                coef[:, :, :, :, 1] * sclt_1 +
                coef[:, :, :, :, 2] * sclt_2 +
                coef[:, :, :, :, 3] * sclt_3 +
                coef[:, :, :, :, 4] * sclt_4 +
                coef[:, :, :, :, 5] * sclt_5
            )

        # Zero out boundary slices
        ddivdx[:, :, kmin-1, :] = rdtype(0.0)
        ddivdy[:, :, kmin-1, :] = rdtype(0.0)
        ddivdz[:, :, kmin-1, :] = rdtype(0.0)
        ddivdx[:, :, kmax+1, :] = rdtype(0.0)
        ddivdy[:, :, kmax+1, :] = rdtype(0.0)
        ddivdz[:, :, kmax+1, :] = rdtype(0.0)

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(lall_pl):
                for k in range(kmin + 1, kmax + 1):
                    for g in range(gall_pl):
                        rhogw_vm_pl[g, k, l] = (
                            vmtr.VMTR_C2WfactGz_pl[g, k, l, 0] * rhogvx_pl[g, k, l] +
                            vmtr.VMTR_C2WfactGz_pl[g, k, l, 1] * rhogvx_pl[g, k - 1, l] +
                            vmtr.VMTR_C2WfactGz_pl[g, k, l, 2] * rhogvy_pl[g, k, l] +
                            vmtr.VMTR_C2WfactGz_pl[g, k, l, 3] * rhogvy_pl[g, k - 1, l] +
                            vmtr.VMTR_C2WfactGz_pl[g, k, l, 4] * rhogvz_pl[g, k, l] +
                            vmtr.VMTR_C2WfactGz_pl[g, k, l, 5] * rhogvz_pl[g, k - 1, l]
                        ) * vmtr.VMTR_RGAMH_pl[g, k, l] + rhogw_pl[g, k, l] * vmtr.VMTR_RGSQRTH_pl[g, k, l]
                    #end loop g
                #end loop k

                rhogw_vm_pl[:, kmin, l] = rdtype(0.0)
                rhogw_vm_pl[:, kmax + 1, l] = rdtype(0.0)
            #end loop l

            for l in range(lall_pl):
                for k in range(kmin, kmax + 1):

                    # Horizontal velocity times RGAM
                    for v in range(gall_pl):
                        rhogvx_vm_pl[v] = rhogvx_pl[v, k, l] * vmtr.VMTR_RGAM_pl[v, k, l]
                        rhogvy_vm_pl[v] = rhogvy_pl[v, k, l] * vmtr.VMTR_RGAM_pl[v, k, l]
                        rhogvz_vm_pl[v] = rhogvz_pl[v, k, l] * vmtr.VMTR_RGAM_pl[v, k, l]

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij = v
                        ijp1 = adm.ADM_gmin_pl if v + 1 > adm.ADM_gmax_pl else v + 1

                        sclt_rhogw_pl = (
                            (rhogw_vm_pl[n, k+1, l] + rhogw_vm_pl[ij, k+1, l] + rhogw_vm_pl[ijp1, k+1, l]) -
                            (rhogw_vm_pl[n, k  , l] + rhogw_vm_pl[ij, k  , l] + rhogw_vm_pl[ijp1, k  , l])
                        ) / rdtype(3.0) * grd.GRD_rdgz[k]

                        sclt_pl[ij] = (
                            coef_intp_pl[v, k0, l, XDIR, 0] * rhogvx_vm_pl[n] + 
                            coef_intp_pl[v, k0, l, XDIR, 1] * rhogvx_vm_pl[ij] +
                            coef_intp_pl[v, k0, l, XDIR, 2] * rhogvx_vm_pl[ijp1] +
                            coef_intp_pl[v, k0, l, YDIR, 0] * rhogvy_vm_pl[n] +
                            coef_intp_pl[v, k0, l, YDIR, 1] * rhogvy_vm_pl[ij] +
                            coef_intp_pl[v, k0, l, YDIR, 2] * rhogvy_vm_pl[ijp1] +
                            coef_intp_pl[v, k0, l, ZDIR, 0] * rhogvz_vm_pl[n] +
                            coef_intp_pl[v, k0, l, ZDIR, 1] * rhogvz_vm_pl[ij] +
                            coef_intp_pl[v, k0, l, ZDIR, 2] * rhogvz_vm_pl[ijp1] +
                            sclt_rhogw_pl
                        )

                    ddivdx_pl[n, k, l] = rdtype(0.0)
                    ddivdy_pl[n, k, l] = rdtype(0.0)
                    ddivdz_pl[n, k, l] = rdtype(0.0)

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij = v
                        ijm1 = adm.ADM_gmax_pl if v - 1 < adm.ADM_gmin_pl else v - 1

                        ddivdx_pl[n, k, l] += coef_diff_pl[v, k0, l, XDIR] * (sclt_pl[ijm1] + sclt_pl[ij])
                        ddivdy_pl[n, k, l] += coef_diff_pl[v, k0, l, YDIR] * (sclt_pl[ijm1] + sclt_pl[ij])
                        ddivdz_pl[n, k, l] += coef_diff_pl[v, k0, l, ZDIR] * (sclt_pl[ijm1] + sclt_pl[ij])
                    #end loop v
                #end loop k
            #end loop l
        else:
            ddivdx_pl[:, :, :] = rdtype(0.0)
            ddivdy_pl[:, :, :] = rdtype(0.0)
            ddivdz_pl[:, :, :] = rdtype(0.0)
        #endif

        prf.PROF_rapend('OPRT3D_divdamp', 2)

        return

