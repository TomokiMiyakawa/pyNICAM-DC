"""Horizontal differential operators on the icosahedral grid (port of
NICAM mod_oprt).

Two halves:

  coefficient setup   OPRT_setup -> OPRT_{divergence,rotation,gradient,
  (host, once)        laplacian,diffusion}_setup precompute the stencil
                      coefficients (OPRT_coef_*) from the GMTR geometry,
                      vectorized over the interior but arithmetic-identical
                      to the Fortran loops. coef_div/coef_intp/coef_diff are
                      also consumed directly by mod_src / mod_vi /
                      mod_numfilter; coef_rot currently has no kernel consumer
                      (kept Fortran-faithful).

  operator kernels    OPRT_gradient / OPRT_horizontalize_vec / OPRT_laplacian /
                      OPRT_diffusion / OPRT_divdamp / OPRT3D_divdamp. Each is a
                      thin dispatcher into its _*_fused body: backend-agnostic
                      (numpy or jax via bk.xp/bk.maybe_jit), stencil math in
                      nhm/dynamics/kernels/*.py, constant geometry cached
                      device-resident via bk.device_consts. resident=True
                      variants return device arrays without a host drain
                      (_oprt3d_divdamp_device likewise, for numfilter_divdamp).

The original Fortran-style loop kernels (*_ij), the unfused numpy bodies and
the experimental jax laplacians were deleted 2026-07-25 after the FUSE_OPRT*
gates were collapsed; recover from git history if ever needed.
"""
import os
import toml
import numpy as np
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_prof import prf
from pynicamdc.share.mod_ppmask import ppm
from pynicamdc.share.mod_backend import backend as bk
from pynicamdc.nhm.dynamics.kernels.oprtdivdamp import (
    OprtDivdampCfg, compute_oprt_divdamp,
)
from pynicamdc.nhm.dynamics.kernels.oprt3ddivdamp import (
    Oprt3DDivdampCfg, compute_oprt3d_divdamp,
)
from pynicamdc.nhm.dynamics.kernels.horizontalizevec import (
    HorizontalizeVecCfg, compute_horizontalize_vec,
)
from pynicamdc.nhm.dynamics.kernels.oprtlaplacian import (
    OprtLaplacianCfg, compute_oprt_laplacian,
)
from pynicamdc.nhm.dynamics.kernels.oprtdiffusion import (
    OprtDiffusionCfg, compute_oprt_diffusion,
)
from pynicamdc.nhm.dynamics.kernels.oprtgradient import (
    OprtGradientCfg, compute_oprt_gradient,
)

class Oprt:


    # ------------------------------------------------------------------
    # Coefficient setup (host numpy, once at init; Fortran-faithful)
    # ------------------------------------------------------------------

    def __init__(self):
        pass

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
        self.OPRT_coef_div[:,:,:,:,:,:] = rdtype(0.0)    #  i , j, KNONE, l, xyz, 7
        self.OPRT_coef_div_pl[:,:,:,:]  = rdtype(0.0)    #  ij,    KNONE, l, xyz

        for l in range(lall):
            for d in range(nxyz):
                #hn = d + HNX - 1
                #         0
                hn = d + HNX
                                # 1  to  16 (inner grid points)
                ii = slice(gmin, gmax + 1)
                ip = slice(gmin + 1, gmax + 2)
                im = slice(gmin - 1, gmax)
                jj = slice(gmin, gmax + 1)
                jp = slice(gmin + 1, gmax + 2)
                jm = slice(gmin - 1, gmax)

                # Vectorized interior (i,j) block: i/i+1/i-1 -> ii/ip/im, j/j+1/j-1
                # -> jj/jp/jm. Per-element arithmetic preserved -> bit-identical.
            #for g in range(gmin, gmax + 1):
            # ij     = g
            # ip1j   = g + iall + 1
            # ip1jp1 = g + iall + 1
            # ijp1   = g + iall
            # i-1, j   = g - 1
            # i-1, jm1 = g - iall - 1
            # ijm1   = g - iall

            # ij
                self.OPRT_coef_div[ii, jj, k0, l, d, 0] = (
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W1] * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W1] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W1] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W1] * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                    + gmtr.GMTR_t[im, jj, k0, l, TI, W2] * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                    - gmtr.GMTR_t[im, jj, k0, l, TI, W2] * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W2] * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W2] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W3] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W3] * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                    - gmtr.GMTR_t[ii, jm, k0, l, TJ, W3] * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                    + gmtr.GMTR_t[ii, jm, k0, l, TJ, W3] * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # ip1j
                self.OPRT_coef_div[ii, jj, k0, l, d, 1] = (
                    - gmtr.GMTR_t[ii, jm, k0, l, TJ, W2] * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                    + gmtr.GMTR_t[ii, jm, k0, l, TJ, W2] * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W2] * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W2] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # ip1jp1
                self.OPRT_coef_div[ii, jj, k0, l, d, 2] = (
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W3] * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W3] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W2] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W2] * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # ijp1
                self.OPRT_coef_div[ii, jj, k0, l, d, 3] = (
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W3] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W3] * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                    + gmtr.GMTR_t[im, jj, k0, l, TI, W3] * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                    - gmtr.GMTR_t[im, jj, k0, l, TI, W3] * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # im1j
                self.OPRT_coef_div[ii, jj, k0, l, d, 4] = (
                    + gmtr.GMTR_t[im, jj, k0, l, TI, W1] * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                    - gmtr.GMTR_t[im, jj, k0, l, TI, W1] * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W3] * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W3] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # i-1,  j-1
                self.OPRT_coef_div[ii, jj, k0, l, d, 5] = (
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W1] * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W1] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W1] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W1] * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # ijm1
                self.OPRT_coef_div[ii, jj, k0, l, d, 6] = (
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W2] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W2] * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                    - gmtr.GMTR_t[ii, jm, k0, l, TJ, W1] * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                    + gmtr.GMTR_t[ii, jm, k0, l, TJ, W1] * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

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
                ii = slice(gmin, gmax + 1)
                ip = slice(gmin + 1, gmax + 2)
                im = slice(gmin - 1, gmax)
                jj = slice(gmin, gmax + 1)
                jp = slice(gmin + 1, gmax + 2)
                jm = slice(gmin - 1, gmax)

                # Vectorized interior (i,j) block: i/i+1/i-1 -> ii/ip/im, j/j+1/j-1
                # -> jj/jp/jm. Per-element arithmetic preserved -> bit-identical.

                # ij
                self.OPRT_coef_rot[ii, jj, k0, l, d, 0] = (
                    + gmtr.GMTR_t[ii, jj,   k0, l, TI, W1] * gmtr.GMTR_a[ii, jj,   k0, l, AI , ht]
                    + gmtr.GMTR_t[ii, jj,   k0, l, TI, W1] * gmtr.GMTR_a[ii, jj,   k0, l, AIJ, ht]
                    + gmtr.GMTR_t[ii, jj,   k0, l, TJ, W1] * gmtr.GMTR_a[ii, jj,   k0, l, AIJ, ht]
                    + gmtr.GMTR_t[ii, jj,   k0, l, TJ, W1] * gmtr.GMTR_a[ii, jj,   k0, l, AJ , ht]
                    + gmtr.GMTR_t[im, jj,   k0, l, TI, W2] * gmtr.GMTR_a[ii, jj,   k0, l, AJ , ht]
                    - gmtr.GMTR_t[im, jj,   k0, l, TI, W2] * gmtr.GMTR_a[im, jj,   k0, l, AI , ht]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W2] * gmtr.GMTR_a[im, jj,   k0, l, AI , ht]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W2] * gmtr.GMTR_a[im, jm, k0, l, AIJ, ht]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W3] * gmtr.GMTR_a[im, jm, k0, l, AIJ, ht]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W3] * gmtr.GMTR_a[ii, jm, k0, l, AJ , ht]
                    - gmtr.GMTR_t[ii, jm, k0, l, TJ, W3] * gmtr.GMTR_a[ii, jm, k0, l, AJ , ht]
                    + gmtr.GMTR_t[ii, jm, k0, l, TJ, W3] * gmtr.GMTR_a[ii, jj,   k0, l, AI , ht]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # ip1j
                self.OPRT_coef_rot[ii, jj, k0, l, d, 1] = (
                    - gmtr.GMTR_t[ii, jm, k0, l, TJ, W2] * gmtr.GMTR_a[ii, jm, k0, l, AJ , ht]
                    + gmtr.GMTR_t[ii, jm, k0, l, TJ, W2] * gmtr.GMTR_a[ii, jj,   k0, l, AI , ht]
                    + gmtr.GMTR_t[ii, jj,   k0, l, TI, W2] * gmtr.GMTR_a[ii, jj,   k0, l, AI , ht]
                    + gmtr.GMTR_t[ii, jj,   k0, l, TI, W2] * gmtr.GMTR_a[ii, jj,   k0, l, AIJ, ht]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # ip1jp1
                self.OPRT_coef_rot[ii, jj, k0, l, d, 2] = (
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W3] * gmtr.GMTR_a[ii, jj, k0, l, AI , ht]
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W3] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, ht]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W2] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, ht]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W2] * gmtr.GMTR_a[ii, jj, k0, l, AJ , ht]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # ijp1
                self.OPRT_coef_rot[ii, jj, k0, l, d, 3] = (
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W3] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, ht]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W3] * gmtr.GMTR_a[ii, jj, k0, l, AJ , ht]
                    + gmtr.GMTR_t[im, jj, k0, l, TI, W3] * gmtr.GMTR_a[ii, jj, k0, l, AJ , ht]
                    - gmtr.GMTR_t[im, jj, k0, l, TI, W3] * gmtr.GMTR_a[im, jj, k0, l, AI , ht]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # im1j
                self.OPRT_coef_rot[ii, jj, k0, l, d, 4] = (
                    + gmtr.GMTR_t[im, jj,   k0, l, TI, W1] * gmtr.GMTR_a[ii, jj,   k0, l, AJ , ht]
                    - gmtr.GMTR_t[im, jj,   k0, l, TI, W1] * gmtr.GMTR_a[im, jj,   k0, l, AI , ht]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W3] * gmtr.GMTR_a[im, jj,   k0, l, AI , ht]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W3] * gmtr.GMTR_a[im, jm, k0, l, AIJ, ht]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # im1jm1
                self.OPRT_coef_rot[ii, jj, k0, l, d, 5] = (
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W1] * gmtr.GMTR_a[im, jj,   k0, l, AI , ht]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W1] * gmtr.GMTR_a[im, jm, k0, l, AIJ, ht]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W1] * gmtr.GMTR_a[im, jm, k0, l, AIJ, ht]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W1] * gmtr.GMTR_a[ii, jm, k0, l, AJ , ht]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # ijm1
                self.OPRT_coef_rot[ii, jj, k0, l, d, 6] = (
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W2] * gmtr.GMTR_a[im, jm, k0, l, AIJ, ht]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W2] * gmtr.GMTR_a[ii, jm, k0, l, AJ , ht]
                    - gmtr.GMTR_t[ii, jm, k0, l, TJ, W1] * gmtr.GMTR_a[ii, jm, k0, l, AJ , ht]
                    + gmtr.GMTR_t[ii, jm, k0, l, TJ, W1] * gmtr.GMTR_a[ii, jj,   k0, l, AI , ht]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                if adm.ADM_have_sgp[l]: # pentagon
                    # ij     = gmin
                    i = 1
                    j = 1
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
        self.OPRT_coef_grad[:,:,:,:,:,:] = rdtype(0.0)   #  i , j, KNONE, l, xyz, 7
        self.OPRT_coef_grad_pl[:,:,:,:]  = rdtype(0.0)   #  ij,    KNONE, l, xyz

        for l in range(lall):
            for d in range(nxyz):
                #hn = d + HNX - 1
                #         0
                hn = d + HNX
                                # 1  to  16 (inner grid points)
                ii = slice(gmin, gmax + 1)
                ip = slice(gmin + 1, gmax + 2)
                im = slice(gmin - 1, gmax)
                jj = slice(gmin, gmax + 1)
                jp = slice(gmin + 1, gmax + 2)
                jm = slice(gmin - 1, gmax)

                # Vectorized interior (i,j) block: i/i+1/i-1 -> ii/ip/im, j/j+1/j-1
                # -> jj/jp/jm. Per-element arithmetic preserved -> bit-identical.

                # ij
                self.OPRT_coef_grad[ii, jj, k0, l, d, 0] = (
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W1] * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W1] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W1] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W1] * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                    + gmtr.GMTR_t[im, jj, k0, l, TI, W2] * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                    - gmtr.GMTR_t[im, jj, k0, l, TI, W2] * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W2] * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W3] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W2] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W3] * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                    - gmtr.GMTR_t[ii, jm, k0, l, TJ, W3] * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                    + gmtr.GMTR_t[ii, jm, k0, l, TJ, W3] * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                    - rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    - rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                    + rdtype(2.0) * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                    + rdtype(2.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    + rdtype(2.0) * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                    - rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # ip1j
                self.OPRT_coef_grad[ii, jj, k0, l, d, 1] = (
                    - gmtr.GMTR_t[ii, jm, k0, l, TJ, W2] * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                    + gmtr.GMTR_t[ii, jm, k0, l, TJ, W2] * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W2] * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W2] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # ip1jp1
                self.OPRT_coef_grad[ii, jj, k0, l, d, 2] = (
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W3] * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TI, W3] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W2] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W2] * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # ijp1
                self.OPRT_coef_grad[ii, jj, k0, l, d, 3] = (
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W3] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + gmtr.GMTR_t[ii, jj, k0, l, TJ, W3] * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                    + gmtr.GMTR_t[im, jj, k0, l, TI, W3] * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                    - gmtr.GMTR_t[im, jj, k0, l, TI, W3] * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # im1j
                self.OPRT_coef_grad[ii, jj, k0, l, d, 4] = (
                    + gmtr.GMTR_t[im, jj, k0, l, TI, W1] * gmtr.GMTR_a[ii, jj, k0, l, AJ , hn]
                    - gmtr.GMTR_t[im, jj, k0, l, TI, W1] * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W3] * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W3] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # im1jm1
                self.OPRT_coef_grad[ii, jj, k0, l, d, 5] = (
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W1] * gmtr.GMTR_a[im, jj, k0, l, AI , hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TJ, W1] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W1] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W1] * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

                # ijm1
                self.OPRT_coef_grad[ii, jj, k0, l, d, 6] = (
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W2] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    - gmtr.GMTR_t[im, jm, k0, l, TI, W2] * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                    - gmtr.GMTR_t[ii, jm, k0, l, TJ, W1] * gmtr.GMTR_a[ii, jm, k0, l, AJ , hn]
                    + gmtr.GMTR_t[ii, jm, k0, l, TJ, W1] * gmtr.GMTR_a[ii, jj, k0, l, AI , hn]
                ) * rdtype(0.5) * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]

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
                ii = slice(gmin, gmax + 1)
                ip = slice(gmin + 1, gmax + 2)
                im = slice(gmin - 1, gmax)
                jj = slice(gmin, gmax + 1)
                jp = slice(gmin + 1, gmax + 2)
                jm = slice(gmin - 1, gmax)

                # Vectorized over the interior (i,j) block: each scalar index
                # i/i+1/i-1, j/j+1/j-1 becomes the matching slice ii/ip/im, jj/jp/jm.
                # Per-element arithmetic, parenthesization and the d-loop order are
                # preserved verbatim, so the result is bit-identical to the loop.

                # coef_lap[ii, jj, k0, l, 0]
                self.OPRT_coef_lap[ii, jj, k0, l, 0] += gmtr.GMTR_t[ii, jj, k0, l, TI, T_RAREA] * (
                    - rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AI,  tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    + rdtype(2.0) * gmtr.GMTR_a[ip, jj, k0, l, AJ,  tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    - rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AI,  tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + rdtype(2.0) * gmtr.GMTR_a[ip, jj, k0, l, AJ,  tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                )

                self.OPRT_coef_lap[ii, jj, k0, l, 0] += gmtr.GMTR_t[ii, jj, k0, l, TJ, T_RAREA] * (
                    - rdtype(1.0) * gmtr.GMTR_a[ii, jj,   k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    - rdtype(2.0) * gmtr.GMTR_a[ii, jp, k0, l, AI,  tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[ii, jj,   k0, l, AJ,  tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    - rdtype(1.0) * gmtr.GMTR_a[ii, jj,   k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    - rdtype(2.0) * gmtr.GMTR_a[ii, jp, k0, l, AI,  tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[ii, jj,   k0, l, AJ,  tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                )

                self.OPRT_coef_lap[ii, jj, k0, l, 0] += gmtr.GMTR_t[im, jj, k0, l, TI, T_RAREA] * (
                    - rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AJ,  tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    - rdtype(2.0) * gmtr.GMTR_a[im, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    - rdtype(1.0) * gmtr.GMTR_a[im, jj, k0, l, AI,  tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AJ,  tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    + rdtype(2.0) * gmtr.GMTR_a[im, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[im, jj, k0, l, AI,  tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                )

                self.OPRT_coef_lap[ii, jj, k0, l, 0] += gmtr.GMTR_t[im, jm, k0, l, TJ, T_RAREA] * (
                    -rdtype(1.0) * gmtr.GMTR_a[im, jj, k0, l, AI, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    + rdtype(2.0) * gmtr.GMTR_a[im, jm, k0, l, AJ, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    - rdtype(1.0) * gmtr.GMTR_a[im, jj, k0, l, AI, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    + rdtype(2.0) * gmtr.GMTR_a[im, jm, k0, l, AJ, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                )

                self.OPRT_coef_lap[ii, jj, k0, l, 0] += gmtr.GMTR_t[im, jm, k0, l, TI, T_RAREA] * (
                    -rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    - rdtype(2.0) * gmtr.GMTR_a[im, jm, k0, l, AI, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[ii, jm, k0, l, AJ, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    - rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    - rdtype(2.0) * gmtr.GMTR_a[im, jm, k0, l, AI, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[ii, jm, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                )

                self.OPRT_coef_lap[ii, jj, k0, l, 0] += gmtr.GMTR_t[ii, jm, k0, l, TJ, T_RAREA] * (
                    -rdtype(1.0) * gmtr.GMTR_a[ii, jm, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    - rdtype(2.0) * gmtr.GMTR_a[ii, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    - rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AI, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[ii, jm, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    + rdtype(2.0) * gmtr.GMTR_a[ii, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AI, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                )

                # coef_lap[ii, jj, k0, l, 1]
                self.OPRT_coef_lap[ii, jj, k0, l, 1] += gmtr.GMTR_t[ii, jm, k0, l, TJ, T_RAREA] * (
                    -rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AI, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    + rdtype(2.0) * gmtr.GMTR_a[ii, jm, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[ii, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    + rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AI, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    - rdtype(2.0) * gmtr.GMTR_a[ii, jm, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    - rdtype(1.0) * gmtr.GMTR_a[ii, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                )

                # coef_lap[ii, jj, k0, l, 1] (continued)
                self.OPRT_coef_lap[ii, jj, k0, l, 1] += gmtr.GMTR_t[ii, jj, k0, l, TI, T_RAREA] * (
                    -rdtype(1.0) * gmtr.GMTR_a[ip, jj, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    -rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AI, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[ip, jj, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    -rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AI, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                )

                # coef_lap[ii, jj, k0, l, 2]
                self.OPRT_coef_lap[ii, jj, k0, l, 2] += gmtr.GMTR_t[ii, jj, k0, l, TI, T_RAREA] * (
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    +rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AI, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[ip, jj, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    +rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AI, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[ip, jj, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                )

                self.OPRT_coef_lap[ii, jj, k0, l, 2] += gmtr.GMTR_t[ii, jj, k0, l, TJ, T_RAREA] * (
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jp, k0, l, AI, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    -rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jp, k0, l, AI, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    -rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                )

                # coef_lap[ii, jj, k0, l, 3]
                self.OPRT_coef_lap[ii, jj, k0, l, 3] += gmtr.GMTR_t[ii, jj, k0, l, TJ, T_RAREA] * (
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    +rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jp, k0, l, AI, tn] * gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    +rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jp, k0, l, AI, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                )

                self.OPRT_coef_lap[ii, jj, k0, l, 3] += gmtr.GMTR_t[im, jj, k0, l, TI, T_RAREA] * (
                    +rdtype(1.0) * gmtr.GMTR_a[im, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    +rdtype(2.0) * gmtr.GMTR_a[im, jj, k0, l, AI, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[im, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    -rdtype(2.0) * gmtr.GMTR_a[im, jj, k0, l, AI, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jj, k0, l, AJ, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                )

                # coef_lap[ii, jj, k0, l, 4]
                self.OPRT_coef_lap[ii, jj, k0, l, 4] += gmtr.GMTR_t[im, jj, k0, l, TI, T_RAREA] * (
                    -rdtype(1.0) * gmtr.GMTR_a[im, jj, k0, l, AI, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    +rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[im, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[im, jj, k0, l, AI, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    -rdtype(2.0) * gmtr.GMTR_a[ii, jj, k0, l, AJ, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[im, jj, k0, l, AIJ, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                )

                # coef_lap[ii, jj, k0, l, 4] (continued)
                self.OPRT_coef_lap[ii, jj, k0, l, 4] += gmtr.GMTR_t[im, jm, k0, l, TJ, T_RAREA] * (
                    -rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AJ, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    -rdtype(2.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[im, jj,   k0, l, AI, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AJ, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    -rdtype(2.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[im, jj,   k0, l, AI, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                )

                # coef_lap[ii, jj, k0, l, 5]
                self.OPRT_coef_lap[ii, jj, k0, l, 5] += gmtr.GMTR_t[im, jm, k0, l, TJ, T_RAREA] * (
                    +rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    +rdtype(2.0) * gmtr.GMTR_a[im, jj,   k0, l, AI, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AJ, tn] * gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    +rdtype(2.0) * gmtr.GMTR_a[im, jj,   k0, l, AI, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AJ, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                )

                self.OPRT_coef_lap[ii, jj, k0, l, 5] += gmtr.GMTR_t[im, jm, k0, l, TI, T_RAREA] * (
                    +rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AI, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    -rdtype(2.0) * gmtr.GMTR_a[ii, jm,   k0, l, AJ, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AI, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    -rdtype(2.0) * gmtr.GMTR_a[ii, jm,   k0, l, AJ, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                )

                # coef_lap[ii, jj, k0, l, 6]
                self.OPRT_coef_lap[ii, jj, k0, l, 6] += gmtr.GMTR_t[im, jm, k0, l, TI, T_RAREA] * (
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jm,   k0, l, AJ, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    +rdtype(2.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AI, tn] * gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jm,   k0, l, AJ, tn] * gmtr.GMTR_a[ii, jm,   k0, l, AJ, hn]
                    +rdtype(2.0) * gmtr.GMTR_a[im, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jm,   k0, l, AJ, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[im, jm, k0, l, AI, tn] * gmtr.GMTR_a[ii, jm,   k0, l, AJ, hn]
                )

                self.OPRT_coef_lap[ii, jj, k0, l, 6] += gmtr.GMTR_t[ii, jm, k0, l, TJ, T_RAREA] * (
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    +rdtype(2.0) * gmtr.GMTR_a[ii, jj,   k0, l, AI, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[ii, jm, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    -rdtype(1.0) * gmtr.GMTR_a[ii, jm, k0, l, AIJ, tn] * gmtr.GMTR_a[ii, jj,   k0, l, AI, hn]
                    -rdtype(2.0) * gmtr.GMTR_a[ii, jj,   k0, l, AI, tn] * gmtr.GMTR_a[ii, jj,   k0, l, AI, hn]
                    +rdtype(1.0) * gmtr.GMTR_a[ii, jm, k0, l, AJ, tn] * gmtr.GMTR_a[ii, jj,   k0, l, AI, hn]
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
                ii = slice(gmin-1, gmax + 1); ip = slice((gmin-1) + 1, (gmax + 1) + 1); im = slice((gmin-1) - 1, (gmax + 1) - 1)
                jj = slice(gmin-1, gmax + 1); jp = slice((gmin-1) + 1, (gmax + 1) + 1); jm = slice((gmin-1) - 1, (gmax + 1) - 1)

                # Vectorized interior (i,j): scalar i/i+/-1, j/j+/-1 -> slices.

                self.OPRT_coef_intp[ii, jj, k0, l, d, 0, TI] = (
                    + gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn] - gmtr.GMTR_a[ii, jj, k0, l, AI, tn]
                ) * rdtype(0.5) * gmtr.GMTR_t[ii, jj, k0, l, TI, T_RAREA]

                self.OPRT_coef_intp[ii, jj, k0, l, d, 1, TI] = (
                    - gmtr.GMTR_a[ii, jj, k0, l, AI, tn] - gmtr.GMTR_a[ip, jj, k0, l, AJ, tn]
                ) * rdtype(0.5) * gmtr.GMTR_t[ii, jj, k0, l, TI, T_RAREA]

                self.OPRT_coef_intp[ii, jj, k0, l, d, 2, TI] = (
                    - gmtr.GMTR_a[ip, jj, k0, l, AJ, tn] + gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn]
                ) * rdtype(0.5) * gmtr.GMTR_t[ii, jj, k0, l, TI, T_RAREA]

                self.OPRT_coef_intp[ii, jj, k0, l, d, 0, TJ] = (
                    + gmtr.GMTR_a[ii, jj, k0, l, AJ, tn] - gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn]
                ) * rdtype(0.5) * gmtr.GMTR_t[ii, jj, k0, l, TJ, T_RAREA]

                self.OPRT_coef_intp[ii, jj, k0, l, d, 1, TJ] = (
                    - gmtr.GMTR_a[ii, jj, k0, l, AIJ, tn] + gmtr.GMTR_a[ii, jp, k0, l, AI, tn]
                ) * rdtype(0.5) * gmtr.GMTR_t[ii, jj, k0, l, TJ, T_RAREA]

                self.OPRT_coef_intp[ii, jj, k0, l, d, 2, TJ] = (
                    + gmtr.GMTR_a[ii, jp, k0, l, AI, tn] + gmtr.GMTR_a[ii, jj, k0, l, AJ, tn]
                ) * rdtype(0.5) * gmtr.GMTR_t[ii, jj, k0, l, TJ, T_RAREA]

        for l in range(lall):
            for d in range(nxyz):

                hn = d + HNX

                                # 1  to  16 (inner grid points)
                ii = slice(gmin, gmax + 1); ip = slice((gmin) + 1, (gmax + 1) + 1); im = slice((gmin) - 1, (gmax + 1) - 1)
                jj = slice(gmin, gmax + 1); jp = slice((gmin) + 1, (gmax + 1) + 1); jm = slice((gmin) - 1, (gmax + 1) - 1)

                # Vectorized interior (i,j): scalar i/i+/-1, j/j+/-1 -> slices.

                self.OPRT_coef_diff[ii, jj, k0, l, d, 0] = (   ##### CCCHHHEEECCCKKK
                    + gmtr.GMTR_a[ii, jj, k0, l, AIJ, hn]
                    * rdtype(0.5)
                    * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]
                )

                self.OPRT_coef_diff[ii, jj, k0, l, d, 1] = (
                    + gmtr.GMTR_a[ii, jj, k0, l, AJ, hn]
                    * rdtype(0.5)
                    * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]
                )

                self.OPRT_coef_diff[ii, jj, k0, l, d, 2] = (
                    - gmtr.GMTR_a[im, jj, k0, l, AI, hn]
                    * rdtype(0.5)
                    * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]
                )

                self.OPRT_coef_diff[ii, jj, k0, l, d, 3] = (
                    - gmtr.GMTR_a[im, jm, k0, l, AIJ, hn]
                    * rdtype(0.5)
                    * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]
                )

                self.OPRT_coef_diff[ii, jj, k0, l, d, 4] = (
                    - gmtr.GMTR_a[ii, jm, k0, l, AJ, hn]
                    * rdtype(0.5)
                    * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]
                )

                self.OPRT_coef_diff[ii, jj, k0, l, d, 5] = (
                    + gmtr.GMTR_a[ii, jj, k0, l, AI, hn]
                    * rdtype(0.5)
                    * gmtr.GMTR_p[ii, jj, k0, l, P_RAREA]
                )

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

    # ------------------------------------------------------------------
    # Operator kernels (public API). Each delegates to its backend-
    # switchable fused body below; the original loop/_ij reference
    # implementations were removed 2026-07-25 (git history has them).
    # ------------------------------------------------------------------

    def OPRT_gradient(self, grad, grad_pl, scl, scl_pl, coef_grad, coef_grad_pl, grd, rdtype,
                      resident=False, scl_pl_d=None, resident_pl=False):

        prf.PROF_rapstart('OPRT_gradient', 2)

        # --- COMM-free body via backend-switchable kernel (numpy<->jax) ---
        # See kernels/oprtgradient.py. RES-TP-2: resident=True returns the device
        # regular grad (scl is device, no host drain); requires the fused path.
        # RES-TRACER-2: scl_pl_d threads a device pole scl in; resident_pl returns the
        # device pole grad (no host grad_pl drain).
        _g = self._oprt_gradient_fused(
            grad, grad_pl, scl, scl_pl, coef_grad, coef_grad_pl, grd,
            resident=resident, scl_pl_d=scl_pl_d, resident_pl=resident_pl,
        )
        prf.PROF_rapend('OPRT_gradient', 2)
        return _g

    def OPRT_horizontalize_vec(self,
            vx, vx_pl,        #[INOUT]
            vy, vy_pl,        #[INOUT]
            vz, vz_pl,        #[INOUT]
            grd, rdtype, resident=False):

        if grd.GRD_grid_type == grd.GRD_grid_type_on_plane:
            # planar grid: no radial component to remove. resident callers expect
            # device arrays back -> return the inputs unchanged.
            return (vx, vy, vz, vx_pl, vy_pl, vz_pl) if resident else None

        prf.PROF_rapstart('OPRT_horizontalize_vec', 2)

        # --- backend-switchable kernel (numpy<->jax). See
        # kernels/horizontalizevec.py. Default ON. ---
        # resident=True (jax only): device in/out, skip the asarray host-gather of
        # the (strided) inputs and the to_numpy drains; returns the projected
        # device arrays (vx,vy,vz, vx_pl,vy_pl,vz_pl) for a caller keeping the
        # field on device. Numpy/non-fused path does not support resident.
        out = self._horizontalize_vec_fused(
            vx, vx_pl, vy, vy_pl, vz, vz_pl, grd, rdtype, resident=resident,
        )
        prf.PROF_rapend('OPRT_horizontalize_vec', 2)
        return out

    def OPRT_laplacian(self, scl, scl_pl, coef_lap, coef_lap_pl, rdtype, resident=False):

        prf.PROF_rapstart('OPRT_laplacian', 2)

        # --- COMM-free body via backend-switchable kernel (numpy<->jax) ---
        # See kernels/oprtlaplacian.py.
        out = self._oprt_laplacian_fused(scl, scl_pl, coef_lap, coef_lap_pl, resident=resident)
        prf.PROF_rapend('OPRT_laplacian', 2)
        return out

    def OPRT_diffusion(self, 
                       scl, scl_pl,              #[IN]    
                       kh, kh_pl,                #[IN]    
                       coef_intp, coef_intp_pl,  #[IN]
                       coef_diff, coef_diff_pl,  #[IN]
                       grd, rdtype, resident=False):

        prf.PROF_rapstart('OPRT_diffusion', 2)

        # --- COMM-free body via backend-switchable kernel (numpy<->jax) ---
        # See kernels/oprtdiffusion.py.
        out = self._oprt_diffusion_fused(
            scl, scl_pl, kh, kh_pl,
            coef_intp, coef_intp_pl, coef_diff, coef_diff_pl, grd,
            resident=resident,
        )
        prf.PROF_rapend('OPRT_diffusion', 2)
        return out

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

        # --- whole COMM-free body via backend-switchable kernel (numpy<->jax) ---
        # See kernels/oprtdivdamp.py. Default OFF until validated.
        self._oprt_divdamp_fused(
            ddivdx, ddivdx_pl, ddivdy, ddivdy_pl, ddivdz, ddivdz_pl,
            vx, vx_pl, vy, vy_pl, vz, vz_pl,
            coef_intp, coef_intp_pl, coef_diff, coef_diff_pl, grd,
        )
        prf.PROF_rapend('OPRT_divdamp', 2)
        return

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

        # --- whole COMM-free body via backend-switchable kernel (numpy<->jax) ---
        # See kernels/oprt3ddivdamp.py. Validated bit-exact (numpy) /
        # single-call numpy-vs-jax (0.0); win in both backends. Default ON.
        self._oprt3d_divdamp_fused(
            ddivdx, ddivdx_pl, ddivdy, ddivdy_pl, ddivdz, ddivdz_pl,
            rhogvx, rhogvx_pl, rhogvy, rhogvy_pl, rhogvz, rhogvz_pl,
            rhogw, rhogw_pl,
            coef_intp, coef_intp_pl, coef_diff, coef_diff_pl,
            grd, vmtr,
        )
        prf.PROF_rapend('OPRT3D_divdamp', 2)
        return

    # ------------------------------------------------------------------
    # Fused backend-switchable bodies (numpy<->jax via bk.maybe_jit;
    # stencil math lives in nhm/dynamics/kernels/*.py)
    # ------------------------------------------------------------------

    def _oprt_gradient_fused(self,
        grad, grad_pl, scl, scl_pl, coef_grad, coef_grad_pl, grd,
        resident=False, scl_pl_d=None, resident_pl=False,
    ):
        """Backend-switchable replacement body for OPRT_gradient.

        coef_grad / coef_grad_pl are constant geometry (same object every call),
        so they are cached device-resident on first use. Results are written
        back in place; grad_pl is left untouched when not have_pl, matching the
        original (whose non-pole branch is a no-op on grad_pl).

        RES-TP-2: when resident=True, ``scl`` is already a device array and the
        regular ``grad`` is NOT drained to host -- the device grad handle is
        returned instead (the caller carries it). The pole (_pl) section still
        drains to host grad_pl. Bit-identical to the host path: the kernel is the
        same and asarray(to_numpy(.)) is a pure f64 copy.
        """
        xp = bk.xp
        if getattr(self, "_oprtgradient_kernel", None) is None:
            self._oprtgradient_cfg = OprtGradientCfg(
                have_pl=adm.ADM_have_pl,
                gslf_pl=adm.ADM_gslf_pl,
                gmax_pl=adm.ADM_gmax_pl,
                k0=adm.ADM_K0,
                XDIR=grd.GRD_XDIR, YDIR=grd.GRD_YDIR, ZDIR=grd.GRD_ZDIR,
            )
            self._oprtgradient_kernel = bk.maybe_jit(
                compute_oprt_gradient, static_argnames=("cfg", "xp"),
            )
        d = bk.device_consts(self, "oprtgradient", lambda: {
            "coef_grad":    coef_grad,
            "coef_grad_pl": coef_grad_pl,
        })

        # RES-TRACER-2: scl_pl_d (device pole scl, e.g. tracer q_pl_d) overrides the
        # host asarray(scl_pl) upload -> the host pole scl is no longer read.
        _scl_pl_in = scl_pl_d if scl_pl_d is not None else xp.asarray(scl_pl)
        _grad, _grad_pl = self._oprtgradient_kernel(
            (scl if resident else xp.asarray(scl)), _scl_pl_in,
            d["coef_grad"], d["coef_grad_pl"],
            cfg=self._oprtgradient_cfg, xp=xp,
        )
        if resident:
            # RES-TP-2: return the device regular grad; pole drained to host UNLESS
            # resident_pl -> then return the device pole grad too (caller keeps it on
            # device through the on-device COMM; no host grad_pl drain).
            if adm.ADM_have_pl:
                if resident_pl:
                    return _grad, _grad_pl
                grad_pl[:, :, :, :] = bk.to_numpy(_grad_pl)
            return _grad
        grad[:, :, :, :, :] = bk.to_numpy(_grad)
        if adm.ADM_have_pl:
            grad_pl[:, :, :, :] = bk.to_numpy(_grad_pl)
        return None

    #> 3D divergence damping operator

    def _horizontalize_vec_fused(self,
        vx, vx_pl, vy, vy_pl, vz, vz_pl, grd, rdtype, resident=False,
    ):
        """Backend-switchable replacement body for OPRT_horizontalize_vec.

        INOUT: only the interior i,j = 1..iall-2 of the regional buffers is
        modified (halos COMM-refilled); the whole pole array is rewritten.
        GRD_x / GRD_x_pl are constant geometry, cached device-resident.

        resident=True (jax only): xp.asarray on the inputs is a no-op when they
        are already device arrays (no host strided gather); skip the to_numpy
        D2H and RETURN the projected device arrays -- regional interior replaced
        functionally via .at[isl,jsl].set() (halo preserved, matching the in-place
        INOUT), pole rewritten full -- as (vx, vy, vz, vx_pl, vy_pl, vz_pl).
        """
        xp = bk.xp
        if getattr(self, "_horizontalize_kernel", None) is None:
            self._horizontalize_cfg = HorizontalizeVecCfg(
                have_pl=adm.ADM_have_pl,
                XDIR=grd.GRD_XDIR, YDIR=grd.GRD_YDIR, ZDIR=grd.GRD_ZDIR,
            )
            self._horizontalize_kernel = bk.maybe_jit(
                compute_horizontalize_vec, static_argnames=("cfg", "xp"),
            )
        # rscale is a scalar -> passes through device_consts unchanged.
        d = bk.device_consts(self, "horizontalize", lambda: {
            "GRD_x":    grd.GRD_x,
            "GRD_x_pl": grd.GRD_x_pl,
            "rscale":   grd.GRD_rscale,
        })

        # asarray is a no-op when inputs are already device arrays (resident path).
        vx_d = xp.asarray(vx); vy_d = xp.asarray(vy); vz_d = xp.asarray(vz)
        vx_pl_d = xp.asarray(vx_pl); vy_pl_d = xp.asarray(vy_pl); vz_pl_d = xp.asarray(vz_pl)

        nvx, nvy, nvz, nvx_pl, nvy_pl, nvz_pl = self._horizontalize_kernel(
            vx_d, vy_d, vz_d, vx_pl_d, vy_pl_d, vz_pl_d,
            d["GRD_x"], d["GRD_x_pl"], d["rscale"],
            cfg=self._horizontalize_cfg, xp=xp,
        )

        iall = adm.ADM_gall_1d
        isl = slice(1, iall - 1)
        jsl = slice(1, iall - 1)
        if resident:
            # device in/out: interior replaced (halo preserved), pole rewritten.
            return (
                vx_d.at[isl, jsl, :, :].set(nvx),
                vy_d.at[isl, jsl, :, :].set(nvy),
                vz_d.at[isl, jsl, :, :].set(nvz),
                nvx_pl, nvy_pl, nvz_pl,
            )
        vx[isl, jsl, :, :] = bk.to_numpy(nvx)
        vy[isl, jsl, :, :] = bk.to_numpy(nvy)
        vz[isl, jsl, :, :] = bk.to_numpy(nvz)
        vx_pl[:, :, :] = bk.to_numpy(nvx_pl)
        vy_pl[:, :, :] = bk.to_numpy(nvy_pl)
        vz_pl[:, :, :] = bk.to_numpy(nvz_pl)

    def _oprt_laplacian_fused(self, scl, scl_pl, coef_lap, coef_lap_pl, resident=False):
        """Backend-switchable replacement body for OPRT_laplacian.

        coef_lap / coef_lap_pl are constant geometry (same object every call),
        so they are cached device-resident on first use.

        resident=True (jax only): skip the bk.to_numpy D2H on the outputs and
        return the jax arrays directly, so a caller (e.g. the resident
        numfilter_hdiffusion lap-order loop) can keep intermediates on device
        across successive operator calls. xp.asarray on the inputs is a no-op
        when they are already device arrays, so the input side needs no change.
        """
        xp = bk.xp
        if getattr(self, "_oprtlaplacian_kernel", None) is None:
            self._oprtlaplacian_cfg = OprtLaplacianCfg(
                have_pl=adm.ADM_have_pl,
                gslf_pl=adm.ADM_gslf_pl,
                gmax_pl=adm.ADM_gmax_pl,
            )
            self._oprtlaplacian_kernel = bk.maybe_jit(
                compute_oprt_laplacian, static_argnames=("cfg", "xp"),
            )
        d = bk.device_consts(self, "oprtlaplacian", lambda: {
            "coef_lap":    coef_lap,
            "coef_lap_pl": coef_lap_pl,
        })

        _dscl, _dscl_pl = self._oprtlaplacian_kernel(
            xp.asarray(scl), xp.asarray(scl_pl),
            d["coef_lap"], d["coef_lap_pl"],
            cfg=self._oprtlaplacian_cfg, xp=xp,
        )
        if resident:
            return _dscl, _dscl_pl
        return bk.to_numpy(_dscl), bk.to_numpy(_dscl_pl)

    def _oprt_diffusion_fused(self,
        scl, scl_pl, kh, kh_pl,
        coef_intp, coef_intp_pl, coef_diff, coef_diff_pl, grd,
        resident=False,
    ):
        """Backend-switchable replacement body for OPRT_diffusion.

        coef_intp / coef_diff and the singular-point mask are constant geometry
        (same object every call), so they are cached device-resident on first
        use. Only the per-call variable fields (scl, kh) cross the boundary.

        resident=True (jax only): skip the bk.to_numpy D2H on the outputs and
        return the jax arrays directly (see _oprt_laplacian_fused).
        """
        xp = bk.xp
        if getattr(self, "_oprtdiffusion_kernel", None) is None:
            self._oprtdiffusion_cfg = OprtDiffusionCfg(
                have_pl=adm.ADM_have_pl,
                gmin=adm.ADM_gmin, gmax=adm.ADM_gmax,
                nxyz=adm.ADM_nxyz,
                gslf_pl=adm.ADM_gslf_pl,
                gmin_pl=adm.ADM_gmin_pl,
                gmax_pl=adm.ADM_gmax_pl,
                k0=adm.ADM_K0,
                TI=adm.ADM_TI, TJ=adm.ADM_TJ,
            )
            self._oprtdiffusion_kernel = bk.maybe_jit(
                compute_oprt_diffusion, static_argnames=("cfg", "xp"),
            )
        d = bk.device_consts(self, "oprtdiffusion", lambda: {
            "coef_intp":    coef_intp,
            "coef_diff":    coef_diff,
            "coef_intp_pl": coef_intp_pl,
            "coef_diff_pl": coef_diff_pl,
            "pntmask":      ppm.pntmask,
        })

        _dscl, _dscl_pl = self._oprtdiffusion_kernel(
            xp.asarray(scl), xp.asarray(scl_pl),
            xp.asarray(kh), xp.asarray(kh_pl),
            d["coef_intp"], d["coef_intp_pl"],
            d["coef_diff"], d["coef_diff_pl"],
            d["pntmask"],
            cfg=self._oprtdiffusion_cfg, xp=xp,
        )
        if resident:
            return _dscl, _dscl_pl
        return bk.to_numpy(_dscl), bk.to_numpy(_dscl_pl)

    def _oprt_divdamp_fused(self,
        ddivdx, ddivdx_pl, ddivdy, ddivdy_pl, ddivdz, ddivdz_pl,
        vx, vx_pl, vy, vy_pl, vz, vz_pl,
        coef_intp, coef_intp_pl, coef_diff, coef_diff_pl, grd,
    ):
        """Backend-switchable replacement body for OPRT_divdamp.

        coef_intp / coef_diff are constant geometry (same object every call),
        so they are cached device-resident on first use.
        """
        xp = bk.xp
        if getattr(self, "_oprtdivdamp_kernel", None) is None:
            self._oprtdivdamp_cfg = OprtDivdampCfg(
                have_pl=adm.ADM_have_pl,
                gmax=adm.ADM_gmax,
                gslf_pl=adm.ADM_gslf_pl,
                gmin_pl=adm.ADM_gmin_pl,
                gmax_pl=adm.ADM_gmax_pl,
                k0=adm.ADM_K0,
                TI=adm.ADM_TI, TJ=adm.ADM_TJ,
                XDIR=grd.GRD_XDIR, YDIR=grd.GRD_YDIR, ZDIR=grd.GRD_ZDIR,
            )
            self._oprtdivdamp_kernel = bk.maybe_jit(
                compute_oprt_divdamp, static_argnames=("cfg", "xp"),
            )
        d = bk.device_consts(self, "oprtdivdamp", lambda: {
            "coef_intp":    coef_intp,
            "coef_diff":    coef_diff,
            "coef_intp_pl": coef_intp_pl,
            "coef_diff_pl": coef_diff_pl,
        })

        _dx, _dy, _dz, _dx_pl, _dy_pl, _dz_pl = self._oprtdivdamp_kernel(
            xp.asarray(vx), xp.asarray(vy), xp.asarray(vz),
            xp.asarray(vx_pl), xp.asarray(vy_pl), xp.asarray(vz_pl),
            d["coef_intp"], d["coef_diff"], d["coef_intp_pl"], d["coef_diff_pl"],
            cfg=self._oprtdivdamp_cfg, xp=xp,
        )

        ddivdx[:, :, :, :] = bk.to_numpy(_dx)
        ddivdy[:, :, :, :] = bk.to_numpy(_dy)
        ddivdz[:, :, :, :] = bk.to_numpy(_dz)
        if adm.ADM_have_pl:
            ddivdx_pl[:, :, :] = bk.to_numpy(_dx_pl)
            ddivdy_pl[:, :, :] = bk.to_numpy(_dy_pl)
            ddivdz_pl[:, :, :] = bk.to_numpy(_dz_pl)

    def _oprt3d_divdamp_fused(self,
        ddivdx, ddivdx_pl, ddivdy, ddivdy_pl, ddivdz, ddivdz_pl,
        rhogvx, rhogvx_pl, rhogvy, rhogvy_pl, rhogvz, rhogvz_pl,
        rhogw, rhogw_pl,
        coef_intp, coef_intp_pl, coef_diff, coef_diff_pl,
        grd, vmtr,
    ):
        """Backend-switchable replacement body for OPRT3D_divdamp.

        coef_intp/coef_diff, the VMTR metric arrays, GRD_rdgz and the
        singular-point mask are constant geometry (same object every call),
        so they are cached device-resident on first use.
        """
        xp = bk.xp
        if getattr(self, "_oprt3ddivdamp_kernel", None) is None:
            self._oprt3ddivdamp_cfg = Oprt3DDivdampCfg(
                have_pl=adm.ADM_have_pl,
                kmin=adm.ADM_kmin, kmax=adm.ADM_kmax,
                gmax=adm.ADM_gmax,
                gslf_pl=adm.ADM_gslf_pl,
                gmin_pl=adm.ADM_gmin_pl,
                gmax_pl=adm.ADM_gmax_pl,
                k0=adm.ADM_K0,
                TI=adm.ADM_TI, TJ=adm.ADM_TJ,
                XDIR=grd.GRD_XDIR, YDIR=grd.GRD_YDIR, ZDIR=grd.GRD_ZDIR,
            )
            self._oprt3ddivdamp_kernel = bk.maybe_jit(
                compute_oprt3d_divdamp, static_argnames=("cfg", "xp"),
            )
        d = bk.device_consts(self, "oprt3ddivdamp", lambda: {
            "coef_intp":    coef_intp,
            "coef_diff":    coef_diff,
            "coef_intp_pl": coef_intp_pl,
            "coef_diff_pl": coef_diff_pl,
            "C2WfactGz":    vmtr.VMTR_C2WfactGz,
            "RGAMH":        vmtr.VMTR_RGAMH,
            "RGSQRTH":      vmtr.VMTR_RGSQRTH,
            "RGAM":         vmtr.VMTR_RGAM,
            "C2WfactGz_pl": vmtr.VMTR_C2WfactGz_pl,
            "RGAMH_pl":     vmtr.VMTR_RGAMH_pl,
            "RGSQRTH_pl":   vmtr.VMTR_RGSQRTH_pl,
            "RGAM_pl":      vmtr.VMTR_RGAM_pl,
            "rdgz":         grd.GRD_rdgz,
            "pntmask":      ppm.pntmask,
        })

        _dx, _dy, _dz, _dx_pl, _dy_pl, _dz_pl = self._oprt3ddivdamp_kernel(
            xp.asarray(rhogvx), xp.asarray(rhogvy), xp.asarray(rhogvz),
            xp.asarray(rhogw),
            xp.asarray(rhogvx_pl), xp.asarray(rhogvy_pl), xp.asarray(rhogvz_pl),
            xp.asarray(rhogw_pl),
            d["coef_intp"], d["coef_diff"], d["coef_intp_pl"], d["coef_diff_pl"],
            d["C2WfactGz"], d["RGAMH"], d["RGSQRTH"], d["RGAM"],
            d["C2WfactGz_pl"], d["RGAMH_pl"], d["RGSQRTH_pl"], d["RGAM_pl"],
            d["rdgz"], d["pntmask"],
            cfg=self._oprt3ddivdamp_cfg, xp=xp,
        )

        ddivdx[:, :, :, :] = bk.to_numpy(_dx)
        ddivdy[:, :, :, :] = bk.to_numpy(_dy)
        ddivdz[:, :, :, :] = bk.to_numpy(_dz)
        if adm.ADM_have_pl:
            ddivdx_pl[:, :, :] = bk.to_numpy(_dx_pl)
            ddivdy_pl[:, :, :] = bk.to_numpy(_dy_pl)
            ddivdz_pl[:, :, :] = bk.to_numpy(_dz_pl)

    def _oprt3d_divdamp_device(self,
        rhogvx, rhogvx_pl, rhogvy, rhogvy_pl, rhogvz, rhogvz_pl,
        rhogw, rhogw_pl,
        coef_intp, coef_intp_pl, coef_diff, coef_diff_pl,
        grd, vmtr,
    ):
        """STEP-7: same fused OPRT3D_divdamp body as _oprt3d_divdamp_fused, but
        RETURNS the device (jax) arrays (dx, dy, dz, dx_pl, dy_pl, dz_pl) WITHOUT
        draining to host. Lets numfilter_divdamp keep vtmp2 device-resident across
        the lap COMM (on-device) and feed the post-COMM fused kernel with no
        intervening D2H/H2D. Shares the same cached kernel/cfg/consts as the fused
        path; jax-only caller. Bit-exact vs the fused path (identical kernel)."""
        xp = bk.xp
        if getattr(self, "_oprt3ddivdamp_kernel", None) is None:
            self._oprt3ddivdamp_cfg = Oprt3DDivdampCfg(
                have_pl=adm.ADM_have_pl,
                kmin=adm.ADM_kmin, kmax=adm.ADM_kmax,
                gmax=adm.ADM_gmax,
                gslf_pl=adm.ADM_gslf_pl,
                gmin_pl=adm.ADM_gmin_pl,
                gmax_pl=adm.ADM_gmax_pl,
                k0=adm.ADM_K0,
                TI=adm.ADM_TI, TJ=adm.ADM_TJ,
                XDIR=grd.GRD_XDIR, YDIR=grd.GRD_YDIR, ZDIR=grd.GRD_ZDIR,
            )
            self._oprt3ddivdamp_kernel = bk.maybe_jit(
                compute_oprt3d_divdamp, static_argnames=("cfg", "xp"),
            )
        d = bk.device_consts(self, "oprt3ddivdamp", lambda: {
            "coef_intp":    coef_intp,
            "coef_diff":    coef_diff,
            "coef_intp_pl": coef_intp_pl,
            "coef_diff_pl": coef_diff_pl,
            "C2WfactGz":    vmtr.VMTR_C2WfactGz,
            "RGAMH":        vmtr.VMTR_RGAMH,
            "RGSQRTH":      vmtr.VMTR_RGSQRTH,
            "RGAM":         vmtr.VMTR_RGAM,
            "C2WfactGz_pl": vmtr.VMTR_C2WfactGz_pl,
            "RGAMH_pl":     vmtr.VMTR_RGAMH_pl,
            "RGSQRTH_pl":   vmtr.VMTR_RGSQRTH_pl,
            "RGAM_pl":      vmtr.VMTR_RGAM_pl,
            "rdgz":         grd.GRD_rdgz,
            "pntmask":      ppm.pntmask,
        })
        return self._oprt3ddivdamp_kernel(
            xp.asarray(rhogvx), xp.asarray(rhogvy), xp.asarray(rhogvz),
            xp.asarray(rhogw),
            xp.asarray(rhogvx_pl), xp.asarray(rhogvy_pl), xp.asarray(rhogvz_pl),
            xp.asarray(rhogw_pl),
            d["coef_intp"], d["coef_diff"], d["coef_intp_pl"], d["coef_diff_pl"],
            d["C2WfactGz"], d["RGAMH"], d["RGSQRTH"], d["RGAM"],
            d["C2WfactGz_pl"], d["RGAMH_pl"], d["RGSQRTH_pl"], d["RGAM_pl"],
            d["rdgz"], d["pntmask"],
            cfg=self._oprt3ddivdamp_cfg, xp=xp,
        )
