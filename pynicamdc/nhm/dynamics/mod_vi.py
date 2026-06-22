import os
import numpy as np
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_prof import prf
from pynicamdc.share.mod_backend import backend as bk
from pynicamdc.nhm.dynamics.kernels.vimatrix import (
    ViMatrixCfg, compute_rhow_matrix_reg, compute_rhow_matrix_pl,
)
from pynicamdc.nhm.dynamics.kernels.virhowsolver import (
    ViSolverCfg, compute_rhow_solver_reg, compute_rhow_solver_pl,
)
from pynicamdc.nhm.dynamics.kernels.fluxconv import FluxConvCfg
from pynicamdc.nhm.dynamics.kernels.advconv import AdvConvCfg
from pynicamdc.nhm.dynamics.kernels.bndcnd import BndCfg
from pynicamdc.nhm.dynamics.kernels.rhogkin import RhogkinCfg
from pynicamdc.nhm.dynamics.kernels.vimain import VimainCfg, compute_vi_main
from pynicamdc.nhm.dynamics.kernels.presgrad import PresGradCfg
from pynicamdc.nhm.dynamics.kernels.vipath1 import ViPath1Cfg, compute_vi_path1
from pynicamdc.nhm.dynamics.kernels.vipath2 import ViPath2Cfg, compute_vi_path2_update

class Vi:
    
    _instance = None
    
    def __init__(self):
        pass

    counter = -1

    def vi_setup(self, cnst, rdtype):

        self.Mc    = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        self.Mc_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        self.Mu    = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        self.Mu_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        self.Ml    = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        self.Ml_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)

        return

    def vi_small_step(self,
            PROG,       PROG_pl,            #INOUT
            vx,         vx_pl,         
            vy,         vy_pl,         
            vz,         vz_pl,         
            eth,        eth_pl,        
            rhog_prim,  rhog_prim_pl,  
            preg_prim,  preg_prim_pl,  
            g_TEND0,    g_TEND0_pl,    
            PROG_split, PROG_split_pl,      #INOUT
            PROG_mean,  PROG_mean_pl,       #OUT
            num_of_itr,                
            dt,                             # DOUBLE
            cnst, comm, grd, oprt, vmtr, tim, rcnf, bndc, cnvv, numf, src, rdtype,                  
    ):
        
        prf.PROF_rapstart('____vi_path0',2)
        prf.PROF_rapstart('_____vp0_halflev',2)   # decompose vi_path0 self-cost (instrument-first)

        gall_1d = adm.ADM_gall_1d
        gall_pl = adm.ADM_gall_pl
        kall = adm.ADM_kall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        lall_pl = adm.ADM_lall_pl
        
        grhogetot0    = np.full(adm.ADM_shape,             cnst.CONST_UNDEF, dtype=rdtype)
        grhogetot0_pl = np.full(adm.ADM_shape_pl,          cnst.CONST_UNDEF, dtype=rdtype)
        rhog_h        = np.full(adm.ADM_shape,             cnst.CONST_UNDEF, dtype=rdtype)
        eth_h         = np.full(adm.ADM_shape,             cnst.CONST_UNDEF, dtype=rdtype)
        rhog_h_pl     = np.full(adm.ADM_shape_pl,          cnst.CONST_UNDEF, dtype=rdtype)
        eth_h_pl      = np.full(adm.ADM_shape_pl,          cnst.CONST_UNDEF, dtype=rdtype)
        drhog         = np.full(adm.ADM_shape,             cnst.CONST_UNDEF, dtype=rdtype)
        drhog_pl      = np.full(adm.ADM_shape_pl,          cnst.CONST_UNDEF, dtype=rdtype)
        dpgrad        = np.full((adm.ADM_shape    + (3,)), cnst.CONST_UNDEF, dtype=rdtype)  # additional dimension for XDIR YDIR ZDIR
        dpgrad_pl     = np.full((adm.ADM_shape_pl + (3,)), cnst.CONST_UNDEF, dtype=rdtype)  # additional dimension for XDIR YDIR ZDIR
        dpgradw       = np.full(adm.ADM_shape,             cnst.CONST_UNDEF, dtype=rdtype)
        dpgradw_pl    = np.full(adm.ADM_shape_pl,          cnst.CONST_UNDEF, dtype=rdtype)
        dbuoiw        = np.full(adm.ADM_shape,             cnst.CONST_UNDEF, dtype=rdtype)
        dbuoiw_pl     = np.full(adm.ADM_shape_pl,          cnst.CONST_UNDEF, dtype=rdtype)
        drhoge        = np.full(adm.ADM_shape,             cnst.CONST_UNDEF, dtype=rdtype)
        drhoge_pl     = np.full(adm.ADM_shape_pl,          cnst.CONST_UNDEF, dtype=rdtype)
        gz_tilde      = np.full(adm.ADM_shape,             cnst.CONST_UNDEF, dtype=rdtype)
        gz_tilde_pl   = np.full(adm.ADM_shape_pl,          cnst.CONST_UNDEF, dtype=rdtype)
        drhoge_pw     = np.full(adm.ADM_shape,             cnst.CONST_UNDEF, dtype=rdtype)
        drhoge_pw_pl  = np.full(adm.ADM_shape_pl,          cnst.CONST_UNDEF, dtype=rdtype)
        drhoge_pwh    = np.full(adm.ADM_shape,             cnst.CONST_UNDEF, dtype=rdtype)
        drhoge_pwh_pl = np.full(adm.ADM_shape_pl,          cnst.CONST_UNDEF, dtype=rdtype)
        g_TEND        = np.full((adm.ADM_shape    + (6,)), cnst.CONST_UNDEF, dtype=rdtype)  
        g_TEND_pl     = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)  # additional dimension for I_RHOG to I_RHOGE

        ddivdvx       = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvx_pl    = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvx_2d    = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvx_2d_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvy       = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvy_pl    = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvy_2d    = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvy_2d_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvz       = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvz_pl    = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvz_2d    = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvz_2d_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        ddivdw        = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        ddivdw_pl     = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)

        preg_prim_split     = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        preg_prim_split_pl  = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)

        drhogw        = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        drhogw_pl     = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)

        drhogw        = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        drhogw_pl     = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)

        diff_vh       = np.full((adm.ADM_shape    + (3,)), cnst.CONST_UNDEF, dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ
        diff_vh_pl    = np.full((adm.ADM_shape_pl + (3,)), cnst.CONST_UNDEF, dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ
        diff_we       = np.full((adm.ADM_shape    + (3,)), cnst.CONST_UNDEF, dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ
        diff_we_pl    = np.full((adm.ADM_shape_pl + (3,)), cnst.CONST_UNDEF, dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ

        XDIR = grd.GRD_XDIR
        YDIR = grd.GRD_YDIR
        ZDIR = grd.GRD_ZDIR     

        GRAV  = cnst.CONST_GRAV
        RovCV = cnst.CONST_Rdry / cnst.CONST_CVdry
        alpha = rdtype(rcnf.NON_HYDRO_ALPHA)

        I_RHOG  = rcnf.I_RHOG
        I_RHOGVX = rcnf.I_RHOGVX
        I_RHOGVY = rcnf.I_RHOGVY
        I_RHOGVZ = rcnf.I_RHOGVZ
        I_RHOGW = rcnf.I_RHOGW
        I_RHOGE = rcnf.I_RHOGE

        grhogetot0[:, :, :, :] = g_TEND0[:, :, :, :, I_RHOGE]
        grhogetot0_pl[:, :, :] = g_TEND0_pl[:, :, :, I_RHOGE]


        # full level -> half level

        kslice = slice(kmin, kmax + 2)       # includes kmax+1
        kslice_m1 = slice(kmin - 1, kmax + 1)  # k-1

        # Vectorized rhog_h
        rhog_h[:, :, kslice, :] = (
            vmtr.VMTR_C2Wfact[:, :, kslice, :, 0] * PROG[:, :, kslice, :, I_RHOG] +
            vmtr.VMTR_C2Wfact[:, :, kslice, :, 1] * PROG[:, :, kslice_m1, :, I_RHOG]
        )

        # Vectorized eth_h
        # expand afact and bfact for broadcasting over i, j, l
        afact = grd.GRD_afact[kslice][None, None, :, None]
        bfact = grd.GRD_bfact[kslice][None, None, :, None]

        eth_h[:, :, kslice, :] = (
            afact * eth[:, :, kslice, :] +
            bfact * eth[:, :, kslice_m1, :]
        )


        rhog_h[:, :, kmin-1, :] = rhog_h[:, :, kmin, :]
        eth_h[:, :, kmin-1, :]  = eth_h[:, :, kmin, :]
        

        if adm.ADM_have_pl:
            #for l in range(adm.ADM_lall_pl):
            # Vectorized computation for kmin to kmax+1
            rhog_h_pl[:, kmin:kmax+2, :] = (
                vmtr.VMTR_C2Wfact_pl[:, kmin:kmax+2, :, 0] * PROG_pl[:, kmin:kmax+2, :, I_RHOG] +
                vmtr.VMTR_C2Wfact_pl[:, kmin:kmax+2, :, 1] * PROG_pl[:, kmin-1:kmax+1, :, I_RHOG]
            )

            eth_h_pl[:, kmin:kmax+2, :] = (
                grd.GRD_afact[kmin:kmax+2][None, :, None] * eth_pl[:, kmin:kmax+2, :] +   #Potential SIZESHAPEERROR Because of k ranges?
                grd.GRD_bfact[kmin:kmax+2][None, :, None] * eth_pl[:, kmin-1:kmax+1, :]
            )

            # Fill ghost level
            rhog_h_pl[:, kmin-1, :] = rhog_h_pl[:, kmin, :]
            eth_h_pl[:, kmin-1, :]  = eth_h_pl[:, kmin, :]
            #end l loop
        #endif

        # prc.prc_mpistop(std.io_l, std.fname_log)

        prf.PROF_rapend  ('_____vp0_halflev',2)
        prf.PROF_rapstart('_____vp0_srcterms',2)   # src_* + divdamp(timed children) + glue
        #---< Calculation of source term for rhog >

        # B.4: skip the numpy src.* islands when device-resident (the resident block
        # below recomputes drhog/dpgrad/dpgradw/dbuoiw/drhoge + gz_tilde on device).
        _resident_vp0 = (bk.type == "jax") and os.environ.get("PYNICAM_RESIDENT_VIPATH0", "0") != "0"

        if not _resident_vp0:
            src.src_flux_convergence(
                    PROG [:,:,:,:,I_RHOGVX], PROG_pl [:,:,:,I_RHOGVX],
                    PROG [:,:,:,:,I_RHOGVY], PROG_pl [:,:,:,I_RHOGVY],
                    PROG [:,:,:,:,I_RHOGVZ], PROG_pl [:,:,:,I_RHOGVZ],
                    PROG [:,:,:,:,I_RHOGW],  PROG_pl [:,:,:,I_RHOGW],
                    drhog[:,:,:,:],          drhog_pl[:,:,:],
                    src.I_SRC_default,
                    cnst, grd, oprt, vmtr, rdtype,
            )

        #---< Calculation of source term for Vh(vx,vy,vz) and W >

        # divergence damping
        numf.numfilter_divdamp(
            PROG   [:,:,:,:,I_RHOGVX], PROG_pl   [:,:,:,I_RHOGVX], # [IN]
            PROG   [:,:,:,:,I_RHOGVY], PROG_pl   [:,:,:,I_RHOGVY], # [IN]
            PROG   [:,:,:,:,I_RHOGVZ], PROG_pl   [:,:,:,I_RHOGVZ], # [IN]
            PROG   [:,:,:,:,I_RHOGW],  PROG_pl   [:,:,:,I_RHOGW],  # [IN]
            ddivdvx[:,:,:,:],          ddivdvx_pl[:,:,:],          # [OUT]   
            ddivdvy[:,:,:,:],          ddivdvy_pl[:,:,:],          # [OUT]
            ddivdvz[:,:,:,:],          ddivdvz_pl[:,:,:],          # [OUT]
            ddivdw [:,:,:,:],          ddivdw_pl [:,:,:],          # [OUT]
            cnst, comm, grd, oprt, vmtr, src, rdtype,
        )

        # with open (std.fname_log, 'a') as log_file:
        #     print("A: in vi_small_step, out of numfilter_divdamp ", file=log_file)
        #     print("ddivdvx[6,5,2,0] ", ddivdvx[6,5,2,0], file=log_file)
        #     print("ddivdvx_pl[0,2,0] ", ddivdvx_pl[0,2,0], file=log_file)
        #     print("ddivdvy[6,5,2,0] ", ddivdvy[6,5,2,0], file=log_file)
        #     print("ddivdvy_pl[0,2,0] ", ddivdvy_pl[0,2,0], file=log_file)
        #     print("ddivdvz[6,5,2,0] ", ddivdvz[6,5,2,0], file=log_file)
        #     print("ddivdvz_pl[0,2,0] ", ddivdvz_pl[0,2,0], file=log_file)
        #     print("ddivdw[6,5,2,0] ", ddivdw[6,5,2,0], file=log_file)
        #     print("ddivdw_pl[0,2,0] ", ddivdw_pl[0,2,0], file=log_file)

        numf.numfilter_divdamp_2d(
            PROG   [:,:,:,:,I_RHOGVX], PROG_pl   [:,:,:,I_RHOGVX], # [IN]
            PROG   [:,:,:,:,I_RHOGVY], PROG_pl   [:,:,:,I_RHOGVY], # [IN]
            PROG   [:,:,:,:,I_RHOGVZ], PROG_pl   [:,:,:,I_RHOGVZ], # [IN]
            ddivdvx_2d[:,:,:,:],       ddivdvx_2d_pl[:,:,:],       # [OUT]
            ddivdvy_2d[:,:,:,:],       ddivdvy_2d_pl[:,:,:],       # [OUT]
            ddivdvz_2d[:,:,:,:],       ddivdvz_2d_pl[:,:,:],       # [OUT]
            cnst, comm, grd, oprt, rdtype,
        )

        # with open (std.fname_log, 'a') as log_file:
        #     print("B: in vi_small_step, out of numfilter_divdamp_2d ", file=log_file)
        #     print("ddivdvx_2d[6,5,2,0] ", ddivdvx_2d[6,5,2,0], file=log_file)
        #     print("ddivdvx_2d_pl[0,2,0] ", ddivdvx_2d_pl[0,2,0], file=log_file)
        #     print("ddivdvy_2d[6,5,2,0] ", ddivdvy_2d[6,5,2,0], file=log_file)
        #     print("ddivdvy_2d_pl[0,2,0] ", ddivdvy_2d_pl[0,2,0], file=log_file)
        #     print("ddivdvz_2d[6,5,2,0] ", ddivdvz_2d[6,5,2,0], file=log_file)
        #     print("ddivdvz_2d_pl[0,2,0] ", ddivdvz_2d_pl[0,2,0], file=log_file)

        if not _resident_vp0:
            # pressure force
            src.src_pres_gradient(
                preg_prim[:,:,:,:],   preg_prim_pl[:,:,:],   # [IN]
                dpgrad   [:,:,:,:,:], dpgrad_pl   [:,:,:,:], # [OUT]
                dpgradw  [:,:,:,:],   dpgradw_pl  [:,:,:],   # [OUT]
                src.I_SRC_default,                           # [IN]
                cnst, grd, oprt, vmtr, rdtype,
            )

            # buoyancy force
            src.src_buoyancy(
                rhog_prim[:,:,:,:], rhog_prim_pl[:,:,:], # [IN]
                dbuoiw   [:,:,:,:], dbuoiw_pl   [:,:,:], # [OUT]
                cnst, vmtr, rdtype,
            )

            #---< Calculation of source term for rhoge >

            # advection convergence for eth

            src.src_advection_convergence(
                PROG  [:,:,:,:,I_RHOGVX], PROG_pl  [:,:,:,I_RHOGVX], # [IN]
                PROG  [:,:,:,:,I_RHOGVY], PROG_pl  [:,:,:,I_RHOGVY], # [IN]
                PROG  [:,:,:,:,I_RHOGVZ], PROG_pl  [:,:,:,I_RHOGVZ], # [IN]
                PROG  [:,:,:,:,I_RHOGW],  PROG_pl  [:,:,:,I_RHOGW],  # [IN]
                eth   [:,:,:,:],          eth_pl   [:,:,:],          # [IN]
                drhoge[:,:,:,:],          drhoge_pl[:,:,:],          # [OUT]
                src.I_SRC_default,                                   # [IN]
                cnst, grd, oprt, vmtr, rdtype,
            )

        prf.PROF_rapend  ('_____vp0_srcterms',2)
        prf.PROF_rapstart('_____vp0_preswork',2)   # gz_tilde + drhoge_pw (W2C interp)
        # pressure work

        # --- First part: compute gz_tilde and drhoge_pwh ---
        gz_tilde[:, :, :, :] = GRAV - (dpgradw - dbuoiw) / rhog_h
        drhoge_pwh[:, :, :, :] = -gz_tilde * PROG[:, :, :, :, I_RHOGW]

        # --- Second part: compute drhoge_pw for kmin ≤ k ≤ kmax ---
        k_slice     = slice(kmin,   kmax + 1)
        kp1_slice   = slice(kmin+1, kmax + 2)

        drhoge_pw[:, :, k_slice, :] = (
            vx[:, :, k_slice, :] * dpgrad[:, :, k_slice, :, XDIR] +
            vy[:, :, k_slice, :] * dpgrad[:, :, k_slice, :, YDIR] +
            vz[:, :, k_slice, :] * dpgrad[:, :, k_slice, :, ZDIR] +
            vmtr.VMTR_W2Cfact[:, :, k_slice, :, 0] * drhoge_pwh[:, :, kp1_slice, :] +
            vmtr.VMTR_W2Cfact[:, :, k_slice, :, 1] * drhoge_pwh[:, :, k_slice, :]
        )

        # --- Boundary values ---
        drhoge_pw[:, :, kmin - 1, :] = rdtype(0.0)
        drhoge_pw[:, :, kmax + 1, :] = rdtype(0.0)



        if adm.ADM_have_pl:
           
            # --- Vectorized gz_tilde_pl and drhoge_pwh_pl
            gz_tilde_pl[:, :, :] = GRAV - (dpgradw_pl[:, :, :] - dbuoiw_pl[:, :, :]) / rhog_h_pl[:, :, :]
            drhoge_pwh_pl[:, :, :] = -gz_tilde_pl[:, :, :] * PROG_pl[:, :, :, I_RHOGW]

            # --- Vectorized drhoge_pw_pl over kmin to kmax
            drhoge_pw_pl[:, kmin:kmax+1, :] = (
                vx_pl[:, kmin:kmax+1, :] * dpgrad_pl[:, kmin:kmax+1, :, XDIR] +
                vy_pl[:, kmin:kmax+1, :] * dpgrad_pl[:, kmin:kmax+1, :, YDIR] +
                vz_pl[:, kmin:kmax+1, :] * dpgrad_pl[:, kmin:kmax+1, :, ZDIR] +
                vmtr.VMTR_W2Cfact_pl[:, kmin:kmax+1, :, 0] * drhoge_pwh_pl[:, kmin+1:kmax+2, :] +
                vmtr.VMTR_W2Cfact_pl[:, kmin:kmax+1, :, 1] * drhoge_pwh_pl[:, kmin:kmax+1,   :]
            )

            # --- Ghost layers at boundaries
            drhoge_pw_pl[:, kmin-1, :] = rdtype(0.0)
            drhoge_pw_pl[:, kmax+1, :] = rdtype(0.0)
       
        #endif


        prf.PROF_rapend  ('_____vp0_preswork',2)
        prf.PROF_rapstart('_____vp0_tendsum',2)   # combine tendencies (+ gated resident block)
        #---< sum of tendencies ( large step + pres-grad + div-damp + div-damp_2d + buoyancy ) >

        g_TEND[:, :, :, :, I_RHOG]   = g_TEND0[:, :, :, :, I_RHOG] + drhog[:, :, :, :]

        g_TEND[:, :, :, :, I_RHOGVX] = (
            g_TEND0[:, :, :, :, I_RHOGVX]
            - dpgrad[:, :, :, :, XDIR]
            + ddivdvx[:, :, :, :]
            + ddivdvx_2d[:, :, :, :]
        )

        g_TEND[:, :, :, :, I_RHOGVY] = (
            g_TEND0[:, :, :, :, I_RHOGVY]
            - dpgrad[:, :, :, :, YDIR]
            + ddivdvy[:, :, :, :]
            + ddivdvy_2d[:, :, :, :]
        )

        g_TEND[:, :, :, :, I_RHOGVZ] = (
            g_TEND0[:, :, :, :, I_RHOGVZ]
            - dpgrad[:, :, :, :, ZDIR]
            + ddivdvz[:, :, :, :]
            + ddivdvz_2d[:, :, :, :]
        )

        g_TEND[:, :, :, :, I_RHOGW] = (
            g_TEND0[:, :, :, :, I_RHOGW]
            + ddivdw[:, :, :, :] * alpha
            - dpgradw[:, :, :, :]
            + dbuoiw[:, :, :, :]
        )

        g_TEND[:, :, :, :, I_RHOGE] = (
            g_TEND0[:, :, :, :, I_RHOGE]
            + drhoge[:, :, :, :]
            + drhoge_pw[:, :, :, :]
        )


        if adm.ADM_have_pl:
            g_TEND_pl[:, :, :, I_RHOG] = g_TEND0_pl[:, :, :, I_RHOG] + drhog_pl

            g_TEND_pl[:, :, :, I_RHOGVX] = (
                g_TEND0_pl[:, :, :, I_RHOGVX]
                - dpgrad_pl[:, :, :, XDIR]
                + ddivdvx_pl
                + ddivdvx_2d_pl
            )

            g_TEND_pl[:, :, :, I_RHOGVY] = (
                g_TEND0_pl[:, :, :, I_RHOGVY]
                - dpgrad_pl[:, :, :, YDIR]
                + ddivdvy_pl
                + ddivdvy_2d_pl
            )

            g_TEND_pl[:, :, :, I_RHOGVZ] = (
                g_TEND0_pl[:, :, :, I_RHOGVZ]
                - dpgrad_pl[:, :, :, ZDIR]
                + ddivdvz_pl
                + ddivdvz_2d_pl
            )

            g_TEND_pl[:, :, :, I_RHOGW] = (
                g_TEND0_pl[:, :, :, I_RHOGW]
                + ddivdw_pl * alpha
                - dpgradw_pl
                + dbuoiw_pl
            )

            g_TEND_pl[:, :, :, I_RHOGE] = (
                g_TEND0_pl[:, :, :, I_RHOGE]
                + drhoge_pl
                + drhoge_pw_pl
            )
        #endif

        # --- Step B.2/B.3 (gated): self-contained device-resident tendency setup.
        #     Recompute g_TEND fully on device: glue (rhog_h, gz_tilde, drhoge_pw) via
        #     functional jnp .at[].set() + resident src.* (jax outputs, no D2H) + combine.
        #     divdamp (ddivd*) stays numpy -> asarray. PYNICAM_RESIDENT_VIPATH0 default off.
        #     Validation-first: still appends after the numpy body (overwrites g_TEND).
        if _resident_vp0:
            _xp = bk.xp
            _UNDEF = cnst.CONST_UNDEF
            _ks  = slice(kmin, kmax + 2)
            _ksm = slice(kmin - 1, kmax + 1)
            _kc  = slice(kmin, kmax + 1)
            _kp1 = slice(kmin + 1, kmax + 2)
            _C2W = _xp.asarray(vmtr.VMTR_C2Wfact)
            _W2C = _xp.asarray(vmtr.VMTR_W2Cfact)
            _PROGd = _xp.asarray(PROG)
            # rhog_h on device (half-level interp + ghost copy)
            _rhogh = _xp.full(adm.ADM_shape, _UNDEF, dtype=rdtype)
            _rhogh = _rhogh.at[:, :, _ks, :].set(
                _C2W[:, :, _ks, :, 0] * _PROGd[:, :, _ks, :, I_RHOG]
                + _C2W[:, :, _ks, :, 1] * _PROGd[:, :, _ksm, :, I_RHOG])
            _rhogh = _rhogh.at[:, :, kmin - 1, :].set(_rhogh[:, :, kmin, :])
            # resident src.* (jax outputs)
            _drhog, _drhog_pl = src.src_flux_convergence(
                PROG[:,:,:,:,I_RHOGVX], PROG_pl[:,:,:,I_RHOGVX],
                PROG[:,:,:,:,I_RHOGVY], PROG_pl[:,:,:,I_RHOGVY],
                PROG[:,:,:,:,I_RHOGVZ], PROG_pl[:,:,:,I_RHOGVZ],
                PROG[:,:,:,:,I_RHOGW],  PROG_pl[:,:,:,I_RHOGW],
                None, None, src.I_SRC_default,
                cnst, grd, oprt, vmtr, rdtype, resident=True)
            _dpg, _dpgw, _dpg_pl, _dpgw_pl = src.src_pres_gradient(
                preg_prim, preg_prim_pl, None, None, None, None,
                src.I_SRC_default, cnst, grd, oprt, vmtr, rdtype, resident=True)
            _dbuo, _dbuo_pl = src.src_buoyancy(
                rhog_prim, rhog_prim_pl, None, None, cnst, vmtr, rdtype, resident=True)
            _drhoge, _drhoge_pl = src.src_advection_convergence(
                PROG[:,:,:,:,I_RHOGVX], PROG_pl[:,:,:,I_RHOGVX],
                PROG[:,:,:,:,I_RHOGVY], PROG_pl[:,:,:,I_RHOGVY],
                PROG[:,:,:,:,I_RHOGVZ], PROG_pl[:,:,:,I_RHOGVZ],
                PROG[:,:,:,:,I_RHOGW],  PROG_pl[:,:,:,I_RHOGW],
                eth, eth_pl, None, None, src.I_SRC_default,
                cnst, grd, oprt, vmtr, rdtype, resident=True)
            # pressure-work glue on device
            _gz  = GRAV - (_dpgw - _dbuo) / _rhogh
            _pwh = -_gz * _PROGd[:, :, :, :, I_RHOGW]
            _vxd = _xp.asarray(vx); _vyd = _xp.asarray(vy); _vzd = _xp.asarray(vz)
            _pw = _xp.full(adm.ADM_shape, _UNDEF, dtype=rdtype)
            _pw = _pw.at[:, :, _kc, :].set(
                _vxd[:, :, _kc, :] * _dpg[:, :, _kc, :, XDIR]
                + _vyd[:, :, _kc, :] * _dpg[:, :, _kc, :, YDIR]
                + _vzd[:, :, _kc, :] * _dpg[:, :, _kc, :, ZDIR]
                + _W2C[:, :, _kc, :, 0] * _pwh[:, :, _kp1, :]
                + _W2C[:, :, _kc, :, 1] * _pwh[:, :, _kc, :])
            _pw = _pw.at[:, :, kmin - 1, :].set(rdtype(0.0))
            _pw = _pw.at[:, :, kmax + 1, :].set(rdtype(0.0))
            # combine
            _g0 = _xp.asarray(g_TEND0)
            g_TEND[:, :, :, :, I_RHOG]   = bk.to_numpy(_g0[:, :, :, :, I_RHOG] + _drhog)
            g_TEND[:, :, :, :, I_RHOGVX] = bk.to_numpy(_g0[:, :, :, :, I_RHOGVX] - _dpg[:, :, :, :, XDIR] + _xp.asarray(ddivdvx) + _xp.asarray(ddivdvx_2d))
            g_TEND[:, :, :, :, I_RHOGVY] = bk.to_numpy(_g0[:, :, :, :, I_RHOGVY] - _dpg[:, :, :, :, YDIR] + _xp.asarray(ddivdvy) + _xp.asarray(ddivdvy_2d))
            g_TEND[:, :, :, :, I_RHOGVZ] = bk.to_numpy(_g0[:, :, :, :, I_RHOGVZ] - _dpg[:, :, :, :, ZDIR] + _xp.asarray(ddivdvz) + _xp.asarray(ddivdvz_2d))
            g_TEND[:, :, :, :, I_RHOGW]  = bk.to_numpy(_g0[:, :, :, :, I_RHOGW] + _xp.asarray(ddivdw) * alpha - _dpgw + _dbuo)
            g_TEND[:, :, :, :, I_RHOGE]  = bk.to_numpy(_g0[:, :, :, :, I_RHOGE] + _drhoge + _pw)
            gz_tilde[:, :, :, :] = bk.to_numpy(_gz)   # rhow_matrix consumes gz_tilde (numpy skipped)
            if adm.ADM_have_pl:
                _C2Wp = _xp.asarray(vmtr.VMTR_C2Wfact_pl)
                _W2Cp = _xp.asarray(vmtr.VMTR_W2Cfact_pl)
                _PROGdp = _xp.asarray(PROG_pl)
                _rhoghp = _xp.full(adm.ADM_shape_pl, _UNDEF, dtype=rdtype)
                _rhoghp = _rhoghp.at[:, _ks, :].set(
                    _C2Wp[:, _ks, :, 0] * _PROGdp[:, _ks, :, I_RHOG]
                    + _C2Wp[:, _ks, :, 1] * _PROGdp[:, _ksm, :, I_RHOG])
                _rhoghp = _rhoghp.at[:, kmin - 1, :].set(_rhoghp[:, kmin, :])
                _gzp  = GRAV - (_dpgw_pl - _dbuo_pl) / _rhoghp
                _pwhp = -_gzp * _PROGdp[:, :, :, I_RHOGW]
                _vxp = _xp.asarray(vx_pl); _vyp = _xp.asarray(vy_pl); _vzp = _xp.asarray(vz_pl)
                _pwp = _xp.full(adm.ADM_shape_pl, _UNDEF, dtype=rdtype)
                _pwp = _pwp.at[:, _kc, :].set(
                    _vxp[:, _kc, :] * _dpg_pl[:, _kc, :, XDIR]
                    + _vyp[:, _kc, :] * _dpg_pl[:, _kc, :, YDIR]
                    + _vzp[:, _kc, :] * _dpg_pl[:, _kc, :, ZDIR]
                    + _W2Cp[:, _kc, :, 0] * _pwhp[:, _kp1, :]
                    + _W2Cp[:, _kc, :, 1] * _pwhp[:, _kc, :])
                _pwp = _pwp.at[:, kmin - 1, :].set(rdtype(0.0))
                _pwp = _pwp.at[:, kmax + 1, :].set(rdtype(0.0))
                _g0p = _xp.asarray(g_TEND0_pl)
                g_TEND_pl[:, :, :, I_RHOG]   = bk.to_numpy(_g0p[:, :, :, I_RHOG] + _drhog_pl)
                g_TEND_pl[:, :, :, I_RHOGVX] = bk.to_numpy(_g0p[:, :, :, I_RHOGVX] - _dpg_pl[:, :, :, XDIR] + _xp.asarray(ddivdvx_pl) + _xp.asarray(ddivdvx_2d_pl))
                g_TEND_pl[:, :, :, I_RHOGVY] = bk.to_numpy(_g0p[:, :, :, I_RHOGVY] - _dpg_pl[:, :, :, YDIR] + _xp.asarray(ddivdvy_pl) + _xp.asarray(ddivdvy_2d_pl))
                g_TEND_pl[:, :, :, I_RHOGVZ] = bk.to_numpy(_g0p[:, :, :, I_RHOGVZ] - _dpg_pl[:, :, :, ZDIR] + _xp.asarray(ddivdvz_pl) + _xp.asarray(ddivdvz_2d_pl))
                g_TEND_pl[:, :, :, I_RHOGW]  = bk.to_numpy(_g0p[:, :, :, I_RHOGW] + _xp.asarray(ddivdw_pl) * alpha - _dpgw_pl + _dbuo_pl)
                g_TEND_pl[:, :, :, I_RHOGE]  = bk.to_numpy(_g0p[:, :, :, I_RHOGE] + _drhoge_pl + _pwp)
                gz_tilde_pl[:, :, :] = bk.to_numpy(_gzp)

        prf.PROF_rapend  ('_____vp0_tendsum',2)
        prf.PROF_rapstart('_____vp0_meanflux',2)   # mean mass-flux init
        # initialization of mean mass flux

        rweight_itr = rdtype(1.0) / rdtype(num_of_itr)
                                # 0 :  5     + 1  # includes I_RHOG (0) to I_RHOGW (5)
        PROG_mean[:, :, :, :, I_RHOG:I_RHOGW + 1] = PROG[:, :, :, :, I_RHOG:I_RHOGW + 1]
        PROG_mean_pl[:, :, :, I_RHOG:I_RHOGW + 1] = PROG_pl[:, :, :, I_RHOG:I_RHOGW + 1]


        # update working matrix for vertical implicit solver
        self.vi_rhow_update_matrix( 
            eth_h   [:,:,:,:], eth_h_pl   [:,:,:], # [IN]
            gz_tilde[:,:,:,:], gz_tilde_pl[:,:,:], # [IN]    
            dt,                                    # [IN]
            cnst, grd, vmtr, rcnf, rdtype,
        )


        prf.PROF_rapend  ('_____vp0_meanflux',2)
        prf.PROF_rapend  ('____vi_path0',2)
        # --- Phase 3a: hoist vi_main's loop-invariant inputs to device-resident
        # handles built ONCE here, before the small-step loop. They are passed to
        # vi_main each iteration where its xp.asarray(...) becomes a no-op (the
        # array is already on-device under jax), so the (num_of_itr-1) redundant
        # host->device transfers are eliminated. PROG, eth and grhogetot0 are not
        # mutated inside the ns loop; Mc/Mu/Ml are rebuilt only once per large
        # step by vi_rhow_update_matrix above. On the numpy backend xp.asarray
        # returns the same (read-only) array, so this is bit-for-bit identical.
        xp = bk.xp
        _rhog0_d      = xp.asarray(PROG[:, :, :, :, I_RHOG])
        _rhogvx0_d    = xp.asarray(PROG[:, :, :, :, I_RHOGVX])
        _rhogvy0_d    = xp.asarray(PROG[:, :, :, :, I_RHOGVY])
        _rhogvz0_d    = xp.asarray(PROG[:, :, :, :, I_RHOGVZ])
        _rhogw0_d     = xp.asarray(PROG[:, :, :, :, I_RHOGW])
        _eth0_d       = xp.asarray(eth)
        _grhogetot0_d = xp.asarray(grhogetot0)
        _rhog0_pl_d      = xp.asarray(PROG_pl[:, :, :, I_RHOG])
        _rhogvx0_pl_d    = xp.asarray(PROG_pl[:, :, :, I_RHOGVX])
        _rhogvy0_pl_d    = xp.asarray(PROG_pl[:, :, :, I_RHOGVY])
        _rhogvz0_pl_d    = xp.asarray(PROG_pl[:, :, :, I_RHOGVZ])
        _rhogw0_pl_d     = xp.asarray(PROG_pl[:, :, :, I_RHOGW])
        _eth0_pl_d       = xp.asarray(eth_pl)
        _grhogetot0_pl_d = xp.asarray(grhogetot0_pl)
        # matrix coefficients (loop-invariant within the small step)
        self._Mc_d = xp.asarray(self.Mc); self._Mu_d = xp.asarray(self.Mu); self._Ml_d = xp.asarray(self.Ml)
        self._Mc_pl_d = xp.asarray(self.Mc_pl); self._Mu_pl_d = xp.asarray(self.Mu_pl); self._Ml_pl_d = xp.asarray(self.Ml_pl)

        # Phase 3 (Option 1): keep the hot segment vipath1 -> COMM(diff_vh) ->
        # vi_main -> COMM(diff_we) -> vipath2c device-resident (diff_vh/diff_we as
        # jax arrays, on-device COMM between) so no to_numpy/asarray drains the
        # async GPU pipeline mid-segment. PROG_split stays numpy at the loop edges.
        # jax-only; gated behind PYNICAM_RESIDENT_VISEG (default off). Bit-exact vs
        # the non-resident jax path (removing to_numpy;asarray is an exact identity,
        # and on-device COMM is bit-exact vs numpy COMM).
        resident_seg = (bk.type == "jax") and getattr(
            self, "use_resident_viseg",
            os.environ.get("PYNICAM_RESIDENT_VISEG", "0") != "0")

        # Option 3 step-1: device-resident ns-loop carry. PROG_split/PROG_mean stay
        # jax across iterations (vi_path2c returns jax, no per-iter D2H); drained after.
        if resident_seg:
            xp = bk.xp
            PROG_split_d    = xp.asarray(PROG_split)
            PROG_mean_d     = xp.asarray(PROG_mean)
            PROG_split_pl_d = xp.asarray(PROG_split_pl)
            PROG_mean_pl_d  = xp.asarray(PROG_mean_pl)

            # Option-3 step-4c: bundle the LARGE-STEP-VARYING device state into a
            # single pytree `_inv` that _ns_body reads from. Passing it as a TRACED
            # arg to the cached-jit ns-loop (below) keeps the XLA signature
            # shape-stable across large steps, so the loop compiles ONCE per trip
            # count and is reused -- instead of step-4b, where these arrays were
            # baked as per-step constants inside the closure and forced a recompile
            # every large step. PROG/g_TEND go to device here (once per large step);
            # Mc/Mu/Ml, eth0, grhogetot0 are already device handles (_*_d above).
            # On the eager (non-fori) path passing these jax arrays is bit-identical
            # to the prior numpy refs (xp.asarray of a jax array is a no-op).
            PROG_d      = xp.asarray(PROG)
            PROG_pl_d   = xp.asarray(PROG_pl)
            g_TEND_d    = xp.asarray(g_TEND)
            g_TEND_pl_d = xp.asarray(g_TEND_pl)
            _inv = (
                PROG_d, PROG_pl_d, g_TEND_d, g_TEND_pl_d,
                self._Mc_d, self._Mu_d, self._Ml_d,
                self._Mc_pl_d, self._Mu_pl_d, self._Ml_pl_d,
                _eth0_d, _eth0_pl_d, _grhogetot0_d, _grhogetot0_pl_d,
            )

        # Option 3 step-4 capability gate: the pure-device fast path (and the
        # forthcoming lax.fori_loop collapse) is only valid when 2D and vertical
        # divdamp are OFF -- then ddivd*_2d and gdvz are identically zero and the
        # body needs no host divdamp compute. When either is ON, the eager
        # resident path below still computes them correctly (numpy numfilter_*),
        # so we fall back to it. (2D divdamp is also non-functional in the port.)
        viseg_pure = (
            resident_seg
            and not numf.NUMFILTER_DOdivdamp_2d
            and not numf.NUMFILTER_DOdivdamp_v
        )
        if resident_seg and not viseg_pure and not getattr(self, "_viseg_pure_warned", False):
            self._viseg_pure_warned = True
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("*** [vi_small_step] resident ns-loop pure/fori_loop path "
                          "disabled: 2D or vertical divdamp enabled "
                          f"(DOdivdamp_2d={numf.NUMFILTER_DOdivdamp_2d}, "
                          f"DOdivdamp_v={numf.NUMFILTER_DOdivdamp_v}); using eager "
                          "resident path.", file=log_file)

        # Option-3 step-4: ONE ns small-step iteration as a PURE carry->carry
        # function. carry = (PROG_split_d, PROG_split_pl_d, PROG_mean_d,
        # PROG_mean_pl_d); everything else is ns-loop-invariant and closed over.
        # No host compute / no prf side-effects -> drivable by the Python loop
        # (step-4a) or by jax.lax.fori_loop (step-4b). Index-independent (ns is
        # never used). Only entered when resident_seg.
        def _ns_body(_carry, _inv):
            PROG_split_d, PROG_split_pl_d, PROG_mean_d, PROG_mean_pl_d = _carry
            # step-4c: the large-step-varying state arrives via _inv (traced under
            # the cached jit), NOT the enclosing closure -> no per-step recompile.
            (PROG_d, PROG_pl_d, g_TEND_d, g_TEND_pl_d,
             _Mc_d, _Mu_d, _Ml_d, _Mc_pl_d, _Mu_pl_d, _Ml_pl_d,
             _eth0_d, _eth0_pl_d, _grhogetot0_d, _grhogetot0_pl_d) = _inv
            xp = bk.xp
            have_pl = adm.ADM_have_pl

            # n-level slices derived from the TRACED PROG_d / PROG_pl_d (cheap
            # slices; these feed vi_main's *0 inputs). Bit-identical to the outer
            # _rhog0_d.. handles (526-539) which the non-resident path still uses.
            _rhog0_d   = PROG_d[:, :, :, :, I_RHOG]
            _rhogvx0_d = PROG_d[:, :, :, :, I_RHOGVX]
            _rhogvy0_d = PROG_d[:, :, :, :, I_RHOGVY]
            _rhogvz0_d = PROG_d[:, :, :, :, I_RHOGVZ]
            _rhogw0_d  = PROG_d[:, :, :, :, I_RHOGW]
            _rhog0_pl_d   = PROG_pl_d[:, :, :, I_RHOG]
            _rhogvx0_pl_d = PROG_pl_d[:, :, :, I_RHOGVX]
            _rhogvy0_pl_d = PROG_pl_d[:, :, :, I_RHOGVY]
            _rhogvz0_pl_d = PROG_pl_d[:, :, :, I_RHOGVZ]
            _rhogw0_pl_d  = PROG_pl_d[:, :, :, I_RHOGW]

            # --- device-resident preg_prim_split from the carry (jax functional) ---
            preg_d = PROG_split_d[:, :, :, :, I_RHOGE] * RovCV
            preg_d = preg_d.at[:, :, kmin - 1, :].set(preg_d[:, :, kmin, :])
            preg_d = preg_d.at[:, :, kmax + 1, :].set(preg_d[:, :, kmax, :])
            PROG_split_d = PROG_split_d.at[:, :, kmin - 1, :, I_RHOGE].set(PROG_split_d[:, :, kmin, :, I_RHOGE])
            PROG_split_d = PROG_split_d.at[:, :, kmax + 1, :, I_RHOGE].set(PROG_split_d[:, :, kmax, :, I_RHOGE])
            if have_pl:
                preg_pl_d = PROG_split_pl_d[:, :, :, I_RHOGE] * RovCV
                preg_pl_d = preg_pl_d.at[:, kmin - 1, :].set(preg_pl_d[:, kmin, :])
                preg_pl_d = preg_pl_d.at[:, kmax + 1, :].set(preg_pl_d[:, kmax, :])
                PROG_split_pl_d = PROG_split_pl_d.at[:, kmin - 1, :, I_RHOGE].set(PROG_split_pl_d[:, kmin, :, I_RHOGE])
                PROG_split_pl_d = PROG_split_pl_d.at[:, kmax + 1, :, I_RHOGE].set(PROG_split_pl_d[:, kmax, :, I_RHOGE])
            else:
                preg_pl_d = preg_prim_split_pl

            # --- divdamp horizontal from the device carry: returns jax ddivd* ---
            _ddx_d = xp.asarray(ddivdvx); _ddxp_d = xp.asarray(ddivdvx_pl)
            _ddy_d = xp.asarray(ddivdvy); _ddyp_d = xp.asarray(ddivdvy_pl)
            _ddz_d = xp.asarray(ddivdvz); _ddzp_d = xp.asarray(ddivdvz_pl)
            _ddw_d = xp.asarray(ddivdw);  _ddwp_d = xp.asarray(ddivdw_pl)
            if tim.TIME_split:
                (_ddx_d, _ddy_d, _ddz_d, _ddxp_d, _ddyp_d, _ddzp_d,
                 _ddw_d, _ddwp_d) = numf.numfilter_divdamp(
                    PROG_split_d[:,:,:,:,I_RHOGVX], PROG_split_pl_d[:,:,:,I_RHOGVX],
                    PROG_split_d[:,:,:,:,I_RHOGVY], PROG_split_pl_d[:,:,:,I_RHOGVY],
                    PROG_split_d[:,:,:,:,I_RHOGVZ], PROG_split_pl_d[:,:,:,I_RHOGVZ],
                    PROG_split_d[:,:,:,:,I_RHOGW ], PROG_split_pl_d[:,:,:,I_RHOGW ],
                    ddivdvx, ddivdvx_pl, ddivdvy, ddivdvy_pl,
                    ddivdvz, ddivdvz_pl, ddivdw,  ddivdw_pl,
                    cnst, comm, grd, oprt, vmtr, src, rdtype, resident=True,
                )
                if not viseg_pure:
                    numf.numfilter_divdamp_2d(
                        PROG_split_d[:,:,:,:,I_RHOGVX], PROG_split_pl_d[:,:,:,I_RHOGVX],
                        PROG_split_d[:,:,:,:,I_RHOGVY], PROG_split_pl_d[:,:,:,I_RHOGVY],
                        PROG_split_d[:,:,:,:,I_RHOGVZ], PROG_split_pl_d[:,:,:,I_RHOGVZ],
                        ddivdvx_2d, ddivdvx_2d_pl, ddivdvy_2d, ddivdvy_2d_pl,
                        ddivdvz_2d, ddivdvz_2d_pl,
                        cnst, comm, grd, oprt, rdtype,
                    )

            # 2D divdamp: identically zero in the supported (viseg_pure) config
            # -> device zeros (no host work); else use the numpy arrays above.
            if viseg_pure:
                _z2  = xp.zeros_like(_ddx_d);   _z2p = xp.zeros_like(_ddxp_d)
                _dd2x_d, _dd2y_d, _dd2z_d    = _z2, _z2, _z2
                _dd2xp_d, _dd2yp_d, _dd2zp_d = _z2p, _z2p, _z2p
            else:
                _dd2x_d, _dd2y_d, _dd2z_d    = ddivdvx_2d, ddivdvy_2d, ddivdvz_2d
                _dd2xp_d, _dd2yp_d, _dd2zp_d = ddivdvx_2d_pl, ddivdvy_2d_pl, ddivdvz_2d_pl

            # --- B1: vipath1 -> diff_vh, drhogw kept on device (jax) ---
            o1 = self._vi_path1_fused(
                PROG_d, PROG_pl_d, PROG_split_d, PROG_split_pl_d,
                preg_d, preg_pl_d,
                g_TEND_d, g_TEND_pl_d,
                _ddx_d, _ddy_d, _ddz_d, _ddw_d,
                _ddxp_d, _ddyp_d, _ddzp_d, _ddwp_d,
                _dd2x_d, _dd2y_d, _dd2z_d,
                _dd2xp_d, _dd2yp_d, _dd2zp_d,
                diff_vh, diff_vh_pl, drhogw, drhogw_pl,
                dt, I_RHOG, I_RHOGVX, I_RHOGVY, I_RHOGVZ, I_RHOGW,
                XDIR, YDIR, ZDIR, alpha,
                cnst, grd, oprt, vmtr, src, bndc, tim, rcnf, rdtype,
                resident=True,
            )
            dvh_d = o1["diff_vh"]
            drhogw_d = o1["drhogw"]
            dvh_pl_d = o1["diff_vh_pl"] if have_pl else xp.asarray(diff_vh_pl)
            drhogw_pl_d = o1["drhogw_pl"] if have_pl else xp.asarray(drhogw_pl)

            # --- COMM(diff_vh) on device (jax in -> jax out) ---
            dvh_d, dvh_pl_d = comm.COMM_data_transfer(dvh_d, dvh_pl_d)

            # --- vertical implicit (vi_main) -> diff_we kept on device ---
            o2 = self.vi_main(
                diff_we[:,:,:,:,0], diff_we_pl[:,:,:,0],
                diff_we[:,:,:,:,1], diff_we_pl[:,:,:,1],
                diff_we[:,:,:,:,2], diff_we_pl[:,:,:,2],
                dvh_d[:,:,:,:,0], dvh_pl_d[:,:,:,0],
                dvh_d[:,:,:,:,1], dvh_pl_d[:,:,:,1],
                dvh_d[:,:,:,:,2], dvh_pl_d[:,:,:,2],
                PROG_split_d[:,:,:,:,I_RHOG],   PROG_split_pl_d[:,:,:,I_RHOG],
                PROG_split_d[:,:,:,:,I_RHOGVX], PROG_split_pl_d[:,:,:,I_RHOGVX],
                PROG_split_d[:,:,:,:,I_RHOGVY], PROG_split_pl_d[:,:,:,I_RHOGVY],
                PROG_split_d[:,:,:,:,I_RHOGVZ], PROG_split_pl_d[:,:,:,I_RHOGVZ],
                PROG_split_d[:,:,:,:,I_RHOGW],  PROG_split_pl_d[:,:,:,I_RHOGW],
                PROG_split_d[:,:,:,:,I_RHOGE],  PROG_split_pl_d[:,:,:,I_RHOGE],
                preg_d[:,:,:,:],     preg_pl_d[:,:,:],
                _rhog0_d,    _rhog0_pl_d,
                _rhogvx0_d,  _rhogvx0_pl_d,
                _rhogvy0_d,  _rhogvy0_pl_d,
                _rhogvz0_d,  _rhogvz0_pl_d,
                _rhogw0_d,   _rhogw0_pl_d,
                _eth0_d,     _eth0_pl_d,
                g_TEND_d[:,:,:,:,I_RHOG],  g_TEND_pl_d[:,:,:,I_RHOG],
                drhogw_d,    drhogw_pl_d,
                g_TEND_d[:,:,:,:,I_RHOGE], g_TEND_pl_d[:,:,:,I_RHOGE],
                _grhogetot0_d, _grhogetot0_pl_d,
                dt,
                rcnf, cnst, vmtr, tim, grd, oprt, bndc, cnvv, src, rdtype,
                resident=True,
                Mc_d=_Mc_d, Mu_d=_Mu_d, Ml_d=_Ml_d,
                Mc_pl_d=_Mc_pl_d, Mu_pl_d=_Mu_pl_d, Ml_pl_d=_Ml_pl_d,
            )
            dwe_d = xp.stack([o2["rhog_split1"], o2["rhogw_split1"], o2["rhoge_split1"]], axis=-1)
            if have_pl:
                dwe_pl_d = xp.stack([o2["rhog_split1_pl"], o2["rhogw_split1_pl"], o2["rhoge_split1_pl"]], axis=-1)
            else:
                dwe_pl_d = xp.asarray(diff_we_pl)

            # --- COMM(diff_we) on device ---
            dwe_d, dwe_pl_d = comm.COMM_data_transfer(dwe_d, dwe_pl_d)

            # --- C2: vipath2c (jax in) -> new (PROG_split, PROG_mean) carry ---
            _o2c = self._vi_path2c_fused(
                PROG_split_d, PROG_split_pl_d, PROG_mean_d, PROG_mean_pl_d,
                dvh_d, dvh_pl_d, dwe_d, dwe_pl_d,
                rweight_itr,
                I_RHOG, I_RHOGVX, I_RHOGVY, I_RHOGVZ, I_RHOGW, I_RHOGE,
                resident=True,
            )
            PROG_split_d = _o2c["PROG_split"]
            PROG_mean_d  = _o2c["PROG_mean"]
            if have_pl:
                PROG_split_pl_d = _o2c["PROG_split_pl"]
                PROG_mean_pl_d  = _o2c["PROG_mean_pl"]
            return (PROG_split_d, PROG_split_pl_d, PROG_mean_d, PROG_mean_pl_d)

        if resident_seg:
            _carry = (PROG_split_d, PROG_split_pl_d, PROG_mean_d, PROG_mean_pl_d)

        # Option-3 step-4b: collapse the Python `for ns` per-iteration dispatch
        # into ONE compiled graph via jax.lax.fori_loop -- the lever for the
        # per-iter host-dispatch floor (~18k host-func calls). Only when
        # viseg_pure (the body is pure-device) AND opt-in PYNICAM_RESIDENT_FORILOOP.
        # mpi4jax sendrecv inside fori_loop is validated bit-exact across 4 ranks
        # (env_check/foriloop_comm_probe). When it fires, the Python loop below
        # runs 0 iters (the carry is already advanced); otherwise the eager Python
        # loop (step-4a) runs unchanged. Default off -> no behavior change.
        _use_foriloop = (
            resident_seg and viseg_pure
            and os.environ.get("PYNICAM_RESIDENT_FORILOOP", "0") != "0"
        )
        if _use_foriloop:
            prf.PROF_rapstart('____vi_seg_foriloop', 2)
            # Warm up all LAZY device-side caches (bk.device_consts dicts, the
            # bk.maybe_jit sub-kernels, the on-device COMM plans/jit fns) with ONE
            # EAGER iteration BEFORE tracing. Those caches populate on first use;
            # if that first use happens inside the fori_loop trace, the trace-time
            # arrays get stored into instance state (self._dev_cache, ...) and
            # escape the loop scope -> jax UnexpectedTracerError. The warm-up runs
            # the body once eagerly (advancing the carry by 1) so the fori_loop
            # trace hits populated caches and mutates no instance state; it then
            # runs the remaining num_of_itr-1 iterations. Bit-exact (same total).
            _carry = _ns_body(_carry, _inv)

            # Option-3 step-4c: COMPILE the fori_loop ONCE per distinct trip count
            # and REUSE it across large steps. The large-step-varying state flows in
            # via the traced `_inv` pytree (see its construction above), so the XLA
            # signature is shape-stable -> cache hit on every later large step. This
            # fixes step-4b, where _inv was baked as closure constants and the whole
            # loop recompiled each large step (the perf regression). num_of_itr is a
            # STATIC trip count drawn from a small fixed set (num_of_iteration_sstep,
            # 3-4 distinct values) -> at most that many compiles total. The cached fn
            # captures the first call's _ns_body, whose remaining free vars are all
            # run-constant (dt/alpha/RovCV/kmin/kmax/I_*/viseg_pure/module refs, and
            # rweight_itr = 1/num_of_itr which is constant for a given key) -> reuse
            # is correct. Must NOT jit a freshly-defined closure each call (new fn
            # object -> recompile every call); hence the per-self cache keyed by N.
            if not hasattr(self, "_ns_loop_jit_cache"):
                self._ns_loop_jit_cache = {}
            _N = int(num_of_itr)   # np.int64 -> python int (stable static bound/key)
            _loop_fn = self._ns_loop_jit_cache.get(_N)
            if _loop_fn is None:
                def _make_ns_loop(_n, _body):
                    def _ns_loop(carry, inv):
                        return bk.jax.lax.fori_loop(
                            0, _n - 1, lambda _i, _c: _body(_c, inv), carry)
                    return bk.jax.jit(_ns_loop)
                _loop_fn = _make_ns_loop(_N, _ns_body)
                self._ns_loop_jit_cache[_N] = _loop_fn
            _carry = _loop_fn(_carry, _inv)
            prf.PROF_rapend('____vi_seg_foriloop', 2)

        #---------------------------------------------------------------------------
        #
        #> Start small step iteration
        #
        #---------------------------------------------------------------------------
        for ns in range(0 if _use_foriloop else num_of_itr):
        #for ns in range(num_of_itr + 1):

            prf.PROF_rapstart('____vi_path1',2)

            # with open (std.fname_log, 'a') as log_file:
            #     print("NNNs num_of_itr ", num_of_itr, file=log_file)
            #     print("ns ", ns, file=log_file)
                
            #---< calculation of preg_prim(*) from rhog(*) & rhoge(*) >

            # Option 3 step-3: in resident mode this numpy preg_prim_split / PROG_split
            # recompute is DEAD WORK — preg is rebuilt on-device as preg_d from
            # PROG_split_d (below), and numpy PROG_split is overwritten by the post-loop
            # drain. Skip it so the resident ns-loop body carries no host compute here.
            if not resident_seg:
                # Main part: compute preg_prim_split for all k and l
                preg_prim_split[:, :, :, :] = PROG_split[:, :, :, :, I_RHOGE] * RovCV

                # Boundary copy (along k axis)
                preg_prim_split[:, :, kmin - 1, :] = preg_prim_split[:, :, kmin, :]
                preg_prim_split[:, :, kmax + 1, :] = preg_prim_split[:, :, kmax, :]

                PROG_split[:, :, kmin - 1, :, I_RHOGE] = PROG_split[:, :, kmin, :, I_RHOGE]
                PROG_split[:, :, kmax + 1, :, I_RHOGE] = PROG_split[:, :, kmax, :, I_RHOGE]

                if adm.ADM_have_pl:
                    preg_prim_split_pl[:, :, :] = PROG_split_pl[:, :, :, I_RHOGE] * RovCV

                    # Ghost layers copy
                    preg_prim_split_pl[:, kmin - 1, :] = preg_prim_split_pl[:, kmin, :]
                    preg_prim_split_pl[:, kmax + 1, :] = preg_prim_split_pl[:, kmax, :]

                    PROG_split_pl[:, kmin - 1, :, I_RHOGE] = PROG_split_pl[:, kmin, :, I_RHOGE]
                    PROG_split_pl[:, kmax + 1, :, I_RHOGE] = PROG_split_pl[:, kmax, :, I_RHOGE]
                #endif
            #endif

            #prc.prc_mpistop(std.io_l, std.fname_log)

            if tim.TIME_split and not resident_seg:   # resident path runs its own divdamp from the device carry
                #---< Calculation of source term for Vh(vx,vy,vz) and W (split) >

                # divergence damping (contains COMM internally)

                numf.numfilter_divdamp(
                    PROG_split[:,:,:,:,I_RHOGVX], PROG_split_pl[:,:,:,I_RHOGVX], # [IN]
                    PROG_split[:,:,:,:,I_RHOGVY], PROG_split_pl[:,:,:,I_RHOGVY], # [IN]
                    PROG_split[:,:,:,:,I_RHOGVZ], PROG_split_pl[:,:,:,I_RHOGVZ], # [IN]
                    PROG_split[:,:,:,:,I_RHOGW ], PROG_split_pl[:,:,:,I_RHOGW ], # [IN]
                    ddivdvx[:,:,:,:],             ddivdvx_pl[:,:,:],             # [OUT]
                    ddivdvy[:,:,:,:],             ddivdvy_pl[:,:,:],             # [OUT]
                    ddivdvz[:,:,:,:],             ddivdvz_pl[:,:,:],             # [OUT]
                    ddivdw [:,:,:,:],             ddivdw_pl [:,:,:],             # [OUT]
                    cnst, comm, grd, oprt, vmtr, src, rdtype,
                )

                # 2d divergence damping (contains COMM internally)
                numf.numfilter_divdamp_2d(
                    PROG_split[:,:,:,:,I_RHOGVX], PROG_split_pl[:,:,:,I_RHOGVX], # [IN]
                    PROG_split[:,:,:,:,I_RHOGVY], PROG_split_pl[:,:,:,I_RHOGVY], # [IN]
                    PROG_split[:,:,:,:,I_RHOGVZ], PROG_split_pl[:,:,:,I_RHOGVZ], # [IN]
                    ddivdvx_2d[:,:,:,:],          ddivdvx_2d_pl[:,:,:],          # [OUT]
                    ddivdvy_2d[:,:,:,:],          ddivdvy_2d_pl[:,:,:],          # [OUT]
                    ddivdvz_2d[:,:,:,:],          ddivdvz_2d_pl[:,:,:],          # [OUT]
                    cnst, comm, grd, oprt, rdtype,
                )
            #endif

            # --- COMM-free "B1" island: presgrad + tendency + diff_vh + BNDCND ---
            # Fused into one backend-switchable pure function by default
            # (kernels/vipath1.py): under jax.jit XLA emits one graph with a single
            # host round-trip instead of three (presgrad / glue / bndcnd). Set
            # self.use_fused_vipath1 = False for the original per-kernel path.
            prf.PROF_rapstart('____vi_path1_fused', 2)
            if resident_seg:
                # Option-3 step-4a: one ns small-step iteration is now the pure
                # carry->carry _ns_body() (defined before the loop): no host
                # compute, no prf side-effects, so it can be driven by this Python
                # loop now and lifted into jax.lax.fori_loop next (step-4b). The
                # two timers it spans (vi_path1 started at the loop top, vi_path1_
                # fused just above) are balanced here.
                _carry = _ns_body(_carry, _inv)
                prf.PROF_rapend  ('____vi_path1_fused', 2)
                prf.PROF_rapend  ('____vi_path1', 2)
                prf.PROF_rapstart('____vi_path2', 2)
                prf.PROF_rapend  ('____vi_path2', 2)
                continue

            if getattr(self, "use_fused_vipath1", os.environ.get("PYNICAM_FUSE_VIPATH1", "1") != "0"):
                self._vi_path1_fused(
                    PROG, PROG_pl, PROG_split, PROG_split_pl,
                    preg_prim_split, preg_prim_split_pl,
                    g_TEND, g_TEND_pl,
                    ddivdvx, ddivdvy, ddivdvz, ddivdw,
                    ddivdvx_pl, ddivdvy_pl, ddivdvz_pl, ddivdw_pl,
                    ddivdvx_2d, ddivdvy_2d, ddivdvz_2d,
                    ddivdvx_2d_pl, ddivdvy_2d_pl, ddivdvz_2d_pl,
                    diff_vh, diff_vh_pl, drhogw, drhogw_pl,
                    dt, I_RHOG, I_RHOGVX, I_RHOGVY, I_RHOGVZ, I_RHOGW,
                    XDIR, YDIR, ZDIR, alpha,
                    cnst, grd, oprt, vmtr, src, bndc, tim, rcnf, rdtype,
                )
            else:
                if tim.TIME_split:
                    # pressure force
                    # dpgradw=0.0_RP because of f_type='HORIZONTAL'.
                    src.src_pres_gradient(
                        preg_prim_split[:,:,:,:],   preg_prim_split_pl[:,:,:],   # [IN]
                        dpgrad         [:,:,:,:,:], dpgrad_pl         [:,:,:,:], # [OUT]
                        dpgradw        [:,:,:,:],   dpgradw_pl        [:,:,:],   # [OUT] not used
                        src.I_SRC_horizontal,                                    # [IN]
                        cnst, grd, oprt, vmtr, rdtype,
                    )

                    # buoyancy force
                    # not calculated, because this term is implicit.

                    #---< sum of tendencies ( large step + split{ pres-grad + div-damp + div-damp_2d } ) >

                    drhogvx = (
                        g_TEND[:, :, :, :, I_RHOGVX]
                        - dpgrad[:, :, :, :, XDIR]
                        + ddivdvx[:, :, :, :]
                        + ddivdvx_2d[:, :, :, :]
                    )
                    drhogvy = (
                        g_TEND[:, :, :, :, I_RHOGVY]
                        - dpgrad[:, :, :, :, YDIR]
                        + ddivdvy[:, :, :, :]
                        + ddivdvy_2d[:, :, :, :]
                    )
                    drhogvz = (
                        g_TEND[:, :, :, :, I_RHOGVZ]
                        - dpgrad[:, :, :, :, ZDIR]
                        + ddivdvz[:, :, :, :]
                        + ddivdvz_2d[:, :, :, :]
                    )
                    drhogw[:, :, :, :] = g_TEND[:, :, :, :, I_RHOGW] + ddivdw[:, :, :, :] * alpha

                    diff_vh[:, :, :, :, 0] = PROG_split[:, :, :, :, I_RHOGVX] + drhogvx * dt
                    diff_vh[:, :, :, :, 1] = PROG_split[:, :, :, :, I_RHOGVY] + drhogvy * dt
                    diff_vh[:, :, :, :, 2] = PROG_split[:, :, :, :, I_RHOGVZ] + drhogvz * dt


                    if adm.ADM_have_pl:
                        #for l in range(adm.ADM_lall_pl):
                        # Vectorized over g and k
                        drhogvx = (
                            g_TEND_pl[:, :, :, I_RHOGVX]
                            - dpgrad_pl[:, :, :, XDIR]
                            + ddivdvx_pl[:, :, :]
                            + ddivdvx_2d_pl[:, :, :]
                        )
                        drhogvy = (
                            g_TEND_pl[:, :, :, I_RHOGVY]
                            - dpgrad_pl[:, :, :, YDIR]
                            + ddivdvy_pl[:, :, :]
                            + ddivdvy_2d_pl[:, :, :]
                        )
                        drhogvz = (
                            g_TEND_pl[:, :, :, I_RHOGVZ]
                            - dpgrad_pl[:, :, :, ZDIR]
                            + ddivdvz_pl[:, :, :]
                            + ddivdvz_2d_pl[:, :, :]
                        )

                        drhogw_pl[:, :, :] = g_TEND_pl[:, :, :, I_RHOGW] + ddivdw_pl[:, :, :] * alpha

                        diff_vh_pl[:, :, :, 0] = PROG_split_pl[:, :, :, I_RHOGVX] + drhogvx * dt
                        diff_vh_pl[:, :, :, 1] = PROG_split_pl[:, :, :, I_RHOGVY] + drhogvy * dt
                        diff_vh_pl[:, :, :, 2] = PROG_split_pl[:, :, :, I_RHOGVZ] + drhogvz * dt
                        #end l loop
                    #endif

                else: # NO-SPLITING

                    #---< sum of tendencies ( large step ) >

                    drhogvx = g_TEND[:, :, :, :, I_RHOGVX]
                    drhogvy = g_TEND[:, :, :, :, I_RHOGVY]
                    drhogvz = g_TEND[:, :, :, :, I_RHOGVZ]
                    drhogw[:, :, :, :] = g_TEND[:, :, :, :, I_RHOGW]

                    diff_vh[:, :, :, :, 0] = PROG_split[:, :, :, :, I_RHOGVX] + drhogvx * dt
                    diff_vh[:, :, :, :, 1] = PROG_split[:, :, :, :, I_RHOGVY] + drhogvy * dt
                    diff_vh[:, :, :, :, 2] = PROG_split[:, :, :, :, I_RHOGVZ] + drhogvz * dt

                    if adm.ADM_have_pl:
                            # Vectorized across g and k
                        drhogvx = g_TEND_pl[:, :, :, I_RHOGVX]
                        drhogvy = g_TEND_pl[:, :, :, I_RHOGVY]
                        drhogvz = g_TEND_pl[:, :, :, I_RHOGVZ]
                        drhogw_pl[:, :, :] = g_TEND_pl[:, :, :, I_RHOGW]

                        diff_vh_pl[:, :, :, 0] = PROG_split_pl[:, :, :, I_RHOGVX] + drhogvx * dt
                        diff_vh_pl[:, :, :, 1] = PROG_split_pl[:, :, :, I_RHOGVY] + drhogvy * dt
                        diff_vh_pl[:, :, :, 2] = PROG_split_pl[:, :, :, I_RHOGVZ] + drhogvz * dt
                    #endif

                #endif    Split/Non-split

                # treatment for boundary condition
                bndc.BNDCND_rhovxvyvz(
                    kmin, kmax,
                    PROG   [:,:,:,:,I_RHOG], # [IN]
                    diff_vh[:,:,:,:,0],      # [INOUT]
                    diff_vh[:,:,:,:,1],      # [INOUT]
                    diff_vh[:,:,:,:,2],      # [INOUT]
                    cnst, rdtype,
                )

                if adm.ADM_have_pl:
                    bndc.BNDCND_rhovxvyvz_pl(
                        kmin, kmax,
                        PROG_pl   [:,:,:,I_RHOG], # [IN]
                        diff_vh_pl[:,:,:,0],      # [INOUT]
                        diff_vh_pl[:,:,:,1],      # [INOUT]
                        diff_vh_pl[:,:,:,2],      # [INOUT]
                        cnst, rdtype,
                    )
                #endif
            #endif  fused / original B1
            prf.PROF_rapend  ('____vi_path1_fused', 2)

            comm.COMM_data_transfer( diff_vh, diff_vh_pl )

            prf.PROF_rapend  ('____vi_path1',2)
            prf.PROF_rapstart('____vi_path2',2)


            #---< vertical implicit scheme >
            self.vi_main(
                diff_we        [:,:,:,:,0],        diff_we_pl        [:,:,:,0],        # [OUT]    # g
                diff_we        [:,:,:,:,1],        diff_we_pl        [:,:,:,1],        # [OUT]    # gw
                diff_we        [:,:,:,:,2],        diff_we_pl        [:,:,:,2],        # [OUT]    # ge
                diff_vh        [:,:,:,:,0],        diff_vh_pl        [:,:,:,0],        # [IN]    #
                diff_vh        [:,:,:,:,1],        diff_vh_pl        [:,:,:,1],        # [IN]    #
                diff_vh        [:,:,:,:,2],        diff_vh_pl        [:,:,:,2],        # [IN]    #   
                PROG_split     [:,:,:,:,I_RHOG],   PROG_split_pl     [:,:,:,I_RHOG],   # [IN]
                PROG_split     [:,:,:,:,I_RHOGVX], PROG_split_pl     [:,:,:,I_RHOGVX], # [IN]
                PROG_split     [:,:,:,:,I_RHOGVY], PROG_split_pl     [:,:,:,I_RHOGVY], # [IN]
                PROG_split     [:,:,:,:,I_RHOGVZ], PROG_split_pl     [:,:,:,I_RHOGVZ], # [IN]
                PROG_split     [:,:,:,:,I_RHOGW],  PROG_split_pl     [:,:,:,I_RHOGW],  # [IN]
                PROG_split     [:,:,:,:,I_RHOGE],  PROG_split_pl     [:,:,:,I_RHOGE],  # [IN]
                preg_prim_split[:,:,:,:],          preg_prim_split_pl[:,:,:],          # [IN]
                _rhog0_d,                          _rhog0_pl_d,                        # [IN] resident
                _rhogvx0_d,                        _rhogvx0_pl_d,                      # [IN] resident
                _rhogvy0_d,                        _rhogvy0_pl_d,                      # [IN] resident
                _rhogvz0_d,                        _rhogvz0_pl_d,                      # [IN] resident
                _rhogw0_d,                         _rhogw0_pl_d,                       # [IN] resident
                _eth0_d,                           _eth0_pl_d,                         # [IN] resident
                g_TEND         [:,:,:,:,I_RHOG],   g_TEND_pl         [:,:,:,I_RHOG],   # [IN]
                drhogw         [:,:,:,:],          drhogw_pl         [:,:,:],          # [IN]
                g_TEND         [:,:,:,:,I_RHOGE],  g_TEND_pl         [:,:,:,I_RHOGE],  # [IN]
                _grhogetot0_d,                     _grhogetot0_pl_d,                   # [IN] resident
                dt,                                                                    # [IN]
                rcnf, cnst, vmtr, tim, grd, oprt, bndc, cnvv, src, rdtype, 
            )



            # treatment for boundary condition   # Halo values before this point should not be used.
            comm.COMM_data_transfer( diff_we, diff_we_pl )

            # update split value and mean mass flux  (COMM-free "C2" island)
            # Fused into one backend-switchable pure function when enabled
            # (kernels/vipath2.py). This island is almost pure data movement, so
            # fusing it in isolation does not reduce memory traffic; it is kept
            # OFF by default and pays off only under Win B (device residency).
            # Set self.use_fused_vipath2c = True / env PYNICAM_FUSE_VIPATH2C=1.
            prf.PROF_rapstart('____vi_path2c_fused', 2)
            if getattr(self, "use_fused_vipath2c", os.environ.get("PYNICAM_FUSE_VIPATH2C", "0") != "0"):
                self._vi_path2c_fused(
                    PROG_split, PROG_split_pl, PROG_mean, PROG_mean_pl,
                    diff_vh, diff_vh_pl, diff_we, diff_we_pl,
                    rweight_itr,
                    I_RHOG, I_RHOGVX, I_RHOGVY, I_RHOGVZ, I_RHOGW, I_RHOGE,
                )
            else:
                PROG_split[:, :, :, :, I_RHOGVX] = diff_vh[:, :, :, :, 0]
                PROG_split[:, :, :, :, I_RHOGVY] = diff_vh[:, :, :, :, 1]
                PROG_split[:, :, :, :, I_RHOGVZ] = diff_vh[:, :, :, :, 2]
                PROG_split[:, :, :, :, I_RHOG]   = diff_we[:, :, :, :, 0]
                PROG_split[:, :, :, :, I_RHOGW]  = diff_we[:, :, :, :, 1]
                PROG_split[:, :, :, :, I_RHOGE]  = diff_we[:, :, :, :, 2]


                PROG_mean[:, :, :, :, I_RHOG:I_RHOGW + 1] += PROG_split[:, :, :, :, I_RHOG:I_RHOGW + 1] * rweight_itr

                if adm.ADM_have_pl:
                    PROG_split_pl[:, :, :, I_RHOGVX] = diff_vh_pl[:, :, :, 0]
                    PROG_split_pl[:, :, :, I_RHOGVY] = diff_vh_pl[:, :, :, 1]
                    PROG_split_pl[:, :, :, I_RHOGVZ] = diff_vh_pl[:, :, :, 2]
                    PROG_split_pl[:, :, :, I_RHOG]   = diff_we_pl[:, :, :, 0]
                    PROG_split_pl[:, :, :, I_RHOGW]  = diff_we_pl[:, :, :, 1]
                    PROG_split_pl[:, :, :, I_RHOGE]  = diff_we_pl[:, :, :, 2]

                    PROG_mean_pl[:, :, :, I_RHOG:I_RHOGW + 1] += (
                        PROG_split_pl[:, :, :, I_RHOG:I_RHOGW + 1] * rweight_itr
                    )
                #endif
            #endif  fused / original C2
            prf.PROF_rapend  ('____vi_path2c_fused', 2)

            prf.PROF_rapend  ('____vi_path2',2)

        #end ns loop  # small step end

        #---------------------------------------------------------------------------
        #
        #
        #
        #---------------------------------------------------------------------------
        prf.PROF_rapstart('____vi_path3',2)

        # update prognostic variables

        # Option 3 step-1: drain the device-resident ns-loop carry back to numpy
        if resident_seg:
            PROG_split_d, PROG_split_pl_d, PROG_mean_d, PROG_mean_pl_d = _carry
            PROG_split[:, :, :, :, :] = bk.to_numpy(PROG_split_d)
            PROG_mean[:, :, :, :, :]  = bk.to_numpy(PROG_mean_d)
            if adm.ADM_have_pl:
                PROG_split_pl[:, :, :, :] = bk.to_numpy(PROG_split_pl_d)
                PROG_mean_pl[:, :, :, :]  = bk.to_numpy(PROG_mean_pl_d)

        PROG[:, :, :, :, I_RHOG:I_RHOGE + 1] += PROG_split[:, :, :, :, I_RHOG:I_RHOGE + 1]

        if adm.ADM_have_pl:
            PROG_pl[:,:,:,:] += PROG_split_pl[:,:,:,:]
        #endif

        oprt.OPRT_horizontalize_vec( 
            PROG[:,:,:,:,I_RHOGVX], PROG_pl[:,:,:,I_RHOGVX], # [INOUT]
            PROG[:,:,:,:,I_RHOGVY], PROG_pl[:,:,:,I_RHOGVY], # [INOUT]
            PROG[:,:,:,:,I_RHOGVZ], PROG_pl[:,:,:,I_RHOGVZ], # [INOUT]
            grd, rdtype,
        )
        
        # communication of mean velocity
        comm.COMM_data_transfer( PROG_mean, PROG_mean_pl )

        prf.PROF_rapend  ('____vi_path3',2)

        return

    def _vi_path1_fused(self,
        PROG, PROG_pl, PROG_split, PROG_split_pl,
        preg_prim_split, preg_prim_split_pl,
        g_TEND, g_TEND_pl,
        ddivdvx, ddivdvy, ddivdvz, ddivdw,
        ddivdvx_pl, ddivdvy_pl, ddivdvz_pl, ddivdw_pl,
        ddivdvx_2d, ddivdvy_2d, ddivdvz_2d,
        ddivdvx_2d_pl, ddivdvy_2d_pl, ddivdvz_2d_pl,
        diff_vh, diff_vh_pl, drhogw, drhogw_pl,
        dt, I_RHOG, I_RHOGVX, I_RHOGVY, I_RHOGVZ, I_RHOGW,
        XDIR, YDIR, ZDIR, alpha,
        cnst, grd, oprt, vmtr, src, bndc, tim, rcnf, rdtype,
        resident=False,
    ):
        # ---------------------------------------------------------------
        # FUSED comm-free "B1" island (numpy<->jax): src_pres_gradient +
        # tendency assembly + diff_vh + BNDCND_rhovxvyvz evaluated inside one
        # pure function so that, under jax.jit, XLA fuses the whole sequence
        # into one graph with a single host round-trip. Under numpy it is
        # bit-for-bit identical to the per-kernel path. See kernels/vipath1.py.
        #
        # The pressure-gradient cfg/constants are reused from src (path0 runs
        # src_pres_gradient once before this loop, so they are already built),
        # which guarantees the gradient math is identical to the standalone use.
        # ---------------------------------------------------------------
        xp = bk.xp
        have_pl = adm.ADM_have_pl

        if getattr(self, "_vipath1_kernel", None) is None:
            self._vipath1_cfg = ViPath1Cfg(
                kmin=adm.ADM_kmin, kmax=adm.ADM_kmax, have_pl=have_pl,
                TIME_split=bool(tim.TIME_split),
                alpha=float(rcnf.NON_HYDRO_ALPHA),
                XDIR=XDIR, YDIR=YDIR, ZDIR=ZDIR,
                I_RHOGVX=I_RHOGVX, I_RHOGVY=I_RHOGVY, I_RHOGVZ=I_RHOGVZ, I_RHOGW=I_RHOGW,
                gradtype=src.I_SRC_horizontal,
                presgrad=src._presgrad_cfg,
                bnd=bndc._bnd_cfg_mom(adm.ADM_kmin, adm.ADM_kmax),
            )
            self._vipath1_kernel = bk.maybe_jit(
                compute_vi_path1, static_argnames=("cfg", "xp"),
            )

        # presgrad constants are staged device-resident by src.src_pres_gradient
        # (path0 runs it once before this loop); reuse that same cached dict.
        C = src._dev_cache["presgrad"]
        cfg = self._vipath1_cfg

        P = {
            "preg":      xp.asarray(preg_prim_split),
            "preg_pl":   xp.asarray(preg_prim_split_pl),
            "g_TEND":    xp.asarray(g_TEND),
            "g_TEND_pl": xp.asarray(g_TEND_pl),
            "ddivdvx": xp.asarray(ddivdvx), "ddivdvy": xp.asarray(ddivdvy),
            "ddivdvz": xp.asarray(ddivdvz), "ddivdw":  xp.asarray(ddivdw),
            "ddivdvx_2d": xp.asarray(ddivdvx_2d), "ddivdvy_2d": xp.asarray(ddivdvy_2d),
            "ddivdvz_2d": xp.asarray(ddivdvz_2d),
            "psvx": xp.asarray(PROG_split[:, :, :, :, I_RHOGVX]),
            "psvy": xp.asarray(PROG_split[:, :, :, :, I_RHOGVY]),
            "psvz": xp.asarray(PROG_split[:, :, :, :, I_RHOGVZ]),
            "prog_rhog": xp.asarray(PROG[:, :, :, :, I_RHOG]),
            # pole (always supplied; consumed only when have_pl)
            "ddivdvx_pl": xp.asarray(ddivdvx_pl), "ddivdvy_pl": xp.asarray(ddivdvy_pl),
            "ddivdvz_pl": xp.asarray(ddivdvz_pl), "ddivdw_pl": xp.asarray(ddivdw_pl),
            "ddivdvx_2d_pl": xp.asarray(ddivdvx_2d_pl), "ddivdvy_2d_pl": xp.asarray(ddivdvy_2d_pl),
            "ddivdvz_2d_pl": xp.asarray(ddivdvz_2d_pl),
            "psvx_pl": xp.asarray(PROG_split_pl[:, :, :, I_RHOGVX]),
            "psvy_pl": xp.asarray(PROG_split_pl[:, :, :, I_RHOGVY]),
            "psvz_pl": xp.asarray(PROG_split_pl[:, :, :, I_RHOGVZ]),
            "prog_rhog_pl": xp.asarray(PROG_pl[:, :, :, I_RHOG]),
        }

        out = self._vipath1_kernel(P, C, dt, cfg=cfg, xp=xp)

        # Phase 3 (device-resident segment): keep outputs on device; the caller
        # threads them (jax) into on-device COMM + vi_main without a host drain.
        if resident:
            return out

        diff_vh[:, :, :, :, :] = bk.to_numpy(out["diff_vh"])
        drhogw[:, :, :, :]     = bk.to_numpy(out["drhogw"])
        if have_pl:
            diff_vh_pl[:, :, :, :] = bk.to_numpy(out["diff_vh_pl"])
            drhogw_pl[:, :, :]     = bk.to_numpy(out["drhogw_pl"])

        return

    def _vi_path2c_fused(self,
        PROG_split, PROG_split_pl, PROG_mean, PROG_mean_pl,
        diff_vh, diff_vh_pl, diff_we, diff_we_pl,
        rweight_itr,
        I_RHOG, I_RHOGVX, I_RHOGVY, I_RHOGVZ, I_RHOGW, I_RHOGE,
        resident=False,
    ):
        # FUSED comm-free "C2" island (numpy<->jax): PROG_split writeback +
        # PROG_mean accumulation in one pure function. See kernels/vipath2.py.
        xp = bk.xp
        have_pl = adm.ADM_have_pl

        if getattr(self, "_vipath2_kernel", None) is None:
            # the kernel builds PROG_split with a fixed stack in prognostic
            # index order; guard against a non-standard reindexing.
            assert (I_RHOG, I_RHOGVX, I_RHOGVY, I_RHOGVZ, I_RHOGW, I_RHOGE) == (0, 1, 2, 3, 4, 5)
            self._vipath2_cfg = ViPath2Cfg(
                have_pl=have_pl,
                I_RHOG=I_RHOG, I_RHOGVX=I_RHOGVX, I_RHOGVY=I_RHOGVY,
                I_RHOGVZ=I_RHOGVZ, I_RHOGW=I_RHOGW, I_RHOGE=I_RHOGE,
            )
            self._vipath2_kernel = bk.maybe_jit(
                compute_vi_path2_update, static_argnames=("cfg", "xp"),
            )

        cfg = self._vipath2_cfg
        P = {
            "diff_vh": xp.asarray(diff_vh), "diff_we": xp.asarray(diff_we),
            "PROG_mean": xp.asarray(PROG_mean),
            "rweight_itr": rweight_itr,
            # pole (always supplied; consumed only when have_pl)
            "diff_vh_pl": xp.asarray(diff_vh_pl), "diff_we_pl": xp.asarray(diff_we_pl),
            "PROG_mean_pl": xp.asarray(PROG_mean_pl),
        }

        out = self._vipath2_kernel(P, None, cfg=cfg, xp=xp)

        if resident:
            # jax dict {PROG_split, PROG_mean, +_pl}: caller threads it as the
            # device-resident ns-loop carry (Option 3 prerequisite). No D2H here.
            return out

        PROG_split[:, :, :, :, :] = bk.to_numpy(out["PROG_split"])
        PROG_mean[:, :, :, :, :]  = bk.to_numpy(out["PROG_mean"])
        if have_pl:
            PROG_split_pl[:, :, :, :] = bk.to_numpy(out["PROG_split_pl"])
            PROG_mean_pl[:, :, :, :]  = bk.to_numpy(out["PROG_mean_pl"])

        return

    #> Update tridiagonal matrix
    def vi_rhow_update_matrix(self,
        eth,     eth_pl,     
        g_tilde, g_tilde_pl, 
        dt,
        cnst, grd, vmtr, rcnf, rdtype,
    ):
            
        #---------------------------------------------------------------------------
        # Original concept
        #
        # A_o(:,:,:) = VMTR_RGSGAM2(:,:,:)
        # A_i(:,:,:) = VMTR_GAM2H(:,:,:) * eth(:,:,:) # [debug] 20120727 H.Yashiro
        # B  (:,:,:) = g_tilde(:,:,:)
        # C_o(:,:,:) = VMTR_RGAM2H (:,:,:) * ( CONST_CVdry / CONST_Rdry * CONST_GRAV )
        # C_i(:,:,:) = 1.0_RP / VMTR_RGAM2H(:,:,:)
        # D  (:,:,:) = CONST_CVdry / CONST_Rdry / ( dt*dt ) / VMTR_RGSQRTH(:,:,:)
        #
        # do k = ADM_kmin+1, ADM_kmax
        #    Mc(:,k,:) = dble(NON_HYDRO_ALPHA) *D(:,k,:)              &
        #              + GRD_rdgzh(k)                                 &
        #              * ( GRD_rdgz (k)   * A_o(:,k  ,:) * A_i(:,k,:) &
        #                + GRD_rdgz (k-1) * A_o(:,k-1,:) * A_i(:,k,:) &
        #                - 0.5_RP * ( GRD_dfact(k) - GRD_cfact(k-1) ) &
        #                * ( B(:,k,:) + C_o(:,k,:) * C_i(:,k,:) )     &
        #                )
        #    Mu(:,k,:) = - GRD_rdgzh(k) * GRD_rdgz(k) * A_o(:,k,:) * A_i(:,k+1,:) &
        #                - GRD_rdgzh(k) * 0.5_RP * GRD_cfact(k)                   &
        #                * ( B(:,k+1,:) + C_o(:,k,:) * C_i(:,k+1,:) )
        #    Ml(:,k,:) = - GRD_rdgzh(k) * GRD_rdgz(k) * A_o(:,k,:) * A_i(:,k-1,:) &
        #                + GRD_rdgzh(k) * 0.5_RP * GRD_dfact(k-1)                 &
        #                * ( B(:,k-1,:) + C_o(:,k,:) * C_i(:,k-1,:) )
        # enddo

        prf.PROF_rapstart('____vi_rhow_update_matrix',2)

        # --- COMM-free tri-diagonal matrix assembly via backend-switchable kernel
        #     (numpy<->jax). See kernels/vimatrix.py. Only the interior rows
        #     k = kmin+1 .. kmax are written; other rows stay at their UNDEF init
        #     and are never read by vi_rhow_solver. ---
        xp = bk.xp
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        ks = slice(kmin + 1, kmax + 1)

        if getattr(self, "_vimatrix_kernels", None) is None:
            self._vimatrix_cfg = ViMatrixCfg(
                kmin=kmin, kmax=kmax,
                have_pl=adm.ADM_have_pl,
                GRAV=float(cnst.CONST_GRAV),
                Rdry=float(cnst.CONST_Rdry),
                CVdry=float(cnst.CONST_CVdry),
                alpha=float(rcnf.NON_HYDRO_ALPHA),
            )
            self._vimatrix_kernels = {
                "reg": bk.maybe_jit(compute_rhow_matrix_reg, static_argnames=("cfg", "xp")),
                "pl":  bk.maybe_jit(compute_rhow_matrix_pl,  static_argnames=("cfg", "xp")),
            }
        d = bk.device_consts(self, "vimatrix", lambda: {
            "RGSQRTH": vmtr.VMTR_RGSQRTH,
            "RGSGAM2": vmtr.VMTR_RGSGAM2,
            "GAM2H":   vmtr.VMTR_GAM2H,
            "RGAMH":   vmtr.VMTR_RGAMH,
            "rdgzh":   grd.GRD_rdgzh,
            "rdgz":    grd.GRD_rdgz,
            "dfact":   grd.GRD_dfact,
            "cfact":   grd.GRD_cfact,
            "RGSQRTH_pl": vmtr.VMTR_RGSQRTH_pl,
            "RGSGAM2_pl": vmtr.VMTR_RGSGAM2_pl,
            "GAM2H_pl":   vmtr.VMTR_GAM2H_pl,
            "RGAMH_pl":   vmtr.VMTR_RGAMH_pl,
        })
        cfg = self._vimatrix_cfg

        _Mc, _Mu, _Ml = self._vimatrix_kernels["reg"](
            xp.asarray(eth), xp.asarray(g_tilde),
            d["RGSQRTH"], d["RGSGAM2"], d["GAM2H"], d["RGAMH"],
            d["rdgzh"], d["rdgz"], d["dfact"], d["cfact"],
            dt, cfg=cfg, xp=xp,
        )
        self.Mc[:, :, ks, :] = bk.to_numpy(_Mc)
        self.Mu[:, :, ks, :] = bk.to_numpy(_Mu)
        self.Ml[:, :, ks, :] = bk.to_numpy(_Ml)

        if adm.ADM_have_pl:
            _Mc_pl, _Mu_pl, _Ml_pl = self._vimatrix_kernels["pl"](
                xp.asarray(eth_pl), xp.asarray(g_tilde_pl),
                d["RGSQRTH_pl"], d["RGSGAM2_pl"], d["GAM2H_pl"], d["RGAMH_pl"],
                d["rdgzh"], d["rdgz"], d["dfact"], d["cfact"],
                dt, cfg=cfg, xp=xp,
            )
            self.Mc_pl[:, ks, :] = bk.to_numpy(_Mc_pl)
            self.Mu_pl[:, ks, :] = bk.to_numpy(_Mu_pl)
            self.Ml_pl[:, ks, :] = bk.to_numpy(_Ml_pl)

        prf.PROF_rapend('____vi_rhow_update_matrix',2)

        return
    
    #> Main part of the vertical implicit scheme
    def vi_main(self,
        rhog_split1,      rhog_split1_pl,      
        rhogw_split1,     rhogw_split1_pl,     
        rhoge_split1,     rhoge_split1_pl,     
        rhogvx_split1,    rhogvx_split1_pl,    
        rhogvy_split1,    rhogvy_split1_pl,    
        rhogvz_split1,    rhogvz_split1_pl,    
        rhog_split0,      rhog_split0_pl,      
        rhogvx_split0,    rhogvx_split0_pl,    
        rhogvy_split0,    rhogvy_split0_pl,    
        rhogvz_split0,    rhogvz_split0_pl,    
        rhogw_split0,     rhogw_split0_pl,     
        rhoge_split0,     rhoge_split0_pl,     
        preg_prim_split0, preg_prim_split0_pl, 
        rhog0,            rhog0_pl,            
        rhogvx0,          rhogvx0_pl,          
        rhogvy0,          rhogvy0_pl,          
        rhogvz0,          rhogvz0_pl,     
        rhogw0,           rhogw0_pl,    
        eth0,             eth0_pl,             
        grhog,            grhog_pl,            
        grhogw,           grhogw_pl,           
        grhoge,           grhoge_pl,           
        grhogetot,        grhogetot_pl,
        dt,
        rcnf, cnst, vmtr, tim, grd, oprt, bndc, cnvv, src, rdtype,
        resident=False,
        Mc_d=None, Mu_d=None, Ml_d=None,
        Mc_pl_d=None, Mu_pl_d=None, Ml_pl_d=None,
    ):

        # vi_main's comm-free core is fused into one pure function by default
        # (kernels/vimain.py). Set self.use_fused_vimain = False to fall back to
        # the original per-kernel path (_vi_main_orig) for A/B timing / debug.
        if not getattr(self, "use_fused_vimain", True):
            return self._vi_main_orig(
                rhog_split1, rhog_split1_pl, rhogw_split1, rhogw_split1_pl,
                rhoge_split1, rhoge_split1_pl, rhogvx_split1, rhogvx_split1_pl,
                rhogvy_split1, rhogvy_split1_pl, rhogvz_split1, rhogvz_split1_pl,
                rhog_split0, rhog_split0_pl, rhogvx_split0, rhogvx_split0_pl,
                rhogvy_split0, rhogvy_split0_pl, rhogvz_split0, rhogvz_split0_pl,
                rhogw_split0, rhogw_split0_pl, rhoge_split0, rhoge_split0_pl,
                preg_prim_split0, preg_prim_split0_pl, rhog0, rhog0_pl,
                rhogvx0, rhogvx0_pl, rhogvy0, rhogvy0_pl, rhogvz0, rhogvz0_pl,
                rhogw0, rhogw0_pl, eth0, eth0_pl, grhog, grhog_pl,
                grhogw, grhogw_pl, grhoge, grhoge_pl, grhogetot, grhogetot_pl,
                dt, rcnf, cnst, vmtr, tim, grd, oprt, bndc, cnvv, src, rdtype,
            )

        prf.PROF_rapstart('____vi_main_fused', 2)

        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax

        # ---------------------------------------------------------------
        # FUSED comm-free core (numpy<->jax). The five sub-kernels that
        # vi_main composes (flux/adv convergence, Thomas solve, rhogw BC,
        # rhogkin) are evaluated inside a single pure function so that, under
        # jax.jit, XLA fuses the whole sequence into one graph with no host
        # round-trips between sub-kernels. Under numpy this is bit-for-bit
        # identical to the per-kernel path. See kernels/vimain.py.
        # ---------------------------------------------------------------
        xp = bk.xp
        have_pl = adm.ADM_have_pl

        if getattr(self, "_vimain_kernel", None) is None:
            flux_cfg = FluxConvCfg(
                kmin=kmin, kmax=kmax, have_pl=have_pl,
                XDIR=grd.GRD_XDIR, YDIR=grd.GRD_YDIR, ZDIR=grd.GRD_ZDIR,
                gslf_pl=adm.ADM_gslf_pl, gmax_pl=adm.ADM_gmax_pl,
                I_SRC_default=src.I_SRC_default,
                I_SRC_horizontal=src.I_SRC_horizontal,
            )
            adv_cfg = AdvConvCfg(
                kmin=kmin, kmax=kmax, have_pl=have_pl,
                I_SRC_default=src.I_SRC_default,
            )
            sol_cfg = ViSolverCfg(
                kmin=kmin, kmax=kmax, have_pl=have_pl,
                GRAV=float(cnst.CONST_GRAV), Rdry=float(cnst.CONST_Rdry),
                CVdry=float(cnst.CONST_CVdry), alpha=float(rcnf.NON_HYDRO_ALPHA),
            )
            bnd_cfg = BndCfg(
                kmin=kmin, kmax=kmax, have_pl=have_pl,
                is_top_tem=bndc.is_top_tem, is_top_epl=bndc.is_top_epl,
                is_btm_tem=bndc.is_btm_tem, is_btm_epl=bndc.is_btm_epl,
                is_top_rigid=bndc.is_top_rigid, is_top_free=bndc.is_top_free,
                is_btm_rigid=bndc.is_btm_rigid, is_btm_free=bndc.is_btm_free,
                GRAV=float(cnst.CONST_GRAV), Rdry=float(cnst.CONST_Rdry),
            )
            kin_cfg = RhogkinCfg(
                kmin=kmin, kmax=kmax, have_pl=have_pl,
                UNDEF=float(cnst.CONST_UNDEF),
            )
            self._vimain_cfg = VimainCfg(
                kmin=kmin, kmax=kmax, have_pl=have_pl,
                TIME_split=bool(tim.TIME_split),
                Rdry=float(cnst.CONST_Rdry), CVdry=float(cnst.CONST_CVdry),
                I_SRC_default=src.I_SRC_default,
                I_SRC_horizontal=src.I_SRC_horizontal,
                flux=flux_cfg, adv=adv_cfg, sol=sol_cfg, bnd=bnd_cfg, kin=kin_cfg,
            )
            self._vimain_kernel = bk.maybe_jit(
                compute_vi_main, static_argnames=("cfg", "xp"),
            )
        C = bk.device_consts(self, "vimain", lambda: {
            "RGAM":      vmtr.VMTR_RGAM,
            "RGAMH":     vmtr.VMTR_RGAMH,
            "RGSQRTH":   vmtr.VMTR_RGSQRTH,
            "C2WfactGz": vmtr.VMTR_C2WfactGz,
            "coef_div":  oprt.OPRT_coef_div,
            "rdgz":      grd.GRD_rdgz,
            "rdgzh":     grd.GRD_rdgzh,
            "afact":     grd.GRD_afact,
            "bfact":     grd.GRD_bfact,
            "RGSGAM2":   vmtr.VMTR_RGSGAM2,
            "RGSGAM2H":  vmtr.VMTR_RGSGAM2H,
            "GSGAM2H":   vmtr.VMTR_GSGAM2H,
            "C2Wfact":   vmtr.VMTR_C2Wfact,
            "W2Cfact":   vmtr.VMTR_W2Cfact,
            "PHI":       vmtr.VMTR_PHI,
            "RGAM_pl":      vmtr.VMTR_RGAM_pl,
            "RGAMH_pl":     vmtr.VMTR_RGAMH_pl,
            "RGSQRTH_pl":   vmtr.VMTR_RGSQRTH_pl,
            "C2WfactGz_pl": vmtr.VMTR_C2WfactGz_pl,
            "coef_div_pl":  oprt.OPRT_coef_div_pl,
            "RGSGAM2_pl":   vmtr.VMTR_RGSGAM2_pl,
            "RGSGAM2H_pl":  vmtr.VMTR_RGSGAM2H_pl,
            "GSGAM2H_pl":   vmtr.VMTR_GSGAM2H_pl,
            "C2Wfact_pl":   vmtr.VMTR_C2Wfact_pl,
            "W2Cfact_pl":   vmtr.VMTR_W2Cfact_pl,
            "PHI_pl":       vmtr.VMTR_PHI_pl,
        })
        cfg = self._vimain_cfg

        P = {
            "rhogvx_s1": xp.asarray(rhogvx_split1), "rhogvy_s1": xp.asarray(rhogvy_split1),
            "rhogvz_s1": xp.asarray(rhogvz_split1),
            "rhog_s0": xp.asarray(rhog_split0),
            "rhogvx_s0": xp.asarray(rhogvx_split0), "rhogvy_s0": xp.asarray(rhogvy_split0),
            "rhogvz_s0": xp.asarray(rhogvz_split0), "rhogw_s0": xp.asarray(rhogw_split0),
            "rhoge_s0": xp.asarray(rhoge_split0), "preg_s0": xp.asarray(preg_prim_split0),
            "rhog0": xp.asarray(rhog0),
            "rhogvx0": xp.asarray(rhogvx0), "rhogvy0": xp.asarray(rhogvy0),
            "rhogvz0": xp.asarray(rhogvz0), "rhogw0": xp.asarray(rhogw0),
            "eth0": xp.asarray(eth0),
            "grhog": xp.asarray(grhog), "grhogw": xp.asarray(grhogw),
            "grhoge": xp.asarray(grhoge), "grhogetot": xp.asarray(grhogetot),
            # step-4c: when the cached-jit ns-loop drives vi_main it passes the
            # matrix coeffs as TRACED args (Mc_d/...), so xp.asarray() is a no-op
            # referencing the loop's traced input rather than baking self._Mc_d as
            # a per-large-step constant. Falls back to self._Mc_d (concrete) for the
            # eager / non-resident path -> bit-identical.
            "Mc": xp.asarray(Mc_d if Mc_d is not None else getattr(self, "_Mc_d", self.Mc)),
            "Mu": xp.asarray(Mu_d if Mu_d is not None else getattr(self, "_Mu_d", self.Mu)),
            "Ml": xp.asarray(Ml_d if Ml_d is not None else getattr(self, "_Ml_d", self.Ml)),
            # pole (always supplied; consumed only when have_pl)
            "rhogvx_s1_pl": xp.asarray(rhogvx_split1_pl), "rhogvy_s1_pl": xp.asarray(rhogvy_split1_pl),
            "rhogvz_s1_pl": xp.asarray(rhogvz_split1_pl),
            "rhog_s0_pl": xp.asarray(rhog_split0_pl),
            "rhogvx_s0_pl": xp.asarray(rhogvx_split0_pl), "rhogvy_s0_pl": xp.asarray(rhogvy_split0_pl),
            "rhogvz_s0_pl": xp.asarray(rhogvz_split0_pl), "rhogw_s0_pl": xp.asarray(rhogw_split0_pl),
            "rhoge_s0_pl": xp.asarray(rhoge_split0_pl), "preg_s0_pl": xp.asarray(preg_prim_split0_pl),
            "rhog0_pl": xp.asarray(rhog0_pl),
            "rhogvx0_pl": xp.asarray(rhogvx0_pl), "rhogvy0_pl": xp.asarray(rhogvy0_pl),
            "rhogvz0_pl": xp.asarray(rhogvz0_pl), "rhogw0_pl": xp.asarray(rhogw0_pl),
            "eth0_pl": xp.asarray(eth0_pl),
            "grhog_pl": xp.asarray(grhog_pl), "grhogw_pl": xp.asarray(grhogw_pl),
            "grhoge_pl": xp.asarray(grhoge_pl), "grhogetot_pl": xp.asarray(grhogetot_pl),
            "Mc_pl": xp.asarray(Mc_pl_d if Mc_pl_d is not None else getattr(self, "_Mc_pl_d", self.Mc_pl)),
            "Mu_pl": xp.asarray(Mu_pl_d if Mu_pl_d is not None else getattr(self, "_Mu_pl_d", self.Mu_pl)),
            "Ml_pl": xp.asarray(Ml_pl_d if Ml_pl_d is not None else getattr(self, "_Ml_pl_d", self.Ml_pl)),
        }

        out = self._vimain_kernel(P, C, dt, cfg=cfg, xp=xp)

        # Phase 3 (device-resident segment): return outputs on device (jax) so the
        # caller can stack -> on-device COMM -> vipath2c with no host drain.
        if resident:
            prf.PROF_rapend('____vi_main_fused', 2)
            return out

        rhog_split1[:, :, :, :]  = bk.to_numpy(out["rhog_split1"])
        rhogw_split1[:, :, :, :] = bk.to_numpy(out["rhogw_split1"])
        rhoge_split1[:, :, :, :] = bk.to_numpy(out["rhoge_split1"])
        if have_pl:
            rhog_split1_pl[:, :, :]  = bk.to_numpy(out["rhog_split1_pl"])
            rhogw_split1_pl[:, :, :] = bk.to_numpy(out["rhogw_split1_pl"])
            rhoge_split1_pl[:, :, :] = bk.to_numpy(out["rhoge_split1_pl"])

        prf.PROF_rapend('____vi_main_fused', 2)

        return

    def _vi_main_orig(self,
        rhog_split1,      rhog_split1_pl,
        rhogw_split1,     rhogw_split1_pl,
        rhoge_split1,     rhoge_split1_pl,
        rhogvx_split1,    rhogvx_split1_pl,
        rhogvy_split1,    rhogvy_split1_pl,
        rhogvz_split1,    rhogvz_split1_pl,
        rhog_split0,      rhog_split0_pl,
        rhogvx_split0,    rhogvx_split0_pl,
        rhogvy_split0,    rhogvy_split0_pl,
        rhogvz_split0,    rhogvz_split0_pl,
        rhogw_split0,     rhogw_split0_pl,
        rhoge_split0,     rhoge_split0_pl,
        preg_prim_split0, preg_prim_split0_pl,
        rhog0,            rhog0_pl,
        rhogvx0,          rhogvx0_pl,
        rhogvy0,          rhogvy0_pl,
        rhogvz0,          rhogvz0_pl,
        rhogw0,           rhogw0_pl,
        eth0,             eth0_pl,
        grhog,            grhog_pl,
        grhogw,           grhogw_pl,
        grhoge,           grhoge_pl,
        grhogetot,        grhogetot_pl,
        dt,
        rcnf, cnst, vmtr, tim, grd, oprt, bndc, cnvv, src, rdtype,
    ):

        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        gall_1d = adm.ADM_gall_1d
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        lall_pl = adm.ADM_lall_pl

        drhog         = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)   # source term at t=n+1
        drhog_pl      = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  
        drhoge        = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  
        drhoge_pl     = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  
        drhogetot     = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  
        drhogetot_pl  = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  

        grhog1        = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  # source term ( large step + t=n+1 )
        grhog1_pl     = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  
        grhoge1       = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  
        grhoge1_pl    = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  
        gpre          = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  
        gpre_pl       = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  

        rhog1         = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  # prognostic vars ( previous + t=n,t=n+1 )
        rhog1_pl      = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  
        rhogvx1       = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  
        rhogvx1_pl    = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  
        rhogvy1       = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  
        rhogvy1_pl    = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  
        rhogvz1       = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  
        rhogvz1_pl    = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  
        rhogw1        = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  
        rhogw1_pl     = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  

        rhogkin0      = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  # kinetic energy ( previous                )
        rhogkin0_pl   = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  
        rhogkin10     = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  # kinetic energy ( previous + split(t=n)   )
        rhogkin10_pl  = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  
        rhogkin11     = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  # kinetic energy ( previous + split(t=n+1) )
        rhogkin11_pl  = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)  
        ethtot0       = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)  # total enthalpy ( h + v^{2}/2 + phi, previous )
        ethtot0_pl    = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype) 

        Rdry  = cnst.CONST_Rdry
        CVdry = cnst.CONST_CVdry


        #---< update grhog & grhoge >

        if tim.TIME_split:
            # horizontal flux convergence
            # with open(std.fname_log, 'a') as log_file:
            #     print("C3637-A", file=log_file)
            src.src_flux_convergence( 
                rhogvx_split1, rhogvx_split1_pl, # [IN]
                rhogvy_split1, rhogvy_split1_pl, # [IN]
                rhogvz_split1, rhogvz_split1_pl, # [IN]
                rhogw_split0,  rhogw_split0_pl,  # [IN]
                drhog,         drhog_pl,         # [OUT]  
                src.I_SRC_horizontal,            # [IN]
                cnst, grd, oprt, vmtr, rdtype,
            )

            # horizontal advection convergence
            # with open(std.fname_log, 'a') as log_file:
            #     print("C3637-B", file=log_file)
            src.src_advection_convergence(
                rhogvx_split1, rhogvx_split1_pl, # [IN]
                rhogvy_split1, rhogvy_split1_pl, # [IN]
                rhogvz_split1, rhogvz_split1_pl, # [IN]
                rhogw_split0,  rhogw_split0_pl,  # [IN]
                eth0,          eth0_pl,          # [IN]
                drhoge,        drhoge_pl,        # [OUT]  
                src.I_SRC_horizontal,            # [IN]
                cnst, grd, oprt, vmtr, rdtype,
            ) 

        else:

            drhog[:, :, :, :] = rdtype(0.0)
            drhoge[:, :, :, :] = rdtype(0.0)

            drhog_pl[:, :, :] = rdtype(0.0)
            drhoge_pl[:, :, :] = rdtype(0.0)

        #endif

        # update grhog, grhoge and calc source term of pressure

        grhog1[:, :, :, :]  = grhog[:, :, :, :]  + drhog[:, :, :, :]
        grhoge1[:, :, :, :] = grhoge[:, :, :, :] + drhoge[:, :, :, :]
        gpre[:, :, :, :]    = grhoge1[:, :, :, :] * Rdry / CVdry
 

        if adm.ADM_have_pl:
            grhog1_pl  = grhog_pl  + drhog_pl      #####CHECK3637
            grhoge1_pl = grhoge_pl + drhoge_pl     #####CHECK3637
            gpre_pl    = grhoge1_pl * Rdry / CVdry
        #endif

        # with open(std.fname_log, 'a') as log_file:
        #     print("C3637", file=log_file)
        #     print("grhog_pl", grhog_pl[:, 36, 0], grhog_pl[:, 37, 0], file=log_file)
        #     print("drhog_pl", drhog_pl[:, 36, 0], drhog_pl[:, 37, 0], file=log_file)      
        #     print("grhoge_pl", grhoge_pl[:, 36, 0], grhoge_pl[:, 37, 0], file=log_file)
        #     print("drhoge_pl", drhoge_pl[:, 36, 0], drhoge_pl[:, 37, 0], file=log_file)   


        #---------------------------------------------------------------------------
        # vertical implict calculation core
        #---------------------------------------------------------------------------

        # boundary condition for rhogw_split1

        rhogw_split1[:, :, :, :] = rdtype(0.0)
        
        # with open(std.fname_log, 'a') as log_file:
        #     print("", file=log_file)
        #     print("check before BNDCND_rhow", file=log_file)
        #     print("rhogvx_split1 k=41", file=log_file)
        #     print(rhogvx_split1[6, 5, 41, 0], file=log_file)
        #     print("rhogvx_split1 k=2", file=log_file)
        #     print(rhogvx_split1[6, 5, 2, 0], file=log_file)
        #     print("rhogvy_split1", file=log_file)
        #     print(rhogvy_split1[6, 5, 41, 0], file=log_file)
        #     print("rhogvz_split1", file=log_file)
        #     print(rhogvz_split1[6, 5, 41, 0], file=log_file)
        #     print("rhogw_split1", file=log_file)
        #     print(rhogw_split1[6, 5, 41, 0], file=log_file)
        #     print("vmtr.VMTR_C2WfactGz", file=log_file)
        #     print(vmtr.VMTR_C2WfactGz[6, 5, 41, :, 0], file=log_file)

        bndc.BNDCND_rhow(
            kmin, kmax,
            rhogvx_split1 [:,:,:,:],       # [IN]
            rhogvy_split1 [:,:,:,:],       # [IN]
            rhogvz_split1 [:,:,:,:],       # [IN]
            rhogw_split1  [:,:,:,:],       # [INOUT]
            vmtr.VMTR_C2WfactGz[:,:,:,:,:], # [IN]
            rdtype,
        )

        # with open(std.fname_log, 'a') as log_file:
        #     print("after BNDCND_rhow", file=log_file)
        #     print("rhogw_split1", file=log_file)
        #     print(rhogw_split1[6, 5, 41, 0], file=log_file)
        #     print(rhogw_split1[6, 5, 40, 0], file=log_file)
        #     print(rhogw_split1[6, 5, 0, 0], file=log_file)
        #     print(rhogw_split1[6, 5, 1, 0], file=log_file)
        #     print("", file=log_file)

        #prc.prc_mpistop(std.io_l, std.fname_log)


        if adm.ADM_have_pl:
            rhogw_split1_pl[:,:,:] = rdtype(0.0)        # Tracing start from here
            
            # for l in range(adm.ADM_lall_pl):
            #     rxpl1=np.empty((gall_pl, kall), dtype=rdtype)
            #     rxpl1[:,:]=rhogvx_split1_pl[:,:,l]
            #     bndc.BNDCND_rhow(
            #         rhogvx_split1_pl [:,np.newaxis,:,l],     # [IN]
            #         rhogvy_split1_pl [:,np.newaxis,:,l],     # [IN]
            #         rhogvz_split1_pl [:,np.newaxis,:,l],     # [IN]
            #         rhogw_split1_pl  [:,np.newaxis,:,l],     # [INOUT]      
            #         vmtr.VMTR_C2WfactGz_pl[:,np.newaxis,:,:,l]    # [IN]
            #     )
            #end loop l
            #for l in range(adm.ADM_lall_pl):
                #$$ rxpl1=np.full((gall_pl, kall), cnst.CONST_UNDEF, dtype=rdtype)
                #$$ rxpl1[:,:]=rhogvx_split1_pl[:,:,l]
            bndc.BNDCND_rhow_pl(
                kmin, kmax,
                rhogvx_split1_pl [:,:,:],     # [IN]
                rhogvy_split1_pl [:,:,:],     # [IN]
                rhogvz_split1_pl [:,:,:],     # [IN]
                rhogw_split1_pl  [:,:,:],     # [INOUT]      
                vmtr.VMTR_C2WfactGz_pl[:,:,:,:],    # [IN]
                rdtype,
            )

            # with open(std.fname_log, 'a') as log_file:
            #     print("after BNDCND_rhow_pl", file=log_file)
            #     print("rhogw_split1_pl", file=log_file)
            #     print(rhogw_split1_pl[:, 0, 0], file=log_file)
            #     print(rhogw_split1_pl[:, 2, 0], file=log_file)
            #     print(rhogw_split1_pl[:,41, 0], file=log_file)  

        #endif

        # self.counter += 1
        # with open(std.fname_log, 'a') as log_file:
        #     print("", file=log_file)
        #     print("rhogw_split1_pl before vi_rhow_solver", file=log_file)
        #     print("counter=", self.counter, file=log_file)
        #     print(rhogw_split1_pl[:, 37, 0], file=log_file)  
        #     print(rhogw_split0_pl[:, 37, 0], file=log_file)
        #     print(preg_prim_split0_pl[:, 37, 0], file=log_file)
        #     print(rhog_split0_pl[:, 37, 0], file=log_file)
        #     print(grhog1_pl[:, 37, 0], file=log_file)             
        #     print(grhogw_pl[:, 37, 0], file=log_file)
        #     print(gpre_pl[:, 37, 0], file=log_file)

        # update rhogw_split1
        self.vi_rhow_solver(
            rhogw_split1,     rhogw_split1_pl,     # [INOUT]     
            rhogw_split0,     rhogw_split0_pl,     # [IN]
            preg_prim_split0, preg_prim_split0_pl, # [IN]
            rhog_split0,      rhog_split0_pl,      # [IN]
            grhog1,           grhog1_pl,           # [IN]
            grhogw,           grhogw_pl,           # [IN]
            gpre,             gpre_pl,             # [IN]
            dt,                                    # [IN]
            cnst, grd, vmtr, rcnf, rdtype, 
        )

        # j=0
        # k=3
        # l=1
        # print(f"cC, j, k, l, {j}, {k}, {l},", rhogw_split1[:,j,k,l])
        
        # with open(std.fname_log, 'a') as log_file:
        #     print("", file=log_file)
        #     print("rhogw_split1_pl after vi_rhow_solver", file=log_file)
        #     print(rhogw_split1_pl[:, 0, 0], file=log_file)
        #     print(rhogw_split1_pl[:, 3, 0], file=log_file)    
        #     print(rhogw_split1_pl[:,41, 0], file=log_file)

        # update rhog_split1
        src.src_flux_convergence(
            rhogvx_split1, rhogvx_split1_pl, # [IN]
            rhogvy_split1, rhogvy_split1_pl, # [IN]
            rhogvz_split1, rhogvz_split1_pl, # [IN]
            rhogw_split1,  rhogw_split1_pl,  # [IN]    ###
            drhog,         drhog_pl,         # [OUT]
            src.I_SRC_default,              # [IN]
            cnst, grd, oprt, vmtr, rdtype,
        )

        rhog_split1[:, :, :, :] = rhog_split0[:, :, :, :] + (grhog[:, :, :, :] + drhog[:, :, :, :]) * dt

        if adm.ADM_have_pl:
            rhog_split1_pl[:, :, :] = rhog_split0_pl[:, :, :] + (grhog_pl[:, :, :] + drhog_pl[:, :, :]) * dt
        #endif

#         with open(std.fname_log, 'a') as log_file:
#             print("", file=log_file)
#             print("rhog_split1_pl before Satoh2002", file=log_file)
# #            print("rhog_split1", file=log_file)
#             print(rhog_split1_pl[:, 39, 0], file=log_file)               
#             print(rhog_split0_pl[:, 39, 0], file=log_file)
#             print(grhog_pl[:, 39, 0], file=log_file)
#             print(drhog_pl[:, 39, 0], file=log_file)         
#             print(rhog_split1_pl[:, 39, 0], file=log_file)
#             print(rhog_split0_pl[:, 39, 0], file=log_file)
#             print(grhog_pl[:, 39, 0], file=log_file)
#             print(drhog_pl[:, 39, 0], file=log_file)
  
#             print("", file=log_file)

        #---------------------------------------------------------------------------
        # energy correction by Etotal (Satoh,2002)
        #---------------------------------------------------------------------------

        # overflow encountered during cnvvar_rhogkin (not always, so it is likely an array issue)

        # with open(std.fname_log, 'a') as log_file:
        #     print("KONATA?", file=log_file)
        #     print("rhog0_pl [:,39,0] ",    rhog0_pl[:,39,0], file=log_file)
        #     print("rhog0_pl [:,40,0] ",    rhog0_pl[:,40,0], file=log_file)
        #     print("rhogvx0_pl [:,39,0] ",  rhogvx0_pl[:,39,0], file=log_file)
        #     print("rhogvx0_pl [:,40,0] ",  rhogvx0_pl[:,40,0], file=log_file)
        #     print("rhogvy0_pl [:,39,0] ",  rhogvy0_pl[:,39,0], file=log_file)
        #     print("rhogvy0_pl [:,40,0] ",  rhogvy0_pl[:,40,0], file=log_file)
        #     print("rhogvz0_pl [:,39,0] ",  rhogvz0_pl[:,39,0], file=log_file)
        #     print("rhogvz0_pl [:,40,0] ",  rhogvz0_pl[:,40,0], file=log_file)
        #     print("rhogw0_pl [:,39,0] ",   rhogw0_pl[:,39,0], file=log_file)
        #     print("rhogw0_pl [:,40,0] ",   rhogw0_pl[:,40,0], file=log_file)

        # calc rhogkin ( previous )

        rhogkin0, rhogkin0_pl = cnvv.cnvvar_rhogkin(
                                    rhog0,    rhog0_pl,    # [IN]
                                    rhogvx0,  rhogvx0_pl,  # [IN]
                                    rhogvy0,  rhogvy0_pl,  # [IN]
                                    rhogvz0,  rhogvz0_pl,  # [IN]
                                    rhogw0,   rhogw0_pl,   # [IN]
                                    cnst, vmtr, rdtype,
                                )

        # with open(std.fname_log, 'a') as log_file:
        #     print("KOCHIRA?", file=log_file)
        #     print("rhogkin0_pl [:, 2,0] ",  rhogkin0_pl[:, 2,0], file=log_file)
        #     print("rhogkin0_pl [:,39,0] ",  rhogkin0_pl[:,39,0], file=log_file)
        #     print("rhog0",   rhog0  [6,5,2,0], file=log_file)
        #     print("rhogvx0", rhogvx0[6,5,2,0], file=log_file)
        #     print("rhogvy0", rhogvy0[6,5,2,0], file=log_file)
        #     print("rhogvz0", rhogvz0[6,5,2,0], file=log_file)
        #     print("rhogw0",  rhogw0 [6,5,2,0], file=log_file)
        #     print("rhog0_pl 0,2 ",   rhog0_pl  [0,2,0], file=log_file)
        #     print("rhogvx0_pl   ", rhogvx0_pl[0,2,0], file=log_file)
        #     print("rhogvy0_pl   ", rhogvy0_pl[0,2,0], file=log_file)
        #     print("rhogvz0_pl   ", rhogvz0_pl[0,2,0], file=log_file)
        #     print("rhogw0_pl    ",  rhogw0_pl[0,2,0], file=log_file)
        #     print("rhog0_pl 2,2 ",   rhog0_pl[2,2,0], file=log_file)
        #     print("rhogvx0_pl   ", rhogvx0_pl[2,2,0], file=log_file)
        #     print("rhogvy0_pl   ", rhogvy0_pl[2,2,0], file=log_file)
        #     print("rhogvz0_pl   ", rhogvz0_pl[2,2,0], file=log_file)
        #     print("rhogw0_pl    ",  rhogw0_pl[2,2,0], file=log_file)
        #     print("rhogkin0        ",       rhogkin0[6,5,2,0], file=log_file)
        #     print("rhogkin0_pl 0,2 ",  rhogkin0_pl[0,2,0], file=log_file)
        #     print("rhogkin0_pl 2,2 ",  rhogkin0_pl[2,2,0], file=log_file)


        # prognostic variables ( previous + split (t=n) )

        rhog1[:, :, :, :]   = rhog0[:, :, :, :]   + rhog_split0[:, :, :, :]
        rhogvx1[:, :, :, :] = rhogvx0[:, :, :, :] + rhogvx_split0[:, :, :, :]
        rhogvy1[:, :, :, :] = rhogvy0[:, :, :, :] + rhogvy_split0[:, :, :, :]
        rhogvz1[:, :, :, :] = rhogvz0[:, :, :, :] + rhogvz_split0[:, :, :, :]
        rhogw1[:, :, :, :]  = rhogw0[:, :, :, :]  + rhogw_split0[:, :, :, :]

        if adm.ADM_have_pl:
            rhog1_pl  [:, :, :] = rhog0_pl  [:, :, :] + rhog_split0_pl  [:, :, :]
            rhogvx1_pl[:, :, :] = rhogvx0_pl[:, :, :] + rhogvx_split0_pl[:, :, :]
            rhogvy1_pl[:, :, :] = rhogvy0_pl[:, :, :] + rhogvy_split0_pl[:, :, :]
            rhogvz1_pl[:, :, :] = rhogvz0_pl[:, :, :] + rhogvz_split0_pl[:, :, :]
            rhogw1_pl [:, :, :] = rhogw0_pl [:, :, :] + rhogw_split0_pl [:, :, :]

        # calc rhogkin ( previous + split(t=n) )

        rhogkin10, rhogkin10_pl = cnvv.cnvvar_rhogkin(
                                        rhog1,    rhog1_pl,      # [IN]
                                        rhogvx1,  rhogvx1_pl,    # [IN]
                                        rhogvy1,  rhogvy1_pl,    # [IN]
                                        rhogvz1,  rhogvz1_pl,    # [IN]
                                        rhogw1,   rhogw1_pl,     # [IN]
                                        cnst, vmtr, rdtype,
                                    )
        
        # with open(std.fname_log, 'a') as log_file:
        #     print("", file=log_file)
        #     print("rhog1",   rhog1  [6,5,2,0], file=log_file)
        #     print("rhogvx1", rhogvx1[6,5,2,0], file=log_file)
        #     print("rhogvy1", rhogvy1[6,5,2,0], file=log_file)
        #     print("rhogvz1", rhogvz1[6,5,2,0], file=log_file)
        #     print("rhogw1",  rhogw1 [6,5,2,0], file=log_file)
        #     print("rhog1_pl 0,2 ",   rhog1_pl  [0,2,0], file=log_file)
        #     print("rhogvx1_pl   ", rhogvx1_pl[0,2,0], file=log_file)
        #     print("rhogvy1_pl   ", rhogvy1_pl[0,2,0], file=log_file)
        #     print("rhogvz1_pl   ", rhogvz1_pl[0,2,0], file=log_file)
        #     print("rhogw1_pl    ",  rhogw1_pl[0,2,0], file=log_file)
        #     print("rhog1_pl 2,2 ",   rhog1_pl[2,2,0], file=log_file)
        #     print("rhogvx1_pl   ", rhogvx1_pl[2,2,0], file=log_file)
        #     print("rhogvy1_pl   ", rhogvy1_pl[2,2,0], file=log_file)
        #     print("rhogvz1_pl   ", rhogvz1_pl[2,2,0], file=log_file)
        #     print("rhogw1_pl    ",  rhogw1_pl[2,2,0], file=log_file)
        #     print("rhogkin10        ",     rhogkin10[6,5,2,0], file=log_file)
        #     print("rhogkin10_pl 0,2 ",  rhogkin10_pl[0,2,0], file=log_file)
        #     print("rhogkin10_pl 2,2 ",  rhogkin10_pl[2,2,0], file=log_file)

        # prognostic variables ( previous + split (t=n+1) )

        rhog1[:, :, :, :]   = rhog0[:, :, :, :]   + rhog_split1[:, :, :, :]
        rhogvx1[:, :, :, :] = rhogvx0[:, :, :, :] + rhogvx_split1[:, :, :, :]
        rhogvy1[:, :, :, :] = rhogvy0[:, :, :, :] + rhogvy_split1[:, :, :, :]
        rhogvz1[:, :, :, :] = rhogvz0[:, :, :, :] + rhogvz_split1[:, :, :, :]
        rhogw1[:, :, :, :]  = rhogw0[:, :, :, :]  + rhogw_split1[:, :, :, :]  

        if adm.ADM_have_pl:
            rhog1_pl[:, :, :]   = rhog0_pl[:, :, :]   + rhog_split1_pl[:, :, :]       
            rhogvx1_pl[:, :, :] = rhogvx0_pl[:, :, :] + rhogvx_split1_pl[:, :, :]     
            rhogvy1_pl[:, :, :] = rhogvy0_pl[:, :, :] + rhogvy_split1_pl[:, :, :]     
            rhogvz1_pl[:, :, :] = rhogvz0_pl[:, :, :] + rhogvz_split1_pl[:, :, :]     
            rhogw1_pl[:, :, :]  = rhogw0_pl[:, :, :]  + rhogw_split1_pl[:, :, :]      

        #### overflow check
        # for l in range(lall):
        #     for k in range(3,kall):
        #         for j  in range(gall_1d):
        #             #for i in range(gall_1d):
        #                 with open(std.fname_log, 'a') as log_file:
        #                     #print("aA, j, k, l", j, k, l, rhogw1[:,j,k,l], file=log_file)    
        #                     #wprint("bB, j, k, l", j, k, l, rhogw0[:,j,k,l], file=log_file)
        #                     print("cC, j, k, l", j, k, l, rhogw_split1[:,j,k,l], file=log_file)  #Halo is corrupted, but no problem?
        #                 #a = rhogw1[i,j,k,l] ** 2

        # calc rhogkin ( previous + split(t=n+1) )
        rhogkin11, rhogkin11_pl = cnvv.cnvvar_rhogkin(
                                        rhog1,    rhog1_pl,      # [IN]
                                        rhogvx1,  rhogvx1_pl,    # [IN]
                                        rhogvy1,  rhogvy1_pl,    # [IN]
                                        rhogvz1,  rhogvz1_pl,    # [IN]
                                        rhogw1,   rhogw1_pl,     # [IN]
                                        cnst, vmtr, rdtype,
                                    )
        
        # l=1
        # k=3
        # with open(std.fname_log, 'a') as log_file:
        #     print(f"aAA, j, k, l: {0}, {k}, {l},", rhogkin11[:,0,k,l], file=log_file) 

        # with open(std.fname_log, 'a') as log_file:
        #     print("", file=log_file)
        #     print("rhog1",   rhog1  [6,5,2,0], file=log_file)
        #     print("rhogvx1", rhogvx1[6,5,2,0], file=log_file)
        #     # print("rhogvy1", rhogvy1[6,5,2,0], file=log_file)
            # print("rhogvz1", rhogvz1[6,5,2,0], file=log_file)
            # print("rhogw1",  rhogw1 [6,5,2,0], file=log_file)
            # print("rhog1_pl 0,2 ",   rhog1_pl  [0,2,0], file=log_file)            #!
            # print("rhogvx1_pl   ", rhogvx1_pl[0,2,0], file=log_file)              
            # print("rhogvy1_pl   ", rhogvy1_pl[0,2,0], file=log_file)
            # print("rhogvz1_pl   ", rhogvz1_pl[0,2,0], file=log_file)              #!
            # print("rhogw1_pl    ",  rhogw1_pl[0,2,0], file=log_file)              #!
            # print("rhog1_pl 2,2 ",   rhog1_pl[2,2,0], file=log_file)
            # print("rhogvx1_pl   ", rhogvx1_pl[2,2,0], file=log_file)
            # print("rhogvy1_pl   ", rhogvy1_pl[2,2,0], file=log_file)
            # print("rhogvz1_pl   ", rhogvz1_pl[2,2,0], file=log_file)
            # print("rhogw1_pl    ",  rhogw1_pl[2,2,0], file=log_file)            
            # print("rhogkin11        ",     rhogkin11[6,5,2,0], file=log_file)
            # print("rhogkin11_pl 0,2 ",  rhogkin11_pl[0,2,0], file=log_file)        #!
            # print("rhogkin11_pl 2,2 ",  rhogkin11_pl[2,2,0], file=log_file)        #!
            # print("rhogkin11_pl :,2 ",  rhogkin11_pl[:,2,0], file=log_file) 
        # calculate total enthalpy ( h + v^{2}/2 + phi, previous )

        ethtot0[:, :, :, :] = (
            eth0[:, :, :, :]
            + rhogkin0[:, :, :, :] / rhog0[:, :, :, :]
            + vmtr.VMTR_PHI[:, :, :, :]
        )

        if adm.ADM_have_pl:
            ethtot0_pl[:, :, :] = (
                eth0_pl[:, :, :]
                + rhogkin0_pl[:, :, :] / rhog0_pl[:, :, :]
                + vmtr.VMTR_PHI_pl[:, :, :]
            )

        # advection convergence for eth + kin + phi
        # with open(std.fname_log, 'a') as log_file:
        #     print("KOKOCA?", file=log_file)
        #     kc=39
        # #     print("self.rhogvxscl (6,5,2,0)", self.rhogvxscl[6, 5, 2, 0], file=log_file) 
        # #     print("self.rhogvyscl (6,5,2,0)", self.rhogvyscl[6, 5, 2, 0], file=log_file) 
        # #     print("self.rhogvzscl (6,5,2,0)", self.rhogvzscl[6, 5, 2, 0], file=log_file) 
        # #     print("self.rhogwscl (6,5,2,0)", self.rhogwscl[6, 5, 2, 0], file=log_file)
        #     print(f"rhogvx1_pl (:,{kc},0)", rhogvx1_pl[:, kc, 0], file=log_file)  
        #     print(f"rhogvy1_pl (:,{kc},0)", rhogvy1_pl[:, kc, 0], file=log_file)  
        #     print(f"rhogvz1_pl (:,{kc},0)", rhogvz1_pl[:, kc, 0], file=log_file)  
        #     print(f"rhogw1_pl  (:,{kc},0)", rhogw1_pl [:, kc, 0], file=log_file)  
        #     print(f"ethtot0_pl (:,{kc},0)", ethtot0_pl[:, kc, 0], file=log_file)  #broken at 39   scl_pl in src_adv_conv
        #     print(f"eth0_pl (:,{kc},0)", eth0_pl[:, kc, 0], file=log_file)
        #     print(f"rhogkin0_pl (:,{kc},0)", rhogkin0_pl[:, kc, 0], file=log_file)  #broken at 39
        #     print(f"rhog0_pl (:,{kc},0)", rhog0_pl[:, kc, 0], file=log_file)
        #     print(f"vmtr.VMTR_PHI_pl (:,{kc},0)", vmtr.VMTR_PHI_pl[:, kc, 0], file=log_file)

        src.src_advection_convergence(
            rhogvx1,    rhogvx1_pl,   # [IN]
            rhogvy1,    rhogvy1_pl,   # [IN]
            rhogvz1,    rhogvz1_pl,   # [IN]
            rhogw1,     rhogw1_pl,    # [IN]
            ethtot0,    ethtot0_pl,   # [IN]
            drhogetot,  drhogetot_pl, # [OUT]
            src.I_SRC_default,        # [IN]
            cnst, grd, oprt, vmtr, rdtype,
        )

        rhoge_split1[:, :, :, :] = (
            rhoge_split0[:, :, :, :]
            + (grhogetot[:, :, :, :] + drhogetot[:, :, :, :]) * dt
            + (rhogkin10[:, :, :, :] - rhogkin11[:, :, :, :])
            + (rhog_split0[:, :, :, :] - rhog_split1[:, :, :, :]) * vmtr.VMTR_PHI[:, :, :, :]
        )

        if adm.ADM_have_pl:
            rhoge_split1_pl[:, :, :] = (
                rhoge_split0_pl[:, :, :]
                + (grhogetot_pl[:, :, :] + drhogetot_pl[:, :, :]) * dt
                + (rhogkin10_pl[:, :, :] - rhogkin11_pl[:, :, :])
                + (rhog_split0_pl[:, :, :] - rhog_split1_pl[:, :, :]) * vmtr.VMTR_PHI_pl[:, :, :]
            )

        # with open(std.fname_log, 'a') as log_file:
        #     print("XXXX rhogw_split1_pl", file=log_file)
        #                         # g  k  l             
        #     print(rhog_split1_pl[0, 3, 0], file=log_file)
        #     print(rhog_split1_pl[1, 3, 0], file=log_file)
        #     print(rhog_split1_pl[2, 3, 0], file=log_file)
        #     print(rhog_split1_pl[3, 3, 0], file=log_file)
        #     print(rhog_split1_pl[4, 3, 0], file=log_file)
        
        #     print(rhogw_split1_pl[0, 3, 0], file=log_file)
        #     print(rhogw_split1_pl[1, 3, 0], file=log_file)
        #     print(rhogw_split1_pl[2, 3, 0], file=log_file)
        #     print(rhogw_split1_pl[3, 3, 0], file=log_file)
        #     print(rhogw_split1_pl[4, 3, 0], file=log_file)

        #     print(rhoge_split1_pl[0, 3, 0], file=log_file)
        #     print(rhoge_split1_pl[1, 3, 0], file=log_file)
        #     print(rhoge_split1_pl[2, 3, 0], file=log_file)
        #     print(rhoge_split1_pl[3, 3, 0], file=log_file)
        #     print(rhoge_split1_pl[4, 3, 0], file=log_file)

        return


    #> Tridiagonal matrix solver
    def vi_rhow_solver(self,
        rhogw,  rhogw_pl,     # rho*w          ( G^1/2 x gam2 ), n+1       [INOUT] ####
        rhogw0, rhogw0_pl,    # rho*w          ( G^1/2 x gam2 )            [IN]    
        preg0,  preg0_pl,     # pressure prime ( G^1/2 x gam2 )            [IN]
        rhog0,  rhog0_pl,     # rho            ( G^1/2 x gam2 )            [IN]
        Srho,   Srho_pl,      # source term for rho  at the full level     [IN]
        Sw,     Sw_pl,        # source term for rhow at the half level     [IN]
        Spre,   Spre_pl,      # source term for pres at the full level     [IN]
        dt,
        cnst, grd, vmtr, rcnf, rdtype,                 
        ):

        prf.PROF_rapstart('____vi_rhow_solver',2)

        # --- COMM-free tri-diagonal (Thomas) solve via backend-switchable
        #     kernel (numpy<->jax). The forward/backward sweeps carry a k
        #     recurrence (not data-parallel); the kernel loops over k while
        #     keeping i/j/l vectorized. See kernels/virhowsolver.py. ---
        xp = bk.xp

        if getattr(self, "_visolver_kernels", None) is None:
            self._visolver_cfg = ViSolverCfg(
                kmin=adm.ADM_kmin, kmax=adm.ADM_kmax,
                have_pl=adm.ADM_have_pl,
                GRAV=float(cnst.CONST_GRAV),
                Rdry=float(cnst.CONST_Rdry),
                CVdry=float(cnst.CONST_CVdry),
                alpha=float(rcnf.NON_HYDRO_ALPHA),
            )
            self._visolver_kernels = {
                "reg": bk.maybe_jit(compute_rhow_solver_reg, static_argnames=("cfg", "xp")),
                "pl":  bk.maybe_jit(compute_rhow_solver_pl,  static_argnames=("cfg", "xp")),
            }
        d = bk.device_consts(self, "visolver", lambda: {
            "RGAMH":    vmtr.VMTR_RGAMH,
            "RGSGAM2":  vmtr.VMTR_RGSGAM2,
            "RGAM":     vmtr.VMTR_RGAM,
            "RGSGAM2H": vmtr.VMTR_RGSGAM2H,
            "GSGAM2H":  vmtr.VMTR_GSGAM2H,
            "rdgzh":    grd.GRD_rdgzh,
            "afact":    grd.GRD_afact,
            "bfact":    grd.GRD_bfact,
            "RGAMH_pl":    vmtr.VMTR_RGAMH_pl,
            "RGSGAM2_pl":  vmtr.VMTR_RGSGAM2_pl,
            "RGAM_pl":     vmtr.VMTR_RGAM_pl,
            "RGSGAM2H_pl": vmtr.VMTR_RGSGAM2H_pl,
            "GSGAM2H_pl":  vmtr.VMTR_GSGAM2H_pl,
        })
        cfg = self._visolver_cfg

        _rhogw = self._visolver_kernels["reg"](
            xp.asarray(rhogw), xp.asarray(rhogw0), xp.asarray(preg0), xp.asarray(rhog0),
            xp.asarray(Srho), xp.asarray(Sw), xp.asarray(Spre),
            xp.asarray(self.Mc), xp.asarray(self.Mu), xp.asarray(self.Ml),
            d["RGAMH"], d["RGSGAM2"], d["RGAM"], d["RGSGAM2H"], d["GSGAM2H"],
            d["rdgzh"], d["afact"], d["bfact"], dt, cfg=cfg, xp=xp,
        )
        rhogw[:, :, :, :] = bk.to_numpy(_rhogw)

        if adm.ADM_have_pl:
            _rhogw_pl = self._visolver_kernels["pl"](
                xp.asarray(rhogw_pl), xp.asarray(rhogw0_pl), xp.asarray(preg0_pl), xp.asarray(rhog0_pl),
                xp.asarray(Srho_pl), xp.asarray(Sw_pl), xp.asarray(Spre_pl),
                xp.asarray(self.Mc_pl), xp.asarray(self.Mu_pl), xp.asarray(self.Ml_pl),
                d["RGAMH_pl"], d["RGSGAM2_pl"], d["RGAM_pl"], d["RGSGAM2H_pl"], d["GSGAM2H_pl"],
                d["rdgzh"], d["afact"], d["bfact"], dt, cfg=cfg, xp=xp,
            )
            rhogw_pl[:, :, :] = bk.to_numpy(_rhogw_pl)

        prf.PROF_rapend('____vi_rhow_solver',2)

        return
