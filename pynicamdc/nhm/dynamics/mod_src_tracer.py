import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf


class Srctr:
    
    _instance = None
    

    def __init__(self,cnst,rdtype):
        pass


    def src_tracer_advection( 
       vmax,                         # [IN] number of tracers   
       rhogq,       rhogq_pl,        # [INOUT] rhogq   ( G^1/2 x gam2 )
       rhog_in,     rhog_in_pl,      # [IN] rho(old)( G^1/2 x gam2 )
       rhog_mean,   rhog_mean_pl,    # [IN] rho     ( G^1/2 x gam2 )
       rhogvx_mean, rhogvx_mean_pl,  # [IN] rho*Vx  ( G^1/2 x gam2 )
       rhogvy_mean, rhogvy_mean_pl,  # [IN] rho*Vy  ( G^1/2 x gam2 )
       rhogvz_mean, rhogvz_mean_pl,  # [IN] rho*Vz  ( G^1/2 x gam2 )
       rhogw_mean,  rhogw_mean_pl,   # [IN] rho*w  ( G^1/2 x gam2 )
       frhog,       frhog_pl,        # [IN] hyperviscosity tendency for rhog
       dt,                           # [IN] delta t
       thuburn_lim,                  # [IN] switch of thuburn limiter [add] 20130613 R.Yoshida   
       thuburn_lim_v, thuburn_lim_h, # [IN] switch of thuburn limiter, optional 
       cnst, grd, gmtr, vmtr, rdtype,
    ):

        TI  = adm.ADM_TI  
        TJ  = adm.ADM_TJ  
        AI  = adm.ADM_AI  
        AIJ = adm.ADM_AIJ
        AJ  = adm.ADM_AJ  
        K0  = adm.ADM_KNONE

        XDIR = grd.GRD_XDIR 
        YDIR = grd.GRD_YDIR 
        ZDIR = grd.GRD_ZDIR

        rhog     = np.full(adm.ADM_shape, cnst.CONST_undef, type=rdtype)
        rhog_pl  = np.full(adm.ADM_shape_pl, cnst.CONST_undef, type=rdtype)
        rhogvx   = np.full(adm.ADM_shape, cnst.CONST_undef, type=rdtype)
        rhogvx_pl= np.full(adm.ADM_shape_pl, cnst.CONST_undef, type=rdtype)
        rhogvy   = np.full(adm.ADM_shape, cnst.CONST_undef, type=rdtype)
        rhogvy_pl= np.full(adm.ADM_shape_pl, cnst.CONST_undef, type=rdtype)
        rhogvz   = np.full(adm.ADM_shape, cnst.CONST_undef, type=rdtype)
        rhogvz_pl= np.full(adm.ADM_shape_pl, cnst.CONST_undef, type=rdtype)

        q        = np.full(adm.ADM_shape, cnst.CONST_undef, type=rdtype)
        q_pl     = np.full(adm.ADM_shape_pl, cnst.CONST_undef, type=rdtype)
        d        = np.full(adm.ADM_shape, cnst.CONST_undef, type=rdtype)
        d_pl     = np.full(adm.ADM_shape_pl, cnst.CONST_undef, type=rdtype)

        q_h      = np.full(adm.ADM_shape, cnst.CONST_undef, type=rdtype)          # q at layer face
        q_h_pl   = np.full(adm.ADM_shape_pl, cnst.CONST_undef, type=rdtype)
        flx_v    = np.full(adm.ADM_shape, cnst.CONST_undef, type=rdtype)          # mass flux
        flx_v_pl = np.full(adm.ADM_shape_pl, cnst.CONST_undef, type=rdtype)
        ck       = np.full(adm.ADM_shape +(2,), cnst.CONST_undef, type=rdtype)    # Courant number
        ck_pl    = np.full(adm.ADM_shape_pl +(2,), cnst.CONST_undef, type=rdtype)

        q_a      = np.full(adm.ADM_shape +(6,), cnst.CONST_undef, type=rdtype)    # q at cell face
        q_a_pl   = np.full(adm.ADM_shape_pl, cnst.CONST_undef, type=rdtype)
        flx_h    = np.full(adm.ADM_shape +(6,), cnst.CONST_undef, type=rdtype)    # mass flux
        flx_h_pl = np.full(adm.ADM_shape_pl, cnst.CONST_undef, type=rdtype)
        ch       = np.full(adm.ADM_shape +(6,), cnst.CONST_undef, type=rdtype)    # Courant number
        ch_pl    = np.full(adm.ADM_shape_pl, cnst.CONST_undef, type=rdtype)
        cmask    = np.full(adm.ADM_shape +(6,), cnst.CONST_undef, type=rdtype)    # upwind direction mask
        cmask_pl = np.full(adm.ADM_shape_pl, cnst.CONST_undef, type=rdtype)
        GRD_xc   = np.full(adm.ADM_shape + (AJ - AI + 1, ZDIR - XDIR +1,), cnst.CONST_undef, type=rdtype)                   # mass centroid position
        GRD_xc_pl= np.full(adm.ADM_shape_pl + (ZDIR - XDIR +1,), cnst.CONST_undef, type=rdtype)

        EPS = cnst.CONST_EPS

        gmin = adm.ADM_gmin
        gmax = adm.ADM_gmax
        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kall = adm.ADM_kall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        lall_pl = adm.ADM_lall_pl

        b1 = 0.0
        b2 = 1.0
        b3 = 1.0 - (b1+b2)

        apply_limiter_v = np.full(vmax, thuburn_lim, dtype=bool)
        apply_limiter_h = np.full(vmax, thuburn_lim, dtype=bool)

        if thuburn_lim_v is not None:
            apply_limiter_v[:] = thuburn_lim_v[:]

        if thuburn_lim_h is not None:
            apply_limiter_h[:] = thuburn_lim_h[:]

        #---------------------------------------------------------------------------
        # Vertical Advection (fractioanl step) : 1st
        #---------------------------------------------------------------------------
        prf.PROF_rapstart('____vertical_adv',2)

        for l in range(lall):
            for k in range(kmin+1, kmax+1):
               flx_v[:, :, k, l] = (
                    (
                        vmtr.VMTR_C2WfactGz[:, :, k, 0, l] * rhogvx_mean[:, :, k,   l] +
                        vmtr.VMTR_C2WfactGz[:, :, k, 1, l] * rhogvx_mean[:, :, k-1, l] +
                        vmtr.VMTR_C2WfactGz[:, :, k, 2, l] * rhogvy_mean[:, :, k,   l] +
                        vmtr.VMTR_C2WfactGz[:, :, k, 3, l] * rhogvy_mean[:, :, k-1, l] +
                        vmtr.VMTR_C2WfactGz[:, :, k, 4, l] * rhogvz_mean[:, :, k,   l] +
                        vmtr.VMTR_C2WfactGz[:, :, k, 5, l] * rhogvz_mean[:, :, k-1, l]
                    ) * vmtr.VMTR_RGAMH[:, :, k, l]
                    + rhogw_mean[:, :, k, l] * vmtr.VMTR_RGSQRTH[:, :, k, l]
                ) * 0.5 * dt
            # end loop k 

            flx_v[:, :, kmin,   l] = 0.0
            flx_v[:, :, kmax+1, l] = 0.0 

            d[:, :, :, l] = b1 * frhog[:, :, :, l] / rhog_in[:, :, :, l] * dt

            for k in range(kmin, kmax+1):
                ck[:, :, k, l, 0] = -flx_v[:, :, k,   l] / rhog_in[:, :, k, l] * grd.GRD_rdgz[k]
                ck[:, :, k, l, 1] =  flx_v[:, :, k+1, l] / rhog_in[:, :, k, l] * grd.GRD_rdgz[k]
            # end loop k

            ck[:, :, kmin-1, l, 0] = 0.0
            ck[:, :, kmin-1, l, 1] = 0.0
            ck[:, :, kmax+1, l, 0] = 0.0
            ck[:, :, kmax+1, l, 1] = 0.0
        # end loop l

        if adm.ADM_have_pl:
            for l in range(lall_pl):
                for k in range(kmin + 1, kmax + 1):
                    for g in range(gall_pl):
                        flx_v_pl[g, k, l] = (
                            (
                                vmtr.VMTR_C2WfactGz_pl[g, k, 0, l] * rhogvx_mean_pl[g, k,   l] +
                                vmtr.VMTR_C2WfactGz_pl[g, k, 1, l] * rhogvx_mean_pl[g, k-1, l] +
                                vmtr.VMTR_C2WfactGz_pl[g, k, 2, l] * rhogvy_mean_pl[g, k,   l] +
                                vmtr.VMTR_C2WfactGz_pl[g, k, 3, l] * rhogvy_mean_pl[g, k-1, l] +
                                vmtr.VMTR_C2WfactGz_pl[g, k, 4, l] * rhogvz_mean_pl[g, k,   l] +
                                vmtr.VMTR_C2WfactGz_pl[g, k, 5, l] * rhogvz_mean_pl[g, k-1, l]
                            ) * vmtr.VMTR_RGAMH_pl[g, k, l] +
                            rhogw_mean_pl[g, k, l] * vmtr.VMTR_RGSQRTH_pl[g, k, l]
                        ) * 0.5 * dt

                flx_v_pl[:, kmin,   l] = 0.0
                flx_v_pl[:, kmax+1, l] = 0.0

                d_pl[:, :, l] = b1 * frhog_pl[:, :, l] / rhog_in_pl[:, :, l] * dt

                for k in range(kmin, kmax + 1):
                    ck_pl[:, k, l, 0] = -flx_v_pl[:, k,   l] / rhog_in_pl[:, k, l] * grd.GRD_rdgz[k]
                    ck_pl[:, k, l, 1] =  flx_v_pl[:, k+1, l] / rhog_in_pl[:, k, l] * grd.GRD_rdgz[k]

                ck_pl[:, kmin-1, l, 0] = 0.0
                ck_pl[:, kmin-1, l, 1] = 0.0
                ck_pl[:, kmax+1, l, 0] = 0.0
                ck_pl[:, kmax+1, l, 1] = 0.0
            # end loop l
        # endif

        #--- vertical advection: 2nd-order centered difference  



        return