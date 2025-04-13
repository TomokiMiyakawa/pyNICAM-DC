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


    def src_tracer_advection(self,
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
        grd_xc   = np.full(adm.ADM_shape + (AJ - AI + 1, ZDIR - XDIR +1,), cnst.CONST_undef, type=rdtype)                   # mass centroid position
        grd_xc_pl= np.full(adm.ADM_shape_pl + (ZDIR - XDIR +1,), cnst.CONST_undef, type=rdtype)

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
        for iq in range (vmax):

            for l in range(lall):
                for k in range(kall):
                    q[:, :, k, l] = rhogq[:, :, k, l, iq] / rhog_in[:, :, k, l]

                for k in range(kmin, kmax + 2):  # +2 to include kmax+1
                    q_h[:, :, k, l] = (
                        grd.GRD_afact[k] * q[:, :, k, l] +
                        grd.GRD_bfact[k] * q[:, :, k - 1, l]
                    )

                q_h[:, :, kmin - 1, l] = 0.0
            # end loop l

            if adm.ADM_have_pl:
                # Compute q_pl across all g and l at once
                q_pl = rhogq_pl[:, :, :, iq] / rhog_in_pl

                # Compute q_h_pl for k in [kmin, kmax+1]
                for k in range(kmin, kmax + 2):
                    q_h_pl[:, k, :] = (
                        grd.GRD_afact[k] * q_pl[:, k, :] +
                        grd.GRD_bfact[k] * q_pl[:, k - 1, :]
                    )

                # Boundary condition
                q_h_pl[:, kmin - 1, :] = 0.0
            #endif

            if apply_limiter_v(iq):
                self.vertical_limiter_thuburn( 
                    q_h[:,:,:,:],   q_h_pl[:,:,:],    # [INOUT]                                                                                          
                    q  [:,:,:,:],   q_pl  [:,:,:],    # [IN]                                                                 
                    d  [:,:,:,:],   d_pl  [:,:,:],    # [IN]                                                                 
                    ck [:,:,:,:,:], ck_pl [:,:,:,:]   # [IN] 
                    )                                                                 
            #endif        

            
            # --- update rhogq 

            for l in range(lall):
                # Zero out boundaries at kmin and kmax+1
                q_h[:, :, kmin, l] = 0.0
                q_h[:, :, kmax + 1, l] = 0.0

                # Update rhogq with flux divergence
                for k in range(kmin, kmax + 1):
                    rhogq[:, :, k, l, iq] -= (
                        flx_v[:, :, k + 1, l] * q_h[:, :, k + 1, l]
                        - flx_v[:, :, k,     l] * q_h[:, :, k,     l]
                    ) * grd.GRD_rdgz[k]

                # Zero out boundaries at kmin-1 and kmax+1
                rhogq[:, :, kmin - 1, l, iq] = 0.0
                rhogq[:, :, kmax + 1, l, iq] = 0.0


            if adm.ADM_have_pl:
                # Set q_h_pl boundaries
                q_h_pl[:, kmin,  :] = 0.0
                q_h_pl[:, kmax+1, :] = 0.0

                for k in range(kmin, kmax + 1):
                    rhogq_pl[:, k, :, iq] -= (
                        flx_v_pl[:, k + 1, :] * q_h_pl[:, k + 1, :] -
                        flx_v_pl[:, k    , :] * q_h_pl[:, k    , :]
                    ) * grd.GRD_rdgz[k]

                # Set rhogq_pl boundaries
                rhogq_pl[:, kmin - 1, :, iq] = 0.0
                rhogq_pl[:, kmax + 1, :, iq] = 0.0
            #endif

        # end loop iq

        #--- update rhog

        for l in range(lall):
            for k in range(kmin, kmax + 1):
                rhog[:, :, k, l] = (
                    rhog_in[:, :, k, l]
                    - (flx_v[:, :, k + 1, l] - flx_v[:, :, k, l]) * grd.GRD_rdgz[k]
                    + b1 * frhog[:, :, k, l] * dt
                )

            # Set boundaries at kmin-1 and kmax+1
            rhog[:, :, kmin - 1, l] = rhog_in[:, :, kmin, l]
            rhog[:, :, kmax + 1, l] = rhog_in[:, :, kmax, l]

        
        if adm.ADM_have_pl:
            for k in range(kmin, kmax + 1):
                rhog_pl[:, k, :] = (
                    rhog_in_pl[:, k, :]
                    - (flx_v_pl[:, k + 1, :] - flx_v_pl[:, k, :]) * grd.GRD_rdgz[k]
                    + b1 * frhog_pl[:, k, :] * dt
                )

            # Set boundaries at kmin-1 and kmax+1
            rhog_pl[:, kmin - 1, :] = rhog_in_pl[:, kmin, :]
            rhog_pl[:, kmax + 1, :] = rhog_in_pl[:, kmax, :]


        prf.PROF_rapend('____vertical_adv',2)
        #---------------------------------------------------------------------------
        # Horizontal advection by MIURA scheme
        #---------------------------------------------------------------------------
        prf.PROF_rapstart('____horizontal_adv',2)


        for l in range(lall):
            for k in range(kall):
                d[:, :, k, l] = b2 * frhog[:, :, k, l] / rhog[:, :, k, l] * dt

        for l in range(lall):
            for k in range(kall):
                rhogvx[:, :, k, l] = rhogvx_mean[:, :, k, l] * vmtr.VMTR_RGAM[:, :, k, l]
                rhogvy[:, :, k, l] = rhogvy_mean[:, :, k, l] * vmtr.VMTR_RGAM[:, :, k, l]
                rhogvz[:, :, k, l] = rhogvz_mean[:, :, k, l] * vmtr.VMTR_RGAM[:, :, k, l]


        if adm.ADM_have_pl:
            d_pl[:, :, :] = b2 * frhog_pl[:, :, :] / rhog_pl[:, :, :] * dt

            rhogvx_pl[:, :, :] = rhogvx_mean_pl[:, :, :] * vmtr.VMTR_RGAM_pl[:, :, :]
            rhogvy_pl[:, :, :] = rhogvy_mean_pl[:, :, :] * vmtr.VMTR_RGAM_pl[:, :, :]
            rhogvz_pl[:, :, :] = rhogvz_mean_pl[:, :, :] * vmtr.VMTR_RGAM_pl[:, :, :]

        # call horizontal_flux( flx_h    (:,:,:,:),   flx_h_pl    (:,:,:),   & ! [OUT]                                                                     
                        #   grd_xc   (:,:,:,:,:), grd_xc_pl   (:,:,:,:), & ! [OUT]                                                                     
                        #   rhog_mean(:,:,:),     rhog_mean_pl(:,:,:),   & ! [IN]                                                                      
                        #   rhogvx   (:,:,:),     rhogvx_pl   (:,:,:),   & ! [IN]                                                                      
                        #   rhogvy   (:,:,:),     rhogvy_pl   (:,:,:),   & ! [IN]                                                                      
                        #   rhogvz   (:,:,:),     rhogvz_pl   (:,:,:),   & ! [IN]                                                                      
                        #   dt                                           ) ! [IN]                                                                      

        #--- Courant number             

        for l in range(lall):
            for k in range(kall):
                ch[:, :, k, l, :] = flx_h[:, :, k, l, :] / rhog[:, :, k, l, None]
                cmask[:, :, k, l, :] = 0.5 - np.sign(0.5 - ch[:, :, k, l, :] + EPS)


        if adm.ADM_have_pl:
            g = adm.ADM_gslf_pl  # scalar index

            ch_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] = (
                flx_h_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] / rhog_pl[g, :, :]
            )

            cmask_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] = (
                0.5 - np.sign(0.5 - ch_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] + EPS)
            )


        for iq in range (vmax):

            for l in range(lall):
                for k in range(kall):
                    q[:, :, k, l] = rhogq[:, :, k, l, iq] / rhog[:, :, k, l]

            if adm.ADM_have_pl:
                q_pl[:, :, :] = rhogq_pl[:, :, :, iq] / rhog_pl[:, :, :]


            # calculate q at cell face, upwind side
            # call horizontal_remap( q_a   (:,:,:,:),   q_a_pl   (:,:,:),   & ! [OUT]
            #                       q     (:,:,:),     q_pl     (:,:,:),   & ! [IN]
            #                       cmask (:,:,:,:),   cmask_pl (:,:,:),   & ! [IN]
            #                       grd_xc(:,:,:,:,:), grd_xc_pl(:,:,:,:)  ) ! [IN]

            # apply flux limiter
            # if ( apply_limiter_h(iq) ) then
            #   call horizontal_limiter_thuburn( q_a  (:,:,:,:),   q_a_pl  (:,:,:), & ! [INOUT]
            #                                    q    (:,:,:),     q_pl    (:,:,:), & ! [IN]
            #                                    d    (:,:,:),     d_pl    (:,:,:), & ! [IN]
            #                                    ch   (:,:,:,:),   ch_pl   (:,:,:), & ! [IN]
            #                                    cmask(:,:,:,:),   cmask_pl(:,:,:)  ) ! [IN]
            # endif

            #--- update rhogq        

            for l in range(lall):
                for k in range(kall):
                    rhogq[:, :, k, l, iq] -= (
                        flx_h[:, :, k, l, 0] * q_a[:, :, k, l, 0] +
                        flx_h[:, :, k, l, 1] * q_a[:, :, k, l, 1] +
                        flx_h[:, :, k, l, 2] * q_a[:, :, k, l, 2] +
                        flx_h[:, :, k, l, 3] * q_a[:, :, k, l, 3] +
                        flx_h[:, :, k, l, 4] * q_a[:, :, k, l, 4] +
                        flx_h[:, :, k, l, 5] * q_a[:, :, k, l, 5]
                    )

            if adm.ADM_have_pl:
                g = adm.ADM_gslf_pl

                for l in range(lall_pl):
                    for k in range(kall):
                        for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                            rhogq_pl[g, k, l, iq] -= flx_h_pl[v, k, l] * q_a_pl[v, k, l]
            #endif

        #end iq LOOP

        #--- update rhog

        for l in range(lall):
            for k in range(kall):
                rhog[:, :, k, l] -= (
                    flx_h[:, :, k, l, 0] +
                    flx_h[:, :, k, l, 1] +
                    flx_h[:, :, k, l, 2] +
                    flx_h[:, :, k, l, 3] +
                    flx_h[:, :, k, l, 4] +
                    flx_h[:, :, k, l, 5]
                )
                rhog[:, :, k, l] += b2 * frhog[:, :, k, l] * dt


        if adm.ADM_have_pl:
            g = adm.ADM_gslf_pl  # Constant index for pole surface

            for l in range(lall_pl):
                for k in range(kall):
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        rhog_pl[g, k, l] -= flx_h_pl[v, k, l]

                    rhog_pl[g, k, l] += b2 * frhog_pl[g, k, l] * dt


        prf.PROF_rapend('____horizontal_adv',2)
        #---------------------------------------------------------------------------
        # Vertical Advection (fractioanl step) : 2nd
        #---------------------------------------------------------------------------
        prf.PROF_rapstart('____vertical_adv',2)

        for l in range(lall):
            d[:, :, :, l] = b3 * frhog[:, :, :, l] / rhog[:, :, :, l] * dt

            for k in range(kmin, kmax + 1):
                ck[:, :, k, l, 0] = -flx_v[:, :, k,   l] / rhog[:, :, k, l] * grd.GRD_rdgz[k]
                ck[:, :, k, l, 1] =  flx_v[:, :, k+1, l] / rhog[:, :, k, l] * grd.GRD_rdgz[k]

            ck[:, :, kmin - 1, l, 0] = 0.0
            ck[:, :, kmin - 1, l, 1] = 0.0
            ck[:, :, kmax + 1, l, 0] = 0.0
            ck[:, :, kmax + 1, l, 1] = 0.0

        if adm.ADM_have_pl:
            d_pl = b3 * frhog_pl / rhog_pl * dt  # fully vectorized over g, k, l

            for k in range(kmin, kmax + 1):
                ck_pl[:, k, :, 0] = -flx_v_pl[:, k,   :] / rhog_pl[:, k, :] * grd.GRD_rdgz[k]
                ck_pl[:, k, :, 1] =  flx_v_pl[:, k+1, :] / rhog_pl[:, k, :] * grd.GRD_rdgz[k]

            ck_pl[:, kmin - 1, :, 0] = 0.0
            ck_pl[:, kmin - 1, :, 1] = 0.0
            ck_pl[:, kmax + 1, :, 0] = 0.0
            ck_pl[:, kmax + 1, :, 1] = 0.0


        #--- vertical advection: 2nd-order centered difference
        for iq in range(vmax):

            for l in range(lall):
                # q = rhogq / rhog
                q[:, :, :, l] = rhogq[:, :, :, l, iq] / rhog[:, :, :, l]

                # q_h = a * q + b * q[-1]
                for k in range(kmin, kmax + 2):
                    q_h[:, :, k, l] = (
                        grd.GRD_afact[k] * q[:, :, k,   l] +
                        grd.GRD_bfact[k] * q[:, :, k-1, l]
                    )

                # Set boundary
                q_h[:, :, kmin - 1, l] = 0.0
            # end loop l

            if adm.ADM_have_pl:
                # q_pl = rhogq_pl / rhog_pl (element-wise division)
                q_pl = rhogq_pl[:, :, :, iq] / rhog_pl

                # q_h_pl = a * q_pl + b * q_pl (shifted k-1)
                q_h_pl[:, kmin:kmax+2, :] = (
                    grd.GRD_afact[kmin:kmax+2][None, :, None] * q_pl[:, kmin:kmax+2, :] +
                    grd.GRD_bfact[kmin:kmax+2][None, :, None] * q_pl[:, kmin-1:kmax+1, :]
                )

                # Boundary at kmin-1
                q_h_pl[:, kmin-1, :] = 0.0
            # endif

            #if apply_limiter_v(iq):
                #       call vertical_limiter_thuburn( q_h(:,:,:),   q_h_pl(:,:,:),  & ! [INOUT]
                #                                      q  (:,:,:),   q_pl  (:,:,:),  & ! [IN]
                #                                      d  (:,:,:),   d_pl  (:,:,:),  & ! [IN]
                #                                      ck (:,:,:,:), ck_pl (:,:,:,:) ) ! [IN]
            # endif

            #--- update rhogq

            for l in range(lall):
                q_h[:, :, kmin, l] = 0.0
                q_h[:, :, kmax+1, l] = 0.0

                for k in range(kmin, kmax+1):
                    rhogq[:, :, k, l, iq] -= (
                        flx_v[:, :, k+1, l] * q_h[:, :, k+1, l] -
                        flx_v[:, :, k,   l] * q_h[:, :, k,   l]
                    ) * grd.GRD_rdgz[k]

                rhogq[:, :, kmin-1, l, iq] = 0.0
                rhogq[:, :, kmax+1, l, iq] = 0.0

            

            if adm.ADM_have_pl:
                q_h_pl[:, kmin,   :] = 0.0
                q_h_pl[:, kmax+1, :] = 0.0

                for k in range(kmin, kmax+1):
                    rhogq_pl[:, k, :, iq] -= (
                        flx_v_pl[:, k+1, :] * q_h_pl[:, k+1, :] -
                        flx_v_pl[:, k,   :] * q_h_pl[:, k,   :]
                    ) * grd.GRD_rdgz[k]

                rhogq_pl[:, kmin-1, :, iq] = 0.0
                rhogq_pl[:, kmax+1, :, iq] = 0.0


            #--- tiny negative fixer

            for l in range(lall):
                for k in range(kmin, kmax + 1):
                    mask = (rhogq[:, :, k, l, iq] > -1.0e-10) & (rhogq[:, :, k, l, iq] < 0.0)
                    rhogq[:, :, k, l, iq][mask] = 0.0

            mask_pl = (rhogq_pl[..., iq] > -1.0e-10) & (rhogq_pl[..., iq] < 0.0)
            rhogq_pl[..., iq][mask_pl] = 0.0

        # end loop iq

        prf.PROF_rapend('____vertical_adv',2)

        return
    
    #> Prepare horizontal advection term: mass flux, grd_xc
    def horizontal_flux(self,
       flx_h,  flx_h_pl,      # [OUT]    # horizontal mass flux
       grd_xc, grd_xc_pl,     # [OUT]    # mass centroid position
       rho,    rho_pl,        # [IN]     # rho at cell center
       rhovx,  rhovx_pl,      # [IN]
       rhovy,  rhovy_pl,      # [IN]
       rhovz,  rhovz_pl,      # [IN]
       dt,
       cnst, grd, gmtr, rdtype,
    ):
    
        prf.PROF_rapstart('____horizontal_adv_flux',2)

        gmin = adm.ADM_gmin
        gmax = adm.ADM_gmax
        kall = adm.ADM_kall
        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        lall_pl = adm.ADM_lall_pl

        EPS = cnst.CONST_EPS

        TI  = adm.ADM_TI,  
        TJ  = adm.ADM_TJ,  
        AI  = adm.ADM_AI,  
        AIJ = adm.ADM_AIJ, 
        AJ  = adm.ADM_AJ,  
        K0  = adm.ADM_KNONE
 
        XDIR = grd.GRD_XDIR, 
        YDIR = grd.GRD_YDIR, 
        ZDIR = grd.GRD_ZDIR
    
        P_RAREA = gmtr.GMTR_p_RAREA, 
        T_RAREA = gmtr.GMTR_t_RAREA, 
        W1      = gmtr.GMTR_t_W1,    
        W2      = gmtr.GMTR_t_W2,    
        W3      = gmtr.GMTR_t_W3,    
        HNX     = gmtr.GMTR_A_HNX,   
        HNY     = gmtr.GMTR_A_HNY,   
        HNZ     = gmtr.GMTR_A_HNZ,   
        TNX     = gmtr.GMTR_A_TNX,   
        TNY     = gmtr.GMTR_A_TNY,   
        TNZ     = gmtr.GMTR_A_TNZ,   
        TN2X    = gmtr.GMTR_A_TN2X,  
        TN2Y    = gmtr.GMTR_A_TN2Y,  
        TN2Z    = gmtr.GMTR_A_TN2Z


        rhot_TI  = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF, type=rdtype)  # rho at cell vertex
        rhot_TJ  = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF, type=rdtype)  # rho at cell vertex
        rhovxt_TI= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF, type=rdtype)
        rhovxt_TJ= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF, type=rdtype)
        rhovyt_TI= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF, type=rdtype)
        rhovyt_TJ= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF, type=rdtype)
        rhovzt_TI= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF, type=rdtype)
        rhovzt_TJ= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF, type=rdtype)

        rhot_pl  = np.full((gall_pl,), cnst.CONST_UNDEF, type=rdtype)
        rhovxt_pl= np.full((gall_pl,), cnst.CONST_UNDEF, type=rdtype)
        rhovyt_pl= np.full((gall_pl,), cnst.CONST_UNDEF, type=rdtype)
        rhovzt_pl= np.full((gall_pl,), cnst.CONST_UNDEF, type=rdtype)


        for l in range(lall):
            for k in range(kall):

                isl = slice(0, iall - 1)
                jsl = slice(0, jall - 1)

                # Prepare slices for second term access
                isl_p = slice(1, iall)
                jsl_p = slice(1, jall)

                # First part: (i,j), (i+1,j)
                rhot_TI[isl, jsl, k]   = rho[isl, jsl, k, l]   * gmtr.GMTR_t[isl, jsl, K0, l, TI, W1] + rho[isl_p, jsl, k, l]   * gmtr.GMTR_t[isl, jsl, K0, l, TI, W2]
                rhovxt_TI[isl, jsl, k] = rhovx[isl, jsl, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TI, W1] + rhovx[isl_p, jsl, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TI, W2]
                rhovyt_TI[isl, jsl, k] = rhovy[isl, jsl, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TI, W1] + rhovy[isl_p, jsl, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TI, W2]
                rhovzt_TI[isl, jsl, k] = rhovz[isl, jsl, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TI, W1] + rhovz[isl_p, jsl, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TI, W2]

                rhot_TJ[isl, jsl, k]   = rho[isl, jsl, k, l]   * gmtr.GMTR_t[isl, jsl, K0, l, TJ, W1]
                rhovxt_TJ[isl, jsl, k] = rhovx[isl, jsl, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TJ, W1]
                rhovyt_TJ[isl, jsl, k] = rhovy[isl, jsl, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TJ, W1]
                rhovzt_TJ[isl, jsl, k] = rhovz[isl, jsl, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TJ, W1]

                # Second part: (i+1,j+1), (i,j+1)
                rhot_TI[isl, jsl, k]   += rho[isl_p, jsl_p, k, l]   * gmtr.GMTR_t[isl, jsl, K0, l, TI, W3]
                rhovxt_TI[isl, jsl, k] += rhovx[isl_p, jsl_p, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TI, W3]
                rhovyt_TI[isl, jsl, k] += rhovy[isl_p, jsl_p, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TI, W3]
                rhovzt_TI[isl, jsl, k] += rhovz[isl_p, jsl_p, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TI, W3]

                rhot_TJ[isl, jsl, k]   += rho[isl_p, jsl_p, k, l]   * gmtr.GMTR_t[isl, jsl, K0, l, TJ, W2] + rho[isl, jsl_p, k, l]   * gmtr.GMTR_t[isl, jsl, K0, l, TJ, W3]
                rhovxt_TJ[isl, jsl, k] += rhovx[isl_p, jsl_p, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TJ, W2] + rhovx[isl, jsl_p, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TJ, W3]
                rhovyt_TJ[isl, jsl, k] += rhovy[isl_p, jsl_p, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TJ, W2] + rhovy[isl, jsl_p, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TJ, W3]
                rhovzt_TJ[isl, jsl, k] += rhovz[isl_p, jsl_p, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TJ, W2] + rhovz[isl, jsl_p, k, l] * gmtr.GMTR_t[isl, jsl, K0, l, TJ, W3]


                if adm.ADM_have_sgp[l]:
                    rhot_TI[0, 0, k]   = rhot_TJ[1, 0, k]
                    rhovxt_TI[0, 0, k] = rhovxt_TJ[1, 0, k]
                    rhovyt_TI[0, 0, k] = rhovyt_TJ[1, 0, k]
                    rhovzt_TI[0, 0, k] = rhovzt_TJ[1, 0, k]


                flx_h[:, :, k, l, :].fill(0.0)      
                grd_xc[:, :, k, l, :, :].fill(0.0)       


                isl = slice(0, iall - 1)
                jsl = slice(1, jall - 1)

                rrhoa2 = 1.0 / np.maximum(
                    rhot_TJ[isl, jsl - 1, k] + rhot_TI[isl, jsl, k], EPS
                )
                rhovxt2 = rhovxt_TJ[isl, jsl - 1, k] + rhovxt_TI[isl, jsl, k]
                rhovyt2 = rhovyt_TJ[isl, jsl - 1, k] + rhovyt_TI[isl, jsl, k]
                rhovzt2 = rhovzt_TJ[isl, jsl - 1, k] + rhovzt_TI[isl, jsl, k]

                flux = 0.5 * (
                    rhovxt2 * gmtr.GMTR_a[isl, jsl, K0, l, AI, HNX] +
                    rhovyt2 * gmtr.GMTR_a[isl, jsl, K0, l, AI, HNY] +
                    rhovzt2 * gmtr.GMTR_a[isl, jsl, K0, l, AI, HNZ]
                )

                flx_h[isl, jsl, k, l, 1]  =  flux * gmtr.GMTR_p[isl, jsl, K0, l, P_RAREA] * dt
                flx_h[isl.start+1:isl.stop+1, jsl, k, l, 4] = -flux * gmtr.GMTR_p[isl.start+1:isl.stop+1, jsl, K0, l, P_RAREA] * dt

                grd_xc[isl, jsl, k, l, AI, XDIR] = grd.GRD_xr[isl, jsl, K0, l, AI, XDIR] - rhovxt2 * rrhoa2 * dt * 0.5
                grd_xc[isl, jsl, k, l, AI, YDIR] = grd.GRD_xr[isl, jsl, K0, l, AI, YDIR] - rhovyt2 * rrhoa2 * dt * 0.5
                grd_xc[isl, jsl, k, l, AI, ZDIR] = grd.GRD_xr[isl, jsl, K0, l, AI, ZDIR] - rhovzt2 * rrhoa2 * dt * 0.5



                isl = slice(0, iall - 1)
                jsl = slice(0, jall - 1)

                rrhoa2 = 1.0 / np.maximum(
                    rhot_TI[isl, jsl, k] + rhot_TJ[isl, jsl, k], EPS
                )
                rhovxt2 = rhovxt_TI[isl, jsl, k] + rhovxt_TJ[isl, jsl, k]
                rhovyt2 = rhovyt_TI[isl, jsl, k] + rhovyt_TJ[isl, jsl, k]
                rhovzt2 = rhovzt_TI[isl, jsl, k] + rhovzt_TJ[isl, jsl, k]

                flux = 0.5 * (
                    rhovxt2 * gmtr.GMTR_a[isl, jsl, K0, l, AIJ, HNX] +
                    rhovyt2 * gmtr.GMTR_a[isl, jsl, K0, l, AIJ, HNY] +
                    rhovzt2 * gmtr.GMTR_a[isl, jsl, K0, l, AIJ, HNZ]
                )

                flx_h[isl, jsl, k, l, 2] =  flux * gmtr.GMTR_p[isl, jsl, K0, l, P_RAREA] * dt
                flx_h[isl.start+1:isl.stop+1, jsl.start+1:jsl.stop+1, k, l, 5] = -flux * gmtr.GMTR_p[isl.start+1:isl.stop+1, jsl.start+1:jsl.stop+1, K0, l, P_RAREA] * dt

                grd_xc[isl, jsl, k, l, AIJ, XDIR] = grd.GRD_xr[isl, jsl, K0, l, AIJ, XDIR] - rhovxt2 * rrhoa2 * dt * 0.5
                grd_xc[isl, jsl, k, l, AIJ, YDIR] = grd.GRD_xr[isl, jsl, K0, l, AIJ, YDIR] - rhovyt2 * rrhoa2 * dt * 0.5
                grd_xc[isl, jsl, k, l, AIJ, ZDIR] = grd.GRD_xr[isl, jsl, K0, l, AIJ, ZDIR] - rhovzt2 * rrhoa2 * dt * 0.5


                isl = slice(1, iall - 1)
                jsl = slice(0, jall - 1)

                rrhoa2 = 1.0 / np.maximum(
                    rhot_TJ[isl, jsl, k] + rhot_TI[isl.start - 1:isl.stop - 1, jsl, k],
                    EPS
                )
                rhovxt2 = rhovxt_TJ[isl, jsl, k] + rhovxt_TI[isl.start - 1:isl.stop - 1, jsl, k]
                rhovyt2 = rhovyt_TJ[isl, jsl, k] + rhovyt_TI[isl.start - 1:isl.stop - 1, jsl, k]
                rhovzt2 = rhovzt_TJ[isl, jsl, k] + rhovzt_TI[isl.start - 1:isl.stop - 1, jsl, k]

                flux = 0.5 * (
                    rhovxt2 * gmtr.GMTR_a[isl, jsl, K0, l, AJ, HNX] +
                    rhovyt2 * gmtr.GMTR_a[isl, jsl, K0, l, AJ, HNY] +
                    rhovzt2 * gmtr.GMTR_a[isl, jsl, K0, l, AJ, HNZ]
                )

                flx_h[isl, jsl, k, l, 3] =  flux * gmtr.GMTR_p[isl, jsl, K0, l, P_RAREA] * dt
                flx_h[isl, jsl.start + 1:jsl.stop + 1, k, l, 6] = -flux * gmtr.GMTR_p[isl, jsl.start + 1:jsl.stop + 1, K0, l, P_RAREA] * dt

                grd_xc[isl, jsl, k, l, AJ, XDIR] = grd.GRD_xr[isl, jsl, K0, l, AJ, XDIR] - rhovxt2 * rrhoa2 * dt * 0.5
                grd_xc[isl, jsl, k, l, AJ, YDIR] = grd.GRD_xr[isl, jsl, K0, l, AJ, YDIR] - rhovyt2 * rrhoa2 * dt * 0.5
                grd_xc[isl, jsl, k, l, AJ, ZDIR] = grd.GRD_xr[isl, jsl, K0, l, AJ, ZDIR] - rhovzt2 * rrhoa2 * dt * 0.5


                if adm.ADM_have_sgp[l]:
                    flx_h(1,1,k,l,6) = 0.0   # really?

            # end loop k
        # end loop l

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(lall_pl):
                for k in range(kall):

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij = v
                        ijp1 = v + 1
                        if ijp1 == adm.ADM_gmax_pl + 1:
                            ijp1 = adm.ADM_gmin_pl

                        rhot_pl[v]   = rho_pl[n,    k, l] * gmtr.GMTR_t_pl[ij, K0, l, W1] + \
                                    rho_pl[ij,   k, l] * gmtr.GMTR_t_pl[ij, K0, l, W2] + \
                                    rho_pl[ijp1, k, l] * gmtr.GMTR_t_pl[ij, K0, l, W3]
                        rhovxt_pl[v] = rhovx_pl[n,    k, l] * gmtr.GMTR_t_pl[ij, K0, l, W1] + \
                                    rhovx_pl[ij,   k, l] * gmtr.GMTR_t_pl[ij, K0, l, W2] + \
                                    rhovx_pl[ijp1, k, l] * gmtr.GMTR_t_pl[ij, K0, l, W3]
                        rhovyt_pl[v] = rhovy_pl[n,    k, l] * gmtr.GMTR_t_pl[ij, K0, l, W1] + \
                                    rhovy_pl[ij,   k, l] * gmtr.GMTR_t_pl[ij, K0, l, W2] + \
                                    rhovy_pl[ijp1, k, l] * gmtr.GMTR_t_pl[ij, K0, l, W3]
                        rhovzt_pl[v] = rhovz_pl[n,    k, l] * gmtr.GMTR_t_pl[ij, K0, l, W1] + \
                                    rhovz_pl[ij,   k, l] * gmtr.GMTR_t_pl[ij, K0, l, W2] + \
                                    rhovz_pl[ijp1, k, l] * gmtr.GMTR_t_pl[ij, K0, l, W3]
                    # end loop v

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij = v
                        ijm1 = v - 1
                        if ijm1 == adm.ADM_gmin_pl - 1:
                            ijm1 = adm.ADM_gmax_pl

                        rrhoa2  = 1.0 / max(rhot_pl[ijm1] + rhot_pl[ij], EPS)
                        rhovxt2 = rhovxt_pl[ijm1] + rhovxt_pl[ij]
                        rhovyt2 = rhovyt_pl[ijm1] + rhovyt_pl[ij]
                        rhovzt2 = rhovzt_pl[ijm1] + rhovzt_pl[ij]

                        flux = 0.5 * (
                            rhovxt2 * gmtr.GMTR_a_pl[ij, K0, l, HNX] +
                            rhovyt2 * gmtr.GMTR_a_pl[ij, K0, l, HNY] +
                            rhovzt2 * gmtr.GMTR_a_pl[ij, K0, l, HNZ]
                        )

                        flx_h_pl[v, k, l] = flux * gmtr.GMTR_p_pl[n, K0, l, P_RAREA] * dt

                        grd_xc_pl[v, k, l, XDIR] = grd.GRD_xr_pl[v, K0, l, XDIR] - rhovxt2 * rrhoa2 * dt * 0.5
                        grd_xc_pl[v, k, l, YDIR] = grd.GRD_xr_pl[v, K0, l, YDIR] - rhovyt2 * rrhoa2 * dt * 0.5
                        grd_xc_pl[v, k, l, ZDIR] = grd.GRD_xr_pl[v, K0, l, ZDIR] - rhovzt2 * rrhoa2 * dt * 0.5
                    # end loop v

                # end loop k
            # end loop l
        # endif

        prf.PROF_rapend  ('____horizontal_adv_flux',2)

        return


    def vertical_limiter_thuburn(self, q_h, q_h_pl, q, q_pl, d, d_pl, ck, ck_pl):
        # Vertical limiter for Thuburn scheme
        # q_h: [INOUT] q at layer face
        # q_h_pl: [INOUT] q at layer face (pl)
        # q: [IN] q at cell center
        # q_pl: [IN] q at cell center (pl)
        # d: [IN] hyperviscosity tendency for rhog
        # d_pl: [IN] hyperviscosity tendency for rhog (pl)
        # ck: [IN] Courant number
        # ck_pl: [IN] Courant number (pl)

        pass