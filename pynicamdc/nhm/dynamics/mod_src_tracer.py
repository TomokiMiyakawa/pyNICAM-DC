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
       cnst, comm, grd, gmtr, oprt, vmtr, rdtype,
    ):

        TI  = adm.ADM_TI  
        TJ  = adm.ADM_TJ  
        AI  = adm.ADM_AI  
        AIJ = adm.ADM_AIJ
        AJ  = adm.ADM_AJ  
        K0  = adm.ADM_K0

        XDIR = grd.GRD_XDIR 
        YDIR = grd.GRD_YDIR 
        ZDIR = grd.GRD_ZDIR

        rhog     = np.full(adm.ADM_shape, cnst.CONST_UNDEF)
        rhog_pl  = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        rhogvx   = np.full(adm.ADM_shape, cnst.CONST_UNDEF)
        rhogvx_pl= np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        rhogvy   = np.full(adm.ADM_shape, cnst.CONST_UNDEF)
        rhogvy_pl= np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        rhogvz   = np.full(adm.ADM_shape, cnst.CONST_UNDEF)
        rhogvz_pl= np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)

        q        = np.full(adm.ADM_shape, cnst.CONST_UNDEF)
        q_pl     = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        d        = np.full(adm.ADM_shape, cnst.CONST_UNDEF)
        d_pl     = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)

        q_h      = np.full(adm.ADM_shape, cnst.CONST_UNDEF)          # q at layer face
        q_h_pl   = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        flx_v    = np.full(adm.ADM_shape, cnst.CONST_UNDEF)          # mass flux
        flx_v_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        ck       = np.full(adm.ADM_shape +(2,), cnst.CONST_UNDEF)    # Courant number
        ck_pl    = np.full(adm.ADM_shape_pl +(2,), cnst.CONST_UNDEF)

        q_a      = np.full(adm.ADM_shape +(6,), cnst.CONST_UNDEF)    # q at cell face
        q_a_pl   = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        flx_h    = np.full(adm.ADM_shape +(6,), cnst.CONST_UNDEF)    # mass flux
        flx_h_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        ch       = np.full(adm.ADM_shape +(6,), cnst.CONST_UNDEF)    # Courant number
        ch_pl    = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        cmask    = np.full(adm.ADM_shape +(6,), cnst.CONST_UNDEF)    # upwind direction mask
        cmask_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        grd_xc   = np.full(adm.ADM_shape + (AJ - AI + 1, ZDIR - XDIR +1,), cnst.CONST_UNDEF)                   # mass centroid position
        grd_xc_pl= np.full(adm.ADM_shape_pl + (ZDIR - XDIR +1,), cnst.CONST_UNDEF)

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

        b1 = rdtype(0.0)
        b2 = rdtype(1.0)
        b3 = rdtype(1.0) - (b1+b2)

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

        kslice     = slice(kmin + 1, kmax + 1)
        kslice_m1  = slice(kmin    , kmax    )
        kslice_p1  = slice(kmin + 2, kmax + 2)

        # ---- flx_v computation (vectorized over k and l) ----
        flx_v[:, :, kslice, :] = (
            (
                vmtr.VMTR_C2WfactGz[:, :, kslice, :, 0] * rhogvx_mean[:, :, kslice, :] +
                vmtr.VMTR_C2WfactGz[:, :, kslice, :, 1] * rhogvx_mean[:, :, kslice_m1, :] +
                vmtr.VMTR_C2WfactGz[:, :, kslice, :, 2] * rhogvy_mean[:, :, kslice, :] +
                vmtr.VMTR_C2WfactGz[:, :, kslice, :, 3] * rhogvy_mean[:, :, kslice_m1, :] +
                vmtr.VMTR_C2WfactGz[:, :, kslice, :, 4] * rhogvz_mean[:, :, kslice, :] +
                vmtr.VMTR_C2WfactGz[:, :, kslice, :, 5] * rhogvz_mean[:, :, kslice_m1, :]
            ) * vmtr.VMTR_RGAMH[:, :, kslice, :] +
            rhogw_mean[:, :, kslice, :] * vmtr.VMTR_RGSQRTH[:, :, kslice, :]
        ) * rdtype(0.5) * dt

        # ---- flx_v vertical boundaries ----
        flx_v[:, :, kmin,   :] = rdtype(0.0)
        flx_v[:, :, kmax+1, :] = rdtype(0.0)

        # ---- d computation ----
        d[:, :, :, :] = b1 * frhog / rhog_in * dt

        # ---- ck computation ----
        # ck[..., 0]
        ck[:, :, kmin:kmax+1, :, 0] = -flx_v[:, :, kmin:kmax+1, :] / rhog_in[:, :, kmin:kmax+1, :] * grd.GRD_rdgz[kmin:kmax+1][None, None, :, None]

        # ck[..., 1]
        ck[:, :, kmin:kmax+1, :, 1] =  flx_v[:, :, kmin+1:kmax+2, :] / rhog_in[:, :, kmin:kmax+1, :] * grd.GRD_rdgz[kmin:kmax+1][None, None, :, None]

        # ---- ck vertical boundaries ----
        ck[:, :, kmin-1, :, 0] = rdtype(0.0)
        ck[:, :, kmin-1, :, 1] = rdtype(0.0)
        ck[:, :, kmax+1, :, 0] = rdtype(0.0)
        ck[:, :, kmax+1, :, 1] = rdtype(0.0)




        # for l in range(lall):
        #     for k in range(kmin+1, kmax+1):
        #        flx_v[:, :, k, l] = (
        #             (
        #                 vmtr.VMTR_C2WfactGz[:, :, k, l, 0] * rhogvx_mean[:, :, k,   l] +
        #                 vmtr.VMTR_C2WfactGz[:, :, k, l, 1] * rhogvx_mean[:, :, k-1, l] +
        #                 vmtr.VMTR_C2WfactGz[:, :, k, l, 2] * rhogvy_mean[:, :, k,   l] +
        #                 vmtr.VMTR_C2WfactGz[:, :, k, l, 3] * rhogvy_mean[:, :, k-1, l] +
        #                 vmtr.VMTR_C2WfactGz[:, :, k, l, 4] * rhogvz_mean[:, :, k,   l] +
        #                 vmtr.VMTR_C2WfactGz[:, :, k, l, 5] * rhogvz_mean[:, :, k-1, l]
        #             ) * vmtr.VMTR_RGAMH[:, :, k, l]
        #             + rhogw_mean[:, :, k, l] * vmtr.VMTR_RGSQRTH[:, :, k, l]
        #         ) * rdtype(0.5) * dt
        #     # end loop k 

        #     flx_v[:, :, kmin,   l] = rdtype(0.0)
        #     flx_v[:, :, kmax+1, l] = rdtype(0.0) 

        #     d[:, :, :, l] = b1 * frhog[:, :, :, l] / rhog_in[:, :, :, l] * dt

        #     for k in range(kmin, kmax+1):
        #         ck[:, :, k, l, 0] = -flx_v[:, :, k,   l] / rhog_in[:, :, k, l] * grd.GRD_rdgz[k]
        #         ck[:, :, k, l, 1] =  flx_v[:, :, k+1, l] / rhog_in[:, :, k, l] * grd.GRD_rdgz[k]
        #     # end loop k

        #     ck[:, :, kmin-1, l, 0] = rdtype(0.0)
        #     ck[:, :, kmin-1, l, 1] = rdtype(0.0)
        #     ck[:, :, kmax+1, l, 0] = rdtype(0.0)
        #     ck[:, :, kmax+1, l, 1] = rdtype(0.0)
        # # end loop l

        if adm.ADM_have_pl:
            for l in range(lall_pl):
                for k in range(kmin + 1, kmax + 1):
                    for g in range(gall_pl):
                        flx_v_pl[g, k, l] = (
                            (
                                vmtr.VMTR_C2WfactGz_pl[g, k, l, 0] * rhogvx_mean_pl[g, k,   l] +
                                vmtr.VMTR_C2WfactGz_pl[g, k, l, 1] * rhogvx_mean_pl[g, k-1, l] +
                                vmtr.VMTR_C2WfactGz_pl[g, k, l, 2] * rhogvy_mean_pl[g, k,   l] +
                                vmtr.VMTR_C2WfactGz_pl[g, k, l, 3] * rhogvy_mean_pl[g, k-1, l] +
                                vmtr.VMTR_C2WfactGz_pl[g, k, l, 4] * rhogvz_mean_pl[g, k,   l] +
                                vmtr.VMTR_C2WfactGz_pl[g, k, l, 5] * rhogvz_mean_pl[g, k-1, l]
                            ) * vmtr.VMTR_RGAMH_pl[g, k, l] +
                            rhogw_mean_pl[g, k, l] * vmtr.VMTR_RGSQRTH_pl[g, k, l]
                        ) * rdtype(0.5) * dt

                flx_v_pl[:, kmin,   l] = rdtype(0.0)
                flx_v_pl[:, kmax+1, l] = rdtype(0.0)

                d_pl[:, :, l] = b1 * frhog_pl[:, :, l] / rhog_in_pl[:, :, l] * dt

                for k in range(kmin, kmax + 1):
                    ck_pl[:, k, l, 0] = -flx_v_pl[:, k,   l] / rhog_in_pl[:, k, l] * grd.GRD_rdgz[k]
                    ck_pl[:, k, l, 1] =  flx_v_pl[:, k+1, l] / rhog_in_pl[:, k, l] * grd.GRD_rdgz[k]

                ck_pl[:, kmin-1, l, 0] = rdtype(0.0)
                ck_pl[:, kmin-1, l, 1] = rdtype(0.0)
                ck_pl[:, kmax+1, l, 0] = rdtype(0.0)
                ck_pl[:, kmax+1, l, 1] = rdtype(0.0)
            # end loop l
        # endif

        #--- vertical advection: 2nd-order centered difference  
        for iq in range (vmax):

            with open(std.fname_log, 'a') as log_file: 
                print("rhogq prep, 6531, iq= ", iq, rhogq[6,5,:4,1,iq],file=log_file)

            for l in range(lall):
                for k in range(kall):
                    q[:, :, k, l] = rhogq[:, :, k, l, iq] / rhog_in[:, :, k, l]

                for k in range(kmin, kmax + 2):  # +2 to include kmax+1
                    q_h[:, :, k, l] = (
                        grd.GRD_afact[k] * q[:, :, k, l] +
                        grd.GRD_bfact[k] * q[:, :, k - 1, l]
                    )
                    if k==3 and l==1:
                        with open(std.fname_log, 'a') as log_file: 
                            print("q_h DEFINE, 6531, iq= ", iq, rhogq[6,5,3,1,iq],file=log_file)
                            print("  q  (k and k-1)", q[6,5,3,1], q[6,5,2,1], file=log_file)
                            print("  abfact", grd.GRD_afact[k], grd.GRD_bfact[k], file=log_file)

                q_h[:, :, kmin - 1, l] = rdtype(0.0)

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
                q_h_pl[:, kmin - 1, :] = rdtype(0.0)
            #endif

            with open(std.fname_log, 'a') as log_file: 
                print("q_h before vlimiter, 6531", iq, q_h[6,5,3,1],file=log_file)
            if apply_limiter_v[iq]:
                self.vertical_limiter_thuburn( 
                    q_h[:,:,:,:],   q_h_pl[:,:,:],    # [INOUT]        #q_h [6,5,3,1] of rank6   -0.006997044776120031 compared to     0.900866536517581  in original                                                                           
                    q  [:,:,:,:],   q_pl  [:,:,:],    # [IN]                                                                 
                    d  [:,:,:,:],   d_pl  [:,:,:],    # [IN]                                                                 
                    ck [:,:,:,:,:], ck_pl [:,:,:,:],   # [IN] 
                    cnst, rdtype,
                    )     
            with open(std.fname_log, 'a') as log_file: 
                print("q_h after vlimiter, 6531", iq, q_h[6,5,3,1],file=log_file)                                                            
            
            # --- update rhogq 

            for l in range(lall):
                # Zero out boundaries at kmin and kmax+1
                q_h[:, :, kmin, l] = rdtype(0.0)
                q_h[:, :, kmax + 1, l] = rdtype(0.0)

                # Update rhogq with flux divergence
                for k in range(kmin, kmax + 1):
                    rhogq[:, :, k, l, iq] -= (
                        flx_v[:, :, k + 1, l] * q_h[:, :, k + 1, l]
                        - flx_v[:, :, k,     l] * q_h[:, :, k,     l]
                    ) * grd.GRD_rdgz[k]

                    # if k==3 and l==1:
                    #     with open(std.fname_log, 'a') as log_file: 
                    #         print(f"STC0.8: rhogq [6,5,{k},{l},:]", rhogq[6, 5, k, l, :], file=log_file)
                    #         print(f"STC0.8: flx_v [6,5,{k+1},{l}]", flx_v[6, 5, k+1, l], file=log_file)
                    #         print(f"STC0.8: flx_v [6,5,{k},{l}]  ", flx_v[6, 5, k, l], file=log_file)
                    #         print(f"STC0.8:   q_h [6,5,{k+1},{l}]", q_h[6, 5, k+1, l], file=log_file)
                    #         print(f"STC0.8:   q_h [6,5,{k},{l}]  ", q_h[6, 5, k, l], file=log_file)   #q_h [6,5,3,1]   -0.006997044776120031 compared to     0.900866536517581  in original   
                    #         print(f"STC0.8:   q [6,5,{k},{l}]    ", q[6, 5, k, l], file=log_file)
                    #         print(f"STC0.8:   d [6,5,{k},{l}]    ", d[6, 5, k, l], file=log_file)
                    #         print(f"STC0.8:   ck [6,5,{k},{l},:] ", ck[6, 5, k, l, :], file=log_file)
                    #         print(f"STC0.8:   grd.GRD_rdgz [{k}] ", grd.GRD_rdgz[k], file=log_file)

                # Zero out boundaries at kmin-1 and kmax+1
                rhogq[:, :, kmin - 1, l, iq] = rdtype(0.0)
                rhogq[:, :, kmax + 1, l, iq] = rdtype(0.0)


            if adm.ADM_have_pl:
                # Set q_h_pl boundaries
                q_h_pl[:, kmin,  :] = rdtype(0.0)
                q_h_pl[:, kmax+1, :] = rdtype(0.0)

                for k in range(kmin, kmax + 1):
                    rhogq_pl[:, k, :, iq] -= (
                        flx_v_pl[:, k + 1, :] * q_h_pl[:, k + 1, :] -
                        flx_v_pl[:, k    , :] * q_h_pl[:, k    , :]
                    ) * grd.GRD_rdgz[k]

                # Set rhogq_pl boundaries
                rhogq_pl[:, kmin - 1, :, iq] = rdtype(0.0)
                rhogq_pl[:, kmax + 1, :, iq] = rdtype(0.0)
            #endif

        # end loop iq

        with open(std.fname_log, 'a') as log_file:
        #     print("STA1:rhogq[0,0,6,1,:]  ", rhogq[0, 0, 6, 1, :], file=log_file)    # 0, 0 is off at step 1 (after step 0))
        #     print("     rhogq[0,0,7,1,:]  ", rhogq[0, 0, 7, 1, :], file=log_file)
        #     print("     rhogq[1,1,6,1,:]  ", rhogq[1, 1, 6, 1, :], file=log_file)
        #     print("     rhogq[1,1,7,1,:]  ", rhogq[1, 1, 7, 1, :], file=log_file)
        #     print("     rhogq[1,1,5,1,:]  ", rhogq[1, 1, 5, 1, :], file=log_file)
        #     print("     rhogq[1,1,8,1,:]  ", rhogq[1, 1, 8, 1, :], file=log_file)

            print("STB1:rhogq [6,5,10,0,:]  ", rhogq[6, 5, 10, 0, :], file=log_file)
        #     print("    :rhogq_pl[0,10,0,:]  ", rhogq_pl[0, 10, 0, :], file=log_file)
        #     print("    :rhogq_pl[1,10,0,:]  ", rhogq_pl[1, 10, 0, :], file=log_file)
        #     print("    :rhogq_pl[2,10,0,:]  ", rhogq_pl[2, 10, 0, :], file=log_file)

            print("STC1:rhogq [6,5,3,1,:]  ", rhogq[6, 5, 3, 1, :], file=log_file)
            print("STD1:rhogq [6,5,2,1,:]  ", rhogq[6, 5, 2, 1, :], file=log_file)
        # if adm.ADM_have_pl:
        #     print("rhogq_pl.shape", rhogq_pl.shape)
        #     print(rhogq_pl[0,3,0,0])

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


        #for l in range(lall):
        #    for k in range(kall):
        d[:, :, :, :] = b2 * frhog[:, :, :, :] / rhog[:, :, :, :] * dt

        #for l in range(lall):
        #    for k in range(kall):
        rhogvx[:, :, :, :] = rhogvx_mean[:, :, :, :] * vmtr.VMTR_RGAM[:, :, :, :]
        rhogvy[:, :, :, :] = rhogvy_mean[:, :, :, :] * vmtr.VMTR_RGAM[:, :, :, :]
        rhogvz[:, :, :, :] = rhogvz_mean[:, :, :, :] * vmtr.VMTR_RGAM[:, :, :, :]


        if adm.ADM_have_pl:
            d_pl[:, :, :] = b2 * frhog_pl[:, :, :] / rhog_pl[:, :, :] * dt

            rhogvx_pl[:, :, :] = rhogvx_mean_pl[:, :, :] * vmtr.VMTR_RGAM_pl[:, :, :]
            rhogvy_pl[:, :, :] = rhogvy_mean_pl[:, :, :] * vmtr.VMTR_RGAM_pl[:, :, :]
            rhogvz_pl[:, :, :] = rhogvz_mean_pl[:, :, :] * vmtr.VMTR_RGAM_pl[:, :, :]

        self.horizontal_flux(
            flx_h, flx_h_pl,            # [OUT]
            grd_xc, grd_xc_pl,          # [OUT]   grd_xc for AIJ and AJ broken?
            rhog_mean, rhog_mean_pl,    # [IN]
            rhogvx, rhogvx_pl,          # [IN]
            rhogvy, rhogvy_pl,          # [IN]
            rhogvz, rhogvz_pl,          # [IN]
            dt,                         # [IN]
            cnst, grd, gmtr, rdtype,
        )


        #--- Courant number             
        # for l in range(lall):
        #     for k in range(kall):
        ch[:, :, :, :, :] = flx_h[:, :, :, :, :] / rhog[:, :, :, :, None]
        cmask[:, :, :, :, :] = rdtype(0.5) - np.copysign(rdtype(0.5), ch[:, :, :, :, :] - EPS)
                #cmask[:, :, k, l, :] = rdtype(0.5) - np.sign(rdtype(0.5) - ch[:, :, k, l, :] + EPS)


        if adm.ADM_have_pl:
            g = adm.ADM_gslf_pl  # scalar index

            ch_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] = (
                flx_h_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] / rhog_pl[g, :, :]
            )

            # cmask_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] = (
            #     rdtype(0.5) - np.sign(rdtype(0.5) - ch_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] + EPS)
            # )
            cmask_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] = (
                rdtype(0.5) - np.copysign(rdtype(0.5), ch_pl[adm.ADM_gmin_pl:adm.ADM_gmax_pl+1, :, :] + EPS)
            )


        for iq in range (vmax):

            # for l in range(lall):
            #     for k in range(kall):
            q[:, :, :, :] = rhogq[:, :, :, :, iq] / rhog[:, :, :, :]

            if adm.ADM_have_pl:
                q_pl[:, :, :] = rhogq_pl[:, :, :, iq] / rhog_pl[:, :, :]

            with open(std.fname_log, 'a') as log_file:
                #print("STC1.3:q_a[6,5,3,1,:]  ", q_a[6, 5, 3, 1, :], file=log_file)

                #print(f"STE1.2:rhogq[16,:,24,1,iq={iq}]", q_a[16,:,24,1,iq]  , file=log_file)
                #print("STE1.2:rhog[16,:,24,1]", rhog[16,:,24,1]  , file=log_file)
                print("STE1.2:q[16,:,24,1]", q[16,:,24,1]  , file=log_file)
                # print("STE1.2:cmask[16,:,24,1,0]", cmask[16,:,24,1,0]  , file=log_file)
                # print("STE1.2:ch[16,:,24,1,0] + EPS", ch[16,:,24,1,0]+EPS  , file=log_file)
                # print("STE1.2:cmask[16,:,24,1,1]", cmask[16,:,24,1,1]  , file=log_file)
                # print("STE1.2:ch[16,:,24,1,1] + EPS", ch[16,:,24,1,1]+EPS  , file=log_file)
                # print("STE1.2:cmask[16,:,24,1,2]", cmask[16,:,24,1,2]  , file=log_file)
                # print("STE1.2:ch[16,:,24,1,2] + EPS", ch[16,:,24,1,2]+EPS  , file=log_file)
                # print("STE1.2:cmask[16,:,24,1,3]", cmask[16,:,24,1,3]  , file=log_file)
                # print("STE1.2:ch[16,:,24,1,3] + EPS", ch[16,:,24,1,3]+EPS  , file=log_file)
                # print("STE1.2:cmask[16,:,24,1,4]", cmask[16,:,24,1,4]  , file=log_file)
                # print("STE1.2:ch[16,:,24,1,4] + EPS", ch[16,:,24,1,4]+EPS  , file=log_file)
                # print("STE1.2:cmask[16,:,24,1,5]", cmask[16,:,24,1,5]  , file=log_file)
                # print("STE1.2:ch[16,:,24,1,5] + EPS", ch[16,:,24,1,5]+EPS  , file=log_file)

                # print("STE1.2:grd_xc[16,:,24,1,0,0]", grd_xc[16,:,24,1,0,0]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,0,1]", grd_xc[16,:,24,1,0,1]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,0,2]", grd_xc[16,:,24,1,0,2]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,1,0]", grd_xc[16,:,24,1,1,0]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,1,1]", grd_xc[16,:,24,1,1,1]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,1,2]", grd_xc[16,:,24,1,1,2]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,2,0]", grd_xc[16,:,24,1,2,0]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,2,1]", grd_xc[16,:,24,1,2,1]  , file=log_file)
                # print("STE1.2:grd_xc[16,:,24,1,2,2]", grd_xc[16,:,24,1,2,2]  , file=log_file)

            # calculate q at cell face, upwind side
            self.horizontal_remap(
                q_a, q_a_pl,            # [OUT]
                q,   q_pl,              # [IN]
                cmask, cmask_pl,        # [IN]
                grd_xc, grd_xc_pl,      # [IN]
                cnst, comm, grd, oprt, rdtype,
            )

            with open(std.fname_log, 'a') as log_file:
                #print("STC1.3:q_a[6,5,3,1,:]  ", q_a[6, 5, 3, 1, :], file=log_file)
                print("STE1.3:q_a[16,:,24,1,1]", q_a[16,:,24,1,1]  , file=log_file)
                #print("STD1.3:q_a[6,5,2,1,:]  ", q_a[6, 5, 2, 1, :], file=log_file)
            #     print("STA1.3 :  q_a[0,0,7,1,:]  ",   q_a[0, 0, 7, 1, :], file=log_file)  # 0.
            #     print("            q[0,0,7,1]    ",   q  [0, 0, 7, 1]   , file=log_file)  # 0.
            #     print("          q_a[1,1,6,1,:]  ",   q_a[1, 1, 6, 1, :], file=log_file)  # 0.
            #     print("            q[1,1,6,1]    ",   q  [1, 1, 6, 1]   , file=log_file)  # 0.
            #     print("          q_a[1,1,7,1,:]  ",   q_a[1, 1, 7, 1, :], file=log_file)  # 0.
            #     print("            q[1,1,7,1]    ",   q  [1, 1, 7, 1]   , file=log_file)  # 0.

            # if adm.ADM_have_pl:
            #     print("q_a_pl")
            #     print(q_a_pl[:,3,0])

            # apply flux limiter
            if apply_limiter_h[iq]:
                self.horizontal_limiter_thuburn(
                    q_a, q_a_pl,            # [INOUT]    #  1 1 6 1 and 1 1 7 1 in SP get undefs out of here 
                    q,   q_pl,              # [IN]
                    d,   d_pl,              # [IN]
                    ch,  ch_pl,             # [IN]
                    cmask, cmask_pl,        # [IN]
                    cnst, comm, rdtype,
                )
            # endif

            # with open(std.fname_log, 'a') as log_file:
            # #     print("STA1.4 :  q_a[0,0,7,1,:]  ",   q_a[0, 0, 7, 1, :], file=log_file)  # 0, 1, 2 are undef
            # #     print("            q[0,0,7,1]    ",   q  [0, 0, 7, 1]   , file=log_file)  # 0.
            # #     print("          q_a[1,1,7,1,:]  ",   q_a[1, 1, 7, 1, :], file=log_file)  # 4 is undef
            # #     print("            q[1,1,7,1]    ",   q  [1, 1, 7, 1]   , file=log_file)  # 0.
            #     print("STA1.4 :  q_a[1,1,6,1,:]  ",   q_a[0, 0, 7, 1, :], file=log_file)  # 0.
            #     #print("            q[0,0,7,1]    ",   q  [0, 0, 7, 1]   , file=log_file)  # 0.
            #     print("          q_a[1,1,7,1,:]  ",   q_a[1, 1, 7, 1, :], file=log_file)  # 0.
            #     print("            q[1,1,6,1]    ",   q  [1, 1, 7, 1]   , file=log_file)  # 0.
            #     print("            q[1,1,7,1]    ",   q  [1, 1, 7, 1]   , file=log_file)  # 0.



            #--- update rhogq        

            # for l in range(lall):
            #     for k in range(kall):
            # rhogq[:, :, :, :, iq] -= (
            #     flx_h[:, :, :, :, 0] * q_a[:, :, :, :, 0] +
            #     flx_h[:, :, :, :, 1] * q_a[:, :, :, :, 1] +
            #     flx_h[:, :, :, :, 2] * q_a[:, :, :, :, 2] +
            #     flx_h[:, :, :, :, 3] * q_a[:, :, :, :, 3] +
            #     flx_h[:, :, :, :, 4] * q_a[:, :, :, :, 4] +
            #     flx_h[:, :, :, :, 5] * q_a[:, :, :, :, 5]
            # )

            # Prepare slices for i=2:iall-1, j=2:jall-1
            isl = slice(1, iall-1)
            jsl = slice(1, jall-1)

            # Fully vectorized calculation
            rhogq[isl, jsl, :, :, iq] -= (
                flx_h[isl, jsl, :, :, 0] * q_a[isl, jsl, :, :, 0] +
                flx_h[isl, jsl, :, :, 1] * q_a[isl, jsl, :, :, 1] +
                flx_h[isl, jsl, :, :, 2] * q_a[isl, jsl, :, :, 2] +
                flx_h[isl, jsl, :, :, 3] * q_a[isl, jsl, :, :, 3] +
                flx_h[isl, jsl, :, :, 4] * q_a[isl, jsl, :, :, 4] +
                flx_h[isl, jsl, :, :, 5] * q_a[isl, jsl, :, :, 5]
            )



            with open(std.fname_log, 'a') as log_file:
            #     print(f"iq=  {iq} ",file=log_file)
            #     print("STA1.5 :rhogq[0,0,7,1,:]  ", rhogq[0, 0, 7, 1, :], file=log_file)  #you  e+23
            #     print("        rhogq[1,1,7,1,:]  ", rhogq[1, 1, 7, 1, :], file=log_file)  #you  e+23
            #     print("        flx_h[0,0,7,1,:]  ", flx_h[0, 0, 7, 1, :], file=log_file)  
            #     print("          q_a[0,0,7,1,:]  ",   q_a[0, 0, 7, 1, :], file=log_file)  # 0, 1, 2 are undef
            #     print("            q[0,0,7,1]    ",   q  [0, 0, 7, 1]   , file=log_file)
            #     print("        flx_h[1,1,7,1,:]  ", flx_h[1, 1, 7, 1, :], file=log_file)  
            #     print("          q_a[1,1,7,1,:]  ",   q_a[1, 1, 7, 1, :], file=log_file)  # 4 is undef
            #     print("            q[1,1,7,1]    ",   q  [1, 1, 7, 1]   , file=log_file)

                print("STB1.5 :rhogq[6,5,10,0,:] ", rhogq[6, 5, 10, 0, :], file=log_file)  #you  e+23
            #     print("        flx_h[6,5,10,0,:] ", flx_h[6, 5, 10, 0, :], file=log_file)  
            #     print("          q_a[6,5,10,0,:] ",   q_a[6, 5, 10, 0, :], file=log_file)  # 0, 1, 2 are undef
            #     print("            q[6,5,10,0]   ",   q  [6, 5, 10, 0]   , file=log_file)
            #     print("          q_a[0,0,7,1,:]  ",   q_a[0, 0, 7, 1, :], file=log_file)  # 0.
            #     print("          q_a[1,1,7,1,:]  ",   q_a[1, 1, 7, 1, :], file=log_file)  # 0.
            #     print("            q[0,0,7,1]    ",     q[0, 0, 7, 1]   , file=log_file)  # 0.
            #     print("            q[1,1,7,1]    ",     q[1, 1, 7, 1]   , file=log_file)  # 0.
                print("STC1.5 :rhogq[6,5,3,1,:] ", rhogq[6, 5, 3, 1, :], file=log_file) 
                print("STD1.5 :rhogq[6,5,2,1,:] ", rhogq[6, 5, 2, 1, :], file=log_file) 
                print("STE1.5 :rhogq[16,:,24,1,1]", rhogq[16,:,24,1,1] , file=log_file)

            if adm.ADM_have_pl:
                g = adm.ADM_gslf_pl

                for l in range(lall_pl):
                    for k in range(kall):
                        for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):   # 1 to 5  range(1,6)
                            rhogq_pl[g, k, l, iq] -= flx_h_pl[v, k, l] * q_a_pl[v, k, l]



            # with open(std.fname_log, 'a') as log_file:
            # #     print("STA2:rhogq[0,0,6,1,:]  ", rhogq[0, 0, 6, 1, :], file=log_file)
            # #     print("     rhogq[0,0,7,1,:]  ", rhogq[0, 0, 7, 1, :], file=log_file)  #you  e+23
            # #     print("     rhogq[1,1,6,1,:]  ", rhogq[1, 1, 6, 1, :], file=log_file)
            # #     print("     rhogq[1,1,7,1,:]  ", rhogq[1, 1, 7, 1, :], file=log_file)  #you  e+23
            # #     print("     rhogq[1,1,5,1,:]  ", rhogq[1, 1, 5, 1, :], file=log_file)
            # #     print("     rhogq[1,1,8,1,:]  ", rhogq[1, 1, 8, 1, :], file=log_file)
            #     print("STA2.0 :  q_a[0,0,7,1,:]  ",   q_a[0, 0, 7, 1, :], file=log_file)  # 0.
            #     print("          q_a[1,1,7,1,:]  ",   q_a[1, 1, 7, 1, :], file=log_file)  # 0.
            #     print("            q[1,1,6,1]    ",   q  [1, 1, 6, 1]   , file=log_file)  # 0.
            #     print("            q[1,1,7,1]    ",   q  [1, 1, 7, 1]   , file=log_file)  # 0.


            # if adm.ADM_have_pl:
            #     print("rhogq_pl.shape", rhogq_pl.shape)
            #     print(rhogq_pl[0,3,0,0])
            #     print("flx_h_pl")
            #     print(flx_h_pl[:,3,0])
            #     print("q_a_pl")
            #     print(q_a_pl[:,3,0])
            #endif

        #end iq LOOP

        # with open(std.fname_log, 'a') as log_file:
        #     print("STA2.1 : rhog[0,0,7,1]  ",  rhog[0, 0, 7, 1], file=log_file)  
        #     print("         rhog[1,1,7,1]  ",  rhog[1, 1, 7, 1], file=log_file)  


        #--- update rhog

        isl = slice(1, iall-1)
        jsl = slice(1, jall-1)

        rhog[isl, jsl, :, :] -= (
            flx_h[isl, jsl, :, :, 0] + flx_h[isl, jsl, :, :, 1] +
            flx_h[isl, jsl, :, :, 2] + flx_h[isl, jsl, :, :, 3] +
            flx_h[isl, jsl, :, :, 4] + flx_h[isl, jsl, :, :, 5]
        ) - (b2 * frhog[isl, jsl, :, :] * dt)


        # for l in range(lall):
        #     for k in range(kall):
        #         rhog[:, :, k, l] -= (
        #             flx_h[:, :, k, l, 0] +
        #             flx_h[:, :, k, l, 1] +
        #             flx_h[:, :, k, l, 2] +
        #             flx_h[:, :, k, l, 3] +
        #             flx_h[:, :, k, l, 4] +
        #             flx_h[:, :, k, l, 5]
        #         )
        #         rhog[:, :, k, l] += b2 * frhog[:, :, k, l] * dt

        # with open(std.fname_log, 'a') as log_file:
        #     print("STA2.2 : rhog[0,0,7,1]  ",  rhog[0, 0, 7, 1], file=log_file)  
        #     print("         rhog[1,1,7,1]  ",  rhog[1, 1, 7, 1], file=log_file)  
        #     print("        frhog[0,0,7,1]  ", frhog[0, 0, 7, 1], file=log_file)  
        #     print("        frhog[1,1,7,1]  ", frhog[1, 1, 7, 1], file=log_file)  
        #     print("        rhogq[0,0,7,1]  ", rhogq[0, 0, 7, 1], file=log_file)  
        #     print("        rhogq[1,1,7,1]  ", rhogq[1, 1, 7, 1], file=log_file) 

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

        # for l in range(lall):
        #     d[:, :, :, l] = b3 * frhog[:, :, :, l] / rhog[:, :, :, l] * dt

        #     for k in range(kmin, kmax + 1):
        #         ck[:, :, k, l, 0] = -flx_v[:, :, k,   l] / rhog[:, :, k, l] * grd.GRD_rdgz[k]
        #         ck[:, :, k, l, 1] =  flx_v[:, :, k+1, l] / rhog[:, :, k, l] * grd.GRD_rdgz[k]

        #     ck[:, :, kmin - 1, l, 0] = rdtype(0.0)
        #     ck[:, :, kmin - 1, l, 1] = rdtype(0.0)
        #     ck[:, :, kmax + 1, l, 0] = rdtype(0.0)
        #     ck[:, :, kmax + 1, l, 1] = rdtype(0.0)

        d[:, :, :, :] = b3 * frhog[:, :, :, :] / rhog[:, :, :, :] * dt

        # Prepare k slice
        k_slice = slice(kmin, kmax + 1)

        # Main ck calculation, fully vectorized over (i, j, k, l)
        ck[:, :, k_slice, :, 0] = -flx_v[:, :, kmin:kmax+1, :] / rhog[:, :, kmin:kmax+1, :] * grd.GRD_rdgz[kmin:kmax+1, np.newaxis]
        ck[:, :, k_slice, :, 1] =  flx_v[:, :, kmin+1:kmax+2, :] / rhog[:, :, kmin:kmax+1, :] * grd.GRD_rdgz[kmin:kmax+1, np.newaxis]

        # Boundary conditions for kmin-1 and kmax+1
        ck[:, :, kmin-1, :, 0] = 0.0
        ck[:, :, kmin-1, :, 1] = 0.0
        ck[:, :, kmax+1, :, 0] = 0.0
        ck[:, :, kmax+1, :, 1] = 0.0


        if adm.ADM_have_pl:
            d_pl = b3 * frhog_pl / rhog_pl * dt  # fully vectorized over g, k, l

            for k in range(kmin, kmax + 1):
                ck_pl[:, k, :, 0] = -flx_v_pl[:, k,   :] / rhog_pl[:, k, :] * grd.GRD_rdgz[k]
                ck_pl[:, k, :, 1] =  flx_v_pl[:, k+1, :] / rhog_pl[:, k, :] * grd.GRD_rdgz[k]

            ck_pl[:, kmin - 1, :, 0] = rdtype(0.0)
            ck_pl[:, kmin - 1, :, 1] = rdtype(0.0)
            ck_pl[:, kmax + 1, :, 0] = rdtype(0.0)
            ck_pl[:, kmax + 1, :, 1] = rdtype(0.0)


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
                q_h[:, :, kmin - 1, l] = rdtype(0.0)
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
                q_h_pl[:, kmin-1, :] = rdtype(0.0)
            # endif


            with open(std.fname_log, 'a') as log_file:
                print(f"iq=  {iq} ",file=log_file)
            #     print("STA2.5 :rhogq[0,0,7,1,:]  ", rhogq[0, 0, 7, 1, :], file=log_file)  #you  bad
            #     print("        rhogq[1,1,7,1,:]  ", rhogq[1, 1, 7, 1, :], file=log_file)  #you  good
            #     print("          q_h[0,0,7,1]    ",   q_h[0, 0, 7, 1]   , file=log_file)  
            #     print("            q[0,0,7,1]    ",     q[0, 0, 7, 1]   , file=log_file)
            #     print("            d[0,0,7,1]    ",     d[0, 0, 7, 1]   , file=log_file)  
            #     print("           ck[0,0,7,1,:]  ",    ck[0, 0, 7, 1, :], file=log_file)    #you bad
            #     print("          q_h[1,1,7,1]  ",     q_h[1, 1, 7, 1]   , file=log_file)    
            #     print("            q[1,1,7,1]  ",       q[1, 1, 7, 1]   , file=log_file)  
            #     print("            d[1,1,7,1]    ",     d[1, 1, 7, 1]   , file=log_file)
            #     print("           ck[1,1,7,1,:]  ",    ck[1, 1, 7, 1, :], file=log_file)    #you good

                print("STB2.5 :rhogq[6,5,10,0,:]  ", rhogq[6, 5, 10, 0, :], file=log_file)  #you  e+23
            #     print("          q_h[6,5,10,0]  ",     q_h[6, 5, 10, 0]   , file=log_file)  
            #     print("            q[6,5,10,0]  ",       q[6, 5, 10, 0]   , file=log_file)  # 0, 1, 2 are undef
                # print("            d[6,5,10,0]    ",     d[6, 5, 10, 0]   , file=log_file)
                # print("           ck[6,5,10,0,:]  ",    ck[6, 5, 10, 0, :], file=log_file)

                print("STC2.5 :rhogq[6,5,3,1,:]  ", rhogq[6, 5, 3, 1, :], file=log_file)  #you  e+23
                print("STD2.5 :rhogq[6,5,2,1,:]  ", rhogq[6, 5, 2, 1, :], file=log_file)  #you  e+23
                print("STD2.5 :rhogq[6,5,1,1,:]  ", rhogq[6, 5, 1, 1, :], file=log_file)  #you  e+23
                print("          q_h[6,5,3,1]  ",     q_h[6, 5, 3, 1]   , file=log_file)
                print("            q[6,5,3,1]  ",       q[6, 5, 3, 1]   , file=log_file)  # 0, 1, 2 are undef
                print("            d[6,5,3,1]    ",     d[6, 5, 3, 1]   , file=log_file)
                print("           ck[6,5,3,1,:]  ",    ck[6, 5, 3, 1, :], file=log_file)    #you good
                print("          q_h[6,5,2,1]  ",     q_h[6, 5, 2, 1]   , file=log_file)
                print("            q[6,5,2,1]  ",       q[6, 5, 2, 1]   , file=log_file)  # 0, 1, 2 are undef
                print("            d[6,5,2,1]    ",     d[6, 5, 2, 1]   , file=log_file)
                print("           ck[6,5,2,1,:]  ",    ck[6, 5, 2, 1, :], file=log_file)    #you good
                print("          q_h[6,5,1,1]  ",     q_h[6, 5, 1, 1]   , file=log_file)
                print("            q[6,5,1,1]  ",       q[6, 5, 1, 1]   , file=log_file)  # 0, 1, 2 are undef
                print("            d[6,5,1,1]    ",     d[6, 5, 1, 1]   , file=log_file)
                print("           ck[6,5,1,1,:]  ",    ck[6, 5, 1, 1, :], file=log_file)    #you good
            if apply_limiter_v[iq]:
                self.vertical_limiter_thuburn(
                    q_h[:,:,:,:],   q_h_pl[:,:,:],  # [INOUT]     # q_h [6,5,2,1]  from 0.9 to  -159.38599569471765 instead of 0.9 (org) at iq = 2 of 1st step in rank 6
                    q  [:,:,:,:],   q_pl  [:,:,:],  # [IN]
                    d  [:,:,:,:],   d_pl  [:,:,:],  # [IN]
                    ck [:,:,:,:,:], ck_pl [:,:,:,:],  # [IN]
                    cnst, rdtype,
                )
            # endif

            #--- update rhogq

            for l in range(lall):
                q_h[:, :, kmin, l] = rdtype(0.0)
                q_h[:, :, kmax+1, l] = rdtype(0.0)

                for k in range(kmin, kmax+1):             
                    rhogq[:, :, k, l, iq] -= (
                        flx_v[:, :, k+1, l] * q_h[:, :, k+1, l] -
                        flx_v[:, :, k,   l] * q_h[:, :, k,   l]
                    ) * grd.GRD_rdgz[k]

                rhogq[:, :, kmin-1, l, iq] = rdtype(0.0)     
                rhogq[:, :, kmax+1, l, iq] = rdtype(0.0)

            

            if adm.ADM_have_pl:
                q_h_pl[:, kmin,   :] = rdtype(0.0)
                q_h_pl[:, kmax+1, :] = rdtype(0.0)

                for k in range(kmin, kmax+1):
                    rhogq_pl[:, k, :, iq] -= (
                        flx_v_pl[:, k+1, :] * q_h_pl[:, k+1, :] -
                        flx_v_pl[:, k,   :] * q_h_pl[:, k,   :]
                    ) * grd.GRD_rdgz[k]

                rhogq_pl[:, kmin-1, :, iq] = rdtype(0.0)
                rhogq_pl[:, kmax+1, :, iq] = rdtype(0.0)

            with open(std.fname_log, 'a') as log_file:
               
            #     print("STA2.6 :rhogq[0,0,7,1,:]  ", rhogq[0, 0, 7, 1, :], file=log_file)  
            #     print("        rhogq[1,1,7,1,:]  ", rhogq[1, 1, 7, 1, :], file=log_file)  
            #     print("        flx_v[0,0,8,1]  ", flx_v[0, 0, 8, 1], file=log_file) 
            #     print("        flx_v[0,0,7,1]  ", flx_v[0, 0, 7, 1], file=log_file)  
            #     print("        flx_v[1,1,8,1]  ", flx_v[1, 1, 8, 1], file=log_file) 
            #     print("        flx_v[1,1,7,1]  ", flx_v[1, 1, 7, 1], file=log_file)  
            #     print("          q_h[0,0,8,1]  ",   q_h[0, 0, 8, 1], file=log_file) 
            #     print("          q_h[0,0,7,1]  ",   q_h[0, 0, 7, 1], file=log_file)  
            #     print("          q_h[1,1,8,1]  ",   q_h[1, 1, 8, 1], file=log_file) 
            #     print("          q_h[1,1,7,1]  ",   q_h[1, 1, 7, 1], file=log_file)  
            #     print("       grd.GRD_rdgz[7]  ",   grd.GRD_rdgz[7], file=log_file)  

            #print("STB2.6 :rhogq[6,5,10,0,:]  ", rhogq[6, 5, 10, 0, :], file=log_file)  
            #print("          q_h[6,5,10,0]  ",     q_h[6, 5, 10, 0]   , file=log_file)  
                print("STD2.6 :rhogq[6,5,3,1,:]  ", rhogq[6, 5, 3, 1, :], file=log_file)  
                print("STD2.6 :rhogq[6,5,2,1,:]  ", rhogq[6, 5, 2, 1, :], file=log_file)  
                print("STD2.6 :rhogq[6,5,1,1,:]  ", rhogq[6, 5, 1, 1, :], file=log_file)  

                print("        flx_v[6,5,3,1]  ", flx_v[6, 5, 3, 1], file=log_file) 
                print("        flx_v[6,5,2,1]  ", flx_v[6, 5, 2, 1], file=log_file)  
                print("        flx_v[6,5,1,1]  ", flx_v[6, 5, 1, 1], file=log_file)  
                print("          q_h[6,5,3,1]  ",   q_h[6, 5, 3, 1], file=log_file) 
                print("          q_h[6,5,2,1]  ",   q_h[6, 5, 2, 1], file=log_file)  
                print("          q_h[6,5,1,1]  ",   q_h[6, 5, 1, 1], file=log_file)  

            #--- tiny negative fixer

            for l in range(lall):
                for k in range(kmin, kmax + 1):
                    mask = (rhogq[:, :, k, l, iq] > -rdtype(1.0e-10)) & (rhogq[:, :, k, l, iq] < rdtype(0.0))
                    rhogq[:, :, k, l, iq][mask] = rdtype(0.0)

            mask_pl = (rhogq_pl[..., iq] > -rdtype(1.0e-10)) & (rhogq_pl[..., iq] < rdtype(0.0))
            rhogq_pl[..., iq][mask_pl] = rdtype(0.0)

        # end loop iq

        prf.PROF_rapend('____vertical_adv',2)

        return
    
    #> Prepare horizontal advection term: mass flux, horizon
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

        TI  = adm.ADM_TI  
        TJ  = adm.ADM_TJ  
        AI  = adm.ADM_AI  
        AIJ = adm.ADM_AIJ 
        AJ  = adm.ADM_AJ  
        K0  = adm.ADM_K0
 
        XDIR = grd.GRD_XDIR 
        YDIR = grd.GRD_YDIR 
        ZDIR = grd.GRD_ZDIR
    
        P_RAREA = gmtr.GMTR_p_RAREA
        T_RAREA = gmtr.GMTR_t_RAREA 
        W1      = gmtr.GMTR_t_W1  
        W2      = gmtr.GMTR_t_W2    
        W3      = gmtr.GMTR_t_W3    
        HNX     = gmtr.GMTR_a_HNX   
        HNY     = gmtr.GMTR_a_HNY   
        HNZ     = gmtr.GMTR_a_HNZ   
    
        rhot_TI  = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)  # rho at cell vertex
        rhot_TJ  = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)  # rho at cell vertex
        rhovxt_TI= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        rhovxt_TJ= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        rhovyt_TI= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        rhovyt_TJ= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        rhovzt_TI= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        rhovzt_TJ= np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)

        rhot_pl  = np.full((gall_pl,), cnst.CONST_UNDEF)
        rhovxt_pl= np.full((gall_pl,), cnst.CONST_UNDEF)
        rhovyt_pl= np.full((gall_pl,), cnst.CONST_UNDEF)
        rhovzt_pl= np.full((gall_pl,), cnst.CONST_UNDEF)


        for l in range(lall):
            for k in range(kall):

                isl = slice(0, iall - 1)
                jsl = slice(0, jall - 1)

                isl_p = slice(1, iall)
                jsl_p = slice(1, jall)

                # First part: (i,j), (i+1,j)
                #print(gmtr.GMTR_t.shape)
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


                flx_h[:, :, k, l, :].fill(rdtype(0.0))      
                grd_xc[:, :, k, l, :, :].fill(rdtype(0.0))       


                isl = slice(0, iall - 1)
                jsl = slice(1, jall - 1)
                jslm1 = slice(0, jall - 2)   # Otameshi to replace jsl-1  (4 of them in the block right below)

                rrhoa2 = rdtype(1.0) / np.maximum(
                    rhot_TJ[isl, jslm1, k] + rhot_TI[isl, jsl, k], EPS
                )
                rhovxt2 = rhovxt_TJ[isl, jslm1, k] + rhovxt_TI[isl, jsl, k]
                rhovyt2 = rhovyt_TJ[isl, jslm1, k] + rhovyt_TI[isl, jsl, k]
                rhovzt2 = rhovzt_TJ[isl, jslm1, k] + rhovzt_TI[isl, jsl, k]

                flux = rdtype(0.5) * (
                    rhovxt2 * gmtr.GMTR_a[isl, jsl, K0, l, AI, HNX] +
                    rhovyt2 * gmtr.GMTR_a[isl, jsl, K0, l, AI, HNY] +
                    rhovzt2 * gmtr.GMTR_a[isl, jsl, K0, l, AI, HNZ]
                )

                flx_h[isl, jsl, k, l, 0]  =  flux * gmtr.GMTR_p[isl, jsl, K0, l, P_RAREA] * dt
                flx_h[isl.start+1:isl.stop+1, jsl, k, l, 3] = -flux * gmtr.GMTR_p[isl.start+1:isl.stop+1, jsl, K0, l, P_RAREA] * dt

                grd_xc[isl, jsl, k, l, AI, XDIR] = grd.GRD_xr[isl, jsl, K0, l, AI, XDIR] - rhovxt2 * rrhoa2 * dt * rdtype(0.5)
                grd_xc[isl, jsl, k, l, AI, YDIR] = grd.GRD_xr[isl, jsl, K0, l, AI, YDIR] - rhovyt2 * rrhoa2 * dt * rdtype(0.5)
                grd_xc[isl, jsl, k, l, AI, ZDIR] = grd.GRD_xr[isl, jsl, K0, l, AI, ZDIR] - rhovzt2 * rrhoa2 * dt * rdtype(0.5)



                isl = slice(0, iall - 1)
                jsl = slice(0, jall - 1)

                rrhoa2 = rdtype(1.0) / np.maximum(
                    rhot_TI[isl, jsl, k] + rhot_TJ[isl, jsl, k], EPS
                )
                rhovxt2 = rhovxt_TI[isl, jsl, k] + rhovxt_TJ[isl, jsl, k]
                rhovyt2 = rhovyt_TI[isl, jsl, k] + rhovyt_TJ[isl, jsl, k]
                rhovzt2 = rhovzt_TI[isl, jsl, k] + rhovzt_TJ[isl, jsl, k]

                flux = rdtype(0.5) * (
                    rhovxt2 * gmtr.GMTR_a[isl, jsl, K0, l, AIJ, HNX] +
                    rhovyt2 * gmtr.GMTR_a[isl, jsl, K0, l, AIJ, HNY] +
                    rhovzt2 * gmtr.GMTR_a[isl, jsl, K0, l, AIJ, HNZ]
                )

                flx_h[isl, jsl, k, l, 1] =  flux * gmtr.GMTR_p[isl, jsl, K0, l, P_RAREA] * dt
                flx_h[isl.start+1:isl.stop+1, jsl.start+1:jsl.stop+1, k, l, 4] = -flux * gmtr.GMTR_p[isl.start+1:isl.stop+1, jsl.start+1:jsl.stop+1, K0, l, P_RAREA] * dt

                grd_xc[isl, jsl, k, l, AIJ, XDIR] = grd.GRD_xr[isl, jsl, K0, l, AIJ, XDIR] - rhovxt2 * rrhoa2 * dt * rdtype(0.5)
                grd_xc[isl, jsl, k, l, AIJ, YDIR] = grd.GRD_xr[isl, jsl, K0, l, AIJ, YDIR] - rhovyt2 * rrhoa2 * dt * rdtype(0.5)
                grd_xc[isl, jsl, k, l, AIJ, ZDIR] = grd.GRD_xr[isl, jsl, K0, l, AIJ, ZDIR] - rhovzt2 * rrhoa2 * dt * rdtype(0.5)

                if l == 1 and k == 24:
                    with open(std.fname_log, 'a') as log_file:
                        print("grd_xc[16,5,24,1,AIJ,XDIR]  ", grd_xc[16, 5, k, l, AIJ, XDIR], file=log_file)  
                        print("grd.GRD_xr[16, 5, K0, l, :, XDIR]  ",     grd.GRD_xr[16, 5, K0, l, :, XDIR]   , file=log_file)  
                        print("grd.GRD_xt[16, 5, K0, l, :, XDIR]  ",     grd.GRD_xt[16, 5, K0, l, :, XDIR]   , file=log_file)  
                        print("grd.GRD_st[16, 5, K0, l, :, XDIR]  ",     grd.GRD_st[16, 5, K0, l, :, XDIR]   , file=log_file)  

                        print("grd.GRD_xr[6, 5, K0, 0, :, XDIR]  ",     grd.GRD_xr[6, 5, K0, 0, :, XDIR]   , file=log_file)  
                        print("rhovxt2[16,5,24,1]  ",  rhovxt2[16, 5]   , file=log_file)  
                        print("rrhoa2[16,5,24,1]  ",  rrhoa2[16, 5]   , file=log_file)  

                isl = slice(1, iall - 1)
                jsl = slice(0, jall - 1)

                rrhoa2 = rdtype(1.0) / np.maximum(
                    rhot_TJ[isl, jsl, k] + rhot_TI[isl.start - 1:isl.stop - 1, jsl, k],
                    EPS
                )
                rhovxt2 = rhovxt_TJ[isl, jsl, k] + rhovxt_TI[isl.start - 1:isl.stop - 1, jsl, k]
                rhovyt2 = rhovyt_TJ[isl, jsl, k] + rhovyt_TI[isl.start - 1:isl.stop - 1, jsl, k]
                rhovzt2 = rhovzt_TJ[isl, jsl, k] + rhovzt_TI[isl.start - 1:isl.stop - 1, jsl, k]

                flux = rdtype(0.5) * (
                    rhovxt2 * gmtr.GMTR_a[isl, jsl, K0, l, AJ, HNX] +
                    rhovyt2 * gmtr.GMTR_a[isl, jsl, K0, l, AJ, HNY] +
                    rhovzt2 * gmtr.GMTR_a[isl, jsl, K0, l, AJ, HNZ]
                )

                flx_h[isl, jsl, k, l, 2] =  flux * gmtr.GMTR_p[isl, jsl, K0, l, P_RAREA] * dt
                flx_h[isl, jsl.start + 1:jsl.stop + 1, k, l, 5] = -flux * gmtr.GMTR_p[isl, jsl.start + 1:jsl.stop + 1, K0, l, P_RAREA] * dt

                grd_xc[isl, jsl, k, l, AJ, XDIR] = grd.GRD_xr[isl, jsl, K0, l, AJ, XDIR] - rhovxt2 * rrhoa2 * dt * rdtype(0.5)
                grd_xc[isl, jsl, k, l, AJ, YDIR] = grd.GRD_xr[isl, jsl, K0, l, AJ, YDIR] - rhovyt2 * rrhoa2 * dt * rdtype(0.5)
                grd_xc[isl, jsl, k, l, AJ, ZDIR] = grd.GRD_xr[isl, jsl, K0, l, AJ, ZDIR] - rhovzt2 * rrhoa2 * dt * rdtype(0.5)


                if adm.ADM_have_sgp[l]:
                    flx_h[1,1,k,l,5] = rdtype(0.0)   # really?

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

                        rrhoa2  = rdtype(1.0) / max(rhot_pl[ijm1] + rhot_pl[ij], EPS)
                        rhovxt2 = rhovxt_pl[ijm1] + rhovxt_pl[ij]
                        rhovyt2 = rhovyt_pl[ijm1] + rhovyt_pl[ij]
                        rhovzt2 = rhovzt_pl[ijm1] + rhovzt_pl[ij]

                        flux = rdtype(0.5) * (
                            rhovxt2 * gmtr.GMTR_a_pl[ij, K0, l, HNX] +
                            rhovyt2 * gmtr.GMTR_a_pl[ij, K0, l, HNY] +
                            rhovzt2 * gmtr.GMTR_a_pl[ij, K0, l, HNZ]
                        )

                        flx_h_pl[v, k, l] = flux * gmtr.GMTR_p_pl[n, K0, l, P_RAREA] * dt

                        grd_xc_pl[v, k, l, XDIR] = grd.GRD_xr_pl[v, K0, l, XDIR] - rhovxt2 * rrhoa2 * dt * rdtype(0.5)
                        grd_xc_pl[v, k, l, YDIR] = grd.GRD_xr_pl[v, K0, l, YDIR] - rhovyt2 * rrhoa2 * dt * rdtype(0.5)
                        grd_xc_pl[v, k, l, ZDIR] = grd.GRD_xr_pl[v, K0, l, ZDIR] - rhovzt2 * rrhoa2 * dt * rdtype(0.5)
                    # end loop v

                # end loop k
            # end loop l
        # endif

        prf.PROF_rapend  ('____horizontal_adv_flux',2)

        return

    def horizontal_remap(self, 
        q_a,    q_a_pl,       # [OUT]    # q at cell face
        q,      q_pl,         # [IN]     # q at cell center
        cmask,  cmask_pl,     # [IN]     # upwind direction mask
        grd_xc, grd_xc_pl,    # [IN]     # position of the mass centroid
        cnst, comm, grd, oprt, rdtype,
    ):
        
        prf.PROF_rapstart('____horizontal_adv_remap',2)
        
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        nxyz = adm.ADM_nxyz

        TI  = adm.ADM_TI  
        TJ  = adm.ADM_TJ  
        AI  = adm.ADM_AI  
        AIJ = adm.ADM_AIJ
        AJ  = adm.ADM_AJ  
        K0  = adm.ADM_K0

        XDIR = grd.GRD_XDIR 
        YDIR = grd.GRD_YDIR 
        ZDIR = grd.GRD_ZDIR

        # nstart1 = suf(ADM_gmin-1,ADM_gmin-1)
        # nstart2 = suf(ADM_gmin  ,ADM_gmin-1)
        # nstart3 = suf(ADM_gmin  ,ADM_gmin  )
        # nstart4 = suf(ADM_gmin-1,ADM_gmin  )
        # nend    = suf(ADM_gmax  ,ADM_gmax  )

        gradq = np.full(adm.ADM_shape + (nxyz,), cnst.CONST_UNDEF)  # grad(q)
        gradq_pl = np.full(adm.ADM_shape_pl + (nxyz,), cnst.CONST_UNDEF)
    
        q_ap1 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_am1 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_ap2 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_am2 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_ap3 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_am3 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_ap4 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_am4 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_ap5 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_am5 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_ap6 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)
        q_am6 = np.full(adm.ADM_shape[:3], cnst.CONST_UNDEF)

        oprt.OPRT_gradient(
            gradq, gradq_pl, 
            q, q_pl,
            oprt.OPRT_coef_grad, oprt.OPRT_coef_grad_pl,
            grd, rdtype,
        )
       
        comm.COMM_data_transfer( gradq[:,:,:,:,:], gradq_pl[:,:,:,:] )


        # interpolated Q at cell arc

        q_ap1.fill(rdtype(0.0))
        q_am1.fill(rdtype(0.0))
        q_ap2.fill(rdtype(0.0))
        q_am2.fill(rdtype(0.0))
        q_ap3.fill(rdtype(0.0))
        q_am3.fill(rdtype(0.0))
        q_ap4.fill(rdtype(0.0))
        q_am4.fill(rdtype(0.0))
        q_ap5.fill(rdtype(0.0))
        q_am5.fill(rdtype(0.0))
        q_ap6.fill(rdtype(0.0))
        q_am6.fill(rdtype(0.0))

        isl = slice(0, iall - 1)
        jsl = slice(0, jall - 1)
        isl_p1 = slice(1, iall)
        jsl_p1 = slice(1, jall)
        isls  = slice(1, iall - 1)
        jsls  = slice(1, jall - 1)
        isls_m1 = slice(0, iall - 2)
        jsls_m1 = slice(0, jall - 2)

        for l in range(lall):
            for k in range(kall):

                # q_ap1
                q_ap1[isl, jsl, k] = (
                    q[isl, jsl, k, l]
                    + gradq[isl, jsl, k, l, XDIR] * (grd_xc[isl, jsl, k, l, AI, XDIR] - grd.GRD_x[isl, jsl, K0, l, XDIR])
                    + gradq[isl, jsl, k, l, YDIR] * (grd_xc[isl, jsl, k, l, AI, YDIR] - grd.GRD_x[isl, jsl, K0, l, YDIR])
                    + gradq[isl, jsl, k, l, ZDIR] * (grd_xc[isl, jsl, k, l, AI, ZDIR] - grd.GRD_x[isl, jsl, K0, l, ZDIR])
                )

                # q_am1
                q_am1[isl, jsl, k] = (
                    q[isl_p1, jsl, k, l]
                    + gradq[isl_p1, jsl, k, l, XDIR] * (grd_xc[isl, jsl, k, l, AI, XDIR] - grd.GRD_x[isl_p1, jsl, K0, l, XDIR])
                    + gradq[isl_p1, jsl, k, l, YDIR] * (grd_xc[isl, jsl, k, l, AI, YDIR] - grd.GRD_x[isl_p1, jsl, K0, l, YDIR])
                    + gradq[isl_p1, jsl, k, l, ZDIR] * (grd_xc[isl, jsl, k, l, AI, ZDIR] - grd.GRD_x[isl_p1, jsl, K0, l, ZDIR])
                )

                # q_ap2
                q_ap2[isl, jsl, k] = (
                    q[isl, jsl, k, l]
                    + gradq[isl, jsl, k, l, XDIR] * (grd_xc[isl, jsl, k, l, AIJ, XDIR] - grd.GRD_x[isl, jsl, K0, l, XDIR])
                    + gradq[isl, jsl, k, l, YDIR] * (grd_xc[isl, jsl, k, l, AIJ, YDIR] - grd.GRD_x[isl, jsl, K0, l, YDIR])
                    + gradq[isl, jsl, k, l, ZDIR] * (grd_xc[isl, jsl, k, l, AIJ, ZDIR] - grd.GRD_x[isl, jsl, K0, l, ZDIR])
                )

                # q_am2
                q_am2[isl, jsl, k] = (
                    q[isl_p1, jsl_p1, k, l]
                    + gradq[isl_p1, jsl_p1, k, l, XDIR] * (grd_xc[isl, jsl, k, l, AIJ, XDIR] - grd.GRD_x[isl_p1, jsl_p1, K0, l, XDIR])
                    + gradq[isl_p1, jsl_p1, k, l, YDIR] * (grd_xc[isl, jsl, k, l, AIJ, YDIR] - grd.GRD_x[isl_p1, jsl_p1, K0, l, YDIR])
                    + gradq[isl_p1, jsl_p1, k, l, ZDIR] * (grd_xc[isl, jsl, k, l, AIJ, ZDIR] - grd.GRD_x[isl_p1, jsl_p1, K0, l, ZDIR])
                )

                # q_ap3
                q_ap3[isl, jsl, k] = (
                    q[isl, jsl, k, l]
                    + gradq[isl, jsl, k, l, XDIR] * (grd_xc[isl, jsl, k, l, AJ, XDIR] - grd.GRD_x[isl, jsl, K0, l, XDIR])
                    + gradq[isl, jsl, k, l, YDIR] * (grd_xc[isl, jsl, k, l, AJ, YDIR] - grd.GRD_x[isl, jsl, K0, l, YDIR])
                    + gradq[isl, jsl, k, l, ZDIR] * (grd_xc[isl, jsl, k, l, AJ, ZDIR] - grd.GRD_x[isl, jsl, K0, l, ZDIR])
                )

                # q_am3
                q_am3[isl, jsl, k] = (
                    q[isl, jsl_p1, k, l]
                    + gradq[isl, jsl_p1, k, l, XDIR] * (grd_xc[isl, jsl, k, l, AJ, XDIR] - grd.GRD_x[isl, jsl_p1, K0, l, XDIR])
                    + gradq[isl, jsl_p1, k, l, YDIR] * (grd_xc[isl, jsl, k, l, AJ, YDIR] - grd.GRD_x[isl, jsl_p1, K0, l, YDIR])
                    + gradq[isl, jsl_p1, k, l, ZDIR] * (grd_xc[isl, jsl, k, l, AJ, ZDIR] - grd.GRD_x[isl, jsl_p1, K0, l, ZDIR])
                )

                # q_ap4
                q_ap4[isls, jsls, k] = (
                    q[isls_m1, jsls, k, l]
                    + gradq[isls_m1, jsls, k, l, XDIR] * (grd_xc[isls_m1, jsls, k, l, AI, XDIR] - grd.GRD_x[isls_m1, jsls, K0, l, XDIR])
                    + gradq[isls_m1, jsls, k, l, YDIR] * (grd_xc[isls_m1, jsls, k, l, AI, YDIR] - grd.GRD_x[isls_m1, jsls, K0, l, YDIR])
                    + gradq[isls_m1, jsls, k, l, ZDIR] * (grd_xc[isls_m1, jsls, k, l, AI, ZDIR] - grd.GRD_x[isls_m1, jsls, K0, l, ZDIR])
                )

                # q_am4
                q_am4[isls, jsls, k] = (
                    q[isls, jsls, k, l]
                    + gradq[isls, jsls, k, l, XDIR] * (grd_xc[isls_m1, jsls, k, l, AI, XDIR] - grd.GRD_x[isls, jsls, K0, l, XDIR])
                    + gradq[isls, jsls, k, l, YDIR] * (grd_xc[isls_m1, jsls, k, l, AI, YDIR] - grd.GRD_x[isls, jsls, K0, l, YDIR])
                    + gradq[isls, jsls, k, l, ZDIR] * (grd_xc[isls_m1, jsls, k, l, AI, ZDIR] - grd.GRD_x[isls, jsls, K0, l, ZDIR])
                )

                # q_ap5
                q_ap5[isls, jsls, k] = (
                    q[isls_m1, jsls_m1, k, l]
                    + gradq[isls_m1, jsls_m1, k, l, XDIR] * (grd_xc[isls_m1, jsls_m1, k, l, AIJ, XDIR] - grd.GRD_x[isls_m1, jsls_m1, K0, l, XDIR])
                    + gradq[isls_m1, jsls_m1, k, l, YDIR] * (grd_xc[isls_m1, jsls_m1, k, l, AIJ, YDIR] - grd.GRD_x[isls_m1, jsls_m1, K0, l, YDIR])
                    + gradq[isls_m1, jsls_m1, k, l, ZDIR] * (grd_xc[isls_m1, jsls_m1, k, l, AIJ, ZDIR] - grd.GRD_x[isls_m1, jsls_m1, K0, l, ZDIR])
                )

                # q_am5
                q_am5[isls, jsls, k] = (
                    q[isls, jsls, k, l]
                    + gradq[isls, jsls, k, l, XDIR] * (grd_xc[isls_m1, jsls_m1, k, l, AIJ, XDIR] - grd.GRD_x[isls, jsls, K0, l, XDIR])
                    + gradq[isls, jsls, k, l, YDIR] * (grd_xc[isls_m1, jsls_m1, k, l, AIJ, YDIR] - grd.GRD_x[isls, jsls, K0, l, YDIR])
                    + gradq[isls, jsls, k, l, ZDIR] * (grd_xc[isls_m1, jsls_m1, k, l, AIJ, ZDIR] - grd.GRD_x[isls, jsls, K0, l, ZDIR])
                )

                # q_ap6
                q_ap6[isls, jsls, k] = (
                    q[isls, jsls_m1, k, l]
                    + gradq[isls, jsls_m1, k, l, XDIR] * (grd_xc[isls, jsls_m1, k, l, AJ, XDIR] - grd.GRD_x[isls, jsls_m1, K0, l, XDIR])
                    + gradq[isls, jsls_m1, k, l, YDIR] * (grd_xc[isls, jsls_m1, k, l, AJ, YDIR] - grd.GRD_x[isls, jsls_m1, K0, l, YDIR])
                    + gradq[isls, jsls_m1, k, l, ZDIR] * (grd_xc[isls, jsls_m1, k, l, AJ, ZDIR] - grd.GRD_x[isls, jsls_m1, K0, l, ZDIR])
                )

                # q_am6
                q_am6[isls, jsls, k] = (
                    q[isls, jsls, k, l]
                    + gradq[isls, jsls, k, l, XDIR] * (grd_xc[isls, jsls_m1, k, l, AJ, XDIR] - grd.GRD_x[isls, jsls, K0, l, XDIR])
                    + gradq[isls, jsls, k, l, YDIR] * (grd_xc[isls, jsls_m1, k, l, AJ, YDIR] - grd.GRD_x[isls, jsls, K0, l, YDIR])
                    + gradq[isls, jsls, k, l, ZDIR] * (grd_xc[isls, jsls_m1, k, l, AJ, ZDIR] - grd.GRD_x[isls, jsls, K0, l, ZDIR])
                )


                q_a[isl, jsl, k, l, 0] = cmask[isl, jsl, k, l, 0] * q_am1[isl, jsl, k] + (rdtype(1.0) - cmask[isl, jsl, k, l, 0]) * q_ap1[isl, jsl, k]
                q_a[isl, jsl, k, l, 1] = cmask[isl, jsl, k, l, 1] * q_am2[isl, jsl, k] + (rdtype(1.0) - cmask[isl, jsl, k, l, 1]) * q_ap2[isl, jsl, k]
                q_a[isl, jsl, k, l, 2] = cmask[isl, jsl, k, l, 2] * q_am3[isl, jsl, k] + (rdtype(1.0) - cmask[isl, jsl, k, l, 2]) * q_ap3[isl, jsl, k]
                q_a[isl, jsl, k, l, 3] = cmask[isl, jsl, k, l, 3] * q_am4[isl, jsl, k] + (rdtype(1.0) - cmask[isl, jsl, k, l, 3]) * q_ap4[isl, jsl, k]
                q_a[isl, jsl, k, l, 4] = cmask[isl, jsl, k, l, 4] * q_am5[isl, jsl, k] + (rdtype(1.0) - cmask[isl, jsl, k, l, 4]) * q_ap5[isl, jsl, k]
                q_a[isl, jsl, k, l, 5] = cmask[isl, jsl, k, l, 5] * q_am6[isl, jsl, k] + (rdtype(1.0) - cmask[isl, jsl, k, l, 5]) * q_ap6[isl, jsl, k]

            # end loop k
        # end loop l

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(adm.ADM_lall_pl):
                for k in range(kall):
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        q_ap = (
                            q_pl[n, k, l]
                            + gradq_pl[n, k, l, XDIR] * (grd_xc_pl[v, k, l, XDIR] - grd.GRD_x_pl[n, K0, l, XDIR])
                            + gradq_pl[n, k, l, YDIR] * (grd_xc_pl[v, k, l, YDIR] - grd.GRD_x_pl[n, K0, l, YDIR])
                            + gradq_pl[n, k, l, ZDIR] * (grd_xc_pl[v, k, l, ZDIR] - grd.GRD_x_pl[n, K0, l, ZDIR])
                        )

                        q_am = (
                            q_pl[v, k, l]
                            + gradq_pl[v, k, l, XDIR] * (grd_xc_pl[v, k, l, XDIR] - grd.GRD_x_pl[v, K0, l, XDIR])
                            + gradq_pl[v, k, l, YDIR] * (grd_xc_pl[v, k, l, YDIR] - grd.GRD_x_pl[v, K0, l, YDIR])
                            + gradq_pl[v, k, l, ZDIR] * (grd_xc_pl[v, k, l, ZDIR] - grd.GRD_x_pl[v, K0, l, ZDIR])
                        )

                        q_a_pl[v, k, l] = (
                            cmask_pl[v, k, l] * q_am + (rdtype(1.0) - cmask_pl[v, k, l]) * q_ap
                        )
                    # end loop v
                # end loop k
            # end loop l
        # endif

        prf.PROF_rapend('____horizontal_adv_remap',2)

        return
        


    def vertical_limiter_thuburn_fast_maybeok(self,
        q_h, q_h_pl, q, q_pl, d, d_pl, ck, ck_pl, 
        cnst, rdtype
    ):
        prf.PROF_rapstart('____vertical_adv_limiter', 2)

        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        gall_pl = adm.ADM_gall_pl

        EPS = cnst.CONST_EPS
        BIG = cnst.CONST_HUGE
        UNDEF = cnst.CONST_UNDEF
        ONE = rdtype(1.0)
        HALF = rdtype(0.5)

        isl = slice(0, iall)
        jsl = slice(0, jall)

        # Allocate once
        Qout_min_km1 = np.full((iall, jall), UNDEF, dtype=rdtype)
        Qout_max_km1 = np.full((iall, jall), UNDEF, dtype=rdtype)
        Qout_min_pl =np.full(adm.ADM_shape_pl, UNDEF)
        Qout_max_pl =np.full(adm.ADM_shape_pl, UNDEF)

        for l in range(lall):
            # Preload slices for efficiency
            q_slice = q[isl, jsl, :, l]  # (i,j,k)
            d_slice = d[isl, jsl, :, l]
            ck_slice = ck[isl, jsl, :, l, :]  # (i,j,k,2)

            # k = kmin separately
            k = kmin   # 1 in p

            inflagL = HALF - np.copysign(HALF, ck_slice[:, :, k, 0])
            inflagU = HALF + np.copysign(HALF, ck_slice[:, :, k+1, 0])

            q_center = q_slice[:, :, k]
            q_below  = q_slice[:, :, k-1]
            q_above  = q_slice[:, :, k+1]

            q_minL = np.minimum(q_center, q_below)
            q_minU = np.minimum(q_center, q_above)
            q_maxL = np.maximum(q_center, q_below)
            q_maxU = np.maximum(q_center, q_above)

            Qin_minL = inflagL * q_minL + (ONE - inflagL) * BIG
            Qin_minU = inflagU * q_minU + (ONE - inflagU) * BIG
            Qin_maxL = inflagL * q_maxL + (ONE - inflagL) * -BIG
            Qin_maxU = inflagU * q_maxU + (ONE - inflagU) * -BIG

            qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, q_center])
            qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, q_center])

            Cin  = inflagL * ck_slice[:, :, k, 0] + inflagU * ck_slice[:, :, k+1, 1]
            Cout = (ONE - inflagL) * ck_slice[:, :, k, 0] + (ONE - inflagU) * ck_slice[:, :, k+1, 1]

            CQin_min = inflagL * ck_slice[:, :, k, 0] * Qin_minL + inflagU * ck_slice[:, :, k+1, 1] * Qin_minU
            CQin_max = inflagL * ck_slice[:, :, k, 0] * Qin_maxL + inflagU * ck_slice[:, :, k+1, 1] * Qin_maxU

            zerosw = HALF - np.copysign(HALF, np.abs(Cout) - EPS)

            Cout_safe = Cout + zerosw
            nonzero_factor = (ONE - zerosw)

            Qout_min_k = ((q_center - qnext_max) + qnext_max * (Cin + Cout - d_slice[:, :, k]) - CQin_max) / Cout_safe * nonzero_factor + q_center * zerosw
            Qout_max_k = ((q_center - qnext_min) + qnext_min * (Cin + Cout - d_slice[:, :, k]) - CQin_min) / Cout_safe * nonzero_factor + q_center * zerosw

            # Store for kmin
            Qout_min_km1[:, :] = Qout_min_k
            Qout_max_km1[:, :] = Qout_max_k

            # Loop kmin+1 to kmax
            for k in range(kmin+1, kmax+1):

                inflagL = HALF - np.copysign(HALF, ck_slice[:, :, k, 0])
                inflagU = HALF + np.copysign(HALF, ck_slice[:, :, k, 1])

                q_center = q_slice[:, :, k]
                q_below  = q_slice[:, :, k-1]
                q_above  = q_slice[:, :, k+1]

                q_minL = np.minimum(q_center, q_below)
                q_minU = np.minimum(q_center, q_above)
                q_maxL = np.maximum(q_center, q_below)
                q_maxU = np.maximum(q_center, q_above)

                Qin_minL = inflagL * q_minL + (ONE - inflagL) * BIG
                Qin_minU = inflagU * q_minU + (ONE - inflagU) * BIG
                Qin_maxL = inflagL * q_maxL + (ONE - inflagL) * -BIG
                Qin_maxU = inflagU * q_maxU + (ONE - inflagU) * -BIG

                qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, q_center])
                qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, q_center])

                Cin  = inflagL * ck_slice[:, :, k, 0] + inflagU * ck_slice[:, :, k, 1]
                Cout = (ONE - inflagL) * ck_slice[:, :, k, 0] + (ONE - inflagU) * ck_slice[:, :, k, 1]

                CQin_min = inflagL * ck_slice[:, :, k, 0] * Qin_minL + inflagU * ck_slice[:, :, k, 1] * Qin_minU
                CQin_max = inflagL * ck_slice[:, :, k, 0] * Qin_maxL + inflagU * ck_slice[:, :, k, 1] * Qin_maxU

                zerosw = HALF - np.copysign(HALF, np.abs(Cout) - EPS)

                Cout_safe = Cout + zerosw
                nonzero_factor = (ONE - zerosw)

                qout_min = ((q_center - qnext_max) + qnext_max * (Cin + Cout - d_slice[:, :, k]) - CQin_max) / Cout_safe * nonzero_factor + q_center * zerosw
                qout_max = ((q_center - qnext_min) + qnext_min * (Cin + Cout - d_slice[:, :, k]) - CQin_min) / Cout_safe * nonzero_factor + q_center * zerosw

                # Manual clipping
                clipped_lower = np.minimum(np.maximum(q_h[isl, jsl, k, l], Qout_min_km1), Qout_max_km1)
                clipped_upper = np.minimum(np.maximum(q_h[isl, jsl, k, l], qout_min), qout_max)

                q_h[isl, jsl, k, l] = inflagL * clipped_lower + (ONE - inflagL) * clipped_upper

                # Update km1 buffers
                Qout_min_km1[:, :] = qout_min
                Qout_max_km1[:, :] = qout_max

        if adm.ADM_have_pl:
            isl_pl = slice(0, gall_pl)

            qgkl = q_pl[isl_pl, kmin:kmax+1, :]  # (gall_pl, k, l)
            qkm1 = q_pl[isl_pl, kmin-1:kmax, :]  # k-1
            qkp1 = q_pl[isl_pl, kmin+1:kmax+2, :]  # k+1

            ck0 = ck_pl[isl_pl, kmin:kmax+1, :, 0]  # (gall_pl, k, l)
            ck1 = ck_pl[isl_pl, kmin:kmax+1, :, 1]  # (gall_pl, k, l)

            inflagL = rdtype(0.5) - np.copysign(rdtype(0.5), ck0)
            inflagU = rdtype(0.5) + np.copysign(rdtype(0.5), ck_pl[isl_pl, kmin+1:kmax+2, :, 0])

            # Precompute min/max
            q_minL = np.minimum(qgkl, qkm1)
            q_minU = np.minimum(qgkl, qkp1)
            q_maxL = np.maximum(qgkl, qkm1)
            q_maxU = np.maximum(qgkl, qkp1)

            # Fuse inflag application (no np.where)
            Qin_minL = inflagL * q_minL + (rdtype(1.0) - inflagL) * BIG
            Qin_minU = inflagU * q_minU + (rdtype(1.0) - inflagU) * BIG
            Qin_maxL = inflagL * q_maxL + (rdtype(1.0) - inflagL) * -BIG
            Qin_maxU = inflagU * q_maxU + (rdtype(1.0) - inflagU) * -BIG

            # Minimize and maximize together
            qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, qgkl])
            qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, qgkl])

            # Fluxes
            Cin  = inflagL * ck0 + inflagU * ck1
            Cout = (rdtype(1.0) - inflagL) * ck0 + (rdtype(1.0) - inflagU) * ck1

            CQin_min = inflagL * ck0 * Qin_minL + inflagU * ck1 * Qin_minU
            CQin_max = inflagL * ck0 * Qin_maxL + inflagU * ck1 * Qin_maxU

            zerosw = rdtype(0.5) - np.copysign(rdtype(0.5), np.abs(Cout) - EPS)

            Cout_safe = Cout + zerosw
            nonzero_factor = rdtype(1.0) - zerosw

            d_slice_pl = d_pl[isl_pl, kmin:kmax+1, :]

            # Final limiter formulas
            Qout_min = ((qgkl - qnext_max) + qnext_max * (Cin + Cout - d_slice_pl) - CQin_max) \
                        / Cout_safe * nonzero_factor + qgkl * zerosw

            Qout_max = ((qgkl - qnext_min) + qnext_min * (Cin + Cout - d_slice_pl) - CQin_min) \
                        / Cout_safe * nonzero_factor + qgkl * zerosw

            # Save output
            Qout_min_pl[isl_pl, kmin:kmax+1, :] = Qout_min
            Qout_max_pl[isl_pl, kmin:kmax+1, :] = Qout_max


            prf.PROF_rapend('____vertical_adv_limiter',2)

        return


    def vertical_limiter_thuburn(self, 
            q_h, q_h_pl,    # [INOUT]
            q, q_pl,        # [IN]
            d, d_pl,        # [IN]
            ck, ck_pl,       # [IN]
            cnst, rdtype,
    ):

        prf.PROF_rapstart('____vertical_adv_limiter',2)

        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        lall_pl = adm.ADM_lall_pl
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax

        Qout_min_km1=np.full(adm.ADM_shape[:2], cnst.CONST_UNDEF)
        Qout_max_km1=np.full(adm.ADM_shape[:2], cnst.CONST_UNDEF)
        Qout_min_pl =np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)
        Qout_max_pl =np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF)

        EPS  = cnst.CONST_EPS
        BIG  = cnst.CONST_HUGE

        #print("BIG", BIG, cnst.CONST_HUGE)
        #prc.prc_mpifinish(std.io_l, std.fname_log)
        #import sys
        #sys.exit(0)

        for l in range(lall):
            k = kmin  # fixed slice   # kmin = 1 in python, 2 in fortran

            # Define slices
            isl = slice(0, iall)
            jsl = slice(0, jall)

            # Incoming flux flags
            inflagL = rdtype(0.5) - np.copysign(rdtype(0.5), ck[isl, jsl, k, l, 0])
            inflagU = rdtype(0.5) + np.copysign(rdtype(0.5), ck[isl, jsl, k + 1, l, 0])

            # Compute bounds with BIG trick

            Qin_minL = np.where(inflagL == rdtype(1.0),
                                np.minimum(q[isl, jsl, k, l], q[isl, jsl, k - 1, l]),
                                BIG)

            Qin_minU = np.where(inflagU == rdtype(1.0),
                                np.minimum(q[isl, jsl, k, l], q[isl, jsl, k + 1, l]),
                                BIG)

            Qin_maxL = np.where(inflagL == rdtype(1.0),
                                np.maximum(q[isl, jsl, k, l], q[isl, jsl, k - 1, l]),
                                -BIG)

            Qin_maxU = np.where(inflagU == rdtype(1.0),
                                np.maximum(q[isl, jsl, k, l], q[isl, jsl, k + 1, l]),
                                -BIG)


            qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, q[isl, jsl, k, l]])
            qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, q[isl, jsl, k, l]])

            # Cin = inflagL * ck[isl, jsl, k, l, 0] + inflagU * ck[isl, jsl, k + 1, l, 0]
            # Cout = (rdtype(1.0) - inflagL) * ck[isl, jsl, k, l, 0] + (rdtype(1.0) - inflagU) * ck[isl, jsl, k + 1, l, 0]

            Cin = inflagL * ck[isl, jsl, k, l, 0] + inflagU * ck[isl, jsl, k + 1, l, 1]
            Cout = (rdtype(1.0) - inflagL) * ck[isl, jsl, k, l, 0] + (rdtype(1.0) - inflagU) * ck[isl, jsl, k + 1, l, 1]


            # if l==1:
            #     with open(std.fname_log, 'a') as log_file:  
            #         print("Cin", Cin[6, 5], "Cout", Cout[6,5], file=log_file)   # Cout -0.005646669245168996  !!! 
            #         print("inflagL", inflagL[6, 5], file=log_file) 
            #         print("inflagU", inflagU[6, 5], file=log_file) 
            #         print("ck k+1", ck[6, 5, k+1, l, :], file=log_file)  
            #         print("ck k", ck[6, 5, k, l, :], file=log_file)

            # CQin_min = inflagL * ck[isl, jsl, k, l, 0] * Qin_minL + inflagU * ck[isl, jsl, k + 1, l, 0] * Qin_minU
            # CQin_max = inflagL * ck[isl, jsl, k, l, 0] * Qin_maxL + inflagU * ck[isl, jsl, k + 1, l, 0] * Qin_maxU

            CQin_min = inflagL * ck[isl, jsl, k, l, 0] * Qin_minL + inflagU * ck[isl, jsl, k + 1, l, 1] * Qin_minU
            CQin_max = inflagL * ck[isl, jsl, k, l, 0] * Qin_maxL + inflagU * ck[isl, jsl, k + 1, l, 1] * Qin_maxU


            #zerosw = rdtype(0.5) - np.sign(rdtype(0.5), np.abs(Cout) - EPS)
            zerosw = rdtype(0.5) - np.copysign(rdtype(0.5), np.abs(Cout) - EPS)

            # Output limits
            Qout_min_k = (
                ((q[isl, jsl, k, l] - qnext_max) + qnext_max * (Cin + Cout - d[isl, jsl, k, l]) - CQin_max)
                / (Cout + zerosw) * (rdtype(1.0) - zerosw) + q[isl, jsl, k, l] * zerosw
            )

            Qout_max_k = (
                ((q[isl, jsl, k, l] - qnext_min) + qnext_min * (Cin + Cout - d[isl, jsl, k, l]) - CQin_min)
                / (Cout + zerosw) * (rdtype(1.0) - zerosw) + q[isl, jsl, k, l] * zerosw
            )
                                                            # Cout
            #  ((0.9 - 0.0) + 0.0 * (***) - 0.0) / (-0.005646669245168996 + 0.) * (1. - 0.) + 0.9 * 0.0

            # if l==1:
            #     with open(std.fname_log, 'a') as log_file:
            #         print("Qout_max_k", Qout_max_k[6, 5], file=log_file) #Qout_max_k  -159.38599569471765
            #         print("q", q[6, 5, k, l], "qnext_min", qnext_min[6, 5], file=log_file)    
            #         print("Cin", Cin[6, 5], "Cout", Cout[6,5], file=log_file)   # Cout -0.005646669245168996  !!! 
            #         print("zerosw", zerosw[6, 5], file=log_file)
            #         print("CQin_min", CQin_min[6, 5], file=log_file)



            # Store to arrays
            Qout_min_km1[isl, jsl] = Qout_min_k
            Qout_max_km1[isl, jsl] = Qout_max_k     #Qout_max_km1 -159.38599569471765



            for k in range(kmin + 1, kmax + 1):
                # Precompute commonly used variables
                #inflagL = rdtype(0.5) - np.sign(rdtype(0.5), ck[isl, jsl, k, l, 0])  # ck[..., 1] in Fortran = ck[..., 0] in Python
                #inflagU = rdtype(0.5) + np.sign(rdtype(0.5), ck[isl, jsl, k + 1, l, 0])
                inflagL = rdtype(0.5) - np.copysign(rdtype(0.5), ck[isl, jsl, k, l, 0])  # ck[..., 1] in Fortran = ck[..., 0] in Python
                inflagU = rdtype(0.5) + np.copysign(rdtype(0.5), ck[isl, jsl, k + 1, l, 0])

                q_center = q[isl, jsl, k, l]
                q_below  = q[isl, jsl, k - 1, l]
                q_above  = q[isl, jsl, k + 1, l]


                Qin_minL = np.where(inflagL == rdtype(1.0),
                                    np.minimum(q_center, q_below),
                                    BIG)

                Qin_minU = np.where(inflagU == rdtype(1.0),
                                    np.minimum(q_center, q_above),
                                    BIG)

                Qin_maxL = np.where(inflagL == rdtype(1.0),
                                    np.maximum(q_center, q_below),
                                    -BIG)

                Qin_maxU = np.where(inflagU == rdtype(1.0),
                                    np.maximum(q_center, q_above),
                                    -BIG)

                qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, q_center])
                qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, q_center])

                ck1 = ck[isl, jsl, k, l, 0]
                ck2 = ck[isl, jsl, k, l, 1]

                Cin = inflagL * ck1 + inflagU * ck2
                Cout = (rdtype(1.0) - inflagL) * ck1 + (rdtype(1.0) - inflagU) * ck2

                CQin_min = inflagL * ck1 * Qin_minL + inflagU * ck2 * Qin_minU
                CQin_max = inflagL * ck1 * Qin_maxL + inflagU * ck2 * Qin_maxU

                #zerosw = rdtype(0.5) - np.sign(rdtype(0.5), np.abs(Cout) - EPS)
                zerosw = rdtype(0.5) - np.copysign(rdtype(0.5), np.abs(Cout) - EPS)

                qout_min_k = (
                    ((q_center - qnext_max) + qnext_max * (Cin + Cout - d[isl, jsl, k, l]) - CQin_max)
                    / (Cout + zerosw) * (rdtype(1.0) - zerosw) + q_center * zerosw
                )

                qout_max_k = (
                    ((q_center - qnext_min) + qnext_min * (Cin + Cout - d[isl, jsl, k, l]) - CQin_min)
                    / (Cout + zerosw) * (rdtype(1.0) - zerosw) + q_center * zerosw
                )

                # if k==2 and l==1:
                #     with open(std.fname_log, 'a') as log_file:
                #         print("q_h", q_h[6, 5, k, l], "Qout_min_km1", Qout_min_km1[6, 5], "Qout_max_km1", Qout_max_km1[6, 5], file=log_file)    #Qout_max_km1 -159.38599569471765
                #         print("qout_min_k", qout_min_k[6, 5], "qout_max_k", qout_max_k[6,5], file=log_file)
                #         print("q_h", q_h[6, 5, k, l], file=log_file)
                #         print("inflagL", inflagL[6, 5], file=log_file)

                # Clip q_h using inflagL
                q_h[isl, jsl, k, l] = (
                    inflagL * np.clip(q_h[isl, jsl, k, l], Qout_min_km1[isl, jsl], Qout_max_km1[isl, jsl]) +
                    (rdtype(1.0) - inflagL) * np.clip(q_h[isl, jsl, k, l], qout_min_k, qout_max_k)
                )

                # Update for next level
                Qout_min_km1[isl, jsl] = qout_min_k
                Qout_max_km1[isl, jsl] = qout_max_k
            # end loop k
        # end loop l

        if adm.ADM_have_pl:

            qgkl = q_pl[:, kmin:kmax+1, :]  # shape (g, k, l)
            qkm1 = q_pl[:, kmin-1:kmax, :]  # k-1
            qkp1 = q_pl[:, kmin+1:kmax+2, :]  # k+1

            ck0 = ck_pl[:, kmin:kmax+1, :, 0]
            ck1 = ck_pl[:, kmin:kmax+1, :, 1]

            inflagL = rdtype(0.5) - np.copysign(rdtype(0.5), ck0)
            inflagU = rdtype(0.5) + np.copysign(rdtype(0.5), ck_pl[:, kmin+1:kmax+2, :, 0])
           
            Qin_minL = np.where(inflagL == rdtype(1.0), np.minimum(qgkl, qkm1), BIG)
            Qin_minU = np.where(inflagU == rdtype(1.0), np.minimum(qgkl, qkp1), BIG)
            Qin_maxL = np.where(inflagL == rdtype(1.0), np.maximum(qgkl, qkm1), -BIG)
            Qin_maxU = np.where(inflagU == rdtype(1.0), np.maximum(qgkl, qkp1), -BIG)

            qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, qgkl])
            qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, qgkl])

            Cin  = inflagL * ck0 + inflagU * ck1
            Cout = (rdtype(1.0) - inflagL) * ck0 + (rdtype(1.0) - inflagU) * ck1

            CQin_min = inflagL * ck0 * Qin_minL + inflagU * ck1 * Qin_minU
            CQin_max = inflagL * ck0 * Qin_maxL + inflagU * ck1 * Qin_maxU

            zerosw = rdtype(0.5) - np.copysign(rdtype(0.5), np.abs(Cout) - EPS)

            Qout_min = ((qgkl - qnext_max) + qnext_max * (Cin + Cout - d_pl[:, kmin:kmax+1, :]) - CQin_max) \
                    / (Cout + zerosw) * (rdtype(1.0) - zerosw) + qgkl * zerosw

            Qout_max = ((qgkl - qnext_min) + qnext_min * (Cin + Cout - d_pl[:, kmin:kmax+1, :]) - CQin_min) \
                    / (Cout + zerosw) * (rdtype(1.0) - zerosw) + qgkl * zerosw

            Qout_min_pl[:, kmin:kmax+1, :] = Qout_min
            Qout_max_pl[:, kmin:kmax+1, :] = Qout_max


            for l in range(lall_pl):
                for k in range(kmin + 1, kmax + 1):
                    for g in range(gall_pl):
                        inflagL = rdtype(0.5) - np.copysign(rdtype(0.5), ck_pl[g, k, l, 0])
                        q_h_pl[g, k, l] = (
                            inflagL * np.clip(q_h_pl[g, k, l], Qout_min_pl[g, k - 1, l], Qout_max_pl[g, k - 1, l])
                            + (rdtype(1.0) - inflagL) * np.clip(q_h_pl[g, k, l], Qout_min_pl[g, k, l], Qout_max_pl[g, k, l])
                        )

            # for l in range(lall_pl):
            #     for k in range(kmin, kmax + 1):
            #         for g in range(gall_pl):
            #             #inflagL = rdtype(0.5) - np.sign(rdtype(0.5), ck_pl[g, k, l, 0])
            #             inflagL = np.copysign(rdtype(0.5), ck[g, k, l, 1])
            #             #inflagU = rdtype(0.5) + np.sign(rdtype(0.5), ck_pl[g, k + 1, l, 0])
            #             inflagU = rdtype(0.5) + np.copysign(rdtype(0.5), ck_pl[g, k + 1, l, 0])

            #             qgkl = q_pl[g, k, l]
            #             Qin_minL = min(qgkl, q_pl[g, k - 1, l]) + (rdtype(1.0) - inflagL) * BIG
            #             Qin_minU = min(qgkl, q_pl[g, k + 1, l]) + (rdtype(1.0) - inflagU) * BIG
            #             Qin_maxL = max(qgkl, q_pl[g, k - 1, l]) - (rdtype(1.0) - inflagL) * BIG
            #             Qin_maxU = max(qgkl, q_pl[g, k + 1, l]) - (rdtype(1.0) - inflagU) * BIG

            #             qnext_min = np.minimum(np.minimum(Qin_minL, Qin_minU), qgkl)
            #             qnext_max = np.maximum(np.maximum(Qin_maxL, Qin_maxU), qgkl)
            #             #qnext_min = np.minimum.reduce([Qin_minL, Qin_minU, qgkl])
            #             #qnext_max = np.maximum.reduce([Qin_maxL, Qin_maxU, qgkl])

            #             ck0 = ck_pl[g, k, l, 0]
            #             ck1 = ck_pl[g, k, l, 1]
            #             Cin  = inflagL * ck0 + inflagU * ck1
            #             Cout = (rdtype(1.0) - inflagL) * ck0 + (rdtype(1.0) - inflagU) * ck1

            #             CQin_min = inflagL * ck0 * Qin_minL + inflagU * ck1 * Qin_minU
            #             CQin_max = inflagL * ck0 * Qin_maxL + inflagU * ck1 * Qin_maxU

            #             zerosw = rdtype(0.5) - np.sign(rdtype(0.5), abs(Cout) - EPS)

            #             #Qout_min_pl[g, k] = ((qgkl - qnext_max) + qnext_max * (Cin + Cout - d_pl[g, k, l]) - CQin_max) / (Cout + zerosw) * (rdtype(1.0) - zerosw) + qgkl * zerosw
            #             #Qout_max_pl[g, k] = ((qgkl - qnext_min) + qnext_min * (Cin + Cout - d_pl[g, k, l]) - CQin_min) / (Cout + zerosw) * (rdtype(1.0) - zerosw) + qgkl * zerosw


            #             Qout_min = ((qgkl - qnext_max) + qnext_max * (Cin + Cout - d_pl[g, k, l]) - CQin_max) / (Cout + zerosw) * (rdtype(1.0) - zerosw) + qgkl * zerosw
            #             Qout_max = ((qgkl - qnext_min) + qnext_min * (Cin + Cout - d_pl[g, k, l]) - CQin_min) / (Cout + zerosw) * (rdtype(1.0) - zerosw) + qgkl * zerosw

            #             print(Qout_min.shape, Qout_max.shape) #, Qout_min.dtype, Qout_max.dtype)
            #             print(qgkl.shape, qnext_max.shape, qnext_min.shape) #, qgkl.dtype, qnext_max.dtype, qnext_min.dtype)
            #             print(Cout.shape, zerosw.shape) #, Cout.dtype, zerosw.dtype)
            #             print(Cin.shape, d_pl[g, k, l].shape) #, Cin.dtype, d_pl[g, k, l].dtype)
            #             print(CQin_min.shape, CQin_max.shape) #, CQin_min.dtype, CQin_max.dtype)
            #             Qout_min_pl[g, k] = Qout_min
            #             Qout_max_pl[g, k] = Qout_max
                        

            #         # end loop g
            #     # end loop k

            #     for k in range(kmin + 1, kmax + 1):
            #         for g in range(gall_pl):
            #             inflagL = rdtype(0.5) - np.sign(rdtype(0.5), ck_pl[g, k, l, 0])
            #             q_h_pl[g, k, l] = (
            #                 inflagL * np.clip(q_h_pl[g, k, l], Qout_min_pl[g, k - 1], Qout_max_pl[g, k - 1])
            #                 + (rdtype(1.0) - inflagL) * np.clip(q_h_pl[g, k, l], Qout_min_pl[g, k], Qout_max_pl[g, k])
            #             )
                    # end loop g
                # end loop k
            # end loop l
        # end if 

#####
        # if adm.ADM_have_pl:
        #     npl = adm.ADM_gslf_pl
        #     for l in range(lall_pl):
        #         for k in range(kmin, kmax + 1):
        #             for g in range(gall_pl):
        #                 inflagL = 0.5 - np.sign(0.5, ck_pl[g, k, l, 0])
        #                 inflagU = 0.5 + np.sign(0.5, ck_pl[g, k + 1, l, 0])

        #                 qgkl = q_pl[g, k, l]
        #                 Qin_minL = min(qgkl, q_pl[g, k - 1, l]) + (1.0 - inflagL) * BIG
        #                 Qin_minU = min(qgkl, q_pl[g, k + 1, l]) + (1.0 - inflagU) * BIG
        #                 Qin_maxL = max(qgkl, q_pl[g, k - 1, l]) - (1.0 - inflagL) * BIG
        #                 Qin_maxU = max(qgkl, q_pl[g, k + 1, l]) - (1.0 - inflagU) * BIG

        #                 qnext_min = min(Qin_minL, Qin_minU, qgkl)
        #                 qnext_max = max(Qin_maxL, Qin_maxU, qgkl)

        #                 ck1 = ck_pl[g, k, l, 0]
        #                 ck2 = ck_pl[g, k, l, 1]
        #                 Cin  = inflagL * ck1 + inflagU * ck2
        #                 Cout = (1.0 - inflagL) * ck1 + (1.0 - inflagU) * ck2

        #                 CQin_min = inflagL * ck1 * Qin_minL + inflagU * ck2 * Qin_minU
        #                 CQin_max = inflagL * ck1 * Qin_maxL + inflagU * ck2 * Qin_maxU

        #                 zerosw = 0.5 - np.sign(0.5, abs(Cout) - EPS)

        #                 Qout_min = ((qgkl - qnext_max) + qnext_max * (Cin + Cout - d_pl[g, k, l]) - CQin_max) / (Cout + zerosw) * (1.0 - zerosw) + qgkl * zerosw
        #                 Qout_max = ((qgkl - qnext_min) + qnext_min * (Cin + Cout - d_pl[g, k, l]) - CQin_min) / (Cout + zerosw) * (1.0 - zerosw) + qgkl * zerosw

        #                 Qout_min_pl[g, k] = Qout_min
        #                 Qout_max_pl[g, k] = Qout_max
        #             # end loop g
        #         # end loop k

        #         for k in range(kmin + 1, kmax + 1):
        #             for g in range(gall_pl):
        #                 inflagL = 0.5 - np.sign(0.5, ck_pl[g, k, l, 0])
        #                 q_h_pl[g, k, l] = (
        #                     inflagL * np.clip(q_h_pl[g, k, l], Qout_min_pl[g, k - 1], Qout_max_pl[g, k - 1])
        #                     + (1.0 - inflagL) * np.clip(q_h_pl[g, k, l], Qout_min_pl[g, k], Qout_max_pl[g, k])
        #                 )
        #             # end loop g
        #         # end loop k
        #     # end loop l
        # # end if 

        ###

        prf.PROF_rapend('____vertical_adv_limiter',2)

        return    
    
    #> Miura(2004)'s scheme with Thuburn(1996) limiter
    def horizontal_limiter_thuburn(self,
        q_a,    q_a_pl,             # [INOUT]    
        q,      q_pl,               # [IN]
        d,      d_pl,               # [IN]
        ch,     ch_pl,              # [IN]
        cmask,  cmask_pl,           # [IN]
        cnst, comm, rdtype,
    ):
        
        prf.PROF_rapstart('____horizontal_adv_limiter',2)

        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        lall_pl = adm.ADM_lall_pl
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax

        I_min = 0
        I_max = 1

        Qin    = np.full(adm.ADM_shape + (2, 6,),  cnst.CONST_UNDEF)
        Qin_pl = np.full(adm.ADM_shape_pl + (2, 2,),  cnst.CONST_UNDEF)
        Qout   = np.full(adm.ADM_shape + (2,),  cnst.CONST_UNDEF)
        Qout_pl= np.full(adm.ADM_shape_pl + (2,),  cnst.CONST_UNDEF)

        # Qin    = np.zeros(adm.ADM_shape + (2,6,), dtype=rdtype)    # set to zero to suppress Possible bug in this scheme 
        # Qin_pl = np.zeros(adm.ADM_shape_pl + (2,2,),  dtype=rdtype)
        # Qout   = np.zeros(adm.ADM_shape + (2,),  dtype=rdtype)
        # Qout_pl= np.zeros(adm.ADM_shape_pl + (2,),  dtype=rdtype)


        EPS  = cnst.CONST_EPS
        BIG  = cnst.CONST_HUGE

        prf.PROF_rapend  ('____horizontal_adv_limiter',2)


        ############  WORKS, and faster, but still has i and j loops #########
        for i in range(iall-1):
            for j in range(jall-1):
                # Build the 4-point stencil at (i,j)
                q0 = q[i,   j,   :, :]  # center
                if j > 0:
                    q1 = q[i,   j-1, :, :]
                else:
                    q1 = q[0,   0,   :, :]
                
                q2 = q[i+1, j,   :, :]
                q3 = q[i+1, j+1, :, :]

                # For AI
                q_min_AI  = np.minimum.reduce([q0, q1, q2, q3])
                q_max_AI  = np.maximum.reduce([q0, q1, q2, q3])

                # For AIJ (no special boundary handling)
                q_min_AIJ = np.minimum.reduce([q0, q2, q3, q[i, j+1, :, :]])
                q_max_AIJ = np.maximum.reduce([q0, q2, q3, q[i, j+1, :, :]])

                # For AJ
                if i > 0:
                    q4 = q[i-1, j, :, :]
                else:
                    q4 = q[0, 0, :, :]
                q_min_AJ = np.minimum.reduce([q0, q3, q[i, j+1, :, :], q4])
                q_max_AJ = np.maximum.reduce([q0, q3, q[i, j+1, :, :], q4])

                # Now fill Qin
                for m, (qmin, qmax) in enumerate([(q_min_AI, q_max_AI), (q_min_AIJ, q_max_AIJ), (q_min_AJ, q_max_AJ)]):
                    Qin[i, j, :, :, I_min, m] = cmask[i, j, :, :, m] * qmin + (1.0 - cmask[i, j, :, :, m]) * BIG
                    Qin[i, j, :, :, I_max, m] = cmask[i, j, :, :, m] * qmax + (1.0 - cmask[i, j, :, :, m]) * (-BIG)

                # For shifted points (neighbors)
                # For AI and AIJ -> (i+1, j), (i+1, j+1)
                Qin[i+1, j,   :, :, I_min, 3] = cmask[i, j, :, :, 0] * BIG + (1.0 - cmask[i, j, :, :, 0]) * q_min_AI
                Qin[i+1, j,   :, :, I_max, 3] = cmask[i, j, :, :, 0] * (-BIG) + (1.0 - cmask[i, j, :, :, 0]) * q_max_AI

                Qin[i+1, j+1, :, :, I_min, 4] = cmask[i, j, :, :, 1] * BIG + (1.0 - cmask[i, j, :, :, 1]) * q_min_AIJ
                Qin[i+1, j+1, :, :, I_max, 4] = cmask[i, j, :, :, 1] * (-BIG) + (1.0 - cmask[i, j, :, :, 1]) * q_max_AIJ

                # For AJ -> (i, j+1)
                Qin[i,   j+1, :, :, I_min, 5] = cmask[i, j, :, :, 2] * BIG + (1.0 - cmask[i, j, :, :, 2]) * q_min_AJ
                Qin[i,   j+1, :, :, I_max, 5] = cmask[i, j, :, :, 2] * (-BIG) + (1.0 - cmask[i, j, :, :, 2]) * q_max_AJ

        # ###########################



        # #### WORKS, but slow ###

        # for l in range(lall):
        #     for k in range(kall):

        #         for j in range(jall - 1):     # Python: 0 to jall-2
        #             for i in range(iall - 1):

        #                 # Handling boundaries (im1j, ijm1 logic is ignored as you seem to apply 1-clamping in the original)
        #                 if i > 0 and j > 0:
        #                     q_min_AI  = np.min([q[i, j, k, l], q[i, j-1, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l]])
        #                     q_max_AI  = np.max([q[i, j, k, l], q[i, j-1, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l]])
        #                     q_min_AIJ = np.min([q[i, j, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l]])
        #                     q_max_AIJ = np.max([q[i, j, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l]])
        #                     q_min_AJ  = np.min([q[i, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l], q[i-1, j, k, l]])
        #                     q_max_AJ  = np.max([q[i, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l], q[i-1, j, k, l]])
        #                 else:
        #                     q_min_AI  = np.min([q[i, j, k, l], q[0, 0, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l]])
        #                     q_max_AI  = np.max([q[i, j, k, l], q[0, 0, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l]])
        #                     q_min_AIJ = np.min([q[i, j, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l]])
        #                     q_max_AIJ = np.max([q[i, j, k, l], q[i+1, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l]])
        #                     q_min_AJ  = np.min([q[i, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l], q[0, 0, k, l]])
        #                     q_max_AJ  = np.max([q[i, j, k, l], q[i+1, j+1, k, l], q[i, j+1, k, l], q[0, 0, k, l]])

        #                 # Now filling Qin array
        #                 Qin[i,   j,   k, l, I_min, 0] = cmask[i, j, k, l, 0] * q_min_AI  + (1.0 - cmask[i, j, k, l, 0]) * BIG
        #                 Qin[i+1, j,   k, l, I_min, 3] = cmask[i, j, k, l, 0] * BIG       + (1.0 - cmask[i, j, k, l, 0]) * q_min_AI
        #                 Qin[i,   j,   k, l, I_max, 0] = cmask[i, j, k, l, 0] * q_max_AI  + (1.0 - cmask[i, j, k, l, 0]) * (-BIG)
        #                 Qin[i+1, j,   k, l, I_max, 3] = cmask[i, j, k, l, 0] * (-BIG)    + (1.0 - cmask[i, j, k, l, 0]) * q_max_AI

        #                 Qin[i,   j,   k, l, I_min, 1] = cmask[i, j, k, l, 1] * q_min_AIJ + (1.0 - cmask[i, j, k, l, 1]) * BIG
        #                 Qin[i+1, j+1, k, l, I_min, 4] = cmask[i, j, k, l, 1] * BIG       + (1.0 - cmask[i, j, k, l, 1]) * q_min_AIJ
        #                 Qin[i,   j,   k, l, I_max, 1] = cmask[i, j, k, l, 1] * q_max_AIJ + (1.0 - cmask[i, j, k, l, 1]) * (-BIG)
        #                 Qin[i+1, j+1, k, l, I_max, 4] = cmask[i, j, k, l, 1] * (-BIG)    + (1.0 - cmask[i, j, k, l, 1]) * q_max_AIJ

        #                 Qin[i,   j,   k, l, I_min, 2] = cmask[i, j, k, l, 2] * q_min_AJ  + (1.0 - cmask[i, j, k, l, 2]) * BIG
        #                 Qin[i,   j+1, k, l, I_min, 5] = cmask[i, j, k, l, 2] * BIG       + (1.0 - cmask[i, j, k, l, 2]) * q_min_AJ
        #                 Qin[i,   j,   k, l, I_max, 2] = cmask[i, j, k, l, 2] * q_max_AJ  + (1.0 - cmask[i, j, k, l, 2]) * (-BIG)
        #                 Qin[i,   j+1, k, l, I_max, 5] = cmask[i, j, k, l, 2] * (-BIG)    + (1.0 - cmask[i, j, k, l, 2]) * q_max_AJ


        # ####################



#         for l in range(lall):
#             for k in range(kall):
#                 # Define slices for interior region
#                 # isl = slice(1, iall - 1)   # size 16
#                 # jsl = slice(1, jall - 1)
#                 # islp1 = slice(2, iall)
#                 # jslp1 = slice(2, jall)
#                 # islm1 = slice(0, iall - 2)
#                 # jslm1 = slice(0, jall - 2)

#                 isl = slice(1, iall - 1)   # size 16
#                 jsl = slice(1, jall - 1)
#                 islp1 = slice(2, iall)
#                 jslp1 = slice(2, jall)
#                 islm1 = slice(0, iall - 2)
#                 jslm1 = slice(0, jall - 2)


#                 # Local slices for broadcasting
# #                cm1 = rdtype(1.0) - cmask[isl, jsl, k, l]   # dimension???
#                 cm1 = rdtype(1.0) - cmask[isl, jsl, k, l, :]   # dimension???  

#                 # q_min and q_max for each stencil
#                 q_min_AI  = np.minimum.reduce([q[isl, jsl, k, l], q[isl, jslm1, k, l], q[islp1, jsl, k, l], q[islp1, jslp1, k, l]])
#                 q_max_AI  = np.maximum.reduce([q[isl, jsl, k, l], q[isl, jslm1, k, l], q[islp1, jsl, k, l], q[islp1, jslp1, k, l]])
#                 q_min_AIJ = np.minimum.reduce([q[isl, jsl, k, l], q[islp1, jsl, k, l], q[islp1, jslp1, k, l], q[isl, jslp1, k, l]])
#                 q_max_AIJ = np.maximum.reduce([q[isl, jsl, k, l], q[islp1, jsl, k, l], q[islp1, jslp1, k, l], q[isl, jslp1, k, l]])
#                 q_min_AJ  = np.minimum.reduce([q[isl, jsl, k, l], q[islp1, jslp1, k, l], q[isl, jslp1, k, l], q[islm1, jsl, k, l]])
#                 q_max_AJ  = np.maximum.reduce([q[isl, jsl, k, l], q[islp1, jslp1, k, l], q[isl, jslp1, k, l], q[islm1, jsl, k, l]])

#                 # min/max indices

#                 Qin[isl,   jsl,   k, l, I_min, 0] = np.where(cmask[isl, jsl, k, l, 0] == rdtype(1.0), q_min_AI,  BIG)
#                 Qin[islp1, jsl,   k, l, I_min, 3] = np.where(cmask[isl, jsl, k, l, 0] == rdtype(1.0), BIG,       q_min_AI)
#                 Qin[isl,   jsl,   k, l, I_max, 0] = np.where(cmask[isl, jsl, k, l, 0] == rdtype(1.0), q_max_AI, -BIG)
#                 Qin[islp1, jsl,   k, l, I_max, 3] = np.where(cmask[isl, jsl, k, l, 0] == rdtype(1.0), -BIG,      q_max_AI)

#                 Qin[isl,   jsl,   k, l, I_min, 1] = np.where(cmask[isl, jsl, k, l, 1] == rdtype(1.0), q_min_AIJ,  BIG)
#                 Qin[islp1, jslp1, k, l, I_min, 4] = np.where(cmask[isl, jsl, k, l, 1] == rdtype(1.0), BIG,       q_min_AIJ)
#                 Qin[isl,   jsl,   k, l, I_max, 1] = np.where(cmask[isl, jsl, k, l, 1] == rdtype(1.0), q_max_AIJ, -BIG)
#                 Qin[islp1, jslp1, k, l, I_max, 4] = np.where(cmask[isl, jsl, k, l, 1] == rdtype(1.0), -BIG,      q_max_AIJ)

#                 Qin[isl,   jsl,   k, l, I_min, 2] = np.where(cmask[isl, jsl, k, l, 2] == rdtype(1.0), q_min_AJ,  BIG)
#                 Qin[isl,   jslp1, k, l, I_min, 5] = np.where(cmask[isl, jsl, k, l, 2] == rdtype(1.0), BIG,       q_min_AJ)
#                 Qin[isl,   jsl,   k, l, I_max, 2] = np.where(cmask[isl, jsl, k, l, 2] == rdtype(1.0), q_max_AJ, -BIG)
#                 Qin[isl,   jslp1, k, l, I_max, 5] = np.where(cmask[isl, jsl, k, l, 2] == rdtype(1.0), -BIG,      q_max_AJ)


#                 #   QQQ1

#                 #         print("q_minmax", q_min_AI, q_min_AIJ, q_min_AJ, q_max_AI, q_max_AIJ, q_max_AJ, file=log_file )
#                 #         print("HLT: Qin 1st: I_min", file=log_file)
#                 #         print(Qin[0, 0, k, l, I_min, :], file=log_file)
#                 #         print(Qin[1, 1, k, l, I_min, :], file=log_file)
#                 #         print(Qin[5, 5, k, l, I_min, :], file=log_file)
#                 #         print("              I_max", file=log_file)
#                 #         print(Qin[0, 0, k, l, I_max, :], file=log_file)
#                 #         print(Qin[1, 1, k, l, I_max, :], file=log_file)
#                 #         print(Qin[5, 5, k, l, I_max, :], file=log_file)
#                 #         print("              cmask", file=log_file)
#                 #         print(cmask[0, 0, k, l, :], file=log_file)
#                 #         print(cmask[1, 1, k, l, :], file=log_file)
#                 #         print(cmask[5, 5, k, l, :], file=log_file)

#                 # < edge treatment for i=0 >
#                 jv = np.arange(1, jall - 1)  # j = 2 to jall-1 (Python 0-based)
#                 i = 0
#                 ip1 = i + 1
#                 jp1 = jv + 1
#                 jm1 = jv - 1

#                 # Extract local cmask slices
#                 cmask0 = cmask[i, jv, k, l, 0]
#                 cmask1 = cmask[i, jv, k, l, 1]
#                 cmask2 = cmask[i, jv, k, l, 2]

#                 # q_min/q_max calculations
#                 q_min_AI  = np.minimum.reduce([q[i, jv,   k, l], q[i, jm1, k, l], q[ip1, jv,   k, l], q[ip1, jp1, k, l]])
#                 q_max_AI  = np.maximum.reduce([q[i, jv,   k, l], q[i, jm1, k, l], q[ip1, jv,   k, l], q[ip1, jp1, k, l]])
#                 q_min_AIJ = np.minimum.reduce([q[i, jv,   k, l], q[ip1, jv,   k, l], q[ip1, jp1, k, l], q[i, jp1, k, l]])
#                 q_max_AIJ = np.maximum.reduce([q[i, jv,   k, l], q[ip1, jv,   k, l], q[ip1, jp1, k, l], q[i, jp1, k, l]])
#                 q_min_AJ  = np.minimum.reduce([q[i, jv,   k, l], q[ip1, jp1, k, l], q[i, jp1, k, l], q[i, jv, k, l]])
#                 q_max_AJ  = np.maximum.reduce([q[i, jv,   k, l], q[ip1, jp1, k, l], q[i, jp1, k, l], q[i, jv, k, l]])

#                 # Assign to Qin
#                 Qin[i, jv,    k, l, I_min, 0] = np.where(cmask0 == rdtype(1.0), q_min_AI,  BIG)
#                 Qin[ip1, jv,  k, l, I_min, 3] = np.where(cmask0 == rdtype(1.0),     BIG,  q_min_AI)
#                 Qin[i, jv,    k, l, I_max, 0] = np.where(cmask0 == rdtype(1.0), q_max_AI, -BIG)
#                 Qin[ip1, jv,  k, l, I_max, 3] = np.where(cmask0 == rdtype(1.0),   -BIG,  q_max_AI)

#                 Qin[i, jv,    k, l, I_min, 1] = np.where(cmask1 == rdtype(1.0), q_min_AIJ,  BIG)
#                 Qin[ip1, jp1, k, l, I_min, 4] = np.where(cmask1 == rdtype(1.0),     BIG,  q_min_AIJ)
#                 Qin[i, jv,    k, l, I_max, 1] = np.where(cmask1 == rdtype(1.0), q_max_AIJ, -BIG)
#                 Qin[ip1, jp1, k, l, I_max, 4] = np.where(cmask1 == rdtype(1.0),   -BIG,  q_max_AIJ)

#                 Qin[i, jv,    k, l, I_min, 2] = np.where(cmask2 == rdtype(1.0), q_min_AJ,  BIG)
#                 Qin[i, jp1,   k, l, I_min, 5] = np.where(cmask2 == rdtype(1.0),     BIG,  q_min_AJ)
#                 Qin[i, jv,    k, l, I_max, 2] = np.where(cmask2 == rdtype(1.0), q_max_AJ, -BIG)
#                 Qin[i, jp1,   k, l, I_max, 5] = np.where(cmask2 == rdtype(1.0),   -BIG,  q_max_AJ)

#                 # if l==1 and k==7:
#                 #     with open(std.fname_log, 'a') as log_file:
#                 #         print("HLT: Qin 2nd: I_min", file=log_file)
#                 #         print(Qin[0, 0, k, l, I_min, :], file=log_file)
#                 #         print(Qin[1, 1, k, l, I_min, :], file=log_file)
#                 #         print(Qin[5, 5, k, l, I_min, :], file=log_file)
#                 #         print("              I_max", file=log_file)
#                 #         print(Qin[0, 0, k, l, I_max, :], file=log_file)
#                 #         print(Qin[1, 1, k, l, I_max, :], file=log_file)
#                 #         print(Qin[5, 5, k, l, I_max, :], file=log_file)

#                 # < edge treatment for j=0 >
#                 iv = np.arange(1, iall - 1)  # i = 2 to iall-1 in Fortran
#                 j = 0
#                 ip1 = iv + 1
#                 jp1 = j + 1
#                 im1 = iv - 1

#                 # Extract cmask components
#                 cmask0 = cmask[iv, j, k, l, 0]
#                 cmask1 = cmask[iv, j, k, l, 1]
#                 cmask2 = cmask[iv, j, k, l, 2]

#                 # Compute min/max values
#                 q_min_AI  = np.minimum.reduce([q[iv, j,   k, l], q[iv, j,   k, l], q[ip1, j,   k, l], q[ip1, jp1, k, l]])
#                 q_max_AI  = np.maximum.reduce([q[iv, j,   k, l], q[iv, j,   k, l], q[ip1, j,   k, l], q[ip1, jp1, k, l]])
#                 q_min_AIJ = np.minimum.reduce([q[iv, j,   k, l], q[ip1, j,   k, l], q[ip1, jp1, k, l], q[iv, jp1, k, l]])
#                 q_max_AIJ = np.maximum.reduce([q[iv, j,   k, l], q[ip1, j,   k, l], q[ip1, jp1, k, l], q[iv, jp1, k, l]])
#                 q_min_AJ  = np.minimum.reduce([q[iv, j,   k, l], q[ip1, jp1, k, l], q[iv, jp1, k, l], q[im1, j, k, l]])
#                 q_max_AJ  = np.maximum.reduce([q[iv, j,   k, l], q[ip1, jp1, k, l], q[iv, jp1, k, l], q[im1, j, k, l]])

#                 # Assign to Qin arrays
#                 Qin[iv,  j,   k, l, I_min, 0] = np.where(cmask0 == rdtype(1.0), q_min_AI,  BIG)
#                 Qin[ip1, j,   k, l, I_min, 3] = np.where(cmask0 == rdtype(1.0),     BIG,  q_min_AI)
#                 Qin[iv,  j,   k, l, I_max, 0] = np.where(cmask0 == rdtype(1.0), q_max_AI, -BIG)
#                 Qin[ip1, j,   k, l, I_max, 3] = np.where(cmask0 == rdtype(1.0),   -BIG,  q_max_AI)

#                 Qin[iv,  j,   k, l, I_min, 1] = np.where(cmask1 == rdtype(1.0), q_min_AIJ,  BIG)
#                 Qin[ip1, jp1, k, l, I_min, 4] = np.where(cmask1 == rdtype(1.0),     BIG,  q_min_AIJ)
#                 Qin[iv,  j,   k, l, I_max, 1] = np.where(cmask1 == rdtype(1.0), q_max_AIJ, -BIG)
#                 Qin[ip1, jp1, k, l, I_max, 4] = np.where(cmask1 == rdtype(1.0),   -BIG,  q_max_AIJ)

#                 Qin[iv,  j,   k, l, I_min, 2] = np.where(cmask2 == rdtype(1.0), q_min_AJ,  BIG)
#                 Qin[iv,  jp1, k, l, I_min, 5] = np.where(cmask2 == rdtype(1.0),     BIG,  q_min_AJ)
#                 Qin[iv,  j,   k, l, I_max, 2] = np.where(cmask2 == rdtype(1.0), q_max_AJ, -BIG)
#                 Qin[iv,  jp1, k, l, I_max, 5] = np.where(cmask2 == rdtype(1.0),   -BIG,  q_max_AJ)



                # if l==1 and k==7:
                #     with open(std.fname_log, 'a') as log_file:
                #         #print("QQQ1", file=log_file)
                #         print("shape", Qin.shape, Qout.shape, file=log_file)
                #         print("QQQ1 Qin min", Qin[1, 1, k, l, I_min, :], file=log_file)
                #         print("QQQ1 Qin max", Qin[1, 1, k, l, I_max, :], file=log_file)
                #         print("QQQ1 Qout", Qout[1, 1, k, l, :], file=log_file)

                ### CORNER treatment for i=0, j=0 missing in vectorized version $$$$$ 

                # if l==1 and k==7:
                #     with open(std.fname_log, 'a') as log_file:
                #         print("HLT: Qin 3rd: I_min", file=log_file)
                #         print(Qin[0, 0, k, l, I_min, :], file=log_file)
                #         print(Qin[1, 1, k, l, I_min, :], file=log_file)
                #         print(Qin[5, 5, k, l, I_min, :], file=log_file)
                #         print("              I_max", file=log_file)
                #         print(Qin[0, 0, k, l, I_max, :], file=log_file)
                #         print(Qin[1, 1, k, l, I_max, :], file=log_file)
                #         print(Qin[5, 5, k, l, I_max, :], file=log_file)

        for l in range(lall):
            for k in range(kall):

                if adm.ADM_have_sgp[l]:
                    i, j = 0, 0  

                    ip1 = i + 1
                    ip2 = i + 2
                    jp1 = j + 1

                    q_min_AIJ = np.min([
                        q[i, j, k, l],
                        q[ip1, jp1, k, l],
                        q[ip2, jp1, k, l],
                        q[i, jp1, k, l],
                    ])
                    q_max_AIJ = np.max([
                        q[i, j, k, l],
                        q[ip1, jp1, k, l],
                        q[ip2, jp1, k, l],
                        q[i, jp1, k, l],
                    ])

                    c1 = cmask[i, j, k, l, 1]

                    Qin[i,     j,    k, l, I_min, 1] = np.where(c1 == rdtype(1.0), q_min_AIJ,  BIG)
                    Qin[ip1,   jp1,  k, l, I_min, 4] = np.where(c1 == rdtype(1.0),      BIG,  q_min_AIJ)
                    Qin[i,     j,    k, l, I_max, 1] = np.where(c1 == rdtype(1.0), q_max_AIJ, -BIG)
                    Qin[ip1,   jp1,  k, l, I_max, 4] = np.where(c1 == rdtype(1.0),    -BIG,  q_max_AIJ)
                # end if
                

                # if l==1 and k==7:
                #     with open(std.fname_log, 'a') as log_file:
                #         print("HLT: Qin 4th: I_min", file=log_file)
                #         print(Qin[0, 0, k, l, I_min, :], file=log_file)
                #         print(Qin[1, 1, k, l, I_min, :], file=log_file)
                #         print(Qin[5, 5, k, l, I_min, :], file=log_file)
                #         print("              I_max", file=log_file)
                #         print(Qin[0, 0, k, l, I_max, :], file=log_file)
                #         print(Qin[1, 1, k, l, I_max, :], file=log_file)
                #         print(Qin[5, 5, k, l, I_max, :], file=log_file)

                #---< (iii) define allowable range of q at next step, eq.(42)&(43) >---   

                isl = slice(1, iall - 1)
                jsl = slice(1, jall - 1)

                qnext_min = np.minimum.reduce([
                    q[isl, jsl, k, l],
                    Qin[isl, jsl, k, l, I_min, 0],
                    Qin[isl, jsl, k, l, I_min, 1],
                    Qin[isl, jsl, k, l, I_min, 2],
                    Qin[isl, jsl, k, l, I_min, 3],
                    Qin[isl, jsl, k, l, I_min, 4],
                    Qin[isl, jsl, k, l, I_min, 5]
                ])

                qnext_max = np.maximum.reduce([
                    q[isl, jsl, k, l],
                    Qin[isl, jsl, k, l, I_max, 0],
                    Qin[isl, jsl, k, l, I_max, 1],
                    Qin[isl, jsl, k, l, I_max, 2],
                    Qin[isl, jsl, k, l, I_max, 3],
                    Qin[isl, jsl, k, l, I_max, 4],
                    Qin[isl, jsl, k, l, I_max, 5]
                ])


                # Apply masking
                ch_masked = np.minimum(ch[isl, jsl, k, l, :], rdtype(0.0))
                Cin_sum = np.sum(ch_masked, axis=-1)
                Cout_sum = np.sum(ch[isl, jsl, k, l, :] - ch_masked, axis=-1)

                CQin_min_sum = np.sum(ch_masked * Qin[isl, jsl, k, l, I_min, :], axis=-1)
                CQin_max_sum = np.sum(ch_masked * Qin[isl, jsl, k, l, I_max, :], axis=-1)

#                 if l==1 and k==7:
#                     with open(std.fname_log, 'a') as log_file:
#                         print("QQQ1x", file=log_file)
# #                        print("MMIN", ch_masked * Qin[isl, jsl, k, l, I_min, :], file=log_file)
#                         print("MMIN", ch_masked * Qin[0, 0, 7, 1, I_min, :], file=log_file)
#                         #print("MMAX", ch_masked * Qin[isl, jsl, k, l, I_max, :], file=log_file)
#                         print("MIN", Qin[0, 0, 7, 1, I_min, :], file=log_file)
#                         print("MASK", ch_masked[0, 0, :], file=log_file)
#                         #print("MAX", Qin[0, 0, k, l, I_max, :], file=log_file)

                #zerosw = rdtype(0.5) - np.sign(rdtype(0.5), np.abs(Cout_sum) - EPS)
                zerosw = rdtype(0.5) - np.copysign(rdtype(0.5), np.abs(Cout_sum) - EPS)

                q_ = q[isl, jsl, k, l]
                d_ = d[isl, jsl, k, l]

                Qout[isl, jsl, k, l, I_min] = (
                    (q_ - CQin_max_sum - qnext_max * (rdtype(1.0) - Cin_sum - Cout_sum + d_)) /
                    (Cout_sum + zerosw) * (rdtype(1.0) - zerosw) +
                    q_ * zerosw
                )

                Qout[isl, jsl, k, l, I_max] = (
                    (q_ - CQin_min_sum - qnext_min * (rdtype(1.0) - Cin_sum - Cout_sum + d_)) /
                    (Cout_sum + zerosw) * (rdtype(1.0) - zerosw) +
                    q_ * zerosw
                )

                # if l==1 and k==7:
                #     with open(std.fname_log, 'a') as log_file:
                #         #print("QQQ1", file=log_file)
                #         print("QQQ2 Qin min", Qin[1, 1, k, l, I_min, :], file=log_file)
                #         print("QQQ2 Qin max", Qin[1, 1, k, l, I_max, :], file=log_file)
                #         print("QQQ2 Qout", Qout[1, 1, k, l, :], file=log_file)

                        # print("QQQ2 d_", d_[1, 1], file=log_file)
                        # print("QQQ2 q_", q_[1, 1], file=log_file)
                        # print("CQin_min_sum shape:", CQin_min_sum.shape, file=log_file)  # 16x16
                        # print("CQin_min_sum", CQin_min_sum[0,0], file=log_file)       # 0, 0 sometimes have a strange value
                        # print("CQin_max_sum", CQin_max_sum[0,0], file=log_file)
                        # print("qnext_min", qnext_min, file=log_file)
                        # print("qnext_max", qnext_max, file=log_file)

                        # print("Cin_sum", Cin_sum, file=log_file)
                        # print("Cout_sum", Cout_sum, file=log_file)
                        # #print("zerosw", zerosw, file=log_file) 

                # j=0 and j=jall-1 edges
                Qout[:, 0,      k, l, I_min] = q[:, 0,      k, l]
                Qout[:, 0,      k, l, I_max] = q[:, 0,      k, l]
                Qout[:, jall-1, k, l, I_min] = q[:, jall-1, k, l]
                Qout[:, jall-1, k, l, I_max] = q[:, jall-1, k, l]

                # i=0 and i=iall-1  edges (excluding corners already set)
                Qout[0,      1:jall-1, k, l, I_min] = q[0,      1:jall-1, k, l]
                Qout[0,      1:jall-1, k, l, I_max] = q[0,      1:jall-1, k, l]
                Qout[iall-1, 1:jall-1, k, l, I_min] = q[iall-1, 1:jall-1, k, l]
                Qout[iall-1, 1:jall-1, k, l, I_max] = q[iall-1, 1:jall-1, k, l]


                # if l==1 and k==7:
                #     with open(std.fname_log, 'a') as log_file:
                #         print("QQQ3 Qin min", Qin[1, 1, k, l, I_min, :], file=log_file)
                #         print("QQQ3 Qin max", Qin[1, 1, k, l, I_max, :], file=log_file)
                #         print("QQQ3 Qout", Qout[1, 1, k, l, :], file=log_file)


            # end loop k
        # end loop l

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl

            for l in range(lall_pl):
                for k in range(kall):
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ij = v
                        ijp1 = adm.ADM_gmin_pl if v + 1 > adm.ADM_gmax_pl else v + 1
                        ijm1 = adm.ADM_gmax_pl if v - 1 < adm.ADM_gmin_pl else v - 1

                        q_min_pl = min(q_pl[n, k, l], q_pl[ij, k, l], q_pl[ijm1, k, l], q_pl[ijp1, k, l])
                        q_max_pl = max(q_pl[n, k, l], q_pl[ij, k, l], q_pl[ijm1, k, l], q_pl[ijp1, k, l])

                        cm = cmask_pl[ij, k, l]

                        Qin_pl[ij, k, l, I_min, 0] = np.where(cm == rdtype(1.0), q_min_pl,  BIG)         #
                        Qin_pl[ij, k, l, I_min, 1] = np.where(cm == rdtype(1.0),     BIG,  q_min_pl)
                        Qin_pl[ij, k, l, I_max, 0] = np.where(cm == rdtype(1.0), q_max_pl, -BIG)         #
                        Qin_pl[ij, k, l, I_max, 1] = np.where(cm == rdtype(1.0),    -BIG,  q_max_pl)

                        # if k == 3 and l == 0:
                        #     print("cm", cm)
                        #     print("q_min_pl", q_min_pl)
                        #     print("q_max_pl", q_max_pl)
                        #     print("Qin_pl", v, k, l, I_min, 0)
                        #     print(Qin_pl[v, k, l, I_min, 0])  #
                            #print(Qin_pl[v, k, l, I_min, 1])
                        #    print(Qin_pl[v, k, l, I_max, 0])  #
                            #print(Qin_pl[v, k, l, I_max, 1])    

                    # Compute min/max over all v
                    qnext_min_pl = q_pl[n, k, l]
                    qnext_max_pl = q_pl[n, k, l]
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        qnext_min_pl = min(qnext_min_pl, Qin_pl[v, k, l, I_min, 0])
                        qnext_max_pl = max(qnext_max_pl, Qin_pl[v, k, l, I_max, 0])
                    # end loop v

                    # Sum contributions
                    Cin_sum_pl = rdtype(0.0)
                    Cout_sum_pl = rdtype(0.0)
                    CQin_min_sum_pl = rdtype(0.0)
                    CQin_max_sum_pl = rdtype(0.0)

                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        ch_m = cmask_pl[v, k, l] * ch_pl[v, k, l]

                        Cin_sum_pl      += ch_m
                        Cout_sum_pl     += ch_pl[v, k, l] - ch_m
                        CQin_min_sum_pl += ch_m * Qin_pl[v, k, l, I_min, 0]
                        CQin_max_sum_pl += ch_m * Qin_pl[v, k, l, I_max, 0]
                    # end loop v

                    Cout_abs = abs(Cout_sum_pl)
                    zerosw = rdtype(0.5) - np.copysign(rdtype(0.5), Cout_abs - EPS)

                    denom = Cout_sum_pl + zerosw
                    factor = rdtype(1.0) - zerosw
                    q_nkl = q_pl[n, k, l]
                    dval = d_pl[n, k, l]

                    Qout_pl[n, k, l, I_min] = ((q_nkl - CQin_max_sum_pl -
                                                qnext_max_pl * (rdtype(1.0) - Cin_sum_pl - Cout_sum_pl + dval))
                                            / denom * factor +
                                            q_nkl * zerosw)

                    Qout_pl[n, k, l, I_max] = ((q_nkl - CQin_min_sum_pl -
                                                qnext_min_pl * (rdtype(1.0) - Cin_sum_pl - Cout_sum_pl + dval))
                                            / denom * factor +
                                            q_nkl * zerosw)
                # end loop k
            # end loop l
        # endif

        comm.COMM_data_transfer( Qout[:,:,:,:,:], Qout_pl[:,:,:,:] )

        #---- apply inflow/outflow limiter

        for l in range(lall):   
            for k in range(kall):

                isl = slice(0, iall - 1)
                jsl = slice(0, jall - 1)
                isl_p1 = slice(1, iall)
                jsl_p1 = slice(1, jall)

                # Direction 1 (index 0) → copied to index 3
                q_a[isl, jsl, k, l, 0] = (
                    cmask[isl, jsl, k, l, 0] * np.minimum(
                        np.maximum(q_a[isl, jsl, k, l, 0], Qin[isl, jsl, k, l, I_min, 0]),
                        Qin[isl, jsl, k, l, I_max, 0]
                    ) + (rdtype(1.0) - cmask[isl, jsl, k, l, 0]) * np.minimum(
                        np.maximum(q_a[isl, jsl, k, l, 0], Qin[isl_p1, jsl, k, l, I_min, 3]),
                        Qin[isl_p1, jsl, k, l, I_max, 3]
                    )
                )
                q_a[isl, jsl, k, l, 0] = (
                    cmask[isl, jsl, k, l, 0] * np.maximum(
                        np.minimum(q_a[isl, jsl, k, l, 0], Qout[isl_p1, jsl, k, l, I_max]),
                        Qout[isl_p1, jsl, k, l, I_min]
                    ) + (rdtype(1.0) - cmask[isl, jsl, k, l, 0]) * np.maximum(
                        np.minimum(q_a[isl, jsl, k, l, 0], Qout[isl, jsl, k, l, I_max]),
                        Qout[isl, jsl, k, l, I_min]
                    )
                )
                q_a[isl_p1, jsl, k, l, 3] = q_a[isl, jsl, k, l, 0]

                # Direction 2 (index 1) → copied to index 4
                q_a[isl, jsl, k, l, 1] = (
                    cmask[isl, jsl, k, l, 1] * np.minimum(
                        np.maximum(q_a[isl, jsl, k, l, 1], Qin[isl, jsl, k, l, I_min, 1]),
                        Qin[isl, jsl, k, l, I_max, 1]
                    ) + (rdtype(1.0) - cmask[isl, jsl, k, l, 1]) * np.minimum(
                        np.maximum(q_a[isl, jsl, k, l, 1], Qin[isl_p1, jsl_p1, k, l, I_min, 4]),
                        Qin[isl_p1, jsl_p1, k, l, I_max, 4]
                    )
                )
                q_a[isl, jsl, k, l, 1] = (
                    cmask[isl, jsl, k, l, 1] * np.maximum(
                        np.minimum(q_a[isl, jsl, k, l, 1], Qout[isl_p1, jsl_p1, k, l, I_max]),
                        Qout[isl_p1, jsl_p1, k, l, I_min]
                    ) + (rdtype(1.0) - cmask[isl, jsl, k, l, 1]) * np.maximum(
                        np.minimum(q_a[isl, jsl, k, l, 1], Qout[isl, jsl, k, l, I_max]),
                        Qout[isl, jsl, k, l, I_min]
                    )
                )
                q_a[isl_p1, jsl_p1, k, l, 4] = q_a[isl, jsl, k, l, 1]

                # Direction 3 (index 2) → copied to index 5
                q_a[isl, jsl, k, l, 2] = (
                    cmask[isl, jsl, k, l, 2] * np.minimum(
                        np.maximum(q_a[isl, jsl, k, l, 2], Qin[isl, jsl, k, l, I_min, 2]),
                        Qin[isl, jsl, k, l, I_max, 2]
                    ) + (rdtype(1.0) - cmask[isl, jsl, k, l, 2]) * np.minimum(
                        np.maximum(q_a[isl, jsl, k, l, 2], Qin[isl, jsl_p1, k, l, I_min, 5]),
                        Qin[isl, jsl_p1, k, l, I_max, 5]
                    )
                )
                q_a[isl, jsl, k, l, 2] = (
                    cmask[isl, jsl, k, l, 2] * np.maximum(
                        np.minimum(q_a[isl, jsl, k, l, 2], Qout[isl, jsl_p1, k, l, I_max]),
                        Qout[isl, jsl_p1, k, l, I_min]
                    ) + (rdtype(1.0) - cmask[isl, jsl, k, l, 2]) * np.maximum(
                        np.minimum(q_a[isl, jsl, k, l, 2], Qout[isl, jsl, k, l, I_max]),
                        Qout[isl, jsl, k, l, I_min]
                    )
                )
                q_a[isl, jsl_p1, k, l, 5] = q_a[isl, jsl, k, l, 2]

                # isl = slice(0, iall - 1)
                # jsl = slice(0, jall - 1)

                # #  (indices 0 and 3)
                # cm = cmask[isl, jsl, k, l, 0]
                # qa = q_a[isl, jsl, k, l, 0]
                # qmin_ai = Qin[isl, jsl, k, l, I_min, 0]
                # qmax_ai = Qin[isl, jsl, k, l, I_max, 0]
                # qmin_ai_p = Qin[isl.start + 1, jsl, k, l, I_min, 3]
                # qmax_ai_p = Qin[isl.start + 1, jsl, k, l, I_max, 3]

                # q_a[isl, jsl, k, l, 0] = cm * np.minimum(np.maximum(qa, qmin_ai), qmax_ai) + (rdtype(1.0) - cm) * np.minimum(np.maximum(qa, qmin_ai_p), qmax_ai_p)

                # qmin_out = Qout[isl.start + 1, jsl, k, l, I_min]
                # qmax_out = Qout[isl.start + 1, jsl, k, l, I_max]
                # qmin_out_p = Qout[isl, jsl, k, l, I_min]
                # qmax_out_p = Qout[isl, jsl, k, l, I_max]

                # q_a[isl, jsl, k, l, 0] = cm * np.maximum(np.minimum(q_a[isl, jsl, k, l, 0], qmax_out), qmin_out) + (rdtype(1.0) - cm) * np.maximum(np.minimum(q_a[isl, jsl, k, l, 0], qmax_out_p), qmin_out_p)
                # q_a[isl.start + 1, jsl, k, l, 3] = q_a[isl, jsl, k, l, 0]

                # #  (indices 1 and 4)
                # cm = cmask[isl, jsl, k, l, 1]
                # qa = q_a[isl, jsl, k, l, 1]
                # qmin = Qin[isl, jsl, k, l, I_min, 1]
                # qmax = Qin[isl, jsl, k, l, I_max, 1]
                # qmin_p = Qin[isl.start + 1, jsl.start + 1, k, l, I_min, 4]
                # qmax_p = Qin[isl.start + 1, jsl.start + 1, k, l, I_max, 4]

                # q_a[isl, jsl, k, l, 1] = cm * np.minimum(np.maximum(qa, qmin), qmax) + (rdtype(1.0) - cm) * np.minimum(np.maximum(qa, qmin_p), qmax_p)

                # qmin_out = Qout[isl.start + 1, jsl.start + 1, k, l, I_min]
                # qmax_out = Qout[isl.start + 1, jsl.start + 1, k, l, I_max]
                # qmin_out_p = Qout[isl, jsl, k, l, I_min]
                # qmax_out_p = Qout[isl, jsl, k, l, I_max]

                # q_a[isl, jsl, k, l, 1] = cm * np.maximum(np.minimum(q_a[isl, jsl, k, l, 1], qmax_out), qmin_out) + (rdtype(1.0) - cm) * np.maximum(np.minimum(q_a[isl, jsl, k, l, 1], qmax_out_p), qmin_out_p)
                # q_a[isl.start + 1, jsl.start + 1, k, l, 4] = q_a[isl, jsl, k, l, 1]

                # #  (indices 2 and 5)
                # cm = cmask[isl, jsl, k, l, 2]
                # qa = q_a[isl, jsl, k, l, 2]
                # qmin = Qin[isl, jsl, k, l, I_min, 2]
                # qmax = Qin[isl, jsl, k, l, I_max, 2]
                # qmin_p = Qin[isl, jsl.start + 1, k, l, I_min, 5]
                # qmax_p = Qin[isl, jsl.start + 1, k, l, I_max, 5]

                # q_a[isl, jsl, k, l, 2] = cm * np.minimum(np.maximum(qa, qmin), qmax) + (rdtype(1.0) - cm) * np.minimum(np.maximum(qa, qmin_p), qmax_p)

                # qmin_out = Qout[isl, jsl.start + 1, k, l, I_min]
                # qmax_out = Qout[isl, jsl.start + 1, k, l, I_max]
                # qmin_out_p = Qout[isl, jsl, k, l, I_min]
                # qmax_out_p = Qout[isl, jsl, k, l, I_max]

                # q_a[isl, jsl, k, l, 2] = cm * np.maximum(np.minimum(q_a[isl, jsl, k, l, 2], qmax_out), qmin_out) + (rdtype(1.0) - cm) * np.maximum(np.minimum(q_a[isl, jsl, k, l, 2], qmax_out_p), qmin_out_p)
                # q_a[isl, jsl.start + 1, k, l, 5] = q_a[isl, jsl, k, l, 2]

            # end loop k
        # end loop l

        if adm.ADM_have_pl:
            n = adm.ADM_gslf_pl
            for l in range(lall_pl):
                for k in range(kall):
                    for v in range(adm.ADM_gmin_pl, adm.ADM_gmax_pl + 1):
                        cm = cmask_pl[v, k, l]

                        # First clamping between min/max inputs
                        q0 = np.minimum(np.maximum(q_a_pl[v, k, l], Qin_pl[v, k, l, I_min, 0]),
                                        Qin_pl[v, k, l, I_max, 0])
                        q1 = np.minimum(np.maximum(q_a_pl[v, k, l], Qin_pl[v, k, l, I_min, 1]),
                                        Qin_pl[v, k, l, I_max, 1])
                        q_a_pl[v, k, l] = cm * q0 + (rdtype(1.0) - cm) * q1

                        #if k == 3 and l == 0:
                        #    print(Qin_pl[v, k, l, I_min, 0])
                        #    print(f"A: q_a_pl[{v}, {k}, {l}] = ", q_a_pl[v, k, l])
                        #    print("q0", q0)
                        #   print("cm", cm)

                        # Then further clamping with output bounds
                        q2 = np.maximum(np.minimum(q_a_pl[v, k, l], Qout_pl[v, k, l, I_max]),
                                        Qout_pl[v, k, l, I_min])
                        q3 = np.maximum(np.minimum(q_a_pl[v, k, l], Qout_pl[n, k, l, I_max]),
                                        Qout_pl[n, k, l, I_min])
                        q_a_pl[v, k, l] = cm * q2 + (rdtype(1.0) - cm) * q3

                        #print(f"B: q_a_pl[{v}, {k}, {l}] = ", q_a_pl[v, k, l])
                    # end loop v
                # end loop k
            # end loop l
        # end if

        return
    