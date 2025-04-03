import numpy as np
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf

class Vi:
    
    _instance = None
    
    def __init__(self):
        pass


    def vi_setup(self, rdtype):

        self.Mc    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall),    dtype=rdtype)
        self.Mc_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)
        self.Mu    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall),    dtype=rdtype)
        self.Mu_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)
        self.Ml    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall),    dtype=rdtype)
        self.Ml_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)

        return

    def vi_small_step(self,
            PROG,       PROG_pl,       
            vx,         vx_pl,         
            vy,         vy_pl,         
            vz,         vz_pl,         
            eth,        eth_pl,        
            rhog_prim,  rhog_prim_pl,  
            preg_prim,  preg_prim_pl,  
            g_TEND0,    g_TEND0_pl,    
            PROG_split, PROG_split_pl, 
            PROG_mean,  PROG_mean_pl,  
            num_of_itr,                
            dt,
            dt_pl,       
            cnst, comm, grd, oprt, vmtr, tim, rcnf, bndc, numf, src, rdtype,                  
    ):
        
        prf.PROF_rapstart('____vi_path0',2)   

        gall_1d = adm.ADM_gall_1d
        gall_pl = gall_pl
        kall = adm.ADM_kdall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        lall_pl = adm.ADM_lall_pl

        grhogetot0    = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        grhogetot0_pl = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        rhog_h        = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        eth_h         = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        rhog_h_pl     = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        eth_h_pl      = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        drhog         = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        drhog_pl      = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        dpgrad        = np.empty((gall_1d, gall_1d, kall, lall,    3,), dtype=rdtype)  # additional dimension for XDIR YDIR ZDIR
        dpgrad_pl     = np.empty((gall_pl,          kall, lall_pl, 3,), dtype=rdtype)  # additional dimension for XDIR YDIR ZDIR
        dpgradw       = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        dpgradw_pl    = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        dbuoiw        = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        dbuoiw_pl     = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        drhoge        = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        drhoge_pl     = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        gz_tilde      = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        gz_tilde_pl   = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        drhoge_pw     = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        drhoge_pw_pl  = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        drhoge_pwh    = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        drhoge_pwh_pl = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        g_TEND        = np.empty((gall_1d, gall_1d, kall, lall,    6,), dtype=rdtype)  # additional dimension for I_RHOG to I_RHOGE
        g_TEND_pl     = np.empty((gall_pl,          kall, lall_pl, 6,), dtype=rdtype)  # additional dimension for I_RHOG to I_RHOGE
      
        ddivdvx       = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        ddivdvx_pl    = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        ddivdvx_2d    = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        ddivdvx_2d_pl = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        ddivdvy       = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        ddivdvy_pl    = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        ddivdvy_2d    = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        ddivdvy_2d_pl = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        ddivdvz       = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        ddivdvz_pl    = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        ddivdvz_2d    = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        ddivdvz_2d_pl = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        ddivdw        = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        ddivdw_pl     = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)

        preg_prim_split     = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        preg_prim_split_pl  = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)

        drhogw        = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        drhogw_pl     = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)

        drhogw        = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        drhogw_pl     = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)

        diff_vh       = np.empty((gall_1d, gall_1d, kall, lall,    3,), dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ
        diff_vh_pl    = np.empty((gall_pl,          kall, lall_pl, 3,), dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ
        diff_we       = np.empty((gall_1d, gall_1d, kall, lall,    3,), dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ
        diff_we_pl    = np.empty((gall_pl,          kall, lall_pl, 3,), dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ

        XDIR = grd.GRD_XDIR
        YDIR = grd.GRD_YDIR
        ZDIR = grd.GRD_ZDIR     

        grav  = cnst.CONST_GRAV
        RovCV = cnst.CONST_Rdry / cnst.CONST_CVdry
        alpha = rdtype(rcnf.NON_HYDRO_ALPHA)

        I_RHOG  = rcnf.I_RHOG
        I_RHOGVX = rcnf.I_RHOGVX
        I_RHOGVY = rcnf.I_RHOGVY
        I_RHOGVZ = rcnf.I_RHOGVZ
        I_RHOGW = rcnf.I_RHOGW
        I_RHOGE = rcnf.I_RHOGE

        for l in range(lall):
            for k in range(kall):
                grhogetot0[:, :, k, l] = g_TEND0[:, :, k, l, I_RHOGE]
            #end k loop
        #end l loop
        grhogetot0_pl[:, :, :] = g_TEND0_pl[:, :, :, I_RHOGE]


        # full level -> half level

        for l in range(lall):
            for k in range(kmin, kmax + 2):  # +2 to include kmax+1
                rhog_h[:, :, k, l] = (
                    vmtr.VMTR_C2Wfact[:, :, k, 0, l] * PROG[:, :, k,   l, I_RHOG] +
                    vmtr.VMTR_C2Wfact[:, :, k, 1, l] * PROG[:, :, k-1, l, I_RHOG]
                )
                eth_h[:, :, k, l] = (
                    grd.GRD_afact[k] * eth[:, :, k,   l] +
                    grd.GRD_bfact[k] * eth[:, :, k-1, l]
                )
            #end k loop

            rhog_h[:, :, kmin-1, l] = rhog_h[:, :, kmin, l]
            eth_h[:, :, kmin-1, l]  = eth_h[:, :, kmin, l]
        #end l loop

        if adm.ADM_have_pl:
            for l in range(adm.ADM_lall_pl):
                # Vectorized computation for kmin to kmax+1
                rhog_h_pl[:, kmin:kmax+2, l] = (
                    vmtr.VMTR_C2Wfact_pl[:, kmin:kmax+2, 0, l] * PROG_pl[:, kmin:kmax+2, l, I_RHOG] +
                    vmtr.VMTR_C2Wfact_pl[:, kmin:kmax+2, 1, l] * PROG_pl[:, kmin-1:kmax+1, l, I_RHOG]
                )

                eth_h_pl[:, kmin:kmax+2, l] = (
                    grd.GRD_afact[kmin:kmax+2][None, :] * eth_pl[:, kmin:kmax+2, l] +
                    grd.GRD_bfact[kmin:kmax+2][None, :] * eth_pl[:, kmin-1:kmax+1, l]
                )

                # Fill ghost level
                rhog_h_pl[:, kmin-1, l] = rhog_h_pl[:, kmin, l]
                eth_h_pl[:, kmin-1, l]  = eth_h_pl[:, kmin, l]
            #end l loop
        #endif

        #---< Calculation of source term for rhog >

        src.src_flux_convergence(
                PROG [:,:,:,:,I_RHOGVX], PROG_pl [:,:,:,I_RHOGVX],
                PROG [:,:,:,:,I_RHOGVY], PROG_pl [:,:,:,I_RHOGVY],
                PROG [:,:,:,:,I_RHOGVZ], PROG_pl [:,:,:,I_RHOGVZ],
                PROG [:,:,:,:,I_RHOGW],  PROG_pl [:,:,:,I_RHOGW],
                drhog[:,:,:,:],          drhog_pl[:,:,:],   
                src.I_SRC_default,   
                grd, oprt, vmtr, rdtype, 
        )


        #---< Calculation of source term for Vh(vx,vy,vz) and W >

        # divergence damping
        # call numfilter_divdamp( PROG   (:,:,:,I_RHOGVX), PROG_pl   (:,:,:,I_RHOGVX), & ! [IN]
        #                 PROG   (:,:,:,I_RHOGVY), PROG_pl   (:,:,:,I_RHOGVY), & ! [IN]
        #                 PROG   (:,:,:,I_RHOGVZ), PROG_pl   (:,:,:,I_RHOGVZ), & ! [IN]
        #                 PROG   (:,:,:,I_RHOGW),  PROG_pl   (:,:,:,I_RHOGW),  & ! [IN]
        #                 ddivdvx(:,:,:),          ddivdvx_pl(:,:,:),          & ! [OUT]
        #                 ddivdvy(:,:,:),          ddivdvy_pl(:,:,:),          & ! [OUT]
        #                 ddivdvz(:,:,:),          ddivdvz_pl(:,:,:),          & ! [OUT]
        #                 ddivdw (:,:,:),          ddivdw_pl (:,:,:)           ) ! [OUT]

        # call numfilter_divdamp_2d( PROG      (:,:,:,I_RHOGVX), PROG_pl      (:,:,:,I_RHOGVX), & ! [IN]
        #                         PROG      (:,:,:,I_RHOGVY), PROG_pl      (:,:,:,I_RHOGVY), & ! [IN]
        #                         PROG      (:,:,:,I_RHOGVZ), PROG_pl      (:,:,:,I_RHOGVZ), & ! [IN]
        #                         ddivdvx_2d(:,:,:),          ddivdvx_2d_pl(:,:,:),          & ! [OUT]
        #                         ddivdvy_2d(:,:,:),          ddivdvy_2d_pl(:,:,:),          & ! [OUT]
        #                         ddivdvz_2d(:,:,:),          ddivdvz_2d_pl(:,:,:)           ) ! [OUT]


        # pressure force
        src.src_pres_gradient(
            preg_prim[:,:,:,:],   preg_prim_pl[:,:,:],   # [IN]
            dpgrad   [:,:,:,:,:], dpgrad_pl   [:,:,:,:], # [OUT]
            dpgradw  [:,:,:,:],   dpgradw_pl  [:,:,:],   # [OUT]
            src.I_SRC_default,                           # [IN]
            grd, oprt, vmtr, rdtype,   
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
            src.I_SRC_default,                                 # [IN]
            grd, oprt, vmtr, rdtype,
        )

        # pressure work

        for l in range(lall):
            # First part: compute gz_tilde and drhoge_pwh
            for k in range(kall):
                gz_tilde[:, :, k, l] = grav - (dpgradw[:, :, k, l] - dbuoiw[:, :, k, l]) / rhog_h[:, :, k, l]
                drhoge_pwh[:, :, k, l] = -gz_tilde[:, :, k, l] * PROG[:, :, k, l, I_RHOGW]
            # end k loop

            # Second part: compute drhoge_pw
            for k in range(kmin, kmax + 1):
                drhoge_pw[:, :, k, l] = (
                    vx[:, :, k, l] * dpgrad[:, :, k, l, XDIR] +
                    vy[:, :, k, l] * dpgrad[:, :, k, l, YDIR] +
                    vz[:, :, k, l] * dpgrad[:, :, k, l, ZDIR] +
                    vmtr.VMTR_W2Cfact[:, :, k, 0, l] * drhoge_pwh[:, :, k + 1, l] +
                    vmtr.VMTR_W2Cfact[:, :, k, 1, l] * drhoge_pwh[:, :, k, l]
                )
            # end k loop

            drhoge_pw[:, :, kmin - 1, l] = 0.0
            drhoge_pw[:, :, kmax + 1, l] = 0.0
        # end l loop

        if adm.ADM_have_pl:
            for l in range(adm.ADM_lall_pl):
                # --- Vectorized gz_tilde_pl and drhoge_pwh_pl
                gz_tilde_pl[:, :, l] = grav - (dpgradw_pl[:, :, l] - dbuoiw_pl[:, :, l]) / rhog_h_pl[:, :, l]
                drhoge_pwh_pl[:, :, l] = -gz_tilde_pl[:, :, l] * PROG_pl[:, :, l, I_RHOGW]

                # --- Vectorized drhoge_pw_pl over kmin to kmax
                drhoge_pw_pl[:, kmin:kmax+1, l] = (
                    vx_pl[:, kmin:kmax+1, l] * dpgrad_pl[:, kmin:kmax+1, XDIR] +
                    vy_pl[:, kmin:kmax+1, l] * dpgrad_pl[:, kmin:kmax+1, YDIR] +
                    vz_pl[:, kmin:kmax+1, l] * dpgrad_pl[:, kmin:kmax+1, ZDIR] +
                    vmtr.VMTR_W2Cfact_pl[:, kmin:kmax+1, 0, l] * drhoge_pwh_pl[:, kmin+1:kmax+2, l] +
                    vmtr.VMTR_W2Cfact_pl[:, kmin:kmax+1, 1, l] * drhoge_pwh_pl[:, kmin:kmax+1,   l]
                )

                # --- Ghost layers at boundaries
                drhoge_pw_pl[:, kmin-1, l] = 0.0
                drhoge_pw_pl[:, kmax+1, l] = 0.0
            # end l loop
        #endif


        #---< sum of tendencies ( large step + pres-grad + div-damp + div-damp_2d + buoyancy ) >

        for l in range(lall):
            for k in range(kall):
                g_TEND[:, :, k, l, I_RHOG]   = g_TEND0[:, :, k, l, I_RHOG] + drhog[:, :, k, l]

                g_TEND[:, :, k, l, I_RHOGVX] = (
                    g_TEND0[:, :, k, l, I_RHOGVX]
                    - dpgrad[:, :, k, l, XDIR]
                    + ddivdvx[:, :, k, l]
                    + ddivdvx_2d[:, :, k, l]
                )

                g_TEND[:, :, k, l, I_RHOGVY] = (
                    g_TEND0[:, :, k, l, I_RHOGVY]
                    - dpgrad[:, :, k, l, YDIR]
                    + ddivdvy[:, :, k, l]
                    + ddivdvy_2d[:, :, k, l]
                )

                g_TEND[:, :, k, l, I_RHOGVZ] = (
                    g_TEND0[:, :, k, l, I_RHOGVZ]
                    - dpgrad[:, :, k, l, ZDIR]
                    + ddivdvz[:, :, k, l]
                    + ddivdvz_2d[:, :, k, l]
                )

                g_TEND[:, :, k, l, I_RHOGW] = (
                    g_TEND0[:, :, k, l, I_RHOGW]
                    + ddivdw[:, :, k, l] * alpha
                    - dpgradw[:, :, k, l]
                    + dbuoiw[:, :, k, l]
                )

                g_TEND[:, :, k, l, I_RHOGE] = (
                    g_TEND0[:, :, k, l, I_RHOGE]
                    + drhoge[:, :, k, l]
                    + drhoge_pw[:, :, k, l]
                )
            #end k loop
        #end l loop 

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


        # initialization of mean mass flux

        rweight_itr = 1.0 / rdtype(num_of_itr)
                                # 0 :  5     + 1  # includes I_RHOG (0) to I_RHOGW (5)
        PROG_mean[:, :, :, :, I_RHOG:I_RHOGW + 1] = PROG[:, :, :, :, I_RHOG:I_RHOGW + 1]
        PROG_mean_pl[:, :, :, I_RHOG:I_RHOGW + 1] = PROG_pl[:, :, :, I_RHOG:I_RHOGW + 1]


        # update working matrix for vertical implicit solver
        # call vi_rhow_update_matrix( eth_h   (:,:,:), eth_h_pl   (:,:,:), & ! [IN]
        #                             gz_tilde(:,:,:), gz_tilde_pl(:,:,:), & ! [IN]
        #                             dt                                   ) ! [IN]

        prf.PROF_rapend  ('____vi_path0',2)
        #---------------------------------------------------------------------------
        #
        #> Start small step iteration
        #
        #---------------------------------------------------------------------------
        for ns in range(num_of_itr):

            prf.PROF_rapstart('____vi_path1',2)

            #---< calculation of preg_prim(*) from rhog(*) & rhoge(*) >

            for l in range(lall):
                for k in range(kall):
                    preg_prim_split[:, :, k, l] = PROG_split[:, :, k, l, I_RHOGE] * RovCV
                #end k loop

                preg_prim_split[:, :, kmin - 1, l] = preg_prim_split[:, :, kmin, l]
                preg_prim_split[:, :, kmax + 1, l] = preg_prim_split[:, :, kmax, l]

                PROG_split[:, :, kmin - 1, l, I_RHOGE] = PROG_split[:, :, kmin, l, I_RHOGE]
                PROG_split[:, :, kmax + 1, l, I_RHOGE] = PROG_split[:, :, kmax, l, I_RHOGE]
            #end l loop

            if adm.ADM_have_pl:
                for l in range(adm.ADM_lall_pl):
                    preg_prim_split_pl[:, :, l] = PROG_split_pl[:, :, l, I_RHOGE] * RovCV

                    # Ghost layers copy
                    preg_prim_split_pl[:, kmin - 1, l] = preg_prim_split_pl[:, kmin, l]
                    preg_prim_split_pl[:, kmax + 1, l] = preg_prim_split_pl[:, kmax, l]

                    PROG_split_pl[:, kmin - 1, l, I_RHOGE] = PROG_split_pl[:, kmin, l, I_RHOGE]
                    PROG_split_pl[:, kmax + 1, l, I_RHOGE] = PROG_split_pl[:, kmax, l, I_RHOGE]
                #end l loop
            #endif


            if tim.TIME_split:
                #---< Calculation of source term for Vh(vx,vy,vz) and W (split) >

                # divergence damping

                # call numfilter_divdamp( PROG_split(:,:,:,I_RHOGVX), PROG_split_pl(:,:,:,I_RHOGVX), & ! [IN]
                #                         PROG_split(:,:,:,I_RHOGVY), PROG_split_pl(:,:,:,I_RHOGVY), & ! [IN]
                #                         PROG_split(:,:,:,I_RHOGVZ), PROG_split_pl(:,:,:,I_RHOGVZ), & ! [IN]
                #                         PROG_split(:,:,:,I_RHOGW),  PROG_split_pl(:,:,:,I_RHOGW),  & ! [IN]
                #                         ddivdvx   (:,:,:),          ddivdvx_pl   (:,:,:),          & ! [OUT]
                #                         ddivdvy   (:,:,:),          ddivdvy_pl   (:,:,:),          & ! [OUT]
                #                         ddivdvz   (:,:,:),          ddivdvz_pl   (:,:,:),          & ! [OUT]
                #                         ddivdw    (:,:,:),          ddivdw_pl    (:,:,:)           ) ! [OUT]

                # 2d divergence damping
                # call numfilter_divdamp_2d( PROG_split(:,:,:,I_RHOGVX), PROG_split_pl(:,:,:,I_RHOGVX), & ! [IN]
                #                             PROG_split(:,:,:,I_RHOGVY), PROG_split_pl(:,:,:,I_RHOGVY), & ! [IN]
                #                             PROG_split(:,:,:,I_RHOGVZ), PROG_split_pl(:,:,:,I_RHOGVZ), & ! [IN]
                #                             ddivdvx_2d(:,:,:),          ddivdvx_2d_pl(:,:,:),          & ! [OUT]
                #                             ddivdvy_2d(:,:,:),          ddivdvy_2d_pl(:,:,:),          & ! [OUT]
                #                             ddivdvz_2d(:,:,:),          ddivdvz_2d_pl(:,:,:)           ) ! [OUT]

                # pressure force
                # dpgradw=0.0_RP becaude of f_type='HORIZONTAL'.
                src.src_pres_gradient( 
                    preg_prim_split[:,:,:,:],   preg_prim_split_pl[:,:,:],   # [IN]
                    dpgrad         [:,:,:,:,:], dpgrad_pl         [:,:,:,:], # [OUT]
                    dpgradw        [:,:,:,:],   dpgradw_pl        [:,:,:],   # [OUT] not used
                    src.I_SRC_horizontal,                                     # [IN]
                    grd, oprt, vmtr, rdtype,
                )

                # buoyancy force
                # not calculated, because this term is implicit.

                #---< sum of tendencies ( large step + split{ pres-grad + div-damp + div-damp_2d } ) >

                for l in range(lall):
                    for k in range(kall):
                        drhogvx = (
                            g_TEND[:, :, k, l, I_RHOGVX]
                            - dpgrad[:, :, k, l, XDIR]
                            + ddivdvx[:, :, k, l]
                            + ddivdvx_2d[:, :, k, l]
                        )
                        drhogvy = (
                            g_TEND[:, :, k, l, I_RHOGVY]
                            - dpgrad[:, :, k, l, YDIR]
                            + ddivdvy[:, :, k, l]
                            + ddivdvy_2d[:, :, k, l]
                        )
                        drhogvz = (
                            g_TEND[:, :, k, l, I_RHOGVZ]
                            - dpgrad[:, :, k, l, ZDIR]
                            + ddivdvz[:, :, k, l]
                            + ddivdvz_2d[:, :, k, l]
                        )
                        drhogw[:, :, k, l] = g_TEND[:, :, k, l, I_RHOGW] + ddivdw[:, :, k, l] * alpha

                        diff_vh[:, :, k, l, 0] = PROG_split[:, :, k, l, I_RHOGVX] + drhogvx * dt
                        diff_vh[:, :, k, l, 1] = PROG_split[:, :, k, l, I_RHOGVY] + drhogvy * dt
                        diff_vh[:, :, k, l, 2] = PROG_split[:, :, k, l, I_RHOGVZ] + drhogvz * dt
                    #end k loop
                #end l loop

                if adm.ADM_have_pl:
                    for l in range(adm.ADM_lall_pl):
                        # Vectorized over g and k
                        drhogvx = (
                            g_TEND_pl[:, :, l, I_RHOGVX]
                            - dpgrad_pl[:, :, l, XDIR]
                            + ddivdvx_pl[:, :, l]
                            + ddivdvx_2d_pl[:, :, l]
                        )
                        drhogvy = (
                            g_TEND_pl[:, :, l, I_RHOGVY]
                            - dpgrad_pl[:, :, l, YDIR]
                            + ddivdvy_pl[:, :, l]
                            + ddivdvy_2d_pl[:, :, l]
                        )
                        drhogvz = (
                            g_TEND_pl[:, :, l, I_RHOGVZ]
                            - dpgrad_pl[:, :, l, ZDIR]
                            + ddivdvz_pl[:, :, l]
                            + ddivdvz_2d_pl[:, :, l]
                        )

                        drhogw_pl[:, :, l] = g_TEND_pl[:, :, l, I_RHOGW] + ddivdw_pl[:, :, l] * alpha

                        diff_vh_pl[:, :, l, 0] = PROG_split_pl[:, :, l, I_RHOGVX] + drhogvx * dt
                        diff_vh_pl[:, :, l, 1] = PROG_split_pl[:, :, l, I_RHOGVY] + drhogvy * dt
                        diff_vh_pl[:, :, l, 2] = PROG_split_pl[:, :, l, I_RHOGVZ] + drhogvz * dt
                    #end l loop
                #endif

            else: # NO-SPLITING
            
                #---< sum of tendencies ( large step ) >

                for l in range(lall):
                    for k in range(kall):
                        drhogvx = g_TEND[:, :, k, l, I_RHOGVX]
                        drhogvy = g_TEND[:, :, k, l, I_RHOGVY]
                        drhogvz = g_TEND[:, :, k, l, I_RHOGVZ]
                        drhogw[:, :, k, l] = g_TEND[:, :, k, l, I_RHOGW]

                        diff_vh[:, :, k, l, 0] = PROG_split[:, :, k, l, I_RHOGVX] + drhogvx * dt
                        diff_vh[:, :, k, l, 1] = PROG_split[:, :, k, l, I_RHOGVY] + drhogvy * dt
                        diff_vh[:, :, k, l, 2] = PROG_split[:, :, k, l, I_RHOGVZ] + drhogvz * dt
                    #end k loop
                #end l loop

                if adm.ADM_have_pl:
                    for l in range(adm.ADM_lall_pl):
                        # Vectorized across g and k
                        drhogvx = g_TEND_pl[:, :, l, I_RHOGVX]
                        drhogvy = g_TEND_pl[:, :, l, I_RHOGVY]
                        drhogvz = g_TEND_pl[:, :, l, I_RHOGVZ]
                        drhogw_pl[:, :, l] = g_TEND_pl[:, :, l, I_RHOGW]

                        diff_vh_pl[:, :, l, 0] = PROG_split_pl[:, :, l, I_RHOGVX] + drhogvx * dt
                        diff_vh_pl[:, :, l, 1] = PROG_split_pl[:, :, l, I_RHOGVY] + drhogvy * dt
                        diff_vh_pl[:, :, l, 2] = PROG_split_pl[:, :, l, I_RHOGVZ] + drhogvz * dt
                    #end l loop
                #endif

            #endif    Split/Non-split

            # treatment for boundary condition
            bndc.BNDCND_rhovxvyvz( 
                PROG   [:,:,:,:,I_RHOG], # [IN]
                diff_vh[:,:,:,:,1],      # [INOUT]
                diff_vh[:,:,:,:,2],      # [INOUT]
                diff_vh[:,:,:,:,3],      # [INOUT]
            )

            if adm.ADM_have_pl:
                bndc.BNDCND_rhovxvyvz(
                    PROG_pl   [:,np.newaxis,:,:,I_RHOG], # [IN]
                    diff_vh_pl[:,np.newaxis,:,:,1],      # [INOUT]
                    diff_vh_pl[:,np.newaxis,:,:,2],      # [INOUT]
                    diff_vh_pl[:,np.newaxis,:,:,3],       # [INOUT]
                )
                # check whether or not squeeze is needed to remove the dummy axis 
            #endif


            comm.COMM_data_transfer( diff_vh, diff_vh_pl )

            prf.PROF_rapend  ('____vi_path1',2)
            prf.PROF_rapstart('____vi_path2',2)

            #---< vertical implicit scheme >

            # call vi_main( diff_we        (:,:,:,1),        diff_we_pl        (:,:,:,1),        & ! [OUT]
            #                 diff_we        (:,:,:,2),        diff_we_pl        (:,:,:,2),        & ! [OUT]
            #                 diff_we        (:,:,:,3),        diff_we_pl        (:,:,:,3),        & ! [OUT]
            #                 diff_vh        (:,:,:,1),        diff_vh_pl        (:,:,:,1),        & ! [IN]
            #                 diff_vh        (:,:,:,2),        diff_vh_pl        (:,:,:,2),        & ! [IN]
            #                 diff_vh        (:,:,:,3),        diff_vh_pl        (:,:,:,3),        & ! [IN]
            #                 PROG_split     (:,:,:,I_RHOG),   PROG_split_pl     (:,:,:,I_RHOG),   & ! [IN]
            #                 PROG_split     (:,:,:,I_RHOGVX), PROG_split_pl     (:,:,:,I_RHOGVX), & ! [IN]
            #                 PROG_split     (:,:,:,I_RHOGVY), PROG_split_pl     (:,:,:,I_RHOGVY), & ! [IN]
            #                 PROG_split     (:,:,:,I_RHOGVZ), PROG_split_pl     (:,:,:,I_RHOGVZ), & ! [IN]
            #                 PROG_split     (:,:,:,I_RHOGW),  PROG_split_pl     (:,:,:,I_RHOGW),  & ! [IN]
            #                 PROG_split     (:,:,:,I_RHOGE),  PROG_split_pl     (:,:,:,I_RHOGE),  & ! [IN]
            #                 preg_prim_split(:,:,:),          preg_prim_split_pl(:,:,:),          & ! [IN]
            #                 PROG           (:,:,:,I_RHOG),   PROG_pl           (:,:,:,I_RHOG),   & ! [IN]
            #                 PROG           (:,:,:,I_RHOGVX), PROG_pl           (:,:,:,I_RHOGVX), & ! [IN]
            #                 PROG           (:,:,:,I_RHOGVY), PROG_pl           (:,:,:,I_RHOGVY), & ! [IN]
            #                 PROG           (:,:,:,I_RHOGVZ), PROG_pl           (:,:,:,I_RHOGVZ), & ! [IN]
            #                 PROG           (:,:,:,I_RHOGW),  PROG_pl           (:,:,:,I_RHOGW),  & ! [IN]
            #                 eth            (:,:,:),          eth_pl            (:,:,:),          & ! [IN]
            #                 g_TEND         (:,:,:,I_RHOG),   g_TEND_pl         (:,:,:,I_RHOG),   & ! [IN]
            #                 drhogw         (:,:,:),          drhogw_pl         (:,:,:),          & ! [IN]
            #                 g_TEND         (:,:,:,I_RHOGE),  g_TEND_pl         (:,:,:,I_RHOGE),  & ! [IN]
            #                 grhogetot0     (:,:,:),          grhogetot0_pl     (:,:,:),          & ! [IN]
            #                 dt                                     ) ! [IN]

            # treatment for boundary condition
            comm.COMM_data_transfer( diff_we, diff_we_pl )

            # update split value and mean mass flux

            for l in range(lall):
                for k in range(kall):
                    PROG_split[:, :, k, l, I_RHOGVX] = diff_vh[:, :, k, l, 0]
                    PROG_split[:, :, k, l, I_RHOGVY] = diff_vh[:, :, k, l, 1]
                    PROG_split[:, :, k, l, I_RHOGVZ] = diff_vh[:, :, k, l, 2]
                    PROG_split[:, :, k, l, I_RHOG]   = diff_we[:, :, k, l, 0]
                    PROG_split[:, :, k, l, I_RHOGW]  = diff_we[:, :, k, l, 1]
                    PROG_split[:, :, k, l, I_RHOGE]  = diff_we[:, :, k, l, 2]
                #end k loop
            #end l loop

            for iv in range(I_RHOG, I_RHOGW + 1):
                for l in range(lall):
                    for k in range(kall):
                        PROG_mean[:, :, k, l, iv] += PROG_split[:, :, k, l, iv] * rweight_itr
                    #end k loop
                #end l loop
            #end iv loop

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

            prf.PROF_rapend  ('____vi_path2',2)

        #end ns loop  # small step end
        #---------------------------------------------------------------------------
        #
        #
        #
        #---------------------------------------------------------------------------
        prf.PROF_rapstart('____vi_path3',2)

        # update prognostic variables

        for l in range(lall):
            for k in range(kall):
                PROG[:, :, k, l, I_RHOG:I_RHOGE + 1] += PROG_split[:, :, k, l, I_RHOG:I_RHOGE + 1]
            #end k loop
        #end l loop

        if adm.ADM_have_pl:
            PROG_pl[:,:,:,:] += PROG_split_pl[:,:,:,:]
        #endif

        oprt.OPRT_horizontalize_vec( 
            PROG[:,:,:,I_RHOGVX], PROG_pl[:,:,:,I_RHOGVX], # [INOUT]
            PROG[:,:,:,I_RHOGVY], PROG_pl[:,:,:,I_RHOGVY], # [INOUT]
            PROG[:,:,:,I_RHOGVZ], PROG_pl[:,:,:,I_RHOGVZ], # [INOUT]
            grd, rdtype,
        )
        
        # communication of mean velocity
        comm.COMM_data_transfer( PROG_mean, PROG_mean_pl )

        prf.PROF_rapend  ('____vi_path3',2)

        return
    
    
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
    ):
        return
    


