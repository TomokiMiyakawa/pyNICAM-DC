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
            cnst, comm, grd, oprt, vmtr, tim, rcnf, bndc, cnvv, numf, src, rdtype,                  
    ):
        
        prf.PROF_rapstart('____vi_path0',2)   

        gall_1d = adm.ADM_gall_1d
        gall_pl = adm.ADM_gall_pl
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

        GRAV  = cnst.CONST_GRAV
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

#                with open(std.fname_log, 'a') as log_file:
#                    log_file.write(f"eth shape: {eth.shape}\n")

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

                # with open (std.fname_log, 'a') as log_file:
                #     log_file.write(f"eth_h_pl shape: {eth_h_pl.shape}\n")
                #     log_file.write(f"eth_pl shape: {eth_pl.shape}\n")
                #     log_file.write(f"kimn, kmax: {kmin}, {kmax}\n")
                # prc.prc_mpistop(std.io_l, std.fname_log)

                eth_h_pl[:, kmin:kmax+2, l] = (
                    grd.GRD_afact[kmin:kmax+2][None, :] * eth_pl[:, kmin:kmax+2, l] +
                    grd.GRD_bfact[kmin:kmax+2][None, :] * eth_pl[:, kmin-1:kmax+1, l]
                )

                # Fill ghost level
                rhog_h_pl[:, kmin-1, l] = rhog_h_pl[:, kmin, l]
                eth_h_pl[:, kmin-1, l]  = eth_h_pl[:, kmin, l]
            #end l loop
        #endif

        # prc.prc_mpistop(std.io_l, std.fname_log)

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

        #prc.prc_mpistop(std.io_l, std.fname_log)

        # with open (std.fname_log, 'a') as log_file:
        #     print("O: in vi_small_step, into numfilter_divdamp ", file=log_file)
        #     print("PROG_pl[0,2,0,:] ", PROG_pl[0,2,0,:], file=log_file)


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
            comm, grd, oprt, vmtr, src, rdtype,
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

            
        # No overflow error upto this point
        #print("really?")
        #prc.prc_mpistop(std.io_l, std.fname_log)

        numf.numfilter_divdamp_2d(
            PROG   [:,:,:,:,I_RHOGVX], PROG_pl   [:,:,:,I_RHOGVX], # [IN]
            PROG   [:,:,:,:,I_RHOGVY], PROG_pl   [:,:,:,I_RHOGVY], # [IN]
            PROG   [:,:,:,:,I_RHOGVZ], PROG_pl   [:,:,:,I_RHOGVZ], # [IN]
            ddivdvx_2d[:,:,:,:],       ddivdvx_2d_pl[:,:,:],       # [OUT]
            ddivdvy_2d[:,:,:,:],       ddivdvy_2d_pl[:,:,:],       # [OUT]
            ddivdvz_2d[:,:,:,:],       ddivdvz_2d_pl[:,:,:],       # [OUT]
            comm, grd, oprt, rdtype,
        )

        # with open (std.fname_log, 'a') as log_file:
        #     print("B: in vi_small_step, out of numfilter_divdamp_2d ", file=log_file)
        #     print("ddivdvx_2d[6,5,2,0] ", ddivdvx_2d[6,5,2,0], file=log_file)
        #     print("ddivdvx_2d_pl[0,2,0] ", ddivdvx_2d_pl[0,2,0], file=log_file)
        #     print("ddivdvy_2d[6,5,2,0] ", ddivdvy_2d[6,5,2,0], file=log_file)
        #     print("ddivdvy_2d_pl[0,2,0] ", ddivdvy_2d_pl[0,2,0], file=log_file)
        #     print("ddivdvz_2d[6,5,2,0] ", ddivdvz_2d[6,5,2,0], file=log_file)
        #     print("ddivdvz_2d_pl[0,2,0] ", ddivdvz_2d_pl[0,2,0], file=log_file)
        
        # overflow error 
        with open(std.fname_log, 'a') as log_file:
            print("really?", file=log_file)
        #prc.prc_mpistop(std.io_l, std.fname_log)


        # pressure force
        src.src_pres_gradient(
            preg_prim[:,:,:,:],   preg_prim_pl[:,:,:],   # [IN]
            dpgrad   [:,:,:,:,:], dpgrad_pl   [:,:,:,:], # [OUT]
            dpgradw  [:,:,:,:],   dpgradw_pl  [:,:,:],   # [OUT]
            src.I_SRC_default,                           # [IN]
            grd, oprt, vmtr, rdtype,   
        )

        #prc.prc_mpistop(std.io_l, std.fname_log)


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
                gz_tilde[:, :, k, l] = GRAV - (dpgradw[:, :, k, l] - dbuoiw[:, :, k, l]) / rhog_h[:, :, k, l]
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
                gz_tilde_pl[:, :, l] = GRAV - (dpgradw_pl[:, :, l] - dbuoiw_pl[:, :, l]) / rhog_h_pl[:, :, l]
                drhoge_pwh_pl[:, :, l] = -gz_tilde_pl[:, :, l] * PROG_pl[:, :, l, I_RHOGW]

                # --- Vectorized drhoge_pw_pl over kmin to kmax
                drhoge_pw_pl[:, kmin:kmax+1, l] = (
                    vx_pl[:, kmin:kmax+1, l] * dpgrad_pl[:, kmin:kmax+1, l, XDIR] +
                    vy_pl[:, kmin:kmax+1, l] * dpgrad_pl[:, kmin:kmax+1, l, YDIR] +
                    vz_pl[:, kmin:kmax+1, l] * dpgrad_pl[:, kmin:kmax+1, l, ZDIR] +
                    vmtr.VMTR_W2Cfact_pl[:, kmin:kmax+1, 0, l] * drhoge_pwh_pl[:, kmin+1:kmax+2, l] +
                    vmtr.VMTR_W2Cfact_pl[:, kmin:kmax+1, 1, l] * drhoge_pwh_pl[:, kmin:kmax+1,   l]
                )

                # --- Ghost layers at boundaries
                drhoge_pw_pl[:, kmin-1, l] = 0.0
                drhoge_pw_pl[:, kmax+1, l] = 0.0
            # end l loop
        #endif

        # overflow error 
        #print("really?")
        #prc.prc_mpistop(std.io_l, std.fname_log)


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

        # with open(std.fname_log, 'a') as log_file:  
        #     print("g_TEND added before smallstep iteration (6,5,2,0,:)", file=log_file) 
        #     print(g_TEND[6, 5, 2, 0, :], file=log_file) 

        # prc.prc_mpistop(std.io_l, std.fname_log)
        # initialization of mean mass flux

        rweight_itr = 1.0 / rdtype(num_of_itr)
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

            #prc.prc_mpistop(std.io_l, std.fname_log)

            if tim.TIME_split:
                #---< Calculation of source term for Vh(vx,vy,vz) and W (split) >

                # divergence damping

                numf.numfilter_divdamp(
                    PROG_split[:,:,:,:,I_RHOGVX], PROG_split_pl[:,:,:,I_RHOGVX], # [IN]
                    PROG_split[:,:,:,:,I_RHOGVY], PROG_split_pl[:,:,:,I_RHOGVY], # [IN]
                    PROG_split[:,:,:,:,I_RHOGVZ], PROG_split_pl[:,:,:,I_RHOGVZ], # [IN]
                    PROG_split[:,:,:,:,I_RHOGW ], PROG_split_pl[:,:,:,I_RHOGW ], # [IN]
                    ddivdvx[:,:,:,:],             ddivdvx_pl[:,:,:],             # [OUT]
                    ddivdvy[:,:,:,:],             ddivdvy_pl[:,:,:],             # [OUT]
                    ddivdvz[:,:,:,:],             ddivdvz_pl[:,:,:],             # [OUT]
                    ddivdw [:,:,:,:],             ddivdw_pl [:,:,:],             # [OUT]
                    comm, grd, oprt, vmtr, src, rdtype,
                )

                # 2d divergence damping
                numf.numfilter_divdamp_2d(
                    PROG_split[:,:,:,:,I_RHOGVX], PROG_split_pl[:,:,:,I_RHOGVX], # [IN]
                    PROG_split[:,:,:,:,I_RHOGVY], PROG_split_pl[:,:,:,I_RHOGVY], # [IN]
                    PROG_split[:,:,:,:,I_RHOGVZ], PROG_split_pl[:,:,:,I_RHOGVZ], # [IN]
                    ddivdvx_2d[:,:,:,:],          ddivdvx_2d_pl[:,:,:],          # [OUT]
                    ddivdvy_2d[:,:,:,:],          ddivdvy_2d_pl[:,:,:],          # [OUT]
                    ddivdvz_2d[:,:,:,:],          ddivdvz_2d_pl[:,:,:],          # [OUT]
                    comm, grd, oprt, rdtype,
                )

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

            # diff_vh at k=41 has issues!!!

            # with open (std.fname_log, 'a') as log_file:
            #     print("UPPER BNDCHECK", file=log_file)
            #     print("diff_vh[6,5,41,0,:]  ", diff_vh[6,5,41,0,:], file=log_file)
            #     print("diff_vh_pl[:,41,0,0] ", diff_vh_pl[:,41,0,0], file=log_file)
            #     print("diff_vh_pl[:,41,0,1] ", diff_vh_pl[:,41,0,1], file=log_file)
            #     print("diff_vh_pl[:,41,0,2] ", diff_vh_pl[:,41,0,2], file=log_file)
            #     print("PROG[6,5,41,0,:] ",     PROG[6,5,41,0,:], file=log_file)

            # treatment for boundary condition
            bndc.BNDCND_rhovxvyvz( 
                PROG   [:,:,:,:,I_RHOG], # [IN]
                diff_vh[:,:,:,:,0],      # [INOUT]
                diff_vh[:,:,:,:,1],      # [INOUT]
                diff_vh[:,:,:,:,2],      # [INOUT]
            )



            if adm.ADM_have_pl:
                #bndc.BNDCND_rhovxvyvz(
                #     PROG_pl   [:,np.newaxis,:,:,I_RHOG], # [IN]
                #     diff_vh_pl[:,np.newaxis,:,:,0],      # [INOUT]
                #     diff_vh_pl[:,np.newaxis,:,:,1],      # [INOUT]
                #     diff_vh_pl[:,np.newaxis,:,:,2],      # [INOUT]
                # )
                bndc.BNDCND_rhovxvyvz_pl(
                    PROG_pl   [:,:,:,I_RHOG], # [IN]
                    diff_vh_pl[:,:,:,0],      # [INOUT]
                    diff_vh_pl[:,:,:,1],      # [INOUT]
                    diff_vh_pl[:,:,:,2],      # [INOUT]
                )
                # check whether or not squeeze is needed to remove the dummy axis 
            #endif

            comm.COMM_data_transfer( diff_vh, diff_vh_pl )

            prf.PROF_rapend  ('____vi_path1',2)
            prf.PROF_rapstart('____vi_path2',2)

            #prc.prc_mpistop(std.io_l, std.fname_log)

            # with open(std.fname_log, 'a') as log_file:  
            #     print("", file=log_file)
            #     print("", file=log_file)
            #     print("check before vi_main", file=log_file) 
            #     print("diff_vh", file=log_file)
            #     print(diff_vh[6, 5, 41, 0, :], file=log_file) 
            #     print("PROG_split", file=log_file)
            #     print(PROG_split [6, 5, 41, 0, :], file=log_file)
            #     print("preg_prim_split", file=log_file)
            #     print(preg_prim_split[6, 5, 41, 0], file=log_file)
            #     print("PROG", file=log_file)
            #     print(PROG [6, 5, 41, 0, :], file=log_file)
            #     print("eth", file=log_file)
            #     print(eth[6, 5, 41, 0], file=log_file)
            #     print("g_TEND", file=log_file)
            #     print(g_TEND[6, 5, 41, 0, :], file=log_file)
            #     print("drhogw", file=log_file)
            #     print(drhogw[6, 5, 41, 0], file=log_file)
            #     print("grhogetot0", file=log_file)
            #     print(grhogetot0[6, 5, 41, 0], file=log_file)
            #     print("dt", dt, file=log_file)
            #     print("", file=log_file)
            #     print("", file=log_file)

                
                # print("check before vi_main", file=log_file) 
                # print("diff_vh", file=log_file)
                # print(diff_vh[6, 5, 0, 0, :], file=log_file) 
                # print("PROG_split", file=log_file)
                # print(PROG_split [6, 5, 0, 0, :], file=log_file)
                # print("preg_prim_split", file=log_file)
                # print(preg_prim_split[6, 5, 0, 0], file=log_file)
                # print("PROG", file=log_file)
                # print(PROG [6, 5, 0, 0, :], file=log_file)
                # print("eth", file=log_file)
                # print(eth[6, 5, 0, 0], file=log_file)
                # print("g_TEND", file=log_file)
                # print(g_TEND[6, 5, 0, 0, :], file=log_file)
                # print("drhogw", file=log_file)
                # print(drhogw[6, 5, 0, 0], file=log_file)
                # print("grhogetot0", file=log_file)
                # print(grhogetot0[6, 5, 0, 0], file=log_file)
                # print("dt", dt, file=log_file)
                # print("", file=log_file)
                # print("", file=log_file)

            # with open(std.fname_log, 'a') as log_file:  
            #     print("", file=log_file)
            #     print("", file=log_file)
            #     print("check before vi_main", file=log_file) 
            #     print("diff_vh", file=log_file)
            #     print(diff_vh[6, 5, 2, 0, :], file=log_file) 
            #     print("PROG_split", file=log_file)
            #     print(PROG_split [6, 5, 2, 0, :], file=log_file)
            #     print("preg_prim_split", file=log_file)
            #     print(preg_prim_split[6, 5, 2, 0], file=log_file)
            #     print("PROG", file=log_file)
            #     print(PROG [6, 5, 2, 0, :], file=log_file)
            #     print("eth", file=log_file)
            #     print(eth[6, 5, 2, 0], file=log_file)
            #     print("g_TEND", file=log_file)
            #     print(g_TEND[6, 5, 2, 0, :], file=log_file)
            #     print("drhogw", file=log_file)
            #     print(drhogw[6, 5, 2, 0], file=log_file)
            #     print("grhogetot0", file=log_file)
            #     print(grhogetot0[6, 5, 2, 0], file=log_file)
            #     print("dt", dt, file=log_file)
            #     print("", file=log_file)
            #     print("", file=log_file)

            #print("stopper")            
            #prc.prc_mpistop(std.io_l, std.fname_log)

            #---< vertical implicit scheme >
            self.vi_main(
                diff_we        [:,:,:,:,0],        diff_we_pl        [:,:,:,0],        # [OUT]
                diff_we        [:,:,:,:,1],        diff_we_pl        [:,:,:,1],        # [OUT]
                diff_we        [:,:,:,:,2],        diff_we_pl        [:,:,:,2],        # [OUT]
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
                PROG           [:,:,:,:,I_RHOG],   PROG_pl           [:,:,:,I_RHOG],   # [IN]
                PROG           [:,:,:,:,I_RHOGVX], PROG_pl           [:,:,:,I_RHOGVX], # [IN]
                PROG           [:,:,:,:,I_RHOGVY], PROG_pl           [:,:,:,I_RHOGVY], # [IN]
                PROG           [:,:,:,:,I_RHOGVZ], PROG_pl           [:,:,:,I_RHOGVZ], # [IN]
                PROG           [:,:,:,:,I_RHOGW],  PROG_pl           [:,:,:,I_RHOGW],  # [IN]
                eth            [:,:,:,:],          eth_pl            [:,:,:],          # [IN]
                g_TEND         [:,:,:,:,I_RHOG],   g_TEND_pl         [:,:,:,I_RHOG],   # [IN]
                drhogw         [:,:,:,:],          drhogw_pl         [:,:,:],          # [IN]
                g_TEND         [:,:,:,:,I_RHOGE],  g_TEND_pl         [:,:,:,I_RHOGE],  # [IN]
                grhogetot0     [:,:,:,:],          grhogetot0_pl     [:,:,:],          # [IN]
                dt,                                                                    # [IN]
                rcnf, cnst, vmtr, tim, grd, oprt, bndc, cnvv, src, rdtype, 
            )

            # with open(std.fname_log, 'a') as log_file:  
            #     print("", file=log_file)
            #     print("check after vi_main", file=log_file) 
            #     print("diff_we", file=log_file)
            #     print(diff_we[6, 5, 2, 0, :], file=log_file) 
            #     print("", file=log_file)

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

            #print("p1stop") 
            #prc.prc_mpistop(std.io_l, std.fname_log)

        #end ns loop  # small step end

        # print("p2stop") # Error remains before this point
        # prc.prc_mpistop(std.io_l, std.fname_log)

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



        #print("pppstop") # Error remains before this point
        #prc.prc_mpistop(std.io_l, std.fname_log)


        prf.PROF_rapend  ('____vi_path3',2)

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

        gall_1d = adm.ADM_gall_1d
        gall_pl = adm.ADM_gall_pl
        kall = adm.ADM_kdall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        lall_pl = adm.ADM_lall_pl

        # Mc     = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        # Mu     = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        # Ml     = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)
        # Mc_pl  = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        # Mu_pl  = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        # Ml_pl  = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)
        Mc     = self.Mc
        Mu     = self.Mu
        Ml     = self.Ml
        Mc_pl  = self.Mc_pl
        Mu_pl  = self.Mu_pl
        Ml_pl  = self.Ml_pl


        GRAV  = cnst.CONST_GRAV
        Rdry  = cnst.CONST_Rdry
        CVdry = cnst.CONST_CVdry

        GCVovR   = GRAV * CVdry / Rdry
        ACVovRt2 = rdtype(rcnf.NON_HYDRO_ALPHA) * CVdry / Rdry / ( dt*dt )

        for l in range(lall):
            for k in range(kmin + 1, kmax + 1):
                # Common vertical scalars
                rgdzh   = grd.GRD_rdgzh[k]
                rgdz    = grd.GRD_rdgz[k]
                rgdzm1  = grd.GRD_rdgz[k - 1]
                dfact   = grd.GRD_dfact[k]
                cfactm1 = grd.GRD_cfact[k - 1]
                dfactm1 = grd.GRD_dfact[k - 1]
                cfact   = grd.GRD_cfact[k]

                # ---- Mc ----
                Mc[:, :, k, l] = (
                    ACVovRt2 / vmtr.VMTR_RGSQRTH[:, :, k, l]
                    + rgdzh * (
                        (vmtr.VMTR_RGSGAM2[:, :, k, l] * rgdz + vmtr.VMTR_RGSGAM2[:, :, k - 1, l] * rgdzm1)
                        * vmtr.VMTR_GAM2H[:, :, k, l] * eth[:, :, k, l]
                        - (dfact - cfactm1) * (g_tilde[:, :, k, l] + GCVovR)
                    )
                )

                # ---- Mu ----
                Mu[:, :, k, l] = -rgdzh * (
                    vmtr.VMTR_RGSGAM2[:, :, k, l] * rgdz
                    * vmtr.VMTR_GAM2H[:, :, k + 1, l] * eth[:, :, k + 1, l]
                    + cfact * (
                        g_tilde[:, :, k + 1, l]
                        + vmtr.VMTR_GAM2H[:, :, k + 1, l] * vmtr.VMTR_RGAMH[:, :, k, l]**2 * GCVovR
                    )
                )

                # ---- Ml ----
                Ml[:, :, k, l] = -rgdzh * (
                    vmtr.VMTR_RGSGAM2[:, :, k, l] * rgdz
                    * vmtr.VMTR_GAM2H[:, :, k - 1, l] * eth[:, :, k - 1, l]
                    - dfactm1 * (
                        g_tilde[:, :, k - 1, l]
                        + vmtr.VMTR_GAM2H[:, :, k - 1, l] * vmtr.VMTR_RGAMH[:, :, k, l]**2 * GCVovR
                    )
                )
            #end k loop
        #end l loop

        if adm.ADM_have_pl:
            for l in range(adm.ADM_lall_pl):
                for k in range(kmin + 1, kmax + 1):  # include kmax
                    # Vectorized over g
                    Mc_pl[:, k, l] = (
                        ACVovRt2 / vmtr.VMTR_RGSQRTH_pl[:, k, l] +
                        grd.GRD_rdgzh[k] * (
                            (vmtr.VMTR_RGSGAM2_pl[:, k, l] * grd.GRD_rdgz[k] +
                            vmtr.VMTR_RGSGAM2_pl[:, k - 1, l] * grd.GRD_rdgz[k - 1]) *
                            vmtr.VMTR_GAM2H_pl[:, k, l] * eth_pl[:, k, l] -
                            (grd.GRD_dfact[k] - grd.GRD_cfact[k - 1]) *
                            (g_tilde_pl[:, k, l] + GCVovR)
                        )
                    )

                    Mu_pl[:, k, l] = -grd.GRD_rdgzh[k] * (
                        vmtr.VMTR_RGSGAM2_pl[:, k, l] * grd.GRD_rdgz[k] *
                        vmtr.VMTR_GAM2H_pl[:, k + 1, l] * eth_pl[:, k + 1, l] +
                        grd.GRD_cfact[k] * (
                            g_tilde_pl[:, k + 1, l] +
                            vmtr.VMTR_GAM2H_pl[:, k + 1, l] * vmtr.VMTR_RGAMH_pl[:, k, l] ** 2 * GCVovR
                        )
                    )

                    Ml_pl[:, k, l] = -grd.GRD_rdgzh[k] * (
                        vmtr.VMTR_RGSGAM2_pl[:, k, l] * grd.GRD_rdgz[k] *
                        vmtr.VMTR_GAM2H_pl[:, k - 1, l] * eth_pl[:, k - 1, l] -
                        grd.GRD_dfact[k - 1] * (
                            g_tilde_pl[:, k - 1, l] +
                            vmtr.VMTR_GAM2H_pl[:, k - 1, l] * vmtr.VMTR_RGAMH_pl[:, k, l] ** 2 * GCVovR
                        )
                    )
                # end k loop
            #end l loop
        #endif 

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
    ):
        
        gall_1d = adm.ADM_gall_1d
        kall = adm.ADM_kdall
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        lall_pl = adm.ADM_lall_pl

        drhog         = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)   # source term at t=n+1
        drhog_pl      = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  
        drhoge        = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  
        drhoge_pl     = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  
        drhogetot     = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  
        drhogetot_pl  = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  

        grhog1         = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  # source term ( large step + t=n+1 )
        grhog1_pl      = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  
        grhoge1        = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  
        grhoge1_pl     = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  
        gpre           = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  
        gpre_pl        = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  

        rhog1          = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  # prognostic vars ( previous + t=n,t=n+1 )
        rhog1_pl       = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  
        rhogvx1        = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  
        rhogvx1_pl     = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  
        rhogvy1        = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  
        rhogvy1_pl     = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  
        rhogvz1        = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  
        rhogvz1_pl     = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  
        rhogw1         = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  
        rhogw1_pl      = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  

        rhogkin0       = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  # kinetic energy ( previous                )
        rhogkin0_pl    = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  
        rhogkin10      = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  # kinetic energy ( previous + split(t=n)   )
        rhogkin10_pl   = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  
        rhogkin11      = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  # kinetic energy ( previous + split(t=n+1) )
        rhogkin11_pl   = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype)  
        ethtot0        = np.empty((gall_1d, gall_1d, kall, lall,   ), dtype=rdtype)  # total enthalpy ( h + v^{2}/2 + phi, previous )
        ethtot0_pl     = np.empty((gall_pl,          kall, lall_pl,), dtype=rdtype) 

        Rdry  = cnst.CONST_Rdry
        CVdry = cnst.CONST_CVdry

        #print("pstop")
        #prc.prc_mpistop(std.io_l, std.fname_log)

        #---< update grhog & grhoge >

        if tim.TIME_split:
            # horizontal flux convergence
            src.src_flux_convergence( 
                rhogvx_split1, rhogvx_split1_pl, # [IN]
                rhogvy_split1, rhogvy_split1_pl, # [IN]
                rhogvz_split1, rhogvz_split1_pl, # [IN]
                rhogw_split0,  rhogw_split0_pl,  # [IN]
                drhog,         drhog_pl,         # [OUT]
                src.I_SRC_horizontal,            # [IN]
                grd, oprt, vmtr, rdtype,
            )

        # horizontal advection convergence
            src.src_advection_convergence(
                rhogvx_split1, rhogvx_split1_pl, # [IN]
                rhogvy_split1, rhogvy_split1_pl, # [IN]
                rhogvz_split1, rhogvz_split1_pl, # [IN]
                rhogw_split0,  rhogw_split0_pl,  # [IN]
                eth0,          eth0_pl,          # [IN]
                drhoge,        drhoge_pl,        # [OUT]
                src.I_SRC_horizontal,            # [IN]
                grd, oprt, vmtr, rdtype,
            ) 

        else:

            for l in range(lall):
                for k in range(kall):
                    drhog[:, :, k, l] = 0.0
                    drhoge[:, :, k, l] = 0.0

            drhog_pl[:, :, :] = 0.0
            drhoge_pl[:, :, :] = 0.0

        #endif

        # update grhog, grhoge and calc source term of pressure

        for l in range(lall):
            for k in range(kall):
                grhog1[:, :, k, l]  = grhog[:, :, k, l]  + drhog[:, :, k, l]
                grhoge1[:, :, k, l] = grhoge[:, :, k, l] + drhoge[:, :, k, l]
                gpre[:, :, k, l]    = grhoge1[:, :, k, l] * Rdry / CVdry
            # end k loop
        # end l loop

        if adm.ADM_have_pl:
            grhog1_pl  = grhog_pl  + drhog_pl
            grhoge1_pl = grhoge_pl + drhoge_pl
            gpre_pl    = grhoge1_pl * Rdry / CVdry
        #endif

        #---------------------------------------------------------------------------
        # verical implict calculation core
        #---------------------------------------------------------------------------

        # boundary condition for rhogw_split1

        for l in range(lall):
            for k in range(kall):
                rhogw_split1[:, :, k, l] = 0.0
        
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


        for l in range(lall):
            bndc.BNDCND_rhow(
                rhogvx_split1 [:,:,:,l],       # [IN]
                rhogvy_split1 [:,:,:,l],       # [IN]
                rhogvz_split1 [:,:,:,l],       # [IN]
                rhogw_split1  [:,:,:,l],       # [INOUT]
                vmtr.VMTR_C2WfactGz[:,:,:,:,l] # [IN]
            )
        #end loop l

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
            rhogw_split1_pl[:,:,:] = 0.0        # Tracing start from here
            
            # for l in range(adm.ADM_lall_pl):
            #     rxpl1=np.empty((gall_pl, kall), dtype=rdtype)
            #     rxpl1[:,:]=rhogvx_split1_pl[:,:,l]
            #     bndc.BNDCND_rhow(
            #         rhogvx_split1_pl [:,np.newaxis,:,l],     # [IN]
            #         rhogvy_split1_pl [:,np.newaxis,:,l],     # [IN]
            #         rhogvz_split1_pl [:,np.newaxis,:,l],     # [IN]
            #         rhogw_split1_pl  [:,np.newaxis,:,l],     # [INOUT]      # Here?
            #         vmtr.VMTR_C2WfactGz_pl[:,np.newaxis,:,:,l]    # [IN]
            #     )
            #end loop l
            for l in range(adm.ADM_lall_pl):
                rxpl1=np.empty((gall_pl, kall), dtype=rdtype)
                rxpl1[:,:]=rhogvx_split1_pl[:,:,l]
                bndc.BNDCND_rhow_pl(
                    rhogvx_split1_pl [:,:,l],     # [IN]
                    rhogvy_split1_pl [:,:,l],     # [IN]
                    rhogvz_split1_pl [:,:,l],     # [IN]
                    rhogw_split1_pl  [:,:,l],     # [INOUT]      # Here?
                    vmtr.VMTR_C2WfactGz_pl[:,:,:,l]    # [IN]
                )

            # with open(std.fname_log, 'a') as log_file:
            #     print("after BNDCND_rhow_pl", file=log_file)
            #     print("rhogw_split1_pl", file=log_file)
            #     print(rhogw_split1_pl[:, 0, 0], file=log_file)
            #     print(rhogw_split1_pl[:, 2, 0], file=log_file)
            #     print(rhogw_split1_pl[:,41, 0], file=log_file)  

        #endif

        # update rhogw_split1
        self.vi_rhow_solver(
            rhogw_split1,     rhogw_split1_pl,     # [INOUT]     #Here?
            rhogw_split0,     rhogw_split0_pl,     # [IN]
            preg_prim_split0, preg_prim_split0_pl, # [IN]
            rhog_split0,      rhog_split0_pl,      # [IN]
            grhog1,           grhog1_pl,           # [IN]
            grhogw,           grhogw_pl,           # [IN]
            gpre,             gpre_pl,             # [IN]
            dt,                                    # [IN]
            cnst, grd, vmtr, rcnf, rdtype, 
        )

        # update rhog_split1
        src.src_flux_convergence(
            rhogvx_split1, rhogvx_split1_pl, # [IN]
            rhogvy_split1, rhogvy_split1_pl, # [IN]
            rhogvz_split1, rhogvz_split1_pl, # [IN]
            rhogw_split1,  rhogw_split1_pl,  # [IN]
            drhog,         drhog_pl,         # [OUT]
            src.I_SRC_default,              # [IN]
            grd, oprt, vmtr, rdtype,
        )

        # check why split1 is different from the original code.

        for l in range(lall):
            for k in range(kall):
                rhog_split1[:, :, k, l] = rhog_split0[:, :, k, l] + (grhog[:, :, k, l] + drhog[:, :, k, l]) * dt

        if adm.ADM_have_pl:
            rhog_split1_pl[:, :, :] = rhog_split0_pl[:, :, :] + (grhog_pl[:, :, :] + drhog_pl[:, :, :]) * dt
        #endif

#         with open(std.fname_log, 'a') as log_file:
#             print("", file=log_file)
#             print("rhog_split1_pl before Satoh2002", file=log_file)
# #            print("rhog_split1", file=log_file)
#             print(rhog_split1_pl[:, 2, 0], file=log_file)
#             print(rhog_split0_pl[:, 2, 0], file=log_file)
#             print(grhog_pl[:, 2, 0], file=log_file)
#             print(drhog_pl[:, 2, 0], file=log_file)         # element zero of axis 0 may have an issue
#             print(rhog_split1_pl[:, 0, 0], file=log_file)
#             print(rhog_split0_pl[:, 0, 0], file=log_file)
#             print(grhog_pl[:, 0, 0], file=log_file)
#             print(drhog_pl[:, 0, 0], file=log_file)
  
#             print("", file=log_file)

        #---------------------------------------------------------------------------
        # energy correction by Etotal (Satoh,2002)
        #---------------------------------------------------------------------------

        # overflow encountered during cnvvar_rhogkin (not always, so it is likely an array issue)

        # calc rhogkin ( previous )
        rhogkin0, rhogkin0_pl = cnvv.cnvvar_rhogkin(
                                    rhog0,    rhog0_pl,    # [IN]
                                    rhogvx0,  rhogvx0_pl,  # [IN]
                                    rhogvy0,  rhogvy0_pl,  # [IN]
                                    rhogvz0,  rhogvz0_pl,  # [IN]
                                    rhogw0,   rhogw0_pl,   # [IN]
                                    vmtr, rdtype,
                                )

        # with open(std.fname_log, 'a') as log_file:
        #     print("", file=log_file)
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

        for l in range(lall):
            for k in range(kall):
                rhog1[:, :, k, l]   = rhog0[:, :, k, l]   + rhog_split0[:, :, k, l]
                rhogvx1[:, :, k, l] = rhogvx0[:, :, k, l] + rhogvx_split0[:, :, k, l]
                rhogvy1[:, :, k, l] = rhogvy0[:, :, k, l] + rhogvy_split0[:, :, k, l]
                rhogvz1[:, :, k, l] = rhogvz0[:, :, k, l] + rhogvz_split0[:, :, k, l]
                rhogw1[:, :, k, l]  = rhogw0[:, :, k, l]  + rhogw_split0[:, :, k, l]

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
                                        vmtr, rdtype,
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

        for l in range(lall):
            for k in range(kall):
                rhog1[:, :, k, l]   = rhog0[:, :, k, l]   + rhog_split1[:, :, k, l]
                rhogvx1[:, :, k, l] = rhogvx0[:, :, k, l] + rhogvx_split1[:, :, k, l]
                rhogvy1[:, :, k, l] = rhogvy0[:, :, k, l] + rhogvy_split1[:, :, k, l]
                rhogvz1[:, :, k, l] = rhogvz0[:, :, k, l] + rhogvz_split1[:, :, k, l]
                rhogw1[:, :, k, l]  = rhogw0[:, :, k, l]  + rhogw_split1[:, :, k, l]  # issue

        if adm.ADM_have_pl:
            rhog1_pl[:, :, :]   = rhog0_pl[:, :, :]   + rhog_split1_pl[:, :, :]       # big issue
            rhogvx1_pl[:, :, :] = rhogvx0_pl[:, :, :] + rhogvx_split1_pl[:, :, :]     # 0,2,0  issue
            rhogvy1_pl[:, :, :] = rhogvy0_pl[:, :, :] + rhogvy_split1_pl[:, :, :]     # 0,2,0
            rhogvz1_pl[:, :, :] = rhogvz0_pl[:, :, :] + rhogvz_split1_pl[:, :, :]     # 0,2,0 
            rhogw1_pl[:, :, :]  = rhogw0_pl[:, :, :]  + rhogw_split1_pl[:, :, :]      # 0,2,0   2,2,0  issue

        # calc rhogkin ( previous + split(t=n+1) )
        rhogkin11, rhogkin11_pl = cnvv.cnvvar_rhogkin(
                                        rhog1,    rhog1_pl,      # [IN]
                                        rhogvx1,  rhogvx1_pl,    # [IN]
                                        rhogvy1,  rhogvy1_pl,    # [IN]
                                        rhogvz1,  rhogvz1_pl,    # [IN]
                                        rhogw1,   rhogw1_pl,     # [IN]
                                        vmtr, rdtype,
                                    )
        
        with open(std.fname_log, 'a') as log_file:
            print("", file=log_file)
            print("rhog1",   rhog1  [6,5,2,0], file=log_file)
            print("rhogvx1", rhogvx1[6,5,2,0], file=log_file)
            print("rhogvy1", rhogvy1[6,5,2,0], file=log_file)
            print("rhogvz1", rhogvz1[6,5,2,0], file=log_file)
            print("rhogw1",  rhogw1 [6,5,2,0], file=log_file)
            print("rhog1_pl 0,2 ",   rhog1_pl  [0,2,0], file=log_file)            #!
            print("rhogvx1_pl   ", rhogvx1_pl[0,2,0], file=log_file)              
            print("rhogvy1_pl   ", rhogvy1_pl[0,2,0], file=log_file)
            print("rhogvz1_pl   ", rhogvz1_pl[0,2,0], file=log_file)              #!
            print("rhogw1_pl    ",  rhogw1_pl[0,2,0], file=log_file)              #!
            print("rhog1_pl 2,2 ",   rhog1_pl[2,2,0], file=log_file)
            print("rhogvx1_pl   ", rhogvx1_pl[2,2,0], file=log_file)
            print("rhogvy1_pl   ", rhogvy1_pl[2,2,0], file=log_file)
            print("rhogvz1_pl   ", rhogvz1_pl[2,2,0], file=log_file)
            print("rhogw1_pl    ",  rhogw1_pl[2,2,0], file=log_file)            
            print("rhogkin11        ",     rhogkin11[6,5,2,0], file=log_file)
            print("rhogkin11_pl 0,2 ",  rhogkin11_pl[0,2,0], file=log_file)        #!
            print("rhogkin11_pl 2,2 ",  rhogkin11_pl[2,2,0], file=log_file)        #!

        # calculate total enthalpy ( h + v^{2}/2 + phi, previous )

        for l in range(lall):
            for k in range(kall):
                ethtot0[:, :, k, l] = (
                    eth0[:, :, k, l]
                    + rhogkin0[:, :, k, l] / rhog0[:, :, k, l]
                    + vmtr.VMTR_PHI[:, :, k, l]
                )

        if adm.ADM_have_pl:
            ethtot0_pl[:, :, :] = (
                eth0_pl[:, :, :]
                + rhogkin0_pl[:, :, :] / rhog0_pl[:, :, :]
                + vmtr.VMTR_PHI_pl[:, :, :]
            )

        # advection convergence for eth + kin + phi
        src.src_advection_convergence(
            rhogvx1,    rhogvx1_pl,   # [IN]
            rhogvy1,    rhogvy1_pl,   # [IN]
            rhogvz1,    rhogvz1_pl,   # [IN]
            rhogw1,     rhogw1_pl,    # [IN]
            ethtot0,    ethtot0_pl,   # [IN]
            drhogetot,  drhogetot_pl, # [OUT]
            src.I_SRC_default,        # [IN]
            grd, oprt, vmtr, rdtype,
        )

        for l in range(lall):
            for k in range(kall):
                rhoge_split1[:, :, k, l] = (
                    rhoge_split0[:, :, k, l]
                    + (grhogetot[:, :, k, l] + drhogetot[:, :, k, l]) * dt
                    + (rhogkin10[:, :, k, l] - rhogkin11[:, :, k, l])
                    + (rhog_split0[:, :, k, l] - rhog_split1[:, :, k, l]) * vmtr.VMTR_PHI[:, :, k, l]
                )

        if adm.ADM_have_pl:
            rhoge_split1_pl[:, :, :] = (
                rhoge_split0_pl[:, :, :]
                + (grhogetot_pl[:, :, :] + drhogetot_pl[:, :, :]) * dt
                + (rhogkin10_pl[:, :, :] - rhogkin11_pl[:, :, :])
                + (rhog_split0_pl[:, :, :] - rhog_split1_pl[:, :, :]) * vmtr.VMTR_PHI_pl[:, :, :]
            )

        return


    #> Tridiagonal matrix solver
    def vi_rhow_solver(self,
        rhogw,  rhogw_pl,     # rho*w          ( G^1/2 x gam2 ), n+1
        rhogw0, rhogw0_pl,    # rho*w          ( G^1/2 x gam2 )
        preg0,  preg0_pl,     # pressure prime ( G^1/2 x gam2 )
        rhog0,  rhog0_pl,     # rho            ( G^1/2 x gam2 )
        Srho,   Srho_pl,      # source term for rho  at the full level
        Sw,     Sw_pl,        # source term for rhow at the half level
        Spre,   Spre_pl,      # source term for pres at the full level
        dt,
        cnst, grd, vmtr, rcnf, rdtype,                 
        ):

        prf.PROF_rapstart('____vi_rhow_solver',2)

        gall_1d = adm.ADM_gall_1d
        kall = adm.ADM_kdall
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax

        Sall     = np.empty((gall_1d, gall_1d, kall,), dtype=rdtype)  
        Sall_pl  = np.empty((gall_pl,          kall,), dtype=rdtype)  
        beta     = np.empty((gall_1d, gall_1d,), dtype=rdtype)  
        beta_pl  = np.empty((gall_pl,         ), dtype=rdtype)  
        gamma    = np.empty((gall_1d, gall_1d, kall,), dtype=rdtype)  
        gamma_pl = np.empty((gall_pl,          kall,), dtype=rdtype)  

        GRAV    = cnst.CONST_GRAV
        CVovRt2 = cnst.CONST_CVdry / cnst.CONST_Rdry / (dt*dt)    # Cv / R / dt**2
        alpha   = rdtype(rcnf.NON_HYDRO_ALPHA)


        for l in range(lall):
            for k in range(kmin + 1, kmax + 1):
                Sall[:, :, k] = (
                    (rhogw0[:, :, k, l] * alpha + dt * Sw[:, :, k, l]) * vmtr.VMTR_RGAMH[:, :, k, l]**2
                    - (
                        (preg0[:, :, k, l] + dt * Spre[:, :, k, l]) * vmtr.VMTR_RGSGAM2[:, :, k, l]
                        - (preg0[:, :, k - 1, l] + dt * Spre[:, :, k - 1, l]) * vmtr.VMTR_RGSGAM2[:, :, k - 1, l]
                    ) * dt * grd.GRD_rdgzh[k]
                    - (
                        (rhog0[:, :, k, l] + dt * Srho[:, :, k, l]) * vmtr.VMTR_RGAM[:, :, k, l]**2 * grd.GRD_afact[k]
                        + (rhog0[:, :, k - 1, l] + dt * Srho[:, :, k - 1, l]) * vmtr.VMTR_RGAM[:, :, k - 1, l]**2 * grd.GRD_bfact[k]
                    ) * dt * GRAV
                ) * CVovRt2

            # Boundary conditions
            rhogw[:, :, kmin, l]   *= vmtr.VMTR_RGSGAM2H[:, :, kmin, l]
            rhogw[:, :, kmax+1, l] *= vmtr.VMTR_RGSGAM2H[:, :, kmax+1, l]
            Sall[:, :, kmin+1] -= self.Ml[:, :, kmin+1, l] * rhogw[:, :, kmin, l]
            Sall[:, :, kmax]   -= self.Mu[:, :, kmax, l]   * rhogw[:, :, kmax+1, l]

            # Solve tri-diagonal matrix
            k = kmin + 1
            beta = self.Mc[:, :, k, l].copy()
            # print('beta', beta)
            # prc.prc_mpistop(std.io_l, std.fname_log)

            rhogw[:, :, k, l] = Sall[:, :, k] / beta    

            # Forward
            gamma = np.zeros((gall_1d, gall_1d, kall))  # Temporary storage for gamma
            for k in range(kmin + 2, kmax + 1):
                gamma[:, :, k] = self.Mu[:, :, k - 1, l] / beta
                beta = self.Mc[:, :, k, l] - self.Ml[:, :, k, l] * gamma[:, :, k]
                rhogw[:, :, k, l] = (Sall[:, :, k] - self.Ml[:, :, k, l] * rhogw[:, :, k - 1, l]) / beta

            # Backward
            for k in range(kmax - 1, kmin, -1):
                rhogw[:, :, k, l]   -= gamma[:, :, k + 1] * rhogw[:, :, k + 1, l]
                rhogw[:, :, k + 1, l] *= vmtr.VMTR_GSGAM2H[:, :, k + 1, l]

            # Boundary treatment
            rhogw[:, :, kmin, l]   *= vmtr.VMTR_GSGAM2H[:, :, kmin, l]
            rhogw[:, :, kmin+1, l] *= vmtr.VMTR_GSGAM2H[:, :, kmin+1, l]
            rhogw[:, :, kmax+1, l] *= vmtr.VMTR_GSGAM2H[:, :, kmax+1, l]

        # end l loop

        if adm.ADM_have_pl:
            for l in range(adm.ADM_lall_pl):
                for k in range(kmin + 1, kmax + 1):
                    for g in range(adm.ADM_gall_pl):
                        Sall_pl[g, k] = (
                            (rhogw0_pl[g, k, l] * alpha + dt * Sw_pl[g, k, l]) * vmtr.VMTR_RGAMH_pl[g, k, l]**2
                            - (
                                (preg0_pl[g, k, l] + dt * Spre_pl[g, k, l]) * vmtr.VMTR_RGSGAM2_pl[g, k, l]
                                - (preg0_pl[g, k - 1, l] + dt * Spre_pl[g, k - 1, l]) * vmtr.VMTR_RGSGAM2_pl[g, k - 1, l]
                            ) * dt * grd.GRD_rdgzh[k]
                            - (
                                (rhog0_pl[g, k, l] + dt * Srho_pl[g, k, l]) * vmtr.VMTR_RGAM_pl[g, k, l]**2 * grd.GRD_afact[k]
                                + (rhog0_pl[g, k - 1, l] + dt * Srho_pl[g, k - 1, l]) * vmtr.VMTR_RGAM_pl[g, k - 1, l]**2 * grd.GRD_bfact[k]
                            ) * dt * GRAV
                        ) * CVovRt2
                    # end g loop
                # end k loop

                # Boundary conditions
                for g in range(adm.ADM_gall_pl):
                    rhogw_pl[g, kmin, l]   *= vmtr.VMTR_RGSGAM2H_pl[g, kmin, l]
                    rhogw_pl[g, kmax+1, l] *= vmtr.VMTR_RGSGAM2H_pl[g, kmax+1, l]
                    Sall_pl[g, kmin+1] -= self.Ml_pl[g, kmin+1, l] * rhogw_pl[g, kmin, l]
                    Sall_pl[g, kmax]   -= self.Mu_pl[g, kmax, l]   * rhogw_pl[g, kmax+1, l]

                # Solve tri-diagonal matrix
                k = kmin + 1
                for g in range(adm.ADM_gall_pl):
                    beta_pl[g]     = self.Mc_pl[g, k, l]
                    rhogw_pl[g, k, l] = Sall_pl[g, k] / beta_pl[g]

                # Forward
                for k in range(kmin + 2, kmax + 1):
                    for g in range(adm.ADM_gall_pl):
                        gamma_pl[g, k] = self.Mu_pl[g, k - 1, l] / beta_pl[g]
                        beta_pl[g]     = self.Mc_pl[g, k, l] - self.Ml_pl[g, k, l] * gamma_pl[g, k]
                        rhogw_pl[g, k, l] = (Sall_pl[g, k] - self.Ml_pl[g, k, l] * rhogw_pl[g, k - 1, l]) / beta_pl[g]

                # Backward
                for k in range(kmax - 1, kmin, -1):
                    for g in range(adm.ADM_gall_pl):
                        rhogw_pl[g, k, l] -= gamma_pl[g, k + 1] * rhogw_pl[g, k + 1, l]
                        rhogw_pl[g, k + 1, l] *= vmtr.VMTR_GSGAM2H_pl[g, k + 1, l]

                # Boundary treatment
                for g in range(adm.ADM_gall_pl):
                    rhogw_pl[g, kmin, l]   *= vmtr.VMTR_GSGAM2H_pl[g, kmin, l]
                    rhogw_pl[g, kmin+1, l] *= vmtr.VMTR_GSGAM2H_pl[g, kmin+1, l]
                    rhogw_pl[g, kmax+1, l] *= vmtr.VMTR_GSGAM2H_pl[g, kmax+1, l]


            # end l loop

        prf.PROF_rapend('____vi_rhow_solver',2)
        
        return
