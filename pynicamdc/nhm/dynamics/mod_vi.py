import numpy as np
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf

class Vi:
    
    _instance = None
    
    def __init__(self):
        pass

    counter = -1

    def vi_setup(self, cnst, rdtype):

        self.Mc    = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self.Mc_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.Mu    = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self.Mu_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        self.Ml    = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        self.Ml_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        # self.Mc    = np.zeros((adm.ADM_shape),    dtype=rdtype)
        # self.Mc_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)
        # self.Mu    = np.zeros((adm.ADM_shape),    dtype=rdtype)
        # self.Mu_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)
        # self.Ml    = np.zeros((adm.ADM_shape),    dtype=rdtype)
        # self.Ml_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)

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
            dt,  
            cnst, comm, grd, oprt, vmtr, tim, rcnf, bndc, cnvv, numf, src, rdtype,                  
    ):
        
        prf.PROF_rapstart('____vi_path0',2)   

        gall_1d = adm.ADM_gall_1d
        gall_pl = adm.ADM_gall_pl
        kall = adm.ADM_kall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        lall_pl = adm.ADM_lall_pl

        # grhogetot0    = np.empty((adm.ADM_shape), dtype=rdtype)
        # grhogetot0_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # rhog_h        = np.empty((adm.ADM_shape), dtype=rdtype)
        # eth_h         = np.empty((adm.ADM_shape), dtype=rdtype)
        # rhog_h_pl     = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # eth_h_pl      = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # drhog         = np.empty((adm.ADM_shape), dtype=rdtype)
        # drhog_pl      = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # dpgrad        = np.empty((adm.ADM_shape 3,), dtype=rdtype)  # additional dimension for XDIR YDIR ZDIR
        # dpgrad_pl     = np.empty((adm.ADM_shape_pl 3,), dtype=rdtype)  # additional dimension for XDIR YDIR ZDIR
        # dpgradw       = np.empty((adm.ADM_shape), dtype=rdtype)
        # dpgradw_pl    = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # dbuoiw        = np.empty((adm.ADM_shape), dtype=rdtype)
        # dbuoiw_pl     = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # drhoge        = np.empty((adm.ADM_shape), dtype=rdtype)
        # drhoge_pl     = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # gz_tilde      = np.empty((adm.ADM_shape), dtype=rdtype)
        # gz_tilde_pl   = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # drhoge_pw     = np.empty((adm.ADM_shape), dtype=rdtype)
        # drhoge_pw_pl  = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # drhoge_pwh    = np.empty((adm.ADM_shape), dtype=rdtype)
        # drhoge_pwh_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # g_TEND        = np.empty((adm.ADM_shape 6,), dtype=rdtype)  # additional dimension for I_RHOG to I_RHOGE
        #g_TEND_pl     = np.empty((adm.ADM_shape_pl 6,), dtype=rdtype)  # additional dimension for I_RHOG to I_RHOGE
        
        grhogetot0    = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        grhogetot0_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        rhog_h        = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        eth_h         = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        rhog_h_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        eth_h_pl      = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        drhog         = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        drhog_pl      = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        dpgrad        = np.full((adm.ADM_shape + (3,)), cnst.CONST_UNDEF, dtype=rdtype)  # additional dimension for XDIR YDIR ZDIR
        dpgrad_pl     = np.full((adm.ADM_shape_pl + (3,)), cnst.CONST_UNDEF, dtype=rdtype)  # additional dimension for XDIR YDIR ZDIR
        dpgradw       = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        dpgradw_pl    = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        dbuoiw        = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        dbuoiw_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        drhoge        = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        drhoge_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        gz_tilde      = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        gz_tilde_pl   = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        drhoge_pw     = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        drhoge_pw_pl  = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        #drhoge_pw_pl  = np.full((adm.ADM_shape_pl), -9.9999E30, dtype=rdtype)
        drhoge_pwh    = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        drhoge_pwh_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        g_TEND        = np.full((adm.ADM_shape + (6,)), cnst.CONST_UNDEF, dtype=rdtype)  
        
        g_TEND_pl     = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)  # additional dimension for I_RHOG to I_RHOGE
      
        # ddivdvx       = np.empty((adm.ADM_shape), dtype=rdtype)
        # ddivdvx_pl    = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # ddivdvx_2d    = np.empty((adm.ADM_shape), dtype=rdtype)
        # ddivdvx_2d_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # ddivdvy       = np.empty((adm.ADM_shape), dtype=rdtype)
        # ddivdvy_pl    = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # ddivdvy_2d    = np.empty((adm.ADM_shape), dtype=rdtype)
        # ddivdvy_2d_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # ddivdvz       = np.empty((adm.ADM_shape), dtype=rdtype)
        # ddivdvz_pl    = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # ddivdvz_2d    = np.empty((adm.ADM_shape), dtype=rdtype)
        # ddivdvz_2d_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # ddivdw        = np.empty((adm.ADM_shape), dtype=rdtype)
        # ddivdw_pl     = np.empty((adm.ADM_shape_pl), dtype=rdtype)

        # preg_prim_split     = np.empty((adm.ADM_shape), dtype=rdtype)
        # preg_prim_split_pl  = np.empty((adm.ADM_shape_pl), dtype=rdtype)

        # drhogw        = np.empty((adm.ADM_shape), dtype=rdtype)
        # drhogw_pl     = np.empty((adm.ADM_shape_pl), dtype=rdtype)

        # drhogw        = np.empty((adm.ADM_shape), dtype=rdtype)
        # drhogw_pl     = np.empty((adm.ADM_shape_pl), dtype=rdtype)

        # diff_vh       = np.empty((adm.ADM_shape 3,), dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ
        # diff_vh_pl    = np.empty((adm.ADM_shape_pl 3,), dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ
        # diff_we       = np.empty((adm.ADM_shape 3,), dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ
        # diff_we_pl    = np.empty((adm.ADM_shape_pl 3,), dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ


        ddivdvx       = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvx_pl    = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvx_2d    = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvx_2d_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvy       = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvy_pl    = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvy_2d    = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvy_2d_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvz       = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvz_pl    = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvz_2d    = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdvz_2d_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdw        = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        ddivdw_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        preg_prim_split     = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        preg_prim_split_pl  = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        drhogw        = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        drhogw_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        drhogw        = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        drhogw_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        diff_vh       = np.full((adm.ADM_shape + (3,)), cnst.CONST_UNDEF, dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ
        diff_vh_pl    = np.full((adm.ADM_shape_pl + (3,)), cnst.CONST_UNDEF, dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ
        diff_we       = np.full((adm.ADM_shape + (3,)), cnst.CONST_UNDEF, dtype=rdtype) # additional dimension for I_RHOGVX I_RHOGVY I_RHOGVZ
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

        # for l in range(lall):
        #     for k in range(kall):
        grhogetot0[:, :, :, :] = g_TEND0[:, :, :, :, I_RHOGE]
            #end k loop
        #end l loop
        grhogetot0_pl[:, :, :] = g_TEND0_pl[:, :, :, I_RHOGE]


        # full level -> half level

        # for l in range(lall):
        for k in range(kmin, kmax + 2):  # +2 to include kmax+1
            rhog_h[:, :, k, :] = (
                vmtr.VMTR_C2Wfact[:, :, k, :, 0] * PROG[:, :, k,   :, I_RHOG] +
                vmtr.VMTR_C2Wfact[:, :, k, :, 1] * PROG[:, :, k-1, :, I_RHOG]
            )

#                with open(std.fname_log, 'a') as log_file:
#                    log_file.write(f"eth shape: {eth.shape}\n")

            eth_h[:, :, k, :] = (
                grd.GRD_afact[k] * eth[:, :, k,   :] +
                grd.GRD_bfact[k] * eth[:, :, k-1, :]
            )
            #end k loop

        rhog_h[:, :, kmin-1, :] = rhog_h[:, :, kmin, :]
        eth_h[:, :, kmin-1, :]  = eth_h[:, :, kmin, :]
        #end l loop

        if adm.ADM_have_pl:
            #for l in range(adm.ADM_lall_pl):
            # Vectorized computation for kmin to kmax+1
            rhog_h_pl[:, kmin:kmax+2, :] = (
                vmtr.VMTR_C2Wfact_pl[:, kmin:kmax+2, :, 0] * PROG_pl[:, kmin:kmax+2, :, I_RHOG] +
                vmtr.VMTR_C2Wfact_pl[:, kmin:kmax+2, :, 1] * PROG_pl[:, kmin-1:kmax+1, :, I_RHOG]
            )

            # with open (std.fname_log, 'a') as log_file:
            #     log_file.write(f"eth_h_pl shape: {eth_h_pl.shape}\n")
            #     log_file.write(f"eth_pl shape: {eth_pl.shape}\n")
            #     log_file.write(f"kimn, kmax: {kmin}, {kmax}\n")
            # prc.prc_mpistop(std.io_l, std.fname_log)

            eth_h_pl[:, kmin:kmax+2, :] = (
                grd.GRD_afact[kmin:kmax+2][None, :, None] * eth_pl[:, kmin:kmax+2, :] +   #SIZESHAPEERROR?
                grd.GRD_bfact[kmin:kmax+2][None, :, None] * eth_pl[:, kmin-1:kmax+1, :]
            )

            # Fill ghost level
            rhog_h_pl[:, kmin-1, :] = rhog_h_pl[:, kmin, :]
            eth_h_pl[:, kmin-1, :]  = eth_h_pl[:, kmin, :]
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
                cnst, grd, oprt, vmtr, rdtype, 
        )

        #---< Calculation of source term for Vh(vx,vy,vz) and W >

        #prc.prc_mpistop(std.io_l, std.fname_log)

        # with open (std.fname_log, 'a') as log_file:
        #     print("O: in vi_small_step, into numfilter_divdamp ", file=log_file)
        #     print("PROG_pl[0,2,0,:] ", PROG_pl[0,2,0,:], file=log_file)

        # with open (std.fname_log, 'a') as log_file:
        #     print("DDDDD", file=log_file)

        # divergence damping
        numf.numfilter_divdamp(
            PROG   [:,:,:,:,I_RHOGVX], PROG_pl   [:,:,:,I_RHOGVX], # [IN]
            PROG   [:,:,:,:,I_RHOGVY], PROG_pl   [:,:,:,I_RHOGVY], # [IN]
            PROG   [:,:,:,:,I_RHOGVZ], PROG_pl   [:,:,:,I_RHOGVZ], # [IN]
            PROG   [:,:,:,:,I_RHOGW],  PROG_pl   [:,:,:,I_RHOGW],  # [IN]
            ddivdvx[:,:,:,:],          ddivdvx_pl[:,:,:],          # [OUT]   # check ddivdvx_pl (e-09 instead of e-19)
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
        
        # overflow error 
        # with open(std.fname_log, 'a') as log_file:
        #     print("really?", file=log_file)
        #prc.prc_mpistop(std.io_l, std.fname_log)


        # pressure force
        src.src_pres_gradient(
            preg_prim[:,:,:,:],   preg_prim_pl[:,:,:],   # [IN]
            dpgrad   [:,:,:,:,:], dpgrad_pl   [:,:,:,:], # [OUT]   #you! undef at halo 
            dpgradw  [:,:,:,:],   dpgradw_pl  [:,:,:],   # [OUT]
            src.I_SRC_default,                           # [IN]
            cnst, grd, oprt, vmtr, rdtype,   
        )

        #prc.prc_mpistop(std.io_l, std.fname_log)


        # buoyancy force
        src.src_buoyancy(
            rhog_prim[:,:,:,:], rhog_prim_pl[:,:,:], # [IN]
            dbuoiw   [:,:,:,:], dbuoiw_pl   [:,:,:], # [OUT]    # you! pole UNDEF at kmax
            cnst, vmtr, rdtype,
        )

        # with open (std.fname_log, 'a') as log_file:
        #     print("UUUUU", file=log_file)
        #     print("dbuoiw[6, 5, kmax, :]", dbuoiw[6, 5, kmax,:], file=log_file)  # you! UNDEF at kmax
        #     print("dbuoiw_pl[:,kmax,:]", dbuoiw_pl[:,kmax,:], file=log_file)     # you! UNDEF at kmax

        #---< Calculation of source term for rhoge >

        # advection convergence for eth
        # with open (std.fname_log, 'a') as log_file:
        #     print("ENTERing src.src_advection_convergence for drhoge_pl", file=log_file)

        src.src_advection_convergence( 
            PROG  [:,:,:,:,I_RHOGVX], PROG_pl  [:,:,:,I_RHOGVX], # [IN]
            PROG  [:,:,:,:,I_RHOGVY], PROG_pl  [:,:,:,I_RHOGVY], # [IN]
            PROG  [:,:,:,:,I_RHOGVZ], PROG_pl  [:,:,:,I_RHOGVZ], # [IN]
            PROG  [:,:,:,:,I_RHOGW],  PROG_pl  [:,:,:,I_RHOGW],  # [IN]
            eth   [:,:,:,:],          eth_pl   [:,:,:],          # [IN]
            drhoge[:,:,:,:],          drhoge_pl[:,:,:],          # [OUT]   #
            src.I_SRC_default,                                 # [IN]
            cnst, grd, oprt, vmtr, rdtype,
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
                    vmtr.VMTR_W2Cfact[:, :, k, l, 0] * drhoge_pwh[:, :, k + 1, l] +
                    vmtr.VMTR_W2Cfact[:, :, k, l, 1] * drhoge_pwh[:, :, k, l]
                )
            # end k loop

            drhoge_pw[:, :, kmin - 1, l] = rdtype(0.0)
            drhoge_pw[:, :, kmax + 1, l] = rdtype(0.0)
        # end l loop

        ###
        # k=3
        # l=1
        # #print(f"eE: drhoge_pwh , j, k, l, {0}, {k}, {l},", drhoge_pwh [:,0,k,l])
        # print(f"eE: dpgrad X, j, k, l, {0}, {k}, {l},", dpgrad[:,0,k,l,XDIR])  #you!  undef
        # print(f"eE: dpgrad Y, j, k, l, {0}, {k}, {l},", dpgrad[:,0,k,l,YDIR])  #you!  undef
        # print(f"eE: dpgrad Z, j, k, l, {0}, {k}, {l},", dpgrad[:,0,k,l,ZDIR])  #you!  undef
        #print(f"eE: vy, j, k, l, {0}, {k}, {l},", vy[:,0,k,l])
        #print(f"eE: vz, j, k, l, {0}, {k}, {l},", vz[:,0,k,l])
                # if adm.ADM_have_pl:
        #     for l in range(adm.ADM_lall_pl):
        #         # --- Vectorized gz_tilde_pl and drhoge_pwh_pl
        #         gz_tilde_pl[:, :, l] = GRAV - (dpgradw_pl[:, :, l] - dbuoiw_pl[:, :, l]) / rhog_h_pl[:, :, l]
        #         drhoge_pwh_pl[:, :, l] = -gz_tilde_pl[:, :, l] * PROG_pl[:, :, l, I_RHOGW]

        #     with open (std.fname_log, 'a') as log_file:
        #         print("ZZZZZ", file=log_file)
        #         #print("gz_tilde_pl[:,kmax+1,:]", gz_tilde_pl[:,kmax+1,:], file=log_file)
        #         print("gz_tilde_pl[:,kmax,:]", gz_tilde_pl[:,kmax,:], file=log_file)       # you! 
        #         #print("gz_tilde_pl[:,kmax-1,:]", gz_tilde_pl[:,kmax-1,:], file=log_file)
        #         #print("dpgradw_pl[:,kmax,:]", dpgradw_pl[:,kmax,:], file=log_file)    
        #         print("dbuoiw_pl[:,kmax,:]", dbuoiw_pl[:,kmax,:], file=log_file)       # you! UNDEF at kmax
        #         #print("rhog_h_pl[:,kmax,:]", rhog_h_pl[:,kmax,:], file=log_file)


        #         # --- Vectorized drhoge_pw_pl over kmin to kmax
        #         drhoge_pw_pl[:, kmin:kmax+1, l] = (
        #             vx_pl[:, kmin:kmax+1, l] * dpgrad_pl[:, kmin:kmax+1, l, XDIR] +
        #             vy_pl[:, kmin:kmax+1, l] * dpgrad_pl[:, kmin:kmax+1, l, YDIR] +
        #             vz_pl[:, kmin:kmax+1, l] * dpgrad_pl[:, kmin:kmax+1, l, ZDIR] +
        #             vmtr.VMTR_W2Cfact_pl[:, kmin:kmax+1, 0, l] * drhoge_pwh_pl[:, kmin+1:kmax+2, l] +
        #             vmtr.VMTR_W2Cfact_pl[:, kmin:kmax+1, 1, l] * drhoge_pwh_pl[:, kmin:kmax+1,   l]
        #         )

        #         # --- Ghost layers at boundaries
        #         drhoge_pw_pl[:, kmin-1, l] = rdtype(0.0)
        #         drhoge_pw_pl[:, kmax+1, l] = rdtype(0.0)
        #     # end l loop
        # #endif

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



        # with open (std.fname_log, 'a') as log_file:
        #     print("BBbBB",file=log_file)
        #     #print("g_TEND_pl[0,3,0,:]", g_TEND_pl[0,3,0,:],file=log_file)   # last axis has nan in 2nd nsloop (ns=1)
        #     #print("g_TEND0_pl[0,3,0,:]", g_TEND0_pl[0,3,0,:],file=log_file)
        #     #print("drhoge_pl[0,3,0] ", drhoge_pl[0,3,0],file=log_file)
        #     print("drhoge_pw_pl[:,3,0] ",   drhoge_pw_pl[:,3,0],file=log_file)   # nan in 2nd nsloop (ns=1)
        #     print("drhoge_pwh_pl[0,3,0] ", drhoge_pwh_pl[0,3,0],file=log_file) 
        #     print("drhoge_pwh_pl[0,4,0] ", drhoge_pwh_pl[0,4,0],file=log_file) 
        #     print("dpgrad_pl[0,3,0,:] ",       dpgrad_pl[0,3,0,:],file=log_file)
        #     print("vx_pl[0,3,0] ",                 vx_pl[0,3,0],file=log_file)
        #     print("vy_pl[0,3,0] ",                 vy_pl[0,3,0],file=log_file)
        #     print("vz_pl[0,3,0] ",                 vz_pl[0,3,0],file=log_file)
        #     print("vmtr.VMTR_W2Cfact_pl[0, 3, 0, 0] ", vmtr.VMTR_W2Cfact_pl[0, 3, 0, 0],file=log_file)
        #     print("vmtr.VMTR_W2Cfact_pl[0, 3, 1, 0] ", vmtr.VMTR_W2Cfact_pl[0, 3, 1, 0],file=log_file)
        #     #print("ddivdvx_2d_pl[0,3,0] ", ddivdvx_2d_pl[0,3,0],file=log_file)         #
        #     #print("ddivdvx_pl[0,3,0] ", ddivdvx_pl[0,3,0], file=log_file)     #  ddivdvx_pl[0,3,0] too big e-09, should be about e-19




        # overflow error 
        #print("really?")
        #prc.prc_mpistop(std.io_l, std.fname_log)


        #---< sum of tendencies ( large step + pres-grad + div-damp + div-damp_2d + buoyancy ) >

        # for l in range(lall):
        #     for k in range(kall):
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
            #end k loop
        #end l loop 

        # k=3
        # l=1
        # print(f"eE: g_TEND , j, k, l, {0}, {k}, {l},", g_TEND [:,0,k,l,I_RHOGE])
        # print(f"eE: g_TEND0, j, k, l, {0}, {k}, {l},", g_TEND0[:,0,k,l,I_RHOGE])
        # print(f"eE: drhoge , j, k, l, {0}, {k}, {l},", drhoge [:,0,k,l])
        # print(f"eE: drhoge_pw , j, k, l, {0}, {k}, {l},", drhoge_pw [:,0,k,l])  # you!


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

#        print("g_TEND_pl[6,5,2,0,:]", g_TEND[6, 5, 2, 0, :])
        # with open (std.fname_log, 'a') as log_file:
        #         print("BBBBB",file=log_file)
        #         print("g_TEND_pl[0,3,0,:]", g_TEND_pl[0,3,0,:],file=log_file)   # last axis has nan in 2nd nsloop (ns=1)
        #         print("g_TEND0_pl[0,3,0,:]", g_TEND0_pl[0,3,0,:],file=log_file)
        #         print("drhoge_pl[0,3,0] ", drhoge_pl[0,3,0],file=log_file)
        #         print("drhoge_pw_pl[0,3,0] ", drhoge_pw_pl[0,3,0],file=log_file)   # nan in 2nd nsloop (ns=1)
        #         print("drhoge_pwh_pl[0,3,0] ", drhoge_pwh_pl[0,3,0],file=log_file) 
        #         print("drhoge_pwh_pl[0,4,0] ", drhoge_pwh_pl[0,4,0],file=log_file) 
        #         print("dpgrad_pl[0,3,0,:] ", dpgrad_pl[0,3,0,:],file=log_file)
        #         print("vx_pl[0,3,0] ", vx_pl[0,3,0],file=log_file)
        #         print("vy_pl[0,3,0] ", vy_pl[0,3,0],file=log_file)
        #         print("vz_pl[0,3,0] ", vz_pl[0,3,0],file=log_file)
        #         print("vmtr.VMTR_W2Cfact_pl[0, 3, 0, 0] ", vmtr.VMTR_W2Cfact_pl[0, 3, 0, 0],file=log_file)
        #         print("vmtr.VMTR_W2Cfact_pl[0, 3, 1, 0] ", vmtr.VMTR_W2Cfact_pl[0, 3, 1, 0],file=log_file)
        #         print("ddivdvx_2d_pl[0,3,0] ", ddivdvx_2d_pl[0,3,0],file=log_file)         #
        #         print("ddivdvx_pl[0,3,0] ", ddivdvx_pl[0,3,0], file=log_file)     #  ddivdvx_pl[0,3,0] too big e-09, should be about e-19

        # with open(std.fname_log, 'a') as log_file:  
        #     print("g_TEND added before smallstep iteration (6,5,2,0,:)", file=log_file) 
        #     print(g_TEND[6, 5, 2, 0, :], file=log_file) 

        # prc.prc_mpistop(std.io_l, std.fname_log)
        # initialization of mean mass flux

        rweight_itr = rdtype(1.0) / rdtype(num_of_itr)
                                # 0 :  5     + 1  # includes I_RHOG (0) to I_RHOGW (5)
        PROG_mean[:, :, :, :, I_RHOG:I_RHOGW + 1] = PROG[:, :, :, :, I_RHOG:I_RHOGW + 1]
        PROG_mean_pl[:, :, :, I_RHOG:I_RHOGW + 1] = PROG_pl[:, :, :, I_RHOG:I_RHOGW + 1]


        # update working matrix for vertical implicit solver
        self.vi_rhow_update_matrix( 
            eth_h   [:,:,:,:], eth_h_pl   [:,:,:], # [IN]
            gz_tilde[:,:,:,:], gz_tilde_pl[:,:,:], # [IN]   #you! pl at kmax 
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
        #for ns in range(num_of_itr + 1):   

            prf.PROF_rapstart('____vi_path1',2)

            # with open (std.fname_log, 'a') as log_file:
            #     print("NNNs num_of_itr ", num_of_itr, file=log_file)
            #     print("ns ", ns, file=log_file)
                
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
                #for l in range(adm.ADM_lall_pl):
                preg_prim_split_pl[:, :, :] = PROG_split_pl[:, :, :, I_RHOGE] * RovCV

                # Ghost layers copy
                preg_prim_split_pl[:, kmin - 1, :] = preg_prim_split_pl[:, kmin, :]
                preg_prim_split_pl[:, kmax + 1, :] = preg_prim_split_pl[:, kmax, :]

                PROG_split_pl[:, kmin - 1, :, I_RHOGE] = PROG_split_pl[:, kmin, :, I_RHOGE]
                PROG_split_pl[:, kmax + 1, :, I_RHOGE] = PROG_split_pl[:, kmax, :, I_RHOGE]
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
                    cnst, comm, grd, oprt, vmtr, src, rdtype,
                )

                # 2d divergence damping
                numf.numfilter_divdamp_2d(
                    PROG_split[:,:,:,:,I_RHOGVX], PROG_split_pl[:,:,:,I_RHOGVX], # [IN]
                    PROG_split[:,:,:,:,I_RHOGVY], PROG_split_pl[:,:,:,I_RHOGVY], # [IN]
                    PROG_split[:,:,:,:,I_RHOGVZ], PROG_split_pl[:,:,:,I_RHOGVZ], # [IN]
                    ddivdvx_2d[:,:,:,:],          ddivdvx_2d_pl[:,:,:],          # [OUT]
                    ddivdvy_2d[:,:,:,:],          ddivdvy_2d_pl[:,:,:],          # [OUT]
                    ddivdvz_2d[:,:,:,:],          ddivdvz_2d_pl[:,:,:],          # [OUT]
                    cnst, comm, grd, oprt, rdtype,
                )

                # pressure force
                # dpgradw=0.0_RP becaude of f_type='HORIZONTAL'.
                src.src_pres_gradient( 
                    preg_prim_split[:,:,:,:],   preg_prim_split_pl[:,:,:],   # [IN]
                    dpgrad         [:,:,:,:,:], dpgrad_pl         [:,:,:,:], # [OUT]
                    dpgradw        [:,:,:,:],   dpgradw_pl        [:,:,:],   # [OUT] not used
                    src.I_SRC_horizontal,                                     # [IN]
                    cnst, grd, oprt, vmtr, rdtype,
                )

                # buoyancy force
                # not calculated, because this term is implicit.

                #---< sum of tendencies ( large step + split{ pres-grad + div-damp + div-damp_2d } ) >

                # for l in range(lall):
                #     for k in range(kall):
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
                    #end k loop
                #end l loop

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

                # for l in range(lall):
                #     for k in range(kall):
                drhogvx = g_TEND[:, :, :, :, I_RHOGVX]
                drhogvy = g_TEND[:, :, :, :, I_RHOGVY]
                drhogvz = g_TEND[:, :, :, :, I_RHOGVZ]
                drhogw[:, :, :, :] = g_TEND[:, :, :, :, I_RHOGW]

                diff_vh[:, :, :, :, 0] = PROG_split[:, :, :, :, I_RHOGVX] + drhogvx * dt
                diff_vh[:, :, :, :, 1] = PROG_split[:, :, :, :, I_RHOGVY] + drhogvy * dt
                diff_vh[:, :, :, :, 2] = PROG_split[:, :, :, :, I_RHOGVZ] + drhogvz * dt
                    #end k loop
                #end l loop

                if adm.ADM_have_pl:
                    #for l in range(adm.ADM_lall_pl):
                        # Vectorized across g and k
                    drhogvx = g_TEND_pl[:, :, :, I_RHOGVX]
                    drhogvy = g_TEND_pl[:, :, :, I_RHOGVY]
                    drhogvz = g_TEND_pl[:, :, :, I_RHOGVZ]
                    drhogw_pl[:, :, :] = g_TEND_pl[:, :, :, I_RHOGW]

                    diff_vh_pl[:, :, :, 0] = PROG_split_pl[:, :, :, I_RHOGVX] + drhogvx * dt
                    diff_vh_pl[:, :, :, 1] = PROG_split_pl[:, :, :, I_RHOGVY] + drhogvy * dt
                    diff_vh_pl[:, :, :, 2] = PROG_split_pl[:, :, :, I_RHOGVZ] + drhogvz * dt
                    #end l loop
                #endif

            #endif    Split/Non-split

            # with open (std.fname_log, 'a') as log_file:
            #     print("diff_vh_pl[0,3,0,:] ", diff_vh_pl[0,3,0,:])
            #     print("PROG_split_pl[0,3,0,:] ", PROG_split_pl[0,3,0,:])
            #     print("g_TEND_pl[0,3,0,:]", g_TEND_pl[0,3,0,:])
#                print("drhogvx[0,3,0,:] ", drhogvx[0,3,0,:])


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
                cnst, rdtype,
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
                    cnst, rdtype,
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
            # #     print("", file=log_file)

            # with open(std.fname_log, 'a') as log_file:
            #     k=3
            #     l=1
            #     print('uho', g_TEND.shape, g_TEND_pl.shape, I_RHOGE, file=log_file)
            #     print(f"bB0: g_TEND, j, k, l, {0}, {k}, {l},", g_TEND[:,0,k,l,I_RHOGE], file=log_file)
            #     print(f"bB1: g_TEND, j, k, l, {1}, {k}, {l},", g_TEND[:,1,k,l,I_RHOGE], file=log_file)
            #     print(f"bB_pl: g_TEND_pl, k, l, {k}, {l},", g_TEND_pl[:,k,l,I_RHOGE],  file=log_file)

            #g_TEND         [:,:,:,:,I_RHOGE],  g_TEND_pl         [:,:,:,I_RHOGE]
            # with open(std.fname_log, 'a') as log_file:
            #     ic = 6
            #     jc = 5
            #     kc= 37
            #     lc= 1
            #     print("BEFOREvimain", file=log_file)  # mostly good execept for small values

            #     print(f"diff_vh[{ic}, {jc}, {kc}, {lc}, :]", diff_vh[ic, jc, kc, lc, :], file=log_file)
            #     print(f"PROG_split[{ic}, {jc}, {kc}, {lc}, :]", PROG_split[ic, jc, kc, lc, :], file=log_file)    
            #     print(f"g_TEND[{ic}, {jc}, {kc}, {lc}, :]", g_TEND[ic, jc, kc, lc, :], file=log_file)        
            #     print(f"PROG[{ic}, {jc}, {kc}, {lc}, :]", PROG[ic, jc, kc, lc, :], file=log_file)
                
            #     print(f"preg_prim_split[{ic}, {jc}, {kc}, {lc}]", preg_prim_split[ic, jc, kc, lc], file=log_file)
            #     print(f"eth[{ic}, {jc}, {kc}, {lc}]", eth[ic, jc, kc, lc], file=log_file)
            #     print(f"drhogw[{ic}, {jc}, {kc}, {lc}]", drhogw[ic, jc, kc, lc], file=log_file)
            #     print(f"grhogetot0[{ic}, {jc}, {kc}, {lc}]", grhogetot0[ic, jc, kc, lc], file=log_file)

            #     print(f"diff_vh_pl[0, {kc}, {lc}, :]", diff_vh_pl[0, kc, lc, :], file=log_file)    #unstable  0 3 0 :  2nd
            #     print(f"diff_vh_pl[1, {kc}, {lc}, :]", diff_vh_pl[1, kc, lc, :], file=log_file)   
            #     print(f"diff_vh_pl[2, {kc}, {lc}, :]", diff_vh_pl[2, kc, lc, :], file=log_file)
            #     print(f"diff_vh_pl[3, {kc}, {lc}, :]", diff_vh_pl[3, kc, lc, :], file=log_file)
            #     print(f"diff_vh_pl[4, {kc}, {lc}, :]", diff_vh_pl[4, kc, lc, :], file=log_file)
            #     print(f"diff_vh_pl[5, {kc}, {lc}, :]", diff_vh_pl[5, kc, lc, :], file=log_file)
            #     print(f"PROG_split_pl[0, {kc}, {lc}, :]", PROG_split_pl[0, kc, lc, :], file=log_file)
            #     print(f"PROG_split_pl[1, {kc}, {lc}, :]", PROG_split_pl[1, kc, lc, :], file=log_file)
            #     print(f"PROG_split_pl[2, {kc}, {lc}, :]", PROG_split_pl[2, kc, lc, :], file=log_file)
            #     print(f"PROG_split_pl[3, {kc}, {lc}, :]", PROG_split_pl[3, kc, lc, :], file=log_file)
            #     print(f"PROG_split_pl[4, {kc}, {lc}, :]", PROG_split_pl[4, kc, lc, :], file=log_file)
            #     print(f"PROG_split_pl[5, {kc}, {lc}, :]", PROG_split_pl[5, kc, lc, :], file=log_file)
            #     print(f"g_TEND_pl[0, {kc}, {lc}, :]", g_TEND_pl[0, kc, lc, :], file=log_file)       #unstable  0 3 0 :  3rd      axes 1-3
            #     print(f"g_TEND_pl[1, {kc}, {lc}, :]", g_TEND_pl[1, kc, lc, :], file=log_file)       #unstable  1 3 0 :  3rd      axes 1-3
            #     print(f"g_TEND_pl[2, {kc}, {lc}, :]", g_TEND_pl[2, kc, lc, :], file=log_file)       #unstable  2 3 0 :  3rd      axes 1-3
            #     print(f"g_TEND_pl[3, {kc}, {lc}, :]", g_TEND_pl[3, kc, lc, :], file=log_file)       #unstable  3 3 0 :  3rd
            #     print(f"g_TEND_pl[4, {kc}, {lc}, :]", g_TEND_pl[4, kc, lc, :], file=log_file)       #unstable  4 3 0 :  3rd
            #     print(f"g_TEND_pl[5, {kc}, {lc}, :]", g_TEND_pl[5, kc, lc, :], file=log_file)       #unstable  5 3 0 :  3rd

            #     print(f"preg_prim_split_pl[:, {kc}, {lc}]", preg_prim_split_pl[:, kc, lc], file=log_file)
            #     print(f"eth_pl[:, {kc}, {lc}]", eth_pl[:, kc, lc], file=log_file)
            #     print(f"drhogw_pl[:, {kc}, {lc}]", drhogw_pl[:, kc, lc], file=log_file)
            #     print(f"grhogetot0_pl[:, {kc}, {lc}]", grhogetot0_pl[:, kc, lc], file=log_file)
            #     print("", file=log_file)
            #     print("", file=log_file)

            #print("stopper")            
            #prc.prc_mpistop(std.io_l, std.fname_log)

            #---< vertical implicit scheme >
            self.vi_main(
                diff_we        [:,:,:,:,0],        diff_we_pl        [:,:,:,0],        # [OUT]    # g
                diff_we        [:,:,:,:,1],        diff_we_pl        [:,:,:,1],        # [OUT]    # gw
                diff_we        [:,:,:,:,2],        diff_we_pl        [:,:,:,2],        # [OUT]    # ge
                diff_vh        [:,:,:,:,0],        diff_vh_pl        [:,:,:,0],        # [IN]    #
                diff_vh        [:,:,:,:,1],        diff_vh_pl        [:,:,:,1],        # [IN]    #
                diff_vh        [:,:,:,:,2],        diff_vh_pl        [:,:,:,2],        # [IN]    #   pole 0 is a bit off at k 0
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
                g_TEND         [:,:,:,:,I_RHOGE],  g_TEND_pl         [:,:,:,I_RHOGE],  # [IN]   #you!?
                grhogetot0     [:,:,:,:],          grhogetot0_pl     [:,:,:],          # [IN]
                dt,                                                                    # [IN]
                rcnf, cnst, vmtr, tim, grd, oprt, bndc, cnvv, src, rdtype, 
            )

            # with open(std.fname_log, 'a') as log_file:
            #     # ic = 6
            #     # jc = 5
            #     # kc= 0
            #     # lc= 0
            #     print("AFTERvimain", file=log_file)
            #     print(f"diff_we[{ic}, {jc}, {kc}, {lc}, :]", diff_we[ic, jc, kc, lc, :], file=log_file)    #good at k 0,  2nd element off at k 41 (< e-22)  off few % at k 3, e-04
            #     print(f"diff_we_pl[0, {kc}, {lc}, :]", diff_we_pl[0, kc, lc, :], file=log_file)            #good at k 0,  2nd element off at k 41 (< e-22)  # unstable? sometimes? off by orders at k 3
            #     print(f"diff_we_pl[1, {kc}, {lc}, :]", diff_we_pl[1, kc, lc, :], file=log_file)            #### unstable? ### sometimes? off by orders at k 3, does not match 2-5, unlike original code
            #     print(f"diff_we_pl[2, {kc}, {lc}, :]", diff_we_pl[2, kc, lc, :], file=log_file)
            #     print(f"diff_we_pl[3, {kc}, {lc}, :]", diff_we_pl[3, kc, lc, :], file=log_file)
            #     print(f"diff_we_pl[4, {kc}, {lc}, :]", diff_we_pl[4, kc, lc, :], file=log_file)
            #     print(f"diff_we_pl[5, {kc}, {lc}, :]", diff_we_pl[5, kc, lc, :], file=log_file)

            # with open(std.fname_log, 'a') as log_file:  
            #     print("", file=log_file)
            #     print("check after vi_main", file=log_file) 
            #     print("diff_we", file=log_file)
            #     print(diff_we[6, 5, 2, 0, :], file=log_file) 
            #     print("", file=log_file)

            # l=1
            # k=3
            # with open(std.fname_log, 'a') as log_file:
            #     print(f"aAA, j, k, l: {0}, {k}, {l},", diff_we[:,0,k,l,2], file=log_file) 


            # treatment for boundary condition   # Halo values before this point should not be used.
            comm.COMM_data_transfer( diff_we, diff_we_pl )

            # l=1
            # k=3
            # with open(std.fname_log, 'a') as log_file:
            #     print(f"bBB, j, k, l: {0}, {k}, {l},", diff_we[:,0,k,l,2], file=log_file) 


            # update split value and mean mass flux

            # for l in range(lall):
            #     for k in range(kall):
            PROG_split[:, :, :, :, I_RHOGVX] = diff_vh[:, :, :, :, 0]
            PROG_split[:, :, :, :, I_RHOGVY] = diff_vh[:, :, :, :, 1]
            PROG_split[:, :, :, :, I_RHOGVZ] = diff_vh[:, :, :, :, 2]
            PROG_split[:, :, :, :, I_RHOG]   = diff_we[:, :, :, :, 0]
            PROG_split[:, :, :, :, I_RHOGW]  = diff_we[:, :, :, :, 1]
            PROG_split[:, :, :, :, I_RHOGE]  = diff_we[:, :, :, :, 2]
                #end k loop
            #end l loop

            #for iv in range(I_RHOG, I_RHOGW + 1):
                # for l in range(lall):
                #     for k in range(kall):
            PROG_mean[:, :, :, :, I_RHOG:I_RHOGW + 1] += PROG_split[:, :, :, :, I_RHOG:I_RHOGW + 1] * rweight_itr
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

        # for l in range(lall):
        #     for k in range(kall):
        PROG[:, :, :, :, I_RHOG:I_RHOGE + 1] += PROG_split[:, :, :, :, I_RHOG:I_RHOGE + 1]
            #end k loop
        #end l loop

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
        kall = adm.ADM_kall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        lall_pl = adm.ADM_lall_pl

        # Mc     = np.empty((adm.ADM_shape), dtype=rdtype)
        # Mu     = np.empty((adm.ADM_shape), dtype=rdtype)
        # Ml     = np.empty((adm.ADM_shape), dtype=rdtype)
        # Mc_pl  = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # Mu_pl  = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # Ml_pl  = np.empty((adm.ADM_shape_pl), dtype=rdtype)
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
            for k in range(kmin + 1, kmax + 1):  # include kmax
                # Vectorized over g
                Mc_pl[:, k, :] = (
                    ACVovRt2 / vmtr.VMTR_RGSQRTH_pl[:, k, :] +
                    grd.GRD_rdgzh[k] * (
                        (vmtr.VMTR_RGSGAM2_pl[:, k, :] * grd.GRD_rdgz[k] +
                        vmtr.VMTR_RGSGAM2_pl[:, k - 1, :] * grd.GRD_rdgz[k - 1]) *
                        vmtr.VMTR_GAM2H_pl[:, k, :] * eth_pl[:, k, :] -
                        (grd.GRD_dfact[k] - grd.GRD_cfact[k - 1]) *
                        (g_tilde_pl[:, k, :] + GCVovR)
                    )
                )

                Mu_pl[:, k, :] = -grd.GRD_rdgzh[k] * (
                    vmtr.VMTR_RGSGAM2_pl[:, k, :] * grd.GRD_rdgz[k] *
                    vmtr.VMTR_GAM2H_pl[:, k + 1, :] * eth_pl[:, k + 1, :] +
                    grd.GRD_cfact[k] * (
                        g_tilde_pl[:, k + 1, :] +
                        vmtr.VMTR_GAM2H_pl[:, k + 1, :] * vmtr.VMTR_RGAMH_pl[:, k, :] ** 2 * GCVovR
                    )
                )


                # if k==kmax:
                # # if k==kmax-1 and l == 0:
                #     with open(std.fname_log, 'a') as log_file:
                #         print("YYYYY", self.counter, file=log_file)
                #         print("Mu_pl", file=log_file)
                #         print(Mu_pl[:, k, 0], file=log_file)
                #         print("vmtr.VMTR_GAM2H_pl[:, k + 1, 0]", vmtr.VMTR_GAM2H_pl[:, k + 1, 0], file=log_file)
                #         print("g_tilde_pl[:, k + 1, 0]", g_tilde_pl[:, k + 1, 0], file=log_file)         # you!  at kmax 
                #         print("eth_pl[:, k + 1, 0]", eth_pl[:, k + 1, 0], file=log_file)

                        # print("vmtr.VMTR_RGAMH_pl[:, k, l]", vmtr.VMTR_RGAMH_pl[:, k, l], file=log_file)
                        # print("GCVovR", GCVovR, file=log_file)
                        # print("grd.GRD_cfact[k]", grd.GRD_cfact[k], file=log_file)
                        # print("grd.GRD_rdgzh[k]", grd.GRD_rdgzh[k], file=log_file)
                        # print("grd.GRD_rdgz[k]", grd.GRD_rdgz[k], file=log_file)
                # if k==kmax-1 and l == 0:
                #     with open(std.fname_log, 'a') as log_file:
                #         print("Mu_pl", file=log_file)
                #         print(Mu_pl[:, k, l], file=log_file)
                #         print("vmtr.VMTR_GAM2H_pl[:, k + 1, l]", vmtr.VMTR_GAM2H_pl[:, k + 1, l], file=log_file)
                #         print("g_tilde_pl[:, k + 1, l]", g_tilde_pl[:, k + 1, l], file=log_file)
                #         print("eth_pl[:, k + 1, l]", eth_pl[:, k + 1, l], file=log_file)
                    


                Ml_pl[:, k, :] = -grd.GRD_rdgzh[k] * (
                    vmtr.VMTR_RGSGAM2_pl[:, k, :] * grd.GRD_rdgz[k] *
                    vmtr.VMTR_GAM2H_pl[:, k - 1, :] * eth_pl[:, k - 1, :] -
                    grd.GRD_dfact[k - 1] * (
                        g_tilde_pl[:, k - 1, :] +
                        vmtr.VMTR_GAM2H_pl[:, k - 1, :] * vmtr.VMTR_RGAMH_pl[:, k, :] ** 2 * GCVovR
                    )
                )
            # end k loop



            # for l in range(adm.ADM_lall_pl):
            #     for k in range(kmin + 1, kmax + 1):  # include kmax
            #         # Vectorized over g
            #         Mc_pl[:, k, l] = (
            #             ACVovRt2 / vmtr.VMTR_RGSQRTH_pl[:, k, l] +
            #             grd.GRD_rdgzh[k] * (
            #                 (vmtr.VMTR_RGSGAM2_pl[:, k, l] * grd.GRD_rdgz[k] +
            #                 vmtr.VMTR_RGSGAM2_pl[:, k - 1, l] * grd.GRD_rdgz[k - 1]) *
            #                 vmtr.VMTR_GAM2H_pl[:, k, l] * eth_pl[:, k, l] -
            #                 (grd.GRD_dfact[k] - grd.GRD_cfact[k - 1]) *
            #                 (g_tilde_pl[:, k, l] + GCVovR)
            #             )
            #         )

            #         Mu_pl[:, k, l] = -grd.GRD_rdgzh[k] * (
            #             vmtr.VMTR_RGSGAM2_pl[:, k, l] * grd.GRD_rdgz[k] *
            #             vmtr.VMTR_GAM2H_pl[:, k + 1, l] * eth_pl[:, k + 1, l] +
            #             grd.GRD_cfact[k] * (
            #                 g_tilde_pl[:, k + 1, l] +
            #                 vmtr.VMTR_GAM2H_pl[:, k + 1, l] * vmtr.VMTR_RGAMH_pl[:, k, l] ** 2 * GCVovR
            #             )
            #         )

            #         if k==kmax and l == 0:
            #         # if k==kmax-1 and l == 0:
            #             with open(std.fname_log, 'a') as log_file:
            #                 print("YYYYY", self.counter, file=log_file)
            #                 print("Mu_pl", file=log_file)
            #                 print(Mu_pl[:, k, l], file=log_file)
            #                 print("vmtr.VMTR_GAM2H_pl[:, k + 1, l]", vmtr.VMTR_GAM2H_pl[:, k + 1, l], file=log_file)
            #                 print("g_tilde_pl[:, k + 1, l]", g_tilde_pl[:, k + 1, l], file=log_file)         # you!  at kmax 
            #                 print("eth_pl[:, k + 1, l]", eth_pl[:, k + 1, l], file=log_file)
            #                 # print("vmtr.VMTR_RGAMH_pl[:, k, l]", vmtr.VMTR_RGAMH_pl[:, k, l], file=log_file)
            #                 # print("GCVovR", GCVovR, file=log_file)
            #                 # print("grd.GRD_cfact[k]", grd.GRD_cfact[k], file=log_file)
            #                 # print("grd.GRD_rdgzh[k]", grd.GRD_rdgzh[k], file=log_file)
            #                 # print("grd.GRD_rdgz[k]", grd.GRD_rdgz[k], file=log_file)
            #         # if k==kmax-1 and l == 0:
            #         #     with open(std.fname_log, 'a') as log_file:
            #         #         print("Mu_pl", file=log_file)
            #         #         print(Mu_pl[:, k, l], file=log_file)
            #         #         print("vmtr.VMTR_GAM2H_pl[:, k + 1, l]", vmtr.VMTR_GAM2H_pl[:, k + 1, l], file=log_file)
            #         #         print("g_tilde_pl[:, k + 1, l]", g_tilde_pl[:, k + 1, l], file=log_file)
            #         #         print("eth_pl[:, k + 1, l]", eth_pl[:, k + 1, l], file=log_file)
                        


            #         Ml_pl[:, k, l] = -grd.GRD_rdgzh[k] * (
            #             vmtr.VMTR_RGSGAM2_pl[:, k, l] * grd.GRD_rdgz[k] *
            #             vmtr.VMTR_GAM2H_pl[:, k - 1, l] * eth_pl[:, k - 1, l] -
            #             grd.GRD_dfact[k - 1] * (
            #                 g_tilde_pl[:, k - 1, l] +
            #                 vmtr.VMTR_GAM2H_pl[:, k - 1, l] * vmtr.VMTR_RGAMH_pl[:, k, l] ** 2 * GCVovR
            #             )
            #         )
            #     # end k loop
            # #end l loop
        #endif 


        # self.Mc     = Mc.copy()
        # self.Mu     = Mu.copy()
        # self.Ml     = Ml.copy()
        # self.Mc_pl  = Mc_pl.copy()
        # self.Mu_pl  = Mu_pl.copy()
        # self.Ml_pl  = Ml_pl.copy()  


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
        grhoge,           grhoge_pl,           # you?!
        grhogetot,        grhogetot_pl,        
        dt,                    
        rcnf, cnst, vmtr, tim, grd, oprt, bndc, cnvv, src, rdtype,           
    ):
        
        gall_1d = adm.ADM_gall_1d
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        lall_pl = adm.ADM_lall_pl

        drhog         = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)   # source term at t=n+1
        drhog_pl      = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  
        drhoge        = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  
        drhoge_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  
        drhogetot     = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  
        drhogetot_pl  = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  

        grhog1         = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  # source term ( large step + t=n+1 )
        grhog1_pl      = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  
        grhoge1        = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  
        grhoge1_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  
        gpre           = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  
        gpre_pl        = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  

        rhog1          = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  # prognostic vars ( previous + t=n,t=n+1 )
        rhog1_pl       = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  
        rhogvx1        = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  
        rhogvx1_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  
        rhogvy1        = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  
        rhogvy1_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  
        rhogvz1        = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  
        rhogvz1_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  
        rhogw1         = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  
        rhogw1_pl      = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  

        rhogkin0       = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  # kinetic energy ( previous                )
        rhogkin0_pl    = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  
        rhogkin10      = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  # kinetic energy ( previous + split(t=n)   )
        rhogkin10_pl   = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  
        rhogkin11      = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  # kinetic energy ( previous + split(t=n+1) )
        rhogkin11_pl   = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)  
        ethtot0        = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)  # total enthalpy ( h + v^{2}/2 + phi, previous )
        ethtot0_pl     = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype) 

        Rdry  = cnst.CONST_Rdry
        CVdry = cnst.CONST_CVdry

        #print("pstop")
        #prc.prc_mpistop(std.io_l, std.fname_log)

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
                drhog,         drhog_pl,         # [OUT]  Broken 37
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
                drhoge,        drhoge_pl,        # [OUT]   Broken 37
                src.I_SRC_horizontal,            # [IN]
                cnst, grd, oprt, vmtr, rdtype,
            ) 

        else:

            # for l in range(lall):
            #     for k in range(kall):
            drhog[:, :, :, :] = rdtype(0.0)
            drhoge[:, :, :, :] = rdtype(0.0)

            drhog_pl[:, :, :] = rdtype(0.0)
            drhoge_pl[:, :, :] = rdtype(0.0)

        #endif

        # update grhog, grhoge and calc source term of pressure

        # for l in range(lall):
        #     for k in range(kall):
        grhog1[:, :, :, :]  = grhog[:, :, :, :]  + drhog[:, :, :, :]
        grhoge1[:, :, :, :] = grhoge[:, :, :, :] + drhoge[:, :, :, :]
        gpre[:, :, :, :]    = grhoge1[:, :, :, :] * Rdry / CVdry
            # end k loop
        # end l loop


        # #j=0
        # k=3
        # l=1
        # print(f"bB0: gpre, j, k, l, {0}, {k}, {l},", gpre[:,0,k,l])
        # print(f"bB1: gpre, j, k, l, {1}, {k}, {l},", gpre[:,1,k,l])
        # print(f"bB0: grhoge, j, k, l, {0}, {k}, {l},", grhoge[:,0,k,l]) # you!
        # print(f"bB0: drhoge, j, k, l, {0}, {k}, {l},", drhoge[:,0,k,l])

        if adm.ADM_have_pl:
            grhog1_pl  = grhog_pl  + drhog_pl      #####CHECK3637
            grhoge1_pl = grhoge_pl + drhoge_pl     #####CHECK3637
            gpre_pl    = grhoge1_pl * Rdry / CVdry
        #endif

        # with open(std.fname_log, 'a') as log_file:
        #     print("C3637", file=log_file)
        #     print("grhog_pl", grhog_pl[:, 36, 0], grhog_pl[:, 37, 0], file=log_file)
        #     print("drhog_pl", drhog_pl[:, 36, 0], drhog_pl[:, 37, 0], file=log_file)      # Broken 37
        #     print("grhoge_pl", grhoge_pl[:, 36, 0], grhoge_pl[:, 37, 0], file=log_file)
        #     print("drhoge_pl", drhoge_pl[:, 36, 0], drhoge_pl[:, 37, 0], file=log_file)   # Broken 37


        #---------------------------------------------------------------------------
        # vertical implict calculation core
        #---------------------------------------------------------------------------

        # boundary condition for rhogw_split1

        #for l in range(lall):
        #    for k in range(kall):
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


        #for l in range(lall):
        bndc.BNDCND_rhow(
            rhogvx_split1 [:,:,:,:],       # [IN]
            rhogvy_split1 [:,:,:,:],       # [IN]
            rhogvz_split1 [:,:,:,:],       # [IN]
            rhogw_split1  [:,:,:,:],       # [INOUT]
            vmtr.VMTR_C2WfactGz[:,:,:,:,:], # [IN]
            rdtype,
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
            rhogw_split1_pl[:,:,:] = rdtype(0.0)        # Tracing start from here
            
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
            #for l in range(adm.ADM_lall_pl):
                #$$ rxpl1=np.full((gall_pl, kall), cnst.CONST_UNDEF, dtype=rdtype)
                #$$ rxpl1[:,:]=rhogvx_split1_pl[:,:,l]
            bndc.BNDCND_rhow_pl(
                rhogvx_split1_pl [:,:,:],     # [IN]
                rhogvy_split1_pl [:,:,:],     # [IN]
                rhogvz_split1_pl [:,:,:],     # [IN]
                rhogw_split1_pl  [:,:,:],     # [INOUT]      # Here?
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

        # j=0
        # k=3
        # l=1
        # print(f"cC, j, k, l, {j}, {k}, {l},", rhogw_split1[:,j,k,l])
        
        # with open(std.fname_log, 'a') as log_file:
        #     print("", file=log_file)
        #     print("rhogw_split1_pl after vi_rhow_solver", file=log_file)
        #     print(rhogw_split1_pl[:, 0, 0], file=log_file)
        #     print(rhogw_split1_pl[:, 3, 0], file=log_file)    # e+27!
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

        # check why split1 is different from the original code.

        # for l in range(lall):
        #     for k in range(kall):
        rhog_split1[:, :, :, :] = rhog_split0[:, :, :, :] + (grhog[:, :, :, :] + drhog[:, :, :, :]) * dt



        if adm.ADM_have_pl:
            rhog_split1_pl[:, :, :] = rhog_split0_pl[:, :, :] + (grhog_pl[:, :, :] + drhog_pl[:, :, :]) * dt
        #endif

#         with open(std.fname_log, 'a') as log_file:
#             print("", file=log_file)
#             print("rhog_split1_pl before Satoh2002", file=log_file)
# #            print("rhog_split1", file=log_file)
#             print(rhog_split1_pl[:, 39, 0], file=log_file)               #
#             print(rhog_split0_pl[:, 39, 0], file=log_file)
#             print(grhog_pl[:, 39, 0], file=log_file)
#             print(drhog_pl[:, 39, 0], file=log_file)         # element zero of axis 0 may have an issue
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

        # for l in range(lall):
        #     for k in range(kall):
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

        # for l in range(lall):  #VectorizeFurther
        #     for k in range(kall):
        rhog1[:, :, :, :]   = rhog0[:, :, :, :]   + rhog_split1[:, :, :, :]
        rhogvx1[:, :, :, :] = rhogvx0[:, :, :, :] + rhogvx_split1[:, :, :, :]
        rhogvy1[:, :, :, :] = rhogvy0[:, :, :, :] + rhogvy_split1[:, :, :, :]
        rhogvz1[:, :, :, :] = rhogvz0[:, :, :, :] + rhogvz_split1[:, :, :, :]
        rhogw1[:, :, :, :]  = rhogw0[:, :, :, :]  + rhogw_split1[:, :, :, :]  # issue

        if adm.ADM_have_pl:
            rhog1_pl[:, :, :]   = rhog0_pl[:, :, :]   + rhog_split1_pl[:, :, :]       # big issue
            rhogvx1_pl[:, :, :] = rhogvx0_pl[:, :, :] + rhogvx_split1_pl[:, :, :]     # 0,2,0  issue
            rhogvy1_pl[:, :, :] = rhogvy0_pl[:, :, :] + rhogvy_split1_pl[:, :, :]     # 0,2,0
            rhogvz1_pl[:, :, :] = rhogvz0_pl[:, :, :] + rhogvz_split1_pl[:, :, :]     # 0,2,0 
            rhogw1_pl[:, :, :]  = rhogw0_pl[:, :, :]  + rhogw_split1_pl[:, :, :]      # 0,2,0   2,2,0  issue

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

        #for l in range(lall):
        #    for k in range(kall):
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

        # for l in range(lall):
        #     for k in range(kall):
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

        gall_1d = adm.ADM_gall_1d
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        gall_pl = adm.ADM_gall_pl
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax

        Sall     = np.full((adm.ADM_shape[:3]), cnst.CONST_UNDEF, dtype=rdtype)  
        Sall_pl  = np.full((adm.ADM_shape_pl[:2]), cnst.CONST_UNDEF, dtype=rdtype)  
        beta     = np.full((adm.ADM_shape[:2]), cnst.CONST_UNDEF, dtype=rdtype)  
        beta_pl  = np.full((adm.ADM_shape_pl[:1]), cnst.CONST_UNDEF, dtype=rdtype)  
        gamma    = np.full((adm.ADM_shape[:3]), cnst.CONST_UNDEF, dtype=rdtype)  
        gamma_pl = np.full((adm.ADM_shape_pl[:2]), cnst.CONST_UNDEF, dtype=rdtype)  

        # Sall     = np.empty((adm.ADM_shape[:3]), dtype=rdtype)  
        # Sall_pl  = np.empty((adm.ADM_shape[:2]), dtype=rdtype)  
        # beta     = np.empty((gall_1d, gall_1d,), dtype=rdtype)  
        # beta_pl  = np.empty((gall_pl,         ), dtype=rdtype)  
        # gamma    = np.empty((adm.ADM_shape[:3]), dtype=rdtype)  
        # gamma_pl = np.empty((adm.ADM_shape[:2]), dtype=rdtype)  

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
                       

            # if l==1:
            #     k=3
            #     with open(std.fname_log, 'a') as log_file:
            #         print(f"preg0, j, k, l, {0}, {k}, {l},", preg0[:,0,k,l],file=log_file)
            #         print(f"preg1, j, k, l, {1}, {k}, {l},", preg0[:,1,k,l],file=log_file)
            #         print(f"Spre0, j, {0},", Spre[:,0,k,l],file=log_file)  # you!!
            #         print(f"Spre1, j, {1},", Spre[:,1,k,l],file=log_file)  
            #         #print(f"Sall0, j, k, {0}, {k},", Srho[:,0,k],file=log_file)
            #         #print(f"Sall1, j, k, {1}, {k},", Srho[:,1,k],file=log_file)

            #         print(f"Sall0, j, k, {0}, {k},", Sall[:,0,k],file=log_file)
            #         print(f"Sall1, j, k, {1}, {k},", Sall[:,1,k],file=log_file)    

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

                # with open (std.fname_log, 'a') as log_file:
                #     print("WEIRDSEARCH3637", file=log_file)
                #     # print(rhogw0_pl[:, 36, l], file=log_file)
                #     # print(rhogw0_pl[:, 37, l], file=log_file)
                #     # print(Sw_pl[:, 36, l], file=log_file)
                #     # print(Sw_pl[:, 37, l], file=log_file)
                #     # print(preg0_pl[:, 36, l], file=log_file)
                #     # print(preg0_pl[:, 37, l], file=log_file)
                #     print(Spre_pl[:, 36, l], file=log_file)   ##
                #     print(Spre_pl[:, 37, l], file=log_file)   ##  gets weird
                #     # print(rhog0_pl[:, 36, l], file=log_file)
                #     # print(rhog0_pl[:, 37, l], file=log_file)
                #     # print(Srho_pl[:, 36, l], file=log_file)
                #     # print(Srho_pl[:, 37, l], file=log_file)
                #     # print(grd.GRD_rdgzh[36], file=log_file)
                #     # print(grd.GRD_rdgzh[37], file=log_file)
                #     # print(grd.GRD_afact[36], file=log_file)
                #     # print(grd.GRD_afact[37], file=log_file)
                #     # print(grd.GRD_bfact[36], file=log_file)
                #     # print(grd.GRD_bfact[37], file=log_file)

                # with open (std.fname_log, 'a') as log_file:
                #     print("Sall_pl36-40", file=log_file)
                #     #print(Sall_pl[:, 3], file=log_file)
                #     #print(Sall_pl[:, 4], file=log_file)
                #     print(Sall_pl[:, 36], file=log_file)
                #     print(Sall_pl[:, 37], file=log_file)
                #     print(Sall_pl[:, 38], file=log_file)
                #     print(Sall_pl[:, 39], file=log_file)
                #     print(Sall_pl[:, 40], file=log_file)


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

                # with open (std.fname_log, 'a') as log_file:    
                #     print("check kmin-1, kmin, kmin+1, kmax+1", kmin-1, kmin, kmin+1, kmax+1, file=log_file)
                #     print('beta_pl[:]', beta_pl[:], file=log_file)
                #     print('rhogw_pl[:,kmin-1,l]', rhogw_pl[:, kmin-1, l], file=log_file)
                #     print('Sall_pl[:,kmin-1]', Sall_pl[:, kmin-1], file=log_file)
                #     print('rhogw_pl[:,kmin,l]', rhogw_pl[:, kmin, l], file=log_file)
                #     print('Sall_pl[:,kmin]', Sall_pl[:, kmin], file=log_file)
                #     print('rhogw_pl[:,kmin+1,l]', rhogw_pl[:, kmin+1, l], file=log_file)
                #     print('Sall_pl[:,kmin+1,l]', Sall_pl[:, kmin+1], file=log_file)
                #     print('rhogw_pl[:,kmax,l]', rhogw_pl[:, kmax, l], file=log_file)
                #     print('Sall_pl[:,kmax,l]', Sall_pl[:, kmax], file=log_file)
                #     print("self.Mu_pl[:, kmax, l]", self.Mu_pl[:, kmax, l], file=log_file)
                #     print('rhogw_pl[:,kmax+1,l]', rhogw_pl[:, kmax+1, l], file=log_file)
                #     print('Sall_pl[:,kmax+1,l]', Sall_pl[:, kmax+1], file=log_file)

                # Forward
                for k in range(kmin + 2, kmax + 1):
                    for g in range(adm.ADM_gall_pl):
                        gamma_pl[g, k] = self.Mu_pl[g, k - 1, l] / beta_pl[g]    # 0th axis of Mu_pl and Mc_pl is nan at counter =2
                        beta_pl[g]     = self.Mc_pl[g, k, l] - self.Ml_pl[g, k, l] * gamma_pl[g, k]
                        rhogw_pl[g, k, l] = (Sall_pl[g, k] - self.Ml_pl[g, k, l] * rhogw_pl[g, k - 1, l]) / beta_pl[g]
                
                    # if k == 3 and l == 0:
                    #if k == kmax and l == 0:
                #     if k > 35 and l == 0:    # k=37 to 40 invalid Sall_pl
                #     #if k == kmin+2 and l == 0:    
                #         with open (std.fname_log, 'a') as log_file:                            
                #             print("HEREISWHEREITGETSWEIRD, k= ", k,  file=log_file)
                #             print('rhogw_pl[:, k, 0]', rhogw_pl[:, k, l], file=log_file)
                #             print('rhogw_pl[:, k-1, 0]', rhogw_pl[:, k-1, l], file=log_file)
                #             print('Sall_pl[:, k]', Sall_pl[:, k],file=log_file)
                #             print('beta_pl[:]', beta_pl[:], file=log_file)
                #             print('gamma_pl[:, k]', gamma_pl[:, k], file=log_file)
                #             # print('self.Ml_pl[g, k, l]', self.Ml_pl[g, k, l])
                #             print('self.Mu_pl[:, k - 1, l]', self.Mu_pl[:, k - 1, l],file=log_file)    ### you!  kmax -1 
                #             print('self.Mc_pl[:, k, l]', self.Mc_pl[:, k, l],file=log_file)
                #             print('self.Ml_pl[:, k, l]', self.Ml_pl[:, k, l],file=log_file) 
                #             #print('self.Mc_pl[:, k, l]', self.Mc_pl[:, k-1, l],file=log_file)
                #             #print('self.Ml_pl[:, k, l]', self.Ml_pl[:, k-1, l],file=log_file)
                #             #print('self.Mu_pl[:, k, l]', self.Mu_pl[:, k, l],file=log_file)  

                # if l == 0:
                #    with open (std.fname_log, 'a') as log_file:
                #         print('rhogw_pl[:,kmax,l]', rhogw_pl[:,kmax,l],file=log_file)
                #         print('gamma_pl[:,kmax]', gamma_pl[:,kmax],file=log_file)      ### you!

                # Backward
                for k in range(kmax - 1, kmin, -1):     # check range!!!!
                    for g in range(adm.ADM_gall_pl):
                        rhogw_pl[g, k, l] -= gamma_pl[g, k + 1] * rhogw_pl[g, k + 1, l]
                        rhogw_pl[g, k + 1, l] *= vmtr.VMTR_GSGAM2H_pl[g, k + 1, l]

                    # if k == 3 and l == 0:
                    #     with open (std.fname_log, 'a') as log_file:
                    #         #print('Sall_pl[:, k]', Sall_pl[:, k],file=log_file)
                    #         print('rhogw_pl[:, k, 0]', rhogw_pl[:, k, l], file=log_file)
                    #         #print('rhogw_pl[:, k+1, 0]', rhogw_pl[:, k+1, l], file=log_file)
                    #         #print('beta_pl[g]', beta_pl[:], file=log_file)
                    #         print('gamma_pl[g, k]', gamma_pl[:, k], file=log_file)
                    #         print('vmtr.VMTR_GSGAM2H_pl[g, k + 1, l]', vmtr.VMTR_GSGAM2H_pl[g, k + 1, l],file=log_file)

                # Boundary treatment
                for g in range(adm.ADM_gall_pl):
                    rhogw_pl[g, kmin, l]   *= vmtr.VMTR_GSGAM2H_pl[g, kmin, l]
                    rhogw_pl[g, kmin+1, l] *= vmtr.VMTR_GSGAM2H_pl[g, kmin+1, l]
                    rhogw_pl[g, kmax+1, l] *= vmtr.VMTR_GSGAM2H_pl[g, kmax+1, l]


            # end l loop

        prf.PROF_rapend('____vi_rhow_solver',2)
        
        return
