import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf


class Src:
    
    _instance = None
    
    I_SRC_horizontal = 1
    I_SRC_vertical   = 2
    I_SRC_default    = 3

    # I_SRC_default    : horizontal & vertical convergence
    # I_SRC_horizontal : horizontal convergence

    first_layer_remedy = True

    def __init__(self,rdtype):

        self.vvx  = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype)
        self.vvy  = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype)
        self.vvz  = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype)
        self.dvvx = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype)
        self.dvvy = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype)
        self.dvvz = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype)
        self.vvx_pl  = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)
        self.vvy_pl  = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)
        self.vvz_pl  = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)
        self.dvvx_pl = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)
        self.dvvy_pl = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)
        self.dvvz_pl = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)
        self.rhogvxscl = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype)
        self.rhogvyscl = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype)
        self.rhogvzscl = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype)
        self.rhogwscl  = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype)
        self.rhogvxscl_pl = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)
        self.rhogvyscl_pl = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)
        self.rhogvzscl_pl = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)
        self.rhogwscl_pl  = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)
        self.rhogvx_vm = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype) #rho*vx / vertical metrics
        self.rhogvy_vm = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype) #rho*vy / vertical metrics
        self.rhogvz_vm = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype) #rho*vz / vertical metrics  
        self.rhogvx_vm_pl = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)  #rho*vx / vertical metrics  
        self.rhogvy_vm_pl = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)  #rho*vy / vertical metrics
        self.rhogvz_vm_pl = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)  #rho*vz / vertical metrics      
        self.rhogw_vmh  = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype) #rho*w / vertical metrics 
        self.rhogw_vmh_pl  = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)  #rho*w / vertical metrics
        self.div_rhogvh = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,), dtype=rdtype) #horizontal convergence
        self.div_rhogvh_pl = np.empty((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl,), dtype=rdtype)  

    def src_advection_convergence_momentum(self, 
                vx,     vx_pl,      
                vy,     vy_pl,      
                vz,     vz_pl,
                w,      w_pl,
                rhog,   rhog_pl,    
                rhogvx, rhogvx_pl,  
                rhogvy,  rhogvy_pl,  
                rhogvz,  rhogvz_pl,  
                rhogw,   rhogw_pl,   
                grhogvx, grhogvx_pl, 
                grhogvy, grhogvy_pl,
                grhogvz, grhogvz_pl,
                grhogw,  grhogw_pl,
                rcnf, cnst, grd, oprt, vmtr, rdtype,
    ):

        prf.PROF_rapstart('____src_advection_conv_m',2)

        gall = adm.ADM_gall
        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        kminm1 = kmin - 1
        kminp1 = kmin + 1
        kmaxp1 = kmax + 1
        kmaxp2 = kmax + 2


        f = rcnf.CORIOLIS_PARAM  # used only for on-plane
        ohm = cnst.CONST_OHM
        rscale = grd.GRD_rscale
        alpha  = rdtype(rcnf.NON_HYDRO_ALPHA)
        XDIR = grd.GRD_XDIR       
        YDIR = grd.GRD_YDIR     
        ZDIR = grd.GRD_ZDIR  

        #---< merge horizontal velocity & vertical velocity >

        if grd.GRD_grid_type == grd.GRD_grid_type_on_plane:

            print("on plane not tested yet!")

            self.vvx[:, :, kmin:kmaxp1, :] = vx[:, :, kmin:kmaxp1, :]
            self.vvy[:, :, kmin:kmaxp1, :] = vy[:, :, kmin:kmaxp1, :]

            # Prepare GRD factors (shape: (kmaxp1 - kmin, 1, 1, 1))
            cfact = grd.GRD_cfact[kmin:kmaxp1][None, None, :, None]
            dfact = grd.GRD_dfact[kmin:kmaxp1][None, None, :, None]

            # Vectorized vvz computation
            self.vvz[:, :, kmin:kmaxp1, :] = (
                cfact * w[:, :, kminp1:kmaxp2, :] +
                dfact * w[:, :, kmin:kmaxp1,     :]
            )

            # Boundary layers
            self.vvx[:, :, kminm1, :] = 0.0
            self.vvx[:, :, kmaxp1, :] = 0.0
            self.vvy[:, :, kminm1, :] = 0.0
            self.vvy[:, :, kmaxp1, :] = 0.0
            self.vvz[:, :, kminm1, :] = 0.0
            self.vvz[:, :, kmaxp1, :] = 0.0

        else:

            # print("kmin,kmax: " , kmin, kmax)
            # prc.prc_mpistop(std.io_l, std.fname_log)


            # Reshape cfact/dfact to broadcast over (i, j, l)
            cfact = grd.GRD_cfact[kmin:kmaxp1][None, None, :, None]  # shape (k, 1, 1, 1)
            dfact = grd.GRD_dfact[kmin:kmaxp1][None, None, :, None]

            # wc = GRD_cfact * w[k+1] + GRD_dfact * w[k]
            wc = (
                cfact * w[:, :, kminp1:kmaxp2, :] +
                dfact * w[:, :, kmin:kmaxp1,     :]
            )  # shape: (i, j, k, l)

            # Prepare GRD_x directional components
            gx = grd.GRD_x[:, :, 0, :, XDIR]  # shape: (i, j, l)
            gy = grd.GRD_x[:, :, 0, :, YDIR]
            gz = grd.GRD_x[:, :, 0, :, ZDIR]

            # Broadcast GRD_x to shape (i, j, k, l)
            gx = gx[:, :, None, :]  # (i, j, 1, l)
            gy = gy[:, :, None, :]
            gz = gz[:, :, None, :]

            # Apply full vectorized updates
            self.vvx[:, :, kmin:kmaxp1, :] = vx[:, :, kmin:kmaxp1, :] + wc * gx / rscale
            self.vvy[:, :, kmin:kmaxp1, :] = vy[:, :, kmin:kmaxp1, :] + wc * gy / rscale
            self.vvz[:, :, kmin:kmaxp1, :] = vz[:, :, kmin:kmaxp1, :] + wc * gz / rscale

            # Set ghost layers to zero
            self.vvx[:, :, kminm1, :] = 0.0
            self.vvx[:, :, kmaxp1, :] = 0.0
            self.vvy[:, :, kminm1, :] = 0.0
            self.vvy[:, :, kmaxp1, :] = 0.0
            self.vvz[:, :, kminm1, :] = 0.0
            self.vvz[:, :, kmaxp1, :] = 0.0

        #endif

        if adm.ADM_have_pl:

            # Allocate temporary buffer
            wc = np.empty((adm.ADM_gall_pl, kmaxp1 - kmin, adm.ADM_lall_pl), dtype=w_pl.dtype)

            # GRD_cfact and GRD_dfact reshaped for broadcasting
            cfact = grd.GRD_cfact[kmin:kmaxp1][None, :, None]  # shape: (k, 1, 1)
            dfact = grd.GRD_dfact[kmin:kmaxp1][None, :, None]

            # Compute wc = GRD_cfact * w[k+1] + GRD_dfact * w[k]
            wc[:] = cfact * w_pl[:, kminp1:kmaxp2, :] + dfact * w_pl[:, kmin:kmaxp1, :]

            # Get GRD_x_pl(g, 1, l, DIRECTION) as GRD_x_pl[:, 0, :, DIR]
            gx = grd.GRD_x_pl[:, 0, :, XDIR]  # (g, l)
            gy = grd.GRD_x_pl[:, 0, :, YDIR]
            gz = grd.GRD_x_pl[:, 0, :, ZDIR]

            # Broadcast to shape (g, 1, l)
            gx = gx[:, None, :]
            gy = gy[:, None, :]
            gz = gz[:, None, :]

            #print(self.vvx_pl[:, kmin:kmaxp1, :].shape, vx_pl[:, kmin:kmaxp1, :].shape, wc.shape, gx.shape)
            #prc.prc_mpistop(std.io_l, std.fname_log)

            # Compute vvx_pl = vx_pl + (wc * gx / rscale)
            self.vvx_pl[:, kmin:kmaxp1, :] = vx_pl[:, kmin:kmaxp1, :] + (wc * gx / rscale)

            # Compute vvy_pl
            self.vvy_pl[:, kmin:kmaxp1, :] = vy_pl[:, kmin:kmaxp1, :] + (wc * gy / rscale)

            # Compute vvz_pl
            self.vvz_pl[:, kmin:kmaxp1, :] = vz_pl[:, kmin:kmaxp1, :] + (wc * gz / rscale)

            # Set ghost layers to zero
            self.vvx_pl[:, kminm1, :] = 0.0
            self.vvx_pl[:, kmaxp1, :] = 0.0
            self.vvy_pl[:, kminm1, :] = 0.0
            self.vvy_pl[:, kmaxp1, :] = 0.0
            self.vvz_pl[:, kminm1, :] = 0.0
            self.vvz_pl[:, kmaxp1, :] = 0.0

        #endif

        #---< advection term for momentum >

        self.src_advection_convergence(
                    rhogvx, rhogvx_pl,
                    rhogvy, rhogvy_pl, 
                    rhogvz, rhogvz_pl, 
                    rhogw,  rhogw_pl, 
                    self.vvx, self.vvx_pl,
                    self.dvvx, self.dvvx_pl,  
                    self.I_SRC_default,
                    grd, oprt, vmtr, rdtype, 
        )

        self.src_advection_convergence(
                    rhogvx, rhogvx_pl,
                    rhogvy, rhogvy_pl, 
                    rhogvz, rhogvz_pl, 
                    rhogw,  rhogw_pl, 
                    self.vvy, self.vvy_pl,
                    self.dvvy, self.dvvy_pl,  
                    self.I_SRC_default,
                    grd, oprt, vmtr, rdtype, 
        )

        self.src_advection_convergence(
                    rhogvx, rhogvx_pl,
                    rhogvy, rhogvy_pl, 
                    rhogvz, rhogvz_pl, 
                    rhogw,  rhogw_pl, 
                    self.vvz, self.vvz_pl,
                    self.dvvz, self.dvvz_pl,  
                    self.I_SRC_default,
                    grd, oprt, vmtr, rdtype, 
        )

 

        if grd.GRD_grid_type == grd.GRD_grid_type_on_plane:

            print("on plane not tested yet!")

            # Main volume computation (for kmin to kmax)
            grhogvx[:, :, kmin:kmaxp1, :] = self.dvvx[:, :, kmin:kmaxp1, :] + f * rhog[:, :, kmin:kmaxp1, :] * self.vvy[:, :, kmin:kmaxp1, :]
            grhogvy[:, :, kmin:kmaxp1, :] = self.dvvy[:, :, kmin:kmaxp1, :] - f * rhog[:, :, kmin:kmaxp1, :] * self.vvx[:, :, kmin:kmaxp1, :]
            grhogvz[:, :, kmin:kmaxp1, :] = 0.0  # Initialize to zero

            # grhogw using VMTR_C2Wfact
            fact1 = vmtr.VMTR_C2Wfact[:, :, kmin:kmaxp1, 0, :]  # (i, j, k, l)
            fact2 = vmtr.VMTR_C2Wfact[:, :, kmin:kmaxp1, 1, :]

            grhogw[:, :, kmin:kmaxp1, :] = alpha * (
                fact1 * self.dvvz[:, :, kmin:kmaxp1, :] +
                fact2 * self.dvvz[:, :, kmin-1:kmax,   :]
            )

            # Set ghost cells (boundary layers) to zero
            grhogvx[:, :, kminm1, :] = 0.0
            grhogvx[:, :, kmaxp1, :] = 0.0
            grhogvy[:, :, kminm1, :] = 0.0
            grhogvy[:, :, kmaxp1, :] = 0.0
            grhogvz[:, :, kminm1, :] = 0.0
            grhogvz[:, :, kmaxp1, :] = 0.0
            grhogw[:, :, kminm1, :]  = 0.0
            grhogw[:, :, kmin,   :]  = 0.0  # note: this matches original zeroing
            grhogw[:, :, kmaxp1, :]  = 0.0

        else:

            # 1. --- Coriolis Force (vectorized) ---
            self.dvvx[:, :, kmin:kmaxp1, :] -= -2.0 * rhog[:, :, kmin:kmaxp1, :] * (ohm * self.vvy[:, :, kmin:kmaxp1, :])
            self.dvvy[:, :, kmin:kmaxp1, :] -=  2.0 * rhog[:, :, kmin:kmaxp1, :] * (ohm * self.vvx[:, :, kmin:kmaxp1, :])

            # 2. --- Horizontalization & Vertical Velocity Separation ---
            # Extract directional vectors and broadcast
            gx = grd.GRD_x[:, :, 0, :, XDIR][:, :, None, :]  # (i, j, 1, l)
            gy = grd.GRD_x[:, :, 0, :, YDIR][:, :, None, :]
            gz = grd.GRD_x[:, :, 0, :, ZDIR][:, :, None, :]

            gx /= rscale
            gy /= rscale
            gz /= rscale

            # Compute prd = projection of dvv* on GRD_x
            prd = (
                self.dvvx[:, :, kmin:kmaxp1, :] * gx +
                self.dvvy[:, :, kmin:kmaxp1, :] * gy +
                self.dvvz[:, :, kmin:kmaxp1, :] * gz
            )

            # grhogv* = dvv* - prd * GRD_x component
            grhogvx[:, :, kmin:kmaxp1, :] = self.dvvx[:, :, kmin:kmaxp1, :] - prd * gx
            grhogvy[:, :, kmin:kmaxp1, :] = self.dvvy[:, :, kmin:kmaxp1, :] - prd * gy
            grhogvz[:, :, kmin:kmaxp1, :] = self.dvvz[:, :, kmin:kmaxp1, :] - prd * gz
            grhogwc = np.empty_like(grhogw)
            grhogwc[:, :, kmin:kmaxp1, :] = prd * alpha

            # 3. --- Compute grhogw ---
            fact1 = vmtr.VMTR_C2Wfact[:, :, kminp1:kmaxp1, 0, :]  # shape (i, j, k, l)
            fact2 = vmtr.VMTR_C2Wfact[:, :, kminp1:kmaxp1, 1, :]

            grhogw[:, :, kminp1:kmaxp1, :] = (
                fact1 * grhogwc[:, :, kminp1:kmaxp1, :] +
                fact2 * grhogwc[:, :, kmin:kmax,     :]
            )

            # 4. --- Ghost Layer Zeroing ---
            grhogvx[:, :, kminm1, :] = 0.0
            grhogvx[:, :, kmaxp1, :] = 0.0
            grhogvy[:, :, kminm1, :] = 0.0
            grhogvy[:, :, kmaxp1, :] = 0.0
            grhogvz[:, :, kminm1, :] = 0.0
            grhogvz[:, :, kmaxp1, :] = 0.0
            grhogw[:, :, kminm1,  :] = 0.0
            grhogw[:, :, kmin,    :] = 0.0
            grhogw[:, :, kmaxp1,  :] = 0.0

        #endif

        if adm.ADM_have_pl:

#alpha = NON_HYDRO_ALPHA  # real scalar

            # --- Coriolis force ---
            self.dvvx_pl[:, kmin:kmaxp1, :] -= -2.0 * rhog_pl[:, kmin:kmaxp1, :] * (-ohm * self.vvy_pl[:, kmin:kmaxp1, :])
            self.dvvy_pl[:, kmin:kmaxp1, :] -=  2.0 * rhog_pl[:, kmin:kmaxp1, :] * ( ohm * self.vvx_pl[:, kmin:kmaxp1, :])

            # --- Horizontalize and separate vertical velocity ---
            gx = grd.GRD_x_pl[:, 0, :, XDIR] / rscale  # shape (g, l)
            gy = grd.GRD_x_pl[:, 0, :, YDIR] / rscale
            gz = grd.GRD_x_pl[:, 0, :, ZDIR] / rscale

            gx = gx[:, None, :]  # shape (g, 1, l)
            gy = gy[:, None, :]
            gz = gz[:, None, :]

            prd = (
                self.dvvx_pl[:, kmin:kmaxp1, :] * gx +
                self.dvvy_pl[:, kmin:kmaxp1, :] * gy +
                self.dvvz_pl[:, kmin:kmaxp1, :] * gz
            )

            grhogvx_pl[:, kmin:kmaxp1, :] = self.dvvx_pl[:, kmin:kmaxp1, :] - prd * gx
            grhogvy_pl[:, kmin:kmaxp1, :] = self.dvvy_pl[:, kmin:kmaxp1, :] - prd * gy
            grhogvz_pl[:, kmin:kmaxp1, :] = self.dvvz_pl[:, kmin:kmaxp1, :] - prd * gz
            grhogwc_pl = np.empty_like(grhogw_pl)
            grhogwc_pl[:, kmin:kmaxp1, :] = prd * alpha

            # --- Compute grhogw_pl from grhogwc_pl ---
            fact1 = vmtr.VMTR_C2Wfact_pl[:, kminp1:kmaxp1, 0, :]
            fact2 = vmtr.VMTR_C2Wfact_pl[:, kminp1:kmaxp1, 1, :]

            grhogw_pl[:, kminp1:kmaxp1, :] = (
                fact1 * grhogwc_pl[:, kminp1:kmaxp1, :] +
                fact2 * grhogwc_pl[:, kmin:kmax,     :]
            )

            # --- Set ghost layers to 0.0
            grhogvx_pl[:, kminm1, :] = 0.0
            grhogvx_pl[:, kmaxp1, :] = 0.0
            grhogvy_pl[:, kminm1, :] = 0.0
            grhogvy_pl[:, kmaxp1, :] = 0.0
            grhogvz_pl[:, kminm1, :] = 0.0
            grhogvz_pl[:, kmaxp1, :] = 0.0
            grhogw_pl[:,  kminm1, :] = 0.0
            grhogw_pl[:,  kmin,   :] = 0.0
            grhogw_pl[:,  kmaxp1, :] = 0.0

        else:
            grhogvx_pl[:,:,:] = 0.0
            grhogvy_pl[:,:,:] = 0.0
            grhogvz_pl[:,:,:] = 0.0
            grhogw_pl [:,:,:] = 0.0
        #endif

        prf.PROF_rapend('____src_advection_conv_m',2)

        return
    
    def src_advection_convergence(self,
            rhogvx, rhogvx_pl,   
            rhogvy, rhogvy_pl,   
            rhogvz, rhogvz_pl,   
            rhogw, rhogw_pl,    
            scl, scl_pl,      
            grhogscl, grhogscl_pl, 
            fluxtype,
            grd, oprt, vmtr, rdtype,
    ):
        
        prf.PROF_rapstart('____src_advection_conv',2)

        gall = adm.ADM_gall
        kall = adm.ADM_kdall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        kminm1 = kmin - 1
        kminp1 = kmin + 1
        kmaxp1 = kmax + 1
        kmaxp2 = kmax + 2

        # rhogvh * scl

        np.multiply(rhogvx, scl, out=self.rhogvxscl)
        np.multiply(rhogvy, scl, out=self.rhogvyscl)
        np.multiply(rhogvz, scl, out=self.rhogvzscl)
        if adm.ADM_have_pl:
            np.multiply(rhogvx_pl, scl_pl, out=self.rhogvxscl_pl)
            np.multiply(rhogvy_pl, scl_pl, out=self.rhogvyscl_pl)
            np.multiply(rhogvz_pl, scl_pl, out=self.rhogvzscl_pl)


        # rhogw * scl at half level
        if fluxtype == self.I_SRC_default:

            # Pre-broadcasted afact and bfact for performance
            afact = grd.GRD_afact[kmin:kmaxp2][None, None, :, None]  # shape (k, 1, 1, 1)
            bfact = grd.GRD_bfact[kmin:kmaxp2][None, None, :, None]

            # Allocate or reuse a temporary array for weighted scalar field
            weighted_scl = np.empty_like(rhogw[:, :, kmin:kmaxp2, :])

            # weighted_scl = afact * scl[k] + bfact * scl[k-1]
            np.multiply(afact, scl[:, :, kmin:kmaxp2, :], out=weighted_scl)
            weighted_scl += bfact * scl[:, :, kmin-1:kmaxp1, :]

            # Apply to rhogw using out=
            np.multiply(rhogw[:, :, kmin:kmaxp2, :], weighted_scl, out=self.rhogwscl[:, :, kmin:kmaxp2, :])

            # Zero out kmin-1 layer
            self.rhogwscl[:, :, kminm1, :] = 0.0


            if adm.ADM_have_pl:

                afact = grd.GRD_afact[kmin:kmaxp2][None, :, None]  # (k, 1, 1)
                bfact = grd.GRD_bfact[kmin:kmaxp2][None, :, None]

                weighted_scl_pl = (
                    afact * scl_pl[:, kmin:kmaxp2, :] +
                    bfact * scl_pl[:, kminm1:kmaxp1,   :]
                )

                self.rhogwscl_pl[:, kmin:kmaxp2, :] = rhogw_pl[:, kmin:kmaxp2, :] * weighted_scl_pl

                self.rhogwscl_pl[:, kminm1, :] = 0.0

        elif fluxtype == self.I_SRC_horizontal:

            self.rhogwscl.fill(0.0)
            
            if adm.ADM_have_pl:
                self.rhogwscl_pl.fill(0.0)

        #endif

        #--- flux convergence step

        self.src_flux_convergence(
                self.rhogvxscl, self.rhogvxscl_pl, 
                self.rhogvyscl, self.rhogvyscl_pl, 
                self.rhogvzscl, self.rhogvzscl_pl, 
                self.rhogwscl,  self.rhogwscl_pl,  
                grhogscl,  grhogscl_pl,  
                fluxtype, 
                grd, oprt, vmtr, rdtype, 
        )

        # call src_flux_convergence( rhogvxscl, rhogvxscl_pl, & ! [IN]
        #                        rhogvyscl, rhogvyscl_pl, & ! [IN]
        #                        rhogvzscl, rhogvzscl_pl, & ! [IN]
        #                        rhogwscl,  rhogwscl_pl,  & ! [IN]
        #                        grhogscl,  grhogscl_pl,  & ! [OUT]
        #                        fluxtype                 ) ! [IN]

        prf.PROF_rapend('____src_advection_conv',2)

        return
    
    
    # > Flux convergence calculation
    #  1. Horizontal flux convergence is calculated by using rhovx, rhovy, and
    #     rhovz which are defined at cell center (vertical) and A-grid (horizontal).
    #  2. Vertical flux convergence is calculated by using rhovx, rhovy, rhovz, and rhow.
    #  3. rhovx, rhovy, and rhovz can be replaced by rhovx*h, rhovy*h, and rhovz*h, respectively.
    def src_flux_convergence(self,
            rhogvx, rhogvx_pl,
            rhogvy, rhogvy_pl,
            rhogvz, rhogvz_pl,
            rhogw,  rhogw_pl, 
            grhog,  grhog_pl,
            fluxtype,
            grd, oprt, vmtr, rdtype,
    ):
        
        prf.PROF_rapstart('____src_flux_conv',2)

        gall = adm.ADM_gall
        kall = adm.ADM_kdall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        kminm1 = kmin - 1
        kminp1 = kmin + 1
        kmaxp1 = kmax + 1
        kmaxp2 = kmax + 2

        if fluxtype == self.I_SRC_default: # Default
           vertical_flag = 1.0
        elif fluxtype == self.I_SRC_horizontal: # Horizontal
           vertical_flag = 0.0
        #endif

        #--- Horizontal flux
        np.multiply(rhogvx, vmtr.VMTR_RGAM, out=self.rhogvx_vm)
        np.multiply(rhogvy, vmtr.VMTR_RGAM, out=self.rhogvy_vm)
        np.multiply(rhogvz, vmtr.VMTR_RGAM, out=self.rhogvz_vm)
                

        #--- Vertical flux

        # Extract VMTR_C2WfactGz components for vectorized use
        fact1 = vmtr.VMTR_C2WfactGz[:, :, kminp1:kmaxp1, 0, :]  # shape: (i, j, k, l)
        fact2 = vmtr.VMTR_C2WfactGz[:, :, kminp1:kmaxp1, 1, :]
        fact3 = vmtr.VMTR_C2WfactGz[:, :, kminp1:kmaxp1, 2, :]
        fact4 = vmtr.VMTR_C2WfactGz[:, :, kminp1:kmaxp1, 3, :]
        fact5 = vmtr.VMTR_C2WfactGz[:, :, kminp1:kmaxp1, 4, :]
        fact6 = vmtr.VMTR_C2WfactGz[:, :, kminp1:kmaxp1, 5, :]

        # Horizontal contribution
        horiz = (
            fact1 * rhogvx[:, :, kminp1:kmaxp1, :] +
            fact2 * rhogvx[:, :, kmin:kmax,     :] +
            fact3 * rhogvy[:, :, kminp1:kmaxp1, :] +
            fact4 * rhogvy[:, :, kmin:kmax,     :] +
            fact5 * rhogvz[:, :, kminp1:kmaxp1, :] +
            fact6 * rhogvz[:, :, kmin:kmax,     :]
        )

        # Multiply horizontal part by RGAMH
        np.multiply(horiz, vmtr.VMTR_RGAMH[:, :, kminp1:kmaxp1, :], out=horiz)

        # Vertical contribution
        vert = vertical_flag * rhogw[:, :, kminp1:kmaxp1, :] * vmtr.VMTR_RGSQRTH[:, :, kminp1:kmaxp1, :]

        # Final sum
        self.rhogw_vmh[:, :, kminp1:kmaxp1, :] = horiz + vert

        # Boundary zeroing
        self.rhogw_vmh[:, :, kmin,   :] = 0.0
        self.rhogw_vmh[:, :, kmaxp1, :] = 0.0


        if adm.ADM_have_pl:

            # --- Horizontal flux (element-wise product)
            np.multiply(rhogvx_pl, vmtr.VMTR_RGAM_pl, out=self.rhogvx_vm_pl)
            np.multiply(rhogvy_pl, vmtr.VMTR_RGAM_pl, out=self.rhogvy_vm_pl)
            np.multiply(rhogvz_pl, vmtr.VMTR_RGAM_pl, out=self.rhogvz_vm_pl)

            # --- Vertical flux
            # Extract factGz coefficients for broadcast
            f1 = vmtr.VMTR_C2WfactGz_pl[:, kminp1:kmaxp1, 0, :]
            f2 = vmtr.VMTR_C2WfactGz_pl[:, kminp1:kmaxp1, 1, :]
            f3 = vmtr.VMTR_C2WfactGz_pl[:, kminp1:kmaxp1, 2, :]
            f4 = vmtr.VMTR_C2WfactGz_pl[:, kminp1:kmaxp1, 3, :]
            f5 = vmtr.VMTR_C2WfactGz_pl[:, kminp1:kmaxp1, 4, :]
            f6 = vmtr.VMTR_C2WfactGz_pl[:, kminp1:kmaxp1, 5, :]

            horiz = (
                f1 * rhogvx_pl[:, kminp1:kmaxp1, :] +
                f2 * rhogvx_pl[:, kmin:kmax,     :] +
                f3 * rhogvy_pl[:, kminp1:kmaxp1, :] +
                f4 * rhogvy_pl[:, kmin:kmax,     :] +
                f5 * rhogvz_pl[:, kminp1:kmaxp1, :] +
                f6 * rhogvz_pl[:, kmin:kmax,     :]
            )

            np.multiply(horiz, vmtr.VMTR_RGAMH_pl[:, kminp1:kmaxp1, :], out=horiz)

            vert = vertical_flag * rhogw_pl[:, kminp1:kmaxp1, :] * vmtr.VMTR_RGSQRTH_pl[:, kminp1:kmaxp1, :]

            self.rhogw_vmh_pl[:, kminp1:kmaxp1, :] = horiz + vert

            # --- Boundary zeroing
            self.rhogw_vmh_pl[:, kmin,   :] = 0.0
            self.rhogw_vmh_pl[:, kmaxp1, :] = 0.0

        #--- Horizontal flux convergence
        oprt.OPRT_divergence(
            self.div_rhogvh, self.div_rhogvh_pl, # [OUT]
            self.rhogvx_vm,  self.rhogvx_vm_pl,  # [IN]
            self.rhogvy_vm,  self.rhogvy_vm_pl,  # [IN]
            self.rhogvz_vm,  self.rhogvz_vm_pl,  # [IN]
            oprt.OPRT_coef_div, oprt.OPRT_coef_div_pl, # [IN]
            grd, rdtype,
        ) 

        #--- Total flux convergence

        # Vertical flux difference (rhogw_vmh[k+1] - rhogw_vmh[k]) * GRD_rdgz[k]
        # GRD_rdgz[k] → reshape for broadcasting: (k, 1, 1, 1)
        rdgz = grd.GRD_rdgz[kmin:kmaxp1][None, None, :, None]

        # Compute difference between k+1 and k
        flux_diff = self.rhogw_vmh[:, :, kminp1:kmaxp2, :] - self.rhogw_vmh[:, :, kmin:kmaxp1, :]

        # Apply vertical term
        vert_term = flux_diff * rdgz  # shape (i, j, k, l)

        # Final grhog update
        grhog[:, :, kmin:kmaxp1, :] = - self.div_rhogvh[:, :, kmin:kmaxp1, :] - vert_term

        # Set ghost layers to zero
        grhog[:, :, kminm1, :] = 0.0
        grhog[:, :, kmaxp1, :] = 0.0


        if adm.ADM_have_pl:

            # Precompute vertical flux difference: (rhogw_vmh[k+1] - rhogw_vmh[k])
            flux_diff_pl = self.rhogw_vmh_pl[:, kminp1:kmaxp2, :] - self.rhogw_vmh_pl[:, kmin:kmaxp1, :]  # shape: (g, k, l)

            # GRD_rdgz[k] needs reshaping for broadcasting: (k, 1)
            rdgz = grd.GRD_rdgz[kmin:kmaxp1][:, None]  # (k, 1)

            # Apply vertical term
            vert_term_pl = flux_diff_pl * rdgz[None, :, :]  # shape: (g, k, l)

            # Final grhog update
            grhog_pl[:, kmin:kmaxp1, :] = - self.div_rhogvh_pl[:, kmin:kmaxp1, :] - vert_term_pl

            # Set ghost layers to zero
            grhog_pl[:, kminm1, :] = 0.0
            grhog_pl[:, kmaxp1, :] = 0.0


        prf.PROF_rapend('____src_flux_conv',2)

        return
    

    def src_pres_gradient(self,
        P,      P_pl,      
        Pgrad,  Pgrad_pl,  
        Pgradw, Pgradw_pl, 
        gradtype,
        grd, oprt, vmtr, rdtype,           
    ):
        
        prf.PROF_rapstart('____src_pres_gradient',2)

        P_vm     = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall,              ), dtype=rdtype)
        P_vm_pl  = np.empty((adm.ADM_gall_pl,                  adm.ADM_kdall, adm.ADM_lall,              ), dtype=rdtype)
        P_vmh    = np.empty((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall, adm.ADM_nxyz,), dtype=rdtype)
        P_vmh_pl = np.empty((adm.ADM_gall_pl,                  adm.ADM_kdall, adm.ADM_lall, adm.ADM_nxyz,), dtype=rdtype)

        gall = adm.ADM_gall
        kall = adm.ADM_kall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        nxyz = adm.ADM_nxyz

        XDIR = grd.GRD_XDIR
        YDIR = grd.GRD_YDIR
        ZDIR = grd.GRD_ZDIR

        #---< horizontal gradient, horizontal contribution >---

        for l in range(lall):
            for k in range(kall):
                P_vm[:, :, k, l] = P[:, :, k, l] * vmtr.VMTR_RGAM[:, :, k, l]

        if adm.ADM_have_pl:
            P_vm_pl[:, :, :] = P_pl[:, :, :] * vmtr.VMTR_RGAM_pl[:, :, :]
        #endif

        oprt.OPRT_gradient(
            Pgrad[:,:,:,:,:], Pgrad_pl[:,:,:,:],                 # [OUT]
            P_vm[:,:,:,:],   P_vm_pl[:,:,:],                     # [IN]
            oprt.OPRT_coef_grad, oprt.OPRT_coef_grad_pl,         # [IN] (array shape omitted for simplicity)
            grd, rdtype,
        )
        
        #---< horizontal gradient, vertical contribution >---

        for l in range(lall):
            for k in range(kmin, kmax + 2):  # includes kmax+1
                P_vmh[:, :, k, l, XDIR] = (
                    vmtr.VMTR_C2WfactGz[:, :, k, 0, l] * P[:, :, k, l] +
                    vmtr.VMTR_C2WfactGz[:, :, k, 1, l] * P[:, :, k - 1, l]
                ) * vmtr.VMTR_RGAMH[:, :, k, l]

                P_vmh[:, :, k, l, YDIR] = (
                    vmtr.VMTR_C2WfactGz[:, :, k, 2, l] * P[:, :, k, l] +
                    vmtr.VMTR_C2WfactGz[:, :, k, 3, l] * P[:, :, k - 1, l]
                ) * vmtr.VMTR_RGAMH[:, :, k, l]

                P_vmh[:, :, k, l, ZDIR] = (
                    vmtr.VMTR_C2WfactGz[:, :, k, 4, l] * P[:, :, k, l] +
                    vmtr.VMTR_C2WfactGz[:, :, k, 5, l] * P[:, :, k - 1, l]
                ) * vmtr.VMTR_RGAMH[:, :, k, l]
            #end k loop

            for d in range(nxyz):
                for k in range(kmin, kmax + 1):
                    Pgrad[:, :, k, l, d] += (
                        P_vmh[:, :, k + 1, l, d] - P_vmh[:, :, k, l, d]
                    ) * grd.GRD_rdgz[k]
                #end k loop
            
                Pgrad[:, :, kmin - 1, l, d] = 0.0
                Pgrad[:, :, kmax + 1, l, d] = 0.0

                if self.first_layer_remedy: #--- At the lowest layer, do not use the extrapolation value      
                    Pgrad[:, :, kmin, l, d] = Pgrad[:, :, kmin + 1, l, d]
                #endif
            #end d loop
        #end l loop

        if adm.ADM_have_pl:
            k_range = np.arange(kmin, kmax + 2)  # includes kmax+1

            # Vectorized computation for P_vmh_pl over all directions
            P_vmh_pl[:, k_range, :, XDIR] = (
                vmtr.VMTR_C2WfactGz_pl[:, k_range, 0, :] * P_pl[:, k_range, :] +
                vmtr.VMTR_C2WfactGz_pl[:, k_range, 1, :] * P_pl[:, k_range - 1, :]
            ) * vmtr.VMTR_RGAMH_pl[:, k_range, :]

            P_vmh_pl[:, k_range, :, YDIR] = (
                vmtr.VMTR_C2WfactGz_pl[:, k_range, 2, :] * P_pl[:, k_range, :] +
                vmtr.VMTR_C2WfactGz_pl[:, k_range, 3, :] * P_pl[:, k_range - 1, :]
            ) * vmtr.VMTR_RGAMH_pl[:, k_range, :]

            P_vmh_pl[:, k_range, :, ZDIR] = (
                vmtr.VMTR_C2WfactGz_pl[:, k_range, 4, :] * P_pl[:, k_range, :] +
                vmtr.VMTR_C2WfactGz_pl[:, k_range, 5, :] * P_pl[:, k_range - 1, :]
            ) * vmtr.VMTR_RGAMH_pl[:, k_range, :]

            # Pressure gradient update
            for d in range(adm.ADM_nxyz):
                k_mid = np.arange(kmin, kmax + 1)
                Pgrad_pl[:, k_mid, :, d] += (
                    P_vmh_pl[:, k_mid + 1, :, d] - P_vmh_pl[:, k_mid, :, d]
                ) * grd.GRD_rdgz[k_mid, None]

                if self.first_layer_remedy: #--- At the lowest layer, do not use the extrapolation value!
                    Pgrad_pl[:, kmin, :, d] = Pgrad_pl[:, kmin + 1, :, d]
                #endif

                Pgrad_pl[:, kmin - 1, :, d] = 0.0
                Pgrad_pl[:, kmax + 1, :, d] = 0.0
            #end d loop
        #endif

        #--- horizontalize
        oprt.OPRT_horizontalize_vec(
            Pgrad[:,:,:,:,XDIR], Pgrad_pl[:,:,:,XDIR], # [INOUT]
            Pgrad[:,:,:,:,YDIR], Pgrad_pl[:,:,:,YDIR], # [INOUT]
            Pgrad[:,:,:,:,ZDIR], Pgrad_pl[:,:,:,ZDIR], # [INOUT]
            grd, rdtype,
        )

        #---< vertical gradient (half level) >---

        if gradtype == self.I_SRC_default:

            for l in range(lall):
                for k in range(kmin + 1, kmax + 1):
                    Pgradw[:, :, k, l] = (
                        vmtr.VMTR_GAM2H[:, :, k, l] *
                        (P[:, :, k, l] * vmtr.VMTR_RGSGAM2[:, :, k, l] -
                        P[:, :, k-1, l] * vmtr.VMTR_RGSGAM2[:, :, k-1, l]) *
                        grd.GRD_rdgzh[k]
                    )
                #end k loop

                # Boundary/ghost layers
                Pgradw[:, :, kmin - 1, l] = 0.0
                Pgradw[:, :, kmin,     l] = 0.0
                Pgradw[:, :, kmax + 1, l] = 0.0
            #end l loop

            if adm.ADM_have_pl:
                k_range = np.arange(kmin + 1, kmax + 1)

                # Vectorized pressure gradient (w-direction)
                Pgradw_pl[:, k_range, :] = (
                    vmtr.VMTR_GAM2H_pl[:, k_range, :] *
                    (P_pl[:, k_range, :] * vmtr.VMTR_RGSGAM2_pl[:, k_range, :] -
                    P_pl[:, k_range - 1, :] * vmtr.VMTR_RGSGAM2_pl[:, k_range - 1, :])
                    * grd.GRD_rdgzh[k_range, None]
                )

                # Set ghost levels to zero
                Pgradw_pl[:, kmin - 1, :] = 0.0
                Pgradw_pl[:, kmin,     :] = 0.0
                Pgradw_pl[:, kmax + 1, :] = 0.0
            #endif

        elif gradtype == self.I_SRC_horizontal:

            Pgradw[:, :, :, :] = 0.0


            if adm.ADM_have_pl:
                Pgradw_pl[:, :, :] = 0.0
            #endif

        #endif

        prf.PROF_rapend('____src_pres_gradient',2)

        return 

    #> Buoyacy force
    #> Note: Upward direction is positive for buoiw.
    def src_buoyancy(self,
        rhog,  rhog_pl, 
        buoiw, buoiw_pl,
        cnst, vmtr, rdtype,
    ):
    
        prf.PROF_rapstart('____src_buoyancy',2)

        gall = adm.ADM_gall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall

        grav = cnst.CONST_GRAV

        for l in range(lall):
            for k in range(kmin + 1, kmax):
                buoiw[:, :, k, l] = -grav * (
                    vmtr.VMTR_C2Wfact[:, :, k, 0, l] * rhog[:, :, k, l] +
                    vmtr.VMTR_C2Wfact[:, :, k, 1, l] * rhog[:, :, k - 1, l]
                )
            #end k loop

            buoiw[:, :, kmin - 1, l] = 0.0
            buoiw[:, :, kmin,     l] = 0.0
            buoiw[:, :, kmax + 1, l] = 0.0
        #end l loop

        # Pole region
        if adm.ADM_have_pl:
            for l in range(adm.ADM_lall_pl):
                buoiw_pl[:, kmin + 1:kmax, l] = -grav * (
                    vmtr.VMTR_C2Wfact_pl[:, kmin + 1:kmax, 0, l] * rhog_pl[:, kmin + 1:kmax, l] +
                    vmtr.VMTR_C2Wfact_pl[:, kmin + 1:kmax, 1, l] * rhog_pl[:, kmin:kmax - 1, l]
                )

                buoiw_pl[:, kmin - 1, l] = 0.0
                buoiw_pl[:, kmin,     l] = 0.0
                buoiw_pl[:, kmax + 1, l] = 0.0
            # end l loop
        #endif

        prf.PROF_rapend('____src_buoyancy',2)

        return
    