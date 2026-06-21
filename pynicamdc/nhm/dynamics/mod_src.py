import toml
import numpy as np
#from mpi4py import MPI
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_prof import prf
from pynicamdc.share.mod_ppmask import ppm
from pynicamdc.share.mod_backend import backend as bk
from pynicamdc.nhm.dynamics.kernels.buoyancy import BuoyCfg, compute_buoyancy
from pynicamdc.nhm.dynamics.kernels.advconv import AdvConvCfg, compute_scaled_fluxes
from pynicamdc.nhm.dynamics.kernels.fluxconv import FluxConvCfg, compute_flux_convergence
from pynicamdc.nhm.dynamics.kernels.presgrad import PresGradCfg, compute_pres_gradient
from pynicamdc.nhm.dynamics.kernels.advconvmom import (
    AdvMomCfg,
    compute_merged_velocity_reg, compute_merged_velocity_pl,
    compute_momentum_tendency_reg, compute_momentum_tendency_pl,
)

class Src:
    
    _instance = None
    
    I_SRC_horizontal = 1
    I_SRC_vertical   = 2
    I_SRC_default    = 3

    # I_SRC_default    : horizontal & vertical convergence
    # I_SRC_horizontal : horizontal convergence

    first_layer_remedy = True

    def __init__(self,cnst,rdtype):

        self.vvx  = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype)
        self.vvy  = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype)
        self.vvz  = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype)
        self.dvvx = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype)
        self.dvvy = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype)
        self.dvvz = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype)
        self.vvx_pl  = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)
        self.vvy_pl  = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)
        self.vvz_pl  = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)
        self.dvvx_pl = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)
        self.dvvy_pl = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)
        self.dvvz_pl = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)
        self.rhogvxscl = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype)
        self.rhogvyscl = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype)
        self.rhogvzscl = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype)
        self.rhogwscl  = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype)
        self.rhogvxscl_pl = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)
        self.rhogvyscl_pl = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)
        self.rhogvzscl_pl = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)
        self.rhogwscl_pl  = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)
        self.rhogvx_vm = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype) #rho*vx / vertical metrics
        self.rhogvy_vm = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype) #rho*vy / vertical metrics
        self.rhogvz_vm = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype) #rho*vz / vertical metrics  
        self.rhogvx_vm_pl = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)  #rho*vx / vertical metrics  
        self.rhogvy_vm_pl = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)  #rho*vy / vertical metrics
        self.rhogvz_vm_pl = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)  #rho*vz / vertical metrics      
        self.rhogw_vmh  = np.full((adm.ADM_shape),cnst.CONST_UNDEF, dtype=rdtype) #rho*w / vertical metrics 
        self.rhogw_vmh_pl  = np.full((adm.ADM_shape_pl),cnst.CONST_UNDEF, dtype=rdtype)  #rho*w / vertical metrics

        # self.vvx  = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.vvy  = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.vvz  = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.dvvx = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.dvvy = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.dvvz = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.vvx_pl  = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # self.vvy_pl  = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # self.vvz_pl  = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # self.dvvx_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # self.dvvy_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # self.dvvz_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # self.rhogvxscl = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.rhogvyscl = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.rhogvzscl = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.rhogwscl  = np.empty((adm.ADM_shape), dtype=rdtype)
        # self.rhogvxscl_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # self.rhogvyscl_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # self.rhogvzscl_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # self.rhogwscl_pl  = np.empty((adm.ADM_shape_pl), dtype=rdtype)
        # self.rhogvx_vm = np.empty((adm.ADM_shape), dtype=rdtype) #rho*vx / vertical metrics
        # self.rhogvy_vm = np.empty((adm.ADM_shape), dtype=rdtype) #rho*vy / vertical metrics
        # self.rhogvz_vm = np.empty((adm.ADM_shape), dtype=rdtype) #rho*vz / vertical metrics  
        # self.rhogvx_vm_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)  #rho*vx / vertical metrics  
        # self.rhogvy_vm_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)  #rho*vy / vertical metrics
        # self.rhogvz_vm_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)  #rho*vz / vertical metrics      
        # self.rhogw_vmh  = np.empty((adm.ADM_shape), dtype=rdtype) #rho*w / vertical metrics 
        # self.rhogw_vmh_pl  = np.empty((adm.ADM_shape_pl), dtype=rdtype)  #rho*w / vertical metrics

        #self.div_rhogvh = np.empty((adm.ADM_shape), dtype=rdtype) #horizontal convergence
        #self.div_rhogvh_pl = np.empty((adm.ADM_shape_pl), dtype=rdtype)  

    def src_advection_convergence_momentum(self, 
                vx,     vx_pl,         # [IN]
                vy,     vy_pl,         # [IN]
                vz,     vz_pl,         # [IN] 
                w,      w_pl,          # [IN]
                rhog,   rhog_pl,       # [IN]
                rhogvx, rhogvx_pl,     # [IN]
                rhogvy,  rhogvy_pl,    # [IN]
                rhogvz,  rhogvz_pl,    # [IN]
                rhogw,   rhogw_pl,     # [IN]
                grhogvx, grhogvx_pl,   # [OUT]   grhogvx very different for vindex > 0
                grhogvy, grhogvy_pl,   # [OUT] 
                grhogvz, grhogvz_pl,   # [OUT] 
                grhogw,  grhogw_pl,    # [OUT] 
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

        # --- backend-switchable COMM-free kernels (numpy<->jax). See kernels/advconvmom.py.
        xp = bk.xp
        if getattr(self, "_advmom_kernels", None) is None:
            self._advmom_cfg = AdvMomCfg(
                kmin=kmin, kmax=kmax, have_pl=adm.ADM_have_pl,
                XDIR=XDIR, YDIR=YDIR, ZDIR=ZDIR,
                rscale=float(rscale), ohm=float(ohm), alpha=float(alpha),
            )
            self._advmom_kernels = {
                "mr": bk.maybe_jit(compute_merged_velocity_reg, static_argnames=("cfg", "xp")),
                "mp": bk.maybe_jit(compute_merged_velocity_pl, static_argnames=("cfg", "xp")),
                "tr": bk.maybe_jit(compute_momentum_tendency_reg, static_argnames=("cfg", "xp")),
                "tp": bk.maybe_jit(compute_momentum_tendency_pl, static_argnames=("cfg", "xp")),
            }
        _amk = self._advmom_kernels
        _amd = bk.device_consts(self, "advmom", lambda: {
            "cfact":      grd.GRD_cfact,
            "dfact":      grd.GRD_dfact,
            "GRD_x":      grd.GRD_x,
            "C2Wfact":    vmtr.VMTR_C2Wfact,
            "GRD_x_pl":   grd.GRD_x_pl,
            "C2Wfact_pl": vmtr.VMTR_C2Wfact_pl,
        })

        if grd.GRD_grid_type == grd.GRD_grid_type_on_plane:

            print("on plane not tested yet!")

            self.vvx[:, :, kmin:kmaxp1, :] = vx[:, :, kmin:kmaxp1, :]
            self.vvy[:, :, kmin:kmaxp1, :] = vy[:, :, kmin:kmaxp1, :]

            # Prepare GRD factors (shape: (kmaxp1 - kmin, 1, 1, 1))
            # ? (shape: (1, 1, kmaxp1 - kmin, 1)) seems correct
            cfact = grd.GRD_cfact[kmin:kmaxp1][None, None, :, None]
            dfact = grd.GRD_dfact[kmin:kmaxp1][None, None, :, None]

            # Vectorized vvz computation
            self.vvz[:, :, kmin:kmaxp1, :] = (
                cfact * w[:, :, kminp1:kmaxp2, :] +
                dfact * w[:, :, kmin:kmaxp1,     :]
            )

            # Boundary layers
            self.vvx[:, :, kminm1, :] = rdtype(0.0)
            self.vvx[:, :, kmaxp1, :] = rdtype(0.0)
            self.vvy[:, :, kminm1, :] = rdtype(0.0)
            self.vvy[:, :, kmaxp1, :] = rdtype(0.0)
            self.vvz[:, :, kminm1, :] = rdtype(0.0)
            self.vvz[:, :, kmaxp1, :] = rdtype(0.0)

        else:

            _vvx, _vvy, _vvz = _amk["mr"](
                xp.asarray(vx), xp.asarray(vy), xp.asarray(vz), xp.asarray(w),
                _amd["cfact"], _amd["dfact"], _amd["GRD_x"],
                cfg=self._advmom_cfg, xp=xp,
            )
            self.vvx[:, :, :, :] = bk.to_numpy(_vvx)
            self.vvy[:, :, :, :] = bk.to_numpy(_vvy)
            self.vvz[:, :, :, :] = bk.to_numpy(_vvz)

        #endif

        if adm.ADM_have_pl:

            _vvx_pl, _vvy_pl, _vvz_pl = _amk["mp"](
                xp.asarray(vx_pl), xp.asarray(vy_pl), xp.asarray(vz_pl), xp.asarray(w_pl),
                _amd["cfact"], _amd["dfact"], _amd["GRD_x_pl"],
                cfg=self._advmom_cfg, xp=xp,
            )
            self.vvx_pl[:, :, :] = bk.to_numpy(_vvx_pl)
            self.vvy_pl[:, :, :] = bk.to_numpy(_vvy_pl)
            self.vvz_pl[:, :, :] = bk.to_numpy(_vvz_pl)

            # with open(std.fname_log, 'a') as log_file:
            #     print("vvxyz_pl check before calculating dvvxyz_pl", file=log_file)
            #     print("self.vvx_pl (:,2,0)", self.vvx_pl [:, 2, 0], file=log_file) 
            #     print("self.vvy_pl (:,2,0)", self.vvy_pl [:, 2, 0], file=log_file) 
            #     print("self.vvz_pl (:,2,0)", self.vvz_pl [:, 2, 0], file=log_file)   # all good at 2,0

        #endif

        #---< advection term for momentum >

        # For X
        self.src_advection_convergence(
                    rhogvx, rhogvx_pl,        # [IN]  rho*Vx ( G^1/2 x gam2 )
                    rhogvy, rhogvy_pl,        # [IN]  rho*Vy ( G^1/2 x gam2 )
                    rhogvz, rhogvz_pl,        # [IN]  rho*Vz ( G^1/2 x gam2 )
                    rhogw,  rhogw_pl,         # [IN]  rho*W ( G^1/2 x gam2 )
                    self.vvx, self.vvx_pl,    # [IN]  scalar
                    self.dvvx, self.dvvx_pl,  # [OUT] scalar tendency
                    self.I_SRC_default,       # default: horizontal & vertical convergence
                    cnst, grd, oprt, vmtr, rdtype, 
        )

        # For Y
        self.src_advection_convergence(
                    rhogvx, rhogvx_pl,
                    rhogvy, rhogvy_pl, 
                    rhogvz, rhogvz_pl, 
                    rhogw,  rhogw_pl, 
                    self.vvy, self.vvy_pl,
                    self.dvvy, self.dvvy_pl,  
                    self.I_SRC_default,
                    cnst, grd, oprt, vmtr, rdtype, 
        )

        # For Z
        self.src_advection_convergence(
                    rhogvx, rhogvx_pl,
                    rhogvy, rhogvy_pl, 
                    rhogvz, rhogvz_pl, 
                    rhogw,  rhogw_pl, 
                    self.vvz, self.vvz_pl,
                    self.dvvz, self.dvvz_pl,  
                    self.I_SRC_default,
                    cnst, grd, oprt, vmtr, rdtype, 
        )

 
        # with open(std.fname_log, 'a') as log_file:  
        #     kc=2
        # #     print("self.vvx (6,5,2,0)", self.vvx [6, 5, 2, 0], file=log_file) 
        # #     print("self.vvy (6,5,2,0)", self.vvy [6, 5, 2, 0], file=log_file) 
        # #     print("self.vvz (6,5,2,0)", self.vvz [6, 5, 2, 0], file=log_file) 
        # #     print("self.dvvx(6,5,2,0)", self.dvvx[6, 5, 2, 0], file=log_file) 
        # #     print("self.dvvy(6,5,2,0)", self.dvvy[6, 5, 2, 0], file=log_file) 
        # #    print("self.dvvz(6,5,2,0)", self.dvvz[6, 5, 2, 0], file=log_file)
        #     print(f"self.vvx_pl (:,{kc},0)", self.vvx_pl [:, kc, 0], file=log_file) 
        #     print(f"self.vvy_pl (:,{kc},0)", self.vvy_pl [:, kc, 0], file=log_file) 
        #     print(f"self.vvz_pl (:,{kc},0)", self.vvz_pl [:, kc, 0], file=log_file) 
        #     print(f"self.dvvx_pl(:,{kc},0)", self.dvvx_pl[:, kc, 0], file=log_file)  # differs from original, but perhaps because the numbers are very small
        #     print(f"self.dvvy_pl(:,{kc},0)", self.dvvy_pl[:, kc, 0], file=log_file) 
        #     print(f"self.dvvz_pl(:,{kc},0)", self.dvvz_pl[:, kc, 0], file=log_file) 

        if grd.GRD_grid_type == grd.GRD_grid_type_on_plane:

            print("on plane not tested yet!")

            # Main volume computation (for kmin to kmax)
            grhogvx[:, :, kmin:kmaxp1, :] = self.dvvx[:, :, kmin:kmaxp1, :] + f * rhog[:, :, kmin:kmaxp1, :] * self.vvy[:, :, kmin:kmaxp1, :]
            grhogvy[:, :, kmin:kmaxp1, :] = self.dvvy[:, :, kmin:kmaxp1, :] - f * rhog[:, :, kmin:kmaxp1, :] * self.vvx[:, :, kmin:kmaxp1, :]
            grhogvz[:, :, kmin:kmaxp1, :] = rdtype(0.0)  # Initialize to zero

            # grhogw using VMTR_C2Wfact
            fact1 = vmtr.VMTR_C2Wfact[:, :, kmin:kmaxp1, :, 0]  # (i, j, k, l)
            fact2 = vmtr.VMTR_C2Wfact[:, :, kmin:kmaxp1, :, 1]

            grhogw[:, :, kmin:kmaxp1, :] = alpha * (
                fact1 * self.dvvz[:, :, kmin:kmaxp1, :] +
                fact2 * self.dvvz[:, :, kmin-1:kmax,   :]
            )

            # Set ghost cells (boundary layers) to zero
            grhogvx[:, :, kminm1, :] = rdtype(0.0)
            grhogvx[:, :, kmaxp1, :] = rdtype(0.0)
            grhogvy[:, :, kminm1, :] = rdtype(0.0)
            grhogvy[:, :, kmaxp1, :] = rdtype(0.0)
            grhogvz[:, :, kminm1, :] = rdtype(0.0)
            grhogvz[:, :, kmaxp1, :] = rdtype(0.0)
            grhogw[:, :, kminm1, :]  = rdtype(0.0)
            grhogw[:, :, kmin,   :]  = rdtype(0.0)  
            grhogw[:, :, kmaxp1, :]  = rdtype(0.0)

        else:

            _gvx, _gvy, _gvz, _gw = _amk["tr"](
                xp.asarray(self.dvvx), xp.asarray(self.dvvy), xp.asarray(self.dvvz),
                xp.asarray(rhog), xp.asarray(self.vvx), xp.asarray(self.vvy),
                _amd["GRD_x"], _amd["C2Wfact"],
                cfg=self._advmom_cfg, xp=xp,
            )
            grhogvx[:, :, :, :] = bk.to_numpy(_gvx)
            grhogvy[:, :, :, :] = bk.to_numpy(_gvy)
            grhogvz[:, :, :, :] = bk.to_numpy(_gvz)
            grhogw[:, :, :, :]  = bk.to_numpy(_gw)

        #endif

        if adm.ADM_have_pl:

            _gvx_pl, _gvy_pl, _gvz_pl, _gw_pl = _amk["tp"](
                xp.asarray(self.dvvx_pl), xp.asarray(self.dvvy_pl), xp.asarray(self.dvvz_pl),
                xp.asarray(rhog_pl), xp.asarray(self.vvx_pl), xp.asarray(self.vvy_pl),
                _amd["GRD_x_pl"], _amd["C2Wfact_pl"],
                cfg=self._advmom_cfg, xp=xp,
            )
            grhogvx_pl[:, :, :] = bk.to_numpy(_gvx_pl)
            grhogvy_pl[:, :, :] = bk.to_numpy(_gvy_pl)
            grhogvz_pl[:, :, :] = bk.to_numpy(_gvz_pl)
            grhogw_pl[:, :, :]  = bk.to_numpy(_gw_pl)

        else:
            grhogvx_pl[:,:,:] = rdtype(0.0)
            grhogvy_pl[:,:,:] = rdtype(0.0)
            grhogvz_pl[:,:,:] = rdtype(0.0)
            grhogw_pl [:,:,:] = rdtype(0.0)
        #endif

        prf.PROF_rapend('____src_advection_conv_m',2)

        return
    
    def src_advection_convergence(self,
            rhogvx, rhogvx_pl,        # [IN]  rho*Vx ( G^1/2 x gam2 )
            rhogvy, rhogvy_pl,        # [IN]  rho*Vy ( G^1/2 x gam2 )
            rhogvz, rhogvz_pl,        # [IN]  rho*Vz ( G^1/2 x gam2 )
            rhogw, rhogw_pl,          # [IN]  rho*W ( G^1/2 x gam2 )
            scl, scl_pl,              # [IN]  scalar
            grhogscl, grhogscl_pl,    # [OUT] scalar tendency
            fluxtype,                 # default: horizontal & vertical convergence
            cnst, grd, oprt, vmtr, rdtype,
    ):
        
        prf.PROF_rapstart('____src_advection_conv',2)

        gall = adm.ADM_gall
        kall = adm.ADM_kall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        kminm1 = kmin - 1
        kminp1 = kmin + 1
        kmaxp1 = kmax + 1
        kmaxp2 = kmax + 2

        # --- scaled fluxes (backend-switchable kernel; numpy<->jax) ---
        xp = bk.xp
        if getattr(self, "_advconv_kernel", None) is None:
            self._advconv_cfg = AdvConvCfg(
                kmin=kmin, kmax=kmax,
                have_pl=adm.ADM_have_pl, I_SRC_default=self.I_SRC_default,
            )
            self._advconv_kernel = bk.maybe_jit(
                compute_scaled_fluxes, static_argnames=("fluxtype", "cfg", "xp"),
            )
        d = bk.device_consts(self, "advconv", lambda: {
            "afact": grd.GRD_afact,
            "bfact": grd.GRD_bfact,
        })

        (_vxscl, _vyscl, _vzscl, _wscl,
         _vxscl_pl, _vyscl_pl, _vzscl_pl, _wscl_pl) = self._advconv_kernel(
            xp.asarray(rhogvx), xp.asarray(rhogvy), xp.asarray(rhogvz),
            xp.asarray(rhogw), xp.asarray(scl),
            xp.asarray(rhogvx_pl), xp.asarray(rhogvy_pl), xp.asarray(rhogvz_pl),
            xp.asarray(rhogw_pl), xp.asarray(scl_pl),
            d["afact"], d["bfact"], fluxtype,
            cfg=self._advconv_cfg, xp=xp,
        )

        self.rhogvxscl[:, :, :, :] = bk.to_numpy(_vxscl)
        self.rhogvyscl[:, :, :, :] = bk.to_numpy(_vyscl)
        self.rhogvzscl[:, :, :, :] = bk.to_numpy(_vzscl)
        self.rhogwscl[:, :, :, :]  = bk.to_numpy(_wscl)
        if adm.ADM_have_pl:
            self.rhogvxscl_pl[:, :, :] = bk.to_numpy(_vxscl_pl)
            self.rhogvyscl_pl[:, :, :] = bk.to_numpy(_vyscl_pl)
            self.rhogvzscl_pl[:, :, :] = bk.to_numpy(_vzscl_pl)
            self.rhogwscl_pl[:, :, :]  = bk.to_numpy(_wscl_pl)

        # with open(std.fname_log, 'a') as log_file:
        #     kc=2
        #     print("before flux convergence", file=log_file)
        #     print("self.rhogvxscl (6,5,2,0)", self.rhogvxscl[6, 5, 2, 0], file=log_file) 
        #     print("self.rhogvyscl (6,5,2,0)", self.rhogvyscl[6, 5, 2, 0], file=log_file) 
        #     print("self.rhogvzscl (6,5,2,0)", self.rhogvzscl[6, 5, 2, 0], file=log_file) 
        #     print("self.rhogwscl (6,5,2,0)", self.rhogwscl[6, 5, 2, 0], file=log_file)
        #     print(f"self.rhogvxscl_pl (:,{kc},0)", self.rhogvxscl_pl[:, kc, 0], file=log_file)  #broken at 39
        #     print(f"self.rhogvyscl_pl (:,{kc},0)", self.rhogvyscl_pl[:, kc, 0], file=log_file)  #broken at 39
        #     print(f"self.rhogvzscl_pl (:,{kc},0)", self.rhogvzscl_pl[:, kc, 0], file=log_file)  #broken at 39
        #     print(f"self.rhogwscl_pl  (:,{kc},0)", self.rhogwscl_pl [:, kc, 0], file=log_file)  #broken at 39
        
        #--- flux convergence step

        self.src_flux_convergence(
                self.rhogvxscl, self.rhogvxscl_pl, 
                self.rhogvyscl, self.rhogvyscl_pl, 
                self.rhogvzscl, self.rhogvzscl_pl, 
                self.rhogwscl,  self.rhogwscl_pl,  
                grhogscl,  grhogscl_pl,  
                fluxtype, 
                cnst, grd, oprt, vmtr, rdtype, 
        )

        # with open(std.fname_log, 'a') as log_file:
        #     print("after flux convergence", file=log_file)
        #     print("grhogscl (6,5,2,0)", grhogscl[6, 5, 37, 0], file=log_file)
        #     print("grhogscl_pl (0,20,0)", grhogscl_pl[0, 37, 0], file=log_file)

        prf.PROF_rapend('____src_advection_conv',2)

        return
    
    
    # > Flux convergence calculation
    #  1. Horizontal flux convergence is calculated by using rhovx, rhovy, and
    #     rhovz which are defined at cell center (vertical) and A-grid (horizontal).
    #  2. Vertical flux convergence is calculated by using rhovx, rhovy, rhovz, and rhow.
    #  3. rhovx, rhovy, and rhovz can be replaced by rhovx*h, rhovy*h, and rhovz*h, respectively.
    def src_flux_convergence(self,
            rhogvx, rhogvx_pl,           # [IN]  
            rhogvy, rhogvy_pl,           # [IN]
            rhogvz, rhogvz_pl,           # [IN]
            rhogw,  rhogw_pl,            # [IN]
            grhog,  grhog_pl,            # [OUT]   #
            fluxtype,
            cnst, grd, oprt, vmtr, rdtype,
            resident=False,
    ):
        
        prf.PROF_rapstart('____src_flux_conv',2)

        # --- whole COMM-free body via backend-switchable kernel (numpy<->jax) ---
        xp = bk.xp
        if getattr(self, "_fluxconv_kernel", None) is None:
            self._fluxconv_cfg = FluxConvCfg(
                kmin=adm.ADM_kmin, kmax=adm.ADM_kmax,
                have_pl=adm.ADM_have_pl,
                XDIR=grd.GRD_XDIR, YDIR=grd.GRD_YDIR, ZDIR=grd.GRD_ZDIR,
                gslf_pl=adm.ADM_gslf_pl, gmax_pl=adm.ADM_gmax_pl,
                I_SRC_default=self.I_SRC_default,
                I_SRC_horizontal=self.I_SRC_horizontal,
            )
            self._fluxconv_kernel = bk.maybe_jit(
                compute_flux_convergence, static_argnames=("fluxtype", "cfg", "xp"),
            )
        d = bk.device_consts(self, "fluxconv", lambda: {
            "RGAM":       vmtr.VMTR_RGAM,
            "RGAMH":      vmtr.VMTR_RGAMH,
            "RGSQRTH":    vmtr.VMTR_RGSQRTH,
            "C2WfactGz":  vmtr.VMTR_C2WfactGz,
            "coef_div":   oprt.OPRT_coef_div,
            "rdgz":       grd.GRD_rdgz,
            "RGAM_pl":      vmtr.VMTR_RGAM_pl,
            "RGAMH_pl":     vmtr.VMTR_RGAMH_pl,
            "RGSQRTH_pl":   vmtr.VMTR_RGSQRTH_pl,
            "C2WfactGz_pl": vmtr.VMTR_C2WfactGz_pl,
            "coef_div_pl":  oprt.OPRT_coef_div_pl,
        })

        _grhog, _grhog_pl = self._fluxconv_kernel(
            xp.asarray(rhogvx), xp.asarray(rhogvy), xp.asarray(rhogvz), xp.asarray(rhogw),
            xp.asarray(rhogvx_pl), xp.asarray(rhogvy_pl), xp.asarray(rhogvz_pl), xp.asarray(rhogw_pl),
            d["RGAM"], d["RGAMH"], d["RGSQRTH"], d["C2WfactGz"], d["coef_div"], d["rdgz"],
            d["RGAM_pl"], d["RGAMH_pl"], d["RGSQRTH_pl"], d["C2WfactGz_pl"], d["coef_div_pl"],
            fluxtype, cfg=self._fluxconv_cfg, xp=xp,
        )

        if resident:
            prf.PROF_rapend('____src_flux_conv',2)
            return _grhog, _grhog_pl

        grhog[:, :, :, :] = bk.to_numpy(_grhog)
        if adm.ADM_have_pl:
            grhog_pl[:, :, :] = bk.to_numpy(_grhog_pl)

        prf.PROF_rapend('____src_flux_conv',2)

        return
    

    def src_pres_gradient(self,
        P,      P_pl,      
        Pgrad,  Pgrad_pl,     #you
        Pgradw, Pgradw_pl, 
        gradtype,
        cnst, grd, oprt, vmtr, rdtype,
        resident=False,           
    ):
        
        prf.PROF_rapstart('____src_pres_gradient',2)

        # --- whole COMM-free body via backend-switchable kernel (numpy<->jax) ---
        # (inlines OPRT_gradient + OPRT_horizontalize_vec). See kernels/presgrad.py.
        xp = bk.xp
        if getattr(self, "_presgrad_kernel", None) is None:
            self._presgrad_cfg = PresGradCfg(
                kmin=adm.ADM_kmin, kmax=adm.ADM_kmax,
                have_pl=adm.ADM_have_pl,
                XDIR=grd.GRD_XDIR, YDIR=grd.GRD_YDIR, ZDIR=grd.GRD_ZDIR,
                gslf_pl=adm.ADM_gslf_pl, gmax_pl=adm.ADM_gmax_pl,
                nxyz=adm.ADM_nxyz,
                first_layer_remedy=self.first_layer_remedy,
                rscale=float(grd.GRD_rscale),
                plmask=int(ppm.plmask),
                horizontalize=(grd.GRD_grid_type != grd.GRD_grid_type_on_plane),
                I_SRC_default=self.I_SRC_default,
                I_SRC_horizontal=self.I_SRC_horizontal,
            )
            self._presgrad_kernel = bk.maybe_jit(
                compute_pres_gradient, static_argnames=("gradtype", "cfg", "xp"),
            )
        d = bk.device_consts(self, "presgrad", lambda: {
            "RGAM":       vmtr.VMTR_RGAM,
            "RGAMH":      vmtr.VMTR_RGAMH,
            "C2WfactGz":  vmtr.VMTR_C2WfactGz,
            "coef_grad":  oprt.OPRT_coef_grad,
            "GRD_x":      grd.GRD_x,
            "rdgz":       grd.GRD_rdgz,
            "rdgzh":      grd.GRD_rdgzh,
            "GAM2H":      vmtr.VMTR_GAM2H,
            "RGSGAM2":    vmtr.VMTR_RGSGAM2,
            "RGAM_pl":      vmtr.VMTR_RGAM_pl,
            "RGAMH_pl":     vmtr.VMTR_RGAMH_pl,
            "C2WfactGz_pl": vmtr.VMTR_C2WfactGz_pl,
            "coef_grad_pl": oprt.OPRT_coef_grad_pl,
            "GRD_x_pl":     grd.GRD_x_pl,
            "GAM2H_pl":     vmtr.VMTR_GAM2H_pl,
            "RGSGAM2_pl":   vmtr.VMTR_RGSGAM2_pl,
        })

        _Pgrad, _Pgradw, _Pgrad_pl, _Pgradw_pl = self._presgrad_kernel(
            xp.asarray(P), xp.asarray(P_pl),
            d["RGAM"], d["RGAMH"], d["C2WfactGz"], d["coef_grad"], d["GRD_x"],
            d["rdgz"], d["rdgzh"], d["GAM2H"], d["RGSGAM2"],
            d["RGAM_pl"], d["RGAMH_pl"], d["C2WfactGz_pl"], d["coef_grad_pl"],
            d["GRD_x_pl"], d["GAM2H_pl"], d["RGSGAM2_pl"],
            gradtype, cfg=self._presgrad_cfg, xp=xp,
        )

        if resident:
            prf.PROF_rapend('____src_pres_gradient',2)
            return _Pgrad, _Pgradw, _Pgrad_pl, _Pgradw_pl

        Pgrad[:, :, :, :, :] = bk.to_numpy(_Pgrad)
        Pgradw[:, :, :, :]   = bk.to_numpy(_Pgradw)
        if adm.ADM_have_pl:
            Pgrad_pl[:, :, :, :] = bk.to_numpy(_Pgrad_pl)
            Pgradw_pl[:, :, :]   = bk.to_numpy(_Pgradw_pl)

        prf.PROF_rapend('____src_pres_gradient',2)

        return

    #> Buoyacy force
    #> Note: Upward direction is positive for buoiw.
    def src_buoyancy(self,
        rhog,  rhog_pl,          # [IN]
        buoiw, buoiw_pl,         # [OUT]
        cnst, vmtr, rdtype,
        resident=False,
    ):
    
        prf.PROF_rapstart('____src_buoyancy',2)

        # Backend-switchable kernel (numpy <-> jax). See kernels/buoyancy.py.
        # Static config + device-staged constants are built once (flavor-B
        # explicit one-time staging). The [OUT] in-place contract is preserved,
        # so the mod_vi.py call site is unchanged.
        xp = bk.xp
        if getattr(self, "_buoy_kernel", None) is None:
            self._buoy_cfg = BuoyCfg(
                kmin=adm.ADM_kmin, kmax=adm.ADM_kmax,
                GRAV=float(cnst.CONST_GRAV), have_pl=adm.ADM_have_pl,
            )
            self._buoy_kernel = bk.maybe_jit(
                compute_buoyancy, static_argnames=("cfg", "xp"))
        d = bk.device_consts(self, "buoy", lambda: {
            "C2Wfact":    vmtr.VMTR_C2Wfact,
            "C2Wfact_pl": vmtr.VMTR_C2Wfact_pl,
        })

        _buoiw, _buoiw_pl = self._buoy_kernel(
            xp.asarray(rhog), xp.asarray(rhog_pl),
            d["C2Wfact"], d["C2Wfact_pl"],
            cfg=self._buoy_cfg, xp=xp,
        )
        if resident:
            prf.PROF_rapend('____src_buoyancy',2)
            return _buoiw, _buoiw_pl

        buoiw[:, :, :, :] = bk.to_numpy(_buoiw)
        if adm.ADM_have_pl:
            buoiw_pl[:, :, :] = bk.to_numpy(_buoiw_pl)

        prf.PROF_rapend('____src_buoyancy',2)

        return
    