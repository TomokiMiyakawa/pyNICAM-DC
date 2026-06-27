import os
import toml
import numpy as np
#from mpi4py import MPI
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.share.mod_prof import prf
from pynicamdc.share.mod_backend import backend as bk
from pynicamdc.nhm.dynamics.kernels.oprtdivdamp import OprtDivdampCfg
from pynicamdc.nhm.dynamics.kernels.horizontalizevec import HorizontalizeVecCfg
from pynicamdc.nhm.dynamics.kernels.divdamppostcomm import compute_divdamp_post_comm


class Numf:
    
    _instance = None
    
    # Numerical filter options
    NUMFILTER_DOrayleigh            = False  # Use Rayleigh damping?
    NUMFILTER_DOhorizontaldiff      = False  # Use horizontal diffusion?
    NUMFILTER_DOhorizontaldiff_lap1 = False  # Use horizontal 1st-order damping? (for upper layer)
    NUMFILTER_DOverticaldiff        = False  # Use vertical diffusion?
    NUMFILTER_DOdivdamp             = False  # Use 3D divergence damping?
    NUMFILTER_DOdivdamp_v           = False  # Use 3D divergence damping for vertical velocity?
    NUMFILTER_DOdivdamp_2d          = False  # Use 2D divergence damping?

    rayleigh_damp_only_w = False  # Damp only w?

    debug = False

    def __init__(self):
        pass

    #def numfilter_setup(self, fname_in, rcnf, cnst, comm, gtl, grd, gmtr, oprt, vmtr, tim, prgv, tdyn, frc, bndc, bsst, rdtype):
    def numfilter_setup(self, fname_in, rcnf, cnst, comm, gtl, grd, gmtr, oprt, vmtr, tim, prgv, tdyn, bndc, bsst, rdtype):
        
        self.lap_order_hdiff = 2  # Laplacian order for horizontal diffusion
        self.hdiff_fact_rho  = rdtype(1.0e-2)
        self.hdiff_fact_q    = rdtype(0.0)
        self.Kh_coef_minlim  = rdtype(0.0)
        self.Kh_coef_maxlim  = rdtype(1.0e+30)

        self.hdiff_nonlinear = False
        self.ZD_hdiff_nl     = rdtype(25000.0)  # Height for decay of nonlinear diffusion

        self.lap_order_divdamp = 2
        self.divdamp_coef_v    = rdtype(0.0)

        # 2D divergence damping coefficients
        self.lap_order_divdamp_2d = 1

        # Grid-related flags and parameters
        self.dep_hgrid = False      # Depends on horizontal grid spacing?
        self.AREA_ave  = None       # Averaged grid area

        self.smooth_1var = True     # Should be False for stretched grid (according to S.Iga)

        deep_effect = False

        # Rayleigh damping
        self.alpha_r = rdtype(0.0)                  # Coefficient for Rayleigh damping
        self.ZD = rdtype(25000.0)                   # Lower limit of Rayleigh damping [m]

        # Horizontal diffusion
        self.hdiff_type = 'NONDIM_COEF'                 # Diffusion type
        self.gamma_h = rdtype(1.0) / rdtype(16.0) / rdtype(10.0)    # Coefficient for horizontal diffusion
        self.tau_h = rdtype(160000.0)               # E-folding time for horizontal diffusion [sec]

        # Horizontal diffusion (1st-order Laplacian)
        self.hdiff_type_lap1 = 'DIRECT'                 # Diffusion type
        self.gamma_h_lap1 = rdtype(0.0)             # Height-dependent gamma_h (1st-order)
        self.tau_h_lap1 = rdtype(160000.0)          # Height-dependent tau_h (1st-order) [sec]
        self.ZD_hdiff_lap1 = rdtype(25000.0)        # Lower limit of horizontal diffusion [m]

        # Vertical diffusion
        self.gamma_v = rdtype(0.0)                  # Coefficient for vertical diffusion

        # 3D divergence damping
        self.divdamp_type = 'NONDIM_COEF'               # Damping type
        self.alpha_d = rdtype(0.0)                  # Coefficient for divergence damping
        self.tau_d = rdtype(132800.0)               # E-folding time for divergence damping [sec]
        self.alpha_dv = rdtype(0.0)                 # Vertical coefficient

        # 2D divergence damping
        self.divdamp_2d_type = 'NONDIM_COEF'            # Damping type
        self.alpha_d_2d = rdtype(0.0)               # Coefficient for 2D divergence damping
        self.tau_d_2d = rdtype(1328000.0)           # E-folding time for 2D divergence damping [sec]
        self.ZD_d_2d = rdtype(25000.0)              # Lower limit of divergence damping [m]

        PI = cnst.CONST_PI
        RADIUS = cnst.CONST_RADIUS

        self.global_area = rdtype(4.0) * PI * RADIUS * RADIUS
        self.global_grid = rdtype(10.0) * (rdtype(4.0) ** adm.ADM_glevel)
        self.AREA_ave = self.global_area / self.global_grid

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[numfilter]/Category[nhm dynamics]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'numfilterparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** numfilterparam not found in toml file! Use default.", file=log_file)
                #prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['numfilterparam']
            self.hdiff_type  = cnfs['hdiff_type']
            self.lap_order_hdiff = cnfs['lap_order_hdiff']
            self.gamma_h = cnfs['gamma_h']
            self.divdamp_type = cnfs['divdamp_type']
            self.lap_order_divdamp = cnfs['lap_order_divdamp']
            self.alpha_d = cnfs['alpha_d']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        # skip for now
        # call numfilter_rayleigh_damping_setup( alpha_r, & ! [IN]                                                                                                  
        #                                     ZD       ) ! [IN]                                                                                                  

        # used in JW06
        self.numfilter_hdiffusion_setup(rcnf, cnst, comm, gtl, grd, gmtr, oprt, tim, rdtype)
        # call numfilter_hdiffusion_setup( hdiff_type,      & ! [IN]                                                                                                
        #                                 dep_hgrid,       & ! [IN]                                                                                                
        #                                 smooth_1var,     & ! [IN]                                                                                                
        #                                 lap_order_hdiff, & ! [IN]                                                                                                
        #                                 gamma_h,         & ! [IN]                                                                                                
        #                                 tau_h,           & ! [IN]                                                                                                
        #                                 hdiff_type_lap1, & ! [IN]                                                                                                
        #                                 gamma_h_lap1,    & ! [IN]                                                                                                
        #                                 tau_h_lap1,      & ! [IN]                                                                                                
        #                                 ZD_hdiff_lap1    ) ! [IN]                                                                                                

        # skip for now
        # call numfilter_vdiffusion_setup( gamma_v ) ! [IN]                                                                                                         

        # used in JW06
        self.numfilter_divdamp_setup(rcnf, cnst, comm, gtl, grd, gmtr, oprt, tim, rdtype)
        # call numfilter_divdamp_setup( divdamp_type,      & ! [IN]                                                                                                 
        #                             dep_hgrid,         & ! [IN]                                                                                                 
        #                             smooth_1var,       & ! [IN]                                                                                                 
        #                             lap_order_divdamp, & ! [IN]                                                                                                 
        #                             alpha_d,           & ! [IN]                                                                                                 
        #                             tau_d,             & ! [IN]                                                                                                 
        #                             alpha_dv           ) ! [IN]                                                                                                 

        # used even if the message says unused! (from orginal code)  
        self.numfilter_divdamp_2d_setup(rcnf, cnst, comm, gtl, grd, gmtr, oprt, tim, rdtype)
        # call numfilter_divdamp_2d_setup( divdamp_2d_type,      & ! [IN]                                                                                           
        #                                 dep_hgrid,            & ! [IN]                                                                                           
        #                                 lap_order_divdamp_2d, & ! [IN]                                                                                           
        #                                 alpha_d_2d,           & ! [IN]                                                                                           
        #                                 tau_d_2d,             & ! [IN]                                                                                           
        #                                 ZD_d_2d               ) ! [IN]                                                                                           

        Kh_deep_factor        = np.zeros(adm.ADM_kall, dtype=rdtype)
        Kh_deep_factor_h      = np.zeros(adm.ADM_kall, dtype=rdtype)
        Kh_lap1_deep_factor   = np.zeros(adm.ADM_kall, dtype=rdtype)
        Kh_lap1_deep_factor_h = np.zeros(adm.ADM_kall, dtype=rdtype)
        divdamp_deep_factor   = np.zeros(adm.ADM_kall, dtype=rdtype)

        if deep_effect:
            print("Sorry, deep_effect is not implemented yet.")
            prc.prc_mpistop(std.io_l, std.fname_log)
            # do k = 1, ADM_kall
            #         Kh_deep_factor       (k) = ( (GRD_gz (k)+RADIUS) / RADIUS )**(2*lap_order_hdiff)
            #         Kh_deep_factor_h     (k) = ( (GRD_gzh(k)+RADIUS) / RADIUS )**(2*lap_order_hdiff)
            #         Kh_lap1_deep_factor  (k) = ( (GRD_gz (k)+RADIUS) / RADIUS )**2
            #         Kh_lap1_deep_factor_h(k) = ( (GRD_gzh(k)+RADIUS) / RADIUS )**2
            #         divdamp_deep_factor  (k) = ( (GRD_gz (k)+RADIUS) / RADIUS )**(2*lap_order_divdamp)
            # enddo

        return
    
    def numfilter_hdiffusion_setup(self, rcnf, cnst, comm, gtl, grd, gmtr, oprt, tim, rdtype):

        PI = cnst.CONST_PI
        EPS = cnst.CONST_EPS

        lap_order = self.lap_order_hdiff
        gamma     = self.gamma_h
        tau       = self.tau_h
        gamma_lap1 = self.gamma_h_lap1
        tau_lap1   = self.tau_h_lap1
        zlimit_lap1 = self.ZD_hdiff_lap1

        e_fold_time    = np.zeros((adm.ADM_shape),    dtype=rdtype)
        e_fold_time_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)

        self.Kh_coef    = np.zeros((adm.ADM_shape),    dtype=rdtype)
        self.Kh_coef_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)

        if self.hdiff_type == 'DIRECT':
            if gamma > rdtype(0.0):
                self.NUMFILTER_DOhorizontaldiff = True

            # gamma is an absolute value
            self.Kh_coef[:, :, :, :] = gamma
            self.Kh_coef_pl[:, :, :] = gamma

        elif self.hdiff_type == 'NONDIM_COEF':
            if gamma > rdtype(0.0):
                self.NUMFILTER_DOhorizontaldiff = True

            large_step_dt = tim.TIME_DTL / rdtype(rcnf.DYN_DIV_NUM)

            # gamma is a non-dimensional number
            if self.dep_hgrid:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kall):
                        self.Kh_coef[:, :, k, l] = gamma / large_step_dt * gmtr.GMTR_area[:, :, l] ** lap_order

                if adm.ADM_have_pl:
                    #for l in range(adm.ADM_lall_pl):
                    for k in range(adm.ADM_kall):
                        self.Kh_coef_pl[:, k, :] = gamma / large_step_dt * gmtr.GMTR_area_pl[:, :] ** lap_order
            else:
                value = gamma / large_step_dt * self.AREA_ave ** lap_order
                self.Kh_coef[:, :, :, :] = value
                self.Kh_coef_pl[:, :, :] = value

        elif self.hdiff_type == 'E_FOLD_TIME':
            if tau > rdtype(0.0):
                self.NUMFILTER_DOhorizontaldiff = True

            # tau is e-folding time for 2*dx waves
            if self.dep_hgrid:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kall):
                        self.Kh_coef[:, :, k, l] = (np.sqrt(gmtr.GMTR_area[:, :, l]) / PI) ** (2 * lap_order) / (tau + EPS)

                if adm.ADM_have_pl:
                    #for l in range(adm.ADM_lall_pl):
                    for k in range(adm.ADM_kall):
                        self.Kh_coef_pl[:, k, :] = (np.sqrt(gmtr.GMTR_area_pl[:, :]) / PI) ** (2 * lap_order) / (tau + EPS)
            else:
                value = (np.sqrt(self.AREA_ave) / PI) ** (2 * lap_order) / (tau + EPS)
                self.Kh_coef[:, :, :, :] = value
                self.Kh_coef_pl[:, :, :] = value

        elif self.hdiff_type == 'NONLINEAR1':
            self.NUMFILTER_DOhorizontaldiff = True
            self.hdiff_nonlinear = True

            self.Kh_coef[:, :, :, :] = rdtype(-999.0)
            self.Kh_coef_pl[:, :, :] = rdtype(-999.0)

        #print("self.hdifftype: ", self.hdiff_type)

        if self.hdiff_type != 'DIRECT' and self.hdiff_type != 'NONLINEAR1':
            if self.smooth_1var:  # Iga 20120721 (add if)
                self.numfilter_smooth_1var(self.Kh_coef, self.Kh_coef_pl, comm, gmtr, oprt, rdtype)

            self.Kh_coef[:, :, :, :] = np.maximum(self.Kh_coef, self.Kh_coef_minlim)


        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print("-----   Horizontal numerical diffusion   -----", file=log_file)

        if self.NUMFILTER_DOhorizontaldiff:
            if not self.hdiff_nonlinear:
                if self.debug:
                    for l in range(adm.ADM_lall):
                        for k in range(adm.ADM_kall):
                            e_fold_time[:, :, k, l] = (np.sqrt(gmtr.GMTR_area[:, :, l]) / PI) ** (2 * lap_order) / (self.Kh_coef[:, :, k, l] + EPS)

                    if adm.ADM_have_pl:
                        #for l in range(adm.ADM_lall_pl):
                        for k in range(adm.ADM_kall):
                            e_fold_time_pl[:, k, :] = (np.sqrt(gmtr.GMTR_area_pl[:, :]) / PI) ** (2 * lap_order) / (self.Kh_coef_pl[:, k, :] + EPS)

                    if std.io_l:
                        with open(std.fname_log, 'a') as log_file:
                            print("    z[m]      max coef      min coef  max eft(2DX)  min eft(2DX)", file=log_file)

                    for k in range(adm.ADM_kmax, adm.ADM_kmin - 1, -1):
                        eft_max  = gtl.GTL_max_k(e_fold_time, e_fold_time_pl, k)
                        eft_min  = gtl.GTL_min_k(e_fold_time, e_fold_time_pl, k)
                        coef_max = gtl.GTL_max_k(self.Kh_coef, self.Kh_coef_pl, k)
                        coef_min = gtl.GTL_min_k(self.Kh_coef, self.Kh_coef_pl, k)
                        if std.io_l:
                            with open(std.fname_log, 'a') as log_file:
                                print(f" {grd.GRD_gz[k]:8.2f}{coef_min:14.6e}{coef_max:14.6e}{eft_max:14.6e}{eft_min:14.6e}", file=log_file)
                else:
                    if std.io_l:
                        with open(std.fname_log, 'a') as log_file:
                            print("=> used.", file=log_file)

            else:
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print("=> Nonlinear filter is used.", file=log_file)
        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("=> not used.", file=log_file)

        # Allocate and initialize Kh_coef_lap1 arrays
        self.Kh_coef_lap1    = np.zeros((adm.ADM_shape), dtype=rdtype)
        self.Kh_coef_lap1_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)


        if self.hdiff_type_lap1 == 'DIRECT':
            if gamma_lap1 > rdtype(0.0):
                self.NUMFILTER_DOhorizontaldiff_lap1 = True

            # gamma is an absolute value
            self.Kh_coef_lap1[:, :, :, :]    = gamma_lap1
            self.Kh_coef_lap1_pl[:, :, :] = gamma_lap1

        elif self.hdiff_type_lap1 == 'NONDIM_COEF':
            if gamma_lap1 > rdtype(0.0):
                self.NUMFILTER_DOhorizontaldiff_lap1 = True

            large_step_dt = tim.TIME_DTL / rdtype(rcnf.DYN_DIV_NUM)

            # gamma is a non-dimensional number
            if self.dep_hgrid:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kall):
                        self.Kh_coef_lap1[:, :, k, l] = gamma_lap1 / large_step_dt * gmtr.GMTR_area[:, :, l]

                if adm.ADM_have_pl:
                    #for l in range(adm.ADM_lall_pl):
                    for k in range(adm.ADM_kall):
                        self.Kh_coef_lap1_pl[:, k, :] = gamma_lap1 / large_step_dt * gmtr.GMTR_area_pl[:, :]
            else:
                value = gamma_lap1 / large_step_dt * self.AREA_ave
                self.Kh_coef_lap1[:, :, :, :]    = value
                self.Kh_coef_lap1_pl[:, :, :] = value

        elif self.hdiff_type_lap1 == 'E_FOLD_TIME':
            if tau_lap1 > rdtype(0.0):
                self.NUMFILTER_DOhorizontaldiff_lap1 = True

            # tau is e-folding time for 2*dx waves
            if self.dep_hgrid:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kall):
                        self.Kh_coef_lap1[:, :, k, l] = (np.sqrt(gmtr.GMTR_area[:, :, l]) / PI) ** 2 / (tau_lap1 + EPS)

                if adm.ADM_have_pl:
                    #for l in range(adm.ADM_lall_pl):
                    for k in range(adm.ADM_kall):
                        self.Kh_coef_lap1_pl[:, k, :] = (np.sqrt(gmtr.GMTR_area_pl[:, :]) / PI) ** 2 / (tau_lap1 + EPS)
            else:
                value = (np.sqrt(self.AREA_ave) / PI) ** 2 / (tau_lap1 + EPS)
                self.Kh_coef_lap1[:, :, :, :] = value
                self.Kh_coef_lap1_pl[:, :, :] = value


        # Apply height factor
        fact = np.full(adm.ADM_kall, cnst.CONST_UNDEF, dtype=rdtype)
        self.height_factor(adm.ADM_kall, grd.GRD_gz, grd.GRD_htop, zlimit_lap1, fact, cnst, rdtype)

        for l in range(adm.ADM_lall):
            for k in range(adm.ADM_kall):
                self.Kh_coef_lap1[:, :, k, l] *= fact[k]

        if adm.ADM_have_pl:
            #for l in range(adm.ADM_lall_pl):
            for k in range(adm.ADM_kall):
                self.Kh_coef_lap1_pl[:, k, :] *= fact[k]

        # Logging
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print("-----   Horizontal numerical diffusion (1st order laplacian)   -----", file=log_file)

        if self.NUMFILTER_DOhorizontaldiff_lap1:
            if self.debug:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kall):
                        e_fold_time[:, :, k, l] = (np.sqrt(gmtr.GMTR_area[:, :, l]) / PI) ** 2 / (self.Kh_coef_lap1[:, :, k, l] + EPS)

                if adm.ADM_have_pl:
                    #for l in range(adm.ADM_lall_pl):
                    for k in range(adm.ADM_kall):
                        e_fold_time_pl[:, k, :] = (np.sqrt(gmtr.GMTR_area_pl[:, :]) / PI) ** 2 / (self.Kh_coef_lap1_pl[:, k, :] + EPS)

                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print("    z[m]      max coef      min coef  max eft(2DX)  min eft(2DX)", file=log_file)

                for k in range(adm.ADM_kmax, adm.ADM_kmin - 1, -1):  # range not checked
                    eft_max  = gtl.GTL_max_k(e_fold_time, e_fold_time_pl, k)
                    eft_min  = gtl.GTL_min_k(e_fold_time, e_fold_time_pl, k)
                    coef_max = gtl.GTL_max_k(self.Kh_coef_lap1, self.Kh_coef_lap1_pl, k)
                    coef_min = gtl.GTL_min_k(self.Kh_coef_lap1, self.Kh_coef_lap1_pl, k)
                    if std.io_l:
                        with open(std.fname_log, 'a') as log_file:
                            print(f" {grd.GRD_gz[k]:8.2f}{coef_min:14.6e}{coef_max:14.6e}{eft_max:14.6e}{eft_min:14.6e}", file=log_file)
            else:
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print("=> used.", file=log_file)
        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("=> not used.", file=log_file)


        return
    
    def numfilter_divdamp_setup(self, rcnf, cnst, comm, gtl, grd, gmtr, oprt, tim, rdtype):

        PI = cnst.CONST_PI
        EPS = cnst.CONST_EPS
        SOUND = cnst.CONST_SOUND

        lap_order = self.lap_order_divdamp
        alpha = self.alpha_d
        tau   = self.tau_d
        alpha_v = self.alpha_dv

        e_fold_time    = np.zeros((adm.ADM_shape),    dtype=rdtype)
        e_fold_time_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)

        self.divdamp_coef    = np.zeros((adm.ADM_shape),    dtype=rdtype)
        self.divdamp_coef_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)


        if self.divdamp_type == 'DIRECT':
            if alpha > rdtype(0.0):
                self.NUMFILTER_DOdivdamp = True

            # alpha_d is an absolute value.
            coef = alpha

            # with open(std.fname_log, 'a') as log_file:
            #     print("coef: ", coef, self.alpha_d, file=log_file)
            #     print("self.divdamp_coef[:, :, :, :] = coef", file=log_file)
            #     print("self.divdamp_coef_pl[:, :, :] = coef", file=log_file)
            # print("coef: ", coef)

            self.divdamp_coef[:, :, :, :] = coef
            self.divdamp_coef_pl[:, :, :] = coef

            # for l in range(adm.ADM_lall):
            #     for k in range(0,3): #adm.ADM_kall):
            #         print(f"self.divdamp_coef[:, :, {k}, {l}]")
            #         print(self.divdamp_coef[:, :, k, l])
            #prc.prc_mpistop(std.io_l, std.fname_log)


        elif self.divdamp_type == 'NONDIM_COEF':
            if alpha > rdtype(0.0):
                self.NUMFILTER_DOdivdamp = True

            small_step_dt = tim.TIME_DTS / rdtype(rcnf.DYN_DIV_NUM)

            # alpha_d is a non-dimensional number.
            # alpha_d * (c_s)^p * dt^{2p-1}
            coef = alpha * (SOUND * SOUND)**lap_order * small_step_dt**(2 * lap_order - 1)

            self.divdamp_coef[:, :, :, :] = coef
            self.divdamp_coef_pl[:, :, :] = coef

        elif self.divdamp_type == 'E_FOLD_TIME':
            if tau > rdtype(0.0):
                self.NUMFILTER_DOdivdamp = True

            # tau_d is e-folding time for 2*dx.
            if self.dep_hgrid:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kall):
                        self.divdamp_coef[:, :, k, l] = (np.sqrt(gmtr.GMTR_area[:, :, l]) / PI)**(2 * lap_order) / (tau + EPS)

                if adm.ADM_have_pl:
                    #for l in range(adm.ADM_lall_pl):
                    for k in range(adm.ADM_kall):
                        self.divdamp_coef_pl[:, k, :] = (np.sqrt(gmtr.GMTR_area_pl[:, :]) / PI)**(2 * lap_order) / (tau + EPS)
            else:
                coef = (np.sqrt(self.AREA_ave) / PI)**(2 * lap_order) / (tau + EPS)

                self.divdamp_coef[:, :, :, :] = coef
                self.divdamp_coef_pl[:, :, :] = coef

        #print("self.divdamp_type: ", self.divdamp_type)
        if self.divdamp_type != 'DIRECT':
            if self.smooth_1var:
                self.numfilter_smooth_1var(self.divdamp_coef, self.divdamp_coef_pl)

            self.divdamp_coef[:, :, :, :] = np.maximum(self.divdamp_coef, self.Kh_coef_minlim)

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print("-----   3D divergence damping   -----", file=log_file)

        if self.NUMFILTER_DOdivdamp:
            if self.debug:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kall):
                        e_fold_time[:, :, k, l] = (np.sqrt(gmtr.GMTR_area[:, :, l]) / PI)**(2 * lap_order) / (self.divdamp_coef[:, :, k, l] + EPS)

                e_fold_time_pl[:, :, :] = rdtype(0.0)

                if adm.ADM_have_pl:
                    #for l in range(adm.ADM_lall_pl):
                    for k in range(adm.ADM_kall):
                        e_fold_time_pl[:, k, :] = (np.sqrt(gmtr.GMTR_area_pl[:, :]) / PI)**(2 * lap_order) / (self.divdamp_coef_pl[:, k, :] + EPS)

                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print('    z[m]      max coef      min coef  max eft(2DX)  min eft(2DX)', file=log_file)

                for k in range(adm.ADM_kmax, adm.ADM_kmin - 1, -1):   # range not checked
                    eft_max = gtl.GTL_max_k(e_fold_time, e_fold_time_pl, k)
                    eft_min = gtl.GTL_min_k(e_fold_time, e_fold_time_pl, k)
                    coef_max = gtl.GTL_max_k(self.divdamp_coef, self.divdamp_coef_pl, k)
                    coef_min = gtl.GTL_min_k(self.divdamp_coef, self.divdamp_coef_pl, k)
                    if std.io_l:
                        with open(std.fname_log, 'a') as log_file:
                            print(f' {grd.GRD_gz[k]:8.2f} {coef_min:14.6e} {coef_max:14.6e} {eft_max:14.6e} {eft_min:14.6e}', file=log_file)
            else:
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print('=> used.', file=log_file)
        else:
            if std.io_l:    
                with open(std.fname_log, 'a') as log_file:
                    print('=> not used.', file=log_file)

        if alpha_v > rdtype(0.0):
            self.NUMFILTER_DOdivdamp_v = True

        small_step_dt = tim.TIME_dts / float(rcnf.DYN_DIV_NUM)
        self.divdamp_coef_v = -alpha_v * SOUND * SOUND * small_step_dt

        return

    def numfilter_divdamp_2d_setup(self, rcnf, cnst, comm, gtl, grd, gmtr, oprt, tim, rdtype):    
 
        PI = cnst.CONST_PI
        EPS = cnst.CONST_EPS
        SOUND = cnst.CONST_SOUND

        divdamp_type = self.divdamp_2d_type
        dep_hgrid = self.dep_hgrid
        lap_order = self.lap_order_divdamp_2d
        alpha = self.alpha_d_2d
        tau   = self.tau_d_2d
        zlimit = self.ZD_d_2d
        #alpha_v = self.alpha_dv

        self.divdamp_2d_coef    = np.zeros((adm.ADM_shape), dtype=rdtype)
        self.divdamp_2d_coef_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)
        e_fold_time    = np.zeros((adm.ADM_shape), dtype=rdtype)
        e_fold_time_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)
        fact = np.full(adm.ADM_kall, cnst.CONST_UNDEF, dtype=rdtype)

        if divdamp_type == 'DIRECT':
            if alpha > rdtype(0.0):
                self.NUMFILTER_DOdivdamp_2d = True
            # endif

            coef = alpha
            self.divdamp_2d_coef[:, :, :, :] = coef
            self.divdamp_2d_coef_pl[:, :, :] = coef

        elif divdamp_type == 'NONDIM_COEF':
            if alpha > rdtype(0.0):
                self.NUMFILTER_DOdivdamp_2d = True
            #endif

            small_step_dt = tim.TIME_dts / rdtype(rcnf.DYN_DIV_NUM)

            # alpha is a non-dimensional number
            coef = alpha * (SOUND * SOUND) ** lap_order * small_step_dt ** (2 * lap_order - 1)
            self.divdamp_2d_coef[:, :, :, :] = coef
            self.divdamp_2d_coef_pl[:, :, :] = coef

        elif divdamp_type == 'E_FOLD_TIME':
            if tau > rdtype(0.0):
                self.NUMFILTER_DOdivdamp_2d = True
            #endif

            if dep_hgrid:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kall):
                        self.divdamp_2d_coef[:, :, k, l] = (
                            (np.sqrt(gmtr.GMTR_area[:, :, l]) / np.pi) ** (2 * lap_order)
                        ) / (tau + EPS)
                    # end k loop
                # end l loop

                if adm.ADM_have_pl:
                    #for l in range(adm.ADM_lall_pl):
                    for k in range(adm.ADM_kall):
                        self.divdamp_2d_coef_pl[:, k, :] = (
                            (np.sqrt(gmtr.GMTR_area_pl[:, :]) / np.pi) ** (2 * lap_order)
                        ) / (tau + EPS)
                        # end k loop
                    # end l loop    
                # end if
            else:
                coef = (np.sqrt(self.AREA_ave) / np.pi) ** (2 * lap_order) / (tau + EPS)
                self.divdamp_2d_coef[:, :, :, :] = coef
                self.divdamp_2d_coef_pl[:, :, :] = coef
            #endif
        #endif

        self.height_factor(adm.ADM_kall, grd.GRD_gz, grd.GRD_htop, zlimit, fact, cnst, rdtype)
        # call height_factor( ADM_kall, GRD_gz(:), GRD_htop, zlimit, fact(:) )

        # for l in range(adm.ADM_lall):
        #     for k in range(adm.ADM_kall):
        self.divdamp_2d_coef[:, :, :, :] *= fact[:][None, None, :, None]
            # end k loop
        # end l loop

        if adm.ADM_have_pl:
            #for l in range(adm.ADM_lall_pl):
            for k in range(adm.ADM_kall):
                self.divdamp_2d_coef_pl[:, k, :] *= fact[k]
                # end k loop
            # end l loop
        # end if


        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print("-----   2D divergence damping   -----", file=log_file)

        if self.NUMFILTER_DOdivdamp_2d:
            if self.debug:
                # Compute e-folding time for the main domain
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kall):
                        e_fold_time[:, :, k, l] = (
                            (np.sqrt(gmtr.GMTR_area[:, :, l]) / np.pi) ** (2 * self.lap_order_divdamp)
                            / (self.divdamp_2d_coef[:, :, k, l] + EPS)
                        )

                # Compute e-folding time for pole region
                if adm.ADM_have_pl:
                    #for l in range(adm.ADM_lall_pl):
                    for k in range(adm.ADM_kall):
                        e_fold_time_pl[:, k, :] = (
                            (np.sqrt(gmtr.GMTR_area_pl[:, :]) / np.pi) ** (2 * self.lap_order_divdamp)
                            / (self.divdamp_2d_coef_pl[:, k, :] + EPS)
                        )
                else:
                    e_fold_time_pl[:, :, :] = rdtype(0.0)

                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print("    z[m]      max coef      min coef  max eft(2DX)  min eft(2DX)", file=log_file)

                for k in range(adm.ADM_kmax, adm.ADM_kmin - 1, -1):
                    eft_max = gtl.GTL_max_k(e_fold_time, e_fold_time_pl, k)
                    eft_min = gtl.GTL_min_k(e_fold_time, e_fold_time_pl, k)
                    coef_max = gtl.GTL_max_k(self.divdamp_2d_coef, self.divdamp_2d_coef_pl, k)
                    coef_min = gtl.GTL_min_k(self.divdamp_2d_coef, self.divdamp_2d_coef_pl, k)

                    if std.io_l:
                        with open(std.fname_log, 'a') as log_file:
                            print(f"{grd.GRD_gz[k]:8.2f}{coef_min:14.6e}{coef_max:14.6e}{eft_max:14.6e}{eft_min:14.6e}", file=log_file)

            else:
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print("=> used.", file=log_file)
        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("=> not used.", file=log_file)
 
        return

    def numfilter_smooth_1var(self, s, s_pl, comm, gmtr, oprt, rdtype):

        vtmp     = np.zeros((adm.ADM_shape    + (1,)), dtype=rdtype)
        vtmp_pl  = np.zeros((adm.ADM_shape_pl + (1,)), dtype=rdtype)
        vtmp2    = np.zeros((adm.ADM_shape    + (1,)), dtype=rdtype)
        vtmp2_pl = np.zeros((adm.ADM_shape_pl + (1,)), dtype=rdtype)

        # Constants
        ggamma_h = rdtype(1.0) / rdtype(16.0) / rdtype(10.0)
        itelim   = 80

        gall_1d = adm.ADM_gall_1d
        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kall = adm.ADM_kall

        #print("itelim=", itelim)

        for ite in range(itelim):
            
            #print(f"ite: {ite}")

            vtmp[:, :, :, :, 0] = s
            if adm.ADM_have_pl:
                vtmp_pl[:, :, :, 0] = s_pl

            comm.COMM_data_transfer(vtmp, vtmp_pl)

            for p in range(2):
                vtmp2[:, :, :, :, :] = rdtype(0.0)
                vtmp2_pl[:, :, :, :] = rdtype(0.0)

                vtmp2[:, :, :, :, 0], vtmp2_pl[:, :, :, 0] = oprt.OPRT_laplacian(
                    vtmp[:, :, :, :, 0], vtmp_pl[:, :, :, 0], 
                    oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl,  rdtype
                )

                comm.COMM_data_transfer(vtmp, vtmp_pl)

            # for i in range(gall_1d):
            #     for j in range(gall_1d):
            for k in range(kall):
                for l in range(adm.ADM_lall):

                    isl = slice(0, iall)
                    jsl = slice(0, jall)

                    s[isl, jsl, k, l] -= (
                        ggamma_h * gmtr.GMTR_area[isl, jsl, l]**2 * vtmp[isl, jsl, k, l, 0]
                    )

                            #s[i, j, k, l] -= ggamma_h * gmtr.GMTR_area[i, j, l] ** 2 * vtmp[i, j, k, l, 0]

            if adm.ADM_have_pl:
                for g in range(adm.ADM_gall_pl):
                    for k in range(adm.ADM_kall):
                        for l in range(adm.ADM_lall_pl):
                            s_pl[g, k, l] -= ggamma_h * gmtr.GMTR_area_pl[g, l] ** 2 * vtmp_pl[g, k, l, 0]

        vtmp[:, :, :, :, 0] = s
        vtmp_pl[:, :, :, 0] = s_pl

        comm.COMM_data_transfer(vtmp, vtmp_pl)

        s[:, :, :, :] = vtmp[:, :, :, :, 0]
        s_pl[:, :, :] = vtmp_pl[:, :, :, 0]

        return
    

    def height_factor(self, kdim, z, z_top, z_bottomlimit, factor, cnst, rdtype):
    
        PI = cnst.CONST_PI

        for k in range(kdim):
            sw = rdtype(0.5) + np.sign(z[k] - z_bottomlimit) * rdtype(0.5)

            factor[k] = sw * rdtype(0.5) * (
                rdtype(1.0) - np.cos(PI * (z[k] - z_bottomlimit) / (z_top - z_bottomlimit))
            )

        return

    def numfilter_hdiffusion(self,
        rhog,       rhog_pl,            # [IN]
        rho,        rho_pl,             # [IN]
        vx,         vx_pl,              # [IN]
        vy,         vy_pl,              # [IN]
        vz,         vz_pl,              # [IN]  
        w,          w_pl,               # [IN]
        tem,        tem_pl,             # [IN]
        q,          q_pl,               # [IN]
        tendency,   tendency_pl,        # [OUT]    #you
        tendency_q, tendency_q_pl,      # [OUT]
        cnst, comm, grd, oprt, vmtr, tim, rcnf, bsst, rdtype,
        prog_d=None, diag_d=None, rho_d=None,   # [IN] optional device-resident PROG/DIAG/rho (RESIDENT_PROG)
        stash_device=False,                     # [IN] stash device f_TEND components for the caller g_TEND assembly (RES-CAPSTONE Phase A)
    ):
        
        prf.PROF_rapstart('____numfilter_hdiffusion',2)
        prf.PROF_rapstart('_____hdiff_setup',2)   # scratch alloc + vtmp pack (decompose the block)
        prf.PROF_rapstart('______hdiff_set_alloc',2)

        # Scratch buffers. Hoist (gated PYNICAM_HDIFF_HOIST): allocate once on
        # self and reuse every call -- the np.full UNDEF-fill of these ~13 large
        # arrays per call is pure setup churn (measured ~0.85s/step steady,
        # residency-independent) that should not recur. Reuse is bit-exact iff
        # every read cell is written each call (KH_coef_h / rhog_h / vtmp / vtmp2
        # are; confirmed by the gl07 A/B). Default off keeps per-call allocation.
        if getattr(self, "use_hdiff_hoist",
                   os.environ.get("PYNICAM_HDIFF_HOIST", "1") != "0"):
            _sc = self._hdiff_scratch(rdtype, cnst)
            KH_coef_h         = _sc["KH_coef_h"]
            KH_coef_lap1_h    = _sc["KH_coef_lap1_h"]
            KH_coef_h_pl      = _sc["KH_coef_h_pl"]
            KH_coef_lap1_h_pl = _sc["KH_coef_lap1_h_pl"]
            fact              = _sc["fact"]
            wk                = _sc["wk"]
            rhog_h            = _sc["rhog_h"]
            vtmp              = _sc["vtmp"]
            vtmp2             = _sc["vtmp2"]
            wk_pl             = _sc["wk_pl"]
            rhog_h_pl         = _sc["rhog_h_pl"]
            vtmp_pl           = _sc["vtmp_pl"]
            vtmp2_pl          = _sc["vtmp2_pl"]
        else:
            KH_coef_h         = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
            KH_coef_lap1_h    = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
            KH_coef_h_pl      = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
            KH_coef_lap1_h_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

            fact = np.full((adm.ADM_kall,), cnst.CONST_UNDEF, dtype=rdtype)

            wk     = np.full((adm.ADM_shape),        cnst.CONST_UNDEF, dtype=rdtype)
            rhog_h = np.full((adm.ADM_shape),        cnst.CONST_UNDEF, dtype=rdtype)
            vtmp   = np.full((adm.ADM_shape + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
            vtmp2  = np.full((adm.ADM_shape + (6,)), cnst.CONST_UNDEF, dtype=rdtype)

            wk_pl     = np.full((adm.ADM_shape_pl),        cnst.CONST_UNDEF, dtype=rdtype)
            rhog_h_pl = np.full((adm.ADM_shape_pl),        cnst.CONST_UNDEF, dtype=rdtype)
            vtmp_pl   = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
            vtmp2_pl  = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)

        # Tracer scratch: only allocate when the tracer hdiffusion actually runs.
        # Under MIURA2004 the tracer block below is gated off, so these six were
        # dead UNDEF allocations every call (the bulk of the setup churn).
        if rcnf.TRC_ADV_TYPE != 'MIURA2004':
            qtmp      = np.full((adm.ADM_shape + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
            qtmp2     = np.full((adm.ADM_shape + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
            qtmp_lap1 = np.full((adm.ADM_shape + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)

            qtmp_pl      = np.full((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
            qtmp2_pl     = np.full((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
            qtmp_lap1_pl = np.full((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)


        cfact = rdtype(2.0)
        T0    = rdtype(300.0)
        gall = adm.ADM_gall
        iall = adm.ADM_gall_1d
        jall = adm.ADM_gall_1d
        kall = adm.ADM_kall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        kminm1 = kmin - 1
        kminp1 = kmin + 1
        kmaxp1 = kmax + 1
        kmaxp2 = kmax + 2

        lall = adm.ADM_lall
        nall = rcnf.TRC_vmax
        CVdry = cnst.CONST_CVdry

        prf.PROF_rapend  ('______hdiff_set_alloc',2)
        prf.PROF_rapstart('______hdiff_set_coef',2)   # height_factor + kh_max + rhog_h interp

        if self.hdiff_nonlinear:
            self.height_factor(adm.ADM_kall, grd.GRD_gz, grd.GRD_htop, self.ZD_hdiff_nl, fact, cnst, rdtype)
            kh_max = (rdtype(1.0) - fact) * self.Kh_coef_maxlim + fact * self.Kh_coef_minlim  
        #endif


        # Extract weights from VMTR_C2Wfact. RES-CAPSTONE-15: these are STRIDED views
        # (stride-2 over the trailing axis) of constant geometry, so the per-step
        # fact1*rhog multiply ran cache-unfriendly (~0.16s/step host = hdiff_set_coef).
        # VMTR_C2Wfact is loop-invariant -> cache the contiguous slices ONCE. Bit-exact
        # (ascontiguousarray preserves values). Gate PYNICAM_HDIFF_C2W_CACHE (default on).
        if os.environ.get("PYNICAM_HDIFF_C2W_CACHE", "1") != "0":
            _c2w = getattr(self, "_hdiff_c2w_cache", None)
            if _c2w is None:
                _c2w = self._hdiff_c2w_cache = {
                    "f1":   np.ascontiguousarray(vmtr.VMTR_C2Wfact[:, :, kmin:kmaxp2, :, 0]),
                    "f2":   np.ascontiguousarray(vmtr.VMTR_C2Wfact[:, :, kmin:kmaxp2, :, 1]),
                    "f1pl": np.ascontiguousarray(vmtr.VMTR_C2Wfact_pl[:, kmin:kmaxp2, :, 0]),
                    "f2pl": np.ascontiguousarray(vmtr.VMTR_C2Wfact_pl[:, kmin:kmaxp2, :, 1]),
                }
            fact1 = _c2w["f1"]; fact2 = _c2w["f2"]
        else:
            fact1 = vmtr.VMTR_C2Wfact[:, :, kmin:kmaxp2, :, 0]  # shape (i, j, k, l)
            fact2 = vmtr.VMTR_C2Wfact[:, :, kmin:kmaxp2, :, 1]

        # Interpolate rhog to cell center
        rhog_h[:, :, kmin:kmaxp2, :] = (
            fact1 * rhog[:, :, kmin:kmaxp2, :] +
            fact2 * rhog[:, :, kminm1:kmaxp1,   :]
        )

        rhog_h[:, :, kminm1, :] = rdtype(0.0)


        #if ADM_have_pl:
        if os.environ.get("PYNICAM_HDIFF_C2W_CACHE", "1") != "0":
            fact1_pl = self._hdiff_c2w_cache["f1pl"]; fact2_pl = self._hdiff_c2w_cache["f2pl"]
        else:
            fact1_pl = vmtr.VMTR_C2Wfact_pl[:, kmin:kmaxp2, :, 0]
            fact2_pl = vmtr.VMTR_C2Wfact_pl[:, kmin:kmaxp2, :, 1]

        rhog_h_pl[:, kmin:kmaxp2, :] = (
            fact1_pl * rhog_pl[:, kmin:kmaxp2, :] +
            fact2_pl * rhog_pl[:, kminm1:kmaxp1,   :]
        )

        rhog_h_pl[:, kminm1, :] = rdtype(0.0)


        prf.PROF_rapend  ('______hdiff_set_coef',2)
        prf.PROF_rapstart('______hdiff_set_pack',2)   # vtmp/vtmp_pl packing from prognostic fields

        # Resident-path flags (computed here so the pack can go on device;
        # reused at the lap-order gate below).
        _resident_hdiff = (
            bk.type == "jax"
            and not self.NUMFILTER_DOhorizontaldiff_lap1
            and getattr(self, "use_resident_hdiff",
                        os.environ.get("PYNICAM_RESIDENT_HDIFF", "1") != "0")
        )
        _resident_full = (
            _resident_hdiff
            and getattr(self, "use_resident_hdiff_full",
                        os.environ.get("PYNICAM_HDIFF_RESIDENT_FULL", "1") != "0")
        )
        # Resident horizontalize (gated PYNICAM_HDIFF_RESIDENT_HORIZ, requires the
        # device tendency of _resident_full): fold OPRT_horizontalize_vec INTO the
        # device tendency so the velocity tendency stays on device across the
        # projection -- the host call below operates on strided numpy views
        # (tendency[...,I_RHOGV*]) and pays an asarray host-gather + to_numpy each
        # otherwise. Inputs are device-resident here (Stage C), so this removes a
        # genuine round-trip (NOT the strided-input ceiling of vi_path0/vi_path3).
        _resident_horiz = (
            _resident_full
            and getattr(self, "use_hdiff_resident_horiz",
                        os.environ.get("PYNICAM_HDIFF_RESIDENT_HORIZ", "1") != "0")
        )
        # RES-CAPSTONE Phase A (g_TEND0 device residency): reset the f_TEND device
        # stash each call. Set (to the 6-tuple of device f_TEND components) only by
        # _hdiff_tendency_resident under fold_horiz, so the caller falls back to
        # asarray(g_TEND0) whenever the resident+horizontalized path did not run.
        self._ftend_d = None
        # C2 (gated PYNICAM_HDIFF_PACK_DEVICE): build vtmp on device, skipping the
        # host packing -- the strided 6-component writes measured ~0.54s/step,
        # ~40x the CPU memory floor. Same H2D volume (6 fields vs the packed
        # vtmp), but the host pack cost disappears. Bit-exact (copies + subtracts).
        _pack_device = (
            _resident_full
            and getattr(self, "use_hdiff_pack_device",
                        os.environ.get("PYNICAM_HDIFF_PACK_DEVICE", "1") != "0")
        )
        # RESIDENT_PROG: caller passed device-resident PROG/DIAG/rho -> feed the
        # device-pack the prognostic/diagnostic fields as on-device views (cheap
        # slices) instead of the host strided-gather asarray of each [...,I_*].
        _rprog = (_pack_device and prog_d is not None
                  and diag_d is not None and rho_d is not None)
        _vtmp_d_pack = _vtmp_pl_d_pack = None
        if _pack_device:
            _reg_device = None
            if _rprog:
                _reg_device = (
                    diag_d[:, :, :, :, rcnf.I_vx], diag_d[:, :, :, :, rcnf.I_vy],
                    diag_d[:, :, :, :, rcnf.I_vz], diag_d[:, :, :, :, rcnf.I_w],
                    diag_d[:, :, :, :, rcnf.I_tem], rho_d,
                )
            _vtmp_d_pack, _vtmp_pl_d_pack = self._hdiff_pack_resident(
                vx, vy, vz, w, tem, rho,
                vx_pl, vy_pl, vz_pl, w_pl, tem_pl, rho_pl, bsst,
                reg_device=_reg_device,
            )
        else:
            vtmp[:, :, :, :, 0] = vx
            vtmp[:, :, :, :, 1] = vy
            vtmp[:, :, :, :, 2] = vz
            vtmp[:, :, :, :, 3] = w
            vtmp[:, :, :, :, 4] = tem - bsst.tem_bs
            vtmp[:, :, :, :, 5] = rho - bsst.rho_bs

            vtmp_pl[:, :, :, 0] = vx_pl
            vtmp_pl[:, :, :, 1] = vy_pl
            vtmp_pl[:, :, :, 2] = vz_pl
            vtmp_pl[:, :, :, 3] = w_pl
            vtmp_pl[:, :, :, 4] = tem_pl - bsst.tem_bs_pl
            vtmp_pl[:, :, :, 5] = rho_pl - bsst.rho_bs_pl


        # copy beforehand
        if self.NUMFILTER_DOhorizontaldiff_lap1:
            vtmp_lap1 = vtmp.copy() 
            vtmp_lap1_pl = vtmp_pl.copy()
        #endif

        prf.PROF_rapend  ('______hdiff_set_pack',2)
        prf.PROF_rapend  ('_____hdiff_setup',2)
        prf.PROF_rapstart('_____hdiff_laploop',2)   # lap-order loop + lap1 (the A/B residency region)

        # high order laplacian
        # Stage-A device residency (gated PYNICAM_RESIDENT_HDIFF, jax + lap1-off
        # only): run the lap-order loop keeping vtmp on device across the oprt
        # calls, draining only once per iter for the host COMM. Removes the
        # per-call H2D/D2H churn (the 4.66s hotspot). Bit-exact: identical
        # kernels/math, only the numpy<->device boundary moves. When it runs,
        # the for-loop below is skipped (range 0) and vtmp/KH_coef_h/self.Kh_coef
        # are left exactly as the non-resident path would leave them.
        # _resident_hdiff / _resident_full / _pack_device computed in the pack
        # section above. Stage C keeps vtmp on device through the tendency
        # (keep_device); C2 also hands in the device-packed vtmp (vtmp_d_in) so
        # vtmp is born on the device and never touches the host.
        _vtmp_d = _vtmp_pl_d = None
        if _resident_hdiff:
            _vtmp_d, _vtmp_pl_d = self._hdiff_laporder_resident(
                vtmp, vtmp_pl, KH_coef_h, KH_coef_h_pl,
                rhog, rhog_pl, kh_max if self.hdiff_nonlinear else None,
                oprt, comm, grd, tim, rcnf, cnst, rdtype,
                keep_device=_resident_full,
                vtmp_d_in=_vtmp_d_pack, vtmp_pl_d_in=_vtmp_pl_d_pack,
            )

        for p in range(0 if _resident_hdiff else self.lap_order_hdiff):  # 2 (0 and 1)

            # for momentum
            vtmp2[:,:,:,:,0], vtmp2_pl[:,:,:,0] = oprt.OPRT_laplacian(
                        vtmp[:,:,:,:,0], vtmp_pl[:,:,:,0], 
                        oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl,
                        rdtype,
            ) 

            vtmp2[:,:,:,:,1], vtmp2_pl[:,:,:,1] = oprt.OPRT_laplacian(
                        vtmp[:,:,:,:,1], vtmp_pl[:,:,:,1], 
                        oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl,
                        rdtype,
            )   

            vtmp2[:,:,:,:,2], vtmp2_pl[:,:,:,2] = oprt.OPRT_laplacian(
                        vtmp[:,:,:,:,2], vtmp_pl[:,:,:,2], 
                        oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl,
                        rdtype,
            )   

            vtmp2[:,:,:,:,3], vtmp2_pl[:,:,:,3] = oprt.OPRT_laplacian(
                        vtmp[:,:,:,:,3], vtmp_pl[:,:,:,3], 
                        oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl,
                        rdtype,
            )     

            # for scalar
            if p == self.lap_order_hdiff-1:  # last iteration 

                if self.hdiff_nonlinear:

                    large_step_dt = tim.TIME_DTL / rdtype(rcnf.DYN_DIV_NUM)
                

                    # Step 1: Compute d2T_dx2 = |vtmp[:,:,:,:,5]| / T0 * AREA_ave
                    d2T_dx2 = np.abs(vtmp[:, :, :, :, 5]) / T0 * self.AREA_ave

                    # Step 2: coef = cfact * AREA_ave² / dt * d2T_dx2
                    coef = cfact * (self.AREA_ave ** 2) / large_step_dt * d2T_dx2

                    # Step 3: Broadcast Kh_max over all dims (k → (1,1,k,1))
                    kh_max_broadcast = kh_max[None, None, :, None]

                    # Step 4: Apply min/max limits
                    self.Kh_coef = np.clip(coef, self.Kh_coef_minlim, kh_max_broadcast)


                    # Step 1: d2T_dx2 = |vtmp_pl[:,:,:,5]| / T0 * AREA_ave
                    d2T_dx2_pl = np.abs(vtmp_pl[:, :, :, 5]) / T0 * self.AREA_ave

                    # Step 2: coef = cfact * AREA_ave² / dt * d2T_dx2
                    coef_pl = cfact * (self.AREA_ave ** 2) / large_step_dt * d2T_dx2_pl

                    # Step 3: Broadcast self.Kh_max(k) over (g, k, l)
                    kh_max_broadcast_pl = self.Kh_max[None, :, None]  # shape (1, k, 1)

                    # Step 4: Clip to limits
                    self.Kh_coef_pl = np.clip(coef_pl, self.Kh_coef_minlim, kh_max_broadcast_pl)

                    # Centered average in vertical direction
                    KH_coef_h[:, :, kminp1:kmax+1, :] = 0.5 * (
                        self.Kh_coef[:, :, kminp1:kmax+1, :] +
                        self.Kh_coef[:, :, kmin:kmax,     :]
                    )

                    # Ghost layers
                    KH_coef_h[:, :, kminm1, :] = rdtype(0.0)
                    KH_coef_h[:, :, kmin,   :] = rdtype(0.0)
                    KH_coef_h[:, :, kmaxp1, :] = rdtype(0.0)

                    # Centered average
                    KH_coef_h_pl[:, kminp1:kmax+1, :] = 0.5 * (
                        self.Kh_coef_pl[:, kminp1:kmax+1, :] +
                        self.Kh_coef_pl[:, kmin:kmax,     :]
                    )

                    # Ghost layers
                    KH_coef_h_pl[:, kminm1, :] = rdtype(0.0)
                    KH_coef_h_pl[:, kmin,   :] = rdtype(0.0)
                    KH_coef_h_pl[:, kmaxp1, :] = rdtype(0.0)

                else:   

                    KH_coef_h[:, :, :, :] = self.Kh_coef
                    KH_coef_h_pl[:, :, :] = self.Kh_coef_pl

                    #KH_coef_h = KH_coef.copy() ?   Check later if I need a copy and not a view.

                # endif # nonlinear1

                # with open (std.fname_log, 'a') as log_file:
                #     print("going into OPRT_diffusion$$$", file=log_file )

                wk = rhog * CVdry * self.Kh_coef                   
                wk_pl = rhog_pl * CVdry * self.Kh_coef_pl

                vtmp2[:,:,:,:,4], vtmp2_pl[:,:,:,4] = oprt.OPRT_diffusion(
                    vtmp[:,:,:,:,4], vtmp_pl[:,:,:,4],                  # pretty good in SP at k=2
                    wk, wk_pl,                                          # good match between SP/DP/F/P
                    oprt.OPRT_coef_intp, oprt.OPRT_coef_intp_pl,        # good match between SP/DP/F/P
                    oprt.OPRT_coef_diff, oprt.OPRT_coef_diff_pl,        # pretty good in SP
                    grd, rdtype,
                )

                # with open (std.fname_log, 'a') as log_file:
                #     print("000A: OPRT_diffusion, lap order: ", p, file=log_file)
                #     print("rhog[6,5,2,0]", rhog[6,5,2,0], file=log_file)
                #     print("rhog[6,5,37,0]", rhog[6,5,37,0], file=log_file)
                #     print("CVdry,", CVdry, file=log_file)
                #     print("self.Kh_coef[6,5,2,0]", self.Kh_coef[6,5,2,0], file=log_file)
                #     print("self.Kh_coef[6,5,37,0]", self.Kh_coef[6,5,37,0], file=log_file)
                #     print("wk[6,5,2,0]", wk[6,5,2,0], file=log_file)
                #     print("wk[6,5,37,0]", wk[6,5,37,0], file=log_file)
                #     print("vtmp[6,5,2,0,:]", file=log_file)
                #     print( vtmp[6,5,2,0,:] , file=log_file)
                #     print("vtmp2[6,5,2,0,:]", file=log_file)
                #     print( vtmp2[6,5,2,0,:] , file=log_file)
                #     print("vtmp[6,5,37,0,:]", file=log_file)
                #     print( vtmp[6,5,37,0,:] , file=log_file)
                #     print("vtmp2[6,5,37,0,:]", file=log_file)
                #     print( vtmp2[6,5,37,0,:] , file=log_file)
                #     print("OPRT_coef_diff[6,5,0,0,0,:]", file=log_file)
                #     print( oprt.OPRT_coef_diff[6,5,0,0,0,:] , file=log_file)
                #     print("OPRT_coef_diff[6,5,0,0,1,:]", file=log_file)
                #     print( oprt.OPRT_coef_diff[6,5,0,0,1,:] , file=log_file)
                #     print("OPRT_coef_diff[6,5,0,0,2,:]", file=log_file)
                #     print( oprt.OPRT_coef_diff[6,5,0,0,2,:] , file=log_file)

                #     print("OPRT_coef_intp[6,5,0,0,0,:0]", file=log_file)
                #     print( oprt.OPRT_coef_intp[6,5,0,0,0,:,0] , file=log_file)
                #     print("OPRT_coef_intp[6,5,0,0,1,:0]", file=log_file)
                #     print( oprt.OPRT_coef_intp[6,5,0,0,1,:,0] , file=log_file)
                #     print("OPRT_coef_intp[6,5,0,0,2,:0]", file=log_file)
                #     print( oprt.OPRT_coef_intp[6,5,0,0,2,:,0] , file=log_file)

                #     print("OPRT_coef_intp[6,5,0,0,0,:,1]", file=log_file)
                #     print( oprt.OPRT_coef_intp[6,5,0,0,0,:,1] , file=log_file)
                #     print("OPRT_coef_intp[6,5,0,0,1,:,1]", file=log_file)
                #     print( oprt.OPRT_coef_intp[6,5,0,0,1,:,1] , file=log_file)
                #     print("OPRT_coef_diff[6,5,0,0,2,:,1]", file=log_file)
                #     print( oprt.OPRT_coef_intp[6,5,0,0,2,:,1] , file=log_file)
                #     print("OPRT_coef_lap[6,5,0,0,:,]", file=log_file)
                #     print( oprt.OPRT_coef_lap[6,5,0,0,:] , file=log_file)



                wk[:, :, :, :] = rhog * self.hdiff_fact_rho * self.Kh_coef
                wk_pl[:, :, :] = rhog_pl * self.hdiff_fact_rho * self.Kh_coef_pl

                vtmp2[:,:,:,:,5], vtmp2_pl[:,:,:,5] = oprt.OPRT_diffusion(
                    vtmp[:,:,:,:,5], vtmp_pl[:,:,:,5], 
                    wk, wk_pl, 
                    oprt.OPRT_coef_intp, oprt.OPRT_coef_intp_pl,   
                    oprt.OPRT_coef_diff, oprt.OPRT_coef_diff_pl,       
                    grd, rdtype,
                )

            else:


                vtmp2[:,:,:,:,4], vtmp2_pl[:,:,:,4] = oprt.OPRT_laplacian(
                            vtmp[:,:,:,:,4], vtmp_pl[:,:,:,4], 
                            oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl,
                            rdtype,
                )   

                vtmp2[:,:,:,:,5], vtmp2_pl[:,:,:,5] = oprt.OPRT_laplacian(
                            vtmp[:,:,:,:,5], vtmp_pl[:,:,:,5], 
                            oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl,
                            rdtype,
                )   

            #endif

            # with open (std.fname_log, 'a') as log_file:
            #     print("OPRT_diffusion, lap order: ", p, file=log_file)
            #     print("vtmp[6,5,2,0,:]", file=log_file)
            #     print( vtmp[6,5,2,0,:] , file=log_file)
            #     print("vtmp2[6,5,2,0,:]", file=log_file)
            #     print( vtmp2[6,5,2,0,:] , file=log_file)
            #     print("vtmp[6,5,37,0,:]", file=log_file)
            #     print( vtmp[6,5,37,0,:] , file=log_file)
            #     print("vtmp2[6,5,37,0,:]", file=log_file)
            #     print( vtmp2[6,5,37,0,:] , file=log_file)
            #     print("OPRT_coef_diff[6,5,0,0,0,:]", file=log_file)
            #     print( oprt.OPRT_coef_diff[6,5,0,0,0,:] , file=log_file)
            #     print("OPRT_coef_diff[6,5,0,0,1,:]", file=log_file)
            #     print( oprt.OPRT_coef_diff[6,5,0,0,1,:] , file=log_file)
            #     print("OPRT_coef_diff[6,5,0,0,2,:]", file=log_file)
            #     print( oprt.OPRT_coef_diff[6,5,0,0,2,:] , file=log_file)



            vtmp[:, :, :, :, :] = -vtmp2[:, :, :, :, :]
            vtmp_pl[:, :, :, :] = -vtmp2_pl[:, :, :, :]

            comm.COMM_data_transfer( vtmp, vtmp_pl )

        #enddo  laplacian order loop

        #--- 1st order laplacian filter
        if self.NUMFILTER_DOhorizontaldiff_lap1:

            KH_coef_lap1_h[:, :, :, :] = self.Kh_coef_lap1[:, :, :, :]
            KH_coef_lap1_h_pl[:, :, :] = self.Kh_coef_lap1_pl[:, :, :]

            vtmp2[:,:,:,:,0], vtmp2_pl[:,:,:,0] = oprt.OPRT_laplacian(
                        vtmp_lap1[:,:,:,:,0], vtmp_lap1_pl[:,:,:,0], 
                        oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl,
                        rdtype,
            )   

            vtmp2[:,:,:,:,1], vtmp2_pl[:,:,:,1] = oprt.OPRT_laplacian(
                        vtmp_lap1[:,:,:,:,1], vtmp_lap1_pl[:,:,:,1], 
                        oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl,
                        rdtype,
            )   

            vtmp2[:,:,:,:,2], vtmp2_pl[:,:,:,2] = oprt.OPRT_laplacian(
                        vtmp_lap1[:,:,:,:,2], vtmp_lap1_pl[:,:,:,2], 
                        oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl,
                        rdtype,
            )   

            vtmp2[:,:,:,:,3], vtmp2_pl[:,:,:,3] = oprt.OPRT_laplacian(
                        vtmp_lap1[:,:,:,:,3], vtmp_lap1_pl[:,:,:,3], 
                        oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl,
                        rdtype,
            )   

            wk[:, :, :, :] = rhog * CVdry * self.Kh_coef_lap1
            wk_pl[:, :, :] = rhog_pl * CVdry * self.Kh_coef_lap1_pl

            vtmp2[:,:,:,:,4], vtmp2_pl[:,:,:,4] = oprt.OPRT_diffusion(
                vtmp_lap1[:,:,:,:,4], vtmp_lap1_pl[:,:,:,4],    
                wk, wk_pl, 
                oprt.OPRT_coef_intp, oprt.OPRT_coef_intp_pl,   
                oprt.OPRT_coef_diff, oprt.OPRT_coef_diff_pl,       
                grd, rdtype,
            )

            wk[:, :, :, :] = rhog * self.hdiff_fact_rho * self.Kh_coef_lap1
            wk_pl[:, :, :] = rhog_pl * self.hdiff_fact_rho * self.Kh_coef_lap1_pl

            vtmp2[:,:,:,:,5], vtmp2_pl[:,:,:,5] = oprt.OPRT_diffusion(
                vtmp_lap1[:,:,:,:,5], vtmp_lap1_pl[:,:,:,5],
                wk[:,:,:,:], wk_pl[:,:,:], 
                oprt.OPRT_coef_intp[:,:,:,:,:,:], oprt.OPRT_coef_intp_pl[:,:,:,:],   
                oprt.OPRT_coef_diff, oprt.OPRT_coef_diff_pl,       
                grd, rdtype,
            )

            vtmp_lap1[:, :, :, :, :] = -vtmp2[:, :, :, :, :]
            vtmp_lap1_pl[:, :, :, :] = -vtmp2_pl[:, :, :, :]

            comm.COMM_data_transfer( vtmp_lap1, vtmp_lap1_pl )

        else:

            KH_coef_lap1_h[:, :, :, :] = rdtype(0.0)
            vtmp_lap1 = np.zeros_like(vtmp)
            #vtmp_lap1[:, :, :, :, :]   = rdtype(0.0)
            KH_coef_lap1_h_pl[:, :, :] = rdtype(0.0)
            vtmp_lap1_pl = np.zeros_like(vtmp_pl)
            #vtmp_lap1_pl[:, :, :, :]   = rdtype(0.0)

        #endif

        # with open (std.fname_log, 'a') as log_file:
        #     print("OPRT_diffusion, update tend: ", file=log_file)
        #     print("vtmp[6,5,2,0,:]", file=log_file)
        #     print( vtmp[6,5,2,0,:] , file=log_file)
        #     print("vtmp_lap1[6,5,2,0,:]", file=log_file)
        #     print( vtmp_lap1[6,5,2,0,:] , file=log_file)
        #     # print("OPRT_coef_diff[6,5,:,0,0]", file=log_file)
        #     # print( oprt.OPRT_coef_diff[6,5,:,0,0] , file=log_file)
        #     # print("OPRT_coef_diff[6,5,:,0,1]", file=log_file)
        #     # print( oprt.OPRT_coef_diff[6,5,:,0,1] , file=log_file)
        #     # print("OPRT_coef_diff[6,5,:,0,2]", file=log_file)
        #     # print( oprt.OPRT_coef_diff[6,5,:,0,2] , file=log_file)




        prf.PROF_rapend  ('_____hdiff_laploop',2)
        prf.PROF_rapstart('_____hdiff_tendency',2)   # tendency assembly + horizontalize + tracer(off)

        #--- Update tendency

        # Vectorized main domain update. Stage C: device tendency from the
        # resident vtmp_d (single drain); else the original host multiply-adds.
        if _resident_full:
            self._hdiff_tendency_resident(
                _vtmp_d, _vtmp_pl_d, tendency, tendency_pl,
                KH_coef_h, KH_coef_h_pl, rhog, rhog_h, rhog_pl, rhog_h_pl,
                rcnf, rdtype, oprt=oprt, grd=grd, fold_horiz=_resident_horiz,
                rhog_d_in=(prog_d[:, :, :, :, rcnf.I_RHOG] if _rprog else None),
                stash_device=stash_device,
            )
        else:
            self._hdiff_tendency_host(
                tendency, tendency_pl, vtmp, vtmp_pl, vtmp_lap1, vtmp_lap1_pl,
                KH_coef_h, KH_coef_lap1_h, KH_coef_h_pl, KH_coef_lap1_h_pl,
                rhog, rhog_h, rhog_pl, rhog_h_pl, rcnf, rdtype,
            )


        # with open (std.fname_log, 'a') as log_file:
        #     print("tendency 0: ", file=log_file)
        #     print("tendency[6,5,2,0,:]", file=log_file)
        #     print( tendency[6,5,2,0,:] , file=log_file)
        #     print("tendency[6,5,37,0,:]", file=log_file)
        #     print( tendency[6,5,37,0,:] , file=log_file)
        #     #print("vtmp_lap1[6,5,2,0,:]", file=log_file)
        #     #print( vtmp_lap1[6,5,2,0,:] , file=log_file)


        # When _resident_horiz, the projection was folded into the device tendency
        # above (on-device, before the drain) -> skip the host call on strided views.
        if not _resident_horiz:
            oprt.OPRT_horizontalize_vec(
                tendency[:, :, :, :, rcnf.I_RHOGVX], tendency_pl[:, :, :, rcnf.I_RHOGVX], # [INOUT]
                tendency[:, :, :, :, rcnf.I_RHOGVY], tendency_pl[:, :, :, rcnf.I_RHOGVY], # [INOUT]
                tendency[:, :, :, :, rcnf.I_RHOGVZ], tendency_pl[:, :, :, rcnf.I_RHOGVZ], # [INOUT]
                grd, rdtype,
            )


        # with open (std.fname_log, 'a') as log_file:
        #     print("tendency 1: ", file=log_file)
        #     print("tendency[6,5,2,0,:]", file=log_file)
        #     print( tendency[6,5,2,0,:] , file=log_file)

        #---------------------------------------------------------------------------
        # For tracer
        #---------------------------------------------------------------------------
        # 08/04/12 [Mod] T.Mitsui, hyper diffusion is needless for tracer if MIURA2004
        #                          because that is upwind-type advection(already diffusive)
        if rcnf.TRC_ADV_TYPE != 'MIURA2004':

            qtmp[:,:,:,:,:]  = q[:,:,:,:,:]
            qtmp_pl[:,:,:,:] = q_pl[:,:,:,:]

            # copy beforehand
            if self.NUMFILTER_DOhorizontaldiff_lap1:
                qtmp_lap1[:,:,:,:,:] = qtmp[:,:,:,:,:].copy()
                qtmp_lap1_pl[:,:,:,:] = qtmp_pl[:,:,:,:].copy()
            #endif

            # high order laplacian filter
            for p in range(self.lap_order_hdiff): # check range later
                if p == self.lap_order_hdiff:

                    wk [:,:,:,:] = rhog * self.hdiff_fact_q * self.Kh_coef   
                    wk_pl[:,:,:] = rhog_pl * self.hdiff_fact_q * self.Kh_coef_pl

                    for nq in range(rcnf.TRC_vmax):

                        qtmp2[:,:,:,:,nq], qtmp2_pl[:,:,:,nq] = oprt.OPRT_diffusion(
                            qtmp[:,:,:,:,nq], qtmp_pl[:,:,:,nq], 
                            wk, wk_pl, 
                            oprt.OPRT_coef_intp, oprt.OPRT_coef_intp_pl,   
                            oprt.OPRT_coef_diff, oprt.OPRT_coef_diff_pl,       
                            grd, rdtype,
                        )

                    #enddo
                else:
                    for nq in range(rcnf.TRC_vmax):
                        qtmp2[:,:,:,:,nq], qtmp2_pl[:,:,:,nq] = oprt.OPRT_laplacian(
                                qtmp[:,:,:,:,nq], qtmp_pl[:,:,:,nq], 
                                oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl,
                        )  
 
                    #enddo
                #endif

                qtmp [:,:,:,:,:] = -qtmp2 [:,:,:,:,:]
                qtmp_pl[:,:,:,:] = -qtmp2_pl[:,:,:,:]

                comm.COMM_data_transfer( qtmp, qtmp_pl )

            #enddo  # laplacian order loop

            #--- 1st order laplacian filter
            if self.NUMFILTER_DOhorizontaldiff_lap1:

                wk [:,:,:,:] = rhog  * self.hdiff_fact_q * self.Kh_coef_lap1 
                wk_pl[:,:,:] = rhog_pl * self.hdiff_fact_q * self.Kh_coef_lap1_pl

                for nq in range(rcnf.TRC_vmax):
                        qtmp2[:,:,:,:,nq], qtmp2_pl[:,:,:,nq] = oprt.OPRT_diffusion(
                        qtmp_lap1[:,:,:,:,nq], qtmp_lap1_pl[:,:,:,nq], 
                        wk, wk_pl, 
                        oprt.OPRT_coef_intp, oprt.OPRT_coef_intp_pl,   
                        oprt.OPRT_coef_diff, oprt.OPRT_coef_diff_pl,       
                        grd, rdtype,
                        )
                #enddo

                qtmp_lap1 [:,:,:,:,:] = -qtmp2 [:,:,:,:,:]
                qtmp_lap1_pl[:,:,:,:] = -qtmp2_pl[:,:,:,:]

                comm.COMM_data_transfer( qtmp_lap1[:,:,:,:,:], qtmp_lap1_pl[:,:,:,:] )

            else:
                qtmp_lap1 [:,:,:,:,:] = rdtype(0.0)
                qtmp_lap1_pl[:,:,:,:] = rdtype(0.0)

            #endif

            tendency_q[:, :, :, :, :] = - (qtmp[:, :, :, :, :] + qtmp_lap1[:, :, :, :, :])


            if adm.ADM_have_pl:
                tendency_q_pl[:] = - (qtmp_pl + qtmp_lap1_pl)
            else:
                tendency_q_pl[:,:,:,:] = rdtype(0.0)
            #endif

        else:           

            tendency_q[:, :, :, :, :] = rdtype(0.0)
            tendency_q_pl[:, :, :, :] = rdtype(0.0)
 
        #endif  # apply filter to tracer?

        prf.PROF_rapend  ('_____hdiff_tendency',2)
        prf.PROF_rapend('____numfilter_hdiffusion',2)

        return

    def _hdiff_pack_resident(
        self, vx, vy, vz, w, tem, rho,
        vx_pl, vy_pl, vz_pl, w_pl, tem_pl, rho_pl, bsst,
        reg_device=None,
    ):
        """C2: build the 6-component vtmp on device (stack of the prognostic
        fields; tem/rho are perturbations from the constant base state). Replaces
        the host packing whose strided 6-wide writes ran ~40x the CPU memory
        floor. tem_bs/rho_bs are constant -> cached device-resident. Bit-exact
        vs the host pack (plain copies + subtracts, no FMA).

        reg_device (RESIDENT_PROG): a tuple of the 6 regular fields
        (vx,vy,vz,w,tem,rho) already on device (views into device PROG/DIAG);
        when given, stack them directly instead of host strided-gather asarray.
        The pole (_pl) fields stay host (tiny)."""
        xp = bk.xp
        d = bk.device_consts(self, "hdiff_basestate", lambda: {
            "tem_bs":    bsst.tem_bs,
            "rho_bs":    bsst.rho_bs,
            "tem_bs_pl": bsst.tem_bs_pl,
            "rho_bs_pl": bsst.rho_bs_pl,
        })
        if reg_device is not None:
            _vxd, _vyd, _vzd, _wd, _temd, _rhod = reg_device
            vtmp_d = xp.stack((
                _vxd, _vyd, _vzd, _wd,
                _temd - d["tem_bs"], _rhod - d["rho_bs"],
            ), axis=-1)
        else:
            vtmp_d = xp.stack((
                xp.asarray(vx), xp.asarray(vy), xp.asarray(vz), xp.asarray(w),
                xp.asarray(tem) - d["tem_bs"], xp.asarray(rho) - d["rho_bs"],
            ), axis=-1)
        vtmp_pl_d = xp.stack((
            xp.asarray(vx_pl), xp.asarray(vy_pl), xp.asarray(vz_pl), xp.asarray(w_pl),
            xp.asarray(tem_pl) - d["tem_bs_pl"], xp.asarray(rho_pl) - d["rho_bs_pl"],
        ), axis=-1)
        return vtmp_d, vtmp_pl_d

    def _hdiff_tendency_host(
        self, tendency, tendency_pl, vtmp, vtmp_pl, vtmp_lap1, vtmp_lap1_pl,
        KH_coef_h, KH_coef_lap1_h, KH_coef_h_pl, KH_coef_lap1_h_pl,
        rhog, rhog_h, rhog_pl, rhog_h_pl, rcnf, rdtype,
    ):
        """Original host tendency assembly (extracted verbatim from
        numfilter_hdiffusion so the resident path can branch around it without
        touching the validated numpy math)."""
        tendency[:, :, :, :, rcnf.I_RHOGVX] = -(
            vtmp[:, :, :, :, 0] * self.Kh_coef + vtmp_lap1[:, :, :, :, 0] * self.Kh_coef_lap1
        ) * rhog

        tendency[:, :, :, :, rcnf.I_RHOGVY] = -(
            vtmp[:, :, :, :, 1] * self.Kh_coef + vtmp_lap1[:, :, :, :, 1] * self.Kh_coef_lap1
        ) * rhog

        tendency[:, :, :, :, rcnf.I_RHOGVZ] = -(
            vtmp[:, :, :, :, 2] * self.Kh_coef + vtmp_lap1[:, :, :, :, 2] * self.Kh_coef_lap1
        ) * rhog

        tendency[:, :, :, :, rcnf.I_RHOGW] = -(
            vtmp[:, :, :, :, 3] * KH_coef_h + vtmp_lap1[:, :, :, :, 3] * KH_coef_lap1_h
        ) * rhog_h

        tendency[:, :, :, :, rcnf.I_RHOGE] = -(
            vtmp[:, :, :, :, 4] + vtmp_lap1[:, :, :, :, 4]
        )

        tendency[:, :, :, :, rcnf.I_RHOG] = -(
            vtmp[:, :, :, :, 5] + vtmp_lap1[:, :, :, :, 5]
        )


        if adm.ADM_have_pl:
            tendency_pl[:, :, :, rcnf.I_RHOGVX] = -(
                vtmp_pl[:, :, :, 0] * self.Kh_coef_pl + vtmp_lap1_pl[:, :, :, 0] * self.Kh_coef_lap1_pl
            ) * rhog_pl

            tendency_pl[:, :, :, rcnf.I_RHOGVY] = -(
                vtmp_pl[:, :, :, 1] * self.Kh_coef_pl + vtmp_lap1_pl[:, :, :, 1] * self.Kh_coef_lap1_pl
            ) * rhog_pl

            tendency_pl[:, :, :, rcnf.I_RHOGVZ] = -(
                vtmp_pl[:, :, :, 2] * self.Kh_coef_pl + vtmp_lap1_pl[:, :, :, 2] * self.Kh_coef_lap1_pl
            ) * rhog_pl

            tendency_pl[:, :, :, rcnf.I_RHOGW] = -(
                vtmp_pl[:, :, :, 3] * KH_coef_h_pl + vtmp_lap1_pl[:, :, :, 3] * KH_coef_lap1_h_pl
            ) * rhog_h_pl

            tendency_pl[:, :, :, rcnf.I_RHOGE] = -(
                vtmp_pl[:, :, :, 4] + vtmp_lap1_pl[:, :, :, 4]
            )

            tendency_pl[:, :, :, rcnf.I_RHOG] = -(
                vtmp_pl[:, :, :, 5] + vtmp_lap1_pl[:, :, :, 5]
            )

        else:
            tendency_pl[:] = rdtype(0.0)
        return

    def _hdiff_tendency_resident(
        self, vtmp_d, vtmp_pl_d, tendency, tendency_pl,
        KH_coef_h, KH_coef_h_pl, rhog, rhog_h, rhog_pl, rhog_h_pl,
        rcnf, rdtype, oprt=None, grd=None, fold_horiz=False, rhog_d_in=None,
        stash_device=False,
    ):
        """Stage C: tendency multiply-adds on device from the resident vtmp_d,
        drained once into the numpy tendency arrays. lap1-off path only (the
        caller's gate guarantees it), so the vtmp_lap1 terms are identically
        zero and dropped. GPU compute -> machine-precision vs the host path
        (validated by cmp_prec rtol 1e-10), not bit-exact.

        fold_horiz=True (gated PYNICAM_HDIFF_RESIDENT_HORIZ): keep the velocity
        tendency (I_RHOGV{X,Y,Z}) on device and run OPRT_horizontalize_vec
        resident BEFORE draining, so the radial-component projection happens on
        device -- no asarray host-gather of the strided tendency[...,I_RHOGV*]
        views + no to_numpy/re-asarray round-trip. The caller skips its host
        OPRT_horizontalize_vec when this is on. Bit-exact vs the host projection
        (regional combined-divide + pole per-term order match the original)."""
        xp = bk.xp
        # RES-CAPSTONE: for the LINEAR hdiff types (DIRECT etc.) Kh_coef/KH_coef_h are
        # loop-invariant -- Kh_coef is set once from grid area + the fixed diffusion
        # gamma (numfilter setup), and KH_coef_h = Kh_coef (its half-level form) is
        # rebuilt to identical values every call. Cache them device-resident instead of
        # re-uploading each nl. SAFETY GUARD: hdiff_type=='NONLINEAR1' (self.hdiff_
        # nonlinear) RECOMPUTES Kh_coef from the temperature Laplacian every step
        # (state-dependent), so it MUST be re-uploaded -> asarray fallback. Bit-exact:
        # cache-once vs asarray-every-call is the same values when not nonlinear.
        if not self.hdiff_nonlinear:
            _khc = bk.device_consts(self, "hdiff_khcoef", lambda: {
                "Kh": self.Kh_coef, "KHh": KH_coef_h})
            Kh  = _khc["Kh"]
            KHh = _khc["KHh"]
        else:
            Kh      = xp.asarray(self.Kh_coef)
            KHh     = xp.asarray(KH_coef_h)
        # RESIDENT_PROG: device view of rhog (PROG[...,I_RHOG]) instead of the
        # host strided-gather asarray. rhog_h is computed scratch (stays host).
        rhog_d  = rhog_d_in if rhog_d_in is not None else xp.asarray(rhog)
        rhogh_d = xp.asarray(rhog_h)

        # velocity tendency components kept on device for the optional fold
        tvx_d = -(vtmp_d[:, :, :, :, 0] * Kh) * rhog_d
        tvy_d = -(vtmp_d[:, :, :, :, 1] * Kh) * rhog_d
        tvz_d = -(vtmp_d[:, :, :, :, 2] * Kh) * rhog_d

        have_pl = adm.ADM_have_pl
        if have_pl:
            Kh_pl   = xp.asarray(self.Kh_coef_pl)
            rhogpl  = xp.asarray(rhog_pl)
            tvx_pl_d = -(vtmp_pl_d[:, :, :, 0] * Kh_pl) * rhogpl
            tvy_pl_d = -(vtmp_pl_d[:, :, :, 1] * Kh_pl) * rhogpl
            tvz_pl_d = -(vtmp_pl_d[:, :, :, 2] * Kh_pl) * rhogpl
        else:
            # horizontalize needs device pole inputs (it zeros them when no pole).
            _z = xp.zeros(adm.ADM_shape_pl, dtype=tvx_d.dtype)
            tvx_pl_d = tvy_pl_d = tvz_pl_d = _z

        if fold_horiz:
            # project the radial component on device (interior + full pole) before
            # the drain; mirrors the host OPRT_horizontalize_vec on tendency[VX,VY,VZ].
            tvx_d, tvy_d, tvz_d, tvx_pl_d, tvy_pl_d, tvz_pl_d = (
                oprt.OPRT_horizontalize_vec(
                    tvx_d, tvx_pl_d, tvy_d, tvy_pl_d, tvz_d, tvz_pl_d,
                    grd, rdtype, resident=True,
                )
            )

        # W/E/RHOG device components, named so they can be both drained and stashed.
        tw_d   = -(vtmp_d[:, :, :, :, 3] * KHh) * rhogh_d
        te_d   = -vtmp_d[:, :, :, :, 4]
        trho_d = -vtmp_d[:, :, :, :, 5]
        # RC-66: skip the dead host REGULAR hdiff tendency drain (~1GB/nl). host f_TEND is
        # unread once the device g_TEND assembly (RESIDENT_GTEND) consumes the _ftend_d
        # stash (below) -- POISON-CONFIRMED dead (hdiftreg PASS job 2267793). The regular
        # analog of RC-64 (pole), found by the dynamic audit. Gate PYNICAM_RESIDENT_HDIFF_TEND
        # (default OFF) + requires the stash (stash_device+fold_horiz) + the consumer gate.
        _skip_tend = (stash_device and fold_horiz
                      and os.environ.get("PYNICAM_RESIDENT_GTEND", "1") != "0"
                      and os.environ.get("PYNICAM_RESIDENT_HDIFF_TEND", "0") != "0")
        if not _skip_tend:
            tendency[:, :, :, :, rcnf.I_RHOGVX] = bk.to_numpy(tvx_d)
            tendency[:, :, :, :, rcnf.I_RHOGVY] = bk.to_numpy(tvy_d)
            tendency[:, :, :, :, rcnf.I_RHOGVZ] = bk.to_numpy(tvz_d)
            tendency[:, :, :, :, rcnf.I_RHOGW]  = bk.to_numpy(tw_d)
            tendency[:, :, :, :, rcnf.I_RHOGE]  = bk.to_numpy(te_d)
            tendency[:, :, :, :, rcnf.I_RHOG]   = bk.to_numpy(trho_d)
        # RESIDENCY-AUDIT POISON (campaign): NaN the host REGULAR hdiff tendency (= f_TEND)
        # AFTER the drain; PASS => host f_TEND unread (device g_TEND assembly uses the
        # _ftend_d stash) -> this ~1GB/nl drain is removable (regular analog of RC-64).
        if "hdiftreg" in os.environ.get("PYNICAM_REG_POISON", ""):
            tendency[:] = np.nan
        # RES-CAPSTONE Phase A: stash the regular device f_TEND components for the
        # caller's device g_TEND assembly. Only under fold_horiz, where tvx/tvy/tvz_d
        # are already horizontalized on device (so they match the host tendency the
        # caller would otherwise asarray). Order = (VX, VY, VZ, W, E, RHOG).
        if stash_device and fold_horiz:
            self._ftend_d = (tvx_d, tvy_d, tvz_d, tw_d, te_d, trho_d)

        if have_pl:
            KHh_pl   = xp.asarray(KH_coef_h_pl)
            rhoghpl  = xp.asarray(rhog_h_pl)
            tw_pl_d   = -(vtmp_pl_d[:, :, :, 3] * KHh_pl) * rhoghpl
            te_pl_d   = -vtmp_pl_d[:, :, :, 4]
            trho_pl_d = -vtmp_pl_d[:, :, :, 5]
            # RES-CAPSTONE-38 (Track B): stash the device POLE hdiff tendency (order
            # VX,VY,VZ,W,E,RHOG) for the caller's device g_TEND_pl assembly -- the pole
            # analog of _ftend_d above. Same gate (stash_device and fold_horiz).
            if stash_device and fold_horiz:
                self._ftend_pl_d = (tvx_pl_d, tvy_pl_d, tvz_pl_d, tw_pl_d, te_pl_d, trho_pl_d)
            # RES-CAPSTONE-64: skip the dead host pole hdiff tendency drain. host f_TEND_pl
            # is unread on the resident path once BOTH its g_TEND0_pl consumers are device:
            # the device pole g_TEND assembly (_g_TEND_pl_d, needs RESIDENT_GTEND_PL) AND
            # the pole grhogetot0_pl (now _g0p[I_RHOGE], RC-64 vi fix). POISON-CONFIRMED
            # dead (hdifftpl PASS, job 2267353). Requires the device stash present + the
            # consumer gate so no half-on combo reads a stale host f_TEND_pl. Gate
            # PYNICAM_RESIDENT_HDIFF_TEND_PL (default OFF; full drain = bit-exact when off).
            _skip_tend_pl = (stash_device and fold_horiz
                             and os.environ.get("PYNICAM_RESIDENT_GTEND_PL", "0") != "0"
                             and os.environ.get("PYNICAM_RESIDENT_HDIFF_TEND_PL", "0") != "0")
            if not _skip_tend_pl:
                tendency_pl[:, :, :, rcnf.I_RHOGVX] = bk.to_numpy(tvx_pl_d)
                tendency_pl[:, :, :, rcnf.I_RHOGVY] = bk.to_numpy(tvy_pl_d)
                tendency_pl[:, :, :, rcnf.I_RHOGVZ] = bk.to_numpy(tvz_pl_d)
                tendency_pl[:, :, :, rcnf.I_RHOGW]  = bk.to_numpy(tw_pl_d)
                tendency_pl[:, :, :, rcnf.I_RHOGE]  = bk.to_numpy(te_pl_d)
                tendency_pl[:, :, :, rcnf.I_RHOG]   = bk.to_numpy(trho_pl_d)
            # Track B POLE-POISON (RC-37 classify): NaN the hdiff pole tendency after the
            # drain; if gl07 still PASSES vs gold, host tendency_pl is unread on the tested
            # path -> the device pole tendency (tvx_pl_d.. + vtmp_pl_d) can be stashed +
            # threaded into the g_TEND_pl assembly (pole analog of _ftend_d/RC-36) and this
            # drain removed. PYNICAM_PL_POISON comma-list; default empty = bit-exact.
            if "hdifftpl" in os.environ.get("PYNICAM_PL_POISON", ""):
                tendency_pl[:] = np.nan
        else:
            tendency_pl[:] = rdtype(0.0)
        return

    def _hdiff_scratch(self, rdtype, cnst):
        """Lazily-allocated, reused scratch buffers for numfilter_hdiffusion
        (gated PYNICAM_HDIFF_HOIST). Allocated once with UNDEF and reused every
        call, removing the per-call np.full alloc+fill of these ~13 large arrays
        from steady state. Shapes are fixed for a run, so a single cache suffices.
        Reuse is bit-exact iff every cell read is overwritten each call (true for
        the buffers cached here; the gl07 A/B confirms)."""
        sc = getattr(self, "_hdiff_scratch_cache", None)
        if sc is None:
            U = cnst.CONST_UNDEF
            sc = {
                "KH_coef_h":         np.full((adm.ADM_shape),        U, dtype=rdtype),
                "KH_coef_lap1_h":    np.full((adm.ADM_shape),        U, dtype=rdtype),
                "KH_coef_h_pl":      np.full((adm.ADM_shape_pl),     U, dtype=rdtype),
                "KH_coef_lap1_h_pl": np.full((adm.ADM_shape_pl),     U, dtype=rdtype),
                "fact":              np.full((adm.ADM_kall,),        U, dtype=rdtype),
                "wk":                np.full((adm.ADM_shape),        U, dtype=rdtype),
                "rhog_h":            np.full((adm.ADM_shape),        U, dtype=rdtype),
                "vtmp":              np.full((adm.ADM_shape + (6,)), U, dtype=rdtype),
                "vtmp2":             np.full((adm.ADM_shape + (6,)), U, dtype=rdtype),
                "wk_pl":             np.full((adm.ADM_shape_pl),        U, dtype=rdtype),
                "rhog_h_pl":         np.full((adm.ADM_shape_pl),        U, dtype=rdtype),
                "vtmp_pl":           np.full((adm.ADM_shape_pl + (6,)), U, dtype=rdtype),
                "vtmp2_pl":          np.full((adm.ADM_shape_pl + (6,)), U, dtype=rdtype),
            }
            self._hdiff_scratch_cache = sc
        return sc

    def _hdiff_laporder_resident(
        self, vtmp, vtmp_pl, KH_coef_h, KH_coef_h_pl,
        rhog, rhog_pl, kh_max, oprt, comm, grd, tim, rcnf, cnst, rdtype,
        keep_device=False, vtmp_d_in=None, vtmp_pl_d_in=None,
    ):
        """Device-resident replacement for the numfilter_hdiffusion lap-order
        loop (Stage A). vtmp is uploaded once, kept on device across the 6 oprt
        calls per iter (resident=True -> no per-call to_numpy), negated on
        device, then drained ONCE for the host COMM and re-uploaded. Per iter:
        1 H2D (re-upload) + 1 D2H (pre-COMM) + 1 D2H of vtmp[...,5] at the last
        iter for the nonlinear Kh_coef host calc -- vs ~12 round-trips before.

        Mutates vtmp / vtmp_pl in place (final values for the post-loop tendency)
        and fills KH_coef_h / KH_coef_h_pl + self.Kh_coef(_pl), so on return the
        caller's state is byte-identical to the non-resident path. lap1 is
        guaranteed off by the caller's gate, so this only covers the high-order
        loop. Geometry coef_* are already device_consts-cached in the oprt fused
        wrappers; only the variable fields cross the boundary.
        """
        xp = bk.xp
        # Stage B (gated PYNICAM_HDIFF_ONDEVICE_COMM): keep vtmp on device across
        # the halo exchange via the on-device COMM path, so it never drains in the
        # loop. Default off -> Stage A (drain once per iter for the host COMM).
        _ondevice_comm = os.environ.get("PYNICAM_HDIFF_ONDEVICE_COMM", "1") != "0"
        cfact = rdtype(2.0)
        T0    = rdtype(300.0)
        CVdry = cnst.CONST_CVdry
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        kminm1 = kmin - 1
        kminp1 = kmin + 1
        kmaxp1 = kmax + 1

        # C2: use the device-packed vtmp if handed in (born on device, no host
        # pack); else upload the host-packed vtmp (one H2D to start the carry).
        vtmp_d    = vtmp_d_in    if vtmp_d_in    is not None else xp.asarray(vtmp)
        vtmp_pl_d = vtmp_pl_d_in if vtmp_pl_d_in is not None else xp.asarray(vtmp_pl)

        # FUSE_HDIFF (Phase 2 / U8): collapse the WHOLE lap-order loop -- per iter the
        # 6 OPRT kernels + the negate/stack + the on-device mpi4jax COMM -- into ONE
        # jit graph (COMM-in-jit; the pure jit'd COMM core _get_ondevice_comm_fn is
        # called directly, NOT the COMM_data_transfer wrapper, so its host-side MPI
        # barrier/PROF timers stay out of the graph). Requires the on-device COMM
        # (Stage B) AND the LINEAR path: the nonlinear last-iteration Kh_coef needs a
        # mid-loop D2H of vtmp[...,5] to host, which cannot live in the graph -> FALL
        # BACK to the eager loop + notify once. In the linear path self.Kh_coef is the
        # DIRECT run-constant, so KH_coef_h + the wk coefficients are loop-invariant
        # and computed host-side here (bit-identical to the eager last iter), with wk
        # fed to the graph as traced inputs. Default OFF. Warm-up: the first call runs
        # the eager loop (warms the OPRT/device_consts/COMM-core caches), then builds +
        # caches the jit; later calls take the fast-path.
        _fuse_hdiff = (bk.type == "jax" and _ondevice_comm
                       and os.environ.get("PYNICAM_FUSE_HDIFF", "0") != "0")
        if (_fuse_hdiff and self.hdiff_nonlinear
                and not getattr(self, "_fuse_hdiff_nl_warned", False)):
            self._fuse_hdiff_nl_warned = True
            _msg = ("[FUSE_HDIFF] disabled: hdiff_nonlinear=True -> the last-iteration "
                    "Kh_coef D2H breaks the COMM-in-jit laploop graph; falling back to "
                    "the eager laploop.")
            if std.io_l:
                with open(std.fname_log, 'a') as _lf:
                    print(_msg, file=_lf)
            if prc.prc_myrank == 0:
                print(_msg, flush=True)
        _fuse_lap = _fuse_hdiff and not self.hdiff_nonlinear

        _run_eager_loop = True
        if _fuse_lap and getattr(self, "_hdiff_laploop_jit", None) is not None:
            # fused fast-path. KH_coef_h + the wk coefs are loop-invariant in the
            # linear path (self.Kh_coef is the DIRECT run-constant); fill/compute them
            # here exactly as the eager last iter (1779-1801) does, then feed wk to the
            # graph. The graph runs the full lap loop (OPRT + stack + COMM) on device.
            KH_coef_h[:, :, :, :] = self.Kh_coef
            KH_coef_h_pl[:, :, :]  = self.Kh_coef_pl
            wk4    = rhog    * CVdry * self.Kh_coef
            wk4_pl = rhog_pl * CVdry * self.Kh_coef_pl
            wk5    = rhog    * self.hdiff_fact_rho * self.Kh_coef
            wk5_pl = rhog_pl * self.hdiff_fact_rho * self.Kh_coef_pl
            vtmp_d, vtmp_pl_d = self._hdiff_laploop_jit(
                vtmp_d, vtmp_pl_d,
                xp.asarray(wk4), xp.asarray(wk4_pl), xp.asarray(wk5), xp.asarray(wk5_pl))
            _run_eager_loop = False

        for p in range(self.lap_order_hdiff if _run_eager_loop else 0):

            o   = [None] * 6
            opl = [None] * 6

            # for momentum (components 0..3)
            for c in range(4):
                o[c], opl[c] = oprt.OPRT_laplacian(
                    vtmp_d[:, :, :, :, c], vtmp_pl_d[:, :, :, c],
                    oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl, rdtype,
                    resident=True,
                )

            # for scalar (components 4,5)
            if p == self.lap_order_hdiff - 1:  # last iteration

                if self.hdiff_nonlinear:
                    large_step_dt = tim.TIME_DTL / rdtype(rcnf.DYN_DIV_NUM)

                    # Kh_coef needs vtmp[...,5] on host (matches non-resident).
                    v5    = bk.to_numpy(vtmp_d[:, :, :, :, 5])
                    v5_pl = bk.to_numpy(vtmp_pl_d[:, :, :, 5])

                    d2T_dx2 = np.abs(v5) / T0 * self.AREA_ave
                    coef = cfact * (self.AREA_ave ** 2) / large_step_dt * d2T_dx2
                    kh_max_broadcast = kh_max[None, None, :, None]
                    self.Kh_coef = np.clip(coef, self.Kh_coef_minlim, kh_max_broadcast)

                    d2T_dx2_pl = np.abs(v5_pl) / T0 * self.AREA_ave
                    coef_pl = cfact * (self.AREA_ave ** 2) / large_step_dt * d2T_dx2_pl
                    kh_max_broadcast_pl = self.Kh_max[None, :, None]
                    self.Kh_coef_pl = np.clip(coef_pl, self.Kh_coef_minlim, kh_max_broadcast_pl)

                    KH_coef_h[:, :, kminp1:kmax+1, :] = 0.5 * (
                        self.Kh_coef[:, :, kminp1:kmax+1, :] +
                        self.Kh_coef[:, :, kmin:kmax,     :]
                    )
                    KH_coef_h[:, :, kminm1, :] = rdtype(0.0)
                    KH_coef_h[:, :, kmin,   :] = rdtype(0.0)
                    KH_coef_h[:, :, kmaxp1, :] = rdtype(0.0)

                    KH_coef_h_pl[:, kminp1:kmax+1, :] = 0.5 * (
                        self.Kh_coef_pl[:, kminp1:kmax+1, :] +
                        self.Kh_coef_pl[:, kmin:kmax,     :]
                    )
                    KH_coef_h_pl[:, kminm1, :] = rdtype(0.0)
                    KH_coef_h_pl[:, kmin,   :] = rdtype(0.0)
                    KH_coef_h_pl[:, kmaxp1, :] = rdtype(0.0)
                else:
                    KH_coef_h[:, :, :, :] = self.Kh_coef
                    KH_coef_h_pl[:, :, :] = self.Kh_coef_pl

                wk    = rhog * CVdry * self.Kh_coef
                wk_pl = rhog_pl * CVdry * self.Kh_coef_pl
                o[4], opl[4] = oprt.OPRT_diffusion(
                    vtmp_d[:, :, :, :, 4], vtmp_pl_d[:, :, :, 4],
                    wk, wk_pl,
                    oprt.OPRT_coef_intp, oprt.OPRT_coef_intp_pl,
                    oprt.OPRT_coef_diff, oprt.OPRT_coef_diff_pl,
                    grd, rdtype, resident=True,
                )

                wk    = rhog * self.hdiff_fact_rho * self.Kh_coef
                wk_pl = rhog_pl * self.hdiff_fact_rho * self.Kh_coef_pl
                o[5], opl[5] = oprt.OPRT_diffusion(
                    vtmp_d[:, :, :, :, 5], vtmp_pl_d[:, :, :, 5],
                    wk, wk_pl,
                    oprt.OPRT_coef_intp, oprt.OPRT_coef_intp_pl,
                    oprt.OPRT_coef_diff, oprt.OPRT_coef_diff_pl,
                    grd, rdtype, resident=True,
                )
            else:
                o[4], opl[4] = oprt.OPRT_laplacian(
                    vtmp_d[:, :, :, :, 4], vtmp_pl_d[:, :, :, 4],
                    oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl, rdtype,
                    resident=True,
                )
                o[5], opl[5] = oprt.OPRT_laplacian(
                    vtmp_d[:, :, :, :, 5], vtmp_pl_d[:, :, :, 5],
                    oprt.OPRT_coef_lap, oprt.OPRT_coef_lap_pl, rdtype,
                    resident=True,
                )

            # assemble + negate on device (matches vtmp = -vtmp2)
            vtmp_d    = -xp.stack(o,   axis=-1)
            vtmp_pl_d = -xp.stack(opl, axis=-1)

            # halo exchange. Stage B (on-device): hand the jax arrays straight to
            # COMM_data_transfer, which routes to the on-device path and returns
            # the updated device arrays -> vtmp never drains. Stage A (default):
            # drain once, host COMM, re-upload for next iter.
            if _ondevice_comm:
                vtmp_d, vtmp_pl_d = comm.COMM_data_transfer(vtmp_d, vtmp_pl_d)
            else:
                vtmp[:, :, :, :, :] = bk.to_numpy(vtmp_d)
                vtmp_pl[:, :, :, :] = bk.to_numpy(vtmp_pl_d)
                comm.COMM_data_transfer(vtmp, vtmp_pl)
                vtmp_d    = xp.asarray(vtmp)
                vtmp_pl_d = xp.asarray(vtmp_pl)

        if _fuse_lap and _run_eager_loop:
            # warm-up done: build + cache the fused laploop jit (oprt/grd + the COMM
            # core fn + lap_order captured static; vtmp + the 4 wk arrays traced). The
            # COMM core (_get_ondevice_comm_fn) is the pure gather->sendrecv->scatter
            # jit graph -- calling it inside this outer jit inlines it (mpi4jax sendrecv
            # composes under nested jit, the validated fori_loop pattern). Mirrors the
            # eager loop above exactly -> bit-exact / machine-eps (XLA may FMA-reassoc).
            _oprt_c = oprt; _grd_c = grd; _rdt = rdtype; _lap = self.lap_order_hdiff
            _comm_fn = comm._get_ondevice_comm_fn(
                vtmp_d.shape[2], vtmp_d.shape[4], vtmp_d.dtype)
            def _laploop_fn(_v, _vpl, _wk4, _wk4pl, _wk5, _wk5pl):
                for _p in range(_lap):
                    o = [None] * 6
                    opl = [None] * 6
                    for _c in range(4):
                        o[_c], opl[_c] = _oprt_c.OPRT_laplacian(
                            _v[:, :, :, :, _c], _vpl[:, :, :, _c],
                            _oprt_c.OPRT_coef_lap, _oprt_c.OPRT_coef_lap_pl, _rdt,
                            resident=True)
                    if _p == _lap - 1:
                        o[4], opl[4] = _oprt_c.OPRT_diffusion(
                            _v[:, :, :, :, 4], _vpl[:, :, :, 4], _wk4, _wk4pl,
                            _oprt_c.OPRT_coef_intp, _oprt_c.OPRT_coef_intp_pl,
                            _oprt_c.OPRT_coef_diff, _oprt_c.OPRT_coef_diff_pl,
                            _grd_c, _rdt, resident=True)
                        o[5], opl[5] = _oprt_c.OPRT_diffusion(
                            _v[:, :, :, :, 5], _vpl[:, :, :, 5], _wk5, _wk5pl,
                            _oprt_c.OPRT_coef_intp, _oprt_c.OPRT_coef_intp_pl,
                            _oprt_c.OPRT_coef_diff, _oprt_c.OPRT_coef_diff_pl,
                            _grd_c, _rdt, resident=True)
                    else:
                        o[4], opl[4] = _oprt_c.OPRT_laplacian(
                            _v[:, :, :, :, 4], _vpl[:, :, :, 4],
                            _oprt_c.OPRT_coef_lap, _oprt_c.OPRT_coef_lap_pl, _rdt,
                            resident=True)
                        o[5], opl[5] = _oprt_c.OPRT_laplacian(
                            _v[:, :, :, :, 5], _vpl[:, :, :, 5],
                            _oprt_c.OPRT_coef_lap, _oprt_c.OPRT_coef_lap_pl, _rdt,
                            resident=True)
                    _v   = -xp.stack(o,   axis=-1)
                    _vpl = -xp.stack(opl, axis=-1)
                    _v, _vpl = _comm_fn(_v, _vpl)
                return _v, _vpl
            self._hdiff_laploop_jit = bk.jax.jit(_laploop_fn)

        # Stage C (keep_device): hand the device vtmp back to the caller for an
        # on-device tendency -- skip the post-loop drain entirely. Pair with
        # on-device COMM so vtmp also never drains inside the loop.
        if keep_device:
            return vtmp_d, vtmp_pl_d

        # Stage B kept vtmp on device for the whole loop; drain once now so the
        # post-loop host tendency code (reads numpy vtmp/vtmp_pl) sees final values.
        # (Stage A already left them current via the per-iter drain.)
        if _ondevice_comm:
            vtmp[:, :, :, :, :] = bk.to_numpy(vtmp_d)
            vtmp_pl[:, :, :, :] = bk.to_numpy(vtmp_pl_d)

        return None, None

    def _divdamp_post_comm_kernel(self, vtmp2_d, vtmp2_pl_d, grd, oprt):
        """Run the fused post-COMM divdamp island (lap_order==2) on device arrays
        vtmp2_d / vtmp2_pl_d (already on device + already halo-exchanged) and
        return the 6 device outputs (gdx,gdy,gdz,gdx_pl,gdy_pl,gdz_pl). Shared by
        the default post-COMM-only fused path (passes xp.asarray(host vtmp2)) and
        the STEP-7 full-resident path (passes the already-device vtmp2). Builds /
        caches the kernel + cfg + device consts on first use."""
        xp = bk.xp
        if getattr(self, "_divdamp_pc_kernel", None) is None:
            self._dd_pc_cfg = OprtDivdampCfg(
                have_pl=adm.ADM_have_pl, gmax=adm.ADM_gmax,
                gslf_pl=adm.ADM_gslf_pl, gmin_pl=adm.ADM_gmin_pl,
                gmax_pl=adm.ADM_gmax_pl, k0=adm.ADM_K0,
                TI=adm.ADM_TI, TJ=adm.ADM_TJ,
                XDIR=grd.GRD_XDIR, YDIR=grd.GRD_YDIR, ZDIR=grd.GRD_ZDIR,
            )
            self._hz_pc_cfg = HorizontalizeVecCfg(
                have_pl=adm.ADM_have_pl,
                XDIR=grd.GRD_XDIR, YDIR=grd.GRD_YDIR, ZDIR=grd.GRD_ZDIR,
            )
            self._divdamp_pc_kernel = bk.maybe_jit(
                compute_divdamp_post_comm, static_argnames=("dd_cfg", "hz_cfg", "xp"),
            )
        _d = bk.device_consts(self, "divdamp_pc", lambda: {
            "coef_intp": oprt.OPRT_coef_intp, "coef_diff": oprt.OPRT_coef_diff,
            "coef_intp_pl": oprt.OPRT_coef_intp_pl, "coef_diff_pl": oprt.OPRT_coef_diff_pl,
            "divdamp_coef": self.divdamp_coef, "divdamp_coef_pl": self.divdamp_coef_pl,
            "GRD_x": grd.GRD_x, "GRD_x_pl": grd.GRD_x_pl, "rscale": grd.GRD_rscale,
        })
        return self._divdamp_pc_kernel(
            vtmp2_d, vtmp2_pl_d,
            _d["divdamp_coef"], _d["divdamp_coef_pl"],
            _d["coef_intp"], _d["coef_diff"], _d["coef_intp_pl"], _d["coef_diff_pl"],
            _d["GRD_x"], _d["GRD_x_pl"], _d["rscale"],
            dd_cfg=self._dd_pc_cfg, hz_cfg=self._hz_pc_cfg, xp=xp,
        )

    def numfilter_divdamp(self,
        rhogvx, rhogvx_pl,    # [IN]
        rhogvy, rhogvy_pl,    # [IN]
        rhogvz, rhogvz_pl,    # [IN]
        rhogw,  rhogw_pl,     # [IN]
        gdx,    gdx_pl,       # [OUT]  #check this   !use undef values for debug in arrays allocated in this function
        gdy,    gdy_pl,       # [OUT]
        gdz,    gdz_pl,       # [OUT]
        gdvz,   gdvz_pl,      # [OUT]
        cnst, comm, grd, oprt, vmtr, src, rdtype,
        resident=False,
        resident_keep_host=False,
    ):

        prf.PROF_rapstart('____numfilter_divdamp',2)


        # if prc.prc_myrank == 0:
        #     print(grd.GRD_x[6, 5, 0, 0, grd.GRD_XDIR])#, file=log_file)
        #     print(grd.GRD_x[6, 5, 0, 0, grd.GRD_YDIR])#, file=log_file)
        #     print(grd.GRD_x[6, 5, 0, 0, grd.GRD_ZDIR])#, file=log_file)
        #     #prc.prc_mpistop(std.io_l, std.fname_log)

        gall_1d = adm.ADM_gall_1d
        gall_pl = adm.ADM_gall_pl
        kall = adm.ADM_kall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
        lall_pl = adm.ADM_lall_pl 

        cnv      = np.full((adm.ADM_shape), cnst.CONST_UNDEF, dtype=rdtype)
        cnv_pl   = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        vtmp     = np.full((adm.ADM_shape + (3,)), cnst.CONST_UNDEF, dtype=rdtype)
        vtmp2    = np.full((adm.ADM_shape + (3,)), cnst.CONST_UNDEF, dtype=rdtype)

        vtmp_pl  = np.zeros((adm.ADM_shape_pl + (3,)), dtype=rdtype)
        vtmp2_pl = np.zeros((adm.ADM_shape_pl + (3,)), dtype=rdtype)
        
         #vtmp_pl  = np.full((adm.ADM_shape_pl 3,), dtype=rdtype)
         #vtmp2_pl = np.full((adm.ADM_shape_pl 3,), dtype=rdtype)

        if not self.NUMFILTER_DOdivdamp:

            gdx   = rdtype(0.0)
            gdy   = rdtype(0.0)
            gdz   = rdtype(0.0)
            gdvz  = rdtype(0.0)
            gdx_pl  = rdtype(0.0)
            gdy_pl  = rdtype(0.0)
            gdz_pl  = rdtype(0.0)
            gdvz_pl = rdtype(0.0)
            # gdx   = np.zeros_like(rhogvx)
            # gdy   = np.zeros_like(rhogvx)
            # gdz   = np.zeros_like(rhogvx)
            # gdvz  = np.zeros_like(rhogvx)
            # gdx_pl  = np.zeros_like(rhogvx_pl)
            # gdy_pl  = np.zeros_like(rhogvx_pl)
            # gdz_pl  = np.zeros_like(rhogvx_pl)
            # gdvz_pl = np.zeros_like(rhogvx_pl)

            prf.PROF_rapend('____numfilter_divdamp',2)
            return
        #endif

        #--- 3D divergence divdamp
        _gd_done = False
        _full_fuse = (
            self.lap_order_divdamp == 2 and bk.type == "jax" and getattr(
                self, "use_fuse_divdamp_full",
                os.environ.get("PYNICAM_FUSE_DIVDAMP_FULL", "1") != "0")
        )

        if _full_fuse:
            # STEP-7: OPRT3D_divdamp -> on-device COMM -> post-COMM island as ONE
            # device-resident chain. vtmp2 stays a jax array end-to-end (no D2H
            # after OPRT3D, no host COMM, no asarray before post-COMM); only the
            # final gd* are drained once. Bit-exact vs the default fused path
            # (identical pure kernels; on-device COMM is bit-exact vs host COMM).
            # Gated PYNICAM_FUSE_DIVDAMP_FULL (default off), jax + lap_order==2 only.
            xp = bk.xp
            _dx, _dy, _dz, _dxp, _dyp, _dzp = oprt._oprt3d_divdamp_device(
                rhogvx, rhogvx_pl, rhogvy, rhogvy_pl, rhogvz, rhogvz_pl,
                rhogw, rhogw_pl,
                oprt.OPRT_coef_intp, oprt.OPRT_coef_intp_pl,
                oprt.OPRT_coef_diff, oprt.OPRT_coef_diff_pl,
                grd, vmtr,
            )
            vtmp2_d    = xp.stack((_dx, _dy, _dz), axis=-1)
            vtmp2_pl_d = xp.stack((_dxp, _dyp, _dzp), axis=-1)
            # halo exchange on device (jax array -> on-device COMM, returns updated)
            vtmp2_d, vtmp2_pl_d = comm.COMM_data_transfer(vtmp2_d, vtmp2_pl_d)
            _gx, _gy, _gz, _gxp, _gyp, _gzp = self._divdamp_post_comm_kernel(
                vtmp2_d, vtmp2_pl_d, grd, oprt,
            )
            # RESIDENT_DIVDAMP_OUT: when resident_keep_host, still drain the device
            # gd* to host (cheap pinned D2H) so host-side readers (numpy g_TEND body,
            # the _inv non-TIME_split fallback, numfilter_divdamp_2d independence)
            # stay valid, AND return the device handles below so the resident caller
            # skips the asarray H2D re-upload. asarray(to_numpy(_gx)) == _gx (f64
            # identity), so the device path is bit-exact vs the host re-upload.
            if (not resident) or resident_keep_host:
                if os.environ.get("PYNICAM_RESIDENT_VI_DRAINOUT", "0") == "0":   # RC-32: gd* host dead (poison PASS); device handles returned below
                    gdx[:, :, :, :] = bk.to_numpy(_gx)
                    gdy[:, :, :, :] = bk.to_numpy(_gy)
                    gdz[:, :, :, :] = bk.to_numpy(_gz)
                if "divdamp" in os.environ.get("PYNICAM_VI_POISON", ""):   # RC-32 divdamp poison
                    gdx[:, :, :, :] = np.nan; gdy[:, :, :, :] = np.nan; gdz[:, :, :, :] = np.nan
                # RES-CAPSTONE-46 (Track B unit C): pole gd*_pl host drains DEAD (vi reads the
                # device pole handles _ddvxp_d.. @vp0; exact analog of the regular RC-32).
                if adm.ADM_have_pl and os.environ.get("PYNICAM_RESIDENT_DIVDAMP_OUT_PL", "0") == "0":
                    gdx_pl[:, :, :] = bk.to_numpy(_gxp)
                    gdy_pl[:, :, :] = bk.to_numpy(_gyp)
                    gdz_pl[:, :, :] = bk.to_numpy(_gzp)
            _gd_done = True

        if not _full_fuse:
            oprt.OPRT3D_divdamp(
                vtmp2 [:, :, :, :, 0],   vtmp2_pl [:, :, :, 0],  # [OUT]
                vtmp2 [:, :, :, :, 1],   vtmp2_pl [:, :, :, 1],  # [OUT]
                vtmp2 [:, :, :, :, 2],   vtmp2_pl [:, :, :, 2],  # [OUT]
                rhogvx,      rhogvx_pl,     # [IN]
                rhogvy,      rhogvy_pl,     # [IN]
                rhogvz,      rhogvz_pl,     # [IN]
                rhogw,       rhogw_pl ,     # [IN]
                oprt.OPRT_coef_intp,     oprt.OPRT_coef_intp_pl, # [IN]
                oprt.OPRT_coef_diff,     oprt.OPRT_coef_diff_pl, # [IN]
                grd, vmtr, rdtype,
            )

        # with open (std.fname_log, 'a') as log_file:
        #     print(f"checking pl: n, ij, ijp1: ij=:, k=2, l=0", file=log_file)
        #     print("vtmp2_pl", vtmp2_pl[:, 2, 0, 0], file=log_file)
        #     print("vtmp2_pl", vtmp2_pl[:, 2, 0, 1], file=log_file)
        #     print("vtmp2_pl", vtmp2_pl[:, 2, 0, 2], file=log_file)


        # with open (std.fname_log, 'a') as log_file:
        #     print("OPRT3D_divdamp, poles: ", file=log_file)
        #     print("vtmp2_pl[0,2,0,:]", vtmp2_pl[0,2,0,:], file=log_file)
        #     print("vtmp2_pl[:,10,1,0]", vtmp2_pl[:,10,1,0], file=log_file)    
        #     print("vtmp2_pl[:,10,1,1]", vtmp2_pl[:,10,1,1], file=log_file)    
        #     print("vtmp2_pl[:,10,1,2]", vtmp2_pl[:,10,1,2], file=log_file)    

        if not _gd_done and self.lap_order_divdamp == 2 and getattr(
            self, "use_fused_divdamp", os.environ.get("PYNICAM_FUSE_DIVDAMP", "1") != "0"
        ):
            # Post-COMM-only fused island (lap_order==2): the post-COMM chain
            #   -vtmp2 -> OPRT_divdamp -> coef*vtmp2 -> horizontalize
            # runs on-device as ONE kernel (kernels/divdamppostcomm.py),
            # collapsing 3 host<->device round-trips into one. The COMM stays on
            # host. (Default fast path; the STEP-7 _full_fuse branch above instead
            # keeps the COMM on-device for the whole chain.) Set
            # use_fused_divdamp=False / PYNICAM_FUSE_DIVDAMP=0 for the original below.
            comm.COMM_data_transfer(vtmp2, vtmp2_pl)
            xp = bk.xp
            _gx, _gy, _gz, _gxp, _gyp, _gzp = self._divdamp_post_comm_kernel(
                xp.asarray(vtmp2), xp.asarray(vtmp2_pl), grd, oprt,
            )
            if os.environ.get("PYNICAM_RESIDENT_VI_DRAINOUT", "0") == "0":   # RC-32: gd* host dead (poison PASS)
                gdx[:, :, :, :] = bk.to_numpy(_gx)
                gdy[:, :, :, :] = bk.to_numpy(_gy)
                gdz[:, :, :, :] = bk.to_numpy(_gz)
            if "divdamp" in os.environ.get("PYNICAM_VI_POISON", ""):   # RC-32 divdamp poison
                gdx[:, :, :, :] = np.nan; gdy[:, :, :, :] = np.nan; gdz[:, :, :, :] = np.nan
            # RES-CAPSTONE-46 (Track B unit C): pole gd*_pl host drains DEAD (vi reads device handles).
            if adm.ADM_have_pl and os.environ.get("PYNICAM_RESIDENT_DIVDAMP_OUT_PL", "0") == "0":
                gdx_pl[:, :, :] = bk.to_numpy(_gxp)
                gdy_pl[:, :, :] = bk.to_numpy(_gyp)
                gdz_pl[:, :, :] = bk.to_numpy(_gzp)
            _gd_done = True

        if not _gd_done:
            # --- original per-kernel path (general lap_order; fallback) ---
            if self.lap_order_divdamp > 1:
                for p in range(self.lap_order_divdamp - 1):
                    comm.COMM_data_transfer(vtmp2, vtmp2_pl)
                    vtmp[:, :, :, :, :] = -vtmp2[:, :, :, :, :]
                    vtmp_pl[:, :, :, :] = -vtmp2_pl[:, :, :, :]
                    oprt.OPRT_divdamp(
                        vtmp2[:, :, :, :, 0], vtmp2_pl[:, :, :, 0],  # [OUT]
                        vtmp2[:, :, :, :, 1], vtmp2_pl[:, :, :, 1],  # [OUT]
                        vtmp2[:, :, :, :, 2], vtmp2_pl[:, :, :, 2],  # [OUT]
                        vtmp [:, :, :, :, 0], vtmp_pl [:, :, :, 0],  # [IN]
                        vtmp [:, :, :, :, 1], vtmp_pl [:, :, :, 1],  # [IN]
                        vtmp [:, :, :, :, 2], vtmp_pl [:, :, :, 2],  # [IN]
                        oprt.OPRT_coef_intp, oprt.OPRT_coef_intp_pl,  # [IN]
                        oprt.OPRT_coef_diff, oprt.OPRT_coef_diff_pl,  # [IN]
                        cnst, grd, rdtype,
                    )

            gdx[:, :, :, :] = self.divdamp_coef * vtmp2[:, :, :, :, 0]
            gdy[:, :, :, :] = self.divdamp_coef * vtmp2[:, :, :, :, 1]
            gdz[:, :, :, :] = self.divdamp_coef * vtmp2[:, :, :, :, 2]

            if adm.ADM_have_pl:
                gdx_pl[:, :, :] = self.divdamp_coef_pl * vtmp2_pl[:, :, :, 0]
                gdy_pl[:, :, :] = self.divdamp_coef_pl * vtmp2_pl[:, :, :, 1]
                gdz_pl[:, :, :] = self.divdamp_coef_pl * vtmp2_pl[:, :, :, 2]

            oprt.OPRT_horizontalize_vec(
                gdx, gdx_pl, gdy, gdy_pl, gdz, gdz_pl, grd, rdtype,
            )

        #prc.prc_mpistop(std.io_l, std.fname_log)

        if self.NUMFILTER_DOdivdamp_v:

            src.SRC_flux_convergence(
                rhogvx[:,:,:,:], rhogvx_pl[:,:,:], # [IN]
                rhogvy[:,:,:,:], rhogvy_pl[:,:,:], # [IN]
                rhogvz[:,:,:,:], rhogvz_pl[:,:,:], # [IN]
                rhogw [:,:,:,:], rhogw_pl [:,:,:], # [IN]
                cnv   [:,:,:,:], cnv_pl   [:,:,:], # [OUT]
                src.I_SRC_default,                 # [IN]
                grd, oprt, vmtr, rdtype, 
            )

                
            k_range = slice(kmin + 1, kmax + 1)
            gdvz[:, :, k_range, :] = self.divdamp_coef_v * (
                cnv[:, :, k_range, :] - cnv[:, :, k_range.start - 1 : k_range.stop - 1, :]
            ) * grd.GRD_rdgzh[k_range, np.newaxis]

            # Zero boundaries
            gdvz[:, :, kmin - 1, :] = rdtype(0.0)
            gdvz[:, :, kmin,     :] = rdtype(0.0)
            gdvz[:, :, kmax + 1, :] = rdtype(0.0)


            if adm.ADM_have_pl:
                # Vectorized over k
                #k_range = slice(kmin + 1, kmax + 1)
                gdvz_pl[:, k_range, :] = (
                    self.divdamp_coef_v
                    * (cnv_pl[:, k_range, :] - cnv_pl[:, k_range.start - 1 : k_range.stop - 1, :])
                    * grd.GRD_rdgzh[k_range, np.newaxis]
                )

                # Zero out boundaries
                gdvz_pl[:, kmin - 1, :] = rdtype(0.0)
                gdvz_pl[:, kmin,     :] = rdtype(0.0)
                gdvz_pl[:, kmax + 1, :] = rdtype(0.0)


        else:

            gdvz[:, :, :, :] = rdtype(0.0)

            if adm.ADM_have_pl:
                gdvz_pl[:, :, :] = rdtype(0.0)
            #endif

        #endif

        if resident:
            # device-resident: return jax gdx/gdy/gdz (from _full_fuse, undrained) +
            # the gdvz (vertical divdamp) slot. RES-CAPSTONE-10: when
            # NUMFILTER_DOdivdamp_v is off (alpha_v==0, the DC/JW config) gdvz was
            # filled with zeros in the else-branch above, so building the device
            # zeros directly avoids asarray-uploading a ~2.2GB host-zeros array
            # (the #1 H2D site after RC-9). Bit-identical (zeros == zeros). When
            # DOdivdamp_v is on, fall back to the asarray of the host vertical part.
            # Gate PYNICAM_RESIDENT_GDVZ (default on).
            xp = bk.xp
            _gdvz_resident = (
                (not self.NUMFILTER_DOdivdamp_v)
                and os.environ.get("PYNICAM_RESIDENT_GDVZ", "1") != "0"
            )
            if _gdvz_resident:
                _gdvz_d    = xp.zeros(adm.ADM_shape,    dtype=rdtype)
                _gdvz_pl_d = xp.zeros(adm.ADM_shape_pl, dtype=rdtype)
            else:
                _gdvz_d    = xp.asarray(gdvz)
                _gdvz_pl_d = xp.asarray(gdvz_pl)
            prf.PROF_rapend('____numfilter_divdamp',2)
            return (_gx, _gy, _gz, _gxp, _gyp, _gzp, _gdvz_d, _gdvz_pl_d)

        prf.PROF_rapend('____numfilter_divdamp',2)

        return
    
    def numfilter_divdamp_2d(self,
        rhogvx, rhogvx_pl, 
        rhogvy, rhogvy_pl, 
        rhogvz, rhogvz_pl, 
        gdx,    gdx_pl,    
        gdy,    gdy_pl,    
        gdz,    gdz_pl,
        cnst, comm, grd, oprt, rdtype,
    ):
        
        prf.PROF_rapstart('____numfilter_divdamp_2d',2)   
        
        gall_1d = adm.ADM_gall_1d
        kall = adm.ADM_kall
        lall = adm.ADM_lall

        vtmp     = np.full((adm.ADM_shape    + (3,)), cnst.CONST_UNDEF, dtype=rdtype)
        vtmp2    = np.full((adm.ADM_shape    + (3,)), cnst.CONST_UNDEF, dtype=rdtype)
        vtmp_pl  = np.full((adm.ADM_shape_pl + (3,)), cnst.CONST_UNDEF, dtype=rdtype)
        vtmp2_pl = np.full((adm.ADM_shape_pl + (3,)), cnst.CONST_UNDEF, dtype=rdtype)


        if not self.NUMFILTER_DOdivdamp_2d:

            gdx[:, :, :, :] = rdtype(0.0)
            gdy[:, :, :, :] = rdtype(0.0)
            gdz[:, :, :, :] = rdtype(0.0)
            gdx_pl[:, :, :] = rdtype(0.0)
            gdy_pl[:, :, :] = rdtype(0.0)
            gdz_pl[:, :, :] = rdtype(0.0)
              
            prf.PROF_rapend('____numfilter_divdamp_2d',2)
            return  
        #endif
    
        #--- 2D dinvergence divdamp
        oprt.OPRT_divdamp(
            vtmp2 [:, :, :, :, 0],   vtmp2_pl [:, :, :, 0],  # [OUT]
            vtmp2 [:, :, :, :, 1],   vtmp2_pl [:, :, :, 1],  # [OUT]
            vtmp2 [:, :, :, :, 2],   vtmp2_pl [:, :, :, 2],  # [OUT]
            rhogvx[:, :, :, :, 0],   rhogvx_pl[:, :, :, 0],  # [IN]
            rhogvy[:, :, :, :, 1],   rhogvy_pl[:, :, :, 1],  # [IN]
            rhogvz[:, :, :, :, 2],   rhogvz_pl[:, :, :, 2],  # [IN]
            oprt.OPRT_coef_intp,   oprt.OPRT_coef_intp_pl,   # [IN]
            oprt.OPRT_coef_diff,   oprt.OPRT_coef_diff_pl,   # [IN]
            grd, rdtype,
        )

        if self.lap_order_divdamp > 1:
            for p in range(self.lap_order_divdamp-1):

                comm.COMM_data_transfer(vtmp2, vtmp2_pl)

                #--- note : sign changes
                # for iv in range(3):  
                #     for l in range(lall):
                #         for k in range(kall):
                #            vtmp[:, :, k, l, iv] = -vtmp2[:, :, k, l, iv]
                vtmp[:, :, :, :, :] = -vtmp2[:, :, :, :, :]
                        #end k loop
                    #end l loop
                #end iv loop

                vtmp_pl[:, :, :, :] = -vtmp2_pl[:, :, :, :]


                #--- 2D dinvergence divdamp
                oprt.OPRT_divdamp(
                    vtmp2 [:, :, :, :, 0],   vtmp2_pl [:, :, :, 0], # [OUT]
                    vtmp2 [:, :, :, :, 1],   vtmp2_pl [:, :, :, 1], # [OUT]
                    vtmp2 [:, :, :, :, 2],   vtmp2_pl [:, :, :, 2], # [OUT]
                    vtmp  [:, :, :, :, 0],   vtmp_pl  [:, :, :, 0], # [IN]
                    vtmp  [:, :, :, :, 1],   vtmp_pl  [:, :, :, 1], # [IN]
                    vtmp  [:, :, :, :, 2],   vtmp_pl  [:, :, :, 2], # [IN]
                    oprt.OPRT_coef_intp,   oprt.OPRT_coef_intp_pl,  # [IN]
                    oprt.OPRT_coef_diff,   oprt.OPRT_coef_diff_pl,  # [IN]
                    grd, rdtype,
                )

            #enddo ! lap_order
        #endif

        #--- X coeffcient

        # for l in range(lall):
        #     for k in range(kall):  # assuming 'kall' is defined appropriately
        gdx[:, :, :, :] = self.divdamp_2d_coef * vtmp2[:, :, :, :, 0]
        gdy[:, :, :, :] = self.divdamp_2d_coef * vtmp2[:, :, :, :, 1]
        gdz[:, :, :, :] = self.divdamp_2d_coef * vtmp2[:, :, :, :, 2]
            #end k loop
        #end l loop

        if adm.ADM_have_pl:
            gdx_pl[:, :, :] = self.divdamp_2d_coef_pl * vtmp2_pl[:, :, :, 0]
            gdy_pl[:, :, :] = self.divdamp_2d_coef_pl * vtmp2_pl[:, :, :, 1]
            gdz_pl[:, :, :] = self.divdamp_2d_coef_pl * vtmp2_pl[:, :, :, 2]
        #endif

        oprt.OPRT_horizontalize_vec(
            gdx, gdx_pl, # [INOUT] 
            gdy, gdy_pl, # [INOUT]
            gdz, gdz_pl, # [INOUT]
            grd, rdtype,
        )

        prf.PROF_rapend('____numfilter_divdamp_2d',2)

        return
