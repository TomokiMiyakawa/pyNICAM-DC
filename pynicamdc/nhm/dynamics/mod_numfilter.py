import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf


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
    ):
        
        prf.PROF_rapstart('____numfilter_hdiffusion',2)

        KH_coef_h         = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        KH_coef_lap1_h    = np.full((adm.ADM_shape),    cnst.CONST_UNDEF, dtype=rdtype)
        KH_coef_h_pl      = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)
        KH_coef_lap1_h_pl = np.full((adm.ADM_shape_pl), cnst.CONST_UNDEF, dtype=rdtype)

        fact = np.full((adm.ADM_kall,), cnst.CONST_UNDEF, dtype=rdtype)

        wk     = np.full((adm.ADM_shape),        cnst.CONST_UNDEF, dtype=rdtype)
        rhog_h = np.full((adm.ADM_shape),        cnst.CONST_UNDEF, dtype=rdtype)
        vtmp   = np.full((adm.ADM_shape + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        vtmp2  = np.full((adm.ADM_shape + (6,)), cnst.CONST_UNDEF, dtype=rdtype)

        qtmp      = np.full((adm.ADM_shape + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        qtmp2     = np.full((adm.ADM_shape + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        qtmp_lap1 = np.full((adm.ADM_shape + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)   

        qtmp_pl      = np.full((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        qtmp2_pl     = np.full((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)
        qtmp_lap1_pl = np.full((adm.ADM_shape_pl + (rcnf.TRC_vmax,)), cnst.CONST_UNDEF, dtype=rdtype)   

        wk_pl     = np.full((adm.ADM_shape_pl),        cnst.CONST_UNDEF, dtype=rdtype)
        rhog_h_pl = np.full((adm.ADM_shape_pl),        cnst.CONST_UNDEF, dtype=rdtype)
        vtmp_pl   = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)
        vtmp2_pl  = np.full((adm.ADM_shape_pl + (6,)), cnst.CONST_UNDEF, dtype=rdtype)


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

        if self.hdiff_nonlinear:
            self.height_factor(adm.ADM_kall, grd.GRD_gz, grd.GRD_htop, self.ZD_hdiff_nl, fact, cnst, rdtype)
            kh_max = (rdtype(1.0) - fact) * self.Kh_coef_maxlim + fact * self.Kh_coef_minlim  
        #endif


        # Extract weights from VMTR_C2Wfact
        fact1 = vmtr.VMTR_C2Wfact[:, :, kmin:kmaxp2, :, 0]  # shape (i, j, k, l)
        fact2 = vmtr.VMTR_C2Wfact[:, :, kmin:kmaxp2, :, 1]

        # Interpolate rhog to cell center
        rhog_h[:, :, kmin:kmaxp2, :] = (
            fact1 * rhog[:, :, kmin:kmaxp2, :] +
            fact2 * rhog[:, :, kminm1:kmaxp1,   :]
        )

        rhog_h[:, :, kminm1, :] = rdtype(0.0)


        #if ADM_have_pl:
        fact1_pl = vmtr.VMTR_C2Wfact_pl[:, kmin:kmaxp2, :, 0]
        fact2_pl = vmtr.VMTR_C2Wfact_pl[:, kmin:kmaxp2, :, 1]

        rhog_h_pl[:, kmin:kmaxp2, :] = (
            fact1_pl * rhog_pl[:, kmin:kmaxp2, :] +
            fact2_pl * rhog_pl[:, kminm1:kmaxp1,   :]
        )

        rhog_h_pl[:, kminm1, :] = rdtype(0.0)


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


        # high order laplacian        
        for p in range(self.lap_order_hdiff):  # 2 (0 and 1)

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




        #--- Update tendency

        # Vectorized main domain update
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

        #endif


        # with open (std.fname_log, 'a') as log_file:
        #     print("tendency 0: ", file=log_file)
        #     print("tendency[6,5,2,0,:]", file=log_file)
        #     print( tendency[6,5,2,0,:] , file=log_file)
        #     print("tendency[6,5,37,0,:]", file=log_file)
        #     print( tendency[6,5,37,0,:] , file=log_file)
        #     #print("vtmp_lap1[6,5,2,0,:]", file=log_file)
        #     #print( vtmp_lap1[6,5,2,0,:] , file=log_file)


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

        prf.PROF_rapend('____numfilter_hdiffusion',2)

        return
    
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

        if self.lap_order_divdamp > 1:
            for p in range(self.lap_order_divdamp-1):

                # with open (std.fname_log, 'a') as log_file:
                #     print("",file=log_file)
                #     print(f"checking before transfer", file=log_file)
                #     print("vtmp2", vtmp2[1, 16, 2, 2, :], file=log_file)
                    

                comm.COMM_data_transfer( vtmp2, vtmp2_pl )


                # with open (std.fname_log, 'a') as log_file:
                #     print(f"checking after transfer", file=log_file)# , pl: n, ij, ijp1: ij=:, k=2, l=0", file=log_file)
                #     #print("vtmp2_pl", vtmp2_pl[:, 2, 0, 0], file=log_file)
                #     #print("vtmp2_pl", vtmp2_pl[:, 2, 0, 1], file=log_file)
                #     #print("vtmp2_pl", vtmp2_pl[:, 2, 0, 2], file=log_file)
                #     print("vtmp2", vtmp2[1, 16, 2, 2, :], file=log_file)
                #     print("",file=log_file)

                #--- note : sign changes

                # for iv in range(3):  
                #     for l in range(lall):
                #         for k in range(kall):
                #             vtmp[:, :, k, l, iv] = -vtmp2[:, :, k, l, iv]
                vtmp[:, :, :, :, :] = -vtmp2[:, :, :, :, :]
                        #end k loop
                    #end l loop
                #end iv loop

                vtmp_pl[:, :, :, :] = -vtmp2_pl[:, :, :, :]


                #--- 2D dinvergence divdamp
                oprt.OPRT_divdamp(
                    vtmp2[:, :, :, :, 0],   vtmp2_pl[:, :, :, 0],  # [OUT]
                    vtmp2[:, :, :, :, 1],   vtmp2_pl[:, :, :, 1],  # [OUT]
                    vtmp2[:, :, :, :, 2],   vtmp2_pl[:, :, :, 2],  # [OUT]
                    vtmp [:, :, :, :, 0],   vtmp_pl [:, :, :, 0],  # [IN]
                    vtmp [:, :, :, :, 1],   vtmp_pl [:, :, :, 1],  # [IN]
                    vtmp [:, :, :, :, 2],   vtmp_pl [:, :, :, 2],  # [IN]
                    oprt.OPRT_coef_intp,   oprt.OPRT_coef_intp_pl, # [IN]
                    oprt.OPRT_coef_diff,   oprt.OPRT_coef_diff_pl, # [IN]
                    cnst, grd, rdtype,
                )

                # with open (std.fname_log, 'a') as log_file:
                #     print("EEEEE", file=log_file)
                #     print("vtmp_pl[0,3,0,:]", vtmp_pl[0,3,0,:], file=log_file)
                #     print("vtmp2_pl[0,3,0,:]", vtmp2_pl[0,3,0,:], file=log_file)
            # enddo  # lap_order
        #endif

        # with open (std.fname_log, 'a') as log_file:
        #     print("OPRT_divdamp, poles: ", file=log_file)
        #     print("vtmp2_pl[0,2,0,:]", vtmp2_pl[0,2,0,:], file=log_file)
        #     print("oprt.OPRT_coef_intp_pl[:,:,0,0]", oprt.OPRT_coef_intp_pl[:,:,0,0], file=log_file)
        #     print("oprt.OPRT_coef_intp_pl[:,:,1,0]", oprt.OPRT_coef_intp_pl[:,:,1,0], file=log_file)
        #     print("oprt.OPRT_coef_intp_pl[:,:,2,0]", oprt.OPRT_coef_intp_pl[:,:,2,0], file=log_file)
        #     print("oprt.OPRT_coef_diff_pl[:,0,0]", oprt.OPRT_coef_diff_pl[:,0,0], file=log_file)
        #     print("oprt.OPRT_coef_diff_pl[:,1,0]", oprt.OPRT_coef_diff_pl[:,1,0], file=log_file)
        #     print("oprt.OPRT_coef_diff_pl[:,2,0]", oprt.OPRT_coef_diff_pl[:,2,0], file=log_file)

        #--- X coeffcient

        # for l in range(lall):
        #     for k in range(kall):
        #         gdx[:, :, k, l] = self.divdamp_coef[:, :, k, l] * vtmp2[:, :, k, l, 0]
        #         gdy[:, :, k, l] = self.divdamp_coef[:, :, k, l] * vtmp2[:, :, k, l, 1]
        #         gdz[:, :, k, l] = self.divdamp_coef[:, :, k, l] * vtmp2[:, :, k, l, 2]
        
        gdx[:, :, :, :] = self.divdamp_coef * vtmp2[:, :, :, :, 0]
        gdy[:, :, :, :] = self.divdamp_coef * vtmp2[:, :, :, :, 1]
        gdz[:, :, :, :] = self.divdamp_coef * vtmp2[:, :, :, :, 2]

                

            #end k loop
        #end l loop


        #prc.prc_mpistop(std.io_l, std.fname_log)


        if adm.ADM_have_pl:
            gdx_pl[:, :, :] = self.divdamp_coef_pl * vtmp2_pl[:, :, :, 0]
            gdy_pl[:, :, :] = self.divdamp_coef_pl * vtmp2_pl[:, :, :, 1]
            gdz_pl[:, :, :] = self.divdamp_coef_pl * vtmp2_pl[:, :, :, 2]
        #endif


        # with open (std.fname_log, 'a') as log_file:
        #     print("CCCCC", file=log_file)
        #     print("gdx_pl[0,3,0,:]", gdx_pl[0,3,0], file=log_file)
        #     print("gdy_pl[0,3,0,:]", gdy_pl[0,3,0], file=log_file)
        #     print("gdz_pl[0,3,0,:]", gdz_pl[0,3,0], file=log_file)
        #     print("vtmp2_pl[0,3,0,:]", vtmp2_pl[0,3,0,:], file=log_file)
        #     #print("self.divdamp_coef_pl", self.divdamp_coef_pl, file=log_file)


        oprt.OPRT_horizontalize_vec(
            gdx, gdx_pl, # [INOUT] 
            gdy, gdy_pl, # [INOUT]
            gdz, gdz_pl, # [INOUT]
            grd, rdtype,
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
