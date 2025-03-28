import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf


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

    def numfilter_setup(self, fname_in, rcnf, cnst, comm, gtl, grd, gmtr, oprt, vmtr, tim, prgv, tdyn, frc, bndc, bsst, rdtype):

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
        self.gamma_h = rdtype(1.0) / 16.0 / 10.0    # Coefficient for horizontal diffusion
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

        # skip for now
        # call numfilter_divdamp_2d_setup( divdamp_2d_type,      & ! [IN]                                                                                           
        #                                 dep_hgrid,            & ! [IN]                                                                                           
        #                                 lap_order_divdamp_2d, & ! [IN]                                                                                           
        #                                 alpha_d_2d,           & ! [IN]                                                                                           
        #                                 tau_d_2d,             & ! [IN]                                                                                           
        #                                 ZD_d_2d               ) ! [IN]                                                                                           

        Kh_deep_factor        = np.zeros(adm.ADM_kdall, dtype=rdtype)
        Kh_deep_factor_h      = np.zeros(adm.ADM_kdall, dtype=rdtype)
        Kh_lap1_deep_factor   = np.zeros(adm.ADM_kdall, dtype=rdtype)
        Kh_lap1_deep_factor_h = np.zeros(adm.ADM_kdall, dtype=rdtype)
        divdamp_deep_factor   = np.zeros(adm.ADM_kdall, dtype=rdtype)

        if deep_effect:
            print("Sorry, deep_effect is not implemented yet.")
            prc.prc_mpistop(std.io_l, std.fname_log)
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

        e_fold_time    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall),    dtype=rdtype)
        e_fold_time_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)

        self.Kh_coef    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall),    dtype=rdtype)
        self.Kh_coef_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)

        if self.hdiff_type == 'DIRECT':
            if gamma > rdtype(0.0):
                self.NUMFILTER_DOhorizontaldiff = True

            # gamma is an absolute value
            self.Kh_coef[:, :, :] = gamma
            self.Kh_coef_pl[:, :, :] = gamma

        elif self.hdiff_type == 'NONDIM_COEF':
            if gamma > rdtype(0.0):
                self.NUMFILTER_DOhorizontaldiff = True

            large_step_dt = tim.TIME_DTL / rdtype(rcnf.DYN_DIV_NUM)

            # gamma is a non-dimensional number
            if self.dep_hgrid:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kdall):
                        self.Kh_coef[:, k, l] = gamma / large_step_dt * gmtr.GMTR_area[:, l] ** lap_order

                if adm.ADM_have_pl:
                    for l in range(adm.ADM_lall_pl):
                        for k in range(adm.ADM_kdall):
                            self.Kh_coef_pl[:, k, l] = gamma / large_step_dt * gmtr.GMTR_area_pl[:, l] ** lap_order
            else:
                value = gamma / large_step_dt * self.AREA_ave ** lap_order
                self.Kh_coef[:, :, :] = value
                self.Kh_coef_pl[:, :, :] = value

        elif self.hdiff_type == 'E_FOLD_TIME':
            if tau > rdtype(0.0):
                self.NUMFILTER_DOhorizontaldiff = True

            # tau is e-folding time for 2*dx waves
            if self.dep_hgrid:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kdall):
                        self.Kh_coef[:, k, l] = (np.sqrt(gmtr.GMTR_area[:, l]) / PI) ** (2 * lap_order) / (tau + EPS)

                if adm.ADM_have_pl:
                    for l in range(adm.ADM_lall_pl):
                        for k in range(adm.ADM_kdall):
                            self.Kh_coef_pl[:, k, l] = (np.sqrt(gmtr.GMTR_area_pl[:, l]) / PI) ** (2 * lap_order) / (tau + EPS)
            else:
                value = (np.sqrt(self.AREA_ave) / PI) ** (2 * lap_order) / (tau + EPS)
                self.Kh_coef[:, :, :] = value
                self.Kh_coef_pl[:, :, :] = value

        elif self.hdiff_type == 'NONLINEAR1':
            self.NUMFILTER_DOhorizontaldiff = True
            self.hdiff_nonlinear = True

            self.Kh_coef[:, :, :] = rdtype(-999.0)
            self.Kh_coef_pl[:, :, :] = rdtype(-999.0)

        #print("self.hdifftype: ", self.hdiff_type)

        if self.hdiff_type != 'DIRECT' and self.hdiff_type != 'NONLINEAR1':
            if self.smooth_1var:  # Iga 20120721 (add if)
                self.numfilter_smooth_1var(self.Kh_coef, self.Kh_coef_pl, comm, gmtr, oprt, rdtype)

            self.Kh_coef[:, :, :] = np.maximum(self.Kh_coef, self.Kh_coef_minlim)


        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print("-----   Horizontal numerical diffusion   -----", file=log_file)

        if self.NUMFILTER_DOhorizontaldiff:
            if not self.hdiff_nonlinear:
                if self.debug:
                    for l in range(adm.ADM_lall):
                        for k in range(adm.ADM_kdall):
                            e_fold_time[:, k, l] = (np.sqrt(gmtr.GMTR_area[:, l]) / PI) ** (2 * lap_order) / (self.Kh_coef[:, k, l] + EPS)

                    if adm.ADM_have_pl:
                        for l in range(adm.ADM_lall_pl):
                            for k in range(adm.ADM_kdall):
                                e_fold_time_pl[:, k, l] = (np.sqrt(gmtr.GMTR_area_pl[:, l]) / PI) ** (2 * lap_order) / (self.Kh_coef_pl[:, k, l] + EPS)

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
        Kh_coef_lap1    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall), dtype=rdtype)
        Kh_coef_lap1_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)


        if self.hdiff_type_lap1 == 'DIRECT':
            if gamma_lap1 > rdtype(0.0):
                self.NUMFILTER_DOhorizontaldiff_lap1 = True

            # gamma is an absolute value
            Kh_coef_lap1[:, :, :, :]    = gamma_lap1
            Kh_coef_lap1_pl[:, :, :] = gamma_lap1

        elif self.hdiff_type_lap1 == 'NONDIM_COEF':
            if gamma_lap1 > rdtype(0.0):
                self.NUMFILTER_DOhorizontaldiff_lap1 = True

            large_step_dt = tim.TIME_DTL / rdtype(rcnf.DYN_DIV_NUM)

            # gamma is a non-dimensional number
            if self.dep_hgrid:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kdall):
                        Kh_coef_lap1[:, :, k, l] = gamma_lap1 / large_step_dt * gmtr.GMTR_area[:, l]

                if adm.ADM_have_pl:
                    for l in range(adm.ADM_lall_pl):
                        for k in range(adm.ADM_kdall):
                            Kh_coef_lap1_pl[:, k, l] = gamma_lap1 / large_step_dt * gmtr.GMTR_area_pl[:, l]
            else:
                value = gamma_lap1 / large_step_dt * self.AREA_ave
                Kh_coef_lap1[:, :, :, :]    = value
                Kh_coef_lap1_pl[:, :, :] = value

        elif self.hdiff_type_lap1 == 'E_FOLD_TIME':
            if tau_lap1 > rdtype(0.0):
                self.NUMFILTER_DOhorizontaldiff_lap1 = True

            # tau is e-folding time for 2*dx waves
            if self.dep_hgrid:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kdall):
                        Kh_coef_lap1[:, :, k, l] = (np.sqrt(gmtr.GMTR_area[:, l]) / PI) ** 2 / (tau_lap1 + EPS)

                if adm.ADM_have_pl:
                    for l in range(adm.ADM_lall_pl):
                        for k in range(adm.ADM_kdall):
                            Kh_coef_lap1_pl[:, k, l] = (np.sqrt(gmtr.GMTR_area_pl[:, l]) / PI) ** 2 / (tau_lap1 + EPS)
            else:
                value = (np.sqrt(self.AREA_ave) / PI) ** 2 / (tau_lap1 + EPS)
                Kh_coef_lap1[:, :, :, :]    = value
                Kh_coef_lap1_pl[:, :, :] = value


        # Apply height factor
        fact = np.empty(adm.ADM_kdall, dtype=rdtype)
        self.height_factor(adm.ADM_kdall, grd.GRD_gz, grd.GRD_htop, zlimit_lap1, fact, cnst, rdtype)

        for l in range(adm.ADM_lall):
            for k in range(adm.ADM_kdall):
                Kh_coef_lap1[:, :, k, l] *= fact[k]

        if adm.ADM_have_pl:
            for l in range(adm.ADM_lall_pl):
                for k in range(adm.ADM_kdall):
                    Kh_coef_lap1_pl[:, k, l] *= fact[k]

        # Logging
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)
                print("-----   Horizontal numerical diffusion (1st order laplacian)   -----", file=log_file)

        if self.NUMFILTER_DOhorizontaldiff_lap1:
            if self.debug:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kdall):
                        e_fold_time[:, :, k, l] = (np.sqrt(gmtr.GMTR_area[:, :, l]) / PI) ** 2 / (Kh_coef_lap1[:, :, k, l] + EPS)

                if adm.ADM_have_pl:
                    for l in range(adm.ADM_lall_pl):
                        for k in range(adm.ADM_kdall):
                            e_fold_time_pl[:, k, l] = (np.sqrt(gmtr.GMTR_area_pl[:, l]) / PI) ** 2 / (Kh_coef_lap1_pl[:, k, l] + EPS)

                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print("    z[m]      max coef      min coef  max eft(2DX)  min eft(2DX)", file=log_file)

                for k in range(adm.ADM_kmax, adm.ADM_kmin - 1, -1):  # range not checked
                    eft_max  = gtl.GTL_max_k(e_fold_time, e_fold_time_pl, k)
                    eft_min  = gtl.GTL_min_k(e_fold_time, e_fold_time_pl, k)
                    coef_max = gtl.GTL_max_k(Kh_coef_lap1, Kh_coef_lap1_pl, k)
                    coef_min = gtl.GTL_min_k(Kh_coef_lap1, Kh_coef_lap1_pl, k)
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

        self.divdamp_type
        self.dep_hgrid
        self.smooth_1var
        lap_order = self.lap_order_divdamp
        alpha = self.alpha_d
        tau   = self.tau_d
        alpha_v = self.alpha_dv

        e_fold_time    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall),    dtype=rdtype)
        e_fold_time_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)

        self.divdamp_coef    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall),    dtype=rdtype)
        self.divdamp_coef_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl), dtype=rdtype)


        if self.divdamp_type == 'DIRECT':
            if alpha > 0.0:
                self.NUMFILTER_DOdivdamp = True

            # alpha_d is an absolute value.
            coef = alpha

            self.divdamp_coef[:, :, :, :] = coef
            self.divdamp_coef_pl[:, :, :] = coef

        elif self.divdamp_type == 'NONDIM_COEF':
            if alpha > 0.0:
                self.NUMFILTER_DOdivdamp = True

            small_step_dt = tim.TIME_DTS / rdtype(rcnf.DYN_DIV_NUM)

            # alpha_d is a non-dimensional number.
            # alpha_d * (c_s)^p * dt^{2p-1}
            coef = alpha * (SOUND * SOUND)**lap_order * small_step_dt**(2 * lap_order - 1)

            self.divdamp_coef[:, :, :, :] = coef
            self.divdamp_coef_pl[:, :, :] = coef

        elif self.divdamp_type == 'E_FOLD_TIME':
            if tau > 0.0:
                self.NUMFILTER_DOdivdamp = True

            # tau_d is e-folding time for 2*dx.
            if self.dep_hgrid:
                for l in range(adm.ADM_lall):
                    for k in range(adm.ADM_kdall):
                        self.divdamp_coef[:, :, k, l] = (np.sqrt(gmtr.GMTR_area[:, :, l]) / PI)**(2 * lap_order) / (tau + EPS)

                if adm.ADM_have_pl:
                    for l in range(adm.ADM_lall_pl):
                        for k in range(adm.ADM_kdall):
                            self.divdamp_coef_pl[:, k, l] = (np.sqrt(gmtr.GMTR_area_pl[:, l]) / PI)**(2 * lap_order) / (tau + EPS)
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
                    for k in range(adm.ADM_kdall):
                        e_fold_time[:, :, k, l] = (np.sqrt(gmtr.GMTR_area[:, :, l]) / PI)**(2 * lap_order) / (self.divdamp_coef[:, :, k, l] + EPS)

                e_fold_time_pl[:, :, :] = 0.0

                if adm.ADM_have_pl:
                    for l in range(adm.ADM_lall_pl):
                        for k in range(adm.ADM_kdall):
                            e_fold_time_pl[:, k, l] = (np.sqrt(gmtr.GMTR_area_pl[:, l]) / PI)**(2 * lap_order) / (self.divdamp_coef_pl[:, k, l] + EPS)

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

        if alpha_v > 0.0:
            self.NUMFILTER_DOdivdamp_v = True

        small_step_dt = tim.TIME_dts / float(rcnf.DYN_DIV_NUM)
        self.divdamp_coef_v = -alpha_v * SOUND * SOUND * small_step_dt

        return
    

    def numfilter_smooth_1var(self, s, s_pl, comm, gmtr, oprt, rdtype):

        vtmp     = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall,    1), dtype=rdtype)
        vtmp_pl  = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, 1), dtype=rdtype)
        vtmp2    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d,    adm.ADM_kdall, adm.ADM_lall,    1), dtype=rdtype)
        vtmp2_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kdall, adm.ADM_lall_pl, 1), dtype=rdtype)

        # Constants
        ggamma_h = rdtype(1.0) / 16.0 / 10.0
        itelim   = 80

        gall_1d = adm.ADM_gall_1d
        kall = adm.ADM_kdall

        print("itelim=", itelim)

        for ite in range(itelim):
            
            print(f"ite: {ite}")

            vtmp[:, :, :, :, 0] = s
            if adm.ADM_have_pl:
                vtmp_pl[:, :, :, 0] = s_pl

            comm.COMM_data_transfer(vtmp, vtmp_pl)

            for p in range(2):
                vtmp2[:, :, :, :, :] = rdtype(0.0)
                vtmp2_pl[:, :, :, :] = rdtype(0.0)

                vtmp2[:, :, :, :, 0], vtmp2_pl[:, :, :, 0] = oprt.OPRT_laplacian(
                    vtmp[:, :, :, :, 0], vtmp_pl[:, :, :, 0], 
                    oprt.OPRT_coef_lap[:, :, :, :], oprt.OPRT_coef_lap_pl[:,:],  rdtype
                )

                comm.COMM_data_transfer(vtmp, vtmp_pl)

            for i in range(gall_1d):
                for j in range(gall_1d):
                    for k in range(kall):
                        for l in range(adm.ADM_lall):            
                            s[i, j, k, l] -= ggamma_h * gmtr.GMTR_area[i, j, l] ** 2 * vtmp[i, j, k, l, 0]

            if adm.ADM_have_pl:
                for g in range(adm.ADM_gall_pl):
                    for k in range(adm.ADM_kdall):
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
