import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf


class Bndc:
    
    _instance = None

    is_top_tem   = False
    is_top_epl   = False
    is_btm_tem   = False
    is_btm_epl   = False
    is_top_rigid = False
    is_top_free  = False
    is_btm_rigid = False
    is_btm_free  = False

    def __init__(self):
        pass

    def BNDCND_setup(self, fname_in, rdtype):

        # Set default boundary types
        BND_TYPE_T_TOP    = 'TEM'
        BND_TYPE_T_BOTTOM = 'TEM'
        BND_TYPE_M_TOP    = 'FREE'
        BND_TYPE_M_BOTTOM = 'RIGID'

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[bndcnd]/Category[nhm share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'bndcndparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** bndcndparam not found in toml file! Use default.", file=log_file)
                #prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['bndcndparam']
            #self.GRD_grid_type = cnfs['GRD_grid_type']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)


        if BND_TYPE_T_BOTTOM == 'TEM':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (temperature, bottom) : equal to lowermost atmosphere', file=log_file)
            self.is_btm_tem = True

        elif BND_TYPE_T_BOTTOM == 'EPL':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (temperature, bottom) : lagrange extrapolation', file=log_file)
            self.is_btm_epl = True

        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('xxx Invalid BND_TYPE_T_BOTTOM. STOP.', file=log_file)
            prc.prc_mpistop(std.io_l, std.fname_log)


        if BND_TYPE_M_TOP == 'RIGID':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (momentum,    top   ) : rigid', file=log_file)
            self.is_top_rigid = True

        elif BND_TYPE_M_TOP == 'FREE':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (momentum,    top   ) : free', file=log_file)
            self.is_top_free = True

        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('xxx Invalid BND_TYPE_M_TOP. STOP.', file=log_file)
            prc.prc_mpistop(std.io_l, std.fname_log)


        if BND_TYPE_M_BOTTOM == 'RIGID':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (momentum,    bottom) : rigid', file=log_file)
            self.is_btm_rigid = True

        elif BND_TYPE_M_BOTTOM == 'FREE':
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('*** Boundary setting type (momentum,    bottom) : free', file=log_file)
            self.is_btm_free = True

        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print('xxx Invalid BND_TYPE_M_BOTTOM. STOP.', file=log_file)
            prc.prc_mpistop(std.io_l, std.fname_log)

        return 
        
    def BNDCND_all(
        self,
        idim, 
        jdim,      # for ij arrays (poles), add a dummy dimension upon calling 
        kdim, 
        ldim, 
        rho,       # (idim, jdim, kdim, ldim)  density
        vx,        # (idim, jdim, kdim, ldim)  horizontal wind (x)
        vy,        # (idim, jdim, kdim, ldim)  horizontal wind (y) 
        vz,        # (idim, jdim, kdim, ldim)  horizontal wind (z)
        w,         # (idim, jdim, kdim, ldim)  vertical wind 
        ein,       # (idim, jdim, kdim, ldim)  internal energy
        tem,       # (idim, jdim, kdim, ldim)  temperature
        pre,       # (idim, jdim, kdim, ldim)  pressure
        rhog,
        rhogvx,
        rhogvy,
        rhogvz,
        rhogw,
        rhoge,
        gsqrtgam2,  
        phi,       # (idim, jdim, kdim, ldim)  geopotential
        c2wfact,    
        c2wfact_Gz,
        cnst,
        rdtype,
    ):

        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        kmaxp1 = kmax + 1
        kminm1 = kmin - 1
        CVdry = cnst.CONST_CVdry


        #--- Thermodynamical variables ( rho, ein, tem, pre, rhog, rhoge ), q = 0 at boundary

        self.BNDCND_thermo(
            tem, rho, pre, phi, 
            cnst, rdtype
        )

        rhog[:, :, kmaxp1, :] = rho[:, :, kmaxp1, :] * gsqrtgam2[:, :, kmaxp1, :]
        rhog[:, :, kminm1, :] = rho[:, :, kminm1, :] * gsqrtgam2[:, :, kminm1, :]
        ein[:, :, kmaxp1, :] = CVdry * tem[:, :, kmaxp1, :]
        ein[:, :, kminm1, :] = CVdry * tem[:, :, kminm1, :]
        rhoge[:, :, kmaxp1, :] = rhog[:, :, kmaxp1, :] * ein[:, :, kmaxp1, :]
        rhoge[:, :, kminm1, :] = rhog[:, :, kminm1, :] * ein[:, :, kminm1, :]

        #--- Momentum ( rhogvx, rhogvy, rhogvz, vx, vy, vz )

            # call BNDCND_rhovxvyvz( ijdim,         & ! [IN]
            #                         kdim,          & ! [IN]
            #                         ldim,          & ! [IN]
            #                         rhog  (:,:,:), & ! [IN]
            #                         rhogvx(:,:,:), & ! [INOUT]
            #                         rhogvy(:,:,:), & ! [INOUT]
            #                         rhogvz(:,:,:)  ) ! [INOUT]


        vx[:, :, kmaxp1, :] = rhogvx[:, :, kmaxp1, :] / rhog[:, :, kmaxp1, :]
        vx[:, :, kminm1, :] = rhogvx[:, :, kminm1, :] / rhog[:, :, kminm1, :]
        vy[:, :, kmaxp1, :] = rhogvy[:, :, kmaxp1, :] / rhog[:, :, kmaxp1, :]
        vy[:, :, kminm1, :] = rhogvy[:, :, kminm1, :] / rhog[:, :, kminm1, :]
        vz[:, :, kmaxp1, :] = rhogvz[:, :, kmaxp1, :] / rhog[:, :, kmaxp1, :]
        vz[:, :, kminm1, :] = rhogvz[:, :, kminm1, :] / rhog[:, :, kminm1, :]

        #--- Momentum ( rhogw, w )
        #for l in range(ldim):
            # call BNDCND_rhow( ijdim,               & ! [IN]
            #                     rhogvx    (:,:,l),   & ! [IN]
            #                     rhogvy    (:,:,l),   & ! [IN]
            #                     rhogvz    (:,:,l),   & ! [IN]
            #                     rhogw     (:,:,l),   & ! [INOUT]
            #                     c2wfact_Gz(:,:,:,l)  ) ! [IN]
        # enddo
        #    pass


        w[:, :, kmaxp1, :] = rhogw[:, :, kmaxp1, :] / (
            c2wfact[:, :, kmaxp1, 0, :] * rhog[:, :, kmaxp1, :] +
            c2wfact[:, :, kmaxp1, 1, :] * rhog[:, :, kmax, :]
        )

        w[:, :, kmin, :] = rhogw[:, :, kmin, :] / (
            c2wfact[:, :, kmin, 0, :] * rhog[:, :, kmin,   :] +
            c2wfact[:, :, kmin, 1, :] * rhog[:, :, kminm1, :]
        )

        w[:, :, kminm1, :] = 0.0


        return
    
    def BNDCND_thermo(
        self,
        tem, rho, pre, phi, 
        cnst, rdtype
    ):

        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        kminm1   = kmin - 1
        kminp1   = kmin + 1
        kminp2   = kmin + 2
        kmaxm1   = kmax - 1
        kmaxm2   = kmax - 2
        kmaxp1   = kmax + 1
        GRAV = cnst.CONST_GRAV
        Rdry = cnst.CONST_Rdry


        # Vectorized Lagrange interpolation
        def lag_intpl_vec(z, z1, p1, z2, p2, z3, p3):
            return (
                ((z - z2) * (z - z3)) / ((z1 - z2) * (z1 - z3)) * p1 +
                ((z - z1) * (z - z3)) / ((z2 - z1) * (z2 - z3)) * p2 +
                ((z - z1) * (z - z2)) / ((z3 - z1) * (z3 - z2)) * p3
            )

        # -----------------------
        # Top temperature boundary
        # -----------------------
        if self.is_top_tem:
            tem[:, :, kmaxp1, :] = tem[:, :, kmax, :]

        elif self.is_top_epl:
            z  = phi[:, :, kmaxp1, :] / GRAV
            z1 = phi[:, :, kmax,   :] / GRAV
            z2 = phi[:, :, kmaxm1, :] / GRAV
            z3 = phi[:, :, kmaxm2, :] / GRAV

            tem[:, :, kmaxp1, :] = lag_intpl_vec(
                z,
                z1, tem[:, :, kmax,   :],
                z2, tem[:, :, kmaxm1, :],
                z3, tem[:, :, kmaxm2, :]
            )

        # -----------------------
        # Bottom temperature boundary
        # -----------------------
        if self.is_btm_tem:
            tem[:, :, kminm1, :] = tem[:, :, kmin, :]

        elif self.is_btm_epl:
            z1 = phi[:, :, kminp2, :] / GRAV
            z2 = phi[:, :, kminp1, :] / GRAV
            z3 = phi[:, :, kmin,   :] / GRAV
            z  = phi[:, :, kminm1, :] / GRAV

            tem[:, :, kminm1, :] = lag_intpl_vec(
                z,
                z1, tem[:, :, kminp2, :],
                z2, tem[:, :, kminp1, :],
                z3, tem[:, :, kmin,   :]
            )

        # -----------------------
        # Pressure boundary (hydrostatic)
        # -----------------------
        pre[:, :, kmaxp1, :] = pre[:, :, kmaxm1, :] - rho[:, :, kmax, :] * (
            phi[:, :, kmaxp1, :] - phi[:, :, kmaxm1, :]
        )
        pre[:, :, kminm1, :] = pre[:, :, kminp1, :] - rho[:, :, kmin, :] * (
            phi[:, :, kminm1, :] - phi[:, :, kminp1, :]
        )

        # -----------------------
        # Density boundary (equation of state)
        # -----------------------
        rho[:, :, kmaxp1, :] = pre[:, :, kmaxp1, :] / (Rdry * tem[:, :, kmaxp1, :])
        rho[:, :, kminm1, :] = pre[:, :, kminm1, :] / (Rdry * tem[:, :, kminm1, :])

        return
    
    # def BNDCND_rhovxvyvz(
    #     self,
    #     idim, jdim, kdim, ldim,
    #     rhog, rhogvx, rhogvy, rhogvz
    # ):

    #     # Vectorized Lagrange interpolation
    #     def lag_intpl_vec(z, z1, p1, z2, p2, z3, p3):
    #         return (
    #             ((z - z2) * (z - z3)) / ((z1 - z2) * (z1 - z3)) * p1 +
    #             ((z - z1) * (z - z3)) / ((z2 - z1) * (z2 - z3)) * p2 +
    #             ((z - z1) * (z - z2)) / ((z3 - z1) * (z3 - z2)) * p3
    #         )

    #     # -----------------------
    #     # Top boundary
    #     # -----------------------
    #     if self.is_top_rigid:
    #         pass

    #     elif self.is_top_free:
    #         for l in range(ldim):
    #             rhogvx[:, :, kmaxp1, l] = rhogvx[:, :, kmaxm1, l]
    #             rhogvy[:, :, kmaxp1, l] = rhogvy[:, :, kmaxm1, l]
    #             rhogvz[:, :, kmaxp1, l] = rhogvz[:, :, kmaxm1, l]

    #     # -----------------------
    #     # Bottom boundary
    #     # -----------------------
    #     if self.is_btm_rigid:
    #         pass

    #     elif self.is_btm_free:
    #         for l in range(ldim):
    #             rhogvx[:, :, kminm1, l] = rhogvx[:, :, kminp1, l]
    #             rhogvy[:, :, kminm1, l] = rhogvy[:, :, kminp1, l]
    #             rhogvz[:, :, kminm1, l] = rhogvz[:, :, kminp1, l]