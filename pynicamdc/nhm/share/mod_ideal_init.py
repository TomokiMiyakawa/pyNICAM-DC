import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_grd import grd
#from mod_prof import prf


class Idi:
    
    _instance = None
    
    DCTEST_type = ''
    DCTEST_case = ''

    # --- Physical Parameters Configurations ---
    Kap = None  # Temporal value (uninitialized)
    d2r = None  # Degree to Radian conversion
    r2d = None  # Radian to Degree conversion
    zero = 0.0  # Zero (float)

    # --- Jablonowski Configuration ---
    clat = 40.0      # Perturbation center: latitude [deg]
    clon = 20.0      # Perturbation center: longitude [deg]
    etaT = 0.2       # Threshold of vertical profile
    eta0 = 0.252     # Threshold of vertical profile
    t0 = 288.0       # Temperature [K]
    delT = 4.8e+5    # Temperature perturbation [K]
    ganma = 0.005    # Temperature lapse rate [K m^-1]
    u0 = 35.0        # Wind speed [m/s]
    uP = 1.0         # Wind perturbation [m/s]
    p0 = 1.0e+5      # Pressure [Pa]

    # --- Constants ---
    message = False  # Boolean flag
    itrmax = 100     # Maximum number of iterations


    def __init__(self):
        pass

    def dycore_input(self, fname_in, cnst, rcnf, grd, idi, rdtype):

        # Equivalent to `real(RP), intent(out) :: DIAG_var(ADM_gall,ADM_kall,ADM_lall,6+TRC_VMAX)`
        DIAG_var = np.zeros((adm.ADM_gall, adm.ADM_kdall, adm.ADM_lall, 6 + rcnf.TRC_vmax), dtype=rdtype)

        # Equivalent to `character(len=H_SHORT) :: init_type = ''`
        init_type = ""  
        test_case = ""  

        # Equivalent to `real(RP) :: eps_geo2prs = 1.E-2_RP`
        eps_geo2prs = 1.0e-2  

        # Equivalent to `logical` variables in Fortran
        nicamcore = True
        chemtracer = False
        prs_rebuild = False

        self.Kap = cnst.CONST_Rdry / cnst.CONST_CPdry
        self.d2r = cnst.CONST_PI/180.0
        self.r2d = 180.0/cnst.CONST_PI


        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[dycoretest]/Category[nhm share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'dycoretestparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** dycoretestparam not found in toml file! Use default.", file=log_file)
                #prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['dycoretestparam']
            init_type = cnfs['init_type']
            test_case = cnfs['test_case']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)


        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(f"*** test case: {test_case.strip()}", file=log_file)

        match init_type:
            # case "DCMIP2012-11" | "DCMIP2012-12" | "DCMIP2012-13" | "DCMIP2012-200" | "DCMIP2012-21" | "DCMIP2012-22":
            #     if IO_L:
            #         print(f"*** test case: {test_case.strip()}")
            #     IDEAL_init_DCMIP2012(adm.ADM_gall, adm.ADM_kdall, adm.ADM_lall, init_type, rcnf.DIAG_var)

            case "Heldsuarez":
                print("Heldsuarez not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                #hs_init(adm.ADM_gall, adm.ADM_kdall, adm.ADM_lall, rcnf.DIAG_var)

            case "Jablonowski":
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:                        
                        print(f"*** test case   : {test_case.strip()}", file=log_file)
                        print(f"*** eps_geo2prs = {eps_geo2prs}", file=log_file)
                        print(f"*** nicamcore   = {nicamcore}", file=log_file)
                DIAG_var = self.jbw_init(adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kdall, adm.ADM_lall, test_case, eps_geo2prs, nicamcore, cnst, rcnf, grd, rdtype)

            case "Jablonowski-Moist":
                print("Jablonowski-Moist not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                #if IO_L:
                #    print(f"*** test case   : {test_case.strip()}")
                #    print(f"*** nicamcore   = {nicamcore}")
                #    print(f"*** chemtracer  = {chemtracer}")
                #jbw_moist_init(adm.ADM_gall, adm.ADM_kdall, adm.ADM_lall, test_case, chemtracer, rcnf.DIAG_var)

            case "Supercell":
                print("Supercell not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                # if IO_L:
                #     print(f"*** test case   : {test_case.strip()}")
                #     print(f"*** nicamcore   = {nicamcore}")
                # sc_init(adm.ADM_gall, adm.ADM_kdall, adm.ADM_lall, test_case, prs_rebuild, rcnf.DIAG_var)

            case "Tropical-Cyclone":
                print("Tropical-Cyclone not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                # if IO_L:
                #     print(f"*** nicamcore   = {nicamcore}")
                # tc_init(adm.ADM_gall, adm.ADM_kdall, adm.ADM_lall, prs_rebuild, rcnf.DIAG_var)

            case "Traceradvection":
                print("Traceradvection not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                # if IO_L:
                #     print(f"*** test case: {test_case.strip()}")
                # tracer_init(adm.ADM_gall, adm.ADM_kdall, adm.ADM_lall, test_case, rcnf.DIAG_var)

            case "Mountainwave":
                print("Mountainwave not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                # if IO_L:
                #     print(f"*** test case: {test_case.strip()}")
                # mountwave_init(adm.ADM_gall, adm.ADM_kdall, adm.ADM_lall, test_case, rcnf.DIAG_var)

            case "Gravitywave":
                print("Gravitywave not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                #gravwave_init(adm.ADM_gall, adm.ADM_kdall, adm.ADM_lall, rcnf.DIAG_var)

            case "Tomita2004":
                print("Tomita2004 not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                #tomita_init(adm.ADM_gall, adm.ADM_kdall, adm.ADM_lall, rcnf.DIAG_var)

            case _:
                print("xxx [dycore_input] Invalid init_type. STOP.")
                raise SystemExit("PRC_MPIstop called")

        return


    def jbw_init(self, idim, jdim, kdim, lall, test_case, eps_geo2prs, nicamcore, cnst, rcnf, grd, rdtype):

        DIAG_var = np.zeros((idim, jdim, kdim, lall, 6 + rcnf.TRC_vmax), dtype=rdtype)

        eta_limit = True
        psgm = False
        logout = True

        lat = 0.0      # Latitude on Icosahedral grid
        lon = 0.0      # Longitude on Icosahedral grid
        ps = 0.0       # Surface pressure

        # --- 1D NumPy Arrays for ICO-grid field ---
        eta = np.zeros((kdim, 2), dtype=rdtype)  # Eta values
        geo = np.zeros(kdim, dtype=rdtype)       # Geopotential

        prs = np.zeros(kdim, dtype=rdtype)       # Pressure
        tmp = np.zeros(kdim, dtype=rdtype)       # Temperature

        wix = np.zeros(kdim, dtype=rdtype)       # Zonal wind component
        wiy = np.zeros(kdim, dtype=rdtype)       # Meridional wind component

        # --- Local Variables ---
        z_local = np.zeros(kdim, dtype=rdtype)   # Local height
        vx_local = np.zeros(kdim, dtype=rdtype)  # Local zonal wind
        vy_local = np.zeros(kdim, dtype=rdtype)  # Local meridional wind
        vz_local = np.zeros(kdim, dtype=rdtype)  # Local vertical wind

        # --- Logical (Boolean) Variables ---
        signal = False     # If True, continue iteration
        pertb = False      # If True, with perturbation
        psgm = False       # If True, PS Gradient Method
        eta_limit = False  # If True, value of eta is limited up to 1.0
        logout = False     # Log output switch for Pressure Convertion




        test_case_trimmed = test_case.strip()

        match test_case_trimmed:
            case "1" | "4-1":  # With perturbation
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print("Jablonowski Initialize - case 1: with perturbation (no rebalance)", file=log_file)
                pertb = True

            case "2" | "4-2":  # Without perturbation
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print("Jablonowski Initialize - case 2: without perturbation (no rebalance)", file=log_file)
                pertb = False

            case "3":  # With perturbation (PS Distribution Method)
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print("Jablonowski Initialize - PS Distribution Method: with perturbation", file=log_file)
                        print("### DO NOT INPUT ANY TOPOGRAPHY ###", file=log_file)
                pertb = True
                psgm = True
                eta_limit = False

            case "4":  # Without perturbation (PS Distribution Method)
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print("Jablonowski Initialize - PS Distribution Method: without perturbation", file=log_file)
                        print("### DO NOT INPUT ANY TOPOGRAPHY ###", file=log_file)
                pertb = False
                psgm = True
                eta_limit = False

            case _:  # Default case (unknown test_case)
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print(f"Unknown test_case: '{test_case_trimmed}' specified.", file=log_file)
                        print("Force changed to case 1 (with perturbation)", file=log_file)
                pertb = True

        # Additional logging
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(f" | eps for geo2prs: {eps_geo2prs}", file=log_file)
                print(f" | nicamcore switch for geo2prs: {nicamcore}", file=log_file)



        for l in range(lall):
            with open(std.fname_log, 'a') as log_file:
                print("*** Processing layer ***", l, file=log_file)
            #prc.prc_mpistop(std.io_l, std.fname_log)
            #import sys
            #sys.exit()

            for i in range(idim):
                with open(std.fname_log, 'a') as log_file:
                    print("*** Processing i ***", i, file=log_file)
                for j in range(jdim):
                    z_local[adm.ADM_kmin] = grd.GRD_vz[i, j, 1, l, grd.GRD_ZH]    # 0th layer
                    #               0inp 1inf    40inp 41inf       for 40layers  
                    for k in range(adm.ADM_kmin + 1, adm.ADM_kmax + 2):  # loop 1 to 41 layers
                                                            # index 
                        z_local[k] = grd.GRD_vz[i, j, k, l, grd.GRD_Z]

                        #lat = rdtype(grd.GRD_LAT[i, j, l])
                        #lon = rdtype(grd.GRD_LON[i, j, l])
                        lat = grd.GRD_LAT[i, j, l]
                        lon = grd.GRD_LON[i, j, l]

                        signal = True

                        # Iteration process
                        for itr in range(self.itrmax):

#                            print(f"ITERATION: {itr}")

                            if itr == 0:
                                eta[:, :] = 1.0e-7  # Jablonowski recommended initial value
                            else:
                                self.eta_vert_coord_NW(kdim, itr, z_local, tmp, geo, eta_limit, eta, signal, cnst, rdtype)

                            self.steady_state(kdim, lat, eta, wix, wiy, tmp, geo, cnst, rdtype)

                            if not signal:
                                break  # Exit iteration loop

                        # Check for convergence failure
                        if itr > self.itrmax:
                            print(f"ETA ITERATION ERROR: NOT CONVERGED at i={i}, j={j} l={l}")
                            prc.prc_mpistop(std.io_l, std.fname_log)
                            raise SystemExit("PRC_MPIstop called")

                        # Pressure estimation
                        if psgm:
                            ps=self.ps_estimation(kdim, lat, eta[:, 0], tmp, geo, wix, nicamcore, cnst, rdtype)
                            self.geo2prs(kdim, ps, lat, tmp, geo, wix, prs, eps_geo2prs, nicamcore, logout, cnst, rdtype)
                        else:
                            self.geo2prs(kdim, self.p0, lat, tmp, geo, wix, prs, eps_geo2prs, nicamcore, logout, cnst, rdtype)  
                        logout = False

                        # Convert velocity components
                        self.conv_vxvyvz(kdim, lat, lon, wix, wiy, vx_local, vy_local, vz_local)

                        # Store results in DIAG_var
                        for k in range(kdim):
                            DIAG_var[i, j, k, l, 0] = prs[k]
                            DIAG_var[i, j, k, l, 1] = tmp[k]
                            DIAG_var[i, j, k, l, 2] = vx_local[k]
                            DIAG_var[i, j, k, l, 3] = vy_local[k]
                            DIAG_var[i, j, k, l, 4] = vz_local[k]



        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(" |            Vertical Coordinate used in JBW initialization              |")
                print(" |------------------------------------------------------------------------|")

                for k in range(kdim):
                    print(f"   (k={k:3}) HGT: {z_local[k]:8.2f} [m]  "
                        f"PRS: {prs[k]:9.2f} [Pa]  "
                        f"GH: {geo[k] / cnst.CONST_GRAV:8.2f} [m]  "
                        f"ETA: {eta[k, 0]:9.5f}", file=log_file)

                print(" |------------------------------------------------------------------------|")

        return DIAG_var

    def eta_vert_coord_NW(self, kdim, itr, z, tmp, geo, eta_limit, eta, signal, cnst, rdtype):

        """
        Computes the eta level vertical coordinate using iteration.

        Parameters:
        kdim (int)       : Number of z-dimension levels
        itr (int)        : Iteration number
        z (np.ndarray)   : z-height vertical coordinate (1D array of size kdim)
        tmp (np.ndarray) : Guessed temperature (1D array of size kdim)
        geo (np.ndarray) : Guessed geopotential (1D array of size kdim)
        eta_limit (bool) : Eta limitation flag
        eta (np.ndarray) : Eta level vertical coordinate (2D array of size (kdim,2))
        signal (bool)    : Iteration signal (modified in-place)
        """

        diff = np.zeros(kdim, dtype=rdtype)
        F = np.zeros(kdim, dtype=rdtype)
        Feta = np.zeros(kdim, dtype=rdtype)

        criteria = max(cnst.CONST_EPS * 10.0, 1.0e-14)  # Equivalent to max(EPS * 10.0_RP, 1.E-14_RP)

        for k in range(kdim):
            F[k] = -cnst.CONST_GRAV * z[k] + geo[k]
            Feta[k] = -1.0 * (cnst.CONST_Rdry / eta[k, 0]) * tmp[k]  # Using eta[:, 0] for Fortran's eta(k,1)

            eta[k, 1] = eta[k, 0] - (F[k] / Feta[k])

            if eta_limit:  # [add] for PSDM (2013/12/20 R.Yoshida)
                eta[k, 1] = min(eta[k, 1], 1.0)  # Not allow eta > 1.0

            eta[k, 1] = max(eta[k, 1], cnst.CONST_EPS)  # Ensure eta ≥ EPS

            diff[k] = abs(eta[k, 1] - eta[k, 0])

        # Update eta[:, 0] with the new values from eta[:, 1]
        eta[:, 0] = eta[:, 1]

        # Logging information
        if self.message:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f" | Eta  {itr}: -- MAX: {np.max(diff):20.10e} MIN: {np.min(diff):20.10e}", file=log_file)
                    print(f" | Diff {itr}: -- MAX: {np.max(diff):20.10e} MIN: {np.min(diff):20.10e}", file=log_file)    

        # Convergence check
        if np.max(diff) < criteria:
            signal = False
        else:
            if self.message and std.io_l: 
                with open(std.fname_log, 'a') as log_file:
                    print(f"| Iterating : {itr} criteria = {criteria:.10e}", file=log_file) 

        return
    
    def steady_state(self, kdim, lat, eta, wix, wiy, tmp, geo, cnst, rdtype):

        # kdim, &  !--- IN : # of z dimension
        # lat,  &  !--- IN : latitude information
        # eta,  &  !--- IN : eta level vertical coordinate
        # wix,  &  !--- INOUT : zonal wind component
        # wiy,  &  !--- INOUT : meridional wind component
        # tmp,  &  !--- INOUT : mean temperature
        # geo   )  !--- INOUT : mean geopotential height

        # ---------- Horizontal Mean ----------
        work1 = cnst.CONST_PI / 2.0
        work2 = cnst.CONST_Rdry * self.ganma / cnst.CONST_GRAV

        for k in range(kdim):
            eta_v = (eta[k, 0] - self.eta0) * work1
            wix[k] = self.u0 * (np.cos(eta_v)) ** 1.5 * (np.sin(2.0 * lat)) ** 2.0

            if eta[k, 0] >= self.etaT:
                tmp[k] = self.t0 * eta[k, 0] ** work2
                geo[k] = self.t0 * cnst.CONST_GRAV / self.ganma * (1.0 - eta[k, 0] ** work2)

            elif eta[k, 0] < self.etaT:
                tmp[k] = self.t0 * eta[k, 0] ** work2 + self.delT * (self.etaT - eta[k, 0]) ** 5.0

                geo[k] = (self.t0 * cnst.CONST_GRAV / self.ganma * (1.0 - eta[k, 0] ** work2) - cnst.CONST_Rdry * self.delT *
                    ((np.log(eta[k, 0] / self.etaT) + 137.0 / 60.0) * self.etaT ** 5.0
                    - 5.0 * self.etaT ** 4.0 * eta[k, 0]
                    + 5.0 * self.etaT ** 3.0 * (eta[k, 0] ** 2.0)
                    - (10.0 / 3.0) * self.etaT ** 2.0 * (eta[k, 0] ** 3.0)
                    + (5.0 / 4.0) * self.etaT * (eta[k, 0] ** 4.0)
                    - (1.0 / 5.0) * (eta[k, 0] ** 5.0))
                )

            else:
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print("|-- ETA BOUNDARY ERROR: [steady state calc.]", file=log_file)
                        print(f"|-- ({k:3d})  eta: {eta[k, 0]:10.4f}", file=log_file)
                    prc.prc_mpi_stop(std.io_l, std.fname_log)   
                    raise SystemExit("STOP")

        # ---------- Meridional Distribution for Temperature and Geopotential ----------
        work1 = cnst.CONST_PI / 2.0
        work2 = 3.0 / 4.0 * (cnst.CONST_PI * self.u0 / cnst.CONST_Rdry)

        for k in range(kdim):
            eta_v = (eta[k, 0] - self.eta0) * work1
            tmp[k] += (work2 * eta[k, 0] * np.sin(eta_v) * (np.cos(eta_v)) ** 0.5 *
                ((-2.0 * (np.sin(lat)) ** 6.0 * (np.cos(lat) ** 2.0 + 1.0 / 3.0) + 10.0 / 63.0)
                * 2.0 * self.u0 * (np.cos(eta_v)) ** 1.5
                + (8.0 / 5.0 * (np.cos(lat)) ** 3.0 * ((np.sin(lat)) ** 2.0 + 2.0 / 3.0) - cnst.CONST_PI / 4.0)
                * cnst.CONST_RADIUS * cnst.CONST_OHM)
            )

            geo[k] += (self.u0 * (np.cos(eta_v)) ** 1.5 *
                ((-2.0 * (np.sin(lat)) ** 6.0 * (np.cos(lat) ** 2.0 + 1.0 / 3.0) + 10.0 / 63.0)
                * self.u0 * (np.cos(eta_v)) ** 1.5
                + (8.0 / 5.0 * (np.cos(lat)) ** 3.0 * ((np.sin(lat)) ** 2.0 + 2.0 / 3.0) - cnst.CONST_PI / 4.0)
                * cnst.CONST_RADIUS * cnst.CONST_OHM)
            )

        wiy[:] = 0.0  # Reset wiy to 0.0

        return
    
    #def ps_estimation(self, kdim, lat, eta, tmp, geo, wix, ps, nicamcore, cnst, rdtype):

    def ps_estimation(self, kdim, lat, eta, tmp, geo, wix, nicamcore, cnst, rdtype):
        """
        Estimates surface pressure (ps) using topography 

        Parameters:
        kdim (int)          : Number of vertical levels (z-dimension)
        lat (float)         : Latitude information
        eta (np.ndarray)    : Eta coordinate (1D array of size kdim)
        tmp (np.ndarray)    : Temperature (1D array of size kdim)
        geo (np.ndarray)    : Geopotential height at full height (1D array of size kdim)
        wix (np.ndarray)    : Zonal wind speed (1D array of size kdim)
        nicamcore (bool)    : Nicamcore switch

        Returns:
        ps (float)          : Estimated surface pressure
        """

        # Constants
        lat0 = 0.691590985442682  
        eta1 = 1.0
        pi_half = cnst.CONST_PI * 0.5

        # Eta-related calculation
        eta_v = (eta1 - self.eta0) * pi_half

        # Temperature at bottom of eta-grid
        tmp0 = (
            self.t0
            + (3.0 / 4.0 * (np.pi * self.u0 / cnst.CONST_Rdry)) * eta1 * np.sin(eta_v) * (np.cos(eta_v)) ** 0.5
            * ((-2.0 * (np.sin(lat0)) ** 6.0 * (np.cos(lat0) ** 2.0 + 1.0 / 3.0) + 10.0 / 63.0)
                * 2.0 * self.u0 * (np.cos(eta_v)) ** 1.5
                + (8.0 / 5.0 * (np.cos(lat0)) ** 3.0 * ((np.sin(lat0)) ** 2.0 + 2.0 / 3.0) - cnst.CONST_PI / 4.0)
                * cnst.CONST_RADIUS * cnst.CONST_OHM)
        )
        tmp1 = tmp[0]  # Equivalent to tmp(1) in Fortran

        # Wind speed at bottom of eta-grid
        ux1 = (self.u0 * np.cos(eta_v) ** 1.5) * (np.sin(2.0 * lat0)) ** 2.0
        ux2 = wix[0]  # Equivalent to wix(1) in Fortran

        # Topography calculation
        cs32ev = (np.cos((1.0 - 0.252) * pi_half)) ** 1.5
        f1 = 10.0 / 63.0 - 2.0 * np.sin(lat) ** 6 * (np.cos(lat) ** 2 + 1.0 / 3.0)
        f2 = 1.6 * np.cos(lat) ** 3 * (np.sin(lat) ** 2 + 2.0 / 3.0) - 0.25 * cnst.CONST_PI
        hgt1 = -1.0 * self.u0 * cs32ev * (f1 * self.u0 * cs32ev + f2 * cnst.CONST_RADIUS * cnst.CONST_OHM) / cnst.CONST_GRAV
        hgt0 = 0.0

        # Pressure estimation
        dz = hgt1 - hgt0
        if nicamcore:
            uave = (ux1 + ux2) * 0.5
            f_cf = 2.0 * cnst.CONST_OHM * uave * np.cos(lat) + (uave ** 2.0) / cnst.CONST_RADIUS
        else:
            f_cf = 0.0

        ps = self.p0 * (1.0 + dz * (f_cf - cnst.CONST_GRAV) / (2.0 * cnst.CONST_Rdry * tmp0)) / (1.0 - dz * (f_cf - cnst.CONST_GRAV) / (2.0 * cnst.CONST_Rdry * tmp1))

        return ps
    

    def geo2prs(self, kdim, ps, lat, tmp, geo, wix, prs, eps_geo2prs, nicamcore, logout, cnst, rdtype):
        """
        Converts geopotential height to pressure.

        Parameters:
        kdim (int)          : Number of vertical levels (z-dimension)
        ps (float)          : Surface pressure
        lat (float)         : Latitude
        tmp (np.ndarray)    : Temperature (1D array of size kdim)
        geo (np.ndarray)    : Geopotential height at full height (1D array of size kdim)
        wix (np.ndarray)    : Zonal wind (1D array of size kdim)
        prs (np.ndarray)    : Pressure (1D array of size kdim) [modified in-place]
        eps_geo2prs (float) : Convergence threshold for pressure iteration
        nicamcore (bool)    : Nicamcore switch
        logout (bool)       : Log output switch
        """


        limit = 400  # Iteration limit
        pp = np.zeros(kdim, dtype=rdtype)  # Temporary pressure array
        iteration = False  # Default no iteration
        do_iter = True

        # Initialize surface pressure
        pp[0] = ps

        # First guess (upward: trapezoidal method)
        for k in range(1, kdim):
            dz = (geo[k] - geo[k - 1]) / cnst.CONST_GRAV
            if nicamcore:
                uave = (wix[k] + wix[k - 1]) * 0.5
                f_cf = 2.0 * cnst.CONST_OHM * uave * np.cos(lat) + (uave ** 2.0) / cnst.CONST_RADIUS
            else:
                f_cf = 0.0

            pp[k] = pp[k - 1] * (1.0 + dz * (f_cf - cnst.CONST_GRAV) / (2.0 * cnst.CONST_Rdry * tmp[k - 1])) / \
                            (1.0 - dz * (f_cf - cnst.CONST_GRAV) / (2.0 * cnst.CONST_Rdry * tmp[k]))

        prs[:] = pp[:]  # Copy values to prs

        # Iteration (Simpson's method)
        if iteration:
            for i in range(limit):
                prs[0] = ps  # Reset surface pressure

                # Upward correction
                for k in range(2, kdim):
                    pp[k] = self.simpson(
                        prs[k], prs[k - 1], prs[k - 2],
                        tmp[k], tmp[k - 1], tmp[k - 2],
                        wix[k], wix[k - 1], wix[k - 2],
                        geo[k], geo[k - 2], lat,
                        False, nicamcore, cnst, rdtype
                    )

                prs[:] = pp[:]  # Copy results to prs

                # Downward correction
                for k in range(kdim - 3, -1, -1):  # Reverse loop in Python
                    pp[k] = self.simpson(
                        prs[k + 2], prs[k + 1], prs[k],
                        tmp[k + 2], tmp[k + 1], tmp[k],
                        wix[k + 2], wix[k + 1], wix[k],
                        geo[k + 2], geo[k], lat,
                        True, nicamcore, cnst, rdtype
                    )

                prs[:] = pp[:]  # Copy results to prs

                diff = pp[0] - ps

                if abs(diff) < eps_geo2prs:
                    do_iter = False
                    break  # Exit iteration loop

        else:
            do_iter = False  # Skip iteration

        # Handle iteration failure
        if do_iter:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(f"ETA ITERATION ERROR: NOT CONVERGED at GEO2PRS, diff = {diff:.10e}", file=log_file)
            raise SystemExit("STOP")


        # Finalize pressure values
        prs[0] = ps
        pp[0] = ps

        # Upward correction using Simpson's method
        for k in range(2, kdim):  # Fortran's k=3:kdim is Python's range(2, kdim)
            pp[k] = self.simpson(
                prs[k], prs[k - 1], prs[k - 2],
                tmp[k], tmp[k - 1], tmp[k - 2],
                wix[k], wix[k - 1], wix[k - 2],
                geo[k], geo[k - 2], lat,
                False, nicamcore, cnst, rdtype
            )
        prs[:] = pp[:]  # Copy values to prs

        # Logging outputs
        if logout:
            if iteration:
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print(f" | diff (guess - ps) : {diff:.2f} [Pa]  --  itr times: {(i - 1)}", file=log_file)
            else:
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print(" | no iteration in geo2prs", file=log_file)

        if self.message:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("\n | ----- Pressure (Final Guess) -----", file=log_file)
                    for k in range(kdim):
                        print(f" | K({k+1:3d}) -- {prs[k]:20.13f}", file=log_file)  # Fortran is 1-based, so adjust index
                    print("", file=log_file)

        return
    

    def conv_vxvyvz(self, kdim, lat, lon, wix, wiy, vx1d, vy1d, vz1d):
        """
        Converts wind components from lat-lon system to absolute system.

        Parameters:
        kdim (int)          : Number of vertical levels (z-dimension)
        lat (float)         : Latitude
        lon (float)         : Longitude
        wix (np.ndarray)    : Zonal wind component on lat-lon (1D array of size kdim)
        wiy (np.ndarray)    : Meridional wind component on lat-lon (1D array of size kdim)
        vx1d (np.ndarray)   : Horizontal-x component on absolute system for horizontal wind (1D array of size kdim) [modified in-place]
        vy1d (np.ndarray)   : Horizontal-y component on absolute system for horizontal wind (1D array of size kdim) [modified in-place]
        vz1d (np.ndarray)   : Vertical component on absolute system for horizontal wind (1D array of size kdim) [modified in-place]
        """

        # Iterate over each vertical level
        for k in range(kdim):
            unit_east = self.Sp_Unit_East(lon)
            unit_north = self.Sp_Unit_North(lon, lat)

            vx1d[k] = unit_east[0] * wix[k] + unit_north[0] * wiy[k]
            vy1d[k] = unit_east[1] * wix[k] + unit_north[1] * wiy[k]
            vz1d[k] = unit_east[2] * wix[k] + unit_north[2] * wiy[k]

        return



    def simpson(self, pin1, pin2, pin3, t1, t2, t3, u1, u2, u3, geo1, geo3, lat, downward, nicamcore, cnst, rdtype):
        """
        Computes pressure at the next level using Simpson's integration method.

        Parameters:
        pin1 (float)      : Pressure at top
        pin2 (float)      : Pressure at middle
        pin3 (float)      : Pressure at bottom
        t1 (float)        : Temperature at top
        t2 (float)        : Temperature at middle
        t3 (float)        : Temperature at bottom
        u1 (float)        : Zonal wind at top
        u2 (float)        : Zonal wind at middle
        u3 (float)        : Zonal wind at bottom
        geo1 (float)      : Geopotential at top
        geo3 (float)      : Geopotential at bottom
        lat (float)       : Latitude
        downward (bool)   : Downward switch
        nicamcore (bool)  : Nicamcore switch

        Returns:
        float             : Computed pressure at next level
        """

        # Compute dz
        dz = (geo1 - geo3) / cnst.CONST_GRAV * 0.5

        # Compute Coriolis and centrifugal forces if nicamcore is enabled
        if nicamcore:
            f_cf = np.array([
                2.0 * cnst.CONST_OHM * u1 * np.cos(lat) + (u1 ** 2.0) / cnst.CONST_RADIUS,
                2.0 * cnst.CONST_OHM * u2 * np.cos(lat) + (u2 ** 2.0) / cnst.CONST_RADIUS,
                2.0 * cnst.CONST_OHM * u3 * np.cos(lat) + (u3 ** 2.0) / cnst.CONST_RADIUS
            ])
        else:
            f_cf = np.zeros(3, dtype=rdtype)

        # Compute density
        rho = np.array([
            pin1 / (cnst.CONST_Rdry * t1),
            pin2 / (cnst.CONST_Rdry * t2),
            pin3 / (cnst.CONST_Rdry * t3)
        ])

        # Compute pressure at next level
        factor = (1.0 / 3.0) * rho[0] * (f_cf[0] - cnst.CONST_GRAV) + (4.0 / 3.0) * rho[1] * (f_cf[1] - cnst.CONST_GRAV) + (1.0 / 3.0) * rho[2] * (f_cf[2] - cnst.CONST_GRAV)

        if downward:
            pout = pin1 - factor * dz
        else:
            pout = pin3 + factor * dz

        return pout



    def Sp_Unit_East(self, lon, rdtype=np.float64):
        """
        Computes the eastward unit vector in a spherical coordinate system.

        Parameters:
        lon (float) : Longitude in radians

        Returns:
        np.ndarray  : 3D unit vector pointing east
        """

        unit_east = np.array([
            -np.sin(lon),  # x-direction
            np.cos(lon),  # y-direction
            0.0           # z-direction
        ], dtype=rdtype)

        return unit_east



    def Sp_Unit_North(self, lon, lat, rdtype=np.float64):
     
        """
        Computes the northward unit vector in a spherical coordinate system.

        Parameters:
        lon (float) : Longitude in radians
        lat (float) : Latitude in radians

        Returns:
        np.ndarray  : 3D unit vector pointing north
        """

        unit_north = np.array([
            -np.sin(lat) * np.cos(lon),  # x-direction
            -np.sin(lat) * np.sin(lon),  # y-direction
            np.cos(lat)                 # z-direction
        ], dtype=rdtype)

        return unit_north
