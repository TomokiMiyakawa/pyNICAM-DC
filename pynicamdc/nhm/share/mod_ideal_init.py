import toml
import numpy as np
#from mpi4py import MPI
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.nhm.share import mod_dcmip_ic as dcmip_ic
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
        DIAG_var = np.zeros((adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, 6 + rcnf.TRC_vmax), dtype=rdtype)

        # Equivalent to `character(len=H_SHORT) :: init_type = ''`
        init_type = ""  
        test_case = ""  

        # Equivalent to `real(RP) :: eps_geo2prs = 1.E-2_RP`
        eps_geo2prs = rdtype(1.0e-2)  

        # Equivalent to `logical` variables in Fortran
        nicamcore = True
        chemtracer = False
        prs_rebuild = False

        self.Kap = cnst.CONST_Rdry / cnst.CONST_CPdry
        self.d2r = cnst.CONST_PI/rdtype(180.0)
        self.r2d = rdtype(180.0)/cnst.CONST_PI


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
            #     IDEAL_init_DCMIP2012(adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, init_type, rcnf.DIAG_var)

            case "Heldsuarez":
                DIAG_var = self.hs_init(adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall, cnst, rcnf, grd, rdtype)

            case "Jablonowski":
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:                        
                        print(f"*** test case   : {test_case.strip()}", file=log_file)
                        print(f"*** eps_geo2prs = {eps_geo2prs}", file=log_file)
                        print(f"*** nicamcore   = {nicamcore}", file=log_file)
                DIAG_var = self.jbw_init(adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall, test_case, eps_geo2prs, nicamcore, cnst, rcnf, grd, rdtype)

            case "Jablonowski-Moist":
                DIAG_var = self.jbw_moist_init(adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall, test_case, chemtracer, cnst, rcnf, grd, rdtype)

            case "Supercell":
                DIAG_var = self.sc_init(adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall, test_case, prs_rebuild, cnst, rcnf, grd, rdtype)

            case "Tropical-Cyclone":
                DIAG_var = self.tc_init(adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall, prs_rebuild, cnst, rcnf, grd, rdtype)

            case "Traceradvection":
                DIAG_var = self.tracer_init(adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall, test_case, cnst, rcnf, grd, rdtype)

            case "Mountainwave":
                DIAG_var = self.mountwave_init(adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall, test_case, cnst, rcnf, grd, rdtype)
                # if IO_L:
                #     print(f"*** test case: {test_case.strip()}")
                # mountwave_init(adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, test_case, rcnf.DIAG_var)

            case "Gravitywave":
                DIAG_var = self.gravwave_init(adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall, cnst, rcnf, grd, rdtype)

            case "Tomita2004":
                DIAG_var = self.tomita_init(adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall, cnst, rcnf, grd, rdtype)

            case _:
                print("xxx [dycore_input] Invalid init_type. STOP.")
                raise SystemExit("PRC_MPIstop called")

        return DIAG_var


    def jbw_init(self, idim, jdim, kdim, lall, test_case, eps_geo2prs, nicamcore, cnst, rcnf, grd, rdtype):
        # Vectorized Jablonowski-Williamson dry baroclinic-wave init.
        return self._jbw_init_vec(idim, jdim, kdim, lall, test_case, eps_geo2prs, nicamcore, cnst, rcnf, grd, rdtype)

    def _jbw_init_OLD(self, idim, jdim, kdim, lall, test_case, eps_geo2prs, nicamcore, cnst, rcnf, grd, rdtype):
        # SUPERSEDED (broken): the eta solve is nested inside the z-build k-loop, the
        # convergence `signal` (a bool passed by value) never propagates so every column
        # runs the full itrmax, geo2prs runs a 400-iter loop, and no perturbation is applied.
        # Kept for reference; not called.

        DIAG_var = np.zeros((idim, jdim, kdim, lall, 6 + rcnf.TRC_vmax), dtype=rdtype)

        k0 = adm.ADM_K0

        eta_limit = True
        psgm = False
        logout = True

        lat = rdtype(0.0)      # Latitude on Icosahedral grid
        lon = rdtype(0.0)      # Longitude on Icosahedral grid
        ps = rdtype(0.0)       # Surface pressure

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
                        lat = grd.GRD_LAT[i, j, k0, l]
                        lon = grd.GRD_LON[i, j, k0, l]

                        signal = True

                        # Iteration process
                        for itr in range(self.itrmax):

#                            print(f"ITERATION: {itr}")

                            if itr == 0:
                                eta[:, :] = rdtype(1.0e-7)  # Jablonowski recommended initial value
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

    def _jbw_steady_state_vec(self, eta, latk, cnst, rdtype):
        # Vectorized nicamdc steady_state: horizontal-mean + meridional distribution of
        # tmp/geo and the zonal wind wix, from the eta level. eta/latk broadcast to (...,k,...).
        Rd = cnst.CONST_Rdry; g = cnst.CONST_GRAV; PI = cnst.CONST_PI
        RADIUS = cnst.CONST_RADIUS; OHM = cnst.CONST_OHM
        eta0 = rdtype(self.eta0); etaT = rdtype(self.etaT); t0 = rdtype(self.t0)
        delT = rdtype(self.delT); ganma = rdtype(self.ganma); u0 = rdtype(self.u0)
        r13 = rdtype(1.0)/rdtype(3.0)

        work1 = PI / rdtype(2.0)
        work2 = Rd * ganma / g
        eta_v = (eta - eta0) * work1
        wix = u0 * np.cos(eta_v) ** rdtype(1.5) * (np.sin(rdtype(2.0) * latk)) ** 2

        etaw2 = eta ** work2
        tmp_hi = t0 * etaw2
        geo_hi = t0 * g / ganma * (rdtype(1.0) - etaw2)
        tmp_lo = t0 * etaw2 + delT * (etaT - eta) ** 5
        geo_lo = ( t0 * g / ganma * (rdtype(1.0) - etaw2) - Rd * delT *
                   ((np.log(eta / etaT) + rdtype(137.0)/rdtype(60.0)) * etaT ** 5
                    - rdtype(5.0) * etaT ** 4 * eta
                    + rdtype(5.0) * etaT ** 3 * eta ** 2
                    - (rdtype(10.0)/rdtype(3.0)) * etaT ** 2 * eta ** 3
                    + (rdtype(5.0)/rdtype(4.0)) * etaT * eta ** 4
                    - (rdtype(1.0)/rdtype(5.0)) * eta ** 5) )
        hi = eta >= etaT
        tmp = np.where(hi, tmp_hi, tmp_lo)
        geo = np.where(hi, geo_hi, geo_lo)

        # meridional distribution
        work2b = rdtype(3.0)/rdtype(4.0) * (PI * u0 / Rd)
        cs32 = np.cos(eta_v) ** rdtype(1.5)
        A = (-rdtype(2.0) * (np.sin(latk)) ** 6 * (np.cos(latk) ** 2 + r13) + rdtype(10.0)/rdtype(63.0))
        B = (rdtype(8.0)/rdtype(5.0) * (np.cos(latk)) ** 3 * ((np.sin(latk)) ** 2 + rdtype(2.0)/rdtype(3.0)) - PI/rdtype(4.0))
        tmp = tmp + work2b * eta * np.sin(eta_v) * np.cos(eta_v) ** rdtype(0.5) * (A * rdtype(2.0) * u0 * cs32 + B * RADIUS * OHM)
        geo = geo + u0 * cs32 * (A * u0 * cs32 + B * RADIUS * OHM)
        return tmp, geo, wix

    def _jbw_geo2prs_vec(self, ps, latk, tmp, geo, wix, nicamcore, cnst, rdtype):
        # Vectorized nicamdc geo2prs (iteration=.false. path): hydrostatic upward integration
        # followed by a single upward Simpson pass. k-recursions are python loops over the
        # (short) vertical axis with array ops over all columns.
        Rd = cnst.CONST_Rdry; g = cnst.CONST_GRAV; OHM = cnst.CONST_OHM; a = cnst.CONST_RADIUS
        cosl = np.cos(latk)
        kdim = tmp.shape[2]
        pp = np.zeros_like(tmp)
        pp[:, :, 0, :] = ps
        for k in range(1, kdim):
            dz = (geo[:, :, k, :] - geo[:, :, k-1, :]) / g
            if nicamcore:
                uave = (wix[:, :, k, :] + wix[:, :, k-1, :]) * rdtype(0.5)
                f_cf = rdtype(2.0) * OHM * uave * cosl[:, :, 0, :] + uave ** 2 / a
            else:
                f_cf = rdtype(0.0)
            num = rdtype(1.0) + dz * (f_cf - g) / (rdtype(2.0) * Rd * tmp[:, :, k-1, :])
            den = rdtype(1.0) - dz * (f_cf - g) / (rdtype(2.0) * Rd * tmp[:, :, k, :])
            pp[:, :, k, :] = pp[:, :, k-1, :] * num / den
        prs = pp.copy()

        # finalize: single upward Simpson pass (downward=False)
        prs[:, :, 0, :] = ps
        pp = pp.copy()
        pp[:, :, 0, :] = ps
        for k in range(2, kdim):
            pp[:, :, k, :] = self._jbw_simpson_vec(
                prs[:, :, k, :], prs[:, :, k-1, :], prs[:, :, k-2, :],
                tmp[:, :, k, :], tmp[:, :, k-1, :], tmp[:, :, k-2, :],
                wix[:, :, k, :], wix[:, :, k-1, :], wix[:, :, k-2, :],
                geo[:, :, k, :], geo[:, :, k-2, :], latk[:, :, 0, :], nicamcore, cnst, rdtype)
        return pp

    def _jbw_simpson_vec(self, pin1, pin2, pin3, t1, t2, t3, u1, u2, u3, geo1, geo3, lat, nicamcore, cnst, rdtype):
        Rd = cnst.CONST_Rdry; g = cnst.CONST_GRAV; OHM = cnst.CONST_OHM; a = cnst.CONST_RADIUS
        dz = (geo1 - geo3) / g * rdtype(0.5)
        if nicamcore:
            cosl = np.cos(lat)
            f0 = rdtype(2.0)*OHM*u1*cosl + u1 ** 2 / a
            f1 = rdtype(2.0)*OHM*u2*cosl + u2 ** 2 / a
            f2 = rdtype(2.0)*OHM*u3*cosl + u3 ** 2 / a
        else:
            f0 = f1 = f2 = rdtype(0.0)
        r0 = pin1 / (Rd * t1); r1 = pin2 / (Rd * t2); r2 = pin3 / (Rd * t3)
        r13 = rdtype(1.0)/rdtype(3.0); r43 = rdtype(4.0)/rdtype(3.0)
        factor = r13 * r0 * (f0 - g) + r43 * r1 * (f1 - g) + r13 * r2 * (f2 - g)
        return pin3 + factor * dz    # downward=False path

    def _jbw_conv_vxvyvz_vec(self, lonk, latk, wix, wiy, rdtype):
        Ex = -np.sin(lonk); Ey = np.cos(lonk)                     # east unit (z=0)
        Nx = -np.sin(latk) * np.cos(lonk); Ny = -np.sin(latk) * np.sin(lonk); Nz = np.cos(latk)
        vx = Ex * wix + Nx * wiy
        vy = Ey * wix + Ny * wiy
        vz = Nz * wiy
        return vx, vy, vz

    def _jbw_perturbation_vec(self, DIAG_var, lat, lon, cnst, rdtype):
        # nicamdc perturbation(): add a localized Gaussian zonal-wind bump about (clat,clon).
        a = cnst.CONST_RADIUS; d2r = cnst.CONST_D2R
        cla = rdtype(self.clat) * d2r; clo = rdtype(self.clon) * d2r
        r = a * np.arccos(np.sin(cla) * np.sin(lat) + np.cos(cla) * np.cos(lat) * np.cos(lon - clo))
        rr = a / rdtype(10.0)
        ptb_wix = rdtype(self.uP) * np.exp(-(r / rr) ** 2)        # (i,j,l), k-independent
        latk = lat[:, :, None, :]; lonk = lon[:, :, None, :]
        pw = ptb_wix[:, :, None, :]
        pvx, pvy, pvz = self._jbw_conv_vxvyvz_vec(lonk, latk, pw, np.zeros_like(pw), rdtype)
        DIAG_var[:, :, :, :, 2] += pvx
        DIAG_var[:, :, :, :, 3] += pvy
        DIAG_var[:, :, :, :, 4] += pvz

    def _jbw_init_vec(self, idim, jdim, kdim, lall, test_case, eps_geo2prs, nicamcore, cnst, rcnf, grd, rdtype):
        g = cnst.CONST_GRAV; Rd = cnst.CONST_Rdry
        DIAG_var = np.zeros((idim, jdim, kdim, lall, 6 + rcnf.TRC_vmax), dtype=rdtype)
        k0 = adm.ADM_K0; kmin = adm.ADM_kmin; kmax = adm.ADM_kmax

        tc = test_case.strip()
        eta_limit = True; pertb = True; psgm = False
        if tc in ("1", "4-1"):   pertb = True
        elif tc in ("2", "4-2"): pertb = False
        elif tc == "3":          pertb = True;  psgm = True; eta_limit = False
        elif tc == "4":          pertb = False; psgm = True; eta_limit = False
        else:                    pertb = True
        if psgm:
            raise NotImplementedError("jbw_init: PS-distribution method (test_case 3/4) not yet vectorized")

        # z (idim,jdim,kdim,lall): z[kmin-1]=GRD_vz ZH@kmin ; z[k]=GRD_vz Z (kmin..kmax+1)
        z = np.zeros((idim, jdim, kdim, lall), dtype=rdtype)
        z[:, :, kmin-1, :] = grd.GRD_vz[:, :, kmin, :, grd.GRD_ZH]
        z[:, :, kmin:kmax+2, :] = grd.GRD_vz[:, :, kmin:kmax+2, :, grd.GRD_Z]
        lat = grd.GRD_LAT[:, :, k0, :]; lon = grd.GRD_LON[:, :, k0, :]     # (i,j,l)
        latk = lat[:, :, None, :]

        criteria = max(cnst.CONST_EPS * rdtype(10.0), rdtype(1.0e-14))
        eta = np.zeros((idim, jdim, kdim, lall), dtype=rdtype)
        tmp = np.zeros_like(eta); geo = np.zeros_like(eta); wix = np.zeros_like(eta)
        active = np.ones((idim, jdim, lall), dtype=bool)
        for itr in range(self.itrmax):
            if itr == 0:
                eta[:] = rdtype(1.0e-7)
                tmp, geo, wix = self._jbw_steady_state_vec(eta, latk, cnst, rdtype)
            else:
                F = -g * z + geo
                Feta = -(Rd / eta) * tmp
                eta_new = eta - F / Feta
                if eta_limit:
                    eta_new = np.minimum(eta_new, rdtype(1.0))
                eta_new = np.maximum(eta_new, cnst.CONST_EPS)
                am = active[:, :, None, :]
                diff = np.where(am, np.abs(eta_new - eta), rdtype(0.0))
                eta = np.where(am, eta_new, eta)
                newly = np.max(diff, axis=2) < criteria
                tmp_n, geo_n, wix_n = self._jbw_steady_state_vec(eta, latk, cnst, rdtype)
                tmp = np.where(am, tmp_n, tmp); geo = np.where(am, geo_n, geo); wix = np.where(am, wix_n, wix)
                active = active & ~newly
                if not active.any():
                    break

        prs = self._jbw_geo2prs_vec(rdtype(self.p0), latk, tmp, geo, wix, nicamcore, cnst, rdtype)
        vx, vy, vz = self._jbw_conv_vxvyvz_vec(lon[:, :, None, :], latk, wix, np.zeros_like(wix), rdtype)

        DIAG_var[:, :, :, :, 0] = prs
        DIAG_var[:, :, :, :, 1] = tmp
        DIAG_var[:, :, :, :, 2] = vx
        DIAG_var[:, :, :, :, 3] = vy
        DIAG_var[:, :, :, :, 4] = vz
        if pertb:
            self._jbw_perturbation_vec(DIAG_var, lat, lon, cnst, rdtype)
        return DIAG_var

    def gravwave_init(self, idim, jdim, kdim, lall, cnst, rcnf, grd, rdtype):
        # Vectorized DCMIP2012-31 non-hydrostatic gravity wave (test 3-1), from nicamdc
        # gravwave_init. The IC hardcodes reduced-Earth X=125 / Om=0, so the run must use
        # small_planet_factor=125 (RADIUS = a/125) for a consistent grid.
        DIAG_var = np.zeros((idim, jdim, kdim, lall, 6 + rcnf.TRC_vmax), dtype=rdtype)
        k0 = adm.ADM_K0; kmin = adm.ADM_kmin; kmax = adm.ADM_kmax

        z = np.zeros((idim, jdim, kdim, lall), dtype=rdtype)
        z[:, :, kmin-1, :] = grd.GRD_vz[:, :, kmin, :, grd.GRD_ZH]
        z[:, :, kmin:kmax+2, :] = grd.GRD_vz[:, :, kmin:kmax+2, :, grd.GRD_Z]
        latk = grd.GRD_LAT[:, :, k0, :][:, :, None, :]
        lonk = grd.GRD_LON[:, :, k0, :][:, :, None, :]
        latb = np.broadcast_to(latk, z.shape)
        lonb = np.broadcast_to(lonk, z.shape)

        p, u, v, w, t, rho = dcmip_ic.test3_gravity_wave(lonb, latb, z, rdtype)
        vx, vy, vz = self._jbw_conv_vxvyvz_vec(lonb, latb, u, v, rdtype)

        DIAG_var[:, :, :, :, 0] = p
        DIAG_var[:, :, :, :, 1] = t
        DIAG_var[:, :, :, :, 2] = vx
        DIAG_var[:, :, :, :, 3] = vy
        DIAG_var[:, :, :, :, 4] = vz
        DIAG_var[:, :, :, :, 5] = w
        return DIAG_var

    def mountwave_init(self, idim, jdim, kdim, lall, test_case, cnst, rcnf, grd, rdtype):
        # DCMIP2012 test 2 over a Schaer mountain (the mountain enters the run through the
        # terrain-following grid GRD_vz, built by IDEAL_topo -- so topo_io_mode must be IDEAL).
        # Two families, selected by test_case:
        #   * 2-0: steady-state atmosphere AT REST -- winds 0 (test2_steady_state_mountain);
        #   * 2-1/2-2: non-hydrostatic mountain WAVES -- a background zonal flow ueq=20 m/s over
        #     the mountain (test2_schaer_mountain), constant (2-1, shear=0) or vertically sheared
        #     (2-2, shear=1); the flow over the mountain is what generates the lee waves.
        # (nicamdc's mountwave_init sets a shear flag for 2-1/2-2 but never calls the windy IC,
        #  leaving them at rest; we wire it through here so the lee waves actually form.)
        DIAG_var = np.zeros((idim, jdim, kdim, lall, 6 + rcnf.TRC_vmax), dtype=rdtype)
        k0 = adm.ADM_K0; kmin = adm.ADM_kmin; kmax = adm.ADM_kmax
        z = np.zeros((idim, jdim, kdim, lall), dtype=rdtype)
        z[:, :, kmin-1, :] = grd.GRD_vz[:, :, kmin, :, grd.GRD_ZH]
        z[:, :, kmin:kmax+2, :] = grd.GRD_vz[:, :, kmin:kmax+2, :, grd.GRD_Z]

        tc = test_case.strip()
        if tc in ('1', '2-1', '2', '2-2'):
            shear = 1 if tc in ('2', '2-2') else 0
            lat = grd.GRD_LAT[:, :, k0, :][:, :, None, :]        # (i,j,1,l), height-independent
            lon = grd.GRD_LON[:, :, k0, :][:, :, None, :]
            latb = np.broadcast_to(lat, z.shape); lonb = np.broadcast_to(lon, z.shape)
            p, u, v, w, t, rho = dcmip_ic.test2_schaer_mountain(latb, z, shear, rdtype)
            vx, vy, vz = self._jbw_conv_vxvyvz_vec(lonb, latb, u, v, rdtype)   # zonal u -> (vx,vy,vz)
            DIAG_var[:, :, :, :, 0] = p
            DIAG_var[:, :, :, :, 1] = t
            DIAG_var[:, :, :, :, 2] = vx
            DIAG_var[:, :, :, :, 3] = vy
            DIAG_var[:, :, :, :, 4] = vz
            # w = 0 (DIAG[5]); passive tracer q = 0
        else:
            # test 2-0: atmosphere at rest; winds u=v=w=0 -> DIAG[2..5] stay zero
            p, u, v, w, t, rho = dcmip_ic.test2_steady_state_mountain(z, rdtype)
            DIAG_var[:, :, :, :, 0] = p
            DIAG_var[:, :, :, :, 1] = t
        return DIAG_var

    def _diag_pressure_vec(self, z, rho, t, q, prs_rebuild, prs_dry, cnst, rdtype):
        # Vectorized nicamdc diag_pressure (uses MODEL constants Rd/Rv/g). z/rho/t/q are
        # (i,j,k,l). ps is unused by the Fortran (kept out).
        Rd = cnst.CONST_Rdry; Rv = cnst.CONST_Rvap; g = cnst.CONST_GRAV
        kdim = t.shape[2]; two = rdtype(2.0); one = rdtype(1.0)
        if prs_dry:
            prs = rho * t * Rd
            if prs_rebuild:
                for k in range(1, kdim):
                    dz = z[:, :, k, :] - z[:, :, k-1, :]
                    prs[:, :, k, :] = (prs[:, :, k-1, :]
                        * (one - dz*g/(two*Rd*t[:, :, k-1, :])) / (one + dz*g/(two*Rd*t[:, :, k, :])))
        else:
            Rmix = (one - q) * Rd + q * Rv
            prs = rho * t * Rmix
            if prs_rebuild:
                for k in range(1, kdim):
                    dz = z[:, :, k, :] - z[:, :, k-1, :]
                    R0 = (one - q[:, :, k-1, :]) * Rd + q[:, :, k-1, :] * Rv
                    R1 = (one - q[:, :, k,   :]) * Rd + q[:, :, k,   :] * Rv
                    prs[:, :, k, :] = (prs[:, :, k-1, :]
                        * (one - dz*g/(two*R0*t[:, :, k-1, :])) / (one + dz*g/(two*R1*t[:, :, k, :])))
        return prs

    def tc_init(self, idim, jdim, kdim, lall, prs_rebuild, cnst, rcnf, grd, rdtype):
        # Vectorized DCMIP2016-12 tropical cyclone (moist). Uses GRD_Z for z at all k.
        DIAG_var = np.zeros((idim, jdim, kdim, lall, 6 + rcnf.TRC_vmax), dtype=rdtype)
        k0 = adm.ADM_K0
        Mvap = rdtype(0.608)
        RdovRv = cnst.CONST_Rdry / cnst.CONST_Rvap
        Mvap2 = (rdtype(1.0) - RdovRv) / RdovRv

        z = grd.GRD_vz[:, :, :, :, grd.GRD_Z].astype(rdtype)              # (i,j,k,l)
        latk = grd.GRD_LAT[:, :, k0, :][:, :, None, :]
        lonk = grd.GRD_LON[:, :, k0, :][:, :, None, :]
        latb = np.broadcast_to(latk, z.shape); lonb = np.broadcast_to(lonk, z.shape)

        u, v, t, thetav, ps, rho, q = dcmip_ic.tropical_cyclone_test(lonb, latb, z, rdtype)
        q = np.maximum(q, rdtype(0.0))
        # re-evaluate temperature from virtual temperature
        tmp = t * ((rdtype(1.0) + Mvap * q) / (rdtype(1.0) + Mvap2 * q))
        prs = self._diag_pressure_vec(z, rho, tmp, q, prs_rebuild, False, cnst, rdtype)
        vx, vy, vz = self._jbw_conv_vxvyvz_vec(lonb, latb, u, v, rdtype)

        DIAG_var[:, :, :, :, 0] = prs
        DIAG_var[:, :, :, :, 1] = tmp
        DIAG_var[:, :, :, :, 2] = vx
        DIAG_var[:, :, :, :, 3] = vy
        DIAG_var[:, :, :, :, 4] = vz
        # DIAG[5]=w=0
        DIAG_var[:, :, :, :, rcnf.DIAG_vmax0 + rcnf.I_QV] = q
        return DIAG_var

    def sc_init(self, idim, jdim, kdim, lall, test_case, prs_rebuild, cnst, rcnf, grd, rdtype):
        # Vectorized DCMIP2016-13 Klemp supercell (moist). Stage-1 background is grid-
        # independent (built once); Stage-2 samples it at GRD_Z (zcoords=1). test_case
        # '1' -> with thermal perturbation, '2' -> no perturbation.
        DIAG_var = np.zeros((idim, jdim, kdim, lall, 6 + rcnf.TRC_vmax), dtype=rdtype)
        k0 = adm.ADM_K0

        tc = test_case.strip()
        if tc == '1':
            pert = 1
        elif tc == '2':
            pert = 0
        else:
            print(f"xxx [sc_init] Invalid test_case: '{tc}'. STOP.")
            prc.prc_mpistop(std.io_l, std.fname_log)

        # NOTE: supercell wrapper uses Mvap=0.61 (hardcoded in nicamdc sc_init),
        # NOT 0.608 as tc_init; Mvap2 from MODEL constants.
        Mvap = rdtype(0.61)
        RdovRv = cnst.CONST_Rdry / cnst.CONST_Rvap
        Mvap2 = (rdtype(1.0) - RdovRv) / RdovRv

        z = grd.GRD_vz[:, :, :, :, grd.GRD_Z].astype(rdtype)             # (i,j,k,l)
        latk = grd.GRD_LAT[:, :, k0, :][:, :, None, :]
        lonk = grd.GRD_LON[:, :, k0, :][:, :, None, :]
        latb = np.broadcast_to(latk, z.shape); lonb = np.broadcast_to(lonk, z.shape)

        bg = dcmip_ic.supercell_init_background(rdtype)
        u, v, t, thetav, rho, q = dcmip_ic.supercell_test(lonb, latb, z, bg, pert, rdtype)

        # fix negative q, and force q=0 above the tropopause (12 km)
        q = np.maximum(q, rdtype(0.0))
        q = np.where(z > rdtype(12000.0), rdtype(0.0), q)
        # re-evaluate temperature from virtual temperature (with the cut q)
        tmp = t * ((rdtype(1.0) + Mvap * q) / (rdtype(1.0) + Mvap2 * q))
        prs = self._diag_pressure_vec(z, rho, tmp, q, prs_rebuild, False, cnst, rdtype)
        vx, vy, vz = self._jbw_conv_vxvyvz_vec(lonb, latb, u, v, rdtype)

        DIAG_var[:, :, :, :, 0] = prs
        DIAG_var[:, :, :, :, 1] = tmp
        DIAG_var[:, :, :, :, 2] = vx
        DIAG_var[:, :, :, :, 3] = vy
        DIAG_var[:, :, :, :, 4] = vz
        # DIAG[5]=w=0
        DIAG_var[:, :, :, :, rcnf.DIAG_vmax0 + rcnf.I_QV] = q
        return DIAG_var

    def jbw_moist_init(self, idim, jdim, kdim, lall, test_case, chemtracer, cnst, rcnf, grd, rdtype):
        # Vectorized DCMIP2016-2x moist baroclinic wave. The IC (baroclinic_wave_test)
        # supplies rho/q/wind; pressure is rebuilt HYDROSTATICALLY (GRD_afact/bfact half-
        # level factors), temperature from the rebuilt pressure + moist gas constant.
        # q is SPECIFIC humidity. NOTE: nicamdc calls BNDCND_thermo here but BNDCND_setup
        # has NOT run yet at IC time, so its BC flags are all .false. -> the call is a
        # NO-OP; the ghost levels keep their raw values (top prs/tem = 0). We match that by
        # NOT applying the BC (the model re-derives the ghosts every step at runtime).
        DIAG_var = np.zeros((idim, jdim, kdim, lall, 6 + rcnf.TRC_vmax), dtype=rdtype)
        k0 = adm.ADM_K0
        kmin = adm.ADM_kmin; kmax = adm.ADM_kmax
        g = cnst.CONST_GRAV; Rd = cnst.CONST_Rdry; Rv = cnst.CONST_Rvap
        one = rdtype(1.0)

        tc = test_case.strip()
        cases = {'1': (1, 0), '2': (1, 1), '3': (0, 0), '4': (0, 1), '5': (1, -99), '6': (0, -99)}
        if tc not in cases:
            print(f"xxx [jbw_moist_init] Invalid test_case: '{tc}'. STOP.")
            prc.prc_mpistop(std.io_l, std.fname_log)
        moist, pertt = cases[tc]
        if moist == 1 and rcnf.NQW_MAX < 3:
            print(f"xxx [jbw_moist_init] NQW_MAX is not enough! requires >= 3. {rcnf.NQW_MAX}")
            prc.prc_mpistop(std.io_l, std.fname_log)

        z = grd.GRD_vz[:, :, :, :, grd.GRD_Z].astype(rdtype)             # (i,j,k,l)
        zh = grd.GRD_vz[:, :, :, :, grd.GRD_ZH].astype(rdtype)
        latk = grd.GRD_LAT[:, :, k0, :][:, :, None, :]
        lonk = grd.GRD_LON[:, :, k0, :][:, :, None, :]
        latb = np.broadcast_to(latk, z.shape); lonb = np.broadcast_to(lonk, z.shape)

        u, v, t, thetav, ps, rho, q = dcmip_ic.baroclinic_wave_test(0, moist, pertt, 1.0, lonb, latb, z, rdtype)
        # mixing ratio -> specific humidity, fix negatives
        q = np.maximum(q / (one + q), rdtype(0.0))

        # hydrostatic pressure rebuild (ps = p0 constant)
        ps_val = rdtype(dcmip_ic.P0_DCMIP)
        prs = np.zeros_like(z)
        prs[:, :, kmin - 1, :] = ps_val
        prs[:, :, kmin, :] = ps_val - rho[:, :, kmin, :] * g * (z[:, :, kmin, :] - zh[:, :, kmin, :])
        for k in range(kmin + 1, kmax + 1):
            dpdz = -g * (rho[:, :, k, :] * grd.GRD_afact[k] + rho[:, :, k - 1, :] * grd.GRD_bfact[k])
            prs[:, :, k, :] = prs[:, :, k - 1, :] + dpdz * (z[:, :, k, :] - z[:, :, k - 1, :])

        # temperature from rebuilt pressure and moist gas constant (specific humidity q).
        # Uses the RAW prs (incl. ghost rows: prs[kmin-1]=ps, prs[kmax+1]=0) -- matching
        # nicamdc, which computes tem BEFORE the pressure ghost BC.
        Rmix = Rd * (one - q) + Rv * q
        tmp = prs / (rho * Rmix)
        # The top-ghost (kmax+1) prs is STILL 0 here (its hydrostatic BC is applied below,
        # matching nicamdc which computes tem before the pressure ghost BC), so tmp[kmax+1]=0.
        # nicamdc's IC also leaves tem[kmax+1]=0, but nicamdc survives forward because its
        # runtime BNDCND_thermo (BND_TYPE_T_TOP='TEM', is_top_tem after BNDCND_setup) resets
        # tem[kmax+1]=tem[kmax] each step BEFORE the ghost can reach a physical cell. pyNICAM
        # instead creates the ghost prognostic at INIT via cnvvar_diag2prg (THRMDYN_rhoein:
        # rho=pre/((qd*Rd+qv*Rv)*tem)) -- BEFORE BNDCND_setup runs -- so tem=0 -> rho=inf ->
        # NaN, which leaks before the first BNDCND_all can contain it. Pre-apply nicamdc's
        # runtime top-BC here (dT/dz=0, zero-gradient) so the init divide stays finite; this
        # is exactly the ghost value nicamdc's forward integration uses. The physical domain
        # (kmin..kmax) is untouched (bit-identical to nicamdc); only the non-physical ghost row
        # differs from nicamdc's IC snapshot (0 vs tem[kmax]) and is BNDCND-overwritten anyway.
        tmp[:, :, kmax + 1, :] = tmp[:, :, kmax, :]

        # Ghost-level pressure BC: unconditional hydrostatic extrapolation, exactly as
        # nicamdc BNDCND_thermo. Its tem/rho BC flags are .false. at IC time (BNDCND_setup
        # runs later), so ONLY pre is set here; the tem ghost rows keep their raw values
        # above. phi = GRD_Z * GRAV.
        phi = z * g
        prs[:, :, kmax + 1, :] = (prs[:, :, kmax - 1, :]
                                  - rho[:, :, kmax, :] * (phi[:, :, kmax + 1, :] - phi[:, :, kmax - 1, :]))
        prs[:, :, kmin - 1, :] = (prs[:, :, kmin + 1, :]
                                  - rho[:, :, kmin, :] * (phi[:, :, kmin - 1, :] - phi[:, :, kmin + 1, :]))

        vx, vy, vz = self._jbw_conv_vxvyvz_vec(lonb, latb, u, v, rdtype)

        DIAG_var[:, :, :, :, 0] = prs
        DIAG_var[:, :, :, :, 1] = tmp
        DIAG_var[:, :, :, :, 2] = vx
        DIAG_var[:, :, :, :, 3] = vy
        DIAG_var[:, :, :, :, 4] = vz
        DIAG_var[:, :, :, :, rcnf.DIAG_vmax0 + rcnf.I_QV] = q
        return DIAG_var

    def tracer_init(self, idim, jdim, kdim, lall, test_case, cnst, rcnf, grd, rdtype):
        # Vectorized DCMIP2012 test 1 pure passive-tracer advection (1-1 deformation,
        # 1-2 hadley, 1-3 orography). Analytic, zcoords=1, time=0. Distinct from the moist
        # inits: w (DIAG slot 5) is NONZERO, and the tracers go into the PASSIVE (NCHEM)
        # slots, not the moisture slot. Requires CHEM_TYPE=PASSIVE (chemvarparam CHEM_TRC_vmax
        # >= tracers). The bottom-ghost z uses the HALF level GRD_ZH (not GRD_Z).
        DIAG_var = np.zeros((idim, jdim, kdim, lall, 6 + rcnf.TRC_vmax), dtype=rdtype)
        k0 = adm.ADM_K0
        kmin = adm.ADM_kmin
        i_pasv = rcnf.DIAG_vmax0 + rcnf.NCHEM_STR                # DIAG slot of the 1st passive tracer

        # z: full levels from GRD_Z, except the bottom ghost uses GRD_ZH at kmin
        z = grd.GRD_vz[:, :, :, :, grd.GRD_Z].astype(rdtype).copy()
        z[:, :, kmin - 1, :] = grd.GRD_vz[:, :, kmin, :, grd.GRD_ZH].astype(rdtype)
        latk = grd.GRD_LAT[:, :, k0, :][:, :, None, :]
        lonk = grd.GRD_LON[:, :, k0, :][:, :, None, :]
        latb = np.broadcast_to(latk, z.shape); lonb = np.broadcast_to(lonk, z.shape)

        tc = test_case.strip()
        if tc in ('1', '1-1'):
            p, u, v, w, t, rho, q1, q2, q3, q4 = dcmip_ic.test1_advection_deformation(lonb, latb, z, rdtype)
            tracers = (q1, q2, q3, q4)
        elif tc in ('2', '1-2'):
            p, u, v, w, t, rho, q1 = dcmip_ic.test1_advection_hadley(lonb, latb, z, rdtype)
            zero = np.zeros_like(p)
            tracers = (q1, zero, zero, zero)
        elif tc in ('3', '1-3'):
            gc = grd.GRD_gz.astype(rdtype)[None, None, :, None]  # bar{z} (Gal-Chen), per k
            gc = np.broadcast_to(gc, z.shape)
            p, u, v, w, t, rho, q1, q2, q3, q4, zs = dcmip_ic.test1_advection_orography(lonb, latb, z, gc, rdtype)
            tracers = (q1, q2, q3, q4)
        else:
            print(f"xxx [tracer_init] Unknown test_case: '{tc}'. STOP.")
            prc.prc_mpistop(std.io_l, std.fname_log)

        vx, vy, vz = self._jbw_conv_vxvyvz_vec(lonb, latb, u, v, rdtype)
        DIAG_var[:, :, :, :, 0] = p
        DIAG_var[:, :, :, :, 1] = t
        DIAG_var[:, :, :, :, 2] = vx
        DIAG_var[:, :, :, :, 3] = vy
        DIAG_var[:, :, :, :, 4] = vz
        DIAG_var[:, :, :, :, 5] = w
        for i, qx in enumerate(tracers):
            DIAG_var[:, :, :, :, i_pasv + i] = qx
        return DIAG_var

    def hs_init(self, idim, jdim, kdim, lall, cnst, rcnf, grd, rdtype):
        # Vectorized Held-Suarez initial condition: hydrostatic column build with the HS
        # equilibrium temperature. Only pre (DIAG 0) and tem (DIAG 1) are set; winds and
        # tracers stay zero. SEQUENTIAL in k (each level's Newton uses the level below),
        # vectorized over (i,j,l) with per-level active-masking (freeze each column at the
        # iteration it converges, as the Fortran `exit` does).
        DIAG_var = np.zeros((idim, jdim, kdim, lall, 6 + rcnf.TRC_vmax), dtype=rdtype)
        kmin = adm.ADM_kmin; kmax = adm.ADM_kmax; k0 = adm.ADM_K0
        g = cnst.CONST_GRAV; Rd = cnst.CONST_Rdry; Cp = cnst.CONST_CPdry; PRE00 = cnst.CONST_PRE00
        RdovCp = Rd / Cp
        deltaT = rdtype(60.0); deltaTh = rdtype(10.0)
        half = rdtype(0.5); one = rdtype(1.0); T200 = rdtype(200.0); T315 = rdtype(315.0)
        prec = np.finfo(rdtype).precision
        eps_hs = max(rdtype(1.0e-10), rdtype(10.0) ** (-prec + 1))
        itrmax = self.itrmax

        gz = grd.GRD_vz[:, :, :, :, grd.GRD_Z].astype(rdtype)         # (i,j,k,l)
        latk = grd.GRD_LAT[:, :, k0, :]                              # (i,j,l)
        sin2 = np.sin(latk) ** 2; cos2 = np.cos(latk) ** 2

        # dz[kmin] = gz[kmin]; dz[k] = gz[k] - gz[k-1]
        dz = np.zeros_like(gz)
        dz[:, :, kmin, :] = gz[:, :, kmin, :]
        dz[:, :, kmin + 1:kmax + 2, :] = gz[:, :, kmin + 1:kmax + 2, :] - gz[:, :, kmin:kmax + 1, :]

        pre = np.zeros_like(gz); tem = np.zeros_like(gz)
        pre_sfc = PRE00
        tem_sfc = T315 - deltaT * sin2                              # (i,j,l)

        def hs_tem(p):                                              # HS equilibrium tem (i,j,l)
            t = (T315 - deltaT * sin2 - deltaTh * np.log(p / PRE00) * cos2) * (p / PRE00) ** RdovCp
            return np.maximum(T200, t)

        def newton(p0, t0, pref, tref, dzk):
            p = p0.copy(); t = t0.copy()
            active = np.ones(p.shape, dtype=bool)
            for _ in range(itrmax):
                p_save = p
                f = np.log(p / pref) / dzk + g / (Rd * half * (t + tref))
                df = one / (p * dzk)
                p_cand = p - f / df
                t_cand = hs_tem(p_cand)
                conv = np.abs(p_save / p_cand - one) <= eps_hs
                p = np.where(active, p_cand, p)
                t = np.where(active, t_cand, t)
                active = active & ~conv
                if not active.any():
                    break
            return p, t

        # bottom level (kmin): reference = surface
        dzk = dz[:, :, kmin, :]
        p0 = np.full(tem_sfc.shape, pre_sfc, dtype=rdtype)
        pre[:, :, kmin, :], tem[:, :, kmin, :] = newton(p0, tem_sfc, pre_sfc, tem_sfc, dzk)

        # upper levels (sequential): reference = level below
        for k in range(kmin + 1, kmax + 2):
            pref = pre[:, :, k - 1, :]; tref = tem[:, :, k - 1, :]
            p0 = pref.copy()
            t0 = np.maximum(T200, rdtype(300.0) * (p0 / PRE00) ** RdovCp)
            pre[:, :, k, :], tem[:, :, k, :] = newton(p0, t0, pref, tref, dz[:, :, k, :])

        # tentative ghost values at kmin-1 = surface
        pre[:, :, kmin - 1, :] = pre_sfc
        tem[:, :, kmin - 1, :] = tem_sfc
        DIAG_var[:, :, :, :, 0] = pre
        DIAG_var[:, :, :, :, 1] = tem
        return DIAG_var

    def tomita_init(self, idim, jdim, kdim, lall, cnst, rcnf, grd, rdtype):
        # Vectorized Tomita & Satoh (2004) Qian98-like mountain-wave IC: an analytic
        # balanced zonal state. Sets pre/tem/vx/vy/vz (winds are purely zonal, wiy=0).
        # NOTE: Gzero/Gphi = (g1/g2)**work with work = N2*a/(4 g Kap); at full-Earth
        # radius work ~ 199 and (g1/g2)**work OVERFLOWS to +Inf -> Pphi = NaN (nicamdc
        # has the SAME overflow). This test is meant for a reduced-radius (small planet).
        DIAG_var = np.zeros((idim, jdim, kdim, lall, 6 + rcnf.TRC_vmax), dtype=rdtype)
        kmin = adm.ADM_kmin; k0 = adm.ADM_K0
        a = cnst.CONST_RADIUS; g = cnst.CONST_GRAV; omega = cnst.CONST_OHM
        Rd = cnst.CONST_Rdry; Cp = cnst.CONST_CPdry; Kap = Rd / Cp
        N = rdtype(0.0187); prs0 = rdtype(1.0e5); ux0 = rdtype(40.0)
        N2 = N ** 2
        work = (N2 * a) / (rdtype(4.0) * g * Kap)

        def bigG(lat):
            c2 = np.cos(rdtype(2.0) * lat); c4 = np.cos(rdtype(4.0) * lat)
            g1 = (rdtype(2.0) * (rdtype(3.0) + rdtype(4.0) * c2 + c4) * ux0 ** 4
                  + rdtype(8.0) * (rdtype(3.0) + rdtype(4.0) * c2 + c4) * ux0 ** 3 * a * omega
                  + rdtype(8.0) * (rdtype(3.0) + rdtype(4.0) * c2 + c4) * ux0 ** 2 * a ** 2 * omega ** 2
                  - rdtype(16.0) * (rdtype(1.0) + c2) * ux0 ** 2 * a * g
                  - rdtype(32.0) * (rdtype(1.0) + c2) * ux0 * a ** 2 * g * omega
                  + rdtype(16.0) * a ** 2 * g ** 2)
            g2 = ux0 ** 4 + rdtype(4.0) * a * omega * ux0 ** 3 + rdtype(4.0) * a ** 2 * omega ** 2 * ux0 ** 2
            return (g1 / g2) ** work

        Gzero = bigG(rdtype(0.0))                                    # scalar (equator)
        latk = grd.GRD_LAT[:, :, k0, :][:, :, None, :]              # (i,j,1,l)
        lonk = grd.GRD_LON[:, :, k0, :][:, :, None, :]
        Gphi = bigG(latk)
        Pphi = prs0 * (Gzero / Gphi)                                # (i,j,1,l)

        # z: bottom ghost from GRD_ZH at kmin, else GRD_Z
        z = grd.GRD_vz[:, :, :, :, grd.GRD_Z].astype(rdtype).copy()
        z[:, :, kmin - 1, :] = grd.GRD_vz[:, :, kmin, :, grd.GRD_ZH].astype(rdtype)

        wix = ux0 * np.cos(latk)                                     # (i,j,1,l), zonal only
        wiy = np.zeros_like(wix)
        prs = Pphi * np.exp((-N2 * z) / (g * Kap))                   # (i,j,k,l)
        tmp = (g * Kap * (g - wix ** 2 / a - rdtype(2.0) * omega * wix * np.cos(latk))) / (N2 * Rd)  # (i,j,1,l)

        shp = prs.shape
        wix_b = np.broadcast_to(wix, shp); wiy_b = np.broadcast_to(wiy, shp)
        vx, vy, vz = self._jbw_conv_vxvyvz_vec(lonk, latk, wix_b, wiy_b, rdtype)
        DIAG_var[:, :, :, :, 0] = prs
        DIAG_var[:, :, :, :, 1] = np.broadcast_to(tmp, shp)
        DIAG_var[:, :, :, :, 2] = vx
        DIAG_var[:, :, :, :, 3] = vy
        DIAG_var[:, :, :, :, 4] = vz
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

        criteria = max(cnst.CONST_EPS * rdtype(10.0), rdtype(1.0e-14))  # Equivalent to max(EPS * 10.0_RP, 1.E-14_RP)

        for k in range(kdim):
            F[k] = -cnst.CONST_GRAV * z[k] + geo[k]
            Feta[k] = -rdtype(1.0) * (cnst.CONST_Rdry / eta[k, 0]) * tmp[k]  # Using eta[:, 0] for Fortran's eta(k,1)

            eta[k, 1] = eta[k, 0] - (F[k] / Feta[k])

            if eta_limit:  # [add] for PSDM (2013/12/20 R.Yoshida)
                eta[k, 1] = min(eta[k, 1], rdtype(1.0))  # Not allow eta > 1.0

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
        work1 = cnst.CONST_PI / rdtype(2.0)
        work2 = cnst.CONST_Rdry * self.ganma / cnst.CONST_GRAV

        for k in range(kdim):
            eta_v = (eta[k, 0] - self.eta0) * work1
            wix[k] = self.u0 * (np.cos(eta_v)) ** rdtype(1.5) * (np.sin(rdtype(2.0) * lat)) ** 2

            if eta[k, 0] >= self.etaT:
                tmp[k] = self.t0 * eta[k, 0] ** work2
                geo[k] = self.t0 * cnst.CONST_GRAV / self.ganma * (rdtype(1.0) - eta[k, 0] ** work2)

            elif eta[k, 0] < self.etaT:
                tmp[k] = self.t0 * eta[k, 0] ** work2 + self.delT * (self.etaT - eta[k, 0]) ** 5

                geo[k] = (self.t0 * cnst.CONST_GRAV / self.ganma * (rdtype(1.0) - eta[k, 0] ** work2) - cnst.CONST_Rdry * self.delT *
                    ((np.log(eta[k, 0] / self.etaT) + rdtype(137.0) / rdtype(60.0)) * self.etaT ** 5
                    - rdtype(5.0) * self.etaT ** 4 * eta[k, 0]
                    + rdtype(5.0) * self.etaT ** 3 * (eta[k, 0] ** 2)
                    - (rdtype(10.0) / rdtype(3.0)) * self.etaT ** 2 * (eta[k, 0] ** 3)
                    + (rdtype(5.0) / rdtype(4.0)) * self.etaT * (eta[k, 0] ** 4)
                    - (rdtype(1.0) / rdtype(5.0)) * (eta[k, 0] ** 5))
                )

            else:
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:
                        print("|-- ETA BOUNDARY ERROR: [steady state calc.]", file=log_file)
                        print(f"|-- ({k:3d})  eta: {eta[k, 0]:10.4f}", file=log_file)
                    prc.prc_mpi_stop(std.io_l, std.fname_log)   
                    raise SystemExit("STOP")

        # ---------- Meridional Distribution for Temperature and Geopotential ----------
        work1 = cnst.CONST_PI / rdtype(2.0)
        work2 = rdtype(3.0) / rdtype(4.0) * (cnst.CONST_PI * self.u0 / cnst.CONST_Rdry)

        for k in range(kdim):
            eta_v = (eta[k, 0] - self.eta0) * work1
            tmp[k] += (work2 * eta[k, 0] * np.sin(eta_v) * (np.cos(eta_v)) ** rdtype(0.5) *
                ((-rdtype(2.0) * (np.sin(lat)) ** 6 * (np.cos(lat) ** 2 + rdtype(1.0) / rdtype(3.0)) + rdtype(10.0) / rdtype(63.0))
                * rdtype(2.0) * self.u0 * (np.cos(eta_v)) ** rdtype(1.5)
                + (rdtype(8.0) / rdtype(5.0) * (np.cos(lat)) ** 3 * ((np.sin(lat)) ** 2 + rdtype(2.0) / rdtype(3.0)) - cnst.CONST_PI / rdtype(4.0))
                * cnst.CONST_RADIUS * cnst.CONST_OHM)
            )

            geo[k] += (self.u0 * (np.cos(eta_v)) ** rdtype(1.5) *
                ((-rdtype(2.0) * (np.sin(lat)) ** 6 * (np.cos(lat) ** 2 + rdtype(1.0) / rdtype(3.0)) + rdtype(10.0) / rdtype(63.0))
                * self.u0 * (np.cos(eta_v)) ** rdtype(1.5)
                + (rdtype(8.0) / rdtype(5.0) * (np.cos(lat)) ** 3 * ((np.sin(lat)) ** 2 + rdtype(2.0) / rdtype(3.0)) - cnst.CONST_PI / rdtype(4.0))
                * cnst.CONST_RADIUS * cnst.CONST_OHM)
            )

        wiy[:] = rdtype(0.0)  # Reset wiy to rdtype(0.0)

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
        lat0 = rdtype(0.691590985442682)
        eta1 = rdtype(1.0)
        pi_half = cnst.CONST_PI * rdtype(0.5)

        # Eta-related calculation
        eta_v = (eta1 - self.eta0) * pi_half

        # Temperature at bottom of eta-grid
        tmp0 = (
            self.t0
            + (rdtype(3.0) / rdtype(4.0) * (np.pi * self.u0 / cnst.CONST_Rdry)) * eta1 * np.sin(eta_v) * (np.cos(eta_v)) ** rdtype(0.5)
            * ((-rdtype(2.0) * (np.sin(lat0)) ** 6 * (np.cos(lat0) ** 2 + rdtype(1.0) / rdtype(3.0)) + rdtype(10.0) / rdtype(63.0))
                * rdtype(2.0) * self.u0 * (np.cos(eta_v)) ** rdtype(1.5)
                + (rdtype(8.0) / rdtype(5.0) * (np.cos(lat0)) ** 3 * ((np.sin(lat0)) ** 2 + rdtype(2.0) / rdtype(3.0)) - cnst.CONST_PI / rdtype(4.0))
                * cnst.CONST_RADIUS * cnst.CONST_OHM)
        )
        tmp1 = tmp[0]  # Equivalent to tmp(1) in Fortran

        # Wind speed at bottom of eta-grid
        ux1 = (self.u0 * np.cos(eta_v) ** rdtype(1.5)) * (np.sin(rdtype(2.0) * lat0)) ** 2
        ux2 = wix[0]  # Equivalent to wix(1) in Fortran

        # Topography calculation
        cs32ev = (np.cos((rdtype(1.0) - rdtype(0.252)) * pi_half)) ** rdtype(1.5)
        f1 = rdtype(10.0) / rdtype(63.0) - rdtype(2.0) * np.sin(lat) ** 6 * (np.cos(lat) ** 2 + rdtype(1.0) / rdtype(3.0))
        f2 = rdtype(1.6) * np.cos(lat) ** 3 * (np.sin(lat) ** 2 + rdtype(2.0) / rdtype(3.0)) - rdtype(0.25) * cnst.CONST_PI
        hgt1 = -rdtype(1.0) * self.u0 * cs32ev * (f1 * self.u0 * cs32ev + f2 * cnst.CONST_RADIUS * cnst.CONST_OHM) / cnst.CONST_GRAV
        hgt0 = rdtype(0.0)

        # Pressure estimation
        dz = hgt1 - hgt0
        if nicamcore:
            uave = (ux1 + ux2) * rdtype(0.5)
            f_cf = rdtype(2.0) * cnst.CONST_OHM * uave * np.cos(lat) + (uave ** 2) / cnst.CONST_RADIUS
        else:
            f_cf = rdtype(0.0)

        ps = self.p0 * (rdtype(1.0) + dz * (f_cf - cnst.CONST_GRAV) / (rdtype(2.0) * cnst.CONST_Rdry * tmp0)) / (rdtype(1.0) - dz * (f_cf - cnst.CONST_GRAV) / (rdtype(2.0) * cnst.CONST_Rdry * tmp1))

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
                uave = (wix[k] + wix[k - 1]) * rdtype(0.5)
                f_cf = rdtype(2.0) * cnst.CONST_OHM * uave * np.cos(lat) + (uave ** 2) / cnst.CONST_RADIUS
            else:
                f_cf = rdtype(0.0)

            pp[k] = pp[k - 1] * (rdtype(1.0) + dz * (f_cf - cnst.CONST_GRAV) / (rdtype(2.0) * cnst.CONST_Rdry * tmp[k - 1])) / \
                            (rdtype(1.0) - dz * (f_cf - cnst.CONST_GRAV) / (rdtype(2.0) * cnst.CONST_Rdry * tmp[k]))

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
        dz = (geo1 - geo3) / cnst.CONST_GRAV * rdtype(0.5)

        # Compute Coriolis and centrifugal forces if nicamcore is enabled
        if nicamcore:
            f_cf = np.array([
                rdtype(2.0) * cnst.CONST_OHM * u1 * np.cos(lat) + (u1 ** 2) / cnst.CONST_RADIUS,
                rdtype(2.0) * cnst.CONST_OHM * u2 * np.cos(lat) + (u2 ** 2) / cnst.CONST_RADIUS,
                rdtype(2.0) * cnst.CONST_OHM * u3 * np.cos(lat) + (u3 ** 2) / cnst.CONST_RADIUS
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
        factor = (rdtype(1.0) / rdtype(3.0)) * rho[0] * (f_cf[0] - cnst.CONST_GRAV) + (rdtype(4.0) / rdtype(3.0)) * rho[1] * (f_cf[1] - cnst.CONST_GRAV) + (rdtype(1.0) / rdtype(3.0)) * rho[2] * (f_cf[2] - cnst.CONST_GRAV)

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
            rdtype(0.0)           # z-direction
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
