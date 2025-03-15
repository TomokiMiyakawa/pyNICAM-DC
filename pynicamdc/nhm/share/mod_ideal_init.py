import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf


class Ide:
    
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

    def dycore_input(self, fname_in, cnst, rcnf, rdtype):

        # Equivalent to `real(RP), intent(out) :: DIAG_var(ADM_gall,ADM_kall,ADM_lall,6+TRC_VMAX)`
        DIAG_var = np.zeros((adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, 6 + rcnf.TRC_VMAX), dtype=rdtype)

        # Equivalent to `character(len=H_SHORT) :: init_type = ''`
        init_type = ""  
        test_case = ""  

        # Equivalent to `real(RP) :: eps_geo2prs = 1.E-2_RP`
        eps_geo2prs = 1.0e-2  

        # Equivalent to `logical` variables in Fortran
        nicamcore = True
        chemtracer = False
        prs_rebuild = False

        self.Kap = cnst.CONST_Rd / cnst.CONST_Cp
        self.d2r = cnst.CONST_pi/180.0
        self.r2d = 180.0/cnst.CONST_pi


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
                print("Heldsuarez not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                #hs_init(adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, rcnf.DIAG_var)

            case "Jablonowski":
                if std.io_l:
                    with open(std.fname_log, 'a') as log_file:                        
                        print(f"*** test case   : {test_case.strip()}", file=log_file)
                        print(f"*** eps_geo2prs = {eps_geo2prs}", file=log_file)
                        print(f"*** nicamcore   = {nicamcore}", file=log_file)
                DIAG_var = self.jbw_init(adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall, test_case, eps_geo2prs, nicamcore)

            case "Jablonowski-Moist":
                print("Jablonowski-Moist not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                #if IO_L:
                #    print(f"*** test case   : {test_case.strip()}")
                #    print(f"*** nicamcore   = {nicamcore}")
                #    print(f"*** chemtracer  = {chemtracer}")
                #jbw_moist_init(adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, test_case, chemtracer, rcnf.DIAG_var)

            case "Supercell":
                print("Supercell not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                # if IO_L:
                #     print(f"*** test case   : {test_case.strip()}")
                #     print(f"*** nicamcore   = {nicamcore}")
                # sc_init(adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, test_case, prs_rebuild, rcnf.DIAG_var)

            case "Tropical-Cyclone":
                print("Tropical-Cyclone not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                # if IO_L:
                #     print(f"*** nicamcore   = {nicamcore}")
                # tc_init(adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, prs_rebuild, rcnf.DIAG_var)

            case "Traceradvection":
                print("Traceradvection not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                # if IO_L:
                #     print(f"*** test case: {test_case.strip()}")
                # tracer_init(adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, test_case, rcnf.DIAG_var)

            case "Mountainwave":
                print("Mountainwave not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                # if IO_L:
                #     print(f"*** test case: {test_case.strip()}")
                # mountwave_init(adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, test_case, rcnf.DIAG_var)

            case "Gravitywave":
                print("Gravitywave not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                #gravwave_init(adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, rcnf.DIAG_var)

            case "Tomita2004":
                print("Tomita2004 not implemented yet")
                prc.prc_mpistop(std.io_l, std.fname_log)
                #tomita_init(adm.ADM_gall, adm.ADM_kall, adm.ADM_lall, rcnf.DIAG_var)

            case _:
                print("xxx [dycore_input] Invalid init_type. STOP.")
                raise SystemExit("PRC_MPIstop called")

        return


    def jbw_init(self, idim, jdim, kdim, lall, test_case, eps_geo2prs, nicamcore, rcnf, rdtype):

        DIAG_var = np.zeros((idim, jdim, kdim, lall, 6 + rcnf.TRC_vmax), dtype=rdtype)

        eta_limit = True
        psgm = False
        logout = True


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
            for i in range(idim):
                for j in range(jdim):

                    z_local[adm.ADM_kmin - 1] = GRD_vz[i, j, 2, l, GRD_ZH]
            for k in range(adm.ADM_kmin, adm.ADM_kmax + 2):  # +2 to match Fortran kmax+1
                z_local[k] = GRD_vz[n, k, l, GRD_Z]

            lat = GRD_LAT[n, l]
            lon = GRD_LON[n, l]

            signal = True

            # Iteration process
            for itr in range(1, itrmax + 1):

                if itr == 1:
                    eta[:, :] = 1.0e-7  # Jablonowski recommended initial value
                else:
                    eta_vert_coord_NW(kdim, itr, z_local, tmp, geo, eta_limit, eta, signal)

                steady_state(kdim, lat, eta, wix, wiy, tmp, geo)

                if not signal:
                    break  # Exit iteration loop

            # Check for convergence failure
            if itr > itrmax:
                print(f"ETA ITERATION ERROR: NOT CONVERGED at n={n}, l={l}")
                raise SystemExit("PRC_MPIstop called")

            # Pressure estimation
            if psgm:
                ps_estimation(kdim, lat, eta[:, 0], tmp, geo, wix, ps, nicamcore)
                geo2prs(kdim, ps, lat, tmp, geo, wix, prs, eps_geo2prs, nicamcore, logout)
            else:
                geo2prs(kdim, p0, lat, tmp, geo, wix, prs, eps_geo2prs, nicamcore, logout)

            logout = False

            # Convert velocity components
            conv_vxvyvz(kdim, lat, lon, wix, wiy, vx_local, vy_local, vz_local)

            # Store results in DIAG_var
            for k in range(1, kdim + 1):
                DIAG_var[n, k, l, 0] = prs[k]
                DIAG_var[n, k, l, 1] = tmp[k]
                DIAG_var[n, k, l, 2] = vx_local[k]
                DIAG_var[n, k, l, 3] = vy_local[k]
                DIAG_var[n, k, l, 4] = vz_local[k]



        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(" |            Vertical Coordinate used in JBW initialization              |")
                print(" |------------------------------------------------------------------------|")

                for k in range(1, kdim + 1):
                    print(f"   (k={k:3}) HGT: {z_local[k]:8.2f} [m]  "
                        f"PRS: {prs[k]:9.2f} [Pa]  "
                        f"GH: {geo[k] / g:8.2f} [m]  "
                        f"ETA: {eta[k, 0]:9.5f}", file=log_file)

                print(" |------------------------------------------------------------------------|")

        return DIAG_var
