import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf

class Frc:
    
    _instance = None

    nmax_TEND     = 7
    nmax_PROG     = 6
    nmax_v_mean_c = 5

    I_RHOG     = 0  # Density x G^1/2 x gamma^2
    I_RHOGVX   = 1  # Density x G^1/2 x gamma^2 x Horizontal velocity (X-direction)
    I_RHOGVY   = 2  # Density x G^1/2 x gamma^2 x Horizontal velocity (Y-direction)
    I_RHOGVZ   = 3  # Density x G^1/2 x gamma^2 x Horizontal velocity (Z-direction)
    I_RHOGW    = 4  # Density x G^1/2 x gamma^2 x Vertical   velocity
    I_RHOGE    = 5  # Density x G^1/2 x gamma^2 x Internal Energy
    I_RHOGETOT = 6  # Density x G^1/2 x gamma^2 x Total Energy

    # Logical flags
    NEGATIVE_FIXER  = False
    UPDATE_TOT_DENS = True
    
    def __init__(self):
        pass

    def forcing_setup(self, fname_in, rcnf, rdtype):

        self.time = 0.0

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[forcing]/Category[nhm]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'forcing_param' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** forcing_param not found in toml file! Use default.", file=log_file)
                #prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['forcing_param']
            #self.GRD_grid_type = cnfs['GRD_grid_type']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print(f"+++ Artificial forcing type: {rcnf.AF_TYPE.strip()}", file=log_file)


        if rcnf.AF_TYPE == 'NONE':
            # do nothing
            pass

        elif rcnf.AF_TYPE == 'HELD-SUAREZ':
            print("sorry, HELD-SUARZ is not implemented yet.")
            #self.AF_heldsuarez_init(moist_case=False)
            prc.prc_mpistop(std.io_l, std.fname_log)

        elif rcnf.AF_TYPE == 'DCMIP':
            print("sorry, DCMIP is not implemented yet.")
            #self.AF_dcmip_init()
            prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            print("xxx unsupported forcing type! STOP.")
            prc.prc_mpistop(std.io_l, std.fname_log)

        return
    

    def forcing_update(self,
                       PROG, PROG_pl,
                       cnst, rcnf, grd, tim, trcadv, rdtype,
                       ):
        
        prf.PROF_rapstart('__Forcing',1)

        vx    = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        vx_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        vy    = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        vy_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        vz    = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        vz_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        w     = np.full(adm.ADM_shape,    cnst.CONST_UNDEF, dtype=rdtype)
        w_pl  = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)

        #--- update velocity
        self.time = self.time + tim.TIME_dtl

        gall_1d= adm.ADM_gall_1d
        kall = adm.ADM_kall
        lall = adm.ADM_lall
        k0 = adm.ADM_K0

        if rcnf.DCTEST_type == 'Traceradvection' and rcnf.DCTEST_case == '1-1':

            trcadv.test11_velocity(self.time,
                                    grd.GRD_LON,
                                    grd.GRD_LAT,
                                    grd.GRD_vz[:,:,:,:,grd.GRD_Z],
                                    grd.GRD_vz[:,:,:,:,grd.GRD_ZH],
                                    vx,
                                    vy,
                                    vz,
                                    w, 
                                    rdtype,
            )

            if adm.ADM_have_pl:      

                trcadv.test11_velocity(self.time,
                                        grd.GRD_LON_pl,
                                        grd.GRD_LAT_pl,
                                        grd.GRD_vz_pl[:,:,:,grd.GRD_Z],
                                        grd.GRD_vz_pl[:,:,:,grd.GRD_ZH],
                                        vx_pl,
                                        vy_pl,
                                        vz_pl,
                                        w_pl, 
                                        rdtype,
                )    

        elif rcnf.DCTEST_type == 'Traceradvection' and rcnf.DCTEST_case == '1-2':
            print("this test case is not implemented yet.")
            prc.prc_mpistop(std.io_l, std.fname_log)

        PROG[:, :, :, :, self.I_RHOGVX] = vx * PROG[:, :, :, :, self.I_RHOG]
        PROG[:, :, :, :, self.I_RHOGVY] = vy * PROG[:, :, :, :, self.I_RHOG]
        PROG[:, :, :, :, self.I_RHOGVZ] = vz * PROG[:, :, :, :, self.I_RHOG]
        PROG[:, :, :, :, self.I_RHOGW ] = w  * PROG[:, :, :, :, self.I_RHOG]

        if adm.ADM_have_pl:
            PROG_pl[:, :, :, self.I_RHOGVX] = vx_pl * PROG_pl[:, :, :, self.I_RHOG]
            PROG_pl[:, :, :, self.I_RHOGVY] = vy_pl * PROG_pl[:, :, :, self.I_RHOG]
            PROG_pl[:, :, :, self.I_RHOGVZ] = vz_pl * PROG_pl[:, :, :, self.I_RHOG]
            PROG_pl[:, :, :, self.I_RHOGW ] = w_pl  * PROG_pl[:, :, :, self.I_RHOG]

        prf.PROF_rapend  ('__Forcing',1)

        return
    
frc = Frc()