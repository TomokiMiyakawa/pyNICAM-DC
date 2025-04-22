import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf

class Idt:
    
    _instance = None
    
    def __init__(self):
        pass

    def IDEAL_topo(self, fname_in, lat, lon, Zsfc, cnst, rdtype):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[ideal topo]/Category[common share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'idealtopoparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** idealtopoparam not found in toml file! STOP.", file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['idealtopoparam']
            topo_type = cnfs['topo_type']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        if topo_type == 'Schar_Moderate':
            print('Schar_Moderate not implemented yet')
            prc.prc_mpistop(std.io_l, std.fname_log)            

        elif topo_type == 'Schar_Steep':
            print('Schar_Steep not implemented yet')
            prc.prc_mpistop(std.io_l, std.fname_log)            

        elif topo_type == 'JW':
            #np.seterr(under='ignore')
            Zsfc[:,:,:,:] = self.IDEAL_topo_JW(lat[:,:,:,:],cnst, rdtype)
            #np.seterr(under='raise')
        else:
            print('xxx [IDEAL_topo] Not appropriate topo_type. STOP.')
            prc.prc_mpistop(std.io_l, std.fname_log)

        return Zsfc[:,:,:,:]
    

    def IDEAL_topo_JW(self, lat, cnst, rdtype):
        #mountain for JW06 testcase

        ETA0 = rdtype(0.252)  # Value of eta at a reference level
        ETAs = rdtype(1.0)  # Value of eta at the surface
        u0 = rdtype(35.0)  # Maximum amplitude of the zonal wind

        #K0 = adm.ADM_KNONE 

        ETAv = (ETAs - ETA0) * (cnst.CONST_PI / rdtype(2.0))
        u0cos32ETAv = u0 * np.cos(ETAv) ** 1.5  #(rdtype(3.0) / rdtype(2.0))

        Zsfc = np.zeros_like(lat)

        for l in range(adm.ADM_lall):
            for j in range(adm.ADM_gall_1d):
                for i in range(adm.ADM_gall_1d):
                    
                    PHI = lat[i,j,0,l]
                    # with open (std.fname_log, 'a') as log_file:
                    #     print("PHI: ", i, j, l, PHI, file=log_file)
                    f1 = -rdtype(2.0) * np.sin(PHI) ** 6 * (np.cos(PHI) ** 2 + rdtype(1.0) / rdtype(3.0)) + rdtype(10.0) / rdtype(63.0)
                    f2 = (rdtype(8.0) / rdtype(5.0)) * np.cos(PHI) ** 3 * (np.sin(PHI) ** 2 + rdtype(2.0) / rdtype(3.0)) - cnst.CONST_PI / rdtype(4.0)
                    Zsfc[i,j,0,l] = u0cos32ETAv * (u0cos32ETAv * f1 + cnst.CONST_RADIUS * cnst.CONST_OHM * f2) / cnst.CONST_GRAV  

                    # with open (std.fname_log, 'a') as log_file:
                    #     print("PHI: ", i, j, l, PHI, file=log_file)
                    #     print("Zsfc: ", Zsfc[i,j,0,l], file=log_file)

                    # if i==1 and j ==17 and l==2:
                    #     with open (std.fname_log, 'a') as log_file:
                    #         print("Zsfc: ", Zsfc[i,j,0,l], file=log_file)
                    #         print("PHI: ", PHI, file=log_file)
                    #         print("f1, f2, u0cos32ETAv", f1, f2,u0cos32ETAv, file=log_file)
                    #         print("cnsts pi omega r g : ", cnst.CONST_PI, cnst.CONST_OHM, cnst.CONST_RADIUS, cnst.CONST_GRAV, file=log_file)

        ### CHECK!!
        return Zsfc[:,:,:,:]
    
