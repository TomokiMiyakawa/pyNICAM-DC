import toml
import numpy as np
#from mpi4py import MPI
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
#from mod_prof import prf

class Idt:
    
    
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
            Zsfc[:,:,:,:] = self.IDEAL_topo_Schar_Moderate(lat[:,:,:,:], lon[:,:,:,:], cnfs, cnst, rdtype)

        elif topo_type == 'Schar_Steep':
            Zsfc[:,:,:,:] = self.IDEAL_topo_Schar_Steep(lat[:,:,:,:], lon[:,:,:,:], cnfs, cnst, rdtype)

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

        # Vectorized over the whole (i,j,k0,l) array: Zsfc is a pure elementwise
        # function of latitude, so the former triple i/j/l loop is replaced by
        # numpy array ops (bit-identical -- same per-element operation order).
        PHI = lat
        f1 = -rdtype(2.0) * np.sin(PHI) ** 6 * (np.cos(PHI) ** 2 + rdtype(1.0) / rdtype(3.0)) + rdtype(10.0) / rdtype(63.0)
        f2 = (rdtype(8.0) / rdtype(5.0)) * np.cos(PHI) ** 3 * (np.sin(PHI) ** 2 + rdtype(2.0) / rdtype(3.0)) - cnst.CONST_PI / rdtype(4.0)
        Zsfc = u0cos32ETAv * (u0cos32ETAv * f1 + cnst.CONST_RADIUS * cnst.CONST_OHM * f2) / cnst.CONST_GRAV

        return Zsfc


    def IDEAL_topo_Schar_Moderate(self, lat, lon, cnfs, cnst, rdtype):
        # Moderately-steep Schar-like circular mountain (DCMIP2012 eq.(48)).
        # Ported from nicamdc IDEAL_topo_Schar_Moderate; params from the
        # [idealtopoparam] toml table (defaults = nicamdc namelist defaults).
        center_lon = rdtype(cnfs.get('center_lon', 270.0))  # [deg]
        center_lat = rdtype(cnfs.get('center_lat',   0.0))  # [deg]
        H0         = rdtype(cnfs.get('H0',        2000.0))   # [m]
        Rm_deg     = rdtype(cnfs.get('Rm_deg',     135.0))   # mountain radius     [deg]
        QSIm_deg   = rdtype(cnfs.get('QSIm_deg',   11.25))   # mountain wavelength [deg]

        PI  = cnst.CONST_PI
        D2R = cnst.CONST_D2R
        LAMBDAm = center_lon * D2R
        PHIm    = center_lat * D2R
        Rm      = Rm_deg     * D2R
        QSIm    = QSIm_deg   * D2R
        sinPHIm = np.sin(PHIm)
        cosPHIm = np.cos(PHIm)

        LAMBDA = lon
        PHI    = lat
        distance = np.arccos(sinPHIm * np.sin(PHI)
                             + cosPHIm * np.cos(PHI) * np.cos(LAMBDA - LAMBDAm))  # great-circle angle [rad]
        # mask = 0 where distance > Rm (Fortran: 0.5 - sign(0.5, distance-Rm))
        mask = rdtype(0.5) - np.copysign(rdtype(0.5), distance - Rm)
        Zsfc = ( H0 / rdtype(2.0)
                 * (rdtype(1.0) + np.cos(PI * distance / Rm))
                 * np.cos(PI * distance / QSIm) ** 2
                 * mask )
        return Zsfc


    def IDEAL_topo_Schar_Steep(self, lat, lon, cnfs, cnst, rdtype):
        # Steep Schar-like circular mountain (DCMIP2012 eq.(76)).
        # Ported from nicamdc IDEAL_topo_Schar_Steep; params from [idealtopoparam].
        center_lon = rdtype(cnfs.get('center_lon',  45.0))  # [deg]
        center_lat = rdtype(cnfs.get('center_lat',   0.0))  # [deg]
        H0         = rdtype(cnfs.get('H0',         250.0))   # [m]
        d          = rdtype(cnfs.get('d',         5000.0))   # half-width  [m]
        QSI        = rdtype(cnfs.get('QSI',       4000.0))   # wavelength  [m]

        PI     = cnst.CONST_PI
        D2R    = cnst.CONST_D2R
        RADIUS = cnst.CONST_RADIUS
        LAMBDAc = center_lon * D2R
        PHIc    = center_lat * D2R
        sinPHIc = np.sin(PHIc)
        cosPHIc = np.cos(PHIc)

        LAMBDA = lon
        PHI    = lat
        distance = RADIUS * np.arccos(sinPHIc * np.sin(PHI)
                                      + cosPHIc * np.cos(PHI) * np.cos(LAMBDA - LAMBDAc))  # [m]
        Zsfc = ( H0
                 * np.exp(-(distance * distance) / (d * d))
                 * np.cos(PI * distance / QSI) ** 2 )
        return Zsfc

