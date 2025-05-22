import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf


class Trcadv:
    
    _instance = None


    def __init__(self, rdtype):
        self.a  = rdtype(6371220.0)            # Earth's Radius [m]
        self.Rd = rdtype(287.0)                # Ideal gas const dry air [J/kg*K]
        self.g  = rdtype(9.80616)              # Gravity [m/s2]
        self.cp = rdtype(1004.5)               # Specific heat capacity [J/kg*K]
        self.pi = rdtype(3.141592653589793238) # pi
        return
    

    def test11_velocity(self,
                        time,     #[IN]
                        lon,      #[IN]
                        lat,      #[IN]
                        zf,       #[IN]
                        zh,       #[IN]
                        vx,       #[OUT]
                        vy,       #[OUT]
                        vz,       #[OUT]
                        w,        #[OUT]
                        rdtype,
                        ):
        
        tau     = rdtype(12.0) * rdtype(86400.0)        # period of motion 12 days
        u0      = rdtype(2.0) * self.pi * self.a / tau  # 2 pi a / 12 days
        k0      = rdtype(10.0) * self.a     / tau       # Velocity Magnitude
        omega0  = rdtype(23000.0) * self.pi / tau       # Velocity Magnitude
        T0      = rdtype(300.0)                         # temperature
        H       = self.Rd * T0 / self.g                 # scale height
        p0      = rdtype(1000.E2)                       # reference pressure (Pa)
        

        #lon = np.expand_dims(lon, axis=-2)   # inserting dummy k axis for broadcast
        #lat = np.expand_dims(lat, axis=-2)

        dlon = rdtype(2.0) * self.pi * time / tau
        lonp = lon - dlon
        bs   = rdtype(0.2)


        # --- Full Level (using zf) ---
        p = p0 * np.exp(-zf / H)                     # array  
        ptop = p0 * np.exp(rdtype(-12000.0) / H)     # scalar

        s = (rdtype(1.0)
            + np.exp((ptop - p0) / (bs * ptop))
            - np.exp((p - p0) / (bs * ptop))
            - np.exp((ptop - p) / (bs * ptop)))      # array       

        ud = (omega0 * self.a) / (bs * ptop) * np.cos(lonp) * np.cos(lat)**2 * np.cos(dlon) * (   # array
            -np.exp((p - p0) / (bs * ptop)) + np.exp((ptop - p) / (bs * ptop))  
        )

        u = k0 * np.sin(rdtype(2.0) * lat) * np.sin(lonp)**2 * np.cos(rdtype(0.5) * dlon) + u0 * np.cos(lat) + ud
        v = k0 * np.sin(rdtype(2.0) * lonp) * np.cos(lat) * np.cos(rdtype(0.5) * dlon)    # arrray

        east = self.Sp_Unit_East(lon)         # This returns an array with additional 3-element axis
        nrth = self.Sp_Unit_North(lon, lat)   # This returns an array with additional 3-element axis


        #print ("a", k0.shape, omega0.shape, self.a.shape, bs.shape, ptop.shape, p.shape, p0.shape, lat.shape, dlon.shape, lonp.shape) #$$$
        #print ("b", lonp.shape, lat.shape, dlon.shape, u0.shape, ud.shape) #$$$
        #print ("c", east.shape, u.shape, nrth.shape, v.shape) #$$$
        vx[:] = east[...,0] * u + nrth[...,0] * v   #free size array
        vy[:] = east[...,1] * u + nrth[...,1] * v   #free size array
        vz[:] = east[...,2] * u + nrth[...,2] * v   #free size array

        # --- Half Level (using zh) ---
        p = p0 * np.exp(-zh / H)                     #array
        ptop = p0 * np.exp(rdtype(-12000.0) / H)     #scalar

        s = (rdtype(1.0)                             #array
            + np.exp((ptop - p0) / (bs * ptop))
            - np.exp((p - p0) / (bs * ptop))
            - np.exp((ptop - p) / (bs * ptop)))

        w[:] = -(self.Rd * T0) / (self.g * p) * omega0 * np.sin(lonp) * np.cos(lat) * np.cos(dlon) * s   # free size array

        # if lon.shape[0] == 18:
        #     with open(std.fname_log, 'a') as log_file:
        #         print("vx: ", vx[6,5,10,0], file=log_file)
        #         print("vy: ", vy[6,5,10,0], file=log_file)
        #         print("vz: ", vz[6,5,10,0], file=log_file)
        #         print("w: ",   w[6,5,10,0], file=log_file)

        return
        
    def Sp_Unit_East(self, lon):
        """Calculate the unit vector in the east direction."""
        # Assuming lon is a scalar or a 1D array
        sp=lon.shape
        unit_east = np.zeros((sp + (3,)), dtype=lon.dtype)
        unit_east[..., 0] = -np.sin(lon)
        unit_east[..., 1] = np.cos(lon)
        unit_east[..., 2] = 0.0

        return unit_east
    
    def Sp_Unit_North(self, lon, lat):
        """Calculate the unit vector in the north direction."""
        # Assuming lon and lat are scalars or 1D arrays
        sp=lon.shape
        unit_north = np.zeros((sp + (3,)), dtype=lon.dtype)
        unit_north[..., 0] = -np.sin(lat) * np.cos(lon)
        unit_north[..., 1] = -np.sin(lat) * np.sin(lon)
        unit_north[..., 2] = np.cos(lat)

        return unit_north