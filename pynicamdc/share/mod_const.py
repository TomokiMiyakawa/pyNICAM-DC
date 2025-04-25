# Import necessary libraries
import numpy as np
import sys
import toml
from mod_stdio import std
class Const:

    _instance = None

    def __init__(self,rdtype):

        #< undefined value (int2, float32, float64))   
        self.CONST_UNDEF2   = np.int16(-32768)
        self.CONST_UNDEF4   = np.float32(-9.9999E+30)
        self.CONST_UNDEF8   = np.float64(-9.9999E+30)

        #< internal energy type      
        self.CONST_THERMODYN_TYPE = 'SIMPLE'    


        self.CONST_UNDEF    = rdtype(-9.9999E+30)

        # Public parameters & variables
        self.CONST_PI       = rdtype(np.pi) # pi 

        #self.CONST_D2R      = rdtype(None)  # degree to radian
        self.CONST_EPS      = rdtype(1.E-16) # small number
        self.CONST_EPS1     = rdtype(0.99999999999999) # small number
        self.CONST_HUGE     = rdtype(1.E+30) # huge number

        # adopted constants 
        self.CONST_RADIUS   = rdtype(6.37122E+6) # radius of the planet [m] 
        self.CONST_OHM      = rdtype(7.2920E-5) # angular velocity of the planet [1/s]  
        self.CONST_GRAV     = rdtype(9.80665) # standard acceleration of gravity [m/s2]

        # physical constants
        self.CONST_STB      = rdtype(5.670373E-8) # Stefan-Boltzman constant [W/m2/K4]
        self.CONST_KARMAN   = rdtype(0.4) # von Karman constant
        self.CONST_R        = rdtype(8.3144621) # universal gas constant [J/mol/K]

        # dry air constants
        self.CONST_Mdry     = rdtype(28.97) # mass weight (dry air)  [g/mol]   
        self.CONST_Rdry     = rdtype(287.04) # specific gas constant (dry air) [J/kg/K]
        self.CONST_CPdry    = rdtype(1004.64) # specific heat (dry air,constant pressure) [J/kg/K]
        #self.CONST_CVdry    = rdtype(None) # specific heat (dry air,constant volume)   [J/kg/K]
        self.CONST_LAPS     = rdtype(6.5E-3) # lapse rate of ISA  [K/m] 
        #self.CONST_LAPSdry  = rdtype(None) # dry adiabatic lapse rate  [K/m]    

        # water constants
        self.CONST_Mvap     = rdtype(18.02) # mass weight (water vapor) [g/mol]  
        self.CONST_Rvap     = rdtype(461.46) # specific gas constant (water vapor) [J/kg/K] 
        self.CONST_CPvap    = rdtype(1845.60) # specific heat (water vapor, constant pressure) [J/kg/K]  
        #self.CONST_CVvap    = rdtype(None) # specific heat (water vapor, constant volume)   [J/kg/K]
        self.CONST_CL       = rdtype(4218.0) # specific heat (liquid water) [J/kg/K]  
        self.CONST_CI       = rdtype(2006.0) # specific heat (ice) [J/kg/K]   

        #self.CONST_EPSvap   = rdtype(None) # Rdry / Rvap 
        #self.CONST_EPSTvap  = rdtype(None) # 1 / epsilon - 1 

        self.CONST_EMELT    = rdtype(3.34E+5) # heat of fusion [J/kg] 
        self.CONST_TMELT    = rdtype(273.15) # Freeze point of water 
        self.CONST_TFRZS    = rdtype(271.35) # Freeze point of sea  

        #self.CONST_LHV      = rdtype(None) # latent heat of vaporizaion for use 
        #self.CONST_LHS      = rdtype(None) # latent heat of sublimation for use
        #self.CONST_LHF      = rdtype(None) # latent heat of fusion      for use
        self.CONST_LHV0     = rdtype(2.5008E+6) # latent heat of vaporizaion at 0C [J/kg] 
        #self.CONST_LHV00    = rdtype(None) # latent heat of vaporizaion at 0K [J/kg]  
        self.CONST_LHS0     = rdtype(2.8342E+6) # latent heat of sublimation at 0C [J/kg]
        #self.CONST_LHS00    = rdtype(None) # latent heat of sublimation at 0K [J/kg]  
        #self.CONST_LHF0     = rdtype(None) # latent heat of fusion      at 0C [J/kg] 
        #self.CONST_LHF00    = rdtype(None) # latent heat of fusion      at 0K [J/kg]  
        self.CONST_PSAT0    = rdtype(610.7) # saturate pressure of water vapor at 0C [Pa] 
        self.CONST_DWATR    = rdtype(1000.0) # density of water [kg/m3]
        self.CONST_DICE     = rdtype(916.8) # density of ice   [kg/m3]  

        # standards
        #self.CONST_SOUND    = rdtype(None) # speed of sound (dry air at 0C) [m/s]  
        self.CONST_Pstd     = rdtype(101325.0) # standard pressure [Pa] 
        self.CONST_PRE00    = rdtype(100000.0) # pressure reference [Pa] 
        self.CONST_Tstd     = rdtype(288.15) # standard temperature (15C) [K]
        self.CONST_TEM00    = rdtype(273.15) # temperature reference (0C) [K] 
        self.CONST_PPM      = rdtype(1.E-6) # parts per million  

    def CONST_setup(self, rdtype, fname_in=None):
        # Setup

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[cnst]", file=log_file)
        
        if fname_in is None:
            with open(std.fname_log, 'a') as log_file:
                if std.io_l: print("*** input toml file is not specified. use default.", file=log_file)

        else:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(f"*** input toml file is ", fname_in, file=log_file)

            with open(fname_in, 'r') as  file:
                cnfs = toml.load(file)

                if 'cnstparam' not in cnfs:

                    if std.io_l:
                        with open(std.fname_log, 'a') as log_file: 
                            print("*** cnstparam not specified in toml file. use default.", file=log_file)
                    
                else:
                    if 'earth_radius' in cnfs['cnstparam']:
                        earth_radius = cnfs['cnstparam']['earth_radius']  
                        self.CONST_RADIUS = rdtype(earth_radius)                 
                    if 'earth_angvel' in cnfs['cnstparam']:
                        earth_angvel = cnfs['cnstparam']['earth_angvel']
                        self.CONST_OHM = rdtype(earth_angvel)
                    if 'small_planet_factor' in cnfs['cnstparam']:
                        small_planet_factor = cnfs['cnstparam']['small_planet_factor']          
                        print("small_planet not implemented yet")   
                    if 'earth_gravity' in cnfs['cnstparam']:
                        earth_gravity = cnfs['cnstparam']['earth_gravity']
                        self.CONST_GRAV = rdtype(earth_gravity)                
                    if 'gas_cnst' in cnfs['cnstparam']:
                        gas_cnst = cnfs['cnstparam']['gas_cnst']
                        self.CONST_Rdry = rdtype(gas_cnst)
                    if 'gas_cnst_vap' in cnfs['cnstparam']:
                        gas_cnst_vap = cnfs['cnstparam']['gas_cnst_vap']
                        self.CONST_Rvap = rdtype(gas_cnst_vap)
                    if 'specific_heat_pre' in cnfs['cnstparam']:
                        specific_heat_pre = cnfs['cnstparam']['specific_heat_pre']
                        self.CONST_CPdry = rdtype(specific_heat_pre)
                    if 'specific_heat_pre_vap' in cnfs['cnstparam']:
                        specific_heat_pre_vap = cnfs['cnstparam']['specific_heat_pre_vap']
                        self.CONST_CPvap = rdtype(specific_heat_pre_vap)
                    if 'latent_heat_vap' in cnfs['cnstparam']:
                        latent_heat_vap = cnfs['cnstparam']['latent_heat_vap']
                        self.CONST_LHV = rdtype(latent_heat_vap)
                    if 'latent_heat_sub' in cnfs['cnstparam']:
                        latent_heat_sub = cnfs['cnstparam']['latent_heat_sub']
                        self.CONST_LHS = rdtype(latent_heat_sub)
                    if 'thermodyn_type' in cnfs['cnstparam']:
                        thermodyn_type = cnfs['cnstparam']['thermodyn_type']
                        self.CONST_THERMODYN_TYPE = thermodyn_type

        #if io_nml: print(cnfs['constparam'])

        # Constants
        self.CONST_PI = rdtype(4.0 * np.arctan(1.0))
        self.CONST_D2R = rdtype(self.CONST_PI / 180.0)
        self.CONST_EPS = np.finfo(rdtype).eps
        self.CONST_EPS1 = rdtype(1.0) - np.finfo(rdtype).eps
        self.CONST_HUGE = np.finfo(rdtype).max

        self.CONST_CVdry = self.CONST_CPdry - self.CONST_Rdry
        self.CONST_LAPSdry = self.CONST_GRAV / self.CONST_CPdry

        self.CONST_CVvap = self.CONST_CPvap - self.CONST_Rvap
        self.CONST_EPSvap = self.CONST_Rdry / self.CONST_Rvap
        self.CONST_EPSTvap = rdtype(1.0 / self.CONST_EPSvap - 1.0)

        self.CONST_LHF0 = self.CONST_LHS0 - self.CONST_LHV0

        self.CONST_LHV00 = self.CONST_LHV0 - (self.CONST_CPvap - self.CONST_CL) * self.CONST_TEM00
        self.CONST_LHS00 = self.CONST_LHS0 - (self.CONST_CPvap - self.CONST_CI) * self.CONST_TEM00
        self.CONST_LHF00 = self.CONST_LHF0 - (self.CONST_CL - self.CONST_CI) * self.CONST_TEM00

        if self.CONST_THERMODYN_TYPE == 'EXACT':
            self.CONST_LHV = self.CONST_LHV00
            self.CONST_LHS = self.CONST_LHS00
            self.CONST_LHF = self.CONST_LHF00
        elif self.CONST_THERMODYN_TYPE in ['SIMPLE', 'SIMPLE2']:
            self.CONST_LHV = self.CONST_LHV0
            self.CONST_LHS = self.CONST_LHS0
            self.CONST_LHF = self.CONST_LHF0
        else:
            print(f'Not appropriate ATMOS_THERMODYN_ENERGY_TYPE. Check! {self.CONST_THERMODYN_TYPE}')
            # Raise an exception or exit the program
            # raise Exception('Invalid thermodynamic type')
            sys.exit(1)

        self.CONST_SOUND = np.sqrt(self.CONST_CPdry * self.CONST_Rdry / (self.CONST_CPdry - self.CONST_Rdry) * self.CONST_TEM00)


# Example usage
#instance = YourClassName()
#instance.calculate_constants()

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:

                print(file=log_file)
                print('*** Precision ***', file=log_file)
                print('*** kind (floating point value) =', np.finfo(rdtype).dtype, file=log_file)
                print('*** precision(floating point value) =', np.finfo(rdtype).precision, file=log_file)
                print('*** range (floating point value) =', (np.finfo(rdtype).min, np.finfo(rdtype).max), file=log_file)
                print(file=log_file)
                print('*** List of constants ***', file=log_file)
                print(f'*** PI : PI = {self.CONST_PI}', file=log_file)
                print(f'*** Small number : EPS = {self.CONST_EPS}', file=log_file)
                print(f'*** Small number (1-EPS) : EPS1 = {self.CONST_EPS1}', file=log_file)
                print(f'*** Huge number : HUGE = {self.CONST_HUGE}', file=log_file)
                print(f'*** undefined number (INT2) : UNDEF2 = {self.CONST_UNDEF2}', file=log_file)
                print(f'*** undefined number (REAL, general use) : UNDEF = {self.CONST_UNDEF}', file=log_file)
                print(f'*** undefined number (REAL4) : UNDEF4 = {self.CONST_UNDEF4}', file=log_file)
                print(f'*** undefined number (REAL8) : UNDEF8 = {self.CONST_UNDEF8}', file=log_file)

                print(f'*** radius of the planet [m] : RADIUS = {self.CONST_RADIUS}', file=log_file)
                print(f'*** angular velocity of the planet [1/s] : OHM = {self.CONST_OHM}', file=log_file)
                print(f'*** standard acceleration of gravity [m/s2] : GRAV = {self.CONST_GRAV}', file=log_file)

                print(f'*** Stefan-Boltzman constant [W/m2/K4] : STB = {self.CONST_STB}', file=log_file)
                print(f'*** von Karman constant : KARMAN = {self.CONST_KARMAN}', file=log_file)
                print(f'*** universal gas constant [J/mol/K] : R = {self.CONST_R}', file=log_file)

                print(f'*** mass weight (dry air) [g/mol] : Mdry = {self.CONST_Mdry}', file=log_file)
                print(f'*** specific gas constant (dry air) [J/kg/K] : Rdry = {self.CONST_Rdry}', file=log_file)
                print(f'*** specific heat (dry air, const. pressure) [J/kg/K] : CPdry = {self.CONST_CPdry}', file=log_file)
                print(f'*** specific heat (dry air, const. volume) [J/kg/K] : Cvdry = {self.CONST_CVdry}', file=log_file)
                print(f'*** lapse rate of ISA [K/m] : LAPS = {self.CONST_LAPS}', file=log_file)
                print(f'*** dry adiabatic lapse rate [K/m] : LAPSdry = {self.CONST_LAPSdry}', file=log_file)

                print(f'*** mass weight (water vapor) [g/mol] : Rvap = {self.CONST_Rvap}', file=log_file)
                print(f'*** specific gas constant (water vapor) [J/kg/K] : Rvap = {self.CONST_Rvap}', file=log_file)    
                print(f'*** specific heat (vapor, const. pressure) [J/kg/K] : CPvap = {self.CONST_CPvap}', file=log_file)
                print(f'*** specific heat (vapor, const. volume) [J/kg/K] : CVvap = {self.CONST_CVvap}', file=log_file)
                print(f'*** specific heat (liquid water) [J/kg/K] : CL = {self.CONST_CL}', file=log_file)
                print(f'*** specific heat (ice) [J/kg/K] : CI = {self.CONST_CI}', file=log_file)
                print(f'*** Rdry / Rvap : EPSvap = {self.CONST_EPSvap}', file=log_file)
                print(f'*** 1 / EPSvap - 1 : EPSTvap = {self.CONST_EPSTvap}', file=log_file)

                print(f'*** latent heat of vaporization at 0C [J/kg] : LHV0 = {self.CONST_LHV0}', file=log_file)
                print(f'*** latent heat of sublimation at 0C [J/kg] : LHS0 = {self.CONST_LHS0}', file=log_file)
                print(f'*** latent heat of fusion at 0C [J/kg] : LHF0 = {self.CONST_LHF0}', file=log_file)
                print(f'*** latent heat of vaporization at 0K [J/kg] : LHV00 = {self.CONST_LHV00}', file=log_file)
                print(f'*** latent heat of sublimation at 0K [J/kg] : LHS00 = {self.CONST_LHS00}', file=log_file)
                print(f'*** latent heat of fusion at 0K [J/kg] : LHF00 = {self.CONST_LHF00}', file=log_file)
                print(f'*** Thermodynamics calculation type : {self.CONST_THERMODYN_TYPE}', file=log_file)
                print(f'*** latent heat of vaporization (used) [J/kg] : LHV = {self.CONST_LHV}', file=log_file)
                print(f'*** latent heat of sublimation (used) [J/kg] : LHS = {self.CONST_LHS}', file=log_file)
                print(f'*** latent heat of fusion (used) [J/kg] : LHF = {self.CONST_LHF}', file=log_file)                
                print(f'*** saturate pressure of water vapor at 0C [Pa] : PSAT0 = {self.CONST_PSAT0}', file=log_file)
                print(f'*** density of water [kg/m3] : DWATR = {self.CONST_DWATR}', file=log_file)
                print(f'*** density of ice [kg/m3] : DICE = {self.CONST_DICE}', file=log_file)

                print(f'*** speed of sound (dry air at 0C) [m/s] : SOUND = {self.CONST_SOUND}', file=log_file)
                print(f'*** standard pressure [Pa] : Pstd = {self.CONST_Pstd}', file=log_file)
                print(f'*** pressure reference [Pa] : PRE00 = {self.CONST_PRE00}', file=log_file)
                print(f'*** standard temperature (15C) [K] : Tstd = {self.CONST_Tstd}', file=log_file)
                print(f'*** temperature reference (0C) [K] : TEM00 = {self.CONST_TEM00}', file=log_file)

## Example usage
#instance = YourClassName()
## Ensure that all the required constants are set in YourClassName
#instance.log_constants(IO_L=True, fname_log='log_file.txt')


