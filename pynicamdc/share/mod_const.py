# Import necessary libraries
import numpy as np
import sys
import toml
from mod_stdio import std
class Const:

    _instance = None

    def __init__(self,single):

        #< undefined value (int2, float32, float64))   
        self.CONST_UNDEF2   = np.int16(-32768)
        self.CONST_UNDEF4   = np.float32(-9.9999E30)
        self.CONST_UNDEF8   = np.float64(-9.9999E30)

        #< internal energy type      
        self.CONST_THERMODYN_TYPE = 'SIMPLE'    

        if single:
            # UNDEF
            self.CONST_UNDEF    = self.CONST_UNDEF4

            # Public parameters & variables
            self.CONST_PI       = np.float32(np.pi) # pi 

            #self.CONST_D2R      = np.float32(None)  # degree to radian
            self.CONST_EPS      = np.float32(1.E-16) # small number
            self.CONST_EPS1     = np.float32(0.99999999999999) # small number
            self.CONST_HUGE     = np.float32(1.E+30) # huge number

            # adopted constants 
            self.CONST_RADIUS   = np.float32(6.37122E+6) # radius of the planet [m] 
            self.CONST_OHM      = np.float32(7.2920E-5) # angular velocity of the planet [1/s]  
            self.CONST_GRAV     = np.float32(9.80665) # standard acceleration of gravity [m/s2]

            # physical constants
            self.CONST_STB      = np.float32(5.670373E-8) # Stefan-Boltzman constant [W/m2/K4]
            self.CONST_KARMAN   = np.float32(0.4) # von Karman constant
            self.CONST_R        = np.float32(8.3144621) # universal gas constant [J/mol/K]

            # dry air constants
            self.CONST_Mdry     = np.float32(28.97) # mass weight (dry air)  [g/mol]   
            self.CONST_Rdry     = np.float32(287.04) # specific gas constant (dry air) [J/kg/K]
            self.CONST_CPdry    = np.float32(1004.64) # specific heat (dry air,constant pressure) [J/kg/K]
            #self.CONST_CVdry    = np.float32(None) # specific heat (dry air,constant volume)   [J/kg/K]
            self.CONST_LAPS     = np.float32(6.5E-3) # lapse rate of ISA  [K/m] 
            #self.CONST_LAPSdry  = np.float32(None) # dry adiabatic lapse rate  [K/m]    

            # water constants
            self.CONST_Mvap     = np.float32(18.02) # mass weight (water vapor) [g/mol]  
            self.CONST_Rvap     = np.float32(461.46) # specific gas constant (water vapor) [J/kg/K] 
            self.CONST_CPvap    = np.float32(1845.60) # specific heat (water vapor, constant pressure) [J/kg/K]  
            #self.CONST_CVvap    = np.float32(None) # specific heat (water vapor, constant volume)   [J/kg/K]
            self.CONST_CL       = np.float32(4218.0) # specific heat (liquid water) [J/kg/K]  
            self.CONST_CI       = np.float32(2006.0) # specific heat (ice) [J/kg/K]   

            #self.CONST_EPSvap   = np.float32(None) # Rdry / Rvap 
            #self.CONST_EPSTvap  = np.float32(None) # 1 / epsilon - 1 

            self.CONST_EMELT    = np.float32(3.34E+5) # heat of fusion [J/kg] 
            self.CONST_TMELT    = np.float32(273.15) # Freeze point of water 
            self.CONST_TFRZS    = np.float32(271.35) # Freeze point of sea  

            #self.CONST_LHV      = np.float32(None) # latent heat of vaporizaion for use 
            #self.CONST_LHS      = np.float32(None) # latent heat of sublimation for use
            #self.CONST_LHF      = np.float32(None) # latent heat of fusion      for use
            self.CONST_LHV0     = np.float32(2.5008E+6) # latent heat of vaporizaion at 0C [J/kg] 
            #self.CONST_LHV00    = np.float32(None) # latent heat of vaporizaion at 0K [J/kg]  
            self.CONST_LHS0     = np.float32(2.8342E+6) # latent heat of sublimation at 0C [J/kg]
            #self.CONST_LHS00    = np.float32(None) # latent heat of sublimation at 0K [J/kg]  
            #self.CONST_LHF0     = np.float32(None) # latent heat of fusion      at 0C [J/kg] 
            #self.CONST_LHF00    = np.float32(None) # latent heat of fusion      at 0K [J/kg]  
            self.CONST_PSAT0    = np.float32(610.7) # saturate pressure of water vapor at 0C [Pa] 
            self.CONST_DWATR    = np.float32(1000.0) # density of water [kg/m3]
            self.CONST_DICE     = np.float32(916.8) # density of ice   [kg/m3]  

            # standards
            #self.CONST_SOUND    = np.float32(None) # speed of sound (dry air at 0C) [m/s]  
            self.CONST_Pstd     = np.float32(101325.0) # standard pressure [Pa] 
            self.CONST_PRE00    = np.float32(100000.0) # pressure reference [Pa] 
            self.CONST_Tstd     = np.float32(288.15) # standard temperature (15C) [K]
            self.CONST_TEM00    = np.float32(273.15) # temperature reference (0C) [K] 
            self.CONST_PPM      = np.float32(1.E-6) # parts per million  

        else:
            # UNDEF
            #self.CONST_UNDEF    = self.CONST_UNDEF8
            self.CONST_UNDEF    = np.nan
            # Public parameters & variables
            self.CONST_PI       = np.float64(np.pi) # pi 

            #self.CONST_D2R      = np.float64(None)  # degree to radian
            self.CONST_EPS      = np.float64(1.E-16) # small number
            self.CONST_EPS1     = np.float64(0.99999999999999) # small number
            self.CONST_HUGE     = np.float64(1.E+30) # huge number

            # adopted constants 
            self.CONST_RADIUS   = np.float64(6.37122E+6) # radius of the planet [m] 
            self.CONST_OHM      = np.float64(7.2920E-5) # angular velocity of the planet [1/s]  
            self.CONST_GRAV     = np.float64(9.80665) # standard acceleration of gravity [m/s2]

            # physical constants
            self.CONST_STB      = np.float64(5.670373E-8) # Stefan-Boltzman constant [W/m2/K4]
            self.CONST_KARMAN   = np.float64(0.4) # von Karman constant
            self.CONST_R        = np.float64(8.3144621) # universal gas constant [J/mol/K]

            # dry air constants
            self.CONST_Mdry     = np.float64(28.97) # mass weight (dry air)  [g/mol]   
            self.CONST_Rdry     = np.float64(287.04) # specific gas constant (dry air) [J/kg/K]
            self.CONST_CPdry    = np.float64(1004.64) # specific heat (dry air,constant pressure) [J/kg/K]
            #self.CONST_CVdry    = np.float64(None) # specific heat (dry air,constant volume)   [J/kg/K]
            self.CONST_LAPS     = np.float64(6.5E-3) # lapse rate of ISA  [K/m] 
            #self.CONST_LAPSdry  = np.float64(None) # dry adiabatic lapse rate  [K/m]    

            # water constants
            self.CONST_Mvap     = np.float64(18.02) # mass weight (water vapor) [g/mol]  
            self.CONST_Rvap     = np.float64(461.46) # specific gas constant (water vapor) [J/kg/K] 
            self.CONST_CPvap    = np.float64(1845.60) # specific heat (water vapor, constant pressure) [J/kg/K]  
            #self.CONST_CVvap    = np.float64(None) # specific heat (water vapor, constant volume)   [J/kg/K]
            self.CONST_CL       = np.float64(4218.0) # specific heat (liquid water) [J/kg/K]  
            self.CONST_CI       = np.float64(2006.0) # specific heat (ice) [J/kg/K]   

            #self.CONST_EPSvap   = np.float64(None) # Rdry / Rvap 
            #self.CONST_EPSTvap  = np.float64(None) # 1 / epsilon - 1 

            self.CONST_EMELT    = np.float64(3.34E+5) # heat of fusion [J/kg] 
            self.CONST_TMELT    = np.float64(273.15) # Freeze point of water 
            self.CONST_TFRZS    = np.float64(271.35) # Freeze point of sea  

            #self.CONST_LHV      = np.float64(None) # latent heat of vaporizaion for use 
            #self.CONST_LHS      = np.float64(None) # latent heat of sublimation for use
            #self.CONST_LHF      = np.float64(None) # latent heat of fusion      for use
            self.CONST_LHV0     = np.float64(2.5008E+6) # latent heat of vaporizaion at 0C [J/kg] 
            #self.CONST_LHV00    = np.float64(None) # latent heat of vaporizaion at 0K [J/kg]  
            self.CONST_LHS0     = np.float64(2.8342E+6) # latent heat of sublimation at 0C [J/kg]
            #self.CONST_LHS00    = np.float64(None) # latent heat of sublimation at 0K [J/kg]  
            #self.CONST_LHF0     = np.float64(None) # latent heat of fusion      at 0C [J/kg] 
            #self.CONST_LHF00    = np.float64(None) # latent heat of fusion      at 0K [J/kg]  
            self.CONST_PSAT0    = np.float64(610.7) # saturate pressure of water vapor at 0C [Pa] 
            self.CONST_DWATR    = np.float64(1000.0) # density of water [kg/m3]
            self.CONST_DICE     = np.float64(916.8) # density of ice   [kg/m3]  

            # standards
            #self.CONST_SOUND    = np.float64(None) # speed of sound (dry air at 0C) [m/s]  
            self.CONST_Pstd     = np.float64(101325.0) # standard pressure [Pa] 
            self.CONST_PRE00    = np.float64(100000.0) # pressure reference [Pa] 
            self.CONST_Tstd     = np.float64(288.15) # standard temperature (15C) [K]
            self.CONST_TEM00    = np.float64(273.15) # temperature reference (0C) [K] 
            self.CONST_PPM      = np.float64(1.E-6) # parts par million  


    def CONST_setup(self, fname_in=None):
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
                        self.CONST_RADIUS = earth_radius                 
                    if 'earth_angvel' in cnfs['cnstparam']:
                        earth_angvel = cnfs['cnstparam']['earth_angvel']
                        self.CONST_OHM = earth_angvel
                    if 'small_planet_factor' in cnfs['cnstparam']:
                        small_planet_factor = cnfs['cnstparam']['small_planet_factor']          
                        print("small_planet not implemented yet")   
                    if 'earth_gravity' in cnfs['cnstparam']:
                        earth_gravity = cnfs['cnstparam']['earth_gravity']
                        self.CONST_GRAV = earth_gravity                
                    if 'gas_cnst' in cnfs['cnstparam']:
                        gas_cnst = cnfs['cnstparam']['gas_cnst']
                        self.CONST_Rdry = gas_cnst
                    if 'gas_cnst_vap' in cnfs['cnstparam']:
                        gas_cnst_vap = cnfs['cnstparam']['gas_cnst_vap']
                        self.CONST_Rvap = gas_cnst_vap
                    if 'specific_heat_pre' in cnfs['cnstparam']:
                        specific_heat_pre = cnfs['cnstparam']['specific_heat_pre']
                        self.CONST_CPdry = specific_heat_pre
                    if 'specific_heat_pre_vap' in cnfs['cnstparam']:
                        specific_heat_pre_vap = cnfs['cnstparam']['specific_heat_pre_vap']
                        self.CONST_CPvap = specific_heat_pre_vap
                    if 'latent_heat_vap' in cnfs['cnstparam']:
                        latent_heat_vap = cnfs['cnstparam']['latent_heat_vap']
                        self.CONST_LHV = latent_heat_vap
                    if 'latent_heat_sub' in cnfs['cnstparam']:
                        latent_heat_sub = cnfs['cnstparam']['latent_heat_sub']
                        self.CONST_LHS = latent_heat_sub
                    if 'thermodyn_type' in cnfs['cnstparam']:
                        thermodyn_type = cnfs['cnstparam']['thermodyn_type']
                        self.CONST_THERMODYN_TYPE = thermodyn_type

        #if io_nml: print(cnfs['constparam'])

        # Constants
        self.CONST_PI = 4.0 * np.arctan(1.0)
        self.CONST_D2R = self.CONST_PI / 180.0
        self.CONST_EPS = np.finfo(float).eps
        self.CONST_EPS1 = 1.0 - np.finfo(float).eps
        self.CONST_HUGE = np.finfo(float).max

        self.CONST_CVdry = self.CONST_CPdry - self.CONST_Rdry
        self.CONST_LAPSdry = self.CONST_GRAV / self.CONST_CPdry

        self.CONST_CVvap = self.CONST_CPvap - self.CONST_Rvap
        self.CONST_EPSvap = self.CONST_Rdry / self.CONST_Rvap
        self.CONST_EPSTvap = 1.0 / self.CONST_EPSvap - 1.0

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
                print('*** kind (floating point value) =', np.finfo(float).dtype, file=log_file)
                print('*** precision(floating point value) =', np.finfo(float).precision, file=log_file)
                print('*** range (floating point value) =', (np.finfo(float).min, np.finfo(float).max), file=log_file)
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


