import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf

class Cldr:
    
    _instance = None

    # Automatic calendar setting:
    # - Year 0-999   : 360-day calendar
    # - Year 1000-1899: 365-day calendar
    # - Year 1900+    : Gregorian calendar
    oauto = True  # Use automatic calendar selection?

    # Use Gregorian calendar?
    ogrego = True

    # Ideal calendar (fixed n days per month)?
    oideal = False

        # Ideal case settings
    imonyr = 12   # Number of months in a year (ideal case)
    idaymo = 30   # Number of days in a month (ideal case)

        # Number of days in each month (normal year / leap year)
    monday =[
            [31, 28], [31, 29],  # January, February
            [31, 30], [31, 30],  # March, April
            [31, 30], [31, 31],  # May, June
            [30, 31], [30, 31],  # July, August
            [30, 31], [30, 31]   # September, October, November, December
            ]

    # Time unit settings
    ihrday = 24  # Hours per day
    iminhr = 60  # Minutes per hour
    isecmn = 60  # Seconds per minute

    # Perpetual calendar settings
    operpt = False  # Use perpetual calendar?
    iyrpp = 0       # Perpetual year
    imonpp = 3      # Perpetual month
    idaypp = 21     # Perpetual day
    
    def __init__(self):
        pass

    def CALENDAR_setup(self, fname_in):

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[calender]/Category[common share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'cldrparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** cldrparam not found in toml file! use default.", file=log_file)
                #prc.prc_mpistop(std.io_l, std.fname_log)

        else:
            cnfs = cnfs['cldrparam']
            #self.GRD_grid_type = cnfs['GRD_grid_type']
            # not ready yet

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(cnfs,file=log_file)

        return

    def CALENDAR_yh2ss(self, idate):
        idays = self.CALENDAR_ym2dd(idate[0], idate[1], idate[2])  # [OUT]
        # Convert hours and minutes to seconds
        rsec = self.CALENDAR_hm2rs(idate[3], idate[4], idate[5])  # [OUT]
        # Convert days and seconds to total seconds
        dsec = self.CALENDAR_ds2ss(idays, rsec)
        return dsec
    
    def CALENDAR_ym2dd(self, iyear, imonth, iday):

        if self.oauto:
            if iyear >= 1900:
                self.ogrego = True
            else:
                self.ogrego = False
                if iyear >= 1000:
                    self.oideal = False
                else:
                    self.oideal = True
                    self.imonyr = 12
                    self.idaymo = 30

        if self.ogrego or not self.oideal:
            if imonth > 0:
                jyear = iyear + (imonth - 1) // 12
                jmonth = (imonth - 1) % 12 + 1
            else:
                jyear = iyear - (-imonth) // 12 - 1
                jmonth = 12 - (-imonth % 12)

        if self.ogrego:
            jy4 = (jyear + 3) // 4
            jcent = (jyear + 99) // 100
            jcent4 = (jyear + 399) // 400

            idays = iday + jyear * 365 + jy4 - jcent + jcent4

            if self.CALENDAR_leapyr(jyear):
                ileap = 2
            else:
                ileap = 1
        elif not self.oideal:
            idays = iday + jyear * 365
            ileap = 1

        if self.ogrego or not self.oideal:
            for m in range(jmonth - 1):
                idays += self.monday[m][ileap - 1]  # Adjust index for leap/non-leap year
        else:
            idayyr = self.idaymo * self.imonyr
            idays = iday
            idays += iyear * idayyr
            idays += (imonth - 1) * self.idaymo

        return idays


    def CALENDAR_hm2rs(self, ihour, imin, isec):
        isecs = ihour * self.iminhr * self.isecmn + imin * self.isecmn + isec
        rsec = float(isecs) 
        return rsec

    def CALENDAR_ds2ss(self, idays, rsec):
        isecdy = self.isecmn * self.iminhr * self.ihrday
        dsec = rsec + float(idays - 1) * float(isecdy)
        return dsec
    
    def CALENDAR_leapyr(self, iyear):
        iy = iyear % 4
        iycen = iyear % 100
        icent = (iyear // 100) % 4
        if iy == 0 and (iycen != 0 or icent == 0):
            CALENDAR_leapyr = True
        else:
            CALENDAR_leapyr = False

        return CALENDAR_leapyr
    
    def CALENDAR_ss2cc(self, dsec, number_only):
        isecdy = self.isecmn * self.iminhr * self.ihrday
        idays = int(dsec / isecdy) + 1
        rsec = dsec - float(idays - 1) * float(isecdy)

        return htime
    
    def CALENDAR_ss2cc(self, dsec, number_only=False):
        itime = self.CALENDAR_ss2yh(dsec)  # Assuming this method returns a list [YYYY, MM, DD, HH, MM, SS]
        if number_only:
            htime=f"{itime[0]:04}{itime[1]:02}{itime[2]:02}{itime[3]:02}{itime[4]:02}{itime[5]:02}"
        else:  
            htime =f"{itime[0]:04}/{itime[1]:02}/{itime[2]:02} - {itime[3]:02}:{itime[4]:02}:{itime[5]:02}"
        return htime
    
    def CALENDAR_ss2yh(self, dsec):
        idays, rsec = self.CALENDAR_ss2ds(dsec)  
        idate = [0] * 6  
        idate[0], idate[1], idate[2] = self.CALENDAR_dd2ym(idays) 
        idate[3], idate[4], idate[5] = self.CALENDAR_rs2hm(rsec)  
        return idate  

    def CALENDAR_ss2ds(self, dsec):
        isecdy = self.isecmn * self.iminhr * self.ihrday
        idays = int(dsec / float(isecdy)) + 1
        rsec = dsec - float(idays - 1) * float(isecdy)
        if round(rsec) >= isecdy:  # Equivalent to nint() in Fortran
            idays += 1
            rsec -= float(isecdy)
        return idays, rsec
    

    def CALENDAR_dd2ym(self, idays):
        if self.oauto:
            if idays >= 693961:  # 1900*365 + 1900/4 - 19 + 5
                self.ogrego = True
            else:
                self.ogrego = False
                if idays >= 1000 * 365:
                    self.oideal = False
                else:
                    self.oideal = True
                    self.imonyr = 12
                    self.idaymo = 30

        if self.ogrego:
            jyear = int(float(self.idays) / 365.24)  # First guess

            while True:
                jy4 = (jyear + 3) // 4
                jcent = (jyear + 99) // 100
                jcent4 = (jyear + 399) // 400

                id = jyear * 365 + jy4 - jcent + jcent4
                if idays <= id:
                    jyear -= 1
                else:
                    break

            iyear = jyear
            jdays = idays - id

            if self.CALENDAR_leapyr(iyear):
                ileap = 2
            else:
                ileap = 1

        elif not self.oideal:
            iyear = idays // 365
            jdays = idays - iyear * 365
            ileap = 1

        if self.ogrego or not self.oideal:
            id = 0
            for m in range(12):
                id += self.monday[m][ileap - 1]  # Adjust for zero-based indexing
                if jdays <= id:
                    imonth = m + 1  # Convert to 1-based month
                    iday = jdays + self.monday[m][ileap - 1] - id
                    break
        else:
            idayyr = self.idaymo * self.imonyr
            iyear = (idays - 1) // idayyr
            imonth = ((idays - iyear * idayyr - 1) // self.idaymo) + 1
            iday = idays - iyear * idayyr - (imonth - 1) * self.idaymo

        return iyear, imonth, iday
    
    def CALENDAR_rs2hm(self, rsec):
        ihour = int(rsec / float(self.iminhr * self.isecmn))
        isecs = rsec - float(ihour * self.iminhr * self.isecmn)
        imin = int(isecs / float(self.isecmn))
        isecs = isecs - float(imin * self.isecmn)
        isec = round(isecs)  # Equivalent to Fortran's `nint()`
        if isec >= self.isecmn:
            imin += 1
            isec -= self.isecmn
        if imin == self.iminhr:
            ihour += 1
            imin -= self.iminhr
        return ihour, imin, isec



    def CALENDAR_dd2ym_autobycopilot(self, idays):
        if self.ogrego or not self.oideal:
            if self.ogrego:
                jcent = idays // 36525
                jyear = idays - 36525 * jcent
                jy4 = (jyear + 3) // 4
                jcent = (jyear + 99) // 100
                jcent4 = (jyear + 399) // 400
                jyear = jyear - jy4 + jcent - jcent4
            else:
                jyear = idays // 365
                jy4 = (jyear + 3) // 4
                jyear = idays - jyear * 365 + jy4
            if self.CALENDAR_leapyr(jyear):
                ileap = 2
            else:
                ileap = 1
            jmonth = 1
            while idays > self.monday[jmonth - 1][ileap - 1]:
                idays -= self.monday[jmonth - 1][ileap - 1]
                jmonth += 1
        else:
            idayyr = self.idaymo * self.imonyr
            jyear = idays // idayyr
            idays -= jyear * idayyr
            jmonth = idays // self.idaymo + 1
            idays -= (jmonth - 1) * self.idaymo
        return jyear, jmonth, idays     


        # Convert total seconds to days and seconds
        idays, rsec = self.CALENDAR_ss2ds(dsec)
        # Convert days to year, month, and day
        iyear, imonth, iday = self.CALENDAR_dd2ym(idays)

cldr=Cldr()
#print("instanciated cldr")
