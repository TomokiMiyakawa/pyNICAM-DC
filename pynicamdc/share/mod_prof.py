from mod_stdio import std 
from mod_process import prc
import toml
#from mod_precision import Pr

class Prof:
    _instance = None    

    def __init__(self):
        #self.rdtype = rdtype
        #self.cnfs = toml.load(fname_in)['param_prof']
        #self.prof_rap_level = self.cnfs['prof_rap_level']
        #self.prof_mpi_barrier = self.cnfs['prof_mpi_barrier']
        pass


    def PROF_setup(self, fname_in, rdtype):

        self.rdtype         = rdtype  #precision
        self.PROF_rapnlimit = 300
        self.PROF_prefix    = ""  # Equivalent to character(len=H_SHORT), initialized as an empty string
        self.PROF_rapnmax   = 0  # Counter for the number of rapid profiling entries
        self.PROF_rapname   = [""] * self.PROF_rapnlimit  # Equivalent to character(len=H_SHORT*2) array
        self.PROF_grpnmax   = 0  # Counter for the number of group profiles
        self.PROF_grpname   = [""] * self.PROF_rapnlimit  # Equivalent to character(len=H_SHORT) array
        self.PROF_grpid     = [0]  * self.PROF_rapnlimit  # Integer array for group IDs
        self.PROF_raptstr   = [0.0]* self.PROF_rapnlimit  # Real (Double Precision) array for timestamps
        self.PROF_rapttot   = [0.0]* self.PROF_rapnlimit  # Real (Double Precision) array for total time
        self.PROF_rapnstr   = [0]  * self.PROF_rapnlimit  # Integer array for start counts
        self.PROF_rapnend   = [0]  * self.PROF_rapnlimit  # Integer array for end counts
        self.PROF_raplevel  = [0]  * self.PROF_rapnlimit  # Integer array for profiling levels
        # Default profiling levels
        self.PROF_default_rap_level = 2
        self.PROF_rap_level = 2  # Current profiling level
        # MPI barrier flag
        self.PROF_mpi_barrier = False  # Equivalent to Fortran `.false.`

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[prof]/Category[common share]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'param_prof' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** param_prof not found in toml file! STOP.", file=log_file)
                #stop

        else:
            profcnfs = cnfs['param_prof']
            self.Prof_rap_level = profcnfs['prof_rap_level']
            self.Prof_mpi_barrier = profcnfs['prof_mpi_barrier']

        if std.io_nml: 
            if std.io_l:
                with open(std.fname_log, 'a') as log_file: 
                    print(profcnfs,file=log_file)

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("", file=log_file)    
                print("*** Rap output level              = ", self.Prof_rap_level, file=log_file)    
                print("*** Add MPI_barrier in every rap? = ", self.Prof_mpi_barrier, file=log_file)

        self.PROF_prefix = ''

        return
    

    def PROF_setprefx(self, prefxname):

        if prefxname == "":  # No prefix
            self.PROF_prefix = ""
        else:
            self.PROF_prefix = prefxname.strip() + "_"

        return
    
    
    def PROF_rapstart(self, rapname_base, level=None):

        #PROF_default_rap_level = 2

        if level is not None:  # Equivalent to Fortran's `present(level)`
            level_ = level
        else:
            level_ = self.PROF_default_rap_level

        if level_ > self.PROF_rap_level:
            return  

        rapname = self.PROF_prefix.strip() + rapname_base.strip()  # Equivalent to `trim()`
        id = self.get_rapid(rapname, level_)  

        if self.PROF_mpi_barrier:
            prc.PRC_MPIbarrier()  

        self.PROF_raptstr[id] = prc.PRC_MPItime()  # Store timestamp
        self.PROF_rapnstr[id] += 1  # Increment counter

        return
    

    def PROF_rapend(self, rapname_base: str, level: int = None):
        """Ends the profiling for a given event."""

        # Check level (if provided) and return early if not within limits
        if level is not None:
            if level > self.PROF_rap_level:
                return

        # Construct full rapname with prefix
        rapname = self.PROF_prefix.strip() + rapname_base.strip()

        # Get profiling ID
        id_ = self.get_rapid(rapname, level)

        if level > self.PROF_rap_level:
            return

        # MPI Barrier if enabled
        if self.PROF_mpi_barrier:
            prc.PRC_MPIbarrier()

        # Update profiling counters
        self.PROF_rapttot[id_] += prc.PRC_MPItime() - self.PROF_raptstr[id_]
        self.PROF_rapnend[id_] += 1

        # Optional performance analysis tools
        #if "_FINEPA_" in globals():  # Mimicking `#ifdef _FINEPA_`
        #    STOP_COLLECTION(rapname.strip())

        #if "_FAPP_" in globals():  # Mimicking `#ifdef _FAPP_`
        #    FAPP_STOP(PROF_grpname[PROF_grpid[id_]].strip(), id_, level)

        return


    def PROF_rapreport(self):

        avgvar = [0.0] * self.PROF_rapnmax
        maxvar = [0.0] * self.PROF_rapnmax
        minvar = [0.0] * self.PROF_rapnmax
        maxidx = [0] * self.PROF_rapnmax
        minidx = [0] * self.PROF_rapnmax

        # Check for mismatches in profiling counts
        for id_ in range(self.PROF_rapnmax):
            if self.PROF_rapnstr[id_] != self.PROF_rapnend[id_]:
                print(f"*** Mismatch Report {id_} {self.PROF_rapname[id_]} {self.PROF_rapnstr[id_]} {self.PROF_rapnend[id_]}")

        if std.io_l: 
            with open(std.fname_log, 'a') as log_file:
                print("", file= log_file)    
                print("*** Computational Time Report", file= log_file)
                print(f"*** Rap level is {self.PROF_rap_level}", file= log_file)

        # Report for each node
        if std.io_log_allnode:
            for gid in range(self.PROF_rapnmax):
                for id_ in range(self.PROF_rapnmax):
                    if self.PROF_raplevel[id_] <= self.PROF_rap_level and self.PROF_grpid[id_] == gid:
                        if std.io_l: 
                            with open(std.fname_log, 'a') as log_file:
                                print(f"*** ID={id_:03d} : {self.PROF_rapname[id_]:<33} T={self.PROF_rapttot[id_]:10.3f} N={self.PROF_rapnstr[id_]}", file= log_file)
        else:
        # Perform MPI time statistics collection
            prc.PRC_MPItimestat(
                avgvar[:self.PROF_rapnmax],  # [OUT]
                maxvar[:self.PROF_rapnmax],  # [OUT]
                minvar[:self.PROF_rapnmax],  # [OUT]
                maxidx[:self.PROF_rapnmax],  # [OUT]
                minidx[:self.PROF_rapnmax],  # [OUT]
                self.PROF_rapttot[:self.PROF_rapnmax]  # [IN]
            )

            if std.io_log_suppress:  # Report to STDOUT
                if prc.PRC_IsMaster():
                    print("*** Computational Time Report")
                    # Reporting time statistics
                    for gid in range(self.PROF_rapnmax):
                        for id_ in range(self.PROF_rapnmax):
                            if self.PROF_raplevel[id_] <= self.PROF_rap_level and self.PROF_grpid[id_] == gid:
                                print(
                                    f"*** ID={id_:03d} : {self.PROF_rapname[id_]} "
                                    f" T(avg)={avgvar[id_]:10.3f}, "
                                    f" T(max)={maxvar[id_]:10.3f}[{maxidx[id_]}], "
                                    f" T(min)={minvar[id_]:10.3f}[{minidx[id_]}], "
                                    f" N={self.PROF_rapnstr[id_]}"
                                )   

            else:
                for gid in range(self.PROF_rapnmax):
                    for id_ in range(self.PROF_rapnmax):
                        if self.PROF_raplevel[id_] <= self.PROF_rap_level and self.PROF_grpid[id_] == gid:
                            if std.io_l: 
                                with open(std.fname_log, 'a') as log_file: 
                                    print(
                                        f"*** ID={id_:03d} : {self.PROF_rapname[id_]} "
                                        f" T(avg)={avgvar[id_]:10.3f}, "
                                        f" T(max)={maxvar[id_]:10.3f}[{maxidx[id_]}], "
                                        f" T(min)={minvar[id_]:10.3f}[{minidx[id_]}], "
                                        f" N={self.PROF_rapnstr[id_]}",
                                        file= log_file
                                    )    
                
            return    


    def get_rapid(self, rapname: str, level: int) -> int:
        
        #global PROF_rapnmax, PROF_rapname, PROF_raplevel, PROF_rapnstr, PROF_rapnend, PROF_rapttot, PROF_grpid
        trapname = rapname.strip()
        trapname2 = rapname.strip()

        # Search for an existing entry
        for id_ in range(self.PROF_rapnmax):  
            if trapname == self.PROF_rapname[id_]:
                level = self.PROF_raplevel[id_]
                return id_

        # If not found, add a new entry
        id_ = self.PROF_rapnmax
        self.PROF_rapnmax += 1        
        self.PROF_rapname[id_] = trapname

        self.PROF_rapnstr[id_] = 0
        self.PROF_rapnend[id_] = 0
        self.PROF_rapttot[id_] = 0.0  # Equivalent to DP (Double Precision)

        self.PROF_grpid[id_] = self.get_grpid(trapname2)
        self.PROF_raplevel[id_] = level

        return id_

    
    def get_grpid(self, rapname: str) -> int:
        
        # Extract the group name (everything before the first space)
        idx = rapname.find(" ")  # Equivalent to Fortran `index(rapname, ' ')`
        if idx > 0:
            grpname = rapname[:idx]  # Extract substring before space
        else:
            grpname = rapname  # If no space, use full name

        # Search for an existing group
        for gid in range(self.PROF_grpnmax):  # Fortran arrays are 1-based
            if grpname == self.PROF_grpname[gid]:
                return gid

        # If not found, add a new group entry
        gid = self.PROF_grpnmax
        self.PROF_grpnmax += 1
        self.PROF_grpname[gid] = grpname

        return gid


prf = Prof()
#print('instantiated prf')    