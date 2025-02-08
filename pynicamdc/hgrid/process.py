
"""                      [ communicator system ]
       MPI_COMM_WORLD
             |
    PRC_LOCAL_COMM_WORLD --split--> BULK_COMM_WORLD
                                        |
                               PRC_GLOBAL_COMM_WORLD --split--> PRC_LOCAL_COMM_WORLD"""

try:
    from mpi4py import MPI
    mpi_available = True
except ImportError:
    mpi_available = False
    raise ImportError('mpi4py library is not installed')

class Process:

    def __init__(self):
        self.parallel_prc = 1 # 1 for parallel, 0 for single: parallel only for now.
        self.prc_masterrank      = 0
        # local world
        self.prc_local_comm_world = -1
        self.prc_nprocs = 1
        self.prc_myrank = 0
        self.prc_ismaster = False
        self.prc_mpi_alive = False

    def prc_mpistart(self):
        self.prc_mpi_alive = True
        self.comm_world=MPI.COMM_WORLD
        self.prc_myrank = self.comm_world.Get_rank()
        self.prc_nprocs = self.comm_world.Get_size()
        if self.prc_myrank == self.prc_masterrank:
            self.prc_ismaster = True
        #    return MPI.COMM_WORLD
        return self.comm_world

    def prc_mpistop(self, io_l, fname_log):

        # flush 1kbyte
        if io_l: 
            with open(fname_log, 'a') as log_file:
                print(f"                                " * 32, file=log_file)
                print("+++ Abort MPI", file=log_file)
                
        if self.prc_ismaster:
            print("+++ Abort MPI")
    
        # Abort MPI     
        if self.prc_mpi_alive:
            self.comm_world.Abort() 


    def prc_mpifinish(self, io_l, fname_log):
    
        if io_l:
            with open(fname_log, 'a') as log_file:
                print("------------", file=log_file)
                print("+++ finalize MPI", file=log_file)

        self.comm_world.barrier()
        self.comm_world.Finalize() # Finalize MPI

        if self.adm_prc_me == self.adm_prc_run_master:
            print("*** MPI is peacefully finalized") 

# Global instance of Process class
prc = Process()
prc.prc_mpistart()
print('instantiated prc')
