class Process:

    """                      [ communicator system ]
       MPI_COMM_WORLD
             |
    PRC_LOCAL_COMM_WORLD --split--> BULK_COMM_WORLD
                                        |
                               PRC_GLOBAL_COMM_WORLD --split--> PRC_LOCAL_COMM_WORLD"""
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
        try:
            from mpi4py import MPI
            self.prc_mpi_alive = True
            #self.comm_world=MPI.COMM_WORLD
            return MPI.COMM_WORLD
        except ImportError:
            raise ImportError('mpi4py library is not installed')
            

    def prc_mpistop(self, mpi_comm_world, io_l, fname_log):

        # flush 1kbyte
        if io_l: 
            with open(fname_log, 'a') as log_file:
                print(f"                                " * 32, file=log_file)
                print("+++ Abort MPI", file=log_file)
                if self.prc_ismaster:
                    print("+++ Abort MPI")
    
        # Abort MPI
        mpi_comm_world.Abort() 

    def prc_mpifinish(self, mpi_comm_world, io_l, fname_log):
    
        if io_l:
            with open(fname_log, 'a') as log_file:
                print("------------", file=log_file)
                print("+++ finalize MPI", file=log_file)

        mpi_comm_world.barrier()
        mpi_comm_world.Finalize() #???

        if self.prc_ismaster:
            print("*** MPI is peacefully finalized") 

prc = Process()
print('instantiated class Process as prc (in mod_process.py)')
