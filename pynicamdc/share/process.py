class Process:
    #-----------------------------------------------------------------------------
    #
    #++ Public parameters & variables
    #
    #-----------------------------------------------------------------------------
    #                          [ communicator system ]
    #    MPI_COMM_WORLD
    #          |
    # PRC_LOCAL_COMM_WORLD --split--> BULK_COMM_WORLD
    #                                     |
    #                            PRC_GLOBAL_COMM_WORLD --split--> PRC_LOCAL_COMM_WORLD
    #-----------------------------------------------------------------------------
    def __init__(self):
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
            return MPI.COMM_WORLD
        except ImportError:
            raise ImportError('mpi4py library is not installed')

            