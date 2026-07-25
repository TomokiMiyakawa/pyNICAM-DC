import sys
import time

from pynicamdc.share import comm_mode

# The mpi-vs-serial decision follows comm_mode.REQUESTED (set from the run's
# driver toml, `comm = "mpi" | "serial" | "auto"`); see comm_mode.py for the
# full policy and rationale.
if comm_mode.REQUESTED == "serial":
    mpi_available = False
    MPI = None   # replaced by the serial stub below
    comm_mode.SELECTED = "serial (requested)"
elif comm_mode.REQUESTED == "mpi":
    try:
        from mpi4py import MPI
    except ImportError as e:
        raise ImportError(
            "comm='mpi' was requested but mpi4py is not importable. "
            "Fix the environment, or set comm='serial' in the driver toml "
            "for an intentional single-process run.") from e
    mpi_available = True
    comm_mode.SELECTED = "mpi (requested)"
else:  # "auto"
    try:
        from mpi4py import MPI
        mpi_available = True
        comm_mode.SELECTED = "mpi (auto: mpi4py available)"
    except ImportError:
        mpi_available = False
        MPI = None
        comm_mode.SELECTED = "serial (auto: mpi4py not importable)"

if not mpi_available:
    # Serial (no-MPI) mode: single process, no mpi4py installed.
    # With one rank the mod_comm halo exchange lands in the Copy lists
    # (COMM_sortdest routes same-rank traffic to Copy_info_*), collectives
    # degenerate to identity, and Waitall is a no-op. One module does real
    # self-directed point-to-point with 1 rank: mod_grd.GRD_gen_plgrid
    # posts Irecv/Isend to itself while building the pole grid (legal MPI).
    # The stub therefore implements self-send with a tag-matched mailbox.

    class _SerialRequest:
        @staticmethod
        def Waitall(requests, statuses=None):
            return None

    class _SerialMPI:
        REQUEST_NULL = None
        MAX = 'MAX'
        MIN = 'MIN'
        SUM = 'SUM'
        Request = _SerialRequest

        @staticmethod
        def Wtime():
            return time.perf_counter()

        @staticmethod
        def Finalize():
            return None

    MPI = _SerialMPI

    class _SerialComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Barrier(self):
            return None

        def barrier(self):
            return None

        def Abort(self, errorcode=1):
            sys.exit(errorcode if errorcode else 1)

        # collectives degenerate to identity for a single rank
        def Allreduce(self, sendbuf, recvbuf, op=None):
            recvbuf[...] = sendbuf

        def allreduce(self, obj, op=None):
            return obj

        def Allgather(self, sendbuf, recvbuf):
            recvbuf[...] = sendbuf

        def allgather(self, obj):
            return [obj]

        def Alltoall(self, sendbuf, recvbuf):
            recvbuf[...] = sendbuf

        def alltoall(self, objs):
            return list(objs)

        def bcast(self, obj, root=0):
            return obj

        # --- self-directed point-to-point (source == dest == 0) ---------
        # Nonblocking semantics via a tag-matched mailbox: Isend copies the
        # payload immediately (buffered send), Irecv fills its buffer on
        # match, whichever side posts first. Waitall is then a no-op.

        def __init__(self):
            self._mailbox = []        # [(payload_copy, tag)] sent, not yet received
            self._pending_recvs = []  # [(recv_buffer, tag)] posted, not yet matched

        def Isend(self, buf, dest=0, tag=0):
            import numpy as _np
            payload = _np.array(buf, copy=True)
            for i, (rbuf, rtag) in enumerate(self._pending_recvs):
                if rtag == tag:
                    rbuf[...] = payload
                    del self._pending_recvs[i]
                    return None
            self._mailbox.append((payload, tag))
            return None

        def Irecv(self, buf, source=0, tag=0):
            for i, (payload, stag) in enumerate(self._mailbox):
                if stag == tag:
                    buf[...] = payload
                    del self._mailbox[i]
                    return None
            self._pending_recvs.append((buf, tag))
            return None

        def Send(self, buf, dest=0, tag=0):
            self.Isend(buf, dest, tag)

        def Recv(self, buf, source=0, tag=0):
            for i, (payload, stag) in enumerate(self._mailbox):
                if stag == tag:
                    buf[...] = payload
                    del self._mailbox[i]
                    return
            raise RuntimeError(
                f"serial mode: blocking Recv(tag={tag}) with no matching self-send")

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
        if mpi_available:
            self.prc_mpi_alive = True
            self.comm_world = MPI.COMM_WORLD
            self.prc_myrank = self.comm_world.Get_rank()
            self.prc_nprocs = self.comm_world.Get_size()
        else:
            # serial mode: prc_mpi_alive stays False so PRC_MPItime and the
            # Abort/Barrier guards take their non-MPI paths
            self.comm_world = _SerialComm()
            self.prc_myrank = 0
            self.prc_nprocs = 1
            print(f"*** pyNICAM comm: {comm_mode.SELECTED}")
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
        
        import sys
        sys.exit(1)

    def prc_mpifinish(self, io_l, fname_log):
    
        if io_l:
            with open(fname_log, 'a') as log_file:
                print("------------", file=log_file)
                print("+++ finalize MPI", file=log_file)

        self.comm_world.barrier()
        #self.comm_world.Finalize() # Finalize MPI
        MPI.Finalize()

        if self.prc_ismaster:
            print("*** MPI is peacefully finalized") 

        return

    def PRC_MPIbarrier(self):

        if self.prc_mpi_alive:  # Assuming PRC_mpi_alive is a global flag
            self.comm_world.Barrier()  # Synchronize all processes

        return


    def PRC_MPItime(self) -> float:

        if self.prc_mpi_alive:  
            return MPI.Wtime()  # Equivalent to MPI_WTIME() in Fortran
        else:
            return time.process_time()  # Equivalent to CPU_TIME(time) in Fortran


# Global instance of Process class
prc = Process()
prc.prc_mpistart()
#print('instantiated proc')
