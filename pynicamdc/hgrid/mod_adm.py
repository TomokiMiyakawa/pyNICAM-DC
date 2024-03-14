#import numpy as np
import toml
from mod_process import prc
from mod_stdio import stdio
class Adm:

    # Basic definition & information

    # Local region and process
    I_l = 1
    I_prc = 2

    # Region ID and direction
    I_RGNID = 1
    I_DIR = 2

    # Identifiers of directions of region edges
    I_SW = 1
    I_NW = 2
    I_NE = 3
    I_SE = 4

    # Identifiers of directions of region vertices
    I_W = 1
    I_N = 2
    I_E = 3
    I_S = 4

    # Identifier of poles (north pole or south pole)
    I_NPL = 1
    I_SPL = 2

    # Identifier of triangle element (i-axis-side or j-axis side)
    ADM_TI = 1
    ADM_TJ = 2

    # Identifier of arc element (i-axis-side, ij-axis side, or j-axis side)
    ADM_AI = 1
    ADM_AIJ = 2
    ADM_AJ = 3

    # Identifier of 1 variable
    ADM_KNONE = 1

    # Dimension of the spatial vector
    ADM_nxyz = 3

    # Maximum number of region per process
    RGNMNG_llim = 2560 

    def __init__(self,comm_world,np):
        self.comm = comm_world
        self.np = np
        
    def RGNMNG_setup(self, io_l, io_nml, fname_log, fname_in=None):
    
        if io_l: 
            with open(fname_log, 'a') as log_file:
                print("+++ Module[rgnmng]", file=log_file)        

        if fname_in is None:
            with open(fname_log, 'a') as log_file:
                if io_l: print("*** input toml file is not specified. use default.", file=log_file)
                # maybe should stop here 
        else:
            if io_l:
                with open(fname_log, 'a') as log_file: 
                    print(f"*** input toml file is ", fname_in, file=log_file)

            with open(fname_in, 'r') as  file:
                cnfs = toml.load(file)

            if 'rgnmngparam' not in cnfs:
                if io_l:
                    with open(fname_log, 'a') as log_file: 
                        print("*** rgnmngparam not specified in toml file. use default.", file=log_file)
                        # maybe should stop here 
            else:
                if 'RGNMNG_in_fname' in cnfs['rgnmngparam']:
                    RGNMNG_in_fname = cnfs['rgnmngparam']['RGNMNG_in_fname']                    

                if 'RGNMNG_out_fname' in cnfs['cnstparam']:
                    RGNMNG_out_fname = cnfs['rgnmngparam']['RGNMNG_out_fname']      


            if self.ADM_lall > self.RGNMNG_llim:
                if io_l:
                    with open(fname_log, 'a') as log_file: 
                        print(f'xxx limit exceed! local region:', self.ADM_lall, RGNMNG_llim, file=log_file)

                prc.prc_mpistop(io_l, fname_log)  
    

    #if io_nml: print(cnfs['rgnmngparam'])


    def output_info(self):
        pass


    def ADM_setup(self, io_l, io_nml, fname_log, fname_in):
        ADM_prc_me = prc.prc_myrank + 1
        ADM_prc_pl = 1

##ifdef _FIXEDINDEX_
        #    if ( PRC_nprocs /= PRC_nprocs ) then
        #       write(*,*) 'xxx Fixed prc_all is not match (fixed,requested): ', PRC_nprocs, PRC_nprocs
        #       stop
        #    endif
##endif

        if io_l: 
            with open(fname_log, 'a') as log_file:
                print("+++ Module[adm]", file=log_file)        
                print(f"*** input toml file is ", fname_in, file=log_file)
 
        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'admparam' not in cnfs:
            with open(fname_log, 'a') as log_file:
                print("*** cnstparam not found in toml file! STOP.", file=log_file)
                #stop

        else:
            glevel = cnfs['admparam']['glevel']  
            rlevel = cnfs['admparam']['rlevel']  
            vlayer = cnfs['admparam']['vlayer']  
            rgnmngfname = cnfs['admparam']['rgnmngfname']  
            ADM_HGRID_SYSTEM = cnfs['admparam']['ADM_HGRID_SYSTEM']  
            ADM_vlink = cnfs['admparam']['ADM_vlink']  
            #ADM_XTMS_MLCP_S = cnfs['admparam']['ADM_XTMS_MLCP_S']  
            debug = cnfs['admparam']['debug']  

            #if io_nml: print(cnfs['constparam'])

            #! Error if glevel & rlevel are not defined
            #if ( glevel < 1 ) then
            #write(*,*) 'xxx [ADM_setup] glevel is not appropriate :', glevel
            #call PRC_MPIstop
            #endif
            #if ( rlevel < 0 ) then
            #write(*,*) 'xxx [ADM_setup] rlevel is not appropriate :', rlevel
            #call PRC_MPIstop
            #endif

            #! Error if glevel & rlevel are different from global parameter
            #if ( GLOBAL_ensemble_on ) then
            #if ( glevel /= GLOBAL_glevel ) then
            #write(*,*) 'xxx glevel is not equal (global,local) :', GLOBAL_glevel, glevel
            #call PRC_MPIstop
            #endif
            #if ( rlevel /= GLOBAL_rlevel ) then
            #write(*,*) 'xxx rlevel is not equal (global,local) :', GLOBAL_rlevel, rlevel
            #call PRC_MPIstop
            #endif
            #endif


#ifdef _FIXEDINDEX_
            #if ( ADM_HGRID_SYSTEM == 'ICO' ) then
            #dmd        = 10
            #elseif( ADM_HGRID_SYSTEM == 'PERIODIC-1DMD' ) then ! T.Ohno 110721
            #dmd        = 1
            #ADM_prc_pl = -999
            #elseif( ADM_HGRID_SYSTEM == '1DMD-ON-SPHERE' ) then ! M.Hara 110721
            #dmd        = 1
            #ADM_prc_pl = -999
            #elseif( ADM_HGRID_SYSTEM == 'ICO-XTMS' ) then
            #dmd        = 10
            #else
            #write(*,*) 'xxx [ADM_setup] Not appropriate param for ADM_HGRID_SYSTEM. STOP.', trim(ADM_HGRID_SYSTEM)
            #call PRC_MPIstop
            #endif
#else

            if ( ADM_HGRID_SYSTEM == 'ICO' ):
                ADM_vlink  = 5
                dmd        = 10
                ADM_prc_pl = 1

            #elseif( ADM_HGRID_SYSTEM == 'LCP' ) then
            #    if( ADM_vlink == -1 ) ADM_vlink = 6
            #    dmd        = 4 * ADM_vlink
            #    ADM_prc_pl = 1
            #    elseif( ADM_HGRID_SYSTEM == 'MLCP-OLD' ) then
            #    if( ADM_vlink == -1 ) ADM_vlink = 6
            #    dmd        = 2 * ADM_vlink
            #    ADM_prc_pl = 1
            #    elseif( ADM_HGRID_SYSTEM == 'MLCP' ) then
            #    if( ADM_vlink == -1 ) ADM_vlink = 6
            #    dmd        = (1+ADM_XTMS_MLCP_S)  * ADM_vlink
            #    ADM_prc_pl = 1
            #    elseif( ADM_HGRID_SYSTEM == 'PERIODIC-1DMD' ) then ! T.Ohno 110721
            #    ADM_vlink  = 5
            #    dmd        = 1
            #    ADM_prc_pl = -999
            #    elseif( ADM_HGRID_SYSTEM == '1DMD-ON-SPHERE' ) then ! M.Hara 110721
            #    ADM_vlink  = 5
            #    dmd        = 1
            #   ADM_prc_pl = -999
            #    elseif( ADM_HGRID_SYSTEM == 'ICO-XTMS' ) then
            #    ADM_vlink  = 5
            #    dmd        = 10
            #    ADM_prc_pl = 1

            else:
                with open(fname_log, 'a') as log_file:
                    print("xxx [ADM_setup] Not appropriate param for ADM_HGRID_SYSTEM. STOP.", ADM_HGRID_SYSTEM, file=log_file)
                    #call PRC_MPIstop

            #endif

            ADM_gall_pl = ADM_vlink + 1
            ADM_gmax_pl = ADM_vlink + 1
#endif

#ifdef _FIXEDINDEX_
            #if ( ADM_glevel /= glevel ) then
            #write(*,*) 'xxx [ADM_setup] Fixed glevel is not match (fixed,requested): ', ADM_glevel, glevel
            #call PRC_MPIstop
            #endif
            #if ( ADM_rlevel /= rlevel ) then
            #write(*,*) 'xxx [ADM_setup] Fixed rlevel is not match (fixed,requested): ', ADM_rlevel, rlevel
            #call PRC_MPIstop
            #endif
            #if ( ADM_vlayer /= vlayer ) then
            #write(*,*) 'xxx [ADM_setup] Fixed vlayer is not match (fixed,requested): ', ADM_vlayer, vlayer
            #call PRC_MPIstop
            #endif
            #if ( ADM_DMD /= dmd ) then
            #write(*,*) 'xxx [ADM_setup] Fixed dmd is not match (fixed,requested): ', ADM_DMD, dmd
            #call PRC_MPIstop
            #endif
#else
           
            self.ADM_glevel = glevel
            self.ADM_rlevel = rlevel
            self.ADM_vlayer = vlayer
            self.ADM_DMD = dmd

            # Calculations
            self.ADM_rgn_nmax = (2 ** self.ADM_rlevel) * (2 ** self.ADM_rlevel) * self.ADM_DMD
            #prc.nprocs=self.comm.Get_size()
            self.ADM_lall = self.ADM_rgn_nmax // prc.nprocs

            nmax = 2 ** (self.ADM_glevel - self.ADM_rlevel)
            self.ADM_gall_1d = 1 + nmax + 1
            self.ADM_gmin = 1 + 1
            self.ADM_gmax = 1 + nmax

            self.ADM_gall = (1 + nmax + 1) * (1 + nmax + 1)
            self.ADM_gall_in = (nmax + 1) * (nmax + 1)

            if self.ADM_vlayer == 1:
                self.ADM_kall = 1
                self.ADM_kmin = 1
                self.ADM_kmax = 1
            else:
                self.ADM_kall = 1 + self.ADM_vlayer + 1
                self.ADM_kmin = 1 + 1
                self.ADM_kmax = 1 + self.ADM_vlayer

            self.RGNMNG_setup(stdio.io_l, stdio.io_nml, stdio.fname_log, fname_in)

            #    allocate( GLOBAL_extension_rgn(ADM_lall) )
            #    allocate( ADM_have_sgp        (ADM_lall) )
            #GLOBAL_extension_rgn(:) = ''
            #ADM_have_sgp        (:) = .false.
            # Allocate and initialize GLOBAL_extension_rgn with empty strings
            GLOBAL_extension_rgn = self.np.empty(self.ADM_lall, dtype=str)
            GLOBAL_extension_rgn[:] = ''
            # Allocate and initialize ADM_have_sgp with False
            ADM_have_sgp = self.np.full(self.ADM_lall, False, dtype=bool)

            #do l = 1, ADM_lall
            #rgnid = RGNMNG_lp2r(l,ADM_prc_me)

            #write(GLOBAL_extension_rgn(l),'(A,I5.5)') '.rgn', rgnid-1

            #    if ( RGNMNG_vert_num(I_W,rgnid) == 3 ) then
            #        ADM_have_sgp(l) = .true.
            #    endif
            #enddo

            for l in range(self.ADM_lall):  
    # Python's range starts from 0 by default, adjust according to your array indexing
                rgnid = RGNMNG_lp2r[l, ADM_prc_me]

    # Formatting the string with '.rgn' prefix and the (rgnid-1) value
                #GLOBAL_extension_rgn[l] = f".rgn{rgnid-1:05d}"
                GLOBAL_extension_rgn[l] = f".rgn{rgnid:05d}"

    # Conditional statement
                #if RGNMNG_vert_num[I_W, rgnid] == 3:
                if RGNMNG_vert_num[I_W, rgnid+1] == 3:
                    ADM_have_sgp[l] = True

            if ADM_prc_me == ADM_prc_pl:
                ADM_have_pl = True
            else:  
                ADM_have_pl = False

            ADM_l_me = 0

            output_info

        return

#adm = Adm()