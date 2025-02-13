!-------------------------------------------------------------------------------
!
!+  Program mkvlayer
!
!-------------------------------------------------------------------------------
program prg_mkvlayer
  !-----------------------------------------------------------------------------
  !
  !++ Description: 
  !       This program makes grid systems based on the icosahedral grid 
  !       configuration.
  ! 
  !++ Current Corresponding Author : H.Tomita
  ! 
  !++ History: 
  !      Version   Date       Comment 
  !      -----------------------------------------------------------------------
  !      0.00      04-02-17  Imported from igdc-4.34
  !      -----------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------------
  !
  !++ Used modules ( shared )
  !
  !=============================================================================
  integer, parameter :: kdum=1
  integer, parameter :: fid=11
  integer :: num_of_layer = 10
  real(8) :: ztop = 1.0D+4
  character(len=256) :: outfname = 'outfile'
  character(len=256) :: infname = 'infile'
  character(len=16) :: layer_type = 'POWER'
  real(8) :: fact = 1.0D0

  namelist / mkvlayer_cnf / &
       num_of_layer,        & !--- number of layers
       ztop,                & !--- height of model top
       outfname,            & !--- output file name
       layer_type,          & !--- type of layer
       fact,                & !--- factor  if layer_type='POWER'
       infname                !--- input file name if layer_type='GIVEN'

  real(8),allocatable :: z_c(:)
  real(8),allocatable :: z_h(:)

  integer :: kmin
  integer :: kmax
  integer :: kall

  open(fid,file='mkvlayer.cnf',status='old',form='formatted')
  read(fid,nml=mkvlayer_cnf)
  close(fid)


  select case(trim(layer_type))
  case('POWER')
     call mk_layer_powerfunc(fact)
  case('GIVEN')
     call mk_layer_given(infname)
  end select
  !
  call output_layer(outfname)

  !=============================================================================
contains
  !-----------------------------------------------------------------------------
  subroutine mk_layer_powerfunc( fact )
    implicit none

    real(8) :: a
    integer :: k
    real(8),intent(in) :: fact

    kmin=kdum+1
    kmax=kdum+num_of_layer
    kall=kdum+num_of_layer+kdum
    allocate(z_c(kall))
    allocate(z_h(kall)) 
    a = ztop/(dble(num_of_layer)**fact)
    do k = kmin, kmax+1
       z_h(k) = a * (dble(k-kmin)**fact)
    enddo
    z_h(kmin-1) = z_h(kmin) - ( z_h(kmin+1) - z_h(kmin) )
    do k= kmin-1, kmax
       z_c(k) = z_h(k) + ( z_h(k+1) - z_h(k) )*0.5D0
    enddo
    z_c(kmax+1) = z_h(kmax+1) + ( z_h(kmax+1) - z_h(kmax) )*0.5D0
  end subroutine mk_layer_powerfunc

  subroutine mk_layer_given( infname )
    implicit none

    character(len=*), intent(in) :: infname
    integer,parameter :: fid=10
    integer :: k
    open(fid,file=trim(infname),status='old',form='formatted')
    read(fid,*) num_of_layer
    kmin=kdum+1
    kmax=kdum+num_of_layer
    kall=kdum+num_of_layer+kdum
    allocate(z_c(kall))
    allocate(z_h(kall))
    do k = kmin, kmax+1
       read(fid,*) z_h(k)
    enddo
    z_h(kmin-1) = z_h(kmin) - ( z_h(kmin+1) - z_h(kmin) )
    do k= kmin-1, kmax
       z_c(k) = z_h(k) + ( z_h(k+1) - z_h(k) )*0.5D0
    enddo
    z_c(kmax+1) = z_h(kmax+1) + ( z_h(kmax+1) - z_h(kmax) )*0.5D0
    close(fid)
  end subroutine mk_layer_given
  !-----------------------------------------------------------------------------
  subroutine output_layer( outfname )
    implicit none

    character(len=*) :: outfname
    integer :: fid=10
    integer :: k
    open(fid,file=trim(outfname),form='unformatted')
    write(fid) num_of_layer
    write(fid) z_c
    write(fid) z_h
    close(fid)
    
    do k = 1, kall
       write(*,*) k, z_h(k), z_c(k)
    enddo

  end subroutine output_layer
  !-----------------------------------------------------------------------------
end program prg_mkvlayer
!-------------------------------------------------------------------------------
