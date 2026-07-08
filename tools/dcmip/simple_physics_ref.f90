program simple_physics_ref
  !-----------------------------------------------------------------------------
  ! Reference driver for the DCMIP simple-physics package.
  !
  ! Builds a controlled multi-column atmosphere, calls SIMPLE_PHYSICS (the
  ! UNMODIFIED nicamdc share/dcmip/simple_physics_v6.f90) for several config
  ! combinations, and dumps inputs + outputs at es24.16 to a self-describing
  ! text file. The numpy port (pynicamdc/nhm/forcing/simple_physics.py) is
  ! validated against this file, column-by-column, level-by-level.
  !
  ! Build/run: tools/dcmip/build_and_run.sh
  !-----------------------------------------------------------------------------
  use mod_simple_physics, only: simple_physics
  implicit none

  integer, parameter :: r8 = selected_real_kind(12)

  integer, parameter :: pcols = 5     ! columns (5 latitudes)
  integer, parameter :: pver  = 30    ! model levels (top-down: 1=top, pver=surface)

  ! --- physical constants (mirror simple_physics_v6.f90 for the q-from-RH setup) ---
  real(r8), parameter :: gravit = 9.80616_r8
  real(r8), parameter :: rair   = 287.0_r8
  real(r8), parameter :: cpair  = 1.0045e3_r8
  real(r8), parameter :: latvap = 2.5e6_r8
  real(r8), parameter :: rh2o   = 461.5_r8
  real(r8), parameter :: epsilo = rair/rh2o
  real(r8), parameter :: T0c    = 273.16_r8
  real(r8), parameter :: e0     = 610.78_r8
  real(r8), parameter :: p0     = 100000.0_r8
  real(r8), parameter :: ptop   = 200.0_r8     ! model top pressure (Pa)
  real(r8), parameter :: pi     = 3.14159265358979323846_r8

  ! --- shared input state (top-down) ---
  real(r8) :: lat  (pcols)
  real(r8) :: t0   (pcols,pver),  q0   (pcols,pver)
  real(r8) :: u0   (pcols,pver),  v0   (pcols,pver)
  real(r8) :: pmid0(pcols,pver),  pint0(pcols,pver+1)
  real(r8) :: pdel0(pcols,pver),  rpdel0(pcols,pver)
  real(r8) :: ps0  (pcols)

  ! --- per-call working copies (simple_physics mutates in place) ---
  real(r8) :: t(pcols,pver), q(pcols,pver), u(pcols,pver), v(pcols,pver)
  real(r8) :: pmid(pcols,pver), pint(pcols,pver+1), pdel(pcols,pver), rpdel(pcols,pver)
  real(r8) :: ps(pcols), precl(pcols)

  integer  :: i, k
  real(r8) :: sig, wfac, rh, qsat, latdeg(pcols)
  integer, parameter :: fid = 21

  !---------------------------------------------------------------------------
  ! Build the reference atmosphere
  !---------------------------------------------------------------------------
  latdeg = (/ -60.0_r8, -30.0_r8, 0.0_r8, 30.0_r8, 60.0_r8 /)
  do i = 1, pcols
     lat(i) = latdeg(i) * pi / 180.0_r8
     ps0(i) = 100000.0_r8
  end do

  ! pressure structure: sigma interfaces top-down, pint(1)=ptop, pint(pver+1)=ps
  do i = 1, pcols
     do k = 1, pver+1
        sig = real(k-1,r8) / real(pver,r8)               ! 0 at top, 1 at surface
        pint0(i,k) = ptop + (ps0(i)-ptop) * sig
     end do
     do k = 1, pver
        pmid0 (i,k) = 0.5_r8*(pint0(i,k)+pint0(i,k+1))
        pdel0 (i,k) = pint0(i,k+1) - pint0(i,k)
        rpdel0(i,k) = 1.0_r8 / pdel0(i,k)
     end do
  end do

  ! temperature: ~200 K aloft -> ~300 K near surface (monotone in pressure)
  do i = 1, pcols
     do k = 1, pver
        t0(i,k) = 200.0_r8 + 100.0_r8 * (pmid0(i,k)/ps0(i))**0.5_r8
     end do
  end do

  ! winds: per-column factor so some columns have |wind|<20 (Cd0+Cd1*w) and
  ! some >20 (Cm branch). Stronger near the surface.
  do i = 1, pcols
     wfac = 0.3_r8 + 0.4_r8*real(i-1,r8)   ! 0.3,0.7,1.1,1.5,1.9
     do k = 1, pver
        u0(i,k) = 40.0_r8 * wfac * (pmid0(i,k)/ps0(i))
        v0(i,k) =  5.0_r8 * wfac * (pmid0(i,k)/ps0(i))
     end do
  end do

  ! specific humidity: RH profile, supersaturated (RH>1) in a lower-mid band
  ! so the large-scale precip branch fires; dry aloft.
  do i = 1, pcols
     do k = 1, pver
        qsat = epsilo*e0/pmid0(i,k)*exp(-latvap/rh2o*((1.0_r8/t0(i,k))-1.0_r8/T0c))
        if ( pmid0(i,k) > 60000.0_r8 .and. pmid0(i,k) < 90000.0_r8 ) then
           rh = 1.10_r8            ! supersaturated band -> triggers condensation
        else
           rh = 0.50_r8 * (pmid0(i,k)/ps0(i))
        end if
        q0(i,k) = max(1.0e-8_r8, rh * qsat)
     end do
  end do

  !---------------------------------------------------------------------------
  ! Dump shared inputs, then run each config and dump outputs
  !---------------------------------------------------------------------------
  open(fid, file='ref_simple_physics.txt', status='replace', action='write')
  write(fid,'(A)') '# simple_physics reference dump'
  write(fid,'(A,2(1x,I0))') 'META pcols pver', pcols, pver
  write(fid,'(A,1x,ES24.16)') 'META dtime', 1200.0_r8

  call dump1('lat',   lat,   pcols)
  call dump1('ps',    ps0,   pcols)
  call dump2('pmid',  pmid0, pcols, pver)
  call dump2i('pint', pint0, pcols, pver+1)
  call dump2('pdel',  pdel0, pcols, pver)
  call dump2('rpdel', rpdel0,pcols, pver)
  call dump2('t_in',  t0,    pcols, pver)
  call dump2('q_in',  q0,    pcols, pver)
  call dump2('u_in',  u0,    pcols, pver)
  call dump2('v_in',  v0,    pcols, pver)

  ! config A: precip on, no Bryan PBL, const SST (test=0)
  call run_and_dump('A_rj_noBryan_test0', 0, .true.,  .false., .false., 1)
  ! config B: precip on, Bryan PBL zi-branch, const SST
  call run_and_dump('B_rj_Bryan_test0',   0, .true.,  .true.,  .false., 1)
  ! config C: precip on, no Bryan, lat-dependent SST (test=1)
  call run_and_dump('C_rj_noBryan_test1', 1, .true.,  .false., .false., 1)
  ! config D: precip OFF (isolate surface flux + PBL)
  call run_and_dump('D_noprecip_test0',   0, .false., .false., .false., 1)
  ! config E: moist Held-Suarez SST branch (use_HS=T, MITC=1)
  call run_and_dump('E_useHS_mitc1',      0, .true.,  .false., .true.,  1)

  close(fid)
  write(*,*) 'wrote ref_simple_physics.txt'

contains

  subroutine reset_state()
    t = t0; q = q0; u = u0; v = v0
    pmid = pmid0; pint = pint0; pdel = pdel0; rpdel = rpdel0; ps = ps0
  end subroutine reset_state

  subroutine run_and_dump(name, test, rj, bryan, useHS, mitc)
    character(*), intent(in) :: name
    integer,      intent(in) :: test, mitc
    logical,      intent(in) :: rj, bryan, useHS
    call reset_state()
    call simple_physics(pcols, pver, 1200.0_r8, lat, t, q, u, v, &
                        pmid, pint, pdel, rpdel, ps, precl, &
                        test, rj, bryan, useHS, mitc)
    write(fid,'(A,1x,A)') 'CONFIG', name
    call dump2('t_out',  t, pcols, pver)
    call dump2('q_out',  q, pcols, pver)
    call dump2('u_out',  u, pcols, pver)
    call dump2('v_out',  v, pcols, pver)
    call dump1('precl',  precl, pcols)
  end subroutine run_and_dump

  subroutine dump1(name, a, n)
    character(*), intent(in) :: name
    integer,      intent(in) :: n
    real(r8),     intent(in) :: a(n)
    integer :: j
    write(fid,'(A,1x,A,1x,I0)') 'ARRAY', name, n
    do j = 1, n
       write(fid,'(ES24.16)') a(j)
    end do
  end subroutine dump1

  subroutine dump2(name, a, nc, nl)
    character(*), intent(in) :: name
    integer,      intent(in) :: nc, nl
    real(r8),     intent(in) :: a(nc,nl)
    integer :: ic, il
    write(fid,'(A,1x,A,1x,I0)') 'ARRAY', name, nc*nl
    do ic = 1, nc          ! column-major flatten: column ic, all levels
       do il = 1, nl
          write(fid,'(ES24.16)') a(ic,il)
       end do
    end do
  end subroutine dump2

  subroutine dump2i(name, a, nc, nl)   ! interface array (nl = pver+1)
    character(*), intent(in) :: name
    integer,      intent(in) :: nc, nl
    real(r8),     intent(in) :: a(nc,nl)
    call dump2(name, a, nc, nl)
  end subroutine dump2i

end program simple_physics_ref
