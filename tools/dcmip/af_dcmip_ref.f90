program af_dcmip_ref
  !-----------------------------------------------------------------------------
  ! Reference driver for the AF_dcmip GLUE (USE_SimpleMicrophys path).
  !
  ! The transformation block below is a VERBATIM copy of mod_af_dcmip.f90
  ! L414-513 (level flip kk=kmax-k+1, vh->uv, pint construction, the real
  ! simple_physics call, uv->vh back-projection, fq/fe tendency assembly),
  ! adapted only to be standalone (hardcoded constants, synthetic inputs).
  ! It calls the UNMODIFIED simple_physics_v6.f90, so it is a faithful golden
  ! for the numpy mod_af_dcmip.py AF_dcmip port.
  !
  !   ./af_dcmip_ref.x [ijdim] [vlayer] [outfile]
  !
  ! Build/run: tools/dcmip/build_af_dcmip_ref.sh
  !-----------------------------------------------------------------------------
  use mod_simple_physics, only: simple_physics
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: DP = r8

  ! model constants (mirror mod_const / mod_runconf for the standalone test)
  real(r8), parameter :: Rdry  = 287.0_r8
  real(r8), parameter :: CPdry = 1004.5_r8
  real(r8), parameter :: CVdry = CPdry - Rdry
  real(r8), parameter :: PRE00 = 100000.0_r8
  real(r8), parameter :: CVW_QV = 1845.6_r8
  real(r8), parameter :: CVW_QC = 4186.0_r8
  real(r8), parameter :: CVW_QR = 4186.0_r8
  real(r8), parameter :: pi = 3.14159265358979323846_r8
  real(DP), parameter :: dt = 1200.0_DP

  integer :: ijdim, vlayer, kdim, kmin, kmax
  integer :: I_QV, I_QC, I_QR

  ! packed per-region model arrays (BOTTOM-UP, halo at 1 and kdim)
  real(r8), allocatable :: lat(:), lon(:), pre_sfc(:)
  real(r8), allocatable :: ix(:), iy(:), iz(:), jx(:), jy(:), jz(:)
  real(r8), allocatable :: alt(:,:), alth(:,:), rho(:,:), pre(:,:), tem(:,:)
  real(r8), allocatable :: vx(:,:), vy(:,:), vz(:,:), ein(:,:)
  real(r8), allocatable :: q(:,:,:)
  real(r8), allocatable :: fvx(:,:), fvy(:,:), fvz(:,:), fe(:,:)
  real(r8), allocatable :: fq(:,:,:), precip(:)

  ! column workspace (top-down) for simple_physics
  real(r8), allocatable :: t(:,:), qvv(:,:), u(:,:), v(:,:)
  real(r8), allocatable :: pmid(:,:), pint(:,:), pdel(:,:), rpdel(:,:), ps(:)
  real(DP), allocatable :: precip2(:), lat_DP(:)
  real(r8), allocatable :: qd(:), qvc(:), qcc(:), qrc(:), cv(:)

  ! flags for this reference
  logical :: SM_LargeScaleCond, SM_PBL_Bryan, USE_HeldSuarez, SM_Latdepend_SST
  integer :: SM_MITC_SST_TYPE, test
  character(len=16) :: RAIN_TYPE

  integer :: ij, k, kk, nq, ntrc
  real(r8) :: sig, u0c, v0c, sinl, cosl, sino, coso, z
  character(len=256) :: outfile, argbuf
  integer, parameter :: fid = 22

  ! ---- args ----
  ijdim = 5; vlayer = 30; outfile = 'ref_af_dcmip.txt'
  if ( command_argument_count() >= 1 ) then
     call get_command_argument(1, argbuf); read(argbuf,*) ijdim
  end if
  if ( command_argument_count() >= 2 ) then
     call get_command_argument(2, argbuf); read(argbuf,*) vlayer
  end if
  if ( command_argument_count() >= 3 ) call get_command_argument(3, outfile)

  kmin = 2; kmax = vlayer + 1; kdim = vlayer + 2
  ntrc = 3; I_QV = 1; I_QC = 2; I_QR = 3    ! 1-based tracer indices

  allocate(lat(ijdim), lon(ijdim), pre_sfc(ijdim))
  allocate(ix(ijdim), iy(ijdim), iz(ijdim), jx(ijdim), jy(ijdim), jz(ijdim))
  allocate(alt(ijdim,kdim), alth(ijdim,kdim), rho(ijdim,kdim), pre(ijdim,kdim), tem(ijdim,kdim))
  allocate(vx(ijdim,kdim), vy(ijdim,kdim), vz(ijdim,kdim), ein(ijdim,kdim))
  allocate(q(ijdim,kdim,ntrc))
  allocate(fvx(ijdim,kdim), fvy(ijdim,kdim), fvz(ijdim,kdim), fe(ijdim,kdim))
  allocate(fq(ijdim,kdim,ntrc), precip(ijdim))
  allocate(t(ijdim,vlayer), qvv(ijdim,vlayer), u(ijdim,vlayer), v(ijdim,vlayer))
  allocate(pmid(ijdim,vlayer), pint(ijdim,vlayer+1), pdel(ijdim,vlayer), rpdel(ijdim,vlayer), ps(ijdim))
  allocate(precip2(ijdim), lat_DP(ijdim))
  allocate(qd(vlayer), qvc(vlayer), qcc(vlayer), qrc(vlayer), cv(vlayer))

  ! ---- synthetic inputs ----
  do ij = 1, ijdim
     lat(ij) = ( -60.0_r8 + 120.0_r8*real(ij-1,r8)/real(ijdim-1,r8) ) * pi/180.0_r8
     lon(ij) = ( 30.0_r8 * real(ij-1,r8) ) * pi/180.0_r8
     pre_sfc(ij) = 100500.0_r8
     sinl = sin(lat(ij)); cosl = cos(lat(ij)); sino = sin(lon(ij)); coso = cos(lon(ij))
     ! local east/north orthonormal basis
     ix(ij) = -sino;         iy(ij) =  coso;        iz(ij) = 0.0_r8
     jx(ij) = -sinl*coso;    jy(ij) = -sinl*sino;   jz(ij) = cosl
  end do

  q(:,:,:) = 0.0_r8
  do ij = 1, ijdim
     do k = 1, kdim
        ! bottom-up: k=kmin bottom (high p), k=kmax top (low p)
        sig = real(k-kmin,r8) / real(kmax-kmin,r8)     ! 0 at bottom .. 1 at top
        sig = min(max(sig,0.0_r8),1.0_r8)
        pre(ij,k) = 100000.0_r8 * exp(-3.5_r8*sig)      ! ~1000 hPa -> ~30 hPa
        z          = 8000.0_r8 * sig                    ! height
        alt (ij,k) = z
        alth(ij,k) = z + 120.0_r8
        tem(ij,k) = 300.0_r8 - 80.0_r8*sig
        rho(ij,k) = pre(ij,k) / (Rdry*tem(ij,k))
        ein(ij,k) = CVdry * tem(ij,k)
        ! tangential wind: V = u0*e + v0*n  (so vh<->uv round trips)
        u0c = 20.0_r8*(1.0_r8-sig) + 5.0_r8
        v0c = 4.0_r8*(1.0_r8-sig)
        vx(ij,k) = u0c*ix(ij) + v0c*jx(ij)
        vy(ij,k) = u0c*iy(ij) + v0c*jy(ij)
        vz(ij,k) = u0c*iz(ij) + v0c*jz(ij)
        ! moisture: supersaturated-ish band in the lower troposphere
        if ( sig < 0.4_r8 ) then
           q(ij,k,I_QV) = 0.015_r8*(1.0_r8-sig)
        else
           q(ij,k,I_QV) = 0.002_r8
        end if
        q(ij,k,I_QC) = 0.0005_r8*(1.0_r8-sig)
        q(ij,k,I_QR) = 0.0002_r8*(1.0_r8-sig)
     end do
  end do

  ! scheme config for this reference
  SM_LargeScaleCond = .true.
  SM_PBL_Bryan      = .false.
  USE_HeldSuarez    = .false.
  SM_Latdepend_SST  = .true.
  SM_MITC_SST_TYPE  = 1

  open(fid, file=trim(outfile), status='replace', action='write')
  write(fid,'(A)') '# af_dcmip glue reference dump'
  write(fid,'(A,3(1x,I0))') 'META ijdim vlayer kdim', ijdim, vlayer, kdim
  write(fid,'(A,2(1x,I0))') 'META kmin kmax', kmin, kmax
  write(fid,'(A,3(1x,I0))') 'META ntrc I_QV_0based', ntrc, I_QV-1
  write(fid,'(A,1x,ES24.16)') 'META dt', dt
  write(fid,'(A,4(1x,ES24.16))') 'META Rdry CPdry CVdry PRE00', Rdry, CPdry, CVdry, PRE00
  write(fid,'(A,3(1x,ES24.16))') 'META CVW_QV CVW_QC CVW_QR', CVW_QV, CVW_QC, CVW_QR

  call dump1('lat', lat, ijdim); call dump1('lon', lon, ijdim)
  call dump1('pre_sfc', pre_sfc, ijdim)
  call dump1('ix', ix, ijdim); call dump1('iy', iy, ijdim); call dump1('iz', iz, ijdim)
  call dump1('jx', jx, ijdim); call dump1('jy', jy, ijdim); call dump1('jz', jz, ijdim)
  call dump2('alt', alt, ijdim, kdim);  call dump2('alth', alth, ijdim, kdim)
  call dump2('rho', rho, ijdim, kdim);  call dump2('pre', pre, ijdim, kdim)
  call dump2('tem', tem, ijdim, kdim);  call dump2('ein', ein, ijdim, kdim)
  call dump2('vx', vx, ijdim, kdim); call dump2('vy', vy, ijdim, kdim); call dump2('vz', vz, ijdim, kdim)
  call dump3('q', q, ijdim, kdim, ntrc)

  ! =========================================================================
  ! VERBATIM AF_dcmip SimpleMicrophys transform (mod_af_dcmip.f90 L414-513)
  ! =========================================================================
  fvx(:,:) = 0.0_r8; fvy(:,:) = 0.0_r8; fvz(:,:) = 0.0_r8
  fe (:,:) = 0.0_r8; fq(:,:,:) = 0.0_r8; precip(:) = 0.0_r8; precip2(:) = 0.0_DP

  if ( SM_Latdepend_SST ) then
     test = 1
  else
     test = 0
  endif
  lat_DP(:) = real(lat(:),kind=DP)

  do k = 1, vlayer
     kk = kmax - k + 1
     t   (:,k) = tem(:,kk)
     qvv (:,k) = q  (:,kk,I_QV)
     u   (:,k) = vx(:,kk)*ix(:) + vy(:,kk)*iy(:) + vz(:,kk)*iz(:)
     v   (:,k) = vx(:,kk)*jx(:) + vy(:,kk)*jy(:) + vz(:,kk)*jz(:)
     pmid(:,k) = pre(:,kk)
  enddo
  pint(:,1) = 0.0_r8
  do k = 2, vlayer
     kk = kmax - k + 1
     pint(:,k) = pre(:,kk) * exp( log( pre(:,kk+1) / pre(:,kk) ) &
                                * (alth(:,kk+1)-alt(:,kk)) / (alt(:,kk+1)-alt(:,kk)) )
  enddo
  pint(:,vlayer+1) = pre_sfc(:)
  do k = 1, vlayer
     pdel (:,k) = pint(:,k+1) - pint(:,k)
     rpdel(:,k) = 1.0_r8 / pdel(:,k)
  enddo
  ps(:) = pre_sfc(:)

  call simple_physics( ijdim, vlayer, dt, lat_DP(:), t(:,:), qvv(:,:), u(:,:), v(:,:), &
                       pmid(:,:), pint(:,:), pdel(:,:), rpdel(:,:), ps(:), precip2(:), &
                       test, SM_LargeScaleCond, SM_PBL_Bryan, USE_HeldSuarez, SM_MITC_SST_TYPE )

  do k = 1, vlayer
     kk = kmax - k + 1
     fvx(:,kk) = fvx(:,kk) + ( u(:,k)*ix(:) + v(:,k)*jx(:) - vx(:,kk) ) / dt
     fvy(:,kk) = fvy(:,kk) + ( u(:,k)*iy(:) + v(:,k)*jy(:) - vy(:,kk) ) / dt
     fvz(:,kk) = fvz(:,kk) + ( u(:,k)*iz(:) + v(:,k)*jz(:) - vz(:,kk) ) / dt
  enddo

  call run_tracer_energy('DRY')
  call dump_out('DRY')
  call run_tracer_energy('WARM')
  call dump_out('WARM')

  close(fid)
  write(*,*) 'wrote ', trim(outfile)

contains

  subroutine run_tracer_energy(rain)
    character(*), intent(in) :: rain
    fq(:,:,:) = 0.0_r8; fe(:,:) = 0.0_r8
    do k = 1, vlayer
       kk = kmax - k + 1
       do ij = 1, ijdim
          if ( rain == 'DRY' ) then
             qvc(k) = qvv(ij,k)
             qd (k) = 1.0_r8 - qvc(k)
             cv (k) = qd(k)*CVdry + qvc(k)*CVW_QV
          else
             qvc(k) = qvv(ij,k)
             qcc(k) = q(ij,kk,I_QC)
             qrc(k) = q(ij,kk,I_QR)
             qd (k) = 1.0_r8 - qvc(k) - qcc(k) - qrc(k)
             cv (k) = qd(k)*CVdry + qvc(k)*CVW_QV + qcc(k)*CVW_QC + qrc(k)*CVW_QR
          endif
          fq(ij,kk,I_QV) = fq(ij,kk,I_QV) + ( qvc(k) - q(ij,kk,I_QV) ) / dt
          fe(ij,kk)      = fe(ij,kk)      + ( cv(k)*t(ij,k) - ein(ij,kk) ) / dt
       enddo
    enddo
    precip(:) = precip2(:)
  end subroutine run_tracer_energy

  subroutine dump_out(name)
    character(*), intent(in) :: name
    write(fid,'(A,1x,A)') 'CONFIG', name
    call dump2('fvx', fvx, ijdim, kdim); call dump2('fvy', fvy, ijdim, kdim)
    call dump2('fvz', fvz, ijdim, kdim); call dump2('fe', fe, ijdim, kdim)
    call dump3('fq', fq, ijdim, kdim, ntrc)
    call dump1('precip', precip, ijdim)
  end subroutine dump_out

  subroutine dump1(nm, a, n)
    character(*), intent(in) :: nm
    integer, intent(in) :: n
    real(r8), intent(in) :: a(n)
    integer :: j
    write(fid,'(A,1x,A,1x,I0)') 'ARRAY', nm, n
    do j = 1, n
       write(fid,'(ES24.16)') a(j)
    end do
  end subroutine dump1

  subroutine dump2(nm, a, nc, nl)
    character(*), intent(in) :: nm
    integer, intent(in) :: nc, nl
    real(r8), intent(in) :: a(nc,nl)
    integer :: ic, il
    write(fid,'(A,1x,A,1x,I0)') 'ARRAY', nm, nc*nl
    do ic = 1, nc
       do il = 1, nl
          write(fid,'(ES24.16)') a(ic,il)
       end do
    end do
  end subroutine dump2

  subroutine dump3(nm, a, nc, nl, nt)
    character(*), intent(in) :: nm
    integer, intent(in) :: nc, nl, nt
    real(r8), intent(in) :: a(nc,nl,nt)
    integer :: ic, il, it
    write(fid,'(A,1x,A,1x,I0)') 'ARRAY', nm, nc*nl*nt
    do ic = 1, nc
       do il = 1, nl
          do it = 1, nt
             write(fid,'(ES24.16)') a(ic,il,it)
          end do
       end do
    end do
  end subroutine dump3

end program af_dcmip_ref
