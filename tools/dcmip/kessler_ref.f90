program kessler_ref
  !-----------------------------------------------------------------------------
  ! Reference driver for the KESSLER warm-rain microphysics column scheme.
  ! Calls the UNMODIFIED nicamdc kessler.f90 (SUBROUTINE KESSLER) per column on
  ! synthetic profiles, and writes BOTH the inputs and the outputs so the numpy
  ! test (test_kessler.py) runs kessler.py on identical inputs and compares.
  !
  !   ./kessler_ref.x [ncol] [nz] [outfile]
  ! Build/run: tools/dcmip/build_kessler_ref.sh
  !-----------------------------------------------------------------------------
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8), parameter :: Rd = 287.0_r8, Cp = 1004.5_r8, p0 = 100000.0_r8
  real(r8), parameter :: grav = 9.80616_r8, kappa = Rd/Cp
  real(r8), parameter :: dt = 1200.0_r8

  integer :: ncol, nz, c, k
  real(r8), allocatable :: theta(:,:), qv(:,:), qc(:,:), qr(:,:)
  real(r8), allocatable :: rho(:,:), pk(:,:), z(:,:)
  real(r8), allocatable :: th0(:,:), qv0(:,:), qc0(:,:), qr0(:,:)
  real(r8), allocatable :: precl(:)
  real(r8) :: tt(1024), qvv(1024), qcc(1024), qrr(1024)
  real(r8) :: rr(1024), pp(1024), zz(1024), pl
  real(r8) :: zt, pres, temp, tsurf, lapse, rhc, qsat, es, rainscale
  character(len=256) :: outfile, argbuf
  integer, parameter :: fid = 23

  call get_command_argument(1, argbuf); read(argbuf,*) ncol
  call get_command_argument(2, argbuf); read(argbuf,*) nz
  call get_command_argument(3, outfile)

  allocate(theta(ncol,nz), qv(ncol,nz), qc(ncol,nz), qr(ncol,nz))
  allocate(rho(ncol,nz), pk(ncol,nz), z(ncol,nz))
  allocate(th0(ncol,nz), qv0(ncol,nz), qc0(ncol,nz), qr0(ncol,nz), precl(ncol))

  ! --- synthetic profiles (surface -> top), varied per column ---
  do c = 1, ncol
     tsurf     = 300.0_r8 - real(c-1,r8)*2.0_r8        ! surface T per column
     lapse     = 6.5e-3_r8                             ! K/m
     rainscale = 0.2_r8 + 0.4_r8*real(c-1,r8)          ! scales qc/qr -> varies rainsplit
     do k = 1, nz
        zt   = 30000.0_r8 * real(k-1,r8)/real(nz-1,r8) ! 0..30 km
        temp = max(tsurf - lapse*zt, 200.0_r8)
        pres = p0 * exp(-grav*zt/(Rd*260.0_r8))        ! isothermal-ish hydrostatic
        z (c,k)  = zt + 50.0_r8
        pk(c,k)  = (pres/p0)**kappa
        theta(c,k) = temp / pk(c,k)
        rho(c,k) = pres/(Rd*temp)
        ! saturation mixing ratio (Teten) for a plausible moisture profile
        es   = 610.78_r8*exp(17.27_r8*(temp-273.16_r8)/(temp-35.86_r8))
        qsat = 0.622_r8*es/max(pres-es,1.0_r8)
        rhc  = max(0.2_r8, 0.95_r8 - 0.7_r8*zt/30000.0_r8)  ! moist below, dry aloft
        qv(c,k) = rhc*qsat
        ! cloud + rain concentrated in the lower/mid troposphere
        if ( zt < 8000.0_r8 ) then
           qc(c,k) = rainscale*1.5e-3_r8*(1.0_r8 - zt/8000.0_r8)
           qr(c,k) = rainscale*0.8e-3_r8*(1.0_r8 - zt/8000.0_r8)
        else
           qc(c,k) = 0.0_r8
           qr(c,k) = 0.0_r8
        end if
     end do
  end do

  th0 = theta; qv0 = qv; qc0 = qc; qr0 = qr

  ! --- call the real KESSLER per column ---
  do c = 1, ncol
     tt(1:nz)  = theta(c,:); qvv(1:nz) = qv(c,:)
     qcc(1:nz) = qc(c,:);    qrr(1:nz) = qr(c,:)
     rr(1:nz)  = rho(c,:);   pp(1:nz)  = pk(c,:); zz(1:nz) = z(c,:)
     call KESSLER(tt(1:nz), qvv(1:nz), qcc(1:nz), qrr(1:nz), rr(1:nz), &
                  pp(1:nz), dt, zz(1:nz), nz, pl)
     theta(c,:) = tt(1:nz); qv(c,:) = qvv(1:nz)
     qc(c,:) = qcc(1:nz);   qr(c,:) = qrr(1:nz); precl(c) = pl
  end do

  ! --- dump inputs + outputs ---
  open(fid, file=trim(outfile), status='replace', action='write')
  write(fid,'(2I8,ES26.17)') ncol, nz, dt
  call wr(th0);  call wr(qv0); call wr(qc0); call wr(qr0)
  call wr(rho);  call wr(pk);  call wr(z)
  call wr(theta); call wr(qv); call wr(qc); call wr(qr)
  do c = 1, ncol
     write(fid,'(ES26.17)') precl(c)
  end do
  close(fid)
  write(*,*) 'OK -> ', trim(outfile)

contains
  subroutine wr(a)
    real(r8), intent(in) :: a(:,:)
    integer :: ic, ik
    do ic = 1, size(a,1)
       do ik = 1, size(a,2)
          write(fid,'(ES26.17)') a(ic,ik)
       end do
    end do
  end subroutine wr
end program kessler_ref
