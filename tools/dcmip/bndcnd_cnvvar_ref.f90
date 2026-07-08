program bndcnd_cnvvar_ref
  !-----------------------------------------------------------------------------
  ! Reference driver for the DCMIP prerequisite kernels:
  !   BNDCND_pre_sfc  (full NICAM mod_bndcnd.f90, Lagrange z-extrap + hydrostatic)
  !   cnvvar_vh2uv    (nicamdc mod_cnvvar.f90, vh->uv metric projection)
  !   cnvvar_uv2vh    (nicamdc mod_cnvvar.f90, uv->vh de-scale + projection)
  !
  ! The kernels below are VERBATIM copies of the reference inner loops, on flat
  ! (ij) horizontal + (k) vertical, with synthetic inputs. Golden for the numpy
  ! ports (mod_bndcnd.BNDCND_pre_sfc, mod_cnvvar.cnvvar_vh2uv/uv2vh).
  !
  !   ./bndcnd_cnvvar_ref.x [ijdim] [kdim] [outfile]
  !-----------------------------------------------------------------------------
  implicit none
  integer, parameter :: DP = selected_real_kind(12)
  real(DP), parameter :: GRAV = 9.80616_DP
  real(DP), parameter :: pi = 3.14159265358979323846_DP

  integer :: ijdim, kdim, kmin
  real(DP), allocatable :: rho(:,:), pre(:,:), zg(:,:), z_srf(:), rho_srf(:), pre_srf(:)
  real(DP), allocatable :: vx(:,:), vy(:,:), vz(:,:), u(:,:), v(:,:)
  real(DP), allocatable :: uc(:,:), vc(:,:), vx2(:,:), vy2(:,:), vz2(:,:)
  real(DP), allocatable :: lat(:), lon(:)
  real(DP), allocatable :: IX(:), IY(:), IZ(:), JX(:), JY(:), JZ(:)

  integer :: ij, k
  real(DP) :: sig, z_ks, z_k1, z_k2, z_k3, coslat, sw, uu, vv, u0c, v0c
  real(DP) :: sinl, cosl, sino, coso
  character(len=256) :: outfile, argbuf
  integer, parameter :: fid = 23

  ijdim = 6; kdim = 12; outfile = 'ref_bndcnd_cnvvar.txt'
  if ( command_argument_count() >= 1 ) then
     call get_command_argument(1, argbuf); read(argbuf,*) ijdim
  end if
  if ( command_argument_count() >= 2 ) then
     call get_command_argument(2, argbuf); read(argbuf,*) kdim
  end if
  if ( command_argument_count() >= 3 ) call get_command_argument(3, outfile)
  kmin = 2   ! 1-based lowest full level (halo at 1)

  allocate(rho(ijdim,kdim), pre(ijdim,kdim), zg(ijdim,kdim))
  allocate(z_srf(ijdim), rho_srf(ijdim), pre_srf(ijdim))
  allocate(vx(ijdim,kdim), vy(ijdim,kdim), vz(ijdim,kdim), u(ijdim,kdim), v(ijdim,kdim))
  allocate(uc(ijdim,kdim), vc(ijdim,kdim), vx2(ijdim,kdim), vy2(ijdim,kdim), vz2(ijdim,kdim))
  allocate(lat(ijdim), lon(ijdim), IX(ijdim), IY(ijdim), IZ(ijdim), JX(ijdim), JY(ijdim), JZ(ijdim))

  ! ---- synthetic inputs ----
  do ij = 1, ijdim
     lat(ij) = ( -75.0_DP + 150.0_DP*real(ij-1,DP)/real(ijdim-1,DP) ) * pi/180.0_DP
     lon(ij) = ( 25.0_DP*real(ij-1,DP) ) * pi/180.0_DP
     sinl = sin(lat(ij)); cosl = cos(lat(ij)); sino = sin(lon(ij)); coso = cos(lon(ij))
     ! east/north orthonormal metric vectors
     IX(ij) = -sino;       IY(ij) =  coso;      IZ(ij) = 0.0_DP
     JX(ij) = -sinl*coso;  JY(ij) = -sinl*sino; JZ(ij) = cosl
     z_srf(ij) = 5.0_DP + 3.0_DP*real(ij-1,DP)        ! surface below z(kmin)
     do k = 1, kdim
        sig = real(k-kmin,DP) / real(kdim-kmin,DP)
        zg (ij,k) = 100.0_DP + 900.0_DP*max(sig,0.0_DP) + 50.0_DP*real(k,DP)
        pre(ij,k) = 100000.0_DP * exp(-3.0_DP*max(sig,0.0_DP))
        rho(ij,k) = pre(ij,k) / (287.0_DP * (300.0_DP - 60.0_DP*max(sig,0.0_DP)))
        u0c = 25.0_DP*(1.0_DP-max(sig,0.0_DP)) + 3.0_DP
        v0c = 6.0_DP*(1.0_DP-max(sig,0.0_DP))
        vx(ij,k) = u0c*IX(ij) + v0c*JX(ij)
        vy(ij,k) = u0c*IY(ij) + v0c*JY(ij)
        vz(ij,k) = u0c*IZ(ij) + v0c*JZ(ij)
     end do
  end do

  ! ============ BNDCND_pre_sfc (verbatim) ============
  do ij = 1, ijdim
     z_ks = z_srf(ij)
     z_k1 = zg(ij,kmin  ); z_k2 = zg(ij,kmin+1); z_k3 = zg(ij,kmin+2)
     rho_srf(ij) = ((z_ks-z_k2)*(z_ks-z_k3))/((z_k1-z_k2)*(z_k1-z_k3))*rho(ij,kmin  ) &
                 + ((z_ks-z_k1)*(z_ks-z_k3))/((z_k2-z_k1)*(z_k2-z_k3))*rho(ij,kmin+1) &
                 + ((z_ks-z_k1)*(z_ks-z_k2))/((z_k3-z_k1)*(z_k3-z_k2))*rho(ij,kmin+2)
     pre_srf(ij) = pre(ij,kmin) + 0.5_DP*(rho_srf(ij)+rho(ij,kmin))*GRAV*(z_k1-z_ks)
  end do

  ! ============ cnvvar_vh2uv (verbatim, withcos=.false.) ============
  do k = 1, kdim
     do ij = 1, ijdim
        coslat = 1.0_DP
        u(ij,k) = ( vx(ij,k)*IX(ij) + vy(ij,k)*IY(ij) + vz(ij,k)*IZ(ij) ) * coslat
        v(ij,k) = ( vx(ij,k)*JX(ij) + vy(ij,k)*JY(ij) + vz(ij,k)*JZ(ij) ) * coslat
     end do
  end do

  ! ============ cnvvar_uv2vh (verbatim; feed ucos=u,vcos=v with coslat=cos(lat)) ============
  do k = 1, kdim
     do ij = 1, ijdim
        coslat = cos(lat(ij))
        sw = 0.5_DP + sign(0.5_DP, -abs(coslat))
        uu = u(ij,k) * ( 1.0_DP - sw ) / ( coslat - sw )
        vv = v(ij,k) * ( 1.0_DP - sw ) / ( coslat - sw )
        vx2(ij,k) = uu*IX(ij) + vv*JX(ij)
        vy2(ij,k) = uu*IY(ij) + vv*JY(ij)
        vz2(ij,k) = uu*IZ(ij) + vv*JZ(ij)
     end do
  end do

  open(fid, file=trim(outfile), status='replace', action='write')
  write(fid,'(A)') '# bndcnd/cnvvar reference dump'
  write(fid,'(A,2(1x,I0))') 'META ijdim kdim', ijdim, kdim
  write(fid,'(A,1x,I0)') 'META kmin_1based', kmin
  write(fid,'(A,1x,ES24.16)') 'META GRAV', GRAV
  call d1('lat',lat); call d1('lon',lon)
  call d1('IX',IX); call d1('IY',IY); call d1('IZ',IZ)
  call d1('JX',JX); call d1('JY',JY); call d1('JZ',JZ)
  call d1('z_srf',z_srf)
  call d2('rho',rho); call d2('pre',pre); call d2('zg',zg)
  call d2('vx',vx); call d2('vy',vy); call d2('vz',vz)
  ! outputs
  call d1('rho_srf',rho_srf); call d1('pre_srf',pre_srf)
  call d2('u',u); call d2('v',v)
  call d2('vx2',vx2); call d2('vy2',vy2); call d2('vz2',vz2)
  close(fid)
  write(*,*) 'wrote ', trim(outfile)

contains
  subroutine d1(nm, a)
    character(*), intent(in) :: nm
    real(DP), intent(in) :: a(:)
    integer :: j
    write(fid,'(A,1x,A,1x,I0)') 'ARRAY', nm, size(a)
    do j = 1, size(a)
       write(fid,'(ES24.16)') a(j)
    end do
  end subroutine d1
  subroutine d2(nm, a)
    character(*), intent(in) :: nm
    real(DP), intent(in) :: a(:,:)
    integer :: ic, il
    write(fid,'(A,1x,A,1x,I0)') 'ARRAY', nm, size(a)
    do ic = 1, size(a,1)
       do il = 1, size(a,2)
          write(fid,'(ES24.16)') a(ic,il)
       end do
    end do
  end subroutine d2
end program bndcnd_cnvvar_ref
