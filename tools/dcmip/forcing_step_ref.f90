program forcing_step_ref
  !-----------------------------------------------------------------------------
  ! Reference driver for forcing_step PART A (compute+apply).
  !
  ! Verbatim copy of nicamdc mod_forcing_driver.f90 forcing_step L253-390 core:
  !   ein = rhoge/rhog ; pre_srf (single-layer hydrostatic) ; AF_dcmip transform
  !   (calling the UNMODIFIED simple_physics_v6.f90) ; tendency apply to
  !   rhog/rhogvx../rhoge and rhogq (UPDATE_TOT_DENS on NQW tracers).
  ! Golden for mod_forcing.Frc.forcing_step.
  !
  !   ./forcing_step_ref.x [ijdim] [vlayer] [outfile]
  !-----------------------------------------------------------------------------
  use mod_simple_physics, only: simple_physics
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: DP = r8

  real(r8), parameter :: Rdry  = 287.0_r8
  real(r8), parameter :: CPdry = 1004.5_r8
  real(r8), parameter :: CVdry = CPdry - Rdry
  real(r8), parameter :: PRE00 = 100000.0_r8
  real(r8), parameter :: CVW_QV = 1845.6_r8, CVW_QC = 4186.0_r8, CVW_QR = 4186.0_r8
  real(r8), parameter :: GRAV = 9.80665_r8
  real(r8), parameter :: pi = 3.14159265358979323846_r8
  real(DP), parameter :: DTL = 1200.0_DP

  integer :: ijdim, vlayer, kdim, kmin, kmax, I_QV, I_QC, I_QR, ntrc
  integer :: NQW_STR, NQW_END

  real(r8), allocatable :: lat(:), lon(:), pre_sfc(:), z_srf(:)
  real(r8), allocatable :: ix(:),iy(:),iz(:),jx(:),jy(:),jz(:)
  real(r8), allocatable :: alt(:,:),alth(:,:),rho(:,:),pre(:,:),tem(:,:)
  real(r8), allocatable :: vx(:,:),vy(:,:),vz(:,:),ein(:,:), gsgam2(:,:), gsgam2h(:,:)
  real(r8), allocatable :: q(:,:,:)
  ! prognostics
  real(r8), allocatable :: rhog(:,:),rhogvx(:,:),rhogvy(:,:),rhogvz(:,:),rhogw(:,:),rhoge(:,:)
  real(r8), allocatable :: rhogq(:,:,:)
  ! forcing tendencies + precip
  real(r8), allocatable :: fvx(:,:),fvy(:,:),fvz(:,:),fe(:,:),fq(:,:,:),precip(:), frhogq(:,:)
  ! column workspace
  real(r8), allocatable :: t(:,:),qvv(:,:),u(:,:),v(:,:),pmid(:,:),pint(:,:),pdel(:,:),rpdel(:,:),ps(:)
  real(DP), allocatable :: precip2(:), lat_DP(:)
  real(r8), allocatable :: qd(:),qvc(:),qcc(:),qrc(:),cv(:)

  logical :: SM_LargeScaleCond, SM_PBL_Bryan, USE_HeldSuarez, SM_Latdepend_SST
  integer :: SM_MITC_SST_TYPE, test, ij, k, kk, nq
  real(r8) :: sig, u0c, v0c, sinl, cosl, sino, coso, zz
  character(len=256) :: outfile, argbuf
  integer, parameter :: fid = 24

  ijdim = 5; vlayer = 30; outfile = 'ref_forcing_step.txt'
  if ( command_argument_count() >= 1 ) then
     call get_command_argument(1, argbuf); read(argbuf,*) ijdim
  end if
  if ( command_argument_count() >= 2 ) then
     call get_command_argument(2, argbuf); read(argbuf,*) vlayer
  end if
  if ( command_argument_count() >= 3 ) call get_command_argument(3, outfile)
  kmin = 2; kmax = vlayer+1; kdim = vlayer+2; ntrc = 3; I_QV=1; I_QC=2; I_QR=3

  allocate(lat(ijdim),lon(ijdim),pre_sfc(ijdim),z_srf(ijdim))
  allocate(ix(ijdim),iy(ijdim),iz(ijdim),jx(ijdim),jy(ijdim),jz(ijdim))
  allocate(alt(ijdim,kdim),alth(ijdim,kdim),rho(ijdim,kdim),pre(ijdim,kdim),tem(ijdim,kdim))
  allocate(vx(ijdim,kdim),vy(ijdim,kdim),vz(ijdim,kdim),ein(ijdim,kdim),gsgam2(ijdim,kdim),gsgam2h(ijdim,kdim))
  allocate(q(ijdim,kdim,ntrc))
  allocate(rhog(ijdim,kdim),rhogvx(ijdim,kdim),rhogvy(ijdim,kdim),rhogvz(ijdim,kdim),rhogw(ijdim,kdim),rhoge(ijdim,kdim))
  allocate(rhogq(ijdim,kdim,ntrc))
  allocate(fvx(ijdim,kdim),fvy(ijdim,kdim),fvz(ijdim,kdim),fe(ijdim,kdim),fq(ijdim,kdim,ntrc),precip(ijdim),frhogq(ijdim,kdim))
  allocate(t(ijdim,vlayer),qvv(ijdim,vlayer),u(ijdim,vlayer),v(ijdim,vlayer))
  allocate(pmid(ijdim,vlayer),pint(ijdim,vlayer+1),pdel(ijdim,vlayer),rpdel(ijdim,vlayer),ps(ijdim))
  allocate(precip2(ijdim),lat_DP(ijdim))
  allocate(qd(vlayer),qvc(vlayer),qcc(vlayer),qrc(vlayer),cv(vlayer))

  ! ---- synthetic inputs ----
  q(:,:,:) = 0.0_r8
  do ij = 1, ijdim
     lat(ij) = ( -60.0_r8 + 120.0_r8*real(ij-1,r8)/real(ijdim-1,r8) ) * pi/180.0_r8
     lon(ij) = ( 30.0_r8*real(ij-1,r8) ) * pi/180.0_r8
     z_srf(ij) = 10.0_r8 + 4.0_r8*real(ij-1,r8)
     sinl=sin(lat(ij)); cosl=cos(lat(ij)); sino=sin(lon(ij)); coso=cos(lon(ij))
     ix(ij)=-sino; iy(ij)=coso; iz(ij)=0.0_r8
     jx(ij)=-sinl*coso; jy(ij)=-sinl*sino; jz(ij)=cosl
     do k = 1, kdim
        sig = real(k-kmin,r8)/real(kmax-kmin,r8); sig=min(max(sig,0.0_r8),1.0_r8)
        pre(ij,k)=100000.0_r8*exp(-3.5_r8*sig)
        zz = 8000.0_r8*sig; alt(ij,k)=zz; alth(ij,k)=zz+120.0_r8
        tem(ij,k)=300.0_r8-80.0_r8*sig
        rho(ij,k)=pre(ij,k)/(Rdry*tem(ij,k))
        u0c=20.0_r8*(1.0_r8-sig)+5.0_r8; v0c=4.0_r8*(1.0_r8-sig)
        vx(ij,k)=u0c*ix(ij)+v0c*jx(ij); vy(ij,k)=u0c*iy(ij)+v0c*jy(ij); vz(ij,k)=u0c*iz(ij)+v0c*jz(ij)
        if (sig<0.4_r8) then; q(ij,k,I_QV)=0.015_r8*(1.0_r8-sig); else; q(ij,k,I_QV)=0.002_r8; end if
        q(ij,k,I_QC)=0.0005_r8*(1.0_r8-sig); q(ij,k,I_QR)=0.0002_r8*(1.0_r8-sig)
        gsgam2 (ij,k)=1.0_r8+0.05_r8*sig
        gsgam2h(ij,k)=1.0_r8+0.04_r8*sig
        ! prognostics (independent synthetic; density-weighted)
        rhog  (ij,k)=rho(ij,k)*gsgam2(ij,k)
        rhogvx(ij,k)=rhog(ij,k)*vx(ij,k); rhogvy(ij,k)=rhog(ij,k)*vy(ij,k); rhogvz(ij,k)=rhog(ij,k)*vz(ij,k)
        rhogw (ij,k)=0.0_r8
        rhoge (ij,k)=rhog(ij,k)*CVdry*tem(ij,k)
        rhogq (ij,k,I_QV)=rhog(ij,k)*q(ij,k,I_QV)
        rhogq (ij,k,I_QC)=rhog(ij,k)*q(ij,k,I_QC)
        rhogq (ij,k,I_QR)=rhog(ij,k)*q(ij,k,I_QR)
     end do
  end do

  SM_LargeScaleCond=.true.; SM_PBL_Bryan=.false.; USE_HeldSuarez=.false.
  SM_Latdepend_SST=.true.;  SM_MITC_SST_TYPE=1

  open(fid, file=trim(outfile), status='replace', action='write')
  write(fid,'(A)') '# forcing_step (part A) reference dump'
  write(fid,'(A,3(1x,I0))') 'META ijdim vlayer kdim', ijdim, vlayer, kdim
  write(fid,'(A,2(1x,I0))') 'META kmin kmax', kmin, kmax
  write(fid,'(A,3(1x,I0))') 'META ntrc I_QV_0based', ntrc, I_QV-1
  write(fid,'(A,1x,ES24.16)') 'META dt', DTL
  write(fid,'(A,4(1x,ES24.16))') 'META Rdry CPdry CVdry PRE00', Rdry, CPdry, CVdry, PRE00
  write(fid,'(A,3(1x,ES24.16))') 'META CVW_QV CVW_QC CVW_QR', CVW_QV, CVW_QC, CVW_QR
  write(fid,'(A,1x,ES24.16)') 'META GRAV', GRAV
  ! dump the shared inputs
  call d1('lat',lat); call d1('lon',lon); call d1('z_srf',z_srf)
  call d1('ix',ix); call d1('iy',iy); call d1('iz',iz); call d1('jx',jx); call d1('jy',jy); call d1('jz',jz)
  call d2('alt',alt); call d2('alth',alth); call d2('rho',rho); call d2('pre',pre); call d2('tem',tem)
  call d2('vx',vx); call d2('vy',vy); call d2('vz',vz); call d2('gsgam2',gsgam2); call d2('gsgam2h',gsgam2h)
  call d3('q',q)
  call d2('rhog',rhog); call d2('rhogvx',rhogvx); call d2('rhogvy',rhogvy); call d2('rhogvz',rhogvz)
  call d2('rhogw',rhogw); call d2('rhoge',rhoge); call d3('rhogq',rhogq)

  call run_apply('DRY',  1, 1)   ! RAIN_TYPE, NQW_STR, NQW_END (1-based)
  call run_apply('WARM', 1, 3)

  close(fid)
  write(*,*) 'wrote ', trim(outfile)

contains

  subroutine run_apply(rain, nqs, nqe)
    character(*), intent(in) :: rain
    integer, intent(in) :: nqs, nqe
    real(r8) :: RGVX(ijdim,kdim),RGVY(ijdim,kdim),RGVZ(ijdim,kdim),RGE(ijdim,kdim),RG(ijdim,kdim)
    real(r8) :: RGQ(ijdim,kdim,ntrc)
    NQW_STR=nqs; NQW_END=nqe
    ! working copies of prognostics (apply mutates them)
    RG=rhog; RGVX=rhogvx; RGVY=rhogvy; RGVZ=rhogvz; RGE=rhoge; RGQ=rhogq

    ! ein = rhoge/rhog
    do k=1,kdim; do ij=1,ijdim; ein(ij,k)=RGE(ij,k)/RG(ij,k); end do; end do
    ! surface pressure (single-layer hydrostatic, L274-275)
    do ij=1,ijdim
       pre_sfc(ij)=pre(ij,kmin)+rho(ij,kmin)*GRAV*(alt(ij,kmin)-z_srf(ij))
    end do

    ! ===== AF_dcmip SimpleMicrophys transform (verbatim) =====
    fvx=0.0_r8; fvy=0.0_r8; fvz=0.0_r8; fe=0.0_r8; fq=0.0_r8; precip=0.0_r8; precip2=0.0_DP
    if (SM_Latdepend_SST) then; test=1; else; test=0; end if
    lat_DP(:)=real(lat(:),kind=DP)
    do k=1,vlayer
       kk=kmax-k+1
       t(:,k)=tem(:,kk); qvv(:,k)=q(:,kk,I_QV)
       u(:,k)=vx(:,kk)*ix(:)+vy(:,kk)*iy(:)+vz(:,kk)*iz(:)
       v(:,k)=vx(:,kk)*jx(:)+vy(:,kk)*jy(:)+vz(:,kk)*jz(:)
       pmid(:,k)=pre(:,kk)
    end do
    pint(:,1)=0.0_r8
    do k=2,vlayer
       kk=kmax-k+1
       pint(:,k)=pre(:,kk)*exp(log(pre(:,kk+1)/pre(:,kk))*(alth(:,kk+1)-alt(:,kk))/(alt(:,kk+1)-alt(:,kk)))
    end do
    pint(:,vlayer+1)=pre_sfc(:)
    do k=1,vlayer; pdel(:,k)=pint(:,k+1)-pint(:,k); rpdel(:,k)=1.0_r8/pdel(:,k); end do
    ps(:)=pre_sfc(:)
    call simple_physics(ijdim,vlayer,DTL,lat_DP,t,qvv,u,v,pmid,pint,pdel,rpdel,ps,precip2,&
                        test,SM_LargeScaleCond,SM_PBL_Bryan,USE_HeldSuarez,SM_MITC_SST_TYPE)
    do k=1,vlayer
       kk=kmax-k+1
       fvx(:,kk)=fvx(:,kk)+(u(:,k)*ix(:)+v(:,k)*jx(:)-vx(:,kk))/DTL
       fvy(:,kk)=fvy(:,kk)+(u(:,k)*iy(:)+v(:,k)*jy(:)-vy(:,kk))/DTL
       fvz(:,kk)=fvz(:,kk)+(u(:,k)*iz(:)+v(:,k)*jz(:)-vz(:,kk))/DTL
    end do
    do k=1,vlayer
       kk=kmax-k+1
       do ij=1,ijdim
          if (rain=='DRY') then
             qvc(k)=qvv(ij,k); qd(k)=1.0_r8-qvc(k); cv(k)=qd(k)*CVdry+qvc(k)*CVW_QV
          else
             qvc(k)=qvv(ij,k); qcc(k)=q(ij,kk,I_QC); qrc(k)=q(ij,kk,I_QR)
             qd(k)=1.0_r8-qvc(k)-qcc(k)-qrc(k)
             cv(k)=qd(k)*CVdry+qvc(k)*CVW_QV+qcc(k)*CVW_QC+qrc(k)*CVW_QR
          end if
          fq(ij,kk,I_QV)=fq(ij,kk,I_QV)+(qvc(k)-q(ij,kk,I_QV))/DTL
          fe(ij,kk)     =fe(ij,kk)     +(cv(k)*t(ij,k)-ein(ij,kk))/DTL
       end do
    end do
    precip(:)=precip2(:)

    ! ===== apply tendencies (L367-390) =====
    do k=1,kdim; do ij=1,ijdim
       RGVX(ij,k)=RGVX(ij,k)+DTL*fvx(ij,k)*rho(ij,k)*gsgam2(ij,k)
       RGVY(ij,k)=RGVY(ij,k)+DTL*fvy(ij,k)*rho(ij,k)*gsgam2(ij,k)
       RGVZ(ij,k)=RGVZ(ij,k)+DTL*fvz(ij,k)*rho(ij,k)*gsgam2(ij,k)
       RGE (ij,k)=RGE (ij,k)+DTL*fe (ij,k)*rho(ij,k)*gsgam2(ij,k)
    end do; end do
    do nq=1,ntrc
       do k=1,kdim; do ij=1,ijdim; frhogq(ij,k)=fq(ij,k,nq)*rho(ij,k)*gsgam2(ij,k); end do; end do
       do k=1,kdim; do ij=1,ijdim; RGQ(ij,k,nq)=RGQ(ij,k,nq)+DTL*frhogq(ij,k); end do; end do
       if (nq>=NQW_STR .and. nq<=NQW_END) then
          do k=1,kdim; do ij=1,ijdim; RG(ij,k)=RG(ij,k)+DTL*frhogq(ij,k); end do; end do
       end if
    end do

    write(fid,'(A,1x,A)') 'CONFIG', rain
    call d2('rhog',RG); call d2('rhogvx',RGVX); call d2('rhogvy',RGVY); call d2('rhogvz',RGVZ)
    call d2('rhoge',RGE); call d3('rhogq',RGQ); call d1('precip',precip)
  end subroutine run_apply

  subroutine d1(nm,a)
    character(*),intent(in)::nm; real(r8),intent(in)::a(:); integer::j
    write(fid,'(A,1x,A,1x,I0)') 'ARRAY',nm,size(a)
    do j=1,size(a); write(fid,'(ES24.16)') a(j); end do
  end subroutine d1
  subroutine d2(nm,a)
    character(*),intent(in)::nm; real(r8),intent(in)::a(:,:); integer::ic,il
    write(fid,'(A,1x,A,1x,I0)') 'ARRAY',nm,size(a)
    do ic=1,size(a,1); do il=1,size(a,2); write(fid,'(ES24.16)') a(ic,il); end do; end do
  end subroutine d2
  subroutine d3(nm,a)
    character(*),intent(in)::nm; real(r8),intent(in)::a(:,:,:); integer::ic,il,it
    write(fid,'(A,1x,A,1x,I0)') 'ARRAY',nm,size(a)
    do ic=1,size(a,1); do il=1,size(a,2); do it=1,size(a,3); write(fid,'(ES24.16)') a(ic,il,it); end do; end do; end do
  end subroutine d3
end program forcing_step_ref
