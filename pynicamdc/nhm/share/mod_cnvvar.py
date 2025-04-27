import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
from mod_prof import prf


class Cnvv:
    
    _instance = None
    
    def __init__(self):
        pass

    def cnvvar_diag2prg(self, diag, diag_pl, cnst, vmtr, rcnf, tdyn, rdtype):

        # Output arrays
        prg    = np.zeros((adm.ADM_shape    + (rcnf.PRG_vmax,)), dtype=rdtype)
        prg_pl = np.zeros((adm.ADM_shape_pl + (rcnf.PRG_vmax,)), dtype=rdtype)

        # Input arrays
        #diag    = np.zeros((adm.ADM_shape,    rcnf.DIAG_vmax), dtype=rdtype)
        #diag_pl = np.zeros((adm.ADM_shape_pl, rcnf.DIAG_vmax), dtype=rdtype)

        # Local arrays
        rho      = np.zeros(adm.ADM_shape, dtype=rdtype)
        rho_pl   = np.zeros(adm.ADM_shape_pl, dtype=rdtype)
        ein      = np.zeros(adm.ADM_shape, dtype=rdtype)
        ein_pl   = np.zeros(adm.ADM_shape_pl, dtype=rdtype)
        rhog_h   = np.zeros(adm.ADM_shape, dtype=rdtype)
        rhog_h_pl= np.zeros(adm.ADM_shape_pl, dtype=rdtype)

        # with open(std.fname_log, 'a') as log_file:
        #     print("diag shape: ", diag.shape, file=log_file)
        #     print("tem, 0, 17, 5, 0:", diag[0,17,5,0,rcnf.I_tem], file=log_file)
        #     print("pre, 0, 17, 5, 0:", diag[0,17,5,0,rcnf.I_pre], file=log_file)
        #     print(diag[0,17,5,0,rcnf.I_qstr:rcnf.I_qend+1], file=log_file)

        #     print("tem, 1, 17, 5, 0:", diag[1,17,5,0,rcnf.I_tem], file=log_file)
        #     print("pre, 1, 17, 5, 0:", diag[1,17,5,0,rcnf.I_pre], file=log_file)
        #     print(diag[1,17,5,0,rcnf.I_qstr:rcnf.I_qend+1], file=log_file)

        #     print("tem, 2, 17, 5, 0:", diag[2,17,5,0,rcnf.I_tem], file=log_file)
        #     print("pre, 2, 17, 5, 0:", diag[2,17,5,0,rcnf.I_pre], file=log_file)
        #     print(diag[2,17,5,0,rcnf.I_qstr:rcnf.I_qend+1], file=log_file)

        # with open(std.fname_log, 'a') as log_file:
        #     print("rcnf.I_qstr, rcnf.I_qend", rcnf.I_qstr, rcnf.I_qend, file=log_file)

        rho, ein = tdyn.THRMDYN_rhoein( adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall,
                                    diag[:, :, :, :, rcnf.I_tem],
                                    diag[:, :, :, :, rcnf.I_pre],
                                    diag[:, :, :, :, rcnf.I_qstr:rcnf.I_qend], # rcnf.I_qstr:rcnf.I_qend+1 ??
                                    cnst, rcnf, rdtype
                                )   
      
        # with open(std.fname_log, 'a') as log_file:
        #     print("rho shape: ", rho.shape, file=log_file)
        #     print(rho[5,5,0,0], file=log_file)
        #     print("ein shape: ", ein.shape, file=log_file)
        #     print(ein[5,5,0,0], file=log_file)
        #     print("diag shape: ", diag.shape, file=log_file)

        
        # Compute RHOG
        prg[:, :, :, :, rcnf.I_RHOG] = rho * vmtr.VMTR_GSGAM2

        # Reuse RHOG to compute momentum and energy
        prg[:, :, :, :, rcnf.I_RHOGVX] = prg[:, :, :, :, rcnf.I_RHOG] * diag[:, :, :, :, rcnf.I_vx]
        prg[:, :, :, :, rcnf.I_RHOGVY] = prg[:, :, :, :, rcnf.I_RHOG] * diag[:, :, :, :, rcnf.I_vy]
        prg[:, :, :, :, rcnf.I_RHOGVZ] = prg[:, :, :, :, rcnf.I_RHOG] * diag[:, :, :, :, rcnf.I_vz]
        prg[:, :, :, :, rcnf.I_RHOGE]  = prg[:, :, :, :, rcnf.I_RHOG] * ein
        
        # with open(std.fname_log, 'a') as log_file:
        #     print("PPPPP", file=log_file)
        #     print(prg[14, 4, 39, 4, rcnf.I_RHOGVX], prg[14, 4, 39, 4, rcnf.I_RHOG], diag[14, 4, 39, 4, rcnf.I_vx], file=log_file)
        #     print(prg[6, 5, 39, 4, rcnf.I_RHOGVX], prg[6, 5, 39, 4, rcnf.I_RHOG], diag[6, 5, 39, 4, rcnf.I_vx], file=log_file)

        for iq in range(rcnf.TRC_vmax):
            prg[:, :, :, :, rcnf.PRG_vmax0 + iq] = (
                prg[:, :, :, :, rcnf.I_RHOG] * diag[:, :, :, :, rcnf.DIAG_vmax0 + iq]
            )
        # with open(std.fname_log, 'a') as log_file:
        #     #print("iq:", iq,rcnf.PRG_vmax0 + iq,rcnf.DIAG_vmax0 + iq, file=log_file)
        #     #kc=10
        #     print(#prg[6, 5, kc, 0, rcnf.PRG_vmax0 + iq], 
        #             prg[6, 5, :, 0, rcnf.I_RHOG], 
        #             diag[6, 5, :, 0, rcnf.I_tem],
        #             diag[6, 5, :, 0, rcnf.I_pre], file=log_file)
        #     print("6 : ", diag[6, 5, :, 0, 6], file=log_file)
        #     print("7 : ", diag[6, 5, :, 0, 7], file=log_file)
        #     print("8 : ", diag[6, 5, :, 0, 8], file=log_file)
        #     print("9 : ", diag[6, 5, :, 0, 9], file=log_file)
        #     print("10: ", diag[6, 5, :, 0, 10], file=log_file)
        #     print("rho: ", rho[6, 5, :, 0], file=log_file)
        #     print("ein: ", ein[6, 5, :, 0], file=log_file)                    

        # k from 1 to kall-1 (inclusive)
        rhog_h[:, :, 1:, :] = (
            vmtr.VMTR_C2Wfact[:, :, 1:, :, 0] * prg[:, :, 1:, :, rcnf.I_RHOG] +
            vmtr.VMTR_C2Wfact[:, :, 1:, :, 1] * prg[:, :, :-1, :, rcnf.I_RHOG]
        )

        # fill k=0 from k=1
        rhog_h[:, :, 0, :] = rhog_h[:, :, 1, :]

        prg[:, :, :, :, rcnf.I_RHOGW] = rhog_h * diag[:, :, :, :, rcnf.I_w]


        if adm.ADM_have_pl:
            rho_pl, ein_pl = tdyn.THRMDYN_rhoein( adm.ADM_gall_pl, 0, adm.ADM_kall, adm.ADM_lall_pl,
                                    diag_pl[:, :, :, rcnf.I_tem],
                                    diag_pl[:, :, :, rcnf.I_pre],
                                    diag_pl[:, :, :, rcnf.I_qstr:rcnf.I_qend + 1], 
                                    cnst, rcnf, rdtype
                                )  

            # Compute RHOG
            prg_pl[..., rcnf.I_RHOG] = rho_pl * vmtr.VMTR_GSGAM2_pl

            # Momentum and energy components
            prg_pl[..., rcnf.I_RHOGVX] = prg_pl[..., rcnf.I_RHOG] * diag_pl[..., rcnf.I_vx]
            prg_pl[..., rcnf.I_RHOGVY] = prg_pl[..., rcnf.I_RHOG] * diag_pl[..., rcnf.I_vy]
            prg_pl[..., rcnf.I_RHOGVZ] = prg_pl[..., rcnf.I_RHOG] * diag_pl[..., rcnf.I_vz]
            prg_pl[..., rcnf.I_RHOGE]  = prg_pl[..., rcnf.I_RHOG] * ein_pl


            # Tracer quantities

            for iq in range(rcnf.TRC_vmax):
                prg_pl[..., rcnf.PRG_vmax0 + iq] = (
                    prg_pl[..., rcnf.I_RHOG] * diag_pl[..., rcnf.DIAG_vmax0 + iq]
                )

            # Interpolation and w-component
            # Vertical interpolation: k = 1 to kall-1
            rhog_h_pl[:, 1:, :] = (
                vmtr.VMTR_C2Wfact_pl[:, 1:, :, 0] * prg_pl[:, 1:, :, rcnf.I_RHOG] +
                vmtr.VMTR_C2Wfact_pl[:, 1:, :, 1] * prg_pl[:, :-1, :, rcnf.I_RHOG]
            )

            # Boundary at k=0 filled from k=1
            rhog_h_pl[:, 0, :] = rhog_h_pl[:, 1, :]

            # Vertical momentum (RHOGW)
            prg_pl[:, :, :, rcnf.I_RHOGW] = rhog_h_pl * diag_pl[:, :, :, rcnf.I_w]

        return prg, prg_pl
    
    def cnvvar_rhogkin(self,
        rhog,    rhog_pl,   
        rhogvx,  rhogvx_pl, 
        rhogvy,  rhogvy_pl, 
        rhogvz,  rhogvz_pl, 
        rhogw,   rhogw_pl,       
        cnst, vmtr, rdtype,
    ):
        
        prf.PROF_rapstart('CNV_rhogkin',2)

        gall_1d = adm.ADM_gall_1d
        gall_pl = adm.ADM_gall_pl
        kall = adm.ADM_kall
        kmin = adm.ADM_kmin
        kmax = adm.ADM_kmax
        lall = adm.ADM_lall
     
        
        rhogkin      = np.full_like(rhog, cnst.CONST_UNDEF)       # 18, 18, 42, 5
        rhogkin_pl   = np.full_like(rhog_pl, cnst.CONST_UNDEF)    #  6,     42, 5

        rhogkin_h    = np.full_like(rhog, cnst.CONST_UNDEF) # rho X ( G^1/2 X gamma2 ) X kin (horizontal)
        #rhogkin_h_pl = np.full_like(rhog_pl[:2], cnst.CONST_UNDEF)
        rhogkin_v    = np.full_like(rhog, cnst.CONST_UNDEF) # rho X ( G^1/2 X gamma2 ) X kin (vertical)
        #rhogkin_v_pl = np.full_like(rhog_pl[:2], cnst.CONST_UNDEF)
        
        #rhogkin_h    = np.full((gall_1d, gall_1d, kall, ), cnst.CONST_UNDEF, dtype=rdtype) # rho X ( G^1/2 X gamma2 ) X kin (horizontal)
        rhogkin_h_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)
        #rhogkin_v    = np.full((gall_1d, gall_1d, kall, ), cnst.CONST_UNDEF, dtype=rdtype) # rho X ( G^1/2 X gamma2 ) X kin (vertical)
        rhogkin_v_pl = np.full(adm.ADM_shape_pl, cnst.CONST_UNDEF, dtype=rdtype)

        # --- Horizontal kinetic energy ---
        rhogkin_h[:, :, kmin:kmax+1, :] = rdtype(0.5) * (
            rhogvx[:, :, kmin:kmax+1, :] ** 2 +
            rhogvy[:, :, kmin:kmax+1, :] ** 2 +
            rhogvz[:, :, kmin:kmax+1, :] ** 2
        ) / rhog[:, :, kmin:kmax+1, :]

        # --- Vertical kinetic energy ---
        denom = (
            vmtr.VMTR_C2Wfact[:, :, kmin+1:kmax+1, :, 0] * rhog[:, :, kmin+1:kmax+1, :] +
            vmtr.VMTR_C2Wfact[:, :, kmin+1:kmax+1, :, 1] * rhog[:, :, kmin:kmax, :]
        )
        rhogkin_v[:, :, kmin+1:kmax+1, :] = rdtype(0.5) * rhogw[:, :, kmin+1:kmax+1, :] ** 2 / denom

        # Boundary values for rhogkin_v
        rhogkin_v[:, :, kmin, :] = rdtype(0.0)
        rhogkin_v[:, :, kmax+1, :] = rdtype(0.0)

        # --- Total kinetic energy ---
        rhogkin[:, :, kmin:kmax+1, :] = (
            rhogkin_h[:, :, kmin:kmax+1, :] +
            vmtr.VMTR_W2Cfact[:, :, kmin:kmax+1, :, 0] * rhogkin_v[:, :, kmin+1:kmax+2, :] +
            vmtr.VMTR_W2Cfact[:, :, kmin:kmax+1, :, 1] * rhogkin_v[:, :, kmin:kmax+1, :]
        )

        # Boundary values for rhogkin
        rhogkin[:, :, kmin-1, :] = rdtype(0.0)
        rhogkin[:, :, kmax+1, :] = rdtype(0.0)


        if adm.ADM_have_pl:
            # Horizontal kinetic energy
            rhogkin_h_pl[:, kmin:kmax+1, :] = (
                rdtype(0.5) * (
                    rhogvx_pl[:, kmin:kmax+1, :]**2 +
                    rhogvy_pl[:, kmin:kmax+1, :]**2 +
                    rhogvz_pl[:, kmin:kmax+1, :]**2
                ) / rhog_pl[:, kmin:kmax+1, :]
            )

            # Vertical kinetic energy
            denom = (
                vmtr.VMTR_C2Wfact_pl[:, kmin+1:kmax+1, :, 0] * rhog_pl[:, kmin+1:kmax+1, :] +
                vmtr.VMTR_C2Wfact_pl[:, kmin+1:kmax+1, :, 1] * rhog_pl[:, kmin:kmax, :]
            )
            rhogkin_v_pl[:, kmin+1:kmax+1, :] = rdtype(0.5) * rhogw_pl[:, kmin+1:kmax+1, :]**2 / denom

            # Vertical boundaries
            rhogkin_v_pl[:, kmin,   :] = rdtype(0.0)
            rhogkin_v_pl[:, kmax+1, :] = rdtype(0.0)

            # Total kinetic energy
            rhogkin_pl[:, kmin:kmax+1, :] = (
                rhogkin_h_pl[:, kmin:kmax+1, :] +
                vmtr.VMTR_W2Cfact_pl[:, kmin:kmax+1, :, 0] * rhogkin_v_pl[:, kmin+1:kmax+2, :] +
                vmtr.VMTR_W2Cfact_pl[:, kmin:kmax+1, :, 1] * rhogkin_v_pl[:, kmin:kmax+1, :]
            )

            # Total boundaries
            rhogkin_pl[:, kmin-1, :] = rdtype(0.0)
            rhogkin_pl[:, kmax+1, :] = rdtype(0.0)

        prf.PROF_rapend('CNV_rhogkin',2)

        return rhogkin, rhogkin_pl
    