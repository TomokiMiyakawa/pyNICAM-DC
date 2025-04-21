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
        rho      = np.zeros((adm.ADM_shape), dtype=rdtype)
        rho_pl   = np.zeros((adm.ADM_shape_pl), dtype=rdtype)
        ein      = np.zeros((adm.ADM_shape), dtype=rdtype)
        ein_pl   = np.zeros((adm.ADM_shape_pl), dtype=rdtype)
        rhog_h   = np.zeros((adm.ADM_shape[:3]), dtype=rdtype)
        rhog_h_pl= np.zeros((adm.ADM_shape_pl[:2]), dtype=rdtype)

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


        rho, ein = tdyn.THRMDYN_rhoein( adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall,
                                    diag[:, :, :, :, rcnf.I_tem],
                                    diag[:, :, :, :, rcnf.I_pre],
                                    diag[:, :, :, :, rcnf.I_qstr:rcnf.I_qend + 1],
                                    cnst, rcnf, rdtype
                                )   
      
        # with open(std.fname_log, 'a') as log_file:
        #     print("rho shape: ", rho.shape, file=log_file)
        #     print(rho[5,5,0,0], file=log_file)
        #     print("ein shape: ", ein.shape, file=log_file)
        #     print(ein[5,5,0,0], file=log_file)
        #     print("diag shape: ", diag.shape, file=log_file)

        
        for i in range(adm.ADM_gall_1d):
            for j in range(adm.ADM_gall_1d):
                for k in range(adm.ADM_kall):
                    for l in range(adm.ADM_lall):
                        prg[i, j, k, l, rcnf.I_RHOG]   = rho[i, j, k, l] * vmtr.VMTR_GSGAM2[i, j, k, l]
                        prg[i, j, k, l, rcnf.I_RHOGVX] = prg[i, j, k, l, rcnf.I_RHOG] * diag[i, j, k, l, rcnf.I_vx]
                        prg[i, j, k, l, rcnf.I_RHOGVY] = prg[i, j, k, l, rcnf.I_RHOG] * diag[i, j, k, l, rcnf.I_vy]
                        prg[i, j, k, l, rcnf.I_RHOGVZ] = prg[i, j, k, l, rcnf.I_RHOG] * diag[i, j, k, l, rcnf.I_vz]
                        prg[i, j, k, l, rcnf.I_RHOGE]  = prg[i, j, k, l, rcnf.I_RHOG] * ein[i, j, k, l]

        # if prc.prc_myrank == 3:
        #     with open(std.fname_log, 'a') as log_file:
        #         print("rhogvx at i=3, j=11, k=11, l=0: ", prg[3, 11, 11, 0, rcnf.I_RHOGVX], file=log_file)
        #         print("rhog", prg[3, 11, 11, 0, rcnf.I_RHOG], file=log_file)
        #         print("rho", rho[3, 11, 11, 0], file=log_file)
        #         print("vmtr", vmtr.VMTR_GSGAM2[3, 11, 11, 0], file=log_file)
        # if prc.prc_myrank == 4:
        #     with open(std.fname_log, 'a') as log_file:
        #         print("rhog aaat i=17, j=0, k=40, l=1: ",file=log_file) 
        #         print("rhog", prg[17, 0, 40, 1, rcnf.I_RHOG], file=log_file)
        #         print("rho", rho[17, 0, 40, 1], file=log_file)
        #         print("vmtr", vmtr.VMTR_GSGAM2[17, 0, 40, 1], file=log_file)
            #print("rhogvx at i=3, j=11, k=11, l=0: ", prg[3, 11, 11, 0, rcnf.I_RHOGVX])
            #print("rhog", prg[3, 11, 11, 0, rcnf.I_RHOG])
            #print("rho", rho[3, 11, 11, 0])
            #print("vmtr", vmtr.VMTR_GSGAM2[3, 11, 11, 0])
        
        for i in range(adm.ADM_gall_1d):
            for j in range(adm.ADM_gall_1d):
                for k in range(adm.ADM_kall):
                    for l in range(adm.ADM_lall):
                        for iq in range(rcnf.TRC_vmax):
                            prg[i, j, k, l, rcnf.PRG_vmax0 + iq] = prg[i, j, k, l, rcnf.I_RHOG] * diag[i, j, k, l, rcnf.DIAG_vmax0 + iq]


        for l in range(adm.ADM_lall):
            # ------ interpolation of rhog_h ------
            for i in range(adm.ADM_gall_1d):
                for j in range(adm.ADM_gall_1d):
                    for k in range(1, adm.ADM_kall):  # starts at 1 to match Fortran's 2-based loop
                        rhog_h[i, j, k] = (
                            vmtr.VMTR_C2Wfact[i, j, k, 0, l] * prg[i, j, k,   l, rcnf.I_RHOG] +
                            vmtr.VMTR_C2Wfact[i, j, k, 1, l] * prg[i, j, k-1, l, rcnf.I_RHOG]
                        )

            for i in range(adm.ADM_gall_1d):
                for j in range(adm.ADM_gall_1d):
                    rhog_h[i, j, 0] = rhog_h[i, j, 1]

            for i in range(adm.ADM_gall_1d):
                for j in range(adm.ADM_gall_1d):
                    for k in range(adm.ADM_kall):
                        prg[i, j, k, l, rcnf.I_RHOGW] = rhog_h[i, j, k] * diag[i, j, k, l, rcnf.I_w]

        if adm.ADM_have_pl:
            
            rho_pl, ein_pl = tdyn.THRMDYN_rhoein( adm.ADM_gall_pl, 0, adm.ADM_kall, adm.ADM_lall_pl,
                                    diag_pl[:, :, :, rcnf.I_tem],
                                    diag_pl[:, :, :, rcnf.I_pre],
                                    diag_pl[:, :, :, rcnf.I_qstr:rcnf.I_qend + 1], 
                                    cnst, rcnf, rdtype
                                )  


            for g in range(adm.ADM_gall_pl):
                for k in range(adm.ADM_kall):
                    for l in range(adm.ADM_lall_pl):
                        prg_pl[g, k, l, rcnf.I_RHOG]   = rho_pl[g, k, l] * vmtr.VMTR_GSGAM2_pl[g, k, l]
                        prg_pl[g, k, l, rcnf.I_RHOGVX] = prg_pl[g, k, l, rcnf.I_RHOG] * diag_pl[g, k, l, rcnf.I_vx]
                        prg_pl[g, k, l, rcnf.I_RHOGVY] = prg_pl[g, k, l, rcnf.I_RHOG] * diag_pl[g, k, l, rcnf.I_vy]
                        prg_pl[g, k, l, rcnf.I_RHOGVZ] = prg_pl[g, k, l, rcnf.I_RHOG] * diag_pl[g, k, l, rcnf.I_vz]
                        prg_pl[g, k, l, rcnf.I_RHOGE]  = prg_pl[g, k, l, rcnf.I_RHOG] * ein_pl[g, k, l]

            # Tracer quantities
            for g in range(adm.ADM_gall_pl):
                for k in range(adm.ADM_kall):
                    for l in range(adm.ADM_lall_pl):
                        for iq in range(rcnf.TRC_vmax):
                            prg_pl[g, k, l, rcnf.PRG_vmax0 + iq] = (
                                prg_pl[g, k, l, rcnf.I_RHOG] * diag_pl[g, k, l, rcnf.DIAG_vmax0 + iq]
                            )

            # Interpolation and w-component
            for l in range(adm.ADM_lall_pl):
                for g in range(adm.ADM_gall_pl):
                    for k in range(1, adm.ADM_kall):  # Start at 1 to match Fortran k=2
                        rhog_h_pl[g, k] = (
                            vmtr.VMTR_C2Wfact_pl[g, k, 0, l] * prg_pl[g, k,   l, rcnf.I_RHOG] +
                            vmtr.VMTR_C2Wfact_pl[g, k, 1, l] * prg_pl[g, k-1, l, rcnf.I_RHOG]
                        )

                for g in range(adm.ADM_gall_pl):
                    rhog_h_pl[g, 0] = rhog_h_pl[g, 1]

                for g in range(adm.ADM_gall_pl):
                    for k in range(adm.ADM_kall):
                        prg_pl[g, k, l, rcnf.I_RHOGW] = rhog_h_pl[g, k] * diag_pl[g, k, l, rcnf.I_w]
    
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
     
        
        rhogkin      = np.full_like(rhog, cnst.CONST_UNDEF)
        rhogkin_pl   = np.full_like(rhog_pl, cnst.CONST_UNDEF)
        rhogkin_h    = np.full((gall_1d, gall_1d, kall, ), cnst.CONST_UNDEF, dtype=rdtype) # rho X ( G^1/2 X gamma2 ) X kin (horizontal)
        rhogkin_h_pl = np.full((gall_pl,          kall, ), cnst.CONST_UNDEF, dtype=rdtype)
        rhogkin_v    = np.full((gall_1d, gall_1d, kall, ), cnst.CONST_UNDEF, dtype=rdtype) # rho X ( G^1/2 X gamma2 ) X kin (vertical)
        rhogkin_v_pl = np.full((gall_pl,          kall, ), cnst.CONST_UNDEF, dtype=rdtype)
        #rhogkin    = np.empty_like(rhog)
        #rhogkin_pl = np.empty_like(rhog_pl)
        #rhogkin_h     = np.empty((gall_1d, gall_1d, kall, ), dtype=rdtype) # rho X ( G^1/2 X gamma2 ) X kin (horizontal)
        #rhogkin_h_pl = np.empty((gall_pl,          kall, ), dtype=rdtype)
        #rhogkin_v    = np.empty((gall_1d, gall_1d, kall, ), dtype=rdtype) # rho X ( G^1/2 X gamma2 ) X kin (vertical)
        #rhogkin_v_pl = np.empty((gall_pl,          kall, ), dtype=rdtype)

        for l in range(lall):
            # --- Horizontal ---
            for k in range(kmin, kmax + 1):
                rhogkin_h[:, :, k] = rdtype(0.5) * (
                    rhogvx[:, :, k, l] ** 2 +
                    rhogvy[:, :, k, l] ** 2 +
                    rhogvz[:, :, k, l] ** 2
                ) / rhog[:, :, k, l]
            #end k loop

            # --- Vertical  ---
            for k in range(kmin + 1, kmax + 1):
                denom = (
                    vmtr.VMTR_C2Wfact[:, :, k, 0, l] * rhog[:, :, k, l] +
                    vmtr.VMTR_C2Wfact[:, :, k, 1, l] * rhog[:, :, k - 1, l]
                )
                rhogkin_v[:, :, k] = rdtype(0.5) * rhogw[:, :, k, l] ** 2 / denom
            #end k loop

            rhogkin_v[:, :, kmin] = rdtype(0.0)
            rhogkin_v[:, :, kmax + 1] = rdtype(0.0)

            # --- Total  ---
            for k in range(kmin, kmax + 1):
                rhogkin[:, :, k, l] = (
                    rhogkin_h[:, :, k] +
                    vmtr.VMTR_W2Cfact[:, :, k, 0, l] * rhogkin_v[:, :, k + 1] +
                    vmtr.VMTR_W2Cfact[:, :, k, 1, l] * rhogkin_v[:, :, k]
                )
            #end k loop

            rhogkin[:, :, kmin - 1, l] = rdtype(0.0)
            rhogkin[:, :, kmax + 1, l] = rdtype(0.0)
        #end l loop

        if adm.ADM_have_pl:
            for l in range(adm.ADM_lall_pl):
                #--- horizontal kinetic energy
                rhogkin_h_pl[:, kmin:kmax+1] = (
                    rdtype(0.5) * (
                        rhogvx_pl[:, kmin:kmax+1, l]**2 +
                        rhogvy_pl[:, kmin:kmax+1, l]**2 +
                        rhogvz_pl[:, kmin:kmax+1, l]**2
                    ) / rhog_pl[:, kmin:kmax+1, l]
                )

                #--- vertical kinetic energy
                rhogkin_v_pl[:, kmin+1:kmax+1] = (
                    rdtype(0.5) * rhogw_pl[:, kmin+1:kmax+1, l]**2 /
                    (
                        vmtr.VMTR_C2Wfact_pl[:, kmin+1:kmax+1, 0, l] * rhog_pl[:, kmin+1:kmax+1, l] +
                        vmtr.VMTR_C2Wfact_pl[:, kmin+1:kmax+1, 1, l] * rhog_pl[:, kmin:kmax, l]
                    )
                )
                rhogkin_v_pl[:, kmin] = rdtype(0.0)
                rhogkin_v_pl[:, kmax+1] = rdtype(0.0)

                #--- total kinetic energy
                rhogkin_pl[:, kmin:kmax+1, l] = (
                    rhogkin_h_pl[:, kmin:kmax+1] +
                    vmtr.VMTR_W2Cfact_pl[:, kmin:kmax+1, 0, l] * rhogkin_v_pl[:, kmin+1:kmax+2] +
                    vmtr.VMTR_W2Cfact_pl[:, kmin:kmax+1, 1, l] * rhogkin_v_pl[:, kmin:kmax+1]
                )

                rhogkin_pl[:, kmin-1, l] = rdtype(0.0)
                rhogkin_pl[:, kmax+1, l] = rdtype(0.0)
            #end l loop
        #endif

        prf.PROF_rapend('CNV_rhogkin',2)

        return rhogkin, rhogkin_pl
    