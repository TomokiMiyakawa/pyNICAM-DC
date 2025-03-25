import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf


class Cnvv:
    
    _instance = None
    
    def __init__(self):
        pass

    def cnvvar_diag2prg(self, diag, diag_pl, cnst, vmtr, rcnf, tdyn, rdtype):

        # Output arrays
        prg    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall,    rcnf.PRG_vmax), dtype=rdtype)
        prg_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kall, adm.ADM_lall_pl, rcnf.PRG_vmax), dtype=rdtype)

        # Input arrays
        #diag    = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall,    rcnf.DIAG_vmax), dtype=rdtype)
        #diag_pl = np.zeros((adm.ADM_gall_pl, adm.ADM_kall, adm.ADM_lall_pl, rcnf.DIAG_vmax), dtype=rdtype)

        # Local arrays
        rho      = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall), dtype=rdtype)
        rho_pl   = np.zeros((adm.ADM_gall_pl, adm.ADM_kall, adm.ADM_lall_pl), dtype=rdtype)
        ein      = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall, adm.ADM_lall), dtype=rdtype)
        ein_pl   = np.zeros((adm.ADM_gall_pl, adm.ADM_kall, adm.ADM_lall_pl), dtype=rdtype)
        rhog_h   = np.zeros((adm.ADM_gall_1d, adm.ADM_gall_1d, adm.ADM_kall), dtype=rdtype)
        rhog_h_pl= np.zeros((adm.ADM_gall_pl, adm.ADM_kall), dtype=rdtype)

        with open(std.fname_log, 'a') as log_file:
            print("diag shape: ", diag.shape, file=log_file)
            print("tem, 0, 17, 5, 0:", diag[0,17,5,0,rcnf.I_tem], file=log_file)
            print("pre, 0, 17, 5, 0:", diag[0,17,5,0,rcnf.I_pre], file=log_file)
            print(diag[0,17,5,0,rcnf.I_qstr:rcnf.I_qend+1], file=log_file)

            print("tem, 1, 17, 5, 0:", diag[1,17,5,0,rcnf.I_tem], file=log_file)
            print("pre, 1, 17, 5, 0:", diag[1,17,5,0,rcnf.I_pre], file=log_file)
            print(diag[1,17,5,0,rcnf.I_qstr:rcnf.I_qend+1], file=log_file)

            print("tem, 2, 17, 5, 0:", diag[2,17,5,0,rcnf.I_tem], file=log_file)
            print("pre, 2, 17, 5, 0:", diag[2,17,5,0,rcnf.I_pre], file=log_file)
            print(diag[2,17,5,0,rcnf.I_qstr:rcnf.I_qend+1], file=log_file)


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