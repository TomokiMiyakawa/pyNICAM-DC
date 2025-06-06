import toml
import numpy as np
#from mpi4py import MPI
from mod_adm import adm
from mod_stdio import std
from mod_process import prc
#from mod_prof import prf


class Tdyn:
    
    _instance = None
    
    def __init__(self):
        pass

    def THRMDYN_rhoein(self, idim, jdim, kdim, ldim, tem, pre, q, cnst, rcnf, rdtype):   # 3.33333333

        nqmax=rcnf.TRC_vmax
        CVdry = cnst.CONST_CVdry
        Rdry  = cnst.CONST_Rdry
        Rvap  = cnst.CONST_Rvap

        if jdim == 0 and ldim == 0:
            # Output arrays
            rho = np.zeros((idim, kdim), dtype=rdtype) # density     [kg/m3]
            ein = np.zeros((idim, kdim), dtype=rdtype) # internal energy [J]
            # Input arrays
            #tem = np.zeros((idim, kdim), dtype=rdtype)        # temperature [K]
            #pre = np.zeros((idim, kdim), dtype=rdtype)        # pressure [Pa]
            #q   = np.zeros((idim, kdim, nqmax), dtype=rdtype) # tracer mass concentration [kg/kg]
            # Local/output arrays
            cv  = np.zeros((idim, kdim), dtype=rdtype)
            qd  = np.full((idim, kdim), rdtype(1.0), dtype=rdtype)

            for nq in range(rcnf.NQW_STR, rcnf.NQW_END):  # Adjusted for 0-based indexing
                # for ij in range(idim):
                #     for k in range(kdim):
                cv[:, :] += q[:, :, nq] * rcnf.CVW[nq]
                qd[:, :] -= q[:, :, nq]

            # for ij in range(idim):
            #     for k in range(kdim):
            cv[:, :] += qd[:, :] * CVdry
            rho[:, :] = pre[:, :] / (
                (qd[:, :] * Rdry + q[:, :, rcnf.I_QV]*Rvap) * tem[:, :]
            )
            ein[:, :] = tem[:, :] * cv[:, :]

        elif jdim == 0 and ldim > 0:
            # Output arrays
            rho = np.zeros((idim, kdim, ldim), dtype=rdtype)
            ein = np.zeros((idim, kdim, ldim), dtype=rdtype)
            # Input arrays
            #tem = np.zeros((idim, kdim, ldim), dtype=rdtype)
            #pre = np.zeros((idim, kdim, ldim), dtype=rdtype)
            #q   = np.zeros((idim, kdim, ldim, nqmax), dtype=rdtype)
            # Local/output arrays
            cv  = np.zeros((idim, kdim, ldim), dtype=rdtype)
            qd  = np.full((idim, kdim, ldim), rdtype(1.0), dtype=rdtype)

            for nq in range(rcnf.NQW_STR, rcnf.NQW_END):  # Adjusted for 0-based indexing
                # for ij in range(idim):
                #     for k in range(kdim):
                #         for l in range(ldim):
                cv[:, :, :] += q[:, :, :, nq] * rcnf.CVW[nq]
                qd[:, :, :] -= q[:, :, :, nq]

            # for ij in range(idim):
            #     for k in range(kdim):
            #         for l in range(ldim):
            cv[:, :, :] += qd[:, :, :] * CVdry
            rho[:, :, :] = pre[:, :, :] / (        #### invalid value divide
                (qd[:, :, :] * Rdry + q[:, :, :, rcnf.I_QV]*Rvap) * tem[:, :, :]
            )
            ein[:, :, :] = tem[:, :, :] * cv[:, :, :]

        elif jdim > 0 and ldim == 0:
            # Output arrays
            rho = np.zeros((idim, jdim, kdim), dtype=rdtype)
            ein = np.zeros((idim, jdim, kdim), dtype=rdtype)
            # Input arrays
            #tem = np.zeros((idim, jdim, kdim), dtype=rdtype)
            #pre = np.zeros((idim, jdim, kdim), dtype=rdtype)
            #q   = np.zeros((idim, jdim, kdim, nqmax), dtype=rdtype)
            # Local/output arrays
            cv  = np.zeros((idim, jdim, kdim), dtype=rdtype)
            qd  = np.full((idim, jdim, kdim), rdtype(1.0), dtype=rdtype)


            for nq in range(rcnf.NQW_STR, rcnf.NQW_END):  # Adjusted for 0-based indexing
                # for i in range(idim):
                #     for j in range(jdim):
                #         for k in range(kdim):
                cv[:, :, :] += q[:, :, :, nq] * rcnf.CVW[nq]
                qd[:, :, :] -= q[:, :, :, nq]
            
            # for i in range(idim):
            #     for j in range(jdim):
            #         for k in range(kdim):
            cv[:, :, :] += qd[:, :, :] * CVdry
            rho[:, :, :] = pre[:, :, :] / (
                (qd[:, :, :] * Rdry + q[:, :, :, rcnf.I_QV]*Rvap) * tem[:, :, :]
            )
            ein[:, :, :] = tem[:, :, :] * cv[:, :, :]
        
        else:

            #print("IIIIII")
            # Output arrays
            rho = np.zeros((idim, jdim, kdim, ldim), dtype=rdtype)
            ein = np.zeros((idim, jdim, kdim, ldim), dtype=rdtype)
            # Input arrays
            #tem = np.zeros((idim, jdim, kdim, ldim), dtype=rdtype)
            #pre = np.zeros((idim, jdim, kdim, ldim), dtype=rdtype)
            #q   = np.zeros((idim, jdim, kdim, ldim, nqmax), dtype=rdtype)
            # Local/output arrays
            cv  = np.zeros((idim, jdim, kdim, ldim), dtype=rdtype)
            qd  = np.full((idim, jdim, kdim, ldim), rdtype(1.0), dtype=rdtype)
            
            for nq in range(rcnf.NQW_STR, rcnf.NQW_END):  # Adjusted for 0-based indexing   # -1  1
                # for i in range(idim):
                #     for j in range(jdim):
                #         for k in range(kdim):
                #             for l in range(ldim):
                cv[:, :, :, :] += q[:, :, :, :, nq] * rcnf.CVW[nq]
                qd[:, :, :, :] -= q[:, :, :, :, nq]


            # with open(std.fname_log, 'a') as log_file:
            #     print("rcnf.NQW_STR, rcnf.NQW_END :", rcnf.NQW_STR, rcnf.NQW_END, file=log_file)
            #     print("cv: ", cv[6,5,:,0], file=log_file) 
            #     print("qd: ", qd[6,5,:,0], file=log_file) 

            # for i in range(idim):
            #     for j in range(jdim):
            #         for k in range(kdim):
            #             for l in range(ldim):

                            # if (qd[i, j, k, l] * Rdry + q[i, j, k, l, rcnf.I_QV]) * tem[i, j, k, l] == 0:
                            # if i==3 and j==11 and k==11 and l==0:
                            #     with open(std.fname_log, 'a') as log_file:
                            # #         print("Zero division error", file=log_file)
                            #         print("i, j, k, l= ", i, j, k, l, file=log_file)
                            #         print("pre, qd, q, tem, cv:", file=log_file)
                            #         print(pre[i,j,k,l], qd[i, j, k, l], q[i, j, k, l, rcnf.I_QV], tem[i, j, k, l], cv[i, j, k, l], file=log_file)
                            #         #print("Rdry= ", Rdry, "kdim= ", kdim, "ldim= ", ldim, file=log_file)
                            #         print("Rdry= ", Rdry, "Rvap= ", Rvap, "CVdry= ", CVdry,   file=log_file)
                            #         break
                            #         prc.prc_mpistop(std.io_l, std.fname_log)
                            #         import sys
                            #         sys.exit(1)

            cv[:, :, :, :] += qd[:, :, :, :] * CVdry
            rho[:, :, :, :] = pre[:, :, :, :] / (    # zero division error!!!
                (qd[:, :, :, :] * Rdry + q[:, :, :, :, rcnf.I_QV]*Rvap) * tem[:, :, :, :]
            )
            ein[:, :, :, :] = tem[:, :, :, :] * cv[:, :, :, :]

                            # if i==3 and j==11 and k==11 and l==0:
                            #     with open(std.fname_log, 'a') as log_file:
                            #         print("rho, ein:", file=log_file)
                            #         print(rho[i, j, k, l], ein[i, j, k, l], file=log_file)
              
        return rho, ein
    

    def THRMDYN_th(
        self, tem, pre, cnst, 
    ):
        
        RovCP = cnst.CONST_Rdry / cnst.CONST_CPdry
        PRE00 = cnst.CONST_PRE00

        #th = np.empty_like(tem)
        th = np.full_like(tem, cnst.CONST_UNDEF)

        # Preallocate buffer for intermediate step
        #ratio = np.empty_like(pre)
        ratio = np.full_like(pre, cnst.CONST_UNDEF)
        np.divide(PRE00, pre, out=ratio)
        #with open(std.fname_log, 'a') as log_file:  
        #    print("ratio1", ratio[6, 5, 2, 0], file=log_file) 
        #    print("ratio1", ratio[5, 6, 2, 0], file=log_file)                
        # Compute exponent (in-place)
        np.power(ratio, RovCP, out=ratio)
        # Final result in-place into th
        np.multiply(tem, ratio, out=th)

        return th
    

    def THRMDYN_eth(
        self, ein, pre, rho, cnst, 
    ):
        
        eth = np.full_like(pre, cnst.CONST_UNDEF)
        np.divide(pre, rho, out=eth)
        np.add(ein, eth, out=eth)

        # if jdim != 1:
        #     with open(std.fname_log, 'a') as log_file:  
        #         # print("ratio2", ratio[6, 5, 2, 0], file=log_file) 
        #         # print("ratio2", ratio[5, 6, 2, 0], file=log_file) 
        #         print("eth", eth[6, 5, 2, 0], file=log_file) 
        #         print("eth", eth[5, 6, 2, 0], file=log_file) 

        # else: 
        #     with open(std.fname_log, 'a') as log_file:  
        #                     # print("ratio2", ratio[6, 5, 2, 0], file=log_file) 
        #                     # print("ratio2", ratio[5, 6, 2, 0], file=log_file) 
        #                     print("eth_pl", eth[0, 0, 2, 0], file=log_file) 
        #                     print("eth_pl", eth[0, 0, 2, 0], file=log_file)

        # Alternative method (commented out for performance)
        #eth[:, :, :, :] = ein[:, :, :, :] + pre[:, :, :, :] / rho[:, :, :, :]

        return eth
    


    