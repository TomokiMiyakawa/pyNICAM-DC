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

    def THRMDYN_rhoein(self, idim, jdim, kdim, ldim, tem, pre, q, cnst, rcnf, rdtype):

        nqmax=rcnf.TRC_vmax
        CVdry = cnst.CONST_CVdry
        Rdry  = cnst.CONST_Rdry
        Rvap  = cnst.CONST_Rvap

        if jdim == 0 and ldim == 0:
            # Output arrays
            rho = np.zeros((idim, kdim), dtype=rdtype) # density     [kg/m3]
            ein = np.zeros((idim, kdim), dtype=rdtype) # internal energy [J]
            # Input arrays
            tem = np.zeros((idim, kdim), dtype=rdtype)        # temperature [K]
            pre = np.zeros((idim, kdim), dtype=rdtype)        # pressure [Pa]
            q   = np.zeros((idim, kdim, nqmax), dtype=rdtype) # tracer mass concentration [kg/kg]
            # Local/output arrays
            cv  = np.zeros((idim, kdim), dtype=rdtype)
            qd  = np.full((idim, kdim), 1.0, dtype=rdtype)

            for nq in range(rcnf.NQW_STR-1, rcnf.NQW_END):  # Adjusted for 0-based indexing
                for ij in range(idim):
                    for k in range(kdim):
                        cv[ij, k] += q[ij, k, nq] * rcnf.CVW[nq]
                        qd[ij, k] -= q[ij, k, nq]

            for ij in range(idim):
                for k in range(kdim):
                    cv[ij, k] += qd[ij, k] * CVdry
                    rho[ij, k] = pre[ij, k] / (
                        (qd[ij, k] * Rdry + q[ij, k, rcnf.I_QV]) * tem[ij, k]
                    )
                    ein[ij, k] = tem[ij, k] * cv[ij, k]

        elif jdim == 0 and ldim > 0:
            # Output arrays
            rho = np.zeros((idim, kdim, ldim), dtype=rdtype)
            ein = np.zeros((idim, kdim, ldim), dtype=rdtype)
            # Input arrays
            tem = np.zeros((idim, kdim, ldim), dtype=rdtype)
            pre = np.zeros((idim, kdim, ldim), dtype=rdtype)
            q   = np.zeros((idim, kdim, ldim, nqmax), dtype=rdtype)
            # Local/output arrays
            cv  = np.zeros((idim, kdim, ldim), dtype=rdtype)
            qd  = np.full((idim, kdim, ldim), 1.0, dtype=rdtype)

            for nq in range(rcnf.NQW_STR-1, rcnf.NQW_END):  # Adjusted for 0-based indexing
                for ij in range(idim):
                    for k in range(kdim):
                        for l in range(ldim):
                            cv[ij, k, l] += q[ij, k, l, nq] * rcnf.CVW[nq]
                            qd[ij, k, l] -= q[ij, k, l, nq]

            for ij in range(idim):
                for k in range(kdim):
                    for l in range(ldim):
                        cv[ij, k, l] += qd[ij, k, l] * CVdry
                        rho[ij, k, l] = pre[ij, k, l] / (        #### invalid value divide, perhaps due to wrong input data storage order in json
                            (qd[ij, k, l] * Rdry + q[ij, k, l, rcnf.I_QV]) * tem[ij, k, l]
                        )
                        ein[ij, k, l] = tem[ij, k, l] * cv[ij, k, l]

        elif jdim > 0 and ldim == 0:
            # Output arrays
            rho = np.zeros((idim, jdim, kdim), dtype=rdtype)
            ein = np.zeros((idim, jdim, kdim), dtype=rdtype)
            # Input arrays
            tem = np.zeros((idim, jdim, kdim), dtype=rdtype)
            pre = np.zeros((idim, jdim, kdim), dtype=rdtype)
            q   = np.zeros((idim, jdim, kdim, nqmax), dtype=rdtype)
            # Local/output arrays
            cv  = np.zeros((idim, jdim, kdim), dtype=rdtype)
            qd  = np.full((idim, jdim, kdim), 1.0, dtype=rdtype)


            for nq in range(rcnf.NQW_STR-1, rcnf.NQW_END):  # Adjusted for 0-based indexing
                for i in range(idim):
                    for j in range(jdim):
                        for k in range(kdim):
                            cv[i, j, k] += q[i, j, k, nq] * rcnf.CVW[nq]
                            qd[i, j, k] -= q[i, j, k, nq]
            
            for i in range(idim):
                for j in range(jdim):
                    for k in range(kdim):
                        cv[i, j, k] += qd[i, j, k] * CVdry
                        rho[i, j, k] = pre[i, j, k] / (
                            (qd[i, j, k] * Rdry + q[i, j, k, rcnf.I_QV]) * tem[i, j, k]
                        )
                        ein[i, j, k] = tem[i, j, k] * cv[i, j, k]
        
        else:
            # Output arrays
            rho = np.zeros((idim, jdim, kdim, ldim), dtype=rdtype)
            ein = np.zeros((idim, jdim, kdim, ldim), dtype=rdtype)
            # Input arrays
            tem = np.zeros((idim, jdim, kdim, ldim), dtype=rdtype)
            pre = np.zeros((idim, jdim, kdim, ldim), dtype=rdtype)
            q   = np.zeros((idim, jdim, kdim, ldim, nqmax), dtype=rdtype)
            # Local/output arrays
            cv  = np.zeros((idim, jdim, kdim, ldim), dtype=rdtype)
            qd  = np.full((idim, jdim, kdim, ldim), 1.0, dtype=rdtype)
            
            for nq in range(rcnf.NQW_STR-1, rcnf.NQW_END):  # Adjusted for 0-based indexing
                for i in range(idim):
                    for j in range(jdim):
                        for k in range(kdim):
                            for l in range(ldim):
                                cv[i, j, k, l] += q[i, j, k, l, nq] * rcnf.CVW[nq]
                                qd[i, j, k, l] -= q[i, j, k, l, nq]

            for i in range(idim):
                for j in range(jdim):
                    for k in range(kdim):
                        for l in range(ldim):
                            cv[i, j, k, l] += qd[i, j, k, l] * CVdry
                            rho[i, j, k, l] = pre[i, j, k, l] / (
                                (qd[i, j, k, l] * Rdry + q[i, j, k, l, rcnf.I_QV]) * tem[i, j, k, l]
                            )
                            ein[i, j, k, l] = tem[i, j, k, l] * cv[i, j, k, l]
              

        return rho, ein