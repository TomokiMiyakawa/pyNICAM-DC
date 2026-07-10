import toml
import numpy as np
#from mpi4py import MPI
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
#from mod_prof import prf


class Bsst:
    
    _instance = None
    
    ref_type  = 'NOBASE' 
    ref_fname = 'ref.dat'
    sounding_fname = ''

    def __init__(self):
        pass

    def bsstate_setup(self, fname_in, cnst, grd, vmtr, bndc, rdtype):

        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("+++ Module[basic state]/Category[nhm share]", file=log_file)
                print(f"*** input toml file is ", fname_in, file=log_file)

        with open(fname_in, 'r') as  file:
            cnfs = toml.load(file)

        if 'bsstateparam' not in cnfs:
            with open(std.fname_log, 'a') as log_file:
                print("*** bsstateparam not found in toml file! Use default.", file=log_file)
                prc.prc_mpistop(std.io_l, std.fname_log)
        else:
            cnfs = cnfs['bsstateparam']
            ref_type = cnfs['ref_type']
            self.ref_type = ref_type
            self.ref_fname = cnfs.get('ref_fname', self.ref_fname)
            self.sounding_fname = cnfs.get('sounding_fname', self.sounding_fname)

        if std.io_nml:
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print(cnfs,file=log_file)

        self.rho_bs    = np.zeros((adm.ADM_shape), dtype=rdtype)
        self.rho_bs_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)
        self.pre_bs    = np.zeros((adm.ADM_shape), dtype=rdtype)
        self.pre_bs_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)
        self.tem_bs    = np.zeros((adm.ADM_shape), dtype=rdtype)
        self.tem_bs_pl = np.zeros((adm.ADM_shape_pl), dtype=rdtype)

        pre_ref = np.zeros(adm.ADM_kall, dtype=rdtype)
        tem_ref = np.zeros(adm.ADM_kall, dtype=rdtype)
        qv_ref  = np.zeros(adm.ADM_kall, dtype=rdtype)

        if ref_type == 'NOBASE':
            pass

        elif ref_type == 'INPUT':
            pre_ref, tem_ref, qv_ref = self._input_ref(self.ref_fname, rdtype)

        else:
            # bsstate_generate (from a sounding / analytic profile) is not ported yet.
            print("Sorry, ref_type '" + str(ref_type) + "' (generate) is not implemented yet.")
            prc.prc_mpistop(std.io_l, std.fname_log)

        if ref_type != 'NOBASE':
            # build the 3-D basic state from the 1-D reference profile
            self._set_basicstate(pre_ref, tem_ref, qv_ref, grd, vmtr, bndc, cnst, rdtype)

            # reference potential temperature / density (diagnostic, for the log)
            Rdry = cnst.CONST_Rdry; Rvap = cnst.CONST_Rvap; CPdry = cnst.CONST_CPdry
            PRE00 = cnst.CONST_PRE00
            th_ref = tem_ref * (PRE00 / pre_ref) ** (Rdry / CPdry)
            rho_ref = pre_ref / tem_ref / ((rdtype(1.0) - qv_ref) * Rdry + qv_ref * Rvap)
            self.pre_ref = pre_ref; self.tem_ref = tem_ref
            self.qv_ref = qv_ref; self.th_ref = th_ref; self.rho_ref = rho_ref
            if std.io_l:
                with open(std.fname_log, 'a') as log_file:
                    print("-------------------------------------------------------", file=log_file)
                    print("Level   Density  Pressure     Temp. Pot. Tem.        qv", file=log_file)
                    for k in range(adm.ADM_kall - 1, -1, -1):
                        print(f"{k+1:4d}{rho_ref[k]:12.4f}{pre_ref[k]:10.2f}"
                              f"{tem_ref[k]:10.2f}{th_ref[k]:10.2f}{qv_ref[k]:10.7f}", file=log_file)
                    print("-------------------------------------------------------", file=log_file)

        return

    def _input_ref(self, fname, rdtype):
        # Read the reference profile (nicamdc bsstate_input_ref): a big-endian
        # (-byteswapio) Fortran sequential-unformatted file with 3 DP records
        # (pre_ref, tem_ref, qv_ref), each of length ADM_kall.
        kall = adm.ADM_kall
        recs = []
        with open(fname, 'rb') as f:
            for _ in range(3):
                n1 = int(np.frombuffer(f.read(4), dtype='>i4')[0])   # leading record marker
                data = np.frombuffer(f.read(n1), dtype='>f8')
                f.read(4)                                            # trailing record marker
                recs.append(data.astype(rdtype))
        pre_ref, tem_ref, qv_ref = recs
        return pre_ref, tem_ref, qv_ref

    def _vintrpl_z2xi(self, var, grd, rdtype):
        # Vertical interpolation z-level -> zstar(xi)-level (nicamdc VINTRPL_Z2Xi):
        # re-sample the column onto GRD_gz using 3-point Lagrange, evaluated at the
        # physical height GRD_Z. Identity for flat topography (GRD_Z == GRD_gz).
        kmin, kmax = adm.ADM_kmin, adm.ADM_kmax
        Xi = grd.GRD_gz                                              # (kall,)
        Z = grd.GRD_vz[:, :, :, :, grd.GRD_Z]                        # (i,j,kall,l)
        tmp = var.copy()
        tmp[:, :, kmin - 1, :] = var[:, :, kmin, :]
        tmp[:, :, kmax + 1, :] = var[:, :, kmax, :]
        out = var.copy()
        Xi_search = Xi[kmin:kmax]                                    # kk range kmin..kmax-1
        for k in range(kmin, kmax + 1):
            Zk = Z[:, :, k, :]                                       # (i,j,l)
            kk = np.searchsorted(Xi_search, Zk, side='left') + kmin  # first kk with Xi[kk] >= Zk
            kk = np.clip(kk, kmin + 1, kmax)                         # kk=max(kk,kmin+1); cap at kmax
            z1 = Xi[kk]; z2 = Xi[kk - 1]; z3 = Xi[kk - 2]
            p1 = np.take_along_axis(tmp, kk[:, :, None, :], axis=2)[:, :, 0, :]
            p2 = np.take_along_axis(tmp, (kk - 1)[:, :, None, :], axis=2)[:, :, 0, :]
            p3 = np.take_along_axis(tmp, (kk - 2)[:, :, None, :], axis=2)[:, :, 0, :]
            out[:, :, k, :] = (((Zk - z2) * (Zk - z3)) / ((z1 - z2) * (z1 - z3)) * p1
                               + ((Zk - z1) * (Zk - z3)) / ((z2 - z1) * (z2 - z3)) * p2
                               + ((Zk - z1) * (Zk - z2)) / ((z3 - z1) * (z3 - z2)) * p3)
        out[:, :, kmin - 1, :] = out[:, :, kmin, :]
        out[:, :, kmax + 1, :] = out[:, :, kmax, :]
        return out

    def _vintrpl_z2xi_pl(self, var, grd, rdtype):
        # Pole-region variant of _vintrpl_z2xi. Pole arrays are 3-D [g,k,l] and
        # the height comes from GRD_vz_pl. Same 3-point Lagrange remap as the
        # regular grid (nicamdc VINTRPL_Z2Xi, ADM_have_pl branch).
        kmin, kmax = adm.ADM_kmin, adm.ADM_kmax
        Xi = grd.GRD_gz                                             # (kall,)
        Z = grd.GRD_vz_pl[:, :, :, grd.GRD_Z]                       # (g,kall,l)
        tmp = var.copy()
        tmp[:, kmin - 1, :] = var[:, kmin, :]
        tmp[:, kmax + 1, :] = var[:, kmax, :]
        out = var.copy()
        Xi_search = Xi[kmin:kmax]                                   # kk range kmin..kmax-1
        for k in range(kmin, kmax + 1):
            Zk = Z[:, k, :]                                         # (g,l)
            kk = np.searchsorted(Xi_search, Zk, side='left') + kmin # first kk with Xi[kk] >= Zk
            kk = np.clip(kk, kmin + 1, kmax)                        # kk=max(kk,kmin+1); cap at kmax
            z1 = Xi[kk]; z2 = Xi[kk - 1]; z3 = Xi[kk - 2]
            p1 = np.take_along_axis(tmp, kk[:, None, :], axis=1)[:, 0, :]
            p2 = np.take_along_axis(tmp, (kk - 1)[:, None, :], axis=1)[:, 0, :]
            p3 = np.take_along_axis(tmp, (kk - 2)[:, None, :], axis=1)[:, 0, :]
            out[:, k, :] = (((Zk - z2) * (Zk - z3)) / ((z1 - z2) * (z1 - z3)) * p1
                            + ((Zk - z1) * (Zk - z3)) / ((z2 - z1) * (z2 - z3)) * p2
                            + ((Zk - z1) * (Zk - z2)) / ((z3 - z1) * (z3 - z2)) * p3)
        out[:, kmin - 1, :] = out[:, kmin, :]
        out[:, kmax + 1, :] = out[:, kmax, :]
        return out

    def _set_basicstate(self, pre_ref, tem_ref, qv_ref, grd, vmtr, bndc, cnst, rdtype):
        # nicamdc set_basicstate: broadcast the 1-D reference to 3-D, remap z->xi,
        # form density (moist gas), then apply the thermodynamic boundary condition.
        Rdry = cnst.CONST_Rdry; Rvap = cnst.CONST_Rvap
        kmin, kmax = adm.ADM_kmin, adm.ADM_kmax
        one = rdtype(1.0)

        self.pre_bs[:] = pre_ref[None, None, :, None]
        self.tem_bs[:] = tem_ref[None, None, :, None]
        qv_bs = np.broadcast_to(qv_ref[None, None, :, None], self.tem_bs.shape).astype(rdtype)

        # z-level -> zstar(xi)-level (identity for flat topo)
        self.pre_bs = self._vintrpl_z2xi(self.pre_bs, grd, rdtype)
        self.tem_bs = self._vintrpl_z2xi(self.tem_bs, grd, rdtype)
        qv_bs = self._vintrpl_z2xi(qv_bs, grd, rdtype)

        self.rho_bs = self.pre_bs / self.tem_bs / ((one - qv_bs) * Rdry + qv_bs * Rvap)

        # thermodynamic boundary condition on the ghost levels (VMTR_PHI = GRD_Z*GRAV)
        phi = grd.GRD_vz[:, :, :, :, grd.GRD_Z] * cnst.CONST_GRAV
        bndc.BNDCND_thermo(kmin, kmax, self.tem_bs, self.rho_bs, self.pre_bs, phi, cnst, rdtype)

        # --- pole region (only on ranks that own a pole) ---
        if adm.ADM_have_pl:
            self.pre_bs_pl[:] = pre_ref[None, :, None]
            self.tem_bs_pl[:] = tem_ref[None, :, None]
            qv_bs_pl = np.broadcast_to(qv_ref[None, :, None], self.tem_bs_pl.shape).astype(rdtype)

            self.pre_bs_pl = self._vintrpl_z2xi_pl(self.pre_bs_pl, grd, rdtype)
            self.tem_bs_pl = self._vintrpl_z2xi_pl(self.tem_bs_pl, grd, rdtype)
            qv_bs_pl = self._vintrpl_z2xi_pl(qv_bs_pl, grd, rdtype)

            self.rho_bs_pl = self.pre_bs_pl / self.tem_bs_pl / ((one - qv_bs_pl) * Rdry + qv_bs_pl * Rvap)

            phi_pl = grd.GRD_vz_pl[:, :, :, grd.GRD_Z] * cnst.CONST_GRAV
            bndc.BNDCND_thermo_pl(kmin, kmax, self.tem_bs_pl, self.rho_bs_pl, self.pre_bs_pl, phi_pl, cnst, rdtype)
        else:
            # nicamdc VINTRPL_Z2Xi fills the pole arrays with UNDEF on non-pole ranks
            self.pre_bs_pl[:] = cnst.CONST_UNDEF
            self.tem_bs_pl[:] = cnst.CONST_UNDEF
        return
