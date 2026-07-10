"""
Energy & mass budget monitor (port of nicamdc mod_embudget.f90).

Every MNT_INTV steps it forms the global-integral atmospheric energy (potential,
internal, kinetic) and mass (dry air, vapour, liquid, ice, total) and appends a
line to BUDGET_energy.log / BUDGET_mass.log. The first record holds absolute
values ([J/m2], [kg/m2], x Mass_budget_factor); subsequent records hold the
per-step difference ([W/m2], [kg/m2/s], x Energy_budget_factor) -- the
conservation residual.

The global sum is volume-weighted (VMTR_VOLUME), reduced with COMM_Stat_sum. Like
the history th_prime diagnostic, the 2 pole singular cells are omitted (interior
only); nicamdc adds them, a ~1e-4 relative contribution to the absolute totals.
"""
import numpy as np
from pynicamdc.share.mod_adm import adm
from pynicamdc.share.mod_stdio import std
from pynicamdc.share.mod_process import prc
from pynicamdc.nhm.dynamics.kernels.diag import compute_diagnostics


class Embudget:

    def __init__(self):
        self.MNT_ON = False
        self.MNT_INTV = 1
        self.MNT_m_fname = 'BUDGET_mass.log'
        self.MNT_e_fname = 'BUDGET_energy.log'
        self._first = True
        self._old = {}                    # previous-step global sums (for the difference)

    def embudget_setup(self, fname_in, msc):
        import toml
        cnst, tim = msc.cnst, msc.tim
        if std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("\n+++ Module[embudget]/Category[nhm share]", file=log_file)

        with open(fname_in, 'r') as f:
            cnfs = toml.load(f)
        if 'embudgetparam' in cnfs:
            p = cnfs['embudgetparam']
            self.MNT_ON   = p.get('MNT_ON', self.MNT_ON)
            self.MNT_INTV = p.get('MNT_INTV', self.MNT_INTV)
        elif std.io_l:
            with open(std.fname_log, 'a') as log_file:
                print("*** EMBUDGETPARAM is not specified. use default.", file=log_file)

        if not self.MNT_ON:
            return

        radius = cnst.CONST_RADIUS
        pi = cnst.CONST_PI
        area = 4.0 * pi * radius * radius
        self.Mass_budget_factor   = 1.0 / area                                    # [J]      -> [J/m2]
        self.Energy_budget_factor = 1.0 / (tim.TIME_dtl * self.MNT_INTV * area)   # [J/step] -> [W/m2]

        if prc.prc_ismaster:                      # (re)create the two log files
            open(self.MNT_e_fname, 'w').close()
            open(self.MNT_m_fname, 'w').close()

        self._store_var(msc)

    def embudget_monitor(self, msc):
        if not self.MNT_ON:
            return
        if msc.tim.TIME_cstep % self.MNT_INTV == 0:
            self._store_var(msc)

    def _gsum(self, var, vmtr):
        # volume-weighted global sum over the interior (nicamdc GTL_global_sum, pole omitted).
        gmin, gmax = adm.ADM_gmin, adm.ADM_gmax
        kmin, kmax = adm.ADM_kmin, adm.ADM_kmax
        sl = (slice(gmin, gmax + 1), slice(gmin, gmax + 1), slice(kmin, kmax + 1), slice(None))
        local = np.sum(var[sl] * vmtr.VMTR_VOLUME[sl])
        return float(self._comm.Comm_Stat_sum(np.asarray(local, dtype=var.dtype)))

    def _store_var(self, msc):
        prgv, vmtr, rcnf, cnst = msc.prgv, msc.vmtr, msc.rcnf, msc.cnst
        self._comm = msc.comm

        # re-derive the diagnostic state from the current prognostic (prgvar_get_withdiag)
        PROG  = prgv.PRG_var[:, :, :, :, 0:6]
        PROGq = prgv.PRG_var[:, :, :, :, 6:]
        rho, DIAG, ein, q, cv, qd = compute_diagnostics(
            PROG, PROGq, msc.dyn.DIAG, vmtr.VMTR_GSGAM2, vmtr.VMTR_C2Wfact, rcnf.CVW,
            cfg=msc.dyn._diag_cfg, xp=np,
        )
        tem = DIAG[:, :, :, :, msc.dyn._diag_cfg.I_tem]

        CVdry, LHV, LHF = cnst.CONST_CVdry, cnst.CONST_LHV, cnst.CONST_LHF
        NQW_STR, NQW_END = rcnf.NQW_STR, rcnf.NQW_END
        I_QV = getattr(rcnf, 'I_QV', -1)
        I_ice = {getattr(rcnf, n, -1) for n in ('I_QI', 'I_QS', 'I_QG')}
        I_liq = {getattr(rcnf, n, -1) for n in ('I_QC', 'I_QR')}

        # ----- energy -----
        pot = self._gsum(rho * vmtr.VMTR_PHI, vmtr)

        eint = qd * CVdry * tem
        for nq in range(NQW_STR, NQW_END + 1):
            eint = eint + q[:, :, :, :, nq] * rcnf.CVW[nq - NQW_STR] * tem
            if nq == I_QV:
                eint = eint + q[:, :, :, :, nq] * LHV
            if nq in I_ice:
                eint = eint - q[:, :, :, :, nq] * LHF
        eint = self._gsum(rho * eint, vmtr)

        rhogkin, _ = msc.cnvv.cnvvar_rhogkin(
            PROG[:, :, :, :, 0], prgv.PRG_var_pl[:, :, :, 0],
            PROG[:, :, :, :, 1], prgv.PRG_var_pl[:, :, :, 1],
            PROG[:, :, :, :, 2], prgv.PRG_var_pl[:, :, :, 2],
            PROG[:, :, :, :, 3], prgv.PRG_var_pl[:, :, :, 3],
            PROG[:, :, :, :, 4], prgv.PRG_var_pl[:, :, :, 4],
            cnst, vmtr, msc.bk.ndtype,
        )
        ekin = self._gsum(rhogkin * vmtr.VMTR_RGSGAM2, vmtr)
        etot = pot + eint + ekin

        # ----- mass -----
        mdry = self._gsum(rho * qd, vmtr)
        rhoq = {nq: self._gsum(rho * q[:, :, :, :, nq], vmtr) for nq in range(NQW_STR, NQW_END + 1)}
        mtot = sum(rhoq.values())
        mvap = rhoq.get(I_QV, 0.0)
        mliq = sum(v for nq, v in rhoq.items() if nq in I_liq)
        mice = sum(v for nq, v in rhoq.items() if nq in I_ice)

        cur = dict(pot=pot, int=eint, kin=ekin, tot=etot,
                   dry=mdry, vap=mvap, liq=mliq, ice=mice, wtot=mtot)
        self._write(msc, cur)

    def _write(self, msc, cur):
        keys_e = ('pot', 'int', 'kin', 'tot')
        keys_m = ('dry', 'vap', 'liq', 'ice', 'wtot')
        if self._first:
            f = self.Mass_budget_factor
            diff = {k: cur[k] * f for k in cur}
        else:
            f = self.Energy_budget_factor
            diff = {k: (cur[k] - self._old[k]) * f for k in cur}

        if prc.prc_ismaster:
            cstep = msc.tim.TIME_cstep
            day = cstep * msc.tim.TIME_dtl / 86400.0
            with open(self.MNT_e_fname, 'a') as fe:
                if self._first:
                    fe.write(f"{'#STEP':>6}{'Day':>16}{'Eng(pot)':>16}{'Eng(int)':>16}"
                             f"{'Eng(kin)':>16}{'Eng(tot)':>16}\n")
                fe.write(f"{cstep:6d}{day:16.8E}" + "".join(f"{diff[k]:16.8E}" for k in keys_e) + "\n")
            with open(self.MNT_m_fname, 'a') as fm:
                if self._first:
                    fm.write(f"{'#STEP':>6}{'Day':>16}{'dry air mass':>16}{'water mass(g)':>16}"
                             f"{'water mass(l)':>16}{'water mass(s)':>16}{'water mass(t)':>16}\n")
                fm.write(f"{cstep:6d}{day:16.8E}" + "".join(f"{diff[k]:16.8E}" for k in keys_m) + "\n")

        self._old = cur
        self._first = False


embudget = Embudget()
