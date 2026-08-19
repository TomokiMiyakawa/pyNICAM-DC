"""Regression guard for the PRG (zarr history) output SCHEDULE / phase.

This pins the property that regressed silently once already: the periodic output
must fire at TIME_cstep = interval, 2*interval, ... (nicamdc mod_history.f90
`mod(TIME_CSTEP, interval) == 0`), NOT one period-phase off. The tier2/tutorial
goldens compare only the final state (PYNICAM_TIMELOOP_DUMP), so they are blind to
*which* intermediate steps get written -- exactly why a +2-step phase error went
unnoticed. These pure-arithmetic checks close that gap with no model run.

The driver maps its 0-based loop index n to TIME_cstep = n+1 (post TIME_advance),
so the driver's fire predicate is prg_output_fires(n+1, interval); the same
function feeds mod_io's zarr time-axis sizing, so this also guards nt == #writes.
"""
from pynicamdc.share.output_schedule import prg_output_fires, prg_output_nslots


def _fire_csteps(lstep_max, interval):
    return [c for c in range(1, lstep_max + 1) if prg_output_fires(c, interval)]


def test_phase_matches_nicamdc():
    # interval=60 -> writes at cstep 60,120,180,240 (nicamdc mod(cstep,60)==0)
    assert _fire_csteps(240, 60) == [60, 120, 180, 240]


def test_rejects_old_buggy_phase():
    # the pre-fix bug fired at cstep = 2, interval+2, ... ((n-1)%interval==0 on the
    # 0-based index while cstep = n+1). Those must NOT be output steps now.
    assert not prg_output_fires(2, 60)
    assert not prg_output_fires(62, 60)
    assert not prg_output_fires(7, 5)


def test_driver_loop_index_mapping():
    # interval=5, lstep=15: driver 0-based n fires at n=4,9,14 -> cstep 5,10,15.
    fired_n = [n for n in range(15) if prg_output_fires(n + 1, 5)]
    assert fired_n == [4, 9, 14]


def test_interval_equal_lstep_is_final_step_only():
    # one snapshot, at the final step (matches nicamdc for interval == lstep_max).
    assert _fire_csteps(48, 48) == [48]


def test_interval_one_fires_every_step():
    assert _fire_csteps(4, 1) == [1, 2, 3, 4]


def test_nslots_equals_write_count():
    # nt MUST equal (#periodic writes) + step0, or the zarr region-write raises
    # "changing dimension size". Kept >= 1 for a valid time axis.
    for lstep, iv, step0 in [(15, 5, True), (66, 6, False), (120, 60, True),
                             (48, 48, True), (4, 1, False), (51, 1000, False)]:
        writes = len(_fire_csteps(lstep, iv)) + (1 if step0 else 0)
        assert prg_output_nslots(lstep, iv, step0) == max(1, writes)


def test_step0_adds_exactly_one_frame():
    assert (prg_output_nslots(120, 60, True)
            == prg_output_nslots(120, 60, False) + 1)


# ---- boundary predicate (S1, FUSION_SCHEDULE_PLAN) --------------------------

from pynicamdc.share.output_schedule import boundary_fires


def _boundary_csteps(lstep_max, intervals):
    return [c for c in range(1, lstep_max + 1) if boundary_fires(c, intervals)]


def test_boundary_is_union_of_schedules():
    # 3D interval 12 + budget 8 -> boundaries at multiples of either
    assert _boundary_csteps(24, [12, 8]) == [8, 12, 16, 24]


def test_boundary_reduces_to_output_when_monitor_off():
    # caller omits MNT_INTV when MNT_ON is false; single interval == prg schedule
    assert _boundary_csteps(36, [12]) == [12, 24, 36]


def test_boundary_ignores_inactive_intervals():
    # 0/None entries (defensive) never fire and never raise
    assert _boundary_csteps(24, [12, 0, None]) == [12, 24]


def test_boundary_mnt_every_step_blocks_all_chunks():
    # MNT_INTV=1 (the mod_embudget default) makes EVERY step a boundary: with the
    # monitor on at that setting, fusion must degrade to per-step, not skip it.
    assert _boundary_csteps(4, [1000, 1]) == [1, 2, 3, 4]


def test_boundary_phase_matches_fire_phase():
    # same 1-based cstep convention as prg_output_fires: no off-by-one
    assert not boundary_fires(2, [60])
    assert boundary_fires(60, [60])
