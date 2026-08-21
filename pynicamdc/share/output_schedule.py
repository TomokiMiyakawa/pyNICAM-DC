"""Single source of truth for the PRG (zarr history) output schedule.

nicamdc writes a periodic history snapshot when ``mod(TIME_CSTEP, interval) == 0``
(mod_history.f90 ``history_out``), plus an optional initial-condition frame at
``TIME_CSTEP = 0`` (``doout_step0``). pyNICAM mirrors this exactly.

Kept deliberately dependency-free (pure integer arithmetic, no numpy/zarr/mpi) so
it is importable in the minimal CI environment and unit-testable in isolation --
both the driver time loop (fire predicate + FUSE_TIMELOOP chunk-trim guard) and
mod_io (zarr time-axis sizing) import from here, so the schedule can never drift
between "when we write" and "how many slots we allocate".

Driver mapping: the time loop uses a 0-based index ``n``; after that step's
``TIME_advance`` the clock reads ``TIME_cstep = n + 1``, so the per-step fire
predicate is ``prg_output_fires(n + 1, interval)``.
"""


def prg_output_fires(cstep, interval):
    """True iff the periodic PRG output writes at this 1-based ``TIME_cstep``.

    Matches nicamdc ``mod(TIME_CSTEP, interval) == 0`` -> writes land at
    ``TIME_cstep = interval, 2*interval, ...`` (e.g. 60, 120). The step-0 frame
    (``TIME_cstep = 0``) is handled separately by the caller via PRGout_step0.
    """
    return cstep >= 1 and cstep % interval == 0


def boundary_fires(cstep, intervals):
    """True iff ANY of the given interval schedules fires at this 1-based
    ``TIME_cstep`` -- i.e. this is a *boundary step*, after which the host must
    see the state (a 3D/2D output writes, or the budget monitor samples).

    The FUSE_TIMELOOP chunk trim must not run past a boundary step: everything
    in ``intervals`` fires host-side after ``TIME_advance``, so a fused chunk
    spanning one would silently skip it (the budget monitor was dropped exactly
    this way before this predicate existed). Callers pass the ACTIVE intervals
    only (e.g. ``MNT_INTV`` only when ``MNT_ON``); non-positive or falsy entries
    are ignored here as a second line of defence.
    """
    return any(iv and iv > 0 and prg_output_fires(cstep, iv) for iv in intervals)


def resolve_chunk(cap, intervals, lstep_max):
    """The fusion chunk length K (S2, FUSION_SCHEDULE_PLAN), pure arithmetic.

    ``cap`` is ``PYNICAM_TIMELOOP_CHUNK`` (default 1). Intervals that never fire
    within ``lstep_max`` (or are falsy/non-positive) are dropped; ``g`` is the
    gcd of the survivors, so every boundary step is a multiple of ``g`` and any
    ``K | g`` tiles the gaps between boundaries exactly.

        K = 1                                   if cap <= 1  (the default)
          = cap                                 if no interval fires in the run
          = max{d <= cap : g mod d == 0}        otherwise

    ``K == 1`` with ``cap > 1`` means fusion cannot engage beyond one-step
    chunks (no divisor of g at or below the cap -- e.g. a prime interval);
    callers should surface that diagnosis rather than compute it silently.
    """
    import math
    active = [iv for iv in intervals if iv and 0 < iv <= lstep_max]
    if cap <= 1:
        return 1
    if not active:
        return cap
    g = math.gcd(*active)
    return max(d for d in range(1, cap + 1) if g % d == 0)


def prg_output_nslots(lstep_max, interval, step0):
    """Number of zarr time slots = periodic fires in ``[1, lstep_max]`` plus the
    optional step-0 frame. This MUST equal the number of ``IO_PRGstep`` writes or
    the zarr region-write raises "changing dimension size".

    Kept >= 1 so the time axis is always valid: ``interval > lstep_max`` yields 0
    periodic fires (output effectively disabled) but still leaves one slot.
    """
    periodic = lstep_max // interval
    return max(1, periodic + (1 if step0 else 0))
