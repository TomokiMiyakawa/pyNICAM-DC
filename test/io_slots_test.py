"""Output-store capacity -- see Io._reserve_slot / Io.IO_finalize in mod_io.py.

IO_setup sizes the store from the OUTPUT SCHEDULE, which is exact for scheduled
output. An unscheduled snapshot (pyNICAM.write()) has no slot reserved, so the time
axis grows to take it instead of the write being dropped, and the tail that block
growth leaves is trimmed at finalize.

The zarr resize itself needs a store, ranks and a barrier; what is unit-testable --
and what actually decides whether a write lands -- is when growth fires and how far
it goes. _resize_axis is stubbed out here and its calls recorded.
"""
import pytest

from pynicamdc.share.mod_io import Io


def _io(nt=0, nt2d=0):
    io = Io()
    io._nt, io._it = nt, 0
    io._nt_2d, io._it_2d = nt2d, 0
    io._out_names, io._diag_names, io._diag_names_2d = ["RHOG"], [], []
    io.calls = []
    io._resize_axis = lambda axis, names, n: io.calls.append((axis, n))
    return io


def test_a_reserved_slot_needs_no_growth():
    io = _io(nt=5)
    for it in range(5):
        assert io._reserve_slot(it, "time", io._out_names, io._nt) == 5
    assert io.calls == []


def test_growth_fires_only_past_the_allocation_and_in_blocks():
    io = _io(nt=5)
    b = io._GROW_BLOCK
    nt = io._nt
    for it in range(5):                      # inside the allocation
        nt = io._reserve_slot(it, "time", io._out_names, nt)
    assert io.calls == []

    nt = io._reserve_slot(5, "time", io._out_names, nt)   # first unscheduled write
    assert nt == b and io.calls == [("time", b)]

    for it in range(6, b):                   # the rest of that block is free
        assert io._reserve_slot(it, "time", io._out_names, nt) == b
    assert len(io.calls) == 1

    nt = io._reserve_slot(b, "time", io._out_names, nt)   # next block
    assert nt == 2 * b and len(io.calls) == 2


def test_growth_always_makes_room_for_the_write():
    # the point of the whole thing: after _reserve_slot, `it` is inside the axis
    io = _io(nt=0)
    nt = 0
    for it in range(25):
        nt = io._reserve_slot(it, "time", io._out_names, nt)
        assert it < nt, (it, nt)


def test_finalize_trims_the_unwritten_tail():
    io = _io(nt=8, nt2d=8)
    io._it, io._it_2d = 6, 3
    io.IO_finalize()
    assert io._nt == 6 and io._nt_2d == 3
    assert io.calls == [("time", 6), ("time2d", 3)]


def test_finalize_leaves_an_exactly_filled_store_alone():
    # a purely scheduled run fills its allocation -- the common case must be a no-op
    io = _io(nt=5, nt2d=5)
    io._it, io._it_2d = 5, 5
    io.IO_finalize()
    assert io.calls == []


def test_finalize_leaves_an_unwritten_axis_alone():
    # nothing written -> shrinking the axis to zero would destroy the store's shape
    # for no gain; leave the allocation as it is
    io = _io(nt=5, nt2d=5)
    io.IO_finalize()
    assert io.calls == []
