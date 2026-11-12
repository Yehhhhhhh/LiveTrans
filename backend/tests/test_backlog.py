import asyncio

import numpy as np
import pytest

import server
from server import MERGE_GAP_S, MERGE_MAX_S, merge_segments, take_batch
from vad import Segment

SR = 16000


def seg(index: int, seconds: float, start: float | None = None) -> Segment:
    audio = np.full(round(seconds * SR), index, dtype=np.float32)
    return Segment(start=index * 10.0 if start is None else start, audio=audio, index=index)


def queue_of(*segs: Segment) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    for s in segs:
        q.put_nowait(s)
    return q


def test_no_merge_while_keeping_up():
    q = queue_of(seg(2, 3))  # one waiting: below MERGE_BACKLOG
    items, carry = take_batch(seg(1, 3), q)
    assert [s.index for s in items] == [1]
    assert carry is None
    assert q.qsize() == 1


def test_backlog_is_merged():
    q = queue_of(seg(2, 3), seg(3, 3), seg(4, 3))
    items, carry = take_batch(seg(1, 3), q)
    assert [s.index for s in items] == [1, 2, 3, 4]
    assert carry is None
    assert q.empty()


def test_merge_stops_at_max_length_and_carries_the_rest():
    per = 7.0  # 3 fit in 24 s with gaps, the 4th doesn't
    q = queue_of(seg(2, per), seg(3, per), seg(4, per), seg(5, per))
    items, carry = take_batch(seg(1, per), q)
    assert [s.index for s in items] == [1, 2, 3]
    assert carry.index == 4
    assert [q.get_nowait().index] == [5]
    merged = merge_segments(items)
    assert len(merged.audio) <= MERGE_MAX_S * SR


def test_merged_segment_keeps_order_start_and_last_index():
    items = [seg(1, 1.0, start=5.0), seg(2, 2.0), seg(3, 0.5)]
    m = merge_segments(items)
    gap = round(MERGE_GAP_S * SR)
    assert m.index == 3
    assert m.start == 5.0
    assert len(m.audio) == SR * 3.5 + 2 * gap
    # audio of segment 1, silence, segment 2, silence, segment 3
    assert m.audio[0] == 1 and m.audio[SR] == 0 and m.audio[SR + gap] == 2
    assert m.audio[-1] == 3


def test_single_segment_is_passed_through():
    s = seg(7, 2)
    assert merge_segments([s]) is s


def test_merge_threshold_constant():
    # Merging changes language detection granularity; keep it for real backlogs only.
    assert server.MERGE_BACKLOG >= 2


def test_batches_never_mix_capture_runs():
    # 开始 pressed again while segments from the previous run still wait (e.g. model loading).
    old = [seg(i, 2) for i in (2, 3)]
    new = [seg(i, 2) for i in (1, 2)]
    for s in new:
        s.run = 1
    q = queue_of(*old, *new)
    items, carry = take_batch(seg(1, 2), q)
    assert [(s.run, s.index) for s in items] == [(0, 1), (0, 2), (0, 3)]
    assert (carry.run, carry.index) == (1, 1)
    assert merge_segments(items).run == 0


def test_duration_spans_the_real_pauses():
    # Segments at 0-2 s, 5-6 s and 10-12 s: merged audio holds two 0.3 s gaps (4.6 s),
    # but the batch covers 12 s of the lecture.
    items = [seg(1, 2, start=0.0), seg(2, 1, start=5.0), seg(3, 2, start=10.0)]
    assert len(merge_segments(items).audio) / SR == pytest.approx(5 + 2 * MERGE_GAP_S)
    assert server.span_s(items) == pytest.approx(12.0)
    assert server.span_s([seg(7, 3, start=4.0)]) == pytest.approx(3.0)
