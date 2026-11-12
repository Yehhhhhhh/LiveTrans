"""Segmenter: where a segment that reaches max_segment_s while speech goes on is cut."""

import numpy as np

from test_resume import ScriptedVad
from vad import FRAME, FRAME_MS, SR, Segmenter, SegmenterParams

MAX_S = 4.0
MAX_FRAMES = int(MAX_S * SR / FRAME)  # 125: the frame count at which the hard cut happens


def numbered(n, first=0):
    """n frames whose samples all hold the frame's own number, so audio can be traced back."""
    out = np.repeat(np.arange(first, first + n, dtype=np.int16), FRAME)
    return out.tobytes()


def frame_numbers(audio):
    return [round(float(x) * 32768) for x in audio[::FRAME]]


def run(probs, n=None):
    seg = Segmenter(ScriptedVad(probs), SegmenterParams(max_segment_s=MAX_S))
    out = seg.feed(numbered(n or len(probs)))
    return seg, out


def test_hard_cut_falls_in_the_dip_before_the_limit():
    probs = [0.9] * 200
    probs[100:103] = [0.6, 0.4, 0.6]  # a short gap between words, 0.8 s before the limit
    seg, out = run(probs)
    assert len(out) == 1
    # Up to the gap; the gap frame itself opens the next segment.
    assert frame_numbers(out[0].audio) == list(range(0, 101))
    assert seg.speaking
    assert frame_numbers(seg.current_audio()) == list(range(101, 200))


def test_no_audio_is_lost_or_repeated_across_hard_cuts():
    rng = np.random.default_rng(1)
    probs = [float(p) for p in rng.uniform(0.6, 1.0, 600)]
    seg, out = run(probs)
    out += seg.flush()
    assert len(out) >= 4
    numbers = [n for s in out for n in frame_numbers(s.audio)]
    assert numbers == list(range(600))
    assert all(len(s.audio) <= MAX_S * SR for s in out)
    # Indices and start times follow the cuts.
    assert [s.index for s in out] == list(range(1, len(out) + 1))
    assert [round(s.start * SR / FRAME) for s in out] == [frame_numbers(s.audio)[0] for s in out]


def test_steady_speech_is_still_cut_at_the_limit():
    seg, out = run([0.9] * 200)
    assert len(out) == 1
    # Nothing to choose between the frames: the whole segment goes out, none is carried over.
    assert frame_numbers(out[0].audio) == list(range(MAX_FRAMES))
    assert frame_numbers(seg.current_audio()) == list(range(MAX_FRAMES, 200))


def test_stopping_right_after_a_hard_cut_loses_nothing():
    # Steady speech: stop on the very frame of the cut.
    seg, out = run([0.9] * MAX_FRAMES)
    out += seg.flush()
    assert [n for s in out for n in frame_numbers(s.audio)] == list(range(MAX_FRAMES))

    # A gap 3 frames before the limit: the 3 carried frames are far less than min_speech_ms,
    # but they are the end of the sentence, not a noise burst.
    probs = [0.9] * MAX_FRAMES
    probs[MAX_FRAMES - 4] = 0.4
    seg, out = run(probs)
    out += seg.flush()
    assert len(out) == 2
    assert [n for s in out for n in frame_numbers(s.audio)] == list(range(MAX_FRAMES))
    assert not seg.take_dropped()


def test_a_short_burst_is_still_dropped_after_an_earlier_hard_cut():
    probs = [0.9] * 200 + [0.0] * 60 + [0.9] * 5 + [0.0] * 60
    seg, out = run(probs)
    # The long speech (cut once), then the 5-frame burst on its own: dropped as before.
    assert len(out) == 2
    assert frame_numbers(out[-1].audio)[-1] < 262
    assert len(seg.take_dropped()) == 1


def test_a_dip_older_than_the_lookback_or_in_the_first_half_is_ignored():
    probs = [0.9] * 200
    probs[30] = 0.4  # 3 s before the limit: outside the 2 s lookback
    seg, out = run(probs)
    assert len(frame_numbers(out[0].audio)) == MAX_FRAMES

    lookback = round(SegmenterParams().hard_cut_lookback_ms / FRAME_MS)
    assert MAX_FRAMES - lookback < 70  # 70 is inside the lookback
    probs = [0.9] * 200
    probs[70] = 0.4
    seg, out = run(probs)
    assert frame_numbers(out[0].audio)[-1] == 69


def test_committed_frames_follow_the_carried_audio():
    probs = [0.9] * 200
    probs[100:103] = [0.6, 0.4, 0.6]
    seg, out = run(probs, n=MAX_FRAMES)
    assert len(out) == 1
    # The next segment began at the gap, not at the limit: a reconnecting page must resend from there.
    assert seg.seg_start_frame == 101
    assert seg.committed_frames() == 101


def test_the_carried_audio_counts_as_speech_when_the_segment_ends_soon_after():
    probs = [0.9] * 200
    probs[100:103] = [0.6, 0.4, 0.6]
    probs[130:] = [0.0] * 70  # the speaker stops 5 frames after the cut
    seg, out = run(probs)
    assert len(out) == 2
    assert not seg.take_dropped()
    # Second segment: from the gap to the end of speech plus the kept tail.
    assert frame_numbers(out[1].audio)[0] == 101
    assert frame_numbers(out[1].audio)[-1] >= 130
