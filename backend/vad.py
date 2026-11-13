"""Streaming Silero VAD + sentence segmenter.

Audio arrives as 16 kHz mono int16 PCM in arbitrary-sized chunks. The
segmenter runs the VAD on 32 ms frames and cuts a segment when the speaker
pauses long enough, or when the segment grows too long.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

SR = 16000
FRAME = 512  # 32 ms at 16 kHz, the frame size Silero expects
FRAME_MS = FRAME * 1000 / SR
CONTEXT = 64  # Silero expects the last 64 samples of the previous frame prepended


def bundled_model_path() -> str:
    """The Silero v6 ONNX model shipped inside faster-whisper."""
    from faster_whisper.utils import get_assets_path

    return os.path.join(get_assets_path(), "silero_vad_v6.onnx")


class SileroModel:
    """Shared ONNX session; per-stream state lives in SileroState."""

    def __init__(self, path: str | None = None):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = ort.InferenceSession(
            path or bundled_model_path(), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def new_state(self) -> "SileroState":
        return SileroState(self)


class SileroState:
    def __init__(self, model: SileroModel):
        self.model = model
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CONTEXT), dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame.reshape(1, -1)], axis=1).astype(np.float32)
        out, self.h, self.c = self.model.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = x[:, -CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])


@dataclass
class SegmenterParams:
    threshold: float = 0.5
    min_silence_ms: int = 600
    max_segment_s: float = 8.0
    pre_roll_ms: int = 300
    # Short bursts (coughs, clicks, chairs) are where Whisper hallucinates most.
    min_speech_ms: int = 400
    # After soft_ratio * max_segment_s, a shorter pause is enough to cut.
    soft_ratio: float = 0.6
    soft_silence_ms: int = 200
    keep_tail_ms: int = 200
    # A segment that reaches max_segment_s while speech goes on is cut where speech was least
    # likely within this long before the limit (a gap between words, with luck), not at the limit
    # itself, which usually falls inside a word; the audio after that point opens the next segment.
    hard_cut_lookback_ms: int = 2000
    # Silero misses quiet / far-field speech, so the VAD (only) sees a level-normalized copy.
    # The tracked level follows loud frames quickly and decays slowly, so the gain is set by
    # the speaker's level, not by the noise between words.
    agc_target_rms: float = 0.05
    agc_max_gain: float = 20.0
    agc_release_s: float = 4.0  # half-life of the tracked level
    # Silero's recurrent state drifts on long unbroken classroom audio until it stops
    # detecting speech at all; reset it at segment boundaries and at least this often.
    vad_reset_s: float = 10.0


@dataclass
class Segment:
    start: float  # seconds since stream start
    audio: np.ndarray  # float32 mono 16 kHz
    index: int  # increments per segment; ties live partials to their final result
    run: int = 0  # capture run (开始 … 停止) it belongs to; batches never mix runs


class Segmenter:
    def __init__(self, model: SileroModel, params: SegmenterParams | None = None):
        self.vad = model.new_state()
        self.params = params or SegmenterParams()
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self.level = 0.0
        self.peak_prob = 0.0  # highest speech probability since take_peak_prob(), for the UI
        self.pending = np.zeros(0, dtype=np.float32)
        self.frames_seen = 0
        self.frames_since_reset = 0
        self.preroll: deque[np.ndarray] = deque(maxlen=max(1, round(self.params.pre_roll_ms / FRAME_MS)))
        self.preroll_probs: deque[float] = deque(maxlen=self.preroll.maxlen)
        self.in_speech = False
        self.seg_frames: list[np.ndarray] = []
        self.seg_probs: list[float] = []  # speech probability of each frame in seg_frames
        self.seg_carried = False  # the segment in progress began with audio left by a hard cut
        self.seg_start_frame = 0
        self.speech_frames = 0
        self.silence_frames = 0
        self.index = 0
        self.dropped: list[int] = []  # indices of segments discarded as too short

    @property
    def speaking(self) -> bool:
        return self.in_speech

    def current_audio(self) -> np.ndarray | None:
        """Audio of the segment in progress, for live partial transcription."""
        if not self.in_speech or not self.seg_frames:
            return None
        return np.concatenate(self.seg_frames)

    def committed_frames(self) -> int:
        """Frames before which no audio can still become part of a future segment: the start of
        the segment in progress, or of the pre-roll a new segment would begin with. A client that
        reconnects resends audio from here (see ws_audio)."""
        if self.in_speech:
            return self.seg_start_frame
        return self.frames_seen - len(self.preroll)

    def take_peak_prob(self) -> float:
        out, self.peak_prob = self.peak_prob, 0.0
        return out

    def take_dropped(self) -> list[int]:
        out, self.dropped = self.dropped, []
        return out

    def feed(self, pcm16: bytes) -> list[Segment]:
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self.pending = np.concatenate([self.pending, samples])
        out: list[Segment] = []
        n = len(self.pending) // FRAME
        for i in range(n):
            seg = self._step(self.pending[i * FRAME : (i + 1) * FRAME])
            if seg is not None:
                out.append(seg)
        self.pending = self.pending[n * FRAME :]
        return out

    def flush(self) -> list[Segment]:
        """Close any segment in progress (called when capture stops)."""
        seg = self._emit() if self.in_speech else None
        self.in_speech = False
        return [seg] if seg is not None else []

    def _step(self, frame: np.ndarray) -> Segment | None:
        p = self.params
        # Reset while silent once the state is old; mid-speech only as a failsafe for long segments
        # (segment boundaries reset it too, see _emit).
        age_ms = self.frames_since_reset * FRAME_MS
        if age_ms >= p.vad_reset_s * 1000 * (2 if self.in_speech else 1):
            self._reset_vad()
        self.frames_since_reset += 1
        prob = self.vad.prob(self._agc(frame))
        self.peak_prob = max(self.peak_prob, prob)
        self.frames_seen += 1

        if not self.in_speech:
            self.preroll.append(frame)
            self.preroll_probs.append(prob)
            if prob >= p.threshold:
                self.in_speech = True
                self.index += 1
                self.seg_frames = list(self.preroll)
                self.seg_probs = list(self.preroll_probs)
                self.seg_carried = False
                self.seg_start_frame = self.frames_seen - len(self.seg_frames)
                self.preroll.clear()
                self.preroll_probs.clear()
                self.speech_frames = 1
                self.silence_frames = 0
            return None

        self.seg_frames.append(frame)
        self.seg_probs.append(prob)
        if prob >= p.threshold:
            self.speech_frames += 1
            self.silence_frames = 0
        elif prob < p.threshold - 0.15:
            self.silence_frames += 1

        dur_s = len(self.seg_frames) * FRAME / SR
        silence_ms = self.silence_frames * FRAME_MS
        if silence_ms >= p.min_silence_ms or (
            dur_s >= p.max_segment_s * p.soft_ratio and silence_ms >= p.soft_silence_ms
        ):
            seg = self._emit()
            self.in_speech = False
            return seg
        if dur_s >= p.max_segment_s:
            # Hard cut while speech continues: emit up to the likeliest gap between words and
            # start the next segment with the audio after it.
            k = self._cut_point()
            rest_frames, rest_probs = self.seg_frames[k:], self.seg_probs[k:]
            self.seg_frames, self.seg_probs = self.seg_frames[:k], self.seg_probs[:k]
            self.speech_frames, self.silence_frames = self._count(self.seg_probs)
            seg = self._emit()
            self.index += 1
            self.seg_frames, self.seg_probs = rest_frames, rest_probs
            self.seg_carried = bool(rest_frames)
            self.seg_start_frame = self.frames_seen - len(rest_frames)
            self.speech_frames, self.silence_frames = self._count(rest_probs)
            return seg
        return None

    def _cut_point(self) -> int:
        """Where to split the segment in progress for a hard cut: the number of frames to emit.

        The frame where speech was least likely (averaged with its two neighbours, so one odd
        frame doesn't decide) among the last hard_cut_lookback_ms, but never in the first half of
        the segment. Of equally unlikely places the frame that is itself least likely wins, then
        the latest. The chosen frame opens the next segment. When no place is less likely than
        the limit itself (steady speech), the whole segment is emitted, as before."""
        probs = self.seg_probs
        n = len(probs)
        look = min(round(self.params.hard_cut_lookback_ms / FRAME_MS), n // 2)

        def score(i: int) -> tuple[float, float]:
            window = probs[max(0, i - 1) : i + 2]
            # Rounded: the same three values summed in another order must compare equal.
            return round(sum(window) / len(window), 6), probs[i]

        limit = score(n - 1)
        best, k = limit, n
        for i in range(n - look, n - 1):
            s = score(i)
            if s < limit and s <= best:
                best, k = s, i
        return k

    def _count(self, probs: list[float]) -> tuple[int, int]:
        """Speech frames and trailing silence frames of a run of frames, counted as _step does."""
        p = self.params
        speech = silence = 0
        for prob in probs:
            if prob >= p.threshold:
                speech += 1
                silence = 0
            elif prob < p.threshold - 0.15:
                silence += 1
        return speech, silence

    def _agc(self, frame: np.ndarray) -> np.ndarray:
        p = self.params
        rms = float(np.sqrt(np.mean(frame * frame)))
        if rms > self.level:
            self.level += (rms - self.level) * 0.3
        else:
            self.level *= 0.5 ** (FRAME_MS / 1000 / p.agc_release_s)
        gain = min(p.agc_max_gain, max(1.0, p.agc_target_rms / max(self.level, 1e-6)))
        return frame if gain == 1.0 else np.clip(frame * gain, -1.0, 1.0)

    def _reset_vad(self) -> None:
        self.vad.reset()
        self.frames_since_reset = 0

    def _emit(self) -> Segment | None:
        p = self.params
        if self.frames_since_reset * FRAME_MS >= 2000:  # a fresh state needs a moment to warm up
            self._reset_vad()
        frames = self.seg_frames
        keep_tail = round(p.keep_tail_ms / FRAME_MS)
        drop = max(0, self.silence_frames - keep_tail)
        if drop:
            frames = frames[:-drop]
        start = self.seg_start_frame * FRAME / SR
        self.seg_frames = []
        self.seg_probs = []
        # Too little speech is dropped as noise, except what a hard cut left over: that is the
        # end of a sentence already under way, however short.
        too_short = self.speech_frames * FRAME_MS < p.min_speech_ms and not (self.seg_carried and self.speech_frames)
        self.seg_carried = False
        if too_short or not frames:
            self.dropped.append(self.index)
            return None
        return Segment(start=start, audio=np.concatenate(frames), index=self.index)
