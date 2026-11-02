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
    min_speech_ms: int = 250
    # After soft_ratio * max_segment_s, a shorter pause is enough to cut.
    soft_ratio: float = 0.6
    soft_silence_ms: int = 200
    keep_tail_ms: int = 200


@dataclass
class Segment:
    start: float  # seconds since stream start
    audio: np.ndarray  # float32 mono 16 kHz
    index: int  # increments per segment; ties live partials to their final result


class Segmenter:
    def __init__(self, model: SileroModel, params: SegmenterParams | None = None):
        self.vad = model.new_state()
        self.params = params or SegmenterParams()
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self.pending = np.zeros(0, dtype=np.float32)
        self.frames_seen = 0
        self.preroll: deque[np.ndarray] = deque(maxlen=max(1, round(self.params.pre_roll_ms / FRAME_MS)))
        self.in_speech = False
        self.seg_frames: list[np.ndarray] = []
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
        prob = self.vad.prob(frame)
        self.frames_seen += 1

        if not self.in_speech:
            self.preroll.append(frame)
            if prob >= p.threshold:
                self.in_speech = True
                self.index += 1
                self.seg_frames = list(self.preroll)
                self.seg_start_frame = self.frames_seen - len(self.seg_frames)
                self.preroll.clear()
                self.speech_frames = 1
                self.silence_frames = 0
            return None

        self.seg_frames.append(frame)
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
            # Hard cut while speech continues: start the next segment right here.
            seg = self._emit()
            self.index += 1
            self.seg_frames = []
            self.seg_start_frame = self.frames_seen
            self.speech_frames = 0
            self.silence_frames = 0
            return seg
        return None

    def _emit(self) -> Segment | None:
        p = self.params
        frames = self.seg_frames
        keep_tail = round(p.keep_tail_ms / FRAME_MS)
        drop = max(0, self.silence_frames - keep_tail)
        if drop:
            frames = frames[:-drop]
        start = self.seg_start_frame * FRAME / SR
        self.seg_frames = []
        if self.speech_frames * FRAME_MS < p.min_speech_ms or not frames:
            self.dropped.append(self.index)
            return None
        return Segment(start=start, audio=np.concatenate(frames), index=self.index)
