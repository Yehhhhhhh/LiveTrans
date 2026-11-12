"""Resuming a stream after a dropped connection: committed positions and the "resume" message."""

from contextlib import contextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from vad import FRAME, Segment, Segmenter, SegmenterParams

CHUNK = FRAME * 32  # 16384 samples, ~1 s


# ------------------------------------------------------------------ Segmenter.committed_frames


class ScriptedVad:
    """Speech probability per frame from a list; silence after it runs out."""

    def __init__(self, probs):
        self.probs = list(probs)

    def new_state(self):
        return self

    def reset(self):
        pass

    def prob(self, frame):
        return self.probs.pop(0) if self.probs else 0.0


def frames(n):
    return np.zeros(n * FRAME, dtype=np.int16).tobytes()


def test_committed_frames_idle_keeps_the_preroll():
    seg = Segmenter(ScriptedVad([]), SegmenterParams())
    seg.feed(frames(50))
    preroll = len(seg.preroll)
    assert preroll > 0
    assert seg.committed_frames() == 50 - preroll


def test_committed_frames_stops_at_the_segment_in_progress():
    seg = Segmenter(ScriptedVad([0.0] * 20 + [0.9] * 30), SegmenterParams())
    seg.feed(frames(50))
    assert seg.speaking
    assert seg.committed_frames() == seg.seg_start_frame
    assert seg.committed_frames() < 20  # the segment includes its pre-roll


# ------------------------------------------------------------------ /ws committed and resume


class FakeSegmenter:
    """Every audio message becomes one finished segment covering exactly that audio."""

    def __init__(self, model, params=None):
        self.params = params
        self.reset()

    def reset(self):
        self.index = 0
        self.pos = 0
        self.speaking = False
        self.frames_seen = 0

    def feed(self, pcm):
        n = len(pcm) // 2
        self.index += 1
        s = Segment(start=self.pos / 16000, audio=np.zeros(n, np.float32), index=self.index)
        self.pos += n
        return [s]

    def committed_frames(self):
        return self.pos // FRAME

    def flush(self):
        return []

    def take_dropped(self):
        return []

    def take_peak_prob(self):
        return 0.0

    def current_audio(self):
        return None


class FakeTranscriber:
    def __init__(self, status):
        self.status = status
        self.error = ""
        self.model_name = "fake"
        self.device = "cpu"
        self.compute_type = "int8"

    def transcribe(self, audio, language="auto", prompt="", context="", context_lang=""):
        return {"lang": "it", "lang_prob": 1.0, "text": "testo", "asr_ms": 1}


@pytest.fixture
def fake(monkeypatch):
    t = FakeTranscriber("ready")
    monkeypatch.setattr(server, "transcriber", t)
    monkeypatch.setattr(server, "Segmenter", FakeSegmenter)
    monkeypatch.setattr(server, "SileroModel", lambda: None)
    monkeypatch.setattr(server, "load_model_in_background", lambda: None)
    monkeypatch.setattr(server, "MODEL_WAIT_POLL_S", 0.01)
    return t


@contextmanager
def session(first="start", **extra):
    with TestClient(server.app, base_url="http://127.0.0.1:8765") as client, client.websocket_connect(
        "ws://127.0.0.1:8765/ws"
    ) as sock:
        assert sock.receive_json()["type"] == "status"
        sock.send_json({"type": "config", "asr": {"partial": False}})
        sock.send_json({"type": first, **extra})
        yield sock


def read_until(sock, pred, limit=200):
    seen = []
    for _ in range(limit):
        m = sock.receive_json()
        seen.append(m)
        if pred(m):
            return seen
    raise AssertionError(f"no matching message in {seen}")


def test_result_carries_the_committed_position(fake):
    with session() as sock:
        sock.send_bytes(b"\0" * (CHUNK * 2))
        seg = read_until(sock, lambda m: m["type"] == "segment")[-1]
        assert seg["committed"] == CHUNK
        assert seg["start"] == 0.0


def test_committed_never_passes_an_unanswered_segment(fake):
    fake.status = "loading"  # segments wait, unanswered
    with session() as sock:
        sock.send_bytes(b"\0" * (CHUNK * 2))
        sock.send_bytes(b"\0" * (CHUNK * 2))
        seen = read_until(sock, lambda m: m["type"] == "backlog" and m["pending"] == 2)
        assert all(m["committed"] == 0 for m in seen if "committed" in m)
        fake.status = "ready"
        first = read_until(sock, lambda m: m["type"] == "segment")[-1]
        assert first["index"] == 1
        # The second segment (starting at CHUNK) is still unanswered: committed stops there.
        assert first["committed"] == CHUNK
        second = read_until(sock, lambda m: m["type"] == "segment")[-1]
        assert second["index"] == 2
        assert second["committed"] == 2 * CHUNK


def test_resume_continues_positions_and_timestamps(fake):
    offset = 10 * 16000
    with session("resume", offset=offset) as sock:
        sock.send_bytes(b"\0" * (CHUNK * 2))
        seg = read_until(sock, lambda m: m["type"] == "segment")[-1]
        assert seg["start"] == 10.0
        assert seg["committed"] == offset + CHUNK


def test_resume_after_stop_transcribes_the_resent_audio(fake):
    """A page that stopped before its last speech was answered resumes the same run, resends that
    audio and stops again: the speech is transcribed with that run's positions."""
    offset = 5 * 16000
    with session("resume", offset=offset, run=3) as sock:
        sock.send_bytes(b"\0" * (CHUNK * 2))
        sock.send_json({"type": "stop"})
        seg = read_until(sock, lambda m: m["type"] == "segment")[-1]
        assert seg["run"] == 3
        assert seg["start"] == 5.0
        assert seg["committed"] == offset + CHUNK


def test_committed_positions_name_their_run(fake):
    """The page ignores positions of an earlier run, so each one says which run it belongs to."""
    with session("start", run=1) as sock:
        sock.send_bytes(b"\0" * (CHUNK * 2))
        seen = read_until(sock, lambda m: m["type"] == "segment")
        seen += read_until(sock, lambda m: m["type"] == "backlog" and m["pending"] == 0)
        assert {m["committed_run"] for m in seen if "committed" in m} == {1}
        sock.send_json({"type": "start", "run": 2})
        sock.send_bytes(b"\0" * (CHUNK * 2))
        seen = read_until(sock, lambda m: m["type"] == "segment")
        assert {m["committed_run"] for m in seen if "committed" in m} == {2}


def test_an_answered_run_is_reported_done_when_the_next_one_starts(fake):
    with session("start", run=1) as sock:
        sock.send_bytes(b"\0" * (CHUNK * 2))
        read_until(sock, lambda m: m["type"] == "segment")
        sock.send_json({"type": "stop"})
        sock.send_json({"type": "start", "run": 2})
        assert read_until(sock, lambda m: m["type"] == "run_done")[-1]["run"] == 1


def test_an_ended_run_still_queued_keeps_its_own_positions(fake):
    """Run 1 (resumed at 10 s) still waits for the model when run 2 starts: its result keeps run 1's
    clock, run 2's committed position ignores it, and run 1 is reported done once answered."""
    fake.status = "loading"
    offset = 10 * 16000
    with session("resume", offset=offset, run=1) as sock:
        sock.send_bytes(b"\0" * (CHUNK * 2))
        sock.send_json({"type": "stop"})
        sock.send_json({"type": "start", "run": 2})
        sock.send_bytes(b"\0" * (CHUNK * 2))
        seen = read_until(sock, lambda m: m["type"] == "backlog" and m["pending"] == 2)
        assert not [m for m in seen if m["type"] == "run_done"]
        # run 2's first segment starts at 0 and is unanswered: its committed stays at 0, not
        # at run 1's unanswered position
        assert [m["committed"] for m in seen if m.get("committed_run") == 2][-1] == 0

        fake.status = "ready"
        seen = read_until(sock, lambda m: m["type"] == "run_done")
        first = [m for m in seen if m["type"] == "segment"][0]
        assert (first["run"], first["start"]) == (1, 10.0)
        assert seen[-1]["run"] == 1
        second = read_until(sock, lambda m: m["type"] == "segment")[-1]
        assert (second["run"], second["start"]) == (2, 0.0)


def test_start_resets_the_offset(fake):
    with session("resume", offset=160000) as sock:
        sock.send_json({"type": "start"})
        sock.send_bytes(b"\0" * (CHUNK * 2))
        seg = read_until(sock, lambda m: m["type"] == "segment")[-1]
        assert seg["start"] == 0.0
        assert seg["committed"] == CHUNK


class FlakyTranscriber(FakeTranscriber):
    def __init__(self, failures):
        super().__init__("ready")
        self.failures = failures
        self.calls = 0

    def transcribe(self, audio, language="auto", prompt="", context="", context_lang=""):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("CUDA error: transient")
        return super().transcribe(audio, language, prompt)


def test_transient_failure_is_retried_and_the_audio_stays_uncommitted(fake, monkeypatch):
    t = FlakyTranscriber(failures=1)
    monkeypatch.setattr(server, "transcriber", t)
    monkeypatch.setattr(server, "TRANSCRIBE_RETRY_S", 0.05, raising=False)
    with session() as sock:
        sock.send_bytes(b"\0" * (CHUNK * 2))
        seen = read_until(sock, lambda m: m["type"] in ("segment", "error"))
        assert seen[-1]["type"] == "segment"  # recovered, no error shown
        assert t.calls == 2
        # until it was answered, the page was never told to drop that audio
        assert all(m["committed"] == 0 for m in seen[:-1] if "committed" in m)
        assert seen[-1]["committed"] == CHUNK


def test_persistent_failure_gives_up_after_the_last_attempt(fake, monkeypatch):
    t = FlakyTranscriber(failures=99)
    monkeypatch.setattr(server, "transcriber", t)
    monkeypatch.setattr(server, "TRANSCRIBE_RETRY_S", 0.01, raising=False)
    with session() as sock:
        sock.send_bytes(b"\0" * (CHUNK * 2))
        err = read_until(sock, lambda m: m["type"] == "error")[-1]
        assert "识别出错" in err["message"]
        assert t.calls == server.TRANSCRIBE_ATTEMPTS
        # given up: the next status moves committed past it
        assert read_until(sock, lambda m: m["type"] == "backlog")[-1]["committed"] == CHUNK
