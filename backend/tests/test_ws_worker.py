"""The /ws transcription worker, with the VAD and Whisper replaced by fakes."""

from contextlib import contextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from vad import Segment


class FakeSegmenter:
    """Every audio message is one finished 1 s segment."""

    def __init__(self, model, params=None):
        self.params = params
        self.index = 0
        self.speaking = False
        self.frames_seen = 0

    def reset(self):
        self.index = 0

    def feed(self, pcm):
        self.index += 1
        return [Segment(start=float(self.index), audio=np.zeros(16000, np.float32), index=self.index)]

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
        self.calls = 0

    def transcribe(self, audio, language="auto", prompt=""):
        self.calls += 1
        return {"lang": "it", "lang_prob": 1.0, "text": f"testo {self.calls}", "asr_ms": 1}


@pytest.fixture
def fake(monkeypatch):
    t = FakeTranscriber("loading")
    monkeypatch.setattr(server, "transcriber", t)
    monkeypatch.setattr(server, "Segmenter", FakeSegmenter)
    monkeypatch.setattr(server, "SileroModel", lambda: None)
    monkeypatch.setattr(server, "load_model_in_background", lambda: None)
    monkeypatch.setattr(server, "MODEL_WAIT_POLL_S", 0.01, raising=False)
    return t


@contextmanager
def session():
    """A started /ws session; always closed, so a failing assertion can't hang TestClient."""
    # Host as the browser sends it (TestClient's default "testserver" is refused by guard.py).
    with TestClient(server.app, base_url="http://127.0.0.1:8765") as client, client.websocket_connect("ws://127.0.0.1:8765/ws") as sock:
        assert sock.receive_json()["type"] == "status"
        sock.send_json({"type": "config", "asr": {"partial": False}})
        sock.send_json({"type": "start"})
        yield sock


def read_until(sock, pred, limit=200):
    """Messages up to and including the first one matching pred."""
    seen = []
    for _ in range(limit):
        m = sock.receive_json()
        seen.append(m)
        if pred(m):
            return seen
    raise AssertionError(f"no matching message in {seen}")


def test_speech_during_model_load_is_kept_and_transcribed_once_ready(fake):
    with session() as sock:
        for _ in range(3):
            sock.send_bytes(b"\0" * 3200)
        seen = read_until(sock, lambda m: m["type"] == "error" or (m["type"] == "backlog" and m["pending"] == 3))
        assert not [m for m in seen if m["type"] in ("segment", "error")]
        assert fake.calls == 0

        fake.status = "ready"
        seen = read_until(sock, lambda m: m["type"] == "segment")
        segment = seen[-1]
        # All three waited and are caught up in one merged pass.
        assert segment["merged"] == 3
        assert segment["index"] == 3
        assert read_until(sock, lambda m: m["type"] == "backlog")[-1]["pending"] == 0


def test_failed_model_reports_an_error_per_batch(fake):
    fake.status = "error"
    with session() as sock:
        sock.send_bytes(b"\0" * 3200)
        err = read_until(sock, lambda m: m["type"] == "error")[-1]
        assert "模型加载失败" in err["message"]


def test_worker_survives_an_unexpected_error(fake, monkeypatch):
    fake.status = "ready"
    real = server.take_batch
    calls = []

    def flaky(first, queue):
        calls.append(first.index)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return real(first, queue)

    monkeypatch.setattr(server, "take_batch", flaky)
    with session() as sock:
        sock.send_bytes(b"\0" * 3200)  # segment 1: the worker hits the error
        read_until(sock, lambda m: m["type"] == "backlog")
        sock.send_bytes(b"\0" * 3200)  # segment 2: must still be transcribed
        seg = read_until(sock, lambda m: m["type"] == "segment")[-1]
        assert seg["index"] == 2
        assert seg["text"] == "testo 1"
