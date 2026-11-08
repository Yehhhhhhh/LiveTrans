"""Changing the recognition language mid-segment re-detects the language for live partials."""

import time
from contextlib import contextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server


class SpeakingSegmenter:
    """One long segment in progress: always speaking, never finished."""

    def __init__(self, model, params=None):
        self.params = params
        self.index = 1
        self.speaking = True
        self.frames_seen = 0

    def reset(self):
        self.frames_seen = 0

    def feed(self, pcm):
        self.frames_seen += 100  # well past PARTIAL_INTERVAL_S
        return []

    def current_audio(self):
        return np.zeros(3 * 16000, np.float32)  # past PARTIAL_LANG_LOCK_S: detection is cached

    def flush(self):
        return []

    def take_dropped(self):
        return []

    def take_peak_prob(self):
        return 0.9

    def committed_frames(self):
        return 0


class FakeTranscriber:
    """Hears Chinese; restricted to Italian / English it picks Italian."""

    def __init__(self):
        self.status = "ready"
        self.error = ""
        self.model_name = "fake"
        self.device = "cpu"
        self.compute_type = "int8"
        self.detections = []

    def pick_language(self, audio, among=("it", "en")):
        self.detections.append(among)
        return ("it", 0.9) if among else ("zh", 0.9)

    def transcribe_partial(self, audio, lang, prompt=""):
        return f"[{lang}]"


@pytest.fixture
def fake(monkeypatch):
    t = FakeTranscriber()
    monkeypatch.setattr(server, "transcriber", t)
    monkeypatch.setattr(server, "Segmenter", SpeakingSegmenter)
    monkeypatch.setattr(server, "SileroModel", lambda: None)
    monkeypatch.setattr(server, "load_model_in_background", lambda: None)
    return t


@contextmanager
def session():
    # Host as the browser sends it, so the test also holds with the same-origin guard (#6).
    with TestClient(server.app, base_url="http://127.0.0.1:8765") as client, client.websocket_connect(
        "ws://127.0.0.1:8765/ws"
    ) as sock:
        assert sock.receive_json()["type"] == "status"
        yield sock


def next_partial(sock):
    time.sleep(0.1)  # let the previous partial task finish
    sock.send_bytes(b"\0" * 3200)
    for _ in range(50):
        m = sock.receive_json()
        if m["type"] == "partial":
            return m["text"]
    raise AssertionError("no partial")


def test_switching_language_mode_redetects_the_segment_in_progress(fake):
    with session() as sock:
        sock.send_json({"type": "config", "asr": {"language": "it_en", "partial": True}})
        sock.send_json({"type": "start"})
        assert next_partial(sock) == "[it]"
        assert next_partial(sock) == "[it]"  # cached for this segment
        assert len(fake.detections) == 1

        sock.send_json({"type": "config", "asr": {"language": "any", "partial": True}})
        assert next_partial(sock) == "[zh]"  # detected again, among all languages
        assert fake.detections[-1] is None


def test_resending_the_same_mode_keeps_the_cached_language(fake):
    with session() as sock:
        sock.send_json({"type": "config", "asr": {"language": "it_en", "partial": True}})
        sock.send_json({"type": "start"})
        assert next_partial(sock) == "[it]"
        # the page resends its config on every settings change (e.g. the glossary prompt)
        sock.send_json({"type": "config", "asr": {"language": "it_en", "partial": True}})
        assert next_partial(sock) == "[it]"
        assert len(fake.detections) == 1


def test_mode_change_during_an_in_flight_partial(fake):
    import threading

    started, release = threading.Event(), threading.Event()
    plain = fake.pick_language

    def slow_first_detection(audio, among=("it", "en")):
        if not started.is_set():
            started.set()
            release.wait(5)  # still detecting under it_en while the mode changes
        return plain(audio, among)

    fake.pick_language = slow_first_detection
    with session() as sock:
        sock.send_json({"type": "config", "asr": {"language": "it_en", "partial": True}})
        sock.send_json({"type": "start"})
        sock.send_bytes(b"\0" * 3200)
        assert started.wait(5)
        sock.send_json({"type": "config", "asr": {"language": "any", "partial": True}})
        time.sleep(0.1)  # the config is applied before the detection finishes
        release.set()
        # The stale "[it]" partial is dropped and doesn't re-fill the cache:
        # the next partial detects again among all languages.
        assert next_partial(sock) == "[zh]"
        assert fake.detections == [("it", "en"), None]
