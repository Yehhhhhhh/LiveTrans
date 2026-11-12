"""Choosing the Whisper model and compute type from the page (设置 → 识别 → 模型), with fakes."""

import sys
import threading
import types

import pytest

import server
from asr import Transcriber
from test_ws_worker import FakeSegmenter, FakeTranscriber, read_until, session

DEFAULT = server.DEFAULT_CHOICE


class SwitchableTranscriber(FakeTranscriber):
    """Loads instantly; choices in `failing` fail and fall back like asr.Transcriber.load."""

    def __init__(self, failing=()):
        super().__init__("ready")
        self.model = object()
        self.choice = DEFAULT
        self.model_name, self.compute_type = DEFAULT
        self.failing = set(failing)
        self.loads = []
        self.gate = None  # set to a threading.Event to hold the next load until it is set
        self.seen_error_while_loading = False

    def _try(self, choice):
        self.loads.append(choice)
        self.model_name = choice[0]
        if choice in self.failing:
            return "boom"
        self.model, self.choice, self.compute_type, self.error = object(), choice, choice[1], ""
        return ""

    def load(self, model_name, device="auto", compute_type="auto", fallback=None):
        self.status = "loading"
        if self.gate is not None:
            gate, self.gate = self.gate, None
            assert gate.wait(5)
        error = self._try((model_name, compute_type))
        if error and fallback and fallback != (model_name, compute_type):
            error = self._try(fallback)
            if not error:
                self.error = f"{model_name}（{compute_type}）加载失败，已换回 {fallback[0]}（{fallback[1]}）。boom"
        if error:
            self.model, self.choice, self.error = None, None, "模型加载失败：boom"
        self.status = "error" if error else "ready"


@pytest.fixture
def fake(monkeypatch):
    t = SwitchableTranscriber()
    monkeypatch.setattr(server, "transcriber", t)
    monkeypatch.setattr(server, "Segmenter", FakeSegmenter)
    monkeypatch.setattr(server, "SileroModel", lambda: None)
    monkeypatch.setattr(server, "load_model_in_background", lambda: None)
    monkeypatch.setattr(server, "MODEL_WAIT_POLL_S", 0.01, raising=False)
    monkeypatch.setattr(server, "wanted_choice", [DEFAULT])
    monkeypatch.setattr(server, "failed_choice", [None])
    return t


def choose(sock, **asr):
    sock.send_json({"type": "config", "asr": {"partial": False, **asr}})


def settled(sock):
    """Messages up to the status that ends a load."""
    return read_until(sock, lambda m: m["type"] == "status" and m["model"] != "loading")


def say(sock):
    sock.send_bytes(b"\0" * 3200)
    return read_until(sock, lambda m: m["type"] == "segment")


def test_status_lists_what_the_page_may_choose(fake):
    msg = server.status_message()
    assert msg["default_model"] == server.MODEL
    assert {server.MODEL, "large-v3-turbo", "large-v3"} == set(msg["models"])
    assert msg["default_compute"] == server.COMPUTE_TYPE
    assert {server.COMPUTE_TYPE, "auto", "float16", "int8_float16", "int8"} == set(msg["compute_types"])


def test_choosing_another_model_loads_it_and_announces_it(fake):
    with session() as sock:
        choose(sock, model="large-v3")
        seen = settled(sock)
        assert [m["model"] for m in seen if m["type"] == "status"] == ["loading", "ready"]
        assert seen[-1]["model_name"] == "large-v3"
        assert fake.loads == [("large-v3", DEFAULT[1])]
        # Speech is transcribed again once the new model is in.
        assert say(sock)[-1]["text"] == "testo 1"


def test_choosing_a_compute_type_reloads_the_same_model(fake):
    with session() as sock:
        choose(sock, compute="int8_float16")
        ready = settled(sock)[-1]
        assert (ready["model_name"], ready["compute_type"]) == (DEFAULT[0], "int8_float16")
        # Model and compute type chosen together load once.
        choose(sock, model="large-v3", compute="int8")
        settled(sock)
        assert fake.loads == [(DEFAULT[0], "int8_float16"), ("large-v3", "int8")]


@pytest.mark.parametrize("asr", [
    {"model": ""}, {"model": DEFAULT[0]}, {"model": "../../somewhere"}, {"model": "tiny"},
    {"model": None}, {"model": 3}, {"compute": ""}, {"compute": "float64"}, {"compute": 8},
    {"model": "nope", "compute": "nope"},
])
def test_default_current_and_unknown_choices_load_nothing(fake, asr):
    with session() as sock:
        choose(sock, **asr)
        seen = say(sock)
        assert not [m for m in seen if m["type"] == "status"]
        assert fake.loads == []


def test_empty_choice_goes_back_to_the_default(fake):
    with session() as sock:
        choose(sock, model="large-v3", compute="int8")
        settled(sock)
        choose(sock, model="", compute="")
        ready = settled(sock)[-1]
        assert ready["model_name"] == DEFAULT[0]
        assert fake.loads == [("large-v3", "int8"), DEFAULT]


def test_a_choice_that_fails_to_load_falls_back_and_is_not_retried(fake):
    bad = ("large-v3", DEFAULT[1])
    fake.failing = {bad}
    with session() as sock:
        choose(sock, model="large-v3")
        seen = settled(sock)
        # Never "error" on the way: queued speech would be given up (see wait_for_model).
        assert [m["model"] for m in seen if m["type"] == "status"] == ["loading", "ready"]
        assert seen[-1]["model_name"] == DEFAULT[0]
        assert "large-v3" in seen[-1]["error"] and "已换回" in seen[-1]["error"]
        assert fake.loads == [bad, DEFAULT]
        # The page resends its settings (any change does): the failed choice isn't loaded again.
        choose(sock, model="large-v3")
        assert say(sock)[-1]["text"] == "testo 1"
        assert fake.loads == [bad, DEFAULT]
        # Another choice is.
        choose(sock, model="large-v3", compute="int8")
        assert settled(sock)[-1]["model_name"] == "large-v3"


def test_a_choice_made_during_a_failing_load_is_applied_afterwards(fake):
    bad = ("large-v3", DEFAULT[1])
    fake.failing = {bad}
    gate = fake.gate = threading.Event()
    with session() as sock:
        choose(sock, model="large-v3")
        read_until(sock, lambda m: m["type"] == "status" and m["model"] == "loading")
        choose(sock, compute="int8")  # while large-v3 is still loading
        sock.send_bytes(b"\0" * 3200)  # handled after the config above: the choice is recorded
        read_until(sock, lambda m: m["type"] == "backlog" and m["pending"] == 1)
        assert server.wanted_choice == [("large-v3", "int8")]
        assert fake.loads == []  # the first load is still held
        # Let it fail: it falls back to the default, then the newer choice is loaded.
        gate.set()
        seen = read_until(
            sock, lambda m: m["type"] == "status" and m["model"] == "ready" and m["compute_type"] == "int8"
        )
        assert seen[-1]["model_name"] == "large-v3"
        assert fake.loads == [bad, DEFAULT, ("large-v3", "int8")]
        assert not [m for m in seen if m["type"] == "error"]
        # The speech queued meanwhile was kept and is transcribed (on the fallback model, between
        # the two loads, or on the final one).
        if not [m for m in seen if m["type"] == "segment"]:
            seen = read_until(sock, lambda m: m["type"] == "segment")
        assert [m["text"] for m in seen if m["type"] == "segment"] == ["testo 1"]


def test_a_choice_made_while_a_model_is_loading_starts_no_second_load(fake):
    fake.status = "loading"
    with session() as sock:
        choose(sock, model="large-v3")
        sock.send_bytes(b"\0" * 3200)
        read_until(sock, lambda m: m["type"] == "backlog" and m["pending"] == 1)
        assert fake.loads == []
        # The choice is kept (server.ensure_model applies it when the running load ends).
        assert server.wanted_choice == [("large-v3", DEFAULT[1])]


def test_gpu_endpoint(fake, monkeypatch):
    from fastapi.testclient import TestClient

    with TestClient(server.app, base_url="http://127.0.0.1:8765") as client:
        monkeypatch.setattr(server, "gpu_memory", lambda: (3000, 8188))
        assert client.get("/api/gpu").json() == {"used_mb": 3000, "total_mb": 8188}
        monkeypatch.setattr(server, "gpu_memory", lambda: None)
        assert client.get("/api/gpu").json() == {}


# ---------------------------------------------------------------- asr.Transcriber.load


def fake_whisper(monkeypatch, failing, seen):
    class WhisperModel:
        def __init__(self, name, device, compute_type):
            seen.append((name, device, compute_type, tr.status))
            if name in failing:
                raise RuntimeError(f"cannot load {name}")

        def transcribe(self, audio, **kw):
            return iter(()), None

    tr = Transcriber()
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WhisperModel))
    return tr


def test_load_falls_back_without_ever_reporting_an_error(monkeypatch):
    seen = []
    tr = fake_whisper(monkeypatch, {"large-v3"}, seen)
    tr.load("large-v3", "cuda", "int8", fallback=("large-v3-turbo", "auto"))
    assert tr.status == "ready"
    assert tr.choice == ("large-v3-turbo", "auto")
    assert (tr.model_name, tr.device, tr.compute_type) == ("large-v3-turbo", "cuda", "float16")
    assert "large-v3（int8）加载失败，已换回 large-v3-turbo" in tr.error
    # Both attempts ran while the status was "loading".
    assert seen == [("large-v3", "cuda", "int8", "loading"), ("large-v3-turbo", "cuda", "float16", "loading")]


def test_load_reports_an_error_when_the_fallback_fails_too(monkeypatch):
    tr = fake_whisper(monkeypatch, {"large-v3", "large-v3-turbo"}, [])
    tr.load("large-v3", "cuda", "auto", fallback=("large-v3-turbo", "auto"))
    assert tr.status == "error" and tr.model is None and tr.choice is None
    assert tr.error.startswith("模型加载失败：")


def test_load_without_fallback(monkeypatch):
    tr = fake_whisper(monkeypatch, set(), [])
    tr.load("large-v3-turbo", "cpu", "auto")
    assert (tr.status, tr.choice, tr.compute_type, tr.error) == ("ready", ("large-v3-turbo", "auto"), "int8", "")
