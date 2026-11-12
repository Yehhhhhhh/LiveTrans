"""Choosing the Whisper model from the page (设置 → 识别 → 模型), with the model replaced by a fake."""

import pytest

import server
from test_ws_worker import FakeSegmenter, FakeTranscriber, read_until, session


class SwitchableTranscriber(FakeTranscriber):
    def __init__(self, failing=()):
        super().__init__("ready")
        self.model = object()
        self.model_name = server.MODEL
        self.failing = set(failing)
        self.loads = []

    def load(self, model_name, device="auto", compute_type="auto"):
        self.loads.append(model_name)
        self.model_name = model_name
        if model_name in self.failing:
            self.model, self.status, self.error = None, "error", "模型加载失败：boom"
        else:
            self.model, self.status, self.error = object(), "ready", ""


@pytest.fixture
def fake(monkeypatch):
    t = SwitchableTranscriber()
    monkeypatch.setattr(server, "transcriber", t)
    monkeypatch.setattr(server, "Segmenter", FakeSegmenter)
    monkeypatch.setattr(server, "SileroModel", lambda: None)
    monkeypatch.setattr(server, "load_model_in_background", lambda: None)
    monkeypatch.setattr(server, "MODEL_WAIT_POLL_S", 0.01, raising=False)
    monkeypatch.setattr(server, "wanted_model", [server.MODEL])
    return t


def choose(sock, model):
    sock.send_json({"type": "config", "asr": {"partial": False, "model": model}})


def test_status_lists_the_models_the_page_may_choose(fake):
    msg = server.status_message()
    assert msg["default_model"] == server.MODEL
    assert {server.MODEL, "large-v3-turbo", "large-v3"} == set(msg["models"])


def test_choosing_another_model_loads_it_and_announces_it(fake):
    with session() as sock:
        choose(sock, "large-v3")
        seen = read_until(sock, lambda m: m["type"] == "status" and m["model"] == "ready")
        assert [m["model"] for m in seen if m["type"] == "status"] == ["loading", "ready"]
        assert seen[-1]["model_name"] == "large-v3"
        assert fake.loads == ["large-v3"]
        # Speech is transcribed again once the new model is in.
        sock.send_bytes(b"\0" * 3200)
        assert read_until(sock, lambda m: m["type"] == "segment")[-1]["text"] == "testo 1"


@pytest.mark.parametrize("model", ["", server.MODEL, "../../somewhere", "tiny", None, 3])
def test_default_current_and_unknown_models_load_nothing(fake, model):
    with session() as sock:
        choose(sock, model)
        sock.send_bytes(b"\0" * 3200)
        seen = read_until(sock, lambda m: m["type"] == "segment")
        assert not [m for m in seen if m["type"] == "status"]
        assert fake.loads == []


def test_empty_choice_goes_back_to_the_default_model(fake):
    with session() as sock:
        choose(sock, "large-v3")
        read_until(sock, lambda m: m["type"] == "status" and m["model"] == "ready")
        choose(sock, "")
        seen = read_until(sock, lambda m: m["type"] == "status" and m["model"] == "ready")
        assert seen[-1]["model_name"] == server.MODEL
        assert fake.loads == ["large-v3", server.MODEL]


def test_a_model_that_fails_to_load_falls_back_to_the_previous_one(fake):
    fake.failing = {"large-v3"}
    with session() as sock:
        choose(sock, "large-v3")
        ready = read_until(sock, lambda m: m["type"] == "status" and m["model"] != "loading")[-1]
        assert ready["model"] == "ready"
        assert ready["model_name"] == server.MODEL
        assert "large-v3 加载失败" in ready["error"]
        assert fake.loads == ["large-v3", server.MODEL]
        sock.send_bytes(b"\0" * 3200)
        assert read_until(sock, lambda m: m["type"] == "segment")[-1]["text"] == "testo 1"


def test_a_choice_made_while_a_model_is_loading_starts_no_second_load(fake):
    fake.status = "loading"
    with session() as sock:
        choose(sock, "large-v3")
        sock.send_bytes(b"\0" * 3200)
        read_until(sock, lambda m: m["type"] == "backlog" and m["pending"] == 1)
        assert fake.loads == []
        # The choice is kept (server.ensure_model applies it when the running load ends).
        assert server.wanted_model == ["large-v3"]
