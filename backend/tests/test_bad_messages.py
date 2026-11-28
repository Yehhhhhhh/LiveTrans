"""Text messages on /ws that can't be used are logged and ignored; the connection stays open.

Ending it would not help: the page reconnects at once and sends the same settings again."""

import json
import logging

import pytest

import server
from server import BadMessage, read_message
from test_ws_worker import FakeSegmenter, FakeTranscriber, read_until, session

# ---------------------------------------------------------------- read_message

# A config message as app.js's asrConfig() builds it.
PAGE_CONFIG = {
    "type": "config",
    "asr": {
        "engine": "assemblyai", "assemblyai_key": "key", "assemblyai_model": "",
        "language": "it_en", "model": "", "compute": "",
        "vad_threshold": 0.5, "min_silence_ms": 600, "max_segment_s": 8,
        "partial": True, "context": True,
    },
    "prompt": "Corso. Fourier, Laplace.",
}


@pytest.mark.parametrize("msg", [
    PAGE_CONFIG,
    {"type": "config"},
    {"type": "config", "asr": None, "prompt": None},  # unset
    {"type": "config", "asr": {"vad_threshold": "0.4", "min_silence_ms": True, "max_segment_s": 1e999}},
    {"type": "config", "asr": {"model": 3, "language": [], "engine": {}, "assemblyai_key": 5}},  # fall back
    {"type": "start", "run": 3, "engine": "local"},
    {"type": "start", "offset": "only resume reads it"},
    {"type": "resume", "run": 3, "offset": 16000, "engine": "assemblyai", "assemblyai_key": "key"},
    {"type": "resume", "offset": "16000"},
    {"type": "resume", "offset": None},
    {"type": "prepare", "engine": "assemblyai", "assemblyai_key": None, "assemblyai_model": 7},
    {"type": "unprepare"},
    {"type": "stop"},
    {"type": "from a newer page"},
    {"type": ["config"]},
    {},
])
def test_messages_that_can_be_handled_pass_unchanged(msg):
    assert read_message(json.dumps(msg)) == msg


@pytest.mark.parametrize("text, reason", [
    ("not json", "not JSON"),
    ('{"type": "config"', "not JSON"),
    ("[" * 100_000, "not JSON"),  # nested too deep for the parser
    ("[1, 2]", "not a JSON object"),
    ('"config"', "not a JSON object"),
    ("null", "not a JSON object"),
    (json.dumps({"type": "config", "asr": ["partial"]}), "asr"),
    (json.dumps({"type": "config", "asr": "partial"}), "asr"),
    (json.dumps({"type": "config", "asr": {"vad_threshold": None}}), "vad_threshold"),  # JSON.stringify(NaN)
    (json.dumps({"type": "config", "asr": {"min_silence_ms": "long"}}), "min_silence_ms"),
    (json.dumps({"type": "config", "asr": {"max_segment_s": [8]}}), "max_segment_s"),
    ('{"type": "config", "asr": {"vad_threshold": 1' + "0" * 400 + "}}", "vad_threshold"),  # too big for float()
    (json.dumps({"type": "config", "prompt": ["Fourier"]}), "prompt"),
    (json.dumps({"type": "config", "prompt": 3}), "prompt"),
    (json.dumps({"type": "resume", "offset": "soon"}), "offset"),
    (json.dumps({"type": "resume", "offset": "1.5"}), "offset"),
    (json.dumps({"type": "resume", "offset": [0]}), "offset"),
    ('{"type": "resume", "offset": NaN}', "offset"),
    ('{"type": "resume", "offset": Infinity}', "offset"),
])
def test_messages_that_would_fail_are_refused(text, reason):
    with pytest.raises(BadMessage, match=reason):
        read_message(text)


# ---------------------------------------------------------------- through /ws


class KeptSegmenter(FakeSegmenter):
    """FakeSegmenter whose instances the test can look at."""

    made: list = []

    def __init__(self, model, params=None):
        super().__init__(model, params)
        KeptSegmenter.made.append(self)


class Recording(FakeTranscriber):
    def __init__(self):
        super().__init__("ready")
        self.args = []

    def transcribe(self, audio, language="auto", prompt="", context="", context_lang=""):
        self.args.append((language, prompt))
        return super().transcribe(audio, language, prompt)


@pytest.fixture
def fake(monkeypatch):
    t = Recording()
    monkeypatch.setattr(server, "transcriber", t)
    monkeypatch.setattr(server, "Segmenter", KeptSegmenter)
    monkeypatch.setattr(server, "SileroModel", lambda: None)
    monkeypatch.setattr(server, "load_model_in_background", lambda: None)
    monkeypatch.setattr(KeptSegmenter, "made", [])
    return t


def say(sock):
    sock.send_bytes(b"\0" * 3200)
    return read_until(sock, lambda m: m["type"] == "segment")[-1]


def warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "livetrans" and r.levelno == logging.WARNING]


def test_bad_messages_are_logged_and_the_connection_keeps_working(fake, caplog):
    bad = [
        "not json",
        "[]",
        json.dumps({"type": "config", "asr": {"vad_threshold": None}}),
        json.dumps({"type": "resume", "offset": "soon"}),
    ]
    with session() as sock:
        for text in bad:
            sock.send_text(text)
        assert say(sock)["text"] == "testo 1"
    assert len(warnings(caplog)) == len(bad)


def test_a_bad_config_is_not_applied_in_part(fake):
    with session() as sock:
        # The good values come before the bad one: none of them is applied either.
        sock.send_json({
            "type": "config",
            "asr": {"language": "en", "min_silence_ms": 300, "context": False, "max_segment_s": "long"},
            "prompt": "Fourier.",
        })
        say(sock)
        params = KeptSegmenter.made[-1].params
        assert (params.min_silence_ms, params.max_segment_s) == (600, 8.0)  # the defaults
        sock.send_json({"type": "config", "asr": {"language": "en", "min_silence_ms": 300}, "prompt": "Fourier."})
        say(sock)
        assert params.min_silence_ms == 300
    assert fake.args == [("it_en", ""), ("en", "Fourier.")]


def test_a_bad_resume_leaves_the_run_as_it_was(fake):
    with session() as sock:  # run 1
        sock.send_json({"type": "resume", "run": 5, "offset": "soon"})
        seg = say(sock)
        assert (seg["run"], seg["start"]) == (1, 1.0)


def test_the_log_does_not_show_what_the_message_said(fake, caplog):
    with session() as sock:
        sock.send_json({"type": "config", "asr": {"assemblyai_key": "secret-key", "vad_threshold": None}})
        sock.send_text('{"assemblyai_key": "secret-key", ')
        say(sock)
    [first, second] = warnings(caplog)
    assert "vad_threshold" in first and "not JSON" in second
    assert not [w for w in (first, second) if "secret-key" in w]


def test_an_unknown_message_type_is_still_ignored_without_a_warning(fake, caplog):
    with session() as sock:
        sock.send_json({"type": "from a newer page", "asr": "anything"})
        sock.send_json({"type": ["config"]})
        say(sock)
    assert warnings(caplog) == []
