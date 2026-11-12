"""The previous sentence as Whisper context: prompt fitting, the echo filter, and the /ws state."""

from types import SimpleNamespace

import numpy as np
import pytest

import server
from asr import PROMPT_TOKENS, Transcriber, echoes_context, fit_prompt
from test_ws_worker import FakeSegmenter, FakeTranscriber, read_until, session
from vad import Segment

AUDIO = np.zeros(16000, dtype=np.float32)


def words(text):
    """Token count stand-in: one token per word."""
    return len(text.split())


def test_prompt_alone_is_unchanged():
    prompt = "Lezione di reti / Networks lecture. Ethernet, Wi-Fi, packet switching."
    assert fit_prompt(words, prompt) == prompt
    assert fit_prompt(words, "") == ""
    assert fit_prompt(words, "Lezione di reti. ") == "Lezione di reti. "


def test_context_goes_last():
    out = fit_prompt(words, "Lezione di reti. Ethernet, Wi-Fi.", "  e quindi il pacchetto  ")
    assert out == "Lezione di reti. Ethernet, Wi-Fi. e quindi il pacchetto"
    assert fit_prompt(words, "", "e quindi il pacchetto") == "e quindi il pacchetto"
    assert fit_prompt(words, "Ethernet, Wi-Fi.", "e quindi") == "Ethernet, Wi-Fi. e quindi"


def test_too_long_drops_the_oldest_terms_and_keeps_the_hint_and_the_context():
    terms = [f"term{i}" for i in range(400)]
    out = fit_prompt(words, "Lezione di reti. " + ", ".join(terms) + ".", "e quindi il pacchetto arriva")
    assert out.startswith("Lezione di reti. term0, term1,")
    assert out.endswith(". e quindi il pacchetto arriva")
    assert "term399" not in out
    assert words(out) <= PROMPT_TOKENS


def test_a_huge_context_loses_its_start_not_the_hint():
    context = " ".join(f"w{i}" for i in range(1000))
    out = fit_prompt(words, "Lezione di reti. Ethernet.", context)
    assert out.startswith("Lezione di reti.")
    assert out.endswith("w999")
    assert "w0 " not in out
    assert words(out) <= PROMPT_TOKENS


def test_echo_of_the_context_is_spotted():
    context = "come abbiamo visto, il segnale viene trasmesso sul canale."
    assert echoes_context("Il segnale viene trasmesso sul canale.", context)
    assert not echoes_context("Sul canale.", context)  # short: could well be said again
    assert not echoes_context("Il segnale viene ricevuto sul canale.", context)
    assert not echoes_context("Il segnale viene trasmesso sul canale.", "")


def test_text_without_spaces_is_handled_by_characters():
    zh = "今天我们继续讲信道容量和香农定理的证明过程"
    # Trimming: one character per token here, so a long context loses its start, not everything.
    out = fit_prompt(len, "课程. ", zh * 20)
    assert out.startswith("课程.") and out.endswith("证明过程")
    assert len(out) <= PROMPT_TOKENS + 1
    # Echo: the whole clause is a single "word".
    assert echoes_context("香农定理的证明过程。", zh)
    assert not echoes_context("证明过程", zh)  # short
    assert not echoes_context("香农定理的推导过程。", zh)
    # The tail kept between segments is not emptied.
    tail = server.context_tail(zh * 20)
    assert len(tail) == server.CONTEXT_CHARS and tail.endswith("证明过程")


class FakeModel:
    def __init__(self, text=" il pacchetto arriva al router di destinazione"):
        self.text = text
        self.calls = []

    def detect_language(self, audio):
        return "it", 0.9, [("it", 0.9), ("en", 0.1)]

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        seg = SimpleNamespace(text=self.text, compression_ratio=1.1, avg_logprob=-0.2, no_speech_prob=0.0)
        return iter([seg]), None


@pytest.fixture
def t():
    tr = Transcriber()
    tr.model = FakeModel()
    tr.status = "ready"
    tr.count_tokens = words
    return tr


def test_context_in_the_same_language_reaches_whisper(t):
    res = t.transcribe(AUDIO, "it_en", "Lezione di reti. Ethernet.", "e quindi", "it")
    assert t.model.calls[0]["initial_prompt"] == "Lezione di reti. Ethernet. e quindi"
    assert res["text"] == "il pacchetto arriva al router di destinazione"


def test_context_in_another_language_is_left_out(t):
    t.transcribe(AUDIO, "it_en", "Lezione di reti. Ethernet.", "and so the packet", "en")
    assert t.model.calls[0]["initial_prompt"] == "Lezione di reti. Ethernet."


def test_no_prompt_and_no_context_sends_none(t):
    t.transcribe(AUDIO, "it")
    assert t.model.calls[0]["initial_prompt"] is None


def test_a_recited_context_is_dropped(t):
    t.model.text = " il pacchetto arriva al router"
    res = t.transcribe(AUDIO, "it", "", "e quindi il pacchetto arriva al router", "it")
    assert res["text"] == ""


# ---------------------------------------------------------------- /ws


class Recording(FakeTranscriber):
    def __init__(self):
        super().__init__("ready")
        self.contexts = []

    def transcribe(self, audio, language="auto", prompt="", context="", context_lang=""):
        self.contexts.append((context, context_lang))
        return super().transcribe(audio, language, prompt)


class SpacedSegmenter(FakeSegmenter):
    """Like FakeSegmenter, with segments GAP seconds apart."""

    GAP = 1.0

    def feed(self, pcm):
        self.index += 1
        return [Segment(start=self.index * self.GAP, audio=np.zeros(16000, np.float32), index=self.index)]


@pytest.fixture
def fake(monkeypatch):
    t = Recording()
    monkeypatch.setattr(server, "transcriber", t)
    monkeypatch.setattr(server, "Segmenter", SpacedSegmenter)
    monkeypatch.setattr(server, "SileroModel", lambda: None)
    monkeypatch.setattr(server, "load_model_in_background", lambda: None)
    monkeypatch.setattr(SpacedSegmenter, "GAP", 1.0)
    return t


def say(sock):
    sock.send_bytes(b"\0" * 3200)
    return read_until(sock, lambda m: m["type"] == "segment")[-1]


def test_each_segment_gets_what_was_said_before_it(fake):
    with session() as sock:
        for _ in range(3):
            say(sock)
    assert fake.contexts == [("", ""), ("testo 1", "it"), ("testo 1 testo 2", "it")]


def test_start_and_a_language_change_clear_the_context(fake):
    with session() as sock:
        say(sock)
        sock.send_json({"type": "start", "run": 7})
        say(sock)
        sock.send_json({"type": "config", "asr": {"language": "en"}})
        say(sock)
    assert fake.contexts == [("", ""), ("", ""), ("", "")]


def test_context_can_be_turned_off(fake):
    with session() as sock:
        sock.send_json({"type": "config", "asr": {"context": False}})
        say(sock)
        say(sock)
    assert fake.contexts == [("", ""), ("", "")]


def test_no_context_after_a_long_pause(fake, monkeypatch):
    monkeypatch.setattr(SpacedSegmenter, "GAP", server.CONTEXT_MAX_GAP_S + 5)
    with session() as sock:
        say(sock)
        say(sock)
    assert fake.contexts == [("", ""), ("", "")]


def test_context_tail_starts_at_a_word():
    text = "parola " * 100
    tail = server.context_tail(text)
    assert len(tail) <= server.CONTEXT_CHARS
    assert tail.startswith("parola") and tail.endswith("parola")
    assert server.context_tail("  due   parole ") == "due parole"
