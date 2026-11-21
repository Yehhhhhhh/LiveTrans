"""The AlignAtt engine's torch-free parts, and how the server hands it a segment's live state."""

import numpy as np
import pytest

import server
from alignatt import AlignAttModel, StreamState
from test_ws_worker import FakeTranscriber, read_until, session
from vad import Segment


class FakeTok:
    """Token i decodes to WORDS[i]."""

    WORDS = [" The", " least", " squ", "ares", ",", " est"]

    def decode(self, tokens):
        return "".join(self.WORDS[t] for t in tokens)


def test_the_last_word_is_held_back():
    new, where = AlignAttModel.drop_last_word(FakeTok(), [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    assert (new, where) == ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
    # "squares," may still go on: only "The least" is certain.
    assert AlignAttModel.drop_last_word(FakeTok(), [0, 1, 2, 3, 4], [1, 2, 3, 4, 5])[0] == [0, 1]
    # A single word, or only the end of the word shown before: nothing new yet.
    assert AlignAttModel.drop_last_word(FakeTok(), [2, 3], [1, 2])[0] == []
    assert AlignAttModel.drop_last_word(FakeTok(), [4], [1])[0] == []


class Stats:
    def __init__(self, no_speech=0.0, logprob=-0.2, ratio=1.2):
        self.no_speech_prob, self.avg_logprob, self.compression_ratio = no_speech, logprob, ratio


class TextTok:
    def __init__(self, words):
        self.words = words

    def decode(self, tokens):
        return "".join(self.words[t] for t in tokens)


def test_unsure_or_hallucinated_words_are_not_shown():
    tok = TextTok([" Good", " morning", " everyone", " Grazie", " a", " tutti", " ruota,"])
    assert AlignAttModel._acceptable(tok, [], [0, 1], Stats(), "")
    assert not AlignAttModel._acceptable(tok, [], [0, 1], Stats(logprob=-1.5), "")
    assert not AlignAttModel._acceptable(tok, [], [0, 1], Stats(no_speech=0.8), "")
    # Speech already shown: a high no-speech score at the start doesn't hold the next words back.
    assert AlignAttModel._acceptable(tok, [0], [1, 2], Stats(no_speech=0.8), "")
    assert not AlignAttModel._acceptable(tok, [], [3, 4, 5], Stats(), "")
    assert not AlignAttModel._acceptable(tok, [], [6, 6, 6], Stats(), "")


class SpeechThenEnd:
    """First audio message: speech in progress (a partial is due). Second: the segment ends."""

    def __init__(self, model, params=None):
        self.params = params
        self.index = 1
        self.speaking = False
        self.frames_seen = 0
        self.calls = 0

    def reset(self):
        pass

    def feed(self, pcm):
        self.calls += 1
        if self.calls == 1:
            self.speaking, self.frames_seen = True, 100
            return []
        self.speaking = False
        return [Segment(start=0.0, audio=np.zeros(32000, np.float32), index=1)]

    def flush(self):
        return []

    def take_dropped(self):
        return []

    def take_peak_prob(self):
        return 0.0

    def current_audio(self):
        return np.zeros(16000, np.float32) if self.speaking else None

    def committed_frames(self):
        return 0


class FakeAlignAtt(FakeTranscriber):
    engine = object()  # what server.alignatt_on looks for

    def __init__(self):
        super().__init__("ready")
        self.partial_states = []
        self.final_states = []

    def transcribe_partial(self, audio, lang, prompt="", state=None):
        self.partial_states.append(state)
        state.lang, state.tokens = lang, [1, 2]
        return "Good morning"

    def transcribe(self, audio, language="auto", prompt="", context="", context_lang="", state=None):
        self.final_states.append(state)
        return {"lang": "en", "lang_prob": 1.0, "text": "Good morning everyone.", "asr_ms": 1}


@pytest.fixture
def engine(monkeypatch):
    t = FakeAlignAtt()
    monkeypatch.setattr(server, "transcriber", t)
    monkeypatch.setattr(server, "Segmenter", SpeechThenEnd)
    monkeypatch.setattr(server, "SileroModel", lambda: None)
    monkeypatch.setattr(server, "load_model_in_background", lambda: None)
    return t


def test_the_final_pass_gets_the_state_its_partials_built(engine):
    with session() as sock:
        sock.send_json({"type": "config", "asr": {"partial": True, "language": "en"}})
        sock.send_bytes(b"\0" * 3200)
        partial = read_until(sock, lambda m: m["type"] == "partial")[-1]
        assert partial["text"] == "Good morning"
        sock.send_bytes(b"\0" * 3200)
        assert read_until(sock, lambda m: m["type"] == "segment")[-1]["text"] == "Good morning everyone."
    [state] = engine.partial_states
    assert isinstance(state, StreamState)
    assert engine.final_states == [state]
