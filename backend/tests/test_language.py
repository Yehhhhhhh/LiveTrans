from types import SimpleNamespace

import numpy as np
import pytest

from asr import ANY, IT_EN, Transcriber, all_languages, detect_among, is_fixed, normalize_language

AUDIO = np.zeros(16000, dtype=np.float32)


def test_modes_and_codes_are_accepted():
    assert normalize_language("any") == ANY
    assert normalize_language("it_en") == IT_EN
    assert normalize_language("it") == "it"
    assert normalize_language("fr") == "fr"
    assert normalize_language("zh") == "zh"


def test_old_auto_setting_keeps_meaning_italian_or_english():
    assert normalize_language("auto") == IT_EN


@pytest.mark.parametrize("bad", ["", "xx", "french", None, 3, "IT"])
def test_invalid_values_are_rejected(bad):
    assert normalize_language(bad) is None


def test_whisper_language_list():
    langs = all_languages()
    assert len(langs) >= 99
    assert {"it", "en", "fr", "de", "es", "zh", "ja"} <= set(langs)


def test_detection_scope():
    assert detect_among(IT_EN) == ("it", "en")
    assert detect_among(ANY) is None
    assert not is_fixed(ANY) and not is_fixed(IT_EN)
    assert is_fixed("it") and is_fixed("fr")


class FakeModel:
    """French is most likely overall; between Italian and English, Italian wins."""

    def __init__(self):
        self.calls = []

    def detect_language(self, audio):
        return "fr", 0.6, [("fr", 0.6), ("it", 0.25), ("en", 0.1), ("es", 0.05)]

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        seg = SimpleNamespace(text=" bonjour à tous les étudiants", compression_ratio=1.1, avg_logprob=-0.2, no_speech_prob=0.0)
        return iter([seg]), None


@pytest.fixture
def t():
    tr = Transcriber()
    tr.model = FakeModel()
    tr.status = "ready"
    return tr


def test_pick_language_any_takes_the_overall_best(t):
    assert t.pick_language(AUDIO, None) == ("fr", 0.6)


def test_pick_language_it_en_ignores_other_languages(t):
    lang, p = t.pick_language(AUDIO, ("it", "en"))
    assert lang == "it"
    assert p == pytest.approx(0.25 / 0.35)


@pytest.mark.parametrize(
    "setting, decoded_as",
    [("any", "fr"), ("it_en", "it"), ("auto", "it"), ("en", "en"), ("de", "de"), ("garbage", "it")],
)
def test_transcribe_uses_the_mode(t, setting, decoded_as):
    res = t.transcribe(AUDIO, setting)
    assert t.model.calls[-1]["language"] == decoded_as
    assert res["lang"] == decoded_as
