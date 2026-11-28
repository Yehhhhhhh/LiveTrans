"""tools/replay.py, the offline replay of a recording, with a scripted VAD and a fake model (no GPU);
and the reasons asr's hallucination checks give for what they throw away."""

import json
import random
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from asr import Transcriber, clean_text, clean_text_why, unreliable, unreliable_why
from test_resume import ScriptedVad
from vad import FRAME, SR, Segmenter, SegmenterParams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import replay  # noqa: E402

SPEECH = [0.9] * 125 + [0.0] * 40  # 4 s of speech (125 frames of 32 ms), then silence
SPEECH_END = 131 * FRAME / SR  # the segment keeps 6 silent frames (keep_tail_ms 200): 4.192 s


class Clock:
    """Stands in for time.perf_counter: only the fake model moves it, by what its calls cost."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeModel:
    """A Transcriber whose calls cost partial_cost / final_cost seconds on the clock. Partials say
    the next of partial_texts (the last one again when they run out), finals the next of
    final_texts; each final reports `dropped` as thrown away."""

    def __init__(self, clock, partial_cost=0.1, final_cost=0.3, partial_texts=("Una",),
                 final_texts=("Una frase.",), dropped=()):
        self.clock = clock
        self.partial_cost, self.final_cost = partial_cost, final_cost
        self.partial_texts, self.final_texts = partial_texts, final_texts
        self.dropped = list(dropped)
        self.detections = 0
        self.partials = []  # (samples, language) of each call
        self.finals = []  # the arguments of each call

    def pick_language(self, audio, among=None):
        self.detections += 1
        return "it", 0.9

    def transcribe_partial(self, audio, lang, prompt=""):
        self.clock.t += self.partial_cost
        self.partials.append((len(audio), lang))
        return self.partial_texts[min(len(self.partials), len(self.partial_texts)) - 1]

    def transcribe(self, audio, language="it_en", prompt="", context="", context_lang="", *, dropped=None):
        self.clock.t += self.final_cost
        self.finals.append({"samples": len(audio), "prompt": prompt, "context": context, "context_lang": context_lang})
        dropped.extend(self.dropped)
        text = self.final_texts[min(len(self.finals), len(self.final_texts)) - 1]
        return {"lang": "it", "lang_prob": 1.0, "text": text, "asr_ms": 1}


def run(probs, model, clock, language="it", **kw):
    rp = replay.Replay(model, Segmenter(ScriptedVad(probs), SegmenterParams()), language, timer=clock, **kw)
    rp.run(np.zeros(len(probs) * FRAME, dtype=np.int16))
    return rp


# ---------------------------------------------------------------------------------- the replay loop


def test_partials_follow_the_server_pace():
    clock = Clock()
    rp = run(SPEECH, FakeModel(clock, partial_cost=0.1, final_cost=0.3), clock)
    # The first partial needs 0.8 s of audio (PARTIAL_MIN_AUDIO_S), each next one 19 frames more
    # (PARTIAL_INTERVAL_S); audio arrives in 100 ms chunks.
    assert [p.start for p in rp.partials] == pytest.approx([0.8, 1.5, 2.1, 2.7, 3.3, 4.0])
    assert all(p.cost == pytest.approx(0.1) and p.shown for p in rp.partials)
    [final] = rp.finals
    # The VAD closes the segment on the 19th silent frame (min_silence_ms 600), in the chunk that
    # ends at 4.7 s; the GPU is free, so its transcription begins at once.
    assert final.indices == [1]
    assert (final.start, final.end, final.closed) == pytest.approx((0.0, SPEECH_END, 4.7))
    assert (final.began, final.done) == pytest.approx((4.7, 5.0))
    assert final.latency == pytest.approx(5.0 - SPEECH_END)


def test_no_partials_when_switched_off():
    clock = Clock()
    model = FakeModel(clock)
    rp = run(SPEECH, model, clock, partial=False)
    assert rp.partials == [] and model.partials == []
    assert [f.text for f in rp.finals] == ["Una frase."]


def test_a_final_waits_for_the_partial_holding_the_gpu():
    clock = Clock()
    # Partials take 1 s: each next one starts when the GPU is free again. The one from 3.8 s still
    # runs when the segment closes at 4.7 s, so its transcription begins at 4.8 s.
    model = FakeModel(clock, partial_cost=1.0, final_cost=0.3)
    rp = run(SPEECH, model, clock)
    assert [p.start for p in rp.partials] == pytest.approx([0.8, 1.8, 2.8, 3.8])
    [final] = rp.finals
    assert (final.closed, final.began, final.done) == pytest.approx((4.7, 4.8, 5.1))
    assert final.began - final.closed == pytest.approx(0.1)  # what the summary calls queue_wait


def test_a_slow_gpu_merges_queued_segments_and_finals_go_first():
    clock = Clock()
    probs = ([0.9] * 40 + [0.0] * 25) * 10  # ten 1.3 s sentences, a new one every 2.1 s
    model = FakeModel(clock, partial_cost=0.2, final_cost=4.0)
    rp = run(probs, model, clock)
    answered = [i for f in rp.finals for i in f.indices]
    assert answered == list(range(1, 11))  # each segment once, in order
    assert any(len(f.indices) > 1 for f in rp.finals)  # it fell behind and caught up (take_batch)
    # One thing at a time on the GPU, and no partial while a closed segment waits for its final.
    jobs = sorted([(p.start, p.done) for p in rp.partials] + [(f.began, f.done) for f in rp.finals])
    assert all(a_done <= b_start + 1e-9 for (_, a_done), (b_start, _) in zip(jobs, jobs[1:]))
    assert not [p for p in rp.partials for f in rp.finals if f.closed <= p.start < f.done]
    s = replay.summarize(rp, len(probs) * FRAME / SR)
    assert s["max_backlog"] >= 3 and s["merged_batches"] >= 1


def test_rewrites_count_the_shown_words_that_change():
    clock = Clock()
    texts = ("Oggi", "Oggi parliamo", "Oggi parlano di", "Oggi parlano di economia")
    model = FakeModel(clock, partial_texts=texts, final_texts=("Oggi parliamo di economia.",))
    rp = run(SPEECH, model, clock)
    s = replay.summarize(rp, len(SPEECH) * FRAME / SR)
    assert [p.text for p in rp.partials if p.shown] == [*texts, texts[-1], texts[-1]]
    assert (s["rewrites"], s["rewritten_words"]) == (1, 1)  # "parliamo" became "parlano"
    # The final text changes the live line ("parlano" back to "parliamo").
    assert (s["final_changed_live"], s["segments_with_live"]) == (1, 1)
    assert s["first_text"]["mean"] == pytest.approx(0.9)  # the first partial, ready at 0.9 s
    assert s["partial_cost"]["mean"] == pytest.approx(0.1)


def test_rewritten_ignores_case_and_punctuation():
    assert replay.rewritten("Oggi, parliamo", "oggi parliamo di economia") == 0
    assert replay.rewritten("a b c", "a x") == 2
    assert replay.rewritten("a b", "a") == 1  # a shown word taken back counts too


def test_a_partial_ready_after_its_segment_was_dropped_is_not_shown():
    clock = Clock()
    # Too little speech to keep (min_speech_ms 400): 10 frames, in a segment dropped at 1.6 s.
    probs = [0.9] * 10 + [0.4] * 20 + [0.0] * 30
    rp = run(probs, FakeModel(clock, partial_cost=0.5, partial_texts=("Ehm",)), clock)
    assert rp.finals == [] and list(rp.gone) == [1]
    # Ready at 1.3 s: shown (the page removes the line at 1.6 s). Ready at 2.0 s: never shown.
    assert [p.done for p in rp.partials] == pytest.approx([1.3, 2.0])
    assert [p.shown for p in rp.partials] == [True, False]


def test_an_empty_partial_is_not_shown():
    clock = Clock()
    rp = run(SPEECH, FakeModel(clock, partial_texts=("",)), clock)
    assert rp.partials and not any(p.shown for p in rp.partials)


def test_filtered_text_is_reported_with_its_reason(capsys):
    clock = Clock()
    model = FakeModel(clock, final_texts=("",), dropped=[("Thanks for watching the course", "phrase: thanks for watching")])
    rp = run(SPEECH, model, clock)
    s = replay.summarize(rp, len(SPEECH) * FRAME / SR)
    assert (s["filtered"], s["filtered_segments"]) == (1, 1)
    assert s["filtered_by_reason"] == {"phrase: thanks for watching": 1}
    replay.print_final(rp.finals[0])
    out = capsys.readouterr().out
    assert "已过滤：幻觉短语「thanks for watching」" in out and "Thanks for watching the course" in out


def test_the_previous_text_is_the_context_of_the_next_segment():
    clock = Clock()
    probs = SPEECH + [0.9] * 40 + [0.0] * 40
    model = FakeModel(clock, final_texts=("Prima frase.", "Seconda frase."))
    run(probs, model, clock, prompt="Lezione di economia. PIL")
    assert [(f["context"], f["context_lang"]) for f in model.finals] == [("", ""), ("Prima frase.", "it")]
    assert {f["prompt"] for f in model.finals} == {"Lezione di economia. PIL"}

    model = FakeModel(clock, final_texts=("Prima frase.", "Seconda frase."))
    run(probs, model, clock, context=False)
    assert [f["context"] for f in model.finals] == ["", ""]


def test_the_language_of_partials_is_detected_until_enough_audio():
    clock = Clock()
    model = FakeModel(clock)
    run(SPEECH, model, clock, language="it_en")
    # Detected for the partials on 0.8 s, 1.47 s and 2.08 s of audio; from 2 s on it is kept.
    assert model.detections == 3
    assert {lang for _, lang in model.partials} == {"it"}
    model = FakeModel(clock)
    run(SPEECH, model, clock, language="it")
    assert model.detections == 0


class FakeAlignAtt(FakeModel):
    engine = object()  # what server.alignatt_on (and the replay) looks for

    def __init__(self, clock):
        super().__init__(clock)
        self.partial_states, self.final_states = [], []

    def transcribe_partial(self, audio, lang, prompt="", state=None):
        self.partial_states.append(state)
        return super().transcribe_partial(audio, lang, prompt)

    def transcribe(self, audio, language="it_en", prompt="", context="", context_lang="", state=None, *, dropped=None):
        self.final_states.append(state)
        return super().transcribe(audio, language, prompt, context, context_lang, dropped=dropped)


def test_alignatt_gets_the_state_its_partials_built():
    clock = Clock()
    model = FakeAlignAtt(clock)
    run(SPEECH, model, clock)
    state = model.partial_states[0]
    assert isinstance(state, replay.StreamState)
    assert all(s is state for s in model.partial_states)
    assert model.final_states == [state]


# ---------------------------------------------------------------------------------- scoring


def levenshtein(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def test_edit_ops_is_a_shortest_edit():
    rng = random.Random(0)
    for _ in range(400):
        ref = [rng.choice("abc") for _ in range(rng.randint(0, 12))]
        hyp = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]
        s, d, i = replay.edit_ops(ref, hyp)
        assert s + d + i == levenshtein(ref, hyp)
        assert len(hyp) - len(ref) == i - d  # each ref word is kept, replaced or deleted
        assert min(s, d, i) >= 0


def test_wer_counts_each_kind_of_error():
    sc = replay.score("Il gatto è sul tavolo.", "il gatto e sul tavolo rosso")
    assert (sc["sub"], sc["del"], sc["ins"]) == (1, 0, 1)  # è -> e, rosso added
    assert sc["wer"] == pytest.approx(2 / 5)
    assert (sc["ref_words"], sc["hyp_words"]) == (5, 6)
    sc = replay.score("uno due tre", "uno tre")
    assert (sc["sub"], sc["del"], sc["ins"]) == (0, 1, 0)
    assert sc["cer"] == pytest.approx(4 / 11)  # "uno due tre" -> "uno tre": 4 characters gone
    assert replay.score("Oggi, parliamo!", "oggi parliamo")["wer"] == 0
    assert replay.score("...", "qualcosa") is None  # nothing to score against


def test_words_ignore_case_punctuation_and_apostrophes():
    assert replay.words("L'Università, oggi… “Economia”!") == ["l", "università", "oggi", "economia"]
    assert replay.words("è") == replay.words("è")  # è written two ways


# ---------------------------------------------------------------------------------- command line


def write_wav(path, n_samples, rate=SR, channels=1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(n_samples * channels, dtype="<i2").tobytes())


def test_the_command_line_replays_a_wav(tmp_path, monkeypatch, capsys):
    wav, ref, out = tmp_path / "rec.wav", tmp_path / "ref.txt", tmp_path / "out.json"
    write_wav(wav, len(SPEECH) * FRAME)
    ref.write_text("﻿Una frase.", encoding="utf-8")  # a BOM, as Notepad may write
    monkeypatch.setattr(replay, "SileroModel", lambda: ScriptedVad(SPEECH))
    model = FakeModel(Clock())
    code = replay.main([str(wav), "--ref", str(ref), "--json", str(out), "--language", "it"], load=lambda *a: model)
    assert code == 0
    text = capsys.readouterr().out
    assert "Una frase." in text and "WER 0.0%" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [f["text"] for f in data["finals"]] == ["Una frase."]
    assert data["score"]["wer"] == 0 and data["summary"]["segments"] == 1
    assert data["vad"]["min_silence_ms"] == SegmenterParams().min_silence_ms


def test_vad_parameters_come_from_the_command_line(tmp_path, monkeypatch, capsys):
    wav = tmp_path / "rec.wav"
    write_wav(wav, len(SPEECH) * FRAME)
    monkeypatch.setattr(replay, "SileroModel", lambda: ScriptedVad(SPEECH))
    model = FakeModel(Clock())
    # 0.9 is not speech any more at this threshold.
    assert replay.main([str(wav), "--threshold", "0.95", "--no-partial"], load=lambda *a: model) == 0
    assert model.finals == []
    assert "断句参数：threshold=0.95" in capsys.readouterr().out


def test_a_wav_in_another_format_is_refused(tmp_path):
    wav = tmp_path / "rec.wav"
    write_wav(wav, 800, rate=8000)
    with pytest.raises(SystemExit, match="16 kHz"):
        replay.read_wav(wav)
    with pytest.raises(SystemExit, match="16 kHz"):
        write_wav(wav, 1600, channels=2)
        replay.read_wav(wav)


def test_an_unknown_language_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="识别语言"):
        replay.main([str(tmp_path / "x.wav"), "--language", "klingon"], load=lambda *a: None)


# ---------------------------------------------------------------------------------- asr's reasons


@pytest.mark.parametrize(
    "text, prompt, reason",
    [
        ("Sottotitoli creati dalla comunità Amara.org", "", "phrase: sottotitoli creati"),
        ("www.unibo.it", "", "url only"),
        ("Trovate le slide su www.unibo.it", "", ""),
        ("Iscriviti al canale!", "", "stock phrase"),
        ("e e e e", "", "filler"),
        ("Grazie. Grazie a tutti.", "", "stock phrase"),
        ("...", "", "no words"),
        ("Lezione di economia.", "Lezione di economia. ", "hint echo"),
        ("PIL, inflazione, deflazione", "PIL, inflazione, deflazione", "glossary echo"),
        ("Oggi parliamo di inflazione.", "", ""),
        ("", "", ""),
    ],
)
def test_clean_text_says_why_it_drops_a_text(text, prompt, reason):
    out, why = clean_text_why(text, prompt)
    assert why == reason
    assert out == clean_text(text, prompt)
    if reason:
        assert out == ""


def piece(text, logprob=-0.2, no_speech=0.0, ratio=1.1):
    return SimpleNamespace(text=text, avg_logprob=logprob, no_speech_prob=no_speech, compression_ratio=ratio)


@pytest.mark.parametrize(
    "s, reason",
    [
        (piece("x", ratio=2.6), "repeats"),
        (piece("x", logprob=-1.2), "unsure"),
        (piece("x", logprob=-0.6, no_speech=0.9), "no speech"),
        (piece("x", logprob=-0.4, no_speech=0.9), ""),
        (piece("x"), ""),
    ],
)
def test_unreliable_says_why(s, reason):
    assert unreliable_why(s).partition(":")[0] == reason
    assert unreliable(s) == bool(reason)


class Pieces:
    """A WhisperModel that decodes every audio as the given pieces."""

    def __init__(self, *pieces):
        self.pieces = pieces

    def detect_language(self, audio):
        return "it", 0.9, [("it", 0.9), ("en", 0.1)]

    def transcribe(self, audio, **kw):
        return iter(self.pieces), None


def transcriber(*pieces):
    tr = Transcriber()
    tr.model, tr.status = Pieces(*pieces), "ready"
    tr.count_tokens = lambda text: len(text.split())
    return tr


def test_transcribe_says_what_it_threw_away_and_returns_the_same():
    audio = np.zeros(16000, np.float32)
    tr = transcriber(piece(" Buongiorno a tutti."), piece(" Grazie per la visione"), piece(" boh", logprob=-1.5))
    plain = tr.transcribe(audio, "it")
    dropped = []
    traced = tr.transcribe(audio, "it", dropped=dropped)
    assert traced == {**plain, "asr_ms": traced["asr_ms"]}
    assert traced["text"] == "Buongiorno a tutti."
    assert dropped == [("Grazie per la visione", "phrase: grazie per la visione"), ("boh", "unsure: avg logprob -1.50")]

    tr = transcriber(piece(" il pacchetto arriva al router"))
    dropped = []
    res = tr.transcribe(audio, "it", "", "e quindi il pacchetto arriva al router", "it", dropped=dropped)
    assert res["text"] == ""
    assert dropped == [("il pacchetto arriva al router", "context echo")]
