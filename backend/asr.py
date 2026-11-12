"""faster-whisper wrapper: Italian / English by default, any Whisper language on request."""

from __future__ import annotations

import logging
import os
import re
import site
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("livetrans.asr")

LANGS = ("it", "en")

# Recognition-language setting: ANY (detect among all of Whisper's languages), IT_EN (each
# segment either Italian or English, the default), or one fixed language code.
ANY, IT_EN = "any", "it_en"


def all_languages() -> tuple[str, ...]:
    """Language codes Whisper (multilingual models) can recognise."""
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES
    except ImportError:  # private name; fall back to the pair this app is built around
        return LANGS
    return tuple(_LANGUAGE_CODES)


def normalize_language(value: object) -> str | None:
    """The setting as used here, or None if it isn't valid. "auto" is the old name of IT_EN."""
    if value == "auto":
        return IT_EN
    if value in (ANY, IT_EN) or (isinstance(value, str) and value in all_languages()):
        return value
    return None


def detect_among(mode: str) -> tuple[str, ...] | None:
    """Languages to choose between when detecting (None = all), for a non-fixed mode."""
    return LANGS if mode == IT_EN else None


def is_fixed(mode: str) -> bool:
    return mode not in (ANY, IT_EN)

# Phrases Whisper tends to hallucinate on noise or silence.
HALLUCINATIONS = re.compile(
    r"(amara\.org|sottotitoli (creati|a cura|e revisione)|grazie per (la visione|l'attenzione|aver guardato)"
    r"|thanks? (you )?for watching|subscribe|iscriviti|www\.|\.com\b)",
    re.I,
)

# Whole outputs Whisper produces on noise / silence (outro-subtitle habits). Matched against
# the entire segment text, after repeats are collapsed, so real sentences containing them survive.
STOCK_PHRASES = re.compile(
    r"^(grazie( a tutti| mille| ancora| a voi)?|thank you( (very much|all|everyone|so much))?|thanks"
    r"|ciao( a tutti)?|bye( bye)?|wtf|you|buona (visione|serata|giornata)|alla prossima)$",
    re.I,
)

# Ignored when deciding whether an output only echoes the prompt.
FUNCTION_WORDS = {"e", "di", "a", "la", "il", "le", "and", "the", "of", "to"}
_CLAUSE = re.compile(r"[.,;:!?…]+")

_WORD = re.compile(r"[\w']+", re.U)


def _norm(token: str) -> str:
    return " ".join(_WORD.findall(token.lower()))


def collapse_repeats(text: str) -> str:
    """Collapse Whisper repeat loops to one copy: a word or phrase repeated 3+ times
    ("ruota, ruota, ruota" -> "ruota,"), or a sentence of 8+ words repeated 2+ times.
    A truncated copy of the unit trailing the loop is dropped too."""
    tokens = text.split()
    norm = [_norm(t) for t in tokens]
    out: list[str] = []
    i = 0
    while i < len(tokens):
        best_n, best_count = 0, 1
        for n in range(1, (len(tokens) - i) // 2 + 1):
            min_repeats = 2 if n >= 8 else 3
            if i + n * min_repeats > len(tokens):
                continue
            unit = norm[i : i + n]
            if not any(unit):
                continue
            count = 1
            while norm[i + count * n : i + (count + 1) * n] == unit:
                count += 1
            if count >= min_repeats and count * n > best_count * best_n:
                best_n, best_count = n, count
        if best_n:
            out.extend(tokens[i : i + best_n])
            i += best_n * best_count
            # A partial result often ends mid-way through yet another copy.
            unit = norm[i - best_n : i]
            tail = 0
            while tail < best_n and i + tail < len(norm) and norm[i + tail] == unit[tail]:
                tail += 1
            if tail >= 2 and i + tail == len(norm):
                i += tail
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def split_prompt(prompt: str) -> tuple[set[str], set[str]]:
    """Words of the course hint and of the glossary terms in a Whisper prompt.

    app.js builds the prompt as "<course hint>. <term>, <term>, ..." (the hint is set in
    设置 → 识别, default "Lezione di telecomunicazioni / Telecommunications lecture"), or just
    "<term>, <term>, ..." when the hint is empty, or "<course hint>. " when there are no terms:
    the hint is everything before the first ". ".
    """
    if not prompt:
        return set(), set()
    hint, sep, terms = prompt.partition(". ")
    if not sep:
        hint, terms = "", prompt
    hint_words = set(_WORD.findall(hint.lower()))
    return hint_words, set(_WORD.findall(terms.lower())) - hint_words


def clean_text(text: str, prompt: str = "") -> str:
    """Return the text with repeat loops collapsed, or "" if it looks like a hallucination."""
    raw = text.strip()
    text = collapse_repeats(raw)
    if not text or HALLUCINATIONS.search(text):
        return ""
    words = _WORD.findall(text.lower())
    # A loop of one or two filler words ("e e e e", "si si si") has nothing worth showing.
    if len(words) <= 2 and text != raw:
        return ""
    # Every clause a stock phrase ("Grazie, grazie.", "Grazie. Grazie a tutti.").
    clauses = [" ".join(_WORD.findall(c)) for c in _CLAUSE.split(text.lower())]
    if all(STOCK_PHRASES.match(c) for c in clauses if c):
        return ""
    # Echo of the glossary prompt: content words that all come from the prompt (or its course
    # hint) are Whisper reciting its prompt, not the lecturer speaking.
    content = [w for w in words if w not in FUNCTION_WORDS]
    hint, vocab = split_prompt(prompt)
    # Only a real echo of the hint: at least two of its content words (a hint with a single content
    # word can't be told apart). A lecturer saying just "Economics." under the hint
    # "Economics lecture" is speech, not an echo.
    hint_content = hint - FUNCTION_WORDS
    echo_min = min(2, len(hint_content))
    if hint and content and all(w in hint for w in content) and len(set(content)) >= echo_min:
        return ""
    if vocab and len(content) >= 3 and all(w in vocab or w in hint for w in content):
        return ""
    return text


def setup_cuda_dlls() -> None:
    """Make the pip-installed CUDA/cuDNN DLLs (nvidia-*-cu12) loadable on Windows."""
    if os.name != "nt":
        return
    roots = [Path(p) for p in site.getsitepackages()]
    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    found = []
    for root in roots:
        for bin_dir in (root / "nvidia").glob("*/bin"):
            if bin_dir.is_dir():
                found.append(str(bin_dir))
    for d in found:
        os.add_dll_directory(d)
    if found:
        os.environ["PATH"] = os.pathsep.join(found + [os.environ.get("PATH", "")])


class Transcriber:
    def __init__(self) -> None:
        self.model = None
        self.status = "idle"  # idle | loading | ready | error
        self.error = ""
        self.device = ""
        self.compute_type = ""
        self.model_name = ""

    def load(self, model_name: str, device: str = "auto", compute_type: str = "auto") -> None:
        from faster_whisper import WhisperModel

        self.status = "loading"
        self.model_name = model_name
        attempts = []
        if device in ("auto", "cuda"):
            attempts.append(("cuda", "float16" if compute_type == "auto" else compute_type))
        if device in ("auto", "cpu"):
            attempts.append(("cpu", "int8" if compute_type == "auto" else compute_type))
        errors = []
        for dev, ct in attempts:
            try:
                t0 = time.time()
                model = WhisperModel(model_name, device=dev, compute_type=ct)
                # Warm up so the first real segment doesn't pay CUDA init / kernel load.
                list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)[0])
                self.model, self.device, self.compute_type = model, dev, ct
                self.status = "ready"
                self.error = "" if dev == "cuda" or device == "cpu" else "未能使用 GPU，已回退到 CPU：" + "; ".join(errors)
                log.info("Loaded %s on %s/%s in %.1fs", model_name, dev, ct, time.time() - t0)
                return
            except Exception as e:  # noqa: BLE001 - report any backend failure and try the next one
                log.exception("Failed to load %s on %s/%s", model_name, dev, ct)
                errors.append(f"{dev}: {e}")
        self.status = "error"
        self.error = "模型加载失败：" + "; ".join(errors)

    def pick_language(self, audio: np.ndarray, among: tuple[str, ...] | None = LANGS) -> tuple[str, float]:
        """Most likely language of the audio, restricted to `among` (None = any), and its
        probability relative to those candidates."""
        _, _, probs = self.model.detect_language(audio)
        if among is None:
            lang, p = max(probs, key=lambda lp: lp[1])
            return lang, p
        scores = {lang: p for lang, p in probs if lang in among}
        lang = max(among, key=lambda k: scores.get(k, 0.0))
        total = sum(scores.values()) or 1.0
        return lang, scores.get(lang, 0.0) / total

    def transcribe_partial(self, audio: np.ndarray, lang: str, prompt: str = "") -> str:
        """Fast greedy pass over a segment still being spoken; the final pass redoes it properly."""
        segments, _ = self.model.transcribe(
            audio,
            language=lang,
            beam_size=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=prompt or None,
        )
        text = clean_text(" ".join(s.text.strip() for s in segments if s.text.strip()), prompt)
        # Whisper ends cut-off audio with "..." or "."; dropping it keeps the last word stable on screen.
        return re.sub(r"[.…]+$", "", text).strip()

    def transcribe(self, audio: np.ndarray, language: str = IT_EN, prompt: str = "") -> dict:
        t0 = time.time()
        mode = normalize_language(language) or IT_EN
        if is_fixed(mode):
            lang, lang_prob = mode, 1.0
        else:
            lang, lang_prob = self.pick_language(audio, detect_among(mode))
        segments, _ = self.model.transcribe(
            audio,
            language=lang,
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=prompt or None,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        parts = []
        for s in segments:
            # Even after temperature fallback Whisper returns its last attempt; reject it here
            # if it is still a repeat loop, very unsure, or probably not speech at all.
            if s.compression_ratio > 2.4 or s.avg_logprob < -1.0:
                continue
            if s.no_speech_prob > 0.5 and s.avg_logprob < -0.5:
                continue
            text = clean_text(s.text, prompt)
            if text:
                parts.append(text)
        return {
            "lang": lang,
            "lang_prob": round(lang_prob, 3),
            "text": clean_text(" ".join(parts), prompt),
            "asr_ms": int((time.time() - t0) * 1000),
        }
