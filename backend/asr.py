"""faster-whisper wrapper restricted to Italian / English."""

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

# The fixed header app.js puts before the glossary terms.
PROMPT_HEADER_WORDS = {"lezione", "di", "telecomunicazioni", "telecommunications", "lecture"}

_WORD = re.compile(r"[\w']+", re.U)


def _norm(token: str) -> str:
    return " ".join(_WORD.findall(token.lower()))


def collapse_repeats(text: str, min_repeats: int = 3, max_unit: int = 6) -> str:
    """Collapse a word or short phrase repeated >= min_repeats times in a row to one copy
    ("ruota, ruota, ruota, ruota" -> "ruota,")."""
    tokens = text.split()
    norm = [_norm(t) for t in tokens]
    out: list[str] = []
    i = 0
    while i < len(tokens):
        best_n, best_count = 0, 1
        for n in range(1, max_unit + 1):
            if i + n * min_repeats > len(tokens):
                break
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
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def prompt_words(prompt: str) -> set[str]:
    return set(_WORD.findall(prompt.lower())) - PROMPT_HEADER_WORDS if prompt else set()


def clean_text(text: str, prompt: str = "") -> str:
    """Return the text with repeat loops collapsed, or "" if it looks like a hallucination."""
    text = collapse_repeats(text.strip())
    if not text or HALLUCINATIONS.search(text):
        return ""
    words = _WORD.findall(text.lower())
    if not words or STOCK_PHRASES.match(" ".join(words)):
        return ""
    # Echo of the glossary prompt: a bare list of 3+ words that all come from the prompt
    # (or the prompt's header) is Whisper reciting its prompt, not the lecturer speaking.
    vocab = prompt_words(prompt)
    if vocab and len(words) >= 3 and all(w in vocab or w in PROMPT_HEADER_WORDS for w in words):
        return ""
    if prompt and all(w in PROMPT_HEADER_WORDS for w in words):
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

    def pick_language(self, audio: np.ndarray) -> tuple[str, float]:
        _, _, probs = self.model.detect_language(audio)
        scores = {lang: p for lang, p in probs if lang in LANGS}
        lang = max(LANGS, key=lambda k: scores.get(k, 0.0))
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

    def transcribe(self, audio: np.ndarray, language: str = "auto", prompt: str = "") -> dict:
        t0 = time.time()
        if language in LANGS:
            lang, lang_prob = language, 1.0
        else:
            lang, lang_prob = self.pick_language(audio)
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
