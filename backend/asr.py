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
        text = " ".join(s.text.strip() for s in segments if s.text.strip())
        # Whisper ends cut-off audio with "..." or "."; dropping it keeps the last word stable on screen.
        text = re.sub(r"[.…]+$", "", text).strip()
        return "" if HALLUCINATIONS.search(text) else text

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
            if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
                continue
            text = s.text.strip()
            if text and not HALLUCINATIONS.search(text):
                parts.append(text)
        return {
            "lang": lang,
            "lang_prob": round(lang_prob, 3),
            "text": " ".join(parts),
            "asr_ms": int((time.time() - t0) * 1000),
        }
