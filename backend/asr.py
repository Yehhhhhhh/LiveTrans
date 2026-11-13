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
    设置 → 识别, blank by default), or just
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


# faster-whisper keeps only the last 223 tokens of a prompt (half of Whisper's 448-token window,
# see WhisperModel.get_prompt) and silently drops what comes before; stay a little under that.
PROMPT_TOKENS = 215


def fit_prompt(count, prompt: str, context: str = "") -> str:
    """The Whisper prompt for one segment: `prompt` (see split_prompt) followed by `context`, the
    text recognized just before, cut down to PROMPT_TOKENS so that nothing is dropped silently.

    The previous sentence goes last because Whisper reads its prompt as the transcript so far:
    it continues in that spelling, casing and vocabulary. What gives way when it doesn't all fit:
    first the glossary terms from the end (app.js lists the newest terms first), then the start
    of the context. The course hint always stays. `count` returns the number of tokens of a text.
    """
    context = context.strip()
    if not prompt and not context:
        return ""
    hint, sep, rest = prompt.partition(". ")
    if not sep:
        hint, rest = "", prompt
    head = f"{hint.strip()}. " if hint.strip() else ""
    terms = [t.strip() for t in rest.strip().rstrip(".").split(", ") if t.strip()]

    budget = PROMPT_TOKENS - (count(head) if head else 0)
    while context and count(context) > budget:
        words = context.split()
        if len(words) > 1:
            context = " ".join(words[max(1, len(words) // 8):])
        else:  # Chinese, Japanese…: no spaces to cut at
            context = context[max(1, len(context) // 8):]
    budget -= count(context) if context else 0

    kept = []
    for term in terms:
        budget -= count(term + ",")
        if budget < 0:
            break
        kept.append(term)
    text = head + (", ".join(kept) + "." if kept else "")
    return f"{text.rstrip()} {context}".strip() if context else text


def echoes_context(text: str, context: str) -> bool:
    """Whether `text` is just the end of `context` again. On noise or a breath Whisper sometimes
    recites the end of its prompt instead of transcribing; a real sentence repeated word for word
    is much rarer, and short ones ("Va bene.") are left alone."""
    words = _WORD.findall(text.lower())
    before = _WORD.findall(context.lower())
    if len(words) >= 4:
        return before[-len(words):] == words
    # Chinese, Japanese…: written without spaces, a whole clause is one "word" here, so
    # compare the characters instead.
    flat = "".join(words)
    unspaced = any(ord(c) >= 0x2E80 for c in flat)
    return unspaced and len(flat) >= 8 and "".join(before).endswith(flat)


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


# The files of a faster-whisper model (as faster_whisper.utils.download_model fetches them).
MODEL_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]
PROGRESS_INTERVAL_S = 0.5


class _NoOutput:
    """Where the download's progress bars print: nowhere."""

    def write(self, text: str) -> None:
        pass

    def flush(self) -> None:
        pass


def fetch_model(name: str, report) -> str:
    """Download the model's files (or find them in the cache) and return their folder.

    faster-whisper downloads a model by itself, but with the progress bars switched off, and a
    3 GB download that shows nothing looks like a hang. This does the same download with bars
    that call `report(done, total, rate)` (bytes, bytes, bytes per second or None) instead of
    printing, at most every PROGRESS_INTERVAL_S. Nothing is reported for a model already cached.
    """
    import huggingface_hub
    from faster_whisper.utils import _MODELS
    from tqdm import tqdm

    repo = name if "/" in name else _MODELS[name]
    bars: list = []
    last = [0.0]

    class Bar(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["file"] = _NoOutput()
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            if self.unit == "B":
                bars.append(self)

        def update(self, n=1):
            out = super().update(n)
            # Two byte bars: received from the network, and written to disk (which lags behind,
            # in bursts). The first to move shows progress; the smaller total is the real size,
            # the other one is stretched by the library when more bytes arrive than expected.
            totals = [b.total for b in bars if b.total]
            now = time.monotonic()
            if self.unit == "B" and totals and now - last[0] >= PROGRESS_INTERVAL_S:
                last[0] = now
                total = min(totals)
                report(min(max(b.n for b in bars), total), total, self.format_dict.get("rate"))
            return out

    return huggingface_hub.snapshot_download(repo, allow_patterns=MODEL_FILES, tqdm_class=Bar)


class Transcriber:
    def __init__(self) -> None:
        self.model = None
        self.status = "idle"  # idle | loading | ready | error
        self.error = ""
        self.device = ""
        self.compute_type = ""
        self.model_name = ""
        # (model, compute type) the loaded model was asked for with; compute_type above is the
        # one in use ("auto" resolved). None while nothing is loaded.
        self.choice: tuple[str, str] | None = None
        # (done, total, rate) of the model download in progress, in bytes (rate: per second or
        # None); None when nothing is being downloaded. on_progress is called when it changes.
        self.progress: tuple[int, int, float | None] | None = None
        self.on_progress = lambda: None

    def load(
        self,
        model_name: str,
        device: str = "auto",
        compute_type: str = "auto",
        fallback: tuple[str, str] | None = None,
    ) -> None:
        """Load a model. `fallback` is a (model, compute type) to load instead when that fails.

        The status stays "loading" until the outcome of both is known: whatever waits for the
        model (see server.wait_for_model) must not act on a failure that is about to be repaired.
        """
        self.status = "loading"
        error = self._load(model_name, device, compute_type)
        if error and fallback and fallback != (model_name, compute_type):
            first = error
            error = self._load(fallback[0], device, fallback[1])
            if error:
                error = f"{first}; {error}"
            else:
                self.error = f"{model_name}（{compute_type}）加载失败，已换回 {fallback[0]}（{fallback[1]}）。{first}"
        if error:
            self.model, self.choice = None, None
            self.error = "模型加载失败：" + error
            self.status = "error"
        else:
            self.status = "ready"

    def _load(self, model_name: str, device: str, compute_type: str) -> str:
        """Try to load a model: "" when it is loaded, else what went wrong."""
        from faster_whisper import WhisperModel

        self.model_name = model_name
        path = model_name
        if not os.path.isdir(model_name):
            try:
                path = fetch_model(model_name, self._report)
            except Exception:  # noqa: BLE001 - let faster-whisper try, and report its error
                log.exception("Downloading %s with progress failed", model_name)
            finally:
                self.progress = None
                self.on_progress()
        attempts = []
        if device in ("auto", "cuda"):
            attempts.append(("cuda", "float16" if compute_type == "auto" else compute_type))
        if device in ("auto", "cpu"):
            attempts.append(("cpu", "int8" if compute_type == "auto" else compute_type))
        errors = []
        for dev, ct in attempts:
            try:
                t0 = time.time()
                model = WhisperModel(path, device=dev, compute_type=ct)
                # Warm up so the first real segment doesn't pay CUDA init / kernel load.
                list(model.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)[0])
                self.model, self.device, self.compute_type = model, dev, ct
                self.choice = (model_name, compute_type)
                self.error = "" if dev == "cuda" or device == "cpu" else "未能使用 GPU，已回退到 CPU：" + "; ".join(errors)
                log.info("Loaded %s on %s/%s in %.1fs", model_name, dev, ct, time.time() - t0)
                return ""
            except Exception as e:  # noqa: BLE001 - report any backend failure and try the next one
                log.exception("Failed to load %s on %s/%s", model_name, dev, ct)
                errors.append(f"{dev}: {e}")
        return "; ".join(errors)

    def _report(self, done: int, total: int, rate: float | None) -> None:
        self.progress = (int(done), int(total), rate)
        self.on_progress()

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

    def count_tokens(self, text: str) -> int:
        # As faster-whisper encodes a prompt: a leading space, no special tokens.
        return len(self.model.hf_tokenizer.encode(" " + text.strip(), add_special_tokens=False).ids)

    def transcribe_partial(self, audio: np.ndarray, lang: str, prompt: str = "") -> str:
        """Fast greedy pass over a segment still being spoken; the final pass redoes it properly."""
        initial_prompt = fit_prompt(self.count_tokens, prompt) if prompt else ""
        segments, _ = self.model.transcribe(
            audio,
            language=lang,
            beam_size=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=initial_prompt or None,
        )
        text = clean_text(" ".join(s.text.strip() for s in segments if s.text.strip()), prompt)
        # Whisper ends cut-off audio with "..." or "."; dropping it keeps the last word stable on screen.
        return re.sub(r"[.…]+$", "", text).strip()

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = IT_EN,
        prompt: str = "",
        context: str = "",
        context_lang: str = "",
    ) -> dict:
        """`context` is the text recognized just before this audio, in `context_lang`. It is given
        to Whisper as the transcript so far (see fit_prompt), but only when this audio is in the
        same language: text in another language pulls the decoding toward that language."""
        t0 = time.time()
        mode = normalize_language(language) or IT_EN
        if is_fixed(mode):
            lang, lang_prob = mode, 1.0
        else:
            lang, lang_prob = self.pick_language(audio, detect_among(mode))
        if context_lang != lang:
            context = ""
        initial_prompt = fit_prompt(self.count_tokens, prompt, context) if prompt or context else ""
        segments, _ = self.model.transcribe(
            audio,
            language=lang,
            beam_size=5,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=initial_prompt or None,
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
        text = clean_text(" ".join(parts), prompt)
        if context and echoes_context(text, context):
            text = ""
        return {
            "lang": lang,
            "lang_prob": round(lang_prob, 3),
            "text": text,
            "asr_ms": int((time.time() - t0) * 1000),
        }
