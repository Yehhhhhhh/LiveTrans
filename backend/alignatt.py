"""Experimental local engine: Whisper decoded with the AlignAtt simultaneous policy.

The idea comes from SimulStreaming (https://github.com/ufal/SimulStreaming, MIT, Macháček & Polák,
IWSLT 2025) and Simul-Whisper. This is a small re-implementation on OpenAI's PyTorch Whisper
(`openai-whisper`), not a copy of their code.

While a segment is still being spoken, each update decodes the audio so far with the words
already shown forced as the start of the transcript, and only adds new words: decoding stops as
soon as the decoder's cross-attention (its alignment heads) reaches the last FRAME_THRESHOLD
frames of the audio, where the words may still be cut off. The last word decoded is held back too
(it may be the first half of a longer word). So the live text only grows and is never rewritten,
and each update decodes only the new words. When the segment ends, the rest is decoded to the end.

CTranslate2 (faster-whisper) doesn't expose the cross-attention, hence PyTorch Whisper. The engine
is chosen as a model ("alignatt-large-v3-turbo") in 设置 → 识别 → 模型 and needs the packages
in requirements-alignatt.txt.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import time
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from asr import (
    PROGRESS_INTERVAL_S,
    clean_text,
    collapse_repeats,
    detect_among,
    echoes_context,
    fit_prompt,
    is_fixed,
    normalize_language,
    unreliable,
    IT_EN,
)

log = logging.getLogger("livetrans.alignatt")

PREFIX = "alignatt-"
# Encoder frames (20 ms each) at the end of the audio that the decoder may not attend to while the
# segment is still being spoken. SimulStreaming's default; larger is steadier but later.
FRAME_THRESHOLD = 25
SAMPLES_PER_FRAME = 320  # one encoder frame: 20 ms at 16 kHz
# Loop guard: at most this many tokens per second of audio (plus a few), well above real speech.
TOKENS_PER_S = 10
TOKENS_BASE = 20


def available() -> bool:
    """Whether the optional packages (requirements-alignatt.txt) are installed."""
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "whisper"))


def is_alignatt(model_name: str) -> bool:
    return model_name.startswith(PREFIX)


@dataclass
class StreamState:
    """What has been shown of one segment so far (see AlignAttModel.transcribe_partial)."""

    lang: str = ""
    tokens: list[int] = field(default_factory=list)  # words shown, as Whisper tokens
    # Audio samples (from the segment's start) that the shown words were attended up to. If the
    # final audio of the segment is shorter (a hard cut put the rest in the next segment), the
    # shown words can't be kept as its start.
    end: int = 0


def model_cache_dir() -> Path:
    default = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"
    return Path(os.environ.get("LIVETRANS_WHISPER_PT_DIR", default))


def fetch_checkpoint(name: str, report) -> str:
    """Download OpenAI's checkpoint of a Whisper model (or find it) and return its path, reporting
    progress as report(done, total, rate) like asr.fetch_model. Checked against its SHA-256."""
    import whisper

    url = whisper._MODELS[name]
    sha = url.split("/")[-2]
    root = model_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / os.path.basename(url)
    if path.is_file():
        return str(path)
    part = path.with_suffix(".part")
    digest = hashlib.sha256()
    t0 = last = time.monotonic()
    with urllib.request.urlopen(url) as r, open(part, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            now = time.monotonic()
            if total and now - last >= PROGRESS_INTERVAL_S:
                last = now
                report(done, total, done / max(now - t0, 1e-3))
    if digest.hexdigest() != sha:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{name} 模型文件校验失败，请重试")
    part.replace(path)
    return str(path)


class AlignAttModel:
    """A Whisper model on PyTorch with AlignAtt decoding. Not thread-safe: the server runs one
    call at a time (gpu_lock)."""

    alignatt = True  # asr.Transcriber hands its calls to this model (see Transcriber.engine)

    def __init__(self, model_name: str, device: str, report) -> None:
        import torch
        import whisper

        name = model_name[len(PREFIX):]
        path = fetch_checkpoint(name, report)
        model = whisper.load_model(path, device="cpu")
        model.set_alignment_heads(whisper._ALIGNMENT_HEADS[name])
        self.heads = [tuple(h) for h in model.alignment_heads.indices().T.tolist()]  # (layer, head)
        if device == "cuda":
            # Half precision except LayerNorm, which whisper runs in float32 (it casts its input).
            for m in model.modules():
                if not isinstance(m, torch.nn.LayerNorm):
                    for p in m.parameters(recurse=False):
                        p.data = p.data.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.model = model.to(device).eval()
        self.device = device
        self.torch = torch
        self.whisper = whisper
        self.compute_type = "float16" if device == "cuda" else "float32"
        # Warm up so the first real segment doesn't pay CUDA init / kernel load.
        self.transcribe(np.zeros(16000, dtype=np.float32), "en")

    # -------------------------------------------------------------- pieces

    def tokenizer(self, lang: str):
        from whisper.tokenizer import get_tokenizer

        m = self.model
        return get_tokenizer(m.is_multilingual, num_languages=m.num_languages, language=lang, task="transcribe")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer("en").encode(" " + text.strip()))

    def encode(self, audio: np.ndarray):
        """Encoder output of the audio (its last 30 s) and the number of encoder frames it fills."""
        from whisper.audio import HOP_LENGTH, N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        audio = audio[-N_SAMPLES:]
        x = self.torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(self.device)
        mel = pad_or_trim(log_mel_spectrogram(x, self.model.dims.n_mels, padding=N_SAMPLES), N_FRAMES)
        with self.torch.no_grad():
            feats = self.model.embed_audio(mel[None].to(self.dtype))
        return feats, max(1, len(audio) // HOP_LENGTH // 2)

    def detect(self, feats, among: tuple[str, ...] | None) -> tuple[str, float]:
        with self.torch.no_grad():
            _, probs = self.model.detect_language(feats, self.tokenizer("en"))
        probs = probs[0]
        if among is None:
            lang = max(probs, key=probs.get)
            return lang, probs[lang]
        scores = {lang: probs.get(lang, 0.0) for lang in among}
        lang = max(among, key=scores.get)
        return lang, scores[lang] / (sum(scores.values()) or 1.0)

    def decode(self, feats, frames: int, lang: str, prompt: str, forced: list[int], is_last: bool):
        """Greedy decoding after `forced`. Unless is_last, it stops where the attention reaches the
        end of the audio. Returns (new tokens, the frame each was attended to, stats)."""
        torch = self.torch
        from whisper.model import disable_sdpa

        tok = self.tokenizer(lang)
        prefix = [tok.sot_prev] + tok.encode(" " + prompt.strip()) if prompt.strip() else []
        start = prefix + list(tok.sot_sequence_including_notimestamps)
        sot_index = len(prefix)
        tokens = start + forced
        budget = TOKENS_BASE + frames * TOKENS_PER_S // 50 - len(forced)
        budget = max(4, min(budget, self.model.dims.n_text_ctx - len(tokens) - 1))

        suppress = torch.zeros(self.model.dims.n_vocab, dtype=torch.bool, device=self.device)
        suppress[list(tok.non_speech_tokens)] = True
        special = [tok.sot, tok.sot_prev, tok.sot_lm, tok.no_speech, tok.transcribe, tok.translate]
        suppress[[t for t in special if t is not None]] = True
        suppress[tok.timestamp_begin:] = True

        qks: dict[int, object] = {}
        cache, hooks = self.model.install_kv_cache_hooks()
        for layer in {layer for layer, _ in self.heads}:
            hooks.append(self.model.decoder.blocks[layer].cross_attn.register_forward_hook(
                lambda _m, _i, out, layer=layer: qks.__setitem__(layer, out[-1])
            ))

        def attended() -> int:
            rows = [qks[layer][0, head, -1].float().softmax(-1)[:frames] for layer, head in self.heads]
            return int(torch.stack(rows).mean(0).argmax())

        new: list[int] = []
        where: list[int] = []
        logprob = 0.0
        try:
            with torch.no_grad(), disable_sdpa():
                logits = self.model.decoder(torch.tensor([tokens], device=self.device), feats, kv_cache=cache)
                no_speech = float(logits[0, sot_index].float().softmax(-1)[tok.no_speech])
                while len(new) < budget:
                    frame = attended()
                    if not is_last and frames - frame <= FRAME_THRESHOLD:
                        break
                    step = logits[0, -1].float()
                    step[suppress] = -np.inf
                    if not new and not forced:  # no empty transcript (as Whisper's SuppressBlank)
                        step[tok.encode(" ") + [tok.eot]] = -np.inf
                    lp = step.log_softmax(-1)
                    nxt = int(lp.argmax())
                    if nxt == tok.eot:
                        break
                    new.append(nxt)
                    where.append(frame)
                    logprob += float(lp[nxt])
                    logits = self.model.decoder(torch.tensor([[nxt]], device=self.device), feats, kv_cache=cache)
        finally:
            for h in hooks:
                h.remove()
        text = tok.decode(forced + new)
        raw = text.encode("utf-8")
        stats = SimpleNamespace(
            no_speech_prob=no_speech,
            avg_logprob=logprob / len(new) if new else 0.0,
            compression_ratio=len(raw) / max(1, len(zlib.compress(raw))) if raw else 0.0,
        )
        return new, where, stats

    @staticmethod
    def drop_last_word(tok, new: list[int], where: list[int]) -> tuple[list[int], list[int]]:
        """The tokens before the last word starts: that word may continue in the next audio."""
        starts = [i for i, t in enumerate(new) if tok.decode([t]).startswith(" ")]
        cut = starts[-1] if starts else 0
        return new[:cut], where[:cut]

    # -------------------------------------------------------------- the Transcriber interface

    def pick_language(self, audio: np.ndarray, among: tuple[str, ...] | None) -> tuple[str, float]:
        feats, _ = self.encode(audio)
        return self.detect(feats, among)

    def transcribe_partial(self, audio: np.ndarray, lang: str, prompt: str = "", state: StreamState | None = None) -> str:
        """The words of a segment still being spoken that are certain by now: those in `state`
        (shown before) plus the new ones. Without a state, just this update's certain words."""
        state = state if state is not None else StreamState()
        if state.lang != lang:
            state.lang, state.tokens, state.end = lang, [], 0
        tok = self.tokenizer(lang)
        initial = fit_prompt(self.count_tokens, prompt) if prompt else ""
        feats, frames = self.encode(audio)
        new, where, stats = self.decode(feats, frames, lang, initial, state.tokens, is_last=False)
        new, where = self.drop_last_word(tok, new, where)
        if new and self._acceptable(tok, state.tokens, new, stats, prompt):
            state.tokens = state.tokens + new
            state.end = (max(where) + FRAME_THRESHOLD // 2) * SAMPLES_PER_FRAME
        return clean_text(tok.decode(state.tokens), prompt) if state.tokens else ""

    @staticmethod
    def _acceptable(tok, shown: list[int], new: list[int], stats, prompt: str) -> bool:
        """Whether new words may be shown (and then forced into the final text): not the start of
        a hallucination, a repeat loop, or a very unsure guess."""
        if unreliable(stats) or (not shown and stats.no_speech_prob > 0.5):
            return False
        text = " ".join(tok.decode(shown + new).split())
        return bool(text) and clean_text(text, prompt) == collapse_repeats(text) == text

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = IT_EN,
        prompt: str = "",
        context: str = "",
        context_lang: str = "",
        state: StreamState | None = None,
    ) -> dict:
        """A whole segment, as asr.Transcriber.transcribe. The words shown while it was spoken
        (`state`) are kept as its start, so the final text only adds to them."""
        t0 = time.time()
        mode = normalize_language(language) or IT_EN
        feats, frames = self.encode(audio)
        if is_fixed(mode):
            lang, lang_prob = mode, 1.0
        elif state and state.tokens:
            lang, lang_prob = state.lang, 1.0  # the shown words were decoded in it
        else:
            lang, lang_prob = self.detect(feats, detect_among(mode))
        forced = []
        if state and state.tokens and state.lang == lang and state.end <= len(audio):
            forced = state.tokens
        if context_lang != lang:
            context = ""
        initial = fit_prompt(self.count_tokens, prompt, context) if prompt or context else ""
        new, _, stats = self.decode(feats, frames, lang, initial, forced, is_last=True)
        if unreliable(stats):
            new = []  # an unsure ending: keep only the words already shown
        tok = self.tokenizer(lang)
        text = clean_text(tok.decode(forced + new), prompt) if forced + new else ""
        if context and echoes_context(text, context):
            text = ""
        return {
            "lang": lang,
            "lang_prob": round(lang_prob, 3),
            "text": text,
            "asr_ms": int((time.time() - t0) * 1000),
            "kept": len(forced),
        }
