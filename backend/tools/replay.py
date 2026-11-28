"""Replay a recording through the local recognition pipeline, offline, and measure the result.

    .venv\\Scripts\\python backend\\tools\\replay.py recordings\\20261128-101500.wav --ref ref.txt

The WAV (16 kHz mono, as LIVETRANS_RECORD=1 saves the audio of a capture) goes to vad.Segmenter in
100 ms chunks, as the page sends it, and the segments to asr.Transcriber: faster-whisper, or the
AlignAtt engine for an "alignatt-..." model. Live partials of the segment in progress follow the
server's rules (see Replay). Time is simulated: each call to the model runs for real and counts for
as long as it ran, on a clock that otherwise moves with the audio, and the GPU does one thing at a
time, finals first, as under server.gpu_lock. A replay so takes about as long as the model is busy,
and the delays it reports are those the page would show on this computer.

Printed: the recognized text with timestamps; the WER / CER against --ref (a plain-text transcript
of the whole recording); how often the live text was rewritten and what each live update cost; how
long each final result came after its segment ended; what the hallucination checks threw away, and
why (to spot real speech lost to them). --json saves all of it, for comparing runs.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import math
import re
import sys
import time
import unicodedata
import wave
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))  # the backend modules import each other by bare name

# server first: it makes the CUDA DLLs loadable before faster-whisper is imported, and its
# constants and helpers are the pacing replayed here.
import server  # noqa: E402
from alignatt import StreamState  # noqa: E402
from asr import IT_EN, Transcriber, detect_among, is_fixed, normalize_language  # noqa: E402
from vad import FRAME, SR, Segment, Segmenter, SegmenterParams, SileroModel  # noqa: E402

CHUNK = 1600  # samples per audio message: the page's AudioWorklet sends 100 ms at a time
# Times within this of each other count as the same moment, so that a sum of floats that should
# equal a chunk's time (a call ending just as audio arrives) doesn't land a microsecond past it.
EPS = 1e-6


@dataclass
class Partial:
    """One live update of the segment in progress."""

    index: int  # the segment's
    start: float  # when it began, in seconds on the stream's clock: also the end of its audio
    done: float  # when its text was ready
    text: str
    shown: bool = False  # whether the page would have shown it (set by Replay.finish)

    @property
    def cost(self) -> float:
        return self.done - self.start


@dataclass
class Final:
    """The final transcription of a batch: one segment, or several merged (server.take_batch)."""

    indices: list[int]
    start: float  # where the batch's audio starts and ends, on the stream's clock
    end: float
    closed: float  # when the VAD closed its last segment
    began: float  # when its transcription began
    done: float  # when its text was ready
    lang: str
    text: str
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (decoded text, reason) thrown away

    @property
    def latency(self) -> float:
        """From the end of the batch's audio to its text: the page's recognition delay."""
        return self.done - self.end


class Replay:
    """The local-Whisper path of server.ws_audio, on a simulated clock (see the module docstring).

    Mirrored from there: when a live partial may start (maybe_partial: the GPU idle and no final
    waiting, at most every PARTIAL_INTERVAL_S, on at least PARTIAL_MIN_AUDIO_S of audio), which
    language it uses (partial_job), and the worker's finals: one batch after the other, segments
    merged when they fall behind (take_batch), the text before them as context (transcribe_batch).
    Server-only concerns (several runs, resending after a reconnect, model loading) don't apply.
    """

    def __init__(
        self,
        transcriber,
        segmenter: Segmenter,
        language: str = IT_EN,
        prompt: str = "",
        *,
        partial: bool = True,
        context: bool = True,
        timer=time.perf_counter,
        on_partial=None,
        on_final=None,
    ) -> None:
        self.tr, self.seg = transcriber, segmenter
        self.language, self.prompt = language, prompt
        self.partial, self.context = partial, context
        self.timer = timer  # measures what each call to the model costs
        self.on_partial = on_partial or (lambda p: None)
        self.on_final = on_final or (lambda f: None)
        self.alignatt = getattr(transcriber, "engine", None) is not None
        self.partial_every = round(server.PARTIAL_INTERVAL_S * SR / FRAME)
        self.now = 0.0  # the stream's clock: the end of the audio received so far
        self.gpu_free = 0.0  # when the GPU is free again (server.gpu_lock)
        self.worker_free = 0.0  # when the worker has answered the batch it took
        self.queue: asyncio.Queue = asyncio.Queue()  # closed segments waiting, for server.take_batch
        self.carry: Segment | None = None  # handed back by take_batch: the worker's next segment
        self.starts: dict[int, float] = {}  # segment index -> where its audio starts
        self.closed: dict[int, float] = {}  # segment index -> when the VAD closed it
        self.gone: dict[int, float] = {}  # segment index -> when the VAD dropped it as noise
        self.last_frame = 0  # the server's live["last_frame"], live["lang_index"], live["lang"]
        self.lang_index, self.lang = 0, ""
        self.states: dict[int, StreamState] = {}  # AlignAtt: what the partials showed (server: shown)
        self.ctx = {"text": "", "lang": "", "end": 0.0}
        self.partials: list[Partial] = []
        self.finals: list[Final] = []

    def run(self, pcm: np.ndarray) -> None:
        """Feed int16 samples as the page sends them, then stop (停止) and answer what is left."""
        for pos in range(0, len(pcm), CHUNK):
            chunk = pcm[pos : pos + CHUNK]
            now = (pos + len(chunk)) / SR
            self.answer(now)  # batches the worker takes up before this audio arrives
            self.now = now
            self.take(self.seg.feed(chunk.tobytes()))
            if self.partial:
                self.maybe_partial()
        self.take(self.seg.flush())
        self.answer(math.inf)
        self.finish()

    def take(self, segments: list[Segment]) -> None:
        """Queue the segments the VAD closed just now; note those it dropped as noise."""
        for s in segments:
            self.starts[s.index], self.closed[s.index] = s.start, self.now
            self.queue.put_nowait(s)
        for index in self.seg.take_dropped():
            self.gone[index] = self.now
            self.states.pop(index, None)

    def answer(self, until: float) -> None:
        """Transcribe the batches the worker takes up by `until` (server: worker).

        Segments are closed only as audio arrives, so when the worker takes one up, the queue
        holds just what the server's would: everything closed by then."""
        while (self.carry or not self.queue.empty()) and self.worker_free <= until + EPS:
            first = self.carry or self.queue.get_nowait()
            taken = max(self.worker_free, self.closed[first.index])
            items, self.carry = server.take_batch(first, self.queue)
            # It may wait for the GPU: a partial started before the segment closed holds it.
            self.transcribe(items, max(taken, self.gpu_free))

    def transcribe(self, items: list[Segment], began: float) -> None:
        """One final transcription, as server.transcribe_batch makes it."""
        item = server.merge_segments(items)
        context = ("", "")
        if self.context and 0 <= item.start - self.ctx["end"] <= server.CONTEXT_MAX_GAP_S:
            context = (self.ctx["text"], self.ctx["lang"])
        args = [item.audio, self.language, self.prompt, *context]
        states = [self.states.pop(it.index, None) for it in items]
        if self.alignatt:  # a merged batch is decoded afresh
            args.append(states[0] if len(items) == 1 else None)
        dropped: list[tuple[str, str]] = []
        t0 = self.timer()
        res = self.tr.transcribe(*args, dropped=dropped)
        done = began + self.timer() - t0
        end = item.start + server.span_s(items)
        if res["text"]:
            joined = context[1] == res["lang"] and context[0]
            text = f"{context[0]} {res['text']}" if joined else res["text"]
            self.ctx.update(text=server.context_tail(text), lang=res["lang"], end=end)
        last = items[-1].index
        final = Final([it.index for it in items], item.start, end, self.closed[last], began, done,
                      res["lang"], res["text"], dropped)
        self.gpu_free = self.worker_free = done
        self.finals.append(final)
        self.on_final(final)

    def maybe_partial(self) -> None:
        """Make a live update of the segment in progress if the server would now (maybe_partial)."""
        seg = self.seg
        if self.gpu_free > self.now + EPS or self.carry or not self.queue.empty() or not seg.speaking:
            return
        if seg.frames_seen - self.last_frame < self.partial_every:
            return
        audio = seg.current_audio()
        if audio is None or len(audio) < server.PARTIAL_MIN_AUDIO_S * SR:
            return
        self.last_frame = seg.frames_seen
        t0 = self.timer()
        text = self.partial_text(audio, seg.index)
        p = Partial(seg.index, self.now, self.now + self.timer() - t0, text)
        self.gpu_free = p.done
        self.partials.append(p)
        self.on_partial(p)

    def partial_text(self, audio: np.ndarray, index: int) -> str:
        """server's partial_job: the language detected once per segment, on enough audio."""
        lang = self.language
        if not is_fixed(lang):
            if self.lang_index == index:
                lang = self.lang
            else:
                lang, _ = self.tr.pick_language(audio, detect_among(lang))
                if len(audio) >= server.PARTIAL_LANG_LOCK_S * SR:
                    self.lang_index, self.lang = index, lang
        if self.alignatt:
            state = self.states.setdefault(index, StreamState())
            return self.tr.transcribe_partial(audio, lang, self.prompt, state)
        return self.tr.transcribe_partial(audio, lang, self.prompt)

    def finish(self) -> None:
        """Mark the partials the page would have shown: those with text whose segment wasn't
        dropped as noise before they were ready (server: run_partial)."""
        for p in self.partials:
            gone = self.gone.get(p.index)
            p.shown = bool(p.text) and (gone is None or gone > p.done + EPS)


# ---------------------------------------------------------------------------------- scoring

_TOKEN = re.compile(r"[^\W_]+")


def words(text: str) -> list[str]:
    """The words of a text as they are scored: case, punctuation and apostrophes don't count
    ("L'anno," is "l anno"), nor do different encodings of the same letter (NFKC)."""
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


def edit_ops(ref: list, hyp: list) -> tuple[int, int, int]:
    """(substitutions, deletions, insertions) of a shortest edit from `ref` to `hyp`.

    The Levenshtein table row by row with numpy, keeping only the current row, with the counts
    of each kind of edit on the way to each cell: a whole lecture is some 10 000 words."""
    if not ref or not hyp:
        return 0, len(ref), len(hyp)
    ids: dict = {}
    r = [ids.setdefault(x, len(ids)) for x in ref]
    h = np.array([ids.setdefault(x, len(ids)) for x in hyp], dtype=np.int64)
    col = np.arange(len(h) + 1, dtype=np.int64)
    # Row 0: hyp[:j] out of nothing, j insertions.
    dist, sub, dele, ins = col.copy(), np.zeros_like(col), np.zeros_like(col), col.copy()
    for i, tok in enumerate(r, 1):
        cost = (h != tok).astype(np.int64)
        # Into cell j from above (delete ref[i-1]) or diagonally (keep or replace it with hyp[j-1]).
        diag, up = dist[:-1] + cost, dist[1:] + 1
        d = diag <= up
        nd = np.concatenate(([i], np.where(d, diag, up)))
        ns = np.concatenate(([0], np.where(d, sub[:-1] + cost, sub[1:])))
        nde = np.concatenate(([i], np.where(d, dele[:-1], dele[1:] + 1)))
        ni = np.concatenate(([0], np.where(d, ins[:-1], ins[1:])))
        # Or from the left, inserting hyp[k:j]: the best over k <= j of nd[k] + (j - k), from the
        # running minimum of nd[k] - k; k is the last place that minimum was reached.
        v = nd - col
        best = np.minimum.accumulate(v)
        k = np.maximum.accumulate(np.where(v == best, col, 0))
        dist = best + col
        sub, dele, ins = ns[k], nde[k], ni[k] + col - k
    return int(sub[-1]), int(dele[-1]), int(ins[-1])


def score(ref_text: str, hyp_text: str) -> dict | None:
    """WER (over words) and CER (over the characters of the words, single-spaced) of a result."""
    ref, hyp = words(ref_text), words(hyp_text)
    if not ref:
        return None
    s, d, i = edit_ops(ref, hyp)
    ref_chars, hyp_chars = " ".join(ref), " ".join(hyp)
    c = sum(edit_ops(list(ref_chars), list(hyp_chars)))
    return {
        "ref_words": len(ref), "hyp_words": len(hyp), "sub": s, "del": d, "ins": i,
        "wer": (s + d + i) / len(ref), "ref_chars": len(ref_chars), "cer": c / len(ref_chars),
    }


def rewritten(before: str, after: str) -> int:
    """How many words of `before` (shown) are no longer there in `after`: 0 when it only adds."""
    a, b = words(before), words(after)
    keep = 0
    while keep < min(len(a), len(b)) and a[keep] == b[keep]:
        keep += 1
    return len(a) - keep


def stats(values: list[float]) -> dict | None:
    if not values:
        return None
    v = np.asarray(values, dtype=float)
    return {"n": len(v), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def reason_key(reason: str) -> str:
    """What a drop counts under: the kind of check, with the phrase for a hallucination phrase."""
    kind, _, _ = reason.partition(": ")
    return reason if kind == "phrase" else kind


def max_backlog(finals: list[Final], closed: dict[int, float]) -> int:
    """Most segments closed and not yet answered at one time (the page's 待识别 N 段)."""
    events = sorted([(closed[i], 1) for f in finals for i in f.indices]
                    + [(f.done, -len(f.indices)) for f in finals])  # at a tie, the answer first
    n = peak = 0
    for _, step in events:
        n += step
        peak = max(peak, n)
    return peak


def summarize(rp: Replay, audio_s: float) -> dict:
    finals, partials = rp.finals, rp.partials
    live: dict[int, list[Partial]] = {}
    for p in partials:
        if p.shown:
            live.setdefault(p.index, []).append(p)
    rewrites = rewritten_words = 0
    for shown in live.values():
        for a, b in zip(shown, shown[1:]):
            n = rewritten(a.text, b.text)
            rewrites += n > 0
            rewritten_words += n
    # The final text against the live text it replaces (merged batches have no single one).
    replaced = [(live[f.indices[0]][-1].text, f.text) for f in finals if len(f.indices) == 1 and f.indices[0] in live]
    busy_final = sum(f.done - f.began for f in finals)
    busy_partial = sum(p.cost for p in partials)
    drops = [why for f in finals for _, why in f.dropped]
    per = (lambda x: x / audio_s) if audio_s else (lambda x: 0.0)
    return {
        "segments": sum(len(f.indices) for f in finals),
        "merged_batches": sum(len(f.indices) > 1 for f in finals),
        "noise_dropped": len(rp.gone),
        "latency": stats([f.latency for f in finals]),
        "vad_wait": stats([f.closed - f.end for f in finals]),
        "queue_wait": stats([f.began - f.closed for f in finals]),
        "asr": stats([f.done - f.began for f in finals]),
        "partials": len(partials),
        "partials_shown": sum(len(v) for v in live.values()),
        "partial_cost": stats([p.cost for p in partials]),
        "rewrites": rewrites,
        "rewritten_words": rewritten_words,
        "final_changed_live": sum(rewritten(shown, text) > 0 for shown, text in replaced),
        "segments_with_live": len(replaced),
        "first_text": stats([v[0].done - rp.starts[i] for i, v in live.items() if i in rp.starts]),
        "gpu_busy": per(busy_final + busy_partial),
        "gpu_final": per(busy_final),
        "gpu_partial": per(busy_partial),
        "max_backlog": max_backlog(finals, rp.closed),
        "filtered": len(drops),
        "filtered_segments": sum(bool(f.dropped) and not f.text for f in finals),
        "filtered_by_reason": dict(Counter(reason_key(why) for why in drops).most_common()),
    }


# ---------------------------------------------------------------------------------- report

REASONS = {  # asr.clean_text_why / unreliable_why / echoes_context, as the report names them
    "phrase": "幻觉短语",
    "url only": "只有网址",
    "filler": "重复的填充词",
    "stock phrase": "只有套话",
    "no words": "只有标点",
    "hint echo": "复读识别提示语",
    "glossary echo": "复读术语表",
    "context echo": "复读上一句",
    "repeats": "重复循环",
    "unsure": "可信度低",
    "no speech": "可能不是语音",
}


def reason_label(reason: str) -> str:
    kind, _, detail = reason.partition(": ")
    label = REASONS.get(kind, kind)
    if kind == "phrase":
        return f"{label}「{detail}」"
    return f"{label}（{detail}）" if detail else label


def clock(t: float) -> str:
    tenths = round(t * 10)
    s, d = divmod(tenths, 10)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}.{d}" if h else f"{m:02d}:{s:02d}.{d}"


def fmt(st: dict | None) -> str:
    if not st:
        return "无"
    return f"平均 {st['mean']:.2f}s，中位 {st['p50']:.2f}s，p90 {st['p90']:.2f}s，最长 {st['max']:.2f}s"


def print_final(f: Final) -> None:
    merged = f" 合并{len(f.indices)}段" if len(f.indices) > 1 else ""
    head = f"[{clock(f.start)}-{clock(f.end)} +{f.latency:.2f}s{merged}] {f.lang or '--'}  "
    lines = [f.text] if f.text else [] if f.dropped else ["（没有识别出文字）"]
    tag = "部分过滤" if f.text else "已过滤"
    lines += [f"（{tag}：{reason_label(why)}）{text}" for text, why in f.dropped]
    print(head + lines[0])
    for line in lines[1:]:
        print("    " + line)


def print_partial(p: Partial) -> None:
    if p.text:
        print(f"    ~{clock(p.done)} ({p.cost:.2f}s) {p.text}")


def print_summary(s: dict, sc: dict | None, no_score: str, audio_s: float, partial: bool) -> None:
    """`no_score`: what to say instead of the WER / CER when `sc` is None."""
    print("\n== 汇总 ==")
    merged = f"（追赶时合并识别 {s['merged_batches']} 次）" if s["merged_batches"] else ""
    print(f"音频 {clock(audio_s)}，识别 {s['segments']} 段{merged}，断句时当作噪音丢掉的短段 {s['noise_dropped']} 个")
    if sc:
        print(f"WER {sc['wer']:.1%}（参考 {sc['ref_words']} 词，识别 {sc['hyp_words']} 词："
              f"替换 {sc['sub']}，删除 {sc['del']}，插入 {sc['ins']}），CER {sc['cer']:.1%}")
    else:
        print(no_score)
    print(f"段结束 → 最终结果：{fmt(s['latency'])}")
    if s["latency"]:
        print(f"  平均等句末停顿 {s['vad_wait']['mean']:.2f}s，排队 {s['queue_wait']['mean']:.2f}s，"
              f"识别 {s['asr']['mean']:.2f}s")
    if partial:
        print(f"实时显示：更新 {s['partials_shown']} 次（调用模型 {s['partials']} 次），每次耗时 {fmt(s['partial_cost'])}")
        print(f"  改写 {s['rewrites']} 次，改掉已显示的词 {s['rewritten_words']} 个；"
              f"最终结果改动了实时文字的段 {s['final_changed_live']} / {s['segments_with_live']}")
        print(f"  每段第一次出字（距这段音频开始）：{fmt(s['first_text'])}")
    print(f"显卡忙碌 {s['gpu_busy']:.0%}（最终识别 {s['gpu_final']:.0%}，实时显示 {s['gpu_partial']:.0%}），"
          f"最多积压 {s['max_backlog']} 段")
    if s["filtered"]:
        reasons = "，".join(f"{reason_label(k)} {n}" for k, n in s["filtered_by_reason"].items())
        print(f"被幻觉过滤丢掉的文字 {s['filtered']} 处（其中整段丢掉 {s['filtered_segments']} 段）：{reasons}")
    else:
        print("幻觉过滤没有丢掉文字")


def rounded(x):
    """For JSON: times to the millisecond."""
    if isinstance(x, float):
        return round(x, 3)
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v) for v in x]
    return x


# ---------------------------------------------------------------------------------- command line


def read_wav(path: Path) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as w:
            fmt_ = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            data = w.readframes(w.getnframes())
    except (OSError, EOFError, wave.Error) as e:
        raise SystemExit(f"{path}：读不了这个 WAV（{e}）") from e
    if fmt_ != (1, 2, SR):
        raise SystemExit(
            f"{path}：需要 16 kHz、单声道、16 位的 WAV（LIVETRANS_RECORD=1 录下的就是），这个文件是 "
            f"{fmt_[2]} Hz、{fmt_[0]} 声道、{8 * fmt_[1]} 位。可以先转换：ffmpeg -i 原文件 -ac 1 -ar 16000 -c:a pcm_s16le 新文件.wav"
        )
    return np.frombuffer(data, dtype="<i2")


def load_transcriber(model: str, device: str, compute: str) -> Transcriber:
    tr = Transcriber()
    shown = [False]

    def progress() -> None:  # the model's first download
        if tr.progress:
            done, total, _ = tr.progress
            print(f"\r下载模型 {done / 2**20:.0f} / {total / 2**20:.0f} MB", end="", file=sys.stderr, flush=True)
            shown[0] = True
        elif shown[0]:
            print(file=sys.stderr)

    tr.on_progress = progress
    tr.load(model, device, compute)
    if tr.status != "ready":
        raise SystemExit(tr.error)
    if tr.error:  # loaded, with a warning (on the CPU after the GPU failed, ...)
        print(tr.error, file=sys.stderr)
    return tr


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="replay.py",
        description="把录好的课堂音频离线回放进本地识别管线（VAD 断句 → Whisper），量化识别质量和实时性。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("wav", type=Path, help="16 kHz 单声道 16 位 WAV，如 recordings/ 里的录音")
    p.add_argument("--ref", type=Path, help="参考文本：整段录音的逐字稿（UTF-8 纯文本），用来算 WER / CER")
    p.add_argument("--json", type=Path, help="把全部结果另存为 JSON，方便比较两次回放")
    p.add_argument("--model", default=server.MODEL, help="Whisper 模型，alignatt-large-v3-turbo 等为 AlignAtt 模式")
    p.add_argument("--device", default=server.DEVICE, choices=("auto", "cuda", "cpu"))
    p.add_argument("--compute", default=server.COMPUTE_TYPE, help="计算精度：auto / float16 / int8_float16 / int8")
    p.add_argument("--language", default=IT_EN, help="识别语言：it_en（意/英）、any（所有语言）、it、en 等")
    p.add_argument("--prompt", default="", help="识别提示，格式同页面：「课程提示语. 术语, 术语」")
    p.add_argument("--no-partial", dest="partial", action="store_false",
                   help="不做实时显示（快很多，只看最终结果）")
    p.add_argument("--no-context", dest="context", action="store_false", help="不把上一句作为上下文")
    p.add_argument("--show-partials", action="store_true", help="同时打印每次实时显示的文字")
    vad = p.add_argument_group("断句参数（vad.SegmenterParams）")
    for f in dataclasses.fields(SegmenterParams):
        vad.add_argument("--" + f.name.replace("_", "-"), dest="vad_" + f.name, type=type(f.default),
                         default=f.default, metavar=type(f.default).__name__.upper())
    return p.parse_args(argv)


def main(argv: list[str] | None = None, load=load_transcriber) -> int:
    args = parse_args(argv)
    language = normalize_language(args.language)
    if language is None:
        raise SystemExit(f"不认识的识别语言：{args.language}（可用 it_en、any，或 it、en 等语言代码）")
    pcm = read_wav(args.wav)
    ref = args.ref.read_text(encoding="utf-8-sig") if args.ref else None  # before the long part
    params = SegmenterParams(**{f.name: getattr(args, "vad_" + f.name) for f in dataclasses.fields(SegmenterParams)})
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # a console code page without some letter
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)  # a line per call otherwise

    tr = load(args.model, args.device, args.compute)
    rp = Replay(tr, Segmenter(SileroModel(), params), language, args.prompt, partial=args.partial,
                context=args.context, on_partial=print_partial if args.show_partials else None,
                on_final=print_final)
    changed = [f"{f.name}={getattr(params, f.name)}" for f in dataclasses.fields(SegmenterParams)
               if getattr(params, f.name) != f.default]
    print(f"回放 {args.wav}（{clock(len(pcm) / SR)}）")
    print(f"模型 {getattr(tr, 'model_name', args.model)}（{getattr(tr, 'device', '')} / {getattr(tr, 'compute_type', '')}），"
          f"识别语言 {language}，实时显示{'开' if args.partial else '关'}，上下文{'开' if args.context else '关'}"
          + (f"，提示：{args.prompt}" if args.prompt else ""))
    print("断句参数：" + ("，".join(changed) if changed else "默认") + "\n")

    t0 = time.perf_counter()
    interrupted = False
    try:
        rp.run(pcm)
    except KeyboardInterrupt:
        interrupted = True
        rp.finish()
    audio_s = rp.now if interrupted else len(pcm) / SR
    summary = summarize(rp, audio_s)
    sc, no_score = None, "没有参考文本：加 --ref <逐字稿.txt> 可以算 WER / CER"
    if ref is not None and interrupted:  # the reference covers the whole recording
        no_score = "回放中断了，不计算 WER / CER"
    elif ref is not None:
        sc = score(ref, " ".join(f.text for f in rp.finals if f.text))
        no_score = "参考文本里没有词，不计算 WER / CER"
    print_summary(summary, sc, no_score, audio_s, args.partial)
    print(f"回放用时 {time.perf_counter() - t0:.0f}s" + (f"（Ctrl+C 中断，只统计了前 {clock(audio_s)}）" if interrupted else ""))

    if args.json:
        data = {
            "wav": str(args.wav), "ref": str(args.ref) if args.ref else None, "audio_s": audio_s,
            "interrupted": interrupted, "model": getattr(tr, "model_name", args.model),
            "device": getattr(tr, "device", ""), "compute_type": getattr(tr, "compute_type", ""),
            "language": language, "prompt": args.prompt, "partial": args.partial, "context": args.context,
            "vad": dataclasses.asdict(params), "score": sc, "summary": summary,
            "finals": [{**dataclasses.asdict(f), "latency": f.latency,
                        "dropped": [{"text": t, "reason": why} for t, why in f.dropped]} for f in rp.finals],
            "partials": [{**dataclasses.asdict(p), "cost": p.cost} for p in rp.partials],
            "noise": [{"index": i, "at": t} for i, t in rp.gone.items()],
        }
        args.json.write_text(json.dumps(rounded(data), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"结果已保存到 {args.json}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
