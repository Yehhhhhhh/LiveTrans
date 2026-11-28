"""LiveTrans backend: audio over WebSocket -> VAD -> Whisper; translation proxy; static frontend."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import subprocess
import threading
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

from asr import setup_cuda_dlls

setup_cuda_dlls()  # must run before ctranslate2 is imported

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.websockets import WebSocketState  # noqa: E402

import alignatt  # noqa: E402
import assembly  # noqa: E402
from asr import IT_EN, Transcriber, all_languages, detect_among, is_fixed, normalize_language  # noqa: E402
from guard import LocalOnlyMiddleware  # noqa: E402
from translate import DEFAULT_DOMAIN, TranslateError, Translator, clean_output, clean_partial  # noqa: E402
from vad import FRAME, Segment, Segmenter, SegmenterParams, SileroModel  # noqa: E402
from winfo import window_info  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

HOST = os.environ.get("LIVETRANS_HOST", "127.0.0.1")
PORT = int(os.environ.get("LIVETRANS_PORT", "8765"))
MODEL = os.environ.get("LIVETRANS_MODEL", "large-v3-turbo")
DEVICE = os.environ.get("LIVETRANS_DEVICE", "auto")
COMPUTE_TYPE = os.environ.get("LIVETRANS_COMPUTE_TYPE", "auto")
# Whisper models the page may choose in 设置 → 识别. Only these names are ever loaded: the name comes
# from the page, and a free one could point faster-whisper at any repository or local path.
ASR_MODELS = tuple(dict.fromkeys((MODEL, "large-v3-turbo", "large-v3")))
# The experimental AlignAtt engine (alignatt.py), offered once its packages are installed.
ALIGNATT_MODELS = tuple(alignatt.PREFIX + m for m in ("large-v3-turbo", "large-v3"))
if alignatt.available():
    ASR_MODELS += ALIGNATT_MODELS
# Compute types (weight precision) it may choose. "auto" is float16 on a GPU and int8 on the CPU;
# the int8 ones roughly halve the VRAM the weights take, at a small cost in accuracy.
COMPUTE_TYPES = tuple(dict.fromkeys((COMPUTE_TYPE, "auto", "float16", "int8_float16", "int8")))
# Set LIVETRANS_RECORD=1 to save each capture's raw 16 kHz audio to recordings/ (for tuning).
RECORD = os.environ.get("LIVETRANS_RECORD", "") == "1"
# Host names (besides localhost and IP addresses) the page may be opened under, comma-separated,
# e.g. this PC's network name when LIVETRANS_HOST=0.0.0.0. See guard.py.
ALLOWED_HOSTS = frozenset(os.environ.get("LIVETRANS_ALLOWED_HOSTS", "").split(","))
RECORDINGS = ROOT / "recordings"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("livetrans")

transcriber = Transcriber()
translator: Translator | None = None
vad_model: SileroModel | None = None
gpu_lock = asyncio.Lock()


async def to_thread_to_end(fn, *args):
    """asyncio.to_thread for Whisper calls made under gpu_lock.

    A thread can't be cancelled. If the awaiting task is (the page disconnected), keep waiting
    until the thread has really finished before passing the cancellation on, so the caller's
    `async with gpu_lock` holds the lock until then and the next call can't overlap it on the GPU.
    """
    fut = asyncio.ensure_future(asyncio.to_thread(fn, *args))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        await asyncio.wait({fut})
        raise


clients: set[WebSocket] = set()
loop: asyncio.AbstractEventLoop | None = None


def download_progress() -> dict | None:
    """The model download in progress, for the page's progress bar; None when there is none."""
    progress = getattr(transcriber, "progress", None)
    if not progress or transcriber.status != "loading":
        return None
    done, total, rate = progress
    return {"done": done, "total": total, "rate": round(rate) if rate else None}


def status_message() -> dict:
    return {
        "type": "status",
        "model": transcriber.status,
        "model_name": transcriber.model_name or MODEL,
        "models": list(ASR_MODELS),
        "default_model": MODEL,
        "compute_types": list(COMPUTE_TYPES),
        "default_compute": COMPUTE_TYPE,
        "download": download_progress(),
        "device": transcriber.device,
        "compute_type": transcriber.compute_type,
        "error": transcriber.error,
    }


async def broadcast_status() -> None:
    msg = status_message()
    for ws in list(clients):
        try:
            await ws.send_json(msg)
        except Exception:  # noqa: BLE001 - client went away
            clients.discard(ws)


DEFAULT_CHOICE = (MODEL, COMPUTE_TYPE)
# The (model, compute type) the page asked for (设置 → 识别 → 模型). It can differ from the loaded
# one while a load is running: the choice is kept and applied when that load ends (ensure_model).
wanted_choice = [DEFAULT_CHOICE]
# The choice that last failed to load. It is not tried again until the page asks for another
# one: the page resends its settings often, and each try would stall recognition for a while.
failed_choice: list[tuple[str, str] | None] = [None]
switch_tasks: set[asyncio.Task] = set()


def load_model_in_background() -> None:
    choice = wanted_choice[0]

    def run() -> None:
        transcriber.load(choice[0], DEVICE, choice[1])
        if loop:
            asyncio.run_coroutine_threadsafe(after_model_load(choice), loop)

    transcriber.status = "loading"
    threading.Thread(target=run, daemon=True).start()


async def need_whisper() -> None:
    """The local engine is chosen or about to be used: load the Whisper model if nothing has yet.

    It is not loaded when the backend starts: with the AssemblyAI engine (the page's default) it
    would be downloaded (1.6 GB) and take about 4 GB of VRAM without ever being used."""
    if transcriber.status != "idle":
        return
    load_model_in_background()
    await broadcast_status()


def request_model(model: object, compute: object) -> None:
    """The page chose a model and/or a compute type; "" is the server's default (LIVETRANS_MODEL,
    LIVETRANS_COMPUTE_TYPE), None leaves that one as it is. Names not on the lists are ignored."""
    new_model, new_compute = wanted_choice[0]
    if isinstance(model, str) and (model or MODEL) in ASR_MODELS:
        new_model = model or MODEL
    if isinstance(compute, str) and (compute or COMPUTE_TYPE) in COMPUTE_TYPES:
        new_compute = compute or COMPUTE_TYPE
    wanted_choice[0] = (new_model, new_compute)
    ensure_model()


def ensure_model() -> None:
    """Start loading the wanted choice unless it is loaded, just failed, or a load is running."""
    want = wanted_choice[0]
    if transcriber.status in ("idle", "loading") or want in (transcriber.choice, failed_choice[0]):
        return
    transcriber.status = "loading"  # from here on finals wait (wait_for_model) and partials stop
    t = asyncio.create_task(switch_model(want))
    switch_tasks.add(t)
    t.add_done_callback(switch_tasks.discard)


def reload_model(choice: tuple[str, str]) -> None:
    """Replace the loaded model; if the new one fails, go back to the one that was loaded."""
    previous = transcriber.choice if transcriber.model is not None else None
    # Free the old model first: two large models don't fit in the VRAM of an 8 GB card.
    transcriber.model = None
    gc.collect()
    transcriber.load(choice[0], DEVICE, choice[1], fallback=previous)


async def switch_model(choice: tuple[str, str]) -> None:
    await broadcast_status()
    # Under gpu_lock: a transcription that is running finishes on the old model first.
    async with gpu_lock:
        await to_thread_to_end(reload_model, choice)
    await after_model_load(choice)


async def after_model_load(choice: tuple[str, str]) -> None:
    failed_choice[0] = None if transcriber.choice == choice else choice
    await broadcast_status()
    ensure_model()  # the page may have chosen something else meanwhile


def gpu_memory() -> tuple[int, int] | None:
    """(used, total) MiB of the first NVIDIA GPU as nvidia-smi reports it; None without one."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
        used, total = (int(x) for x in out.splitlines()[0].split(","))
        return used, total
    except Exception:  # noqa: BLE001 - no NVIDIA GPU / driver, or output we can't read
        return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global translator, vad_model, loop
    loop = asyncio.get_running_loop()
    # Called from the thread that loads the model, as its download advances.
    transcriber.on_progress = lambda: asyncio.run_coroutine_threadsafe(broadcast_status(), loop)
    vad_model = SileroModel()
    translator = Translator()
    # The Whisper model is loaded when a page asks for the local engine (need_whisper).
    log.info("Open http://%s:%d in Chrome", HOST, PORT)
    yield
    await translator.close()


app = FastAPI(lifespan=lifespan)
# Other websites open in the browser must not reach /ws or the API (see guard.py).
app.add_middleware(LocalOnlyMiddleware, allowed_hosts=ALLOWED_HOSTS)


@app.get("/api/gpu")
def api_gpu() -> dict:
    """VRAM in use on the whole GPU (all programs), for 设置 → 识别 → 模型. Empty without one."""
    mem = gpu_memory()
    return {"used_mb": mem[0], "total_mb": mem[1]} if mem else {}


@app.get("/api/languages")
async def api_languages() -> dict:
    """Language codes the recognition-language setting may choose from ("其他")."""
    return {"languages": list(all_languages())}


@app.get("/api/status")
async def api_status() -> dict:
    return status_message()


@app.get("/api/window")
def api_window(hwnd: int) -> dict:
    """Name of a shared window: Chrome's window-capture track label carries its native handle."""
    # Not async: the Windows calls block (asking a window for its icon waits up to 200 ms per
    # try), so this runs in a worker thread and the audio WebSocket keeps being served meanwhile.
    info = window_info(hwnd)
    if info is None:
        raise HTTPException(404, "window not found")
    return info


class TranslateRequest(BaseModel):
    provider: str
    config: dict
    text: str
    context: list[dict] = []
    glossary: list[dict] = []
    domain: str = DEFAULT_DOMAIN  # course domain for the system prompt; "" = unspecified


@app.post("/api/translate")
async def api_translate(req: TranslateRequest) -> dict:
    try:
        out = await translator.translate(
            req.provider, req.config, req.text, req.context, req.glossary, req.domain
        )
    except TranslateError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"translation": out}


class WarmupRequest(BaseModel):
    config: dict
    reload: bool = False  # unload the model first (see Translator.warmup)


@app.post("/api/translate/warmup")
async def api_translate_warmup(req: WarmupRequest) -> dict:
    try:
        await translator.warmup(req.config, req.reload)
    except TranslateError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"ok": True}


@app.post("/api/translate/stream")
async def api_translate_stream(req: TranslateRequest) -> StreamingResponse:
    """NDJSON: {"text": cleaned translation so far} per update, then {"done": true, "translation"} or {"error"}."""

    async def gen():
        raw, shown = "", ""
        try:
            async for delta in translator.translate_stream(
                req.provider, req.config, req.text, req.context, req.glossary, req.domain
            ):
                raw += delta
                text = clean_partial(raw)
                if text != shown:
                    shown = text
                    yield json.dumps({"text": text}, ensure_ascii=False) + "\n"
        except TranslateError as e:
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"
            return
        yield json.dumps({"done": True, "translation": clean_output(raw)}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def apply_config(seg: Segmenter, asr: dict) -> None:
    p = seg.params
    if "vad_threshold" in asr:
        p.threshold = min(0.95, max(0.1, float(asr["vad_threshold"])))
    if "min_silence_ms" in asr:
        p.min_silence_ms = int(min(3000, max(150, float(asr["min_silence_ms"]))))
    if "max_segment_s" in asr:
        p.max_segment_s = min(28.0, max(2.0, float(asr["max_segment_s"])))


class BadMessage(ValueError):
    """A text message from the page that can't be used. It is logged and ignored: ending the
    connection over it would not help, as the page reconnects and sends the same settings again."""


# The VAD settings of a "config" message, which apply_config takes with float().
VAD_SETTINGS = ("vad_threshold", "min_silence_ms", "max_segment_s")


def converts(to: type, value: object) -> bool:
    """Whether to(value) works: float("0.5") does, float(None) and int("1.5") don't."""
    try:
        to(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def read_message(text: str) -> dict:
    """A text message from the page, checked before any of it is applied, so that a bad value
    can't stop its handler half-way. Raises BadMessage. Everything the handlers ignore or fall
    back from (unknown types and fields, a model name that isn't one, ...) is let through."""
    try:
        data = json.loads(text)
    except (ValueError, RecursionError) as e:
        raise BadMessage(f"not JSON ({e})") from None
    if not isinstance(data, dict):
        raise BadMessage("not a JSON object")
    kind = data.get("type")
    if kind == "config":
        asr = data.get("asr") or {}
        if not isinstance(asr, dict):
            raise BadMessage("config: asr is not an object")
        for name in VAD_SETTINGS:
            if name in asr and not converts(float, asr[name]):
                raise BadMessage(f"config: asr.{name} is not a number")
        if not isinstance(data.get("prompt") or "", str):
            raise BadMessage("config: prompt is not a string")
    elif kind == "resume" and not converts(int, data.get("offset") or 0):
        raise BadMessage("resume: offset is not a whole number")
    return data


# Live partials: re-transcribe the segment in progress at most this often.
PARTIAL_INTERVAL_S = 0.6
PARTIAL_MIN_AUDIO_S = 0.8
# Below this length, language detection is unreliable, so it isn't cached yet.
PARTIAL_LANG_LOCK_S = 2.0
# How often the live speech probability is sent to the status bar.
VAD_LEVEL_INTERVAL_S = 0.25
# Catching up: when the final transcriptions fall behind the speaker (slow GPU / CPU), segments
# that queued up are transcribed together in one pass. Whisper's encoder costs about the same for
# 1 s or 28 s of audio and dominates the time (beam size barely matters), so 3 segments merged
# take ~1.3x one segment instead of 3x. Only done once MERGE_BACKLOG segments wait behind the one
# about to be transcribed, because a merged pass detects the language once (auto mode) and loses
# the per-segment split.
MERGE_BACKLOG = 2
MERGE_MAX_S = 24.0  # stay well inside Whisper's 30 s window
MERGE_GAP_S = 0.3  # silence between merged segments, so Whisper sees the pause
# While the Whisper model is still loading, queued speech waits; status is checked this often.
MODEL_WAIT_POLL_S = 0.2
# An AssemblyAI connection opened when 开始 was pressed is closed again if no capture has started
# after this long (the window picker was left open): an open session is billed.
SPARE_KEEP_S = 60.0
# A failed final transcription is retried this many times in all, this long apart, before its
# audio is given up (and stops being kept for resending, see AudioSession.committed).
TRANSCRIBE_ATTEMPTS = 3
TRANSCRIBE_RETRY_S = 1.0
# Context: the end of what was just recognized is given to Whisper with the next segment (see
# asr.fit_prompt), so a sentence cut in two, a name or a term is decoded consistently. Only this
# much of it, and only if the next segment starts within this long: after a longer pause the
# lecturer has usually moved on, and old text would only bias the decoding.
CONTEXT_CHARS = 200
CONTEXT_MAX_GAP_S = 15.0


def context_tail(text: str) -> str:
    """The last CONTEXT_CHARS characters of the text, starting at a word where there is one."""
    text = " ".join(text.split())
    if len(text) <= CONTEXT_CHARS:
        return text
    tail = text[-CONTEXT_CHARS:]
    cut = tail.find(" ", 0, 40)
    # No space near the start (Chinese, Japanese… are written without): keep the tail as it is.
    return tail[cut + 1 :] if cut >= 0 else tail


def merge_segments(items: list[Segment]) -> Segment:
    """One segment holding the audio of consecutive segments; carries the last one's index."""
    if len(items) == 1:
        return items[0]
    gap = np.zeros(round(MERGE_GAP_S * 16000), dtype=np.float32)
    parts = []
    for i, s in enumerate(items):
        if i:
            parts.append(gap)
        parts.append(s.audio)
    return Segment(start=items[0].start, audio=np.concatenate(parts), index=items[-1].index, run=items[0].run)


def span_s(items: list[Segment]) -> float:
    """Seconds from the first segment's start to the end of the last one's audio: the real time
    the batch covers (merge_segments' audio has fixed short gaps instead of the real pauses)."""
    last = items[-1]
    return last.start + len(last.audio) / 16000 - items[0].start


def take_batch(first: Segment, queue: asyncio.Queue) -> tuple[list[Segment], Segment | None]:
    """The segments to transcribe next: just `first`, or, when MERGE_BACKLOG or more wait behind it,
    as many queued ones as fit in MERGE_MAX_S. A taken segment that doesn't fit, or that belongs to
    another capture run (开始 pressed again meanwhile), is returned as carry."""
    items = [first]
    if queue.qsize() < MERGE_BACKLOG:
        return items, None
    total = len(first.audio)
    while not queue.empty():
        nxt = queue.get_nowait()
        total += round(MERGE_GAP_S * 16000) + len(nxt.audio)
        if total > MERGE_MAX_S * 16000 or nxt.run != first.run:
            return items, nxt
        items.append(nxt)
    return items, None


def open_recording() -> wave.Wave_write:
    RECORDINGS.mkdir(exist_ok=True)
    path = RECORDINGS / time.strftime("%Y%m%d-%H%M%S.wav")
    w = wave.open(str(path), "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(16000)
    log.info("Recording raw audio to %s", path)
    return w


def alignatt_on() -> bool:
    """The loaded model is the AlignAtt engine's (see alignatt.py)."""
    return getattr(transcriber, "engine", None) is not None


class AudioSession:
    """One /ws connection: the page's audio and settings come in, its subtitles go out.

    Each capture run (开始 … 停止) goes to one recognition engine. Whisper: the VAD (Segmenter)
    cuts the audio into segments, `worker` transcribes them in order, and `maybe_partial` shows
    the segment in progress meanwhile. AssemblyAI: the audio goes to the run's stream, and
    `on_stream` passes its events on as the same messages. ws_audio hands it what the page sends.
    """

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.seg = Segmenter(vad_model, SegmenterParams())

        # The page's settings ("config"). The engine, its API key and its model are applied by
        # the next "start" / "resume".
        self.language = IT_EN
        self.prompt = ""
        self.use_partials = True
        self.use_context = True
        self.engine = "local"  # "local" (Whisper) or "assemblyai" (see assembly.py)
        self.key = ""
        self.ai_model = assembly.DEFAULT_MODEL

        # Capture run, set by every "start" (the page numbers its runs); tags queued segments (see
        # take_batch) and is echoed in their results, so a run that is still being transcribed
        # after 开始 was pressed again keeps its own time base on the page and can't touch the new
        # run's state.
        self.run = 0
        # run -> its AssemblyAI stream, for runs captured with that engine. `cloud`: the current
        # run is one of them, so its audio goes to the stream and not through the VAD to Whisper.
        self.streams: dict[int, assembly.AssemblyStream] = {}
        self.cloud = False
        # A stream opened ahead of its run ("prepare": the page is about to start capturing), so
        # that the connection is there when the audio begins. Audio that waits for the connection
        # is sent at the speed it was captured at, and the whole run's results would stay that far
        # behind. Its run is 0 until a "start" takes it. spare_state: its last reported state.
        self.spare: assembly.AssemblyStream | None = None
        self.spare_state = ""
        self.spare_timer: asyncio.TimerHandle | None = None

        # Resuming after a dropped connection: the page keeps the audio it sent and, on reconnect,
        # resends it from the last "committed" position, i.e. the point before which every segment
        # has been answered (or was silence). Positions are samples since the page pressed 开始.
        # run -> position of that run's first sample on this connection (set by "resume"). Per
        # run: a reconnecting page may resend several runs, and a run's segments can still be
        # transcribed after the next run began.
        self.bases: dict[int, int] = {}
        self.unsent: dict[tuple[int, int], int] = {}  # (run, index) -> start position, not answered yet

        # Segments waiting for their final transcription, and how many the worker has taken off
        # the queue whose result hasn't been sent yet.
        self.queue: asyncio.Queue = asyncio.Queue()
        self.in_hand = 0
        # What the last answered segment said, in which language, and when (seconds, on the same
        # clock as a result's "start") it ended. Used as context for the next segment of the same
        # run.
        self.ctx_text = ""
        self.ctx_lang = ""
        self.ctx_run = 0
        self.ctx_end = 0.0

        # Live partials: one at a time (partial_busy), every partial_every frames at most
        # (partial_frame: seg.frames_seen when the last one started), and none for a segment whose
        # final result was sent (final_sent: the highest index of the current run sent).
        self.partial_every = round(PARTIAL_INTERVAL_S * 16000 / 512)
        self.partial_busy = False
        self.partial_frame = 0
        self.final_sent = 0
        self.partial_tasks: set[asyncio.Task] = set()
        # Segments of the current run discarded as noise (too little speech). A partial of one may
        # still be running in its worker thread when the page is told to drop the line: its text
        # must not be sent afterwards, or the line would come back and stay ("识别中…") with
        # nothing to replace it until the next sentence.
        self.gone: set[int] = set()
        # AlignAtt engine only: (run, index) -> what its partials have shown of the segment in
        # progress, which its final transcription keeps as its start (see alignatt.py).
        self.shown: dict[tuple[int, int], alignatt.StreamState] = {}
        # The language detected for the segment in progress (detected_index), kept for its next
        # partials. mode_gen is bumped whenever the recognition language mode changes: a partial
        # that started under an older mode (it runs in a worker thread) neither caches the
        # language it detected nor shows its text. lang_lock keeps the mode, its generation and
        # the cache consistent across threads.
        self.detected_index = 0
        self.detected_lang = ""
        self.mode_gen = 0
        self.lang_lock = threading.Lock()

        # The status bar's speech indicator: what was sent last, and when.
        self.speaking = False
        self.last_level = 0.0
        self.rec: wave.Wave_write | None = None  # LIVETRANS_RECORD

        self.worker_task = asyncio.create_task(self.worker())

    def close(self) -> None:
        """The page went away: stop everything this connection started."""
        self.close_recording()
        self.worker_task.cancel()
        for t in self.partial_tasks:
            t.cancel()
        for stream in self.streams.values():
            stream.abort()
        if self.spare_timer:
            self.spare_timer.cancel()
        if self.spare:
            self.spare.abort()

    def close_recording(self) -> None:
        if self.rec is not None:
            self.rec.close()
            self.rec = None

    # ------------------------------------------------------------ the page's messages

    async def on_audio(self, pcm: bytes) -> None:
        """A chunk of the capture: 16 kHz mono PCM16."""
        if RECORD:
            if self.rec is None:
                self.rec = open_recording()
            self.rec.writeframes(pcm)
        new = self.seg.feed(pcm)
        if self.cloud:
            # AssemblyAI finds the sentences itself; the VAD only feeds the status bar.
            self.seg.take_dropped()
            if self.run in self.streams:
                self.streams[self.run].feed(pcm)
        else:
            await self.enqueue(new)
        now = time.monotonic()
        if self.seg.speaking != self.speaking or now - self.last_level >= VAD_LEVEL_INTERVAL_S:
            self.speaking = self.seg.speaking
            self.last_level = now
            await self.ws.send_json({
                "type": "vad",
                "speaking": self.speaking,
                "prob": round(self.seg.take_peak_prob(), 2),
                **self.commit_info(),
            })
        if not self.cloud:
            self.maybe_partial()

    async def on_text(self, text: str) -> None:
        """A JSON message. One that can't be used is logged and ignored (see read_message); one of
        a type not handled here is ignored without a word."""
        try:
            data = read_message(text)
        except BadMessage as e:
            # Not the message itself: a config message carries the AssemblyAI API key.
            log.warning("Ignored a message from the page: %s", e)
            return
        match data.get("type"):
            case "config":
                await self.on_config(data)
            case "prepare":
                await self.on_prepare(data)
            case "unprepare":
                self.on_unprepare()
            case "start":
                await self.on_start(data)
            case "resume":
                await self.on_start(data, resume=True)
            case "stop":
                await self.on_stop()

    async def on_config(self, data: dict) -> None:
        """The recognition settings (设置 → 识别), sent on connecting and after every change."""
        asr = data.get("asr") or {}
        apply_config(self.seg, asr)
        mode = normalize_language(asr.get("language"))
        if mode and mode != self.language:
            # The language detected for the segment in progress was chosen under the old mode:
            # detect again for its next partials, and drop partials still running under the old
            # mode (see partial_job).
            with self.lang_lock:
                self.language = mode
                self.mode_gen += 1
                self.detected_index, self.detected_lang = 0, ""
            self.ctx_text, self.ctx_lang = "", ""
        if "partial" in asr:
            self.use_partials = bool(asr["partial"])
        if "context" in asr:
            self.use_context = bool(asr["context"])
        if "model" in asr or "compute" in asr:
            request_model(asr.get("model"), asr.get("compute"))
        self.prompt = (data.get("prompt") or "")[:1000]
        if asr.get("engine") in ("local", "assemblyai"):
            self.engine = asr["engine"]
        if isinstance(asr.get("assemblyai_key"), str):
            self.key = asr["assemblyai_key"].strip()
        self.ai_model = assembly.normalize_model(asr.get("assemblyai_model")) or self.ai_model
        if self.engine == "local":
            await need_whisper()
        if self.run in self.streams:  # language or glossary changed while capturing
            self.streams[self.run].update(self.language, assembly.keyterms(self.prompt))

    async def on_prepare(self, data: dict) -> None:
        """开始 was pressed: the page now asks for its audio sources (a window picker, a permission
        prompt), and "start" follows once it has them. Connect to AssemblyAI meanwhile."""
        self.drop_spare()
        key, ai_model = self.assembly_choice(data)
        if data.get("engine") == "assemblyai" and key:
            self.spare = self.new_stream(key, ai_model)
            self.spare_timer = asyncio.get_running_loop().call_later(SPARE_KEEP_S, self.drop_spare)
            self.spare.start()

    def on_unprepare(self) -> None:
        """The page got no audio sources after all."""
        self.drop_spare()

    async def on_start(self, data: dict, resume: bool = False) -> None:
        """"start": a new capture run. "resume" (`resume`): a reconnecting page continues its run
        at `offset` and resends the audio from there; segment start times continue from that
        position."""
        old = self.run
        r = data.get("run")
        self.run = r if isinstance(r, int) and not isinstance(r, bool) else self.run + 1
        self.seg.reset()
        self.gone.clear()
        self.shown.clear()
        self.partial_frame = self.final_sent = self.detected_index = 0
        self.detected_lang = ""
        self.ctx_text, self.ctx_lang = "", ""
        self.bases[self.run] = max(0, int(data.get("offset") or 0)) if resume else 0
        if old != self.run and old in self.streams:
            self.streams[old].close()  # normally closed already, by "stop"
        if self.run in self.streams:
            self.streams.pop(self.run).abort()
        # The run's engine, key and model come with the message: the page fixes them when the run
        # starts and repeats them on "resume". The settings may have changed since (they apply
        # from the next 开始), and audio captured for the local engine must not be sent to
        # AssemblyAI when the run is resumed.
        engine = data.get("engine") if data.get("engine") in ("local", "assemblyai") else self.engine
        key, ai_model = self.assembly_choice(data)
        self.cloud = engine == "assemblyai"
        spare_state = self.spare_state
        if self.cloud and not resume:
            stream = self.take_spare(key, ai_model)
        else:
            stream = None
            self.drop_spare()
        if stream:
            # Connected (or connecting) since "prepare": the settings may have changed.
            stream.run = self.run
            self.streams[self.run] = stream
            stream.update(self.language, assembly.keyterms(self.prompt))
            await self.ws.send_json({
                "type": "engine", "state": spare_state or "connecting",
                "model": stream.model, "run": stream.run,
            })
        elif self.cloud:
            stream = self.new_stream(key, ai_model)
            stream.run = self.run
            self.streams[self.run] = stream
            stream.start()
        else:
            await need_whisper()
        if old and old != self.run:  # 0: no run yet on this connection
            await self.send_run_done(old)

    async def on_stop(self) -> None:
        """停止: the segment in progress is cut off and queued, or the run's AssemblyAI stream is
        closed."""
        self.close_recording()
        new = self.seg.flush()
        if self.cloud:
            new = []
            self.seg.take_dropped()
            if self.run in self.streams:
                self.streams[self.run].close()
        await self.enqueue(new)
        self.speaking = False
        await self.ws.send_json({"type": "vad", "speaking": False})

    # ------------------------------------------------------------ AssemblyAI

    def assembly_choice(self, data: dict) -> tuple[str, str]:
        """The API key and model a prepare / start / resume message names, else the settings'."""
        key = data.get("assemblyai_key")
        key = key.strip() if isinstance(key, str) else self.key
        return key, assembly.normalize_model(data.get("assemblyai_model")) or self.ai_model

    def new_stream(self, key: str, model: str) -> assembly.AssemblyStream:
        return assembly.AssemblyStream(
            key, model, self.language, assembly.keyterms(self.prompt), self.on_stream,
        )

    def drop_spare(self) -> None:
        s, timer = self.spare, self.spare_timer
        self.spare, self.spare_state, self.spare_timer = None, "", None
        if timer:
            timer.cancel()
        if s:
            s.close()  # ends the session properly: an open one is billed

    def take_spare(self, key: str, model: str) -> assembly.AssemblyStream | None:
        """The prepared stream, if it fits the run that starts now and is still good."""
        s, state = self.spare, self.spare_state
        if s and not s.done and not s.dead and state != "error" and (s.key, s.model) == (key, model):
            self.spare = None  # kept open: drop_spare only clears the rest
            self.drop_spare()
            return s
        self.drop_spare()
        return None

    async def on_stream(self, s: assembly.AssemblyStream, ev: dict) -> None:
        """An AssemblyAI stream's events, as the messages the page knows from the Whisper path."""
        current = s.run == self.run  # an older run's indices mean nothing for the current one
        kind = ev["kind"]
        if not s.run:
            # Opened ahead of its run and not taken (yet): the page hears of it when it is.
            if s is self.spare and kind == "state":
                self.spare_state = ev["state"]
            return
        if kind == "partial":
            if self.use_partials and current and ev["index"] > self.final_sent:
                out = {"type": "partial", "index": ev["index"], "text": ev["text"]}
                if ev.get("end") is not None:
                    out["audio_end"] = round(self.bases.get(s.run, 0) / 16000 + ev["end"], 3)
                await self.ws.send_json(out)
        elif kind == "drop":
            if current:
                await self.ws.send_json({"type": "partial_drop", "index": ev["index"]})
        elif kind == "final":
            if current:
                self.final_sent = max(self.final_sent, ev["index"])
            log.info("[%s %.2fs %s] %s", ev["lang"], ev["duration"], s.model, ev["text"])
            await self.ws.send_json({
                "type": "segment",
                "index": ev["index"],
                "start": round(self.bases.get(s.run, 0) / 16000 + ev["start"], 3),
                "duration": round(ev["duration"], 2),
                "lang": ev["lang"],
                "lang_prob": ev["lang_prob"],
                "text": ev["text"],
                "asr_ms": 0,
                "merged": 1,
                "run": s.run,
                **self.stream_commit(s),
            })
        elif kind == "state":
            await self.ws.send_json({"type": "engine", "state": ev["state"], "model": s.model, "run": s.run})
            if ev["state"] == "error":
                await self.ws.send_json({"type": "error", "message": ev["message"]})
        elif kind == "closed":
            # Everything is answered now: the page can drop the audio it kept for this run.
            await self.ws.send_json(
                {"type": "engine", "state": "closed", "model": s.model, "run": s.run, **self.stream_commit(s)}
            )
            await self.send_run_done(s.run)

    def stream_commit(self, s: assembly.AssemblyStream) -> dict:
        return {"committed": self.bases.get(s.run, 0) + s.committed, "committed_run": s.run}

    # ------------------------------------------------------------ resuming: what is answered

    def committed(self) -> int:
        if self.run in self.streams:
            return self.bases.get(self.run, 0) + self.streams[self.run].committed
        pos = self.seg.committed_frames() * FRAME
        mine = [p for (r, _), p in self.unsent.items() if r == self.run]
        if mine:
            pos = min(pos, min(mine))
        return self.bases.get(self.run, 0) + pos

    def commit_info(self) -> dict:
        # The run the position belongs to: a message sent before a "start" was processed may reach
        # the page after it started the new run, and must not move that run's resend cursor.
        return {"committed": self.committed(), "committed_run": self.run}

    async def send_run_done(self, r: int) -> None:
        # An ended run (开始 pressed again, or the next run resumed) whose segments are all
        # answered: the page can drop the audio it kept for it.
        if r == self.run or any(k[0] == r for k in self.unsent):
            return
        if r in self.streams and not self.streams[r].done:
            return  # its last turn is still coming (see on_stream, "closed")
        self.streams.pop(r, None)
        await self.ws.send_json({"type": "run_done", "run": r})

    # ------------------------------------------------------------ Whisper: final transcriptions

    async def enqueue(self, new: list[Segment]) -> None:
        """Queue the segments the VAD finished, and tell the page of those it dropped."""
        for s in new:
            s.run = self.run
            self.unsent[(s.run, s.index)] = round(s.start * 16000)
            self.queue.put_nowait(s)
        if new:
            await self.send_backlog()
        await self.send_dropped()

    async def send_backlog(self) -> None:
        # Segments waiting for (or in) their final transcription; the UI warns when it grows.
        await self.ws.send_json(
            {"type": "backlog", "pending": self.queue.qsize() + self.in_hand, **self.commit_info()}
        )

    async def send_dropped(self) -> None:
        for index in self.seg.take_dropped():
            self.gone.add(index)
            self.shown.pop((self.run, index), None)
            await self.ws.send_json({"type": "partial_drop", "index": index})

    async def worker(self) -> None:
        carry: Segment | None = None
        while True:
            try:
                first = carry or await self.queue.get()
                carry = None
                self.in_hand = 1
                await self.wait_for_model()
                items, carry = take_batch(first, self.queue)
                self.in_hand = len(items) + (carry is not None)
                try:
                    await self.transcribe_batch(items)
                    await self.send_run_done(items[0].run)
                finally:
                    self.in_hand = carry is not None
                await self.send_backlog()
            except Exception:  # noqa: BLE001 - one bad batch must not stop transcription for good
                if self.ws.client_state != WebSocketState.CONNECTED:
                    return  # the page went away; close() cancels this task anyway
                log.exception("ASR worker failed on a batch; continuing")
                carry, self.in_hand = None, 0

    async def wait_for_model(self) -> None:
        # The first launch downloads the model (~1.6 GB) while the page may already be capturing:
        # keep that speech queued and transcribe it (merged, see take_batch) once the model is
        # ready, instead of dropping it. The page shows the backlog meanwhile.
        if transcriber.status in ("idle", "loading"):
            await self.send_backlog()
            while transcriber.status in ("idle", "loading"):
                await asyncio.sleep(MODEL_WAIT_POLL_S)

    async def transcribe_batch(self, items: list[Segment]) -> None:
        if transcriber.status != "ready":
            self.answered(items)
            await self.ws.send_json({"type": "error", "message": "模型加载失败，这段语音无法识别"})
            return
        item = merge_segments(items)
        start = item.start + self.bases.get(item.run, 0) / 16000
        context = ("", "")
        if self.use_context and self.ctx_run == item.run and 0 <= start - self.ctx_end <= CONTEXT_MAX_GAP_S:
            context = (self.ctx_text, self.ctx_lang)
        # Answering a batch advances "committed", and the page then drops the kept audio. So a
        # failure (e.g. a transient CUDA error) is retried before the batch is given up.
        args = [item.audio, self.language, self.prompt, *context]
        states = [self.shown.pop((it.run, it.index), None) for it in items]
        if alignatt_on():
            # A merged batch is decoded afresh: its partials' words belong to one part of it.
            args.append(states[0] if len(items) == 1 else None)
        for attempt in range(1, TRANSCRIBE_ATTEMPTS + 1):
            try:
                async with gpu_lock:
                    res = await to_thread_to_end(transcriber.transcribe, *args)
                break
            except Exception as e:  # noqa: BLE001
                log.exception("Transcription failed (attempt %d of %d)", attempt, TRANSCRIBE_ATTEMPTS)
                if attempt == TRANSCRIBE_ATTEMPTS:
                    self.answered(items)
                    await self.ws.send_json({"type": "error", "message": f"识别出错：{e}"})
                    return
                await asyncio.sleep(TRANSCRIBE_RETRY_S)
        self.answered(items)
        for it in items:  # a partial may have started again after the state was taken
            self.shown.pop((it.run, it.index), None)
        if res["text"]:
            # Same language and still connected to it: keep the older text too, so a short
            # "Sì." doesn't leave the next segment with almost no context.
            joined = context[1] == res["lang"] and context[0]
            self.ctx_text = context_tail(f"{context[0]} {res['text']}" if joined else res["text"])
            self.ctx_lang = res["lang"]
            self.ctx_run = item.run
            self.ctx_end = start + span_s(items)
        res.update(
            type="segment",
            index=item.index,
            start=round(start, 3),
            # Sent with the result itself: if the connection drops right after, the page won't
            # resend (and show twice) the audio of a sentence it already has.
            **self.commit_info(),
            duration=round(span_s(items), 2),
            merged=len(items),
            run=item.run,
        )
        if item.run == self.run:  # an older run's indices mean nothing for the current one
            self.final_sent = max(self.final_sent, item.index)
        log.info(
            "[%s %.2fs %dms%s] %s",
            res["lang"], res["duration"], res["asr_ms"],
            f" merged {len(items)}" if len(items) > 1 else "", res["text"],
        )
        await self.ws.send_json(res)

    def answered(self, items: list[Segment]) -> None:
        for it in items:
            self.unsent.pop((it.run, it.index), None)

    # ------------------------------------------------------------ Whisper: live partials

    def maybe_partial(self) -> None:
        # Finals take priority: skip while one is queued or running.
        if not self.use_partials or self.partial_busy or transcriber.status != "ready":
            return
        if not self.queue.empty() or gpu_lock.locked() or not self.seg.speaking:
            return
        if self.seg.frames_seen - self.partial_frame < self.partial_every:
            return
        audio = self.seg.current_audio()
        if audio is None or len(audio) < PARTIAL_MIN_AUDIO_S * 16000:
            return
        self.partial_busy = True
        self.partial_frame = self.seg.frames_seen
        audio_end = round((self.bases.get(self.run, 0) + self.seg.frames_seen * FRAME) / 16000, 3)
        # The event loop only keeps weak references to tasks: hold one until it is done.
        t = asyncio.create_task(self.run_partial(audio, self.seg.index, audio_end))
        self.partial_tasks.add(t)
        t.add_done_callback(self.partial_tasks.discard)

    async def run_partial(self, audio, index: int, audio_end: float) -> None:
        # audio_end: the stream position (seconds) of the newest audio in `audio`; the page shows
        # how long after sending that audio its text came back.
        started_in = self.run  # 开始 pressed again meanwhile: the index means another segment now
        try:
            async with gpu_lock:
                if index <= self.final_sent:
                    return
                text = await to_thread_to_end(self.partial_job, audio, index, started_in)
            if text and index > self.final_sent and index not in self.gone and self.run == started_in:
                await self.ws.send_json({"type": "partial", "index": index, "text": text, "audio_end": audio_end})
        except Exception:  # noqa: BLE001 - a lost partial is harmless; the final pass follows
            log.debug("Partial transcription failed", exc_info=True)
        finally:
            self.partial_busy = False

    def partial_job(self, audio, index: int, started_in: int) -> str:
        """Runs in a worker thread (see lang_lock)."""
        with self.lang_lock:
            gen, lang = self.mode_gen, self.language
            cached = self.detected_lang if self.detected_index == index else ""
        if not is_fixed(lang):
            if cached:
                lang = cached
            else:
                lang, _ = transcriber.pick_language(audio, detect_among(lang))
                if len(audio) >= PARTIAL_LANG_LOCK_S * 16000:
                    with self.lang_lock:
                        if self.mode_gen == gen:
                            self.detected_index, self.detected_lang = index, lang
        if alignatt_on():
            state = self.shown.setdefault((started_in, index), alignatt.StreamState())
            text = transcriber.transcribe_partial(audio, lang, self.prompt, state)
        else:
            text = transcriber.transcribe_partial(audio, lang, self.prompt)
        with self.lang_lock:
            return text if self.mode_gen == gen else ""


@app.websocket("/ws")
async def ws_audio(ws: WebSocket) -> None:
    """The page's capture: binary messages are its audio, text messages JSON (see AudioSession)."""
    await ws.accept()
    clients.add(ws)
    await ws.send_json(status_message())
    session = AudioSession(ws)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                await session.on_audio(msg["bytes"])
            elif msg.get("text"):
                await session.on_text(msg["text"])
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)
        session.close()


class NoCacheStaticFiles(StaticFiles):
    """Make the browser revalidate the frontend on every load, so edits show up on a plain refresh."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/", NoCacheStaticFiles(directory=FRONTEND, html=True), name="frontend")


def open_browser_when_ready(server: uvicorn.Server) -> None:
    """Open the frontend only once uvicorn is actually accepting connections."""

    def run() -> None:
        while not server.started:
            if server.should_exit:
                return
            time.sleep(0.1)
        import webbrowser

        webbrowser.open(f"http://127.0.0.1:{PORT}")

    threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning"))
    # start.bat sets LIVETRANS_OPEN_BROWSER=1 so double-clicking it opens the page once the server is up.
    if os.environ.get("LIVETRANS_OPEN_BROWSER", "") == "1":
        open_browser_when_ready(server)
    server.run()
