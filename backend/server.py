"""LiveTrans backend: audio over WebSocket -> VAD -> Whisper; translation proxy; static frontend."""

from __future__ import annotations

import asyncio
import json
import logging
import os
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

from asr import Transcriber  # noqa: E402
from guard import LocalOnlyMiddleware  # noqa: E402
from translate import TranslateError, Translator, clean_output, clean_partial  # noqa: E402
from vad import FRAME, Segment, Segmenter, SegmenterParams, SileroModel  # noqa: E402
from winfo import window_info  # noqa: E402
from asr import IT_EN, all_languages, detect_among, is_fixed, normalize_language  # noqa: E402
from translate import DEFAULT_DOMAIN  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

HOST = os.environ.get("LIVETRANS_HOST", "127.0.0.1")
PORT = int(os.environ.get("LIVETRANS_PORT", "8765"))
MODEL = os.environ.get("LIVETRANS_MODEL", "large-v3-turbo")
DEVICE = os.environ.get("LIVETRANS_DEVICE", "auto")
COMPUTE_TYPE = os.environ.get("LIVETRANS_COMPUTE_TYPE", "auto")
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


def status_message() -> dict:
    return {
        "type": "status",
        "model": transcriber.status,
        "model_name": transcriber.model_name or MODEL,
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


def load_model_in_background() -> None:
    def run() -> None:
        transcriber.load(MODEL, DEVICE, COMPUTE_TYPE)
        if loop:
            asyncio.run_coroutine_threadsafe(broadcast_status(), loop)

    transcriber.status = "loading"
    threading.Thread(target=run, daemon=True).start()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global translator, vad_model, loop
    loop = asyncio.get_running_loop()
    vad_model = SileroModel()
    translator = Translator()
    load_model_in_background()
    log.info("Open http://%s:%d in Chrome", HOST, PORT)
    yield
    await translator.close()


app = FastAPI(lifespan=lifespan)
# Other websites open in the browser must not reach /ws or the API (see guard.py).
app.add_middleware(LocalOnlyMiddleware, allowed_hosts=ALLOWED_HOSTS)


@app.get("/api/languages")
async def api_languages() -> dict:
    """Language codes the recognition-language setting may choose from ("其他")."""
    return {"languages": list(all_languages())}


@app.get("/api/status")
async def api_status() -> dict:
    return status_message()


@app.get("/api/window")
async def api_window(hwnd: int) -> dict:
    """Name of a shared window: Chrome's window-capture track label carries its native handle."""
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


@app.post("/api/translate/warmup")
async def api_translate_warmup(req: WarmupRequest) -> dict:
    try:
        await translator.warmup(req.config)
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
# A failed final transcription is retried this many times in all, this long apart, before its
# audio is given up (and stops being kept for resending, see ws_audio).
TRANSCRIBE_ATTEMPTS = 3
TRANSCRIBE_RETRY_S = 1.0


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


@app.websocket("/ws")
async def ws_audio(ws: WebSocket) -> None:
    await ws.accept()
    clients.add(ws)
    await ws.send_json(status_message())

    seg = Segmenter(vad_model, SegmenterParams())
    opts = {"language": IT_EN, "prompt": "", "partial": True}
    queue: asyncio.Queue = asyncio.Queue()
    live = {"busy": False, "last_frame": 0, "final_sent": 0, "lang_index": 0, "lang": ""}
    partial_every = round(PARTIAL_INTERVAL_S * 16000 / 512)

    in_hand = [0]  # segments taken off the queue whose result hasn't been sent yet
    # Capture run, set by every "start" (the page numbers its runs); tags queued segments (see
    # take_batch) and is echoed in their results, so a run that is still being transcribed after
    # 开始 was pressed again keeps its own time base on the page and can't touch the new run's state.
    run = [0]

    # Resuming after a dropped connection: the page keeps the audio it sent and, on reconnect,
    # resends it from the last "committed" position, i.e. the point before which every segment
    # has been answered (or was silence). Positions are samples since the page pressed 开始.
    # run -> position of that run's first sample on this connection (set by "resume"). Per run:
    # a reconnecting page may resend several runs, and a run's segments can still be transcribed
    # after the next run began.
    bases: dict[int, int] = {}
    unsent: dict[tuple[int, int], int] = {}  # (run, index) -> start position, not answered yet

    def committed() -> int:
        pos = seg.committed_frames() * FRAME
        mine = [p for (r, _), p in unsent.items() if r == run[0]]
        if mine:
            pos = min(pos, min(mine))
        return bases.get(run[0], 0) + pos

    async def send_run_done(r: int) -> None:
        # An ended run (开始 pressed again, or the next run resumed) whose segments are all
        # answered: the page can drop the audio it kept for it.
        if r != run[0] and not any(k[0] == r for k in unsent):
            await ws.send_json({"type": "run_done", "run": r})

    def commit_info() -> dict:
        # The run the position belongs to: a message sent before a "start" was processed may reach
        # the page after it started the new run, and must not move that run's resend cursor.
        return {"committed": committed(), "committed_run": run[0]}

    async def send_backlog() -> None:
        # Segments waiting for (or in) their final transcription; the UI warns when it grows.
        await ws.send_json(
            {"type": "backlog", "pending": queue.qsize() + in_hand[0], **commit_info()}
        )

    async def transcribe_batch(items: list[Segment]) -> None:
        if transcriber.status != "ready":
            answered(items)
            await ws.send_json({"type": "error", "message": "模型加载失败，这段语音无法识别"})
            return
        item = merge_segments(items)
        # Answering a batch advances "committed", and the page then drops the kept audio. So a
        # failure (e.g. a transient CUDA error) is retried before the batch is given up.
        for attempt in range(1, TRANSCRIBE_ATTEMPTS + 1):
            try:
                async with gpu_lock:
                    res = await to_thread_to_end(
                        transcriber.transcribe, item.audio, opts["language"], opts["prompt"]
                    )
                break
            except Exception as e:  # noqa: BLE001
                log.exception("Transcription failed (attempt %d of %d)", attempt, TRANSCRIBE_ATTEMPTS)
                if attempt == TRANSCRIBE_ATTEMPTS:
                    answered(items)
                    await ws.send_json({"type": "error", "message": f"识别出错：{e}"})
                    return
                await asyncio.sleep(TRANSCRIBE_RETRY_S)
        answered(items)
        res.update(
            type="segment",
            index=item.index,
            start=round(item.start + bases.get(item.run, 0) / 16000, 3),
            # Sent with the result itself: if the connection drops right after, the page won't
            # resend (and show twice) the audio of a sentence it already has.
            **commit_info(),
            duration=round(span_s(items), 2),
            merged=len(items),
            run=item.run,
        )
        if item.run == run[0]:  # an older run's indices mean nothing for the current one
            live["final_sent"] = max(live["final_sent"], item.index)
        log.info(
            "[%s %.2fs %dms%s] %s",
            res["lang"], res["duration"], res["asr_ms"],
            f" merged {len(items)}" if len(items) > 1 else "", res["text"],
        )
        await ws.send_json(res)

    def answered(items: list[Segment]) -> None:
        for it in items:
            unsent.pop((it.run, it.index), None)

    async def wait_for_model() -> None:
        # The first launch downloads the model (~1.6 GB) while the page may already be capturing:
        # keep that speech queued and transcribe it (merged, see take_batch) once the model is
        # ready, instead of dropping it. The page shows the backlog meanwhile.
        if transcriber.status in ("idle", "loading"):
            await send_backlog()
            while transcriber.status in ("idle", "loading"):
                await asyncio.sleep(MODEL_WAIT_POLL_S)

    async def worker() -> None:
        carry: Segment | None = None
        while True:
            try:
                first = carry or await queue.get()
                carry = None
                in_hand[0] = 1
                await wait_for_model()
                items, carry = take_batch(first, queue)
                in_hand[0] = len(items) + (carry is not None)
                try:
                    await transcribe_batch(items)
                    await send_run_done(items[0].run)
                finally:
                    in_hand[0] = carry is not None
                await send_backlog()
            except Exception:  # noqa: BLE001 - one bad batch must not stop transcription for good
                if ws.client_state != WebSocketState.CONNECTED:
                    return  # the page went away; ws_audio cancels this task anyway
                log.exception("ASR worker failed on a batch; continuing")
                carry, in_hand[0] = None, 0

    # Bumped whenever the recognition language mode changes. A partial that started under an
    # older mode (it runs in a worker thread) neither caches the language it detected nor shows
    # its text. lang_lock keeps the mode, its generation and the cache consistent across threads.
    mode_gen = [0]
    lang_lock = threading.Lock()

    def partial_job(audio, index: int) -> str:
        with lang_lock:
            gen, lang = mode_gen[0], opts["language"]
            cached = live["lang"] if live["lang_index"] == index else ""
        if not is_fixed(lang):
            if cached:
                lang = cached
            else:
                lang, _ = transcriber.pick_language(audio, detect_among(lang))
                if len(audio) >= PARTIAL_LANG_LOCK_S * 16000:
                    with lang_lock:
                        if mode_gen[0] == gen:
                            live["lang_index"], live["lang"] = index, lang
        text = transcriber.transcribe_partial(audio, lang, opts["prompt"])
        with lang_lock:
            return text if mode_gen[0] == gen else ""

    async def run_partial(audio, index: int) -> None:
        try:
            async with gpu_lock:
                if index <= live["final_sent"]:
                    return
                text = await to_thread_to_end(partial_job, audio, index)
            if text and index > live["final_sent"]:
                await ws.send_json({"type": "partial", "index": index, "text": text})
        except Exception:  # noqa: BLE001 - a lost partial is harmless; the final pass follows
            log.debug("Partial transcription failed", exc_info=True)
        finally:
            live["busy"] = False

    def maybe_partial() -> None:
        # Finals take priority: skip while one is queued or running.
        if not opts["partial"] or live["busy"] or transcriber.status != "ready":
            return
        if not queue.empty() or gpu_lock.locked() or not seg.speaking:
            return
        if seg.frames_seen - live["last_frame"] < partial_every:
            return
        audio = seg.current_audio()
        if audio is None or len(audio) < PARTIAL_MIN_AUDIO_S * 16000:
            return
        live["busy"] = True
        live["last_frame"] = seg.frames_seen
        # The event loop only keeps weak references to tasks: hold one until it is done.
        t = asyncio.create_task(run_partial(audio, seg.index))
        partial_tasks.add(t)
        t.add_done_callback(partial_tasks.discard)

    partial_tasks: set[asyncio.Task] = set()
    task = asyncio.create_task(worker())
    speaking = False
    last_level = 0.0
    rec: list[wave.Wave_write | None] = [None]

    def close_recording() -> None:
        if rec[0] is not None:
            rec[0].close()
            rec[0] = None
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if RECORD:
                    if rec[0] is None:
                        rec[0] = open_recording()
                    rec[0].writeframes(msg["bytes"])
                new = seg.feed(msg["bytes"])
                for s in new:
                    s.run = run[0]
                    unsent[(s.run, s.index)] = round(s.start * 16000)
                    queue.put_nowait(s)
                if new:
                    await send_backlog()
                for index in seg.take_dropped():
                    await ws.send_json({"type": "partial_drop", "index": index})
                now = time.monotonic()
                if seg.speaking != speaking or now - last_level >= VAD_LEVEL_INTERVAL_S:
                    speaking = seg.speaking
                    last_level = now
                    await ws.send_json({
                        "type": "vad",
                        "speaking": speaking,
                        "prob": round(seg.take_peak_prob(), 2),
                        **commit_info(),
                    })
                maybe_partial()
            elif msg.get("text"):
                data = json.loads(msg["text"])
                kind = data.get("type")
                if kind == "config":
                    asr = data.get("asr") or {}
                    apply_config(seg, asr)
                    mode = normalize_language(asr.get("language"))
                    if mode and mode != opts["language"]:
                        # The language detected for the segment in progress was chosen under the
                        # old mode: detect again for its next partials, and drop partials still
                        # running under the old mode (see partial_job).
                        with lang_lock:
                            opts["language"] = mode
                            mode_gen[0] += 1
                            live.update(lang_index=0, lang="")
                    if "partial" in asr:
                        opts["partial"] = bool(asr["partial"])
                    opts["prompt"] = (data.get("prompt") or "")[:1000]
                elif kind in ("start", "resume"):
                    # "resume": a reconnecting page continues its stream at `offset` and resends
                    # the audio from there; segment start times continue from that position.
                    old = run[0]
                    r = data.get("run")
                    run[0] = r if isinstance(r, int) and not isinstance(r, bool) else run[0] + 1
                    seg.reset()
                    live.update(last_frame=0, final_sent=0, lang_index=0, lang="")
                    bases[run[0]] = max(0, int(data.get("offset") or 0)) if kind == "resume" else 0
                    if old and old != run[0]:  # 0: no run yet on this connection
                        await send_run_done(old)
                elif kind == "stop":
                    close_recording()
                    new = seg.flush()
                    for s in new:
                        s.run = run[0]
                        unsent[(s.run, s.index)] = round(s.start * 16000)
                        queue.put_nowait(s)
                    if new:
                        await send_backlog()
                    for index in seg.take_dropped():
                        await ws.send_json({"type": "partial_drop", "index": index})
                    speaking = False
                    await ws.send_json({"type": "vad", "speaking": False})
    except WebSocketDisconnect:
        pass
    finally:
        close_recording()
        clients.discard(ws)
        task.cancel()
        for t in partial_tasks:
            t.cancel()


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
