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

from asr import setup_cuda_dlls

setup_cuda_dlls()  # must run before ctranslate2 is imported

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from asr import Transcriber  # noqa: E402
from translate import TranslateError, Translator, clean_output, clean_partial  # noqa: E402
from vad import Segmenter, SegmenterParams, SileroModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"

HOST = os.environ.get("LIVETRANS_HOST", "127.0.0.1")
PORT = int(os.environ.get("LIVETRANS_PORT", "8765"))
MODEL = os.environ.get("LIVETRANS_MODEL", "large-v3-turbo")
DEVICE = os.environ.get("LIVETRANS_DEVICE", "auto")
COMPUTE_TYPE = os.environ.get("LIVETRANS_COMPUTE_TYPE", "auto")
# Set LIVETRANS_RECORD=1 to save each capture's raw 16 kHz audio to recordings/ (for tuning).
RECORD = os.environ.get("LIVETRANS_RECORD", "") == "1"
RECORDINGS = ROOT / "recordings"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("livetrans")

transcriber = Transcriber()
translator: Translator | None = None
vad_model: SileroModel | None = None
gpu_lock = asyncio.Lock()
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


@app.get("/api/status")
async def api_status() -> dict:
    return status_message()


class TranslateRequest(BaseModel):
    provider: str
    config: dict
    text: str
    context: list[dict] = []
    glossary: list[dict] = []


@app.post("/api/translate")
async def api_translate(req: TranslateRequest) -> dict:
    try:
        out = await translator.translate(req.provider, req.config, req.text, req.context, req.glossary)
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
                req.provider, req.config, req.text, req.context, req.glossary
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
    opts = {"language": "auto", "prompt": "", "partial": True}
    queue: asyncio.Queue = asyncio.Queue()
    live = {"busy": False, "last_frame": 0, "final_sent": 0, "lang_index": 0, "lang": ""}
    partial_every = round(PARTIAL_INTERVAL_S * 16000 / 512)

    async def worker() -> None:
        while True:
            item = await queue.get()
            if transcriber.status != "ready":
                await ws.send_json({"type": "error", "message": "模型尚未就绪，这段语音被跳过"})
                continue
            try:
                async with gpu_lock:
                    res = await asyncio.to_thread(
                        transcriber.transcribe, item.audio, opts["language"], opts["prompt"]
                    )
            except Exception as e:  # noqa: BLE001
                log.exception("Transcription failed")
                await ws.send_json({"type": "error", "message": f"识别出错：{e}"})
                continue
            res.update(
                type="segment",
                index=item.index,
                start=round(item.start, 3),
                duration=round(len(item.audio) / 16000, 2),
            )
            live["final_sent"] = max(live["final_sent"], item.index)
            log.info("[%s %.2fs %dms] %s", res["lang"], res["duration"], res["asr_ms"], res["text"])
            await ws.send_json(res)

    def partial_job(audio, index: int) -> str:
        lang = opts["language"]
        if lang not in ("it", "en"):
            if live["lang_index"] == index and live["lang"]:
                lang = live["lang"]
            else:
                lang, _ = transcriber.pick_language(audio)
                if len(audio) >= PARTIAL_LANG_LOCK_S * 16000:
                    live["lang_index"], live["lang"] = index, lang
        return transcriber.transcribe_partial(audio, lang, opts["prompt"])

    async def run_partial(audio, index: int) -> None:
        try:
            async with gpu_lock:
                if index <= live["final_sent"]:
                    return
                text = await asyncio.to_thread(partial_job, audio, index)
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
        asyncio.create_task(run_partial(audio, seg.index))

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
                for s in seg.feed(msg["bytes"]):
                    queue.put_nowait(s)
                for index in seg.take_dropped():
                    await ws.send_json({"type": "partial_drop", "index": index})
                now = time.monotonic()
                if seg.speaking != speaking or now - last_level >= VAD_LEVEL_INTERVAL_S:
                    speaking = seg.speaking
                    last_level = now
                    await ws.send_json(
                        {"type": "vad", "speaking": speaking, "prob": round(seg.take_peak_prob(), 2)}
                    )
                maybe_partial()
            elif msg.get("text"):
                data = json.loads(msg["text"])
                kind = data.get("type")
                if kind == "config":
                    asr = data.get("asr") or {}
                    apply_config(seg, asr)
                    if asr.get("language") in ("auto", "it", "en"):
                        opts["language"] = asr["language"]
                    if "partial" in asr:
                        opts["partial"] = bool(asr["partial"])
                    opts["prompt"] = (data.get("prompt") or "")[:1000]
                elif kind == "start":
                    seg.reset()
                    live.update(last_frame=0, final_sent=0, lang_index=0, lang="")
                elif kind == "stop":
                    close_recording()
                    for s in seg.flush():
                        queue.put_nowait(s)
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
