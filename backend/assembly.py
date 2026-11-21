"""AssemblyAI streaming speech-to-text (Universal-Streaming v3) as a cloud recognition engine.

One AssemblyStream per capture run: the page's 16 kHz PCM is forwarded as it arrives, and
AssemblyAI's turns (it does its own end-of-sentence detection) come back as events that
server.py turns into the page's "partial" / "segment" messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from urllib.parse import urlencode

from asr import ANY, IT_EN, LANGS, is_fixed, normalize_language, prompt_parts

log = logging.getLogger("livetrans.assembly")

URL = "wss://streaming.assemblyai.com/v3/ws"

ENGLISH = "universal-streaming-english"  # English only; reports no language
MULTILINGUAL = "universal-streaming-multilingual"  # en/es/de/fr/pt/it, one language per turn
PRO = "universal-3-6-pro"  # 32 languages, switches within a sentence
STT37 = "assemblyai-stt-3-7-realtime"  # 32 languages, switches within a sentence; for notetakers
# Models whose turns always come formatted (they take no format_turns: each turn ends once) and
# that can be steered toward the recognition languages (language_codes).
STEERABLE = (PRO, STT37)
# Only these names are ever sent: the name comes from the page.
MODELS = (ENGLISH, MULTILINGUAL, PRO, STT37)
DEFAULT_MODEL = ENGLISH

# Audio goes out in chunks of this size (the API accepts 20-1000 ms per message).
CHUNK_BYTES = 1600  # 50 ms of 16 kHz PCM16
def _speed() -> float:
    try:
        return max(1.0, float(os.environ.get("LIVETRANS_AAI_SPEED") or 1.0))
    except ValueError:
        return 1.0


# Audio must reach AssemblyAI no faster than it was captured: sent faster, it is transcribed
# worse or the session is closed (error 3007). Audio that queued up before the session began
# (while connecting, or kept for a new session after the connection dropped) therefore goes out
# at this speed too, and the results stay that much behind until the capture is stopped.
# SPEED: seconds of audio per second. LIVETRANS_AAI_SPEED sets it for trying out a faster
# catch-up (e.g. 1.1): AssemblyAI states no tolerance, so anything above 1 is at one's own risk.
SPEED = _speed()
# The sender may run ahead of that schedule by this much audio, to make up for its own late
# wake-ups. Well within the API's limit: one message may carry up to a second of audio.
BURST_S = 0.2
# Audio captured while the session was open cannot add up to more than the session's age, however
# late it is sent. So time during which nothing was waiting to be sent is kept as credit, up to
# this much audio: what queues up later (a stalled page, a late wake-up) then goes out at once
# instead of staying behind for the rest of the run.
MAX_BURST_S = 1.0
# The sender reports how far behind the capture it is, this often, while it is this far behind.
BEHIND_LOG_EVERY_S = 10.0
BEHIND_LOG_MIN_S = 0.2
# Audio is kept until it is committed (see AssemblyStream.committed), so that a new session can
# be given it again after the connection to AssemblyAI drops. At most this much.
KEEP_S = 120
# While nobody is speaking, audio older than this counts as answered. Not all of it: the first
# words of a sentence take a moment to come back as a turn.
SILENCE_MARGIN_S = 2.0
# A failed connection is retried this many times in a row, this long apart (times the attempt).
RETRIES = 4
RETRY_S = 1.0
# A session that lasted this long was working: failures before it no longer count.
HEALTHY_S = 30.0
# After "Terminate", how long to wait for the last turn and the "Termination" that follows it.
CLOSE_TIMEOUT_S = 10.0
MAX_KEYTERMS = 100
MAX_KEYTERM_CHARS = 50


def normalize_model(value: object) -> str | None:
    """The model to use for the page's choice ("" is the default), or None if it isn't one."""
    if not isinstance(value, str):
        return None
    name = value or DEFAULT_MODEL
    return name if name in MODELS else None


def keyterms(prompt: str) -> list[str]:
    """The glossary terms of the page's recognition prompt (see asr.split_prompt), as AssemblyAI
    takes them: at most MAX_KEYTERMS, and none longer than MAX_KEYTERM_CHARS (it ignores those)."""
    _, terms = prompt_parts(prompt or "")
    return [t for t in terms if len(t) <= MAX_KEYTERM_CHARS][:MAX_KEYTERMS]


def language_codes(model: str, language: object) -> list[str] | None:
    """Languages to steer the model toward, for the recognition-language setting; None = no
    steering. Only Universal-3.6 Pro and STT-3.7 Realtime take them: the other models choose by
    themselves."""
    if model not in STEERABLE:
        return None
    mode = normalize_language(language) or IT_EN
    if mode == ANY:
        return None
    return list(LANGS) if mode == IT_EN else [mode]


def fixed_language(model: str, language: object) -> str:
    """The language of a result that carries none: English for the English model, else the
    recognition-language setting when it is one fixed language ("" when it isn't)."""
    if model == ENGLISH:
        return "en"
    mode = normalize_language(language) or IT_EN
    return mode if is_fixed(mode) else ""


def stream_url(model: str, language: object) -> str:
    query = {"speech_model": model, "sample_rate": "16000", "encoding": "pcm_s16le"}
    if model not in STEERABLE:
        # Punctuation and casing: each turn then ends twice, unformatted and then formatted.
        query["format_turns"] = "true"
    if model != ENGLISH:
        query["language_detection"] = "true"
    codes = language_codes(model, language)
    if codes:
        query["language_codes"] = json.dumps(codes)
    return f"{URL}?{urlencode(query)}"


def open_socket(url: str, key: str):
    """The connection to AssemblyAI (an async context manager). Replaced by a fake in the tests."""
    from websockets.asyncio.client import connect

    return connect(url, additional_headers={"Authorization": key}, open_timeout=10, max_size=2**22)


def is_fatal(e: Exception) -> bool:
    """Whether trying again is pointless: the key was refused (invalid, disabled, out of funds)."""
    code = getattr(getattr(e, "rcvd", None), "code", None)  # websockets.ConnectionClosed
    status = getattr(getattr(e, "response", None), "status_code", None)  # websockets.InvalidStatus
    return code == 1008 or (status is not None and 400 <= status < 500 and status not in (408, 429))


class AssemblyStream:
    """The audio of one capture run, sent to AssemblyAI. Positions are samples since the run's
    first sample on this stream.

    `on_event(stream, event)` is awaited with dicts whose "kind" is:
      state    "state": connecting | ready | error (then "message"; the stream has given up)
      partial  "index", "text": the turn being spoken, so far
      final    "index", "text", "start", "duration" (seconds), "lang", "lang_prob"
      drop     "index": a partial that will get no final
      closed   nothing more will come
    Turn indices count from 1 and keep growing across reconnections.
    """

    def __init__(self, key: str, model: str, language: object, terms: list[str], on_event) -> None:
        self.key = key
        self.model = model
        self.language = language
        self.terms = list(terms)
        self.on_event = on_event
        self.run = 0  # set by the owner
        self.fed = 0  # samples taken in
        # Everything before this position is answered (its turns were sent) or was silence.
        self.committed = 0
        self.done = False  # the task has ended
        self.dead = False  # gave up: what is fed from now on is thrown away
        self._buf = bytearray()
        self._chunks: deque[tuple[int, bytes]] = deque()  # (position, audio), not committed yet
        self._sent_count = 0  # how many of _chunks went to the current session
        self._sent_pos = 0  # position after the last chunk sent
        self._session_start = 0  # position of the current session's first sample
        self._order_base = 0  # turn indices used by earlier sessions
        self._max_index = 0
        self._open: dict | None = None  # the turn being spoken: {"index", "text"}
        self._last_end = 0.0  # where the last turn ended (seconds), for a turn without words
        self._pending: dict = {}  # configuration changes not sent yet
        self._closing = False
        self._terminate_sent = False
        self._error = ""  # what AssemblyAI's last "Error" message said
        self._wake = asyncio.Event()
        self._begun = asyncio.Event()
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------ owner's side

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    def feed(self, pcm: bytes) -> None:
        if self._closing:
            return
        self._buf += pcm
        while len(self._buf) >= CHUNK_BYTES:
            self._push(bytes(self._buf[:CHUNK_BYTES]))
            del self._buf[:CHUNK_BYTES]

    def update(self, language: object, terms: list[str]) -> None:
        """The recognition language or the glossary changed while capturing."""
        change: dict = {}
        if language_codes(self.model, language) != language_codes(self.model, self.language):
            change["language_codes"] = language_codes(self.model, language) or []
        if list(terms) != self.terms:
            change["keyterms_prompt"] = list(terms)
        self.language, self.terms = language, list(terms)
        if change:
            self._pending.update(change)
            self._wake.set()

    def close(self) -> None:
        """No more audio: send what is left, let the last turn come back, then end ("closed")."""
        if self._closing:
            return
        if self._buf:
            self._push(bytes(self._buf) + b"\0" * (CHUNK_BYTES - len(self._buf)))
            self._buf.clear()
        self._closing = True
        self._wake.set()

    def abort(self) -> None:
        """The page went away: stop at once, without reporting anything."""
        if self._task and not self._task.done():
            self._task.cancel()

    # ------------------------------------------------------------ audio kept for resending

    def _push(self, chunk: bytes) -> None:
        pos = self.fed
        self.fed += len(chunk) // 2
        if self.dead:
            self.committed = self.fed
            return
        self._chunks.append((pos, chunk))
        while self._chunks and self.fed - self._chunks[0][0] > KEEP_S * 16000:
            self._chunks.popleft()
            self._sent_count = max(0, self._sent_count - 1)
        self._wake.set()

    def _commit(self, pos: int) -> None:
        if pos <= self.committed:
            return
        self.committed = pos
        while self._chunks and self._chunks[0][0] + len(self._chunks[0][1]) // 2 <= pos:
            self._chunks.popleft()
            self._sent_count = max(0, self._sent_count - 1)

    # ------------------------------------------------------------ the connection

    async def _emit(self, **event) -> None:
        try:
            await self.on_event(self, event)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the page went away; the owner aborts this stream
            log.debug("Event not delivered", exc_info=True)

    async def _run(self) -> None:
        failures = 0
        try:
            while True:
                problem = ""
                if not self.key:
                    problem = "请先在「设置 → 识别」填写 AssemblyAI API Key"
                    break
                await self._emit(kind="state", state="connecting")
                t0 = time.monotonic()
                self._error = ""
                try:
                    terminated = await self._session()
                    if terminated and self._closing:
                        break
                    detail = self._error or "连接被关闭"
                    fatal = False
                except Exception as e:  # noqa: BLE001 - any failure of the connection
                    detail = self._error or str(e) or type(e).__name__
                    fatal = is_fatal(e)
                    log.warning("AssemblyAI session failed: %s", detail)
                if time.monotonic() - t0 >= HEALTHY_S:
                    failures = 0
                failures += 1
                if fatal or failures > RETRIES:
                    hint = "，请检查 API Key 和账户余额" if fatal else ""
                    problem = f"AssemblyAI 连接失败：{detail}{hint}"
                    break
                await asyncio.sleep(RETRY_S * failures)
            if problem:
                await self._give_up(problem)
            else:
                await self._flush_open()
            self._commit(self.fed)
            self.done = True
            await self._emit(kind="closed")
        finally:
            self.done = True

    async def _give_up(self, message: str) -> None:
        self.dead = True
        self._chunks.clear()
        if self._open:
            await self._emit(kind="drop", index=self._open["index"])
            self._open = None
        await self._emit(kind="state", state="error", message=message)

    async def _flush_open(self) -> None:
        # Terminated before the turn's (formatted) end arrived: what was heard is the result.
        turn, self._open = self._open, None
        if turn:
            await self._final(turn["index"], turn["text"], turn["words"], {})

    async def _session(self) -> bool:
        """One connection: True when AssemblyAI ended it properly ("Termination")."""
        # What an earlier session was sent without answering is sent again.
        self._sent_count = 0
        self._session_start = self._chunks[0][0] if self._chunks else self.fed
        self._sent_pos = self._session_start
        self._order_base = self._max_index
        self._open = None
        self._pending = {}
        self._terminate_sent = False
        self._begun = asyncio.Event()
        t0 = time.monotonic()
        async with open_socket(stream_url(self.model, self.language), self.key) as ws:
            sender = asyncio.create_task(self._send(ws))
            try:
                while True:
                    timeout = CLOSE_TIMEOUT_S if self._terminate_sent else None
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout))
                    kind = msg.get("type")
                    if kind == "Begin":
                        # Audio waiting now was captured before the session: it is sent at the
                        # speed it was captured at, and the results stay this far behind.
                        log.info(
                            "AssemblyAI session ready after %.2f s, %.2f s of audio waiting",
                            time.monotonic() - t0, (self.fed - self._session_start) / 16000,
                        )
                        self._begun.set()
                        await self._emit(kind="state", state="ready")
                    elif kind == "Turn":
                        await self._turn(msg)
                    elif kind == "Error":
                        self._error = str(msg.get("error") or msg.get("error_code") or "")
                    elif kind == "Termination":
                        return True
            finally:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)

    async def _send(self, ws) -> None:
        await self._begun.wait()
        if self.terms:
            await ws.send(json.dumps({"type": "UpdateConfiguration", "keyterms_prompt": self.terms}))
        next_at = 0.0  # on time.monotonic(): the next chunk is not sent before this (see SPEED)
        begun = time.monotonic()
        behind, behind_at = 0.0, begun  # the furthest behind the capture since the last report
        while True:
            if self._pending:
                change, self._pending = self._pending, {}
                await ws.send(json.dumps({"type": "UpdateConfiguration", **change}))
            if self._sent_count < len(self._chunks):
                wait = next_at - time.monotonic()
                if wait > 0:
                    # The queue can change meanwhile (audio committed or too old is dropped).
                    await asyncio.sleep(wait)
                    continue
                pos, chunk = self._chunks[self._sent_count]
                self._sent_count += 1
                await ws.send(chunk)
                now = time.monotonic()
                # How far ahead of the schedule this chunk may have been (see MAX_BURST_S).
                credit = min(MAX_BURST_S, BURST_S + (now - begun) * SPEED)
                next_at = max(next_at, now - credit / SPEED) + len(chunk) / 32000 / SPEED
                self._sent_pos = pos + len(chunk) // 2
                behind = max(behind, (self.fed - self._sent_pos) / 16000)
                if now - behind_at >= BEHIND_LOG_EVERY_S:
                    if behind >= BEHIND_LOG_MIN_S:
                        log.info("AssemblyAI: sending was up to %.2f s behind the capture", behind)
                    behind, behind_at = 0.0, now
                if self._open is None:
                    self._commit(self._sent_pos - round(SILENCE_MARGIN_S * 16000))
                continue
            if self._closing and not self._pending:
                self._terminate_sent = True
                await ws.send(json.dumps({"type": "Terminate"}))
                return
            self._wake.clear()
            await self._wake.wait()

    async def _turn(self, msg: dict) -> None:
        index = self._order_base + int(msg.get("turn_order") or 0) + 1
        self._max_index = max(self._max_index, index)
        words = msg.get("words") or []
        text = (msg.get("transcript") or "").strip()
        # With format_turns a turn ends twice: unformatted first, then with punctuation.
        final = bool(msg.get("end_of_turn")) and (self.model in STEERABLE or bool(msg.get("turn_is_formatted")))
        if final:
            await self._final(index, text, words, msg)
            return
        if not msg.get("end_of_turn"):
            # "transcript" holds the words that are settled; "words" also has the newest ones.
            text = " ".join(str(w.get("text") or "").strip() for w in words).strip() or text
        if text:
            self._open = {"index": index, "text": text, "words": words}
            # "end": where the newest word ends (seconds of this stream's audio), so the page can
            # tell how long ago the audio behind this text was sent.
            end = self._session_start / 16000 + float(words[-1].get("end") or 0) / 1000 if words else None
            await self._emit(kind="partial", index=index, text=text, end=end)

    async def _final(self, index: int, text: str, words: list, msg: dict) -> None:
        self._open = None
        base = self._session_start / 16000
        start = base + float(words[0].get("start") or 0) / 1000 if words else self._last_end
        end = max(start, base + float(words[-1].get("end") or 0) / 1000) if words else start
        self._last_end = end
        self._commit(min(round(end * 16000), self._sent_pos))
        lang = msg.get("language_code") or fixed_language(self.model, self.language)
        confidence = msg.get("language_confidence")
        await self._emit(
            kind="final",
            index=index,
            text=text,
            start=start,
            duration=end - start,
            lang=lang,
            lang_prob=round(float(confidence), 3) if confidence is not None else (1.0 if lang else 0.0),
        )
