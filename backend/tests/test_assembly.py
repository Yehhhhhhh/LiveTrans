"""The AssemblyAI engine (设置 → 识别 → 识别引擎), with the connection to AssemblyAI faked."""

import asyncio
import json
import time
import types
from contextlib import contextmanager
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

import assembly
import server
from assembly import ENGLISH, MULTILINGUAL, PRO, STT37, AssemblyStream
from test_ws_worker import FakeSegmenter, FakeTranscriber, read_until

CHUNK = b"\0" * assembly.CHUNK_BYTES  # 50 ms


def turn(order, text, words=(), end=False, formatted=False, **extra):
    return {
        "type": "Turn", "turn_order": order, "transcript": text, "end_of_turn": end,
        "turn_is_formatted": formatted,
        "words": [{"text": t, "start": a, "end": b} for t, a, b in words],
        **extra,
    }


class FakeSocket:
    """AssemblyAI's end of a session. `script` says what it sends back: {number of audio chunks
    received: [messages]}, and under "end" what comes after "Terminate" (before "Termination").
    An exception among the messages is raised by recv, like a dropped connection."""

    def __init__(self, script):
        self.script = script
        self.sent = []
        self.chunks = 0
        self.inbox = asyncio.Queue()
        self.put({"type": "Begin", "id": "x", "expires_at": 0})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def put(self, m):
        self.inbox.put_nowait(m if isinstance(m, Exception) else json.dumps(m))

    async def send(self, data):
        self.sent.append(data)
        if isinstance(data, bytes):
            self.chunks += 1
            for m in self.script.get(self.chunks, []):
                self.put(m)
        elif json.loads(data)["type"] == "Terminate":
            for m in self.script.get("end", []):
                self.put(m)
            self.put({"type": "Termination"})

    async def recv(self):
        m = await self.inbox.get()
        if isinstance(m, Exception):
            raise m
        return m

    def texts(self):
        return [json.loads(d) for d in self.sent if isinstance(d, str)]


class Refused:
    def __init__(self, error):
        self.error = error

    async def __aenter__(self):
        raise self.error

    async def __aexit__(self, *exc):
        return False


class Hub:
    """Stands in for assembly.open_socket: one script (or an exception: the connection fails)
    per connection, in order; connections beyond the list get an empty script."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.calls = []
        self.sockets = []

    def __call__(self, url, key):
        self.calls.append((url, key))
        script = self.scripts.pop(0) if self.scripts else {}
        if isinstance(script, Exception):
            return Refused(script)
        self.sockets.append(FakeSocket(script))
        return self.sockets[-1]


def closed(code):
    e = RuntimeError(f"closed {code}")
    e.rcvd = types.SimpleNamespace(code=code)
    return e


@pytest.fixture
def hub(monkeypatch):
    h = Hub()
    monkeypatch.setattr(assembly, "open_socket", h)
    monkeypatch.setattr(assembly, "RETRY_S", 0)
    monkeypatch.setattr(assembly, "SPEED", 1000.0)  # the tests don't wait for audio to play out
    return h


def query(url):
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


# ---------------------------------------------------------------- settings -> connection


def test_default_model_is_universal_streaming_english():
    assert assembly.DEFAULT_MODEL == "universal-streaming-english"
    assert assembly.normalize_model("") == ENGLISH
    assert assembly.normalize_model(PRO) == PRO
    assert assembly.normalize_model(STT37) == "assemblyai-stt-3-7-realtime"


@pytest.mark.parametrize("value", [None, 3, "whisper", "universal-2", "../x"])
def test_unknown_models_are_refused(value):
    assert assembly.normalize_model(value) is None


def test_url_of_each_model():
    assert assembly.stream_url(ENGLISH, "it_en").startswith("wss://streaming.assemblyai.com/v3/ws?")
    common = {"sample_rate": "16000", "encoding": "pcm_s16le"}
    # The English model knows no other language: nothing about languages is sent.
    assert query(assembly.stream_url(ENGLISH, "it")) == {
        **common, "speech_model": ENGLISH, "format_turns": "true",
    }
    assert query(assembly.stream_url(MULTILINGUAL, "it_en")) == {
        **common, "speech_model": MULTILINGUAL, "format_turns": "true", "language_detection": "true",
    }
    assert query(assembly.stream_url(PRO, "it_en")) == {
        **common, "speech_model": PRO, "language_detection": "true", "language_codes": '["it", "en"]',
    }
    assert query(assembly.stream_url(PRO, "fr"))["language_codes"] == '["fr"]'
    assert "language_codes" not in query(assembly.stream_url(PRO, "any"))
    # STT-3.7 Realtime is sent like Universal-3.6 Pro: always formatted, steered by language.
    assert query(assembly.stream_url(STT37, "it_en")) == {
        **common, "speech_model": STT37, "language_detection": "true", "language_codes": '["it", "en"]',
    }
    assert query(assembly.stream_url(STT37, "it"))["language_codes"] == '["it"]'


def test_language_of_a_result_that_carries_none():
    assert assembly.fixed_language(ENGLISH, "it") == "en"
    assert assembly.fixed_language(PRO, "it") == "it"
    assert assembly.fixed_language(PRO, "it_en") == ""
    assert assembly.fixed_language(STT37, "en") == "en"


def test_keyterms_are_the_glossary_terms_within_the_limits():
    assert assembly.keyterms("Lezione di economia. PIL, tasso di sconto, Keynes.") == ["PIL", "tasso di sconto", "Keynes"]
    assert assembly.keyterms("PIL, Keynes") == ["PIL", "Keynes"]  # no course hint
    assert assembly.keyterms("Lezione di economia. ") == []
    assert assembly.keyterms("") == []
    many = ", ".join(f"term{i}" for i in range(150))
    assert len(assembly.keyterms(many)) == 100
    assert assembly.keyterms("ok, " + "x" * 51) == ["ok"]


# ---------------------------------------------------------------- AssemblyStream


async def until(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "timed out"
        await asyncio.sleep(0.005)



def run_stream(hub, model, language, terms, body):
    """Run `body(stream, events)` with a started stream; returns the events it reported."""
    events = []

    async def on_event(s, ev):
        events.append(ev)

    async def main():
        s = AssemblyStream("key", model, language, terms, on_event)
        s.start()
        try:
            await body(s, events)
        finally:
            s.abort()

    asyncio.run(main())
    return events


def kinds(events, kind):
    return [e for e in events if e["kind"] == kind]


def test_audio_is_sent_in_50ms_chunks_after_the_keyterms(hub):
    async def body(s, events):
        s.feed(b"\0" * 1000)
        s.feed(b"\0" * 3000)  # 4000 bytes: two whole chunks, 800 bytes wait for more
        await until(lambda: hub.sockets and hub.sockets[0].chunks == 2)
        s.close()  # the rest goes out padded to a whole chunk
        await until(lambda: s.done)

    events = run_stream(hub, ENGLISH, "en", ["Fourier"], body)
    sock = hub.sockets[0]
    assert hub.calls == [(assembly.stream_url(ENGLISH, "en"), "key")]
    assert sock.texts()[0] == {"type": "UpdateConfiguration", "keyterms_prompt": ["Fourier"]}
    assert [len(d) for d in sock.sent if isinstance(d, bytes)] == [1600, 1600, 1600]
    assert sock.texts()[-1] == {"type": "Terminate"}
    assert [e["state"] for e in kinds(events, "state")] == ["connecting", "ready"]
    assert events[-1] == {"kind": "closed"}


def sent_at(sock):
    """Records when each audio chunk reaches the fake AssemblyAI (seconds, time.monotonic)."""
    times = []
    send = sock.send

    async def timed(data):
        if isinstance(data, bytes):
            times.append(time.monotonic())
        await send(data)

    sock.send = timed
    return times


def test_queued_audio_is_sent_no_faster_than_it_was_captured(hub, monkeypatch):
    """Audio that queued up before the session was ready (or is resent to a new session) must
    not go out in one burst: AssemblyAI wants it at the speed of a live microphone."""
    monkeypatch.setattr(assembly, "SPEED", 10.0)  # ten times faster than real, to keep the test short
    monkeypatch.setattr(assembly, "BURST_S", 0.1)

    async def body(s, events):
        s.feed(CHUNK * 40)  # 2 s of audio, before the connection is there
        await until(lambda: hub.sockets)
        times = sent_at(hub.sockets[0])
        await until(lambda: hub.sockets[0].chunks == 40)
        # 2 s of audio at 10x: 0.2 s, less the 0.1 s (of audio) it may run ahead.
        assert times[-1] - times[0] >= 0.17
        s.close()
        await until(lambda: s.done)

    run_stream(hub, ENGLISH, "en", [], body)


def test_live_audio_is_not_held_back(hub, monkeypatch):
    """Audio arriving as it is captured goes out at once, and a change of settings is not kept
    waiting behind queued audio."""
    monkeypatch.setattr(assembly, "SPEED", 1.0)

    async def body(s, events):
        await until(lambda: kinds(events, "state")[-1:] == [{"kind": "state", "state": "ready"}])
        sock = hub.sockets[0]
        t0 = time.monotonic()
        s.feed(CHUNK * 4)
        await until(lambda: sock.chunks == 4)
        assert time.monotonic() - t0 < 0.1  # within what it may run ahead: not 50 ms apart
        s.feed(CHUNK * 20)  # a second of audio at once: this does take its time
        s.update("en", ["Keynes"])
        await until(lambda: {"type": "UpdateConfiguration", "keyterms_prompt": ["Keynes"]} in sock.texts())
        assert sock.chunks < 24

    run_stream(hub, ENGLISH, "en", [], body)


def test_audio_that_queues_up_during_the_session_is_sent_at_once(hub, monkeypatch):
    """Only audio from before the session has to be held back. While the session is open, time
    with nothing to send is credit: a page that stalled and then delivers its audio in one go is
    not kept behind for the rest of the run."""
    monkeypatch.setattr(assembly, "SPEED", 1.0)

    async def body(s, events):
        await until(lambda: kinds(events, "state")[-1:] == [{"kind": "state", "state": "ready"}])
        sock = hub.sockets[0]
        await asyncio.sleep(0.5)  # the session is half a second old, and nothing was sent
        t0 = time.monotonic()
        s.feed(CHUNK * 12)  # 0.6 s of audio at once
        await until(lambda: sock.chunks == 12)
        assert time.monotonic() - t0 < 0.2
        s.feed(CHUNK * 40)  # the credit is used up, and it is never more than MAX_BURST_S
        await asyncio.sleep(0.2)
        assert sock.chunks < 12 + 20

    run_stream(hub, ENGLISH, "en", [], body)


def test_formatted_models_end_a_turn_twice_and_report_it_once(hub):
    hub.scripts = [{
        2: [turn(0, "", [("hello", 20, 60)])],
        4: [
            turn(0, "hello world", [("hello", 20, 60), ("world", 70, 150)], end=True),
            turn(0, "Hello world.", [("Hello", 20, 60), ("world.", 70, 150)], end=True, formatted=True),
        ],
    }]

    async def body(s, events):
        s.feed(CHUNK * 4)
        await until(lambda: kinds(events, "final"))

    events = run_stream(hub, ENGLISH, "it_en", [], body)
    assert [(e["index"], e["text"]) for e in kinds(events, "partial")] == [(1, "hello"), (1, "hello world")]
    # Where the newest word of each partial ends, for the page's delay display.
    assert [e["end"] for e in kinds(events, "partial")] == [0.06, 0.15]
    assert kinds(events, "final") == [{
        "kind": "final", "index": 1, "text": "Hello world.", "start": 0.02,
        "duration": pytest.approx(0.13), "lang": "en", "lang_prob": 1.0,
    }]


@pytest.mark.parametrize("model", [PRO, STT37])
def test_steerable_models_end_a_turn_once_and_report_the_detected_language(hub, model):
    hub.scripts = [{
        2: [turn(0, "Buongiorno a tutti.", [("Buongiorno", 0, 50), ("tutti.", 60, 90)], end=True,
                 language_code="it", language_confidence=0.9234)],
        4: [turn(1, "Good morning.", [("Good", 110, 150), ("morning.", 160, 190)], end=True)],
    }]

    async def body(s, events):
        s.feed(CHUNK * 4)
        await until(lambda: len(kinds(events, "final")) == 2)

    finals = kinds(run_stream(hub, model, "it_en", [], body), "final")
    assert [(e["index"], e["text"], e["lang"], e["lang_prob"]) for e in finals] == [
        (1, "Buongiorno a tutti.", "it", 0.923),
        (2, "Good morning.", "", 0.0),  # no language reported, and none fixed
    ]


def test_committed_follows_the_answered_turns_and_silence(hub):
    hub.scripts = [{4: [turn(0, "Hi.", [("Hi.", 20, 150)], end=True, formatted=True)]}]

    async def body(s, events):
        s.feed(CHUNK * 4)
        await until(lambda: kinds(events, "final"))
        assert s.committed == 2400  # the end of the turn, 150 ms
        # Silence: everything but the last SILENCE_MARGIN_S counts as answered.
        s.feed(CHUNK * 100)
        await until(lambda: hub.sockets[0].chunks == 104)
        assert s.committed == 104 * 800 - 32000
        s.close()
        await until(lambda: s.done)
        assert s.committed == s.fed == 104 * 800

    run_stream(hub, ENGLISH, "en", [], body)


def test_the_turn_being_spoken_is_reported_when_the_stream_is_closed(hub):
    hub.scripts = [{2: [turn(0, "", [("so", 10, 40)])], "end": []}]

    async def body(s, events):
        s.feed(CHUNK * 2)
        await until(lambda: kinds(events, "partial"))
        s.close()
        await until(lambda: s.done)

    events = run_stream(hub, ENGLISH, "en", [], body)
    assert [(e["index"], e["text"]) for e in kinds(events, "final")] == [(1, "so")]


def test_a_dropped_session_is_replaced_and_given_the_unanswered_audio_again(hub):
    hub.scripts = [
        {2: [turn(0, "", [("one", 10, 40)])], 4: [closed(1006)]},
        {4: [turn(0, "One two.", [("One", 10, 40), ("two.", 50, 190)], end=True, formatted=True)]},
    ]

    async def body(s, events):
        s.feed(CHUNK * 4)
        await until(lambda: kinds(events, "final"))

    events = run_stream(hub, ENGLISH, "en", [], body)
    assert len(hub.calls) == 2
    assert hub.sockets[1].chunks == 4  # nothing was committed: all of it is sent again
    assert [e["state"] for e in kinds(events, "state")] == ["connecting", "ready", "connecting", "ready"]
    # Indices keep growing, so the page replaces the line of the lost partial.
    assert [e["index"] for e in kinds(events, "partial")] == [1]
    assert [(e["index"], e["text"]) for e in kinds(events, "final")] == [(2, "One two.")]


def test_a_failing_connection_is_retried_then_given_up(hub):
    hub.scripts = [OSError("no route")] * (assembly.RETRIES + 1)

    async def body(s, events):
        s.feed(CHUNK * 2)
        await until(lambda: s.done)
        s.feed(CHUNK * 2)  # thrown away, but counted as answered
        assert s.committed == s.fed == 3200

    events = run_stream(hub, ENGLISH, "en", [], body)
    assert len(hub.calls) == assembly.RETRIES + 1
    error = kinds(events, "state")[-1]
    assert error["state"] == "error" and "no route" in error["message"]
    assert events[-1] == {"kind": "closed"}


@pytest.mark.parametrize("error", [
    closed(1008),
    type("InvalidStatus", (Exception,), {"response": types.SimpleNamespace(status_code=401)})(),
])
def test_a_refused_key_is_not_retried(hub, error):
    hub.scripts = [error]

    async def body(s, events):
        await until(lambda: s.done)

    events = run_stream(hub, ENGLISH, "en", [], body)
    assert len(hub.calls) == 1
    assert "API Key" in kinds(events, "state")[-1]["message"]


def test_refused_while_speaking_drops_the_partial(hub):
    hub.scripts = [{2: [turn(0, "", [("one", 10, 40)]), {"type": "Error", "error": "Insufficient funds"}, closed(1008)]}]

    async def body(s, events):
        s.feed(CHUNK * 2)
        await until(lambda: s.done)

    events = run_stream(hub, ENGLISH, "en", [], body)
    assert kinds(events, "drop") == [{"kind": "drop", "index": 1}]
    assert "Insufficient funds" in kinds(events, "state")[-1]["message"]


def test_no_key_connects_nowhere(hub):
    events = []

    async def on_event(s, ev):
        events.append(ev)

    async def main():
        s = AssemblyStream("", ENGLISH, "en", [], on_event)
        s.start()
        await until(lambda: s.done)

    asyncio.run(main())
    assert hub.calls == []
    assert [e["kind"] for e in events] == ["state", "closed"]
    assert "API Key" in events[0]["message"]


def test_language_and_glossary_changes_are_sent_while_capturing(hub):
    async def body(s, events):
        await until(lambda: kinds(events, "state")[-1:] == [{"kind": "state", "state": "ready"}])
        s.update("it_en", [])  # nothing changed
        s.update("en", ["Fourier"])
        await until(lambda: hub.sockets[0].texts())
        s.update("any", ["Fourier"])
        await until(lambda: len(hub.sockets[0].texts()) == 2)

    run_stream(hub, PRO, "it_en", [], body)
    assert hub.sockets[0].texts() == [
        {"type": "UpdateConfiguration", "language_codes": ["en"], "keyterms_prompt": ["Fourier"]},
        {"type": "UpdateConfiguration", "language_codes": []},
    ]


# ---------------------------------------------------------------- through /ws


@pytest.fixture
def fake(monkeypatch, hub):
    t = FakeTranscriber("ready")
    monkeypatch.setattr(server, "transcriber", t)
    monkeypatch.setattr(server, "Segmenter", FakeSegmenter)
    monkeypatch.setattr(server, "SileroModel", lambda: None)
    monkeypatch.setattr(server, "load_model_in_background", lambda: None)
    return t


@contextmanager
def cloud_session(start=None, **asr):
    with TestClient(server.app, base_url="http://127.0.0.1:8765") as client, client.websocket_connect("ws://127.0.0.1:8765/ws") as sock:
        assert sock.receive_json()["type"] == "status"
        config = {"engine": "assemblyai", "assemblyai_key": " key ", **asr}
        sock.send_json({"type": "config", "asr": config, "prompt": "Corso. Fourier, Laplace."})
        sock.send_json(start or {"type": "start", "run": 1})
        yield sock


HELLO = {
    2: [turn(0, "", [("hello", 20, 60)])],
    4: [turn(0, "Hello world.", [("Hello", 20, 60), ("world.", 70, 150)], end=True, formatted=True)],
}


def test_a_cloud_run_goes_to_assemblyai_and_not_to_whisper(fake, hub):
    hub.scripts = [HELLO]
    with cloud_session() as sock:
        sock.send_bytes(CHUNK * 2)
        sock.send_bytes(CHUNK * 2)
        seen = read_until(sock, lambda m: m["type"] == "segment")
        assert [m["state"] for m in seen if m["type"] == "engine"] == ["connecting", "ready"]
        assert [(m["index"], m["text"]) for m in seen if m["type"] == "partial"] == [(1, "hello")]
        assert [m["audio_end"] for m in seen if m["type"] == "partial"] == [0.06]
        assert seen[-1] == {
            "type": "segment", "index": 1, "start": 0.02, "duration": 0.13, "lang": "en", "lang_prob": 1.0,
            "text": "Hello world.", "asr_ms": 0, "merged": 1, "run": 1, "committed": 2400, "committed_run": 1,
        }
        # 停止: the stream is closed, and everything counts as answered.
        sock.send_json({"type": "stop"})
        done = read_until(sock, lambda m: m["type"] == "engine" and m["state"] == "closed")[-1]
        assert (done["committed"], done["committed_run"]) == (3200, 1)
    assert fake.calls == 0
    assert not [m for m in seen if m["type"] in ("backlog", "error")]
    # The default model, the trimmed key, and the glossary terms without the course hint.
    assert hub.calls == [(assembly.stream_url(ENGLISH, "it_en"), "key")]
    assert hub.sockets[0].texts() == [
        {"type": "UpdateConfiguration", "keyterms_prompt": ["Fourier", "Laplace"]},
        {"type": "Terminate"},
    ]


def test_partials_can_be_switched_off(fake, hub):
    hub.scripts = [HELLO]
    with cloud_session(partial=False) as sock:
        sock.send_bytes(CHUNK * 4)
        seen = read_until(sock, lambda m: m["type"] == "segment")
        assert not [m for m in seen if m["type"] == "partial"]


def test_the_chosen_model_is_used_and_unknown_ones_are_not(fake, hub):
    with cloud_session(assemblyai_model=PRO, language="it"):
        pass
    with cloud_session(assemblyai_model="whisper-9"):
        pass
    assert [url for url, _ in hub.calls] == [assembly.stream_url(PRO, "it"), assembly.stream_url(ENGLISH, "it_en")]


def test_a_resumed_run_keeps_its_clock(fake, hub):
    hub.scripts = [HELLO]
    with cloud_session(start={"type": "resume", "run": 7, "offset": 16000}) as sock:
        sock.send_bytes(CHUNK * 4)
        segment = read_until(sock, lambda m: m["type"] == "segment")[-1]
        assert (segment["start"], segment["run"]) == (1.02, 7)
        assert (segment["committed"], segment["committed_run"]) == (16000 + 2400, 7)


def test_a_stopped_run_is_reported_done_once_its_last_turn_is_in(fake, hub):
    hub.scripts = [{2: [turn(0, "", [("so", 10, 40)])], "end": []}]
    with cloud_session() as sock:
        sock.send_bytes(CHUNK * 2)
        read_until(sock, lambda m: m["type"] == "partial")
        sock.send_json({"type": "stop"})
        sock.send_json({"type": "start", "run": 2})
        seen = read_until(sock, lambda m: m["type"] == "run_done")
        # The first run's last words arrive under its own run number, then it is done.
        assert [(m["run"], m["text"]) for m in seen if m["type"] == "segment"] == [(1, "so")]
        assert seen[-1] == {"type": "run_done", "run": 1}


def test_a_missing_key_is_an_error_on_the_page(fake, hub):
    with cloud_session(assemblyai_key="") as sock:
        seen = read_until(sock, lambda m: m["type"] == "error")
        assert "API Key" in seen[-1]["message"]
        assert [m["state"] for m in seen if m["type"] == "engine"] == ["error"]
        sock.send_bytes(CHUNK * 2)  # not sent anywhere, not transcribed locally
    assert hub.calls == [] and fake.calls == 0


def test_the_local_engine_is_the_default(fake, hub):
    with TestClient(server.app, base_url="http://127.0.0.1:8765") as client, client.websocket_connect("ws://127.0.0.1:8765/ws") as sock:
        assert sock.receive_json()["type"] == "status"
        sock.send_json({"type": "config", "asr": {"partial": False, "assemblyai_key": "key"}})
        sock.send_json({"type": "start"})
        sock.send_bytes(b"\0" * 3200)
        assert read_until(sock, lambda m: m["type"] == "segment")[-1]["text"] == "testo 1"
    assert hub.calls == []


def test_a_resumed_run_keeps_the_engine_it_was_started_with(fake, hub):
    # Captured with the local engine; AssemblyAI was chosen meanwhile (applies from the next 开始),
    # then the connection dropped: the page resumes the run and resends its kept audio.
    with cloud_session(start={"type": "resume", "run": 3, "offset": 0, "engine": "local"}) as sock:
        sock.send_bytes(b"\0" * 3200)
        assert read_until(sock, lambda m: m["type"] == "segment")[-1]["text"] == "testo 1"
    assert hub.calls == []  # nothing went to AssemblyAI


def test_a_resumed_cloud_run_keeps_its_key_and_model(fake, hub):
    pinned = {"engine": "assemblyai", "assemblyai_key": "old key", "assemblyai_model": PRO}
    with TestClient(server.app, base_url="http://127.0.0.1:8765") as client, client.websocket_connect("ws://127.0.0.1:8765/ws") as sock:
        assert sock.receive_json()["type"] == "status"
        sock.send_json({"type": "config", "asr": {"engine": "local", "assemblyai_key": "new key", "language": "it"}})
        sock.send_json({"type": "resume", "run": 3, "offset": 0, **pinned})
        read_until(sock, lambda m: m["type"] == "engine" and m["state"] == "ready")
    assert hub.calls == [(assembly.stream_url(PRO, "it"), "old key")]
    assert fake.calls == 0


# ---------------------------------------------------------------- connecting ahead of the run


def prepared_session():
    return cloud_session(start={"type": "prepare", "engine": "assemblyai", "assemblyai_key": "key"})


def test_the_connection_opened_when_start_is_pressed_serves_the_run(fake, hub):
    hub.scripts = [HELLO]
    with prepared_session() as sock:
        # Connected while the page asks for its sources: nothing is reported for it yet.
        asyncio.run(until(lambda: hub.sockets))
        sock.send_json({"type": "start", "run": 1, "engine": "assemblyai", "assemblyai_key": "key"})
        sock.send_bytes(CHUNK * 4)
        seen = read_until(sock, lambda m: m["type"] == "segment")
        assert {m["run"] for m in seen if m["type"] == "engine"} == {1}
        assert [m["state"] for m in seen if m["type"] == "engine"][-1] == "ready"
        assert (seen[-1]["text"], seen[-1]["run"]) == ("Hello world.", 1)
    assert len(hub.calls) == 1  # the run used that connection


def test_a_prepared_connection_is_closed_when_no_capture_follows(fake, hub):
    with prepared_session() as sock:
        asyncio.run(until(lambda: hub.sockets))
        sock.send_json({"type": "unprepare"})
        asyncio.run(until(lambda: {"type": "Terminate"} in hub.sockets[0].texts()))
        sock.send_json({"type": "stop"})
        seen = read_until(sock, lambda m: m["type"] == "vad")
        assert not [m for m in seen if m["type"] in ("engine", "run_done", "error")]
    assert len(hub.calls) == 1


def test_a_prepared_connection_is_not_used_for_another_key_or_engine(fake, hub):
    with prepared_session() as sock:
        asyncio.run(until(lambda: hub.sockets))
        sock.send_json({"type": "start", "run": 1, "engine": "assemblyai", "assemblyai_key": "other"})
        read_until(sock, lambda m: m["type"] == "engine" and m["state"] == "ready")
        assert [key for _, key in hub.calls] == ["key", "other"]
        assert {"type": "Terminate"} in hub.sockets[0].texts()
        sock.send_json({"type": "prepare", "engine": "assemblyai", "assemblyai_key": "key"})
        asyncio.run(until(lambda: len(hub.sockets) == 3))
        sock.send_json({"type": "start", "run": 2, "engine": "local"})
        asyncio.run(until(lambda: {"type": "Terminate"} in hub.sockets[2].texts()))
    assert fake.calls == 0
