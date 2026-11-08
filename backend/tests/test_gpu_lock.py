"""A cancelled transcription must keep gpu_lock until its (uncancellable) thread has finished."""

import asyncio
import threading

import server


def test_cancelled_call_keeps_the_lock_until_its_thread_finishes():
    async def scenario():
        lock = asyncio.Lock()
        started, release = threading.Event(), threading.Event()
        overlap = []
        running = [0]

        def whisper_call():
            running[0] += 1
            overlap.append(running[0])
            started.set()
            release.wait(5)
            running[0] -= 1
            return "done"

        async def transcribe():
            async with lock:
                return await server.to_thread_to_end(whisper_call)

        first = asyncio.create_task(transcribe())
        while not started.is_set():
            await asyncio.sleep(0.01)
        first.cancel()  # the page disconnected mid-call
        await asyncio.sleep(0.1)
        assert lock.locked(), "lock released while the thread still runs"

        second = asyncio.create_task(transcribe())  # e.g. the page reconnected
        await asyncio.sleep(0.1)
        assert not second.done()  # waits for the lock

        release.set()
        assert await second == "done"
        assert first.cancelled()
        assert max(overlap) == 1  # the two calls never ran at the same time

    asyncio.run(scenario())


def test_normal_call_returns_its_result():
    async def scenario():
        return await server.to_thread_to_end(lambda a, b: a + b, 2, 3)

    assert asyncio.run(scenario()) == 5
