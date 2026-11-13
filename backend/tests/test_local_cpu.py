"""The "run the local model on the CPU" setting (设置 → 翻译 → 本地模型)."""

import asyncio
import json
import os

import httpx

from translate import OLLAMA_CPU_THREADS, OLLAMA_CTX, Translator


def ollama(seen: list) -> Translator:
    """A Translator whose Ollama answers from here; the request bodies are appended to `seen`."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        if request.url.path == "/api/chat":
            lines = [{"message": {"content": "你好"}}, {"done": True}]
            return httpx.Response(200, text="".join(json.dumps(x) + "\n" for x in lines))
        return httpx.Response(200, json={})

    t = Translator()
    t.http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return t


def test_translation_keeps_the_model_off_the_gpu_only_when_asked():
    seen: list = []
    t = ollama(seen)
    for config in ({"model": "m"}, {"model": "m", "cpu": True}):
        assert asyncio.run(t.translate("ollama", config, "ciao", [], [])) == "你好"
    (_, on_gpu), (_, on_cpu) = seen
    assert "num_gpu" not in on_gpu["options"] and "num_thread" not in on_gpu["options"]
    assert on_cpu["options"]["num_gpu"] == 0
    # Not every core: the PC must stay usable while a translation is generated.
    assert on_cpu["options"]["num_thread"] == OLLAMA_CPU_THREADS
    assert 2 <= OLLAMA_CPU_THREADS <= max(2, (os.cpu_count() or 8) // 2)
    assert on_gpu["options"]["num_ctx"] == on_cpu["options"]["num_ctx"] == OLLAMA_CTX


def test_warmup_loads_the_model_where_the_setting_says():
    seen: list = []
    t = ollama(seen)
    asyncio.run(t.warmup({"model": "m", "cpu": True}))
    asyncio.run(t.warmup({"model": "m"}))
    assert [path for path, _ in seen] == ["/api/generate"] * 2
    assert seen[0][1]["options"]["num_gpu"] == 0
    assert "num_gpu" not in seen[1][1]["options"]


def test_warmup_with_reload_unloads_the_model_first():
    # Ollama leaves a model that is loaded on the CPU there, so going back to the GPU needs this.
    seen: list = []
    asyncio.run(ollama(seen).warmup({"model": "m"}, reload=True))
    unload, load = (body for _, body in seen)
    assert unload == {"model": "m", "keep_alive": 0}
    assert load["keep_alive"] != 0 and "num_gpu" not in load["options"]
