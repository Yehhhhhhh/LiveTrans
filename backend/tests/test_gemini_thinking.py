"""Gemini models are asked not to think, or to think as little as they allow."""

import asyncio
import json

import httpx

from translate import Translator

DONE = 'data: {"choices":[{"delta":{"content":"你好"}}]}\n\ndata: [DONE]\n\n'


def endpoint(seen: list, rejected: set) -> Translator:
    """An OpenAI-compatible endpoint that rejects the reasoning_effort values in `rejected`."""

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body.get("reasoning_effort"))
        if body.get("reasoning_effort") in rejected:
            return httpx.Response(400, text="Budget 0 is invalid. This model only works in thinking mode.")
        return httpx.Response(200, text=DONE)

    t = Translator()
    t.http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return t


def run(t: Translator, model: str) -> str:
    return asyncio.run(t.translate("openai", {"baseUrl": "https://x/v1", "model": model}, "ciao", [], []))


def test_gemini_thinking_is_switched_off():
    seen: list = []
    assert run(endpoint(seen, set()), "gemini-2.5-flash") == "你好"
    assert seen == ["none"]


def test_a_model_that_must_think_gets_the_lowest_level_it_takes_and_it_is_remembered():
    seen: list = []
    t = endpoint(seen, {"none", "minimal"})
    assert run(t, "models/gemini-3-pro") == "你好"
    assert run(t, "models/gemini-3-pro") == "你好"
    assert seen == ["none", "minimal", "low", "low"]


def test_without_any_level_the_parameter_is_left_out():
    seen: list = []
    t = endpoint(seen, {"none", "minimal", "low"})
    assert run(t, "gemini-x") == "你好"
    assert seen == ["none", "minimal", "low", None]


def test_other_models_are_left_alone():
    seen: list = []
    assert run(endpoint(seen, set()), "gpt-5.5") == "你好"
    assert seen == [None]
