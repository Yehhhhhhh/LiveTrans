"""Translation providers: any OpenAI-compatible endpoint, or Claude via the Anthropic SDK."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

import anthropic
import httpx

SYSTEM_PROMPT = """你是一名专业的同声传译员，正在为一门通信工程（telecommunications）大学课程做实时字幕翻译。
讲者混用意大利语和英语，你要把每一段讲话翻译成自然、准确、简洁的简体中文。

要求：
- 只输出译文本身，不要解释、注释、引号或任何前缀。
- 原文来自语音识别，可能有误识别、断句不完整或同音错词；请结合上文和专业知识理解原意后翻译，不要逐字硬译。
- 术语优先使用给出的术语表译法；常见缩写（如 OFDM、MIMO、SNR、QAM）保留英文缩写。
- 公式、变量和数字按原样保留，例如 "E b su N zero" 写作 Eb/N0。
- 念出来的希腊字母写成符号：rho→ρ，lambda→λ，mu→μ，sigma→σ，alpha→α，beta→β，tau→τ，theta→θ，phi→φ，omega→ω，delta→δ，pi→π 等。
- 念出来的公式写成数学式，不要用文字复述运算：例如 "rho divided by 1 minus rho" / "rho fratto uno meno rho" 写作 ρ/(1−ρ)，"lambda over mu" 写作 λ/μ，"e to the minus x" 写作 e^(−x)，"x squared" 写作 x²，"uno meno rho per rho alla n" 写作 (1−ρ)ρⁿ。
- 简单式子用上面的纯文本写法；求和、积分、复杂分式、上下标较多的公式改用 LaTeX，并且只用 \\( 和 \\) 包裹（界面会渲染），例如 "the sum for i from 10 to 150 of p i" 写作 \\(\\sum_{i=10}^{150} p_i\\)。不要使用其他 Markdown 格式。
- 如果这一段只是口头语（如 "allora"、"ok"、"quindi"），给出简短的对应中文即可。
- 上文只用于理解语境，不要重复翻译上文。"""

# Models that accept output_config.effort and the server-side refusal fallback.
EFFORT_MODEL = re.compile(r"^claude-(opus|sonnet|haiku|fable|mythos)-5")
FALLBACK_MODEL = re.compile(r"^claude-(opus-5|sonnet-5-5|fable-5)")


OLLAMA_URL = "http://127.0.0.1:11434"
# Glossary + context + sentence fit easily; a small window keeps VRAM free for Whisper.
OLLAMA_CTX = 4096
OLLAMA_KEEP_ALIVE = "1h"


class TranslateError(Exception):
    pass


def ollama_error(status: int, text: str, model: str) -> str:
    if status == 404 and "not found" in text:
        return f"本地没有模型 {model}，请先运行：ollama pull {model}"
    return f"Ollama HTTP {status}：{text[:300]}"


def build_user_message(text: str, context: list[dict], glossary: list[dict]) -> str:
    parts = []
    if glossary:
        lines = [f"- {g.get('it') or '-'} / {g.get('en') or '-'} → {g.get('zh') or '-'}" for g in glossary]
        parts.append("术语表（意大利语 / 英语 → 中文）：\n" + "\n".join(lines))
    if context:
        lines = []
        for c in context:
            lines.append(f"原文：{c.get('src', '')}")
            # Earlier segments still being translated (translations run in parallel) come without one.
            if c.get("tgt"):
                lines.append(f"译文：{c['tgt']}")
        parts.append("上文（仅供参考）：\n" + "\n".join(lines))
    parts.append("请翻译这一段：\n" + text)
    return "\n\n".join(parts)


def clean_output(text: str) -> str:
    # Reasoning models served via Ollama etc. may inline their thoughts.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.strip()
    for prefix in ("译文：", "译文:", "翻译：", "翻译:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def clean_partial(text: str) -> str:
    """clean_output for a stream still in progress: hide an unfinished <think> block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:
        text = text[: text.index("<think>")]
    stripped = text.lstrip()
    if stripped and "<think>".startswith(stripped):
        return ""
    return clean_output(text)


class Translator:
    def __init__(self) -> None:
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        self._claude_clients: dict[str, anthropic.AsyncAnthropic] = {}
        self._no_temperature: set[tuple[str, str]] = set()

    async def close(self) -> None:
        await self.http.aclose()
        for c in self._claude_clients.values():
            await c.close()

    async def translate(self, provider: str, config: dict, text: str, context: list[dict], glossary: list[dict]) -> str:
        out = "".join([d async for d in self.translate_stream(provider, config, text, context, glossary)])
        return clean_output(out)

    def translate_stream(
        self, provider: str, config: dict, text: str, context: list[dict], glossary: list[dict]
    ) -> AsyncIterator[str]:
        """Yield raw text deltas of the translation as the model produces them."""
        user = build_user_message(text, context, glossary)
        if provider == "claude":
            return self._claude(config, user)
        if provider == "openai":
            return self._openai(config, user)
        if provider == "ollama":
            return self._ollama(config, user)
        raise TranslateError(f"未知服务商：{provider}")

    @staticmethod
    def _ollama_base(config: dict) -> str:
        # Accept the OpenAI-style URL too; the native API lives at the root.
        return re.sub(r"/v1/?$", "", (config.get("baseUrl") or OLLAMA_URL).rstrip("/"))

    async def warmup(self, config: dict) -> None:
        """Load the local model into VRAM now and keep it there, so the first sentence isn't slow."""
        model = config.get("model") or ""
        if not model:
            raise TranslateError("请填写本地模型名")
        try:
            r = await self.http.post(
                f"{self._ollama_base(config)}/api/generate",
                json={"model": model, "keep_alive": OLLAMA_KEEP_ALIVE, "options": {"num_ctx": OLLAMA_CTX}},
            )
        except httpx.HTTPError as e:
            raise TranslateError(f"无法连接 Ollama（{e}），请确认 Ollama 已安装并在运行") from e
        if r.status_code >= 400:
            raise TranslateError(ollama_error(r.status_code, r.text, model))

    async def _ollama(self, config: dict, user: str) -> AsyncIterator[str]:
        model = config.get("model") or ""
        if not model:
            raise TranslateError("请填写本地模型名")
        body = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "stream": True,
            "think": False,
            # Stay loaded between sentences: a reload adds seconds; freed after an hour idle.
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0.2, "num_ctx": OLLAMA_CTX},
        }
        url = f"{self._ollama_base(config)}/api/chat"
        try:
            for _ in range(2):
                async with self.http.stream("POST", url, json=body) as r:
                    if r.status_code >= 400:
                        err = (await r.aread()).decode("utf-8", "replace")
                        if r.status_code == 400 and "think" in body and "think" in err:
                            del body["think"]  # Older Ollama or a model without a thinking switch.
                            continue
                        raise TranslateError(ollama_error(r.status_code, err, model))
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except ValueError as e:
                            raise TranslateError(f"无法解析返回：{line[:300]}") from e
                        if data.get("error"):
                            raise TranslateError(f"Ollama：{data['error']}")
                        delta = (data.get("message") or {}).get("content")
                        if delta:
                            yield delta
                        if data.get("done"):
                            return
                    return
        except httpx.HTTPError as e:
            raise TranslateError(f"无法连接 Ollama（{e}），请确认 Ollama 已安装并在运行") from e

    async def _openai(self, config: dict, user: str) -> AsyncIterator[str]:
        base = (config.get("baseUrl") or "").rstrip("/")
        model = config.get("model") or ""
        if not base or not model:
            raise TranslateError("请在设置中填写 Base URL 和模型名")
        headers = {"Content-Type": "application/json"}
        if config.get("apiKey"):
            headers["Authorization"] = f"Bearer {config['apiKey']}"
        body = {
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            "stream": True,
        }
        key = (base, model)
        if key not in self._no_temperature:
            body["temperature"] = 0.2
        try:
            for _ in range(2):
                async with self.http.stream("POST", f"{base}/chat/completions", json=body, headers=headers) as r:
                    if r.status_code >= 400:
                        err = (await r.aread()).decode("utf-8", "replace")
                        if r.status_code == 400 and "temperature" in body and "temperature" in err:
                            # Some models (e.g. reasoning models) only accept the default temperature.
                            self._no_temperature.add(key)
                            del body["temperature"]
                            continue
                        raise TranslateError(f"HTTP {r.status_code}：{err[:300]}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        try:
                            choices = json.loads(data).get("choices") or []
                        except ValueError as e:
                            raise TranslateError(f"无法解析返回：{data[:300]}") from e
                        # Reasoning text (delta.reasoning_content) is skipped; only the answer is shown.
                        delta = (choices[0].get("delta") or {}).get("content") if choices else None
                        if delta:
                            yield delta
                    return
        except httpx.HTTPError as e:
            raise TranslateError(f"连接失败：{e}") from e

    def _claude_client(self, api_key: str) -> anthropic.AsyncAnthropic:
        client = self._claude_clients.get(api_key)
        if client is None:
            client = anthropic.AsyncAnthropic(api_key=api_key, timeout=60.0, max_retries=1)
            self._claude_clients[api_key] = client
        return client

    async def _claude(self, config: dict, user: str) -> AsyncIterator[str]:
        api_key = config.get("apiKey") or ""
        model = config.get("model") or "claude-opus-5-5"
        if not api_key:
            raise TranslateError("请在设置中填写 Claude API Key")
        client = self._claude_client(api_key)
        kwargs = dict(
            model=model,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        if EFFORT_MODEL.match(model):
            kwargs["output_config"] = {"effort": config.get("effort") or "low"}
        if FALLBACK_MODEL.match(model):
            stream = client.beta.messages.stream(**kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        else:
            stream = client.messages.stream(**kwargs)
        try:
            async with stream as s:
                async for delta in s.text_stream:
                    yield delta
                resp = await s.get_final_message()
        except anthropic.AuthenticationError as e:
            raise TranslateError("Claude API Key 无效") from e
        except anthropic.RateLimitError as e:
            raise TranslateError("Claude 请求过于频繁（429），请稍后重试") from e
        except anthropic.APIStatusError as e:
            raise TranslateError(f"Claude HTTP {e.status_code}：{e.message}") from e
        except anthropic.APIConnectionError as e:
            raise TranslateError(f"连接 Claude 失败：{e}") from e
        if resp.stop_reason == "refusal":
            raise TranslateError("Claude 拒绝了这段内容")
