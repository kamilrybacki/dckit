"""LLM adapter — any OpenAI-compatible chat endpoint.

Wraps prompts in delimited sections, validates JSON output, strips Markdown
code fences. Used by `discover` (proposing macro pairs) and `OracleJudge`
(area selection).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, runtime_checkable

import httpx

from ..exceptions import LLMResponseError

_logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


def strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrapping from LLM output."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


def parse_json_response(text: str) -> Any:
    """Parse LLM output as JSON, tolerating Markdown fences.

    Raises LLMResponseError with the offending payload on parse failure.
    """
    cleaned = strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            f"LLM output is not valid JSON: {exc.msg} (payload: {cleaned[:200]!r})"
        ) from exc


@runtime_checkable
class LLMClient(Protocol):
    """Chat-completion adapter."""

    def chat(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system: str | None = None,
    ) -> str: ...


class OpenAICompatLLM:
    """Reference impl for OpenAI-compatible HTTP endpoints (OmniRoute, vLLM,
    Ollama with /v1/chat/completions, OpenRouter, the OpenAI API itself).
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    def chat(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system: str | None = None,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.post(
            f"{self._base_url}/v1/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        response.raise_for_status()
        body = response.json()
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"unexpected chat response shape: {body!r}") from exc

    def close(self) -> None:
        self._client.close()


def make_delimited_prompt(
    instruction: str,
    sections: dict[str, str],
    *,
    output_format: str = "JSON object",
) -> str:
    """Build a prompt with explicit delimiters around input sections.

    Mitigates prompt-injection from user-supplied content and helps the LLM
    distinguish instructions from data.
    """
    parts = [instruction.strip(), ""]
    for name, body in sections.items():
        parts.append(f"=== BEGIN {name} ===")
        parts.append(body.strip())
        parts.append(f"=== END {name} ===")
        parts.append("")
    parts.append(f"Respond with ONLY a {output_format}. No prose. No fences.")
    return "\n".join(parts)
