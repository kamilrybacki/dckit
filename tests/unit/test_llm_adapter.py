"""LLM adapter helpers — fence stripping, JSON parsing, prompt delimiters."""

from __future__ import annotations

import pytest

from dckit.adapters.llm import make_delimited_prompt, parse_json_response, strip_fences
from dckit.exceptions import LLMResponseError


@pytest.mark.unit
def test_strip_fences_json_label() -> None:
    out = strip_fences('```json\n{"a": 1}\n```')
    assert out == '{"a": 1}'


@pytest.mark.unit
def test_strip_fences_no_label() -> None:
    out = strip_fences('```\n[1, 2]\n```')
    assert out == "[1, 2]"


@pytest.mark.unit
def test_strip_fences_unwrapped_passthrough() -> None:
    out = strip_fences('  {"clean": true}  ')
    assert out == '{"clean": true}'


@pytest.mark.unit
def test_parse_json_response_handles_fences() -> None:
    payload = parse_json_response('```json\n{"k": 16}\n```')
    assert payload == {"k": 16}


@pytest.mark.unit
def test_parse_json_response_raises_on_invalid() -> None:
    with pytest.raises(LLMResponseError):
        parse_json_response("not json {{")


@pytest.mark.unit
def test_make_delimited_prompt_includes_sections() -> None:
    prompt = make_delimited_prompt(
        "Classify the passage.",
        {"PASSAGE": "Hello world.", "AREAS": "[0] foo\n[1] bar"},
    )
    assert "=== BEGIN PASSAGE ===" in prompt
    assert "=== END PASSAGE ===" in prompt
    assert "=== BEGIN AREAS ===" in prompt
    assert "Respond with ONLY a JSON object" in prompt
