"""Robust JSON-array extraction from LLM responses."""

from __future__ import annotations

import json
from typing import Any


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extract first balanced [...] block from LLM output.

    Handles prose around the JSON and ignores `]` inside string literals.
    Raises json.JSONDecodeError on malformed input.
    """
    start = text.find("[")
    if start < 0:
        raise json.JSONDecodeError("no '[' found", text, 0)
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : i + 1])
                if not isinstance(parsed, list):
                    raise json.JSONDecodeError("expected array", text, start)
                return parsed
    raise json.JSONDecodeError("unbalanced brackets", text, start)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract first balanced {...} block."""
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("no '{' found", text, 0)
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parsed = json.loads(text[start : i + 1])
                if not isinstance(parsed, dict):
                    raise json.JSONDecodeError("expected object", text, start)
                return parsed
    raise json.JSONDecodeError("unbalanced braces", text, start)
