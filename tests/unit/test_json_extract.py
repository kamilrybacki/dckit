"""Robust JSON-array / JSON-object extraction."""

from __future__ import annotations

import json

import pytest

from dckit._internal.json_extract import extract_json_array, extract_json_object


@pytest.mark.unit
def test_extract_array_handles_prose_around_json() -> None:
    text = "Sure! Here is the result:\n[{\"a\": 1}, {\"a\": 2}]\nThanks."
    assert extract_json_array(text) == [{"a": 1}, {"a": 2}]


@pytest.mark.unit
def test_extract_array_ignores_brackets_in_strings() -> None:
    text = '[{"name": "with ] bracket"}]'
    assert extract_json_array(text) == [{"name": "with ] bracket"}]


@pytest.mark.unit
def test_extract_array_raises_on_missing() -> None:
    with pytest.raises(json.JSONDecodeError):
        extract_json_array("no array here")


@pytest.mark.unit
def test_extract_object_basic() -> None:
    text = "Verdict: {\"area_index\": 3, \"confidence\": \"high\"}"
    assert extract_json_object(text) == {"area_index": 3, "confidence": "high"}


@pytest.mark.unit
def test_extract_object_handles_braces_in_strings() -> None:
    text = '{"k": "with } brace"}'
    assert extract_json_object(text) == {"k": "with } brace"}
