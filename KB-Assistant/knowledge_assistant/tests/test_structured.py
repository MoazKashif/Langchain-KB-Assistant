"""Tests for the resilient structured-output helpers (repair logic)."""
from __future__ import annotations

from src.structured import _coerce, _coerce_list, _extract_json


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_fence():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_surrounded_by_text():
    assert _extract_json('Here you go: {"answer": "x"} hope that helps') == {"answer": "x"}


def test_coerce_string_list_comma_separated():
    assert _coerce_list("[KB-0011, KB-0012, KB-0013]") == ["KB-0011", "KB-0012", "KB-0013"]


def test_coerce_string_list_json_style():
    assert _coerce_list('"KB-0001", "KB-0002"') == ["KB-0001", "KB-0002"]


def test_coerce_string_list_single_value():
    assert _coerce_list("HR Handbook v4.2") == ["HR Handbook v4.2"]


def test_coerce_list_passthrough():
    assert _coerce_list(["KB-0001", "KB-0002"]) == ["KB-0001", "KB-0002"]


def test_coerce_full_dict():
    data = _coerce(
        {
            "answer": "ok",
            "matched_records": "[KB-0011, KB-0012]",
            "sources": '"HR Handbook v4.2"',
            "confidence": "0.9",
            "needs_human_review": "true",
        }
    )
    assert data["matched_records"] == ["KB-0011", "KB-0012"]
    assert data["sources"] == ["HR Handbook v4.2"]
    assert data["confidence"] == 0.9
    assert data["needs_human_review"] is True
