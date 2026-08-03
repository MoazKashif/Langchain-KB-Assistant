"""Resilient structured output generation.

Llama-class models on Groq occasionally emit JSON array fields as comma
separated strings (e.g. `"matched_records": "[KB-0011, KB-0012]"`). Strict
tool-calling structured output then fails API-side with a 400. Instead we use
`json_mode` (no API-side schema enforcement), parse defensively, repair common
type mistakes, and retry once with corrective feedback before giving up.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

_SCHEMA_INSTRUCTION = """\
Respond with a single JSON object matching the schema below and nothing else.
- answer: string
- matched_records: array of strings; use ONLY record IDs that appear in the
  <context> blocks, written exactly like "KB-0001"
- sources: array of strings; use ONLY the Source document names shown in the
  context blocks (e.g. "HR Handbook v4.2"). Never use keywords or record titles.
- confidence: number between 0 and 1
- needs_human_review: boolean

Schema:
{schema}

Return ONLY the JSON object. matched_records and sources MUST be JSON arrays,
for example ["KB-0001", "KB-0002"], never a comma-separated string.
"""


def _extract_json(text: str) -> Any:
    """Pull the first JSON object out of a model response (strips fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in: {text[:200]}")
    return json.loads(text[start : end + 1])


def _coerce_list(value: Any) -> list[str]:
    """Coerce a JSON array field that arrived as a string into a list."""
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    cleaned = str(value).strip()
    if not cleaned:
        return []
    if any(c in cleaned for c in "[],;\""):
        items = [p.strip().strip("\"'") for p in re.split(r"[,\[\]\";]+", cleaned) if p.strip()]
        return items
    return [cleaned]


def _coerce(data: dict) -> dict:
    """Repair common type mistakes before pydantic validation."""
    for key in ("matched_records", "sources"):
        if key in data:
            data[key] = _coerce_list(data[key])
    if isinstance(data.get("confidence"), str):
        try:
            data["confidence"] = float(data["confidence"])
        except ValueError:
            data.pop("confidence", None)
    if isinstance(data.get("needs_human_review"), str):
        data["needs_human_review"] = data["needs_human_review"].strip().lower() in {
            "true",
            "yes",
            "1",
        }
    return data


class StructuredGenerator:
    """Wraps a chat model with json-mode structured output + repair + retry.

    `invoke(messages)` returns a tuple `(AssistantResponse, usage_metadata)`
    or `(None, usage_metadata)` if generation/parsing keeps failing.
    """

    def __init__(self, llm, schema: type[BaseModel], max_retries: int = 1):
        self.schema = schema
        self.max_retries = max_retries
        try:
            self._runnable = llm.with_structured_output(
                schema, method="json_mode", include_raw=True
            )
        except Exception:
            self._runnable = llm.with_structured_output(schema, include_raw=True)

    def invoke(self, messages: list) -> tuple[Any, dict | None]:
        work = list(messages)
        work.append(
            (
                "system",
                _SCHEMA_INSTRUCTION.format(
                    schema=json.dumps(self.schema.model_json_schema())
                ),
            )
        )
        last_error: str | None = None
        usage: dict | None = None

        for _ in range(self.max_retries + 1):
            if last_error:
                work.append(
                    (
                        "system",
                        f"Your previous response was not valid: {last_error}. "
                        "Return ONLY a JSON object that conforms exactly to the schema.",
                    )
                )
            out = self._runnable.invoke(work)
            raw = out.get("raw")
            parsed = out.get("parsed")
            parsing_error = out.get("parsing_error")
            usage = getattr(raw, "usage_metadata", None) or usage

            if parsed is not None:
                return self.schema.model_validate(parsed), usage
            if raw is not None and raw.content:
                try:
                    data = _coerce(_extract_json(raw.content))
                    return self.schema.model_validate(data), usage
                except Exception as exc:  # noqa: BLE001 - any parse/validate error
                    last_error = f"{parsing_error or exc}"[:400]
                    continue
            last_error = f"{parsing_error or 'empty model output'}"[:400]

        return None, usage
