"""Tools exposed to the agent.

- `lookup_record_by_id`: exact retrieval of a single record by its record ID,
  used when the user asks about a specific known record.
"""
from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from .models import KnowledgeRecord
from .retrieval import KnowledgeBase

_UNKNOWN_MESSAGE = "No knowledge record found with ID {record_id}. Valid IDs include {sample}."


def _format_record(record: KnowledgeRecord) -> str:
    return record.to_context_block()


def create_lookup_tool(base: KnowledgeBase) -> BaseTool:
    """Create the exact-record lookup tool bound to a knowledge base."""

    valid_ids = base.ids
    sample_ids = ", ".join(valid_ids[:6]) + ("..." if len(valid_ids) > 6 else "")

    @tool("lookup_record_by_id")
    def lookup_record_by_id(record_id: str) -> str:
        """Look up a single knowledge record by its exact record ID (e.g. 'KB-0001').
        Use this when the user references a specific record ID or asks for the
        exact content of one record."""
        record = base.get_by_id(record_id)
        if record is None:
            return _UNKNOWN_MESSAGE.format(record_id=record_id, sample=sample_ids)
        return _format_record(record)

    return lookup_record_by_id
