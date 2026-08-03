"""Pydantic schemas for knowledge records and the assistant's structured output."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class KnowledgeRecord(BaseModel):
    """A single knowledge base entry."""

    record_id: str
    title: str
    category: str
    description: str
    keywords: list[str]
    source: str
    last_updated: str

    def to_search_text(self) -> str:
        """Flat text used for splitting and lexical retrieval."""
        return (
            f"{self.record_id}\n{self.title}\n{self.category}\n"
            f"{self.description}\nKeywords: {', '.join(self.keywords)}"
        )

    def to_context_block(self) -> str:
        """Formatted evidence block handed to the model."""
        return (
            f"[{self.record_id}] {self.title} ({self.category})\n"
            f"{self.description}\n"
            f"Keywords: {', '.join(self.keywords)}\n"
            f"Source: {self.source} | Last updated: {self.last_updated}"
        )


class RetrievedRecord(BaseModel):
    """A knowledge record returned by the retriever with its relevance score."""

    record: KnowledgeRecord
    score: float


class ToolCall(BaseModel):
    """A recorded tool invocation for observability."""

    name: str
    arguments: dict
    result_summary: str


class AssistantResponse(BaseModel):
    """Structured output returned to the user.

    `matched_records` lists the record IDs that served as evidence.
    `sources` lists the human-readable source documents.
    `confidence` is a 0-1 estimate of answer reliability.
    `needs_human_review` is true when evidence is insufficient or uncertain.
    """

    answer: str = Field(description="The final answer to the user's question.")
    matched_records: list[str] = Field(
        description="Record IDs used as evidence for the answer."
    )
    sources: list[str] = Field(
        description="Human-readable source documents cited in the answer."
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1 that the answer is correct."
    )
    needs_human_review: bool = Field(
        description="True when evidence is insufficient or the answer is uncertain."
    )
