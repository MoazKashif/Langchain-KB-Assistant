"""Tests for the knowledge base assistant.

The LLM is replaced by a fake structured model so the suite runs offline
(no API key required). Retrieval, tools and guardrails are tested for real.
"""
from __future__ import annotations

import pytest

from src.agent import KnowledgeBaseAgent
from src.config import Settings
from src.models import AssistantResponse
from src.retrieval import KnowledgeBase, KnowledgeBaseRetriever
from src.tools import create_lookup_tool

BASE_DIR = Settings().knowledge_base_path.parent.parent


@pytest.fixture(scope="session")
def base() -> KnowledgeBase:
    return KnowledgeBase.from_csv(BASE_DIR / "data" / "knowledge_base.csv")


@pytest.fixture(scope="session")
def retriever(base: KnowledgeBase) -> KnowledgeBaseRetriever:
    return KnowledgeBaseRetriever(base, top_k=4)


class FakeStructuredModel:
    """Stands in for `llm.with_structured_output(AssistantResponse)`."""

    def __init__(self, response: AssistantResponse):
        self.response = response

    def invoke(self, messages) -> AssistantResponse:
        return self.response


def make_agent(
    base: KnowledgeBase,
    retriever: KnowledgeBaseRetriever,
    structured_model: object | None = None,
    min_score: float = 2.0,
) -> KnowledgeBaseAgent:
    return KnowledgeBaseAgent(
        base=base,
        retriever=retriever,
        settings=Settings(),
        structured_model=structured_model,
        min_score=min_score,
    )


# ------------------------------------------------------------------- dataset
def test_knowledge_base_has_at_least_20_records(base: KnowledgeBase):
    assert len(base.records) >= 20


def test_records_have_all_required_fields(base: KnowledgeBase):
    required = {
        "record_id", "title", "category", "description",
        "keywords", "source", "last_updated",
    }
    for record in base.records:
        for field in required:
            assert getattr(record, field), f"{record.record_id} missing {field}"


def test_record_ids_are_unique(base: KnowledgeBase):
    assert len(base.ids) == len(set(base.ids))


# ----------------------------------------------------------------- retrieval
def test_retriever_exact_record_id(retriever: KnowledgeBaseRetriever):
    results = retriever.retrieve("What does KB-0001 say about remote work?")
    assert results[0].record.record_id == "KB-0001"


def test_retriever_keyword_match(retriever: KnowledgeBaseRetriever):
    results = retriever.retrieve("What is the password policy?")
    assert results[0].record.record_id == "KB-0005"


def test_retriever_category_filter(retriever: KnowledgeBaseRetriever):
    results = retriever.retrieve("health insurance coverage benefits")
    assert results[0].record.record_id == "KB-0016"


def test_retriever_low_scores_for_unsupported(retriever: KnowledgeBaseRetriever):
    results = retriever.retrieve("how do I book a flight to the moon?")
    assert all(r.score < 5.0 for r in results)


# --------------------------------------------------------------------- tools
def test_lookup_tool_valid_id(base: KnowledgeBase):
    tool = create_lookup_tool(base)
    out = tool.invoke({"record_id": "kb-0012"})
    assert "KB-0012" in out
    assert "Product Warranty Terms" in out


def test_lookup_tool_invalid_id(base: KnowledgeBase):
    tool = create_lookup_tool(base)
    out = tool.invoke({"record_id": "KB-9999"})
    assert "No knowledge record found" in out


# ------------------------------------------------------------------- guardrail
def test_guardrail_unsupported_question(base: KnowledgeBase, retriever: KnowledgeBaseRetriever):
    agent = make_agent(base, retriever)
    result = agent.answer("How do I book a flight to the moon?")
    assert result.needs_human_review is True
    assert result.matched_records == []
    assert result.sources == []
    assert result.confidence == 0.0
    assert "not available" in result.answer.lower()


def test_guardrail_invalid_record_id(base: KnowledgeBase, retriever: KnowledgeBaseRetriever):
    agent = make_agent(base, retriever)
    result = agent.answer("Please show me the full content of KB-9999.")
    assert result.needs_human_review is True
    assert "not available" in result.answer.lower()


# --------------------------------------------------------------- full pipeline
def test_agent_uses_lookup_tool_for_exact_id(
    base: KnowledgeBase, retriever: KnowledgeBaseRetriever
):
    canned = AssistantResponse(
        answer="Employees may work remotely up to 4 days per week.",
        matched_records=["KB-0001"],
        sources=["HR Handbook v4.2"],
        confidence=0.95,
        needs_human_review=False,
    )
    agent = make_agent(base, retriever, structured_model=FakeStructuredModel(canned))
    result = agent.answer_with_meta("What is the remote work policy in KB-0001?")

    assert result.tool_calls
    assert result.tool_calls[0].name == "lookup_record_by_id"
    assert result.tool_calls[0].arguments == {"record_id": "KB-0001"}
    assert result.response == canned
    assert result.request_id.startswith("req-")


def test_agent_calls_lookup_for_each_id_in_comparison(
    base: KnowledgeBase, retriever: KnowledgeBaseRetriever
):
    canned = AssistantResponse(
        answer="Comparison of remote work and annual leave.",
        matched_records=["KB-0001", "KB-0002"],
        sources=["HR Handbook v4.2"],
        confidence=0.9,
        needs_human_review=False,
    )
    agent = make_agent(base, retriever, structured_model=FakeStructuredModel(canned))
    result = agent.answer_with_meta("Compare KB-0001 with KB-0002.")

    args = [tc.arguments["record_id"] for tc in result.tool_calls]
    assert args == ["KB-0001", "KB-0002"]


def test_agent_falls_back_to_guardrail_when_structured_output_fails(
    base: KnowledgeBase, retriever: KnowledgeBaseRetriever
):
    class FailingModel:
        def invoke(self, messages):
            return None

    agent = make_agent(base, retriever, structured_model=FailingModel())
    result = agent.answer("What is the remote work policy in KB-0001?")

    assert result.needs_human_review is True
    assert result.confidence == 0.0
    assert "not available" in result.answer.lower()


def test_agent_does_not_call_tool_without_id(
    base: KnowledgeBase, retriever: KnowledgeBaseRetriever
):
    canned = AssistantResponse(
        answer="Password policy requires 12+ characters.",
        matched_records=["KB-0005"],
        sources=["IT Security Policy v3.0"],
        confidence=0.9,
        needs_human_review=False,
    )
    agent = make_agent(base, retriever, structured_model=FakeStructuredModel(canned))
    result = agent.answer_with_meta("What are the password requirements?")

    assert result.tool_calls == []
    assert result.retrieved_records[0] == "KB-0005"


def test_sources_are_reconciled_from_matched_records(
    base: KnowledgeBase, retriever: KnowledgeBaseRetriever
):
    canned = AssistantResponse(
        answer="You accrue 25 leave days.",
        matched_records=["KB-0002"],
        sources=["vacation"],
        confidence=0.9,
        needs_human_review=False,
    )
    agent = make_agent(base, retriever, structured_model=FakeStructuredModel(canned))
    result = agent.answer("How many leave days do I get?")

    assert result.sources == ["HR Handbook v4.2"]


def test_token_usage_is_captured(
    base: KnowledgeBase, retriever: KnowledgeBaseRetriever
):
    canned = AssistantResponse(
        answer="Annual leave accrues 25 days.",
        matched_records=["KB-0002"],
        sources=["HR Handbook v4.2"],
        confidence=0.9,
        needs_human_review=False,
    )
    object.__setattr__(canned, "usage_metadata", {"total_tokens": 42, "prompt_tokens": 30, "completion_tokens": 12})
    agent = make_agent(base, retriever, structured_model=FakeStructuredModel(canned))
    result = agent.answer_with_meta("How many leave days do I get?")

    assert result.token_usage == {"total_tokens": 42, "prompt_tokens": 30, "completion_tokens": 12}
    assert result.execution_time_ms >= 0
