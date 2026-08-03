"""Agent pipeline: retrieval + exact-ID lookup tool + structured generation.

Flow (a small LangGraph state machine):

    START -> retrieve (BM25 + optional lookup tool) -> decide
        -> generate  (structured LLM output)
        -> guardrail (insufficient evidence -> needs_human_review=True)

Observability: every request logs a request ID, the question, retrieved record
IDs, tool calls, token usage, execution time and errors. LangSmith tracing is
enabled only when credentials are present.
"""
from __future__ import annotations

import logging
import os
import re
import time
import uuid
from typing import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .config import Settings
from .models import AssistantResponse, ToolCall
from .retrieval import KnowledgeBase, KnowledgeBaseRetriever
from .structured import StructuredGenerator
from .tools import create_lookup_tool

logger = logging.getLogger("kb_assistant")

_RECORD_ID_PATTERN = re.compile(r"\b(KB-\d{4})\b", re.IGNORECASE)

SYSTEM_PROMPT = """\
You are a knowledge base assistant. Answer the user's question using ONLY the
evidence provided in the <context> blocks below. Do not use outside knowledge.

Rules:
1. If the evidence fully answers the question, answer concisely and cite the
   matched record IDs in matched_records and their Source document names in
   sources (use the exact "Source:" values from the context blocks, e.g.
   "HR Handbook v4.2" — never keywords or record titles).
2. Set needs_human_review=true ONLY when the question is genuinely ambiguous
   (e.g. "what is the policy?" with no subject) or the evidence is missing or
   contradictory. For "list/filter" questions, listing the relevant policies is
   a normal answer — do NOT request review just because several records apply.
3. Ignore context blocks that are not relevant to the question. Having extra
   irrelevant blocks is not a reason to request human review.
4. If the evidence is insufficient or the requested record does not exist, state
   that the information is unavailable in the knowledge base, set
   matched_records=[] and sources=[], confidence=0.0, and
   needs_human_review=true.
5. confidence reflects how well the evidence supports the answer (0.0-1.0).
6. When exact record ID(s) were requested and those records were found, prefer
   the exact lookup results as the primary source.
7. matched_records and sources are JSON arrays of strings, e.g.
   ["KB-0001", "KB-0002"]. Never return them as comma-separated strings.
"""


def setup_logging(settings: Settings) -> None:
    """Configure console + rotating file logging once per process."""
    root = logging.getLogger("kb_assistant")
    if root.handlers:
        return
    root.setLevel(settings.log_level.upper())
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    if settings.log_file:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)


def configure_langsmith(settings: Settings) -> bool:
    """Enable LangSmith tracing only when credentials are present."""
    if settings.has_langsmith:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
        logger.info("LangSmith tracing enabled (project=%s)", settings.langchain_project)
        return True
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    logger.info("LangSmith tracing disabled (no API key configured)")
    return False


def create_llm(settings: Settings) -> BaseChatModel:
    """Create the chat model for the configured provider."""
    provider = settings.model_provider
    if provider == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        return ChatGroq(model=settings.model_name, api_key=settings.groq_api_key, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.model_name, api_key=settings.openai_api_key or None, temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.model_name, api_key=settings.anthropic_api_key or None, temperature=0)
    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(model=settings.model_name, base_url=settings.ollama_base_url, temperature=0)
    raise ValueError(f"Unsupported MODEL_PROVIDER: {provider}")


class AgentState(TypedDict):
    question: str
    request_id: str
    context_blocks: list[str]
    record_ids: list[str]
    retrieved_scores: list[float]
    tool_calls: list[ToolCall]
    token_usage: dict | None
    response: AssistantResponse | None


class AgentResult(BaseModel):
    """Full result of a request, including observability metadata."""

    request_id: str
    question: str
    response: AssistantResponse
    retrieved_records: list[str]
    tool_calls: list[ToolCall]
    execution_time_ms: float
    token_usage: dict | None


def _insufficient_response(question: str) -> AssistantResponse:
    return AssistantResponse(
        answer=(
            "The requested information is not available in the knowledge base, "
            "or the question is too vague to answer from it. No sufficient "
            "evidence was found for: "
            f"'{question}'. A human review is recommended."
        ),
        matched_records=[],
        sources=[],
        confidence=0.0,
        needs_human_review=True,
    )


class KnowledgeBaseAgent:
    """Tool-using, retrieval-augmented agent with structured output."""

    def __init__(
        self,
        base: KnowledgeBase,
        retriever: KnowledgeBaseRetriever,
        settings: Settings | None = None,
        llm: BaseChatModel | None = None,
        structured_model: object | None = None,
        min_score: float | None = None,
    ):
        self.settings = settings or Settings()
        self.base = base
        self.retriever = retriever
        self.min_score = (
            min_score if min_score is not None else self.settings.retrieval_min_score
        )
        self.lookup_tool = create_lookup_tool(base)

        if structured_model is None and llm is not None:
            structured_model = llm.with_structured_output(AssistantResponse)
        self.structured_model = structured_model

        self.graph = self._build_graph()

    def _get_structured_model(self) -> object:
        """Create the structured generator lazily so guardrail-only paths never
        require an API key."""
        if self.structured_model is None:
            self.structured_model = StructuredGenerator(
                create_llm(self.settings), AssistantResponse
            )
        return self.structured_model

    def _reconcile_sources(self, response: AssistantResponse) -> AssistantResponse:
        """Derive authoritative `sources` from `matched_records`.

        Models sometimes put keywords or record titles in `sources`. Since the
        source document name for a record is known exactly, we recompute it from
        the cited record IDs (order-preserving, de-duplicated)."""
        if not response.matched_records:
            return response
        sources: list[str] = []
        for record_id in response.matched_records:
            record = self.base.get_by_id(record_id)
            if record is not None and record.source not in sources:
                sources.append(record.source)
        return response.model_copy(update={"sources": sources})

    # ---------------------------------------------------------------- graph
    def _build_graph(self):
        g = StateGraph(AgentState)

        def retrieve(state: AgentState) -> AgentState:
            question = state["question"]
            tool_calls: list[ToolCall] = []
            context_blocks: list[str] = []

            match = _RECORD_ID_PATTERN.findall(question)
            if match:
                for raw_id in sorted({m.upper() for m in match}):
                    record_id = raw_id
                    tool_result = self.lookup_tool.invoke({"record_id": record_id})
                    tool_calls.append(
                        ToolCall(
                            name="lookup_record_by_id",
                            arguments={"record_id": record_id},
                            result_summary=tool_result[:120],
                        )
                    )
                    logger.info(
                        "request=%s tool=%s args=%s",
                        state.get("request_id", "-"),
                        "lookup_record_by_id",
                        {"record_id": record_id},
                    )
                    if not tool_result.startswith("No knowledge record found"):
                        context_blocks.append(tool_result)

            retrieved = self.retriever.retrieve(question)
            record_ids = [r.record.record_id for r in retrieved]
            scores = [r.score for r in retrieved]
            for r in retrieved:
                context_blocks.append(r.record.to_context_block())
            logger.info(
                "request=%s retrieved=%s scores=%s",
                state.get("request_id", "-"),
                record_ids,
                [round(s, 2) for s in scores],
            )
            return {
                "context_blocks": context_blocks,
                "record_ids": record_ids,
                "retrieved_scores": scores,
                "tool_calls": tool_calls,
            }

        def has_evidence(state: AgentState) -> str:
            tool_calls = state.get("tool_calls", [])
            if tool_calls:
                found = any(
                    not tc.result_summary.startswith("No knowledge record found")
                    for tc in tool_calls
                )
                return "generate" if found else "guardrail"
            good = [s for s in state.get("retrieved_scores", []) if s >= self.min_score]
            return "generate" if good else "guardrail"

        def generate(state: AgentState) -> AgentState:
            context = "\n\n".join(state["context_blocks"])
            messages = [
                ("system", SYSTEM_PROMPT),
                ("human", f"<context>\n{context}\n</context>\n\nQuestion: {state['question']}"),
            ]
            result = self._get_structured_model().invoke(messages)
            token_usage = None
            if isinstance(result, tuple):
                response, token_usage = result
            else:
                response = result
                token_usage = getattr(response, "usage_metadata", None)
            logger.info(
                "request=%s llm_usage=%s", state.get("request_id", "-"), token_usage
            )
            if response is None:
                logger.warning(
                    "request=%s structured_output_failed_after_retries",
                    state.get("request_id", "-"),
                )
                return {
                    "response": _insufficient_response(state["question"]),
                    "token_usage": token_usage,
                }
            if not isinstance(response, AssistantResponse):
                response = AssistantResponse.model_validate(response)
            return {"response": self._reconcile_sources(response), "token_usage": token_usage}

        def guardrail(state: AgentState) -> AgentState:
            logger.warning(
                "request=%s guardrail=insufficient_evidence",
                state.get("request_id", "-"),
            )
            return {"response": _insufficient_response(state["question"])}

        g.add_node("retrieve", retrieve)
        g.add_node("generate", generate)
        g.add_node("guardrail", guardrail)
        g.add_edge(START, "retrieve")
        g.add_conditional_edges("retrieve", has_evidence, {"generate": "generate", "guardrail": "guardrail"})
        g.add_edge("generate", END)
        g.add_edge("guardrail", END)
        return g.compile()

    # ------------------------------------------------------------- public API
    def answer(self, question: str, request_id: str | None = None) -> AssistantResponse:
        """Answer a question and return the structured response."""
        return self.answer_with_meta(question, request_id).response

    def answer_with_meta(
        self, question: str, request_id: str | None = None
    ) -> AgentResult:
        """Answer a question and return the structured response plus metadata."""
        start = time.perf_counter()
        request_id = request_id or f"req-{uuid.uuid4().hex[:10]}"
        logger.info("request=%s question=%s", request_id, question)
        try:
            final = self.graph.invoke(
                {"question": question, "request_id": request_id},
                config={"run_name": f"kb-assistant-{request_id}"},
            )
            response: AssistantResponse = final["response"]
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "request=%s completed elapsed_ms=%.1f needs_human_review=%s confidence=%.2f",
                request_id,
                elapsed_ms,
                response.needs_human_review,
                response.confidence,
            )
            return AgentResult(
                request_id=request_id,
                question=question,
                response=response,
                retrieved_records=final.get("record_ids", []),
                tool_calls=final.get("tool_calls", []),
                execution_time_ms=round(elapsed_ms, 2),
                token_usage=final.get("token_usage"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "request=%s error=%s elapsed_ms=%.1f", request_id, exc, elapsed_ms
            )
            raise
