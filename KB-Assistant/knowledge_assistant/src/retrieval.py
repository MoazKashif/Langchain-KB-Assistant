"""Knowledge base loading, splitting, indexing and retrieval.

We use a lexical BM25 index (rank_bm25) built over per-record chunks. This
requires no embedding API key, is deterministic (great for tests) and returns
interpretable scores that feed the confidence / guardrail logic.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from .config import Settings
from .models import KnowledgeRecord, RetrievedRecord


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer shared by query and documents."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_records(path: Path) -> list[KnowledgeRecord]:
    """Load knowledge records from a CSV file."""
    records: list[KnowledgeRecord] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            keywords = [k.strip() for k in row.get("keywords", "").split(",") if k.strip()]
            records.append(
                KnowledgeRecord(
                    record_id=row["record_id"].strip(),
                    title=row["title"].strip(),
                    category=row["category"].strip(),
                    description=row["description"].strip(),
                    keywords=keywords,
                    source=row["source"].strip(),
                    last_updated=row["last_updated"].strip(),
                )
            )
    return records


class KnowledgeBase:
    """In-memory store of records with exact ID lookup."""

    def __init__(self, records: list[KnowledgeRecord]):
        self.records = records
        self._by_id: dict[str, KnowledgeRecord] = {r.record_id: r for r in records}

    @classmethod
    def from_csv(cls, path: Path) -> "KnowledgeBase":
        return cls(load_records(path))

    def get_by_id(self, record_id: str) -> KnowledgeRecord | None:
        """Exact record lookup by record ID."""
        key = record_id.strip().upper()
        return self._by_id.get(key)

    @property
    def ids(self) -> list[str]:
        return list(self._by_id.keys())


class KnowledgeBaseRetriever(BaseRetriever):
    """BM25 retriever over split record chunks.

    Documents are split with a recursive character splitter so a single record
    may span several chunks. Retrieved chunks are grouped back by record ID and
    the best chunk score is used as the record's relevance score.
    """

    base: "KnowledgeBase"
    top_k: int = 4
    chunk_size: int = 400
    chunk_overlap: int = 60

    _chunks: list[Document] = []
    _records_by_id: dict[str, KnowledgeRecord] = {}
    _bm25: BM25Okapi | None = None

    def __init__(self, base: KnowledgeBase, top_k: int = 4, **kwargs):
        kwargs.setdefault("base", base)
        kwargs.setdefault("top_k", top_k)
        super().__init__(**kwargs)
        self._build_index()

    def _build_index(self) -> None:
        self._records_by_id = {r.record_id: r for r in self.base.records}
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n", " ", ""],
        )
        chunks: list[Document] = []
        for record in self.base.records:
            for i, piece in enumerate(splitter.split_text(record.to_search_text())):
                chunks.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "record_id": record.record_id,
                            "chunk_index": i,
                        },
                    )
                )
        self._chunks = chunks
        tokenized = [_tokenize(c.page_content) for c in chunks]
        self._bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedRecord]:
        """Return the most relevant records for a query with their scores."""
        docs = self._get_relevant_documents(query)
        k = top_k or self.top_k
        return [
            RetrievedRecord(
                record=self._records_by_id[d.metadata["record_id"]],
                score=float(d.metadata.get("score", 0.0)),
            )
            for d in docs[:k]
        ]

    def _get_relevant_documents(self, query: str) -> list[Document]:
        if self._bm25 is None:
            return []
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        best_score: dict[str, float] = defaultdict(lambda: float("-inf"))
        for chunk, score in zip(self._chunks, scores):
            rid = chunk.metadata["record_id"]
            if score > best_score[rid]:
                best_score[rid] = score

        ranked = sorted(best_score.items(), key=lambda kv: kv[1], reverse=True)
        documents: list[Document] = []
        for rid, score in ranked[: self.top_k]:
            record = self._records_by_id[rid]
            documents.append(
                Document(
                    page_content=record.to_context_block(),
                    metadata={"record_id": rid, "score": score},
                )
            )
        return documents


@lru_cache
def get_retriever(settings: Settings | None = None) -> KnowledgeBaseRetriever:
    settings = settings or Settings()
    base = KnowledgeBase.from_csv(settings.knowledge_base_path)
    return KnowledgeBaseRetriever(base, top_k=settings.retriever_top_k)
