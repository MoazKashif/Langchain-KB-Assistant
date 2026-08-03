"""Run the evaluation test cases against the agent.

Requires a configured model (see .env). Reports per-case retrieval, guardrail
flags and prints a small summary. Soft checks are used because the exact model
wording varies; inspect the answers to judge quality.

Usage:
    python -m evaluation.run_evaluation
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from src.agent import KnowledgeBaseAgent, configure_langsmith, create_llm, setup_logging
from src.config import get_settings
from src.retrieval import KnowledgeBase, KnowledgeBaseRetriever

EVAL_CSV = Path(__file__).resolve().parent / "test_cases.csv"


def main() -> int:
    settings = get_settings()
    setup_logging(settings)
    configure_langsmith(settings)

    base = KnowledgeBase.from_csv(settings.knowledge_base_path)
    retriever = KnowledgeBaseRetriever(base, top_k=settings.retriever_top_k)
    agent = KnowledgeBaseAgent(base=base, retriever=retriever, settings=settings)

    rows = list(csv.DictReader(EVAL_CSV.open(encoding="utf-8", newline="")))
    results = []
    for row in rows:
        result = agent.answer_with_meta(row["question"])
        expected_ids = set(row["expected_record_ids"].split())
        retrieved = set(result.response.matched_records)
        expected_unavailable = row["expects_unavailable"].strip().lower() == "true"
        expected_review = row["expects_human_review"].strip().lower() == "true"

        retrieval_ok = (not expected_ids) or expected_ids.issubset(retrieved) or retrieved & expected_ids
        guardrail_ok = (
            result.response.needs_human_review == expected_review
        ) or (
            expected_unavailable
            and result.response.needs_human_review
            and not retrieved
        )
        passed = retrieval_ok and guardrail_ok

        results.append(
            {
                "case_id": row["case_id"],
                "type": row["type"],
                "passed": passed,
                "matched_records": sorted(retrieved),
                "sources": list(result.response.sources),
                "confidence": round(result.response.confidence, 2),
                "needs_human_review": result.response.needs_human_review,
                "answer": result.response.answer,
            }
        )

    passed = sum(1 for r in results if r["passed"])
    print(f"\n=== Evaluation summary: {passed}/{len(results)} passed ===\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['case_id']} ({r['type']})")
        print(f"    matched_records={r['matched_records']} confidence={r['confidence']} review={r['needs_human_review']}")
        print(f"    answer: {r['answer'][:200]}")
        print()

    out = EVAL_CSV.parent / "evaluation_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Full results written to {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
