"""Command-line entry point.

Usage:
    python -m src.main "What is the remote work policy?"
    python -m src.main                      # interactive REPL
    python -m src.main --list-records       # list available record IDs
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import KnowledgeBaseAgent, configure_langsmith, create_llm, setup_logging
from .config import get_settings
from .retrieval import KnowledgeBase, KnowledgeBaseRetriever


def build_agent():
    settings = get_settings()
    setup_logging(settings)
    configure_langsmith(settings)
    base = KnowledgeBase.from_csv(settings.knowledge_base_path)
    retriever = KnowledgeBaseRetriever(base, top_k=settings.retriever_top_k)
    agent = KnowledgeBaseAgent(base=base, retriever=retriever, settings=settings)
    return agent, settings


def run_single(agent: KnowledgeBaseAgent, question: str) -> None:
    response = agent.answer(question)
    print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))


def run_repl(agent: KnowledgeBaseAgent) -> None:
    print("Knowledge Base Assistant. Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        response = agent.answer(question)
        print(f"\nAssistant: {response.answer}")
        print(
            f"(records: {', '.join(response.matched_records) or '-'} | "
            f"confidence: {response.confidence:.2f} | "
            f"needs_human_review: {response.needs_human_review})\n"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Knowledge Base Assistant")
    parser.add_argument("question", nargs="?", help="Single question to answer")
    parser.add_argument(
        "--list-records", action="store_true", help="Print available record IDs"
    )
    args = parser.parse_args(argv)

    agent, settings = build_agent()

    if args.list_records:
        for rid in agent.base.ids:
            print(rid)
        return 0

    if args.question:
        run_single(agent, args.question)
        return 0

    run_repl(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
