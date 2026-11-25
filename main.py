"""CLI entry: build index, answer a sample question, print eval metrics."""

from __future__ import annotations

import json

from src.build_index import main as build_index
from src.evaluate import evaluate_retrieval
from src.pipeline import RAGPipeline
from src.retrieve import get_or_build_index


def main() -> None:
    build_index()
    pipe = RAGPipeline(get_or_build_index())
    demo_q = "How do I reset my password?"
    result = pipe.ask(demo_q)
    print("\n=== Demo answer ===")
    print(result["answer"])
    print("\nSources:", [s["doc_id"] for s in result["sources"]])

    report = evaluate_retrieval(index=pipe.index)
    print("\n=== Retrieval eval ===")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
