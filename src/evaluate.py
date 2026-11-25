"""Retrieval quality metrics on a gold question set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import EVAL_DIR, TOP_K
from src.retrieve import TfidfIndex, get_or_build_index


def load_eval_set(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or (EVAL_DIR / "questions.json")
    return json.loads(path.read_text(encoding="utf-8"))


def hit_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str], k: int) -> float:
    top = retrieved_doc_ids[:k]
    return 1.0 if any(d in gold_doc_ids for d in top) else 0.0


def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(
    index: TfidfIndex | None = None,
    eval_path: Path | None = None,
    k: int = TOP_K,
) -> dict[str, Any]:
    index = index or get_or_build_index()
    cases = load_eval_set(eval_path)
    hits = []
    mrrs = []
    details = []

    for case in cases:
        query = case["question"]
        gold = case["relevant_doc_ids"]
        retrieved = index.search(query, top_k=max(k, 5), min_score=0.0)
        doc_ids = [h.doc_id for h in retrieved]
        # de-duplicate while preserving order
        seen: set[str] = set()
        ordered: list[str] = []
        for d in doc_ids:
            if d not in seen:
                seen.add(d)
                ordered.append(d)
        h = hit_at_k(ordered, gold, k)
        rr = reciprocal_rank(ordered, gold)
        hits.append(h)
        mrrs.append(rr)
        details.append(
            {
                "id": case.get("id"),
                "question": query,
                "gold": gold,
                "retrieved": ordered[:k],
                "hit": h,
                "rr": round(rr, 4),
            }
        )

    n = max(len(hits), 1)
    return {
        "n": len(cases),
        "hit_at_k": round(sum(hits) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "k": k,
        "details": details,
    }


if __name__ == "__main__":
    report = evaluate_retrieval()
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
