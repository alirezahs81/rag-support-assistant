"""End-to-end retrieve → generate pipeline."""

from __future__ import annotations

from typing import Any

from src.config import MIN_SCORE, TOP_K
from src.generate import synthesize_answer
from src.retrieve import TfidfIndex, get_or_build_index


class RAGPipeline:
    def __init__(self, index: TfidfIndex | None = None):
        self.index = index or get_or_build_index()

    def ask(
        self,
        query: str,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
    ) -> dict[str, Any]:
        hits = self.index.search(query, top_k=top_k, min_score=min_score)
        answer = synthesize_answer(query, hits)
        return {
            "query": query,
            "answer": answer,
            "sources": [h.to_dict() for h in hits],
            "grounded": bool(hits),
        }
