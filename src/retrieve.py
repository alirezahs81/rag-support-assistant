"""Offline TF-IDF retrieval index (CPU-only, CI-friendly)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.chunking import Chunk, build_chunks
from src.config import INDEX_FILE, MIN_SCORE, TOP_K


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TfidfIndex:
    def __init__(self, vectorizer: TfidfVectorizer, matrix, chunks: list[Chunk]):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[Chunk] | None = None) -> "TfidfIndex":
        chunks = chunks or build_chunks()
        if not chunks:
            raise ValueError("No knowledge chunks found. Add markdown files under data/knowledge/.")
        texts = [c.text for c in chunks]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(texts)
        return cls(vectorizer=vectorizer, matrix=matrix, chunks=chunks)

    def save(self, path: Path = INDEX_FILE) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "matrix": self.matrix,
                "chunks": self.chunks,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path = INDEX_FILE) -> "TfidfIndex":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index not found at {path}. Run: python -m src.build_index")
        payload = joblib.load(path)
        return cls(
            vectorizer=payload["vectorizer"],
            matrix=payload["matrix"],
            chunks=payload["chunks"],
        )

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
    ) -> list[Hit]:
        if not query or not query.strip():
            return []
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        if scores.size == 0:
            return []
        order = np.argsort(scores)[::-1]
        hits: list[Hit] = []
        for idx in order[: max(top_k * 3, top_k)]:
            score = float(scores[idx])
            if score < min_score:
                continue
            ch = self.chunks[int(idx)]
            hits.append(
                Hit(
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    title=ch.title,
                    text=ch.text,
                    score=round(score, 4),
                )
            )
            if len(hits) >= top_k:
                break
        return hits


def get_or_build_index(path: Path = INDEX_FILE) -> TfidfIndex:
    path = Path(path)
    if path.exists():
        return TfidfIndex.load(path)
    index = TfidfIndex.build()
    index.save(path)
    return index
