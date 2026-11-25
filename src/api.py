"""FastAPI service for the support knowledge RAG assistant."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MIN_SCORE, TOP_K
from src.evaluate import evaluate_retrieval
from src.pipeline import RAGPipeline
from src.retrieve import get_or_build_index

app = FastAPI(
    title="RAG Support Knowledge Assistant",
    description=(
        "Offline retrieval-augmented answers over a local support knowledge base. "
        "TF-IDF retrieval + grounded synthesis (no external LLM required)."
    ),
    version="1.0.0",
)

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = RAGPipeline(get_or_build_index())
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _pipeline


class AskIn(BaseModel):
    question: str = Field(..., min_length=1, examples=["How do I reset my password?"])
    top_k: int = Field(default=TOP_K, ge=1, le=10)
    min_score: float = Field(default=MIN_SCORE, ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        pipe = get_pipeline()
        n = len(pipe.index.chunks)
        return {"status": "ok", "chunks": n, "index_ready": True}
    except HTTPException:
        return {"status": "degraded", "chunks": 0, "index_ready": False}


@app.post("/ask")
def ask(body: AskIn) -> dict[str, Any]:
    pipe = get_pipeline()
    return pipe.ask(body.question, top_k=body.top_k, min_score=body.min_score)


@app.get("/eval")
def eval_endpoint() -> dict[str, Any]:
    """Run retrieval evaluation on the bundled gold question set."""
    pipe = get_pipeline()
    report = evaluate_retrieval(index=pipe.index)
    return {k: v for k, v in report.items() if k != "details"}
