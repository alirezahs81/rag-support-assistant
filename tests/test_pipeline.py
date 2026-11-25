from src.pipeline import RAGPipeline
from src.retrieve import TfidfIndex


def test_ask_grounded():
    pipe = RAGPipeline(TfidfIndex.build())
    out = pipe.ask("How do I reset my password?")
    assert out["grounded"] is True
    assert out["sources"]
    assert "password" in out["answer"].lower() or "reset" in out["answer"].lower()


def test_ask_emptyish_still_returns_structure():
    pipe = RAGPipeline(TfidfIndex.build())
    out = pipe.ask("zzzz not a real support topic xyzzy")
    assert "query" in out
    assert "answer" in out
    assert "sources" in out
