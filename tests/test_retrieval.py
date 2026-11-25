from src.evaluate import evaluate_retrieval
from src.retrieve import TfidfIndex


def test_search_password_query():
    index = TfidfIndex.build()
    hits = index.search("How do I reset my password?", top_k=3, min_score=0.0)
    assert hits
    assert any(h.doc_id == "password-reset" for h in hits)


def test_search_billing_query():
    index = TfidfIndex.build()
    hits = index.search("duplicate charge refund invoice", top_k=3, min_score=0.0)
    assert hits
    assert any(h.doc_id == "billing-refunds" for h in hits)


def test_eval_hit_rate_reasonable():
    index = TfidfIndex.build()
    report = evaluate_retrieval(index=index, k=3)
    assert report["n"] >= 6
    assert report["hit_at_k"] >= 0.75
    assert report["mrr"] >= 0.6
