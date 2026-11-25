from fastapi.testclient import TestClient

from src.api import app
from src.config import INDEX_FILE
from src.retrieve import TfidfIndex


def setup_module():
    index = TfidfIndex.build()
    index.save(INDEX_FILE)


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["index_ready"] is True
    assert body["chunks"] > 0


def test_ask():
    client = TestClient(app)
    r = client.post("/ask", json={"question": "How do I reset my password?"})
    assert r.status_code == 200
    body = r.json()
    assert body["grounded"] is True
    assert body["sources"]


def test_ask_empty_rejected():
    client = TestClient(app)
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422


def test_eval_endpoint():
    client = TestClient(app)
    r = client.get("/eval")
    assert r.status_code == 200
    body = r.json()
    assert body["hit_at_k"] >= 0.75
