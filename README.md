# RAG Support Knowledge Assistant

[![CI](https://github.com/alirezahs81/rag-support-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/rag-support-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Retrieval-Augmented Generation** for support knowledge — offline, deterministic, and production-shaped.

Chunked markdown knowledge base · **TF-IDF** retrieval · grounded answers · **Hit@K / MRR** evaluation · **FastAPI** · **Docker** · **pytest** + GitHub Actions.

> No external LLM or embedding API is required. Answers are synthesized strictly from retrieved local documents so CI stays offline-reliable. The pipeline boundary is clean if you later swap the retriever for dense embeddings or the synthesizer for an LLM.

---

## Why this design

| Concern | Choice |
|---------|--------|
| Retrieval | Word **TF-IDF + cosine** (CPU, no model download) |
| Generation | **Grounded synthesis** from top chunks (no free-form hallucination) |
| Evaluation | Gold questions with **Hit@K** and **MRR** |
| Serving | FastAPI `/ask`, `/health`, `/eval` |
| Ops | Index built at Docker build time; CI on 3.10–3.12 |

---

## Project layout

```text
rag-support-assistant/
├── data/
│   ├── knowledge/          # support articles (.md)
│   └── eval/questions.json # gold retrieval set
├── src/
│   ├── chunking.py
│   ├── retrieve.py         # TF-IDF index
│   ├── generate.py         # grounded answer
│   ├── pipeline.py
│   ├── evaluate.py
│   ├── build_index.py
│   └── api.py
├── tests/
├── main.py
└── Dockerfile
```

---

## Quick start

```bash
git clone https://github.com/alirezahs81/rag-support-assistant.git
cd rag-support-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.build_index
python main.py                 # demo answer + eval metrics
pytest tests/ -v
uvicorn src.api:app --reload --port 8000
```

API docs: http://localhost:8000/docs

```bash
curl -s http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I reset my password?"}'
```

### Docker

```bash
docker build -t rag-support .
docker run --rm -p 8000:8000 rag-support
```

---

## Methodology

1. Load markdown articles under `data/knowledge/`.
2. Split into overlapping chunks; prefix each with the article title for better lexical match.
3. Fit a **TF-IDF** vectorizer on chunks; retrieve top-k by cosine similarity.
4. **Synthesize** an answer only from retrieved text (refuse when empty).
5. Evaluate retrieval on `data/eval/questions.json` with Hit@K and MRR.

On the bundled knowledge base, retrieval Hit@3 is typically **≥ 0.75** (asserted in tests).

---

## Extending toward dense RAG / LLM

- Replace `TfidfIndex` with sentence-transformers or a vector DB; keep the same `Hit` schema.
- Replace `synthesize_answer` with an OpenAI-compatible chat call that receives the same sources as context.
- Add your own docs by dropping `.md` files into `data/knowledge/` and re-running `python -m src.build_index`.

---

## License

[MIT](LICENSE) © 2026 Alireza Hasheminasab
