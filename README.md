# RAG Support Knowledge Assistant

[![CI](https://github.com/alirezahs81/rag-support-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/rag-support-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Retrieval-Augmented Generation** for support knowledge — offline, deterministic, and production-shaped.

Chunked markdown knowledge base · **TF-IDF** retrieval · grounded answers · **Hit@K / MRR** · **FastAPI** · **Streamlit live demo** · **Docker** · **pytest** + GitHub Actions.

> No external LLM required. Answers are synthesized only from retrieved local documents.

---

## Live demo (Streamlit)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 — sample questions, answer + sources, optional Hit@K / MRR in the sidebar.

### Deploy on Streamlit Community Cloud (free)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → repository `alirezahs81/rag-support-assistant`.
3. Branch: `main` · Main file path: `app.py`.
4. Deploy. After it goes live, put the public URL in your profile README.

Step-by-step notes: [docs/DEPLOY_STREAMLIT.md](docs/DEPLOY_STREAMLIT.md).

---

## Why this design

| Concern | Choice |
|---------|--------|
| Retrieval | Word **TF-IDF + cosine** (CPU, no model download) |
| Generation | **Grounded synthesis** from top chunks |
| Evaluation | Gold questions with **Hit@K** and **MRR** |
| Serving | FastAPI `/ask` · Streamlit `app.py` |
| Ops | Index on first use / Docker build; CI on 3.10–3.12 |

---

## Project layout

```text
rag-support-assistant/
├── app.py                  # Streamlit demo
├── data/knowledge/         # support articles (.md)
├── data/eval/questions.json
├── src/                    # chunk · retrieve · generate · api
├── tests/
├── main.py
└── Dockerfile              # API image
```

---

## Quick start (API)

```bash
git clone https://github.com/alirezahs81/rag-support-assistant.git
cd rag-support-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.build_index
python main.py
pytest tests/ -v
uvicorn src.api:app --reload --port 8000
```

```bash
curl -s http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I reset my password?"}'
```

### Docker (API)

```bash
docker build -t rag-support .
docker run --rm -p 8000:8000 rag-support
```

---

## Methodology

1. Load markdown under `data/knowledge/`.
2. Chunk with overlap; prefix titles for lexical match.
3. TF-IDF + cosine top-k.
4. Synthesize only from hits (refuse if empty).
5. Evaluate on `data/eval/questions.json` (Hit@K, MRR).

---

## License

[MIT](LICENSE) © 2026 Alireza Hasheminasab
