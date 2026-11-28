# RAG Support Knowledge Assistant

[![CI](https://github.com/alirezahs81/rag-support-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/rag-support-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Retrieval-Augmented Generation** for support knowledge — offline, deterministic, and production-shaped.

Chunked markdown knowledge base · **TF-IDF** retrieval · grounded answers · **Hit@K / MRR** · **FastAPI** · **Streamlit demo** · **Docker** · **pytest** + GitHub Actions.

> No external LLM required. Answers are synthesized only from retrieved local documents.

---

## Live demo

### Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

http://localhost:8501 — sample questions, answer + sources, Hit@K / MRR in the sidebar.

### Deploy: **Hugging Face Spaces** (recommended)

Streamlit Community Cloud can be flaky; use HF Spaces instead.

1. [Create a new Space](https://huggingface.co/new-space) → SDK **Streamlit** → CPU free.  
2. Connect GitHub repo `alirezahs81/rag-support-assistant` (branch `main`, app file `app.py`).  
3. Wait for the build; open the Space URL and add it to your profile README.

Full steps: [docs/DEPLOY_HUGGINGFACE.md](docs/DEPLOY_HUGGINGFACE.md).

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
├── app.py                  # Streamlit demo (HF Spaces entrypoint)
├── data/knowledge/
├── data/eval/questions.json
├── src/
├── tests/
├── main.py
└── Dockerfile              # FastAPI image
```

---

## Quick start (API)

```bash
git clone https://github.com/alirezahs81/rag-support-assistant.git
cd rag-support-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.build_index
uvicorn src.api:app --reload --port 8000
```

### Docker (API)

```bash
docker build -t rag-support .
docker run --rm -p 8000:8000 rag-support
```

---

## License

[MIT](LICENSE) © 2026 Alireza Hasheminasab
