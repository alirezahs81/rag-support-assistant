# RAG Support Knowledge Assistant

[![CI](https://github.com/alirezahs81/rag-support-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezahs81/rag-support-assistant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Retrieval-Augmented Generation** for support knowledge — offline, deterministic, and production-shaped.

**Live demo:** [https://alirezahs81-rag-support-assistant-app-4ryhln.streamlit.app/](https://alirezahs81-rag-support-assistant-app-4ryhln.streamlit.app/)

Chunked markdown knowledge base · **TF-IDF** retrieval · grounded answers · **Hit@K / MRR** · **FastAPI** · **Streamlit demo** · **Docker** · **pytest** + GitHub Actions.

> No external LLM required. Answers are synthesized only from retrieved local documents.

---

## Live demo

- **Public app:** [RAG Support Assistant on Streamlit](https://alirezahs81-rag-support-assistant-app-4ryhln.streamlit.app/)
- **Local:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

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
