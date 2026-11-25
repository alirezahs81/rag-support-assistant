from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
EVAL_DIR = DATA_DIR / "eval"
MODELS_DIR = BASE_DIR / "models"

for _d in (KNOWLEDGE_DIR, EVAL_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

INDEX_FILE = MODELS_DIR / "tfidf_index.joblib"

# Retrieval
TOP_K = 3
MIN_SCORE = 0.05
CHUNK_SIZE = 420
CHUNK_OVERLAP = 60
