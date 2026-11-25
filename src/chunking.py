"""Load markdown knowledge files and split into overlapping chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, KNOWLEDGE_DIR


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def _split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Prefer break near a sentence boundary
        window = text[start:end]
        if end < len(text):
            for sep in (". ", "? ", "! ", "\n"):
                idx = window.rfind(sep)
                if idx > size // 3:
                    end = start + idx + len(sep)
                    window = text[start:end]
                    break
        chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def load_documents(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[tuple[str, str, str]]:
    """Return list of (doc_id, title, body)."""
    docs: list[tuple[str, str, str]] = []
    paths = sorted(knowledge_dir.glob("*.md"))
    for path in paths:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        lines = raw.splitlines()
        title = path.stem.replace("-", " ").title()
        body_start = 0
        if lines and lines[0].startswith("#"):
            title = lines[0].lstrip("# ").strip()
            body_start = 1
        body = "\n".join(lines[body_start:]).strip()
        docs.append((path.stem, title, body))
    return docs


def build_chunks(
    knowledge_dir: Path = KNOWLEDGE_DIR,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, title, body in load_documents(knowledge_dir):
        pieces = _split_text(body, size=chunk_size, overlap=overlap)
        if not pieces and body:
            pieces = [body]
        for i, piece in enumerate(pieces):
            # Prefix title so retrieval can match topic terms
            text = f"{title}. {piece}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::{i}",
                    doc_id=doc_id,
                    title=title,
                    text=text,
                )
            )
    return chunks
