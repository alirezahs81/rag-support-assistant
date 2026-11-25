"""Build and persist the TF-IDF retrieval index."""

from __future__ import annotations

from src.retrieve import TfidfIndex
from src.config import INDEX_FILE


def main() -> None:
    index = TfidfIndex.build()
    path = index.save()
    print(f"Indexed {len(index.chunks)} chunks → {path}")


if __name__ == "__main__":
    main()
