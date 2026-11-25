"""Grounded answer generation without an external LLM dependency.

Answers are composed strictly from retrieved knowledge chunks so the system
stays offline, deterministic, and CI-friendly. Swap `synthesize_answer` for an
OpenAI-compatible call later if you want generative phrasing.
"""

from __future__ import annotations

from src.retrieve import Hit


def synthesize_answer(query: str, hits: list[Hit]) -> str:
    if not hits:
        return (
            "I could not find relevant information in the knowledge base. "
            "Please rephrase your question or contact support."
        )

    # Prefer the highest-scoring chunk as the primary answer body
    primary = hits[0]
    body = primary.text
    # Drop the "Title. " prefix we injected for retrieval if present
    if body.startswith(primary.title):
        body = body[len(primary.title) :].lstrip(" .")

    lines = [
        f"Based on **{primary.title}**:",
        body.strip(),
    ]

    if len(hits) > 1:
        extra_titles = []
        for h in hits[1:]:
            if h.title != primary.title and h.title not in extra_titles:
                extra_titles.append(h.title)
        if extra_titles:
            lines.append("")
            lines.append("Related topics: " + "; ".join(extra_titles) + ".")

    lines.append("")
    lines.append(
        "_This answer is grounded in the local knowledge base "
        "(no external LLM required)._"
    )
    return "\n".join(lines)
