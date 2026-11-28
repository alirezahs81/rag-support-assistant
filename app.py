"""Streamlit demo — ask the support knowledge base (offline RAG).

Run locally::

    streamlit run app.py

Deploy: Streamlit Community Cloud → point to this repo, Main file path: app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.evaluate import evaluate_retrieval
from src.pipeline import RAGPipeline
from src.retrieve import get_or_build_index

SAMPLE_QUESTIONS = [
    "How do I reset my password?",
    "I was charged twice — can I get a refund?",
    "My account is locked after failed logins",
    "The app crashes when I upload files",
    "How do I upgrade to the Business plan?",
    "How can I download or delete my data?",
]


@st.cache_resource(show_spinner="Building retrieval index…")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline(get_or_build_index())


def main() -> None:
    st.set_page_config(
        page_title="RAG Support Assistant",
        page_icon="💬",
        layout="centered",
    )

    st.title("RAG Support Knowledge Assistant")
    st.caption(
        "Offline retrieval-augmented answers over a local FAQ — "
        "TF-IDF retrieval + grounded synthesis (no external LLM)."
    )

    pipe = load_pipeline()

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Top-K sources", 1, 5, 3)
        min_score = st.slider("Min similarity", 0.0, 0.5, 0.05, 0.01)
        st.markdown("---")
        if st.button("Run retrieval eval (Hit@K / MRR)"):
            with st.spinner("Evaluating…"):
                report = evaluate_retrieval(index=pipe.index, k=top_k)
            st.metric("Hit@K", f"{report['hit_at_k']:.2f}")
            st.metric("MRR", f"{report['mrr']:.2f}")
            st.caption(f"n = {report['n']}, k = {report['k']}")
        st.markdown("---")
        st.markdown(
            "[GitHub repo](https://github.com/alirezahs81/rag-support-assistant)"
        )

    st.subheader("Try a sample question")
    cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 2].button(q, use_container_width=True):
            st.session_state["question"] = q

    question = st.text_input(
        "Your question",
        value=st.session_state.get("question", ""),
        placeholder="e.g. Where can I find my invoices?",
    )

    if st.button("Ask", type="primary") or (
        question and st.session_state.get("_auto_ask")
    ):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Retrieving…"):
                result = pipe.ask(question, top_k=top_k, min_score=min_score)

            st.markdown("### Answer")
            st.markdown(result["answer"])

            badge = "grounded" if result["grounded"] else "no matching source"
            st.caption(f"Status: **{badge}**")

            if result["sources"]:
                st.markdown("### Sources")
                for s in result["sources"]:
                    with st.expander(
                        f"{s['title']}  ·  score={s['score']:.3f}  ·  `{s['doc_id']}`"
                    ):
                        st.write(s["text"])

    st.markdown("---")
    st.caption(
        "Demo knowledge base: password, billing, access, outages, plans, privacy. "
        "Answers are composed only from retrieved chunks."
    )


if __name__ == "__main__":
    main()
