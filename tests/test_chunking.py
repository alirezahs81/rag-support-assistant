from src.chunking import build_chunks, load_documents


def test_load_documents_not_empty():
    docs = load_documents()
    assert len(docs) >= 6
    ids = {d[0] for d in docs}
    assert "password-reset" in ids
    assert "billing-refunds" in ids


def test_chunks_have_ids_and_text():
    chunks = build_chunks()
    assert len(chunks) >= len(load_documents())
    for c in chunks:
        assert c.chunk_id
        assert c.doc_id
        assert c.text
