import pytest

from rag.chunking import chunk_document, chunk_text


def test_short_text_stays_one_chunk():
    text = "Короткий текст."
    assert chunk_text(text, size=400, overlap=80) == [text]


def test_respects_overlap_boundary():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


def test_chunks_join_back_to_original_content():
    text = "Предложение номер один. " * 50
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)


def test_no_infinite_loop_on_no_separators():
    text = "а" * 1000
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1


def test_chunk_document_keeps_source_and_position():
    chunks = chunk_document("vpn.md", "Предложение. " * 200, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(c.source == "vpn.md" for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert chunks[0].label == "vpn.md, фрагмент 1"
