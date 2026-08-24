"""Регрессия на размеченном наборе вопросов.

Тест поднимает настоящую модель эмбеддингов, поэтому по умолчанию пропускается:

    RAG_SLOW_TESTS=1 pytest
"""
import os

import pytest

from rag.corpus import load_documents, load_golden

pytestmark = pytest.mark.skipif(
    os.getenv("RAG_SLOW_TESTS") != "1",
    reason="нужен intfloat/multilingual-e5-large; включается RAG_SLOW_TESTS=1",
)


@pytest.fixture(scope="module")
def rag():
    from rag.pipeline import MiniRAG
    return MiniRAG(load_documents())


@pytest.fixture(scope="module")
def golden():
    return load_golden()


def hit_rate(rag, golden, rank_fn, k):
    sources = [c.source for c in rag.chunks]
    hits = 0
    for item in golden:
        ranking = rank_fn(item["question"])[:k]
        hits += any(sources[i] == item["source"] for i, _ in ranking)
    return hits / len(golden)


def test_hybrid_hits_top5_almost_always(rag, golden):
    assert hit_rate(rag, golden, rag.hybrid_ranking, 5) >= 0.90


def test_hybrid_is_not_worse_than_vector_alone(rag, golden):
    """Гибрид обязан быть не хуже вектора — ровно это и ломалось раньше."""
    hybrid = hit_rate(rag, golden, rag.hybrid_ranking, 3)
    vector = hit_rate(rag, golden, rag.vector_ranking, 3)
    assert hybrid >= vector - 0.02, f"гибрид {hybrid:.2%} против вектора {vector:.2%}"


def test_lexical_queries_survive(rag, golden):
    """Коды и идентификаторы — та часть, ради которой в схеме вообще есть BM25."""
    sources = [c.source for c in rag.chunks]
    for item in golden:
        if item["question"].isascii() and len(item["question"].split()) == 1:
            ranking = rag.hybrid_ranking(item["question"])[:3]
            assert any(sources[i] == item["source"] for i, _ in ranking), item["question"]


def test_retrieve_returns_chunks_with_sources(rag):
    chunks = rag.retrieve("как получить впн", k=3)
    assert len(chunks) == 3
    assert all(c.source.endswith(".md") for c in chunks)
