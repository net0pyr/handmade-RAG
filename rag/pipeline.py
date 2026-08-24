from .chunking import chunk_text
from .embeddings import embed
from .search import BM25Index, rrf


class MiniRAG:
    def __init__(self, documents: list[str]):
        self.chunks = [c for d in documents for c in chunk_text(d)]
        self.matrix = embed(self.chunks, "passage: ")
        self.bm25 = BM25Index(self.chunks)

    def _vector_ranking(self, query: str) -> list[tuple[int, float]]:
        qv = embed([query], "query: ")[0]
        scores = self.matrix @ qv                # косинус за одно произведение
        return sorted(enumerate(scores), key=lambda x: -x[1])

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Гибридный поиск: вектор + BM25, объединённые через RRF."""
        vector_ranking = self._vector_ranking(query)
        bm25_ranking = self.bm25.search(query)
        fused = rrf([vector_ranking, bm25_ranking])
        return [self.chunks[i] for i, _ in fused[:k]]

    def ask(self, question: str, k: int = 5) -> str:
        from .generate import generate
        return generate(question, self.retrieve(question, k=k))
