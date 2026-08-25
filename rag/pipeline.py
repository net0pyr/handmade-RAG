from .chunking import Chunk, chunk_document
from .embeddings import embed
from .search import BM25Index, rrf

Ranking = list[tuple[int, float]]


class MiniRAG:
    """Гибридный ретривер: вектор + BM25, объединённые через RRF.

    candidates — сколько кандидатов берётся из каждого поиска перед слиянием;
    почему не вся база, написано в rrf().
    """

    def __init__(self, documents: dict[str, str], candidates: int = 20):
        self.chunks: list[Chunk] = [
            c for source, text in documents.items() for c in chunk_document(source, text)
        ]
        self.candidates = candidates
        self.matrix = embed([c.text for c in self.chunks], "passage: ")
        self.bm25 = BM25Index([c.text for c in self.chunks])

    def vector_ranking(self, query: str, limit: int | None = None) -> Ranking:
        qv = embed([query], "query: ")[0]
        scores = self.matrix @ qv                # косинус за одно произведение
        ranking = sorted(enumerate(scores), key=lambda x: -x[1])
        return ranking[:limit] if limit else ranking

    def bm25_ranking(self, query: str, limit: int | None = None) -> Ranking:
        return self.bm25.search(query, limit=limit)

    def hybrid_ranking(self, query: str) -> Ranking:
        return rrf([
            self.vector_ranking(query, limit=self.candidates),
            self.bm25_ranking(query, limit=self.candidates),
        ])

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        return [self.chunks[i] for i, _ in self.hybrid_ranking(query)[:k]]
