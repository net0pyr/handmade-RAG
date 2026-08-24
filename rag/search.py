import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c) for c in chunks])

    def search(self, query: str) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        return sorted(enumerate(scores), key=lambda x: -x[1])


def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])
