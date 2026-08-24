import re

import snowballstemmer
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STEMMER = snowballstemmer.stemmer("russian")


def tokenize(text: str) -> list[str]:
    """Слова в нижнем регистре, приведённые к основе.

    Без стемминга BM25 на русском рассыпается: «отпуску» и «отпусков» не находят
    документ, где написано «отпуск», потому что это буквально разные строки.
    """
    return _STEMMER.stemWords(_TOKEN_RE.findall(text.lower()))


class BM25Index:
    def __init__(self, texts: list[str]):
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, limit: int | None = None) -> list[tuple[int, float]]:
        """Ранжирование по BM25 без нулевых score.

        score == 0 означает «ни одного общего слова с запросом», то есть документ
        не найден. Если оставить такие записи в списке, RRF всё равно засчитает им
        ранг и они начнут вытеснять нормальные результаты векторного поиска.
        """
        scores = self._bm25.get_scores(tokenize(query))
        hits = [(i, s) for i, s in enumerate(scores) if s > 0.0]
        hits.sort(key=lambda x: -x[1])
        return hits[:limit] if limit else hits


def rrf(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    Складывает не веса, а обратные ранги, поэтому косинус и score BM25 не нужно
    приводить к одной шкале. На вход подавайте списки кандидатов, а не полные
    ранжирования всей базы: вклад 1/(k+rank) убывает медленно, и длинный хвост
    нерелевантных документов заметно смазывает картину.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])
