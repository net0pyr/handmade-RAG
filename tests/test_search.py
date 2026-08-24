from rag.search import BM25Index, rrf, tokenize


def test_stemming_collapses_russian_cases():
    forms = ["отпуск", "отпуска", "отпуску", "отпусков"]
    stems = {tokenize(f)[0] for f in forms}
    assert len(stems) == 1, f"падежи разошлись по разным основам: {stems}"


# BM25 считает IDF как log((N - df + 0.5) / (df + 0.5)): на паре документов вес
# любого слова схлопывается в ноль, а у частых слов уходит в минус. Поэтому во
# всех тестах ниже база нарочно не маленькая.
CORPUS = [
    "Ежегодный отпуск составляет 28 календарных дней.",
    "Заявка на отпуск подаётся через HR-портал заранее.",
    "Компенсация занятий в тренажёрном зале до 3000 рублей.",
    "Доступ к VPN оформляется заявкой в проекте IT-ACCESS.",
    "Инцидент SEV1 означает полную недоступность сервиса.",
    "Дежурный инженер назначается в течение 15 минут.",
    "Ноутбук меняется планово раз в четыре года.",
    "Суточные в командировке составляют 2500 рублей.",
    "Пароль должен быть не короче 14 символов.",
    "Образовательный бюджет равен 100000 рублей в год.",
]


def test_bm25_finds_document_in_another_case():
    hits = BM25Index(CORPUS).search("сколько отпусков положено")
    assert hits, "запрос в другом падеже не нашёл ничего"
    assert hits[0][0] in (0, 1)


def test_bm25_drops_non_positive_scores():
    hits = BM25Index(CORPUS).search("отпуск")
    assert len(hits) == 2, "в выдаче должны остаться только документы со словом «отпуск»"
    assert all(score > 0 for _, score in hits)


def test_bm25_returns_nothing_for_unknown_words():
    assert BM25Index(CORPUS).search("криптовалюта") == []


def test_bm25_respects_limit():
    assert len(BM25Index(CORPUS).search("рублей", limit=2)) == 2


def test_rrf_favors_consistent_middle_over_single_top():
    # doc 1: третий в обоих списках. doc 0: первый, но только в одном.
    ranking_a = [(0, 0.9), (2, 0.5), (1, 0.4)]
    ranking_b = [(3, 0.9), (1, 0.6), (2, 0.3)]
    fused = rrf([ranking_a, ranking_b])
    fused_ids = [doc_id for doc_id, _ in fused]
    assert fused_ids.index(1) < fused_ids.index(0)


def test_rrf_scores_are_sorted_descending():
    ranking = [(0, 1.0), (1, 0.5)]
    fused = rrf([ranking])
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_zero_score_tail_sinks_the_right_answer():
    """Зачем BM25Index отбрасывает нулевые score.

    Документ 12 — уверенно первый по вектору, но у BM25 к нему нет ни одного
    совпадения, и в полном ранжировании он оказывается в самом хвосте. RRF
    смотрит только на позицию, поэтому хвост из нулей утаскивает правильный
    ответ вниз, пропуская вперёд документы, которые не выигрывают нигде.
    """
    vector = [(12, 0.9)] + [(i, 0.5) for i in range(12)]
    # у 0 есть лексическое совпадение, у остальных score ровно 0
    bm25_with_zeros = [(0, 1.8)] + [(i, 0.0) for i in range(1, 12)] + [(12, 0.0)]
    bm25_clean = [(0, 1.8)]

    def position_of(ranking, doc_id):
        return [d for d, _ in ranking].index(doc_id) + 1

    assert position_of(rrf([vector, bm25_with_zeros]), 12) > 3
    assert position_of(rrf([vector, bm25_clean]), 12) <= 3
