from rag.search import rrf


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
