#!/usr/bin/env python3
"""Оценка качества поиска на размеченном наборе вопросов.

Считает метрики для трёх ретриверов: только BM25, только вектор и гибрид через RRF.

    python eval.py
"""
import sys

from rag.corpus import load_documents, load_golden
from rag.pipeline import MiniRAG

K_VALUES = (1, 3, 5)
MRR_DEPTH = 10


def reciprocal_rank(ranking, sources, expected, depth=MRR_DEPTH):
    for pos, (i, _) in enumerate(ranking[:depth], start=1):
        if sources[i] == expected:
            return 1.0 / pos
    return 0.0


def evaluate(name, rank_fn, golden, sources):
    """hit@k и MRR. Правильный документ ровно один, поэтому hit@k здесь — это recall@k."""
    row = {"название": name}
    rankings = [(g["source"], rank_fn(g["question"])) for g in golden]
    n = len(golden)
    for k in K_VALUES:
        row[f"hit@{k}"] = sum(
            any(sources[i] == exp for i, _ in r[:k]) for exp, r in rankings
        ) / n
    row[f"MRR@{MRR_DEPTH}"] = sum(
        reciprocal_rank(r, sources, exp) for exp, r in rankings
    ) / n
    return row


def print_table(rows):
    headers = [h for h in rows[0] if h != "название"]
    width = max(len(r["название"]) for r in rows)
    print(f"{'ретривер':<{width}} " + " ".join(f"{h:>10}" for h in headers))
    for r in rows:
        print(f"{r['название']:<{width}} " + " ".join(f"{r[h]:>10.1%}" for h in headers))


def main() -> None:
    documents = load_documents()
    golden = load_golden()
    rag = MiniRAG(documents)
    sources = [c.source for c in rag.chunks]

    print(f"документов: {len(documents)}, чанков: {len(rag.chunks)}, вопросов: {len(golden)}")
    print(f"кандидатов из каждого поиска перед слиянием: {rag.candidates}")

    retrievers = [
        ("только BM25", rag.bm25_ranking),
        ("только вектор", rag.vector_ranking),
        ("гибрид (RRF)", rag.hybrid_ranking),
    ]

    print("\n--- весь набор ---")
    print_table([evaluate(n, f, golden, sources) for n, f in retrievers])

    for kind, title in (("natural", "вопросы своими словами"), ("lexical", "точные коды и числа")):
        subset = [g for g in golden if g.get("kind") == kind]
        if subset:
            print(f"\n--- {title} ({len(subset)}) ---")
            print_table([evaluate(n, f, subset, sources) for n, f in retrievers])

    print(
        "\nhit@k — доля вопросов, где нужный документ попал в top-k."
        f"\nMRR@{MRR_DEPTH} — усреднённая 1/позиция первого правильного фрагмента."
    )

    scored = [
        (reciprocal_rank(rag.hybrid_ranking(g["question"]), sources, g["source"]), g)
        for g in golden
    ]
    print("\nхуже всего гибрид отвечает на:")
    for rr, g in sorted(scored, key=lambda x: x[0])[:5]:
        pos = int(round(1 / rr)) if rr else f">{MRR_DEPTH}"
        print(f"  {g['question'][:50]:52} ждём {g['source']:14} позиция {pos}")


if __name__ == "__main__":
    sys.exit(main())
