#!/usr/bin/env python3
"""Оценка качества поиска на размеченном наборе вопросов.

Считает метрики для трёх ретриверов (BM25, вектор, гибрид через RRF) и для
«наивного» гибрида — такого, каким он получается без стемминга, с нулевыми
score BM25 и с полными ранжированиями на входе RRF.

    python eval.py
"""
import re
import sys

from rank_bm25 import BM25Okapi

from rag.corpus import load_documents, load_golden
from rag.pipeline import MiniRAG
from rag.search import rrf

K_VALUES = (1, 3, 5)
MRR_DEPTH = 10
CONTEXT_K = 5

_PLAIN_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def reciprocal_rank(ranking, sources, expected, depth=MRR_DEPTH):
    for pos, (i, _) in enumerate(ranking[:depth], start=1):
        if sources[i] == expected:
            return 1.0 / pos
    return 0.0


def evaluate(name, rank_fn, golden, sources):
    """hit@k, MRR и «мусор@5» — сколько фрагментов контекста не из того документа."""
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
    row[f"мусор@{CONTEXT_K}"] = sum(
        sum(1 for i, _ in r[:CONTEXT_K] if sources[i] != exp) for exp, r in rankings
    ) / n
    return row


def print_table(rows):
    headers = [h for h in rows[0] if h != "название"]
    width = max(len(r["название"]) for r in rows)
    print(f"{'ретривер':<{width}} " + " ".join(f"{h:>10}" for h in headers))
    for r in rows:
        cells = [
            f"{r[h]:>10.2f}" if h.startswith("мусор") else f"{r[h]:>10.1%}"
            for h in headers
        ]
        print(f"{r['название']:<{width}} " + " ".join(cells))


def main() -> None:
    documents = load_documents()
    golden = load_golden()
    rag = MiniRAG(documents)
    sources = [c.source for c in rag.chunks]

    print(f"документов: {len(documents)}, чанков: {len(rag.chunks)}, вопросов: {len(golden)}")
    print(f"кандидатов из каждого поиска перед слиянием: {rag.candidates}")

    # «Наивный» BM25: без стемминга и без отсева нулевых score.
    naive_bm25 = BM25Okapi([_PLAIN_TOKEN_RE.findall(c.text.lower()) for c in rag.chunks])

    def naive_hybrid(q):
        scores = naive_bm25.get_scores(_PLAIN_TOKEN_RE.findall(q.lower()))
        lexical = sorted(enumerate(scores), key=lambda x: -x[1])
        return rrf([rag.vector_ranking(q), lexical])

    retrievers = [
        ("только BM25", rag.bm25_ranking),
        ("только вектор", rag.vector_ranking),
        ("гибрид (RRF)", rag.hybrid_ranking),
        ("наивный гибрид", naive_hybrid),
    ]

    print("\n--- весь набор ---")
    print_table([evaluate(n, f, golden, sources) for n, f in retrievers])

    for kind, title in (("natural", "вопросы своими словами"), ("lexical", "точные коды и числа")):
        subset = [g for g in golden if g.get("kind") == kind]
        if subset:
            print(f"\n--- {title} ({len(subset)}) ---")
            print_table([evaluate(n, f, subset, sources) for n, f in retrievers])

    print(
        f"\nhit@k — доля вопросов, где нужный документ попал в top-k."
        f"\nмусор@{CONTEXT_K} — сколько из {CONTEXT_K} фрагментов контекста в среднем"
        f" пришли не из того документа."
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
