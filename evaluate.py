#!/usr/bin/env python3
"""Retrieval-quality eval harness.

Builds a temp index from a labeled corpus and reports P@k / Recall@k / MRR /
nDCG@k for keyword vs semantic vs hybrid search, so retrieval quality is
measurable and regressions are visible. Semantic/hybrid use whatever
CARDINAL_EMBEDDER selects (hash = meaningless placeholder; specter2 = real):

    python evaluate.py
    CARDINAL_EMBEDDER=specter2 python evaluate.py     # measure real semantic quality
"""
import argparse
import json
import math
import os
import tempfile

import build_index
import db
import queries
from embed import get_embedder

DEFAULT_CORPUS = os.path.join(os.path.dirname(__file__), "tests", "fixtures", "eval_corpus.json")
MODES = ("keyword", "semantic", "hybrid")


def _dcg(rels):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def query_metrics(ranked, gold, k):
    """Return (precision@k, recall@k, reciprocal-rank, nDCG@k) for one query."""
    gold = set(gold)
    topk = ranked[:k]
    hits = sum(1 for x in topk if x in gold)
    prec = hits / k if k else 0.0
    rec = hits / len(gold) if gold else 0.0
    rr = next((1.0 / (i + 1) for i, x in enumerate(ranked) if x in gold), 0.0)
    idcg = _dcg([1.0] * min(len(gold), k))
    ndcg = _dcg([1.0 if x in gold else 0.0 for x in topk]) / idcg if idcg else 0.0
    return prec, rec, rr, ndcg


def build_index_from_corpus(corpus, db_path, embedder):
    build_index.build(db_path).close()
    conn = db.connect(db_path)
    for p in corpus["papers"]:
        queries.ingest(conn, {"raw": p}, embedder=embedder)
    conn.commit()
    return conn


def run(corpus_path=DEFAULT_CORPUS, db_path=None, k=5, quiet=False):
    """Return {mode: (P@k, R@k, MRR, nDCG@k) | None} averaged over the corpus queries."""
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    embedder = get_embedder()
    tmp = db_path or os.path.join(tempfile.mkdtemp(), "eval.db")
    conn = build_index_from_corpus(corpus, tmp, embedder)
    try:
        results = {}
        for mode in MODES:
            agg, n = [0.0, 0.0, 0.0, 0.0], 0
            for query in corpus["queries"]:
                try:
                    ranked = [r["paper_id"] for r in queries.search(
                        conn, query["q"], mode=mode, embedder=embedder, k=k)]
                except queries.EmbedderUnavailable:
                    agg = None
                    break
                agg = [a + b for a, b in zip(agg, query_metrics(ranked, query["gold"], k))]
                n += 1
            results[mode] = None if agg is None else tuple(x / n for x in agg)
    finally:
        conn.close()
    if not quiet:
        _print(results, k, embedder)
    return results


def _print(results, k, embedder):
    ename = type(embedder).__name__ if embedder else "none"
    print(f"eval  k={k}  embedder={ename}")
    print(f"{'mode':9}{'P@'+str(k):>8}{'R@'+str(k):>8}{'MRR':>8}{'nDCG@'+str(k):>9}")
    for mode in MODES:
        r = results[mode]
        if r is None:
            print(f"{mode:9}{'  — no embedder configured':<33}")
        else:
            print(f"{mode:9}{r[0]:8.3f}{r[1]:8.3f}{r[2]:8.3f}{r[3]:9.3f}")


def main():
    ap = argparse.ArgumentParser(description="Retrieval-quality eval harness")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    run(a.corpus, k=a.k)


if __name__ == "__main__":
    main()
