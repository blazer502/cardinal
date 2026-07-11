import evaluate


def test_keyword_baseline(tmp_path, monkeypatch):
    # hash embedder keeps it deterministic and torch-free; keyword ignores the embedder anyway
    monkeypatch.setenv("CARDINAL_EMBEDDER", "hash")
    res = evaluate.run(db_path=str(tmp_path / "eval.db"), k=5, quiet=True)
    assert set(res) == {"keyword", "semantic", "hybrid"}
    p_at5, r_at5, mrr, ndcg = res["keyword"]
    # keyword must ace the term-overlap queries -> healthy aggregate retrieval
    assert mrr >= 0.6, f"keyword MRR regressed: {mrr}"
    assert r_at5 >= 0.4, f"keyword recall@5 regressed: {r_at5}"


def test_query_metrics_math():
    # gold={a,b,c}; ranked puts a@1, b@3 within top-5
    p, r, rr, ndcg = evaluate.query_metrics(["a", "x", "b", "y", "z"], ["a", "b", "c"], k=5)
    assert rr == 1.0                       # first relevant at rank 1
    assert round(r, 3) == round(2 / 3, 3)  # 2 of 3 gold in top-5
    assert p == 2 / 5
