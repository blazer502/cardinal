import pytest

import queries
from embed import HashEmbedder


# ---- ingest / amortization ----
def test_ingest_cache_hit(seeded):
    calls = {"n": 0}
    def extractor(t, a):
        calls["n"] += 1
        return {"problem": "p"}
    extractor.model = "x"
    raw = dict(paper_id="S2:14", title="BadNets Backdoor Attack",
               abstract="Backdoor via poisoned training data.")
    res = queries.ingest(seeded, {"raw": raw}, extractor=extractor)
    assert res["cached"] is True and res["llm_calls"] == 0 and calls["n"] == 0
    assert res["fields_status"] == "extracted"


def test_ingest_requires_raw(seeded):
    with pytest.raises(NotImplementedError):
        queries.ingest(seeded, {"s2_id": "x"})


# ---- search ----
def test_keyword_search(seeded):
    ids = [r["paper_id"] for r in queries.search(seeded, "trojaning", mode="keyword")]
    assert "S2:22" in ids


def test_semantic_returns_scored(seeded):
    res = queries.search(seeded, "backdoor", mode="semantic", embedder=HashEmbedder(), k=3)
    assert len(res) == 3 and all("score" in r for r in res)


def test_semantic_requires_embedder(seeded):
    with pytest.raises(queries.EmbedderUnavailable):
        queries.search(seeded, "x", mode="semantic", embedder=None)


def test_hybrid_falls_back_to_keyword_without_embedder(seeded):
    res = queries.search(seeded, "trojaning", mode="hybrid", embedder=None)
    assert any(r["paper_id"] == "S2:22" for r in res)


def test_filter_year(seeded):
    ids = [r["paper_id"] for r in queries.search(seeded, "backdoor", mode="keyword",
                                                 filters={"year_min": 2019})]
    assert "S2:31" not in ids  # 2018 excluded


def test_filter_tags(seeded):
    ids = [r["paper_id"] for r in queries.search(seeded, "backdoor", mode="keyword",
                                                 filters={"tags": ["trojan"]})]
    assert "S2:22" in ids and "S2:14" not in ids


def test_bad_mode(seeded):
    with pytest.raises(ValueError):
        queries.search(seeded, "x", mode="bogus")


# ---- neighbors ----
def test_neighbors_cited_by(seeded):
    ids = {r["paper_id"] for r in queries.neighbors(seeded, "S2:14", "cited_by")}
    assert ids == {"S2:22", "S2:31"}


def test_neighbors_cites(seeded):
    ids = [r["paper_id"] for r in queries.neighbors(seeded, "S2:22", "cites")]
    assert ids == ["S2:14"]


def test_neighbors_similar_ordered(seeded):
    res = queries.neighbors(seeded, "S2:14", "similar")
    assert [r["paper_id"] for r in res] == ["S2:22", "S2:31"]  # weight desc
    assert res[0]["weight"] == 0.83


def test_neighbors_min_weight(seeded):
    res = queries.neighbors(seeded, "S2:14", "similar", min_weight=0.7)
    assert [r["paper_id"] for r in res] == ["S2:22"]


def test_neighbors_bad_kind(seeded):
    with pytest.raises(ValueError):
        queries.neighbors(seeded, "S2:14", "nonsense")


# ---- expand ----
def test_expand_fields(seeded):
    out = queries.expand(seeded, ["S2:14", "S2:22"], "fields")
    assert out["S2:14"]["problem"] == "poisoned data"
    assert out["S2:22"] is None


def test_expand_abstract(seeded):
    assert "poisoned" in queries.expand(seeded, ["S2:14"], "abstract")["S2:14"]


def test_expand_chunks(seeded):
    out = queries.expand(seeded, ["S2:14"], "chunks")
    assert out["S2:14"][0]["text"].startswith("we poison")


def test_expand_bad_level(seeded):
    with pytest.raises(ValueError):
        queries.expand(seeded, ["S2:14"], "bogus")


# ---- subgraph ----
def test_subgraph(seeded):
    g = queries.subgraph(seeded, ["S2:14"], hops=1)
    assert {n["paper_id"] for n in g["nodes"]} == {"S2:14", "S2:22", "S2:31"}
    assert len(g["edges"]) == 4


def test_subgraph_node_cap(seeded):
    g = queries.subgraph(seeded, ["S2:14"], hops=1, max_nodes=1)
    assert len(g["nodes"]) == 1


# ---- get_cluster ----
def test_get_cluster_by_id(seeded):
    c = queries.get_cluster(seeded, cluster_id=3)
    assert c["label"] == "Backdoors" and len(c["top_papers"]) == 3
    assert c["top_papers"][0]["paper_id"] == "S2:14"  # citations desc


def test_get_cluster_by_paper(seeded):
    assert queries.get_cluster(seeded, paper_id="S2:22")["cluster_id"] == 3


def test_get_cluster_requires_arg(seeded):
    with pytest.raises(ValueError):
        queries.get_cluster(seeded)


# ---- RRF ----
def test_rrf_orders_by_fused_score():
    order, scores = queries._rrf([["a", "b"], ["a", "c"]])
    assert order[0] == "a"  # top of both rankings
