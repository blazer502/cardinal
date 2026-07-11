"""Shared pytest fixtures. Deterministic, no network, no torch (uses the hash embedder)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_index  # noqa: E402
import db  # noqa: E402
import queries  # noqa: E402
from embed import HashEmbedder  # noqa: E402


def _vec(n):
    return [((k * n) % 97) / 97.0 for k in range(768)]


def _extractor(title, abstract):
    return {"problem": "poisoned data", "method": "trigger implant", "dataset": "CIFAR",
            "metric": "ASR", "result": "high asr", "limitation": "visible trigger"}


_extractor.model = "test-extractor"


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def conn(db_path):
    build_index.build(db_path).close()
    c = db.connect(db_path)
    yield c
    c.close()


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def seeded(conn, embedder):
    """A small corpus with fields, abstracts, precomputed vectors, edges, and a cluster."""
    papers = [
        dict(paper_id="S2:14", title="BadNets Backdoor Attack",
             abstract="Backdoor via poisoned training data.", year=2019, venue="IEEE Access",
             tldr="backdoor via poisoned data", tags=["backdoor", "dnn"], n_citations=980,
             embedding=_vec(3), ext_ids=[{"scheme": "doi", "value": "10.1/a"}]),
        dict(paper_id="S2:22", title="Trojaning Attack on Neural Networks",
             abstract="Implanting triggers into trained nets.", year=2021, venue="NDSS",
             tldr="trigger implant attack", tags=["backdoor", "trojan"], n_citations=410,
             embedding=_vec(5)),
        dict(paper_id="S2:31", title="Fine-Pruning Defense",
             abstract="Pruning and fine-tuning defense.", year=2018, venue="RAID",
             tldr="pruning defense", tags=["backdoor", "defense"], n_citations=260,
             embedding=_vec(7)),
    ]
    for p in papers:
        queries.ingest(conn, {"raw": p}, embedder=embedder,
                       extractor=_extractor if p["paper_id"] == "S2:14" else None)
    conn.executemany("INSERT INTO edge(src,dst,kind,weight,intent) VALUES (?,?,?,?,?)", [
        ("S2:22", "S2:14", "cites", 1.0, "method"),
        ("S2:31", "S2:14", "cites", 1.0, "background"),
        ("S2:14", "S2:22", "similar", 0.83, None),
        ("S2:14", "S2:31", "similar", 0.61, None),
    ])
    conn.execute("INSERT INTO cluster(cluster_id,label,summary,size) "
                 "VALUES (3,'Backdoors','Data-poisoning backdoors',3)")
    conn.execute("UPDATE paper SET cluster_id=3")
    conn.execute("INSERT INTO paper_chunk(paper_id,section,ord,text) "
                 "VALUES ('S2:14','Method',0,'we poison images with a trigger patch')")
    conn.commit()
    return conn
