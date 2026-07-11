import pytest

import embed
from embed import HashEmbedder, Specter2Embedder


def test_factory_none(monkeypatch):
    monkeypatch.setenv("CARDINAL_EMBEDDER", "none")
    assert embed.get_embedder() is None


def test_factory_hash(monkeypatch):
    monkeypatch.setenv("CARDINAL_EMBEDDER", "hash")
    assert isinstance(embed.get_embedder(), HashEmbedder)


def test_factory_specter2_is_lazy(monkeypatch):
    monkeypatch.setenv("CARDINAL_EMBEDDER", "specter2")
    e = embed.get_embedder()
    assert isinstance(e, Specter2Embedder)
    assert e._model is None  # model must NOT load at construction (keyword-only pays nothing)


def test_factory_bogus(monkeypatch):
    monkeypatch.setenv("CARDINAL_EMBEDDER", "bogus")
    with pytest.raises(ValueError):
        embed.get_embedder()


def test_hash_deterministic_and_dim():
    e = HashEmbedder()
    assert e.encode("hello") == e.encode("hello")          # deterministic
    assert len(e.encode("hello")) == 768 * 4               # 768 float32
    assert e.encode("hello") != e.encode("world")          # content-sensitive
