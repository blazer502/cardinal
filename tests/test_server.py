import asyncio
import json

import server
from embed import HashEmbedder


def test_all_tools_registered():
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert names == {"search", "neighbors", "expand", "subgraph", "get_cluster",
                     "ingest", "seed_topic", "export_okf"}


def test_instructions_guide_the_agent():
    assert "seed_topic" in server.INSTRUCTIONS and "0 model tokens" in server.INSTRUCTIONS


def test_search_tool_returns_cards(seeded, monkeypatch):
    monkeypatch.setattr(server, "_conn", seeded)
    monkeypatch.setattr(server, "_embedder", HashEmbedder())
    out = server.search("trojaning", mode="keyword", format="cards")
    assert out.startswith("cards[") and "S2:22" in out


def test_neighbors_tool_ids(seeded, monkeypatch):
    monkeypatch.setattr(server, "_conn", seeded)
    out = json.loads(server.neighbors("S2:14", "cited_by", format="ids"))
    assert {n["id"] for n in out["neighbors"]} == {"S2:22", "S2:31"}


def test_export_okf_tool(seeded, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_conn", seeded)
    out = json.loads(server.export_okf(str(tmp_path / "okf")))
    assert out["concepts"] == 3
