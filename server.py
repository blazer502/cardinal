"""Cardinal MCP server — agent-friendly related-work DB (PLAN.md §3, §7.5).

Thin MCP layer over queries.py. Tools search/neighbors/expand/subgraph/get_cluster
cost 0 LLM tokens; ingest is the single (amortized) LLM touchpoint. Default returns
are the compressed cards projection (§4); pass format="json"|"ids" for full objects.

Run as a stdio MCP server:  python server.py   (needs index.db built via build_index.py)
"""
import json
import sqlite3

from mcp.server.fastmcp import FastMCP

import cards
import db
import okf
import queries
import seed as seeder
from embed import get_embedder

INSTRUCTIONS = """Cardinal is a related-work knowledge graph for constructing and \
analyzing literature. Typical workflow:
  1. seed_topic(query) — construct/expand the graph from Semantic Scholar/OpenAlex (network).
  2. search(query) — cheap hybrid scan returning compact cards. Start analysis here.
  3. neighbors(paper_id, kind) to walk citations/similarity; subgraph(seeds) for a map.
  4. expand(paper_ids, level) only for the few papers needing structured fields/abstract/chunks.
  5. get_cluster to read a precomputed cluster; export_okf to persist the canonical bundle.
search/neighbors/expand/subgraph/get_cluster cost 0 model tokens (deterministic SQL). \
Default returns are compressed cards; pass format="json"|"ids" for full objects."""

mcp = FastMCP("cardinal-related-work", instructions=INSTRUCTIONS)

_conn: sqlite3.Connection | None = None
_embedder = get_embedder()


def conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = db.connect()
    return _conn


def _err(msg: str) -> str:
    return json.dumps({"error": msg})


def _fmt_results(rows: list[dict], fmt: str) -> str:
    if fmt == "ids":
        return json.dumps([r["paper_id"] for r in rows])
    if fmt == "json":
        return json.dumps({"results": rows}, ensure_ascii=False)
    extra = ["score"] if rows and "score" in rows[0] else None
    return cards.render_cards(rows, extra=extra)


@mcp.tool()
def search(query: str, k: int = 20, mode: str = "hybrid",
           filters: dict | None = None, format: str = "cards") -> str:
    """Hybrid search (BM25 + vector RRF), 0 LLM tokens.
    mode: hybrid|keyword|semantic. filters: year_min/year_max/tags/cluster_id/venue.
    format: cards (default, compact) | json | ids. Without an embedder, hybrid falls back to keyword."""
    try:
        rows = queries.search(conn(), query, k=k, mode=mode, filters=filters, embedder=_embedder)
    except (queries.EmbedderUnavailable, ValueError) as e:
        return _err(str(e))
    except sqlite3.OperationalError as e:
        return _err(f"query error: {e}")
    return _fmt_results(rows, format)


@mcp.tool()
def neighbors(paper_id: str, kind: str, k: int = 15,
              min_weight: float = 0.0, format: str = "cards") -> str:
    """Graph neighbors of a paper by edge kind, 0 LLM tokens.
    kind: cites|cited_by|similar|shared_method|shared_dataset. format: cards|json|ids."""
    try:
        rows = queries.neighbors(conn(), paper_id, kind, k=k, min_weight=min_weight)
    except ValueError as e:
        return _err(str(e))
    if format == "ids":
        return json.dumps({"paper_id": paper_id, "kind": kind,
                           "neighbors": [{"id": r["paper_id"], "weight": r["weight"],
                                          "intent": r["intent"]} for r in rows]})
    if format == "json":
        return json.dumps({"paper_id": paper_id, "kind": kind, "neighbors": rows}, ensure_ascii=False)
    return cards.render_cards(rows, extra=["weight"])


@mcp.tool()
def expand(paper_ids: list[str], level: str, query: str | None = None) -> str:
    """Drill down into heavy layers (opt-in), 0 LLM tokens.
    level: fields | abstract | chunks. query: for chunks, selects the most relevant chunks."""
    try:
        out = queries.expand(conn(), paper_ids, level, query=query, embedder=_embedder)
    except ValueError as e:
        return _err(str(e))
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
def subgraph(seeds: list[str], hops: int = 1, kinds: list[str] | None = None,
             max_nodes: int = 60, min_weight: float = 0.0) -> str:
    """Extract a subgraph for the graph view (nodes + edges), 0 LLM tokens.
    kinds default: [cites, similar]. Node size=n_citations, color=year."""
    return json.dumps(
        queries.subgraph(conn(), seeds, hops=hops, kinds=kinds,
                         max_nodes=max_nodes, min_weight=min_weight), ensure_ascii=False)


@mcp.tool()
def get_cluster(cluster_id: int | None = None, paper_id: str | None = None) -> str:
    """Look up a precomputed cluster (label, summary, top papers), 0 LLM tokens.
    Pass cluster_id, or paper_id to resolve its cluster."""
    try:
        res = queries.get_cluster(conn(), cluster_id=cluster_id, paper_id=paper_id)
    except ValueError as e:
        return _err(str(e))
    return json.dumps(res, ensure_ascii=False)


@mcp.tool()
def ingest(source: dict, prompt_version: str = "v1", force: bool = False) -> str:
    """Add/update a paper — the only LLM touchpoint (amortized via ingest_cache).
    Scaffold: pass source={"raw": {"title":..., "abstract":..., ...}}; external
    fetch (S2/OpenAlex) and LLM field extraction are wired in later steps of §7."""
    try:
        res = queries.ingest(conn(), source, prompt_version=prompt_version,
                             force=force, embedder=_embedder, extractor=None)
    except NotImplementedError as e:
        return _err(str(e))
    return json.dumps(res, ensure_ascii=False)


@mcp.tool()
def seed_topic(query: str, source: str = "openalex", limit: int = 30,
               similar_k: int = 0, okf_dir: str | None = None) -> str:
    """Construct/expand the graph from Semantic Scholar or OpenAlex (network fetch).
    source: openalex (no key) | s2 (adds TLDR + SPECTER2 vectors; needs S2_API_KEY).
    similar_k: also add N vector-similarity edges/paper. okf_dir: also emit the OKF bundle.
    After this, use search/neighbors/subgraph to analyze the newly added papers."""
    try:
        stats = seeder.seed(db.DEFAULT_DB, query=query, source=source, limit=limit,
                            similar_k=similar_k, okf_dir=okf_dir)
    except Exception as e:  # network/API failures surface as a tool error, not a crash
        return _err(f"{type(e).__name__}: {e}")
    return json.dumps(stats)


@mcp.tool()
def export_okf(out_dir: str = "okf") -> str:
    """Write the OKF canonical bundle (one markdown+YAML concept per paper + index.md)
    from the current index, recording each concept's path onto the paper."""
    return json.dumps({"concepts": okf.export(conn(), out_dir), "out": f"{out_dir}/papers/"})


if __name__ == "__main__":
    mcp.run()
