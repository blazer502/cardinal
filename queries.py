"""Deterministic query layer for the six MCP tools (PLAN.md §3).

Pure functions over a sqlite3 connection — no MCP, no I/O beyond the DB — so
they are unit-testable on their own. search/neighbors/expand/subgraph/get_cluster
are 0 LLM tokens. ingest is the single LLM touchpoint (amortized via ingest_cache).

Security invariant (PLAN.md §9): every agent-supplied value is a bind parameter,
including the FTS5 MATCH string. Never string-interpolate agent input into SQL.
"""
import hashlib
import json
import re
import sqlite3
import struct

_WORD = re.compile(r"[0-9a-zA-Z]+")

# card columns pulled straight from the paper table (superset of the v_card view;
# venue is included so format="json" can return it, per §3.1).
_SELECT = (
    "p.paper_id, p.year, p.n_citations, p.cluster_id, p.fields_status, "
    "p.title, p.tldr, p.tags, p.venue"
)


class EmbedderUnavailable(RuntimeError):
    """Raised when semantic search is requested but no query embedder is configured."""


# ---------------------------------------------------------------------------
# filters (§3.1) — pre-applied as bound WHERE on the card table
# ---------------------------------------------------------------------------
def _filter_sql(filters: dict | None):
    clauses, params = [], []
    f = filters or {}
    if f.get("year_min") is not None:
        clauses.append("p.year >= ?"); params.append(f["year_min"])
    if f.get("year_max") is not None:
        clauses.append("p.year <= ?"); params.append(f["year_max"])
    if f.get("cluster_id") is not None:
        clauses.append("p.cluster_id = ?"); params.append(f["cluster_id"])
    if f.get("venue"):
        clauses.append("p.venue = ?"); params.append(f["venue"])
    for tag in f.get("tags") or []:  # tags is a comma-separated flat string
        clauses.append("(',' || IFNULL(p.tags,'') || ',') LIKE ?"); params.append(f"%,{tag},%")
    return " AND ".join(clauses), params


def _row_matches(row: dict, filters: dict | None) -> bool:
    """Same predicate as _filter_sql, in Python (for post-KNN filtering)."""
    f = filters or {}
    if f.get("year_min") is not None and (row["year"] is None or row["year"] < f["year_min"]):
        return False
    if f.get("year_max") is not None and (row["year"] is None or row["year"] > f["year_max"]):
        return False
    if f.get("cluster_id") is not None and row["cluster_id"] != f["cluster_id"]:
        return False
    if f.get("venue") and row.get("venue") != f["venue"]:
        return False
    tags = set((row["tags"] or "").split(","))
    return all(t in tags for t in (f.get("tags") or []))


# ---------------------------------------------------------------------------
# search (§3.1)
# ---------------------------------------------------------------------------
def _fts_query(text):
    """Turn a natural-language query into an OR of quoted tokens.

    FTS5's implicit operator is AND, which makes multi-word NL queries brittle
    (every word, incl. stopwords, must appear). We OR the alphanumeric tokens so
    BM25 ranks by term overlap; quoting each token also neutralizes FTS operators,
    so the (still bound) MATCH string can't be injected with."""
    return " OR ".join(f'"{t}"' for t in _WORD.findall(text.lower()))


def _keyword(conn, query, k, where, wparams):
    match = _fts_query(query or "")
    if not match:
        return []
    sql = (f"SELECT {_SELECT}, bm25(fts_paper) AS score "
           "FROM fts_paper JOIN paper p ON p.rowid = fts_paper.rowid "
           "WHERE fts_paper MATCH ?")
    params = [match]
    if where:
        sql += " AND " + where; params += wparams
    sql += " ORDER BY score LIMIT ?"; params.append(k)  # bm25: lower = better
    return [dict(r) for r in conn.execute(sql, params)]


def _semantic(conn, query, k, filters, embedder):
    if embedder is None:
        raise EmbedderUnavailable(
            "semantic search needs a query embedder — set CARDINAL_EMBEDDER "
            "and wire a SPECTER2-family model (see embed.py / PLAN.md §3)")
    qvec = embedder.encode(query)
    kk = k * 4 if filters else k  # over-fetch, then apply metadata filters post-KNN
    sql = (f"SELECT {_SELECT}, v.distance AS score "
           "FROM vec_paper v JOIN paper p ON p.rowid = v.paper_rowid "
           "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance")
    rows = [dict(r) for r in conn.execute(sql, [qvec, kk])]
    if filters:
        rows = [r for r in rows if _row_matches(r, filters)]
    return rows[:k]


def _rrf(rankings, k0=60):
    """Reciprocal Rank Fusion: score = Σ 1/(k0 + rank) (§3.1 hybrid)."""
    scores: dict = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k0 + rank + 1)
    return sorted(scores, key=scores.get, reverse=True), scores


def _hybrid(conn, query, k, filters, where, wparams, embedder):
    kw = _keyword(conn, query, max(k, 50), where, wparams)
    sem = _semantic(conn, query, max(k, 50), filters, embedder)
    order, scores = _rrf([[r["paper_id"] for r in kw], [r["paper_id"] for r in sem]])
    by_id = {r["paper_id"]: r for r in kw}
    for r in sem:
        by_id.setdefault(r["paper_id"], r)
    out = []
    for pid in order[:k]:
        row = dict(by_id[pid]); row["score"] = round(scores[pid], 6); out.append(row)
    return out


def search(conn, query, k=20, mode="hybrid", filters=None, embedder=None):
    where, wparams = _filter_sql(filters)
    if mode == "keyword" or (mode == "hybrid" and embedder is None):
        return _keyword(conn, query, k, where, wparams)  # hybrid degrades to keyword w/o embedder
    if mode == "semantic":
        return _semantic(conn, query, k, filters, embedder)
    if mode == "hybrid":
        return _hybrid(conn, query, k, filters, where, wparams, embedder)
    raise ValueError(f"unknown search mode {mode!r} (expected hybrid|keyword|semantic)")


# ---------------------------------------------------------------------------
# neighbors (§3.2) — pure index range scan over edge
# ---------------------------------------------------------------------------
# kind -> (key column to match :id, column to return, stored edge.kind)
_KIND_DIR = {
    "cites":          ("src", "dst", "cites"),
    "cited_by":       ("dst", "src", "cites"),
    "similar":        ("src", "dst", "similar"),
    "shared_method":  ("src", "dst", "shared_method"),
    "shared_dataset": ("src", "dst", "shared_dataset"),
}


def neighbors(conn, paper_id, kind, k=15, min_weight=0.0):
    if kind not in _KIND_DIR:
        raise ValueError(f"unknown kind {kind!r} (expected {'|'.join(_KIND_DIR)})")
    key_col, ret_col, edge_kind = _KIND_DIR[kind]
    sql = (f"SELECT {_SELECT}, e.weight, e.intent "
           f"FROM edge e JOIN paper p ON p.paper_id = e.{ret_col} "
           f"WHERE e.{key_col} = ? AND e.kind = ? AND e.weight >= ? "
           "ORDER BY e.weight DESC LIMIT ?")
    return [dict(r) for r in conn.execute(sql, [paper_id, edge_kind, min_weight, k])]


# ---------------------------------------------------------------------------
# expand (§3.3) — heavy layers, opt-in only
# ---------------------------------------------------------------------------
_FIELD_COLS = ("problem", "method", "dataset", "metric", "result", "limitation")


def expand(conn, paper_ids, level, query=None, embedder=None, k_chunks=6):
    if level == "fields":
        out = {}
        for pid in paper_ids:
            r = conn.execute(
                f"SELECT {','.join(_FIELD_COLS)} FROM paper_fields WHERE paper_id=?",
                (pid,)).fetchone()
            out[pid] = dict(r) if r else None
        return out
    if level == "abstract":
        out = {}
        for pid in paper_ids:
            r = conn.execute("SELECT abstract FROM paper_abstract WHERE paper_id=?",
                             (pid,)).fetchone()
            out[pid] = r["abstract"] if r else None
        return out
    if level == "chunks":
        return _expand_chunks(conn, paper_ids, query, embedder, k_chunks)
    raise ValueError(f"unknown level {level!r} (expected fields|abstract|chunks)")


def _expand_chunks(conn, paper_ids, query, embedder, k):
    out: dict = {pid: [] for pid in paper_ids}
    wanted = set(paper_ids)
    if query and embedder is not None:  # rank chunks by similarity to the query
        qvec = embedder.encode(query)
        rows = conn.execute(
            "SELECT c.paper_id, c.section, c.ord, c.text "
            "FROM vec_chunk v JOIN paper_chunk c ON c.chunk_id = v.chunk_id "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            [qvec, k * max(len(paper_ids), 1) * 8]).fetchall()  # over-fetch, filter below
        for r in rows:
            pid = r["paper_id"]
            if pid in wanted and len(out[pid]) < k:
                out[pid].append({"section": r["section"], "ord": r["ord"], "text": r["text"]})
    else:  # no query: return chunks in document order
        ph = ",".join("?" * len(paper_ids))
        rows = conn.execute(
            f"SELECT paper_id, section, ord, text FROM paper_chunk "
            f"WHERE paper_id IN ({ph}) ORDER BY paper_id, ord", list(paper_ids))
        for r in rows:
            out[r["paper_id"]].append({"section": r["section"], "ord": r["ord"], "text": r["text"]})
    return out


# ---------------------------------------------------------------------------
# subgraph (§3.4) — BFS for the graph view
# ---------------------------------------------------------------------------
def subgraph(conn, seeds, hops=1, kinds=None, max_nodes=60, min_weight=0.0):
    kinds = kinds or ["cites", "similar"]
    kph = ",".join("?" * len(kinds))
    nodes: dict = {}
    edges = []
    seen_edges = set()

    def add_node(pid):
        if pid in nodes:
            return
        r = conn.execute(
            "SELECT paper_id, title, year, n_citations, cluster_id FROM paper WHERE paper_id=?",
            (pid,)).fetchone()
        if r:
            nodes[pid] = dict(r)

    frontier = list(dict.fromkeys(seeds))
    for pid in frontier:
        add_node(pid)
    visited = set()
    for _ in range(hops):
        nxt = []
        for pid in frontier:
            if pid in visited or pid not in nodes:
                continue
            visited.add(pid)
            for e in conn.execute(
                f"SELECT src, dst, kind, weight FROM edge "
                f"WHERE (src=? OR dst=?) AND kind IN ({kph}) AND weight >= ? "
                "ORDER BY weight DESC", [pid, pid, *kinds, min_weight]):
                other = e["dst"] if e["src"] == pid else e["src"]
                if other not in nodes and len(nodes) >= max_nodes:
                    continue  # node cap reached — don't grow further
                add_node(other)
                ekey = (e["src"], e["dst"], e["kind"])
                if ekey not in seen_edges and e["src"] in nodes and e["dst"] in nodes:
                    seen_edges.add(ekey)
                    edges.append({"src": e["src"], "dst": e["dst"],
                                  "kind": e["kind"], "weight": e["weight"]})
                    nxt.append(other)
        frontier = nxt
    return {"nodes": list(nodes.values()), "edges": edges}


# ---------------------------------------------------------------------------
# get_cluster (§3.5) — precomputed lookup, not an LLM call
# ---------------------------------------------------------------------------
def get_cluster(conn, cluster_id=None, paper_id=None, top_k=10):
    if cluster_id is None:
        if paper_id is None:
            raise ValueError("get_cluster requires cluster_id or paper_id")
        r = conn.execute("SELECT cluster_id FROM paper WHERE paper_id=?", (paper_id,)).fetchone()
        if not r or r["cluster_id"] is None:
            return None
        cluster_id = r["cluster_id"]
    c = conn.execute("SELECT cluster_id, label, summary, size FROM cluster WHERE cluster_id=?",
                     (cluster_id,)).fetchone()
    if not c:
        return None
    tops = [dict(r) for r in conn.execute(
        f"SELECT {_SELECT} FROM paper p WHERE p.cluster_id=? "
        "ORDER BY p.n_citations DESC LIMIT ?", (cluster_id, top_k))]
    out = dict(c)
    out["top_papers"] = tops
    return out


# ---------------------------------------------------------------------------
# ingest (§3.6) — the only LLM touchpoint, amortized via ingest_cache
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    return " ".join((text or "").split()).lower()


def content_hash(title: str, abstract: str | None) -> str:
    return hashlib.sha256(
        f"{_normalize(title)}\n{_normalize(abstract or '')}".encode("utf-8")).hexdigest()


def _store_vec(conn, paper_id, vec):
    """vec: a list/tuple of floats (dim-checked) OR already-packed float32 bytes."""
    if isinstance(vec, (list, tuple)):
        if len(vec) != 768:
            raise ValueError(
                f"embedding dim {len(vec)} != 768 (schema vec_paper is FLOAT[768]); paper {paper_id}")
        blob = struct.pack(f"{len(vec)}f", *vec)
    else:
        blob = vec
    rid = conn.execute("SELECT rowid FROM paper WHERE paper_id=?", (paper_id,)).fetchone()["rowid"]
    conn.execute("DELETE FROM vec_paper WHERE paper_rowid=?", (rid,))
    conn.execute("INSERT INTO vec_paper(paper_rowid, embedding) VALUES (?,?)", (rid, blob))


def ingest(conn, source, prompt_version="v1", force=False, embedder=None, extractor=None):
    """Add/update one paper.

    SCAFFOLD SCOPE: the deterministic pipeline (hash → ingest_cache → upsert L0/L2
    → vector) is implemented. Two plug-in points are left for later steps of §7:
      - external metadata fetch (S2/OpenAlex): pass source={"raw": {...}} for now;
        an id-only source raises NotImplementedError.
      - L1 field extraction: pass an `extractor(title, abstract) -> dict` (a small
        model, §7.4). Without one, the card is stored with fields_status='none'.
    """
    raw = (source or {}).get("raw")
    if not raw:
        raise NotImplementedError(
            "S2/OpenAlex fetch not wired in this scaffold — pass "
            "source={'raw': {'title':..., 'abstract':...}} (see PLAN.md §7.1)")

    title = raw["title"]
    abstract = raw.get("abstract")
    chash = content_hash(title, abstract)
    paper_id = raw.get("paper_id") or ("H:" + chash[:16])

    # L1 fields: cache-first (amortization). §3.6 step 3-4.
    fields, llm_calls, cached = None, 0, True
    hit = conn.execute(
        "SELECT output FROM ingest_cache WHERE content_hash=? AND task='fields' AND prompt_version=?",
        (chash, prompt_version)).fetchone()
    if hit is not None and not force:
        fields = json.loads(hit["output"])
    elif extractor is not None:
        fields = extractor(title, abstract)
        llm_calls, cached = 1, False
        conn.execute(
            "INSERT OR REPLACE INTO ingest_cache(content_hash,task,prompt_version,model,output) "
            "VALUES (?,?,?,?,?)",
            (chash, "fields", prompt_version, getattr(extractor, "model", None), json.dumps(fields)))

    tags = raw.get("tags")
    if isinstance(tags, list):
        tags = ",".join(tags)

    # L0 card upsert (FTS stays in sync via the paper_ai/au triggers)
    conn.execute(
        "INSERT INTO paper(paper_id,title,year,venue,tldr,contribution,tags,n_citations,"
        "content_hash,fields_status,okf_path,okf_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(paper_id) DO UPDATE SET title=excluded.title, year=excluded.year, "
        "venue=excluded.venue, tldr=excluded.tldr, contribution=excluded.contribution, "
        "tags=excluded.tags, n_citations=excluded.n_citations, content_hash=excluded.content_hash, "
        "fields_status=excluded.fields_status, okf_path=excluded.okf_path, okf_version=excluded.okf_version",
        (paper_id, title, raw.get("year"), raw.get("venue"), raw.get("tldr"),
         raw.get("contribution"), tags, raw.get("n_citations", 0), chash,
         "extracted" if fields else "none", raw.get("okf_path"), raw.get("okf_version")))

    if fields:
        conn.execute(
            "INSERT INTO paper_fields(paper_id,problem,method,dataset,metric,result,limitation,"
            "extractor_model,prompt_version,source_hash) VALUES (?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(paper_id) DO UPDATE SET problem=excluded.problem, method=excluded.method, "
            "dataset=excluded.dataset, metric=excluded.metric, result=excluded.result, "
            "limitation=excluded.limitation, prompt_version=excluded.prompt_version, "
            "source_hash=excluded.source_hash",
            (paper_id, *(fields.get(c) for c in _FIELD_COLS),
             getattr(extractor, "model", None), prompt_version, chash))

    if abstract is not None:
        conn.execute(
            "INSERT INTO paper_abstract(paper_id,abstract) VALUES (?,?) "
            "ON CONFLICT(paper_id) DO UPDATE SET abstract=excluded.abstract", (paper_id, abstract))

    # vectors: a precomputed corpus embedding (e.g. S2 SPECTER2) wins; else the local embedder
    vec = raw.get("embedding")
    if vec is not None:
        _store_vec(conn, paper_id, vec)
    elif embedder is not None:
        _store_vec(conn, paper_id, embedder.encode(f"{title}\n{abstract or ''}"))

    for eid in raw.get("ext_ids") or []:  # external id mapping (dedup/lookup, not hot path)
        conn.execute("INSERT OR IGNORE INTO paper_ext_id(paper_id,scheme,value) VALUES (?,?,?)",
                     (paper_id, eid["scheme"], eid["value"]))

    conn.commit()
    return {"paper_id": paper_id, "content_hash": chash,
            "fields_status": "extracted" if fields else "none",
            "cached": cached, "llm_calls": llm_calls, "okf_path": raw.get("okf_path")}
