#!/usr/bin/env python3
"""Seed the derived index from Semantic Scholar / OpenAlex (PLAN.md §7.1).

Fetches nodes + edges + TLDR + SPECTER2 via public APIs (no custom crawler),
normalizes each record, and upserts through queries.ingest. Citation edges are
built among the seeded set; optional `similar` edges come from SPECTER2 top-k.

Backends:
  s2       — Semantic Scholar Graph API: TLDR + SPECTER2 vectors + citations.
             Rate-limited without a key; set S2_API_KEY for reliable/bulk use.
  openalex — OpenAlex: title/abstract/venue/citations + references. No key
             needed (set OPENALEX_MAILTO for the polite pool). No embeddings —
             semantic search needs a local embedder (CARDINAL_EMBEDDER).

Usage:
  python seed.py --query "backdoor attacks on neural networks" --source openalex --limit 30
  S2_API_KEY=... python seed.py --query "..." --source s2 --similar-k 5
  python seed.py --ids W2018,W2019 --source openalex
All paths/keys are relative or env-driven — no machine-specific configuration.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import build_index
import db
import okf
import queries
from embed import get_embedder

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = ("corpusId,title,abstract,year,venue,citationCount,tldr,externalIds,"
             "fieldsOfStudy,embedding.specter_v2,references.corpusId")
OA_BASE = "https://api.openalex.org"


# ---------------------------------------------------------------------------
# HTTP (stdlib only) with backoff on 429/5xx
# ---------------------------------------------------------------------------
def _request(url, *, headers=None, data=None, timeout=20, retries=4):
    body = json.dumps(data).encode() if data is not None else None
    hdrs = {"User-Agent": "cardinal-seeder/0.1", "Accept": "application/json"}
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    delay = 1.0
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=hdrs,
                                     method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay); delay *= 2; continue
            if e.code == 429:
                raise RuntimeError(
                    "429 rate-limited by the API. Set S2_API_KEY (S2) or retry later, "
                    "or use --source openalex.") from e
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(delay); delay *= 2; continue
            raise
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------
def _s2_headers():
    key = os.environ.get("S2_API_KEY")
    return {"x-api-key": key} if key else {}


def s2_search_ids(query, limit):
    url = (f"{S2_BASE}/paper/search?query={urllib.parse.quote(query)}"
           f"&limit={min(limit, 100)}&fields=paperId")
    d = _request(url, headers=_s2_headers())
    return [p["paperId"] for p in d.get("data", []) if p.get("paperId")]


def s2_fetch(paper_ids):
    if not paper_ids:
        return []
    url = f"{S2_BASE}/paper/batch?fields={S2_FIELDS}"
    return _request(url, headers=_s2_headers(), data={"ids": paper_ids})


def _s2_to_raw(rec):
    if not rec or rec.get("corpusId") is None or not rec.get("title"):
        return None
    ext = rec.get("externalIds") or {}
    ext_ids = [{"scheme": s, "value": str(ext[k])}
               for s, k in (("doi", "DOI"), ("arxiv", "ArXiv"),
                            ("mag", "MAG"), ("openalex", "OpenAlex")) if ext.get(k)]
    ext_ids.append({"scheme": "s2", "value": str(rec["corpusId"])})
    tags = [f["category"] if isinstance(f, dict) else f
            for f in (rec.get("fieldsOfStudy") or [])]
    refs = [f"S2:{r['corpusId']}" for r in (rec.get("references") or [])
            if r.get("corpusId") is not None]
    return {
        "paper_id": f"S2:{rec['corpusId']}",
        "title": rec["title"], "abstract": rec.get("abstract"),
        "year": rec.get("year"), "venue": rec.get("venue"),
        "tldr": (rec.get("tldr") or {}).get("text"),
        "tags": tags[:6], "n_citations": rec.get("citationCount") or 0,
        "embedding": (rec.get("embedding") or {}).get("vector"),
        "ext_ids": ext_ids, "refs": refs,
    }


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------
def _oa_mailto():
    return os.environ.get("OPENALEX_MAILTO", "cardinal@example.com")


def oa_search(query, limit):
    url = (f"{OA_BASE}/works?search={urllib.parse.quote(query)}"
           f"&per-page={min(limit, 200)}&mailto={_oa_mailto()}")
    return _request(url).get("results", [])


def oa_fetch_ids(ids):
    return [_request(f"{OA_BASE}/works/{i}?mailto={_oa_mailto()}") for i in ids]


def _oa_short(oa_id):  # "https://openalex.org/W123" -> "W123"
    return (oa_id or "").rsplit("/", 1)[-1]


def _reconstruct_abstract(inv):
    if not inv:
        return None
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos)) or None


def _oa_to_raw(w):
    oid = _oa_short(w.get("id"))
    title = w.get("title") or w.get("display_name")
    if not oid or not title:
        return None
    ids = w.get("ids") or {}
    ext_ids = [{"scheme": "openalex", "value": oid}]
    if ids.get("doi"):
        ext_ids.append({"scheme": "doi", "value": ids["doi"].split("doi.org/")[-1]})
    if ids.get("mag"):
        ext_ids.append({"scheme": "mag", "value": str(ids["mag"])})
    venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
    tags = [c["display_name"] for c in (w.get("concepts") or [])[:5] if c.get("display_name")]
    refs = [f"OA:{_oa_short(r)}" for r in (w.get("referenced_works") or [])]
    return {
        "paper_id": f"OA:{oid}",
        "title": title, "abstract": _reconstruct_abstract(w.get("abstract_inverted_index")),
        "year": w.get("publication_year"), "venue": venue,
        "tldr": None, "tags": tags, "n_citations": w.get("cited_by_count") or 0,
        "embedding": None, "ext_ids": ext_ids, "refs": refs,
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def _ensure_schema(db_path):
    conn = db.connect(db_path)
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper'").fetchone()
    conn.close()
    if not exists:  # fresh DB — apply schema.sql (build removes the empty file first)
        build_index.build(db_path).close()


def build_similar_edges(conn, paper_ids, k):
    """Add `similar` edges from each paper's top-k nearest vectors (any embedding source)."""
    n = 0
    for pid in paper_ids:
        row = conn.execute(
            "SELECT v.embedding FROM vec_paper v JOIN paper p ON p.rowid = v.paper_rowid "
            "WHERE p.paper_id = ?", (pid,)).fetchone()
        if row is None:  # paper has no stored vector — skip
            continue
        for opid, dist in conn.execute(
            "SELECT p.paper_id, v.distance FROM vec_paper v "
            "JOIN paper p ON p.rowid = v.paper_rowid "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance", (row["embedding"], k + 1)):
            if opid == pid:
                continue
            conn.execute("INSERT OR IGNORE INTO edge(src,dst,kind,weight,intent) VALUES (?,?,?,?,?)",
                         (pid, opid, "similar", 1.0 / (1.0 + dist), None))
            n += 1
    conn.commit()
    return n


def seed(db_path, *, query=None, ids=None, source="openalex", limit=30, similar_k=0, okf_dir=None):
    _ensure_schema(db_path)
    conn = db.connect(db_path)
    embedder = get_embedder()

    if source == "s2":
        recs = s2_fetch(ids if ids else s2_search_ids(query, limit))
        recs = [_s2_to_raw(r) for r in recs]
    elif source == "openalex":
        works = oa_fetch_ids(ids) if ids else oa_search(query, limit)
        recs = [_oa_to_raw(w) for w in works]
    else:
        raise SystemExit(f"unknown source {source!r}")
    recs = [r for r in recs if r]

    seeded = set()
    for r in recs:
        res = queries.ingest(conn, {"raw": r}, embedder=embedder)
        seeded.add(res["paper_id"])

    n_cites = 0  # citation edges, closed over the seeded set (FK-safe)
    for r in recs:
        for ref in r["refs"]:
            if ref in seeded and ref != r["paper_id"]:
                conn.execute("INSERT OR IGNORE INTO edge(src,dst,kind,weight,intent) VALUES (?,?,?,?,?)",
                             (r["paper_id"], ref, "cites", 1.0, None))
                n_cites += 1
    conn.commit()

    n_sim = build_similar_edges(conn, seeded, similar_k) if similar_k > 0 else 0
    n_okf = okf.export(conn, okf_dir) if okf_dir else 0
    n_vec = conn.execute("SELECT count(*) FROM vec_paper").fetchone()[0]
    conn.close()
    msg = (f"seeded {len(seeded)} papers, {n_cites} cites edges, {n_sim} similar edges "
           f"from {source} ({n_vec} vectors in index)")
    if okf_dir:
        msg += f"; wrote {n_okf} OKF concepts to {okf_dir}/papers/"
    print(msg)
    return {"papers": len(seeded), "cites": n_cites, "similar": n_sim, "okf": n_okf}


def main():
    ap = argparse.ArgumentParser(description="Seed the derived index from S2/OpenAlex (PLAN.md §7.1)")
    ap.add_argument("--query", help="search query")
    ap.add_argument("--ids", help="comma-separated source ids (S2 paperIds or OpenAlex work ids)")
    ap.add_argument("--source", choices=["s2", "openalex"], default="openalex")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--db", default=db.DEFAULT_DB)
    ap.add_argument("--similar-k", type=int, default=0, dest="similar_k",
                    help="build N similar edges/paper from SPECTER2 (needs embeddings)")
    ap.add_argument("--okf-dir", default=None, dest="okf_dir",
                    help="also emit the OKF canonical bundle to this dir (PLAN.md §7.2)")
    a = ap.parse_args()
    if not a.query and not a.ids:
        ap.error("provide --query or --ids")
    ids = [x.strip() for x in a.ids.split(",")] if a.ids else None
    seed(a.db, query=a.query, ids=ids, source=a.source, limit=a.limit,
         similar_k=a.similar_k, okf_dir=a.okf_dir)


if __name__ == "__main__":
    main()
