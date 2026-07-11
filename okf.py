#!/usr/bin/env python3
"""OKF canonical bundle <-> derived index (PLAN.md §7.2, §7.3, §3.6 step 5).

The OKF bundle is the source of truth: one concept = one markdown+YAML file,
cross-links = edges, plus a progressive-disclosure index.md. SPECTER2 vectors
(which can't be recomputed offline) travel as per-concept `.vec.json` sidecars,
so the bundle is self-contained. The SQLite index is a derived artifact:

  export(conn, dir)   index      -> OKF bundle   (materialize the source of truth)
  rebuild(dir, db)    OKF bundle -> index        (regenerate the derived index)

rebuild reproduces papers, structured fields, abstracts, external ids, edges, and
vectors — and repopulates ingest_cache so LLM amortization survives a rebuild.
Similar-edge weights are preserved to 6 decimals; citation edges are weight 1.0.

Frontmatter values are JSON-encoded (valid YAML) so the bundle round-trips with
no external YAML dependency.

Usage:
  python okf.py export  --db index.db --out okf/
  python okf.py rebuild --in okf/    --db index.db
"""
import argparse
import json
import os
import re
import struct
from collections import defaultdict

import build_index
import db
import queries
from embed import get_embedder

OKF_VERSION = "0.1"
_FIELD_COLS = ("problem", "method", "dataset", "metric", "result", "limitation")


def _slug(paper_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", paper_id)


def _frontmatter(meta: dict) -> str:
    # JSON scalars/sequences are valid YAML flow syntax -> unambiguous round-trip
    return "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items())


# =========================================================================== export
def _render_concept(r, fields, abstract, cites, similar, ext_ids, titles, ver):
    meta = {
        "okf_version": ver, "id": r["paper_id"], "title": r["title"],
        "year": r["year"], "venue": r["venue"], "n_citations": r["n_citations"],
        "tags": r["tags"].split(",") if r["tags"] else [],
        "cluster": r["cluster_id"], "content_hash": r["content_hash"],
        "tldr": r["tldr"], "contribution": r["contribution"],
        "ext_ids": [{"scheme": e["scheme"], "value": e["value"]} for e in ext_ids],
    }
    if fields:  # keep extractor provenance so paper_fields + ingest_cache round-trip
        meta["fields_prompt_version"] = fields["prompt_version"]
        meta["fields_extractor"] = fields["extractor_model"]
    out = ["---", _frontmatter(meta), "---", "", f"# {r['title']}", ""]
    if r["tldr"]:
        out += [r["tldr"], ""]
    if fields and any(fields[c] for c in _FIELD_COLS):
        out.append("## Structured fields")
        out += [f"- **{c}:** {fields[c]}" for c in _FIELD_COLS if fields[c]]
        out.append("")
    if abstract and abstract["abstract"]:
        out += ["## Abstract", abstract["abstract"], ""]
    if cites:
        out.append("## Cites")
        out += [f"- [{titles.get(e['dst'], e['dst'])}]({_slug(e['dst'])}.md)" for e in cites]
        out.append("")
    if similar:
        out.append("## Similar")
        out += [f"- [{titles.get(e['dst'], e['dst'])}]({_slug(e['dst'])}.md) — {e['weight']:.6f}"
                for e in similar]
        out.append("")
    return "\n".join(out)


def _write_index(papers_dir, rows, ver):
    by_year = defaultdict(list)
    for r in rows:
        by_year[r["year"] or 0].append(r)
    out = ["---", _frontmatter({"okf_version": ver, "kind": "index", "count": len(rows)}),
           "---", "", "# Related-work concepts", ""]
    for year in sorted(by_year, reverse=True):
        out.append(f"## {year if year else 'Unknown year'}")
        for r in sorted(by_year[year], key=lambda x: -(x["n_citations"] or 0)):
            out.append(f"- [{r['title']}]({_slug(r['paper_id'])}.md) — {r['n_citations'] or 0} cit")
        out.append("")
    with open(os.path.join(papers_dir, "index.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))


def export(conn, okf_dir, okf_version=OKF_VERSION):
    """Write the full OKF bundle (+ vector sidecars) and record paper.okf_path."""
    papers_dir = os.path.join(okf_dir, "papers")
    os.makedirs(papers_dir, exist_ok=True)
    rows = conn.execute("SELECT rowid, * FROM paper ORDER BY paper_id").fetchall()
    titles = {r["paper_id"]: r["title"] for r in rows}
    for r in rows:
        pid, slug = r["paper_id"], _slug(r["paper_id"])
        fields = conn.execute(
            f"SELECT {','.join(_FIELD_COLS)}, prompt_version, extractor_model "
            "FROM paper_fields WHERE paper_id=?", (pid,)).fetchone()
        abstract = conn.execute("SELECT abstract FROM paper_abstract WHERE paper_id=?", (pid,)).fetchone()
        cites = conn.execute(
            "SELECT dst, weight, intent FROM edge WHERE src=? AND kind='cites' ORDER BY dst", (pid,)).fetchall()
        similar = conn.execute(
            "SELECT dst, weight FROM edge WHERE src=? AND kind='similar' ORDER BY weight DESC", (pid,)).fetchall()
        ext_ids = conn.execute(
            "SELECT scheme, value FROM paper_ext_id WHERE paper_id=? ORDER BY scheme", (pid,)).fetchall()
        with open(os.path.join(papers_dir, slug + ".md"), "w", encoding="utf-8") as f:
            f.write(_render_concept(r, fields, abstract, cites, similar, ext_ids, titles, okf_version))
        # vector sidecar keeps the bundle self-contained (SPECTER2 is not recomputable)
        vrow = conn.execute("SELECT embedding FROM vec_paper WHERE paper_rowid=?", (r["rowid"],)).fetchone()
        vpath = os.path.join(papers_dir, slug + ".vec.json")
        if vrow is not None:
            vec = list(struct.unpack(f"{len(vrow['embedding']) // 4}f", vrow["embedding"]))
            with open(vpath, "w", encoding="utf-8") as f:
                json.dump(vec, f)
        elif os.path.exists(vpath):
            os.remove(vpath)  # drop a stale sidecar if the vector went away
        conn.execute("UPDATE paper SET okf_path=?, okf_version=? WHERE paper_id=?",
                     (f"papers/{slug}.md", okf_version, pid))
    _write_index(papers_dir, rows, okf_version)
    conn.commit()
    return len(rows)


# ============================================================================= read
def read_concept(path):
    """Round-trip reader: returns (frontmatter_dict, body_markdown)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing OKF frontmatter")
    _, fm_block, body = text.split("---\n", 2)
    meta = {}
    for line in fm_block.strip().splitlines():
        key, _, val = line.partition(": ")
        meta[key] = json.loads(val)
    return meta, body.lstrip("\n")


def _section(body, header):
    """Lines under a '## header' block, up to the next '## ' (or end)."""
    out, capture = [], False
    for line in body.splitlines():
        if line.startswith("## "):
            capture = line[3:].strip() == header
            continue
        if capture:
            out.append(line)
    return out


_FIELD_RE = re.compile(r"^- \*\*(\w+):\*\* (.*)$")
_LINK_RE = re.compile(r"^- \[.*?\]\((?P<slug>.+?)\.md\)(?:\s+—\s+(?P<w>[-\d.]+))?\s*$")


def _parse_fields(body):
    return {m.group(1): m.group(2).strip()
            for line in _section(body, "Structured fields") if (m := _FIELD_RE.match(line))}


def _parse_abstract(body):
    return "\n".join(_section(body, "Abstract")).strip() or None


def _parse_links(body, header):
    out = []
    for line in _section(body, header):
        m = _LINK_RE.match(line)
        if m:
            out.append((m.group("slug"), float(m.group("w")) if m.group("w") else None))
    return out


# ========================================================================== rebuild
def _restore_paper(conn, meta, body):
    pid = meta["id"]
    fields, abstract = _parse_fields(body), _parse_abstract(body)
    conn.execute(
        "INSERT INTO paper(paper_id,title,year,venue,tldr,contribution,tags,n_citations,"
        "cluster_id,content_hash,fields_status,okf_path,okf_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (pid, meta["title"], meta.get("year"), meta.get("venue"), meta.get("tldr"),
         meta.get("contribution"), ",".join(meta.get("tags") or []), meta.get("n_citations") or 0,
         meta.get("cluster"), meta["content_hash"], "extracted" if fields else "none",
         f"papers/{_slug(pid)}.md", meta.get("okf_version")))
    for e in meta.get("ext_ids") or []:
        conn.execute("INSERT OR IGNORE INTO paper_ext_id(paper_id,scheme,value) VALUES (?,?,?)",
                     (pid, e["scheme"], e["value"]))
    if abstract:
        conn.execute("INSERT INTO paper_abstract(paper_id,abstract) VALUES (?,?)", (pid, abstract))
    if fields:
        pv = meta.get("fields_prompt_version") or "okf"
        conn.execute(
            "INSERT INTO paper_fields(paper_id,problem,method,dataset,metric,result,limitation,"
            "extractor_model,prompt_version,source_hash) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (pid, *(fields.get(c) for c in _FIELD_COLS), meta.get("fields_extractor"),
             pv, meta["content_hash"]))
        conn.execute(  # rebuild the amortization cache so re-ingest stays a cache hit
            "INSERT OR REPLACE INTO ingest_cache(content_hash,task,prompt_version,model,output) "
            "VALUES (?,?,?,?,?)",
            (meta["content_hash"], "fields", pv, meta.get("fields_extractor"), json.dumps(fields)))


def _restore_vector(conn, papers_dir, meta, embedder):
    vpath = os.path.join(papers_dir, _slug(meta["id"]) + ".vec.json")
    if os.path.exists(vpath):
        with open(vpath, encoding="utf-8") as f:
            queries._store_vec(conn, meta["id"], json.load(f))
    elif embedder is not None:  # bundle authored without vectors -> derive from text
        queries._store_vec(conn, meta["id"], embedder.encode(f"{meta['title']}\n{meta.get('tldr') or ''}"))


def _restore_edges(conn, src, body, slug_to_id):
    n = 0
    for kind, header in (("cites", "Cites"), ("similar", "Similar")):
        for slug, w in _parse_links(body, header):
            dst = slug_to_id.get(slug)
            if dst:
                conn.execute("INSERT OR IGNORE INTO edge(src,dst,kind,weight,intent) VALUES (?,?,?,?,?)",
                             (src, dst, kind, w if w is not None else 1.0, None))
                n += 1
    return n


def rebuild(okf_dir, db_path, embedder=None):
    """Regenerate the derived index from the OKF bundle (PLAN.md §7.3)."""
    build_index.build(db_path).close()  # fresh schema
    conn = db.connect(db_path)
    papers_dir = os.path.join(okf_dir, "papers")
    concepts = [(fn, *read_concept(os.path.join(papers_dir, fn)))
                for fn in sorted(os.listdir(papers_dir))
                if fn.endswith(".md") and fn != "index.md"]
    slug_to_id = {_slug(meta["id"]): meta["id"] for _, meta, _ in concepts}

    for _, meta, body in concepts:               # pass 1: nodes (must exist before edges)
        _restore_paper(conn, meta, body)
        _restore_vector(conn, papers_dir, meta, embedder)
    conn.commit()

    n_edges = 0                                   # pass 2: edges (endpoints now exist -> FK-safe)
    for _, meta, body in concepts:
        n_edges += _restore_edges(conn, meta["id"], body, slug_to_id)
    conn.commit()

    n_vec = conn.execute("SELECT count(*) FROM vec_paper").fetchone()[0]
    conn.close()
    return {"papers": len(concepts), "edges": n_edges, "vectors": n_vec}


# ============================================================================== cli
def main():
    ap = argparse.ArgumentParser(description="OKF bundle <-> derived index (PLAN.md §7.2/§7.3)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("export", help="index -> OKF bundle")
    pe.add_argument("--db", default=db.DEFAULT_DB)
    pe.add_argument("--out", default=os.environ.get("CARDINAL_OKF", "okf"))
    pr = sub.add_parser("rebuild", help="OKF bundle -> index")
    pr.add_argument("--in", dest="src", default=os.environ.get("CARDINAL_OKF", "okf"))
    pr.add_argument("--db", default=db.DEFAULT_DB)
    a = ap.parse_args()
    if a.cmd == "export":
        conn = db.connect(a.db)
        n = export(conn, a.out)
        conn.close()
        print(f"wrote {n} OKF concepts + index to {a.out}/papers/")
    else:
        stats = rebuild(a.src, a.db, embedder=get_embedder())
        print(f"rebuilt {a.db} from {a.src}: {stats}")


if __name__ == "__main__":
    main()
