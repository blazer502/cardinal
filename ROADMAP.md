# Cardinal — Roadmap & Status

Design lives in [`PLAN.md`](PLAN.md). This file tracks **what's built** and **what's
next**, so work can be picked up later. Section numbers reference `PLAN.md §7` (MVP
build order).

## Status (on `main`)

Built and verified end-to-end (tests: `pytest`; retrieval quality: `python evaluate.py`):

| PLAN §7 step | State | Where |
|---|---|---|
| 1. Seed S2/OpenAlex → nodes/edges/TLDR/SPECTER2 | ✅ | `seed.py` |
| 2. OKF canonical bundle (+ lossless index↔bundle loop) | ✅ | `okf.py` (`export`/`rebuild`) |
| 3. Derived index (FTS5 + sqlite-vec + triggers) | ✅ | `schema.sql`, `build_index.py` |
| 4. ingest cache / amortization | ✅ | `queries.ingest`, `ingest_cache` |
| 4b. **L1 field extraction (LLM)** | ⏳ stubbed | `queries.ingest(extractor=...)` — not wired to a real model |
| 5. MCP server (8 tools + workflow instructions) | ✅ | `server.py` |
| 6. Graph view (force-directed frontend) | ⏳ | `subgraph` tool exists; no UI yet |

Also done beyond the base plan: real local **SPECTER2** semantic search (`embed.py`,
optional extra), a **47-test** pytest suite, and a **retrieval eval harness**
(`evaluate.py`) — with SPECTER2, semantic MRR ~0.94 and hybrid best recall@5.

## Next up (prioritized)

### 1. LLM field extraction  (PLAN §7.4 / §3.6 step 4) — **recommended next**
Populate L1 `problem/method/dataset/metric/result/limitation` (+ `contribution`, `tags`)
so `expand(fields)` and the card `f` flag become real.
- Add an `extractor(title, abstract) -> dict` backed by a small model (Claude Haiku via
  the Anthropic API). Give it a `.model` attr and a `prompt_version`.
- Wire it into `seed.py` / `queries.ingest` (the plumbing + `ingest_cache` amortization
  already exist — a cache hit means 0 LLM calls).
- Ship it as an **optional extra** (like the embedder) so the core stays dependency-free;
  read the API key from env. Guard/skip gracefully when unconfigured.
- Touchpoints: `queries.ingest`, `seed.py`, new `extract.py`, `requirements-embed.txt`
  or a new `requirements-llm.txt`, README.

### 2. Batch clustering  (PLAN §3.6 step 7)
Fill the `cluster` table so `get_cluster` returns real data.
- Cluster SPECTER2 vectors (k-means / HDBSCAN over `vec_paper`), assign `paper.cluster_id`.
- Generate one `label`+`summary` per cluster with a small model (cache the LLM output).
- Touchpoints: new `cluster.py` (batch), `queries.get_cluster` (already reads it).

### 3. Graph-view frontend  (PLAN §7.6)
A connected-style force-directed viz consuming `subgraph` output (node size = citations,
color = year). d3-force / cytoscape / sigma, or reuse an OKF reference visualizer.

### 4. Embedder refinements
- Use SPECTER2 `adhoc_query` adapter for the query side (asymmetric) vs `proximity` for
  docs; currently both use `proximity` for space-consistency (`embed.py`).
- Optional lightweight ONNX embedder (configurable dim) for a torch-free semantic path.

### 5. Seed / OKF polish
- Persist S2 **citation intent** into `cites` edges (currently dropped) and round-trip it
  through OKF (`## Cites` format).
- S2 batch pagination for larger seeds.

## Resume checklist
```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python build_index.py
.venv/bin/pytest                                   # 47 tests, no network/torch
.venv/bin/python evaluate.py                       # retrieval baseline
# optional semantic: pip install -r requirements-embed.txt ; CARDINAL_EMBEDDER=specter2
```
