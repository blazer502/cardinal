# Cardinal — agent-friendly related-work DB

Token-efficient related-work index + MCP server. Deterministic search (0 LLM
tokens) over a SQLite index (sqlite-vec + FTS5), built from public metadata.
See [`PLAN.md`](PLAN.md) for the full design and [`schema.sql`](schema.sql) for the schema.

## Reproduce anywhere

All paths resolve relative to the repo and all config is env-driven — nothing is
tied to a specific machine. From a fresh checkout:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python build_index.py                 # apply schema.sql -> index.db
.venv/bin/python seed.py --query "backdoor attacks on neural networks" \
    --source openalex --limit 30                 # fetch nodes/edges/metadata
./run-server.sh                                  # start the MCP server (stdio)
```

## Seeding (`seed.py`, PLAN.md §7.1)

| Source | Gets | Needs |
|---|---|---|
| `--source openalex` (default) | title/abstract/venue/citations + references | nothing (set `OPENALEX_MAILTO` for the polite pool) |
| `--source s2` | + TLDR + **SPECTER2** vectors + citations | `S2_API_KEY` (rate-limited without one) |

```sh
# OpenAlex, no key:
.venv/bin/python seed.py --query "trojan neural network" --source openalex --limit 40
# Semantic Scholar with embeddings + SPECTER2 similar-edges:
S2_API_KEY=xxx .venv/bin/python seed.py --query "backdoor defense" --source s2 --similar-k 5
# By id:
.venv/bin/python seed.py --ids W2018,W2019 --source openalex
```

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `CARDINAL_DB` | `./index.db` | index location |
| `CARDINAL_EMBEDDER` | `none` | semantic-search embedder: `none` / `specter2` / `hash` |
| `S2_API_KEY` | — | Semantic Scholar API key |
| `OPENALEX_MAILTO` | — | OpenAlex polite-pool contact |

## Semantic search (optional extra)

Keyword search works out of the box. For real semantic/hybrid search, install the
SPECTER2 embedder (heavier — torch + transformers + adapters) and select it:

```sh
pip install -r requirements-embed.txt      # CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu first
export CARDINAL_EMBEDDER=specter2
```

SPECTER2 (768-dim) shares Semantic Scholar's `embedding.specter_v2` space, so locally
embedded queries align with S2-provided corpus vectors — and OpenAlex papers (no S2
vector) can be embedded into the same space. The model lazy-loads on the first
semantic query, so keyword-only sessions pay nothing. `hash` is a test-only
placeholder; with `none`, hybrid search falls back to keyword.

## OKF canonical bundle (`okf.py`, PLAN.md §7.2/§7.3)

The **source of truth** is an OKF bundle: one markdown+YAML concept file per paper,
cross-links = edges, plus `index.md` for progressive disclosure. The SQLite index
is a **derived, fully rebuildable** artifact — the loop runs both ways:

```sh
.venv/bin/python okf.py export  --db index.db --out okf/   # index      -> bundle
.venv/bin/python okf.py rebuild --in  okf/    --db index.db # bundle     -> index
```

Each `okf/papers/<id>.md` has JSON-encoded YAML frontmatter (valid YAML, no PyYAML
dependency — round-trips via `okf.read_concept`) plus `## Structured fields`,
`## Abstract`, `## Cites`, and `## Similar` link sections. SPECTER2 vectors ride
alongside as `<id>.vec.json` sidecars, so `rebuild` reconstructs papers, fields,
abstracts, external ids, edges, vectors, **and** the `ingest_cache` — meaning a
rebuilt index is functionally identical (search, semantic, graph) and LLM
amortization survives. `seed.py … --okf-dir okf` emits the bundle while seeding.

## Use it as an agent tool (Claude Code & Codex)

Cardinal is an **MCP server**, not a REST API — agents call its tools directly to
construct and analyze related work. It exposes eight tools:

| Tool | Cost | Purpose |
|---|---|---|
| `seed_topic` | network | **construct**: pull a topic from S2/OpenAlex into the graph |
| `search` | 0 tokens | hybrid BM25+vector scan → compact cards |
| `neighbors` / `subgraph` | 0 tokens | walk / map citations + similarity |
| `expand` | 0 tokens | drill into fields / abstract / chunks for a chosen few |
| `get_cluster` | 0 tokens | read a precomputed cluster |
| `ingest` | 1 LLM call | add/update one paper (amortized) |
| `export_okf` | 0 tokens | persist the canonical OKF bundle |

The server ships workflow `instructions` so the agent knows the cheap path
(cards first, drill down only when needed). `<repo>` is your checkout path — the
config is per-machine; the repo itself stays path-agnostic.

**Claude Code**
```sh
claude mcp add cardinal -- <repo>/run-server.sh
```
or a project-scoped `.mcp.json`:
```jsonc
{ "mcpServers": { "cardinal": {
    "command": "<repo>/run-server.sh",
    "env": { "CARDINAL_EMBEDDER": "none" }   // set once a real embedder is wired
} } }
```

**Codex** — in `~/.codex/config.toml`:
```toml
[mcp_servers.cardinal]
command = "<repo>/run-server.sh"
env = { CARDINAL_EMBEDDER = "none" }
```

Default returns are the compressed cards projection; pass `format:"json"|"ids"`
for full objects.

## Layout

```
schema.sql       derived-index schema (tables/FTS5/vec0/triggers/v_card view)
build_index.py   apply schema.sql -> index.db (rebuildable anytime)
seed.py          S2/OpenAlex -> nodes/edges/TLDR/SPECTER2
okf.py           emit/read the OKF canonical bundle (source of truth)
db.py            connection factory (loads sqlite-vec)
embed.py         query-embedder interface + factory
cards.py         §4 compressed card projection
queries.py       pure query layer for the tools (bind-parameterized)
server.py        FastMCP server (thin wrapper over queries.py)
run-server.sh    portable launcher
```
