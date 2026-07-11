-- =============================================================================
-- Agent-friendly related-work DB — SQLite schema
-- Design principles
--   1) progressive disclosure: L0 cards (flat) settle most decisions. L1~L3 are explicit drill-down.
--   2) LLM cost is amortized once at ingest time. Query time is deterministic (0 LLM tokens).
--   3) Edges/clusters/summaries are all precomputed → the agent traverses IDs, not text.
--   4) The hot path (L0) uses flat columns with no nested JSON → lossless projection to TOON/TSV (fewer input tokens).
--   5) The source of truth is the OKF bundle (markdown+YAML, 1 concept = 1 file, cross-links = edges).
--      This SQLite is the derived search index (vector/BM25/graph acceleration) built from that bundle. OKF has no query engine.
--
-- Required extensions: sqlite-vec (vectors), FTS5 (built-in, BM25). Load example:
--   .load ./vec0            -- sqlite-vec
--   (FTS5 ships by default in most SQLite builds)
-- Security: never string-interpolate agent-supplied strings into SQL (everything is a bind parameter).
-- =============================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

-- -----------------------------------------------------------------------------
-- L0: paper card (hot path). No nested/heavy fields allowed here.
--   One row is exactly one "card" the agent scans (≈30~60 tokens target).
-- -----------------------------------------------------------------------------
CREATE TABLE paper (
    paper_id      TEXT PRIMARY KEY,      -- internal stable ID (e.g. S2 corpusId, or content_hash prefix)
    title         TEXT NOT NULL,
    year          INTEGER,
    venue         TEXT,
    tldr          TEXT,                  -- L0 one-sentence summary (reuse S2 TLDR when possible → no generation needed)
    contribution  TEXT,                  -- L0 core contribution, 1 line (LLM-extracted or S2)
    tags          TEXT,                  -- comma-separated flat string. For card rendering/projection
    n_citations   INTEGER DEFAULT 0,     -- visualization node size
    cluster_id    INTEGER,               -- FK -> cluster.cluster_id
    content_hash  TEXT NOT NULL,         -- sha256(normalize(title+"\n"+abstract)) — basis for the cache key
    fields_status TEXT NOT NULL DEFAULT 'none', -- 'none' | 'extracted' : whether L1 exists (= whether the agent has read it)
    okf_path      TEXT,                          -- OKF file path of this paper's concept (source of truth). SQLite is this bundle's derived index.
    okf_version   TEXT                           -- target OKF spec version (e.g. '0.1')
);
CREATE INDEX idx_paper_year     ON paper(year);
CREATE INDEX idx_paper_cluster  ON paper(cluster_id);
CREATE INDEX idx_paper_hash     ON paper(content_hash);
CREATE INDEX idx_paper_fstatus  ON paper(fields_status);

-- external ID mapping (for dedup/lookup, not hot path)
CREATE TABLE paper_ext_id (
    paper_id  TEXT NOT NULL REFERENCES paper(paper_id) ON DELETE CASCADE,
    scheme    TEXT NOT NULL,   -- 'doi' | 'arxiv' | 's2' | 'openalex' | 'mag'
    value     TEXT NOT NULL,
    PRIMARY KEY (scheme, value)
);
CREATE INDEX idx_ext_paper ON paper_ext_id(paper_id);

-- -----------------------------------------------------------------------------
-- L1: structured fields (extracted once by the LLM at ingest, cached). Replaces re-reading the abstract.
-- -----------------------------------------------------------------------------
CREATE TABLE paper_fields (
    paper_id        TEXT PRIMARY KEY REFERENCES paper(paper_id) ON DELETE CASCADE,
    problem         TEXT,
    method          TEXT,
    dataset         TEXT,
    metric          TEXT,
    result          TEXT,
    limitation      TEXT,
    extractor_model TEXT,          -- for tracking (e.g. 'claude-haiku-4-5')
    prompt_version  TEXT NOT NULL, -- cache-invalidation axis
    source_hash     TEXT NOT NULL, -- paper.content_hash at extraction time (change detection)
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- -----------------------------------------------------------------------------
-- L2: abstract (stored separately → an L0 scan never loads the abstract)
-- -----------------------------------------------------------------------------
CREATE TABLE paper_abstract (
    paper_id  TEXT PRIMARY KEY REFERENCES paper(paper_id) ON DELETE CASCADE,
    abstract  TEXT
);

-- -----------------------------------------------------------------------------
-- L3: full-text chunks (accessed only in explicit drill-down)
-- -----------------------------------------------------------------------------
CREATE TABLE paper_chunk (
    chunk_id  INTEGER PRIMARY KEY,
    paper_id  TEXT NOT NULL REFERENCES paper(paper_id) ON DELETE CASCADE,
    section   TEXT,
    ord       INTEGER,
    text      TEXT NOT NULL
);
CREATE INDEX idx_chunk_paper ON paper_chunk(paper_id, ord);

-- -----------------------------------------------------------------------------
-- edges: single normalized table. Designed so neighbors() is a pure index range scan.
--   kind: 'cites' | 'similar' | 'shared_method' | 'shared_dataset'
--   weight: similarity score / citation influence
--   intent: for 'cites', the S2 citation intent ('background'|'method'|'result')
-- -----------------------------------------------------------------------------
CREATE TABLE edge (
    src     TEXT NOT NULL REFERENCES paper(paper_id) ON DELETE CASCADE,
    dst     TEXT NOT NULL REFERENCES paper(paper_id) ON DELETE CASCADE,
    kind    TEXT NOT NULL,
    weight  REAL DEFAULT 0.0,
    intent  TEXT,
    PRIMARY KEY (src, dst, kind)
);
-- (src, kind, weight DESC): "top-k kind-neighbors of id" finishes as a 0-token index scan
CREATE INDEX idx_edge_src ON edge(src, kind, weight DESC);
CREATE INDEX idx_edge_dst ON edge(dst, kind);

-- -----------------------------------------------------------------------------
-- clusters: precomputed labels/summaries → "what is this cluster?" is a lookup, not an LLM call.
-- -----------------------------------------------------------------------------
CREATE TABLE cluster (
    cluster_id  INTEGER PRIMARY KEY,
    label       TEXT,
    summary     TEXT,    -- generated once, then reused
    size        INTEGER DEFAULT 0
);

-- -----------------------------------------------------------------------------
-- ingest cache: the amortization point for every LLM output.
--   Before ingest, look up by (content_hash, task, prompt_version) → if present, skip the LLM call.
--   Bumping prompt_version re-extracts only that task, globally.
-- -----------------------------------------------------------------------------
CREATE TABLE ingest_cache (
    content_hash    TEXT NOT NULL,
    task            TEXT NOT NULL,   -- 'fields' | 'contribution' | 'tags' | 'cluster_summary'
    prompt_version  TEXT NOT NULL,
    model           TEXT,
    output          TEXT,            -- raw LLM output (JSON/text)
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (content_hash, task, prompt_version)
);

-- =============================================================================
-- Search indexes
-- =============================================================================

-- BM25 keyword search (external-content FTS: indexes only paper's card fields)
CREATE VIRTUAL TABLE fts_paper USING fts5(
    title, tldr, tags,
    content='paper', content_rowid='rowid',
    tokenize='porter unicode61'
);
-- paper <-> fts_paper sync triggers
CREATE TRIGGER paper_ai AFTER INSERT ON paper BEGIN
    INSERT INTO fts_paper(rowid, title, tldr, tags)
    VALUES (new.rowid, new.title, new.tldr, new.tags);
END;
CREATE TRIGGER paper_ad AFTER DELETE ON paper BEGIN
    INSERT INTO fts_paper(fts_paper, rowid, title, tldr, tags)
    VALUES ('delete', old.rowid, old.title, old.tldr, old.tags);
END;
CREATE TRIGGER paper_au AFTER UPDATE ON paper BEGIN
    INSERT INTO fts_paper(fts_paper, rowid, title, tldr, tags)
    VALUES ('delete', old.rowid, old.title, old.tldr, old.tags);
    INSERT INTO fts_paper(rowid, title, tldr, tags)
    VALUES (new.rowid, new.title, new.tldr, new.tags);
END;

-- vector search (sqlite-vec). SPECTER2 = 768 dims. Joined via paper.rowid. Derived index built from OKF/embeddings.
CREATE VIRTUAL TABLE vec_paper USING vec0(
    paper_rowid  INTEGER PRIMARY KEY,   -- = paper.rowid
    embedding    FLOAT[768]
);
-- chunk embeddings (for L3 RAG). Adjust to the model's dimensionality.
CREATE VIRTUAL TABLE vec_chunk USING vec0(
    chunk_id   INTEGER PRIMARY KEY,     -- = paper_chunk.chunk_id
    embedding  FLOAT[768]
);

-- =============================================================================
-- Token-oriented projection view: flat cards for agent returns.
--   The MCP serializer declares the TOON/TSV header "once" in this column order, then emits only value rows.
--   (This is where you save 30~50% input tokens vs JSON array-of-objects)
-- =============================================================================
CREATE VIEW v_card AS
SELECT paper_id, year, n_citations, cluster_id, fields_status, title, tldr, tags
FROM   paper;

-- =============================================================================
-- Frequently used deterministic queries (all 0 LLM tokens) — parameters MUST be bound
-- =============================================================================
-- [neighbors] top-k kind-neighbors of id (index range scan)
--   SELECT dst AS paper_id, weight, intent
--   FROM   edge WHERE src = :id AND kind = :kind
--   ORDER  BY weight DESC LIMIT :k;
--
-- [semantic search] SPECTER2 nearest-neighbor → join to cards
--   SELECT p.paper_id, p.title, p.tldr, v.distance
--   FROM   vec_paper v JOIN paper p ON p.rowid = v.paper_rowid
--   WHERE  v.embedding MATCH :qvec AND k = :k
--   ORDER  BY v.distance;
--
-- [keyword search] BM25
--   SELECT p.paper_id, p.title, bm25(fts_paper) AS score
--   FROM   fts_paper f JOIN paper p ON p.rowid = f.rowid
--   WHERE  fts_paper MATCH :q
--   ORDER  BY score LIMIT :k;
--
-- [hybrid: BM25 + vector fusion (Reciprocal Rank Fusion)]
--   1) get FTS BM25 top-N and vec_paper nearest top-N separately (both internal to sqlite, 0 LLM tokens)
--   2) merge the two rankings with RRF: score = Σ 1/(k0 + rank_i)  (k0≈60)
--   Meta filters (cluster/year/tags) are pre-applied as WHERE on each subquery.
--
-- [ingest cache lookup] always check before an LLM call
--   SELECT output FROM ingest_cache
--   WHERE content_hash = :h AND task = :task AND prompt_version = :pv;
-- =============================================================================
