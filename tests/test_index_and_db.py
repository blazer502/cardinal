import struct


def test_schema_objects(conn):
    names = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}
    for expected in ("paper", "edge", "cluster", "ingest_cache", "paper_fields",
                     "paper_abstract", "paper_chunk", "paper_ext_id",
                     "fts_paper", "vec_paper", "vec_chunk", "v_card",
                     "paper_ai", "paper_ad", "paper_au"):
        assert expected in names, f"missing {expected}"


def test_sqlite_vec_loaded(conn):
    # vec0 KNN only works if db.connect loaded the sqlite-vec extension
    conn.execute("INSERT INTO paper(paper_id,title,content_hash) VALUES ('x','t','h')")
    rid = conn.execute("SELECT rowid FROM paper WHERE paper_id='x'").fetchone()[0]
    blob = struct.pack("768f", *([0.1] * 768))
    conn.execute("INSERT INTO vec_paper(paper_rowid, embedding) VALUES (?,?)", (rid, blob))
    row = conn.execute("SELECT paper_rowid FROM vec_paper "
                       "WHERE embedding MATCH ? AND k=1 ORDER BY distance", (blob,)).fetchone()
    assert row[0] == rid


def test_fts_insert_trigger(conn):
    conn.execute("INSERT INTO paper(paper_id,title,tldr,tags,content_hash) "
                 "VALUES ('p','Backdoor Survey','x','y','h')")
    conn.commit()
    hits = conn.execute("SELECT p.paper_id FROM fts_paper f JOIN paper p ON p.rowid=f.rowid "
                        "WHERE fts_paper MATCH 'backdoor'").fetchall()
    assert ("p",) in [tuple(r) for r in hits]
