import os

import db
import okf


def test_export_read_roundtrip(seeded, tmp_path):
    out = str(tmp_path / "okf")
    assert okf.export(seeded, out) == 3
    files = os.listdir(os.path.join(out, "papers"))
    assert "index.md" in files

    # S2:14 has fields, an abstract, and outgoing 'similar' edges (it is cited by others,
    # so it has no outgoing 'cites' section — that lives on the citing papers).
    meta, body = okf.read_concept(os.path.join(out, "papers", "S2_14.md"))
    assert meta["id"] == "S2:14"
    assert isinstance(meta["tags"], list) and isinstance(meta["ext_ids"], list)
    for section in ("## Structured fields", "## Abstract", "## Similar"):
        assert section in body
    _, body22 = okf.read_concept(os.path.join(out, "papers", "S2_22.md"))
    assert "## Cites" in body22 and "S2_14.md" in body22  # S2:22 cites S2:14


def _snapshot(c):
    q = lambda s: c.execute(s).fetchone()[0]
    return {t: q(f"SELECT count(*) FROM {t}")
            for t in ("paper", "paper_ext_id", "paper_abstract", "paper_fields")} | {
        "cites": q("SELECT count(*) FROM edge WHERE kind='cites'"),
        "similar": q("SELECT count(*) FROM edge WHERE kind='similar'"),
        "vec": q("SELECT count(*) FROM vec_paper"),
        "cache": q("SELECT count(*) FROM ingest_cache WHERE task='fields'"),
    }


def _vec14(c):
    return c.execute("SELECT embedding FROM vec_paper v JOIN paper p ON p.rowid=v.paper_rowid "
                     "WHERE p.paper_id='S2:14'").fetchone()[0]


def test_rebuild_is_faithful(seeded, tmp_path):
    out = str(tmp_path / "okf")
    okf.export(seeded, out)
    rebuilt_path = str(tmp_path / "rebuilt.db")
    stats = okf.rebuild(out, rebuilt_path)
    assert stats["papers"] == 3

    rebuilt = db.connect(rebuilt_path)
    try:
        assert _snapshot(rebuilt) == _snapshot(seeded)      # every table count matches
        assert _vec14(rebuilt) == _vec14(seeded)            # vector bytes identical
    finally:
        rebuilt.close()
