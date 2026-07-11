#!/usr/bin/env python3
"""Build the derived SQLite search index from schema.sql.

Deterministic, re-runnable (PLAN.md §7.3): the derived index can be rebuilt
from scratch at any time from the OKF bundle. This step applies schema.sql
into a fresh DB with the sqlite-vec extension loaded (vec0 virtual tables).

Usage: python build_index.py [db_path]   (default: index.db)
"""
import os
import sqlite3
import sys

import sqlite_vec

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")


def build(db_path: str) -> sqlite3.Connection:
    # fresh build: drop any prior DB + WAL sidecars so the schema applies clean
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db_path + suffix)
        except FileNotFoundError:
            pass

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    return conn


def inventory(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT type, name FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
    ).fetchall()
    by_type: dict[str, list[str]] = {}
    for typ, name in rows:
        by_type.setdefault(typ, []).append(name)
    for typ in ("table", "index", "trigger", "view"):
        names = by_type.get(typ, [])
        print(f"{typ+'s':9} ({len(names):2}): {', '.join(names)}")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "index.db"
    conn = build(db)
    print(f"built {db}  (sqlite {sqlite3.sqlite_version}, sqlite-vec loaded)\n")
    inventory(conn)
    conn.close()
