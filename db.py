"""SQLite connection helper for the derived index.

Every connection must load sqlite-vec (the schema's vec0 tables depend on it),
exactly as build_index.py does. Rows come back as sqlite3.Row (dict-like).
"""
import os
import sqlite3

import sqlite_vec

DEFAULT_DB = os.environ.get(
    "CARDINAL_DB", os.path.join(os.path.dirname(__file__), "index.db")
)


def connect(db_path: str = DEFAULT_DB, *, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
