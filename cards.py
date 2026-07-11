"""Compressed card projection (PLAN.md §4).

Serialize uniform paper rows as one column-header declaration + tab-separated
value rows, saving 30~50% input tokens vs a JSON array-of-objects. Columns
mirror the v_card view (schema.sql).

    cards[2]{paper_id,year,cit,cl,f,title,tldr,tags}:
    S2:14<TAB>2019<TAB>980<TAB>3<TAB>Y<TAB>BadNets<TAB>...<TAB>backdoor,dnn
"""
from typing import Mapping, Sequence

# (source key in row dict) -> (short header shown to the agent)
CARD_COLUMNS = [
    ("paper_id", "paper_id"),
    ("year", "year"),
    ("n_citations", "cit"),
    ("cluster_id", "cl"),
    ("fields_status", "f"),
    ("title", "title"),
    ("tldr", "tldr"),
    ("tags", "tags"),
]


def _cell(col: str, value) -> str:
    if value is None:
        return ""
    if col == "fields_status":  # Y/N = does the agent already have structured L1?
        return "Y" if value == "extracted" else "N"
    if isinstance(value, float):
        return f"{value:.4f}"
    # keep every cell single-line + tab-safe so rows stay losslessly parseable
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def render_cards(rows: Sequence[Mapping], extra: list[str] | None = None) -> str:
    rows = list(rows)
    cols = CARD_COLUMNS + [(e, e) for e in (extra or [])]
    header = f"cards[{len(rows)}]{{{','.join(short for _, short in cols)}}}:"
    lines = [header]
    for r in rows:
        lines.append("\t".join(_cell(src, r.get(src)) for src, _ in cols))
    return "\n".join(lines)
