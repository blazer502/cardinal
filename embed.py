"""Query embedding provider for semantic/hybrid search.

The query must be embedded by a LOCAL model in the SAME family as the corpus
vectors (SPECTER2, 768-dim) so that agent token cost stays 0 (PLAN.md §3).
No such model is bundled yet, so get_embedder() returns None by default and
semantic/hybrid search reports that it is unconfigured (hybrid falls back to
keyword; see queries.search).

Wire a real model by implementing encode() -> packed float32[DIM] bytes and
returning it from get_embedder() (selected via CARDINAL_EMBEDDER). HashEmbedder
is a deterministic PLACEHOLDER for plumbing/tests only — its vectors carry no
semantic meaning; never use it for real retrieval.
"""
import hashlib
import os
import struct
from typing import Optional, Protocol

DIM = 768


class Embedder(Protocol):
    def encode(self, text: str) -> bytes:  # packed float32[DIM], ready for vec0 MATCH
        ...


class HashEmbedder:
    """Deterministic placeholder — NOT semantically meaningful. Tests/plumbing only."""

    model = "hash-placeholder"

    def encode(self, text: str) -> bytes:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vals = [h[i % len(h)] / 255.0 for i in range(DIM)]
        return struct.pack(f"{DIM}f", *vals)


def get_embedder() -> Optional[Embedder]:
    kind = os.environ.get("CARDINAL_EMBEDDER", "none").lower()
    if kind == "none":
        return None
    if kind == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown CARDINAL_EMBEDDER={kind!r} (expected none|hash|<your model>)")
