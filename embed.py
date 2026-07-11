"""Text embedding providers for semantic/hybrid search.

Text is embedded by a LOCAL model in the SAME family as the corpus vectors
(SPECTER2, 768-dim) so agent token cost stays 0 (PLAN.md §3). Selected via
CARDINAL_EMBEDDER:

  none      (default) semantic search disabled; hybrid falls back to keyword.
  specter2  real SPECTER2 (allenai/specter2) — best for scientific papers.
            OPTIONAL extra: pip install -r requirements-embed.txt (torch +
            transformers + adapters). The core tool needs none of this — keyword
            search always works, and the heavy model lazy-loads only on the first
            semantic query.
  hash      deterministic PLACEHOLDER for tests/plumbing only — NOT meaningful.

Every embedder returns packed float32[DIM] bytes, ready for vec0 MATCH / _store_vec.
"""
import hashlib
import logging
import os
import struct
from typing import Optional, Protocol

DIM = 768


class Embedder(Protocol):
    def encode(self, text: str) -> bytes:  # packed float32[DIM]
        ...


def _pack(floats) -> bytes:
    if len(floats) != DIM:
        raise ValueError(f"embedding dim {len(floats)} != {DIM}")
    return struct.pack(f"{DIM}f", *floats)


class HashEmbedder:
    """Deterministic placeholder — NOT semantically meaningful. Tests/plumbing only."""

    model = "hash-placeholder"

    def encode(self, text: str) -> bytes:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return _pack([h[i % len(h)] / 255.0 for i in range(DIM)])


class Specter2Embedder:
    """Local SPECTER2 (allenai/specter2 proximity adapter), 768-dim.

    Shares Semantic Scholar's `embedding.specter_v2` vector space, so locally
    embedded queries/docs align with S2-provided corpus vectors — and OpenAlex
    papers (no S2 vector) can be embedded locally into the same space. The model
    lazy-loads on first encode(), so keyword-only sessions pay nothing. Queries use
    the proximity adapter for space-consistency; the adhoc_query adapter is a
    possible refinement for short queries.
    """

    model = "specter2"
    _BASE = "allenai/specter2_base"
    _ADAPTER = "allenai/specter2"  # proximity adapter

    def __init__(self):
        self._tok = self._model = self._torch = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch
            from adapters import AutoAdapterModel
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError("CARDINAL_EMBEDDER=specter2 needs the embedding extra: "
                               "pip install -r requirements-embed.txt") from e
        self._torch = torch
        alog = logging.getLogger("adapters")  # silence the transient setup-time adapter notice
        prev = alog.level
        alog.setLevel(logging.ERROR)
        try:
            self._tok = AutoTokenizer.from_pretrained(self._BASE)
            model = AutoAdapterModel.from_pretrained(self._BASE)
            model.load_adapter(self._ADAPTER, source="hf", load_as="proximity", set_active=True)
            model.set_active_adapters("proximity")  # ensure the adapter drives the forward pass
        finally:
            alog.setLevel(prev)
        model.eval()
        self._model = model

    def encode(self, text: str) -> bytes:
        self._ensure_loaded()
        inp = self._tok(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with self._torch.no_grad():
            out = self._model(**inp)
        cls = out.last_hidden_state[:, 0, :].squeeze(0)  # [CLS] pooling (SPECTER2 convention)
        return _pack(cls.tolist())


def get_embedder() -> Optional[Embedder]:
    kind = os.environ.get("CARDINAL_EMBEDDER", "none").lower()
    if kind == "none":
        return None
    if kind == "specter2":
        return Specter2Embedder()
    if kind == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown CARDINAL_EMBEDDER={kind!r} (expected none|specter2|hash)")
