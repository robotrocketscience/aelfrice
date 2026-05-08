"""Leaf module: types and constants for the document/semantic linker.

Extracted from `aelfrice.doc_linker` so `aelfrice.store` can read the
`DocAnchor` shape without closing a `doc_linker ↔ store` import cycle
(`doc_linker.link_belief_to_document` calls `store.link_belief_to_document`,
and `store` returns `DocAnchor` rows back). Mirrors the leaf-module
pattern used in #500 for `np_pattern` / `classification_core` / `db_paths`.

This module imports nothing from the rest of `aelfrice` — keep it that way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

ANCHOR_INGEST: Final[str] = "ingest"
ANCHOR_MANUAL: Final[str] = "manual"
ANCHOR_DERIVED: Final[str] = "derived"

ANCHOR_TYPES: Final[frozenset[str]] = frozenset(
    {ANCHOR_INGEST, ANCHOR_MANUAL, ANCHOR_DERIVED},
)


@dataclass(frozen=True)
class DocAnchor:
    """One stored anchor row from ``belief_documents``.

    ``doc_uri`` is opaque to the linker. Recommended encodings at v2.0.0:

    - ``file:///abs/path/to/source.py#Lstart-Lend`` — local-source ingest.
    - ``https://host/path#fragment`` — external doc / web ingest.

    Other forms are accepted; the linker only rejects empty input.

    ``position_hint`` is a free-form string (e.g. ``"L42-L60"``,
    ``"#section"``); ``None`` when the URI itself encodes the position or
    no hint is available.

    ``created_at`` is a unix timestamp (seconds since epoch).
    """

    belief_id: str
    doc_uri: str
    anchor_type: str
    position_hint: str | None
    created_at: float
