"""Document / semantic linker (#435).

Connects a belief to the document anchor it describes — a file path with a
line range, a URL with a section fragment, etc. — so retrieval can return the
canonical reference alongside the bare belief snippet. Spec:
``docs/design/feature-doc-linker.md``.

The linker stores opaque URI strings in a sibling table ``belief_documents``
keyed on ``(belief_id, doc_uri)``. Idempotent re-ingest: ``INSERT OR IGNORE``
collapses repeats to the first row. Ingest-time writes happen inside the
derivation worker when ``DerivationInput.source_path`` is set; manual writes
happen via ``aelf lock --doc=URI``. Retrieval consumers opt in via the
``with_doc_anchors=True`` kwarg on ``retrieve()`` / ``retrieve_v2()``.

Out of scope at v2.0.0 (per spec):

- ``anchor_type='derived'`` writers (retrieval-time inference). Enum value is
  reserved; no writer in this module.
- URI validation beyond non-empty.
- ``source_path`` normalisation. The ingest writer stores whatever path the
  caller passes; consumers normalise on the way out.
"""
from __future__ import annotations

from pathlib import Path

# Types + constants live in the leaf module so `aelfrice.store` can read
# the `DocAnchor` shape without closing a `doc_linker ↔ store` cycle
# (see #501). Re-exported here for backward compatibility.
from aelfrice.doc_linker_types import (
    ANCHOR_DERIVED,
    ANCHOR_INGEST,
    ANCHOR_MANUAL,
    ANCHOR_TYPES,
    DocAnchor,
)

__all__ = (
    "ANCHOR_DERIVED",
    "ANCHOR_INGEST",
    "ANCHOR_MANUAL",
    "ANCHOR_TYPES",
    "DocAnchor",
    "file_uri_from_path",
    "get_doc_anchors",
    "link_belief_to_document",
)


def file_uri_from_path(
    source_path: str,
    *,
    project_root: Path | None = None,
    position_hint: str | None = None,
) -> str:
    """Build a ``file://`` URI from a local source path.

    When ``project_root`` is set and ``source_path`` is inside it, the URI
    encodes the path relative to the project root (avoids leaking the
    operator's filesystem layout into the store). Otherwise the absolute
    path is used.

    ``position_hint`` is appended as a fragment when provided; the linker
    stores the same value separately so consumers can read it without
    parsing the URI.
    """
    p = Path(source_path)
    rel: str
    if project_root is not None:
        try:
            rel = str(p.resolve().relative_to(project_root.resolve()))
        except ValueError:
            rel = str(p)
    else:
        rel = str(p)
    uri = f"file:{rel}" if not rel.startswith("/") else f"file://{rel}"
    if position_hint:
        uri = f"{uri}#{position_hint}"
    return uri


def link_belief_to_document(
    store: "MemoryStore",
    belief_id: str,
    doc_uri: str,
    *,
    anchor_type: str = ANCHOR_INGEST,
    position_hint: str | None = None,
) -> DocAnchor:
    """Persist a ``belief_id ↔ doc_uri`` anchor and return the row.

    Idempotent on ``(belief_id, doc_uri)`` via ``INSERT OR IGNORE`` in the
    storage layer: re-calling with the same pair is a no-op write but always
    returns a ``DocAnchor`` reflecting the canonical (first-write) row.

    Raises ``ValueError`` on empty ``doc_uri`` or unknown ``anchor_type``.
    """
    if not doc_uri:
        raise ValueError("doc_uri must be non-empty")
    if anchor_type not in ANCHOR_TYPES:
        raise ValueError(
            f"Unknown anchor_type {anchor_type!r}. "
            f"Must be one of {sorted(ANCHOR_TYPES)}"
        )
    return store.link_belief_to_document(
        belief_id=belief_id,
        doc_uri=doc_uri,
        anchor_type=anchor_type,
        position_hint=position_hint,
    )


def get_doc_anchors(
    store: "MemoryStore",
    belief_id: str,
) -> list[DocAnchor]:
    """Return every anchor for a belief, ordered by ``created_at`` ASC."""
    return store.get_doc_anchors(belief_id)
