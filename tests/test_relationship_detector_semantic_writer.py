"""Unit tests for the #988 CONTRADICTS edge writer + ingest wiring.

Covers ``write_semantic_edges`` (high-confidence CONTRADICTS edges, the
complement of the sub-confidence ``write_potentially_stale_edges`` set),
the default-off ``auto_detect`` flag resolver, the determinism guarantee,
the per-belief write-gate, and the byte-identical off-path through
``ingest_turn``.

All tests use a real ``MemoryStore(":memory:")`` — no mocks.
"""
from __future__ import annotations

import hashlib

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    LOCK_NONE,
    Belief,
)
from aelfrice.relationship_detector import (
    DEFAULT_MAX_EDGES_PER_BELIEF,
    ENV_AUTO_RELATIONSHIPS,
    is_auto_relationship_detection_enabled,
    write_semantic_edges,
)
from aelfrice.store import MemoryStore

# High-confidence contradiction: "always X" vs "never X" — universal
# affirmation vs negation over identical residual content → score 1.0.
_ALWAYS = "the deployment script always runs the database migration step"
_NEVER = "the deployment script never runs the database migration step"
# Lexically distant filler that relates to nothing else.
_UNRELATED = "the harbor seals bask on the warm rocks at noon each day"


def _make_belief(
    store: MemoryStore,
    *,
    belief_id: str,
    content: str,
    created_at: str = "2026-01-01T00:00:00Z",
) -> Belief:
    b = Belief(
        id=belief_id,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=None,
    )
    store.insert_belief(b)
    return b


def _contradicts_edges(store: MemoryStore) -> list[tuple[str, str]]:
    """Return all CONTRADICTS edges as sorted (src, dst) tuples."""
    rows = store._conn.execute(  # type: ignore[attr-defined]
        "SELECT src, dst FROM edges WHERE type = ? ORDER BY src, dst",
        (EDGE_CONTRADICTS,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


# ---------------------------------------------------------------------------
# Flag resolver precedence
# ---------------------------------------------------------------------------


def test_flag_defaults_off(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv(ENV_AUTO_RELATIONSHIPS, raising=False)
    # start at an empty dir so no repo .aelfrice.toml is found
    assert is_auto_relationship_detection_enabled(start=tmp_path) is False


def test_flag_env_wins_over_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_AUTO_RELATIONSHIPS, "off")
    assert is_auto_relationship_detection_enabled(explicit=True) is False
    monkeypatch.setenv(ENV_AUTO_RELATIONSHIPS, "on")
    assert is_auto_relationship_detection_enabled(explicit=False) is True


def test_flag_unrecognised_env_is_not_decisive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_AUTO_RELATIONSHIPS, "maybe")
    assert is_auto_relationship_detection_enabled(explicit=True) is True


def test_flag_toml_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv(ENV_AUTO_RELATIONSHIPS, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[relationship_detector]\nauto_detect = true\n"
    )
    assert is_auto_relationship_detection_enabled(start=tmp_path) is True


# ---------------------------------------------------------------------------
# write_semantic_edges — write / scope / idempotency
# ---------------------------------------------------------------------------


def test_writes_contradicts_edge_for_high_confidence_pair(
    store: MemoryStore,
) -> None:
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_NEVER)
    report = write_semantic_edges(store)
    assert report.n_contradicts_high == 1
    assert report.n_edges_written == 1
    # Canonical direction: src = min(id), dst = max(id).
    assert _contradicts_edges(store) == [("b1", "b2")]


def test_idempotent_second_run_writes_nothing(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_NEVER)
    write_semantic_edges(store)
    report = write_semantic_edges(store)
    assert report.n_edges_written == 0
    assert report.n_edges_skipped_existing == 1
    assert _contradicts_edges(store) == [("b1", "b2")]


def test_unrelated_pair_writes_no_edge(store: MemoryStore) -> None:
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_UNRELATED)
    report = write_semantic_edges(store)
    assert report.n_edges_written == 0
    assert _contradicts_edges(store) == []


def test_agreeing_modality_is_refines_not_contradicts(
    store: MemoryStore,
) -> None:
    # Two "never" statements over the same residual content agree in
    # modality → refines, which this CONTRADICTS-only writer ignores.
    _make_belief(store, belief_id="b1", content=_NEVER)
    _make_belief(
        store,
        belief_id="b2",
        content=_NEVER + " before launch",
    )
    report = write_semantic_edges(store)
    assert report.n_edges_written == 0
    assert _contradicts_edges(store) == []


# ---------------------------------------------------------------------------
# Determinism — byte-equal edge sets across two fresh stores
# ---------------------------------------------------------------------------


def test_determinism_byte_equal_edges() -> None:
    def build() -> list[tuple[str, str]]:
        s = MemoryStore(":memory:")
        _make_belief(s, belief_id="b1", content=_ALWAYS)
        _make_belief(s, belief_id="b2", content=_NEVER)
        _make_belief(s, belief_id="b3", content=_UNRELATED)
        write_semantic_edges(s)
        return _contradicts_edges(s)

    assert build() == build()


# ---------------------------------------------------------------------------
# Write-gate — per-belief edge cap (Exp-48 dilution guard)
# ---------------------------------------------------------------------------


def test_write_gate_caps_per_belief_edges(store: MemoryStore) -> None:
    # b1 contradicts both b2 and b3 (both negate b1's universal claim).
    # b2 and b3 agree with each other (both "never") → no edge between them.
    _make_belief(store, belief_id="b1", content=_ALWAYS)
    _make_belief(store, belief_id="b2", content=_NEVER)
    _make_belief(
        store, belief_id="b3", content=_NEVER + " before each launch"
    )
    report = write_semantic_edges(store, max_edges_per_belief=1)
    # b1 may only accrue one CONTRADICTS edge; the second pair is gated.
    assert report.n_edges_written == 1
    assert report.n_edges_skipped_gated == 1
    edges = _contradicts_edges(store)
    assert len(edges) == 1
    # The surviving edge is the first in deterministic (a_id, b_id) order.
    assert edges == [("b1", "b2")]


def test_default_cap_is_positive() -> None:
    assert DEFAULT_MAX_EDGES_PER_BELIEF >= 1


# ---------------------------------------------------------------------------
# Ingest wiring — off-path byte-identical, on-path writes
# ---------------------------------------------------------------------------


def test_ingest_off_path_writes_no_contradicts_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice.ingest import ingest_turn

    monkeypatch.delenv(ENV_AUTO_RELATIONSHIPS, raising=False)
    s = MemoryStore(":memory:")
    ingest_turn(s, _ALWAYS, source="t", session_id="sess")
    ingest_turn(s, _NEVER, source="t", session_id="sess")
    assert _contradicts_edges(s) == []


def test_ingest_on_path_writes_contradicts_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aelfrice.ingest import ingest_turn

    monkeypatch.setenv(ENV_AUTO_RELATIONSHIPS, "1")
    s = MemoryStore(":memory:")
    ingest_turn(s, _ALWAYS, source="t", session_id="sess")
    ingest_turn(s, _NEVER, source="t", session_id="sess")
    assert len(_contradicts_edges(s)) == 1
