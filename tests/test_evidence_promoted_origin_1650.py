"""#1650 part 2: the evidence_promoted origin.

An automatically promoted phantom must never claim that a person
validated it (docs/design/feature-supports-writer.md). It gets its own
origin, ranked with user_transcript and below user_validated, through a
promote that accepts only phantoms, and an undo that `aelf demote` runs.
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.cli import main as cli_main
from aelfrice.contradiction import precedence_class
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_EVIDENCE_PROMOTED,
    ORIGIN_RETRIEVAL_PRIORITY,
    ORIGIN_SPECULATIVE,
    ORIGIN_USER_TRANSCRIPT,
    ORIGIN_USER_VALIDATED,
    ORIGINS,
    Belief,
)
from aelfrice.promotion import (
    SOURCE_PROMOTE_EVIDENCE,
    SOURCE_REVERT_EVIDENCE,
    promote_on_evidence,
    revert_evidence_promotion,
)
from aelfrice.store import MemoryStore


def _belief(origin: str, *, bid: str = "p1", lock: str = LOCK_NONE) -> Belief:
    return Belief(
        id=bid, content="The cache expires after ten minutes.", content_hash=f"h_{bid}",
        alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=lock,
        locked_at="2026-10-01T00:00:00Z" if lock == LOCK_USER else None,
        created_at="2026-10-01T00:00:00Z", last_retrieved_at=None, origin=origin,
    )


@pytest.fixture
def store(tmp_path: Path):  # noqa: ANN201
    s = MemoryStore(str(tmp_path / "m.db"))
    yield s
    s.close()


def _audit(store: MemoryStore) -> list[str]:
    return [str(r[0]) for r in store._conn.execute(  # noqa: SLF001
        "SELECT source FROM feedback_history ORDER BY id").fetchall()]


def test_the_origin_is_declared_and_ranked_with_user_transcript() -> None:
    assert ORIGIN_EVIDENCE_PROMOTED in ORIGINS
    assert ORIGIN_RETRIEVAL_PRIORITY[ORIGIN_EVIDENCE_PROMOTED] == ORIGIN_RETRIEVAL_PRIORITY[ORIGIN_USER_TRANSCRIPT]
    assert ORIGIN_RETRIEVAL_PRIORITY[ORIGIN_EVIDENCE_PROMOTED] < ORIGIN_RETRIEVAL_PRIORITY[ORIGIN_USER_VALIDATED]


def test_contradiction_precedence_matches_user_transcript() -> None:
    assert precedence_class(_belief(ORIGIN_EVIDENCE_PROMOTED)) == precedence_class(_belief(ORIGIN_USER_TRANSCRIPT))


def test_a_phantom_is_promoted_with_an_audit_row(store: MemoryStore) -> None:
    store.insert_belief(_belief(ORIGIN_SPECULATIVE))
    result = promote_on_evidence(store, "p1", now="2026-10-06T00:00:00Z")
    b = store.get_belief("p1")
    assert b is not None and b.origin == ORIGIN_EVIDENCE_PROMOTED
    assert result.prior_origin == ORIGIN_SPECULATIVE and not result.already_validated
    assert _audit(store) == [SOURCE_PROMOTE_EVIDENCE]


def test_promoting_twice_is_a_no_op(store: MemoryStore) -> None:
    store.insert_belief(_belief(ORIGIN_SPECULATIVE))
    promote_on_evidence(store, "p1")
    again = promote_on_evidence(store, "p1")
    assert again.already_validated
    assert _audit(store) == [SOURCE_PROMOTE_EVIDENCE]


@pytest.mark.parametrize("origin", [ORIGIN_AGENT_INFERRED, ORIGIN_USER_TRANSCRIPT, ORIGIN_USER_VALIDATED])
def test_only_a_phantom_can_be_promoted(store: MemoryStore, origin: str) -> None:
    store.insert_belief(_belief(origin))
    with pytest.raises(ValueError, match="speculative"):
        promote_on_evidence(store, "p1")
    b = store.get_belief("p1")
    assert b is not None and b.origin == origin
    assert _audit(store) == []


def test_a_locked_or_missing_belief_is_refused(store: MemoryStore) -> None:
    store.insert_belief(_belief(ORIGIN_SPECULATIVE, lock=LOCK_USER))
    with pytest.raises(ValueError, match="locked"):
        promote_on_evidence(store, "p1")
    with pytest.raises(ValueError, match="not found"):
        promote_on_evidence(store, "missing")


def test_revert_returns_it_to_a_phantom(store: MemoryStore) -> None:
    store.insert_belief(_belief(ORIGIN_SPECULATIVE))
    promote_on_evidence(store, "p1")
    revert_evidence_promotion(store, "p1")
    b = store.get_belief("p1")
    assert b is not None and b.origin == ORIGIN_SPECULATIVE
    assert _audit(store) == [SOURCE_PROMOTE_EVIDENCE, SOURCE_REVERT_EVIDENCE]


def test_revert_refuses_any_other_origin(store: MemoryStore) -> None:
    store.insert_belief(_belief(ORIGIN_USER_VALIDATED))
    with pytest.raises(ValueError, match="not evidence_promoted"):
        revert_evidence_promotion(store, "p1")


def test_aelf_demote_reverts_an_evidence_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "cli.db"
    s = MemoryStore(str(db))
    s.insert_belief(_belief(ORIGIN_SPECULATIVE))
    promote_on_evidence(s, "p1")
    s.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    out = io.StringIO()
    assert cli_main(["demote", "p1"], out=out) == 0
    assert "reverted to speculative: p1" in out.getvalue()
    s = MemoryStore(str(db))
    try:
        b = s.get_belief("p1")
        assert b is not None and b.origin == ORIGIN_SPECULATIVE
    finally:
        s.close()


def test_it_renders_with_the_inferred_tier() -> None:
    # The words are the agent's; promotion on the user's restatements does
    # not make them observed (#1650).
    from aelfrice.provenance_render import SECTION_BY_ORIGIN, SECTION_INFERRED

    assert SECTION_BY_ORIGIN[ORIGIN_EVIDENCE_PROMOTED] == SECTION_INFERRED
