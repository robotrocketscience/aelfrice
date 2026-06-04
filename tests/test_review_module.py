"""Tests for the aelfrice.review module (#936 Phase A).

Covers:
- select_candidates: ordering with synthetic beliefs
- render_review_file: exact-string snapshot on hand-built fixture
- parse_review_file: empty row, single-box rows (each verdict),
  ambiguous row raises, malformed row raises
- apply_decisions: each verdict's side-effect, missing-id error, mixed batch
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_USER_STATED,
    Belief,
)
from aelfrice.review import (
    AmbiguousRowError,
    ApplyReport,
    MalformedRowError,
    ParsedDecision,
    apply_decisions,
    parse_review_file,
    render_review_file,
    select_candidates,
)
from aelfrice.store import MemoryStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 6, 4, 10, 0, 0, tzinfo=timezone.utc)
_NOW_ISO = "2026-06-04T10:00:00Z"


def _mk_belief(
    bid: str,
    content: str = "belief content",
    created_at: str = "2026-01-01T00:00:00Z",
    last_retrieved_at: str | None = None,
    last_confirmed_at: str | None = None,
    lock_level: str = LOCK_NONE,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash="h_" + bid,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=last_retrieved_at,
        last_confirmed_at=last_confirmed_at,
    )


def _store_with(*beliefs: Belief) -> MemoryStore:
    s = MemoryStore(":memory:")
    for b in beliefs:
        s.insert_belief(b)
    return s


# ── select_candidates ─────────────────────────────────────────────────────────

def test_select_candidates_empty_store() -> None:
    s = MemoryStore(":memory:")
    assert select_candidates(s) == []


def test_select_candidates_ordering() -> None:
    """Never-confirmed, never-retrieved beliefs should surface before confirmed."""
    b_confirmed = _mk_belief(
        "confirmed",
        content="confirmed belief",
        created_at="2026-01-01T00:00:00Z",
        last_confirmed_at="2026-05-01T00:00:00Z",
    )
    b_cold = _mk_belief(
        "cold",
        content="cold belief",
        created_at="2026-01-02T00:00:00Z",
        last_retrieved_at="2026-01-05T00:00:00Z",
    )
    b_never = _mk_belief(
        "never",
        content="never touched",
        created_at="2026-01-01T00:00:00Z",
    )
    s = _store_with(b_confirmed, b_cold, b_never)
    result = select_candidates(s)
    ids = [b.id for b in result]
    # never-confirmed null before retrieved-once, both before confirmed
    assert ids.index("never") < ids.index("cold")
    assert ids.index("cold") < ids.index("confirmed")


def test_select_candidates_excludes_locked() -> None:
    b_open = _mk_belief("open", content="open belief")
    b_locked = _mk_belief("locked", content="locked belief", lock_level=LOCK_USER)
    s = _store_with(b_open, b_locked)
    ids = [b.id for b in select_candidates(s)]
    assert "open" in ids
    assert "locked" not in ids


def test_select_candidates_excludes_soft_deleted() -> None:
    b_active = _mk_belief("active", content="active")
    b_deleted = _mk_belief("deleted", content="deleted")
    s = _store_with(b_active, b_deleted)
    s.soft_delete_belief("deleted")
    ids = [b.id for b in select_candidates(s)]
    assert "active" in ids
    assert "deleted" not in ids


def test_select_candidates_limit() -> None:
    s = MemoryStore(":memory:")
    for i in range(15):
        s.insert_belief(_mk_belief(
            f"b{i:02d}",
            content=f"belief number {i:02d}",
            created_at=f"2026-01-{i+1:02d}T00:00:00Z",
        ))
    assert len(select_candidates(s, limit=5)) == 5
    assert len(select_candidates(s, limit=10)) == 10
    assert len(select_candidates(s, limit=20)) == 15


# ── render_review_file ────────────────────────────────────────────────────────

def test_render_review_file_empty_candidates() -> None:
    out = render_review_file([], now=_NOW)
    assert "## aelfrice review — generated 2026-06-04" in out
    # No belief rows in an empty file
    assert "[ ] keep" not in out


def test_render_review_file_snapshot() -> None:
    """Exact format check on a two-belief fixture."""
    b1 = _mk_belief(
        "abc123",
        content="pipx is no longer a supported install path",
        created_at="2026-01-10T00:00:00Z",
        last_retrieved_at="2026-01-20T00:00:00Z",
    )
    b2 = _mk_belief(
        "def456",
        content="atomic commits beat batched",
        created_at="2026-01-01T00:00:00Z",
    )
    out = render_review_file([b1, b2], now=_NOW)
    assert "## aelfrice review — generated 2026-06-04" in out
    # b1 row
    assert "- [ ] keep   [ ] remove   [ ] lock" in out
    assert "| abc123" in out
    assert "pipx is no longer a supported install path" in out
    # b2 row
    assert "| def456" in out
    assert "atomic commits beat batched" in out


def test_render_review_file_truncates_long_content() -> None:
    # Use alternating chars so we can distinguish the tail from the prefix.
    long_content = "ab" * 50  # 100 chars total
    b = _mk_belief("bid1", content=long_content, created_at="2026-01-01T00:00:00Z")
    out = render_review_file([b], now=_NOW)
    assert "..." in out
    # The snippet must contain the first 80 chars
    assert long_content[:80] in out
    # The full 100-char content must NOT appear (it was truncated)
    assert long_content not in out


def test_render_review_file_age_and_cold_days() -> None:
    """Verify age_days and cold_days appear in the row."""
    b = _mk_belief(
        "b1",
        content="content",
        created_at="2026-01-01T00:00:00Z",  # 154d before 2026-06-04
    )
    out = render_review_file([b], now=_NOW)
    assert "154d old" in out
    assert "154d cold" in out  # cold == age when no retrieval


def test_render_review_file_cold_days_uses_retrieval() -> None:
    b = _mk_belief(
        "b1",
        content="content",
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at="2026-05-01T00:00:00Z",  # 34d before 2026-06-04
    )
    out = render_review_file([b], now=_NOW)
    assert "34d cold" in out


# ── parse_review_file ─────────────────────────────────────────────────────────

def test_parse_review_file_empty_row_yields_skip() -> None:
    text = "- [ ] keep   [ ] remove   [ ] lock   | abc123 (1d old, 1d cold) — content\n"
    decisions = parse_review_file(text)
    assert len(decisions) == 1
    assert decisions[0].verdict == "skip"
    assert decisions[0].belief_id == "abc123"


def test_parse_review_file_keep() -> None:
    text = "- [x] keep   [ ] remove   [ ] lock   | abc123 (1d old, 1d cold) — content\n"
    decisions = parse_review_file(text)
    assert decisions[0].verdict == "keep"
    assert decisions[0].belief_id == "abc123"


def test_parse_review_file_remove() -> None:
    text = "- [ ] keep   [x] remove   [ ] lock   | abc123 (1d old, 1d cold) — content\n"
    decisions = parse_review_file(text)
    assert decisions[0].verdict == "remove"


def test_parse_review_file_lock() -> None:
    text = "- [ ] keep   [ ] remove   [x] lock   | abc123 (1d old, 1d cold) — content\n"
    decisions = parse_review_file(text)
    assert decisions[0].verdict == "lock"


def test_parse_review_file_uppercase_x() -> None:
    text = "- [X] keep   [ ] remove   [ ] lock   | abc123 (1d old, 1d cold) — content\n"
    decisions = parse_review_file(text)
    assert decisions[0].verdict == "keep"


def test_parse_review_file_ambiguous_row_raises() -> None:
    text = "- [x] keep   [x] remove   [ ] lock   | abc123 (1d old, 1d cold) — content\n"
    with pytest.raises(AmbiguousRowError) as exc_info:
        parse_review_file(text)
    assert "abc123" in str(exc_info.value)


def test_parse_review_file_all_three_checked_raises() -> None:
    text = "- [x] keep   [x] remove   [x] lock   | abc123 (1d old, 1d cold) — content\n"
    with pytest.raises(AmbiguousRowError):
        parse_review_file(text)


def test_parse_review_file_mixed_batch() -> None:
    text = (
        "- [x] keep   [ ] remove   [ ] lock   | id1 (1d old, 1d cold) — a\n"
        "- [ ] keep   [x] remove   [ ] lock   | id2 (1d old, 1d cold) — b\n"
        "- [ ] keep   [ ] remove   [x] lock   | id3 (1d old, 1d cold) — c\n"
        "- [ ] keep   [ ] remove   [ ] lock   | id4 (1d old, 1d cold) — d\n"
    )
    decisions = parse_review_file(text)
    verdicts = {d.belief_id: d.verdict for d in decisions}
    assert verdicts["id1"] == "keep"
    assert verdicts["id2"] == "remove"
    assert verdicts["id3"] == "lock"
    assert verdicts["id4"] == "skip"


def test_parse_review_file_ignores_non_matching_lines() -> None:
    text = (
        "## aelfrice review — generated 2026-06-04\n"
        "For each belief: leave checkboxes empty to no-op; check exactly one verdict per row.\n"
        "\n"
        "- [x] keep   [ ] remove   [ ] lock   | abc123 (1d old, 1d cold) — content\n"
    )
    decisions = parse_review_file(text)
    assert len(decisions) == 1
    assert decisions[0].belief_id == "abc123"


# ── apply_decisions ───────────────────────────────────────────────────────────

def _decision(belief_id: str, verdict: str) -> ParsedDecision:
    return ParsedDecision(
        belief_id=belief_id,
        verdict=verdict,  # type: ignore[arg-type]
        row_text=f"- [{verdict[0] if verdict != 'skip' else ' '}] ...",
    )


def test_apply_decisions_keep_updates_last_confirmed_at() -> None:
    b = _mk_belief("b1", content="keep me")
    s = _store_with(b)
    report = apply_decisions(s, [_decision("b1", "keep")], now=_NOW_ISO)
    assert report.kept == ["b1"]
    assert report.errors == []
    got = s.get_belief("b1")
    assert got is not None
    assert got.last_confirmed_at == _NOW_ISO


def test_apply_decisions_remove_soft_deletes() -> None:
    b = _mk_belief("b1", content="remove me")
    s = _store_with(b)
    report = apply_decisions(s, [_decision("b1", "remove")], now=_NOW_ISO)
    assert report.removed == ["b1"]
    assert report.errors == []
    got = s.get_belief("b1")
    assert got is not None
    assert got.valid_to is not None  # soft-deleted


def test_apply_decisions_remove_writes_audit_row() -> None:
    b = _mk_belief("b1", content="remove me")
    s = _store_with(b)
    apply_decisions(s, [_decision("b1", "remove")], now=_NOW_ISO)
    events = s.list_feedback_events("b1")
    sources = [e.source for e in events]
    assert "review:remove" in sources


def test_apply_decisions_lock_sets_lock_level() -> None:
    b = _mk_belief("b1", content="lock me")
    s = _store_with(b)
    report = apply_decisions(s, [_decision("b1", "lock")], now=_NOW_ISO)
    assert report.locked == ["b1"]
    assert report.errors == []
    got = s.get_belief("b1")
    assert got is not None
    assert got.lock_level == LOCK_USER
    assert got.locked_at == _NOW_ISO
    assert got.origin == ORIGIN_USER_STATED


def test_apply_decisions_lock_writes_audit_row() -> None:
    b = _mk_belief("b1", content="lock me")
    s = _store_with(b)
    apply_decisions(s, [_decision("b1", "lock")], now=_NOW_ISO)
    events = s.list_feedback_events("b1")
    sources = [e.source for e in events]
    assert "review:lock" in sources


def test_apply_decisions_skip_is_noop() -> None:
    b = _mk_belief("b1", content="skip me")
    s = _store_with(b)
    report = apply_decisions(s, [_decision("b1", "skip")], now=_NOW_ISO)
    assert report.skipped == ["b1"]
    got = s.get_belief("b1")
    assert got is not None
    assert got.last_confirmed_at is None
    assert got.valid_to is None
    assert got.lock_level == LOCK_NONE


def test_apply_decisions_missing_id_adds_to_errors() -> None:
    s = MemoryStore(":memory:")
    report = apply_decisions(s, [_decision("nonexistent", "keep")], now=_NOW_ISO)
    assert "nonexistent" in report.errors[0]
    assert report.kept == []


def test_apply_decisions_missing_id_does_not_abort_rest() -> None:
    b = _mk_belief("b2", content="real belief")
    s = _store_with(b)
    decisions = [
        _decision("missing", "keep"),
        _decision("b2", "keep"),
    ]
    report = apply_decisions(s, decisions, now=_NOW_ISO)
    assert len(report.errors) == 1
    assert "b2" in report.kept
    got = s.get_belief("b2")
    assert got is not None
    assert got.last_confirmed_at == _NOW_ISO


def test_apply_decisions_mixed_batch() -> None:
    b_keep = _mk_belief("keep_me", content="keep")
    b_remove = _mk_belief("remove_me", content="remove")
    b_lock = _mk_belief("lock_me", content="lock")
    b_skip = _mk_belief("skip_me", content="skip")
    s = _store_with(b_keep, b_remove, b_lock, b_skip)
    decisions = [
        _decision("keep_me", "keep"),
        _decision("remove_me", "remove"),
        _decision("lock_me", "lock"),
        _decision("skip_me", "skip"),
    ]
    report = apply_decisions(s, decisions, now=_NOW_ISO)
    assert "keep_me" in report.kept
    assert "remove_me" in report.removed
    assert "lock_me" in report.locked
    assert "skip_me" in report.skipped
    assert report.errors == []
