"""Tests for `aelf doctor --promote-retention` (issue #290 phase-3).

Covers:
* find_promotable_snapshots query: respects N>=3 corroborations,
  M>=2 distinct sessions, retention_class='snapshot' filter, and the
  inbound-CONTRADICTS-edge exclusion.
* promote_retention() flips retention_class to 'fact', writes a
  synthetic feedback_history row with source='retention_promotion'
  and valence=0.0, and respects dry_run / max_n.
* set_retention_class store helper updates the row and rejects
  unknown class names.
* CLI wiring: --promote-retention --dry-run prints the candidate
  count without mutating; --promote-retention writes through.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

import aelfrice.cli as cli_module
from aelfrice.doctor import (
    FEEDBACK_SOURCE_RETENTION_PROMOTION,
    PROMOTE_RETENTION_MIN_CORROBORATIONS,
    PROMOTE_RETENTION_MIN_SESSIONS,
    PromotionRunReport,
    format_promotion_report,
    promote_retention,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    LOCK_NONE,
    RETENTION_FACT,
    RETENTION_SNAPSHOT,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(
    store: MemoryStore,
    bid: str,
    *,
    retention_class: str = RETENTION_SNAPSHOT,
) -> Belief:
    b = Belief(
        id=bid,
        content=f"content for {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
        retention_class=retention_class,
    )
    store.insert_belief(b)
    return b


def _corr(
    store: MemoryStore, belief_id: str, *, session: str | None
) -> None:
    store.record_corroboration(
        belief_id,
        source_type="filesystem_ingest",
        session_id=session,
    )


# --- thresholds are spec-anchored ---------------------------------------


def test_promotion_thresholds_match_spec() -> None:
    """docs/belief_retention_class.md §4 nails N=3, M=2."""
    assert PROMOTE_RETENTION_MIN_CORROBORATIONS == 3
    assert PROMOTE_RETENTION_MIN_SESSIONS == 2


# --- find_promotable_snapshots query ------------------------------------


def test_promotable_requires_three_corroborations(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _mk(store, "b1")
    _corr(store, "b1", session="s1")
    _corr(store, "b1", session="s2")
    # Only N=2; below the threshold.
    assert store.find_promotable_snapshots() == []


def test_promotable_requires_two_distinct_sessions(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _mk(store, "b1")
    _corr(store, "b1", session="s1")
    _corr(store, "b1", session="s1")
    _corr(store, "b1", session="s1")
    # N=3 but M=1; not promotable.
    assert store.find_promotable_snapshots() == []


def test_promotable_excludes_null_sessions_from_distinct_count(
    tmp_path: Path,
) -> None:
    """Pre-#192-T3 corroborations may have session_id=NULL. NULL must
    not count as a distinct session, otherwise one un-attributed
    re-ingest could masquerade as cross-session reuse."""
    store = MemoryStore(str(tmp_path / "m.db"))
    _mk(store, "b1")
    _corr(store, "b1", session="s1")
    _corr(store, "b1", session=None)
    _corr(store, "b1", session=None)
    # COUNT(DISTINCT session_id) drops NULLs → distinct == 1 → not promotable.
    assert store.find_promotable_snapshots() == []


def test_promotable_only_returns_snapshot_class(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _mk(store, "b_fact", retention_class=RETENTION_FACT)
    _corr(store, "b_fact", session="s1")
    _corr(store, "b_fact", session="s2")
    _corr(store, "b_fact", session="s3")
    # Already a fact; promotion is a no-op for non-snapshots.
    assert store.find_promotable_snapshots() == []


def test_promotable_excludes_beliefs_with_inbound_contradicts(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _mk(store, "b1")
    _mk(store, "b2", retention_class=RETENTION_FACT)
    _corr(store, "b1", session="s1")
    _corr(store, "b1", session="s2")
    _corr(store, "b1", session="s3")
    store.insert_edge(
        Edge(src="b2", dst="b1", type=EDGE_CONTRADICTS, weight=1.0)
    )
    assert store.find_promotable_snapshots() == []


def test_promotable_returns_qualifying_belief(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _mk(store, "b1")
    _corr(store, "b1", session="s1")
    _corr(store, "b1", session="s2")
    _corr(store, "b1", session="s3")
    promotable = store.find_promotable_snapshots()
    assert [b.id for b in promotable] == ["b1"]


# --- set_retention_class --------------------------------------------------


def test_set_retention_class_updates_row(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _mk(store, "b1")
    store.set_retention_class("b1", RETENTION_FACT)
    refreshed = store.get_belief("b1")
    assert refreshed is not None
    assert refreshed.retention_class == RETENTION_FACT


def test_set_retention_class_rejects_unknown_value(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _mk(store, "b1")
    with pytest.raises(ValueError):
        store.set_retention_class("b1", "garbage")


# --- promote_retention end-to-end ---------------------------------------


def _seed_one_promotable(store: MemoryStore, bid: str = "b1") -> None:
    _mk(store, bid)
    _corr(store, bid, session="s1")
    _corr(store, bid, session="s2")
    _corr(store, bid, session="s3")


def test_promote_retention_dry_run_does_not_mutate(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _seed_one_promotable(store)
    report = promote_retention(store, dry_run=True)
    assert report.candidates_found == 1
    assert report.promoted == 0
    assert report.dry_run is True
    # Belief still snapshot.
    refreshed = store.get_belief("b1")
    assert refreshed is not None
    assert refreshed.retention_class == RETENTION_SNAPSHOT
    # No synthetic feedback row written.
    events = store.list_feedback_events(belief_id="b1")
    assert all(
        e.source != FEEDBACK_SOURCE_RETENTION_PROMOTION for e in events
    )


def test_promote_retention_flips_class_and_records_feedback(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _seed_one_promotable(store)
    report = promote_retention(store, dry_run=False)
    assert report.candidates_found == 1
    assert report.promoted == 1

    refreshed = store.get_belief("b1")
    assert refreshed is not None
    assert refreshed.retention_class == RETENTION_FACT
    # Bayesian prior must be untouched (valence=0.0 contract).
    assert refreshed.alpha == 1.0
    assert refreshed.beta == 1.0

    events = store.list_feedback_events(belief_id="b1")
    matching = [
        e for e in events if e.source == FEEDBACK_SOURCE_RETENTION_PROMOTION
    ]
    assert len(matching) == 1
    assert matching[0].valence == 0.0


def test_promote_retention_respects_max_n(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _seed_one_promotable(store, "b1")
    _seed_one_promotable(store, "b2")
    report = promote_retention(store, dry_run=False, max_n=1)
    assert report.candidates_found == 2
    assert report.promoted == 1


def test_promote_retention_no_candidates_is_noop(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _mk(store, "b1")  # snapshot, no corroborations
    report = promote_retention(store, dry_run=False)
    assert report.candidates_found == 0
    assert report.promoted == 0


# --- format helper -------------------------------------------------------


def test_format_promotion_report_dry_run() -> None:
    text = format_promotion_report(
        PromotionRunReport(candidates_found=2, promoted=0, dry_run=True)
    )
    assert "[dry-run]" in text
    assert "2 snapshot" in text


def test_format_promotion_report_applied() -> None:
    text = format_promotion_report(
        PromotionRunReport(candidates_found=2, promoted=2, dry_run=False)
    )
    assert "promoted: 2" in text


# --- CLI wiring ----------------------------------------------------------


def _doctor_args(**overrides: object) -> argparse.Namespace:
    base = dict(
        scope=None,
        user_settings=None,
        project_root=None,
        hook_failures_log=None,
        classify_orphans=False,
        gc_orphan_feedback=False,
        promote_retention=False,
        replay=False,
        apply=False,
        dry_run=False,
        max=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cli_promote_retention_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "m.db"
    store = MemoryStore(str(db))
    _seed_one_promotable(store)
    store.close()

    monkeypatch.setattr(
        cli_module, "_open_store", lambda: MemoryStore(str(db))
    )
    out = io.StringIO()
    args = _doctor_args(promote_retention=True, dry_run=True)
    rc = cli_module._cmd_doctor(args, out)
    assert rc == 0
    assert "[dry-run]" in out.getvalue()
    # Belief unchanged.
    s2 = MemoryStore(str(db))
    refreshed = s2.get_belief("b1")
    s2.close()
    assert refreshed is not None
    assert refreshed.retention_class == RETENTION_SNAPSHOT


def test_cli_promote_retention_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "m.db"
    store = MemoryStore(str(db))
    _seed_one_promotable(store)
    store.close()

    monkeypatch.setattr(
        cli_module, "_open_store", lambda: MemoryStore(str(db))
    )
    out = io.StringIO()
    args = _doctor_args(promote_retention=True)
    rc = cli_module._cmd_doctor(args, out)
    assert rc == 0
    assert "promoted: 1" in out.getvalue()

    s2 = MemoryStore(str(db))
    refreshed = s2.get_belief("b1")
    s2.close()
    assert refreshed is not None
    assert refreshed.retention_class == RETENTION_FACT
