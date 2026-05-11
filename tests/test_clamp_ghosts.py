"""Tests for `aelfrice.clamp_ghosts` + the `aelf clamp-ghosts` CLI.

Verifies:
- The structural ghost-α predicate (no feedback_history, no
  belief_corroborations, α > threshold, lock_level='none').
- Dry-run is read-only.
- Apply clamps α and writes one negative-valence feedback_history row
  per belief.
- Idempotent: re-applying after a successful clamp finds zero rows
  because the EXISTS filter now excludes them.
- Locked beliefs are never touched.
- Beliefs with existing feedback_history or belief_corroborations
  are skipped.
- Reversibility: the clamped α can be restored from the
  negative-valence row's magnitude.
- CLI flag plumbing.
- Argument validation (threshold/target positivity, ordering).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from aelfrice.cli import build_parser
from aelfrice.clamp_ghosts import (
    CLAMP_SOURCE,
    DEFAULT_TARGET_ALPHA,
    DEFAULT_THRESHOLD_ALPHA,
    clamp_ghost_alphas,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


# -- belief / store helpers -------------------------------------------------

def _mk(bid: str, *, alpha: float = 1.0, beta: float = 1.0, lock: str = LOCK_NONE) -> Belief:
    return Belief(
        id=bid,
        content=f"belief content for {bid}",
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        locked_at=("2026-04-01T00:00:00Z" if lock != LOCK_NONE else None),
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
        session_id=None,
        origin="unknown",
        corroboration_count=0,
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "test.db")
    yield s
    s.close()


def _alpha_of(store: MemoryStore, bid: str) -> float:
    b = store.get_belief(bid)
    assert b is not None
    return b.alpha


def _feedback_rows_for(store: MemoryStore, bid: str) -> list[tuple[float, str]]:
    cur = store._conn.execute(  # noqa: SLF001
        "SELECT valence, source FROM feedback_history WHERE belief_id = ?",
        (bid,),
    )
    return [(float(r["valence"]), str(r["source"])) for r in cur.fetchall()]


# -- structural predicate ---------------------------------------------------

def test_dry_run_does_not_mutate(store: MemoryStore) -> None:
    ghost = _mk("g1", alpha=10.0)
    store.insert_belief(ghost)

    result = clamp_ghost_alphas(store, dry_run=True)

    assert result.matched == 1
    assert result.clamped == 0
    assert result.skipped == 1
    assert result.dry_run is True
    assert _alpha_of(store, "g1") == 10.0
    assert _feedback_rows_for(store, "g1") == []


def test_apply_clamps_and_writes_audit_row(store: MemoryStore) -> None:
    ghost = _mk("g1", alpha=10.0)
    store.insert_belief(ghost)

    result = clamp_ghost_alphas(store, dry_run=False)

    assert result.matched == 1
    assert result.clamped == 1
    assert result.skipped == 0
    assert _alpha_of(store, "g1") == DEFAULT_TARGET_ALPHA  # 4.0

    rows = _feedback_rows_for(store, "g1")
    assert len(rows) == 1
    valence, source = rows[0]
    assert source == CLAMP_SOURCE
    # negative valence equal to the clamped magnitude (10 - 4)
    assert valence == pytest.approx(-(10.0 - DEFAULT_TARGET_ALPHA))


def test_idempotent_after_apply(store: MemoryStore) -> None:
    """Re-running after a successful clamp finds zero rows.

    The first apply writes feedback_history; the second invocation's
    EXISTS filter excludes the row.
    """
    store.insert_belief(_mk("g1", alpha=10.0))

    first = clamp_ghost_alphas(store, dry_run=False)
    assert first.clamped == 1

    second = clamp_ghost_alphas(store, dry_run=False)
    assert second.matched == 0
    assert second.clamped == 0


def test_skips_locked_beliefs(store: MemoryStore) -> None:
    """Locked beliefs are never clamped.

    Their α reflects an explicit user assertion (e.g. via aelf lock),
    not an unaudited write.
    """
    store.insert_belief(_mk("locked", alpha=100.0, lock=LOCK_USER))

    result = clamp_ghost_alphas(store, dry_run=False)

    assert result.matched == 0
    assert _alpha_of(store, "locked") == 100.0


def test_skips_beliefs_with_feedback_history(store: MemoryStore) -> None:
    """Beliefs with any prior feedback event are skipped — their α has
    a known audit trail."""
    store.insert_belief(_mk("audited", alpha=10.0))
    # Insert one feedback row manually to mark this belief as "explained"
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO feedback_history (belief_id, valence, source, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("audited", 0.1, "hook", "2026-04-26T00:00:01Z"),
    )
    store._conn.commit()  # noqa: SLF001

    result = clamp_ghost_alphas(store, dry_run=False)

    assert result.matched == 0
    assert _alpha_of(store, "audited") == 10.0


def test_skips_beliefs_with_corroborations(store: MemoryStore) -> None:
    """Beliefs with prior corroborations are skipped — their α is at
    least defensible against a multi-source ingest history."""
    store.insert_belief(_mk("corroborated", alpha=10.0))
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO belief_corroborations "
        "(belief_id, ingested_at, source_type, session_id, source_path_hash) "
        "VALUES (?, ?, ?, ?, ?)",
        ("corroborated", "2026-04-26T00:00:01Z", "transcript_ingest", None, None),
    )
    store._conn.commit()  # noqa: SLF001

    result = clamp_ghost_alphas(store, dry_run=False)

    assert result.matched == 0
    assert _alpha_of(store, "corroborated") == 10.0


def test_skips_beliefs_below_threshold(store: MemoryStore) -> None:
    """α at or below the threshold is not a ghost candidate."""
    store.insert_belief(_mk("just_under", alpha=4.0))  # == threshold
    store.insert_belief(_mk("comfortably_under", alpha=2.5))

    result = clamp_ghost_alphas(store, dry_run=False, threshold_alpha=4.0)

    assert result.matched == 0
    assert _alpha_of(store, "just_under") == 4.0
    assert _alpha_of(store, "comfortably_under") == 2.5


def test_reversibility_via_feedback_history(store: MemoryStore) -> None:
    """The clamp event's negative valence magnitude restores prior α."""
    store.insert_belief(_mk("g1", alpha=84.0))

    clamp_ghost_alphas(store, dry_run=False)
    assert _alpha_of(store, "g1") == DEFAULT_TARGET_ALPHA

    # Reverse the clamp: read the negative valence and add its magnitude back.
    cur = store._conn.execute(  # noqa: SLF001
        "SELECT valence FROM feedback_history "
        "WHERE belief_id = ? AND source = ?",
        ("g1", CLAMP_SOURCE),
    )
    rows = cur.fetchall()
    assert len(rows) == 1
    valence = float(rows[0]["valence"])
    store._conn.execute(  # noqa: SLF001
        "UPDATE beliefs SET alpha = alpha + ? WHERE id = ?",
        (-valence, "g1"),
    )
    store._conn.commit()  # noqa: SLF001

    assert _alpha_of(store, "g1") == 84.0  # restored exactly


def test_limit_caps_processed_rows(store: MemoryStore) -> None:
    for i in range(5):
        store.insert_belief(_mk(f"g{i}", alpha=10.0))

    result = clamp_ghost_alphas(store, dry_run=False, limit=2)

    assert result.matched == 2
    assert result.clamped == 2
    # The other three remain unclamped at α=10
    unclamped = sum(
        1 for i in range(5) if _alpha_of(store, f"g{i}") == 10.0
    )
    assert unclamped == 3


def test_sample_capped_at_10(store: MemoryStore) -> None:
    for i in range(15):
        store.insert_belief(_mk(f"g{i:02d}", alpha=10.0))

    result = clamp_ghost_alphas(store, dry_run=True)

    assert result.matched == 15
    assert len(result.sample) == 10


def test_validation_rejects_target_above_threshold() -> None:
    """clamp_ghost_alphas raises if target > threshold (would no-op)."""

    class _NopStore:  # never reached
        _conn = None

    with pytest.raises(ValueError, match="target_alpha"):
        clamp_ghost_alphas(_NopStore(), threshold_alpha=4.0, target_alpha=8.0)  # type: ignore[arg-type]


def test_validation_rejects_nonpositive_alphas() -> None:
    class _NopStore:
        _conn = None

    with pytest.raises(ValueError, match="must be positive"):
        clamp_ghost_alphas(_NopStore(), threshold_alpha=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be positive"):
        clamp_ghost_alphas(_NopStore(), target_alpha=-1.0)  # type: ignore[arg-type]


# -- CLI surface ------------------------------------------------------------

@pytest.fixture
def cli_store_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    return db


def _run_cli(*args: str) -> tuple[int, str]:
    parser = build_parser()
    ns = parser.parse_args(["clamp-ghosts", *args])
    out = io.StringIO()
    code: int = ns.func(ns, out)  # type: ignore[attr-defined]
    return code, out.getvalue()


def test_clamp_ghosts_subcommand_registered() -> None:
    from aelfrice.cli import _known_cli_subcommands
    assert "clamp-ghosts" in _known_cli_subcommands()


def test_clamp_ghosts_cli_dry_run_default(cli_store_path: Path) -> None:
    s = MemoryStore(cli_store_path)
    s.insert_belief(_mk("g1", alpha=10.0))
    s.close()

    code, output = _run_cli()  # no flags → dry-run

    assert code == 0
    assert "matched=1" in output
    assert "clamped=0" in output
    assert "dry_run=True" in output
    assert "--apply" in output  # the "run with --apply" hint

    s2 = MemoryStore(cli_store_path)
    try:
        assert s2.get_belief("g1").alpha == 10.0
    finally:
        s2.close()


def test_clamp_ghosts_cli_apply_clamps(cli_store_path: Path) -> None:
    s = MemoryStore(cli_store_path)
    s.insert_belief(_mk("g1", alpha=10.0))
    s.close()

    code, output = _run_cli("--apply")

    assert code == 0
    assert "clamped=1" in output
    assert "dry_run=False" in output

    s2 = MemoryStore(cli_store_path)
    try:
        assert s2.get_belief("g1").alpha == DEFAULT_TARGET_ALPHA
    finally:
        s2.close()


def test_clamp_ghosts_cli_threshold_and_target_flags(cli_store_path: Path) -> None:
    s = MemoryStore(cli_store_path)
    s.insert_belief(_mk("g1", alpha=10.0))
    s.close()

    code, output = _run_cli("--threshold", "8.0", "--target", "6.0", "--apply")

    assert code == 0
    assert "threshold=8.0" in output
    assert "target=6.0" in output

    s2 = MemoryStore(cli_store_path)
    try:
        assert s2.get_belief("g1").alpha == 6.0
    finally:
        s2.close()
