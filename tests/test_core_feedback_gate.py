"""The core posterior arm counts feedback, not the ingest prior's mass.

`derive()` stamps the undeflated `TYPE_PRIORS` on user-sourced rows: a
`factual` row lands at Beta(3, 1) and a `correction` or `requirement` at
Beta(9, 0.5). Both clear the arm's mean (2/3) and mass (4) thresholds on
insert, so before this change every such row entered `aelf core` with no
evidence. The arm now also needs `CORE_MIN_POSTERIOR_FEEDBACK_EVENTS`
feedback rows that moved the posterior: a source outside
`EXPOSURE_ONLY_FEEDBACK_SOURCES` and a non-zero valence.

The three copies of the rule are pinned together: `aelf core` (cli), the
SessionStart `<core>` lane (`hook._belief_qualifies_core`), and the
callable doctor passes around (`cli.default_core_rule`).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import default_core_rule, main
from aelfrice.doctor import core_gate_candidates
from aelfrice.feedback import apply_feedback
from aelfrice.hook import _belief_qualifies_core  # pyright: ignore[reportPrivateUsage]
from aelfrice.hook_search import HOOK_FEEDBACK_SOURCE
from aelfrice.models import (
    BELIEF_CORRECTION,
    BELIEF_FACTUAL,
    CORE_MIN_POSTERIOR_FEEDBACK_EVENTS,
    FEEDBACK_SOURCE_LOCK_EXPIRE,
    LOCK_NONE,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
)
from aelfrice.sentiment_feedback import SENTIMENT_INFERRED_SOURCE
from aelfrice.store import MemoryStore

_TS = "2026-11-09T00:00:00Z"
_FACTUAL = "b0factual00000000a"
_CORRECTION = "b0correction00000a"


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _belief(bid: str, btype: str, alpha: float, beta: float) -> Belief:
    return Belief(
        id=bid,
        content=f"content of {bid}",
        content_hash="hash_" + bid,
        alpha=alpha,
        beta=beta,
        type=btype,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=_TS,
        last_retrieved_at=None,
        origin=ORIGIN_USER_TRANSCRIPT,
    )


def _seed(db: Path) -> None:
    """One belief at each ingest prior that clears the old posterior arm."""
    s = MemoryStore(str(db))
    try:
        s.insert_belief(_belief(_FACTUAL, BELIEF_FACTUAL, 3.0, 1.0))
        s.insert_belief(_belief(_CORRECTION, BELIEF_CORRECTION, 9.0, 0.5))
    finally:
        s.close()


def _feedback(db: Path, bid: str, n: int, source: str = SENTIMENT_INFERRED_SOURCE) -> None:
    s = MemoryStore(str(db))
    try:
        for _ in range(n):
            apply_feedback(s, bid, 1.0, source, now=_TS, propagate=False)
    finally:
        s.close()


def _audit_rows(db: Path, bid: str, n: int, source: str) -> None:
    s = MemoryStore(str(db))
    try:
        for _ in range(n):
            s.insert_feedback_event(bid, 0.0, source, _TS)
    finally:
        s.close()


def _core_ids(*extra: str) -> list[str]:
    buf = io.StringIO()
    assert main(argv=["core", "--json", "--no-locked", *extra], out=buf) == 0
    text = buf.getvalue()
    if text.strip() == "no core beliefs":
        return []
    rows: list[dict[str, object]] = json.loads(text)
    return sorted(str(r["id"]) for r in rows)


def test_threshold_is_two() -> None:
    assert CORE_MIN_POSTERIOR_FEEDBACK_EVENTS == 2


def test_ingest_priors_alone_do_not_enter_core(isolated_db: Path) -> None:
    _seed(isolated_db)
    assert _core_ids() == []


def test_one_event_is_not_enough(isolated_db: Path) -> None:
    _seed(isolated_db)
    _feedback(isolated_db, _FACTUAL, 1)
    assert _core_ids() == []


def test_two_events_admit(isolated_db: Path) -> None:
    _seed(isolated_db)
    _feedback(isolated_db, _FACTUAL, 2)
    assert _core_ids() == [_FACTUAL]


def test_posterior_signal_label_needs_the_events(isolated_db: Path) -> None:
    _seed(isolated_db)
    _feedback(isolated_db, _FACTUAL, 2)
    buf = io.StringIO()
    assert main(argv=["core", "--json", "--no-locked"], out=buf) == 0
    rows: list[dict[str, object]] = json.loads(buf.getvalue())
    assert [(r["id"], r["signals"]) for r in rows] == [(_FACTUAL, ["posterior"])]
    text = io.StringIO()
    assert main(argv=["core", "--no-locked"], out=text) == 0
    assert f"{_FACTUAL} [α=5.0,β=1.0,μ=0.833]" in text.getvalue()


def test_hook_exposure_rows_do_not_count(isolated_db: Path) -> None:
    _seed(isolated_db)
    s = MemoryStore(str(isolated_db))
    try:
        for _ in range(5):
            apply_feedback(
                s, _FACTUAL, 1.0, HOOK_FEEDBACK_SOURCE, now=_TS,
                propagate=False, update_posterior=False,
            )
    finally:
        s.close()
    assert _core_ids() == []


def test_zero_valence_audit_rows_do_not_count(isolated_db: Path) -> None:
    _seed(isolated_db)
    _audit_rows(isolated_db, _CORRECTION, 3, FEEDBACK_SOURCE_LOCK_EXPIRE)
    assert _core_ids() == []


def test_min_feedback_events_flag_widens_the_lens(isolated_db: Path) -> None:
    _seed(isolated_db)
    assert _core_ids("--min-feedback-events", "0") == [_CORRECTION, _FACTUAL]


def test_posterior_feedback_counts(isolated_db: Path) -> None:
    _seed(isolated_db)
    _feedback(isolated_db, _FACTUAL, 2)
    _audit_rows(isolated_db, _FACTUAL, 1, FEEDBACK_SOURCE_LOCK_EXPIRE)
    _audit_rows(isolated_db, _CORRECTION, 4, HOOK_FEEDBACK_SOURCE)
    s = MemoryStore(str(isolated_db))
    try:
        apply_feedback(s, _CORRECTION, -1.0, SENTIMENT_INFERRED_SOURCE, now=_TS, propagate=False)
        assert s.posterior_feedback_counts() == {_FACTUAL: 2, _CORRECTION: 1}
    finally:
        s.close()


@pytest.mark.parametrize(
    ("events", "expected"),
    [(0, False), (1, False), (2, True), (3, True)],
)
def test_all_three_rule_copies_agree(events: int, expected: bool) -> None:
    b = _belief(_FACTUAL, BELIEF_FACTUAL, 3.0, 1.0)
    assert default_core_rule(b, 0, events) is expected
    assert _belief_qualifies_core(b, 0, events) is expected


def test_doctor_candidates_exclude_prior_only(isolated_db: Path) -> None:
    _seed(isolated_db)
    _feedback(isolated_db, _CORRECTION, 2)
    s = MemoryStore(str(isolated_db))
    try:
        candidates, _ = core_gate_candidates(s, default_core_rule)
    finally:
        s.close()
    assert [b.id for b in candidates] == [_CORRECTION]


def test_session_start_core_lane_needs_the_events(
    isolated_db: Path, tmp_path: Path,
) -> None:
    from aelfrice.hook import _build_session_start_subblock  # pyright: ignore[reportPrivateUsage]

    _seed(isolated_db)
    _feedback(isolated_db, _CORRECTION, 2)
    s = MemoryStore(str(isolated_db))
    try:
        block = _build_session_start_subblock(s, cwd=tmp_path)
    finally:
        s.close()
    assert f'id="{_CORRECTION}"' in block
    assert f'id="{_FACTUAL}"' not in block
