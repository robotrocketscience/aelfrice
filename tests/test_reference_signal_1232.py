"""Close-the-loop reference signal: recall floor and operator reporting (#1232).

The signal read 0.03% positive on a live store for months without anything
reporting it. Two things are pinned here:

  * the recall defect itself, as an explicit assertion of current behaviour
    rather than a skip, so a fix has to come here and flip it; and
  * the doctor section that makes the rate visible, so the next dead signal
    is caught by reading the report instead of by a store scan.

The fixture beliefs are synthetic prose sized to the live distribution
(p50 437 characters, ~70 words); no store content is reproduced here.
"""
from __future__ import annotations

import sqlite3

import pytest

from aelfrice.doctor import (
    REFERENCE_SIGNAL_DEAD_RATE,
    REFERENCE_SIGNAL_MIN_RESOLVED,
    DoctorReport,
    ReferenceSignalStats,
    _format_reference_signal_section,
    diagnose_reference_signal,
)
from aelfrice.relevance_detection import is_referenced, score_references

# ~490 characters, matching the mean length of a belief that actually gets
# injected on the live store. The exact-substring detector requires every
# one of these characters to reappear in the response.
REPRESENTATIVE_BELIEF = (
    "The retrieval budget defaults to two thousand tokens per injection, and "
    "locked beliefs are packed ahead of the ranked candidate list so that a "
    "user-asserted fact is never displaced by a higher-scoring inference. "
    "When the budget is exhausted the remaining candidates are dropped in "
    "score order rather than truncated mid-belief, because a half-injected "
    "statement reads as a complete one to the consuming agent and there is "
    "no marker that would tell it otherwise."
)

# What an agent actually emits after reading the belief above: the content is
# used, the wording is not reproduced.
PARAPHRASING_RESPONSE = (
    "Right — the cap is 2000 tokens by default, and anything you've locked "
    "gets packed first, so a pinned fact won't lose its slot to something "
    "the ranker happened to score higher. Once the budget runs out the rest "
    "are dropped whole rather than cut off partway, since a partial belief "
    "would look complete to whatever reads it."
)

SHORT_BELIEF = "atomic commits beat batched"


def test_representative_length_belief_is_not_detected() -> None:
    """PINNED DEFECT (#1232): the shipped detector has ~zero recall.

    This is an assertion of current behaviour, not a desired property. The
    belief is used by the response and the detector says it was not, because
    exact-substring detection asks for all 490 characters back verbatim.

    Measured consequence on a live store: 5 positives across 16,355 resolved
    `injection_events`, all of them the same two ~26-character beliefs.

    A fix must come here and flip this assertion. If it starts failing
    because detection improved, that is the fix landing — update it to
    assert detection instead of relaxing it.
    """
    assert is_referenced(REPRESENTATIVE_BELIEF, PARAPHRASING_RESPONSE) is False


def test_the_same_belief_is_detected_when_quoted_verbatim() -> None:
    """Control: the fixture belief is not malformed and the detector works.

    Without this, a `normalize_text` bug that made everything undetectable
    would satisfy the pinned defect above and look like the known problem.
    """
    quoted = f'Per the note: "{REPRESENTATIVE_BELIEF}" — so we cap at 2000.'
    assert is_referenced(REPRESENTATIVE_BELIEF, quoted) is True


def test_a_short_belief_is_detected_in_the_same_response_shape() -> None:
    """Control: length is the discriminator, not the response's shape.

    The only beliefs that ever scored on the live store were short enough to
    be quoted whole. Reproducing that here shows the pinned defect is about
    belief length rather than about this particular paraphrase.
    """
    response = f"Agreed, and remember: {SHORT_BELIEF}. So I'll split the diff."
    assert is_referenced(SHORT_BELIEF, response) is True
    assert is_referenced(REPRESENTATIVE_BELIEF, response) is False


def test_score_references_reports_the_dead_rate_on_a_realistic_batch() -> None:
    """The batch API agrees with the per-belief one, and the rate is zero.

    A test asserting only `is_referenced` would miss a `score_references`
    that diverged from it, which is the path the sweeper actually calls.
    """
    pairs = [(i, REPRESENTATIVE_BELIEF) for i in range(20)]
    scored = score_references(pairs, PARAPHRASING_RESPONSE)
    assert [r for _, r in scored] == [0] * 20
    assert sum(r for _, r in scored) / len(scored) < REFERENCE_SIGNAL_DEAD_RATE


# --- doctor reporting (AC3) ------------------------------------------------


def _mk_store(path, rows: list[tuple[str, int | None]]) -> None:
    """Write a minimal `injection_events` table: (belief_id, referenced)."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE injection_events ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  belief_id TEXT NOT NULL,"
        "  referenced INTEGER)"
    )
    conn.executemany(
        "INSERT INTO injection_events (belief_id, referenced) VALUES (?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_stats_count_resolved_pending_and_distinct_positives(tmp_path) -> None:
    db = tmp_path / "m.db"
    _mk_store(db, [("b1", 1), ("b1", 1), ("b2", 0), ("b3", None)])
    st = diagnose_reference_signal(str(db))
    assert st == ReferenceSignalStats(
        resolved=3, pending=1, positives=2, distinct_positive_beliefs=1
    )
    assert st.positive_rate == pytest.approx(2 / 3)


def test_distinct_positive_beliefs_separates_rate_from_breadth(
    tmp_path,
) -> None:
    """The live failure looked like 5 positives and was really 2 beliefs.

    A report that printed only the rate would have hidden that, so the
    distinct count is asserted to move independently of the positive count.
    """
    db = tmp_path / "m.db"
    _mk_store(db, [("b1", 1)] * 9 + [("b2", 0)])
    st = diagnose_reference_signal(str(db))
    assert st.positives == 9
    assert st.distinct_positive_beliefs == 1


@pytest.mark.parametrize(
    ("n_resolved", "n_positive", "expected_dead"),
    [
        # Enough events to judge, rate under the floor -> dead.
        (REFERENCE_SIGNAL_MIN_RESOLVED, 0, True),
        # Same rate, one event short of the minimum -> not judged.
        (REFERENCE_SIGNAL_MIN_RESOLVED - 1, 0, False),
        # Enough events, rate above the floor -> alive.
        (REFERENCE_SIGNAL_MIN_RESOLVED, REFERENCE_SIGNAL_MIN_RESOLVED, False),
    ],
)
def test_is_dead_needs_both_a_low_rate_and_enough_evidence(
    tmp_path, n_resolved: int, n_positive: int, expected_dead: bool
) -> None:
    """Both halves of the predicate are load-bearing.

    Dropping the minimum-resolved clause would call a fresh store dead;
    dropping the rate clause would call a healthy one dead. Each row here
    fails under exactly one of those mutations.
    """
    db = tmp_path / "m.db"
    rows = [("b%d" % i, 1) for i in range(n_positive)]
    rows += [("bz", 0)] * (n_resolved - n_positive)
    _mk_store(db, rows)
    st = diagnose_reference_signal(str(db))
    assert st.resolved == n_resolved
    assert st.is_dead is expected_dead


def test_report_names_the_rate_and_the_dead_status(tmp_path) -> None:
    db = tmp_path / "m.db"
    _mk_store(db, [("b1", 1)] + [("bz", 0)] * (REFERENCE_SIGNAL_MIN_RESOLVED - 1))
    report = DoctorReport()
    report.reference_signal = diagnose_reference_signal(str(db))
    lines: list[str] = []
    _format_reference_signal_section(report, lines)
    rendered = "\n".join(lines)
    assert "close-the-loop reference signal:" in rendered
    assert f"1 of {REFERENCE_SIGNAL_MIN_RESOLVED} resolved" in rendered
    assert "distinct beliefs ever referenced: 1" in rendered
    assert "DEAD" in rendered
    assert "#1232" in rendered


def test_report_is_silent_when_the_store_supplies_nothing() -> None:
    """No store section rather than a zeroed one — an absent store is not
    evidence of a dead signal."""
    report = DoctorReport()
    lines: list[str] = []
    _format_reference_signal_section(report, lines)
    assert lines == []


@pytest.mark.parametrize("path", [":memory:", "/nonexistent/aelfrice/m.db"])
def test_unreadable_store_degrades_to_none(path: str) -> None:
    assert diagnose_reference_signal(path) is None


def test_store_without_injection_events_degrades_to_none(tmp_path) -> None:
    """A store predating the table must not crash the whole report."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE beliefs (id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()
    assert diagnose_reference_signal(str(db)) is None


def test_empty_injection_events_reports_zero_without_dividing_by_zero(
    tmp_path,
) -> None:
    db = tmp_path / "m.db"
    _mk_store(db, [])
    st = diagnose_reference_signal(str(db))
    assert st == ReferenceSignalStats(
        resolved=0, pending=0, positives=0, distinct_positive_beliefs=0
    )
    assert st.positive_rate == 0.0
    assert st.is_dead is False


def test_diagnose_does_not_write_to_the_store(tmp_path) -> None:
    """Read-only contract: a diagnostic must not mutate the store it reads.

    Constructing a `MemoryStore` would run pending one-shot migrations;
    this path opens `mode=ro` instead. Asserted by mtime and size rather
    than trusted from the connection string.
    """
    db = tmp_path / "m.db"
    _mk_store(db, [("b1", 1), ("b2", 0)])
    before = (db.stat().st_mtime_ns, db.stat().st_size)
    diagnose_reference_signal(str(db))
    assert (db.stat().st_mtime_ns, db.stat().st_size) == before
