"""Tests for `replay_full_equality` (#262 shape-equality probe).

Each test states a falsifiable hypothesis. All tests use an in-memory or
tmp_path SQLite store. Target runtime < 500 ms total.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.derivation import (
    DerivationInput,
    _belief_id,
    _content_hash,
    derive,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_PREFERENCE,
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_FILESYSTEM,
    INGEST_SOURCE_LEGACY_UNKNOWN,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_UNKNOWN,
    Belief,
)
from aelfrice.replay import FullEqualityReport, replay_full_equality
from aelfrice.store import MemoryStore

_TS = "2026-01-01T00:00:00+00:00"

# A sentence that the classifier reliably persists as factual.
_FACTUAL_SENTENCE = (
    "The configuration database is stored at /var/lib/aelfrice/brain.db "
    "and is backed up nightly."
)
_FACTUAL_SENTENCE_2 = (
    "The default listening port for the web server is 8443 on all interfaces."
)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "test_replay.db"))
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_belief(raw_text: str, source_path: str = "test:source") -> Belief:
    """Derive a Belief from raw_text via the filesystem path."""
    inp = DerivationInput(
        raw_text=raw_text,
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path=source_path,
        ts=_TS,
    )
    out = derive(inp)
    assert out.belief is not None, f"derive returned None for {raw_text!r}"
    return out.belief


def _ingest(store: MemoryStore, raw_text: str, source_path: str = "test:source") -> str:
    """Insert a belief + log row into the store. Returns belief_id."""
    b = _make_belief(raw_text, source_path)
    log_id = store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path=source_path,
        raw_text=raw_text,
        derived_belief_ids=[b.id],
        ts=_TS,
    )
    store.insert_belief(b)
    return b.id


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------


def test_empty_store_all_zero(store: MemoryStore) -> None:
    """Hypothesis: an empty store produces an all-zero report with has_drift False
    and implemented=True. Falsifiable if any counter is non-zero or has_drift True."""
    report = replay_full_equality(store)
    assert report.implemented is True
    assert report.total_log_rows == 0
    assert report.excluded_legacy_unknown == 0
    assert report.matched == 0
    assert report.mismatched == 0
    assert report.derived_orphan == 0
    assert report.canonical_orphan == 0
    assert report.legacy_origin_backfill == 0
    assert report.feedback_derived_edges == 0
    assert report.has_drift is False
    assert report.drift_examples == {
        "mismatched": [],
        "derived_orphan": [],
        "canonical_orphan": [],
    }


# ---------------------------------------------------------------------------
# Single ingested belief — replay matches
# ---------------------------------------------------------------------------


def test_single_belief_replay_matches(store: MemoryStore) -> None:
    """Hypothesis: a freshly ingested belief replays as matched=1, no drift.
    Falsifiable by matched != 1 or any non-zero drift counter."""
    _ingest(store, _FACTUAL_SENTENCE)
    report = replay_full_equality(store)
    assert report.implemented is True
    assert report.total_log_rows == 1
    assert report.matched == 1
    assert report.mismatched == 0
    assert report.derived_orphan == 0
    assert report.has_drift is False


# ---------------------------------------------------------------------------
# Posterior drift is not drift (shape-equality ignores alpha/beta)
# ---------------------------------------------------------------------------


def test_posterior_mutation_not_drift(store: MemoryStore) -> None:
    """Hypothesis: updating alpha/beta after ingest does not cause drift.
    Replay compares content_hash/type/origin only; alpha/beta are excluded.
    Falsifiable by mismatched > 0 after an alpha/beta mutation."""
    bid = _ingest(store, _FACTUAL_SENTENCE)

    # Manually mutate alpha/beta to simulate feedback events.
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "UPDATE beliefs SET alpha = 5.0, beta = 1.5 WHERE id = ?",
        (bid,),
    )
    store._conn.commit()  # pyright: ignore[reportPrivateUsage]

    report = replay_full_equality(store)
    assert report.mismatched == 0
    assert report.matched == 1
    assert report.has_drift is False


# ---------------------------------------------------------------------------
# Origin backfill cohort — canonical origin NULL counts as match
# ---------------------------------------------------------------------------


def test_origin_null_backfill_cohort(store: MemoryStore) -> None:
    """Hypothesis: a canonical belief with origin=NULL/unknown is matched
    (not mismatched) when derived belief has a non-unknown origin.
    The legacy_origin_backfill counter is incremented.
    Falsifiable by mismatched > 0 or legacy_origin_backfill == 0."""
    b = _make_belief(_FACTUAL_SENTENCE)

    # Insert with origin='unknown' (legacy schema default).
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        """
        INSERT INTO beliefs (
            id, content, content_hash, alpha, beta, type, lock_level,
            locked_at, created_at, last_retrieved_at,
            session_id, origin
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'unknown')
        """,
        (
            b.id, b.content, b.content_hash,
            b.alpha, b.beta, b.type, b.lock_level,
            b.locked_at, b.created_at,
            b.last_retrieved_at, b.session_id,
        ),
    )
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "INSERT INTO beliefs_fts (id, content) VALUES (?, ?)",
        (b.id, b.content),
    )
    store._conn.commit()  # pyright: ignore[reportPrivateUsage]

    # Write the log row pointing at this belief.
    store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="test:source",
        raw_text=_FACTUAL_SENTENCE,
        derived_belief_ids=[b.id],
        ts=_TS,
    )

    report = replay_full_equality(store)
    assert report.mismatched == 0
    assert report.matched == 1
    assert report.legacy_origin_backfill == 1
    assert report.has_drift is False


# ---------------------------------------------------------------------------
# Genuine mismatch — corrupted content_hash
# ---------------------------------------------------------------------------


def test_genuine_mismatch_content_hash(store: MemoryStore) -> None:
    """Hypothesis: corrupting a canonical belief's content_hash post-ingest
    causes mismatched=1 and captures a drift example with raw_text truncated
    to ≤200 chars.
    Falsifiable by mismatched != 1 or absent/wrong drift example."""
    bid = _ingest(store, _FACTUAL_SENTENCE)

    # Corrupt the canonical content_hash.
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "UPDATE beliefs SET content_hash = 'CORRUPTED' WHERE id = ?",
        (bid,),
    )
    store._conn.commit()  # pyright: ignore[reportPrivateUsage]

    report = replay_full_equality(store)
    assert report.mismatched == 1
    assert report.matched == 0
    assert report.has_drift is True

    examples = report.drift_examples["mismatched"]
    assert len(examples) == 1
    ex = examples[0]
    assert ex["belief_id"] == bid
    assert len(str(ex["raw_text"])) <= 200
    assert "content_hash" in ex["fields_diff"]
    assert ex["fields_diff"]["content_hash"]["canonical"] == "CORRUPTED"


# ---------------------------------------------------------------------------
# Derived orphan — belief deleted but log row remains
# ---------------------------------------------------------------------------


def test_derived_orphan(store: MemoryStore) -> None:
    """Hypothesis: when the canonical belief is deleted but the log row
    remains, derived_orphan=1 and the example captures synthesized_belief_id.
    Falsifiable by derived_orphan != 1."""
    bid = _ingest(store, _FACTUAL_SENTENCE)

    # Delete the canonical belief (leave the log row).
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM beliefs WHERE id = ?", (bid,)
    )
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM beliefs_fts WHERE id = ?", (bid,)
    )
    store._conn.commit()  # pyright: ignore[reportPrivateUsage]

    report = replay_full_equality(store)
    assert report.derived_orphan == 1
    assert report.matched == 0
    assert report.has_drift is True

    examples = report.drift_examples["derived_orphan"]
    assert len(examples) == 1
    ex = examples[0]
    assert ex["synthesized_belief_id"] == bid
    assert len(str(ex["raw_text"])) <= 200


# ---------------------------------------------------------------------------
# Canonical orphan — belief with no non-legacy log row
# ---------------------------------------------------------------------------


def test_canonical_orphan(store: MemoryStore) -> None:
    """Hypothesis: a belief inserted directly without any log row counts as
    canonical_orphan=1.  This covers the pre-#205 and pre-legacy-migration paths.
    Falsifiable by canonical_orphan != 1."""
    b = _make_belief(_FACTUAL_SENTENCE)

    # Insert belief directly; do NOT write any ingest_log row.
    store.insert_belief(b)

    report = replay_full_equality(store)
    assert report.canonical_orphan == 1
    # canonical_orphan does NOT trigger has_drift per spec.
    assert report.has_drift is False

    examples = report.drift_examples["canonical_orphan"]
    assert len(examples) == 1
    assert examples[0]["belief_id"] == b.id
    assert examples[0]["content_hash"] == b.content_hash


# ---------------------------------------------------------------------------
# legacy_unknown exclusion
# ---------------------------------------------------------------------------


def test_legacy_unknown_rows_excluded(store: MemoryStore) -> None:
    """Hypothesis: log rows with source_kind=legacy_unknown are excluded
    from total_log_rows and do not trigger any drift bucket.
    Falsifiable if excluded_legacy_unknown == 0 or total_log_rows > 0."""
    # Insert a belief without a non-legacy log row.
    b = _make_belief(_FACTUAL_SENTENCE)
    store.insert_belief(b)

    # Write a legacy_unknown log row pointing at the belief.
    store.record_ingest(
        source_kind=INGEST_SOURCE_LEGACY_UNKNOWN,
        source_path=None,
        raw_text=_FACTUAL_SENTENCE,
        derived_belief_ids=[b.id],
        ts=_TS,
    )

    report = replay_full_equality(store)
    assert report.excluded_legacy_unknown == 1
    assert report.total_log_rows == 0
    assert report.matched == 0
    assert report.mismatched == 0
    assert report.derived_orphan == 0
    # The belief's only log row is legacy_unknown → canonical_orphan.
    assert report.canonical_orphan == 1
    assert report.has_drift is False


def test_legacy_unknown_mixed_with_non_legacy(store: MemoryStore) -> None:
    """Hypothesis: if a belief has both a legacy_unknown row and a real row,
    the real row is processed and the belief is matched (not orphan).
    Falsifiable by canonical_orphan > 0 or matched != 1."""
    bid = _ingest(store, _FACTUAL_SENTENCE)

    # Add a legacy_unknown row pointing at the same belief.
    store.record_ingest(
        source_kind=INGEST_SOURCE_LEGACY_UNKNOWN,
        source_path=None,
        raw_text=_FACTUAL_SENTENCE,
        derived_belief_ids=[bid],
        ts=_TS,
    )

    report = replay_full_equality(store)
    assert report.excluded_legacy_unknown == 1
    assert report.total_log_rows == 1
    assert report.matched == 1
    assert report.canonical_orphan == 0


# ---------------------------------------------------------------------------
# Drift example cap
# ---------------------------------------------------------------------------


def test_drift_example_cap(store: MemoryStore) -> None:
    """Hypothesis: with 25 mismatches and drift_examples=10, only 10 examples
    are captured.  Falsifiable if len(drift_examples['mismatched']) > 10."""
    source = "test:source"
    inserted_ids = []
    for i in range(25):
        raw = f"The system parameter {i} controls processing behaviour on startup."
        inp = DerivationInput(
            raw_text=raw,
            source_kind=INGEST_SOURCE_FILESYSTEM,
            source_path=source,
            ts=_TS,
        )
        out = derive(inp)
        if out.belief is None:
            continue
        b = out.belief
        store.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM,
            source_path=source,
            raw_text=raw,
            derived_belief_ids=[b.id],
            ts=_TS,
        )
        store.insert_belief(b)
        inserted_ids.append(b.id)

    # Corrupt all canonical beliefs' content_hash.
    for bid in inserted_ids:
        store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "UPDATE beliefs SET content_hash = 'CORRUPTED_' || id WHERE id = ?",
            (bid,),
        )
    store._conn.commit()  # pyright: ignore[reportPrivateUsage]

    report = replay_full_equality(store, drift_examples=10)
    assert report.mismatched == len(inserted_ids)
    assert len(report.drift_examples["mismatched"]) == 10


# ---------------------------------------------------------------------------
# Scope flag
# ---------------------------------------------------------------------------


def test_scope_since_v2_equivalent_to_all(store: MemoryStore) -> None:
    """Hypothesis: post-#263 migration, scope='since-v2' and scope='all'
    produce identical counts.  Falsifiable by any counter differing between
    the two calls."""
    _ingest(store, _FACTUAL_SENTENCE)
    _ingest(store, _FACTUAL_SENTENCE_2)

    report_all = replay_full_equality(store, scope="all")
    report_v2 = replay_full_equality(store, scope="since-v2")

    assert report_all.total_log_rows == report_v2.total_log_rows
    assert report_all.matched == report_v2.matched
    assert report_all.mismatched == report_v2.mismatched
    assert report_all.derived_orphan == report_v2.derived_orphan
    assert report_all.canonical_orphan == report_v2.canonical_orphan


# ---------------------------------------------------------------------------
# has_drift semantics
# ---------------------------------------------------------------------------


def test_has_drift_false_when_canonical_orphan_only(store: MemoryStore) -> None:
    """Hypothesis: canonical_orphan alone does not trigger has_drift.
    Per spec, only mismatched + derived_orphan constitute drift.
    Falsifiable by has_drift True."""
    b = _make_belief(_FACTUAL_SENTENCE)
    store.insert_belief(b)

    report = replay_full_equality(store)
    assert report.canonical_orphan == 1
    assert report.has_drift is False


def test_has_drift_false_when_legacy_backfill_only(store: MemoryStore) -> None:
    """Hypothesis: legacy_origin_backfill alone does not trigger has_drift.
    Falsifiable by has_drift True."""
    b = _make_belief(_FACTUAL_SENTENCE)
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        """
        INSERT INTO beliefs (
            id, content, content_hash, alpha, beta, type, lock_level,
            locked_at, created_at, last_retrieved_at,
            session_id, origin
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'unknown')
        """,
        (
            b.id, b.content, b.content_hash,
            b.alpha, b.beta, b.type, b.lock_level,
            b.locked_at, b.created_at,
            b.last_retrieved_at, b.session_id,
        ),
    )
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "INSERT INTO beliefs_fts (id, content) VALUES (?, ?)",
        (b.id, b.content),
    )
    store._conn.commit()  # pyright: ignore[reportPrivateUsage]

    store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="test:source",
        raw_text=_FACTUAL_SENTENCE,
        derived_belief_ids=[b.id],
        ts=_TS,
    )

    report = replay_full_equality(store)
    assert report.legacy_origin_backfill == 1
    assert report.matched == 1
    assert report.has_drift is False



# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_doctor_replay_exit_0_clean_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hypothesis: `aelf doctor --replay` exits 0 on a store with no drift.
    Falsifiable by any non-zero exit code."""
    import io
    from aelfrice.cli import main

    db = str(tmp_path / "brain.db")
    monkeypatch.setenv("AELFRICE_DB", db)

    # Ingest a belief so total_log_rows >= 1.
    s = MemoryStore(db)
    _ingest(s, _FACTUAL_SENTENCE)
    s.close()

    out = io.StringIO()
    rc = main(["doctor", "--replay"], out=out)
    assert rc == 0


def test_cli_doctor_replay_exit_1_on_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hypothesis: `aelf doctor --replay` exits 1 when mismatched > 0.
    Falsifiable by exit code 0."""
    import io
    from aelfrice.cli import main

    db = str(tmp_path / "brain.db")
    monkeypatch.setenv("AELFRICE_DB", db)

    s = MemoryStore(db)
    bid = _ingest(s, _FACTUAL_SENTENCE)
    s._conn.execute(
        "UPDATE beliefs SET content_hash = 'CORRUPTED' WHERE id = ?", (bid,)
    )
    s._conn.commit()
    s.close()

    out = io.StringIO()
    rc = main(["doctor", "--replay"], out=out)
    assert rc == 1


def test_cli_doctor_replay_max_drift_exit_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hypothesis: `aelf doctor --replay --max-drift 5` exits 0 when
    mismatched + derived_orphan <= 5.
    Falsifiable by exit code != 0."""
    import io
    from aelfrice.cli import main

    db = str(tmp_path / "brain.db")
    monkeypatch.setenv("AELFRICE_DB", db)

    s = MemoryStore(db)
    # Create 3 mismatches.
    for i in range(3):
        raw = f"The server process {i} listens on port {8000 + i} by default."
        inp = DerivationInput(
            raw_text=raw,
            source_kind=INGEST_SOURCE_FILESYSTEM,
            source_path="test:source",
            ts=_TS,
        )
        out_d = derive(inp)
        if out_d.belief is None:
            continue
        b = out_d.belief
        s.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM,
            source_path="test:source",
            raw_text=raw,
            derived_belief_ids=[b.id],
            ts=_TS,
        )
        s.insert_belief(b)
        s._conn.execute(
            "UPDATE beliefs SET content_hash = 'BAD' || id WHERE id = ?", (b.id,)
        )
    s._conn.commit()
    s.close()

    out = io.StringIO()
    rc = main(["doctor", "--replay", "--max-drift", "5"], out=out)
    assert rc == 0


def test_cli_doctor_replay_output_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hypothesis: `aelf doctor --replay` prints one summary line per bucket
    and a drift section when drift > 0.
    Falsifiable by missing expected output lines."""
    import io
    from aelfrice.cli import main

    db = str(tmp_path / "brain.db")
    monkeypatch.setenv("AELFRICE_DB", db)

    s = MemoryStore(db)
    bid = _ingest(s, _FACTUAL_SENTENCE)
    s._conn.execute(
        "UPDATE beliefs SET content_hash = 'CORRUPTED' WHERE id = ?", (bid,)
    )
    s._conn.commit()
    s.close()

    buf = io.StringIO()
    main(["doctor", "--replay"], out=buf)
    text = buf.getvalue()

    # Summary lines present.
    assert "matched:" in text
    assert "mismatched:" in text
    assert "derived_orphan:" in text
    assert "canonical_orphan:" in text
    # Drift section present.
    assert "drift" in text.lower()


def test_cli_doctor_replay_negative_max_drift_clamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hypothesis: a negative `--max-drift` value is clamped to zero, so a
    clean store still exits 0. Falsifiable by exit code != 0."""
    import io
    from aelfrice.cli import main

    db = str(tmp_path / "brain.db")
    monkeypatch.setenv("AELFRICE_DB", db)

    s = MemoryStore(db)
    _ingest(s, _FACTUAL_SENTENCE)
    s.close()

    out = io.StringIO()
    rc = main(["doctor", "--replay", "--max-drift", "-1"], out=out)
    assert rc == 0


def test_cli_doctor_replay_drift_examples_zero_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hypothesis: explicit `--drift-examples 0` produces a report with no
    example entries even when drift is present. Falsifiable by any
    `[bucket]` sub-block appearing in the output."""
    import io
    from aelfrice.cli import main

    db = str(tmp_path / "brain.db")
    monkeypatch.setenv("AELFRICE_DB", db)

    s = MemoryStore(db)
    bid = _ingest(s, _FACTUAL_SENTENCE)
    s._conn.execute(
        "UPDATE beliefs SET content_hash = 'CORRUPTED' WHERE id = ?", (bid,)
    )
    s._conn.commit()
    s.close()

    buf = io.StringIO()
    rc = main(
        ["doctor", "--replay", "--drift-examples", "0", "--max-drift", "999"],
        out=buf,
    )
    text = buf.getvalue()

    # Drift exists but no per-bucket example sub-block was emitted.
    assert rc == 0
    assert "mismatched:" in text
    assert "[mismatched]" not in text
    assert "[derived_orphan]" not in text
    assert "[canonical_orphan]" not in text


def test_cli_doctor_replay_canonical_orphan_only_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hypothesis: a belief whose only ingest_log row is legacy_unknown is a
    canonical_orphan, which is informational-only; `aelf doctor --replay`
    must exit 0. Falsifiable by exit code != 0."""
    import io
    from aelfrice.cli import main

    db = str(tmp_path / "brain.db")
    monkeypatch.setenv("AELFRICE_DB", db)

    s = MemoryStore(db)
    b = _make_belief(_FACTUAL_SENTENCE)
    s.insert_belief(b)
    s.record_ingest(
        source_kind=INGEST_SOURCE_LEGACY_UNKNOWN,
        source_path=None,
        raw_text=_FACTUAL_SENTENCE,
        derived_belief_ids=[b.id],
        ts=_TS,
    )
    s.close()

    out = io.StringIO()
    rc = main(["doctor", "--replay"], out=out)
    assert rc == 0
