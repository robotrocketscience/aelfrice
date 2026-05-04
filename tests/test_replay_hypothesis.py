"""Property-based replay-equality probe for #403 (replay-soak gate, B).

Generates random sequences of (raw_text, source_kind, source_path) tuples via
hypothesis, ingests each through the live `derive() -> record_ingest +
insert_belief` path, then asserts `replay_full_equality(store)` reports
`mismatched + derived_orphan == 0`.

Catches nondeterminism in `derive()` under arbitrary public-domain inputs:
order effects, source_path leakage between rows, hidden mutable state, or
any future change that lets two derive() calls on the same raw_text
produce different (id, content_hash, type, origin) tuples.

The seed pool is small and deliberately product-domain (no `~/.claude/`-derived
text); hypothesis explores subsets, ordering, multiplicity, and source_kind /
source_path combinations. Bounded scope per spec — does not change
`replay_full_equality()` itself.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aelfrice.derivation import DerivationInput, derive
from aelfrice.models import (
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS,
    INGEST_SOURCE_FILESYSTEM,
    INGEST_SOURCE_GIT,
    INGEST_SOURCE_MCP_REMEMBER,
    INGEST_SOURCE_PYTHON_AST,
)
from aelfrice.replay import replay_full_equality
from aelfrice.store import MemoryStore

_TS = "2026-01-01T00:00:00+00:00"

# Public-domain / product-domain seed sentences. Each is shaped to
# reliably produce a belief through the current classifier; hypothesis
# composes them into sequences. Adding a sentence that derive() rejects
# is harmless — replay treats persist=False as informational.
_SEED_TEXTS: tuple[str, ...] = (
    "The configuration database is stored at /var/lib/aelfrice/brain.db.",
    "The default listening port for the web server is 8443 on all interfaces.",
    "The retention class for filesystem ingest is short.",
    "Backups run nightly at 02:00 UTC against the canonical store.",
    "FTS5 search powers the L1 retrieval lane in aelfrice v1.x.",
    "Pre-push hook enforces the directory-of-origin boundary.",
    "Python 3.12 is the minimum supported runtime for the package.",
    "The signing key for release tags is stored at ~/.ssh/id_rrs.",
)

_SOURCE_KINDS: tuple[str, ...] = (
    INGEST_SOURCE_FILESYSTEM,
    INGEST_SOURCE_GIT,
    INGEST_SOURCE_PYTHON_AST,
    INGEST_SOURCE_MCP_REMEMBER,
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS,
)


# Each seed text gets a stable (source_kind, source_path) identity tuple.
# `derive()` keys `belief.id` on (raw_text, source_path or source_kind),
# while `content_hash` keys on raw_text only. Mixing the same raw_text
# under two different identity tuples collides on UNIQUE(content_hash)
# at insert and produces derived_orphan rows on replay (because replay
# re-derives per-row using the row's identity). Production handles
# dedup-by-hash on insert; the replay-soak invariant the gate enforces
# is "each raw_text has a stable identity across its lifetime in the log."
# The strategy honors that invariant by binding identity to text.
_IDENTITY_BY_TEXT: dict[str, tuple[str, str | None]] = {
    t: (
        _SOURCE_KINDS[i % len(_SOURCE_KINDS)],
        None if i % 3 == 0 else f"src:{i:02d}",
    )
    for i, t in enumerate(_SEED_TEXTS)
}


@st.composite
def _row(draw: st.DrawFn) -> tuple[str, str, str | None]:
    text = draw(st.sampled_from(_SEED_TEXTS))
    kind, path = _IDENTITY_BY_TEXT[text]
    return text, kind, path


_row_strategy = _row()


@given(rows=st.lists(_row_strategy, min_size=0, max_size=12))
@settings(
    max_examples=40,
    deadline=None,
)
@pytest.mark.timeout(60)
def test_replay_full_equality_no_drift_on_random_ingest(
    rows: list[tuple[str, str, str | None]],
) -> None:
    """Hypothesis: for any sequence of valid (raw_text, source_kind, source_path)
    rows ingested via the production write path, replay_full_equality reports
    `mismatched + derived_orphan == 0`. Falsifiable if any rerun of derive()
    on the same raw_text produces a non-equal belief, or any row produces a
    belief id absent from the canonical store.
    """
    # Per-example store: hypothesis re-runs the body many times; each run
    # needs a clean db so prior-example state does not bleed in.
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "hypothesis_replay.db"
        store = MemoryStore(str(db_path))
        try:
            _run_one(store, rows)
        finally:
            store.close()


def _run_one(
    store: MemoryStore,
    rows: list[tuple[str, str, str | None]],
) -> None:
    for i, (raw_text, source_kind, source_path) in enumerate(rows):
        inp = DerivationInput(
            raw_text=raw_text,
            source_kind=source_kind,
            source_path=source_path,
            ts=_TS,
        )
        out = derive(inp)
        if out.belief is None:
            # Classifier declined to persist; informational only — skip
            # both record_ingest and insert_belief. Replay walks
            # ingest_log rows, so a missing log row is consistent with a
            # missing belief.
            continue
        # The store enforces UNIQUE on (id) and on (content_hash). Two
        # different (raw_text, source_path) inputs can produce the same
        # content_hash with different ids; collapse to the existing
        # canonical row in that case so replay sees a consistent store.
        existing_by_hash = store.get_belief_by_content_hash(out.belief.content_hash)
        belief_id_for_log = (
            existing_by_hash.id if existing_by_hash is not None else out.belief.id
        )
        store.record_ingest(
            source_kind=source_kind,
            source_path=source_path,
            raw_text=raw_text,
            derived_belief_ids=[belief_id_for_log],
            ts=_TS,
            log_id=f"log_{i:03d}_{belief_id_for_log[:8]}",
        )
        if existing_by_hash is None and store.get_belief(out.belief.id) is None:
            store.insert_belief(out.belief)

    report = replay_full_equality(store)
    assert report.implemented is True
    assert report.mismatched == 0, (
        f"replay drift: {report.mismatched} mismatched rows out of "
        f"{report.total_log_rows}"
    )
    assert report.derived_orphan == 0, (
        f"replay orphan: {report.derived_orphan} derived-orphan rows"
    )


def test_inconsistent_identity_for_same_text_produces_drift(tmp_path: Path) -> None:
    """Tripwire: documents the soak-gate corpus invariant.

    If the SAME raw_text is ingested under two different identity tuples
    (source_kind/source_path), insert dedups on UNIQUE(content_hash) but
    replay re-derives the second log row's identity to a fresh belief id
    that is absent from the canonical store, producing a derived_orphan.
    The replay-soak corpus authoring discipline (#403 § A) must enforce
    text → identity stability; this test will start failing the day the
    derivation scheme decouples id from source_path, and the corpus rule
    can be relaxed at that point.
    """
    store = MemoryStore(str(tmp_path / "drift.db"))
    try:
        text = "FTS5 search powers the L1 retrieval lane in aelfrice v1.x."
        for i, (kind, path) in enumerate(
            [(INGEST_SOURCE_FILESYSTEM, "src:a"), (INGEST_SOURCE_FILESYSTEM, "src:b")]
        ):
            inp = DerivationInput(
                raw_text=text, source_kind=kind, source_path=path, ts=_TS,
            )
            out = derive(inp)
            assert out.belief is not None
            existing = store.get_belief_by_content_hash(out.belief.content_hash)
            chosen_id = existing.id if existing is not None else out.belief.id
            store.record_ingest(
                source_kind=kind,
                source_path=path,
                raw_text=text,
                derived_belief_ids=[chosen_id],
                ts=_TS,
                log_id=f"drift_{i}",
            )
            if existing is None:
                store.insert_belief(out.belief)
        report = replay_full_equality(store)
        assert report.derived_orphan >= 1, (
            "expected derived_orphan>=1 when same text has inconsistent identity"
        )
    finally:
        store.close()
