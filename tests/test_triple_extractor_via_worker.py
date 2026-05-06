"""Integration tests for the post-#264 ingest_triples → derivation_worker
call shape.

`ingest_triples` is the commit-ingest terminus: triples extracted from
commit messages get materialized as belief pairs + edges. Under slice 2
the entry point appends one unstamped log row per unique phrase (with
`raw_meta.call_site = commit_ingest`), invokes `run_worker(store)` once
at end-of-batch, then constructs edges against the worker-stamped
canonical ids.

These tests anchor the *new* invariants. They catch any future
regression that re-introduces an inline `derive()` / `insert_belief()`
call in the entry point.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.replay import replay_full_equality
from aelfrice.store import MemoryStore
from aelfrice.triple_extractor import Triple, ingest_triples


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "triples-via-worker.db"))
    yield s
    s.close()


def _t(subj: str, rel: str, obj: str, anchor: str = "") -> Triple:
    return Triple(subject=subj, relation=rel, object=obj, anchor_text=anchor or f"{subj} {rel} {obj}")


def test_every_log_row_is_stamped_after_ingest_triples(
    store: MemoryStore,
) -> None:
    """Hypothesis: after ingest_triples returns, no log row remains
    unstamped — the worker ran end-of-batch. Falsifiable by any row
    whose `derived_belief_ids IS NULL`."""
    triples = [
        _t("the scheduler", "supports", "the worker"),
        _t("the worker", "implements", "derivation"),
    ]
    ingest_triples(store, triples, session_id="commit-abc")
    assert store.list_unstamped_ingest_log() == []


def test_log_row_carries_call_site_metadata(store: MemoryStore) -> None:
    """Hypothesis: each unique phrase stamps `raw_meta.call_site=commit_ingest`
    so the worker resolves the corroboration source unambiguously.
    Falsifiable if `raw_meta` is missing or carries a different
    call_site."""
    ingest_triples(store, [_t("alpha", "depends_on", "beta")], session_id="c1")
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT raw_meta FROM ingest_log"
    ).fetchall()
    assert rows
    import json
    for row in rows:
        meta = json.loads(row["raw_meta"]) if row["raw_meta"] else None
        assert isinstance(meta, dict)
        assert meta.get("call_site") == "commit_ingest"


def test_unique_phrase_per_log_row(store: MemoryStore) -> None:
    """Hypothesis: convergent triples (same phrase appearing as the
    subject across multiple triples) reuse one log row per phrase.
    Falsifiable if `ingest_log` row count exceeds the unique-phrase
    count for the batch."""
    triples = [
        _t("the scheduler", "supports", "the worker"),
        _t("the scheduler", "supports", "the dashboard"),  # subj reused
    ]
    ingest_triples(store, triples, session_id="c2")
    n_rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT COUNT(*) AS n FROM ingest_log"
    ).fetchone()["n"]
    # 3 unique phrases: "the scheduler", "the worker", "the dashboard".
    assert n_rows == 3


def test_ingest_triples_idempotent_on_re_run(store: MemoryStore) -> None:
    """Hypothesis: re-ingesting the same triple batch is idempotent on
    the canonical belief and edge sets, even though the log appends
    new rows. Falsifiable if a duplicate edge appears OR if
    new_beliefs is non-empty on the second call."""
    triples = [_t("the cache", "supports", "the api")]
    r1 = ingest_triples(store, triples, session_id="c3")
    edges_after_first = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT COUNT(*) AS n FROM edges"
    ).fetchone()["n"]
    r2 = ingest_triples(store, triples, session_id="c3")
    assert r2.new_beliefs == []
    assert r2.new_edges == []
    edges_after_second = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT COUNT(*) AS n FROM edges"
    ).fetchone()["n"]
    assert edges_after_first == edges_after_second
    # And both invocations created log rows (canonical history).
    n_rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT COUNT(*) AS n FROM ingest_log"
    ).fetchone()["n"]
    assert n_rows == 4  # 2 unique phrases × 2 invocations
    assert r1.new_beliefs  # sanity: first call did create beliefs


def test_replay_full_equality_passes_after_triples(store: MemoryStore) -> None:
    """Hypothesis (CI gate): after a representative ingest_triples run,
    the full-equality replay probe reports zero drift. Falsifiable by
    any non-zero drift counter — triples that bypass the worker would
    show up as canonical_orphan immediately."""
    triples = [
        _t("the scheduler", "supports", "the worker"),
        _t("the worker", "implements", "derivation"),
        _t("the api", "depends_on", "the cache"),
    ]
    ingest_triples(store, triples, session_id="c4")
    report = replay_full_equality(store)
    assert report.total_log_rows > 0
    assert report.matched == report.total_log_rows, (
        f"replay drift: matched={report.matched}, "
        f"mismatched={report.mismatched}, "
        f"derived_orphan={report.derived_orphan}, "
        f"canonical_orphan={report.canonical_orphan}, "
        f"examples={report.drift_examples}"
    )
    assert report.mismatched == 0
    assert report.derived_orphan == 0
    assert report.canonical_orphan == 0
