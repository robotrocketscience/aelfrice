"""Integration tests for the post-#264 scan_repo (regex path) →
derivation_worker call shape.

Each test states a falsifiable hypothesis about the new contract:

    scan_repo (with no llm_router) writes one log row per
    noise-survived candidate and invokes run_worker once at
    end-of-scan. The worker derives + dedup-or-corroborates +
    stamps every row.

These tests anchor the new invariants for the scanner regex path. The
LLM-classify path is intentionally untouched in this PR — its
alpha/beta/origin/audit_source are not yet representable in
`raw_meta` for the worker to reconstruct.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.replay import replay_full_equality
from aelfrice.scanner import scan_repo
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "scanner-via-worker.db"))
    yield s
    s.close()


def _seed_repo(root: Path) -> None:
    """Drop a tiny doc-only tree the filesystem extractor will see.

    Two paragraphs in a single .md file is enough to produce multiple
    candidates without depending on git or AST extraction.
    """
    (root / "README.md").write_text(
        "The configuration file lives at /etc/aelfrice/conf.\n"
        "\n"
        "The default port is 8080 for the dashboard.\n"
        "\n"
        "Aelfrice stores beliefs in a SQLite database.\n",
        encoding="utf-8",
    )


def test_every_log_row_is_stamped_after_scan(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: after scan_repo returns, no row in `ingest_log`
    remains unstamped — the worker ran end-of-scan. Falsifiable by any
    row whose `derived_belief_ids IS NULL` after the call (worker did
    not run, or scanner bypassed the worker for some candidates)."""
    _seed_repo(tmp_path)
    scan_repo(store, tmp_path)
    unstamped = store.list_unstamped_ingest_log()
    assert unstamped == [], f"unstamped rows: {unstamped}"


def test_log_row_carries_call_site_metadata(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: the regex path stamps
    `raw_meta.call_site = filesystem_ingest` so the worker resolves
    the corroboration source unambiguously even though the scanner
    shares `source_kind=filesystem` with other entry points.
    Falsifiable if any scanner-written row is missing the key or
    carries a different call_site."""
    _seed_repo(tmp_path)
    scan_repo(store, tmp_path)
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT raw_meta FROM ingest_log"
    ).fetchall()
    assert rows
    for row in rows:
        meta = json.loads(row["raw_meta"]) if row["raw_meta"] else None
        assert isinstance(meta, dict)
        assert meta.get("call_site") == "filesystem_ingest"


def test_replay_full_equality_passes_after_scan(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis (CI gate): after a scan_repo run, the full-equality
    replay probe reports zero drift. Tripwire for the regex path
    bypassing the worker (`canonical_orphan`) or the worker producing
    a different canonical belief than direct derive (`mismatched`).
    Falsifiable by any non-zero drift counter."""
    _seed_repo(tmp_path)
    scan_repo(store, tmp_path)
    # Second scan exercises the corroboration path inside the worker.
    scan_repo(store, tmp_path)
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


def test_scan_repo_idempotent_after_worker_path(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Hypothesis: re-scanning the same tree under the worker path is
    idempotent on the canonical belief set even though it adds new log
    rows (one per candidate, per scan). `ScanResult.inserted` is 0 on
    the second run; the second scan's candidates all classify as
    skipped_existing. Falsifiable by a duplicate belief OR by a
    non-zero `inserted` on the second scan."""
    _seed_repo(tmp_path)
    first = scan_repo(store, tmp_path)
    assert first.inserted >= 1
    beliefs_after_first = store.count_beliefs()

    second = scan_repo(store, tmp_path)
    assert second.inserted == 0
    assert second.skipped_existing >= first.inserted
    assert store.count_beliefs() == beliefs_after_first

    # Two log rows for each persistable candidate (one per scan).
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT COUNT(*) AS n FROM ingest_log",
    ).fetchone()
    assert rows["n"] >= 2 * first.inserted
