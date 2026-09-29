"""#1636: an inter-turn DERIVED_FROM edge never links a belief to itself.

Since #1364 the inter-turn chain in `ingest_jsonl` anchors on the belief a
turn resolved to, so a user turn that repeats the one before it resolves to
the same head and the writer linked that belief to itself. A self-loop is
not a derivation: BFS and propagation skip it, but it counts in edge
statistics and in anything that reads `DERIVED_FROM` as provenance.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.ingest import ingest_jsonl
from aelfrice.models import EDGE_DERIVED_FROM
from aelfrice.store import MemoryStore

_SESSION = "s-1636"
_A = "The configuration file lives at /etc/aelfrice/conf."
_B = "Astronomers process supernova imagery nightly using clusters."
_C = "Radio telescopes calibrate against known pulsar timings."


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's repo-local live store out of every test."""
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "loop.db"))
    yield s
    s.close()


def _ingest(store: MemoryStore, path: Path, texts: list[str]) -> None:
    with path.open("w") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({
                "role": "user", "text": t, "session_id": _SESSION,
                "ts": f"2026-08-01T00:{i // 60:02d}:{i % 60:02d}Z",
            }) + "\n")
    ingest_jsonl(store, path)


def _derived_from(store: MemoryStore) -> list[tuple[str, str]]:
    return [
        (e.src, e.dst) for e in store.iter_all_edges()
        if e.type == EDGE_DERIVED_FROM
    ]


def _bid(store: MemoryStore, text: str) -> str:
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT id FROM beliefs WHERE content = ?", (text,)
    ).fetchall()
    assert len(rows) == 1, (text, rows)
    return rows[0][0]


@pytest.mark.timeout(30)
def test_adjacent_duplicate_turns_write_no_self_loop(
    store: MemoryStore, tmp_path: Path,
) -> None:
    _ingest(store, tmp_path / "t.jsonl", [_A, _B, _B])
    assert [e for e in _derived_from(store) if e[0] == e[1]] == []


@pytest.mark.timeout(60)
def test_the_issues_fixture_writes_no_self_loop(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """50 duplicate turns in 45 runs: 45 self-loops on main, 0 now."""
    texts: list[str] = []
    runs = 0
    dups = 0
    for i in range(500):
        # Each run of duplicates repeats one distinct sentence; five runs
        # are two duplicates long, so 50 duplicates fall in 45 runs.
        base = f"Distinct observation number {i} about telescope calibration."
        texts.append(base)
        if i % 11 == 0 and runs < 45:
            reps = 2 if runs < 5 else 1
            texts.extend([base] * reps)
            runs += 1
            dups += reps
    assert (runs, dups) == (45, 50)
    _ingest(store, tmp_path / "t.jsonl", texts)
    assert [e for e in _derived_from(store) if e[0] == e[1]] == []


@pytest.mark.timeout(30)
def test_the_chain_steps_over_a_duplicate_turn(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """A, B, B, C links C to B: the pointer still advances on a duplicate."""
    _ingest(store, tmp_path / "t.jsonl", [_A, _B, _B, _C])
    a, b, c = (_bid(store, t) for t in (_A, _B, _C))
    assert sorted(_derived_from(store)) == sorted([(b, a), (c, b)])


@pytest.mark.timeout(30)
def test_the_ingest_count_matches_the_edges_written(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """`edges_inserted` counts writes, not attempts.

    The refused self-loop must not be counted: `insert_edge` reports it
    skipped the write, and the writer counts from that.
    """
    path = tmp_path / "t.jsonl"
    with path.open("w") as f:
        for i, t in enumerate([_A, _B, _B, _C]):
            f.write(json.dumps({"role": "user", "text": t,
                                "session_id": _SESSION,
                                "ts": f"2026-08-01T00:00:0{i}Z"}) + "\n")
    result = ingest_jsonl(store, path)
    assert result.edges_inserted == len(_derived_from(store)) == 2


@pytest.mark.timeout(30)
def test_the_anchor_after_a_skipped_loop_is_the_later_turns_text(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """The pointer advances past a skipped self-loop.

    Two different turns that end on the same sentence resolve to the same
    head, so the edge between them is a self-loop and is skipped. The next
    turn's edge must anchor on the second turn's text, which it can only do
    if the pointer moved to it.
    """
    shared = "Every deploy runs the full regression suite first."
    t1 = f"We shipped the parser yesterday. {shared}"
    t2 = f"The cache layer changed this morning. {shared}"
    _ingest(store, tmp_path / "t.jsonl", [t1, t2, _C])
    c, s_id = _bid(store, _C), _bid(store, shared)
    edges = [e for e in store.iter_all_edges()
             if e.type == EDGE_DERIVED_FROM and e.src == c]
    assert [(e.dst, e.anchor_text) for e in edges] == [(s_id, t2)]


@pytest.mark.timeout(30)
def test_insert_edge_refuses_a_self_loop_from_any_caller(
    store: MemoryStore,
) -> None:
    """Skipped and reported, never raised: hooks call this."""
    from aelfrice.models import Edge

    wrote = store.insert_edge(Edge(src="x" * 16, dst="x" * 16,
                                   type=EDGE_DERIVED_FROM, weight=1.0))
    assert wrote is False
    assert store.get_edge("x" * 16, "x" * 16, EDGE_DERIVED_FROM) is None
    assert store.insert_edge(Edge(src="x" * 16, dst="y" * 16,
                                  type=EDGE_DERIVED_FROM, weight=1.0)) is True


@pytest.mark.timeout(30)
def test_a_belief_paired_with_itself_is_not_a_resolved_contradiction(
    store: MemoryStore,
) -> None:
    """Found by review: a skipped SUPERSEDES self-loop read as created, and a
    legacy CONTRADICTS self-loop was re-resolved, with an audit row, forever.
    """
    from aelfrice.contradiction import (
        auto_resolve_all_contradictions,
        find_unresolved_contradictions,
        resolve_contradiction,
    )
    from aelfrice.models import EDGE_CONTRADICTS, BELIEF_FACTUAL, LOCK_NONE, Belief

    store.insert_belief(Belief(
        id="z" * 16, content="a lone belief", content_hash="hz",
        alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
        locked_at=None, created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    ))
    assert resolve_contradiction(store, "z" * 16, "z" * 16).supersedes_created is False
    # A legacy self-loop, written by raw SQL as an older version would have.
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "INSERT INTO edges (src, dst, type, weight) VALUES (?, ?, ?, 1.0)",
        ("z" * 16, "z" * 16, EDGE_CONTRADICTS),
    )
    assert find_unresolved_contradictions(store) == []
    assert auto_resolve_all_contradictions(store) == []


@pytest.mark.timeout(30)
def test_migrate_counts_only_the_edges_it_wrote(tmp_path: Path) -> None:
    """Found by review: a legacy self-loop was counted as inserted.

    `migrate` copies a legacy store's edges; the target now refuses a
    self-loop, and the report must not say it was copied.
    """
    from aelfrice.migrate import migrate
    from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief

    src_db = tmp_path / "legacy" / "memory.db"
    src_db.parent.mkdir()
    legacy = MemoryStore(str(src_db))
    try:
        for bid in ("a" * 16, "b" * 16):
            legacy.insert_belief(Belief(
                id=bid, content=f"belief {bid}", content_hash=f"h{bid}",
                alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE, locked_at=None,
                created_at="2026-06-18T00:00:00Z", last_retrieved_at=None,
            ))
        conn = legacy._conn  # pyright: ignore[reportPrivateUsage]
        conn.execute(
            "INSERT INTO edges (src, dst, type, weight) VALUES (?, ?, ?, 1.0)",
            ("a" * 16, "a" * 16, EDGE_DERIVED_FROM))
        conn.execute(
            "INSERT INTO edges (src, dst, type, weight) VALUES (?, ?, ?, 1.0)",
            ("b" * 16, "a" * 16, EDGE_DERIVED_FROM))
        conn.commit()
    finally:
        legacy.close()
    tgt_db = tmp_path / "target.db"
    report = migrate(legacy_path=src_db, target_path=tgt_db,
                     project_root=Path("/"), apply=True, copy_all=True)
    target = MemoryStore(str(tgt_db))
    try:
        written = len(list(target.iter_all_edges()))
    finally:
        target.close()
    assert report.counts.legacy_edges == 2
    assert report.counts.inserted_edges == written == 1
