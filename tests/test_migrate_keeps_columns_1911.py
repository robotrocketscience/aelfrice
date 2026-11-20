"""`aelf migrate --apply` copies every column of each belief (#1911).

Before the fix, migrate built each target `Belief` from 13 named fields,
so `valid_to`, `lock_tier`, `lock_expires_at`, `session_id`,
`retention_class`, and `last_confirmed_at` took their defaults, the
`belief_corroborations` rows behind `corroboration_count` were not
copied, and edges lost `anchor_text`. A retired belief was active in the
target, and a timed reference lock was a permanent frozen lock there.

Each test seeds a source store, runs `migrate(..., apply=True,
copy_all=True)` into a fresh target in `tmp_path`, and reads the target
back. The conftest session fixture pins HOME, AELFRICE_DB, and
AELFRICE_USER_DB to a temp dir.
"""
from __future__ import annotations

import dataclasses
import io
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aelfrice.cli import _print_migrate_report
from aelfrice.migrate import migrate
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_TYPES,
    CORROBORATION_SOURCE_CLI_REMEMBER,
    CORROBORATION_SOURCE_TYPES,
    CORROBORATION_SPEAKERS,
    DEFAULT_LOCK_TIER,
    EDGE_SUPPORTS,
    LOCK_LEVELS,
    LOCK_NONE,
    LOCK_TIER_REFERENCE,
    LOCK_TIERS,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_UNKNOWN,
    ORIGINS,
    RETENTION_CLASSES,
    RETENTION_FACT,
    RETENTION_UNKNOWN,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore

_FAR_FUTURE = "2099-01-01T00:00:00+00:00"


def _belief(bid: str = "b1", **overrides: Any) -> Belief:
    base = Belief(
        id=bid,
        content=f"content of {bid}",
        content_hash=f"h-{bid}",
        alpha=2.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-01-01T00:00:00+00:00",
        last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    )
    return dataclasses.replace(base, **overrides)


def _migrate_one(tmp_path: Path, belief: Belief) -> sqlite3.Row:
    """Migrate `belief` alone and return its raw target row."""
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    s = MemoryStore(str(src))
    try:
        s.insert_belief(belief)
    finally:
        s.close()
    migrate(legacy_path=src, target_path=dst, project_root=tmp_path,
            apply=True, copy_all=True)
    conn = sqlite3.connect(dst)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM beliefs WHERE id = ?", (belief.id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return row


# --- One test per column the old copy dropped -------------------------------


def test_valid_to_survives_so_a_retired_belief_stays_retired(
    tmp_path: Path,
) -> None:
    row = _migrate_one(tmp_path, _belief(valid_to="2026-03-01T00:00:00+00:00"))
    assert row["valid_to"] == "2026-03-01T00:00:00+00:00"


def test_a_retired_belief_has_no_fts_row_in_the_target(tmp_path: Path) -> None:
    """The source's retire pruned the FTS row; the copy matches it."""
    _migrate_one(tmp_path, _belief(valid_to="2026-03-01T00:00:00+00:00"))
    conn = sqlite3.connect(tmp_path / "dst.db")
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM beliefs_fts WHERE id = 'b1'",
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 0


def test_lock_tier_survives_so_a_reference_lock_stays_reference(
    tmp_path: Path,
) -> None:
    row = _migrate_one(tmp_path, _belief(
        lock_level=LOCK_USER, locked_at="2026-01-02T00:00:00+00:00",
        lock_tier=LOCK_TIER_REFERENCE,
    ))
    assert row["lock_tier"] == LOCK_TIER_REFERENCE


def test_lock_expires_at_survives_so_a_timed_lock_stays_timed(
    tmp_path: Path,
) -> None:
    row = _migrate_one(tmp_path, _belief(
        lock_level=LOCK_USER, locked_at="2026-01-02T00:00:00+00:00",
        lock_expires_at=_FAR_FUTURE,
    ))
    assert row["lock_expires_at"] == _FAR_FUTURE
    assert row["lock_level"] == LOCK_USER


def test_session_id_survives(tmp_path: Path) -> None:
    row = _migrate_one(tmp_path, _belief(session_id="sess-1911"))
    assert row["session_id"] == "sess-1911"


def test_retention_class_survives(tmp_path: Path) -> None:
    row = _migrate_one(tmp_path, _belief(retention_class=RETENTION_FACT))
    assert row["retention_class"] == RETENTION_FACT


def test_last_confirmed_at_survives(tmp_path: Path) -> None:
    row = _migrate_one(tmp_path, _belief(
        last_confirmed_at="2026-02-01T00:00:00+00:00",
    ))
    assert row["last_confirmed_at"] == "2026-02-01T00:00:00+00:00"


def test_corroboration_count_survives_with_its_rows(tmp_path: Path) -> None:
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    s = MemoryStore(str(src))
    try:
        s.insert_belief(_belief())
        for i in range(2):
            s.record_corroboration(
                "b1", source_type=CORROBORATION_SOURCE_CLI_REMEMBER,
                session_id=f"s-{i}", ts=f"2026-01-0{i + 2}T00:00:00+00:00",
            )
    finally:
        s.close()
    migrate(legacy_path=src, target_path=dst, project_root=tmp_path,
            apply=True, copy_all=True)
    t = MemoryStore(str(dst))
    try:
        got = t.get_belief("b1")
        assert got is not None
        assert got.corroboration_count == 2
    finally:
        t.close()
    conn = sqlite3.connect(dst)
    try:
        rows = conn.execute(
            "SELECT session_id, ingested_at FROM belief_corroborations "
            "ORDER BY id",
        ).fetchall()
    finally:
        conn.close()
    assert rows == [
        ("s-0", "2026-01-02T00:00:00+00:00"),
        ("s-1", "2026-01-03T00:00:00+00:00"),
    ]


def test_a_belief_the_target_holds_keeps_the_targets_rows(
    tmp_path: Path,
) -> None:
    """Migrating into a store that already holds b1 adds none of the
    source's rows to it, so its corroboration count does not grow."""
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    for path, sessions in ((src, ("s-0", "s-1")), (dst, ("t-0",))):
        s = MemoryStore(str(path))
        try:
            s.insert_belief(_belief())
            for sid in sessions:
                s.record_corroboration(
                    "b1", source_type=CORROBORATION_SOURCE_CLI_REMEMBER,
                    session_id=sid, ts="2026-01-02T00:00:00+00:00",
                )
        finally:
            s.close()
    report = migrate(legacy_path=src, target_path=dst, project_root=tmp_path,
                     apply=True, copy_all=True)
    assert report.counts.skipped_existing_beliefs == 1
    assert report.counts.copied_corroborations == 0
    t = MemoryStore(str(dst))
    try:
        rows = t.list_corroborations("b1")
    finally:
        t.close()
    assert [r[2] for r in rows] == ["t-0"]


def test_edge_anchor_text_survives(tmp_path: Path) -> None:
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    s = MemoryStore(str(src))
    try:
        s.insert_belief(_belief("b1"))
        s.insert_belief(_belief("b2"))
        s.insert_edge(Edge(src="b1", dst="b2", type=EDGE_SUPPORTS,
                           weight=0.5, anchor_text="the anchor"))
    finally:
        s.close()
    migrate(legacy_path=src, target_path=dst, project_root=tmp_path,
            apply=True, copy_all=True)
    t = MemoryStore(str(dst))
    try:
        e = t.get_edge("b1", "b2", EDGE_SUPPORTS)
    finally:
        t.close()
    assert e is not None
    assert e.anchor_text == "the anchor"


# --- The round trip covers the whole schema ---------------------------------


def test_every_beliefs_column_is_a_belief_field(tmp_path: Path) -> None:
    """A column with no Belief field would be dropped by the copy."""
    s = MemoryStore(str(tmp_path / "fresh.db"))
    s.close()
    conn = sqlite3.connect(tmp_path / "fresh.db")
    try:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(beliefs)")}
    finally:
        conn.close()
    fields = {f.name for f in dataclasses.fields(Belief)}
    assert columns - fields == set()


_ISO = st.datetimes(
    min_value=datetime(2020, 1, 1), max_value=datetime(2090, 1, 1),
).map(lambda d: d.replace(tzinfo=timezone.utc).isoformat())
_FUTURE_ISO = st.integers(min_value=1, max_value=3650).map(
    lambda days: (datetime(2099, 1, 1, tzinfo=timezone.utc)
                  + timedelta(days=days)).isoformat()
)
_TEXT = st.text(
    alphabet=st.characters(min_codepoint=0x20, max_codepoint=0x7E),
    min_size=1, max_size=20,
)
_SCOPES = st.one_of(
    st.sampled_from(["project", "global"]),
    st.from_regex(r"\Ashared:[a-z0-9_-]{1,8}\Z"),
)
# (source_type, speaker, ingested_at, session_id, source_path_hash). Two
# session ids let rows share a session, so rows that differ only in
# source_path_hash stay distinct under the #1020 unique key.
_CorroborationDraw = tuple[str, str | None, str, str | None, str | None]
_CORROBORATION = st.tuples(
    st.sampled_from(sorted(CORROBORATION_SOURCE_TYPES)),
    st.none() | st.sampled_from(sorted(CORROBORATION_SPEAKERS)),
    _ISO,
    st.none() | st.sampled_from(["s0", "s1"]),
    st.none() | _TEXT,
)


@st.composite
def _beliefs(draw: st.DrawFn) -> list[tuple[Belief, list[_CorroborationDraw]]]:
    n = draw(st.integers(min_value=1, max_value=4))
    out: list[tuple[Belief, list[_CorroborationDraw]]] = []
    for i in range(n):
        lock_level = draw(st.sampled_from(sorted(LOCK_LEVELS)))
        kwargs: dict[str, Any] = {
            "id": f"b{i}",
            "content": f"{i} {draw(_TEXT)}",
            "content_hash": f"h{i}",
            "alpha": draw(st.floats(min_value=0.01, max_value=1e6)),
            "beta": draw(st.floats(min_value=0.01, max_value=1e6)),
            "type": draw(st.sampled_from(sorted(BELIEF_TYPES))),
            "lock_level": lock_level,
            "locked_at": draw(st.none() | _ISO),
            "created_at": draw(_ISO),
            "last_retrieved_at": draw(st.none() | _ISO),
            "session_id": draw(st.none() | _TEXT),
            # ORIGIN_UNKNOWN is rewritten on purpose (#1135); its mapping
            # has its own tests in test_migrate.py.
            "origin": draw(st.sampled_from(sorted(ORIGINS - {ORIGIN_UNKNOWN}))),
            "corroboration_count": 0,
            "retention_class": draw(st.sampled_from(sorted(RETENTION_CLASSES))),
            "valid_to": draw(st.none() | _ISO),
            "scope": draw(_SCOPES),
            # A non-empty value is never restamped (#970).
            "project_context": draw(st.just("") | _TEXT),
            "last_confirmed_at": draw(st.none() | _ISO),
            "lock_tier": draw(st.sampled_from(sorted(LOCK_TIERS))),
            # A past expiry is swept to unlocked when a store opens, so
            # only a future one, or none, round-trips unchanged.
            "lock_expires_at": draw(st.none() | _FUTURE_ISO),
        }
        assert set(kwargs) == {f.name for f in dataclasses.fields(Belief)}
        corroborations = draw(st.lists(_CORROBORATION, max_size=3))
        out.append((Belief(**kwargs), corroborations))
    return out


def _snapshot(path: Path) -> tuple[dict[str, dict[str, Any]], list[tuple[Any, ...]]]:
    """Every belief as `get_belief` returns it, plus its corroboration rows."""
    conn = sqlite3.connect(path)
    try:
        ids = [r[0] for r in conn.execute("SELECT id FROM beliefs ORDER BY id")]
        corroborations = conn.execute(
            "SELECT belief_id, ingested_at, source_type, session_id, "
            "source_path_hash, speaker FROM belief_corroborations "
            "ORDER BY belief_id, ingested_at, session_id, source_path_hash, "
            "source_type, speaker",
        ).fetchall()
    finally:
        conn.close()
    store = MemoryStore(str(path))
    try:
        beliefs: dict[str, dict[str, Any]] = {}
        for bid in ids:
            b = store.get_belief(bid, include_retired=True)
            assert b is not None
            beliefs[bid] = dataclasses.asdict(b)
    finally:
        store.close()
    return beliefs, corroborations


@pytest.mark.timeout(120)
@settings(max_examples=25, deadline=None, derandomize=True, database=None)
@given(rows=_beliefs())
def test_migrate_round_trips_every_belief_field(
    rows: list[tuple[Belief, list[_CorroborationDraw]]],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    d = tmp_path_factory.mktemp("roundtrip")
    src, dst = d / "src.db", d / "dst.db"
    s = MemoryStore(str(src))
    try:
        for b, corroborations in rows:
            s.insert_belief(dataclasses.replace(b, valid_to=None))
            if b.valid_to is not None:
                s.soft_delete_belief(b.id, ts=b.valid_to)
            for source_type, speaker, ts, session_id, path_hash in corroborations:
                s.record_corroboration(
                    b.id, source_type=source_type, session_id=session_id,
                    source_path_hash=path_hash, ts=ts, speaker=speaker,
                )
    finally:
        s.close()
    before = _snapshot(src)
    migrate(legacy_path=src, target_path=dst, project_root=d,
            apply=True, copy_all=True)
    assert _snapshot(dst) == before


# --- Older source schemas still migrate -------------------------------------

_V1_0_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE beliefs (
        id TEXT PRIMARY KEY, content TEXT NOT NULL,
        content_hash TEXT NOT NULL, alpha REAL NOT NULL, beta REAL NOT NULL,
        type TEXT NOT NULL, lock_level TEXT NOT NULL, locked_at TEXT,
        demotion_pressure INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL, last_retrieved_at TEXT
    )
    """,
    """
    CREATE TABLE edges (
        src TEXT NOT NULL, dst TEXT NOT NULL, type TEXT NOT NULL,
        weight REAL NOT NULL, PRIMARY KEY (src, dst, type)
    )
    """,
)


def _seed_v1_0(path: Path, *, corroborations_without_speaker: bool) -> None:
    conn = sqlite3.connect(path)
    try:
        for stmt in _V1_0_SCHEMA:
            conn.execute(stmt)
        for bid in ("old1", "old2"):
            conn.execute(
                "INSERT INTO beliefs (id, content, content_hash, alpha, beta, "
                "type, lock_level, locked_at, created_at, last_retrieved_at) "
                "VALUES (?, ?, ?, 1.0, 1.0, 'factual', 'none', NULL, "
                "'2025-01-01T00:00:00+00:00', NULL)",
                (bid, f"text {bid}", f"h-{bid}"),
            )
        conn.execute(
            "INSERT INTO edges VALUES ('old1', 'old2', 'CITES', 1.0)",
        )
        if corroborations_without_speaker:
            # The v1.5.0 table, before the #1650 speaker column.
            conn.execute(
                "CREATE TABLE belief_corroborations ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, belief_id TEXT NOT NULL, "
                "ingested_at TEXT NOT NULL, source_type TEXT NOT NULL, "
                "session_id TEXT, source_path_hash TEXT)"
            )
            conn.execute(
                "INSERT INTO belief_corroborations (belief_id, ingested_at, "
                "source_type, session_id, source_path_hash) VALUES "
                "('old1', '2025-02-01T00:00:00+00:00', 'cli_remember', 's', NULL)"
            )
        conn.commit()
    finally:
        conn.close()


def test_a_v1_0_source_migrates_with_defaults(tmp_path: Path) -> None:
    """No session_id, origin, scope, corroboration table or anchor_text."""
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    _seed_v1_0(src, corroborations_without_speaker=False)
    report = migrate(legacy_path=src, target_path=dst, project_root=tmp_path,
                     apply=True, copy_all=True)
    assert report.counts.inserted_beliefs == 2
    assert report.counts.inserted_edges == 1
    t = MemoryStore(str(dst))
    try:
        b = t.get_belief("old1")
        e = t.get_edge("old1", "old2", "CITES")
    finally:
        t.close()
    assert b is not None and e is not None
    assert (b.session_id, b.valid_to, b.lock_tier, b.retention_class,
            b.corroboration_count) == (None, None, "frozen", "unknown", 0)
    assert e.anchor_text is None


def test_a_source_without_the_speaker_column_copies_its_corroborations(
    tmp_path: Path,
) -> None:
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    _seed_v1_0(src, corroborations_without_speaker=True)
    migrate(legacy_path=src, target_path=dst, project_root=tmp_path,
            apply=True, copy_all=True)
    t = MemoryStore(str(dst))
    try:
        b = t.get_belief("old1")
    finally:
        t.close()
    assert b is not None
    assert b.corroboration_count == 1


# --- A run that stops partway can be rerun ----------------------------------


class _Stop(Exception):
    """Raised by a patched store method to stop a migrate partway."""


def _fail_once(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make `MemoryStore.<name>` raise `_Stop` on its first call only."""
    real = getattr(MemoryStore, name)
    calls = {"n": 0}

    def wrapper(self: MemoryStore, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Stop(name)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(MemoryStore, name, wrapper)


def _seed_retired_with_rows(path: Path) -> None:
    s = MemoryStore(str(path))
    try:
        s.insert_belief(_belief("r1"))
        for i in range(2):
            s.record_corroboration(
                "r1", source_type=CORROBORATION_SOURCE_CLI_REMEMBER,
                session_id=f"s-{i}", ts=f"2026-01-0{i + 2}T00:00:00+00:00",
            )
        s.soft_delete_belief("r1", ts="2026-03-01T00:00:00+00:00")
    finally:
        s.close()


@pytest.mark.parametrize("fails", ["soft_delete_belief", "record_corroboration"])
def test_a_rerun_after_a_stop_copies_the_belief_in_full(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: str,
) -> None:
    """A stop after the insert leaves no belief behind in the target, so
    the rerun copies it retired and with its corroboration rows."""
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    _seed_retired_with_rows(src)
    with monkeypatch.context() as m:
        _fail_once(m, fails)
        with pytest.raises(_Stop):
            migrate(legacy_path=src, target_path=dst, project_root=tmp_path,
                    apply=True, copy_all=True)
    report = migrate(legacy_path=src, target_path=dst, project_root=tmp_path,
                     apply=True, copy_all=True)
    assert report.counts.inserted_beliefs == 1
    conn = sqlite3.connect(dst)
    try:
        valid_to = conn.execute(
            "SELECT valid_to FROM beliefs WHERE id = 'r1'",
        ).fetchone()[0]
        fts = conn.execute(
            "SELECT COUNT(*) FROM beliefs_fts WHERE id = 'r1'",
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT COUNT(*) FROM belief_corroborations WHERE belief_id = 'r1'",
        ).fetchone()[0]
    finally:
        conn.close()
    assert (valid_to, fts, rows) == ("2026-03-01T00:00:00+00:00", 0, 2)


# --- Rows this version does not know are skipped ----------------------------


def _seed_with_unknown_rows(path: Path) -> None:
    """b1 with one known row and two rows `record_corroboration` refuses."""
    s = MemoryStore(str(path))
    try:
        s.insert_belief(_belief("b1"))
        s.insert_belief(_belief("b2"))
        s.insert_edge(Edge(src="b1", dst="b2", type=EDGE_SUPPORTS, weight=0.5))
        s.record_corroboration(
            "b1", source_type=CORROBORATION_SOURCE_CLI_REMEMBER,
            session_id="s-known", ts="2026-01-02T00:00:00+00:00",
        )
    finally:
        s.close()
    conn = sqlite3.connect(path)
    try:
        conn.executemany(
            "INSERT INTO belief_corroborations (belief_id, ingested_at, "
            "source_type, session_id, source_path_hash, speaker) "
            "VALUES ('b1', '2026-01-03T00:00:00+00:00', ?, ?, NULL, ?)",
            [("future_ingest", "s-type", None),
             (CORROBORATION_SOURCE_CLI_REMEMBER, "s-speaker", "future_speaker")],
        )
        conn.commit()
    finally:
        conn.close()


def test_an_unknown_corroboration_row_is_skipped_and_counted(
    tmp_path: Path,
) -> None:
    """A newer source type or speaker does not stop the migrate (#1911)."""
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    _seed_with_unknown_rows(src)
    report = migrate(legacy_path=src, target_path=dst, project_root=tmp_path,
                     apply=True, copy_all=True)
    c = report.counts
    assert (c.inserted_beliefs, c.inserted_edges) == (2, 1)
    assert (c.copied_corroborations, c.skipped_unknown_corroborations) == (1, 2)
    t = MemoryStore(str(dst))
    try:
        b = t.get_belief("b1")
    finally:
        t.close()
    assert b is not None
    assert b.corroboration_count == 1


def test_a_dry_run_counts_the_corroboration_rows_it_would_copy(
    tmp_path: Path,
) -> None:
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    _seed_with_unknown_rows(src)
    report = migrate(legacy_path=src, target_path=dst, project_root=tmp_path,
                     apply=False, copy_all=True)
    c = report.counts
    assert (c.copied_corroborations, c.skipped_unknown_corroborations) == (1, 2)
    assert not dst.exists()


# --- Belief values this version does not know are normalized ----------------


def test_an_unknown_retention_class_or_lock_tier_is_normalized_and_counted(
    tmp_path: Path,
) -> None:
    """A newer retention class or lock tier does not stop the migrate.

    The target gets `unknown` and the default tier for those values, as
    before #1911, and the later beliefs and edges still copy.
    """
    src, dst = tmp_path / "src.db", tmp_path / "dst.db"
    s = MemoryStore(str(src))
    try:
        for bid in ("b1", "b2", "b3"):
            s.insert_belief(_belief(bid))
        s.insert_edge(Edge(src="b1", dst="b3", type=EDGE_SUPPORTS, weight=0.5))
    finally:
        s.close()
    conn = sqlite3.connect(src)
    try:
        conn.execute("PRAGMA ignore_check_constraints=ON")
        conn.execute(
            "UPDATE beliefs SET retention_class = 'archival' WHERE id = 'b2'",
        )
        conn.execute("UPDATE beliefs SET lock_tier = 'pinned' WHERE id = 'b3'")
        conn.commit()
    finally:
        conn.close()
    dry = migrate(legacy_path=src, target_path=dst, project_root=tmp_path,
                  apply=False, copy_all=True)
    assert dry.counts.normalized_unknown_values == 2
    report = migrate(legacy_path=src, target_path=dst, project_root=tmp_path,
                     apply=True, copy_all=True)
    conn = sqlite3.connect(dst)
    try:
        ids = [r[0] for r in conn.execute("SELECT id FROM beliefs ORDER BY id")]
        edges = conn.execute("SELECT src, dst FROM edges").fetchall()
        b2 = conn.execute(
            "SELECT retention_class FROM beliefs WHERE id = 'b2'",
        ).fetchone()[0]
        b3 = conn.execute(
            "SELECT lock_tier FROM beliefs WHERE id = 'b3'",
        ).fetchone()[0]
    finally:
        conn.close()
    assert ids == ["b1", "b2", "b3"]
    assert edges == [("b1", "b3")]
    assert (b2, b3) == (RETENTION_UNKNOWN, DEFAULT_LOCK_TIER)
    assert report.counts.normalized_unknown_values == 2
    out = io.StringIO()
    _print_migrate_report(report, out=out)
    assert "  normalized unknown values: 2\n" in out.getvalue()
