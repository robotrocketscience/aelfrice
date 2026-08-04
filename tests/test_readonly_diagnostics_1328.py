"""#1328: a diagnostic must not mutate the store it measures.

`MemoryStore(path)` is a **write** open. It runs the DDL battery, any
pending migrations, the `schema_meta` seed, `_resolve_local_scope_id`
(which mints and persists an id on a store that has none) and — since
#1314 — `sweep_expired_locks`, which flips expired user locks to unlocked.
Two shipped benchmarks pointed their store argument at the live
`.git/aelfrice/memory.db` in their own usage text and opened it read-write,
so running them changed the corpus they existed to measure.

The arms here are deliberately of two kinds. The behavioural ones prove
`read_only=True` actually prevents the mutation — including the specific
one observed, a lock being swept away. The static one enumerates
`benchmarks/` from the directory rather than from a literal list, because
the failure mode is a *new* benchmark reaching for the convenient call; a
list would pass forever while the directory grew around it.
"""
from __future__ import annotations

import ast
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_USER,
    Belief,
)
from aelfrice.store import MemoryStore

_BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"


def _store_with_an_expired_lock(path: Path) -> str:
    """Build a store holding one time-boxed lock whose window has closed."""
    past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    store = MemoryStore(str(path))
    try:
        store.insert_belief(
            Belief(
                id="expired_lock",
                content="the release key rotates at the end of the quarter",
                content_hash="h_expired_lock",
                alpha=1.0,
                beta=1.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_USER,
                locked_at=past,
                created_at=past,
                last_retrieved_at=None,
                lock_expires_at=past,
            )
        )
    finally:
        store.close()
    return "expired_lock"


class TestReadOnlyOpenDoesNotMutate:
    def test_a_read_only_open_does_not_sweep_an_expired_lock(
        self, tmp_path: Path
    ) -> None:
        """The exact mutation observed on the live store.

        Distinguishing: the sibling test below opens the same store
        read-write and asserts the lock *is* swept, so this cannot pass
        because the fixture failed to arm.
        """
        db = tmp_path / "ro.db"
        bid = _store_with_an_expired_lock(db)

        store = MemoryStore(str(db), read_only=True)
        try:
            assert store.get_belief(bid).lock_level == LOCK_USER
        finally:
            store.close()

        con = sqlite3.connect(str(db))
        try:
            level = con.execute(
                "SELECT lock_level FROM beliefs WHERE id = ?", (bid,)
            ).fetchone()[0]
        finally:
            con.close()
        assert level == LOCK_USER, (
            "a read-only open swept the lock: the write window is not gated"
        )

    def test_a_write_open_does_sweep_it(self, tmp_path: Path) -> None:
        """The control. Without it the arm above passes on a broken
        fixture that never armed an expiring lock in the first place."""
        db = tmp_path / "rw.db"
        bid = _store_with_an_expired_lock(db)
        store = MemoryStore(str(db))
        try:
            assert store.get_belief(bid).lock_level != LOCK_USER
        finally:
            store.close()

    def test_a_read_only_open_leaves_the_file_bytes_alone(
        self, tmp_path: Path
    ) -> None:
        """Stronger than "no logical change": no bytes move at all.

        Migrations, the generation seed and the scope-id mint would each
        show up here even when they change nothing a query can see.
        """
        db = tmp_path / "bytes.db"
        _store_with_an_expired_lock(db)
        # Settle the WAL so the comparison is against a quiescent file.
        con = sqlite3.connect(str(db))
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.close()
        before = db.read_bytes()

        store = MemoryStore(str(db), read_only=True)
        try:
            store.list_beliefs_for_indexing()
        finally:
            store.close()

        assert db.read_bytes() == before

    def test_writes_are_refused_by_the_engine_not_by_convention(
        self, tmp_path: Path
    ) -> None:
        """`mode=ro` is what makes this a guarantee.

        Gating the open-time passes stops the *known* writes. Only the
        engine stops the ones nobody thought of, which is the difference
        between this and the docstring that was there before.
        """
        db = tmp_path / "refuse.db"
        _store_with_an_expired_lock(db)
        store = MemoryStore(str(db), read_only=True)
        try:
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                store._conn.execute(  # noqa: SLF001 - asserting the handle
                    "UPDATE beliefs SET lock_level = 'none'"
                )
        finally:
            store.close()

    def test_a_missing_file_raises_rather_than_being_created(
        self, tmp_path: Path
    ) -> None:
        """A diagnostic pointed at the wrong path should say so.

        The read-write default creates an empty store, which reads as "the
        corpus is empty" rather than as "that is not the corpus".
        """
        with pytest.raises(sqlite3.OperationalError):
            MemoryStore(str(tmp_path / "does-not-exist.db"), read_only=True)


class TestBenchmarksDoNotOpenTheLiveStoreForWrite:
    """Static guard, enumerated from the directory.

    A literal list of the two offending files would pass forever while
    `benchmarks/` grew around it, and growth is the failure mode: three
    separate benchmarks reached for `MemoryStore(path)` independently.
    """

    @staticmethod
    def _live_store_paths(module: Path) -> bool:
        text = module.read_text(encoding="utf-8", errors="replace")
        return "aelfrice/memory.db" in text or "AELFRICE_DB" in text

    @staticmethod
    def _bare_memorystore_calls(module: Path) -> list[int]:
        """Line numbers of `MemoryStore(...)` calls with no `read_only=`."""
        try:
            tree = ast.parse(module.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken benchmark
            return []
        bare: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (
                fn.id if isinstance(fn, ast.Name)
                else fn.attr if isinstance(fn, ast.Attribute)
                else None
            )
            if name != "MemoryStore":
                continue
            if not any(k.arg == "read_only" for k in node.keywords):
                bare.append(node.lineno)
        return bare

    def test_no_benchmark_opens_a_named_live_store_read_write(self) -> None:
        offenders: list[str] = []
        for module in sorted(_BENCHMARKS.glob("*.py")):
            if not self._live_store_paths(module):
                continue
            for line in self._bare_memorystore_calls(module):
                offenders.append(f"{module.name}:{line}")
        assert offenders == [], (
            "these benchmarks name a live store path and open it without "
            f"read_only=: {offenders}. Pass read_only=True, or "
            "read_only=<writes-were-requested> if the script has a "
            "deliberate write mode."
        )

    def test_the_guard_can_actually_fire(self, tmp_path: Path) -> None:
        """The guard above is a no-op if its detector is broken.

        Feeds it a synthetic offender and asserts it is caught, so a
        refactor that silently stops matching `MemoryStore(` fails here
        instead of turning the real check green.
        """
        offender = tmp_path / "b.py"
        offender.write_text(
            "from aelfrice.store import MemoryStore\n"
            "s = MemoryStore('.git/aelfrice/memory.db')\n"
        )
        assert self._live_store_paths(offender)
        assert self._bare_memorystore_calls(offender) == [2]

        clean = tmp_path / "c.py"
        clean.write_text(
            "from aelfrice.store import MemoryStore\n"
            "s = MemoryStore('.git/aelfrice/memory.db', read_only=True)\n"
        )
        assert self._bare_memorystore_calls(clean) == []


def test_a_read_only_open_logs_no_migration_failures(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The gate in `_run_guarded_migration` earns its keep here.

    `mode=ro` already refuses the writes, so removing that gate does not
    break correctness — every pass simply raises and the guard swallows
    it. What it does break is quiet: eleven passes would raise, each
    logging at ERROR and each attempting to write a `migration_failed:`
    marker that also fails. A diagnostic that prints eleven store errors
    every run is one people stop reading.

    This is the arm that fails when the gate is removed; without it the
    gate is untested and the next refactor drops it as dead weight.
    """
    import logging

    db = tmp_path / "quiet.db"
    _store_with_an_expired_lock(db)

    with caplog.at_level(logging.ERROR, logger="aelfrice"):
        store = MemoryStore(str(db), read_only=True)
        try:
            store.list_beliefs_for_indexing()
        finally:
            store.close()

    failures = [r for r in caplog.records if "migration" in r.getMessage()]
    assert failures == [], (
        "a read-only open logged migration failures: the guarded-migration "
        f"gate is not firing — {[r.getMessage()[:80] for r in failures]}"
    )
