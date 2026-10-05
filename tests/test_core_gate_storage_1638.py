"""Core admission gate storage: the label cache and the batch table (#1638).

Each test names the mutation that kills it. The fixture text is neutral
and synthetic; no test opens a store other than one under `tmp_path`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

import pytest

from aelfrice.core_gate import MAX_BATCH
from aelfrice.models import CoreGateBatchItem
from aelfrice.store import (
    CoreGateAcceptRefused,
    MemoryStore,
    core_gate_batch_id,
)

VERSION = "core-gate-widgetversion"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "memory.db"


@pytest.fixture
def store(db: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(db))
    try:
        yield s
    finally:
        s.close()


def _items(n: int = 2) -> list[CoreGateBatchItem]:
    return [
        CoreGateBatchItem(index=i, belief_id=f"widget{i}", content_hash=f"hash{i}")
        for i in range(n)
    ]


def _batch(store: MemoryStore, n: int = 2, version: str = VERSION) -> str:
    return store.create_core_gate_batch(
        _items(n), classifier_version=version, origin="doctor",
        session_id=None, created_at="2026-10-05T00:00:00+00:00",
    )


def _tables(db: Path) -> set[str]:
    conn = sqlite3.connect(str(db))
    try:
        return {
            str(r[0]) for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        conn.close()


def _label_rows(db: Path) -> list[tuple[str, str, str]]:
    conn = sqlite3.connect(str(db))
    try:
        return [
            (str(r[0]), str(r[1]), str(r[2])) for r in conn.execute(
                "SELECT content_hash, classifier_version, label "
                "FROM core_gate_labels ORDER BY content_hash"
            )
        ]
    finally:
        conn.close()


# --- schema ---------------------------------------------------------------


def test_a_fresh_store_has_both_tables(store: MemoryStore, db: Path) -> None:
    """Killed by: deleting either CREATE TABLE from `_SCHEMA`."""
    assert {"core_gate_labels", "core_gate_batches"} <= _tables(db)


def test_an_older_store_gains_both_tables_on_open(db: Path) -> None:
    """A store written before the tables existed gets them on its next open.

    Killed by: deleting either CREATE TABLE from `_SCHEMA`.
    """
    MemoryStore(str(db)).close()
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "DROP TABLE core_gate_labels; DROP TABLE core_gate_batches;"
    )
    conn.close()
    assert not {"core_gate_labels", "core_gate_batches"} & _tables(db)
    MemoryStore(str(db)).close()
    assert {"core_gate_labels", "core_gate_batches"} <= _tables(db)


def test_the_label_column_refuses_a_label_outside_the_set(
    store: MemoryStore,
) -> None:
    """Killed by: dropping the CHECK on `core_gate_labels.label`."""
    with pytest.raises(sqlite3.IntegrityError):
        store.put_core_gate_labels(
            {"hash0": "D"}, classifier_version=VERSION, batch_id=None,
            labeled_at="t",
        )


def test_the_origin_column_refuses_an_origin_outside_the_set(db: Path) -> None:
    """Killed by: dropping the CHECK on `core_gate_batches.origin`."""
    MemoryStore(str(db)).close()
    conn = sqlite3.connect(str(db))
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO core_gate_batches (batch_id, classifier_version, "
                "origin, session_id, items_json, created_at) "
                "VALUES ('x', 'v', 'elsewhere', NULL, '[]', 't')"
            )
    finally:
        conn.close()


# --- batch creation -------------------------------------------------------


def test_the_batch_id_is_the_pinned_digest_of_its_inputs() -> None:
    """The id is a pure function of version, timestamp, and sorted hashes.

    Killed by: dropping `created_at` from the hashed parts, dropping the
    version from them, or changing the separator or the prefix length.
    """
    got = core_gate_batch_id("v", ["hash1", "hash0"], "t")
    # sha256(b"v\0t\0hash0\0hash1")[:16], computed outside Python.
    assert got == "6748c4398a7233e0"
    assert core_gate_batch_id("v", ["hash0", "hash1"], "t") == got
    assert core_gate_batch_id("w", ["hash0", "hash1"], "t") != got
    assert core_gate_batch_id("v", ["hash0", "hash1"], "u") != got


def test_create_returns_the_derived_id_whatever_the_item_order(
    store: MemoryStore,
) -> None:
    """Killed by: removing `sorted(...)` in `core_gate_batch_id`."""
    items = _items(3)
    first = store.create_core_gate_batch(
        items, classifier_version=VERSION, origin="doctor",
        session_id=None, created_at="t",
    )
    again = store.create_core_gate_batch(
        list(reversed(items)), classifier_version=VERSION, origin="doctor",
        session_id=None, created_at="t",
    )
    assert first == again == core_gate_batch_id(
        VERSION, ["hash0", "hash1", "hash2"], "t"
    )


def test_recreating_an_identical_batch_is_a_no_op(
    store: MemoryStore, db: Path,
) -> None:
    """Killed by: `INSERT OR IGNORE` -> `INSERT` (the re-create raises)."""
    first = _batch(store)
    assert _batch(store) == first
    conn = sqlite3.connect(str(db))
    try:
        n = conn.execute("SELECT COUNT(*) FROM core_gate_batches").fetchone()[0]
    finally:
        conn.close()
    assert n == 1


def test_a_different_batch_with_the_same_id_is_refused(
    store: MemoryStore,
) -> None:
    """Same hashes, version, and timestamp, but other belief ids.

    Killed by: forcing `same = True` in the collision branch.
    """
    _batch(store)
    other = [
        CoreGateBatchItem(index=i, belief_id=f"gadget{i}", content_hash=f"hash{i}")
        for i in range(2)
    ]
    with pytest.raises(ValueError, match="different batch"):
        store.create_core_gate_batch(
            other, classifier_version=VERSION, origin="doctor",
            session_id=None, created_at="2026-10-05T00:00:00+00:00",
        )


@pytest.mark.parametrize(
    ("items", "match"),
    [
        # Killed by: removing the empty-batch check.
        ([], "at least one"),
        # Killed by: removing the MAX_BATCH check.
        (
            [
                CoreGateBatchItem(index=i, belief_id=f"w{i}", content_hash=f"h{i}")
                for i in range(MAX_BATCH + 1)
            ],
            "at most",
        ),
        # Killed by: removing the unique-index check.
        (
            [
                CoreGateBatchItem(index=0, belief_id="w0", content_hash="h0"),
                CoreGateBatchItem(index=0, belief_id="w1", content_hash="h1"),
            ],
            "indexes must be unique",
        ),
        # Killed by: removing the unique-hash check.
        (
            [
                CoreGateBatchItem(index=0, belief_id="w0", content_hash="h0"),
                CoreGateBatchItem(index=1, belief_id="w1", content_hash="h0"),
            ],
            "hashes must be unique",
        ),
    ],
    ids=["empty", "over-max", "repeated-index", "repeated-hash"],
)
def test_create_refuses_a_malformed_batch(
    store: MemoryStore, items: list[CoreGateBatchItem], match: str, db: Path,
) -> None:
    with pytest.raises(ValueError, match=match):
        store.create_core_gate_batch(
            items, classifier_version=VERSION, origin="doctor",
            session_id=None, created_at="t",
        )
    conn = sqlite3.connect(str(db))
    try:
        n = conn.execute("SELECT COUNT(*) FROM core_gate_batches").fetchone()[0]
    finally:
        conn.close()
    assert n == 0


def test_create_refuses_an_unknown_origin_before_sqlite_does(
    store: MemoryStore,
) -> None:
    """Killed by: removing the origin check (SQLite then raises
    IntegrityError, which is not a ValueError)."""
    with pytest.raises(ValueError, match="origin"):
        store.create_core_gate_batch(
            _items(), classifier_version=VERSION, origin="elsewhere",
            session_id=None, created_at="t",
        )


def test_a_batch_round_trips_with_its_items_in_index_order(
    store: MemoryStore,
) -> None:
    """Killed by: dropping the index sort in `_core_gate_items_json`."""
    items = list(reversed(_items(3)))
    bid = store.create_core_gate_batch(
        items, classifier_version=VERSION, origin="session_end",
        session_id="widgetsession", created_at="t",
    )
    got = store.get_core_gate_batch(bid)
    assert got is not None
    assert [i.index for i in got.items] == [0, 1, 2]
    assert got.items[2] == CoreGateBatchItem(2, "widget2", "hash2")
    assert (got.origin, got.session_id, got.accepted_at) == (
        "session_end", "widgetsession", None,
    )
    assert store.get_core_gate_batch("nosuchbatch") is None


# --- the label cache ------------------------------------------------------


def test_labels_are_keyed_by_content_hash_and_classifier_version(
    store: MemoryStore,
) -> None:
    """A label made under one version never answers for another.

    Killed by: replacing `classifier_version = ?` in the lookup with
    `? IS NOT NULL`.
    """
    store.put_core_gate_labels(
        {"hash0": "A"}, classifier_version=VERSION, batch_id=None,
        labeled_at="t",
    )
    assert store.core_gate_labels_for(["hash0", "hash9"], VERSION) == {
        "hash0": "A",
    }
    assert store.core_gate_labels_for(["hash0"], "core-gate-other") == {}


def test_the_lookup_covers_more_hashes_than_one_chunk(
    store: MemoryStore,
) -> None:
    """Killed by: running only the first chunk of the lookup loop."""
    labels = {f"hash{i:05d}": "B" for i in range(1201)}
    store.put_core_gate_labels(
        labels, classifier_version=VERSION, batch_id=None, labeled_at="t",
    )
    assert store.core_gate_labels_for(labels, VERSION) == labels


def test_a_later_label_replaces_an_earlier_one(
    store: MemoryStore, db: Path,
) -> None:
    """Killed by: `DO UPDATE SET ...` -> `DO NOTHING`."""
    store.put_core_gate_labels(
        {"hash0": "A"}, classifier_version=VERSION, batch_id="one",
        labeled_at="t1",
    )
    store.put_core_gate_labels(
        {"hash0": "C"}, classifier_version=VERSION, batch_id="two",
        labeled_at="t2",
    )
    assert store.core_gate_labels_for(["hash0"], VERSION) == {"hash0": "C"}
    assert _label_rows(db) == [("hash0", VERSION, "C")]


def test_a_batch_is_marked_accepted_once(store: MemoryStore) -> None:
    """Killed by: dropping `AND accepted_at IS NULL` from the UPDATE."""
    bid = _batch(store)
    assert store.mark_core_gate_batch_accepted(bid, "t1") is True
    assert store.mark_core_gate_batch_accepted(bid, "t2") is False
    got = store.get_core_gate_batch(bid)
    assert got is not None and got.accepted_at == "t1"


# --- accept ---------------------------------------------------------------


def test_accept_caches_each_label_under_its_items_content_hash(
    store: MemoryStore, db: Path,
) -> None:
    """Killed by: skipping `put_core_gate_labels` in accept, or keying the
    labels by belief id instead of content hash."""
    bid = _batch(store, 3)
    store.accept_core_gate_batch(
        bid, {0: "A", 1: "B", 2: "C"}, classifier_version=VERSION,
        accepted_at="t",
    )
    assert _label_rows(db) == [
        ("hash0", VERSION, "A"), ("hash1", VERSION, "B"), ("hash2", VERSION, "C"),
    ]


def test_accept_marks_the_batch_accepted(store: MemoryStore) -> None:
    """Killed by: skipping `mark_core_gate_batch_accepted` in accept."""
    bid = _batch(store)
    store.accept_core_gate_batch(
        bid, {0: "A", 1: "A"}, classifier_version=VERSION, accepted_at="t9",
    )
    got = store.get_core_gate_batch(bid)
    assert got is not None and got.accepted_at == "t9"


def _refused_writes_nothing(
    store: MemoryStore, db: Path, bid: str, labels: dict[int, str],
    version: str, match: str,
) -> None:
    before = store.get_core_gate_batch(bid)
    rows_before = _label_rows(db)
    with pytest.raises(CoreGateAcceptRefused, match=match):
        store.accept_core_gate_batch(
            bid, labels, classifier_version=version, accepted_at="t-refused",
        )
    assert store.get_core_gate_batch(bid) == before
    assert _label_rows(db) == rows_before


def test_accept_refuses_a_batch_from_another_classifier_version(
    store: MemoryStore, db: Path,
) -> None:
    """Killed by: removing the classifier-version check in accept."""
    bid = _batch(store, version="core-gate-stale")
    _refused_writes_nothing(
        store, db, bid, {0: "A", 1: "A"}, VERSION, "emitted under",
    )


def test_accept_refuses_a_batch_already_accepted(
    store: MemoryStore, db: Path,
) -> None:
    """Killed by: removing the `accepted_at is not None` check (the
    UPDATE guard then refuses with another message)."""
    bid = _batch(store)
    store.accept_core_gate_batch(
        bid, {0: "A", 1: "A"}, classifier_version=VERSION, accepted_at="t1",
    )
    _refused_writes_nothing(
        store, db, bid, {0: "C", 1: "C"}, VERSION, "already accepted",
    )


def test_accept_refuses_an_unknown_batch(store: MemoryStore, db: Path) -> None:
    """Killed by: removing the `batch is None` check (AttributeError)."""
    with pytest.raises(CoreGateAcceptRefused, match="no core-gate batch"):
        store.accept_core_gate_batch(
            "nosuchbatch", {0: "A"}, classifier_version=VERSION,
            accepted_at="t",
        )
    assert _label_rows(db) == []


@pytest.mark.parametrize(
    "labels",
    [{0: "A"}, {0: "A", 1: "A", 2: "A"}],
    ids=["missing-index", "extra-index"],
)
def test_accept_refuses_labels_that_do_not_cover_the_batch(
    store: MemoryStore, db: Path, labels: dict[int, str],
) -> None:
    """Killed by: removing the coverage check (a partial set is written,
    or an extra index raises KeyError)."""
    bid = _batch(store)
    _refused_writes_nothing(store, db, bid, labels, VERSION, "cover exactly")


def test_accept_refuses_a_label_outside_the_set(
    store: MemoryStore, db: Path,
) -> None:
    """Killed by: removing the label-set check (SQLite's CHECK then
    raises IntegrityError, which is not a refusal)."""
    bid = _batch(store)
    _refused_writes_nothing(
        store, db, bid, {0: "A", 1: "D"}, VERSION, "not A, B, or C",
    )


def test_a_failure_after_marking_rolls_the_mark_back(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept is one transaction: a failure while writing labels leaves
    the batch unaccepted.

    Killed by: replacing `with self.transaction(immediate=True):` in
    accept with a no-op context (the mark commits on its own).
    """
    bid = _batch(store)

    def _boom(*_a: object, **_k: object) -> int:
        raise RuntimeError("widget failure")

    monkeypatch.setattr(store, "put_core_gate_labels", _boom)
    with pytest.raises(RuntimeError, match="widget failure"):
        store.accept_core_gate_batch(
            bid, {0: "A", 1: "A"}, classifier_version=VERSION,
            accepted_at="t",
        )
    got = store.get_core_gate_batch(bid)
    assert got is not None and got.accepted_at is None
