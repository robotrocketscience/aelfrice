"""#1283 AC2: the spine is recomputable from the log, and the gap is named.

The recompute keys on `(created_at, ingest_log ULID, id)` against the
shipped writer's `(created_at, rowid)`, where `rowid` is implicit and
`VACUUM` may renumber it. The ULID component is durable as an *observed
property, not a guarantee* — `ulid.make_generator` is monotone only
within a process — so nothing here may be written as if the key were
guaranteed; see `spine_recompute.recompute_spine_edges` for the measured
exposure.

**A test here must not assert that divergence is zero.** The writer has
not been changed yet, so writer and recompute key on different things
and a zero would mean the recompute had been fitted to the defect. What
these pin instead is that each *bucket* of the gap is what it claims to
be, because only one of the three is a defect anyone can fix.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aelfrice import cli
from aelfrice.models import EDGE_TEMPORAL_NEXT, LOCK_USER, Belief, Edge
from aelfrice.spine_recompute import (
    SYNTH_SOURCE_KIND,
    recompute_spine_edges,
    spine_divergence,
)
from aelfrice.store import MemoryStore


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's repo-local live store out of every test."""
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))


@pytest.fixture()
def store() -> MemoryStore:
    return MemoryStore(":memory:")


def _belief(
    store: MemoryStore, bid: str, *, session: str, created_at: str
) -> None:
    store.insert_belief(Belief(
        id=bid,
        content=f"content for {bid}",
        content_hash=f"hash-{bid}",
        alpha=1.0,
        beta=1.0,
        type="fact",
        lock_level="none",
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=None,
        session_id=session,
    ))


def _log(
    store: MemoryStore,
    ulid: str,
    belief_ids: list[str],
    *,
    source_kind: str = "transcript",
) -> None:
    """Plant an ingest_log row directly.

    Written through the connection rather than `record_ingest` because
    the ULID is the thing under test: it has to be chosen, not minted.
    """
    store._conn.execute(
        "INSERT INTO ingest_log (id, source_kind, raw_text, ts, "
        "derived_belief_ids) VALUES (?, ?, '', '2026-01-01T00:00:00Z', ?)",
        (ulid, source_kind, json.dumps(belief_ids)),
    )
    store._conn.commit()


# --- the key ------------------------------------------------------------

def test_ulid_orders_beliefs_that_share_a_created_at(
    store: MemoryStore,
) -> None:
    """The whole point: a `created_at` tie is broken by the log, not rowid.

    All three beliefs carry the same timestamp and are inserted in an
    order that disagrees with their ULIDs, so a recompute that fell back
    to insertion order would chain them b-a-c instead of a-b-c.
    """
    same = "2026-03-01T00:00:00Z"
    for bid in ("b", "a", "c"):
        _belief(store, bid, session="s1", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["a"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["b"])
    _log(store, "01CCCCCCCCCCCCCCCCCCCCCCCC", ["c"])

    edges, no_log = recompute_spine_edges(store)
    assert no_log == set()
    assert edges == {("b", "a"), ("c", "b")}


def test_ulid_beats_belief_id_when_the_two_orders_disagree(
    store: MemoryStore,
) -> None:
    """The ULID is the key — not the belief id that usually agrees with it.

    The sibling test above picks ids whose alphabetical order happens to
    match their ULID order, so `(created_at, id)` and
    `(created_at, ULID, id)` produce the same chain and it cannot tell
    them apart. Replacing the log key with a constant leaves it green.

    Here the two orders are deliberately opposed: `aaa` carries the
    LAST ULID and `ccc` the first, so the log says ccc-bbb-aaa while the
    id says aaa-bbb-ccc. Only a recompute that actually consults the log
    produces the expected set — which matters because 96.5% of this
    store shares a `created_at`, so the id is doing the ordering
    whenever the ULID is ignored.
    """
    same = "2026-03-01T00:00:00Z"
    for bid in ("aaa", "bbb", "ccc"):
        _belief(store, bid, session="s1", created_at=same)
    _log(store, "01CCCCCCCCCCCCCCCCCCCCCCCC", ["aaa"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["bbb"])
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["ccc"])

    edges, no_log = recompute_spine_edges(store)
    assert no_log == set()
    # ULID order ccc < bbb < aaa. An `(created_at, id)` sort would give
    # aaa < bbb < ccc and therefore {("bbb", "aaa"), ("ccc", "bbb")}.
    assert edges == {("bbb", "ccc"), ("aaa", "bbb")}


def test_a_belief_takes_its_earliest_log_row(store: MemoryStore) -> None:
    """Later rows are corroborations; only the first records insertion."""
    same = "2026-03-01T00:00:00Z"
    _belief(store, "first", session="s1", created_at=same)
    _belief(store, "second", session="s1", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["first"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["second"])
    # `first` is corroborated later; that must not reorder it after
    # `second`.
    _log(store, "01ZZZZZZZZZZZZZZZZZZZZZZZZ", ["first"])

    edges, _ = recompute_spine_edges(store)
    assert edges == {("second", "first")}


def test_created_at_dominates_the_ulid(store: MemoryStore) -> None:
    """The key is `(created_at, ULID)`, not the ULID alone.

    Without this, a backfill — whose ULIDs are minted long after the
    content — would reorder a whole session by processing time.
    """
    _belief(store, "early", session="s1", created_at="2026-01-01T00:00:00Z")
    _belief(store, "late", session="s1", created_at="2026-06-01T00:00:00Z")
    # ULIDs deliberately inverted against the timestamps.
    _log(store, "01ZZZZZZZZZZZZZZZZZZZZZZZZ", ["early"])
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["late"])

    edges, _ = recompute_spine_edges(store)
    assert edges == {("late", "early")}


def test_sessions_do_not_chain_into_each_other(store: MemoryStore) -> None:
    same = "2026-03-01T00:00:00Z"
    _belief(store, "a1", session="s1", created_at=same)
    _belief(store, "b1", session="s2", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["a1"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["b1"])

    edges, _ = recompute_spine_edges(store)
    assert edges == set(), "beliefs from different sessions were chained"


# --- the synth exclusion ------------------------------------------------

def test_synth_rows_supply_no_ordering_key(store: MemoryStore) -> None:
    """A belief covered only by a synth row counts as having no log row.

    The #263 synthesis relabelled `beliefs.rowid` as a ULID, so honouring
    those keys would launder rowid order into the key the contract calls
    durable. Excluded by `source_kind`, a stated column — not by a
    prefix or density heuristic, both of which were measured and failed.
    """
    same = "2026-03-01T00:00:00Z"
    _belief(store, "real", session="s1", created_at=same)
    _belief(store, "synthetic", session="s1", created_at=same)
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["real"])
    _log(
        store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["synthetic"],
        source_kind=SYNTH_SOURCE_KIND,
    )

    edges, no_log = recompute_spine_edges(store)
    assert no_log == {"synthetic"}, (
        "a synth-covered belief was given an ordering key; its ULID is "
        "migration wall clock and its order is rowid relabelled"
    )
    # `synthetic` sorts last by the no-log convention despite its ULID
    # sorting first, which is the observable consequence.
    assert edges == {("synthetic", "real")}


# --- the no-log convention ----------------------------------------------

def test_no_log_beliefs_sort_last_within_their_group(
    store: MemoryStore,
) -> None:
    """A stated forward convention, not a recovery of historical order.

    The module docstring is explicit that the historical ordering of
    this bucket is unreconstructible — 93.9% of it sits inside a
    `created_at` tie where the only other durable column is a
    content-addressed id. What this pins is that the convention is
    applied consistently, so the recompute is deterministic.
    """
    same = "2026-03-01T00:00:00Z"
    for bid in ("logged", "orphan_b", "orphan_a"):
        _belief(store, bid, session="s1", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["logged"])

    edges, no_log = recompute_spine_edges(store)
    assert no_log == {"orphan_a", "orphan_b"}
    # logged -> orphan_a -> orphan_b: no-log last, then id ASC.
    assert edges == {("orphan_a", "logged"), ("orphan_b", "orphan_a")}


def test_recompute_is_deterministic(store: MemoryStore) -> None:
    """Same store state, same edge set — including the no-log bucket.

    Determinism is the property the no-log convention exists to buy, so
    it is asserted over a store that has one.
    """
    same = "2026-03-01T00:00:00Z"
    for bid in ("d", "a", "c", "b"):
        _belief(store, bid, session="s1", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["a"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["b"])

    first, first_no_log = recompute_spine_edges(store)
    second, second_no_log = recompute_spine_edges(store)
    assert first == second
    assert first_no_log == second_no_log


# --- divergence attribution ---------------------------------------------

def test_divergence_is_zero_when_the_writer_agrees(
    store: MemoryStore,
) -> None:
    """The control. Without it every bucket assertion below is vacuous —
    a recompute that produced nothing would file every shipped edge under
    some bucket and look like a correct attribution."""
    same = "2026-03-01T00:00:00Z"
    _belief(store, "a", session="s1", created_at=same)
    _belief(store, "b", session="s1", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["a"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["b"])
    store.insert_edge(Edge(src="b", dst="a", type=EDGE_TEMPORAL_NEXT, weight=1.0))

    report = spine_divergence(store)
    assert report.n_shipped == 1
    assert report.n_reproduced == 1
    assert report.reproduced_share == 1.0
    assert report.missing_other == 0


def test_a_no_log_miss_lands_in_its_own_bucket(store: MemoryStore) -> None:
    """Unreconstructible misses must never be counted as drift."""
    same = "2026-03-01T00:00:00Z"
    _belief(store, "orphan", session="s1", created_at=same)
    _belief(store, "logged", session="s1", created_at=same)
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["logged"])
    # The writer chained orphan first; the recompute puts it last.
    store.insert_edge(
        Edge(src="logged", dst="orphan", type=EDGE_TEMPORAL_NEXT, weight=1.0)
    )

    report = spine_divergence(store)
    assert report.n_reproduced == 0
    assert report.missing_touching_no_log == 1
    assert report.missing_fan_in == 0
    assert report.missing_other == 0


def test_a_fan_in_miss_lands_in_its_own_bucket(store: MemoryStore) -> None:
    """Two predecessors for one successor is a writer defect, not a key
    disagreement — a chain gives each successor exactly one."""
    for i, bid in enumerate(("a", "b", "c")):
        _belief(store, bid, session="s1", created_at=f"2026-03-0{i+1}T00:00:00Z")
    _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["a"])
    _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["b"])
    _log(store, "01CCCCCCCCCCCCCCCCCCCCCCCC", ["c"])
    store.insert_edge(Edge(src="b", dst="a", type=EDGE_TEMPORAL_NEXT, weight=1.0))
    store.insert_edge(Edge(src="c", dst="b", type=EDGE_TEMPORAL_NEXT, weight=1.0))
    # The defect: `c` gains a second predecessor.
    store.insert_edge(Edge(src="c", dst="a", type=EDGE_TEMPORAL_NEXT, weight=1.0))

    report = spine_divergence(store)
    assert report.n_shipped == 3
    assert report.n_reproduced == 2
    assert report.missing_fan_in == 1
    assert report.missing_touching_no_log == 0
    assert report.missing_other == 0


def test_an_empty_store_reports_full_reproduction(
    store: MemoryStore,
) -> None:
    """A zero-edge store is not a 0% reproduction — it is nothing to
    reproduce. Getting this wrong makes a fresh store look broken."""
    report = spine_divergence(store)
    assert report.n_shipped == 0
    assert report.reproduced_share == 1.0


# --- the CLI surface ----------------------------------------------------
#
# `aelf spine verify` is the only way anybody runs the recompute, and the
# thing that makes it safe to point at a live store is one keyword. A test
# that merely invokes the command cannot see that keyword go missing, so
# the arms below are split: one reads the printed report, one reads the
# open itself, and one reads the file bytes. Each names the mutation it
# is there to catch.


@pytest.fixture()
def cli_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A file-backed store whose divergence lands in all three buckets.

    Built so the report cannot be right by accident. Every count it
    prints is a different number — 10 shipped, 7 recomputed, 4
    reproduced, then 3 / 2 / 1 across the buckets — so an implementation
    that prints the right values in the wrong slots fails here. A
    fixture with equal buckets cannot tell those two apart.

    * `s1` chains a-b-c-d by the log. The writer shipped that chain plus
      two extra predecessors, so `c` and `d` each carry two: **2** in
      the fan-in bucket, and 3 links reproduced.
    * `s2` holds two beliefs with no log row at all. All three links the
      writer shipped there touch one: **3** in the no-log bucket.
    * `s3` chains x-y-z; the writer shipped `z <- y` correctly and
      reversed `x <- y`. One successor, one predecessor, no orphan —
      **1** in `other`, the only bucket a key disagreement moves.

    It also carries a user lock whose window closed two days ago. That
    belief is deliberately session-less, so it contributes no edge and
    cannot perturb the counts; it is here so a read-write open has
    something observable to break, which is what makes the
    byte-identity arm below non-vacuous.
    """
    db = tmp_path / "spine.db"
    store = MemoryStore(str(db))
    try:
        same = "2026-03-01T00:00:00Z"
        for bid in ("a", "b", "c", "d"):
            _belief(store, bid, session="s1", created_at=same)
        _log(store, "01AAAAAAAAAAAAAAAAAAAAAAAA", ["a"])
        _log(store, "01BBBBBBBBBBBBBBBBBBBBBBBB", ["b"])
        _log(store, "01CCCCCCCCCCCCCCCCCCCCCCCC", ["c"])
        _log(store, "01DDDDDDDDDDDDDDDDDDDDDDDD", ["d"])

        for bid in ("logged", "orphan1", "orphan2"):
            _belief(store, bid, session="s2", created_at=same)
        _log(store, "01EEEEEEEEEEEEEEEEEEEEEEEE", ["logged"])

        for bid in ("x", "y", "z"):
            _belief(store, bid, session="s3", created_at=same)
        _log(store, "01FFFFFFFFFFFFFFFFFFFFFFFF", ["x"])
        _log(store, "01GGGGGGGGGGGGGGGGGGGGGGGG", ["y"])
        _log(store, "01HHHHHHHHHHHHHHHHHHHHHHHH", ["z"])

        for src, dst in (
            ("b", "a"),                 # reproduced
            ("c", "b"),                 # reproduced
            ("d", "c"),                 # reproduced
            ("c", "a"),                 # fan-in: `c` gains a 2nd predecessor
            ("d", "b"),                 # fan-in: so does `d`
            ("logged", "orphan1"),      # no-log endpoint
            ("orphan1", "orphan2"),     # no-log endpoint
            ("orphan2", "logged"),      # no-log endpoint
            ("x", "y"),                 # the log says y <- x; `other`
            ("z", "y"),                 # reproduced
        ):
            store.insert_edge(
                Edge(src=src, dst=dst, type=EDGE_TEMPORAL_NEXT, weight=1.0)
            )

        past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        store.insert_belief(Belief(
            id="expired_lock",
            content="the release key rotates at the end of the quarter",
            content_hash="hash-expired_lock",
            alpha=1.0,
            beta=1.0,
            type="fact",
            lock_level=LOCK_USER,
            locked_at=past,
            created_at=past,
            last_retrieved_at=None,
            lock_expires_at=past,
        ))
    finally:
        store.close()

    # Settle the WAL so the byte comparison is against a quiescent file.
    con = sqlite3.connect(str(db))
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()

    # `_cmd_spine_verify` resolves `db_path` off the `cli` module, so that
    # is the name to patch (verify the call site, not the resolver). The
    # env var is pinned to the same file as a backstop: a refactor onto
    # `_open_store()` would bypass the patched name and reach for
    # `db_paths.db_path`, and the byte-identity arm should catch it there
    # rather than silently measuring the developer's own store.
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setattr(cli, "db_path", lambda: db)
    return db


def _verify(out: io.StringIO) -> int:
    """Drive the command the way the parser does: through `_cmd_spine`.

    Calling `_cmd_spine_verify` directly would leave the dispatch arm in
    `_cmd_spine` uncovered, and that arm is what keeps `verify` off the
    read-write backfill path.
    """
    return cli._cmd_spine(
        argparse.Namespace(action="verify", dry_run=False), out
    )


def test_spine_verify_reports_every_bucket_and_names_the_gap(
    cli_store: Path,
) -> None:
    """The report's shape, over a store that populates all three buckets.

    Catches: the `action == "verify"` dispatch being dropped (the output
    becomes a backfill line), any bucket being dropped from the report,
    a bucket being printed from the wrong field (each of the three
    carries a distinct count), and the deletion of the gap-not-drift
    note — without which a reader takes a non-zero divergence for
    corruption rather than the expected state of an unchanged writer.
    """
    out = io.StringIO()
    assert _verify(out) == 0
    text = out.getvalue()

    # Anchored on the line end / the trailing space before each
    # parenthetical, so `: 1` cannot quietly match a printed `12`.
    assert "shipped TEMPORAL_NEXT : 10\n" in text
    assert "recomputed            : 7\n" in text
    assert "reproduced            : 4 (40.00%)\n" in text
    assert "--- misses, by cause ---" in text
    assert "no-log endpoint     : 3 (" in text
    assert "fan-in > 1          : 2 (" in text
    assert "other               : 1 (" in text
    assert "gap against the ratified key, not drift" in text


def test_spine_verify_opens_the_store_read_only(
    cli_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keyword, asserted at the call site.

    Catches: `read_only=True` being dropped or flipped to False. The
    byte-identity arm below catches the same mutation behaviourally;
    this one catches it *by name*, so the failure message points at the
    line instead of at a hash mismatch.

    `MemoryStore` is patched on `cli` because that is where
    `_cmd_spine_verify` resolves it — patching `aelfrice.store` would
    leave the already-bound module global untouched and the assertion
    would never see the call.
    """
    real = cli.MemoryStore
    opens: list[dict[str, object]] = []

    def _spy(path: str, **kwargs: object) -> MemoryStore:
        opens.append(dict(kwargs))
        return real(path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "MemoryStore", _spy)
    assert _verify(io.StringIO()) == 0

    assert len(opens) == 1, (
        f"expected exactly one store open, saw {len(opens)}"
    )
    assert opens[0].get("read_only") is True, (
        "`aelf spine verify` opened the store read-write. This is a "
        "diagnostic: a bare open runs the DDL battery, pending "
        "migrations, the scope-id mint and the #1314 lock sweep against "
        "the store it exists to measure (#1328)."
    )


def test_spine_verify_leaves_the_store_bytes_alone(cli_store: Path) -> None:
    """Stronger than "no logical change": no bytes move at all.

    Catches: `read_only=True` being dropped, and also the wider
    regression the kwarg spy cannot see — `db_path()` being swapped for
    `_open_store()`, which reaches past the patched name and opens
    read-write anyway.

    The control below proves this store is one a write open would in
    fact change, so a green here is a fact about the open and not about
    the fixture.
    """
    before = hashlib.sha256(cli_store.read_bytes()).hexdigest()
    assert _verify(io.StringIO()) == 0
    after = hashlib.sha256(cli_store.read_bytes()).hexdigest()
    assert after == before, (
        "`aelf spine verify` rewrote the store it was measuring"
    )


def test_the_same_store_is_swept_by_a_write_open(cli_store: Path) -> None:
    """The control. Without it the two arms above pass on a store that
    nothing would have written to in the first place."""
    store = MemoryStore(str(cli_store))
    try:
        assert store.get_belief("expired_lock").lock_level != LOCK_USER
    finally:
        store.close()


def test_spine_verify_leaves_an_expired_lock_locked(cli_store: Path) -> None:
    """The specific mutation observed on the live store (#1328).

    Read through a fresh connection rather than through `MemoryStore`,
    because opening one to check would itself run the sweep.
    """
    assert _verify(io.StringIO()) == 0
    con = sqlite3.connect(f"file:{cli_store}?mode=ro", uri=True)
    try:
        level = con.execute(
            "SELECT lock_level FROM beliefs WHERE id = ?", ("expired_lock",)
        ).fetchone()[0]
    finally:
        con.close()
    assert level == LOCK_USER, (
        "running the diagnostic unlocked a belief: the #1314 sweep ran"
    )


def test_spine_verify_exits_zero_on_an_empty_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "exits 0 regardless" promise, on the degenerate store.

    A zero-edge store reports full reproduction — there is nothing to
    reproduce — and that has to survive the trip through the CLI's
    formatting, where a naive percentage would divide by zero.
    """
    db = tmp_path / "empty.db"
    MemoryStore(str(db)).close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setattr(cli, "db_path", lambda: db)

    out = io.StringIO()
    assert _verify(out) == 0
    text = out.getvalue()
    assert "shipped TEMPORAL_NEXT : 0\n" in text
    assert "reproduced            : 0 (100.00%)\n" in text


def test_spine_verify_says_so_when_there_is_no_store_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that was never built is a message, not a traceback.

    `read_only=True` opens through SQLite's `mode=ro` URI, which refuses
    to create the file — so unlike `backfill` and `clear`, which reach
    the store through `_open_store()`, `verify` is the one spine action
    that meets a fresh repo with nothing there. Without the existence
    guard this raises `sqlite3.OperationalError` out through `main()`,
    which wraps nothing, and the user sees a raw traceback that reads
    identically to a corrupt store or a permissions fault.

    Catches: deletion of the guard. The assertion is on the exit code
    and the message, not on the absence of an exception, so a guard
    that swallows the error and returns 1 fails too.
    """
    missing = tmp_path / "never-built" / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(missing))
    monkeypatch.setattr(cli, "db_path", lambda: missing)

    out = io.StringIO()
    assert _verify(out) == 0
    assert "nothing to verify" in out.getvalue()
    assert not missing.exists(), (
        "a read-only diagnostic created the store it was asked to read"
    )
