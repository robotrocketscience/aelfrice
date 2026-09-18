"""#1561 — an observational open of a migrated store must write nothing.

`aelf search`, `stats`, `locked`, `speculative` and `show` open through
`db_paths.open_store_for_read()`, which attempts the ordinary **writable**
open first and only falls back to `mode=ro` when that open is refused for
lack of write access (#1416). Against a writable store the handle is the
writable one, so the whole open-time write window — the DDL battery, the
one-shot migrations, the scope-id mint and the #1314 expired-lock sweep —
runs on a command whose contract is observational.

#1561 was filed as a performance issue and the performance premise was
refuted: the avoidable component is 82 statements, none of them a write,
and a fraction of a millisecond. `benchmarks/store_open_cost.py` is the
producer for those figures. **This module is the part that outlives them.**

What it pins, and the difference between the two halves
-------------------------------------------------------
On an already-migrated store the open mutates no row, issues no
`INSERT`/`UPDATE`/`DELETE` at all, and leaves the database file's bytes
identical. That is the invariant a future migration would break silently:
a one-shot pass that stamps its marker on *every* open rather than once
turns every read command into a writer, and nothing else in the suite
would notice — `tests/test_readonly_store_1416.py` checks the bytes only
on stores frozen at 0444, where the engine forbids the write anyway, so
it cannot see a write on an ordinary store.

The invariant is **conditioned on the store being already migrated**, and
the condition is load-bearing rather than a hedge. A store carrying an
unrun one-shot backfill *is* migrated by an observational open, by design
— that is what the writable-first order buys. The second half of this
module builds exactly that store and asserts the write happens, so the
first half cannot be misread as "an observational open never writes", and
so that a future change making the open unconditionally read-only fails
here with the semantics it removed named.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from aelfrice.db_paths import open_store_for_read, repo_identity_from_db_path
from aelfrice.models import LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore
from benchmarks.store_open_cost import (
    ENV_PREFIX,
    REPO_STORE_DIRNAME,
    build_fixture,
    figures,
    main,
    measure_open,
)

_REPO = Path(__file__).resolve().parents[1]
_GATE = _REPO / "scripts" / "check_derived_figures.py"

_spec = importlib.util.spec_from_file_location("_cdf_1561", _GATE)
assert _spec and _spec.loader
# Declared `Any` rather than left implicit: pyright runs `tests/` in strict
# mode, where an implicitly-typed module object makes every attribute read an
# `Unknown`. Same load shape as `tests/test_derived_figures_1469.py` — the
# gate is a script, not an importable package.
cdf: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdf)

# Small enough to build in well under a second, large enough that the
# project-context backfill below has several rows to stamp. The published
# census is size-invariant over 200..20,000 beliefs, re-derived by the
# producer on every run, so nothing here depends on the count.
BELIEFS = 40


@pytest.fixture()
def migrated_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """A store at the repo-store layout, with every one-shot already run.

    Built through `benchmarks.store_open_cost.build_fixture` rather than a
    local copy of it, so the store this module guards and the store the
    published figures are measured on cannot drift apart into two shapes
    that agree only by coincidence.

    The layout matters and is not decoration: `repo_identity_from_db_path`
    returns an identity only when the database's parent directory is named
    `aelfrice`, and an empty identity switches the project-context backfill
    off entirely. A fixture at any other path would satisfy the assertions
    below while never reaching the migration they are about.
    """
    db = tmp_path / REPO_STORE_DIRNAME / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_PROJECT_CONTEXT", raising=False)
    build_fixture(db, BELIEFS)
    yield db


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the invariant ---------------------------------------------------------


def test_an_observational_open_mutates_no_row(migrated_store: Path) -> None:
    """The headline guard. `total_changes` is the engine's own count.

    Read off the connection rather than by diffing tables: an open that
    wrote a row and deleted it again would leave every table identical and
    `total_changes` at 2, and a migration that rewrites a column to the
    value it already holds is the likeliest shape of the defect.
    """
    store = open_store_for_read()
    try:
        assert store.read_only is False, (
            "the fixture fell back to the read-only handle, so this asserts "
            "the engine's refusal rather than aelfrice's behaviour"
        )
        assert store._conn.total_changes == 0
    finally:
        store.close()


def test_an_observational_open_issues_no_write_statement(
    migrated_store: Path,
) -> None:
    """Stronger than the row count, and a different claim.

    No row changed is compatible with an `UPDATE` that matched nothing, and
    a future migration whose predicate happens to be empty on this fixture
    would pass the test above while shipping a writer. This one counts the
    statements the open issues, so the write has to be absent rather than
    merely ineffective.
    """
    measured = measure_open(migrated_store)
    writable = measured["writable"]
    assert writable["dml_statements"] == 0
    assert writable["statements"] > 0, (
        "a census of nothing would pass this test vacuously"
    )
    assert writable["other"] == 0, (
        "the open issued a statement class this module does not classify; "
        "read benchmarks/store_open_cost.py's report before widening it"
    )


def test_an_observational_open_leaves_the_file_bytes_identical(
    migrated_store: Path,
) -> None:
    """What a user can check without instrumenting anything.

    Taken after `close()`, which is when a WAL checkpoint would land, so
    this covers a write that reaches the log and not yet the main file.
    """
    before = _digest(migrated_store)
    store = open_store_for_read()
    store.close()
    assert _digest(migrated_store) == before
    for suffix in ("-wal", "-shm"):
        sidecar = migrated_store.with_name(migrated_store.name + suffix)
        assert not sidecar.exists() or sidecar.stat().st_size == 0


def test_the_read_only_arm_is_strictly_cheaper(migrated_store: Path) -> None:
    """The floor exists, so "avoidable" is a measured gap and not a guess.

    If this ever reads equal, the issue's whole premise has changed shape:
    there would be nothing for a fix to remove, and the published
    `avoidable_statements` would be zero rather than stale.
    """
    measured = measure_open(migrated_store)
    assert (
        measured["readonly"]["statements"] < measured["writable"]["statements"]
    )
    assert measured["readonly"]["rows_mutated"] == 0


# --- the precondition, which is load-bearing -------------------------------


def _unstamped_store(db: Path) -> int:
    """Build a store whose project-context backfill has not run.

    Written with `project_context_default=''`, which is what a direct
    `MemoryStore(path)` open does: no row is stamped on insert and the
    one-shot's completion marker is never set. Returns the number of rows
    the backfill is due to stamp, counted off the store rather than
    computed from `BELIEFS`, so the assertion below is derived from the
    fixture instead of restating the generator's arithmetic.
    """
    db.parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(db))
    try:
        for index in range(BELIEFS):
            content = f"unstamped belief {index} lorem ipsum dolor sit amet"
            locked = index % 10 == 0
            store.insert_belief(
                Belief(
                    id=hashlib.sha256(str(index).encode()).hexdigest()[:16],
                    content=content,
                    content_hash=hashlib.sha256(content.encode()).hexdigest(),
                    alpha=5.0,
                    beta=1.0,
                    type="factual",
                    lock_level=LOCK_USER if locked else LOCK_NONE,
                    locked_at="2026-01-01T00:00:00Z" if locked else None,
                    created_at="2026-01-01T00:00:00Z",
                    last_retrieved_at=None,
                    origin="user_asserted" if locked else "agent_inferred",
                )
            )
        due = int(
            store._conn.execute(
                "SELECT count(*) FROM beliefs WHERE project_context = '' "
                "AND scope = 'project' AND lock_level != 'user'"
            ).fetchone()[0]
        )
    finally:
        store.close()
    return due


def test_an_unmigrated_store_is_written_by_an_observational_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The condition on the invariant, asserted rather than assumed.

    This is the behaviour `open_store_for_read`'s writable-first order
    exists to preserve: a store an older binary wrote is brought up to the
    current shape by whichever command opens it next, including a read
    command. Stating it here keeps the zero-mutation guard above honest
    about what it does and does not cover, and makes the cost of switching
    the open to `mode=ro` unconditionally visible as a failing test rather
    than as a silent semantic change.
    """
    db = tmp_path / REPO_STORE_DIRNAME / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv("AELFRICE_PROJECT_CONTEXT", raising=False)
    due = _unstamped_store(db)
    assert due > 0, "a store with nothing due would pass this test vacuously"
    assert repo_identity_from_db_path(db), (
        "without a repo identity the backfill is switched off and this test "
        "would assert nothing"
    )

    store = open_store_for_read()
    try:
        # `due` rows stamped, plus the one `schema_meta` row that records
        # the one-shot as complete.
        assert store._conn.total_changes == due + 1
    finally:
        store.close()

    # And it is a one-shot: the very next observational open is back inside
    # the invariant above.
    store = open_store_for_read()
    try:
        assert store._conn.total_changes == 0
    finally:
        store.close()


# --- the producer's contract ----------------------------------------------


def test_the_producer_has_no_store_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#1328. Two shipped benchmarks defaulted `--store` to the live store.

    A run with no mode selected must refuse rather than resolve anything.
    `db_path()` is monkeypatched to raise so that a future edit reaching for
    it fails loudly here instead of quietly opening whatever the operator's
    environment points at.
    """
    import aelfrice.db_paths as db_paths

    def _forbidden() -> Path:
        raise AssertionError("the benchmark resolved a live store path")

    monkeypatch.setattr(db_paths, "db_path", _forbidden)
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_the_producer_leaves_no_store_path_behind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A producer that redirects the suite's default store is a time bomb.

    `figures()` assigns `AELFRICE_DB` inside its hermetic block, pointing at
    a database in a temporary directory that the block then deletes.
    Restoring only the variables that existed on *entry* leaves that
    assignment standing, so the next test in the session to resolve the
    default store is silently pointed at a dead path — and `MemoryStore`
    would create a fresh empty store there rather than fail.

    Run on a one-element grid: the leak is a property of the block, not of
    the grid, and the published values are the derived-figure gate's job.
    """
    monkeypatch.delenv("AELFRICE_DB", raising=False)
    figures(grid=(20,))
    leaked = sorted(k for k in os.environ if k.startswith(ENV_PREFIX))
    assert not leaked, f"the producer left these set: {leaked}"


def test_emit_figures_refuses_a_named_store() -> None:
    """A published figure has to be re-derivable by whoever reads it.

    `scripts/check_derived_figures.py --mode producers` runs the producer
    with `--emit-figures` and nothing else, so a figure measured against a
    store only its author has could never be re-derived by the gate that
    claims to guard it.
    """
    with pytest.raises(SystemExit) as excinfo:
        main(["--emit-figures", "--store", "/nonexistent/memory.db"])
    assert excinfo.value.code == 2


def test_every_marker_naming_this_producer_has_a_key_it_emits() -> None:
    """AC1's actual requirement: a marker whose key nothing emits is inert.

    Run on a one-element grid, which emits the same key set as the
    published run at a fraction of the cost; the *values* are the gate's
    job, and `tests/test_derived_figures_1469.py` already runs it.
    """
    files = cdf.iter_files(list(cdf.DEFAULT_ROOTS))
    report = cdf.Report(github=False)
    cited = {
        marker.key
        for marker in cdf.check_text(files, report)
        if marker.producer == "benchmarks/store_open_cost.py"
    }
    assert cited, "no marker names this producer, so it guards nothing"
    emitted = set(figures(grid=(20,)))
    assert cited <= emitted, (
        "these markers name keys the producer does not emit: "
        f"{sorted(cited - emitted)}"
    )


def test_the_producer_refuses_a_grid_whose_census_disagrees(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Averaging a moving count would publish a figure that is true nowhere.

    Simulated at the measurement seam rather than by finding a real
    divergence: the point under test is the producer's refusal, and a
    genuine size-dependent count is the thing that must never exist.
    """
    import benchmarks.store_open_cost as producer

    real = producer.measure_open
    calls: list[int] = []

    def flaky(db: Path) -> dict[str, Any]:
        measured: dict[str, Any] = real(db)
        calls.append(1)
        if len(calls) == 1:
            writable: dict[str, Any] = dict(measured["writable"])
            writable["statements"] = int(writable["statements"]) + 1
            measured["writable"] = writable
        return measured

    monkeypatch.setattr(producer, "measure_open", flaky)
    with pytest.raises(AssertionError, match="not a schema property"):
        producer.figures(grid=(20, 40))


def test_a_measurement_of_a_named_store_copies_it_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening a store is a write, so measuring one in place would mutate it.

    Asserted on the bytes of the named database, which is the property the
    operator cares about, rather than on the copy having happened.
    """
    db = tmp_path / REPO_STORE_DIRNAME / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    build_fixture(db, BELIEFS)
    before = _digest(db)
    real_connect = sqlite3.connect

    def _no_timings(db: Path, repeats: int = 1) -> dict[str, float]:
        return {
            "writable_open_ms": 0.0,
            "readonly_open_ms": 0.0,
            "avoidable_ms": 0.0,
        }

    def _no_command(repeats: int = 1) -> float:
        return 1.0

    monkeypatch.setattr(
        "benchmarks.store_open_cost.time_opens", _no_timings
    )
    monkeypatch.setattr(
        "benchmarks.store_open_cost.time_stats_command", _no_command
    )
    assert main(["--store", str(db)]) == 0
    assert _digest(db) == before
    # The census replaces `sqlite3.connect` while it runs. A run that left
    # the wrapper installed would silently trace every later test's store.
    assert sqlite3.connect is real_connect
