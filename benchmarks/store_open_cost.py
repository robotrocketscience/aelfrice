#!/usr/bin/env python3
"""#1561 — what an observational command's store open actually costs.

`aelf search`, `stats`, `locked`, `speculative` and `show` open through
`db_paths.open_store_for_read()`, which attempts the ordinary **writable**
open first and falls back to `mode=ro` only when that open is refused for
lack of write access (#1416). Against a writable store — every ordinary
store — the handle is the writable one, so an observational command pays
the schema battery, the migrations, the scope-id mint and the expired-lock
sweep.

#1561 was filed on the premise that this is a cost worth removing. **This
module is the producer that refutes that premise**, and it changes nothing
in `src/aelfrice/`: the open order is not touched here and is not proposed
to be.

## What it measures, and why these are the figures

Every statement the open issues is captured by installing a
`sqlite3` trace callback on the connection `MemoryStore` opens, and every
row it changes is read off that connection's `total_changes`. Three
things come out, and only the first two are published:

* **A statement census, decomposed by component.** On a settled store at
  the repo-store layout the writable open issues 87 statements: 56
  `CREATE ... IF NOT EXISTS`, 26 `SELECT` and 5 `PRAGMA`. AC1 asks for the
  four components separately, so the census attributes every statement to
  the one that issued it — 4 connection `PRAGMA`s, 58 DDL battery, 23
  migration check, 1 scope-id mint, 1 expired-lock sweep — by execution
  nesting rather than by leading keyword. Not one is `INSERT`, `UPDATE` or
  `DELETE`, no row changes, and the database file's bytes are identical
  afterwards. The read-only floor — what `mode=ro` pays, which is the
  cheapest open that can exist — is 5 statements, so 82 statements are
  what an ideal fix would remove.
* **The sweep, on a store that has something for it to do.** The census
  above is taken where nothing is due, which is the sweep's *empty* cost
  and not its cost. A separate arm back-dates one lock's
  `lock_expires_at` and measures the same open again: 95 statements, the
  sweep's own 9 of them, 6 `INSERT`/`UPDATE`, 6 rows, and the database
  file's bytes changed. That arm is why the zero-mutation figures are
  reported against *a settled store* and not against an already-migrated
  one — see "The invariant" below.
* **The counts are a property of the shipped schema, not of the corpus.**
  They are identical at 200, 2,000 and 20,000 beliefs, and this module
  refuses to emit a figure unless the whole grid agrees, so a count that
  starts moving with store size fails the gate rather than being averaged.
  They are *not* independent of what the store holds: the arms above
  differ by 8 statements on one back-dated column, which is a property of
  the store's state rather than of its size. The grid pins the second and
  the expired-lock arm publishes the first.
* **Wall clock, which is deliberately not published.** The report modes
  print it; `--emit-figures` does not emit it. A latency is a property of
  the machine that ran it, so a marker over one would go stale on the
  first runner whose clock differs — and `docs/design/derived_figure_markers.md`'s
  own rule is that a marker stamping a value nobody can recompute
  publishes a second unchecked number beside the first.

## Store-free, and why that was the choice

The figures carry no `corpus=` attribute. This module builds its own
fixture from shipped code in a temporary directory, so a public runner
can rebuild it and `scripts/check_derived_figures.py --mode producers`
re-runs it on every pull request, hard-failing on any difference. A
store-backed marker would have recorded which corpus on what day and
bought only self-consistency and a staleness warning, which is strictly
weaker. The figures are eligible for that stronger class precisely
because they are counts of statements rather than of beliefs.

## `--store` has no default, and it copies

Opening a store is a write. Two shipped benchmarks defaulted `--store` to
the live `.git/aelfrice/memory.db` and mutated the corpus they existed to
measure (#1328). So `--store` here has no default, nothing in this file
calls `db_paths.db_path()`, and a run with no mode selected exits 2 rather
than resolving anything. A named store is **copied** into a temporary
directory before it is opened, for the same reason: this module's whole
subject is that the open under measurement is a write attempt, and
measuring it in place would run that attempt against the operator's
corpus.

## The invariant, and where it is enforced

`rows_mutated = 0` holds on a **settled** store, which is two conditions
and not one:

1. every one-shot migration has already run, and
2. no time-boxed lock is due to expire.

Neither is a hedge and neither is rare in the other direction. A store
carrying an unrun one-shot backfill is migrated by the writable open —
the project-context backfill was measured changing 301 rows on such a
store — and a store holding a lock past its `lock_expires_at` is swept by
it, which the `expired_lock_*` figures above measure at 6 rows. The
second condition is the one an ordinary user meets repeatedly: any store
that has ever held a time-boxed lock reaches the sweep's write path on
whichever observational command opens it next. Both are the writable-first
order working as designed, and both are why the zero figures are published
with their precondition attached rather than as a property of
observational opens.

That precondition is what makes the figure worth guarding — a future
migration that stamps its marker on every open rather than once would
turn a read command into a writer on a settled store, silently. The
permanent guard is `tests/test_store_open_cost_1561.py`, which pins the
zero on a settled store and the write on each unsettled one; this module
is the producer for the numbers.

Usage:

    uv run python benchmarks/store_open_cost.py --synthetic
    uv run python benchmarks/store_open_cost.py --store /path/to/memory.db
    uv run python benchmarks/store_open_cost.py --emit-figures
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

# Belief counts the published census is required to agree across. Two orders
# of magnitude, because the claim being published is that the open's cost is
# a function of the schema and not of the corpus; a single-size measurement
# could not distinguish those.
SIZE_GRID: tuple[int, ...] = (200, 2_000, 20_000)

# The fixture size the report modes use when no store is named. Large enough
# that a per-row cost would show, small enough to build in well under a second.
DEFAULT_BELIEFS = 2_000

# Repetitions behind each reported median. Opens are sub-millisecond, so the
# loop is cheap and a single reading would be dominated by scheduler noise.
TIMING_REPEATS = 31

# The directory name that makes a path *the repo store*. `db_paths.
# repo_identity_from_db_path` returns a repo identity only when the database's
# parent directory is named this, and a non-empty identity is what puts the
# project-context backfill's marker probe into the open. A fixture at any
# other path measures a different, cheaper open — which is why both shapes
# are published rather than one.
REPO_STORE_DIRNAME = "aelfrice"

ENV_PREFIX = "AELFRICE_"

# Git's own location variables. `repo_identity_from_db_path` derives its
# identity from the path rather than by forking git, but `MemoryStore` open
# and the CLI command timed beside it read the surrounding repository through
# these, and `$TMPDIR` inside a git work tree is a real configuration on a CI
# runner. Both barriers go up: `GIT_CEILING_DIRECTORIES` is list-parsed and a
# colon in the tempdir's parent voids it, while `GIT_DIR` is single-valued and
# stops discovery outright.
_GIT_LOCATION_VARS: tuple[str, ...] = (
    "GIT_CEILING_DIRECTORIES",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
)

# Nothing creates it: a `GIT_DIR` naming a directory that does not exist is
# what makes every git spawn exit 128 rather than read a repository.
_NO_SUCH_REPOSITORY = "no-such-repository.git"

# SQLite sidecars that belong to a store and must travel with a copy of it.
_SIDECAR_SUFFIXES: tuple[str, ...] = ("-wal", "-shm")

# The repo-identity sidecar `db_paths` writes beside a store (#1415). It
# travels with a copy too: the identity is otherwise derived from the path,
# and a copy at a new path would resolve a different one — which decides
# whether the project-context backfill considers itself due, and so whether
# the census sees a migration the original open would not have run.
_IDENTITY_SIDECAR_NAME = "identity"

# Statement classes the census reports. Anything outside this set lands in
# `other`, which is emitted so a new class cannot be absorbed silently.
# `BEGIN` and `COMMIT` are named rather than left to `other`: an open that
# writes wraps its writes in a transaction, and an unnamed transaction pair
# would make `other` fire with "a statement class this module does not
# classify" on the one case whose cause is known exactly.
_REPORTED_KINDS: tuple[str, ...] = (
    "CREATE",
    "SELECT",
    "PRAGMA",
    "BEGIN",
    "COMMIT",
)

# The classes that change rows. Counted separately from `rows_mutated`
# because they are a different claim: no row changed is compatible with an
# `UPDATE` that matched nothing, and the published figure is the stronger
# one — no such statement is issued at all.
_DML_KINDS: frozenset[str] = frozenset({"INSERT", "UPDATE", "DELETE", "REPLACE"})

# The components #1561's AC1 asks to be measured separately — the DDL
# battery, the migration check, the scope-id mint and the #1314 expired-lock
# sweep — plus `connection`, which is the set-up `PRAGMA`s that run before
# any of them. `connection` exists so the parts sum to the total: a
# decomposition with a residue nobody names is a total wearing four labels.
COMPONENTS: tuple[str, ...] = (
    "connection",
    "ddl",
    "migration_check",
    "scope_id",
    "sweep",
)

# The component a statement issued outside every instrumented span belongs
# to. Reached by the four `PRAGMA`s `MemoryStore.__init__` runs on the raw
# connection, and by nothing else on the paths measured here.
_UNSPANNED_COMPONENT = "connection"

# (component, dotted attribute of `aelfrice.store`) for every span. Resolved
# against the live module at instrumentation time, so a rename in the store
# raises here rather than silently emptying a component and publishing the
# zero.
#
# Attribution is by **execution nesting**, innermost span wins — not by
# statement index. `_resolve_local_scope_id` runs inside
# `_apply_open_schema` and keeps its own `SELECT`; `sweep_expired_locks`
# runs inside `_run_guarded_migration` and keeps its own writes. A
# positional slice of the trace would agree with this today and have to be
# re-derived by hand the first time a one-shot is added or reordered.
_SPAN_TARGETS: tuple[tuple[str, str], ...] = (
    # The schema battery proper: the stale-log probe, then every statement
    # of `_SCHEMA`, `_MIGRATIONS` and `_POST_MIGRATION_INDEXES`, all of
    # which reach the connection through `_execute_reprepare`.
    ("ddl", "_drop_stale_ingest_log"),
    ("ddl", "_execute_reprepare"),
    # What is left inside `_apply_open_schema` once the DDL spans above are
    # subtracted is the marker-gated work: the `store_generation` seed probe
    # and the origin-backfill marker. `_run_guarded_migration` wraps each
    # remaining one-shot.
    ("migration_check", "MemoryStore._apply_open_schema"),
    ("migration_check", "MemoryStore._run_guarded_migration"),
    ("scope_id", "MemoryStore._resolve_local_scope_id"),
    ("scope_id", "MemoryStore._read_only_scope_id"),
    ("sweep", "MemoryStore.sweep_expired_locks"),
)


@contextlib.contextmanager
def _hermetic_environment(tmp: Path) -> Iterator[None]:
    """Pin the environment this run reads through, for its duration.

    Two halves, and neither closes the hole alone. The `AELFRICE_` prefix is
    cleared as a class rather than a name list, so a resolver added later
    that reads an `AELFRICE_` variable is covered without an edit here —
    `AELFRICE_DB` above all, which decides which database every measurement
    below opens. The git location variables are enumerated, because a
    `$TMPDIR` inside a git work tree lets discovery ascend out of the
    fixture and find the enclosing repository.

    Cleared here rather than by the caller, and inside the function rather
    than at import scope. A module that clears the environment when it is
    imported breaks any CI gate that imports it for something else, which
    this repository has already paid for; a producer that publishes figures
    is the right place for its own hermeticity, and the right time is while
    it is running.
    """
    saved = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(ENV_PREFIX) or k in _GIT_LOCATION_VARS
    }
    for name in saved:
        del os.environ[name]
    os.environ["GIT_DIR"] = str(tmp / _NO_SUCH_REPOSITORY)
    os.environ["GIT_CEILING_DIRECTORIES"] = str(tmp.parent)
    cwd = os.getcwd()
    # A tempdir has no ancestor `.aelfrice.toml`, and this repository sits
    # under a home directory that has one.
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(cwd)
        for name in _GIT_LOCATION_VARS:
            os.environ.pop(name, None)
        # Every `AELFRICE_` name the block *set*, not only the ones it
        # saved. `figures()` and `main()` assign `AELFRICE_DB` inside this
        # block, so restoring `saved` alone leaves the process pointed at a
        # database inside a temporary directory that is about to be deleted
        # — and `MemoryStore` would recreate an empty store there for
        # whatever ran next.
        for name in [k for k in os.environ if k.startswith(ENV_PREFIX)]:
            if name not in saved:
                del os.environ[name]
        os.environ.update(saved)


class Statement(NamedTuple):
    """One traced statement, and the open component that issued it."""

    component: str
    sql: str


# Innermost open span, maintained by `_instrumented`. Module-level rather
# than threaded through, because the trace callback is invoked by SQLite
# from inside `MemoryStore.__init__` and has no other way to learn where it
# is. Nothing here is thread-safe and nothing here is threaded.
_SPAN_STACK: list[str] = []

# Per-frame accumulator of time spent in *nested* spans, so the wall clock
# reported per component is exclusive. Parallel to `_SPAN_STACK`.
_CHILD_SECONDS: list[float] = []


def _current_component() -> str:
    return _SPAN_STACK[-1] if _SPAN_STACK else _UNSPANNED_COMPONENT


@contextlib.contextmanager
def _instrumented() -> Iterator[dict[str, float]]:
    """Attribute every statement in the block to one open component.

    Each entry point in `_SPAN_TARGETS` is wrapped so that, while it runs,
    `_current_component()` names it. The trace callback installed by
    `_traced` reads that, which is what makes the census a decomposition
    rather than one aggregate by SQL keyword: AC1 asks for the DDL battery,
    the migration check, the scope-id mint and the #1314 sweep separately,
    and a leading-keyword count cannot tell them apart.

    Yields a component -> exclusive seconds mapping, filled as the block
    runs. Exclusive because a nested span's time is subtracted from its
    parent's: `_execute_reprepare` runs inside `_apply_open_schema`, and
    charging those statements to both would make the parts sum to more than
    the open.
    """
    store_module = importlib.import_module("aelfrice.store")

    elapsed = dict.fromkeys(COMPONENTS, 0.0)
    undo: list[tuple[Any, str, Any]] = []

    def _wrap(component: str, original: Any) -> Any:
        def span(*args: Any, **kwargs: Any) -> Any:
            _SPAN_STACK.append(component)
            _CHILD_SECONDS.append(0.0)
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                total = time.perf_counter() - start
                child = _CHILD_SECONDS.pop()
                _SPAN_STACK.pop()
                # `.get`, not `[]`. A component named here but absent from
                # `COMPONENTS` is a live defect — half a rename — and the
                # census's sum check is what reports it. A `KeyError` raised
                # from inside a span would instead be swallowed by
                # `_run_guarded_migration`, which degrades the pass rather
                # than failing, so the instrumentation would have changed
                # the behaviour it exists to observe.
                elapsed[component] = elapsed.get(component, 0.0) + total - child
                if _CHILD_SECONDS:
                    _CHILD_SECONDS[-1] += total

        return span

    for component, dotted in _SPAN_TARGETS:
        *path, name = dotted.split(".")
        holder: Any = store_module
        for step in path:
            holder = getattr(holder, step)
        if not hasattr(holder, name):
            raise AssertionError(
                f"aelfrice.store has no {dotted}: the {component} component "
                "would be published as zero because its entry point was "
                "renamed, not because it stopped issuing statements"
            )
        original = getattr(holder, name)
        undo.append((holder, name, original))
        setattr(holder, name, _wrap(component, original))

    depth = len(_SPAN_STACK)
    try:
        yield elapsed
    finally:
        for holder, name, original in reversed(undo):
            setattr(holder, name, original)
        del _SPAN_STACK[depth:]
        _CHILD_SECONDS.clear()


@contextlib.contextmanager
def _traced() -> Iterator[list[tuple[sqlite3.Connection, list[Statement]]]]:
    """Record every statement each connection opened in the block executes.

    `sqlite3.connect` is replaced for the duration rather than a callback
    installed on a handle already in hand, because the connection under
    measurement is created *inside* `MemoryStore.__init__` and the whole
    subject here is what that constructor does before it returns.

    The connections are held in the yielded list, not keyed by `id()`. An
    `id` is only unique among live objects, and a closed-and-collected
    handle's id is reusable — a census keyed on one can be attributed to the
    wrong open.

    Each statement is recorded with the component that was open when SQLite
    traced it. The callback reads `_current_component()` at trace time,
    which is inside the `execute()` that issued the statement, so the
    attribution is the call stack's and not a later reconstruction of it.
    """
    real_connect = sqlite3.connect
    seen: list[tuple[sqlite3.Connection, list[Statement]]] = []

    def wrapper(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)
        statements: list[Statement] = []
        conn.set_trace_callback(
            lambda sql: statements.append(
                Statement(_current_component(), sql)
            )
        )
        seen.append((conn, statements))
        return conn

    sqlite3.connect = wrapper  # type: ignore[assignment]
    try:
        yield seen
    finally:
        sqlite3.connect = real_connect  # type: ignore[assignment]


def _statements_for(
    seen: list[tuple[sqlite3.Connection, list[Statement]]],
    conn: sqlite3.Connection,
) -> list[Statement]:
    for candidate, statements in seen:
        if candidate is conn:
            return statements
    raise AssertionError(
        "the store's connection was not created inside the traced block; the "
        "census would be of nothing"
    )


def _kind(sql: str) -> str:
    words = sql.split()
    return words[0].upper() if words else ""


def _census(statements: list[Statement]) -> dict[str, int]:
    """Classify statements by leading keyword, and by issuing component.

    Two independent cuts of the same trace. `other` is emitted even when it
    is zero — a class this file does not name is the one thing a census must
    not absorb into a total silently — and so is every component, for the
    same reason.
    """
    kinds = Counter(_kind(s.sql) for s in statements if s.sql.split())
    out = {kind.lower(): kinds.get(kind, 0) for kind in _REPORTED_KINDS}
    out["statements"] = len(statements)
    out["dml_statements"] = sum(kinds.get(kind, 0) for kind in _DML_KINDS)
    out["other"] = out["statements"] - sum(
        out[kind.lower()] for kind in _REPORTED_KINDS
    ) - out["dml_statements"]
    # A `CREATE` that is not `IF NOT EXISTS` would be a statement that fails
    # rather than no-ops on the second open, so the two are not interchangeable
    # and the census says which it counted.
    out["creates_if_not_exists"] = sum(
        1
        for s in statements
        if _kind(s.sql) == "CREATE"
        and "IF NOT EXISTS" in " ".join(s.sql.split()).upper()
    )
    for component in COMPONENTS:
        member = [s for s in statements if s.component == component]
        out[f"{component}_statements"] = len(member)
        out[f"{component}_dml_statements"] = sum(
            1 for s in member if _kind(s.sql) in _DML_KINDS
        )
    # A decomposition whose parts do not sum to the whole is not one. This
    # is the guard on `_SPAN_TARGETS` drifting out of step with the store:
    # a component that stops being reached lands its statements in
    # `connection` and is caught by the census agreeing with itself, not by
    # a reader noticing a zero.
    parts = sum(out[f"{c}_statements"] for c in COMPONENTS)
    if parts != out["statements"]:
        raise AssertionError(
            f"the component census sums to {parts} but the open issued "
            f"{out['statements']} statements; a component is missing or "
            "double-counted, so none of the per-component figures can be "
            "published"
        )
    return out


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar_bytes(db: Path) -> int:
    total = 0
    for suffix in _SIDECAR_SUFFIXES:
        sidecar = db.with_name(db.name + suffix)
        if sidecar.exists():
            total += sidecar.stat().st_size
    return total


def build_fixture(db: Path, beliefs: int, *, seed: int = 1561) -> None:
    """Write a store of `beliefs` rows, opened the way production opens it.

    Built through `MemoryStore` with the same `project_context_default` that
    `db_paths._open_store` injects, so the rows land already stamped and the
    one-shot project-context backfill is complete before the measurement
    opens the store. Building the fixture any other way would measure a
    store mid-migration and publish 301.

    The locked rows carry `locked_at` and **no** `lock_expires_at`, which
    is the store's other settled condition: nothing is due for the #1314
    sweep, so the census measures the sweep's empty probe. That is a choice
    and not an oversight — the cost of a sweep that has work is measured on
    its own fixture, built by back-dating a lock through `expire_one_lock`,
    and published under the `expired_lock_*` keys. A fixture that could
    only produce the empty case would publish the component AC1 calls the
    load-bearing one as a structural zero.

    Deterministic: content is a function of the row index and `seed`, so the
    same arguments always produce the same store.
    """
    from aelfrice.db_paths import repo_identity_from_db_path
    from aelfrice.models import LOCK_NONE, LOCK_USER, Belief
    from aelfrice.store import MemoryStore

    db.parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(
        str(db),
        project_context_default=repo_identity_from_db_path(db, create=True),
        # Spelled out rather than left to the default, because
        # `tests/test_readonly_diagnostics_1328.py` requires every
        # `MemoryStore(...)` in a module that names `AELFRICE_DB` to declare
        # which mode it wants. This is the deliberate write mode that guard
        # sanctions: `db` is a path under a temporary directory this process
        # created, and building a fixture is the one thing here that must
        # write. Every open under measurement is elsewhere, and the only
        # database a caller can name reaches this function as a copy.
        read_only=False,
    )
    try:
        for index in range(beliefs):
            content = (
                f"belief {index} seed {seed} "
                + "lorem ipsum dolor sit amet consectetur " * 3
            )
            locked = index % 50 == 0
            store.insert_belief(
                Belief(
                    id=hashlib.sha256(
                        f"{index}:{seed}".encode()
                    ).hexdigest()[:16],
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
    finally:
        store.close()


def expire_one_lock(db: Path, *, when: str = "2020-01-01T00:00:00+00:00") -> int:
    """Back-date one user lock's expiry, and return the rows it makes due.

    Written with a plain `sqlite3` connection rather than through
    `MemoryStore`, because opening the store is the thing under measurement
    and an open here would run the sweep this is setting up.

    The point of this fixture is that `build_fixture` cannot produce it.
    `build_fixture` sets `locked_at` and never `lock_expires_at`, so every
    store the census is taken on has nothing due and measures the #1314
    sweep at zero *by construction*. AC1 names that sweep as the one
    component that must run for correctness of what observational commands
    print, so measuring only the empty case would publish the component
    this issue most needs a number for as a structural zero.
    """
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "UPDATE beliefs SET lock_expires_at = ? WHERE id = "
            "(SELECT id FROM beliefs WHERE lock_level = 'user' "
            " ORDER BY id LIMIT 1)",
            (when,),
        )
        conn.commit()
        row = conn.execute(
            "SELECT count(*) FROM beliefs WHERE lock_expires_at IS NOT NULL "
            "AND lock_expires_at <= ? AND lock_level = 'user'",
            (when,),
        ).fetchone()
    finally:
        conn.close()
    due = int(row[0]) if row else 0
    if due != 1:
        raise AssertionError(
            f"expected exactly one lock due to expire, found {due}; the "
            "fixture has no user lock to back-date and the sweep arm would "
            "measure the empty case a second time"
        )
    return due


def measure_open(db: Path) -> dict[str, Any]:
    """Census both arms of one store's open. Assumes `AELFRICE_DB` is `db`.

    The writable arm is `open_store_for_read()` itself, not a direct
    `MemoryStore(...)`: the figures are about the function the observational
    commands call, including its identity resolution and its fallback
    attempt, and a hand-rolled equivalent would be a different code path
    wearing the same name.

    The read-only arm is the counterfactual floor rather than a proposal.
    It is what `mode=ro` costs, so the difference between the two arms is
    the whole of what any fix to the open order could recover.
    """
    from aelfrice.db_paths import open_store_for_read
    from aelfrice.store import MemoryStore

    before_digest = _digest(db)
    before_sidecars = _sidecar_bytes(db)

    with _instrumented() as writable_seconds, _traced() as seen:
        store = open_store_for_read()
        try:
            writable = _census(_statements_for(seen, store._conn))
            writable["rows_mutated"] = store._conn.total_changes
            writable["read_only_handle"] = int(store.read_only)
        finally:
            store.close()

    after_digest = _digest(db)
    writable["db_bytes_changed"] = int(after_digest != before_digest)
    writable["sidecar_bytes_before"] = before_sidecars
    writable["sidecar_bytes_after"] = _sidecar_bytes(db)

    with _instrumented(), _traced() as seen:
        handle = MemoryStore(str(db), read_only=True)
        try:
            readonly = _census(_statements_for(seen, handle._conn))
            readonly["rows_mutated"] = handle._conn.total_changes
        finally:
            handle.close()

    return {
        "db_bytes": db.stat().st_size,
        "writable": writable,
        "readonly": readonly,
        # Wall clock, so never emitted — see this module's docstring. Kept
        # on the result because AC1 asks for the components separated by
        # wall clock as well as by write volume, and the report prints it.
        "writable_component_ms": {
            component: seconds * 1000.0
            for component, seconds in writable_seconds.items()
        },
    }


def time_opens(db: Path, repeats: int = TIMING_REPEATS) -> dict[str, float]:
    """Median milliseconds for each arm, and the difference between them.

    Reported, never emitted. See this module's docstring: a wall-clock
    reading is a property of the machine that took it, and no runner can
    re-derive another's.
    """
    from aelfrice.db_paths import open_store_for_read
    from aelfrice.store import MemoryStore

    writable: list[float] = []
    readonly: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        store = open_store_for_read()
        writable.append((time.perf_counter() - start) * 1000)
        store.close()
        start = time.perf_counter()
        handle = MemoryStore(str(db), read_only=True)
        readonly.append((time.perf_counter() - start) * 1000)
        handle.close()
    writable_ms = statistics.median(writable)
    readonly_ms = statistics.median(readonly)
    return {
        "writable_open_ms": writable_ms,
        "readonly_open_ms": readonly_ms,
        "avoidable_ms": writable_ms - readonly_ms,
    }


def time_stats_command(repeats: int = 7) -> float:
    """Median milliseconds for the whole `aelf stats` command, in process.

    The denominator the avoidable component is a share *of*. In process
    rather than as a subprocess on purpose: a subprocess reading is mostly
    interpreter start-up and import, which is not work the store open can
    be a share of, and quoting the avoidable component against it flatters
    the answer by an order of magnitude. Both denominators are stated in
    the entry this module produces; this is the honest one.
    """
    import io

    from aelfrice.cli import main as cli_main

    readings: list[float] = []
    for _ in range(repeats):
        buffer = io.StringIO()
        start = time.perf_counter()
        with contextlib.redirect_stdout(buffer):
            cli_main(["stats"])
        readings.append((time.perf_counter() - start) * 1000)
    return statistics.median(readings)


def figures(*, grid: tuple[int, ...] = SIZE_GRID) -> dict[str, Any]:
    """Re-derive every #1561 figure this repo publishes.

    Raises rather than averaging when the grid disagrees. A census that
    moves with store size is not a schema property, and publishing a mean
    of three readings that disagree would be exactly the class of figure
    `scripts/check_derived_figures.py` exists to stop.
    """
    if not grid:
        raise ValueError("an empty grid would emit a figure measured on nothing")

    values: dict[str, Any] = {"beliefs_grid": list(grid)}
    with tempfile.TemporaryDirectory(prefix="store-open-cost-") as td:
        tmp = Path(td)
        with _hermetic_environment(tmp):
            per_size: dict[int, dict[str, Any]] = {}
            for beliefs in grid:
                db = tmp / f"n{beliefs}" / REPO_STORE_DIRNAME / "memory.db"
                build_fixture(db, beliefs)
                os.environ["AELFRICE_DB"] = str(db)
                per_size[beliefs] = measure_open(db)

            largest = per_size[grid[-1]]
            for arm in ("writable", "readonly"):
                for key, value in largest[arm].items():
                    differing = {
                        beliefs: per_size[beliefs][arm][key]
                        for beliefs in grid
                        if per_size[beliefs][arm][key] != value
                    }
                    if differing:
                        raise AssertionError(
                            f"{arm}.{key} is not a schema property: it reads "
                            f"{value} at {grid[-1]} beliefs and "
                            f"{differing} elsewhere. A count that moves with "
                            "the corpus cannot be published as one figure."
                        )

            writable = largest["writable"]
            readonly = largest["readonly"]
            if writable["read_only_handle"]:
                raise AssertionError(
                    "the fixture fell back to the read-only handle, so the "
                    "writable arm measured the fallback and not the open "
                    "this issue is about"
                )
            values["statements"] = writable["statements"]
            values["creates"] = writable["create"]
            values["selects"] = writable["select"]
            values["pragmas"] = writable["pragma"]
            values["creates_if_not_exists"] = writable["creates_if_not_exists"]
            values["other_statements"] = writable["other"]
            values["dml_statements"] = writable["dml_statements"]
            values["rows_mutated"] = writable["rows_mutated"]
            values["db_bytes_changed"] = writable["db_bytes_changed"]
            values["readonly_statements"] = readonly["statements"]
            values["avoidable_statements"] = (
                writable["statements"] - readonly["statements"]
            )

            # AC1: the four components separately, plus the connection
            # set-up, so a reader deciding AC2 can see which part of the
            # total a short-cut would be short-cutting. Emitted for every
            # component including the ones that read zero — a component
            # that has become free is a fact about the open, and dropping
            # its key would make that indistinguishable from the component
            # having been removed from the measurement.
            for component in COMPONENTS:
                values[f"{component}_statements"] = writable[
                    f"{component}_statements"
                ]
                values[f"{component}_dml_statements"] = writable[
                    f"{component}_dml_statements"
                ]

            # The sweep, on a store that has something for it to do. Every
            # arm above has nothing due, so every `sweep_statements` above
            # is the empty probe. This arm is the one that says what the
            # component actually costs, and it is also the counterexample
            # to reading the zero-mutation figures as unconditional: an
            # ordinary user store that has ever held a time-boxed lock
            # reaches this on the next `aelf search`.
            due = tmp / "due" / REPO_STORE_DIRNAME / "memory.db"
            build_fixture(due, grid[0])
            values["expired_locks_due"] = expire_one_lock(due)
            os.environ["AELFRICE_DB"] = str(due)
            due_measured = measure_open(due)["writable"]
            if due_measured["read_only_handle"]:
                raise AssertionError(
                    "the expired-lock fixture fell back to the read-only "
                    "handle, which runs no sweep at all"
                )
            values["expired_lock_statements"] = due_measured["statements"]
            values["expired_lock_sweep_statements"] = due_measured[
                "sweep_statements"
            ]
            values["expired_lock_dml_statements"] = due_measured[
                "dml_statements"
            ]
            values["expired_lock_rows_mutated"] = due_measured["rows_mutated"]
            values["expired_lock_db_bytes_changed"] = due_measured[
                "db_bytes_changed"
            ]

            # The same open against a database outside the repo-store layout.
            # `repo_identity_from_db_path` returns '' there, which skips the
            # project-context backfill's marker probe, so this arm is one
            # SELECT cheaper. Published beside the repo-store figure because
            # a census taken on the wrong layout is the likeliest way for a
            # re-derivation of this issue to disagree with itself.
            bare = tmp / "bare" / "memory.db"
            build_fixture(bare, grid[0])
            os.environ["AELFRICE_DB"] = str(bare)
            bare_writable = measure_open(bare)["writable"]
            if bare_writable["read_only_handle"]:
                raise AssertionError(
                    "the no-repo-identity fixture fell back to the read-only "
                    "handle, so this arm measured `mode=ro` and the "
                    "one-statement gap it publishes is not the one described"
                )
            values["no_repo_identity_statements"] = bare_writable["statements"]
            values["no_repo_identity_rows_mutated"] = bare_writable[
                "rows_mutated"
            ]
    return values


def _report(db: Path, *, label: str) -> str:
    measured = measure_open(db)
    timings = time_opens(db)
    stats_ms = time_stats_command()
    writable = measured["writable"]
    readonly = measured["readonly"]
    avoidable = writable["statements"] - readonly["statements"]
    share = 100.0 * timings["avoidable_ms"] / stats_ms if stats_ms else 0.0
    wall_clock_heading = (
        "wall clock on THIS machine — not published, not re-derivable "
        + "elsewhere:"
    )
    lines = [
        f"store: {label}",
        f"  file bytes                {measured['db_bytes']}",
        "",
        "writable-first open (what every observational command pays):",
        f"  statements                {writable['statements']}",
        f"    CREATE                  {writable['create']} "
        f"({writable['creates_if_not_exists']} IF NOT EXISTS)",
        f"    SELECT                  {writable['select']}",
        f"    PRAGMA                  {writable['pragma']}",
        f"    INSERT/UPDATE/DELETE    {writable['dml_statements']}",
        f"    unclassified            {writable['other']}",
        f"  rows mutated              {writable['rows_mutated']}",
        f"  database bytes changed    {bool(writable['db_bytes_changed'])}",
        f"  handle was read-only      {bool(writable['read_only_handle'])}",
        "",
        "by component (#1561 AC1) — statements, writes, exclusive ms:",
    ]
    for component in COMPONENTS:
        lines.append(
            f"  {component:<24}{writable[f'{component}_statements']:>4}"
            f"{writable[f'{component}_dml_statements']:>6}"
            f"{measured['writable_component_ms'][component]:>10.3f}"
        )
    lines += [
        "",
        "read-only open (the floor any fix could reach):",
        f"  statements                {readonly['statements']}",
        f"  rows mutated              {readonly['rows_mutated']}",
        "",
        f"avoidable statements        {avoidable}",
        "",
        wall_clock_heading,
        f"  writable open median      {timings['writable_open_ms']:.3f} ms",
        f"  read-only open median     {timings['readonly_open_ms']:.3f} ms",
        f"  avoidable median          {timings['avoidable_ms']:.3f} ms",
        f"  `aelf stats` in process   {stats_ms:.1f} ms",
        f"  avoidable share           {share:.3f}%",
    ]
    return "\n".join(lines)


def _copy_store(source: Path, into: Path) -> Path:
    """Copy `source` and its sidecars into `into`, and return the copy.

    Not a convenience. Opening a store is a write — the DDL battery, the
    migrations, the scope-id mint and the expired-lock sweep all run — so a
    module that opened the operator's named store to measure the cost of
    opening it would mutate the corpus it was measuring, which is #1328
    exactly.
    """
    if not source.is_file():
        raise SystemExit(f"--store: no such database: {source}")
    into.mkdir(parents=True, exist_ok=True)
    copy = into / source.name
    shutil.copy2(source, copy)
    for suffix in _SIDECAR_SUFFIXES:
        sidecar = source.with_name(source.name + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, copy.with_name(copy.name + suffix))
    identity = source.parent / _IDENTITY_SIDECAR_NAME
    if identity.is_file():
        shutil.copy2(identity, copy.parent / _IDENTITY_SIDECAR_NAME)
    return copy


def main(argv: list[str] | None = None) -> int:
    """Emit the figures, or print a report over a synthetic or named store."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument(
        "--emit-figures",
        action="store_true",
        help=(
            "emit a flat JSON object of key -> value on stdout and nothing "
            "else, measured on fixtures this script builds"
        ),
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="print a report over a fixture this script builds",
    )
    parser.add_argument(
        "--store",
        metavar="PATH",
        help=(
            "print a report over a COPY of this database; there is no "
            "default and no live path is ever resolved (#1328)"
        ),
    )
    parser.add_argument(
        "--beliefs",
        type=int,
        default=DEFAULT_BELIEFS,
        help=f"rows in the synthetic fixture (default {DEFAULT_BELIEFS})",
    )
    args = parser.parse_args(argv)

    if args.emit_figures and (args.store or args.synthetic):
        parser.error(
            "--emit-figures publishes figures from the fixture this script "
            "builds; it does not take a store, because a published figure "
            "has to be re-derivable by whoever reads it"
        )
    if not args.emit_figures and not args.synthetic and not args.store:
        parser.error(
            "choose a mode: --emit-figures, --synthetic, or --store PATH. "
            "--store has no default on purpose — a benchmark that resolved "
            "the live store would open it, and opening a store is a write "
            "(#1328)"
        )

    if args.emit_figures:
        json.dump(figures(), sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    with tempfile.TemporaryDirectory(prefix="store-open-cost-") as td:
        tmp = Path(td)
        with _hermetic_environment(tmp):
            if args.store:
                source = Path(args.store).expanduser()
                db = _copy_store(source, tmp / "copy" / REPO_STORE_DIRNAME)
                label = f"{source} (measured on a copy)"
            else:
                db = tmp / "synthetic" / REPO_STORE_DIRNAME / "memory.db"
                build_fixture(db, args.beliefs)
                label = f"synthetic, {args.beliefs} beliefs"
            os.environ["AELFRICE_DB"] = str(db)
            print(_report(db, label=label))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
