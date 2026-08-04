"""SQLITE_SCHEMA ("database schema has changed") retry on store open (#1310).

`MemoryStore.__init__` runs a DDL + seed + backfill battery. When a
second process commits DDL between one of those statements' prepare and
step, SQLite raises `OperationalError: database schema has changed`.
`busy_timeout` does not cover it — that pragma retries `database is
locked`, a different error — so the open failed outright and took a
required CI check down with it at random.

What these tests prove, and what they do not:

- They prove the retry is **wired**: an injected SQLITE_SCHEMA error at
  a DDL statement, at a bare `schema_meta` read, and at the scope-id
  write all leave `MemoryStore(...)` constructing successfully, and a
  non-schema `OperationalError` still propagates.
- They do **not** prove the CI flake is gone. Real-concurrency
  reproduction was attempted and did not fire (0/180 rounds locally), so
  a race-based guard would assert nothing. Fault injection is the only
  arm that distinguishes the fixed code from the broken code
  deterministically.

Injection works by wrapping the real connection. Two mechanics matter:

- `__getattr__` is not consulted for dunder lookups on new-style
  classes, so the context-manager protocol (`with self._conn:`, used by
  a one-shot migration) is forwarded explicitly.
- `aelfrice.store.sqlite3` *is* the stdlib `sqlite3` module, so patching
  `connect` through it is process-global. Every patch here is scoped to
  a single `MemoryStore(...)` call and restored in a `finally`.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from aelfrice import store as store_mod
from aelfrice.store import (
    MemoryStore,
    _execute_reprepare,
    _retry_on_schema_change,
)

_SCHEMA_ERR = "database schema has changed"


class _FaultingConnection:
    """Wrap a real connection; raise `error` when `match` hits.

    Fires once by default (or on every matching call when `once=False`),
    so a retry that re-prepares the same statement sees a success on its
    second attempt. `fired` is the arm that keeps a test from passing
    vacuously because the injection never triggered.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        match: Callable[[str], bool],
        error: Exception,
        *,
        once: bool = True,
    ) -> None:
        # Bypass __setattr__, which forwards to the wrapped connection.
        object.__setattr__(
            self,
            "_state",
            {
                "conn": conn,
                "match": match,
                "error": error,
                "once": once,
                "fired": 0,
            },
        )

    @property
    def fired(self) -> int:
        return int(self._state["fired"])

    def execute(self, sql: str, *args: object, **kwargs: object) -> object:
        state = self._state
        already = int(state["fired"])
        if state["match"](sql) and not (state["once"] and already):
            state["fired"] = already + 1
            raise state["error"]
        return state["conn"].execute(sql, *args, **kwargs)

    # Dunders are looked up on the type, so __getattr__ below never sees
    # them. `with self._conn:` in a one-shot migration needs these.
    def __enter__(self) -> object:
        return self._state["conn"].__enter__()

    def __exit__(self, *exc: object) -> object:
        return self._state["conn"].__exit__(*exc)

    def __getattr__(self, name: str) -> object:
        return getattr(self._state["conn"], name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(self._state["conn"], name, value)


@contextmanager
def _faulting_connect(
    match: Callable[[str], bool],
    error: Exception,
    *,
    once: bool = True,
) -> Iterator[list[_FaultingConnection]]:
    """Patch `sqlite3.connect` to hand back a faulting wrapper.

    PROCESS-GLOBAL: `aelfrice.store.sqlite3` is the stdlib module, not a
    module-local alias. Keep the body to the one `MemoryStore(...)` call
    under test; the original is restored unconditionally.
    """
    real = sqlite3.connect
    made: list[_FaultingConnection] = []

    def fake(*args: object, **kwargs: object) -> _FaultingConnection:
        wrapper = _FaultingConnection(
            real(*args, **kwargs), match, error, once=once
        )
        made.append(wrapper)
        return wrapper

    store_mod.sqlite3.connect = fake  # type: ignore[assignment]
    try:
        yield made
    finally:
        store_mod.sqlite3.connect = real  # type: ignore[assignment]


def _startswith(prefix: str) -> Callable[[str], bool]:
    return lambda sql: sql.strip().upper().startswith(prefix.upper())


# --- the helpers themselves -------------------------------------------


def test_retry_on_schema_change_retries_then_returns() -> None:
    calls: list[int] = []

    def op() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError(_SCHEMA_ERR)
        return "ok"

    assert _retry_on_schema_change(op) == "ok"
    assert len(calls) == 2


def test_retry_on_schema_change_reraises_other_operational_errors() -> None:
    """A malformed statement must stay loud, not be retried into silence."""
    calls: list[int] = []

    def op() -> None:
        calls.append(1)
        raise sqlite3.OperationalError('near "CRATE": syntax error')

    with pytest.raises(sqlite3.OperationalError, match="CRATE"):
        _retry_on_schema_change(op)
    assert len(calls) == 1, "non-schema errors must not be retried"


def test_retry_on_schema_change_is_bounded() -> None:
    """A schema that keeps changing fails rather than spinning forever."""
    calls: list[int] = []

    def op() -> None:
        calls.append(1)
        raise sqlite3.OperationalError(_SCHEMA_ERR)

    with pytest.raises(sqlite3.OperationalError, match="schema has changed"):
        _retry_on_schema_change(op, attempts=3)
    assert len(calls) == 3


def test_execute_reprepare_reruns_the_statement(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "x.db"))
    wrapper = _FaultingConnection(
        conn,
        _startswith("CREATE TABLE"),
        sqlite3.OperationalError(_SCHEMA_ERR),
    )
    _execute_reprepare(  # type: ignore[arg-type]
        wrapper, "CREATE TABLE t (a INTEGER)"
    )
    assert wrapper.fired == 1
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    assert [r[0] for r in rows] == ["t"]
    conn.close()


# --- the wiring, through MemoryStore.__init__ --------------------------


@pytest.mark.parametrize(
    ("label", "match"),
    [
        # The _SCHEMA loop — the site the traceback in #1310 named.
        ("schema_ddl", _startswith("CREATE TABLE")),
        # A bare read the issue's narrower patch would have left exposed.
        ("generation_read", _startswith("SELECT 1 FROM schema_meta")),
        # _resolve_local_scope_id's write, the last statement in the window.
        ("scope_id_write", _startswith("INSERT OR REPLACE INTO schema_meta")),
    ],
)
def test_open_survives_injected_schema_change(
    tmp_path: Path, label: str, match: Callable[[str], bool]
) -> None:
    db = str(tmp_path / f"{label}.db")
    err = sqlite3.OperationalError(_SCHEMA_ERR)
    with _faulting_connect(match, err) as made:
        store = MemoryStore(db)
    assert made, "connect was never patched"
    assert made[0].fired == 1, "injection never fired — test is vacuous"
    assert store.local_scope_id
    store.close()

    # The store is usable and the battery left the canonical schema.
    plain = MemoryStore(db)
    assert plain.get_schema_meta("local_scope_id")
    plain.close()


def test_open_still_raises_on_a_non_schema_error(tmp_path: Path) -> None:
    """AC2: a genuinely broken statement is not retried into silence."""
    db = str(tmp_path / "bad.db")
    with _faulting_connect(
        _startswith("CREATE TABLE"),
        sqlite3.OperationalError('near "CRATE": syntax error'),
    ):
        with pytest.raises(sqlite3.OperationalError, match="CRATE"):
            MemoryStore(db)


def test_open_gives_up_on_a_persistently_changing_schema(
    tmp_path: Path,
) -> None:
    """Bounded attempts: a permanent SQLITE_SCHEMA fails, it does not hang."""
    db = str(tmp_path / "spin.db")
    with _faulting_connect(
        _startswith("CREATE TABLE"),
        sqlite3.OperationalError(_SCHEMA_ERR),
        once=False,
    ) as made:
        with pytest.raises(
            sqlite3.OperationalError, match="schema has changed"
        ):
            MemoryStore(db)
    # 3 outer window attempts x 3 statement-level attempts, not unbounded.
    assert made[0].fired <= 9
