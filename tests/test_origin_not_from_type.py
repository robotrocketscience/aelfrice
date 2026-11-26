"""#2013: the origin backfill and `aelf migrate` don't set origin from the type.

Two writers used to give every `correction` whose origin is `unknown` the
origin `user_corrected`: the one-shot origin backfill in `store.py`, and the
row conversion in `migrate.py`. Nothing shows that the user made those
corrections. Both writers keep the lock rule: a locked belief whose origin
is `unknown` becomes `user_stated`.

The opt-in Stop-hook auto-lock (`AELF_AUTOLOCK_CORRECTIONS=1`) still picks
beliefs of type `correction` and sets their origin to `user_stated`. #2022
tracks it.

The tests drive a real store file. The backfill runs only on a writable
open of a store that has no `origin_backfill_complete` marker, so each
store test deletes that row first.
"""
from __future__ import annotations

import ast
import io
import re
import sqlite3
from pathlib import Path

import pytest

from aelfrice import models
from aelfrice.cli import main
from aelfrice.models import (
    BELIEF_CORRECTION,
    BELIEF_FACTUAL,
    BELIEF_TYPES,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_UNKNOWN,
    ORIGIN_USER_CORRECTED,
    ORIGIN_USER_STATED,
    Belief,
)
from aelfrice.store import SCHEMA_META_ORIGIN_BACKFILL, MemoryStore

_SRC = Path(__file__).resolve().parents[1] / "src" / "aelfrice"

UNLOCKED_CORRECTION = "c0rrect10n00001"
LOCKED_FACTUAL = "l0cked0fact0001"
LOCKED_CORRECTION = "l0cked0c0rr0001"
ALREADY_CORRECTED = "a1readyc0rr0001"


def _belief(bid: str, btype: str, *, locked: bool, origin: str) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid} for the #2013 origin test",
        content_hash=f"h-{bid}",
        alpha=9.0 if locked else 1.0,
        beta=0.5,
        type=btype,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-01-01T00:00:00+00:00" if locked else None,
        created_at="2026-01-01T00:00:00+00:00",
        last_retrieved_at=None,
        origin=origin,
    )


def _seed(db: Path) -> None:
    """Write a store whose rows look like a store from before the backfill.

    The rows are inserted with an explicit origin, then set back to
    `unknown` with raw SQL, and the backfill marker is deleted. The next
    writable open then runs the backfill on these rows.
    """
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_belief(
            UNLOCKED_CORRECTION, BELIEF_CORRECTION,
            locked=False, origin=ORIGIN_UNKNOWN,
        ))
        store.insert_belief(_belief(
            LOCKED_FACTUAL, BELIEF_FACTUAL,
            locked=True, origin=ORIGIN_USER_STATED,
        ))
        store.insert_belief(_belief(
            LOCKED_CORRECTION, BELIEF_CORRECTION,
            locked=True, origin=ORIGIN_USER_STATED,
        ))
        store.insert_belief(_belief(
            ALREADY_CORRECTED, BELIEF_CORRECTION,
            locked=False, origin=ORIGIN_USER_CORRECTED,
        ))
    finally:
        store.close()
    raw = sqlite3.connect(str(db))
    try:
        raw.execute(
            "UPDATE beliefs SET origin = ? WHERE id IN (?, ?, ?)",
            (ORIGIN_UNKNOWN, UNLOCKED_CORRECTION, LOCKED_FACTUAL,
             LOCKED_CORRECTION),
        )
        raw.execute(
            "DELETE FROM schema_meta WHERE key = ?",
            (SCHEMA_META_ORIGIN_BACKFILL,),
        )
        raw.commit()
    finally:
        raw.close()


def _raw_origins(db: Path) -> dict[str, str]:
    raw = sqlite3.connect(str(db))
    try:
        return dict(raw.execute("SELECT id, origin FROM beliefs").fetchall())
    finally:
        raw.close()


def _raw_marker(db: Path) -> str | None:
    raw = sqlite3.connect(str(db))
    try:
        row = raw.execute(
            "SELECT value FROM schema_meta WHERE key = ?",
            (SCHEMA_META_ORIGIN_BACKFILL,),
        ).fetchone()
    finally:
        raw.close()
    return None if row is None else str(row[0])


@pytest.fixture
def seeded_db(tmp_path: Path) -> Path:
    db = tmp_path / "pre_backfill.db"
    _seed(db)
    origins = _raw_origins(db)
    # Confirms the seed, so a later pass proves the open did the work.
    assert origins[UNLOCKED_CORRECTION] == ORIGIN_UNKNOWN
    assert origins[LOCKED_FACTUAL] == ORIGIN_UNKNOWN
    assert origins[LOCKED_CORRECTION] == ORIGIN_UNKNOWN
    assert origins[ALREADY_CORRECTED] == ORIGIN_USER_CORRECTED
    assert _raw_marker(db) is None
    return db


def _open_writable(db: Path) -> None:
    MemoryStore(str(db)).close()


# --- store backfill -------------------------------------------------------


def test_backfill_leaves_unlocked_correction_unknown(seeded_db: Path) -> None:
    _open_writable(seeded_db)
    assert _raw_origins(seeded_db)[UNLOCKED_CORRECTION] == ORIGIN_UNKNOWN


def test_backfill_writes_its_marker(seeded_db: Path) -> None:
    _open_writable(seeded_db)
    assert _raw_marker(seeded_db) is not None


def test_backfill_marks_locked_unknown_user_stated(seeded_db: Path) -> None:
    _open_writable(seeded_db)
    origins = _raw_origins(seeded_db)
    assert origins[LOCKED_FACTUAL] == ORIGIN_USER_STATED
    assert origins[LOCKED_CORRECTION] == ORIGIN_USER_STATED


def test_backfill_keeps_existing_user_corrected(seeded_db: Path) -> None:
    _open_writable(seeded_db)
    assert _raw_origins(seeded_db)[ALREADY_CORRECTED] == ORIGIN_USER_CORRECTED


def test_read_only_open_runs_no_backfill(seeded_db: Path) -> None:
    store = MemoryStore(str(seeded_db), read_only=True)
    try:
        assert store.read_only
    finally:
        store.close()
    origins = _raw_origins(seeded_db)
    # The locked row is the one the backfill would change after the fix,
    # so it is the row that shows whether the backfill ran.
    assert origins[LOCKED_FACTUAL] == ORIGIN_UNKNOWN
    assert origins[UNLOCKED_CORRECTION] == ORIGIN_UNKNOWN
    assert _raw_marker(seeded_db) is None


# --- aelf migrate -----------------------------------------------------------


@pytest.fixture
def migrated_origins(
    seeded_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, str]:
    """Run `aelf migrate --apply --all` from `seeded_db` into a new store.

    The source is read through a read-only connection, so its rows keep
    origin `unknown`. The target store is new: its first open stamps the
    marker before any row lands, so the target's own backfill cannot set
    an origin on the copied rows. Only the migrate conversion can.
    """
    target = tmp_path / "target" / "memory.db"
    target.parent.mkdir()
    monkeypatch.setenv("AELFRICE_DB", str(target))
    out = io.StringIO()
    code = main(
        ["migrate", "--from", str(seeded_db), "--apply", "--all", "--yes"],
        out=out,
    )
    assert code == 0, out.getvalue()
    assert _raw_origins(seeded_db)[UNLOCKED_CORRECTION] == ORIGIN_UNKNOWN
    origins = _raw_origins(target)
    assert set(origins) == {
        UNLOCKED_CORRECTION, LOCKED_FACTUAL, LOCKED_CORRECTION,
        ALREADY_CORRECTED,
    }, out.getvalue()
    return origins


def test_migrate_leaves_unlocked_correction_unknown(
    migrated_origins: dict[str, str],
) -> None:
    assert migrated_origins[UNLOCKED_CORRECTION] == ORIGIN_UNKNOWN


def test_migrate_marks_locked_unknown_user_stated(
    migrated_origins: dict[str, str],
) -> None:
    assert migrated_origins[LOCKED_FACTUAL] == ORIGIN_USER_STATED
    assert migrated_origins[LOCKED_CORRECTION] == ORIGIN_USER_STATED


def test_migrate_keeps_existing_user_corrected(
    migrated_origins: dict[str, str],
) -> None:
    assert migrated_origins[ALREADY_CORRECTED] == ORIGIN_USER_CORRECTED


# --- source scan ------------------------------------------------------------

# A SQL origin write keyed on the type column. The pattern matches an
# `UPDATE beliefs ... SET ... origin = ...` statement that, before the
# statement ends at `;`, compares a column named exactly `type` with `=`,
# `==`, or `IN`. `\btype\b` does not match `content_type` or `edge_type`,
# because `_` is a word character. The lock rule (`WHERE origin = 'unknown'
# AND lock_level = 'user'`) has no `type` column, so it does not match.
_SQL_ORIGIN_FROM_TYPE = re.compile(
    r"\bUPDATE\s+beliefs\s+SET\b[^;]*?\borigin\s*=[^;]*?"
    r"\btype\b\s*(?:==|=|\bIN\b)",
    re.IGNORECASE | re.DOTALL,
)

# The names in models.py that hold a belief type, such as BELIEF_CORRECTION.
# Scope names such as BELIEF_SCOPE_PROJECT share the prefix but hold no
# type, so the set lists only names whose value is in BELIEF_TYPES.
_BELIEF_TYPE_NAMES: frozenset[str] = frozenset(
    name for name, value in vars(models).items()
    if name.startswith("BELIEF_") and isinstance(value, str)
    and value in BELIEF_TYPES
)


def _reads_belief_type(node: ast.AST) -> bool:
    """True when `node` reads a belief's type.

    A read is one of `x.type`, `x["type"]`, a name `belief_type`, or a
    name from `_BELIEF_TYPE_NAMES`. A call to the builtin `type()` is a
    name `type`, not any of these, so `type(x) is dict` does not count.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr == "type":
            return True
        if (
            isinstance(sub, ast.Subscript)
            and isinstance(sub.slice, ast.Constant)
            and sub.slice.value == "type"
        ):
            return True
        if isinstance(sub, ast.Name) and (
            sub.id == "belief_type" or sub.id in _BELIEF_TYPE_NAMES
        ):
            return True
    return False


def _is_origin_target(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "origin"
    if isinstance(node, ast.Attribute):
        return node.attr == "origin"
    if isinstance(node, ast.Subscript):
        return (
            isinstance(node.slice, ast.Constant)
            and node.slice.value == "origin"
        )
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_is_origin_target(elt) for elt in node.elts)
    return False


def _origin_write_value(node: ast.AST) -> ast.expr | None:
    """The value `node` writes to an origin, or None when it writes none.

    A write is an assignment whose target is an origin, or an `origin=`
    keyword in a call. An annotated assignment with no value writes none.
    """
    targets: list[ast.expr] = []
    value: ast.expr | None = None
    if isinstance(node, ast.Assign):
        targets, value = node.targets, node.value
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets, value = [node.target], node.value
    elif isinstance(node, ast.keyword) and node.arg == "origin":
        return node.value
    if any(_is_origin_target(t) for t in targets):
        return value
    return None


def _line(node: ast.AST) -> int:
    """The line `node` starts on. Every node the scan reports has one."""
    return int(getattr(node, "lineno", 0))


def _origin_writes(body: list[ast.stmt]) -> list[ast.AST]:
    """Every assignment to an origin, and every `origin=` keyword, in `body`.

    Nested statements count, so a write inside a nested block of an
    `if` body is found. An `elif` lives in the `orelse` of the `if`, not
    its body, so each branch is checked against its own test.
    """
    return [
        sub for stmt in body for sub in ast.walk(stmt)
        if _origin_write_value(sub) is not None
    ]


def _origin_from_type_sites(tree: ast.AST) -> list[int]:
    """Line numbers where Python code sets an origin from a belief type.

    A site is one of these:

    - An `if` whose test reads a belief type and whose body writes an
      origin, as in `elif row["type"] == BELIEF_CORRECTION: origin = ...`.
    - An origin write whose value reads a belief type, as in
      `origin = X if b.type == BELIEF_CORRECTION else Y`, or
      `origin=TYPE_TO_ORIGIN[b.type]`.
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _reads_belief_type(node.test):
            lines.extend(_line(w) for w in _origin_writes(node.body))
        value = _origin_write_value(node)
        if value is not None and _reads_belief_type(value):
            lines.append(_line(node))
    return sorted(set(lines))


def _sql_sites(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(
                v.value for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            )
        else:
            continue
        if _SQL_ORIGIN_FROM_TYPE.search(text):
            lines.append(node.lineno)
    return lines


@pytest.mark.source_scan
def test_no_source_file_sets_origin_in_a_belief_type_check() -> None:
    """Find a belief type check and an origin write in the same statement.

    The scan flags an `if` whose test reads a belief type and whose body
    writes an origin, an origin write whose value reads a belief type, and a
    SQL `UPDATE` that sets origin with a type predicate. It can miss other
    shapes. The known one is the opt-in Stop-hook auto-lock that #2022
    tracks, which picks beliefs by type in one function and sets their
    origin in another.
    """
    files = sorted(_SRC.rglob("*.py"))
    assert files, f"no source files found under {_SRC}"
    hits: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(_SRC)
        hits.extend(f"{rel}:{n} (SQL)" for n in _sql_sites(tree))
        hits.extend(f"{rel}:{n}" for n in _origin_from_type_sites(tree))
    assert not hits, (
        "these sites set a belief's origin from its type (#2013): "
        + ", ".join(hits)
    )


def test_scan_patterns_catch_both_removed_writers() -> None:
    """The scan flags the two writers #2013 removes, written out here.

    Without this, a scan that matched nothing would also pass.
    """
    store_rule = (
        "UPDATE beliefs SET origin = 'user_corrected' "
        "WHERE origin = 'unknown' AND type = 'correction'"
    )
    lock_rule = (
        "UPDATE beliefs SET origin = 'user_stated' "
        "WHERE origin = 'unknown' AND lock_level = 'user'"
    )
    assert _SQL_ORIGIN_FROM_TYPE.search(store_rule)
    assert not _SQL_ORIGIN_FROM_TYPE.search(lock_rule)

    migrate_rule = ast.parse(
        "if origin == ORIGIN_UNKNOWN:\n"
        "    if lock_level == LOCK_USER:\n"
        "        origin = ORIGIN_USER_STATED\n"
        "    elif row['type'] == BELIEF_CORRECTION:\n"
        "        origin = ORIGIN_USER_CORRECTED\n"
    )
    assert _origin_from_type_sites(migrate_rule) == [5]
    lock_only = ast.parse(
        "if origin == ORIGIN_UNKNOWN:\n"
        "    if lock_level == LOCK_USER:\n"
        "        origin = ORIGIN_USER_STATED\n"
    )
    assert _origin_from_type_sites(lock_only) == []
    unrelated = ast.parse(
        "if type(x) is dict and scope == BELIEF_SCOPE_PROJECT:\n"
        "    origin = ORIGIN_USER_STATED\n"
        "if belief.type == BELIEF_CORRECTION:\n"
        "    precedence = PRECEDENCE_USER_CORRECTED\n"
    )
    assert _origin_from_type_sites(unrelated) == []
