"""Result, context and probe types for the fire census (#1887).

A probe is a function `Ctx -> Result`. It reads one store through a
read-only sqlite connection (`ctx.db`) or the JSONL sidecars beside the
store's `memory.db`, and returns integers and fixed labels only.

Output privacy is enforced here, not by convention. `Result` validation
rejects a count that is not an int, a detail label outside `LABEL_RE`, a
detail value that is not an int, and a note outside `NOTE_RE`. `NOTE_RE`
has no `/` or `\\`, so a note cannot carry a path. A probe that needs a
value from the store in its output has to name the allowed values in code
(`Ctx.group_count`, `jsonl_tally`); anything else folds into `other`.

Store locations and the tables and sidecars the probes read are listed in
the docstring of each `probes_*` module, next to the writer each probe
keys on.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import sqlite3
import stat
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

Kind = Literal["store", "sidecar", "global-log", "untraceable"]
Status = Literal["fired", "ambiguous", "silent", "untraceable", "error"]
Layout = Literal["repo", "named", "legacy", "global", "user", "none"]

KINDS: frozenset[str] = frozenset({"store", "sidecar", "global-log", "untraceable"})
STATUSES: frozenset[str] = frozenset({"fired", "ambiguous", "silent", "untraceable", "error"})
LAYOUTS: frozenset[str] = frozenset({"repo", "named", "legacy", "global", "user", "none"})

LABEL_RE = re.compile(r"^[a-z0-9_.:-]{1,64}$")
NOTE_RE = re.compile(r"^[A-Za-z0-9 _.,:;()=<>+#*-]{0,160}$")
ITEM_ID_RE = re.compile(r"^A[0-9]{3}$")
STORE_LABEL_RE = re.compile(r"^(store[0-9]{2}|named[0-9]{2}|legacy[0-9]{2}|global|user)$")

DEFAULT_PROBE_DEADLINE_S = 60.0
MAX_JSONL_BYTES = 2 * 1024 ** 3

# Row fields that attribute a global-log row to a folder or project.
ATTRIBUTION_KEYS: frozenset[str] = frozenset({
    "cwd", "project", "path", "project_root", "project_dir", "repo",
    "repo_root", "root", "db", "db_path", "store", "store_path", "folder",
    "transcript_path", "source_path", "file", "dir", "workdir",
})


# --- result types ----------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Result:
    """One probe outcome for one store: a status, an int count and int detail."""

    status: Status
    count: int | None = None
    detail: dict[str, int] = dataclasses.field(default_factory=lambda: {})
    note: str = ""

    def __post_init__(self) -> None:
        validate_result(self)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status, "count": self.count, "detail": dict(self.detail),
        }
        if self.note:
            out["note"] = self.note
        return out


def validate_result(r: Result) -> None:
    """Raise unless `r` carries only ints and labels from the fixed patterns."""
    if r.status not in STATUSES:
        raise ValueError("bad status")
    if r.count is not None and type(r.count) is not int:
        raise TypeError("count must be int or None")
    if not isinstance(r.detail, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("detail must be dict")
    for k, v in r.detail.items():
        if not isinstance(k, str) or not LABEL_RE.match(k):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("detail label outside LABEL_RE")
        if type(v) is not int:
            raise TypeError("detail value must be int")
    if not isinstance(r.note, str) or not NOTE_RE.match(r.note):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("note outside NOTE_RE")
    if r.status in ("fired", "ambiguous") and (r.count is None or r.count <= 0):
        raise ValueError("fired and ambiguous need a count above 0")
    if r.status == "silent" and r.count not in (0, None):
        raise ValueError("silent needs a count of 0 or null")
    if r.status == "untraceable" and r.count is not None:
        raise ValueError("untraceable has a null count")


def fired(count: int, detail: dict[str, int] | None = None, note: str = "") -> Result:
    return Result("fired", count, detail or {}, note)


def silent(count: int | None = 0, detail: dict[str, int] | None = None,
           note: str = "") -> Result:
    return Result("silent", count, detail or {}, note)


def untraceable(note: str = "no durable trace") -> Result:
    return Result("untraceable", None, {}, note)


def from_count(count: int, detail: dict[str, int] | None = None, note: str = "") -> Result:
    """`fired` when count > 0, else `silent`."""
    return (fired if count > 0 else silent)(count, detail, note)


def ambiguous(count: int, detail: dict[str, int] | None = None, note: str = "") -> Result:
    """The trace is there, but another writer leaves the same rows; the note names it."""
    return Result("ambiguous", count, detail or {}, note)


def from_shared_count(count: int, detail: dict[str, int] | None = None,
                      note: str = "") -> Result:
    """`ambiguous` when count > 0, else `silent`: for a trace another writer shares."""
    return (ambiguous if count > 0 else silent)(count, detail, note)


# --- context ---------------------------------------------------------------


class UnfoldedWalError(Exception):
    """The db has a non-empty `-wal` file, so an immutable read would miss rows."""


def wal_file(db_file: Path) -> Path:
    return db_file.with_name(db_file.name + "-wal")


def has_unfolded_wal(db_file: Path) -> bool:
    wal = wal_file(db_file)
    return wal.is_file() and wal.stat().st_size > 0


class IdentityMismatchError(Exception):
    """The db file is not the file its copy token recorded."""


def _dev_fd_dir() -> Path | None:
    d = Path("/dev/fd")
    return d if os.name == "posix" and d.is_dir() else None


def open_ro(db_file: Path, expect: tuple[int, int] | None = None) -> sqlite3.Connection:
    """Open a sqlite file read-only through a descriptor this function opened.

    The file is opened with `O_NOFOLLOW`, so a symlink is refused, and
    `fstat` on that descriptor must show a regular file and, when
    `expect` is given, the `(st_dev, st_ino)` its copy token recorded.
    sqlite then opens `/dev/fd/<n>`, the same open file, so a path swapped
    after the check can't change what is read. Where `/dev/fd` doesn't
    exist (Windows), sqlite opens the path and the identity is checked
    again after the open.

    The URI sets `mode=ro&immutable=1` and the connection `query_only`.
    `mode=ro` alone creates `-wal` and `-shm` beside a WAL-mode database
    and rewrites the `-shm` index; `immutable=1` opens neither, so the
    read changes no byte in the folder. It also ignores the `-wal` file,
    so a db whose `-wal` still holds frames raises `UnfoldedWalError`.
    """
    if has_unfolded_wal(db_file):
        raise UnfoldedWalError("db has a non-empty -wal file")
    fd = os.open(db_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise IdentityMismatchError("db is not a regular file")
        if expect is not None and (st.st_dev, st.st_ino) != expect:
            raise IdentityMismatchError("db is not the file the copy token recorded")
        fd_dir = _dev_fd_dir()
        target = f"file:{fd_dir}/{fd}" if fd_dir else db_file.resolve().as_uri()
        conn = sqlite3.connect(target + "?mode=ro&immutable=1", uri=True, timeout=5.0)
        conn.execute("PRAGMA query_only = 1")
        conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()
        if fd_dir is None:
            again = os.stat(db_file, follow_symlinks=False)
            if (again.st_dev, again.st_ino) != (st.st_dev, st.st_ino):
                conn.close()
                raise IdentityMismatchError("db changed while it was opened")
    finally:
        os.close(fd)
    return conn


@dataclasses.dataclass
class Ctx:
    """One store under census: its db connection, its folder and its layout.

    `store_dir` is the folder that holds `memory.db` and its sidecars.
    `global_dir` is a copy of `~/.aelfrice`, or None when the run was
    given none. `layout` says which kind of store this is, because a few
    features apply to one kind only.
    """

    name: str
    store_dir: Path
    global_dir: Path | None
    is_global: bool = False
    db_file: Path | None = None
    layout: Layout = "repo"
    expect: tuple[int, int] | None = None
    db: sqlite3.Connection | None = None
    _tables: set[str] | None = dataclasses.field(default=None, repr=False)

    def open(self) -> None:
        if self.db is None and self.db_file is not None and (
                self.expect is not None or self.db_file.is_file()):
            self.db = open_ro(self.db_file, self.expect)

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None
        self._tables = None

    def require_db(self) -> sqlite3.Connection:
        if self.db is None:
            raise LookupError("no db")
        return self.db

    # The census scripts this package came from used the private name.
    _require_db = require_db

    def tables(self) -> set[str]:
        if self._tables is None:
            rows = self.require_db().execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            ).fetchall()
            self._tables = {str(r[0]) for r in rows}
        return self._tables

    def table_exists(self, table: str) -> bool:
        return self.db is not None and table in self.tables()

    def column_exists(self, table: str, column: str) -> bool:
        if not self.table_exists(table):
            return False
        cols = self.require_db().execute(
            "SELECT name FROM pragma_table_info(?)", (table,)
        ).fetchall()
        return any(c[0] == column for c in cols)

    def count(self, sql: str, params: Iterable[Any] = ()) -> int:
        """First column of the first row, as int (NULL -> 0)."""
        row = self.require_db().execute(sql, tuple(params)).fetchone()
        if row is None or row[0] is None:
            return 0
        v = row[0]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError("count query returned non-numeric")
        return int(v)

    def counts(self, sql: str, params: Iterable[Any] = ()) -> list[int]:
        """Every column of the first row, as ints (NULL -> 0)."""
        row = self.require_db().execute(sql, tuple(params)).fetchone()
        if row is None:
            return []
        out: list[int] = []
        for v in row:
            if v is None:
                out.append(0)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(int(v))
            else:
                raise TypeError("counts query returned non-numeric")
        return out

    def group_count(self, sql: str, allowed: Iterable[str],
                    params: Iterable[Any] = ()) -> dict[str, int]:
        """Run `SELECT key, n ...`; keys not in `allowed` fold into `other`.

        `allowed` is a fixed set of labels named in code, so a value from
        the store reaches output only when the probe named it.
        """
        allow = set(allowed)
        for a in allow:
            if not LABEL_RE.match(a):
                raise ValueError("allowed label outside LABEL_RE")
        out: dict[str, int] = {}
        for key, n in self.require_db().execute(sql, tuple(params)):
            label = key if isinstance(key, str) and key in allow else "other"
            if key is None:
                label = "null"
            out[label] = out.get(label, 0) + int(n or 0)
        return out

    def sidecar(self, *parts: str) -> Path:
        return self.store_dir.joinpath(*parts)

    def global_file(self, *parts: str) -> Path:
        """A file under the global dir. Raises LookupError when the run has none."""
        if self.global_dir is None:
            raise LookupError("no global dir")
        return self.global_dir.joinpath(*parts)


# --- module-level helpers --------------------------------------------------


def table_exists(ctx: Ctx, table: str) -> bool:
    return ctx.table_exists(table)


def column_exists(ctx: Ctx, table: str, column: str) -> bool:
    return ctx.column_exists(table, column)


def count(ctx: Ctx, sql: str, params: Iterable[Any] = ()) -> int:
    return ctx.count(sql, params)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Parsed dict rows; malformed and non-dict lines are skipped."""
    if not path.is_file() or path.stat().st_size > MAX_JSONL_BYTES:
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj: Any = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield dict(obj)  # pyright: ignore[reportUnknownArgumentType]


def jsonl_paths(path: Path) -> list[Path]:
    """The live file plus its single-slot `.1` rotation, those that exist."""
    return [p for p in (path, path.with_name(path.name + ".1")) if p.is_file()]


def _as_paths(path: Path | Iterable[Path]) -> list[Path]:
    return [path] if isinstance(path, Path) else list(path)


def jsonl_count(path: Path | Iterable[Path],
                predicate: Callable[[dict[str, Any]], bool] = lambda r: True) -> int:
    """Rows where predicate(row) is truthy; a predicate error counts as False."""
    n = 0
    for p in _as_paths(path):
        for row in iter_jsonl(p):
            try:
                if predicate(row):
                    n += 1
            except Exception:  # noqa: BLE001
                continue
    return n


def jsonl_tally(path: Path | Iterable[Path], keyfn: Callable[[dict[str, Any]], str | None],
                allowed: Iterable[str]) -> dict[str, int]:
    """Bucket rows by keyfn(row); labels outside `allowed` fold into `other`."""
    allow = set(allowed)
    out: dict[str, int] = {}
    for p in _as_paths(path):
        for row in iter_jsonl(p):
            try:
                k = keyfn(row)
            except Exception:  # noqa: BLE001
                k = "error"
            if k is None:
                continue
            label = k if k in allow else "other"
            out[label] = out.get(label, 0) + 1
    return out


def row_attribution(row: Any) -> str:
    """`attributed` when a log row names a folder or project, else `unattributed`."""
    stack: list[Any] = [row]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():  # pyright: ignore[reportUnknownVariableType]
                if isinstance(k, str) and k.lower() in ATTRIBUTION_KEYS and v not in (None, ""):
                    return "attributed"
                stack.append(v)
        elif isinstance(cur, list):
            stack.extend(cur)  # pyright: ignore[reportUnknownArgumentType]
    return "unattributed"


def jsonl_split_count(path: Path | Iterable[Path],
                      predicate: Callable[[dict[str, Any]], bool] = lambda r: True
                      ) -> dict[str, int]:
    """Matching rows split into `attributed` and `unattributed` (global logs)."""
    out = {"attributed": 0, "unattributed": 0}
    for p in _as_paths(path):
        for row in iter_jsonl(p):
            try:
                if not predicate(row):
                    continue
            except Exception:  # noqa: BLE001
                continue
            out[row_attribution(row)] += 1
    return out


def safe(fn: Callable[[Ctx], Result]) -> Callable[[Ctx], Result]:
    """Wrap a probe: an exception becomes status `error` with its class name only."""
    def wrapped(ctx: Ctx) -> Result:
        try:
            r = fn(ctx)
            if not isinstance(r, Result):  # pyright: ignore[reportUnnecessaryIsInstance]
                return Result("error", None, {}, "ProbeReturnedNonResult")
            return r
        except Exception as exc:  # noqa: BLE001
            return Result("error", None, {}, type(exc).__name__[:80])
    wrapped.__name__ = getattr(fn, "__name__", "probe")
    return wrapped


@dataclasses.dataclass(frozen=True)
class Probe:
    """One feature: its inventory id, a fixed name, what it reads, and the probe."""

    item_id: str
    name: str
    kind: Kind
    fn: Callable[[Ctx], Result]

    def __post_init__(self) -> None:
        if not ITEM_ID_RE.match(self.item_id):
            raise ValueError("bad item_id")
        if self.kind not in KINDS:
            raise ValueError("bad kind")


def untraceable_probe(item_id: str, name: str, note: str = "no durable trace") -> Probe:
    """A feature that leaves no store row or log line to count."""
    return Probe(item_id, name, "untraceable", lambda ctx: untraceable(note))


def run_one(p: Probe, ctx: Ctx | None, deadline_s: float = DEFAULT_PROBE_DEADLINE_S) -> Result:
    """Run one probe on one context with a wall-clock bound on its sqlite work."""
    if ctx is None:
        if p.kind == "untraceable":
            return safe(p.fn)(Ctx("_", Path("."), None, layout="none"))
        return Result("error", None, {}, "NoGlobalContext")
    db = ctx.db
    if db is not None:
        end = time.monotonic() + deadline_s
        db.set_progress_handler(lambda: 1 if time.monotonic() > end else 0, 10000)
    try:
        return safe(p.fn)(ctx)
    finally:
        if db is not None:
            db.set_progress_handler(None, 0)


def outcome(statuses: Iterable[str]) -> str:
    """Fold per-store statuses into one per-feature outcome.

    `fired` if any store fired, else `ambiguous` if any store holds the
    shared trace, else `silent` if any was silent, else `error` when only
    errors and untraceable results remain and at least one is an error,
    else `untraceable`.
    """
    s = set(statuses)
    if "fired" in s:
        return "fired"
    if "ambiguous" in s:
        return "ambiguous"
    if "silent" in s:
        return "silent"
    if "error" in s:
        return "error"
    return "untraceable"
