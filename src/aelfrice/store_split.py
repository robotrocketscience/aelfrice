"""`aelf store split`: copy file-sourced beliefs into marker stores (#1796).

A `[store]` marker gives a folder its own store, but the beliefs ingested
before the marker existed stay in the repository store. This module copies
each belief whose source file is under a marker folder into that folder's
store. It copies and never moves: the repository store is opened
`mode=ro` and is never written.

A belief qualifies only when it has a file source the scanner wrote:
`doc:<rel>:p<N>` or `ast:<rel>:module|func:<name>|class:<name>`, read from
`ingest_log.source_path` (with `derived_belief_ids`) or from a
`belief_documents` URI of the form `file:<source>`, with an optional
`#...` fragment. `<rel>` is relative to the onboard root, which is the
`repo_path` of the `onboard_sessions` row for the ingest row's session.
A relative `repo_path`, such as `.`, resolves against the work-tree root,
and is kept only when at least `RELATIVE_ROOT_MIN_PRESENT` of the
session's distinct file paths exist as files under it; otherwise the
session names no root. Every ingest row with a file source must name the
same root, or the belief is skipped as `no_root`. A root outside the
work tree, present or not, or inside a linked worktree or nested
repository below it, is skipped as `outside_work_tree`. A root inside
the work tree that is no longer a folder on disk is skipped as
`missing_root`, because a removed linked worktree leaves no `.git` entry
to tell it from a deleted folder. The folder is then the deepest
real folder of `<root>/<rel>` inside the work tree (`db_paths.owning_folder`),
and the belief is copied only when that folder resolves to a marker store
other than the repository store.

A copy carries the belief under its own id, with `project_context`
restamped to the target store's identity for a project-scope belief and
left empty for a `global` or `shared:` one, and its `belief_documents`,
`feedback_history` and `belief_corroborations` rows. Edges and
`belief_support_evidence` rows come along when both ends land in the
target, as a copy or as a belief already there. Nothing else is copied.

A belief is skipped, and counted under its reason, when the target already
holds its id or content hash, when it is user-locked, when it is retired,
and when it is superseded by nothing that lands in the target. A lock is
not copied because a user lock keeps an empty `project_context` so that
it applies under every context, and the copy restamps that column. A
retired belief's search index row was removed when it was retired, and an
insert would add it back. A SUPERSEDES edge is what marks a belief
superseded, so the belief keeps that state only when the edge comes along.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Final

from aelfrice.config_discovery import (
    CONFIG_FILENAME,
    WORKTREE_MARKER,
    config_discovery_scope,
)
from aelfrice.db_paths import (
    STORE_RULE_MARKER,
    FolderAccessError,
    StoreSelection,
    owning_folder,
    repository_store_path,
    resolve_store_in_work_tree,
    same_store_file,
    store_file_present,
    store_identity,
)
from aelfrice.migrate import copy_belief
from aelfrice.models import EDGE_SUPERSEDES, LOCK_USER, Belief, Edge, SupportEvidence
from aelfrice.store import MemoryStore, ReadOnlyStoreUnavailable, read_only_uri

#: Reasons a belief is not copied, as the report keys them.
SKIP_NO_FILE_SOURCE: Final[str] = "no_file_source"
SKIP_NO_ROOT: Final[str] = "no_root"
SKIP_OUTSIDE_WORK_TREE: Final[str] = "outside_work_tree"
SKIP_MISSING_ROOT: Final[str] = "missing_root"
SKIP_SOURCE_STORE: Final[str] = "source_store"
SKIP_SEVERAL_STORES: Final[str] = "several_stores"
SKIP_ALREADY_PRESENT: Final[str] = "already_present"
SKIP_LOCKED: Final[str] = "locked"
SKIP_RETIRED: Final[str] = "retired"
SKIP_SUPERSEDED: Final[str] = "superseded"

#: Reasons decided before a belief has a target store.
UNROUTED_REASONS: Final[tuple[str, ...]] = (
    SKIP_NO_FILE_SOURCE,
    SKIP_NO_ROOT,
    SKIP_OUTSIDE_WORK_TREE,
    SKIP_MISSING_ROOT,
    SKIP_SEVERAL_STORES,
    SKIP_SOURCE_STORE,
)

#: Reasons decided against one target store.
TARGET_REASONS: Final[tuple[str, ...]] = (
    SKIP_ALREADY_PRESENT,
    SKIP_LOCKED,
    SKIP_RETIRED,
    SKIP_SUPERSEDED,
)

#: The row kinds a copy writes, in the order it writes them.
COPIED_KINDS: Final[tuple[str, ...]] = (
    "beliefs",
    "documents",
    "feedback",
    "corroborations",
    "edges",
    "support_evidence",
)

# The scanner's file sources (`scanner.extract_filesystem`, `extract_ast`).
# The greedy `rel` group backtracks to the last suffix, so a path that
# holds a colon still parses.
_DOC_SOURCE_RE: Final[re.Pattern[str]] = re.compile(r"doc:(?P<rel>.+):p\d+")
_AST_SOURCE_RE: Final[re.Pattern[str]] = re.compile(
    r"ast:(?P<rel>.+):(?:module|(?:func|class):[^:]+)",
)
_FILE_URI_PREFIX: Final[str] = "file:"

_GIT_TIMEOUT_S: Final[float] = 5.0

#: The share of a relative-root onboard session's distinct file paths that
#: must exist as files under the work-tree folder it resolves to, for that
#: folder to count as the session's root. The bar is inclusive.
RELATIVE_ROOT_MIN_PRESENT: Final[float] = 0.9

_BELIEF_COLUMNS: Final[str] = (
    "b.id, b.content, b.content_hash, b.alpha, b.beta, b.type, "
    "b.lock_level, b.locked_at, b.created_at, b.last_retrieved_at, "
    "b.session_id, b.origin, b.retention_class, b.valid_to, b.scope, "
    "b.project_context, b.last_confirmed_at, b.lock_tier, b.lock_expires_at"
)


class SplitRefused(Exception):
    """`aelf store split` can't run here; the message is one line."""


class SplitError(Exception):
    """The repository store couldn't be read."""


@dataclass(frozen=True)
class _DocRow:
    belief_id: str
    doc_uri: str
    anchor_type: str
    position_hint: str | None
    created_at: float


@dataclass(frozen=True)
class _FeedbackRow:
    belief_id: str
    valence: float
    source: str
    created_at: str


@dataclass(frozen=True)
class _CorroborationRow:
    belief_id: str
    ingested_at: str
    source_type: str
    session_id: str | None
    source_path_hash: str | None
    speaker: str | None


@dataclass(frozen=True)
class _SourceRows:
    """The rows of the candidate beliefs, read once from the source."""

    beliefs: dict[str, Belief]
    documents: list[_DocRow]
    feedback: list[_FeedbackRow]
    corroborations: list[_CorroborationRow]
    edges: list[Edge]
    evidence: list[SupportEvidence]
    superseders: dict[str, set[str]]


@dataclass(frozen=True)
class _Plan:
    beliefs: list[Belief]
    documents: list[_DocRow]
    feedback: list[_FeedbackRow]
    corroborations: list[_CorroborationRow]
    edges: list[Edge]
    evidence: list[SupportEvidence]
    skipped: dict[str, int]

    def counts(self) -> dict[str, int]:
        return {
            "beliefs": len(self.beliefs),
            "documents": len(self.documents),
            "feedback": len(self.feedback),
            "corroborations": len(self.corroborations),
            "edges": len(self.edges),
            "support_evidence": len(self.evidence),
        }


@dataclass(frozen=True)
class TargetReport:
    """What one marker store received, or would receive in a dry run."""

    store: Path
    marker: Path | None
    copied: dict[str, int]
    skipped: dict[str, int]
    error: str | None = None


@dataclass(frozen=True)
class SplitReport:
    """The result of one `aelf store split` run."""

    work_tree: Path
    source: Path
    applied: bool
    targets: tuple[TargetReport, ...]
    not_copied: dict[str, int]

    @property
    def failed(self) -> bool:
        return any(t.error is not None for t in self.targets)

    def to_json(self) -> dict[str, object]:
        return {
            "work_tree": str(self.work_tree),
            "source": str(self.source),
            "applied": self.applied,
            "targets": [
                {
                    "store": str(t.store),
                    "marker": None if t.marker is None else str(t.marker),
                    "copied": t.copied,
                    "skipped": t.skipped,
                    "error": t.error,
                }
                for t in self.targets
            ],
            "not_copied": self.not_copied,
        }


def _safe_rel(rel: str) -> bool:
    """True for a relative POSIX path with no `.` or `..` step."""
    if not rel or "\x00" in rel or "\\" in rel:
        return False
    if PurePosixPath(rel).is_absolute():
        return False
    return all(part not in ("", ".", "..") for part in rel.split("/"))


def file_source_rel(source: str) -> str | None:
    """The `<rel>` of a scanner file source, or None for any other source."""
    for pattern in (_DOC_SOURCE_RE, _AST_SOURCE_RE):
        found = pattern.fullmatch(source)
        if found is not None:
            rel = found["rel"]
            return rel if _safe_rel(rel) else None
    return None


def doc_uri_rel(uri: str) -> str | None:
    """The `<rel>` of a `file:<source>` anchor URI, or None.

    The `#` fragment is optional and a path may hold `#` too, so the
    whole URI is tried first, then each cut at a `#` from the right.
    """
    if not uri.startswith(_FILE_URI_PREFIX) or uri.startswith("file://"):
        return None
    body = uri[len(_FILE_URI_PREFIX):]
    cut = len(body)
    for _ in range(body.count("#") + 1):
        rel = file_source_rel(body[:cut])
        if rel is not None:
            return rel
        cut = body.rfind("#", 0, cut)
        if cut < 0:
            return None
    return None


def work_tree_root(start: Path) -> Path | None:
    """The root of the git work tree holding `start`, or None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8",
            errors="strict", check=False, timeout=_GIT_TIMEOUT_S, cwd=start,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    top = result.stdout.strip()
    if result.returncode != 0 or not top:
        return None
    return Path(top).resolve()


def check_markers(work_tree: Path) -> None:
    """Raise `StoreMarkerError` for a broken marker anywhere in the work
    tree, and `FolderAccessError` for a folder in it that can't be listed
    or searched.

    Walks down from the root and resolves the store of every folder that
    holds the project config file. The walk skips `.git`, nested
    repositories, and symlinked folders, which belong to no folder of
    this work tree. A folder it can't list or search may hold a marker
    or a `.git` entry it can't check, so the walk stops there.
    """

    def refuse(exc: OSError) -> None:
        # A folder removed during the walk holds nothing to check.
        if isinstance(exc, (FileNotFoundError, NotADirectoryError)):
            return
        folder = work_tree if exc.filename is None else Path(
            os.fsdecode(exc.filename),
        )
        raise FolderAccessError(folder) from exc

    for dirpath, dirnames, filenames in os.walk(work_tree, onerror=refuse):
        here = Path(dirpath)
        dirnames[:] = sorted(
            d for d in dirnames
            if d != WORKTREE_MARKER and _walks_into(here / d)
        )
        if CONFIG_FILENAME in filenames:
            resolve_store_in_work_tree(here, work_tree)


def _walks_into(folder: Path) -> bool:
    """True when the marker walk enters `folder`: a real folder, not a
    symlink, with no `.git` entry.

    Raises `FolderAccessError` when it can't check for a `.git` entry,
    which is a folder it can't search.
    """
    try:
        if stat.S_ISLNK(os.lstat(folder).st_mode):
            return False
        os.lstat(folder / WORKTREE_MARKER)
    except (FileNotFoundError, NotADirectoryError):
        return True
    except OSError as exc:
        raise FolderAccessError(folder) from exc
    return False


def _resolved(path: Path) -> Path | None:
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _enough_present(root: Path, rels: set[str]) -> bool:
    """True when at least `RELATIVE_ROOT_MIN_PRESENT` of `rels` are files
    under `root`."""
    if not rels:
        return False
    present = sum(1 for rel in sorted(rels) if os.path.isfile(root / rel))
    return present / len(rels) >= RELATIVE_ROOT_MIN_PRESENT


def _onboard_roots(
    conn: sqlite3.Connection, work_tree: Path,
) -> dict[str, Path | None]:
    """Each onboard session's root, None when it names no usable one.

    An absolute `repo_path` is the root. A relative one, which `aelf
    onboard .` records, resolves against `work_tree`. A root that lands
    outside `work_tree` is kept, so the belief counts as outside the
    work tree. One inside it is kept only when the session's files are
    there, since the folder the onboard ran from isn't recorded.
    """
    roots: dict[str, Path | None] = {}
    relative: dict[str, Path] = {}
    for sid, repo_path in conn.execute(
        "SELECT session_id, repo_path FROM onboard_sessions",
    ).fetchall():
        key = str(sid)
        if not isinstance(repo_path, str) or not repo_path:
            roots[key] = None
            continue
        given = Path(repo_path)
        root = _resolved(given if given.is_absolute() else work_tree / given)
        roots[key] = root
        if root is not None and not given.is_absolute() and root.is_relative_to(
            work_tree,
        ):
            relative[key] = root
    if not relative:
        return roots
    rels: dict[str, set[str]] = {}
    for sid, source_path in conn.execute(
        "SELECT DISTINCT session_id, source_path FROM ingest_log "
        "WHERE session_id IS NOT NULL AND source_path IS NOT NULL",
    ).fetchall():
        rel = file_source_rel(str(source_path))
        if rel is not None and str(sid) in relative:
            rels.setdefault(str(sid), set()).add(rel)
    for key, root in relative.items():
        if not _enough_present(root, rels.get(key, set())):
            roots[key] = None
    return roots


def _read_locations(
    conn: sqlite3.Connection, work_tree: Path,
) -> tuple[list[str], dict[str, list[tuple[Path | None, str]]], dict[str, set[str]]]:
    """Every belief id, and each belief's file sources.

    Returns `(ids, ingested, anchored)`. `ingested` maps a belief to
    `(root, rel)` for each ingest row with a file source, `root` None
    when the row's session names no usable onboard root. `anchored` maps
    a belief to the `rel` of each anchor URI with a file source.
    """
    roots = _onboard_roots(conn, work_tree)
    ids = [str(r[0]) for r in conn.execute("SELECT id FROM beliefs ORDER BY id")]
    ingested: dict[str, list[tuple[Path | None, str]]] = {}
    # A malformed `derived_belief_ids` reads as no ids. The CASE guards
    # the function call itself, since `json_each` raises on bad JSON
    # whatever the WHERE clause says.
    for bid, source_path, sid in conn.execute(
        "SELECT je.value, il.source_path, il.session_id "
        "FROM ingest_log il, json_each(CASE "
        "WHEN json_valid(il.derived_belief_ids) "
        "AND json_type(il.derived_belief_ids) = 'array' "
        "THEN il.derived_belief_ids ELSE '[]' END) je "
        "WHERE il.source_path IS NOT NULL "
        "ORDER BY il.id",
    ).fetchall():
        rel = file_source_rel(str(source_path))
        if rel is None or not isinstance(bid, str):
            continue
        root = roots.get(str(sid)) if sid is not None else None
        ingested.setdefault(bid, []).append((root, rel))
    anchored: dict[str, set[str]] = {}
    for bid, uri in conn.execute(
        "SELECT belief_id, doc_uri FROM belief_documents",
    ).fetchall():
        rel = doc_uri_rel(str(uri))
        if rel is not None:
            anchored.setdefault(str(bid), set()).add(rel)
    return ids, ingested, anchored


def _root_skip(work_tree: Path, under: Path) -> str | None:
    """Why the onboard root `work_tree / under` can't place a belief, or
    None when it can.

    Each folder from below `work_tree` down to the root must be a folder
    with no `.git` entry. A `.git` entry puts the root in a linked
    worktree, a nested repository, or a submodule (`outside_work_tree`).
    A linked worktree shares the repository store, but its files belong
    to its own work tree and its own markers. A folder that is gone gives
    `missing_root`: `git worktree remove` deletes the folder and its
    `.git` entry, so a removed worktree looks like any deleted folder,
    and its beliefs must not fall to a marker above it. A folder it can't
    search raises `FolderAccessError`.
    """
    current = work_tree
    for part in under.parts:
        parent, current = current, current / part
        try:
            if not stat.S_ISDIR(os.stat(current).st_mode):
                return SKIP_MISSING_ROOT
        except (FileNotFoundError, NotADirectoryError):
            return SKIP_MISSING_ROOT
        except OSError as exc:
            raise FolderAccessError(parent) from exc
        try:
            os.lstat(current / WORKTREE_MARKER)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as exc:
            raise FolderAccessError(current) from exc
        return SKIP_OUTSIDE_WORK_TREE
    return None


def _belief_folders(
    work_tree: Path,
    ingested: list[tuple[Path | None, str]],
    anchored: set[str],
) -> tuple[str | None, set[Path]]:
    """`(reason, folders)`: the folders a belief's files are in, or why
    none can be named."""
    rels = {rel for _, rel in ingested} | anchored
    if not rels:
        return SKIP_NO_FILE_SOURCE, set()
    roots = {root for root, _ in ingested}
    if len(roots) != 1:
        return SKIP_NO_ROOT, set()
    root = roots.pop()
    if root is None:
        return SKIP_NO_ROOT, set()
    try:
        under = Path(*root.relative_to(work_tree).parts)
    except ValueError:
        return SKIP_OUTSIDE_WORK_TREE, set()
    skip = _root_skip(work_tree, under)
    if skip is not None:
        return skip, set()
    return None, {
        owning_folder(work_tree, str(under / rel)) for rel in sorted(rels)
    }


def _store_key(
    selection: StoreSelection, known: list[tuple[str, StoreSelection]],
) -> tuple[str, StoreSelection]:
    """The target key and selection for the store `selection` names.

    Two markers can name one store under two spellings, such as `a/` and
    `b/` that both say `../shared`, so a store already seen that is the
    same file (`same_store_file`) keeps its key and its selection.
    """
    for key, seen in known:
        if same_store_file(seen.path, selection.path):
            return key, seen
    entry = (str(selection.path), selection)
    known.append(entry)
    return entry


def _route(
    work_tree: Path,
    source: Path,
    ids: list[str],
    ingested: dict[str, list[tuple[Path | None, str]]],
    anchored: dict[str, set[str]],
) -> tuple[dict[str, tuple[StoreSelection, set[str]]], dict[str, int]]:
    """Group the beliefs by target store.

    Returns `(targets, not_copied)`: `targets` maps a store path to its
    selection and the ids bound for it, and `not_copied` counts the
    beliefs that have no target, by reason.
    """
    not_copied = {reason: 0 for reason in UNROUTED_REASONS}
    targets: dict[str, tuple[StoreSelection, set[str]]] = {}
    selections: dict[Path, tuple[str, StoreSelection]] = {}
    known: list[tuple[str, StoreSelection]] = []
    for bid in ids:
        reason, folders = _belief_folders(
            work_tree, ingested.get(bid, []), anchored.get(bid, set()),
        )
        if reason is not None:
            not_copied[reason] += 1
            continue
        chosen: dict[str, StoreSelection] = {}
        for folder in sorted(folders):
            if folder not in selections:
                selections[folder] = _store_key(
                    resolve_store_in_work_tree(folder, work_tree), known,
                )
            key, selection = selections[folder]
            chosen.setdefault(key, selection)
        if len(chosen) != 1:
            not_copied[SKIP_SEVERAL_STORES] += 1
            continue
        key, selection = next(iter(chosen.items()))
        if selection.rule != STORE_RULE_MARKER or same_store_file(
            selection.path, source,
        ):
            not_copied[SKIP_SOURCE_STORE] += 1
            continue
        targets.setdefault(key, (selection, set()))[1].add(bid)
    return targets, not_copied


def _row_belief(r: tuple[Any, ...]) -> Belief:
    def text(v: object) -> str | None:
        return None if v is None else str(v)

    return Belief(
        id=str(r[0]), content=str(r[1]), content_hash=str(r[2]),
        alpha=float(r[3]), beta=float(r[4]), type=str(r[5]),
        lock_level=str(r[6]), locked_at=text(r[7]), created_at=str(r[8]),
        last_retrieved_at=text(r[9]), session_id=text(r[10]),
        origin=str(r[11]), retention_class=str(r[12]), valid_to=text(r[13]),
        scope=str(r[14]), project_context=str(r[15] or ""),
        last_confirmed_at=text(r[16]), lock_tier=str(r[17]),
        lock_expires_at=text(r[18]),
    )


def _read_rows(conn: sqlite3.Connection, ids: set[str]) -> _SourceRows:
    """Read the candidate beliefs and every row a copy may carry."""
    cand = json.dumps(sorted(ids))
    beliefs = {
        str(r[0]): _row_belief(tuple(r))
        for r in conn.execute(
            f"SELECT {_BELIEF_COLUMNS} FROM beliefs b "
            "JOIN json_each(?) AS c ON c.value = b.id",
            (cand,),
        ).fetchall()
    }
    documents = [
        _DocRow(str(r[0]), str(r[1]), str(r[2]),
                None if r[3] is None else str(r[3]), float(r[4]))
        for r in conn.execute(
            "SELECT d.belief_id, d.doc_uri, d.anchor_type, d.position_hint, "
            "d.created_at FROM belief_documents d "
            "JOIN json_each(?) AS c ON c.value = d.belief_id "
            "ORDER BY d.belief_id, d.created_at, d.doc_uri",
            (cand,),
        ).fetchall()
    ]
    feedback = [
        _FeedbackRow(str(r[0]), float(r[1]), str(r[2]), str(r[3]))
        for r in conn.execute(
            "SELECT f.belief_id, f.valence, f.source, f.created_at "
            "FROM feedback_history f "
            "JOIN json_each(?) AS c ON c.value = f.belief_id ORDER BY f.id",
            (cand,),
        ).fetchall()
    ]
    corroborations = [
        _CorroborationRow(
            str(r[0]), str(r[1]), str(r[2]),
            None if r[3] is None else str(r[3]),
            None if r[4] is None else str(r[4]),
            None if r[5] is None else str(r[5]),
        )
        for r in conn.execute(
            "SELECT k.belief_id, k.ingested_at, k.source_type, k.session_id, "
            "k.source_path_hash, k.speaker FROM belief_corroborations k "
            "JOIN json_each(?) AS c ON c.value = k.belief_id ORDER BY k.id",
            (cand,),
        ).fetchall()
    ]
    edges = [
        Edge(src=str(r[0]), dst=str(r[1]), type=str(r[2]), weight=float(r[3]),
             anchor_text=None if r[4] is None else str(r[4]))
        for r in conn.execute(
            "SELECT src, dst, type, weight, anchor_text FROM edges "
            "WHERE src IN (SELECT value FROM json_each(?)) "
            "OR dst IN (SELECT value FROM json_each(?)) "
            "ORDER BY src, dst, type",
            (cand, cand),
        ).fetchall()
    ]
    evidence = [
        SupportEvidence(
            supporter_id=str(r[0]), target_id=str(r[1]), kind=str(r[2]),
            target_sentence_index=int(r[3]),
            supporter_session_id=None if r[4] is None else str(r[4]),
            supporter_ingest_log_id=None if r[5] is None else str(r[5]),
            matched_span=str(r[6]), applied=bool(r[7]), created_at=str(r[8]),
        )
        for r in conn.execute(
            "SELECT supporter_id, target_id, kind, target_sentence_index, "
            "supporter_session_id, supporter_ingest_log_id, matched_span, "
            "applied, created_at FROM belief_support_evidence "
            "WHERE supporter_id IN (SELECT value FROM json_each(?)) "
            "OR target_id IN (SELECT value FROM json_each(?)) "
            "ORDER BY supporter_id, target_id, kind",
            (cand, cand),
        ).fetchall()
    ]
    superseders: dict[str, set[str]] = {}
    for e in edges:
        if e.type == EDGE_SUPERSEDES and e.src != e.dst:
            superseders.setdefault(e.dst, set()).add(e.src)
    return _SourceRows(
        beliefs, documents, feedback, corroborations, edges, evidence,
        superseders,
    )


def _evidence_present(target: MemoryStore | None, row: SupportEvidence) -> bool:
    if target is None:
        return False
    return any(
        e.supporter_id == row.supporter_id and e.kind == row.kind
        for e in target.list_support_evidence(row.target_id)
    )


def _plan(
    rows: _SourceRows, ids: set[str], target: MemoryStore | None,
) -> _Plan:
    """Decide what a copy of `ids` into `target` writes.

    `target` is None for a store that doesn't exist yet, which holds
    nothing.
    """
    beliefs = [rows.beliefs[bid] for bid in sorted(ids) if bid in rows.beliefs]
    endpoints = set(ids)
    for e in rows.edges:
        endpoints.update((e.src, e.dst))
    for v in rows.evidence:
        endpoints.update((v.supporter_id, v.target_id))
    present_ids: set[str] = set()
    present_hashes: set[str] = set()
    if target is not None:
        present_ids, present_hashes = target.existing_belief_keys(
            sorted(endpoints), [b.content_hash for b in beliefs],
        )
    skipped = {reason: 0 for reason in TARGET_REASONS}
    copy: dict[str, Belief] = {}
    for b in beliefs:
        if b.id in present_ids or b.content_hash in present_hashes:
            skipped[SKIP_ALREADY_PRESENT] += 1
        elif b.lock_level == LOCK_USER:
            skipped[SKIP_LOCKED] += 1
        elif b.valid_to is not None:
            skipped[SKIP_RETIRED] += 1
        else:
            copy[b.id] = b
    landed = set(copy) | present_ids
    # A superseded belief stays only while one of its superseders lands.
    # Dropping one can strand a belief it superseded, so repeat until
    # nothing changes; each pass drops at least one, so this is bounded.
    for _ in range(len(copy) + 1):
        drop = sorted(
            bid for bid in copy
            if rows.superseders.get(bid) and not (rows.superseders[bid] & landed)
        )
        if not drop:
            break
        for bid in drop:
            del copy[bid]
            landed.discard(bid)
            skipped[SKIP_SUPERSEDED] += 1

    def lands(a: str, b: str) -> bool:
        return a != b and a in landed and b in landed and (a in copy or b in copy)

    return _Plan(
        beliefs=[copy[bid] for bid in sorted(copy)],
        documents=[d for d in rows.documents if d.belief_id in copy],
        feedback=[f for f in rows.feedback if f.belief_id in copy],
        corroborations=[c for c in rows.corroborations if c.belief_id in copy],
        edges=[
            e for e in rows.edges
            if lands(e.src, e.dst)
            and (target is None or target.get_edge(e.src, e.dst, e.type) is None)
        ],
        evidence=[
            v for v in rows.evidence
            if lands(v.supporter_id, v.target_id)
            and not _evidence_present(target, v)
        ],
        skipped=skipped,
    )


def _write(target: MemoryStore, plan: _Plan) -> dict[str, int]:
    """Write `plan` into `target`; the caller holds the transaction.

    Each copy goes in with an empty `project_context`, so the target
    applies its own rule: it stamps its identity on a project-scope row
    and leaves a `global` or `shared:` row empty, visible under every
    context.
    """
    for b in plan.beliefs:
        copy_belief(target, replace(b, project_context=""))
    for d in plan.documents:
        # The store annotates the return as a string forward reference.
        target.link_belief_to_document(  # pyright: ignore[reportUnknownMemberType]
            belief_id=d.belief_id, doc_uri=d.doc_uri,
            anchor_type=d.anchor_type, position_hint=d.position_hint,
            created_at=d.created_at,
        )
    for f in plan.feedback:
        target.insert_feedback_event(
            f.belief_id, f.valence, f.source, f.created_at,
        )
    for c in plan.corroborations:
        target.record_corroboration(
            c.belief_id, source_type=c.source_type, session_id=c.session_id,
            source_path_hash=c.source_path_hash, ts=c.ingested_at,
            speaker=c.speaker,
        )
    edges = sum(1 for e in plan.edges if target.insert_edge(e))
    evidence = sum(
        1 for v in plan.evidence
        if target.record_support_evidence(
            v.supporter_id, v.target_id, kind=v.kind,
            target_sentence_index=v.target_sentence_index,
            supporter_session_id=v.supporter_session_id,
            supporter_ingest_log_id=v.supporter_ingest_log_id,
            matched_span=v.matched_span, applied=v.applied, ts=v.created_at,
        )
    )
    counts = plan.counts()
    counts["edges"] = edges
    counts["support_evidence"] = evidence
    return counts


def _dry_run_target(
    rows: _SourceRows, selection: StoreSelection, ids: set[str],
) -> TargetReport:
    """Plan one target without writing; a store that can't be read is
    reported as failed, as `_apply_target` reports it.

    A read-only open also fails with `ReadOnlyStoreUnavailable` for a WAL
    store whose sidecars it can't create, or an old schema it can't
    migrate."""
    path = selection.path
    try:
        if not store_file_present(path):
            plan = _plan(rows, ids, None)
        else:
            target = MemoryStore(str(path), read_only=True)
            try:
                plan = _plan(rows, ids, target)
            finally:
                target.close()
    except (sqlite3.Error, OSError, ValueError, ReadOnlyStoreUnavailable) as exc:
        return TargetReport(
            path, selection.marker, {kind: 0 for kind in COPIED_KINDS},
            {reason: 0 for reason in TARGET_REASONS}, str(exc),
        )
    return TargetReport(path, selection.marker, plan.counts(), plan.skipped)


def _apply_target(
    rows: _SourceRows, selection: StoreSelection, ids: set[str],
) -> TargetReport:
    """Copy into one target in one `BEGIN IMMEDIATE` transaction, so a
    failure leaves its rows as they were."""
    path = selection.path
    empty = {kind: 0 for kind in COPIED_KINDS}
    skipped = {reason: 0 for reason in TARGET_REASONS}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = store_identity(selection, create=True)
        target = MemoryStore(str(path), project_context_default=identity)
        try:
            with target.transaction(immediate=True):
                plan = _plan(rows, ids, target)
                skipped = plan.skipped
                copied = _write(target, plan)
        finally:
            target.close()
    except (sqlite3.Error, OSError, ValueError) as exc:
        return TargetReport(path, selection.marker, empty, skipped, str(exc))
    return TargetReport(path, selection.marker, copied, skipped)


def split_store(start: Path, *, apply: bool) -> SplitReport:
    """Copy the beliefs under each marker folder of `start`'s work tree.

    A dry run unless `apply`. Raises `SplitRefused` when the command
    can't run here, including a folder of the work tree it can't list or
    search, `StoreMarkerError` for a broken marker anywhere in the work
    tree (both before anything is written), and `SplitError` when the
    repository store can't be read.
    """
    if apply and os.environ.get("AELFRICE_DB"):
        raise SplitRefused(
            "--apply refuses to run while AELFRICE_DB is set, because it "
            "hides every [store] marker; unset it and run again",
        )
    try:
        path = _check_start(start)
        work_tree = work_tree_root(path)
        if work_tree is None:
            raise SplitRefused(f"{path} is not in a git work tree")
        source = repository_store_path(work_tree)
        if source is None:
            raise SplitRefused(f"{work_tree} is not in a git repository")
        return _split(work_tree, source, apply=apply)
    except FolderAccessError as exc:
        raise SplitRefused(str(exc)) from exc


def _check_start(start: Path) -> Path:
    """`start` as an absolute path, once git can run there.

    git fails there for a file, a missing path, or a folder it can't
    reach, so this names that problem first: `SplitRefused` for a
    `start` that is missing or a file, and `FolderAccessError` for the
    first folder on the way to `start`, or `start` itself, that can't
    be searched. A relative `start` also needs the current folder's
    path, which the system can't give for a removed folder, or, on
    macOS, for one you can't search; that is a `SplitRefused` too.
    """
    try:
        path = Path(os.path.abspath(start))
    except FileNotFoundError:
        raise SplitRefused("the current folder no longer exists") from None
    except OSError:
        raise SplitRefused(
            "cannot list or search the current folder",
        ) from None
    reach = next(
        (
            p for p in (path, *path.parents)
            if os.path.isdir(p) and os.access(p, os.X_OK)
        ),
        None,
    )
    if reach == path:
        return path
    if reach is None:
        raise SplitRefused(f"no folder at {start}")
    first = reach / path.relative_to(reach).parts[0]
    try:
        mode = os.stat(first).st_mode
    except OSError:
        raise SplitRefused(f"no folder at {start}") from None
    if not stat.S_ISDIR(mode):
        if first == path:
            raise SplitRefused(f"{start} is not a folder")
        raise SplitRefused(f"no folder at {start}")
    raise FolderAccessError(first)


def _split(work_tree: Path, source: Path, *, apply: bool) -> SplitReport:
    """`split_store()` once the work tree and its repository store are
    known."""
    with config_discovery_scope():
        check_markers(work_tree)
        try:
            present = store_file_present(source)
        except OSError as exc:
            # A folder it can't search: the store may be there.
            raise SplitError(f"cannot read {source}: {exc}") from exc
        if not present:
            raise SplitRefused(f"no repository store at {source}")
        try:
            conn = sqlite3.connect(
                read_only_uri(str(source)), uri=True, isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise SplitError(f"cannot open {source}: {exc}") from exc
        try:
            # One read transaction for every SELECT, so a hook that writes
            # the store mid-read can't pair a belief's old posterior with
            # its new feedback rows. Closing the connection ends it.
            conn.execute("BEGIN")
            ids, ingested, anchored = _read_locations(conn, work_tree)
            routed, not_copied = _route(
                work_tree, source, ids, ingested, anchored,
            )
            wanted: set[str] = set()
            for _, bound in routed.values():
                wanted |= bound
            rows = _read_rows(conn, wanted)
        except sqlite3.Error as exc:
            raise SplitError(f"cannot read {source}: {exc}") from exc
        finally:
            conn.close()
    reports = tuple(
        (_apply_target if apply else _dry_run_target)(rows, selection, bound)
        for _, (selection, bound) in sorted(routed.items())
    )
    return SplitReport(work_tree, source, apply, reports, not_copied)


_LABELS: Final[dict[str, str]] = {
    SKIP_NO_FILE_SOURCE: "no file source",
    SKIP_NO_ROOT: "no root",
    SKIP_OUTSIDE_WORK_TREE: "outside the work tree",
    SKIP_MISSING_ROOT: "missing root",
    SKIP_SEVERAL_STORES: "several stores",
    SKIP_SOURCE_STORE: "stays in the repository store",
    SKIP_ALREADY_PRESENT: "already present",
    SKIP_LOCKED: "locked",
    SKIP_RETIRED: "retired",
    SKIP_SUPERSEDED: "superseded",
    "support_evidence": "support evidence",
}


def _counts_line(counts: dict[str, int], *, skip_zero: bool) -> str:
    parts = [
        f"{_LABELS.get(key, key)} {n}"
        for key, n in counts.items() if n or not skip_zero
    ]
    return ", ".join(parts) if parts else "none"


def format_report(report: SplitReport) -> str:
    """The text `aelf store split` prints."""
    mode = "applied" if report.applied else "dry run"
    verb = "copied" if report.applied else "would copy"
    lines = [
        f"store split ({mode}): {report.work_tree}",
        f"source: {report.source}",
    ]
    if not report.targets:
        lines.append("no marker store receives a belief")
    for t in report.targets:
        marker = "" if t.marker is None else f" (marker {t.marker})"
        lines.append(f"target: {t.store}{marker}")
        if t.error is not None:
            lines.append(f"  failed, nothing copied: {t.error}")
        else:
            lines.append(f"  {verb}: {_counts_line(t.copied, skip_zero=False)}")
        lines.append(f"  skipped: {_counts_line(t.skipped, skip_zero=True)}")
    lines.append(
        f"not copied: {_counts_line(report.not_copied, skip_zero=True)}",
    )
    return "\n".join(lines)
