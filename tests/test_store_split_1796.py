"""`aelf store split` copies file-sourced beliefs into marker stores (#1796).

Every test builds a temporary repository and temporary stores. The source
is the repository store, `<git-common-dir>/aelfrice/memory.db`, filled
through `record_ingest` and the derivation worker, the path the onboard
handshake takes, so the beliefs carry the ingest rows and the document
anchors a real onboard writes.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from aelfrice import cli, db_paths, store_split
from aelfrice.classification import (
    HostClassification,
    accept_classifications,
    start_onboard_session,
)
from aelfrice.config_discovery import CONFIG_FILENAME
from aelfrice.derivation_worker import run_worker
from aelfrice.doc_linker_types import ANCHOR_MANUAL
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    INGEST_SOURCE_FILESYSTEM,
    LOCK_NONE,
    LOCK_USER,
    ONBOARD_STATE_COMPLETED,
    SUPPORT_EVIDENCE_KIND_REWORDED,
    Belief,
    Edge,
    OnboardSession,
)
from aelfrice.store import MemoryStore

pytestmark = pytest.mark.timeout(60)

_TS = "2026-10-01T00:00:00+00:00"


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, timeout=30,
    )


def _marker(folder: Path, value: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    config = folder / CONFIG_FILENAME
    config.write_text(f'[store]\npath = "{value}"\n', encoding="utf-8")
    return config


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A sandboxed home and no store override, inside `tmp_path`."""
    monkeypatch.delenv("AELFRICE_DB", raising=False)
    monkeypatch.delenv("AELFRICE_TRANSCRIPTS_DIR", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(db_paths, "DEFAULT_DB_DIR", home / ".aelfrice")
    db_paths.clear_git_common_dir_cache()
    return tmp_path


@pytest.fixture()
def repo(env: Path) -> Path:
    """A repository whose `a/` folder has a marker store."""
    root = (env / "mono").resolve()
    root.mkdir()
    _git("init", "-q", cwd=root)
    _marker(root / "a", ".aelfrice/store")
    (root / "a" / "docs").mkdir()
    (root / "b").mkdir()
    return root


def _source_path(repo: Path) -> Path:
    return repo / ".git" / "aelfrice" / "memory.db"


def _target_path(repo: Path) -> Path:
    return repo / "a" / ".aelfrice" / "store" / "memory.db"


def _open(path: Path) -> MemoryStore:
    path.parent.mkdir(parents=True, exist_ok=True)
    return MemoryStore(str(path))


def _open_target(repo: Path) -> MemoryStore:
    """Open the `a/` marker store the way aelfrice opens it, with its
    identity, so opening it again changes no row."""
    selection = db_paths.resolve_store_in_work_tree(repo / "a", repo)
    selection.path.parent.mkdir(parents=True, exist_ok=True)
    return MemoryStore(
        str(selection.path),
        project_context_default=db_paths.store_identity(selection, create=True),
    )


def _onboard(store: MemoryStore, root: Path | str, sid: str) -> None:
    store.insert_onboard_session(OnboardSession(
        session_id=sid, repo_path=str(root), state=ONBOARD_STATE_COMPLETED,
        candidates_json="[]", created_at=_TS, completed_at=_TS,
    ))


def _ingest(store: MemoryStore, source: str, text: str, sid: str | None) -> str:
    """Ingest one candidate the way an onboard does; return its belief id."""
    log_id = store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM, source_path=source,
        raw_text=text, session_id=sid, ts=_TS,
        raw_meta={"call_site": CORROBORATION_SOURCE_FILESYSTEM_INGEST},
    )
    run_worker(store)
    entry = store.get_ingest_log_entry(log_id)
    assert entry is not None
    ids = cast("list[str]", entry["derived_belief_ids"])
    assert len(ids) == 1, entry
    return ids[0]


def _plain(bid: str, text: str) -> Belief:
    return Belief(
        id=bid, content=text, content_hash=f"{bid}-hash", alpha=1.0,
        beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE, locked_at=None,
        created_at=_TS, last_retrieved_at=None,
    )


def _text(n: int) -> str:
    return (
        f"Module number {n} parses the configuration file and validates "
        f"every key it reads before the server starts."
    )


def _run(
    repo: Path, *flags: str, capsys: pytest.CaptureFixture[str],
) -> tuple[int, str, str]:
    capsys.readouterr()
    buf = io.StringIO()
    code = cli.main(["store", "split", *flags, "--path", str(repo)], out=buf)
    return code, buf.getvalue(), capsys.readouterr().err


def _report(
    repo: Path, *flags: str, capsys: pytest.CaptureFixture[str],
) -> dict[str, object]:
    code, out, err = _run(repo, "--json", *flags, capsys=capsys)
    assert code == 0, err
    parsed: dict[str, object] = json.loads(out)
    return parsed


def _only_target(report: dict[str, object]) -> dict[str, dict[str, int]]:
    targets = cast("list[dict[str, dict[str, int]]]", report["targets"])
    assert len(targets) == 1, targets
    return targets[0]


def _file_hash(path: Path) -> str | None:
    """Digest of a store's database and write-ahead log, or None."""
    if not path.exists():
        return None
    h = hashlib.sha256(path.read_bytes())
    wal = path.with_name(path.name + "-wal")
    if wal.exists():
        h.update(wal.read_bytes())
    return h.hexdigest()


_TABLES = (
    "beliefs", "edges", "feedback_history", "belief_corroborations",
    "belief_documents", "belief_support_evidence",
)


def _rows(path: Path) -> dict[str, list[tuple[object, ...]]]:
    """Every row of the tables a copy writes, read `mode=ro`."""
    conn = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
    try:
        return {
            t: sorted(conn.execute(f"SELECT * FROM {t}").fetchall(), key=repr)
            for t in _TABLES
        }
    finally:
        conn.close()


# --- parsing ---------------------------------------------------------------


def test_the_scanner_file_sources_qualify_and_nothing_else_does() -> None:
    rel = store_split.file_source_rel
    assert rel("doc:a/docs/x.md:p3") == "a/docs/x.md"
    assert rel("doc:a/with:colon.md:p0") == "a/with:colon.md"
    assert rel("ast:a/m.py:module") == "a/m.py"
    assert rel("ast:a/m.py:func:run") == "a/m.py"
    assert rel("ast:a/m.py:class:Thing") == "a/m.py"
    for other in (
        "git:abc123", "transcript", "lock", "doc:a/x.md", "ast:a/m.py:method:x",
        "doc:/abs/x.md:p0", "doc:../x.md:p0", "doc:a/../x.md:p0", "",
    ):
        assert rel(other) is None, other
    uri = store_split.doc_uri_rel
    assert uri("file:doc:a/x.md:p0") == "a/x.md"
    assert uri("file:doc:a/x.md:p0#hint") == "a/x.md"
    assert uri("file:doc:a/x#y.md:p0") == "a/x#y.md"
    assert uri("file:doc:a/x#y.md:p0#hint") == "a/x#y.md"
    assert uri("file:transcript#s/t/h") is None
    assert uri("file:///abs/doc:a/x.md:p0") is None
    assert uri("doc:a/x.md:p0") is None


# --- which beliefs qualify -------------------------------------------------


def test_only_beliefs_with_a_file_source_are_copied(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        _ingest(src, "doc:a/docs/x.md:p0", _text(1), "s1")
        _ingest(src, "ast:a/m.py:func:run", _text(2), "s1")
        _ingest(src, "git:abc123", _text(3), "s1")
        src.insert_belief(_plain("nosource0000001", _text(4)))
    finally:
        src.close()
    report = _report(repo, capsys=capsys)
    target = _only_target(report)
    assert target["copied"]["beliefs"] == 2
    assert report["not_copied"] == {
        "no_file_source": 2, "no_root": 0, "outside_work_tree": 0,
        "missing_root": 0, "several_stores": 0, "source_store": 0,
    }


def test_the_root_comes_from_the_onboard_session(
    repo: Path, env: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Every ingest row must name one usable root, or the belief is
    skipped as `no_root`."""
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        _onboard(src, repo / "a", "s-sub")
        _onboard(src, repo / "b", "s2")
        _onboard(src, "relative/path", "s-rel")
        _onboard(src, env / "elsewhere", "s-out")
        # Onboarded from `a/`: the rel is relative to that root.
        _ingest(src, "doc:docs/x.md:p0", _text(1), "s-sub")
        # The same paragraph from two roots: ambiguous.
        both = _ingest(src, "doc:a/docs/y.md:p0", _text(2), "s1")
        assert _ingest(src, "doc:a/docs/y.md:p0", _text(2), "s2") == both
        # No onboard session, and a relative root whose file isn't there.
        _ingest(src, "doc:a/docs/z.md:p0", _text(3), "no-such-session")
        _ingest(src, "doc:a/docs/w.md:p0", _text(4), "s-rel")
        _ingest(src, "doc:a/docs/v.md:p0", _text(5), None)
        # A root outside this work tree.
        _ingest(src, "doc:a/docs/u.md:p0", _text(6), "s-out")
    finally:
        src.close()
    report = _report(repo, capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 1
    not_copied = report["not_copied"]
    assert isinstance(not_copied, dict)
    assert not_copied["no_root"] == 4
    assert not_copied["outside_work_tree"] == 1


def _relative_session(
    src: MemoryStore, repo: Path, sid: str, *, total: int, present: int,
    n: int, root: str = ".",
) -> str:
    """A session onboarded from the relative `root`, with `total` distinct
    document paths under `a/docs/`, the first `present` of them written
    as files under the work tree. Returns the belief from the first."""
    _onboard(src, root, sid)
    rels = [f"a/docs/{sid}-{i}.md" for i in range(total)]
    for rel in rels[:present]:
        (repo / rel).write_text("text\n", encoding="utf-8")
    bid = _ingest(src, f"doc:{rels[0]}:p0", _text(n), sid)
    for rel in rels[1:]:
        src.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM, source_path=f"doc:{rel}:p0",
            raw_text=rel, session_id=sid, ts=_TS, derived_belief_ids=[],
        )
    return bid


@pytest.mark.parametrize(
    ("present", "total"), [(10, 10), (9, 10)], ids=["all", "at-the-bar"],
)
def test_a_relative_root_whose_files_are_there_routes(
    repo: Path, capsys: pytest.CaptureFixture[str], present: int, total: int,
) -> None:
    """A session onboarded from `.` places its beliefs from the work-tree
    root when at least 90% of its files are there; the bar is inclusive."""
    assert store_split.RELATIVE_ROOT_MIN_PRESENT == 0.9
    src = _open(_source_path(repo))
    try:
        bid = _relative_session(
            src, repo, "s-dot", total=total, present=present, n=1,
        )
    finally:
        src.close()
    report = _report(repo, "--apply", capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 1
    assert {r[0] for r in _rows(_target_path(repo))["beliefs"]} == {bid}


def test_a_relative_root_just_below_the_bar_is_no_root(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """8 of 9 files is just under 90%, so that session names no root,
    while a session with all its files there still routes."""
    src = _open(_source_path(repo))
    try:
        routed = _relative_session(src, repo, "s-all", total=10, present=10, n=1)
        _relative_session(src, repo, "s-below", total=9, present=8, n=2)
    finally:
        src.close()
    report = _report(repo, "--apply", capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 1
    assert cast("dict[str, int]", report["not_copied"])["no_root"] == 1
    assert {r[0] for r in _rows(_target_path(repo))["beliefs"]} == {routed}


def test_a_relative_root_that_resolves_outside_the_work_tree_is_outside(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """`..` resolves above the work-tree root, so the belief counts as
    outside the work tree, whether or not its files are there."""
    src = _open(_source_path(repo))
    try:
        _relative_session(
            src, repo, "s-up", total=1, present=0, n=1, root="..",
        )
    finally:
        src.close()
    report = _report(repo, capsys=capsys)
    assert report["targets"] == []
    not_copied = cast("dict[str, int]", report["not_copied"])
    assert not_copied["outside_work_tree"] == 1
    assert not_copied["no_root"] == 0


def test_a_relative_root_counts_the_files_under_the_folder_it_names(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A session onboarded from `a` names its files relative to `a/`, so
    the bar counts the files under `a/`, not under the work-tree root."""
    (repo / "a" / "docs" / "x.md").write_text("text\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "y.md").write_text("text\n", encoding="utf-8")
    src = _open(_source_path(repo))
    try:
        _onboard(src, "a", "s-in-a")
        routed = _ingest(src, "doc:docs/x.md:p0", _text(1), "s-in-a")
        # Its file exists only at the top, not under `a/`.
        _onboard(src, "a", "s-at-top")
        _ingest(src, "doc:docs/y.md:p0", _text(2), "s-at-top")
    finally:
        src.close()
    report = _report(repo, "--apply", capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 1
    assert cast("dict[str, int]", report["not_copied"])["no_root"] == 1
    assert {r[0] for r in _rows(_target_path(repo))["beliefs"]} == {routed}


def test_the_relative_root_bar_counts_distinct_files_not_paragraphs(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Nine paragraphs of one present file and one paragraph of a missing
    file are nine of ten source paths but one of two files, under the bar."""
    (repo / "a" / "docs" / "x.md").write_text("text\n", encoding="utf-8")
    src = _open(_source_path(repo))
    try:
        _onboard(src, ".", "s-dot")
        _ingest(src, "doc:a/docs/x.md:p0", _text(1), "s-dot")
        sources = [f"doc:a/docs/x.md:p{i}" for i in range(1, 9)]
        for source in [*sources, "doc:a/docs/gone.md:p0"]:
            src.record_ingest(
                source_kind=INGEST_SOURCE_FILESYSTEM, source_path=source,
                raw_text=source, session_id="s-dot", ts=_TS,
                derived_belief_ids=[],
            )
    finally:
        src.close()
    report = _report(repo, capsys=capsys)
    assert report["targets"] == []
    assert cast("dict[str, int]", report["not_copied"])["no_root"] == 1


def test_a_folder_at_a_session_file_path_does_not_count_as_its_file(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """8 of 10 paths are files and a 9th is a folder, so the session is
    under the bar; a folder would lift it to 90%."""
    src = _open(_source_path(repo))
    try:
        _relative_session(src, repo, "s-dir", total=10, present=8, n=1)
    finally:
        src.close()
    (repo / "a" / "docs" / "s-dir-8.md").mkdir()
    report = _report(repo, capsys=capsys)
    assert report["targets"] == []
    assert cast("dict[str, int]", report["not_copied"])["no_root"] == 1


def _other_case(repo: Path) -> Path:
    """`repo` with its name in upper case, skipping the test where that
    names no folder, as on a case-sensitive filesystem."""
    upper = repo.with_name(repo.name.upper())
    if upper.name == repo.name or not upper.is_dir():
        pytest.skip("needs a case-insensitive filesystem")
    return upper


def test_an_absolute_root_in_another_letter_case_is_the_work_tree(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A root recorded as `.../MONO` for the folder `.../mono` is compared
    to the work tree by file identity, so its beliefs route, and it is
    the same root as `.../mono`."""
    upper = _other_case(repo)
    src = _open(_source_path(repo))
    try:
        _onboard(src, upper, "s-upper")
        top = _ingest(src, "doc:a/docs/x.md:p0", _text(1), "s-upper")
        _onboard(src, repo, "s-lower")
        assert _ingest(src, "doc:a/docs/x.md:p0", _text(1), "s-lower") == top
        _onboard(src, upper / "a", "s-sub")
        sub = _ingest(src, "doc:docs/y.md:p0", _text(2), "s-sub")
    finally:
        src.close()
    report = _report(repo, "--apply", capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 2
    assert {r[0] for r in _rows(_target_path(repo))["beliefs"]} == {top, sub}


def test_a_relative_root_in_another_letter_case_keeps_the_bar(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """`../MONO` resolves into the work tree `.../mono`, so the 90% bar
    applies to it as it does to `.`."""
    upper = _other_case(repo)
    src = _open(_source_path(repo))
    try:
        routed = _relative_session(
            src, repo, "s-here", total=1, present=1, n=1,
            root=f"../{upper.name}",
        )
        _relative_session(
            src, repo, "s-gone", total=1, present=0, n=2,
            root=f"../{upper.name}",
        )
    finally:
        src.close()
    report = _report(repo, "--apply", capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 1
    not_copied = cast("dict[str, int]", report["not_copied"])
    assert not_copied["no_root"] == 1
    assert not_copied["outside_work_tree"] == 0
    assert {r[0] for r in _rows(_target_path(repo))["beliefs"]} == {routed}


def _repeated_session(
    src: MemoryStore, repo: Path, given: str, *, total: int, also: Path,
    also_present: int,
) -> str:
    """A session onboarded from the relative `given`, with `total`
    distinct paths under `docs/`, each written as a file under
    `<repo>/<given>`, and the first `also_present` of them written under
    `<also>/<given>` too. Returns the belief from the first."""
    _onboard(src, given, "s-rep")
    rels = [f"docs/s-rep-{i}.md" for i in range(total)]
    for i, rel in enumerate(rels):
        folders = [repo / given] + ([also / given] if i < also_present else [])
        for folder in folders:
            (folder / rel).parent.mkdir(parents=True, exist_ok=True)
            (folder / rel).write_text("text\n", encoding="utf-8")
    bid = _ingest(src, f"doc:{rels[0]}:p0", _text(1), "s-rep")
    for rel in rels[1:]:
        src.record_ingest(
            source_kind=INGEST_SOURCE_FILESYSTEM, source_path=f"doc:{rel}:p0",
            raw_text=rel, session_id="s-rep", ts=_TS, derived_belief_ids=[],
        )
    return bid


@pytest.mark.parametrize("given", [".", "lib"], ids=["dot", "below"])
@pytest.mark.parametrize(
    ("also_present", "copied"), [(10, 0), (9, 0), (8, 1)],
    ids=["all", "at-the-bar", "below-the-bar"],
)
def test_a_relative_session_also_present_under_a_marker_folder_is_no_root(
    repo: Path, capsys: pytest.CaptureFixture[str], given: str,
    also_present: int, copied: int,
) -> None:
    """The folder a relative onboard ran from isn't recorded. A session
    whose files reach the 90% bar under the work-tree root and also under
    the marker folder `a/` could have run from either, so its beliefs
    count as `no_root`. The bar is inclusive there too."""
    _marker(repo / given / "docs", ".aelfrice/store")
    src = _open(_source_path(repo))
    try:
        bid = _repeated_session(
            src, repo, given, total=10, also=repo / "a",
            also_present=also_present,
        )
    finally:
        src.close()
    report = _report(repo, "--apply", capsys=capsys)
    not_copied = cast("dict[str, int]", report["not_copied"])
    assert not_copied["no_root"] == 1 - copied
    target = (repo / given / "docs" / ".aelfrice" / "store" / "memory.db").resolve()
    if copied:
        assert _only_target(report)["copied"]["beliefs"] == 1
        assert {r[0] for r in _rows(target)["beliefs"]} == {bid}
    else:
        assert report["targets"] == []
        assert not target.exists()


def test_only_a_marker_folder_makes_a_relative_session_ambiguous(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """`b/` has no project config file, and `a/c/` has one with no
    `[store]` table, so its store comes from the marker in `a/`. The
    files they repeat leave the session's root alone."""
    _marker(repo / "docs", ".aelfrice/store")
    plain = repo / "a" / "c"
    (plain / "docs").mkdir(parents=True)
    (plain / CONFIG_FILENAME).write_text("# no store table\n", encoding="utf-8")
    src = _open(_source_path(repo))
    try:
        bid = _repeated_session(
            src, repo, ".", total=2, also=repo / "b", also_present=2,
        )
    finally:
        src.close()
    for i in range(2):
        (plain / "docs" / f"s-rep-{i}.md").write_text("text\n", encoding="utf-8")
    report = _report(repo, "--apply", capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 1
    target = repo / "docs" / ".aelfrice" / "store" / "memory.db"
    assert {r[0] for r in _rows(target)["beliefs"]} == {bid}


@pytest.mark.parametrize("start", ["root", "pkg"])
def test_an_onboard_from_the_root_is_placed_and_one_from_a_marker_is_not(
    repo: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch, start: str,
) -> None:
    """The CONFIG.md example: `pkg/` holds a marker, and the work-tree
    root repeats its `README.md` and `docs/index.md`. A real `onboard .`
    from `pkg/` reaches the bar under both folders, so it names no root.
    One from the root also reads `pkg/README.md` and `pkg/docs/index.md`,
    which aren't under `pkg/pkg/`, so it stays under the bar there and
    its `pkg/` beliefs are copied."""
    _marker(repo / "pkg", ".aelfrice/pkg")
    for base, n in ((repo, 1), (repo / "pkg", 3)):
        (base / "docs").mkdir(exist_ok=True)
        (base / "README.md").write_text(f"{_text(n)}\n", encoding="utf-8")
        (base / "docs" / "index.md").write_text(
            f"{_text(n + 1)}\n", encoding="utf-8",
        )
    monkeypatch.chdir(repo if start == "root" else repo / "pkg")
    src = _open(_source_path(repo))
    try:
        started = start_onboard_session(src, Path("."))
        accept_classifications(src, started.session_id, [
            HostClassification(index=s.index, belief_type="factual", persist=True)
            for s in started.sentences
        ])
    finally:
        src.close()
    sources = {s.source for s in started.sentences}
    assert sources == (
        {"doc:README.md:p0", "doc:docs/index.md:p0"}
        | ({"doc:pkg/README.md:p0", "doc:pkg/docs/index.md:p0"}
           if start == "root" else set())
    ), sources
    report = _report(repo, "--apply", capsys=capsys)
    not_copied = cast("dict[str, int]", report["not_copied"])
    if start == "pkg":
        assert not_copied["no_root"] == 2
        assert report["targets"] == []
    else:
        assert not_copied["no_root"] == 0
        assert not_copied["source_store"] == 2
        assert _only_target(report)["copied"]["beliefs"] == 2


@pytest.mark.parametrize("case", ["same", "other"], ids=["dot", "other-case"])
def test_a_marker_at_the_work_tree_root_leaves_a_relative_session_alone(
    repo: Path, capsys: pytest.CaptureFixture[str], case: str,
) -> None:
    """A marker folder that resolves to the session's own root is the
    root it was scored under, not a second candidate. It is compared by
    file identity, so `../MONO` for the work tree `.../mono` is that
    root too."""
    given = "." if case == "same" else f"../{_other_case(repo).name}"
    _marker(repo, ".aelfrice/root")
    src = _open(_source_path(repo))
    try:
        bid = _relative_session(
            src, repo, "s-rel", total=2, present=2, n=1, root=given,
        )
    finally:
        src.close()
    report = _report(repo, "--apply", capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 1
    assert {r[0] for r in _rows(_target_path(repo))["beliefs"]} == {bid}


def test_an_onboard_root_in_a_nested_worktree_is_outside_the_work_tree(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A linked worktree inside this one shares the repository store, but
    its files are not this work tree's, so a marker here can't claim them."""
    _marker(repo, ".aelfrice/store")
    _git(
        "-c", "user.name=t", "-c", "user.email=t@example.com",
        "commit", "-q", "--allow-empty", "-m", "init", cwd=repo,
    )
    _git("worktree", "add", "-q", "wt/x", cwd=repo)
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo / "wt" / "x", "s-x")
        _onboard(src, repo, "s1")
        _ingest(src, "doc:docs/f.md:p1", _text(1), "s-x")
        here = _ingest(src, "doc:docs/g.md:p1", _text(2), "s1")
    finally:
        src.close()
    report = _report(repo, "--apply", capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 1
    not_copied = cast("dict[str, int]", report["not_copied"])
    assert not_copied["outside_work_tree"] == 1
    root_store = repo / ".aelfrice" / "store" / "memory.db"
    assert {r[0] for r in _rows(root_store)["beliefs"]} == {here}


def test_an_onboard_root_below_a_nested_worktree_root_is_outside(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The `.git` entry sits in a folder above the onboard root, not in
    the root itself, so every folder on the way down is checked."""
    _marker(repo, ".aelfrice/store")
    _git(
        "-c", "user.name=t", "-c", "user.email=t@example.com",
        "commit", "-q", "--allow-empty", "-m", "init", cwd=repo,
    )
    _git("worktree", "add", "-q", "wt/x", cwd=repo)
    (repo / "wt" / "x" / "docs").mkdir()
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo / "wt" / "x" / "docs", "s-x")
        _onboard(src, repo, "s1")
        _ingest(src, "doc:f.md:p1", _text(1), "s-x")
        here = _ingest(src, "doc:docs/g.md:p1", _text(2), "s1")
    finally:
        src.close()
    report = _report(repo, "--apply", capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 1
    not_copied = cast("dict[str, int]", report["not_copied"])
    assert not_copied["outside_work_tree"] == 1
    root_store = repo / ".aelfrice" / "store" / "memory.db"
    assert {r[0] for r in _rows(root_store)["beliefs"]} == {here}


def test_an_onboard_root_in_a_removed_nested_worktree_is_not_copied(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """`git worktree remove` deletes the folder and its `.git` entry, so
    the root no longer looks like a worktree. Its belief must not fall to
    the marker above it, and the root counts as missing."""
    _marker(repo, "git:whole")
    _git(
        "-c", "user.name=t", "-c", "user.email=t@example.com",
        "commit", "-q", "--allow-empty", "-m", "init", cwd=repo,
    )
    _git("worktree", "add", "-q", ".wt/x", cwd=repo)
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo / ".wt" / "x", "s-x")
        _ingest(src, "doc:a/docs/x.md:p1", _text(1), "s-x")
    finally:
        src.close()
    _git("worktree", "remove", "--force", ".wt/x", cwd=repo)
    assert not (repo / ".wt" / "x").exists()
    report = _report(repo, "--apply", capsys=capsys)
    assert report["targets"] == []
    not_copied = cast("dict[str, int]", report["not_copied"])
    assert not_copied["missing_root"] == 1
    assert not_copied["source_store"] == 0
    assert not (repo / ".git" / "aelfrice" / "stores").exists()


def test_a_relative_root_from_a_removed_worktree_resolves_here(
    repo: Path, env: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A `.` session doesn't record the worktree it ran in, so with its
    files present here its belief lands in this work tree's marker store.
    The same belief under an absolute root in that worktree doesn't."""
    _git(
        "-c", "user.name=t", "-c", "user.email=t@example.com",
        "commit", "-q", "--allow-empty", "-m", "init", cwd=repo,
    )
    other = env / "other"
    _git("worktree", "add", "-q", str(other), cwd=repo)
    (repo / "a" / "docs" / "x.md").write_text("text\n", encoding="utf-8")
    src = _open(_source_path(repo))
    try:
        _onboard(src, ".", "s-rel")
        routed = _ingest(src, "doc:a/docs/x.md:p0", _text(1), "s-rel")
        _onboard(src, other, "s-abs")
        _ingest(src, "doc:a/docs/x.md:p1", _text(2), "s-abs")
    finally:
        src.close()
    _git("worktree", "remove", "--force", str(other), cwd=repo)
    assert not other.exists()
    report = _report(repo, "--apply", capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 1
    assert cast("dict[str, int]", report["not_copied"])["outside_work_tree"] == 1
    assert {r[0] for r in _rows(_target_path(repo))["beliefs"]} == {routed}


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs POSIX permissions that the user can't bypass",
)
def test_an_onboard_root_it_cannot_search_is_refused(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A root whose `.git` entry can't be checked may be a worktree, so
    the command stops and names the root instead of placing its belief.
    The marker walk finds the root first, so the root check is also
    called directly."""
    root = repo / "a" / "docs"
    src = _open(_source_path(repo))
    try:
        _onboard(src, root, "s1")
        _ingest(src, "doc:x.md:p0", _text(1), "s1")
    finally:
        src.close()
    root_skip = store_split._root_skip  # pyright: ignore[reportPrivateUsage]
    root.chmod(0o644)
    try:
        code, out, err = _run(repo, "--apply", capsys=capsys)
        with pytest.raises(db_paths.FolderAccessError) as raised:
            root_skip(repo, Path("a", "docs"))
    finally:
        root.chmod(0o755)
    assert raised.value.folder == root
    assert code == 2
    assert out == ""
    assert err == f"aelf store split: cannot list or search {root}\n"
    assert not _target_path(repo).parent.exists()


def test_the_folder_is_the_deepest_real_folder_in_the_work_tree(
    repo: Path, env: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A symlink, a missing folder, and a nested repository each stop the
    walk, and the folder above them picks the store."""
    _marker(repo / "a" / "inner", ".aelfrice/store")
    inner_store = repo / "a" / "inner" / ".aelfrice" / "store" / "memory.db"
    (repo / "link").symlink_to(repo / "a" / "inner", target_is_directory=True)
    nested = repo / "n"
    nested.mkdir()
    _git("init", "-q", cwd=nested)
    _marker(nested, ".aelfrice/store")
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        _ingest(src, "doc:a/inner/x.md:p0", _text(1), "s1")
        # `gone/` doesn't exist, so the walk stops at `a/inner/`.
        _ingest(src, "doc:a/inner/gone/deeper/x.md:p0", _text(2), "s1")
        # `link/` is a symlink into `a/inner/`, so the walk stops at the root.
        _ingest(src, "doc:link/x.md:p0", _text(3), "s1")
        # `n/` is a nested repository with its own marker.
        _ingest(src, "doc:n/x.md:p0", _text(4), "s1")
    finally:
        src.close()
    report = _report(repo, capsys=capsys)
    target = _only_target(report)
    assert target["store"] == str(inner_store)
    assert target["copied"]["beliefs"] == 2
    not_copied = report["not_copied"]
    assert isinstance(not_copied, dict)
    assert not_copied["source_store"] == 2


def test_a_belief_whose_files_resolve_to_two_stores_is_skipped(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        bid = _ingest(src, "doc:a/docs/x.md:p0", _text(1), "s1")
        assert _ingest(src, "doc:b/x.md:p0", _text(1), "s1") == bid
    finally:
        src.close()
    report = _report(repo, capsys=capsys)
    assert report["targets"] == []
    not_copied = report["not_copied"]
    assert isinstance(not_copied, dict)
    assert not_copied["several_stores"] == 1


def test_two_markers_that_name_one_store_are_one_target(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """`c/` and `d/` both name `../shared`: one store under two spellings,
    so a belief in both folders is copied, and an edge between them lands
    in the dry run and in `--apply` alike."""
    _marker(repo / "c", "../shared")
    _marker(repo / "d", "../shared")
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        both = _ingest(src, "doc:c/x.md:p1", _text(1), "s1")
        assert _ingest(src, "doc:d/y.md:p1", _text(1), "s1") == both
        in_c = _ingest(src, "doc:c/z.md:p0", _text(2), "s1")
        in_d = _ingest(src, "doc:d/w.md:p0", _text(3), "s1")
        assert src.insert_edge(Edge(in_c, in_d, EDGE_RELATES_TO, 0.5))
    finally:
        src.close()
    for flags in ((), ("--apply",)):
        report = _report(repo, *flags, capsys=capsys)
        target = _only_target(report)
        assert target["copied"]["beliefs"] == 3, flags
        assert target["copied"]["edges"] == 1, flags
        not_copied = cast("dict[str, int]", report["not_copied"])
        assert not_copied["several_stores"] == 0, flags
    got = _rows(repo / "shared" / "memory.db")
    assert {r[0] for r in got["beliefs"]} == {both, in_c, in_d}


def test_a_store_seen_first_keeps_its_spelling_and_marker(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """`c/` resolves first, for a belief that also spans `e/` and so lands
    nowhere. The target that `d/`'s belief then reaches is the same file,
    and the report names it as `c/` spelled it, with `c/`'s marker."""
    c_marker = _marker(repo / "c", "../shared")
    _marker(repo / "d", "../shared")
    _marker(repo / "e", "../other")
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        for bid, text, rels in (
            ("aaaaaaaaaaaaaa1", _text(1), ("c/x.md", "e/y.md")),
            ("bbbbbbbbbbbbbb1", _text(2), ("d/w.md",)),
        ):
            src.insert_belief(_plain(bid, text))
            for rel in rels:
                src.record_ingest(
                    source_kind=INGEST_SOURCE_FILESYSTEM,
                    source_path=f"doc:{rel}:p0", raw_text=text,
                    session_id="s1", ts=_TS, derived_belief_ids=[bid],
                )
    finally:
        src.close()
    report = _report(repo, capsys=capsys)
    target = _only_target(report)
    assert target["store"] == str(repo / "c" / "../shared" / "memory.db")
    assert target["marker"] == str(c_marker)
    assert target["copied"]["beliefs"] == 1
    not_copied = cast("dict[str, int]", report["not_copied"])
    assert not_copied["several_stores"] == 1


def test_a_document_anchor_adds_its_folder(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """An anchor whose path differs from the ingest row's counts too: a
    belief ingested from `b/` and anchored in `a/` spans two stores."""
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        bid = _ingest(src, "doc:b/x.md:p0", _text(1), "s1")
        src.link_belief_to_document(  # pyright: ignore[reportUnknownMemberType]
            belief_id=bid, doc_uri="file:doc:a/docs/x.md:p0",
            anchor_type=ANCHOR_MANUAL, position_hint=None,
        )
    finally:
        src.close()
    report = _report(repo, capsys=capsys)
    assert report["targets"] == []
    not_copied = cast("dict[str, int]", report["not_copied"])
    assert not_copied["several_stores"] == 1
    assert not_copied["source_store"] == 0


# --- dry run ---------------------------------------------------------------


def _seed_pair(repo: Path) -> tuple[str, str]:
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        a1 = _ingest(src, "doc:a/docs/x.md:p0", _text(1), "s1")
        b1 = _ingest(src, "doc:b/y.md:p0", _text(2), "s1")
    finally:
        src.close()
    return a1, b1


def test_a_dry_run_writes_nothing(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_pair(repo)
    source = _source_path(repo)
    before = _file_hash(source)
    code, out, err = _run(repo, capsys=capsys)
    assert code == 0, err
    assert "would copy: beliefs 1" in out
    assert _file_hash(source) == before
    assert not _target_path(repo).parent.exists()

    _open(_target_path(repo)).close()
    target_before = _file_hash(_target_path(repo))
    assert target_before is not None
    code, out, err = _run(repo, capsys=capsys)
    assert code == 0, err
    assert "would copy: beliefs 1" in out
    assert _file_hash(source) == before
    assert _file_hash(_target_path(repo)) == target_before
    assert not (_target_path(repo).parent / "identity").exists()


_REPO = Path(__file__).resolve().parents[1]
_CONFIG_DOC = _REPO / "docs" / "user" / "CONFIG.md"


def _split_section() -> str:
    """The "Split an existing store" section of the configuration doc."""
    text = _CONFIG_DOC.read_text(encoding="utf-8")
    start = text.index("### Split an existing store")
    return text[start:text.index("\n## ", start)]


def test_a_run_leaves_only_sqlite_sidecars_beside_the_source_and_says_so(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The database file is never written, but a `mode=ro` open of a WAL
    store creates its `-wal` and `-shm` files, and the doc names them."""
    _seed_pair(repo)
    source = _source_path(repo)
    before = source.read_bytes()
    names = {p.name for p in source.parent.iterdir()}
    for flags in ((), ("--apply",)):
        code, _, err = _run(repo, *flags, capsys=capsys)
        assert code == 0, err
        assert source.read_bytes() == before, flags
        added = {p.name for p in source.parent.iterdir()} - names
        assert added <= {"memory.db-wal", "memory.db-shm"}, added
    section = _split_section()
    assert "never writes to it" not in section
    assert "`-wal` and `-shm`" in section


def test_the_doc_names_every_skip_reason_by_its_key_and_its_label() -> None:
    """The text report prints a label, not the `--json` key, so each
    reason's table row gives both."""
    section = _split_section()
    labels = store_split._LABELS  # pyright: ignore[reportPrivateUsage]
    for reason in (*store_split.UNROUTED_REASONS, *store_split.TARGET_REASONS):
        row = f"| `{reason}` | {labels[reason]} | "
        assert row in section, row


def test_a_dry_run_reports_a_target_store_it_cannot_read(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_pair(repo)
    target = _target_path(repo)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not a database " * 64)
    code, out, err = _run(repo, "--json", capsys=capsys)
    assert code == 1, err
    report = _only_target(json.loads(out))
    assert "not a database" in str(report["error"])
    code, out, _ = _run(repo, capsys=capsys)
    assert code == 1
    assert "failed, nothing copied: file is not a database" in out


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs POSIX permissions that the user can't bypass",
)
def test_a_dry_run_reports_a_target_store_it_cannot_open_read_only(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A WAL store with no sidecars, in a folder you can't write, can't
    be read: SQLite must create its `-shm` file first. The report marks
    that target as failed and still reports the other target."""
    _seed_pair(repo)
    _marker(repo / "b", ".aelfrice/store")
    target = _target_path(repo)
    _open(target).close()
    assert {p.name for p in target.parent.iterdir()} == {"memory.db"}
    target.parent.chmod(0o555)
    try:
        code, out, err = _run(repo, "--json", capsys=capsys)
        text_code, text_out, text_err = _run(repo, capsys=capsys)
    finally:
        target.parent.chmod(0o755)
    assert (code, text_code) == (1, 1)
    assert (err, text_err) == ("", "")
    targets = {
        str(t["store"]): t
        for t in cast("list[dict[str, object]]", json.loads(out)["targets"])
    }
    assert "without write access" in str(targets[str(target)]["error"])
    other = targets[str(repo / "b" / ".aelfrice" / "store" / "memory.db")]
    assert other["error"] is None
    assert cast("dict[str, int]", other["copied"])["beliefs"] == 1
    assert "failed, nothing copied: cannot read" in text_out


# --- apply -------------------------------------------------------------------


def _seed_graph(repo: Path) -> dict[str, str]:
    """Two beliefs under `a/`, one under `b/`, one already in the target,
    with rows of every kind a copy carries."""
    t = _open_target(repo)
    try:
        t.insert_belief(_plain("present00000001", _text(9)))
    finally:
        t.close()
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        ids = {
            "a1": _ingest(src, "doc:a/docs/x.md:p0", _text(1), "s1"),
            "a2": _ingest(src, "ast:a/m.py:module", _text(2), "s1"),
            "b1": _ingest(src, "doc:b/y.md:p0", _text(3), "s1"),
        }
        src.insert_belief(_plain("present00000001", _text(9)))
        a1, a2, b1 = ids["a1"], ids["a2"], ids["b1"]
        for edge in (
            Edge(a1, a2, EDGE_RELATES_TO, 0.5, anchor_text="both ends"),
            Edge(a1, b1, EDGE_RELATES_TO, 0.5),
            Edge(a2, "present00000001", EDGE_RELATES_TO, 0.5),
        ):
            assert src.insert_edge(edge)
        src.insert_feedback_event(a1, 1.0, "test", _TS)
        src.insert_feedback_event(b1, 1.0, "test", _TS)
        src.record_corroboration(
            a1, source_type=CORROBORATION_SOURCE_FILESYSTEM_INGEST,
            session_id="s1", ts=_TS,
        )
        for supporter, target in ((a1, a2), (a1, b1)):
            src.record_support_evidence(
                supporter, target, kind=SUPPORT_EVIDENCE_KIND_REWORDED,
                target_sentence_index=0, supporter_session_id="s1",
                supporter_ingest_log_id=None, matched_span="span", ts=_TS,
            )
    finally:
        src.close()
    return ids


def test_apply_copies_beliefs_with_their_rows_and_both_end_edges(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    ids = _seed_graph(repo)
    a1, a2, b1 = ids["a1"], ids["a2"], ids["b1"]
    report = _report(repo, "--apply", capsys=capsys)
    target = _only_target(report)
    assert target["copied"] == {
        "beliefs": 2, "documents": 2, "feedback": 1, "corroborations": 1,
        "edges": 2, "support_evidence": 1,
    }
    got = _rows(_target_path(repo))
    assert {r[0] for r in got["beliefs"]} == {a1, a2, "present00000001"}
    assert {(r[0], r[1]) for r in got["edges"]} == {
        (a1, a2), (a2, "present00000001"),
    }
    assert {r[0] for r in got["belief_documents"]} == {a1, a2}
    assert {r[1] for r in got["feedback_history"]} == {a1}
    assert {r[1] for r in got["belief_corroborations"]} == {a1}
    assert {(r[0], r[1]) for r in got["belief_support_evidence"]} == {(a1, a2)}
    assert b1 not in {r[0] for r in got["beliefs"]}
    source = _rows(_source_path(repo))
    for table in ("belief_documents",):
        want = {r for r in source[table] if r[0] in (a1, a2)}
        assert set(got[table]) == want


def test_an_edge_the_target_already_holds_is_not_copied_again(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The edges table has no foreign key, so a target can hold an edge
    before it holds either belief. Inserting it again would fail the
    whole target on the primary key."""
    ids = _seed_graph(repo)
    t = _open_target(repo)
    try:
        assert t.insert_edge(Edge(ids["a1"], ids["a2"], EDGE_RELATES_TO, 0.5))
        assert t.get_belief(ids["a1"]) is None
    finally:
        t.close()
    for flags in ((), ("--apply",)):
        target = _only_target(_report(repo, *flags, capsys=capsys))
        assert target["error"] is None, flags
        assert target["copied"]["beliefs"] == 2, flags
        assert target["copied"]["edges"] == 1, flags
    got = _rows(_target_path(repo))
    assert {(r[0], r[1]) for r in got["edges"]} == {
        (ids["a1"], ids["a2"]), (ids["a2"], "present00000001"),
    }


def test_a_copy_is_restamped_with_the_target_identity(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    a1, _ = _seed_pair(repo)
    src = MemoryStore(str(_source_path(repo)))
    try:
        stamped = src.get_belief(a1)
        assert stamped is not None
        src.update_belief(dataclasses.replace(stamped, project_context="repo-x"))
    finally:
        src.close()
    _report(repo, "--apply", capsys=capsys)
    selection = db_paths.resolve_store_in_work_tree(repo / "a", repo)
    identity = db_paths.store_identity(selection)
    assert identity.startswith("a-")
    t = MemoryStore(str(_target_path(repo)), read_only=True)
    try:
        copied = t.get_belief(a1)
    finally:
        t.close()
    assert copied is not None
    assert copied.project_context == identity
    assert copied.id == a1


def test_a_copy_of_a_shared_or_global_belief_keeps_an_empty_context(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The store stamps its identity only on project-scope rows, so a
    federation-shared row stays visible under every context."""
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        ids = {
            scope: _ingest(src, f"doc:a/docs/{n}.md:p0", _text(n), "s1")
            for n, scope in enumerate(("project", "global", "shared:team"))
        }
        for scope, bid in ids.items():
            b = src.get_belief(bid)
            assert b is not None
            src.update_belief(dataclasses.replace(
                b, scope=scope, project_context="",
            ))
    finally:
        src.close()
    _report(repo, "--apply", capsys=capsys)
    selection = db_paths.resolve_store_in_work_tree(repo / "a", repo)
    identity = db_paths.store_identity(selection)
    conn = sqlite3.connect(_target_path(repo).as_uri() + "?mode=ro", uri=True)
    try:
        got = dict(conn.execute(
            "SELECT scope, project_context FROM beliefs",
        ).fetchall())
    finally:
        conn.close()
    assert got == {"project": identity, "global": "", "shared:team": ""}


def test_apply_skips_a_present_id_or_hash_and_a_second_apply_copies_nothing(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        by_id = _ingest(src, "doc:a/docs/x.md:p0", _text(1), "s1")
        by_hash = _ingest(src, "doc:a/docs/y.md:p0", _text(2), "s1")
        fresh = _ingest(src, "doc:a/docs/z.md:p0", _text(3), "s1")
        same_id = src.get_belief(by_id)
        same_hash = src.get_belief(by_hash)
    finally:
        src.close()
    assert same_id is not None and same_hash is not None
    t = _open_target(repo)
    try:
        t.insert_belief(dataclasses.replace(
            same_id, content="other text", content_hash="other-hash",
        ))
        t.insert_belief(dataclasses.replace(same_hash, id="otherid00000001"))
    finally:
        t.close()
    first = _only_target(_report(repo, "--apply", capsys=capsys))
    assert first["copied"]["beliefs"] == 1
    assert first["skipped"]["already_present"] == 2
    after_first = _rows(_target_path(repo))
    assert fresh in {r[0] for r in after_first["beliefs"]}
    second = _only_target(_report(repo, "--apply", capsys=capsys))
    assert all(n == 0 for n in second["copied"].values()), second
    assert second["skipped"]["already_present"] == 3
    assert _rows(_target_path(repo)) == after_first


def test_a_present_belief_brings_no_feedback_or_corroboration(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The rows of a belief skipped as already present stay behind."""
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        a1 = _ingest(src, "doc:a/docs/x.md:p0", _text(1), "s1")
        src.insert_feedback_event(a1, 1.0, "test", _TS)
        src.record_corroboration(
            a1, source_type=CORROBORATION_SOURCE_FILESYSTEM_INGEST,
            session_id="s1", ts=_TS,
        )
        belief = src.get_belief(a1)
    finally:
        src.close()
    assert belief is not None
    t = _open_target(repo)
    try:
        t.insert_belief(belief)
    finally:
        t.close()
    before = _rows(_target_path(repo))
    target = _only_target(_report(repo, "--apply", capsys=capsys))
    assert target["skipped"]["already_present"] == 1
    assert all(n == 0 for n in target["copied"].values()), target
    after = _rows(_target_path(repo))
    for table in ("feedback_history", "belief_corroborations"):
        assert after[table] == before[table], table


def test_a_failure_mid_copy_leaves_the_target_unchanged(
    repo: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_graph(repo)
    before = _rows(_target_path(repo))
    calls: list[str] = []

    def failing_insert_edge(self: MemoryStore, e: Edge) -> bool:
        calls.append(e.src)
        raise sqlite3.IntegrityError("injected mid-copy failure")

    monkeypatch.setattr(MemoryStore, "insert_edge", failing_insert_edge)
    code, out, err = _run(repo, "--apply", "--json", capsys=capsys)
    assert code == 1, err
    target = _only_target(json.loads(out))
    assert "injected mid-copy failure" in str(target["error"])
    assert calls, "the copy never reached the edges"
    assert _rows(_target_path(repo)) == before


# --- lock and retirement -----------------------------------------------------


def test_locked_retired_and_stranded_superseded_beliefs_are_reported(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    src = _open(_source_path(repo))
    try:
        _onboard(src, repo, "s1")
        locked = _ingest(src, "doc:a/docs/l.md:p0", _text(1), "s1")
        retired = _ingest(src, "doc:a/docs/r.md:p0", _text(2), "s1")
        # `old` is superseded by `new`, and both land: both are copied.
        old = _ingest(src, "doc:a/docs/o.md:p0", _text(3), "s1")
        new = _ingest(src, "doc:a/docs/n.md:p0", _text(4), "s1")
        # `mid` is superseded by `outside`, which stays in the source, and
        # `low` is superseded by `mid`: dropping `mid` strands `low`.
        outside = _ingest(src, "doc:b/x.md:p0", _text(5), "s1")
        mid = _ingest(src, "doc:a/docs/m.md:p0", _text(6), "s1")
        low = _ingest(src, "doc:a/docs/w.md:p0", _text(7), "s1")
        for src_id, dst_id in ((new, old), (outside, mid), (mid, low)):
            assert src.insert_edge(Edge(src_id, dst_id, EDGE_SUPERSEDES, 1.0))
        got = src.get_belief(locked)
        assert got is not None
        src.update_belief(dataclasses.replace(
            got, lock_level=LOCK_USER, locked_at=_TS,
        ))
        src.soft_delete_belief(retired, ts=_TS)
    finally:
        src.close()
    target = _only_target(_report(repo, "--apply", capsys=capsys))
    assert target["skipped"] == {
        "already_present": 0, "locked": 1, "retired": 1, "superseded": 2,
    }
    assert target["copied"]["beliefs"] == 2
    got_rows = _rows(_target_path(repo))
    assert {r[0] for r in got_rows["beliefs"]} == {old, new}
    assert {(r[0], r[1], r[2]) for r in got_rows["edges"]} == {
        (new, old, EDGE_SUPERSEDES),
    }


# --- refusals ---------------------------------------------------------------


def test_a_broken_marker_anywhere_aborts_before_writing(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_pair(repo)
    # A broken marker in a folder no belief comes from.
    _marker(repo / "b" / "deep", "store.db")
    before = _file_hash(_source_path(repo))
    code, out, err = _run(repo, "--apply", capsys=capsys)
    assert code == 2
    assert "store.db" in err and out == ""
    assert err.startswith("aelf store split: "), err
    assert _file_hash(_source_path(repo)) == before
    assert not _target_path(repo).parent.exists()


def test_a_broken_marker_in_a_nested_repository_is_not_checked(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A nested repository's markers belong to its own work tree."""
    _seed_pair(repo)
    nested = repo / "n"
    nested.mkdir()
    _git("init", "-q", cwd=nested)
    _marker(nested, "store.db")
    code, _, err = _run(repo, capsys=capsys)
    assert code == 0, err


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs POSIX permissions that the user can't bypass",
)
@pytest.mark.parametrize(
    "mode", [0, 0o111, 0o444], ids=["sealed", "search-only", "list-only"],
)
def test_a_folder_it_cannot_list_or_search_is_refused_before_writing(
    repo: Path, capsys: pytest.CaptureFixture[str], mode: int,
) -> None:
    """A folder without read or search permission, such as a container's
    data folder, may hold a marker or a `.git` entry the command can't
    check. A dry run and `--apply` alike stop with one line naming it."""
    _seed_pair(repo)
    before = _file_hash(_source_path(repo))
    sealed = repo / "b" / "pgdata"
    sealed.mkdir()
    sealed.chmod(mode)
    try:
        runs = [_run(repo, *flags, capsys=capsys) for flags in ((), ("--apply",))]
    finally:
        sealed.chmod(0o755)
    for code, out, err in runs:
        assert code == 2
        assert out == ""
        assert err == f"aelf store split: cannot list or search {sealed}\n"
    assert _file_hash(_source_path(repo)) == before
    assert not _target_path(repo).parent.exists()


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs POSIX permissions that the user can't bypass",
)
@pytest.mark.parametrize(
    "mode", [0, 0o111, 0o444], ids=["sealed", "search-only", "list-only"],
)
@pytest.mark.parametrize("below", [False, True], ids=["at", "below"])
def test_a_path_at_or_below_a_folder_it_cannot_list_or_search_is_refused(
    repo: Path, capsys: pytest.CaptureFixture[str], mode: int, below: bool,
) -> None:
    """git can't run in a folder it can't search, so a `--path` at or
    below one gets the same one-line refusal naming that folder."""
    _seed_pair(repo)
    before = _file_hash(_source_path(repo))
    sealed = repo / "b" / "pgdata"
    (sealed / "sub").mkdir(parents=True)
    start = sealed / "sub" if below else sealed
    sealed.chmod(mode)
    try:
        runs = [
            _run(start, *flags, capsys=capsys) for flags in ((), ("--apply",))
        ]
    finally:
        sealed.chmod(0o755)
    for code, out, err in runs:
        assert code == 2
        assert out == ""
        assert err == f"aelf store split: cannot list or search {sealed}\n"
    assert _file_hash(_source_path(repo)) == before
    assert not _target_path(repo).parent.exists()


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs POSIX permissions that the user can't bypass",
)
@pytest.mark.parametrize("mode", [0, 0o444], ids=["sealed", "list-only"])
@pytest.mark.parametrize("rel", [".", "sub"], ids=["here", "below"])
def test_a_relative_path_from_a_folder_it_cannot_search_is_refused(
    repo: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch, mode: int, rel: str,
) -> None:
    """A relative `--path` needs the current folder's path. Some systems,
    such as macOS, can't give it for a folder you can't search, so the
    refusal names the current folder instead of the path."""
    _seed_pair(repo)
    before = _file_hash(_source_path(repo))
    sealed = repo / "b" / "pgdata"
    (sealed / "sub").mkdir(parents=True)
    monkeypatch.chdir(sealed)
    sealed.chmod(mode)
    try:
        runs = [
            _run(Path(rel), *flags, capsys=capsys)
            for flags in ((), ("--apply",))
        ]
    finally:
        sealed.chmod(0o755)
    named = {
        f"aelf store split: cannot list or search {sealed}\n",
        "aelf store split: cannot list or search the current folder\n",
    }
    for code, out, err in runs:
        assert code == 2
        assert out == ""
        assert err in named
    assert _file_hash(_source_path(repo)) == before
    assert not _target_path(repo).parent.exists()


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows can't remove the current folder",
)
def test_a_relative_path_from_a_removed_folder_is_refused(
    repo: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gone = repo / "b" / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    code, out, err = _run(Path("."), capsys=capsys)
    assert code == 2
    assert out == ""
    assert err == "aelf store split: the current folder no longer exists\n"


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows can't remove the current folder",
)
def test_an_absolute_path_from_a_removed_folder_runs(
    repo: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absolute `--path` doesn't need the current folder's path, so a
    removed current folder doesn't stop the command."""
    _seed_pair(repo)
    gone = repo / "b" / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    code, out, err = _run(repo, capsys=capsys)
    assert code == 0, err
    assert "would copy: beliefs 1" in out
    assert not _target_path(repo).parent.exists()
    code, out, err = _run(repo, "--apply", capsys=capsys)
    assert code == 0, err
    assert "copied: beliefs 1" in out


def _run_here(
    *flags: str, capsys: pytest.CaptureFixture[str],
) -> tuple[int, str, str]:
    """`_run` with no `--path`, so the command starts from the current
    folder."""
    capsys.readouterr()
    buf = io.StringIO()
    code = cli.main(["store", "split", *flags], out=buf)
    return code, buf.getvalue(), capsys.readouterr().err


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs POSIX permissions that the user can't bypass",
)
@pytest.mark.parametrize("mode", [0, 0o444], ids=["sealed", "list-only"])
def test_no_path_from_a_folder_it_cannot_search_is_refused(
    repo: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch, mode: int,
) -> None:
    _seed_pair(repo)
    before = _file_hash(_source_path(repo))
    sealed = repo / "b" / "pgdata"
    sealed.mkdir()
    monkeypatch.chdir(sealed)
    sealed.chmod(mode)
    try:
        runs = [_run_here(*flags, capsys=capsys) for flags in ((), ("--apply",))]
    finally:
        sealed.chmod(0o755)
    named = {
        f"aelf store split: cannot list or search {sealed}\n",
        "aelf store split: cannot list or search the current folder\n",
    }
    for code, out, err in runs:
        assert code == 2
        assert out == ""
        assert err in named
    assert _file_hash(_source_path(repo)) == before
    assert not _target_path(repo).parent.exists()


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs POSIX permissions that the user can't bypass",
)
@pytest.mark.parametrize("mode", [0, 0o444], ids=["sealed", "list-only"])
@pytest.mark.parametrize("rel", [None, "."], ids=["no-path", "dot"])
def test_a_run_below_a_folder_it_cannot_search_names_that_folder(
    repo: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch, mode: int, rel: str | None,
) -> None:
    """macOS can't give the current folder's path from below a folder you
    can't search, though the current folder itself is fine. The refusal
    names the folder above it that can't be searched."""
    _seed_pair(repo)
    sealed = repo / "b" / "pgdata"
    (sealed / "sub").mkdir(parents=True)
    monkeypatch.chdir(sealed / "sub")
    sealed.chmod(mode)
    try:
        runs = [
            _run_here(*flags, capsys=capsys) if rel is None
            else _run(Path(rel), *flags, capsys=capsys)
            for flags in ((), ("--apply",))
        ]
    finally:
        sealed.chmod(0o755)
    for code, out, err in runs:
        assert code == 2
        assert out == ""
        assert err == f"aelf store split: cannot list or search {sealed}\n"
    assert not _target_path(repo).parent.exists()


@pytest.mark.skipif(
    sys.platform != "darwin" or os.geteuid() == 0,
    reason="needs macOS, whose getcwd fails below a folder you can't "
    "search, and POSIX permissions that the user can't bypass",
)
@pytest.mark.parametrize("mode", [0, 0o444], ids=["sealed", "list-only"])
@pytest.mark.parametrize("rel", [None, "."], ids=["no-path", "dot"])
def test_a_run_from_a_search_only_folder_below_one_it_cannot_search(
    repo: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch, mode: int, rel: str | None,
) -> None:
    """The current folder can be searched but not listed. A handle opened
    for search needs no read permission, so it still gives the path, and
    the refusal names the folder above it that can't be searched."""
    _seed_pair(repo)
    sealed = repo / "b" / "pgdata"
    sub = sealed / "sub"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    sub.chmod(0o111)
    sealed.chmod(mode)
    try:
        runs = [
            _run_here(*flags, capsys=capsys) if rel is None
            else _run(Path(rel), *flags, capsys=capsys)
            for flags in ((), ("--apply",))
        ]
    finally:
        sealed.chmod(0o755)
        sub.chmod(0o755)
    for code, out, err in runs:
        assert code == 2
        assert out == ""
        assert err == f"aelf store split: cannot list or search {sealed}\n"
    assert not _target_path(repo).parent.exists()


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows can't remove the current folder",
)
def test_no_path_from_a_removed_folder_is_refused(
    repo: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gone = repo / "b" / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    code, out, err = _run_here(capsys=capsys)
    assert code == 2
    assert out == ""
    assert err == "aelf store split: the current folder no longer exists\n"


def test_no_path_starts_from_the_current_folder(
    repo: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_pair(repo)
    monkeypatch.chdir(repo / "b")
    code, out, err = _run_here("--json", capsys=capsys)
    assert code == 0, err
    assert json.loads(out)["work_tree"] == str(repo)


@pytest.mark.skipif(
    sys.platform == "win32", reason="symlinks need extra rights on Windows",
)
def _other_repo(env: Path) -> Path:
    """A second repository, `other`, whose `o/` marker store receives one
    belief, with an empty `sub/` folder."""
    other = (env / "other").resolve()
    (other / "sub").mkdir(parents=True)
    _git("init", "-q", cwd=other)
    _marker(other / "o", ".aelfrice/store")
    (other / "o" / "docs").mkdir()
    src = _open(_source_path(other))
    try:
        _onboard(src, other, "s1")
        _ingest(src, "doc:o/docs/z.md:p0", _text(3), "s1")
    finally:
        src.close()
    return other


def test_a_path_through_a_symlink_and_up_splits_the_folder_it_names(
    repo: Path, env: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """`link/..` names the folder above the link's target, as the system
    and git read it, so the split runs in that folder's work tree."""
    _seed_pair(repo)
    other = _other_repo(env)
    (repo / "b" / "link").symlink_to(other / "sub")
    code, out, err = _run(
        repo / "b" / "link" / "..", "--apply", "--json", capsys=capsys,
    )
    assert code == 0, err
    assert json.loads(out)["work_tree"] == str(other)
    assert (other / "o" / ".aelfrice" / "store" / "memory.db").exists()
    assert not _target_path(repo).parent.exists()


@pytest.mark.skipif(
    sys.platform == "win32", reason="symlinks need extra rights on Windows",
)
def test_a_relative_path_through_a_symlink_and_up_without_getcwd(
    repo: Path, env: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the system can't give the current folder's path, the path
    read from a handle to it still resolves its symlinks, so `link/..`
    names the folder above the link's target there too."""
    _seed_pair(repo)
    other = _other_repo(env)
    (repo / "b" / "link").symlink_to(other / "sub")
    monkeypatch.chdir(repo / "b")

    def no_getcwd() -> str:
        raise PermissionError(13, "Permission denied")

    with monkeypatch.context() as patch:
        patch.setattr(store_split, "_current_folder", lambda: str(repo / "b"))
        patch.setattr(os, "getcwd", no_getcwd)
        code, out, err = _run(
            Path("link") / "..", "--apply", "--json", capsys=capsys,
        )
    assert code == 0, err
    assert json.loads(out)["work_tree"] == str(other)
    assert (other / "o" / ".aelfrice" / "store" / "memory.db").exists()
    assert not _target_path(repo).parent.exists()


def test_apply_refuses_while_aelfrice_db_is_set(
    repo: Path, env: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_pair(repo)
    monkeypatch.setenv("AELFRICE_DB", str(env / "pinned" / "memory.db"))
    code, out, err = _run(repo, "--apply", capsys=capsys)
    assert code == 2
    assert out == ""
    assert len(err.strip().splitlines()) == 1
    assert "AELFRICE_DB" in err
    assert not _target_path(repo).parent.exists()
    assert not (env / "pinned").exists()


def test_a_folder_outside_a_work_tree_is_refused(
    env: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    plain = env / "plain"
    plain.mkdir()
    code, _, err = _run(plain, capsys=capsys)
    assert code == 2
    assert "not in a git work tree" in err


def test_a_path_that_is_a_file_or_missing_is_refused_as_such(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The refusal names the real problem, not a missing work tree."""
    marker = repo / "a" / CONFIG_FILENAME
    code, _, err = _run(marker, capsys=capsys)
    assert code == 2
    assert f"{marker} is not a folder" in err
    missing = repo / "a" / "missing"
    code, _, err = _run(missing, capsys=capsys)
    assert code == 2
    assert f"no folder at {missing}" in err


# --- the source store ------------------------------------------------------


def test_a_missing_repository_store_is_refused(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    code, out, err = _run(repo, capsys=capsys)
    assert code == 2
    assert out == ""
    assert f"no repository store at {_source_path(repo)}" in err


def test_an_unreadable_repository_store_exits_1(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    source = _source_path(repo)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"not a database " * 64)
    code, out, err = _run(repo, capsys=capsys)
    assert code == 1
    assert out == ""
    assert f"aelf store split: cannot read {source}" in err


@pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="needs POSIX permissions that the user can't bypass",
)
def test_a_repository_store_in_a_folder_it_cannot_search_exits_1(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The store may be there, so the command reports it as unreadable
    in one line rather than as missing or with a traceback."""
    _seed_pair(repo)
    source = _source_path(repo)
    source.parent.chmod(0)
    try:
        code, out, err = _run(repo, "--apply", capsys=capsys)
    finally:
        source.parent.chmod(0o755)
    assert code == 1
    assert out == ""
    assert err.startswith(f"aelf store split: cannot read {source}: "), err
    assert len(err.strip().splitlines()) == 1
    assert not _target_path(repo).parent.exists()


def test_the_source_store_is_only_ever_opened_read_only(
    repo: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_graph(repo)
    source = _source_path(repo)
    before = _file_hash(source)
    opened: list[str] = []
    real_connect: Callable[..., sqlite3.Connection] = sqlite3.connect

    def spy(database: str, *args: object, **kwargs: object) -> sqlite3.Connection:
        opened.append(str(database))
        return real_connect(database, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sqlite3, "connect", spy)
    report = _report(repo, "--apply", capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 2
    names = (str(source), source.as_uri())
    on_source = [db for db in opened if any(n in db for n in names)]
    assert on_source, opened
    assert all("mode=ro" in db for db in on_source), on_source
    assert _file_hash(source) == before


def test_the_source_is_read_in_one_snapshot(
    repo: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write that lands after the beliefs are read, such as a hook's
    feedback event, reaches neither the copied posterior nor the copied
    feedback rows, so the two stay consistent."""
    a1, _ = _seed_pair(repo)
    source = _source_path(repo)
    src = MemoryStore(str(source), read_only=True)
    try:
        belief = src.get_belief(a1)
    finally:
        src.close()
    assert belief is not None
    real_row_belief = store_split._row_belief  # pyright: ignore[reportPrivateUsage]
    wrote: list[bool] = []

    def row_belief_then_write(r: tuple[object, ...]) -> Belief:
        got = real_row_belief(r)
        if not wrote:
            wrote.append(True)
            writer = sqlite3.connect(source, timeout=5)
            try:
                writer.execute(
                    "UPDATE beliefs SET alpha = alpha + 1 WHERE id = ?", (a1,),
                )
                writer.execute(
                    "INSERT INTO feedback_history "
                    "(belief_id, valence, source, created_at) "
                    "VALUES (?, 1.0, 'hook', ?)",
                    (a1, _TS),
                )
                writer.commit()
            finally:
                writer.close()
        return got

    monkeypatch.setattr(store_split, "_row_belief", row_belief_then_write)
    target = _only_target(_report(repo, "--apply", capsys=capsys))
    assert wrote
    assert target["copied"]["beliefs"] == 1
    assert target["copied"]["feedback"] == 0
    got = _rows(_target_path(repo))
    assert got["feedback_history"] == []
    copied = MemoryStore(str(_target_path(repo)), read_only=True)
    try:
        copy = copied.get_belief(a1)
    finally:
        copied.close()
    assert copy is not None
    assert copy.alpha == belief.alpha


def test_a_dry_run_with_aelfrice_db_set_finds_no_marker_store(
    repo: Path, env: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The variable hides every marker, so no folder selects a marker
    store, and the run says so instead of routing beliefs to the pin."""
    _seed_pair(repo)
    monkeypatch.setenv("AELFRICE_DB", str(env / "pinned" / "memory.db"))
    code, out, err = _run(repo, "--json", capsys=capsys)
    assert code == 0, err
    assert json.loads(out)["targets"] == []
    assert "AELFRICE_DB" in err
    assert not (env / "pinned").exists()


def test_a_dry_run_with_aelfrice_db_set_reads_no_marker(
    repo: Path, env: Path, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the variable set no marker is read, so a broken one isn't
    reported, and every placed belief counts under `source_store`, as the
    docs say."""
    _seed_pair(repo)
    _marker(repo / "b", "")
    monkeypatch.setenv("AELFRICE_DB", str(env / "pinned" / "memory.db"))
    code, out, err = _run(repo, "--json", capsys=capsys)
    assert code == 0, err
    report = json.loads(out)
    assert report["targets"] == []
    assert report["not_copied"]["source_store"] == 2


def test_a_marker_that_names_the_repository_store_copies_nothing(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _marker(repo / "b", str(repo / ".git" / "aelfrice"))
    _seed_pair(repo)
    before = _file_hash(_source_path(repo))
    report = _report(repo, "--apply", capsys=capsys)
    assert _only_target(report)["copied"]["beliefs"] == 1
    not_copied = cast("dict[str, int]", report["not_copied"])
    assert not_copied["source_store"] == 1
    assert _file_hash(_source_path(repo)) == before


def test_a_marker_store_file_that_links_to_the_repository_store_copies_nothing(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A `memory.db` that is a symlink to the repository store is that
    store, so `--apply` never opens it for writing."""
    _seed_pair(repo)
    source = _source_path(repo)
    link = _target_path(repo)
    link.parent.mkdir(parents=True)
    link.symlink_to(source)
    assert db_paths.same_store_file(link, source)
    before = _file_hash(source)
    report = _report(repo, "--apply", capsys=capsys)
    assert report["targets"] == []
    not_copied = cast("dict[str, int]", report["not_copied"])
    assert not_copied["source_store"] == 2
    assert _file_hash(source) == before


def test_malformed_derived_ids_read_as_no_ids(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    a1, _ = _seed_pair(repo)
    src = _open(_source_path(repo))
    try:
        for raw in ("not json", '{"a": "b"}'):
            src.record_ingest(
                source_kind=INGEST_SOURCE_FILESYSTEM,
                source_path="doc:a/docs/q.md:p0", raw_text=raw,
                session_id="s1", ts=_TS, derived_belief_ids=[a1],
            )
    finally:
        src.close()
    conn = sqlite3.connect(_source_path(repo))
    try:
        conn.execute(
            "UPDATE ingest_log SET derived_belief_ids = raw_text "
            "WHERE raw_text IN ('not json', '{\"a\": \"b\"}')",
        )
        conn.commit()
    finally:
        conn.close()
    target = _only_target(_report(repo, capsys=capsys))
    assert target["copied"]["beliefs"] == 1
