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
import sqlite3
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from aelfrice import cli, db_paths, store_split
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
        "several_stores": 0, "source_store": 0,
    }


def test_the_root_comes_from_the_onboard_session(
    repo: Path, env: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """One absolute root for every ingest row, or the belief is skipped."""
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
        # No onboard session, and a relative root.
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
