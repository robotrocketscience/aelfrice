"""The commit hook runs, exits 0, and writes nothing.

The commit hook stays installed, but it no longer stores relation phrases
from commit messages. These tests drive its real entry points and read
the store file back:

1. A commit whose message holds relation phrases changes no row in the
   store and no file beside it, and the hook exits 0.
2. The read bank that `rebuild_log` uses keeps its patterns and types.
3. The query that `rebuild_log` records for fixed prompts is unchanged.
4. Controls: the same checks report a write when there is one, and the
   installed hook process runs to completion for the commit.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from aelfrice import hook_commit_ingest as hk
from aelfrice.db_paths import db_path
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_SUPERSEDES,
    INGEST_SOURCE_TRANSCRIPT,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.rebuild_log import record_user_prompt_submit_log
from aelfrice.setup import install_commit_ingest_git_hook
from aelfrice.store import MemoryStore
from aelfrice.triple_extractor import (
    _PATTERNS,  # pyright: ignore[reportPrivateUsage]
    extract_triples,
)

_CONSOLE_SCRIPT = Path(sys.executable).parent / "aelf-commit-ingest"

#: One message per relation phrasing. Each one produced rows before the
#: hook stopped writing.
_MESSAGES: dict[str, str] = {
    "implements": "the retry worker implements the queue contract",
    "is_an_implementation_of": "the kiln driver is an implementation of the heat spec",
    "fulfills": "the label printer fulfills the packing promise",
    "is_tested_by": "the index writer is tested by the orchard runner",
    "is_a_test_for": "the smoke suite is a test for the upgrade path",
    "is_covered_by": "the harbor import is covered by the tide suite",
    "is_derived_from": "the ferry timetable is derived from the prior memo",
    "is_based_on": "the bakery menu is based on the winter survey",
    "cites": "the release note cites the audit memo",
    "mentions": "the planner note mentions a missing path",
    "contradicts": "the cache note contradicts the storage memo",
    "disagrees_with": "the lock budget disagrees with the injection cap",
    "supersedes": "the batch importer supersedes the nightly export",
    "is_supported_by": "the new index is supported by faster queries",
    "relates_to": "our session table relates to the audit log",
    "comes_after": "the second rollout comes after the first rollout",
    # Subject and body, with the single-word verbs the ingest bank skipped.
    "subject_and_body": (
        "feat: the orchard worker implements the harvest contract\n"
        "\n"
        "the ripeness suite tests the harvest contract\n"
        "the ripeness suite covers the frost path\n"
        "the harvest contract is derived from the winter survey\n"
        "the harvest note cites the frost memo\n"
        "the frost memo contradicts the summer memo\n"
        "the orchard worker replaces the hand picker\n"
        "the picking plan extends the harvest plan\n"
        "the frost path follows the harvest path\n"
        "the winter survey supports the frost memo\n"
    ),
}

#: A message with no relation phrase.
_PLAIN_MESSAGE = "tidy the workshop shelf"


# --- the store, read from outside aelfrice -------------------------------------


def _rows(db: Path) -> dict[str, list[str]]:
    """Every row of every table in the store file, by table name."""
    conn = sqlite3.connect(str(db), timeout=10)
    try:
        names = [
            str(r[0])
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        ]
        out: dict[str, list[str]] = {}
        for name in names:
            rows = cast(
                "list[tuple[object, ...]]",
                conn.execute(f'SELECT * FROM "{name}"').fetchall(),
            )
            out[name] = sorted(repr(r) for r in rows)
        return out
    finally:
        conn.close()


def _counts(db: Path) -> dict[str, int]:
    rows = _rows(db)
    return {t: len(rows[t]) for t in ("beliefs", "edges", "ingest_log")}


def _changed_tables(
    before: dict[str, list[str]], after: dict[str, list[str]],
) -> list[str]:
    return sorted(t for t in {*before, *after} if before.get(t) != after.get(t))


def _files(folder: Path) -> list[str]:
    """Files beside the store, without SQLite's own sidecar files."""
    return sorted(
        p.name for p in folder.iterdir() if not p.name.endswith(("-wal", "-shm"))
    )


def _belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
    )


# --- fixtures ------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The store path the hook resolves. The file doesn't exist yet."""
    home = tmp_path / "home"
    home.mkdir()
    folder = tmp_path / "store"
    folder.mkdir()
    p = folder / "memory.db"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AELFRICE_DB", str(p))
    monkeypatch.setenv("AELFRICE_USER_DB", str(tmp_path / "user" / "memory.db"))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    return p


@pytest.fixture
def seeded(db: Path) -> Path:
    """A store at that path that already holds two beliefs and an edge."""
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_belief("seed-a", "the greenhouse opens at nine"))
        store.insert_belief(_belief("seed-b", "the greenhouse opens at eight"))
        store.insert_edge(Edge("seed-a", "seed-b", EDGE_SUPERSEDES, 1.0))
    finally:
        store.close()
    return db


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        encoding="utf-8", check=False, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {args!r} failed: {r.stderr}")
    return r.stdout


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "T")
    _git(r, "config", "commit.gpgsign", "false")
    return r


def _commit(repo: Path, message: str, name: str = "x") -> str:
    """Commit `message` and return the commit's hash."""
    (repo / name).write_text(message, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD").strip()


# --- the hook's entry points ---------------------------------------------------


def _run_git_hook(
    repo: Path, rev: str, monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, str]:
    """The post-commit form: `aelf-commit-ingest --git-hook REV`."""
    err = io.StringIO()
    with monkeypatch.context() as mp:
        mp.chdir(repo)
        rc = hk.main(stdin=io.StringIO(""), stderr=err, argv=["--git-hook", rev])
    return rc, err.getvalue()


def _run_tool_hook(
    repo: Path, rev: str, monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, str]:
    """The PostToolUse form: a payload on stdin."""
    subject = _git(repo, "log", "-1", "--format=%s", rev).strip()
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "git commit -F message.txt"},
        "tool_response": {
            "stdout": f"[main {rev[:8]}] {subject}\n 1 file changed",
            "stderr": "",
            "isError": False,
            "interrupted": False,
        },
        "cwd": str(repo),
    }
    err = io.StringIO()
    with monkeypatch.context() as mp:
        mp.chdir(repo)
        rc = hk.main(stdin=io.StringIO(json.dumps(payload)), stderr=err, argv=[])
    return rc, err.getvalue()


_ENTRY_POINTS = {"git_hook": _run_git_hook, "tool_hook": _run_tool_hook}


@dataclass(frozen=True)
class _HookRun:
    """What the installed hook process reported when it finished."""

    exit_code: int
    store: str
    args: list[str]
    stderr: str


def _install_recording_hook(repo: Path, folder: Path) -> Path:
    """Install the real post-commit block around the real console script.

    The block runs its command in the background and discards its output,
    so the command is a wrapper that runs `aelf-commit-ingest` with the
    same arguments and then records how it ended. The record appears only
    when the process has finished. A missing console script is recorded
    as a nonzero exit.
    """
    folder.mkdir()
    record = folder / "record"
    script = folder / "wrapper"
    q = shlex.quote
    script.write_text(
        "#!/bin/sh\n"
        f'{q(str(_CONSOLE_SCRIPT))} "$@" 2>{q(str(folder / "stderr"))}\n'
        "rc=$?\n"
        f'printf "%s\\n" "$rc" "$AELFRICE_DB" "$@" >{q(str(record) + ".tmp")}\n'
        f'mv {q(str(record) + ".tmp")} {q(str(record))}\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    assert install_commit_ingest_git_hook(repo, command=str(script)).status == "installed"
    return record


def _wait_for_hook(record: Path, timeout: float = 40.0) -> _HookRun:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if record.exists():
            lines = record.read_text(encoding="utf-8").splitlines()
            return _HookRun(
                exit_code=int(lines[0]),
                store=lines[1],
                args=lines[2:],
                stderr=(record.parent / "stderr").read_text(encoding="utf-8"),
            )
        time.sleep(0.1)
    raise AssertionError(f"the installed hook didn't finish within {timeout} s")


# --- 1. a commit writes nothing ------------------------------------------------


@pytest.mark.timeout(60)
@pytest.mark.parametrize("entry", sorted(_ENTRY_POINTS))
@pytest.mark.parametrize("message", list(_MESSAGES.values()), ids=list(_MESSAGES))
def test_a_relation_phrase_writes_nothing(
    message: str, entry: str, repo: Path, seeded: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rev = _commit(repo, message)
    before, files = _rows(seeded), _files(seeded.parent)
    counts = _counts(seeded)

    rc, err = _ENTRY_POINTS[entry](repo, rev, monkeypatch)

    assert (rc, err) == (0, "")
    assert _counts(seeded) == counts
    assert _changed_tables(before, _rows(seeded)) == []
    # The git hook logs a swallowed exception to a file beside the store.
    assert _files(seeded.parent) == files


@pytest.mark.timeout(60)
@pytest.mark.parametrize("entry", sorted(_ENTRY_POINTS))
def test_the_same_message_twice_writes_nothing(
    entry: str, repo: Path, seeded: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated message used to corroborate the first one's beliefs."""
    message = _MESSAGES["subject_and_body"]
    before, files = _rows(seeded), _files(seeded.parent)
    for name in ("a", "b"):
        rev = _commit(repo, message, name)
        assert _ENTRY_POINTS[entry](repo, rev, monkeypatch) == (0, "")
    assert _changed_tables(before, _rows(seeded)) == []
    assert _files(seeded.parent) == files


@pytest.mark.timeout(60)
@pytest.mark.parametrize("entry", sorted(_ENTRY_POINTS))
def test_no_store_is_created(
    entry: str, repo: Path, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rev = _commit(repo, _MESSAGES["subject_and_body"])
    assert _ENTRY_POINTS[entry](repo, rev, monkeypatch) == (0, "")
    assert _files(db.parent) == []


@pytest.mark.timeout(90)
def test_the_installed_hook_writes_nothing_on_a_real_commit(
    repo: Path, seeded: Path, tmp_path: Path,
) -> None:
    record = _install_recording_hook(repo, tmp_path / "hook")
    before, files = _rows(seeded), _files(seeded.parent)
    counts = _counts(seeded)

    rev = _commit(repo, _MESSAGES["subject_and_body"])

    run = _wait_for_hook(record)
    assert run == _HookRun(0, str(seeded), ["--git-hook", rev], "")
    assert _counts(seeded) == counts
    assert _changed_tables(before, _rows(seeded)) == []
    assert _files(seeded.parent) == files


# --- 2. the read bank ----------------------------------------------------------

#: Type strings of the read bank, in order. They are literals here because
#: the bank keeps each string when its edge type's constant is deleted.
_READ_BANK_TYPES = [
    "SUPPORTS", "SUPPORTS",
    "CITES", "CITES",
    "CONTRADICTS", "CONTRADICTS",
    "SUPERSEDES", "SUPERSEDES",
    "RELATES_TO", "RELATES_TO",
    "DERIVED_FROM", "DERIVED_FROM", "DERIVED_FROM",
    "IMPLEMENTS", "IMPLEMENTS", "IMPLEMENTS", "IMPLEMENTS",
    "TEMPORAL_NEXT", "TEMPORAL_NEXT", "TEMPORAL_NEXT", "TEMPORAL_NEXT",
    "TESTS", "TESTS", "TESTS", "TESTS",
]

_READ_BANK_DIGEST = "299ae9b1cce16d17792fae781da3ca65338a594fb4708f2555657d20a60f68b5"


def test_the_read_bank_types_are_unchanged() -> None:
    assert [p.edge_type for p in _PATTERNS] == _READ_BANK_TYPES


def test_the_read_bank_digest_is_unchanged() -> None:
    bank = [(p.regex.pattern, p.regex.flags, p.edge_type, p.swap) for p in _PATTERNS]
    raw = json.dumps(bank, ensure_ascii=True).encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == _READ_BANK_DIGEST


# --- 3. the query terms that rebuild_log records -------------------------------

#: Prompt and the query recorded for it. The last group holds prompts that
#: the read path and the commit hook's own bank extracted differently: a
#: match across a line break, an uppercase verb, single-word verbs, and a
#: phrase of function words.
_QUERY_TERMS: dict[str, tuple[str, str]] = {
    "supports": (
        "the winter survey supports the frost memo",
        "the winter survey supports the frost memo the winter survey the "
        "frost memo",
    ),
    "is_supported_by": (
        "the new index is supported by faster queries",
        "the new index is supported by faster queries the new index",
    ),
    "cites": (
        "the release note cites the audit memo",
        "the release note cites the audit memo the release note the "
        "audit memo",
    ),
    "contradicts": (
        "the cache note contradicts the storage memo",
        "the cache note contradicts the storage memo the cache note the "
        "storage memo",
    ),
    "disagrees_with": (
        "the lock budget disagrees with the injection cap",
        "the lock budget disagrees with the injection cap the lock "
        "budget the injection cap",
    ),
    "is_related_to": (
        "the tide chart is related to the harbor log",
        "the tide chart is related to the harbor log the tide chart",
    ),
    "is_derived_from": (
        "the ferry timetable is derived from the prior memo",
        "the ferry timetable is derived from the prior memo the ferry "
        "timetable",
    ),
    "implements": (
        "a single worker implements the queue contract",
        "a single worker implements the queue contract a single worker "
        "the queue contract",
    ),
    "is_a_test_for": (
        "the smoke suite is a test for the upgrade path",
        "the smoke suite is a test for the upgrade path the smoke suite "
        "the upgrade path",
    ),
    "comes_after": (
        "the second rollout comes after the first rollout",
        "the second rollout comes after the first rollout the second "
        "rollout the first rollout",
    ),
    "entities": (
        "open src/orchard/harvest.py and check HarvestPlan before the "
        "frost_path runs",
        "src/orchard/harvest.py HarvestPlan frost_path",
    ),
    "no_relation": (
        "please water plants before lunch tomorrow",
        "please water plants before lunch tomorrow",
    ),
    "short_words": (
        "and so it was",
        "and so it was",
    ),
    "two_sentences": (
        "the batch importer supersedes the nightly export. the planner "
        "mentions a missing path on startup",
        "the batch importer supersedes the nightly export the planner "
        "mentions a missing path on startup the batch importer the "
        "nightly export the planner a missing path on startup",
    ),
    "across_lines": (
        "the greenhouse\nimplements\nthe heating spec",
        "the greenhouse\nimplements\nthe heating spec the greenhouse the "
        "heating spec",
    ),
    "two_lines": (
        "the batch importer supersedes the nightly export\nthe planner "
        "mentions a missing path",
        "the batch importer supersedes the nightly export\nthe planner "
        "mentions a missing path the batch importer the nightly export "
        "the planner mentions the nightly export the planner a missing "
        "path",
    ),
    "uppercase_verb": (
        "describe the SUPERSEDES writer for new readers",
        "describe the SUPERSEDES writer for new readers describe the "
        "writer for new readers",
    ),
    "single_word_verbs": (
        "the ripeness suite tests the frost path and the hand picker "
        "replaces the old cart",
        "the ripeness suite tests the frost path and the hand picker "
        "replaces the old cart the ripeness suite the frost path and the "
        "hand the old cart",
    ),
    "covers_and_follows": (
        "the tide suite covers the harbor import while the frost path "
        "follows the harvest path",
        "the tide suite covers the harbor import while the frost path "
        "follows the harvest path the tide suite the harbor import while "
        "the frost the harvest path",
    ),
    "extends": (
        "the picking plan extends the harvest plan",
        "the picking plan extends the harvest plan the picking plan the "
        "harvest plan",
    ),
    "function_word_subject": (
        "it's supersedes the old parser",
        "it s supersedes the old parser s the old parser",
    ),
    "function_word_object": (
        "the new parser supersedes it",
        "the new parser supersedes it the new parser it",
    ),
}


@pytest.mark.parametrize(
    ("prompt", "expected"), list(_QUERY_TERMS.values()), ids=list(_QUERY_TERMS),
)
def test_rebuild_log_query_terms_are_unchanged(
    prompt: str, expected: str, db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AELFRICE_REBUILD_LOG", raising=False)
    log = db.parent / "rebuild_logs" / "s.jsonl"
    hits = [_belief("hit", "the greenhouse opens at nine")]
    record_user_prompt_submit_log(
        prompt=prompt, session_id="s", hits_pre_dedup=hits,
        hits_post_dedup=hits, log_path=log, enabled=True, stderr=io.StringIO(),
    )
    [line] = log.read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["input"]["extracted_query"] == expected


# --- 4. controls ---------------------------------------------------------------


def test_the_checks_report_a_direct_write(repo: Path, seeded: Path) -> None:
    """The store the hook resolves is the file these tests read, and a write
    to it shows up in the counts and in the row comparison."""
    assert db_path(start=repo) == seeded
    before = _rows(seeded)
    assert _counts(seeded) == {"beliefs": 2, "edges": 1, "ingest_log": 0}

    store = MemoryStore(str(db_path(start=repo)))
    try:
        store.insert_belief(_belief("new", "the bakery closes on Monday"))
        store.insert_edge(Edge("new", "seed-a", EDGE_SUPERSEDES, 1.0))
        store.record_ingest(
            source_kind=INGEST_SOURCE_TRANSCRIPT,
            raw_text="the bakery closes on Monday",
            derived_belief_ids=["new"],
        )
        public = (store.count_beliefs(), store.count_edges(), store.count_ingest_log())
    finally:
        store.close()

    assert public == (3, 2, 1)
    assert _counts(seeded) == {"beliefs": 3, "edges": 2, "ingest_log": 1}
    changed = _changed_tables(before, _rows(seeded))
    assert {"beliefs", "edges", "ingest_log"} <= set(changed)


def test_opening_the_store_changes_no_row(seeded: Path) -> None:
    """The row comparison doesn't fail on an open that writes nothing."""
    before, files = _rows(seeded), _files(seeded.parent)
    MemoryStore(str(seeded)).close()
    assert _changed_tables(before, _rows(seeded)) == []
    assert _files(seeded.parent) == files


def test_the_messages_hold_relation_phrases() -> None:
    """The combined message holds implement, test, derive, cite, and
    contradict phrasings, as the extractor that stays reads them."""
    relations = {t.relation for t in extract_triples(_MESSAGES["subject_and_body"])}
    assert {"IMPLEMENTS", "TESTS", "DERIVED_FROM", "CITES", "CONTRADICTS"} <= relations
    assert extract_triples(_PLAIN_MESSAGE) == []


@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    "message", [_PLAIN_MESSAGE, _MESSAGES["implements"]], ids=["plain", "relation"],
)
def test_the_installed_hook_runs_for_the_commit(
    message: str, repo: Path, seeded: Path, tmp_path: Path,
) -> None:
    """Setup's block still starts the console script for the new commit, in
    the environment that names this store, and the process exits 0."""
    assert os.access(_CONSOLE_SCRIPT, os.X_OK)
    record = _install_recording_hook(repo, tmp_path / "hook")
    rev = _commit(repo, message)
    run = _wait_for_hook(record)
    assert run == _HookRun(0, str(seeded), ["--git-hook", rev], "")


@pytest.mark.timeout(60)
@pytest.mark.parametrize("entry", sorted(_ENTRY_POINTS))
def test_a_plain_message_writes_nothing(
    entry: str, repo: Path, seeded: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The commit, the fixtures, and the hook run change nothing on their
    own, so a difference in the first group comes from the message."""
    rev = _commit(repo, _PLAIN_MESSAGE)
    before, files = _rows(seeded), _files(seeded.parent)
    assert _ENTRY_POINTS[entry](repo, rev, monkeypatch) == (0, "")
    assert _changed_tables(before, _rows(seeded)) == []
    assert _files(seeded.parent) == files
