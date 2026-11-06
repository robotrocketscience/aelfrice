"""#1874: the commit hook's patterns take a lowercase verb on one line.

`re.IGNORECASE` let an uppercase edge-type name used as a noun, as in "the
SUPERSEDES writer", read as the verb. `\\s` in the noun-phrase pattern and
in the verb's whitespace let a match join words from two lines. Both changes
apply to the ingest bank only: `rebuild_log` queries with the read bank, and
`entity_extractor` shares the noun-phrase pattern, so neither may move.

Every message here is invented.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from aelfrice import triple_extractor as te
from aelfrice.derivation import DerivationInput, derive
from aelfrice.entity_extractor import extract_entities
from aelfrice.hook_commit_ingest import _ingest_message, git_hook_ingest
from aelfrice.models import (
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TESTS,
    INGEST_SOURCE_GIT,
)
from aelfrice.np_pattern import NOUN_PHRASE_PATTERN
from aelfrice.rebuild_log import RecentTurn, _query_for_recent_turns
from aelfrice.replay import replay_full_equality
from aelfrice.setup import (
    GIT_HOOK_BEGIN,
    GIT_HOOK_END,
    git_hook_command_line,
    install_commit_ingest_git_hook,
)
from aelfrice.store import MemoryStore
from aelfrice.triple_extractor import (
    _INGEST_PATTERNS,
    _PATTERNS,
    Triple,
    _build_pattern,
    extract_triples,
    ingest_triples,
)
from benchmarks import triple_precision_1376
from benchmarks.triple_precision_1376 import _fires_by_template

_SUBJECT_AND_BODY = (
    "fix: handle empty input\n\n"
    "This fix supersedes the old fetcher because it handles retries badly"
)

# Every character `str.splitlines` splits on. `\s` matches each of them, so
# each one is a way for a match to cross a line.
_LINE_BREAKS = (
    "\n", "\r\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85",
    " ", " ",
)

_SINGLE_LINE_LOWERCASE = (
    "feat: the new fetcher supersedes the old fetcher",
    "The big old red fast car supersedes a slow blue van",
    "THE NEW FETCHER supersedes THE OLD FETCHER",
    "fix: crash when the config mentions a missing path on startup",
    "the spec is derived from the prior memo",
    "the new index is supported by faster queries",
    "this parser is tested by the golden corpus and is covered by nightly runs",
    "the planner contradicts the memo, and the memo cites the paper",
    "Our writer implements the spec. Their reader is based on the writer.",
)

# Rich in what the ingest bank now refuses, so a change that reaches the read
# path shows up in the pinned hash below.
_READ_PATH_CORPUS = (
    *_SINGLE_LINE_LOWERCASE,
    _SUBJECT_AND_BODY,
    "docs: describe the SUPERSEDES writer for new readers",
    "docs: add a CITES reader in the graph view",
    "The New Fetcher Supersedes The Old Fetcher",
    "refactor: tidy the fetch\n\nThe new fetcher\nsupersedes\nthe old fetcher",
    "two tests that encoded the old behaviour are removed",
    "this covers the empty case and follows the spec\r\nand extends the parser",
)

# Computed on `github/main` at fecbe3e1, before this change.
_READ_BANK_SIGNATURE = "299ae9b1cce16d17792fae781da3ca65338a594fb4708f2555657d20a60f68b5"
_READ_OUTPUT_DIGEST = "5297f4cca211f6c83901845526cb278bc22cfc40cb728c351a95ac440f3c05a3"
_READ_QUERY_DIGEST = "0c38f30b7d6f96fbbc4ee67aa3f31a898233dda1eb8cea389fe9ecae7104f331"
_NP_PATTERN_DIGEST = "bf82703a2320d4c2cf5ab05b7b38168ea32894bf04b0a29606e962a9f68997eb"
_ENTITY_OUTPUT_DIGEST = "7263d5d003e349b710ff0d8ab1e0ef4bb0fcaa17a331908aa67783d78edf5aa5"


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _read_bank_signature() -> str:
    return _digest(
        [(p.regex.pattern, p.regex.flags, p.edge_type, p.swap) for p in _PATTERNS]
    )


def _read_output_digest() -> str:
    return _digest([
        [(t.subject, t.relation, t.object, t.anchor_text) for t in extract_triples(text)]
        for text in _READ_PATH_CORPUS
    ])


def _read_query_digest() -> str:
    return _digest([
        _query_for_recent_turns([RecentTurn(role="user", text=text)])
        for text in _READ_PATH_CORPUS
    ])


def _entity_output_digest() -> str:
    return _digest([
        [(e.kind, e.raw, e.span_start, e.span_end) for e in extract_entities(text)]
        for text in _READ_PATH_CORPUS
    ])


def _ingest(text: str) -> list[Triple]:
    return extract_triples(text, constrain_collision_verbs=True)


def _pairs(text: str) -> list[tuple[str, str, str]]:
    return [(t.subject, t.relation, t.object) for t in _ingest(text)]


# --- The verb's case ---------------------------------------------------------


@pytest.mark.parametrize("pat", _INGEST_PATTERNS, ids=lambda p: p.template)
def test_every_ingest_template_matches_only_in_lowercase(
    pat: te._RelationPattern,
) -> None:
    def sentence(verb: str) -> str:
        return f"the new fetcher {verb} the old fetcher"

    subject, obj = "the new fetcher", "the old fetcher"
    if pat.swap:
        subject, obj = obj, subject
    assert _pairs(sentence(pat.template)) == [(subject, pat.edge_type, obj)]
    assert _pairs(sentence(pat.template.upper())) == []
    assert _pairs(sentence(pat.template.title())) == []


def test_a_passive_form_still_swaps_its_phrases() -> None:
    assert _pairs("the new index is supported by faster queries") == [
        ("faster queries", EDGE_SUPPORTS, "the new index"),
    ]
    assert _pairs("the new parser is tested by the golden corpus") == [
        ("the golden corpus", EDGE_TESTS, "the new parser"),
    ]


def test_one_uppercase_token_in_a_multiword_verb_stops_the_match() -> None:
    assert _pairs("the spec is Based on the prior memo") == []


def test_an_edge_type_name_used_as_a_noun_is_not_a_verb() -> None:
    assert _pairs("docs: describe the SUPERSEDES writer for new readers") == []
    assert _pairs("docs: add a CITES reader in the graph view") == []


def test_the_phrases_on_each_side_keep_ignoring_case() -> None:
    """A global flag removal keeps `The` out of the determiner slot, so a
    five-token phrase loses it."""
    assert _pairs("The big old red fast car supersedes THE slow blue van") == [
        ("The big old red fast car", EDGE_SUPERSEDES, "THE slow blue van"),
    ]


@pytest.mark.parametrize(
    "text",
    (
        *_SINGLE_LINE_LOWERCASE,
        "This supersedes the old parser",
        "the new parser supersedes this",
        "the new parser supersedes the",
    ),
)
def test_a_lowercase_verb_on_one_line_stores_what_the_old_bank_stored(
    text: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    new = _ingest(text)
    old_bank = tuple(
        _build_pattern(p.template, p.edge_type, swap=p.swap)
        for p in _INGEST_PATTERNS
    )
    monkeypatch.setattr(te, "_INGEST_PATTERNS", old_bank)
    assert new == _ingest(text)
    assert new


# --- Line crossing -----------------------------------------------------------


@pytest.mark.parametrize("brk", _LINE_BREAKS, ids=repr)
@pytest.mark.parametrize(
    "text",
    [
        "the new fetcher{brk}supersedes the old fetcher",
        "the new fetcher supersedes{brk}the old fetcher",
        "the spec is{brk}derived from the prior memo",
    ],
)
def test_a_line_break_touching_the_verb_stops_the_match(text: str, brk: str) -> None:
    assert _pairs(text.format(brk=brk)) == []


@pytest.mark.parametrize("brk", _LINE_BREAKS, ids=repr)
def test_a_phrase_stops_at_a_line_break(brk: str) -> None:
    assert _pairs(f"fix the bug{brk}the new fetcher supersedes the old{brk}fetcher") == [
        ("the new fetcher", EDGE_SUPERSEDES, "the old"),
    ]


def test_a_subject_line_never_joins_the_body() -> None:
    assert _pairs(_SUBJECT_AND_BODY) == [
        ("This fix", EDGE_SUPERSEDES, "the old fetcher because it handles"),
    ]


def test_a_relation_on_each_of_two_lines_stores_both() -> None:
    """Across lines, the first match's object used to swallow the second verb."""
    assert _pairs("the writer supersedes the old\nthe reader supersedes the new") == [
        ("the writer", EDGE_SUPERSEDES, "the old"),
        ("the reader", EDGE_SUPERSEDES, "the new"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "the new parser supersedes the\nold parser",
        "the new parser supersedes the  \nold parser",
        "the new parser supersedes A\nold parser",
        "the parser is tested by the\ngolden corpus",
    ],
)
def test_a_determiner_a_line_break_cut_off_stores_nothing(text: str) -> None:
    assert _pairs(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "the new parser supersedes the",
        "the new parser supersedes the\n",
        "the new parser supersedes this, for now\nmore text",
    ],
)
def test_a_determiner_no_line_break_cut_off_still_stores(text: str) -> None:
    [(subject, relation, obj)] = _pairs(text)
    assert (subject, relation) == ("the new parser", EDGE_SUPERSEDES)
    assert obj in {"the", "this"}


def test_a_body_line_that_opens_with_a_determiner_keeps_its_subject() -> None:
    assert _pairs("fix: tidy the parser\n\nThis supersedes the old parser") == [
        ("This", EDGE_SUPERSEDES, "the old parser"),
    ]


def test_the_anchor_still_reads_past_the_matched_line() -> None:
    [triple] = _ingest("fix: retry\n\nthe writer supersedes the reader\n\nmore context")
    assert triple.subject == "the writer"
    assert triple.anchor_text == (
        "fix: retry\n\nthe writer supersedes the reader\n\nmore context"
    )


# --- The read path and the entity index --------------------------------------


def test_the_read_bank_is_byte_identical_to_base() -> None:
    assert _read_bank_signature() == _READ_BANK_SIGNATURE


def test_read_path_output_is_byte_identical_to_base() -> None:
    assert _read_output_digest() == _READ_OUTPUT_DIGEST


def test_the_rebuild_log_query_is_byte_identical_to_base() -> None:
    assert _read_query_digest() == _READ_QUERY_DIGEST


def test_the_read_path_still_ignores_case_and_crosses_lines() -> None:
    upper = extract_triples("docs: describe the SUPERSEDES writer for new readers")
    assert [t.relation for t in upper] == [EDGE_SUPERSEDES]
    [crossed] = extract_triples(_SUBJECT_AND_BODY)
    assert crossed.subject == "handle empty input This fix"


def test_the_entity_index_is_byte_identical_to_base() -> None:
    assert _digest(NOUN_PHRASE_PATTERN) == _NP_PATTERN_DIGEST
    assert _entity_output_digest() == _ENTITY_OUTPUT_DIGEST


# --- The hook ----------------------------------------------------------------


def _edges(db: Path) -> list[tuple[str, str, str]]:
    if not db.exists():
        return []
    store = MemoryStore(str(db))
    try:
        rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "SELECT s.content, e.type, d.content FROM edges e "
            "JOIN beliefs s ON s.id = e.src JOIN beliefs d ON d.id = e.dst "
            "ORDER BY s.content"
        ).fetchall()
    finally:
        store.close()
    return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def test_the_hook_opens_no_store_for_an_uppercase_noun(db: Path) -> None:
    _ingest_message(
        "docs: describe the SUPERSEDES writer for new readers",
        session_id="s", project_context="p",
    )
    assert not db.exists()


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        encoding="utf-8", check=True, timeout=30,
    )
    return r.stdout


@pytest.mark.timeout(60)
def test_real_commits_store_only_the_lowercase_one_line_relation(
    tmp_path: Path, db: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "commit.gpgsign", "false")
    for i, message in enumerate((
        "docs: describe the SUPERSEDES writer for new readers",
        "refactor: tidy the fetch\n\nThe new fetcher\nsupersedes\nthe old fetcher",
        _SUBJECT_AND_BODY,
        "feat: the new parser supersedes the old parser",
    )):
        (repo / f"f{i}").write_text(message, encoding="utf-8")
        _git(repo, "add", f"f{i}")
        _git(repo, "commit", "-q", "-m", message)
        git_hook_ingest(_git(repo, "rev-parse", "HEAD").strip(), str(repo))
    assert _edges(db) == [
        ("This fix", EDGE_SUPERSEDES, "the old fetcher because it handles"),
        ("the new parser", EDGE_SUPERSEDES, "the old parser"),
    ]


# --- Replay ------------------------------------------------------------------


def _log_rows(store: MemoryStore) -> list[str]:
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT raw_text FROM ingest_log WHERE source_kind = ? ORDER BY rowid",
        (INGEST_SOURCE_GIT,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def test_a_stored_row_replays_under_the_patterns_that_wrote_it() -> None:
    """A `git` log row holds a phrase and no message, so `derive` never reruns
    the patterns. A row the old bank wrote keeps its shape, with no drift."""
    old_subject = "handle empty input This fix"
    old_object = "the old fetcher because it handles"
    store = MemoryStore(":memory:")
    try:
        ingest_triples(store, [
            Triple(old_subject, EDGE_SUPERSEDES, old_object, _SUBJECT_AND_BODY),
        ])
        report = replay_full_equality(store)
        assert not report.has_drift
        assert report.matched == report.total_log_rows == 2
        assert _log_rows(store) == [old_subject, old_object]
        derived = derive(
            DerivationInput(raw_text=old_subject, source_kind=INGEST_SOURCE_GIT),
        )
        assert derived.belief is not None
        assert derived.belief.content == old_subject
        assert old_subject not in {t.subject for t in _ingest(_SUBJECT_AND_BODY)}
    finally:
        store.close()


def test_a_fresh_ingest_replays_to_the_same_beliefs() -> None:
    store = MemoryStore(":memory:")
    try:
        ingest_triples(store, _ingest(_SUBJECT_AND_BODY))
        report = replay_full_equality(store)
        assert not report.has_drift
        assert report.matched == report.total_log_rows == 2
        assert _log_rows(store) == [
            "This fix", "the old fetcher because it handles",
        ]
    finally:
        store.close()


# --- The post-commit block ---------------------------------------------------


def _hook(repo: Path) -> Path:
    return repo / ".git" / "hooks" / "post-commit"


@pytest.mark.timeout(60)
def test_the_block_comment_describes_what_the_hook_stores(tmp_path: Path) -> None:
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, timeout=30,
    )
    assert install_commit_ingest_git_hook(tmp_path, command="/x/ingest").status == "installed"
    comments = " ".join(
        ln.lstrip("# ") for ln in _hook(tmp_path).read_text().splitlines()
        if ln.startswith("# ") and "aelfrice commit-ingest" not in ln
    )
    assert "Records this commit's message" not in comments
    assert "stores the words on each side" in comments
    assert "Delete this block to stop it." in comments


@pytest.mark.timeout(60)
def test_setup_replaces_a_block_with_the_old_comment(tmp_path: Path) -> None:
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, timeout=30,
    )
    command = "/x/ingest"
    old_block = (
        f"{GIT_HOOK_BEGIN}\n"
        "# Records this commit's message in aelfrice memory, in the\n"
        "# background, so the commit doesn't wait. Skips commits that a\n"
        "# rebase replays. Delete this block to stop it.\n"
        + git_hook_command_line(command)
        + f"{GIT_HOOK_END}\n"
    )
    hook = _hook(tmp_path)
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(f"#!/bin/sh\n{old_block}echo mine\n", encoding="utf-8")
    hook.chmod(0o755)
    assert install_commit_ingest_git_hook(tmp_path, command=command).status == "installed"
    text = hook.read_text()
    assert text.count(GIT_HOOK_BEGIN) == 1
    assert "Records this commit's message" not in text
    assert text.endswith(f"{GIT_HOOK_END}\necho mine\n")
    assert install_commit_ingest_git_hook(tmp_path, command=command).status == "already_present"


# --- The benchmark's replica loop --------------------------------------------


_BENCHMARK_BODIES = (
    _SUBJECT_AND_BODY,
    "refactor: tidy the fetch\n\nThe new fetcher\nsupersedes\nthe old fetcher",
    "the writer supersedes the old\nthe reader supersedes the new",
    "the new parser supersedes the\nold parser",
    "docs: describe the SUPERSEDES writer for new readers",
)


def test_the_benchmark_replica_agrees_with_extract_triples() -> None:
    """`triple_precision_1376` exits when its replica loop and `extract_triples`
    disagree, so its ingest walk has to match per line too."""
    for body in _BENCHMARK_BODIES:
        ingest_counts, _ = _fires_by_template([body], _INGEST_PATTERNS, per_line=True)
        assert sum(ingest_counts.values()) == len(_ingest(body)), body
        read_counts, _ = _fires_by_template([body], _PATTERNS)
        assert sum(read_counts.values()) == len(extract_triples(body)), body


@pytest.mark.parametrize("body", _BENCHMARK_BODIES)
def test_the_benchmark_cross_check_passes_on_a_line_crossing_body(
    body: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        triple_precision_1376, "_commit_bodies",
        lambda limit, ref, root: [body],
    )
    assert triple_precision_1376.main(["--json"]) == 0
