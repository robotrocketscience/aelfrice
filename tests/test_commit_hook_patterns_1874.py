"""#1874: the read bank and the entity index stay byte-identical.

#1874 made the commit hook's own bank take a lowercase verb on one line and
skip a phrase of only function words. That bank went when the commit hook
stopped writing. `rebuild_log` still queries with the read bank, and
`entity_extractor` shares the noun-phrase pattern, so neither may move. A
`git` row already in a store still replays to its belief, and the installed
block's comment says what the hook does.

Every message here is invented.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from aelfrice.classification_core import INGEST_CLASSIFIER_VERSION, rule_set_hash
from aelfrice.derivation import DerivationInput, derive
from aelfrice.derivation_worker import run_worker
from aelfrice.entity_extractor import extract_entities
from aelfrice.models import (
    CORROBORATION_SOURCE_COMMIT_INGEST,
    EDGE_SUPERSEDES,
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
from aelfrice.triple_extractor import _PATTERNS, extract_triples

_SUBJECT_AND_BODY = (
    "fix: handle empty input\n\n"
    "This fix supersedes the old fetcher because it handles retries badly"
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

# Rich in what the commit hook's bank refused, so a change that reaches the
# read path shows up in the pinned hash below.
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

# Computed on `github/main` at fecbe3e1, before #1874.
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


def test_the_read_path_keeps_contraction_fragments() -> None:
    assert [
        (t.subject, t.object) for t in extract_triples("it's supersedes the old parser")
    ] == [("s", "the old parser")]


def test_the_read_path_keeps_function_word_phrases() -> None:
    assert [
        (t.subject, t.object) for t in extract_triples("This supersedes the old parser")
    ] == [("This", "the old parser")]
    assert [
        (t.subject, t.object) for t in extract_triples("the new parser supersedes it")
    ] == [("the new parser", "it")]


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


# --- Replay ------------------------------------------------------------------


def _log_rows(store: MemoryStore) -> list[str]:
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT raw_text FROM ingest_log WHERE source_kind = ? ORDER BY rowid",
        (INGEST_SOURCE_GIT,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def _store_git_rows(store: MemoryStore, phrases: list[str]) -> None:
    """The rows the commit hook wrote, one per phrase, then the worker."""
    for phrase in phrases:
        store.record_ingest(
            source_kind=INGEST_SOURCE_GIT,
            raw_text=phrase,
            session_id="commit-session",
            ts="2026-01-01T00:00:00+00:00",
            raw_meta={"call_site": CORROBORATION_SOURCE_COMMIT_INGEST},
            classifier_version=INGEST_CLASSIFIER_VERSION,
            rule_set_hash=rule_set_hash(),
        )
    run_worker(store)


def test_a_stored_row_replays_under_the_patterns_that_wrote_it() -> None:
    """A `git` log row holds a phrase and no message, so `derive` never reruns
    the patterns. A row an old bank wrote keeps its shape, with no drift."""
    old_subject = "handle empty input This fix"
    old_object = "the old fetcher because it handles"
    store = MemoryStore(":memory:")
    try:
        _store_git_rows(store, [old_subject, old_object])
        report = replay_full_equality(store)
        assert not report.has_drift
        assert report.matched == report.total_log_rows == 2
        assert _log_rows(store) == [old_subject, old_object]
        derived = derive(
            DerivationInput(raw_text=old_subject, source_kind=INGEST_SOURCE_GIT),
        )
        assert derived.belief is not None
        assert derived.belief.content == old_subject
        assert store.get_belief(derived.belief.id) is not None
    finally:
        store.close()


# --- The post-commit block ---------------------------------------------------


def _hook(repo: Path) -> Path:
    return repo / ".git" / "hooks" / "post-commit"


@pytest.mark.timeout(60)
def test_the_block_comment_says_the_hook_stores_nothing(tmp_path: Path) -> None:
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, timeout=30,
    )
    assert install_commit_ingest_git_hook(tmp_path, command="/x/ingest").status == "installed"
    comments = " ".join(
        ln.lstrip("# ") for ln in _hook(tmp_path).read_text().splitlines()
        if ln.startswith("# ") and "aelfrice commit-ingest" not in ln
    )
    assert comments == (
        "aelfrice created this file. "
        "Starts aelf-commit-ingest in the background after each commit, "
        "so the commit doesn't wait. The command stores nothing in "
        "aelfrice memory. Skips commits that a rebase replays. Delete "
        "this block to stop it."
    )


#: Comments that earlier releases wrote into the block, oldest first.
_OLD_COMMENTS = (
    "# Records this commit's message in aelfrice memory, in the\n"
    "# background, so the commit doesn't wait. Skips commits that a\n"
    "# rebase replays. Delete this block to stop it.\n",
    "# Reads this commit's message in the background, so the commit\n"
    "# doesn't wait. On a phrase such as \"X supersedes Y\", stores the\n"
    "# words on each side in aelfrice memory. Skips commits that a\n"
    "# rebase replays. Delete this block to stop it.\n",
)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("old_comment", _OLD_COMMENTS, ids=["1698", "1874"])
def test_setup_replaces_a_block_with_an_old_comment(
    tmp_path: Path, old_comment: str,
) -> None:
    """A block an older release installed keeps running the same command,
    and the next `aelf setup` rewrites only its comment."""
    subprocess.run(
        ["git", "init", "-q", str(tmp_path)], check=True, timeout=30,
    )
    command = "/x/ingest"
    old_block = (
        f"{GIT_HOOK_BEGIN}\n"
        + old_comment
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
    assert old_comment not in text
    assert "stores nothing" in text
    assert text.count(git_hook_command_line(command)) == 1
    assert text.endswith(f"{GIT_HOOK_END}\necho mine\n")
    assert install_commit_ingest_git_hook(tmp_path, command=command).status == "already_present"
