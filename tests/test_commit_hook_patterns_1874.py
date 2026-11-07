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
import re
import subprocess
import sys
import textwrap
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest

from aelfrice import triple_extractor as te
from aelfrice.derivation import DerivationInput, derive
from aelfrice.entity_extractor import extract_entities
from aelfrice.hook_commit_ingest import _ingest_message, git_hook_ingest
from aelfrice.models import (
    EDGE_DERIVED_FROM,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TESTS,
    INGEST_SOURCE_GIT,
)
from aelfrice.np_pattern import _DET, NOUN_PHRASE_PATTERN
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
    "\u2028", "\u2029",
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


# What base's ingest bank wrote for each message, frozen as literals on
# `github/main` at fecbe3e1, so a bug in the new loop can't also move the
# reference. Each is lowercase in its verb, on one line, and free of a phrase
# that is only function words, so nothing here may change.
_FROZEN_SINGLE_LINE: dict[str, list[tuple[str, str, str, str]]] = {
    "feat: the new fetcher supersedes the old fetcher": [
        ("the new fetcher", "SUPERSEDES", "the old fetcher",
         "feat: the new fetcher supersedes the old fetcher"),
    ],
    "The big old red fast car supersedes a slow blue van": [
        ("The big old red fast car", "SUPERSEDES", "a slow blue van",
         "The big old red fast car supersedes a slow blue van"),
    ],
    "the slow blue van supersedes The big old red fast car": [
        ("the slow blue van", "SUPERSEDES", "The big old red fast car",
         "the slow blue van supersedes The big old red fast car"),
    ],
    "THE NEW FETCHER supersedes THE OLD FETCHER": [
        ("THE NEW FETCHER", "SUPERSEDES", "THE OLD FETCHER",
         "THE NEW FETCHER supersedes THE OLD FETCHER"),
    ],
    "fix: crash when the config mentions a missing path on startup": [
        ("crash when the config", "CITES", "a missing path on startup",
         "fix: crash when the config mentions a missing path on startup"),
    ],
    "the spec is derived from the prior memo": [
        ("the spec", "DERIVED_FROM", "the prior memo",
         "the spec is derived from the prior memo"),
    ],
    "the new index is supported by faster queries": [
        ("faster queries", "SUPPORTS", "the new index",
         "the new index is supported by faster queries"),
    ],
    "this parser is tested by the golden corpus and is covered by nightly runs": [
        ("the golden corpus and is covered", "TESTS", "this parser",
         "this parser is tested by the golden corpus and is covered by nightly runs"),
        ("nightly runs", "TESTS", "by the golden corpus and",
         "parser is tested by the golden corpus and is covered by nightly runs"),
    ],
    "the planner contradicts the memo, and the memo cites the paper": [
        ("the planner", "CONTRADICTS", "the memo",
         "the planner contradicts the memo, and the memo cites the"),
        ("and the memo", "CITES", "the paper",
         "planner contradicts the memo, and the memo cites the paper"),
    ],
    "Our writer implements the spec. Their reader is based on the writer.": [
        ("Our writer", "IMPLEMENTS", "the spec",
         "Our writer implements the spec. Their reader is based on"),
        ("Their reader", "DERIVED_FROM", "the writer",
         "implements the spec. Their reader is based on the writer."),
    ],
    "the writer supersedes the old, and the reader supersedes the new": [
        ("the writer", "SUPERSEDES", "the old",
         "the writer supersedes the old, and the reader supersedes"),
        ("and the reader", "SUPERSEDES", "the new",
         "writer supersedes the old, and the reader supersedes the new"),
    ],
}

_FUNCTION_WORDS = (
    "the", "a", "an", "our", "their", "its", "this", "that", "these", "those",
    "my", "your", "his", "her",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them",
    "myself", "yourself", "himself", "herself", "itself", "ourselves",
    "yourselves", "themselves", "mine", "yours", "hers", "ours", "theirs",
    "which", "who", "whom", "whose", "what", "where", "when", "why", "how",
    "all", "both", "each", "every", "either", "neither", "some", "any", "none",
    "many", "most", "few", "several", "another", "such", "other", "others",
    "one", "more", "less", "much",
    "something", "anything", "everything", "nothing", "someone", "anyone",
    "everyone",
    "and", "or", "but", "nor", "yet", "so", "for", "if", "unless", "because",
    "although", "though", "while", "whereas", "whether", "than", "as", "since",
    "until", "till", "once",
    "of", "to", "in", "on", "at", "by", "with", "from", "about", "into", "onto",
    "over", "under", "after", "before", "between", "among", "through", "during",
    "without", "within", "against", "above", "below", "across", "along",
    "around", "behind", "beside", "besides", "beyond", "despite", "except",
    "inside", "outside", "near", "off", "out", "up", "down", "per", "via",
    "toward", "towards", "upon", "unlike", "versus",
    "am", "is", "are", "was", "were", "be", "been", "being", "do", "does",
    "did", "has", "have", "had", "having", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must", "ought", "isn", "aren", "wasn",
    "weren", "don", "doesn", "didn", "hasn", "haven", "hadn", "won", "wouldn",
    "shouldn", "couldn", "mustn", "shan", "ain",
    "not", "no", "never", "cannot",
    "also", "then", "just", "only", "even", "still", "too", "very", "now",
    "here", "there", "thus", "hence", "however", "therefore", "instead",
    "already", "again", "ever", "rather", "quite", "perhaps", "maybe",
)

_WRAP_TABLE = {"identical": 311, "shorter": 284, "lost": 269}

# The relation sentences behind the wrapped-text table in the PR description.
_WRAP_SENTENCES = (
    "the new fetcher supersedes the old fetcher",
    "this parser is based on the older grammar",
    "the retry queue contradicts the delivery guarantee in the old design note",
    "that config loader is derived from the first bootstrap script",
    "the index writer is tested by the golden corpus runner",
    "our session table relates to the hook audit log",
    "the planner mentions a missing path on startup",
    "the batch importer supersedes the nightly export job in the deploy runbook",
    "the store migration is covered by the upgrade smoke test",
    "the lock budget disagrees with the injection cap",
    "a single worker implements the queue contract",
    "the short form cites the spec",
)


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


@pytest.mark.parametrize("space", ["\t", "  ", " \t "], ids=["tab", "two", "mixed"])
def test_any_whitespace_inside_a_line_separates_the_verb(space: str) -> None:
    assert _pairs(f"the new fetcher{space}supersedes{space}the old fetcher") == [
        ("the new fetcher", EDGE_SUPERSEDES, "the old fetcher"),
    ]
    assert _pairs(f"the spec{space}is{space}derived{space}from{space}the memo") == [
        ("the spec", EDGE_DERIVED_FROM, "the memo"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "3way supersedes the old parser",
        "--force supersedes the old parser",
        "_private supersedes the old parser",
    ],
)
def test_a_phrase_doesnt_start_inside_a_word(text: str) -> None:
    assert _pairs(text) == []


def test_non_ascii_letters_stay_inside_their_word() -> None:
    assert _pairs("the caf\u00e9menu supersedes the na\u00efve parser") == [
        ("the caf\u00e9menu", EDGE_SUPERSEDES, "the na\u00efve parser"),
    ]


def test_one_uppercase_token_in_a_multiword_verb_stops_the_match() -> None:
    assert _pairs("the spec is Based on the prior memo") == []


def test_an_edge_type_name_used_as_a_noun_is_not_a_verb() -> None:
    assert _pairs("docs: describe the SUPERSEDES writer for new readers") == []
    assert _pairs("docs: add a CITES reader in the graph view") == []


def test_the_phrases_on_each_side_keep_ignoring_case() -> None:
    """Six tokens with an uppercase determiner fit a phrase only when case is
    ignored there, so this fails if either side loses it."""
    assert _pairs(
        "The big old red fast car supersedes The slow blue old rusty van"
    ) == [
        ("The big old red fast car", EDGE_SUPERSEDES, "The slow blue old rusty van"),
    ]


@pytest.mark.parametrize("text", _FROZEN_SINGLE_LINE)
def test_a_lowercase_verb_on_one_line_stores_what_base_stored(text: str) -> None:
    assert [
        (t.subject, t.relation, t.object, t.anchor_text) for t in _ingest(text)
    ] == _FROZEN_SINGLE_LINE[text]


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


def test_two_relations_with_one_verb_on_one_line_store_both() -> None:
    assert _pairs(
        "the writer supersedes the old, and the reader supersedes the new"
    ) == [
        ("the writer", EDGE_SUPERSEDES, "the old"),
        ("and the reader", EDGE_SUPERSEDES, "the new"),
    ]


def test_the_anchor_still_reads_past_the_matched_line() -> None:
    [triple] = _ingest("fix: retry\n\nthe writer supersedes the reader\n\nmore context")
    assert triple.subject == "the writer"
    assert triple.anchor_text == (
        "fix: retry\n\nthe writer supersedes the reader\n\nmore context"
    )


# --- Function words ----------------------------------------------------------


def test_the_function_word_list_is_exactly_the_reviewed_one() -> None:
    assert te._INGEST_FUNCTION_WORDS == frozenset(_FUNCTION_WORDS)
    assert len(_FUNCTION_WORDS) == len(set(_FUNCTION_WORDS))


def test_every_determiner_the_noun_phrase_admits_is_a_function_word() -> None:
    determiners = re.findall(r"[a-z]+", _DET)
    assert len(determiners) == 14
    assert set(determiners) <= set(_FUNCTION_WORDS)


@pytest.mark.parametrize("word", _FUNCTION_WORDS)
@pytest.mark.parametrize("form", [str.lower, str.title], ids=["low", "title"])
def test_a_function_word_on_either_side_stores_nothing(
    word: str, form: Callable[[str], str],
) -> None:
    word = form(word)
    assert _pairs(f"{word} supersedes the old parser") == []
    assert _pairs(f"the new parser supersedes {word}") == []
    assert _pairs(f"the new parser is tested by {word}") == []
    assert _pairs(f"{word} is tested by the golden corpus") == []


@pytest.mark.parametrize("word", _FUNCTION_WORDS)
def test_a_function_word_inside_a_longer_phrase_doesnt_count(word: str) -> None:
    assert _pairs(f"parser {word} supersedes {word} parser") == [
        (f"parser {word}", EDGE_SUPERSEDES, f"{word} parser"),
    ]


@pytest.mark.parametrize("word", [w for w in _FUNCTION_WORDS if len(w) > 1])
def test_a_word_in_all_capitals_counts_as_a_name(word: str) -> None:
    name = word.upper()
    assert _pairs(f"{name} supersedes the old parser") == [
        (name, EDGE_SUPERSEDES, "the old parser"),
    ]
    assert _pairs(f"the new parser supersedes {name}") == [
        ("the new parser", EDGE_SUPERSEDES, name),
    ]


@pytest.mark.parametrize("word", ["A", "I", "A-", "I_"])
def test_a_single_capital_letter_is_still_a_function_word(word: str) -> None:
    assert _pairs(f"{word} supersedes the old parser") == []
    assert _pairs(f"the new parser supersedes {word}") == []


def test_a_trailing_hyphen_on_a_name_keeps_it_a_name() -> None:
    assert _pairs("IT- supersedes the old parser") == [
        ("IT-", EDGE_SUPERSEDES, "the old parser"),
    ]


@pytest.mark.parametrize("word", ["None", "Any", "Some", "One", "Each", "Will", "May"])
def test_a_capitalized_listed_word_stores_nothing(word: str) -> None:
    """The loss the docs name: a code name or a person's name matches the list."""
    assert _pairs(f"{word} supersedes the old parser") == []
    assert _pairs(f"the new parser supersedes {word}") == []


@pytest.mark.parametrize(
    "word",
    ["one-shot", "each-other", "any-of", "if-else", "all_of", "each_line", "one_of"],
)
def test_a_word_with_an_inner_hyphen_or_underscore_counts_as_one_word(word: str) -> None:
    assert _pairs(f"{word} supersedes the old parser") == [
        (word, EDGE_SUPERSEDES, "the old parser"),
    ]
    assert _pairs(f"the new parser supersedes {word}") == [
        ("the new parser", EDGE_SUPERSEDES, word),
    ]


@pytest.mark.parametrize("word", ["it-", "the-", "this-", "all-", "one_", "all_", "it_", "the-_"])
def test_a_trailing_hyphen_or_underscore_is_a_fragment_of_the_word(word: str) -> None:
    assert _pairs(f"{word} supersedes the old parser") == []
    assert _pairs(f"the new parser supersedes {word}") == []


def test_a_trailing_hyphen_on_a_content_word_still_stores() -> None:
    assert _pairs("parser- supersedes the old one") == [
        ("parser-", EDGE_SUPERSEDES, "the old one"),
    ]


@pytest.mark.parametrize("apostrophe", ["'", "\u2019"], ids=["straight", "curly"])
@pytest.mark.parametrize(
    "text",
    [
        "it{a}s supersedes the old parser",
        "that{a}s cites the memo",
        "doesn{a}t cites the memo",
        "the old parser is based on it{a}s",
        "the new parser supersedes doesn{a}t",
        "the new parser supersedes isn{a}t",
    ],
)
def test_a_contraction_leaves_no_fragment(text: str, apostrophe: str) -> None:
    assert _pairs(text.format(a=apostrophe)) == []


@pytest.mark.parametrize("apostrophe", ["'", "\u2019"], ids=["straight", "curly"])
def test_a_phrase_doesnt_start_after_an_apostrophe(apostrophe: str) -> None:
    assert _pairs(f"the parser{apostrophe}s cache supersedes the old one") == [
        ("cache", EDGE_SUPERSEDES, "the old one"),
    ]
    assert _pairs(f"the new parser supersedes the module{apostrophe}s own") == [
        ("the new parser", EDGE_SUPERSEDES, "the module"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "the new parser supersedes all the",
        "both these supersedes the old parser",
        "the new parser supersedes this one",
        "the median separation the comment cites does not",
        "which also contradicts the documented",
        "or one that contradicts pyproject",
    ],
)
def test_a_phrase_of_only_function_words_stores_nothing(text: str) -> None:
    assert _pairs(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "feat: new parser supersedes that",
        "feat: new parser supersedes that\n\nCloses 12",
        "X supersedes it\nlater text",
        "X supersedes all\nlater text",
        "X supersedes both\nlater text",
        "the new parser supersedes the\nold parser",
        "the new parser supersedes the `old` parser",
        "and the claim contradicts the `doc` text",
        "so a function whose prose cites a `ref` here",
        "fix: tidy the parser\n\nThis supersedes the old parser",
        "the new parser handles retries\nand supersedes the old parser",
        "the new parser handles retries\nand contradicts the memo",
        "the new parser handles retries\nalso cites the paper",
        "the planner cites of\nthe paper",
        "the doc is derived from and\nthe rest",
    ],
)
def test_a_review_case_stores_nothing(text: str) -> None:
    assert _pairs(text) == []


@pytest.mark.parametrize(
    ("text", "obj"),
    [
        ("the new fetcher supersedes the old\nfetcher", "the old"),
        ("the new fetcher supersedes the earlier\nfetcher", "the earlier"),
        ("X supersedes old\nlater text", "old"),
    ],
)
def test_an_adjective_with_no_noun_still_stores(text: str, obj: str) -> None:
    """Telling `the old` from `the parser` takes a grammar check, so the rule
    leaves these in. A later change to that decision has to change this test."""
    assert [(r, o) for _, r, o in _pairs(text)] == [(EDGE_SUPERSEDES, obj)]


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


def test_the_wrapped_text_table_reproduces() -> None:
    """Each sentence at all 72 offsets behind a run of letters and a period,
    which no phrase can cross, wrapped at 72 columns and compared with the same
    text unwrapped. A lost relation either has the break touching its verb or
    leaves only function words on a side."""
    counts: Counter[str] = Counter()
    for sentence in _WRAP_SENTENCES:
        for pad in range(72):
            flat = ("a" * pad + ". " if pad else "") + sentence
            unwrapped = _pairs(flat)
            assert len(unwrapped) == 1, flat
            wrapped = _pairs(textwrap.fill(flat, 72))
            if wrapped == unwrapped:
                counts["identical"] += 1
            elif wrapped:
                counts["shorter"] += 1
            else:
                counts["lost"] += 1
    assert counts == _WRAP_TABLE


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


def _hook_ran(done: Path) -> int:
    return len(done.read_text(encoding="utf-8").splitlines()) if done.exists() else 0


@pytest.mark.timeout(120)
def test_the_installed_block_stores_only_what_the_rules_allow(
    tmp_path: Path, db: Path,
) -> None:
    """Commits go through the real `post-commit` block. A wrapper records each
    finished background run, so the test waits on that and not on a clock."""
    script = Path(sys.executable).parent / "aelf-commit-ingest"
    if not script.exists():
        pytest.skip("aelf-commit-ingest is not installed next to the interpreter")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "commit.gpgsign", "false")
    done = tmp_path / "done"
    wrapper = tmp_path / "ingest-and-report"
    wrapper.write_text(
        f"#!/bin/sh\n'{script}' \"$@\"\necho ran >> '{done}'\n", encoding="utf-8",
    )
    wrapper.chmod(0o755)
    assert install_commit_ingest_git_hook(repo, command=str(wrapper)).status == "installed"
    messages = (
        "docs: describe the SUPERSEDES writer for new readers",
        "refactor: tidy the fetch\n\nThe new fetcher\nsupersedes\nthe old fetcher",
        "feat: new parser supersedes that\n\nCloses 12",
        "fix: tidy the parser\n\nThis supersedes the old parser",
        "fix: tidy the reader\n\nThe new reader retries\nand supersedes the old reader",
        "feat: the new codec supersedes the old codec",
    )
    for i, message in enumerate(messages, start=1):
        (repo / f"f{i}").write_text(message, encoding="utf-8")
        _git(repo, "add", f"f{i}")
        _git(repo, "commit", "-q", "-m", message)
        deadline = time.monotonic() + 30
        while _hook_ran(done) < i:
            assert time.monotonic() < deadline, f"the block didn't run for commit {i}"
            time.sleep(0.05)
    assert _edges(db) == [("the new codec", EDGE_SUPERSEDES, "the old codec")]


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
    assert comments == (
        "aelfrice created this file. "
        "Reads this commit's message in the background, so the commit "
        "doesn't wait. On a phrase such as \"X supersedes Y\", stores the "
        "words on each side in aelfrice memory. Skips commits that a "
        "rebase replays. Delete this block to stop it."
    )


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
    "This supersedes the old parser",
    "docs: describe the SUPERSEDES writer for new readers",
)


def test_the_benchmark_replica_agrees_with_extract_triples() -> None:
    """`triple_precision_1376` exits when its replica loop and `extract_triples`
    disagree, so its ingest walk has to match per line too."""
    for body in _BENCHMARK_BODIES:
        ingest_counts, _ = _fires_by_template([body], _INGEST_PATTERNS, ingest_rules=True)
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
