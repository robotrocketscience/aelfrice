"""Regression corpora for #1376 — the extractor used to read bare
plural nouns as relation verbs.

Two corpora, deliberately opposed:

* `COMMIT_PROSE_ZERO_TRIPLES` — verbatim subjects and body paragraphs
  from this repository's own commit history. Every row produced at
  least one triple before the fix, and every triple was a fragment
  pair, not an assertion. `hook_commit_ingest` hands exactly this kind
  of text to `extract_triples` on every successful `git commit`, and
  `ingest_triples` mints a belief per phrase, so each of these rows was
  worth two junk belief rows per commit.
* `RELATIONAL_PROSE` — genuine relational statements, at least one per
  registered relation family. The fix has to stop the fragments
  without degenerating into "extract nothing".

Line breaks inside the zero-triple rows are the original commit's
wrapping and are load-bearing: the noun-phrase regex joins on `\\s+`,
so a wrapped line is exactly where the greedy phrase runs on.
"""
from __future__ import annotations

import pytest

from aelfrice.models import (
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TEMPORAL_NEXT,
    EDGE_TESTS,
)
from aelfrice.np_pattern import DETERMINER_LED_NOUN_PHRASE_PATTERN
from aelfrice.triple_extractor import _PATTERNS, extract_triples

# (id, verbatim commit prose). Ids name the collision that fired.
COMMIT_PROSE_ZERO_TRIPLES: list[tuple[str, str]] = [
    # The two examples executed in the issue body.
    ("tests-quantifier-subject",
     "Two tests that encoded the old behaviour are removed."),
    ("tests-bare-subject-adjective-object",
     "codex-skills tests unaffected"),
    # Real bodies. Every one of these is a plural noun.
    ("tests-det-led-subject",
     "audit-touching test surface stays green -- 29 passed here plus 46\n"
     "across two hook files. The SessionStart tests\n"
     "structurally cannot catch it: they read row [0] of a file that\n"
     "contains only SessionStart rows."),
    ("tests-singular-verb-lookalike",
     "The file now holds a FIFO window of recently-seen ids (bound 128)\n"
     "and the\npredicate tests membership. Eviction can only cause an\n"
     "extra fire, never a missed one."),
    ("tests-numeral-modifier",
     "The four new tests fail on f1fa6d12. The shipped #578 tests cover\n"
     "only A,A and A,B, both of which pass under the single slot"),
    ("covers-adverb-subject",
     "Discovery now covers both\nconventional scopes, filtered to files\n"
     "that exist and deduplicated by\nresolved path"),
    ("covers-conjunction-subject",
     "generation seed, the scope-id mint), and an engine-refusal arm\n"
     "covers the\nwrites nobody thought of"),
    ("follows-conforms-to-sense",
     "Contemporary writers set origin explicitly, so the flip only\n"
     "matters once per legacy DB. Marker\nfollows the entity-backfill\n"
     "convention; the pass rides the existing\nopen commit."),
    ("supports-edge-type-name-as-noun",
     "DERIVED_FROM is 0.5 and SUPPORTS is 1.0. Rewrite both rationales"),
    ("replaces-pytest-table-row",
     "  rescue replaces the cap               1 failed, 36 passed"),
    ("extends-bare-gerund-subject",
     "Sharpening extends the single clause with a three-step note"),
]

# (text, relation that must appear). One row per template that survived
# the fix, so a template silently dropped later turns this red.
RELATIONAL_PROSE: list[tuple[str, str]] = [
    ("the new index supports faster queries", EDGE_SUPPORTS),
    ("the new index is supported by the cache layer", EDGE_SUPPORTS),
    ("the proposal cites the prior memo", EDGE_CITES),
    ("the proposal mentions the prior memo", EDGE_CITES),
    ("the new finding contradicts the earlier paper", EDGE_CONTRADICTS),
    ("the benchmark disagrees with the published table", EDGE_CONTRADICTS),
    ("this commit supersedes the legacy parser", EDGE_SUPERSEDES),
    ("this commit replaces the legacy parser", EDGE_SUPERSEDES),
    ("the cache layer relates to retrieval", EDGE_RELATES_TO),
    ("the cache layer is related to retrieval", EDGE_RELATES_TO),
    ("the spec is derived from the prior memo", EDGE_DERIVED_FROM),
    ("the spec is based on the prior memo", EDGE_DERIVED_FROM),
    ("the spec extends the prior memo", EDGE_DERIVED_FROM),
    ("the worker implements the ingest spec", EDGE_IMPLEMENTS),
    ("the worker is an implementation of the ingest spec", EDGE_IMPLEMENTS),
    ("the worker realizes the ingest spec", EDGE_IMPLEMENTS),
    ("the worker fulfills the ingest spec", EDGE_IMPLEMENTS),
    ("the reindex comes after the backfill", EDGE_TEMPORAL_NEXT),
    ("the reindex is after the backfill", EDGE_TEMPORAL_NEXT),
    ("the backfill is followed by the reindex", EDGE_TEMPORAL_NEXT),
    ("the reindex succeeds the backfill", EDGE_TEMPORAL_NEXT),
    ("the idempotency suite is a test for the merge path", EDGE_TESTS),
    ("the idempotency suite is test of the merge path", EDGE_TESTS),
    ("the merge path is tested by the idempotency suite", EDGE_TESTS),
    ("the merge path is covered by the idempotency suite", EDGE_TESTS),
]

_BARE_PLURAL_NOUN_VERBS = frozenset({"tests", "covers", "follows"})


@pytest.mark.parametrize(
    "case_id,text",
    COMMIT_PROSE_ZERO_TRIPLES,
    ids=[c for c, _ in COMMIT_PROSE_ZERO_TRIPLES],
)
def test_commit_prose_yields_no_triples(case_id: str, text: str) -> None:
    triples = extract_triples(text)
    assert triples == [], (
        f"{case_id}: commit prose still mints fragment triples "
        f"{[(t.subject, t.relation, t.object) for t in triples]}"
    )


@pytest.mark.parametrize(
    "text,relation", RELATIONAL_PROSE, ids=[t for t, _ in RELATIONAL_PROSE],
)
def test_relational_prose_still_extracts(text: str, relation: str) -> None:
    triples = extract_triples(text)
    assert any(t.relation == relation for t in triples), (
        f"{text!r} no longer yields {relation}: "
        f"{[(t.subject, t.relation, t.object) for t in triples]}"
    )


def test_passive_tests_form_points_test_at_the_thing_under_test() -> None:
    """`is tested by` / `is covered by` replaced bare `tests` / `covers`,
    so their direction has to be pinned: EDGE_TESTS runs test -> spec."""
    for text in (
        "the merge path is tested by the idempotency suite",
        "the merge path is covered by the idempotency suite",
    ):
        triples = [t for t in extract_triples(text) if t.relation == EDGE_TESTS]
        assert len(triples) == 1, text
        assert triples[0].subject == "the idempotency suite", text
        assert triples[0].object == "the merge path", text


def test_passive_temporal_form_points_successor_at_predecessor() -> None:
    """`is followed by` replaced bare `follows`; TEMPORAL_NEXT runs
    successor -> predecessor."""
    triples = [
        t for t in extract_triples("the backfill is followed by the reindex")
        if t.relation == EDGE_TEMPORAL_NEXT
    ]
    assert len(triples) == 1
    assert triples[0].subject == "the reindex"
    assert triples[0].object == "the backfill"


def test_no_bare_plural_noun_verb_is_registered() -> None:
    """Restoring any of these as a single-token template is the mutation
    that turns COMMIT_PROSE_ZERO_TRIPLES red; fail here first, with a
    name, rather than in eleven parametrized corpus rows."""
    single_token = {p.template for p in _PATTERNS if " " not in p.template}
    collisions = single_token & _BARE_PLURAL_NOUN_VERBS
    assert not collisions, (
        f"bare plural-noun relation verbs re-registered: {sorted(collisions)}"
    )


def test_every_single_token_template_frames_its_subject() -> None:
    """A single-token verb has to demand a determiner-led subject NP;
    bare containment is what produced the fragment triples."""
    unframed = [
        p.template for p in _PATTERNS
        if " " not in p.template and not p.det_subject
    ]
    assert unframed == [], (
        f"single-token templates matching on bare containment: {unframed}"
    )
    for p in _PATTERNS:
        if p.det_subject:
            assert DETERMINER_LED_NOUN_PHRASE_PATTERN in p.regex.pattern, (
                f"{p.template!r} claims det_subject but did not compile it in"
            )
