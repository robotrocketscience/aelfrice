"""#1159 §3 / #1376: single-token relation verbs that are also plural nouns.

`triple_extractor` registered bare `tests`, `covers`, `follows`, `extends`,
`supports` and `replaces` as relation verbs. Every one of them is also an
ordinary plural noun or a non-relational verb in commit prose, so the ingest
path minted fragment "beliefs" from commit bodies — and because
`ingest_triples` records *both* phrases of a triple, one bogus match becomes
two pieces of junk in the store.

Operator ruling 2026-08-06 scopes the fix to the **write path**: the six
templates are dropped from an ingest-only bank selected by an explicit
parameter, and `context_rebuilder`'s read-path call — made on every prompt,
with behaviour that was never measured — stays byte-identical.

Three corpora, because any one of them alone is passed by a wrong fix:

* `MUST_YIELD_NO_TRIPLES` — real commit prose. Passed by deleting the
  extractor entirely.
* `MUST_STILL_FIRE` — genuine relations. Passed by changing nothing.
* the read-path arm — the same collision rows, asserted to *still* fire
  under the default bank. Passed by neither, and it is what makes the
  change write-path-scoped rather than global.
"""

from __future__ import annotations

import ast

import hashlib
import json

import pytest

from aelfrice.triple_extractor import (
    _INGEST_EXCLUDED_TEMPLATES,
    _INGEST_PATTERNS,
    _PATTERNS,
    _RelationPattern,
    extract_triples,
)

# --- Corpus 1: real commit prose that must mint nothing ---------------
#
# The first two rows are #1376's own executed examples. The rest are drawn
# verbatim from `github/main` commit bodies in the 400-body window the
# precision harness walks (`benchmarks/triple_precision_1376.py`), chosen
# because each fires a *different* one of the six dropped templates.
MUST_YIELD_NO_TRIPLES: tuple[str, ...] = (
    # #1376's seed rows, both executed in the issue body.
    "Two tests that encoded the old behaviour are removed.",
    "codex-skills tests unaffected",
    # `tests` — the dominant collision, 51 of 82 fires on the sample.
    "The two tests that missed their branches are covered now.",
    "Both tests pin that None stays distinguishable from absent.",
    # `covers` — 12 of 82.
    "The recognition predicate covers all four shapes this project emits.",
    "Once the gate covers every non-advisory check the required set is whole.",
    # `replaces` — 3 of 82.
    "The way the tests it replaces were written is not preserved.",
    # `follows` — 1 of 82.
    "The lowercase stage that follows it is str.lower, not casefold.",
    # `supports` and `extends` never fired on the sample, but they are in
    # the dropped set, so the corpus carries a constructed row for each —
    # otherwise nothing here would go red if one were quietly restored.
    "The three supports that hold the frame are bolted, not welded.",
    "Every belief that extends past the window is dropped.",
)

# --- Corpus 2: genuine relations that must survive the constraint -----
#
# Each row is (text, expected_subject, relation, expected_object). Every
# dropped template's surviving sibling appears at least once, which is what
# makes "each dropped verb keeps a reachable relation" a tested claim rather
# than a comment.
MUST_STILL_FIRE: tuple[tuple[str, str, str, str], ...] = (
    # SUPERSEDES survives `replaces` via `supersedes`.
    (
        "The v4 entry supersedes the earlier triage note.",
        "The v4 entry",
        "SUPERSEDES",
        "the earlier triage note",
    ),
    # DERIVED_FROM survives `extends` via `is derived from` / `is based on`.
    (
        "The retry cap is derived from the SLA budget.",
        "The retry cap",
        "DERIVED_FROM",
        "the SLA budget",
    ),
    (
        "The cache layer is based on the LRU spec.",
        "The cache layer",
        "DERIVED_FROM",
        "the LRU spec",
    ),
    # TEMPORAL_NEXT survives `follows` via `comes after` / `succeeds`.
    (
        "The rollout comes after the migration.",
        "The rollout",
        "TEMPORAL_NEXT",
        "the migration",
    ),
    (
        "The staging deploy succeeds the smoke run.",
        "The staging deploy",
        "TEMPORAL_NEXT",
        "the smoke run",
    ),
    # SUPPORTS survives `supports` via the passive `is supported by`.
    (
        "The v3 gate is supported by the bench results.",
        "the bench results",
        "SUPPORTS",
        "The v3 gate",
    ),
    # TESTS lost both single-token forms, so it gains the two passive
    # multi-token forms #1376 names as the obvious replacement.
    (
        "The parser is tested by test_parser.",
        "test_parser",
        "TESTS",
        "The parser",
    ),
    (
        "The audit row is covered by test_audit_rows.",
        "test_audit_rows",
        "TESTS",
        "The audit row",
    ),
    # Relations with no dropped template must be untouched by all of this.
    (
        "The new scheduler contradicts the documented ordering.",
        "The new scheduler",
        "CONTRADICTS",
        "the documented ordering",
    ),
    (
        "The adapter implements the storage protocol.",
        "The adapter",
        "IMPLEMENTS",
        "the storage protocol",
    ),
)


# --- The write path mints nothing from commit prose -------------------


@pytest.mark.parametrize("text", MUST_YIELD_NO_TRIPLES)
def test_ingest_bank_mints_no_triples_from_commit_prose(text: str) -> None:
    assert extract_triples(text, constrain_collision_verbs=True) == []


@pytest.mark.parametrize(("text", "subject", "relation", "obj"), MUST_STILL_FIRE)
def test_ingest_bank_still_extracts_genuine_relations(
    text: str, subject: str, relation: str, obj: str
) -> None:
    got = extract_triples(text, constrain_collision_verbs=True)
    assert (subject, relation, obj) in [
        (t.subject, t.relation, t.object) for t in got
    ], got


# --- The read path is byte-identical ----------------------------------


@pytest.mark.parametrize("text", MUST_YIELD_NO_TRIPLES)
def test_read_path_still_fires_on_the_same_rows(text: str) -> None:
    """The constraint is write-path-only, and this is what proves it.

    Without this arm, dropping the six templates from `_PATTERNS` outright
    — which would silently change what `context_rebuilder` queries on every
    prompt — passes the entire rest of this file.
    """
    assert extract_triples(text) != [], (
        f"{text!r} no longer fires under the default bank; the ingest "
        "constraint has leaked onto the read path"
    )


def test_default_bank_is_the_unconstrained_one() -> None:
    assert _INGEST_PATTERNS is not _PATTERNS
    for text in MUST_YIELD_NO_TRIPLES:
        assert extract_triples(text) == extract_triples(
            text, constrain_collision_verbs=False
        )


def _extract_triples_calls(module: object) -> list[ast.Call]:
    """Every `extract_triples(...)` call in `module`, parsed rather than grepped.

    These two call sites are the whole scoping decision — the write path opts
    in, the read path must not — and both were asserted by scanning source
    *lines* for a substring. That is not a behaviour assert wearing a
    behaviour assert's docstring; it is a formatting assert. Rewriting the
    rebuilder's call across several lines moves the keyword onto a line that
    does not contain `extract_triples(`, so the negative guard stops seeing it
    and the read path can be constrained on every prompt with the full suite
    green (verified: 7,667 passed, byte-identical to the unmutated run). The
    positive guard has the opposite defect — it requires one exact literal, so
    a harmless line wrap fails it, and a mention in a comment satisfies it.

    Parsing the module removes both. Formatting is invisible to the AST, and a
    string in a comment is not a `Call` node.
    """
    from pathlib import Path

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "extract_triples"
    ]


def test_context_rebuilder_does_not_pass_the_constraint() -> None:
    """The read-path call site must stay byte-identical (ruling 2026-08-06).

    Asserted against the source text rather than by calling it, because the
    rebuilder's own call is buried behind a store and a query, and a test
    that drove it would not fail for the reason this one is named after.
    """
    import aelfrice.context_rebuilder as cr

    calls = _extract_triples_calls(cr)
    assert calls, "extract_triples call site vanished from context_rebuilder"
    for call in calls:
        passed = {kw.arg for kw in call.keywords}
        assert "constrain_collision_verbs" not in passed, (
            "the read path opted into the write-path constraint at "
            f"context_rebuilder.py:{call.lineno} — every prompt would be "
            "filtered, which the 2026-08-06 ruling forbids"
        )


def test_hook_commit_ingest_does_pass_the_constraint() -> None:
    """The mirror image: the write path must actually opt in.

    Threading a parameter nothing passes is the failure mode that would
    leave this whole change inert while every other test here stays green.
    """
    import aelfrice.hook_commit_ingest as hci

    calls = _extract_triples_calls(hci)
    assert calls, "extract_triples call site vanished from hook_commit_ingest"
    for call in calls:
        passed = {
            kw.arg: kw.value
            for kw in call.keywords
            if isinstance(kw.value, ast.Constant)
        }
        assert passed.get("constrain_collision_verbs") is not None, (
            "the write path stopped opting in at "
            f"hook_commit_ingest.py:{call.lineno} — the constraint would be "
            "threaded but inert, and every other test here would stay green"
        )
        assert passed["constrain_collision_verbs"].value is True


# --- The bank composition itself --------------------------------------


def test_excluded_templates_are_exactly_the_six_named_by_the_issue() -> None:
    assert _INGEST_EXCLUDED_TEMPLATES == {
        "supports",
        "replaces",
        "extends",
        "follows",
        "tests",
        "covers",
    }


def test_every_excluded_template_leaves_its_edge_type_reachable() -> None:
    """Dropping a verb must not remove a relation from the ingest surface.

    This is the claim the module comment makes; without the assert it is
    prose, and a seventh exclusion could silently orphan an edge type.
    """
    reachable = {p.edge_type for p in _INGEST_PATTERNS}
    for pat in _PATTERNS:
        if pat.template in _INGEST_EXCLUDED_TEMPLATES:
            assert pat.edge_type in reachable, (
                f"dropping {pat.template!r} orphaned {pat.edge_type}"
            )


def test_read_path_bank_behaviour_is_byte_identical_to_pre_1376() -> None:
    """`_PATTERNS`' *digest* moved under #1376; its behaviour must not have.

    `_RelationPattern` gained a `template` field so a bank can be filtered
    by verb, which changed the dataclass repr and therefore the pinned
    manifest digest — forcing a `DETECTOR_THRESHOLDS_VERSION` bump for a
    change that alters nothing the extractor does. `DIGEST_HISTORY[2]`
    claims exactly that, and this is the check behind the claim.

    The signature below is over (regex source, edge type, swap) — the
    three things that decide what the read path matches and emits. It was
    computed on `github/main` at dde5a5f5, before this change, and is
    equal at this head. If a later change moves it, the read path moved
    with it and `context_rebuilder` is affected.
    """
    signature = hashlib.sha256(
        json.dumps(
            [(p.regex.pattern, p.edge_type, p.swap) for p in _PATTERNS]
        ).encode("utf-8")
    ).hexdigest()
    assert signature == (
        "ed7dc10d976d0f0ec0b0e222b8378987e4791227cd80c71ed8caaed27729afb7"
    )


def test_no_excluded_template_survives_in_the_ingest_bank() -> None:
    assert not (
        {p.template for p in _INGEST_PATTERNS} & _INGEST_EXCLUDED_TEMPLATES
    )


def test_restoring_a_collision_verb_turns_the_zero_corpus_red() -> None:
    """The mutation guard AC4 asks for, run rather than described.

    Rebuilds the ingest bank with `tests` and `covers` put back and asserts
    the zero-triple corpus stops being zero. A corpus that cannot detect its
    own regression is the failure mode this repo keeps shipping by accident.
    """
    restored: tuple[_RelationPattern, ...] = tuple(
        p for p in _PATTERNS if p.template in ("tests", "covers")
    ) + _INGEST_PATTERNS
    assert restored != _INGEST_PATTERNS

    fired = []
    for text in MUST_YIELD_NO_TRIPLES:
        for pat in restored:
            if pat.template not in ("tests", "covers"):
                continue
            hit = pat.regex.search(text)
            if hit and hit.group("subject").strip() and hit.group("object").strip():
                fired.append(text)
                break
    assert len(fired) >= 4, (
        "restoring `tests`/`covers` should re-fire on several corpus rows; "
        f"only {fired} fired, so the corpus does not pin the regression"
    )
