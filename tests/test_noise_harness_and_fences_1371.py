"""#1371 §9 and §10 — harness scaffolding, and fence parity between the paths.

Scope note: this covers §9 and §10 only. **§1 is deliberately not fixed here**
— the ack/shell regexes still discard durable beliefs like
"No vector embeddings, ever.", and `test_section_1_is_still_open` pins that so
the split cannot be mistaken for a complete fix. §1 is held behind a funded
`scan_repo` admission re-measurement (#1398).

Each fix gets a must-survive arm as well as a must-filter arm. A denylist is
easy to widen until it eats prose, and an assertion that only checks the bad
input passes just as well for a rule that filters everything.
"""
from __future__ import annotations

import pytest

from aelfrice.extraction import extract_sentences
from aelfrice.noise_filter import is_transcript_noise
from aelfrice.scanner import _split_paragraphs

# The scaffolding the harness injects into a turn. These arrive at ingest as
# `type in ("user","assistant")` like any other transcript text, so before
# #1371 §9 `derivation` stamped them ORIGIN_USER_TRANSCRIPT with the
# undeflated user prior — stored as if the user had typed them.
HARNESS_SCAFFOLDING = [
    "<system-reminder>As you answer the user's questions, use this context.</system-reminder>",
    "<command-name>clear</command-name>",
    "<command-message>clear</command-message>",
    "<command-args></command-args>",
    "<local-command-stdout></local-command-stdout>",
]

# Prose that must NOT be caught by the §9 prefixes. Each is chosen to sit close
# to one of them: a word that merely starts with the same letters, an unrelated
# angle-bracket tag, and prose that names the scaffolding rather than being it.
MUST_SURVIVE_PROSE = [
    "The command-line interface exposes a search subcommand for beliefs.",
    "Use <system> design docs when planning the rollout of the new indexer.",
    "Commands are dispatched through a single entry point in the CLI module.",
    "The system reminder mechanism is documented in the hook design notes.",
]


@pytest.mark.parametrize("text", HARNESS_SCAFFOLDING)
def test_harness_scaffolding_is_filtered(text: str) -> None:
    """§9: each scaffolding form is recognised as transcript noise."""
    assert is_transcript_noise(text) is True


@pytest.mark.parametrize("text", MUST_SURVIVE_PROSE)
def test_prose_near_the_new_prefixes_survives(text: str) -> None:
    """§9 must-survive: the prefixes are narrow enough not to eat prose.

    Without this arm, widening a prefix to `<command` or `<system` would look
    like a passing change: every must-filter assertion would still hold.
    """
    assert is_transcript_noise(text) is False


def test_section_9_prefixes_are_anchored_not_substring() -> None:
    """The rule is `startswith`, so scaffolding *quoted mid-sentence* survives.

    A belief that discusses the scaffolding is a real belief. This pins the
    distinction the prefix list depends on.
    """
    quoted = (
        "When the harness injects <system-reminder> blocks the ingest path "
        "must not treat them as user-authored content."
    )
    assert is_transcript_noise(quoted) is False


# --- §10 -----------------------------------------------------------------

_FENCED_DOC = (
    "The scanner walks the tree and emits one candidate per paragraph.\n\n"
    "```xml\n"
    "<system-reminder>scaffolding, not prose, and comfortably long</system-reminder>\n"
    "```\n\n"
    "Beliefs are deduplicated by a content hash across rescans of one tree."
)


def test_onboard_path_strips_fences() -> None:
    """§10: a fenced block no longer survives `_split_paragraphs`.

    Before the fix the fenced paragraph was over `_MIN_PARAGRAPH_CHARS`, had no
    matching `is_noise` category, and was stored verbatim.
    """
    paragraphs = _split_paragraphs(_FENCED_DOC)
    assert len(paragraphs) == 2
    assert not any("```" in p for p in paragraphs)
    assert not any("system-reminder" in p for p in paragraphs)


def test_both_ingest_paths_agree_on_fenced_content() -> None:
    """§10's acceptance criterion: one document, two paths, no fence in either.

    The paths tokenise differently — paragraphs vs sentences — so the assertion
    is that neither retains the fenced region, not that the units match.
    """
    onboard = _split_paragraphs(_FENCED_DOC)
    transcript = extract_sentences(_FENCED_DOC)
    for unit in [*onboard, *transcript]:
        assert "```" not in unit
        assert "system-reminder" not in unit
    # Both paths still carry the surrounding prose, so the strip is targeted
    # rather than a blanket drop of the document.
    assert any("scanner walks the tree" in u for u in onboard)
    assert any("scanner walks the tree" in u for u in transcript)


def test_prose_paragraph_survives_an_inline_fence() -> None:
    """A fence inside a prose paragraph removes the code, keeps the prose."""
    doc = (
        "Run the migration before upgrading:\n"
        "```\naelf spine clear\n```\n"
        "and confirm the edge count afterwards."
    )
    paragraphs = _split_paragraphs(doc)
    assert len(paragraphs) == 1
    assert "aelf spine clear" not in paragraphs[0]
    assert "Run the migration before upgrading" in paragraphs[0]
    assert "confirm the edge count afterwards" in paragraphs[0]


def test_unterminated_fence_keeps_its_content_on_both_paths() -> None:
    """The malformed case: neither path silently swallows the text.

    `CODE_FENCE_RE` requires a closing delimiter, so an unterminated fence is
    not matched and its content survives. That is the transcript path's
    long-standing behaviour and onboard now inherits it rather than inventing a
    stricter rule — a stricter one would delete real prose after a stray
    backtick run.

    One residual difference is asserted rather than glossed: onboard keeps the
    literal ``` marker, the transcript path does not, because
    `extract_sentences` also strips *inline* backticks (its step 2). That strip
    is outside §10's scope, which is the fenced-region rule only. Pinned here so
    the divergence is on the record instead of being discovered as a surprise.
    """
    tail = "this fence is never closed and the text after it runs on long enough"
    doc = f"Intro paragraph that is long enough to be kept.\n\n```\n{tail}"

    onboard = _split_paragraphs(doc)
    transcript = extract_sentences(doc)
    assert any(tail in p for p in onboard)
    assert any(tail in s for s in transcript)

    assert any("```" in p for p in onboard)
    assert not any("```" in s for s in transcript)


def test_shared_pattern_is_one_object_not_two_copies() -> None:
    """Both paths must reference the *same* compiled pattern.

    The defect §10 fixes was one path having a fence rule and the other not.
    Re-inlining a second copy in either module would reintroduce exactly that
    divergence, and every behavioural test above would still pass.
    """
    from aelfrice import extraction, scanner

    assert scanner.CODE_FENCE_RE is extraction.CODE_FENCE_RE


# --- the part deliberately NOT fixed -------------------------------------


def test_section_1_is_still_open() -> None:
    """§1 is held behind the #1398 admission re-measurement — pin that.

    `is_transcript_noise` still discards durable product statements. If a later
    change fixes §1, this test fails and must be deleted *with* that change, so
    the fix is recorded rather than absorbed silently.
    """
    assert is_transcript_noise("No vector embeddings, ever.") is True
