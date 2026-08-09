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
    (
        "<local-command-caveat>Caveat: the messages below were generated "
        "while running local commands.</local-command-caveat>"
    ),
]

# Prose that must NOT be caught by the §9 prefixes. Each is chosen to sit close
# to one of them: a word that merely starts with the same letters, an unrelated
# angle-bracket tag, and prose that names the scaffolding rather than being it.
MUST_SURVIVE_PROSE = [
    "The command-line interface exposes a search subcommand for beliefs.",
    "Use <system> design docs when planning the rollout of the new indexer.",
    "Commands are dispatched through a single entry point in the CLI module.",
    "The system reminder mechanism is documented in the hook design notes.",
    # The two that make the widening claim in the docstring below true. The
    # rule is `startswith`, so a must-survive string can only engage a widened
    # prefix if it *begins* with the widened stem — none of the four above do,
    # and without these two, collapsing the six prefixes to "<command" and
    # "<system" passes the entire suite.
    "<system> blocks in the prompt template are rendered before the user turn.",
    "<command> elements in the legacy XML schema map to CLI subcommands.",
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


def test_an_inline_fence_mention_does_not_open_a_fenced_region() -> None:
    """A ``` inside prose must not flip fence parity for the rest of the doc.

    `CODE_FENCE_RE` is anchored to line starts. Unanchored, the mention below
    pairs with the *opening* delimiter of the real fence that follows: the
    prose tail is deleted as fence interior, and the code body — token literal
    included — is promoted into a prose paragraph, clears `is_noise`, and is
    stored as a belief. That is the inverse of what §10 set out to do, and the
    onboard path is the one that walks a repo's own documentation, where prose
    about fences is common: on this repo `docs/user/CONFIG.md` drops from 248
    paragraphs to 120 unanchored, against 206 anchored.

    Every other §10 case here uses a well-formed fence and cannot see this.
    """
    doc = (
        "Markdown opens a code block with ``` at the start of a line.\n\n"
        '```python\nAPI_TOKEN = "do-not-store-me"\n```\n\n'
        "Beliefs are deduplicated by a content hash across rescans of one tree."
    )
    paragraphs = _split_paragraphs(doc)

    assert len(paragraphs) == 2
    # The prose survives whole. Unanchored it was truncated at the mention,
    # losing "at the start of a line." and gaining the fence body.
    assert paragraphs[0] == "Markdown opens a code block with ``` at the start of a line."
    assert paragraphs[1].startswith("Beliefs are deduplicated")
    # And the fenced body is dropped rather than stored as prose.
    assert not any("API_TOKEN" in p for p in paragraphs)


def test_a_language_tagged_line_inside_a_block_does_not_close_it() -> None:
    """A closing delimiter may not carry an info string (CommonMark).

    ``` ```python ``` on its own line is body text when a block is already
    open. Accepting `[^\\n]*` on the closing side ends the span there, which
    leaks the rest of the block into the cleaned output *and* leaves the real
    closing delimiter behind as stray prose — the opposite of what a fence
    stripper is for.

    Every other case in this module closes with a bare ```, so none can see
    it. Reachability is not bounded by this repo: 0 of its 176 markdown files
    change under the fix, but both paths read documents the repo does not own.
    """
    doc = (
        "Intro paragraph that must survive the strip entirely.\n\n"
        "```\nfirst = 1\n```python\nSECRET = \"do-not-store-me\"\n```\n\n"
        "Closing paragraph that must also survive."
    )
    paragraphs = _split_paragraphs(doc)

    assert paragraphs[0] == "Intro paragraph that must survive the strip entirely."
    assert paragraphs[-1] == "Closing paragraph that must also survive."
    # The whole block goes, including the half after the language-tagged line.
    assert not any("SECRET" in p for p in paragraphs)
    assert not any("first = 1" in p for p in paragraphs)
    # And no orphaned delimiter is promoted to prose.
    assert not any(p.strip().startswith("```") for p in paragraphs)


def test_shared_pattern_is_one_object_not_two_copies() -> None:
    """Both paths must reference the *same* compiled pattern.

    The defect §10 fixes was one path having a fence rule and the other not.
    Re-inlining a second copy in either module would reintroduce exactly that
    divergence, and every behavioural test above would still pass.
    """
    from aelfrice import extraction, scanner

    assert scanner.CODE_FENCE_RE is extraction.CODE_FENCE_RE


# --- the part deliberately NOT fixed -------------------------------------


def test_section_1_is_closed() -> None:
    """§1 is fixed; this is the inverse of the marker #1406 left here.

    That marker asserted `is_transcript_noise("No vector embeddings, ever.")
    was still True, and said in its own docstring that whichever change fixed
    §1 must delete it so the fix is recorded rather than absorbed silently.
    This is that change. Inverted rather than deleted, so the sentence #1159
    named first stays pinned in the file that tracked it.

    The full corpus lives in `tests/test_noise_filter_ack_shell_1371.py`.
    """
    assert is_transcript_noise("No vector embeddings, ever.") is False
