"""#1371 §9 — harness scaffolding stored as user-authored belief.

Scope note: this covers §9 only. **§1 is deliberately not fixed here**
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

from aelfrice.noise_filter import is_transcript_noise

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


# --- the part deliberately NOT fixed -------------------------------------


def test_section_1_is_still_open() -> None:
    """§1 is held behind the #1398 admission re-measurement — pin that.

    `is_transcript_noise` still discards durable product statements. If a later
    change fixes §1, this test fails and must be deleted *with* that change, so
    the fix is recorded rather than absorbed silently.
    """
    assert is_transcript_noise("No vector embeddings, ever.") is True
