"""#1638: the core admission gate's rubric, prompt, and label parser."""
from __future__ import annotations

import json

import pytest

from aelfrice import core_gate as cg

#: One digest per classifier version. APPEND ONLY: never edit an existing
#: entry. A change to the rubric, the prompt, MAX_BATCH, or the model tier
#: needs a new CLASSIFIER_VERSION and a new entry here, so a label cached
#: under one classifier is never reused under another.
PINNED_DIGESTS: dict[str, str] = {
    "core-gate-1": "ceb65a5e721cbc6399db5f507c90fc17f990f4e6b88e41de6e6c208b37023506",
}

#: The examples that replace the reference raters' private ones. Everything
#: else in the rubric is the raters' text, word for word.
SWAPPED_EXAMPLES = (
    "Python's json module is part of the standard library.",
    "It failed on the second run.",
    "That value is stale.",
    "The limit is 50, not 100.",
    "returns nothing.",
    "in the second column.",
    "Run the tests",
    "Never delete the backup",
    "Open the next file",
    "The general principle:",
    "Always sort the list",
)

#: The raters' rubric with each swapped example masked as <EX>.
REFERENCE_RUBRIC_MASKED = """\
Assign each snippet exactly one class:

- **A = self-contained truth-apt proposition.** A declarative claim that is true or false on its own, without the surrounding conversation. It may use proper names, file names, issue numbers or project terms, as long as it does not depend on unresolved pronouns or deictic references to the conversation. Example: "<EX>"
- **B = truth-apt only with context.** It is a declarative claim, but its truth depends on unresolved references ("it", "this step", "the cap", "Step 2", "I", "we", "the above") or on the missing surrounding turn. Examples: "<EX>"; "<EX>"; "<EX>"
- **C = not truth-apt.** Sentence fragments with no complete claim ("<EX>", "<EX>"), commands or instructions ("<EX>", "<EX>", "<EX>"), questions, headings or labels ("<EX>"), and code, shell, tables or markup (including XML-like `<belief ...>` blocks).

Decision rules:

- Imperatives and rules addressed to an agent ("<EX>", "Do not X") are C, even when phrased as policy.
- A heading or lead-in ending in ":" with no claim is C.
- A multi-sentence snippet is classed by its main content. If it contains at least one complete self-contained claim and is not mainly code or markup, use A or B for that claim.
- When torn between A and B, ask: would a stranger with no access to the conversation know what the claim is about? If not, B.
"""


def test_the_current_version_has_its_pinned_digest() -> None:
    """If this fails, the classifier changed: add a new version, don't edit.

    Bump `CLASSIFIER_VERSION` and add `{version: prompt_digest()}` to
    PINNED_DIGESTS. Editing the existing entry instead would let labels made
    under the old text be reused under the new one.
    """
    assert cg.CLASSIFIER_VERSION in PINNED_DIGESTS
    assert cg.prompt_digest() == PINNED_DIGESTS[cg.CLASSIFIER_VERSION]


def test_no_two_versions_share_a_digest() -> None:
    """A new version that names the same classifier would split one cache."""
    assert len(set(PINNED_DIGESTS.values())) == len(PINNED_DIGESTS)


def test_the_digest_covers_the_assembly_the_limit_and_the_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = cg.prompt_digest()
    for name, value in (
        ("MAX_BATCH", cg.MAX_BATCH + 1),
        ("CLASSIFIER_MODEL_TIER", "other"),
        ("PROMPT_FOOTER", cg.PROMPT_FOOTER + " "),
    ):
        with monkeypatch.context() as m:
            m.setattr(cg, name, value)
            assert cg.prompt_digest() != base, name


def test_the_rubric_is_the_reference_text_except_the_examples() -> None:
    masked = cg.RUBRIC
    for example in SWAPPED_EXAMPLES:
        assert f'"{example}"' in masked, example
        masked = masked.replace(f'"{example}"', '"<EX>"')
    assert masked == REFERENCE_RUBRIC_MASKED


def test_build_prompt_carries_rubric_and_snippets() -> None:
    prompt = cg.build_prompt([(3, "The widgetprompt cache holds 50 entries."), (7, "Run it.")])
    assert cg.RUBRIC in prompt
    payload = json.loads(prompt.rsplit("\n", 2)[-2])
    assert payload == [
        {"index": 3, "text": "The widgetprompt cache holds 50 entries."},
        {"index": 7, "text": "Run it."},
    ]


@pytest.mark.parametrize("snippets", [
    [],
    [(i, "x") for i in range(cg.MAX_BATCH + 1)],
    [(1, "a"), (1, "b")],
])
def test_build_prompt_rejects_bad_batches(snippets: list[tuple[int, str]]) -> None:
    with pytest.raises(ValueError):
        cg.build_prompt(snippets)


def test_build_prompt_accepts_a_full_batch() -> None:
    assert cg.build_prompt([(i, "x") for i in range(cg.MAX_BATCH)])


def test_parse_labels_accepts_a_complete_reply() -> None:
    reply = '[{"index": 2, "label": "B"}, {"index": 1, "label": "A"}, {"index": 3, "label": "C"}]'
    assert cg.parse_labels(reply, {1, 2, 3}) == {1: "A", 2: "B", 3: "C"}


@pytest.mark.parametrize("reply", [
    "not json",
    '{"index": 1, "label": "A"}',
    '["A"]',
    '[{"index": "1", "label": "A"}]',
    '[{"index": true, "label": "A"}]',
    '[{"index": 1, "label": "D"}]',
    '[{"index": 1, "label": "a"}]',
    '[{"index": 1, "label": "A"}, {"index": 1, "label": "B"}]',
    '[{"index": 1, "label": "A"}]',
    '[{"index": 1, "label": "A"}, {"index": 2, "label": "A"}, {"index": 9, "label": "C"}]',
])
def test_parse_labels_rejects_anything_short_of_a_full_valid_reply(reply: str) -> None:
    with pytest.raises(ValueError):
        cg.parse_labels(reply, {1, 2})


def test_parse_labels_rejects_an_index_labeled_twice_even_when_complete() -> None:
    """Two labels for one index are ambiguous; keeping either is a guess."""
    with pytest.raises(ValueError, match="labeled twice"):
        cg.parse_labels('[{"index": 1, "label": "A"}, {"index": 1, "label": "C"}]', {1})


def test_parse_labels_rejects_a_boolean_index_even_when_it_equals_one() -> None:
    """`True == 1` in Python, so a bare int check would accept it."""
    with pytest.raises(ValueError, match="integer"):
        cg.parse_labels('[{"index": true, "label": "A"}]', {1})
