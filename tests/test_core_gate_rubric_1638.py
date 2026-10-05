"""#1638: the core admission gate's rubric, prompt and label parser."""
from __future__ import annotations

import json

import pytest

from aelfrice import core_gate as cg


def test_the_digest_pins_the_rubric_and_prompt_text() -> None:
    """Editing the rubric or prompt must come with a version bump.

    If this fails, the classifier text changed. Bump `CLASSIFIER_VERSION`
    and set `CLASSIFIER_DIGEST` to the new `text_digest()`, so labels made
    under the old text are not reused.
    """
    assert cg.text_digest() == cg.CLASSIFIER_DIGEST


def test_the_rubric_keeps_the_reference_rules() -> None:
    """The definitions and decision rules stay as the raters had them."""
    for line in (
        "- **A = self-contained truth-apt proposition.** A declarative claim that is true or false on its own, without the surrounding conversation.",
        "- **B = truth-apt only with context.** It is a declarative claim, but its truth depends on unresolved references",
        "- **C = not truth-apt.** Sentence fragments with no complete claim",
        "- A heading or lead-in ending in \":\" with no claim is C.",
        "- When torn between A and B, ask: would a stranger with no access to the conversation know what the claim is about? If not, B.",
    ):
        assert line in cg.RUBRIC


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
