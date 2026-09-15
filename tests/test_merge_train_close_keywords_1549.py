r"""The merge-train's auto-close emulates GitHub's close keywords (#1549).

#1541 made the whole pull-request body reach the matcher and deliberately left
two fidelity gaps open, because both change *which* issues close. #1549 ruled
the framing: step 7/7 stands in for a merge-commit close that the fast-forward
model cannot produce, so it reads the body the way GitHub reads it.

Three things are pinned here, and they pull against each other, which is why
none of them is sufficient alone:

1. **All nine keywords link.** A parser that matched only `closes` / `fixes` /
   `resolves` left `Fixed #N` open, and the train logged the same line a body
   with no keyword at all produces.
2. **A keyword a Markdown reader would not render as prose does not link.** A
   parser that matched everything passes (1) and closes the issue named in a
   body that merely documents the syntax. Fences, indented blocks, inline code
   spans, block quotes and HTML comments are all covered.
3. **A refused keyword is named on stderr.** This is the acceptance criterion
   #1549 calls the one that matters: (1) and (2) together still let a
   rejection look exactly like an absence in the step log. stdout must stay
   bare numbers, because the workflow reads it into a shell variable.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "merge_train_linked_issues.py"
_WORKFLOW = _REPO / ".github" / "workflows" / "merge-train.yml"

sys.path.insert(0, str(_REPO / "scripts"))

from merge_train_linked_issues import (  # noqa: E402
    CROSS_REPO,
    IN_COMMENT,
    IN_FENCE,
    IN_INDENT,
    IN_QUOTE,
    IN_SPAN,
    KEYWORDS,
    linked_issues,
    parse,
)

# --------------------------------------------------------------------------
# Gap 1: the keyword set is GitHub's, not the shell's three.
# --------------------------------------------------------------------------

_GITHUB_KEYWORDS = (
    "close",
    "closes",
    "closed",
    "fix",
    "fixes",
    "fixed",
    "resolve",
    "resolves",
    "resolved",
)


def test_the_module_names_exactly_githubs_nine_keywords() -> None:
    """Spelled out here rather than imported, so a narrowing edit fails.

    Asserting `KEYWORDS == KEYWORDS` would pass whatever the module said.
    """
    assert sorted(KEYWORDS) == sorted(_GITHUB_KEYWORDS)


@pytest.mark.parametrize("keyword", _GITHUB_KEYWORDS)
def test_every_github_keyword_links(keyword: str) -> None:
    assert linked_issues(f"{keyword} #7") == [7]
    assert linked_issues(f"{keyword.upper()} #7") == [7]
    assert linked_issues(f"{keyword.capitalize()} #7") == [7]


@pytest.mark.parametrize(
    "keyword", ["closed", "fix", "fixed", "close", "resolve", "resolved"]
)
def test_the_six_spellings_1541_missed_now_link(keyword: str) -> None:
    """The six #1549 gap 1 named. Redundant with the sweep above on purpose.

    If a future author re-narrows the set to the shell's three, the sweep
    fails on six rows and this fails on six more, and the reason is in the
    test name rather than only in a parametrisation id.
    """
    assert linked_issues(f"{keyword.capitalize()} #4242.") == [4242]


@pytest.mark.parametrize(
    "body",
    [
        "precloses #7",  # \b must still hold at the front
        "prefixes #7",
        "unresolved #7",
        "closesX #7",  # no boundary between the keyword and the digit run
        "Closes#7",  # whitespace is still required
        "Closes issue #7",  # GitHub does not accept this either
        "#7",
        "See #7 for context",
    ],
)
def test_widening_the_keywords_did_not_widen_what_counts_as_a_link(
    body: str,
) -> None:
    assert linked_issues(body) == []


# --------------------------------------------------------------------------
# Gap 2: a keyword a Markdown reader renders as code, a quote or a comment.
# --------------------------------------------------------------------------

# Every body below links #8 from prose and names #7 somewhere inert. Asserting
# both halves is what stops a parser that returns nothing from passing.
_INERT_BODIES = [
    ("```\nCloses #7\n```\n\nFixes #8", IN_FENCE),
    ("```markdown\nCloses #7\n```\n\nFixes #8", IN_FENCE),
    ("~~~\nCloses #7\n~~~\n\nFixes #8", IN_FENCE),
    ("~~~text\nCloses #7\n~~~\n\nFixes #8", IN_FENCE),
    ("`````\nCloses #7\n`````\n\nFixes #8", IN_FENCE),
    ("````\n```\nCloses #7\n```\n````\n\nFixes #8", IN_FENCE),
    ("  ```\nCloses #7\n  ```\n\nFixes #8", IN_FENCE),
    ("prose\n\n    Closes #7\n\nFixes #8", IN_INDENT),
    ("prose\n\n\tCloses #7\n\nFixes #8", IN_INDENT),
    ("prose\n\n        Closes #7\n\nFixes #8", IN_INDENT),
    ("prose\n\n    Closes #7\n    still code\n\nFixes #8", IN_INDENT),
    ("Write `Closes #7` in the body.\n\nFixes #8", IN_SPAN),
    ("``a `tick` and Closes #7``\n\nFixes #8", IN_SPAN),
    ("> Closes #7\n\nFixes #8", IN_QUOTE),
    ("> quoting a review:\n> Closes #7\n\nFixes #8", IN_QUOTE),
    ("   > Closes #7\n\nFixes #8", IN_QUOTE),
    ("<!-- Closes #7 -->\n\nFixes #8", IN_COMMENT),
    ("<!--\nCloses #7\n-->\n\nFixes #8", IN_COMMENT),
    ("Fixes #8 <!-- template line: Closes #7 -->", IN_COMMENT),
]


@pytest.mark.parametrize(
    ("body", "reason"), _INERT_BODIES, ids=[b[:24] for b, _ in _INERT_BODIES]
)
def test_an_inert_keyword_does_not_link_but_the_prose_one_does(
    body: str, reason: str
) -> None:
    found, refused = parse(body)
    assert found == [8], "the prose keyword must still link"
    assert [r.reason for r in refused] == [reason]
    assert [r.text for r in refused] == ["Closes #7"]


def test_a_pr_documenting_this_very_syntax_closes_nothing() -> None:
    """#1549's motivating wrong close, written as a real PR body would be."""
    body = (
        "## Summary\n"
        "\n"
        "The parser acts on a trailer such as:\n"
        "\n"
        "```\n"
        "Closes #1234\n"
        "Fixes #1235\n"
        "Resolves #1236\n"
        "```\n"
        "\n"
        "That is all it does.\n"
    )
    found, refused = parse(body)
    assert found == []
    assert [r.text for r in refused] == ["Closes #1234", "Fixes #1235", "Resolves #1236"]
    assert {r.reason for r in refused} == {IN_FENCE}


def test_an_unclosed_fence_swallows_the_rest_of_the_body() -> None:
    """The documented choice: CommonMark ends the block at end of document.

    The two errors are not symmetric -- refusing a close leaves an open issue
    a human notices, making one closes an issue nobody asked to close -- so an
    unterminated fence stays a fence rather than reverting to literal text.
    """
    found, refused = parse("```\nCloses #7\n\nFixes #8\n")
    assert found == []
    assert [(r.text, r.reason) for r in refused] == [
        ("Closes #7", IN_FENCE),
        ("Fixes #8", IN_FENCE),
    ]


def test_an_unclosed_html_comment_swallows_the_rest_of_the_body() -> None:
    found, refused = parse("<!-- Closes #7\n\nFixes #8\n")
    assert found == []
    assert {r.reason for r in refused} == {IN_COMMENT}


def test_a_fence_closes_only_on_its_own_character_and_length() -> None:
    """A shorter run, or the other fence character, does not close it."""
    assert linked_issues("````\n```\n~~~~\nCloses #7\n````\n\nFixes #8") == [8]
    assert linked_issues("```\n~~~\nCloses #7\n```\n\nFixes #8") == [8]


def test_a_closing_fence_carrying_text_does_not_close() -> None:
    """CommonMark allows an info string on the opener only."""
    assert linked_issues("```\ncode\n``` trailing\nCloses #7\n") == []


def test_an_indented_block_cannot_interrupt_a_paragraph() -> None:
    """A documented divergence, pinned so it is a decision and not a surprise.

    CommonMark folds an indented line directly under a paragraph into that
    paragraph, so the keyword is prose and does link. It needs a blank line
    above it to become code.
    """
    assert linked_issues("a paragraph line\n    Closes #7\n") == [7]
    assert linked_issues("a paragraph line\n\n    Closes #7\n") == []


def test_an_unmatched_backtick_marks_nothing_inert() -> None:
    """Otherwise one stray tick in a body silences every keyword after it."""
    assert linked_issues("a ` stray tick, and Closes #7") == [7]


def test_text_after_a_comment_closes_is_live_again() -> None:
    assert linked_issues("<!-- ignore Closes #7 --> and Fixes #8") == [8]


@pytest.mark.parametrize(
    "body",
    [
        "<!-- note -->Fixes #8",  # the `-->` and the keyword touch
        "`x`Fixes #8",  # so do the closing backtick and the keyword
    ],
)
def test_a_keyword_starting_where_an_inert_span_ends_is_prose(body: str) -> None:
    """An inert span is half-open, and the first live offset is its end.

    The cases above sit one character tighter than
    `test_text_after_a_comment_closes_is_live_again`, which leaves a space
    after the `-->`. Without that space the keyword starts at exactly the
    span's end offset, so widening the containment test to `offset <= end`
    turns both of these into refusals.
    """
    assert linked_issues(body) == [8]


def test_a_cross_repository_link_is_refused_out_loud() -> None:
    """Out of scope to *follow*; in scope to stop passing over in silence."""
    found, refused = parse("Closes robotrocketscience/aelfrice#7\n\nFixes #8")
    assert found == [8]
    assert [(r.text, r.reason) for r in refused] == [
        ("Closes robotrocketscience/aelfrice#7", CROSS_REPO)
    ]


def test_a_rejection_reports_the_line_it_was_found_on() -> None:
    body = "one\ntwo\n```\nCloses #7\n```\n"
    (_, refused) = parse(body)
    assert [r.line for r in refused] == [4]


def test_a_rejection_names_the_keyword_on_one_normalised_line() -> None:
    """GitHub accepts a newline between the keyword and the `#N`, so a match
    can span one. The warning is read out of a step log a line at a time, so
    the quoted text collapses its whitespace instead of breaking the message
    in two and leaving `#7` on a line of its own.
    """
    (_, refused) = parse("```\nResolves\n#7\n```\n")
    assert [r.text for r in refused] == ["Resolves #7"]
    assert "\n" not in refused[0].message()
    assert refused[0].message() == (
        'warning: ignored "Resolves #7" on line 2: inside a fenced code block.'
    )


def test_a_keyword_split_across_lines_takes_its_keyword_line_context() -> None:
    """A documented divergence: classification is by where the match starts."""
    assert linked_issues("```\nResolves\n#7\n```\n") == []
    assert linked_issues("Resolves\n#7\n") == [7]


# --------------------------------------------------------------------------
# AC5: a rejection must not print what an absence prints.
# --------------------------------------------------------------------------

# One interpreter per CLI test, reading a string and printing numbers -- no
# network, no store, no lock. Scaled by the suite's own knob so a loaded
# machine reports contention as slowness rather than as a failure (#1307).
_CLI_TIMEOUT = 30 * int(os.environ.get("AELF_TEST_TIMEOUT_SCALE", "4"))


def _run(args: list[str], stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        timeout=_CLI_TIMEOUT,
    )


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_a_rejected_keyword_is_not_silent() -> None:
    """The defect #1549 calls the one that matters.

    A body whose only keyword the parser refused used to produce exactly what
    a body with no keyword produces: empty stdout and empty stderr.
    """
    absent = _run([], stdin="no trailer here\n")
    refused = _run([], stdin="```\nCloses #7\n```\n")

    assert absent.returncode == refused.returncode == 0
    assert absent.stdout == refused.stdout == ""
    assert absent.stderr == "", "an absence has nothing to report"
    assert refused.stderr != absent.stderr, (
        "a refused keyword printed the same thing as no keyword at all"
    )


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_the_rejection_names_the_text_and_the_reason() -> None:
    refused = _run([], stdin="intro\n\n```\nCloses #7\n```\n")
    assert refused.returncode == 0
    assert "Closes #7" in refused.stderr
    assert IN_FENCE in refused.stderr
    assert "line 4" in refused.stderr


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_rejections_never_reach_stdout() -> None:
    """stdout is read into a shell variable and split on whitespace."""
    proc = _run([], stdin="```\nCloses #7\n```\n\nFixes #8\n")
    assert proc.returncode == 0
    assert proc.stdout.split() == ["8"]
    assert "#7" in proc.stderr


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_every_rejection_gets_its_own_line() -> None:
    proc = _run([], stdin="```\nCloses #7\nFixes #8\nResolves #9\n```\n")
    lines = [ln for ln in proc.stderr.splitlines() if ln.startswith("warning:")]
    assert len(lines) == 3
    assert proc.stdout == ""


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_dry_run_still_reports_rejections() -> None:
    proc = _run(["--dry-run"], stdin="```\nCloses #7\n```\n\nFixes #8\n")
    assert proc.stdout.strip() == "would close #8"
    assert "Closes #7" in proc.stderr


# --------------------------------------------------------------------------
# The workflow must surface that stderr.
# --------------------------------------------------------------------------


def _live_lines() -> list[str]:
    """The merge-train's command lines, with the comment lines dropped.

    The step comment quotes both `2>/dev/null` and `2>&1` to say not to add
    them, so any check that reads the raw file text sees those spellings
    whether or not a command still uses one.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if not ln.strip().startswith("#")]


def _script_invocation() -> str:
    r"""The merge-train lines that run the parser, comments dropped.

    The invocation is wrapped over two lines with a `\` continuation, so the
    redirect that would swallow stderr could sit on either of them.
    """
    lines = _live_lines()
    for i, line in enumerate(lines):
        if "merge_train_linked_issues.py" in line:
            return "\n".join(lines[i : i + 3])
    raise AssertionError("merge-train.yml no longer runs the parser")


def test_the_workflow_does_not_swallow_the_parsers_stderr() -> None:
    """Command substitution captures stdout only, so stderr reaches the log.

    A `2>/dev/null` or a `2>&1` on this call would undo #1549 without touching
    the parser: the first discards the diagnostics, the second folds them into
    the issue-number list the shell then loops over.
    """
    call = _script_invocation()
    assert "2>/dev/null" not in call
    assert "2>&1" not in call
    assert "2>" not in call


def test_that_assertion_is_not_vacuous() -> None:
    """Other commands in this workflow do redirect stderr, so the check is narrow.

    Read the same comment-filtered lines `_script_invocation` reads, not the
    raw file: the step comment added for #1549 names both spellings, so a
    whole-file search reports a redirect that no command runs.
    """
    live = "\n".join(_live_lines())
    assert "2>/dev/null" in live
    assert "2>&1" in live


def test_the_workflow_no_longer_calls_the_gaps_undecided() -> None:
    """The step comment named both gaps as open; #1549 closed them."""
    whole = _WORKFLOW.read_text(encoding="utf-8")
    assert "left for a decision" not in whole


def test_the_docstring_states_the_decision_and_what_remains() -> None:
    """AC1: the gaps section is replaced, not merely amended."""
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert "Two fidelity gaps this deliberately does NOT close" not in doc
    assert "this step emulates GitHub" in doc
    assert "Divergences that REMAIN, deliberately" in doc
    for remaining in ("Commit messages", "Cross-repository", "target branch"):
        assert remaining in doc


def test_the_docstring_cites_githubs_keyword_list() -> None:
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert "docs.github.com" in doc
    assert re.search(r"linking-a-pull-request-to-an-issue", doc)
