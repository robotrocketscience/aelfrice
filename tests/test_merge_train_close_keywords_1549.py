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
    ISSUE_URL,
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
        "Closes#7",  # a delimiter is still required
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
# Gap 1b: the colon GitHub's own page accepts after a keyword.
# --------------------------------------------------------------------------

# Verbatim from the page cited in the module docstring: "The keywords can be
# followed by colons or in uppercase. For example: `Closes: #10`,
# `CLOSES #10`, or `CLOSES: #10`."
_GITHUB_COLON_EXAMPLES = ["Closes: #10", "CLOSES #10", "CLOSES: #10"]


@pytest.mark.parametrize("body", _GITHUB_COLON_EXAMPLES)
def test_githubs_own_three_examples_all_link(body: str) -> None:
    """The colon form used to match nothing, and matching nothing is silent.

    `parse("Closes: #10")` returned `([], [])`: no link and no rejection, so
    the step log printed what it prints for a body with no keyword at all --
    verbatim the failure the whole of AC5 exists to kill.
    """
    assert linked_issues(body) == [10]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Closes: #10", [10]),  # GitHub's published example
        ("Closes:#10", [10]),  # the colon is itself the delimiter
        ("Closes : #10", []),  # the colon must touch the keyword
        ("Closes::#10", []),  # one colon, not a run of them
        ("Closes#10", []),  # no delimiter at all
    ],
    ids=["colon-space", "colon-tight", "space-colon", "double-colon", "bare"],
)
def test_the_colon_binds_to_the_keyword_and_does_not_repeat(
    body: str, expected: list[int]
) -> None:
    """GitHub publishes the colon but not its spacing; the module rules on it.

    The three undocumented forms are decided in the module docstring, under
    'What "followed by a colon" means here', and pinned here so the reading is
    a decision rather than whatever the regex happened to do.
    """
    assert linked_issues(body) == expected


def test_the_colon_form_still_obeys_the_block_exclusions() -> None:
    """Widening the separator must not widen where a keyword may fire."""
    found, refused = parse("```\nCloses: #7\n```\n\nFixes: #8")
    assert found == [8]
    assert [(r.text, r.reason) for r in refused] == [("Closes: #7", IN_FENCE)]


def test_the_1504_shaped_prose_links_and_that_is_the_accepted_cost() -> None:
    """The known false positive the colon buys, recorded rather than hidden.

    Merged PR #1504's body ends a clause on one of the nine words, and no
    rule of text tells that apart from a close directive. GitHub closes #1329
    from this body on an ordinary merge, so a train standing in for GitHub
    closes it too.
    """
    body = "All ruled prerequisites are closed: #1329, #1412 and #1428.\n"
    found, refused = parse(body)
    assert found == [1329]
    assert refused == [], "a prose match is a link, not a rejection"


def test_the_docstring_records_the_false_positive_the_colon_buys() -> None:
    """A cost accepted in a ruling and left out of the file is not recorded."""
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert "#1504" in doc
    assert "false positive" in doc


# --------------------------------------------------------------------------
# Gap 1c: the reference half, and the forms GitHub's own pages document for it.
# --------------------------------------------------------------------------

# Every casing of `GH-` GitHub's renderer links to the same issue. Measured
# against GitHub rather than assumed, with:
#     gh api --method POST /markdown -f mode=gfm -f context=OWNER/REPO \
#         -f text='GH-10 and gh-10 and Gh-10 and gH-10 and #10'
# whose output carries one issue-link anchor to /issues/10 per spelling.
_GH_CASINGS = ["GH-10", "gh-10", "Gh-10", "gH-10"]


@pytest.mark.parametrize("reference", _GH_CASINGS)
def test_the_gh_reference_form_links_in_any_case(reference: str) -> None:
    """`Closes GH-10` used to be the silent no-op AC5 forbids outright.

    `parse("Closes GH-10")` returned `([], [])`, so the step printed for it
    exactly what it prints for a body carrying no keyword: empty stdout,
    empty stderr, `no linked issues parsed from PR body`. The repository's
    own advisory `pr-metadata.yml` job matches `GH-` and had already told the
    author the link was fine.
    """
    assert linked_issues(f"Closes {reference}") == [10]
    assert linked_issues(f"resolved {reference}") == [10]


def test_the_gh_form_takes_the_colon_separator_too() -> None:
    """The two halves of the pattern are independent, and stay that way."""
    assert linked_issues("Closes: GH-10") == [10]
    assert linked_issues("CLOSES:gh-10") == [10]


@pytest.mark.parametrize(
    "body",
    [
        "Closes GH10",  # the hyphen is part of the form
        "ClosesGH-10",  # a delimiter is still required
        "Closes GH-",  # a reference needs a number
        "Closes ugh-10",  # the form has to start where the reference does
    ],
)
def test_the_gh_form_did_not_widen_what_counts_as_a_link(body: str) -> None:
    assert linked_issues(body) == []


def test_the_gh_form_obeys_the_block_exclusions() -> None:
    """Widening the reference half must not widen where a keyword may fire."""
    found, refused = parse("```\nCloses GH-7\n```\n\nFixes GH-8")
    assert found == [8]
    assert [(r.text, r.reason) for r in refused] == [("Closes GH-7", IN_FENCE)]


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/octo-org/octo-repo/issues/26",
        "https://github.com/octo-org/octo-repo/pull/26",
        "https://github.com/robotrocketscience/aelfrice/issues/26",
    ],
    ids=["issue", "pull", "this-repository"],
)
def test_a_full_issue_url_is_refused_out_loud(url: str) -> None:
    """Refused rather than followed, and never in silence.

    The parser carries no repository identity, so it cannot tell its own URL
    from another repository's -- the third case is this repository's own and
    is refused with the other two, exactly as `robotrocketscience/aelfrice#7`
    already is. Matching the form is what turns a silent miss into a named
    rejection.
    """
    found, refused = parse(f"Closes {url}\n\nFixes #8")
    assert found == [8]
    assert [(r.text, r.reason) for r in refused] == [(f"Closes {url}", ISSUE_URL)]


# Every reference form GitHub documents: the keyword page's syntax table and
# its formatting note, plus the "Autolinked references and URLs" page that
# same page links for what a reference to an issue may look like.
_DOCUMENTED_FORMS = [
    ("#N", "Closes #10", [10], []),
    ("GH-N", "Closes GH-10", [10], []),
    ("OWNER/REPOSITORY#N", "Fixes octo-org/octo-repo#100", [], [CROSS_REPO]),
    (
        "issue URL",
        "Closes https://github.com/octo-org/octo-repo/issues/26",
        [],
        [ISSUE_URL],
    ),
    (
        "multiple issues",
        "Resolves #10, resolves #123, resolves octo-org/octo-repo#100",
        [10, 123],
        [CROSS_REPO],
    ),
    ("a colon", "Closes: #10", [10], []),
    ("uppercase", "CLOSES #10", [10], []),
    ("uppercase and a colon", "CLOSES: #10", [10], []),
]


@pytest.mark.parametrize(
    ("body", "found", "reasons"),
    [(body, found, reasons) for _, body, found, reasons in _DOCUMENTED_FORMS],
    ids=[name for name, *_ in _DOCUMENTED_FORMS],
)
def test_no_documented_reference_form_is_silently_unmatched(
    body: str, found: list[int], reasons: list[str]
) -> None:
    """AC5 over the whole documented surface, not only the forms #1549 named.

    Linking is a fine outcome and refusing by name is a fine outcome. The one
    outcome forbidden is `([], [])`: a form that neither links nor says why
    prints what a body with no keyword at all prints, which is the failure
    this issue exists to remove. Both `GH-N` and the URL row did exactly that
    until this commit.
    """
    got, refused = parse(body)
    assert (got, [r.reason for r in refused]) != ([], []), (
        "a documented form matched nothing and reported nothing"
    )
    assert got == found
    assert [r.reason for r in refused] == reasons


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


def test_a_code_span_after_a_comment_closes_mid_line_is_still_code() -> None:
    """A wrong close: the remainder used to be scanned for comments only.

    The comment opens on line 1 and closes part-way through line 2, so the
    rest of line 2 is live text. It reached a comment-only scan rather than
    the live-line path, so the backticks around `Closes #7` marked nothing and
    the parser returned `([7, 8], [])` -- closing an issue whose keyword sits
    inside a code span, a context the docstring lists as excluded.
    """
    found, refused = parse("<!--\nnote --> `Closes #7` and Fixes #8")
    assert found == [8]
    assert [(r.text, r.reason) for r in refused] == [("Closes #7", IN_SPAN)]


def test_a_comment_marker_inside_a_code_span_does_not_open_a_comment() -> None:
    """The mirror of the case above, and a wrong refusal rather than a close.

    Comments were scanned before code spans on a live line, so a `<!--`
    written between backticks opened a comment that swallowed the rest of the
    body: `Fixes #8` came back refused as `inside an HTML comment`.
    """
    found, refused = parse("a `comment marker <!-- inside` a code span Fixes #8")
    assert found == [8]
    assert refused == []


def test_a_backtick_inside_a_comment_does_not_open_a_code_span() -> None:
    """The other direction of the same rule: the comment opened first.

    Ranking spans over comments instead would end the comment at the backtick
    run and hand `Closes #7` back as prose.
    """
    found, refused = parse("<!-- a `tick` and Closes #7 -->\n\nFixes #8")
    assert found == [8]
    assert [(r.text, r.reason) for r in refused] == [("Closes #7", IN_COMMENT)]


def test_an_unterminated_comment_opened_inside_a_code_span_is_inert_text() -> None:
    """The span wins, so nothing is left open and the next line is live."""
    assert linked_issues("`<!-- still code`\nFixes #8") == [8]


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
    """A match that spans a newline still warns on a single log line.

    GitHub accepts any whitespace between the keyword and the `#N`, so the
    match can carry a newline. A step log is read a line at a time, so the
    quoted text collapses its whitespace rather than breaking the warning in
    two and leaving `#7` on a line of its own.
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


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_the_gh_form_no_longer_prints_what_an_absent_keyword_prints() -> None:
    """The defect at the step's own boundary, where a human reads the log."""
    absent = _run([], stdin="no trailer here\n")
    gh_form = _run([], stdin="Closes GH-10\n")

    assert (absent.stdout, absent.stderr) == ("", "")
    assert gh_form.stdout.split() == ["10"]


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_a_url_link_is_refused_on_stderr_and_not_in_silence() -> None:
    url = "https://github.com/octo-org/octo-repo/issues/26"
    proc = _run([], stdin=f"Closes {url}\n")

    assert proc.returncode == 0
    assert proc.stdout == "", "stdout is the issue-number list the shell splits"
    assert ISSUE_URL in proc.stderr
    assert url in proc.stderr


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


def test_the_docstring_enumerates_every_documented_reference_form() -> None:
    """A form ruled on in review and left out of the file is not ruled on.

    The enumeration is the deliverable, not only the two forms it changed:
    the next reader has to be able to check the list against GitHub's pages
    without re-deriving which forms were considered.
    """
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert "Every documented reference form is acted on or refused out loud" in doc
    for form in ("`#N`", "`GH-N`", "`OWNER/REPOSITORY#N`", "/issues/26"):
        assert form in doc, f"the docstring does not rule on {form}"


def test_the_docstring_cites_githubs_keyword_list() -> None:
    """The citation must be one whole URL, not a host and a path that happen
    to both appear.

    Matched as a single pattern rather than as two `in` checks. Two
    independent substring assertions pass on a docstring that names the host
    in one sentence and the page in another, which is not a citation a reader
    can follow; and a bare `"docs.github.com" in doc` also reads to CodeQL as
    an incomplete URL sanitization check, since that is the shape of one.
    """
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert re.search(
        r"https://docs\.github\.com/\S*linking-a-pull-request-to-an-issue", doc
    ), "the docstring does not cite GitHub's closing-keyword page by URL"
