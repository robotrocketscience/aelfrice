r"""The merge-train asks GitHub which issues a body closes (#1549).

#1541 made the whole pull-request body reach the matcher and left two fidelity
gaps open, because both change *which* issues close. #1549 ruled the framing:
step 7/7 stands in for a merge-commit close that the fast-forward model cannot
produce, so it must read the body the way GitHub reads it.

An earlier revision emulated that reading with a line-state Markdown scanner and
was wrong once per review round, the last time on this repository's own
pull-request template. The scanner is gone. `scripts/merge_train_linked_issues.py`
now renders the body through GitHub's `/markdown` endpoint and acts on the
issue-link anchors GitHub itself produced.

Four things are pinned here, and none is sufficient alone:

1. **What GitHub actually renders.** Replayed from
   `tests/data/merge_train_github_renders.json`, which holds verbatim responses
   from the real endpoint. Without these the claim "a fence produces no anchor"
   would rest on the parser under test agreeing with itself.
2. **Which anchor is a close directive.** GitHub anchors every reference, so
   adjacency in the rendered text decides, and that logic is this module's own.
3. **The call itself.** The tests never reach the network, so the seam is
   pinned twice: once by injecting the runner and asserting the exact argv and
   payload, and once by putting a `gh` of the tests' own on `PATH` and driving
   the shipped command end to end. A suite that only ever sees an injected
   callable proves nothing about what ships.
4. **The failure policy.** Unreachable, non-200 and unreadable all exit 2 with
   an `error:` line and an empty stdout, because a wrong close is worse in kind
   than a missed one -- and because a silent "no linked issues" is the exact
   failure #1549 exists to kill.

The recorded renders were produced against `robotrocketscience/aelfrice` on
2026-09-15; `python3 scripts/record_merge_train_renders.py --dry-run` re-sends
every body and reports any that GitHub now answers differently.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from merge_train_fake_gh import (
    CLI_TIMEOUT as _CLI_TIMEOUT,
    fake_gh,
    issue_anchor,
    recorded_call,
    run_cli,
)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "merge_train_linked_issues.py"
_WORKFLOW = _REPO / ".github" / "workflows" / "merge-train.yml"
_RECORDS_FILE = _REPO / "tests" / "data" / "merge_train_github_renders.json"

sys.path.insert(0, str(_REPO / "scripts"))

import merge_train_linked_issues as module  # noqa: E402

# Bound off the one module object rather than imported a second way. Several
# tests monkeypatch and read attributes through `module`, so that name has to
# exist; adding `from merge_train_linked_issues import ...` beside it imports
# the same module twice and trips CodeQL's py/import-and-import-from. Binding
# here says explicitly that every name below is an attribute of that object.
ADJACENT_RE = module.ADJACENT_RE
CROSS_REPO = module.CROSS_REPO
DECLINED_SEPARATOR = module.DECLINED_SEPARATOR
FROM_DISCUSSION = module.FROM_DISCUSSION
IN_QUOTE = module.IN_QUOTE
KEYWORDS = module.KEYWORDS
MAX_BODY_CHARACTERS = module.MAX_BODY_CHARACTERS
NOT_AN_ISSUE = module.NOT_AN_ISSUE
NOT_LINKED = module.NOT_LINKED
RENDER_LIMIT_BYTES = module.RENDER_LIMIT_BYTES
RENDER_TIMEOUT_SECONDS = module.RENDER_TIMEOUT_SECONDS
UNLABELLED_TARGET = module.UNLABELLED_TARGET
RendererUnavailable = module.RendererUnavailable
close_directives = module.close_directives
discussion_targets = module.discussion_targets
linked_issues = module.linked_issues
parse = module.parse
render_markdown = module.render_markdown

_RECORDS = json.loads(_RECORDS_FILE.read_text(encoding="utf-8"))
_CONTEXT = _RECORDS["_context"]


def _replay(name: str):
    """The recorded body, and a renderer that replays what GitHub answered.

    The renderer asserts it was handed that exact body and context, so a test
    cannot quietly pass by replaying one record against another body.
    """
    record = _RECORDS["records"][name]

    def render(body: str, repo: str) -> str:
        assert body == record["body"], f"the {name} record replayed another body"
        assert repo == _CONTEXT
        return record["html"]

    return record["body"], render


def _parse(name: str) -> tuple[list[int], list[tuple[str, str]]]:
    body, render = _replay(name)
    found, refused = parse(body, _CONTEXT, render=render)
    return found, [(r.text, r.reason) for r in refused]


# --------------------------------------------------------------------------
# What GitHub renders, and what therefore cannot close.
# --------------------------------------------------------------------------


def test_the_recorded_renders_are_githubs_and_not_this_modules() -> None:
    """The fixtures must look like the endpoint's output, not like a stand-in.

    Every replay below is only worth as much as this: the HTML carries
    GitHub's own `issue-link js-issue-link` anchor class and its `data-url`
    attribute, neither of which anything in this repository writes.
    """
    for name, record in _RECORDS["records"].items():
        assert record["body"], f"{name} recorded an empty body"
        assert 'class="issue-link js-issue-link"' in record["html"], name
        assert f'data-url="https://github.com/{_CONTEXT}/issues/' in record["html"], (
            name
        )


@pytest.mark.parametrize(
    ("record", "issue"),
    [
        ("blocks", 8),  # a fenced code block
        ("blocks", 9),  # an indented code block
        ("blocks", 11),  # an HTML comment
        ("blocks", 12),  # an inline code span
        ("template", 7),  # the indented line under `## Linked issues`
        ("template", 8),  # an indent after a setext underline
        ("template", 9),  # an indent after a thematic break
        ("template", 12),  # an indent inside a block quote
        ("template", 13),  # an indent inside a list item
    ],
)
def test_github_renders_no_anchor_for_a_keyword_it_reads_as_code(
    record: str, issue: int
) -> None:
    """Gap 2, closed by the renderer rather than by a block grammar.

    Asserted against the recorded HTML directly, not through the parser: this
    is the claim the whole change rests on, and reading it off the parser
    would be reading it off the thing under test. Each of these was a wrong
    close the line-state scanner made, and the first `template` row is this
    repository's own pull-request template, where an indented `Fixed #7`
    under a `## Linked issues` heading linked #7.
    """
    html = _RECORDS["records"][record]["html"]
    assert f"/issues/{issue}" not in html
    assert "<pre" in html or "<code" in html


@pytest.mark.parametrize(
    ("record", "issue"),
    [
        ("blocks", 1546),
        ("blocks", 1548),
        ("template", 1522),
        ("template", 1521),
    ],
)
def test_github_does_anchor_the_references_those_bodies_close(
    record: str, issue: int
) -> None:
    """The control. Without it the assertion above passes on empty HTML."""
    html = _RECORDS["records"][record]["html"]
    assert f'data-url="https://github.com/{_CONTEXT}/issues/{issue}"' in html


def test_a_body_documenting_the_syntax_closes_nothing_and_says_so() -> None:
    found, refused = _parse("blocks")
    assert found == [1546, 1548], "the prose keyword and the GH- form still link"
    assert (
        "Fixes #8",
        NOT_LINKED,
    ) in refused, "a refused keyword must never be silent"
    assert {text for text, _ in refused} == {
        "Closes #1547",
        "Fixes #8",
        "Resolves #9",
        "Closes #11",
        "Closes #12",
        "Closes octo-org/octo-repo#14",
    }


def test_the_pull_request_template_no_longer_wrong_closes() -> None:
    """#1549's fifth-round defect, on this repository's own template.

    `## Linked issues` followed by an indented `Fixed #7` linked #7 under the
    line-state scanner. GitHub renders it inside `<pre><code>`.
    """
    found, refused = _parse("template")
    assert 7 not in found
    assert ("Fixed #7", NOT_LINKED) in refused
    assert found == [1513, 1521, 1522]


@pytest.mark.parametrize("issue", [8, 9, 12, 13])
def test_every_block_type_round_five_found_is_inert_now(issue: int) -> None:
    """A setext underline, a thematic break, a quote and a list item.

    Each was a separate wrong close, and none of them costs a line of code
    here: they are inert because GitHub rendered no anchor.
    """
    found, refused = _parse("template")
    assert issue not in found
    assert any(str(issue) in text for text, _ in refused)


def test_a_table_row_and_a_heading_do_close() -> None:
    """The other half: a block GitHub renders as prose is prose.

    Both were also wrong under the line-state scanner, in the other
    direction. Adjacency is per block, so a `<td>` and an `<h2>` are blocks
    like any other.
    """
    found, _ = _parse("template")
    assert 1522 in found and 1521 in found


# --------------------------------------------------------------------------
# Which anchor is a close directive: adjacency.
# --------------------------------------------------------------------------


def test_the_module_names_exactly_githubs_nine_keywords() -> None:
    """Spelled out here rather than imported, so a narrowing edit fails.

    Asserting `KEYWORDS == KEYWORDS` would pass whatever the module said.
    """
    assert sorted(KEYWORDS) == sorted(
        (
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
    )


def _anchor(
    number: int, repo: str = _CONTEXT, text: str | None = None, *, kind: str = "issue"
) -> str:
    """One issue-link anchor shaped like GitHub's, for the logic tests."""
    return issue_anchor(number, repo, text, kind=kind)


@pytest.mark.parametrize("keyword", KEYWORDS)
@pytest.mark.parametrize("case", [str.lower, str.upper, str.capitalize])
def test_every_keyword_arms_the_anchor_after_it_in_any_case(keyword, case) -> None:
    html = f"<p>{case(keyword)} {_anchor(7)}</p>"
    assert close_directives(html, _CONTEXT) == ([7], [])


@pytest.mark.parametrize(
    ("before", "found"),
    [
        ("Closes ", [7]),
        ("Closes: ", [7]),  # GitHub's published example
        ("Closes:", [7]),  # the colon is itself the delimiter
        ("Closes : ", []),  # the colon must touch the keyword
        ("Closes::", []),  # one colon, not a run of them
        ("precloses ", []),  # the boundary at the front still holds
        ("Closes issue ", []),  # GitHub does not act on this either
        ("See ", []),  # a mention, not a directive
        ("", []),  # an anchor with nothing before it
        ("Closes\n", [7]),  # GitHub accepts any whitespace
    ],
    ids=lambda v: repr(v)[:20],
)
def test_what_counts_as_the_keyword_immediately_before_an_anchor(
    before: str, found: list[int]
) -> None:
    """The colon spacings are this module's ruling, not GitHub's.

    GitHub renders an anchor for all five colon rows -- checked in the
    `separators` record -- and publishes no close processor, so the module
    docstring rules on them under "What 'followed by a colon' means here" and
    they are pinned here rather than left to whatever the regex did.
    """
    assert close_directives(f"<p>{before}{_anchor(7)}</p>", _CONTEXT)[0] == found


def test_the_separator_rulings_hold_against_a_real_github_render() -> None:
    """The rows above, driven through the body GitHub actually rendered.

    `Closes : #4`, `Closes::#5`, `Closes#6`, `precloses #7` and
    `Closes issue #8` close nothing, and the split between them is the point.
    The first two are this module's rulings against a keyword GitHub anchored,
    so they are refusals and AC5 requires each to say so. The last three are
    forms GitHub itself does not act on, so there is no close directive in
    them to report and they are silent for the same reason a body with no
    keyword is silent.
    """
    found, refused = _parse("separators")
    assert found == [1526, 1527, 1541, 1542, 1556]
    assert refused == [
        ("Closes : #1558", DECLINED_SEPARATOR),
        ("Closes::#1528", DECLINED_SEPARATOR),
    ]


@pytest.mark.parametrize(
    ("before", "warned"),
    [
        ("Closes : ", "Closes : #7"),  # the colon must touch the keyword
        ("Closes::", "Closes::#7"),  # one colon, not a run of them
        ("Resolves :: ", "Resolves :: #7"),  # both at once
        ("Fixed:\n : ", "Fixed: : #7"),  # normalised across the line break
        ("precloses ", None),  # GitHub does not act on it either
        ("Closes issue ", None),  # nor on this
        ("See ", None),  # an ordinary mention
        ("", None),  # an anchor with nothing before it
    ],
    ids=lambda v: repr(v)[:20],
)
def test_only_a_declined_keyword_warns_and_a_mention_stays_quiet(
    before: str, warned: str | None
) -> None:
    """The line between a refusal and an absence.

    Warning about `See #7` would teach a reader to ignore the warnings, which
    is the same failure as not warning at all.
    """
    _, refused = close_directives(f"<p>{before}{_anchor(7)}</p>", _CONTEXT)
    if warned is None:
        assert refused == []
    else:
        assert [(r.text, r.reason) for r in refused] == [
            (warned, DECLINED_SEPARATOR)
        ]


def test_a_keyword_split_across_a_line_still_arms_the_anchor() -> None:
    """`Resolves\\n#9` in the recorded body; GitHub renders a `<br>`."""
    assert 1527 in _parse("separators")[0]


def test_a_line_break_does_not_break_the_run() -> None:
    """`<br>` is inline, and adds no whitespace of its own because none is needed.

    Written with no newline beside the tag, which is the shape the recorded
    render does not have: GitHub emits `Resolves<br>\\n<a ...>`, so a tool
    that read only the literal newline would pass the row above. Adjacency
    allows zero whitespace, so what has to hold here is only that `<br>` does
    not end the run -- dropping it from the inline set does end it, and the
    recorded `#9` stops linking.
    """
    assert close_directives(f"<p>Resolves<br>{_anchor(7)}</p>", _CONTEXT) == ([7], [])


def test_one_keyword_arms_one_anchor() -> None:
    """`Closes #10 #11` links #10 alone, which is GitHub's rule.

    An anchor ends the run of text, so nothing before the first anchor can
    reach the second.
    """
    assert 1523 not in _parse("separators")[0]
    assert close_directives(
        f"<p>Closes {_anchor(10)} {_anchor(11)}</p>", _CONTEXT
    ) == ([10], [])


def test_text_before_a_block_boundary_cannot_arm_an_anchor_after_it() -> None:
    """Adjacency is per block. Two paragraphs are two blocks."""
    assert close_directives(f"<p>Closes</p><p>{_anchor(7)}</p>", _CONTEXT) == ([], [])
    assert close_directives(
        f"<ul><li>Closes</li><li>{_anchor(7)}</li></ul>", _CONTEXT
    ) == ([], [])


def test_a_keyword_written_as_code_does_not_arm_the_anchor_beside_it() -> None:
    """A keyword inside a code span is not a keyword.

    `<code>` is deliberately absent from the inline set, so it breaks the run
    like a block boundary. The tie-break is the asymmetry the docstring
    states: refusing costs an open issue a human sees.
    """
    assert close_directives(
        f"<p><code>Closes</code> {_anchor(7)}</p>", _CONTEXT
    ) == ([], [])


def test_an_inline_wrapper_does_not_break_the_run() -> None:
    """Emphasis around the reference is still the same block of text."""
    assert close_directives(
        f"<p>Closes <em>{_anchor(7)}</em></p>", _CONTEXT
    ) == ([7], [])


# Every tag GitHub emits that may sit between a keyword and its reference
# without ending the run. Spelled out here rather than read from the module,
# because a list derived from `_INLINE_TAGS` loses a row exactly when a member
# is deleted -- which is the deletion these rows exist to catch. Written
# derived first, and dropping `strong` then left 216 passed.
_GITHUB_INLINE_TAGS = (
    "a", "abbr", "b", "br", "cite", "del", "em", "font", "g-emoji", "i",
    "img", "ins", "kbd", "mark", "q", "s", "small", "span", "strong",
    "sub", "sup", "time", "tt", "u",
)


def test_the_module_names_exactly_the_inline_tags_github_emits() -> None:
    """The set, against a list written out rather than imported from it."""
    assert sorted(module._INLINE_TAGS) == sorted(_GITHUB_INLINE_TAGS)


@pytest.mark.parametrize("tag", [t for t in _GITHUB_INLINE_TAGS if t != "a"])
def test_every_inline_tag_leaves_the_keyword_inside_the_run(tag: str) -> None:
    """One row per member, with the tag wrapped around the KEYWORD.

    `**Closes** #7` renders `<strong>Closes</strong> <a class="issue-link">`.
    With `strong` missing from the set, `</strong>` ends the run, the anchor is
    never armed, and the close is lost in SILENCE: no keyword was found, so no
    warning is printed either. That is the one outcome AC5 forbids, and it was
    reachable for 20 of the 24 members -- only `a`, `br` and the start tag of
    `em` were exercised anywhere, so the rest could be deleted with the suite
    still green. All 24 are real GitHub output.

    `a` is excluded deliberately rather than forgotten. An anchor ENDS the run
    -- `Closes #1 #2` links #1 alone -- so `<a>Closes</a>` arms nothing and the
    row would read backwards. What its membership decides is whether an
    issue-link anchor is opened at all: without it `handle_starttag` resets the
    run and returns before reading the URL, so nothing in the body ever closes
    and most of this file fails.
    """
    html = f"<p><{tag}>Closes</{tag}> {_anchor(7)}</p>"
    assert close_directives(html, _CONTEXT) == ([7], [])


def test_an_unknown_wrapper_breaks_the_run_rather_than_arming_it() -> None:
    """The conservative direction for HTML GitHub has not emitted yet."""
    assert close_directives(
        f"<p>Closes <some-future-element>{_anchor(7)}</some-future-element></p>",
        _CONTEXT,
    ) == ([], [])


def test_an_ordinary_link_is_not_an_issue_reference() -> None:
    """Only GitHub's `issue-link` anchor counts, and it ends the run too."""
    url = "https://example.invalid/x"
    assert close_directives(
        f'<p>Closes <a href="{url}">docs</a> {_anchor(7)}</p>', _CONTEXT
    ) == ([], [])


def test_an_unterminated_link_still_ends_the_run_before_it() -> None:
    """The case the reset on a non-issue-link `<a>` start tag is actually for.

    With a `</a>` the end-tag handler ends the run, so the reset on the start
    tag changes nothing and the row above passes either way. `html.parser`
    synthesises no close tag, so an unterminated `<a>` -- which is what this
    module would see if GitHub ever emitted malformed HTML, or if a render
    arrived truncated -- leaves the keyword in the run and arms the anchor
    that follows. Deleting the reset turns this into a close of #7.
    """
    url = "https://example.invalid/x"
    assert close_directives(
        f'<p>Closes <a href="{url}">{_anchor(7)}</p>', _CONTEXT
    ) == ([], [])


def test_the_self_closing_spelling_behaves_like_the_start_and_end_pair() -> None:
    """The self-closing spelling is the only input reaching `handle_startendtag`.

    GitHub emits `<br>`, so nothing in the recorded renders reaches that
    handler; it exists for HTML this module did not write. It must run both
    halves: `<br/>` leaves the run alone like `<br>`, an unknown element still
    ends it, and a self-closing anchor is only ever recorded by the end-tag
    half, so dropping that call loses the close entirely.
    """
    assert close_directives(f"<p>Resolves<br/>{_anchor(7)}</p>", _CONTEXT) == ([7], [])
    assert close_directives(
        f"<p>Closes <some-future-element/>{_anchor(7)}</p>", _CONTEXT
    ) == ([], [])
    self_closed = (
        '<a class="issue-link" data-hovercard-type="issue" '
        f'data-url="https://github.com/{_CONTEXT}/issues/7"/>'
    )
    assert close_directives(f"<p>Closes {self_closed}</p>", _CONTEXT) == ([7], [])


# --------------------------------------------------------------------------
# The adjacency window, which is a cost bound and must not be a ruling.
# --------------------------------------------------------------------------

# Prose words that end in one of the nine keywords. Each is a wrong close
# waiting for a window short enough to cut the letters in front of it away.
_PROSE_ENDING_IN_A_KEYWORD = (
    "prefixed",
    "prefixes",
    "affixed",
    "unresolved",
    "disclosed",
    "foreclosed",
)


@pytest.mark.parametrize("word", _PROSE_ENDING_IN_A_KEYWORD)
def test_no_window_size_turns_prose_into_a_close(
    word: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clipping the run must never invent a boundary that was not there.

    The window is a suffix of the run, so a window short enough to start
    exactly at `fixed` used to let `(?:^|\\W)` read the cut as the start of a
    block: at a six-character window `... the last one is prefixed #7` closes
    #7. Requiring a real non-word character in a clipped window makes every
    match on the window a match on the whole run, so clipping can only lose a
    close. Every window size is swept rather than the one that flips, because
    which size flips depends on the keyword -- 6 for `prefixed`, 9 for
    `unresolved`.
    """
    lead = "a long paragraph of prose whose last word is "
    html = f"<p>{lead}{word} {_anchor(7)}</p>"
    for window in range(1, 65):
        monkeypatch.setattr(module, "_TAIL", window)
        assert close_directives(html, _CONTEXT) == ([], []), (
            f"a {window}-character window turned {word!r} into a close"
        )


def test_a_real_close_survives_every_window_the_bound_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control: the sweep above would also pass if nothing ever closed."""
    lead = "a long paragraph of prose that ends in a directive. "
    html = f"<p>{lead}Resolves: {_anchor(7)}</p>"
    for window in range(12, 65):
        monkeypatch.setattr(module, "_TAIL", window)
        assert close_directives(html, _CONTEXT) == ([7], []), (
            f"a {window}-character window lost a real close"
        )


def test_a_long_block_does_not_poison_the_block_after_it() -> None:
    """The clipped flag has to come back down with the run it describes.

    A directive that opens a block starts at position 0 of a window that was
    never cut, so `^` is a real boundary there and must stay accepted. Left
    set from the previous block, the flag makes the rule demand a non-word
    character that a block-opening keyword cannot have, and the close is lost
    in silence -- the same shape of failure as one quote silencing a body.
    """
    long_block = "x" * (module._TAIL * 3)
    assert close_directives(
        f"<p>{long_block}</p><p>Closes {_anchor(7)}</p>", _CONTEXT
    ) == ([7], [])
    assert close_directives(
        f"<p>{long_block} Closes {_anchor(7)}</p>", _CONTEXT
    ) == ([7], []), "and clipping mid-block must not lose it either"


def test_the_adjacency_window_cannot_be_trimmed_silently() -> None:
    """`_TAIL` carries a comment inviting a trim, so the bound is asserted.

    The lower bound is what the accepted spellings need: the longest keyword,
    its colon, one space and the non-word character in front of it. The
    equality is a change detector on a constant whose comment calls the rest
    slack -- a trim below the bound loses real closes silently, and the sweep
    above is what stops one from inventing closes.
    """
    longest = max(len(keyword) for keyword in KEYWORDS)
    assert module._TAIL >= longest + 3
    assert module._TAIL == 64, "the adjacency window changed; re-run the sweep above"


# --------------------------------------------------------------------------
# Repository identity, which the emulating revision did not have.
# --------------------------------------------------------------------------


def test_a_link_to_another_repository_is_refused_out_loud() -> None:
    """Out of scope to *follow*; in scope to stop passing over in silence."""
    found, refused = _parse("elsewhere")
    assert refused == [
        ("Closes cli/cli#1", CROSS_REPO),
        ("Fixes cli/cli#2", CROSS_REPO),
    ]
    assert found == [1549]


def test_a_full_url_to_this_repository_now_closes_it() -> None:
    """A behaviour change, and a deliberate one.

    The emulating revision refused every URL, its own included, because it
    carried no repository identity and could not tell them apart. This one
    sends `--repo` as the render context and compares it against every
    anchor's `data-url`, so `Resolves <this repo>/issues/3` closes #3 -- which
    is what GitHub does with the same body.
    """
    assert _parse("elsewhere")[0] == [1549]


def test_the_comparison_is_case_insensitive_like_github() -> None:
    html = f"<p>Closes {_anchor(7, repo='RobotRocketScience/Aelfrice')}</p>"
    assert close_directives(html, "robotrocketscience/aelfrice") == ([7], [])


def test_a_near_miss_repository_name_is_refused_not_closed() -> None:
    """The control for the case fold: a different repo is still a different repo.

    The near-miss name is a suffix of this repository's, so a comparison that
    fell back to a prefix or substring test would close #7 here.
    """
    html = f"<p>Closes {_anchor(7, repo='robotrocketscience/aelfrice-x')}</p>"
    found, refused = close_directives(html, _CONTEXT)
    assert found == []
    assert [r.reason for r in refused] == [CROSS_REPO]


# --------------------------------------------------------------------------
# A name grammar on the anchor side is a way to abort the whole step.
# --------------------------------------------------------------------------


def test_github_anchors_a_repository_whose_name_begins_with_a_dot() -> None:
    """The premise, read off GitHub's own bytes: `.github` is a repository.

    GitHub only anchors `owner/repo#N` for a repository it can resolve, so the
    anchor in the recorded render is itself the proof that the name is legal.
    """
    html = _RECORDS["records"]["identity"]["html"]
    assert 'data-url="https://github.com/github/.github/issues/5"' in html


def test_a_legal_repository_name_is_a_refusal_and_not_an_abort() -> None:
    """A body naming another repository must still close this one's issues.

    `Closes github/.github#5` beside a reference to this repository: the name
    pattern required an alphanumeric at each end, so `_ISSUE_HREF_RE` returned
    no match for `.github`, `close_directives` raised, and the shipped CLI
    answered rc=2 with an empty stdout -- closing NOTHING for the whole body,
    the issue in its own repository included. The docstring's ruling for an
    anchor resolving elsewhere is CROSS_REPO, so that is what it has to be.
    """
    found, refused = _parse("identity")
    assert found == [1553], "the abort took this repository's own close with it"
    assert ("Closes github/.github#5", CROSS_REPO) in refused


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_the_shipped_cli_closes_its_own_issue_beside_a_dotted_name(
    tmp_path: Path,
) -> None:
    """End to end, because the defect was rc=2 and an empty shipped stdout."""
    body = _RECORDS["records"]["identity"]["body"]
    bin_dir = _fake_gh(tmp_path, _RECORDS["records"]["identity"]["html"])
    proc = _run_cli(["--repo", _CONTEXT], bin_dir=bin_dir, stdin=body)
    assert proc.returncode == 0
    assert proc.stdout.split() == ["1553"]
    assert CROSS_REPO in proc.stderr


def test_every_anchor_in_every_recorded_render_is_readable() -> None:
    """The grammar has to cover GitHub's whole output, not a sample of it.

    `close_directives` raises on a `data-url` it cannot read, and that raise
    costs every close in the body rather than one. So a URL shape the pattern
    misses is not a missed close, it is a merge that closes nothing -- which
    is why this sweeps every anchor GitHub returned for every recorded body,
    and why a re-record that brings back a new shape fails here.
    """
    seen = 0
    for name, record in _RECORDS["records"].items():
        for url in re.findall(
            r'class="issue-link[^"]*"[^>]*data-url="([^"]+)"', record["html"]
        ):
            assert module._ISSUE_HREF_RE.match(url) is not None, (name, url)
            seen += 1
    assert seen >= 20, f"the sweep only found {seen} anchors to read"


@pytest.mark.parametrize(
    ("url", "closes"),
    [
        # A legal repository name with a leading dot, which is the shape that
        # aborted the step. `github/.github` is real and common.
        ("https://github.com/github/.github/issues/5", False),
        # An owner spelled the same way. No GitHub login may begin with a dot,
        # so the renderer cannot produce this -- but reading it costs nothing
        # and refusing to read it costs every close in the body.
        ("https://github.com/.dotowner/.dotname/issues/5", False),
        # A name at GitHub's documented 100-character maximum. No length bound
        # is written into the pattern, because a bound here is another abort.
        ("https://github.com/owner/" + "n" * 100 + "/issues/5", False),
        # Percent-encoding: `aelfric%65` decodes to this repository's name and
        # is refused all the same. Nothing here decodes, so the comparison
        # fails and the answer is a missed close rather than a wrong one --
        # the asymmetry the docstring states, applied to an escape.
        ("https://github.com/robotrocketscience/aelfric%65/issues/5", False),
        # This repository, in every shape a URL may carry after the number.
        ("https://github.com/robotrocketscience/aelfrice/issues/5", True),
        ("https://github.com/robotrocketscience/aelfrice/issues/5/", True),
        ("https://github.com/robotrocketscience/aelfrice/issues/5?utm_source=x", True),
        ("https://github.com/robotrocketscience/aelfrice/issues/5#issuecomment-1", True),
        # A `/pull/` path, which the pattern must read: `href` is the
        # fallback when an anchor carries no `data-url`, and GitHub spells a
        # pull request's `href` that way. It closes here because the path is
        # not what refuses a pull request -- `data-hovercard-type` is, and
        # every anchor in this sweep carries `issue`. The refusal has its own
        # rows in `test_a_pull_request_is_refused_in_every_spelling`.
        ("https://github.com/robotrocketscience/aelfrice/pull/5", True),
    ],
    ids=lambda v: str(v)[-40:],
)
def test_no_url_a_legal_anchor_can_carry_aborts_the_step(
    url: str, closes: bool
) -> None:
    """Each shape the review asked about, decided one row at a time.

    The owner and the name are read as whole path segments, which is what
    makes this list closed rather than a sample: a path segment cannot contain
    a `/`, so every owner and name GitHub is able to put here parses, whatever
    its length or spelling. Being wider than the legal set decides nothing,
    because the segments are only ever compared against `--repo`.
    """
    html = (
        '<p>Closes <a class="issue-link js-issue-link" '
        f'data-hovercard-type="issue" data-url="{url}">#5</a></p>'
    )
    found, refused = close_directives(html, _CONTEXT)
    if closes:
        assert (found, refused) == ([5], [])
    else:
        assert found == []
        assert [r.reason for r in refused] == [CROSS_REPO]


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/whatever",
        "https://github.com/robotrocketscience/aelfrice/milestone/5",
        "https://github.com/robotrocketscience/issues/5",
    ],
)
def test_a_url_that_is_not_an_issue_reference_still_stops_the_step(url: str) -> None:
    """The control: widening the segments must not empty the raise.

    What the raise is for is an output shape GitHub does not emit today --
    another host, or a path that is not `OWNER/NAME/(issues|pull)/N`. Stopping
    is right there, because the module cannot tell whose issue it is looking
    at; stopping for a name it simply could not spell was not.
    """
    html = f'<p>Closes <a class="issue-link js-issue-link" data-url="{url}">#5</a></p>'
    with pytest.raises(RendererUnavailable):
        close_directives(html, _CONTEXT)


def test_a_discussion_in_a_dotted_repository_is_still_found_at_source() -> None:
    """On the source side the same grammar decides a close, not a diagnostic.

    `DISCUSSION_RE` is what stops a discussions URL from closing the issue
    that happens to share its number, and it is built from the same name
    fragment. A repository whose name the fragment cannot spell got no such
    protection: run against `--repo github/.github`, a body writing that
    repository's own discussion would have closed its issue 5.
    """
    body = "Closes https://github.com/github/.github/discussions/5\n"
    assert discussion_targets(body) == frozenset({("github", ".github", 5)})

    html = f"<p>Closes {_anchor(5, repo='github/.github')}</p>"
    found, refused = parse(body, "github/.github", render=lambda b, r: html)
    assert found == []
    assert [(r.text, r.reason) for r in refused] == [("Closes #5", FROM_DISCUSSION)]


def test_a_block_quote_is_refused_out_loud() -> None:
    """The one place this refuses where adjacency alone would close.

    Divergence 4 in the module docstring: a quoted review comment saying
    `Fixes #N` is a real wrong close, and the rendered tree says exactly
    which anchors are inside a `<blockquote>`, so the refusal costs no
    grammar. It is a ruling; GitHub publishes nothing either way.
    """
    _, refused = _parse("blocks")
    assert ("Closes #1547", IN_QUOTE) in refused


def test_leaving_a_quote_makes_the_next_anchor_live_again() -> None:
    """The depth counter must come back down, or one quote silences the body."""
    html = f"<blockquote><p>Closes {_anchor(7)}</p></blockquote><p>Fixes {_anchor(8)}</p>"
    found, refused = close_directives(html, _CONTEXT)
    assert found == [8]
    assert [r.reason for r in refused] == [IN_QUOTE]


# --------------------------------------------------------------------------
# The render says a reference exists. It does not say what the reference named.
# --------------------------------------------------------------------------


def test_github_rewrites_a_discussions_url_into_an_issue_anchor() -> None:
    """The premise of the refusal below, read off GitHub's own bytes.

    `.../discussions/3` comes back as an `issue-link` anchor whose `data-url`
    is `.../issues/3` and whose text is `#3` -- byte-for-byte what a plain
    `#3` produces. Nothing in the rendered document distinguishes them, which
    is why the check has to run against the source body.
    """
    html = _RECORDS["records"]["urlforms"]["html"]
    assert 'href="https://github.com/robotrocketscience/aelfrice/issues/1512"' in html
    assert 'data-url="https://github.com/robotrocketscience/aelfrice/issues/1512"' in html
    assert "discussions/1512" not in html


def test_a_discussions_url_is_refused_rather_than_closing_that_issue_number() -> None:
    """A live wrong close: discussions are numbered independently of issues.

    This repository has discussions enabled, so `Closes <a discussion URL>`
    named an object that has nothing to do with the issue of the same number
    and the train would have run `gh issue close` on it.
    """
    found, refused = _parse("urlforms")
    assert 1512 not in found
    assert ("Closes #1512", FROM_DISCUSSION) in refused


def test_the_other_rewritten_url_forms_are_ruled_on_one_by_one() -> None:
    """The audit in the module docstring, replayed against the recorded render.

    A comment fragment names the issue the comment is on, so it closes. A
    pull-request URL renders the same anchor and does not close: GitHub
    labels it `pull_request`, and a pull request is not a closing reference.
    `/pull/N/files`, `/commit/SHA` and an organisation discussion are not
    issue references at all.
    """
    found, refused = _parse("urlforms")
    assert found == [1511], "a comment fragment closes; a pull request does not"
    assert ("Closes #1504", NOT_AN_ISSUE) in refused
    assert ("Closes #1504", FROM_DISCUSSION) not in refused
    assert ("Closes cli/cli#8", FROM_DISCUSSION) in refused
    # `/pull/N/files` renders an ordinary link, so only the source scan sees it.
    assert (
        "Closes https://github.com/robotrocketscience/aelfrice/pull/1505",
        NOT_LINKED,
    ) in refused
    # A commit link and an organisation discussion are not references at all.
    assert not any("abc1234" in text for text, _ in refused)
    assert not any("orgs/robotrocketscience" in text for text, _ in refused)


def test_github_lower_cases_the_url_it_hangs_on_a_mixed_case_reference() -> None:
    """The premise of the fold, read off GitHub's own bytes.

    The body writes `RobotRocketScience/AelfRice`; `data-url` comes back
    lower-cased, and the author's capitalisation survives nowhere in the
    render. So the two sides of the discussions comparison -- one read from
    the source, one from the anchor -- meet only if both are folded.
    """
    record = _RECORDS["records"]["identity"]
    assert "RobotRocketScience/AelfRice/discussions/1551" in record["body"]
    assert (
        'data-url="https://github.com/robotrocketscience/aelfrice/issues/1551"'
        in record["html"]
    )
    assert "RobotRocketScience" not in record["html"]


def test_a_mixed_case_discussions_url_is_refused_like_a_lower_case_one() -> None:
    """The round-six wrong close, spelled the way an author actually types it.

    Only the SOURCE-side fold catches this: GitHub has already lower-cased the
    anchor, so folding the anchor again is a no-op here. That half was
    unguarded -- dropping it left the whole suite green while
    `Closes https://github.com/RobotRocketScience/AelfRice/discussions/21`
    closed the unrelated issue 21.
    """
    found, refused = _parse("identity")
    assert 1551 not in found
    assert ("Closes #1551", FROM_DISCUSSION) in refused


def test_a_mixed_case_spelling_of_this_repository_still_closes() -> None:
    """The control: folding must not turn every mixed-case reference away."""
    assert _parse("identity")[0] == [1553]


def test_the_source_scan_is_what_finds_a_discussion_not_the_render() -> None:
    """`discussion_targets` reads the body, because the render cannot say."""
    body = "Closes https://github.com/robotrocketscience/aelfrice/discussions/3\n"
    assert discussion_targets(body) == frozenset(
        {("robotrocketscience", "aelfrice", 3)}
    )
    assert discussion_targets("Closes #3\n") == frozenset()


def test_parse_is_what_wires_the_source_scan_to_the_render() -> None:
    """The edge, not the two ends.

    `close_directives` cannot find a discussion on its own and does not
    pretend to: called without the triples it closes #3 like any anchor. What
    makes the refusal real is that `parse` passes them, so this drives `parse`
    rather than asserting the helper works when handed the right argument.
    """
    body = "Closes https://github.com/robotrocketscience/aelfrice/discussions/3\n"
    html = f"<p>Closes {_anchor(3)}</p>"
    assert close_directives(html, _CONTEXT) == ([3], [])
    found, refused = parse(body, _CONTEXT, render=lambda b, r: html)
    assert found == []
    assert [(r.text, r.reason) for r in refused] == [("Closes #3", FROM_DISCUSSION)]


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_the_shipped_cli_refuses_a_discussions_url(tmp_path: Path) -> None:
    """End to end, because the defect was a number on the shipped stdout."""
    html = f"<p>Closes {_anchor(3)}</p>"
    bin_dir = _fake_gh(tmp_path, html)
    proc = _run_cli(
        ["--repo", _CONTEXT],
        bin_dir=bin_dir,
        stdin="Closes https://github.com/robotrocketscience/aelfrice/discussions/3\n",
    )
    assert proc.returncode == 0
    assert proc.stdout == "", "the train must not close an issue for a discussion"
    assert FROM_DISCUSSION in proc.stderr


def test_a_discussion_shadows_an_issue_of_the_same_number_deliberately() -> None:
    """The blunt edge of the rule, pinned so it is a decision and not a bug.

    A body that writes both spellings for one number loses the close. The
    anchors are identical, so no rule on the render can tell which of them the
    discussion produced; the module's standing asymmetry says a missed close
    is an open issue a human sees.
    """
    body = (
        "Closes #3\n\n"
        "Closes https://github.com/robotrocketscience/aelfrice/discussions/3\n"
    )
    html = f"<p>Closes {_anchor(3)}</p><p>Closes {_anchor(3)}</p>"
    found, refused = parse(body, _CONTEXT, render=lambda b, r: html)
    assert found == []
    assert [r.reason for r in refused] == [FROM_DISCUSSION, FROM_DISCUSSION]


# --------------------------------------------------------------------------
# A pull request is not a closing reference.
# --------------------------------------------------------------------------

# GitHub's `data-hovercard-type`, spelled out here rather than imported. The
# attribute name and the one value that closes are the whole rule, so reading
# them back off the module would pass for any spelling -- including one that
# never matches, which closes nothing, and one that matches everything, which
# closes pull requests again.
_KIND_ATTRIBUTE = "data-hovercard-type"
_KIND_THAT_CLOSES = "issue"


def test_the_module_reads_githubs_own_attribute_and_only_issue_closes() -> None:
    assert module._ANCHOR_KIND_ATTRIBUTE == _KIND_ATTRIBUTE
    assert module._ISSUE_KIND == _KIND_THAT_CLOSES


def test_github_labels_every_issue_link_anchor_with_what_it_names() -> None:
    """The premise, read off GitHub's own bytes rather than off the parser.

    `data-url` cannot answer this: GitHub rewrites a `/pull/N` reference into
    an `/issues/N` `data-url`, so the URL of a pull-request anchor is
    byte-for-byte an issue's. The label is the only thing that differs, it is
    on every anchor in every record, and the corpus holds both values -- which
    is what lets the rows below prove a refusal AND a close.
    """
    labels: list[str] = []
    for name, record in _RECORDS["records"].items():
        for tag in re.findall(r"<a [^>]*>", record["html"]):
            if "issue-link" not in tag:
                continue
            found = re.search(rf'{_KIND_ATTRIBUTE}="([^"]*)"', tag)
            assert found is not None, f"{name}: an anchor carried no label: {tag}"
            labels.append(found.group(1))
    assert len(labels) >= 25, f"the sweep only found {len(labels)} anchors"
    assert set(labels) == {"issue", "pull_request"}


def test_the_kinds_record_is_one_issue_and_four_pull_requests() -> None:
    """The control for the row below: the numbers really are what it claims.

    Every spelling in that body names a pull request except the first, so a
    rule that closed everything would answer five and one that closed nothing
    would answer none.
    """
    html = _RECORDS["records"]["kinds"]["html"]
    for number, label in (
        (1509, "issue"),
        (1504, "pull_request"),
        (1505, "pull_request"),
        (1506, "pull_request"),
        (1507, "pull_request"),
    ):
        anchor = re.search(rf'<a [^>]*/issues/{number}"[^>]*>', html)
        assert anchor is not None, number
        assert f'{_KIND_ATTRIBUTE}="{label}"' in anchor.group(0), number


def test_a_pull_request_is_refused_in_every_spelling() -> None:
    """GitHub links issues. `Closes #1504` on a pull request closes nothing.

    An earlier round on this branch ruled the other way for the `/pull/N` URL
    -- pull requests and issues share one number space, so the anchor names
    "the same object GitHub would close". They do share the space; that is
    what makes the reference render, not what makes it close. The train would
    have acted on it: step 7/7 asks `gh issue view N --json state`, which
    resolves a pull request and answers MERGED rather than CLOSED, so the
    idempotency skip does not fire and `gh issue close N` runs.
    """
    found, refused = _parse("kinds")
    assert found == [1509], "only the issue closes"
    assert refused == [
        ("Closes #1504", NOT_AN_ISSUE),  # `#N`
        ("Closes GH-1505", NOT_AN_ISSUE),  # `GH-N`
        ("Closes #1506", NOT_AN_ISSUE),  # a full `/pull/N` URL
        ("Closes #1507", NOT_AN_ISSUE),  # `OWNER/NAME#N`
    ]


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_the_shipped_cli_closes_nothing_for_a_pull_request(tmp_path: Path) -> None:
    """End to end, because what the workflow reads is stdout.

    Exit 0 and an empty stdout, not exit 2: nothing failed, and the body
    genuinely links no issue. The reason still has to reach the log.
    """
    record = _RECORDS["records"]["kinds"]
    bin_dir = _fake_gh(tmp_path, record["html"])
    proc = _run_cli(["--repo", _CONTEXT], bin_dir=bin_dir, stdin=record["body"])
    assert proc.returncode == 0
    assert proc.stdout.split() == ["1509"]
    assert proc.stderr.count(NOT_AN_ISSUE) == 4
    assert "1504" in proc.stderr


def test_the_label_and_not_the_url_path_is_what_refuses_a_pull_request() -> None:
    """The two are separable, and only one of them may decide.

    `data-url` is `/issues/N` for a pull request, and `href` -- the fallback
    when an anchor carries no `data-url` -- is `/pull/N` for a pull request
    and `/issues/N` for an issue. So a rule written on the path would refuse
    by accident here and miss every `#N` spelling, which is four of the five
    rows above.
    """
    pull_url = f"https://github.com/{_CONTEXT}/pull/5"
    as_issue = (
        f'<p>Closes <a class="issue-link" {_KIND_ATTRIBUTE}="issue" '
        f'data-url="{pull_url}">#5</a></p>'
    )
    as_pull = (
        f'<p>Closes <a class="issue-link" {_KIND_ATTRIBUTE}="pull_request" '
        f'data-url="https://github.com/{_CONTEXT}/issues/5">#5</a></p>'
    )
    assert close_directives(as_issue, _CONTEXT) == ([5], [])
    found, refused = close_directives(as_pull, _CONTEXT)
    assert found == []
    assert [(r.text, r.reason) for r in refused] == [("Closes #5", NOT_AN_ISSUE)]


def test_a_pull_request_in_another_repository_still_reads_as_cross_repo() -> None:
    """The order of the two refusals, pinned because both apply.

    `cli/cli#1` is a pull request AND another repository. The train would not
    close it either way, and "a link to another repository" is the sentence
    that tells a reader why -- so the repository check stays in front.
    """
    html = f"<p>Closes {_anchor(1, repo='cli/cli', kind='pull_request')}</p>"
    _, refused = close_directives(html, _CONTEXT)
    assert [r.reason for r in refused] == [CROSS_REPO]


def test_an_unlabelled_anchor_is_refused_and_not_raised() -> None:
    """The absence must not become a new way to abort the whole step.

    Raising here would cost every close in the body and print the workflow's
    `::error::`, which tells a human to close the linked issues by hand --
    an invitation to make exactly the wrong close. A refusal costs one open
    issue and one warning line, which is this module's standing asymmetry.
    """
    html = f"<p>Closes {_anchor(7, kind='')}</p><p>Fixes {_anchor(8)}</p>"
    assert _KIND_ATTRIBUTE not in html.split("</p>")[0]
    found, refused = close_directives(html, _CONTEXT)
    assert found == [8], "one unlabelled anchor must not silence the body"
    assert [(r.text, r.reason) for r in refused] == [("Closes #7", UNLABELLED_TARGET)]


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_an_unlabelled_anchor_leaves_the_cli_exiting_zero(tmp_path: Path) -> None:
    """The half of the row above that the workflow can actually see."""
    bin_dir = _fake_gh(tmp_path, f"<p>Closes {_anchor(7, kind='')}</p>")
    proc = _run_cli(["--repo", _CONTEXT], bin_dir=bin_dir, stdin="Closes #7\n")
    assert proc.returncode == 0, "an absent label must not abort the step"
    assert proc.stdout == ""
    assert UNLABELLED_TARGET in proc.stderr


def test_a_label_github_has_not_emitted_yet_names_itself() -> None:
    """One code path for every non-issue label, so the warning never lies.

    Writing "a link to a pull request" for whatever GitHub sends next would
    put a sentence in the merge-train log that is not true of the anchor it
    describes.
    """
    html = f"<p>Closes {_anchor(7, kind='discussion')}</p>"
    _, refused = close_directives(html, _CONTEXT)
    assert [r.reason for r in refused] == [
        "a reference GitHub labelled discussion, and this train closes an issue only"
    ]
    assert refused[0].reason != NOT_AN_ISSUE


def test_the_docstring_corrects_the_earlier_ruling_rather_than_editing_it_away() -> None:
    """A reversed ruling has to be readable as reversed.

    The file said a `/pull/N` URL "names the same object GitHub would close".
    A reader who found that sentence gone would have no way to know it had
    been decided against, or why.
    """
    doc = module.__doc__ or ""
    assert "\n### A pull request is not a closing reference\n" in doc, (
        "the section is cross-referenced from three bullets, so the phrase "
        "alone survives deleting the section that argues it"
    )
    assert "ruled the opposite" in doc
    assert "gh issue view" in doc, "the docstring must say what the wrong close costs"
    assert _KIND_ATTRIBUTE in doc
    assert "names the same object GitHub would close" in doc, (
        "the corrected ruling is quoted, so the correction is legible"
    )
    assert "When the attribute is absent" in doc


# --------------------------------------------------------------------------
# Nothing is silently unmatched.
# --------------------------------------------------------------------------


def test_a_reference_github_declined_to_render_is_reported() -> None:
    """`owner/repo#N` for a repository GitHub cannot resolve renders no anchor.

    There is then nothing in the document to hang a reason on, so the source
    scan reports it. It is the only refusal reason that carries a line number,
    because it is the only one derived from the body rather than the render.
    """
    body, render = _replay("blocks")
    _, refused = parse(body, _CONTEXT, render=render)
    unrendered = [r for r in refused if r.reason == NOT_LINKED]
    assert ("Closes octo-org/octo-repo#14", 17) in [
        (r.text, r.line) for r in unrendered
    ]
    assert all(r.line is not None for r in unrendered)


def test_a_number_that_was_closed_is_not_also_reported_as_unmatched() -> None:
    """A body that documents `Closes #7` in a fence and closes #7 in prose.

    The source scan is diagnostics, so an accounted-for number is not worth a
    warning; reporting it would teach a reader to ignore the warnings.
    """
    body = "Closes #7\n\n```\nCloses #7\n```\n"
    html = f"<p>Closes {_anchor(7)}</p><pre><code>Closes #7\n</code></pre>"
    found, refused = parse(body, _CONTEXT, render=lambda b, r: html)
    assert found == [7]
    assert refused == []


def test_the_source_scan_never_decides_a_close() -> None:
    """It is a flat regex with no idea of blocks; only the render decides.

    A body whose source is full of keywords and whose render carries no
    anchor closes nothing -- which is what makes it safe for it to be a
    superset in the places it is one.
    """
    body = "Closes #7 Fixes #8 Resolves #9\n"
    found, refused = parse(body, _CONTEXT, render=lambda b, r: "<p>nothing</p>")
    assert found == []
    assert {r.reason for r in refused} == {NOT_LINKED}
    assert len(refused) == 3


def test_an_empty_body_is_answered_without_calling_the_renderer() -> None:
    def explode(body: str, repo: str) -> str:
        raise AssertionError("an empty body must not cost a network call")

    assert parse("", _CONTEXT, render=explode) == ([], [])
    assert parse("   \n\n", _CONTEXT, render=explode) == ([], [])


# --------------------------------------------------------------------------
# The seam: the call this step actually makes.
# --------------------------------------------------------------------------


class _Runner:
    """A stand-in for `subprocess.run` that records how it was called."""

    def __init__(self, *, returncode: int = 0, stdout: str = "<p>ok</p>", stderr: str = ""):
        self.result = subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr=stderr
        )
        self.argv: list[str] | None = None
        self.kwargs: dict[str, object] = {}

    def __call__(self, argv: list[str], **kwargs: object):
        self.argv = argv
        self.kwargs = kwargs
        return self.result


def test_the_real_call_is_pinned_argument_by_argument() -> None:
    """What ships, not what a fake accepts.

    `mode` must be `gfm` or `#N` is literal text and nothing links; `context`
    must be the repository or `#N` resolves nowhere; the body must go on
    stdin, because a quarter-megabyte one does not fit in argv.
    """
    runner = _Runner()
    render_markdown("Closes #7", _CONTEXT, run=runner)

    assert runner.argv == ["gh", "api", "--method", "POST", "/markdown", "--input", "-"]
    assert json.loads(runner.kwargs["input"]) == {
        "mode": "gfm",
        "context": _CONTEXT,
        "text": "Closes #7",
    }
    assert runner.kwargs["capture_output"] is True
    assert runner.kwargs["text"] is True
    assert runner.kwargs["check"] is False
    assert runner.kwargs["timeout"] == RENDER_TIMEOUT_SECONDS
    assert runner.kwargs["encoding"] == "utf-8", (
        "the payload is no longer pure ASCII, so the encoding of stdin cannot "
        "be left to the runner's locale"
    )


def test_the_whole_body_is_sent_however_long_it_is() -> None:
    """#1541's property, restated for the renderer.

    The cap that caused #1541 sat between the body and the matcher. The
    matcher is now GitHub, so the property is that the whole body reaches it.
    """
    body = "x" * 200_000 + "\nCloses #4242."
    runner = _Runner()
    render_markdown(body, _CONTEXT, run=runner)
    assert json.loads(runner.kwargs["input"])["text"] == body


def test_a_non_ascii_body_is_sent_as_itself_and_not_as_escapes() -> None:
    """`json.dumps` escapes non-ASCII by default, and the renderer counts bytes."""
    body = "Closes #7 \U0001F600"
    runner = _Runner()
    render_markdown(body, _CONTEXT, run=runner)
    payload = runner.kwargs["input"]
    assert "\\ud83d" not in payload, "the emoji was escaped into six-fold ASCII"
    assert "\U0001F600" in payload
    assert json.loads(payload)["text"] == body


def test_the_largest_body_github_accepts_fits_inside_the_renderers_cap() -> None:
    """The reintroduced input bound, bracketed from both sides.

    `POST /markdown` refuses a request over 400 KB, which is a cap on a step
    whose defect was a cap. Escaped, a body of 65,536 astral characters is a
    786,501-byte request and GitHub answers HTTP 403; unescaped it is 262,213
    bytes and renders. The relation is what makes the cap unreachable from any
    body GitHub would accept, so it is the relation that is pinned.

    Both halves of the relation are asserted, not one. `sent < LIMIT` alone
    lets the constant be widened to 4 MB with the suite green, and the
    docstring beside it -- which is what a reader trusts -- would then be
    false. The escaped figure exceeding the cap is the other half, and it is
    also what makes `ensure_ascii=False` load-bearing rather than tidy.
    """
    body = "\U0001F600" * MAX_BODY_CHARACTERS
    runner = _Runner()
    render_markdown(body, _CONTEXT, run=runner)
    sent = len(runner.kwargs["input"].encode("utf-8"))
    escaped = len(
        json.dumps({"mode": "gfm", "context": _CONTEXT, "text": body}).encode("utf-8")
    )

    assert escaped == 786_501
    assert escaped > RENDER_LIMIT_BYTES, (
        "the cap has to be reachable by the escaping this module stopped "
        "doing, or `ensure_ascii=False` guards nothing"
    )
    assert sent == 262_213
    assert sent < RENDER_LIMIT_BYTES


def test_the_recorded_caps_are_the_figures_the_docstring_publishes() -> None:
    """Two constants no production path reads, kept honest by the prose.

    Neither bounds anything this module does -- deliberately, because an input
    bound was #1541's whole defect -- so they are assertions ABOUT GitHub, and
    what a reader trusts is the docstring. Read each figure back out of that
    prose and compare, rather than reading the constant against its own
    definition, which would pass for any value.
    """
    doc = module.__doc__ or ""
    limit = re.search(r"refuses a request above ([\d,]+) KB", doc)
    assert limit is not None, "the docstring no longer publishes the render cap"
    assert RENDER_LIMIT_BYTES == int(limit.group(1).replace(",", "")) * 1024

    body_cap = re.search(
        r"caps an issue or pull-request body at ([\d,]+) characters", doc
    )
    assert body_cap is not None, "the docstring no longer publishes the body cap"
    assert MAX_BODY_CHARACTERS == int(body_cap.group(1).replace(",", ""))


def test_the_module_applies_no_size_bound_of_its_own() -> None:
    """The constants record GitHub's caps; they must not become this tool's.

    #1541 was an input bound on this step, so its fix cannot be a different
    input bound. A body larger than `RENDER_LIMIT_BYTES` is handed over whole
    and the refusal is left to GitHub, which is loud about it -- see
    `test_a_render_refused_for_size_closes_nothing_loudly`.
    """
    body = "x" * (RENDER_LIMIT_BYTES + 1)
    runner = _Runner()
    render_markdown(body, _CONTEXT, run=runner)
    assert json.loads(runner.kwargs["input"])["text"] == body
    assert len(runner.kwargs["input"].encode("utf-8")) > RENDER_LIMIT_BYTES


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_a_render_refused_for_size_closes_nothing_loudly(tmp_path: Path) -> None:
    """The cap is nothing like `head -c 8192`: `gh` exits non-zero.

    A silent truncation is what #1541 was about. A refusal that reaches the
    failure policy costs one hand-run of `gh issue close` and says so in the
    step log, which is the safe side of the same question.
    """
    bin_dir = _fake_gh(
        tmp_path,
        "",
        returncode=1,
        stderr=(
            "gh: This API renders Markdown text up to 400 KB in size. The "
            "requested text is too large to render via the API. (HTTP 403)\n"
        ),
    )
    proc = _run_cli(["--repo", _CONTEXT], bin_dir=bin_dir, stdin="Closes #7\n")
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "400 KB" in proc.stderr
    assert "nothing was closed" in proc.stderr


def test_the_render_seam_is_late_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """Written as `render=render_markdown` the seam would be a decoration.

    A default argument is evaluated once, at import, so the signature form
    binds the real renderer before any test can replace it: replacing the
    module attribute would leave the network call in place and the suite
    would look injected while reaching GitHub on every run. Caught exactly
    that way -- this test failed against the live endpoint before the default
    moved into the body.
    """
    monkeypatch.setattr(
        "merge_train_linked_issues.render_markdown",
        lambda body, repo: f"<p>Closes {issue_anchor(7, repo)}</p>",
    )
    assert parse("Closes #7", _CONTEXT)[0] == [7]
    assert linked_issues("Closes #7", _CONTEXT) == [7]


def test_the_runner_seam_is_late_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same trap one level down, where the subprocess is spawned."""
    calls: list[list[str]] = []

    def record(argv: list[str], **kwargs: object):
        calls.append(argv)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout="<p>ok</p>", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", record)
    render_markdown("Closes #7", _CONTEXT)
    assert calls and calls[0][0] == "gh"


def _fake_gh(tmp_path: Path, body: str, *, returncode: int = 0, stderr: str = "") -> Path:
    """A `gh` of our own on `PATH`, so the shipped command runs for real.

    It records the argv and stdin it was given, which is how the end-to-end
    tests below check that `subprocess.run` was handed what the unit test
    above pins -- an injected callable alone could not tell.
    """
    return fake_gh(tmp_path, body, returncode=returncode, stderr=stderr)


def _run_cli(
    args: list[str], *, bin_dir: Path, stdin: str = ""
) -> subprocess.CompletedProcess[str]:
    """The CLI, with `PATH` holding only our `gh`, so nothing reaches GitHub."""
    return run_cli(_SCRIPT, args, bin_dir=bin_dir, stdin=stdin)


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_the_shipped_command_runs_gh_with_the_pinned_arguments(
    tmp_path: Path,
) -> None:
    """End to end through the real `subprocess.run`, with no network.

    The unit test above pins the argv against an injected runner; this one
    proves the shipped default runner is `subprocess.run` and that it invokes
    `gh` -- the two together are what make the seam tested rather than only
    the fake.
    """
    html = f"<p>Closes {_anchor(7)}</p>"
    bin_dir = _fake_gh(tmp_path, html)
    proc = _run_cli(["--repo", _CONTEXT], bin_dir=bin_dir, stdin="Closes #7\n")

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["7"]

    call = recorded_call(tmp_path)
    assert call["argv"] == ["api", "--method", "POST", "/markdown", "--input", "-"]
    assert json.loads(call["stdin"]) == {
        "mode": "gfm",
        "context": _CONTEXT,
        "text": "Closes #7\n",
    }


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_the_repo_flag_is_what_reaches_github_as_the_context(tmp_path: Path) -> None:
    bin_dir = _fake_gh(tmp_path, "<p>nothing</p>")
    _run_cli(["--repo", "octo-org/octo-repo"], bin_dir=bin_dir, stdin="hello\n")
    call = recorded_call(tmp_path)
    assert json.loads(call["stdin"])["context"] == "octo-org/octo-repo"


# --------------------------------------------------------------------------
# The failure policy: loud, and closing nothing.
# --------------------------------------------------------------------------


def test_a_non_zero_renderer_exit_raises_rather_than_returning_nothing() -> None:
    runner = _Runner(returncode=1, stdout="", stderr="HTTP 503: unavailable")
    with pytest.raises(RendererUnavailable) as exc:
        render_markdown("Closes #7", _CONTEXT, run=runner)
    assert "503" in str(exc.value), "the failure must name what GitHub said"


def test_an_empty_render_is_a_render_and_not_a_failure() -> None:
    """A 200 carrying nothing is GitHub's answer for a body with nothing in it.

    This raised. `<!-- Closes #1549 -->` renders as the empty string with `gh`
    exiting 0, so the raise made `main` return 2 and the workflow print
    `merge-train could not determine the linked issues ... Close them by
    hand.` -- an invitation to make exactly the wrong close, for a body whose
    GitHub answer is "nothing is linked".
    """
    runner = _Runner(stdout="")
    assert render_markdown("<!-- Closes #1549 -->", _CONTEXT, run=runner) == ""


def test_an_empty_render_is_said_out_loud_rather_than_passed_over(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The half of the old raise that survives: the log still tells them apart.

    "Rendered nothing" and "rendered a document with no directive" are
    different answers, and only one of them means the body had nothing in it.
    """
    render_markdown("<!-- Closes #1549 -->", _CONTEXT, run=_Runner(stdout=""))
    empty = capsys.readouterr().err
    render_markdown("prose", _CONTEXT, run=_Runner(stdout="<p>prose</p>"))
    rendered = capsys.readouterr().err

    assert "empty document" in empty
    assert str(len("<!-- Closes #1549 -->")) in empty
    assert rendered == ""


def test_a_comment_only_body_closes_nothing_and_is_still_not_silent() -> None:
    """AC5 is what the raise was defending, and the source scan keeps it.

    The keyword inside the comment is reported as NOT_LINKED, which is true
    of it: GitHub renders no anchor there, so a merge commit would not close
    it either. That report is what makes exit 0 the right answer rather than
    a silence.
    """
    body = "<!-- Closes #1549 -->\n"
    found, refused = parse(body, _CONTEXT, render=lambda b, r: "")
    assert found == []
    assert [(r.text, r.reason, r.line) for r in refused] == [
        ("Closes #1549", NOT_LINKED, 1)
    ]


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_the_shipped_cli_does_not_abort_on_a_comment_only_body(
    tmp_path: Path,
) -> None:
    """End to end, because what the workflow reads is the exit code.

    A template stripped to its guidance comments is a realistic body, and
    exit 2 on it puts an `::error::` in the step log telling a human to close
    the linked issues by hand.
    """
    bin_dir = _fake_gh(tmp_path, "")
    proc = _run_cli(["--repo", _CONTEXT], bin_dir=bin_dir, stdin="<!-- Closes #7 -->\n")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert "empty document" in proc.stderr
    assert NOT_LINKED in proc.stderr


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_a_renderer_that_answers_nothing_reports_every_keyword_it_lost(
    tmp_path: Path,
) -> None:
    """The case the old raise was written for, and what replaces it.

    A `gh` that exits 0 having printed nothing is byte-for-byte a
    comment-only body's answer, so no rule on the transport can tell them
    apart. What the step does either way is close nothing and name every
    keyword in the body -- a missed close a human sees, which is this
    module's standing asymmetry, rather than an abort that asks for a
    hand-close.
    """
    bin_dir = _fake_gh(tmp_path, "")
    proc = _run_cli(
        ["--repo", _CONTEXT], bin_dir=bin_dir, stdin="Closes #7\n\nFixes #8\n"
    )
    assert proc.returncode == 0
    assert proc.stdout == ""
    assert "Closes #7" in proc.stderr
    assert "Fixes #8" in proc.stderr
    assert proc.stderr.count(NOT_LINKED) == 2


def test_the_docstring_rules_on_the_empty_document() -> None:
    """A reversed raise has to be readable as reversed, and argued.

    The distinction the review asked for -- a successful empty render against
    an unusable answer -- cannot be drawn on the exit-0 side, and the file has
    to say that rather than leave the raise's disappearance unexplained.
    """
    doc = module.__doc__ or ""
    assert "\n### An empty document is an answer\n" in doc, (
        "the failure-policy section points at this one by name, so the phrase "
        "alone survives deleting the section that argues it"
    )
    assert "Close them by hand" in doc
    for construct in ("comment", "link\nreference definition", "footnote"):
        assert construct in doc, f"the docstring does not name the {construct} case"
    assert "still raise" in doc


def test_a_missing_gh_raises_rather_than_closing_nothing_quietly() -> None:
    def absent(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "gh")

    with pytest.raises(RendererUnavailable) as exc:
        render_markdown("Closes #7", _CONTEXT, run=absent)
    assert "gh" in str(exc.value)


def test_a_renderer_timeout_raises() -> None:
    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=RENDER_TIMEOUT_SECONDS)

    with pytest.raises(RendererUnavailable) as exc:
        render_markdown("Closes #7", _CONTEXT, run=slow)
    assert str(RENDER_TIMEOUT_SECONDS) in str(exc.value)


# The worst of 17 live renders measured against GitHub on 2026-09-15, over
# three body sizes -- one line, the 264,096-character body the docstring
# builds, and the 262,213-byte astral payload. The slowest was the largest
# body. The command is in the comment on `RENDER_TIMEOUT_SECONDS`.
_WORST_MEASURED_RENDER_SECONDS = 1.03


def _job_timeout_seconds() -> int:
    """The merge-train job's own bound, read from the workflow, not written."""
    match = re.search(
        r"^\s*timeout-minutes:\s*(\d+)\s*$",
        _WORKFLOW.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None, "merge-train.yml no longer bounds the job"
    return int(match.group(1)) * 60


def test_the_render_timeout_cannot_be_trimmed_silently() -> None:
    """Both of the older assertions read the value back from the constant.

    `kwargs["timeout"] == RENDER_TIMEOUT_SECONDS` and
    `str(RENDER_TIMEOUT_SECONDS) in str(exc)` hold for any value at all, so 30
    could become 1 with the suite green -- and then every render slower than a
    second aborts the close of every issue in the body, on exactly the loaded
    runner where a render is slow. Bracketed here from two sources that are
    not this constant: the measured worst render below it, and the workflow's
    own job bound above it.
    """
    assert RENDER_TIMEOUT_SECONDS >= 10 * _WORST_MEASURED_RENDER_SECONDS
    assert RENDER_TIMEOUT_SECONDS <= _job_timeout_seconds() / 10
    assert RENDER_TIMEOUT_SECONDS == 30, (
        "the render timeout changed; re-measure it against the live endpoint"
    )


def test_an_anchor_with_an_unreadable_url_raises_rather_than_being_skipped() -> None:
    """If GitHub's output shape changes, the step must stop, not miss closes."""
    html = (
        '<p>Closes <a class="issue-link js-issue-link" '
        'data-url="https://example.invalid/whatever">#7</a></p>'
    )
    with pytest.raises(RendererUnavailable):
        close_directives(html, _CONTEXT)


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_the_cli_exits_two_and_prints_nothing_when_the_renderer_fails(
    tmp_path: Path,
) -> None:
    """Exit 2, an `error:` line, and an empty stdout -- the ruled policy.

    Closing nothing is recoverable by a human reading the step log. Closing
    the wrong issue is not, and printing nothing on a failure while exiting 0
    would be the silent "no linked issues" #1549 removed.
    """
    bin_dir = _fake_gh(tmp_path, "", returncode=1, stderr="HTTP 503\n")
    proc = _run_cli(["--repo", _CONTEXT], bin_dir=bin_dir, stdin="Closes #7\n")

    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "error:" in proc.stderr
    assert "nothing was closed" in proc.stderr


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_a_renderer_failure_does_not_look_like_a_body_with_no_keyword(
    tmp_path: Path,
) -> None:
    """The two outcomes must differ in the exit code *and* in the log."""
    ok = _fake_gh(tmp_path / "ok", "<p>nothing here</p>")
    broken = _fake_gh(tmp_path / "broken", "", returncode=1, stderr="boom\n")

    absent = _run_cli(["--repo", _CONTEXT], bin_dir=ok, stdin="no trailer here\n")
    failed = _run_cli(["--repo", _CONTEXT], bin_dir=broken, stdin="no trailer here\n")

    assert (absent.returncode, absent.stdout, absent.stderr) == (0, "", "")
    assert failed.returncode == 2
    assert failed.stderr != ""


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_the_cli_refuses_to_guess_the_repository(tmp_path: Path) -> None:
    """Without a context `#N` resolves nowhere, so guessing one is a wrong close."""
    bin_dir = _fake_gh(tmp_path, "<p>nothing</p>")
    proc = _run_cli([], bin_dir=bin_dir, stdin="Closes #7\n")
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "--repo" in proc.stderr


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_the_repository_may_come_from_the_workflow_environment(
    tmp_path: Path,
) -> None:
    bin_dir = _fake_gh(tmp_path, f"<p>Closes {_anchor(7)}</p>")
    proc = run_cli(
        _SCRIPT,
        [],
        bin_dir=bin_dir,
        stdin="Closes #7\n",
        env={"GITHUB_REPOSITORY": _CONTEXT},
    )
    assert proc.returncode == 0
    assert proc.stdout.split() == ["7"]


# --------------------------------------------------------------------------
# AC5 at the step's own boundary.
# --------------------------------------------------------------------------


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_a_refused_keyword_is_not_silent(tmp_path: Path) -> None:
    """The defect #1549 calls the one that matters.

    A body whose only keyword the tool refused used to produce exactly what a
    body with no keyword produces: empty stdout and empty stderr.
    """
    bin_dir = _fake_gh(tmp_path, "<pre><code>Closes #7\n</code></pre>")
    refused = _run_cli(["--repo", _CONTEXT], bin_dir=bin_dir, stdin="```\nCloses #7\n```\n")

    quiet = _fake_gh(tmp_path / "quiet", "<p>no trailer here</p>")
    absent = _run_cli(["--repo", _CONTEXT], bin_dir=quiet, stdin="no trailer here\n")

    assert absent.returncode == refused.returncode == 0
    assert absent.stdout == refused.stdout == ""
    assert absent.stderr == "", "an absence has nothing to report"
    assert "Closes #7" in refused.stderr
    assert NOT_LINKED in refused.stderr


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_rejections_never_reach_stdout(tmp_path: Path) -> None:
    """stdout is read into a shell variable and split on whitespace."""
    html = f"<pre><code>Closes #7\n</code></pre><p>Fixes {_anchor(8)}</p>"
    bin_dir = _fake_gh(tmp_path, html)
    proc = _run_cli(
        ["--repo", _CONTEXT],
        bin_dir=bin_dir,
        stdin="```\nCloses #7\n```\n\nFixes #8\n",
    )
    assert proc.returncode == 0
    assert proc.stdout.split() == ["8"]
    assert "#7" in proc.stderr


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_every_rejection_gets_its_own_line(tmp_path: Path) -> None:
    bin_dir = _fake_gh(tmp_path, "<pre><code>x\n</code></pre>")
    proc = _run_cli(
        ["--repo", _CONTEXT],
        bin_dir=bin_dir,
        stdin="```\nCloses #7\nFixes #8\nResolves #9\n```\n",
    )
    lines = [ln for ln in proc.stderr.splitlines() if ln.startswith("warning:")]
    assert len(lines) == 3
    assert proc.stdout == ""


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_dry_run_still_reports_rejections(tmp_path: Path) -> None:
    html = f"<pre><code>Closes #7\n</code></pre><p>Fixes {_anchor(8)}</p>"
    bin_dir = _fake_gh(tmp_path, html)
    proc = _run_cli(
        ["--repo", _CONTEXT, "--dry-run"],
        bin_dir=bin_dir,
        stdin="```\nCloses #7\n```\n\nFixes #8\n",
    )
    assert proc.stdout.strip() == "would close #8"
    assert "Closes #7" in proc.stderr


def test_a_rejection_reads_as_one_normalised_log_line() -> None:
    """A step log is read a line at a time, and a match can span a newline."""
    body = "```\nResolves\n#7\n```\n"
    _, refused = parse(body, _CONTEXT, render=lambda b, r: "<pre><code>x</code></pre>")
    assert [r.text for r in refused] == ["Resolves #7"]
    assert "\n" not in refused[0].message()
    assert refused[0].message() == (
        f'warning: ignored "Resolves #7" on line 2: {NOT_LINKED}.'
    )


def test_a_rejection_without_a_source_line_still_names_itself() -> None:
    """A refusal read off the render has no line, and must not print `line None`."""
    _, refused = close_directives(
        f"<blockquote><p>Closes {_anchor(7)}</p></blockquote>", _CONTEXT
    )
    assert refused[0].message() == f'warning: ignored "Closes #7": {IN_QUOTE}.'
    assert "None" not in refused[0].message()


# --------------------------------------------------------------------------
# The workflow must invoke it correctly and surface its failures.
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
    r"""The merge-train lines that run the tool, comments dropped.

    The invocation is wrapped over two lines with a `\` continuation, so a
    redirect that would swallow stderr could sit on either of them.
    """
    lines = _live_lines()
    for i, line in enumerate(lines):
        if "merge_train_linked_issues.py" in line:
            return "\n".join(lines[i : i + 3])
    raise AssertionError("merge-train.yml no longer runs the tool")


def test_the_workflow_does_not_swallow_the_tools_stderr() -> None:
    """Command substitution captures stdout only, so stderr reaches the log.

    A `2>/dev/null` or a `2>&1` on this call would undo #1549 without touching
    the tool: the first discards the diagnostics, the second folds them into
    the issue-number list the shell then loops over.
    """
    call = _script_invocation()
    assert "2>/dev/null" not in call
    assert "2>&1" not in call
    assert "2>" not in call


def test_that_assertion_is_not_vacuous() -> None:
    """Other commands in this workflow do redirect stderr, so the check is narrow.

    Read the same comment-filtered lines `_script_invocation` reads, not the
    raw file: the step comment names both spellings, so a whole-file search
    reports a redirect that no command runs.
    """
    live = "\n".join(_live_lines())
    assert "2>/dev/null" in live
    assert "2>&1" in live


def test_the_workflow_passes_the_repository_to_the_tool() -> None:
    """Without it the tool exits 2 and the train closes nothing, every merge."""
    assert "--repo" in _script_invocation()


def test_the_workflow_surfaces_a_failed_determination_instead_of_swallowing_it() -> None:
    """`|| true` on this call would restore the silence #1549 removed.

    A non-zero exit means the tool could not decide; the step must say so
    where a human will see it rather than report the same "no linked issues"
    a body with no keyword reports.
    """
    live = "\n".join(_live_lines())
    assert "merge_train_linked_issues.py \\\n              --repo" in live
    assert "::error::merge-train could not determine the linked issues" in live
    assert "|| true)" not in _script_invocation()


def test_the_workflow_log_tells_undetermined_apart_from_no_keyword() -> None:
    """Three outcomes, three messages, or the annotation stands alone.

    Collapsing the failure branch back into the empty one would put
    `no linked issues parsed from PR body` under an `::error::` -- the same
    sentence a body with no keyword produces, which is the equivalence #1549
    exists to break. Driving the extracted fragment under `set -euo pipefail`
    with a stub tool prints `linked issues UNDETERMINED` for exit 2, `no
    linked issues parsed from PR body` for an empty stdout, and the close
    loop otherwise.
    """
    live = "\n".join(_live_lines())
    assert "linked issues UNDETERMINED" in live
    assert "no linked issues parsed from PR body" in live
    assert live.index("UNDETERMINED") < live.index("no linked issues parsed")


def test_the_workflow_no_longer_claims_to_emulate_markdown() -> None:
    """The step comment described the scanner this change deleted."""
    whole = _WORKFLOW.read_text(encoding="utf-8")
    assert "left for a decision" not in whole
    assert "ignores one written inside a" not in whole


# --------------------------------------------------------------------------
# The module must state what it decided.
# --------------------------------------------------------------------------


def test_the_docstring_states_the_ruling_and_the_deleted_emulation() -> None:
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert "ask GitHub rather than emulate it" in doc
    assert "gh api --method POST /markdown" in doc


def test_the_docstring_settles_the_failure_policy() -> None:
    """AC5 forbids a silent failure, so the choice has to be argued in the file."""
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert "Failure policy" in doc
    for rejected_alternative in ("Falling back to a local parse", "no links"):
        assert rejected_alternative in doc
    assert "RendererUnavailable" in doc


def test_the_docstring_states_the_injection_seam() -> None:
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert "This is the seam" in doc


def test_the_docstring_enumerates_every_documented_reference_form() -> None:
    """A form ruled on in review and left out of the file is not ruled on."""
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert "Every documented reference form is acted on or refused out loud" in doc
    for form in ("`#N`", "`GH-N`", "`OWNER/REPOSITORY#N`", "full issue"):
        assert form in doc, f"the docstring does not rule on {form}"


def test_the_docstring_audits_what_github_rewrites_into_an_issue_anchor() -> None:
    """A URL shape ruled on in review and left out of the file is not ruled on.

    The audit is the finding, not the discussions fix alone: the renderer
    normalises several shapes into one anchor, and each had to be decided.
    """
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert "The render is not an oracle for which object a reference names" in doc
    for shape in (
        "`/issues/N`",
        "`/pull/N`",
        "`/discussions/N`",
        "`/pull/N/files`",
        "`/commit/SHA`",
        "issuecomment",
        "orgs/ORG/discussions",
    ):
        assert shape in doc, f"the docstring does not rule on {shape}"


def test_the_docstring_rules_on_what_an_unreadable_url_costs() -> None:
    """A raise that closes nothing for the whole body is a ruling, not a detail.

    The pattern reading GitHub's own URLs is the one place where being too
    strict aborts the step, so what it accepts has to be argued in the file
    and each shape decided. Read off `__doc__`, because the same words appear
    in the comment on `_ISSUE_HREF_RE` and a whole-file search would pass on a
    docstring that no longer says any of it.
    """
    doc = module.__doc__ or ""
    assert "Reading GitHub's own URL must not be able to abort the step" in doc
    for shape in (
        "leading dot",
        "100-character maximum",
        "Percent-encoding",
        "trailing slash",
    ):
        assert shape in doc, f"the docstring does not rule on {shape}"
    assert "creating-a-new-repository" in doc, (
        "the name grammar is GitHub's, so the file must cite where GitHub "
        "writes it rather than assert it"
    )


def test_the_docstring_states_where_the_case_fold_happens() -> None:
    """Folding in two places is what left the source half unguarded."""
    doc = module.__doc__ or ""
    assert "One case fold, in one place" in doc
    assert "_identity" in doc


def test_the_docstring_states_the_renderers_own_cap() -> None:
    """A reintroduced input bound on the step whose defect was an input bound.

    #1541's framing means a cap cannot be left implicit: it has to be named,
    measured, tied to the failure policy, and shown to be unreachable from a
    body GitHub would accept.

    Read off `__doc__` rather than the file: the same words appear in the
    comments on `RENDER_LIMIT_BYTES` and `MAX_BODY_CHARACTERS`, so a whole-file
    search passes on a docstring that no longer says any of it.
    """
    doc = module.__doc__ or ""
    assert "The renderer has a cap of its own, and it must be said out loud" in doc
    assert "400 KB" in doc
    assert "ensure_ascii" in doc
    assert "65,536 characters" in doc
    assert "UNVERIFIED" in doc, "the body limit was not measured here and must say so"


def test_the_docstrings_big_body_figure_matches_the_command_it_publishes() -> None:
    """The command in the file must produce the number beside it.

    The figure published here was 264,036 for a command that builds a
    108,036-character body. Re-derived rather than re-copied.
    """
    doc = _SCRIPT.read_text(encoding="utf-8")
    rebuilt = "\n".join(
        ["prose line, and more of it"] * 9780
        + ["Closes #11.", "Fixes #22.", "Resolves #33."]
    )
    assert len(rebuilt) == 264_096
    assert "* 9780" in doc, "the docstring no longer builds the body it measured"
    assert "264,096-character body" in doc
    assert "264,036" not in doc


def test_the_docstring_states_what_still_diverges_from_github() -> None:
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert "Divergences that REMAIN, deliberately" in doc
    for remaining in ("Commit messages", "Cross-repository", "target branch", "block quote"):
        assert remaining in doc


def test_the_docstring_cites_githubs_keyword_page_as_one_url() -> None:
    """Two independent substring checks pass on a host in one sentence and a
    path in another, which is not a citation a reader can follow."""
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert re.search(
        r"https://docs\.github\.com/\S*linking-a-pull-request-to-an-issue", doc
    ), "the docstring does not cite GitHub's closing-keyword page by URL"


def test_the_keyword_alternation_is_built_from_the_nine() -> None:
    """The regex that decides a close must carry every keyword, not three."""
    for keyword in KEYWORDS:
        assert ADJACENT_RE.search(f"{keyword} ") is not None
    assert ADJACENT_RE.search("mentions ") is None
