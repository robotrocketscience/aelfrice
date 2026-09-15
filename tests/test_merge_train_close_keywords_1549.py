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

from merge_train_linked_issues import (  # noqa: E402
    ADJACENT_RE,
    CROSS_REPO,
    IN_QUOTE,
    KEYWORDS,
    NOT_LINKED,
    RENDER_TIMEOUT_SECONDS,
    RendererUnavailable,
    close_directives,
    linked_issues,
    parse,
    render_markdown,
)

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
    [("blocks", 7), ("blocks", 13), ("template", 10), ("template", 11)],
)
def test_github_does_anchor_the_references_those_bodies_close(
    record: str, issue: int
) -> None:
    """The control. Without it the assertion above passes on empty HTML."""
    html = _RECORDS["records"][record]["html"]
    assert f'data-url="https://github.com/{_CONTEXT}/issues/{issue}"' in html


def test_a_body_documenting_the_syntax_closes_nothing_and_says_so() -> None:
    found, refused = _parse("blocks")
    assert found == [7, 13], "the prose keyword and the GH- form still link"
    assert (
        "Fixes #8",
        NOT_LINKED,
    ) in refused, "a refused keyword must never be silent"
    assert {text for text, _ in refused} == {
        "Closes #10",
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
    assert found == [10, 11, 15]


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
    assert 10 in found and 11 in found


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


def _anchor(number: int, repo: str = _CONTEXT, text: str | None = None) -> str:
    """One issue-link anchor shaped like GitHub's, for the logic tests."""
    return issue_anchor(number, repo, text)


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
    `Closes issue #8` close nothing. The first two are rulings; the last
    three are forms GitHub itself does not act on, so refusing them is
    fidelity rather than a decision, and each is silent for the same reason a
    body with no keyword is silent -- there is no close directive in it.
    """
    found, refused = _parse("separators")
    assert found == [1, 2, 3, 9, 10]
    assert refused == []


def test_a_keyword_split_across_a_line_still_arms_the_anchor() -> None:
    """`Resolves\\n#9` in the recorded body; GitHub renders a `<br>`."""
    assert 9 in _parse("separators")[0]


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
    assert 11 not in _parse("separators")[0]
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
    assert found == [3]


def test_a_full_url_to_this_repository_now_closes_it() -> None:
    """A behaviour change, and a deliberate one.

    The emulating revision refused every URL, its own included, because it
    carried no repository identity and could not tell them apart. This one
    sends `--repo` as the render context and compares it against every
    anchor's `data-url`, so `Resolves <this repo>/issues/3` closes #3 -- which
    is what GitHub does with the same body.
    """
    assert _parse("elsewhere")[0] == [3]


def test_the_comparison_is_case_insensitive_like_github() -> None:
    html = f"<p>Closes {_anchor(7, repo='RobotRocketScience/Aelfrice')}</p>"
    assert close_directives(html, "robotrocketscience/aelfrice") == ([7], [])


def test_a_near_miss_repository_name_is_refused_not_closed() -> None:
    """The control for the case fold: a different repo is still a different repo."""
    html = f"<p>Closes {_anchor(7, repo='robotrocketscience/aelfrice-lab')}</p>"
    found, refused = close_directives(html, _CONTEXT)
    assert found == []
    assert [r.reason for r in refused] == [CROSS_REPO]


def test_a_block_quote_is_refused_out_loud() -> None:
    """The one place this refuses where adjacency alone would close.

    Divergence 4 in the module docstring: a quoted review comment saying
    `Fixes #N` is a real wrong close, and the rendered tree says exactly
    which anchors are inside a `<blockquote>`, so the refusal costs no
    grammar. It is a ruling; GitHub publishes nothing either way.
    """
    _, refused = _parse("blocks")
    assert ("Closes #10", IN_QUOTE) in refused


def test_leaving_a_quote_makes_the_next_anchor_live_again() -> None:
    """The depth counter must come back down, or one quote silences the body."""
    html = f"<blockquote><p>Closes {_anchor(7)}</p></blockquote><p>Fixes {_anchor(8)}</p>"
    found, refused = close_directives(html, _CONTEXT)
    assert found == [8]
    assert [r.reason for r in refused] == [IN_QUOTE]


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


def test_the_whole_body_is_sent_however_long_it_is() -> None:
    """#1541's property, restated for the renderer.

    The cap that caused #1541 sat between the body and the matcher. The
    matcher is now GitHub, so the property is that the whole body reaches it.
    """
    body = "x" * 200_000 + "\nCloses #4242."
    runner = _Runner()
    render_markdown(body, _CONTEXT, run=runner)
    assert json.loads(runner.kwargs["input"])["text"] == body


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


def test_an_empty_render_of_a_non_empty_body_raises() -> None:
    """The unparseable case that matters: a 200 carrying nothing.

    Returning `([], [])` here is indistinguishable from a body with no
    keyword, which is the silence AC5 forbids.
    """
    runner = _Runner(stdout="")
    with pytest.raises(RendererUnavailable):
        render_markdown("Closes #7", _CONTEXT, run=runner)


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


def test_the_docstring_states_what_still_diverges_from_github() -> None:
    doc = _SCRIPT.read_text(encoding="utf-8")
    assert "Divergences that REMAIN, deliberately" in doc
    for remaining in ("Commit messages", "Cross-repository", "target branch", "block quote"):
        assert remaining in doc


def test_the_docstring_cites_githubs_keyword_page_as_one_url() -> None:
    """Two independent substring checks pass on a host in one sentence and a
    path in another, which is not a citation a reader can follow."""
    import re

    doc = _SCRIPT.read_text(encoding="utf-8")
    assert re.search(
        r"https://docs\.github\.com/\S*linking-a-pull-request-to-an-issue", doc
    ), "the docstring does not cite GitHub's closing-keyword page by URL"


def test_the_keyword_alternation_is_built_from_the_nine() -> None:
    """The regex that decides a close must carry every keyword, not three."""
    for keyword in KEYWORDS:
        assert ADJACENT_RE.search(f"{keyword} ") is not None
    assert ADJACENT_RE.search("mentions ") is None
