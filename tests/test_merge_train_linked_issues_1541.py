r"""The merge-train reads the whole PR body before closing issues (#1541).

The defect this pins: `.github/workflows/merge-train.yml` step 7/7 piped the
body through `head -c 8192` **before** matching `Closes #N`, so a keyword
written past the 8,192nd byte was truncated away. The merge landed, the train
printed "no linked issues parsed from PR body", and the issue stayed open with
nothing anywhere reporting a failure. Measured across five consecutive merges
the keyword sat at bytes 463, 9148, 8992, 0 and 0; the two over the cut both
lost their close.

#1549 moved the decision itself to GitHub: `scripts/merge_train_linked_issues.py`
renders the body through GitHub's `/markdown` endpoint and acts on the anchors
GitHub produced. That moves the boundary this module guards without changing the
property -- the cut used to sit between the body and the matcher, and the matcher
is now GitHub, so the property is that **the whole body reaches the renderer**.
Which references GitHub anchors, and which anchor is a close directive, belong to
`tests/test_merge_train_close_keywords_1549.py` and are not restated here.

The tests below assert both directions, because one alone is satisfied by a
broken tool: one that sends everything passes "a late keyword is found", and one
that sends nothing passes "a truncated body finds nothing". The offset cases
carry the old 8,192-byte cut explicitly, so a future author who re-introduces an
input cap to bound the read fails here rather than in production six merges
later.
"""
from __future__ import annotations

import html
import io
import json
import re
import sys
from pathlib import Path

import pytest
from merge_train_fake_gh import (
    CLI_TIMEOUT,
    fake_gh,
    issue_anchor,
    recorded_call,
    run_cli,
)

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "merge_train_linked_issues.py"
_WORKFLOW = _REPO / ".github" / "workflows" / "merge-train.yml"

sys.path.insert(0, str(_REPO / "scripts"))

from merge_train_linked_issues import (  # noqa: E402
    ADJACENT_RE,
    NOISY_COUNT,
    linked_issues,
    main,
)

# GitHub's nine closing keywords, spelled out here rather than imported from
# the module under test. Comparing the compiled pattern against the tuple it
# was built from holds for any value of that tuple, so it would pass a
# narrowing back to the three the shell used.
_GITHUB_KEYWORDS = (
    "close",
    "closed",
    "closes",
    "fix",
    "fixed",
    "fixes",
    "resolve",
    "resolved",
    "resolves",
)

_CONTEXT = "robotrocketscience/aelfrice"

# The cut that caused #1541. Referenced by name so the tests read as being
# about the defect rather than about an arbitrary number.
_OLD_TRUNCATION = 8192

_REFERENCE_RE = re.compile(r"(?:(?P<repo>[A-Za-z0-9._-]+/[A-Za-z0-9._-]+))?#(?P<n>\d+)")


def _one_paragraph(body: str, repo: str) -> str:
    """A stand-in renderer that anchors every `#N`, whatever its offset.

    The tests must not reach the network, and what these tests are about is
    the *transport*: whether the whole body gets to the renderer. So this
    stand-in is deliberately offset-blind -- it has no cut of any kind, which
    is the property a truncating tool has to fail against. It is not a claim
    about GitHub's own rendering; that is replayed from recorded responses in
    `tests/test_merge_train_close_keywords_1549.py`.

    `test_the_old_cut_would_have_failed_the_cases_above` is the control that
    keeps it honest: the same offsets, rendered after a `head -c 8192`, lose
    their close.
    """
    escaped = html.escape(body, quote=False)
    return "<p>" + _REFERENCE_RE.sub(
        lambda m: issue_anchor(m.group("n"), m.group("repo") or repo), escaped
    ) + "</p>"


def _body_with_keyword_at(offset: int, issue: int = 4242) -> str:
    """A body whose `Closes #<issue>` begins at exactly `offset` bytes.

    The filler ends in a newline on purpose. The keyword needs a word boundary
    in front of it, and there is none between a filler `x` and the `C` of
    `Closes`, so filler-up-to-the-offset would produce a body nothing should
    match and the offsets below would pass for the wrong reason.
    """
    filler = ("x" * (offset - 1) + "\n") if offset else ""
    body = f"{filler}Closes #{issue}."
    assert body.encode().index(b"Closes") == offset
    return body


# --------------------------------------------------------------------------
# The regression itself: offset must not decide whether a link is seen.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset",
    [
        0,
        463,  # the PR that worked
        _OLD_TRUNCATION - 20,  # just under the old cut
        _OLD_TRUNCATION,  # exactly at it
        8992,  # a PR that silently lost its close
        9148,  # the other one
        _OLD_TRUNCATION * 8,  # far past any plausible re-added cap
    ],
)
def test_a_closing_keyword_is_found_at_any_offset(offset: int) -> None:
    body = _body_with_keyword_at(offset)
    assert linked_issues(body, _CONTEXT, render=_one_paragraph) == [4242]


def test_the_old_cut_would_have_failed_the_cases_above() -> None:
    """The control. Without it, the parametrisation above proves nothing.

    A renderer that truncates the way the shell pipeline did loses the late
    keyword, which is both the original defect and the proof that
    `_one_paragraph` is not simply answering `[4242]` to everything.
    """

    def truncating(body: str, repo: str) -> str:
        return _one_paragraph(body[:_OLD_TRUNCATION], repo)

    late = _body_with_keyword_at(9148)
    assert linked_issues(late, _CONTEXT, render=truncating) == []
    assert linked_issues(late, _CONTEXT, render=_one_paragraph) == [4242]


def test_the_whole_body_reaches_the_renderer_byte_for_byte() -> None:
    """The property, stated at the boundary it now lives on.

    Asserted as identity rather than as a length: a tool that sent the last
    8 KiB instead of the first would pass a length check on a body that size.
    """
    sent: list[str] = []

    def recording(body: str, repo: str) -> str:
        sent.append(body)
        return _one_paragraph(body, repo)

    body = _body_with_keyword_at(_OLD_TRUNCATION * 8)
    assert linked_issues(body, _CONTEXT, render=recording) == [4242]
    assert sent == [body]


def test_a_body_larger_than_any_cap_still_parses_every_link() -> None:
    body = "\n".join(
        ["prose line, and more of it, to make this look like a real PR body"] * 4000
        + ["Closes #11.", "Fixes #22.", "Resolves #33."]
    )
    assert len(body) > 64 * 1024
    assert linked_issues(body, _CONTEXT, render=_one_paragraph) == [11, 22, 33]


def test_duplicates_collapse_and_order_is_numeric_not_textual() -> None:
    body = "Closes #10. Closes #9. Fixes #10."
    assert linked_issues(body, _CONTEXT, render=_one_paragraph) == [9, 10]


def test_an_empty_body_is_not_an_error() -> None:
    def explode(body: str, repo: str) -> str:
        raise AssertionError("an empty body must not cost a render")

    assert linked_issues("", _CONTEXT, render=explode) == []


# --------------------------------------------------------------------------
# The CLI the workflow actually invokes.
# --------------------------------------------------------------------------


@pytest.mark.timeout(CLI_TIMEOUT)
def test_cli_sends_a_whole_body_file_and_prints_bare_numbers(tmp_path: Path) -> None:
    """The #1541 property at the boundary the workflow crosses.

    The fake `gh` records what it was handed, so this asserts the file's whole
    9,161 bytes arrived rather than only that the right number came back.
    """
    body = _body_with_keyword_at(9148, issue=1526)
    f = tmp_path / "body.md"
    f.write_text(body, encoding="utf-8")

    bin_dir = fake_gh(tmp_path, f"<p>Closes {issue_anchor(1526, _CONTEXT)}</p>")
    proc = run_cli(
        _SCRIPT, ["--repo", _CONTEXT, "--body-file", str(f)], bin_dir=bin_dir
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["1526"]
    assert json.loads(recorded_call(tmp_path)["stdin"])["text"] == body


@pytest.mark.timeout(CLI_TIMEOUT)
def test_cli_reads_stdin(tmp_path: Path) -> None:
    bin_dir = fake_gh(tmp_path, f"<p>Closes {issue_anchor(99, _CONTEXT)}</p>")
    proc = run_cli(_SCRIPT, ["--repo", _CONTEXT], bin_dir=bin_dir, stdin="Closes #99.\n")
    assert proc.returncode == 0
    assert proc.stdout.split() == ["99"]


@pytest.mark.timeout(CLI_TIMEOUT)
def test_cli_dry_run_names_what_it_would_close(tmp_path: Path) -> None:
    bin_dir = fake_gh(tmp_path, f"<p>Closes {issue_anchor(99, _CONTEXT)}</p>")
    proc = run_cli(
        _SCRIPT, ["--repo", _CONTEXT, "--dry-run"], bin_dir=bin_dir, stdin="Closes #99.\n"
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "would close #99"


@pytest.mark.timeout(CLI_TIMEOUT)
def test_cli_exits_nonzero_on_an_unreadable_body_file(tmp_path: Path) -> None:
    bin_dir = fake_gh(tmp_path, "<p>unused</p>")
    proc = run_cli(
        _SCRIPT,
        ["--repo", _CONTEXT, "--body-file", str(tmp_path / "absent.md")],
        bin_dir=bin_dir,
    )
    assert proc.returncode != 0
    assert "cannot read" in proc.stderr


@pytest.mark.timeout(CLI_TIMEOUT)
def test_a_body_with_no_links_is_success_and_silence(tmp_path: Path) -> None:
    bin_dir = fake_gh(tmp_path, "<p>no trailer here</p>")
    proc = run_cli(
        _SCRIPT, ["--repo", _CONTEXT], bin_dir=bin_dir, stdin="no trailer here\n"
    )
    assert proc.returncode == 0
    assert proc.stdout == ""
    assert proc.stderr == ""


@pytest.mark.timeout(CLI_TIMEOUT)
def test_a_noisy_count_warns_but_drops_nothing(tmp_path: Path) -> None:
    n = NOISY_COUNT + 5
    body = " ".join(f"Closes #{i}." for i in range(1, n + 1))
    bin_dir = fake_gh(tmp_path, _one_paragraph(body, _CONTEXT))
    proc = run_cli(_SCRIPT, ["--repo", _CONTEXT], bin_dir=bin_dir, stdin=body)

    assert proc.returncode == 0
    assert [int(x) for x in proc.stdout.split()] == list(range(1, n + 1))
    assert "none dropped" in proc.stderr


def test_main_returns_zero_for_a_body_on_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("Fixes #5.\n"))
    monkeypatch.setattr(
        "merge_train_linked_issues.render_markdown", _one_paragraph
    )
    assert main(["--repo", _CONTEXT]) == 0
    assert capsys.readouterr().out.split() == ["5"]


# --------------------------------------------------------------------------
# The workflow must actually use it.
# --------------------------------------------------------------------------


def _workflow_code_lines() -> list[str]:
    """The workflow's executable lines, with `#` comments dropped.

    The comment explaining #1541 necessarily quotes the pipeline that caused
    it, so a naive substring search over the whole file finds the defect in
    its own postmortem.
    """
    return [
        raw
        for raw in _WORKFLOW.read_text(encoding="utf-8").splitlines()
        if not raw.strip().startswith("#")
    ]


def test_the_workflow_no_longer_truncates_the_body_before_matching() -> None:
    code = "\n".join(_workflow_code_lines())
    assert "head -c" not in code, (
        "merge-train.yml truncates the PR body again; #1541 is back"
    )


def test_that_assertion_is_not_vacuous() -> None:
    """The comment does quote the old pipeline, so the filter is load-bearing."""
    whole = _WORKFLOW.read_text(encoding="utf-8")
    assert "head -c 8192" in whole


def test_the_workflow_calls_this_script() -> None:
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "scripts/merge_train_linked_issues.py" in text


def test_the_inline_grep_pipeline_is_gone() -> None:
    """The shell parse must not survive beside the tool that replaced it."""
    code = "\n".join(_workflow_code_lines())
    assert "closes|fixes|resolves" not in code


def test_the_keyword_set_matches_what_the_docs_claim() -> None:
    """The docstring names GitHub's nine keywords; the regex must carry them.

    The alternation read is the one that decides a close -- the keyword group
    of `ADJACENT_RE`, which is matched against the text immediately before
    each anchor GitHub returned.

    It is compared against `_GITHUB_KEYWORDS`, not against the module's own
    `KEYWORDS`. The pattern is built from `KEYWORDS`, so comparing the two
    pins nothing: narrowing `KEYWORDS` back to the shell's three narrows the
    alternation with it and the assertion still holds.
    """
    alternation = re.search(r"\(\?P<keyword>([^)]*)\)", ADJACENT_RE.pattern)
    assert alternation is not None, "the keyword group is gone from the regex"
    assert sorted(alternation.group(1).split("|")) == sorted(_GITHUB_KEYWORDS)
