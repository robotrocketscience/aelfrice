"""The merge-train reads the whole PR body before closing issues (#1541).

The defect this pins: `.github/workflows/merge-train.yml` step 7/7 piped the
body through `head -c 8192` **before** matching `Closes #N`, so a keyword
written past the 8,192nd byte was truncated away. The merge landed, the train
printed "no linked issues parsed from PR body", and the issue stayed open with
nothing anywhere reporting a failure. Measured across five consecutive merges
the keyword sat at bytes 463, 9148, 8992, 0 and 0; the two over the cut both
lost their close.

The tests below assert both directions, because one alone is satisfied by a
broken parser: a parser that returns everything passes "a late keyword is
found", and one that returns nothing passes "a bare `#N` is not a link". The
offset cases carry the old 8,192-byte cut explicitly, so a future author who
re-introduces an input cap to bound the read fails here rather than in
production six merges later.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "merge_train_linked_issues.py"
_WORKFLOW = _REPO / ".github" / "workflows" / "merge-train.yml"

sys.path.insert(0, str(_REPO / "scripts"))

from merge_train_linked_issues import (  # noqa: E402
    LINK_RE,
    NOISY_COUNT,
    linked_issues,
    main,
)

# The cut that caused #1541. Referenced by name so the tests read as being
# about the defect rather than about an arbitrary number.
_OLD_TRUNCATION = 8192


def _body_with_keyword_at(offset: int, issue: int = 4242) -> str:
    """A body whose `Closes #<issue>` begins at exactly `offset` bytes.

    The filler ends in a newline on purpose. `LINK_RE` opens with `\b`, and
    there is no word boundary between a filler `x` and the `C` of `Closes`,
    so filler-up-to-the-offset would produce a body no parser should match
    and the offsets below would pass for the wrong reason.
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
    assert linked_issues(_body_with_keyword_at(offset)) == [4242]


def test_the_old_cut_would_have_failed_the_cases_above() -> None:
    """The control. Without it, the parametrisation above proves nothing.

    If this ever passes, the offsets chosen above no longer straddle the
    truncation they were chosen to straddle, and the test above has stopped
    testing the defect.
    """
    late = _body_with_keyword_at(9148)
    assert linked_issues(late[:_OLD_TRUNCATION]) == []
    assert linked_issues(late) == [4242]


def test_a_body_larger_than_any_cap_still_parses_every_link() -> None:
    body = "\n".join(
        ["prose line, and more of it, to make this look like a real PR body"] * 4000
        + ["Closes #11.", "Fixes #22.", "Resolves #33."]
    )
    assert len(body) > 64 * 1024
    assert linked_issues(body) == [11, 22, 33]


# --------------------------------------------------------------------------
# What counts as a link, and what does not.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "Closes #7",
        "closes #7",
        "CLOSES #7",
        "Fixes #7",
        "fixed?  no -- resolves #7",
        "Resolves\n#7",
        "Closes  \t #7",
        "Closes #7.",
        "(Closes #7)",
    ],
)
def test_forms_that_link(body: str) -> None:
    assert linked_issues(body) == [7]


@pytest.mark.parametrize(
    "body",
    [
        "#7",  # a bare reference is not a link
        "See #7 for context",
        "precloses #7",  # \b must not let a suffix match
        "Closes issue #7",  # GitHub does not accept this either
        "Closes#7",  # whitespace is required
        "Closes owner/repo#7",  # cross-repo: not ours to close
        "Closes #",  # no number
    ],
)
def test_forms_that_do_not_link(body: str) -> None:
    assert linked_issues(body) == []


def test_one_keyword_links_one_issue() -> None:
    """`Closes #1 #2` links only #1 -- GitHub needs a keyword per issue."""
    assert linked_issues("Closes #1 #2") == [1]
    assert linked_issues("Closes #1, closes #2") == [1, 2]


def test_duplicates_collapse_and_order_is_numeric_not_textual() -> None:
    assert linked_issues("Closes #10. Closes #9. Fixes #10.") == [9, 10]


def test_an_empty_body_is_not_an_error() -> None:
    assert linked_issues("") == []


# --------------------------------------------------------------------------
# The CLI the workflow actually invokes.
# --------------------------------------------------------------------------


# Every CLI test below spawns one interpreter that reads a string and prints
# numbers -- no network, no store, no lock. The budget is generous against
# that work and still bounded, so contention reports as slowness rather than
# as a hang (#1307). Scaled by the suite's own knob so a loaded machine does
# not turn a slow start into a failure.
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
def test_cli_reads_a_body_file_and_prints_bare_numbers(tmp_path: Path) -> None:
    f = tmp_path / "body.md"
    f.write_text(_body_with_keyword_at(9148, issue=1526), encoding="utf-8")
    proc = _run(["--body-file", str(f)])
    assert proc.returncode == 0
    assert proc.stdout.split() == ["1526"]


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_cli_reads_stdin() -> None:
    proc = _run([], stdin="Closes #99.\n")
    assert proc.returncode == 0
    assert proc.stdout.split() == ["99"]


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_cli_dry_run_names_what_it_would_close() -> None:
    proc = _run(["--dry-run"], stdin="Closes #99.\n")
    assert proc.returncode == 0
    assert proc.stdout.strip() == "would close #99"


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_cli_exits_nonzero_on_an_unreadable_body_file(tmp_path: Path) -> None:
    proc = _run(["--body-file", str(tmp_path / "absent.md")])
    assert proc.returncode != 0
    assert "cannot read" in proc.stderr


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_a_body_with_no_links_is_success_and_silence() -> None:
    proc = _run([], stdin="no trailer here\n")
    assert proc.returncode == 0
    assert proc.stdout == ""


@pytest.mark.timeout(_CLI_TIMEOUT)
def test_a_noisy_count_warns_but_drops_nothing() -> None:
    n = NOISY_COUNT + 5
    body = " ".join(f"Closes #{i}." for i in range(1, n + 1))
    proc = _run([], stdin=body)
    assert proc.returncode == 0
    assert [int(x) for x in proc.stdout.split()] == list(range(1, n + 1))
    assert "none dropped" in proc.stderr


# --------------------------------------------------------------------------
# The workflow must actually use it.
# --------------------------------------------------------------------------


def _workflow_code_lines() -> list[str]:
    """The workflow's executable lines, with `#` comments dropped.

    The comment explaining #1541 necessarily quotes the pipeline that caused
    it, so a naive substring search over the whole file finds the defect in
    its own postmortem.
    """
    lines = []
    for raw in _WORKFLOW.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        lines.append(raw)
    return lines


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
    """The shell parse must not survive beside the script that replaced it."""
    code = "\n".join(_workflow_code_lines())
    assert "closes|fixes|resolves" not in code


def test_the_regex_keyword_set_matches_what_the_docs_claim() -> None:
    """The docstring names three keywords; the regex must carry those three."""
    assert LINK_RE.pattern.count("|") == 2
    for kw in ("closes", "fixes", "resolves"):
        assert kw in LINK_RE.pattern


def test_main_returns_zero_for_a_body_on_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("Fixes #5.\n"))
    assert main([]) == 0
    assert capsys.readouterr().out.split() == ["5"]
