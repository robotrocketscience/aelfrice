"""#1691: pasted terminal and tool output is not the user's belief.

The #1649 gate keeps a `<pasted_content>` block, because a paste is the
user's choice of what to say. When the paste was a terminal's output, each
line of it became a belief: a pasted `git status` came back on the next
turn as "On branch ..." and "(use "git restore <file>..." ...)". Pasted
prose is still kept; only output is dropped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.ingest import _user_speech, ingest_jsonl
from aelfrice.store import MemoryStore

PROSE = "The retry loop stops after three attempts."
OTHER_PROSE = "The cache now expires after ten minutes."


def _paste(body: str, pid: str = "ab12") -> str:
    return f'<pasted_content id="{pid}">\n{body}\n</pasted_content id="{pid}">'


GIT_PASTE = "\n".join([
    "[feat/x 1a2b3c4] fix(api): handle an empty body",
    " 1 file changed, 8 insertions(+)",
    " src/api.py | 8 ++++++++",
    "On branch feat/x",
    "Your branch is up to date with 'origin/feat/x'.",
    "Changes not staged for commit:",
    '  (use "git add <file>..." to update what will be committed)',
    "\tmodified:   src/api.py",
    "\tdeleted:    docs/old.md",
    "Untracked files:",
    "\tnotes/",
    'no changes added to commit (use "git add" and/or "git commit -a")',
])


def test_a_pasted_git_status_leaves_nothing() -> None:
    assert _user_speech(_paste(GIT_PASTE)) is None


def test_a_terminal_transcript_goes_whole() -> None:
    # The block opens with a prompt, so even an output line that reads like
    # a sentence is the command's output, not the user's.
    body = "dev@laptop ~/proj (main)> make test\n" + PROSE
    assert _user_speech(_paste(body)) is None


@pytest.mark.parametrize("prompt", [
    "dev@laptop ~/proj (main)> ls",
    "dev@laptop:~/proj$ ls",
    "dev@laptop proj % ls",
])
def test_each_shell_prompt_form_marks_a_transcript(prompt: str) -> None:
    assert _user_speech(_paste(prompt + "\n" + PROSE)) is None


@pytest.mark.parametrize("line", [
    "dev@laptop:~/proj$ make test",
    "Ran 2 shell commands",
    "Ran 1 command",
    "✻ Cooked for 3m 12s",
    "✻ Worked for 1h 2m 5s · done 4:10 PM",
    "fish: Unknown command: frobnicate",
    "09:14:02  cache miss for key users:42",
    "2026-10-02T12:00:01Z worker started",
    "Updating 5347899..3fdc540",
    "Fast-forward",
    " create mode 100644 docs/new.md",
    "Unmerged paths:",
    "nothing to commit, working tree clean",
    "\tnew file:   src/new.py",
    "\trenamed:    a.py -> b.py",
    ".bashrc    .config    Documents",
    "412  enabled=True  3.1MB  utf-8",
    "https://example.com/docs/setup",
    "quick+brown+fox",
    "assets/images/logo",
    "v1.2.3",
])
def test_each_output_line_is_dropped_and_the_prose_kept(line: str) -> None:
    speech = _user_speech(_paste("\n".join([PROSE, line, OTHER_PROSE])))
    assert speech is not None
    assert PROSE in speech and OTHER_PROSE in speech
    assert line.strip() not in speech


@pytest.mark.parametrize("prose", [
    "Thanks.",
    "On branch protection, we require signed commits.",
    "Fast-forward merges are required on main.",
    "Use two spaces  after a period.  It reads fine.",
    "The deploy ran at 12:56 and nothing failed.",
    "Contact dev@laptop.example about the outage.",
])
def test_prose_that_looks_like_output_is_kept(prose: str) -> None:
    assert _user_speech(_paste(prose)) == prose


def test_the_users_words_around_a_paste_are_kept() -> None:
    text = "Here is what I saw:\n" + _paste(GIT_PASTE) + "\nPlease fix the branch."
    speech = _user_speech(text) or ""
    assert "Here is what I saw:" in speech
    assert "Please fix the branch." in speech
    assert "On branch" not in speech


def test_a_paste_without_an_id_is_filtered_too() -> None:
    text = "<pasted_content>\n" + GIT_PASTE + "\n</pasted_content>"
    assert _user_speech(text) is None


def test_no_pasted_git_line_reaches_the_store(tmp_path: Path) -> None:
    rec = {"schema_version": 1, "role": "user", "session_id": "s1",
           "ts": "2026-10-02T10:00:00Z",
           "text": _paste(GIT_PASTE + "\n" + PROSE)}
    log = tmp_path / "turns.jsonl"
    log.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    store = MemoryStore(str(tmp_path / "memory.db"))
    try:
        ingest_jsonl(store, log)
        rows = [r[0] for r in store._conn.execute(  # noqa: SLF001
            "SELECT content FROM beliefs").fetchall()]
    finally:
        store.close()
    assert any(PROSE in r for r in rows)
    for r in rows:
        assert "On branch" not in r and "git add" not in r and "modified:" not in r


@pytest.mark.parametrize("prose", [
    "jon@corp.com: 50% of tests fail on main.",
    "Your branch is behind on reviews, please catch up.",
    "nothing to commit to yet, we are still deciding.",
    "modified: the plan was changed yesterday",
    "Ran 3 commands and nothing happened",
    "2026-10-01 12:00:00 was the outage window",
    "10:30:00 is when the deploy kicked off, please check",
    "bash: is the shell we standardise on",
    "I moved it.  Then I ran tests.  Everything passed, mostly",
    "and/or",
    "e.g.",
    "C++",
    "x=1",
])
def test_prose_that_opens_like_output_is_kept(prose: str) -> None:
    # Each rule matches its output format whole, not a sentence that only
    # starts the same way (review, PR #1693).
    assert _user_speech(_paste(prose)) == prose


def test_a_paste_opening_with_an_email_is_not_a_transcript() -> None:
    body = "jon@corp.com: 50% of tests fail on main.\n" + PROSE
    speech = _user_speech(_paste(body)) or ""
    assert PROSE in speech and "jon@corp.com" in speech


@pytest.mark.parametrize("line", [
    "." * 200_000 + " [100%]",
    ":" * 200_000 + " x",
    "/" * 200_000 + " x",
    "a." * 100_000 + " x",
    "a+" * 100_000 + " x",
    "/a" * 100_000 + " x",
])
def test_a_long_line_is_judged_in_linear_time(
    line: str, request: pytest.FixtureRequest,
) -> None:
    # A wall-clock budget, so it is opt-in (#1473): `pytest --run-perf`.
    try:
        run_perf = bool(request.config.getoption("--run-perf", default=False))
    except (AttributeError, ValueError):
        run_perf = False
    if not run_perf:
        pytest.skip("perf test gated on --run-perf")
    import time
    start = time.perf_counter()
    _user_speech(_paste(line))
    # A quadratic pattern took over two minutes on a 200,000-character run.
    assert time.perf_counter() - start < 2.0


def test_an_unclosed_paste_is_filtered_to_the_end() -> None:
    text = "Look at this:\n<pasted_content>\nOn branch main\n" + PROSE
    speech = _user_speech(text) or ""
    assert "Look at this:" in speech and PROSE in speech
    assert "On branch" not in speech and "pasted_content" not in speech


def test_a_stray_closing_tag_is_removed() -> None:
    text = _paste("On branch x") + "\n" + PROSE + "\n</pasted_content>"
    assert _user_speech(text) == PROSE


def test_a_host_glyph_is_judged_by_what_follows_it() -> None:
    speech = _user_speech(_paste("\n".join([
        "❯   Ran 1 shell command", "❯ " + PROSE]))) or ""
    assert "Ran 1 shell command" not in speech
    assert PROSE in speech


def test_words_after_a_closed_transcript_paste_are_kept() -> None:
    # A closed paste ends at its closing tag; the user's line after it is not
    # part of the terminal transcript.
    text = _paste("dev@laptop ~/proj> ls\nsrc  docs  tests") + "\n" + PROSE
    assert _user_speech(text) == PROSE
