"""The pre-push hooks must actually call the #1617 checks.

Both hook surfaces had zero test coverage, so nothing pinned that the
checks were wired in at all — and one of them turned out not to be live.
There are two, and the difference matters:

- `.githooks/pre-push` is TRACKED and is what `core.hooksPath` points at
  after `scripts/setup-hooks.sh`. Editing it takes effect on the next
  pull, with no install step.
- `scripts/install-discretion-hook.sh` is an INSTALLER. It writes a copy
  into `.git/hooks/pre-push`, so editing it changes nothing on a machine
  until somebody re-runs it with `--force`.

These arms assert the wiring exists in both, so a future edit cannot
silently drop it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRACKED_HOOK = _REPO_ROOT / ".githooks" / "pre-push"
_INSTALLER = _REPO_ROOT / "scripts" / "install-discretion-hook.sh"

_CHECKS = ("check_no_personal_paths", "check_commit_identity")


@pytest.mark.timeout(30)
@pytest.mark.parametrize("check", _CHECKS)
def test_the_tracked_hook_calls_the_check(check: str) -> None:
    text = _TRACKED_HOOK.read_text(encoding="utf-8")
    assert check in text, (
        f"{check} is not invoked by .githooks/pre-push, so the tracked "
        "hook does not gate it"
    )


@pytest.mark.timeout(30)
@pytest.mark.parametrize("check", _CHECKS)
def test_the_installer_writes_the_check(check: str) -> None:
    text = _INSTALLER.read_text(encoding="utf-8")
    assert check in text, (
        f"{check} is absent from the installed hook's body in "
        "scripts/install-discretion-hook.sh"
    )


@pytest.mark.timeout(30)
def test_a_disclosure_is_not_waivable_by_the_staleness_override() -> None:
    """`ALLOW_STALE_BRANCH_PUSH` waives staleness, never a disclosure.

    The two failures share one hook, so a single `fail` flag would have
    let a branch-freshness override wave a personal path straight past.
    """
    text = _TRACKED_HOOK.read_text(encoding="utf-8")
    assert "disclosure_fail" in text, (
        "disclosure failures are not tracked separately from staleness"
    )
    stale_idx = text.index("ALLOW_STALE_BRANCH_PUSH:-0")
    disclosure_idx = text.index('if [ "$disclosure_fail" -ne 0 ]')
    assert disclosure_idx < stale_idx, (
        "the disclosure gate must be checked BEFORE the staleness "
        "override, or the override reaches it"
    )


@pytest.mark.timeout(30)
def test_the_installer_refuses_every_host_configuration_directory() -> None:
    """`BANNED_PATHS` must cover all three, not just `.claude/`.

    `.gemini/` walked past this list in PR #1610 because it named only
    `.claude/`.
    """
    text = _INSTALLER.read_text(encoding="utf-8")
    line = next(
        (ln for ln in text.splitlines() if ln.startswith("BANNED_PATHS=")), ""
    )
    assert line, "BANNED_PATHS not found in the installer"
    for directory in (r"\.claude/", r"\.gemini/", r"\.codex/"):
        assert directory in line, f"{directory} missing from BANNED_PATHS: {line}"


@pytest.mark.timeout(30)
@pytest.mark.parametrize("path", [_TRACKED_HOOK, _INSTALLER], ids=lambda p: p.name)
def test_each_hook_parses_under_its_own_interpreter(path: Path) -> None:
    """Parse-check each file with the shell its shebang names.

    They are not the same language: the tracked hook is `#!/bin/sh` and
    stays POSIX so it runs anywhere, while the installer is bash and uses
    process substitution. Checking both with `sh -n` fails the installer
    for conforming to its own shebang.
    """
    shebang = path.read_text(encoding="utf-8").splitlines()[0]
    shell = "bash" if "bash" in shebang else "sh"
    r = subprocess.run(
        [shell, "-n", str(path)], capture_output=True, text=True, timeout=30
    )
    assert r.returncode == 0, f"{path.name} is not valid {shell}:\n{r.stderr}"


def _vocab_filter() -> tuple[str, str, list[str]]:
    """The installer's `BANNED_VOCAB`, `ALLOWED_VOCAB_LINE`, and `vocab_hits`
    definitions as shell source, the allowed line, and the banned words.

    Read from the installer rather than spelled here, so this file carries
    none of the words the hook bans.
    """
    lines = _INSTALLER.read_text(encoding="utf-8").splitlines()
    vocab = next(ln for ln in lines if ln.startswith("BANNED_VOCAB="))
    allowed = next(ln for ln in lines if ln.startswith("ALLOWED_VOCAB_LINE="))
    start = lines.index("vocab_hits() {")
    func = lines[start:lines.index("}", start) + 1]
    words = vocab.split("(", 1)[1].split(")", 1)[0].split("|")
    allowed_line = allowed.split("=", 1)[1][1:-1]
    return "\n".join([vocab, allowed, *func]), allowed_line, words


def _hits(source: str, diff: str) -> list[str]:
    r = subprocess.run(
        ["bash", "-c", source + "\nvocab_hits"],
        input=diff, capture_output=True, text=True, timeout=20, check=True,
    )
    return r.stdout.splitlines()


@pytest.mark.timeout(30)
def test_only_the_whole_allowed_line_passes_the_vocabulary_check() -> None:
    """Killed by: dropping the filter, matching the allowed line as a
    substring, or letting it exempt the word anywhere else (#1741)."""
    source, allowed, words = _vocab_filter()
    word = allowed.split('"')[1]
    assert word.lower() in (w.lower() for w in words)
    assert _hits(source, f"{allowed}\n context\n") == []
    # The same word anywhere else is still caught: a longer line that
    # contains the allowed one, another assignment, and the allowed line
    # behind a second `+`.
    for line in (
        f"{allowed}  # x", f"+x = {word!r}", f"+{allowed}",
        allowed.replace(word, word.capitalize()),
    ):
        assert len(_hits(source, line + "\n")) == 1, line
    # Other banned words are unaffected.
    assert len(_hits(source, f"+{words[0]}\n")) == 1


@pytest.mark.timeout(30)
def test_the_diff_vocabulary_check_uses_the_filter() -> None:
    """Killed by: a diff check that greps `BANNED_VOCAB` directly again."""
    text = _INSTALLER.read_text(encoding="utf-8")
    block = text[text.index("bad_vocab=$(git diff"):text.index('if [ -n "$bad_vocab" ]')]
    assert "| vocab_hits)" in block
    assert "BANNED_VOCAB" not in block
