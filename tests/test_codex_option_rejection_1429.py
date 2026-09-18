"""#1429 — `--host codex` refuses the options the codex path never reads.

`aelf setup --host codex` reads only `--force` and `--codex-skills`;
`aelf unsetup --host codex` reads only `--host`. Everything else the shared
parser accepts used to be parsed, discarded, and reported as success. The
ratified disposition is to refuse the whole inapplicable set, with no
per-option exemption, before anything on disk moves.

Isolation notes, both of which a verifier reproduced as false results:

* `_cmd_unsetup_codex` resolves `skills_dest=None` through `Path.home()`,
  not through `$CODEX_HOME`, so the omission control (which runs the real
  teardown) reaches the developer's own `~/.agents/skills` unless `HOME`
  itself is redirected. Every test here redirects `HOME`, and the omission
  control asserts the resolved directory is the sandbox one.
* `$CODEX_HOME` has to exist before setup runs, or the command exits 1 on
  the missing-home check and every case passes for the wrong reason.
"""
from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import pytest

from aelfrice.cli import (
    _CODEX_APPLICABLE_DESTS,
    _codex_option_rejection,
    _explicitly_supplied_dests,
    _option_dests,
    _subcommand_parser,
    build_parser,
    codex_inapplicable_options,
    main,
)
from aelfrice.host_codex import _bundled_codex_skills

# Measured from the parser on 2026-09-17: setup rejects 18 options and
# unsetup rejects 14, matching the counts the issue reports. The sets
# themselves are always derived below, never hard-coded, so an option
# added to either parser is covered by the gate the moment it is
# registered; these two numbers exist so that a *silent* change in the
# population is visible in a diff.
_EXPECTED_INAPPLICABLE_COUNTS: dict[str, int] = {"setup": 18, "unsetup": 14}


def _sample_value(action: argparse.Action) -> str | None:
    """A usable value for `action`, or None when it takes none."""
    if action.nargs == 0:
        return None
    if action.choices:
        return str(sorted(action.choices)[0])
    if action.type is int:
        return "5"
    return "sample"


def _inapplicable_cases(cmd: str) -> list[tuple[str, str, str | None]]:
    """Every (cmd, option string, value) the codex path must refuse.

    Enumerated from the live parser, including the `--no-` half of each
    `BooleanOptionalAction`, which is the form a caller actually types.
    """
    parser = build_parser()
    sub = _subcommand_parser(parser, cmd)
    assert sub is not None
    rejected = set(codex_inapplicable_options(parser, cmd))
    cases: list[tuple[str, str, str | None]] = []
    for action in sub._actions:  # noqa: SLF001 — no public accessor
        if action.dest not in rejected:
            continue
        for opt in action.option_strings:
            cases.append((cmd, opt, _sample_value(action)))
    return cases


_CASES = _inapplicable_cases("setup") + _inapplicable_cases("unsetup")


@pytest.fixture()
def codex_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> "CodexEnv":
    """A sandbox home plus an existing `$CODEX_HOME`."""
    import aelfrice.auto_install as auto_install
    import aelfrice.setup as setup_mod

    home = tmp_path / "home"
    home.mkdir()
    codex = tmp_path / "codex"
    codex.mkdir()  # hazard 2: setup exits 1 against a missing codex home
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.home() on Windows
    monkeypatch.setenv("CODEX_HOME", str(codex))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(
        auto_install, "AELFRICE_DOTDIR", home / ".aelfrice"
    )
    monkeypatch.setattr(
        auto_install, "OPT_OUT_PATH", home / ".aelfrice" / "opt-out-hooks.json"
    )
    monkeypatch.setattr(
        setup_mod, "USER_SETTINGS_PATH", home / ".claude" / "settings.json"
    )
    return CodexEnv(home=home, codex=codex)


class CodexEnv:
    def __init__(self, home: Path, codex: Path) -> None:
        self.home = home
        self.codex = codex

    @property
    def hooks(self) -> Path:
        return self.codex / "hooks.json"

    @property
    def skills(self) -> Path:
        return self.home / ".agents" / "skills"

    def is_untouched(self) -> bool:
        return not self.hooks.exists() and not self.skills.exists()


# --- the population the gate covers ----------------------------------------


@pytest.mark.timeout(60)
def test_inapplicable_sets_come_from_the_parser() -> None:
    parser = build_parser()
    for cmd, expected in _EXPECTED_INAPPLICABLE_COUNTS.items():
        sub = _subcommand_parser(parser, cmd)
        assert sub is not None
        every = set(_option_dests(sub))
        rejected = set(codex_inapplicable_options(parser, cmd))
        # Partition, exactly: nothing is both applicable and rejected, and
        # nothing escapes the partition.
        assert rejected == every - _CODEX_APPLICABLE_DESTS[cmd]
        assert _CODEX_APPLICABLE_DESTS[cmd] <= every
        assert len(rejected) == expected, sorted(rejected)


_REPO = Path(__file__).resolve().parents[1]
_ENTRY = _REPO / "CHANGELOG" / "unreleased" / "1429-codex-option-rejection.md"
_NUMBER_WORDS: dict[int, str] = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


@pytest.mark.timeout(60)
def test_changelog_count_matches_the_command_the_entry_quotes() -> None:
    """The entry's one spelled-out figure is re-derived, not trusted.

    The entry quotes a command line and then says how many of its options
    the executor never saw. `--host codex` is on that line and is the
    option that routes the call into `_cmd_setup_codex`, so it is
    applicable and must not be counted. Deriving the figure from the same
    parser the gate uses keeps the prose from drifting off the code.
    """
    text = _ENTRY.read_text(encoding="utf-8")
    quoted = re.search(r"`(aelf setup --host codex [^`]+)`", text)
    assert quoted is not None, "the entry no longer quotes a setup command"
    argv = quoted.group(1).split()[1:]

    parser = build_parser()
    inapplicable = codex_inapplicable_options(parser, "setup")
    supplied = _explicitly_supplied_dests(argv)
    discarded = sorted(
        opt for dest, opt in inapplicable.items() if dest in supplied
    )
    # The routing option is honoured, so it is never part of the figure.
    assert "host" in supplied
    assert "host" not in inapplicable

    claimed = re.search(r"a success report for (\w+) instructions", text)
    assert claimed is not None, "the entry no longer states the figure"
    assert claimed.group(1) == _NUMBER_WORDS[len(discarded)], discarded


@pytest.mark.timeout(60)
def test_applicable_dests_are_the_ones_the_codex_path_reads() -> None:
    """The applicable set is the executor's real read set, not a guess."""
    import inspect

    from aelfrice.cli import _cmd_setup_codex, _cmd_unsetup_codex

    setup_src = inspect.getsource(_cmd_setup_codex)
    assert 'getattr(args, "force"' in setup_src
    assert 'getattr(args, "codex_skills"' in setup_src
    # unsetup reads nothing off the namespace at all.
    unsetup_src = inspect.getsource(_cmd_unsetup_codex)
    assert re.search(r"^\s+_ = args$", unsetup_src, re.M)


# --- refusal ---------------------------------------------------------------


@pytest.mark.timeout(60)
@pytest.mark.parametrize(("cmd", "option", "value"), _CASES, ids=[
    f"{c}{o}" for c, o, _ in _CASES
])
def test_explicit_inapplicable_option_exits_2_before_any_mutation(
    cmd: str,
    option: str,
    value: str | None,
    codex_env: CodexEnv,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [cmd, "--host", "codex", option]
    if value is not None:
        argv.append(value)
    out = io.StringIO()

    rc = main(argv, out)

    assert rc == 2
    err = capsys.readouterr().err
    assert option in err or option.replace("--no-", "--", 1) in err
    assert "#1429" in err
    assert codex_env.is_untouched()
    assert out.getvalue() == ""


@pytest.mark.timeout(60)
def test_equals_form_and_prefix_abbreviation_are_refused(
    codex_env: CodexEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    """A literal argv scan would miss both spellings; argparse does not."""
    assert main(["setup", "--host", "codex", "--scope=user"], io.StringIO()) == 2
    assert "--scope" in capsys.readouterr().err
    assert main(["setup", "--host", "codex", "--proj", "x"], io.StringIO()) == 2
    assert "--project-root" in capsys.readouterr().err
    assert codex_env.is_untouched()


@pytest.mark.timeout(60)
def test_several_inapplicable_options_are_all_named(
    codex_env: CodexEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        ["setup", "--host", "codex", "--rebuilder", "--no-statusline"],
        io.StringIO(),
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "--rebuilder" in err
    assert "--no-statusline" in err
    assert codex_env.is_untouched()


# --- the controls: what must still work ------------------------------------


@pytest.mark.timeout(60)
def test_omission_control_setup_then_unsetup_exit_0(
    codex_env: CodexEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bare `--host codex` still installs and still tears down.

    The teardown half resolves its skills directory from `HOME`; asserting
    the reported directory is the sandbox one is what keeps this test off
    the developer's real `~/.agents/skills`.
    """
    out = io.StringIO()
    assert main(["setup", "--host", "codex"], out) == 0
    assert codex_env.hooks.is_file()
    assert (codex_env.skills / "aelf-setup" / "SKILL.md").is_file()

    out = io.StringIO()
    assert main(["unsetup", "--host", "codex"], out) == 0
    text = out.getvalue()
    assert str(codex_env.skills) in text
    assert not list(codex_env.skills.glob("aelf-*"))
    capsys.readouterr()


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "extra",
    [["--force"], ["--codex-skills"], ["--no-codex-skills"]],
)
def test_applicable_options_are_still_accepted(
    extra: list[str], codex_env: CodexEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["setup", "--host", "codex", *extra], io.StringIO())
    assert rc == 0
    assert codex_env.hooks.is_file()
    capsys.readouterr()


@pytest.mark.timeout(60)
def test_claude_host_is_not_gated() -> None:
    """The gate keys on `--host codex`; the default host keeps every flag."""
    parser = build_parser()
    args = parser.parse_args(
        ["setup", "--scope", "user", "--rebuilder", "--no-statusline"]
    )
    verdict = _codex_option_rejection(
        parser,
        "setup",
        args,
        ["setup", "--scope", "user", "--rebuilder", "--no-statusline"],
    )
    assert verdict is None


@pytest.mark.timeout(60)
def test_other_subcommands_are_not_gated() -> None:
    """`doctor` and `uninstall` accept `--host codex`; #1429 is setup-only."""
    parser = build_parser()
    argv = ["uninstall", "--host", "codex", "--keep-db", "--keep-hook"]
    args = parser.parse_args(argv)
    assert _codex_option_rejection(parser, "uninstall", args, argv) is None


# --- the first-party caller: the generated `$aelf-setup` skill -------------


def _rejected_option_strings() -> frozenset[str]:
    parser = build_parser()
    strings: set[str] = set()
    for cmd in _CODEX_APPLICABLE_DESTS:
        sub = _subcommand_parser(parser, cmd)
        assert sub is not None
        rejected = set(codex_inapplicable_options(parser, cmd))
        for action in sub._actions:  # noqa: SLF001 — no public accessor
            if action.dest in rejected:
                strings.update(action.option_strings)
    return frozenset(strings)


def _normalize(block: str) -> str:
    """Fold backticks and hyphens to spaces so host qualifiers compare."""
    return " ".join(re.sub(r"[`\-]", " ", block).split()).lower()


@pytest.mark.timeout(60)
def test_generated_skills_never_instruct_a_refused_option_unqualified() -> None:
    """Every mention of a refused option is marked as claude-host guidance.

    The `$aelf-setup` skill mandates `--host codex` on every invocation, so
    an unqualified "opt out with `--no-pre-issue-guard`" in its prose tells
    a codex agent to run a command this gate now refuses.
    """
    rejected = _rejected_option_strings()
    offenders: list[tuple[str, str]] = []
    for skill_name, text in sorted(_bundled_codex_skills().items()):
        for block in text.split("\n\n"):
            named = sorted(opt for opt in rejected if opt in block)
            if not named:
                continue
            if "claude host" not in _normalize(block):
                offenders.append((skill_name, ", ".join(named)))
    assert offenders == []


@pytest.mark.timeout(60)
def test_generated_setup_skill_states_the_refusal_contract() -> None:
    text = _bundled_codex_skills()["aelf-setup"]
    assert "--host codex" in text
    assert "exit 2" in text
    assert "--force" in text
    assert "--codex-skills" in text
    assert "#1429" in text
