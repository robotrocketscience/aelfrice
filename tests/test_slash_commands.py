"""Verify the slash-command markdown files match the CLI surface.

These files ship inside the aelfrice package so `setup.py` (v0.7.0) can
copy them into `~/.claude/commands/aelf/`. The tests check structural
properties — frontmatter, required fields, and that each command's
shell action references the matching CLI subcommand.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# The CLI's user-facing surface -- must match cli.py's subparsers exactly.
# Lifecycle commands (upgrade/uninstall/statusline) added in v1.0.1.
# `resolve` joins at v1.0.1 alongside the contradiction tie-breaker.
# `status` (alias for health), `regime` (preserved v1.0 classifier),
# and `migrate` (legacy-DB import) join at v1.1.0.
# `rebuild` (context rebuilder MVP) joins at v1.1.0 alpha.
# v1.3 user-facing surface — must match cli.py's non-SUPPRESS subparsers.
# Hidden subparsers (rebuild, statusline, bench, regime, migrate, unsetup,
# health, stats) keep working but ship no slash file.
EXPECTED_COMMANDS = (
    "onboard",
    "search",
    "lock",
    "locked",
    "unlock",
    "promote",
    "status",
    "doctor",
    "setup",
    # Slash command name is `upgrade` (imperative — does the upgrade
    # via Bash orchestration). The CLI verb stays `upgrade-cmd`
    # (advisory: prints the install-aware command line); see
    # HIDDEN_SUBCOMMANDS. The slash file invokes `aelf upgrade-cmd`
    # (no flags — the no-flag form has emitted `run:` on every
    # released CLI; `--check` short-circuited it on ≤2.0.1, see #530),
    # then runs the printed command, then `aelf setup`.
    "upgrade",
    "uninstall",
    "rebuild",
    "tail",
    # v2.0 / Track B (#389) — graph-walk surfaces.
    "reason",
    "wonder",
    # v2.0 (#441) — explicit affirmation, sibling of unlock.
    "confirm",
    # v2.0 (#440) — hard-delete escape hatch with confirmation prompt.
    # Registered here so test_no_extra_files_in_slash_commands_dir and
    # test_slash_commands_match_visible_cli_subcommands both enforce the
    # slash file exists and matches the CLI surface.
    "delete",
    # v2.0 (#439) — load-bearing belief lens (locked ∪ corroborated ∪ high-posterior).
    "core",
    # v2.0 (#365 R4) — operator-facing relevance-calibration harness
    # (P@K / ROC-AUC / Spearman ρ).
    "eval",
)


def _slash_dir() -> Path:
    """Locate src/aelfrice/slash_commands without importing aelfrice."""
    import aelfrice
    return Path(aelfrice.__file__).parent / "slash_commands"


@pytest.mark.parametrize("cmd", EXPECTED_COMMANDS)
def test_slash_command_file_exists(cmd: str) -> None:
    path = _slash_dir() / f"{cmd}.md"
    assert path.exists(), f"missing slash command: {path}"


@pytest.mark.parametrize("cmd", EXPECTED_COMMANDS)
def test_slash_command_file_nonempty(cmd: str) -> None:
    path = _slash_dir() / f"{cmd}.md"
    assert path.stat().st_size > 0


@pytest.mark.parametrize("cmd", EXPECTED_COMMANDS)
def test_slash_command_has_frontmatter(cmd: str) -> None:
    text = (_slash_dir() / f"{cmd}.md").read_text()
    assert text.startswith("---\n"), f"{cmd}.md missing opening frontmatter delimiter"
    assert "\n---\n" in text[4:], f"{cmd}.md missing closing frontmatter delimiter"


@pytest.mark.parametrize("cmd", EXPECTED_COMMANDS)
def test_slash_command_name_matches_filename(cmd: str) -> None:
    text = (_slash_dir() / f"{cmd}.md").read_text()
    m = re.search(r"^name:\s*(\S+)", text, re.MULTILINE)
    assert m is not None, f"{cmd}.md missing name: line"
    assert m.group(1) == f"aelf:{cmd}", f"{cmd}.md name mismatch: {m.group(1)}"


@pytest.mark.parametrize("cmd", EXPECTED_COMMANDS)
def test_slash_command_has_description(cmd: str) -> None:
    text = (_slash_dir() / f"{cmd}.md").read_text()
    m = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    assert m is not None, f"{cmd}.md missing description: line"
    assert len(m.group(1).strip()) >= 20, f"{cmd}.md description too short"


# Slash files that invoke a *different* CLI verb than their name (or
# orchestrate multiple CLI calls) are exempt from the strict
# slash-name == cli-verb check. Each entry must justify why.
_INVOKE_EXEMPT: dict[str, str] = {
    # Imperative orchestrator: calls `aelf upgrade-cmd` (no flags), runs
    # the printed install-aware upgrade command via Bash (separate
    # process, no mid-process replacement), then `aelf setup` to
    # refresh slash-command bundle, then `aelf upgrade-cmd` to clear
    # the stale update-banner cache. No single `uv run aelf upgrade`.
    "upgrade": "imperative orchestrator over upgrade-cmd + setup",
}


@pytest.mark.parametrize("cmd", EXPECTED_COMMANDS)
def test_slash_command_invokes_matching_cli(cmd: str) -> None:
    if cmd in _INVOKE_EXEMPT:
        pytest.skip(f"{cmd} is exempt: {_INVOKE_EXEMPT[cmd]}")
    text = (_slash_dir() / f"{cmd}.md").read_text()
    assert f"uv run aelf {cmd}" in text, (
        f"{cmd}.md does not invoke `uv run aelf {cmd}`"
    )


def test_no_extra_files_in_slash_commands_dir() -> None:
    """Catches typos like `loocked.md` and accidental .md.bak files."""
    files = sorted(p.name for p in _slash_dir().iterdir() if p.is_file())
    expected = sorted(f"{c}.md" for c in EXPECTED_COMMANDS)
    assert files == expected


def test_upgrade_slash_uses_no_flag_upgrade_cmd() -> None:
    """`/aelf:upgrade` step 1 must use the no-flag form of `aelf
    upgrade-cmd`, not `--check`.

    Issue #530 / lesson from #522: pre-#522 CLIs (≤2.0.1) short-circuit
    the `run:` line under `--check`. The slash file ships *with* the
    CLI, so a v2.0.1 user's installed slash uses whatever form the v2.0.1
    bundle had — and v2.0.1 shipped the slash that called `--check` and
    the CLI that suppressed `run:` under `--check` together. To prevent
    a recurrence (and to keep the slash robust against any future
    flag-gated regression), step 1 must use the form that has emitted
    `run:` on every released CLI: `aelf upgrade-cmd` with no flags.
    """
    text = (_slash_dir() / "upgrade.md").read_text()
    assert "aelf upgrade-cmd --check" not in text, (
        "/aelf:upgrade must not depend on `--check` for the `run:` line "
        "(see #530); use the no-flag form instead."
    )
    assert "aelf upgrade-cmd" in text, (
        "/aelf:upgrade step 1 must invoke `aelf upgrade-cmd`."
    )


def test_upgrade_slash_keeps_step_2_imperative() -> None:
    """`/aelf:upgrade` must keep step 2 as a hard execution step.

    Issue #611: agents were stopping after step 1 — printing the
    detection output and never running the `run:` command. The slash
    body's job is to keep step 2 explicit enough that an agent reading
    it cannot reasonably stop early. Lock in two markers:

    - the word `verbatim` appears in step 2 (signals "don't substitute
      a guess of what to run").
    - the body anticipates the parenthetical advisory by name
      (`(installed via pipx — use pipx to upgrade)`) and tells the
      agent it is not a substitute for executing the `run:` line.

    If a future edit softens either, this test fires. The point of the
    test is to keep the imperative load-bearing under copy edits.
    """
    text = (_slash_dir() / "upgrade.md").read_text()
    assert "verbatim" in text, (
        "/aelf:upgrade step 2 must instruct the agent to run the `run:` "
        "command verbatim (see #611 — guess-substitution failure mode)."
    )
    assert "installed via pipx" in text, (
        "/aelf:upgrade must call out the parenthetical advisory by name "
        "so the agent does not treat it as a substitute for step 2 "
        "(see #611 — stop-after-step-1 failure mode)."
    )


# Subparsers intentionally hidden from --help at v1.3. Aliases live one
# minor and are deleted at v1.4 (`health`, `stats`); the rest stay
# callable indefinitely as scripting / hook entry points.
HIDDEN_SUBCOMMANDS = frozenset({
    "statusline", "bench", "regime", "migrate", "unsetup",
    "health", "stats", "project-warm", "session-delta",
    "demote", "validate", "resolve", "feedback", "ingest-transcript",
    "sweep-feedback",
    # v2.1 (#475) — operator-side gate aggregator. CLI ships hidden;
    # slash command shape (`/aelf:gate-list` vs `/aelf:gate`) is a
    # separate ship.
    "gate",
    # `upgrade-cmd` is the advisory CLI verb (prints the
    # install-aware upgrade command). The imperative slash command
    # `/aelf:upgrade` calls it. CLI verb has no slash file of its own.
    "upgrade-cmd",
    # FastMCP server entrypoint — `aelf mcp` runs the MCP server over
    # stdio for host integration. Hidden because it's not a user-facing
    # workflow verb; hosts wire it via their MCP server config.
    "mcp",
    # `scan-derivation` is a discretion-gate CLI (#681): N-gram Jaccard
    # similarity against a reference document, intended for git
    # pre-commit / pre-push hook wiring. No slash command — it's a
    # reviewer/automation surface, not a workflow verb.
    "scan-derivation",
})


def test_slash_commands_match_visible_cli_subcommands() -> None:
    """The CLI's full subparser set must equal EXPECTED ∪ HIDDEN, and the
    slash directory must equal EXPECTED. Catches drift in either
    direction."""
    from aelfrice.cli import build_parser
    parser = build_parser()
    sub_actions = [a for a in parser._subparsers._actions  # type: ignore[union-attr]
                    if a.__class__.__name__ == "_SubParsersAction"]
    assert sub_actions, "cli has no subparsers"
    cli_names = set(sub_actions[0].choices.keys())  # type: ignore[attr-defined]
    expected_total = set(EXPECTED_COMMANDS) | HIDDEN_SUBCOMMANDS
    assert cli_names == expected_total, (
        f"cli subcommands {sorted(cli_names)} != "
        f"EXPECTED ∪ HIDDEN {sorted(expected_total)}"
    )
