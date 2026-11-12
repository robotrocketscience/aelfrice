"""Pin what each host's setup, unsetup, uninstall, and doctor do (#1812).

#1812 threads the host through the host-specific code so a third host can
plug in. That refactor must not change what the `claude` or `codex` host
writes or prints. These tests pass against the code the refactor starts
from and pass unchanged after it.

Each test drives `cli.main` in a sandbox home and compares the normalized
result with a literal: the settings or hooks file written, the slash
commands or skills installed, and the lines printed. Normalization removes
only what varies by machine: the sandbox root and the directory the hook
launchers resolve to. A hook command is reduced to its launcher key plus
its arguments, so `<venv>/bin/aelf-hook` and `aelf-hook.exe` both read
`aelf-hook`.
"""
from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

import aelfrice.auto_install as auto_install
import aelfrice.cli as cli
import aelfrice.setup as setup_mod
from aelfrice.host_codex import _bundled_codex_skills
from aelfrice.launcher import command_tokens, launcher_key


@dataclass(frozen=True)
class Sandbox:
    root: Path
    home: Path

    @property
    def settings(self) -> Path:
        return self.home / ".claude" / "settings.json"

    @property
    def slash_dir(self) -> Path:
        return self.home / ".claude" / "commands" / "aelf"

    @property
    def codex(self) -> Path:
        return self.home / ".codex"

    @property
    def skills(self) -> Path:
        return self.home / ".agents" / "skills"

    @property
    def opt_out(self) -> Path:
        return self.home / ".aelfrice" / "opt-out-hooks.json"


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sandbox:
    home = tmp_path / "userhome"
    (home / ".codex").mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.home() on Windows
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setenv("AELFRICE_DB", str(home / "memory.db"))
    # The pinned setup lines expect the opt-out branch of the commit hook.
    monkeypatch.setenv("AELF_NO_GIT_HOOK_INSTALL", "1")
    dotdir = home / ".aelfrice"
    monkeypatch.setattr(auto_install, "AELFRICE_DOTDIR", dotdir)
    monkeypatch.setattr(
        auto_install, "OPT_OUT_PATH", dotdir / "opt-out-hooks.json",
    )
    monkeypatch.setattr(
        auto_install, "STAMP_PATH", dotdir / "installed-manifest-version",
    )
    monkeypatch.setattr(
        setup_mod, "USER_SETTINGS_PATH", home / ".claude" / "settings.json",
    )
    monkeypatch.setattr(
        setup_mod, "SLASH_COMMANDS_DIR_DEFAULT",
        home / ".claude" / "commands" / "aelf",
    )
    monkeypatch.setattr(
        cli, "_CLAUDE_PROJECTS_DIR", home / ".claude" / "projects",
    )
    return Sandbox(root=tmp_path, home=home)


def _command_key(command: str) -> str:
    """A hook command with its program reduced to the launcher key."""
    tokens = command_tokens(command)
    if not tokens:
        return command
    return " ".join([launcher_key(tokens[0]), *tokens[1:]])


def _normalize_text(text: str, sb: Sandbox) -> list[str]:
    text = re.sub(
        r"command='([^']*)'",
        lambda m: f"command='{_command_key(m.group(1))}'",
        text,
    )
    text = text.replace(str(sb.root), "<TMP>").replace("\\", "/")
    return text.splitlines()


def _normalize_hooks(doc: dict[str, object]) -> dict[str, object]:
    """Copy of a settings or hooks.json document with keyed commands."""
    out = json.loads(json.dumps(doc))
    hooks = out.get("hooks")
    if isinstance(hooks, dict):
        for groups in hooks.values():
            for group in groups:
                for handler in group.get("hooks", []):
                    if isinstance(handler.get("command"), str):
                        handler["command"] = _command_key(handler["command"])
    return out


def _run(argv: list[str], sb: Sandbox) -> tuple[int, list[str]]:
    out = io.StringIO()
    rc = cli.main(argv, out)
    return rc, _normalize_text(out.getvalue(), sb)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slash_names() -> list[str]:
    return sorted(setup_mod.bundled_slash_files())


def _skill_names() -> list[str]:
    return sorted(_bundled_codex_skills())


def _claude_setup(sb: Sandbox) -> tuple[int, list[str]]:
    return _run(["setup", "--settings-path", str(sb.settings)], sb)


def _codex_setup(sb: Sandbox) -> tuple[int, list[str]]:
    return _run(["setup", "--host", "codex"], sb)


# --- the --host option -------------------------------------------------------


@pytest.mark.timeout(60)
def test_every_host_option_defaults_to_claude() -> None:
    parser = cli.build_parser(show_advanced=True)
    seen: dict[str, tuple[object, tuple[str, ...]]] = {}
    for cmd in ("setup", "unsetup", "uninstall", "doctor"):
        sub = cli._subcommand_parser(parser, cmd)
        assert sub is not None, cmd
        for action in sub._actions:
            if "--host" in action.option_strings:
                seen[cmd] = (action.default, tuple(action.choices or ()))
    assert seen == {
        cmd: ("claude", ("claude", "codex", "gemini"))
        for cmd in ("setup", "unsetup", "uninstall", "doctor")
    }


# --- tool names the hooks act on -------------------------------------------

_TOOL_NAMES = (
    "Grep", "Glob", "WebSearch", "WebFetch", "Bash", "Read", "Write",
    "Edit", "MultiEdit", "Agent", "Task", "grep", "bash",
)


@pytest.mark.timeout(60)
def test_search_tool_hook_acts_on_the_pinned_tool_names() -> None:
    from aelfrice import hook_search_tool

    search = [
        name for name in _TOOL_NAMES
        if hook_search_tool._is_search_tool_call({"tool_name": name})
    ]
    assert search == ["Grep", "Glob", "WebSearch", "WebFetch"]
    shell = [
        name for name in _TOOL_NAMES
        if hook_search_tool._extract_bash_query({
            "tool_name": name,
            "tool_input": {"command": "grep -rn needle_token src"},
        }) is not None
    ]
    assert shell == ["Bash"]


@pytest.mark.timeout(60)
def test_memory_mirror_hook_acts_on_the_pinned_tool_names(
    tmp_path: Path,
) -> None:
    from aelfrice import hook_claude_memory_mirror

    fact = tmp_path / ".claude" / "projects" / "p" / "memory" / "fact.md"
    mirrored = [
        name for name in _TOOL_NAMES
        if hook_claude_memory_mirror._file_path_for_memory_write({
            "tool_name": name, "tool_input": {"file_path": str(fact)},
        }) is not None
    ]
    assert mirrored == ["Write", "Edit", "MultiEdit"]


# --- claude host -------------------------------------------------------------

_SETTINGS_FILE = "<TMP>/userhome/.claude/settings.json"

_CLAUDE_SETUP_LINES = [
    f"installed UserPromptSubmit hook in {_SETTINGS_FILE} "
    "(command='aelf-hook')",
    "installed transcript-ingest hooks (UserPromptSubmit, Stop, PreCompact, "
    f"PostCompact) in {_SETTINGS_FILE} (command='aelf-transcript-logger')",
    f"installed SessionStart hook in {_SETTINGS_FILE} "
    "(command='aelf-session-start-hook')",
    f"installed Stop hook in {_SETTINGS_FILE} (command='aelf-stop-hook')",
    f"installed statusline in {_SETTINGS_FILE} (command='aelf statusline')",
    "commit-ingest git hook not installed (AELF_NO_GIT_HOOK_INSTALL=1)",
    f"installed search-tool PreToolUse hook in {_SETTINGS_FILE} "
    "(command='aelf-search-tool-hook')",
    f"installed search-tool-bash PreToolUse:Bash hook in {_SETTINGS_FILE} "
    "(command='aelf-search-tool-hook')",
    f"installed pre-issue-guard PreToolUse:Bash hook in {_SETTINGS_FILE} "
    "(command='aelf-pre-issue-hook')",
    f"installed claude-memory-mirror PostToolUse hook in {_SETTINGS_FILE} "
    "(command='aelf-claude-memory-mirror'); inert until "
    "AELFRICE_MIRROR_CLAUDE_MEMORY or [memory] mirror_claude_memory is set",
    f"installed agent-context PreToolUse:Agent hook in {_SETTINGS_FILE} "
    "(command='aelf-agent-context-hook')",
    "<slash commands>",
    "next step: this folder isn't in a git repository. Open a project "
    "first, then run `/aelf:onboard .` in your agent.",
]


def _handler(command: str, timeout: int) -> dict[str, object]:
    return {"type": "command", "command": command, "timeout": timeout}


_CLAUDE_SETTINGS = {
    "hooks": {
        "UserPromptSubmit": [
            {"hooks": [_handler("aelf-hook", 15)]},
            {"hooks": [_handler("aelf-transcript-logger", 30)]},
        ],
        "Stop": [
            {"hooks": [_handler("aelf-transcript-logger", 30)]},
            {"hooks": [_handler("aelf-stop-hook", 30)]},
        ],
        "PreCompact": [{"hooks": [_handler("aelf-transcript-logger", 30)]}],
        "PostCompact": [{"hooks": [_handler("aelf-transcript-logger", 30)]}],
        "SessionStart": [{"hooks": [_handler("aelf-session-start-hook", 15)]}],
        "PreToolUse": [
            {
                "hooks": [_handler("aelf-search-tool-hook", 15)],
                "matcher": "Grep|Glob|WebSearch|WebFetch",
            },
            {
                "hooks": [_handler("aelf-search-tool-hook", 15)],
                "matcher": "Bash",
            },
            {
                "hooks": [_handler("aelf-pre-issue-hook", 15)],
                "matcher": "Bash",
            },
            {
                "hooks": [_handler("aelf-agent-context-hook", 15)],
                "matcher": "^(Agent|Task)$",
            },
        ],
        "PostToolUse": [
            {
                "hooks": [_handler("aelf-claude-memory-mirror", 30)],
                "matcher": "Write|Edit|MultiEdit",
            },
        ],
    },
    "statusLine": {"type": "command", "command": "aelf statusline"},
}


def _slash_line(verb: str, sb: Sandbox) -> str:
    where = str(sb.slash_dir).replace(str(sb.root), "<TMP>").replace("\\", "/")
    return f"{verb} {len(_slash_names())} slash command(s) {where}: " + ", ".join(
        _slash_names()
    )


@pytest.mark.timeout(120)
def test_claude_setup_writes_the_pinned_settings(sandbox: Sandbox) -> None:
    rc, lines = _claude_setup(sandbox)
    assert rc == 0
    expected = [
        _slash_line("installed", sandbox).replace(
            "slash command(s) ", "slash command(s) in ",
        )
        if line == "<slash commands>" else line
        for line in _CLAUDE_SETUP_LINES
    ]
    assert lines == expected
    assert _normalize_hooks(_read_json(sandbox.settings)) == _CLAUDE_SETTINGS
    assert sorted(p.name for p in sandbox.slash_dir.iterdir()) == _slash_names()
    # The claude host leaves the Codex home and the agent skills alone.
    assert list(sandbox.codex.iterdir()) == []
    assert not sandbox.skills.exists()


@pytest.mark.timeout(120)
def test_claude_doctor_hooks_reports_the_pinned_scan(sandbox: Sandbox) -> None:
    assert _claude_setup(sandbox)[0] == 0
    rc, lines = _run(
        ["doctor", "hooks", "--user-settings", str(sandbox.settings)], sandbox,
    )
    assert rc == 0
    assert lines[:3] == [
        f"scanned user: {_SETTINGS_FILE}",
        "",
        "summary: 13 ok, 0 broken, 0 skipped",
    ]


_CLAUDE_UNSETUP_LINES = [
    f"removed 1 hook entry from {_SETTINGS_FILE} (basename='aelf-hook')",
    f"removed 1 SessionStart entry from {_SETTINGS_FILE}",
    f"removed 1 Stop entry from {_SETTINGS_FILE}",
    f"removed 1 UserPromptSubmit entry from {_SETTINGS_FILE}",
    f"removed 1 Stop entry from {_SETTINGS_FILE}",
    f"removed 1 PreCompact entry from {_SETTINGS_FILE}",
    f"removed 1 PostCompact entry from {_SETTINGS_FILE}",
    f"removed statusline from {_SETTINGS_FILE}",
    "no commit-ingest hook to remove",
    f"removed 1 claude-memory-mirror entry from {_SETTINGS_FILE}",
    f"removed 2 search-tool entries from {_SETTINGS_FILE}",
    f"no search-tool-bash hook in {_SETTINGS_FILE}",
    f"removed 1 pre-issue-guard entry from {_SETTINGS_FILE}",
    f"removed 1 agent-context entry from {_SETTINGS_FILE}",
    "<slash commands>",
]

# Removal empties each event's list and keeps the key.
_CLAUDE_SETTINGS_AFTER_REMOVAL = {
    "hooks": {event: [] for event in _CLAUDE_SETTINGS["hooks"]},
}


@pytest.mark.timeout(120)
def test_claude_unsetup_removes_the_pinned_entries(sandbox: Sandbox) -> None:
    assert _claude_setup(sandbox)[0] == 0
    rc, lines = _run(["unsetup", "--settings-path", str(sandbox.settings)], sandbox)
    assert rc == 0
    expected = [
        _slash_line("removed", sandbox).replace(
            "slash command(s) ", "slash command(s) from ",
        )
        if line == "<slash commands>" else line
        for line in _CLAUDE_UNSETUP_LINES
    ]
    assert lines == expected
    assert _read_json(sandbox.settings) == _CLAUDE_SETTINGS_AFTER_REMOVAL
    assert list(sandbox.slash_dir.iterdir()) == []


# Uninstall disposes of the install stamp the setup run wrote.
_STAMP_REMOVAL = "removed <TMP>/userhome/.aelfrice/installed-manifest-version"


@pytest.mark.timeout(120)
def test_claude_uninstall_cleans_the_claude_host_only(sandbox: Sandbox) -> None:
    assert _claude_setup(sandbox)[0] == 0
    assert _codex_setup(sandbox)[0] == 0
    codex_before = _read_json(sandbox.codex / "hooks.json")
    rc, lines = _run(["uninstall", "--keep-db"], sandbox)
    assert rc == 0
    unsetup = _CLAUDE_UNSETUP_LINES[:7] + [
        f"no rebuilder PreCompact hook in {_SETTINGS_FILE}",
        *_CLAUDE_UNSETUP_LINES[7:-1],
        _slash_line("removed", sandbox).replace(
            "slash command(s) ", "slash command(s) from ",
        ),
    ]
    assert lines == [
        "DB preserved at <TMP>/userhome/memory.db.",
        _STAMP_REMOVAL,
        *unsetup,
        "opted this host out of auto-install — `aelf` will no longer "
        "reinstall hooks at CLI entry (undo with `aelf setup`).",
        "",
        f"hook cleanup ran on: {_SETTINGS_FILE}",
        "the aelfrice package itself is still installed. Finish removing "
        "it with: uv tool uninstall aelfrice",
    ]
    assert _read_json(sandbox.settings) == _CLAUDE_SETTINGS_AFTER_REMOVAL
    assert _read_json(sandbox.opt_out) == {
        "opt_out": [], "opt_out_hosts": ["claude"],
    }
    # The codex host keeps its wiring and its skills.
    assert _read_json(sandbox.codex / "hooks.json") == codex_before
    assert sorted(p.name for p in sandbox.skills.iterdir()) == _skill_names()


@pytest.mark.timeout(120)
def test_claude_setup_clears_the_claude_opt_out(sandbox: Sandbox) -> None:
    # #1053: a claude setup is re-consent, so it drops the claude opt-out
    # that a codex setup writes.
    assert _codex_setup(sandbox)[0] == 0
    rc, lines = _claude_setup(sandbox)
    assert rc == 0
    assert lines[0] == "cleared claude host auto-install opt-out (#1053)"
    assert _read_json(sandbox.opt_out) == {"opt_out": []}


# --- codex host --------------------------------------------------------------

_HOOKS_FILE = "<TMP>/userhome/.codex/hooks.json"
_SKILLS_DIR = "<TMP>/userhome/.agents/skills"
_CODEX_EVENTS = (
    "PostCompact, PostToolUse, PreCompact, PreToolUse, SessionStart, Stop, "
    "UserPromptSubmit"
)


def _codex_handler(command: str) -> dict[str, object]:
    return {"type": "command", "command": command}


_CODEX_HOOKS = {
    "hooks": {
        "UserPromptSubmit": [
            {"hooks": [
                _codex_handler("aelf-hook"),
                _codex_handler("aelf-transcript-logger"),
            ]},
        ],
        "Stop": [
            {"hooks": [
                _codex_handler("aelf-transcript-logger"),
                _codex_handler("aelf-stop-hook"),
            ]},
        ],
        "PreCompact": [{"hooks": [_codex_handler("aelf-transcript-logger")]}],
        "PostCompact": [{"hooks": [_codex_handler("aelf-transcript-logger")]}],
        "SessionStart": [
            {
                "matcher": "startup|resume|clear|compact",
                "hooks": [_codex_handler("aelf-session-start-hook")],
            },
        ],
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    _codex_handler("aelf-search-tool-hook"),
                    _codex_handler("aelf-pre-issue-hook"),
                ],
            },
        ],
        "PostToolUse": [
            {"matcher": "Bash", "hooks": [_codex_handler("aelf-commit-ingest")]},
        ],
    },
}


def _codex_setup_lines() -> list[str]:
    return [
        f"wrote {_HOOKS_FILE} (events: {_CODEX_EVENTS})",
        "  next: Codex runs a hook only after per-hook trust approval: open a "
        "Codex session and run /hooks to approve the new entries (automation "
        "that vets its own hook sources may pass "
        "--dangerously-bypass-hook-trust instead).",
        "  next: The Codex `hooks` feature is stable and enabled by default; "
        "no action is needed unless you disabled it ([features].hooks = "
        "false in config.toml).",
        f"installed {len(_skill_names())} Codex skill(s) in {_SKILLS_DIR}: "
        "invoke as $" + ", $".join(_skill_names()),
        "  note: opted the claude host out of auto-install (#1053) — aelf "
        "will no longer rewrite ~/.claude/settings.json at CLI entry. Undo "
        "with `aelf setup` (default claude host).",
        "next step: this folder isn't in a git repository. Open a project "
        "first, then run `$aelf-onboard .` in Codex.",
    ]


def _codex_removal_lines() -> list[str]:
    return [
        f"removed aelfrice entries from {_HOOKS_FILE} (events: {_CODEX_EVENTS})",
        f"removed {len(_skill_names())} Codex skill(s) from {_SKILLS_DIR}: "
        + ", ".join(_skill_names()),
    ]


@pytest.mark.timeout(120)
def test_codex_setup_writes_the_pinned_hooks(sandbox: Sandbox) -> None:
    rc, lines = _codex_setup(sandbox)
    assert rc == 0
    assert lines == _codex_setup_lines()
    assert _normalize_hooks(_read_json(sandbox.codex / "hooks.json")) == _CODEX_HOOKS
    assert sorted(p.name for p in sandbox.skills.iterdir()) == _skill_names()
    assert _read_json(sandbox.opt_out) == {
        "opt_out": [], "opt_out_hosts": ["claude"],
    }
    # The codex host writes no Claude settings and no slash commands.
    assert not sandbox.settings.exists()
    assert not sandbox.slash_dir.exists()


@pytest.mark.timeout(120)
def test_codex_doctor_reports_the_pinned_lines(sandbox: Sandbox) -> None:
    assert _codex_setup(sandbox)[0] == 0
    rc, lines = _run(["doctor", "--host", "codex"], sandbox)
    assert rc == 0
    assert lines == [
        "[i] codex home: <TMP>/userhome/.codex",
        "[ok] codex hooks.json: present=True valid=True aelfrice_handlers=10",
        "[i] codex `hooks` feature: unknown; approved [hooks.state] entries: "
        "0 (all hooks on this host, not only aelfrice)",
        f"[i] codex skills ($aelf-*): {len(_skill_names())} installed "
        f"(0 orphaned) in {_SKILLS_DIR}",
        "[warn] 10 aelfrice handler(s) configured but no approved "
        "[hooks.state] entries exist — unapproved hooks are silently "
        "skipped; run /hooks in a Codex session to approve them",
    ]


@pytest.mark.timeout(120)
def test_codex_unsetup_removes_the_pinned_entries(sandbox: Sandbox) -> None:
    assert _codex_setup(sandbox)[0] == 0
    rc, lines = _run(["unsetup", "--host", "codex"], sandbox)
    assert rc == 0
    assert lines == _codex_removal_lines()
    assert _read_json(sandbox.codex / "hooks.json") == {"hooks": {}}
    assert list(sandbox.skills.iterdir()) == []


@pytest.mark.timeout(120)
def test_codex_uninstall_cleans_the_codex_host_only(sandbox: Sandbox) -> None:
    assert _claude_setup(sandbox)[0] == 0
    settings_before = _read_json(sandbox.settings)
    assert _codex_setup(sandbox)[0] == 0
    rc, lines = _run(["uninstall", "--keep-db", "--host", "codex"], sandbox)
    assert rc == 0
    assert lines == [
        "DB preserved at <TMP>/userhome/memory.db.",
        _STAMP_REMOVAL,
        *_codex_removal_lines(),
        "",
        "hook cleanup ran on the codex host; no Claude settings.json was "
        "touched (re-run without `--host` to clean the Claude host too).",
        "the aelfrice package itself is still installed. Finish removing "
        "it with: uv tool uninstall aelfrice",
    ]
    assert _read_json(sandbox.codex / "hooks.json") == {"hooks": {}}
    # The claude host keeps its settings, its slash commands, and its
    # auto-install: a codex teardown writes no claude opt-out.
    assert _read_json(sandbox.settings) == settings_before
    assert sorted(p.name for p in sandbox.slash_dir.iterdir()) == _slash_names()
    assert not sandbox.opt_out.exists() or _read_json(sandbox.opt_out).get(
        "opt_out_hosts", [],
    ) == []
