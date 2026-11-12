"""`aelf setup`, `unsetup`, `uninstall`, and `doctor` with `--host gemini` (#1812).

Each test drives `cli.main` in a sandbox home holding a Gemini directory,
a Claude settings file, and a Codex home, and checks that the gemini host
path writes and removes only its own files. The hook command resolves to a
file created in the test.
"""
from __future__ import annotations

import inspect
import io
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

import aelfrice.auto_install as auto_install
import aelfrice.cli as cli
import aelfrice.host_gemini as host_gemini
import aelfrice.setup as setup_mod
from aelfrice import launcher
from aelfrice.cli import build_parser, host_inapplicable_options, main

pytestmark = pytest.mark.timeout(60)

EVENTS = "AfterAgent, BeforeAgent, PreCompress, SessionStart"


@dataclass(frozen=True)
class Sandbox:
    root: Path
    home: Path
    hook: Path

    @property
    def gemini(self) -> Path:
        return self.home / ".gemini"

    @property
    def settings(self) -> Path:
        return self.gemini / "settings.json"

    @property
    def commands(self) -> Path:
        return self.gemini / "commands" / "aelf"

    @property
    def claude_settings(self) -> Path:
        return self.home / ".claude" / "settings.json"

    @property
    def codex_hooks(self) -> Path:
        return self.home / ".codex" / "hooks.json"

    @property
    def opt_out(self) -> Path:
        return self.home / ".aelfrice" / "opt-out-hooks.json"


@pytest.fixture
def sb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sandbox:
    home = tmp_path / "userhome"
    (home / ".gemini").mkdir(parents=True)
    (home / ".codex").mkdir()
    project = tmp_path / "project"
    project.mkdir()
    hook = tmp_path / "bin" / "aelf-gemini-hook"
    hook.parent.mkdir()
    hook.write_text("", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setenv("AELFRICE_DB", str(home / "memory.db"))
    monkeypatch.delenv("GEMINI_CLI_HOME", raising=False)
    dotdir = home / ".aelfrice"
    monkeypatch.setattr(auto_install, "AELFRICE_DOTDIR", dotdir)
    monkeypatch.setattr(auto_install, "OPT_OUT_PATH", dotdir / "opt-out-hooks.json")
    monkeypatch.setattr(
        auto_install, "STAMP_PATH", dotdir / "installed-manifest-version",
    )
    monkeypatch.setattr(
        setup_mod, "USER_SETTINGS_PATH", home / ".claude" / "settings.json",
    )
    monkeypatch.setattr(
        setup_mod, "SLASH_COMMANDS_DIR_DEFAULT", home / ".claude" / "commands" / "aelf",
    )
    monkeypatch.setattr(cli, "_CLAUDE_PROJECTS_DIR", home / ".claude" / "projects")
    monkeypatch.setattr(
        host_gemini, "resolve_gemini_hook_command", lambda _scope: str(hook),
    )
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    return Sandbox(root=tmp_path, home=home, hook=hook)


def _run(argv: list[str], sb: Sandbox) -> tuple[int, list[str]]:
    out = io.StringIO()
    rc = main(argv, out)
    text = out.getvalue().replace(str(sb.root), "<TMP>").replace("\\", "/")
    return rc, text.splitlines()


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _n_commands() -> int:
    return len(host_gemini.bundled_gemini_commands())


# --- setup ------------------------------------------------------------------------


def test_setup_writes_the_hooks_and_commands(sb: Sandbox) -> None:
    rc, lines = _run(["setup", "--host", "gemini"], sb)
    assert rc == 0
    *lines, note = lines
    assert lines == [
        f"wrote <TMP>/userhome/.gemini/settings.json (events: {EVENTS})",
        "  next: restart Gemini CLI to load the hooks. Gemini runs them "
        "while hooksConfig.enabled is true, which "
        "is the default.",
        f"installed {_n_commands()} Gemini command(s) in "
        "<TMP>/userhome/.gemini/commands/aelf: invoke as /aelf:<name>",
    ]
    # The #1053 tests own the wording of the consent note.
    assert note.startswith(
        "  note: opted the claude host out of auto-install (#1053)",
    )
    assert _read(sb.settings) == {
        "hooks": host_gemini.desired_gemini_hooks(str(sb.hook)),
    }
    assert len(list(sb.commands.glob("*.toml"))) == _n_commands()


def test_setup_writes_nothing_for_another_host(sb: Sandbox) -> None:
    _run(["setup", "--host", "gemini"], sb)
    assert not sb.claude_settings.exists()
    assert not (sb.home / ".claude" / "commands").exists()
    assert list((sb.home / ".codex").iterdir()) == []
    assert not (sb.home / ".agents").exists()


def test_a_second_setup_changes_nothing(sb: Sandbox) -> None:
    _run(["setup", "--host", "gemini"], sb)
    before = _files(sb.gemini)
    rc, lines = _run(["setup", "--host", "gemini"], sb)
    assert rc == 0
    assert lines[0] == (
        f"already current: <TMP>/userhome/.gemini/settings.json (events: {EVENTS})"
    )
    assert lines[2] == (
        "Gemini commands already up to date in <TMP>/userhome/.gemini/commands/aelf"
    )
    assert _files(sb.gemini) == before


def test_setup_leaves_claude_auto_install_on_a_dual_host_machine(sb: Sandbox) -> None:
    """The #1053 consent rule, shared with the codex host."""
    sb.claude_settings.parent.mkdir(parents=True)
    sb.claude_settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [
        {"hooks": [{"type": "command", "command": "/x/bin/aelf-hook"}]},
    ]}}), encoding="utf-8")
    rc, lines = _run(["setup", "--host", "gemini"], sb)
    assert rc == 0
    assert lines[-1] == (
        "  note: Claude-host aelfrice hooks detected; leaving Claude "
        "auto-install enabled (dual-host)."
    )
    assert not sb.opt_out.exists()


def test_setup_refuses_a_gemini_cli_home_that_does_not_exist(
    sb: Sandbox, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    missing = sb.root / "nowhere"
    monkeypatch.setenv("GEMINI_CLI_HOME", str(missing))
    rc, _ = _run(["setup", "--host", "gemini"], sb)
    assert rc == 1
    assert "GEMINI_CLI_HOME" in capsys.readouterr().err
    assert not missing.exists()
    assert not sb.settings.exists()


def test_gemini_cli_home_moves_the_install(
    sb: Sandbox, monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = sb.root / "otherhome"
    other.mkdir()
    monkeypatch.setenv("GEMINI_CLI_HOME", str(other))
    assert _run(["setup", "--host", "gemini"], sb)[0] == 0
    assert (other / ".gemini" / "settings.json").is_file()
    assert not sb.settings.exists()


def test_setup_refuses_an_unparseable_settings_file(
    sb: Sandbox, capsys: pytest.CaptureFixture[str],
) -> None:
    sb.settings.write_text('{\n  // mine\n  "theme": "x"\n}\n', encoding="utf-8")
    assert _run(["setup", "--host", "gemini"], sb)[0] == 1
    assert "not plain JSON" in capsys.readouterr().err
    assert not sb.commands.exists()
    assert _run(["setup", "--host", "gemini", "--force"], sb)[0] == 0
    assert "hooks" in _read(sb.settings)


# --- unsetup and uninstall -------------------------------------------------------


def _other_hosts(sb: Sandbox) -> dict[str, bytes]:
    """Claude and codex wiring that a gemini teardown must leave alone."""
    sb.claude_settings.parent.mkdir(parents=True, exist_ok=True)
    sb.claude_settings.write_text(json.dumps({"hooks": {"UserPromptSubmit": [
        {"hooks": [{"type": "command", "command": "/x/bin/aelf-hook"}]},
    ]}}), encoding="utf-8")
    sb.codex_hooks.write_text(json.dumps({"hooks": {"Stop": [
        {"hooks": [{"type": "command", "command": "/x/bin/aelf-stop-hook"}]},
    ]}}), encoding="utf-8")
    return {
        "claude": sb.claude_settings.read_bytes(),
        "codex": sb.codex_hooks.read_bytes(),
    }


def test_unsetup_removes_exactly_what_setup_added(sb: Sandbox) -> None:
    original = {"theme": "widget", "hooks": {"BeforeTool": [
        {"matcher": "write_file", "hooks": [
            {"type": "command", "command": "/opt/tools/other-hook"},
        ]},
    ]}}
    sb.settings.write_text(json.dumps(original), encoding="utf-8")
    other = _other_hosts(sb)
    _run(["setup", "--host", "gemini"], sb)
    rc, lines = _run(["unsetup", "--host", "gemini"], sb)
    assert rc == 0
    assert lines == [
        "removed aelfrice hooks from <TMP>/userhome/.gemini/settings.json "
        f"(events: {EVENTS})",
        f"removed {_n_commands()} Gemini command(s) from "
        "<TMP>/userhome/.gemini/commands/aelf",
    ]
    assert _read(sb.settings) == original
    assert not sb.commands.exists()
    assert sb.claude_settings.read_bytes() == other["claude"]
    assert sb.codex_hooks.read_bytes() == other["codex"]


def test_setup_then_uninstall_leaves_the_gemini_dir_as_it_was(sb: Sandbox) -> None:
    """Killed by: leaving `{}`, a lock file, or an empty commands dir."""
    rc = _run(["setup", "--host", "gemini"], sb)[0]
    assert rc == 0
    rc = _run(["uninstall", "--keep-db", "--host", "gemini"], sb)[0]
    assert rc == 0
    assert list(sb.gemini.iterdir()) == []


def test_unsetup_on_a_machine_without_gemini_creates_nothing(
    sb: Sandbox,
) -> None:
    sb.gemini.rmdir()
    rc, lines = _run(["unsetup", "--host", "gemini"], sb)
    assert rc == 0
    assert lines == [
        "no aelfrice hooks in <TMP>/userhome/.gemini/settings.json",
        "no aelfrice Gemini commands in <TMP>/userhome/.gemini/commands/aelf",
    ]
    assert not sb.gemini.exists()


def test_uninstall_tears_down_the_named_host_only(sb: Sandbox) -> None:
    """#1332: the claude opt-out is written for a claude teardown only."""
    other = _other_hosts(sb)
    _run(["setup", "--host", "gemini"], sb)
    opt_out_before = sb.opt_out.read_bytes() if sb.opt_out.exists() else None
    rc, lines = _run(["uninstall", "--keep-db", "--host", "gemini"], sb)
    assert rc == 0
    assert (
        "hook cleanup ran on the gemini host; no Claude settings.json was "
        "touched (re-run without `--host` to clean the Claude host too)."
    ) in lines
    assert not sb.settings.exists()
    assert not sb.commands.exists()
    assert sb.claude_settings.read_bytes() == other["claude"]
    assert sb.codex_hooks.read_bytes() == other["codex"]
    after = sb.opt_out.read_bytes() if sb.opt_out.exists() else None
    assert after == opt_out_before


# --- doctor -----------------------------------------------------------------------


def test_doctor_on_a_clean_install(
    sb: Sandbox, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run(["setup", "--host", "gemini"], sb)
    monkeypatch.setattr(launcher, "which_on_path", lambda name: sb.hook)
    rc, lines = _run(["doctor", "--host", "gemini"], sb)
    assert rc == 0
    assert lines == [
        "[i] gemini dir: <TMP>/userhome/.gemini",
        "[ok] gemini settings.json: present=True valid=True aelfrice_handlers=4",
        f"[i] gemini commands (/aelf:*): {_n_commands()} installed (0 orphaned)",
    ]


@pytest.fixture
def no_aelf_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """No `aelf` on PATH, so a doctor check that ignores the #1430 gate
    fails here whatever the machine running the test has installed."""
    monkeypatch.setattr(launcher, "which_on_path", lambda name: None)


@pytest.mark.usefixtures("no_aelf_on_path")
def test_doctor_without_wiring_exits_0(sb: Sandbox) -> None:
    rc, lines = _run(["doctor", "--host", "gemini"], sb)
    assert rc == 0
    assert lines[1] == (
        "[-] gemini settings.json: present=False valid=False aelfrice_handlers=0"
    )
    assert not any(line.startswith("[FAIL]") for line in lines)


def test_doctor_fails_without_aelf_on_path(
    sb: Sandbox, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run(["setup", "--host", "gemini"], sb)
    monkeypatch.setattr(launcher, "which_on_path", lambda name: None)
    rc, lines = _run(["doctor", "--host", "gemini"], sb)
    assert rc == 1
    assert any("`aelf` is not on PATH" in line for line in lines)


def test_doctor_fails_on_an_edited_command(
    sb: Sandbox, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run(["setup", "--host", "gemini"], sb)
    monkeypatch.setattr(launcher, "which_on_path", lambda name: sb.hook)
    edited = sb.commands / "search.toml"
    edited.write_text(edited.read_text(encoding="utf-8") + "# edit\n", encoding="utf-8")
    rc, lines = _run(["doctor", "--host", "gemini"], sb)
    assert rc == 1
    assert any(
        line.startswith("[FAIL] installed /aelf:* command(s) do not match")
        and line.endswith("search.toml; re-run `aelf setup --host gemini`")
        for line in lines
    )


@pytest.mark.usefixtures("no_aelf_on_path")
def test_doctor_passes_a_commented_settings_file_without_aelfrice(sb: Sandbox) -> None:
    """A machine that never ran setup exits 0, even when Gemini's own file
    holds a comment aelfrice can't parse."""
    sb.settings.write_text('{\n  // my theme\n  "theme": "Dracula"\n}', encoding="utf-8")
    rc, lines = _run(["doctor", "--host", "gemini"], sb)
    assert rc == 0
    assert lines[1] == (
        "[-] gemini settings.json: present=True valid=False aelfrice_handlers=0"
    )
    assert any(line.startswith("[warn] ") and "as JSON" in line for line in lines)
    assert not any(line.startswith("[FAIL]") for line in lines)


def test_doctor_fails_on_an_unparseable_file_that_names_the_hook(sb: Sandbox) -> None:
    _run(["setup", "--host", "gemini"], sb)
    text = sb.settings.read_text(encoding="utf-8")
    sb.settings.write_text(text.replace("{", "{\n  // mine", 1), encoding="utf-8")
    rc, lines = _run(["doctor", "--host", "gemini"], sb)
    assert rc == 1
    assert any(
        line.startswith("[FAIL] ") and "can't check the aelfrice hooks" in line
        for line in lines
    )


@pytest.mark.usefixtures("no_aelf_on_path")
def test_doctor_skips_the_graph_checks(
    sb: Sandbox, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli, "_cmd_doctor_classify_orphans",
        lambda _a, _o: pytest.fail("the full doctor ran"),
    )
    assert _run(["doctor", "--host", "gemini", "--classify-orphans"], sb)[0] == 0


# --- options the gemini path never reads (#1429) ----------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["setup", "--host", "gemini", "--scope", "user"],
        ["setup", "--host", "gemini", "--no-codex-skills"],
        ["setup", "--host", "gemini", "--rebuilder"],
        ["unsetup", "--host", "gemini", "--command", "x"],
    ],
)
def test_an_inapplicable_option_exits_2_before_any_write(
    sb: Sandbox, argv: list[str], capsys: pytest.CaptureFixture[str],
) -> None:
    rc, lines = _run(argv, sb)
    assert rc == 2
    assert lines == []
    err = capsys.readouterr().err
    assert f"{argv[0]} --host gemini: refusing {argv[3]}" in err
    assert "; the gemini host path never reads these options" in err
    assert "#1429" in err
    assert not sb.settings.exists()
    assert not sb.commands.exists()


def test_the_applicable_sets_partition_the_parser() -> None:
    parser = build_parser()
    for cmd, applicable in (("setup", {"host", "force"}), ("unsetup", {"host"})):
        sub = cli._subcommand_parser(parser, cmd)  # noqa: SLF001
        assert sub is not None
        every = set(cli._option_dests(sub))  # noqa: SLF001
        assert set(host_inapplicable_options(parser, cmd, "gemini")) == (
            every - applicable
        )


def test_the_claude_host_has_no_inapplicable_options() -> None:
    parser = build_parser()
    assert host_inapplicable_options(parser, "setup", "claude") == {}


@pytest.mark.source_scan
def test_the_applicable_sets_are_the_gemini_read_sets() -> None:
    setup_src = inspect.getsource(cli._cmd_setup_gemini)  # noqa: SLF001
    assert re.findall(r'getattr\(args, "(\w+)"', setup_src) == ["force"]
    unsetup_src = inspect.getsource(cli._cmd_unsetup_gemini)  # noqa: SLF001
    assert re.search(r"^\s+_ = args$", unsetup_src, re.M)
    assert "getattr(args" not in unsetup_src


def _normalize(block: str) -> str:
    return " ".join(re.sub(r"[`\-]", " ", block).split()).lower()


def test_generated_commands_never_instruct_a_refused_option_unqualified() -> None:
    """A refused option named in a command prompt is marked as another
    host's guidance, as the codex skills are (#1429)."""
    parser = build_parser()
    refused: set[str] = set()
    for cmd in ("setup", "unsetup"):
        sub = cli._subcommand_parser(parser, cmd)  # noqa: SLF001
        assert sub is not None
        dests = set(host_inapplicable_options(parser, cmd, "gemini"))
        for action in sub._actions:  # noqa: SLF001
            if action.dest in dests:
                refused.update(action.option_strings)
    offenders: list[tuple[str, str]] = []
    for name, text in sorted(host_gemini.bundled_gemini_commands().items()):
        for block in text.split("\n\n"):
            named = sorted(opt for opt in refused if opt in block)
            if not named:
                continue
            norm = _normalize(block)
            if "claude host" not in norm and "codex host" not in norm:
                offenders.append((name, ", ".join(named)))
    assert offenders == []
