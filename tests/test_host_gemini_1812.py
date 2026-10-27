"""Gemini settings hooks, custom commands, and the doctor scan (#1812).

Every file is under `tmp_path`, and every settings document is synthetic.
The hook command is a file created in the test, so nothing resolves a real
install.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import tempfile
import tomllib
from pathlib import Path

import pytest

import aelfrice.host_gemini as hgm
from aelfrice import launcher
from aelfrice.file_lock import HAVE_ADVISORY_LOCKS, try_lock_exclusive
from aelfrice.setup import bundled_slash_files

pytestmark = pytest.mark.timeout(60)

EVENTS = ["AfterAgent", "AfterTool", "BeforeAgent", "PreCompress", "SessionStart"]


@pytest.fixture(autouse=True)
def _scratch_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The settings lock goes to the temporary directory; keep it here."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))


@pytest.fixture
def cmd(tmp_path: Path) -> str:
    exe = tmp_path / "bin" / "aelf-gemini-hook"
    exe.parent.mkdir()
    exe.write_text("", encoding="utf-8")
    return str(exe)


@pytest.fixture
def gdir(tmp_path: Path) -> Path:
    path = tmp_path / "home" / ".gemini"
    path.mkdir(parents=True)
    return path


def _install(path: Path, cmd: str, *, force: bool = False) -> hgm.GeminiSettingsResult:
    return hgm.install_gemini_hooks(path, force=force, command=cmd)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, doc: object) -> None:
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _ours(cmd: str, event: str) -> dict[str, object]:
    return hgm.desired_gemini_hooks(cmd)[event][0]


_FOREIGN_HANDLER = {"type": "command", "command": "/opt/tools/other-hook", "name": "other"}
_FOREIGN_GROUP = {"matcher": "write_file", "hooks": [_FOREIGN_HANDLER]}


# --- the hook set -------------------------------------------------------------


def test_the_hook_set_covers_five_events_in_milliseconds(cmd: str) -> None:
    desired = hgm.desired_gemini_hooks(cmd)
    assert sorted(desired) == EVENTS
    specs = {
        event: (
            groups[0]["matcher"],
            groups[0]["hooks"][0]["name"],  # type: ignore[index]
            groups[0]["hooks"][0]["timeout"],  # type: ignore[index]
        )
        for event, groups in desired.items()
    }
    assert specs == {
        "SessionStart": ("*", "aelfrice-session-start", 15000),
        "BeforeAgent": ("*", "aelfrice-prompt", 45000),
        "AfterAgent": ("*", "aelfrice-turn-end", 60000),
        "PreCompress": ("*", "aelfrice-pre-compress", 30000),
        "AfterTool": ("^save_memory$", "aelfrice-memory-mirror", 30000),
    }
    for groups in desired.values():
        assert groups[0]["hooks"] == [{
            "type": "command", "command": cmd,
            "name": groups[0]["hooks"][0]["name"],  # type: ignore[index]
            "timeout": groups[0]["hooks"][0]["timeout"],  # type: ignore[index]
        }]


def _commands(path: Path) -> set[str]:
    hooks = _read(path)["hooks"]
    assert isinstance(hooks, dict)
    return {
        handler["command"]
        for groups in hooks.values()
        for group in groups
        for handler in group["hooks"]
    }


def test_setup_writes_the_path_it_resolves_at_install_time(
    gdir: Path, cmd: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: writing the bare script name instead of resolving it."""
    monkeypatch.setattr(
        launcher, "which_on_path",
        lambda name: Path(cmd) if name == "aelf-gemini-hook" else None,
    )
    path = gdir / "settings.json"
    hgm.install_gemini_hooks(path)
    assert _commands(path) == {cmd}


def _spaced_hook(tmp_path: Path) -> Path:
    name = "aelf-gemini-hook.exe" if os.name == "nt" else "aelf-gemini-hook"
    exe = tmp_path / "my bin" / name
    exe.parent.mkdir()
    exe.write_text("", encoding="utf-8")
    return exe


def test_a_spaced_hook_path_is_one_word_to_gemini_shell(
    gdir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: writing the resolved path unquoted.

    Gemini runs the command under `bash -c`, which splits a bare path at
    the space and runs `.../my`.
    """
    exe = _spaced_hook(tmp_path)
    monkeypatch.setattr(
        launcher, "which_on_path",
        lambda name: exe if name == "aelf-gemini-hook" else None,
    )
    path = gdir / "settings.json"
    hgm.install_gemini_hooks(path)
    (command,) = _commands(path)
    if os.name == "nt":
        assert command == f"& '{exe}'"
    else:
        assert shlex.split(command) == [str(exe)]
    report = hgm.doctor_gemini(gdir)
    assert report.owned_handler_count == 5
    assert report.faults() == []


def test_a_spaced_windows_path_is_quoted_for_powershell() -> None:
    exe = "C:\\Users\\First Last\\.local\\bin\\aelf-gemini-hook.exe"
    command = hgm.gemini_hook_command(exe, windows=True)
    assert command == f"& '{exe}'"
    assert hgm.shell_program(command, windows=True) == exe
    assert hgm._handler_is_owned({"command": command}, windows=True)  # noqa: SLF001
    odd = "C:\\tools\\it's\\bin\\aelf-gemini-hook.exe"
    quoted = hgm.gemini_hook_command(odd, windows=True)
    # PowerShell reads two single quotes inside a single-quoted string as one.
    assert quoted == "& 'C:\\tools\\it''s\\bin\\aelf-gemini-hook.exe'"
    assert hgm.shell_program(quoted, windows=True) == odd


@pytest.mark.parametrize(
    ("program", "windows"),
    [
        ("/usr/local/bin/aelf-gemini-hook", False),
        ("C:\\Users\\w\\.local\\bin\\aelf-gemini-hook.exe", True),
    ],
)
def test_a_plain_hook_path_is_written_unchanged(program: str, windows: bool) -> None:
    assert hgm.gemini_hook_command(program, windows=windows) == program


# --- install -------------------------------------------------------------------


def test_install_into_a_missing_file(gdir: Path, cmd: str) -> None:
    path = gdir / "settings.json"
    result = _install(path, cmd)
    assert (result.changed, result.error, result.events) == (True, None, EVENTS)
    assert _read(path) == {"hooks": hgm.desired_gemini_hooks(cmd)}


def test_install_is_idempotent(gdir: Path, cmd: str) -> None:
    path = gdir / "settings.json"
    _install(path, cmd)
    before = path.read_bytes()
    result = _install(path, cmd)
    assert (result.changed, result.error) == (False, None)
    assert path.read_bytes() == before


def test_install_keeps_foreign_settings_and_handlers(gdir: Path, cmd: str) -> None:
    path = gdir / "settings.json"
    original = {
        "theme": "widget",
        "hooksConfig": {"notifications": False},
        "hooks": {
            "BeforeAgent": [{"matcher": "*", "hooks": [_FOREIGN_HANDLER]}],
            "BeforeTool": [_FOREIGN_GROUP],
        },
    }
    _write(path, original)
    _install(path, cmd)
    doc = _read(path)
    assert doc["theme"] == "widget"
    assert doc["hooksConfig"] == {"notifications": False}
    hooks = doc["hooks"]
    assert isinstance(hooks, dict)
    assert hooks["BeforeTool"] == [_FOREIGN_GROUP]
    assert hooks["BeforeAgent"] == [
        {"matcher": "*", "hooks": [_FOREIGN_HANDLER]},
        _ours(cmd, "BeforeAgent"),
    ]


def test_a_reinstall_replaces_an_older_handler_in_a_mixed_group(
    gdir: Path, cmd: str,
) -> None:
    """Killed by: deciding ownership per group instead of per handler."""
    path = gdir / "settings.json"
    old = {"type": "command", "command": "/old/venv/bin/aelf-gemini-hook"}
    _write(path, {"hooks": {"AfterAgent": [
        {"matcher": "*", "hooks": [_FOREIGN_HANDLER, old]},
    ]}})
    _install(path, cmd)
    hooks = _read(path)["hooks"]
    assert isinstance(hooks, dict)
    assert hooks["AfterAgent"] == [
        {"matcher": "*", "hooks": [_FOREIGN_HANDLER]},
        _ours(cmd, "AfterAgent"),
    ]


def test_a_claude_host_handler_is_not_claimed(gdir: Path, cmd: str) -> None:
    """Only `aelf-gemini-hook` is ours here, even among aelfrice scripts."""
    path = gdir / "settings.json"
    claude_style = {"type": "command", "command": "/x/bin/aelf-hook"}
    _write(path, {"hooks": {"BeforeAgent": [{"hooks": [claude_style]}]}})
    _install(path, cmd)
    hgm.remove_gemini_hooks(path)
    assert _read(path) == {"hooks": {"BeforeAgent": [{"hooks": [claude_style]}]}}


def test_a_windows_launcher_is_owned() -> None:
    handler = {
        "type": "command",
        "command": "C:\\Users\\w\\.local\\bin\\aelf-gemini-hook.EXE",
    }
    assert hgm._handler_is_owned(handler, windows=True)  # noqa: SLF001
    assert not hgm._handler_is_owned(  # noqa: SLF001
        {"command": "C:\\tools\\other.exe"}, windows=True,
    )


@pytest.mark.parametrize(
    ("content", "needle"),
    [
        ('{\n  // a comment\n  "theme": "x"\n}\n', "not plain JSON"),
        ("[1, 2]\n", "not a JSON object"),
        ('{"hooks": null}\n', "non-object `hooks` value (null)"),
        ('{"hooks": 42}\n', "non-object `hooks` value (number)"),
        ('{"hooks": {"BeforeAgent": {"x": 1}}}\n', "`BeforeAgent` event (object)"),
        ('{"hooks": {"AfterTool": null}}\n', "`AfterTool` event (null)"),
    ],
)
def test_install_refuses_what_it_would_reshape(
    gdir: Path, cmd: str, content: str, needle: str,
) -> None:
    path = gdir / "settings.json"
    path.write_text(content, encoding="utf-8")
    result = _install(path, cmd)
    assert result.error is not None and needle in result.error
    assert path.read_text(encoding="utf-8") == content
    forced = _install(path, cmd, force=True)
    assert forced.error is None
    hooks = _read(path)["hooks"]
    assert isinstance(hooks, dict)
    assert set(EVENTS) <= set(hooks)


# --- remove ---------------------------------------------------------------------


def test_remove_undoes_install_exactly(gdir: Path, cmd: str) -> None:
    """Install then remove gives back the document setup started from."""
    path = gdir / "settings.json"
    original = {
        "theme": "widget",
        "hooks": {
            "BeforeAgent": [{"matcher": "*", "hooks": [_FOREIGN_HANDLER]}],
            "BeforeTool": [_FOREIGN_GROUP],
        },
    }
    _write(path, original)
    _install(path, cmd)
    result = hgm.remove_gemini_hooks(path)
    assert (result.changed, result.events) == (True, EVENTS)
    assert _read(path) == original


def test_remove_drops_a_hooks_object_it_empties(gdir: Path, cmd: str) -> None:
    """Killed by: leaving `"hooks": {}` behind in a file that had none."""
    path = gdir / "settings.json"
    _write(path, {"theme": "widget"})
    _install(path, cmd)
    hgm.remove_gemini_hooks(path)
    assert _read(path) == {"theme": "widget"}


def test_no_lock_file_is_left_in_the_gemini_dir(gdir: Path, cmd: str) -> None:
    """Killed by: locking a sibling `settings.json.lock` again."""
    path = gdir / "settings.json"
    _install(path, cmd)
    hgm.remove_gemini_hooks(path)
    assert sorted(p.name for p in gdir.iterdir()) == ["settings.json"]


def test_the_lock_is_keyed_by_the_settings_path(tmp_path: Path) -> None:
    one = hgm.settings_lock_anchor(tmp_path / "a" / ".gemini" / "settings.json")
    two = hgm.settings_lock_anchor(tmp_path / "b" / ".gemini" / "settings.json")
    assert one != two
    assert one.parent == Path(tempfile.gettempdir())


@pytest.mark.skipif(
    not HAVE_ADVISORY_LOCKS, reason="no advisory locking on this filesystem",
)
def test_the_lock_is_held_across_the_read_modify_write(
    gdir: Path, cmd: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second opener can't take the lock while the plan runs.

    `flock` is held per open file description, so a re-open in this
    process contends as another process would.
    """
    path = gdir / "settings.json"
    _write(path, {"theme": "widget"})
    anchor = hgm.settings_lock_anchor(path)
    lock_path = anchor.with_name(anchor.name + ".lock")
    observed: list[bool] = []
    real = Path.read_bytes

    def probing(self: Path) -> bytes:
        if self == path:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            try:
                try_lock_exclusive(fd)
                observed.append(False)
            except OSError:
                observed.append(True)
            finally:
                os.close(fd)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", probing)
    assert _install(path, cmd).error is None
    assert observed and all(observed)


def test_remove_without_a_file_creates_nothing(tmp_path: Path) -> None:
    gdir = tmp_path / "absent" / ".gemini"
    result = hgm.remove_gemini_hooks(gdir / "settings.json")
    assert (result.changed, result.error) == (False, None)
    assert not (tmp_path / "absent").exists()
    # Not even a lock file in the temporary directory.
    assert list(Path(tempfile.gettempdir()).iterdir()) == []


def test_remove_leaves_an_unparseable_file_alone(gdir: Path) -> None:
    path = gdir / "settings.json"
    path.write_text("{ not json", encoding="utf-8")
    result = hgm.remove_gemini_hooks(path)
    assert result.error is not None
    assert path.read_text(encoding="utf-8") == "{ not json"


def test_remove_with_nothing_of_ours_changes_nothing(gdir: Path) -> None:
    path = gdir / "settings.json"
    _write(path, {"hooks": {"BeforeTool": [_FOREIGN_GROUP]}})
    before = path.read_bytes()
    result = hgm.remove_gemini_hooks(path)
    assert (result.changed, result.error) == (False, None)
    assert path.read_bytes() == before


# --- custom commands ---------------------------------------------------------------


def test_every_slash_command_becomes_one_toml_command() -> None:
    bundle = hgm.bundled_gemini_commands()
    assert sorted(bundle) == sorted(
        name.removesuffix(".md") + ".toml" for name in bundled_slash_files()
    )
    for name, text in bundle.items():
        assert text.startswith(f"# {hgm.COMMAND_MARKER}:"), name
        parsed = tomllib.loads(text)
        assert set(parsed) == {"description", "prompt"}, name
        prompt = parsed["prompt"]
        assert isinstance(prompt, str) and prompt.strip(), name
        assert "$ARGUMENTS" not in prompt, name
        assert "uv run aelf" not in prompt, name
        # `!{...}` runs a shell command and `@{...}` inlines a file in a
        # Gemini command prompt; `{{args}}` is the only template allowed.
        assert "!{" not in prompt and "@{" not in prompt, name
        assert re.sub(r"\{\{args\}\}", "", prompt).count("{{") == 0, name


def test_arguments_become_the_gemini_placeholder() -> None:
    sources = bundled_slash_files()
    bundle = hgm.bundled_gemini_commands()
    for filename, text in sources.items():
        prompt = tomllib.loads(bundle[filename.removesuffix(".md") + ".toml"])["prompt"]
        assert ("$ARGUMENTS" in text) == ("{{args}}" in prompt), filename
    search = tomllib.loads(bundle["search.toml"])["prompt"]
    assert 'aelf search "{{args}}"' in search


@pytest.mark.parametrize("name", ["setup", "doctor", "uninstall", "upgrade"])
def test_host_commands_are_steered_to_the_gemini_host(name: str) -> None:
    prompt = tomllib.loads(hgm.bundled_gemini_commands()[f"{name}.toml"])["prompt"]
    assert "--host gemini" in prompt


def test_fan_out_commands_point_at_gemini_sub_agents() -> None:
    """A source that lists the `Task` tool gets the note; others don't."""
    bundle = hgm.bundled_gemini_commands()
    for filename, text in bundled_slash_files().items():
        prompt = tomllib.loads(bundle[filename.removesuffix(".md") + ".toml"])["prompt"]
        assert ("Gemini's sub-agents" in prompt) == ("Task" in text), filename
    assert "Gemini's sub-agents" in tomllib.loads(bundle["onboard.toml"])["prompt"]


def test_other_commands_carry_no_host_note() -> None:
    prompt = tomllib.loads(hgm.bundled_gemini_commands()["search.toml"])["prompt"]
    assert "--host gemini" not in prompt


def test_the_setup_command_describes_the_gemini_install() -> None:
    parsed = tomllib.loads(hgm.bundled_gemini_commands()["setup.toml"])
    assert "Gemini's user settings.json" in parsed["description"]


@pytest.mark.parametrize(
    "value",
    ["one line", "two\nlines", "it's '''quoted'''\nhere", "tab\there\n", "cr\r\nline"],
)
def test_toml_strings_round_trip(value: str) -> None:
    doc = tomllib.loads(f"v = {hgm._toml_string(value)}\n")  # noqa: SLF001
    assert doc["v"] == value


def test_the_generation_is_deterministic() -> None:
    assert hgm.bundled_gemini_commands() == hgm.bundled_gemini_commands()


def test_install_and_remove_commands(gdir: Path) -> None:
    target = gdir / "commands" / "aelf"
    first = hgm.install_gemini_commands(gdir)
    assert first.dest_dir == target
    assert sorted(first.written) == sorted(hgm.bundled_gemini_commands())
    second = hgm.install_gemini_commands(gdir)
    assert (second.written, len(second.already)) == ((), len(first.written))
    removed = hgm.remove_gemini_commands(gdir)
    assert sorted(removed.pruned) == sorted(first.written)
    assert not target.exists()
    assert (gdir / "commands").is_dir()


def test_a_user_command_is_never_overwritten_or_removed(gdir: Path) -> None:
    """Killed by: dropping the marker check on replace or on remove."""
    target = gdir / "commands" / "aelf"
    target.mkdir(parents=True)
    (target / "search.toml").write_text('prompt = "mine"\n', encoding="utf-8")
    (target / "mine.toml").write_text('prompt = "also mine"\n', encoding="utf-8")
    result = hgm.install_gemini_commands(gdir)
    assert result.skipped == ("search.toml",)
    assert (target / "search.toml").read_text(encoding="utf-8") == 'prompt = "mine"\n'
    hgm.remove_gemini_commands(gdir)
    assert sorted(p.name for p in target.iterdir()) == ["mine.toml", "search.toml"]


def test_orphans_and_edits_are_reported_and_pruned(gdir: Path) -> None:
    target = gdir / "commands" / "aelf"
    hgm.install_gemini_commands(gdir)
    orphan = target / "retired-widget.toml"
    orphan.write_text(f"# {hgm.COMMAND_MARKER}: old\nprompt = \"x\"\n", encoding="utf-8")
    edited = target / "search.toml"
    edited.write_text(edited.read_text(encoding="utf-8") + "# edit\n", encoding="utf-8")
    assert hgm.orphaned_gemini_commands(gdir) == ["retired-widget.toml"]
    assert hgm.modified_gemini_commands(gdir) == ["search.toml"]
    result = hgm.install_gemini_commands(gdir)
    assert result.pruned == ("retired-widget.toml",)
    assert result.written == ("search.toml",)
    assert hgm.orphaned_gemini_commands(gdir) == []
    assert hgm.modified_gemini_commands(gdir) == []


# --- doctor -------------------------------------------------------------------------


def test_doctor_without_a_gemini_dir(tmp_path: Path) -> None:
    report = hgm.doctor_gemini(tmp_path / "none" / ".gemini")
    assert not report.gemini_dir_present
    assert report.faults() == []
    assert report.warnings


def test_doctor_on_a_clean_install(gdir: Path, cmd: str) -> None:
    _install(gdir / "settings.json", cmd)
    report = hgm.doctor_gemini(gdir)
    assert (report.settings_valid, report.owned_handler_count) == (True, 5)
    assert report.missing_events == []
    assert report.faults() == []


def test_doctor_on_absent_wiring_is_not_a_fault(gdir: Path) -> None:
    _write(gdir / "settings.json", {"theme": "widget"})
    report = hgm.doctor_gemini(gdir)
    assert report.owned_handler_count == 0
    assert report.faults() == []
    assert any("aelf setup --host gemini" in w for w in report.warnings)


def test_doctor_faults_a_missing_event(gdir: Path, cmd: str) -> None:
    path = gdir / "settings.json"
    _install(path, cmd)
    doc = _read(path)
    del doc["hooks"]["AfterAgent"]  # type: ignore[attr-defined]
    _write(path, doc)
    report = hgm.doctor_gemini(gdir)
    assert report.missing_events == ["AfterAgent"]
    assert any("missing events: AfterAgent" in f for f in report.faults())


def test_doctor_faults_a_command_not_on_disk(gdir: Path, tmp_path: Path) -> None:
    gone = str(tmp_path / "gone" / "aelf-gemini-hook")
    _install(gdir / "settings.json", gone)
    report = hgm.doctor_gemini(gdir)
    assert report.stale_commands == [gone]
    assert any("not present on disk" in f for f in report.faults())


def test_doctor_faults_an_unquoted_spaced_command(gdir: Path, tmp_path: Path) -> None:
    """Killed by: accepting a command because some rejoined reading of it
    exists, when the shell splits it at the space."""
    exe = str(_spaced_hook(tmp_path))
    _install(gdir / "settings.json", exe)
    report = hgm.doctor_gemini(gdir)
    assert report.owned_handler_count == 5
    assert (report.unquoted_commands, report.stale_commands) == ([exe], [])
    assert any("is not quoted" in f for f in report.faults())


@pytest.mark.parametrize(
    ("section", "key"), [("tools", "enableHooks"), ("hooksConfig", "enabled")],
)
def test_doctor_faults_hooks_turned_off(
    gdir: Path, cmd: str, section: str, key: str,
) -> None:
    path = gdir / "settings.json"
    _install(path, cmd)
    doc = _read(path)
    doc[section] = {key: False}
    _write(path, doc)
    report = hgm.doctor_gemini(gdir)
    assert report.hooks_turned_off_by == [f"{section}.{key}"]
    assert any("turned off" in f for f in report.faults())


def test_doctor_reports_a_disabled_hook_name_without_failing(
    gdir: Path, cmd: str,
) -> None:
    path = gdir / "settings.json"
    _install(path, cmd)
    doc = _read(path)
    doc["hooksConfig"] = {"disabled": ["aelfrice-memory-mirror", "other"]}
    _write(path, doc)
    report = hgm.doctor_gemini(gdir)
    assert report.disabled_hook_names == ["aelfrice-memory-mirror"]
    assert report.faults() == []


def test_doctor_reports_an_unparseable_file(gdir: Path) -> None:
    """Not a fault: nothing in the file says aelfrice was installed."""
    (gdir / "settings.json").write_text("{ nope", encoding="utf-8")
    report = hgm.doctor_gemini(gdir)
    assert report.settings_present and not report.settings_valid
    assert report.parse_error
    assert report.faults() == []


def test_doctor_faults_an_unparseable_file_that_names_the_hook(
    gdir: Path, cmd: str,
) -> None:
    path = gdir / "settings.json"
    _install(path, cmd)
    path.write_text("// mine\n" + path.read_text(encoding="utf-8"), encoding="utf-8")
    report = hgm.doctor_gemini(gdir)
    assert report.owned_handler_count == 0
    assert any("can't check the aelfrice hooks" in f for f in report.faults())
