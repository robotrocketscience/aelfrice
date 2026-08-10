"""`$CODEX_HOME` routing for the Codex host (#1427).

Codex reads its configuration from `$CODEX_HOME` when that variable is
set; aelfrice bound ``Path.home() / ".codex"`` into a module-level
constant at import time, so `aelf setup --host codex` wrote hooks into
the conventional directory while the running Codex read the configured
one — setup reported success against a file Codex never loads, and
doctor/unsetup inspected and stripped the wrong directory.

Every test here drives `$HOME` and `$CODEX_HOME` apart, so a resolver
that consults only one of them fails visibly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.host_codex import (
    CodexHomeError,
    codex_config_path,
    codex_hooks_path,
    resolve_agents_skills_dir,
    resolve_codex_home,
)


@pytest.fixture()
def isolated_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point `Path.home()` at a tmp dir on every platform."""
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


# --- resolution ------------------------------------------------------------


def test_codex_home_env_wins(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "custom codex home"
    custom.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(custom))
    assert resolve_codex_home() == custom
    assert codex_hooks_path() == custom / "hooks.json"
    assert codex_config_path() == custom / "config.toml"


def test_unset_codex_home_keeps_conventional_dir(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert resolve_codex_home() == isolated_home / ".codex"


def test_empty_codex_home_keeps_conventional_dir(
    isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex ignores an empty value; so must we."""
    monkeypatch.setenv("CODEX_HOME", "")
    assert resolve_codex_home() == isolated_home / ".codex"


def test_resolution_is_late_bound_not_import_time(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An environment change between two calls is honoured (#1427/#1320)."""
    first = tmp_path / "home-one"
    second = tmp_path / "home-two"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(first))
    assert resolve_codex_home() == first
    monkeypatch.setenv("CODEX_HOME", str(second))
    assert resolve_codex_home() == second


def test_awkward_characters_survive(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "cödex home's dir"
    custom.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(custom))
    assert resolve_codex_home() == custom


def test_relative_codex_home_is_made_absolute(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rel-codex").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CODEX_HOME", "rel-codex")
    resolved = resolve_codex_home()
    assert resolved.is_absolute()
    assert resolved.name == "rel-codex"


def test_non_directory_codex_home_is_an_error_not_a_fallback(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicitly configured bad value must not silently revert."""
    bogus = tmp_path / "not-a-dir"
    bogus.write_text("x", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(bogus))
    with pytest.raises(CodexHomeError) as excinfo:
        resolve_codex_home()
    assert "CODEX_HOME" in str(excinfo.value)
    assert "not a directory" in str(excinfo.value)
    assert not (isolated_home / ".codex").exists()


def test_nonexistent_codex_home_is_an_error_not_a_fresh_directory(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex's own rule for a missing `$CODEX_HOME`, mirrored.

    Measured against codex-cli 0.145.0 on the machine this was written
    on::

        $ CODEX_HOME=/tmp/definitely-not-here codex mcp list
        Error: failed to load configuration
        Caused by:
            CODEX_HOME points to "/tmp/definitely-not-here", but that
            path does not exist

    Resolving it anyway is not a harmless convenience: setup would
    `mkdir -p` a brand-new configuration home and report success against
    a directory Codex refuses to start in — the "setup succeeded, hooks
    unwired" failure #1427 is about, reached by one typo instead of by
    ignoring the variable.
    """
    missing = tmp_path / "typo'd codex home"
    monkeypatch.setenv("CODEX_HOME", str(missing))
    with pytest.raises(CodexHomeError) as excinfo:
        resolve_codex_home()
    assert "does not exist" in str(excinfo.value)
    assert not missing.exists()
    assert not (isolated_home / ".codex").exists()


def test_setup_does_not_create_a_nonexistent_codex_home(
    tmp_path: Path,
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI half: exit 1, nothing created, neither directory touched."""
    missing = tmp_path / "typo-codex"
    monkeypatch.setenv("CODEX_HOME", str(missing))

    rc = main(["setup", "--host", "codex", "--no-codex-skills"])

    assert rc == 1
    assert "does not exist" in capsys.readouterr().err
    assert not missing.exists()
    assert not (isolated_home / ".codex").exists()


def test_doctor_and_unsetup_refuse_a_nonexistent_codex_home(
    tmp_path: Path,
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All three verbs resolve through one function, so all three refuse."""
    missing = tmp_path / "typo-codex"
    monkeypatch.setenv("CODEX_HOME", str(missing))

    assert main(["doctor", "--host", "codex"]) == 1
    assert "does not exist" in capsys.readouterr().err
    assert main(["unsetup", "--host", "codex"]) == 1
    assert "does not exist" in capsys.readouterr().err
    assert not missing.exists()


def test_skills_dir_ignores_codex_home(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent skills stay on the cross-agent standard path."""
    custom = tmp_path / "custom-codex"
    custom.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(custom))
    assert resolve_agents_skills_dir() == isolated_home / ".agents" / "skills"


# --- CLI end-to-end --------------------------------------------------------


def test_setup_doctor_unsetup_all_route_to_codex_home(
    tmp_path: Path,
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The issue's reproduction, as one CLI round trip.

    Setup must write below `$CODEX_HOME`, doctor must report that
    directory and count the handlers it holds, and unsetup must strip
    them from there — with nothing created under `$HOME/.codex`.
    """
    custom = tmp_path / "custom codex home"
    custom.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(custom))
    skills = tmp_path / "skills"

    rc = main(["setup", "--host", "codex", "--no-codex-skills"])
    assert rc == 0
    assert (custom / "hooks.json").is_file()
    assert not (isolated_home / ".codex").exists()

    doc = json.loads((custom / "hooks.json").read_text(encoding="utf-8"))
    assert "UserPromptSubmit" in doc["hooks"]

    capsys.readouterr()
    rc = main(["doctor", "--host", "codex"])
    out = capsys.readouterr().out
    assert str(custom) in out
    assert "aelfrice_handlers=0" not in out
    assert rc in (0, 1)

    rc = main(["unsetup", "--host", "codex"])
    assert rc == 0
    doc = json.loads((custom / "hooks.json").read_text(encoding="utf-8"))
    assert doc.get("hooks", {}) == {}
    assert not (isolated_home / ".codex").exists()
    assert not skills.exists()


def test_unsetup_creates_nothing_on_a_host_that_never_had_codex(
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`unsetup` on a fresh host must leave the filesystem as it found it.

    The CLI half of the `remove_codex_hooks` absence check. With
    `$CODEX_HOME` unset, the resolved home is `~/.codex`, which does not
    exist here — and an uninstall verb that mkdirs it (to create a lock
    file it never sweeps) is creating the configuration directory it is
    supposed to be removing from. The assertion is over the whole home,
    so any residue anywhere below it fails.
    """
    monkeypatch.delenv("CODEX_HOME", raising=False)
    before = sorted(p.name for p in isolated_home.rglob("*"))

    rc = main(["unsetup", "--host", "codex"])

    assert rc == 0
    assert "no aelfrice entries" in capsys.readouterr().out
    assert not (isolated_home / ".codex").exists()
    assert sorted(p.name for p in isolated_home.rglob("*")) == before


def test_setup_reports_bad_codex_home_without_touching_fallback(
    tmp_path: Path,
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bogus = tmp_path / "file-not-dir"
    bogus.write_text("x", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(bogus))
    rc = main(["setup", "--host", "codex", "--no-codex-skills"])
    assert rc == 1
    assert "CODEX_HOME" in capsys.readouterr().err
    assert not (isolated_home / ".codex").exists()
