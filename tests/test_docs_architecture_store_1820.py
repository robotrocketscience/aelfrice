"""ARCHITECTURE.md claims about commit ingest and the store marker (#1820).

The #1820 docs sweep left two statements on the page that disagreed with
the code:

  1. The module table called `hook_commit_ingest.py` the `PostToolUse:Bash`
     hook. Since #1698, `aelf setup` installs it as the git `post-commit`
     hook and removes the old settings entry; the page's own hook table
     already said so.
  2. The `marker` step of "Store location" said the nearest `[store]`
     table applies everywhere. Outside a git work tree, only the nearest
     `.aelfrice.toml` counts, so a nearer file without the table hides one
     above it and the `global` rule applies.

Each test checks the code fact first, then the page. The page checks match
terms, not sentences: the terms carry the claim, so rewording the prose
keeps these quiet, and dropping the claim fails them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from aelfrice import db_paths
from aelfrice.auto_install import _merge_hooks_into_settings, load_manifest
from aelfrice.config_discovery import CONFIG_FILENAME
from aelfrice.db_paths import STORE_RULE_ENV, STORE_RULE_GLOBAL, resolve_store
from aelfrice.setup import COMMIT_INGEST_EVENT, COMMIT_INGEST_SCRIPT_NAME

REPO = Path(__file__).resolve().parents[1]
ARCHITECTURE = REPO / "docs" / "concepts" / "ARCHITECTURE.md"
SRC = REPO / "src" / "aelfrice"


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def _module_row(name: str) -> str:
    text = ARCHITECTURE.read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if ln.startswith(f"| `{name}` |")]
    assert len(rows) == 1, f"expected one module-table row for {name}, got {rows}"
    return rows[0]


def _store_step(prefix: str) -> str:
    text = ARCHITECTURE.read_text(encoding="utf-8")
    heading = "## Store location"
    assert heading in text, "ARCHITECTURE.md lost its Store location section"
    section = text.split(heading, 1)[1].split("\n## ", 1)[0]
    steps = [ln for ln in section.splitlines() if ln.startswith(prefix)]
    assert len(steps) == 1, f"expected one {prefix} step, got {steps}"
    return _collapse(steps[0])


def _marker_step() -> str:
    text = ARCHITECTURE.read_text(encoding="utf-8")
    heading = "## Store location"
    assert heading in text, "ARCHITECTURE.md lost its Store location section"
    section = text.split(heading, 1)[1].split("\n## ", 1)[0]
    steps = [ln for ln in section.splitlines() if ln.startswith("2. `marker`")]
    assert len(steps) == 1, f"expected one marker step, got {steps}"
    return _collapse(steps[0])


@pytest.mark.source_scan
def test_commit_ingest_module_row_names_the_git_hook(tmp_path: Path) -> None:
    """Setup wires the git hook and neither install path writes the
    settings entry, so the module row must name `post-commit` and not
    `PostToolUse`.

    Two paths could write the entry: a direct call to
    `install_commit_ingest_hook(` anywhere in the package, and the
    manifest merge, which holds the installer by reference in
    `_DISPATCH` and runs it for every default-on hook. The regex covers
    the first. Running the merge into a scratch settings file covers the
    second.
    """
    cli_text = (SRC / "cli.py").read_text(encoding="utf-8")
    assert "install_commit_ingest_git_hook(" in cli_text, (
        "aelf setup no longer installs the commit-ingest git hook; "
        "revisit ARCHITECTURE.md's hook_commit_ingest.py row"
    )
    callers = [
        p.name
        for p in sorted(SRC.rglob("*.py"))
        if re.search(
            r"(?<!def )(?<!un)\binstall_commit_ingest_hook\(",
            p.read_text(encoding="utf-8"),
        )
    ]
    assert not callers, (
        f"{callers} install the PostToolUse commit-ingest entry again; "
        "revisit ARCHITECTURE.md's hook_commit_ingest.py row"
    )
    settings = tmp_path / "settings.json"
    installed, already, _ = _merge_hooks_into_settings(
        manifest=load_manifest(),
        opt_outs=set(),
        scope="user",
        settings_path=settings,
        timeout=None,
    )
    assert "commit_ingest" not in installed + already, (
        "the manifest merge installs the PostToolUse commit-ingest entry "
        "again; revisit ARCHITECTURE.md's hook_commit_ingest.py row"
    )
    data = json.loads(settings.read_text(encoding="utf-8"))
    commands = [
        hook.get("command", "")
        for entry in data.get("hooks", {}).get(COMMIT_INGEST_EVENT, [])
        for hook in entry.get("hooks", [])
    ]
    assert not [c for c in commands if COMMIT_INGEST_SCRIPT_NAME in c], commands
    row = _module_row("hook_commit_ingest.py")
    assert "post-commit" in row, row
    assert "PostToolUse" not in row, row


def test_marker_step_states_the_outside_work_tree_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside a work tree a nearer config without `[store]` hides an
    outer marker; the page's marker step has to say so."""
    monkeypatch.delenv("AELFRICE_DB", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(db_paths, "DEFAULT_DB_DIR", home / ".aelfrice")
    db_paths.clear_git_common_dir_cache()
    outer = tmp_path / "notes"
    outer.mkdir()
    (outer / CONFIG_FILENAME).write_text(
        '[store]\npath = "outer-store"\n', encoding="utf-8",
    )
    inner = outer / "inner"
    inner.mkdir()
    (inner / CONFIG_FILENAME).write_text("[retrieval]\n", encoding="utf-8")
    monkeypatch.chdir(inner)
    assert resolve_store().rule == STORE_RULE_GLOBAL

    step = _marker_step()
    assert "outside a git work tree" in step, step
    assert "only the nearest" in step, step
    assert "inside a git work tree" in step, step


def test_env_step_says_an_empty_value_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty `AELFRICE_DB` falls through to the other rules, so
    the page's env step has to say the value must be non-empty."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(db_paths, "DEFAULT_DB_DIR", home / ".aelfrice")
    db_paths.clear_git_common_dir_cache()
    folder = tmp_path / "plain"
    folder.mkdir()
    monkeypatch.chdir(folder)
    monkeypatch.setenv("AELFRICE_DB", "")
    assert resolve_store().rule == STORE_RULE_GLOBAL
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))
    assert resolve_store().rule == STORE_RULE_ENV

    step = _store_step("1. `env`")
    assert "non-empty" in step, step
    assert "empty value counts as unset" in step, step
