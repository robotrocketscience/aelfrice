"""Hook-install convergence and host-level bounds (#1161).

Three defects, one theme: what `aelf setup` writes into settings.json was
neither convergent nor observable.

* `_install_or_replace_entry` returned on the *first* `(matcher,
  basename)` match, so a settings.json already holding two entries for
  one logical hook kept them forever — and because the first match was
  byte-identical, it reported "already installed" and skipped the write.
  Every aelfrice hook fired twice per event, permanently, while `aelf
  setup` and `aelf doctor --prune` both reported success. Reproduced in
  the field on the maintainer's own machine across all ten default-on
  hooks.
* Nothing detected that state. `--prune` removes entries whose program
  path is *broken*; a duplicate resolves perfectly well.
* Every installed entry omitted `timeout`, so the module's "never block
  the user's prompt" contract had no host-level enforcement at all.

The timeout values are grounded rather than guessed: a cold-start
UserPromptSubmit fire against a real 45,919-belief store measured 1.74s
to 2.67s when it had to rebuild a stale BM25F sidecar, and 0.36s warm.
The manifest's 15s interactive budget is ~5.6x the worst observed cold
path, which bounds a wedged hook without truncating a legitimate rebuild
on a store several times larger.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from aelfrice.auto_install import (
    load_manifest,
    manifest_timeouts_by_installer,
)
from aelfrice.doctor import (
    DoctorReport,
    _entry_duplicate_key,
    diagnose,
    find_duplicate_hook_entries,
    format_report,
    prune_broken_aelf_hooks,
)
from aelfrice.setup import (
    install_search_tool_hook,
    install_user_prompt_submit_hook,
)

@pytest.fixture(autouse=True)
def _no_ambient_opt_outs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the opt-out set empty for every test in this module.

    `_default_on_hook_basenames` and `diagnose` read the real
    `~/.aelfrice/opt-out-hooks.json`. Without this, a developer who has
    ever run `aelf setup --no-commit-ingest` sees different results than
    CI — the ambient-state failure mode that already bites
    `test_doctor_cli_exit_0_when_clean`.
    """
    monkeypatch.setattr(
        "aelfrice.auto_install.read_opt_outs", lambda *a, **k: set()
    )


# --- helpers -------------------------------------------------------------


def _entry(command: str, *, matcher: str | None = None,
           timeout: int | None = None) -> dict[str, object]:
    inner: dict[str, object] = {"type": "command", "command": command}
    if timeout is not None:
        inner["timeout"] = timeout
    entry: dict[str, object] = {"hooks": [inner]}
    if matcher is not None:
        entry["matcher"] = matcher
    return entry


def _write(path: Path, hooks: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hooks": hooks}, indent=2), encoding="utf-8")


def _event(path: Path, event: str) -> list[dict[str, object]]:
    return json.loads(path.read_text())["hooks"][event]


def _live(tmp_path: Path, name: str) -> Path:
    p = tmp_path / "bin" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/bin/sh\n", encoding="utf-8")
    p.chmod(0o755)
    return p


# --- the regression: install collapses duplicates ------------------------


def test_installing_over_a_duplicated_entry_collapses_it(
    tmp_path: Path,
) -> None:
    """The #1161 regression, in the exact live shape: byte-identical twins."""
    settings = tmp_path / "settings.json"
    twin = _entry("/opt/bin/aelf-hook")
    _write(settings, {"UserPromptSubmit": [twin, twin]})

    result = install_user_prompt_submit_hook(
        settings, command="/opt/bin/aelf-hook", timeout=15,
    )

    entries = _event(settings, "UserPromptSubmit")
    assert len(entries) == 1
    # Reported as a mutation, so the caller performs the atomic write.
    # Pre-fix this returned already_present=True and skipped it, which is
    # precisely why the duplicate was permanent.
    assert result.already_present is False


def test_collapse_is_idempotent(tmp_path: Path) -> None:
    """Second run is a genuine no-op, not another rewrite."""
    settings = tmp_path / "settings.json"
    twin = _entry("/opt/bin/aelf-hook")
    _write(settings, {"UserPromptSubmit": [twin, twin, twin]})

    install_user_prompt_submit_hook(
        settings, command="/opt/bin/aelf-hook", timeout=15,
    )
    again = install_user_prompt_submit_hook(
        settings, command="/opt/bin/aelf-hook", timeout=15,
    )
    assert again.already_present is True
    assert len(_event(settings, "UserPromptSubmit")) == 1


def test_collapse_keeps_the_first_occurrence_preserving_order(
    tmp_path: Path,
) -> None:
    """Hook order within an event decides injection order; keep it stable."""
    settings = tmp_path / "settings.json"
    _write(settings, {
        "UserPromptSubmit": [
            _entry("/usr/local/bin/user-own-hook.sh"),
            _entry("/opt/bin/aelf-hook"),
            _entry("/other/user-hook-two.sh"),
            _entry("/opt/bin/aelf-hook"),
        ],
    })
    install_user_prompt_submit_hook(
        settings, command="/opt/bin/aelf-hook", timeout=15,
    )
    commands = [
        h["command"]
        for e in _event(settings, "UserPromptSubmit")
        for h in e["hooks"]
    ]
    # aelf-hook stays at index 1 (where the user had it), not appended.
    assert commands == [
        "/usr/local/bin/user-own-hook.sh",
        "/opt/bin/aelf-hook",
        "/other/user-hook-two.sh",
    ]


def test_duplicates_are_collapsed_even_when_paths_differ(
    tmp_path: Path,
) -> None:
    """A venv move changes the path, not the basename — still one hook."""
    settings = tmp_path / "settings.json"
    _write(settings, {
        "UserPromptSubmit": [
            _entry("/old/venv/bin/aelf-hook"),
            _entry("/new/venv/bin/aelf-hook"),
        ],
    })
    install_user_prompt_submit_hook(
        settings, command="/new/venv/bin/aelf-hook", timeout=15,
    )
    entries = _event(settings, "UserPromptSubmit")
    assert len(entries) == 1
    assert entries[0]["hooks"][0]["command"] == "/new/venv/bin/aelf-hook"


def test_matcher_scoped_duplicates_do_not_collapse_across_matchers(
    tmp_path: Path,
) -> None:
    """Same program under two matchers is two real hooks, not a duplicate.

    `search_tool` and `search_tool_bash` are distinct manifest rows that
    install `aelf-search-tool-hook` under `Grep|Glob` and `Bash`. Collapsing
    on basename alone would silently delete one of them.
    """
    settings = tmp_path / "settings.json"
    _write(settings, {
        "PreToolUse": [
            _entry("/opt/bin/aelf-search-tool-hook", matcher="Grep|Glob"),
            _entry("/opt/bin/aelf-search-tool-hook", matcher="Bash"),
        ],
    })
    install_search_tool_hook(
        settings, command="/opt/bin/aelf-search-tool-hook", timeout=15,
    )
    matchers = sorted(e.get("matcher") for e in _event(settings, "PreToolUse"))
    assert matchers == ["Bash", "Grep|Glob"]


# --- detection -----------------------------------------------------------


def test_find_duplicate_hook_entries_reports_the_live_shape(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    twin = _entry("/opt/bin/aelf-hook")
    _write(settings, {"UserPromptSubmit": [twin, twin, twin]})

    dupes = find_duplicate_hook_entries(settings)
    assert len(dupes) == 1
    assert dupes[0].event == "UserPromptSubmit"
    assert dupes[0].count == 3
    assert "aelf-hook" in dupes[0].describe()


def test_find_duplicate_hook_entries_ignores_user_owned_hooks(
    tmp_path: Path,
) -> None:
    """A user listing their own hook twice is their business, not ours."""
    settings = tmp_path / "settings.json"
    twin = _entry("/usr/local/bin/conversation-logger.sh")
    _write(settings, {"UserPromptSubmit": [twin, twin]})
    assert find_duplicate_hook_entries(settings) == []


def test_duplicate_key_is_none_for_non_aelf_entries() -> None:
    """The predicate that keeps the collapse blind to user hooks."""
    assert _entry_duplicate_key(_entry("/bin/my-own-hook.sh")) is None
    assert _entry_duplicate_key(_entry("/bin/aelf-hook")) is not None


def test_find_duplicate_hook_entries_quiet_on_a_clean_file(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    _write(settings, {
        "UserPromptSubmit": [_entry("/opt/bin/aelf-hook")],
        "Stop": [_entry("/opt/bin/aelf-stop-hook")],
    })
    assert find_duplicate_hook_entries(settings) == []


def test_find_duplicate_hook_entries_survives_a_malformed_file(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text("{not json", encoding="utf-8")
    assert find_duplicate_hook_entries(settings) == []


# --- repair via --prune --------------------------------------------------


def test_prune_collapses_duplicates(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    live = str(_live(tmp_path, "aelf-hook"))
    _write(settings, {
        "UserPromptSubmit": [_entry(live), _entry(live), _entry(live)],
    })
    result = prune_broken_aelf_hooks(settings)
    assert result.total_duplicates_removed == 2
    assert result.duplicates_per_event == {"UserPromptSubmit": 2}
    # The stale-path pass found nothing — these all resolved fine.
    assert result.total_removed == 0
    assert len(_event(settings, "UserPromptSubmit")) == 1


def test_prune_dry_run_reports_duplicates_without_writing(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    live = str(_live(tmp_path, "aelf-hook"))
    _write(settings, {"UserPromptSubmit": [_entry(live), _entry(live)]})
    before = settings.read_text()
    result = prune_broken_aelf_hooks(settings, dry_run=True)
    assert result.total_duplicates_removed == 1
    assert settings.read_text() == before


def test_prune_keeps_the_surviving_copy_when_a_duplicate_is_broken(
    tmp_path: Path,
) -> None:
    """Broken-path pass runs first, so the live copy is the keeper."""
    settings = tmp_path / "settings.json"
    live = str(_live(tmp_path, "aelf-hook"))
    _write(settings, {
        "UserPromptSubmit": [
            _entry("/vanished/venv/bin/aelf-hook"),
            _entry(live),
        ],
    })
    result = prune_broken_aelf_hooks(settings)
    assert result.total_removed == 1
    assert result.total_duplicates_removed == 0
    entries = _event(settings, "UserPromptSubmit")
    assert len(entries) == 1
    assert entries[0]["hooks"][0]["command"] == live


def test_prune_never_touches_user_owned_duplicates(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    mine = str(_live(tmp_path, "my-logger.sh"))
    _write(settings, {"UserPromptSubmit": [_entry(mine), _entry(mine)]})
    result = prune_broken_aelf_hooks(settings)
    assert result.total_duplicates_removed == 0
    assert len(_event(settings, "UserPromptSubmit")) == 2


# --- doctor reporting ----------------------------------------------------


def test_doctor_reports_duplicates(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    live = str(_live(tmp_path, "aelf-hook"))
    _write(settings, {"UserPromptSubmit": [_entry(live), _entry(live)]})
    report = diagnose(
        user_settings=settings, project_root=tmp_path / "noproj",
    )
    assert len(report.duplicate_hook_entries) == 1
    rendered = format_report(report)
    assert "duplicate aelf-* hook entries" in rendered
    assert "aelf doctor --prune" in rendered


def test_doctor_quiet_about_duplicates_when_clean(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    live = str(_live(tmp_path, "aelf-hook"))
    _write(settings, {"UserPromptSubmit": [_entry(live)]})
    report = diagnose(
        user_settings=settings, project_root=tmp_path / "noproj",
    )
    assert report.duplicate_hook_entries == []
    assert "duplicate aelf-* hook entries" not in format_report(report)


def test_empty_report_renders_without_duplicate_section() -> None:
    assert "duplicate aelf-* hook entries" not in format_report(DoctorReport())


# --- timeouts ------------------------------------------------------------


def test_manifest_declares_a_timeout_for_every_default_on_hook() -> None:
    """The bundled manifest must leave no default-on hook unbounded.

    `_parse_timeout` is fail-soft (a malformed cell degrades to None so a
    packaging typo cannot break the CLI), so this is the check that
    actually holds the shipped data to the contract.
    """
    missing = [
        h.name
        for h in load_manifest().hooks
        if h.default_on and h.timeout is None
    ]
    assert missing == []


def test_manifest_timeouts_are_positive_ints() -> None:
    for hook in load_manifest().hooks:
        if hook.timeout is None:
            continue
        assert isinstance(hook.timeout, int)
        assert not isinstance(hook.timeout, bool)
        assert hook.timeout > 0


def test_interactive_hooks_clear_the_measured_cold_start() -> None:
    """Budgets must exceed the worst measured cold path with headroom.

    Measured on a real 45,919-belief store: 2.67s worst-case cold
    UserPromptSubmit fire (stale BM25F sidecar rebuilt from scratch),
    0.36s warm. A budget at or below the cold path would convert a
    legitimate rebuild into a killed hook and a silently empty injection
    — a worse failure than the unbounded wait it replaces.
    """
    worst_measured_cold_seconds = 2.67
    timeouts = manifest_timeouts_by_installer()
    for installer in (
        "user_prompt_submit", "session_start", "search_tool",
        "search_tool_bash", "agent_context",
    ):
        budget = timeouts[installer]
        assert budget is not None
        assert budget >= 4 * worst_measured_cold_seconds, installer


def test_install_writes_the_manifest_timeout(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    _write(settings, {})
    expected = manifest_timeouts_by_installer()["user_prompt_submit"]
    install_user_prompt_submit_hook(
        settings, command="/opt/bin/aelf-hook", timeout=expected,
    )
    inner = _event(settings, "UserPromptSubmit")[0]["hooks"][0]
    assert inner["timeout"] == expected


def test_manifest_timeouts_cover_every_dispatchable_installer() -> None:
    """No default-on installer may be missing from the timeout map."""
    timeouts = manifest_timeouts_by_installer()
    for hook in load_manifest().hooks:
        if hook.default_on:
            assert timeouts.get(hook.installer) is not None, hook.name


def test_manifest_timeouts_fail_soft_when_manifest_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A packaging error must not break `aelf setup`."""
    def _boom() -> object:
        raise ValueError("malformed")

    monkeypatch.setattr("aelfrice.auto_install.load_manifest", _boom)
    assert manifest_timeouts_by_installer() == {}


def test_parse_timeout_rejects_bools_and_nonpositive() -> None:
    """`True` is an `int` in Python; installing `"timeout": true` is invalid."""
    from aelfrice.auto_install import _parse_timeout

    assert _parse_timeout(True) is None
    assert _parse_timeout(False) is None
    assert _parse_timeout(0) is None
    assert _parse_timeout(-5) is None
    assert _parse_timeout("15") is None
    assert _parse_timeout(None) is None
    assert _parse_timeout(15) == 15


# --- manifest-driven doctor coverage ------------------------------------


def test_doctor_flags_a_missing_primary_retrieval_hook(
    tmp_path: Path,
) -> None:
    """The concrete regression: everything installed except `aelf-hook`.

    Pre-#1161 doctor's covered set was a 4-tuple that omitted `aelf-hook`
    entirely, so this settings.json reported clean while no belief could
    ever reach a prompt again.
    """
    settings = tmp_path / "settings.json"
    _write(settings, {
        "UserPromptSubmit": [
            _entry(str(_live(tmp_path, "aelf-transcript-logger"))),
        ],
        "SessionStart": [
            _entry(str(_live(tmp_path, "aelf-session-start-hook"))),
        ],
        "Stop": [_entry(str(_live(tmp_path, "aelf-stop-hook")))],
        "PostToolUse": [
            _entry(str(_live(tmp_path, "aelf-commit-ingest")), matcher="Bash"),
        ],
    })
    report = diagnose(
        user_settings=settings, project_root=tmp_path / "noproj",
    )
    assert "aelf-hook" in report.missing_auto_capture_hooks
    assert "default-on hooks not installed" in format_report(report)


def test_default_on_basenames_have_no_duplicates() -> None:
    """Two manifest rows share `aelf-search-tool-hook`; report it once."""
    from aelfrice.doctor import _default_on_hook_basenames

    names = _default_on_hook_basenames()
    assert len(names) == len(set(names))
    counts = Counter(
        h.basename for h in load_manifest().hooks if h.default_on
    )
    assert any(c > 1 for c in counts.values()), (
        "fixture assumption: a shared basename should exist in the manifest"
    )
