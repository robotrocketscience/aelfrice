"""Order-of-operations test for `_cmd_setup` migration wiring (#733).

The migration must run BEFORE hook reconciliation so that the
freshly-installed uv-tool `aelf-hook` shim is what `resolve_hook_command`
sees when it next walks PATH. If migration ran AFTER hook reconcile,
the hook commands would still point at the now-orphaned pipx binary.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

from aelfrice import cli as cli_mod
from aelfrice.lifecycle import MigrationResult


def test_migration_runs_before_hook_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`_cmd_setup` must call `maybe_migrate_to_uv` strictly before any
    hook-installer function. We record call order via a shared list
    and assert `maybe_migrate_to_uv` comes first.

    The fail mode this guards against: a refactor moves the migration
    call below `install_user_prompt_submit_hook`, so hooks land pointing
    at the pipx binary (which is about to be orphaned by the migration).
    The user would have to re-run `aelf setup` to fix the paths.
    """
    order: list[str] = []

    def _record_migration() -> MigrationResult:
        order.append("migrate")
        return MigrationResult(False, False, "already on uv tool")

    class _Result:
        already_present = True
        path = tmp_path / "settings.json"
        installed: tuple[str, ...] = ()
        already: tuple[str, ...] = ()

    def _record_clean() -> object:
        order.append("clean_dangling")
        class _Cleanup:
            removed: list[Path] = []
        return _Cleanup()

    def _record_install_ups(*a: object, **k: object) -> _Result:
        order.append("install_user_prompt_submit_hook")
        return _Result()

    def _record_install_ti(*a: object, **k: object) -> _Result:
        order.append("install_transcript_ingest_hooks")
        return _Result()

    monkeypatch.setattr(cli_mod, "maybe_migrate_to_uv", _record_migration)
    monkeypatch.setattr(cli_mod, "clean_dangling_shims", _record_clean)
    monkeypatch.setattr(
        cli_mod, "install_user_prompt_submit_hook", _record_install_ups
    )
    monkeypatch.setattr(
        cli_mod, "install_transcript_ingest_hooks", _record_install_ti
    )
    # Stub the rest of the hook installers — we only care about ordering
    # vs. the first hook installer, which is `install_user_prompt_submit_hook`.
    monkeypatch.setattr(
        cli_mod, "install_session_start_hook",
        lambda *a, **k: _Result(),
    )
    monkeypatch.setattr(
        cli_mod, "install_stop_hook", lambda *a, **k: _Result(),
    )
    monkeypatch.setattr(
        cli_mod, "install_commit_ingest_hook", lambda *a, **k: _Result(),
    )
    monkeypatch.setattr(
        cli_mod, "install_pre_compact_hook", lambda *a, **k: _Result(),
    )
    monkeypatch.setattr(
        cli_mod, "install_search_tool_hook", lambda *a, **k: _Result(),
    )
    monkeypatch.setattr(
        cli_mod, "install_search_tool_bash_hook", lambda *a, **k: _Result(),
    )
    monkeypatch.setattr(
        cli_mod, "install_slash_commands", lambda *a, **k: ([], []),
    )
    monkeypatch.setattr(
        cli_mod, "install_statusline", lambda *a, **k: _Result(),
    )

    args = argparse.Namespace(
        scope="user",
        project_root=None,
        settings=None,
        command=None,
        timeout=5,
        status_message=None,
        transcript_ingest=True,
        session_start=True,
        stop_hook=True,
        commit_ingest=True,
        rebuilder=False,
        search_tool=False,
        no_slash_commands=False,
        slash_dir=None,
        no_statusline=True,
        statusline_command=None,
        statusline_global=False,
        no_auto_install=True,
    )
    buf = io.StringIO()
    # _cmd_setup may bail or print to stderr; we only care about ordering.
    try:
        cli_mod._cmd_setup(args, buf)
    except Exception:
        # Even if a downstream stub doesn't perfectly match the signature
        # cli.py expects, the migration call should have happened first.
        pass
    assert "migrate" in order, f"migration was never called; order={order}"
    assert order.index("migrate") == 0, (
        f"migration must be the first recorded call but order was {order}"
    )
    # Specifically: migration before the first hook installer.
    if "install_user_prompt_submit_hook" in order:
        assert order.index("migrate") < order.index(
            "install_user_prompt_submit_hook"
        ), f"migration must precede hook install; got {order}"
