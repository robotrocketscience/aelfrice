"""#1652: say so when `$HOME/.aelfrice.toml` is ignored.

Since #1582 the config walk stops before it examines `$HOME`, so a
per-user file is never read. Settings that lived there stopped applying
with no message. The SessionStart hook and `aelf doctor` now name the
file, say it is ignored, and give the fix, but only when the project has
no config of its own.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice import hook
from aelfrice.config_discovery import ignored_home_config, ignored_home_config_notice
from aelfrice.doctor import diagnose, format_report


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    (h / ".aelfrice.toml").write_text("[cadence]\nenabled = true\n")
    return h


def _project(root: Path, *, with_config: bool) -> Path:
    root.mkdir(parents=True)
    (root / ".git").mkdir()
    if with_config:
        (root / ".aelfrice.toml").write_text("[cadence]\nenabled = true\n")
    return root


def test_a_project_without_config_reports_the_home_file(home: Path) -> None:
    proj = _project(home / "projects" / "app", with_config=False)
    assert ignored_home_config(proj) == (home / ".aelfrice.toml").resolve()
    line = ignored_home_config_notice(proj) or ""
    assert str((home / ".aelfrice.toml").resolve()) in line
    assert "ignored" in line and "project's root" in line


def test_a_project_with_config_reports_nothing(home: Path) -> None:
    proj = _project(home / "projects" / "app", with_config=True)
    assert ignored_home_config(proj) is None
    assert ignored_home_config_notice(proj) is None


def test_no_home_file_reports_nothing(home: Path) -> None:
    (home / ".aelfrice.toml").unlink()
    proj = _project(home / "projects" / "app", with_config=False)
    assert ignored_home_config(proj) is None


def test_a_project_outside_home_is_checked_too(
    home: Path, tmp_path: Path,
) -> None:
    proj = _project(tmp_path / "elsewhere" / "app", with_config=False)
    assert ignored_home_config(proj) == (home / ".aelfrice.toml").resolve()


def _session_start(payload: dict[str, object]) -> str:
    """The hook's stdout. The notice goes there, like the lock notice: a
    hook's stderr on exit 0 reaches only the host's debug log."""
    out = io.StringIO()
    rc = hook.session_start(stdin=io.StringIO(json.dumps(payload)),
                            stdout=out, stderr=io.StringIO())
    assert rc == 0
    return out.getvalue()


def test_session_start_warns_from_the_payload_cwd(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The hook process sits in a project that has config; the agent's cwd,
    # in the payload, is a project that has none. The payload decides.
    configured = _project(home / "projects" / "configured", with_config=True)
    bare = _project(home / "projects" / "bare", with_config=False)
    monkeypatch.chdir(configured)
    out = _session_start({"session_id": "s1", "source": "startup", "cwd": str(bare)})
    assert out.count(".aelfrice.toml is ignored") == 1


def test_session_start_is_quiet_when_the_project_has_config(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    proj = _project(home / "projects" / "app", with_config=True)
    monkeypatch.chdir(proj)
    out = _session_start({"session_id": "s1", "source": "startup", "cwd": str(proj)})
    assert "is ignored" not in out


def test_session_start_does_not_repeat_after_compaction(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    proj = _project(home / "projects" / "app", with_config=False)
    monkeypatch.chdir(proj)
    out = _session_start({"session_id": "s1", "source": "compact", "cwd": str(proj)})
    assert "is ignored" not in out


def _diagnose(tmp_path: Path, proj: Path):  # noqa: ANN202
    return diagnose(
        user_settings=tmp_path / "no-settings.json",
        project_root=proj,
        hook_failures_log=tmp_path / "no-failures.log",
        aelfrice_projects_dir=tmp_path / "no-projects",
    )


def test_doctor_warns_without_failing(home: Path, tmp_path: Path) -> None:
    proj = _project(home / "projects" / "app", with_config=False)
    report = _diagnose(tmp_path, proj)
    assert report.ignored_home_config == (home / ".aelfrice.toml").resolve()
    text = format_report(report)
    assert f"warning: {(home / '.aelfrice.toml').resolve()} is ignored" in text
    assert "fix: copy it to the project's root" in text
    # A warning, not a failure: nothing is marked broken.
    assert report.broken == []


def test_doctor_is_quiet_when_the_project_has_config(
    home: Path, tmp_path: Path,
) -> None:
    proj = _project(home / "projects" / "app", with_config=True)
    report = _diagnose(tmp_path, proj)
    assert report.ignored_home_config is None
    assert "is ignored" not in format_report(report)


def test_doctor_warns_on_the_settings_path_too(home: Path, tmp_path: Path) -> None:
    # With a settings.json, format_report takes its full path, which renders
    # its sections separately from the no-settings early return.
    proj = _project(home / "projects" / "app", with_config=False)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    report = diagnose(
        user_settings=settings,
        project_root=proj,
        hook_failures_log=tmp_path / "no-failures.log",
        aelfrice_projects_dir=tmp_path / "no-projects",
    )
    assert report.scopes_scanned
    assert "is ignored" in format_report(report)


@pytest.mark.parametrize("kind", ["directory", "dangling symlink"])
def test_a_home_path_that_is_not_a_file_reports_nothing(
    home: Path, kind: str,
) -> None:
    cfg = home / ".aelfrice.toml"
    cfg.unlink()
    if kind == "directory":
        cfg.mkdir()
    else:
        cfg.symlink_to(home / "missing.toml")
    proj = _project(home / "projects" / "app", with_config=False)
    assert ignored_home_config(proj) is None


def test_a_failing_check_does_not_break_session_start(
    home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(start: Path | None = None) -> str | None:
        raise RuntimeError("discovery failed")

    from aelfrice.models import BELIEF_FACTUAL, LOCK_USER, Belief
    from aelfrice.store import MemoryStore

    db = home / "memory.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(Belief(
            id="L1", content="Run pyright before every release.",
            content_hash="h_L1", alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
            lock_level=LOCK_USER, locked_at="2026-10-01T00:00:00Z",
            created_at="2026-10-01T00:00:00Z", last_retrieved_at=None,
        ))
    finally:
        store.close()
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setattr("aelfrice.config_discovery.ignored_home_config_notice", boom)
    proj = _project(home / "projects" / "app", with_config=False)
    monkeypatch.chdir(proj)
    out = _session_start({"session_id": "s1", "source": "startup", "cwd": str(proj)})
    assert "is ignored" not in out
    # The failure is contained: the locked-belief block still goes out.
    assert "Run pyright before every release." in out
