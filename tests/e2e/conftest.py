"""E2E test fixtures (#334).

Boundary rule: e2e tests invoke the binary as installed (subprocess),
never via in-process imports. Fixtures here are deliberately minimal —
each test owns its own ephemeral DB and (where needed) its own
synthetic git project. No mocks for store, store init, or schema.

`installed_aelf` resolves which `aelf` to run:
    AELFRICE_E2E_BIN env var > shutil.which('aelf') > 'uv run aelf'.

CI sets AELFRICE_E2E_BIN per install-method matrix leg
(see .github/workflows/e2e.yml). Locally the fallback to `uv run aelf`
keeps the suite runnable without an explicit install step.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest


@pytest.fixture
def ephemeral_db(tmp_path: Path) -> Path:
    """Per-test SQLite path. Set AELFRICE_DB to this in subprocess env."""
    return tmp_path / "aelf.sqlite3"


@pytest.fixture
def installed_aelf() -> Sequence[str]:
    """Argv prefix for invoking `aelf`. Tuple of strings; tests append args."""
    explicit = os.environ.get("AELFRICE_E2E_BIN")
    if explicit:
        return (explicit,)
    on_path = shutil.which("aelf")
    if on_path:
        return (on_path,)
    # Local fallback for `pytest tests/e2e/` without an explicit install.
    return ("uv", "run", "aelf")


@pytest.fixture
def aelf_run(
    installed_aelf: Sequence[str],
    ephemeral_db: Path,
) -> Iterator[
    "callable[..., subprocess.CompletedProcess[str]]"  # type: ignore[name-defined]
]:
    """Run `aelf <args>` with AELFRICE_DB pinned to the ephemeral DB.

    Returns a callable so tests can capture stdout/stderr and exit code.
    Inherits the parent environment (PATH etc.) and overlays AELFRICE_DB.
    """

    def _run(
        *args: str,
        cwd: Path | None = None,
        check: bool = True,
        timeout: float = 60.0,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["AELFRICE_DB"] = str(ephemeral_db)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(  # noqa: S603 — argv list, not shell
            [*installed_aelf, *args],
            cwd=str(cwd) if cwd else None,
            env=env,
            check=check,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    yield _run


@pytest.fixture
def tiny_project(tmp_path: Path) -> Path:
    """Public-safe synthetic git repo with a handful of distinctive commits.

    Deliberately not a snapshot of any real project. Content is generated
    inline so the fixture has no on-disk dependencies and no
    directory-of-origin concerns.
    """
    proj = tmp_path / "tiny-project"
    proj.mkdir()

    files: dict[str, str] = {
        "README.md": textwrap.dedent(
            """
            # tiny-project

            Synthetic fixture for aelfrice e2e tests. The phrase
            `quokka calibration knob` is a deliberate distinctive
            token that search tests look for.
            """
        ).strip()
        + "\n",
        "src/widgets.py": textwrap.dedent(
            '''
            """Widgets module — distinctive token: aardvark-counter."""


            def make_widget(label: str) -> dict[str, str]:
                """Construct a labeled widget."""
                return {"label": label, "kind": "widget"}
            '''
        ).lstrip(),
        "docs/notes.md": "Calibrating the quokka knob requires three turns.\n",
    }

    for relpath, body in files.items():
        path = proj / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)

    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "tiny-project",
        "GIT_AUTHOR_EMAIL": "tiny@example.invalid",
        "GIT_COMMITTER_NAME": "tiny-project",
        "GIT_COMMITTER_EMAIL": "tiny@example.invalid",
    }

    def git(*args: str) -> None:
        subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=proj,
            env=git_env,
            check=True,
            capture_output=True,
        )

    git("init", "-q", "-b", "main")
    git("add", "README.md")
    git("commit", "-q", "-m", "init: tiny-project README")
    git("add", "src/widgets.py")
    git("commit", "-q", "-m", "feat: add widgets module")
    git("add", "docs/notes.md")
    git("commit", "-q", "-m", "docs: note quokka calibration cadence")

    return proj
