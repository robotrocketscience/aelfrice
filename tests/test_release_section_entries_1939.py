"""`release-docs-check` refuses an empty dated CHANGELOG section (#1939).

The job checked for the `## [X.Y.Z]` heading, the compare-link footnote
and two drained unreleased surfaces, but not that the dated section
listed anything. A cut from a tree with no entry files collates to a
bare heading, and that release passed the check and published an empty
section and an empty release PR body.

These tests run the workflow's own shell, extracted from
`staging-gate.yml`, from the `major=` assignment up to the directory
drain guard. That span holds every check on the `CHANGELOG/vN.md` file,
so a test here fails if the new predicate is edited into one that
matches nothing, and it failed before the predicate existed.
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO / ".github" / "workflows" / "staging-gate.yml"

_spec = importlib.util.spec_from_file_location(
    "_collate_1939", _REPO / "scripts" / "collate_changelog.py"
)
assert _spec and _spec.loader
collate_changelog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collate_changelog)

_EMPTY_MESSAGE = "has no entries"

_FOOTNOTES = (
    "[5.4.1]: https://github.com/robotrocketscience/aelfrice/compare/v5.4.0...v5.4.1\n"
    "[5.4.0]: https://github.com/robotrocketscience/aelfrice/compare/v5.3.0...v5.4.0\n"
)

_SHIPPED = (
    "## [5.4.0] - 2026-11-11\n"
    "\n"
    "### Fixed\n"
    "\n"
    "- **Something that already shipped ([#1](u)).** Prose.\n"
    "\n"
)


def _changelog_block() -> str:
    """The step's `CHANGELOG/vN.md` checks, dedented."""
    lines = _WORKFLOW.read_text(encoding="utf-8").split("\n")
    start = end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if start is None and stripped.startswith('major="${head_ver%%.*}"'):
            start = i
        if start is not None and stripped.startswith(
            "if [ ! -d CHANGELOG/unreleased ]"
        ):
            end = i
            break
    assert start is not None and end is not None, (
        "release-docs-check no longer has the CHANGELOG/vN.md block"
    )
    body = lines[start:end]
    indent = len(body[0]) - len(body[0].lstrip())
    return "\n".join(ln[indent:] for ln in body)


def _run(root: Path, version: str) -> subprocess.CompletedProcess[str]:
    """Run the block under the step's `set -euo pipefail`, in bash.

    GitHub's default `run:` shell on Linux is `bash -e {0}`, and the
    step opens `set -euo pipefail`; dash would reject `pipefail`.
    """
    script = (
        "set -euo pipefail\n"
        f'head_ver="{version}"\n'
        'base_ver="0.0.0"\n'
        "fail=0\n"
        + _changelog_block()
        + '\nexit "$fail"\n'
    )
    return subprocess.run(
        ["bash", "-c", script],
        cwd=root, capture_output=True, text=True, check=False, timeout=20,
    )


def _tree(tmp_path: Path, text: str, major: int = 5) -> Path:
    directory = tmp_path / "CHANGELOG"
    directory.mkdir()
    # Bytes, so a CRLF fixture reaches the file unchanged on every platform.
    (directory / f"v{major}.md").write_bytes(text.encode("utf-8"))
    return tmp_path


def _with_section(section: str) -> str:
    return (
        "# Changelog\n\n## [Unreleased]\n\n"
        + section
        + _SHIPPED
        + _FOOTNOTES
    )


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "section",
    [
        pytest.param("## [5.4.1] - 2026-11-19\n\n", id="empty"),
        pytest.param("## [5.4.1] - 2026-11-19\n", id="no-blank-line"),
        pytest.param(
            "## [5.4.1] - 2026-11-19\n\n### Fixed\n\n### Changed\n\n",
            id="category-headings-only",
        ),
        pytest.param(
            "## [5.4.1] - 2026-11-19\n\n### Fixed\n\n#### Retrieval\n\n",
            id="sub-heading-only",
        ),
        pytest.param(
            "## [5.4.1] - 2026-11-19\n\n   ### Fixed\n\n",
            id="heading-indented-three-spaces",
        ),
        pytest.param(
            "## [5.4.1] - 2026-11-19\n   \n\t\n \t \n\n",
            id="whitespace-only",
        ),
    ],
)
def test_an_empty_dated_section_fails_naming_the_version_and_file(
    tmp_path: Path, section: str
) -> None:
    proc = _run(_tree(tmp_path, _with_section(section)), "5.4.1")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    errors = [ln for ln in proc.stdout.split("\n") if _EMPTY_MESSAGE in ln]
    assert len(errors) == 1, proc.stdout
    assert "[5.4.1]" in errors[0]
    assert "CHANGELOG/v5.md" in errors[0].split("::", 2)[2]


@pytest.mark.timeout(30)
def test_the_footnotes_after_a_last_section_are_not_entries(
    tmp_path: Path,
) -> None:
    """A new major's first file ends with its section, then the footnote.

    The section runs to end of file, so without the footnote exclusion
    the link line alone would read as content.
    """
    text = (
        "# Changelog\n\n## [Unreleased]\n\n"
        "## [6.0.0] - 2026-12-01\n\n### Fixed\n\n"
        "[6.0.0]: https://github.com/robotrocketscience/aelfrice/compare/v5.4.1...v6.0.0\n"
    )
    proc = _run(_tree(tmp_path, text, major=6), "6.0.0")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert _EMPTY_MESSAGE in proc.stdout


@pytest.mark.timeout(30)
def test_a_heading_for_a_longer_version_does_not_stand_in(
    tmp_path: Path,
) -> None:
    """`[5.4.10]` with entries must not satisfy an empty `[5.4.1]`."""
    text = (
        "# Changelog\n\n## [Unreleased]\n\n"
        "## [5.4.10] - 2026-12-01\n\n### Fixed\n\n- **A ([#2](u)).** x\n\n"
        "## [5.4.1] - 2026-11-19\n\n"
        + _SHIPPED
        + _FOOTNOTES
    )
    proc = _run(_tree(tmp_path, text), "5.4.1")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert _EMPTY_MESSAGE in proc.stdout


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "section",
    [
        pytest.param(
            "## [5.4.1] - 2026-11-19\n\n### Fixed\n\n- **A ([#2](u)).** x\n\n",
            id="one-entry",
        ),
        pytest.param(
            "## [5.4.1] - 2026-11-19\n\nA summary paragraph and nothing else.\n\n",
            id="summary-only",
        ),
        pytest.param(
            "## [5.4.1] - 2026-11-19\n\n    # four spaces make a code block\n\n",
            id="indented-code-block",
        ),
    ],
)
def test_a_dated_section_with_content_passes(
    tmp_path: Path, section: str
) -> None:
    """The summary-only shape is how `[2.0.0]` shipped in v2.md."""
    proc = _run(_tree(tmp_path, _with_section(section)), "5.4.1")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _EMPTY_MESSAGE not in proc.stdout


def _dated_versions() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for path in sorted((_REPO / "CHANGELOG").glob("v*.md")):
        for line in path.read_text(encoding="utf-8").split("\n"):
            match = re.match(r"## \[(\d[^\]]*)\] - ", line)
            if match:
                found.append((path.name, match.group(1)))
    return found


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("name", "version"),
    _dated_versions(),
    ids=lambda value: str(value),
)
def test_every_committed_dated_section_has_entries(
    name: str, version: str
) -> None:
    """Every shipped section would pass the stricter check.

    Only the new message is asserted: older files lack some footnotes,
    and those checks are not under test here.
    """
    proc = _run(_REPO, version)

    assert f"v{version.split('.')[0]}.md" == name
    assert _EMPTY_MESSAGE not in proc.stdout, proc.stdout


def _cut(files: list[tuple[str, str]]) -> str:
    changelog = "# Changelog\n\n## [Unreleased]\n\n" + _SHIPPED + _FOOTNOTES
    return collate_changelog.collate(changelog, files, "5.4.1", "2026-11-19")


@pytest.mark.timeout(30)
def test_a_cut_with_no_entry_files_fails_the_check(tmp_path: Path) -> None:
    """The issue's scenario end to end: collation emits a bare heading."""
    out = _cut([])
    assert "## [5.4.1] - 2026-11-19" in out

    proc = _run(_tree(tmp_path, out), "5.4.1")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert _EMPTY_MESSAGE in proc.stdout


@pytest.mark.timeout(30)
def test_a_cut_with_one_entry_file_passes_the_check(tmp_path: Path) -> None:
    entry = (
        "1939-slug.md",
        "### Fixed\n\n- **A ([#1939](u)).** x\n",
    )
    out = _cut([entry])

    proc = _run(_tree(tmp_path, out), "5.4.1")

    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.timeout(30)
def test_a_crlf_empty_section_fails(tmp_path: Path) -> None:
    """A CRLF line holds a bare `\\r`, which awk counts as a field."""
    text = _with_section("## [5.4.1] - 2026-11-19\n\n### Fixed\n\n")
    proc = _run(_tree(tmp_path, text.replace("\n", "\r\n")), "5.4.1")

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert _EMPTY_MESSAGE in proc.stdout, proc.stdout


@pytest.mark.timeout(30)
def test_a_crlf_changelog_with_entries_passes(tmp_path: Path) -> None:
    """The [Unreleased] drain check shares the bare-CR blank line."""
    text = _with_section(
        "## [5.4.1] - 2026-11-19\n\n### Fixed\n\n- **A ([#2](u)).** x\n\n"
    )
    proc = _run(_tree(tmp_path, text.replace("\n", "\r\n")), "5.4.1")

    assert proc.returncode == 0, proc.stdout + proc.stderr
