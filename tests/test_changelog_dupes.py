"""The CHANGELOG duplicate-entry guard (#1211).

Resolving a CHANGELOG conflict insert-only is correct for *added*
entries and wrong for *amended* ones — it leaves the superseded
revision beside the one that replaced it. Two entries reached
`[Unreleased]` that way before this guard existed.
"""
from __future__ import annotations

import importlib.util
import re
import shlex
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "check_changelog_dupes.py"

_spec = importlib.util.spec_from_file_location("_ccd", _SCRIPT)
assert _spec and _spec.loader
check_changelog_dupes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_changelog_dupes)

find_duplicates = check_changelog_dupes.find_duplicates
THRESHOLD = check_changelog_dupes.PREFIX_THRESHOLD


def _entry(head: str, tail: str) -> str:
    """A bullet whose opening is `head`, padded past the threshold."""
    return f"- **{head}** {'x' * (THRESHOLD + 50)} {tail}"


def test_an_amended_entry_beside_its_own_revision_is_caught() -> None:
    """The shape that actually happened: same opening, corrected tail."""
    shared = _entry("Feedback lost evidence ([#1168])", "beta 16 of 121")
    revised = _entry("Feedback lost evidence ([#1168])", "beta 16.0 not 121.0")
    text = "\n".join(["## [Unreleased]", "", "### Fixed", shared, revised])

    found = find_duplicates(text)

    assert len(found) == 1
    section, overlap, older = found[0]
    assert section == "[Unreleased]"
    assert overlap >= THRESHOLD
    assert older == min(shared, revised, key=len), "must name the shorter"


def test_full_containment_is_caught_too() -> None:
    """The other shape: the amendment appends rather than rewrites."""
    short = _entry("Superseded beliefs ([#1187])", "tail")
    long = short + " and the exclusion arm backfills the pack."
    found = find_duplicates(
        "\n".join(["## [Unreleased]", "### Fixed", short, long])
    )
    assert [f[2] for f in found] == [short]


def test_separate_fixes_under_one_umbrella_are_not_duplicates() -> None:
    """#1160 legitimately appears four times, #1161 three.

    Keying on the issue number would fire on every umbrella.
    """
    text = "\n".join([
        "## [Unreleased]", "### Fixed",
        _entry("Path filters omitted benchmarks ([#1160])", "a"),
        _entry("Band-check passed on absent data ([#1160])", "b"),
        _entry("Perf flag was never registered ([#1160])", "c"),
    ])
    assert find_duplicates(text) == []


def test_a_short_reused_title_is_not_a_duplicate() -> None:
    """v1.1.0 has two entries both titled with the same bare path.

    Keying on the bold title would fire on those; they diverge
    immediately, so the shared opening stays far below the threshold.
    """
    text = "\n".join([
        "## [1.1.0] - 2026-04-27", "### Added",
        "- **`docs/promotion_path.md`** describes the promotion ladder.",
        "- **`docs/promotion_path.md`** gained a worked example.",
    ])
    assert find_duplicates(text) == []


def test_entries_in_different_sections_do_not_collide() -> None:
    """The same entry may legitimately recur across release sections."""
    entry = _entry("Some fix ([#1])", "tail")
    text = "\n".join([
        "## [Unreleased]", "### Fixed", entry,
        "## [4.2.0] - 2026-07-21", "### Fixed", entry,
    ])
    assert find_duplicates(text) == []


@pytest.mark.parametrize(
    "path", sorted((_REPO / "CHANGELOG").glob("*.md")) + [
        _REPO / "CHANGELOG.md"
    ],
)
def test_committed_changelogs_are_clean(path: Path) -> None:
    """The guard must pass on every changelog actually in the tree."""
    found = find_duplicates(path.read_text(encoding="utf-8"))
    assert found == [], (
        f"{path.name} has a restated entry: "
        + "; ".join(f"{s} ({n} chars) {e[:90]}" for s, n, e in found)
    )


# ---------------------------------------------------------------------------
# `CHANGELOG/unreleased/` coverage (#1475).
#
# One entry per file means no single file can hold a duplicate, so a
# per-file check would report a pass having examined nothing that could
# fail — the #1160 defect class. These pin the two pairs that can
# actually duplicate: file-vs-file, and file-vs-`[Unreleased]` block
# during the transition.
# ---------------------------------------------------------------------------

scan_entry_dir = check_changelog_dupes.scan_entry_dir
main = check_changelog_dupes.main

_WORKFLOW = _REPO / ".github" / "workflows" / "staging-gate.yml"
_INVOCATION_RE = re.compile(
    r"python3 scripts/check_changelog_dupes\.py ([^\n]*)"
)


def _ci_arguments() -> list[str]:
    """The arguments `release-docs-check` actually passes."""
    found = _INVOCATION_RE.search(_WORKFLOW.read_text(encoding="utf-8"))
    assert found, "release-docs-check no longer runs the duplicate check"
    return shlex.split(found.group(1))


def _tree(tmp_path: Path, block: str, files: dict[str, str]) -> list[str]:
    """A changelog + entry directory; returns argv for `main`."""
    directory = tmp_path / "unreleased"
    directory.mkdir()
    (directory / "README.md").write_text(
        _entry("Not an entry, the directory note", "z") + "\n",
        encoding="utf-8",
    )
    for name, text in files.items():
        (directory / name).write_text(text + "\n", encoding="utf-8")
    changelog = tmp_path / "v4.md"
    changelog.write_text(
        "\n".join(["## [Unreleased]", "", "### Fixed", block, ""]),
        encoding="utf-8",
    )
    return ["check_changelog_dupes.py", str(changelog), str(directory)]


def test_a_duplicate_across_two_entry_files_is_caught(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Neither file is a duplicate on its own."""
    argv = _tree(tmp_path, "", {
        "1-a.md": "### Fixed\n\n" + _entry("Same fix ([#1])", "16 of 121"),
        "2-b.md": "### Fixed\n\n" + _entry("Same fix ([#1])", "16.0 not 121"),
    })
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert "1-a.md" in err and "2-b.md" in err


def test_a_duplicate_between_the_block_and_an_entry_file_is_caught(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The transition shape: the same entry in both conventions.

    Collation emits the block and then the files, so this reaches the
    released section twice.
    """
    argv = _tree(
        tmp_path,
        _entry("Same fix ([#1])", "in the block"),
        {"1-a.md": "### Fixed\n\n" + _entry("Same fix ([#1])", "in a file")},
    )
    assert main(argv) == 1
    assert "[Unreleased]" in capsys.readouterr().err


def test_distinct_entry_files_pass(tmp_path: Path) -> None:
    """The control: the check is not simply failing on any directory."""
    argv = _tree(tmp_path, "", {
        "1-a.md": "### Fixed\n\n" + _entry("One fix ([#1])", "a"),
        "2-b.md": "### Fixed\n\n" + _entry("Another fix ([#2])", "b"),
    })
    assert main(argv) == 0


def test_a_path_that_is_not_an_entry_file_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A skipped path is an entry compared against nothing.

    The control below shows the directory is otherwise clean, so this
    fails on the stray paths and not on the entry beside them.
    """
    argv = _tree(tmp_path, "", {
        "1-a.md": "### Fixed\n\n" + _entry("One fix ([#1])", "a"),
    })
    directory = tmp_path / "unreleased"
    (directory / "2-b.txt").write_text(
        "### Fixed\n\n" + _entry("Another fix ([#2])", "b") + "\n",
        encoding="utf-8",
    )
    (directory / "sub").mkdir()

    assert main(argv) == 1

    err = capsys.readouterr().err
    assert str(directory / "2-b.txt") in err
    assert str(directory / "sub") in err


def test_an_empty_entry_directory_is_not_an_error(tmp_path: Path) -> None:
    """The steady state right after a release cut."""
    assert main(_tree(tmp_path, "", {})) == 0


def test_a_missing_directory_is_still_an_error(tmp_path: Path) -> None:
    """A gate pointed at nothing must not report a pass."""
    assert main(["check_changelog_dupes.py", str(tmp_path / "gone")]) == 1


def test_the_ci_invocation_names_the_unreleased_directory() -> None:
    """`CHANGELOG/*.md` does not match a directory.

    If the argument is dropped, entry files stop being examined and the
    check still reports green — so assert the argument, not the result.
    """
    directory = (_REPO / "CHANGELOG" / "unreleased").resolve()
    matched = [
        arg for arg in _ci_arguments()
        if any(p.resolve() == directory for p in _REPO.glob(arg))
    ]
    assert matched, (
        "release-docs-check does not pass CHANGELOG/unreleased to "
        "check_changelog_dupes.py; entry files would go unchecked"
    )


def test_the_committed_tree_passes_the_ci_invocation(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the gate's own command line against the tree.

    The workflow's `CHANGELOG/*.md` is expanded by the shell before the
    script sees it, so expand it here too rather than handing the script
    a literal it would report as missing.
    """
    monkeypatch.chdir(_REPO)
    expanded: list[str] = []
    for arg in _ci_arguments():
        matches = sorted(str(p) for p in _REPO.glob(arg))
        expanded.extend(matches or [arg])
    assert main(["check_changelog_dupes.py", *expanded]) == 0
