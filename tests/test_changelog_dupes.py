"""The CHANGELOG duplicate-entry guard (#1211).

Resolving a CHANGELOG conflict insert-only is correct for *added*
entries and wrong for *amended* ones — it leaves the superseded
revision beside the one that replaced it. Two entries reached
`[Unreleased]` that way before this guard existed.
"""
from __future__ import annotations

import importlib.util
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
