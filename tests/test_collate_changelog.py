"""Collation of `CHANGELOG/unreleased/` into a dated section (#1475).

One file per entry removes the merge-time conflict, and buys a
release-time failure mode in exchange: an entry that quietly never
appears. That trade is only acceptable if the quiet failure is pinned,
so the two load-bearing tests here are
`test_every_input_file_contributes_exactly_one_entry` and
`test_the_directory_is_empty_after_a_cut` — the two ways collation can
lose an entry. Both were mutation-checked: `files[:-1]` in `collate`'s
merge loop reddens the first, `paths[:-1]` in `cut`'s unlink loop
reddens the second, and dropping the `sorted()` in `entry_files`
reddens `test_order_does_not_depend_on_filesystem_iteration`.
"""
from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "collate_changelog.py"

_spec = importlib.util.spec_from_file_location("_collate", _SCRIPT)
assert _spec and _spec.loader
collate_changelog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collate_changelog)

CATEGORIES = collate_changelog.CATEGORIES
CollationError = collate_changelog.CollationError
collate = collate_changelog.collate
cut = collate_changelog.cut
entry_files = collate_changelog.entry_files
main = collate_changelog.main
parse_entry_file = collate_changelog.parse_entry_file

HEADER = "\n".join([
    "# Changelog — v4.x",
    "",
    "## [Unreleased]",
    "",
    "",
])

FOOTER = "\n".join([
    "## [4.2.0] - 2026-07-21",
    "",
    "### Added",
    "",
    "- **Something that already shipped ([#1132](u)).** Prose.",
    "",
    "[4.2.0]: https://github.com/robotrocketscience/aelfrice/compare/v4.1.0...v4.2.0",
    "",
])


def _changelog(*block: str) -> str:
    """A changelog whose `[Unreleased]` body is `block`."""
    body = "\n".join(block)
    return HEADER + (body + "\n\n" if body else "") + FOOTER


def _issues(bullets: list[str]) -> list[str]:
    """The issue number each bullet links, in order."""
    return [b.split("([#")[1].split("]")[0] for b in bullets]


def _file(issue: int, category: str = "Fixed") -> tuple[str, str]:
    """An entry file `(name, text)` identifiable by its issue number."""
    return (
        f"{issue}-slug.md",
        f"### {category}\n\n- **Entry {issue} ([#{issue}](u)).** Prose.\n",
    )


def _bullets(section: str) -> list[str]:
    """Top-level bullet lines of one `## [...]` section."""
    out: list[str] = []
    inside = False
    for line in section.split("\n"):
        if line.startswith("## ["):
            inside = line.startswith("## [4.3.0]")
        elif inside and line.startswith("- "):
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# The two mandatory pins: no entry may be lost, and none may be left.
# ---------------------------------------------------------------------------


def test_every_input_file_contributes_exactly_one_entry() -> None:
    """One entry per input file — the silent-drop failure, pinned.

    Counting is not enough on its own: a collation that emitted the
    first file twice and dropped the second would keep the count. So
    assert per-file identity as well.
    """
    files = [_file(n) for n in (101, 102, 103, 104)]
    out = collate(_changelog(), files, "4.3.0", "2026-08-10")

    bullets = _bullets(out)
    assert len(bullets) == len(files)
    for name, _ in files:
        issue = name.split("-")[0]
        matches = [b for b in bullets if f"([#{issue}](u))" in b]
        assert len(matches) == 1, f"{name} appears {len(matches)} times"


def test_the_directory_is_empty_after_a_cut(tmp_path: Path) -> None:
    """`unreleased/` is drained by the cut, README aside.

    A file left behind is re-collated into the *next* release as a
    duplicate of an entry that already shipped.
    """
    directory = tmp_path / "unreleased"
    directory.mkdir()
    (directory / "README.md").write_text("not an entry\n", encoding="utf-8")
    for issue in (101, 102, 103):
        name, text = _file(issue)
        (directory / name).write_text(text, encoding="utf-8")
    changelog = tmp_path / "v4.md"
    changelog.write_text(_changelog(), encoding="utf-8")

    cut(changelog, directory, "4.3.0", "2026-08-10")

    assert entry_files(directory) == []
    assert (directory / "README.md").is_file(), "the note must survive"


def test_a_dry_run_leaves_both_the_file_and_the_directory_alone(
    tmp_path: Path,
) -> None:
    """The emptiness assertion above must be caused by the cut, not by
    running the script at all."""
    directory = tmp_path / "unreleased"
    directory.mkdir()
    name, text = _file(101)
    (directory / name).write_text(text, encoding="utf-8")
    changelog = tmp_path / "v4.md"
    changelog.write_text(_changelog(), encoding="utf-8")

    out = cut(changelog, directory, "4.3.0", "2026-08-10", dry_run=True)

    assert "[#101](u)" in out
    assert [p.name for p in entry_files(directory)] == [name]
    assert changelog.read_text(encoding="utf-8") == _changelog()


# ---------------------------------------------------------------------------
# Determinism and byte-exactness.
# ---------------------------------------------------------------------------


def test_collation_is_byte_identical_to_hand_assembly() -> None:
    """The whole file, not just the new section."""
    block = ["### Fixed", "", "- **Already in the block ([#99](u)).** Prose."]
    files = [_file(102, "Fixed"), _file(101, "Added")]

    out = collate(_changelog(*block), files, "4.3.0", "2026-08-10")

    assert out == HEADER + "\n".join([
        "## [4.3.0] - 2026-08-10",
        "",
        "### Added",
        "",
        "- **Entry 101 ([#101](u)).** Prose.",
        "",
        "### Fixed",
        "",
        "- **Already in the block ([#99](u)).** Prose.",
        "- **Entry 102 ([#102](u)).** Prose.",
    ]) + "\n\n" + FOOTER


def test_order_does_not_depend_on_filesystem_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`glob` order is not collation order.

    Handing back the same directory in the opposite order must not
    change a byte — otherwise the release section's order would depend
    on which machine cut it.
    """
    directory = tmp_path / "unreleased"
    directory.mkdir()
    for issue in (101, 102, 103):
        name, text = _file(issue)
        (directory / name).write_text(text, encoding="utf-8")

    forward = [p.name for p in entry_files(directory)]

    real_glob = Path.glob

    def reversed_glob(self: Path, pattern: str) -> Iterator[Path]:
        return iter(sorted(real_glob(self, pattern), reverse=True))

    monkeypatch.setattr(Path, "glob", reversed_glob)
    assert [p.name for p in entry_files(directory)] == forward
    assert forward == ["101-slug.md", "102-slug.md", "103-slug.md"]


def test_perturbing_the_input_set_changes_the_output() -> None:
    """The reverse of the pin above: collation is not inert."""
    base = collate(_changelog(), [_file(101)], "4.3.0", "2026-08-10")
    more = collate(
        _changelog(), [_file(101), _file(102)], "4.3.0", "2026-08-10"
    )
    assert base != more
    assert len(_bullets(more)) == len(_bullets(base)) + 1


# ---------------------------------------------------------------------------
# Transition: the block stays valid and comes first.
# ---------------------------------------------------------------------------


def test_existing_block_entries_come_first_within_a_category() -> None:
    """No in-flight PR is forced onto the new convention to merge."""
    block = [
        "### Fixed",
        "",
        "- **In the block ([#98](u)).** Prose.",
        "- **Also in the block ([#99](u)).** Prose.",
    ]
    out = collate(_changelog(*block), [_file(101)], "4.3.0", "2026-08-10")
    assert _issues(_bullets(out)) == ["98", "99", "101"]


def test_a_multi_paragraph_block_entry_survives_verbatim() -> None:
    """The v4 block has entries with indented continuation paragraphs."""
    block = [
        "### Fixed",
        "",
        "- **Multi-part ([#98](u)).** First paragraph.",
        "",
        "  **Second paragraph.** Detail.",
        "- **Next ([#99](u)).** Prose.",
    ]
    out = collate(_changelog(*block), [], "4.3.0", "2026-08-10")
    assert "\n".join(block[2:]) in out


def test_the_unreleased_header_survives_the_cut_emptied() -> None:
    """`release-docs-check` reads the block; the next cycle needs it."""
    out = collate(
        _changelog("### Fixed", "", "- **Gone ([#98](u)).** Prose."),
        [], "4.3.0", "2026-08-10",
    )
    unreleased = out.split("## [Unreleased]")[1].split("## [")[0]
    assert unreleased.strip() == ""


def test_collating_an_empty_block_and_no_files_still_cuts_a_section() -> None:
    out = collate(_changelog(), [], "4.3.0", "2026-08-10")
    assert "## [4.3.0] - 2026-08-10" in out
    assert _bullets(out) == []


# ---------------------------------------------------------------------------
# Refusals. A malformed entry file must be loud, never a guess.
# ---------------------------------------------------------------------------


def test_two_bullets_in_one_file_is_an_error() -> None:
    """Two entries in one file re-open the collision being removed."""
    with pytest.raises(CollationError, match="exactly one top-level"):
        parse_entry_file(
            "### Fixed\n\n- **A ([#1](u)).** x\n- **B ([#2](u)).** y\n",
            "1-slug.md",
        )


def test_a_file_with_no_category_heading_is_an_error() -> None:
    with pytest.raises(CollationError, match="exactly one '### <Category>'"):
        parse_entry_file("- **A ([#1](u)).** x\n", "1-slug.md")


def test_an_unknown_category_is_an_error() -> None:
    with pytest.raises(CollationError, match="unknown category 'Fixt'"):
        parse_entry_file("### Fixt\n\n- **A ([#1](u)).** x\n", "1-slug.md")


def test_a_changelog_without_an_unreleased_header_is_an_error() -> None:
    with pytest.raises(CollationError, match="no '## \\[Unreleased\\]'"):
        collate("# Changelog\n", [], "4.3.0", "2026-08-10")


def test_a_missing_entry_directory_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A typo'd `--unreleased` must not read as "no entries this cycle".

    `Path.glob` on a path that does not exist returns `[]` with no
    error, so without this the cut collates nothing, exits 0 and prints
    "[Unreleased] drained into [4.3.0]" — the release ships with every
    entry file still sitting in the real directory.
    """
    changelog = tmp_path / "v4.md"
    before = _changelog("### Fixed", "", "- **Stays ([#98](u)).** Prose.")
    changelog.write_text(before, encoding="utf-8")

    code = main([
        "collate_changelog.py", "--version", "4.3.0", "--date",
        "2026-08-10", "--changelog", str(changelog), "--unreleased",
        str(tmp_path / "unrleased"),
    ])

    assert code == 1
    assert "no such directory" in capsys.readouterr().err
    assert changelog.read_text(encoding="utf-8") == before


def test_main_reports_a_malformed_file_and_changes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "unreleased"
    directory.mkdir()
    (directory / "101-slug.md").write_text(
        "- **No heading ([#101](u)).** x\n", encoding="utf-8"
    )
    changelog = tmp_path / "v4.md"
    changelog.write_text(_changelog(), encoding="utf-8")

    code = main([
        "collate_changelog.py", "--version", "4.3.0", "--date",
        "2026-08-10", "--changelog", str(changelog), "--unreleased",
        str(directory),
    ])

    assert code == 1
    assert "101-slug.md" in capsys.readouterr().err
    assert changelog.read_text(encoding="utf-8") == _changelog()
    assert len(entry_files(directory)) == 1


# ---------------------------------------------------------------------------
# The committed tree.
# ---------------------------------------------------------------------------


def test_the_committed_unreleased_directory_parses() -> None:
    """Every entry file in the tree can actually be collated.

    A file that only fails at release time is the failure mode this
    whole change has to buy off.
    """
    directory = _REPO / "CHANGELOG" / "unreleased"
    assert (directory / "README.md").is_file(), "keeps the dir tracked"
    for path in entry_files(directory):
        category, entry = parse_entry_file(
            path.read_text(encoding="utf-8"), path.name
        )
        assert category in CATEGORIES
        assert entry.startswith("- ")


def test_the_committed_v4_changelog_collates_without_loss() -> None:
    """99 block entries in the real file; none may vanish."""
    text = (_REPO / "CHANGELOG" / "v4.md").read_text(encoding="utf-8")
    before = [
        line for line in
        text.split("## [Unreleased]")[1].split("\n## [")[0].split("\n")
        if line.startswith("- ")
    ]
    directory = _REPO / "CHANGELOG" / "unreleased"
    files = [
        (p.name, p.read_text(encoding="utf-8"))
        for p in entry_files(directory)
    ]

    out = collate(text, files, "4.3.0", "2026-08-10")

    assert sorted(_bullets(out)) == sorted(
        before + [
            parse_entry_file(t, n)[1].split("\n")[0] for n, t in files
        ]
    )


# ---------------------------------------------------------------------------
# `release-docs-check` must know the new location.
#
# The job already refuses a release PR whose `[Unreleased]` block still
# has content. Half the unreleased surface now lives in a directory, and
# a file stranded there is quieter than a stranded bullet: nothing
# renders it, so it reappears only as a duplicate in the next release.
# The shell predicate is extracted from the workflow and run, so these
# fail if it is edited into something that matches nothing.
# ---------------------------------------------------------------------------

_WORKFLOW = _REPO / ".github" / "workflows" / "staging-gate.yml"


def _leftover_command() -> str:
    """The workflow's own `leftover=$(...)` scan, as written."""
    for line in _WORKFLOW.read_text(encoding="utf-8").split("\n"):
        stripped = line.strip()
        if stripped.startswith("leftover=$(") and "unreleased" in stripped:
            return stripped
    raise AssertionError(
        "release-docs-check has no CHANGELOG/unreleased drain check"
    )


def _run_leftover(root: Path) -> list[str]:
    import subprocess

    proc = subprocess.run(
        ["sh", "-c", _leftover_command() + '\nprintf "%s\\n" "$leftover"'],
        cwd=root, capture_output=True, text=True, check=True,
        timeout=20,
    )
    return [line for line in proc.stdout.split("\n") if line]


@pytest.mark.timeout(30)
def test_the_release_gate_lists_a_stranded_entry_file(tmp_path: Path) -> None:
    directory = tmp_path / "CHANGELOG" / "unreleased"
    directory.mkdir(parents=True)
    (directory / "README.md").write_text("note\n", encoding="utf-8")
    (directory / "1475-slug.md").write_text("### Fixed\n", encoding="utf-8")

    assert _run_leftover(tmp_path) == ["CHANGELOG/unreleased/1475-slug.md"]


@pytest.mark.timeout(30)
def test_the_release_gate_ignores_the_directory_note(tmp_path: Path) -> None:
    """A drained directory keeps its README; that is not an entry."""
    directory = tmp_path / "CHANGELOG" / "unreleased"
    directory.mkdir(parents=True)
    (directory / "README.md").write_text("note\n", encoding="utf-8")

    assert _run_leftover(tmp_path) == []
