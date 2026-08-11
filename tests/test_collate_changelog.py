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

An entry can be lost by halves as well as whole, so the same standard
applies to its body: `entry = lines[starts[0]]` in `parse_entry_file`
— keeping the bullet, discarding every continuation paragraph —
reddens `test_a_multi_paragraph_entry_file_survives_verbatim` and
`test_the_committed_v4_changelog_collates_without_loss`. The latter
compares whole entry blocks, and derives its expectation without the
parser it is checking.
"""
from __future__ import annotations

import importlib.util
import subprocess
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
scan_entry_dir = collate_changelog.scan_entry_dir

# The duplicate check keeps its own copy of the same walk — both are
# standalone stdlib scripts. Loaded here so the two can be compared
# against each other and against the workflow's scan in one place.
_DUPES = _REPO / "scripts" / "check_changelog_dupes.py"
_dupes_spec = importlib.util.spec_from_file_location("_dupes", _DUPES)
assert _dupes_spec and _dupes_spec.loader
check_changelog_dupes = importlib.util.module_from_spec(_dupes_spec)
_dupes_spec.loader.exec_module(check_changelog_dupes)

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
    """Directory order is not collation order.

    Handing back the same directory in the opposite order must not
    change a byte — otherwise the release section's order would depend
    on which machine cut it. `iterdir` is what the walk actually reads,
    so that is what is perturbed.
    """
    directory = tmp_path / "unreleased"
    directory.mkdir()
    for issue in (101, 102, 103):
        name, text = _file(issue)
        (directory / name).write_text(text, encoding="utf-8")

    forward = [p.name for p in entry_files(directory)]

    real_iterdir = Path.iterdir

    def reversed_iterdir(self: Path) -> Iterator[Path]:
        return iter(sorted(real_iterdir(self), reverse=True))

    monkeypatch.setattr(Path, "iterdir", reversed_iterdir)
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


def test_a_multi_paragraph_entry_file_survives_verbatim() -> None:
    """The twin of the block test below, for the new convention.

    An entry's continuation paragraphs are most of its text — on the
    committed #1475 entry file the bullet line is under a third of the
    entry. Collation is a re-arrangement and never a re-wrap, so the
    whole block has to arrive byte for byte; keeping only the bullet
    would drop the rest and leave a plausible-looking one-liner in the
    release. No character count is quoted: the entry is edited like any
    other prose, and a figure in a docstring nothing re-derives goes
    stale silently.
    """
    entry = "\n".join([
        "- **Multi-part ([#101](u)).** First paragraph.",
        "",
        "  **Second paragraph.** Detail.",
        "",
        "  Third paragraph, indented like the committed entries.",
    ])
    out = collate(
        _changelog(),
        [("101-slug.md", f"### Fixed\n\n{entry}\n")],
        "4.3.0", "2026-08-10",
    )
    assert entry in out


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


def test_entry_file_content_before_the_bullet_is_an_error() -> None:
    """The entry-file twin of the block's refusal.

    `parse_entry_file` reads from the first `- ` onward, so anything
    above it that is not the category heading is discarded — and unlike
    the block, nothing else ever reads this file, so the loss is silent
    once and permanent. The block grew this refusal when a
    release-summary paragraph turned out to be a live way to lose prose;
    an entry file is now the default authoring surface and takes the
    same prose.
    """
    entry = "- **A ([#1](u)).** x"
    with pytest.raises(CollationError, match="'Context for this entry.'"):
        parse_entry_file(
            "\n".join(["Context for this entry.", "", "### Fixed", "", entry]),
            "1-slug.md",
        )
    with pytest.raises(CollationError, match="'#### Retrieval'"):
        parse_entry_file(
            "\n".join(["### Fixed", "", "#### Retrieval", "", entry]),
            "1-slug.md",
        )


def test_a_heading_after_the_bullet_is_an_error() -> None:
    """Otherwise it is swallowed into the body and re-emitted inline.

    The one-heading and one-bullet counts both pass in this shape, so
    order is the only thing that catches it.
    """
    with pytest.raises(CollationError, match="after the entry on line"):
        parse_entry_file(
            "- **A ([#1](u)).** x\n\n### Fixed\n", "1-slug.md"
        )


def test_block_content_that_is_not_an_entry_is_an_error() -> None:
    """A note paragraph in the block has nowhere to be collated to.

    Dropping it is silent twice: the prose never reaches the dated
    section, and `release-docs-check`'s drain check then reads the
    emptied block and passes. Every dated section in v0-v4 opens with a
    release-summary paragraph, so this shape is one keystroke away.
    """
    with pytest.raises(CollationError, match="neither a '### <Category>'"):
        collate(
            _changelog(
                "### Fixed",
                "",
                "This release concentrates on retrieval.",
                "",
                "- **A ([#98](u)).** Prose.",
            ),
            [], "4.3.0", "2026-08-10",
        )


def test_a_sub_heading_before_the_first_entry_is_an_error() -> None:
    """The other shape of the same loss, and above any category."""
    entry = "- **A ([#98](u)).** x"
    with pytest.raises(CollationError, match="'#### Retrieval'"):
        collate(
            _changelog("### Fixed", "", "#### Retrieval", "", entry),
            [], "4.3.0", "2026-08-10",
        )
    with pytest.raises(CollationError, match="neither a '### <Category>'"):
        collate(
            _changelog("Summary.", "", "### Fixed", "", entry),
            [], "4.3.0", "2026-08-10",
        )


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


def _entry_blocks(text: str) -> list[str]:
    """Every `- ` bullet with its continuation lines, verbatim.

    Re-derived here rather than reusing `parse_entry_file`: an
    expectation computed with the code under test cannot catch that
    code dropping something. A `#` heading closes the open block.
    """
    blocks: list[str] = []
    inside = False
    for line in text.split("\n"):
        if line.startswith("- "):
            blocks.append(line)
            inside = True
        elif line.startswith("#"):
            inside = False
        elif inside:
            blocks[-1] += "\n" + line
    return [block.rstrip() for block in blocks]


def test_the_committed_v4_changelog_collates_without_loss() -> None:
    """Whole entries, not opening lines. None may vanish, none truncate.

    Most of an entry's text is its indented continuation paragraphs —
    the bullet line of the committed entry file is under a third of it
    — so comparing first lines would pass just as happily on a
    collation that discarded every paragraph after the first.

    No entry count is quoted here. 96a196a1 deleted the pinned
    `assert len(expected) == 99` on purpose: a count re-couples this
    test to the block's contents, which is the coupling #1475 removes.
    """
    text = (_REPO / "CHANGELOG" / "v4.md").read_text(encoding="utf-8")
    block = text.split("## [Unreleased]")[1].split("\n## [")[0]
    directory = _REPO / "CHANGELOG" / "unreleased"
    files = [
        (p.name, p.read_text(encoding="utf-8"))
        for p in entry_files(directory)
    ]

    out = collate(text, files, "4.3.0", "2026-08-10")

    expected = _entry_blocks(block)
    for name, file_text in files:
        one = _entry_blocks(file_text)
        assert len(one) == 1, f"{name} holds {len(one)} entries"
        expected += one
    # Deliberately NOT a pinned count. Pinning one would re-couple this
    # branch to the `[Unreleased]` block's contents — the exact coupling
    # #1475 exists to remove — and every in-flight PR that adds an entry
    # would turn this red for a reason that has nothing to do with
    # collation. What must hold is that the comparison below is not
    # vacuously true against an empty set, and that every entry survives.
    #
    # The guard counts BOTH surfaces, and the ordering is the whole point.
    # Against the block alone it is a time bomb with a known fuse: this
    # convention drains `[Unreleased]` at the next release cut and
    # CONTRIBUTING.md then forbids refilling it, so a block-only guard
    # goes red on the first release PR and stays red on main afterwards —
    # while entry files, the surface that replaces it, sit right there
    # unexamined.
    if not expected:
        pytest.skip("no unreleased entries on either surface; nothing to prove")
    section = out.split("## [4.3.0] - 2026-08-10")[1].split("\n## [")[0]
    assert sorted(_entry_blocks(section)) == sorted(expected)


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
    """The workflow's drain check, from its directory guard onward.

    The guard is part of the predicate, not preamble: the step opens
    `set -euo pipefail`, so without it a missing directory makes `find`
    exit 1 and the assignment itself kills the step. Extracting the
    assignment alone would leave that invisible to every test here.
    """
    lines = _WORKFLOW.read_text(encoding="utf-8").split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("if [ ! -d CHANGELOG/unreleased ]"):
            start = i
        if line.strip().startswith("leftover=$(") and "unreleased" in line:
            assert start is not None, (
                "the drain check lost its directory guard; under "
                "`set -e` the assignment becomes the failure"
            )
            body = lines[start:i + 1]
            indent = len(body[0]) - len(body[0].lstrip())
            return "\n".join(ln[indent:] for ln in body)
    raise AssertionError(
        "release-docs-check has no CHANGELOG/unreleased drain check"
    )


def _run_leftover(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the predicate in the shell the workflow actually gives it.

    `set -euo pipefail` is the step's first line. Running without it was
    what hid the missing-directory abort: the harness returned 0 and an
    empty list where the shipped step exits 1.
    """
    return subprocess.run(
        [
            "sh", "-c",
            "set -euo pipefail\n"
            + _leftover_command()
            + '\nprintf "%s\\n" "$leftover"',
        ],
        cwd=root, capture_output=True, text=True, check=False,
        timeout=20,
    )


def _leftovers(root: Path) -> list[str]:
    proc = _run_leftover(root)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return [line for line in proc.stdout.split("\n") if line]


@pytest.mark.timeout(30)
def test_the_release_gate_lists_a_stranded_entry_file(tmp_path: Path) -> None:
    directory = tmp_path / "CHANGELOG" / "unreleased"
    directory.mkdir(parents=True)
    (directory / "README.md").write_text("note\n", encoding="utf-8")
    (directory / "1475-slug.md").write_text("### Fixed\n", encoding="utf-8")

    assert _leftovers(tmp_path) == ["CHANGELOG/unreleased/1475-slug.md"]


@pytest.mark.timeout(30)
def test_the_release_gate_ignores_the_directory_note(tmp_path: Path) -> None:
    """A drained directory keeps its README; that is not an entry."""
    directory = tmp_path / "CHANGELOG" / "unreleased"
    directory.mkdir(parents=True)
    (directory / "README.md").write_text("note\n", encoding="utf-8")

    assert _leftovers(tmp_path) == []


@pytest.mark.timeout(30)
def test_the_release_gate_names_a_missing_entry_directory(
    tmp_path: Path,
) -> None:
    """The third call site now names the path the other two already do.

    Without the guard this is not a nicer message, it is a different
    step: `find` exits 1 on a missing directory, `set -euo pipefail`
    turns the assignment into the step's failure, and `2>/dev/null`
    swallows the one diagnostic — so the README roadmap check and the
    ROADMAP warning below it never run either.
    """
    (tmp_path / "CHANGELOG").mkdir()

    proc = _run_leftover(tmp_path)

    assert proc.returncode == 1
    assert "CHANGELOG/unreleased" in proc.stdout + proc.stderr
    assert "::error" in proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# A path that is not an entry file must be loud in all three places.
#
# Globbing `*.md` one level deep is a filter, and a filter is silent
# about what it drops. The collator, the duplicate check and the
# release gate all filtered the same way, so a `notes.txt` — or an
# entry hidden one directory down — was invisible to all three at once:
# never collated, never dupe-checked, never reported as stranded, exit
# 0 with a success message. That is the release-time silence this whole
# convention has to buy off, so it is an error naming the path.
# ---------------------------------------------------------------------------

# One of each way a path can fail to be a top-level `<issue>-<slug>.md`.
_STRAY_NAMES = ("102-b.txt", "103-c.markdown", "104-d", "105-e.MD", "sub")


def _mixed_tree(root: Path) -> Path:
    """`CHANGELOG/unreleased/` holding one real entry and five strays."""
    directory = root / "CHANGELOG" / "unreleased"
    directory.mkdir(parents=True)
    (directory / "README.md").write_text("note\n", encoding="utf-8")
    name, text = _file(101)
    (directory / name).write_text(text, encoding="utf-8")
    for stray in _STRAY_NAMES:
        if stray == "sub":
            (directory / stray).mkdir()
            hidden, hidden_text = _file(106)
            (directory / stray / hidden).write_text(
                hidden_text, encoding="utf-8"
            )
        else:
            (directory / stray).write_text(text, encoding="utf-8")
    return directory


def test_a_path_that_is_not_an_entry_file_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Named, not skipped — and the cut writes nothing."""
    directory = _mixed_tree(tmp_path)

    entries, stray = scan_entry_dir(directory)

    assert [p.name for p in entries] == ["101-slug.md"]
    assert [str(p.relative_to(directory)) for p in stray] == [
        "102-b.txt", "103-c.markdown", "104-d", "105-e.MD",
        "sub", "sub/106-slug.md",
    ]

    with pytest.raises(CollationError) as raised:
        entry_files(directory)
    for path in stray:
        assert str(path) in str(raised.value)

    changelog = tmp_path / "v4.md"
    changelog.write_text(_changelog(), encoding="utf-8")
    code = main([
        "collate_changelog.py", "--version", "4.3.0", "--date",
        "2026-08-10", "--changelog", str(changelog), "--unreleased",
        str(directory),
    ])

    assert code == 1
    assert "102-b.txt" in capsys.readouterr().err
    assert changelog.read_text(encoding="utf-8") == _changelog()
    assert (directory / "101-slug.md").is_file(), "nothing unlinked"


@pytest.mark.timeout(30)
def test_the_three_call_sites_see_the_same_paths(tmp_path: Path) -> None:
    """The collator, the duplicate check and the release gate agree.

    The two scripts must classify identically — a path one collates and
    the other skips is an entry that ships unchecked. The gate must be
    *broader* than either: it lists everything but the note, so the
    only way to satisfy it is a directory genuinely drained.
    """
    directory = _mixed_tree(tmp_path)

    assert scan_entry_dir(directory) == (
        check_changelog_dupes.scan_entry_dir(directory)
    )

    entries, stray = scan_entry_dir(directory)
    assert sorted(_leftovers(tmp_path)) == sorted(
        str(p.relative_to(tmp_path)) for p in entries + stray
    )
