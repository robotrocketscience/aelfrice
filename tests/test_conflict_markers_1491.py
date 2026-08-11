"""The repo carries no committed merge-conflict residue (#1491).

`CHANGELOG/v4.md` sat on `main` with three marker lines and the suite
was fully green, because nothing read any tracked file for shape. The
live-repo test below is the one that would have caught it; the unit
tests beside it exist so that a passing live scan means the scanner
works, rather than that it looked at nothing.

Marker literals are constructed rather than written out, so this file
does not trip the check it exercises.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "check_conflict_markers.py"

_spec = importlib.util.spec_from_file_location("_ccm", _SCRIPT)
assert _spec and _spec.loader
check_conflict_markers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_conflict_markers)

find_markers = check_conflict_markers.find_markers
scan = check_conflict_markers.scan
tracked_files = check_conflict_markers.tracked_files

OPEN = "<" * 7
CLOSE = ">" * 7
BASE = "|" * 7
SEP = "=" * 7


def _conflicted(ours: str, theirs: str) -> str:
    """The exact shape `git merge` leaves in a file it could not resolve."""
    return "\n".join([
        "intro line",
        f"{OPEN} HEAD",
        ours,
        SEP,
        theirs,
        f"{CLOSE} be03af9f (docs(changelog): a subject line)",
        "tail line",
    ])


def test_the_v4_incident_shape_is_caught() -> None:
    """Three lines reported, at the positions git wrote them."""
    found = find_markers(_conflicted("- **ours**", "- **theirs**"))

    assert [n for n, _ in found] == [2, 4, 6]
    assert found[0][1].startswith(OPEN)
    assert found[1][1] == SEP
    assert found[2][1].startswith(CLOSE)


@pytest.mark.parametrize("ancestor", [
    BASE,
    f"{BASE} merged common ancestors",
    f"{BASE} main",
])
def test_diff3_style_base_marker_is_caught(ancestor: str) -> None:
    """`merge.conflictStyle=diff3` writes a fourth marker.

    Bare from `git merge-file`, labelled from a merge or a
    `--conflict=diff3` checkout. The labelled forms are the ones a real
    merge produces, so a rule matching only the bare line would miss
    every marker this repo could actually acquire.
    """
    text = "\n".join([f"{OPEN} HEAD", "a", ancestor, "base", SEP, "b",
                      f"{CLOSE} x"])

    assert [n for n, _ in find_markers(text)] == [1, 3, 5, 7]


def test_a_markdown_table_row_of_empty_cells_is_not_a_conflict() -> None:
    """The control for the ancestor marker, and why it is conditional.

    Seven `|` is a table row of six empty cells. It is legal Markdown
    and would be reported by an unconditional rule — so the ancestor
    marker, like the separator, counts only beside an open or close
    marker.
    """
    table = "\n".join(["| a | b |", "| - | - |", BASE, "| c | d |"])

    assert find_markers(table) == []


def test_a_setext_underline_alone_is_not_a_conflict() -> None:
    """The control. Without it the separator rule could be unconditional.

    Markdown Setext headings and reStructuredText section rules are
    lines of `=`; both occur in `docs/`. Reporting them would make the
    check unrunnable and it would be turned off, not fixed.
    """
    assert find_markers("\n".join(["A heading", SEP, "", "Body text."])) == []
    assert find_markers("\n".join([SEP, "Title", SEP])) == []


def test_a_separator_is_caught_when_the_file_also_conflicts() -> None:
    """A partly hand-cleaned conflict still reports every line."""
    text = "\n".join(["Heading", SEP, "prose", f"{CLOSE} deadbeef (subject)"])

    assert [n for n, _ in find_markers(text)] == [2, 4]


def test_a_marker_needs_its_trailing_space() -> None:
    """`<<<<<<<` opens a conflict; `<<<<<<<<` in prose about shell here-docs
    or a run of arrows in a diagram does not."""
    assert find_markers(f"{OPEN}{OPEN}\n{CLOSE}{CLOSE}") == []


def test_scan_skips_binary_without_counting_it(tmp_path: Path) -> None:
    """A repository that is all images must not report a vacuous pass."""
    binary = tmp_path / "logo.png"
    binary.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x00\x01")
    text = tmp_path / "notes.md"
    text.write_text("clean\n", encoding="utf-8")

    read, findings = scan([binary, text])

    assert read == 1, "the binary must not be counted as scanned"
    assert findings == []


def test_a_symlink_is_read_as_its_target_string(tmp_path: Path) -> None:
    """git stores a symlink's target, so that is what gets scanned.

    Following the link would scan whatever it points at — content that
    may be outside the repository and tracked by nobody — and would
    report findings against a path whose committed blob is clean.
    """
    outside = tmp_path / "outside.md"
    outside.write_text(_conflicted("x", "y"), encoding="utf-8")
    link = tmp_path / "link.md"
    link.symlink_to(outside)

    read, findings = scan([link])

    assert read == 1
    assert findings == [], "the link's own content is a path, not residue"


def test_a_broken_symlink_is_read_rather_than_skipped(tmp_path: Path) -> None:
    """A dangling link still has content: the target string git stored."""
    link = tmp_path / "dangling.md"
    link.symlink_to(tmp_path / "does-not-exist.md")

    read, findings = scan([link])

    assert read == 1, "a broken link must not silently shrink the scan"
    assert findings == []


def test_an_unreadable_file_fails_the_scan_instead_of_shrinking_it(
    tmp_path: Path,
) -> None:
    """The hole a bare `except OSError: continue` leaves.

    A skip is indistinguishable from a clean file in the summary, so a
    path that cannot be read would quietly reduce coverage while the
    check went on reporting success. It raises instead, and `main`
    turns that into exit 2 rather than exit 0.
    """
    missing = tmp_path / "vanished.md"

    with pytest.raises(check_conflict_markers.UnreadableFile) as caught:
        scan([missing])

    assert "vanished.md" in str(caught.value)
    assert check_conflict_markers.main(["prog", str(missing)]) == 2


def test_scan_reports_the_file_and_line(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text(_conflicted("x", "y"), encoding="utf-8")

    read, findings = scan([doc])

    assert read == 1
    assert [(p, n) for p, n, _ in findings] == [(doc, 2), (doc, 4), (doc, 6)]


@pytest.mark.parametrize("marker", [OPEN, CLOSE])
def test_no_tracked_file_carries_a_marker(marker: str) -> None:
    """The live guard, and the one the incident needed.

    Parameterised so a failure names which marker was found. The
    non-vacuity assert matters as much as the emptiness one: if
    `git ls-files` returned nothing the scan would pass having read no
    file, which is how this class of check quietly stops working.
    """
    read, findings = scan(tracked_files(_REPO))

    assert read > 100, f"only {read} tracked text files scanned"
    hits = [(str(p), n, line) for p, n, line in findings
            if line.startswith(marker)]
    assert hits == [], f"conflict markers committed: {hits}"


def test_the_live_scan_reports_no_separator_either() -> None:
    """Separators are conditional, so they need their own live assert."""
    _, findings = scan(tracked_files(_REPO))

    assert [(str(p), n) for p, n, _ in findings] == []
