"""The CI `code` paths-filter must cover every in-repo package the suite uses.

`.github/workflows/ci.yml` always runs so that `pytest (3.12)` and
`pytest (3.13)` — both required status checks — report on every PR,
including docs-only ones (#413/#427). Which PRs actually *run* the suite
is decided inside the job by a `dorny/paths-filter` glob list. A path the
suite depends on but the filter omits is therefore worse than an
uncovered path: the required check reports success from an `echo`,
having run nothing.

That is what #1160 found. `benchmarks/**` and `scripts/**` were absent
while `pythonpath = ["."]` made both importable and ~30 test modules
used them, so breaking either merged green.

These tests **derive** the expected filter contents by scanning
`tests/` for what it actually imports and reads, rather than comparing
the filter against a hand-maintained list of names. A hand-maintained
list is the failure mode #1161 hit on the doctor side: its drift guard
compared a tuple against the same constants the tuple was copied from,
so it could only fail on a rename, never on the drift that occurred.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CI_WORKFLOW = _REPO / ".github" / "workflows" / "ci.yml"

# A top-level directory holding Python the suite could depend on. Excludes
# directories with no .py at all (`CHANGELOG/`, and the untracked
# `telemetry/` a local run may leave behind), so the scan does not depend
# on working-tree state.
_IN_REPO_PACKAGES = frozenset(
    p.name
    for p in _REPO.iterdir()
    if p.is_dir() and not p.name.startswith(".") and any(p.rglob("*.py"))
)

_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", re.MULTILINE)
_ROOT_PATH_RES = (
    re.compile(r'parents\[1\]\s*/\s*"([^"]+)"'),
    re.compile(r'parent\.parent\s*/\s*"([^"]+)"'),
)


def _code_filter_globs() -> list[str]:
    """Return the glob list under the `code:` key of the paths-filter."""
    lines = _CI_WORKFLOW.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "code:"]
    assert len(starts) == 1, (
        f"expected exactly one `code:` filter key in {_CI_WORKFLOW.name}, "
        f"found {len(starts)} — the parser below cannot be trusted, and a "
        f"parser that silently finds nothing turns every assertion in this "
        f"module green for free."
    )
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    globs: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        matched = re.fullmatch(r"-\s*'([^']+)'", stripped)
        assert matched, f"unparsed entry in the code filter: {stripped!r}"
        globs.append(matched.group(1))
    return globs


def _suite_dependencies() -> set[str]:
    """Top-level in-repo packages that `tests/` imports or reads."""
    found: set[str] = set()
    for path in sorted(_REPO.joinpath("tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found |= {n for n in _IMPORT_RE.findall(text) if n in _IN_REPO_PACKAGES}
        for pattern in _ROOT_PATH_RES:
            found |= {n for n in pattern.findall(text) if n in _IN_REPO_PACKAGES}
    return found


def _is_covered(directory: str, globs: list[str]) -> bool:
    return any(g == directory or g.startswith(f"{directory}/") for g in globs)


def test_the_discovery_scan_is_not_vacuous() -> None:
    """Guard the guard: an empty scan would pass every test below."""
    deps = _suite_dependencies()
    assert {"src", "tests"} <= deps, (
        f"the scan found {sorted(deps)}, which is missing the packages the "
        f"suite unarguably uses — the regexes have stopped matching and the "
        f"coverage assertion below is now vacuous"
    )


def test_code_filter_covers_every_in_repo_package_the_suite_uses() -> None:
    globs = _code_filter_globs()
    uncovered = sorted(d for d in _suite_dependencies() if not _is_covered(d, globs))
    assert not uncovered, (
        f"{uncovered} are used by tests/ but absent from the `code` filter in "
        f"{_CI_WORKFLOW.name}. A PR touching only those paths takes the `echo` "
        f"branch and reports a passing pytest matrix without running the tests "
        f"that cover them. Add '<dir>/**' to the filter."
    )


def test_benchmarks_and_scripts_are_in_the_code_filter() -> None:
    """Pin the #1160 regression by name, not only by derivation."""
    globs = _code_filter_globs()
    for directory in ("benchmarks", "scripts"):
        assert _is_covered(directory, globs), (
            f"'{directory}/**' dropped out of the code filter; this is the "
            f"#1160 defect returning"
        )


def test_docs_only_changes_still_skip_the_suite() -> None:
    """`docs/` is excluded deliberately, and that costs no coverage.

    Verified by mutation rather than assumed: truncating `README.md` to a
    quarter of its length leaves the full suite passing, so no test
    asserts on prose content. Re-run that check before widening the
    filter here — if documentation ever becomes load-bearing for a test,
    this expectation is the thing that should change first.
    """
    globs = _code_filter_globs()
    for directory in ("docs", "CHANGELOG"):
        assert not _is_covered(directory, globs), (
            f"'{directory}' entered the code filter, so docs-only PRs now run "
            f"the full matrix — reversing #413/#427. Intentional? Confirm a "
            f"test actually depends on those files first."
        )
