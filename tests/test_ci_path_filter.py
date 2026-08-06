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


# --- e2e trigger coverage (#1420 §1) -------------------------------------

_E2E_WORKFLOW = _REPO / ".github" / "workflows" / "e2e.yml"


def _block_range(lines: list[str], start: int) -> range:
    """Indices of the block nested under `lines[start]`.

    Index range rather than a list of line *contents*: two sibling blocks
    can hold byte-identical lines (`branches: [main]` appears under both
    triggers in e2e.yml), and membership-testing on content would match
    the wrong one.
    """
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and not line.strip().startswith("#"):
            if len(line) - len(line.lstrip()) <= indent:
                break
        end += 1
    return range(start + 1, end)


def _sole_key(lines: list[str], within: range, key: str, where: str) -> int:
    """The single index in `within` whose stripped content is `key`."""
    hits = [i for i in within if lines[i].strip() == key]
    assert len(hits) == 1, (
        f"expected exactly one `{key}` {where} in {_E2E_WORKFLOW.name}, found "
        f"{len(hits)} — a parser that finds the wrong block, or none, turns "
        f"every assertion below green for free"
    )
    return hits[0]


def _e2e_pull_request_paths() -> list[str]:
    """The glob list under `on.pull_request.paths` in e2e.yml.

    Walks `on:` -> `pull_request:` -> `paths:` by indentation rather than
    grepping the file for `paths:`. The difference is load-bearing twice
    over. A file-wide search cannot tell a `paths:` under `pull_request:`
    from one under `push:`, so it would (a) bind these assertions to the
    wrong trigger without saying so, and (b) fail — with a message about
    parser trust — the moment someone legitimately adds a `push: paths:`,
    which is a confusing signal for an unrelated edit.

    Asserted non-empty for the same reason `_code_filter_globs` is: a
    parser that quietly finds nothing turns every assertion below green
    for free.
    """
    lines = _E2E_WORKFLOW.read_text(encoding="utf-8").splitlines()

    on_key = _sole_key(lines, range(len(lines)), "on:", "top-level key")
    pr_key = _sole_key(lines, _block_range(lines, on_key), "pull_request:", "trigger under `on:`")
    start = _sole_key(lines, _block_range(lines, pr_key), "paths:", "key under `on.pull_request`")
    indent = len(lines[start]) - len(lines[start].lstrip())
    globs: list[str] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        matched = re.fullmatch(r"-\s*'([^']+)'", stripped)
        assert matched, f"unparsed entry in the e2e paths list: {stripped!r}"
        globs.append(matched.group(1))
    assert globs, "the e2e paths list parsed empty"
    return globs


def test_e2e_runs_on_code_prs_without_an_opt_in_label() -> None:
    """#1420 §1: e2e must not require a label to run on a PR.

    Before this, the job carried
    `if: github.event_name == 'push' || contains(...labels..., 'e2e')`, so an
    installed-package regression — which the unit suite structurally cannot
    see, because it imports from the source tree — was caught on `main` after
    merge instead of on the PR. Asserting the absence of the label condition
    is the whole point; a test that only checked the workflow parses would
    have passed in the broken state.
    """
    lines = _E2E_WORKFLOW.read_text(encoding="utf-8").splitlines()
    text = "\n".join(lines)

    # The invariant, not one spelling of its violation: the job carries no
    # `if:` at all. Asserting only the absence of the old label expression
    # lets any re-gate through — `if: github.event_name == 'push'`, a
    # differently-quoted `contains(...)`, or a `vars.`-driven toggle would
    # each fully negate AC1 while keeping that string absent.
    jobs_key = _sole_key(lines, range(len(lines)), "jobs:", "top-level key")
    e2e_job = _sole_key(lines, _block_range(lines, jobs_key), "e2e:", "job under `jobs:`")
    job_body = _block_range(lines, e2e_job)
    job_indent = min(
        (len(lines[i]) - len(lines[i].lstrip()) for i in job_body if lines[i].strip()),
        default=0,
    )
    gating = [
        lines[i].strip()
        for i in job_body
        if lines[i].strip().startswith("if:")
        and len(lines[i]) - len(lines[i].lstrip()) == job_indent
    ]
    assert not gating, (
        f"the `e2e` job is conditionally gated again ({gating}) — AC1 says it "
        f"runs on any code PR, and *any* job-level `if:` can negate that. "
        f"Trigger-level `paths` is where docs-only PRs are excluded."
    )
    assert "labels.*.name, 'e2e'" not in text, (
        "e2e is gated on an opt-in label again — a PR without the label gets "
        "no end-to-end coverage and the regression lands on main"
    )
    assert "types: [opened, synchronize, reopened]" in text, (
        "e2e's pull_request trigger must include `opened`; with only "
        "[labeled, synchronize, reopened] a freshly-opened PR never fires it"
    )


def test_e2e_paths_cover_the_ci_code_filter() -> None:
    """The two path lists must agree, or e2e silently under-triggers.

    `e2e.yml` says to keep its `paths` in sync with ci.yml's `code` filter.
    A comment cannot enforce that: a path added to ci.yml and forgotten here
    means a PR that changes installed behaviour runs the unit suite and skips
    e2e, with nothing red to show for it.

    Each list may legitimately reference its *own* workflow file, so those are
    excluded from the comparison rather than the assertion being loosened.
    """
    ci_globs = {g for g in _code_filter_globs() if not g.startswith(".github/")}
    e2e_globs = {g for g in _e2e_pull_request_paths() if not g.startswith(".github/")}
    missing = sorted(ci_globs - e2e_globs)
    assert not missing, (
        f"{missing} are in ci.yml's `code` filter but not in e2e.yml's "
        f"`paths`. A PR touching only those runs pytest but not e2e."
    )


def test_e2e_still_skips_docs_only_prs() -> None:
    """#1420 §1 AC2, which the subset check above cannot express.

    `test_e2e_paths_cover_the_ci_code_filter` asserts `ci - e2e == {}` — e2e's
    list is a *superset* of ci's. A superset assertion is blind to additions,
    so appending `docs/**` (or `'**'`) to the e2e trigger keeps every other
    test in this module green while putting the 3-leg install matrix on every
    docs-only and CHANGELOG-only PR. That is AC2 reversed, with nothing red to
    show for it.

    The mirror of `test_docs_only_changes_still_skip_the_suite`, which is the
    same guard on the ci.yml side and the reason that side cannot drift.
    """
    globs = _e2e_pull_request_paths()
    for directory in ("docs", "CHANGELOG"):
        assert not _is_covered(directory, globs), (
            f"'{directory}' entered e2e's trigger paths, so docs-only PRs now "
            f"run the install matrix — reversing #1420 §1 AC2. Intentional? "
            f"Confirm the installed-package path can actually regress from a "
            f"prose change first."
        )


def test_e2e_is_path_filtered_at_the_trigger_not_inside_the_job() -> None:
    """Pin *why* the trigger-level filter is safe here.

    A path-filtered **required** context never reports on a PR it skips and
    leaves that PR permanently pending — which is exactly why ci.yml has no
    `paths` and filters inside the job instead. e2e is not a required context,
    so filtering at the trigger is fine and cheaper. If e2e is ever promoted
    to required, this must move inside the job first.
    """
    ci_text = _CI_WORKFLOW.read_text(encoding="utf-8")
    assert "dorny/paths-filter" in ci_text, (
        "ci.yml stopped filtering inside the job; if it gained a trigger-level "
        "`paths` it would leave docs-only PRs permanently pending on the "
        "required pytest contexts"
    )
    assert "dorny/paths-filter" not in _E2E_WORKFLOW.read_text(encoding="utf-8")
