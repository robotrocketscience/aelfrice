"""The eval-calibration trigger must cover everything its metric depends on.

`.github/workflows/eval-calibration.yml` pins `aelf eval --json` output
byte-for-byte against `benchmarks/posterior_ranking/baseline.json`. It
runs on PRs behind a `paths:` filter, and unconditionally on push to
main.

That asymmetry is the hazard. If a PR changes the measured output but
does not match the PR filter, the job never runs on the PR, the PR
merges, and the push-to-main run then fails the baseline assertion with
no owning PR to revert. #1160 found exactly that: the filter named
`eval_harness.py`, `calibration_metrics.py` and `cli.py`, but the metric
is produced by `retrieve()` (`eval_harness.py:167,182`), so the whole
retrieval and scoring stack was outside it.

This test derives the dependency set by walking imports from the
harness rather than restating a list of module names, since a
hand-maintained list is what drifted.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO / ".github" / "workflows" / "eval-calibration.yml"
_PACKAGE = _REPO / "src" / "aelfrice"
_ROOT_MODULE = "eval_harness"


def _pr_path_globs() -> list[str]:
    """Return the `paths:` list under the workflow's `pull_request:` trigger."""
    lines = _WORKFLOW.read_text(encoding="utf-8").splitlines()
    starts = [
        i for i, line in enumerate(lines) if line.strip() == "pull_request:"
    ]
    assert len(starts) == 1, (
        f"expected one `pull_request:` trigger in {_WORKFLOW.name}, found "
        f"{len(starts)} — a parser that finds nothing would make every "
        f"assertion here pass for free"
    )
    paths_at = None
    for i in range(starts[0] + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped == "paths:":
            paths_at = i
            break
        # a new top-level trigger key (e.g. `push:`) ends the block
        if stripped and not lines[i].startswith("    "):
            break
    assert paths_at is not None, (
        "the pull_request trigger has no `paths:` filter — if it was dropped "
        "deliberately the job now runs on every PR, which is also a valid fix "
        "for #1160; delete this test rather than weakening it"
    )
    indent = len(lines[paths_at]) - len(lines[paths_at].lstrip())
    globs: list[str] = []
    for line in lines[paths_at + 1 :]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= indent:
            break
        matched = re.fullmatch(r"-\s*'([^']+)'", stripped)
        assert matched, f"unparsed entry in the paths filter: {stripped!r}"
        globs.append(matched.group(1))
    return globs


def _glob_matches(glob: str, path: str) -> bool:
    """Match a GitHub-Actions path glob, where `**` spans separators."""
    pattern = "".join(
        r"[^/]*" if part == "*" else r".*" if part == "**" else re.escape(part)
        for part in re.split(r"(\*\*|\*)", glob)
    )
    return re.fullmatch(pattern, path) is not None


def _path_is_included(globs: list[str], path: str) -> bool:
    """Whether GitHub would trigger on `path` given `globs`, in order.

    A `paths:` list may negate with `!`, and GitHub evaluates the entries
    top to bottom: a later negative excludes a path an earlier positive
    matched, and a later positive re-includes it. Matching with `any()`
    ignores negation entirely, so `src/aelfrice/**` followed by
    `!src/aelfrice/foo.py` would read as covering `foo.py` when the job
    would in fact skip it — the guard failing *open* in exactly the case
    it exists to catch. There are no negated entries in the filter today;
    this keeps the guard correct if one is ever added.
    """
    included = False
    for glob in globs:
        negated = glob.startswith("!")
        if _glob_matches(glob[1:] if negated else glob, path):
            included = not negated
    return included


def _reachable_modules() -> set[str]:
    """`aelfrice.*` module names transitively imported from the harness.

    Walks the AST, so imports written inside a function body — which is
    how the harness reaches `retrieve` — are found too.
    """
    seen: set[str] = set()
    queue = [_ROOT_MODULE]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        source = _PACKAGE / f"{name}.py"
        if not source.is_file():
            continue
        seen.add(name)
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            found: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("aelfrice."):
                    found.append(node.module.split(".", 1)[1])
            elif isinstance(node, ast.Import):
                found += [
                    alias.name.split(".", 1)[1]
                    for alias in node.names
                    if alias.name.startswith("aelfrice.")
                ]
            queue += [m for m in found if m not in seen]
    return seen


def test_the_import_walk_is_not_vacuous() -> None:
    """Guard the guard: an empty walk would satisfy the coverage test."""
    reachable = _reachable_modules()
    assert {"eval_harness", "retrieval", "scoring"} <= reachable, (
        f"the import walk reached {len(reachable)} modules and is missing the "
        f"ones #1160 is about, so the coverage assertion below is vacuous: "
        f"{sorted(reachable)}"
    )


def test_pr_filter_covers_every_module_the_metric_depends_on() -> None:
    globs = _pr_path_globs()
    uncovered = sorted(
        module
        for module in _reachable_modules()
        if not _path_is_included(globs, f"src/aelfrice/{module}.py")
    )
    assert not uncovered, (
        f"{uncovered} are reachable from `aelf eval` but do not match the "
        f"pull_request paths filter in {_WORKFLOW.name}. A PR touching only "
        f"those files skips this gate, merges, and then fails the "
        f"unconditional push-to-main run with no owning PR. Prefer widening "
        f"to 'src/aelfrice/**' over adding names."
    )


def test_the_push_trigger_stays_unconditional() -> None:
    """The post-merge re-assertion is what makes a subset filter dangerous.

    Kept as an explicit expectation: if the push trigger ever grows a
    `paths:` filter, the failure mode this module guards changes shape
    and the reasoning above needs revisiting.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    push_block = text.split("push:", 1)
    assert len(push_block) == 2, "no `push:` trigger in the workflow"
    following = push_block[1].split("permissions:", 1)[0]
    assert "paths:" not in following, (
        "the push-to-main trigger grew a paths filter; re-check whether a "
        "PR-side subset filter can still leave main red with no owning PR"
    )


def test_a_negated_glob_excludes_a_path_an_earlier_glob_matched() -> None:
    """Order matters, and `any()` would get this wrong.

    Pinned as a unit on the helper rather than by editing the workflow,
    since the filter carries no negation and should not grow one just to
    be tested.
    """
    globs = ["src/aelfrice/**", "!src/aelfrice/scoring.py"]
    assert _path_is_included(globs, "src/aelfrice/retrieval.py")
    assert not _path_is_included(globs, "src/aelfrice/scoring.py")
    # A later positive wins over an earlier negative.
    assert _path_is_included(
        ["src/aelfrice/**", "!src/aelfrice/scoring.py", "src/aelfrice/scoring.py"],
        "src/aelfrice/scoring.py",
    )
    # No entry matches at all -> not included.
    assert not _path_is_included(globs, "docs/README.md")
