"""The documented dev setup must install what the workflows install (#1548).

`CONTRIBUTING.md` § Development setup told a contributor to run
`uv sync --all-groups`. `--all-groups` installs dependency *groups*, and
`archive` is an *extra*, so that command left `cryptography` uninstalled.
Eight uninstall-archive tests then skipped locally and ran in CI, and
`pyright` gained 23 phantom errors in `src/aelfrice/lifecycle.py` (31 with
the extra, 54 without), which fails `scripts/check_pyright_baseline.py` on a
correctly set-up machine. Both failures are silent at both ends: the local
run is green, and CI never sees the weaker environment.

The drift is what these tests pin, not the specific extra. If a later change
adds an extra to `ci.yml` and not to the setup block, the same gap reopens
with a different name, so the assertion is a subset relation between what the
workflows install and what the documentation tells a human to install
(#1548 AC3/AC4).

Parsed by line rather than with PyYAML on purpose: `yaml` is not in this
project's dependency set — it reaches a local venv only as a transitive
dependency of optional extras, so an `import yaml` here passes locally and
fails under CI's `uv sync --frozen --group dev --extra archive`. The parser
is small enough to be wrong quietly, so `test_parse_sync_extras_*` pins its
shape directly instead of trusting it through the assertions above.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.conftest import (
    ARCHIVE_EXTRA_NAME,
    ARCHIVE_EXTRA_SKIP_REASON,
    _report_archive_extra_skips,
)

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO / ".github" / "workflows"
_CONTRIBUTING = _REPO / "CONTRIBUTING.md"

# `--all-extras` is not a name, so it cannot live in the same set as one. It
# is the top of the lattice: a documented setup carrying it satisfies every
# workflow, whatever extras are added later.
ALL_EXTRAS = "*"

_SYNC_RE = re.compile(r"\buv sync\b(?P<args>.*)")
_EXTRA_RE = re.compile(r"--extra[= ]+(?P<name>[A-Za-z0-9._-]+)")

# The four modules whose archive gates AC2 covers. Stated rather than
# discovered, because a discovery pass that finds nothing would assert
# nothing and read as a pass (#1161).
_ARCHIVE_GATED_MODULES = (
    "test_lifecycle.py",
    "test_cli_uninstall_gates.py",
    "test_uninstall_artifacts.py",
    "test_uninstall_dotdir.py",
)
_EXPECTED_ARCHIVE_GATES = 8


def _parse_sync_extras(command: str) -> set[str]:
    """The extras a `uv sync` command line installs.

    `{ALL_EXTRAS}` when the line carries `--all-extras`; otherwise every
    name given to `--extra`. A line with neither installs no extra, which
    is the defect this module exists for, so it returns an empty set
    rather than raising.
    """
    m = _SYNC_RE.search(command)
    if m is None:
        raise AssertionError(f"not a `uv sync` command line: {command!r}")
    args = m.group("args")
    if "--all-extras" in args:
        return {ALL_EXTRAS}
    return {e.group("name") for e in _EXTRA_RE.finditer(args)}


def _covers(documented: set[str], required: set[str]) -> bool:
    """Does the documented sync install everything `required` names?"""
    if ALL_EXTRAS in documented:
        return True
    if ALL_EXTRAS in required:
        # A workflow syncing `--all-extras` is only covered by a documented
        # setup that does the same; naming today's extras one by one would
        # silently stop covering it the day an extra is added.
        return False
    return required <= documented


def _job_block(workflow: str, job: str) -> list[str]:
    """The lines of one job in a workflow, by indentation.

    A job header sits at indent 2 under `jobs:`; the block runs to the
    next line at indent 2 or less that is not blank.
    """
    lines = (_WORKFLOWS / workflow).read_text(encoding="utf-8").splitlines()
    want = f"  {job}:"
    start = next((i for i, ln in enumerate(lines) if ln.rstrip() == want), None)
    assert start is not None, f"no job {job!r} in {workflow}"
    out: list[str] = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip(" ")) <= 2:
            break
        out.append(line)
    assert out, f"job {job!r} in {workflow} has no body"
    return out


def _workflow_sync_extras(workflow: str, job: str) -> set[str]:
    block = _job_block(workflow, job)
    syncs = [ln for ln in block if "uv sync" in ln and not ln.lstrip().startswith("#")]
    assert len(syncs) == 1, (
        f"expected exactly one `uv sync` in {workflow} job {job!r}, got {syncs}"
    )
    return _parse_sync_extras(syncs[0])


def _documented_sync_line() -> str:
    """The `uv sync` line inside the § Development setup fenced block."""
    lines = _CONTRIBUTING.read_text(encoding="utf-8").splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.strip() == "## Development setup"),
        None,
    )
    assert start is not None, "CONTRIBUTING.md has no `## Development setup` heading"
    fence_open = next(
        (i for i in range(start, len(lines)) if lines[i].startswith("```")), None
    )
    assert fence_open is not None, "the setup section opens no fenced block"
    fence_close = next(
        (i for i in range(fence_open + 1, len(lines)) if lines[i].startswith("```")),
        None,
    )
    assert fence_close is not None, "the setup block is never closed"
    syncs = [
        ln for ln in lines[fence_open + 1 : fence_close] if "uv sync" in ln
    ]
    assert len(syncs) == 1, f"expected one `uv sync` in the setup block, got {syncs}"
    return syncs[0]


# --------------------------------------------------------------------------
# AC4: the documented setup and the workflows cannot drift apart silently.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("workflow", "job"),
    [
        # The job behind the `pytest (3.12)` / `pytest (3.13)` required
        # checks — the green that actually gates a merge.
        ("ci.yml", "pytest"),
        # The setup block runs `check_pyright_baseline.py`, and this job is
        # the environment that check's baseline is recorded in.
        ("pyright-ratchet.yml", "ratchet"),
    ],
)
def test_the_documented_setup_installs_what_the_workflow_installs(
    workflow: str, job: str
) -> None:
    documented = _parse_sync_extras(_documented_sync_line())
    required = _workflow_sync_extras(workflow, job)
    assert _covers(documented, required), (
        f"CONTRIBUTING.md § Development setup installs extras {sorted(documented)}, "
        f"but {workflow} job {job!r} installs {sorted(required)}. A contributor "
        f"following the documentation runs a weaker suite than the one that "
        f"gates the merge. Widen the setup command, or say in the setup block "
        f"which command installs the rest."
    )


def test_the_archive_extra_is_one_of_the_extras_ci_installs() -> None:
    """Guard against the assertion above passing because both sides are empty.

    `_covers` is true for two empty sets, so the drift test alone would go
    green if the `uv sync` lines lost their extras entirely. This names the
    extra the issue is about, so at least one real element is in play.
    """
    assert ARCHIVE_EXTRA_NAME in _workflow_sync_extras("ci.yml", "pytest")


# --------------------------------------------------------------------------
# The parser, pinned directly rather than through the assertions above.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("uv sync --all-groups --all-extras", {ALL_EXTRAS}),
        ("uv sync --frozen --all-extras", {ALL_EXTRAS}),
        ("        run: uv sync --frozen --group dev --extra archive", {"archive"}),
        ("      - run: uv sync --frozen --group dev --extra archive --extra benchmarks",
         {"archive", "benchmarks"}),
        ("uv sync --extra=archive", {"archive"}),
        # The shape the defect had: groups only, no extra.
        ("uv sync --all-groups", set()),
        ("uv sync --frozen --group dev", set()),
    ],
)
def test_parse_sync_extras_reads_the_shapes_the_workflows_use(
    command: str, expected: set[str]
) -> None:
    assert _parse_sync_extras(command) == expected


def test_parse_sync_extras_refuses_a_line_that_is_not_a_sync() -> None:
    with pytest.raises(AssertionError, match="not a `uv sync` command line"):
        _parse_sync_extras("uv run pytest tests/ -q")


@pytest.mark.parametrize(
    ("documented", "required", "covered"),
    [
        ({ALL_EXTRAS}, {"archive"}, True),
        ({ALL_EXTRAS}, {ALL_EXTRAS}, True),
        ({"archive"}, {"archive"}, True),
        ({"archive", "benchmarks"}, {"archive"}, True),
        ({"archive"}, {"archive", "benchmarks"}, False),
        # Naming today's extras does not cover a workflow that syncs them all.
        ({"archive", "benchmarks", "onboard-llm"}, {ALL_EXTRAS}, False),
        (set(), {"archive"}, False),
    ],
)
def test_covers_is_a_subset_relation_with_all_extras_on_top(
    documented: set[str], required: set[str], covered: bool
) -> None:
    assert _covers(documented, required) is covered


# --------------------------------------------------------------------------
# AC2: a skip caused by a missing optional dependency names the extra.
# --------------------------------------------------------------------------


def test_the_skip_reason_names_the_extra_and_the_command_that_installs_it() -> None:
    """`pytest -rs` must tell a contributor what to install, not what failed
    to import."""
    assert f"'{ARCHIVE_EXTRA_NAME}' extra" in ARCHIVE_EXTRA_SKIP_REASON
    assert "uv sync" in ARCHIVE_EXTRA_SKIP_REASON
    assert "--all-extras" in ARCHIVE_EXTRA_SKIP_REASON


def test_every_archive_gate_uses_the_shared_helper() -> None:
    """A bare `importorskip("cryptography")` reports the import, not the fix.

    Counted, not merely absent: a module that lost its gates entirely would
    also have no bare call, and would read as a pass.
    """
    bare = re.compile(r"importorskip\(\s*\n?\s*\"cryptography\"")
    found = 0
    for name in _ARCHIVE_GATED_MODULES:
        text = (_REPO / "tests" / name).read_text(encoding="utf-8")
        assert not bare.search(text), (
            f"tests/{name} skips on `cryptography` without naming the extra; "
            f"use `requires_archive_extra()` from tests/conftest.py"
        )
        found += text.count("requires_archive_extra()")
    assert found == _EXPECTED_ARCHIVE_GATES, (
        f"expected {_EXPECTED_ARCHIVE_GATES} archive gates across "
        f"{list(_ARCHIVE_GATED_MODULES)}, found {found}"
    )


class _FakeReport:
    """Enough of a pytest skip report for the summary hook to classify."""

    def __init__(self, reason: str) -> None:
        self.longrepr = ("path.py", 1, reason)


class _FakeReporter:
    def __init__(self, skipped: list[_FakeReport]) -> None:
        self.stats = {"skipped": skipped}
        self.lines: list[str] = []
        self.seps: list[str] = []

    def write_sep(self, _char: str, title: str) -> None:
        self.seps.append(title)

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def test_the_terminal_summary_names_the_extra_when_archive_gates_skip() -> None:
    rep = _FakeReporter([_FakeReport(ARCHIVE_EXTRA_SKIP_REASON)] * 8)
    _report_archive_extra_skips(rep)
    assert rep.seps == ["optional extras"]
    assert len(rep.lines) == 1
    assert "8 test(s) skipped" in rep.lines[0]
    assert ARCHIVE_EXTRA_NAME in rep.lines[0]
    assert "--all-extras" in rep.lines[0]


def test_the_terminal_summary_is_silent_when_nothing_skipped_for_the_extra() -> None:
    """Classification is by reason, so an unrelated skip must not be folded in."""
    rep = _FakeReporter([_FakeReport("no corpus root; bench-gate test")])
    _report_archive_extra_skips(rep)
    assert rep.seps == []
    assert rep.lines == []
