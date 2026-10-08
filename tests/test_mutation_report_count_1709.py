"""The mutation report counts method mutants, and its counts add up (#1709).

mutmut 3 names a module-level function's mutants `pkg.mod.x_name__mutmut_N`
and a method's mutants `pkg.mod.xǁClassǁmethod__mutmut_N`. The workflow's
mutant pattern used a `[A-Za-z0-9_.]` key class, which cannot match `ǁ`, so
the report's total skipped every method mutant while the killed and survived
counts, which match on the status, included them.

These tests run the workflow's own step bodies with bash and the system grep
against a fixed report, rather than re-implementing the count in Python. On
the CI runner that is GNU grep, the grep the workflow runs under. Both C and
C.UTF-8 are run because grep reads `ǁ` as two bytes under one and as one
character under the other.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_mutation_workflow_1457 import _step_body

_BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(_BASH is None, reason="needs bash")

# The first three key shapes are copied from a real `mutmut results
# --all=true` run (mutmut 3.8.0) over a module with one function and one
# method; the rest reuse those shapes with other statuses.
_FUNCTION_KILLED = "    pkg.mod.x_kept__mutmut_1: killed"
_METHOD_KILLED = "    pkg.mod.xǁBoxǁshrink__mutmut_1: killed"
_METHOD_SURVIVED = "    pkg.mod.xǁBoxǁshrink__mutmut_2: survived"

# One line per status in mutmut 3's `status_by_exit_code`, half of them on
# method keys.
_OTHER_STATUSES = (
    "no tests",
    "timeout",
    "suspicious",
    "skipped",
    "not checked",
    "caught by type check",
    "segfault",
    "check was interrupted by user",
)


def _report(*extra: str) -> str:
    lines = [_FUNCTION_KILLED, _METHOD_KILLED, _METHOD_SURVIVED]
    for i, status in enumerate(_OTHER_STATUSES):
        key = f"pkg.mod.xǁBoxǁshrink__mutmut_{10 + i}" if i % 2 else (
            f"pkg.mod.x_kept__mutmut_{10 + i}"
        )
        lines.append(f"    {key}: {status}")
    lines.extend(extra)
    return "\n".join(lines) + "\n"


def _run_step(
    name: str, report: str, tmp_path: Path, locale: str
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run step `name`'s `run:` body in `tmp_path` against `report`."""
    (tmp_path / "mutation-report.txt").write_text(report, encoding="utf-8")
    summary = tmp_path / "summary.md"
    summary.write_text("", encoding="utf-8")
    script = tmp_path / "step.sh"
    # Everything after the step's `run: |` line; the step's other keys
    # (`if:`) come before it. The indentation is harmless to bash.
    _, sep, body = _step_body(name).partition("run: |\n")
    assert sep, f"step {name!r} has no `run: |` block"
    script.write_text(body, encoding="utf-8")
    env = {
        **os.environ,
        "LC_ALL": locale,
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    assert _BASH is not None
    proc = subprocess.run(
        [_BASH, "-e", str(script)],
        cwd=tmp_path, env=env, capture_output=True, text=True,
        encoding="utf-8", check=False,
    )
    return proc, summary.read_text(encoding="utf-8")


@pytest.mark.parametrize("locale", ["C", "C.UTF-8"])
def test_the_pr_report_total_counts_method_mutants(
    tmp_path: Path, locale: str
) -> None:
    """AC2/AC3: the table's total counts function and method mutants alike.

    Fails under the `[A-Za-z0-9_.]` key class, which counts only the
    function-mutant lines.
    """
    report = _report()
    proc, summary = _run_step("Report", report, tmp_path, locale)
    assert proc.returncode == 0, proc.stderr
    n_lines = len(report.splitlines())
    assert f"| {n_lines} | 2 | 1 |" in summary, summary


@pytest.mark.parametrize("locale", ["C", "C.UTF-8"])
def test_the_pr_report_counts_add_up_to_the_total(
    tmp_path: Path, locale: str
) -> None:
    """AC4: killed, survived, and every listed status sum to the total.

    Fails if a mutmut status is missing from the step's list, because the
    step then prints an `unlisted status` line.
    """
    proc, summary = _run_step("Report", _report(), tmp_path, locale)
    assert proc.returncode == 0, proc.stderr
    for status in _OTHER_STATUSES:
        assert f"{status}: 1\n" in summary, (status, summary)
    assert "unlisted status" not in summary, summary


def test_the_pr_report_names_a_status_it_does_not_list(tmp_path: Path) -> None:
    """AC4: a status a later mutmut adds is reported, not silently dropped."""
    report = _report("    pkg.mod.xǁBoxǁshrink__mutmut_99: zapped")
    proc, summary = _run_step("Report", report, tmp_path, "C.UTF-8")
    assert proc.returncode == 0, proc.stderr
    assert "unlisted status: 1\n" in summary, summary


@pytest.mark.parametrize("locale", ["C", "C.UTF-8"])
def test_the_weekly_guard_total_counts_method_mutants(
    tmp_path: Path, locale: str
) -> None:
    """AC2/AC3 for the weekly job: its printed total includes method mutants.

    The guard fails the job on `not checked`, so the report here holds only
    killed and survived mutants.
    """
    report = "\n".join([_FUNCTION_KILLED, _METHOD_KILLED, _METHOD_SURVIVED]) + "\n"
    proc, _ = _run_step(
        "Assert the run actually produced results", report, tmp_path, locale
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert re.search(r"mutants: 3 total, 2 killed, 1 survived", proc.stdout), (
        proc.stdout
    )
