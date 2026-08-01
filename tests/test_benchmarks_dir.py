"""Smoke tests for the academic benchmark suite scaffold (P1).

Distinct from `tests/test_benchmark.py` (singular), which covers the
in-tree synthetic regression harness at `src/aelfrice/benchmark.py`.
This file covers the top-level `benchmarks/` directory: contamination
gate, score utilities, and the `aelf bench` dispatcher's handling of
inert academic-suite targets.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import main as cli_main


def _run_cli(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = cli_main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def test_verify_clean_passes_on_clean_file(tmp_path: Path) -> None:
    """verify_clean accepts a retrieval file with only safe keys."""
    from benchmarks import verify_clean

    clean_file = tmp_path / "retrieval.json"
    clean_file.write_text(json.dumps([
        {"id": "q1", "question": "what?", "retrieved_context": "..."},
        {"id": "q2", "question": "why?", "retrieved_context": "..."},
    ]))
    assert verify_clean.verify_file(str(clean_file)) is True


def test_verify_clean_rejects_contaminated_file(tmp_path: Path) -> None:
    """verify_clean rejects a retrieval file containing ground truth."""
    from benchmarks import verify_clean

    bad_file = tmp_path / "retrieval.json"
    bad_file.write_text(json.dumps([
        {"id": "q1", "question": "what?", "answer": "leaked!"},
    ]))
    assert verify_clean.verify_file(str(bad_file)) is False


def test_verify_clean_rejects_missing_file(tmp_path: Path) -> None:
    from benchmarks import verify_clean

    assert verify_clean.verify_file(str(tmp_path / "nope.json")) is False


@pytest.mark.parametrize(
    "target",
    ["mab", "locomo", "longmemeval", "structmemeval", "amabench"],
)
def test_aelf_bench_inert_targets_exit_2_with_pointer(target: str) -> None:
    """Each scaffolded but not-yet-runnable target exits 2 with a
    pointer to benchmarks/README.md so users know where to look.

    `all` is no longer in this list — it dispatches via
    benchmarks.run.main_all() per the v2.0 reproducibility harness
    (#437). See test_aelf_bench_all_requires_out below.
    """
    code, output = _run_cli("bench", target)
    assert code == 2
    assert "benchmarks/README.md" in output
    assert target in output


def test_aelf_bench_all_requires_out() -> None:
    """`aelf bench all` without --out exits 2 with a clear message (#437)."""
    code, output = _run_cli("bench", "all")
    assert code == 2
    assert "--out PATH is required" in output


def test_aelf_bench_unknown_target_exits_2() -> None:
    code, output = _run_cli("bench", "not-a-real-target")
    assert code == 2
    assert "unknown target" in output


def test_aelf_bench_default_still_runs_synthetic_harness() -> None:
    """Backward-compat: bare `aelf bench` must remain the synthetic
    regression harness (it's the v0.9.0-rc → v1.0.0 contract).
    """
    code, output = _run_cli("bench")
    assert code == 0
    report = json.loads(output)
    assert report["benchmark_name"] == "aelfrice-bench-v1"
    assert report["corpus_size"] == 16


def test_aelf_bench_synthetic_target_explicit_runs_synthetic() -> None:
    """`aelf bench synthetic` is an alias for the default."""
    code, output = _run_cli("bench", "synthetic")
    assert code == 0
    report = json.loads(output)
    assert report["corpus_size"] == 16


def test_aelf_bench_verify_clean_target_redirects_to_module() -> None:
    """The dev-only target now points users at `python -m benchmarks.verify_clean`."""
    code, output = _run_cli("bench", "verify-clean")
    assert code == 2
    assert "python -m benchmarks.verify_clean" in output


def test_aelf_bench_longmemeval_score_target_redirects_to_module() -> None:
    code, output = _run_cli("bench", "longmemeval-score")
    assert code == 2
    assert "python -m benchmarks.longmemeval_score" in output


def test_aelf_bench_posterior_residual_target_redirects_to_module() -> None:
    code, output = _run_cli("bench", "posterior-residual")
    assert code == 2
    assert "python -m benchmarks.posterior_ranking" in output


def test_aelf_bench_unknown_target_lists_moved_dev_targets() -> None:
    code, output = _run_cli("bench", "no-such-target")
    assert code == 2
    assert "verify-clean" in output
    assert "longmemeval-score" in output
    assert "posterior-residual" in output


def test_benchmarks_package_imports() -> None:
    """The benchmarks/ directory is a valid (empty) package."""
    import benchmarks
    assert benchmarks is not None


@pytest.mark.timeout(120)
def test_posterior_channel_audit_holds_the_documented_defaults() -> None:
    """`benchmarks/posterior_channel_audit.py` exits 0 on the shipped defaults.

    The script is the regression guard behind the LIMITATIONS entry that
    says no automatic channel moves a posterior (#1267). A guard nothing
    invokes cannot keep a doc entry honest, and no workflow runs
    `benchmarks/*.py` — `bench-smoke` only runs two named test modules.
    Running it from the required `pytest` job is what makes the claim
    enforceable.

    Driven as a subprocess rather than imported: the script deletes every
    ambient `AELFRICE_*` variable at import time so it measures defaults
    rather than the developer's opt-ins, and that must not leak into the
    rest of the test session.
    """
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "benchmarks" / "posterior_channel_audit.py"
    assert script.is_file(), f"missing {script}"

    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert proc.returncode == 0, (
        f"posterior-channel audit failed (exit {proc.returncode}); a "
        f"documented default moved:\n{proc.stdout}\n{proc.stderr}"
    )
