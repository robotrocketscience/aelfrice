"""PR-smoke tests for the four benchmark adapters (#476).

Each test invokes one adapter's `--retrieve-only` CLI path against a
schema-matching micro-fixture under `tests/fixtures/bench_smoke/`,
parses the JSON output, and asserts non-zero retrieved-belief counts.

Scope is dispatcher-contract validation only: does the adapter
`load → ingest → retrieve` sequence run end-to-end against
schema-shaped input? Real-data shape coverage lives in the nightly
`bench-canonical` cron, not here.

Total wall-budget for all four adapters: 2 minutes (issue #476
acceptance). Individual per-adapter timeout: 60s.

Requires the `[benchmarks]` install extras (`nltk`, `tiktoken`,
`datasets`, `huggingface_hub`). When those extras are absent the
whole module is skipped — the CI job `bench-smoke` installs them,
the default `pytest` run does not.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Skip the entire module if benchmark extras are not installed.
# The bench-smoke CI job installs them; the default pytest run does
# not, and these tests should not pollute that run.
_REQUIRED_EXTRAS = ("nltk", "tiktoken", "datasets")
for _mod in _REQUIRED_EXTRAS:
    pytest.importorskip(_mod, reason=f"bench-smoke requires [benchmarks] extras (missing {_mod})")


_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
_FIXTURE_DIR: Path = _REPO_ROOT / "tests" / "fixtures" / "bench_smoke"


def _run_adapter(module: str, args: list[str], out_path: Path) -> subprocess.CompletedProcess[str]:
    """Run `python -m benchmarks.<module> <args> --retrieve-only <out>`."""
    cmd: list[str] = [
        sys.executable, "-m", f"benchmarks.{module}",
        *args,
        "--retrieve-only", str(out_path),
    ]
    return subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _load_items(out_path: Path) -> list[dict[str, object]]:
    """Parse the retrieve-only JSON; assert it exists and is a non-empty list."""
    assert out_path.exists(), f"adapter did not write {out_path}"
    with out_path.open("r", encoding="utf-8") as f:
        items: list[dict[str, object]] = json.load(f)
    assert isinstance(items, list), f"expected list, got {type(items)}"
    assert len(items) > 0, "adapter wrote zero retrieval items"
    return items


def _context_len(item: dict[str, object]) -> int:
    """LoCoMo/MAB/StructMemEval use `context`; LongMemEval uses `retrieved_context`."""
    for key in ("retrieved_context", "context"):
        val = item.get(key)
        if isinstance(val, str):
            return len(val)
    return 0


@pytest.mark.timeout(60)
def test_locomo_smoke(tmp_path: Path) -> None:
    """LoCoMo adapter retrieves against the micro fixture."""
    out: Path = tmp_path / "locomo_out.json"
    proc = _run_adapter(
        "locomo_adapter",
        ["--data", str(_FIXTURE_DIR / "locomo_micro.json")],
        out,
    )
    assert proc.returncode == 0, (
        f"locomo_adapter exited {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    items = _load_items(out)
    for i, item in enumerate(items):
        assert _context_len(item) > 0, f"locomo item {i} has empty context"


@pytest.mark.timeout(60)
def test_mab_smoke(tmp_path: Path) -> None:
    """MAB adapter retrieves against the micro fixture (all four splits in one file)."""
    out: Path = tmp_path / "mab_out.json"
    proc = _run_adapter(
        "mab_adapter",
        ["--data", str(_FIXTURE_DIR / "mab_micro.json")],
        out,
    )
    assert proc.returncode == 0, (
        f"mab_adapter exited {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    items = _load_items(out)
    for i, item in enumerate(items):
        assert _context_len(item) > 0, f"mab item {i} has empty context"


@pytest.mark.timeout(60)
def test_longmemeval_smoke(tmp_path: Path) -> None:
    """LongMemEval adapter retrieves against the micro fixture."""
    out: Path = tmp_path / "lme_out.json"
    proc = _run_adapter(
        "longmemeval_adapter",
        ["--data", str(_FIXTURE_DIR / "longmemeval_micro.json")],
        out,
    )
    assert proc.returncode == 0, (
        f"longmemeval_adapter exited {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    items = _load_items(out)
    for i, item in enumerate(items):
        # LongMemEval also exposes num_beliefs in its retrieve-only output.
        assert _context_len(item) > 0, f"lme item {i} has empty retrieved_context"
        n_beliefs = item.get("num_beliefs", 0)
        assert isinstance(n_beliefs, int) and n_beliefs > 0, (
            f"lme item {i} reports num_beliefs={n_beliefs}"
        )


@pytest.mark.timeout(60)
@pytest.mark.parametrize("task", ["location", "accounting", "recommendations", "tree"])
def test_structmemeval_smoke(tmp_path: Path, task: str) -> None:
    """StructMemEval adapter retrieves against the micro fixture, one task at a time."""
    out: Path = tmp_path / f"structmem_{task}_out.json"
    proc = _run_adapter(
        "structmemeval_adapter",
        [
            "--data", str(_FIXTURE_DIR / "structmemeval_micro"),
            "--task", task,
        ],
        out,
    )
    assert proc.returncode == 0, (
        f"structmemeval_adapter task={task} exited {proc.returncode}\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    items = _load_items(out)
    for i, item in enumerate(items):
        assert _context_len(item) > 0, f"structmemeval task={task} item {i} has empty context"
