"""End-to-end regression: aelf bench produces a JSON report meeting the
v0.9.0-rc score floor.

Marks the v0.9.0-rc benchmark milestone as observable from outside the
package: a user with `aelf` on their PATH can run `aelf bench` and
publish the resulting JSON without touching internals.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import cast

import pytest

from aelfrice.benchmark import BENCHMARK_NAME
from aelfrice.cli import main as cli_main

pytestmark = pytest.mark.regression


def _run_cli(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = cli_main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _parse_report(stdout: str) -> dict[str, object]:
    parsed = json.loads(stdout)  # pyright: ignore[reportAny]
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


@pytest.mark.timeout(30)
def test_aelf_bench_default_in_memory_meets_score_floor() -> None:
    code, output = _run_cli("bench")
    assert code == 0
    report = _parse_report(output)
    assert report["benchmark_name"] == BENCHMARK_NAME
    assert report["corpus_size"] == 16
    assert report["query_count"] == 16
    hit5 = report["hit_at_5"]
    assert isinstance(hit5, float)
    assert hit5 >= 0.75, f"hit_at_5={hit5} below v0.9.0-rc floor 0.75"


@pytest.mark.timeout(30)
def test_aelf_bench_with_explicit_db_writes_file(tmp_path: Path) -> None:
    db = tmp_path / "bench.db"
    code, output = _run_cli("bench", "--db", str(db))
    assert code == 0
    assert db.exists()
    report = _parse_report(output)
    assert report["corpus_size"] == 16


@pytest.mark.timeout(30)
def test_aelf_bench_top_k_override_changes_report_field() -> None:
    code, output = _run_cli("bench", "--top-k", "3")
    assert code == 0
    report = _parse_report(output)
    assert report["top_k"] == 3
