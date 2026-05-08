"""Tests for benchmarks.run — the v2.0 reproducibility dispatcher.

The dispatcher subprocesses each adapter, so all tests stub the runner
to avoid spawning real benchmarks. Spec: docs/v2_reproducibility_harness.md.
Issue: #437.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks import run as bench_run


def _make_runner(per_adapter_payloads: dict[str, dict] | None = None,
                 per_adapter_returncode: dict[str, int] | None = None):
    """Return a stub runner that writes a canned JSON to out_path.

    `per_adapter_payloads` keys can be either the adapter name ("mab")
    or the adapter/sub_key form ("mab/Conflict_Resolution"); the runner
    matches the most specific key it can.
    """
    per_adapter_payloads = per_adapter_payloads or {}
    per_adapter_returncode = per_adapter_returncode or {}

    def runner(cmd: list[str], out_path: Path) -> subprocess.CompletedProcess[str]:
        # cmd shape: [python, -m, benchmarks.<adapter>_adapter, ..., --output, PATH]
        adapter_module = cmd[2]
        adapter = adapter_module.removeprefix("benchmarks.").removesuffix("_adapter")
        sub_key = None
        for flag in ("--split", "--task"):
            if flag in cmd:
                idx = cmd.index(flag)
                sub_key = cmd[idx + 1]
                break
        compound_key = f"{adapter}/{sub_key}" if sub_key else adapter
        rc = per_adapter_returncode.get(compound_key,
                                        per_adapter_returncode.get(adapter, 0))
        payload = per_adapter_payloads.get(compound_key,
                                           per_adapter_payloads.get(adapter,
                                                                    {"score": 0.5}))
        if rc == 0:
            with out_path.open("w") as f:
                json.dump(payload, f)
        return subprocess.CompletedProcess(
            args=cmd, returncode=rc, stdout="", stderr="adapter said: data missing" if rc == 2 else "",
        )

    return runner


def test_smoke_invocations_count(tmp_path):
    """Smoke registry has the documented 2 entries."""
    assert len(bench_run.SMOKE_INVOCATIONS) == 2
    adapters = {i.adapter for i in bench_run.SMOKE_INVOCATIONS}
    assert adapters == {"mab", "amabench"}


def test_canonical_invocations_count(tmp_path):
    """Canonical registry: 4 MAB + 1 LoCoMo + 1 LongMemEval + 4 StructMemEval + 1 AMA = 11."""
    assert len(bench_run.CANONICAL_INVOCATIONS) == 11
    by_adapter: dict[str, int] = {}
    for inv in bench_run.CANONICAL_INVOCATIONS:
        by_adapter[inv.adapter] = by_adapter.get(inv.adapter, 0) + 1
    assert by_adapter == {
        "mab": 4, "locomo": 1, "longmemeval": 1,
        "structmemeval": 4, "amabench": 1,
    }


def test_smoke_run_writes_schema_v2(tmp_path):
    """End-to-end: smoke run with stubbed adapters produces a schema-v2 JSON."""
    out = tmp_path / "smoke.json"
    rc = bench_run.main_all(
        out_path=out, canonical=False, smoke=True,
        runner=_make_runner({"mab/Conflict_Resolution": {"f1": 0.6},
                             "amabench": {"f1": 0.4}}),
        tmp_root=tmp_path / "tmp",
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["schema_version"] == 2
    assert data["label"] == "v2.0.0 smoke"
    assert "mab" in data["results"]
    assert "amabench" in data["results"]
    assert data["results"]["mab"]["Conflict_Resolution"]["_status"] == "ok"
    assert data["results"]["mab"]["Conflict_Resolution"]["output"] == {"f1": 0.6}


def test_canonical_run_with_partial_adapters_refuses(tmp_path):
    """--canonical + --adapters=mab should refuse: cut doesn't match."""
    out = tmp_path / "bad.json"
    with pytest.raises(SystemExit, match="refusing to write --canonical"):
        bench_run.main_all(
            out_path=out, canonical=True, adapters=("mab",),
            runner=_make_runner(), tmp_root=tmp_path / "tmp",
        )
    assert not out.exists()


def test_canonical_run_with_full_set_accepted(tmp_path):
    """--canonical with no --adapters filter accepts the full registry."""
    out = tmp_path / "canon.json"
    rc = bench_run.main_all(
        out_path=out, canonical=True, smoke=False,
        runner=_make_runner(), tmp_root=tmp_path / "tmp",
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["label"] == "v2.0.0 canonical"
    # All 11 invocations represented.
    assert sum(
        len([k for k in v if k != "_"]) or (1 if "_" in v else 0)
        for v in data["results"].values()
    ) == 11


def test_skipped_data_missing_propagates(tmp_path):
    """Adapter exit-code 2 → status=skipped_data_missing, overall rc=2."""
    out = tmp_path / "skip.json"
    rc = bench_run.main_all(
        out_path=out, canonical=False, smoke=True,
        runner=_make_runner(per_adapter_returncode={"amabench": 2}),
        tmp_root=tmp_path / "tmp",
    )
    assert rc == 2
    data = json.loads(out.read_text())
    assert data["results"]["amabench"]["_"]["_status"] == "skipped_data_missing"
    assert "data missing" in data["results"]["amabench"]["_"]["_error_message"]


def test_error_overrides_skip(tmp_path):
    """If any adapter is `error` AND another is `skip`, overall rc=1 (error wins)."""
    out = tmp_path / "err.json"
    rc = bench_run.main_all(
        out_path=out, canonical=False, smoke=True,
        runner=_make_runner(per_adapter_returncode={
            "amabench": 2, "mab/Conflict_Resolution": 1,
        }),
        tmp_root=tmp_path / "tmp",
    )
    assert rc == 1


def test_unknown_adapter_filter_raises(tmp_path):
    """Filter to a nonexistent adapter → SystemExit."""
    out = tmp_path / "x.json"
    with pytest.raises(SystemExit, match="no adapters matched"):
        bench_run.main_all(
            out_path=out, canonical=False,
            adapters=("nonexistent_adapter",),
            runner=_make_runner(), tmp_root=tmp_path / "tmp",
        )


def test_headline_cut_recorded(tmp_path):
    """The merged JSON includes the headline_cut declaration."""
    out = tmp_path / "h.json"
    rc = bench_run.main_all(
        out_path=out, canonical=True, smoke=False,
        runner=_make_runner(), tmp_root=tmp_path / "tmp",
    )
    assert rc == 0
    data = json.loads(out.read_text())
    cut = data["headline_cut"]
    # MAB has 4 sub-invocations; LoCoMo has 1.
    assert len(cut["mab"]) == 4
    assert len(cut["locomo"]) == 1
    # LongMemEval is full → no flags in args (override from spec).
    locome_args = cut["longmemeval"][0]["args"]
    assert locome_args == []
    # StructMemEval --bench big in every entry (override from spec).
    for entry in cut["structmemeval"]:
        assert "big" in entry["args"]


def test_per_question_detail_stripped_from_merged_output(tmp_path):
    """The merged JSON drops `per_question` and `per_case` lists — they bloat
    the file and aren't read by the band-check (#437 calibration finding
    2026-05-06; `per_case` added 2026-05-08 after structmemeval bloated the
    canonical to 6.4 MB).
    """
    out = tmp_path / "stripped.json"
    payload = {
        "f1": 0.5, "exact_match": 0.3,
        "per_question": [{"id": i, "score": 0.5} for i in range(2000)],
        "per_case": [{"case_id": f"c{i}", "accuracy": 1.0} for i in range(50)],
    }
    rc = bench_run.main_all(
        out_path=out, canonical=False, smoke=True,
        runner=_make_runner({"mab/Conflict_Resolution": payload,
                             "amabench": payload}),
        tmp_root=tmp_path / "tmp",
    )
    assert rc == 0
    data = json.loads(out.read_text())
    mab_out = data["results"]["mab"]["Conflict_Resolution"]["output"]
    ama_out = data["results"]["amabench"]["_"]["output"]
    assert "per_question" not in mab_out
    assert "per_question" not in ama_out
    assert "per_case" not in mab_out
    assert "per_case" not in ama_out
    # Summary metrics retained.
    assert mab_out["f1"] == 0.5
    assert ama_out["exact_match"] == 0.3


def test_runner_crash_recorded_as_error(tmp_path):
    """If the runner raises, status=error and message captured."""
    def crashing_runner(cmd, out_path):
        raise RuntimeError("boom")
    out = tmp_path / "crash.json"
    rc = bench_run.main_all(
        out_path=out, canonical=False, smoke=True,
        runner=crashing_runner, tmp_root=tmp_path / "tmp",
    )
    assert rc == 1
    data = json.loads(out.read_text())
    err = data["results"]["amabench"]["_"]
    assert err["_status"] == "error"
    assert "boom" in err["_error_message"]
