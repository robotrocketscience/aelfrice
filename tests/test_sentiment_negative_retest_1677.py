"""The #1677 negative-sentiment re-test producer selects, samples, and scores."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import sentiment_negative_retest_1677 as rt


def _row(ts: str, prompt: str, *, abstained: str = "negative_disabled",
         hook: str = "sentiment_feedback", session: str = "s1",
         targets: list[str] | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "ts": ts, "hook": hook, "session_id": session, "prompt_prefix": prompt,
        "sentiment": "negative", "pattern": "wrong", "matched_text": "wrong",
        "belief_ids": [], "n_beliefs": 0, "abstained": abstained,
    }
    if targets is not None:
        row["target_ids"] = targets
    return row


def test_wilson_lower_matches_a_known_value() -> None:
    assert rt.wilson_lower(7, 10) == pytest.approx(0.3968, abs=1e-4)
    assert rt.wilson_lower(0, 0) == 0.0


def test_the_population_is_fresh_short_disabled_negative_fires() -> None:
    keep = _row("2026-10-02T00:00:00Z", "no, that's wrong", targets=["F1"])
    rows = [
        keep,
        _row("2026-09-30T23:59:59Z", "no, that's wrong"),  # before the window
        _row("2026-10-02T00:00:00Z", "x" * 200),  # at the cap: may be truncated
        _row("2026-10-02T00:00:00Z", "still broken", abstained="no_prior_injection"),
        _row("2026-10-02T00:00:00Z", "no", hook="user_prompt_submit"),
        _row("2026-10-02T00:00:00Z", ""),
        dict(keep),  # the same fire read from a rotated file
    ]
    assert rt.fresh_negative_fires(rows) == [keep]


def test_a_prompt_one_under_the_cap_is_kept() -> None:
    row = _row("2026-10-02T00:00:00Z", "x" * 199)
    assert rt.fresh_negative_fires([row]) == [row]


def test_the_sample_is_seeded_capped_and_order_independent() -> None:
    fires = [_row(f"2026-10-0{1 + i % 9}T00:00:{i:02d}Z", f"wrong {i}") for i in range(30)]
    pop = rt.fresh_negative_fires(fires)
    one = rt.draw_sample(pop, seed=1677, max_n=10)
    again = rt.draw_sample(rt.fresh_negative_fires(list(reversed(fires))), seed=1677, max_n=10)
    other = rt.draw_sample(pop, seed=7, max_n=10)
    assert len(one) == 10
    assert one == again
    assert [r["id"] for r in one] != [r["id"] for r in other]
    assert len(rt.draw_sample(pop, seed=1677, max_n=100)) == 30


def test_score_reports_each_grader_and_the_bar() -> None:
    ids = [f"f{i}" for i in range(10)]
    a = {i: n < 8 for n, i in enumerate(ids)}  # 8 of 10
    b = {i: n < 6 for n, i in enumerate(ids)}  # 6 of 10
    report = rt.score(ids, a, b)
    assert [g["precision"] for g in report["graders"]] == [0.8, 0.6]
    assert report["graders"][0]["wilson_lower"] == pytest.approx(rt.wilson_lower(8, 10))
    assert report["agreed"] == 8 and report["kappa_n"] == 10
    assert report["passed"] is False
    b_pass = {i: n < 7 for n, i in enumerate(ids)}  # exactly the bar
    assert rt.score(ids, a, b_pass)["passed"] is True


def test_an_unclear_fire_is_left_out_of_that_graders_figures() -> None:
    ids = ["f0", "f1", "f2"]
    a: dict[str, bool | None] = {"f0": True, "f1": True, "f2": None}
    b: dict[str, bool | None] = {"f0": True, "f1": False, "f2": True}
    report = rt.score(ids, a, b)
    assert report["graders"][0]["graded"] == 2
    assert report["graders"][1]["graded"] == 3
    assert report["kappa_n"] == 2


def test_score_refuses_a_sheet_fire_without_both_labels() -> None:
    with pytest.raises(ValueError, match="lack a label"):
        rt.score(["f0", "f1"], {"f0": True, "f1": True}, {"f0": True})


def test_the_cli_counts_samples_and_scores(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    audit = tmp_path / "hook_audit.jsonl"
    fires = [_row(f"2026-10-02T00:00:{i:02d}Z", f"wrong {i}", targets=["F1"]) for i in range(4)]
    audit.write_text("".join(json.dumps(r) + "\n" for r in fires), encoding="utf-8")
    assert rt.main(["count", "--audit", str(audit)]) == 0
    assert "fires=4 with_targets=4" in capsys.readouterr().out
    sheet = tmp_path / "sheet.jsonl"
    assert rt.main(["sample", "--audit", str(audit), "--seed", "1", "--dry-run"]) == 0
    assert not sheet.exists()
    assert rt.main(["sample", "--audit", str(audit), "--seed", "1"]) == 1
    assert rt.main(["sample", "--audit", str(audit), "--seed", "1", "--out", str(sheet)]) == 0
    ids = [json.loads(line)["id"] for line in sheet.read_text(encoding="utf-8").splitlines()]
    for name, value in (("a", True), ("b", True)):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps({"id": i, "correct": value}) + "\n" for i in ids), encoding="utf-8")
    capsys.readouterr()
    labels = [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]
    assert rt.main(["score", "--sheet", str(sheet), "--labels", *labels]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    (tmp_path / "b.jsonl").write_text(
        "".join(json.dumps({"id": i, "correct": False}) + "\n" for i in ids), encoding="utf-8")
    assert rt.main(["score", "--sheet", str(sheet), "--labels", *labels]) == 1
