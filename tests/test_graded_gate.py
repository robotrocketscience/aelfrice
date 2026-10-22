"""The two-grader scorer shared by the #1677 and #1650 graded gates."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks import graded_gate as gg


def test_the_wilson_interval_matches_a_known_value() -> None:
    lower, upper = gg.wilson_interval(7, 10)
    assert math.isclose(lower, 0.3968, abs_tol=1e-4)
    assert math.isclose(upper, 0.8922, abs_tol=1e-4)
    assert gg.wilson_lower(7, 10) == lower
    assert gg.wilson_interval(0, 0) == (0.0, 0.0)


def test_an_unclear_label_is_left_out_of_precision_and_kappa() -> None:
    ids = ["i0", "i1", "i2"]
    a: dict[str, bool | None] = {"i0": True, "i1": True, "i2": None}
    b: dict[str, bool | None] = {"i0": True, "i1": False, "i2": False}
    assert gg.grader_figures(ids, a)["graded"] == 2
    assert gg.grader_figures(ids, a)["precision"] == 1.0
    # Only i0 and i1 count: agreement 1/2 against chance 1/2. Counting the
    # unclear i2 as False would give 0.4.
    assert gg.paired_kappa(ids, a, b) == (0.0, 2, 1)


def test_the_verdict_needs_min_graded_items_on_each_grader() -> None:
    enough = {"graded": gg.MIN_GRADED, "precision": 1.0}
    short = {"graded": gg.MIN_GRADED - 1, "precision": 1.0}
    assert gg.verdict([enough, enough]) == "pass"
    assert gg.verdict([enough, short]) == "insufficient"


def test_the_verdict_needs_the_bar_on_each_grader() -> None:
    at_bar = {"graded": 40, "precision": gg.BAR}
    below = {"graded": 40, "precision": gg.BAR - 0.01}
    assert gg.verdict([at_bar, at_bar]) == "pass"
    assert gg.verdict([at_bar, below]) == "fail"


def test_require_labels_names_the_missing_count() -> None:
    gg.require_labels(["i0"], {"i0": True}, {"i0": None})
    with pytest.raises(ValueError, match="1 pairs lack a label from both graders"):
        gg.require_labels(["i0", "i1"], {"i0": True, "i1": True}, {"i0": True}, "pairs")


def test_read_labels_refuses_a_label_outside_true_false_null(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps({"id": "i0", "correct": None}) + "\n\n", encoding="utf-8")
    assert gg.read_labels(path) == {"i0": None}
    path.write_text(json.dumps({"id": "i0", "correct": "yes"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="a.jsonl:1"):
        gg.read_labels(path)
