"""#915 StructMemEval reader-prediction scoring seam.

`score_predictions` completes the `--retrieve-only` workflow: an external
reader (a sub-agent) produces predictions from the dumped context, and this
function grades them with the task-aware `score_prediction` instead of the
raw-retrieval `check_state_correctness`. These tests lock the seam —
matching on (case_id, question), per-item task dispatch, unmatched
accounting, and (the load-bearing one) that a raw-retrieval-style answer is
now scored WRONG because it routes through the task-aware scorer.
"""
from __future__ import annotations

from benchmarks.structmemeval_adapter import score_predictions

_LOC_REF = (
    "Since you currently live in Sydney (Bondi), your Saturday activity "
    "is sunrise yoga on Bondi Beach. The answer should NOT mention "
    "activities from your previous cities (Lisbon, Tokyo, London)."
)
_TREE_REF = "Yes, they are indirect colleagues according to their graph relations."


def _gt(rows: list[tuple[str, str, str]]) -> list[dict[str, object]]:
    return [
        {"case_id": c, "question": q, "reference_answer": r} for c, q, r in rows
    ]


def test_correct_location_prediction_scores_correct() -> None:
    gt = _gt([("loc_1", "Where do you live and what is Saturday?", _LOC_REF)])
    preds = [{
        "case_id": "loc_1",
        "question": "Where do you live and what is Saturday?",
        "prediction": "You live in Sydney (Bondi); Saturday is sunrise yoga on Bondi Beach.",
        "task": "location",
    }]
    out = score_predictions(preds, gt, default_task="location")
    assert out["total"] == 1
    assert out["correct"] == 1
    assert out["score_pct"] == 100.0
    assert out["unmatched"] == 0


def test_raw_retrieval_style_answer_scores_wrong() -> None:
    # The whole point of #914/#915: an answer that names every stale city
    # (what raw retrieval surfaces) scored HIGH under check_state_correctness.
    # Routed through score_prediction it must be WRONG (leaks distractors).
    gt = _gt([("loc_1", "Where do you live?", _LOC_REF)])
    preds = [{
        "case_id": "loc_1",
        "question": "Where do you live?",
        "prediction": "Sydney Bondi yoga, and previously Lisbon, Tokyo, London.",
        "task": "location",
    }]
    out = score_predictions(preds, gt, default_task="location")
    assert out["correct"] == 0
    assert out["score_pct"] == 0.0


def test_per_item_task_dispatch() -> None:
    gt = _gt([
        ("loc_1", "loc?", _LOC_REF),
        ("tree_1", "connected?", _TREE_REF),
    ])
    preds = [
        {"case_id": "loc_1", "question": "loc?",
         "prediction": "Sydney Bondi sunrise yoga Bondi Beach.", "task": "location"},
        {"case_id": "tree_1", "question": "connected?",
         "prediction": "Yes, a path connects them.", "task": "tree"},
    ]
    out = score_predictions(preds, gt)
    assert out["correct"] == 2
    assert out["total"] == 2


def test_default_task_used_when_item_omits_task() -> None:
    gt = _gt([("tree_1", "connected?", _TREE_REF)])
    preds = [{"case_id": "tree_1", "question": "connected?",
              "prediction": "No, disconnected components."}]
    out = score_predictions(preds, gt, default_task="tree")
    # Opposite polarity ⇒ wrong, but it must have dispatched to the tree scorer.
    assert out["correct"] == 0
    assert out["total"] == 1


def test_unmatched_prediction_counted_not_scored() -> None:
    gt = _gt([("loc_1", "q1", _LOC_REF)])
    preds = [
        {"case_id": "loc_1", "question": "q1",
         "prediction": "Sydney Bondi sunrise yoga Bondi Beach.", "task": "location"},
        {"case_id": "ghost", "question": "no-such-q",
         "prediction": "whatever", "task": "location"},
    ]
    out = score_predictions(preds, gt, default_task="location")
    assert out["total"] == 1  # only the matched one is scored
    assert out["correct"] == 1
    assert out["unmatched"] == 1


def test_empty_predictions_yields_zero_not_crash() -> None:
    out = score_predictions([], _gt([("c", "q", _LOC_REF)]), default_task="location")
    assert out["total"] == 0
    assert out["score_pct"] == 0.0
