"""The 0 to 3 score reader shared by the #1804 and #1795 harnesses (#1809)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.relevance_scale import InputError, parse_scores

_SRC = Path("labels.jsonl")


def _data(*rows: object) -> bytes:
    return "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")


def test_scores_and_null_are_read() -> None:
    data = _data(*({"id": f"r{s}", "score": s} for s in (0, 1, 2, 3)),
                 {"id": "rn", "score": None})
    assert parse_scores(_SRC, data) == {"r0": 0, "r1": 1, "r2": 2, "r3": 3, "rn": None}


@pytest.mark.parametrize("score", [2.0, 1e0, 0.0, 3.0, True, False])
def test_a_float_or_boolean_score_is_refused(score: object) -> None:
    # 2.0 == 2 and True == 1 in Python, so only the type tells them apart.
    with pytest.raises(InputError, match="1: need an id and a score of 0, 1, 2, 3, or null"):
        parse_scores(_SRC, _data({"id": "r1", "score": score}))


@pytest.mark.parametrize("correct", [True, False, None])
def test_a_binary_label_file_is_refused_by_name(correct: bool | None) -> None:
    with pytest.raises(InputError, match=r'labels\.jsonl:1: a binary "correct" label'):
        parse_scores(_SRC, _data({"id": "r1", "correct": correct}))


def test_a_row_with_a_score_and_a_binary_label_is_refused() -> None:
    data = _data({"id": "r1", "score": 2}, {"id": "r2", "score": 3, "correct": True})
    with pytest.raises(InputError, match=r'labels\.jsonl:2: a binary "correct" label'):
        parse_scores(_SRC, data)


def test_a_repeated_id_and_a_non_object_line_are_still_refused() -> None:
    with pytest.raises(InputError, match="2: repeats row r1"):
        parse_scores(_SRC, _data({"id": "r1", "score": 2}, {"id": "r1", "score": 0}))
    with pytest.raises(InputError, match="1: not a JSON object"):
        parse_scores(_SRC, _data([{"id": "r1", "score": 2}]))
