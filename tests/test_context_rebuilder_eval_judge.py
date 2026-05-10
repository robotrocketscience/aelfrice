"""Tests for the LLM-judge stage of the context-rebuilder eval harness.

Round-trip only. CI never invokes a real model — the judge response
file is hand-authored in each test fixture, exactly as it would be
hand-authored by an operator running the host-side dispatch.

Two contracts are load-bearing and asserted explicitly:

  1. The contamination boundary (`docs/BENCHMARKS.md`): the judge
     request file carries only `(turn_idx, expected, actual)` — no
     rebuilt block, no user turn.

  2. The cost cap: `max_judge_calls=0` (the default) writes no file
     and dispatches no calls, so CI runs are free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The benchmarks/context-rebuilder directory is hyphenated, so `python -m`
# imports do not work. Resolve the module via path hop, the same shim the
# eval harness uses.
_HARNESS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "context-rebuilder"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

from judges import llm_judge  # noqa: E402


def _replay_row(
    *, turn_idx: int, reason: str, expected: str, actual: str,
) -> dict:
    return {
        "turn_idx": turn_idx,
        "expected": expected,
        "actual": actual,
        "matched": False,
        "reason": reason,
    }


# --- 1. contamination boundary -------------------------------------------

def test_judge_request_schema_carries_only_turn_idx_expected_actual(
    tmp_path: Path,
) -> None:
    """The request file must not surface retrieval context to the judge.

    `docs/BENCHMARKS.md` requires that generation and scoring run as
    separate passes and that the judge never sees the rebuilt context.
    This test fails loud if a future edit widens the schema to include
    `rebuilt_block`, `user_turn`, or any other retrieval-pass field.
    """
    rows = [
        _replay_row(
            turn_idx=4, reason=llm_judge.JUDGE_REASON,
            expected="cached locks are demoted after 30 days idle",
            actual="locks are evicted by the idle-30d sweep",
        ),
    ]
    n = llm_judge.write_judge_requests(rows, tmp_path, max_judge_calls=5)
    assert n == 1
    path = tmp_path / llm_judge.JUDGE_REQUESTS_FILENAME
    assert path.exists()
    line = path.read_text(encoding="utf-8").splitlines()[0]
    obj = json.loads(line)
    assert set(obj.keys()) == {"turn_idx", "expected", "actual"}, (
        f"judge request schema widened to {sorted(obj.keys())} — would leak "
        "retrieval context into the judge prompt"
    )


def test_judge_request_writer_skips_non_judge_rows(tmp_path: Path) -> None:
    """Rows tagged with other reasons (substring-match settled,
    pending_replay, needs_replay_client) must not appear in the judge
    request file. Only rows the deterministic path could not settle
    are the judge's concern."""
    rows = [
        _replay_row(
            turn_idx=1, reason="",  # already settled by substring match
            expected="x", actual="x",
        ),
        _replay_row(
            turn_idx=2, reason="pending_replay",
            expected="y", actual="",
        ),
        _replay_row(
            turn_idx=3, reason=llm_judge.JUDGE_REASON,
            expected="z", actual="paraphrased z",
        ),
    ]
    n = llm_judge.write_judge_requests(rows, tmp_path, max_judge_calls=5)
    assert n == 1
    payload = (tmp_path / llm_judge.JUDGE_REQUESTS_FILENAME).read_text(
        encoding="utf-8"
    )
    objs = [json.loads(line) for line in payload.splitlines() if line.strip()]
    assert [o["turn_idx"] for o in objs] == [3]


def test_judge_request_writer_skips_empty_expected_or_actual(
    tmp_path: Path,
) -> None:
    """A judge-tagged row with no `expected` text or no `actual` text
    has nothing to compare. Including it would waste a call and
    confuse the verdict."""
    rows = [
        _replay_row(
            turn_idx=1, reason=llm_judge.JUDGE_REASON,
            expected="", actual="something",
        ),
        _replay_row(
            turn_idx=2, reason=llm_judge.JUDGE_REASON,
            expected="something", actual="",
        ),
    ]
    n = llm_judge.write_judge_requests(rows, tmp_path, max_judge_calls=5)
    assert n == 0
    assert not (tmp_path / llm_judge.JUDGE_REQUESTS_FILENAME).exists()


# --- 2. cost cap ---------------------------------------------------------

def test_judge_disabled_by_default_writes_no_file(tmp_path: Path) -> None:
    """With `max_judge_calls=0` (the default), no file is written and
    no requests are produced even when eligible rows are present.
    This is what keeps CI free."""
    rows = [
        _replay_row(
            turn_idx=i, reason=llm_judge.JUDGE_REASON,
            expected=f"ref-{i}", actual=f"cand-{i}",
        )
        for i in range(3)
    ]
    n = llm_judge.write_judge_requests(rows, tmp_path)
    assert n == 0
    assert not (tmp_path / llm_judge.JUDGE_REQUESTS_FILENAME).exists()


def test_judge_cost_cap_binds_before_writing(tmp_path: Path) -> None:
    """`max_judge_calls` is the hard ceiling on requests written, not
    just an advisory. With 5 eligible rows and cap=2, exactly 2 rows
    land in the file. The cap binds before write to protect cost
    even if eligibility logic over-counts."""
    rows = [
        _replay_row(
            turn_idx=i, reason=llm_judge.JUDGE_REASON,
            expected=f"ref-{i}", actual=f"cand-{i}",
        )
        for i in range(5)
    ]
    n = llm_judge.write_judge_requests(rows, tmp_path, max_judge_calls=2)
    assert n == 2
    payload = (tmp_path / llm_judge.JUDGE_REQUESTS_FILENAME).read_text(
        encoding="utf-8"
    )
    assert len(payload.splitlines()) == 2


# --- 3. response round-trip ---------------------------------------------

def test_judge_response_reader_joins_by_turn_idx(tmp_path: Path) -> None:
    """Pre-write a hand-authored response file (as an operator would
    after running the host-side dispatch), read it back, assert each
    response is keyed by turn_idx and carries the verdict."""
    path = tmp_path / llm_judge.JUDGE_RESPONSES_FILENAME
    path.write_text(
        "\n".join(
            [
                json.dumps({"turn_idx": 3, "matched": True, "rationale": "paraphrase"}),
                json.dumps({"turn_idx": 7, "matched": False, "rationale": "omits load-bearing fact"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    responses = llm_judge.read_judge_responses(tmp_path)
    assert set(responses.keys()) == {3, 7}
    assert responses[3].matched is True
    assert responses[3].rationale == "paraphrase"
    assert responses[7].matched is False


def test_judge_response_reader_returns_empty_when_file_absent(
    tmp_path: Path,
) -> None:
    """No response file → empty dict, not an exception. The harness
    runs the read-step optimistically; the operator may not have run
    the dispatch yet."""
    assert llm_judge.read_judge_responses(tmp_path) == {}


def test_judge_response_reader_skips_malformed_lines(tmp_path: Path) -> None:
    """A partially-corrupted response file still returns the well-formed
    rows. Better partial coverage than full failure when iterating on
    many turns."""
    path = tmp_path / llm_judge.JUDGE_RESPONSES_FILENAME
    path.write_text(
        "\n".join(
            [
                "not-json",
                json.dumps({"turn_idx": "wrong-type", "matched": True}),
                json.dumps({"turn_idx": 5, "matched": True, "rationale": "ok"}),
                json.dumps({"turn_idx": 6}),  # missing matched
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    responses = llm_judge.read_judge_responses(tmp_path)
    assert set(responses.keys()) == {5}


# --- 4. fold verdicts into replay rows ----------------------------------

def test_apply_judge_verdicts_clears_reason_and_sets_matched() -> None:
    """For each judge-tagged row with a matching response: `matched`
    becomes the verdict, `reason` clears, and `judge_rationale` is
    added. Non-judge rows pass through untouched. Rows without a
    response keep their needs_llm_judge tag for the next retry."""
    rows = [
        _replay_row(
            turn_idx=1, reason="",
            expected="x", actual="x",
        ),
        _replay_row(
            turn_idx=2, reason=llm_judge.JUDGE_REASON,
            expected="ref-2", actual="cand-2",
        ),
        _replay_row(
            turn_idx=3, reason=llm_judge.JUDGE_REASON,
            expected="ref-3", actual="cand-3",
        ),
    ]
    responses = {
        2: llm_judge.JudgeResponse(turn_idx=2, matched=True, rationale="ok"),
        # 3 has no response — operator hasn't run dispatch for it yet.
    }
    out = llm_judge.apply_judge_verdicts(rows, responses)
    assert out[0]["matched"] is False  # unchanged (non-judge row)
    assert out[0]["reason"] == ""
    assert out[1]["matched"] is True
    assert out[1]["reason"] == ""
    assert out[1]["judge_rationale"] == "ok"
    assert out[2]["matched"] is False  # unchanged (no response yet)
    assert out[2]["reason"] == llm_judge.JUDGE_REASON


def test_apply_judge_verdicts_does_not_mutate_input() -> None:
    """The function is pure — mutating callers would couple it to
    whatever upstream owns the replay-row list."""
    rows = [
        _replay_row(
            turn_idx=1, reason=llm_judge.JUDGE_REASON,
            expected="x", actual="y",
        ),
    ]
    snapshot = json.dumps(rows, sort_keys=True)
    llm_judge.apply_judge_verdicts(
        rows,
        {1: llm_judge.JudgeResponse(turn_idx=1, matched=True)},
    )
    assert json.dumps(rows, sort_keys=True) == snapshot


# --- 5. round-trip end-to-end -------------------------------------------

def test_end_to_end_round_trip(tmp_path: Path) -> None:
    """Eligible rows → write → operator writes responses (here,
    hand-authored) → read → apply → assert verdicts folded in."""
    rows = [
        _replay_row(
            turn_idx=1, reason=llm_judge.JUDGE_REASON,
            expected="lock posterior is Beta-Bernoulli",
            actual="locks use a Beta-Bernoulli posterior",
        ),
        _replay_row(
            turn_idx=2, reason=llm_judge.JUDGE_REASON,
            expected="locks survive demotion sweeps",
            actual="the demotion sweep ignores locks",
        ),
    ]
    n = llm_judge.write_judge_requests(rows, tmp_path, max_judge_calls=10)
    assert n == 2

    # Operator dispatches host-side and writes the response file.
    (tmp_path / llm_judge.JUDGE_RESPONSES_FILENAME).write_text(
        "\n".join(
            [
                json.dumps({"turn_idx": 1, "matched": True, "rationale": "paraphrase"}),
                json.dumps({"turn_idx": 2, "matched": True, "rationale": "equivalent"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    responses = llm_judge.read_judge_responses(tmp_path)
    folded = llm_judge.apply_judge_verdicts(rows, responses)
    assert all(r["matched"] for r in folded)
    assert all(r["reason"] == "" for r in folded)


# --- 6. anchor constants --------------------------------------------------

def test_judge_model_tier_is_anchor() -> None:
    """The judge tier label is "anchor" — the operator maps this to
    the host CLI's calibration-baseline tier, not the cheaper small-
    model tier below it. Prior calibration work (Cohen's-κ against a
    zero-LLM baseline) was measured at the anchor tier; dropping to
    a cheaper tier weakens comparability. This is a flag-flip canary;
    intentional changes should update the constant *and* this test
    together."""
    assert llm_judge.JUDGE_MODEL_TIER == "anchor"


@pytest.mark.parametrize(
    "field",
    ["expected", "actual"],
)
def test_judge_prompt_template_contains_required_field(field: str) -> None:
    """The prompt template must surface the expected and actual fields
    so the host-side dispatcher can render concrete prompts."""
    assert "{" + field + "}" in llm_judge.JUDGE_PROMPT_TEMPLATE
