"""LLM-judge stage for the context-rebuilder eval harness.

Open-ended eval turns produce free-text continuations that substring
match (the deterministic path in #600's `score_fidelity`) misclassifies
as wrong. Those rows surface with `reason="needs_llm_judge"`. This
module turns each such row into a request file the host CLI dispatches
off-band, then folds the verdicts back into the run.

## Contamination protocol

`docs/concepts/BENCHMARKS.md` is explicit: generation and scoring run as separate
passes, and **the judge never sees the retrieval context.** The request
schema therefore carries only `(turn_idx, expected, actual)` — not the
rebuilt block, not the user turn. Including the rebuilt block in the
judge prompt would let the judge "patch" the prediction with details it
sees in the retrieval window, inflating fidelity.

## Cost posture

`max_judge_calls` defaults to 0, which disables the stage entirely so
CI runs are free. Operators opt in by passing a positive cap; the cap
binds before any request file is written, so even a misconfigured run
cannot exceed the budget.

The chosen judge tier is the host CLI's anchor model — the tier the
prior calibration work (Cohen's-κ disagreement against a zero-LLM
baseline) was measured against. A cheaper-tier knob is intentionally
not exposed: dropping to the small model below the anchor weakens
comparability on the open-ended turns this stage exists to score.

## Polymorphic dispatch (no SDK in this module)

This module imports nothing from any provider SDK. The flow is:

  1. Replay phase tags non-substring-match rows `needs_llm_judge`.
  2. `write_judge_requests(...)` writes `judge_requests.jsonl` into the
     per-run directory, one strict-schema row per eligible turn.
  3. Operator runs the host-side dispatch — a host-CLI skill, MCP host
     equivalent, or hand-driven `gh`/`uv run` call — which reads the
     requests, issues one off-band model call per row using
     `JUDGE_PROMPT_TEMPLATE`, and writes `judge_responses.jsonl`.
  4. `read_judge_responses(...)` joins the responses back by turn_idx.
  5. `apply_judge_verdicts(...)` updates the replay rows in place.

Steps 2/4/5 are pure-Python and exercised by CI. Step 3 is operator-
driven; CI never invokes a real model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Iterable

JUDGE_MODEL_TIER: Final[str] = "anchor"
"""Logical tier label for the judge model. The operator maps this to a
concrete model name in their host CLI's dispatch step; the value
"anchor" means the host's calibration-baseline tier (not the small /
cheaper tier below it). Encoding the tier as a label keeps this module
free of vendor-specific model identifiers and decoupled from any one
host CLI's catalog."""

JUDGE_REASON: Final[str] = "needs_llm_judge"
"""Reason tag carried by replay rows that this stage processes. Set by
the deterministic `score_fidelity` path when an `actual` is present
but no substring match was found."""

JUDGE_REQUESTS_FILENAME: Final[str] = "judge_requests.jsonl"
JUDGE_RESPONSES_FILENAME: Final[str] = "judge_responses.jsonl"

JUDGE_PROMPT_TEMPLATE: Final[str] = """\
You are scoring whether one free-text answer means the same thing as
a reference answer for a memory-system continuation eval. You see only
the two answers — no retrieval context, no user prompt, no history.

REFERENCE:
{expected}

CANDIDATE:
{actual}

## How to score

1. Identify the specific facts the reference commits to. A specific
   fact is a concrete atom — a name, path, value, action, or numeric
   quantity — that the reference is asserting. A reference may commit
   to one fact or several. Surrounding prose (restating the question,
   tooling tips, error explanations) is supporting material, not a
   fact for scoring purposes.
2. Check whether the candidate, read alone, conveys EVERY specific
   fact from the reference. Synonyms, paraphrases, equivalent
   restatements, and reordered phrasing all count as conveying the
   same fact.
3. `matched` is true iff a reader of the candidate alone — given no
   other context — could correctly state each specific fact the
   reference commits to.

## Worked examples

REFERENCE: "Use binary search; runs in O(log n)."
Specific facts: "binary search" AND "O(log n)".
- CANDIDATE "Binary search, which is O(log n)." → matched=true
- CANDIDATE "Use binary search." → matched=false (omits complexity)
- CANDIDATE "It's an O(log n) algorithm." → matched=false (omits name)

REFERENCE: "Yes."
Specific facts: none — the reference commits only to affirmation.
- CANDIDATE "Yes, that should work." → matched=true
- CANDIDATE "No." → matched=false (contradicts)
- CANDIDATE "" → matched=false (empty)

## Verdict

Reply with a single JSON object on one line:
  {{"matched": true|false, "rationale": "<= 240 chars"}}

Do not penalize style differences, missing supporting material, or
extra phrasing not present in the reference.
"""
"""Prompt the host-side dispatcher should send to each off-band call.
Anchors the judge in a no-retrieval-context posture (contamination
protocol) and constrains the response to a one-line JSON verdict so
`read_judge_responses` can parse without ambiguity."""


@dataclass(frozen=True)
class JudgeRequest:
    """One judge request. Strict schema — `turn_idx` is the join key,
    `expected` is the reference answer, `actual` is the candidate.
    No `rebuilt_block`, no `user_turn` — that is the contamination
    boundary."""
    turn_idx: int
    expected: str
    actual: str


@dataclass(frozen=True)
class JudgeResponse:
    """One judge response. `matched` is the verdict; `rationale` is a
    short free-text explanation surfaced in the run report."""
    turn_idx: int
    matched: bool
    rationale: str = ""


def _eligible_rows(replay_results: Iterable[dict]) -> list[dict]:
    """Filter the replay rows the judge stage processes.

    Eligibility: `reason == JUDGE_REASON` AND `actual` is a non-empty
    string AND `expected` is a non-empty string. Rows missing either
    field have nothing to compare; rows with the wrong reason were
    already settled by the deterministic path.
    """
    out: list[dict] = []
    for r in replay_results:
        if r.get("reason") != JUDGE_REASON:
            continue
        expected = r.get("expected")
        actual = r.get("actual")
        if not isinstance(expected, str) or not expected:
            continue
        if not isinstance(actual, str) or not actual:
            continue
        out.append(r)
    return out


def write_judge_requests(
    replay_results: Iterable[dict],
    run_dir: Path,
    *,
    max_judge_calls: int = 0,
) -> int:
    """Write eligible replay rows to `<run_dir>/judge_requests.jsonl`.

    Returns the number of rows written. `max_judge_calls` caps the
    request count and defaults to 0, which is the disabled state —
    no file is written when no calls are budgeted. Operators opt in
    by passing a positive cap.

    Each row's on-disk shape is exactly the three fields of
    `JudgeRequest`. Asserted by the test suite to prevent retrieval
    context from leaking into the judge prompt.
    """
    if max_judge_calls <= 0:
        return 0
    eligible = _eligible_rows(replay_results)
    if not eligible:
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / JUDGE_REQUESTS_FILENAME
    written = 0
    with path.open("w", encoding="utf-8") as f:
        for row in eligible:
            if written >= max_judge_calls:
                break
            req = JudgeRequest(
                turn_idx=int(row["turn_idx"]),
                expected=row["expected"],
                actual=row["actual"],
            )
            f.write(json.dumps(asdict(req), ensure_ascii=False) + "\n")
            written += 1
    return written


def read_judge_responses(run_dir: Path) -> dict[int, JudgeResponse]:
    """Read `<run_dir>/judge_responses.jsonl` into a turn_idx-keyed dict.

    Returns an empty dict if the file is absent (the operator hasn't
    run the dispatch step yet) or if every line is malformed. Lines
    that fail to parse, or that lack the required fields, are silently
    skipped — partial responses are preferable to a full run failure
    when the harness is iterating across many turns.
    """
    path = run_dir / JUDGE_RESPONSES_FILENAME
    if not path.exists():
        return {}
    out: dict[int, JudgeResponse] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            turn_idx = obj.get("turn_idx")
            matched = obj.get("matched")
            if not isinstance(turn_idx, int) or not isinstance(matched, bool):
                continue
            rationale = obj.get("rationale", "")
            if not isinstance(rationale, str):
                rationale = ""
            out[turn_idx] = JudgeResponse(
                turn_idx=turn_idx, matched=matched, rationale=rationale
            )
    return out


def apply_judge_verdicts(
    replay_results: list[dict],
    responses: dict[int, JudgeResponse],
) -> list[dict]:
    """Fold judge verdicts back into the replay rows.

    Pure function — returns a new list with judge-eligible rows
    updated. For each row tagged `JUDGE_REASON` with a matching
    response: `matched` becomes the judge verdict, `reason` is
    cleared (set to empty string), and a `judge_rationale` field is
    added. Rows without a response keep `reason=JUDGE_REASON` so the
    next harness invocation can retry.

    Non-eligible rows pass through untouched.
    """
    updated: list[dict] = []
    for row in replay_results:
        if row.get("reason") != JUDGE_REASON:
            updated.append(dict(row))
            continue
        turn_idx = row.get("turn_idx")
        if not isinstance(turn_idx, int) or turn_idx not in responses:
            updated.append(dict(row))
            continue
        verdict = responses[turn_idx]
        new_row = dict(row)
        new_row["matched"] = verdict.matched
        new_row["reason"] = ""
        new_row["judge_rationale"] = verdict.rationale
        updated.append(new_row)
    return updated
