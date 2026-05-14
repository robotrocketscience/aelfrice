"""Dynamic-mode trigger investigation for v1.4/v1.5 (issues #141, #188).

Measures two heuristic-driven trigger candidates against a fixture or
directory of fixtures. The output is **the evidence** for the v1.4/v1.5
ship-or-park decision.

Spec gate: "Dynamic mode ships only if its fidelity delta beats
the threshold default by a documented margin (≥ 5% absolute
fidelity at same-or-lower token cost). Otherwise: park to v1.5."

The two candidates:

  1. **rate-of-growth.** Fire at the first turn whose 4-turn
     rolling token-count delta exceeds the median per-turn token
     count of the fixture. Rationale: catches moments of rapid
     working-state expansion -- the failure mode the spec calls
     out as "agent stops being able to compress its own state".

  2. **entity-density-delta.** Fire at the first turn whose new-
     entity count drops below half the fixture-wide median.
     Rationale: the rebuilder's job is easier once the agent has
     stopped introducing new state -- low new-entity density is
     a signal that working state has stabilised and a rebuild
     captures the steady-state knowledge cleanly.

Both candidates use the same content-overlap proxy and seed-store
machinery as `calibrate.py`. The output JSON has the same shape so
downstream tooling can compare apples-to-apples.

If a future v1.5.x investigation finds a candidate that clears the
gate, this module is the place to extend; the threshold-mode
calibration acts as the baseline-to-beat.

CLI modes
---------
Single-fixture mode (positional ``fixture`` arg)
    Run against one ``turns.jsonl`` file; emit a single
    ``DynamicProbeResult`` JSON object.

Corpus mode (``--corpus DIR``)
    Run against every ``turns.jsonl`` found directly under ``DIR``
    (non-recursive). Emit a JSON object whose top-level keys are
    ``fixture_results`` (list of per-fixture ``DynamicProbeResult``
    dicts), ``corpus`` (the ``DIR`` path or ``<lab-side>`` when
    scrubbed), ``n_fixtures``, and ``aggregate_verdict`` (``"ship"``
    if any fixture yields a winning candidate, ``"park"`` otherwise).
    The corpus path is intentionally kept separate from individual
    fixture paths so callers can scrub it to ``"<lab-side>"`` without
    touching per-result structure.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final

from aelfrice.context_rebuilder import RecentTurn, rebuild_v14
from aelfrice.entity_extractor import extract_entities
from aelfrice.store import MemoryStore
from benchmarks.context_rebuilder.calibrate import (
    CALIBRATION_TOKEN_BUDGET,
    content_overlap_score,
    seed_store_with_pre_clear_turns,
)
from benchmarks.context_rebuilder.measure import estimate_tokens
from benchmarks.context_rebuilder.replay import ReplayTurn, load_turns

#: Spec ship-gate margin for dynamic mode. Dynamic-mode fidelity must
#: exceed threshold-mode fidelity by at least this absolute amount AT
#: SAME-OR-LOWER token cost to ship. Sourced from issue #141 spec.
DYNAMIC_FIDELITY_MARGIN: Final[float] = 0.05

#: Threshold-mode reference point for the comparison. Set from the
#: v1.4 calibration; the dynamic_probe asserts the comparison is
#: against the same number that drove the threshold default.
THRESHOLD_REFERENCE_FRACTION: Final[float] = 0.6


@dataclass(frozen=True)
class CandidateMeasurement:
    name: str
    clear_at: int
    rebuild_block_tokens: int
    full_replay_baseline_tokens: int
    token_budget_ratio: float
    continuation_fidelity: float
    n_post_clear_assistant_turns: int
    fire_rule: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "clear_at": self.clear_at,
            "rebuild_block_tokens": self.rebuild_block_tokens,
            "full_replay_baseline_tokens": (
                self.full_replay_baseline_tokens
            ),
            "token_budget_ratio": self.token_budget_ratio,
            "continuation_fidelity": self.continuation_fidelity,
            "n_post_clear_assistant_turns": (
                self.n_post_clear_assistant_turns
            ),
            "fire_rule": self.fire_rule,
        }


@dataclass(frozen=True)
class DynamicProbeResult:
    fixture: str
    n_turns: int
    threshold_reference: dict[str, object]
    candidates: list[CandidateMeasurement]
    verdict: str
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "fixture": self.fixture,
            "n_turns": self.n_turns,
            "threshold_reference": self.threshold_reference,
            "candidates": [c.to_dict() for c in self.candidates],
            "verdict": self.verdict,
            "rationale": self.rationale,
        }


def _rate_of_growth_clear_at(turns: list[ReplayTurn]) -> int:
    """Fire at the first turn whose 4-turn rolling token-delta exceeds
    the fixture-wide median per-turn token count.

    Returns a `clear_at` index in `[1, n_turns - 1]`. If the rule
    never fires, returns `n_turns - 1` (degenerate fall-back).
    """
    per_turn = [estimate_tokens(t.text) for t in turns]
    if not per_turn:
        return 0
    median = statistics.median(per_turn)
    window = 4
    for i in range(window, len(turns)):
        delta = sum(per_turn[i - window:i])
        # Average per-turn delta over the window vs. the per-turn
        # median; fire when the window's average exceeds the median
        # by at least the median (i.e. the window is double-fast).
        if delta / float(window) > median * 1.5:
            return i
    return max(1, len(turns) - 1)


def _entity_density_delta_clear_at(turns: list[ReplayTurn]) -> int:
    """Fire at the first turn whose new-entity count drops below half
    the fixture-wide median new-entity count.

    Returns a `clear_at` index. If the rule never fires, returns
    `n_turns - 1`.
    """
    seen: set[str] = set()
    new_per_turn: list[int] = []
    for t in turns:
        ents = extract_entities(t.text, max_entities=32)
        new = 0
        for e in ents:
            key = e.lower
            if key not in seen:
                seen.add(key)
                new += 1
        new_per_turn.append(new)
    if not new_per_turn:
        return 0
    median = statistics.median(new_per_turn)
    floor = max(1.0, median * 0.5)
    # Skip the first 4 turns (every turn there is "all new") so we're
    # measuring the steady-state phase, not the cold-start phase.
    for i in range(4, len(new_per_turn)):
        if new_per_turn[i] < floor:
            return i
    return max(1, len(turns) - 1)


def _measure_candidate(
    *, name: str, fixture_turns: list[ReplayTurn], clear_at: int,
    full_replay_tokens: int, fire_rule: str,
) -> CandidateMeasurement:
    """Measure one candidate trigger at its computed `clear_at`."""
    n = len(fixture_turns)
    if clear_at >= n:
        clear_at = n - 1
    if clear_at < 1:
        clear_at = 1

    store = MemoryStore(":memory:")
    try:
        seed_store_with_pre_clear_turns(store, fixture_turns, clear_at)
        recent = [
            RecentTurn(
                role=t.role, text=t.text, session_id=t.session_id,
            )
            for t in fixture_turns
            if t.index < clear_at
        ]
        block = rebuild_v14(
            recent, store, token_budget=CALIBRATION_TOKEN_BUDGET,
        )
    finally:
        store.close()
    block_tokens = estimate_tokens(block)
    fidelity, n_scoreable = content_overlap_score(
        block, fixture_turns, clear_at,
    )
    ratio = (
        block_tokens / float(full_replay_tokens)
        if full_replay_tokens > 0 else 0.0
    )
    return CandidateMeasurement(
        name=name,
        clear_at=clear_at,
        rebuild_block_tokens=block_tokens,
        full_replay_baseline_tokens=full_replay_tokens,
        token_budget_ratio=round(ratio, 6),
        continuation_fidelity=round(fidelity, 6),
        n_post_clear_assistant_turns=n_scoreable,
        fire_rule=fire_rule,
    )


def probe(fixture: Path) -> DynamicProbeResult:
    """Run the dynamic-mode investigation against `fixture`."""
    turns, _ = load_turns(fixture)
    full_replay_tokens = sum(estimate_tokens(t.text) for t in turns)

    # Reference point: re-derive the threshold-mode measurement at
    # the calibrated default. Lock the comparison to the same number
    # that drove the threshold default so the verdict is honest.
    threshold_clear_at = int(len(turns) * THRESHOLD_REFERENCE_FRACTION)
    if threshold_clear_at < 1:
        threshold_clear_at = 1
    threshold_meas = _measure_candidate(
        name=f"threshold_{THRESHOLD_REFERENCE_FRACTION}",
        fixture_turns=turns,
        clear_at=threshold_clear_at,
        full_replay_tokens=full_replay_tokens,
        fire_rule=(
            f"clear_at = floor(n_turns * {THRESHOLD_REFERENCE_FRACTION})"
        ),
    )

    # Candidate 1: rate-of-growth.
    rog_clear = _rate_of_growth_clear_at(turns)
    rog = _measure_candidate(
        name="rate_of_growth",
        fixture_turns=turns,
        clear_at=rog_clear,
        full_replay_tokens=full_replay_tokens,
        fire_rule=(
            "fire at first turn whose 4-turn rolling per-turn token "
            "average exceeds 1.5x the fixture-wide median per-turn "
            "token count"
        ),
    )

    # Candidate 2: entity-density-delta.
    edd_clear = _entity_density_delta_clear_at(turns)
    edd = _measure_candidate(
        name="entity_density_delta",
        fixture_turns=turns,
        clear_at=edd_clear,
        full_replay_tokens=full_replay_tokens,
        fire_rule=(
            "fire at first turn (after warmup of 4 turns) whose new-"
            "entity count drops below max(1, 0.5 * fixture-median "
            "new-entity count)"
        ),
    )

    # Verdict.
    threshold_fid = threshold_meas.continuation_fidelity
    threshold_ratio = threshold_meas.token_budget_ratio
    winners: list[str] = []
    for cand in (rog, edd):
        fid_delta = cand.continuation_fidelity - threshold_fid
        cost_ok = cand.token_budget_ratio <= threshold_ratio
        if fid_delta >= DYNAMIC_FIDELITY_MARGIN and cost_ok:
            winners.append(cand.name)
    if winners:
        verdict = "ship"
        rationale = (
            f"candidates {winners} clear the v1.4 ship-gate "
            f"(>= {DYNAMIC_FIDELITY_MARGIN} absolute fidelity over "
            f"threshold at same-or-lower token cost)."
        )
    else:
        verdict = "park"
        deltas = ", ".join(
            f"{c.name} fid_delta="
            f"{(c.continuation_fidelity - threshold_fid):+.4f}, "
            f"ratio={c.token_budget_ratio:.3f} vs "
            f"threshold {threshold_ratio:.3f}"
            for c in (rog, edd)
        )
        rationale = (
            f"no candidate clears the v1.4 ship-gate "
            f"(>= {DYNAMIC_FIDELITY_MARGIN} absolute fidelity at "
            f"same-or-lower token cost). Measurements: {deltas}. "
            f"Parked for v1.5; see docs/design/context_rebuilder.md "
            f"§ Dynamic mode (parked v1.5)."
        )

    return DynamicProbeResult(
        fixture=str(fixture),
        n_turns=len(turns),
        threshold_reference=threshold_meas.to_dict(),
        candidates=[rog, edd],
        verdict=verdict,
        rationale=rationale,
    )


def probe_corpus(
    corpus_dir: Path,
    *,
    corpus_label: str | None = None,
) -> dict[str, object]:
    """Run the probe against every ``turns.jsonl`` in *corpus_dir*.

    Returns a JSON-serialisable dict with keys:
    - ``corpus`` — the corpus path (use ``corpus_label`` to override,
      e.g. to scrub to ``"<lab-side>"`` before committing).
    - ``n_fixtures`` — number of fixtures found and run.
    - ``fixture_results`` — list of ``DynamicProbeResult.to_dict()``
      for each fixture.
    - ``aggregate_verdict`` — ``"ship"`` if any fixture produced a
      winning candidate; ``"park"`` otherwise.
    - ``aggregate_rationale`` — human-readable summary.
    """
    fixtures = sorted(corpus_dir.glob("*.jsonl"))
    results: list[DynamicProbeResult] = []
    for f in fixtures:
        results.append(probe(f))
    label = corpus_label if corpus_label is not None else str(corpus_dir)
    ship_count = sum(1 for r in results if r.verdict == "ship")
    if results and ship_count > 0:
        agg_verdict = "ship"
        agg_rationale = (
            f"{ship_count}/{len(results)} fixture(s) produced a candidate "
            f"clearing the ship gate."
        )
    elif results:
        agg_verdict = "park"
        agg_rationale = (
            f"0/{len(results)} fixture(s) produced a candidate clearing "
            f"the ship gate."
        )
    else:
        agg_verdict = "park"
        agg_rationale = "no fixtures found in corpus directory."
    return {
        "corpus": label,
        "n_fixtures": len(results),
        "fixture_results": [r.to_dict() for r in results],
        "aggregate_verdict": agg_verdict,
        "aggregate_rationale": agg_rationale,
    }


# --- CLI ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.context_rebuilder.dynamic_probe",
        description=(
            "Dynamic-mode trigger investigation for v1.4/v1.5 (#141, #188). "
            "Measures two heuristic candidates and emits a JSON verdict. "
            "Pass a single ``fixture`` positional arg or ``--corpus DIR`` "
            "to run against all *.jsonl files in a directory."
        ),
    )
    # Mutually exclusive: single fixture OR corpus directory.
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "fixture",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Path to a single turns.jsonl fixture "
            "(mutually exclusive with --corpus)."
        ),
    )
    group.add_argument(
        "--corpus",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory containing *.jsonl fixtures. "
            "Run against all fixtures found; emit aggregate JSON. "
            "Mutually exclusive with the positional fixture arg."
        ),
    )
    p.add_argument(
        "--corpus-label",
        default=None,
        metavar="LABEL",
        help=(
            "Override the corpus path in output JSON (e.g. '<lab-side>'). "
            "Only meaningful with --corpus."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write probe JSON to PATH. Default: stdout.",
    )
    return p


def main(
    argv: list[str] | None = None,
    *,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    out_stream = stdout if stdout is not None else sys.stdout
    _ = stderr  # reserved for future error reporting
    parser = _build_parser()
    args = parser.parse_args(argv)
    out_path: Path | None = args.out  # type: ignore[assignment]

    if args.corpus is not None:
        corpus_dir: Path = args.corpus  # type: ignore[assignment]
        corpus_label: str | None = args.corpus_label
        result_obj = probe_corpus(corpus_dir, corpus_label=corpus_label)
        payload = json.dumps(result_obj, indent=2, sort_keys=True)
    else:
        fixture: Path = args.fixture  # type: ignore[assignment]
        result = probe(fixture)
        payload = json.dumps(result.to_dict(), indent=2, sort_keys=True)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {out_path}", file=out_stream)
    else:
        print(payload, file=out_stream)
    return 0


if __name__ == "__main__":
    sys.exit(main())
