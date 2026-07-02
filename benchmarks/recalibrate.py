"""Adapter scorer drift check — verifier recalibration auditor.

Runs each scored adapter's scorer function against a pinned oracle fixture
and asserts the computed score falls within the expected band.  A score
outside the band means the scorer has drifted from its known-correct
behaviour; any benchmark run produced by that scorer is suspect.

This guard is complementary to `verify_clean.py` (which checks retrieval
files for ground-truth contamination).  `recalibrate.py` checks that the
scorer *logic* still maps retrieval output to a score correctly.

Usage:
    uv run python -m benchmarks.recalibrate           # all adapters
    uv run python -m benchmarks.recalibrate locomo    # one adapter by name
    uv run python -m benchmarks.recalibrate --list    # list registered adapters

Exit codes:
    0 — all oracle checks passed
    1 — one or more oracle checks failed (scorer drift detected)
    2 — usage error

Design notes:
    - Oracle fixtures live under benchmarks/oracle_fixtures/ as JSON files.
    - Each fixture is a dict with an "oracles" list of input/band tuples.
    - The scorer is called directly (no live LLM, no retrieval, no I/O).
    - For adapters whose "score" is produced by a live LLM judge
      (LongMemEval, StructMemEval, AMA-Bench), the judge call itself cannot
      be oracle-checked deterministically — see docs/concepts/BENCHMARKS.md
      for the documented limitation.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_HERE: Path = Path(__file__).parent
_ORACLE_DIR: Path = _HERE / "oracle_fixtures"

# Registry: adapter_name → oracle fixture filename.
# Add an entry here and a matching JSON file in oracle_fixtures/ when
# adding oracle coverage for a new adapter's scorer.
ADAPTER_FIXTURES: dict[str, str] = {
    "locomo": "locomo_scorer.json",
    "mab": "mab_scorer.json",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleFail:
    """One oracle tuple that produced a score outside its expected band."""

    adapter: str
    label: str
    observed: float
    expected_lower: float
    expected_upper: float

    def __str__(self) -> str:
        return (
            f"  FAIL  [{self.adapter}] {self.label!r}\n"
            f"        observed {self.observed:.6f}  "
            f"expected [{self.expected_lower:.6f}, {self.expected_upper:.6f}]"
        )


# ---------------------------------------------------------------------------
# Band check primitive
# ---------------------------------------------------------------------------


def _check_oracle_tuple(
    *,
    adapter: str,
    label: str,
    observed: float,
    expected_lower: float,
    expected_upper: float,
) -> OracleFail | None:
    """Return an OracleFail if observed is outside [expected_lower, expected_upper],
    else None.
    """
    if observed < expected_lower or observed > expected_upper:
        return OracleFail(
            adapter=adapter,
            label=label,
            observed=observed,
            expected_lower=expected_lower,
            expected_upper=expected_upper,
        )
    return None


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def load_fixture(adapter_name: str) -> dict[str, Any]:
    """Load and parse the oracle fixture JSON for *adapter_name*."""
    filename = ADAPTER_FIXTURES.get(adapter_name)
    if filename is None:
        known = ", ".join(sorted(ADAPTER_FIXTURES))
        msg = f"Unknown adapter {adapter_name!r}. Known: {known}"
        raise KeyError(msg)
    path = _ORACLE_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Per-adapter scorer runners
# ---------------------------------------------------------------------------


def _run_locomo_oracle(
    oracles: list[dict[str, Any]],
    scorer_fn: Callable[..., float] | None = None,
) -> list[OracleFail]:
    """Run the LoCoMo oracle tuples against *scorer_fn*.

    *scorer_fn* defaults to ``benchmarks.locomo_adapter.score_qa``.
    Pass a replacement to test a broken scorer in unit tests.
    """
    if scorer_fn is None:
        from benchmarks.locomo_adapter import score_qa as scorer_fn  # type: ignore[assignment]

    fails: list[OracleFail] = []
    for oracle in oracles:
        label: str = oracle["label"]
        prediction: str = oracle["prediction"]
        ground_truth: str = oracle["ground_truth"]
        category: int = int(oracle["category"])
        expected_lower: float = float(oracle["expected_lower"])
        expected_upper: float = float(oracle["expected_upper"])

        observed: float = scorer_fn(prediction, ground_truth, category)
        result = _check_oracle_tuple(
            adapter="locomo",
            label=label,
            observed=observed,
            expected_lower=expected_lower,
            expected_upper=expected_upper,
        )
        if result is not None:
            fails.append(result)
    return fails


def _run_mab_oracle(
    oracles: list[dict[str, Any]],
    scorer_fn: Callable[..., Any] | None = None,
) -> list[OracleFail]:
    """Run the MAB/qa_scoring oracle tuples.

    *scorer_fn* is only consulted for the simple single-output scorers
    (score_exact_match, score_substring_exact_match, score_f1).  For
    score_multi_answer the function is always imported from qa_scoring
    (it returns a dict, not a float, so the injected scorer API would
    need special casing — pass a wrapped callable if you need to test it).
    """
    import benchmarks.qa_scoring as qs  # noqa: PLC0415

    _DEFAULT_SCORERS: dict[str, Callable[..., float]] = {
        "score_exact_match": qs.score_exact_match,
        "score_substring_exact_match": qs.score_substring_exact_match,
        "score_f1": qs.score_f1,
    }

    fails: list[OracleFail] = []
    for oracle in oracles:
        label: str = oracle["label"]
        fn_name: str = oracle["scorer_fn"]
        expected_lower: float = float(oracle["expected_lower"])
        expected_upper: float = float(oracle["expected_upper"])

        if fn_name == "score_multi_answer":
            prediction: str = oracle["prediction"]
            ground_truths: list[str] = oracle["ground_truths"]
            score_key: str = oracle["score_key"]
            scores: dict[str, float] = qs.score_multi_answer(prediction, ground_truths)
            observed: float = scores[score_key]
        else:
            # Simple single-output scorer — allow injection via scorer_fn
            if scorer_fn is not None:
                fn: Callable[..., float] = scorer_fn
            else:
                fn = _DEFAULT_SCORERS.get(fn_name, _DEFAULT_SCORERS["score_f1"])
            prediction = oracle["prediction"]
            ground_truth: str = oracle["ground_truth"]
            observed = fn(prediction, ground_truth)

        result = _check_oracle_tuple(
            adapter="mab",
            label=label,
            observed=observed,
            expected_lower=expected_lower,
            expected_upper=expected_upper,
        )
        if result is not None:
            fails.append(result)
    return fails


# ---------------------------------------------------------------------------
# Public API (used by tests)
# ---------------------------------------------------------------------------

_RUNNERS: dict[str, Callable[..., list[OracleFail]]] = {
    "locomo": _run_locomo_oracle,
    "mab": _run_mab_oracle,
}


def run_adapter_oracles(
    adapter_name: str,
    scorer_fn: Callable[..., Any] | None = None,
) -> list[OracleFail]:
    """Load and run the oracle fixture for *adapter_name*.

    *scorer_fn* is forwarded to the adapter-specific runner — useful in
    tests to substitute a broken scorer and verify the mechanism fires.

    Returns a (possibly empty) list of OracleFail objects.
    """
    fixture = load_fixture(adapter_name)
    runner = _RUNNERS[adapter_name]
    return runner(fixture["oracles"], scorer_fn)


def run_all_oracles() -> list[OracleFail]:
    """Run all registered adapters.  Returns the combined list of failures."""
    all_fails: list[OracleFail] = []
    for name in ADAPTER_FIXTURES:
        all_fails.extend(run_adapter_oracles(name))
    return all_fails


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_header() -> None:
    print("=" * 60)
    print("Adapter scorer recalibration check")
    print("=" * 60)


def main() -> None:
    """CLI entry point."""
    args = sys.argv[1:]

    if "--list" in args:
        print("Registered adapters:")
        for name, fname in sorted(ADAPTER_FIXTURES.items()):
            print(f"  {name:20s}  {_ORACLE_DIR / fname}")
        sys.exit(0)

    if not args:
        # Run all adapters
        targets = list(ADAPTER_FIXTURES)
    else:
        targets = args
        unknown = [t for t in targets if t not in ADAPTER_FIXTURES]
        if unknown:
            print(f"Unknown adapter(s): {', '.join(unknown)}")
            print(f"Known: {', '.join(sorted(ADAPTER_FIXTURES))}")
            sys.exit(2)

    _print_header()
    all_fails: list[OracleFail] = []
    for name in targets:
        print(f"\nChecking [{name}] ...")
        try:
            fails = run_adapter_oracles(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR loading adapter {name!r}: {exc}")
            all_fails.append(
                OracleFail(
                    adapter=name,
                    label="<load error>",
                    observed=float("nan"),
                    expected_lower=float("nan"),
                    expected_upper=float("nan"),
                )
            )
            continue

        if fails:
            for f in fails:
                print(str(f))
            all_fails.extend(fails)
        else:
            fixture = load_fixture(name)
            n = len(fixture.get("oracles", []))
            print(f"  PASS  {n} oracle tuples checked")

    print()
    print("=" * 60)
    if all_fails:
        print(f"DRIFT DETECTED: {len(all_fails)} oracle tuple(s) failed.")
        print("Scorer logic has changed relative to the pinned oracle.")
        print("Fix the scorer or update the oracle fixture (with review).")
        sys.exit(1)
    else:
        total = sum(
            len(load_fixture(n).get("oracles", [])) for n in targets
        )
        print(f"ALL PASS: {total} oracle tuples checked across {len(targets)} adapter(s).")
        sys.exit(0)


if __name__ == "__main__":
    main()
