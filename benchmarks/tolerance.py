"""Tolerance-band classification for the v2.0 reproducibility harness.

Per the 2026-05-06 ratification on #437, bands are relative-with-floor:

- Relative band: ±X% of the canonical value, where X is per-metric.
  Defaults: F1 ±7%, exact-match ±10%, latency ±25%.
- Absolute floor: bands never fall below ±2 percentage points (prevents
  tiny-value flapping).
- Per-metric override: canonical JSON can declare wider bands for
  known-noisy metrics; overrides take precedence over defaults.
- Soft warning: drift inside the band but >50% of the band width
  emits a notice without failing.
- Direction (#1160): bands are enforced on the regression side only,
  per `METRIC_DIRECTIONS`. Leaving the band on the improving side
  WARNs instead of failing. Unclassified metrics stay two-sided.

Spec: docs/design/v2_reproducibility_harness.md.
Issue: #437, #1160.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Default per-metric relative-band percentages. Keys are matched as
# substrings of the leaf metric path so "f1_avg" matches "f1" and
# "median_latency_ms" matches "latency".
DEFAULT_RELATIVE_BANDS: dict[str, float] = {
    "exact_match": 0.10,
    "f1": 0.07,
    "latency": 0.25,
}
# Catch-all for any metric not covered above.
FALLBACK_RELATIVE_BAND = 0.10
# Absolute floor (in metric units, not percent — for 0..1 metrics this
# is 2 percentage points).
ABSOLUTE_FLOOR = 0.02


class Direction(str, Enum):
    """Which side of the band is a regression, per #1160.

    Two-sided bands treat any movement as a failure, so a real ranking
    win registers as FAIL — canonical `mab.Accurate_Retrieval.
    exact_match = 0.0` with the ±0.02 absolute floor means *every*
    improvement lands outside the band. One-sided bands enforce only
    the regression side.
    """

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    TWO_SIDED = "two_sided"


# Per-metric direction, keyed by the leaf metric name and falling back
# to the leaf's parent key (see `direction_for`).
#
# Unlisted metrics are TWO_SIDED on purpose. A metric given the *wrong*
# direction goes blind to regressions in its real direction, which is
# strictly worse than the false failure one-sided bands exist to fix,
# so the default has to be the conservative one: a new metric fails
# loudly as out-of-band until someone classifies it here. Do not
# replace this with substring matching on the metric name — an oddly
# named metric would then silently inherit the wrong direction.
METRIC_DIRECTIONS: dict[str, Direction] = {
    # --- quality: a drop is the regression -------------------------
    "accuracy": Direction.HIGHER_IS_BETTER,
    "accuracy_pct": Direction.HIGHER_IS_BETTER,
    "category_f1": Direction.HIGHER_IS_BETTER,  # parent of "1".."5"
    "correct": Direction.HIGHER_IS_BETTER,
    "exact_match": Direction.HIGHER_IS_BETTER,
    "exact_match_pct": Direction.HIGHER_IS_BETTER,
    "f1": Direction.HIGHER_IS_BETTER,
    "f1_pct": Direction.HIGHER_IS_BETTER,
    "items_with_any_nonzero_metric": Direction.HIGHER_IS_BETTER,
    "overall_f1": Direction.HIGHER_IS_BETTER,
    "perfect_cases": Direction.HIGHER_IS_BETTER,
    "score_pct": Direction.HIGHER_IS_BETTER,
    "substring_exact_match": Direction.HIGHER_IS_BETTER,
    "substring_exact_match_pct": Direction.HIGHER_IS_BETTER,
    "total_correct": Direction.HIGHER_IS_BETTER,
    # --- cost: a rise is the regression ----------------------------
    "avg_latency_ms": Direction.LOWER_IS_BETTER,
    "total_ingest_time_s": Direction.LOWER_IS_BETTER,
    # --- corpus invariants: any drift is a defect ------------------
    # These size the run rather than score it. A change means the
    # corpus or the dispatcher moved, which invalidates the comparison
    # in either direction.
    "count": Direction.TWO_SIDED,
    "domain_counts": Direction.TWO_SIDED,  # parent of the domain keys
    "n": Direction.TWO_SIDED,
    "n_runs": Direction.TWO_SIDED,
    "total": Direction.TWO_SIDED,
    "total_cases": Direction.TWO_SIDED,
    "total_episodes": Direction.TWO_SIDED,
    "total_ingest_turns": Direction.TWO_SIDED,
    "total_qa": Direction.TWO_SIDED,
    "total_queries": Direction.TWO_SIDED,
    "total_questions": Direction.TWO_SIDED,
    "type_counts": Direction.TWO_SIDED,  # parent of the type keys
    # --- deliberately ambiguous ------------------------------------
    # Retrieval volume is neither good nor bad on its own, and per
    # #1160 it is what inflates token-F1: halving the budget doubles
    # reported F1 while retrieving strictly less. Movement in either
    # direction is worth a look, so both sides stay enforced.
    "avg_beliefs": Direction.TWO_SIDED,
    "avg_beliefs_per_query": Direction.TWO_SIDED,
    # Hop-class summaries: `sem` may be a standard error (lower
    # better) or a mean score (higher better), and `split_delta_pp` is
    # a signed gap. Unresolved, so conservative.
    "multi_hop_mean_sem_pct": Direction.TWO_SIDED,
    "single_hop_mean_sem_pct": Direction.TWO_SIDED,
    "split_delta_pp": Direction.TWO_SIDED,
}


def direction_for(path: tuple[str, ...]) -> Direction:
    """Return the band direction for a leaf metric path.

    Looks up the leaf name first, then its parent. The fallback exists
    because bucketed metrics key their leaves by bucket id, not by
    metric name: LoCoMo's per-category F1 lands at
    ``...category_f1.1`` through ``.5``, so the leaf name is ``"1"``
    and only the parent says what is being measured. Leaf wins on a
    tie, so ``...count.correct`` resolves as ``correct`` (higher is
    better) while ``...temporal-reasoning.count`` resolves as the
    invariant ``count``.

    Anything unresolved is TWO_SIDED — see `METRIC_DIRECTIONS`.
    """
    if not path:
        return Direction.TWO_SIDED
    if path[-1] in METRIC_DIRECTIONS:
        return METRIC_DIRECTIONS[path[-1]]
    if len(path) > 1 and path[-2] in METRIC_DIRECTIONS:
        return METRIC_DIRECTIONS[path[-2]]
    return Direction.TWO_SIDED


class Verdict(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    # v2.1 #479. The observed sub-result has `_status:
    # skipped_data_missing`, meaning the adapter ran but its data dir
    # was absent. The canonical metrics for this leaf simply cannot be
    # computed; this is not a regression. summarize() treats SKIP as
    # neither PASS nor FAIL — it ignores the leaf for the rollup.
    SKIP = "skip"
    # #1160. Rollup-only: never the verdict of an individual leaf.
    # Ignoring each SKIP is right per #479, but ignoring *every* SKIP
    # meant a run where nothing could be computed rolled up to PASS and
    # the cron reported success having measured nothing. Distinct from
    # FAIL because the two demand different responses — a regression
    # means read the diff, no data means fix the runner.
    NO_DATA = "no_data"


@dataclass(frozen=True)
class BandCheck:
    """Result of comparing a single observed metric to its canonical band."""
    path: tuple[str, ...]   # e.g. ("mab", "Conflict_Resolution", "f1_avg")
    canonical: float
    observed: float
    lower: float
    upper: float
    band_kind: str          # "relative" | "absolute" | "override"
    verdict: Verdict
    note: str = ""
    # #1160. Which side of the band was enforced. Defaulted so the
    # SKIP/missing BandChecks, which never consult a direction, stay
    # constructible without one.
    direction: Direction = Direction.TWO_SIDED


def _relative_band_pct(metric_name: str, overrides: dict[str, float]) -> float:
    if metric_name in overrides:
        return overrides[metric_name]
    name_low = metric_name.lower()
    for key, pct in DEFAULT_RELATIVE_BANDS.items():
        if key in name_low:
            return pct
    return FALLBACK_RELATIVE_BAND


def compute_band(
    metric_name: str,
    canonical: float,
    *,
    overrides: dict[str, float] | None = None,
    floor: float = ABSOLUTE_FLOOR,
) -> tuple[float, float, str]:
    """Return (lower, upper, band_kind) for a canonical value.

    Relative band picked first; falls back to absolute floor when the
    relative band would be tighter than the floor.
    """
    overrides = overrides or {}
    pct = _relative_band_pct(metric_name, overrides)
    relative_half = abs(canonical) * pct
    if relative_half >= floor:
        return canonical - relative_half, canonical + relative_half, (
            "override" if metric_name in overrides else "relative"
        )
    return canonical - floor, canonical + floor, "absolute"


def classify(
    canonical: float, observed: float, lower: float, upper: float,
    *, direction: Direction = Direction.TWO_SIDED,
) -> tuple[Verdict, str]:
    """Map (observed) into pass/warn/fail given the band.

    `direction` (#1160) decides which side of the band is enforced.
    The band itself is unchanged — both bounds are still computed and
    reported — but leaving the band on the *improving* side is a WARN
    rather than a FAIL.

    It is not a silent PASS. This repo's own canonical numbers are the
    argument: token-F1 over a retrieval blob rises when the token
    budget falls, so a large unexplained gain is as likely to be a
    measurement artifact as a win. WARN keeps it visible without
    failing the nightly on a genuine improvement.

    Defaults to TWO_SIDED so existing positional callers are unchanged.
    """
    below, above = observed < lower, observed > upper
    if below or above:
        regressed = (
            (direction is not Direction.LOWER_IS_BETTER) if below
            else (direction is not Direction.HIGHER_IS_BETTER)
        )
        detail = (
            f"observed {observed:.4f} outside band "
            f"[{lower:.4f}, {upper:.4f}]"
        )
        if regressed:
            return Verdict.FAIL, detail
        return Verdict.WARN, (
            f"{detail} on the improving side ({direction.value}); "
            f"not a regression — confirm it is real before recutting"
        )
    half = (upper - lower) / 2.0
    if half == 0:
        return Verdict.PASS, "zero-width band; exact match required"
    drift = abs(observed - canonical) / half
    if drift > 0.5:
        return Verdict.WARN, (
            f"drift {drift:.0%} of band half-width "
            f"(observed {observed:.4f} vs canonical {canonical:.4f})"
        )
    return Verdict.PASS, ""


def _ancestor_skipped(obs_results: Any, path: tuple[str, ...]) -> bool:
    """Return True if any ancestor sub-result of `path` carries
    `_status: skipped_data_missing` in `obs_results`.

    Walks one prefix at a time. As soon as a level is missing or a
    non-dict shows up, the answer is "no skip ancestor here" — the
    enclosing logic falls through to the existing missing-leaf path.

    Per #479: when an adapter sub-result is skipped because data was
    absent, the canonical metrics nested under it are uncomputable
    (not regressions); they collapse to Verdict.SKIP rather than FAIL.
    """
    cursor: Any = obs_results
    for k in path:
        if not isinstance(cursor, dict):
            return False
        if cursor.get("_status") == "skipped_data_missing":
            return True
        cursor = cursor.get(k)
    if isinstance(cursor, dict) and cursor.get("_status") == "skipped_data_missing":
        return True
    return False


def _walk_leaves(
    obj: Any, path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], float]]:
    """Yield (path, value) for every numeric leaf in a nested dict.

    Skips non-numeric leaves (strings, lists, None) silently — those
    aren't metrics. Skips keys starting with `_` (reserved for
    metadata like `_status`, `_elapsed_sec`) EXCEPT the bare `_`
    sentinel: `benchmarks/run.py:241` uses `_` as the sub-bucket key
    for single-invocation adapters (locomo, longmemeval, amabench),
    so their `output.*` leaves live under
    `results.<adapter>._.output.*` and must be walked. Recursion
    re-filters at each level so metadata like `_status` nested
    inside `_` still gets skipped.
    """
    leaves: list[tuple[tuple[str, ...], float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("_") and k != "_":
                continue
            leaves.extend(_walk_leaves(v, (*path, str(k))))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        leaves.append((path, float(obj)))
    return leaves


def check_report(
    canonical: dict[str, Any],
    observed: dict[str, Any],
    *,
    metric_overrides: dict[str, float] | None = None,
    floor: float = ABSOLUTE_FLOOR,
) -> list[BandCheck]:
    """Walk the canonical results tree and band-check every leaf in observed.

    Missing leaves in `observed` are reported as FAIL ("not present").
    Extra leaves in `observed` not in `canonical` are silently ignored
    — the canonical JSON is the source of truth for which metrics
    matter.

    `metric_overrides` defaults to canonical["metric_overrides"] if the
    canonical JSON carries one. Explicitly-passed overrides take
    precedence (used by tests).
    """
    if metric_overrides is None:
        cano_overrides = canonical.get("metric_overrides")
        if isinstance(cano_overrides, dict):
            metric_overrides = {
                str(k): float(v) for k, v in cano_overrides.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
    cano_results = canonical.get("results", {})
    obs_results = observed.get("results", {})
    checks: list[BandCheck] = []
    for path, cano_val in _walk_leaves(cano_results):
        if _ancestor_skipped(obs_results, path):
            checks.append(BandCheck(
                path=path, canonical=cano_val, observed=float("nan"),
                lower=cano_val, upper=cano_val, band_kind="skipped",
                verdict=Verdict.SKIP,
                note=(
                    f"observed sub-result skipped (data missing) at "
                    f"{'/'.join(path)}"
                ),
            ))
            continue
        leaf = obs_results
        try:
            for k in path:
                leaf = leaf[k]
        except (KeyError, TypeError):
            checks.append(BandCheck(
                path=path, canonical=cano_val, observed=float("nan"),
                lower=cano_val, upper=cano_val, band_kind="missing",
                verdict=Verdict.FAIL,
                note=f"observed report has no leaf at {'/'.join(path)}",
            ))
            continue
        if not isinstance(leaf, (int, float)) or isinstance(leaf, bool):
            checks.append(BandCheck(
                path=path, canonical=cano_val, observed=float("nan"),
                lower=cano_val, upper=cano_val, band_kind="missing",
                verdict=Verdict.FAIL,
                note=f"observed leaf at {'/'.join(path)} is not numeric",
            ))
            continue
        obs_val = float(leaf)
        metric_name = path[-1]
        lower, upper, kind = compute_band(
            metric_name, cano_val,
            overrides=metric_overrides, floor=floor,
        )
        direction = direction_for(path)
        verdict, note = classify(
            cano_val, obs_val, lower, upper, direction=direction,
        )
        checks.append(BandCheck(
            path=path, canonical=cano_val, observed=obs_val,
            lower=lower, upper=upper, band_kind=kind,
            verdict=verdict, note=note, direction=direction,
        ))
    return checks


def summarize(checks: list[BandCheck]) -> tuple[Verdict, dict[str, int]]:
    """Roll up per-leaf verdicts to one overall verdict + counts.

    SKIP leaves (per #479) are tallied but do not raise the rollup
    above PASS — they represent uncomputable metrics, not regressions.

    But PASS is a claim that something was measured and stayed in band,
    so it requires at least one leaf that actually passed. Without that
    the rollup is NO_DATA (#1160): previously an all-SKIP run — every
    adapter exiting because its data dir was absent, e.g. a failed
    dataset download on the runner — returned PASS, and an empty check
    list did too, so the nightly reported success having measured
    nothing. #479 is preserved exactly: a SKIP alongside any real PASS
    still rolls up to PASS.
    """
    counts = {
        Verdict.PASS.value: 0, Verdict.WARN.value: 0,
        Verdict.FAIL.value: 0, Verdict.SKIP.value: 0,
    }
    for c in checks:
        # `.get` rather than direct indexing so a leaf carrying an
        # unexpected verdict is counted instead of raising KeyError
        # inside the gate that is supposed to report it.
        counts[c.verdict.value] = counts.get(c.verdict.value, 0) + 1
    if counts[Verdict.FAIL.value] > 0:
        return Verdict.FAIL, counts
    if counts[Verdict.WARN.value] > 0:
        return Verdict.WARN, counts
    if counts[Verdict.PASS.value] == 0:
        return Verdict.NO_DATA, counts
    return Verdict.PASS, counts


def load_report(path: Path) -> dict[str, Any]:
    """Read a harness report and validate schema_version=2."""
    with path.open() as f:
        data = json.load(f)
    if data.get("schema_version") != 2:
        raise ValueError(
            f"{path}: expected schema_version=2, got "
            f"{data.get('schema_version')!r}"
        )
    return data
