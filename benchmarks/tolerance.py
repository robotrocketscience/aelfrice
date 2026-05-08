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

Spec: docs/v2_reproducibility_harness.md.
Issue: #437.
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
) -> tuple[Verdict, str]:
    """Map (observed) into pass/warn/fail given the band."""
    if observed < lower or observed > upper:
        return Verdict.FAIL, (
            f"observed {observed:.4f} outside band "
            f"[{lower:.4f}, {upper:.4f}]"
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
    metadata like `_status`, `_elapsed_sec`).
    """
    leaves: list[tuple[tuple[str, ...], float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("_"):
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
        verdict, note = classify(cano_val, obs_val, lower, upper)
        checks.append(BandCheck(
            path=path, canonical=cano_val, observed=obs_val,
            lower=lower, upper=upper, band_kind=kind,
            verdict=verdict, note=note,
        ))
    return checks


def summarize(checks: list[BandCheck]) -> tuple[Verdict, dict[str, int]]:
    """Roll up per-leaf verdicts to one overall verdict + counts.

    SKIP leaves (per #479) are tallied but do not raise the rollup
    above PASS — they represent uncomputable metrics, not regressions.
    """
    counts = {
        Verdict.PASS.value: 0, Verdict.WARN.value: 0,
        Verdict.FAIL.value: 0, Verdict.SKIP.value: 0,
    }
    for c in checks:
        counts[c.verdict.value] += 1
    if counts[Verdict.FAIL.value] > 0:
        return Verdict.FAIL, counts
    if counts[Verdict.WARN.value] > 0:
        return Verdict.WARN, counts
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
