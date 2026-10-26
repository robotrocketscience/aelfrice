"""Two-grader precision scoring shared by the graded gates (#1677, #1650).

A graded gate samples what a detector or matcher would do, has two
graders label each item independently, and scores the labels here. A
label is `true` when the item is right, `false` when it is wrong, and
`null` when the grader can't tell. An unclear label is left out of that
grader's precision and out of Cohen's kappa, which is computed on the
items both graders labeled `true` or `false`.

The verdict is `insufficient` until each grader has graded at least
`MIN_GRADED` items (operator ruling, 2026-10-08), `pass` when each
grader's precision reaches `BAR`, and `fail` otherwise.

Label files are JSONL, one `{"id": str, "correct": true|false|null}`
object per line. A line that is not such an object, or that repeats an
id, is refused.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from benchmarks.context_rebuilder.kappa import cohens_kappa

MIN_GRADED: Final[int] = 30
"""Items each grader must grade before the verdict can pass (2026-10-08)."""
BAR: Final[float] = 0.70
"""Pre-registered precision bar, on each grader (#1647)."""
Z_95: Final[float] = 1.959963984540054

Labels = Mapping[str, bool | None]


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval `(lower, upper)`; `(0.0, 0.0)` when n is 0."""
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom, (centre + margin) / denom


def wilson_lower(successes: int, n: int, z: float = Z_95) -> float:
    """Wilson score interval lower bound; 0.0 when n is 0."""
    return wilson_interval(successes, n, z)[0]


def read_labels(path: Path) -> dict[str, bool | None]:
    """{item id: True | False | None} from a JSONL file of {"id", "correct"}.

    ValueError on a line that is not a JSON object or repeats an id.
    """
    labels: dict[str, bool | None] = {}
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj: object = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{n}: not a JSON object")
            row = cast(dict[str, Any], obj)
            correct = row.get("correct")
            if not isinstance(row.get("id"), str) or correct not in (True, False, None):
                raise ValueError(f"{path}:{n}: need an id and correct=true|false|null")
            if row["id"] in labels:
                raise ValueError(f"{path}:{n}: repeats item {row['id']}")
            labels[row["id"]] = correct
    return labels


def require_labels(
    ids: Sequence[str], a: Labels, b: Labels, noun: str = "items",
) -> None:
    """Raise ValueError unless both graders labeled every id."""
    missing = [i for i in ids if i not in a or i not in b]
    if missing:
        raise ValueError(f"{len(missing)} {noun} lack a label from both graders")


def grader_figures(ids: Sequence[str], labels: Labels) -> dict[str, Any]:
    """One grader's graded count, correct count, precision, and Wilson lower bound."""
    graded = [labels[i] for i in ids if labels[i] is not None]
    correct = sum(1 for x in graded if x)
    n = len(graded)
    return {
        "graded": n,
        "correct": correct,
        "precision": correct / n if n else 0.0,
        "wilson_lower": wilson_lower(correct, n),
    }


def paired_kappa(ids: Sequence[str], a: Labels, b: Labels) -> tuple[float, int, int]:
    """`(kappa, items both graded, items they agreed on)`, unclear labels left out."""
    both = [i for i in ids if a[i] is not None and b[i] is not None]
    kappa = cohens_kappa([bool(a[i]) for i in both], [bool(b[i]) for i in both])
    return kappa, len(both), sum(1 for i in both if a[i] == b[i])


def verdict(
    graders: Sequence[Mapping[str, Any]], bar: float = BAR, min_graded: int = MIN_GRADED,
) -> str:
    """`insufficient`, `pass`, or `fail` from each grader's figures."""
    if any(g["graded"] < min_graded for g in graders):
        return "insufficient"
    if all(g["precision"] >= bar for g in graders):
        return "pass"
    return "fail"


def score_pair(
    ids: Sequence[str], a: Labels, b: Labels,
    bar: float = BAR, min_graded: int = MIN_GRADED,
) -> dict[str, Any]:
    """Per-grader figures, kappa, and the verdict over `ids`."""
    graders = [grader_figures(ids, a), grader_figures(ids, b)]
    kappa, kappa_n, agreed = paired_kappa(ids, a, b)
    return {
        "graders": graders,
        "kappa": kappa,
        "kappa_n": kappa_n,
        "agreed": agreed,
        "bar": bar,
        "min_graded": min_graded,
        "verdict": verdict(graders, bar, min_graded),
    }
