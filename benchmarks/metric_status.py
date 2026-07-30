"""The sentinel a benchmark adapter writes when a metric cannot be computed.

Some metrics in the canonical cut are not merely zero, they are
unmeasurable by the harness that reports them. `aelf bench all` runs no
reader (`benchmarks/run.py`), so every adapter hands the joined retrieval
context to a scorer that was written for a model's answer. Exact match
asks whether the prediction *equals* the gold string; a two-thousand-token
context never does, for any corpus, at any retrieval quality. LoCoMo's
adversarial category asks whether the answer was a refusal; nothing in the
pipeline can refuse.

Reporting those as `0.0` states a measurement that was never taken, and
`0.0` is the worst possible score, so the canonical file reads as if
retrieval failed where in fact nothing was asked of it. Worse, a
tolerance band around `0.0` turns any real improvement into a band
excursion (#1160).

Writing `NOT_APPLICABLE` instead says what happened. `tolerance.py`
recognises the sentinel and records the leaf as `Verdict.NOT_APPLICABLE`:
tallied, visible in the report, excluded from the rollup, and — unlike a
`PASS` — not counted as evidence that anything was measured.

Issue: #1160.
"""
from __future__ import annotations

from typing import Any, Final

#: Written in place of a float wherever a metric is structurally
#: uncomputable. A bare string keeps the canonical JSON readable and
#: keeps the leaf out of `tolerance._walk_leaves`, which only collects
#: numeric leaves.
NOT_APPLICABLE: Final[str] = "n/a"

#: Key under which an adapter records *why* each n/a metric is n/a. The
#: leading underscore keeps the block off the band-check walk (see
#: `tolerance._walk_leaves`) while leaving the reason in the artifact for
#: whoever reads it next.
NOT_APPLICABLE_REASONS_KEY: Final[str] = "_not_applicable"


def is_not_applicable(value: Any) -> bool:
    """True when `value` is the not-applicable sentinel.

    Compares case-insensitively against the stripped string so a report
    hand-edited to `"N/A"` still reads as deliberate rather than as a
    non-numeric leaf (which `tolerance.check_report` fails closed on).
    """
    return isinstance(value, str) and value.strip().lower() == NOT_APPLICABLE
