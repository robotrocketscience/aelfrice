"""The rescued-sentence split is total, reachable, and open (#1490).

`rescued_distinct` counts what the relaxed transcript filter admits. It does
not say what KIND of thing was admitted, and the kinds behave very differently
once stored: a durable policy statement earns its slot, while an ephemeral
session-status line ("No failures yet.") is true of one run and of nothing
afterwards — yet both enter at the same undeflated `user_transcript` prior of
0.75 and compete for the same injection slots.

Measured population is small and did not grow: 6 distinct / 9 rows, unchanged
across a corpus that went 17,592 -> 17,628 sentences. Twice ruled not worth a
mechanism on the ingest path. What ships is the instrument, not a filter.

## Three properties, and why each is here

1. **Total.** The buckets sum to the input. A sentence matching no rule must
   land in `unclassified`, never vanish — a silent drop would make the split
   under-report and hide the novel shape it exists to surface.
2. **Reachable.** Every declared bucket is returned by some input. A declared
   but dead bucket is a permanently zero column that reads as a measurement.
3. **Open.** An unseen shape reaches `unclassified` rather than being absorbed
   by the closed set of three. CI can never watch the real number here — the
   corpus is an untracked local archive — so a growing `unclassified` count is
   the only growth signal this instrument can ever have.

Nothing under test reaches the ingest write path. A wrong bucket produces a
wrong number in a manually-run report, never a dropped belief.
"""
from __future__ import annotations

import inspect
import re

from benchmarks import transcript_noise_admission as harness

# The five sentences the shipped disclosure at `noise_filter.py` names, with
# the class it assigns each. Pinned so a rule change that re-buckets the
# documented examples fails here rather than silently contradicting the
# comment that cites them.
NAMED = {
    "No failures yet.": "ephemeral_status",
    "No mutations attempted.": "ephemeral_status",
    "No telemetry, no network calls, no accounts.": "durable_policy",
    "No action required after aelf setup.": "durable_policy",
    "No work around.": "operator_directive",
}


def test_the_named_sentences_land_in_their_declared_bucket() -> None:
    for sentence, expected in NAMED.items():
        assert harness._rescue_bucket(sentence) == expected, (
            f"{sentence!r} bucketed as "
            f"{harness._rescue_bucket(sentence)!r}, but the disclosure in "
            f"src/aelfrice/noise_filter.py calls it {expected}"
        )


def test_the_buckets_partition_the_input() -> None:
    """No silent drop: the counts sum to exactly what went in."""
    sentences = [*NAMED, "Deploys run from the release branch.", "", "   "]
    counts = harness._bucket_counts(sentences)
    assert sum(counts.values()) == len(sentences), (
        f"buckets sum to {sum(counts.values())} for {len(sentences)} inputs — "
        "a sentence was dropped rather than counted"
    )


def test_every_declared_bucket_is_reachable() -> None:
    """A declared-but-dead bucket is a permanent zero that reads as data.

    Reads the returned string literals out of the source, the same shape used
    by `tests/test_scan_admission_funnel_1398.py`, so an arm that is returned
    but not declared (dropped from the total) and a bucket that is declared
    but never returned both fail.
    """
    src = inspect.getsource(harness._rescue_bucket)
    returned = set(re.findall(r'return "([a-z_]+)"', src))
    assert returned == set(harness.RESCUE_BUCKETS), (
        f"_rescue_bucket returns {sorted(returned)} but RESCUE_BUCKETS "
        f"declares {sorted(harness.RESCUE_BUCKETS)}"
    )


def test_an_unseen_shape_reaches_unclassified() -> None:
    """The closed set of three must not absorb a new shape.

    Without this the instrument reports the same three numbers forever, which
    is the failure the 2026-08-12 ruling names.
    """
    for sentence in (
        "Deploys run from the release branch.",
        "The north aisle holds eleven barrels.",
        "Migrations are reversible.",
    ):
        assert harness._rescue_bucket(sentence) == "unclassified", (
            f"{sentence!r} was absorbed as "
            f"{harness._rescue_bucket(sentence)!r}; a shape none of the three "
            "rules was written for must stay visible"
        )


def test_the_report_carries_both_denominators() -> None:
    """Rows and distinct are different claims; neither derives from the other.

    Three sentences account for most of the rescue at four occurrences each,
    so a row count reads as content items and overstates by ~2x.
    """
    rescued = ["No failures yet.", "No failures yet.", "No work around."]
    rows = harness._bucket_counts(rescued)
    distinct = harness._bucket_counts(sorted(set(rescued)))
    assert rows["ephemeral_status"] == 2
    assert distinct["ephemeral_status"] == 1
    assert sum(rows.values()) == 3
    assert sum(distinct.values()) == 2
