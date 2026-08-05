"""#1345: every `Belief` field is classified, or the guard says so.

The replay probe is what #1157 calls "the falsification instrument for
every other claim here". Before this, 13 of 22 `Belief` fields were
compared by neither the strict contract nor the #1167 mutable bucket --
including `valid_to` (so an all-soft-deleted store passed), and
`lock_tier` / `lock_expires_at` / `locked_at` (so the #1314 time-boxed
lock window was invisible to it).

The enumeration is the smaller half. The gap existed because nothing
forced a decision when a field was added, so a fix that only names
today's 13 goes stale on the next schema change -- exactly how the parent
umbrella's own "3 of 21" went stale after #1167 shipped and nobody
noticed. The guard below is what stops that.
"""
from __future__ import annotations

import dataclasses

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.replay import (
    EXCLUDED_FIELDS,
    FullEqualityReport,
    MUTABLE_FIELDS,
    STRICT_FIELDS,
    _FLOAT_MUTABLE_FIELDS,
    _mutable_fields_diff,
)

# No pseudo-fields remain in `MUTABLE_FIELDS`. `edge_set` used to sit
# there as a reporting key despite not being a `Belief` column; #1354
# moved it out into its own drift-triggering counter, so every name in
# the classified set is now a real column. Kept as an empty frozenset so
# the set algebra below still states that the exclusion is deliberate.
_PSEUDO_FIELDS: frozenset[str] = frozenset()


def _belief_field_names() -> set[str]:
    return {f.name for f in dataclasses.fields(Belief)}


def _classified() -> set[str]:
    return (
        set(STRICT_FIELDS) | set(MUTABLE_FIELDS) | set(EXCLUDED_FIELDS)
    ) - _PSEUDO_FIELDS


def test_every_belief_field_is_classified() -> None:
    """The rule this issue exists to make permanent.

    A field classified nowhere is compared by nothing, and nothing else in
    the suite notices -- which is the whole defect. Adding a column to
    `Belief` must force a decision about whether it is log-derivable.
    """
    unclassified = sorted(_belief_field_names() - _classified())
    assert unclassified == [], (
        f"{len(unclassified)} Belief field(s) are compared by neither the "
        "strict contract nor the mutable bucket, and are not on the "
        "excluded list with a reason (#1345): " + ", ".join(unclassified)
    )


def test_the_classification_is_a_partition() -> None:
    """Exactly one set each, or the report double-counts or contradicts."""
    strict, mutable, excluded = (
        set(STRICT_FIELDS),
        set(MUTABLE_FIELDS) - _PSEUDO_FIELDS,
        set(EXCLUDED_FIELDS),
    )
    assert strict & mutable == set(), strict & mutable
    assert strict & excluded == set(), strict & excluded
    assert mutable & excluded == set(), mutable & excluded


def test_no_classified_name_is_a_phantom() -> None:
    """The inverse direction: a renamed column leaves a dead entry behind,
    and a dead entry in `STRICT_FIELDS` silently stops being compared
    while still reading as covered."""
    phantom = sorted(_classified() - _belief_field_names())
    assert phantom == [], (
        "these are classified but are not `Belief` fields, so they are "
        "compared against nothing: " + ", ".join(phantom)
    )


def test_the_classification_is_not_empty() -> None:
    """A scan that sees no fields satisfies every assertion above.

    `dataclasses.fields` on the wrong object, or an import that resolves
    to a stub, would make the guard vacuous and it would read exactly like
    a fully-classified tree.
    """
    assert len(_belief_field_names()) >= 20, _belief_field_names()
    assert len(STRICT_FIELDS) >= 3
    assert len(set(MUTABLE_FIELDS) - _PSEUDO_FIELDS) >= 15


def test_edge_set_is_not_folded_back_into_the_informational_bucket() -> None:
    """`edge_set` must stay out of `MUTABLE_FIELDS` and keep its own
    drift-triggering counter (#1354).

    Every name in `MUTABLE_FIELDS` is informational by construction — the
    bucket exists for fields no post-ingest operation can be blamed for.
    The edge set is not one of those: `derive()` reconstructs it from the
    log row, so a divergence is a derivation regression. Folding it back
    in would silently restore exactly the blindness #1345 exists to
    prevent, and no other assertion in this file would notice.
    """
    assert "edge_set" not in set(MUTABLE_FIELDS)
    assert "edge_set_divergence" in {
        f.name for f in dataclasses.fields(FullEqualityReport)
    }


def test_every_exclusion_carries_a_reason() -> None:
    """An exclusion without a stated reason is indistinguishable from an
    oversight, which is the state this issue found the 13 fields in."""
    for name, reason in EXCLUDED_FIELDS.items():
        assert isinstance(reason, str) and len(reason) >= 40, (
            f"{name} is excluded with no substantive reason: {reason!r}"
        )


@pytest.mark.parametrize(
    "name",
    ["valid_to", "lock_tier", "lock_expires_at", "locked_at",
     "corroboration_count", "session_id", "project_context"],
)
def test_the_fields_this_issue_was_filed_over_are_compared(name: str) -> None:
    """Named individually so the fix cannot be satisfied by excluding them.

    Moving any of these to `EXCLUDED_FIELDS` would make
    `test_every_belief_field_is_classified` green again while restoring
    the exact blindness #1345 reports -- a store whose retirement flags or
    lock windows were corrupted passing the probe.
    """
    assert name in set(MUTABLE_FIELDS), (
        f"{name} was moved out of the compared set; that restores the "
        "#1345 blindness rather than fixing it"
    )


def test_every_float_field_is_also_in_the_mutable_tuple() -> None:
    """`_mutable_fields_diff` walks `MUTABLE_FIELDS` and branches on this set.

    A float field named here but absent from `MUTABLE_FIELDS` would never be
    reached, so it would stop being compared while still reading as handled.
    """
    assert _FLOAT_MUTABLE_FIELDS <= set(MUTABLE_FIELDS), (
        sorted(_FLOAT_MUTABLE_FIELDS - set(MUTABLE_FIELDS))
    )


def test_the_diff_is_keyed_in_declared_field_order() -> None:
    """The report order is `MUTABLE_FIELDS` order, not a set's iteration order.

    `MUTABLE_FIELDS` documents itself as "the reporting order", and the diff
    dict is rendered verbatim into the drift examples `aelf doctor replay`
    prints. Walking a `frozenset` instead keys that output on string hash
    randomisation, so the same store prints a different field order on every
    process — nondeterministic output from the instrument whose purpose is to
    falsify the determinism claim.
    """
    base = Belief(
        id="b1",
        content="c",
        content_hash="h",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
    )
    other = dataclasses.replace(
        base,
        alpha=2.0,
        beta=3.0,
        hibernation_score=0.5,
        lock_level=LOCK_USER,
        session_id="s",
    )
    diff = _mutable_fields_diff(base, other)
    changed = [n for n in MUTABLE_FIELDS if n in diff]
    assert list(diff) == changed, (list(diff), changed)
    # Guard the guard: an assertion over an empty diff is vacuous.
    assert len(diff) >= 4, diff


def test_created_at_is_excluded_on_the_record() -> None:
    """The one field measurement moved back out of the compared set.

    Comparing it reported divergence on 25 of 25 rows of a clean synthetic
    store -- the canonical value is when the writer ran, the re-derived
    one is the log row's `ts`. A field that diverges on every row is not a
    signal, it is a constant that hides the ones that are. Pinned so the
    next person to enumerate fields does not "fix" the omission and drown
    `mutable_divergence` again.
    """
    assert "created_at" in EXCLUDED_FIELDS
    assert "created_at" not in set(MUTABLE_FIELDS)
