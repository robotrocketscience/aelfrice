"""Adversarial bench fixture for the Surface B phantom-promotion rule.

Loads tests/fixtures/promotion_adversarial.json and runs each case against
``aelfrice.promotion.find_phantom_lock_matches``. Two case categories:

- ``regression_cases``: rule currently behaves as expected — strict assertion.
  A regression here means Surface B's behavior on a case the rule HANDLES
  correctly today started returning the wrong verdict.
- ``edge_cases``: rule does NOT currently behave as expected — marked
  ``xfail(strict=True)``. Each documents a known failure mode (synonym
  substitution, antonym substitution, etc.). When a future Surface B
  improvement starts handling the case correctly, the case **fails** —
  deliberately. Move it to ``regression_cases`` with ``status: passes``,
  update ``_meta`` counts, and record what closed it in the rationale.

  Strict is the point. Under ``strict=False`` an improved rule produces
  XPASS, which is a non-failing outcome nothing gates on, so the marker
  outlives the defect it documents and quietly suppresses a guard that
  has started working. C6-01..04 sat that way after 39745247 fixed the
  all-stopword promotion path.

Skips entirely if Surface B is not yet shipped (gated on #616 merge).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "promotion_adversarial.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def _surface_b_available() -> bool:
    try:
        from aelfrice.promotion import find_phantom_lock_matches  # noqa: F401
        return True
    except ImportError:
        return False


def _evaluate(phantom_text: str, lock_text: str) -> bool:
    """Run Surface B against an in-memory store containing the phantom.

    Returns True if the rule promotes the phantom under the lock_text.
    """
    import hashlib

    from aelfrice.models import (
        BELIEF_SPECULATIVE,
        LOCK_NONE,
        ORIGIN_SPECULATIVE,
        Belief,
    )
    from aelfrice.promotion import find_phantom_lock_matches
    from aelfrice.store import MemoryStore

    content_hash = hashlib.sha256(phantom_text.encode("utf-8")).hexdigest()
    belief_id = f"phantom-{content_hash[:12]}"
    phantom = Belief(
        id=belief_id,
        content=phantom_text,
        content_hash=content_hash,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_SPECULATIVE,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-11T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_SPECULATIVE,
    )
    store = MemoryStore(":memory:")
    store.insert_belief(phantom)
    matches = find_phantom_lock_matches(store, lock_text)
    return belief_id in matches


def _build_param_list(category: str) -> list:
    if not FIXTURE_PATH.exists():
        return []
    fixture = _load_fixture()
    cases = fixture.get(category, [])
    params = []
    for case in cases:
        marks = []
        if case.get("status") == "known_failure":
            marks.append(pytest.mark.xfail(
                reason=f"{case['class_id']}: {case.get('rationale', '')[:80]}",
                # strict=True so a case the rule starts handling FAILS
                # rather than reporting XPASS. Under strict=False the
                # marker cannot tell anyone it has gone stale: XPASS is a
                # non-failing outcome that no gate reads, so a
                # known_failure entry silently outlives the failure it
                # documents. That happened — C6-01..04 were fixed by
                # 39745247 and sat here as decorative markers, describing
                # a defect the rule no longer has and suppressing a guard
                # that was working. A red build naming the case is the
                # only signal that reliably gets the fixture updated.
                strict=True,
            ))
        params.append(pytest.param(case, marks=marks, id=case["id"]))
    return params


@pytest.mark.skipif(
    not _surface_b_available(),
    reason="Surface B (find_phantom_lock_matches) not importable — shipped via #616 "
           "but the predicate stays defensive for back-revision benches against pre-#616 trees",
)
class TestPromotionAdversarial:
    """Adversarial bench fixture for Surface B."""

    @pytest.mark.parametrize("case", _build_param_list("regression_cases"))
    def test_regression_case(self, case: dict) -> None:
        """Cases where Surface B currently behaves as expected.

        A failure here is a regression in Surface B's correct-behavior surface.
        """
        actual = _evaluate(case["phantom_text"], case["lock_text"])
        assert actual == case["expected_should_promote"], (
            f"{case['id']} ({case['class_id']}): "
            f"expected {case['expected_should_promote']}, got {actual}. "
            f"Rationale: {case.get('rationale', '')}"
        )

    @pytest.mark.parametrize("case", _build_param_list("edge_cases"))
    def test_edge_case(self, case: dict) -> None:
        """Cases where Surface B does NOT currently behave as expected.

        Marked xfail. When Surface B is improved (e.g. acronym expansion,
        stemming, synonym matching), individual cases will start passing —
        flip them from edge_cases to regression_cases and remove xfail.
        """
        actual = _evaluate(case["phantom_text"], case["lock_text"])
        assert actual == case["expected_should_promote"], (
            f"{case['id']} ({case['class_id']}): "
            f"expected {case['expected_should_promote']}, got {actual}. "
            f"Rationale: {case.get('rationale', '')}"
        )


def test_fixture_self_consistent() -> None:
    """Sanity check: the fixture file is well-formed and counts agree."""
    fixture = _load_fixture()
    meta = fixture.get("_meta", {})
    edge_cases = fixture.get("edge_cases", [])
    regression_cases = fixture.get("regression_cases", [])
    assert meta.get("edge_count") == len(edge_cases), (
        f"edge_count mismatch: meta={meta.get('edge_count')}, actual={len(edge_cases)}"
    )
    assert meta.get("regression_count") == len(regression_cases), (
        f"regression_count mismatch: meta={meta.get('regression_count')}, actual={len(regression_cases)}"
    )
    assert meta.get("total") == len(edge_cases) + len(regression_cases)
    # Every case has the required fields
    for case in edge_cases + regression_cases:
        for field in ["id", "class_id", "axis", "phantom_text", "lock_text",
                      "expected_should_promote", "rationale", "status"]:
            assert field in case, f"case {case.get('id')} missing field {field}"


def test_known_failures_are_strict() -> None:
    """A stale `known_failure` must fail the build, not report XPASS.

    This is the guard on the guard. XPASS is a non-failing outcome that
    nothing gates on, so under `strict=False` a case the rule has started
    handling correctly stays marked as a known defect indefinitely — the
    marker describes a failure that no longer exists and suppresses a
    working assertion. C6-01..04 sat that way after 39745247 closed the
    all-stopword promotion path.

    Falsifiable by flipping either the per-marker `strict` or the
    `xfail_strict` default back."""
    import tomllib

    params = _build_param_list("edge_cases")
    assert params, "no edge cases found — this assertion would pass for free"
    for param in params:
        xfails = [m for m in param.marks if m.name == "xfail"]
        assert xfails, f"{param.id} lost its xfail marker"
        assert xfails[0].kwargs.get("strict") is True, (
            f"{param.id} is xfail(strict=False): if the rule starts "
            f"handling it, the case reports XPASS and nothing notices"
        )

    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)
    assert config["tool"]["pytest"]["ini_options"].get("xfail_strict") is True, (
        "xfail_strict is not the default, so a future marker added without "
        "an explicit strict= inherits the silent behaviour"
    )


def test_no_case_is_both_a_known_failure_and_a_regression() -> None:
    """The two buckets must stay disjoint by id.

    Moving a closed case is a two-step edit — append to
    `regression_cases`, remove from `edge_cases` — and doing only the
    first leaves the case asserted strictly *and* marked as a known
    failure, which under strict=True is a guaranteed red build with a
    confusing cause."""
    fixture = _load_fixture()
    edge_ids = {c["id"] for c in fixture.get("edge_cases", [])}
    regression_ids = {c["id"] for c in fixture.get("regression_cases", [])}
    overlap = sorted(edge_ids & regression_ids)
    assert not overlap, f"cases in both buckets: {overlap}"
    # And statuses match the bucket they sit in.
    for case in fixture.get("edge_cases", []):
        assert case["status"] == "known_failure", (
            f"{case['id']} sits in edge_cases with status "
            f"{case['status']!r} — move it to regression_cases"
        )
    for case in fixture.get("regression_cases", []):
        assert case["status"] != "known_failure", (
            f"{case['id']} sits in regression_cases still marked "
            f"known_failure"
        )
