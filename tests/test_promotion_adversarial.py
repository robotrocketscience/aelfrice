"""Adversarial bench fixture for the Surface B phantom-promotion rule.

Loads tests/fixtures/promotion_adversarial.json and runs each case against
``aelfrice.promotion.find_phantom_lock_matches``. Two case categories:

- ``regression_cases``: rule currently behaves as expected — strict assertion.
  A regression here means Surface B's behavior on a case the rule HANDLES
  correctly today started returning the wrong verdict.
- ``edge_cases``: rule does NOT currently behave as expected — marked xfail.
  Each documents a known failure mode (synonym substitution, antonym
  substitution, etc.). When a future Surface B improvement starts handling
  the case correctly, the xfail flips to "unexpectedly passing" and the
  marker should be removed.

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
        demotion_pressure=0,
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
                strict=False,
            ))
        params.append(pytest.param(case, marks=marks, id=case["id"]))
    return params


@pytest.mark.skipif(
    not _surface_b_available(),
    reason="Surface B (find_phantom_lock_matches) not yet on main — gated on #616 merge",
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
