"""#899 StructMemEval accounting settlement-tuple scorer.

The legacy `check_state_correctness` word-overlap never checks settlement
amounts or directions. `score_accounting_prediction` parses both the
reference (a " | "-joined list of acceptable alternative settlements) and
the prediction into directed (payer, payee, amount) sets and accepts the
prediction if it matches ANY alternative within an amount tolerance.
"""
from __future__ import annotations

from benchmarks.structmemeval_adapter import (
    _parse_settlement_transactions,
    _transaction_sets_match,
    score_accounting_prediction,
    score_prediction,
)

_REF = (
    "Settlement: Alice -> Bob : 91.00 EUR, Alice -> Charlie : 16.00 EUR | "
    "Settlement: Alice -> Bob : 145.00 EUR, Alice -> Charlie : 52.00 EUR | "
    "Settlement: Alice -> Bob : 284.00 EUR, Charlie -> Bob : 33.50 EUR"
)


def test_parse_arrow_form() -> None:
    tx = _parse_settlement_transactions("Alice -> Bob : 91.00 EUR, Charlie -> Bob : 33.50 EUR")
    assert tx == {("alice", "bob", 91.0), ("charlie", "bob", 33.5)}


def test_parse_pays_form() -> None:
    tx = _parse_settlement_transactions("Alice pays 91 to Bob\nCharlie pays 33.50 to Bob")
    assert tx == {("alice", "bob", 91.0), ("charlie", "bob", 33.5)}


def test_match_exact_alternative() -> None:
    assert score_accounting_prediction("Alice pays 91 to Bob. Alice pays 16 to Charlie.", _REF) is True


def test_match_third_alternative_with_cents() -> None:
    assert score_accounting_prediction("Alice pays 284.00 to Bob, Charlie pays 33.50 to Bob", _REF) is True


def test_wrong_amount_fails() -> None:
    assert score_accounting_prediction("Alice pays 99 to Bob. Alice pays 16 to Charlie.", _REF) is False


def test_wrong_direction_fails() -> None:
    # Right amounts, reversed payer/payee.
    assert score_accounting_prediction("Bob pays 91 to Alice. Charlie pays 16 to Alice.", _REF) is False


def test_missing_transaction_fails() -> None:
    assert score_accounting_prediction("Alice pays 91 to Bob.", _REF) is False


def test_extra_transaction_fails() -> None:
    assert score_accounting_prediction(
        "Alice pays 91 to Bob. Alice pays 16 to Charlie. Bob pays 5 to Alice.", _REF,
    ) is False


def test_no_transactions_fails() -> None:
    assert score_accounting_prediction("I cannot determine the settlement.", _REF) is False


def test_within_tolerance_passes() -> None:
    # 91.00 vs 91.4 is within the 0.5 tolerance.
    assert score_accounting_prediction("Alice pays 91.4 to Bob. Alice pays 16 to Charlie.", _REF) is True


def test_transaction_sets_match_requires_same_cardinality() -> None:
    a = {("alice", "bob", 91.0)}
    b = {("alice", "bob", 91.0), ("alice", "charlie", 16.0)}
    assert _transaction_sets_match(a, b) is False


def test_score_prediction_dispatches_accounting() -> None:
    assert score_prediction("Alice pays 91 to Bob. Alice pays 16 to Charlie.", _REF, "accounting") is True
    assert score_prediction("Bob pays 91 to Alice. Charlie pays 16 to Alice.", _REF, "accounting") is False
