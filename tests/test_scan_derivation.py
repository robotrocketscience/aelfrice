"""Fixture-backed regression tests for the N-gram Jaccard similarity gate.

Covers:
  - similarity_to_reference detects a near-verbatim paraphrase fixture
    at n=3, threshold=0.6.
  - A clean-control fixture is not flagged.
  - aelf scan-derivation CLI: exit 1 on paraphrase, exit 0 on clean.
  - aelf scan-derivation CLI: exit 2 when reference is missing.

Fixtures live under tests/fixtures/derivation_gate/:
  reference.txt    — the reference document
  paraphrase.txt   — near-verbatim paraphrase (must be flagged)
  clean_control.txt — unrelated text (must not be flagged)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.noise_filter import similarity_to_reference

_FIXTURES = Path(__file__).parent / "fixtures" / "derivation_gate"
_REFERENCE = _FIXTURES / "reference.txt"
_PARAPHRASE = _FIXTURES / "paraphrase.txt"
_CLEAN = _FIXTURES / "clean_control.txt"


# --- Pure-function fixture regression -----------------------------------


def test_paraphrase_detected() -> None:
    """The paraphrase fixture crosses threshold=0.6 at n=3."""
    text = _PARAPHRASE.read_text(encoding="utf-8")
    over, score, excerpt = similarity_to_reference(
        text, _REFERENCE, n=3, threshold=0.6
    )
    assert over is True, f"expected paraphrase to be flagged; score={score:.3f}"
    assert excerpt is not None


def test_clean_control_not_flagged() -> None:
    """The clean-control fixture stays below threshold=0.6 at n=3."""
    text = _CLEAN.read_text(encoding="utf-8")
    over, score, _ = similarity_to_reference(
        text, _REFERENCE, n=3, threshold=0.6
    )
    assert over is False, f"expected clean control not to be flagged; score={score:.3f}"


# --- CLI end-to-end tests ------------------------------------------------


def test_cli_exit_1_on_paraphrase() -> None:
    """aelf scan-derivation exits 1 when the input exceeds threshold."""
    from aelfrice.cli import main

    code = main(
        argv=[
            "scan-derivation",
            "--reference", str(_REFERENCE),
            "--threshold", "0.6",
            "--n", "3",
            str(_PARAPHRASE),
        ]
    )
    assert code == 1, f"expected exit 1 for paraphrase; got {code}"


def test_cli_exit_0_on_clean() -> None:
    """aelf scan-derivation exits 0 when the input is below threshold."""
    from aelfrice.cli import main

    code = main(
        argv=[
            "scan-derivation",
            "--reference", str(_REFERENCE),
            "--threshold", "0.6",
            "--n", "3",
            str(_CLEAN),
        ]
    )
    assert code == 0, f"expected exit 0 for clean control; got {code}"


def test_cli_exit_2_on_missing_reference(tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
    """aelf scan-derivation exits 2 when --reference does not exist."""
    from aelfrice.cli import main

    missing = tmp_path / "does_not_exist.txt"  # type: ignore[operator]
    code = main(
        argv=[
            "scan-derivation",
            "--reference", str(missing),
            str(_CLEAN),
        ]
    )
    assert code == 2, f"expected exit 2 for missing reference; got {code}"
