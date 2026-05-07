"""Tests for the v2.x #265 view-flip flag reader.

PR-A scope: the flag is read deterministically and defaults off. No
behavior is gated yet. Subsequent PRs (B/C/D) consume the flag to
gate `insert_belief()` direct calls, drive `aelf rebuild`, and
surface `aelf doctor --explain` lineage.

Each test is a falsifiable hypothesis about flag resolution:

- unset env → off (default of "off" is the rollback path)
- truthy values ("1", "true", "yes", "on") → on (case-insensitive)
- falsy values ("0", "false", "no", "off") → off
- unrecognised values → off (conservative; does not silently flip)
- whitespace tolerated
"""
from __future__ import annotations

import pytest

from aelfrice.derivation_worker import (
    ENV_WRITE_LOG_AUTHORITATIVE,
    is_write_log_authoritative,
)


def test_unset_env_returns_false() -> None:
    assert is_write_log_authoritative(env={}) is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "On", " yes "])
def test_truthy_values_return_true(value: str) -> None:
    assert is_write_log_authoritative(env={ENV_WRITE_LOG_AUTHORITATIVE: value}) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "Off"])
def test_falsy_values_return_false(value: str) -> None:
    assert is_write_log_authoritative(env={ENV_WRITE_LOG_AUTHORITATIVE: value}) is False


@pytest.mark.parametrize("value", ["", "maybe", "2", "yesno"])
def test_unrecognised_values_return_false(value: str) -> None:
    """Conservative: only the defined truthy set flips. Anything else
    stays off so a typo never silently flips authority."""
    assert is_write_log_authoritative(env={ENV_WRITE_LOG_AUTHORITATIVE: value}) is False


def test_live_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read live process env when no kwarg passed."""
    monkeypatch.delenv(ENV_WRITE_LOG_AUTHORITATIVE, raising=False)
    assert is_write_log_authoritative() is False
    monkeypatch.setenv(ENV_WRITE_LOG_AUTHORITATIVE, "1")
    assert is_write_log_authoritative() is True
    monkeypatch.setenv(ENV_WRITE_LOG_AUTHORITATIVE, "off")
    assert is_write_log_authoritative() is False
