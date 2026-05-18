"""Resolver tests for v3.2 #858 — `active_project_context()`.

Reads `$AELFRICE_PROJECT_CONTEXT` and returns the stripped string,
empty when unset / whitespace-only. The hook lane's project-context
filter consults this resolver once per UserPromptSubmit; the empty-
string return is the "no filter" marker that preserves pre-#858
retrieval behaviour.
"""
from __future__ import annotations

import pytest

from aelfrice.db_paths import PROJECT_CONTEXT_ENV, active_project_context


def test_unset_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROJECT_CONTEXT_ENV, raising=False)
    assert active_project_context() == ""


def test_empty_string_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit '' env value reads back as '' — same as unset."""
    monkeypatch.setenv(PROJECT_CONTEXT_ENV, "")
    assert active_project_context() == ""


def test_whitespace_only_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace-only env value is treated as unset (no-filter marker)."""
    monkeypatch.setenv(PROJECT_CONTEXT_ENV, "   \t  ")
    assert active_project_context() == ""


def test_simple_value_returned_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PROJECT_CONTEXT_ENV, "retrieval-v3")
    assert active_project_context() == "retrieval-v3"


def test_surrounding_whitespace_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leading / trailing whitespace stripped; internal preserved."""
    monkeypatch.setenv(PROJECT_CONTEXT_ENV, "  retrieval v3  ")
    assert active_project_context() == "retrieval v3"


def test_env_var_name_is_public_constant() -> None:
    """The env var name is exported as a Final constant so callers do
    not depend on a magic string. Stability contract: this name is part
    of the project's external API."""
    assert PROJECT_CONTEXT_ENV == "AELFRICE_PROJECT_CONTEXT"
