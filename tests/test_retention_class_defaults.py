"""Tests for #290 phase-2: per-ingest-source retention_class defaults.

Covers the spec table at docs/design/belief_retention_class.md § 2 and its
wiring through derivation.derive() and scanner.

Phase-1 (PR #351) shipped the column + python validator. Phase-2
(this PR) ratifies the per-source defaults so live ingest paths stop
landing every new belief as `unknown`.
"""
from __future__ import annotations

import pytest

from aelfrice.derivation import DerivationInput, derive
from aelfrice.models import (
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS,
    INGEST_SOURCE_FILESYSTEM,
    INGEST_SOURCE_GIT,
    INGEST_SOURCE_LEGACY_UNKNOWN,
    INGEST_SOURCE_MCP_REMEMBER,
    INGEST_SOURCE_PYTHON_AST,
    RETENTION_FACT,
    RETENTION_SNAPSHOT,
    RETENTION_UNKNOWN,
    retention_class_for_source,
)


# ---- helper surface -----------------------------------------------------


@pytest.mark.parametrize(
    "source_kind,expected",
    [
        (INGEST_SOURCE_FILESYSTEM, RETENTION_FACT),
        (INGEST_SOURCE_GIT, RETENTION_FACT),
        (INGEST_SOURCE_PYTHON_AST, RETENTION_FACT),
        (INGEST_SOURCE_MCP_REMEMBER, RETENTION_FACT),
        (INGEST_SOURCE_CLI_REMEMBER, RETENTION_FACT),
        (INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS, RETENTION_SNAPSHOT),
        (INGEST_SOURCE_LEGACY_UNKNOWN, RETENTION_UNKNOWN),
    ],
)
def test_retention_class_for_source_table(
    source_kind: str, expected: str
) -> None:
    assert retention_class_for_source(source_kind) == expected


def test_unknown_source_kind_falls_back_to_unknown() -> None:
    # Future / test-only / not-yet-ratified source labels must not
    # silently become 'fact' — that would lie about provenance.
    assert retention_class_for_source("not-a-real-source") == RETENTION_UNKNOWN


def test_no_source_defaults_to_transient() -> None:
    # The spec is explicit: `transient` is operator-supplied only,
    # never assigned by default. Guard against accidental reintroduction.
    for value in {retention_class_for_source(s) for s in [
        INGEST_SOURCE_FILESYSTEM,
        INGEST_SOURCE_GIT,
        INGEST_SOURCE_PYTHON_AST,
        INGEST_SOURCE_MCP_REMEMBER,
        INGEST_SOURCE_CLI_REMEMBER,
        INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS,
        INGEST_SOURCE_LEGACY_UNKNOWN,
    ]}:
        assert value != "transient"


# ---- derive() wiring ----------------------------------------------------


def test_derive_cli_remember_belief_is_fact() -> None:
    out = derive(DerivationInput(
        raw_text="A real fact the operator wants preserved.",
        source_kind=INGEST_SOURCE_CLI_REMEMBER,
        ts="2026-05-02T00:00:00Z",
    ))
    assert out.belief is not None
    assert out.belief.retention_class == RETENTION_FACT


def test_derive_mcp_remember_belief_is_fact() -> None:
    out = derive(DerivationInput(
        raw_text="MCP-written fact.",
        source_kind=INGEST_SOURCE_MCP_REMEMBER,
        ts="2026-05-02T00:00:00Z",
    ))
    assert out.belief is not None
    assert out.belief.retention_class == RETENTION_FACT


def test_derive_git_commit_belief_is_fact() -> None:
    out = derive(DerivationInput(
        raw_text="alice fixed bug",
        source_kind=INGEST_SOURCE_GIT,
        ts="2026-05-02T00:00:00Z",
    ))
    assert out.belief is not None
    assert out.belief.retention_class == RETENTION_FACT
