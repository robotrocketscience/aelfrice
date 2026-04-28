"""Unit tests for derivation.derive() — pure function, no store needed.

Covers:
- each INGEST_SOURCE_KIND produces a Belief (or rejects correctly)
- classify-based path: persist=False yields belief=None + skip_reason
- lock-based path: yields LOCK_USER + ORIGIN_USER_STATED belief
- purity: identical inputs produce equal outputs
- invalid source_kind raises ValueError
"""
from __future__ import annotations

import pytest

from aelfrice.derivation import DerivationInput, DerivationOutput, derive
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_PREFERENCE,
    BELIEF_REQUIREMENT,
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS,
    INGEST_SOURCE_FILESYSTEM,
    INGEST_SOURCE_GIT,
    INGEST_SOURCE_LEGACY_UNKNOWN,
    INGEST_SOURCE_MCP_REMEMBER,
    INGEST_SOURCE_PYTHON_AST,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
)

_TS = "2026-04-28T00:00:00Z"


def _inp(
    raw_text: str,
    source_kind: str,
    source_path: str | None = "doc:README.md:p0",
    session_id: str | None = None,
) -> DerivationInput:
    return DerivationInput(
        raw_text=raw_text,
        source_kind=source_kind,
        source_path=source_path,
        raw_meta=None,
        session_id=session_id,
        ts=_TS,
        classifier_version=None,
        rule_set_hash=None,
    )


# --- Purity -----------------------------------------------------------------


def test_derive_is_pure_identical_inputs_equal_outputs() -> None:
    """Calling derive() twice with the same input yields structurally equal
    results — same belief id, same alpha, same everything."""
    inp = _inp(
        "The configuration file lives at the default path.",
        INGEST_SOURCE_FILESYSTEM,
    )
    out1 = derive(inp)
    out2 = derive(inp)
    assert out1.belief is not None
    assert out2.belief is not None
    assert out1.belief.id == out2.belief.id
    assert out1.belief.alpha == out2.belief.alpha
    assert out1.belief.beta == out2.belief.beta
    assert out1.belief.type == out2.belief.type
    assert out1.skip_reason == out2.skip_reason


# --- source_kind: filesystem ------------------------------------------------


def test_filesystem_factual_statement_produces_belief() -> None:
    out = derive(_inp(
        "The default port is 8080 for the dashboard service.",
        INGEST_SOURCE_FILESYSTEM,
    ))
    assert out.belief is not None
    assert out.belief.type == BELIEF_FACTUAL
    assert out.belief.lock_level == LOCK_NONE
    assert out.belief.origin == ORIGIN_AGENT_INFERRED
    assert out.skip_reason is None


def test_filesystem_question_rejected() -> None:
    out = derive(_inp(
        "What is the default port for the dashboard service?",
        INGEST_SOURCE_FILESYSTEM,
    ))
    assert out.belief is None
    assert out.skip_reason == "persist=False"
    assert out.edges == []


def test_filesystem_empty_text_rejected() -> None:
    out = derive(_inp("", INGEST_SOURCE_FILESYSTEM))
    assert out.belief is None
    assert out.skip_reason == "persist=False"


def test_filesystem_preference_classified_correctly() -> None:
    out = derive(_inp(
        "I prefer using uv for all Python package management.",
        INGEST_SOURCE_FILESYSTEM,
        source_path="user",
    ))
    assert out.belief is not None
    assert out.belief.type == BELIEF_PREFERENCE


def test_filesystem_requirement_classified_correctly() -> None:
    out = derive(_inp(
        "You must use SSH key authentication for all deployments.",
        INGEST_SOURCE_FILESYSTEM,
        source_path="user",
    ))
    assert out.belief is not None
    assert out.belief.type == BELIEF_REQUIREMENT


def test_filesystem_belief_id_stable_across_calls() -> None:
    inp = _inp("Aelfrice stores beliefs in a local SQLite database.", INGEST_SOURCE_FILESYSTEM)
    out1 = derive(inp)
    out2 = derive(inp)
    assert out1.belief is not None
    assert out2.belief is not None
    assert out1.belief.id == out2.belief.id


def test_filesystem_session_id_stamped_on_belief() -> None:
    out = derive(_inp(
        "The project uses conventional commits for all changes.",
        INGEST_SOURCE_FILESYSTEM,
        session_id="session-abc123",
    ))
    assert out.belief is not None
    assert out.belief.session_id == "session-abc123"


def test_filesystem_ts_written_to_created_at() -> None:
    out = derive(_inp(
        "The benchmark harness evaluates retrieval quality.",
        INGEST_SOURCE_FILESYSTEM,
    ))
    assert out.belief is not None
    assert out.belief.created_at == _TS


# --- source_kind: git -------------------------------------------------------


def test_git_factual_commit_subject_produces_belief() -> None:
    out = derive(_inp(
        "feat: add BM25 retrieval index for offline search",
        INGEST_SOURCE_GIT,
        source_path="git:commit:abc1234",
    ))
    assert out.belief is not None
    assert out.belief.type == BELIEF_FACTUAL
    assert out.belief.origin == ORIGIN_AGENT_INFERRED


def test_git_question_rejected() -> None:
    out = derive(_inp(
        "What does this commit change in the retrieval layer?",
        INGEST_SOURCE_GIT,
        source_path="git:commit:abc1234",
    ))
    assert out.belief is None
    assert out.skip_reason == "persist=False"


# --- source_kind: python_ast ------------------------------------------------


def test_python_ast_docstring_produces_belief() -> None:
    out = derive(_inp(
        "Sentence classification: assign one of the four belief types and a "
        "source-adjusted Beta prior.",
        INGEST_SOURCE_PYTHON_AST,
        source_path="ast:src/aelfrice/classification.py:module",
    ))
    assert out.belief is not None
    assert out.belief.origin == ORIGIN_AGENT_INFERRED
    assert out.belief.lock_level == LOCK_NONE


# --- source_kind: mcp_remember ----------------------------------------------


def test_mcp_remember_produces_locked_belief() -> None:
    out = derive(_inp(
        "Always use SSH key authentication for production deployments.",
        INGEST_SOURCE_MCP_REMEMBER,
        source_path=None,
    ))
    assert out.belief is not None
    assert out.belief.lock_level == LOCK_USER
    assert out.belief.origin == ORIGIN_USER_STATED
    assert out.belief.locked_at == _TS
    assert out.skip_reason is None


def test_mcp_remember_high_confidence_prior() -> None:
    out = derive(_inp(
        "Never commit secrets to the repository.",
        INGEST_SOURCE_MCP_REMEMBER,
        source_path=None,
    ))
    assert out.belief is not None
    # Lock path uses fixed high-confidence priors matching the requirement prior.
    assert out.belief.alpha == 9.0
    assert out.belief.beta == 0.5


def test_mcp_remember_question_still_persists() -> None:
    """Lock path never rejects — caller is asserting a belief, not classifying."""
    out = derive(_inp(
        "What should we use for authentication?",
        INGEST_SOURCE_MCP_REMEMBER,
        source_path=None,
    ))
    # Lock path bypasses classify_sentence entirely — always produces a belief.
    assert out.belief is not None
    assert out.belief.lock_level == LOCK_USER


# --- source_kind: cli_remember ----------------------------------------------


def test_cli_remember_produces_locked_belief() -> None:
    out = derive(_inp(
        "Prefer uv over pip for all Python environment management.",
        INGEST_SOURCE_CLI_REMEMBER,
        source_path=None,
    ))
    assert out.belief is not None
    assert out.belief.lock_level == LOCK_USER
    assert out.belief.origin == ORIGIN_USER_STATED


def test_cli_remember_and_mcp_remember_same_text_same_id() -> None:
    """mcp_remember and cli_remember with no source_path both fall back to
    source_kind as the id-hash key — so they produce different ids for the
    same text (different source_kind = different derivation context)."""
    out_mcp = derive(_inp("Use uv for Python.", INGEST_SOURCE_MCP_REMEMBER, source_path=None))
    out_cli = derive(_inp("Use uv for Python.", INGEST_SOURCE_CLI_REMEMBER, source_path=None))
    assert out_mcp.belief is not None
    assert out_cli.belief is not None
    # Different source_kind → different id hash
    assert out_mcp.belief.id != out_cli.belief.id


# --- source_kind: feedback_loop_synthesis -----------------------------------


def test_feedback_loop_synthesis_produces_belief() -> None:
    out = derive(_inp(
        "The feedback loop converges after three correction cycles.",
        INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS,
        source_path="synthesis:session-abc",
    ))
    assert out.belief is not None
    assert out.belief.origin == ORIGIN_AGENT_INFERRED


# --- source_kind: legacy_unknown --------------------------------------------


def test_legacy_unknown_produces_belief() -> None:
    out = derive(_inp(
        "Pre-migration belief backfilled from the old schema.",
        INGEST_SOURCE_LEGACY_UNKNOWN,
        source_path="legacy:row:42",
    ))
    assert out.belief is not None
    assert out.belief.origin == ORIGIN_AGENT_INFERRED
    assert out.belief.lock_level == LOCK_NONE


# --- Invalid source_kind ----------------------------------------------------


def test_invalid_source_kind_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown source_kind"):
        derive(_inp("Some text.", "bogus_kind"))


# --- Edge list --------------------------------------------------------------


def test_derive_returns_empty_edges_list() -> None:
    out = derive(_inp(
        "The context rebuilder stitches beliefs into a coherent block.",
        INGEST_SOURCE_FILESYSTEM,
    ))
    assert out.edges == []


# --- DerivationOutput invariants -------------------------------------------


def test_belief_none_implies_skip_reason_set() -> None:
    out = derive(_inp("What is the answer?", INGEST_SOURCE_FILESYSTEM))
    assert out.belief is None
    assert out.skip_reason is not None


def test_belief_present_implies_skip_reason_none() -> None:
    out = derive(_inp(
        "The retrieval layer uses BM25 plus TF-IDF scoring.",
        INGEST_SOURCE_FILESYSTEM,
    ))
    assert out.belief is not None
    assert out.skip_reason is None
