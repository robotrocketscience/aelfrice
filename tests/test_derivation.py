"""Unit tests for the pure derive() function in derivation.py.

Each test states a falsifiable hypothesis and exercises one or more
source_kind values from INGEST_SOURCE_KINDS. No store I/O; all tests
are pure-function calls.
"""
from __future__ import annotations

import hashlib

import pytest

from aelfrice.derivation import (
    DerivationInput,
    DerivationOutput,
    _belief_id,
    _lock_id,
    _triple_belief_id,
    derive,
)
from aelfrice.models import (
    BELIEF_CORRECTION,
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

_TS = "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# DerivationInput / DerivationOutput shape
# ---------------------------------------------------------------------------


def test_derivation_input_required_fields_only() -> None:
    """Hypothesis: DerivationInput accepts raw_text + source_kind with all
    optional fields defaulting to None / empty string.  Falsifiable if
    construction raises."""
    inp = DerivationInput(raw_text="hello", source_kind=INGEST_SOURCE_FILESYSTEM)
    assert inp.source_path is None
    assert inp.session_id is None
    assert inp.ts == ""
    assert inp.override_belief_type is None


def test_derivation_output_belief_none_has_skip_reason() -> None:
    """Hypothesis: a DerivationOutput with belief=None always carries a
    non-empty skip_reason.  Falsifiable by any skip output that leaves
    skip_reason=None."""
    out = derive(DerivationInput(
        raw_text="What is the default port?",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:README.md:p0",
        ts=_TS,
    ))
    assert out.belief is None
    assert out.skip_reason is not None
    assert out.skip_reason != ""


# ---------------------------------------------------------------------------
# filesystem (classifier path)
# ---------------------------------------------------------------------------


def test_filesystem_factual_belief() -> None:
    """Hypothesis: a plain factual sentence via filesystem yields a factual
    belief with LOCK_NONE and ORIGIN_AGENT_INFERRED.  Falsifiable by any
    other type, lock, or origin."""
    out = derive(DerivationInput(
        raw_text="The default port is 8080 for the dashboard.",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:README.md:p0",
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.type == BELIEF_FACTUAL
    assert out.belief.lock_level == LOCK_NONE
    assert out.belief.origin == ORIGIN_AGENT_INFERRED
    assert out.belief.created_at == _TS


def test_filesystem_requirement_belief() -> None:
    """Hypothesis: a sentence with a requirement keyword yields
    belief_type=requirement.  Falsifiable by any other type."""
    out = derive(DerivationInput(
        raw_text="This project must use uv for environment management.",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:README.md:p1",
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.type == BELIEF_REQUIREMENT


def test_filesystem_preference_belief() -> None:
    """Hypothesis: a sentence with a preference keyword yields
    belief_type=preference.  Falsifiable by any other type."""
    out = derive(DerivationInput(
        raw_text="I prefer atomic commits over batched commits.",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:README.md:p2",
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.type == BELIEF_PREFERENCE


def test_filesystem_correction_belief() -> None:
    """Hypothesis: a sentence that triggers the correction detector yields
    belief_type=correction.  Falsifiable by any other type."""
    out = derive(DerivationInput(
        raw_text="Actually, the default port is 9090, not 8080.",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:README.md:p3",
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.type == BELIEF_CORRECTION


def test_filesystem_question_skipped() -> None:
    """Hypothesis: a question-form sentence via filesystem is skipped
    (persist=False).  Falsifiable if a belief is returned."""
    out = derive(DerivationInput(
        raw_text="What is the default port?",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:README.md:p4",
        ts=_TS,
    ))
    assert out.belief is None
    assert out.skip_reason == "persist=False"


def test_filesystem_empty_text_skipped() -> None:
    """Hypothesis: empty raw_text is skipped.  Falsifiable if a belief
    is returned."""
    out = derive(DerivationInput(
        raw_text="",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:README.md:p5",
        ts=_TS,
    ))
    assert out.belief is None


def test_filesystem_belief_id_stable_and_matches_scheme() -> None:
    """Hypothesis: the belief id is sha256(source_path NUL text)[:16],
    matching the scheme used by ingest._belief_id.  Falsifiable by any
    mismatch."""
    text = "The configuration file lives at /etc/aelfrice/conf."
    source = "doc:README.md:p0"
    expected = hashlib.sha256(f"{source}\x00{text}".encode()).hexdigest()[:16]
    out = derive(DerivationInput(
        raw_text=text,
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path=source,
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.id == expected


def test_filesystem_session_id_propagated() -> None:
    """Hypothesis: session_id on DerivationInput is stamped onto the
    belief.  Falsifiable by a None or different session_id on the belief."""
    out = derive(DerivationInput(
        raw_text="The deploy uses uv only.",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="user",
        session_id="test-session-123",
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.session_id == "test-session-123"


# ---------------------------------------------------------------------------
# python_ast path
# ---------------------------------------------------------------------------


def test_python_ast_factual_belief() -> None:
    """Hypothesis: source_kind=python_ast with a plain docstring sentence
    yields a factual belief.  Falsifiable by any other type or a skip."""
    out = derive(DerivationInput(
        raw_text="Parse a Python module and extract top-level docstrings.",
        source_kind=INGEST_SOURCE_PYTHON_AST,
        source_path="ast:src/aelfrice/scanner.py:func:extract_ast",
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.type == BELIEF_FACTUAL


# ---------------------------------------------------------------------------
# git (triple-extraction path)
# ---------------------------------------------------------------------------


def test_git_always_yields_belief() -> None:
    """Hypothesis: INGEST_SOURCE_GIT always produces a belief regardless
    of text content (no classifier skip).  Falsifiable by any None belief."""
    for phrase in ["the index", "What is this?", "", "  "]:
        out = derive(DerivationInput(
            raw_text=phrase,
            source_kind=INGEST_SOURCE_GIT,
            ts=_TS,
        ))
        assert out.belief is not None, f"expected belief for phrase {phrase!r}"


def test_git_belief_alpha_beta_and_type() -> None:
    """Hypothesis: git-path beliefs have alpha=1.0, beta=1.0, type=factual.
    Falsifiable by any other value."""
    out = derive(DerivationInput(
        raw_text="the new index",
        source_kind=INGEST_SOURCE_GIT,
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.alpha == 1.0
    assert out.belief.beta == 1.0
    assert out.belief.type == BELIEF_FACTUAL
    assert out.belief.lock_level == LOCK_NONE


def test_git_belief_id_matches_triple_extractor_scheme() -> None:
    """Hypothesis: the belief id from derive() for git source_kind matches
    triple_extractor._belief_id_for_phrase (sha256(triple NUL lower)[:16]).
    Falsifiable by any mismatch — a mismatch would break idempotency with
    the triple-ingest path."""
    phrase = "the new index"
    normalized = " ".join(phrase.split()).lower()
    expected = hashlib.sha256(
        f"triple\x00{normalized}".encode("utf-8")
    ).hexdigest()[:16]
    out = derive(DerivationInput(
        raw_text=phrase,
        source_kind=INGEST_SOURCE_GIT,
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.id == expected


def test_git_belief_content_is_normalized_phrase() -> None:
    """Hypothesis: git-path belief content is the whitespace-normalised
    phrase (not the raw text with extra spaces).  Falsifiable by any
    non-normalised content."""
    out = derive(DerivationInput(
        raw_text="  the   new   index  ",
        source_kind=INGEST_SOURCE_GIT,
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.content == "the new index"


# ---------------------------------------------------------------------------
# mcp_remember path
# ---------------------------------------------------------------------------


def test_mcp_remember_yields_user_locked_belief() -> None:
    """Hypothesis: mcp_remember always yields a LOCK_USER belief with
    ORIGIN_USER_STATED and alpha=9.0 / beta=0.5.  Falsifiable by any
    other lock level, origin, or prior."""
    out = derive(DerivationInput(
        raw_text="Always use uv for package management.",
        source_kind=INGEST_SOURCE_MCP_REMEMBER,
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.lock_level == LOCK_USER
    assert out.belief.origin == ORIGIN_USER_STATED
    assert out.belief.alpha == 9.0
    assert out.belief.beta == 0.5
    assert out.belief.locked_at == _TS


def test_mcp_remember_belief_id_matches_lock_scheme() -> None:
    """Hypothesis: mcp_remember id is sha256(lock NUL text)[:16], matching
    cli._lock_id_for and mcp_server._lock_id_for.  Falsifiable by mismatch."""
    stmt = "Always use uv for package management."
    expected = hashlib.sha256(f"lock\x00{stmt}".encode()).hexdigest()[:16]
    out = derive(DerivationInput(
        raw_text=stmt,
        source_kind=INGEST_SOURCE_MCP_REMEMBER,
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.id == expected


# ---------------------------------------------------------------------------
# cli_remember path
# ---------------------------------------------------------------------------


def test_cli_remember_yields_user_locked_belief() -> None:
    """Hypothesis: cli_remember behaves identically to mcp_remember for
    the lock/prior/origin fields.  Falsifiable by any difference."""
    out = derive(DerivationInput(
        raw_text="The deploy must use the staging environment first.",
        source_kind=INGEST_SOURCE_CLI_REMEMBER,
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.lock_level == LOCK_USER
    assert out.belief.origin == ORIGIN_USER_STATED
    assert out.belief.alpha == 9.0
    assert out.belief.beta == 0.5


def test_cli_remember_belief_id_matches_lock_scheme() -> None:
    """Hypothesis: cli_remember id uses the same lock scheme as
    mcp_remember.  Falsifiable by any mismatch with the sha256 formula."""
    stmt = "The deploy must use the staging environment first."
    expected = hashlib.sha256(f"lock\x00{stmt}".encode()).hexdigest()[:16]
    out = derive(DerivationInput(
        raw_text=stmt,
        source_kind=INGEST_SOURCE_CLI_REMEMBER,
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.id == expected


# ---------------------------------------------------------------------------
# override_belief_type (accept_classifications path)
# ---------------------------------------------------------------------------


def test_override_belief_type_bypasses_classify_sentence() -> None:
    """Hypothesis: when override_belief_type is set, derive() uses that
    type directly rather than calling classify_sentence. Concretely: a
    question-form sentence (which classify_sentence would reject with
    persist=False) should persist when override_belief_type is set.
    Falsifiable if the result is skipped."""
    out = derive(DerivationInput(
        raw_text="What is the default port?",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:README.md:p0",
        ts=_TS,
        override_belief_type=BELIEF_FACTUAL,
    ))
    assert out.belief is not None
    assert out.belief.type == BELIEF_FACTUAL


def test_override_belief_type_all_valid_types() -> None:
    """Hypothesis: every belief type in BELIEF_TYPES is accepted by
    override_belief_type and produces a belief.  Falsifiable by any type
    that errors or returns None."""
    for btype in (BELIEF_FACTUAL, BELIEF_CORRECTION, BELIEF_PREFERENCE, BELIEF_REQUIREMENT):
        out = derive(DerivationInput(
            raw_text="The configuration file is at /etc/aelf/conf.",
            source_kind=INGEST_SOURCE_FILESYSTEM,
            source_path="doc:README.md:p0",
            ts=_TS,
            override_belief_type=btype,
        ))
        assert out.belief is not None, f"expected belief for type {btype!r}"
        assert out.belief.type == btype


# ---------------------------------------------------------------------------
# feedback_loop_synthesis and legacy_unknown
# ---------------------------------------------------------------------------


def test_feedback_loop_synthesis_uses_classifier() -> None:
    """Hypothesis: feedback_loop_synthesis goes through classify_sentence
    (classifier path, not lock path).  Falsifiable if lock_level is USER
    or if a plain factual sentence is skipped."""
    out = derive(DerivationInput(
        raw_text="The feedback loop synthesis produces factual beliefs.",
        source_kind=INGEST_SOURCE_FEEDBACK_LOOP_SYNTHESIS,
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.lock_level == LOCK_NONE
    assert out.belief.type == BELIEF_FACTUAL


def test_legacy_unknown_uses_classifier() -> None:
    """Hypothesis: legacy_unknown goes through classify_sentence.
    Falsifiable if the result is a lock or if an obvious factual is skipped."""
    out = derive(DerivationInput(
        raw_text="Pre-migration belief content from an old session.",
        source_kind=INGEST_SOURCE_LEGACY_UNKNOWN,
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.lock_level == LOCK_NONE


# ---------------------------------------------------------------------------
# ts defaults
# ---------------------------------------------------------------------------


def test_empty_ts_triggers_utc_now() -> None:
    """Hypothesis: when ts is empty, derive() stamps a non-empty ISO-8601
    string on the belief.  Falsifiable by an empty or None created_at."""
    out = derive(DerivationInput(
        raw_text="The deploy uses uv only.",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="user",
    ))
    assert out.belief is not None
    assert out.belief.created_at
    assert "T" in out.belief.created_at  # rough ISO-8601 check


# ---------------------------------------------------------------------------
# edges field
# ---------------------------------------------------------------------------


def test_derive_always_returns_empty_edges() -> None:
    """Hypothesis: the edges field is always an empty list in v2.0
    (placeholder for future use).  Falsifiable by any non-empty edges list."""
    for source_kind in (
        INGEST_SOURCE_FILESYSTEM,
        INGEST_SOURCE_GIT,
        INGEST_SOURCE_MCP_REMEMBER,
        INGEST_SOURCE_CLI_REMEMBER,
        INGEST_SOURCE_PYTHON_AST,
    ):
        out = derive(DerivationInput(
            raw_text="Some sentence.",
            source_kind=source_kind,
            ts=_TS,
        ))
        assert out.edges == [], f"expected empty edges for source_kind={source_kind!r}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_derive_is_deterministic() -> None:
    """Hypothesis: calling derive() twice with identical inputs produces
    identical outputs (same id, type, alpha, beta, content).  Falsifiable
    by any field that changes between calls."""
    inp = DerivationInput(
        raw_text="This project must use uv for environment management.",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:README.md:p1",
        ts=_TS,
    )
    out1 = derive(inp)
    out2 = derive(inp)
    assert out1.belief is not None
    assert out2.belief is not None
    assert out1.belief.id == out2.belief.id
    assert out1.belief.type == out2.belief.type
    assert out1.belief.alpha == out2.belief.alpha
    assert out1.belief.beta == out2.belief.beta
    assert out1.belief.content == out2.belief.content
