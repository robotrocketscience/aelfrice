"""Unit tests for type-aware compression (#434).

Covers the strategy table, the headline edge cases, the locked-override
rule, and the token-monotone invariant.
"""
from __future__ import annotations

import pytest

from aelfrice.compression import (
    MAX_HEADLINE_CHARS,
    STRATEGY_HEADLINE,
    STRATEGY_STUB,
    STRATEGY_VERBATIM,
    CompressedBelief,
    _estimate_tokens,
    compress_for_retrieval,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    RETENTION_FACT,
    RETENTION_SNAPSHOT,
    RETENTION_TRANSIENT,
    RETENTION_UNKNOWN,
    Belief,
)


def _mk(
    content: str,
    *,
    bid: str = "b1",
    retention_class: str = RETENTION_UNKNOWN,
    lock_level: str = LOCK_NONE,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        created_at="2026-05-08T00:00:00Z",
        last_retrieved_at=None,
        retention_class=retention_class,
    )


# --- Strategy table ----------------------------------------------------


def test_fact_unlocked_is_verbatim() -> None:
    b = _mk("a long fact-class belief content. with more sentences. here.",
            retention_class=RETENTION_FACT)
    cb = compress_for_retrieval(b, locked=False)
    assert cb.strategy == STRATEGY_VERBATIM
    assert cb.rendered == b.content


def test_fact_locked_is_verbatim() -> None:
    b = _mk("locked fact content.", retention_class=RETENTION_FACT)
    cb = compress_for_retrieval(b, locked=True)
    assert cb.strategy == STRATEGY_VERBATIM


def test_unknown_class_is_verbatim() -> None:
    b = _mk("uncategorised belief content. another sentence.",
            retention_class=RETENTION_UNKNOWN)
    cb = compress_for_retrieval(b, locked=False)
    assert cb.strategy == STRATEGY_VERBATIM
    assert cb.rendered == b.content


def test_snapshot_unlocked_takes_headline() -> None:
    content = (
        "the first sentence is the headline. "
        "the second sentence and rest of body should be dropped. "
        "and a third sentence."
    )
    b = _mk(content, retention_class=RETENTION_SNAPSHOT)
    cb = compress_for_retrieval(b, locked=False)
    assert cb.strategy == STRATEGY_HEADLINE
    assert cb.rendered.startswith("the first sentence is the headline")
    assert cb.rendered.endswith("…")
    assert cb.rendered_tokens < _estimate_tokens(content)


def test_snapshot_locked_is_verbatim() -> None:
    content = "first sentence. second sentence. third."
    b = _mk(content, retention_class=RETENTION_SNAPSHOT, lock_level=LOCK_USER)
    cb = compress_for_retrieval(b, locked=True)
    assert cb.strategy == STRATEGY_VERBATIM
    assert cb.rendered == content


def test_transient_unlocked_is_stub() -> None:
    content = "ephemeral PR-window note about a temporary fix."
    b = _mk(content, bid="abcd1234", retention_class=RETENTION_TRANSIENT)
    cb = compress_for_retrieval(b, locked=False)
    assert cb.strategy == STRATEGY_STUB
    assert cb.rendered == "[stub: belief=abcd1234 class=transient]"


def test_transient_locked_is_verbatim() -> None:
    b = _mk("locked transient content.", retention_class=RETENTION_TRANSIENT)
    cb = compress_for_retrieval(b, locked=True)
    assert cb.strategy == STRATEGY_VERBATIM


# --- Headline edge cases -----------------------------------------------


def test_headline_short_no_boundary_renders_verbatim() -> None:
    content = "short content with no period in it"
    b = _mk(content, retention_class=RETENTION_SNAPSHOT)
    cb = compress_for_retrieval(b, locked=False)
    assert cb.strategy == STRATEGY_VERBATIM
    assert cb.rendered == content


def test_headline_long_no_boundary_hard_truncates() -> None:
    content = "word " * 100
    b = _mk(content, retention_class=RETENTION_SNAPSHOT)
    cb = compress_for_retrieval(b, locked=False)
    assert cb.strategy == STRATEGY_HEADLINE
    assert cb.rendered.endswith("…")
    assert len(cb.rendered) <= MAX_HEADLINE_CHARS + len("…")


def test_headline_split_on_newline_period() -> None:
    content = "first line headline.\nsecond line should be dropped."
    b = _mk(content, retention_class=RETENTION_SNAPSHOT)
    cb = compress_for_retrieval(b, locked=False)
    assert cb.strategy == STRATEGY_HEADLINE
    assert "second line" not in cb.rendered


def test_headline_preserves_balanced_code_fence_only_content() -> None:
    content = "```python\nx = 1\nprint(x)\n```"
    b = _mk(content, retention_class=RETENTION_SNAPSHOT)
    cb = compress_for_retrieval(b, locked=False)
    assert cb.strategy == STRATEGY_VERBATIM
    assert cb.rendered == content


def test_headline_does_not_split_on_period_inside_code_fence() -> None:
    content = (
        "first sentence with no boundary then code "
        "```\nfoo. bar.\n``` and more prose. final sentence."
    )
    b = _mk(content, retention_class=RETENTION_SNAPSHOT)
    cb = compress_for_retrieval(b, locked=False)
    assert cb.strategy == STRATEGY_HEADLINE
    assert "final sentence" not in cb.rendered
    assert cb.rendered.endswith("…")


# --- Invariants --------------------------------------------------------


def test_token_monotone_non_increasing() -> None:
    contents = [
        "short.",
        "first sentence. second sentence.",
        "word " * 200,
        "```\nprint(1)\n```",
        "transient note text.",
        "",
    ]
    for rc in (RETENTION_FACT, RETENTION_SNAPSHOT, RETENTION_TRANSIENT, RETENTION_UNKNOWN):
        for content in contents:
            for locked in (False, True):
                b = _mk(content, retention_class=rc)
                cb = compress_for_retrieval(b, locked=locked)
                source_cost = _estimate_tokens(content)
                assert cb.rendered_tokens <= source_cost, (
                    f"rendered_tokens={cb.rendered_tokens} > source={source_cost} "
                    f"for rc={rc} locked={locked} content={content!r}"
                )


def test_compress_is_deterministic() -> None:
    content = "first sentence. second sentence. third sentence here."
    for rc in (RETENTION_FACT, RETENTION_SNAPSHOT, RETENTION_TRANSIENT, RETENTION_UNKNOWN):
        for locked in (False, True):
            b = _mk(content, retention_class=rc)
            a = compress_for_retrieval(b, locked=locked)
            c = compress_for_retrieval(b, locked=locked)
            assert a == c


def test_total_function_returns_compressed_belief_for_every_input() -> None:
    for rc in (RETENTION_FACT, RETENTION_SNAPSHOT, RETENTION_TRANSIENT, RETENTION_UNKNOWN):
        b = _mk("any content", retention_class=rc)
        cb = compress_for_retrieval(b, locked=False)
        assert isinstance(cb, CompressedBelief)
        assert cb.belief is b


def test_unknown_retention_value_falls_back_to_verbatim() -> None:
    # Defensive: a belief whose retention_class is some out-of-band string
    # (e.g. a future class added by a downgrade-then-upgrade) should not
    # crash and should not silently compress. Verbatim is the safe fallback.
    b = _mk("content", retention_class="unrecognised_class")
    cb = compress_for_retrieval(b, locked=False)
    assert cb.strategy == STRATEGY_VERBATIM


# --- Stub format ------------------------------------------------------


def test_stub_format_is_grep_friendly() -> None:
    # Use content longer than the stub marker so the stub strategy fires.
    payload = "this is a long-enough transient note " * 4
    b = _mk(payload, bid="xyz789", retention_class=RETENTION_TRANSIENT)
    cb = compress_for_retrieval(b, locked=False)
    assert cb.strategy == STRATEGY_STUB
    assert "[stub: belief=xyz789" in cb.rendered
    assert "class=transient" in cb.rendered


def test_stub_falls_back_to_verbatim_when_source_smaller() -> None:
    # Pathological case: a transient belief whose content is shorter than
    # the stub marker (~38 chars). Compressor must not expand cost.
    b = _mk("x", bid="i", retention_class=RETENTION_TRANSIENT)
    cb = compress_for_retrieval(b, locked=False)
    assert cb.rendered_tokens <= _estimate_tokens("x")
    assert cb.strategy == STRATEGY_VERBATIM
