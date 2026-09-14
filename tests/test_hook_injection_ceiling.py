"""Bounds on the UserPromptSubmit block: per-belief cap and block ceiling."""
from __future__ import annotations

from aelfrice.hook import (
    BELIEF_CONTENT_CHAR_CAP,
    HOOK_BLOCK_TOKEN_CEILING,
    _audit_tokens_from_block,
    _cap_belief_content,
    enforce_block_ceiling,
    resolve_block_ceiling,
)


def _element(bid: str, chars: int) -> str:
    return f'<belief id="{bid}" lock="none">{"x" * chars}</belief>\n'


def test_short_content_is_returned_unchanged() -> None:
    content = "a" * (BELIEF_CONTENT_CHAR_CAP - 1)
    assert _cap_belief_content(content) == content


def test_oversized_content_is_capped() -> None:
    capped = _cap_belief_content("a" * 35_164)
    assert len(capped) < 35_164
    assert capped.endswith("[…truncated]")
    assert capped.startswith("a" * BELIEF_CONTENT_CHAR_CAP)


def test_block_under_ceiling_is_untouched() -> None:
    body = "<aelfrice-memory>\n" + _element("a" * 16, 100) + "</aelfrice-memory>"
    out, dropped = enforce_block_ceiling(body)
    assert dropped == 0
    assert out == body


def test_block_over_ceiling_drops_whole_elements_from_the_end() -> None:
    # Ten elements of ~1k tokens each: each one fits on its own, the block
    # does not. This is the shape the per-belief cap leaves behind.
    body = "<aelfrice-memory>\n" + "".join(
        _element(str(i) * 16, 4_000) for i in range(10)
    ) + "</aelfrice-memory>"
    assert _audit_tokens_from_block(body) > HOOK_BLOCK_TOKEN_CEILING
    out, dropped = enforce_block_ceiling(body)
    assert 0 < dropped < 10
    assert _audit_tokens_from_block(out) <= HOOK_BLOCK_TOKEN_CEILING
    # Framing survives and no element is left half-removed.
    assert out.startswith("<aelfrice-memory>")
    assert out.endswith("</aelfrice-memory>")
    assert out.count("<belief ") == out.count("</belief>")
    # Dropping is from the end: the highest-ranked element is kept.
    assert '<belief id="' + "0" * 16 + '"' in out
    assert '<belief id="' + "9" * 16 + '"' not in out


def test_capped_beliefs_keep_a_block_of_many_hits_under_the_ceiling() -> None:
    # 40 hits, each at the per-belief cap, is the realistic worst case.
    body = "<aelfrice-memory>\n" + "".join(
        _element(str(i).zfill(16), BELIEF_CONTENT_CHAR_CAP) for i in range(40)
    ) + "</aelfrice-memory>"
    out, _ = enforce_block_ceiling(body)
    assert _audit_tokens_from_block(out) <= HOOK_BLOCK_TOKEN_CEILING


def test_ceiling_is_disabled_by_zero(monkeypatch) -> None:
    monkeypatch.setenv("AELFRICE_HOOK_BLOCK_CEILING", "0")
    assert resolve_block_ceiling() == 0
    body = "<aelfrice-memory>\n" + _element("a" * 16, 400_000) + "</aelfrice-memory>"
    out, dropped = enforce_block_ceiling(body)
    assert dropped == 0
    assert out == body


def test_non_integer_env_falls_back_to_the_default(monkeypatch) -> None:
    monkeypatch.setenv("AELFRICE_HOOK_BLOCK_CEILING", "not-a-number")
    assert resolve_block_ceiling() == HOOK_BLOCK_TOKEN_CEILING
