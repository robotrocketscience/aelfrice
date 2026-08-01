"""#1286 two-tier render bound — the parse must track the real renderer.

The measurement in `benchmarks/two_tier_render_bound.py` is a replay: it
reads blocks the hook already emitted and re-renders them under a proposed
rule. Its one silent failure mode is the parse drifting from
`hook._split_belief_lines` — a changed attribute order or a new attribute
makes `BELIEF_RE` match nothing, every block parses as zero beliefs, and the
script reports a 0.00% saving that looks exactly like a real null. The
verdict recorded on #1286 rests on that not happening, so it is pinned here
rather than assumed.

Every assertion is *distinguishing*: it separates the rule from the identity
render, which is the failure a count-only test would pass through.
"""

from __future__ import annotations

from benchmarks.two_tier_render_bound import BELIEF_RE, rewrite_block
from aelfrice.hook import _split_belief_lines
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_SPECULATIVE,
    Belief,
)

# Comfortably past SHORT_TOKENS=60 at 4 chars/token, and carrying a sentence
# boundary so `_headline` cuts rather than hard-truncating.
LONG = "First sentence of the belief. " + ("filler words here " * 40)


def _mk(bid: str, content: str, lock_level: str = LOCK_NONE, **kw) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at="2026-07-31T00:00:00Z" if lock_level == LOCK_USER else None,
        created_at="2026-07-31T00:00:00Z",
        last_retrieved_at=None,
        **kw,
    )


def _render(hits: list[Belief]) -> str:
    lines, _manifest = _split_belief_lines(hits)
    return "\n".join(lines)


def test_the_regex_parses_what_the_renderer_emits() -> None:
    """The drift guard. Renders through the production function and asserts
    the parse recovers every belief, its id and its tier — so a change to
    the render that this regex cannot read fails here instead of silently
    zeroing the measurement."""
    hits = [
        _mk("a" * 16, "plain non-locked belief."),
        _mk("b" * 16, "a user lock.", lock_level=LOCK_USER),
        _mk("c" * 16, "a phantom.", origin=ORIGIN_SPECULATIVE),
    ]
    found = BELIEF_RE.findall(_render(hits))
    assert [m[0] for m in found] == [h.id for h in hits]
    assert [m[1] for m in found] == ["none", "user", "none"]
    # The speculative marker is an optional group; it must be captured
    # rather than breaking the match, or phantoms would vanish from the
    # eligible set and the ceiling would read low.
    assert found[2][2] == ' speculative="1"'


def test_content_survives_the_escape_round_trip() -> None:
    """Belief text with angle brackets is escaped at render time. The cost
    and the headline are computed from the unescaped text, so a broken
    inverse would misprice exactly the beliefs most likely to be long."""
    hits = [_mk("d" * 16, "before <tag> after. rest of it.")]
    (_bid, _lock, _spec, rendered), = BELIEF_RE.findall(_render(hits))
    assert "&lt;tag&gt;" in rendered


def test_a_long_non_locked_belief_is_headlined() -> None:
    block = _render([_mk(f"{i:016x}", LONG) for i in range(8)])
    out, counts = rewrite_block(block)
    # Ranks 0-4 are exempt by rank; 5, 6, 7 are long enough to headline.
    assert counts["headlined"] == 3
    assert counts["exempt_rank"] == 5
    assert 'truncated="1"' in out
    assert len(out) < len(block)


def test_a_user_lock_is_never_headlined_however_long() -> None:
    """The one exemption the proposal calls non-negotiable, and the reason
    the ceiling is 6.1% rather than the whole block."""
    block = _render(
        [_mk(f"{i:016x}", LONG, lock_level=LOCK_USER) for i in range(8)]
    )
    out, counts = rewrite_block(block)
    assert counts["exempt_lock"] == 8
    assert counts["headlined"] == 0
    assert out == block


def test_a_short_belief_is_left_verbatim() -> None:
    """Distinguishing against the long case above: same rank positions,
    same tier, different outcome — so the cost clause is doing work rather
    than the rank clause carrying every exemption."""
    block = _render([_mk(f"{i:016x}", "short one.") for i in range(8)])
    _out, counts = rewrite_block(block)
    assert counts["exempt_short"] == 3
    assert counts["headlined"] == 0


def test_the_rank_clause_is_reachable_when_it_is_not_shadowed() -> None:
    """`rank < K_verbatim` counted 0 on the live audit because locks lead
    every block and the lock clause fires first. That is a property of the
    data, not of the rule — pinned so a future reader does not conclude the
    clause is dead code."""
    block = _render([_mk(f"{i:016x}", LONG) for i in range(3)])
    _out, counts = rewrite_block(block)
    assert counts["exempt_rank"] == 3
    assert counts["exempt_lock"] == 0
