"""#1442 — the session-end block is bounded on count and on content length.

The Stop hook writes this block to stderr **once per assistant turn** and
re-emits the whole cumulative session list each time. Before this it was
bounded on neither axis: the worst session on the development store
rendered 6,427 entries and 3,448,428 bytes, and its longest single
candidate was 14,360 characters on one `aelf lock '<...>'` line.

Measured against that same distribution, so the constants are falsifiable
rather than tasteful: p50=10 candidates/session, p90=69, max=6,427;
content p50=86 chars, p95=605, max=14,360.
"""
from __future__ import annotations

import pytest

from aelfrice.hook import (
    STOP_PROMPT_MAX_CONTENT,
    STOP_PROMPT_MAX_ITEMS,
    _format_stop_prompt,
)
from aelfrice.models import (
    BELIEF_CORRECTION,
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
)

_SESSION = "s-1442"


def _belief(bid: str, content: str, *, created: str, type_: str = BELIEF_FACTUAL,
            origin: str = ORIGIN_USER_TRANSCRIPT) -> Belief:
    return Belief(
        id=bid, content=content, content_hash=f"h_{bid}", alpha=1.0, beta=1.0,
        type=type_, lock_level=LOCK_NONE, locked_at=None, created_at=created,
        last_retrieved_at=None, session_id=_SESSION, origin=origin,
    )


def test_the_shipped_limits_are_the_measured_ones() -> None:
    """Pins the values, not just the mechanism.

    Every behavioural test below reads the constants, so it would pass
    just as happily at `MAX_ITEMS = 10_000` — which is the bound not
    existing. These two asserts are what make the rest mean something.
    """
    assert STOP_PROMPT_MAX_ITEMS == 20
    assert STOP_PROMPT_MAX_CONTENT == 1000


def test_the_list_is_capped_and_says_how_many_it_withheld() -> None:
    """Silent truncation reads as "that was all of them", which is worse
    than the flood — the user acts on a list believing it complete.

    Falsifiable by dropping the slice (all 30 render) or by dropping the
    trailing line (the count disappears while the items stay hidden).
    """
    n = STOP_PROMPT_MAX_ITEMS + 10
    cands = [
        _belief(f"b{i:03d}", f"Always use rule {i}.", created=f"2026-08-09T00:{i:02d}:00Z")
        for i in range(n)
    ]
    out = _format_stop_prompt(cands)

    assert out.count("aelf lock ") == STOP_PROMPT_MAX_ITEMS
    # The header states the true total, not the shown count — otherwise
    # the cap would misreport how much is unlocked.
    assert f"Found {n} beliefs" in out
    assert "and 10 older beliefs from this session, not shown" in out


def test_the_cap_keeps_the_newest_not_an_arbitrary_slice() -> None:
    """The cap is only safe if it keeps the turn that just ended.

    `_collect_lock_candidates` walks `list_belief_ids()`, whose `ORDER BY
    id ASC` is content-hash order — ids are `sha256(source + NUL +
    text)[:16]`, so on the development store all 44,683 active rows sit
    at a different position there than in `created_at` order. A head-cap
    over that input would show a fixed arbitrary subset and hide the
    newest belief except by chance.

    So the ids here are deliberately ordered *against* recency: `b000` is
    newest. Falsifiable by dropping the sort — the slice then keeps
    `b000..b019`, which are the twenty OLDEST, and the newest-present
    assertion fails.
    """
    n = STOP_PROMPT_MAX_ITEMS + 5
    cands = [
        _belief(f"b{i:03d}", f"Always use rule {i}.",
                created=f"2026-08-09T00:{(n - i):02d}:00Z")
        for i in range(n)
    ]
    out = _format_stop_prompt(cands)

    assert "Always use rule 0." in out, "the newest belief was dropped by the cap"
    assert "Always use rule 24." not in out, "the oldest belief survived the cap"


def test_ties_on_created_at_are_broken_deterministically() -> None:
    """`created_at` ties are the norm on this store, so sorting on it
    alone is not a total order and the cap would be nondeterministic —
    against a stated invariant of the project.

    Falsifiable by dropping `b.id` from the sort key: which twenty
    survive then depends on the input list's order, so the two calls
    below disagree.
    """
    same = "2026-08-09T00:00:00Z"
    cands = [
        _belief(f"b{i:03d}", f"Always use rule {i}.", created=same)
        for i in range(STOP_PROMPT_MAX_ITEMS + 5)
    ]
    first = _format_stop_prompt(cands)
    second = _format_stop_prompt(list(reversed(cands)))
    assert first == second


def test_an_overlong_belief_is_listed_but_gets_no_pasteable_command() -> None:
    """`aelf lock` takes the statement *text* and has no id form, so a
    truncated command would lock text the user never wrote — silently,
    and as user-asserted ground truth. Withholding the command is the
    only safe option.

    Falsifiable by rendering the command anyway: the full 2,000-char
    content then appears inside `aelf lock '...'`.
    """
    long_content = "Always " + ("x" * (STOP_PROMPT_MAX_CONTENT + 1000))
    out = _format_stop_prompt([
        _belief("b_long", long_content, created="2026-08-09T00:00:00Z")
    ])

    assert "b_long" in out, "an overlong belief must still be listed"
    assert f"aelf lock '{long_content}'" not in out
    assert "too long to paste as a command" in out
    # The 120-char snippet still shows, so the user can tell what it is.
    assert "Always xxx" in out


def test_a_belief_at_the_limit_still_renders_its_command() -> None:
    """Boundary control for the test above. Without it the guard could be
    `len(content) > 0` — withholding every command — and the suite stays
    green.
    """
    at_limit = "Always " + ("x" * (STOP_PROMPT_MAX_CONTENT - len("Always ")))
    assert len(at_limit) == STOP_PROMPT_MAX_CONTENT
    out = _format_stop_prompt([
        _belief("bfit", at_limit, created="2026-08-09T00:00:00Z")
    ])
    assert f"aelf lock '{at_limit}'" in out
    assert "too long to paste" not in out


def test_the_autolock_caveat_counts_the_whole_set_not_the_shown_slice() -> None:
    """`AELF_AUTOLOCK_CORRECTIONS=1` writes every correction-class
    candidate, including ones the cap withheld. A caveat computed over
    the visible slice would tell the user the flag covers everything
    listed when it also covers items they cannot see.

    Here all 25 are correction-class, so the flag covers the entire set
    and the "does not cover the rest" caveat must be absent — which it
    would not be if the comparison used the 20 shown against the 25
    total. Falsifiable by scoping `covered` to `shown`.
    """
    cands = [
        _belief(f"c{i:03d}", f"Actually it is {i}, not {i + 1}.",
                created=f"2026-08-09T00:{i:02d}:00Z", type_=BELIEF_CORRECTION)
        for i in range(STOP_PROMPT_MAX_ITEMS + 5)
    ]
    out = _format_stop_prompt(cands)
    assert "AELF_AUTOLOCK_CORRECTIONS=1." in out
    assert "does not cover the rest" not in out


@pytest.mark.parametrize("n", [1, STOP_PROMPT_MAX_ITEMS])
def test_an_uncapped_list_gains_no_withheld_line(n: int) -> None:
    """The trailing line must not appear when nothing was withheld —
    including exactly at the limit, the off-by-one the cap invites.
    """
    cands = [
        _belief(f"b{i:03d}", f"Always use rule {i}.",
                created=f"2026-08-09T00:{i:02d}:00Z")
        for i in range(n)
    ]
    out = _format_stop_prompt(cands)
    assert "not shown" not in out
    assert out.count("aelf lock ") == n
