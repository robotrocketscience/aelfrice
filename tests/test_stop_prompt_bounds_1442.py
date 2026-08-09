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

    Every behavioural test below reads the constants rather than
    hard-coding them, so raising a limit is largely invisible to them:
    at `MAX_CONTENT = 1_000_000` this is the *only* test that fails, and
    at `MAX_ITEMS = 10_000` only this and the collector-order test do.
    Without these two asserts the bound could be widened to
    non-existence with the suite green.
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


def test_the_collector_returns_newest_first_through_a_real_store() -> None:
    """The recency guarantee lives at the collector, because that is the
    only place with access to the key that discriminates.

    Every belief here shares one `created_at`, which is not a contrived
    fixture: `created_at` has 2,772 tie groups on this repo's store and
    the 6,427-belief session this whole bound is named for shares a single
    timestamp across all of them. Under a `(created_at, id)` sort the
    survivors would be chosen by content-hash order.

    **The ids are hash-shaped, and that is the whole fixture.** An earlier
    draft numbered them `a000, b001, … y024` — monotone with insertion —
    so `ORDER BY id DESC` and `ORDER BY rowid DESC` returned the identical
    sequence and swapping the shipped query for the former (a plausible
    "use the PRIMARY KEY index instead of a rowid scan" edit) left the
    **whole suite** green. Real ids are `sha256(source + NUL + text)[:16]`,
    so these are generated the same way and carry no temporal signal; the
    two adequacy asserts below fail if a future edit makes them monotone
    again.

    Falsifiable by pointing `_collect_lock_candidates` back at
    `list_belief_ids()`, by re-introducing a `(created_at, id)` sort in
    `_format_stop_prompt`, or by changing `list_belief_ids_newest_first`
    to `ORDER BY id DESC`: all three put an arbitrary belief at the head
    and drop the newest.
    """
    import hashlib
    import tempfile
    from pathlib import Path as _Path

    from aelfrice.hook import _collect_lock_candidates
    from aelfrice.store import MemoryStore

    same = "2026-08-09T00:00:00Z"
    n = STOP_PROMPT_MAX_ITEMS + 40
    ids = [
        hashlib.sha256(f"1442\x00Always use rule {i}.".encode()).hexdigest()[:16]
        for i in range(n)
    ]
    # Adequacy: id order must carry no insertion signal in either
    # direction, or `ORDER BY id` would pass by accident.
    assert sorted(ids) != ids, "fixture ids ascend with insertion order"
    assert sorted(ids, reverse=True) != ids[::-1], (
        "fixture ids descend with insertion order"
    )

    with tempfile.TemporaryDirectory() as td:
        db = _Path(td) / "m.db"
        s = MemoryStore(str(db))
        try:
            for i, bid in enumerate(ids):        # inserted oldest-first
                s.insert_belief(
                    _belief(bid, f"Always use rule {i}.", created=same)
                )
            got = _collect_lock_candidates(s, _SESSION)
        finally:
            s.close()

    assert len(got) == n
    # The last row inserted must come first, whatever its id sorts like.
    assert got[0].content == f"Always use rule {n - 1}."
    assert got[-1].content == "Always use rule 0."

    out = _format_stop_prompt(got)
    assert f"Always use rule {n - 1}." in out, (
        "the newest belief was dropped by the cap"
    )
    assert "Always use rule 0." not in out, "the oldest belief survived the cap"


def test_the_renderer_takes_the_head_and_does_not_re_sort() -> None:
    """`_format_stop_prompt` must preserve the caller's order.

    A re-sort here cannot be a recency guarantee — the only keys on a
    `Belief` are `created_at` (tied) and `id` (content-hash) — so it would
    silently override the collector's `rowid` order with an arbitrary one.
    Falsifiable by restoring `sorted(..., key=(created_at, id), reverse=True)`:
    `zzz` sorts first and displaces the intended head.
    """
    same = "2026-08-09T00:00:00Z"
    ordered = [
        _belief("aaa", "Always use rule NEWEST.", created=same),
        _belief("zzz", "Always use rule OLDEST.", created=same),
    ]
    out = _format_stop_prompt(ordered[:1] + ordered[1:])
    first = out.index("Always use rule NEWEST.")
    second = out.index("Always use rule OLDEST.")
    assert first < second, "the renderer re-sorted its input"


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


def test_the_pointer_offered_instead_of_the_command_actually_resolves(
    tmp_path, monkeypatch, capsys,
) -> None:
    """The withheld-command line is the user's only route to content up
    to 14,360 characters they are being asked to lock, so the command it
    names has to work on *this* population — an unlocked belief, looked
    up by id.

    `aelf search '<id>'` did not. Ids are not part of belief content and
    so are not in the FTS index, and `_cmd_search` is `retrieve()` plus a
    peer overlay; it appears to work only for *locked* beliefs, because
    the L0 lane emits those whatever the query, and a lock candidate is
    by construction never `lock_level=user`. The command is now taken
    from the rendered block verbatim and run, rather than asserted as a
    string, because a string assertion is exactly what let a dead
    pointer ship.
    """
    import re
    import shlex

    from aelfrice import cli
    from aelfrice.store import MemoryStore

    long_content = "Always " + ("x" * (STOP_PROMPT_MAX_CONTENT + 1000))
    bid = "deadbeefcafe0001"          # hash-shaped, and absent from the content
    b = _belief(bid, long_content, created="2026-08-09T00:00:00Z")

    db = tmp_path / "m.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    s = MemoryStore(str(db))
    try:
        s.insert_belief(b)
    finally:
        s.close()

    out = _format_stop_prompt([b])
    quoted = [m for m in re.findall(r"`([^`]+)`", out) if m.startswith("aelf ")]
    assert len(quoted) == 1, f"expected exactly one command, got {quoted!r}"
    argv = shlex.split(quoted[0])
    assert argv[0] == "aelf"

    capsys.readouterr()
    assert cli.main(argv[1:]) == 0, "the pointer's command failed"
    printed = capsys.readouterr().out
    assert bid in printed, "the pointer did not resolve the belief"
    assert long_content in printed, (
        "the pointer resolved but did not show the content the user is "
        "being asked to lock"
    )


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
    total. Falsifiable by comparing `len(covered)` against the shown
    count while leaving `covered` itself over the whole set.

    This case alone is **not** sufficient: scoping `covered` *and* the
    comparison to `shown` together is self-consistent and passes here,
    because an all-correction-class list agrees under either scoping.
    `test_the_autolock_caveat_survives_a_mixed_list_across_the_cap`
    is the arm that separates them.
    """
    cands = [
        _belief(f"c{i:03d}", f"Actually it is {i}, not {i + 1}.",
                created=f"2026-08-09T00:{i:02d}:00Z", type_=BELIEF_CORRECTION)
        for i in range(STOP_PROMPT_MAX_ITEMS + 5)
    ]
    out = _format_stop_prompt(cands)
    assert "AELF_AUTOLOCK_CORRECTIONS=1." in out
    assert "does not cover the rest" not in out


def test_the_autolock_caveat_survives_a_mixed_list_across_the_cap() -> None:
    """The class boundary and the cap boundary are put in different
    places, which is the only configuration that pins the scope.

    The shown slice is entirely correction-class and everything the cap
    withheld is not, so `AELF_AUTOLOCK_CORRECTIONS=1` covers 20 of 25
    candidates while covering 20 of the 20 the user can see. Computing
    `covered` over `shown` — the self-consistent half of the mutation,
    which the all-correction-class case above passes — drops the caveat
    and tells the user the flag covers a list of which five items it does
    not. The flag writes over the whole candidate set, cap or no cap, so
    the caveat has to be computed there too.
    """
    corrections = [
        _belief(f"c{i:03d}", f"Actually it is {i}, not {i + 1}.",
                created=f"2026-08-09T00:{i:02d}:00Z", type_=BELIEF_CORRECTION)
        for i in range(STOP_PROMPT_MAX_ITEMS)
    ]
    # Withheld by the cap and outside the flag's population: `factual` /
    # `user_transcript` is what production `derive()` types a #1315
    # directive as.
    directives = [
        _belief(f"d{i:03d}", f"Always use rule {i}.",
                created=f"2026-08-09T01:{i:02d}:00Z", type_=BELIEF_FACTUAL,
                origin=ORIGIN_USER_TRANSCRIPT)
        for i in range(5)
    ]
    out = _format_stop_prompt(corrections + directives)

    assert "and 5 older beliefs from this session, not shown" in out
    assert "does not cover the rest" in out, (
        "the caveat was computed over the shown slice, not the whole set"
    )


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
