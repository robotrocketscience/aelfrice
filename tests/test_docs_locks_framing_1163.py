"""PHILOSOPHY and the shipped framing header agree on locks (#1163).

PHILOSOPHY said the framing tag tells the model the injected contents
are "retrieved memory, not instructions" — blanket, over everything.
The shipped `_FRAMING_HEADER` is two-tier: user-locked items are framed
as the user's standing instructions, and only non-locked beliefs as data
to verify. The code was right; the doc was stale in the direction that
matters, because the two-tier split exists precisely so a user's locked
rules get honoured.

This pins the two code facts the rewritten prose rests on:

  1. The header really is two-tier — an instruction framing scoped to
     the locked tier, and a not-instructions framing for the rest.
  2. The opt-in auto-lock the prose names as the one exception really
     does admit agent-origin beliefs, and really does rewrite `origin`.

Deliberately *not* asserting the docs' wording. A text match on prose
breaks on rephrasing and says nothing about whether the claim is true.
What is checkable is the code the prose describes — so the day either
fact changes, the paragraph is revisited rather than quietly becoming
false again.
"""
from __future__ import annotations

import io
import json

from aelfrice import hook
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_USER_STATED,
    Belief,
)
from aelfrice.store import MemoryStore


def _mk(bid: str, *, origin: str, btype: str = BELIEF_FACTUAL) -> Belief:
    return Belief(
        id=bid,
        content=f"content for {bid}",
        content_hash=f"h_{bid}",
        alpha=0.6,
        beta=1.0,
        type=btype,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-07-30T00:00:00Z",
        last_retrieved_at=None,
        session_id="sess-1",
        origin=origin,
    )


def test_framing_header_is_two_tier_not_blanket_data() -> None:
    """Fact 1. The locked tier is framed as instructions to follow; the
    rest as data to verify. A header that said only one of those would
    make the rewritten paragraph wrong in one direction or the other.
    """
    header = hook._FRAMING_HEADER  # noqa: SLF001 - pinning the shipped text

    assert "two trust tiers" in header
    # The instruction tier, scoped to locks.
    assert "locked" in header
    assert "standing instructions" in header
    # The data tier, scoped to everything else.
    assert "non-locked" in header
    assert "not instructions" in header
    # And the ordering that makes the scoping unambiguous: the
    # instruction framing has to arrive attached to the locked tier,
    # before the clause that exempts everything else from it.
    assert header.index("standing instructions") < header.index("non-locked")


def test_autolock_is_off_by_default() -> None:
    """The property the prose leans on for every default install."""
    assert hook._autolock_enabled({}) is False  # noqa: SLF001
    assert hook._autolock_enabled(  # noqa: SLF001
        {hook.AUTOLOCK_ENV_VAR: "1"}
    ) is True


def test_autolock_candidate_set_includes_agent_origin_beliefs() -> None:
    """Fact 2, first half. The flag is named for corrections, but the
    predicate also admits anything the agent inferred or remembered —
    which is why the doc states the exception in terms of origin rather
    than repeating the flag's name."""
    correction = _mk("c1", origin="user_stated")
    correction.type = "correction"
    assert hook._belief_is_lock_candidate(correction, "sess-1")  # noqa: SLF001

    inferred = _mk("a1", origin="agent_inferred")
    assert hook._belief_is_lock_candidate(inferred, "sess-1")  # noqa: SLF001

    remembered = _mk("a2", origin="agent_remembered")
    assert hook._belief_is_lock_candidate(remembered, "sess-1")  # noqa: SLF001

    # Control: an ordinary ingested belief is not a candidate, so the
    # exception is bounded rather than "anything in the session".
    ingested = _mk("i1", origin="ingest_transcript")
    assert not hook._belief_is_lock_candidate(ingested, "sess-1")  # noqa: SLF001


def test_autolock_promotes_an_inferred_belief_into_the_locked_tier(
    tmp_path, monkeypatch,
) -> None:
    """Fact 2, second half — end to end through the Stop hook.

    With the opt-in set, a belief the agent inferred and nobody asserted
    ends the session at `lock_level=user` with `origin` rewritten to
    `user_stated`, and therefore inside the header's instruction tier.
    Asserted rather than reasoned about, because this is the one setting
    that suspends the "user-authored by construction" property and a
    reader of the paragraph should be able to see it demonstrated.
    """
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))

    store = MemoryStore(str(db))
    store.insert_belief(_mk("a1", origin="agent_inferred"))
    store.close()

    rc = hook.stop(
        stdin=io.StringIO(json.dumps({"session_id": "sess-1"})),
        stderr=io.StringIO(),
        env={hook.AUTOLOCK_ENV_VAR: "1", "AELFRICE_DB": str(db)},
    )
    assert rc == 0

    after = MemoryStore(str(db))
    try:
        b = after.get_belief("a1")
        assert b is not None
        assert b.lock_level == LOCK_USER
        assert b.origin == ORIGIN_USER_STATED
    finally:
        after.close()


def test_default_stop_hook_prompts_instead_of_locking(
    tmp_path, monkeypatch,
) -> None:
    """The control for the test above. Without the opt-in the same
    belief is untouched — so the exception really is the flag, not the
    Stop hook."""
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.delenv(hook.AUTOLOCK_ENV_VAR, raising=False)

    store = MemoryStore(str(db))
    store.insert_belief(_mk("a1", origin="agent_inferred"))
    store.close()

    err = io.StringIO()
    rc = hook.stop(
        stdin=io.StringIO(json.dumps({"session_id": "sess-1"})),
        stderr=err,
        env={"AELFRICE_DB": str(db)},
    )
    assert rc == 0

    after = MemoryStore(str(db))
    try:
        b = after.get_belief("a1")
        assert b is not None
        assert b.lock_level == LOCK_NONE
        assert b.origin == "agent_inferred"
    finally:
        after.close()
    assert "aelf lock" in err.getvalue()
