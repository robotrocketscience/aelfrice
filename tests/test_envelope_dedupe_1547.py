"""A belief shown in the session-start sub-block is not re-shown below it (#1547).

`_format_hits_with_session_start` builds one `<aelfrice-memory>` envelope out
of two halves: the embedded session-start sub-block (its `<locked>` and
`<core>` sections) and the per-turn retrieval hits. Both halves are rendered
by that one call, from the same frame, and nothing stopped a belief appearing
in both.

Measured on 1,127 live `UserPromptSubmit` rows before the fix: 1,589 redundant
`<belief>` elements, 164 of 1,124 rows (14.6%) carrying at least one, and the
pairings concentrated in `<core>` vs per-turn (829) and `<locked>` vs per-turn
(740). 729,527 of 17,179,106 characters, 4.25%, were the duplicate copies.

Two things these tests are careful about, because both are easy to get wrong:

**The id pattern must not assume hex.** Live stores carry 16-character hex ids
and 26-character ULIDs. Four independent readers of this code wrote a hex-only
regex and silently dropped the ULIDs, which are attached to the *largest*
elements in the corpus. `test_a_ulid_id_dedupes_too` is that guard.

**A store with nothing to dedupe must be byte-identical.** This change is a
reduction, and the thing held neutral is the no-duplicate case. A test that
only asserts "bytes went down" passes for a formatter that drops beliefs at
random.
"""
from __future__ import annotations

import re

import pytest

from aelfrice.hook import (
    _format_hits,
    _format_hits_with_session_start,
    _ids_rendered_verbatim_in,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)

_HEX_ID = "0123456789abcdef"
_ULID_ID = "01KS5RN4S24D5NRJ47TFSAT31C"


def _mk(bid: str, content: str = "a belief with enough text to be worth a line",
        *, lock_level: str = LOCK_NONE) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=5.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at="2026-09-01T00:00:00Z" if lock_level == LOCK_USER else None,
        created_at="2026-09-01T00:00:00Z",
        last_retrieved_at=None,
        lock_tier="frozen",
        corroboration_count=2,
    )


def _sub_block(*ids: str, shape: str = "locked") -> str:
    """A session-start sub-block rendering `ids`, in one of its two shapes."""
    if shape == "locked":
        body = "\n".join(
            f'<belief id="{i}" lock="user">locked content for {i}</belief>'
            for i in ids
        )
        return f"<session-start>\n<locked>\n{body}\n</locked>\n</session-start>"
    body = "\n".join(
        f'<belief id="{i}" corr="2" posterior="0.833">core content for {i}'
        f"</belief>"
        for i in ids
    )
    return f"<session-start>\n<core>\n{body}\n</core>\n</session-start>"


def _belief_ids(block: str) -> list[str]:
    return re.findall(r'<belief\s+id="([^"]+)"', block)


# ---------------------------------------------------------------------------
# The id scan
# ---------------------------------------------------------------------------


def test_the_scan_finds_both_shipped_id_forms() -> None:
    block = (
        f'<belief id="{_HEX_ID}" lock="user">x</belief>\n'
        f'<belief id="{_ULID_ID}" corr="2" posterior="0.9">y</belief>'
    )
    assert _ids_rendered_verbatim_in(block) == {_HEX_ID, _ULID_ID}


def test_the_scan_finds_all_three_element_shapes() -> None:
    """Per-turn, `<locked>` and `<core>` elements differ after the id."""
    block = "\n".join([
        f'<belief id="aaa" lock="none">per-turn</belief>',
        f'<belief id="bbb" lock="user">locked</belief>',
        f'<belief id="ccc" corr="7" posterior="0.5">core</belief>',
    ])
    assert _ids_rendered_verbatim_in(block) == {"aaa", "bbb", "ccc"}


def test_an_empty_block_needs_no_special_case() -> None:
    assert _ids_rendered_verbatim_in("") == frozenset()


def test_a_manifest_pointer_is_not_a_verbatim_render() -> None:
    """`seen <id>: "topic"` lines must not count as already rendered.

    They are the *output* of this mechanism. Counting them as input would
    make the predicate self-satisfying on a second pass.
    """
    assert _ids_rendered_verbatim_in('seen abc123: "the topic"') == frozenset()


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["locked", "core"])
def test_a_belief_shown_in_the_sub_block_is_not_shown_again(shape: str) -> None:
    dup = _mk(_HEX_ID, "content that would otherwise be emitted twice")
    fresh = _mk("ffffffffffffffff", "content seen only in the per-turn pack")
    block = _format_hits_with_session_start(
        [dup, fresh], _sub_block(_HEX_ID, shape=shape)
    )
    ids = _belief_ids(block)
    assert ids.count(_HEX_ID) == 1, f"{_HEX_ID} rendered {ids.count(_HEX_ID)}x"
    assert ids.count("ffffffffffffffff") == 1


@pytest.mark.parametrize("shape", ["locked", "core"])
def test_the_duplicate_leaves_a_pointer_rather_than_vanishing(
    shape: str,
) -> None:
    """It must still be findable. A silent drop is a different behaviour."""
    dup = _mk(_HEX_ID, "content that would otherwise be emitted twice")
    block = _format_hits_with_session_start(
        [dup], _sub_block(_HEX_ID, shape=shape)
    )
    assert f"seen {_HEX_ID}" in block


def test_a_ulid_id_dedupes_too() -> None:
    """The regression a hex-only pattern would reintroduce."""
    dup = _mk(_ULID_ID, "a long document-shaped belief " * 20)
    block = _format_hits_with_session_start([dup], _sub_block(_ULID_ID))
    assert _belief_ids(block).count(_ULID_ID) == 1
    assert f"seen {_ULID_ID}" in block


def test_the_no_duplicate_case_is_byte_identical() -> None:
    """The neutrality contract. This is what stops the fix being a cut."""
    hits = [_mk(f"{i:016d}", f"belief number {i} with some content")
            for i in range(6)]
    sub = _sub_block("cccccccccccccccc")  # shares nothing with the hits
    before = _format_hits(hits)
    after = _format_hits_with_session_start(hits, sub)
    # The sub-block is additive; everything else must match exactly.
    assert sub in after
    assert after.replace(sub + "\n", "") == before


def test_dedupe_shortens_the_envelope() -> None:
    long_text = "a long document-shaped belief, repeated to matter. " * 60
    hits = [_mk(_HEX_ID, long_text), _mk("ffffffffffffffff", long_text)]
    sub = _sub_block(_HEX_ID)
    deduped = _format_hits_with_session_start(hits, sub)
    # The control: the same render with nothing shared.
    not_deduped = _format_hits_with_session_start(hits, _sub_block("dddddddddddddddd"))
    assert len(deduped) < len(not_deduped)
    saved = len(not_deduped) - len(deduped)
    assert saved > len(long_text) * 0.8, saved


def test_every_hit_duplicated_still_renders_the_sub_block() -> None:
    """Degenerate case: nothing survives verbatim in the per-turn half."""
    hits = [_mk(_HEX_ID), _mk(_ULID_ID)]
    sub = _sub_block(_HEX_ID, _ULID_ID)
    block = _format_hits_with_session_start(hits, sub)
    assert sub in block
    assert f"seen {_HEX_ID}" in block and f"seen {_ULID_ID}" in block
    for bid in (_HEX_ID, _ULID_ID):
        assert _belief_ids(block).count(bid) == 1


def test_an_explicit_already_rendered_set_still_applies() -> None:
    """The #1382 ledger set must not be overwritten by the new union."""
    ledger_id = "1111111111111111"
    hits = [_mk(ledger_id), _mk(_HEX_ID)]
    block = _format_hits_with_session_start(
        hits, _sub_block(_HEX_ID), already_rendered=frozenset({ledger_id}),
    )
    assert f"seen {ledger_id}" in block
    assert f"seen {_HEX_ID}" in block
    assert _belief_ids(block).count(ledger_id) == 0


def test_no_sub_block_leaves_the_plain_formatter_untouched() -> None:
    hits = [_mk(f"{i:016d}") for i in range(3)]
    assert _format_hits_with_session_start(hits, "") == _format_hits(hits)
