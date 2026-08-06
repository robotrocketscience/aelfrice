"""#1422: `mcp_remember` must stay readable after the writer is deleted.

The MCP surface is gone, so nothing produces `source_kind='mcp_remember'`
any more. The constants and their frozenset membership deliberately stay,
because **existing stores already hold rows carrying it** — anyone who used
`aelf_lock` before the surface broke in v2.0.1, and anyone whose store was
migrated forward from one.

This is the #1161 failure class: a reader that rejects a `source_kind` it
wrote last release makes the store unopenable, and that is not recoverable
in the field. So the removal's rule is *stop producing, keep accepting*, and
this module is what enforces it — otherwise the constants look like dead
code to the next person cleaning up after the removal, and deleting them is
a one-line change with no test to stop it.

`tests/corpus/replay_soak/v0.1/mcp_remember_v0_1.jsonl` covers the same
contract end-to-end through the soak runner; these are the direct,
fast-failing assertions that name *why* the constants exist.
"""
from __future__ import annotations

import pytest

from aelfrice.derivation import DerivationInput, derive
from aelfrice.models import (
    CORROBORATION_SOURCE_MCP_REMEMBER,
    CORROBORATION_SOURCE_TYPES,
    CORROBORATION_SOURCES_USER_EXPLICIT,
    INGEST_SOURCE_KINDS,
    INGEST_SOURCE_MCP_REMEMBER,
    LOCK_USER,
    ORIGIN_USER_STATED,
)
from aelfrice.store import MemoryStore

_TS = "2026-05-01T00:00:00Z"


def test_mcp_remember_is_still_a_recognised_ingest_source() -> None:
    """Dropping it from `INGEST_SOURCE_KINDS` is the store-bricking change."""
    assert INGEST_SOURCE_MCP_REMEMBER == "mcp_remember"
    assert INGEST_SOURCE_MCP_REMEMBER in INGEST_SOURCE_KINDS


def test_mcp_remember_is_still_a_recognised_corroboration_source() -> None:
    """It must also stay in the user-explicit set, not just the general one.

    `CORROBORATION_SOURCES_USER_EXPLICIT` is what `insert_or_corroborate`
    consults to decide whether re-asserting a statement revives a *retired*
    belief (#1215). A historical `mcp_remember` corroboration was a person
    typing a statement, so demoting it to the non-explicit tier would
    silently change how an old store's tombstones behave.
    """
    assert CORROBORATION_SOURCE_MCP_REMEMBER in CORROBORATION_SOURCE_TYPES
    assert CORROBORATION_SOURCE_MCP_REMEMBER in CORROBORATION_SOURCES_USER_EXPLICIT


def test_derive_still_accepts_an_mcp_remember_row() -> None:
    """A historical row derives to the same locked belief it always did.

    Membership in a frozenset is necessary but not sufficient — the
    derivation branch that gives lock/remember rows their priors has to keep
    matching on it too.
    """
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


def test_a_store_holding_an_mcp_remember_row_still_opens_and_reads() -> None:
    """The end the constraint is actually about: the store stays usable.

    Writes a row the way a pre-removal release would have, then re-reads it.
    A reader that rejected the `source_kind` would fail here rather than in a
    user's terminal.
    """
    store = MemoryStore(":memory:")
    try:
        out = derive(DerivationInput(
            raw_text="Commits on main are signed.",
            source_kind=INGEST_SOURCE_MCP_REMEMBER,
            ts=_TS,
        ))
        assert out.belief is not None
        store.insert_belief(out.belief)

        read_back = store.get_belief(out.belief.id)
        assert read_back is not None
        assert read_back.content == "Commits on main are signed."
        assert read_back.lock_level == LOCK_USER
    finally:
        store.close()


@pytest.mark.parametrize(
    "constant",
    [INGEST_SOURCE_MCP_REMEMBER, CORROBORATION_SOURCE_MCP_REMEMBER],
)
def test_the_wire_string_is_unchanged(constant: str) -> None:
    """The literal is the on-disk value; renaming it orphans historical rows."""
    assert constant == "mcp_remember"
