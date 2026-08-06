"""#1422: `mcp_remember` must stay readable after the writer is deleted.

The MCP surface is gone, so nothing produces `source_kind='mcp_remember'`
any more. The constants and their frozenset membership deliberately stay,
because **existing stores already hold rows carrying it** — anyone who used
`aelf_lock` before the surface broke in v2.0.1, and anyone whose store was
migrated forward from one.

What deleting the constants would actually cost, stated precisely rather
than as the worst case: `ingest_log.source_kind` carries no SQL CHECK, and
`_ingest_row_to_dict` does not validate, so a historical row still *reads*
either way — this is not, today, the #1161 unopenable-store class. The two
reachable consequences are that `record_ingest` (store.py, the sole consumer
of `INGEST_SOURCE_KINDS`) would reject the value, which breaks replaying any
historical row including the soak corpus below; and
`retention_class_for_source` would silently fall through its `.get` default,
reclassifying the row `unknown` instead of `fact`.

The durable risk is the one a test has to hold: nothing stops a *future*
reader from validating `source_kind` on the way in, and by then the rows are
already on disk. So the removal's rule is *stop producing, keep accepting*,
and this module is what enforces it — otherwise the constants look like dead
code to the next person cleaning up after the removal, and deleting them is
a one-line change with no test to stop it.

`tests/corpus/replay_soak/v0.1/mcp_remember_v0_1.jsonl` covers the same
contract end-to-end through the soak runner; these are the direct,
fast-failing assertions that name *why* the constants exist.
"""
from __future__ import annotations

from pathlib import Path

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
    RETENTION_FACT,
    retention_class_for_source,
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


def test_a_persisted_mcp_remember_row_survives_close_and_reopen(
    tmp_path: Path,
) -> None:
    """The end the constraint is actually about: the store stays usable.

    Seeds the row with raw SQL rather than `record_ingest`, because
    `record_ingest` validates against `INGEST_SOURCE_KINDS` — going through
    it would make this test pass by construction on exactly the change it
    exists to catch. A pre-removal release left the row on disk without
    re-validating it, and that is the state being reproduced.

    File-backed and reopened, because opening a store is a *write* — DDL and
    the migration sweep run on open, which is where a value the current code
    no longer recognises would actually get rejected or rewritten.
    """
    db = tmp_path / "legacy.db"
    store = MemoryStore(str(db))
    try:
        store._conn.execute(
            "INSERT INTO ingest_log (id, ts, source_kind, raw_text) "
            "VALUES (?, ?, ?, ?)",
            ("01LEGACYMCPREMEMBER00000000", _TS,
             INGEST_SOURCE_MCP_REMEMBER, "Commits on main are signed."),
        )
        store._conn.commit()
    finally:
        store.close()

    reopened = MemoryStore(str(db))
    try:
        row = reopened.get_ingest_log_entry("01LEGACYMCPREMEMBER00000000")
        assert row is not None, "reopening dropped the historical row"
        assert row["source_kind"] == "mcp_remember"
        assert row["raw_text"] == "Commits on main are signed."
    finally:
        reopened.close()


def test_deriving_from_the_legacy_source_still_classifies_it_as_fact() -> None:
    """`retention_class_for_source` is the other reachable consequence.

    Its `.get` default means dropping the mapping degrades silently to
    `unknown` rather than raising — so only an assertion on the value
    catches it.
    """
    assert retention_class_for_source(INGEST_SOURCE_MCP_REMEMBER) == (
        RETENTION_FACT
    )

    out = derive(DerivationInput(
        raw_text="Commits on main are signed.",
        source_kind=INGEST_SOURCE_MCP_REMEMBER,
        ts=_TS,
    ))
    assert out.belief is not None
    assert out.belief.retention_class == RETENTION_FACT
    assert out.belief.lock_level == LOCK_USER


@pytest.mark.parametrize(
    "constant",
    [INGEST_SOURCE_MCP_REMEMBER, CORROBORATION_SOURCE_MCP_REMEMBER],
)
def test_the_wire_string_is_unchanged(constant: str) -> None:
    """The literal is the on-disk value; renaming it orphans historical rows."""
    assert constant == "mcp_remember"
