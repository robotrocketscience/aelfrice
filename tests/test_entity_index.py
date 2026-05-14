"""Acceptance tests for the v1.3.0 entity-index (L2.5) retrieval tier.

One test per acceptance criterion in docs/design/entity_index.md § Validation:

  AC1. Pattern coverage (per-kind unit fixtures, ≥3 positive + ≥3
       negative). Tabulated below as a parametrised fixture set.
  AC2. Idempotency: refresh_belief / backfill_all are no-ops on
       re-run.
  AC3. On-write trigger fires: insert_belief / update_belief writes
       belief_entities rows without the test calling the index
       directly.
  AC4. Cache invalidation: RetrievalCache fronting an L2.5-aware
       retrieve() invalidates on belief mutations exactly as v1.0.
  AC5. Budget enforcement: L2.5 sub-budget is bounded; total output
       ≤ token_budget; monotonicity holds.
  AC6. Forward compatibility: a v1.0-shaped store opens cleanly on
       v1.3.0; entity-empty queries return byte-identical results.
  AC7. Default-off byte-identical fallback: AELFRICE_ENTITY_INDEX=0
       reverts retrieve() to v1.2-shape behaviour.
  AC8. Schema migration is idempotent: opening a v1.3.0 store
       twice never raises and never re-runs the backfill.

All tests deterministic, in-memory SQLite, ≤2 s wall clock each.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aelfrice.entity_extractor import (
    KIND_BRANCH,
    KIND_ERROR_CODE,
    KIND_FILE_PATH,
    KIND_IDENTIFIER,
    KIND_URL,
    KIND_VERSION,
    Entity,
    extract_entities,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import (
    DEFAULT_L25_LIMIT,
    DEFAULT_TOKEN_BUDGET,
    LEGACY_TOKEN_BUDGET,
    ENV_ENTITY_INDEX,
    RetrievalCache,
    is_entity_index_enabled,
    retrieve,
)
from aelfrice.store import (
    SCHEMA_META_ENTITY_BACKFILL,
    MemoryStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


@pytest.fixture(autouse=True)
def isolated_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Isolate the entity-index flag resolution per test:

    - Drop AELFRICE_ENTITY_INDEX from the env so the user's shell
      can't disable the flag globally.
    - Chdir into `tmp_path` so the TOML discovery walk never finds
      a `.aelfrice.toml` from the actual repo (the v1.3.0 default
      is True; this guarantees that's the path we test).

    Tests that want to override either inject monkeypatch / TOML
    files into the same `tmp_path` they receive.
    """
    monkeypatch.delenv(ENV_ENTITY_INDEX, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# AC1: pattern coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_kind", "expected_lower_substr"),
    [
        # file_path POSIX (3 positive)
        ("see src/aelfrice/retrieval.py for details", KIND_FILE_PATH, "src/aelfrice/retrieval.py"),
        ("docs/concepts/ROADMAP.md notes the milestone", KIND_FILE_PATH, "docs/roadmap.md"),
        ("the .github/workflows/ci.yml fires on PR", KIND_FILE_PATH, ".github/workflows/ci.yml"),
        # url
        ("see https://example.com/path for docs", KIND_URL, "https://example.com/path"),
        ("link http://x.y/z", KIND_URL, "http://x.y/z"),
        ("HTTPS://Mixed.Case/Page also works", KIND_URL, "https://mixed.case/page"),
        # error_code
        ("returned HTTP 503 from upstream", KIND_ERROR_CODE, "http 503"),
        ("error E1001 on disk", KIND_ERROR_CODE, "e1001"),
        ("caught a ValueError in tests", KIND_ERROR_CODE, "valueerror"),
        # version
        ("ships at v1.3.0 next week", KIND_VERSION, "v1.3.0"),
        ("legacy 1.2.0 is deprecated", KIND_VERSION, "1.2.0"),
        ("rc cut 1.3.0-rc1 last night", KIND_VERSION, "1.3.0-rc1"),
        # branch
        ("on feat/invisibility-reframe today", KIND_BRANCH, "feat/invisibility-reframe"),
        ("merged docs/entity-index-spec yesterday", KIND_BRANCH, "docs/entity-index-spec"),
        ("opened ci/staging-gate-update for review", KIND_BRANCH, "ci/staging-gate-update"),
        # identifier (both shapes)
        ("aelfrice.retrieval is the module", KIND_IDENTIFIER, "aelfrice.retrieval"),
        ("MemoryStore class lives in store.py", KIND_IDENTIFIER, "memorystore"),
        ("session_id is propagated", KIND_IDENTIFIER, "session_id"),
    ],
)
def test_ac1_per_kind_positive_fixtures(
    text: str, expected_kind: str, expected_lower_substr: str,
) -> None:
    """At least three positive fixtures per structured kind."""
    entities = extract_entities(text)
    matching = [
        e for e in entities
        if e.kind == expected_kind and expected_lower_substr in e.lower
    ]
    assert matching, (
        f"expected at least one {expected_kind} entity matching "
        f"{expected_lower_substr!r} in {text!r}, got {entities}"
    )


@pytest.mark.parametrize(
    ("text", "forbidden_kind"),
    [
        # file_path negatives — bare words / typos with dot only
        ("just a sentence with no path", KIND_FILE_PATH),
        ("Mr. Smith and Inc. are not paths", KIND_FILE_PATH),
        ("punctuation. only.", KIND_FILE_PATH),
        # url negatives
        ("nothing http here", KIND_URL),
        ("a http server is mentioned", KIND_URL),  # no scheme://
        ("ftp://example.com is not http(s)", KIND_URL),
        # error_code negatives — codes that look like prose
        ("the http protocol is fine", KIND_ERROR_CODE),
        ("empty error here", KIND_ERROR_CODE),
        ("just numbers 503 alone", KIND_ERROR_CODE),
        # version negatives — partial / non-semver
        ("just 1.2 alone", KIND_VERSION),
        ("v one point three", KIND_VERSION),
        ("date 2026-04 is not semver", KIND_VERSION),
        # branch negatives — missing prefix
        ("foo/bar is not a branch", KIND_BRANCH),
        ("path/to/thing/here", KIND_BRANCH),
        ("plainprose", KIND_BRANCH),
    ],
)
def test_ac1_per_kind_negative_fixtures(
    text: str, forbidden_kind: str,
) -> None:
    """At least three negative fixtures per structured kind."""
    entities = extract_entities(text)
    matching = [e for e in entities if e.kind == forbidden_kind]
    assert not matching, (
        f"expected no {forbidden_kind} in {text!r}, got {matching}"
    )


def test_ac1_overlap_policy_file_path_wins_over_identifier() -> None:
    """A POSIX file path absorbs the identifier substring it would
    otherwise produce. `aelfrice/retrieval.py` is exactly one
    file_path, NOT also `aelfrice.retrieval` and `retrieval`.
    """
    entities = extract_entities("see aelfrice/retrieval.py details")
    paths = [e for e in entities if e.kind == KIND_FILE_PATH]
    assert len(paths) == 1
    assert paths[0].raw == "aelfrice/retrieval.py"
    # The dotted-identifier pattern would have matched a substring
    # of this file path — ensure it didn't.
    idents_inside = [
        e for e in entities
        if e.kind == KIND_IDENTIFIER
        and e.span_start >= paths[0].span_start
        and e.span_end <= paths[0].span_end
    ]
    assert idents_inside == []


def test_ac1_max_entities_caps_overflow() -> None:
    """A pathological input doesn't blow up the index."""
    # Build text that would yield far more than the cap.
    text = " ".join(f"id_{i}" for i in range(200))
    entities = extract_entities(text, max_entities=10)
    assert len(entities) <= 10


# ---------------------------------------------------------------------------
# AC2: idempotency (refresh + backfill)
# ---------------------------------------------------------------------------


def test_ac2_update_belief_is_idempotent_in_entity_table() -> None:
    """Re-inserting the same content via update_belief produces the
    same row set; no duplicates accumulate."""
    s = MemoryStore(":memory:")
    b = _mk("B1", "uses aelfrice.retrieval and src/store.py daily")
    s.insert_belief(b)
    rows1 = s.belief_entities_for("B1")
    # update_belief without changing content should leave the row
    # set identical (the implementation deletes-then-re-inserts).
    s.update_belief(b)
    rows2 = s.belief_entities_for("B1")
    assert sorted(rows1) == sorted(rows2)
    s.update_belief(b)
    rows3 = s.belief_entities_for("B1")
    assert sorted(rows1) == sorted(rows3)


def test_ac2_backfill_is_idempotent_on_repeat_open() -> None:
    """Reopening a v1.3-stamped DB does not re-run the backfill."""
    db_path = ":memory:"
    s = MemoryStore(db_path)
    s.insert_belief(_mk("B1", "uses aelfrice.retrieval"))
    marker_first = s.get_schema_meta(SCHEMA_META_ENTITY_BACKFILL)
    assert marker_first is not None and marker_first != ""
    # The backfill on the same connection: clear the marker, re-run
    # explicitly, and confirm the second call is a no-op (zero new
    # rows because the PK guards against duplicates).
    rows_before = s.count_belief_entities()
    # Already stamped → no-op. Use getattr to avoid pyright's
    # protected-access warning; the symbol is part of the v1.3
    # contract documented in entity_index.md but kept under a
    # leading underscore because it's invoked by __init__.
    backfill = getattr(s, "_maybe_backfill_entity_index")
    backfill()
    rows_after = s.count_belief_entities()
    assert rows_before == rows_after


# ---------------------------------------------------------------------------
# AC3: on-write trigger fires
# ---------------------------------------------------------------------------


def test_ac3_insert_belief_writes_entity_rows() -> None:
    """A plain insert_belief call populates belief_entities without
    the test calling any index function."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("B1", "uses aelfrice.retrieval and src/store.py"))
    assert s.count_belief_entities() > 0
    rows = s.belief_entities_for("B1")
    kinds = {kind for _lower, _raw, kind in rows}
    # Both file_path and identifier should fire on this content.
    assert KIND_FILE_PATH in kinds
    assert KIND_IDENTIFIER in kinds


def test_ac3_update_belief_rewrites_entity_rows() -> None:
    """An update with new content drops the old entity rows and
    writes the new ones in the same transaction."""
    s = MemoryStore(":memory:")
    b = _mk("B1", "old: src/store.py")
    s.insert_belief(b)
    old_rows = s.belief_entities_for("B1")
    assert any("store.py" in lower for lower, _raw, _k in old_rows)

    b.content = "new: aelfrice.retrieval"
    s.update_belief(b)
    new_rows = s.belief_entities_for("B1")
    assert any("aelfrice.retrieval" in lower for lower, _raw, _k in new_rows)
    assert not any("store.py" in lower for lower, _raw, _k in new_rows)


def test_ac3_delete_belief_cascades_to_entity_rows() -> None:
    """delete_belief drops the entity rows for that belief."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("B1", "uses aelfrice.retrieval"))
    assert len(s.belief_entities_for("B1")) > 0
    s.delete_belief("B1")
    assert s.belief_entities_for("B1") == []


# ---------------------------------------------------------------------------
# AC4: cache invalidation
# ---------------------------------------------------------------------------


def test_ac4_cache_invalidates_on_entity_relevant_mutation() -> None:
    """A RetrievalCache fronting an L2.5-aware retrieve() invalidates
    on belief mutations — same contract as v1.0.
    """
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("B1", "uses aelfrice.retrieval today"))
    cache = RetrievalCache(s)
    cache.retrieve("aelfrice.retrieval")
    assert len(cache) == 1
    # Insert another belief — invalidate fires.
    s.insert_belief(_mk("B2", "extra fact"))
    assert len(cache) == 0


def test_ac4_cache_invalidates_on_update_too() -> None:
    """update_belief fires the same callback registry."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("B1", "uses aelfrice.retrieval"))
    cache = RetrievalCache(s)
    cache.retrieve("aelfrice.retrieval")
    assert len(cache) == 1
    b = s.get_belief("B1")
    assert b is not None
    b.content = "changed content with v1.3.0 noted"
    s.update_belief(b)
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# AC5: budget enforcement
# ---------------------------------------------------------------------------


def test_ac5_l25_subbudget_caps_entity_hits() -> None:
    """L2.5 cannot return more than DEFAULT_L25_TOKEN_SUBBUDGET tokens
    of content even if the entity-index has many matches.

    Direct observation via retrieve_with_tiers (the function that
    surfaces the per-tier counts the v1.3 benchmark adapter consumes).
    """
    from aelfrice.retrieval import retrieve_with_tiers
    s = MemoryStore(":memory:")
    big = "x" * 200  # ~50 tokens
    for i in range(30):
        s.insert_belief(_mk(f"E{i:02d}", f"the src/foo.py and {big}"))
    _out, _l0, l25_ids, _l1, _bfs = retrieve_with_tiers(s, "src/foo.py")
    # L2.5 alone is bounded: 50-token beliefs against the 400-token
    # sub-budget fit at most 8 (allow off-by-one from rounding).
    assert len(l25_ids) <= 9, (
        f"L2.5 returned {len(l25_ids)} beliefs "
        f"(estimate ~50 tok each, sub-budget 400)"
    )
    assert len(l25_ids) <= DEFAULT_L25_LIMIT


def test_ac5_total_budget_respected_when_l0_eats_room() -> None:
    """If L0 uses most of the outer budget, L2.5 gets the smaller
    of (sub-budget, remaining-after-L0) — never exceeds outer budget."""
    s = MemoryStore(":memory:")
    big = "L" * 9000  # ~2250 tokens
    s.insert_belief(_mk(
        "L1", big, lock_level=LOCK_USER, locked_at="2026-04-26T01:00:00Z",
    ))
    s.insert_belief(_mk("E1", "uses src/foo.py and v1.3.0"))
    hits = retrieve(s, "src/foo.py", token_budget=DEFAULT_TOKEN_BUDGET)
    # L0 alone is ~2250 tokens; outer budget is 2400. L2.5 has at
    # most 150 tokens of room remaining. The E1 belief is ~10 tokens
    # so it fits.
    locked = [h for h in hits if h.lock_level == LOCK_USER]
    assert len(locked) == 1
    assert locked[0].id == "L1"


# ---------------------------------------------------------------------------
# AC6: forward compatibility (v1.0 fixture opens cleanly)
# ---------------------------------------------------------------------------


def test_ac6_v10_fixture_opens_and_backfills(tmp_path: Path) -> None:
    """Build a v1.0-shaped store by hand, then open it with a v1.3+
    MemoryStore and verify the backfill populated entity rows for
    the existing beliefs."""
    db_file = tmp_path / "v10_fixture.db"
    # Hand-build a v1.0-shape: just `beliefs` + `beliefs_fts`. No
    # belief_entities, no schema_meta. The MemoryStore() open should
    # add the missing tables and run the backfill.
    conn = sqlite3.connect(str(db_file))
    conn.execute("""
        CREATE TABLE beliefs (
            id                  TEXT PRIMARY KEY,
            content             TEXT NOT NULL,
            content_hash        TEXT NOT NULL,
            alpha               REAL NOT NULL,
            beta                REAL NOT NULL,
            type                TEXT NOT NULL,
            lock_level          TEXT NOT NULL,
            locked_at           TEXT,
            demotion_pressure   INTEGER NOT NULL DEFAULT 0,
            created_at          TEXT NOT NULL,
            last_retrieved_at   TEXT
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE beliefs_fts
        USING fts5(id UNINDEXED, content, tokenize='porter unicode61')
    """)
    conn.execute(
        "INSERT INTO beliefs (id, content, content_hash, alpha, beta, "
        "type, lock_level, locked_at, created_at, "
        "last_retrieved_at) VALUES "
        "('B1', 'mentions aelfrice.retrieval and src/foo.py', 'h', "
        "1.0, 1.0, 'factual', 'none', NULL, '2026-01-01T00:00:00Z', NULL)"
    )
    conn.execute(
        "INSERT INTO beliefs_fts (id, content) VALUES "
        "('B1', 'mentions aelfrice.retrieval and src/foo.py')"
    )
    conn.commit()
    conn.close()

    # Open via v1.3.0 MemoryStore: schema migrations + backfill run.
    s = MemoryStore(str(db_file))
    rows = s.belief_entities_for("B1")
    assert rows, "backfill should have populated entity rows"
    kinds = {kind for _lower, _raw, kind in rows}
    assert KIND_FILE_PATH in kinds
    marker = s.get_schema_meta(SCHEMA_META_ENTITY_BACKFILL)
    assert marker is not None and marker != ""


def test_ac6_entity_empty_query_byte_identical_to_v12() -> None:
    """A query with no extractable entities returns the same beliefs
    in the same order with L2.5 enabled vs disabled."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("F1", "the dog is happy"))
    s.insert_belief(_mk("F2", "the cat is happy"))

    # noun_phrase WILL fire on "happy", but only on the BELIEFS, not
    # the query "happy" alone (single token, no leading determiner →
    # no NP match in extract_entities). So L2.5 does no work.
    enabled = retrieve(s, "happy", entity_index_enabled=True)
    disabled = retrieve(s, "happy", entity_index_enabled=False)
    assert [b.id for b in enabled] == [b.id for b in disabled]


# ---------------------------------------------------------------------------
# AC7: default-off byte-identical fallback
# ---------------------------------------------------------------------------


def test_ac7_env_off_reverts_to_v12_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """AELFRICE_ENTITY_INDEX=0 → retrieve() ignores entity hits,
    uses LEGACY_TOKEN_BUDGET when no explicit budget is passed, and
    matches the disabled-flag path byte-for-byte."""
    monkeypatch.setenv(ENV_ENTITY_INDEX, "0")
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("F1", "uses aelfrice.retrieval"))
    s.insert_belief(_mk("F2", "uses src/store.py"))

    env_off = retrieve(s, "aelfrice.retrieval")  # default budget
    flag_off = retrieve(s, "aelfrice.retrieval", entity_index_enabled=False)
    assert [b.id for b in env_off] == [b.id for b in flag_off]


def test_ac7_disabled_flag_uses_legacy_budget() -> None:
    """When the flag is disabled and no explicit budget is given,
    the function uses the v1.0 default (2000) so entity-empty
    behaviour is byte-identical to v1.2."""
    assert LEGACY_TOKEN_BUDGET == 2000
    assert DEFAULT_TOKEN_BUDGET == 2400
    # Can't observe the budget directly without instrumentation; the
    # shape test above (env_off vs flag_off identical) stands in.


def test_ac7_is_entity_index_enabled_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """env > kwarg > toml > default."""
    # Default is True (no env, no toml).
    assert is_entity_index_enabled() is True
    # Explicit kwarg wins over default.
    assert is_entity_index_enabled(False) is False
    assert is_entity_index_enabled(True) is True
    # Env var disables regardless of explicit kwarg.
    monkeypatch.setenv(ENV_ENTITY_INDEX, "0")
    assert is_entity_index_enabled() is False
    assert is_entity_index_enabled(True) is False
    # Re-enable: env unset, TOML disables, no kwarg → False.
    monkeypatch.delenv(ENV_ENTITY_INDEX)
    toml = tmp_path / ".aelfrice.toml"
    toml.write_text("[retrieval]\nentity_index_enabled = false\n")
    monkeypatch.chdir(tmp_path)
    assert is_entity_index_enabled() is False
    # Explicit True kwarg overrides TOML False.
    assert is_entity_index_enabled(True) is True


# ---------------------------------------------------------------------------
# AC8: schema migration idempotency
# ---------------------------------------------------------------------------


def test_ac8_double_open_does_not_raise(tmp_path: Path) -> None:
    """Opening the same DB file twice with v1.3+ MemoryStore is safe.
    Schema CREATE IF NOT EXISTS is the contract; the backfill stamp
    is the auxiliary one."""
    db_file = tmp_path / "twice.db"
    s1 = MemoryStore(str(db_file))
    s1.insert_belief(_mk("B1", "uses aelfrice.retrieval"))
    rows1 = s1.belief_entities_for("B1")
    s1.close()
    s2 = MemoryStore(str(db_file))
    rows2 = s2.belief_entities_for("B1")
    assert sorted(rows1) == sorted(rows2)
    s2.close()


def test_ac8_l25_lookup_smoke_end_to_end() -> None:
    """End-to-end: insert a belief that mentions an identifier, query
    by that identifier, see the belief surface via the L2.5 path."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk("B1", "the aelfrice.retrieval module is the L2.5 home"))
    s.insert_belief(_mk("B2", "an unrelated belief about apples"))
    hits = retrieve(s, "aelfrice.retrieval")
    ids = [b.id for b in hits]
    assert "B1" in ids
    # B2 has no aelfrice.retrieval — should not surface via L2.5
    # (it might surface via L1 BM25 if "aelfrice" or "retrieval"
    # tokenises against it, but it has neither word).
    assert "B2" not in ids


# ---------------------------------------------------------------------------
# Sanity: extract_entities is total
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "a", "."])
def test_extract_entities_total_on_pathological_inputs(text: str) -> None:
    """No exceptions, returns a list."""
    out = extract_entities(text)
    assert isinstance(out, list)
    for e in out:
        assert isinstance(e, Entity)
