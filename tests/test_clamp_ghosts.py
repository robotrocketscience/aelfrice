"""Tests for `aelfrice.clamp_ghosts` + the `aelf clamp-ghosts` CLI.

Verifies:
- The structural ghost-α predicate (no feedback_history, no
  belief_corroborations, α > threshold, lock_level='none').
- Dry-run is read-only.
- Apply clamps α and writes one negative-valence feedback_history row
  per belief.
- Idempotent: re-applying after a successful clamp finds zero rows
  because the EXISTS filter now excludes them.
- Locked beliefs are never touched.
- Beliefs with existing feedback_history or belief_corroborations
  are skipped.
- Reversibility: the clamped α can be restored from the
  negative-valence row's magnitude.
- CLI flag plumbing.
- Argument validation (threshold/target positivity, ordering).
"""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from aelfrice import classification_core as cc
from aelfrice import clamp_ghosts
from aelfrice.cli import build_parser
from aelfrice.classification_core import TYPE_PRIORS, get_source_adjusted_prior
from aelfrice.clamp_ghosts import (
    CLAMP_SOURCE,
    DEFAULT_THRESHOLD_ALPHA,
    DEFAULT_TARGET_ALPHA,
    USER_PRIOR_ORIGINS,
    _ELIGIBILITY_SQL,
    _GHOST_RECHECK_SQL,
    _GHOST_SELECT_SQL,
    _eligibility_params,
    clamp_ghost_alphas,
)
from aelfrice.models import BELIEF_FACTUAL, BELIEF_TYPES, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore


# -- belief / store helpers -------------------------------------------------

def _mk(bid: str, *, alpha: float = 1.0, beta: float = 1.0, lock: str = LOCK_NONE) -> Belief:
    return Belief(
        id=bid,
        content=f"belief content for {bid}",
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        locked_at=("2026-04-01T00:00:00Z" if lock != LOCK_NONE else None),
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
        session_id=None,
        origin="unknown",
        corroboration_count=0,
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(tmp_path / "test.db")
    yield s
    s.close()


def _alpha_of(store: MemoryStore, bid: str) -> float:
    b = store.get_belief(bid)
    assert b is not None
    return b.alpha


def _feedback_rows_for(store: MemoryStore, bid: str) -> list[tuple[float, str]]:
    cur = store._conn.execute(  # noqa: SLF001
        "SELECT valence, source FROM feedback_history WHERE belief_id = ?",
        (bid,),
    )
    return [(float(r["valence"]), str(r["source"])) for r in cur.fetchall()]


# -- structural predicate ---------------------------------------------------

def test_dry_run_does_not_mutate(store: MemoryStore) -> None:
    ghost = _mk("g1", alpha=10.0)
    store.insert_belief(ghost)

    result = clamp_ghost_alphas(store, dry_run=True)

    assert result.matched == 1
    assert result.clamped == 0
    assert result.skipped == 1
    assert result.dry_run is True
    assert _alpha_of(store, "g1") == 10.0
    assert _feedback_rows_for(store, "g1") == []


def test_apply_clamps_and_writes_audit_row(store: MemoryStore) -> None:
    ghost = _mk("g1", alpha=10.0)
    store.insert_belief(ghost)

    result = clamp_ghost_alphas(store, dry_run=False)

    assert result.matched == 1
    assert result.clamped == 1
    assert result.skipped == 0
    assert _alpha_of(store, "g1") == DEFAULT_TARGET_ALPHA  # 4.0

    rows = _feedback_rows_for(store, "g1")
    assert len(rows) == 1
    valence, source = rows[0]
    assert source == CLAMP_SOURCE
    # negative valence equal to the clamped magnitude (10 - 4)
    assert valence == pytest.approx(-(10.0 - DEFAULT_TARGET_ALPHA))


def test_idempotent_after_apply(store: MemoryStore) -> None:
    """Re-running after a successful clamp finds zero rows.

    The first apply writes feedback_history; the second invocation's
    EXISTS filter excludes the row.
    """
    store.insert_belief(_mk("g1", alpha=10.0))

    first = clamp_ghost_alphas(store, dry_run=False)
    assert first.clamped == 1

    second = clamp_ghost_alphas(store, dry_run=False)
    assert second.matched == 0
    assert second.clamped == 0


def test_skips_locked_beliefs(store: MemoryStore) -> None:
    """Locked beliefs are never clamped.

    Their α reflects an explicit user assertion (e.g. via aelf lock),
    not an unaudited write.
    """
    store.insert_belief(_mk("locked", alpha=100.0, lock=LOCK_USER))

    result = clamp_ghost_alphas(store, dry_run=False)

    assert result.matched == 0
    assert _alpha_of(store, "locked") == 100.0


def test_skips_beliefs_with_feedback_history(store: MemoryStore) -> None:
    """Beliefs with any prior feedback event are skipped — their α has
    a known audit trail."""
    store.insert_belief(_mk("audited", alpha=10.0))
    # Insert one feedback row manually to mark this belief as "explained"
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO feedback_history (belief_id, valence, source, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("audited", 0.1, "hook", "2026-04-26T00:00:01Z"),
    )
    store._conn.commit()  # noqa: SLF001

    result = clamp_ghost_alphas(store, dry_run=False)

    assert result.matched == 0
    assert _alpha_of(store, "audited") == 10.0


def test_skips_beliefs_with_corroborations(store: MemoryStore) -> None:
    """Beliefs with prior corroborations are skipped — their α is at
    least defensible against a multi-source ingest history."""
    store.insert_belief(_mk("corroborated", alpha=10.0))
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO belief_corroborations "
        "(belief_id, ingested_at, source_type, session_id, source_path_hash) "
        "VALUES (?, ?, ?, ?, ?)",
        ("corroborated", "2026-04-26T00:00:01Z", "transcript_ingest", None, None),
    )
    store._conn.commit()  # noqa: SLF001

    result = clamp_ghost_alphas(store, dry_run=False)

    assert result.matched == 0
    assert _alpha_of(store, "corroborated") == 10.0


def test_skips_beliefs_below_threshold(store: MemoryStore) -> None:
    """α at or below the threshold is not a ghost candidate."""
    store.insert_belief(_mk("just_under", alpha=4.0))  # == threshold
    store.insert_belief(_mk("comfortably_under", alpha=2.5))

    result = clamp_ghost_alphas(store, dry_run=False, threshold_alpha=4.0)

    assert result.matched == 0
    assert _alpha_of(store, "just_under") == 4.0
    assert _alpha_of(store, "comfortably_under") == 2.5


def test_reversibility_via_feedback_history(store: MemoryStore) -> None:
    """The clamp event's negative valence magnitude restores prior α."""
    store.insert_belief(_mk("g1", alpha=84.0))

    clamp_ghost_alphas(store, dry_run=False)
    assert _alpha_of(store, "g1") == DEFAULT_TARGET_ALPHA

    # Reverse the clamp: read the negative valence and add its magnitude back.
    cur = store._conn.execute(  # noqa: SLF001
        "SELECT valence FROM feedback_history "
        "WHERE belief_id = ? AND source = ?",
        ("g1", CLAMP_SOURCE),
    )
    rows = cur.fetchall()
    assert len(rows) == 1
    valence = float(rows[0]["valence"])
    store._conn.execute(  # noqa: SLF001
        "UPDATE beliefs SET alpha = alpha + ? WHERE id = ?",
        (-valence, "g1"),
    )
    store._conn.commit()  # noqa: SLF001

    assert _alpha_of(store, "g1") == 84.0  # restored exactly


def test_limit_caps_processed_rows(store: MemoryStore) -> None:
    for i in range(5):
        store.insert_belief(_mk(f"g{i}", alpha=10.0))

    result = clamp_ghost_alphas(store, dry_run=False, limit=2)

    assert result.matched == 2
    assert result.clamped == 2
    # The other three remain unclamped at α=10
    unclamped = sum(
        1 for i in range(5) if _alpha_of(store, f"g{i}") == 10.0
    )
    assert unclamped == 3


def test_sample_capped_at_10(store: MemoryStore) -> None:
    for i in range(15):
        store.insert_belief(_mk(f"g{i:02d}", alpha=10.0))

    result = clamp_ghost_alphas(store, dry_run=True)

    assert result.matched == 15
    assert len(result.sample) == 10


def test_validation_rejects_target_above_threshold(store: MemoryStore) -> None:
    """clamp_ghost_alphas raises if target > threshold (would no-op)."""
    with pytest.raises(ValueError, match="target_alpha"):
        clamp_ghost_alphas(store, threshold_alpha=4.0, target_alpha=8.0)


def test_validation_rejects_nonpositive_alphas(store: MemoryStore) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        clamp_ghost_alphas(store, threshold_alpha=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        clamp_ghost_alphas(store, target_alpha=-1.0)


# -- CLI surface ------------------------------------------------------------

@pytest.fixture
def cli_store_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    return db


def _run_cli(*args: str) -> tuple[int, str]:
    parser = build_parser()
    ns = parser.parse_args(["clamp-ghosts", *args])
    out = io.StringIO()
    code: int = ns.func(ns, out)  # type: ignore[attr-defined]
    return code, out.getvalue()


def test_clamp_ghosts_subcommand_registered() -> None:
    from aelfrice.cli import _known_cli_subcommands
    assert "clamp-ghosts" in _known_cli_subcommands()


def test_clamp_ghosts_cli_dry_run_default(cli_store_path: Path) -> None:
    s = MemoryStore(cli_store_path)
    s.insert_belief(_mk("g1", alpha=10.0))
    s.close()

    code, output = _run_cli()  # no flags → dry-run

    assert code == 0
    assert "matched=1" in output
    assert "clamped=0" in output
    assert "dry_run=True" in output
    assert "--apply" in output  # the "run with --apply" hint

    s2 = MemoryStore(cli_store_path)
    try:
        assert s2.get_belief("g1").alpha == 10.0
    finally:
        s2.close()


def test_clamp_ghosts_cli_apply_clamps(cli_store_path: Path) -> None:
    s = MemoryStore(cli_store_path)
    s.insert_belief(_mk("g1", alpha=10.0))
    s.close()

    code, output = _run_cli("--apply")

    assert code == 0
    assert "clamped=1" in output
    assert "dry_run=False" in output

    s2 = MemoryStore(cli_store_path)
    try:
        assert s2.get_belief("g1").alpha == DEFAULT_TARGET_ALPHA
    finally:
        s2.close()


def test_clamp_ghosts_cli_threshold_and_target_flags(cli_store_path: Path) -> None:
    s = MemoryStore(cli_store_path)
    s.insert_belief(_mk("g1", alpha=10.0))
    s.close()

    code, output = _run_cli("--threshold", "8.0", "--target", "6.0", "--apply")

    assert code == 0
    assert "threshold=8.0" in output
    assert "target=6.0" in output

    s2 = MemoryStore(cli_store_path)
    try:
        assert s2.get_belief("g1").alpha == 6.0
    finally:
        s2.close()


# -- legitimately-ingested rows are not ghosts (#1374 §12) ------------------

def _fresh_user_belief() -> Belief:
    """A brand-new user-sourced belief, built by the production insert
    path rather than by hand.

    `derive()` on a user transcript turn that classifies as a
    `requirement` stamps the undeflated TYPE_PRIORS prior — alpha=9.0,
    lock_level='none', origin='user_transcript' — and the row has no
    feedback_history and no belief_corroborations because it was just
    created. That is every condition the pre-fix selector tested for.
    """
    from aelfrice.derivation import DerivationInput, derive
    from aelfrice.models import INGEST_SOURCE_TRANSCRIPT

    out = derive(DerivationInput(
        raw_text="Deploys must run the migration check before shipping.",
        source_kind=INGEST_SOURCE_TRANSCRIPT,
        source_path="transcript",
        raw_meta={"role": "user"},
        ts="2026-04-26T00:00:00Z",
    ))
    assert out.belief is not None
    # Pin the properties that make this a hard case. If the insert path
    # ever stops writing an above-threshold alpha here, the test below
    # would start passing for the wrong reason.
    assert out.belief.alpha > DEFAULT_TARGET_ALPHA
    assert out.belief.lock_level == LOCK_NONE
    return out.belief


def test_skips_freshly_ingested_user_belief_but_still_clamps_a_ghost(
    store: MemoryStore,
) -> None:
    legit = _fresh_user_belief()
    store.insert_belief(legit)
    # A real ghost sits alongside it: same alpha band, same empty audit
    # trail, but a legacy origin no insert path writes at that alpha.
    store.insert_belief(_mk("ghost", alpha=10.0))

    result = clamp_ghost_alphas(store, dry_run=False)

    assert [s["id"] for s in result.sample] == ["ghost"]
    assert result.matched == 1 and result.clamped == 1
    assert _alpha_of(store, "ghost") == DEFAULT_TARGET_ALPHA
    # The legitimate row keeps its birth prior and gains no fabricated
    # audit row attributing a clamp to the tool.
    assert _alpha_of(store, legit.id) == legit.alpha
    assert _feedback_rows_for(store, legit.id) == []


def test_user_prior_origins_are_all_excluded(store: MemoryStore) -> None:
    # Enumerated literally rather than read off USER_PRIOR_ORIGINS: a
    # fixture built from the constant under test shrinks to nothing the
    # moment the constant does, and would stay green either way.
    user_origins = (
        "user_corrected",
        "user_stated",
        "user_transcript",
        "user_validated",
    )
    assert set(user_origins) == set(USER_PRIOR_ORIGINS)
    for i, origin in enumerate(user_origins):
        store.insert_belief(replace(_mk(f"u{i}", alpha=9.0), origin=origin))
    store.insert_belief(_mk("ghost", alpha=9.0))

    result = clamp_ghost_alphas(store)

    assert [s["id"] for s in result.sample] == ["ghost"]


def test_created_before_cutoff_excludes_newer_rows(store: MemoryStore) -> None:
    old = replace(_mk("old", alpha=10.0), created_at="2026-04-01T00:00:00Z")
    new = replace(_mk("new", alpha=10.0), created_at="2026-04-30T00:00:00Z")
    store.insert_belief(old)
    store.insert_belief(new)

    result = clamp_ghost_alphas(
        store, dry_run=False, created_before="2026-04-15T00:00:00Z"
    )

    assert [s["id"] for s in result.sample] == ["old"]
    assert result.clamped == 1
    assert _alpha_of(store, "old") == DEFAULT_TARGET_ALPHA
    assert _alpha_of(store, "new") == 10.0


def test_clamp_ghosts_cli_created_before_flag(cli_store_path: Path) -> None:
    s = MemoryStore(cli_store_path)
    s.insert_belief(
        replace(_mk("old", alpha=10.0), created_at="2026-04-01T00:00:00Z")
    )
    s.insert_belief(
        replace(_mk("new", alpha=10.0), created_at="2026-04-30T00:00:00Z")
    )
    s.close()

    code, output = _run_cli("--created-before", "2026-04-15T00:00:00Z", "--apply")

    assert code == 0
    assert "matched=1" in output and "clamped=1" in output

    s2 = MemoryStore(cli_store_path)
    try:
        assert s2.get_belief("old").alpha == DEFAULT_TARGET_ALPHA
        assert s2.get_belief("new").alpha == 10.0
    finally:
        s2.close()


# -- static-SQL selector invariants (#1374) ---------------------------------

def test_both_selectors_share_one_eligibility_predicate() -> None:
    # The enumeration query and the under-the-write-lock re-check must
    # apply the SAME predicate: a looser re-check would clamp a row the
    # enumeration never offered. Asserting the substring is what makes
    # that structural rather than a comment — inlining the predicate into
    # either query separately fails here.
    assert _ELIGIBILITY_SQL in _GHOST_SELECT_SQL
    assert _ELIGIBILITY_SQL in _GHOST_RECHECK_SQL


def test_selector_sql_is_static_and_fully_parameterised() -> None:
    # Both query strings are module constants with no interpolation, so
    # no caller value can reach the SQL text. Pinned as a test because
    # the previous shape composed them per-call from an origin-count
    # placeholder list and a conditionally-appended cutoff clause.
    for sql in (_GHOST_SELECT_SQL, _GHOST_RECHECK_SQL):
        assert "{" not in sql and "}" not in sql
        for origin in USER_PRIOR_ORIGINS:
            assert origin not in sql
    # Three bound parameters per splice, in the order _eligibility_params
    # emits them; a drift between the two desynchronises every call site.
    assert _ELIGIBILITY_SQL.count("?") == 3
    assert len(_eligibility_params(None)) == 3


def test_origins_bind_as_one_sorted_json_array() -> None:
    # The SHIPPED value, written out. Comparing against
    # sorted(USER_PRIOR_ORIGINS) would pass either way today, because
    # the constant already happens to be in alphabetical order — so it
    # would pin nothing and would go green if the exclusion shrank.
    assert json.loads(_eligibility_params(None)[0]) == [
        "user_corrected",
        "user_stated",
        "user_transcript",
        "user_validated",
    ]


def test_origins_array_is_sorted_regardless_of_constant_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The MECHANISM, which the shipped value cannot exercise. The
    # docstring claims the array is serialised sorted so its byte value
    # is stable if the tuple is ever reordered; without this, dropping
    # the sort is a silent no-op edit.
    monkeypatch.setattr(
        clamp_ghosts, "USER_PRIOR_ORIGINS", ("user_validated", "user_corrected")
    )
    assert json.loads(_eligibility_params(None)[0]) == [
        "user_corrected",
        "user_validated",
    ]


def test_empty_created_before_means_no_cutoff(store: MemoryStore) -> None:
    # "" is falsy and previously selected the no-clause branch. Under a
    # bound `? IS NULL` cutoff an empty string is NOT null, so it would
    # read as "created before the empty string" and match nothing. The
    # normalisation that preserves the old meaning is only visible here.
    store.insert_belief(replace(_mk("old", alpha=10.0), created_at="2026-04-01T00:00:00Z"))
    store.insert_belief(replace(_mk("new", alpha=10.0), created_at="2026-04-30T00:00:00Z"))

    result = clamp_ghost_alphas(store, created_before="")

    assert sorted(s["id"] for s in result.sample) == ["new", "old"]
    assert result.matched == 2


def test_none_limit_processes_every_match(store: MemoryStore) -> None:
    # LIMIT is always bound now; the no-cap case passes a negative
    # sentinel rather than omitting the clause. A sentinel of 0 (or any
    # non-negative value) silently returns nothing.
    for i in range(5):
        store.insert_belief(_mk(f"g{i}", alpha=9.0))

    result = clamp_ghost_alphas(store, dry_run=False, limit=None)

    assert result.matched == 5
    assert result.clamped == 5


# -- the α ceiling the origin exclusion rests on (#1374) --------------------

# Source labels the non-user insert paths actually mint (scanner's
# `doc:` / `ast:` / `git:` prefixes), plus case and whitespace variants:
# the deflation gate is `source != USER_SOURCE`, an exact case-sensitive
# comparison, so "User" and " user" must deflate like any other.
_NON_USER_SOURCES: tuple[str, ...] = (
    "",
    "agent",
    "transcript",
    "doc:README.md:p0",
    "ast:src/aelfrice/store.py:module",
    "git:commit:0123abc",
    "User",
    "USER",
    " user",
    "user ",
)


def test_no_deterministic_non_user_insert_can_reach_the_clamp_threshold() -> None:
    """The module docstring's safety argument, as an executable assertion.

    The selector excludes only USER_PRIOR_ORIGINS; every other origin is
    clampable. The justification is a numeric claim — a non-user source's
    α is deflated to at most 1.8, below the α=4.0 threshold, so no insert
    on a clampable origin can be selected.

    That claim is the product of four shipped constants. Asserting only
    the inequality would let the margin erode silently: deflation
    0.2 → 0.4 leaves the max at 3.6, still under 4.0, while the
    docstring's "1.8" quietly becomes false. So the constants, the
    derived ceiling, and the inequality are pinned separately.

    SCOPE, so this does not read as a stronger guarantee than it is. It
    bounds α by *source*. It says nothing about `route_overrides`, which
    bypasses `get_source_adjusted_prior` and writes the producer's α
    verbatim, nor about migration-preserved legacy rows on
    `origin='unknown'` — both named in the module docstring, and neither
    constrained here.
    """
    # Shipped values written out literally: a fixture read off the
    # constant it guards stays green through any edit to that constant.
    assert TYPE_PRIORS == {
        "requirement": (9.0, 0.5),
        "correction": (9.0, 0.5),
        "preference": (7.0, 1.0),
        "factual": (3.0, 1.0),
    }
    assert cc._AGENT_INFERRED_DEFLATION == 0.2  # noqa: SLF001
    assert cc._DEFLATED_ALPHA_FLOOR == 0.5  # noqa: SLF001
    assert cc.USER_SOURCE == "user"
    assert DEFAULT_THRESHOLD_ALPHA == 4.0

    # Every type the classifier can emit, plus an unmapped string that
    # exercises the unknown-type fallback. `speculative` is in
    # BELIEF_TYPES but has no prior, so it takes that fallback too.
    belief_types = sorted({*TYPE_PRIORS, *BELIEF_TYPES, "not-a-belief-type"})
    assert "speculative" in belief_types and "speculative" not in TYPE_PRIORS
    # The fallback is the factual prior. Repointing it at `requirement`
    # would raise the real ceiling while every aggregate below still
    # passed, so it is pinned on its own.
    assert get_source_adjusted_prior("not-a-belief-type", "user") == TYPE_PRIORS[
        "factual"
    ]

    ceiling = max(
        get_source_adjusted_prior(t, s)[0]
        for t in belief_types
        for s in _NON_USER_SOURCES
    )
    assert ceiling == pytest.approx(1.8)
    assert ceiling < DEFAULT_THRESHOLD_ALPHA
    # The margin, named. The reviewed claim was that headroom is 2.2 on
    # the insert path; if a prior moves it, that fails here.
    assert DEFAULT_THRESHOLD_ALPHA - ceiling == pytest.approx(2.2)

    # And the contrast that makes the origin exclusion necessary at all:
    # the user source is deliberately NOT bounded by that ceiling.
    user_max = max(
        get_source_adjusted_prior(t, cc.USER_SOURCE)[0] for t in belief_types
    )
    assert user_max == pytest.approx(9.0)
    assert user_max > DEFAULT_THRESHOLD_ALPHA
