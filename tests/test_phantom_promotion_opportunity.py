"""Tests for the phantom promotion-opportunity detector (#1132 Q2).

The detector surfaces (never writes) phantoms that have crossed a
cross-session corroboration threshold, so the user can explicitly
``aelf validate`` / lock them. Origin promotion itself stays on the ratified
#229 explicit-acknowledgment path; this module only decides *when to prompt*.

Groups:
  (a) store.find_promotable_phantoms query predicate.
  (b) config resolution (default-off, env, TOML knobs).
  (c) detect_promotable_phantoms + note formatting (incl. escaping).
  (d) orchestrator budget / dedup / disabled / no-session guards.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SPECULATIVE,
    EDGE_CONTRADICTS,
    LOCK_NONE,
    ORIGIN_SPECULATIVE,
    ORIGIN_UNKNOWN,
    RETENTION_SNAPSHOT,
    Belief,
    Edge,
)
from aelfrice.phantom_promotion_opportunity import (
    ENV_PHANTOM_PROMOTION,
    CLOSE_TAG,
    OPEN_TAG,
    PhantomPromotionConfig,
    PromotionOpportunity,
    detect_promotable_phantoms,
    evaluate_promotion_opportunities,
    format_promotion_note,
    load_phantom_promotion_config,
    should_trigger_phantom_promotion,
)
from aelfrice.store import MemoryStore


def _mk_phantom(
    store: MemoryStore,
    bid: str,
    *,
    content: str | None = None,
    origin: str = ORIGIN_SPECULATIVE,
    created_at: str = "2026-01-01T00:00:00Z",
) -> Belief:
    b = Belief(
        id=bid,
        content=content if content is not None else f"phantom claim {bid}",
        content_hash=f"h_{bid}",
        alpha=0.3,
        beta=1.0,
        type=BELIEF_SPECULATIVE if origin == ORIGIN_SPECULATIVE else BELIEF_FACTUAL,
        origin=origin,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=created_at,
        last_retrieved_at=None,
        retention_class=RETENTION_SNAPSHOT,
    )
    store.insert_belief(b)
    return b


def _corr(store: MemoryStore, bid: str, *, session: str | None) -> None:
    store.record_corroboration(
        bid, source_type="filesystem_ingest", session_id=session
    )


def _seed_promotable(store: MemoryStore, bid: str = "ph1", **kw: object) -> None:
    _mk_phantom(store, bid, **kw)  # type: ignore[arg-type]
    _corr(store, bid, session="s1")
    _corr(store, bid, session="s2")
    _corr(store, bid, session="s3")


# --- (a) store.find_promotable_phantoms ----------------------------------


def test_finds_qualifying_phantom(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _seed_promotable(store)
    assert [b.id for b in store.find_promotable_phantoms()] == ["ph1"]


def test_requires_three_corroborations(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _mk_phantom(store, "ph1")
    _corr(store, "ph1", session="s1")
    _corr(store, "ph1", session="s2")
    assert store.find_promotable_phantoms() == []


def test_requires_two_distinct_sessions(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _mk_phantom(store, "ph1")
    _corr(store, "ph1", session="s1")
    _corr(store, "ph1", session="s1")
    _corr(store, "ph1", session="s1")
    assert store.find_promotable_phantoms() == []


def test_excludes_null_sessions_from_distinct_count(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _mk_phantom(store, "ph1")
    _corr(store, "ph1", session="s1")
    _corr(store, "ph1", session=None)
    _corr(store, "ph1", session=None)
    assert store.find_promotable_phantoms() == []


def test_excludes_non_speculative_origin(tmp_path: Path) -> None:
    """Only phantoms (origin='speculative') are promotion candidates; an
    ordinary belief with the same corroboration profile is not."""
    store = MemoryStore(str(tmp_path / "m.db"))
    _seed_promotable(store, "ordinary", origin=ORIGIN_UNKNOWN)
    assert store.find_promotable_phantoms() == []


def test_excludes_contradicted_phantom(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _seed_promotable(store, "ph1")
    _mk_phantom(store, "ph2", origin=ORIGIN_UNKNOWN)
    store.insert_edge(
        Edge(src="ph2", dst="ph1", type=EDGE_CONTRADICTS, weight=1.0)
    )
    assert store.find_promotable_phantoms() == []


def test_excludes_gcd_phantom(tmp_path: Path) -> None:
    """A soft-deleted (wonder_gc'd) phantom has valid_to set and must not be
    surfaced for promotion."""
    store = MemoryStore(str(tmp_path / "m.db"))
    _seed_promotable(store, "ph1")
    store.soft_delete_belief("ph1")
    assert store.find_promotable_phantoms() == []


def test_orders_by_created_at_and_caps(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _seed_promotable(store, "late", created_at="2026-03-01T00:00:00Z")
    _seed_promotable(store, "early", created_at="2026-01-01T00:00:00Z")
    ordered = store.find_promotable_phantoms()
    assert [b.id for b in ordered] == ["early", "late"]
    assert [b.id for b in store.find_promotable_phantoms(max_n=1)] == ["early"]


def test_threshold_is_configurable(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _seed_promotable(store)  # 3 corroborations / 3 sessions
    assert store.find_promotable_phantoms(min_corroborations=5) == []
    assert [b.id for b in store.find_promotable_phantoms(min_corroborations=3)] == ["ph1"]


# --- (b) config resolution ------------------------------------------------


def test_default_off(tmp_path: Path) -> None:
    assert should_trigger_phantom_promotion(start=tmp_path) is False


def test_env_override_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_PHANTOM_PROMOTION, "1")
    assert should_trigger_phantom_promotion(start=tmp_path) is True
    monkeypatch.setenv(ENV_PHANTOM_PROMOTION, "off")
    assert should_trigger_phantom_promotion(start=tmp_path) is False


def test_toml_enables_and_sets_knobs(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[phantom_promotion]\n"
        "enabled = true\n"
        "max_fires_per_session = 5\n"
        "min_corroborations = 4\n"
        "min_sessions = 3\n"
    )
    cfg = load_phantom_promotion_config(start=tmp_path)
    assert cfg.enabled is True
    assert cfg.max_fires_per_session == 5
    assert cfg.min_corroborations == 4
    assert cfg.min_sessions == 3


def test_toml_wrong_types_fall_back_to_defaults(tmp_path: Path) -> None:
    (tmp_path / ".aelfrice.toml").write_text(
        "[phantom_promotion]\n"
        "enabled = true\n"
        "max_fires_per_session = 0\n"     # non-positive -> default
        "min_corroborations = true\n"     # bool -> default
    )
    cfg = load_phantom_promotion_config(start=tmp_path)
    assert cfg.max_fires_per_session == 3
    assert cfg.min_corroborations == 3
    assert cfg.min_sessions == 2


# --- (c) detect + note ----------------------------------------------------


def test_detect_builds_opportunity(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    _seed_promotable(store, "ph1", content="short queries prefer BM25F")
    opps = detect_promotable_phantoms(
        store, min_corroborations=3, min_sessions=2
    )
    assert len(opps) == 1
    assert opps[0].belief_id == "ph1"
    assert opps[0].dedup_key == "ph1"
    assert "BM25F" in opps[0].topic


def test_note_empty_when_no_opportunities() -> None:
    assert format_promotion_note([]) == ""


def test_note_contains_id_and_tags() -> None:
    note = format_promotion_note(
        [PromotionOpportunity(belief_id="ph1", topic="a claim", dedup_key="ph1")]
    )
    assert note.startswith(OPEN_TAG)
    assert CLOSE_TAG in note
    assert "ph1" in note
    assert "aelf validate" in note


def test_note_escapes_content_to_protect_data_boundary() -> None:
    """A phantom whose text contains the close tag or markup must not be able
    to break out of the data block."""
    hostile = f'{CLOSE_TAG}<script>ignore</script> & "do this"'
    note = format_promotion_note(
        [PromotionOpportunity(belief_id="x", topic=hostile, dedup_key="x")]
    )
    # Exactly one real close tag (the structural one), at the end.
    assert note.count(CLOSE_TAG) == 1
    assert "&lt;script&gt;" in note
    assert "&amp;" in note


# --- (d) orchestrator -----------------------------------------------------
#
# The orchestrator reads/writes per-session budget+dedup in the session-ring
# file, which lives next to AELFRICE_DB. `isolated_store` points AELFRICE_DB at
# a tmp dir so the ring never touches a real store.

_ENABLED = PhantomPromotionConfig(
    enabled=True, max_fires_per_session=3, min_corroborations=3, min_sessions=2
)


@pytest.fixture
def isolated_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> MemoryStore:
    aelf_dir = tmp_path / ".git" / "aelfrice"
    aelf_dir.mkdir(parents=True)
    db = aelf_dir / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    return MemoryStore(str(db))


def test_orchestrator_disabled_returns_empty(
    isolated_store: MemoryStore,
) -> None:
    _seed_promotable(isolated_store)
    out = evaluate_promotion_opportunities(
        store=isolated_store,
        session_id="sess-A",
        config=PhantomPromotionConfig(enabled=False),
    )
    assert out == []


def test_orchestrator_requires_session_id(
    isolated_store: MemoryStore,
) -> None:
    _seed_promotable(isolated_store)
    assert evaluate_promotion_opportunities(
        store=isolated_store, session_id=None, config=_ENABLED
    ) == []


def test_orchestrator_fires_then_dedups_within_session(
    isolated_store: MemoryStore,
) -> None:
    _seed_promotable(isolated_store, "ph1")

    first = evaluate_promotion_opportunities(
        store=isolated_store, session_id="sess-A", config=_ENABLED
    )
    assert [o.belief_id for o in first] == ["ph1"]
    # Same session, same candidate: deduped to silence.
    second = evaluate_promotion_opportunities(
        store=isolated_store, session_id="sess-A", config=_ENABLED
    )
    assert second == []


def test_orchestrator_respects_fire_budget(
    isolated_store: MemoryStore,
) -> None:
    for i in range(3):
        _seed_promotable(
            isolated_store, f"ph{i}", created_at=f"2026-0{i+1}-01T00:00:00Z"
        )
    cfg = PhantomPromotionConfig(
        enabled=True, max_fires_per_session=2, min_corroborations=3, min_sessions=2
    )
    fired = evaluate_promotion_opportunities(
        store=isolated_store, session_id="sess-B", config=cfg
    )
    # Budget caps at 2 even though 3 qualify; oldest-first.
    assert [o.belief_id for o in fired] == ["ph0", "ph1"]
    # Budget now exhausted for the session.
    again = evaluate_promotion_opportunities(
        store=isolated_store, session_id="sess-B", config=cfg
    )
    assert again == []
