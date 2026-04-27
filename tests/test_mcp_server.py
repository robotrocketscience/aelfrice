"""Atomic tests for the 8 MCP tool handlers.

Tests target the pure `tool_*` functions, not the FastMCP-decorated
`aelf_*` wrappers. The fastmcp dependency is optional and not installed
in the dev environment; importing `aelfrice.mcp_server` must work
without it (only `serve()` requires it).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.classification import (
    HostClassification,
    accept_classifications,
    start_onboard_session,
)
from aelfrice.mcp_server import (
    tool_demote,
    tool_feedback,
    tool_health,
    tool_lock,
    tool_locked,
    tool_onboard,
    tool_onboard_sync,
    tool_search,
    tool_stats,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_REQUIREMENT,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import Store


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


def _put_belief(
    store: Store,
    *,
    id: str = "b1",
    content: str = "hello world",
    alpha: float = 2.0,
    beta: float = 1.0,
    lock_level: str = LOCK_NONE,
    demotion_pressure: int = 0,
) -> Belief:
    locked_at = "2026-04-26T00:00:00Z" if lock_level == LOCK_USER else None
    b = Belief(
        id=id,
        content=content,
        content_hash="hh",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        demotion_pressure=demotion_pressure,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )
    store.insert_belief(b)
    return b


def _populate_repo(root: Path) -> None:
    (root / "README.md").write_text(
        "This project must use uv for environment management.\n\n"
        "We always prefer atomic commits over batched commits.\n\n"
        "The system follows a Bayesian feedback loop with locks.\n"
    )


# --- import surface ----------------------------------------------------


def test_module_imports_without_fastmcp() -> None:
    """The module must be importable in environments where fastmcp is
    not installed; only `serve()` requires it."""
    import importlib
    mod = importlib.import_module("aelfrice.mcp_server")
    assert hasattr(mod, "serve")


def test_serve_raises_clear_error_when_fastmcp_missing() -> None:
    from aelfrice.mcp_server import serve
    # fastmcp is not in dev deps; serve() must raise RuntimeError, not ImportError.
    with pytest.raises(RuntimeError, match="aelfrice\\[mcp\\]"):
        serve()


# --- search ------------------------------------------------------------


def test_search_returns_results_kind(store: Store) -> None:
    _put_belief(store, content="the quick brown fox")
    out = tool_search(store, query="brown")
    assert out["kind"] == "search.results"


def test_search_returns_hit_payload_shape(store: Store) -> None:
    _put_belief(store, content="the quick brown fox")
    out = tool_search(store, query="brown")
    assert out["n_hits"] == 1
    hit = out["hits"][0]
    assert set(hit.keys()) == {"id", "content", "lock_level", "type"}


def test_search_no_match_returns_zero_hits(store: Store) -> None:
    out = tool_search(store, query="zzznonexistent")
    assert out["n_hits"] == 0


# --- lock --------------------------------------------------------------


def test_lock_creates_new_belief(store: Store) -> None:
    out = tool_lock(store, statement="we always sign commits with ssh")
    assert out["action"] == "locked"
    bid = out["id"]
    inserted = store.get_belief(bid)
    assert inserted is not None
    assert inserted.lock_level == LOCK_USER


def test_lock_upgrades_existing_unlocked_belief(store: Store) -> None:
    first = tool_lock(store, statement="claude is my friend")
    bid = first["id"]
    # demote then re-lock to exercise the upgrade path
    b = store.get_belief(bid)
    assert b is not None
    b.lock_level = LOCK_NONE
    b.locked_at = None
    store.update_belief(b)
    second = tool_lock(store, statement="claude is my friend")
    assert second["action"] == "upgraded"
    assert second["id"] == bid


# --- locked ------------------------------------------------------------


def test_locked_lists_user_locks(store: Store) -> None:
    tool_lock(store, statement="alpha")
    tool_lock(store, statement="beta")
    out = tool_locked(store)
    assert out["n"] == 2


def test_locked_pressured_filter_excludes_unpressured(store: Store) -> None:
    tool_lock(store, statement="alpha")
    out = tool_locked(store, pressured=True)
    assert out["n"] == 0


def test_locked_pressured_filter_includes_pressured(store: Store) -> None:
    tool_lock(store, statement="alpha")
    bid = tool_locked(store)["locked"][0]["id"]
    b = store.get_belief(bid)
    assert b is not None
    b.demotion_pressure = 3
    store.update_belief(b)
    out = tool_locked(store, pressured=True)
    assert out["n"] == 1


# --- demote ------------------------------------------------------------


def test_demote_unlocks_a_locked_belief(store: Store) -> None:
    bid = tool_lock(store, statement="x")["id"]
    out = tool_demote(store, belief_id=bid)
    assert out["demoted"] is True
    b = store.get_belief(bid)
    assert b is not None
    assert b.lock_level == LOCK_NONE


def test_demote_unknown_belief_returns_not_found(store: Store) -> None:
    out = tool_demote(store, belief_id="deadbeef")
    assert out["kind"] == "demote.not_found"
    assert out["demoted"] is False


def test_demote_unlocked_belief_is_no_op(store: Store) -> None:
    _put_belief(store, id="b1", lock_level=LOCK_NONE)
    out = tool_demote(store, belief_id="b1")
    assert out["kind"] == "demote.not_locked"
    assert out["demoted"] is False


# --- feedback ----------------------------------------------------------


def test_feedback_used_increments_alpha(store: Store) -> None:
    _put_belief(store, id="b1", alpha=2.0, beta=1.0)
    out = tool_feedback(store, belief_id="b1", signal="used")
    assert out["kind"] == "feedback.applied"
    assert out["new_alpha"] > out["prior_alpha"]


def test_feedback_harmful_increments_beta(store: Store) -> None:
    _put_belief(store, id="b1", alpha=2.0, beta=1.0)
    out = tool_feedback(store, belief_id="b1", signal="harmful")
    assert out["new_beta"] > out["prior_beta"]


def test_feedback_bad_signal_returns_error(store: Store) -> None:
    _put_belief(store, id="b1")
    out = tool_feedback(store, belief_id="b1", signal="bogus")
    assert out["kind"] == "feedback.bad_signal"
    assert "error" in out


def test_feedback_unknown_belief_returns_error(store: Store) -> None:
    out = tool_feedback(store, belief_id="nope", signal="used")
    assert out["kind"] == "feedback.unknown_belief"


# --- stats / health ----------------------------------------------------


def test_stats_returns_count_keys(store: Store) -> None:
    out = tool_stats(store)
    assert {
        "beliefs", "edges", "locked", "feedback_events",
        "onboard_sessions_total",
    }.issubset(out.keys())


def test_stats_counts_match_after_inserts(store: Store) -> None:
    _put_belief(store, id="a")
    _put_belief(store, id="b")
    out = tool_stats(store)
    assert out["beliefs"] == 2


def test_health_returns_regime_field(store: Store) -> None:
    out = tool_health(store)
    assert "regime" in out
    assert "description" in out


def test_health_insufficient_data_omits_features(store: Store) -> None:
    """Empty store -> insufficient_data regime; features must not be
    reported (they would be undefined)."""
    out = tool_health(store)
    if out["regime"] == "insufficient_data":
        assert "features" not in out


# --- onboard (polymorphic) --------------------------------------------


def test_onboard_path_starts_session(store: Store, tmp_path: Path) -> None:
    _populate_repo(tmp_path)
    out = tool_onboard(store, path=str(tmp_path))
    assert out["kind"] == "onboard.session_started"
    assert "session_id" in out
    assert isinstance(out["sentences"], list)


def test_onboard_status_no_pending_returns_zero(store: Store) -> None:
    out = tool_onboard(store)
    assert out["kind"] == "onboard.status"
    assert out["n_pending"] == 0


def test_onboard_status_lists_started_session(
    store: Store, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    started = tool_onboard(store, path=str(tmp_path))
    sid = started["session_id"]
    out = tool_onboard(store)
    assert sid in out["pending_session_ids"]


def test_onboard_accept_completes_session(
    store: Store, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    started = tool_onboard(store, path=str(tmp_path))
    sid = started["session_id"]
    classifications = [
        {"index": s["index"], "belief_type": BELIEF_FACTUAL, "persist": True}
        for s in started["sentences"]
    ]
    out = tool_onboard(store, session_id=sid, classifications=classifications)
    assert out["kind"] == "onboard.session_completed"
    assert out["inserted"] == len(classifications)


def test_onboard_accept_with_no_classifications_completes_empty(
    store: Store, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    started = tool_onboard(store, path=str(tmp_path))
    out = tool_onboard(store, session_id=started["session_id"])
    assert out["inserted"] == 0
    assert out["skipped_unclassified"] == len(started["sentences"])


def test_onboard_after_complete_no_longer_pending(
    store: Store, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    started = tool_onboard(store, path=str(tmp_path))
    tool_onboard(store, session_id=started["session_id"])
    status = tool_onboard(store)
    assert status["n_pending"] == 0


def test_onboard_sync_falls_back_to_regex_classifier(
    store: Store, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    out = tool_onboard_sync(store, path=str(tmp_path))
    assert out["kind"] == "onboard.sync_completed"
    assert out["total_candidates"] > 0


# --- end-to-end smoke ---------------------------------------------------


def test_end_to_end_lock_search_feedback_demote(store: Store) -> None:
    locked = tool_lock(store, statement="we use uv exclusively")
    bid = locked["id"]

    hits = tool_search(store, query="uv exclusively")
    assert any(h["id"] == bid for h in hits["hits"])

    fb = tool_feedback(store, belief_id=bid, signal="used")
    assert fb["new_alpha"] > fb["prior_alpha"]

    dem = tool_demote(store, belief_id=bid)
    assert dem["demoted"] is True

    stats = tool_stats(store)
    assert stats["locked"] == 0
    assert stats["beliefs"] == 1


def test_end_to_end_polymorphic_onboard_then_search(
    store: Store, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    started = tool_onboard(store, path=str(tmp_path))
    classifications = [
        {"index": s["index"], "belief_type": BELIEF_REQUIREMENT, "persist": True}
        for s in started["sentences"]
    ]
    tool_onboard(store, session_id=started["session_id"],
                  classifications=classifications)
    hits = tool_search(store, query="uv environment")
    assert hits["n_hits"] >= 1


# --- internal-helper sanity checks -------------------------------------


def test_polymorphic_onboard_handlers_share_state_with_classification_module(
    store: Store, tmp_path: Path
) -> None:
    """Round-trip via the bare classification API should match the MCP
    tool's view of the world. Catches drift between the two surfaces."""
    _populate_repo(tmp_path)
    started_via_mcp = tool_onboard(store, path=str(tmp_path))
    sid = started_via_mcp["session_id"]
    accept_classifications(
        store,
        sid,
        [
            HostClassification(index=s["index"], belief_type=BELIEF_FACTUAL, persist=True)
            for s in started_via_mcp["sentences"]
        ],
        now="2026-04-26T01:00:00Z",
    )
    pending = tool_onboard(store)
    assert pending["n_pending"] == 0


def test_unused_classification_import_does_not_drift(
    store: Store, tmp_path: Path
) -> None:
    _populate_repo(tmp_path)
    via_classification = start_onboard_session(
        store, tmp_path, now="2026-04-26T00:00:00Z"
    )
    # Same store viewed through the MCP status tool — pending sessions
    # from the bare API show up in the MCP status response.
    out = tool_onboard(store)
    assert via_classification.session_id in out["pending_session_ids"]
