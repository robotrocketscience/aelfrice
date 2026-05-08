"""Tests for the worker's `route_overrides` plumbing (#265 PR-B commit 1).

Each test is a falsifiable hypothesis about the LLM-router post-derivation
splice contract ratified on PR #478:

  - `route_overrides` flows through `raw_meta -> DerivationInput`.
  - When set, `derive()` skips the regex classifier and uses the router's
    `(type, origin, alpha, beta)`. Byte-identical to today's scanner.py
    direct-write LLM-classify path on the same input.
  - The worker emits a `feedback_history` audit row when
    `audit_source` is set AND the belief was newly inserted (matches
    today's scanner.py:301-310 `if was_inserted` guard).
  - Absent `route_overrides`, today's regex-path behavior is unchanged.

Scanner-side migration to `record_ingest + run_worker` lives in commit 2.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.derivation import (
    DerivationInput,
    RouteOverrides,
    derive,
)
from aelfrice.derivation_worker import (
    _derivation_input_from_row,
    _route_overrides_from_raw_meta,
    run_worker,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    INGEST_SOURCE_FILESYSTEM,
    ORIGIN_AGENT_INFERRED,
    ORIGIN_USER_STATED,
)
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "ro.db"))
    yield s
    s.close()


def _record_with_overrides(
    store: MemoryStore,
    text: str,
    overrides: dict | None,
    *,
    source_path: str = "doc:notes.md",
    call_site: str = CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    ts: str = "2026-05-08T00:00:00+00:00",
) -> str:
    raw_meta: dict[str, object] = {"call_site": call_site}
    if overrides is not None:
        raw_meta["route_overrides"] = overrides
    return store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path=source_path,
        raw_text=text,
        raw_meta=raw_meta,
        ts=ts,
    )


# ---------------------------------------------------------------------------
# `RouteOverrides` dataclass
# ---------------------------------------------------------------------------


def test_route_overrides_is_frozen():
    """Frozen so callers can stash it as a dict value safely."""
    ro = RouteOverrides(
        belief_type="factual",
        origin="agent_inferred",
        alpha=1.4,
        beta=0.6,
    )
    with pytest.raises(Exception):
        ro.alpha = 2.0  # type: ignore[misc]


def test_route_overrides_audit_source_optional():
    ro = RouteOverrides(belief_type="factual", origin="x", alpha=1.0, beta=1.0)
    assert ro.audit_source is None


# ---------------------------------------------------------------------------
# `_route_overrides_from_raw_meta`
# ---------------------------------------------------------------------------


def test_parses_well_formed_block():
    block = {
        "belief_type": "factual",
        "origin": ORIGIN_AGENT_INFERRED,
        "alpha": 1.4,
        "beta": 0.6,
        "audit_source": "llm_router_v1",
    }
    ro = _route_overrides_from_raw_meta({"route_overrides": block})
    assert ro is not None
    assert ro.belief_type == "factual"
    assert ro.origin == ORIGIN_AGENT_INFERRED
    assert ro.alpha == pytest.approx(1.4)
    assert ro.beta == pytest.approx(0.6)
    assert ro.audit_source == "llm_router_v1"


def test_returns_none_when_block_absent():
    assert _route_overrides_from_raw_meta({"call_site": "x"}) is None


def test_returns_none_for_missing_required_field():
    block = {"belief_type": "factual", "origin": "x", "alpha": 1.0}
    # missing beta
    assert _route_overrides_from_raw_meta({"route_overrides": block}) is None


def test_returns_none_for_wrong_type():
    block = {
        "belief_type": "factual",
        "origin": "x",
        "alpha": "not a number",
        "beta": 1.0,
    }
    assert _route_overrides_from_raw_meta({"route_overrides": block}) is None


def test_returns_none_for_empty_strings():
    block = {"belief_type": "", "origin": "x", "alpha": 1.0, "beta": 1.0}
    assert _route_overrides_from_raw_meta({"route_overrides": block}) is None


def test_int_alpha_beta_coerce_to_float():
    block = {"belief_type": "f", "origin": "x", "alpha": 2, "beta": 3}
    ro = _route_overrides_from_raw_meta({"route_overrides": block})
    assert ro is not None
    assert isinstance(ro.alpha, float) and ro.alpha == 2.0
    assert isinstance(ro.beta, float) and ro.beta == 3.0


def test_audit_source_none_when_empty_string():
    block = {
        "belief_type": "f", "origin": "x", "alpha": 1.0, "beta": 1.0,
        "audit_source": "",
    }
    ro = _route_overrides_from_raw_meta({"route_overrides": block})
    assert ro is not None
    assert ro.audit_source is None


def test_returns_none_when_raw_meta_is_none():
    assert _route_overrides_from_raw_meta(None) is None


# ---------------------------------------------------------------------------
# `_derivation_input_from_row` forwards route_overrides
# ---------------------------------------------------------------------------


def test_derivation_input_carries_route_overrides():
    row = {
        "id": "log-1",
        "raw_text": "x",
        "source_kind": INGEST_SOURCE_FILESYSTEM,
        "source_path": "doc:x.md",
        "ts": "2026-05-08T00:00:00+00:00",
        "raw_meta": {
            "call_site": CORROBORATION_SOURCE_FILESYSTEM_INGEST,
            "route_overrides": {
                "belief_type": "factual",
                "origin": ORIGIN_AGENT_INFERRED,
                "alpha": 1.5,
                "beta": 0.5,
            },
        },
    }
    inp = _derivation_input_from_row(row)
    assert inp.route_overrides is not None
    assert inp.route_overrides.belief_type == "factual"
    assert inp.route_overrides.alpha == pytest.approx(1.5)


def test_derivation_input_route_overrides_default_none():
    row = {
        "id": "log-2",
        "raw_text": "x",
        "source_kind": INGEST_SOURCE_FILESYSTEM,
        "raw_meta": {"call_site": "x"},
    }
    inp = _derivation_input_from_row(row)
    assert inp.route_overrides is None


# ---------------------------------------------------------------------------
# `derive()` honors route_overrides
# ---------------------------------------------------------------------------


def test_derive_uses_router_fields_when_set():
    """Hypothesis: derive() with route_overrides produces a Belief whose
    (type, origin, alpha, beta) match the overrides verbatim. Falsifiable
    by any of the four fields differing."""
    inp = DerivationInput(
        raw_text="The agent stores beliefs in SQLite.",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:notes.md",
        ts="2026-05-08T00:00:00+00:00",
        route_overrides=RouteOverrides(
            belief_type="opinion",
            origin=ORIGIN_USER_STATED,
            alpha=2.0,
            beta=0.5,
        ),
    )
    out = derive(inp)
    assert out.belief is not None
    assert out.belief.type == "opinion"
    assert out.belief.origin == ORIGIN_USER_STATED
    assert out.belief.alpha == pytest.approx(2.0)
    assert out.belief.beta == pytest.approx(0.5)


def test_derive_id_unchanged_by_route_overrides():
    """Belief id is sha256(source\\x00text)[:16] regardless of overrides
    — same scheme used by scanner._derive_belief_id today, so byte-
    identical canonical state holds with or without the override."""
    common_kwargs = {
        "raw_text": "Some content here for hashing.",
        "source_kind": INGEST_SOURCE_FILESYSTEM,
        "source_path": "doc:x.md",
        "ts": "2026-05-08T00:00:00+00:00",
    }
    base = derive(DerivationInput(**common_kwargs))
    routed = derive(DerivationInput(
        **common_kwargs,
        route_overrides=RouteOverrides(
            belief_type=BELIEF_FACTUAL,
            origin=ORIGIN_AGENT_INFERRED,
            alpha=1.0,
            beta=1.0,
        ),
    ))
    assert base.belief is not None
    assert routed.belief is not None
    assert base.belief.id == routed.belief.id


def test_derive_skips_classifier_when_route_overrides_set():
    """Question-form text is normally rejected by classify_sentence
    (persist=False). With route_overrides, the router's persist=True
    decision wins and a belief lands. Matches scanner.py LLM-path
    today."""
    inp = DerivationInput(
        raw_text="Are we storing this as a belief?",
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:notes.md",
        route_overrides=RouteOverrides(
            belief_type="factual",
            origin=ORIGIN_AGENT_INFERRED,
            alpha=1.0,
            beta=1.0,
        ),
    )
    out = derive(inp)
    assert out.belief is not None  # would be None without overrides


# ---------------------------------------------------------------------------
# End-to-end through `run_worker`
# ---------------------------------------------------------------------------


def test_worker_writes_belief_with_overrides_from_log_row(
    store: MemoryStore,
) -> None:
    """Hypothesis: a log row carrying raw_meta.route_overrides yields a
    canonical belief whose router-owned fields match the sub-dict
    verbatim. End-to-end through run_worker()."""
    log_id = _record_with_overrides(
        store,
        "The retrieval engine uses BM25.",
        overrides={
            "belief_type": "factual",
            "origin": ORIGIN_AGENT_INFERRED,
            "alpha": 1.7,
            "beta": 0.3,
        },
    )
    result = run_worker(store)
    assert result.beliefs_inserted == 1

    row = store.get_ingest_log_entry(log_id)
    assert row is not None
    derived = row["derived_belief_ids"]
    assert isinstance(derived, list) and len(derived) == 1

    belief = store.get_belief(derived[0])
    assert belief is not None
    assert belief.type == "factual"
    assert belief.origin == ORIGIN_AGENT_INFERRED
    assert belief.alpha == pytest.approx(1.7)
    assert belief.beta == pytest.approx(0.3)


def test_worker_emits_audit_row_on_new_insert_with_audit_source(
    store: MemoryStore,
) -> None:
    """Hypothesis: when route_overrides.audit_source is set AND the
    belief is newly inserted, the worker writes one feedback_history
    row tagged with the supplied source string. Mirrors scanner's
    pre-migration behavior at scanner.py:304-310."""
    _record_with_overrides(
        store,
        "Source-marked belief content.",
        overrides={
            "belief_type": "factual",
            "origin": ORIGIN_AGENT_INFERRED,
            "alpha": 1.0,
            "beta": 1.0,
            "audit_source": "llm_router_v1",
        },
    )
    run_worker(store)

    events = store.list_feedback_events()
    assert len(events) == 1
    assert events[0].source == "llm_router_v1"
    assert events[0].valence == pytest.approx(0.0)


def test_worker_skips_audit_row_when_audit_source_absent(
    store: MemoryStore,
) -> None:
    """Hypothesis: an LLM-routed row without audit_source produces a
    belief but no audit row — only the optional `audit_source` field
    triggers the feedback_history emission."""
    _record_with_overrides(
        store,
        "Different content for a unique id.",
        overrides={
            "belief_type": "factual",
            "origin": ORIGIN_AGENT_INFERRED,
            "alpha": 1.0,
            "beta": 1.0,
        },
    )
    run_worker(store)

    assert store.count_feedback_events() == 0


def test_worker_skips_audit_row_on_corroboration(
    store: MemoryStore,
) -> None:
    """Hypothesis: a second log row carrying the same text + audit_source
    does NOT write a second audit row, because the belief was
    corroborated rather than inserted. Matches scanner's `if
    was_inserted` guard at scanner.py:301-310."""
    overrides = {
        "belief_type": "factual",
        "origin": ORIGIN_AGENT_INFERRED,
        "alpha": 1.0,
        "beta": 1.0,
        "audit_source": "llm_router_v1",
    }
    _record_with_overrides(store, "Repeated content.", overrides=overrides)
    run_worker(store)
    assert store.count_feedback_events() == 1

    _record_with_overrides(
        store,
        "Repeated content.",
        overrides=overrides,
        ts="2026-05-08T01:00:00+00:00",
    )
    run_worker(store)

    # Still 1 — second pass should corroborate, not insert.
    assert store.count_feedback_events() == 1


def test_worker_unchanged_when_no_overrides(store: MemoryStore) -> None:
    """Regex-path baseline: a log row without route_overrides produces a
    classifier-derived belief, no audit row. Today's behavior must hold."""
    store.record_ingest(
        source_kind=INGEST_SOURCE_FILESYSTEM,
        source_path="doc:r.md",
        raw_text="The system uses an SQLite store for beliefs.",
        raw_meta={"call_site": CORROBORATION_SOURCE_FILESYSTEM_INGEST},
        ts="2026-05-08T00:00:00+00:00",
    )
    result = run_worker(store)
    assert result.beliefs_inserted == 1
    assert store.count_feedback_events() == 0
