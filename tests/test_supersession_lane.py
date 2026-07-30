"""The supersession retrieval lane, both arms (#1187).

Superseded beliefs were never demoted or excluded at retrieval: the user
corrects "deploy target is heroku" to "fly.io", contradiction resolution
records the supersession, and the next prompt still injects heroku *first*.
`uri_baki.apply_supersession_demote` was the only implementation and had no
importer.

Both arms ship behind a default-OFF flag; the ratified three-arm bench
(demote vs exclusion vs control) picks the default. These tests pin the
mechanism, the resolvers, and — load-bearing — that the demote is additive
in the log domain rather than multiplicative on the score.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from aelfrice import retrieval
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_SUPERSEDES,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.retrieval import (
    SUPERSESSION_DEMOTE_FACTOR,
    SUPERSESSION_TREATMENT_DEMOTE,
    SUPERSESSION_TREATMENT_EXCLUDE,
    _l1_hits,
    _supersession_penalty,
    is_supersession_demote_enabled,
    resolve_supersession_factor,
    resolve_supersession_treatment,
    retrieve_v2,
)
from aelfrice.store import MemoryStore

_OLD_ID = "B" + "1" * 15
_NEW_ID = "B" + "2" * 15


def _belief(bid: str, content: str, created: str) -> Belief:
    return Belief(
        id=bid, content=content, content_hash="h" + bid[1:],
        alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
        locked_at=None, created_at=created, last_retrieved_at=None,
    )


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    """The issue's reproduction: heroku superseded by fly.io."""
    s = MemoryStore(str(tmp_path / "memory.db"))
    s.insert_belief(_belief(_OLD_ID, "deploy target is heroku", "2026-01-01T00:00:00Z"))
    s.insert_belief(_belief(_NEW_ID, "deploy target is fly.io", "2026-06-01T00:00:00Z"))
    # Producers' canonical direction (#1170): src = newer, dst = retired.
    s.insert_edge(
        Edge(src=_NEW_ID, dst=_OLD_ID, type=EDGE_SUPERSEDES, weight=1.0)
    )
    yield s
    s.close()


# --- The batched store query ---------------------------------------------


def test_returns_the_edge_target_not_its_source(store: MemoryStore) -> None:
    """A belief is superseded when it is the `dst`, never the `src`.

    Getting this backwards would demote the *replacement* — the #1170
    inversion, reintroduced one layer down.
    """
    assert store.superseded_belief_ids([_OLD_ID, _NEW_ID]) == {_OLD_ID}


def test_restricted_to_the_candidate_set(store: MemoryStore) -> None:
    """Only ids asked about come back, so the L1 rerank stays scoped."""
    assert store.superseded_belief_ids([_NEW_ID]) == set()
    assert store.superseded_belief_ids([_OLD_ID]) == {_OLD_ID}


def test_ignores_other_edge_types(store: MemoryStore) -> None:
    """CONTRADICTS is not supersession — it is the unresolved state."""
    other = "B" + "3" * 15
    store.insert_belief(_belief(other, "deploy target is render", "2026-02-01T00:00:00Z"))
    store.insert_edge(
        Edge(src=_NEW_ID, dst=other, type=EDGE_CONTRADICTS, weight=1.0)
    )

    assert store.superseded_belief_ids([_OLD_ID, _NEW_ID, other]) == {_OLD_ID}


def test_empty_input_does_no_sql(store: MemoryStore) -> None:
    assert store.superseded_belief_ids([]) == set()


def test_duplicate_ids_collapse(store: MemoryStore) -> None:
    """Two SUPERSEDES edges onto one belief still yield one entry."""
    third = "B" + "4" * 15
    store.insert_belief(_belief(third, "deploy target is fly", "2026-07-01T00:00:00Z"))
    store.insert_edge(
        Edge(src=third, dst=_OLD_ID, type=EDGE_SUPERSEDES, weight=1.0)
    )

    assert store.superseded_belief_ids([_OLD_ID, _OLD_ID]) == {_OLD_ID}


# --- The penalty: additive, not multiplicative ---------------------------


def test_penalty_is_log_of_the_factor() -> None:
    """`factor` keeps its multiplicative meaning, in the log domain."""
    sup = frozenset({_OLD_ID})

    assert _supersession_penalty(sup, _OLD_ID, 0.5) == pytest.approx(math.log(0.5))
    assert _supersession_penalty(sup, _NEW_ID, 0.5) == 0.0
    assert _supersession_penalty(None, _OLD_ID, 0.5) == 0.0


def test_penalty_never_promotes() -> None:
    """A factor of 1 is a no-op and above 1 cannot become a boost.

    `resolve_supersession_factor` clamps, but the penalty clamps too, so a
    direct caller cannot turn the demote into a promotion either.
    """
    sup = frozenset({_OLD_ID})

    assert _supersession_penalty(sup, _OLD_ID, 1.0) == 0.0
    assert _supersession_penalty(sup, _OLD_ID, 4.0) == 0.0


def test_penalty_is_finite_at_factor_zero() -> None:
    """`log(0)` is -inf, which would make the score non-comparable."""
    got = _supersession_penalty(frozenset({_OLD_ID}), _OLD_ID, 0.0)

    assert got < 0.0
    assert math.isfinite(got)


def test_a_multiplicative_demote_would_have_inverted(store: MemoryStore) -> None:
    """Why this lane is additive — the regression the design avoids.

    The composite rerank score is a log-domain quantity and is routinely
    negative (measured ~-13 on this two-belief store). `score * 0.5` on a
    negative score *raises* it, so wiring `uri_baki.apply_supersession_demote`
    as the issue suggested would have promoted the superseded belief to the
    top of the pack. This asserts the premise directly, so nobody
    "simplifies" the additive penalty back into a multiplication.
    """
    from aelfrice.scoring import partial_bayesian_score

    scored = store.search_beliefs_scored("deploy target", limit=10)
    assert scored, "the reproduction needs both beliefs to match"

    for belief, bm25_raw in scored:
        composite = partial_bayesian_score(bm25_raw, belief.alpha, belief.beta, 0.0)
        assert composite < 0.0, "premise: the composite score is negative"
        assert composite * SUPERSESSION_DEMOTE_FACTOR > composite, (
            "premise: multiplying a negative score by 0.5 raises it"
        )
        additive = composite + _supersession_penalty(
            frozenset({belief.id}), belief.id, SUPERSESSION_DEMOTE_FACTOR,
        )
        assert additive < composite, "the additive penalty always demotes"


# --- Both arms through the L1 rerank -------------------------------------


def _order(beliefs: list[Belief]) -> list[str]:
    return ["superseded" if b.id == _OLD_ID else "current" for b in beliefs]


def test_control_still_ranks_the_superseded_belief_first(
    store: MemoryStore,
) -> None:
    """The defect, unchanged with the lane off — this is the baseline arm."""
    got = _l1_hits(store, "deploy target", l1_limit=10, posterior_weight=0.0)

    assert _order(got) == ["superseded", "current"]


def test_demote_arm_reorders(store: MemoryStore) -> None:
    """The demote arm moves the retired belief below its replacement.

    `use_entity_persist_demote=False` isolates this lane; the two compose
    additively and the default-ON entity lane has its own opinion about
    these two beliefs (see the composition test below).
    """
    got = _l1_hits(
        store, "deploy target", l1_limit=10, posterior_weight=0.0,
        use_supersession_demote=True,
        supersession_treatment=SUPERSESSION_TREATMENT_DEMOTE,
        use_entity_persist_demote=False,
    )

    assert _order(got) == ["current", "superseded"]


def test_exclude_arm_drops_it_entirely(store: MemoryStore) -> None:
    got = _l1_hits(
        store, "deploy target", l1_limit=10, posterior_weight=0.0,
        use_supersession_demote=True,
        supersession_treatment=SUPERSESSION_TREATMENT_EXCLUDE,
    )

    assert _order(got) == ["current"]


def test_exclude_arm_can_empty_the_result(tmp_path: Path) -> None:
    """Every candidate superseded is a legitimate empty pack, not a crash."""
    s = MemoryStore(str(tmp_path / "m.db"))
    s.insert_belief(_belief(_OLD_ID, "deploy target is heroku", "2026-01-01T00:00:00Z"))
    s.insert_belief(_belief(_NEW_ID, "unrelated content", "2026-06-01T00:00:00Z"))
    s.insert_edge(Edge(src=_NEW_ID, dst=_OLD_ID, type=EDGE_SUPERSEDES, weight=1.0))

    got = _l1_hits(
        s, "heroku", l1_limit=10, posterior_weight=0.0,
        use_supersession_demote=True,
        supersession_treatment=SUPERSESSION_TREATMENT_EXCLUDE,
    )

    assert got == []
    s.close()


def test_lane_off_touches_no_edges(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default-off must be byte-identical, including no extra query.

    The short-circuit that returns the un-reranked FTS5 order has to stay
    reachable, or the default path pays for a lane nobody enabled.
    """
    def _boom(_ids: list[str]) -> set[str]:
        raise AssertionError("superseded_belief_ids called with the lane off")

    monkeypatch.setattr(store, "superseded_belief_ids", _boom)

    got = _l1_hits(store, "deploy target", l1_limit=10, posterior_weight=0.0)

    assert _order(got) == ["superseded", "current"]


def test_the_two_demote_lanes_compose_additively(store: MemoryStore) -> None:
    """Measured interaction, recorded because the bench has to account for it.

    At factor 0.5 the supersession penalty is log(0.5) = -0.693 — the same
    order of magnitude as the entity-persistence penalty, whose floor is
    log(1e-3) = -6.9. Here they very nearly cancel: the *current* belief
    extracts entities (S1 = 0.5, penalty -0.691) while the superseded one
    extracts none (penalty 0), so with both lanes on the pre-existing bm25
    gap survives and the order does not change. The demote is applied
    correctly either way; it is simply a weak term in composition, which is
    why the bench should sweep the factor rather than test 0.5 alone.
    """
    ep = store.entity_persistence_scores([_OLD_ID, _NEW_ID])
    assert _OLD_ID not in ep, "superseded belief extracts no entities here"
    assert ep[_NEW_ID] == pytest.approx(0.5)

    both_on = _l1_hits(
        store, "deploy target", l1_limit=10, posterior_weight=0.0,
        use_supersession_demote=True, use_entity_persist_demote=True,
    )
    sup_only = _l1_hits(
        store, "deploy target", l1_limit=10, posterior_weight=0.0,
        use_supersession_demote=True, use_entity_persist_demote=False,
    )

    assert _order(sup_only) == ["current", "superseded"]
    assert _order(both_on) == ["superseded", "current"]


# --- Resolvers -----------------------------------------------------------


def test_lane_defaults_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The flip is gated on the bench, so the default must stay False."""
    monkeypatch.delenv(retrieval.ENV_SUPERSESSION_DEMOTE, raising=False)

    assert is_supersession_demote_enabled(start=tmp_path) is False


def test_env_beats_kwarg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(retrieval.ENV_SUPERSESSION_DEMOTE, "0")

    assert is_supersession_demote_enabled(True, start=tmp_path) is False


def test_kwarg_beats_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(retrieval.ENV_SUPERSESSION_DEMOTE, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[retrieval]\nuse_supersession_demote = true\n", encoding="utf-8",
    )

    assert is_supersession_demote_enabled(False, start=tmp_path) is False
    assert is_supersession_demote_enabled(start=tmp_path) is True


def test_treatment_defaults_to_the_safer_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong demote leaves a ranking signal; a wrong exclusion does not."""
    monkeypatch.delenv(retrieval.ENV_SUPERSESSION_TREATMENT, raising=False)

    assert resolve_supersession_treatment(start=tmp_path) == (
        SUPERSESSION_TREATMENT_DEMOTE
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("exclude", SUPERSESSION_TREATMENT_EXCLUDE),
        ("EXCLUDE", SUPERSESSION_TREATMENT_EXCLUDE),
        ("  demote  ", SUPERSESSION_TREATMENT_DEMOTE),
    ],
)
def test_treatment_is_normalised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str, expected: str,
) -> None:
    monkeypatch.setenv(retrieval.ENV_SUPERSESSION_TREATMENT, raw)

    assert resolve_supersession_treatment(start=tmp_path) == expected


def test_unknown_treatment_falls_back_rather_than_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A typo in config must not break retrieval on the default path."""
    monkeypatch.setenv(retrieval.ENV_SUPERSESSION_TREATMENT, "delete")

    got = resolve_supersession_treatment(start=tmp_path)

    assert got == SUPERSESSION_TREATMENT_DEMOTE
    assert "ignoring supersession treatment" in capsys.readouterr().err


def test_factor_defaults_and_clamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(retrieval.ENV_SUPERSESSION_FACTOR, raising=False)

    assert resolve_supersession_factor(start=tmp_path) == SUPERSESSION_DEMOTE_FACTOR
    # Above 1 would promote; below 0 has no meaning.
    assert resolve_supersession_factor(4.0, start=tmp_path) == 1.0
    assert resolve_supersession_factor(-1.0, start=tmp_path) > 0.0


def test_non_numeric_factor_env_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(retrieval.ENV_SUPERSESSION_FACTOR, "half")

    assert resolve_supersession_factor(start=tmp_path) == SUPERSESSION_DEMOTE_FACTOR
    assert "expected a number" in capsys.readouterr().err


# --- Reachable from the production path ----------------------------------


def test_retrieve_v2_threads_the_lane(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lane has to be reachable from `retrieve_v2`, not just `_l1_hits`.

    `uri_baki.apply_supersession_demote` had no importer for exactly this
    reason — a primitive nothing calls is not a fix. Uses the exclusion arm
    because it is unambiguous end-to-end (the demote arm composes with the
    default-ON entity lane, per the composition test above).
    """
    monkeypatch.delenv(retrieval.ENV_SUPERSESSION_DEMOTE, raising=False)

    off = retrieve_v2(store, "deploy target")
    on = retrieve_v2(
        store, "deploy target", use_supersession_demote=True,
        supersession_treatment=SUPERSESSION_TREATMENT_EXCLUDE,
    )

    assert _order(off.beliefs) == ["superseded", "current"]
    assert _order(on.beliefs) == ["current"]
