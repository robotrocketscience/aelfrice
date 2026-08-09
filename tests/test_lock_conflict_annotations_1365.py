"""#1365 (#1175 proposal 2): wire lock-conflict annotations into retrieval.

The measurement half shipped in #1244 and sat unwired — `lock_consistency.
lock_conflict_annotations` had no production caller. These tests cover the
wiring: the flag, the process snapshot, the compute inside
`retrieve_with_tiers`, and the render.
"""
from __future__ import annotations

import pytest

from aelfrice import hook, retrieval
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief

ENV = "AELFRICE_LOCK_CONFLICT_ANNOTATIONS"


def _mk(bid: str, content: str, *, locked: bool = False) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash="h_" + bid,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-08-06T00:00:00Z" if locked else None,
        created_at="2026-07-21T00:00:00Z",
        last_retrieved_at=None,
    )


# --- flag resolver -------------------------------------------------------


def test_flag_defaults_off(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv(ENV, raising=False)
    assert retrieval.is_lock_conflict_annotations_enabled(start=tmp_path) is False


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "TRUE", " on "])
def test_flag_env_truthy(
    raw: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv(ENV, raw)
    assert retrieval.is_lock_conflict_annotations_enabled(start=tmp_path) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off"])
def test_flag_env_falsy(
    raw: str, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv(ENV, raw)
    assert retrieval.is_lock_conflict_annotations_enabled(start=tmp_path) is False


def test_env_beats_an_explicit_false_kwarg(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Env-first is the repo convention, and it is load-bearing: an operator
    setting the env var must win over a caller's hard-coded default."""
    monkeypatch.setenv(ENV, "1")
    assert (
        retrieval.is_lock_conflict_annotations_enabled(False, start=tmp_path)
        is True
    )


def test_unrecognised_env_falls_through_rather_than_pinning_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """None and False are different answers. A typo must not silently pin
    the flag off — it must defer to the next rung."""
    monkeypatch.setenv(ENV, "yepp")
    assert retrieval._env_lock_conflict_annotations_override() is None
    assert (
        retrieval.is_lock_conflict_annotations_enabled(True, start=tmp_path)
        is True
    )


def test_flag_reads_toml(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv(ENV, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[retrieval]\nuse_lock_conflict_annotations = true\n", encoding="utf-8"
    )
    assert retrieval.is_lock_conflict_annotations_enabled(start=tmp_path) is True


# --- the process snapshot ------------------------------------------------


def test_snapshot_round_trips() -> None:
    retrieval._reset_last_lock_conflict_annotations({"b1": "LK1"})
    assert dict(retrieval.last_lock_conflict_annotations()) == {"b1": "LK1"}
    retrieval._reset_last_lock_conflict_annotations({})


def test_snapshot_is_read_only() -> None:
    """The accessor hands out a process global. A bare dict would let any
    caller mutate what every later reader sees."""
    retrieval._reset_last_lock_conflict_annotations({"b1": "LK1"})
    with pytest.raises(TypeError):
        retrieval.last_lock_conflict_annotations()["b2"] = "LK2"  # type: ignore[index]
    retrieval._reset_last_lock_conflict_annotations({})


def test_snapshot_copies_its_source() -> None:
    """Mutating the caller's dict afterwards must not reach the snapshot."""
    src = {"b1": "LK1"}
    retrieval._reset_last_lock_conflict_annotations(src)
    src["b2"] = "LK2"
    assert dict(retrieval.last_lock_conflict_annotations()) == {"b1": "LK1"}
    retrieval._reset_last_lock_conflict_annotations({})


# --- the render ----------------------------------------------------------


def test_render_is_byte_identical_without_annotations() -> None:
    b = _mk("b1", "latency is 9 ms")
    plain, _ = hook._split_belief_lines([b], annotations={})
    assert plain[0] == '<belief id="b1" lock="none">latency is 9 ms</belief>'


def test_render_names_the_conflicting_lock() -> None:
    b = _mk("b1", "latency is 9 ms")
    annotated, _ = hook._split_belief_lines([b], annotations={"b1": "LK7"})
    assert (
        annotated[0]
        == '<belief id="b1" lock="none" conflicts_with="LK7">'
        "latency is 9 ms</belief>"
    )


def test_render_leaves_unannotated_beliefs_untouched() -> None:
    """A pack where only one belief conflicts must not perturb the others."""
    hits = [_mk("b1", "latency is 9 ms"), _mk("b2", "unrelated content")]
    lines, _ = hook._split_belief_lines(hits, annotations={"b1": "LK7"})
    assert 'conflicts_with="LK7"' in lines[0]
    assert "conflicts_with" not in lines[1]


def test_render_annotation_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no annotations passed and the flag unset, the snapshot is not
    consulted at all — so a stale snapshot cannot leak into the block."""
    monkeypatch.delenv(ENV, raising=False)
    retrieval._reset_last_lock_conflict_annotations({"b1": "STALE"})
    lines, _ = hook._split_belief_lines([_mk("b1", "latency is 9 ms")])
    assert "conflicts_with" not in lines[0]
    retrieval._reset_last_lock_conflict_annotations({})


def test_render_consults_the_snapshot_when_the_flag_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV, "1")
    retrieval._reset_last_lock_conflict_annotations({"b1": "LK9"})
    lines, _ = hook._split_belief_lines([_mk("b1", "latency is 9 ms")])
    assert 'conflicts_with="LK9"' in lines[0]
    retrieval._reset_last_lock_conflict_annotations({})


# --- suppression: symmetric and slot-scoped ------------------------------
#
# These are the mutation-checked arms. Each asserts a *presence* and an
# *absence* on the same input, so deleting the suppression makes them go
# red rather than merely changing a count.


def test_suppressed_literal_alone_yields_no_annotation() -> None:
    """A version literal is not a disagreement. It is the dominant source of
    noise measured in #1175 (6.12% -> 1.38% once suppressed)."""
    from aelfrice.lock_consistency import lock_conflict_annotations
    from aelfrice.value_compare import extract_values

    lock = _mk("LK1", "version is 2", locked=True)
    cand = _mk("b1", "version is 3")
    out = lock_conflict_annotations(
        [("b1", extract_values(cand.content))],
        [(lock, extract_values(lock.content))],
    )
    assert out == {}


def test_a_genuine_disagreement_is_still_annotated_alongside_a_suppressed_one(
) -> None:
    """Slot-scoped, not belief-scoped. A belief carrying BOTH a suppressed
    version literal AND a real numeric disagreement must still be annotated
    on the real one — suppressing the whole belief would lose the signal
    this feature exists to surface."""
    from aelfrice.lock_consistency import lock_conflict_annotations
    from aelfrice.value_compare import extract_values

    lock = _mk("LK1", "version is 2 and retry limit is 5", locked=True)
    cand = _mk("b1", "version is 3 and retry limit is 9")
    out = lock_conflict_annotations(
        [("b1", extract_values(cand.content))],
        [(lock, extract_values(lock.content))],
    )
    assert out == {"b1": "LK1"}


def test_suppression_applies_to_the_candidate_side_too() -> None:
    """Symmetric. Filtering only the lock side would let a candidate's own
    version literal manufacture a conflict against a real lock slot."""
    from aelfrice.lock_consistency import annotation_slots
    from aelfrice.value_compare import extract_values

    cand_slots = annotation_slots(extract_values("version is 3"))
    assert not cand_slots.numeric


# --- the compute inside retrieve_with_tiers ------------------------------


@pytest.fixture
def conflict_store():
    """A user lock and an unlocked belief that numerically disagrees with it
    on a slot no suppression rule covers."""
    from aelfrice.store import MemoryStore

    s = MemoryStore(":memory:")
    s.insert_belief(_mk("LK1", "retry limit is 5", locked=True))
    s.insert_belief(_mk("b1", "retry limit is 9"))
    yield s
    s.close()


def test_compute_does_not_run_when_the_flag_is_off(
    conflict_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate must precede the compute, not merely the render — otherwise
    the default path pays for value extraction it never uses.

    Exercising a real `retrieve_with_tiers` is the point: asserting only
    that the resolver returns False would pass even if the compute ran
    unconditionally.
    """
    monkeypatch.delenv(ENV, raising=False)

    def boom(*_a: object, **_k: object) -> object:  # pragma: no cover
        raise AssertionError("extract_values ran with the flag off")

    monkeypatch.setattr("aelfrice.value_compare.extract_values", boom)
    retrieval.retrieve_with_tiers(conflict_store, "retry limit")
    assert dict(retrieval.last_lock_conflict_annotations()) == {}


def test_compute_annotates_a_real_conflict_end_to_end(
    conflict_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV, "1")
    retrieval.retrieve_with_tiers(conflict_store, "retry limit")
    assert dict(retrieval.last_lock_conflict_annotations()) == {"b1": "LK1"}


def test_snapshot_is_cleared_between_calls(
    conflict_store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A call that finds no conflict must not serve the previous call's
    annotations — the stale-snapshot failure #1366 closed for lane firings."""
    monkeypatch.setenv(ENV, "1")
    retrieval.retrieve_with_tiers(conflict_store, "retry limit")
    assert dict(retrieval.last_lock_conflict_annotations()) == {"b1": "LK1"}

    monkeypatch.delenv(ENV, raising=False)
    retrieval.retrieve_with_tiers(conflict_store, "retry limit")
    assert dict(retrieval.last_lock_conflict_annotations()) == {}
