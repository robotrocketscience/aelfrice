"""Regression: BM25F must weight a query term by how often it is asked for.

`BM25Index.score` built its query vector with ``q_vec[j] = self.idf[j]`` —
an assignment inside the token loop, so a term repeated N times contributed
exactly once and ``score("t") == score("t t t")``.

That is not a cosmetic gap. BM25F is the default L1 lane
(`resolve_use_bm25f_anchors` -> True) and three shipped components encode
their boost *as a duplicated token*:

* `query_understanding.entity_expand` emits `qf_multiplier` copies
* `query_understanding.idf_clip` emits `boost_qf` copies for high-IDF terms
* `hook._build_conversation_aware_query` repeats the live prompt
  `prompt_weight` times "so they keep the dominant BM25 term-frequency
  contribution"

All three were therefore inert on the production lane.

The fix accumulates qf and applies Robertson & Walker (1994) query
saturation, ``idf * (k3 + 1) * qf / (k3 + qf)``. `k3` defaults to 0.0,
where the factor is exactly 1.0 for every qf >= 1 — so the shipped ranking
is unchanged and turning the mechanism on is a separate, bench-gated flip.

Falsifiable hypotheses:
  1. At ``k3 == 0`` scoring is bit-identical to the pre-fix lane.
  2. At ``k3 > 0`` a repeated term scores strictly higher, by exactly the
     Robertson factor, and can reorder the top-K.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.bm25 import DEFAULT_K3, BM25Index, BM25IndexCache
from aelfrice.derivation import DerivationInput, derive
from aelfrice.models import INGEST_SOURCE_FILESYSTEM
from aelfrice.query_understanding.entity_expand import (
    expand_with_capitalised_entities,
)
from aelfrice.retrieval import ENV_BM25_K3, resolve_bm25_k3
from aelfrice.store import MemoryStore

CORPUS = [
    "retrieval ranking uses bm25 scoring over the belief corpus",
    "the ranking pipeline also considers posterior confidence",
    "unrelated note about coffee brewing temperature and grind size",
    "retrieval retrieval retrieval is not how you boost a term",
]


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "qtf.db"))
    for i, text in enumerate(CORPUS):
        out = derive(
            DerivationInput(
                source_kind=INGEST_SOURCE_FILESYSTEM,
                raw_text=text,
                source_path=f"f{i}.md",
                session_id=None,
                ts="2026-01-01T00:00:00+00:00",
            ),
        )
        assert out.belief is not None
        s.insert_or_corroborate(out.belief, source_type="filesystem_ingest")
    yield s
    s.close()


# --- Hypothesis 1: k3 = 0 is byte-exact parity with the old lane ---------


def test_default_k3_is_zero(store: MemoryStore) -> None:
    """The shipped default must be the parity value, not a live flip."""
    assert DEFAULT_K3 == 0.0
    assert BM25Index.build(store).k3 == 0.0
    assert BM25IndexCache(store).k3 == 0.0


def test_k3_zero_ignores_repetition_exactly(store: MemoryStore) -> None:
    """Parity: identical float scores, not merely identical ordering."""
    index = BM25Index.build(store)
    once = index.score("retrieval")
    thrice = index.score("retrieval retrieval retrieval")

    assert once == thrice
    assert [s for _, s in once] == [s for _, s in thrice]


def test_k3_zero_parity_holds_for_mixed_term_queries(
    store: MemoryStore,
) -> None:
    """Weighting one term of a multi-term query must also be inert at k3=0."""
    index = BM25Index.build(store)
    assert index.score("ranking retrieval") == index.score(
        "ranking retrieval retrieval",
    )


# --- Hypothesis 2: k3 > 0 makes qf load-bearing --------------------------


def test_repetition_raises_score_when_k3_enabled(store: MemoryStore) -> None:
    index = BM25Index.build(store, k3=8.0)
    once = dict(index.score("retrieval"))
    thrice = dict(index.score("retrieval retrieval retrieval"))

    assert set(once) == set(thrice)
    for bid, base in once.items():
        assert thrice[bid] > base


@pytest.mark.parametrize(("k3", "qf"), [(8.0, 2), (8.0, 3), (1.2, 2), (1000.0, 3)])
def test_weight_matches_robertson_closed_form(
    store: MemoryStore, k3: float, qf: int,
) -> None:
    """The uplift must be exactly (k3+1)*qf/(k3+qf), not merely monotone."""
    index = BM25Index.build(store, k3=k3)
    base = dict(index.score("retrieval"))
    boosted = dict(index.score(" ".join(["retrieval"] * qf)))
    expected = (k3 + 1.0) * qf / (k3 + qf)

    for bid, score in base.items():
        assert boosted[bid] == pytest.approx(score * expected, rel=1e-5)


def test_repetition_can_reorder_top_k(store: MemoryStore) -> None:
    """The user-visible consequence: a boost actually changes what is picked."""
    index = BM25Index.build(store, k3=8.0)
    plain = index.score("ranking retrieval")
    weighted = index.score("ranking retrieval retrieval")

    assert plain[0][0] != weighted[0][0]


def test_k3_zero_and_k3_high_differ(store: MemoryStore) -> None:
    """Guards against a resolver that silently pins k3 to the default."""
    q = "retrieval retrieval retrieval"
    assert BM25Index.build(store).score(q) != BM25Index.build(
        store, k3=8.0,
    ).score(q)


def test_negative_k3_is_rejected(store: MemoryStore) -> None:
    with pytest.raises(ValueError, match="k3 must be >= 0"):
        BM25Index.build(store, k3=-1.0)


# --- The shipped boost mechanisms (issue #1166 AC4) ----------------------


def test_entity_expansion_is_inert_at_the_shipped_default(
    store: MemoryStore,
) -> None:
    """`expand_with_capitalised_entities` reaches the lane only when k3 > 0.

    This pins the AC4 measurement rather than the wish: at the shipped
    `k3 = 0` the expansion changes *nothing*, which is why the component
    has had no production effect since BM25F became the default L1 lane.
    The assertion is deliberately "no effect" — if a later change makes
    it live at k3 = 0, that is a ranking move that must be benched, and
    this test is the tripwire.
    """
    base = ["update", "the", "retrieval", "config"]
    expanded = expand_with_capitalised_entities(
        "Update the Retrieval config", base, qf_multiplier=2,
    )
    assert expanded != base  # the expansion itself does emit duplicates

    plain_q, boosted_q = " ".join(base), " ".join(expanded)
    off = BM25Index.build(store, k3=0.0)
    assert off.score(plain_q) == off.score(boosted_q)

    on = BM25Index.build(store, k3=8.0)
    assert on.score(plain_q) != on.score(boosted_q)


# --- Serialisation / sidecar: k3 must not leak across configurations -----


def test_serialize_round_trips_k3(store: MemoryStore) -> None:
    for k3 in (0.0, 8.0, 1000.0):
        index = BM25Index.build(store, k3=k3)
        restored = BM25Index.deserialize(index.serialize())
        assert restored.k3 == k3
        assert restored.score("retrieval retrieval") == index.score(
            "retrieval retrieval",
        )


def test_sidecar_is_not_reused_across_k3(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blob written with qf off must not be served to a cache with it on.

    This is the #1135 bug class applied to the new parameter: `k3` rides on
    the built index, so reusing another config's sidecar would score with a
    stale value forever.
    """
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "qtf.db"))
    cold = BM25IndexCache(store, k3=0.0)
    assert cold.get().k3 == 0.0

    warm = BM25IndexCache(store, k3=8.0)
    assert warm.get().k3 == 8.0
    q = "retrieval retrieval retrieval"
    assert warm.get().score(q) != cold.get().score(q)


# --- Resolver precedence -------------------------------------------------


def test_resolver_defaults_to_parity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv(ENV_BM25_K3, raising=False)
    assert resolve_bm25_k3(start=tmp_path) == DEFAULT_K3


def test_env_overrides_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env wins over the kwarg, matching `resolve_posterior_weight`."""
    monkeypatch.setenv(ENV_BM25_K3, "8.0")
    assert resolve_bm25_k3(explicit=2.0) == 8.0


def test_explicit_used_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_BM25_K3, raising=False)
    assert resolve_bm25_k3(explicit=2.0) == 2.0


def test_toml_key_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv(ENV_BM25_K3, raising=False)
    (tmp_path / ".aelfrice.toml").write_text(
        "[retrieval]\nbm25_k3 = 8.0\n", encoding="utf-8",
    )
    assert resolve_bm25_k3(start=tmp_path) == 8.0


@pytest.mark.parametrize("raw", ["", "   ", "not-a-float"])
def test_malformed_env_falls_through(
    monkeypatch: pytest.MonkeyPatch, raw: str,
) -> None:
    """Fail-soft: a bad env value must not raise or move the ranking."""
    monkeypatch.setenv(ENV_BM25_K3, raw)
    assert resolve_bm25_k3(explicit=2.0) == 2.0


def test_negative_env_clamps_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_BM25_K3, "-5")
    assert resolve_bm25_k3() == 0.0
