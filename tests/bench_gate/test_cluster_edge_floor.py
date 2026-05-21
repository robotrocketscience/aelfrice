"""Bench gate for #724: cluster_edge_floor substrate verification.

**Scope.** This is a substrate-only regression catch for the
``cluster_candidates`` floor parameter. It does **not** reproduce the
lab finding that motivated #724 (3x recall@k uplift at floor=0.6 due
to ``EDGE_CITES`` rows dropping out at weight 0.5). The labeled
rerank-relevance corpus v0_1 (#819) carries no edges, so the lab
finding cannot be re-bench'd against this corpus shape.

What this test does verify:
  * The floor parameter has the *correct mechanism* — raising it
    from 0.4 to 0.6 produces strictly fewer non-singleton clusters
    when the synthesised edge weights span that range.
  * No regression in ``cluster_candidates`` output when the floor
    moves: candidate-induced subgraph rules hold, singleton
    fallback still works.

What this test does **NOT** verify:
  * Whether ``DEFAULT_CLUSTER_EDGE_FLOOR = 0.6`` would actually
    improve recall@k on real workloads. That is gated on a v0_2
    corpus extension that adds typed edges (``EDGE_CITES`` etc.)
    with realistic weights, tracked separately.
  * Latency budget at the higher floor — the synthetic edges are
    deterministic and small; real-store edge counts are different.

Determinism (#605): edge synthesis uses pure-stdlib token-set
Jaccard; same belief texts -> same edges -> same clusters at any
floor. No randomness, no wall-clock.

Discretion (`ab96e9d3501b1c14`): the bench reads only the labeled
corpus through ``load_corpus_module``; no ``~/.claude/``-derived
state enters this file. Synthesised edges are deterministic
functions of belief text content, so their existence and weights
ride entirely on the corpus's own discretion-scrub posture.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest

from aelfrice.clustering import (
    DEFAULT_CLUSTER_EDGE_FLOOR,
    Edge,
    cluster_candidates,
)
from aelfrice.models import Belief
from tests.conftest import load_corpus_module


_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> frozenset[str]:
    """Lowercase alnum tokens. Pure function; deterministic."""
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Standard Jaccard over frozensets. Pure; deterministic."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union


_EDGE_WEIGHT_FLOOR: Final[float] = 0.30
"""Below this Jaccard, the pair gets no edge at all (sparser graph)."""

_EDGE_WEIGHT_SPAN_LO: Final[float] = 0.30
_EDGE_WEIGHT_SPAN_HI: Final[float] = 0.80
"""Synthesised edge weights map [_EDGE_WEIGHT_FLOOR, 1.0] linearly
into [_EDGE_WEIGHT_SPAN_LO, _EDGE_WEIGHT_SPAN_HI], so a non-trivial
fraction of edges sit in (0.4, 0.6) and the floor parameter actually
filters at the test's two operating points."""


def _synth_edges(beliefs: list[dict]) -> list[Edge]:
    """Synthesize edges between candidate beliefs via token-Jaccard.

    For each pair (i, j) with i < j, compute Jaccard over lowercase
    alnum tokens of belief text. If Jaccard >= ``_EDGE_WEIGHT_FLOOR``,
    emit a single directed edge i -> j with weight scaled into
    ``[_EDGE_WEIGHT_SPAN_LO, _EDGE_WEIGHT_SPAN_HI]``.

    Edge type is fixed at ``"synth_sim"`` (not a real edge-type from
    the brain graph; this is a bench-substrate label).
    """
    token_sets = [_tokens(b["text"]) for b in beliefs]
    out: list[Edge] = []
    span = _EDGE_WEIGHT_SPAN_HI - _EDGE_WEIGHT_SPAN_LO
    denom = 1.0 - _EDGE_WEIGHT_FLOOR
    for i in range(len(beliefs)):
        for j in range(i + 1, len(beliefs)):
            sim = _jaccard(token_sets[i], token_sets[j])
            if sim < _EDGE_WEIGHT_FLOOR:
                continue
            weight = _EDGE_WEIGHT_SPAN_LO + ((sim - _EDGE_WEIGHT_FLOOR) / denom) * span
            out.append(
                Edge(src=beliefs[i]["id"], dst=beliefs[j]["id"],
                     type="synth_sim", weight=weight),
            )
    return out


def _to_belief(row_belief: dict) -> Belief:
    """Build a minimal-fields Belief from a corpus row's belief entry."""
    return Belief(
        id=row_belief["id"],
        content=row_belief["text"],
        content_hash="",
        alpha=1.0,
        beta=1.0,
        type="snapshot",
        lock_level=row_belief.get("lock_level", "none"),
        locked_at=None,
        created_at="",
        last_retrieved_at=None,
    )


def _scores_descending(beliefs: list[dict]) -> dict[str, float]:
    """Assign monotonically descending scores by corpus row order.

    The cluster algorithm uses scores only to pick seed/representative
    ordering. For substrate testing, the absolute values don't matter
    as long as they're distinct; row order is the deterministic source.
    """
    return {b["id"]: 1.0 - (i / max(1, len(beliefs)))
            for i, b in enumerate(beliefs)}


@pytest.mark.bench_gated
def test_v0_1_corpus_loads(aelfrice_corpus_root: Path) -> None:
    """Smoke: corpus mounts, rows parse, every row has the fields
    the bench depends on. Catches schema drift early."""
    rows = load_corpus_module(aelfrice_corpus_root, "rerank_relevance")
    assert rows, "rerank_relevance corpus produced zero rows"
    for row in rows:
        assert "beliefs" in row, f"row {row.get('id')} missing 'beliefs'"
        assert isinstance(row["beliefs"], list)
        assert row["beliefs"], f"row {row.get('id')} has empty belief pool"
        for b in row["beliefs"]:
            assert isinstance(b, dict)
            assert "id" in b
            assert "text" in b


@pytest.mark.bench_gated
def test_synth_edges_span_floor_range(aelfrice_corpus_root: Path) -> None:
    """The bench's synthetic edges must straddle the 0.4-0.6 floor
    range across the corpus, or the floor parameter can't be
    meaningfully tested. This guard fires if Jaccard scores cluster
    too narrowly (e.g. all rows on identical topics)."""
    rows = load_corpus_module(aelfrice_corpus_root, "rerank_relevance")
    in_range = 0
    total = 0
    for row in rows:
        for e in _synth_edges(row["beliefs"]):
            total += 1
            if 0.4 <= e.weight < 0.6:
                in_range += 1
    # Loose gate: at least some edges sit in the cluster_edge_floor
    # transition zone across the whole corpus. If this fails the
    # corpus is too homogeneous for floor-parameter testing.
    assert total > 0, "no synthetic edges generated across the corpus"
    assert in_range >= max(1, total // 50), (
        f"only {in_range}/{total} edges in (0.4, 0.6) "
        "— floor parameter has no transition zone"
    )


@pytest.mark.bench_gated
def test_higher_floor_yields_fewer_non_singleton_clusters(
    aelfrice_corpus_root: Path,
) -> None:
    """Raising the floor from 0.4 to 0.6 must produce monotonically
    weakly-fewer non-singleton clusters per row: edges drop out, some
    components fragment, singleton count goes up.

    'Weakly fewer' means: at least one row shows strict decrease,
    no row shows increase. Strict-decrease-everywhere would over-
    constrain on rows whose edges all sit below 0.4 or all above
    0.6 (those rows are insensitive to the floor in this range)."""
    rows = load_corpus_module(aelfrice_corpus_root, "rerank_relevance")
    any_decrease = False
    for row in rows:
        beliefs = [_to_belief(b) for b in row["beliefs"]]
        edges = _synth_edges(row["beliefs"])
        scores = _scores_descending(row["beliefs"])
        cl_04 = cluster_candidates(
            beliefs, scores, edges=edges, edge_weight_floor=0.4,
        )
        cl_06 = cluster_candidates(
            beliefs, scores, edges=edges, edge_weight_floor=0.6,
        )
        non_singleton_04 = sum(1 for c in cl_04 if len(c.member_ids) > 1)
        non_singleton_06 = sum(1 for c in cl_06 if len(c.member_ids) > 1)
        assert non_singleton_06 <= non_singleton_04, (
            f"row {row.get('id')}: non-singleton count went UP "
            f"at higher floor ({non_singleton_06} > {non_singleton_04}); "
            "monotonicity property violated"
        )
        if non_singleton_06 < non_singleton_04:
            any_decrease = True
    assert any_decrease, (
        "no row showed any structural change moving floor 0.4 -> 0.6; "
        "either the corpus is too homogeneous OR the floor parameter "
        "is no-op'd in cluster_candidates"
    )


@pytest.mark.bench_gated
def test_singletons_preserved_at_both_floors(
    aelfrice_corpus_root: Path,
) -> None:
    """Every belief in every row must appear in exactly one cluster
    at both floors (no orphans, no duplicates). Singleton fallback
    must work; the floor parameter doesn't change membership totals."""
    rows = load_corpus_module(aelfrice_corpus_root, "rerank_relevance")
    for row in rows:
        beliefs = [_to_belief(b) for b in row["beliefs"]]
        edges = _synth_edges(row["beliefs"])
        scores = _scores_descending(row["beliefs"])
        expected_ids = {b.id for b in beliefs}
        for floor in (0.4, 0.6):
            clusters = cluster_candidates(
                beliefs, scores, edges=edges, edge_weight_floor=floor,
            )
            seen: set[str] = set()
            for c in clusters:
                for mid in c.member_ids:
                    assert mid not in seen, (
                        f"row {row.get('id')} floor={floor}: belief "
                        f"{mid} appears in multiple clusters"
                    )
                    seen.add(mid)
            assert seen == expected_ids, (
                f"row {row.get('id')} floor={floor}: missing "
                f"{expected_ids - seen}, extra {seen - expected_ids}"
            )


@pytest.mark.bench_gated
def test_default_floor_is_documented_value(
    aelfrice_corpus_root: Path,
) -> None:
    """Pin DEFAULT_CLUSTER_EDGE_FLOOR. When v0_2 evidence justifies a
    flip 0.4 -> 0.6, this assertion changes alongside the constant in
    a single commit (same pattern as test_clustering.py's k-pin)."""
    assert DEFAULT_CLUSTER_EDGE_FLOOR == 0.4, (
        f"unexpected default {DEFAULT_CLUSTER_EDGE_FLOOR}; "
        "flip requires v0_2 corpus evidence — see #724"
    )
