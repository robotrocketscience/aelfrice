"""Tests for the semantic relationship detector (#201, audit slice).

Covers the verdict logic on hand-crafted pairs and one end-to-end
pass through `detect_relationships()` on a tiny store. Audit-only:
no edges are inserted; tests assert the returned report, not store
state.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.bm25 import tokenize
from aelfrice.relationship_detector import (
    DEFAULT_CONFIDENCE_MIN,
    RELATIONSHIP_SECTION,
    VERDICT_CONTRADICTS,
    VERDICT_REFINES,
    VERDICT_UNRELATED,
    RelationshipDetectorConfig,
    classify_pair,
    detect_relationships,
    format_audit_report,
    load_relationship_config,
)
from aelfrice.store import MemoryStore


def _classify(a: str, b: str, *, confidence_min: float = DEFAULT_CONFIDENCE_MIN):
    ta = frozenset(tokenize(a))
    tb = frozenset(tokenize(b))
    # jaccard_score is informational on the returned pair; the
    # candidate-pair pipeline computes it separately. Pass 1.0 to
    # signal "candidate cleared the prefilter" — verdict logic does
    # not consult it directly.
    return classify_pair(
        content_a=a,
        content_b=b,
        tokens_a=ta,
        tokens_b=tb,
        jaccard_score=1.0,
        confidence_min=confidence_min,
    )


# --- classify_pair: verdict logic --------------------------------------


def test_quantifier_axis_contradiction_always_vs_never():
    pair = _classify(
        "always push to main when CI is green",
        "never push to main when CI is green",
    )
    assert pair.verdict == VERDICT_CONTRADICTS
    assert pair.confidence >= DEFAULT_CONFIDENCE_MIN
    assert pair.auto_emit is True


def test_quantifier_axis_disagreement_often_vs_rarely_below_gate():
    pair = _classify(
        "deployments often fail without manual review",
        "deployments rarely fail without manual review",
    )
    # often vs rarely -> 1.0 axis delta -> q_term 0.5 -> contradicts,
    # but with a softer disagreement signal than always↔never.
    assert pair.verdict == VERDICT_CONTRADICTS
    # Confidence should be lower than the always↔never case.
    stronger = _classify(
        "deployments always fail without manual review",
        "deployments never fail without manual review",
    )
    assert pair.confidence < stronger.confidence


def test_negation_flip_contradiction():
    pair = _classify(
        "the build pipeline runs uv install before pytest",
        "the build pipeline does not run uv install before pytest",
    )
    assert pair.verdict == VERDICT_CONTRADICTS


def test_negation_contraction_caught():
    # Contracted form should still trip the negation flag even though
    # the bm25 tokenizer splits `don't` into `["don", "t"]`.
    pair = _classify(
        "we use uv for python deps",
        "we don't use uv for python deps",
    )
    assert pair.verdict == VERDICT_CONTRADICTS


def test_hedge_vs_certainty_asymmetric_disagreement():
    pair = _classify(
        "tests must run on every push to main",
        "tests might run on every push to main",
    )
    assert pair.verdict == VERDICT_CONTRADICTS
    # Lower than negation-flip — m_term caps at 0.5.
    flip = _classify(
        "tests run on every push to main",
        "tests do not run on every push to main",
    )
    assert pair.confidence <= flip.confidence


def test_agreeing_paraphrase_is_unrelated_or_refines():
    # No modality conflict, no quantifier conflict. The longer one is
    # a strict token-superset of the shorter, so refines fires.
    pair = _classify(
        "use uv for python deps",
        "always use uv for python deps in this project",
    )
    assert pair.verdict in (VERDICT_REFINES, VERDICT_UNRELATED)


def test_different_aspects_same_subject_not_flagged():
    # The classic false-positive shape: both reference Python tooling
    # but neither contradicts the other. The residual-content overlap
    # check should prevent the contradiction verdict.
    pair = _classify(
        "use uv for python deps",
        "never use pip in this project",
    )
    # Even with overlapping `use` token, the residual content (uv,
    # python, deps) vs (pip, project) does not pass residual >= 0.5.
    assert pair.verdict != VERDICT_CONTRADICTS


def test_unrelated_pair_low_confidence():
    pair = _classify(
        "the build runs on every push",
        "the cat sat on the mat in the kitchen",
    )
    assert pair.verdict == VERDICT_UNRELATED
    assert pair.confidence == 0.0


# --- classify_pair: confidence gate ------------------------------------


def test_auto_emit_respects_confidence_min_override():
    # Lower the floor — a soft contradiction now auto-emits.
    pair = _classify(
        "deployments often fail without manual review",
        "deployments rarely fail without manual review",
        confidence_min=0.4,
    )
    assert pair.verdict == VERDICT_CONTRADICTS
    assert pair.auto_emit is True


# --- detect_relationships: end-to-end ----------------------------------


def _belief_id_for(content: str) -> str:
    # Use the live store's id format (sha256 prefix). Inserting via the
    # public API does this for us — we just need a Belief constructor.
    from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
    import hashlib
    h = hashlib.sha256(("test:" + content).encode()).hexdigest()[:16]
    return Belief(
        id=h,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest()[:12],
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-03T00:00:00Z",
        last_retrieved_at=None,
    )


def test_detect_relationships_finds_contradiction_pair():
    store = MemoryStore(":memory:")
    a = _belief_id_for("always push to main when CI is green")
    b = _belief_id_for("never push to main when CI is green")
    c = _belief_id_for("the cat sat on the mat in the kitchen")
    store.insert_belief(a)
    store.insert_belief(b)
    store.insert_belief(c)
    report = detect_relationships(store, jaccard_min=0.4)
    store.close()
    assert report.n_beliefs_scanned == 3
    assert report.n_contradicts == 1
    assert report.n_auto_emit == 1
    assert len(report.pairs) == 1
    pair = report.pairs[0]
    assert pair.verdict == VERDICT_CONTRADICTS
    assert {pair.belief_a_id, pair.belief_b_id} == {a.id, b.id}


def test_detect_relationships_empty_store():
    store = MemoryStore(":memory:")
    report = detect_relationships(store)
    store.close()
    assert report.n_beliefs_scanned == 0
    assert report.pairs == ()


def test_detect_relationships_rejects_bad_thresholds():
    store = MemoryStore(":memory:")
    try:
        with pytest.raises(ValueError):
            detect_relationships(store, jaccard_min=1.5)
        with pytest.raises(ValueError):
            detect_relationships(store, confidence_min=-0.1)
        with pytest.raises(ValueError):
            detect_relationships(store, max_candidate_pairs=0)
    finally:
        store.close()


# --- format_audit_report -----------------------------------------------


def test_format_audit_report_empty_store_is_human_readable():
    store = MemoryStore(":memory:")
    report = detect_relationships(store)
    store.close()
    text = format_audit_report(report)
    assert "aelf doctor relationships" in text
    assert "Beliefs scanned" in text
    assert "0" in text


def test_format_audit_report_marks_auto_emit_rows():
    store = MemoryStore(":memory:")
    a = _belief_id_for("always push to main when CI is green")
    b = _belief_id_for("never push to main when CI is green")
    store.insert_belief(a)
    store.insert_belief(b)
    report = detect_relationships(store, jaccard_min=0.4)
    store.close()
    text = format_audit_report(report)
    assert "contradicts" in text
    # The `*` marker indicates auto_emit.
    assert "* contradicts" in text


# --- Config loader -----------------------------------------------------


def test_config_defaults_when_no_toml(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_relationship_config(start=tmp_path)
    assert cfg == RelationshipDetectorConfig()


def test_config_reads_overrides(tmp_path: Path):
    (tmp_path / ".aelfrice.toml").write_text(
        f"[{RELATIONSHIP_SECTION}]\n"
        "jaccard_min = 0.6\n"
        "confidence_min = 0.7\n"
        "max_candidate_pairs = 1000\n"
    )
    cfg = load_relationship_config(start=tmp_path)
    assert cfg.jaccard_min == 0.6
    assert cfg.confidence_min == 0.7
    assert cfg.max_candidate_pairs == 1000


def test_config_rejects_out_of_range_floats(tmp_path: Path):
    (tmp_path / ".aelfrice.toml").write_text(
        f"[{RELATIONSHIP_SECTION}]\n"
        "jaccard_min = 1.5\n"
        "confidence_min = -0.1\n"
    )
    cfg = load_relationship_config(start=tmp_path)
    # Out-of-range values fall back to defaults; loader does not raise.
    assert cfg == RelationshipDetectorConfig()
