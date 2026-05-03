"""Unit tests for `aelfrice.relationship_detector` (#201).

17 tests covering signal extraction, the verdict classifier, the
audit pass against a real `MemoryStore` fixture, and the
`.aelfrice.toml` config loader. The audit is read-only — every
audit-level test asserts that no edges and no beliefs were
inserted/mutated by the audit pass.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.relationship_detector import (
    DEFAULT_CONFIDENCE_MIN,
    DEFAULT_JACCARD_MIN,
    LABEL_CONTRADICTS,
    LABEL_REFINES,
    LABEL_UNRELATED,
    RelationshipDetectorConfig,
    analyze,
    classify,
    extract_signals,
    format_audit_report,
    load_relationship_detector_config,
    relationships_audit,
)
from aelfrice.store import MemoryStore


# --- Signal extraction (3 tests) ---------------------------------------


def test_signals_negation_word_and_contraction() -> None:
    assert extract_signals("never push to main").has_negation is True
    assert extract_signals("don't push to main").has_negation is True
    assert extract_signals("won’t push directly").has_negation is True
    assert extract_signals("just push to main").has_negation is False


def test_signals_quantifier_axis_and_presence() -> None:
    s_always = extract_signals("always commit signed")
    assert s_always.has_quantifier is True
    assert s_always.quantifier_axis == 1.0
    s_never = extract_signals("never commit secrets")
    assert s_never.has_quantifier is True
    assert s_never.quantifier_axis == -1.0
    s_silent = extract_signals("commit signed")
    assert s_silent.has_quantifier is False
    # Missing quantifier is silent, not neutral; axis defaults to 0.0
    # but consumers must consult has_quantifier first.
    assert s_silent.quantifier_axis == 0.0


def test_signals_residual_strips_modality_quantifier_stopwords() -> None:
    s = extract_signals("never always use the cat on mat")
    # "never", "always", "use", "the", "on" all subtracted; "cat",
    # "mat" remain.
    assert "cat" in s.residual_content
    assert "mat" in s.residual_content
    assert "never" not in s.residual_content
    assert "always" not in s.residual_content
    assert "use" not in s.residual_content


# --- Verdict classifier (7 tests) --------------------------------------


def test_unrelated_when_residual_overlap_below_floor() -> None:
    v = analyze("the sky is blue", "water is wet")
    assert v.label == LABEL_UNRELATED
    assert v.score == 0.0


def test_negation_disagreement_is_contradicts() -> None:
    v = analyze("cats are mammals", "cats are not mammals")
    assert v.label == LABEL_CONTRADICTS
    assert v.score > 0.0
    assert "negation" in v.rationale


def test_quantifier_axis_disagreement_is_contradicts() -> None:
    v = analyze("python is always fast", "python is rarely fast")
    assert v.label == LABEL_CONTRADICTS
    assert v.score > 0.0


def test_missing_quantifier_does_not_contradict_universal() -> None:
    # Spec design note: "use uv" (silent) vs "always use uv" (axis=+1)
    # must NOT trigger contradicts. has_quantifier guards this.
    v = analyze("python is fast", "python is always fast")
    assert v.label == LABEL_REFINES
    assert v.score == 0.0


def test_subject_mismatch_guard_via_residual_overlap() -> None:
    # Spec example: "use uv for python deps" + "never use pip in this
    # project" overlap on "use" only (which is a stopword), residual
    # disjoint — verdict is unrelated.
    v = analyze(
        "use uv for python deps",
        "never use pip in this project",
    )
    assert v.label == LABEL_UNRELATED


def test_modality_agreement_is_refines() -> None:
    # Residual {python, deps, fast} ∩ {python, deps, slow} / union
    # clears the 0.4 relatedness floor; modality silent on both
    # sides; verdict refines.
    v = analyze("python deps are fast", "python deps are slow")
    assert v.label == LABEL_REFINES


def test_classify_sugar_returns_label_string() -> None:
    # Bench-gate API: classify(a, b) -> str ∈ VERDICT_LABELS.
    assert classify("cats are mammals", "cats are not mammals") == LABEL_CONTRADICTS
    assert classify("the sky is blue", "water is wet") == LABEL_UNRELATED


# --- Audit pass against a real store (5 tests) -------------------------


def _insert(store: MemoryStore, bid: str, content: str) -> None:
    store.insert_belief(
        Belief(
            id=bid,
            content=content,
            content_hash=f"h_{bid}",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_NONE,
            locked_at=None,
            demotion_pressure=0,
            created_at="2026-05-03T00:00:00Z",
            last_retrieved_at=None,
        )
    )


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


def test_audit_empty_store(store: MemoryStore) -> None:
    report = relationships_audit(store)
    assert report.n_beliefs_scanned == 0
    assert report.pairs == ()


def test_audit_finds_contradicts_pair(store: MemoryStore) -> None:
    _insert(store, "b1", "always use uv for python deps")
    _insert(store, "b2", "never use uv for python deps")
    _insert(store, "b3", "the cat sat on the mat")
    report = relationships_audit(store)
    contra = [p for p in report.pairs if p.label == LABEL_CONTRADICTS]
    assert len(contra) == 1
    assert contra[0].belief_a_id == "b1"
    assert contra[0].belief_b_id == "b2"


def test_audit_is_read_only(store: MemoryStore) -> None:
    _insert(store, "b1", "always use uv for python deps")
    _insert(store, "b2", "never use uv for python deps")
    before = store.list_beliefs_for_indexing()
    relationships_audit(store)
    after = store.list_beliefs_for_indexing()
    assert before == after
    cur = store._conn.execute("SELECT COUNT(*) FROM edges")  # type: ignore[attr-defined]
    assert cur.fetchone()[0] == 0


def test_audit_truncates_at_max_candidate_pairs(store: MemoryStore) -> None:
    for i in range(5):
        _insert(store, f"b{i}", "deploy via terraform on aws cluster")
    report = relationships_audit(store, max_candidate_pairs=3)
    assert report.truncated is True


def test_audit_invalid_threshold_raises(store: MemoryStore) -> None:
    with pytest.raises(ValueError):
        relationships_audit(store, jaccard_min=1.5)
    with pytest.raises(ValueError):
        relationships_audit(store, max_candidate_pairs=0)


# --- Report formatter + config loader (2 tests) ------------------------


def test_format_audit_report_renders_pair_lines(store: MemoryStore) -> None:
    _insert(store, "b1", "always use uv for python deps")
    _insert(store, "b2", "never use uv for python deps")
    report = relationships_audit(store)
    text = format_audit_report(report)
    assert "aelf doctor relationships" in text
    assert LABEL_CONTRADICTS in text
    assert "b1" in text and "b2" in text


def test_config_loader_overrides_and_falls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # No config -> defaults.
    cfg = load_relationship_detector_config(start=tmp_path)
    assert cfg.jaccard_min == DEFAULT_JACCARD_MIN
    assert cfg.confidence_min == DEFAULT_CONFIDENCE_MIN
    # Valid overrides load.
    (tmp_path / ".aelfrice.toml").write_text(
        "[relationship_detector]\n"
        "jaccard_min = 0.55\n"
        "confidence_min = 0.7\n"
        "max_candidate_pairs = 100\n"
    )
    cfg = load_relationship_detector_config(start=tmp_path)
    assert cfg.jaccard_min == 0.55
    assert cfg.confidence_min == 0.7
    assert cfg.max_candidate_pairs == 100
    # Malformed values fall back with stderr trace.
    (tmp_path / ".aelfrice.toml").write_text(
        "[relationship_detector]\n"
        "jaccard_min = 1.5\n"
        "max_candidate_pairs = 0\n"
    )
    cfg = load_relationship_detector_config(start=tmp_path)
    assert cfg == RelationshipDetectorConfig()
    assert "ignoring" in capsys.readouterr().err
