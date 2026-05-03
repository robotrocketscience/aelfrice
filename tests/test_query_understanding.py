"""Unit tests for the R1+R3 query understanding stack (#291 PR-1)."""
from __future__ import annotations

import numpy as np
import pytest

from aelfrice.query_understanding import (
    clip_with_quantile_thresholds,
    compute_idf_quantile_thresholds,
    expand_with_capitalised_entities,
)


# --- R1 entity expansion ----------------------------------------------------

def test_r1_expand_appends_lowercased_capitalised_at_2x():
    out = expand_with_capitalised_entities(
        raw_query="why does the Rebuilder use Bayesian inference",
        base_terms=["why", "does", "the", "rebuilder", "use", "bayesian", "inference"],
    )
    assert out[:7] == ["why", "does", "the", "rebuilder", "use", "bayesian", "inference"]
    assert out[7:] == ["rebuilder", "rebuilder", "bayesian", "bayesian"]


def test_r1_expand_preserves_match_order():
    out = expand_with_capitalised_entities(
        raw_query="Zebra Apple Mango",
        base_terms=[],
    )
    assert out == ["zebra", "zebra", "apple", "apple", "mango", "mango"]


def test_r1_expand_qf_multiplier_one_is_no_boost():
    out = expand_with_capitalised_entities(
        raw_query="Foo bar Baz",
        base_terms=["x"],
        qf_multiplier=1,
    )
    assert out == ["x", "foo", "baz"]


def test_r1_expand_no_capitalised_returns_base():
    out = expand_with_capitalised_entities(
        raw_query="all lowercase here",
        base_terms=["all", "lowercase", "here"],
    )
    assert out == ["all", "lowercase", "here"]


def test_r1_expand_empty_query():
    out = expand_with_capitalised_entities(raw_query="", base_terms=["a"])
    assert out == ["a"]


def test_r1_expand_skips_single_letter_capital():
    # Sentence pronoun "I" is a single uppercase character with no
    # trailing lowercase -- the pattern must not match it.
    out = expand_with_capitalised_entities(
        raw_query="I see Foo",
        base_terms=[],
    )
    assert out == ["foo", "foo"]


def test_r1_expand_does_not_mutate_base_terms():
    base = ["one", "two"]
    _ = expand_with_capitalised_entities("Foo", base)
    assert base == ["one", "two"]


def test_r1_expand_rejects_invalid_qf():
    with pytest.raises(ValueError):
        expand_with_capitalised_entities("Foo", [], qf_multiplier=0)
    with pytest.raises(ValueError):
        expand_with_capitalised_entities("Foo", [], qf_multiplier=-1)


def test_r1_expand_handles_camelcase_as_single_token():
    out = expand_with_capitalised_entities(
        raw_query="The MemoryStore loads BeliefRows",
        base_terms=[],
    )
    # "The" matches the rule too -- the campaign relies on R3's IDF
    # clip (downstream) to drop high-frequency words like "the".
    # CamelCase tokens collapse to a single lowercased entity each.
    assert out == [
        "the", "the",
        "memorystore", "memorystore",
        "beliefrows", "beliefrows",
    ]


# --- R3 quantile threshold computation --------------------------------------

def test_quantiles_basic():
    idf = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    low, high = compute_idf_quantile_thresholds(idf, 0.25, 0.75)
    assert low == 2.0
    assert high == 4.0


def test_quantiles_default_is_25_75():
    idf = np.arange(101, dtype=float)
    low, high = compute_idf_quantile_thresholds(idf)
    assert low == 25.0
    assert high == 75.0


def test_quantiles_empty_idf_returns_zero_zero():
    low, high = compute_idf_quantile_thresholds(np.array([]))
    assert (low, high) == (0.0, 0.0)


def test_quantiles_single_element_idf():
    low, high = compute_idf_quantile_thresholds(np.array([3.5]))
    assert low == 3.5
    assert high == 3.5


def test_quantiles_reject_swapped_args():
    with pytest.raises(ValueError):
        compute_idf_quantile_thresholds(np.array([1.0]), 0.9, 0.1)


def test_quantiles_reject_equal_args():
    with pytest.raises(ValueError):
        compute_idf_quantile_thresholds(np.array([1.0]), 0.5, 0.5)


def test_quantiles_reject_out_of_range_args():
    with pytest.raises(ValueError):
        compute_idf_quantile_thresholds(np.array([1.0]), -0.1, 0.5)
    with pytest.raises(ValueError):
        compute_idf_quantile_thresholds(np.array([1.0]), 0.5, 1.1)


# --- R3 clip ----------------------------------------------------------------

def test_clip_drops_low_keeps_mid_boosts_high():
    vocab = {"low": 0, "mid": 1, "high": 2}
    idf = np.array([0.5, 2.0, 5.0])
    out = clip_with_quantile_thresholds(
        terms=["low", "mid", "high"],
        vocabulary=vocab,
        idf=idf,
        low_threshold=1.0,
        high_threshold=4.0,
    )
    assert out == ["mid", "high", "high"]


def test_clip_passes_through_oov_terms():
    vocab = {"in": 0}
    idf = np.array([3.0])
    out = clip_with_quantile_thresholds(
        terms=["in", "outofvocab", "in"],
        vocabulary=vocab,
        idf=idf,
        low_threshold=1.0,
        high_threshold=5.0,
    )
    assert out == ["in", "outofvocab", "in"]


def test_clip_boost_qf_three_emits_three_copies():
    vocab = {"rare": 0}
    idf = np.array([10.0])
    out = clip_with_quantile_thresholds(
        terms=["rare"],
        vocabulary=vocab,
        idf=idf,
        low_threshold=1.0,
        high_threshold=2.0,
        boost_qf=3,
    )
    assert out == ["rare", "rare", "rare"]


def test_clip_boundary_idf_exactly_at_threshold_keeps_once():
    # Strict < and > -- IDF exactly at threshold falls in the mid band.
    vocab = {"a": 0, "b": 1}
    idf = np.array([1.0, 5.0])
    out = clip_with_quantile_thresholds(
        terms=["a", "b"],
        vocabulary=vocab,
        idf=idf,
        low_threshold=1.0,
        high_threshold=5.0,
    )
    assert out == ["a", "b"]


def test_clip_reject_invalid_boost_qf():
    with pytest.raises(ValueError):
        clip_with_quantile_thresholds(
            terms=[],
            vocabulary={},
            idf=np.array([]),
            low_threshold=0.0,
            high_threshold=1.0,
            boost_qf=0,
        )


def test_clip_empty_terms():
    out = clip_with_quantile_thresholds(
        terms=[],
        vocabulary={"x": 0},
        idf=np.array([1.0]),
        low_threshold=0.5,
        high_threshold=2.0,
    )
    assert out == []


def test_clip_repeated_low_term_dropped_each_time():
    vocab = {"the": 0, "rare": 1}
    idf = np.array([0.1, 9.0])
    out = clip_with_quantile_thresholds(
        terms=["the", "rare", "the", "the"],
        vocabulary=vocab,
        idf=idf,
        low_threshold=0.5,
        high_threshold=5.0,
    )
    assert out == ["rare", "rare"]


# --- Composition smoke (R1 then R3) -----------------------------------------

def test_r1_then_r3_pipeline_smoke():
    """End-to-end shape check: R1 expand -> R3 clip preserves the
    documented stack ordering. Real bench numbers come from PR-2
    integration; this is a wiring smoke check only."""
    raw = "Foo bar Baz quux"
    base = ["foo", "bar", "baz", "quux"]
    after_r1 = expand_with_capitalised_entities(raw, base)
    assert after_r1 == ["foo", "bar", "baz", "quux", "foo", "foo", "baz", "baz"]

    vocab = {"foo": 0, "bar": 1, "baz": 2, "quux": 3}
    idf = np.array([2.0, 0.1, 5.0, 1.0])
    after_r3 = clip_with_quantile_thresholds(
        after_r1, vocab, idf, low_threshold=0.5, high_threshold=4.0,
    )
    # "bar" -- IDF 0.1 < 0.5 -> dropped (every occurrence).
    # "baz" -- 3 occurrences in after_r1 (1 base + 2 from R1 boost),
    #          IDF 5.0 > 4.0 -> each emitted boost_qf=2 times = 6.
    # "foo" -- 3 occurrences (1 base + 2 R1), IDF 2.0 mid-band = 3.
    # "quux" -- 1 occurrence (lowercase in raw, no R1 boost), mid = 1.
    assert "bar" not in after_r3
    assert after_r3.count("baz") == 6
    assert after_r3.count("foo") == 3
    assert after_r3.count("quux") == 1
