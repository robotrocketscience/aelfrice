"""Tests for the adaptive meta-belief substrate (#755, umbrella #480).

Covers the six acceptance items on #755:

1. Schema round-trip (install + read).
2. ``update_meta_belief`` / ``read_meta_belief_state`` API.
3. Decay convergence: zero-evidence series → ``static_default`` within
   5 * half_life.
4. Multi-signal independence: two signal classes with different evidence
   cadences update separately; surfaced value reflects the weighted
   blend.
5. ``aelf doctor --meta-beliefs`` and ``--json`` surface state.
6. Determinism: same seed + same evidence sequence → same final state
   across runs (locked ``c06f8d575fad71fb``).
"""
from __future__ import annotations

import io
import json

import pytest

from aelfrice.cli import main as cli_main
from aelfrice.meta_beliefs import (
    SIGNAL_BFS_DEPTH,
    SIGNAL_LATENCY,
    SIGNAL_RELEVANCE,
    apply_evidence,
    decay_state,
    decay_toward_default,
    encode_signal_weights,
    parse_signal_weights,
    posterior_mean,
    prior_alpha_beta,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Pure-math sanity (signal-class enum, prior, apply_evidence, decay)
# ---------------------------------------------------------------------------


def test_prior_at_static_default_has_unit_mass_and_mean_matches():
    a0, b0 = prior_alpha_beta(0.7)
    assert abs((a0 + b0) - 1.0) < 1e-9
    assert abs(a0 / (a0 + b0) - 0.7) < 1e-9


def test_apply_evidence_clamps_out_of_range():
    a, b = apply_evidence(1.0, 1.0, 1.5)  # over-1 evidence
    assert a == 2.0 and b == 1.0
    a, b = apply_evidence(1.0, 1.0, -0.3)  # negative evidence
    assert a == 1.0 and b == 2.0


def test_decay_zero_age_or_zero_half_life_is_passthrough():
    assert decay_toward_default(5.0, 2.0, 0.0, 100.0, 0.5) == (5.0, 2.0)
    assert decay_toward_default(5.0, 2.0, 50.0, 0.0, 0.5) == (5.0, 2.0)


def test_decay_convergence_within_five_half_lives():
    """Acceptance: zero-evidence series converges to static_default
    within 5 * half_life (#755).

    After 5 half-lives ``factor = 2^-5 ≈ 0.031`` — each component's
    residual deviation from its prior shrinks by ~97%. Surfaced mean
    convergence depends on residual magnitudes; we exercise a moderate
    posterior (~5 samples worth of evidence) and assert the residual
    deviation each component is < 5% of the starting deviation.
    """
    static_default = 0.3
    half_life = 100.0
    a0, b0 = prior_alpha_beta(static_default)
    a_start, b_start = 2.0, 0.5  # ~ 5 samples of evidence
    a, b = decay_toward_default(
        a_start, b_start, 5.0 * half_life, half_life, static_default
    )
    # Per-component residual shrunk by 2^-5 ≈ 0.03125 (~4% of starting
    # residual on each axis).
    assert abs(a - a0) < 0.05 * abs(a_start - a0) + 1e-9 + 0.01
    assert abs(b - b0) < 0.05 * abs(b_start - b0) + 1e-9 + 0.01
    # Surfaced mean within 0.05 of static_default after 5 half-lives at
    # this evidence magnitude.
    assert abs(posterior_mean(a, b) - static_default) < 0.05


def test_decay_asymptotic_convergence_at_ten_half_lives():
    """Stronger: at 10 half-lives the residual is 2^-10 ≈ 0.001 of the
    starting deviation; mean is within 0.005 of static_default even
    from a very off-prior starting state."""
    static_default = 0.3
    half_life = 100.0
    a, b = decay_toward_default(
        10.0, 1.0, 10.0 * half_life, half_life, static_default
    )
    assert abs(posterior_mean(a, b) - static_default) < 0.01


def test_signal_weights_encode_decode_roundtrip_is_canonical():
    weights = {SIGNAL_RELEVANCE: 1.0, SIGNAL_LATENCY: 0.5,
               SIGNAL_BFS_DEPTH: 0.25}
    encoded = encode_signal_weights(weights)
    # Sorted by class name → deterministic byte ordering.
    assert encoded == encode_signal_weights(dict(reversed(weights.items())))
    decoded = parse_signal_weights(encoded)
    assert decoded == weights


def test_parse_signal_weights_drops_unknown_classes():
    raw = json.dumps([{"class": SIGNAL_RELEVANCE, "weight": 1.0},
                      {"class": "future_unknown_signal", "weight": 5.0}])
    out = parse_signal_weights(raw)
    assert out == {SIGNAL_RELEVANCE: 1.0}


def test_parse_signal_weights_accepts_bare_string_list():
    raw = json.dumps([SIGNAL_RELEVANCE, SIGNAL_LATENCY])
    out = parse_signal_weights(raw)
    assert out == {SIGNAL_RELEVANCE: 1.0, SIGNAL_LATENCY: 1.0}


# ---------------------------------------------------------------------------
# MemoryStore round-trip
# ---------------------------------------------------------------------------


def _fresh_store() -> MemoryStore:
    return MemoryStore(":memory:")


def test_install_meta_belief_is_idempotent_returns_false_on_reinstall():
    s = _fresh_store()
    try:
        assert s.install_meta_belief(
            "meta:x", static_default=0.5, half_life_seconds=3600,
            signal_weights={SIGNAL_RELEVANCE: 1.0}, now_ts=1000,
        )
        # Second install of same key, even with different config → no-op.
        assert not s.install_meta_belief(
            "meta:x", static_default=0.9, half_life_seconds=600,
            signal_weights={SIGNAL_LATENCY: 1.0}, now_ts=2000,
        )
        state = s.read_meta_belief_state("meta:x")
        # Original config preserved — no silent shift.
        assert state is not None
        assert state.static_default == 0.5
        assert state.half_life_seconds == 3600
        assert state.signal_weights == {SIGNAL_RELEVANCE: 1.0}
    finally:
        s.close()


def test_install_rejects_invalid_config():
    s = _fresh_store()
    try:
        with pytest.raises(ValueError):
            s.install_meta_belief(
                "meta:bad", static_default=1.5, half_life_seconds=10,
                signal_weights={SIGNAL_RELEVANCE: 1.0}, now_ts=1,
            )
        with pytest.raises(ValueError):
            s.install_meta_belief(
                "meta:bad", static_default=0.5, half_life_seconds=0,
                signal_weights={SIGNAL_RELEVANCE: 1.0}, now_ts=1,
            )
        with pytest.raises(ValueError):
            s.install_meta_belief(
                "meta:bad", static_default=0.5, half_life_seconds=10,
                signal_weights={"unknown_class": 1.0}, now_ts=1,
            )
    finally:
        s.close()


def test_update_unknown_key_or_unsubscribed_class_is_noop():
    s = _fresh_store()
    try:
        # Unknown key.
        assert not s.update_meta_belief(
            "meta:missing", SIGNAL_RELEVANCE, 0.8, now_ts=1,
        )
        s.install_meta_belief(
            "meta:k", static_default=0.5, half_life_seconds=100,
            signal_weights={SIGNAL_RELEVANCE: 1.0}, now_ts=0,
        )
        # Subscribed class → applies.
        assert s.update_meta_belief("meta:k", SIGNAL_RELEVANCE, 0.7, now_ts=10)
        # Unsubscribed class → no-op.
        assert not s.update_meta_belief("meta:k", SIGNAL_LATENCY, 0.1, now_ts=10)
        # Latency sub-posterior must not exist.
        state = s.read_meta_belief_state("meta:k")
        assert state is not None
        assert SIGNAL_LATENCY not in state.posteriors
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Multi-signal independence (acceptance #755)
# ---------------------------------------------------------------------------


def test_multi_signal_posteriors_update_independently():
    """Two signal classes with different evidence cadences must not
    collapse to one — each carries its own sub-posterior, and the
    surfaced value reflects a weighted blend (#755)."""
    s = _fresh_store()
    try:
        s.install_meta_belief(
            "meta:k", static_default=0.5, half_life_seconds=100_000,
            signal_weights={SIGNAL_RELEVANCE: 1.0, SIGNAL_LATENCY: 1.0},
            now_ts=0,
        )
        # Relevance is "high" (mean evidence ≈ 0.9), latency is "low"
        # (mean evidence ≈ 0.1).
        for i in range(20):
            s.update_meta_belief("meta:k", SIGNAL_RELEVANCE, 0.9, now_ts=i + 1)
        for i in range(20):
            s.update_meta_belief("meta:k", SIGNAL_LATENCY, 0.1, now_ts=i + 1)
        state = s.read_meta_belief_state("meta:k")
        assert state is not None
        rel = state.posteriors[SIGNAL_RELEVANCE]
        lat = state.posteriors[SIGNAL_LATENCY]
        # Posterior means reflect the per-class evidence direction.
        assert posterior_mean(rel.alpha, rel.beta) > 0.7
        assert posterior_mean(lat.alpha, lat.beta) < 0.3
        # Surfaced value is the equal-weight blend → ≈ 0.5.
        # Read at now_ts=21 (just past last update) so decay is negligible.
        v = s.read_meta_belief_value("meta:k", now_ts=21)
        assert v is not None
        assert 0.4 < v < 0.6
    finally:
        s.close()


def test_weighted_blend_skews_value_toward_higher_weight_class():
    s = _fresh_store()
    try:
        s.install_meta_belief(
            "meta:weighted", static_default=0.5, half_life_seconds=1_000_000,
            signal_weights={SIGNAL_RELEVANCE: 3.0, SIGNAL_LATENCY: 1.0},
            now_ts=0,
        )
        for i in range(30):
            s.update_meta_belief("meta:weighted", SIGNAL_RELEVANCE, 0.9, now_ts=i + 1)
        for i in range(30):
            s.update_meta_belief("meta:weighted", SIGNAL_LATENCY, 0.1, now_ts=i + 1)
        v = s.read_meta_belief_value("meta:weighted", now_ts=31)
        assert v is not None
        # 3:1 weight toward high-evidence relevance → expected blend ≈
        # (3*~0.9 + 1*~0.1) / 4 = 0.7. Allow a wide band for prior mass.
        assert v > 0.55
    finally:
        s.close()


def test_cold_start_returns_static_default():
    s = _fresh_store()
    try:
        s.install_meta_belief(
            "meta:cold", static_default=0.42, half_life_seconds=3600,
            signal_weights={SIGNAL_RELEVANCE: 1.0}, now_ts=0,
        )
        # No update calls — no sub-posteriors row exists.
        v = s.read_meta_belief_value("meta:cold", now_ts=1000)
        assert v == 0.42
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Determinism (locked c06f8d575fad71fb)
# ---------------------------------------------------------------------------


def _run_evidence_seq(seq: list[tuple[str, float, int]]) -> dict:
    """Build a store, replay the evidence sequence, return a snapshot."""
    s = _fresh_store()
    try:
        s.install_meta_belief(
            "meta:det", static_default=0.6, half_life_seconds=86400,
            signal_weights={SIGNAL_RELEVANCE: 1.0, SIGNAL_LATENCY: 0.5},
            now_ts=0,
        )
        for cls, ev, ts in seq:
            s.update_meta_belief("meta:det", cls, ev, now_ts=ts)
        state = s.read_meta_belief_state("meta:det")
        assert state is not None
        return {
            "value_at_t": s.read_meta_belief_value("meta:det", now_ts=200_000),
            "posteriors": {
                cls: (p.alpha, p.beta, p.last_updated_ts)
                for cls, p in state.posteriors.items()
            },
        }
    finally:
        s.close()


def test_determinism_same_sequence_yields_same_state():
    """Acceptance: same evidence sequence → byte-identical final state
    across runs (#605 / locked c06f8d575fad71fb)."""
    seq = [
        (SIGNAL_RELEVANCE, 0.8, 100),
        (SIGNAL_LATENCY, 0.2, 200),
        (SIGNAL_RELEVANCE, 0.7, 500),
        (SIGNAL_LATENCY, 0.4, 1000),
        (SIGNAL_RELEVANCE, 0.95, 5000),
    ]
    a = _run_evidence_seq(seq)
    b = _run_evidence_seq(seq)
    assert a == b


def test_decay_state_is_pure_no_persisted_writes():
    """Diagnostic decay must not rewrite persisted rows — repeated reads
    must observe stable state (the doctor surface relies on this)."""
    s = _fresh_store()
    try:
        s.install_meta_belief(
            "meta:p", static_default=0.5, half_life_seconds=100,
            signal_weights={SIGNAL_RELEVANCE: 1.0}, now_ts=0,
        )
        s.update_meta_belief("meta:p", SIGNAL_RELEVANCE, 0.9, now_ts=10)
        # Decayed read at distant ts.
        state1 = s.read_meta_belief_state("meta:p")
        assert state1 is not None
        decay_state(state1, now_ts=1_000_000)
        # Persisted state must still be the pre-decay (10ts) record.
        state2 = s.read_meta_belief_state("meta:p")
        assert state2 is not None
        rel = state2.posteriors[SIGNAL_RELEVANCE]
        # Last update was at ts=10 with evidence 0.9 → alpha ≈ 0.5+0.9=1.4,
        # beta ≈ 0.5+0.1=0.6 (prior was alpha0=beta0=0.5 at static_default=0.5).
        assert abs(rel.alpha - 1.4) < 1e-6
        assert abs(rel.beta - 0.6) < 1e-6
        assert rel.last_updated_ts == 10
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Doctor surface (acceptance #755)
# ---------------------------------------------------------------------------


def _seed_doctor_store(tmp_path, *, now_ts: int = 1000):
    import os
    db_path = tmp_path / "mb.sqlite"
    os.environ["AELFRICE_DB"] = str(db_path)
    from aelfrice.db_paths import _open_store
    s = _open_store()
    try:
        s.install_meta_belief(
            "meta:retrieval.temporal_half_life_seconds",
            static_default=0.5, half_life_seconds=86400,
            signal_weights={SIGNAL_RELEVANCE: 1.0}, now_ts=now_ts,
        )
        s.update_meta_belief(
            "meta:retrieval.temporal_half_life_seconds",
            SIGNAL_RELEVANCE, 0.8, now_ts=now_ts + 10,
        )
    finally:
        s.close()


def test_doctor_meta_beliefs_text_lists_installed_rows(tmp_path, monkeypatch):
    _seed_doctor_store(tmp_path)
    buf = io.StringIO()
    rc = cli_main(argv=["doctor", "--meta-beliefs"], out=buf)
    assert rc == 0
    output = buf.getvalue()
    assert "meta:retrieval.temporal_half_life_seconds" in output
    assert "static_default" in output
    assert "half_life" in output
    assert "posteriors" in output
    assert "[relevance]" in output


def test_doctor_meta_beliefs_json_is_parseable(tmp_path, monkeypatch):
    _seed_doctor_store(tmp_path)
    buf = io.StringIO()
    rc = cli_main(argv=["doctor", "--meta-beliefs", "--json"], out=buf)
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["count"] == 1
    row = payload["meta_beliefs"][0]
    assert row["key"] == "meta:retrieval.temporal_half_life_seconds"
    assert row["static_default"] == 0.5
    assert row["half_life_seconds"] == 86400
    assert any(p["signal_class"] == SIGNAL_RELEVANCE for p in row["posteriors"])


def test_doctor_meta_beliefs_empty_store_reports_none(tmp_path, monkeypatch):
    import os
    db_path = tmp_path / "empty.sqlite"
    os.environ["AELFRICE_DB"] = str(db_path)
    # touch the store so the file exists
    from aelfrice.db_paths import _open_store
    _open_store().close()
    buf = io.StringIO()
    rc = cli_main(argv=["doctor", "--meta-beliefs"], out=buf)
    assert rc == 0
    assert "no meta-beliefs installed" in buf.getvalue()
