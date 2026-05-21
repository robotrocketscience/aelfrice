"""Unit tests for #876 P3 cadence policies — enum + resolvers + predicates.

This PR is surface-only — no Stop/UPS dispatcher wiring, no session-ring
state additions. Those land in subsequent PRs of the #876 stack.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from aelfrice.cadence import (
    CONFIG_FILENAME,
    CadenceConfig,
    DEFAULT_P3_SUBSTANTIVE_THRESHOLD,
    DEFAULT_P3_SUBSTANTIVE_WINDOW,
    DEFAULT_P3_VELOCITY_THRESHOLD,
    ENV_CADENCE_P3_SUBSTANTIVE_THRESHOLD,
    ENV_CADENCE_P3_SUBSTANTIVE_WINDOW,
    ENV_CADENCE_P3_VELOCITY_THRESHOLD,
    POLICY_OFF,
    POLICY_P1_EVERY_K_TURNS,
    POLICY_P3_SUBSTANTIVE,
    POLICY_P3_VELOCITY,
    _VALID_POLICIES,
    is_substantive_turn,
    resolve_cadence_p3_substantive_threshold,
    resolve_cadence_p3_substantive_window,
    resolve_cadence_p3_velocity_threshold,
    should_fire_p3_substantive,
    should_fire_p3_velocity,
    would_fire_p3_substantive,
    would_fire_p3_velocity,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        ENV_CADENCE_P3_VELOCITY_THRESHOLD,
        ENV_CADENCE_P3_SUBSTANTIVE_WINDOW,
        ENV_CADENCE_P3_SUBSTANTIVE_THRESHOLD,
    ):
        monkeypatch.delenv(var, raising=False)


def _write_toml(repo: Path, body: str) -> None:
    (repo / CONFIG_FILENAME).write_text(textwrap.dedent(body))


# --- Policy enum -----------------------------------------------------


def test_p3_policies_in_valid_set() -> None:
    assert POLICY_P3_VELOCITY in _VALID_POLICIES
    assert POLICY_P3_SUBSTANTIVE in _VALID_POLICIES
    assert POLICY_P3_VELOCITY == "p3_velocity"
    assert POLICY_P3_SUBSTANTIVE == "p3_substantive"


# --- Resolvers --------------------------------------------------------


def test_velocity_threshold_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_cadence_p3_velocity_threshold() == DEFAULT_P3_VELOCITY_THRESHOLD


def test_velocity_threshold_env_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_toml(tmp_path, "[cadence]\np3_velocity_threshold = 5000\n")
    monkeypatch.setenv(ENV_CADENCE_P3_VELOCITY_THRESHOLD, "7000")
    assert resolve_cadence_p3_velocity_threshold(explicit=4000) == 7000


def test_velocity_threshold_kwarg_over_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_toml(tmp_path, "[cadence]\np3_velocity_threshold = 5000\n")
    assert resolve_cadence_p3_velocity_threshold(explicit=4000) == 4000


def test_velocity_threshold_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_toml(tmp_path, "[cadence]\np3_velocity_threshold = 5000\n")
    assert resolve_cadence_p3_velocity_threshold() == 5000


def test_substantive_window_resolver_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_cadence_p3_substantive_window() == DEFAULT_P3_SUBSTANTIVE_WINDOW
    _write_toml(tmp_path, "[cadence]\np3_substantive_window = 20\n")
    assert resolve_cadence_p3_substantive_window() == 20
    assert resolve_cadence_p3_substantive_window(explicit=15) == 15
    monkeypatch.setenv(ENV_CADENCE_P3_SUBSTANTIVE_WINDOW, "25")
    assert resolve_cadence_p3_substantive_window(explicit=15) == 25


def test_substantive_threshold_resolver_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_cadence_p3_substantive_threshold() == DEFAULT_P3_SUBSTANTIVE_THRESHOLD
    _write_toml(tmp_path, "[cadence]\np3_substantive_threshold = 0.75\n")
    assert resolve_cadence_p3_substantive_threshold() == pytest.approx(0.75)
    assert resolve_cadence_p3_substantive_threshold(explicit=0.5) == 0.5
    monkeypatch.setenv(ENV_CADENCE_P3_SUBSTANTIVE_THRESHOLD, "0.9")
    assert resolve_cadence_p3_substantive_threshold(explicit=0.5) == pytest.approx(0.9)


def test_substantive_threshold_out_of_range_kwarg_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_toml(tmp_path, "[cadence]\np3_substantive_threshold = 0.5\n")
    assert resolve_cadence_p3_substantive_threshold(explicit=1.5) == pytest.approx(0.5)
    assert resolve_cadence_p3_substantive_threshold(explicit=-0.1) == pytest.approx(0.5)


# --- is_substantive_turn ---------------------------------------------


def test_is_substantive_turn_inverts_phase_boundary() -> None:
    assert is_substantive_turn("how does the rebuilder budget work")
    assert not is_substantive_turn("ok thanks")
    assert not is_substantive_turn("next task")
    assert not is_substantive_turn("ship it")


def test_is_substantive_turn_empty_or_none() -> None:
    assert not is_substantive_turn(None)
    assert not is_substantive_turn("")
    assert not is_substantive_turn("   ")


# --- would_fire_p3_velocity ------------------------------------------


def _cfg_velocity(enabled: bool = True, threshold: int = 3000) -> CadenceConfig:
    return CadenceConfig(
        enabled=enabled, policy=POLICY_P3_VELOCITY, p3_velocity_threshold=threshold,
    )


def test_velocity_disabled_no_fire() -> None:
    cfg = _cfg_velocity(enabled=False)
    fired, reason = would_fire_p3_velocity(
        bytes_at_last_fire=0, transcript_bytes=10_000,
        turns_since_last_fire=2, config=cfg,
    )
    assert not fired
    assert "disabled" in reason


def test_velocity_fires_when_density_at_threshold() -> None:
    cfg = _cfg_velocity(threshold=3000)
    fired, reason = would_fire_p3_velocity(
        bytes_at_last_fire=10_000, transcript_bytes=16_000,
        turns_since_last_fire=2, config=cfg,
    )
    assert fired, reason
    assert "bytes/turn" in reason


def test_velocity_no_fire_below_threshold() -> None:
    cfg = _cfg_velocity(threshold=3000)
    fired, reason = would_fire_p3_velocity(
        bytes_at_last_fire=10_000, transcript_bytes=15_000,
        turns_since_last_fire=2, config=cfg,
    )
    assert not fired
    assert "below threshold" in reason


def test_velocity_zero_turns_no_fire() -> None:
    cfg = _cfg_velocity()
    fired, reason = would_fire_p3_velocity(
        bytes_at_last_fire=0, transcript_bytes=10_000,
        turns_since_last_fire=0, config=cfg,
    )
    assert not fired
    assert "turns_since_last_fire=0" in reason


def test_velocity_transcript_shrunk_no_fire() -> None:
    cfg = _cfg_velocity()
    fired, reason = would_fire_p3_velocity(
        bytes_at_last_fire=10_000, transcript_bytes=5_000,
        turns_since_last_fire=1, config=cfg,
    )
    assert not fired
    assert "below" in reason


def test_velocity_threshold_must_be_positive() -> None:
    cfg = _cfg_velocity(threshold=0)
    fired, reason = would_fire_p3_velocity(
        bytes_at_last_fire=0, transcript_bytes=10_000,
        turns_since_last_fire=1, config=cfg,
    )
    assert not fired
    assert "p3_velocity_threshold=0" in reason


def test_should_fire_velocity_policy_gated() -> None:
    cfg_velocity = _cfg_velocity(threshold=1000)
    assert should_fire_p3_velocity(
        bytes_at_last_fire=0, transcript_bytes=2000,
        turns_since_last_fire=1, config=cfg_velocity,
    )
    cfg_other = CadenceConfig(
        enabled=True, policy=POLICY_P1_EVERY_K_TURNS, p3_velocity_threshold=1000,
    )
    assert not should_fire_p3_velocity(
        bytes_at_last_fire=0, transcript_bytes=2000,
        turns_since_last_fire=1, config=cfg_other,
    )


# --- would_fire_p3_substantive ---------------------------------------


def _cfg_substantive(
    enabled: bool = True, window: int = 10, threshold: float = 0.6,
) -> CadenceConfig:
    return CadenceConfig(
        enabled=enabled, policy=POLICY_P3_SUBSTANTIVE,
        p3_substantive_window=window, p3_substantive_threshold=threshold,
    )


def test_substantive_disabled_no_fire() -> None:
    fired, reason = would_fire_p3_substantive(
        substantive_count=8, config=_cfg_substantive(enabled=False),
    )
    assert not fired
    assert "disabled" in reason


def test_substantive_fires_at_threshold() -> None:
    cfg = _cfg_substantive(window=10, threshold=0.6)
    fired, reason = would_fire_p3_substantive(substantive_count=6, config=cfg)
    assert fired, reason
    assert "6/10" in reason


def test_substantive_below_threshold_no_fire() -> None:
    cfg = _cfg_substantive(window=10, threshold=0.6)
    fired, reason = would_fire_p3_substantive(substantive_count=5, config=cfg)
    assert not fired
    assert "below" in reason


def test_substantive_count_exceeds_window_no_fire() -> None:
    cfg = _cfg_substantive(window=10)
    fired, reason = would_fire_p3_substantive(substantive_count=15, config=cfg)
    assert not fired
    assert "outside" in reason


def test_substantive_negative_count_no_fire() -> None:
    cfg = _cfg_substantive(window=10)
    fired, reason = would_fire_p3_substantive(substantive_count=-1, config=cfg)
    assert not fired
    assert "outside" in reason


def test_substantive_threshold_out_of_range_no_fire() -> None:
    cfg = _cfg_substantive(threshold=1.5)
    fired, reason = would_fire_p3_substantive(substantive_count=6, config=cfg)
    assert not fired
    assert "outside" in reason


def test_substantive_zero_window_no_fire() -> None:
    cfg = _cfg_substantive(window=0)
    fired, reason = would_fire_p3_substantive(substantive_count=0, config=cfg)
    assert not fired
    assert "p3_substantive_window=0" in reason


def test_should_fire_substantive_policy_gated() -> None:
    cfg_sub = _cfg_substantive(window=10, threshold=0.6)
    assert should_fire_p3_substantive(substantive_count=7, config=cfg_sub)
    cfg_other = CadenceConfig(
        enabled=True, policy=POLICY_OFF,
        p3_substantive_window=10, p3_substantive_threshold=0.6,
    )
    assert not should_fire_p3_substantive(substantive_count=7, config=cfg_other)


# --- Determinism -----------------------------------------------------


def test_velocity_predicate_is_pure() -> None:
    cfg = _cfg_velocity(threshold=2000)
    inputs = dict(
        bytes_at_last_fire=5000, transcript_bytes=15_000,
        turns_since_last_fire=4, config=cfg,
    )
    r1 = would_fire_p3_velocity(**inputs)
    r2 = would_fire_p3_velocity(**inputs)
    assert r1 == r2


def test_substantive_predicate_is_pure() -> None:
    cfg = _cfg_substantive(window=8, threshold=0.5)
    r1 = would_fire_p3_substantive(substantive_count=4, config=cfg)
    r2 = would_fire_p3_substantive(substantive_count=4, config=cfg)
    assert r1 == r2
