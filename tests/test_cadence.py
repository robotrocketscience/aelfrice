"""Unit tests for src/aelfrice/cadence.py (#749 P1 every-K-turns)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from aelfrice import cadence


# --- should_fire predicate (pure function) --------------------------------


def _on_cfg(k: int = 15) -> cadence.CadenceConfig:
    return cadence.CadenceConfig(
        enabled=True,
        policy=cadence.POLICY_P1_EVERY_K_TURNS,
        k=k,
    )


def test_should_fire_default_config_never_fires() -> None:
    cfg = cadence.CadenceConfig()
    assert cfg.enabled is False
    for fire_idx in range(0, 50):
        assert cadence.should_fire(fire_idx, cfg) is False


def test_should_fire_disabled_never_fires_regardless_of_policy_and_k() -> None:
    cfg = cadence.CadenceConfig(
        enabled=False,
        policy=cadence.POLICY_P1_EVERY_K_TURNS,
        k=1,
    )
    for fire_idx in range(0, 50):
        assert cadence.should_fire(fire_idx, cfg) is False


def test_should_fire_policy_off_never_fires_even_when_enabled() -> None:
    cfg = cadence.CadenceConfig(
        enabled=True, policy=cadence.POLICY_OFF, k=15,
    )
    for fire_idx in range(0, 50):
        assert cadence.should_fire(fire_idx, cfg) is False


def test_should_fire_p1_k15_fires_at_multiples_of_k() -> None:
    cfg = _on_cfg(k=15)
    expected = {15, 30, 45}
    for fire_idx in range(1, 50):
        assert cadence.should_fire(fire_idx, cfg) is (fire_idx in expected), (
            f"fire_idx={fire_idx} should be {fire_idx in expected}"
        )


def test_should_fire_zero_fire_idx_suppresses_spurious_cold_start() -> None:
    # 0 % k == 0 is True numerically; the predicate must reject it so
    # the very first Stop after session start does not fire cadence.
    assert cadence.should_fire(0, _on_cfg(k=15)) is False
    assert cadence.should_fire(0, _on_cfg(k=1)) is False


def test_should_fire_negative_fire_idx_never_fires() -> None:
    for fire_idx in (-1, -10, -100):
        assert cadence.should_fire(fire_idx, _on_cfg(k=15)) is False


def test_should_fire_k_zero_or_negative_never_fires() -> None:
    cfg_zero = cadence.CadenceConfig(
        enabled=True, policy=cadence.POLICY_P1_EVERY_K_TURNS, k=0,
    )
    cfg_neg = cadence.CadenceConfig(
        enabled=True, policy=cadence.POLICY_P1_EVERY_K_TURNS, k=-5,
    )
    for fire_idx in range(0, 30):
        assert cadence.should_fire(fire_idx, cfg_zero) is False
        assert cadence.should_fire(fire_idx, cfg_neg) is False


def test_should_fire_k1_fires_every_turn() -> None:
    cfg = _on_cfg(k=1)
    for fire_idx in range(1, 20):
        assert cadence.should_fire(fire_idx, cfg) is True


# --- load_cadence_config (TOML parser) ------------------------------------


def _write_toml(dirpath: Path, body: str) -> None:
    (dirpath / cadence.CONFIG_FILENAME).write_text(textwrap.dedent(body))


def test_load_config_missing_file_returns_defaults(tmp_path: Path) -> None:
    cfg = cadence.load_cadence_config(start=tmp_path)
    assert cfg == cadence.CadenceConfig()


def test_load_config_missing_section_returns_defaults(tmp_path: Path) -> None:
    _write_toml(tmp_path, """
        [retrieval]
        posterior_weight = 0.5
    """)
    cfg = cadence.load_cadence_config(start=tmp_path)
    assert cfg == cadence.CadenceConfig()


def test_load_config_full_section(tmp_path: Path) -> None:
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 20
    """)
    cfg = cadence.load_cadence_config(start=tmp_path)
    assert cfg.enabled is True
    assert cfg.policy == cadence.POLICY_P1_EVERY_K_TURNS
    assert cfg.k == 20


def test_load_config_partial_section_uses_defaults_for_missing_fields(
    tmp_path: Path,
) -> None:
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
    """)
    cfg = cadence.load_cadence_config(start=tmp_path)
    assert cfg.enabled is True
    assert cfg.policy == cadence.DEFAULT_POLICY  # not flipped by enabled
    assert cfg.k == cadence.DEFAULT_K


def test_load_config_malformed_toml_returns_defaults(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / cadence.CONFIG_FILENAME).write_text("[[cadence broken")
    cfg = cadence.load_cadence_config(start=tmp_path)
    assert cfg == cadence.CadenceConfig()
    err = capsys.readouterr().err
    assert "malformed TOML" in err


def test_load_config_wrong_typed_enabled_ignored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _write_toml(tmp_path, """
        [cadence]
        enabled = "yes"
        k = 10
    """)
    cfg = cadence.load_cadence_config(start=tmp_path)
    # `enabled = "yes"` is wrong type (we require bool); falls back.
    # k=10 still reads correctly.
    assert cfg.enabled is False
    assert cfg.k == 10
    err = capsys.readouterr().err
    assert "expected bool" in err


def test_load_config_unknown_policy_ignored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p99_warp_drive"
    """)
    cfg = cadence.load_cadence_config(start=tmp_path)
    assert cfg.policy == cadence.DEFAULT_POLICY
    err = capsys.readouterr().err
    assert "p99_warp_drive" not in err  # the bad value isn't echoed
    assert cadence.POLICY_P1_EVERY_K_TURNS in err  # but valid options are


def test_load_config_non_positive_k_ignored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _write_toml(tmp_path, """
        [cadence]
        k = 0
    """)
    cfg = cadence.load_cadence_config(start=tmp_path)
    assert cfg.k == cadence.DEFAULT_K
    err = capsys.readouterr().err
    assert "expected positive int" in err


def test_load_config_k_as_bool_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    # bool is a subclass of int in Python; we reject it explicitly to
    # avoid `k = true` silently meaning k=1.
    _write_toml(tmp_path, """
        [cadence]
        k = true
    """)
    cfg = cadence.load_cadence_config(start=tmp_path)
    assert cfg.k == cadence.DEFAULT_K
    err = capsys.readouterr().err
    assert "expected positive int" in err


def test_load_config_walks_up_parent_dirs(tmp_path: Path) -> None:
    # Write TOML at tmp_path, start the walk from a nested child.
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
        policy = "p1_every_k_turns"
        k = 7
    """)
    child = tmp_path / "a" / "b" / "c"
    child.mkdir(parents=True)
    cfg = cadence.load_cadence_config(start=child)
    assert cfg.enabled is True
    assert cfg.k == 7


# --- Resolvers (env > kwarg > TOML > default) -----------------------------


def test_resolve_enabled_default_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(cadence.ENV_CADENCE_ENABLED, raising=False)
    assert cadence.resolve_cadence_enabled(start=tmp_path) is False


def test_resolve_enabled_env_wins_over_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
    """)
    monkeypatch.setenv(cadence.ENV_CADENCE_ENABLED, "0")
    # TOML says true, env says falsy → env wins.
    assert cadence.resolve_cadence_enabled(start=tmp_path) is False


def test_resolve_enabled_kwarg_wins_over_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
    """)
    monkeypatch.delenv(cadence.ENV_CADENCE_ENABLED, raising=False)
    # Explicit kwarg overrides TOML.
    assert cadence.resolve_cadence_enabled(explicit=False, start=tmp_path) is False


def test_resolve_enabled_env_beats_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cadence.ENV_CADENCE_ENABLED, "1")
    # Env beats kwarg (operator override wins over caller-supplied value).
    assert cadence.resolve_cadence_enabled(explicit=False, start=tmp_path) is True


def test_resolve_enabled_unparseable_env_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cadence.ENV_CADENCE_ENABLED, "maybe")
    _write_toml(tmp_path, """
        [cadence]
        enabled = true
    """)
    # Unparseable env returns None from the env reader; TOML wins.
    assert cadence.resolve_cadence_enabled(start=tmp_path) is True


def test_resolve_k_default_15(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(cadence.ENV_CADENCE_K, raising=False)
    assert cadence.resolve_cadence_k(start=tmp_path) == cadence.DEFAULT_K


def test_resolve_k_env_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cadence.ENV_CADENCE_K, "30")
    _write_toml(tmp_path, """
        [cadence]
        k = 5
    """)
    assert cadence.resolve_cadence_k(start=tmp_path) == 30


def test_resolve_k_env_non_int_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cadence.ENV_CADENCE_K, "fifteen")
    _write_toml(tmp_path, """
        [cadence]
        k = 7
    """)
    assert cadence.resolve_cadence_k(start=tmp_path) == 7


def test_resolve_k_env_non_positive_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cadence.ENV_CADENCE_K, "0")
    _write_toml(tmp_path, """
        [cadence]
        k = 7
    """)
    assert cadence.resolve_cadence_k(start=tmp_path) == 7


def test_resolve_policy_default_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(cadence.ENV_CADENCE_POLICY, raising=False)
    assert cadence.resolve_cadence_policy(start=tmp_path) == cadence.POLICY_OFF


def test_resolve_policy_env_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        cadence.ENV_CADENCE_POLICY, "p1_every_k_turns",
    )
    assert cadence.resolve_cadence_policy(start=tmp_path) == (
        cadence.POLICY_P1_EVERY_K_TURNS
    )


def test_resolve_policy_env_unknown_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(cadence.ENV_CADENCE_POLICY, "warp")
    _write_toml(tmp_path, """
        [cadence]
        policy = "p1_every_k_turns"
    """)
    assert cadence.resolve_cadence_policy(start=tmp_path) == (
        cadence.POLICY_P1_EVERY_K_TURNS
    )
