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


# --- would_fire_p1 (policy-agnostic shadow predicate) ---------------------


def test_would_fire_p1_disabled_returns_reason() -> None:
    cfg = cadence.CadenceConfig()  # default: enabled=False
    fired, reason = cadence.would_fire_p1(fire_idx=15, config=cfg)
    assert fired is False
    assert "disabled" in reason


def test_would_fire_p1_k_zero_or_negative_returns_reason() -> None:
    for bad_k in (0, -1, -15):
        cfg = cadence.CadenceConfig(
            enabled=True,
            policy=cadence.POLICY_P1_EVERY_K_TURNS,
            k=bad_k,
        )
        fired, reason = cadence.would_fire_p1(fire_idx=15, config=cfg)
        assert fired is False
        assert f"k={bad_k}" in reason


def test_would_fire_p1_zero_or_negative_fire_idx_returns_reason() -> None:
    cfg = _on_cfg(k=15)
    for bad_idx in (0, -1, -100):
        fired, reason = cadence.would_fire_p1(fire_idx=bad_idx, config=cfg)
        assert fired is False
        assert f"fire_idx={bad_idx}" in reason


def test_would_fire_p1_not_divisible_returns_remainder_reason() -> None:
    cfg = _on_cfg(k=15)
    fired, reason = cadence.would_fire_p1(fire_idx=7, config=cfg)
    assert fired is False
    assert "fire_idx=7" in reason
    assert "k=15" in reason
    assert "7" in reason  # remainder


def test_would_fire_p1_divisible_returns_true_with_reason() -> None:
    cfg = _on_cfg(k=15)
    for idx in (15, 30, 45, 150):
        fired, reason = cadence.would_fire_p1(fire_idx=idx, config=cfg)
        assert fired is True
        assert f"fire_idx={idx}" in reason
        assert "k=15" in reason


def test_would_fire_p1_ignores_policy_field() -> None:
    # The whole point of would_fire_p1 vs should_fire: shadow-mode
    # evaluates "would P1 fire if it WERE selected", so the predicate
    # does not gate on config.policy. Same config with three different
    # policies; the verdict only depends on (enabled, k, fire_idx).
    for policy in (
        cadence.POLICY_OFF,
        cadence.POLICY_P1_EVERY_K_TURNS,
        cadence.POLICY_P2_CTX_THRESHOLD,
    ):
        cfg = cadence.CadenceConfig(enabled=True, policy=policy, k=15)
        fired, _ = cadence.would_fire_p1(fire_idx=15, config=cfg)
        assert fired is True, f"policy={policy!r} should not block P1 predicate"


def test_would_fire_p1_determinism_replay() -> None:
    # Same inputs -> same (bool, str) tuple, character-for-character.
    cfg = _on_cfg(k=15)
    inputs = [0, 1, 14, 15, 16, 29, 30, 31, 45]
    first = [cadence.would_fire_p1(fire_idx=i, config=cfg) for i in inputs]
    second = [cadence.would_fire_p1(fire_idx=i, config=cfg) for i in inputs]
    assert first == second


def test_should_fire_delegates_to_would_fire_p1() -> None:
    # Behavior invariant: should_fire returns True iff would_fire_p1's
    # bool is True AND config.policy == p1_every_k_turns.
    cfg_p1 = _on_cfg(k=5)
    cfg_off = cadence.CadenceConfig(
        enabled=True, policy=cadence.POLICY_OFF, k=5,
    )
    for idx in range(0, 20):
        would, _ = cadence.would_fire_p1(fire_idx=idx, config=cfg_p1)
        assert cadence.should_fire(idx, cfg_p1) is would
        # Same idx, policy=off -> should_fire False even when would_fire_p1 True
        assert cadence.should_fire(idx, cfg_off) is False



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

# --- resolve_cadence_shadow_mode_enabled (#875) ---------------------------


def test_resolve_shadow_mode_default_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AELFRICE_CADENCE_SHADOW_MODE_ENABLED", raising=False)
    monkeypatch.chdir(tmp_path)
    assert cadence.resolve_cadence_shadow_mode_enabled() is False


def test_resolve_shadow_mode_env_wins_over_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _write_toml(tmp_path, "[cadence]\nshadow_mode_enabled = false\n")
    monkeypatch.setenv("AELFRICE_CADENCE_SHADOW_MODE_ENABLED", "true")
    monkeypatch.chdir(tmp_path)
    assert cadence.resolve_cadence_shadow_mode_enabled() is True


def test_resolve_shadow_mode_kwarg_wins_over_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _write_toml(tmp_path, "[cadence]\nshadow_mode_enabled = false\n")
    monkeypatch.delenv("AELFRICE_CADENCE_SHADOW_MODE_ENABLED", raising=False)
    monkeypatch.chdir(tmp_path)
    assert cadence.resolve_cadence_shadow_mode_enabled(explicit=True) is True


def test_resolve_shadow_mode_env_beats_kwarg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("AELFRICE_CADENCE_SHADOW_MODE_ENABLED", "false")
    monkeypatch.chdir(tmp_path)
    assert cadence.resolve_cadence_shadow_mode_enabled(explicit=True) is False


def test_resolve_shadow_mode_unparseable_env_falls_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _write_toml(tmp_path, "[cadence]\nshadow_mode_enabled = true\n")
    monkeypatch.setenv("AELFRICE_CADENCE_SHADOW_MODE_ENABLED", "garbage-value")
    monkeypatch.chdir(tmp_path)
    # Env is unparseable -> falls through to TOML, which says true.
    assert cadence.resolve_cadence_shadow_mode_enabled() is True


def test_load_config_shadow_mode_true(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _write_toml(tmp_path, "[cadence]\nshadow_mode_enabled = true\n")
    monkeypatch.chdir(tmp_path)
    cfg = cadence.load_cadence_config()
    assert cfg.shadow_mode_enabled is True


def test_load_config_shadow_mode_wrong_type_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_toml(tmp_path, '[cadence]\nshadow_mode_enabled = "yes"\n')
    monkeypatch.chdir(tmp_path)
    cfg = cadence.load_cadence_config()
    assert cfg.shadow_mode_enabled is False  # falls back to default
    captured = capsys.readouterr()
    assert "shadow_mode_enabled" in captured.err

# --- Shadow-log writer (#875) ---------------------------------------------


def test_shadow_log_path_layout() -> None:
    d = Path("/some/proj/.git/aelfrice")
    path = cadence.shadow_log_path(
        project_aelfrice_dir=d, session_id="abc-123",
    )
    assert path == Path("/some/proj/.git/aelfrice/cadence_shadow/abc-123.jsonl")


def test_shadow_log_path_does_not_touch_disk(tmp_path: Path) -> None:
    path = cadence.shadow_log_path(
        project_aelfrice_dir=tmp_path, session_id="x",
    )
    # Resolving the path must not create any directories.
    assert not (tmp_path / cadence.CADENCE_SHADOW_DIRNAME).exists()
    assert not path.exists()


def test_format_shadow_row_schema_and_newline() -> None:
    import json as _json
    line = cadence.format_shadow_row(
        session_id="sid",
        selected_policy="p2_ctx_threshold",
        fired=True,
        shadow={
            "p1_every_k_turns": {"would_fire": False, "reason": "r1"},
            "p2_ctx_threshold": {"would_fire": True, "reason": "r2"},
        },
        now="2026-05-20T23:59:00Z",
    )
    assert line.endswith("\n")
    parsed = _json.loads(line.rstrip("\n"))
    assert parsed == {
        "ts": "2026-05-20T23:59:00Z",
        "session_id": "sid",
        "selected": "p2_ctx_threshold",
        "fired": True,
        "shadow": {
            "p1_every_k_turns": {"would_fire": False, "reason": "r1"},
            "p2_ctx_threshold": {"would_fire": True, "reason": "r2"},
        },
    }


def test_format_shadow_row_determinism() -> None:
    # Same inputs -> identical string, byte-for-byte.
    args = dict(
        session_id="sid",
        selected_policy="p1_every_k_turns",
        fired=False,
        shadow={"p1_every_k_turns": {"would_fire": False, "reason": "x"}},
        now="2026-05-20T00:00:00Z",
    )
    a = cadence.format_shadow_row(**args)  # type: ignore[arg-type]
    b = cadence.format_shadow_row(**args)  # type: ignore[arg-type]
    assert a == b


def test_format_shadow_row_extra_keys_passthrough() -> None:
    import json as _json
    line = cadence.format_shadow_row(
        session_id="sid",
        selected_policy="off",
        fired=False,
        shadow={
            "p3_turn_density": {
                "would_fire": True,
                "reason": "density=0.92",
                "density": 0.92,
                "window_size": 10,
            },
        },
        now="2026-05-20T00:00:00Z",
    )
    parsed = _json.loads(line.rstrip("\n"))
    p3 = parsed["shadow"]["p3_turn_density"]
    assert p3["density"] == 0.92
    assert p3["window_size"] == 10


def test_append_shadow_row_creates_parent_and_appends(tmp_path: Path) -> None:
    lp = cadence.shadow_log_path(
        project_aelfrice_dir=tmp_path, session_id="s",
    )
    cadence.append_shadow_row(log_path=lp, row_line='{"a":1}\n')
    cadence.append_shadow_row(log_path=lp, row_line='{"a":2}\n')
    assert lp.exists()
    assert lp.read_text() == '{"a":1}\n{"a":2}\n'


def test_append_shadow_row_failsoft_on_unwritable(tmp_path: Path) -> None:
    # Make the parent of the cadence_shadow dir unwritable. The
    # function must NOT raise; it must swallow the OSError.
    shadow_dir = tmp_path / "cadence_shadow"
    # Pre-create as a *file* so mkdir(parents=True, exist_ok=True)
    # fails with FileExistsError (a subclass of OSError).
    shadow_dir.write_text("not a dir")
    lp = shadow_dir / "s.jsonl"
    cadence.append_shadow_row(log_path=lp, row_line='{"a":1}\n')
    # No raise -> pass. File-as-dir blocks any write.
    assert shadow_dir.read_text() == "not a dir"
