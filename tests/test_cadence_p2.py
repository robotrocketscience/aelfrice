"""Unit tests for P2 cadence — ctx-threshold + phase-boundary (#871).

P1 tests live in :mod:`tests.test_cadence` and are unchanged. This
module covers only the surface introduced by #871.
"""
from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path

import pytest

from aelfrice.cadence import (
    CONFIG_FILENAME,
    DEFAULT_CTX_BYTE_WINDOW,
    DEFAULT_CTX_THRESHOLD,
    DEFAULT_K,
    DEFAULT_POLICY,
    ENV_CADENCE_CTX_BYTE_WINDOW,
    ENV_CADENCE_CTX_THRESHOLD,
    POLICY_P1_EVERY_K_TURNS,
    POLICY_P2_CTX_THRESHOLD,
    CadenceConfig,
    estimate_transcript_bytes,
    is_phase_boundary_signal,
    load_cadence_config,
    read_last_user_prompt,
    resolve_cadence_ctx_byte_window,
    resolve_cadence_ctx_threshold,
    should_fire,
    should_fire_p2,
    would_fire_p1,
    would_fire_p2,
)


# --- Fixtures -------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    for var in (
        "AELFRICE_CADENCE_ENABLED",
        "AELFRICE_CADENCE_POLICY",
        "AELFRICE_CADENCE_K",
        ENV_CADENCE_CTX_THRESHOLD,
        ENV_CADENCE_CTX_BYTE_WINDOW,
    ):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    return tmp_path


# --- Phase-boundary detector ---------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "done",
        "Done",
        "DONE.",
        "thanks",
        "Thanks!",
        "thank you",
        "ok",
        "okay",
        "ok thanks",
        "Ok, thanks.",
        "okay great",
        "perfect",
        "Perfect.",
        "great",
        "awesome",
        "got it",
        "sounds good",
        "good to go",
        "all good",
        "looks good",
        "next",
        "next task",
        "next step",
        "move on",
        "ship it",
        "merge it",
    ],
)
def test_phase_boundary_exact_match(prompt: str) -> None:
    """Exact-match allowlist entries trigger boundary detection."""
    assert is_phase_boundary_signal(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "switch to PR review",
        "Switch to the database task",
        "let's move on to the next bug",
        "Let's move on to PR #869",
        "lets switch to the iceberg PR",
        "let us switch to a different topic",
        "now let's fix the auth bug",
        "moving on to the test suite",
    ],
)
def test_phase_boundary_prefix_match(prompt: str) -> None:
    """Prefix-allowlist transitions trigger boundary detection."""
    assert is_phase_boundary_signal(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        None,
        "",
        "   ",
        # Substantive prompts that may begin with ack tokens but
        # extend past the 80-char cap or contain new work.
        "thanks for the help, now can you also look at the file at line 42 and explain it carefully",
        "ok but first explain what is happening in the database query before we move on",
        "can you refactor the database query for me",
        "lets investigate this further and figure out the bug in the rebuilder code",
        # Random non-boundary input
        "what does this function do",
        "fix the typo on line 12",
    ],
)
def test_phase_boundary_negative(prompt: str | None) -> None:
    """Long/substantive/empty prompts are not boundaries."""
    assert not is_phase_boundary_signal(prompt)


def test_phase_boundary_max_len_cap() -> None:
    """Boundary signals over 80 normalized chars are rejected."""
    # An ack-like prompt padded to exceed the cap.
    long_ack = "ok thanks " + "x " * 50
    assert len(long_ack) > 80
    assert not is_phase_boundary_signal(long_ack)


def test_phase_boundary_apostrophe_normalization() -> None:
    """Apostrophes are folded — ``let's`` matches the allowlist's
    apostrophe-free entries."""
    assert is_phase_boundary_signal("let's move on to v3.1")
    assert is_phase_boundary_signal("Lets move on to v3.1")


def test_phase_boundary_punctuation_collapse() -> None:
    """Punctuation collapses to whitespace, words preserved."""
    assert is_phase_boundary_signal("ok,thanks!")  # comma → space
    assert is_phase_boundary_signal("ok... thanks.")


def test_phase_boundary_whitespace_only() -> None:
    assert not is_phase_boundary_signal("\n\t  \n")


# --- estimate_transcript_bytes ------------------------------------------


def test_estimate_transcript_bytes_none() -> None:
    assert estimate_transcript_bytes(None) == 0


def test_estimate_transcript_bytes_missing(empty_dir: Path) -> None:
    assert estimate_transcript_bytes(empty_dir / "nope.jsonl") == 0


def test_estimate_transcript_bytes_real(empty_dir: Path) -> None:
    p = empty_dir / "t.jsonl"
    p.write_text("x" * 1234)
    assert estimate_transcript_bytes(p) == 1234


def test_estimate_transcript_bytes_empty(empty_dir: Path) -> None:
    p = empty_dir / "empty.jsonl"
    p.write_text("")
    assert estimate_transcript_bytes(p) == 0


# --- read_last_user_prompt -----------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_read_last_user_prompt_none() -> None:
    assert read_last_user_prompt(None) is None


def test_read_last_user_prompt_missing(empty_dir: Path) -> None:
    assert read_last_user_prompt(empty_dir / "nope.jsonl") is None


def test_read_last_user_prompt_inlined_role(empty_dir: Path) -> None:
    """Schema variant: role/content at top level, no `message` wrapper."""
    p = empty_dir / "t.jsonl"
    _write_jsonl(p, [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "last user line"},
    ])
    assert read_last_user_prompt(p) == "last user line"


def test_read_last_user_prompt_wrapped_message(empty_dir: Path) -> None:
    """Schema variant: role/content under `message` key."""
    p = empty_dir / "t.jsonl"
    _write_jsonl(p, [
        {"message": {"role": "user", "content": "first"}},
        {"message": {"role": "assistant", "content": "reply"}},
        {"message": {"role": "user", "content": "second user"}},
    ])
    assert read_last_user_prompt(p) == "second user"


def test_read_last_user_prompt_content_blocks(empty_dir: Path) -> None:
    """Claude API content list-of-blocks format."""
    p = empty_dir / "t.jsonl"
    _write_jsonl(p, [
        {"message": {"role": "user", "content": [
            {"type": "text", "text": "block one"},
            {"type": "text", "text": "block two"},
        ]}},
    ])
    assert read_last_user_prompt(p) == "block one\nblock two"


def test_read_last_user_prompt_no_user_turns(empty_dir: Path) -> None:
    p = empty_dir / "t.jsonl"
    _write_jsonl(p, [
        {"message": {"role": "assistant", "content": "only assistant"}},
    ])
    assert read_last_user_prompt(p) is None


def test_read_last_user_prompt_malformed_lines_tolerated(empty_dir: Path) -> None:
    """Non-JSON / empty lines are skipped without breaking parse."""
    p = empty_dir / "t.jsonl"
    with p.open("w") as f:
        f.write("not json\n")
        f.write("\n")
        f.write(json.dumps({"role": "user", "content": "good"}) + "\n")
        f.write("garbage{{{\n")
    assert read_last_user_prompt(p) == "good"


def test_read_last_user_prompt_skips_non_text_blocks(empty_dir: Path) -> None:
    """tool_use / tool_result blocks in content list are ignored."""
    p = empty_dir / "t.jsonl"
    _write_jsonl(p, [
        {"message": {"role": "user", "content": [
            {"type": "tool_result", "content": "ignore me"},
            {"type": "text", "text": "real content"},
            {"type": "tool_use", "id": "abc"},
        ]}},
    ])
    assert read_last_user_prompt(p) == "real content"


def test_read_last_user_prompt_large_file_tail(empty_dir: Path) -> None:
    """Files >64KB are read from the tail; last user turn still found."""
    p = empty_dir / "big.jsonl"
    with p.open("w") as f:
        # Push past 64KB with assistant noise...
        for _ in range(200):
            f.write(json.dumps({"role": "assistant", "content": "x" * 500}) + "\n")
        # ...then the real last user turn.
        f.write(json.dumps({"role": "user", "content": "tail user"}) + "\n")
    assert p.stat().st_size > 65536
    assert read_last_user_prompt(p) == "tail user"


# --- should_fire_p2 ------------------------------------------------------


def _p2_cfg(**kwargs: object) -> CadenceConfig:
    defaults: dict[str, object] = {
        "enabled": True,
        "policy": POLICY_P2_CTX_THRESHOLD,
        "k": DEFAULT_K,
        "ctx_threshold": 0.50,
        "ctx_byte_window": 600_000,
    }
    defaults.update(kwargs)
    return CadenceConfig(**defaults)  # type: ignore[arg-type]


def test_p2_disabled(empty_dir: Path) -> None:
    p = empty_dir / "t.jsonl"
    p.write_text("x" * 1_000_000)
    cfg = _p2_cfg(enabled=False)
    assert not should_fire_p2(transcript_path=p, last_user_prompt="done", config=cfg)


def test_p2_wrong_policy(empty_dir: Path) -> None:
    p = empty_dir / "t.jsonl"
    p.write_text("x" * 1_000_000)
    cfg = _p2_cfg(policy=POLICY_P1_EVERY_K_TURNS)
    assert not should_fire_p2(transcript_path=p, last_user_prompt="done", config=cfg)


def test_p2_zero_byte_window(empty_dir: Path) -> None:
    p = empty_dir / "t.jsonl"
    p.write_text("x" * 1_000_000)
    cfg = _p2_cfg(ctx_byte_window=0)
    assert not should_fire_p2(transcript_path=p, last_user_prompt="done", config=cfg)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.1, 2.0])
def test_p2_bad_threshold_rejected(empty_dir: Path, bad: float) -> None:
    p = empty_dir / "t.jsonl"
    p.write_text("x" * 1_000_000)
    cfg = _p2_cfg(ctx_threshold=bad)
    assert not should_fire_p2(transcript_path=p, last_user_prompt="done", config=cfg)


def test_p2_below_threshold_no_fire(empty_dir: Path) -> None:
    """Transcript byte-count under watermark — no fire even with boundary."""
    p = empty_dir / "t.jsonl"
    p.write_text("x" * 100)  # tiny
    cfg = _p2_cfg(ctx_threshold=0.5, ctx_byte_window=600_000)  # watermark 300k
    assert not should_fire_p2(transcript_path=p, last_user_prompt="done", config=cfg)


def test_p2_above_threshold_no_boundary(empty_dir: Path) -> None:
    """Transcript byte-count over watermark but non-boundary prompt."""
    p = empty_dir / "t.jsonl"
    p.write_text("x" * 400_000)
    cfg = _p2_cfg(ctx_threshold=0.5, ctx_byte_window=600_000)
    assert not should_fire_p2(
        transcript_path=p,
        last_user_prompt="please refactor this function",
        config=cfg,
    )


def test_p2_above_threshold_with_boundary_fires(empty_dir: Path) -> None:
    """Both conditions met → fire."""
    p = empty_dir / "t.jsonl"
    p.write_text("x" * 400_000)
    cfg = _p2_cfg(ctx_threshold=0.5, ctx_byte_window=600_000)
    assert should_fire_p2(transcript_path=p, last_user_prompt="done", config=cfg)


def test_p2_missing_transcript_no_fire(empty_dir: Path) -> None:
    """No transcript path → 0 bytes → below threshold → no fire."""
    cfg = _p2_cfg(ctx_threshold=0.001, ctx_byte_window=600_000)
    assert not should_fire_p2(
        transcript_path=empty_dir / "missing.jsonl",
        last_user_prompt="done",
        config=cfg,
    )


def test_p2_at_exact_threshold_fires(empty_dir: Path) -> None:
    """Boundary case: bytes == watermark → fire (>= comparison)."""
    p = empty_dir / "t.jsonl"
    p.write_text("x" * 300_000)
    cfg = _p2_cfg(ctx_threshold=0.5, ctx_byte_window=600_000)  # watermark = 300k
    assert should_fire_p2(transcript_path=p, last_user_prompt="done", config=cfg)


def test_p2_threshold_one_full_window(empty_dir: Path) -> None:
    """ctx_threshold = 1.0 — fire only at full window."""
    p = empty_dir / "t.jsonl"
    p.write_text("x" * 600_000)
    cfg = _p2_cfg(ctx_threshold=1.0, ctx_byte_window=600_000)
    assert should_fire_p2(transcript_path=p, last_user_prompt="done", config=cfg)


def test_p2_does_not_fire_under_p1_predicate(empty_dir: Path) -> None:
    """should_fire (P1) returns False for P2 configs even at fire_idx % k == 0."""
    cfg = _p2_cfg()
    assert not should_fire(15, cfg)
    assert not should_fire(30, cfg)


# --- TOML / resolver tests for P2 fields --------------------------------


def _write_toml(dir_: Path, body: str) -> Path:
    p = dir_ / CONFIG_FILENAME
    p.write_text(body)
    return p


def test_load_p2_full_config(empty_dir: Path) -> None:
    _write_toml(empty_dir, (
        "[cadence]\n"
        "enabled = true\n"
        'policy = "p2_ctx_threshold"\n'
        "ctx_threshold = 0.65\n"
        "ctx_byte_window = 800000\n"
    ))
    cfg = load_cadence_config(start=empty_dir)
    assert cfg.enabled is True
    assert cfg.policy == POLICY_P2_CTX_THRESHOLD
    assert cfg.ctx_threshold == pytest.approx(0.65)
    assert cfg.ctx_byte_window == 800_000


def test_load_p2_missing_keys_fall_back_to_defaults(empty_dir: Path) -> None:
    """[cadence] table with only enabled set — P2 defaults apply."""
    _write_toml(empty_dir, "[cadence]\nenabled = true\n")
    cfg = load_cadence_config(start=empty_dir)
    assert cfg.ctx_threshold == DEFAULT_CTX_THRESHOLD
    assert cfg.ctx_byte_window == DEFAULT_CTX_BYTE_WINDOW


def test_load_p2_wrong_typed_threshold(empty_dir: Path, capfd: pytest.CaptureFixture[str]) -> None:
    _write_toml(empty_dir, (
        "[cadence]\n"
        "ctx_threshold = \"sixty percent\"\n"
    ))
    cfg = load_cadence_config(start=empty_dir)
    assert cfg.ctx_threshold == DEFAULT_CTX_THRESHOLD
    captured = capfd.readouterr()
    assert "ctx_threshold" in captured.err


def test_load_p2_out_of_range_threshold(empty_dir: Path, capfd: pytest.CaptureFixture[str]) -> None:
    _write_toml(empty_dir, "[cadence]\nctx_threshold = 1.5\n")
    cfg = load_cadence_config(start=empty_dir)
    assert cfg.ctx_threshold == DEFAULT_CTX_THRESHOLD


def test_load_p2_bool_as_byte_window_rejected(empty_dir: Path) -> None:
    _write_toml(empty_dir, "[cadence]\nctx_byte_window = true\n")
    cfg = load_cadence_config(start=empty_dir)
    assert cfg.ctx_byte_window == DEFAULT_CTX_BYTE_WINDOW


def test_load_p2_negative_byte_window_rejected(empty_dir: Path) -> None:
    _write_toml(empty_dir, "[cadence]\nctx_byte_window = -1\n")
    cfg = load_cadence_config(start=empty_dir)
    assert cfg.ctx_byte_window == DEFAULT_CTX_BYTE_WINDOW


def test_resolve_ctx_threshold_env_beats_toml(
    empty_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_toml(empty_dir, "[cadence]\nctx_threshold = 0.55\n")
    monkeypatch.setenv(ENV_CADENCE_CTX_THRESHOLD, "0.80")
    assert resolve_cadence_ctx_threshold(start=empty_dir) == pytest.approx(0.80)


def test_resolve_ctx_threshold_kwarg_beats_toml(empty_dir: Path) -> None:
    _write_toml(empty_dir, "[cadence]\nctx_threshold = 0.55\n")
    assert resolve_cadence_ctx_threshold(0.70, start=empty_dir) == pytest.approx(0.70)


def test_resolve_ctx_threshold_env_unparseable_falls_through(
    empty_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_toml(empty_dir, "[cadence]\nctx_threshold = 0.55\n")
    monkeypatch.setenv(ENV_CADENCE_CTX_THRESHOLD, "not a float")
    assert resolve_cadence_ctx_threshold(start=empty_dir) == pytest.approx(0.55)


def test_resolve_ctx_threshold_env_out_of_range_falls_through(
    empty_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_toml(empty_dir, "[cadence]\nctx_threshold = 0.55\n")
    monkeypatch.setenv(ENV_CADENCE_CTX_THRESHOLD, "1.5")
    assert resolve_cadence_ctx_threshold(start=empty_dir) == pytest.approx(0.55)


def test_resolve_ctx_byte_window_env_beats_toml(
    empty_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_toml(empty_dir, "[cadence]\nctx_byte_window = 700000\n")
    monkeypatch.setenv(ENV_CADENCE_CTX_BYTE_WINDOW, "900000")
    assert resolve_cadence_ctx_byte_window(start=empty_dir) == 900_000


def test_resolve_ctx_byte_window_env_zero_falls_through(
    empty_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_toml(empty_dir, "[cadence]\nctx_byte_window = 700000\n")
    monkeypatch.setenv(ENV_CADENCE_CTX_BYTE_WINDOW, "0")
    assert resolve_cadence_ctx_byte_window(start=empty_dir) == 700_000


def test_resolve_ctx_threshold_no_config_returns_default() -> None:
    assert resolve_cadence_ctx_threshold(start=Path("/nonexistent")) == DEFAULT_CTX_THRESHOLD


def test_p2_policy_in_toml_resolves_correctly(empty_dir: Path) -> None:
    """The TOML policy reader accepts p2_ctx_threshold."""
    _write_toml(empty_dir, '[cadence]\npolicy = "p2_ctx_threshold"\n')
    cfg = load_cadence_config(start=empty_dir)
    assert cfg.policy == POLICY_P2_CTX_THRESHOLD


def test_unknown_policy_rejected_with_stderr_log(
    empty_dir: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    _write_toml(empty_dir, '[cadence]\npolicy = "p99_nonsense"\n')
    cfg = load_cadence_config(start=empty_dir)
    assert cfg.policy == DEFAULT_POLICY
    captured = capfd.readouterr()
    assert "policy" in captured.err


def test_non_dict_cadence_section_logs_to_stderr(
    empty_dir: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Drive-by fix on top of #869: non-table [cadence] now logs."""
    _write_toml(empty_dir, "cadence = 42\n")
    cfg = load_cadence_config(start=empty_dir)
    assert cfg == CadenceConfig()
    captured = capfd.readouterr()
    assert "cadence" in captured.err


# --- would_fire_p2 (policy-agnostic shadow predicate) --------------------


def _wf_p2_cfg(
    *,
    enabled: bool = True,
    threshold: float = 0.5,
    window: int = 100,
) -> CadenceConfig:
    return CadenceConfig(
        enabled=enabled,
        policy=POLICY_P2_CTX_THRESHOLD,
        ctx_threshold=threshold,
        ctx_byte_window=window,
    )


def test_would_fire_p2_disabled_returns_reason(empty_dir: Path) -> None:
    cfg = _wf_p2_cfg(enabled=False)
    fired, reason = would_fire_p2(
        transcript_path=None, last_user_prompt="ok", config=cfg
    )
    assert fired is False
    assert "disabled" in reason


def test_would_fire_p2_non_positive_window_returns_reason() -> None:
    for bad in (0, -1, -100):
        cfg = CadenceConfig(
            enabled=True,
            policy=POLICY_P2_CTX_THRESHOLD,
            ctx_threshold=0.5,
            ctx_byte_window=bad,
        )
        fired, reason = would_fire_p2(
            transcript_path=None, last_user_prompt="ok", config=cfg
        )
        assert fired is False
        assert "ctx_byte_window" in reason


def test_would_fire_p2_out_of_range_threshold_returns_reason() -> None:
    for bad in (0.0, -0.5, 1.5, 2.0):
        cfg = CadenceConfig(
            enabled=True,
            policy=POLICY_P2_CTX_THRESHOLD,
            ctx_threshold=bad,
            ctx_byte_window=100,
        )
        fired, reason = would_fire_p2(
            transcript_path=None, last_user_prompt="ok", config=cfg
        )
        assert fired is False
        assert "ctx_threshold" in reason


def test_would_fire_p2_below_watermark_returns_reason(empty_dir: Path) -> None:
    cfg = _wf_p2_cfg(threshold=0.5, window=100)
    # Write a 30-byte file, well below the 50-byte watermark.
    fp = empty_dir / "transcript.jsonl"
    fp.write_text("x" * 30)
    fired, reason = would_fire_p2(
        transcript_path=fp, last_user_prompt="ok", config=cfg
    )
    assert fired is False
    assert "bytes=30" in reason
    assert "watermark=50" in reason


def test_would_fire_p2_above_watermark_but_no_boundary_returns_reason(
    empty_dir: Path,
) -> None:
    cfg = _wf_p2_cfg(threshold=0.5, window=100)
    fp = empty_dir / "transcript.jsonl"
    fp.write_text("x" * 80)  # above 50-byte watermark
    fired, reason = would_fire_p2(
        transcript_path=fp,
        last_user_prompt="please continue the deep architectural refactor",
        config=cfg,
    )
    assert fired is False
    assert "phase-boundary" in reason


def test_would_fire_p2_fires_when_all_conditions_met(empty_dir: Path) -> None:
    cfg = _wf_p2_cfg(threshold=0.5, window=100)
    fp = empty_dir / "transcript.jsonl"
    fp.write_text("x" * 80)  # above 50-byte watermark
    fired, reason = would_fire_p2(
        transcript_path=fp, last_user_prompt="ok thanks", config=cfg
    )
    assert fired is True
    assert "watermark" in reason
    assert "phase-boundary" in reason


def test_would_fire_p2_ignores_policy_field(empty_dir: Path) -> None:
    # As with would_fire_p1: policy-agnostic, for shadow-mode use.
    fp = empty_dir / "transcript.jsonl"
    fp.write_text("x" * 80)
    for policy in (
        cadence_POLICY_OFF := "off",
        POLICY_P1_EVERY_K_TURNS,
        POLICY_P2_CTX_THRESHOLD,
    ):
        cfg = CadenceConfig(
            enabled=True,
            policy=policy,
            ctx_threshold=0.5,
            ctx_byte_window=100,
        )
        fired, _ = would_fire_p2(
            transcript_path=fp, last_user_prompt="ok", config=cfg
        )
        assert fired is True, f"policy={policy!r} should not block P2 predicate"


def test_would_fire_p2_determinism_replay(empty_dir: Path) -> None:
    cfg = _wf_p2_cfg(threshold=0.5, window=100)
    fp = empty_dir / "transcript.jsonl"
    fp.write_text("x" * 80)
    prompts = [None, "ok", "ok thanks", "please continue", "done"]
    first = [
        would_fire_p2(
            transcript_path=fp, last_user_prompt=pr, config=cfg
        )
        for pr in prompts
    ]
    second = [
        would_fire_p2(
            transcript_path=fp, last_user_prompt=pr, config=cfg
        )
        for pr in prompts
    ]
    assert first == second


def test_should_fire_p2_delegates_to_would_fire_p2(empty_dir: Path) -> None:
    fp = empty_dir / "transcript.jsonl"
    fp.write_text("x" * 80)
    cfg_p2 = _wf_p2_cfg(threshold=0.5, window=100)
    cfg_off = CadenceConfig(
        enabled=True,
        policy="off",
        ctx_threshold=0.5,
        ctx_byte_window=100,
    )
    # When policy matches, should_fire_p2 mirrors would_fire_p2's bool.
    would, _ = would_fire_p2(
        transcript_path=fp, last_user_prompt="ok", config=cfg_p2
    )
    assert should_fire_p2(
        transcript_path=fp, last_user_prompt="ok", config=cfg_p2
    ) is would
    # Policy off: should_fire_p2 returns False even when conditions match.
    assert should_fire_p2(
        transcript_path=fp, last_user_prompt="ok", config=cfg_off
    ) is False
