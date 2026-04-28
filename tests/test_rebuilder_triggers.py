"""v1.4 rebuilder trigger-mode acceptance tests (issue #141).

Spec acceptance criteria:

  TM1. Manual mode is the v1.4.0 default; the PreCompact hook fires
       only on explicit invocation. The slash command and `aelf
       rebuild` CLI both reach the rebuild block.
  TM2. Threshold mode fires when the hook is called; below-band /
       wrong-mode configurations no-op.
  TM3. Threshold default value is sourced from the calibration file,
       not hand-picked. The `DEFAULT_THRESHOLD_FRACTION` constant
       matches the chosen value in
       `benchmarks/context-rebuilder/calibration_v1_4_0.json`.
  TM4. Dynamic mode is parked at v1.4 -- setting it raises a clear
       "parked v1.5" trace and no-ops the hook.
  TM5. The dynamic-mode parking decision is *evidence-backed*:
       running the dynamic_probe against the same synthetic fixture
       used for calibration produces a `verdict='park'` JSON.

All tests deterministic, no real network, well under the 2-second
per-test budget.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import cast

import pytest

from aelfrice.context_rebuilder import (
    DEFAULT_THRESHOLD_FRACTION,
    DEFAULT_TRIGGER_MODE,
    HOOK_EVENT_NAME,
    TRIGGER_MODE_DYNAMIC,
    TRIGGER_MODE_MANUAL,
    TRIGGER_MODE_THRESHOLD,
    VALID_TRIGGER_MODES,
    RebuilderConfig,
    load_rebuilder_config,
)
from aelfrice.hook import pre_compact
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore

# --- Fixtures (subset duplicated from test_context_rebuilder_hook to keep
#     this module self-contained -- intentional: the trigger-mode tests
#     should not hard-depend on internals of the AC1-AC4 suite).


_REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_FIXTURE = (
    _REPO_ROOT
    / "benchmarks"
    / "context-rebuilder"
    / "fixtures"
    / "synthetic"
    / "debugging_session_001.jsonl"
)
CALIBRATION_JSON = (
    _REPO_ROOT
    / "benchmarks"
    / "context-rebuilder"
    / "calibration_v1_4_0.json"
)


def _mk(
    bid: str,
    content: str,
    *,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
    session_id: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        demotion_pressure=0,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
        session_id=session_id,
    )


def _seed_db(db_path: Path, beliefs: list[Belief]) -> None:
    store = MemoryStore(str(db_path))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _set_db(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(path))


def _aelfrice_log(cwd: Path, lines: list[dict[str, object]]) -> Path:
    p = cwd / ".git" / "aelfrice" / "transcripts" / "turns.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return p


def _payload(cwd: Path) -> str:
    return json.dumps(
        {
            "session_id": "s1",
            "transcript_path": "",
            "cwd": str(cwd),
            "hook_event_name": "PreCompact",
        }
    )


def _write_config(cwd: Path, body: str) -> None:
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / ".aelfrice.toml").write_text(body, encoding="utf-8")


# --- TM1: manual mode is the default ------------------------------------


def test_tm1_default_trigger_mode_is_manual() -> None:
    """Ship default at v1.4.0 must be manual.

    The spec is explicit: threshold mode is opt-in until production
    telemetry. If this test fails, someone has flipped the ship
    default without re-reading the issue's "hard rules".
    """
    assert DEFAULT_TRIGGER_MODE == TRIGGER_MODE_MANUAL
    assert RebuilderConfig().trigger_mode == TRIGGER_MODE_MANUAL


def test_tm1_manual_mode_pre_compact_hook_no_ops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In manual mode, the PreCompact hook must NOT emit a block.

    Even with a populated store + a non-empty transcript, the hook
    short-circuits before reading the transcript or opening the
    store.
    """
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk(
        "L1", "user prefers uv over pip",
        lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z",
    )])
    _set_db(monkeypatch, db)
    cwd = tmp_path / "repo"
    (cwd / ".git").mkdir(parents=True)
    _write_config(cwd, '[rebuilder]\ntrigger_mode = "manual"\n')
    _aelfrice_log(cwd, [
        {"role": "user", "text": "test prompt"},
        {"role": "assistant", "text": "test answer"},
    ])
    sin = io.StringIO(_payload(cwd))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = pre_compact(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    # Manual mode = silent on stdout. Block must not appear.
    assert sout.getvalue() == ""


def test_tm1_manual_explicit_path_via_aelf_rebuild_still_fires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`aelf rebuild` (the manual surface) bypasses trigger_mode.

    The CLI subcommand must work regardless of `trigger_mode`. This
    is the explicit testing surface — gating it on trigger_mode
    would defeat the point.
    """
    from aelfrice.context_rebuilder import (
        RecentTurn,
        rebuild_v14,
    )

    db = tmp_path / "memory.db"
    _seed_db(db, [_mk(
        "L1", "user prefers uv over pip",
        lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z",
    )])
    _set_db(monkeypatch, db)
    store = MemoryStore(str(db))
    try:
        # `rebuild_v14` is the function `aelf rebuild` drives. Must
        # produce a non-empty block regardless of trigger_mode (which
        # is a hook-level concern, not a function-level one).
        block = rebuild_v14(
            [RecentTurn(role="user", text="anything")], store,
        )
    finally:
        store.close()
    assert "<aelfrice-rebuild>" in block
    assert 'id="L1"' in block


# --- TM2: threshold mode fires ------------------------------------------


def test_tm2_threshold_mode_pre_compact_emits_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trigger_mode='threshold' makes the hook fire as in v1.2.0a0."""
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk(
        "Lk", "user prefers uv",
        lock_level=LOCK_USER, locked_at="2026-04-26T00:00:00Z",
    )])
    _set_db(monkeypatch, db)
    cwd = tmp_path / "repo"
    (cwd / ".git").mkdir(parents=True)
    _write_config(cwd, '[rebuilder]\ntrigger_mode = "threshold"\n')
    _aelfrice_log(cwd, [
        {"role": "user", "text": "explain uv"},
        {"role": "assistant", "text": "uv is a fast Python pkg mgr"},
    ])
    sin = io.StringIO(_payload(cwd))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = pre_compact(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    out = sout.getvalue()
    assert out, "threshold-mode hook produced no envelope"
    # Parse and verify shape.
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    payload = cast(dict[str, object], parsed)
    spec_obj = payload.get("hookSpecificOutput")
    assert isinstance(spec_obj, dict)
    spec = cast(dict[str, object], spec_obj)
    assert spec.get("hookEventName") == HOOK_EVENT_NAME
    ctx = spec.get("additionalContext")
    assert isinstance(ctx, str)
    assert 'id="Lk"' in ctx


def test_tm2_threshold_mode_default_fraction_is_calibrated_value(
    tmp_path: Path,
) -> None:
    """The default threshold_fraction must match the calibrated value.

    `RebuilderConfig().threshold_fraction` and the chosen value in
    the committed calibration JSON must agree, otherwise a future
    calibration re-run could silently diverge from what ships.
    """
    cfg = RebuilderConfig()
    assert cfg.threshold_fraction == DEFAULT_THRESHOLD_FRACTION

    assert CALIBRATION_JSON.is_file(), (
        f"calibration JSON missing: {CALIBRATION_JSON}"
    )
    raw = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    chosen_obj = cast(dict[str, object], raw).get("chosen")
    assert isinstance(chosen_obj, dict)
    chosen = cast(dict[str, object], chosen_obj)
    chosen_fraction = chosen.get("threshold_fraction")
    assert isinstance(chosen_fraction, (int, float))
    assert float(chosen_fraction) == DEFAULT_THRESHOLD_FRACTION


def test_tm2_threshold_mode_invalid_fraction_falls_back_to_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """An out-of-range threshold_fraction degrades to the default."""
    cfg_path = tmp_path / ".aelfrice.toml"
    cfg_path.write_text(
        '[rebuilder]\nthreshold_fraction = 2.5\n', encoding="utf-8",
    )
    cfg = load_rebuilder_config(start=tmp_path)
    assert cfg.threshold_fraction == DEFAULT_THRESHOLD_FRACTION
    captured = capsys.readouterr()
    assert "threshold_fraction" in captured.err


def test_tm2_threshold_mode_loads_override_from_toml(
    tmp_path: Path,
) -> None:
    """A valid override in `.aelfrice.toml` wins over the default."""
    cfg_path = tmp_path / ".aelfrice.toml"
    cfg_path.write_text(
        '[rebuilder]\n'
        'trigger_mode = "threshold"\n'
        'threshold_fraction = 0.5\n',
        encoding="utf-8",
    )
    cfg = load_rebuilder_config(start=tmp_path)
    assert cfg.trigger_mode == TRIGGER_MODE_THRESHOLD
    assert cfg.threshold_fraction == 0.5


# --- TM3: calibration source-of-truth ------------------------------------


def test_tm3_calibration_json_is_committed_and_well_formed() -> None:
    """The calibration JSON must exist and have the documented shape."""
    raw = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    payload = cast(dict[str, object], raw)
    assert "fixture" in payload
    assert "method" in payload
    assert "sweep" in payload
    assert "chosen" in payload
    sweep = payload.get("sweep")
    assert isinstance(sweep, list)
    sweep_typed = cast(list[object], sweep)
    assert len(sweep_typed) >= 3
    for item in sweep_typed:
        assert isinstance(item, dict)
        i = cast(dict[str, object], item)
        # Each sweep point must have all required fields.
        for k in (
            "threshold_fraction",
            "clear_at",
            "rebuild_block_tokens",
            "full_replay_baseline_tokens",
            "token_budget_ratio",
            "continuation_fidelity",
            "n_post_clear_assistant_turns",
        ):
            assert k in i, f"calibration sweep point missing {k}: {i}"


def test_tm3_calibration_is_reproducible_byte_for_byte() -> None:
    """Re-running the calibration script produces byte-identical JSON.

    Reproducibility contract: the calibration is deterministic on the
    same fixture + same code. If this test fails, either someone
    touched `_choose_threshold` / the proxy metric, or the fixture
    changed; both require re-running the calibration and committing
    a fresh JSON.

    The `fixture` field in the JSON is the path the calibration was
    invoked with; it is excluded from the byte-for-byte comparison
    because absolute-vs-relative path is a property of the caller,
    not the calibration. Every other field is compared byte-for-
    byte.
    """
    from benchmarks.context_rebuilder.calibrate import calibrate

    cal = calibrate(SYNTHETIC_FIXTURE)
    re_derived_obj = cal.to_dict()
    committed_obj = json.loads(
        CALIBRATION_JSON.read_text(encoding="utf-8")
    )
    assert isinstance(committed_obj, dict)
    committed_typed = cast(dict[str, object], committed_obj)
    # Path-field is the caller's choice; re-derive without it.
    _ = re_derived_obj.pop("fixture", None)
    _ = committed_typed.pop("fixture", None)
    re_derived = json.dumps(re_derived_obj, indent=2, sort_keys=True)
    committed = json.dumps(committed_typed, indent=2, sort_keys=True)
    assert re_derived == committed, (
        "calibration drift: re-running calibrate.py produced output "
        "that differs from the committed calibration_v1_4_0.json. "
        "Re-run `python -m benchmarks.context_rebuilder.calibrate "
        f"{SYNTHETIC_FIXTURE} --out {CALIBRATION_JSON}` and commit "
        "the new JSON."
    )


# --- TM4: dynamic mode is parked ----------------------------------------


def test_tm4_dynamic_mode_in_valid_modes_set() -> None:
    """All three trigger modes are valid config values.

    The string is a *legitimate* config value (it doesn't fall back
    to default the way a typo would); the hook handles it specially.
    """
    assert TRIGGER_MODE_DYNAMIC in VALID_TRIGGER_MODES


def test_tm4_dynamic_mode_pre_compact_hook_logs_parked_and_no_ops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """trigger_mode='dynamic' must produce a clear parked trace.

    Spec: "If dynamic parks: a regression test asserts
    `trigger_mode = "dynamic"` raises a clear 'parked at v1.4, ships
    v1.5' error."

    The hook contract is non-blocking (exit 0, no raise), so 'raise'
    here is interpreted as 'surface a clear parked trace on stderr'
    -- the analog of an error in a non-blocking codepath.
    """
    db = tmp_path / "memory.db"
    _seed_db(db, [_mk(
        "Lk", "x", lock_level=LOCK_USER,
        locked_at="2026-04-26T00:00:00Z",
    )])
    _set_db(monkeypatch, db)
    cwd = tmp_path / "repo"
    (cwd / ".git").mkdir(parents=True)
    _write_config(cwd, '[rebuilder]\ntrigger_mode = "dynamic"\n')
    _aelfrice_log(cwd, [
        {"role": "user", "text": "anything"},
        {"role": "assistant", "text": "anything"},
    ])
    sin = io.StringIO(_payload(cwd))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = pre_compact(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    # No additionalContext written.
    assert sout.getvalue() == ""
    # Stderr surfaces the parked-v1.5 trace, with the v1.5 reference
    # so a user can find the ship plan.
    err = serr.getvalue()
    assert "parked" in err.lower()
    assert "v1.5" in err
    assert "dynamic" in err


def test_tm4_dynamic_mode_via_context_rebuilder_main_also_parks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The alternate hook entry point honors the same trigger gate."""
    from aelfrice.context_rebuilder import main as cr_main

    db = tmp_path / "memory.db"
    _seed_db(db, [_mk(
        "Lk", "x", lock_level=LOCK_USER,
        locked_at="2026-04-26T00:00:00Z",
    )])
    _set_db(monkeypatch, db)
    cwd = tmp_path / "repo"
    (cwd / ".git").mkdir(parents=True)
    _write_config(cwd, '[rebuilder]\ntrigger_mode = "dynamic"\n')
    _aelfrice_log(cwd, [
        {"role": "user", "text": "anything"},
    ])
    sin = io.StringIO(_payload(cwd))
    sout = io.StringIO()
    serr = io.StringIO()
    rc = cr_main(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    assert sout.getvalue() == ""
    assert "parked" in serr.getvalue().lower()


# --- TM5: dynamic-mode parking is evidence-backed -----------------------


@pytest.mark.timeout(15)
def test_tm5_dynamic_probe_verdict_is_park() -> None:
    """The dynamic_probe must produce verdict='park' on the bundled
    synthetic fixture.

    This is the regression-test hook on the parking decision: if a
    future change to the rebuilder, the entity extractor, or the
    fixture flips the verdict to 'ship', we *want* CI red so the
    park documentation can be re-evaluated.
    """
    from benchmarks.context_rebuilder.dynamic_probe import probe

    result = probe(SYNTHETIC_FIXTURE)
    assert result.verdict == "park", (
        f"dynamic_probe flipped to ship: {result.rationale}. "
        f"This is a doc-update event: re-read "
        f"docs/context_rebuilder.md § Dynamic mode (parked v1.5) "
        f"and either ship dynamic mode (with PR-body acknowledgment "
        f"of the verdict change) or update the probe."
    )
    # And the candidates must include both probes the doc names.
    cand_names = {c.name for c in result.candidates}
    assert "rate_of_growth" in cand_names
    assert "entity_density_delta" in cand_names


def test_tm5_dynamic_probe_threshold_reference_matches_calibrated_default() -> None:
    """The probe's threshold reference uses the calibrated default.

    Otherwise the comparison is against a stale baseline and the
    park verdict means nothing.
    """
    from benchmarks.context_rebuilder.dynamic_probe import (
        THRESHOLD_REFERENCE_FRACTION,
    )

    assert THRESHOLD_REFERENCE_FRACTION == DEFAULT_THRESHOLD_FRACTION


# --- Edge: config parser validates trigger_mode strings -----------------


def test_config_parser_rejects_unknown_trigger_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """An unknown trigger_mode string must fall back to default."""
    cfg_path = tmp_path / ".aelfrice.toml"
    cfg_path.write_text(
        '[rebuilder]\ntrigger_mode = "lasagna"\n',
        encoding="utf-8",
    )
    cfg = load_rebuilder_config(start=tmp_path)
    assert cfg.trigger_mode == DEFAULT_TRIGGER_MODE
    captured = capsys.readouterr()
    assert "trigger_mode" in captured.err


def test_config_parser_accepts_each_valid_trigger_mode(
    tmp_path: Path,
) -> None:
    """Every valid trigger mode round-trips through the parser."""
    for mode in VALID_TRIGGER_MODES:
        cfg_path = tmp_path / ".aelfrice.toml"
        cfg_path.write_text(
            f'[rebuilder]\ntrigger_mode = "{mode}"\n',
            encoding="utf-8",
        )
        cfg = load_rebuilder_config(start=tmp_path)
        assert cfg.trigger_mode == mode
