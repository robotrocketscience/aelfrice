"""#1634: a headless host session is not ingested as user knowledge.

An evaluation harness drives the host headlessly, one scripted prompt per
session, and ingest stored each run's prompt as if a user had typed it:
one run became N "independent" sessions of corroboration. The host names
how a session started -- its entrypoint -- in the session log's records
and in a hook's `CLAUDE_CODE_ENTRYPOINT`. The headless entrypoints are
`sdk-cli` (`claude -p`), `sdk-ts`, and `sdk-py`.

Both capture paths skip them by default. `[ingest] capture_print_mode =
true`, or `AELFRICE_CAPTURE_PRINT_MODE=1`, restores capture.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.ingest import ingest_jsonl
from aelfrice.store import MemoryStore

_TEXTS = [
    "The configuration file lives at /etc/aelfrice/conf.",
    "Astronomers process supernova imagery nightly using clusters.",
    "Radio telescopes calibrate against known pulsar timings.",
]
_HEADLESS = ["sdk-cli", "sdk-ts", "sdk-py"]
_INTERACTIVE = ["cli", "claude-vscode", "remote", "claude-desktop"]


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))
    monkeypatch.delenv("AELFRICE_CAPTURE_PRINT_MODE", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ATTENDED", raising=False)
    # The process cwd holds no `.aelfrice.toml`, so only a file a test
    # writes can decide a result.
    work = tmp_path / "proc-cwd"
    work.mkdir()
    monkeypatch.chdir(work)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "pm.db"))
    yield s
    s.close()


def _session_log(path: Path, entrypoint: str | None, session: str) -> Path:
    """A host session log (shape 2) of three user records."""
    with path.open("w") as f:
        for i, text in enumerate(_TEXTS):
            rec: dict[str, object] = {
                "type": "user",
                "message": {"role": "user", "content": text},
                "sessionId": session,
                "timestamp": f"2026-08-01T00:00:0{i}Z",
                "cwd": "/work",
            }
            if entrypoint is not None:
                rec["entrypoint"] = entrypoint
            f.write(json.dumps(rec) + "\n")
    return path


# --- the session-log path --------------------------------------------------

@pytest.mark.timeout(30)
@pytest.mark.parametrize("entrypoint", _HEADLESS)
def test_a_headless_session_log_adds_no_belief(
    store: MemoryStore, tmp_path: Path, entrypoint: str,
) -> None:
    result = ingest_jsonl(store, _session_log(tmp_path / "p.jsonl", entrypoint, "p"))
    assert result.beliefs_inserted == 0
    assert result.skipped_lines == len(_TEXTS)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("entrypoint", [*_INTERACTIVE, None])
def test_interactive_and_older_logs_ingest_as_before(
    store: MemoryStore, tmp_path: Path, entrypoint: str | None,
) -> None:
    """Interactive entrypoints, and a log with no `entrypoint`, still count."""
    result = ingest_jsonl(store, _session_log(tmp_path / "i.jsonl", entrypoint, "i"))
    assert result.beliefs_inserted == len(_TEXTS)


@pytest.mark.timeout(30)
def test_the_logger_shape_is_not_touched_by_the_session_log_check(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """aelfrice's own turns.jsonl carries no `entrypoint`; it ingests."""
    path = tmp_path / "turns.jsonl"
    path.write_text("".join(
        json.dumps({"role": "user", "text": t, "session_id": "s",
                    "ts": f"2026-08-01T00:00:0{i}Z"}) + "\n"
        for i, t in enumerate(_TEXTS)
    ))
    assert ingest_jsonl(store, path).beliefs_inserted == len(_TEXTS)


# --- the override, and its precedence ---------------------------------------

@pytest.mark.timeout(30)
def test_the_env_override_restores_capture(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_CAPTURE_PRINT_MODE", "1")
    result = ingest_jsonl(store, _session_log(tmp_path / "p.jsonl", "sdk-cli", "p"))
    assert result.beliefs_inserted == len(_TEXTS)


@pytest.mark.timeout(30)
def test_the_toml_override_restores_capture(
    store: MemoryStore, tmp_path: Path,
) -> None:
    Path(".aelfrice.toml").write_text("[ingest]\ncapture_print_mode = true\n")
    result = ingest_jsonl(store, _session_log(tmp_path / "p.jsonl", "sdk-cli", "p"))
    assert result.beliefs_inserted == len(_TEXTS)


@pytest.mark.timeout(30)
def test_the_env_value_wins_over_the_toml_key(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TOML says capture; the environment says no; the environment wins."""
    Path(".aelfrice.toml").write_text("[ingest]\ncapture_print_mode = true\n")
    monkeypatch.setenv("AELFRICE_CAPTURE_PRINT_MODE", "0")
    result = ingest_jsonl(store, _session_log(tmp_path / "p.jsonl", "sdk-cli", "p"))
    assert result.beliefs_inserted == 0


@pytest.mark.timeout(30)
@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " TRUE "])
def test_every_truthy_env_spelling_restores_capture(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("AELFRICE_CAPTURE_PRINT_MODE", value)
    result = ingest_jsonl(store, _session_log(tmp_path / "p.jsonl", "sdk-cli", "p"))
    assert result.beliefs_inserted == len(_TEXTS)


@pytest.mark.timeout(30)
@pytest.mark.parametrize("value", ["0", "false", "no", "off", " Off "])
def test_every_falsy_env_spelling_overrides_a_toml_yes(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    Path(".aelfrice.toml").write_text("[ingest]\ncapture_print_mode = true\n")
    monkeypatch.setenv("AELFRICE_CAPTURE_PRINT_MODE", value)
    result = ingest_jsonl(store, _session_log(tmp_path / "p.jsonl", "sdk-cli", "p"))
    assert result.beliefs_inserted == 0


@pytest.mark.timeout(30)
def test_an_unrecognized_env_value_defers_to_the_toml_key(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`maybe` is not a decision; the TOML key decides."""
    Path(".aelfrice.toml").write_text("[ingest]\ncapture_print_mode = true\n")
    monkeypatch.setenv("AELFRICE_CAPTURE_PRINT_MODE", "maybe")
    result = ingest_jsonl(store, _session_log(tmp_path / "p.jsonl", "sdk-cli", "p"))
    assert result.beliefs_inserted == len(_TEXTS)


@pytest.mark.timeout(30)
def test_an_unrecognized_env_value_alone_keeps_the_default(
    store: MemoryStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found by review: with no TOML key, `maybe` must not read as a yes."""
    monkeypatch.setenv("AELFRICE_CAPTURE_PRINT_MODE", "maybe")
    result = ingest_jsonl(store, _session_log(tmp_path / "p.jsonl", "sdk-cli", "p"))
    assert result.beliefs_inserted == 0


@pytest.mark.timeout(30)
@pytest.mark.parametrize("toml", [
    "[ingest]\ncapture_print_mode = \"true\"\n",   # a string, not a bool
    "[ingest\ncapture_print_mode = true\n",        # malformed TOML
    "ingest = 5\n",                                 # [ingest] not a table
], ids=["non-bool", "malformed", "not-a-table"])
def test_an_unusable_toml_value_keeps_the_default(
    store: MemoryStore, tmp_path: Path, toml: str,
) -> None:
    """Never raises, and never reads a non-bool as a yes."""
    Path(".aelfrice.toml").write_text(toml)
    result = ingest_jsonl(store, _session_log(tmp_path / "p.jsonl", "sdk-cli", "p"))
    assert result.beliefs_inserted == 0


@pytest.mark.timeout(30)
@pytest.mark.parametrize("value", [["sdk-cli"], {"x": 1}, 7, None])
def test_a_malformed_entrypoint_is_ingested_not_raised(
    store: MemoryStore, tmp_path: Path, value: object,
) -> None:
    """Found by review: an unhashable value raised TypeError mid-ingest."""
    path = tmp_path / "odd.jsonl"
    path.write_text(json.dumps({
        "type": "user", "entrypoint": value, "sessionId": "odd",
        "message": {"role": "user", "content": _TEXTS[0]},
    }) + "\n")
    result = ingest_jsonl(store, path)
    assert result.beliefs_inserted == 1


@pytest.mark.timeout(30)
def test_a_nul_in_the_start_path_keeps_the_default() -> None:
    """Found by review: path resolution raised ValueError, outside the
    reader's `except OSError`."""
    from aelfrice import print_mode

    assert print_mode.is_print_mode_capture_enabled(
        start=Path("a\x00b"), env={}) is False


@pytest.mark.timeout(30)
def test_a_toml_nested_too_deep_to_parse_keeps_the_default() -> None:
    """Found by review: a deep nest raised RecursionError from the parser."""
    from aelfrice import print_mode

    Path(".aelfrice.toml").write_text(
        "[ingest]\ncapture_print_mode = true\nx = " + "[" * 50_000
        + "]" * 50_000 + "\n")
    assert print_mode.is_print_mode_capture_enabled(env={}) is False


@pytest.mark.timeout(30)
def test_a_non_utf8_toml_does_not_raise(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Found by review: undecodable bytes must not escape the reader."""
    Path(".aelfrice.toml").write_bytes(
        b"[ingest]\ncapture_print_mode = true\n# \xff\xfe\n")
    result = ingest_jsonl(store, _session_log(tmp_path / "p.jsonl", "sdk-cli", "p"))
    assert result.beliefs_inserted == len(_TEXTS)


@pytest.mark.timeout(30)
def test_an_unreadable_directory_keeps_the_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Found by review: config discovery itself could raise, so the reader
    was not the "never raises" its docstring said."""
    from aelfrice import print_mode

    def _denied(start: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(print_mode, "discover_config", _denied)
    assert print_mode.is_print_mode_capture_enabled(env={}) is False
    assert "cannot look for .aelfrice.toml" in capsys.readouterr().err


@pytest.mark.timeout(30)
def test_a_malformed_toml_is_reported_on_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from aelfrice import print_mode

    Path(".aelfrice.toml").write_text("[ingest\n")
    assert print_mode.is_print_mode_capture_enabled(env={}) is False
    assert "cannot read capture_print_mode" in capsys.readouterr().err


@pytest.mark.timeout(30)
def test_an_explicit_env_mapping_replaces_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found by review: the `env=` parameter had no test."""
    from aelfrice import print_mode

    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "sdk-cli")
    monkeypatch.setenv("AELFRICE_CAPTURE_PRINT_MODE", "1")
    assert print_mode.is_headless_hook_env({"CLAUDE_CODE_ENTRYPOINT": "cli"}) is False
    assert print_mode.is_print_mode_capture_enabled(env={}) is False


# --- the live logger path --------------------------------------------------

def _log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cwd: Path | None = None,
) -> list[dict[str, object]]:
    """Drive the real UserPromptSubmit handler; return the rows it wrote."""
    from aelfrice import transcript_logger

    tdir = tmp_path / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(transcript_logger, "transcripts_dir", lambda: tdir)
    transcript_logger._handle_user_prompt_submit(  # pyright: ignore[reportPrivateUsage]
        {"session_id": "s1", "prompt": _TEXTS[0],
         "cwd": str(cwd if cwd is not None else tmp_path)}
    )
    out = tdir / transcript_logger.TURNS_FILENAME
    if not out.exists():
        return []
    return [json.loads(x) for x in out.read_text().splitlines() if x.strip()]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("entrypoint", _HEADLESS)
def test_the_logger_skips_a_headless_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", entrypoint)
    assert _log(tmp_path, monkeypatch) == []


@pytest.mark.timeout(30)
@pytest.mark.parametrize("entrypoint", [*_INTERACTIVE, None])
def test_the_logger_records_an_interactive_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str | None,
) -> None:
    if entrypoint is not None:
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", entrypoint)
    assert len(_log(tmp_path, monkeypatch)) == 1


@pytest.mark.timeout(30)
def test_an_unattended_interactive_session_is_still_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Background, daemon, and teammate sessions set SESSION_ATTENDED=0.

    They are a user's own work, so the logger must not read that flag as
    "headless": only the entrypoint decides.
    """
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ATTENDED", "0")
    assert len(_log(tmp_path, monkeypatch)) == 1


@pytest.mark.timeout(30)
def test_the_logger_skips_a_headless_session_without_a_payload_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No `cwd` in the payload: the default decides, and it still skips."""
    from aelfrice import transcript_logger

    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    monkeypatch.setattr(transcript_logger, "transcripts_dir", lambda: tdir)
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "sdk-cli")
    transcript_logger._handle_user_prompt_submit(  # pyright: ignore[reportPrivateUsage]
        {"session_id": "s1", "prompt": _TEXTS[0]}
    )
    assert not (tdir / transcript_logger.TURNS_FILENAME).exists()


@pytest.mark.timeout(30)
def test_a_malformed_setting_skips_rather_than_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed override reads as the default; the turn is not recorded."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".aelfrice.toml").write_text("[ingest\ncapture_print_mode = true\n")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "sdk-cli")
    assert _log(tmp_path, monkeypatch, cwd=project) == []


@pytest.mark.timeout(30)
def test_an_error_in_the_check_records_the_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found by review: only an unexpected error records a headless turn,
    so a broken install never costs the logger one; that had no test."""
    from aelfrice import print_mode

    def _boom(env: object = None) -> bool:
        raise RuntimeError("broken install")

    monkeypatch.setattr(print_mode, "is_headless_hook_env", _boom)
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "sdk-cli")
    assert len(_log(tmp_path, monkeypatch)) == 1


def _stop_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    flushed: list[Path] | None = None, cwd: Path | None = None,
) -> list[dict[str, object]]:
    """Drive the real Stop handler with an assistant reply in the payload.

    The flush is stubbed, and each call's directory goes to `flushed`.
    """
    from aelfrice import transcript_logger

    tdir = tmp_path / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(transcript_logger, "transcripts_dir", lambda: tdir)
    sink: list[Path] = flushed if flushed is not None else []
    monkeypatch.setattr(transcript_logger, "_maybe_stop_flush",
                        lambda d: sink.append(d) or False)
    monkeypatch.setattr(transcript_logger, "_stop_payload_assistant_text",
                        lambda _p: "The deploy target is the staging cluster.")
    transcript_logger._handle_stop(  # pyright: ignore[reportPrivateUsage]
        {"session_id": "s1", "cwd": str(cwd if cwd is not None else tmp_path)}
    )
    out = tdir / transcript_logger.TURNS_FILENAME
    if not out.exists():
        return []
    return [json.loads(x) for x in out.read_text().splitlines() if x.strip()]


@pytest.mark.timeout(30)
def test_the_logger_skips_a_headless_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found by review: only the prompt handler checked the entrypoint."""
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "sdk-cli")
    assert _stop_rows(tmp_path, monkeypatch) == []


@pytest.mark.timeout(30)
def test_the_logger_records_an_interactive_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    rows = _stop_rows(tmp_path, monkeypatch)
    assert [r["role"] for r in rows] == ["assistant"]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("entrypoint", ["sdk-cli", "cli"])
def test_the_stop_flush_runs_whether_or_not_the_reply_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrypoint: str,
) -> None:
    """Found by review: the flush folds turns other sessions logged, so a
    headless session must still run it; the stub hid its absence."""
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", entrypoint)
    flushed: list[Path] = []
    _stop_rows(tmp_path, monkeypatch, flushed)
    assert flushed == [tmp_path / "transcripts"]


@pytest.mark.timeout(30)
def test_the_logger_reads_the_override_from_the_payload_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The project is the payload's `cwd`, not the hook's process cwd."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".aelfrice.toml").write_text("[ingest]\ncapture_print_mode = true\n")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "sdk-cli")
    assert len(_log(tmp_path, monkeypatch, cwd=project)) == 1


@pytest.mark.timeout(30)
def test_the_stop_handler_reads_the_override_from_the_payload_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Found by review: only the prompt handler's `cwd` had a test."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".aelfrice.toml").write_text("[ingest]\ncapture_print_mode = true\n")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "sdk-cli")
    rows = _stop_rows(tmp_path, monkeypatch, cwd=project)
    assert [r["role"] for r in rows] == ["assistant"]
