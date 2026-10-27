"""The Gemini hook adapter, `aelf-gemini-hook` (#1812).

Each test feeds the adapter a synthetic Gemini payload, built from the
field names in the Gemini CLI hooks reference, and checks what reaches the
aelfrice handlers and what Gemini reads back on stdout. Stores, turn logs,
and Gemini homes are under `tmp_path`. No Gemini CLI runs here.
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import aelfrice.hook_gemini as hg
from aelfrice.gemini_paths import GeminiHomeError, resolve_gemini_dir
from aelfrice.hosts import HOST_ENV
from aelfrice.models import BELIEF_FACTUAL, LOCK_USER, Belief
from aelfrice.store import MemoryStore

pytestmark = pytest.mark.timeout(60)

SESSION = "widgetsession-gemini"


def _payload(event: str, **extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "session_id": SESSION,
        "transcript_path": "/tmp/widget/chats/session-widget.json",
        "cwd": "/tmp/widget",
        "hook_event_name": event,
        "timestamp": "2026-10-27T00:00:00.000Z",
    }
    base.update(extra)
    return base


def _run(payload: object) -> tuple[str, str]:
    out, err = io.StringIO(), io.StringIO()
    text = payload if isinstance(payload, str) else json.dumps(payload)
    assert hg.main(stdin=io.StringIO(text), stdout=out, stderr=err) == 0
    return out.getvalue(), err.getvalue()


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, object]]]:
    """Replace each handler with a recorder that names itself."""
    calls: list[tuple[str, dict[str, object]]] = []

    def recorder(name: str, context: str = ""):
        def handler(text: str, out: io.StringIO, err: io.StringIO) -> None:
            _ = err
            calls.append((name, json.loads(text)))
            if context:
                out.write(context)
        return handler

    monkeypatch.setitem(
        hg.HANDLERS, "SessionStart", (recorder("session_start", "baseline"),),
    )
    monkeypatch.setitem(
        hg.HANDLERS, "BeforeAgent",
        (recorder("retrieval", "<aelfrice-memory>hit</aelfrice-memory>"),
         recorder("logger")),
    )
    monkeypatch.setitem(
        hg.HANDLERS, "AfterAgent",
        (recorder("logger"), recorder("stop", "ignored on AfterAgent")),
    )
    monkeypatch.setitem(hg.HANDLERS, "PreCompress", (recorder("logger"),))
    return calls


# --- payload translation ----------------------------------------------------


@pytest.mark.parametrize(
    ("event", "expected"),
    [("SessionStart", "SessionStart"), ("BeforeAgent", "UserPromptSubmit"),
     ("AfterAgent", "Stop"), ("PreCompress", "PreCompact")],
)
def test_each_mapped_event_runs_under_its_aelfrice_name(
    event: str, expected: str,
) -> None:
    mapped = hg.to_aelfrice_payload(_payload(event))
    assert mapped is not None
    assert mapped["hook_event_name"] == expected
    assert mapped["session_id"] == SESSION
    assert mapped["cwd"] == "/tmp/widget"


@pytest.mark.parametrize(
    "event",
    ["SessionEnd", "BeforeTool", "AfterTool", "BeforeModel", "AfterModel",
     "BeforeToolSelection", "Notification", "Stop", "UserPromptSubmit"],
)
def test_an_unmapped_event_has_no_aelfrice_payload(event: str) -> None:
    """Killed by: passing an unknown event through under its own name."""
    assert hg.to_aelfrice_payload(_payload(event)) is None


def test_the_transcript_path_is_blanked() -> None:
    """Gemini's transcript is one JSON document, not JSON lines. Killed
    by: passing `transcript_path` through."""
    mapped = hg.to_aelfrice_payload(_payload("BeforeAgent", prompt="p"))
    assert mapped is not None
    assert mapped["transcript_path"] == ""


def test_the_reply_moves_to_last_assistant_message() -> None:
    mapped = hg.to_aelfrice_payload(_payload(
        "AfterAgent", prompt="widget question",
        prompt_response="widget answer", stop_hook_active=False,
    ))
    assert mapped is not None
    assert mapped["last_assistant_message"] == "widget answer"
    assert mapped["prompt"] == "widget question"
    assert mapped["stop_hook_active"] is False


def test_a_non_string_reply_is_not_copied() -> None:
    mapped = hg.to_aelfrice_payload(_payload("AfterAgent", prompt_response=7))
    assert mapped is not None
    assert "last_assistant_message" not in mapped


# --- stdout contract ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", ""), ("  \n", ""), ("plain block\n", "plain block"),
     (json.dumps({"hookSpecificOutput": {"additionalContext": " ctx "}}), "ctx"),
     (json.dumps({"decision": "allow"}), ""),
     (json.dumps({"hookSpecificOutput": {"additionalContext": 3}}), ""),
     ("[1, 2]", "[1, 2]")],
)
def test_context_from_output(text: str, expected: str) -> None:
    assert hg.context_from_output(text) == expected


def test_render_output_on_a_context_event() -> None:
    assert json.loads(hg.render_output("BeforeAgent", "ctx")) == {
        "hookSpecificOutput": {
            "hookEventName": "BeforeAgent", "additionalContext": "ctx",
        },
    }


@pytest.mark.parametrize(
    ("event", "context"),
    [("BeforeAgent", ""), ("AfterAgent", "ctx"), ("PreCompress", "ctx"),
     ("AfterTool", "ctx"), ("", "ctx")],
)
def test_render_output_is_an_empty_object_otherwise(
    event: str, context: str,
) -> None:
    """Killed by: emitting context on an event Gemini documents none for."""
    assert hg.render_output(event, context) == "{}\n"


# --- the entry point, per event ----------------------------------------------


def test_the_handler_table_runs_each_event_in_order() -> None:
    """The fixture below replaces the table, so this pins the real one.
    Retrieval reads the turn log before the logger appends this prompt,
    and the logger records the reply before the Stop hook runs."""
    assert dict(hg.HANDLERS) == {
        "SessionStart": (hg._run_session_start,),  # noqa: SLF001
        "BeforeAgent": (hg._run_retrieval, hg._run_logger),  # noqa: SLF001
        "AfterAgent": (hg._run_logger, hg._run_stop),  # noqa: SLF001
        "PreCompress": (hg._run_logger,),  # noqa: SLF001
    }


def test_session_start_injects_the_handler_context(
    recorded: list[tuple[str, dict[str, object]]],
) -> None:
    out, _ = _run(_payload("SessionStart", source="startup"))
    assert [name for name, _ in recorded] == ["session_start"]
    assert recorded[0][1]["source"] == "startup"
    assert json.loads(out)["hookSpecificOutput"] == {
        "hookEventName": "SessionStart", "additionalContext": "baseline",
    }


def test_before_agent_runs_retrieval_then_the_logger(
    recorded: list[tuple[str, dict[str, object]]],
) -> None:
    out, _ = _run(_payload("BeforeAgent", prompt="widget prompt"))
    assert [name for name, _ in recorded] == ["retrieval", "logger"]
    for _, seen in recorded:
        assert seen["hook_event_name"] == "UserPromptSubmit"
        assert seen["prompt"] == "widget prompt"
    assert json.loads(out)["hookSpecificOutput"]["additionalContext"] == (
        "<aelfrice-memory>hit</aelfrice-memory>"
    )


def test_after_agent_runs_the_logger_then_stop_and_prints_no_context(
    recorded: list[tuple[str, dict[str, object]]],
) -> None:
    out, _ = _run(_payload(
        "AfterAgent", prompt="q", prompt_response="widget reply",
        stop_hook_active=False,
    ))
    assert [name for name, _ in recorded] == ["logger", "stop"]
    assert recorded[0][1]["last_assistant_message"] == "widget reply"
    assert recorded[1][1]["hook_event_name"] == "Stop"
    assert out == "{}\n"


def test_pre_compress_runs_the_logger(
    recorded: list[tuple[str, dict[str, object]]],
) -> None:
    out, _ = _run(_payload("PreCompress", trigger="auto"))
    assert recorded == [("logger", recorded[0][1])]
    assert recorded[0][1]["hook_event_name"] == "PreCompact"
    assert recorded[0][1]["trigger"] == "auto"
    assert out == "{}\n"


@pytest.mark.parametrize("event", ["SessionEnd", "BeforeTool", "Notification"])
def test_an_unmapped_event_runs_nothing(
    recorded: list[tuple[str, dict[str, object]]], event: str,
) -> None:
    out, _ = _run(_payload(event))
    assert recorded == []
    assert out == "{}\n"


@pytest.mark.parametrize("text", ["", "not json", "[1]", "null"])
def test_an_unreadable_payload_prints_an_empty_object(
    recorded: list[tuple[str, dict[str, object]]], text: str,
) -> None:
    out, _ = _run(text)
    assert recorded == []
    assert out == "{}\n"


def test_a_failing_handler_does_not_stop_the_next(
    recorded: list[tuple[str, dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: dropping the per-handler guard."""
    logger = hg.HANDLERS["BeforeAgent"][1]

    def broken(text: str, out: io.StringIO, err: io.StringIO) -> None:
        raise RuntimeError("widget failure")

    monkeypatch.setitem(hg.HANDLERS, "BeforeAgent", (broken, logger))
    out, err = _run(_payload("BeforeAgent", prompt="p"))
    assert [name for name, _ in recorded] == ["logger"]
    assert "widget failure" in err
    assert out == "{}\n"


def test_a_handler_printing_to_sys_stdout_cannot_break_the_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini shows non-JSON stdout to the user and drops it from the
    model's context. Killed by: removing the stdout redirect."""
    def chatty(text: str, out: io.StringIO, err: io.StringIO) -> None:
        print("stray line")

    monkeypatch.setitem(hg.HANDLERS, "PreCompress", (chatty,))
    real_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", real_stdout)
    out, _ = _run(_payload("PreCompress", trigger="manual"))
    assert out == "{}\n"
    assert real_stdout.getvalue() == ""


def test_the_host_marker_is_set_during_the_call_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str | None] = []

    def probe(text: str, out: io.StringIO, err: io.StringIO) -> None:
        seen.append(os.environ.get(HOST_ENV))

    monkeypatch.setitem(hg.HANDLERS, "PreCompress", (probe,))
    monkeypatch.delenv(HOST_ENV, raising=False)
    _run(_payload("PreCompress"))
    assert seen == ["gemini"]
    assert HOST_ENV not in os.environ
    monkeypatch.setenv(HOST_ENV, "codex")
    _run(_payload("PreCompress"))
    assert os.environ[HOST_ENV] == "codex"


# --- the real handlers --------------------------------------------------------


@pytest.fixture
def store_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setenv("AELFRICE_INGEST_STOP_FLUSH_TURNS", "0")
    monkeypatch.chdir(tmp_path)
    return db


def test_session_start_carries_a_locked_belief(store_env: Path) -> None:
    store = MemoryStore(str(store_env))
    try:
        store.insert_belief(Belief(
            id="g1", content="The widget deploy runs from the gemini branch.",
            content_hash="hash-g1", alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
            lock_level=LOCK_USER, locked_at="2026-10-01T00:00:00Z",
            created_at="2026-10-01T00:00:00Z", last_retrieved_at=None,
        ))
    finally:
        store.close()
    out, _ = _run(_payload("SessionStart", source="startup"))
    specific = json.loads(out)["hookSpecificOutput"]
    assert specific["hookEventName"] == "SessionStart"
    assert "widget deploy runs from the gemini branch" in specific[
        "additionalContext"
    ]


def test_a_turn_reaches_the_turn_log(store_env: Path, tmp_path: Path) -> None:
    _run(_payload("BeforeAgent", prompt="Where does the widget deploy run?"))
    _run(_payload(
        "AfterAgent", prompt="Where does the widget deploy run?",
        prompt_response="The widget deploy runs from the gemini branch.",
        stop_hook_active=False,
    ))
    rows = [
        json.loads(line)
        for line in (tmp_path / "transcripts" / "turns.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    turns = [(r["role"], r["text"], r["session_id"]) for r in rows if "role" in r]
    assert turns == [
        ("user", "Where does the widget deploy run?", SESSION),
        ("assistant", "The widget deploy runs from the gemini branch.", SESSION),
    ]


# --- the saved-memory mirror ---------------------------------------------------


_GEMINI_MD = (
    "# Project notes\n\nKeep this.\n\n"
    "## Gemini Added Memories\n"
    "- The widget service listens on port 8443.\n"
    "* The widget team ships on Thursdays.\n"
    "\n"
    "The widget cache lives in the shared volume.\n"
    "## Later heading\n"
    "- Not a saved memory.\n"
)


def test_saved_memories_reads_only_the_heading_section() -> None:
    assert hg.saved_memories(_GEMINI_MD) == [
        "The widget service listens on port 8443.",
        "The widget team ships on Thursdays.",
        "The widget cache lives in the shared volume.",
    ]


def test_saved_memories_without_the_heading_is_empty() -> None:
    assert hg.saved_memories("# notes\n- a fact\n") == []


def _gemini_dir(tmp_path: Path) -> Path:
    gdir = tmp_path / "ghome" / ".gemini"
    gdir.mkdir(parents=True)
    (gdir / "GEMINI.md").write_text(_GEMINI_MD, encoding="utf-8")
    return gdir


_COUNT_SQL = {
    "beliefs": "SELECT COUNT(*) FROM beliefs",
    "belief_corroborations": "SELECT COUNT(*) FROM belief_corroborations",
}


def _count(db: Path, table: str) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return int(conn.execute(_COUNT_SQL[table]).fetchone()[0])
    finally:
        conn.close()


def _save(**response: object) -> dict[str, object]:
    return _payload(
        "AfterTool", tool_name="save_memory", tool_input={},
        tool_response=response or {"llmContent": "ok", "returnDisplay": "ok"},
    )


def test_a_saved_memory_becomes_a_belief_once(
    store_env: Path, tmp_path: Path,
) -> None:
    """Killed by: corroborating a fact the store already holds."""
    gdir = _gemini_dir(tmp_path)
    added = hg.mirror_saved_memories(_save(), gemini_dir=gdir)
    assert added == 3
    beliefs = _count(store_env, "beliefs")
    corroborations = _count(store_env, "belief_corroborations")
    assert hg.mirror_saved_memories(_save(), gemini_dir=gdir) == 0
    assert _count(store_env, "beliefs") == beliefs
    assert _count(store_env, "belief_corroborations") == corroborations


@pytest.mark.parametrize(
    "payload",
    [_save(error="widget failure"),
     _payload("AfterTool", tool_name="write_file", tool_response={}),
     _payload("AfterTool", tool_response={})],
)
def test_only_a_successful_save_is_mirrored(
    store_env: Path, tmp_path: Path, payload: dict[str, object],
) -> None:
    gdir = _gemini_dir(tmp_path)
    assert hg.mirror_saved_memories(payload, gemini_dir=gdir) == 0
    assert not store_env.exists()


def test_a_missing_context_file_mirrors_nothing(
    store_env: Path, tmp_path: Path,
) -> None:
    assert hg.mirror_saved_memories(_save(), gemini_dir=tmp_path) == 0
    assert not store_env.exists()


def test_after_tool_runs_the_mirror_from_the_gemini_home(
    store_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gdir = _gemini_dir(tmp_path)
    monkeypatch.setenv("GEMINI_CLI_HOME", str(gdir.parent))
    out, _ = _run(_save())
    assert out == "{}\n"
    assert _count(store_env, "beliefs") == 3


# --- the Gemini home -----------------------------------------------------------


def test_the_gemini_dir_defaults_to_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert resolve_gemini_dir({}) == tmp_path / ".gemini"
    assert resolve_gemini_dir({"GEMINI_CLI_HOME": ""}) == tmp_path / ".gemini"


def test_gemini_cli_home_moves_the_gemini_dir(tmp_path: Path) -> None:
    assert resolve_gemini_dir({"GEMINI_CLI_HOME": str(tmp_path)}) == (
        tmp_path / ".gemini"
    )


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_a_gemini_cli_home_that_is_no_directory_is_refused(
    tmp_path: Path, kind: str,
) -> None:
    target = tmp_path / "ghome"
    if kind == "file":
        target.write_text("x", encoding="utf-8")
    with pytest.raises(GeminiHomeError, match="GEMINI_CLI_HOME"):
        resolve_gemini_dir({"GEMINI_CLI_HOME": str(target)})


# --- the installed console script ---------------------------------------------


def test_the_console_script_prints_one_json_object(
    tmp_path: Path,
) -> None:
    scripts_dir = Path(sys.executable).parent
    script = next(
        (
            candidate
            for suffix in ("", ".exe", ".cmd", ".bat")
            if (candidate := scripts_dir / f"aelf-gemini-hook{suffix}").exists()
        ),
        None,
    )
    assert script is not None, f"aelf-gemini-hook is not installed in {scripts_dir}"
    env = dict(os.environ)
    env["AELFRICE_DB"] = str(tmp_path / "memory.db")
    env["AELFRICE_TRANSCRIPTS_DIR"] = str(tmp_path / "transcripts")
    result = subprocess.run(
        [str(script)],
        input=json.dumps(_payload("PreCompress", trigger="manual")),
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(tmp_path), env=env, timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "{}\n"
