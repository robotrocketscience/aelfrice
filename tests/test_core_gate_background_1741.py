"""Background labeling of core-gate batches (#1741).

The Stop hook records a session-end batch and starts a detached child that
runs the host's `claude` CLI; nothing is written to the hook's stdout. These
tests never run a real CLI: the hook tests replace `find_claude` and `spawn`,
and the runner tests pass a fake `run`. Each test names the mutation that
kills it. Every store is under `tmp_path`, and the belief text is synthetic.
"""
from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import aelfrice.core_gate_background as bg
import aelfrice.hook as hook
from aelfrice.core_gate import CLASSIFIER_VERSION, MAX_BATCH, build_prompt
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
    CoreGateBatchItem,
)
from aelfrice.store import MemoryStore

pytestmark = pytest.mark.timeout(60)

SESSION = "widgetsession-bg"
CLAUDE = "/opt/widget/bin/claude"


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DB", str(path))
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("AELFRICE_NO_AUTO_INSTALL", "1")
    monkeypatch.delenv(hook.CORE_GATE_SESSION_END_ENV, raising=False)
    monkeypatch.delenv(bg.BACKGROUND_ENV, raising=False)
    return path


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    """Report `claude` on PATH and record each spawn instead of running it."""
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(bg, "find_claude", lambda: CLAUDE)

    def fake_spawn(batch_id: str, *, db_path: str, claude: str) -> bool:
        calls.append((batch_id, db_path, claude))
        return True

    monkeypatch.setattr(bg, "spawn", fake_spawn)
    return calls


def _belief(bid: str, *, content: str | None = None, created: str = "2026-10-01T00:00:00Z") -> Belief:
    """A belief with the (3, 1) prior, which passes the posterior core arm."""
    return Belief(
        id=bid,
        content=content if content is not None else f"The widgetbg {bid} setting is stable.",
        content_hash=f"hash-{bid}",
        alpha=3.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
        locked_at=None, created_at=created, last_retrieved_at=None,
        session_id=SESSION, origin=ORIGIN_USER_TRANSCRIPT,
    )


def _seed(db: Path, beliefs: list[Belief]) -> None:
    store = MemoryStore(str(db))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _stop(
    tmp_path: Path, *, env: dict[str, str] | None = None,
    extra: dict[str, object] | None = None,
) -> tuple[str, str]:
    payload: dict[str, object] = {"session_id": SESSION, "cwd": str(tmp_path)}
    payload.update(extra or {})
    out, err = io.StringIO(), io.StringIO()
    assert hook.stop(
        stdin=io.StringIO(json.dumps(payload)), stdout=out, stderr=err,
        env=env if env is not None else {},
    ) == 0
    return out.getvalue(), err.getvalue()


def _batches(db: Path) -> list[tuple[str, str, list[str]]]:
    """(batch_id, origin, belief ids in index order) per batch."""
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT batch_id, origin, items_json FROM core_gate_batches ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()
    return [
        (bid, origin, [i["belief_id"] for i in sorted(json.loads(items), key=lambda i: i["index"])])
        for bid, origin, items in rows
    ]


# --- the Stop hook ---------------------------------------------------------


def test_default_spawns_and_writes_nothing(
    db: Path, tmp_path: Path, spawned: list[tuple[str, str, str]],
) -> None:
    """Killed by: not taking the background path by default, writing the
    continuation as well, or spawning without the batch's id, store, or CLI."""
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path)
    assert out == ""
    batches = _batches(db)
    assert [(o, ids) for _, o, ids in batches] == [("session_end", ["w1"])]
    assert spawned == [(batches[0][0], str(db), CLAUDE)]


def test_no_candidates_no_spawn(
    db: Path, tmp_path: Path, spawned: list[tuple[str, str, str]],
) -> None:
    """Killed by: spawning when every candidate is already claimed. The
    first Stop batches w1; the second finds nothing new."""
    _seed(db, [_belief("w1")])
    _stop(tmp_path)
    assert len(spawned) == 1
    out, _ = _stop(tmp_path)
    assert out == ""
    assert len(spawned) == 1
    assert len(_batches(db)) == 1


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_env_off_skips_background(
    db: Path, tmp_path: Path, spawned: list[tuple[str, str, str]], value: str,
) -> None:
    """Killed by: ignoring the env override. With background off and no
    session_end opt-in, nothing is batched or written."""
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path, env={bg.BACKGROUND_ENV: value})
    assert (out, spawned, _batches(db)) == ("", [], [])


def test_env_off_falls_back_to_the_opt_in_turn(
    db: Path, tmp_path: Path, spawned: list[tuple[str, str, str]],
) -> None:
    """Killed by: returning on background-off before the #1740 opt-in path."""
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path, env={bg.BACKGROUND_ENV: "0", hook.CORE_GATE_SESSION_END_ENV: "1"})
    assert json.loads(out)["hookSpecificOutput"]["hookEventName"] == "Stop"
    assert spawned == []


def test_background_wins_over_the_opt_in_turn(
    db: Path, tmp_path: Path, spawned: list[tuple[str, str, str]],
) -> None:
    """Killed by: falling through to the #1740 continuation after a spawn,
    which would add the turn this issue removes."""
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path, env={hook.CORE_GATE_SESSION_END_ENV: "1"})
    assert out == ""
    assert len(spawned) == 1
    assert len(_batches(db)) == 1


def test_toml_off_skips_background(
    db: Path, tmp_path: Path, spawned: list[tuple[str, str, str]],
) -> None:
    """Killed by: not reading `[core_gate] background`."""
    (tmp_path / ".aelfrice.toml").write_text("[core_gate]\nbackground = false\n", encoding="utf-8")
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path)
    assert (out, spawned, _batches(db)) == ("", [], [])


def test_env_on_overrides_toml_off(
    db: Path, tmp_path: Path, spawned: list[tuple[str, str, str]],
) -> None:
    """Killed by: reading the TOML key when the env var says on."""
    (tmp_path / ".aelfrice.toml").write_text("[core_gate]\nbackground = false\n", encoding="utf-8")
    _seed(db, [_belief("w1")])
    _stop(tmp_path, env={bg.BACKGROUND_ENV: "1"})
    assert len(spawned) == 1


def test_non_bool_toml_reads_as_on(
    db: Path, tmp_path: Path, spawned: list[tuple[str, str, str]],
) -> None:
    """Killed by: a default of False on the non-bool branch, or no trace."""
    (tmp_path / ".aelfrice.toml").write_text('[core_gate]\nbackground = "nah"\n', encoding="utf-8")
    _seed(db, [_belief("w1")])
    _, err = _stop(tmp_path)
    assert len(spawned) == 1
    assert "[core_gate] background (expected bool)" in err


def test_no_cli_no_batch(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: recording a batch, or spawning, without a CLI to run it.
    The beliefs stay unclaimed for a later Stop or the drain."""
    calls: list[str] = []
    monkeypatch.setattr(bg, "find_claude", lambda: None)
    monkeypatch.setattr(bg, "spawn", lambda batch_id, **_: calls.append(batch_id) or True)
    _seed(db, [_belief("w1")])
    out, _ = _stop(tmp_path)
    assert (out, calls, _batches(db)) == ("", [], [])


@pytest.mark.parametrize("case", ["stop_hook_active", "headless", "codex"])
def test_guards_apply_to_background(
    db: Path, tmp_path: Path, spawned: list[tuple[str, str, str]], case: str,
) -> None:
    """Killed by: moving the background path above any of the three guards."""
    _seed(db, [_belief("w1")])
    if case == "stop_hook_active":
        _stop(tmp_path, extra={"stop_hook_active": True})
    elif case == "headless":
        _stop(tmp_path, env={"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"})
    else:
        _stop(tmp_path, extra={"turn_id": "widget-turn"})
    assert (spawned, _batches(db)) == ([], [])


def test_batch_is_capped_at_max_batch(
    db: Path, tmp_path: Path, spawned: list[tuple[str, str, str]],
) -> None:
    """Killed by: batching past MAX_BATCH (the store refuses, so nothing
    spawns), or dropping the cap's newest-first order."""
    beliefs = [
        _belief(f"w{i:03d}", created=f"2026-10-01T00:{i // 60:02d}:{i % 60:02d}Z")
        for i in range(MAX_BATCH + 3)
    ]
    _seed(db, beliefs)
    _stop(tmp_path)
    (_, _, ids), = _batches(db)
    # Candidates come newest first (rowid DESC), so the cut keeps the
    # newest MAX_BATCH and leaves the three oldest for the next Stop.
    newest_first = [b.id for b in reversed(beliefs)]
    assert ids == newest_first[:MAX_BATCH]
    assert len(spawned) == 1
    _stop(tmp_path)
    assert [i for _, _, i in _batches(db)][1] == newest_first[MAX_BATCH:]


def test_overflow_never_takes_the_opt_in_turn(
    db: Path, tmp_path: Path, spawned: list[tuple[str, str, str]],
) -> None:
    """Killed by: falling through to the #1740 continuation after a spawn.
    Past MAX_BATCH candidates, the leftovers would get the turn."""
    _seed(db, [_belief(f"w{i:03d}") for i in range(MAX_BATCH + 3)])
    out, _ = _stop(tmp_path, env={hook.CORE_GATE_SESSION_END_ENV: "1"})
    assert out == ""
    assert len(_batches(db)) == 1


# --- the runner --------------------------------------------------------------


def test_runner_argv_is_the_measured_configuration() -> None:
    """Killed by: dropping or changing any measured flag."""
    assert bg.runner_argv(CLAUDE) == [
        CLAUDE, "-p", "--model", "haiku", "--tools", "",
        "--settings", '{"disableAllHooks": true}', "--no-session-persistence",
        "--output-format", "json", "--max-budget-usd", "0.10",
        "--system-prompt", bg.RUNNER_SYSTEM_PROMPT,
        "--strict-mcp-config", "--disable-slash-commands",
    ]


def test_runner_env_keeps_thinking_and_drops_context() -> None:
    """Killed by: keeping a thinking cap, or not setting either disable."""
    env = bg.runner_env({"MAX_THINKING_TOKENS": "0", "PATH": "/x"})
    assert "MAX_THINKING_TOKENS" not in env
    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert env["PATH"] == "/x"


@pytest.mark.parametrize(("raw", "want"), [
    ('[{"index": 0}]', '[{"index": 0}]'),
    ('```json\n[1]\n```', "[1]"),
    ('```\n[1]\n```', "[1]"),
    ('  ```json\n[1]\n```\n', "[1]"),
    ("```json [1]", "```json [1]"),
    ("```json\n[1]", "```json\n[1]"),
])
def test_strip_fence(raw: str, want: str) -> None:
    """Killed by: stripping an unclosed or single-line fence, or not
    stripping a closed one."""
    assert bg.strip_fence(raw) == want


def _open_batch(db: Path, ids: list[str], *, version: str | None = None) -> tuple[MemoryStore, str]:
    store = MemoryStore(str(db))
    batch_id = store.create_core_gate_batch(
        [CoreGateBatchItem(index=i, belief_id=b, content_hash=f"hash-{b}") for i, b in enumerate(ids)],
        classifier_version=version or str(CLASSIFIER_VERSION),
        origin="session_end", session_id=SESSION, created_at="2026-10-01T00:00:00Z",
    )
    return store, batch_id


class FakeRun:
    def __init__(self, *, stdout: str = "", returncode: int = 0, exc: Exception | None = None):
        self.stdout, self.returncode, self.exc = stdout, returncode, exc
        self.calls: list[dict[str, Any]] = []

    def __call__(self, argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append({"argv": argv, **kw})
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, "")


def _reply(labels: list[tuple[int, str]], *, fence: bool = True, cost: float = 0.04) -> str:
    body = json.dumps([{"index": i, "label": lab} for i, lab in labels])
    if fence:
        body = f"```json\n{body}\n```"
    return json.dumps({"type": "result", "result": body, "total_cost_usd": cost})


def _labels(db: Path) -> dict[str, str]:
    conn = sqlite3.connect(str(db))
    try:
        return dict(conn.execute("SELECT content_hash, label FROM core_gate_labels").fetchall())
    finally:
        conn.close()


def test_label_batch_accepts_a_fenced_reply(db: Path) -> None:
    """Killed by: not stripping the fence, sending a prompt other than
    `build_prompt` over the batch, or not caching the labels."""
    _seed(db, [_belief("w1"), _belief("w2")])
    store, batch_id = _open_batch(db, ["w1", "w2"])
    run = FakeRun(stdout=_reply([(0, "A"), (1, "C")]))
    try:
        record = bg.label_batch(store, batch_id, claude=CLAUDE, run=run, env={})
    finally:
        store.close()
    assert record["outcome"] == bg.OUTCOME_ACCEPTED
    assert record["cost_usd"] == 0.04
    assert _labels(db) == {"hash-w1": "A", "hash-w2": "C"}
    (call,) = run.calls
    assert call["argv"] == bg.runner_argv(CLAUDE)
    assert call["input"] == build_prompt([
        (0, "The widgetbg w1 setting is stable."), (1, "The widgetbg w2 setting is stable."),
    ])
    assert call["timeout"] == bg.RUNNER_TIMEOUT_SECONDS
    assert call["env"]["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"


@pytest.mark.parametrize(("name", "run", "outcome"), [
    ("nonzero exit", FakeRun(stdout=_reply([(0, "A")]), returncode=1), bg.OUTCOME_RUN_FAILED),
    ("not json", FakeRun(stdout="widget"), bg.OUTCOME_RUN_FAILED),
    ("json not an object", FakeRun(stdout="[1]"), bg.OUTCOME_RUN_FAILED),
    ("no result", FakeRun(stdout=json.dumps({"type": "result"})), bg.OUTCOME_BAD_REPLY),
    ("partial labels", FakeRun(stdout=_reply([(0, "A")])), bg.OUTCOME_BAD_REPLY),
    ("timeout", FakeRun(exc=subprocess.TimeoutExpired("claude", 300)), bg.OUTCOME_RUN_FAILED),
    ("no cli", FakeRun(exc=FileNotFoundError()), bg.OUTCOME_RUN_FAILED),
])
def test_label_batch_writes_nothing_on_failure(
    db: Path, name: str, run: FakeRun, outcome: str,
) -> None:
    """Killed by: accepting on any failed run or partial reply."""
    _seed(db, [_belief("w1"), _belief("w2")])
    store, batch_id = _open_batch(db, ["w1", "w2"])
    try:
        record = bg.label_batch(store, batch_id, claude=CLAUDE, run=run, env={})
    finally:
        store.close()
    assert record["outcome"] == outcome, name
    assert _labels(db) == {}


def test_label_batch_skips_unknown_spent_and_stale(db: Path) -> None:
    """Killed by: running the CLI for a batch that can't be accepted."""
    _seed(db, [_belief("w1")])
    run = FakeRun(stdout=_reply([(0, "A")]))
    store, batch_id = _open_batch(db, ["w1"])
    try:
        assert bg.label_batch(store, "widget-none", claude=CLAUDE, run=run, env={})["outcome"] == bg.OUTCOME_NO_BATCH
        assert bg.label_batch(store, batch_id, claude=CLAUDE, run=run, env={})["outcome"] == bg.OUTCOME_ACCEPTED
        assert bg.label_batch(store, batch_id, claude=CLAUDE, run=run, env={})["outcome"] == bg.OUTCOME_NOT_OPEN
        assert len(run.calls) == 1
        _, old = _open_batch(db, ["w1"], version="core-gate-widgetold")
        assert bg.label_batch(store, old, claude=CLAUDE, run=run, env={})["outcome"] == bg.OUTCOME_STALE_VERSION
        assert len(run.calls) == 1
    finally:
        store.close()


def test_label_batch_refuses_a_changed_belief(db: Path) -> None:
    """Killed by: labeling content other than the hash the batch names."""
    _seed(db, [_belief("w1")])
    store, batch_id = _open_batch(db, ["w1"])
    store._conn.execute("UPDATE beliefs SET content_hash = 'hash-other' WHERE id = 'w1'")  # noqa: SLF001
    store._conn.commit()  # noqa: SLF001
    run = FakeRun(stdout=_reply([(0, "A")]))
    try:
        assert bg.label_batch(store, batch_id, claude=CLAUDE, run=run, env={})["outcome"] == bg.OUTCOME_STALE_BELIEF
    finally:
        store.close()
    assert run.calls == []


# --- the spawn -----------------------------------------------------------------


def test_spawn_detaches_the_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed by: not detaching, not passing the store path, or a wrong
    child command."""
    seen: dict[str, Any] = {}

    class FakePopen:
        def __init__(self, argv: list[str], **kw: Any) -> None:
            seen["argv"], seen["kw"] = argv, kw

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    assert bg.spawn("widgetbatch", db_path="/x/memory.db", claude=CLAUDE) is True
    assert seen["argv"] == [sys.executable, "-m", "aelfrice.core_gate_background", "widgetbatch", CLAUDE]
    assert seen["kw"]["env"]["AELFRICE_DB"] == "/x/memory.db"
    assert seen["kw"]["stdout"] is subprocess.DEVNULL
    if sys.platform != "win32":
        assert seen["kw"]["start_new_session"] is True


def test_spawn_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed by: letting a Popen failure reach the hook."""
    def boom(*_a: Any, **_k: Any) -> None:
        raise OSError("widget")

    monkeypatch.setattr(subprocess, "Popen", boom)
    assert bg.spawn("widgetbatch", db_path="/x/memory.db", claude=CLAUDE) is False


def test_main_logs_one_record_without_text(
    db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: not logging the outcome, or logging snippet text."""
    _seed(db, [_belief("w1")])
    store, batch_id = _open_batch(db, ["w1"])
    store.close()
    monkeypatch.setattr(bg.subprocess, "run", FakeRun(stdout=_reply([(0, "B")])))
    assert bg.main([batch_id, CLAUDE]) == 0
    lines = (db.parent / bg.LOG_FILENAME).read_text(encoding="utf-8").splitlines()
    (rec,) = [json.loads(x) for x in lines]
    assert rec["batch_id"] == batch_id and rec["outcome"] == bg.OUTCOME_ACCEPTED
    assert "widgetbg" not in lines[0]
    assert bg.main(["only-one"]) == 2
