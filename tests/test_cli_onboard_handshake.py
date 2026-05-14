"""CLI tests for the polymorphic onboard handshake flags (#238).

`--emit-candidates` and `--accept-classifications` are the
shell-callable equivalents of the MCP `aelf:onboard` polymorphic
surface, used by the /aelf:onboard slash command to drive the
onboard handshake from a Claude Code session via Haiku Task
subagents (no API key, no network).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.models import ONBOARD_STATE_COMPLETED


@pytest.fixture(autouse=True)
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _run(*argv: str, stdin: str | None = None,
         monkeypatch: pytest.MonkeyPatch | None = None) -> tuple[int, str]:
    if stdin is not None:
        assert monkeypatch is not None, "stdin redirection needs monkeypatch"
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def _populate_repo(root: Path) -> None:
    (root / "README.md").write_text(
        "This project must use uv for environment management.\n\n"
        "We always prefer atomic commits over batched commits.\n\n"
        "The system follows a Bayesian feedback loop with locks.\n"
    )
    (root / "module.py").write_text(
        '"""Top-level module docstring describing the module purpose."""\n\n'
        "def f():\n"
        '    """A top-level function that returns a constant value."""\n'
        "    return 1\n"
    )


def test_emit_candidates_returns_valid_json(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)
    code, out = _run("onboard", str(repo), "--emit-candidates")
    assert code == 0
    payload = json.loads(out)
    assert "session_id" in payload
    assert "n_already_present" in payload
    assert "sentences" in payload
    assert isinstance(payload["sentences"], list)


def test_emit_candidates_persists_pending_session(tmp_path: Path) -> None:
    from aelfrice.cli import db_path
    from aelfrice.store import MemoryStore
    from aelfrice.models import ONBOARD_STATE_PENDING

    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)
    code, out = _run("onboard", str(repo), "--emit-candidates")
    assert code == 0
    sid = json.loads(out)["session_id"]
    store = MemoryStore(str(db_path()))
    try:
        session = store.get_onboard_session(sid)
    finally:
        store.close()
    assert session is not None
    assert session.state == ONBOARD_STATE_PENDING


def test_emit_candidates_requires_path() -> None:
    code, _ = _run("onboard", "--emit-candidates")
    assert code == 2


def test_accept_classifications_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)
    code, out = _run("onboard", str(repo), "--emit-candidates")
    assert code == 0
    payload = json.loads(out)
    sid = payload["session_id"]
    sentences = payload["sentences"]
    assert sentences, "fixture should produce at least one candidate"

    classifications = [
        {"index": s["index"], "belief_type": "factual", "persist": True}
        for s in sentences
    ]
    cls_file = tmp_path / "cls.json"
    cls_file.write_text(json.dumps(classifications))

    code, out = _run(
        "onboard",
        "--accept-classifications",
        "--session-id", sid,
        "--classifications-file", str(cls_file),
    )
    assert code == 0
    summary = json.loads(out)
    assert summary["session_id"] == sid
    assert summary["inserted"] == len(sentences)
    assert summary["skipped_unclassified"] == 0


def test_accept_classifications_reads_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)
    code, out = _run("onboard", str(repo), "--emit-candidates")
    payload = json.loads(out)
    sid = payload["session_id"]
    classifications = [
        {"index": s["index"], "belief_type": "preference", "persist": True}
        for s in payload["sentences"]
    ]
    code, out = _run(
        "onboard",
        "--accept-classifications",
        "--session-id", sid,
        "--classifications-file", "-",
        stdin=json.dumps(classifications),
        monkeypatch=monkeypatch,
    )
    assert code == 0
    summary = json.loads(out)
    assert summary["inserted"] == len(classifications)


def test_accept_classifications_marks_session_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aelfrice.cli import db_path
    from aelfrice.store import MemoryStore

    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)
    code, out = _run("onboard", str(repo), "--emit-candidates")
    payload = json.loads(out)
    sid = payload["session_id"]
    cls_file = tmp_path / "cls.json"
    cls_file.write_text(json.dumps([]))
    code, _ = _run(
        "onboard",
        "--accept-classifications",
        "--session-id", sid,
        "--classifications-file", str(cls_file),
    )
    assert code == 0
    store = MemoryStore(str(db_path()))
    try:
        session = store.get_onboard_session(sid)
    finally:
        store.close()
    assert session is not None
    assert session.state == ONBOARD_STATE_COMPLETED


def test_accept_classifications_requires_session_id(tmp_path: Path) -> None:
    cls_file = tmp_path / "cls.json"
    cls_file.write_text("[]")
    code, _ = _run(
        "onboard",
        "--accept-classifications",
        "--classifications-file", str(cls_file),
    )
    assert code == 2


def test_accept_classifications_requires_file() -> None:
    code, _ = _run(
        "onboard",
        "--accept-classifications",
        "--session-id", "abc123",
    )
    assert code == 2


def test_accept_classifications_rejects_invalid_json(tmp_path: Path) -> None:
    f = tmp_path / "bad.json"
    f.write_text("not json")
    code, _ = _run(
        "onboard",
        "--accept-classifications",
        "--session-id", "abc",
        "--classifications-file", str(f),
    )
    assert code == 1


def test_accept_classifications_rejects_unknown_session(tmp_path: Path) -> None:
    f = tmp_path / "cls.json"
    f.write_text("[]")
    code, _ = _run(
        "onboard",
        "--accept-classifications",
        "--session-id", "deadbeef" * 4,
        "--classifications-file", str(f),
    )
    assert code == 1


# --- onboard --check (#761) --------------------------------------------


def test_check_requires_path() -> None:
    code, _ = _run("onboard", "--check")
    assert code == 2


def test_check_on_empty_dir_reports_zero(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    code, out = _run("onboard", str(repo), "--check")
    assert code == 0
    assert "already present: 0 candidates" in out
    assert "new since last onboard: 0 candidates" in out


def test_check_on_fresh_repo_reports_new_candidates(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)
    code, out = _run("onboard", str(repo), "--check")
    assert code == 0
    assert "already present: 0 candidates" in out
    # at least one candidate extracted from _populate_repo content
    assert "new since last onboard: " in out
    assert "new since last onboard: 0 candidates" not in out


def test_check_does_not_persist_session(tmp_path: Path) -> None:
    """Pre-scan must not leave an onboard_sessions row behind."""
    from aelfrice.cli import db_path
    from aelfrice.store import MemoryStore

    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)
    code, _ = _run("onboard", str(repo), "--check")
    assert code == 0
    store = MemoryStore(str(db_path()))
    try:
        assert store.count_onboard_sessions() == 0
    finally:
        store.close()


def test_check_reports_already_present_after_accept(tmp_path: Path) -> None:
    """A second --check after a real onboard reports the inserted
    candidates as already-present, demonstrating the idempotency signal
    the issue requested."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)

    emit_code, emit_out = _run("onboard", str(repo), "--emit-candidates")
    assert emit_code == 0
    emit_payload = json.loads(emit_out)
    sid = emit_payload["session_id"]
    sentences = emit_payload["sentences"]
    classifications = [
        {"index": s["index"], "belief_type": "factual", "persist": True}
        for s in sentences
    ]
    cf = tmp_path / "classifications.json"
    cf.write_text(json.dumps(classifications))
    accept_code, _ = _run(
        "onboard",
        "--accept-classifications",
        "--session-id", sid,
        "--classifications-file", str(cf),
    )
    assert accept_code == 0

    check_code, check_out = _run("onboard", str(repo), "--check")
    assert check_code == 0
    assert f"already present: {len(sentences)} candidates" in check_out
    assert "new since last onboard: 0 candidates" in check_out


def test_check_bypasses_emit_candidates(tmp_path: Path) -> None:
    """--check must short-circuit before --emit-candidates path even when
    both flags are passed. The pre-scan path is read-only; the emit path
    persists a session. --check wins."""
    from aelfrice.cli import db_path
    from aelfrice.store import MemoryStore

    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)
    code, out = _run(
        "onboard", str(repo), "--check", "--emit-candidates"
    )
    assert code == 0
    # human-readable text, not JSON
    assert "path: " in out
    assert "session_id" not in out
    store = MemoryStore(str(db_path()))
    try:
        assert store.count_onboard_sessions() == 0
    finally:
        store.close()


# --- #801 rejection-ledger CLI wiring -----------------------------------


def _reject_all(repo: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    """Run emit + accept-all-persist:false. Returns the number of
    sentences rejected.
    """
    _, emit_out = _run("onboard", str(repo), "--emit-candidates")
    payload = json.loads(emit_out)
    sid = payload["session_id"]
    sentences = payload["sentences"]
    cls = [
        {"index": s["index"], "belief_type": "factual", "persist": False}
        for s in sentences
    ]
    code, _ = _run(
        "onboard",
        "--accept-classifications",
        "--session-id", sid,
        "--classifications-file", "-",
        stdin=json.dumps(cls),
        monkeypatch=monkeypatch,
    )
    assert code == 0
    return len(sentences)


def test_emit_candidates_json_exposes_n_already_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)
    n = _reject_all(repo, monkeypatch)
    assert n > 0
    code, out = _run("onboard", str(repo), "--emit-candidates")
    assert code == 0
    payload = json.loads(out)
    assert payload["n_already_rejected"] == n
    assert payload["sentences"] == []


def test_check_reports_already_rejected_after_persist_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)
    n = _reject_all(repo, monkeypatch)
    assert n > 0
    code, out = _run("onboard", str(repo), "--check")
    assert code == 0
    assert f"already rejected: {n} candidates" in out
    assert "new since last onboard: 0 candidates" in out


def test_force_flag_re_emits_previously_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)
    n = _reject_all(repo, monkeypatch)
    assert n > 0
    code, out = _run("onboard", str(repo), "--emit-candidates", "--force")
    assert code == 0
    payload = json.loads(out)
    assert len(payload["sentences"]) == n
    assert payload["n_already_rejected"] == 0


def test_check_force_notes_ledger_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _populate_repo(repo)
    n = _reject_all(repo, monkeypatch)
    assert n > 0
    code, out = _run("onboard", str(repo), "--check", "--force")
    assert code == 0
    assert "--force: ledger bypassed" in out
    assert f"new since last onboard: {n} candidates" in out
