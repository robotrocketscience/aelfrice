"""Tests for `aelf label` (#859) — interactive corpus-row labelling.

Covers happy-path one row, --resume skips, invalid-input re-prompt,
`0`-skip behaviour, EOF-mid-prompt clean shutdown, and Ctrl-C between
rows. All in-process via `aelfrice.cli.main(argv=...)` with a stdin
StringIO injected through args, so no subprocess.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from aelfrice import cli as cli_mod
from aelfrice.label_cli import _parse_indices, cmd_label


def _stub_rows(tmp_path: Path, *rows: dict[str, Any]) -> Path:
    p = tmp_path / "stubs.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return p


def _row(rid: str, *bids: str) -> dict[str, Any]:
    return {
        "id": rid,
        "query": f"q for {rid}",
        "beliefs": [{"id": b, "text": f"belief text {b}"} for b in bids],
    }


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class _Args:
    def __init__(self, **kw: Any) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


# ---------- _parse_indices unit ----------

def test_parse_indices_basic() -> None:
    assert _parse_indices("1,3,7", 10) == [0, 2, 6]


def test_parse_indices_zero_means_skip() -> None:
    assert _parse_indices("0", 10) == []


def test_parse_indices_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        _parse_indices("11", 10)


def test_parse_indices_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        _parse_indices("2,2", 10)


def test_parse_indices_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        _parse_indices("foo", 10)


def test_parse_indices_rejects_empty() -> None:
    with pytest.raises(ValueError):
        _parse_indices("", 10)


# ---------- cmd_label integration ----------

def test_happy_path_one_row(tmp_path: Path) -> None:
    stubs = _stub_rows(tmp_path, _row("rr-001", "b-a", "b-b", "b-c"))
    out_path = tmp_path / "out.jsonl"
    stdin = io.StringIO("1,3\n\nthis is the labeller note\n")
    out_stream = io.StringIO()
    args = _Args(
        module="rerank_relevance",
        input=str(stubs),
        output=str(out_path),
        resume=False,
        k=10,
        no_ordering=False,
        stdin=stdin,
    )
    rc = cmd_label(args, out_stream)
    assert rc == 0, out_stream.getvalue()
    rows = _read_jsonl(out_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "rr-001"
    assert row["label"] == "graded"
    assert row["provenance"] == "synthetic-v0.1"
    assert row["gold_top_k"] == ["b-a", "b-c"]
    assert "gold_ordering" not in row
    assert row["k"] == 10
    assert row["labeller_note"] == "this is the labeller note"


def test_gold_ordering_captured(tmp_path: Path) -> None:
    stubs = _stub_rows(tmp_path, _row("rr-002", "b-a", "b-b", "b-c"))
    out_path = tmp_path / "out.jsonl"
    stdin = io.StringIO("1,3\n1,3,2\nnote\n")
    args = _Args(
        module="rerank_relevance",
        input=str(stubs),
        output=str(out_path),
        resume=False,
        k=10,
        no_ordering=False,
        stdin=stdin,
    )
    rc = cmd_label(args, io.StringIO())
    assert rc == 0
    rows = _read_jsonl(out_path)
    assert rows[0]["gold_ordering"] == ["b-a", "b-c", "b-b"]


def test_gold_ordering_must_contain_top_k(tmp_path: Path) -> None:
    stubs = _stub_rows(tmp_path, _row("rr-003", "b-a", "b-b", "b-c"))
    out_path = tmp_path / "out.jsonl"
    # First ordering omits b-c (id 3); re-prompt; second is valid.
    stdin = io.StringIO("1,3\n1,2\n1,3,2\nnote\n")
    out_stream = io.StringIO()
    args = _Args(
        module="rerank_relevance",
        input=str(stubs),
        output=str(out_path),
        resume=False,
        k=10,
        no_ordering=False,
        stdin=stdin,
    )
    rc = cmd_label(args, out_stream)
    assert rc == 0
    assert "must contain every gold_top_k id" in out_stream.getvalue()
    rows = _read_jsonl(out_path)
    assert rows[0]["gold_ordering"] == ["b-a", "b-c", "b-b"]


def test_zero_skips_row(tmp_path: Path) -> None:
    stubs = _stub_rows(tmp_path, _row("rr-004", "b-a", "b-b"))
    out_path = tmp_path / "out.jsonl"
    stdin = io.StringIO("0\n")
    args = _Args(
        module="rerank_relevance",
        input=str(stubs),
        output=str(out_path),
        resume=False,
        k=10,
        no_ordering=False,
        stdin=stdin,
    )
    rc = cmd_label(args, io.StringIO())
    assert rc == 0
    assert not out_path.exists() or _read_jsonl(out_path) == []


def test_invalid_top_k_reprompts(tmp_path: Path) -> None:
    stubs = _stub_rows(tmp_path, _row("rr-005", "b-a", "b-b"))
    out_path = tmp_path / "out.jsonl"
    # "9" out of range → re-prompt; "1" valid; no-ordering blank; note.
    stdin = io.StringIO("9\n1\n\nfine\n")
    out_stream = io.StringIO()
    args = _Args(
        module="rerank_relevance",
        input=str(stubs),
        output=str(out_path),
        resume=False,
        k=10,
        no_ordering=False,
        stdin=stdin,
    )
    rc = cmd_label(args, out_stream)
    assert rc == 0
    assert "out of range" in out_stream.getvalue()
    rows = _read_jsonl(out_path)
    assert rows[0]["gold_top_k"] == ["b-a"]


def test_no_ordering_flag_skips_ordering_prompt(tmp_path: Path) -> None:
    stubs = _stub_rows(tmp_path, _row("rr-006", "b-a", "b-b"))
    out_path = tmp_path / "out.jsonl"
    stdin = io.StringIO("1\nfine\n")
    args = _Args(
        module="rerank_relevance",
        input=str(stubs),
        output=str(out_path),
        resume=False,
        k=10,
        no_ordering=True,
        stdin=stdin,
    )
    rc = cmd_label(args, io.StringIO())
    assert rc == 0
    rows = _read_jsonl(out_path)
    assert "gold_ordering" not in rows[0]


def test_resume_skips_existing_ids(tmp_path: Path) -> None:
    stubs = _stub_rows(
        tmp_path,
        _row("rr-007", "b-a", "b-b"),
        _row("rr-008", "b-a", "b-b"),
    )
    out_path = tmp_path / "out.jsonl"
    # Pre-populate output with rr-007.
    out_path.write_text(
        json.dumps({"id": "rr-007", "label": "graded"}) + "\n",
        encoding="utf-8",
    )
    stdin = io.StringIO("1\n\nnote-for-rr-008\n")
    args = _Args(
        module="rerank_relevance",
        input=str(stubs),
        output=str(out_path),
        resume=True,
        k=10,
        no_ordering=False,
        stdin=stdin,
    )
    rc = cmd_label(args, io.StringIO())
    assert rc == 0
    rows = _read_jsonl(out_path)
    assert [r["id"] for r in rows] == ["rr-007", "rr-008"]


def test_empty_note_reprompts(tmp_path: Path) -> None:
    stubs = _stub_rows(tmp_path, _row("rr-009", "b-a"))
    out_path = tmp_path / "out.jsonl"
    stdin = io.StringIO("1\n\n   \nfinal note\n")
    out_stream = io.StringIO()
    args = _Args(
        module="rerank_relevance",
        input=str(stubs),
        output=str(out_path),
        resume=False,
        k=10,
        no_ordering=False,
        stdin=stdin,
    )
    rc = cmd_label(args, out_stream)
    assert rc == 0
    assert "labeller_note must be non-empty" in out_stream.getvalue()
    rows = _read_jsonl(out_path)
    assert rows[0]["labeller_note"] == "final note"


def test_eof_mid_prompt_returns_cleanly(tmp_path: Path) -> None:
    stubs = _stub_rows(
        tmp_path,
        _row("rr-010", "b-a"),
        _row("rr-011", "b-a"),
    )
    out_path = tmp_path / "out.jsonl"
    # First row labelled successfully; second row EOF at top-k prompt.
    stdin = io.StringIO("1\n\nnote-10\n")
    args = _Args(
        module="rerank_relevance",
        input=str(stubs),
        output=str(out_path),
        resume=False,
        k=10,
        no_ordering=False,
        stdin=stdin,
    )
    # EOFError on second row's top-k prompt propagates as a SKIP, not crash.
    rc = cmd_label(args, io.StringIO())
    assert rc == 0
    rows = _read_jsonl(out_path)
    assert [r["id"] for r in rows] == ["rr-010"]


def test_keyboard_interrupt_flushes_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = _stub_rows(
        tmp_path,
        _row("rr-012", "b-a"),
        _row("rr-013", "b-a"),
    )
    out_path = tmp_path / "out.jsonl"
    # First row labelled; second row raises KeyboardInterrupt at top-k.
    call_count = {"n": 0}
    pre_stdin = io.StringIO("1\n\nnote-12\n")

    class IntStdin:
        def readline(self) -> str:
            call_count["n"] += 1
            if call_count["n"] <= 3:
                return pre_stdin.readline()
            raise KeyboardInterrupt

    args = _Args(
        module="rerank_relevance",
        input=str(stubs),
        output=str(out_path),
        resume=False,
        k=10,
        no_ordering=False,
        stdin=IntStdin(),
    )
    out_stream = io.StringIO()
    rc = cmd_label(args, out_stream)
    assert rc == 130
    rows = _read_jsonl(out_path)
    assert [r["id"] for r in rows] == ["rr-012"]
    assert "interrupted" in out_stream.getvalue()


def test_missing_input_returns_error(tmp_path: Path) -> None:
    out_path = tmp_path / "out.jsonl"
    args = _Args(
        module="rerank_relevance",
        input=str(tmp_path / "nope.jsonl"),
        output=str(out_path),
        resume=False,
        k=10,
        no_ordering=False,
        stdin=io.StringIO(""),
    )
    rc = cmd_label(args, io.StringIO())
    assert rc == 2


def test_malformed_stub_row_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("not json\n", encoding="utf-8")
    args = _Args(
        module="rerank_relevance",
        input=str(p),
        output=str(tmp_path / "out.jsonl"),
        resume=False,
        k=10,
        no_ordering=False,
        stdin=io.StringIO(""),
    )
    with pytest.raises(SystemExit):
        cmd_label(args, io.StringIO())


def test_malformed_belief_item_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    # belief is a bare string instead of {id, text}
    p.write_text(
        json.dumps({"id": "rr-x", "query": "q", "beliefs": ["just-a-string"]}) + "\n",
        encoding="utf-8",
    )
    args = _Args(
        module="rerank_relevance",
        input=str(p),
        output=str(tmp_path / "out.jsonl"),
        resume=False,
        k=10,
        no_ordering=False,
        stdin=io.StringIO(""),
    )
    with pytest.raises(SystemExit, match="must be an object"):
        cmd_label(args, io.StringIO())


def test_belief_missing_id_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(
        json.dumps({"id": "rr-y", "query": "q", "beliefs": [{"text": "no id here"}]}) + "\n",
        encoding="utf-8",
    )
    args = _Args(
        module="rerank_relevance",
        input=str(p),
        output=str(tmp_path / "out.jsonl"),
        resume=False,
        k=10,
        no_ordering=False,
        stdin=io.StringIO(""),
    )
    with pytest.raises(SystemExit, match="missing non-empty string 'id'"):
        cmd_label(args, io.StringIO())


def test_cli_main_dispatches_to_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stubs = _stub_rows(tmp_path, _row("rr-014", "b-a"))
    out_path = tmp_path / "out.jsonl"
    monkeypatch.setattr("sys.stdin", io.StringIO("1\n\nvia-main\n"))
    rc = cli_mod.main(
        [
            "label", "rerank_relevance",
            "--input", str(stubs),
            "--output", str(out_path),
        ],
        out=io.StringIO(),
    )
    assert rc == 0
    rows = _read_jsonl(out_path)
    assert rows[0]["labeller_note"] == "via-main"
