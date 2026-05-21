"""Unit tests for src/aelfrice/cadence_score.py (#875 scoring)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.cadence import (
    CADENCE_SHADOW_DIRNAME,
    POLICY_OFF,
    POLICY_P1_EVERY_K_TURNS,
    POLICY_P2_CTX_THRESHOLD,
)
from aelfrice.cadence_score import (
    compute_summary,
    format_report,
    iter_shadow_rows,
    resolve_shadow_dir,
)


def _row(
    *,
    ts: str = "2026-05-20T00:00:00Z",
    sid: str = "s1",
    selected: str = POLICY_P2_CTX_THRESHOLD,
    fired: bool = False,
    p1: bool = False,
    p2: bool = False,
) -> dict:
    return {
        "ts": ts,
        "session_id": sid,
        "selected": selected,
        "fired": fired,
        "shadow": {
            POLICY_P1_EVERY_K_TURNS: {"would_fire": p1, "reason": "p1r"},
            POLICY_P2_CTX_THRESHOLD: {"would_fire": p2, "reason": "p2r"},
        },
    }


# --- compute_summary ------------------------------------------------------


def test_compute_summary_empty() -> None:
    summ = compute_summary([])
    assert summ.total_rows == 0
    assert summ.sessions == 0
    assert summ.earliest_ts is None
    assert summ.per_policy_fire_count == {}
    # Agreement keys exist with zero counts.
    assert summ.agreement_matrix[(False, False)] == 0


def test_compute_summary_totals_and_sessions() -> None:
    rows = [_row(sid="a"), _row(sid="b"), _row(sid="a")]
    summ = compute_summary(rows)
    assert summ.total_rows == 3
    assert summ.sessions == 2


def test_compute_summary_timestamp_range() -> None:
    rows = [
        _row(ts="2026-05-20T00:00:01Z"),
        _row(ts="2026-05-20T00:00:00Z"),
        _row(ts="2026-05-20T00:00:02Z"),
    ]
    summ = compute_summary(rows)
    assert summ.earliest_ts == "2026-05-20T00:00:00Z"
    assert summ.latest_ts == "2026-05-20T00:00:02Z"


def test_compute_summary_per_policy_fire_counts() -> None:
    rows = [
        _row(p1=True, p2=False),
        _row(p1=True, p2=True),
        _row(p1=False, p2=True),
        _row(p1=False, p2=False),
    ]
    summ = compute_summary(rows)
    assert summ.per_policy_fire_count[POLICY_P1_EVERY_K_TURNS] == 2
    assert summ.per_policy_fire_count[POLICY_P2_CTX_THRESHOLD] == 2
    assert summ.per_policy_total[POLICY_P1_EVERY_K_TURNS] == 4
    assert summ.per_policy_total[POLICY_P2_CTX_THRESHOLD] == 4


def test_compute_summary_selected_fire_rate() -> None:
    rows = [
        _row(selected=POLICY_P2_CTX_THRESHOLD, fired=True),
        _row(selected=POLICY_P2_CTX_THRESHOLD, fired=False),
        _row(selected=POLICY_OFF, fired=False),
    ]
    summ = compute_summary(rows)
    assert summ.selected_fire_count[POLICY_P2_CTX_THRESHOLD] == 1
    assert summ.selected_total[POLICY_P2_CTX_THRESHOLD] == 2
    assert summ.selected_total[POLICY_OFF] == 1
    assert summ.selected_fire_count.get(POLICY_OFF, 0) == 0


def test_compute_summary_agreement_matrix() -> None:
    rows = [
        _row(p1=True, p2=True),
        _row(p1=True, p2=True),
        _row(p1=True, p2=False),
        _row(p1=False, p2=True),
        _row(p1=False, p2=False),
        _row(p1=False, p2=False),
        _row(p1=False, p2=False),
    ]
    summ = compute_summary(rows)
    assert summ.agreement_matrix[(True, True)] == 2
    assert summ.agreement_matrix[(True, False)] == 1
    assert summ.agreement_matrix[(False, True)] == 1
    assert summ.agreement_matrix[(False, False)] == 3


def test_compute_summary_session_filter() -> None:
    rows = [_row(sid="x", p1=True), _row(sid="y", p1=False)]
    summ = compute_summary(rows, session_filter="x")
    assert summ.total_rows == 1
    assert summ.per_policy_fire_count.get(POLICY_P1_EVERY_K_TURNS) == 1


def test_compute_summary_skips_malformed_decision() -> None:
    rows = [{
        "ts": "x",
        "session_id": "s",
        "selected": "off",
        "fired": False,
        "shadow": {
            "p1_every_k_turns": "not-a-dict",
            "p2_ctx_threshold": {"would_fire": "not-a-bool"},
        },
    }]
    summ = compute_summary(rows)
    # No policy counts increment because both decisions are malformed.
    assert summ.per_policy_total == {}


# --- iter_shadow_rows -----------------------------------------------------


def test_iter_shadow_rows_missing_dir(tmp_path: Path) -> None:
    rows = list(iter_shadow_rows(tmp_path / "nope"))
    assert rows == []


def test_iter_shadow_rows_reads_jsonl(tmp_path: Path) -> None:
    sd = tmp_path / "cadence_shadow"
    sd.mkdir()
    (sd / "a.jsonl").write_text(
        '{"a":1}\n{"a":2}\n', encoding="utf-8",
    )
    (sd / "b.jsonl").write_text('{"b":3}\n', encoding="utf-8")
    rows = sorted(iter_shadow_rows(sd), key=lambda r: tuple(r.items()))
    assert {tuple(r.items()) for r in rows} == {
        (("a", 1),), (("a", 2),), (("b", 3),),
    }


def test_iter_shadow_rows_skips_malformed_lines(tmp_path: Path) -> None:
    sd = tmp_path / "cadence_shadow"
    sd.mkdir()
    (sd / "a.jsonl").write_text(
        'not json\n{"a":1}\n[]\n', encoding="utf-8",
    )
    # `not json` and `[]` (list, not dict) are skipped.
    rows = list(iter_shadow_rows(sd))
    assert rows == [{"a": 1}]


# --- format_report --------------------------------------------------------


def test_format_report_text_has_headers() -> None:
    summ = compute_summary([_row(p1=True, p2=False, fired=False)])
    text = format_report(summ, as_json=False)
    assert "total rows:" in text
    assert "per-policy would_fire rate:" in text
    assert "selected-policy live fire rate:" in text
    assert "agreement matrix" in text


def test_format_report_json_roundtrips() -> None:
    summ = compute_summary([_row(p1=True, p2=True, fired=True)])
    out = format_report(summ, as_json=True)
    parsed = json.loads(out)
    assert parsed["total_rows"] == 1
    assert parsed["per_policy_fire_count"][POLICY_P1_EVERY_K_TURNS] == 1
    # Agreement matrix keys are stringified bool pairs.
    assert "p1=1,p2=1" in parsed["agreement_matrix"]


# --- resolve_shadow_dir ---------------------------------------------------


def test_resolve_shadow_dir_layout() -> None:
    p = resolve_shadow_dir(Path("/proj"))
    assert p == Path("/proj/.git/aelfrice/cadence_shadow")


# --- CLI surface ---------------------------------------------------------


def test_cli_cadence_score_emits_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`aelf cadence-score --project PATH` reads the project's shadow log."""
    import io as _io
    from aelfrice import cli

    sd = tmp_path / ".git" / "aelfrice" / CADENCE_SHADOW_DIRNAME
    sd.mkdir(parents=True)
    (sd / "s1.jsonl").write_text(
        json.dumps({
            "ts": "2026-05-20T00:00:00Z",
            "session_id": "s1",
            "selected": POLICY_P2_CTX_THRESHOLD,
            "fired": True,
            "shadow": {
                POLICY_P1_EVERY_K_TURNS: {"would_fire": False, "reason": "p1r"},
                POLICY_P2_CTX_THRESHOLD: {"would_fire": True, "reason": "p2r"},
            },
        }) + "\n",
    )
    out = _io.StringIO()
    rc = cli.main(
        ["cadence-score", "--project", str(tmp_path)], out=out,
    )
    assert rc == 0
    text = out.getvalue()
    assert "total rows:    1" in text
    assert "p2_ctx_threshold" in text
    assert "agreement matrix" in text


def test_cli_cadence_score_json_emits_object(tmp_path: Path) -> None:
    import io as _io
    from aelfrice import cli

    sd = tmp_path / ".git" / "aelfrice" / CADENCE_SHADOW_DIRNAME
    sd.mkdir(parents=True)
    (sd / "s.jsonl").write_text(json.dumps({
        "ts": "2026-05-20T00:00:00Z",
        "session_id": "s",
        "selected": POLICY_OFF,
        "fired": False,
        "shadow": {
            POLICY_P1_EVERY_K_TURNS: {"would_fire": True, "reason": "x"},
            POLICY_P2_CTX_THRESHOLD: {"would_fire": False, "reason": "y"},
        },
    }) + "\n")
    out = _io.StringIO()
    rc = cli.main(
        ["cadence-score", "--project", str(tmp_path), "--json"], out=out,
    )
    assert rc == 0
    parsed = json.loads(out.getvalue())
    assert parsed["total_rows"] == 1
    assert parsed["per_policy_fire_count"][POLICY_P1_EVERY_K_TURNS] == 1


def test_cli_cadence_score_session_filter(tmp_path: Path) -> None:
    import io as _io
    from aelfrice import cli

    sd = tmp_path / ".git" / "aelfrice" / CADENCE_SHADOW_DIRNAME
    sd.mkdir(parents=True)
    rows = []
    for sid, p1 in (("alpha", True), ("beta", False), ("alpha", True)):
        rows.append(json.dumps({
            "ts": "2026-05-20T00:00:00Z",
            "session_id": sid,
            "selected": POLICY_OFF,
            "fired": False,
            "shadow": {
                POLICY_P1_EVERY_K_TURNS: {"would_fire": p1, "reason": "x"},
                POLICY_P2_CTX_THRESHOLD: {"would_fire": False, "reason": "y"},
            },
        }))
    (sd / "log.jsonl").write_text("\n".join(rows) + "\n")

    out = _io.StringIO()
    rc = cli.main([
        "cadence-score", "--project", str(tmp_path),
        "--session", "alpha", "--json",
    ], out=out)
    assert rc == 0
    parsed = json.loads(out.getvalue())
    assert parsed["total_rows"] == 2
    assert parsed["per_policy_fire_count"][POLICY_P1_EVERY_K_TURNS] == 2


def test_cli_cadence_score_missing_project_returns_2(tmp_path: Path) -> None:
    import io as _io
    from aelfrice import cli
    out = _io.StringIO()
    rc = cli.main([
        "cadence-score", "--project", str(tmp_path / "does-not-exist"),
    ], out=out)
    assert rc == 2


def test_cli_cadence_score_empty_dir_emits_zero_report(tmp_path: Path) -> None:
    """Project with no shadow log -> exit 0, all zeros."""
    import io as _io
    from aelfrice import cli
    out = _io.StringIO()
    rc = cli.main([
        "cadence-score", "--project", str(tmp_path), "--json",
    ], out=out)
    assert rc == 0
    parsed = json.loads(out.getvalue())
    assert parsed["total_rows"] == 0
