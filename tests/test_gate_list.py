"""Unit tests for `aelf gate list` aggregator (#475).

The runner is injected so tests never shell out to gh. Each test passes
a fake runner that returns canned JSON for `issue list` / `issue view`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from aelfrice import gate_list as gl


def _make_runner(responses: dict[str, str]):
    """Map keyed-by-args-tuple gh-call → stdout. Unrecognised calls error."""
    def runner(args):
        key = tuple(args)
        if key not in responses:
            raise AssertionError(f"unexpected gh call: {key}")
        return responses[key]
    return runner


def _list_call(label: str) -> tuple:
    return (
        "issue", "list",
        "--label", label,
        "--state", "open",
        "--json", "number,title,createdAt",
        "--limit", "200",
    )


def _view_call(num: int) -> tuple:
    return ("issue", "view", str(num), "--json", "comments")


# ---------------------------------------------------------------------------
# parse_asks_count
# ---------------------------------------------------------------------------


def test_parse_asks_count_typical():
    body = (
        "[gate:ratify]\n\nThree asks open.\n\n"
        "**Ask 1: foo?**\n  A. one.\n\n"
        "**Ask 2: bar?**\n  A. two.\n\n"
        "**Ask 3: baz?**\n  A. three.\n"
    )
    assert gl.parse_asks_count(body) == 3


def test_parse_asks_count_zero_when_empty():
    assert gl.parse_asks_count("") == 0
    assert gl.parse_asks_count("[gate:ratify]\nno asks here") == 0


def test_parse_asks_count_ignores_inline_mentions():
    body = "Some prose mentioning Ask 1 inline\n**Ask 1: real header?**\n"
    assert gl.parse_asks_count(body) == 1


def test_parse_asks_count_handles_double_digit():
    body = "\n".join(f"**Ask {i}: q?**" for i in range(1, 12))
    assert gl.parse_asks_count(body) == 11


# ---------------------------------------------------------------------------
# collect — section assembly
# ---------------------------------------------------------------------------


def test_collect_empty_sections():
    runner = _make_runner({
        _list_call("gate:ratify"): "[]",
        _list_call("gate:prereq"): "[]",
        _list_call("bench-gated"): "[]",
        _list_call("gate:license"): "[]",
    })
    report = gl.collect(runner=runner)
    for header in ("gate:ratify", "gate:prereq", "gate:bench", "gate:license"):
        assert report.sections[header] == []


def test_collect_ratify_counts_asks_and_records_age():
    issues = [
        {"number": 473, "title": "StructMemEval temporal_sort",
         "createdAt": "2026-05-08T07:00:00Z"},
    ]
    comments = [
        {"body": "informational prose; not a gate block",
         "createdAt": "2026-05-08T07:01:00Z"},
        {"body": "[gate:ratify]\n\n**Ask 1: a?**\n**Ask 2: b?**\n**Ask 3: c?**",
         "createdAt": "2026-05-08T07:28:34Z"},
    ]
    runner = _make_runner({
        _list_call("gate:ratify"): json.dumps(issues),
        _view_call(473): json.dumps({"comments": comments}),
        _list_call("gate:prereq"): "[]",
        _list_call("bench-gated"): "[]",
        _list_call("gate:license"): "[]",
    })
    report = gl.collect(runner=runner)
    rats = report.sections["gate:ratify"]
    assert len(rats) == 1
    assert rats[0].number == 473
    assert rats[0].asks_count == 3
    assert rats[0].most_recent_gate_comment_at == datetime(
        2026, 5, 8, 7, 28, 34, tzinfo=timezone.utc
    )


def test_collect_ratify_picks_most_recent_gate_comment():
    issues = [
        {"number": 100, "title": "x", "createdAt": "2026-01-01T00:00:00Z"}
    ]
    comments = [
        {"body": "[gate:ratify]\n**Ask 1: stale?**",
         "createdAt": "2026-01-01T00:00:00Z"},
        {"body": "[gate:ratify]\n**Ask 1: fresh?**\n**Ask 2: also?**",
         "createdAt": "2026-05-08T07:30:00Z"},
        {"body": "[gate:ratify]\n**Ask 1: middle?**",
         "createdAt": "2026-03-01T00:00:00Z"},
    ]
    runner = _make_runner({
        _list_call("gate:ratify"): json.dumps(issues),
        _view_call(100): json.dumps({"comments": comments}),
        _list_call("gate:prereq"): "[]",
        _list_call("bench-gated"): "[]",
        _list_call("gate:license"): "[]",
    })
    report = gl.collect(runner=runner)
    item = report.sections["gate:ratify"][0]
    assert item.asks_count == 2  # asks come from the freshest comment
    assert item.most_recent_gate_comment_at == datetime(
        2026, 5, 8, 7, 30, 0, tzinfo=timezone.utc
    )


def test_collect_ratify_handles_no_gate_comment():
    issues = [
        {"number": 50, "title": "y", "createdAt": "2026-01-01T00:00:00Z"}
    ]
    runner = _make_runner({
        _list_call("gate:ratify"): json.dumps(issues),
        _view_call(50): json.dumps({"comments": []}),
        _list_call("gate:prereq"): "[]",
        _list_call("bench-gated"): "[]",
        _list_call("gate:license"): "[]",
    })
    report = gl.collect(runner=runner)
    item = report.sections["gate:ratify"][0]
    assert item.asks_count == 0
    assert item.most_recent_gate_comment_at is None


def test_collect_ratify_sorted_oldest_first_no_comment_last():
    issues = [
        {"number": 10, "title": "no-comment", "createdAt": "2026-01-01T00:00:00Z"},
        {"number": 20, "title": "fresh", "createdAt": "2026-01-01T00:00:00Z"},
        {"number": 30, "title": "stale", "createdAt": "2026-01-01T00:00:00Z"},
    ]
    runner = _make_runner({
        _list_call("gate:ratify"): json.dumps(issues),
        _view_call(10): json.dumps({"comments": []}),
        _view_call(20): json.dumps({"comments": [
            {"body": "[gate:ratify]\n**Ask 1: q?**",
             "createdAt": "2026-05-08T07:30:00Z"},
        ]}),
        _view_call(30): json.dumps({"comments": [
            {"body": "[gate:ratify]\n**Ask 1: q?**",
             "createdAt": "2026-03-01T00:00:00Z"},
        ]}),
        _list_call("gate:prereq"): "[]",
        _list_call("bench-gated"): "[]",
        _list_call("gate:license"): "[]",
    })
    report = gl.collect(runner=runner)
    nums = [i.number for i in report.sections["gate:ratify"]]
    # stale (Mar) before fresh (May), no-comment last
    assert nums == [30, 20, 10]


def test_collect_simple_sections_sorted_by_number():
    prereqs = [
        {"number": 365, "title": "close-the-loop", "createdAt": "2026-04-01T00:00:00Z"},
        {"number": 291, "title": "rebuild redesign", "createdAt": "2026-03-01T00:00:00Z"},
    ]
    bench = [
        {"number": 197, "title": "dedup eval", "createdAt": "2026-02-01T00:00:00Z"},
    ]
    runner = _make_runner({
        _list_call("gate:ratify"): "[]",
        _list_call("gate:prereq"): json.dumps(prereqs),
        _list_call("bench-gated"): json.dumps(bench),
        _list_call("gate:license"): "[]",
    })
    report = gl.collect(runner=runner)
    assert [i.number for i in report.sections["gate:prereq"]] == [291, 365]
    assert [i.number for i in report.sections["gate:bench"]] == [197]


# ---------------------------------------------------------------------------
# format_text / format_json
# ---------------------------------------------------------------------------


def _fixed_now() -> datetime:
    return datetime(2026, 5, 8, 11, 28, 34, tzinfo=timezone.utc)


def test_format_text_all_sections_present_when_empty():
    report = gl.GateReport(sections={
        "gate:ratify": [],
        "gate:prereq": [],
        "gate:bench": [],
        "gate:license": [],
    })
    text = gl.format_text(report, now=_fixed_now())
    assert "gate:ratify (0 open, 0 total asks):" in text
    assert "gate:prereq (0 open):" in text
    assert "gate:bench (0 open):" in text
    assert "gate:license (0 open):" in text
    # Each empty section prints "(none)" line.
    assert text.count("(none)") == 4


def test_format_text_ratify_line_shape():
    item = gl.GateItem(
        number=473,
        title="StructMemEval temporal_sort",
        asks_count=3,
        most_recent_gate_comment_at=datetime(
            2026, 5, 8, 7, 28, 34, tzinfo=timezone.utc
        ),
    )
    report = gl.GateReport(sections={
        "gate:ratify": [item],
        "gate:prereq": [],
        "gate:bench": [],
        "gate:license": [],
    })
    text = gl.format_text(report, now=_fixed_now())
    assert "gate:ratify (1 open, 3 total asks):" in text
    assert "#473" in text
    assert "3 asks" in text
    assert "oldest 4h" in text  # 11:28 - 07:28 = 4h
    assert "StructMemEval temporal_sort" in text


def test_format_text_singular_ask_label():
    item = gl.GateItem(
        number=999,
        title="single ask",
        asks_count=1,
        most_recent_gate_comment_at=datetime(
            2026, 5, 8, 11, 28, 34, tzinfo=timezone.utc
        ),
    )
    report = gl.GateReport(sections={
        "gate:ratify": [item],
        "gate:prereq": [],
        "gate:bench": [],
        "gate:license": [],
    })
    text = gl.format_text(report, now=_fixed_now())
    assert "1 ask " in text  # singular, no trailing 's'


def test_format_text_simple_section_line_shape():
    report = gl.GateReport(sections={
        "gate:ratify": [],
        "gate:prereq": [
            gl.GateItem(number=291, title="rebuild redesign"),
            gl.GateItem(number=365, title="close-the-loop"),
        ],
        "gate:bench": [],
        "gate:license": [],
    })
    text = gl.format_text(report, now=_fixed_now())
    assert "gate:prereq (2 open):" in text
    assert "  #291  rebuild redesign" in text
    assert "  #365  close-the-loop" in text


@pytest.mark.parametrize("seconds, expected", [
    (5, "5s"),
    (90, "1m"),
    (3600 * 2, "2h"),
    (86400 * 3, "3d"),
])
def test_humanise_age_buckets(seconds, expected):
    now = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
    ts = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc).fromtimestamp(
        now.timestamp() - seconds, tz=timezone.utc
    )
    assert gl._humanise_age(now, ts) == expected


def test_humanise_age_no_comment_marker():
    now = datetime(2026, 5, 8, 12, 0, 0, tzinfo=timezone.utc)
    assert gl._humanise_age(now, None) == "no asks-block"


def test_format_json_round_trips():
    report = gl.GateReport(sections={
        "gate:ratify": [
            gl.GateItem(
                number=473,
                title="t",
                asks_count=3,
                most_recent_gate_comment_at=datetime(
                    2026, 5, 8, 7, 28, 34, tzinfo=timezone.utc
                ),
            )
        ],
        "gate:prereq": [gl.GateItem(number=291, title="p")],
        "gate:bench": [],
        "gate:license": [],
    })
    payload = json.loads(gl.format_json(report))
    assert payload["sections"]["gate:ratify"][0]["number"] == 473
    assert payload["sections"]["gate:ratify"][0]["asks_count"] == 3
    assert payload["sections"]["gate:ratify"][0]["most_recent_gate_comment_at"] == (
        "2026-05-08T07:28:34+00:00"
    )
    assert payload["sections"]["gate:prereq"][0]["number"] == 291
    assert payload["sections"]["gate:prereq"][0]["most_recent_gate_comment_at"] is None
    assert payload["sections"]["gate:bench"] == []
    assert payload["sections"]["gate:license"] == []


# ---------------------------------------------------------------------------
# Error surfaces
# ---------------------------------------------------------------------------


def test_default_runner_raises_when_gh_missing(monkeypatch):
    def fake_run(*a, **kw):
        raise FileNotFoundError("no gh")
    monkeypatch.setattr("aelfrice.gate_list.subprocess.run", fake_run)
    with pytest.raises(gl.GhError, match="gh CLI not found"):
        gl._default_runner(["issue", "list"])


def test_default_runner_raises_on_nonzero(monkeypatch):
    class _R:
        returncode = 1
        stdout = ""
        stderr = "boom"
    monkeypatch.setattr(
        "aelfrice.gate_list.subprocess.run", lambda *a, **kw: _R()
    )
    with pytest.raises(gl.GhError, match="boom"):
        gl._default_runner(["issue", "list"])
