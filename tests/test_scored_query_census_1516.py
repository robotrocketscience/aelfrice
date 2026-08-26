"""The #1516 scored-query census counts what the bound is actually on.

The bound is "**>= 500 distinct queries spanning >= 30 days**", and each half
of that has a cheaper number sitting next to it that would pass the gate early:

* rows carrying a scored query is not distinct queries -- one query re-asked
  ten times is one query, and counting rows inflates the population by the
  operator's repetition rate;
* days with at least one scored query is not the calendar span -- eight active
  days scattered over a fortnight is not a 30-day span, and a 30-day span with
  two active days in it is not 30 days of accrual either.

So the tests below feed synthetic JSONL rows in which those numbers are
deliberately different, and pin each one separately. A census that confused
either pair would pass a test that only checked totals.

The bounds themselves are pinned as constants with no CLI flag, because the
whole point of #1516 is that they were committed before any measurement.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import scored_query_census as sqc


def _row(ts: str | None, query: str | None, *, omit: bool = False) -> dict:
    """One rebuild_log row, shaped like `_build_rebuild_log_record` writes it.

    `omit=True` drops the `scored_query` key entirely, which is the shape of
    every row written before #1405; the key-present-but-null shape is what a
    caller that passes nothing produces today.
    """
    inner: dict[str, object] = {"n_recent_turns": 1, "extracted_query": "x"}
    if not omit:
        inner["scored_query"] = query
    return {"ts": ts, "session_id": "s", "input": inner}


def _write(dirpath: Path, name: str, rows: list[dict]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / name).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


@pytest.mark.timeout(30)
def test_distinct_is_not_the_row_count() -> None:
    """Four scored rows, two queries. The bound is on the two."""
    rows = [
        _row("2026-08-13T01:00:00Z", "alpha"),
        _row("2026-08-13T02:00:00Z", "alpha"),
        _row("2026-08-14T03:00:00Z", "alpha"),
        _row("2026-08-14T04:00:00Z", "beta"),
    ]
    counts = sqc.census(rows)
    assert counts["rows_scanned"] == 4
    assert counts["rows_with_scored_query"] == 4
    assert counts["distinct_scored_queries"] == 2
    # And the per-day split keeps the same distinction, so a day histogram
    # cannot be summed into a fake distinct total either.
    per_day = {e["day"]: e for e in counts["per_day"]}
    assert per_day["2026-08-13"]["rows"] == 2
    assert per_day["2026-08-13"]["distinct"] == 1
    assert per_day["2026-08-14"]["rows"] == 2
    assert per_day["2026-08-14"]["distinct"] == 2


@pytest.mark.timeout(30)
def test_absent_null_and_blank_scored_query_are_all_excluded() -> None:
    """Only a non-blank string counts toward the population.

    Absent means "unknown" (#1405 is forward-only) and null means the caller
    passed nothing. Neither is a query, and a census that counted key presence
    would report the whole post-#1405 corpus as scored.
    """
    rows = [
        _row("2026-08-13T01:00:00Z", None, omit=True),
        _row("2026-08-13T02:00:00Z", None),
        _row("2026-08-13T03:00:00Z", "   "),
        _row("2026-08-13T04:00:00Z", "real query"),
    ]
    counts = sqc.census(rows)
    assert counts["rows_scanned"] == 4
    assert counts["rows_with_scored_query"] == 1
    assert counts["distinct_scored_queries"] == 1


@pytest.mark.timeout(30)
def test_active_days_and_calendar_span_are_different_numbers() -> None:
    """Three active days inside a fifteen-day window.

    Pinning both is the point: the committed bound is on the span, and
    reporting active days as though it were the span understates the window
    an accrual rate is computed over.
    """
    rows = [
        _row("2026-08-01T00:00:00Z", "q1"),
        _row("2026-08-05T00:00:00Z", "q2"),
        _row("2026-08-15T00:00:00Z", "q3"),
    ]
    counts = sqc.census(rows)
    assert counts["active_days"] == 3
    assert counts["span_days"] == 15
    assert counts["first_day"] == "2026-08-01"
    assert counts["last_day"] == "2026-08-15"
    # One day of data is a one-day span, not a zero-day one -- otherwise the
    # accrual rate divides by zero on a fresh corpus.
    single = sqc.census([_row("2026-08-01T00:00:00Z", "q1")])
    assert single["active_days"] == 1
    assert single["span_days"] == 1


@pytest.mark.timeout(30)
def test_unusable_timestamps_leave_the_day_histogram_alone() -> None:
    """A torn or malformed `ts` is counted and set aside, never bucketed."""
    rows = [
        _row("2026-08-13T01:00:00Z", "good"),
        _row("not-a-timestamp", "bad-ts"),
        _row(None, "no-ts"),
        _row("2026-13-45T00:00:00Z", "impossible-date"),
    ]
    counts = sqc.census(rows)
    assert counts["rows_with_scored_query"] == 4
    assert counts["distinct_scored_queries"] == 4
    assert counts["rows_without_usable_ts"] == 3
    assert counts["active_days"] == 1
    assert [e["day"] for e in counts["per_day"]] == ["2026-08-13"]


@pytest.mark.timeout(30)
def test_empty_input_reports_zeroes_rather_than_dividing_by_zero() -> None:
    counts = sqc.census([])
    assert counts["rows_scanned"] == 0
    assert counts["distinct_scored_queries"] == 0
    assert counts["active_days"] == 0
    assert counts["span_days"] == 0
    assert counts["first_day"] is None and counts["last_day"] is None
    assert counts["per_day"] == []

    v = sqc.verdict(counts)
    assert v["runnable"] is False
    assert v["distinct_per_span_day"] == 0.0
    assert v["days_to_bound_linear"] is None
    assert v["queries_short_by"] == sqc.THRESHOLD_DISTINCT_QUERIES


@pytest.mark.timeout(30)
def test_verdict_needs_both_bounds() -> None:
    """Either bound alone is not enough, and both together are."""
    # 500 distinct queries, all on one day: query bound met, span bound not.
    one_day = [_row("2026-08-01T00:00:00Z", f"q{i}") for i in range(500)]
    v = sqc.verdict(sqc.census(one_day))
    assert v["queries_bound_met"] is True
    assert v["span_bound_met"] is False
    assert v["runnable"] is False

    # A 30-day span with far too few queries: the mirror case.
    sparse = [
        _row("2026-08-01T00:00:00Z", "a"),
        _row("2026-08-30T00:00:00Z", "b"),
    ]
    v = sqc.verdict(sqc.census(sparse))
    assert v["queries_bound_met"] is False
    assert v["span_bound_met"] is True
    assert v["runnable"] is False

    both = one_day + [_row("2026-08-30T00:00:00Z", "q0")]
    v = sqc.verdict(sqc.census(both))
    assert v["runnable"] is True


@pytest.mark.timeout(30)
def test_lumpiness_is_reported_so_the_rate_carries_its_caveat() -> None:
    """The busiest-day share is what makes a linear extrapolation weak."""
    rows = [_row("2026-08-02T00:00:00Z", f"q{i}") for i in range(9)]
    rows.append(_row("2026-08-01T00:00:00Z", "lone"))
    v = sqc.verdict(sqc.census(rows))
    assert v["busiest_day_distinct"] == 9
    assert v["busiest_day_share"] == pytest.approx(0.9)
    assert v["distinct_per_span_day"] == pytest.approx(5.0)


@pytest.mark.timeout(30)
def test_exit_codes_separate_failure_from_not_yet(tmp_path: Path) -> None:
    """2 is "could not census", 3 is "censused, bound unmet", 0 is runnable."""
    missing = tmp_path / "nope"
    assert sqc.main(["--logs", str(missing)]) == 2

    empty = tmp_path / "empty"
    empty.mkdir()
    assert sqc.main(["--logs", str(empty)]) == 2

    short = tmp_path / "short"
    _write(short, "s.jsonl", [_row("2026-08-13T00:00:00Z", "only one")])
    assert sqc.main(["--logs", str(short)]) == 3

    big = tmp_path / "big"
    _write(big, "s.jsonl", [
        _row(f"2026-08-{1 + (i % 30):02d}T00:00:00Z", f"q{i}")
        for i in range(600)
    ])
    assert sqc.main(["--logs", str(big)]) == 0


@pytest.mark.timeout(30)
def test_dry_run_lists_the_files_and_reads_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dry run must survive a corpus it could not parse, having not tried."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "torn.jsonl").write_text("{not json at all\n", encoding="utf-8")
    assert sqc.main(["--logs", str(logs), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "torn.jsonl" in out
    assert "nothing read" in out


@pytest.mark.timeout(30)
def test_the_committed_bounds_are_not_cli_flags() -> None:
    """#1516's bounds were fixed in advance; the CLI must not move them.

    A flag would let a re-run lower the bar it is checking and still look like
    the same census, which is exactly the failure the pre-commitment exists to
    prevent.
    """
    assert sqc.THRESHOLD_DISTINCT_QUERIES == 500
    assert sqc.THRESHOLD_SPAN_DAYS == 30
    for flag in ("--min-queries", "--min-days", "--threshold", "--span-days"):
        with pytest.raises(SystemExit):
            sqc.main([flag, "1"])


@pytest.mark.timeout(30)
def test_json_output_carries_the_corpus_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A population figure without its corpus identity is unquotable."""
    logs = tmp_path / "logs"
    _write(logs, "s.jsonl", [_row("2026-08-13T00:00:00Z", "q")])
    assert sqc.main(["--logs", str(logs), "--json"]) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["corpus"]["n_files"] == 1
    assert payload["corpus"]["n_log_dirs"] == 1
    assert payload["corpus"]["digest"]
    assert payload["thresholds"] == {"distinct_queries": 500, "span_days": 30}
    assert payload["counts"]["distinct_scored_queries"] == 1
    assert payload["verdict"]["runnable"] is False


@pytest.mark.timeout(30)
def test_discovery_finds_both_store_layouts(tmp_path: Path) -> None:
    """Repo-local `<root>/.git/aelfrice/rebuild_logs` and the home fallback.

    The live store is repo-local, so a machine-wide census that only knew
    about `~/.aelfrice` would report a near-empty population.
    """
    repo = tmp_path / "projects" / "repo"
    repo_logs = repo / ".git" / "aelfrice" / "rebuild_logs"
    _write(repo_logs, "a.jsonl", [_row("2026-08-13T00:00:00Z", "repo query")])
    home_logs = tmp_path / ".aelfrice" / "rebuild_logs"
    _write(home_logs, "b.jsonl", [_row("2026-08-14T00:00:00Z", "home query")])

    found = sqc.discover_log_dirs(tmp_path)
    assert repo_logs.resolve() in found
    assert home_logs.resolve() in found
    assert len(sqc.jsonl_files(found)) == 2


@pytest.mark.timeout(30)
def test_a_malformed_row_falls_out_instead_of_aborting_the_census() -> None:
    """`input` is written as an object, but the census must not assume it.

    The module's contract is that a bad row drops out of the count, the way
    `_day` drops a torn timestamp -- a census that dies on one malformed line
    reports nothing at all about the other four thousand.
    """
    rows: list[dict] = [
        {"ts": "2026-08-13T00:00:00Z", "input": "oops"},
        {"ts": "2026-08-13T00:00:00Z", "input": []},
        {"ts": "2026-08-13T00:00:00Z", "input": 7},
        {"ts": "2026-08-13T00:00:00Z"},
        _row("2026-08-13T00:00:00Z", "real"),
    ]
    counts = sqc.census(rows)
    assert counts["rows_scanned"] == 5
    assert counts["rows_with_scored_query"] == 1
    assert counts["distinct_scored_queries"] == 1


@pytest.mark.timeout(30)
def test_surrounding_whitespace_does_not_split_one_query_into_three() -> None:
    """Blankness is judged on the stripped form, so identity must be too.

    Otherwise `"q"`, `" q"` and `"q "` are three distinct queries and the
    number checked against #1516's 500-query bound is inflated by however
    often the writer's whitespace differs.
    """
    rows = [
        _row("2026-08-13T00:00:00Z", "same query"),
        _row("2026-08-13T01:00:00Z", " same query"),
        _row("2026-08-13T02:00:00Z", "same query\n"),
    ]
    counts = sqc.census(rows)
    assert counts["rows_with_scored_query"] == 3
    assert counts["distinct_scored_queries"] == 1
    assert counts["per_day"][0]["distinct"] == 1


@pytest.mark.timeout(30)
def test_one_directory_named_twice_is_read_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both flags are repeatable, so one directory can arrive twice.

    `jsonl_files` dedupes by resolved path, so the corpus identity would keep
    saying "one file" while `load_rows` read the directory once per spelling
    and doubled every row count -- and the row counts are what a coverage
    claim is a ratio of.
    """
    logs = tmp_path / "logs"
    _write(logs, "s.jsonl", [
        _row("2026-08-13T00:00:00Z", "a"),
        _row("2026-08-13T01:00:00Z", "b"),
        _row("2026-08-14T00:00:00Z", "c"),
    ])
    # The same physical directory, spelled two ways.
    detour = logs / ".." / logs.name
    assert detour.is_dir() and str(detour) != str(logs)

    assert sqc.main(["--logs", str(logs), "--logs", str(detour), "--json"]) == 3
    payload = json.loads(capsys.readouterr().out)

    assert payload["corpus"]["n_log_dirs"] == 1
    assert payload["corpus"]["n_files"] == 1
    assert payload["counts"]["rows_scanned"] == 3
    assert payload["counts"]["rows_with_scored_query"] == 3
    assert payload["counts"]["distinct_scored_queries"] == 3
