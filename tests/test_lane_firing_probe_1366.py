"""#1366 lane-firing probe — the probe's own instrument checks.

The probe's output is only worth reading if the instrument is sound, and
"the probe ran and reported nothing" is the exact shape of both a clean
result and a broken instrument. These tests pin the differences:

* both diff directions are computed, so removing the second one (the
  direction nobody has looked for) fails here rather than silently
  halving the finding;
* the corpus reader keeps untruncated user text and drops the record
  types that never reached the retrieval path;
* the lane table is wired to `LaneTelemetry` fields that exist, so a
  renamed field is a red test rather than a lane that reads as dead;
* the flags with no telemetry field are reported as *unobservable*, not
  as never-fired — scoring them as never-fired would fabricate the
  finding the probe exists to detect;
* importing this module has no side effect on the environment.
"""
from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path

import pytest

from aelfrice.hook import AUDIT_PROMPT_PREFIX_CAP
from aelfrice.retrieval import LaneTelemetry
from benchmarks import lane_firing_probe as probe


# --- The two-directional diff -------------------------------------------


def test_diff_reports_enabled_but_never_fired() -> None:
    """A lane the resolver reports on that no call witnessed."""
    out = probe.diff_lanes({"ghost": True, "live": True}, {"ghost": 0, "live": 7})
    assert out["enabled_but_never_fired"] == ["ghost"]


def test_diff_reports_fired_but_not_reported_enabled() -> None:
    """The other direction: a lane firing that nothing reports as on.

    This is the assertion that distinguishes a two-directional probe from
    a one-directional one. Delete the `fired_but_not_reported_enabled`
    branch in `diff_lanes` and this fails while every
    enabled-but-never-fired test still passes — which is precisely the
    half-finding #1366 warns against.
    """
    out = probe.diff_lanes({"stealth": False}, {"stealth": 3})
    assert out["fired_but_not_reported_enabled"] == ["stealth"]
    assert out["enabled_but_never_fired"] == []


def test_diff_directions_are_independent() -> None:
    """Both asymmetries in one call, and neither swallows the other."""
    out = probe.diff_lanes(
        {"ghost": True, "stealth": False, "ok_on": True, "ok_off": False},
        {"ghost": 0, "stealth": 5, "ok_on": 5, "ok_off": 0},
    )
    assert out["enabled_but_never_fired"] == ["ghost"]
    assert out["fired_but_not_reported_enabled"] == ["stealth"]


def test_diff_is_empty_when_everything_agrees() -> None:
    out = probe.diff_lanes({"a": True, "b": False}, {"a": 4, "b": 0})
    assert out == {
        "enabled_but_never_fired": [],
        "fired_but_not_reported_enabled": [],
    }


def test_diff_counts_not_booleans() -> None:
    """One fire in a thousand calls is *not* a dead lane.

    A boolean accumulator would collapse this case into the same answer
    as a lane that fired on every call. The diff must treat a single
    observation as firing.
    """
    out = probe.diff_lanes({"rare": True}, {"rare": 1})
    assert out["enabled_but_never_fired"] == []


# --- Lane table ----------------------------------------------------------


def test_observable_lanes_name_real_telemetry_fields() -> None:
    """Every observable lane points at a field `LaneTelemetry` has.

    A renamed or removed field would otherwise make the lane read as
    permanently dead — a fabricated "enabled but never fires".
    """
    known = {f.name for f in fields(LaneTelemetry)}
    for lane in probe.OBSERVABLE_LANES:
        assert lane.field in known, lane.name


def test_observable_lane_names_are_unique() -> None:
    names = [lane.name for lane in probe.OBSERVABLE_LANES]
    assert len(names) == len(set(names))


def test_observed_predicates_are_false_on_a_zero_telemetry() -> None:
    """A default `LaneTelemetry` is "nothing fired".

    If a predicate reads True on the zero value it reports a lane firing
    on every call including the ones where retrieval returned nothing —
    the fired-but-not-enabled direction, manufactured by the instrument.
    """
    zero = LaneTelemetry()
    for lane in probe.OBSERVABLE_LANES:
        assert lane.observed(zero) is False or lane.observed(zero) == 0, lane.name


def test_observed_predicates_are_true_when_their_field_is_set() -> None:
    """Each predicate reads its own field and reacts to it."""
    cases = {
        "l0_locked": LaneTelemetry(locked=1),
        "l25_entity_index": LaneTelemetry(l25=1),
        "l1_bm25": LaneTelemetry(l1=1),
        "l1_bm25f_anchors": LaneTelemetry(bm25f_used=True),
        "bfs_multihop": LaneTelemetry(bfs=1),
        "hrr_expand": LaneTelemetry(hrr_expand=1),
        "temporal_spine": LaneTelemetry(temporal_spine=1),
        "heat_kernel": LaneTelemetry(heat_used=True),
        "posterior_weight_rerank": LaneTelemetry(posterior_weight=0.5),
        "expansion_gate": LaneTelemetry(expansion_gate_reason="narrow"),
    }
    by_name = {lane.name: lane for lane in probe.OBSERVABLE_LANES}
    assert set(cases) == set(by_name), "lane table and this table disagree"
    for name, tel in cases.items():
        assert by_name[name].observed(tel), name


def test_unobservable_lanes_are_not_scored_as_never_fired() -> None:
    """Flags with no telemetry field never enter the diff.

    Feeding them in would report every one of them as "enabled but never
    fires" on the first run, which is a fabricated finding: the probe
    cannot see them at all. They are reported separately with
    `observable: false`.
    """
    observable = {lane.name for lane in probe.OBSERVABLE_LANES}
    unobservable = {name for name, _, _ in probe.UNOBSERVABLE_LANES}
    assert observable.isdisjoint(unobservable)
    known = {f.name for f in fields(LaneTelemetry)}
    for name in unobservable:
        assert name not in known


# --- Corpus reader -------------------------------------------------------


def _write_transcript(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8",
    )


def _user(text: str, **extra: object) -> dict:
    return {"type": "user", "message": {"role": "user", "content": text},
            **extra}


def test_corpus_keeps_prompts_untruncated(tmp_path: Path) -> None:
    """The corpus must carry full prompt text.

    Prompt length gates the #741 expansion gate, so any truncation in the
    reader would under-fire the expansion-dependent lanes and produce the
    false "enabled but never fires" this probe exists to catch. The
    length asserted here is well past the `hook_audit` cap that makes a
    `hook_audit` corpus unusable.
    """
    long_prompt = "x" * (AUDIT_PROMPT_PREFIX_CAP * 10)
    _write_transcript(tmp_path / "s.jsonl", [_user(long_prompt)])
    got = probe.collect_corpus(tmp_path, 10)
    assert got == [long_prompt]
    assert len(got[0]) > AUDIT_PROMPT_PREFIX_CAP


def test_corpus_drops_tool_results_and_meta(tmp_path: Path) -> None:
    """Only turns a user actually typed count."""
    _write_transcript(tmp_path / "s.jsonl", [
        {"type": "assistant", "message": {"role": "assistant",
                                          "content": "not a user turn"}},
        _user("meta bookkeeping", isMeta=True),
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "tool output"},
        ]}},
        _user("<command-name>slash</command-name>"),
        _user("   "),
        _user("real prompt"),
    ])
    assert probe.collect_corpus(tmp_path, 10) == ["real prompt"]


def test_corpus_reads_text_blocks(tmp_path: Path) -> None:
    """Structured content keeps its text blocks and drops the rest."""
    _write_transcript(tmp_path / "s.jsonl", [
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "ignored"},
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]}},
    ])
    assert probe.collect_corpus(tmp_path, 10) == ["first\nsecond"]


def test_corpus_is_deterministic_and_ordered(tmp_path: Path) -> None:
    """No RNG: sorted file order, then line order, then first N.

    Written out of order on purpose — a reader that walked the directory
    in `os.scandir` order would pass on some filesystems and not others.
    """
    _write_transcript(tmp_path / "b.jsonl", [_user("b1"), _user("b2")])
    _write_transcript(tmp_path / "a.jsonl", [_user("a1"), _user("a2")])
    _write_transcript(tmp_path / "nested" / "c.jsonl", [_user("c1")])
    assert probe.collect_corpus(tmp_path, 5) == ["a1", "a2", "b1", "b2", "c1"]
    assert probe.collect_corpus(tmp_path, 3) == ["a1", "a2", "b1"]
    assert probe.collect_corpus(tmp_path, 3) == probe.collect_corpus(tmp_path, 3)


def test_corpus_survives_a_malformed_line(tmp_path: Path) -> None:
    """One bad line is not a reason to lose the corpus."""
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps(_user("before")) + "\n"
        + "{not json\n"
        + json.dumps(_user("after")) + "\n",
        encoding="utf-8",
    )
    assert probe.collect_corpus(tmp_path, 10) == ["before", "after"]


# --- Gate-reason normalisation ------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [
    ("broad:long(2324>80)", "broad:long"),
    ("broad:long(122>80),no-markers", "broad:long,no-markers"),
    ("narrow", "narrow"),
    ("", ""),
])
def test_gate_reason_normalisation_drops_operands(
    raw: str, expected: str,
) -> None:
    """The measured operand is a prompt length; the tag is the branch.

    Keeping the operands makes one histogram bucket per distinct prompt
    length and publishes a length distribution into a report meant to be
    pasted into an issue.
    """
    assert probe.normalise_gate_reason(raw) == expected


# --- Report shape --------------------------------------------------------


def _observed(fired: dict[str, int]) -> dict[str, object]:
    return {
        "fired_calls": {lane.name: fired.get(lane.name, 0)
                        for lane in probe.OBSERVABLE_LANES},
        "expansion_gate_reason_histogram": {"narrow": 1},
        "expansion_gate_skipped_bfs_calls": 0,
        "temporal_spine_candidate_calls": 1,
        "temporal_spine_packed_calls": 0,
        "l1_trimmed_calls": 0,
        "empty_output_calls": 0,
    }


def test_report_carries_both_diff_directions() -> None:
    report = probe.build_report(["a prompt"], _observed({}))
    assert set(report["diff"]) == {
        "enabled_but_never_fired", "fired_but_not_reported_enabled",
    }


def test_report_emits_counts_per_lane_not_booleans() -> None:
    report = probe.build_report(
        ["p1", "p2", "p3", "p4"], _observed({"l1_bm25": 1}),
    )
    row = next(r for r in report["observable_lanes"] if r["lane"] == "l1_bm25")
    assert row["fired_calls"] == 1
    assert row["fire_rate"] == 0.25


def test_report_contains_no_prompt_text() -> None:
    """Aggregate counts only — the report is meant to be pasteable.

    The prompt is a distinctive literal; if any reporting path ever
    echoes corpus text (a sample, a longest-prompt field, an un-normalised
    gate reason) it lands here.
    """
    secret = "zzq-distinctive-corpus-literal-zzq"
    report = probe.build_report([secret], _observed({}))
    assert secret not in json.dumps(report)


def test_report_records_the_audit_cap_it_refuses() -> None:
    """The corpus block cites the live constant, not a copy of it."""
    report = probe.build_report(["p"], _observed({}))
    assert report["corpus"]["audit_prompt_prefix_cap"] == AUDIT_PROMPT_PREFIX_CAP


def test_report_marks_unobservable_lanes_as_unobservable() -> None:
    """They carry `observable: false` and stay out of the diff."""
    report = probe.build_report(["p"], _observed({}))
    rows = report["unobservable_lanes"]
    assert rows, "the unobservable list must not be silently empty"
    diffed = set(report["diff"]["enabled_but_never_fired"])
    for row in rows:
        assert row["observable"] is False
        assert row["lane"] not in diffed


# --- Truncation control --------------------------------------------------


def test_truncation_control_reports_moved_lanes_only() -> None:
    """The control names the lanes a truncated corpus would misjudge."""
    full = _observed({"l25_entity_index": 379, "l1_bm25": 500})
    cut = _observed({"l25_entity_index": 483, "l1_bm25": 500})
    out = probe.truncation_control(full, cut, 500)
    assert set(out["lanes_whose_fire_count_moved"]) == {"l25_entity_index"}
    assert out["lanes_whose_fire_count_moved"]["l25_entity_index"] == {
        "full": 379, "truncated": 483, "delta": 104,
    }
    assert out["prompt_chars"] == AUDIT_PROMPT_PREFIX_CAP


def test_truncation_control_flags_a_falsely_dead_lane() -> None:
    """A lane that fires on full prompts and not on truncated ones.

    This is the manufactured "enabled but never fires" the corpus hazard
    produces. The control has to name it, not merely record a delta.
    """
    full = _observed({"temporal_spine": 209})
    cut = _observed({"temporal_spine": 0})
    out = probe.truncation_control(full, cut, 500)
    assert out["lanes_falsely_dead_under_truncation"] == ["temporal_spine"]


def test_truncation_control_is_quiet_when_nothing_moves() -> None:
    same = _observed({"l1_bm25": 500})
    out = probe.truncation_control(same, _observed({"l1_bm25": 500}), 500)
    assert out["lanes_whose_fire_count_moved"] == {}
    assert out["lanes_falsely_dead_under_truncation"] == []


# --- Import-time side effects -------------------------------------------


def test_importing_the_probe_leaves_ambient_config_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing this module must not touch the caller's environment.

    The clear used to run at module scope. `tests/conftest.py` reads
    ``AELFRICE_CORPUS_ROOT`` at test time to decide whether the
    corpus-gated tests run, and this module is imported at pytest
    *collection* — so a full ``pytest tests/`` silently skipped every
    bench gate, the same class of defect as #1278.

    Re-executing module scope is the direct form: move the clear back up
    there and the sentinel is gone when this returns.
    """
    import importlib

    monkeypatch.setenv("AELFRICE_SENTINEL_1366", "kept")
    monkeypatch.setenv("AELF_SENTINEL_1366", "kept")
    importlib.reload(probe)
    assert os.environ.get("AELFRICE_SENTINEL_1366") == "kept"
    assert os.environ.get("AELF_SENTINEL_1366") == "kept"


def test_pinned_environment_clears_then_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AELFRICE_SENTINEL_1366", "kept")
    monkeypatch.setenv("AELF_SENTINEL_1366", "kept")
    monkeypatch.setenv("UNRELATED_SENTINEL_1366", "kept")
    with probe.pinned_environment() as cleared:
        assert "AELFRICE_SENTINEL_1366" in cleared
        assert "AELF_SENTINEL_1366" in cleared
        assert os.environ.get("AELFRICE_SENTINEL_1366") is None
        assert os.environ.get("AELF_SENTINEL_1366") is None
        # Only the aelfrice-prefixed vars, never the whole environment.
        assert os.environ.get("UNRELATED_SENTINEL_1366") == "kept"
    assert os.environ.get("AELFRICE_SENTINEL_1366") == "kept"
    assert os.environ.get("AELF_SENTINEL_1366") == "kept"


def test_pinned_environment_restores_on_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed run must not leave the caller stripped of its config."""
    monkeypatch.setenv("AELFRICE_SENTINEL_1366", "kept")
    with pytest.raises(RuntimeError), probe.pinned_environment():
        raise RuntimeError("boom")
    assert os.environ.get("AELFRICE_SENTINEL_1366") == "kept"


# --- Committed result ----------------------------------------------------


def test_committed_report_is_well_formed_and_content_free() -> None:
    """The checked-in run carries the shape the report promises.

    Guards the artifact as well as the code: a report regenerated by a
    future edit that starts emitting belief ids or prompt text fails
    here.
    """
    path = (Path(__file__).resolve().parents[1]
            / "benchmarks" / "results" / "lane_firing_probe_1366.json")
    report = json.loads(path.read_text())
    assert report["issue"] == 1366
    assert report["corpus"]["prompts"] >= probe.MIN_PROMPTS
    assert set(report["diff"]) == {
        "enabled_but_never_fired", "fired_but_not_reported_enabled",
    }
    assert {r["lane"] for r in report["observable_lanes"]} == {
        lane.name for lane in probe.OBSERVABLE_LANES
    }
    # Aggregates only: every leaf is a number, a bool, or one of the
    # fixed vocabularies the report defines. No free text from the corpus.
    for row in report["observable_lanes"]:
        assert isinstance(row["fired_calls"], int)
        assert 0.0 <= row["fire_rate"] <= 1.0
    for reason in report["expansion_gate"]["reason_histogram"]:
        assert "(" not in reason, "gate operands leaked into the histogram"
