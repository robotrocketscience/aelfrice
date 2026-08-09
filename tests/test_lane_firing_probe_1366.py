"""#1366 lane-firing probe — the probe's own instrument checks.

The probe's output is only worth reading if the instrument is sound, and
"the probe ran and reported nothing" is the exact shape of both a clean
result and a broken instrument. These tests pin the differences:

* both diff directions are computed, so removing the second one (the
  direction nobody has looked for) fails here rather than silently
  halving the finding — *and* the report discloses that only one of the
  two directions has any sensitivity, so an empty set in the other is
  never presented as a null result;
* the corpus reader keeps untruncated user text and drops the record
  types that never reached the retrieval path;
* the lane table is wired to `LaneTelemetry` fields that exist, so a
  renamed field is a red test rather than a lane that reads as dead;
* the flags with no telemetry field are reported as *unobservable*, not
  as never-fired — scoring them as never-fired would fabricate the
  finding the probe exists to detect;
* the lanes whose telemetry field is written from the resolved flag are
  marked as such, because a 100% fire rate read off a resolver is the
  false positive this probe exists to catch;
* importing this module has no side effect on the environment, and the
  method constraints `main()` promises (read-only store, cleared env,
  pinned TOML walk, production `retrieve()` kwargs, corpus floor) each
  have an assertion that goes red when they are removed.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from aelfrice.hook import AUDIT_PROMPT_PREFIX_CAP
from aelfrice import retrieval

# Bound from the module rather than imported separately: the tests below
# monkeypatch `retrieval.retrieve`, which needs the module object, and
# importing the same module both ways is what CodeQL 564/565 flagged.
LaneTelemetry = retrieval.LaneTelemetry
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
        "type_aware_compression": LaneTelemetry(compression_renders=1),
        "intentional_clustering": LaneTelemetry(cluster_packed=1),
        "max_coverage_pack": LaneTelemetry(max_coverage_packed=1),
        "entity_persist_demote": LaneTelemetry(entity_persist_demoted=1),
        "supersession_demote": LaneTelemetry(supersession_demoted=1),
        "origin_tiebreak": LaneTelemetry(origin_tiebreak_decided=1),
        "gamma_posterior_temperature": LaneTelemetry(gamma_rerank_scored=1),
        "zeta_posterior_rerank": LaneTelemetry(zeta_rerank_scored=1),
        "fan_effect": LaneTelemetry(fan_effect_ranked=1),
        "hrr_structural": LaneTelemetry(hrr_structural_hit=True),
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
    unobservable = {name for name, _, _, _ in probe.UNOBSERVABLE_LANES}
    assert observable.isdisjoint(unobservable)
    known = {f.name for f in fields(LaneTelemetry)}
    for name in unobservable:
        assert name not in known


def test_every_unobservable_lane_says_why_it_is_out_of_reach() -> None:
    """"No telemetry field" invites the reader to assume nobody looked.

    Each remaining hole names where its leaf actually is, so the next
    reader can tell "not instrumented yet" from "cannot be instrumented
    from this module".
    """
    for name, _, _, why in probe.UNOBSERVABLE_LANES:
        assert why.strip(), name
        assert len(why) > 40, name


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


# --- Flag-tracking disclosure -------------------------------------------

_FLAG_TRACKING_LANES = {
    "l1_bm25f_anchors", "posterior_weight_rerank", "fan_effect",
    # Not a resolver echo like the three above, but uninformative for the
    # same reason: all six `should_run_expansion` return paths set a
    # non-empty reason, so `bool(expansion_gate_reason)` is constant-True
    # inside `retrieve_with_tiers`. A 1.0000 rate that cannot be anything
    # else is not evidence, and the table must not read as if it were.
    "expansion_gate",
}


def test_the_flag_tracking_set_is_exactly_the_audited_one() -> None:
    """Drift guard on the audit of which fields restate their resolver.

    A new lane whose field is written from the flag has to be added
    here deliberately; a lane that stops being flag-tracking has to be
    removed. Either way the audit is re-done rather than inherited.
    """
    got = {lane.name for lane in probe.OBSERVABLE_LANES if lane.tracks_flag}
    assert got == _FLAG_TRACKING_LANES


def test_posterior_weight_rerank_is_disclosed_not_reported_as_firing() -> None:
    """`posterior_weight` is the resolved flag, not a firing observation.

    `LaneTelemetry(posterior_weight=weight)` is fed by
    `resolve_posterior_weight(...)` at the same call site, so a 500/500
    fire rate on it is the resolver echoed back — the exact false
    positive this probe exists to catch. It stays in the table (dropping
    it would hide that the flag is all the pipeline records) but every
    output path has to say so.
    """
    lane = next(
        x for x in probe.OBSERVABLE_LANES
        if x.name == "posterior_weight_rerank"
    )
    assert lane.tracks_flag is True
    assert "resolved weight" in lane.note


def test_every_flag_tracking_lane_carries_a_note() -> None:
    for lane in probe.OBSERVABLE_LANES:
        if lane.tracks_flag:
            assert lane.note.strip(), lane.name


def test_report_marks_the_flag_tracking_rows() -> None:
    report = probe.build_report(["p"], _observed({}))
    marked = {
        r["lane"] for r in report["observable_lanes"]
        if r["tracks_flag_by_construction"]
    }
    assert marked == _FLAG_TRACKING_LANES
    assert set(report["lanes_whose_field_tracks_the_flag"]) == marked


def test_render_marks_the_flag_tracking_rows() -> None:
    """The human rendering is what gets pasted into an issue.

    Disclosing only in the JSON leaves the table anyone actually reads
    claiming an unqualified 100% for a lane nothing observed firing.
    """
    text = probe.render(
        probe.build_report(["p"], _observed({"posterior_weight_rerank": 1}))
    )
    row = next(
        line for line in text.splitlines()
        if "posterior_weight_rerank" in line and "1.0000" in line
    )
    assert row.startswith("*")
    assert "restate the resolver" in text


# --- Diff sensitivity ----------------------------------------------------


def test_report_declares_the_sensitivity_of_each_diff_direction() -> None:
    report = probe.build_report(["p"], _observed({}))
    assert set(report["diff_sensitivity"]) == set(report["diff"])


def test_the_fired_but_not_enabled_direction_declares_zero_sensitivity(
) -> None:
    """An empty set from an instrument that cannot produce a non-empty
    one is not a null result.

    Every counter is written inside a branch guarded by the same
    resolver the `reported` side queries, so no observation on this
    table can populate this direction. Presenting the empty set as a
    finding would be the R3 IDF-clip failure mode. The direction is
    kept as a re-wiring guard, which is a different claim.
    """
    report = probe.build_report(["p"], _observed({}))
    entry = report["diff_sensitivity"]["fired_but_not_reported_enabled"]
    assert entry["can_be_populated_by_this_instrument"] is False
    assert entry["what_would_make_it_reachable"].strip()
    other = report["diff_sensitivity"]["enabled_but_never_fired"]
    assert other["can_be_populated_by_this_instrument"] is True


def test_render_qualifies_the_zero_sensitivity_direction() -> None:
    text = probe.render(probe.build_report(["p"], _observed({})))
    assert "CANNOT populate this direction" in text
    assert "NOT a null result" in text


def test_the_zero_sensitivity_direction_is_still_computed() -> None:
    """Kept, not deleted: re-wiring is what would make it fire."""
    out = probe.diff_lanes({"stealth": False}, {"stealth": 3})
    assert out["fired_but_not_reported_enabled"] == ["stealth"]


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


# --- Method constraints of the run --------------------------------------
#
# `main()` and `probe()` had no coverage at all: the instrument checks
# above exercise pure helpers, and every constraint the module docstring
# promises could be deleted with all of them still green.


class _FakeStore:
    """Records how it was opened; does nothing else."""

    opened: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, path: str, **kwargs: Any) -> None:
        type(self).opened.append((path, kwargs))

    def close(self) -> None:
        pass


def _write_corpus(root: Path, n: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "s.jsonl").write_text(
        "\n".join(json.dumps(_user(f"prompt number {i}")) for i in range(n))
        + "\n",
        encoding="utf-8",
    )


def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prompts: int = probe.MIN_PROMPTS,
) -> tuple[list[str], list[dict[str, Any]], list[str | None]]:
    """Wire `main()` to fakes and return its recorded observations.

    Returns `(argv, retrieve_kwargs, env_seen_inside_the_run)`.
    """

    store_path = tmp_path / "memory.db"
    sqlite3.connect(store_path).close()
    transcripts = tmp_path / "transcripts"
    _write_corpus(transcripts, prompts)

    _FakeStore.opened = []
    monkeypatch.setattr(probe, "MemoryStore", _FakeStore)
    monkeypatch.setattr(probe, "last_lane_telemetry", LaneTelemetry)

    seen_kwargs: list[dict[str, Any]] = []
    seen_env: list[str | None] = []

    def _fake_retrieve(store: object, query: str, **kwargs: Any) -> list[str]:
        seen_kwargs.append(kwargs)
        seen_env.append(os.environ.get("AELFRICE_SENTINEL_1366"))
        return []

    monkeypatch.setattr(retrieval, "retrieve", _fake_retrieve)
    argv = [
        "--store", str(store_path),
        "--transcripts", str(transcripts),
        "--prompts", str(probe.MIN_PROMPTS),
    ]
    return argv, seen_kwargs, seen_env


def test_main_opens_the_store_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare `MemoryStore(...)` open is a write.

    DDL, migrations, scope-id persistence and — since #1314 — the
    lock-expiry sweep, which flips a user's expired locks. Measuring a
    store is not a reason to mutate it. Drop the `read_only=True` and
    this goes red.
    """
    argv, _, _ = _harness(tmp_path, monkeypatch)
    assert probe.main(argv) == 0
    assert len(_FakeStore.opened) == 1
    assert _FakeStore.opened[0][1] == {"read_only": True}


def test_main_clears_ambient_config_for_the_run_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retrieval calls see no ambient opt-in; the caller keeps its.

    Without the clear the diff measures the developer's own environment
    rather than the shipped defaults. Delete the `pinned_environment()`
    wrapper and the first assertion fails; delete the restore and the
    last one does.
    """
    monkeypatch.setenv("AELFRICE_SENTINEL_1366", "ambient")
    argv, _, seen_env = _harness(tmp_path, monkeypatch)
    assert probe.main(argv) == 0
    assert seen_env, "the probe made no retrieval calls"
    assert set(seen_env) == {None}
    assert os.environ.get("AELFRICE_SENTINEL_1366") == "ambient"


def test_main_passes_the_production_manifest_reference_locks_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe measures the shipped call, not a convenient one.

    `hook_search.search_and_record` passes
    `manifest_reference_locks=True`; a probe that omits it measures a
    path production never takes.
    """
    argv, seen_kwargs, _ = _harness(tmp_path, monkeypatch)
    assert probe.main(argv) == 0
    assert seen_kwargs
    assert all(
        kw.get("manifest_reference_locks") is True for kw in seen_kwargs
    )


def test_main_refuses_when_the_toml_walk_is_not_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_read_toml_flag_for` walks up from the working directory.

    Clearing the environment pins only the env tier (#1295). If a
    `.aelfrice.toml` sits above the scratch cwd the TOML tier is
    unpinned and the run must refuse rather than report a diff against
    somebody's config. Delete the `scratch_walk_hits` check and this
    returns 0 with the store opened anyway.
    """
    argv, _, _ = _harness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        probe, "scratch_walk_hits", lambda _s: ["/somewhere/.aelfrice.toml"],
    )
    assert probe.main(argv) == 1
    assert _FakeStore.opened == []


def test_main_refuses_a_corpus_below_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thin corpus reads exactly like a real "every lane is dead".

    A lane that fires on 1% of calls is indistinguishable from a dead
    one on a few dozen prompts, so the run refuses instead of
    manufacturing the finding. Delete the floor check and this returns
    0 on a corpus of `MIN_PROMPTS - 1`.
    """
    argv, _, _ = _harness(
        tmp_path, monkeypatch, prompts=probe.MIN_PROMPTS - 1,
    )
    assert probe.main(argv) == 1
    assert _FakeStore.opened == []


def test_main_rejects_a_prompts_flag_below_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI guard, which the corpus-floor test does not reach.

    `_harness` always passes `--prompts MIN_PROMPTS` and varies only how
    many prompts the corpus holds, so
    `test_main_refuses_a_corpus_below_the_floor` exercises the *substantive*
    floor (exit 1) and the argument check at the top of `main()` (exit 2)
    was never executed. Two different refusals with two different exit
    codes; asserting the code is what keeps them apart.
    """
    argv, _, _ = _harness(tmp_path, monkeypatch)
    argv[argv.index("--prompts") + 1] = str(probe.MIN_PROMPTS - 1)

    assert probe.main(argv) == 2
    assert _FakeStore.opened == [], (
        "the guard must refuse before opening the store"
    )


def test_probe_calls_retrieve_once_per_prompt_with_the_shipped_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`probe()` itself, without the `main()` scaffolding."""

    seen: list[tuple[str, dict[str, Any]]] = []

    def _fake_retrieve(store: object, query: str, **kwargs: Any) -> list[str]:
        seen.append((query, kwargs))
        return []

    monkeypatch.setattr(retrieval, "retrieve", _fake_retrieve)
    monkeypatch.setattr(probe, "last_lane_telemetry", LaneTelemetry)
    out = probe.probe(object(), ["one", "two"])
    assert [q for q, _ in seen] == ["one", "two"]
    assert all(kw == {"manifest_reference_locks": True} for _, kw in seen)
    assert out["empty_output_calls"] == 2


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


# --- what `main()` actually produces ---------------------------------------
#
# Everything above `main()` asserts how it *calls* things. Nothing asserted
# what came out, so the report's own half of the diff — the `reported`
# column, resolved by `build_report` — was unpinned. That is the side whose
# constraint the module docstring states first, and moving `build_report`
# outside `pinned_environment()` left every test green while turning the
# `reported` column into a readout of the developer's shell.


def _run_and_read_report(
    argv: list[str], tmp_path: Path,
) -> dict[str, Any]:
    """Run `main()` with `--json` and return the report it wrote."""
    out = tmp_path / "report.json"
    assert probe.main([*argv, "--json", str(out)]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_reported_column_is_resolved_inside_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`reported` must describe the shipped default, not the ambient shell.

    The two halves of the diff have to be resolved under the same
    conditions or the comparison is meaningless: `observed` comes from
    retrieval calls made inside `pinned_environment()`, so `reported` has
    to be read there too. `AELFRICE_BFS` ships false; exporting it true and
    seeing `reported_enabled` come back true would mean `build_report` ran
    after the environment was restored, and every lane's "reported enabled"
    claim would be about the machine that ran the probe.

    `test_main_clears_ambient_config_for_the_run_only` does not cover this —
    it watches the *retrieve* calls, and `build_report` is a separate call
    that can drift out of the `with` block on its own.
    """
    monkeypatch.setenv("AELFRICE_BFS", "1")
    argv, _, _ = _harness(tmp_path, monkeypatch)

    report = _run_and_read_report(argv, tmp_path)

    rows = {row["lane"]: row for row in report["observable_lanes"]}
    assert "bfs_multihop" in rows, "the bfs_multihop lane vanished"
    assert rows["bfs_multihop"]["reported_enabled"] is False, (
        "AELFRICE_BFS leaked into the reported column — build_report ran "
        "outside pinned_environment(), so the diff compares shipped-default "
        "firing against this machine's configuration"
    )
    # And the caller's environment is still its own afterwards.
    assert os.environ.get("AELFRICE_BFS") == "1"


def test_the_report_counts_what_the_run_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clear is evidence, and it is per-run rather than per-import.

    A reader cannot tell a clean run from one whose clear silently matched
    nothing unless the run says how much it removed. The report carries a
    count rather than the names, because it is meant to be pasteable into a
    public issue — so the count is the only thing that can carry it, and it
    has to move with the environment.
    """
    argv, _, _ = _harness(tmp_path, monkeypatch)
    baseline = _run_and_read_report(argv, tmp_path)["environment"]

    monkeypatch.setenv("AELFRICE_BFS", "1")
    monkeypatch.setenv("AELF_SESSION_ID", "probe-1366")
    with_ambient = _run_and_read_report(argv, tmp_path)["environment"]

    assert with_ambient["cleared_env_vars"] == baseline["cleared_env_vars"] + 2, (
        "the cleared-variable count did not move when two AELF* variables "
        "were exported, so it is not measuring this run"
    )
    assert with_ambient["env_prefixes_cleared"] == ["AELFRICE_", "AELF_"]


def test_truncate_control_is_off_unless_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control is a second full pass; it must not run by default."""
    argv, seen_kwargs, _ = _harness(tmp_path, monkeypatch)

    report = _run_and_read_report(argv, tmp_path)

    assert report.get("truncation_control") in (None, {}), (
        "the truncation control ran without --truncate-control"
    )
    assert len(seen_kwargs) == probe.MIN_PROMPTS, (
        "one corpus pass expected; the control replayed it anyway"
    )


def test_truncate_control_replays_the_corpus_cut_to_the_audit_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC: the flag is wired from argv all the way into the report.

    `truncation_control()` was unit-tested against hand-built dicts, so the
    wiring in `main()` — the part that decides whether the measurement
    happens at all — had no coverage. Deleting the `if args.truncate_control`
    branch leaves the unit tests green and this red.
    """
    argv, seen_kwargs, _ = _harness(tmp_path, monkeypatch)

    report = _run_and_read_report([*argv, "--truncate-control"], tmp_path)

    assert report.get("truncation_control"), (
        "--truncate-control produced no control section"
    )
    assert len(seen_kwargs) == probe.MIN_PROMPTS * 2, (
        "the control did not replay the corpus a second time"
    )


def test_render_surfaces_the_truncation_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measurement nobody can see in the printed report is not reported.

    `render()`'s control block is the only place the truncation finding
    reaches a human reading stdout, and it was unasserted.
    """
    argv, _, _ = _harness(tmp_path, monkeypatch)
    with_control = _run_and_read_report(
        [*argv, "--truncate-control"], tmp_path
    )
    without = _run_and_read_report(argv, tmp_path)

    shown = probe.render(with_control)
    hidden = probe.render(without)

    assert "truncation control" in shown.lower(), (
        "render() dropped the truncation control from its output"
    )
    assert len(shown) > len(hidden), (
        "render() produced the same text with and without the control"
    )
