"""Tests for `aelf introspect` — honest-signal belief view (#1081).

The deterministic core is `aelfrice.introspect.build_report`; the CLI wrapper
(`aelf introspect`) formats it as text or JSON. Module tests drive an in-memory
store directly; CLI tests use the `main(argv=..., out=...)` harness over an
isolated file DB. Entity rows are inserted directly (mirroring the #1096 test)
so grounding is controlled precisely.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.introspect import (
    GROUNDING_DURABLE,
    GROUNDING_EPHEMERAL,
    GROUNDING_NEUTRAL,
    STATUS_DECIDED,
    STATUS_DECIDES,
    STATUS_FLOATED,
    STATUS_STALE,
    build_report,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_POTENTIALLY_STALE,
    EDGE_RESOLVES,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    content: str = "plain belief content",
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    session_id: str | None = None,
    project_context: str = "",
    lock_level: str = LOCK_NONE,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        created_at="2026-06-01T00:00:00Z",
        last_retrieved_at=None,
        session_id=session_id,
        origin=ORIGIN_AGENT_INFERRED,
        project_context=project_context,
    )


def _add_entity(store: MemoryStore, bid: str, lower: str, kind: str) -> None:
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO belief_entities(belief_id, entity_lower, entity_raw, "
        "kind, span_start, span_end) VALUES (?,?,?,?,0,0)",
        (bid, lower, lower, kind),
    )
    store._conn.commit()


def _clear_entities(store: MemoryStore, bid: str) -> None:
    store._conn.execute(  # noqa: SLF001
        "DELETE FROM belief_entities WHERE belief_id=?", (bid,)
    )
    store._conn.commit()


def _by_id(report: object) -> dict:
    out = {}
    for g in report.groups:  # type: ignore[attr-defined]
        for b in g.beliefs:
            out[b.id] = b
    return out


# --- empty -------------------------------------------------------------------


def test_empty_store_reports_nothing() -> None:
    s = MemoryStore(":memory:")
    try:
        report = build_report(s)
    finally:
        s.close()
    assert report.total == 0
    assert report.groups == ()
    assert report.noise_total == 0


# --- grouping ----------------------------------------------------------------


def test_group_by_session_splits_and_orders_no_session_last() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("a", session_id="sess-b"))
        s.insert_belief(_mk("b", session_id="sess-a"))
        s.insert_belief(_mk("c", session_id=None))
        report = build_report(s)
    finally:
        s.close()
    labels = [g.label for g in report.groups]
    assert labels == ["sess-a", "sess-b", "(no session)"]


def test_group_by_project() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("a", project_context="proj-x"))
        s.insert_belief(_mk("b", project_context=""))
        report = build_report(s, group_by="project")
    finally:
        s.close()
    labels = [g.label for g in report.groups]
    assert labels == ["proj-x", "(no project)"]


def test_invalid_group_by_raises() -> None:
    s = MemoryStore(":memory:")
    try:
        with pytest.raises(ValueError, match="group_by"):
            build_report(s, group_by="theme")
    finally:
        s.close()


def test_session_filter() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("a", session_id="keep"))
        s.insert_belief(_mk("b", session_id="drop"))
        report = build_report(s, session="keep")
    finally:
        s.close()
    assert set(_by_id(report)) == {"a"}


# --- signals: posterior + evidence + recurrence ------------------------------


def test_posterior_and_evidence() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("a", alpha=3.0, beta=1.0))
        report = build_report(s)
    finally:
        s.close()
    sig = _by_id(report)["a"]
    assert sig.posterior_mean == pytest.approx(0.75)
    assert sig.evidence == pytest.approx(4.0)


# --- signals: grounding ------------------------------------------------------


def test_grounding_durable() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("a"))
        _clear_entities(s, "a")
        _add_entity(s, "a", "src/foo.py", "file_path")
        report = build_report(s)
    finally:
        s.close()
    assert _by_id(report)["a"].grounding == GROUNDING_DURABLE


def test_grounding_ephemeral() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("a"))
        _clear_entities(s, "a")
        _add_entity(s, "a", "#879", "identifier")  # bare number -> transient
        report = build_report(s)
    finally:
        s.close()
    assert _by_id(report)["a"].grounding == GROUNDING_EPHEMERAL


def test_grounding_neutral_when_absent() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("a"))
        _clear_entities(s, "a")
        report = build_report(s)
    finally:
        s.close()
    assert _by_id(report)["a"].grounding == GROUNDING_NEUTRAL


# --- signals: status (floated vs decided) ------------------------------------


def test_status_floated_default() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("a"))
        report = build_report(s)
    finally:
        s.close()
    assert _by_id(report)["a"].status == STATUS_FLOATED


def test_status_decided_on_incoming_resolves() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("winner"))
        s.insert_belief(_mk("target"))
        s.insert_edge(Edge(src="winner", dst="target", type=EDGE_RESOLVES, weight=1.0))
        report = build_report(s)
    finally:
        s.close()
    sigs = _by_id(report)
    assert sigs["target"].status == STATUS_DECIDED
    assert sigs["winner"].status == STATUS_DECIDES


def test_status_stale_on_incoming_potentially_stale() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("a"))
        s.insert_belief(_mk("marker"))
        s.insert_edge(
            Edge(src="marker", dst="a", type=EDGE_POTENTIALLY_STALE, weight=0.0)
        )
        report = build_report(s)
    finally:
        s.close()
    assert _by_id(report)["a"].status == STATUS_STALE


# --- signals: noise + lock ---------------------------------------------------


def test_noise_flag_on_stranded_header() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("junk", content="Recommendation:"))
        s.insert_belief(_mk("real", content="the parser handles nested quotes"))
        report = build_report(s)
    finally:
        s.close()
    sigs = _by_id(report)
    assert sigs["junk"].noise is True
    assert sigs["real"].noise is False
    assert report.noise_total == 1


def test_lock_signal() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("a", lock_level=LOCK_USER))
        report = build_report(s)
    finally:
        s.close()
    assert _by_id(report)["a"].lock_level == LOCK_USER


# --- ordering within a group -------------------------------------------------


def test_noise_sorts_first_in_group() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("clean", content="a durable technical statement"))
        s.insert_belief(_mk("junk", content="$ python run_all.py"))
        report = build_report(s)
    finally:
        s.close()
    order = [b.id for b in report.groups[0].beliefs]
    assert order[0] == "junk"


def test_ephemeral_sorts_before_durable() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("dur"))
        _clear_entities(s, "dur")
        _add_entity(s, "dur", "src/a.py", "file_path")  # durable
        s.insert_belief(_mk("eph"))
        _clear_entities(s, "eph")
        _add_entity(s, "eph", "#12", "identifier")  # ephemeral
        report = build_report(s)
    finally:
        s.close()
    order = [b.id for b in report.groups[0].beliefs]
    assert order.index("eph") < order.index("dur")


# --- only_noise + limit + soft-delete ----------------------------------------


def test_only_noise_filters_to_junk() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("junk", content="Recommendation:"))
        s.insert_belief(_mk("real", content="a real belief about the code"))
        report = build_report(s, only_noise=True)
    finally:
        s.close()
    assert set(_by_id(report)) == {"junk"}


def test_limit_caps_count() -> None:
    s = MemoryStore(":memory:")
    try:
        for i in range(5):
            s.insert_belief(_mk(f"b{i}"))
        report = build_report(s, limit=2)
    finally:
        s.close()
    assert report.total == 2


def test_soft_deleted_beliefs_excluded() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("live"))
        s.insert_belief(_mk("gone"))
        s.soft_delete_belief("gone")
        report = build_report(s)
    finally:
        s.close()
    assert set(_by_id(report)) == {"live"}


# --- CLI wrapper -------------------------------------------------------------


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "aelf.db"
    monkeypatch.setenv("AELFRICE_DB", str(p))
    return p


def _seed(db: Path, *beliefs: Belief) -> None:
    s = MemoryStore(str(db))
    try:
        for b in beliefs:
            s.insert_belief(b)
    finally:
        s.close()


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    code = main(argv=list(argv), out=buf)
    return code, buf.getvalue()


def test_cli_text_view_has_header_legend_and_footer(isolated_db: Path) -> None:
    _seed(isolated_db, _mk("aabbccdd11223344", content="a stored belief", session_id="s1"))
    code, out = _run("introspect")
    assert code == 0
    assert "introspect:" in out
    assert "aabbccdd11223344" in out
    assert "legend:" in out
    assert "aelf retire <id>" in out


def test_cli_empty_message(isolated_db: Path) -> None:
    code, out = _run("introspect")
    assert code == 0
    assert "no active beliefs match" in out


def test_cli_json_structure(isolated_db: Path) -> None:
    _seed(
        isolated_db,
        _mk("aabbccdd11223344", content="belief one", session_id="s1", alpha=3.0, beta=1.0),
    )
    code, out = _run("introspect", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["group_by"] == "session"
    assert payload["total"] == 1
    b = payload["groups"][0]["beliefs"][0]
    assert b["id"] == "aabbccdd11223344"
    assert b["posterior_mean"] == 0.75
    assert b["status"] == STATUS_FLOATED
    assert b["grounding"] == GROUNDING_NEUTRAL


def test_cli_only_noise_flag(isolated_db: Path) -> None:
    _seed(
        isolated_db,
        _mk("1111ccdd11223344", content="Recommendation:", session_id="s1"),
        _mk("2222ccdd11223344", content="a genuine belief", session_id="s1"),
    )
    code, out = _run("introspect", "--only-noise", "--json")
    assert code == 0
    payload = json.loads(out)
    ids = [b["id"] for g in payload["groups"] for b in g["beliefs"]]
    assert ids == ["1111ccdd11223344"]


def test_cli_by_project(isolated_db: Path) -> None:
    _seed(isolated_db, _mk("aabbccdd11223344", project_context="proj-x"))
    code, out = _run("introspect", "--by", "project", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["group_by"] == "project"
    assert payload["groups"][0]["label"] == "proj-x"


def test_cli_limit_zero_means_no_cap(isolated_db: Path) -> None:
    beliefs = [_mk(f"{i:016x}") for i in range(3)]
    _seed(isolated_db, *beliefs)
    code, out = _run("introspect", "--limit", "0", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["total"] == 3
