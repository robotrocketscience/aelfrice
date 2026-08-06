"""`aelf doctor` counts edges whose endpoints do not exist (#1375).

The `edges` table is `PRIMARY KEY (src, dst, type)` with no foreign key,
and `insert_edge` gates only on federation ownership —
`assert_local_ownership` is a documented no-op for an id the store has
never seen. So an edge naming a belief that does not exist is written
without complaint, and until #1375 nothing counted them:
`--classify-orphans` is about *belief* orphans (#206), which is a
different set entirely.

Report-only. Repairing the rows means a foreign key on `edges`, which
needs a table rebuild, and an `edges` migration is what bricked stores in
#1161 — explicitly out of scope for this leaf.
"""
from __future__ import annotations

from pathlib import Path

from aelfrice.doctor import (
    DanglingEdgeStats,
    diagnose,
    diagnose_dangling_edges,
    format_report,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_CONTRADICTS,
    EDGE_DERIVED_FROM,
    EDGE_IMPLEMENTS,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TEMPORAL_NEXT,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk(belief_id: str) -> Belief:
    return Belief(
        id=belief_id,
        content=belief_id,
        content_hash=f"h_{belief_id}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level="none",
        locked_at=None,
        created_at="2026-08-05T00:00:00Z",
        last_retrieved_at=None,
    )


def _store(tmp_path: Path) -> tuple[Path, MemoryStore]:
    path = tmp_path / "memory.db"
    store = MemoryStore(str(path))
    for bid in ("A", "B", "C"):
        store.insert_belief(_mk(bid))
    return path, store


def test_clean_store_reports_zero(tmp_path: Path) -> None:
    path, store = _store(tmp_path)
    try:
        store.insert_edge(
            Edge(src="A", dst="B", type=EDGE_SUPPORTS, weight=1.0)
        )
        store.insert_edge(
            Edge(src="B", dst="C", type=EDGE_RELATES_TO, weight=1.0)
        )
    finally:
        store.close()

    stats = diagnose_dangling_edges(str(path))

    assert stats == DanglingEdgeStats(
        total=0, total_edges=2, missing_src=0, missing_dst=0, by_type=()
    )


def test_counts_each_missing_endpoint_separately(tmp_path: Path) -> None:
    """One missing src, one missing dst, and one edge missing both.

    The both-missing edge is the fixture that separates a correct query
    from an `AND`, from a `UNION` of two per-endpoint scans, and from a
    per-endpoint sum: it is one row, but two absent endpoints.
    """
    path, store = _store(tmp_path)
    try:
        store.insert_edge(
            Edge(src="A", dst="B", type=EDGE_SUPPORTS, weight=1.0)
        )
        store.insert_edge(
            Edge(src="GHOST", dst="B", type=EDGE_SUPPORTS, weight=1.0)
        )
        store.insert_edge(
            Edge(src="A", dst="GHOST", type=EDGE_RELATES_TO, weight=1.0)
        )
        store.insert_edge(
            Edge(
                src="GHOST_SRC",
                dst="GHOST_DST",
                type=EDGE_TEMPORAL_NEXT,
                weight=1.0,
            )
        )
    finally:
        store.close()

    stats = diagnose_dangling_edges(str(path))

    assert stats is not None
    assert stats.total == 3, "the both-missing edge must count once"
    assert stats.total_edges == 4
    assert stats.missing_src == 2
    assert stats.missing_dst == 2
    assert stats.by_type == (
        (EDGE_RELATES_TO, 1),
        (EDGE_SUPPORTS, 1),
        (EDGE_TEMPORAL_NEXT, 1),
    )


def test_retired_belief_endpoint_is_not_dangling(tmp_path: Path) -> None:
    """A soft-deleted belief is still a row, so its edges are intact.

    `aelf retire` sets `valid_to` and deliberately preserves the evidence
    trail, `aelf restore` being the inverse. A check that joined against
    the live view instead of the table would report every retired
    belief's edges as broken and bury the real ones.
    """
    path, store = _store(tmp_path)
    try:
        store.insert_edge(
            Edge(src="A", dst="B", type=EDGE_SUPPORTS, weight=1.0)
        )
        store.soft_delete_belief("B")
        assert store.get_belief("B", include_retired=True) is not None
        assert store.get_belief("B") is None, (
            "the fixture must actually retire B, or this test is vacuous"
        )
    finally:
        store.close()

    stats = diagnose_dangling_edges(str(path))

    assert stats is not None
    assert stats.total == 0


def test_fail_soft_on_unreadable_store(tmp_path: Path) -> None:
    assert diagnose_dangling_edges(str(tmp_path / "nope.db")) is None
    assert diagnose_dangling_edges(":memory:") is None


def test_format_report_renders_the_count(tmp_path: Path) -> None:
    path, store = _store(tmp_path)
    try:
        store.insert_edge(
            Edge(src="A", dst="GHOST", type=EDGE_RELATES_TO, weight=1.0)
        )
    finally:
        store.close()

    report = diagnose(
        user_settings=tmp_path / "missing-user.json",
        project_root=tmp_path / "missing-project",
        store_path=str(path),
    )

    assert report.dangling_edges is not None
    assert report.dangling_edges.total == 1
    text = format_report(report)
    assert "dangling edges" in text
    assert "1 of 1 edge(s)" in text
    assert f"    {EDGE_RELATES_TO}: 1" in text


def test_format_report_says_so_when_clean(tmp_path: Path) -> None:
    """The zero reading is rendered too.

    A section that appears only on failure cannot be told apart from a
    section that never ran, which is the state #1375 found the whole
    check in.
    """
    path, store = _store(tmp_path)
    try:
        store.insert_edge(
            Edge(src="A", dst="B", type=EDGE_SUPPORTS, weight=1.0)
        )
    finally:
        store.close()

    report = diagnose(
        user_settings=tmp_path / "missing-user.json",
        project_root=tmp_path / "missing-project",
        store_path=str(path),
    )

    assert "none of 1 edge(s)" in format_report(report)


def test_section_absent_without_a_store_path(tmp_path: Path) -> None:
    report = diagnose(
        user_settings=tmp_path / "missing-user.json",
        project_root=tmp_path / "missing-project",
    )
    assert report.dangling_edges is None
    assert "dangling edges" not in format_report(report)


def test_asymmetric_fixture_pins_src_dst_order_sort_and_truncation(
    tmp_path: Path,
) -> None:
    """One fixture for the three constructs a symmetric one cannot see.

    Every other case here is symmetric — equal `missing_src` and
    `missing_dst`, one dangling edge per type, and fewer types than
    `DANGLING_EDGE_TYPES_SHOWN`. Under those inputs three shipped
    constructs are free variables: the `SUM(bs.id IS NULL),
    SUM(bd.id IS NULL)` column order transposes undetected, `ORDER BY n
    DESC` can be `ASC` because the `e.type` tie-break carries the
    assertion when every count is 1, and the "... and N more type(s)"
    remainder never renders. All three were confirmed dead by mutation
    before this was written.

    So: unequal src/dst counts, unequal per-type counts, and **seven**
    dangling types against a limit of five. The rendered line is read
    here too, because `_format_dangling_edges_section` is where the
    src/dst pair reaches a human and no other test asserts its text.
    """
    path, store = _store(tmp_path)
    # 7 types; counts deliberately distinct so DESC and ASC disagree on
    # the head, and so the head is not reachable by the type tie-break.
    plan = (
        (EDGE_SUPPORTS, 4, "src"),
        (EDGE_RELATES_TO, 3, "src"),
        (EDGE_TEMPORAL_NEXT, 2, "dst"),
        (EDGE_CITES, 1, "dst"),
        (EDGE_CONTRADICTS, 1, "dst"),
        (EDGE_SUPERSEDES, 1, "dst"),
        (EDGE_DERIVED_FROM, 1, "dst"),
    )
    try:
        for edge_type, count, missing_end in plan:
            for i in range(count):
                ghost = f"GHOST_{edge_type}_{i}"
                src = ghost if missing_end == "src" else "A"
                dst = "B" if missing_end == "src" else ghost
                store.insert_edge(
                    Edge(src=src, dst=dst, type=edge_type, weight=1.0)
                )
    finally:
        store.close()

    stats = diagnose_dangling_edges(str(path))
    assert stats is not None

    # (a) src/dst are not interchangeable: 7 edges miss src, 6 miss dst.
    assert stats.missing_src == 7
    assert stats.missing_dst == 6

    # (b) largest first, and the head is decided by the count rather
    # than by the alphabetical tie-break — SUPPORTS sorts last of the
    # seven by name, so seeing it first can only be the DESC ordering.
    assert stats.by_type[0] == (EDGE_SUPPORTS, 4)
    assert stats.by_type[1] == (EDGE_RELATES_TO, 3)
    assert stats.by_type[2] == (EDGE_TEMPORAL_NEXT, 2)
    assert [n for _t, n in stats.by_type] == sorted(
        [n for _t, n in stats.by_type], reverse=True
    )

    report = diagnose(
        user_settings=tmp_path / "missing-user.json",
        project_root=tmp_path / "missing-project",
        store_path=str(path),
    )
    rendered = format_report(report)

    # (c) the rendered src/dst pair, which no other test reads.
    assert "7 missing src, 6 missing dst" in rendered
    # (d) truncation fires: 7 types, 5 shown, 2 held back.
    assert f"{EDGE_SUPPORTS}: 4" in rendered
    assert "... and 2 more type(s)" in rendered
