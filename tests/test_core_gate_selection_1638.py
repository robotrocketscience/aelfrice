"""Core admission gate in core selection (#1638).

The gate applies to the two non-lock arms of core at all three consumers:
`aelf core`, the SessionStart `<core>` lane, and the core membership the
`aelf doctor --gc-filesystem-corroboration` report reads. Rule
(`core_gate.admit`): no label passes on today's rule; A is admitted; B is
admitted only with fewer than 2 episode-qualified corroborations; C never.
Locked beliefs are unaffected.

Each test names the mutation that kills it. The fixture text is neutral
and synthetic; every store lives under `tmp_path`.
"""
from __future__ import annotations

import io
import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from aelfrice.core_gate import (
    B_CORROBORATION_LIMIT,
    CLASSIFIER_VERSION,
    LABEL_NEEDS_CONTEXT,
    LABEL_NOT_A_CLAIM,
    LABEL_SELF_CONTAINED,
    admit,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_AGENT_INFERRED,
    Belief,
    episode_qualified_corroborations,
)
from aelfrice.store import MemoryStore

_TX = CORROBORATION_SOURCE_TRANSCRIPT_INGEST
_FS = CORROBORATION_SOURCE_FILESYSTEM_INGEST

# In core through the corroboration arm: two rows on separate days, so its
# episode-qualified count is 2.
CORR_ARM = "corrarm000000001"
# In core through the posterior arm, with no corroboration rows: count 0.
POST_ARM = "postarm000000001"
# Posterior arm plus one row a day later: two episodes, count 1.
POST_ONE = "postone000000001"
# Posterior arm plus two rows on separate days: count 2 (and the
# corroboration arm fires too).
POST_TWO = "posttwo000000001"
# Posterior arm plus thirteen rows inside the creation hour: one episode,
# so the episode-qualified count is 0 although the raw row count is 13.
POST_BURST = "postburst0000001"
# Locked: in core through the lock arm, whatever its label.
LOCKED = "locked0000000001"
# Meets no arm: never in core.
WEAK = "weak000000000001"

UNLOCKED_CORE = frozenset({CORR_ARM, POST_ARM, POST_ONE, POST_TWO, POST_BURST})
ALL_IDS = (CORR_ARM, POST_ARM, POST_ONE, POST_TWO, POST_BURST, LOCKED, WEAK)


def _hash(bid: str) -> str:
    return f"h_{bid}"


def _mk(store: MemoryStore, bid: str, *, alpha: float = 1.0,
        created_at: str = "2026-08-01T09:00:00+00:00",
        locked: bool = False) -> None:
    store.insert_belief(Belief(
        id=bid, content=f"the widget {bid} has a stated property",
        content_hash=_hash(bid), alpha=alpha, beta=1.0, type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-08-01T09:00:00+00:00" if locked else None,
        created_at=created_at, last_retrieved_at=None,
        origin=ORIGIN_AGENT_INFERRED,
    ))


def _rows(store: MemoryStore, bid: str, times: list[str],
          source: str = _TX) -> None:
    for i, ts in enumerate(times):
        store.record_corroboration(
            bid, source_type=source, session_id=f"{bid}-s{i}", ts=ts,
        )


def _day(d: int) -> str:
    return f"2026-08-{d:02d}T09:00:00+00:00"


def build_fixture(path: Path) -> None:
    """The fixture store, with no labels. Also used to derive the golden
    output below from the tree before the gate."""
    store = MemoryStore(str(path))
    try:
        _mk(store, CORR_ARM)
        _rows(store, CORR_ARM, [_day(2), _day(5)])
        _mk(store, POST_ARM, alpha=9.0)
        _mk(store, POST_ONE, alpha=8.0)
        _rows(store, POST_ONE, [_day(3)])
        _mk(store, POST_TWO, alpha=7.0)
        _rows(store, POST_TWO, [_day(2), _day(3)])
        _mk(store, POST_BURST, alpha=6.0, created_at="2026-08-01T12:00:00+00:00")
        _rows(store, POST_BURST, [
            f"2026-08-01T12:{(i * 9) // 60:02d}:{(i * 9) % 60:02d}+00:00"
            for i in range(13)
        ])
        _mk(store, LOCKED, locked=True)
        _mk(store, WEAK)
    finally:
        store.close()


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "memory.db"
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(path))
    build_fixture(path)
    return path


def _label(db: Path, labels: dict[str, str],
           version: str = CLASSIFIER_VERSION) -> None:
    store = MemoryStore(str(db))
    try:
        store.put_core_gate_labels(
            {_hash(bid): lab for bid, lab in labels.items()},
            classifier_version=version, batch_id=None,
            labeled_at="2026-10-05T00:00:00+00:00",
        )
    finally:
        store.close()


def _label_all(db: Path, label: str) -> None:
    _label(db, {bid: label for bid in ALL_IDS})


# --- the three consumers, each returning its unlocked core ids ------------


def _cli_out(argv: list[str]) -> str:
    from aelfrice.cli import main

    buf = io.StringIO()
    assert main(argv=argv, out=buf) == 0
    return buf.getvalue()


def _cli_rows() -> list[dict[str, object]]:
    rows = json.loads(_cli_out(["core", "--json"]))
    assert isinstance(rows, list)
    return rows  # pyright: ignore[reportUnknownVariableType]


def _via_cli(db: Path, tmp_path: Path) -> set[str]:
    return {str(r["id"]) for r in _cli_rows() if r["lock_level"] == LOCK_NONE}


def _subblock(store: MemoryStore, tmp_path: Path) -> str:
    from aelfrice.hook import _build_session_start_subblock  # pyright: ignore[reportPrivateUsage]

    return _build_session_start_subblock(store, cwd=tmp_path)


def _core_section_ids(block: str) -> set[str]:
    m = re.search(r"<core>(.*?)</core>", block, re.S)
    assert m is not None, block
    return set(re.findall(r'<belief id="([^"]+)"', m.group(1)))


def _via_hook(db: Path, tmp_path: Path) -> set[str]:
    store = MemoryStore(str(db))
    try:
        return _core_section_ids(_subblock(store, tmp_path))
    finally:
        store.close()


def _via_doctor(db: Path, tmp_path: Path) -> set[str]:
    from aelfrice.cli import default_core_rule
    from aelfrice.doctor import _core_members  # pyright: ignore[reportPrivateUsage]

    store = MemoryStore(str(db))
    try:
        return _core_members(store, list(ALL_IDS), default_core_rule)
    finally:
        store.close()


Consumer = Callable[[Path, Path], set[str]]
CONSUMERS: list[Consumer] = [_via_cli, _via_hook, _via_doctor]
CONSUMER_IDS = ["aelf-core", "session-start-core", "doctor-core-members"]


# --- the predicate --------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "count", "expected"),
    [
        (None, 0, True), (None, 5, True),
        (LABEL_SELF_CONTAINED, 0, True), (LABEL_SELF_CONTAINED, 5, True),
        (LABEL_NEEDS_CONTEXT, 0, True), (LABEL_NEEDS_CONTEXT, 1, True),
        (LABEL_NEEDS_CONTEXT, 2, False), (LABEL_NEEDS_CONTEXT, 5, False),
        (LABEL_NOT_A_CLAIM, 0, False), (LABEL_NOT_A_CLAIM, 5, False),
    ],
)
def test_admit_rule(label: str | None, count: int, expected: bool) -> None:
    """Mutations: drop the C branch; `<` to `<=` on B; unlabeled -> False;
    A -> False. Each flips at least one row."""
    assert admit(label, count) is expected


def test_b_threshold_is_two() -> None:
    """Pins the ruled 1x constant. Mutation: B_CORROBORATION_LIMIT = 3."""
    assert B_CORROBORATION_LIMIT == 2


@pytest.mark.parametrize(
    ("count", "episodes", "expected"),
    [(13, 1, 0), (1, 1, 0), (1, 2, 1), (2, 2, 2), (2, 3, 2), (0, 0, 0)],
)
def test_episode_qualified_count(count: int, episodes: int, expected: int) -> None:
    """The count the corroboration arm uses: rows count only across two or
    more episodes. Mutations: return the raw count (kills (13, 1)); return
    `episodes` (kills (1, 2) and (2, 3))."""
    assert episode_qualified_corroborations(count, episodes) == expected


# --- per-consumer outcomes ------------------------------------------------


@pytest.mark.timeout(60)
@pytest.mark.parametrize("consumer", CONSUMERS, ids=CONSUMER_IDS)
def test_unlabeled_passes_through(db: Path, tmp_path: Path, consumer: Consumer) -> None:
    """Mutation: unlabeled -> False in `admit`."""
    assert consumer(db, tmp_path) == UNLOCKED_CORE


@pytest.mark.timeout(60)
@pytest.mark.parametrize("consumer", CONSUMERS, ids=CONSUMER_IDS)
def test_label_a_admits_both_arms(db: Path, tmp_path: Path, consumer: Consumer) -> None:
    """Mutation: A -> False in `admit`."""
    _label_all(db, LABEL_SELF_CONTAINED)
    assert consumer(db, tmp_path) == UNLOCKED_CORE


@pytest.mark.timeout(60)
@pytest.mark.parametrize("consumer", CONSUMERS, ids=CONSUMER_IDS)
def test_label_c_drops_both_arms(db: Path, tmp_path: Path, consumer: Consumer) -> None:
    """Mutations: drop the C branch in `admit`; drop the
    `gate_core_candidates` call at this consumer."""
    _label_all(db, LABEL_NOT_A_CLAIM)
    assert consumer(db, tmp_path) == set()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("consumer", CONSUMERS, ids=CONSUMER_IDS)
def test_label_b_admits_below_two_episode_qualified(
    db: Path, tmp_path: Path, consumer: Consumer,
) -> None:
    """B with count 0 or 1 is admitted, with count 2 is not, on both arms.

    Mutations: `<` to `<=` on B (admits CORR_ARM and POST_TWO); pass the
    raw row count (drops POST_BURST, whose 13 rows are one episode); pass
    the episode count (drops POST_ONE, whose one row makes two episodes);
    drop the `gate_core_candidates` call at this consumer.
    """
    _label_all(db, LABEL_NEEDS_CONTEXT)
    assert consumer(db, tmp_path) == {POST_ARM, POST_ONE, POST_BURST}


@pytest.mark.timeout(60)
@pytest.mark.parametrize("consumer", CONSUMERS, ids=CONSUMER_IDS)
def test_label_under_another_version_is_ignored(
    db: Path, tmp_path: Path, consumer: Consumer,
) -> None:
    """Mutation: drop the `classifier_version` filter from the lookup."""
    _label(db, {bid: LABEL_NOT_A_CLAIM for bid in ALL_IDS},
           version="core-gate-0000000000000000")
    assert consumer(db, tmp_path) == UNLOCKED_CORE


@pytest.mark.timeout(60)
@pytest.mark.parametrize("consumer", CONSUMERS, ids=CONSUMER_IDS)
def test_mixed_labels(db: Path, tmp_path: Path, consumer: Consumer) -> None:
    """Labels are matched per content hash, not applied to the whole
    selection. Mutation: key the lookup result by belief id."""
    _label(db, {CORR_ARM: LABEL_SELF_CONTAINED, POST_ARM: LABEL_NOT_A_CLAIM,
                POST_TWO: LABEL_NEEDS_CONTEXT})
    assert consumer(db, tmp_path) == {CORR_ARM, POST_ONE, POST_BURST}


# --- locked beliefs -------------------------------------------------------


@pytest.mark.timeout(60)
def test_locked_unaffected_in_aelf_core(db: Path) -> None:
    """Mutation: run the locked list through `gate_core_candidates`."""
    _label_all(db, LABEL_NOT_A_CLAIM)
    rows = _cli_rows()
    assert [r["id"] for r in rows] == [LOCKED]
    assert "core_gate_label" not in rows[0]


@pytest.mark.timeout(60)
def test_locked_unaffected_in_session_start(db: Path, tmp_path: Path) -> None:
    """Mutation: run the locked list through `gate_core_candidates`."""
    _label_all(db, LABEL_NOT_A_CLAIM)
    store = MemoryStore(str(db))
    try:
        block = _subblock(store, tmp_path)
    finally:
        store.close()
    locked = re.search(r"<locked>(.*?)</locked>", block, re.S)
    assert locked is not None
    assert f'<belief id="{LOCKED}"' in locked.group(1)


# --- `aelf core` output ---------------------------------------------------

# Derived from the tree before the gate (github/feat/issue-1638-core-gate-
# storage at 8426dc27) by running `aelf core` on `build_fixture`'s store.
_GOLDEN_TEXT = (
    f"{LOCKED} [LOCK]: the widget {LOCKED} has a stated property\n"
    f"{POST_ARM} [α=9.0,β=1.0,μ=0.900]: the widget {POST_ARM} has a stated property\n"
    f"{POST_ONE} [α=8.0,β=1.0,μ=0.889]: the widget {POST_ONE} has a stated property\n"
    f"{POST_TWO} [CORR=2,α=7.0,β=1.0,μ=0.875]: the widget {POST_TWO} has a stated property\n"
    f"{POST_BURST} [α=6.0,β=1.0,μ=0.857]: the widget {POST_BURST} has a stated property\n"
    f"{CORR_ARM} [CORR=2]: the widget {CORR_ARM} has a stated property\n"
)


@pytest.mark.timeout(60)
def test_text_output_without_labels_is_unchanged(db: Path) -> None:
    """Byte-identical to the tree before the gate. Mutation: render a label
    tag in text mode; unlabeled -> False."""
    assert _cli_out(["core"]) == _GOLDEN_TEXT


@pytest.mark.timeout(60)
def test_json_output_without_labels_has_no_label_field(db: Path) -> None:
    """Mutation: always emit `core_gate_label` (as null when unlabeled)."""
    rows = _cli_rows()
    assert [r["id"] for r in rows] == [
        LOCKED, POST_ARM, POST_ONE, POST_TWO, POST_BURST, CORR_ARM,
    ]
    assert all("core_gate_label" not in r for r in rows)


@pytest.mark.timeout(60)
def test_json_output_carries_the_label_when_one_exists(db: Path) -> None:
    """Mutation: drop the `core_gate_label` field."""
    _label(db, {POST_ARM: LABEL_SELF_CONTAINED, POST_ONE: LABEL_NEEDS_CONTEXT,
                LOCKED: LABEL_SELF_CONTAINED})
    by_id = {str(r["id"]): r for r in _cli_rows()}
    assert by_id[POST_ARM]["core_gate_label"] == LABEL_SELF_CONTAINED
    assert by_id[POST_ONE]["core_gate_label"] == LABEL_NEEDS_CONTEXT
    assert "core_gate_label" not in by_id[CORR_ARM]
    # Locked beliefs are not gated, so their label is not shown.
    assert "core_gate_label" not in by_id[LOCKED]


@pytest.mark.timeout(60)
def test_text_output_drops_gated_beliefs_only(db: Path) -> None:
    """Mutation: drop the `gate_core_candidates` call in `_cmd_core`."""
    _label(db, {POST_ARM: LABEL_NOT_A_CLAIM})
    expected = "".join(
        line + "\n" for line in _GOLDEN_TEXT.splitlines()
        if not line.startswith(POST_ARM)
    )
    assert _cli_out(["core"]) == expected


# --- a store written before the label table, opened read-only -------------


def _strip_gate_tables(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("DROP TABLE core_gate_labels")
        conn.execute("DROP TABLE core_gate_batches")
        conn.commit()
    finally:
        conn.close()


@pytest.mark.timeout(60)
def test_read_only_store_without_the_label_table_passes_through(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-gate store on the `open_store_for_read` fallback has no label
    table; every consumer reads that as no labels. Mutation: drop the
    missing-table branch in `core_gate_labels_for` (it raises)."""
    import aelfrice.cli as cli

    _strip_gate_tables(db)
    ro = MemoryStore(str(db), read_only=True)
    try:
        assert "core_gate_labels" not in {
            str(r[0]) for r in ro._conn.execute(  # pyright: ignore[reportPrivateUsage]
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert ro.core_gate_labels_for([_hash(POST_ARM)], CLASSIFIER_VERSION) == {}
        assert _core_section_ids(_subblock(ro, tmp_path)) == UNLOCKED_CORE
        from aelfrice.doctor import _core_members  # pyright: ignore[reportPrivateUsage]

        assert _core_members(ro, list(ALL_IDS), cli.default_core_rule) == UNLOCKED_CORE
    finally:
        ro.close()

    monkeypatch.setattr(
        cli, "open_store_for_read",
        lambda: MemoryStore(str(db), read_only=True),
    )
    assert _cli_out(["core"]) == _GOLDEN_TEXT


# --- the doctor report ----------------------------------------------------


@pytest.fixture
def fs_store(tmp_path: Path) -> Iterator[MemoryStore]:
    store = MemoryStore(str(tmp_path / "fs.db"))
    # Posterior arm, with two filesystem rows on separate days: its
    # episode-qualified count is 2 now and 0 once the rows are gone.
    _mk(store, "fsposterior00001", alpha=9.0)
    _rows(store, "fsposterior00001", [_day(2), _day(3)], source=_FS)
    try:
        yield store
    finally:
        store.close()


@pytest.mark.timeout(60)
def test_gc_report_lists_a_b_belief_entering_core(fs_store: MemoryStore) -> None:
    """Deleting rows moves a B belief into core. Mutation: drop the
    `entering_core` assignment, or its lines in the formatter."""
    from aelfrice.cli import default_core_rule
    from aelfrice.doctor import (
        format_filesystem_corroboration_report,
        gc_filesystem_corroboration,
    )

    fs_store.put_core_gate_labels(
        {_hash("fsposterior00001"): LABEL_NEEDS_CONTEXT},
        classifier_version=CLASSIFIER_VERSION, batch_id=None,
        labeled_at="2026-10-05T00:00:00+00:00",
    )
    report = gc_filesystem_corroboration(fs_store, qualifies=default_core_rule)
    assert report.leaving_core == []
    assert report.entering_core == ["fsposterior00001"]
    text = format_filesystem_corroboration_report(report)
    assert "beliefs entering `aelf core`: 1\n  fsposterior00001" in text


@pytest.mark.timeout(60)
def test_gc_report_without_labels_has_no_entering_line(fs_store: MemoryStore) -> None:
    """Unlabeled, the belief is in core before and after, and the report
    reads as before the gate. Mutation: print the entering line when empty."""
    from aelfrice.cli import default_core_rule
    from aelfrice.doctor import (
        format_filesystem_corroboration_report,
        gc_filesystem_corroboration,
    )

    report = gc_filesystem_corroboration(fs_store, qualifies=default_core_rule)
    assert report.entering_core == []
    assert report.leaving_core == []
    assert "entering" not in format_filesystem_corroboration_report(report)


@pytest.mark.timeout(60)
def test_gc_report_lists_an_episode_merge_entering_core(tmp_path: Path) -> None:
    """Unlabeled, and still entering: the filesystem row splits a gap, so
    deleting it merges two 40-minute gaps into one 80-minute gap, which
    starts a second episode and puts the belief in core on today's rule.
    Pins the docstring's second non-monotone case. Mutation: drop the
    `entering_core` assignment."""
    from aelfrice.cli import default_core_rule
    from aelfrice.doctor import gc_filesystem_corroboration

    store = MemoryStore(str(tmp_path / "merge.db"))
    try:
        bid = "episodemerge0001"
        _mk(store, bid, created_at="2026-08-01T00:00:00+00:00")
        _rows(store, bid, ["2026-08-01T00:40:00+00:00"], source=_FS)
        _rows(store, bid, ["2026-08-01T01:20:00+00:00",
                           "2026-08-01T01:30:00+00:00"])
        assert store.corroboration_episodes()[bid] == 1
        report = gc_filesystem_corroboration(store, qualifies=default_core_rule)
        assert report.entering_core == [bid]
        assert report.leaving_core == []
    finally:
        store.close()
