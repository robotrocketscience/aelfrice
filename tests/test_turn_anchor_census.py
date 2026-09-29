"""#1602 AC6 — the producer of the AC1 figures counts what it claims to."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from aelfrice.ingest import ingest_jsonl
from aelfrice.store import MemoryStore

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "turn_anchor_census.py"
)


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's repo-local live store out of every test."""
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))


@pytest.fixture(scope="module")
def census_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "turn_anchor_census", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["turn_anchor_census"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.timeout(30)
def test_census_counts_turns_and_anchor_kinds(
    census_module: ModuleType, tmp_path: Path,
) -> None:
    turns = [
        # Two sentences in one turn: two rows, one turn, two beliefs.
        {"session_id": "s1", "ts": "2026-08-01T00:00:00Z",
         "text": "The configuration file lives at /etc/aelfrice/conf. "
                 "Radio telescopes calibrate against known pulsar timings."},
        # aelfrice's own transcript logger writes `+00:00`, not `Z`.
        {"session_id": "s2", "ts": "2026-08-02T00:00:00+00:00",
         "text": "Astronomers process supernova imagery nightly using "
                 "clusters."},
        # No ts of its own: ingest stamps the clock, the belief keeps the
        # label-only anchor, and the census cannot tell this row from a
        # logged turn -- which is why the count is an upper bound.
        {"session_id": "s2",
         "text": "Glaciers retreat measurably faster in warmer decades."},
    ]
    path = tmp_path / "t.jsonl"
    path.write_text("".join(
        json.dumps({"role": "user", **t}) + "\n" for t in turns
    ))
    db = tmp_path / "census.db"
    store = MemoryStore(str(db))
    ingest_jsonl(store, path)
    store.close()

    assert census_module.census(db) == {
        "transcript_rows": 4,
        "rows_with_turn_identity": 4,
        "distinct_turns": 3,
        "distinct_sessions": 2,
        "beliefs_reachable": 4,
        "rows_with_turn_sha": 3,
        "anchors_by_kind": {"turn": 3, "label_only": 1},
        # The two sentences of one turn share its anchor URI.
        "distinct_doc_uris": 3,
    }


@pytest.mark.timeout(30)
def test_missing_store_exits_2(
    census_module: ModuleType, tmp_path: Path,
) -> None:
    assert census_module.main(["--db", str(tmp_path / "absent.db")]) == 2
