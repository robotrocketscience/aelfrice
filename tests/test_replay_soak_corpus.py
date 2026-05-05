"""Schema + replay-equality validation for the public replay-soak corpus (#403 A).

Two independent assertions:

1. Every JSONL row under `tests/corpus/replay_soak/v0.1/` parses and
   conforms to the README schema.
2. The full corpus replays through `aelfrice.derivation.derive` +
   `MemoryStore.record_ingest` + `replay_full_equality` with
   `mismatched + derived_orphan == 0`. This is the v0.1 soak gate's
   per-run invariant; the scheduled workflow runs the same code path
   daily and appends the result to `.replay-soak-status.json`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from aelfrice.models import INGEST_SOURCE_KINDS, INGEST_SOURCE_LEGACY_UNKNOWN
from aelfrice.store import MemoryStore

from tests.replay_soak_runner import REPLAY_SOAK_CORPUS_ROOT, run_replay_soak

# `legacy_unknown` is excluded from the corpus by design — `derive`
# does not have a re-derivation path for it.
_ALLOWED_KINDS = INGEST_SOURCE_KINDS - {INGEST_SOURCE_LEGACY_UNKNOWN}

_REQUIRED_FIELDS = ("id", "source_kind", "source_path", "raw_text",
                    "raw_meta", "provenance", "note")


def _iter_corpus_files() -> Iterator[Path]:
    yield from sorted(REPLAY_SOAK_CORPUS_ROOT.glob("*.jsonl"))


def test_corpus_root_has_readme() -> None:
    """The schema contract README must live alongside the corpus dir."""
    readme = REPLAY_SOAK_CORPUS_ROOT.parent / "README.md"
    assert readme.is_file(), (
        f"{readme} is required — it documents the row schema and "
        "authoring discipline."
    )


def test_corpus_covers_every_kind() -> None:
    """One JSONL file per non-legacy source kind must exist at v0.1."""
    present = {p.stem.replace("_v0_1", "") for p in _iter_corpus_files()}
    missing = _ALLOWED_KINDS - present
    assert not missing, (
        f"v0.1 corpus is missing files for source kinds: {sorted(missing)}. "
        f"Expected one `<kind>_v0_1.jsonl` per kind in {_ALLOWED_KINDS}."
    )


@pytest.mark.parametrize("path", list(_iter_corpus_files()),
                         ids=lambda p: p.name)
def test_corpus_file_rows_valid(path: Path) -> None:
    """Every row in this file conforms to the README schema."""
    expected_kind = path.stem.replace("_v0_1", "")
    seen_ids: set[str] = set()

    with path.open() as f:
        rows = [
            (lineno, line.strip())
            for lineno, line in enumerate(f, 1)
            if line.strip()
        ]

    assert rows, f"{path.name}: corpus file is empty"

    for lineno, raw in rows:
        where = f"{path.name}:{lineno}"
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{where}: invalid JSON — {exc}")

        assert isinstance(row, dict), f"{where}: row must be JSON object"

        for field in _REQUIRED_FIELDS:
            assert field in row, f"{where}: missing required field {field!r}"

        for field in ("id", "source_kind", "raw_text", "provenance", "note"):
            val = row[field]
            assert isinstance(val, str) and val, (
                f"{where}: field {field!r} must be non-empty string"
            )

        # `source_path` and `raw_meta` are nullable.
        sp = row["source_path"]
        assert sp is None or (isinstance(sp, str) and sp), (
            f"{where}: source_path must be non-empty string or null"
        )
        rm = row["raw_meta"]
        assert rm is None or isinstance(rm, dict), (
            f"{where}: raw_meta must be object or null"
        )

        assert row["source_kind"] == expected_kind, (
            f"{where}: source_kind {row['source_kind']!r} does not match "
            f"file's kind {expected_kind!r}"
        )

        assert row["id"] not in seen_ids, (
            f"{where}: duplicate id {row['id']!r} within {path.name}"
        )
        seen_ids.add(row["id"])


def test_corpus_replays_with_zero_drift(tmp_path: Path) -> None:
    """The v0.1 soak invariant: full corpus replay reports no drift.

    `mismatched + derived_orphan == 0` is the soak-gate per-run check.
    Anything else means deterministic-derivation has broken under the
    public fixture and the gate must fail loudly.
    """
    db_path = tmp_path / "replay_soak.db"
    store = MemoryStore(str(db_path))
    try:
        result = run_replay_soak(store)
    finally:
        store.close()

    assert result.has_drift is False, (
        f"replay-soak drift on public fixture: "
        f"mismatched={result.mismatched}, "
        f"derived_orphan={result.derived_orphan}. "
        f"drift_examples={result.drift_examples}"
    )
    assert result.total_log_rows >= 60, (
        f"v0.1 floor is 60 rows; harness loaded {result.total_log_rows}"
    )
