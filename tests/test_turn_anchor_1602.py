"""#1602 — a transcript belief anchors to the turn it came from.

Before this, every transcript anchor was `file:<source label>`, and the
labels in use are a handful of constants, so an anchor named no part of
any transcript. A transcript row whose turn carries both a session id and
its own timestamp now anchors to
`file:<label>#<session_id>/<ts>/<sha256(turn)[:16]>`.

Operator rulings 2026-09-28: hash the whole turn, replace the label-only
anchor rather than add beside it, and no backfill.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.derivation import META_TURN_SHA, DerivationInput, derive
from aelfrice.ingest import ingest_jsonl
from aelfrice.store import MemoryStore

_SESSION = "s-1602"

_TURN_A = "The configuration file lives at /etc/aelfrice/conf."
_TURN_B = "Astronomers process supernova imagery nightly using clusters."
_TURN_C = "Radio telescopes calibrate against known pulsar timings."


@pytest.fixture(autouse=True)
def _pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's repo-local live store out of every test."""
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "pinned.db"))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "anchor.db"))
    yield s
    s.close()


def _write(path: Path, turns: list[dict[str, str]]) -> Path:
    with path.open("w") as f:
        for t in turns:
            f.write(json.dumps({"role": "user", **t}) + "\n")
    return path


def _turn(text: str, ts: str | None) -> dict[str, str]:
    row = {"text": text, "session_id": _SESSION}
    if ts is not None:
        row["ts"] = ts
    return row


def _anchors(store: MemoryStore) -> list[tuple[str, str | None]]:
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT doc_uri, position_hint FROM belief_documents "
        "ORDER BY doc_uri"
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@pytest.mark.timeout(30)
def test_each_turn_gets_its_own_anchor(
    store: MemoryStore, tmp_path: Path,
) -> None:
    ts_a, ts_b = "2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z"
    path = _write(tmp_path / "t.jsonl", [
        _turn(_TURN_A, ts_a), _turn(_TURN_B, ts_b),
    ])
    ingest_jsonl(store, path, source_label="session-end")

    hint_a = f"{_SESSION}/{ts_a}/{_sha(_TURN_A)}"
    hint_b = f"{_SESSION}/{ts_b}/{_sha(_TURN_B)}"
    assert _anchors(store) == sorted([
        (f"file:session-end#{hint_a}", hint_a),
        (f"file:session-end#{hint_b}", hint_b),
    ])


@pytest.mark.timeout(30)
def test_the_hash_covers_the_whole_turn(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Two sentences of one turn share the hash of the turn, not their own."""
    ts = "2026-08-01T00:00:00Z"
    turn = f"{_TURN_A} {_TURN_C}"
    ingest_jsonl(store, _write(tmp_path / "t.jsonl", [_turn(turn, ts)]))

    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT belief_id, position_hint FROM belief_documents"
    ).fetchall()
    assert len({r[0] for r in rows}) == 2
    assert {r[1] for r in rows} == {f"{_SESSION}/{ts}/{_sha(turn)}"}


@pytest.mark.timeout(30)
def test_reingest_writes_no_second_anchor(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """AC4: the same turn re-ingested resolves to the same anchor row."""
    path = _write(tmp_path / "t.jsonl", [
        _turn(_TURN_A, "2026-08-01T00:00:00Z"),
        _turn(_TURN_B, "2026-08-02T00:00:00Z"),
    ])
    ingest_jsonl(store, path)
    first = _anchors(store)
    ingest_jsonl(store, path)
    assert _anchors(store) == first


@pytest.mark.timeout(30)
def test_a_sentence_asserted_in_two_turns_anchors_to_both(
    store: MemoryStore, tmp_path: Path,
) -> None:
    ts_1, ts_2 = "2026-08-01T00:00:00Z", "2026-08-02T00:00:00Z"
    ingest_jsonl(store, _write(tmp_path / "t.jsonl", [
        _turn(_TURN_A, ts_1), _turn(_TURN_A, ts_2),
    ]))
    rows = store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "SELECT belief_id, position_hint FROM belief_documents"
    ).fetchall()
    assert len({r[0] for r in rows}) == 1
    assert {r[1] for r in rows} == {
        f"{_SESSION}/{ts_1}/{_sha(_TURN_A)}",
        f"{_SESSION}/{ts_2}/{_sha(_TURN_A)}",
    }


@pytest.mark.timeout(30)
def test_a_turn_without_its_own_timestamp_keeps_the_label_anchor(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """AC5: a clock-minted ts would change on every re-ingest, so no hint."""
    ingest_jsonl(store, _write(tmp_path / "t.jsonl", [_turn(_TURN_A, None)]))
    assert _anchors(store) == [("file:transcript", None)]


@pytest.mark.timeout(30)
def test_derive_ignores_the_turn_hash() -> None:
    """The hash is the worker's input, never `derive()`'s, so replay
    equality cannot depend on it."""
    base = DerivationInput(
        raw_text=_TURN_A, source_kind="transcript",
        source_path="transcript", raw_meta={"role": "user"},
        session_id=_SESSION, ts="2026-08-01T00:00:00Z",
    )
    with_hash = DerivationInput(
        raw_text=_TURN_A, source_kind="transcript",
        source_path="transcript",
        raw_meta={"role": "user", META_TURN_SHA: _sha(_TURN_A)},
        session_id=_SESSION, ts="2026-08-01T00:00:00Z",
    )
    assert derive(base) == derive(with_hash)
