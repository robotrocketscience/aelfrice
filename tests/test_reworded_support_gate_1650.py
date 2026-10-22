"""The #1650 reworded-support gate: sweep, sample, and score.

Every store is built from synthetic text in a temp directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aelfrice.ingest import _ingest_turn  # pyright: ignore[reportPrivateUsage]
from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SPECULATIVE,
    LOCK_NONE,
    Belief,
)
from aelfrice.store import MemoryStore
from aelfrice.supports_log import ENV_LOG_REWORDED_SUPPORT
from benchmarks import reworded_support_gate_1650 as gate

_TARGET = (
    "Our release checklist covers the changelog. "
    "The scanner reliably detects leaked secrets in committed source code."
)
_NEGATED = "The scanner never detects leaked secrets in source code."
_USER = "The scanner detects leaked secrets in source code."
_SPAN = "The scanner reliably detects leaked secrets in committed source code."


@pytest.fixture(autouse=True)
def pinned_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_DOTDIR", str(tmp_path / "dotdir"))
    monkeypatch.delenv(ENV_LOG_REWORDED_SUPPORT, raising=False)
    work = tmp_path / "proc-cwd"
    work.mkdir()
    monkeypatch.chdir(work)


def _target(
    s: MemoryStore, bid: str, content: str = _TARGET, *,
    session: str = "s1", btype: str = BELIEF_FACTUAL, origin: str = "agent_inferred",
) -> None:
    s.insert_belief(Belief(
        id=bid, content=content, content_hash=f"h_{bid}", alpha=1.0,
        beta=1.0, type=btype, lock_level=LOCK_NONE, locked_at=None,
        created_at="2026-10-01T00:00:00Z", last_retrieved_at=None,
        session_id=session, origin=origin,
    ))


def _typed(
    s: MemoryStore, text: str = _USER, *, session: str = "s2",
    role: str = "user", ts: str = "2026-10-02T00:00:00Z",
) -> None:
    _ingest_turn(s, text, "transcript", session_id=session, created_at=ts, role=role)


def _store(tmp_path: Path, name: str = "src.db") -> tuple[Path, MemoryStore]:
    path = tmp_path / name
    return path, MemoryStore(str(path))


def _sweep(path: Path, tmp_path: Path) -> tuple[list[gate.Pair], list[gate.Pair], int, int]:
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return gate.sweep(path, work)


# --- Sweep -------------------------------------------------------------------


def test_a_match_and_a_rejected_candidate_land_in_their_arms(tmp_path: Path) -> None:
    path, s = _store(tmp_path)
    _target(s, "p1")
    _target(s, "p2", _NEGATED)
    _typed(s)
    s.close()
    matches, rejected, skipped, supporters = _sweep(path, tmp_path)
    assert [p.target_id for p in matches] == ["p1"]
    assert [p.target_id for p in rejected] == ["p2"]
    assert (skipped, supporters) == (0, 1)
    (m,) = matches
    assert (m.arm, m.target_sentence_index, m.matched_span) == (gate.ARM_MATCH, 1, _SPAN)
    assert (rejected[0].arm, rejected[0].matched_span) == (gate.ARM_NULL, None)


def test_a_phantom_target_is_left_out_of_both_arms(tmp_path: Path) -> None:
    path, s = _store(tmp_path)
    _target(s, "p1", btype=BELIEF_SPECULATIVE, origin="speculative")
    _target(s, "p2", _NEGATED, btype=BELIEF_SPECULATIVE, origin="speculative")
    _target(s, "p3")
    _typed(s)
    s.close()
    matches, rejected, skipped, _ = _sweep(path, tmp_path)
    assert [p.target_id for p in matches] == ["p3"]
    assert rejected == []
    assert skipped == 2


def test_only_the_row_that_inserted_a_belief_is_replayed(tmp_path: Path) -> None:
    path, s = _store(tmp_path)
    _typed(s, session="s2")
    # The same sentence in a later session corroborates the first belief.
    _typed(s, session="s3", ts="2026-10-03T00:00:00Z")
    s.close()
    supporters = gate.replay_supporters(path)
    assert len(supporters) == 1
    check = MemoryStore(str(path), read_only=True)
    entry = check.get_ingest_log_entry(supporters[0][1])
    check.close()
    assert entry is not None and entry["session_id"] == "s2"


def test_the_log_only_rules_decide_who_supports_what(tmp_path: Path) -> None:
    path, s = _store(tmp_path)
    _target(s, "p1", session="s2")  # the supporter's own session
    _target(s, "p2", _TARGET + " It runs nightly.", origin="user_transcript")
    _typed(s, session="s2")
    _typed(s, "The scanner flags leaked secrets in source code.", role="assistant")
    s.close()
    matches, rejected, _, supporters = _sweep(path, tmp_path)
    assert supporters == 2
    assert matches == [] and rejected == []


def test_the_sweep_reads_a_read_only_copy_and_leaves_the_store_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, s = _store(tmp_path)
    _target(s, "p1")
    _typed(s)
    s.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    opened: list[tuple[str, bool]] = []
    real_store = gate.MemoryStore

    def spy(db: str, *, read_only: bool = False) -> MemoryStore:
        opened.append((db, read_only))
        return real_store(db, read_only=read_only)

    monkeypatch.setattr(gate, "MemoryStore", spy)
    matches, _, _, _ = _sweep(path, tmp_path)
    assert len(matches) == 1
    assert opened == [(str(tmp_path / "work" / "memory.db"), True)]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    check = MemoryStore(str(path), read_only=True)
    rows = check.count_support_evidence(applied=False)
    check.close()
    assert rows == 0


# --- Sample ------------------------------------------------------------------


def _pairs(arm: str, n: int) -> list[gate.Pair]:
    return sorted((
        gate.Pair(
            id=gate.pair_id(f"{arm}-s{i}", f"t{i}"), arm=arm,
            supporter_id=f"{arm}-s{i}", target_id=f"t{i}", ingest_log_id=f"l{i}",
            user_sentence=f"user {arm} {i}", target_belief=f"belief {i}",
            target_sentence_index=0 if arm == gate.ARM_MATCH else None,
            matched_span=f"belief {i}" if arm == gate.ARM_MATCH else None,
        ) for i in range(n)), key=lambda p: p.id)


def test_the_pair_id_hashes_the_two_belief_ids_only() -> None:
    assert gate.pair_id("a", "b") == hashlib.sha256(b"a\0b").hexdigest()[:16]
    assert gate.pair_id("a", "b") != gate.pair_id("b", "a")


def test_the_sample_is_seeded_capped_and_mixes_the_arms_in_id_order() -> None:
    matches, rejected = _pairs(gate.ARM_MATCH, 50), _pairs(gate.ARM_NULL, 80)
    one = gate.draw_sample(matches, rejected, seed=1650, max_n=10)
    assert one == gate.draw_sample(matches, rejected, seed=1650, max_n=10)
    assert one != gate.draw_sample(matches, rejected, seed=7, max_n=10)
    assert [p.id for p in one] == sorted(p.id for p in one)
    assert sum(p.arm == gate.ARM_MATCH for p in one) == 10
    assert sum(p.arm == gate.ARM_NULL for p in one) == 10
    some = gate.draw_sample(matches, rejected, seed=1, max_n=10, null_n=25)
    assert sum(p.arm == gate.ARM_NULL for p in some) == 25
    # The null arm is capped by the rejected pairs there are.
    few = gate.draw_sample(matches, rejected[:3], seed=1, max_n=100)
    assert (len(few), sum(p.arm == gate.ARM_NULL for p in few)) == (53, 3)


def test_the_packet_hides_the_arm_and_the_matched_span(tmp_path: Path) -> None:
    sample = gate.draw_sample(_pairs(gate.ARM_MATCH, 3), _pairs(gate.ARM_NULL, 3), seed=1)
    gate.write_sample(sample, tmp_path / "out")
    packet = [json.loads(x) for x in (tmp_path / "out" / "packet.jsonl").read_text().splitlines()]
    key = [json.loads(x) for x in (tmp_path / "out" / "key.jsonl").read_text().splitlines()]
    assert all(set(row) == {"id", "user_sentence", "target_belief"} for row in packet)
    assert [row["id"] for row in packet] == [row["id"] for row in key]
    assert {row["arm"] for row in key} == {gate.ARM_MATCH, gate.ARM_NULL}
    assert (tmp_path / "out" / "instructions.txt").read_text() == gate.INSTRUCTIONS


def test_one_store_and_seed_give_byte_identical_files(tmp_path: Path) -> None:
    path, s = _store(tmp_path)
    _target(s, "p1")
    _target(s, "p2", _NEGATED)
    _typed(s)
    s.close()
    for name in ("a", "b"):
        assert gate.main(["sweep", "--store", str(path), "--seed", "1650",
                          "--out", str(tmp_path / name)]) == 0
    files = ("packet.jsonl", "key.jsonl", "instructions.txt")
    assert [(tmp_path / "a" / f).read_bytes() for f in files] == [
        (tmp_path / "b" / f).read_bytes() for f in files]
    assert len((tmp_path / "a" / "packet.jsonl").read_text().splitlines()) == 2


# --- CLI ---------------------------------------------------------------------


def test_the_cli_sweeps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    path, s = _store(tmp_path)
    _target(s, "p1")
    _target(s, "p2", _NEGATED)
    _typed(s)
    s.close()
    out = tmp_path / "out"
    assert gate.main(["sweep", "--store", str(path), "--seed", "1", "--dry-run",
                      "--out", str(out)]) == 0
    assert not out.exists()
    assert "matches=1 rejected=1 phantom_skipped=0" in capsys.readouterr().out
    assert gate.main(["sweep", "--store", str(path), "--seed", "1"]) == 1
    assert gate.main(["sweep", "--store", str(tmp_path / "none.db"), "--seed", "1",
                      "--out", str(out)]) == 1
    assert gate.main(["sweep", "--store", str(path), "--seed", "1", "--out", str(out)]) == 0
    assert len((out / "packet.jsonl").read_text().splitlines()) == 2
