"""The #1804 grader calibration: split, cut fitting, choice on half A, and test on half B.

Every key, packet, and label is synthetic text built in a temp directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from benchmarks import grader_calibration_1804 as cal
from benchmarks.real_store_ab_1795 import InputError

_ARMS_RUN = ["A0", "A1"]


def _rid(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def _source(
    root: Path, snap: str, prompts: Mapping[str, int], *, nulls: int = 1,
) -> tuple[Path, Path]:
    """A session-1 key and packet directory: `prompts` maps prompt id to judged rows."""
    root.mkdir(parents=True, exist_ok=True)
    key: list[dict[str, Any]] = [{
        "kind": "store", "snapshot": snap, "prompts_sampled": len(prompts) + 1,
        "arms_run": _ARMS_RUN,
    }]
    packet: list[dict[str, str]] = []
    for pid, n in prompts.items():
        rows = [(str(k), {"A0": k + 1}) for k in range(n)]
        rows += [(f"null{k}", None) for k in range(nulls)]
        for tag, arms in rows:
            rid = _rid(snap, pid, tag)
            key.append({"kind": "item", "id": rid, "prompt_id": pid, "prompt": f"ask {pid}",
                        "belief_id": f"b{tag}", "arms": arms, "shares_prompt_stem": False,
                        "snapshot": snap})
            packet.append({"id": rid, "prompt": f"ask {pid}", "belief": f"note {tag}"})
    key_path = root / f"{snap}-key.jsonl"
    key_path.write_text("".join(json.dumps(r) + "\n" for r in key), encoding="utf-8")
    pdir = root / f"{snap}-packet"
    pdir.mkdir()
    (pdir / "packet.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in packet), encoding="utf-8")
    return key_path, pdir


def _stores(root: Path, n: int = 6) -> tuple[list[Path], list[Path]]:
    """`n` stores, each with prompts of 1 to 4 judged rows plus one null row.

    The prompt ids are written out of sorted order.
    """
    keys: list[Path] = []
    packets: list[Path] = []
    for s in range(n):
        prompts = {f"p{s}{(2 * j) % 5}": 1 + (s + j) % 4 for j in range(5)}
        k, p = _source(root / "s1", f"snap{s}", prompts)
        keys.append(k)
        packets.append(p)
    return keys, packets


def _split(keys: Sequence[Path], packets: Sequence[Path], out: Path, *,
           seed: int = cal.DEFAULT_SEED, min_items: int = 8, stores: int | None = None) -> int:
    """Run `split`; `stores` defaults to the count of stores with item rows in `keys`."""
    if stores is None:
        stores = len({r["snapshot"] for r in _source_items(keys)})
    return cal.main(["split", "--keys", *map(str, keys), "--packets", *map(str, packets),
                     "--seed", str(seed), "--min-items", str(min_items),
                     "--stores", str(stores), "--out", str(out)])


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _source_items(keys: Sequence[Path]) -> list[dict[str, Any]]:
    return [r for k in keys for r in _read(k) if r["kind"] == "item"]


# --- Split ----------------------------------------------------------------------


def test_the_split_defaults_are_the_pre_registered_ones() -> None:
    assert cal.DEFAULT_SEED == 1804
    assert cal.DEFAULT_MIN_ITEMS == 2000
    assert cal.DEFAULT_STORES == 17
    half_a, half_b = cal.split_stores([f"snap{s}" for s in range(17)], cal.DEFAULT_SEED)
    assert (len(half_a), len(half_b)) == (8, 9)


@pytest.mark.parametrize("stores", [5, 7, None])
def test_a_store_count_other_than_stores_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], stores: int | None,
) -> None:
    # Six stores hold item rows. None leaves --stores at its default, 17.
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    argv = ["split", "--keys", *map(str, keys), "--packets", *map(str, packets),
            "--min-items", "8", "--out", str(out)]
    if stores is not None:
        argv += ["--stores", str(stores)]
    assert cal.main(argv) == 1
    assert f"hold 6 stores with item rows, not --stores {stores or 17}" in capsys.readouterr().err
    assert not (out / "key.jsonl").exists()
    assert cal.main([*argv, "--stores", "6"]) == 0


def test_the_split_is_reproducible_byte_for_byte(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    assert _split(keys, packets, tmp_path / "one") == 0
    assert _split(keys, packets, tmp_path / "two") == 0
    for rel in ("key.jsonl", "A/packet.jsonl", "B/packet.jsonl", "A/instructions.txt"):
        assert (tmp_path / "one" / rel).read_bytes() == (tmp_path / "two" / rel).read_bytes()


def test_the_stores_split_in_snapshot_order_shuffled_with_the_seed(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path, n=7)
    out = tmp_path / "out"
    assert _split(keys, packets, out, min_items=1) == 0
    order = sorted(f"snap{s}" for s in range(7))
    random.Random(cal.DEFAULT_SEED).shuffle(order)
    rows = _read(out / "key.jsonl")
    half_a = {r["snapshot"] for r in rows if r["half"] == "A"}
    half_b = {r["snapshot"] for r in rows if r["half"] == "B"}
    # With one row enough, each half takes one prompt, so it names one store.
    assert half_a <= set(order[:3])
    assert half_b <= set(order[3:])
    assert cal.split_stores([f"snap{s}" for s in reversed(range(7))], cal.DEFAULT_SEED) == (
        order[:3], order[3:])


def test_the_halves_share_no_store(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    assert _split(keys, packets, out, min_items=10) == 0
    rows = _read(out / "key.jsonl")
    a = {r["snapshot"] for r in rows if r["half"] == "A"}
    b = {r["snapshot"] for r in rows if r["half"] == "B"}
    assert a and b and not a & b


def test_the_split_depends_on_the_seed() -> None:
    snaps = [f"snap{s}" for s in range(6)]
    halves = {seed: cal.split_stores(snaps, seed) for seed in (1804, 1805)}
    assert halves[1804] != halves[1805]
    assert all(len(a) == 3 and len(b) == 3 for a, b in halves.values())


def test_a_store_row_without_item_rows_joins_neither_half(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path, n=2)
    k, p = _source(tmp_path / "s1", "empty", {})
    out = tmp_path / "out"
    assert _split([*keys, k], [*packets, p], out, min_items=1) == 0
    rows = _read(out / "key.jsonl")
    # Counted, it would make three stores, and seed 1804 would give it half A
    # alone, which holds no row.
    assert cal.split_stores(["empty", "snap0", "snap1"], cal.DEFAULT_SEED)[0] == ["empty"]
    assert {r["snapshot"] for r in rows if r["half"] == "A"} == {"snap0"}
    assert {r["snapshot"] for r in rows if r["half"] == "B"} == {"snap1"}
    assert _split([*keys, k], [*packets, p], tmp_path / "three", min_items=1, stores=3) == 1


def test_fewer_than_two_stores_with_rows_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    keys, packets = _stores(tmp_path, n=1)
    assert _split(keys, packets, tmp_path / "out", min_items=1, stores=1) == 1
    assert "--stores must be at least 2" in capsys.readouterr().err


def test_each_half_takes_whole_prompts_in_seeded_order_until_min_items(
    tmp_path: Path,
) -> None:
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    min_items = 9
    assert _split(keys, packets, out, min_items=min_items) == 0
    source = _source_items(keys)
    rows = _read(out / "key.jsonl")
    snaps = sorted({r["snapshot"] for r in source})
    for half, members in zip("AB", cal.split_stores(snaps, cal.DEFAULT_SEED), strict=True):
        prompts = sorted({(r["prompt_id"], r["snapshot"]) for r in source
                          if r["snapshot"] in members})
        random.Random(cal.DEFAULT_SEED).shuffle(prompts)
        taken: list[tuple[str, str]] = []
        count = 0
        for p in prompts:
            if count >= min_items:
                break
            taken.append(p)
            count += sum(1 for r in source if (r["prompt_id"], r["snapshot"]) == p)
        mine = [r for r in rows if r["half"] == half]
        assert {(r["prompt_id"], r["snapshot"]) for r in mine} == set(taken)
        # Whole prompts: every source row of a taken prompt is in the half.
        assert len(mine) == count
        assert len(mine) >= min_items
        # Minimal: without the last prompt the half falls short.
        last = sum(1 for r in source if (r["prompt_id"], r["snapshot"]) == taken[-1])
        assert len(mine) - last < min_items


def test_a_prompt_id_two_stores_share_is_two_prompts(tmp_path: Path) -> None:
    keys: list[Path] = []
    packets: list[Path] = []
    for s in range(4):
        k, p = _source(tmp_path / "s1", f"snap{s}", {"x": 2, "y": 2})
        keys.append(k)
        packets.append(p)
    out = tmp_path / "out"
    assert _split(keys, packets, out, min_items=3) == 0
    for half in "AB":
        mine = [r for r in _read(out / "key.jsonl") if r["half"] == half]
        # Each prompt holds three rows, so three rows is exactly one prompt.
        assert len(mine) == 3
        assert len({r["snapshot"] for r in mine}) == 1


def test_a_half_short_of_min_items_is_refused(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    assert _split(keys, packets, out, min_items=10_000) == 1
    assert not (out / "key.jsonl").exists()


@pytest.mark.parametrize("min_items", [0, -5])
def test_a_min_items_under_one_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], min_items: int,
) -> None:
    # Every half holds at least zero rows, so without the refusal both halves
    # would take no prompt and the split would write empty packets.
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    assert _split(keys, packets, out, min_items=min_items) == 1
    assert "--min-items must be at least 1" in capsys.readouterr().err
    assert not (out / "key.jsonl").exists()


def test_the_packets_carry_only_id_prompt_and_belief(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    assert _split(keys, packets, out) == 0
    key = {r["id"]: r for r in _read(out / "key.jsonl")}
    for half in "AB":
        text = (out / half / "packet.jsonl").read_text(encoding="utf-8")
        assert "snap" not in text and "null_arm" not in text and "A0" not in text
        rows = _read(out / half / "packet.jsonl")
        assert all(set(r) == {"id", "prompt", "belief"} for r in rows)
        assert sorted(r["id"] for r in rows) == sorted(i for i, r in key.items()
                                                       if r["half"] == half)
        # Shuffled with the seed: neither id order nor the source order.
        ordered = sorted(rows, key=lambda r: r["id"])
        random.Random(cal.DEFAULT_SEED).shuffle(ordered)
        assert rows == ordered
        assert sorted(os.listdir(out / half)) == ["instructions.txt", "packet.jsonl"]


def test_the_packet_rows_keep_the_session_1_ids_and_text(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    assert _split(keys, packets, out) == 0
    source = {r["id"]: r for p in packets for r in _read(p / "packet.jsonl")}
    for half in "AB":
        for r in _read(out / half / "packet.jsonl"):
            assert r == source[r["id"]]


def test_the_key_records_half_store_prompt_and_null_arm(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    assert _split(keys, packets, out) == 0
    source = {r["id"]: r for r in _source_items(keys)}
    rows = _read(out / "key.jsonl")
    assert {r["half"] for r in rows} == {"A", "B"}
    for r in rows:
        assert set(r) == {"id", "half", "snapshot", "prompt_id", "null_arm"}
        s = source[r["id"]]
        assert (r["snapshot"], r["prompt_id"]) == (s["snapshot"], s["prompt_id"])
        assert r["null_arm"] is (s["arms"] is None)
    assert any(r["null_arm"] for r in rows) and not all(r["null_arm"] for r in rows)


def test_the_instructions_state_the_scale_table_verbatim(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    assert _split(keys, packets, out) == 0
    table = {
        "3": "Needed: without the note, the answer would be wrong or would miss a rule,"
             " a preference, or an earlier decision about this exact task.",
        "2": "Would help: reading the note would change or directly inform what the"
             " assistant does for this prompt.",
        "1": "Same topic only: the note is about the same project, file, tool, or topic,"
             " but wouldn't change the answer.",
        "0": "Unrelated, or it only shares words with the prompt.",
        "null": "You can't tell what the prompt asks or what the note means.",
    }
    for half in "AB":
        lines = (out / half / "instructions.txt").read_text(encoding="utf-8").splitlines()
        for score, meaning in table.items():
            assert f"  {score:<5} {meaning}" in lines
        assert '  {"id": "<row id>", "score": 2}' in lines


def test_keys_pair_with_packets_by_row_ids_in_any_order(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    assert _split(keys, packets, tmp_path / "one") == 0
    assert _split(keys, list(reversed(packets)), tmp_path / "two") == 0
    assert (tmp_path / "one" / "key.jsonl").read_bytes() == (
        tmp_path / "two" / "key.jsonl").read_bytes()


def test_a_key_without_its_packet_is_refused(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    assert _split(keys, packets[:-1], tmp_path / "out") == 1
    assert _split(keys[:-1], packets, tmp_path / "out") == 1
    assert not (tmp_path / "out" / "key.jsonl").exists()


def test_a_packet_with_one_extra_row_is_refused(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    with (packets[0] / "packet.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "extra", "prompt": "p", "belief": "b"}) + "\n")
    assert _split(keys, packets, tmp_path / "out") == 1


def test_the_same_key_twice_is_refused(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    assert _split([*keys, keys[0]], [*packets, packets[0]], tmp_path / "out") == 1


def test_two_keys_with_one_snapshot_are_refused(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    k, p = _source(tmp_path / "again", "snap0", {"other": 2})
    rows = _read(k)
    # A sample size that covers both keys' prompts, so only the repeat is wrong.
    rows[0]["prompts_sampled"] = 100
    k.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert _split([*keys, k], [*packets, p], tmp_path / "out") == 1


def test_two_keys_with_one_item_id_are_refused(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    k, p = _source(tmp_path / "again", "snap9", {"other": 1}, nulls=0)
    item = _read(keys[0])[1]
    k.write_text("".join(json.dumps(r) + "\n" for r in [
        _read(k)[0], {**item, "snapshot": "snap9"}]), encoding="utf-8")
    (p / "packet.jsonl").write_text(json.dumps(
        {"id": item["id"], "prompt": "p", "belief": "b"}) + "\n", encoding="utf-8")
    assert _split([*keys, k], [*packets, p], tmp_path / "out") == 1


def test_an_item_of_a_store_the_key_lacks_is_refused(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    rows = _read(keys[0])
    rows[1]["snapshot"] = "elsewhere"
    keys[0].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert _split(keys, packets, tmp_path / "out") == 1


def test_a_second_store_row_in_one_key_is_refused(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    rows = _read(keys[0])
    keys[0].write_text("".join(json.dumps(r) + "\n" for r in [*rows, rows[0]]),
                       encoding="utf-8")
    assert _split(keys, packets, tmp_path / "out") == 1


def test_a_key_without_a_store_row_is_refused(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    # An empty key and an empty packet pair with each other and add no row,
    # so only the missing store row is wrong.
    empty_key = tmp_path / "empty-key.jsonl"
    empty_key.write_text("", encoding="utf-8")
    empty_packet = tmp_path / "empty-packet"
    empty_packet.mkdir()
    (empty_packet / "packet.jsonl").write_text("", encoding="utf-8")
    assert _split([*keys, empty_key], [*packets, empty_packet], tmp_path / "out") == 1
    assert not (tmp_path / "out" / "key.jsonl").exists()


def test_a_repeated_row_in_a_source_packet_is_refused(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    path = packets[0] / "packet.jsonl"
    rows = _read(path)
    path.write_text("".join(json.dumps(r) + "\n" for r in [*rows, rows[0]]),
                    encoding="utf-8")
    assert _split(keys, packets, tmp_path / "out") == 1


@pytest.mark.parametrize("name", ["A/packet.jsonl", "B/instructions.txt", "key.jsonl"])
@pytest.mark.parametrize("link", ["symlink", "hard link"])
def test_a_link_at_a_split_file_name_is_refused_before_any_write(
    tmp_path: Path, name: str, link: str,
) -> None:
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    (out / "A").mkdir(parents=True)
    (out / "B").mkdir()
    other = tmp_path / "other.txt"
    other.write_text("old\n", encoding="utf-8")
    if link == "symlink":
        (out / name).symlink_to(other)
    else:
        os.link(other, out / name)
    assert _split(keys, packets, out) == 1
    assert other.read_text(encoding="utf-8") == "old\n"
    written = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert written == [name]


def test_a_half_directory_linked_to_the_output_is_refused(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "A").symlink_to(out)
    assert _split(keys, packets, out) == 1
    assert not (out / "key.jsonl").exists()


def test_half_directories_linked_together_are_refused(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    (out / "A").mkdir(parents=True)
    (out / "B").symlink_to(out / "A")
    assert _split(keys, packets, out) == 1
    assert not (out / "A" / "packet.jsonl").exists()


def test_plain_split_files_are_rewritten(tmp_path: Path) -> None:
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    (out / "A").mkdir(parents=True)
    (out / "A" / "packet.jsonl").write_text("stale\n", encoding="utf-8")
    assert _split(keys, packets, out) == 0
    assert "stale" not in (out / "A" / "packet.jsonl").read_text(encoding="utf-8")


# --- Score ----------------------------------------------------------------------

Row = tuple[str, bool, int | None, int | None, int | None]
"""`(half, null_arm, grader a's score, grader b's, grader c's)`."""


def _case(
    rows: Sequence[Row],
) -> tuple[list[cal.CalibrationItem], dict[str, dict[str, int | None]]]:
    items: list[cal.CalibrationItem] = []
    scores: dict[str, dict[str, int | None]] = {g: {} for g in cal.GRADERS}
    for n, (half, null_arm, *graded) in enumerate(rows):
        rid = f"r{n}"
        items.append(cal.CalibrationItem(id=rid, half=half, snapshot=f"s{half}",
                                         prompt_id=f"p{n}", null_arm=null_arm))
        for g, s in zip(cal.GRADERS, graded, strict=True):
            scores[g][rid] = s
    return items, scores


def _pattern(half: str, a: str, b: str, c: str, *, null: str = "0000000111") -> list[Row]:
    """Rows from 0/1 strings, 1 scored 3 and 0 scored 0; `null` marks null-arm rows."""
    return [(half, null[k] == "1", 3 * int(a[k]), 3 * int(b[k]), 3 * int(c[k]))
            for k in range(len(a))]


def _combo(report: Mapping[str, Any], rule: str, pair: str) -> dict[str, Any]:
    return next(c for c in report["combinations"]
                if c["rule"] == rule and "".join(c["pair"]) == pair)


# Pairwise kappas on these rows, by hand from the 2x2 tables of ten rows:
#   b, c: b says yes 4, c 5, 9 agree. p_o = 0.9, p_e = 0.4*0.5 + 0.6*0.5 = 0.5,
#         kappa = 0.4 / 0.5 = 0.8.
#   a, c: a says yes 7, c 5, 8 agree. p_o = 0.8, p_e = 0.5, kappa = 0.6.
#   a, b: 7 agree. p_o = 0.7, p_e = 0.7*0.4 + 0.3*0.6 = 0.46,
#         kappa = 0.24 / 0.54 = 4/9.
_ORDER_A = _pattern("A", "1111111000", "1111000000", "1111100000")


def test_the_bars_are_the_pre_registered_values() -> None:
    assert cal.KAPPA_BAR == 0.60
    assert cal.NULL_RATE_CEILING == 0.20
    assert cal.FIXED_CUT == 2
    assert cal.CUTS == (1, 2, 3)
    assert cal.CUT_TIE_TARGET == (2, 2)
    assert cal.RULES == ("fixed", "per_grader")
    assert cal.PAIRS[0] == ("a", "b")


def test_kappa_matches_the_hand_computed_values() -> None:
    items, scores = _case([*_ORDER_A, *_pattern("B", "1111111000", "1111000000", "1111100000")])
    report = cal.score(items, scores)
    for half in "AB":
        assert math.isclose(_combo(report, "fixed", "bc")["halves"][half]["kappa"], 0.8)
        assert math.isclose(_combo(report, "fixed", "ac")["halves"][half]["kappa"], 0.6)
        assert math.isclose(_combo(report, "fixed", "ab")["halves"][half]["kappa"], 4 / 9)
    ab = _combo(report, "fixed", "ab")["halves"]["A"]
    assert (ab["n"], ab["agreed"], ab["raw_agreement"]) == (10, 7, 0.7)
    assert ab["disagreements"] == {"a_only": 3, "b_only": 0,
                                   "a_only_share": 1.0, "b_only_share": 0.0}
    assert ab["relevant"]["a"] == {"graded": 10, "relevant": 7, "rate": 0.7}


def test_kappa_binarizes_each_score_at_its_cut() -> None:
    # a scores 2 where b scores 3: at cut 2 both say relevant, at cut 3 only b.
    rows: list[Row] = [("A", False, 2, 3, 0), ("A", False, 2, 3, 0),
                       ("A", False, 0, 0, 0), ("A", True, 0, 0, 0)]
    items, scores = _case(rows)
    fig = cal.half_result(items, "A", ("a", "b"), (2, 2), scores).figures
    assert fig["kappa"] == 1.0
    fig = cal.half_result(items, "A", ("a", "b"), (3, 3), scores).figures
    assert fig["kappa"] == 0.0
    assert fig["disagreements"]["b_only"] == 2


def test_kappa_leaves_out_a_row_either_grader_scored_null() -> None:
    rows: list[Row] = [("A", False, 3, 3, 0), ("A", False, 0, 0, 0),
                       ("A", False, None, 3, 0), ("A", False, 3, None, 0)]
    items, scores = _case(rows)
    fig = cal.half_result(items, "A", ("a", "b"), (2, 2), scores).figures
    assert (fig["n"], fig["kappa"]) == (2, 1.0)


def test_raw_agreement_counts_the_rows_both_graders_scored() -> None:
    # Five rows, two with a null score: a and b agree on 2 of the 3 rows
    # both scored, so 2/3, the rows kappa reads. Over all five it would be 2/5.
    rows: list[Row] = [("A", False, 3, 3, 0), ("A", False, 0, 0, 0), ("A", False, 3, 0, 0),
                       ("A", False, None, 3, 0), ("A", True, 0, None, 0)]
    items, scores = _case(rows)
    fig = cal.half_result(items, "A", ("a", "b"), (2, 2), scores).figures
    assert (fig["rows"], fig["n"], fig["agreed"]) == (5, 3, 2)
    assert math.isclose(fig["raw_agreement"], 2 / 3)


def test_weighted_kappa_matches_a_hand_computed_value() -> None:
    # Scores (0,0) (1,2) (2,2) (3,3): the squared differences sum to 1, so the
    # observed mean is 1/4. Marginals x {0,1,2,3}: 1 each, y {0: 1, 2: 2, 3: 1};
    # the expected sum of count products times squared differences is
    # 17 + 7 + 5 + 11 = 40 over 16 pairs, 2.5. Kappa = 1 - 0.25 / 2.5 = 0.9.
    assert math.isclose(cal.weighted_kappa([(0, 0), (1, 2), (2, 2), (3, 3)]), 0.9)
    # (0,2) (1,1) (2,0) (3,3): squared differences 4 + 0 + 4 + 0 over 4 pairs,
    # 2. Both marginals put one pair on each score, so the expected mean is
    # 40 / 16 = 2.5. Kappa = 1 - 2 / 2.5 = 0.2.
    assert math.isclose(cal.weighted_kappa([(0, 2), (1, 1), (2, 0), (3, 3)]), 0.2)
    assert cal.weighted_kappa([(1, 1), (2, 2)]) == 1.0
    assert cal.weighted_kappa([]) == 0.0
    # A miss by three costs nine times a miss by one.
    near = cal.weighted_kappa([(0, 0), (3, 3), (0, 1), (3, 3)])
    far = cal.weighted_kappa([(0, 0), (3, 3), (0, 3), (3, 3)])
    assert far < near


def test_the_report_carries_the_weighted_kappa_of_the_raw_scores() -> None:
    rows: list[Row] = [("A", False, 0, 0, 0), ("A", False, 1, 2, 0), ("A", False, 2, 2, 0),
                       ("A", True, 3, 3, 0), ("A", False, None, 1, 0)]
    items, scores = _case(rows)
    fig = cal.half_result(items, "A", ("a", "b"), (2, 2), scores).figures
    assert math.isclose(fig["weighted_kappa"], 0.9)


def test_the_null_arm_rate_is_the_share_of_non_null_null_arm_scores_at_the_cut() -> None:
    # Null-arm rows: a scores 3, 0, 1, null, 2. Four graded, two at or above 2.
    rows: list[Row] = [("A", True, 3, 0, 0), ("A", True, 0, 0, 0), ("A", True, 1, 0, 0),
                       ("A", True, None, 0, 0), ("A", True, 2, 0, 0), ("A", False, 3, 3, 0)]
    items, scores = _case(rows)
    fig = cal.half_result(items, "A", ("a", "b"), (2, 2), scores).figures
    assert fig["null_arm"]["a"] == {"graded": 4, "called_relevant": 2, "rate": 0.5}
    assert fig["null_arm"]["b"] == {"graded": 5, "called_relevant": 0, "rate": 0.0}
    assert fig["null_rows"] == 5
    fig = cal.half_result(items, "A", ("a", "b"), (3, 2), scores).figures
    assert fig["null_arm"]["a"]["rate"] == 0.25


def test_a_null_arm_rate_on_the_ceiling_meets_it_and_one_row_more_does_not() -> None:
    perfect = [("A", False, 3, 3, 3)] * 5 + [("A", False, 0, 0, 0)] * 5
    on_bar: list[Row] = [*perfect, ("A", True, 3, 3, 0), *[("A", True, 0, 0, 0)] * 4]
    items, scores = _case(on_bar)
    assert cal.half_result(items, "A", ("a", "b"), (2, 2), scores).meets_bars
    over: list[Row] = [*perfect, ("A", True, 3, 3, 0), *[("A", True, 0, 0, 0)] * 3]
    items, scores = _case(over)
    assert not cal.half_result(items, "A", ("a", "b"), (2, 2), scores).meets_bars


def test_the_null_arm_ceiling_divides_by_the_graded_null_arm_rows() -> None:
    # a and b score the five null-arm rows 3, 0, 0, null, null: one of three
    # graded rows is 1/3, over the ceiling. Over all five rows it would be
    # 1/5, on the ceiling.
    perfect = [("A", False, 3, 3, 3)] * 5 + [("A", False, 0, 0, 0)] * 5
    rows: list[Row] = [*perfect, ("A", True, 3, 3, 0), *[("A", True, 0, 0, 0)] * 2,
                       *[("A", True, None, None, 0)] * 2]
    items, scores = _case(rows)
    result = cal.half_result(items, "A", ("a", "b"), (2, 2), scores)
    assert result.kappa == 1
    assert not result.meets_bars


def test_each_half_reads_only_its_own_null_arm_rows() -> None:
    # Both halves agree on every row. Half A's five null-arm rows score 0;
    # half B scores 3 on two of its five, a rate of 2/5 that fails the
    # ceiling. Pooled over both halves it would be 2/10, on the ceiling.
    perfect = [(3, 3, 3)] * 5 + [(0, 0, 0)] * 5
    rows: list[Row] = [
        *[("A", False, *s) for s in perfect], *[("A", True, 0, 0, 0)] * 5,
        *[("B", False, *s) for s in perfect], *[("B", True, 3, 3, 3)] * 2,
        *[("B", True, 0, 0, 0)] * 3,
    ]
    items, scores = _case(rows)
    report = cal.score(items, scores)
    ab = _combo(report, "fixed", "ab")["halves"]
    assert ab["A"]["null_arm"]["a"] == {"graded": 5, "called_relevant": 0, "rate": 0.0}
    assert ab["B"]["null_arm"]["a"] == {"graded": 5, "called_relevant": 2, "rate": 0.4}
    assert report["chosen"] == {"rule": "fixed", "pair": ["a", "b"], "cuts": {"a": 2, "b": 2}}
    assert report["verdict"] == "fail"


def test_no_graded_null_arm_row_fails_the_ceiling() -> None:
    rows: list[Row] = [("A", False, 3, 3, 0), ("A", False, 0, 0, 0), ("A", True, None, 0, 0)]
    items, scores = _case(rows)
    assert not cal.half_result(items, "A", ("a", "b"), (2, 2), scores).meets_bars


def test_a_kappa_on_the_bar_qualifies_and_one_under_does_not() -> None:
    items, scores = _case(_ORDER_A)
    assert cal.half_result(items, "A", ("a", "c"), (2, 2), scores).meets_bars
    assert not cal.half_result(items, "A", ("a", "b"), (2, 2), scores).meets_bars


def test_an_exact_kappa_on_the_bar_qualifies_where_the_float_falls_under() -> None:
    # Both say yes on 2 rows, each alone on 1, both no on 14: p_o = 16/18,
    # p_e = (3*3 + 15*15) / 18**2 = 13/18, kappa = (3/18) / (5/18) = 3/5.
    # In floating point the same table gives just under 0.6.
    rows: list[Row] = [*[("A", False, 3, 3, 0)] * 2, ("A", False, 3, 0, 0),
                       ("A", False, 0, 3, 0), *[("A", False, 0, 0, 0)] * 9,
                       *[("A", True, 0, 0, 0)] * 5]
    items, scores = _case(rows)
    result = cal.half_result(items, "A", ("a", "b"), (2, 2), scores)
    assert result.kappa == Fraction(3, 5)
    assert result.figures["kappa"] < cal.KAPPA_BAR
    assert result.meets_bars


def test_fit_cuts_takes_the_least_rate_difference() -> None:
    # a's rates at cuts 1, 2, 3: 1/2, 1/2, 1/2. b's: 1/2, 0, 0.
    ids = ["0", "1"]
    assert cal.fit_cuts(ids, {"0": 3, "1": 0}, {"0": 1, "1": 0}) == (2, 1)


def test_fit_cuts_breaks_a_tie_toward_two_two() -> None:
    # a's rates: 3/4, 1/2, 1/4. b's: 1/4 at every cut. Cut 3 for a matches
    # any cut for b, and (3, 2) is the nearest of those to (2, 2).
    ids = ["0", "1", "2", "3"]
    a = {"0": 0, "1": 1, "2": 2, "3": 3}
    b = {"0": 0, "1": 0, "2": 0, "3": 3}
    assert cal.fit_cuts(ids, a, b) == (3, 2)
    assert cal.fit_cuts(ids, a, a) == (2, 2)


def test_fit_cuts_breaks_a_remaining_tie_lexicographically() -> None:
    # a's rates: 1, 0, 0. b's: 1, 1, 0. (1, 1), (1, 2), (2, 3), and (3, 3)
    # match; (1, 2) and (2, 3) are both one step from (2, 2).
    ids = ["0", "1"]
    assert cal.fit_cuts(ids, {"0": 1, "1": 1}, {"0": 2, "1": 2}) == (1, 2)


def test_fit_cuts_skip_null_scores() -> None:
    # a's null row would halve its rate if it counted as not relevant.
    ids = ["0", "1", "2"]
    assert cal.fit_cuts(ids, {"0": 3, "1": 0, "2": None}, {"0": 3, "1": 0, "2": 0}) == (2, 2)
    assert cal.relevant_rate(ids, {"0": 3, "1": 0, "2": None}, 2) == Fraction(1, 2)


def test_the_cuts_are_fitted_on_half_a_only() -> None:
    # On A, a's rate is 1/2 at every cut and b's is 1/2 at cut 1, so (2, 1).
    # Over both halves a's is 2/5 and b's 3/5, 1/5, 1/5, which ties every
    # cut pair at 1/5 and would fit (2, 2).
    rows: list[Row] = [("A", False, 3, 1, 0), ("A", True, 0, 0, 0),
                       ("B", False, 3, 3, 0), ("B", False, 0, 1, 0), ("B", True, 0, 0, 0)]
    items, scores = _case(rows)
    report = cal.score(items, scores)
    assert _combo(report, "per_grader", "ab")["cuts"] == {"a": 2, "b": 1}


def test_the_a_b_pair_comes_first_then_the_others_by_descending_kappa() -> None:
    items, scores = _case([*_ORDER_A, *_pattern("B", "1111111000", "1111000000", "1111100000")])
    report = cal.score(items, scores)
    order = [(c["rule"], "".join(c["pair"])) for c in report["combinations"]]
    # a, b fails (4/9); b, c (0.8) is tried before a, c (0.6).
    assert order[:3] == [("fixed", "ab"), ("fixed", "bc"), ("fixed", "ac")]
    assert [c["rank"] for c in report["combinations"]] == [1, 2, 3, 4, 5, 6]
    assert report["chosen"] == {"rule": "fixed", "pair": ["b", "c"], "cuts": {"b": 2, "c": 2}}
    assert report["verdict"] == "pass"


def test_the_other_pairs_are_ordered_by_half_a_kappa_not_half_b() -> None:
    # Half A is _ORDER_A: b, c (0.8) before a, c (0.6). On half B a scores 3
    # on five rows, so a, c reaches 1.0 and b, c stays 0.8: ordered by half B,
    # a, c would come first and be chosen.
    items, scores = _case([*_ORDER_A, *_pattern("B", "1111100000", "1111000000", "1111100000")])
    report = cal.score(items, scores)
    assert _combo(report, "fixed", "ac")["halves"]["B"]["kappa"] == 1.0
    order = [(c["rule"], "".join(c["pair"])) for c in report["combinations"]]
    assert order[:3] == [("fixed", "ab"), ("fixed", "bc"), ("fixed", "ac")]
    assert report["chosen"] == {"rule": "fixed", "pair": ["b", "c"], "cuts": {"b": 2, "c": 2}}
    assert report["verdict"] == "pass"


def test_each_rule_orders_its_pairs_by_its_own_kappa() -> None:
    # c scores one step below a: 2 on rows 0 and 1, 1 on rows 2 and 3, where
    # a scores 3. b scores 3 on rows 0, 1, and 4.
    #   Fixed cut: a, c agree on 8 rows, a says yes 4, c 2. p_e = 0.4*0.2 +
    #     0.6*0.8 = 0.56, kappa = 0.24 / 0.44 = 6/11. b, c agree on 9, b says
    #     yes 3: p_e = 0.3*0.2 + 0.7*0.8 = 0.62, kappa = 0.28 / 0.38 = 14/19.
    #   Per-grader cuts: a's rate is 0.4 at every cut and c's is 0.4 at cut 1,
    #     so a, c fit (2, 1) and agree on every row, kappa 1. b's rate is 0.3
    #     at every cut, as near c's 0.4 as its 0.2, so b, c fit (2, 2) and
    #     keep 14/19.
    # So the fixed rule tries b, c before a, c, and the per-grader rule a, c
    # before b, c.
    a, b, c = "3333000000", "3300300000", "2211000000"
    rows: list[Row] = [(h, k >= 7, int(a[k]), int(b[k]), int(c[k]))
                       for h in "AB" for k in range(10)]
    items, scores = _case(rows)
    report = cal.score(items, scores)
    assert _combo(report, "fixed", "ac")["halves"]["A"]["kappa"] == pytest.approx(6 / 11)
    assert _combo(report, "fixed", "bc")["halves"]["A"]["kappa"] == pytest.approx(14 / 19)
    assert _combo(report, "per_grader", "ac")["cuts"] == {"a": 2, "c": 1}
    assert _combo(report, "per_grader", "ac")["halves"]["A"]["kappa"] == 1.0
    assert _combo(report, "per_grader", "bc")["cuts"] == {"b": 2, "c": 2}
    assert _combo(report, "per_grader", "bc")["halves"]["A"]["kappa"] == pytest.approx(14 / 19)
    order = [(c["rule"], "".join(c["pair"])) for c in report["combinations"]]
    assert order == [("fixed", "ab"), ("fixed", "bc"), ("fixed", "ac"),
                     ("per_grader", "ab"), ("per_grader", "ac"), ("per_grader", "bc")]


def test_the_a_b_pair_is_chosen_over_a_pair_with_a_higher_kappa() -> None:
    # a, b: 0.6, qualifies. a, c: 1.0. a, b is still first.
    rows = _pattern("A", "1111100000", "1111111000", "1111100000")
    items, scores = _case([*rows, *_pattern("B", "1111100000", "1111111000", "1111100000")])
    report = cal.score(items, scores)
    assert _combo(report, "fixed", "ac")["halves"]["A"]["kappa"] == 1.0
    assert report["chosen"]["pair"] == ["a", "b"]


def test_the_fixed_cut_comes_before_per_grader_cuts() -> None:
    items, scores = _case([*_ORDER_A, *_pattern("B", "1111111000", "1111000000", "1111100000")])
    report = cal.score(items, scores)
    assert [c["rule"] for c in report["combinations"]] == ["fixed"] * 3 + ["per_grader"] * 3
    # Scores of 0 and 3 fit (2, 2), so per-grader a, b is fixed a, b again;
    # fixed b, c qualifies and is chosen though per-grader comes later.
    assert _combo(report, "per_grader", "bc")["halves"]["A"]["meets_bars"]
    assert report["chosen"]["rule"] == "fixed"


def _per_grader_rows(half: str) -> list[Row]:
    # a scores 3 where b scores 1. At the fixed cut b calls nothing relevant,
    # and c disagrees with both, so no fixed pair qualifies. Per-grader cuts
    # (2, 1) make a and b agree on every row.
    a, c = "1111100000", "1000011000"
    return [(half, k >= 7, 3 * int(a[k]), int(a[k]), 3 * int(c[k])) for k in range(10)]


def test_per_grader_cuts_are_chosen_when_no_fixed_combination_qualifies() -> None:
    items, scores = _case([*_per_grader_rows("A"), *_per_grader_rows("B")])
    report = cal.score(items, scores)
    assert not any(c["halves"]["A"]["meets_bars"] for c in report["combinations"]
                   if c["rule"] == "fixed")
    assert report["chosen"] == {"rule": "per_grader", "pair": ["a", "b"],
                                "cuts": {"a": 2, "b": 1}}
    assert report["verdict"] == "pass"


def test_half_b_decides_a_combination_that_passed_half_a() -> None:
    rows = [*_pattern("A", "1111100000", "1111100000", "1111100000"),
            *_pattern("B", "1111100000", "0000011000", "1111100000")]
    items, scores = _case(rows)
    report = cal.score(items, scores)
    assert report["chosen"]["pair"] == ["a", "b"]
    assert _combo(report, "fixed", "ab")["halves"]["A"]["meets_bars"]
    assert not _combo(report, "fixed", "ab")["halves"]["B"]["meets_bars"]
    # a, c passes half B, but it was not the half-A choice.
    assert _combo(report, "fixed", "ac")["halves"]["B"]["meets_bars"]
    assert report["verdict"] == "fail"


def test_half_b_uses_the_cuts_fitted_on_half_a() -> None:
    # On B, a scores 2 where b scores 1: at (2, 1) they agree, at (2, 2) not.
    a_rows = _per_grader_rows("A")
    b_rows: list[Row] = [("B", k >= 7, 2 * int("1111100000"[k]), int("1111100000"[k]), 0)
                         for k in range(10)]
    items, scores = _case([*a_rows, *b_rows])
    report = cal.score(items, scores)
    assert report["chosen"]["cuts"] == {"a": 2, "b": 1}
    assert _combo(report, "per_grader", "ab")["halves"]["B"]["kappa"] == 1.0
    assert report["verdict"] == "pass"


def test_half_a_alone_chooses_and_a_failed_half_a_never_reaches_half_b() -> None:
    # c scores nothing on A, and a and b disagree there; on B all three agree.
    a_rows: list[Row] = [("A", k >= 7, 3 * int("1111100000"[k]), 3 * int("1000011000"[k]), None)
                         for k in range(10)]
    items, scores = _case([*a_rows, *_pattern("B", "1111100000", "1111100000", "1111100000")])
    report = cal.score(items, scores)
    assert all(c["halves"]["B"]["meets_bars"] for c in report["combinations"])
    assert report["chosen"] is None
    assert report["verdict"] == "no qualifying combination on half A"


def test_session1_agreement_reads_the_score_at_cut_2() -> None:
    rows: list[Row] = [("A", False, 3, 0, 0), ("A", False, 1, 2, 0),
                       ("A", False, 2, 3, 0), ("B", True, None, 0, 0)]
    items, scores = _case(rows)
    s1a = {"r0": True, "r1": True, "r2": False, "r3": True}
    s1b = {"r0": False, "r1": True, "r2": None, "r3": False, "other": True}
    report = cal.score(items, scores, [s1a, s1b])
    assert report["session1"]["a"] == {"n": 3, "agreed": 1, "agreement": 1 / 3}
    assert report["session1"]["b"] == {"n": 3, "agreed": 3, "agreement": 1.0}


def test_session1_labels_must_cover_every_key_row() -> None:
    items, scores = _case([("A", False, 3, 0, 0), ("B", False, 0, 0, 0)])
    with pytest.raises(ValueError, match="lack a label"):
        cal.score(items, scores, [{"r0": True, "r1": False}, {"r0": True}])


# --- Score CLI ------------------------------------------------------------------


def _write_key(path: Path, items: Sequence[cal.CalibrationItem]) -> None:
    path.write_text("".join(json.dumps({
        "id": i.id, "half": i.half, "snapshot": i.snapshot, "prompt_id": i.prompt_id,
        "null_arm": i.null_arm}) + "\n" for i in items), encoding="utf-8")


def _write_labels(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _cli_case(tmp_path: Path) -> tuple[Path, list[Path]]:
    items, scores = _case([*_pattern("A", "1111100000", "1111100000", "1111100000"),
                           *_pattern("B", "1111100000", "1111100000", "1111100000")])
    key = tmp_path / "key.jsonl"
    _write_key(key, items)
    labels: list[Path] = []
    for g in cal.GRADERS:
        p = tmp_path / f"{g}.jsonl"
        _write_labels(p, [{"id": i, "score": s} for i, s in scores[g].items()])
        labels.append(p)
    return key, labels


def test_the_score_cli_prints_the_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    key, labels = _cli_case(tmp_path)
    s1 = tmp_path / "s1.jsonl"
    _write_labels(s1, [{"id": f"r{n}", "correct": n % 2 == 0} for n in range(20)])
    argv = ["score", "--key", str(key), "--labels", *map(str, labels),
            "--session1", str(s1), str(s1)]
    assert cal.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "pass"
    assert report["rows"] == {"A": 10, "B": 10}
    assert report["null_rows"] == {"A": 3, "B": 3}
    assert set(report["session1"]) == {"a", "b"}


@pytest.mark.parametrize("bad", [
    "missing", "extra", 4, -1, 2.5, True, "2", "no score",
])
def test_bad_labels_are_refused(tmp_path: Path, bad: object) -> None:
    key, labels = _cli_case(tmp_path)
    rows = _read(labels[2])
    if bad == "missing":
        rows = rows[1:]
    elif bad == "extra":
        rows.append({"id": "not-in-key", "score": 1})
    elif bad == "no score":
        del rows[0]["score"]
    else:
        rows[0]["score"] = bad
    _write_labels(labels[2], rows)
    assert cal.main(["score", "--key", str(key), "--labels", *map(str, labels)]) == 1


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_score_names_the_grader_with_a_missing_or_extra_row(change: str) -> None:
    items, scores = _case([("A", False, 3, 3, 3), ("B", False, 0, 0, 0)])
    if change == "missing":
        del scores["c"]["r1"]
    else:
        scores["c"]["r9"] = 1
    with pytest.raises(InputError, match="grader c"):
        cal.score(items, scores)


@pytest.mark.parametrize("change", ["no grader c", "grader d"])
def test_score_needs_graders_a_b_and_c_exactly(change: str) -> None:
    items, scores = _case([("A", False, 3, 3, 3), ("B", False, 0, 0, 0)])
    if change == "no grader c":
        del scores["c"]
    else:
        scores["d"] = dict(scores["c"])
    with pytest.raises(InputError, match="need scores from graders a, b, c"):
        cal.score(items, scores)


def test_a_repeated_label_id_is_refused(tmp_path: Path) -> None:
    key, labels = _cli_case(tmp_path)
    rows = _read(labels[0])
    _write_labels(labels[0], [*rows, rows[0]])
    assert cal.main(["score", "--key", str(key), "--labels", *map(str, labels)]) == 1


@pytest.mark.parametrize("bad", ["half C", "null_arm string", "null_arm 0", "empty", "repeat"])
def test_a_bad_calibration_key_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], bad: str,
) -> None:
    key, labels = _cli_case(tmp_path)
    rows = _read(key)
    if bad == "half C":
        rows[0]["half"] = "C"
    elif bad == "null_arm string":
        rows[0]["null_arm"] = "false"
    elif bad == "null_arm 0":
        rows[0]["null_arm"] = 0
    elif bad == "empty":
        rows = []
    else:
        rows.append(rows[0])
    _write_labels(key, rows)
    assert cal.main(["score", "--key", str(key), "--labels", *map(str, labels)]) == 1
    assert str(key) in capsys.readouterr().err


def test_a_null_score_is_accepted(tmp_path: Path) -> None:
    key, labels = _cli_case(tmp_path)
    rows = _read(labels[0])
    rows[0]["score"] = None
    _write_labels(labels[0], rows)
    assert cal.main(["score", "--key", str(key), "--labels", *map(str, labels)]) == 0


def test_score_needs_three_label_files(tmp_path: Path) -> None:
    key, labels = _cli_case(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cal.main(["score", "--key", str(key), "--labels", *map(str, labels[:2])])
    assert exc.value.code == 1


def test_split_then_score_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    keys, packets = _stores(tmp_path)
    out = tmp_path / "out"
    assert _split(keys, packets, out) == 0
    capsys.readouterr()
    rows = _read(out / "key.jsonl")
    labels: list[Path] = []
    for g in cal.GRADERS:
        p = tmp_path / f"{g}.jsonl"
        _write_labels(p, [{"id": r["id"], "score": 0 if r["null_arm"] else 3} for r in rows])
        labels.append(p)
    assert cal.main(["score", "--key", str(out / "key.jsonl"),
                     "--labels", *map(str, labels)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["chosen"]["pair"] == ["a", "b"]
    assert report["verdict"] == "pass"


def test_the_harness_says_data_stays_out_of_the_repository() -> None:
    doc = cal.__doc__ or ""
    assert "Keep them out of the repository" in doc
    assert "grader_calibration_1804 split" in doc and "grader_calibration_1804 score" in doc


def test_input_error_is_the_1795_harness_error() -> None:
    assert cal.InputError is InputError
