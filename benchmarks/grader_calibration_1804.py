"""Grader calibration for real-store relevance on the #1795 session-1 rows (#1804).

The #1795 harness asks two graders for a binary relevance label. This
harness runs the calibration that #1804 pre-registered before any of it
was graded: three graders, a, b, and c, score the session-1 rows from 0
to 3, the cut that turns a score into "relevant" is fitted on half of
the stores, and one confirmatory test on the other half decides.

**Subcommands.** Each one is deterministic given its inputs and
`--seed`, and exits 1 on bad input.

- `split`: reads session-1 keys and their packet directories, and pairs
  each key with the packet that holds exactly its row ids. The stores
  with at least one item row must number `--stores`, 17 by default as
  pre-registered. In snapshot order, they are shuffled with the seed;
  the first half, rounded down, is half A (8 of 17) and the rest is
  half B. In each half the prompts are shuffled with the seed, and whole
  prompts are taken in that order until the half holds `--min-items`
  rows. It writes `A/` and `B/` under `--out`, each with `packet.jsonl`
  and `instructions.txt` for the graders, and the hidden key `key.jsonl`
  beside them. Give each grader the `A/` and `B/` directories only.
- `score`: three graders' scores, fitted and chosen on half A and
  tested on half B, as the next section states. As amendment 1
  requires, it refuses a key with a half under 2,000 rows, whatever
  `--min-items` the split ran with. With `--session1`, it also reports
  how often grader a's and grader b's score at cut 2 agrees with their
  own session-1 label.

**Procedure.** Under each rule, each pair of graders gets cuts from half
A: the fixed rule cuts every grader at 2, and the per-grader rule fits
cuts in {1, 2, 3} as amendment 1 of the pre-registration reads it. A cut
pair is eligible when both graders' relevant rates on half A at their
cuts are above 0. Among eligible pairs the fit takes the least rate
difference, then the least summed distance to (2, 2), then the higher
half-A kappa, then the lexicographically smaller cuts. A pair of graders
with no eligible cuts is reported without cuts or figures and doesn't
qualify under the per-grader rule. Kappa counts the rows both graders
scored, each score read at its grader's cut. When both graders give all
those rows one label, chance agreement is 1 and kappa is undefined,
which never reaches the bar. A grader's null-arm rate is the share of
its scored null-arm rows at or above its cut. A rule and pair qualify on
half A when kappa reaches `KAPPA_BAR` and each grader's null-arm rate is
at most `NULL_RATE_CEILING`. The fixed rule is tried before the
per-grader rule. Within a rule, the a and b pair comes first, then the
other pairs by descending half-A kappa, and a pair without cuts comes
after every pair with them. The first qualifying combination, with its
half-A cuts, must meet the same bars on half B. Half B alone gives the
verdict. Everything else in the report decides nothing.

**Data.** Packets, keys, and labels hold user text or ids from a real
store. Keep them out of the repository.

Usage::

    uv run python -m benchmarks.grader_calibration_1804 split --keys key-1.jsonl key-2.jsonl --packets <packet-1> <packet-2> --seed 1804 --out <calibration-dir>
    uv run python -m benchmarks.grader_calibration_1804 score --key <calibration-dir>/key.jsonl --labels a.jsonl b.jsonl c.jsonl --session1 a-session1.jsonl b-session1.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import stat
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, cast

from benchmarks.graded_gate import (
    Labels,
    grader_figures,
    paired_kappa,
    read_labels,
    require_labels,
)
from benchmarks.real_store_ab_1795 import (
    KEY_ITEM,
    KEY_STORE,
    InputError,
    KeyItem,
    KeyStore,
    _dump,  # pyright: ignore[reportPrivateUsage]
    _jsonl,  # pyright: ignore[reportPrivateUsage]
    _key_item,  # pyright: ignore[reportPrivateUsage]
    _key_store,  # pyright: ignore[reportPrivateUsage]
    _Parser,  # pyright: ignore[reportPrivateUsage]
    exact_bar,
    exact_kappa,
    make_key,
)

DEFAULT_SEED: Final[int] = 1804
MIN_HALF_ROWS: Final[int] = 2000
"""Rows each half must hold at least; `score` refuses a smaller half (#1804 amendment 1)."""
DEFAULT_MIN_ITEMS: Final[int] = MIN_HALF_ROWS
"""Rows `split` takes for each half at least, in whole prompts (#1804)."""
DEFAULT_STORES: Final[int] = 17
"""Stores with item rows in the session-1 keys; the first 8 go to half A (#1804)."""

HALF_A: Final[str] = "A"
HALF_B: Final[str] = "B"
HALVES: Final[tuple[str, ...]] = (HALF_A, HALF_B)
GRADERS: Final[tuple[str, ...]] = ("a", "b", "c")
"""Grader a and grader b graded session 1; grader c is the third family."""
PAIRS: Final[tuple[tuple[str, str], ...]] = (("a", "b"), ("a", "c"), ("b", "c"))
"""Every pair, the a and b pair first, as the order of preference starts."""
SESSION1_GRADERS: Final[tuple[str, ...]] = ("a", "b")

SCORES: Final[tuple[int, ...]] = (0, 1, 2, 3)
FIXED_CUT: Final[int] = 2
"""Fixed rule: a row is relevant when its score is 2 or higher, for every grader."""
CUTS: Final[tuple[int, ...]] = (1, 2, 3)
"""Per-grader rule: each grader's cut is one of these."""
CUT_TIE_TARGET: Final[tuple[int, int]] = (2, 2)
"""Per-grader rule: equal rate differences go to the cuts nearest this pair.

Nearest is by the sum of absolute differences. On the 3 by 3 grid of cuts
it orders the pairs as the Euclidean distance does.
"""
SESSION1_CUT: Final[int] = FIXED_CUT
"""Cut a 0 to 3 score is read at against the session-1 binary label."""

RULE_FIXED: Final[str] = "fixed"
RULE_PER_GRADER: Final[str] = "per_grader"
RULES: Final[tuple[str, ...]] = (RULE_FIXED, RULE_PER_GRADER)
"""Rules in the order of preference."""

# Pre-registered bars, from the #1804 pre-registration.
KAPPA_BAR: Final[float] = 0.60
"""A pair's kappa must reach this, on half A to qualify and on half B to pass."""
NULL_RATE_CEILING: Final[float] = 0.20
"""Neither grader may call more than this share of the null arm relevant."""

V_PASS: Final[str] = "pass"
V_FAIL: Final[str] = "fail"
V_NONE: Final[str] = "no qualifying combination on half A"

PACKET_FILE: Final[str] = "packet.jsonl"
INSTRUCTIONS_FILE: Final[str] = "instructions.txt"
KEY_FILE: Final[str] = "key.jsonl"
PACKET_FIELDS: Final[tuple[str, ...]] = ("id", "prompt", "belief")

SCALE: Final[tuple[tuple[str, str], ...]] = (
    ("3", ("Needed: without the note, the answer would be wrong or would miss a rule,"
           " a preference, or an earlier decision about this exact task.")),
    ("2", ("Would help: reading the note would change or directly inform what the"
           " assistant does for this prompt.")),
    ("1", ("Same topic only: the note is about the same project, file, tool, or topic,"
           " but wouldn't change the answer.")),
    ("0", "Unrelated, or it only shares words with the prompt."),
    ("null", "You can't tell what the prompt asks or what the note means."),
)
"""The pre-registration's scale table, one `(score, meaning)` row per score."""

INSTRUCTIONS: Final[str] = (
    "Each row of packet.jsonl pairs a prompt a user typed to a coding assistant\n"
    "with one note from that project's memory. Score each row from 0 to 3, or\n"
    "null:\n\n"
    + "".join(f"  {score:<5} {meaning}\n" for score, meaning in SCALE)
    + "\nWrite one JSON object per line to your own labels file:\n"
    '  {"id": "<row id>", "score": 2}\n'
    "The score is 0, 1, 2, 3, or null. Score every row. Rows come in no\n"
    "particular order. Don't discuss the rows with the other graders.\n"
)

Scores = Mapping[str, int | None]


# --- Split --------------------------------------------------------------------


@dataclass(frozen=True)
class SourceKey:
    """One session-1 key file: its item rows and its store rows by snapshot."""

    path: Path
    items: tuple[KeyItem, ...]
    stores: Mapping[str, KeyStore]

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(i.id for i in self.items)


@dataclass(frozen=True)
class SourcePacket:
    """One session-1 packet directory's rows by id."""

    path: Path
    rows: Mapping[str, Mapping[str, str]]

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(self.rows)


def read_source_key(path: Path) -> SourceKey:
    """A session-1 key's rows; InputError on a malformed row or a repeated store or id.

    Unlike `read_key` of the #1795 harness, a key with no item rows, or
    with no A0 item, is read: such a store adds no rows to the split.
    """
    items: list[KeyItem] = []
    stores: dict[str, KeyStore] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj: object = json.loads(line)
        if not isinstance(obj, dict):
            raise InputError(f"{path}:{n}: not a JSON object")
        row = cast(dict[str, Any], obj)
        snapshot: object = row.get("snapshot")
        if not isinstance(snapshot, str) or not snapshot:
            raise InputError(f"{path}:{n}: need a snapshot")
        kind: object = row.get("kind")
        if kind == KEY_STORE:
            store = _key_store(row)
            if store is None or snapshot in stores:
                raise InputError(f"{path}:{n}: a malformed or second store row for {snapshot}")
            stores[snapshot] = store
        elif kind == KEY_ITEM:
            item = _key_item(row, snapshot)
            if item is None:
                raise InputError(f"{path}:{n}: a malformed item row")
            items.append(item)
        else:
            raise InputError(f"{path}:{n}: need kind {KEY_STORE} or {KEY_ITEM}")
    if not stores:
        raise InputError(f"{path}: no store row")
    if len({i.id for i in items}) != len(items):
        raise InputError(f"{path}: repeats an item id")
    return SourceKey(path=path, items=tuple(items), stores=stores)


def read_source_packet(directory: Path) -> SourcePacket:
    """A session-1 packet directory's `packet.jsonl` rows; InputError when malformed."""
    path = directory / PACKET_FILE
    rows: dict[str, Mapping[str, str]] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj: object = json.loads(line)
        if not isinstance(obj, dict):
            raise InputError(f"{path}:{n}: not a JSON object")
        row = cast(dict[str, object], obj)
        if not all(isinstance(row.get(f), str) for f in PACKET_FIELDS):
            raise InputError(f"{path}:{n}: need a string id, prompt, and belief")
        rid = cast(str, row["id"])
        if rid in rows:
            raise InputError(f"{path}:{n}: repeats row {rid}")
        rows[rid] = {f: cast(str, row[f]) for f in PACKET_FIELDS}
    return SourcePacket(path=directory, rows=rows)


def pair_sources(
    keys: Sequence[SourceKey], packets: Sequence[SourcePacket],
) -> list[tuple[SourceKey, SourcePacket]]:
    """Each key with the packet holding exactly its row ids; InputError unless one to one.

    Two keys can hold the same ids only when both hold none, and then
    either pairing gives the same split.
    """
    unmatched = Counter(k.ids for k in keys)
    unmatched.subtract(p.ids for p in packets)
    for k in keys:
        if unmatched[k.ids] > 0:
            raise InputError(f"no packet directory holds exactly the row ids of {k.path}")
    for p in packets:
        if unmatched[p.ids] < 0:
            raise InputError(f"no key holds exactly the row ids of {p.path}")
    free = list(packets)
    pairs: list[tuple[SourceKey, SourcePacket]] = []
    for k in keys:
        packet = next(p for p in free if p.ids == k.ids)
        free.remove(packet)
        pairs.append((k, packet))
    return pairs


def split_stores(snapshots: Sequence[str], seed: int) -> tuple[list[str], list[str]]:
    """`(half A, half B)`: the snapshots in order, shuffled, cut at half rounded down."""
    order = sorted(snapshots)
    random.Random(seed).shuffle(order)
    cut = len(order) // 2
    return order[:cut], order[cut:]


def sample_prompts(items: Sequence[KeyItem], seed: int, min_items: int) -> list[KeyItem]:
    """Whole prompts, in seeded order, until the rows reach `min_items`.

    A prompt is its id and its store's snapshot, since two stores can
    share a prompt id. InputError when every prompt together holds fewer
    rows.
    """
    by_prompt: dict[tuple[str, str], list[KeyItem]] = {}
    for i in items:
        by_prompt.setdefault((i.prompt_id, i.snapshot), []).append(i)
    order = sorted(by_prompt)
    random.Random(seed).shuffle(order)
    taken: list[KeyItem] = []
    for prompt in order:
        if len(taken) >= min_items:
            break
        taken.extend(by_prompt[prompt])
    if len(taken) < min_items:
        raise InputError(f"the half holds {len(taken)} rows, fewer than {min_items}")
    return taken


def build_split(
    pairs: Sequence[tuple[SourceKey, SourcePacket]], seed: int, min_items: int, stores: int,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    """`({half: packet rows}, key rows)` for the paired keys and packets.

    InputError unless exactly `stores` stores hold item rows. The packet
    rows of each half are shuffled with `seed`, so their order tells
    neither the store nor the arm. Key rows are in half and id order.
    """
    if min_items < 1:
        raise InputError("--min-items must be at least 1")
    if stores < 2:
        raise InputError("--stores must be at least 2")
    items: list[KeyItem] = []
    store_rows: dict[str, KeyStore] = {}
    rows: dict[str, Mapping[str, str]] = {}
    for key, packet in pairs:
        repeated = sorted(key.stores.keys() & store_rows.keys())
        if repeated:
            raise InputError(f"two keys hold a store row for snapshot {repeated[0]}")
        store_rows.update(key.stores)
        items.extend(key.items)
        rows.update(packet.rows)
    if len({i.id for i in items}) != len(items):
        raise InputError("two keys hold the same item id")
    make_key(items, store_rows)
    snapshots = sorted({i.snapshot for i in items})
    if len(snapshots) != stores:
        raise InputError(
            f"the keys hold {len(snapshots)} stores with item rows, not --stores {stores}")
    halves = dict(zip(HALVES, split_stores(snapshots, seed), strict=True))
    packets: dict[str, list[dict[str, str]]] = {}
    key_rows: list[dict[str, Any]] = []
    for half in HALVES:
        members = set(halves[half])
        taken = sample_prompts([i for i in items if i.snapshot in members], seed, min_items)
        packet = [dict(rows[i.id]) for i in sorted(taken, key=lambda i: i.id)]
        random.Random(seed).shuffle(packet)
        packets[half] = packet
        key_rows.extend({
            "id": i.id, "half": half, "snapshot": i.snapshot, "prompt_id": i.prompt_id,
            "null_arm": i.arms is None,
        } for i in sorted(taken, key=lambda i: i.id))
    return packets, key_rows


def write_split(
    packets: Mapping[str, Sequence[Mapping[str, str]]], key_rows: Sequence[Mapping[str, Any]],
    out: Path,
) -> None:
    """Write each half's packet and instructions to `out/<half>`, and the key to `out`.

    The graders get the half directories, so the key must lie outside
    both. A half directory that links to `out` or to the other half is
    refused, as `write_packet` of the #1795 harness refuses a key in its
    packet directory. Every file is written through any link at its
    name, so a symlink or a second hard link at one could send a packet
    over the key, or the key into a packet; links are refused before
    anything is written.
    """
    key_path = out / KEY_FILE
    dirs = [out / h for h in HALVES]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    if dirs[0].samefile(dirs[1]):
        raise InputError(f"{dirs[0]} and {dirs[1]} are one directory")
    spellings = (key_path.absolute(), key_path.resolve())
    for d in dirs:
        if any(p.exists() and p.samefile(d) for k in spellings for p in (k, *k.parents)):
            raise InputError(f"the key {key_path} must be outside the packet directory {d}")
    files = [f for d in dirs for f in (d / PACKET_FILE, d / INSTRUCTIONS_FILE)]
    for f in (*files, key_path):
        try:
            st = f.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(st.st_mode) or st.st_nlink > 1:
            raise InputError(f"{f} is a link; remove it before writing the split")
    for half, d in zip(HALVES, dirs, strict=True):
        (d / PACKET_FILE).write_text(_jsonl(packets[half]), encoding="utf-8")
        (d / INSTRUCTIONS_FILE).write_text(INSTRUCTIONS, encoding="utf-8")
    key_path.write_text(_jsonl(key_rows), encoding="utf-8")


# --- Score --------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationItem:
    """One row of the split's hidden key."""

    id: str
    half: str
    snapshot: str
    prompt_id: str
    null_arm: bool


def read_calibration_key(path: Path) -> list[CalibrationItem]:
    """The split's key rows; InputError on a malformed row, a repeated id, or no row."""
    items: list[CalibrationItem] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj: object = json.loads(line)
        if not isinstance(obj, dict):
            raise InputError(f"{path}:{n}: not a JSON object")
        row = cast(dict[str, object], obj)
        if (not all(isinstance(row.get(f), str) for f in ("id", "snapshot", "prompt_id"))
                or row.get("half") not in HALVES or not isinstance(row.get("null_arm"), bool)):
            raise InputError(
                f"{path}:{n}: need id, half A or B, snapshot, prompt_id, and null_arm")
        items.append(CalibrationItem(
            id=cast(str, row["id"]), half=cast(str, row["half"]),
            snapshot=cast(str, row["snapshot"]), prompt_id=cast(str, row["prompt_id"]),
            null_arm=cast(bool, row["null_arm"]),
        ))
    if not items:
        raise InputError(f"{path}: empty key")
    if len({i.id for i in items}) != len(items):
        raise InputError(f"{path}: repeats an item id")
    return items


def read_scores(path: Path) -> dict[str, int | None]:
    """{id: 0 to 3 or None} from a JSONL file of {"id", "score"}; InputError when malformed."""
    scores: dict[str, int | None] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj: object = json.loads(line)
        if not isinstance(obj, dict):
            raise InputError(f"{path}:{n}: not a JSON object")
        row = cast(dict[str, object], obj)
        rid, score = row.get("id"), row.get("score")
        # `True == 1` in Python, so a boolean is refused by type, not by value.
        valid = score is None or (type(score) is int and score in SCORES)
        if not isinstance(rid, str) or "score" not in row or not valid:
            raise InputError(f"{path}:{n}: need an id and a score of 0, 1, 2, 3, or null")
        if rid in scores:
            raise InputError(f"{path}:{n}: repeats row {rid}")
        scores[rid] = cast(int | None, score)
    return scores


def require_scores(ids: Sequence[str], scores: Mapping[str, Scores]) -> None:
    """InputError unless each grader scored exactly the key's ids."""
    wanted = set(ids)
    for grader, s in scores.items():
        missing = sorted(wanted - s.keys())
        if missing:
            raise InputError(f"grader {grader} has no score for {len(missing)} key rows,"
                             f" {missing[0]} among them")
        extra = sorted(s.keys() - wanted)
        if extra:
            raise InputError(f"grader {grader} scored {len(extra)} rows the key lacks,"
                             f" {extra[0]} among them")


def require_disjoint_halves(items: Sequence[CalibrationItem]) -> None:
    """InputError when one store has rows in both halves.

    The pre-registration puts no store in both halves, and `split` keeps
    to it, but `score` reads any key file, so it checks again.
    """
    half_a, half_b = ({i.snapshot for i in items if i.half == h} for h in HALVES)
    shared = sorted(half_a & half_b)
    if shared:
        raise InputError(f"store {shared[0]} has rows in both halves")


def require_half_rows(items: Sequence[CalibrationItem]) -> None:
    """InputError when a half holds fewer than `MIN_HALF_ROWS` rows.

    Amendment 1 refuses such a half. `split` stops at `--min-items`, which
    a caller can set lower, and `score` reads any key file, so it checks
    the count itself.
    """
    for half in HALVES:
        rows = sum(1 for i in items if i.half == half)
        if rows < MIN_HALF_ROWS:
            raise InputError(f"half {half} holds {rows} rows, fewer than {MIN_HALF_ROWS}")


def binarize(scores: Scores, cut: int) -> dict[str, bool | None]:
    """{id: score at or above `cut`}, a null score staying None."""
    return {i: None if s is None else s >= cut for i, s in scores.items()}


def relevant_rate(ids: Sequence[str], scores: Scores, cut: int) -> Fraction:
    """Share of the grader's non-null scores over `ids` at or above `cut`; 0 when none."""
    graded = [s for i in ids if (s := scores[i]) is not None]
    return Fraction(sum(1 for s in graded if s >= cut), len(graded)) if graded else Fraction(0)


def cut_kappa(
    ids: Sequence[str], lx: Mapping[str, bool | None], ly: Mapping[str, bool | None],
) -> Fraction | None:
    """Exact kappa of two graders' labels over `ids`, or None when it is undefined.

    With no row both scored, or one label from both graders on every row,
    chance agreement is 1 and kappa is 0/0. `exact_kappa` gives 0 and 1
    there, and a 1 would pass a rule that calls no row relevant.
    """
    both = [i for i in ids if lx[i] is not None and ly[i] is not None]
    values = {lx[i] for i in both} | {ly[i] for i in both}
    return None if len(values) < 2 else exact_kappa(ids, lx, ly)


def fit_cuts(ids: Sequence[str], x: Scores, y: Scores) -> tuple[int, int] | None:
    """Per-grader rule: the cuts for graders `x` and `y` over `ids`, or None.

    A cut pair is eligible when both graders' relevant rates at their cuts
    are above 0. Among eligible pairs the fit takes the least rate
    difference, then the least summed absolute distance to
    `CUT_TIE_TARGET`, then the higher kappa, an undefined kappa ranking
    below every defined one, then the lexicographically smaller cuts.
    None when no cut pair is eligible.
    """
    rx = {c: relevant_rate(ids, x, c) for c in CUTS}
    ry = {c: relevant_rate(ids, y, c) for c in CUTS}
    tx, ty = CUT_TIE_TARGET

    def rank(c: tuple[int, int]) -> tuple[Fraction, int, bool, Fraction, tuple[int, int]]:
        kappa = cut_kappa(ids, binarize(x, c[0]), binarize(y, c[1]))
        return (abs(rx[c[0]] - ry[c[1]]), abs(c[0] - tx) + abs(c[1] - ty),
                kappa is None, -kappa if kappa is not None else Fraction(0), c)

    eligible = [(cx, cy) for cx in CUTS for cy in CUTS if rx[cx] > 0 and ry[cy] > 0]
    return min(eligible, key=rank) if eligible else None


def weighted_kappa(pairs: Sequence[tuple[int, int]]) -> float:
    """Cohen's kappa with quadratic weights on `(x, y)` scores from 0 to 3.

    `1 - sum(w * observed) / sum(w * expected)`, `w = (x - y)**2`, the
    expected counts from the two graders' marginals. Like `cohens_kappa`,
    no pair gives 0 and graders who agree on every pair give 1.
    """
    n = len(pairs)
    if n == 0:
        return 0.0
    if all(x == y for x, y in pairs):
        return 1.0
    cx, cy = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    observed = Fraction(sum((x - y) ** 2 for x, y in pairs), n)
    expected = Fraction(
        sum(cx[i] * cy[j] * (i - j) ** 2 for i in SCORES for j in SCORES), n * n)
    return float(1 - observed / expected)


@dataclass(frozen=True)
class HalfResult:
    """One rule and pair on one half: the report, whether it meets the bars, and kappa.

    `kappa` is None when it is undefined.
    """

    figures: dict[str, Any]
    meets_bars: bool
    kappa: Fraction | None


def half_result(
    items: Sequence[CalibrationItem], half: str, pair: tuple[str, str],
    cuts: tuple[int, int], scores: Mapping[str, Scores],
) -> HalfResult:
    """Kappa, null-arm and relevant rates, agreement, and weighted kappa on one half."""
    ids = [i.id for i in items if i.half == half]
    null_ids = [i.id for i in items if i.half == half and i.null_arm]
    (gx, gy), (cx, cy) = pair, cuts
    lx, ly = binarize(scores[gx], cx), binarize(scores[gy], cy)
    float_kappa, n, agreed = paired_kappa(ids, lx, ly)
    both = [i for i in ids if lx[i] is not None and ly[i] is not None]
    kappa = cut_kappa(ids, lx, ly)
    x_only = sum(1 for i in both if lx[i] and not ly[i])
    y_only = sum(1 for i in both if ly[i] and not lx[i])
    disagreed = x_only + y_only
    null: dict[str, Any] = {}
    relevant: dict[str, Any] = {}
    nulls_ok = True
    for g, labels in ((gx, lx), (gy, ly)):
        fig = grader_figures(null_ids, labels)
        graded, called = int(fig["graded"]), int(fig["correct"])
        null[g] = {"graded": graded, "called_relevant": called, "rate": fig["precision"]}
        nulls_ok = nulls_ok and graded > 0 and (
            Fraction(called, graded) <= exact_bar(NULL_RATE_CEILING))
        rel = grader_figures(ids, labels)
        relevant[g] = {"graded": rel["graded"], "relevant": rel["correct"],
                       "rate": rel["precision"]}
    raw = [(cast(int, scores[gx][i]), cast(int, scores[gy][i])) for i in both]
    figures: dict[str, Any] = {
        "rows": len(ids), "null_rows": len(null_ids),
        "kappa": None if kappa is None else float_kappa,
        "n": n, "agreed": agreed, "raw_agreement": agreed / n if n else 0.0,
        "null_arm": null, "relevant": relevant,
        "disagreements": {
            f"{gx}_only": x_only, f"{gy}_only": y_only,
            f"{gx}_only_share": x_only / disagreed if disagreed else 0.0,
            f"{gy}_only_share": y_only / disagreed if disagreed else 0.0,
        },
        "weighted_kappa": weighted_kappa(raw),
    }
    meets = kappa is not None and kappa >= exact_bar(KAPPA_BAR) and nulls_ok
    figures["meets_bars"] = meets
    return HalfResult(figures=figures, meets_bars=meets, kappa=kappa)


Combo = tuple[str, tuple[str, str]]
"""A rule and a pair of graders."""


def _order_key(pair: tuple[str, str], result: Mapping[str, HalfResult] | None,
               ) -> tuple[bool, bool, bool, Fraction]:
    """Sort key among the pairs of a rule.

    A pair without cuts (None) comes after every pair with them. Then the
    first pair of `PAIRS`, then the higher half-A kappa, and an undefined
    kappa last.
    """
    if result is None:
        return (True, False, False, Fraction(0))
    kappa = result[HALF_A].kappa
    return (False, pair != PAIRS[0], kappa is None,
            -kappa if kappa is not None else Fraction(0))


def preference_order(
    combos: Mapping[Combo, Mapping[str, HalfResult] | None],
) -> list[Combo]:
    """`(rule, pair)` in the order of preference: rule order, then within a rule
    the first pair of `PAIRS`, then the others by descending half-A kappa.

    A pair without cuts (None), which never qualifies, sorts after every
    pair with cuts, in `PAIRS` order. So does an undefined kappa among the
    pairs with cuts. Equal kappas keep `PAIRS` order."""
    order: list[Combo] = []
    for rule in RULES:
        ranked = sorted(PAIRS, key=lambda p: _order_key(p, combos[(rule, p)]))
        order.extend((rule, p) for p in ranked)
    return order


def choose(
    order: Sequence[Combo], combos: Mapping[Combo, Mapping[str, HalfResult] | None],
) -> Combo | None:
    """The first combination of `order` with cuts that qualifies on half A, or None."""
    for combo in order:
        result = combos[combo]
        if result is not None and result[HALF_A].meets_bars:
            return combo
    return None


def session1_agreement(
    ids: Sequence[str], scores: Scores, labels: Labels,
) -> dict[str, Any]:
    """How often a score at `SESSION1_CUT` matches the grader's own session-1 label.

    Rows where either is null are left out.
    """
    both = [i for i in ids if scores[i] is not None and labels[i] is not None]
    agreed = sum(1 for i in both if (cast(int, scores[i]) >= SESSION1_CUT) == labels[i])
    return {"n": len(both), "agreed": agreed,
            "agreement": agreed / len(both) if both else 0.0}


def score(
    items: Sequence[CalibrationItem], scores: Mapping[str, Scores],
    session1: Sequence[Labels] | None = None,
) -> dict[str, Any]:
    """The calibration report: every rule and pair on both halves, and the verdict.

    InputError when a half holds fewer than `MIN_HALF_ROWS` rows.
    `scores` maps each of `GRADERS` to its scores. Cuts are fitted on half
    A, the combination is chosen on half A, and the verdict is half B's
    alone. `session1`, when given, is grader a's and grader b's session-1
    labels, which must cover every key row.
    """
    ids = [i.id for i in items]
    if sorted(scores) != sorted(GRADERS):
        raise InputError(f"need scores from graders {', '.join(GRADERS)}")
    require_scores(ids, scores)
    require_disjoint_halves(items)
    require_half_rows(items)
    ids_a = [i.id for i in items if i.half == HALF_A]
    combos: dict[Combo, dict[str, HalfResult] | None] = {}
    cuts_of: dict[Combo, tuple[int, int] | None] = {}
    for rule in RULES:
        for pair in PAIRS:
            cuts = ((FIXED_CUT, FIXED_CUT) if rule == RULE_FIXED
                    else fit_cuts(ids_a, scores[pair[0]], scores[pair[1]]))
            cuts_of[(rule, pair)] = cuts
            combos[(rule, pair)] = None if cuts is None else {
                h: half_result(items, h, pair, cuts, scores) for h in HALVES}
    order = preference_order(combos)
    chosen = choose(order, combos)
    picked = None if chosen is None else combos[chosen]
    verdict = V_NONE if picked is None else V_PASS if picked[HALF_B].meets_bars else V_FAIL
    report_combos: list[dict[str, Any]] = []
    for rank, combo in enumerate(order, 1):
        rule, pair = combo
        cuts, result = cuts_of[combo], combos[combo]
        # A pair with no eligible per-grader cuts is reported, not scored.
        report_combos.append({
            "rank": rank, "rule": rule, "pair": list(pair),
            "cuts": None if cuts is None else dict(zip(pair, cuts, strict=True)),
            "halves": None if result is None else {h: result[h].figures for h in HALVES},
        })
    report: dict[str, Any] = {
        "rows": {h: sum(1 for i in items if i.half == h) for h in HALVES},
        "null_rows": {h: sum(1 for i in items if i.half == h and i.null_arm) for h in HALVES},
        "combinations": report_combos,
        "chosen": None if chosen is None else {
            "rule": chosen[0], "pair": list(chosen[1]),
            "cuts": report_combos[order.index(chosen)]["cuts"],
        },
        "verdict": verdict,
        "bars": {"kappa": KAPPA_BAR, "null_rate_ceiling": NULL_RATE_CEILING,
                 "fixed_cut": FIXED_CUT, "cuts": list(CUTS)},
    }
    if session1 is not None:
        s1a, s1b = session1
        require_labels(ids, s1a, s1b, "key rows")
        report["session1"] = {
            g: session1_agreement(ids, scores[g], labels)
            for g, labels in zip(SESSION1_GRADERS, session1, strict=True)
        }
    return report


# --- CLI ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True, parser_class=_Parser)
    s = sub.add_parser("split", help="half A and half B packets and the hidden key")
    s.add_argument("--keys", type=Path, nargs="+", required=True, help="session-1 keys")
    s.add_argument("--packets", type=Path, nargs="+", required=True,
                   help="session-1 packet directories, one per key, in any order")
    s.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s.add_argument("--min-items", type=int, default=DEFAULT_MIN_ITEMS)
    s.add_argument("--stores", type=int, default=DEFAULT_STORES,
                   help="stores with item rows the keys must hold")
    s.add_argument("--out", type=Path, required=True,
                   help="directory for A/, B/, and key.jsonl; give graders A/ and B/ only")
    sc = sub.add_parser("score", help="fit on half A, test on half B")
    sc.add_argument("--key", type=Path, required=True)
    sc.add_argument("--labels", type=Path, nargs=3, required=True,
                    help="grader a's, b's, and c's scores, in that order")
    sc.add_argument("--session1", type=Path, nargs=2,
                    help="grader a's and b's session-1 labels, in that order")
    return parser


def _dispatch(args: argparse.Namespace) -> None:
    if cast(str, args.cmd) == "split":
        keys = [read_source_key(p) for p in args.keys]
        packets = [read_source_packet(p) for p in args.packets]
        halves, key_rows = build_split(
            pair_sources(keys, packets), args.seed, args.min_items, args.stores)
        write_split(halves, key_rows, args.out)
        for half in HALVES:
            rows = [r for r in key_rows if r["half"] == half]
            print(f"{half}: stores={len({r['snapshot'] for r in rows})}"
                  f" prompts={len({(r['snapshot'], r['prompt_id']) for r in rows})}"
                  f" rows={len(rows)} null={sum(1 for r in rows if r['null_arm'])}")
    else:
        scores = {g: read_scores(p) for g, p in zip(GRADERS, args.labels, strict=True)}
        session1 = None if args.session1 is None else [read_labels(p) for p in args.session1]
        _dump(score(read_calibration_key(args.key), scores, session1), None)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _dispatch(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
