"""Real-store A/B harness for the HRR expand, edge-writer, and PMI arms (#1795).

The HRR expand lane, the semantic-edge writers, and the vocabulary-bridge
work were kept off or declined on LoCoMo10 and synthetic graphs. This
harness reruns those decisions on copies of real project stores, with
prompts sampled from each store's own transcripts and blind graders
judging the result.

**Store safety.** Every subcommand that reads a store copies it first,
with SQLite's backup API from a `mode=ro` source URI, and works on the
copy in a temporary directory that is deleted on exit. Nothing opens the
source store for writing. `--work-dir` picks where the temporary copy
goes.

**Subcommands.** Each one is deterministic given its inputs and
`--seed`, and exits 1 on bad input.

- `census`: edge counts by type, active beliefs, and probed-type edges
  per 100 beliefs for one or more stores, optionally after the A1
  writers ran on the copy. With `--writers` it also reports H1 on the
  median store.
- `sample`: user prompts from the store's sibling transcripts
  (`transcripts/turns.jsonl` and every file under `transcripts/archive`),
  minus the prompts the production prompt-shape gate sends to locks-only,
  deduplicated on their stripped text and sampled with `--seed`.
- `run`: one arm's non-locked packed belief ids per prompt, in rank
  order, retrieved through the production `retrieve()` with the
  arguments the UserPromptSubmit hook passes it.
- `packet`: the blind grader packet, its instructions, and the hidden key.
- `score`: the H0 to H6 verdicts from two graders' labels.
- `proxy`: whether a later turn of the same session cites or restates
  each retrieved belief. Reported beside the graders, never gating.

**Arms.**

- `A0`: defaults.
- `A1`: on the copy, the CONTRADICTS writer from `relationship_detector`
  plus the SUPPORTS and SUPERSEDES writers that #999 added and 01d4dcfb
  and 6c630223 deleted (restored below, not in `src`), then retrieval
  with `AELFRICE_HRR_EXPAND=1`.
- `A1W`: A1's writers with the expand lane off. Default retrieval reads
  the new edges too (the intentional-clustering pack merges candidates
  any edge joins), so A1 against A0 mixes that reranking with the lane.
  A1W against A0 measures the writers alone, and A1 against A1W the lane.
- `A2`: `AELFRICE_HRR_EXPAND=1`, probing `PROBE_KINDS_A2` instead of the
  lane's default kinds.
- `A3`: the prompt's top PMI neighbors from the copy's own belief texts,
  appended to the query, then retrieval with defaults.

Every arm runs with each `AELFRICE_*` and `AELF_*` variable removed from
the environment, then sets only the variables the arm needs. The working
directory moves to the temporary directory, which holds an empty
`.aelfrice.toml`, so no project config reaches the resolvers.

**Retrieval call.** The hook calls `hook_search.search_for_prompt`,
which calls `retrieve(store, query, token_budget=DEFAULT_HOOK_TOKEN_BUDGET,
manifest_reference_locks=True, belief_cost_fn=hook._ups_belief_line_cost)`
with `record_exposure=False`. `run` makes the same call with two
deliberate differences: `user_locks=()`, so the user-scope store on the
machine running the harness doesn't change the budget, and a query built
by `hook._build_conversation_aware_query` from the turns of the same
session that precede the prompt in the transcript (the bare prompt when
there are none, as in the hook). The hook reads those turns from the
live log while the prompt is submitted, so the recorded context is an
approximation of what it saw.

**Data.** Prompt files, run files, packets, keys, labels, and proxy files
hold user text or ids from a real store. Keep them out of the repository.

Usage::

    uv run python -m benchmarks.real_store_ab_1795 census --store <memory.db> --writers
    uv run python -m benchmarks.real_store_ab_1795 sample --store <memory.db> --seed 1795 --out prompts.jsonl
    uv run python -m benchmarks.real_store_ab_1795 run --store <memory.db> --prompts prompts.jsonl --arm A0 --out A0.json
    uv run python -m benchmarks.real_store_ab_1795 packet --runs A0.json A1.json A1W.json A2.json A3.json --store <memory.db> --seed 1795 --out <dir>
    uv run python -m benchmarks.real_store_ab_1795 proxy --store <memory.db> --runs A0.json A1.json A1W.json A2.json A3.json --prompts prompts.jsonl --out proxy.json
    uv run python -m benchmarks.real_store_ab_1795 score --key <dir>/key.jsonl --labels a.jsonl b.jsonl --proxy proxy.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
import statistics
import sys
import tempfile
from collections import Counter
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NoReturn, cast

from aelfrice import hrr_expand
from aelfrice.bm25 import tokenize, tokenize_stemmed
from aelfrice.dedup import (
    DEFAULT_JACCARD_MIN as DEDUP_JACCARD_MIN,
    DEFAULT_LEVENSHTEIN_MIN,
    DEFAULT_MAX_CANDIDATE_PAIRS as DEDUP_MAX_CANDIDATE_PAIRS,
    dedup_audit,
)
from aelfrice.hook import (
    DEFAULT_CONV_AWARE_WEIGHT,
    DEFAULT_CONV_AWARE_WINDOW,
    DEFAULT_HOOK_TOKEN_BUDGET,
    _build_conversation_aware_query,  # pyright: ignore[reportPrivateUsage]
    _should_skip_bm25,  # pyright: ignore[reportPrivateUsage]
    _ups_belief_line_cost,  # pyright: ignore[reportPrivateUsage]
)
from aelfrice.models import (
    EDGE_DERIVED_FROM,
    EDGE_RELATES_TO,
    EDGE_SUPERSEDES,
    EDGE_SUPPORTS,
    EDGE_TEMPORAL_NEXT,
    LOCK_NONE,
    Edge,
)
from aelfrice.rebuild_log import RecentTurn
from aelfrice.relationship_detector import (
    DEFAULT_CONFIDENCE_MIN,
    DEFAULT_JACCARD_MIN,
    DEFAULT_MAX_CANDIDATE_PAIRS,
    DEFAULT_MAX_EDGES_PER_BELIEF,
    DEFAULT_RESIDUAL_OVERLAP_MIN,
    LABEL_REFINES,
    relationships_audit,
    write_semantic_edges,
)
from aelfrice.retrieval import ENV_HRR_EXPAND, retrieve
from aelfrice.store import MemoryStore, read_only_uri
from benchmarks.graded_gate import (
    MIN_GRADED,
    Labels,
    grader_figures,
    paired_kappa,
    read_labels,
    require_labels,
    wilson_interval,
)

ARM_A0: Final[str] = "A0"
ARM_A1: Final[str] = "A1"
ARM_A1W: Final[str] = "A1W"
ARM_A2: Final[str] = "A2"
ARM_A3: Final[str] = "A3"
ARMS: Final[tuple[str, ...]] = (ARM_A0, ARM_A1, ARM_A1W, ARM_A2, ARM_A3)
WRITER_ARMS: Final[tuple[str, ...]] = (ARM_A1, ARM_A1W)
"""Arms that run the A1 writers on the copy before retrieval."""
CONTROL_ARM: Final[str] = ARM_A0

ENV_PREFIXES: Final[tuple[str, ...]] = ("AELFRICE_", "AELF_")
"""Variables with these prefixes are removed for every arm's run."""
ARM_ENV: Final[Mapping[str, Mapping[str, str]]] = {
    ARM_A0: {},
    ARM_A1: {ENV_HRR_EXPAND: "1"},
    ARM_A1W: {},
    ARM_A2: {ENV_HRR_EXPAND: "1"},
    ARM_A3: {},
}
PROBE_KINDS_A2: Final[tuple[str, ...]] = (
    EDGE_DERIVED_FROM, EDGE_RELATES_TO, EDGE_TEMPORAL_NEXT,
)
"""Edge kinds A2 probes: the ones real stores already hold (#1795)."""

TOP_K: Final[int] = 10
"""The top 10 non-locked beliefs of the packed result: H5's precision set."""
COMPARISONS: Final[tuple[tuple[str, str], ...]] = (
    (ARM_A1, ARM_A0), (ARM_A1W, ARM_A0), (ARM_A2, ARM_A0), (ARM_A3, ARM_A0),
    (ARM_A1, ARM_A1W),
)
"""`(arm, control)` pairs whose packed results the packet judges in full.

The expand lane appends its beliefs after every L1 hit, so when L1 fills
the top 10 an A1 or A2 addition lands past the cut, and a gain counted
in the top 10 alone is zero by construction. The gain is counted over
the whole packed result, which the hook injects, instead. Beliefs both
arms packed cancel, so the packet judges only the top 10s and the
beliefs one arm of a pair packed and the other didn't.
"""
DEFAULT_MAX_PROMPTS: Final[int] = 40
"""Most prompts sampled per store (#1795)."""

# A3: the PMI table. A belief is the co-occurrence window, and each stem
# counts once per belief. A stem, and a stem pair, must occur in at least
# `PMI_MIN_COUNT` beliefs. A stem in more than `PMI_MAX_DF_SHARE` of the
# beliefs neither expands nor is added, which keeps function words out. Each
# prompt stem contributes its `PMI_TOP_N_PER_TERM` strongest positive-PMI
# neighbors, and the query gains at most `PMI_MAX_ADDED` words in all.
PMI_MIN_COUNT: Final[int] = 3
PMI_MAX_DF_SHARE: Final[float] = 0.10
PMI_TOP_N_PER_TERM: Final[int] = 2
PMI_MAX_ADDED: Final[int] = 6

# Pre-registered bars, from the #1795 hypothesis table.
KAPPA_BAR: Final[float] = 0.60
"""H0: the two graders' agreement must reach this."""
NULL_RATE_CEILING: Final[float] = 0.20
"""H0: neither grader may call more than this share of the null arm relevant."""
SUBSTRATE_BAR: Final[float] = 1.0
"""H1: probed-type edges per 100 beliefs on the median store."""
GAIN_BAR: Final[float] = 0.3
"""H2 to H4: mean paired gain in graded-relevant beliefs per prompt."""
GAIN_KILL: Final[float] = 0.1
"""H2 to H4: a mean paired gain under this kills the hypothesis."""
GAINING_SHARE_BAR: Final[float] = 0.10
"""H2 to H4: share of prompts that gain at least one relevant belief."""
BRIDGING_SHARE_BAR: Final[float] = 0.5
"""H4: share of A3's relevant additions that share no stem with the prompt."""
PRECISION_DROP_BAR: Final[float] = 0.05
"""H5: largest allowed drop in an arm's precision share against A0."""
PROXY_OVERLAP_BAR: Final[float] = 0.6
"""Usage proxy: share of a belief's stems a later turn must hold."""
BAR_TOLERANCE: Final[float] = 1e-9
"""Float slack for a figure that lands exactly on a bar, such as 0.8 - 0.75."""

BOOTSTRAP_RESAMPLES: Final[int] = 10_000
"""Paired bootstrap resamples of the prompts for the 95% interval."""
CI_LOW_Q: Final[float] = 0.025
CI_HIGH_Q: Final[float] = 0.975

HYPOTHESIS_ARMS: Final[Mapping[str, str]] = {"H2": ARM_A1, "H3": ARM_A2, "H4": ARM_A3}

V_PASS: Final[str] = "pass"
V_FAIL: Final[str] = "fail"
V_INCONCLUSIVE: Final[str] = "inconclusive"
V_INSUFFICIENT: Final[str] = "insufficient"
V_UNRELIABLE: Final[str] = "unreliable"
V_NOT_MEASURED: Final[str] = "not measured"
V_STARVED: Final[str] = "substrate-starved"
V_NEEDS_CENSUS: Final[str] = "needs census"
V_AGREE: Final[str] = "agree"
V_DISAGREE: Final[str] = "disagree"

INSTRUCTIONS: Final[str] = """\
Each row of packet.jsonl pairs a prompt a user typed to a coding assistant
with one note from that project's memory. For each row, decide whether the
note is relevant to the prompt: whether an assistant answering the prompt
would be better off having read it.

  true   relevant: the note bears on what the prompt asks or on how to do it
  false  not relevant: the note is about something else, or only shares words
         with the prompt
  null   unclear: you can't tell from the prompt and the note

Write one JSON object per line to your own labels file:
  {"id": "<row id>", "correct": true}
Label every row. Rows come in no particular order. Don't discuss the rows
with the other grader.
"""


class InputError(ValueError):
    """Bad input: a malformed file, a missing field, or a bad argument."""


# --- Store copies -------------------------------------------------------------


def copy_store(src: Path, dst: Path) -> None:
    """Copy `src` to `dst` through a `mode=ro` handle and SQLite's backup API."""
    if not src.is_file():
        raise InputError(f"no store at {src}")
    source = sqlite3.connect(read_only_uri(str(src)), uri=True)
    try:
        target = sqlite3.connect(str(dst))
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def prepare_copy(src: Path, work: Path) -> Path:
    """Copy the store into `work` and open the copy once read-write.

    The read-write open runs any pending migration on the copy, so every
    arm reads the same schema and can open it read-only afterwards.
    """
    db = work / "memory.db"
    copy_store(src, db)
    MemoryStore(str(db)).close()
    return db


def active_beliefs(db: Path) -> list[tuple[str, str, str]]:
    """`(id, content, lock_level)` of every active belief, in id order."""
    conn = sqlite3.connect(read_only_uri(str(db)), uri=True)
    try:
        rows = conn.execute(
            "SELECT id, content, lock_level FROM beliefs "
            "WHERE valid_to IS NULL ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]


def belief_texts(db: Path, ids: Sequence[str]) -> dict[str, str]:
    """{id: content} for `ids`, retired beliefs included; InputError on a missing id."""
    conn = sqlite3.connect(read_only_uri(str(db)), uri=True)
    try:
        out: dict[str, str] = {}
        for bid in sorted(set(ids)):
            row = conn.execute("SELECT content FROM beliefs WHERE id = ?", (bid,)).fetchone()
            if row is None:
                raise InputError(f"belief {bid} is not in the store")
            out[bid] = str(row[0])
    finally:
        conn.close()
    return out


# --- A1 writers -----------------------------------------------------------------
#
# The SUPPORTS and SUPERSEDES writers below are the ones #999 added.
# c3a500fa dropped their ingest wiring, and 01d4dcfb and 6c630223 deleted
# them from `relationship_detector` and `dedup`. The audits they call are
# unchanged. Their per-belief cap was `DEFAULT_MAX_EDGES_PER_BELIEF` in
# each module, 8 in both; only `relationship_detector` keeps it now.


def write_supports_edges(store: MemoryStore) -> dict[str, int]:
    """SUPPORTS edges for the audit's REFINES pairs, `src = min(id)`.

    Restored from the `relationship_detector` writer that 01d4dcfb deleted.
    """
    report = relationships_audit(
        store,
        jaccard_min=DEFAULT_JACCARD_MIN,
        residual_overlap_min=DEFAULT_RESIDUAL_OVERLAP_MIN,
        confidence_min=DEFAULT_CONFIDENCE_MIN,
        max_candidate_pairs=DEFAULT_MAX_CANDIDATE_PAIRS,
    )
    refines = [p for p in report.pairs if p.label == LABEL_REFINES]
    per_belief: dict[str, int] = {}
    written = existing = gated = self_pairs = 0
    for pair in refines:
        a_id, b_id = pair.belief_a_id, pair.belief_b_id
        if a_id == b_id:
            self_pairs += 1
            continue
        if (
            per_belief.get(a_id, 0) >= DEFAULT_MAX_EDGES_PER_BELIEF
            or per_belief.get(b_id, 0) >= DEFAULT_MAX_EDGES_PER_BELIEF
        ):
            gated += 1
            continue
        src_id, dst_id = (a_id, b_id) if a_id < b_id else (b_id, a_id)
        if store.get_edge(src_id, dst_id, EDGE_SUPPORTS) is not None:
            existing += 1
        else:
            store.insert_edge(Edge(src=src_id, dst=dst_id, type=EDGE_SUPPORTS, weight=1.0))
            written += 1
        per_belief[a_id] = per_belief.get(a_id, 0) + 1
        per_belief[b_id] = per_belief.get(b_id, 0) + 1
    return {
        "n_refines": len(refines), "n_edges_written": written,
        "n_edges_skipped_existing": existing, "n_edges_skipped_gated": gated,
        "n_edges_skipped_self_pair": self_pairs,
    }


def write_supersedes_edges(store: MemoryStore) -> dict[str, int]:
    """SUPERSEDES edges from each near-duplicate to its cluster's oldest member.

    Restored from the `dedup` writer that 6c630223 deleted.
    """
    report = dedup_audit(
        store,
        jaccard_min=DEDUP_JACCARD_MIN,
        levenshtein_min=DEFAULT_LEVENSHTEIN_MIN,
        max_candidate_pairs=DEDUP_MAX_CANDIDATE_PAIRS,
    )
    per_belief: dict[str, int] = {}
    written = existing = gated = 0
    for cluster in report.clusters:
        members = [b for mid in cluster.member_ids if (b := store.get_belief(mid)) is not None]
        if len(members) < 2:
            continue
        oldest = min(members, key=lambda b: (b.created_at, b.id))
        for b in members:
            if b.id == oldest.id:
                continue
            src_id, dst_id = b.id, oldest.id
            if (
                per_belief.get(src_id, 0) >= DEFAULT_MAX_EDGES_PER_BELIEF
                or per_belief.get(dst_id, 0) >= DEFAULT_MAX_EDGES_PER_BELIEF
            ):
                gated += 1
                continue
            if store.get_edge(src_id, dst_id, EDGE_SUPERSEDES) is not None:
                existing += 1
            else:
                store.insert_edge(
                    Edge(src=src_id, dst=dst_id, type=EDGE_SUPERSEDES, weight=1.0)
                )
                written += 1
            per_belief[src_id] = per_belief.get(src_id, 0) + 1
            per_belief[dst_id] = per_belief.get(dst_id, 0) + 1
    return {
        "n_clusters": len(report.clusters), "n_edges_written": written,
        "n_edges_skipped_existing": existing, "n_edges_skipped_gated": gated,
    }


def apply_writers(db: Path) -> dict[str, dict[str, int]]:
    """Run the three A1 writers on the copy at `db`."""
    store = MemoryStore(str(db))
    try:
        contradicts = write_semantic_edges(store)
        supports = write_supports_edges(store)
        supersedes = write_supersedes_edges(store)
    finally:
        store.close()
    return {
        "contradicts": {
            "n_edges_written": contradicts.n_edges_written,
            "n_edges_skipped_existing": contradicts.n_edges_skipped_existing,
            "n_edges_skipped_gated": contradicts.n_edges_skipped_gated,
        },
        "supports": supports,
        "supersedes": supersedes,
    }


# --- Census --------------------------------------------------------------------


def census_store(src: Path, work: Path, *, writers: bool) -> dict[str, Any]:
    """Edge counts by type, active beliefs, and probed-type edges per 100 beliefs."""
    db = prepare_copy(src, work)
    report: dict[str, Any] = {"store": str(src)}
    if writers:
        report["writers"] = apply_writers(db)
    conn = sqlite3.connect(read_only_uri(str(db)), uri=True)
    try:
        edges = {
            str(t): int(n) for t, n in conn.execute(
                "SELECT type, COUNT(*) FROM edges GROUP BY type ORDER BY type"
            ).fetchall()
        }
        beliefs = int(conn.execute(
            "SELECT COUNT(*) FROM beliefs WHERE valid_to IS NULL"
        ).fetchone()[0])
    finally:
        conn.close()
    kinds = hrr_expand.hrr_expand_edge_types()
    probed = sum(edges.get(k, 0) for k in kinds)
    report.update({
        "edges_by_type": edges,
        "edges": sum(edges.values()),
        "beliefs": beliefs,
        "probed_kinds": list(kinds),
        "probed_edges": probed,
        "probed_per_100_beliefs": 100.0 * probed / beliefs if beliefs else 0.0,
    })
    return report


def census(stores: Sequence[Path], work_dir: Path | None, *, writers: bool) -> dict[str, Any]:
    """Per-store census and, after the writers, the H1 verdict on the median store."""
    rows: list[dict[str, Any]] = []
    for src in stores:
        with tempfile.TemporaryDirectory(dir=work_dir) as work:
            rows.append(census_store(src, Path(work), writers=writers))
    median = statistics.median(r["probed_per_100_beliefs"] for r in rows)
    h1 = V_NOT_MEASURED
    if writers:
        h1 = V_PASS if median >= SUBSTRATE_BAR else V_FAIL
    return {
        "writers": writers,
        "stores": rows,
        "median_probed_per_100_beliefs": median,
        "substrate_bar": SUBSTRATE_BAR,
        "H1": h1,
    }


# --- Transcripts and the prompt sample ----------------------------------------


@dataclass(frozen=True)
class Turn:
    """One user or assistant line of the turn log, in file order."""

    seq: int
    role: str
    text: str
    session_id: str | None
    ts: str | None
    turn_id: str | None


def transcript_files(store: Path) -> list[Path]:
    """Every file under `transcripts/archive`, by path, then `transcripts/turns.jsonl`."""
    tdir = store.parent / "transcripts"
    archive = tdir / "archive"
    files = sorted(
        (p for p in archive.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(archive).as_posix(),
    ) if archive.is_dir() else []
    live = tdir / "turns.jsonl"
    if live.is_file():
        files.append(live)
    return files


def _opt_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def read_turns(files: Sequence[Path]) -> list[Turn]:
    """Turn lines of the transcript logger's format, in file and line order.

    A line counts when it is a JSON object whose `role` is `user` or
    `assistant` and whose `text` is a non-blank string, the rule
    `context_rebuilder._parse_aelfrice_turns` applies. Compaction markers
    and malformed lines are skipped. Undecodable bytes become U+FFFD, so a
    stray binary file in the archive is skipped line by line rather than
    failing the read.
    """
    turns: list[Turn] = []
    for path in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                obj: object = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            row = cast(dict[str, object], obj)
            role, text = row.get("role"), row.get("text")
            if role not in ("user", "assistant") or not isinstance(text, str) or not text.strip():
                continue
            turns.append(Turn(
                seq=len(turns), role=cast(str, role), text=text,
                session_id=_opt_str(row.get("session_id")), ts=_opt_str(row.get("ts")),
                turn_id=_opt_str(row.get("turn_id")),
            ))
    return turns


def prompt_id(turn: Turn) -> str:
    """Stable id from the turn's session, turn id, timestamp, and text."""
    key = "\0".join((turn.session_id or "", turn.turn_id or "", turn.ts or "", turn.text))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def gate_sends_to_locks_only(text: str) -> bool:
    """The production prompt-shape gate's verdict on `text`."""
    return _should_skip_bm25(text)[0]


def sample_prompts(turns: Sequence[Turn], seed: int, max_n: int) -> list[dict[str, Any]]:
    """Seeded sample of distinct user prompts the gate lets through, in turn order.

    Each row carries the prompt, where it sits in the log, and `context`:
    up to `DEFAULT_CONV_AWARE_WINDOW` earlier turns of the same session,
    which `run` folds into the query the way the hook does.
    """
    if max_n < 1:
        raise InputError("--max must be at least 1")
    if not turns:
        raise InputError("no transcript turns beside the store")
    seen: set[str] = set()
    candidates: list[Turn] = []
    for t in turns:
        if t.role != "user" or gate_sends_to_locks_only(t.text):
            continue
        key = t.text.strip()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(t)
    if not candidates:
        raise InputError("no user prompt passes the prompt-shape gate")
    rng = random.Random(seed)
    picked = candidates if len(candidates) <= max_n else rng.sample(candidates, max_n)
    picked = sorted(picked, key=lambda t: t.seq)
    rows: list[dict[str, Any]] = []
    for t in picked:
        before = [{"role": u.role, "text": u.text} for u in turns[:t.seq]
                  if t.session_id is not None and u.session_id == t.session_id]
        rows.append({
            "prompt_id": prompt_id(t), "text": t.text, "session_id": t.session_id,
            "turn_id": t.turn_id, "ts": t.ts, "seq": t.seq,
            "context": before[-DEFAULT_CONV_AWARE_WINDOW:],
        })
    return rows


def _is_context_turn(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    row = cast(dict[str, object], obj)
    return isinstance(row.get("role"), str) and isinstance(row.get("text"), str)


def read_prompts(path: Path) -> list[dict[str, Any]]:
    """Rows of a `sample` output file; InputError on a malformed row."""
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj: object = json.loads(line)
        if not isinstance(obj, dict):
            raise InputError(f"{path}:{n}: not a JSON object")
        row = cast(dict[str, Any], obj)
        ctx: object = row.get("context", [])
        if (
            not isinstance(row.get("prompt_id"), str) or not isinstance(row.get("text"), str)
            or not isinstance(ctx, list)
            or not all(_is_context_turn(c) for c in cast(list[object], ctx))
        ):
            raise InputError(f"{path}:{n}: need prompt_id, text, and a context list")
        rows.append(row)
    if not rows:
        raise InputError(f"{path}: no prompts")
    if len({r["prompt_id"] for r in rows}) != len(rows):
        raise InputError(f"{path}: duplicate prompt ids")
    return rows


# --- A3: PMI query expansion -------------------------------------------------


class PMIExpander:
    """Deterministic PMI neighbors of a prompt's stems over a belief corpus.

    `PMI(x, y) = log(N * c(x, y) / (c(x) * c(y)))`, where `N` is the
    number of beliefs, `c(x)` the number holding stem `x`, and `c(x, y)`
    the number holding both. Stems come from `bm25.tokenize_stemmed`, the
    BM25 index vocabulary. An added stem is written as its most frequent
    surface word in the corpus, ties broken by the smaller word, so the
    retrieval tokenizer stems it back to the same stem.
    """

    def __init__(
        self, texts: Sequence[str], *,
        min_count: int = PMI_MIN_COUNT, max_df_share: float = PMI_MAX_DF_SHARE,
        top_n_per_term: int = PMI_TOP_N_PER_TERM, max_added: int = PMI_MAX_ADDED,
    ) -> None:
        self.min_count = min_count
        self.top_n_per_term = top_n_per_term
        self.max_added = max_added
        self._docs = [frozenset(tokenize_stemmed(t)) for t in texts]
        self._n = len(self._docs)
        self._max_df = max_df_share * self._n
        postings: dict[str, list[int]] = {}
        for i, doc in enumerate(self._docs):
            for stem in doc:
                postings.setdefault(stem, []).append(i)
        self._postings = postings
        self._cache: dict[str, list[tuple[str, float, int]]] = {}
        words: Counter[str] = Counter()
        for t in texts:
            words.update(tokenize(t))
        surface: dict[str, tuple[int, str]] = {}
        for word in sorted(words):
            stems = tokenize_stemmed(word)
            if len(stems) != 1:
                continue
            best = surface.get(stems[0])
            if best is None or words[word] > best[0]:
                surface[stems[0]] = (words[word], word)
        self._surface = {stem: w for stem, (_c, w) in surface.items()}

    def _eligible(self, stem: str) -> bool:
        df = len(self._postings.get(stem, ()))
        return self.min_count <= df <= self._max_df

    def neighbors(self, stem: str) -> list[tuple[str, float, int]]:
        """`(stem, pmi, co-count)` with positive PMI, strongest first.

        Sorted by PMI descending, then co-count descending, then stem.
        """
        cached = self._cache.get(stem)
        if cached is not None:
            return cached
        out: list[tuple[str, float, int]] = []
        if self._eligible(stem):
            co: Counter[str] = Counter()
            for i in self._postings[stem]:
                co.update(self._docs[i])
            del co[stem]
            df_x = len(self._postings[stem])
            for y, c in co.items():
                if c < self.min_count or not self._eligible(y):
                    continue
                pmi = math.log(self._n * c / (df_x * len(self._postings[y])))
                if pmi > 0:
                    out.append((y, pmi, c))
            out.sort(key=lambda r: (-r[1], -r[2], r[0]))
        self._cache[stem] = out
        return out

    def expand(self, prompt: str) -> list[str]:
        """Surface words to append to the query for `prompt`, strongest first."""
        q = set(tokenize_stemmed(prompt))
        best: dict[str, tuple[float, int]] = {}
        for stem in sorted(q):
            picked = [r for r in self.neighbors(stem) if r[0] not in q][: self.top_n_per_term]
            for y, pmi, c in picked:
                if (pmi, c) > best.get(y, (float("-inf"), -1)):
                    best[y] = (pmi, c)
        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], -kv[1][1], kv[0]))
        words = [self._surface[y] for y, _ in ranked if y in self._surface]
        return words[: self.max_added]


# --- Retrieval runs -----------------------------------------------------------


@contextmanager
def arm_environment(arm: str, work: Path) -> Generator[None]:
    """Run the block with only `arm`'s settings reaching retrieval.

    Removes every `ENV_PREFIXES` variable, sets the arm's own, moves the
    working directory to `work` (which gets an empty `.aelfrice.toml`, so
    config discovery stops there), and sets A2's probe kinds. Everything
    is restored on exit.
    """
    saved = {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIXES)}
    saved_cwd = Path.cwd()
    saved_kinds = hrr_expand.probe_kinds_override
    (work / ".aelfrice.toml").write_text("", encoding="utf-8")
    try:
        for k in saved:
            del os.environ[k]
        os.environ.update(ARM_ENV[arm])
        if arm == ARM_A2:
            hrr_expand.probe_kinds_override = PROBE_KINDS_A2
        os.chdir(work)
        yield
    finally:
        os.chdir(saved_cwd)
        hrr_expand.probe_kinds_override = saved_kinds
        for k in [k for k in os.environ if k.startswith(ENV_PREFIXES)]:
            del os.environ[k]
        os.environ.update(saved)


def hook_query(prompt: Mapping[str, Any]) -> str:
    """The query the hook's conversation-aware default builds for `prompt`.

    Like the hook, a prompt with no earlier turns is the query unchanged,
    not repeated `DEFAULT_CONV_AWARE_WEIGHT` times.
    """
    context = cast(list[dict[str, str]], prompt.get("context", []))
    if not context:
        return str(prompt["text"])
    return _build_conversation_aware_query(
        str(prompt["text"]),
        [RecentTurn(role=c["role"], text=c["text"]) for c in context],
        turn_window=DEFAULT_CONV_AWARE_WINDOW,
        prompt_weight=DEFAULT_CONV_AWARE_WEIGHT,
    )


def retrieve_packed(store: MemoryStore, query: str) -> list[str]:
    """Every non-locked belief id the hook's call packs, in rank order."""
    hits = retrieve(
        store, query,
        token_budget=DEFAULT_HOOK_TOKEN_BUDGET,
        manifest_reference_locks=True,
        belief_cost_fn=_ups_belief_line_cost,
        user_locks=(),
    )
    return [b.id for b in hits if b.lock_level == LOCK_NONE]


def run_arm(
    src: Path, prompts: Sequence[Mapping[str, Any]], arm: str, work: Path,
) -> dict[str, Any]:
    """One arm's packed non-locked ids per prompt, from a fresh copy of `src`."""
    if arm not in ARMS:
        raise InputError(f"unknown arm {arm!r}")
    db = prepare_copy(src, work)
    out: dict[str, Any] = {"arm": arm, "top_k": TOP_K}
    if arm in WRITER_ARMS:
        out["writers"] = apply_writers(db)
    expander = (
        PMIExpander([c for _i, c, _l in active_beliefs(db)]) if arm == ARM_A3 else None
    )
    rows: list[dict[str, Any]] = []
    store = MemoryStore(str(db), read_only=True)
    try:
        with arm_environment(arm, work):
            for p in prompts:
                query = hook_query(p)
                row: dict[str, Any] = {"prompt_id": p["prompt_id"], "prompt": p["text"]}
                if expander is not None:
                    added = expander.expand(str(p["text"]))
                    row["added_terms"] = added
                    if added:
                        query = f"{query} {' '.join(added)}"
                row["ranked"] = retrieve_packed(store, query)
                rows.append(row)
    finally:
        store.close()
    out["prompts"] = rows
    return out


def read_run(path: Path) -> dict[str, Any]:
    """A `run` output file; InputError when malformed."""
    obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise InputError(f"{path}: not a JSON object")
    run = cast(dict[str, Any], obj)
    prompts: object = run.get("prompts")
    if run.get("arm") not in ARMS or not isinstance(prompts, list):
        raise InputError(f"{path}: need an arm of {', '.join(ARMS)} and a prompts list")
    for p in cast(list[object], prompts):
        if not isinstance(p, dict):
            raise InputError(f"{path}: a prompt row is not an object")
        row = cast(dict[str, Any], p)
        ranked: object = row.get("ranked")
        if (
            not isinstance(row.get("prompt_id"), str) or not isinstance(row.get("prompt"), str)
            or not isinstance(ranked, list)
            or not all(isinstance(b, str) for b in cast(list[object], ranked))
        ):
            raise InputError(f"{path}: need prompt_id, prompt, and ranked ids")
    return run


def read_runs(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """{arm: run}; InputError unless the arms are distinct, include A0, and
    cover the same prompts."""
    runs: dict[str, dict[str, Any]] = {}
    for path in paths:
        run = read_run(path)
        if run["arm"] in runs:
            raise InputError(f"two runs for arm {run['arm']}")
        runs[run["arm"]] = run
    if CONTROL_ARM not in runs:
        raise InputError(f"no {CONTROL_ARM} run")
    ids = {a: [p["prompt_id"] for p in r["prompts"]] for a, r in runs.items()}
    ref = sorted(ids[CONTROL_ARM])
    if any(sorted(v) != ref for v in ids.values()):
        raise InputError("the runs cover different prompts")
    return runs


# --- Packet -------------------------------------------------------------------


def item_id(seed: int, pid: str, belief_id: str) -> str:
    """Packet row id from the seed, prompt id, and belief id only."""
    key = f"{seed}\0{pid}\0{belief_id}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def shares_prompt_stem(prompt: str, belief: str) -> bool:
    """True when the belief holds at least one of the prompt's stems."""
    return bool(set(tokenize_stemmed(prompt)) & set(tokenize_stemmed(belief)))


def build_packet(
    runs: Mapping[str, Mapping[str, Any]], db: Path, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`(packet rows, key rows)`, both in item-id order.

    Per prompt, the union of every arm's top 10 and, for each pair in
    `COMPARISONS`, the beliefs one arm packed and the other didn't. Each
    judged belief's key row holds its rank in every arm that packed it,
    past the top 10 included. A null arm of active non-locked beliefs no
    arm packed for the prompt is mixed in: as many per prompt as the
    median judged count (the lower median), drawn with `seed` in
    prompt-id order.
    """
    prompts: dict[str, str] = {}
    packed: dict[str, dict[str, dict[str, int]]] = {}
    for arm in sorted(runs):
        for p in runs[arm]["prompts"]:
            pid = str(p["prompt_id"])
            prompts[pid] = str(p["prompt"])
            slot = packed.setdefault(pid, {})
            for rank, bid in enumerate(p["ranked"], 1):
                slot.setdefault(str(bid), {})[arm] = rank
    if not prompts:
        raise InputError("the runs hold no prompts")
    judged: dict[str, dict[str, dict[str, int]]] = {}
    for pid in prompts:
        ranks = packed[pid]
        keep = {b for b, arms in ranks.items() if min(arms.values()) <= TOP_K}
        for x, c in COMPARISONS:
            if x in runs and c in runs:
                keep |= {b for b, arms in ranks.items() if (x in arms) != (c in arms)}
        judged[pid] = {b: ranks[b] for b in keep}
    null_n = int(statistics.median_low(len(judged[pid]) for pid in prompts))
    pool = [bid for bid, _c, lock in active_beliefs(db) if lock == LOCK_NONE]
    rng = random.Random(seed)
    entries: list[tuple[str, str, dict[str, int] | None]] = []
    for pid in sorted(prompts):
        retrieved = judged[pid]
        for bid in sorted(retrieved):
            entries.append((pid, bid, retrieved[bid]))
        candidates = [b for b in pool if b not in packed[pid]]
        for bid in sorted(rng.sample(candidates, min(null_n, len(candidates)))):
            entries.append((pid, bid, None))
    texts = belief_texts(db, [bid for _p, bid, _a in entries])
    packet: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for pid, bid, arms in entries:
        iid = item_id(seed, pid, bid)
        packet.append({"id": iid, "prompt": prompts[pid], "belief": texts[bid]})
        key.append({
            "id": iid, "prompt_id": pid, "prompt": prompts[pid], "belief_id": bid,
            "arms": None if arms is None else dict(sorted(arms.items())),
            "shares_prompt_stem": shares_prompt_stem(prompts[pid], texts[bid]),
        })
    if len({r["id"] for r in packet}) != len(packet):
        raise InputError("packet id collision; pick another seed")
    packet.sort(key=lambda r: r["id"])
    key.sort(key=lambda r: r["id"])
    return packet, key


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def write_packet(packet: Sequence[Mapping[str, Any]], key: Sequence[Mapping[str, Any]],
                 out: Path) -> None:
    """Write packet.jsonl, instructions.txt, and key.jsonl to `out`."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "packet.jsonl").write_text(_jsonl(packet), encoding="utf-8")
    (out / "instructions.txt").write_text(INSTRUCTIONS, encoding="utf-8")
    (out / "key.jsonl").write_text(_jsonl(key), encoding="utf-8")


# --- Usage proxy --------------------------------------------------------------


def _find_turn(turns: Sequence[Turn], prompt: Mapping[str, Any]) -> int | None:
    """Index of the prompt's turn: by turn id, else by session, timestamp, and text."""
    tid = prompt.get("turn_id")
    for i, t in enumerate(turns):
        if tid is not None and t.turn_id == tid and t.session_id == prompt.get("session_id"):
            return i
    for i, t in enumerate(turns):
        if (t.role == "user" and t.session_id == prompt.get("session_id")
                and t.ts == prompt.get("ts") and t.text == prompt.get("text")):
            return i
    return None


def proxy_used(belief_id: str, belief_stems: frozenset[str],
               later: Sequence[tuple[str, frozenset[str]]]) -> bool:
    """True when a later turn cites the id or holds `PROXY_OVERLAP_BAR` of the stems."""
    for text, stems in later:
        if belief_id in text:
            return True
        if belief_stems and len(belief_stems & stems) / len(belief_stems) >= PROXY_OVERLAP_BAR:
            return True
    return False


def proxy(
    runs: Mapping[str, Mapping[str, Any]], prompts: Sequence[Mapping[str, Any]],
    turns: Sequence[Turn], db: Path,
) -> dict[str, Any]:
    """Per prompt and arm, whether a later turn of the session used each belief."""
    by_pid = {str(p["prompt_id"]): p for p in prompts}
    retrieved: dict[str, dict[str, list[str]]] = {}
    for arm in sorted(runs):
        for p in runs[arm]["prompts"]:
            pid = str(p["prompt_id"])
            if pid not in by_pid:
                raise InputError(f"prompt {pid} is not in the prompts file")
            retrieved.setdefault(pid, {})[arm] = [str(b) for b in p["ranked"]]
    texts = belief_texts(
        db, [b for arms in retrieved.values() for ids in arms.values() for b in ids],
    )
    stems = {bid: frozenset(tokenize_stemmed(t)) for bid, t in texts.items()}
    rows: list[dict[str, Any]] = []
    for pid in sorted(retrieved):
        prompt = by_pid[pid]
        idx = _find_turn(turns, prompt)
        sid = prompt.get("session_id")
        later: list[tuple[str, frozenset[str]]] = []
        if idx is not None and sid is not None:
            later = [(t.text, frozenset(tokenize_stemmed(t.text)))
                     for t in turns[idx + 1:] if t.session_id == sid]
        rows.append({
            "prompt_id": pid, "found": idx is not None, "later_turns": len(later),
            "arms": {
                arm: [{"id": b, "used": proxy_used(b, stems[b], later)} for b in ids]
                for arm, ids in sorted(retrieved[pid].items())
            },
        })
    return {"overlap_bar": PROXY_OVERLAP_BAR, "prompts": rows}


# --- Score --------------------------------------------------------------------


@dataclass(frozen=True)
class KeyItem:
    id: str
    prompt_id: str
    arms: Mapping[str, int] | None
    shares_prompt_stem: bool


def read_key(path: Path) -> list[KeyItem]:
    """Rows of a key file; InputError on a malformed row or a key without A0."""
    items: list[KeyItem] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj: object = json.loads(line)
        if not isinstance(obj, dict):
            raise InputError(f"{path}:{n}: not a JSON object")
        row = cast(dict[str, Any], obj)
        arms: object = row.get("arms")
        ok_arms = arms is None or (
            isinstance(arms, dict) and all(
                a in ARMS and isinstance(r, int) and not isinstance(r, bool)
                for a, r in cast(dict[str, object], arms).items()
            )
        )
        if (not isinstance(row.get("id"), str) or not isinstance(row.get("prompt_id"), str)
                or not ok_arms or not isinstance(row.get("shares_prompt_stem"), bool)):
            raise InputError(f"{path}:{n}: need id, prompt_id, arms, and shares_prompt_stem")
        items.append(KeyItem(
            id=row["id"], prompt_id=row["prompt_id"],
            arms=None if arms is None else dict(cast(dict[str, int], arms)),
            shares_prompt_stem=row["shares_prompt_stem"],
        ))
    if not items:
        raise InputError(f"{path}: empty key")
    if not any(i.arms and CONTROL_ARM in i.arms for i in items):
        raise InputError(f"{path}: no {CONTROL_ARM} item")
    return items


def paired_bootstrap_ci(gains: Sequence[float], seed: int,
                        resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    """Percentile 95% interval of the mean paired gain, resampling prompts."""
    n = len(gains)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(gains[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    lo = means[int(math.floor(CI_LOW_Q * (resamples - 1)))]
    hi = means[int(math.ceil(CI_HIGH_Q * (resamples - 1)))]
    return lo, hi


def at_least(value: float, bar: float) -> bool:
    """`value >= bar`, counting a value within `BAR_TOLERANCE` of the bar as on it."""
    return value >= bar - BAR_TOLERANCE


def at_most(value: float, bar: float) -> bool:
    """`value <= bar`, counting a value within `BAR_TOLERANCE` of the bar as on it."""
    return value <= bar + BAR_TOLERANCE


def gain_verdict(mean: float, ci: tuple[float, float], share: float) -> str:
    """H2/H3 bars: kill when the mean is under `GAIN_KILL` or the interval holds 0."""
    if mean < GAIN_KILL or ci[0] <= 0.0 <= ci[1]:
        return V_FAIL
    if mean >= GAIN_BAR and share >= GAINING_SHARE_BAR:
        return V_PASS
    return V_INCONCLUSIVE


def combine(verdicts: Sequence[str]) -> str:
    """Both graders must pass; either one failing fails."""
    if V_INSUFFICIENT in verdicts:
        return V_INSUFFICIENT
    if V_FAIL in verdicts:
        return V_FAIL
    if all(v == V_PASS for v in verdicts):
        return V_PASS
    return V_INCONCLUSIVE


def _top(items: Sequence[KeyItem], arm: str) -> dict[str, list[KeyItem]]:
    """{prompt id: the arm's top `TOP_K` items}."""
    out: dict[str, list[KeyItem]] = {}
    for i in items:
        if i.arms is not None and i.arms.get(arm, TOP_K + 1) <= TOP_K:
            out.setdefault(i.prompt_id, []).append(i)
    return out


def _only(items: Sequence[KeyItem], arm: str, other: str) -> dict[str, list[KeyItem]]:
    """{prompt id: the judged items `arm` packed and `other` didn't}."""
    out: dict[str, list[KeyItem]] = {}
    for i in items:
        if i.arms is not None and arm in i.arms and other not in i.arms:
            out.setdefault(i.prompt_id, []).append(i)
    return out


def _relevant(items: Sequence[KeyItem], labels: Labels) -> int:
    return sum(1 for i in items if labels[i.id] is True)


def arm_figures(
    items: Sequence[KeyItem], arm: str, labels: Labels, prompts: Sequence[str], seed: int,
    control: str = CONTROL_ARM,
) -> dict[str, Any]:
    """One grader's paired gain, precision share, and drop for `arm` against `control`.

    The gain is counted over the whole packed result (see `COMPARISONS`):
    relevant beliefs only `arm` packed minus those only `control` packed.
    `top10_mean_gain` counts the top 10s alone. Precision and its drop
    are over the top 10s, as H5 states.
    """
    top_arm, top_a0 = _top(items, arm), _top(items, control)
    gained, lost = _only(items, arm, control), _only(items, control, arm)
    gains = [
        float(_relevant(gained.get(p, []), labels) - _relevant(lost.get(p, []), labels))
        for p in prompts
    ]
    top10_gains = [
        _relevant(top_arm.get(p, []), labels) - _relevant(top_a0.get(p, []), labels)
        for p in prompts
    ]
    mean = sum(gains) / len(gains)
    ci = paired_bootstrap_ci(gains, seed)
    share = sum(1 for g in gains if g >= 1) / len(gains)
    fig_arm = grader_figures([i.id for p in prompts for i in top_arm.get(p, [])], labels)
    fig_a0 = grader_figures([i.id for p in prompts for i in top_a0.get(p, [])], labels)
    g_arm, g_a0 = int(fig_arm["graded"]), int(fig_a0["graded"])
    precision, precision_a0 = float(fig_arm["precision"]), float(fig_a0["precision"])
    sufficient = g_arm >= MIN_GRADED and g_a0 >= MIN_GRADED
    drop = precision_a0 - precision
    figures: dict[str, Any] = {
        "graded": g_arm, "graded_a0": g_a0,
        "mean_gain": mean, "ci95": list(ci), "share_gaining": share,
        "top10_mean_gain": sum(top10_gains) / len(top10_gains),
        "precision": precision, "precision_a0": precision_a0, "precision_drop": drop,
        "gain_verdict": gain_verdict(mean, ci, share) if sufficient else V_INSUFFICIENT,
        "h5_verdict": (
            (V_PASS if at_most(drop, PRECISION_DROP_BAR) else V_FAIL) if sufficient else V_INSUFFICIENT
        ),
    }
    if arm == ARM_A3:
        additions = [i for p in prompts for i in gained.get(p, []) if labels[i.id] is True]
        bridging = sum(1 for i in additions if not i.shares_prompt_stem)
        share_b = bridging / len(additions) if additions else None
        figures.update({"relevant_additions": len(additions), "bridging": bridging,
                        "bridging_share": share_b})
        # A positive gain needs a relevant addition, so a passing gain
        # always has a bridging share.
        v = figures["gain_verdict"]
        if v not in (V_INSUFFICIENT, V_FAIL) and share_b is not None and (
                share_b < BRIDGING_SHARE_BAR):
            v = V_FAIL
        figures["gain_verdict"] = v
    return figures


def null_figures(items: Sequence[KeyItem], labels: Labels) -> dict[str, Any]:
    """One grader's relevant rate on the null arm, with its Wilson interval."""
    fig = grader_figures([i.id for i in items if i.arms is None], labels)
    graded, called = int(fig["graded"]), int(fig["correct"])
    lo, hi = wilson_interval(called, graded)
    return {"graded": graded, "called_relevant": called,
            "rate": float(fig["precision"]), "wilson_lower": lo, "wilson_upper": hi}


def proxy_gain_signs(proxy_report: Mapping[str, Any], arms: Sequence[str]) -> dict[str, int]:
    """Sign of the proxy's mean paired gain in used beliefs, per arm against A0."""
    rows = cast(list[dict[str, Any]], proxy_report.get("prompts", []))
    out: dict[str, int] = {}
    for arm in arms:
        gains = [
            sum(1 for b in r["arms"].get(arm, []) if b["used"])
            - sum(1 for b in r["arms"].get(CONTROL_ARM, []) if b["used"])
            for r in rows
        ]
        mean = sum(gains) / len(gains) if gains else 0.0
        out[arm] = (mean > 0) - (mean < 0)
    return out


def score(
    items: Sequence[KeyItem], a: Labels, b: Labels, *, seed: int,
    proxy_report: Mapping[str, Any] | None = None,
    census_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """H0 to H6 from the key, two graders' labels, and the optional proxy and census.

    Each grader is scored on its own labels, and an arm passes only when
    both graders pass it. A grader with fewer than `MIN_GRADED` graded
    items in the arm's or A0's top 10 makes the result `insufficient`. A
    failed H0 makes every arm result, H6 included, `unreliable`.

    H2 is a lane verdict only when the census passed H1. Under #1795's H1
    kill condition a failed H1 reports H2 as `substrate-starved`, and with
    no writers census H2 is `needs census`. The arm's own figures and
    verdict stay under `arms`.

    With an A1W run, `H2_split` splits H2's gain into the writers alone
    (A1W against A0) and the lane given the writers (A1 against A1W),
    under the same H0 and H1 rules as H2. A1's figures against A1W are
    under `arms["A1"]["vs_A1W"]`.
    """
    ids = [i.id for i in items]
    require_labels(ids, a, b, "key items")
    prompts = sorted({i.prompt_id for i in items})
    arms = sorted({arm for i in items if i.arms for arm in i.arms} - {CONTROL_ARM})
    kappa, kappa_n, agreed = paired_kappa(ids, a, b)
    null = [null_figures(items, a), null_figures(items, b)]
    reliable = at_least(kappa, KAPPA_BAR) and all(
        n["graded"] > 0 and n["rate"] <= NULL_RATE_CEILING for n in null
    )
    # H0's kill condition: when either bar fails, every arm result is
    # `unreliable`, whatever the graded counts.
    per_arm: dict[str, Any] = {}
    for arm in arms:
        graders = [arm_figures(items, arm, labels, prompts, seed) for labels in (a, b)]
        gain = combine([g["gain_verdict"] for g in graders])
        h5 = combine([g["h5_verdict"] for g in graders])
        if not reliable:
            gain = h5 = V_UNRELIABLE
        per_arm[arm] = {"graders": graders, "gain_verdict": gain, "h5_verdict": h5}
    if ARM_A1 in per_arm and ARM_A1W in per_arm:
        lane = [arm_figures(items, ARM_A1, labels, prompts, seed, control=ARM_A1W)
                for labels in (a, b)]
        per_arm[ARM_A1]["vs_A1W"] = {
            "graders": lane,
            "gain_verdict": (combine([g["gain_verdict"] for g in lane]) if reliable
                             else V_UNRELIABLE),
        }
    hyp: dict[str, Any] = {"H0": V_PASS if reliable else V_FAIL}
    hyp["H1"] = (census_report or {}).get("H1", V_NOT_MEASURED)
    for h, arm in HYPOTHESIS_ARMS.items():
        hyp[h] = per_arm[arm]["gain_verdict"] if arm in per_arm else V_NOT_MEASURED
    if ARM_A1 in per_arm and reliable and hyp["H1"] != V_PASS:
        hyp["H2"] = V_STARVED if hyp["H1"] == V_FAIL else V_NEEDS_CENSUS
    if ARM_A1 in per_arm and ARM_A1W in per_arm:
        split = {"writers": per_arm[ARM_A1W]["gain_verdict"],
                 "lane": per_arm[ARM_A1]["vs_A1W"]["gain_verdict"]}
        if reliable and hyp["H1"] != V_PASS:
            split = {k: hyp["H2"] for k in split}
        hyp["H2_split"] = split
    else:
        hyp["H2_split"] = V_NOT_MEASURED
    hyp["H5"] = {arm: per_arm[arm]["h5_verdict"] for arm in arms}
    if proxy_report is None:
        hyp["H6"] = V_NOT_MEASURED
    else:
        signs = proxy_gain_signs(proxy_report, arms)
        h6: dict[str, Any] = {}
        for arm in arms:
            grader_signs = [(g["mean_gain"] > 0) - (g["mean_gain"] < 0)
                            for g in per_arm[arm]["graders"]]
            agree = all(s == signs[arm] for s in grader_signs)
            h6[arm] = {"proxy_sign": signs[arm], "grader_signs": grader_signs,
                       "verdict": V_UNRELIABLE if not reliable
                       else V_AGREE if agree else V_DISAGREE}
        hyp["H6"] = h6
    return {
        "prompts": len(prompts), "items": len(ids),
        "null_items": sum(1 for i in items if i.arms is None),
        "kappa": kappa, "kappa_n": kappa_n, "agreed": agreed, "null_arm": null,
        "arms": per_arm, "hypotheses": hyp,
        "bars": {
            "kappa": KAPPA_BAR, "null_rate_ceiling": NULL_RATE_CEILING,
            "substrate_per_100": SUBSTRATE_BAR, "gain": GAIN_BAR, "gain_kill": GAIN_KILL,
            "gaining_share": GAINING_SHARE_BAR, "bridging_share": BRIDGING_SHARE_BAR,
            "precision_drop": PRECISION_DROP_BAR, "min_graded": MIN_GRADED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": seed,
        },
    }


# --- CLI ----------------------------------------------------------------------


class _Parser(argparse.ArgumentParser):
    """An argument parser that exits 1, not 2, on a usage error."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _read_json(path: Path) -> dict[str, Any]:
    obj: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise InputError(f"{path}: not a JSON object")
    return cast(dict[str, Any], obj)


def _dump(obj: object, out: Path | None) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if out is None:
        sys.stdout.write(text)
    else:
        out.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True, parser_class=_Parser)

    def work_dir(p: argparse.ArgumentParser) -> None:
        p.add_argument("--work-dir", type=Path,
                       help="where the temporary store copy goes (default: system temp)")

    c = sub.add_parser("census", help="edge census of store copies")
    c.add_argument("--store", type=Path, nargs="+", required=True)
    c.add_argument("--writers", action="store_true", help="run the A1 writers on each copy first")
    work_dir(c)
    s = sub.add_parser("sample", help="seeded prompt sample from the store's transcripts")
    s.add_argument("--store", type=Path, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--max", type=int, default=DEFAULT_MAX_PROMPTS)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="one arm's packed beliefs per prompt")
    r.add_argument("--store", type=Path, required=True)
    r.add_argument("--prompts", type=Path, required=True)
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--out", type=Path, required=True)
    work_dir(r)
    k = sub.add_parser("packet", help="blind grader packet and hidden key")
    k.add_argument("--runs", type=Path, nargs="+", required=True)
    k.add_argument("--store", type=Path, required=True)
    k.add_argument("--seed", type=int, required=True)
    k.add_argument("--out", type=Path, required=True)
    work_dir(k)
    sc = sub.add_parser("score", help="H0 to H6 from two graders' labels")
    sc.add_argument("--key", type=Path, required=True)
    sc.add_argument("--labels", type=Path, nargs=2, required=True)
    sc.add_argument("--proxy", type=Path)
    sc.add_argument("--census", type=Path, help="census output, for H1")
    sc.add_argument("--seed", type=int, default=1795, help="bootstrap seed")
    p = sub.add_parser("proxy", help="usage proxy from later turns")
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    work_dir(p)
    return parser


def _dispatch(args: argparse.Namespace) -> None:
    cmd = cast(str, args.cmd)
    if cmd == "census":
        _dump(census(args.store, args.work_dir, writers=args.writers), None)
    elif cmd == "sample":
        if not args.store.is_file():
            raise InputError(f"no store at {args.store}")
        rows = sample_prompts(read_turns(transcript_files(args.store)), args.seed, args.max)
        args.out.write_text(_jsonl(rows), encoding="utf-8")
        print(f"sampled={len(rows)} seed={args.seed}")
    elif cmd == "run":
        prompts = read_prompts(args.prompts)
        with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
            _dump(run_arm(args.store, prompts, args.arm, Path(work)), args.out)
    elif cmd == "packet":
        runs = read_runs(args.runs)
        with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
            packet, key = build_packet(runs, prepare_copy(args.store, Path(work)), args.seed)
        write_packet(packet, key, args.out)
        print(f"items={len(packet)} null={sum(1 for r in key if r['arms'] is None)}")
    elif cmd == "score":
        report = score(
            read_key(args.key), read_labels(args.labels[0]), read_labels(args.labels[1]),
            seed=args.seed,
            proxy_report=_read_json(args.proxy) if args.proxy else None,
            census_report=_read_json(args.census) if args.census else None,
        )
        _dump(report, None)
    else:
        runs = read_runs(args.runs)
        prompts = read_prompts(args.prompts)
        turns = read_turns(transcript_files(args.store))
        with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
            report = proxy(runs, prompts, turns, prepare_copy(args.store, Path(work)))
        _dump(report, args.out)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _dispatch(args)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
