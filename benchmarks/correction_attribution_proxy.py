"""#1291 / #1177 proposal 16 — free precision proxy for attributed correction.

Proposal 16 replaces the uniform valence smear in
`sentiment_feedback.apply_sentiment_to_pending` — which credits every belief
injected on the prior turn with the *same* valence — with winner-take-all
attribution plus an abstention rule. Its own cheapest kill experiment is a free
precision proxy: over real injection sets paired with the correction that
followed them, compute the distribution of max-Jaccard.

Pre-registered kill rule, quoted from the proposal and unchanged:

    KILL if the top candidate rarely clears theta=0.15, or clears it for many
    candidates at once. PROCEED only if a substantial share of correction turns
    yield exactly one clear winner with a margin over the runner-up.

**The corpus the proposal names cannot run it.** It specifies `hook_audit.jsonl`;
that file held 4 rows spanning one 7-minute window when this was written. The
population that can run it is the `injection_events` table joined to the session
transcript archive, which is what this script measures.

What it reports, in order:

  1. **Join feasibility** — how much of `injection_events` survives the join to
     a transcript, and where the rest goes.
  2. **The correction funnel** — how many user turns clear
     `sentiment_feedback.MAX_PROMPT_CHARS` and then trip `detect_sentiment`
     negative. This is the population proposal 16 acts on, and it is the number
     that decides whether the proxy is runnable at all.
  3. **max-Jaccard and |B*| distributions**, split at the 2026-06-30 regime
     break (`injection_events` is 2.1% non-locked before it and 50.3% after,
     so the two windows are different products and must not be pooled).

Both inputs are explicit arguments — nothing is read from a default home-
directory location. The store is opened **read-only**; point `--db` at a copy if
you want belt and braces. Output is aggregate statistics only: no prompt text
and no belief text is printed, written or retained.

Run:
    python benchmarks/correction_attribution_proxy.py \
        --db <store.db> --transcripts <dir-of-session-jsonl>
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from datetime import datetime
from pathlib import Path

from aelfrice.bm25 import tokenize_stemmed
from aelfrice.relationship_detector import _STOPWORDS
from aelfrice.sentiment_feedback import (
    MAX_PROMPT_CHARS,
    NEGATIVE,
    detect_sentiment,
)

# The proposal's theta. Not a tunable here — changing it invalidates the
# pre-registered rule this script exists to adjudicate.
THETA = 0.15

# `injection_events` is 2.1% non-locked before this date and 50.3% after
# (#1016-B landed reference-tier locks). Never pool across it.
REGIME_BREAK = "2026-06-30"

# An injection fires at UserPromptSubmit for the turn it belongs to, so the
# injection row and the user turn share a wall-clock instant to within hook
# latency. 30s is generous enough to absorb a slow hook and far tighter than
# the gap between consecutive turns.
ALIGN_TOLERANCE_S = 30.0


def content_tokens(text: str) -> set[str]:
    """The proposal's `content_tokens`: stemmed tokens minus stopwords."""
    return {t for t in tokenize_stemmed(text) if t not in _STOPWORDS}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def parse_ts(raw: str) -> float | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def load_injections(db: Path) -> tuple[dict[str, list], dict[str, set[str]]]:
    """Read-only. Returns (session -> [(t, injected_at, belief_ids)], tokens)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT session_id, turn_id, injected_at, belief_id "
            "FROM injection_events"
        ).fetchall()
        beliefs = dict(con.execute("SELECT id, content FROM beliefs").fetchall())
    finally:
        con.close()

    turns: dict[tuple[str, str], dict] = {}
    for sid, tid, at, bid in rows:
        slot = turns.setdefault(
            (sid, tid), {"t": parse_ts(at), "at": at, "ids": []}
        )
        slot["ids"].append(bid)

    by_session: dict[str, list] = {}
    for (sid, _tid), v in turns.items():
        if v["t"] is None:
            continue
        by_session.setdefault(sid, []).append((v["t"], v["at"], v["ids"]))
    for sid in by_session:
        by_session[sid].sort()

    return by_session, {
        bid: content_tokens(c) for bid, c in beliefs.items() if c
    }


def user_turns(path: Path) -> list[tuple[float, str]]:
    """`[(epoch, text)]` for genuine user prose turns, oldest first.

    Harness-generated entries (slash-command envelopes, tool results) are not
    user prose and would inflate the denominator of the correction funnel.
    """
    out: list[tuple[float, str]] = []
    with path.open(errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "user":
                continue
            msg = rec.get("message") or {}
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                text = "\n".join(
                    c.get("text", "")
                    for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            elif isinstance(content, str):
                text = content
            else:
                text = ""
            if not text.strip():
                continue
            if "<local-command" in text or "tool_use_id" in text:
                continue
            t = parse_ts(rec.get("timestamp", ""))
            if t is not None:
                out.append((t, text))
    out.sort()
    return out


def _describe(label: str, subset: list[dict]) -> None:
    print()
    print("=" * 70)
    print(f"{label}   n={len(subset)}")
    print("=" * 70)
    if not subset:
        print("  (empty)")
        return
    mx = sorted(r["max"] for r in subset)
    print(f"  |P| median            : "
          f"{statistics.median(r['p'] for r in subset):.0f}")
    print(f"  max-Jaccard           : min={mx[0]:.4f} median="
          f"{statistics.median(mx):.4f} p90={mx[int(len(mx) * 0.9)]:.4f} "
          f"max={mx[-1]:.4f}")
    clears = [r for r in subset if r["max"] >= THETA]
    print(f"  clears theta={THETA}      : {len(clears)}/{len(subset)} "
          f"({100 * len(clears) / len(subset):.1f}%)")
    if clears:
        nc = sorted(r["n_clear"] for r in clears)
        margins = [r["max"] - r["runner"] for r in clears]
        print(f"    |B*| median         : {statistics.median(nc):.0f} "
              f"(max {nc[-1]})")
        print(f"    exactly one clears  : "
              f"{sum(1 for r in clears if r['n_clear'] == 1)}/{len(clears)}")
        print(f"    winner-runner margin: {statistics.median(margins):.4f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, type=Path,
                    help="store containing injection_events (opened read-only)")
    ap.add_argument("--transcripts", required=True, type=Path,
                    help="directory searched recursively for <session-id>.jsonl")
    args = ap.parse_args(argv)

    if not args.db.is_file():
        print(f"no such store: {args.db}", file=sys.stderr)
        return 2
    if not args.transcripts.is_dir():
        print(f"no such directory: {args.transcripts}", file=sys.stderr)
        return 2

    by_session, tok = load_injections(args.db)
    index = {p.stem: p for p in args.transcripts.rglob("*.jsonl")}

    total_turns = sum(len(v) for v in by_session.values())
    print(f"beliefs with content       : {len(tok)}")
    print(f"sessions with injections   : {len(by_session)}")
    print(f"injection turns            : {total_turns}")

    joined = 0
    joined_injection_turns = 0
    user_turn_total = 0
    over_guard = 0
    fired = 0
    negative = 0
    records: list[dict] = []

    for sid, turns in by_session.items():
        tpath = index.get(sid)
        if tpath is None:
            continue
        joined += 1
        joined_injection_turns += len(turns)

        uturns = user_turns(tpath)
        user_turn_total += len(uturns)
        if not uturns:
            continue

        # Align each injection to the nearest user turn in time.
        by_time: dict[float, list[tuple[str, list[str]]]] = {}
        for t, at, ids in turns:
            best = min(uturns, key=lambda u: abs(u[0] - t))
            if abs(best[0] - t) <= ALIGN_TOLERANCE_S:
                by_time.setdefault(best[0], []).append((at, ids))

        for idx, (_t_now, text_now) in enumerate(uturns):
            if len(text_now) > MAX_PROMPT_CHARS:
                over_guard += 1
                continue
            signal = detect_sentiment(text_now)
            if signal is None:
                continue
            fired += 1
            if signal.sentiment != NEGATIVE:
                continue
            negative += 1

            # A correction at turn i is about the set injected at turn i-1.
            if idx == 0:
                continue
            prior = by_time.get(uturns[idx - 1][0])
            if not prior:
                continue
            at_prev, ids = prior[0]

            c_tok = content_tokens(text_now)
            scores = sorted(
                (jaccard(c_tok, tok[b]) for b in ids if b in tok), reverse=True
            )
            if not scores:
                continue
            records.append({
                "at": at_prev,
                "p": len(scores),
                "max": scores[0],
                "runner": scores[1] if len(scores) > 1 else 0.0,
                "n_clear": sum(1 for s in scores if s >= THETA),
            })

    print()
    print("=" * 70)
    print("JOIN FEASIBILITY")
    print("=" * 70)
    pct = 100 * joined_injection_turns / total_turns if total_turns else 0.0
    print(f"  sessions joined to a transcript : {joined}/{len(by_session)}")
    print(f"  injection turns reachable       : {joined_injection_turns}"
          f"/{total_turns} ({pct:.1f}%)")

    print()
    print("=" * 70)
    print("CORRECTION FUNNEL  (the population proposal 16 acts on)")
    print("=" * 70)
    denom = max(user_turn_total, 1)
    print(f"  user prose turns                : {user_turn_total}")
    print(f"  over MAX_PROMPT_CHARS ({MAX_PROMPT_CHARS})     : {over_guard} "
          f"({100 * over_guard / denom:.1f}%)  <- None before any pattern runs")
    print(f"  detect_sentiment fired          : {fired} "
          f"({100 * fired / denom:.2f}%)")
    print(f"  ...and NEGATIVE                 : {negative} "
          f"({100 * negative / denom:.2f}%)")
    print(f"  ...with a prior injected set    : {len(records)}")

    _describe("ALL", records)
    _describe(f"POST-BREAK (> {REGIME_BREAK})",
              [r for r in records if r["at"] > REGIME_BREAK])
    _describe(f"PRE-BREAK (<= {REGIME_BREAK})",
              [r for r in records if r["at"] <= REGIME_BREAK])

    print()
    print("=" * 70)
    print("READING THIS")
    print("=" * 70)
    print("  The verdict lives in the CORRECTION FUNNEL, not in the Jaccard")
    print("  tables. If the NEGATIVE count is small, the max-Jaccard rows are")
    print("  underpowered and must not be quoted as a precision result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
