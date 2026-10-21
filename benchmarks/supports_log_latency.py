"""Per-turn latency of the #1650 reworded-support log pass.

Builds a synthetic store of non-user beliefs, then ingests the same user
turns twice through ``ingest_jsonl``, once with
``AELFRICE_LOG_REWORDED_SUPPORT=0`` and once with ``=1``, each arm on a
fresh copy of the store. It reports the median and p99 wall time per
turn for each arm, the median and p99 of the pass alone
(``supports_log.log_reworded_support``) in the on arm, and the median
cost of resolving the flag with the env var unset, which ``ingest_jsonl``
pays once per call. Each arm ingests one untimed warm-up turn first.

Half the turns restate a stored belief with one word dropped, so the on
arm has matches to log; the run exits 1 if it logs none, because a pass
that matched nothing measures only the candidate query. The corpus is
pseudo-words, not a real-world distribution. Words are drawn Zipf-weighted
by default, so most are rare and each candidate search hits few beliefs;
``--uniform`` with a small ``--vocabulary`` draws every word equally
often, and ``--turn-words`` shortens each user turn. Together they make
each search hit many beliefs, the costly case. Every path the script
touches is under a fresh temporary directory, including ``HOME`` and
the store paths, so it never opens a real store, and the directory is
removed when the run ends.

Usage:

    uv run python benchmarks/supports_log_latency.py
    uv run python benchmarks/supports_log_latency.py --beliefs 10000 --turns 400
    uv run python benchmarks/supports_log_latency.py --vocabulary 120 --uniform --turn-words 5
    uv run python benchmarks/supports_log_latency.py --dry-run

Exit status: 0 on a completed run or a dry run, 1 when the on arm logs
no row.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

_SANDBOX = Path(tempfile.mkdtemp(prefix="supports-log-latency-"))
os.environ["HOME"] = str(_SANDBOX)
os.environ["AELFRICE_DB"] = str(_SANDBOX / "unused.db")
os.environ["AELFRICE_USER_DB"] = str(_SANDBOX / "unused-user.db")

import aelfrice.supports_log as supports_log  # noqa: E402
from aelfrice.ingest import ingest_jsonl  # noqa: E402
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief  # noqa: E402
from aelfrice.store import MemoryStore  # noqa: E402

_SYLLABLES = (
    "ka", "lo", "mi", "ne", "ru", "sa", "te", "vo", "zi", "pa", "do", "fe",
    "gu", "hi", "jo", "ba", "ce", "di", "ko", "la", "mu", "no", "pi", "ro",
)


def _vocabulary(rng: random.Random, size: int) -> list[str]:
    words: set[str] = set()
    while len(words) < size:
        words.add("".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 4))))
    return sorted(words)


def _sentence(rng: random.Random, vocab: list[str], weights: list[float]) -> str:
    # Weighted draw: Zipf by default, so a few words are common and most
    # are rare, or equal weights with --uniform.
    # Distinct words only: the matcher rejects a word typed 3 times.
    size = rng.randint(8, 14)
    words = list(dict.fromkeys(rng.choices(vocab, weights=weights, k=size * 3)))
    return " ".join(words[:size]).capitalize() + "."


def _build_store(path: Path, beliefs: list[str]) -> None:
    store = MemoryStore(str(path), read_only=False)  # a sandbox store it builds
    try:
        with store.transaction():
            for i, content in enumerate(beliefs):
                store.insert_belief(Belief(
                    id=f"b{i:06d}", content=content, content_hash=f"h{i:06d}",
                    alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
                    lock_level=LOCK_NONE, locked_at=None,
                    created_at="2026-10-01T00:00:00Z", last_retrieved_at=None,
                    session_id=f"s{i % 50}", origin="agent_inferred",
                ))
    finally:
        store.close()


def _turns(rng: random.Random, vocab: list[str], weights: list[float],
           beliefs: list[str], count: int, cap: int) -> list[str]:
    """User turns, alternating a restatement and an unrelated sentence.

    A `cap` above 0 keeps the first `cap` words of each turn. A prefix of
    a restatement is still contained in its belief, so it still matches.
    """
    turns: list[str] = []
    for i in range(count):
        if i % 2 == 0:
            words = rng.choice(beliefs).rstrip(".").split()
            del words[rng.randrange(1, len(words))]
        else:
            words = _sentence(rng, vocab, weights).rstrip(".").split()
        turns.append(" ".join(words[:cap] if cap > 0 else words) + ".")
    return turns


_WARM_UP = "The warm up turn loads every module before the timed turns start."


def _flag_check_us(work: Path, calls: int = 1000) -> float:
    """Median cost of one flag resolution with the env var unset.

    Resolved from `work`, which holds no `.aelfrice.toml`, so this times
    the full env-then-TOML walk that `ingest_jsonl` pays once per call.
    """
    os.environ.pop(supports_log.ENV_LOG_REWORDED_SUPPORT, None)
    samples: list[float] = []
    for _ in range(calls):
        start = time.perf_counter()
        supports_log.is_reworded_support_log_enabled(start=work)
        samples.append((time.perf_counter() - start) * 1_000_000)
    return round(statistics.median(samples), 2)


def _pct(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))]


def _arm(work: Path, template: Path, turns: list[str], flag: str) -> dict[str, object]:
    db = work / f"arm-{flag}.db"
    db.write_bytes(template.read_bytes())
    os.environ[supports_log.ENV_LOG_REWORDED_SUPPORT] = flag
    store = MemoryStore(str(db), read_only=False)  # a sandbox copy it writes
    pass_ms: list[float] = []
    real_pass: Callable[..., int] = supports_log.log_reworded_support

    def timed_pass(s: MemoryStore, pairs: tuple[tuple[str, str], ...]) -> int:
        start = time.perf_counter()
        try:
            return real_pass(s, pairs)
        finally:
            pass_ms.append((time.perf_counter() - start) * 1000)

    supports_log.log_reworded_support = timed_pass  # type: ignore[assignment]
    turn_ms: list[float] = []
    line = work / "turn.jsonl"
    try:
        # One untimed turn first, so imports and the first FTS query are
        # not charged to either arm.
        line.write_text(json.dumps({
            "role": "user", "text": _WARM_UP, "session_id": "bench-warm-up",
            "ts": "2026-10-02T00:00:00Z",
        }) + "\n")
        ingest_jsonl(store, line)
        pass_ms.clear()
        for i, text in enumerate(turns):
            line.write_text(json.dumps({
                "role": "user", "text": text, "session_id": "bench-user",
                "ts": f"2026-10-02T01:{i // 60 % 60:02d}:{i % 60:02d}Z",
            }) + "\n")
            start = time.perf_counter()
            ingest_jsonl(store, line)
            turn_ms.append((time.perf_counter() - start) * 1000)
        logged = store.count_support_evidence(applied=False)
    finally:
        supports_log.log_reworded_support = real_pass  # type: ignore[assignment]
        store.close()
    result: dict[str, object] = {
        "flag": flag, "turns": len(turn_ms), "logged_rows": logged,
        "turn_median_ms": round(statistics.median(turn_ms), 3),
        "turn_p99_ms": round(_pct(turn_ms, 99), 3),
    }
    if pass_ms:
        result["pass_calls"] = len(pass_ms)
        result["pass_median_ms"] = round(statistics.median(pass_ms), 3)
        result["pass_p99_ms"] = round(_pct(pass_ms, 99), 3)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--beliefs", type=int, default=10_000)
    parser.add_argument("--turns", type=int, default=400)
    parser.add_argument("--vocabulary", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=1650)
    parser.add_argument("--turn-words", type=int, default=0,
                        help="keep the first N words of each user turn "
                             "(0 keeps every word)")
    parser.add_argument("--uniform", action="store_true",
                        help="draw every word equally often, not Zipf-weighted")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit 0 without building a store")
    args = parser.parse_args(argv)
    if args.dry_run:
        draw = "uniform" if args.uniform else "Zipf-weighted"
        print(f"would build {args.beliefs} beliefs from {args.vocabulary} "
              f"{draw} pseudo-words (seed {args.seed}) under {_SANDBOX}, then time "
              f"{args.turns} user turns with the log off and on")
        return 0
    rng = random.Random(args.seed)
    vocab = _vocabulary(rng, args.vocabulary)
    weights = [1.0 if args.uniform else 1.0 / (rank + 1)
               for rank in range(len(vocab))]
    beliefs = [_sentence(rng, vocab, weights) for _ in range(args.beliefs)]
    turns = _turns(rng, vocab, weights, beliefs, args.turns, args.turn_words)
    template = _SANDBOX / "template.db"
    _build_store(template, beliefs)
    off = _arm(_SANDBOX, template, turns, "0")
    on = _arm(_SANDBOX, template, turns, "1")
    report = {
        "beliefs": args.beliefs, "turns": args.turns, "seed": args.seed,
        "vocabulary": args.vocabulary, "uniform": args.uniform,
        "turn_words": args.turn_words,
        "flag_resolution_median_us": _flag_check_us(_SANDBOX),
        "off": off, "on": on,
    }
    print(json.dumps(report, indent=2))
    if on["logged_rows"] == 0:
        print("FAIL: the on arm logged no row, so the pass matched nothing",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(_SANDBOX, ignore_errors=True)
