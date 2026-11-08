"""The run half of the automatic wonder pass, for the replay harness (#1811).

The pass reads a session's prompts, picks the one the store covers least,
asks the host CLI's smallest model about it with the store's own snippets
(`wonder.auto_prompt`), and writes the claims it accepts as phantoms. The
replay (`benchmarks/wonder_auto_replay.py`) runs it on store copies with the
#1741 labeler's runner: `core_gate_background.runner_argv`, `runner_env`, and
`run_bounded`, with its own system prompt and a $0.10 cap.

Measured that way, the pass failed its pre-registered bars, and the operator
ruled that it doesn't ship. Nothing starts it on its own: no hook, setting,
or command calls this module. It stays so a later version can be measured
the same way.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Final, cast

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore
    from aelfrice.wonder.auto_prompt import AutoPrompt
    from aelfrice.wonder.dispatch import DispatchPayload

RUNNER_BUDGET_USD: Final[str] = "0.10"
"""`--max-budget-usd` for one run (operator ruling on #1811)."""

RUNNER_SYSTEM_PROMPT: Final[str] = (
    "You write short research notes from text snippets. "
    "Follow the user's instructions exactly."
)
MAX_PROMPTS: Final[int] = 20
"""The newest prompts of the session that compete for the topic."""
ARCHIVE_FILES_READ: Final[int] = 5
"""Newest rotated turn logs read for the session's earlier prompts."""

OUTCOME_ACCEPTED: Final[str] = "accepted"
OUTCOME_RUN_FAILED: Final[str] = "run_failed"
OUTCOME_BAD_REPLY: Final[str] = "bad_reply"
OUTCOME_STALE_ANCHOR: Final[str] = "stale_anchor"

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def checked_cost(value: object) -> float | None:
    """`value` as dollars, or None unless it is a finite number of at least 0.

    `json.loads` reads `NaN` and `Infinity` as numbers, and a negative
    cost would hand budget back.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        cost = float(value)
    except OverflowError:
        return None
    return cost if math.isfinite(cost) and cost >= 0 else None


def session_prompts(turns: Path, session_id: str) -> list[str]:
    """The newest `MAX_PROMPTS` prompts of `session_id` worth researching.

    Reads the newest `ARCHIVE_FILES_READ` rotated logs beside `turns` and
    then `turns` itself, since a compaction moves the session's earlier
    turns to the archive. A prompt the prompt-shape gate (#674) sends to
    locks only, such as a system envelope or an acknowledgement, doesn't
    compete: the store never had a chance to cover it.
    """
    from aelfrice.context_rebuilder import read_recent_turns_aelfrice  # noqa: PLC0415
    from aelfrice.hook import _should_skip_bm25  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]
    from aelfrice.transcript_logger import ARCHIVE_DIRNAME  # noqa: PLC0415

    archive = sorted((turns.parent / ARCHIVE_DIRNAME).glob("turns-*.jsonl"))
    prompts: list[str] = []
    for path in (*archive[-ARCHIVE_FILES_READ:], turns):
        rows = read_recent_turns_aelfrice(path, 1_000_000, session_id=session_id)
        prompts.extend(t.text for t in rows if t.role == "user")
    return [p for p in prompts if not _should_skip_bm25(p)[0]][-MAX_PROMPTS:]


def pick_topic(
    store: "MemoryStore", prompts: list[str],
) -> tuple["DispatchPayload", "AutoPrompt"] | None:
    """The prompt with the weakest coverage that has something to ask.

    Each prompt, cut to the snippet length, is the query for
    `build_dispatch_payload`. The lowest `query_term_coverage` wins; a
    tie goes to the newer prompt. A prompt whose payload leaves no store
    axis or no snippet that shares a word with it can't be asked about,
    so it is passed over and costs no run.
    """
    from aelfrice.wonder.auto_prompt import build_auto_prompt, snippet  # noqa: PLC0415
    from aelfrice.wonder.dispatch import build_dispatch_payload  # noqa: PLC0415

    best: tuple[tuple[float, int], DispatchPayload, AutoPrompt] | None = None
    for i, raw in enumerate(prompts):
        query = snippet(raw)
        if not query:
            continue
        payload = build_dispatch_payload(store, query)
        prompt = build_auto_prompt(payload)
        if prompt is None:
            continue
        key = (payload.gap_analysis.query_term_coverage, -i)
        if best is None or key < best[0]:
            best = (key, payload, prompt)
    return None if best is None else (best[1], best[2])


def _reply_json(stdout: str | None) -> object:
    """The CLI's output as JSON, or None when it isn't.

    A number of 5,000 digits or more raises a plain `ValueError` under
    Python's integer limit, past `JSONDecodeError`, and deep nesting raises
    `RecursionError`.
    """
    try:
        return json.loads(stdout or "")
    except (ValueError, TypeError, RecursionError):
        return None


def reported_cost(stdout: str | None) -> float | None:
    """The `total_cost_usd` of the CLI's JSON output, or None when unusable."""
    out = _reply_json(stdout)
    if not isinstance(out, dict):
        return None
    return checked_cost(cast(dict[str, Any], out).get("total_cost_usd"))


def execute_run(
    prompt: "AutoPrompt",
    *,
    cli: str,
    record: dict[str, object],
    run: Runner | None = None,
    env: dict[str, str] | None = None,
) -> "subprocess.CompletedProcess[str] | None":
    """Run the CLI on `prompt` with the #1741 flags. None when it failed.

    A failure fills `record`'s outcome and error. A run that started may
    have spent up to its cap, so `record` holds the cap until `apply_reply`
    reads the CLI's own report; a CLI that never started has spent nothing.
    """
    from aelfrice import core_gate_background as background  # noqa: PLC0415

    runner = run if run is not None else background.run_bounded
    started = time.monotonic()
    record["cost_usd"] = float(RUNNER_BUDGET_USD)
    try:
        with background.settings_file() as settings:
            # `encoding`: the CLI writes UTF-8 (#1441). `errors`: a byte that
            # isn't UTF-8 must fail the reply and not the pass.
            proc = runner(
                background.runner_argv(
                    cli, settings,
                    system_prompt=RUNNER_SYSTEM_PROMPT, budget_usd=RUNNER_BUDGET_USD,
                ),
                input=prompt.text, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=background.RUNNER_TIMEOUT_SECONDS,
                env=background.runner_env(dict(env if env is not None else os.environ)),
            )
    except (OSError, subprocess.SubprocessError) as exc:
        if isinstance(exc, OSError):
            del record["cost_usd"]
        record["outcome"] = OUTCOME_RUN_FAILED
        record["error"] = type(exc).__name__
        return None
    record["seconds"] = round(time.monotonic() - started, 1)
    return proc


def apply_reply(
    store: "MemoryStore",
    session_id: str,
    prompt: "AutoPrompt",
    proc: "subprocess.CompletedProcess[str]",
    record: dict[str, object],
) -> None:
    """Fill `record` from the CLI's output for `prompt`, and write its phantoms.

    The half of a pass after the CLI returned: the cost it reported, the
    strict parse, and on `accepted` the phantom write. Other processes
    write to the store during the run, so the anchors are checked again
    and the phantoms written in one `BEGIN IMMEDIATE` transaction: a
    reply citing an anchor retired since the prompt was built is
    `stale_anchor` and writes nothing. The replay harness
    (`benchmarks/wonder_auto_replay.py`) calls this on store copies.
    """
    from aelfrice.wonder.auto_prompt import parse_claims, to_phantoms  # noqa: PLC0415
    from aelfrice.wonder.lifecycle import wonder_ingest  # noqa: PLC0415

    out = _reply_json(proc.stdout)
    cost = reported_cost(proc.stdout)
    if cost is not None:
        record["cost_usd"] = cost
    elif "total_cost_usd" in (proc.stdout or ""):
        # The record keeps the cap, and the replay's kill rule counts this.
        record["cost_unusable"] = True
    if proc.returncode != 0 or not isinstance(out, dict):
        record["outcome"] = OUTCOME_RUN_FAILED
        record["returncode"] = proc.returncode
        return
    reply = cast(dict[str, Any], out).get("result")
    if not isinstance(reply, str):
        record["outcome"] = OUTCOME_BAD_REPLY
        return
    try:
        results = parse_claims(reply, prompt)
    except ValueError as exc:
        # The parser's messages name positions and keys, never text.
        record["outcome"] = OUTCOME_BAD_REPLY
        record["reason"] = str(exc)
        return
    phantoms = to_phantoms(results)
    record["claims"] = sum(len(r.claims) for r in results)
    record["documents"] = len(results)
    anchors = {a for p in phantoms for a in p.constituent_belief_ids}
    with store.transaction(immediate=True):
        if any(store.get_belief(a) is None for a in anchors):
            record["outcome"] = OUTCOME_STALE_ANCHOR
            return
        # `wonder_ingest` skips a known (anchors, generator) key only, so a
        # text this pass already wrote, live or retired, is dropped first.
        known = store.auto_wonder_contents([p.content for p in phantoms])
        fresh = [p for p in phantoms if p.content not in known]
        ingested = wonder_ingest(store, fresh, session_id=session_id)
    skipped = ingested.skipped + len(phantoms) - len(fresh)
    record["inserted"] = ingested.inserted
    record["skipped"] = skipped
    record["edges_created"] = ingested.edges_created
    record["outcome"] = OUTCOME_ACCEPTED
