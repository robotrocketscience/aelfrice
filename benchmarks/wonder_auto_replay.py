"""Replay of the automatic wonder pass on store copies (#1811).

The automatic wonder pass (`aelfrice.wonder.auto_run`) asks the host CLI's
smallest model about one topic per session and writes the claims it returns
as phantoms. It was built to start from the Stop hook. #1811 pre-registered
H1 to H7 before any run; the paid replay killed it, so nothing starts it on
its own, and this harness stays to measure a later version the same way.
This harness measures them on read-only copies of real stores,
with a runner you choose, and starts no run past `TOTAL_CEILING_USD`. The
paragraph on what is bounded, under `replay`, says what that does and doesn't
promise.

**Steps.** Each subcommand checks its inputs, writes nothing with
`--dry-run`, exits 1 on bad input, and is deterministic given its inputs
and `--seed`, apart from what the runner replies.

- `prepare`: copies `<projects root>/<name>/.git/aelfrice/memory.db` and
  the transcripts beside it to `--out/<name>/`, through a `mode=ro` handle.
  It skips every `--exclude` project by name, and any project with a link
  that resolves into one, and reads no store or transcript in it. It refuses
  a copy that would leave less free disk than the copies need. The names
  come from the command line because they are private to the person running
  it, and this file is public. It marks each finished copy with
  `COPY_MARK_FILE`, leaves a marked copy as it is, and stops at a `memory.db`
  with no mark, which it doesn't delete because it may be a real store.
- `replay`: refuses two `--store` values that are one file or hold the same
  bytes, before anything runs. For each store copy, the newest `--sessions`
  sessions the Stop hook would start the pass for (see Sessions), then the
  runner on each request with the writer's own flags and environment, then
  `run`. Requests go round the stores in a seeded order, so a budget stop
  still leaves several stores measured. The spend ledger is one file,
  `~/.aelfrice-replay/wonder-auto-spend.jsonl`, which no flag moves, so a new
  `--out` anywhere doesn't reset the total. It holds
  one charge per real start of the runner. Under a lock held across the
  check and the claim, a run starts only when everything the ledger records
  plus `RUN_RESERVE_USD` fits `--budget-usd`, at most `TOTAL_CEILING_USD`, and
  the ledger holds fewer than `--max-calls` calls. Its `started` row reserves its
  cap before the runner starts. A run that started and reported no cost that
  is a finite number of at least 0, such as one that timed out, failed, or
  reported NaN, charges `RUN_RESERVE_USD`, and a runner that never started
  charges nothing. Repeating the command with the same `--out` resumes: a
  request already run isn't run again, so raising `--max-calls` continues a
  replay that stopped on it. When the resume runs anything, it moves the
  earlier `run/` to `run-1` (then `run-2`, and so on) before it builds the new
  one, and when it runs nothing it leaves `run/` as it was. A run the harness died inside counts
  at `RUN_RESERVE_USD`, the most the ceiling allows a run, since nothing shows
  what it spent, and isn't run again either.

  What is bounded: the total stays at or under `--budget-usd` while each run
  costs at most `RUN_RESERVE_USD`, which is the $0.10 cap and 30% over it. The
  last run starts with at most `--budget-usd` minus `RUN_RESERVE_USD` spent,
  $0.87 at the $1.00 ceiling. What isn't: a run that costs more than that adds
  what it costs on top of those $0.87, and a runner that ignores its
  `--max-budget-usd` is bounded only by the runner timeout,
  `core_gate_background.RUNNER_TIMEOUT_SECONDS`. A run reporting less than
  its cap leaves room for more runs, and one reporting $0 uses none, so
  `--max-calls` is what bounds the number of runs. It counts every call the
  ledger holds, from every replay under this `HOME`, and its default is the
  runs `--budget-usd` admits at the cap: 9 at $1.00. Deleting the ledger
  resets the total and the count, and a different `HOME` reads a different
  ledger.

  Only one replay runs at a time, so only one runner does. A timeout, the
  runner's own exit, Ctrl-C, or a signal whose default ends the harness ends
  the runner's process group (see `core_gate_background.run_bounded`).
  SIGKILL of the harness can't: the runner keeps going with no timeout, and
  the ledger charges it at
  the cap whatever it spends. It holds the replay lock it inherited, so no
  second replay starts until it and its children exit. A child that leaves
  the process group is out of reach of the kill, and holds the lock the same way.

  Before the first run, and in a dry run, `replay` prints how many runs and
  claims the budget reaches against H2's `MIN_GRADED`, and warns when the
  target is out of reach. Each axis yields at most `MAX_CLAIMS` claims, a
  fully covered prompt offers one axis, and a reply may answer only one of
  the axes offered, so the claims are a range: one axis a run at the low end,
  every axis offered at the high end. The dry run also prints `--max-calls`
  and `--max`, which sets the real prompts sampled per store for retrieval.
- `requests` and `run`: the first and last halves of `replay` as their own
  commands. `run` takes saved replies (`{"request_id": ..., "stdout": <the
  CLI's stdout>, "returncode": <its exit code>}` per line), spends nothing,
  and so can rebuild the output from what `replay` recorded. For each
  store, two copies: on one, each reply goes through the writer's
  `apply_reply`, which parses, refuses, costs, and writes it as the live
  pass would. The other copy stays as it was. It samples the store's real
  prompts (`real_store_ab_1795.sample_prompts`), leaving out every prompt
  the pass read for a request, and retrieves for each from both copies
  (`real_store_ab_1795.retrieve_packed`).
- Grading retrieval: give graders a and c, the pair #1804 calibrated, one
  fresh reader each, the `retrieval/` directory only. Each scores every row
  from 0 to 3, or null, as `{"id": ..., "score": ...}` lines in one labels
  file per grader. A row is relevant when its score is 2 or higher (#1804,
  #1809). Neither grader sees the key, the arm, the other's labels, or
  `claims/` yet.
- `claims`: once both labels files score every retrieval row, writes
  `claims/`, the claims packet and its instructions, and records each
  labels file's digest. A claim's text can also be a retrieval row that
  only the phantom copy packed, so a grader who saw the claims first
  could tell H3's arms apart.
- Grading claims: give the same graders `claims/`. Each scores every row
  the same way into a new labels file. A claim is good at 2 or higher.
- `score`: H1 to H7 and the decision, from the run directory and both
  graders' labels for each packet. It refuses retrieval labels whose
  digests differ from the ones `claims` recorded.

**Output.** Under `replay --out`: `requests.jsonl`, `requests-run.jsonl`
(what the budget admitted), `replies.jsonl`, and `run/`. Under `run/`: `retrieval/` and later `claims/`, the only directories
a grader gets, each a `packet.jsonl` of opaque ids in id order with
`instructions.txt`; `phantoms.jsonl`, every claim and document with its axis,
anchors, and session, for you and not for graders; `key.jsonl`; `report.json`;
and `GRADING.txt`, these steps with this run's paths.

**Sessions.** The trigger the pass was built for starts it on the first
Stop after a transcript ingest landed for the session, and the transcript logger ingests
every `DEFAULT_STOP_FLUSH_TURNS` turn lines. A session counts when the store
holds an ingest row from it and it has at least `HORIZON_TURNS` turn lines:
the flush at line 12, then one more exchange to the next Stop. The pass
reads the prompts among those lines only (`--horizon 0` reads the whole
session). Every prompt is built on the store as it is now, so a later
session's prompt doesn't see the phantoms an earlier one wrote, and the
snippets can include beliefs written after the session ended.

**Hypotheses** (#1811). H2, H3, and H4 must pass. A failure of any of
them, or a run that spent more than its cap, kills the pass. The rest
are reported.

- H1 supply: the share of eligible sessions that wrote at least one
  claim, and the share of requests refused as a bad reply, at most
  `PARSE_FAILURE_BAR`. A session is eligible when it is among the
  `--sessions` newest that the hook would start the pass for, and the
  budget admitted it, so one the pass found no prompt or no topic in
  counts as a session with no claim.
- H2 claim precision: each grader's share of good claims reaches
  `CLAIM_PRECISION_BAR`, and the graders' kappa reaches `KAPPA_BAR`.
  It fails when every grader is under the bar, whatever the kappa.
- H3 value: per sampled prompt, relevant beliefs only the phantom copy
  packed minus those only the unchanged copy packed; for each grader the
  mean reaches `VALUE_BAR` and its 95% interval lies above 0. The gain,
  interval, and `MIN_GRADED` rule are `real_store_ab_1795.arm_figures`.
  The sample holds out the prompts the pass read. It draws from the
  store's whole history, so a sampled prompt can predate the phantoms.
- H4 harm: each grader's relevant share of the top 10 drops at most
  `PRECISION_DROP_BAR` (`arm_figures`), and the share of the top 10 that
  is assistant narration (#785) is no higher with the phantoms. A belief
  is narration when every ingest row that derived it records the
  assistant role.
- H5 usage: in the phantom copy, how often a later turn of the session
  used a phantom hit against the other hits of the same prompts, by
  `real_store_ab_1795.proxy_used`.
- H6 conversion: phantoms written, phantoms evidence promotion could
  promote now, and the share wonder GC would keep at its TTL with no
  new support, a lower bound on day-14 survival. Support events need
  the live trial.
- H7 cost: per run from the CLI's reported cost, or the run's cap when
  it reported none, and per day projected from the busiest day's count
  of sessions. A reported cost that isn't a finite number of at least 0
  counts as a run over its cap.

H3 and H4's precision count only when the retrieval labels are reliable,
as #1795's H0 sets: kappa at least `KAPPA_BAR` and each grader's
null-arm rate at most `NULL_RATE_CEILING`.

**Data.** Copies, requests, replies, packets, keys, labels, and the
report's directory hold user text or ids from a real store. Every output
and `--work-dir` must sit outside every checkout of this repository, its
linked worktrees included, and only its owner can read it. Stdout gets
counts, never belief or prompt text. `--store` takes only a copy `prepare`
made, since a `[store]` marker can place a real store anywhere. It also
refuses a path under a `.git` directory or `~/.aelfrice`, in any letter
case, a file with a second hard link, and a path naming an `--exclude`
project.

Usage::

    uv run python -m benchmarks.wonder_auto_replay prepare --projects-root <projects> --exclude <name> --out <scratch>/stores
    uv run python -m benchmarks.wonder_auto_replay replay --store <scratch>/stores/*/memory.db --runner <cli> --out <scratch>/replay/one
    uv run python -m benchmarks.wonder_auto_replay claims --run <scratch>/replay/one/run --labels a.jsonl c.jsonl
    uv run python -m benchmarks.wonder_auto_replay score --run <scratch>/replay/one/run --labels a.jsonl c.jsonl --claim-labels a-claims.jsonl c-claims.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import Any, Final, cast

from aelfrice import core_gate_background as background
from aelfrice import file_lock
from aelfrice.wonder import auto_run as wb
from aelfrice.models import (
    AUTO_WONDER_GENERATOR_FAMILY,
    CORROBORATION_SOURCE_WONDER_INGEST,
    EDGE_RELATES_TO,
)
from aelfrice.phantom_promotion_opportunity import (
    find_evidence_promotable_phantoms,
    promotion_guarded_ids,
)
from aelfrice.store import MemoryStore, read_only_uri
from aelfrice.transcript_logger import DEFAULT_STOP_FLUSH_TURNS
from aelfrice.wonder.auto_prompt import DOCUMENT_KEY, MAX_CLAIMS, AutoPrompt
from benchmarks.graded_gate import MIN_GRADED, Labels, grader_figures, require_labels
from benchmarks.grader_calibration_1804 import (
    FIXED_CUT,
    KAPPA_BAR,
    NULL_RATE_CEILING,
    private_dir,
    write_private,
)
from benchmarks.real_store_ab_1795 import (
    ARM_A0,
    DEFAULT_MAX_PROMPTS,
    KEY_ITEM,
    KEY_STORE,
    PRECISION_DROP_BAR,
    TOP_K,
    V_FAIL,
    V_INCONCLUSIVE,
    V_INSUFFICIENT,
    V_NOT_MEASURED,
    V_PASS,
    V_UNRELIABLE,
    InputError,
    Key,
    KeyItem,
    KeyStore,
    Turn,
    _jsonl,  # pyright: ignore[reportPrivateUsage]
    _Parser,  # pyright: ignore[reportPrivateUsage]
    arm_environment,
    arm_figures,
    belief_texts,
    build_packet,
    combine,
    copy_store,
    exact_bar,
    exact_kappa,
    hook_query,
    make_key,
    null_figures,
    prepare_snapshot,
    proxy,
    read_turns,
    retrieve_packed,
    sample_prompts,
    transcript_files,
)
from benchmarks.relevance_scale import INSTRUCTIONS as RELEVANCE_INSTRUCTIONS
from benchmarks.relevance_scale import binarize, parse_scores

BASE_ARM: Final[str] = "base"
"""The store copy without the replayed phantoms."""
PHANTOM_ARM: Final[str] = "phantom"
"""The store copy with the replayed phantoms."""
ARMS_RUN: Final[tuple[str, ...]] = (BASE_ARM, PHANTOM_ARM)
COMPARISON: Final[tuple[tuple[str, str], ...]] = ((PHANTOM_ARM, BASE_ARM),)
"""The packet judges every belief one copy packed and the other didn't."""

DEFAULT_SEED: Final[int] = 1811
DEFAULT_SESSIONS: Final[int] = 10
"""Newest sessions per store that the Stop hook would start the pass for."""
DEFAULT_REPLAY_SESSIONS: Final[int] = 3
"""The same, for `replay`, where the budget runs out long before 10 per store."""
LIVE_DAILY_CEILING_USD: Final[float] = 0.50
"""The per-store daily ceiling the pass was built to run under (operator ruling
on #1811). H7 reports the busiest day's projected cost against it."""
TOTAL_CEILING_USD: Final[float] = 1.00
"""The most replays may spend, by operator ruling on #1811, across the one
spend ledger."""
RUN_RESERVE_USD: Final[float] = 0.13
"""What one more run may cost for the ceiling to hold: its $0.10 cap and 30% over
it. The CLI checks its cap between steps. In #1741's precision check, the shipped
configuration's costliest run was $0.0505 against a $0.05 cap, and two other
configurations reached $0.113 and $0.146 against the same cap, so 30% is a margin
and not a measured limit."""
HORIZON_TURNS: Final[int] = DEFAULT_STOP_FLUSH_TURNS + 2
"""Turn lines the pass has seen: the first ingest flush, then one exchange."""

# Pre-registered bars, from the #1811 hypothesis table.
PARSE_FAILURE_BAR: Final[float] = 0.10
"""H1, reported: share of requests refused as a bad reply."""
CLAIM_PRECISION_BAR: Final[float] = 0.70
"""H2: each grader's share of claims scored correct and not a restatement."""
VALUE_BAR: Final[float] = 0.1
"""H3: mean paired gain in relevant beliefs per prompt."""
COST_MEAN_BAR: Final[float] = 0.05
"""H7, reported: mean cost per run in dollars."""

KEY_CLAIM: Final[str] = "claim"
"""`kind` of a key row for one claims-packet row."""
ROW_STORE: Final[str] = "store"
ROW_REQUEST: Final[str] = "request"
CLAIMS_DIR: Final[str] = "claims"
RETRIEVAL_DIR: Final[str] = "retrieval"
PACKET_FILE: Final[str] = "packet.jsonl"
INSTRUCTIONS_FILE: Final[str] = "instructions.txt"
KEY_FILE: Final[str] = "key.jsonl"
REPORT_FILE: Final[str] = "report.json"
PHANTOMS_FILE: Final[str] = "phantoms.jsonl"
"""Every claim and document a run wrote, for the operator, never for a grader."""
GRADING_FILE: Final[str] = "GRADING.txt"
REQUESTS_FILE: Final[str] = "requests.jsonl"
ADMITTED_FILE: Final[str] = "requests-run.jsonl"
"""The requests the budget admitted, which `replay` hands to `run`."""
REPLIES_FILE: Final[str] = "replies.jsonl"
SPEND_DIR: Final[str] = ".aelfrice-replay"
SPEND_LEDGER_FILE: Final[str] = "wonder-auto-spend.jsonl"
SPEND_LOCK_FILE: Final[str] = "wonder-auto-spend.lock"
COPY_MARK_FILE: Final[str] = ".wonder-replay-copy"
"""Beside a store copy `prepare` finished. `--store` takes nothing without it."""
REPLAY_LOCK_FILE: Final[str] = "wonder-auto-replay.lock"
"""Held for a whole replay, beside the ledger, so one runner runs at a time."""
RUN_DIR: Final[str] = "run"
CLAIMS_HELD_FILE: Final[str] = "claims-held.jsonl"
"""The claims packet's rows, held back until both graders score every retrieval row."""
RELEASE_FILE: Final[str] = "claims-release.json"
"""The digests of the retrieval labels files the claims packet was released against."""

S_SHORT: Final[str] = "short"
S_NO_BELIEFS: Final[str] = "no_beliefs"
S_NO_PROMPTS: Final[str] = "no_prompts"
S_NO_TOPIC: Final[str] = "no_topic"
S_REQUEST: Final[str] = "request"
S_BUDGET_STOPPED: Final[str] = "budget_stopped"

STATE_STARTED: Final[str] = "started"
STATE_DONE: Final[str] = "done"
FAILED_RETURNCODE: Final[int] = -1
"""A saved reply's exit code when the runner never returned one."""

DECISION_KILL: Final[str] = "kill"
DECISION_PASS: Final[str] = "pass"
DECISION_OPEN: Final[str] = "inconclusive"

GC_ALL_CUTOFF: Final[str] = "9999-12-31T23:59:59+00:00"
"""A GC cutoff every phantom is older than: GC as it would run at the TTL."""

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

CLAIM_SCALE: Final[tuple[tuple[str, str], ...]] = (
    ("3", "Correct and new: it follows from its notes, and no single note states it."),
    ("2", "Correct and adds to its notes: it joins two of them or draws a consequence."),
    ("1", "Correct, but one of its notes already states it."),
    ("0", "Wrong, or its notes don't support it."),
    ("null", "You can't tell what the claim or the notes mean."),
)
"""The claims packet's scale. A score of 2 or higher is a good claim (H2)."""

CLAIM_INSTRUCTIONS: Final[str] = (
    "Each row of packet.jsonl holds one claim and the notes it cites, from a\n"
    "coding project's memory. A program wrote the claim from those notes.\n"
    "Score each row from 0 to 3, or null:\n\n"
    + "".join(f"  {score:<5} {meaning}\n" for score, meaning in CLAIM_SCALE)
    + "\nWrite one JSON object per line to your own labels file:\n"
    '  {"id": "<row id>", "score": 2}\n'
    "The score is 0, 1, 2, 3, or null. Score every row. Rows come in no\n"
    "particular order. Don't discuss the rows with the other grader.\n"
)


GRADING_TEXT: Final[str] = """\
Grade this run with graders a and c, the pair #1804 calibrated: one fresh
reader each, who see no key, no arm, and not each other's labels.

1. Give each grader {run}/retrieval/ and nothing else. Each writes one labels
   file, one {{"id": ..., "score": 0|1|2|3|null}} line per row.
2. Release the claims:
   uv run python -m benchmarks.wonder_auto_replay claims --run {run} --labels <a> <c>
3. Give the same graders {run}/claims/. Each writes a new labels file.
4. Score:
   uv run python -m benchmarks.wonder_auto_replay score --run {run} --labels <a> <c> --claim-labels <a-claims> <c-claims>

Keep key.jsonl, phantoms.jsonl, and report.json from the graders.
"""


# --- Output safety ------------------------------------------------------------


def repository_roots(root: Path) -> list[Path]:
    """Every checkout of the repository at `root`: itself, its main checkout, and its worktrees.

    Read from git's files: a linked worktree's `.git` file names its git
    dir, whose `commondir` names the shared git dir, and each
    `worktrees/*/gitdir` there names a linked worktree's `.git`. A root
    with no `.git` is its own only checkout.
    """
    roots = [root]
    dotgit = root / ".git"
    if dotgit.is_dir():
        common = dotgit
    elif dotgit.is_file():
        gitdir = root / dotgit.read_text(encoding="utf-8").strip().removeprefix("gitdir:").strip()
        link = gitdir / "commondir"
        common = (gitdir / link.read_text(encoding="utf-8").strip() if link.is_file()
                  else gitdir).resolve()
    else:
        return roots
    if common.name == ".git":
        roots.append(common.parent)
    for link in sorted((common / "worktrees").glob("*/gitdir")):
        try:
            roots.append(Path(link.read_text(encoding="utf-8").strip()).parent)
        except OSError:
            # A worktree removed since the listing has no checkout left to guard.
            continue
    return roots


def _same_folder(a: Path, b: Path) -> bool:
    """Whether `a` and `b` are one folder; False when either is gone."""
    try:
        return a.samefile(b)
    except OSError:
        return False


def require_outside_repo(path: Path, root: Path = REPO_ROOT) -> None:
    """InputError when `path`, as given or with links resolved, is in a checkout of `root`'s repository.

    A run from a linked worktree would otherwise write into the main
    checkout or a sibling worktree. Compared by file identity on the
    nearest existing directory, so a filesystem that folds case can't
    hide a checkout. A worktree whose directory is gone is skipped, even
    when it goes during the comparison.
    """
    roots = repository_roots(root)
    for p in (Path(os.path.abspath(path)), path.resolve()):
        for d in (p, *p.parents):
            if any(_same_folder(d, r) for r in roots):
                raise InputError(f"{path} is inside the repository; write it to a scratch folder")


def require_distinct_stores(stores: Sequence[Path]) -> None:
    """InputError when two `--store` values are one file or hold the same bytes.

    A request's id comes from its store's bytes and its session, so a repeated
    store would run each of its requests again and then fail to pair them up,
    after spending. A missing file passes here, and the command that wanted it
    says so.
    """
    seen: dict[str, Path] = {}
    for src in stores:
        if not src.is_file():
            continue
        with src.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest in seen:
            raise InputError(f"{src} and {seen[digest]} are one store; list it once")
        seen[digest] = src


def excluded_names(names: Sequence[str] | None) -> frozenset[str]:
    """`--exclude` values, compared without case, since macOS folds it on disk."""
    return frozenset(name.lower() for name in names or ())


def _under_folder(path: Path, folder: Path) -> bool:
    """Whether `path` is `folder` or under it, in any letter case, since macOS folds it."""
    return Path(str(path).lower()).is_relative_to(str(folder).lower())


def require_store_copy(path: Path, excluded: frozenset[str] = frozenset()) -> None:
    """InputError unless `path` is a copy of a store this harness may read.

    Checked on the path as given and with links resolved, before anything
    opens it, so an excluded project's store is never touched. A missing
    file passes here, and the command that wanted it says so. Only stat
    calls reach the file: a second hard link means another name may be the
    real store, and `COPY_MARK_FILE` shows `prepare` made it, which is the
    one check a store a `[store]` marker placed elsewhere can't pass.
    """
    home_store = Path.home() / ".aelfrice"
    for p in (Path(os.path.abspath(path)), path.resolve()):
        parts = {part.lower() for part in p.parts}
        if parts & excluded:
            raise InputError("that store belongs to an excluded project")
        if ".git" in parts or _under_folder(p, home_store) or _under_folder(p, home_store.resolve()):
            raise InputError(f"{path} is a real store; copy it with `prepare` first")
    if not path.is_file():
        return
    if path.stat().st_nlink > 1:
        raise InputError(f"{path} has another name on disk; copy it with `prepare` first")
    if not (path.resolve().parent / COPY_MARK_FILE).is_file():
        raise InputError(f"{path} isn't a copy `prepare` made; copy it with `prepare` first")


def new_private_dir(path: Path) -> None:
    """Make the new directory `path` for the owner only; its parent may exist or not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    private_dir(path)


@contextmanager
def pinned_store_env(db: Path) -> Generator[None]:
    """Point every store lookup at `db` and turn the feed log off for the block.

    The feed log sits beside the store `AELFRICE_DB` names, so anything a
    copy's writes log would otherwise reach the machine's own feed.
    """
    keys = ("AELFRICE_DB", "AELFRICE_USER_DB", "AELFRICE_FEED_LOG")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["AELFRICE_DB"] = str(db)
    os.environ["AELFRICE_USER_DB"] = str(db.parent / "user-store" / "memory.db")
    os.environ["AELFRICE_FEED_LOG"] = "0"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _sha16(*parts: object) -> str:
    return hashlib.sha256("\0".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every row of a JSONL file; InputError on a line that isn't a JSON object."""
    rows: list[dict[str, Any]] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj: object = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}:{n}: not JSON") from exc
        if not isinstance(obj, dict):
            raise InputError(f"{path}:{n}: not a JSON object")
        rows.append(cast(dict[str, Any], obj))
    return rows


# --- Requests -----------------------------------------------------------------


def sessions_of(turns: Sequence[Turn]) -> dict[str, list[Turn]]:
    """{session id: its turns in file order}; turns with no session are dropped."""
    out: dict[str, list[Turn]] = {}
    for t in turns:
        if t.session_id is not None:
            out.setdefault(t.session_id, []).append(t)
    return out


def newest_first(by_session: Mapping[str, Sequence[Turn]]) -> list[str]:
    """Session ids by their last turn's place in the log, newest first."""
    return sorted(by_session, key=lambda s: by_session[s][-1].seq, reverse=True)


def hook_would_start(store: MemoryStore, turns: Sequence[Turn], horizon: int) -> str | None:
    """None when the Stop hook would start the pass for the session, else why not."""
    if horizon > 0 and len(turns) < horizon:
        return S_SHORT
    if not store.session_has_transcript_ingest(cast(str, turns[0].session_id)):
        return S_NO_BELIEFS
    return None


def seen_turns(turns: Sequence[Turn], horizon: int) -> Sequence[Turn]:
    """The session's first `horizon` turn lines, the ones the pass reads; all of them for 0."""
    return turns[:horizon] if horizon > 0 else turns


def seen_prompts(turns: Sequence[Turn], horizon: int, work: Path) -> list[str]:
    """The session's prompts the pass reads, through the writer's `session_prompts`.

    The `seen_turns` go to a turn log of their own in `work`, which
    `session_prompts` reads as it reads the live log.
    """
    seen = seen_turns(turns, horizon)
    sid = cast(str, turns[0].session_id)
    log = work / f"turns-{_sha16(sid)}.jsonl"
    log.write_text(_jsonl([
        {"role": t.role, "text": t.text, "session_id": sid, "ts": t.ts} for t in seen
    ]), encoding="utf-8")
    try:
        return wb.session_prompts(log, sid)
    finally:
        log.unlink()


def store_requests(
    src: Path, work: Path, *, sessions: int, horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """`(store row, request rows)` for one store, from a fresh copy of it.

    The store row counts each considered session's outcome and, per UTC
    day, the sessions the hook would start the pass for (H7). Request
    rows are in session order, oldest first, the order `run` ingests them.
    """
    if sessions < 1:
        raise InputError("--sessions must be at least 1")
    if horizon < 0:
        raise InputError("--horizon must be 0 or more")
    turns = read_turns(transcript_files(src))
    if not turns:
        raise InputError(f"no transcript turns beside {src}")
    by_session = sessions_of(turns)
    db, snapshot = prepare_snapshot(src, work)
    outcomes: Counter[str] = Counter()
    days: Counter[str] = Counter()
    picked: list[tuple[int, dict[str, Any]]] = []
    store = MemoryStore(str(db), read_only=True)
    try:
        considered = 0
        for sid in newest_first(by_session):
            sturns = by_session[sid]
            skip = hook_would_start(store, sturns, horizon)
            if skip is not None:
                outcomes[skip] += 1
                continue
            day = next((t.ts[:10] for t in sturns if t.ts), None)
            if day is not None:
                days[day] += 1
            if considered >= sessions:
                continue
            considered += 1
            prompts = seen_prompts(sturns, horizon, work)
            if not prompts:
                outcomes[S_NO_PROMPTS] += 1
                continue
            topic = wb.pick_topic(store, prompts)
            if topic is None:
                outcomes[S_NO_TOPIC] += 1
                continue
            payload, prompt = topic
            outcomes[S_REQUEST] += 1
            picked.append((sturns[-1].seq, {
                "kind": ROW_REQUEST, "request_id": _sha16(snapshot, sid),
                "snapshot": snapshot, "session_id": sid, "prompts": len(prompts),
                "coverage": payload.gap_analysis.query_term_coverage,
                "topic": prompt.topic, "axes": list(prompt.axes),
                "anchor_ids": list(prompt.anchor_ids), "input": prompt.text,
            }))
    finally:
        store.close()
    row = {
        "kind": ROW_STORE, "snapshot": snapshot, "store": str(src.resolve()),
        "sessions": len(by_session), "considered": considered,
        "outcomes": dict(sorted(outcomes.items())), "hook_sessions_by_day": dict(sorted(days.items())),
        "horizon": horizon,
    }
    return row, [r for _seq, r in sorted(picked, key=lambda x: x[0])]


@dataclass(frozen=True)
class Requests:
    """A `requests` file: its store rows by snapshot, and its request rows in order."""

    stores: Mapping[str, Mapping[str, Any]]
    requests: tuple[Mapping[str, Any], ...]


def read_requests(path: Path) -> Requests:
    """A `requests` output file; InputError when malformed."""
    return parse_requests(_read_jsonl(path), path)


def parse_requests(rows: Sequence[Mapping[str, Any]], path: Path) -> Requests:
    """Store and request rows as `Requests`; InputError when malformed, naming `path`."""
    stores: dict[str, Mapping[str, Any]] = {}
    reqs: list[Mapping[str, Any]] = []
    for row in rows:
        snap = row.get("snapshot")
        if not isinstance(snap, str) or not snap:
            raise InputError(f"{path}: a row has no snapshot")
        if row.get("kind") == ROW_STORE:
            if snap in stores:
                raise InputError(f"{path}: two store rows for snapshot {snap}")
            stores[snap] = row
        elif row.get("kind") == ROW_REQUEST:
            ok = (
                isinstance(row.get("request_id"), str) and isinstance(row.get("session_id"), str)
                and isinstance(row.get("topic"), str) and isinstance(row.get("input"), str)
                and isinstance(row.get("axes"), list) and isinstance(row.get("anchor_ids"), list)
            )
            if not ok:
                raise InputError(f"{path}: a request row lacks a field")
            reqs.append(row)
        else:
            raise InputError(f"{path}: a row of unknown kind")
    if not stores:
        raise InputError(f"{path}: no store rows")
    ids = [str(r["request_id"]) for r in reqs]
    if len(set(ids)) != len(ids):
        raise InputError(f"{path}: repeats a request id")
    orphan = sorted({str(r["snapshot"]) for r in reqs} - stores.keys())
    if orphan:
        raise InputError(f"{path}: requests of snapshot {orphan[0]} have no store row")
    return Requests(stores=stores, requests=tuple(reqs))


# --- Replies and the ingest ---------------------------------------------------


def read_replies(path: Path, requests: Requests) -> dict[str, tuple[str, int]]:
    """{request id: (stdout, returncode)}; InputError unless one reply per request."""
    replies: dict[str, tuple[str, int]] = {}
    for n, row in enumerate(_read_jsonl(path), 1):
        rid, stdout, code = row.get("request_id"), row.get("stdout"), row.get("returncode")
        if (not isinstance(rid, str) or not isinstance(stdout, str)
                or type(code) is not int):
            raise InputError(f"{path}: reply {n} needs request_id, stdout, and returncode")
        if rid in replies:
            raise InputError(f"{path}: two replies to request {rid}")
        replies[rid] = (stdout, code)
    wanted = {str(r["request_id"]) for r in requests.requests}
    missing = sorted(wanted - replies.keys())
    if missing:
        raise InputError(f"{path}: no reply to request {missing[0]} ({len(missing)} missing)")
    extra = sorted(replies.keys() - wanted)
    if extra:
        raise InputError(f"{path}: reply to unknown request {extra[0]}")
    return replies


def to_prompt(row: Mapping[str, Any]) -> AutoPrompt:
    """The `AutoPrompt` a request row records, which its reply answered."""
    return AutoPrompt(
        topic=str(row["topic"]), axes=tuple(str(a) for a in row["axes"]),
        anchor_ids=tuple(str(a) for a in row["anchor_ids"]), text=str(row["input"]),
    )


def ingest_replies(
    db: Path, rows: Sequence[Mapping[str, Any]], replies: Mapping[str, tuple[str, int]],
) -> list[dict[str, object]]:
    """Each reply through the writer's `apply_reply` on the copy at `db`; the records.

    As `execute_run` does, a record starts at the run's cap and takes the
    CLI's reported cost when there is one.
    """
    records: list[dict[str, object]] = []
    with pinned_store_env(db):
        # A deliberate write: `db` is always the phantom arm's own copy,
        # made by `run_store` in a new temporary folder, never a store given.
        store = MemoryStore(str(db), read_only=False)
        try:
            for row in rows:
                stdout, code = replies[str(row["request_id"])]
                record: dict[str, object] = {
                    "session_id": row["session_id"], "cost_usd": float(wb.RUNNER_BUDGET_USD),
                }
                proc = subprocess.CompletedProcess[str]([], code, stdout, "")
                wb.apply_reply(store, str(row["session_id"]), to_prompt(row), proc, record)
                records.append(record)
        finally:
            store.close()
    return records


@dataclass(frozen=True)
class AutoPhantom:
    """One phantom the background pass wrote: its id, text, axis, anchors, and session."""

    id: str
    content: str
    axis: str
    document: bool
    anchors: tuple[str, ...]
    session_id: str | None


def auto_phantoms(db: Path) -> list[AutoPhantom]:
    """Every phantom of the pass in the store at `db`, by id."""
    pattern = AUTO_WONDER_GENERATOR_FAMILY.replace("_", "\\_") + ":%"
    conn = sqlite3.connect(read_only_uri(str(db)), uri=True)
    try:
        rows = conn.execute(
            "SELECT b.id, b.content, c.source_path_hash, b.session_id FROM beliefs b "
            "JOIN belief_corroborations c ON c.belief_id = b.id "
            "WHERE b.type = 'speculative' AND c.source_type = ? "
            "AND c.source_path_hash LIKE ? ESCAPE '\\' ORDER BY b.id",
            (CORROBORATION_SOURCE_WONDER_INGEST, pattern),
        ).fetchall()
        out: list[AutoPhantom] = []
        for bid, content, meta, sid in rows:
            parts = str(meta).rsplit("@", 1)[0].split(":")
            anchors = tuple(str(r[0]) for r in conn.execute(
                "SELECT dst FROM edges WHERE src = ? AND type = ? ORDER BY rowid",
                (bid, EDGE_RELATES_TO),
            ).fetchall())
            out.append(AutoPhantom(
                id=str(bid), content=str(content), axis=parts[2] if len(parts) > 2 else "",
                document=len(parts) == 5 and parts[3] == DOCUMENT_KEY, anchors=anchors,
                session_id=None if sid is None else str(sid),
            ))
    finally:
        conn.close()
    return out


def narration_ids(db: Path) -> set[str]:
    """Beliefs every ingest row of which records the assistant role (#785)."""
    roles: dict[str, set[str]] = {}
    conn = sqlite3.connect(read_only_uri(str(db)), uri=True)
    try:
        rows = conn.execute(
            "SELECT raw_meta, derived_belief_ids FROM ingest_log "
            "WHERE derived_belief_ids IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    for meta_text, ids_text in rows:
        try:
            meta: object = json.loads(meta_text) if meta_text else None
            ids: object = json.loads(ids_text)
        except (json.JSONDecodeError, TypeError):
            continue
        role = cast(dict[str, object], meta).get("role") if isinstance(meta, dict) else None
        if not isinstance(ids, list):
            continue
        for bid in cast(list[object], ids):
            if isinstance(bid, str):
                roles.setdefault(bid, set()).add(role if isinstance(role, str) else "")
    return {bid for bid, r in roles.items() if r == {"assistant"}}


# --- Retrieval ----------------------------------------------------------------


def run_copy(db: Path, prompts: Sequence[Mapping[str, Any]], arm: str, work: Path,
             snapshot: str) -> dict[str, Any]:
    """One copy's packed non-locked ids per prompt, as the hook's call packs them.

    Retrieval runs with the defaults `real_store_ab_1795`'s A0 runs with.
    """
    rows: list[dict[str, Any]] = []
    store = MemoryStore(str(db), read_only=True)
    try:
        with arm_environment(ARM_A0, work):
            for p in prompts:
                rows.append({"prompt_id": p["prompt_id"], "prompt": p["text"],
                             "ranked": retrieve_packed(store, hook_query(p))})
    finally:
        store.close()
    return {"arm": arm, "snapshot": snapshot, "top_k": TOP_K, "prompts": rows}


def retrieval_packet(
    runs: Mapping[str, Mapping[str, Any]], db: Path, seed: int, *, snapshot: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """`real_store_ab_1795.build_packet` over the two copies.

    Past the top 10s it judges every belief one copy packed and the other
    didn't, so a phantom packed below rank 10 still counts in H3's gain.
    """
    return build_packet(runs, db, seed, snapshot=snapshot, comparisons=COMPARISON)


def top_counts(run: Mapping[str, Any], marked: set[str]) -> tuple[int, int]:
    """`(marked, all)` beliefs over each prompt's top `TOP_K` in `run`."""
    hit = total = 0
    for p in run["prompts"]:
        top = p["ranked"][:TOP_K]
        total += len(top)
        hit += sum(1 for b in top if b in marked)
    return hit, total


def usage_counts(report: Mapping[str, Any], phantoms: set[str]) -> dict[str, int]:
    """H5 counts from a proxy report: phantom hits, and the other hits of their prompts."""
    out = {"phantom_hits": 0, "phantom_used": 0, "other_hits": 0, "other_used": 0}
    for row in report["prompts"]:
        hits = row["arms"][PHANTOM_ARM]
        if not any(h["id"] in phantoms for h in hits):
            continue
        for h in hits:
            side = "phantom" if h["id"] in phantoms else "other"
            out[f"{side}_hits"] += 1
            out[f"{side}_used"] += int(bool(h["used"]))
    return out


# --- Run ----------------------------------------------------------------------


def read_by_pass(turns: Sequence[Turn], rows: Sequence[Mapping[str, Any]], horizon: int) -> set[str]:
    """The stripped user prompts the pass read for `rows`' sessions.

    H3's sample leaves them out: each phantom answers a topic taken from
    them, so retrieval scored on them would grade the phantoms on their
    own question.
    """
    by_session = sessions_of(turns)
    return {t.text.strip() for row in rows
            for t in seen_turns(by_session.get(str(row["session_id"]), []), horizon)
            if t.role == "user"}


@dataclass
class StoreRun:
    """What `run` makes for one store: packet and key rows, and the store's report."""

    retrieval: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    key: list[dict[str, Any]]
    report: dict[str, Any]
    phantoms: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


def run_store(
    src: Path, store_row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
    replies: Mapping[str, tuple[str, int]], work: Path, *, seed: int, max_prompts: int,
) -> StoreRun:
    """Ingest the store's replies on a copy, retrieve from both copies, and build its rows."""
    snapshot = str(store_row["snapshot"])
    (work / BASE_ARM).mkdir()
    (work / PHANTOM_ARM).mkdir()
    base, snap = prepare_snapshot(src, work / BASE_ARM)
    if snap != snapshot:
        raise InputError(f"{src} changed since its requests; build them again")
    phantom = work / PHANTOM_ARM / "memory.db"
    copy_store(base, phantom)
    records = ingest_replies(phantom, rows, replies)
    before = {p.id for p in auto_phantoms(base)}
    new = [p for p in auto_phantoms(phantom) if p.id not in before]
    new_ids = {p.id for p in new}

    turns = read_turns(transcript_files(src))
    held_out = read_by_pass(turns, rows, int(store_row["horizon"]))
    try:
        prompts = sample_prompts(turns, seed, max_prompts, exclude=held_out)
    except InputError:
        # A small store whose every prompt the pass read has none left to
        # sample. Its claims still count, and the money is already spent.
        prompts = []
    runs = {arm: run_copy(db, prompts, arm, work, snapshot)
            for arm, db in ((BASE_ARM, base), (PHANTOM_ARM, phantom))}
    packet: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = [{"kind": KEY_STORE, "snapshot": snapshot, "prompts_sampled": 0,
                                  "arms_run": sorted(ARMS_RUN)}]
    if prompts:
        packet, key = retrieval_packet(runs, phantom, seed, snapshot=snapshot)

    claims = [p for p in new if not p.document]
    texts = belief_texts(phantom, [a for p in claims for a in p.anchors])
    claim_rows: list[dict[str, Any]] = []
    for p in claims:
        cid = _sha16(seed, snapshot, KEY_CLAIM, p.id)
        claim_rows.append({"id": cid, "claim": p.content, "notes": [texts[a] for a in p.anchors]})
        key.append({"kind": KEY_CLAIM, "id": cid, "snapshot": snapshot, "phantom_id": p.id,
                    "axis": p.axis})

    usage = usage_counts(proxy(runs, prompts, turns, phantom, snapshot=snapshot), new_ids)
    narration = narration_ids(base)
    phantom_hits = sum(1 for p in runs[PHANTOM_ARM]["prompts"] for b in p["ranked"] if b in new_ids)
    store = MemoryStore(str(phantom), read_only=True)
    try:
        promotable = {b.id for b in find_evidence_promotable_phantoms(store)} & new_ids
        stale = [b for b in store.query_wonder_gc_candidates(cutoff_ts=GC_ALL_CUTOFF)
                 if b in new_ids]
        kept = promotion_guarded_ids(store, stale)
    finally:
        store.close()
    outcomes = Counter(str(r.get("outcome")) for r in records)
    report: dict[str, Any] = {
        "snapshot": snapshot,
        "sessions": store_row.get("sessions"), "session_outcomes": store_row.get("outcomes"),
        "hook_sessions_by_day": store_row.get("hook_sessions_by_day", {}),
        "eligible_sessions": int(store_row["considered"]),
        "requests": len(rows), "run_outcomes": dict(sorted(outcomes.items())),
        # One request per session, so a session holding a new claim is a
        # run that wrote one; a run whose claims were all known wrote none.
        "runs_with_claim": len({p.session_id for p in claims}),
        "costs": [float(cast(float, r["cost_usd"])) for r in records if "cost_usd" in r],
        "cost_unusable": sum(1 for r in records if r.get("cost_unusable")),
        "claims_written": len(claims), "documents_written": len(new) - len(claims),
        "prompts_sampled": len(prompts), "prompts_held_out": len(held_out),
        "phantom_hits": phantom_hits,
        "narration": {arm: dict(zip(("narration", "top"), top_counts(runs[arm], narration),
                                    strict=True)) for arm in ARMS_RUN},
        "usage": usage,
        "promotable": len(promotable),
        "gc_collected_at_ttl": len(set(stale) - kept),
    }
    phantom_rows: list[dict[str, Any]] = [{
        "snapshot": snapshot, "session_id": p.session_id, "axis": p.axis,
        "document": p.document, "text": p.content, "anchors": list(p.anchors),
    } for p in new]
    return StoreRun(retrieval=packet, claims=claim_rows, key=key, report=report, phantoms=phantom_rows)


def run_all(
    requests: Requests, replies: Mapping[str, tuple[str, int]], stores: Sequence[Path],
    work_dir: Path | None, *, seed: int, max_prompts: int,
) -> StoreRun:
    """Every store's rows, pooled; InputError unless the stores are the requests' stores."""
    by_snap: dict[str, Path] = {}
    for src in stores:
        if not src.is_file():
            raise InputError(f"no store at {src}")
        with tempfile.TemporaryDirectory(dir=work_dir) as work:
            snap = prepare_snapshot(src, Path(work))[1]
        if snap in by_snap:
            raise InputError(f"{src} and {by_snap[snap]} are the same snapshot")
        by_snap[snap] = src
    missing = sorted(requests.stores.keys() - by_snap.keys())
    if missing:
        raise InputError(
            f"no --store holds the requests' snapshot {missing[0]}; a store that"
            " changed since its requests needs them built again")
    extra = sorted(by_snap.keys() - requests.stores.keys())
    if extra:
        raise InputError(f"{by_snap[extra[0]]} has no requests; build them first")
    pooled = StoreRun(retrieval=[], claims=[], key=[], report={"stores": []})
    for snap in sorted(by_snap):
        rows = [r for r in requests.requests if r["snapshot"] == snap]
        with tempfile.TemporaryDirectory(dir=work_dir) as work:
            one = run_store(by_snap[snap], requests.stores[snap], rows, replies, Path(work),
                            seed=seed, max_prompts=max_prompts)
        pooled.retrieval.extend(one.retrieval)
        pooled.claims.extend(one.claims)
        pooled.key.extend(one.key)
        pooled.phantoms.extend(one.phantoms)
        pooled.report["stores"].append(one.report)
    pooled.retrieval.sort(key=lambda r: r["id"])
    pooled.claims.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in (*pooled.retrieval, *pooled.claims)]
    if len(set(ids)) != len(ids):
        raise InputError("packet id collision; pick another seed")
    pooled.report.update({"seed": seed, "max_prompts": max_prompts})
    return pooled


def write_run(result: StoreRun, out: Path) -> None:
    """The retrieval packet, held claims, key, and report under the new directory `out`."""
    new_private_dir(out)
    private_dir(out / RETRIEVAL_DIR)
    write_private(out / RETRIEVAL_DIR / PACKET_FILE, _jsonl(result.retrieval))
    write_private(out / RETRIEVAL_DIR / INSTRUCTIONS_FILE, RELEVANCE_INSTRUCTIONS)
    write_private(out / CLAIMS_HELD_FILE, _jsonl(result.claims))
    write_private(out / KEY_FILE, _jsonl(result.key))
    write_private(out / REPORT_FILE,
                  json.dumps(result.report, indent=2, sort_keys=True) + "\n")
    write_private(out / PHANTOMS_FILE, _jsonl(result.phantoms))
    write_private(out / GRADING_FILE, GRADING_TEXT.format(run=out))


# --- Prepare ------------------------------------------------------------------


def _resolves_into(path: Path, excluded: frozenset[str]) -> bool:
    """Whether `path`, links followed, has an excluded name in it; True on a link loop."""
    try:
        parts = path.resolve().parts
    except (OSError, RuntimeError):
        return True
    return any(part.lower() in excluded for part in parts)


def project_stores(root: Path, excluded: frozenset[str]) -> list[tuple[str, Path]]:
    """`(project name, store path)` for each project under `root`, by name.

    A project is dropped on its name, or a link on where it resolves, before
    anything reads a store or a transcript. The links checked are the project,
    `.git`, `.git/aelfrice`, the store and its sidecar, the transcripts folder,
    and each transcript.
    """
    if not root.is_dir():
        raise InputError(f"no projects folder at {root}")
    with os.scandir(root) as entries:
        names = sorted(entry.name for entry in entries)
    found: list[tuple[str, Path]] = []
    for name in names:
        project = root / name
        if name.lower() in excluded:
            continue
        # The store's resolved path holds the project's own link target too.
        db = project / ".git" / "aelfrice" / "memory.db"
        if _resolves_into(db, excluded) or not db.is_file():
            continue
        beside = (db.with_name(db.name + "-wal"), db.parent / "transcripts", db.parent / "transcripts" / "archive")
        if any(_resolves_into(path, excluded) for path in beside):
            continue
        if not any(_resolves_into(path, excluded) for path in transcript_files(db)):
            found.append((name, db))
    return found


def copy_size(db: Path) -> int:
    """Bytes `prepare` writes for the store at `db`, sidecars and transcripts included."""
    wal = db.with_name(db.name + "-wal")
    files = [db, *([wal] if wal.exists() else []), *transcript_files(db)]
    return sum(p.stat().st_size for p in files)


def nearest_existing(path: Path) -> Path:
    """`path`, or the nearest parent of it that exists."""
    for d in (path, *path.parents):
        if d.exists():
            return d
    return path


def copy_state(target: Path) -> str:
    """`new`, `finished`, or raises: a `memory.db` with no mark isn't known to be a copy.

    It may be a copy that a killed `prepare` left half written, which `replay`
    refuses, or a real store. Nothing here deletes it, so nothing here can lose one.
    """
    if not (target / "memory.db").exists():
        return "new"
    if (target / COPY_MARK_FILE).exists():
        return "finished"
    raise InputError(f"{target} holds a memory.db with no {COPY_MARK_FILE}; "
                     "if it is a copy prepare left unfinished, delete the folder and run prepare again")


def copy_project(name: str, db: Path, out: Path) -> int:
    """The store at `db` and its transcripts, copied to `out/name`; the bytes on disk.

    `_prepare` checked the folder first with `copy_state`.
    """
    target = out / name
    copied = target / "memory.db"
    private_dir(target)
    copy_store(db, copied)
    os.chmod(copied, 0o600)
    tdir = db.parent / "transcripts"
    for src in transcript_files(db):
        dst = target / "transcripts" / src.relative_to(tdir)
        dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(src, dst)
        os.chmod(dst, 0o600)
    write_private(target / COPY_MARK_FILE, "")
    return sum(p.stat().st_size for p in target.rglob("*") if p.is_file())


def _prepare(args: argparse.Namespace) -> None:
    require_outside_repo(args.out)
    stores = project_stores(args.projects_root, excluded_names(args.exclude))
    pending = [(name, db) for name, db in stores if copy_state(args.out / name) == "new"]
    needed = sum(copy_size(db) for _name, db in pending)
    free = shutil.disk_usage(nearest_existing(args.out)).free
    if free < 2 * needed:
        raise InputError(f"copies need {needed} bytes and {free} are free; free some disk first")
    if args.dry_run:
        _print({"dry_run": True, "projects": [name for name, _db in stores],
                "bytes_needed": needed, "bytes_free": free})
        return
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    written = sum(copy_project(name, db, args.out) for name, db in pending)
    _print({"projects": len(stores), "copied": len(pending),
            "already_copied": len(stores) - len(pending), "bytes_copied": written})


# --- Replay -------------------------------------------------------------------


def budget_order(rows: Sequence[Mapping[str, Any]], seed: int) -> list[Mapping[str, Any]]:
    """`rows` in the order the budget spends on them: round the stores, newest first.

    `rows` arrive oldest first per store. The stores' order is shuffled
    with `seed`, so which stores a budget stop leaves out doesn't follow
    from their snapshots' sort order.
    """
    by_store: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_store.setdefault(str(row["snapshot"]), []).append(row)
    order = sorted(by_store)
    random.Random(seed).shuffle(order)
    queues = [list(reversed(by_store[snap])) for snap in order]
    out: list[Mapping[str, Any]] = []
    while any(queues):
        for queue in queues:
            if queue:
                out.append(queue.pop(0))
    return out


def read_ledger(path: Path) -> list[dict[str, Any]]:
    """A JSONL file's rows, or [] when it doesn't exist."""
    return _read_jsonl(path) if path.is_file() else []


def spend_ledger_path() -> Path:
    """The one spend ledger: every replay of this home folder, whatever its `--out`."""
    return Path.home() / SPEND_DIR / SPEND_LEDGER_FILE


@contextmanager
def spend_lock() -> Generator[Path]:
    """Hold the ledger's lock for the block and yield the ledger's path."""
    if not file_lock.HAVE_ADVISORY_LOCKS:
        raise InputError("this host has no file locks, so the spend ceiling can't hold")
    ledger = spend_ledger_path()
    ledger.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(ledger.with_name(SPEND_LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        file_lock.lock_exclusive(fd)
        yield ledger
    finally:
        # Closing the descriptor releases the lock.
        os.close(fd)


@contextmanager
def replay_lock() -> Generator[int]:
    """Hold the one replay slot for the block and yield the lock's descriptor.

    InputError when another replay holds it. Parallel replays each start a
    runner, so runners that ignore their caps would multiply what the ceiling
    can't stop. A resume also closes every open run of its `--out` as dead,
    which would be wrong for a live replay. A runner given the descriptor
    keeps the lock after a SIGKILL of the harness, until it exits.
    """
    ledger = spend_ledger_path()
    ledger.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(ledger.with_name(REPLAY_LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            file_lock.try_lock_exclusive(fd)
        except BlockingIOError as exc:
            raise InputError(
                "another replay is running, or the runner of a killed one is still alive; "
                "wait for it to finish") from exc
        yield fd
    finally:
        os.close(fd)


@dataclass
class LedgerCall:
    """One real start of the runner, as the ledger records it."""

    out: str
    request_id: str
    charge: float | None = None
    """None until a `done` row closes the call."""


def ledger_calls(rows: Sequence[Mapping[str, Any]]) -> dict[object, LedgerCall]:
    """Each call the ledger records, by its number.

    The number, not the request id, keys the charge, so no later row can
    overwrite what an earlier call cost. A row with no number counts as a
    call of its own, and a `done` row whose cost isn't a finite number of at
    least 0 charges `RUN_RESERVE_USD`.
    """
    calls: dict[object, LedgerCall] = {}
    for i, row in enumerate(rows):
        number = row.get("call")
        key: object = number if type(number) is int else f"row-{i}"
        call = calls.setdefault(key, LedgerCall(str(row.get("out")), str(row.get("request_id"))))
        if row.get("state") == STATE_DONE:
            cost = wb.checked_cost(row.get("cost_usd"))
            charge = RUN_RESERVE_USD if cost is None else cost
            call.charge = charge if call.charge is None else max(call.charge, charge)
    return calls


def ledger_state(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, object]]:
    """`(charge of each finished request, the open call of each unfinished one)`."""
    done: dict[str, float] = {}
    open_calls: dict[str, object] = {}
    for key, call in ledger_calls(rows).items():
        if call.charge is None:
            open_calls[call.request_id] = key
        else:
            done[call.request_id] = done.get(call.request_id, 0.0) + call.charge
    return done, {rid: key for rid, key in open_calls.items() if rid not in done}


def total_spend() -> float:
    """Dollars the spend ledger records, per real call, an unfinished one at `RUN_RESERVE_USD`."""
    return sum(RUN_RESERVE_USD if c.charge is None else c.charge
               for c in ledger_calls(read_ledger(spend_ledger_path())).values())


def admit_run(out_key: str, request_id: str, budget: float, max_calls: int) -> int | None:
    """Reserve one run in the ledger: its call number, or None when it can't start.

    It can't when what's left doesn't cover `RUN_RESERVE_USD`, or the ledger
    already holds `max_calls` calls. The lock spans the check and the `started`
    row, so parallel replays can't all pass the check on the same total.
    """
    with spend_lock() as ledger:
        if total_spend() + RUN_RESERVE_USD > budget + 1e-9:
            return None
        calls = ledger_calls(read_ledger(ledger))
        if len(calls) >= max_calls:
            return None
        numbers = [k for k in calls if isinstance(k, int)]
        number = max(numbers, default=0) + 1
        _append_line(ledger, {"out": out_key, "call": number, "request_id": request_id,
                              "state": STATE_STARTED})
    return number


def close_run(out_key: str, request_id: str, call: object, cost: object, **fields: object) -> None:
    """Write a call's `done` row, which replaces its reservation with its charge."""
    with spend_lock() as ledger:
        _append_line(ledger, {"out": out_key, "call": call, "request_id": request_id,
                              "state": STATE_DONE, "cost_usd": cost, **fields})


def _append_line(path: Path, row: Mapping[str, Any]) -> None:
    """One JSON line, on disk before this returns, in a file only its owner reads."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def calls_admitted_at_the_cap(budget: float) -> int:
    """How many runs `budget` admits from an empty ledger when each costs its cap."""
    cap = float(wb.RUNNER_BUDGET_USD)
    runs, total = 0, 0.0
    while total + RUN_RESERVE_USD <= budget + 1e-9:
        runs, total = runs + 1, total + cap
    return runs


def run_plan(
    rows: Sequence[Mapping[str, Any]], out: Path, *, budget: float, max_calls: int, seed: int,
) -> dict[str, Any]:
    """What the budget and `max_calls` can reach, from the ledger as it stands.

    `runs_at_the_cap` assumes each run costs its cap. `runs_at_most` assumes
    runs that cost nothing, so only `max_calls` and the requests limit it. A
    claims figure is an upper bound: `MAX_CLAIMS` per axis, every claim
    accepted, and no null score. `..._one_axis_each` counts one axis a run, since
    a reply may answer one of the axes offered.
    """
    cap = float(wb.RUNNER_BUDGET_USD)
    ledger = read_ledger(spend_ledger_path())
    spent = total_spend()
    done, _open = ledger_state([r for r in ledger if r.get("out") == str(out.resolve())])
    order = [r for r in budget_order(rows, seed) if str(r["request_id"]) not in done]
    claims = [MAX_CLAIMS * len(cast(list[str], r["axes"])) for r in order]
    claims_done = sum(MAX_CLAIMS * len(cast(list[str], r["axes"])) for r in rows
                      if str(r["request_id"]) in done)
    room = max(0, max_calls - len(ledger_calls(ledger)))
    at_cap, total = 0, spent
    while at_cap < min(room, len(order)) and total + RUN_RESERVE_USD <= budget + 1e-9:
        at_cap, total = at_cap + 1, total + cap
    at_most = min(room, len(order)) if spent + RUN_RESERVE_USD <= budget + 1e-9 else 0
    claims_at_cap = claims_done + sum(claims[:at_cap])
    claims_at_most = claims_done + sum(claims[:at_most])
    one_axis_done = MAX_CLAIMS * sum(1 for r in rows if str(r["request_id"]) in done)
    one_axis_at_cap = one_axis_done + MAX_CLAIMS * at_cap
    one_axis_at_most = one_axis_done + MAX_CLAIMS * at_most
    needed = next((i + 1 for i in range(at_most)
                   if claims_done + sum(claims[:i + 1]) >= MIN_GRADED), None)
    cost_for_target = None
    if needed is not None:
        cost_for_target = cap if needed == 1 else min(
            cap, (budget - RUN_RESERVE_USD - spent) / (needed - 1))
    plan: dict[str, Any] = {
        "budget_usd": budget, "spent_usd": round(spent, 6), "max_calls": max_calls,
        "max_calls_counts": "every call in the ledger under this HOME",
        "requests_left": len(order), "runs_at_the_cap": at_cap, "runs_at_most": at_most,
        "last_run_starts_at_or_under_usd": round(budget - RUN_RESERVE_USD, 6),
        "each_run_within_budget_up_to_usd": RUN_RESERVE_USD,
        "over_that_bounded_only_by_timeout_seconds": background.RUNNER_TIMEOUT_SECONDS,
        "claims_target": MIN_GRADED, "claims_at_the_cap": claims_at_cap,
        "claims_at_most": claims_at_most,
        "claims_at_the_cap_one_axis_each": one_axis_at_cap,
        "claims_at_most_one_axis_each": one_axis_at_most,
        "target_reachable_at_the_cap": claims_at_cap >= MIN_GRADED,
        "target_reachable_at_most": claims_at_most >= MIN_GRADED,
        "cost_per_run_for_the_target_usd": None if cost_for_target is None else round(cost_for_target, 6),
    }
    warning = None
    if claims_at_most < MIN_GRADED:
        warning = (f"H2 needs {MIN_GRADED} graded claims and these requests reach at most "
                   f"{claims_at_most}, so H2 would be insufficient and the decision inconclusive; "
                   "more sessions or stores are the only way past that")
    elif claims_at_cap < MIN_GRADED:
        warning = (f"H2 needs {MIN_GRADED} graded claims and at the cap this budget reaches at most "
                   f"{claims_at_cap}, so H2 would be insufficient and the decision inconclusive, "
                   f"unless runs cost ${cost_for_target:.4f} or less with no null score")
    elif one_axis_at_cap < MIN_GRADED:
        warning = (f"H2 needs {MIN_GRADED} graded claims and at the cap this budget reaches "
                   f"{one_axis_at_cap} to {claims_at_cap}: {one_axis_at_cap} if each reply answers "
                   "one axis, which would leave H2 insufficient and the decision inconclusive")
    if warning is not None:
        plan["warning"] = warning
    return plan


def run_requests(
    rows: Sequence[Mapping[str, Any]], *, runner: str, out: Path, budget: float, seed: int,
    max_calls: int,
) -> tuple[int, int]:
    """Run the runner on each request the budget admits. `(runs now, stopped)`.

    Each run's `started` row reaches the ledger before the run, so a harness
    killed mid-run still counts it. On a resume that run is closed at
    `RUN_RESERVE_USD` with an empty reply, never run again. Other `--out`
    folders' open rows count at the reserve too, for that folder's resume
    to close.
    """
    out_key = str(out.resolve())
    replies = out / REPLIES_FILE
    with replay_lock() as lock_fd:
        mine = [r for r in read_ledger(spend_ledger_path()) if r.get("out") == out_key]
        done, open_calls = ledger_state(mine)
        answered = {str(r.get("request_id")) for r in read_ledger(replies)}
        for rid, call in sorted(open_calls.items()):
            if rid not in answered:
                _append_line(replies, {"request_id": rid, "stdout": "",
                                       "returncode": FAILED_RETURNCODE})
            close_run(out_key, rid, call, RUN_RESERVE_USD, returncode=FAILED_RETURNCODE)
            done[rid] = RUN_RESERVE_USD
        # The runner runs from the output folder, so no project's files are near it.
        run = partial(background.run_bounded, cwd=str(out),
                      pass_fds=(lock_fd,) if os.name == "posix" else ())
        ran = stopped = 0
        for row in budget_order(rows, seed):
            rid = str(row["request_id"])
            if rid in done:
                continue
            call = admit_run(out_key, rid, budget, max_calls)
            if call is None:
                stopped += 1
                continue
            record: dict[str, object] = {"session_id": row["session_id"]}
            proc = wb.execute_run(to_prompt(row), cli=runner, record=record, run=run)
            stdout = proc.stdout if proc is not None else ""
            code = proc.returncode if proc is not None else FAILED_RETURNCODE
            reported = wb.reported_cost(stdout)
            # `execute_run` drops `cost_usd` only when the runner never started.
            unknown = RUN_RESERVE_USD if "cost_usd" in record else 0.0
            cost = reported if reported is not None else unknown
            _append_line(replies, {"request_id": rid, "stdout": stdout or "", "returncode": code})
            close_run(out_key, rid, call, cost, seconds=record.get("seconds"), returncode=code)
            done[rid] = float(cast(float, cost))
            ran += 1
    return ran, stopped


def admitted_rows(
    store_rows: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], answered: set[str],
) -> list[dict[str, Any]]:
    """The requests with a reply, and store rows that count what the budget stopped.

    A stopped session wasn't eligible: it never had its chance. A store the
    budget left with no request drops out, since retrieval on it would count
    only zeros against a pass that never ran there.
    """
    out: list[dict[str, Any]] = []
    for store_row in store_rows:
        snap = store_row["snapshot"]
        mine = [r for r in rows if r["snapshot"] == snap]
        kept = [dict(r) for r in mine if r["request_id"] in answered]
        lost = len(mine) - len(kept)
        if lost and not kept:
            continue
        outcomes = dict(store_row["outcomes"])
        if lost:
            outcomes[S_REQUEST] = len(kept)
            outcomes[S_BUDGET_STOPPED] = lost
        out.append({**store_row, "outcomes": dict(sorted(outcomes.items())),
                    "considered": int(store_row["considered"]) - lost})
        out.extend(kept)
    return out


def _replay_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Request and store rows for every `--store` that has a transcript."""
    rows: list[dict[str, Any]] = []
    for src in args.store:
        if not read_turns(transcript_files(src)):
            continue
        with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
            store_row, reqs = store_requests(src, Path(work), sessions=args.sessions,
                                             horizon=args.horizon)
        rows.extend([store_row, *reqs])
    return rows


def _replay(args: argparse.Namespace) -> None:
    if not args.budget_usd > 0 or args.budget_usd > TOTAL_CEILING_USD:
        raise InputError(f"--budget-usd must be above 0 and at most {TOTAL_CEILING_USD}")
    require_outside_repo(args.out)
    if args.max < 1:
        raise InputError("--max must be at least 1")
    if args.max_calls is None:
        # A budget too small for one run admits none at the reserve check.
        args.max_calls = max(1, calls_admitted_at_the_cap(args.budget_usd))
    if args.max_calls < 1:
        raise InputError("--max-calls must be at least 1")
    runner = shutil.which(args.runner)
    if runner is None:
        raise InputError(f"no runner at {args.runner}")
    for src in args.store:
        if not src.is_file():
            raise InputError(f"no store at {src}")
    requests_path = args.out / REQUESTS_FILE
    if requests_path.is_file():
        rows = _read_jsonl(requests_path)
    else:
        rows = _replay_requests(args)
    # Every failure of these rows to pair up comes before the first run.
    parse_requests(rows, requests_path)
    reqs = [r for r in rows if r.get("kind") == ROW_REQUEST]
    store_rows = [r for r in rows if r.get("kind") == ROW_STORE]
    plan = run_plan(reqs, args.out, budget=args.budget_usd, max_calls=args.max_calls, seed=args.seed)
    if args.dry_run:
        _print({"dry_run": True, "stores": len(store_rows), "requests": len(reqs), "plan": plan,
                "max_calls": args.max_calls, "max_prompts": args.max,
                "input_chars": sum(len(str(r["input"])) for r in reqs)})
        return
    print(json.dumps({"plan": plan}, sort_keys=True), file=sys.stderr, flush=True)
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not requests_path.exists():
        write_private(requests_path, _jsonl(rows))
    ran, stopped = run_requests(reqs, runner=runner, out=args.out, budget=args.budget_usd,
                                seed=args.seed, max_calls=args.max_calls)
    replies_path = args.out / REPLIES_FILE
    answered = {str(r["request_id"]) for r in read_ledger(replies_path)}
    summary: dict[str, Any] = {"requests": len(reqs), "ran_now": ran, "budget_stopped": stopped,
                               "spent_usd": round(total_spend(), 6),
                               "budget_usd": args.budget_usd}
    if not answered:
        _print({**summary, "note": "no request ran, so there is nothing to grade"})
        return
    run_dir = args.out / RUN_DIR
    if run_dir.exists():
        if ran == 0:
            _print({**summary, "note": f"nothing new ran, so {run_dir} stays as it was"})
            return
        # A grader may already hold the earlier packets, so they keep their folder.
        moved = next(args.out / f"{RUN_DIR}-{k}" for k in range(1, 10_000)
                     if not (args.out / f"{RUN_DIR}-{k}").exists())
        run_dir.rename(moved)
        summary["earlier_run_moved_to"] = str(moved)
    admitted = args.out / ADMITTED_FILE
    if admitted.exists():
        admitted.unlink()
    write_private(admitted, _jsonl(admitted_rows(store_rows, reqs, answered)))
    requests = read_requests(admitted)
    stores = [src for src in args.store if _snapshot_of(src, args.work_dir) in requests.stores]
    result = run_all(requests, read_replies(replies_path, requests), stores, args.work_dir,
                     seed=args.seed, max_prompts=args.max)
    write_run(result, args.out / RUN_DIR)
    run_stores = cast(list[dict[str, Any]], result.report["stores"])
    _print({**summary, "stores_run": len(run_stores), "retrieval_rows": len(result.retrieval),
            "claim_rows": len(result.claims), "H1": h1_supply(run_stores), "H7": h7_cost(run_stores)})


def _snapshot_of(src: Path, work_dir: Path | None) -> str:
    with tempfile.TemporaryDirectory(dir=work_dir) as work:
        return prepare_snapshot(src, Path(work))[1]


# --- Score --------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayKey:
    """A `run` key: the retrieval items as `real_store_ab_1795` keys them, and the claims."""

    retrieval: Key
    claims: tuple[str, ...]


def read_key(path: Path) -> ReplayKey:
    """A `run` key file; InputError on a malformed row or rows that don't fit together."""
    items: list[KeyItem] = []
    stores: dict[str, KeyStore] = {}
    claims: list[str] = []
    for n, row in enumerate(_read_jsonl(path), 1):
        snap, kind, rid = row.get("snapshot"), row.get("kind"), row.get("id")
        if not isinstance(snap, str) or not snap:
            raise InputError(f"{path}:{n}: need a snapshot")
        if kind == KEY_STORE:
            sampled = row.get("prompts_sampled")
            if (type(sampled) is not int or sampled < 0
                    or row.get("arms_run") != sorted(ARMS_RUN) or snap in stores):
                raise InputError(f"{path}:{n}: a malformed or repeated store row")
            stores[snap] = KeyStore(prompts_sampled=sampled, arms_run=tuple(sorted(ARMS_RUN)))
        elif kind == KEY_ITEM:
            arms: object = row.get("arms")
            ok = arms is None or (isinstance(arms, dict) and all(
                a in ARMS_RUN and type(r) is int for a, r in cast(dict[str, object], arms).items()))
            if (not isinstance(rid, str) or not isinstance(row.get("prompt_id"), str) or not ok
                    or not isinstance(row.get("shares_prompt_stem"), bool)):
                raise InputError(f"{path}:{n}: a malformed item row")
            items.append(KeyItem(
                id=rid, prompt_id=str(row["prompt_id"]),
                arms=None if arms is None else dict(cast(dict[str, int], arms)),
                shares_prompt_stem=bool(row["shares_prompt_stem"]), snapshot=snap,
            ))
        elif kind == KEY_CLAIM:
            if not isinstance(rid, str):
                raise InputError(f"{path}:{n}: a malformed claim row")
            claims.append(rid)
        else:
            raise InputError(f"{path}:{n}: a row of unknown kind")
    if not stores:
        raise InputError(f"{path}: no store rows")
    ids = [i.id for i in items] + claims
    if len(set(ids)) != len(ids):
        raise InputError(f"{path}: repeats a row id")
    return ReplayKey(retrieval=make_key(items, stores), claims=tuple(claims))


def read_report(path: Path, key: ReplayKey) -> dict[str, Any]:
    """A `run` report; InputError unless it covers exactly the key's stores."""
    obj: object = json.loads(path.read_text(encoding="utf-8"))
    rows: object = cast(dict[str, object], obj).get("stores") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        raise InputError(f"{path}: not a run report")
    report = cast(dict[str, Any], obj)
    snaps = [cast(dict[str, Any], r).get("snapshot") for r in cast(list[object], rows)
             if isinstance(r, dict)]
    if len(snaps) != len(cast(list[object], rows)) or sorted(map(str, snaps)) != sorted(
            key.retrieval.stores):
        raise InputError(f"{path}: its stores are not the key's stores")
    return report


def read_scores(path: Path) -> Labels:
    """A grader's 0 to 3 scores, read at the #1804 cut of 2."""
    return binarize(parse_scores(path, path.read_bytes()), FIXED_CUT)


def _digests(paths: Sequence[Path]) -> list[str]:
    return [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]


def release_claims(run: Path, labels: Sequence[Path], *, dry_run: bool = False) -> int:
    """Write `run`'s claims packet once both graders scored every retrieval row; the row count.

    The release records each retrieval labels file's digest, which
    `require_released` checks, so H3 and H4 read labels written before
    any grader saw a claim. InputError when the packet is already out or
    the held rows are not the key's claims.
    """
    key = read_key(run / KEY_FILE)
    if (run / RELEASE_FILE).exists():
        raise InputError(f"{run}: the claims packet was already released")
    a, c = (read_scores(p) for p in labels)
    require_labels([i.id for i in key.retrieval.items], a, c, "retrieval rows")
    rows = _read_jsonl(run / CLAIMS_HELD_FILE)
    if sorted(str(r.get("id")) for r in rows) != sorted(key.claims):
        raise InputError(f"{run / CLAIMS_HELD_FILE}: its rows are not the key's claims")
    if not dry_run:
        write_private(run / RELEASE_FILE, json.dumps({"retrieval_labels": _digests(labels)}) + "\n")
        private_dir(run / CLAIMS_DIR)
        write_private(run / CLAIMS_DIR / PACKET_FILE, _jsonl(rows))
        write_private(run / CLAIMS_DIR / INSTRUCTIONS_FILE, CLAIM_INSTRUCTIONS)
    return len(rows)


def require_released(run: Path, labels: Sequence[Path]) -> None:
    """InputError unless `labels` are the retrieval labels `run`'s claims were released against."""
    path = run / RELEASE_FILE
    if not path.is_file():
        raise InputError(f"{run}: release the claims packet with `claims` before scoring")
    obj: object = json.loads(path.read_text(encoding="utf-8"))
    recorded = cast(dict[str, object], obj).get("retrieval_labels") if isinstance(obj, dict) else None
    if recorded != _digests(labels):
        raise InputError(
            "the retrieval labels differ from the ones the claims packet was released against")


def _share(num: int, den: int) -> float | None:
    return num / den if den else None


def h1_supply(stores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """H1: eligible sessions and their share that wrote a claim, and the bad-reply share."""
    eligible = sum(int(s["eligible_sessions"]) for s in stores)
    requests = sum(int(s["requests"]) for s in stores)
    outcomes: Counter[str] = Counter()
    sessions: Counter[str] = Counter()
    for s in stores:
        outcomes.update(cast(dict[str, int], s["run_outcomes"]))
        sessions.update(cast(dict[str, int], s.get("session_outcomes") or {}))
    bad = outcomes.get(wb.OUTCOME_BAD_REPLY, 0)
    with_claim = sum(int(s["runs_with_claim"]) for s in stores)
    return {
        "session_outcomes": dict(sorted(sessions.items())), "eligible_sessions": eligible,
        "requests": requests, "run_outcomes": dict(sorted(outcomes.items())),
        "runs_with_claim": with_claim, "supply": _share(with_claim, eligible),
        "parse_failures": bad, "parse_failure_share": _share(bad, requests),
        "parse_failure_bar": PARSE_FAILURE_BAR,
        "within_bar": (Fraction(bad, requests) <= exact_bar(PARSE_FAILURE_BAR)
                       if requests else None),
    }


def h2_precision(claims: Sequence[str], a: Labels, c: Labels) -> dict[str, Any]:
    """H2: each grader's share of good claims, and the graders' kappa.

    When every grader is under the bar, H2 fails whatever the kappa: no
    reading of the labels passes. Otherwise a kappa under `KAPPA_BAR`, or
    one `exact_kappa` leaves undefined because both graders gave every
    claim one label, makes the labels unreliable.
    """
    graders = [grader_figures(claims, labels) for labels in (a, c)]
    kappa = exact_kappa(claims, a, c)
    if any(int(g["graded"]) < MIN_GRADED for g in graders):
        verdict = V_INSUFFICIENT
    else:
        meets = [Fraction(int(g["correct"]), int(g["graded"])) >= exact_bar(CLAIM_PRECISION_BAR)
                 for g in graders]
        if not any(meets):
            verdict = V_FAIL
        elif kappa is None or kappa < exact_bar(KAPPA_BAR):
            verdict = V_UNRELIABLE
        else:
            verdict = V_PASS if all(meets) else V_FAIL
    return {"claims": len(claims), "graders": graders,
            "kappa": None if kappa is None else float(kappa),
            "bar": CLAIM_PRECISION_BAR, "kappa_bar": KAPPA_BAR, "verdict": verdict}


def value_verdict(figures: Mapping[str, Any]) -> str:
    """H3 for one grader: `insufficient` under `MIN_GRADED`, else the #1811 bar."""
    if int(figures["graded"]) < MIN_GRADED or int(figures["graded_a0"]) < MIN_GRADED:
        return V_INSUFFICIENT
    low = float(figures["ci95"][0])
    return V_PASS if float(figures["mean_gain"]) >= VALUE_BAR and low > 0.0 else V_FAIL


def retrieval_reliable(key: Key, a: Labels, c: Labels) -> tuple[bool, dict[str, Any]]:
    """#1795's H0 on the retrieval items, with the #1804 kappa: the bars and figures."""
    ids = [i.id for i in key.items]
    kappa = exact_kappa(ids, a, c)
    null = [null_figures(key.items, labels) for labels in (a, c)]
    ok = kappa is not None and kappa >= exact_bar(KAPPA_BAR) and all(
        int(n["graded"]) > 0
        and Fraction(int(n["called_relevant"]), int(n["graded"])) <= exact_bar(NULL_RATE_CEILING)
        for n in null)
    return ok, {"kappa": None if kappa is None else float(kappa), "null_arm": null,
                "kappa_bar": KAPPA_BAR, "null_rate_ceiling": NULL_RATE_CEILING,
                "verdict": V_PASS if ok else V_FAIL}


def narration_verdict(stores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """H4's narration half: the phantom copy's top-10 narration share is no higher."""
    sums = {arm: [sum(int(s["narration"][arm][k]) for s in stores) for k in ("narration", "top")]
            for arm in ARMS_RUN}
    (bn, bt), (pn, pt) = sums[BASE_ARM], sums[PHANTOM_ARM]
    if not bt or not pt:
        verdict = V_NOT_MEASURED
    else:
        verdict = V_PASS if Fraction(pn, pt) <= Fraction(bn, bt) else V_FAIL
    return {"base_share": _share(bn, bt), "phantom_share": _share(pn, pt),
            "base_top": bt, "phantom_top": pt, "verdict": verdict}


def h3_h4(key: Key, a: Labels, c: Labels, stores: Sequence[Mapping[str, Any]],
          seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """`(H0, H3, H4)`: the labels' reliability, the value, and the harm."""
    reliable, h0 = retrieval_reliable(key, a, c)
    prompts = sorted({i.prompt_key for i in key.items})
    unkeyed = sum(s.prompts_sampled for s in key.stores.values()) - len(prompts)
    graders = [arm_figures(key.items, PHANTOM_ARM, labels, prompts, seed, control=BASE_ARM,
                           unkeyed=unkeyed) for labels in (a, c)]
    value = combine([value_verdict(g) for g in graders])
    precision = combine([str(g["h5_verdict"]) for g in graders])
    if not reliable:
        value = precision = V_UNRELIABLE
    narration = narration_verdict(stores)
    if narration["verdict"] == V_FAIL:
        harm = V_FAIL
    elif narration["verdict"] == V_NOT_MEASURED and precision == V_PASS:
        harm = V_INCONCLUSIVE
    else:
        harm = precision
    figs = [{k: g[k] for k in ("graded", "graded_a0", "mean_gain", "ci95", "share_gaining",
                               "precision", "precision_a0", "precision_drop")} for g in graders]
    h3 = {"graders": figs, "bar": VALUE_BAR, "verdict": value}
    h4 = {"precision_verdict": precision, "precision_drop_bar": PRECISION_DROP_BAR,
          "narration": narration, "verdict": harm}
    return h0, h3, h4


def h5_usage(stores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """H5, reported: the proxy's use rate of phantom hits against their prompts' other hits."""
    u = {k: sum(int(s["usage"][k]) for s in stores)
         for k in ("phantom_hits", "phantom_used", "other_hits", "other_used")}
    ph, oh = _share(u["phantom_used"], u["phantom_hits"]), _share(u["other_used"], u["other_hits"])
    return {**u, "phantom_rate": ph, "other_rate": oh,
            "at_least_as_often": None if ph is None or oh is None else (
                Fraction(u["phantom_used"], u["phantom_hits"])
                >= Fraction(u["other_used"], u["other_hits"]))}


def h6_conversion(stores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """H6, reported: phantoms written, promotable now, and kept by GC at its TTL."""
    written = sum(int(s["claims_written"]) + int(s["documents_written"]) for s in stores)
    collected = sum(int(s["gc_collected_at_ttl"]) for s in stores)
    return {"claims_written": sum(int(s["claims_written"]) for s in stores),
            "documents_written": sum(int(s["documents_written"]) for s in stores),
            "phantom_hits": sum(int(s["phantom_hits"]) for s in stores),
            "promotable": sum(int(s["promotable"]) for s in stores),
            "gc_collected_at_ttl": collected,
            "survival_lower_bound": _share(written - collected, written),
            "support_events": V_NOT_MEASURED, "baseline_promoted": 0}


def h7_cost(stores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """H7, reported: cost per run, runs over their cap, and the projected busiest day.

    A cost that isn't a finite number of at least 0 counts at the cap in the
    sums and as a run over its cap, since nothing shows what that run spent.
    """
    cap = float(wb.RUNNER_BUDGET_USD)
    checked = [wb.checked_cost(c) for s in stores for c in s["costs"]]
    costs = [cap if c is None else c for c in checked]
    unusable = checked.count(None) + sum(int(s.get("cost_unusable", 0)) for s in stores)
    mean = sum(costs) / len(costs) if costs else None
    busiest = max((int(n) for s in stores for n in s["hook_sessions_by_day"].values()),
                  default=0)
    return {"runs": len(costs), "mean_usd": mean, "max_usd": max(costs, default=None),
            "total_usd": sum(costs), "cap_usd": cap,
            "over_cap": sum(1 for c in costs if c > cap + 1e-9) + unusable,
            "cost_unusable": unusable,
            "mean_bar_usd": COST_MEAN_BAR,
            "mean_within_bar": None if mean is None else mean <= COST_MEAN_BAR + 1e-12,
            "busiest_day_sessions": busiest,
            "projected_busiest_day_usd": None if mean is None else mean * busiest,
            "daily_ceiling_usd": LIVE_DAILY_CEILING_USD}


def decide(h2: str, h3: str, h4: str, over_cap: int) -> str:
    """Kill on a failed H2, H3, or H4 or a run over its cap; pass only when all three pass."""
    if V_FAIL in (h2, h3, h4) or over_cap:
        return DECISION_KILL
    if (h2, h3, h4) == (V_PASS, V_PASS, V_PASS):
        return DECISION_PASS
    return DECISION_OPEN


def score(key: ReplayKey, report: Mapping[str, Any], retrieval: tuple[Labels, Labels],
          claims: tuple[Labels, Labels], seed: int) -> dict[str, Any]:
    """H1 to H7 and the decision from both graders' labels of each packet.

    Counts and rates only, never text or ids.
    """
    retrieval_ids = [i.id for i in key.retrieval.items]
    a, c = retrieval
    require_labels(retrieval_ids, a, c, "retrieval rows")
    require_labels(key.claims, *claims, "claims")
    stores = cast(list[dict[str, Any]], report["stores"])
    h0, h3, h4 = h3_h4(key.retrieval, a, c, stores, seed)
    h2 = h2_precision(key.claims, *claims)
    h7 = h7_cost(stores)
    return {
        "stores": len(stores), "retrieval_rows": len(retrieval_ids), "claim_rows": len(key.claims),
        "cut": FIXED_CUT, "min_graded": MIN_GRADED,
        "H0": h0, "H1": h1_supply(stores), "H2": h2, "H3": h3, "H4": h4,
        "H5": h5_usage(stores), "H6": h6_conversion(stores), "H7": h7,
        "decision": decide(h2["verdict"], h3["verdict"], h4["verdict"], int(h7["over_cap"])),
    }


# --- CLI ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True, parser_class=_Parser)
    pre = sub.add_parser("prepare", help="read-only copies of the real stores under a projects folder")
    pre.add_argument("--projects-root", type=Path, required=True)
    pre.add_argument("--exclude", action="append", required=True, metavar="NAME",
                     help="a project folder never to touch; repeat for each")
    pre.add_argument("--out", type=Path, required=True, help="a folder outside the repository")
    rep = sub.add_parser("replay", help="the pass on store copies with a chosen runner, under a budget")
    rep.add_argument("--store", type=Path, nargs="+", required=True, help="copies from `prepare`")
    rep.add_argument("--runner", required=True, help="the CLI to run, or a fake with its interface")
    rep.add_argument("--budget-usd", type=float, default=TOTAL_CEILING_USD,
                     help=f"total to spend, at most {TOTAL_CEILING_USD}")
    rep.add_argument("--sessions", type=int, default=DEFAULT_REPLAY_SESSIONS)
    rep.add_argument("--horizon", type=int, default=HORIZON_TURNS,
                     help="turn lines of a session the pass has seen; 0 for all")
    rep.add_argument("--seed", type=int, default=DEFAULT_SEED)
    rep.add_argument("--max", type=int, default=DEFAULT_MAX_PROMPTS,
                     help="real prompts sampled per store for retrieval")
    rep.add_argument("--max-calls", type=int, default=None,
                     help="the most calls the spend ledger may hold, from every replay under "
                          "this HOME; default: the runs --budget-usd admits at the cap")
    rep.add_argument("--out", type=Path, required=True,
                     help="a folder outside the repository; the same folder resumes")
    q = sub.add_parser("requests", help="the prompts the pass would send, per recent session")
    q.add_argument("--store", type=Path, nargs="+", required=True)
    q.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS)
    q.add_argument("--horizon", type=int, default=HORIZON_TURNS,
                   help="turn lines of a session the pass has seen; 0 for all")
    q.add_argument("--out", type=Path, required=True, help="a new file outside the repository")
    r = sub.add_parser("run", help="ingest the replies on a copy and build the packets")
    r.add_argument("--requests", type=Path, required=True)
    r.add_argument("--replies", type=Path, required=True)
    r.add_argument("--store", type=Path, nargs="+", required=True)
    r.add_argument("--seed", type=int, default=DEFAULT_SEED)
    r.add_argument("--max", type=int, default=DEFAULT_MAX_PROMPTS,
                   help="real prompts sampled per store for retrieval")
    r.add_argument("--out", type=Path, required=True,
                   help="a new directory outside the repository")
    c = sub.add_parser("claims", help="release the claims packet once retrieval is graded")
    s = sub.add_parser("score", help="H1 to H7 from the run and two graders' labels")
    for p in (c, s):
        p.add_argument("--run", type=Path, required=True, help="the directory `run` wrote")
        p.add_argument("--labels", type=Path, nargs=2, required=True,
                       help="grader a's and grader c's retrieval scores")
    s.add_argument("--claim-labels", type=Path, nargs=2, required=True,
                   help="grader a's and grader c's claim scores")
    s.add_argument("--seed", type=int, default=DEFAULT_SEED, help="bootstrap seed")
    for p in (rep, q, r):
        p.add_argument("--work-dir", type=Path,
                       help="where the temporary store copies go (default: system temp)")
    for p in (rep, q, r):
        p.add_argument("--exclude", action="append", metavar="NAME",
                       help="refuse a --store whose path names this folder; repeat for each")
    for p in (pre, rep, q, r, c, s):
        p.add_argument("--dry-run", action="store_true",
                       help="check the inputs, print what would run, write nothing")
    return parser


def _print(obj: object) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _requests(args: argparse.Namespace) -> None:
    require_outside_repo(args.out)
    if args.out.exists():
        raise InputError(f"{args.out} exists; pick a new file")
    stores: list[Path] = list(args.store)
    for src in stores:
        if not src.is_file():
            raise InputError(f"no store at {src}")
    if args.dry_run:
        turns = [read_turns(transcript_files(src)) for src in stores]
        _print({"dry_run": True, "stores": len(stores),
                "sessions": [len(sessions_of(t)) for t in turns],
                "sessions_per_store": args.sessions, "horizon": args.horizon})
        return
    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for src in stores:
        with tempfile.TemporaryDirectory(dir=args.work_dir) as work:
            store_row, reqs = store_requests(src, Path(work), sessions=args.sessions,
                                             horizon=args.horizon)
        rows.extend([store_row, *reqs])
        summary.append({"sessions": store_row["sessions"], "outcomes": store_row["outcomes"],
                        "input_chars": [len(str(r["input"])) for r in reqs]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_private(args.out, _jsonl(rows))
    _print({"stores": summary, "requests": sum(len(s["input_chars"]) for s in summary)})


def _run(args: argparse.Namespace) -> None:
    require_outside_repo(args.out)
    if args.out.exists():
        raise InputError(f"{args.out} exists; pick a new directory")
    if args.max < 1:
        raise InputError("--max must be at least 1")
    requests = read_requests(args.requests)
    replies = read_replies(args.replies, requests)
    if args.dry_run:
        for src in args.store:
            if not src.is_file():
                raise InputError(f"no store at {src}")
        _print({"dry_run": True, "stores": len(args.store), "requests": len(requests.requests),
                "replies": len(replies), "seed": args.seed, "max_prompts": args.max})
        return
    result = run_all(requests, replies, args.store, args.work_dir,
                     seed=args.seed, max_prompts=args.max)
    write_run(result, args.out)
    stores = cast(list[dict[str, Any]], result.report["stores"])
    _print({"stores": len(stores), "retrieval_rows": len(result.retrieval),
            "claim_rows": len(result.claims), "H1": h1_supply(stores),
            "H5": h5_usage(stores), "H6": h6_conversion(stores), "H7": h7_cost(stores),
            "H4_narration": narration_verdict(stores)})


def _claims(args: argparse.Namespace) -> None:
    n = release_claims(args.run, args.labels, dry_run=args.dry_run)
    _print({"dry_run": bool(args.dry_run), "claim_rows": n})


def _score(args: argparse.Namespace) -> None:
    key = read_key(args.run / KEY_FILE)
    report = read_report(args.run / REPORT_FILE, key)
    require_released(args.run, args.labels)
    ra, rc = (read_scores(p) for p in args.labels)
    ca, cc = (read_scores(p) for p in args.claim_labels)
    if args.dry_run:
        # The release checked the retrieval labels, and the digests match.
        require_labels(key.claims, ca, cc, "claims")
        _print({"dry_run": True, "retrieval_rows": len(key.retrieval.items),
                "claim_rows": len(key.claims), "stores": len(key.retrieval.stores)})
        return
    _print(score(key, report, (ra, rc), (ca, cc), args.seed))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"prepare": _prepare, "replay": _replay, "requests": _requests, "run": _run,
                "claims": _claims, "score": _score}
    try:
        for src in getattr(args, "store", None) or ():
            require_store_copy(src, excluded_names(getattr(args, "exclude", None)))
        require_distinct_stores(getattr(args, "store", None) or ())
        if getattr(args, "work_dir", None) is not None:
            require_outside_repo(args.work_dir)
        handlers[cast(str, args.cmd)](args)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
