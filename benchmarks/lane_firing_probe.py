"""#1366 lane-firing probe — which retrieval lanes actually fire, and how
that compares to the set the resolvers report as enabled.

This repo keeps rediscovering the same failure mode one lane at a time: a
lane the resolvers report as ON that no production call ever reaches (the
heat-kernel lane, `edge_rerank`, the R3 IDF-clip boost arm). Every one of
those was found by reading call sites, which is the method that had been
missing them. **Observing what actually fires is an independent method**,
and that independence is the whole point — where it disagrees with a
static reachability sweep, one of the two is wrong.

The output is a two-directional diff:

* **enabled but never fires** — the dead-lane case above.
* **fires but not reported enabled** — the more alarming direction, and
  the one nobody has looked for. Reporting only the first direction is
  satisfied by a probe that observes nothing at all, so both ship.

Counts, not booleans. "Fired once in 500 calls" and "fired on every call"
are different findings and a boolean erases the difference.

Corpus hazard — read before changing the input
----------------------------------------------
The corpus must come from the **session transcript archive**, never from
`hook_audit.jsonl`. `hook.py` writes
`"prompt_prefix": prompt[:AUDIT_PROMPT_PREFIX_CAP]` at both of its audit
sites, and that cap is 200 characters — this module imports the constant
rather than restating it, so the citation cannot go stale. On this corpus
the median prompt is several times that cap. Prompt length gates the
#741 expansion gate, so a truncated
corpus systematically under-fires the expansion-dependent lanes and
manufactures exactly the false "enabled but never fires" verdict this
probe exists to catch — inverting the defect instead of finding it. The
same truncation has already produced a 4x error in a prior retrieval A/B.
`--transcripts` therefore takes a directory of untruncated `*.jsonl`
session transcripts and there is no `--hook-audit` option.

Method constraints
------------------
* **Environment pinned for the duration of the run, and only then.**
  Every ambient ``AELFRICE_*`` / ``AELF_*`` variable is deleted at the
  top of `main()` and restored before it returns; without the clear the
  diff measures the developer's own opt-ins rather than the shipped
  defaults. `benchmarks/posterior_channel_audit.py` establishes the
  pattern.

  The clear is deliberately **not** at module scope. It used to be, on
  the theory that resolvers read the environment at import time — they
  do not: there is not one module-scope ``os.environ`` read in
  ``src/aelfrice``, every resolver reads it per call. What the
  import-time clear did instead was delete the variables of any process
  that merely *imported* this module, including pytest at collection
  time, where `tests/conftest.py` reads ``AELFRICE_CORPUS_ROOT`` to
  decide whether the corpus-gated tests run. A full ``pytest tests/``
  then silently skipped every one of them — the same class of defect as
  #1278. `test_importing_the_probe_leaves_ambient_config_alone` fails if
  the clear moves back.
* **TOML pinned too.** These resolvers are env -> kwarg -> TOML ->
  default and `_read_toml_flag_for` walks *up from the working
  directory*, so clearing the environment alone is not enough (#1295).
  The run chdirs into a scratch directory for the duration — `retrieve()`
  resolves its own flags internally with no `start=` to pass — and
  `scratch_walk_hits` fails the run if anything above that scratch
  directory carries a `.aelfrice.toml` after all. The pin is verified,
  not assumed.
* **The store is opened read-only.** A bare `MemoryStore(...)` open is a
  write: DDL, migrations, scope-id persistence, and since #1314 the
  lock-expiry sweep, which flips a user's expired locks. Measuring a
  store is not a reason to mutate it.
* **No RNG.** Transcript files are walked in sorted path order and lines
  in file order; the first `--prompts` qualifying user turns are taken.
  Re-running on the same archive selects the same corpus.
* **Nothing is persisted.** The probe reads `LaneTelemetry` through
  `last_lane_telemetry()` and writes only its own JSON report.
  `_LAST_TELEMETRY` is documented as not thread-safe and is contended by
  other work; a parallel observation surface would be a second thing to
  keep true, so the probe reuses the carrier that already ships.

What the probe can and cannot see
---------------------------------
`LaneTelemetry` observes twenty lanes. Ten of those are the #1366
record-at-site fields added to the shipped carrier for this work: each
is written at the leaf where the lane does its work, never derived from
a tier count and never re-resolved from the flag, because a derived
field keeps reporting the old answer after a lane is re-wired.

Three retrieval flags still have a resolver but **no telemetry field**,
so their firing cannot be observed by any runtime record — the probe
reports what they resolve to and lists them as unobservable rather than
silently scoring them as "never fired". A lane with no observation
surface is a finding about the pipeline, not a null result. Why each is
still unobservable is recorded per-lane in `UNOBSERVABLE_LANES`.

Two lanes are observable but their telemetry field **tracks the resolved
flag by construction** (`bm25f_used`, `posterior_weight`): the field
records what the resolver returned, so a 100% fire rate on them is a
restatement of the flag, not evidence that anything fired. Those rows
carry `tracks_flag_by_construction: true` and are named in the report,
because "fired 500/500" read off a resolver is exactly the false
positive this probe exists to catch.

Diff sensitivity
----------------
The two diff directions do **not** have the same sensitivity, and an
empty `fired_but_not_reported_enabled` must not be read as a null
result — see `diff_sensitivity` in the report and `DIFF_SENSITIVITY`
below for the reason and for what would have to change to make that
direction reachable.

Output is aggregate counts only — no belief content, no belief ids, no
paths — so the JSON is safe to paste into an issue.

Run::

    python benchmarks/lane_firing_probe.py \\
        --store <git-common-dir>/aelfrice/memory.db \\
        --transcripts <session-transcript-archive-dir> \\
        --prompts 500 \\
        --json benchmarks/results/lane_firing_probe_1366.json

Exits 0 when the probe completed, 1 when it could not run (empty corpus,
unpinned config, ambient environment left behind). A non-empty diff is a
*result*, not a failure, so it does not change the exit code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ENV_PREFIXES = ("AELFRICE_", "AELF_")
"""Prefixes of the ambient opt-ins `main()` clears for the run.

**No side effect at module scope.** Importing this module must leave the
caller's environment exactly as it found it: `tests/conftest.py` reads
``AELFRICE_CORPUS_ROOT`` at test time to decide whether the corpus-gated
tests run, and an import-time clear made a full ``pytest tests/`` skip
every one of them without saying so.
"""

# Imported rather than restated: the cap is the reason this probe refuses a
# `hook_audit` corpus, and a citation that cannot go stale is worth the
# import. `hook` is on the numpy-free import graph, so this is cheap.
from aelfrice.hook import AUDIT_PROMPT_PREFIX_CAP  # noqa: E402
from aelfrice.retrieval import (  # noqa: E402
    CONFIG_FILENAME,
    LaneTelemetry,
    is_bfs_enabled,
    is_entity_index_enabled,
    is_entity_persist_demote_enabled,
    is_exploration_enabled,
    is_fan_effect_enabled,
    is_heat_kernel_enabled,
    is_hrr_expand_enabled,
    is_hrr_structural_enabled,
    is_max_coverage_pack_enabled,
    is_origin_tiebreak_enabled,
    is_supersession_demote_enabled,
    is_temporal_spine_enabled,
    last_lane_telemetry,
    resolve_bm25_k3,
    resolve_bm25f_per_field,
    resolve_posterior_weight,
    resolve_use_bm25f_anchors,
    resolve_use_gamma_posterior_temperature,
    resolve_use_intentional_clustering,
    resolve_use_type_aware_compression,
    resolve_use_zeta_posterior_rerank,
)
from aelfrice.store import MemoryStore  # noqa: E402

DEFAULT_PROMPTS: int = 500
"""Corpus size. #1366 asks for >=200; 500 keeps a rare lane's fire rate
resolvable to ~0.2% without making the run unreasonably long."""

MIN_PROMPTS: int = 200
"""Below this the run refuses: a lane that fires on 1% of calls is
indistinguishable from a dead lane on a corpus of a few dozen, and an
under-powered "never fires" is the exact false finding #1366 targets."""


# --- Lane table ----------------------------------------------------------


@dataclass(frozen=True)
class ObservableLane:
    """A lane whose firing `LaneTelemetry` can actually witness.

    `reported` is the *independent* side of the diff: it calls the lane's
    resolver with no kwargs, so it reports what the shipped default
    resolves to rather than what the retrieval call happened to be
    passed. `observed` reads the telemetry the same call produced.

    `tracks_flag` marks a lane whose telemetry field is written from the
    resolved flag rather than from anything the lane did. The two sides
    of the diff are then the same number twice, its fire rate is a
    restatement of the flag, and reporting "fired 500/500" off it would
    be the exact false positive this probe exists to catch. Those lanes
    stay in the table — dropping them would hide that the flag is all
    the pipeline records — but every output path marks them.
    """

    name: str
    resolver: str
    field: str
    reported: Callable[[], bool]
    observed: Callable[[LaneTelemetry], bool]
    note: str = ""
    tracks_flag: bool = False


def _always_on() -> bool:
    """A lane with no flag. Unconditional lanes still belong in the diff:
    "reported enabled" is a claim about the pipeline whether or not a
    resolver makes it, and an unconditional lane that never fires is the
    single most alarming reading this probe can produce."""
    return True


OBSERVABLE_LANES: tuple[ObservableLane, ...] = (
    ObservableLane(
        name="l0_locked",
        resolver="(unconditional — no flag)",
        field="locked",
        reported=_always_on,
        observed=lambda t: t.locked > 0,
    ),
    ObservableLane(
        name="l25_entity_index",
        resolver="is_entity_index_enabled()",
        field="l25",
        reported=is_entity_index_enabled,
        observed=lambda t: t.l25 > 0,
    ),
    ObservableLane(
        name="l1_bm25",
        resolver="(unconditional — no flag)",
        field="l1",
        reported=_always_on,
        observed=lambda t: t.l1 > 0,
    ),
    ObservableLane(
        name="l1_bm25f_anchors",
        resolver="resolve_use_bm25f_anchors()",
        field="bm25f_used",
        reported=resolve_use_bm25f_anchors,
        observed=lambda t: t.bm25f_used,
        note="bm25f_used records the resolved L1 implementation, not a "
             "packed-hit count, so it tracks the flag by construction: "
             "its fire rate restates the resolver and is not evidence "
             "that the BM25F scorer produced anything.",
        tracks_flag=True,
    ),
    ObservableLane(
        name="bfs_multihop",
        resolver="is_bfs_enabled()",
        field="bfs",
        reported=is_bfs_enabled,
        observed=lambda t: t.bfs > 0,
    ),
    ObservableLane(
        name="hrr_expand",
        resolver="is_hrr_expand_enabled()",
        field="hrr_expand",
        reported=is_hrr_expand_enabled,
        observed=lambda t: t.hrr_expand > 0,
    ),
    ObservableLane(
        name="temporal_spine",
        resolver="is_temporal_spine_enabled()",
        field="temporal_spine",
        reported=is_temporal_spine_enabled,
        observed=lambda t: t.temporal_spine > 0,
    ),
    ObservableLane(
        name="heat_kernel",
        resolver="is_heat_kernel_enabled()",
        field="heat_used",
        reported=is_heat_kernel_enabled,
        observed=lambda t: t.heat_used,
        note="heat_used is True only when the heat branch rewrote the L1 "
             "ordering, not when the flag resolved True (#1162).",
    ),
    ObservableLane(
        name="posterior_weight_rerank",
        resolver="resolve_posterior_weight() > 0",
        field="posterior_weight",
        reported=lambda: resolve_posterior_weight() > 0.0,
        observed=lambda t: t.posterior_weight > 0.0,
        note="posterior_weight carries the resolved weight, not a count "
             "of rerank decisions — `LaneTelemetry(posterior_weight=...)` "
             "is fed by `resolve_posterior_weight(...)` at the same call "
             "site. Its fire rate is the flag restated; nothing here "
             "observes the rerank doing work.",
        tracks_flag=True,
    ),
    ObservableLane(
        name="expansion_gate",
        resolver="(unconditional — no flag)",
        field="expansion_gate_reason",
        reported=_always_on,
        observed=lambda t: bool(t.expansion_gate_reason),
        note="all six `should_run_expansion` return paths set a non-empty "
             "reason, so this predicate is constant-True inside "
             "`retrieve_with_tiers` and its 1.0000 rate is uninformative. "
             "The gate's real signal is the reason histogram below, not "
             "this row.",
        tracks_flag=True,
    ),
    # --- #1366 record-at-site lanes -------------------------------------
    # Each field below is written where the lane does its work, so a
    # non-zero count is an observation and a zero is a real absence.
    ObservableLane(
        name="type_aware_compression",
        resolver="resolve_use_type_aware_compression()",
        field="compression_renders",
        reported=resolve_use_type_aware_compression,
        observed=lambda t: t.compression_renders > 0,
        note="counts only beliefs the compressor actually SHORTENED. A "
             "`STRATEGY_VERBATIM` return costs exactly the uncompressed "
             "token count, so counting it would report a fire for a call "
             "that changed nothing — held to the same standard as "
             "`entity_persist_demoted`.",
    ),
    ObservableLane(
        name="intentional_clustering",
        resolver="resolve_use_intentional_clustering()",
        field="cluster_packed",
        reported=resolve_use_intentional_clustering,
        observed=lambda t: t.cluster_packed > 0,
        note="counts beliefs the cluster pack selected. The max-coverage "
             "selector takes precedence over this arm when both resolve "
             "on, so a zero here can mean 'lost the precedence contest' "
             "as well as 'off' — read it against max_coverage_pack.",
    ),
    ObservableLane(
        name="max_coverage_pack",
        resolver="is_max_coverage_pack_enabled()",
        field="max_coverage_packed",
        reported=is_max_coverage_pack_enabled,
        observed=lambda t: t.max_coverage_packed > 0,
    ),
    ObservableLane(
        name="entity_persist_demote",
        resolver="is_entity_persist_demote_enabled()",
        field="entity_persist_demoted",
        reported=is_entity_persist_demote_enabled,
        observed=lambda t: t.entity_persist_demoted > 0,
        note="counts beliefs whose score the demotion actually moved; a "
             "belief with no extracted entities, or with S1 → 1, clamps "
             "to 0.0 and keeps the lane-off ordering.",
    ),
    ObservableLane(
        name="supersession_demote",
        resolver="is_supersession_demote_enabled()",
        field="supersession_demoted",
        reported=is_supersession_demote_enabled,
        observed=lambda t: t.supersession_demoted > 0,
    ),
    ObservableLane(
        name="origin_tiebreak",
        resolver="is_origin_tiebreak_enabled()",
        field="origin_tiebreak_decided",
        reported=is_origin_tiebreak_enabled,
        observed=lambda t: t.origin_tiebreak_decided > 0,
        note="counts adjacent pairs in the final L1 order that the origin "
             "term decided (equal composite score, different origin "
             "priority) — not that the sort key carried the term.",
    ),
    ObservableLane(
        name="gamma_posterior_temperature",
        resolver="resolve_use_gamma_posterior_temperature()",
        field="gamma_rerank_scored",
        reported=resolve_use_gamma_posterior_temperature,
        observed=lambda t: t.gamma_rerank_scored > 0,
    ),
    ObservableLane(
        name="zeta_posterior_rerank",
        resolver="resolve_use_zeta_posterior_rerank()",
        field="zeta_rerank_scored",
        reported=resolve_use_zeta_posterior_rerank,
        observed=lambda t: t.zeta_rerank_scored > 0,
    ),
    ObservableLane(
        name="fan_effect",
        resolver="is_fan_effect_enabled()",
        field="fan_effect_ranked",
        reported=is_fan_effect_enabled,
        observed=lambda t: t.fan_effect_ranked > 0,
        note="recorded where the fan-weighted entity ordering is consumed "
             "rather than where the flag resolves, but the only thing "
             "between the two is whether L2.5 returned any hits at all — "
             "treat a non-zero count as 'flag on and L2.5 non-empty', "
             "not as evidence the fan weighting reordered anything.",
        tracks_flag=True,
    ),
    ObservableLane(
        name="hrr_structural",
        resolver="is_hrr_structural_enabled()",
        field="hrr_structural_hit",
        reported=is_hrr_structural_enabled,
        observed=lambda t: t.hrr_structural_hit,
        note="True only when the marker parsed AND the struct index "
             "answered, i.e. when the lane took the whole call; the two "
             "fall-through returns record nothing. A zero here is a fact "
             "about the corpus, not a reachability verdict: the lane is "
             "reachable (a `KIND:target` prompt routes through it, pinned "
             "by tests/test_retrieve_v2_hrr_structural.py), and no "
             "natural-language prompt parses as a structural marker. "
             "'Enabled but never fires on real prompts' is the finding; "
             "'unreachable from src/' is a different claim this does not "
             "support.",
    ),
)


UNOBSERVABLE_LANES: tuple[tuple[str, str, Callable[[], bool], str], ...] = (
    # (name, resolver expression, reported-enabled callable, why)
    ("bm25f_per_field", "resolve_bm25f_per_field()", resolve_bm25f_per_field,
     "the leaf is the two-field scorer inside `aelfrice.bm25`, below the "
     "module that owns LaneTelemetry; recording there means threading a "
     "channel across the module boundary, which is a larger change than "
     "the record-at-site extension this probe shipped"),
    ("bm25_k3_query_saturation", "resolve_bm25_k3() > 0",
     lambda: resolve_bm25_k3() > 0.0,
     "same leaf as bm25f_per_field — the k3 saturation term is applied "
     "inside the scorer. Note the shipped default is 0.0, at which the "
     "boost arm is a verified no-op, so a firing record would be about "
     "an arm nothing reaches"),
    ("exploration_slots", "is_exploration_enabled()", is_exploration_enabled,
     "resolved and applied in `aelfrice.hook`, outside the retrieval call "
     "LaneTelemetry describes; a field here would be recorded by a "
     "different process stage than the one it is filed under"),
)
"""Retrieval flags with a resolver but no `LaneTelemetry` field.

These cannot be diffed by observation at all — not "observed as never
firing", *unobservable*. Scoring them as never-fired would fabricate the
finding; omitting them would hide that the observation surface has holes.
They are reported with their resolved value, an explicit
`observable: false`, and the reason recording them was out of reach —
"no field" on its own invites the reader to assume nobody looked.
"""


# --- Corpus --------------------------------------------------------------


def iter_user_prompts(root: Path) -> Iterator[str]:
    """Yield untruncated user-turn text from a transcript archive.

    Deterministic: files in sorted path order, lines in file order, no
    RNG and no sampling. A caller wanting N prompts takes the first N.

    A transcript line qualifies when `type == "user"` and
    `message.role == "user"` and it carries real text. Excluded:

    * `isMeta` records — harness bookkeeping, not something a user typed.
    * tool-result turns, which are `content` lists with no `text` block.
      These are the bulk of `type == "user"` records and none of them
      ever reached the retrieval path as a prompt.
    * slash-command wrappers (`<command-name>...`), which the host
      expands before the hook sees them.

    Malformed lines and unreadable files are skipped rather than raising:
    an archive is an accumulated artifact, and one bad line is not a
    reason to lose the corpus.
    """
    for path in sorted(root.rglob("*.jsonl")):
        try:
            handle = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (ValueError, TypeError):
                    continue
                text = _user_turn_text(rec)
                if text is not None:
                    yield text


def _user_turn_text(rec: Any) -> str | None:
    """Return the user-typed text of one transcript record, or None."""
    if not isinstance(rec, dict):
        return None
    if rec.get("type") != "user" or rec.get("isMeta"):
        return None
    msg = rec.get("message")
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return None
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if not parts:
            return None
        text = "\n".join(parts)
    else:
        return None
    text = text.strip()
    if not text or text.startswith(("<command-name>", "<local-command")):
        return None
    return text


def normalise_gate_reason(reason: str) -> str:
    """Strip the measured operands out of an expansion-gate reason tag.

    `should_run_expansion` returns tags like ``broad:long(2324>80)``, where
    the parenthesised part is the prompt's own character count. Left as-is
    the histogram is one bucket per distinct prompt length — high
    cardinality, and it publishes a length distribution of the operator's
    prompts into a file meant to be pasteable. The tag without its
    operands (``broad:long``) is the part that says which branch ran.
    """
    out: list[str] = []
    depth = 0
    for ch in reason:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def collect_corpus(root: Path, limit: int) -> list[str]:
    """First `limit` user prompts from the archive, in archive order."""
    out: list[str] = []
    for text in iter_user_prompts(root):
        out.append(text)
        if len(out) >= limit:
            break
    return out


# --- Diff ----------------------------------------------------------------


DIFF_SENSITIVITY: dict[str, dict[str, Any]] = {
    "enabled_but_never_fired": {
        "can_be_populated_by_this_instrument": True,
        "why": (
            "A lane's resolver can report enabled while the branch that "
            "writes its counter never runs — the flag resolves against "
            "env/TOML/default with no kwarg, but `retrieve()` passes an "
            "explicit kwarg to some lanes and pins others OFF. The "
            "resolver and the counter can therefore disagree, so an "
            "entry here is a measurement."
        ),
    },
    "fired_but_not_reported_enabled": {
        "can_be_populated_by_this_instrument": False,
        "why": (
            "Structurally unreachable on the current wiring, for every "
            "lane in the table. Each counter is written only inside a "
            "branch guarded by the same resolver `reported` queries, and "
            "the kwarg `retrieve()` passes is either None (so the "
            "no-kwarg resolver gives the same answer) or False (which "
            "can only produce the *other* direction). The unconditional "
            "lanes report True by definition and cannot appear here "
            "either. **An empty set in this direction is a property of "
            "the instrument, not a null result** — reporting it as a "
            "finding would be the R3 IDF-clip failure mode, where an "
            "unreachable arm was read as a measured absence."
        ),
        "what_would_make_it_reachable": (
            "a caller that forces a lane ON against its resolver (an "
            "explicit `True` kwarg from `retrieve()` / `retrieve_v2`, or "
            "a branch guarded by a different flag than the one the lane "
            "is filed under). The direction is kept precisely so that "
            "re-wiring shows up here instead of going unnoticed."
        ),
    },
}
"""Per-direction sensitivity of `diff_lanes`, reported alongside it.

The two directions are not equally informative and the report must not
present them as if they were. See each entry's `why`.
"""


def diff_lanes(
    reported: dict[str, bool],
    fired: dict[str, int],
) -> dict[str, list[str]]:
    """Both directions of the enabled-vs-observed asymmetry.

    `reported` maps lane -> what its resolver said with no kwargs.
    `fired` maps lane -> how many calls the telemetry witnessed it on.

    Returns `enabled_but_never_fired` and `fired_but_not_reported_enabled`.
    Reporting only the first is satisfied by a probe that observes
    nothing, which is why both are computed here and both are asserted
    on in the tests.
    """
    names = sorted(set(reported) | set(fired))
    return {
        "enabled_but_never_fired": [
            n for n in names if reported.get(n, False) and fired.get(n, 0) == 0
        ],
        "fired_but_not_reported_enabled": [
            n for n in names if fired.get(n, 0) > 0 and not reported.get(n, False)
        ],
    }


# --- Config pinning ------------------------------------------------------


@contextmanager
def pinned_environment() -> Iterator[list[str]]:
    """Delete every ambient `AELFRICE_*` / `AELF_*` var for the block.

    Yields the sorted names that were removed, and puts them back on the
    way out — including on an exception, so a failed run does not leave
    the caller's shell or the pytest session stripped of its config.

    Restoring matters as much as clearing. This clear used to run at
    module import, which stripped the environment of every process that
    imported the module and never gave it back; a `pytest tests/` run
    that collected the probe's test file lost ``AELFRICE_CORPUS_ROOT``
    and silently skipped every corpus-gated test.
    """
    saved = {
        k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIXES)
    }
    for k in saved:
        del os.environ[k]
    try:
        yield sorted(saved)
    finally:
        os.environ.update(saved)


def scratch_walk_hits(scratch: Path) -> list[str]:
    """Config files an upward walk from `scratch` would still find.

    Clearing the environment pins only the env tier. `_read_toml_flag_for`
    walks up from the working directory, so a scratch cwd bounds the walk
    only if nothing above it carries a config either. Resolve first:
    `Path.parents` is lexical, and on darwin `tempfile` hands back
    `/var/folders/...` whose resolved chain runs through `/private`.
    """
    resolved = scratch.resolve()
    return [
        str(parent / CONFIG_FILENAME)
        for parent in (resolved, *resolved.parents)
        if (parent / CONFIG_FILENAME).exists()
    ]


# --- Run -----------------------------------------------------------------


def probe(
    store: MemoryStore,
    prompts: Sequence[str],
) -> dict[str, Any]:
    """Drive `retrieve()` over `prompts` and accumulate per-lane counts.

    Counts, never booleans: `fired_calls` is how many of the calls the
    lane was witnessed on, so a lane that fired once is distinguishable
    from one that fired every time.
    """
    from aelfrice.retrieval import retrieve  # noqa: PLC0415

    fired: dict[str, int] = {lane.name: 0 for lane in OBSERVABLE_LANES}
    gate_reasons: dict[str, int] = {}
    gate_skipped_bfs = 0
    spine_candidates_calls = 0
    spine_packed_calls = 0
    l1_trimmed_calls = 0
    empty_output_calls = 0

    for prompt in prompts:
        # `manifest_reference_locks=True` mirrors the production hook
        # call in `hook_search.search_and_record`; the probe measures the
        # shipped path, not a convenient one.
        hits = retrieve(store, prompt, manifest_reference_locks=True)
        tel = last_lane_telemetry()
        for lane in OBSERVABLE_LANES:
            if lane.observed(tel):
                fired[lane.name] += 1
        reason = normalise_gate_reason(tel.expansion_gate_reason) or "(empty)"
        gate_reasons[reason] = gate_reasons.get(reason, 0) + 1
        if tel.expansion_gate_skipped_bfs:
            gate_skipped_bfs += 1
        if tel.temporal_spine_candidates > 0:
            spine_candidates_calls += 1
        if tel.temporal_spine > 0:
            spine_packed_calls += 1
        if tel.l1_candidates > tel.l1:
            l1_trimmed_calls += 1
        if not hits:
            empty_output_calls += 1

    return {
        "fired_calls": fired,
        "expansion_gate_reason_histogram": dict(sorted(gate_reasons.items())),
        "expansion_gate_skipped_bfs_calls": gate_skipped_bfs,
        # The trim seam: a lane can discover candidates on a call and
        # still pack none of them. "Discovered but trimmed" is a third
        # state that neither side of the diff names, and conflating it
        # with "never fires" would misattribute a budget problem to a
        # reachability problem.
        "temporal_spine_candidate_calls": spine_candidates_calls,
        "temporal_spine_packed_calls": spine_packed_calls,
        "l1_trimmed_calls": l1_trimmed_calls,
        "empty_output_calls": empty_output_calls,
    }


def truncation_control(
    full: dict[str, Any],
    truncated: dict[str, Any],
    n: int,
) -> dict[str, Any]:
    """Same corpus, truncated to the `hook_audit` cap — what changes.

    The corpus hazard in this module's docstring is an argument. This
    turns it into a measurement: the identical prompts are replayed cut
    to `AUDIT_PROMPT_PREFIX_CAP` characters, which is exactly what a
    `hook_audit`-sourced corpus would have handed the retrieval path.
    Any lane whose fire count moves is a lane whose verdict a truncated
    corpus would have got wrong, and any lane that drops to zero is a
    false "enabled but never fires" the truncated instrument would have
    manufactured.
    """
    deltas = {
        name: {
            "full": full["fired_calls"][name],
            "truncated": truncated["fired_calls"][name],
            "delta": truncated["fired_calls"][name] - full["fired_calls"][name],
        }
        for name in sorted(full["fired_calls"])
        if truncated["fired_calls"][name] != full["fired_calls"][name]
    }
    return {
        "prompt_chars": AUDIT_PROMPT_PREFIX_CAP,
        "prompts": n,
        "lanes_whose_fire_count_moved": deltas,
        "lanes_falsely_dead_under_truncation": sorted(
            name for name, d in deltas.items()
            if d["truncated"] == 0 and d["full"] > 0
        ),
        "expansion_gate_reason_histogram":
            truncated["expansion_gate_reason_histogram"],
        "temporal_spine_candidate_calls":
            truncated["temporal_spine_candidate_calls"],
        "temporal_spine_packed_calls":
            truncated["temporal_spine_packed_calls"],
    }


def build_report(
    prompts: Sequence[str],
    observed: dict[str, Any],
    control: dict[str, Any] | None = None,
    *,
    cleared_env: Sequence[str] = (),
) -> dict[str, Any]:
    """Assemble the JSON report. Aggregate counts only.

    Nothing derived from a prompt or a belief goes in here beyond
    lengths and counts: the report is meant to be pasteable into a
    public issue.

    `cleared_env` is the list `pinned_environment()` yielded for this
    run — passed in rather than read from a module global, because the
    clear is a property of the run and not of importing this file.
    """
    reported = {lane.name: bool(lane.reported()) for lane in OBSERVABLE_LANES}
    fired = observed["fired_calls"]
    n = len(prompts)
    lengths = sorted(len(p) for p in prompts)

    lanes = []
    for lane in OBSERVABLE_LANES:
        count = fired[lane.name]
        lanes.append({
            "lane": lane.name,
            "telemetry_field": lane.field,
            "resolver": lane.resolver,
            "reported_enabled": reported[lane.name],
            "fired_calls": count,
            "fire_rate": round(count / n, 4) if n else 0.0,
            "note": lane.note,
            # A row whose field is written from the resolver rather than
            # from the lane's work. Its counts are the flag restated.
            "tracks_flag_by_construction": lane.tracks_flag,
        })

    return {
        "issue": 1366,
        "corpus": {
            "source": "session transcript archive (untruncated user turns)",
            "prompts": n,
            "median_prompt_chars": lengths[n // 2] if n else 0,
            "max_prompt_chars": lengths[-1] if n else 0,
            "prompts_over_audit_prefix_cap": sum(
                1 for x in lengths if x > AUDIT_PROMPT_PREFIX_CAP
            ),
            "audit_prompt_prefix_cap": AUDIT_PROMPT_PREFIX_CAP,
            "why_not_hook_audit": (
                "hook_audit truncates prompt_prefix at "
                f"AUDIT_PROMPT_PREFIX_CAP = {AUDIT_PROMPT_PREFIX_CAP} "
                "(src/aelfrice/hook.py); prompt length gates the expansion "
                "gate, so a truncated corpus under-fires expansion-dependent "
                "lanes — see truncation_control for the measured effect"
            ),
        },
        "environment": {
            "cleared_env_vars": len(cleared_env),
            "env_prefixes_cleared": list(_ENV_PREFIXES),
        },
        "observable_lanes": lanes,
        # Named separately as well as flagged per-row: a reader skimming
        # the table for 100% fire rates has to be told which of them are
        # the resolver echoed back.
        "lanes_whose_field_tracks_the_flag": [
            lane.name for lane in OBSERVABLE_LANES if lane.tracks_flag
        ],
        "unobservable_lanes": [
            {
                "lane": name,
                "resolver": expr,
                "reported_enabled": bool(fn()),
                "observable": False,
                "reason": (
                    "no LaneTelemetry field records this lane firing: "
                    + why
                ),
            }
            for name, expr, fn, why in UNOBSERVABLE_LANES
        ],
        "diff": diff_lanes(reported, fired),
        "diff_sensitivity": DIFF_SENSITIVITY,
        "trim_seam": {
            "temporal_spine_candidate_calls":
                observed["temporal_spine_candidate_calls"],
            "temporal_spine_packed_calls":
                observed["temporal_spine_packed_calls"],
            "l1_trimmed_calls": observed["l1_trimmed_calls"],
        },
        "expansion_gate": {
            "reason_histogram": observed["expansion_gate_reason_histogram"],
            "skipped_bfs_calls": observed["expansion_gate_skipped_bfs_calls"],
        },
        "empty_output_calls": observed["empty_output_calls"],
        "truncation_control": control,
    }


def render(report: dict[str, Any]) -> str:
    """Human-readable rendering of the same numbers the JSON carries."""
    out: list[str] = []
    corpus = report["corpus"]
    out.append("=" * 72)
    out.append("#1366 lane-firing probe — observed lanes vs reported-enabled")
    out.append("=" * 72)
    out.append(
        f"  corpus                : {corpus['prompts']} untruncated user "
        f"turns from the transcript archive"
    )
    out.append(
        f"  median prompt chars   : {corpus['median_prompt_chars']} "
        f"({corpus['prompts_over_audit_prefix_cap']} of "
        f"{corpus['prompts']} exceed the "
        f"{corpus['audit_prompt_prefix_cap']}-char hook_audit cap)"
    )
    out.append(
        f"  ambient env cleared   : "
        f"{report['environment']['cleared_env_vars']} variables"
    )
    out.append("")
    out.append(f"  {'lane':<28} {'reported':>9} {'fired':>8} {'rate':>8}")
    out.append(f"  {'-' * 28} {'-' * 9} {'-' * 8} {'-' * 8}")
    for row in report["observable_lanes"]:
        mark = "*" if row["tracks_flag_by_construction"] else " "
        out.append(
            f"{mark} {row['lane']:<28} {str(row['reported_enabled']):>9} "
            f"{row['fired_calls']:>8} {row['fire_rate']:>8.4f}"
        )
    tracks = report["lanes_whose_field_tracks_the_flag"]
    if tracks:
        out.append("")
        out.append(
            "  * telemetry field is written from the resolved flag, not "
            "from the lane's work:"
        )
        out.append(
            f"    {', '.join(tracks)} — these fire rates restate the "
            f"resolver and are NOT"
        )
        out.append("    evidence that anything fired.")
    out.append("")
    diff = report["diff"]
    sens = report["diff_sensitivity"]
    out.append("  DIFF — enabled but never fired:")
    out.append("    " + (", ".join(diff["enabled_but_never_fired"]) or "(none)"))
    out.append("  DIFF — fired but not reported enabled:")
    out.append(
        "    " + (", ".join(diff["fired_but_not_reported_enabled"]) or "(none)")
    )
    for direction, entry in sens.items():
        if not entry["can_be_populated_by_this_instrument"]:
            out.append(
                f"    ^ {direction}: this instrument CANNOT populate this "
                f"direction on the"
            )
            out.append(
                "      current wiring — every counter is written inside a "
                "branch guarded by"
            )
            out.append(
                "      the same resolver the 'reported' side queries. An "
                "empty set here is"
            )
            out.append(
                "      NOT a null result; it is zero sensitivity. Kept as "
                "a re-wiring guard."
            )
    out.append("")
    unobs = [r["lane"] for r in report["unobservable_lanes"]]
    out.append(
        f"  UNOBSERVABLE ({len(unobs)} flags have a resolver but no "
        f"LaneTelemetry field —"
    )
    out.append("  these are not 'never fired', they are not witnessable):")
    for row in report["unobservable_lanes"]:
        out.append(
            f"    {row['lane']:<28} reported_enabled="
            f"{row['reported_enabled']}"
        )
    out.append("")
    seam = report["trim_seam"]
    out.append(
        f"  trim seam             : temporal_spine discovered candidates on "
        f"{seam['temporal_spine_candidate_calls']} calls, packed on "
        f"{seam['temporal_spine_packed_calls']}"
    )
    out.append(
        f"  L1 trimmed by budget  : {seam['l1_trimmed_calls']} calls"
    )
    out.append(
        f"  empty output          : {report['empty_output_calls']} calls"
    )
    control = report.get("truncation_control")
    if control is not None:
        out.append("")
        out.append(
            f"  TRUNCATION CONTROL — same {control['prompts']} prompts cut to "
            f"{control['prompt_chars']} chars (what a hook_audit corpus "
            f"would give):"
        )
        moved = control["lanes_whose_fire_count_moved"]
        if not moved:
            out.append("    no lane's fire count moved")
        for name, delta in moved.items():
            out.append(
                f"    {name:<26} {delta['full']:>5} -> "
                f"{delta['truncated']:<5} ({delta['delta']:+d})"
            )
        falsely_dead = control["lanes_falsely_dead_under_truncation"]
        out.append(
            "    falsely dead under truncation: "
            + (", ".join(falsely_dead) or "(none)")
        )
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, required=True,
                    help="store to read (opened read-only; never written)")
    ap.add_argument("--transcripts", type=Path, required=True,
                    help="directory searched recursively for untruncated "
                         "*.jsonl session transcripts. NOT hook_audit.jsonl "
                         "— it truncates prompts at 200 chars")
    ap.add_argument("--prompts", type=int, default=DEFAULT_PROMPTS,
                    help=f"corpus size (default {DEFAULT_PROMPTS}, "
                         f"minimum {MIN_PROMPTS})")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the aggregate report here")
    ap.add_argument("--truncate-control", action="store_true",
                    help="replay the same corpus cut to "
                         f"AUDIT_PROMPT_PREFIX_CAP={AUDIT_PROMPT_PREFIX_CAP} "
                         "chars and report which lanes' fire counts move — "
                         "measures the hook_audit corpus hazard instead of "
                         "asserting it")
    args = ap.parse_args(argv)

    if args.prompts < MIN_PROMPTS:
        print(f"--prompts must be at least {MIN_PROMPTS}", file=sys.stderr)
        return 2
    if not args.store.is_file():
        print(f"no such store: {args.store}", file=sys.stderr)
        return 2
    if not args.transcripts.is_dir():
        print(f"no such directory: {args.transcripts}", file=sys.stderr)
        return 2

    store_path = args.store.resolve()
    transcripts = args.transcripts.resolve()
    out_json = args.json.resolve() if args.json is not None else None

    prompts = collect_corpus(transcripts, args.prompts)
    if len(prompts) < MIN_PROMPTS:
        # An empty or thin corpus reads exactly like a real result — every
        # lane "never fires" — and that is the finding this probe exists
        # to avoid manufacturing. Refuse instead.
        print(f"only {len(prompts)} user turns found under {transcripts}; "
              f"need at least {MIN_PROMPTS}. Wrong directory, or an archive "
              f"of tool-result turns only", file=sys.stderr)
        return 1

    # The env clear scopes to the run, not to importing this module. Both
    # the retrieval calls and `build_report`'s resolver reads happen
    # inside it, so the two sides of the diff see the same tier.
    with pinned_environment() as cleared, tempfile.TemporaryDirectory() as tmp:
        stragglers = sorted(
            k for k in os.environ if k.startswith(_ENV_PREFIXES)
        )
        if stragglers:
            print(f"ambient config survived the clear: "
                  f"{', '.join(stragglers)}", file=sys.stderr)
            return 1
        scratch = Path(tmp)
        stray = scratch_walk_hits(scratch)
        if stray:
            print("the scratch walk is not clean, so the TOML tier is "
                  "unpinned: " + ", ".join(stray), file=sys.stderr)
            return 1
        cwd = Path.cwd()
        # `retrieve()` resolves its own flags with no `start=` to pass, so
        # the working directory *is* the TOML tier for this run.
        os.chdir(scratch)
        try:
            store = MemoryStore(str(store_path), read_only=True)
            try:
                observed = probe(store, prompts)
                control = None
                if args.truncate_control:
                    cut = [p[:AUDIT_PROMPT_PREFIX_CAP] for p in prompts]
                    control = truncation_control(
                        observed, probe(store, cut), len(cut),
                    )
                report = build_report(
                    prompts, observed, control, cleared_env=cleared,
                )
            finally:
                store.close()
        finally:
            os.chdir(cwd)

    print(render(report))
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {out_json.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
