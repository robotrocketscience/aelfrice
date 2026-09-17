"""#1526 — what the corrected per-belief accounting does to each injection lane.

#1526 changed what the injection packers charge: from the belief's *content* to
the *line the renderer actually emits*. Every budget constant is unchanged, so
the effect of the change is a change in what each budget buys. This module
measures that effect, per lane, and is the producer for every #1526 figure the
repo publishes.

## What it measures

Eight lanes, each at its own shipped `(budget, l1_limit)` and through its own
renderer:

* `ups` — the per-turn `<aelfrice-memory>` block (`hook.DEFAULT_HOOK_TOKEN_BUDGET`).
* `first_prompt` — the composed first-prompt envelope: the session-start
  sub-block and the per-turn pack rendered into one `<aelfrice-memory>` block
  by `hook._format_hits_with_session_start` (#1547).
* `core` — the first-prompt `<core>` section
  (`hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET`).
* `session_start` — the `<aelfrice-baseline>` block (no shipped budget: the
  production lane passes none, and this module probes it at
  `SESSION_START_PROBE_BUDGET`).
* `search_tool` / `search_tool_bash` — the PreToolUse Grep|Glob and Bash lanes
  (`hook_search_tool.INJECTED_TOKEN_BUDGET` / `BASH_INJECTED_TOKEN_BUDGET`).
* `agent_context` — the Agent/Task worker-context lane
  (`hook_agent_context.INJECTED_TOKEN_BUDGET`).
* `cli_search` — `aelf search` (`retrieval.DEFAULT_TOKEN_BUDGET`).

The *before* arm rebinds the pre-#1526 cost functions; the *after* arm runs the
shipped ones. **Both arms run at the same, unchanged budget constant**, so the
difference between them is the accounting change and nothing else. Both render
through the shipped renderer and the counts are of rendered bytes.

## Saturation: what is suppressed, and what that suppression means

A budget that ends no pack is not evidence about that budget. This module
checks that by re-running an arm with **every budget that can end its pack**
raised to `_probe_budget(store)` — `token_budget` *and* `l25_token_subbudget`,
because the L2.5 sub-pack has its own cap and a pack ended by that cap does not
move when `token_budget` alone is raised.

The probe is a bound, not a factor: the sum over the store of the largest
per-belief charge any packer here can bill, so a pack run at it can afford
every candidate it was offered. `SATURATION_PROBE_FACTOR` survives as a floor
under it. It used to be the whole probe, and it was sized for a grid topping
out at 300 content characters; a multiple of the *lane's* budget is not a
multiple of what a belief costs, and at 18,600 content characters the cells
that read `pool` first move at a `token_budget` near 9,300 — twice a single
belief's charge — which only `cli_search`'s 4x probe of 9,600 clears. Every
other lane's is below it, `ups`'s 6,000 included, so a 4x probe could not
admit even one more belief there and 20 of the 45 `pool` labels on the
extended grid were false. Every label is
published with the probe that produced it (`{arm}_probe_budget`), and a `pool`
is re-rendered at `POOL_CONFIRM_MULTIPLE` times the probe before it is
published — a probe that turns out to be too small raises `PoolProbeTooSmall`
rather than emitting the label.

Round 2 of #1526 found exactly that error, and it reproduces on the shipped
tree. On `ups` at 92 content characters, against a store of 300 beliefs, the
pack returns 58 hits at `token_budget=6000` and 58 at 1,500,000 with the
sub-budget left at its default 400 — and 60 at `token_budget=6000` with the
sub-budget at 1600. A probe that varied `token_budget` alone read that as a
budget binding on nothing, and suppressed the cell.

Each arm records which cap ended it as `before_binds_on` / `after_binds_on`,
and **a byte count is printed either way**. An arm that ended on the candidate
pool is weak evidence about the budget; its bytes are still the bytes the model
receives, and the earlier version of this module suppressed whole cells whose
deviation ran to 53%.

Only one case is genuinely uninformative: **both** arms ending on the pool,
where the two arms return the same candidates and their equality is an equality
of pools. Those cells are marked `pool_equality: true`. Their bytes are still
reported.

`session_start` is `pool_equality` at every length, and that is the finding
rather than a gap in the measurement. `hook.session_start` retrieves with an
**empty query**, and in `retrieve_with_tiers` every relevance lane is gated on
`query.strip()`; only L0 contributes, and L0 is never trimmed by a budget
(#379). No budget can bind on that lane, at any value, which is why #1546
deleted the constant it used to carry. The lane is still measured — the bytes
it emits reach the model on every session — and this module renders it with
the empty query it actually issues rather than with `QUERY`, so the reader
sees why the row reads `pool`.

## The effect is length-dependent, and this module reports the whole curve

What the correction removes from a pack is the ratio of a rendered line to its
content: largest where beliefs are shortest, shrinking towards parity as they
grow. On the `search_tool` lanes it changes sign, because those lanes truncate
to `PER_LINE_CHAR_CAP` and past that cap the old accounting charged *more* than
the lane emits. So there is no single multiplier for this change, and any
figure taken from it is only readable beside the belief length it was measured
at. `--curve` prints the whole grid.

## #1547: the arm this module did not have, and why it could not see the defect

`retrieve_with_tiers._cost` charges `compress_for_retrieval(b).rendered_tokens`
when `use_type_aware_compression` is on — the shipped default — and **no caller
passes it a `belief_cost_fn`**, and nothing renders that compressed form. On a
`snapshot` belief the packer pays a headline price and the hook emits the
document.

Both halves of that sentence moved after this arm was built, and the arm now
measures what is left rather than what it was built for. #1551 and #1552 wired
`hook._ups_belief_line_cost` onto the UPS lane as a `belief_cost_fn`, which
short-circuits `_cost` at `retrieval.py:4798` before compression is reached, so
`ups` and `first_prompt` charge the line they emit and their pack ratio is 1.0
in every retention class at every grid length. #1552 also capped the rendered
content at `hook.BELIEF_CONTENT_CHAR_CAP = 1200`, which bounds the gap that is
left: `undercharge_table`'s snapshot ratio plateaus at 7.2x rather than growing
with belief length. `agent_context` is the only lane here that still passes no
cost function, so it is the lane the charge-vs-emit claim is quoted from — see
`SNAPSHOT_ARM_HEADLINE_LANE`. The two closed lanes are kept in the arm, because
"closed here" is a measurement and it is what would regress if the cost
function were unwired.

Two properties of this module made that invisible, and both are now closed:

1. **The corpus held no snapshot belief.** `_synthetic_store` passed no
   `retention_class`, so every row landed on `RETENTION_UNKNOWN`, which
   `compress_for_retrieval` maps to `verbatim`. Every charge matched every
   emission by construction. `snapshot_every` builds a second corpus that is
   identical except for that one field, and `corpus_shape` reads the class back
   off both stores so the published figures carry a measured count and not the
   generator's arithmetic — the same standard the locked and speculative counts
   are already held to.

2. **No lane composed.** Seven lanes each rendered one block; the first prompt
   of a session renders two into one envelope, and that is the shape #1547's
   live 66,165-character row came from. `first_prompt` renders it.

Three figures come out of this, none of which the seven-lane version could
produce. `undercharge` is charged tokens against emitted tokens for one belief
at each grid length in each retention class. `snapshot_arm` is the three
corpora run through `SNAPSHOT_ARM_LANES`, with `_measure`'s binding probe on
every side. `dedupe` is what #1547's AC2 envelope dedupe recovers — that fix shipped
with no producer able to measure it, because it only fires where a belief
appears in both halves of a composed envelope.

**This module measures the defect. It does not fix it**, and nothing here
changes `src/aelfrice/`.

Usage:

    uv run python benchmarks/injection_budget_bytes.py
    uv run python benchmarks/injection_budget_bytes.py --curve
    uv run python benchmarks/injection_budget_bytes.py --emit-figures
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tests"))


# Beliefs in the synthetic store. 300 is well past every lane's L1 cap, so the
# candidate pool is never the first thing that ends a pack.
STORE_BELIEFS = 300

# Every `LOCK_EVERY`-th belief is a user lock, and every `SPECULATIVE_EVERY`-th
# is speculative-origin. Both arms this change rewrote — `lock_injection_tokens`
# and the `speculative="1"` attribute width — are dead code against a corpus of
# plain agent-inferred beliefs, so the corpus carries both. The lock share is
# small on purpose: L0 is never trimmed, so a store that is mostly locks spends
# the whole budget before the pack loop is reached and measures nothing.
LOCK_EVERY = 50
SPECULATIVE_EVERY = 5

# The #1547 snapshot arm's stride: every `SNAPSHOT_EVERY`-th belief carries
# `retention_class = RETENTION_SNAPSHOT`. Used only by the arm's own corpus;
# the control corpus passes no stride and measures zero of them (`corpus_shape`
# reports the count either way, so the reader sees which corpus is which).
#
# The stride is not a model of the live rate. Live stores run 0.09% `snapshot`
# by candidate count, and 0.09% of 300 beliefs is zero — which is exactly the
# corpus that made the undercharge invisible, whatever its size (#1547 priced
# it at 150x on an uncapped renderer; `undercharge_table` measures 7.2x
# post-#1552, and a corpus with no snapshot belief in it reports neither). The
# arm exists so the class is
# reachable by every lane's pack, including the Bash lane's `l1_limit` of 5, so
# the stride is set where a pool that small still contains one.
#
# It is deliberately coprime with neither of the strides above: a belief that is
# both speculative-origin and snapshot-class, or both locked and snapshot-class,
# is an ordinary row in a live store, and the locked one exercises
# compression's lock override (a locked snapshot renders verbatim). Both are
# reported separately by `corpus_shape` rather than assumed away.
SNAPSHOT_EVERY = 7

# Every synthetic belief is terminated into sentences of about this many
# characters. The headline strategy's first-sentence branch needs a `. ` or
# `.\n` outside a code fence, ending at or before
# `compression.MAX_HEADLINE_CHARS` (240). With no such boundary
# `compression._headline` either returns the content unchanged (content no
# longer than the cap) or hard-truncates at the last space inside it (content
# longer) — two different code paths with two different byte counts, and
# neither is what this arm measures. The generator this module shipped before
# #1547 joined vocabulary words with spaces and produced no boundary at any
# length, so it never reached the branch. 120 is half that cap, so
# the first boundary lands well inside it at every grid length above 120 and the
# headline is a first sentence rather than a truncation. Below 120 a belief
# carries no boundary at all and the headline strategy returns it unchanged,
# which is why the two shortest grid points are the arm's inert control.
SENTENCE_CHARS = 120

# Content lengths the curve is reported at, in characters. Each point, and why
# it is here:
#
# * 40 — below `SENTENCE_CHARS`: no sentence boundary, so a snapshot belief
#   compresses to itself. The arm is inert here by construction, which makes
#   this and 92 the control for every ratio below.
# * 92 — the committed `tests/corpus/replay_soak` corpus's median belief
#   length, and the length every #1526 headline figure is quoted at. Also below
#   `SENTENCE_CHARS`, so also inert.
# * 150 — the first grid point that carries a sentence boundary. The headline
#   strategy fires and the charged-vs-emitted ratio is barely above 1.
# * 200 — `hook_search_tool.PER_LINE_CHAR_CAP`, where that lane's truncation arm
#   starts to dominate; 300 is past it.
# * 300 — the top of this grid before #1547. The largest ratio the old producer
#   could have reported even if its corpus had carried the class, which is the
#   second half of why the defect was invisible here.
# * 1000 — p90 of live per-turn `<belief>` element size (#1547 measured 1,005
#   over n=24,083 elements).
# * 6004 — the edge at which the two arms' accountings of a single `<core>`
#   line part company, and the first grid length whose **before** arm packs no
#   `<core>` line at all. #1552 caps the rendered content at
#   `hook.BELIEF_CONTENT_CHAR_CAP`, so the shipped charge plateaus at 320
#   tokens for any content past 1,200 characters and can never reach the
#   1,500-token `DEFAULT_SESSION_START_CORE_TOKEN_BUDGET`. The pre-#1526
#   charge `max(1, len(content) // 4)` is uncapped and crosses that budget at
#   exactly 6,004 characters: 1,500 at 6,003, 1,501 here. A `<core>` zero is
#   read in the currency of the arm that packed it (see
#   `test_the_producer_names_which_budget_ended_every_pack`), and this is the
#   length that makes the before arm's half of that branch decide something.
#   It replaces 5,950, which was chosen while the crossing ran the other way —
#   the shipped charge was the uncapped one and was the half that overran the
#   budget first. Post-#1552 that relation is inverted and 5,950 sits 54
#   characters below the new edge, on the side where both arms pack a line.
# * 7170 — p99 of that same distribution.
# * 18600 — the length #1547's charged-vs-emitted table is measured at, and the
#   only grid point where this producer's ratio can be compared with the
#   issue's prior.
CORPUS_MEDIAN_CHARS = 92
LENGTH_GRID: tuple[int, ...] = (40, 92, 150, 200, 300, 1000, 6004, 7170, 18600)

# Lengths the snapshot arm builds its own corpus at. A subset of the grid: the
# arm's corpus is a second set of stores, and building one at every grid length
# would double the producer's runtime to report the same shape twice. 92 is the
# control point (no sentence boundary, so the class changes nothing); the rest
# are the grid above `MAX_HEADLINE_CHARS`.
SNAPSHOT_ARM_LENGTHS: tuple[int, ...] = (92, 300, 1000, 7170, 18600)

# Lanes the snapshot arm is measured on. All three compose `<belief>` elements
# out of a budgeted `retrieve()` and render them through
# `hook._split_belief_lines`; the PreToolUse lanes emit a different shape
# entirely and are measured by `LANES` alone.
SNAPSHOT_ARM_LANES: tuple[str, ...] = ("ups", "first_prompt", "agent_context")

# The arm lane the charge-vs-emit claim is quoted from, and the one
# `_flat_1547_keys` lifts to scalar keys for the CHANGELOG gate.
#
# It is `agent_context` and not `ups` because `agent_context` is the only lane
# left in this module whose pack still charges through compression: it passes
# no `belief_cost_fn`, so `retrieve_with_tiers._cost` reaches the branch
# `_charged_tokens` transcribes, while `hook_agent_context._build_block`
# renders the whole `<belief …>` element through `hook._split_belief_lines`.
# #1551 and #1552 closed that gap on `ups` and `first_prompt` — they charge
# `hook._ups_belief_line_cost`, which *is* the emitted line, so their pack
# ratio is 1.0 at every arm length and in every retention class. Those two arms
# are kept, because "the defect is closed here" is a measurement worth
# publishing and is the thing that would regress if the cost function were ever
# unwired.
#
# The consequence for the published ratio is that `undercharge_table` is now
# this lane's table and no other lane's: `_emitted_chars` renders through
# `_split_belief_lines`, which is what `ups`, `first_prompt` and
# `agent_context` all emit, but only `agent_context` still pays
# `_charged_tokens` for it. Its `snapshot` cell — 44 charged against 317
# emitted, 7.2x at 18,600 content characters — is the arm's headline number.
#
# The pack-level companion to it is weaker than the single-belief cell at the
# top of the grid, and that is a measurement rather than a gap. This lane's
# budget is 600 tokens against six user locks whose content is exempt from
# `BELIEF_CONTENT_CHAR_CAP`; at 7,170 and 18,600 content characters those six
# spend the budget before the pack loop reaches a candidate, so the pack is all
# locks, `_pack_charge` has nothing non-locked to sum over and the ratio is
# undefined. Where the pack is not lock-starved it carries the gap: 1.3x at 300
# characters, on 4 non-locked hits charged 266 against 352 emitted.
SNAPSHOT_ARM_HEADLINE_LANE = "agent_context"

# The retention classes the charged-vs-emitted table is reported for, in the
# order #1547's own table lists them.
RETENTION_CLASSES_MEASURED: tuple[str, ...] = (
    "fact", "snapshot", "transient", "unknown",
)

# Seed for the one belief the charged-vs-emitted table is built from. Its own
# seed, not the store's: that table is a single-belief measurement with no
# store behind it, and drawing it off the store's shared stream would make the
# published ratio depend on how many stores had been built first.
UNDERCHARGE_SEED = 1547

# The floor under `_probe_budget`, as a multiple of the cap being probed. Both
# `token_budget` and `l25_token_subbudget` are raised: the L2.5 sub-pack has
# its own cap, and raising `token_budget` alone leaves a pack that ended on
# that cap unchanged, which reads as "the budget does not bind".
#
# This was the whole probe until the grid reached lengths where one belief
# costs several times a lane's entire budget; see `_probe_budget` for what it
# got wrong and what replaced it. It is kept as a floor so that no cell is
# probed more weakly than it was before, which on this corpus binds only on
# `cli_search` at 40 content characters (9,600 against an 8,100 bound).
SATURATION_PROBE_FACTOR = 4

# A query broad enough that the L1 candidate cap, not the query, decides how
# many beliefs reach the pack. Fixed text so the measurement is reproducible.
QUERY = (
    "belief retrieval store commit scanner memory python file git hook "
    "MemoryStore EntityIndex"
)

# Vocabulary the synthetic beliefs are drawn from. Every word appears in QUERY
# or is a near neighbour of one, so BM25 ranks the whole store and the L1 cap
# is what bounds the candidate set.
_VOCAB: tuple[str, ...] = tuple(
    (
        "belief retrieval store commit scanner memory python file git hook "
        "budget render packer token session prompt injection lane corpus "
        "index entity cluster lock manifest baseline worker context"
    ).split()
)

# Identifier-shaped tokens, one per belief, cycled. Without them the entity
# index extracts nothing from a bag of lowercase words, L2.5 returns empty and
# the candidate pool collapses to `l1_limit` alone -- which on the two
# PreToolUse lanes (l1_limit 10 and 5) is small enough that no budget can bind
# on it. Measured: at 92 content chars the Grep|Glob lane's pool is 10 beliefs
# without these and 18 with them. A production store carries identifiers; a
# corpus that does not would silently measure the L1 cap instead of the budget.
_ENTITIES: tuple[str, ...] = tuple(
    (
        "MemoryStore RetrievalLane BeliefPacker HookSearchTool ContextRebuilder "
        "src/aelfrice/retrieval.py tests/test_hook.py BudgetTuner LaneSpec "
        "PerLineCap TokenBudget SessionRing ReplaySoak EntityIndex"
    ).split()
)


def synthetic_content(
    rng: random.Random, i: int, content_chars: int, *, sentences: bool = False,
) -> str:
    """One belief's content, exactly `content_chars` long.

    A per-belief unique token first — `beliefs.content_hash` is UNIQUE, and at
    40 characters a shared prefix plus a small vocabulary collides — then one
    identifier, so L2.5 has something to index, then vocabulary words drawn
    from `rng` until the length is reached.

    `sentences` closes a sentence every `SENTENCE_CHARS` characters. That is
    what the #1547 arm needs and what nothing else may have: without a `. ` the
    headline strategy never takes its first-sentence branch, and **with** one
    the corpus is no longer the corpus every published #1526 figure was
    measured on. So it is off by default and the arm carries its own corpus,
    and the two are separated by a third arm that changes the text without the
    class, so the class effect is attributable rather than asserted.

    Off, this is byte-identical to the generator that shipped before #1547 —
    same shared `rng`, same draw order, same truncation — which is checked by
    `scripts/check_derived_figures.py --mode all`: it re-runs this producer and
    diffs every derived-figure marker naming it against the value published
    beside it. No count of those markers is given, because no command prints
    one. What changed is that the running
    length is accumulated instead of re-joining the whole word list per word.
    That join was quadratic and was 9.8 of the 12.5 seconds one
    18,600-character store took to build (`cProfile`, 797,843 calls to
    `str.join`), which is the mechanical reason the grid could not previously
    reach the lengths the defect lives at.
    """
    parts: list[str] = [f"b{i:04d}", _ENTITIES[i % len(_ENTITIES)]]
    total = len(parts[0]) + 1 + len(parts[1])
    since = 0
    while total < content_chars:
        word = rng.choice(_VOCAB)
        total += 1 + len(word)
        if sentences and total - since >= SENTENCE_CHARS:
            word += "."
            total += 1
            since = total
        parts.append(word)
    return " ".join(parts)[:content_chars]


def _synthetic_store(
    path: Path,
    content_chars: int,
    *,
    seed: int = 1526,
    snapshot_every: int = 0,
    sentences: bool = False,
) -> Any:
    """A store of `STORE_BELIEFS` beliefs, each `content_chars` long.

    Seeded, so the same length always produces the same store. Every belief
    carries alpha=5/beta=1: that clears both `<core>` gates (alpha+beta >= 4,
    mean 0.833 >= 2/3) and renders a five-character `posterior` attribute,
    which is the common case for that attribute's width. A freshly derived
    belief carries alpha=beta=1 and would render an empty `<core>`.

    Every `LOCK_EVERY`-th belief is a user lock and every
    `SPECULATIVE_EVERY`-th is speculative-origin, so the two per-belief arms
    #1526 rewrote for those cases (`retrieval.lock_injection_tokens` and the
    `speculative="1"` attribute in `render_cost.belief_line_chars`) are
    exercised by the published figures rather than by the unit tests alone.

    `snapshot_every` and `sentences` are the #1547 arm. At the defaults — which
    is what every `LANES` figure is measured on — no belief carries a retention
    class and none carries a sentence boundary, so every row lands on
    `RETENTION_UNKNOWN`, which `compress_for_retrieval` maps to `verbatim`.
    That is the corpus this module shipped with, and it is why nothing here
    could see a charge-only discount.

    Separate corpora rather than a changed one, deliberately: this module is
    the producer for every #1526 figure the repo publishes, and folding either
    flag into its one store would re-denominate all of them — every marker
    naming this file that `scripts/check_derived_figures.py --mode all`
    re-derives — while leaving nothing to attribute the change to. Three corpora make the attribution explicit:
    control (neither flag), prose (`sentences` alone), snapshot (both). The
    class effect is prose → snapshot; control → prose is the cost of the text
    change, which is measured rather than asserted to be nil.
    """
    from aelfrice.models import (
        LOCK_NONE,
        LOCK_USER,
        ORIGIN_SPECULATIVE,
        RETENTION_SNAPSHOT,
        RETENTION_UNKNOWN,
        Belief,
    )
    from aelfrice.store import MemoryStore

    # One shared stream across the whole store, which is what the pre-#1547
    # generator used; a per-belief seed would be tidier and would change every
    # belief's content.
    rng = random.Random(seed)
    store = MemoryStore(str(path))
    for i in range(STORE_BELIEFS):
        content = synthetic_content(rng, i, content_chars, sentences=sentences)
        bid = hashlib.sha256(f"{i}:{seed}".encode()).hexdigest()[:16]
        locked = i % LOCK_EVERY == 0
        speculative = not locked and i % SPECULATIVE_EVERY == 0
        snapshot = snapshot_every > 0 and i % snapshot_every == 0
        store.insert_belief(
            Belief(
                id=bid,
                content=content,
                retention_class=(
                    RETENTION_SNAPSHOT if snapshot else RETENTION_UNKNOWN
                ),
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                alpha=5.0,
                beta=1.0,
                type="factual",
                lock_level=LOCK_USER if locked else LOCK_NONE,
                locked_at="2026-01-01T00:00:00Z" if locked else None,
                created_at="2026-01-01T00:00:00Z",
                last_retrieved_at=None,
                origin=(
                    "user_asserted"
                    if locked
                    else ORIGIN_SPECULATIVE
                    if speculative
                    else "agent_inferred"
                ),
            )
        )
    return store


def corpus_shape(store: Any) -> dict[str, int]:
    """Locked, speculative-origin and snapshot-class counts, read off the store.

    Emitted with the figures so a reader can check that the arms those cases
    exercise were populated, rather than trusting `_synthetic_store`'s
    arithmetic. Same justification for all three, and the snapshot count is the
    one that most needs it: the class is a column the store writes and reads
    back, and a generator that set it on a `Belief` the store then dropped
    would leave every ratio below reading 1.0 with nothing to say why.

    `snapshot_unlocked` is reported separately because `compress_for_retrieval`
    renders a *locked* snapshot verbatim — locks override retention class, the
    same rule as "L0 is never trimmed" — so only the unlocked ones can produce
    a shortened charge. On the control corpus both snapshot keys — `snapshot`
    and `snapshot_unlocked` — are 0.

    `sentence_headline` counts the beliefs that satisfy the *other* condition
    the arm needs: a sentence boundary at or before `MAX_HEADLINE_CHARS`, which
    is what takes `compression._headline` down its first-sentence branch. It is
    what separates the middle corpus from the control, and without it that
    middle column is unfalsifiable — a prose corpus generated with the sentence
    boundaries left off is byte-identical to the control at every length, and
    every assertion written against it stays true while it attributes nothing.
    The predicate is `compression`'s own function, not a copy of its rule, so
    this counts what the renderer keys on rather than what this module believes
    it keys on.
    """
    from aelfrice.compression import (
        MAX_HEADLINE_CHARS,
        _first_sentence_end_outside_fence,
    )
    from aelfrice.models import LOCK_USER, ORIGIN_SPECULATIVE, RETENTION_SNAPSHOT

    locked = 0
    speculative = 0
    snapshot = 0
    snapshot_unlocked = 0
    sentence_headline = 0
    for bid in store.list_belief_ids():
        b = store.get_belief(bid)
        if b is None:
            continue
        if b.lock_level == LOCK_USER:
            locked += 1
        if b.origin == ORIGIN_SPECULATIVE:
            speculative += 1
        if b.retention_class == RETENTION_SNAPSHOT:
            snapshot += 1
            if b.lock_level != LOCK_USER:
                snapshot_unlocked += 1
        end = _first_sentence_end_outside_fence(b.content)
        if end is not None and end <= MAX_HEADLINE_CHARS:
            sentence_headline += 1
    return {
        "locked": locked,
        "speculative": speculative,
        "snapshot": snapshot,
        "snapshot_unlocked": snapshot_unlocked,
        "sentence_headline": sentence_headline,
    }


def _legacy_belief_tokens(b: Any) -> int:
    """`retrieval._belief_tokens`'s pre-#1526 body.

    Transcribed from `github/main`: `return _estimate_tokens(b.content)`,
    where `_estimate_tokens` is `ceil(len(text) / 4)` and returns 0 on empty
    text.
    """
    if not b.content:
        return 0
    return int((len(b.content) + 3) // 4)


def _legacy_lock_injection_tokens(b: Any, *, manifest_reference_locks: bool) -> int:
    """`retrieval.lock_injection_tokens`'s pre-#1526 body, from `github/main`."""
    from aelfrice.retrieval import is_reference_lock, lock_manifest_line

    if manifest_reference_locks and is_reference_lock(b):
        text = lock_manifest_line(b)
        return int((len(text) + 3) // 4) if text else 0
    return _legacy_belief_tokens(b)


def _legacy_core_cost(b: Any) -> int:
    """`<core>`'s pre-#1526 per-belief cost.

    Transcribed from `hook._build_session_start_subblock` on `github/main`:
    `cost = max(1, len(b.content) // _CORE_CHARS_PER_TOKEN)`.
    """
    return max(1, len(b.content) // 4)


def _legacy_wrapper_tokens(b: Any) -> int:
    """`retrieval._render_wrapper_tokens`'s pre-#1526 body: it did not exist.

    The compressed arm of `retrieve_with_tiers._cost` added nothing for the
    `<belief>` element before #1526, so the legacy body is a constant zero.
    """
    del b
    return 0


# Every cost function #1526 changed, as `(module stem, attribute, legacy body)`.
# `_legacy_accounting` rebinds each one; the before arm of any lane that reaches
# one and does not rebind it measures a hybrid of the two accountings.
#
# Enumerated as data rather than as five assignments so the set is readable
# from a test. `tests/test_render_cost_1526.py` scans `src/aelfrice` for the
# #1526 cost functions and fails when one appears in neither this tuple nor
# `LEGACY_COST_NOT_REBOUND` — which is the check that was missing when the
# `first_prompt` lane shipped a before arm mixing pre-#1526 retrieval cost with
# post-#1526 `<core>` cost.
LEGACY_COST_REBINDS: tuple[tuple[str, str, Callable[..., int]], ...] = (
    # The uncompressed pack cost, and the L2.5 sub-pack's.
    ("retrieval", "_belief_tokens", _legacy_belief_tokens),
    # What the compressed arm of `retrieve_with_tiers._cost` adds.
    ("retrieval", "_render_wrapper_tokens", _legacy_wrapper_tokens),
    # The L0 arm.
    ("retrieval", "lock_injection_tokens", _legacy_lock_injection_tokens),
    # The cluster and max-coverage packs' default `cost_fn`.
    ("clustering", "_belief_tokens", _legacy_belief_tokens),
    # `<core>`'s packer, reached by every lane that composes a session-start
    # sub-block: `_build_session_start_subblock` calls `_pack_core_candidates`
    # with no `cost_fn`, so the default — this name — is what it packs with.
    ("hook", "_core_belief_cost", _legacy_core_cost),
)

# The #1526 cost functions the before arm deliberately leaves alone, each with
# the reason. Kept beside the rebind table so the scan above has somewhere to
# put a name that is genuinely not a rebind, rather than a tolerance.
LEGACY_COST_NOT_REBOUND: tuple[tuple[str, str, str], ...] = (
    (
        "retrieval",
        "_estimate_tokens",
        "#1526 did not change it. It is `ceil(len(text) / 4)` on both sides, "
        "and it is the primitive the legacy bodies here are written in terms "
        "of, so rebinding it would move both arms together.",
    ),
    (
        "hook_search_tool",
        "_belief_line_cost",
        "Not read off a module global. The Grep|Glob and Bash lanes pass it in "
        "as `belief_cost_fn`, and `_render_search_tool`'s before arm passes "
        "None in its place, so a rebind here would reach nothing.",
    ),
    (
        "hook",
        "_ups_belief_line_cost",
        "Not read off a module global. The UPS lane passes it in as "
        "`belief_cost_fn` (`hook.py:3385`), and the before arms of "
        "`_render_ups` and `_render_first_prompt` pass None in its place to "
        "reproduce the pre-#1526 lane, so a rebind here would reach nothing.",
    ),
)


@contextlib.contextmanager
def _legacy_accounting() -> Iterator[None]:
    """Rebind every cost function #1526 changed to its pre-#1526 body.

    Driven by `LEGACY_COST_REBINDS`, which enumerates the names rather
    than summarising them, because a before arm that misses one measures a
    hybrid. The `first_prompt` lane shipped exactly that defect: it reaches
    `hook._core_belief_cost` through `_build_session_start_subblock`, that name
    was not in the set, and every composed before cell mixed pre-#1526
    retrieval cost with post-#1526 `<core>` cost. Measured at 92 content
    characters the composed before arm read 15,861 bytes against a true legacy
    19,999, and the published change read -15.7% against a consistent -33.2%.

    The packers read these off the module global rather than closing over
    them, so rebinding reaches them. `_render_wrapper_tokens` exists as a
    module-level name for exactly this reason: compression resolves ON by
    default, so the compressed arm is the production one, and with its
    wrapper addition inlined in the closure there was nothing for a before
    arm to rebind. `_pack_core_candidates` resolves its default `cost_fn` to
    `_core_belief_cost` per call for the same reason.

    `LEGACY_COST_NOT_REBOUND` holds the #1526 cost functions this
    deliberately does not touch, with the reason for each. Neither tuple is
    published with a length: the count is whatever the tuple holds, nothing
    asserts it, and it has already gone stale once — it read "two" while
    `hook._ups_belief_line_cost` made it three.
    """
    import importlib

    modules = [
        importlib.import_module(f"aelfrice.{stem}")
        for stem, _attr, _body in LEGACY_COST_REBINDS
    ]
    saved = [
        getattr(mod, attr)
        for mod, (_stem, attr, _body) in zip(modules, LEGACY_COST_REBINDS)
    ]
    try:
        for mod, (_stem, attr, body) in zip(modules, LEGACY_COST_REBINDS):
            setattr(mod, attr, body)
        yield
    finally:
        for mod, (_stem, attr, _body), previous in zip(
            modules, LEGACY_COST_REBINDS, saved
        ):
            setattr(mod, attr, previous)


# --- lanes ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Arm:
    """One rendered block: how many beliefs reached it and how big it is."""

    n_items: int
    n_bytes: int


def _core_candidates(store: Any) -> list[Any]:
    """Every store belief in `<core>`'s order: posterior descending, id ascending."""
    out: list[Any] = []
    for bid in store.list_belief_ids():
        b = store.get_belief(bid)
        if b is not None:
            out.append(b)
    out.sort(key=lambda b: (-(b.alpha / (b.alpha + b.beta)), b.id))
    return out


def _render_core(store: Any, budget: int, sub: int, *, legacy: bool) -> Arm:
    """`<core>` packs directly, with no retrieval, so `sub` is unused here.

    `legacy` is unused too, and that is the point: `_pack_core_candidates`
    resolves its default `cost_fn` to `hook._core_belief_cost` per call, so the
    `_legacy_accounting` rebind `_measure` is already holding open reaches this
    packer the same way it reaches every other. Passing the legacy cost in here
    explicitly is what let the composed `first_prompt` lane — which packs
    `<core>` through `_build_session_start_subblock` and has no such
    parameter — render a before arm the rebind never touched.
    """
    del sub, legacy
    from aelfrice import hook

    candidates = _core_candidates(store)
    packed = hook._pack_core_candidates(candidates, budget)
    return Arm(len(packed), sum(len(hook._core_belief_line(b)) + 1 for b in packed))


def _lane_belief_cost(lane: str, *, legacy: bool) -> Callable[[Any], int] | None:
    """The `belief_cost_fn` `lane` passes to `retrieve()`, or None if it passes none.

    `retrieve_with_tiers._cost` short-circuits to this function when the caller
    supplies one (`retrieval.py:4798`), *before* compression is consulted. So a
    lane that passes one does not charge what `_charged_tokens` transcribes,
    and a producer that omits it models a cost path the lane does not run.

    `None` on the before arm for the same reason `_render_search_tool`'s before
    arm passes None: these names are not read off a module global, so
    `_legacy_accounting` cannot reach them, and passing None is what reproduces
    the pre-#1526 lane. Both are listed in `LEGACY_COST_NOT_REBOUND` with that
    reason.

    The lanes absent from this mapping — `agent_context` above all — pass no
    cost function and still charge through compression, which is what makes
    `agent_context` the surviving charge-vs-emit exposure after #1551/#1552.
    """
    from aelfrice import hook, hook_search_tool

    if legacy:
        return None
    return {
        # #1551, `hook.py:3385` in `hook._retrieve`.
        "ups": hook._ups_belief_line_cost,
        "first_prompt": hook._ups_belief_line_cost,
        # #1526 item 4, `hook_search_tool.py:866`.
        "search_tool": hook_search_tool._belief_line_cost,
        "search_tool_bash": hook_search_tool._belief_line_cost,
    }.get(lane)


def _lane_hits(
    lane: str, store: Any, budget: int, sub: int, *, legacy: bool,
) -> list[Any]:
    """The hits `lane`'s renderer packs, retrieved the way that lane retrieves.

    One retrieval per lane, called by the renderer *and* by `_pack_charge`, so
    the pack a byte count is read off and the pack a charge is summed over
    cannot be two different packs. They were: `_pack_charge` passed no
    `belief_cost_fn` while the lane passes one, so its `unlocked_hits` counted
    a pack `snapshot_items` was never taken from.

    Only the lanes the #1547 arm measures are routed through here. The rest
    build their own call because nothing else reads their hit list.
    """
    from aelfrice import hook_agent_context, retrieval

    kwargs: dict[str, Any] = {}
    if lane == "agent_context":
        kwargs["l1_limit"] = hook_agent_context.INJECTED_L1_LIMIT
    cost_fn = _lane_belief_cost(lane, legacy=legacy)
    if cost_fn is not None:
        kwargs["belief_cost_fn"] = cost_fn
    return list(
        retrieval.retrieve(
            store, QUERY, token_budget=budget, l25_token_subbudget=sub,
            manifest_reference_locks=True, **kwargs,
        )
    )


def _render_ups(store: Any, budget: int, sub: int, *, legacy: bool) -> Arm:
    from aelfrice import hook

    hits = _lane_hits("ups", store, budget, sub, legacy=legacy)
    return Arm(len(hits), len(hook._format_hits(hits)))


def _session_start_block(store: Any) -> str:
    """The `<session-start>` sub-block, built on a cwd that carries no git.

    `_build_session_start_subblock` appends a `<recent-work>` section resolved
    from git plumbing under `cwd`, which would make this lane's byte count a
    function of the branch name and the commit subjects
    `hook.DEFAULT_RECENT_WORK_COMMIT_LIMIT` admits — eight at the shipped
    value — of whatever
    checkout the producer happened to run in. `figures()` has already chdir'd
    into a tempdir for the same hermeticity reason the `[retrieval]` flags
    need, so `Path.cwd()` is a non-git directory and `_resolve_branch` returns
    None before any subprocess reads a log. The measured size of that section
    is published as `first_prompt_recent_work_chars` rather than asserted here.
    """
    from aelfrice import hook

    return hook._build_session_start_subblock(store, cwd=Path.cwd())


def _recent_work_chars_in(block: str) -> int:
    """`<recent-work>`'s size *inside the block the lane rendered*, tag to tag.

    Read back off the composed block rather than measured by a second call to
    `hook._build_recent_work_subblock`. A separate call agrees with the lane
    only while both happen to resolve the same cwd, so an assertion on it
    constrains nothing about the lane: mutating `_session_start_block` to
    `cwd=REPO_ROOT` moves every composed-lane byte count while a separate
    probe on `Path.cwd()` goes on reporting 0. Derived from the block, the
    published `first_prompt_recent_work_chars` moves with the lane and the
    assertion in `tests/test_render_cost_1526.py` fails, which is the whole
    reason the figure is published.
    """
    from aelfrice.hook import RECENT_WORK_CLOSE_TAG, RECENT_WORK_OPEN_TAG

    start = block.find(RECENT_WORK_OPEN_TAG)
    if start < 0:
        return 0
    end = block.find(RECENT_WORK_CLOSE_TAG, start)
    if end < 0:
        return 0
    return end + len(RECENT_WORK_CLOSE_TAG) - start


def _render_first_prompt(store: Any, budget: int, sub: int, *, legacy: bool) -> Arm:
    """The composed first-prompt envelope: session-start sub-block plus hits.

    This is the shape #1547's live 66,165-character row came from, and the
    lane this module did not have. The other seven lanes each render one block;
    the first prompt of a session renders two into one envelope —
    `_build_session_start_subblock` (`<locked>` + `<core>` + `<recent-work>`)
    inside `_format_hits_with_session_start` alongside the per-turn pack — and
    a per-block measurement cannot see what composing them costs.

    `n_items` counts the per-turn hits only. The `<locked>` and `<core>`
    elements above them are not hits and are not packed by this lane's budget:
    `<locked>` is exempt by #379 and `<core>` is packed by its own separate
    `DEFAULT_SESSION_START_CORE_TOKEN_BUDGET`. They are in the byte count
    because the model receives them, which is the whole point of composing.

    `<core>` reaches its cost function through a module global
    `_legacy_accounting` rebinds — `hook._core_belief_cost`, which
    `_pack_core_candidates` resolves per call. That fifth name was missing from
    the rebind set when this lane shipped, and the composed before arm was a
    hybrid: 15,861 bytes at 92 content characters against a true legacy 19,999,
    published as -15.7% where the consistent figure is -33.2%.

    The per-turn half is not reached that way. Since #1551 the lane passes
    `hook._ups_belief_line_cost` as `belief_cost_fn` (`hook.py:3385`), which no
    rebind can reach, so `_lane_belief_cost` selects it on the after arm and
    returns None on the before arm, the same way `_render_search_tool` selects
    its lane's cost function.
    """
    from aelfrice import hook

    hits = _lane_hits("first_prompt", store, budget, sub, legacy=legacy)
    block = _session_start_block(store)
    return Arm(len(hits), len(hook._format_hits_with_session_start(list(hits), block)))


def _render_session_start(store: Any, budget: int, sub: int, *, legacy: bool) -> Arm:
    """The SessionStart baseline block, retrieved the way the lane retrieves it.

    The empty query is the point. `hook._retrieve_baseline_with_block` calls
    `retrieve(store, "", ...)`, and every relevance lane in
    `retrieve_with_tiers` is gated on `query.strip()`, so only L0 reaches the
    block and L0 is never trimmed (#379). Rendering this lane with `QUERY`
    would measure a lane that does not exist.

    The production call passes no `token_budget` at all (#1546). `budget`
    here is this module's probe value, varied so the row can report that
    nothing moves.
    """
    del legacy
    from aelfrice import hook, retrieval

    hits = retrieval.retrieve(
        store, "", token_budget=budget, l25_token_subbudget=sub,
        manifest_reference_locks=True,
    )
    return Arm(len(hits), len(hook._format_baseline_hits(hits)))


def _render_agent_context(store: Any, budget: int, sub: int, *, legacy: bool) -> Arm:
    """The Agent/Task worker-context block.

    This lane passes no `belief_cost_fn`, so its pack charges
    `retrieve_with_tiers._cost`'s compression branch — the one
    `_charged_tokens` transcribes — while `hook_agent_context._build_block`
    renders through `hook._split_belief_lines`, the whole `<belief …>` element.
    That is the charge-vs-emit gap #1547 opened on, and after #1551/#1552 wired
    a cost function onto the UPS lane it is the only lane in this module still
    carrying it.
    """
    from aelfrice import hook_agent_context

    hits = _lane_hits("agent_context", store, budget, sub, legacy=legacy)
    return Arm(len(hits), len(hook_agent_context._build_block(list(hits))))


def _render_search_tool(
    store: Any, budget: int, sub: int, *, legacy: bool, bash: bool = False,
) -> Arm:
    from aelfrice import hook_search_tool, retrieval

    l1_limit = (
        hook_search_tool.BASH_INJECTED_L1_LIMIT
        if bash
        else hook_search_tool.INJECTED_L1_LIMIT
    )
    locked_ids = {
        b.id for b in store.list_locked_beliefs()
    }
    hits = retrieval.retrieve(
        store,
        QUERY,
        token_budget=budget,
        l25_token_subbudget=sub,
        l1_limit=l1_limit,
        manifest_reference_locks=True,
        # Pre-#1526 this lane had no cost function of its own and was charged
        # the `<belief …>` element it does not emit. Resolved through
        # `_lane_belief_cost` so this lane's entry in that table is the one
        # thing that names it, rather than a second copy beside it.
        belief_cost_fn=_lane_belief_cost(
            "search_tool_bash" if bash else "search_tool", legacy=legacy,
        ),
    )
    block = hook_search_tool._format_results(
        QUERY,
        list(hits),
        locked_ids,
        bash_source=("rg", QUERY) if bash else None,
    )
    return Arm(len(hits), len(block))


def _render_cli_search(store: Any, budget: int, sub: int, *, legacy: bool) -> Arm:
    """`aelf search`'s own renderer: `{prefix} {id}: {content}` per line.

    `_cmd_search` prints an 8-character lock prefix, a space, the full belief
    id, `": "`, the content and a newline. Reproduced here rather than called
    because `_cmd_search` opens the ambient store and writes to a stream; the
    line shape is the whole of what it emits per hit.
    """
    del legacy
    from aelfrice import retrieval
    from aelfrice.models import LOCK_USER

    hits = retrieval.retrieve(
        store, QUERY, token_budget=budget, l25_token_subbudget=sub,
    )
    n_bytes = 0
    for h in hits:
        prefix = "[locked]" if h.lock_level == LOCK_USER else "        "
        n_bytes += len(f"{prefix} {h.id}: {h.content}") + 1
    return Arm(len(hits), n_bytes)


_RENDERERS: dict[str, Callable[..., Arm]] = {
    "ups": _render_ups,
    "first_prompt": _render_first_prompt,
    "core": _render_core,
    "session_start": _render_session_start,
    "search_tool": _render_search_tool,
    "search_tool_bash": lambda s, b, sub, *, legacy: _render_search_tool(
        s, b, sub, legacy=legacy, bash=True,
    ),
    "agent_context": _render_agent_context,
    "cli_search": _render_cli_search,
}

LANES: tuple[str, ...] = (
    "ups",
    "first_prompt",
    "core",
    "session_start",
    "search_tool",
    "search_tool_bash",
    "agent_context",
    "cli_search",
)

# The budget the `session_start` arm is probed at. It is a literal of this
# module, not a shipped constant: the production lane passes NO token budget
# at all. #1546 deleted `hook.DEFAULT_SESSION_START_TOKEN_BUDGET` because no
# value of it could change this block — the lane retrieves on an empty query,
# only L0 contributes, and L0 is never trimmed (#379).
#
# The lane is still measured, at the value the deleted constant carried, for
# two reasons. Its rendered bytes are bytes the model receives on every
# session, whatever ends the pack. And `_measure` varies this number by
# `SATURATION_PROBE_FACTOR` and reports `pool` for both arms, which is the
# evidence for the deletion rather than an assertion of it: change the lane so
# a budget can bind and this row stops reading `pool`.
SESSION_START_PROBE_BUDGET: int = 1500


def _subbudget() -> int:
    from aelfrice.retrieval import DEFAULT_L25_TOKEN_SUBBUDGET

    return DEFAULT_L25_TOKEN_SUBBUDGET


class PoolProbeTooSmall(RuntimeError):
    """A cell about to be labelled `pool` moved when the probe was doubled.

    `pool` is defined in-module as "this arm is not evidence about any budget",
    and a reader who takes that at face value takes it from the probe. So the
    label is confirmed before it is published rather than asserted from the
    probe's arithmetic: a probe that turns out to be too small is a crash here,
    not a false label in the emitted figures.

    That confirmation is not redundant with `_probe_budget`'s upper bound.
    `_charged_tokens` is a hand transcription of a closure with no reachable
    name (`retrieve_with_tiers._cost`), and `_legacy_core_cost` and
    `_legacy_belief_tokens` are hand transcriptions of bodies on `github/main`.
    Any of them drifting from the shipped code makes the bound too low and
    every `pool` label unsafe. This render is what turns that drift into a
    failure instead of a silent under-size.
    """

    def __init__(
        self,
        lane: str,
        chars: int,
        arm_bytes: int,
        probe: int,
        moved_bytes: int,
    ) -> None:
        super().__init__(
            f"{lane} at {chars} content chars was about to be labelled `pool`: "
            f"{arm_bytes} bytes held at a probe of {probe} on both caps, but "
            f"moved to {moved_bytes} at {POOL_CONFIRM_MULTIPLE}x that probe. "
            f"`_probe_budget` is too small for this cell — one of the cost "
            f"transcriptions it sums has drifted from the shipped code."
        )
        self.lane = lane
        self.chars = chars
        self.arm_bytes = arm_bytes
        self.probe = probe
        self.moved_bytes = moved_bytes


# `_probe_budget` keyed by the store's file. Memoised because it is a scan of
# every belief in the store, it has exactly one call site — `_measure`, which
# runs two or three times per lane per grid cell — and the stores it is handed
# are an order of magnitude fewer than the calls. The memo turns that into one
# scan per store. No call or store count is published here: an earlier revision
# gave 1,192 calls against 21 stores, and instrumenting the full grid returns
# neither. Keyed by path rather than by `id(store)`: these stores are
# closed and freed between `figures()` calls and CPython reuses addresses, so
# an identity key can serve one store's bound for another's. Every store this
# module builds has its own file under a per-run tempdir, so the key is unique
# for as long as the value is valid.
_PROBE_BUDGETS: dict[str, int] = {}

# `_measure` confirms a `pool` label by re-rendering at this multiple of the
# probe on both caps. 2, because the probe is already an upper bound on what
# any candidate can cost and the question this render answers is whether that
# bound is sound, not how far past it the cell might move.
POOL_CONFIRM_MULTIPLE = 2


def _probe_budget(store: Any) -> int:
    """A budget at which every belief in `store` is affordable to every packer.

    The sum, over every belief, of the largest per-belief charge any arm or
    packer this module drives can bill for it: the shipped pack cost
    (`_charged_tokens`), its pre-#1526 body (`_legacy_belief_tokens`), and
    `<core>`'s two (`hook._core_belief_cost` and `_legacy_core_cost`). A pack
    run at this budget can afford every candidate it was offered, so a pack
    that still does not move at it was not ended by a budget.

    **This replaces a factor with a bound, and the factor was wrong.**
    `SATURATION_PROBE_FACTOR = 4` was sized for a grid topping out at 300
    content characters, and it multiplies the *lane's* budget, which is not
    what a belief costs. At 18,600 the cells that read `pool` first move at a
    `token_budget` near 9,300 — 9,300 on the before arm of every lane, 9,326
    on `agent_context`'s after arm — and the only 4x probe on this module that
    reaches it is `cli_search`'s 9,600. Every other lane's is below it,
    `ups`'s 6,000 included, so `_measure` returned `pool` — "not evidence
    about any budget" — for cells a larger probe moves. 45 of this module's
    189 arms read `pool` under the factor and 20 of them were false; under the
    bound 25 survive, including all 18 `session_start` arms (#1546).

    Two other sizings were measured and both fail:

    * *Double until two successive renders agree.* Packs are integer-quantised,
      so bytes sit flat across wide budget ranges and then jump.
      `agent_context` and `search_tool` at 18,600 are flat at 4x and 8x and
      move at 16x; `search_tool_bash` at 18,600 is flat at 4x, 8x **and** 16x
      and moves at 32x. The loop stops on the first plateau and republishes the
      false label.
    * *The largest single-belief charge.* Too small, by a factor of two. At
      18,600 that charge is 4,667 tokens, while the arms that are still flat
      at 4x — six items, all of them the store's user locks, charged 4,663
      each — first move at 9,300 (`agent_context` after arm: 9,326), which is
      twice one belief's charge in whichever currency the arm pays. Measured
      by bisecting `token_budget` against the rendered bytes with
      `l25_token_subbudget` left at its default.

    The sum is the smallest bound that survives both, and it is an upper bound
    rather than a search, so it does not depend on where the plateaus fall.
    """
    from aelfrice import hook

    key = store._db_path
    memo = _PROBE_BUDGETS.get(key)
    if memo is not None:
        return memo
    total = 0
    for bid in store.list_belief_ids():
        b = store.get_belief(bid)
        if b is None:
            continue
        total += max(
            _charged_tokens(b),
            _legacy_belief_tokens(b),
            hook._core_belief_cost(b),
            _legacy_core_cost(b),
        )
    _PROBE_BUDGETS[key] = total
    return total


def _measure(
    lane: str, store: Any, budget: int, *, chars: int, legacy: bool,
) -> tuple[Arm, str, int]:
    """Render one arm, and name which budget ended its pack.

    Returns `(arm, binds_on, probe)`, where `binds_on` is one of:

    * `token_budget` — raising `token_budget` alone moves the bytes.
    * `l25_subbudget` — raising `l25_token_subbudget` alone moves them; the
      pack ended on the L2.5 sub-cap and the lane's own budget bound nothing.
    * `both` — neither alone moves them but raising both does. The two caps
      are binding together.
    * `pool` — raising both moves nothing. The pack ended on the candidate
      pool, and this arm is not evidence about any budget.

    **Every budget that can end the pack is varied, which is the round-2
    correction.** The earlier version raised only `token_budget` while
    `l25_token_subbudget` stayed at the module default, so a pack the L2.5
    sub-cap had ended read as a budget that does not bind, and the cell was
    suppressed. Measured on `ups` at 92 content characters, against a store of
    300 beliefs: 58 hits at `token_budget=6000` and 58 at 1,500,000 with the
    sub-budget at its default 400, but 60 at `token_budget=6000` with the
    sub-budget at 1600.

    The arm is returned in every case. A `pool` arm is weak evidence about
    the budget; its bytes are still the bytes the model receives, and
    suppressing them is how the earlier version hid a +53% deviation.

    The probe is `_probe_budget(store)`, floored at
    `budget * SATURATION_PROBE_FACTOR` so no cell is probed more weakly than it
    was before. It is returned so every label is auditable from the emitted
    figures rather than from the constant behind it, and it is confirmed by a
    render at `POOL_CONFIRM_MULTIPLE` times itself before any `pool` is
    published — see `PoolProbeTooSmall`.

    `arm` itself is rendered at the shipped `budget` and the shipped `sub` in
    every case, so nothing about the probe moves a published byte or item
    count. Only the label moves.
    """
    render = _RENDERERS[lane]
    sub = _subbudget()
    f = SATURATION_PROBE_FACTOR
    # `sub * f` is in the floor for the same reason `budget * f` is: the old
    # constant survives as a floor on **both** caps. On this corpus it is never
    # the binding term (`_probe_budget` is 8,100 at the shortest grid length
    # against a 1,600 sub-floor), which is why it costs nothing to keep.
    probe = max(_probe_budget(store), budget * f, sub * f)
    ctx = _legacy_accounting if legacy else contextlib.nullcontext
    # The four labels are a precedence and not a set: the first widening whose
    # bytes differ from the arm decides, and the widenings below it are never
    # read. So each one is rendered only if the label above it did not fire.
    # `render` is a pure function of `(store, budget, sub, legacy)` — it reads
    # the store and returns counts — so this changes no label and no byte
    # count; it changes how many renders a cell costs, which on the full grid
    # is the difference between five per `_measure` call and one for the
    # common `token_budget` cell.
    with ctx():
        arm = render(store, budget, sub, legacy=legacy)
        for label, wide_budget, wide_sub in (
            ("token_budget", probe, sub),
            ("l25_subbudget", budget, probe),
            ("both", probe, probe),
        ):
            wide = render(store, wide_budget, wide_sub, legacy=legacy)
            if wide.n_bytes != arm.n_bytes:
                return (arm, label, probe)
        confirm = render(
            store,
            probe * POOL_CONFIRM_MULTIPLE,
            probe * POOL_CONFIRM_MULTIPLE,
            legacy=legacy,
        )
    if confirm.n_bytes != arm.n_bytes:
        raise PoolProbeTooSmall(
            lane, chars, arm.n_bytes, probe, confirm.n_bytes,
        )
    return (arm, "pool", probe)


# --- #1547: what the pack charges against what the lane emits ---------------


def _charged_tokens(b: Any) -> int:
    """What `retrieve_with_tiers._cost` charges for one belief.

    The compressed branch of that closure, transcribed:
    `compress_for_retrieval(b, locked=...).rendered_tokens +
    _render_wrapper_tokens(b)`. Transcribed rather than called because `_cost`
    is a closure over `retrieve_with_tiers`'s arguments and has no name a
    measurement can reach — the same reason `_render_wrapper_tokens` was given
    a module-level name in the first place.

    The `compress_on=False` branch is not reproduced:
    `resolve_use_type_aware_compression()` is published with these figures and
    defaults True.

    The `belief_cost_fn` branch is not reproduced either, and that is now a
    statement about *which lanes this applies to* rather than about none of
    them. An earlier revision of this docstring said "the two lanes that pass
    their own `belief_cost_fn` are not in `SNAPSHOT_ARM_LANES`"; since #1551
    wired `hook._ups_belief_line_cost` onto the UPS lane (`hook.py:3385`) that
    is the opposite of the truth — `ups` and `first_prompt` short-circuit
    `_cost` at `retrieval.py:4798` and never reach compression at all. This
    function is what a lane pays when it passes **no** `belief_cost_fn`, which
    among the lanes this module drives is `agent_context` alone. `_pack_charge`
    therefore charges each lane through `_lane_belief_cost`, not through this
    name unconditionally.
    """
    from aelfrice.compression import compress_for_retrieval
    from aelfrice.models import LOCK_USER
    from aelfrice.retrieval import _render_wrapper_tokens

    cb = compress_for_retrieval(b, locked=(b.lock_level == LOCK_USER))
    return cb.rendered_tokens + _render_wrapper_tokens(b)


def _emitted_chars(b: Any) -> int:
    """What the per-turn renderer emits for one belief, newline included.

    `hook._split_belief_lines` rather than an arithmetic reconstruction: this
    is the number the charge above is supposed to match, so reconstructing it
    from `render_cost` would compare two arithmetics and agree by
    construction. The resolvers this leaves unpinned (`order_policy`,
    `provenance_render`) are the same ones every other lane in this module
    resolves, and `figures()` runs in a directory with no ancestor config.
    """
    from aelfrice import hook

    belief_lines, _manifest = hook._split_belief_lines([b])
    return sum(len(line) + 1 for line in belief_lines)


def _probe_belief(content: str, retention_class: str) -> Any:
    """One unlocked, agent-inferred belief carrying `retention_class`.

    A 16-character id, because `render_cost.BELIEF_LINE_WRAPPER_CHARS` is
    itemised against that form; the 26-character ULID form undercharges the
    wrapper by 10 characters per line and is filed separately by #1547.
    """
    from aelfrice.models import LOCK_NONE, Belief

    return Belief(
        id="1547" + "0" * 12,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=5.0,
        beta=1.0,
        type="factual",
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
        origin="agent_inferred",
        retention_class=retention_class,
    )


def undercharge_table(lengths: tuple[int, ...] = LENGTH_GRID) -> dict[str, Any]:
    """Charged tokens against emitted tokens, per retention class, per length.

    One belief, one text, four retention classes — the same design as #1547's
    own table, so the two are comparable cell for cell. **It is
    `agent_context`'s table and no other lane's.** `_emitted_chars` renders
    through `hook._split_belief_lines`, which is what `ups`, `first_prompt` and
    `agent_context` all emit, but #1551 and #1552 gave the first two a
    `belief_cost_fn` that short-circuits `retrieve_with_tiers._cost` before
    compression; `agent_context` passes none, so it is the one lane that still
    pays `_charged_tokens` for a line it renders in full. See
    `SNAPSHOT_ARM_HEADLINE_LANE`.

    The ratio grows with belief length and then stops. The headline is a
    fixed-size prefix, so the charged side is flat at 44 tokens from 150
    content characters up; the emitted side grows until
    `hook.BELIEF_CONTENT_CHAR_CAP` binds and is flat at 1,265 characters — 317
    tokens — from 1,201 up. So the undercharge has a **ceiling of 7.2x** on
    this text, reached at 1,201 characters and unchanged at 18,600, and any
    single multiplier quoted from this table is still unreadable without the
    length beside it.

    **This disagrees with #1547's prior, and #1552 moved the disagreement to
    the other column.** At 18,600 characters the issue's table reads snapshot
    150.4x and transient 388.6x; this measures 7.2x and 12.7x. The charged side
    is 44 tokens here against the issue's 31, and 25 against 12, all of which
    is the `<belief>` wrapper: `_render_wrapper_tokens` adds 13 tokens to every
    compressed render, which is #1526's correction and postdates the figure the
    issue quotes; net of it the two charges agree (31 and 31, 12 and 12). The
    emitted side used to agree exactly at 4,663 tokens. It no longer does,
    because `_belief_element_line` caps the content it renders at 1,200
    characters (#1552), so the element the issue measured at 18,600 characters
    is the element this measures at 1,201. The gap the issue reported is real
    and was measured on an uncapped renderer; what is left of it is bounded by
    the cap.

    **The verbatim classes now run the other way.** `fact` and `unknown` are
    charged the whole content and emit the capped line, so above the cap they
    **over**charge: 4,663 charged against 317 emitted at 18,600 characters,
    0.1x. That is the same cap seen from the other side and it is published in
    the same columns rather than filtered out.

    The ratio is still a property of the corpus as much as of the defect — it
    is set by where the first sentence ends, which here is character 122 for a
    123-character headline — so the charged and emitted columns are published
    beside it and the multiplier is not quoted alone.
    """
    from aelfrice.compression import compress_for_retrieval
    from aelfrice.render_cost import chars_to_tokens

    out: dict[str, Any] = {}
    for chars in lengths:
        content = synthetic_content(
            random.Random(UNDERCHARGE_SEED), 0, chars, sentences=True,
        )
        row: dict[str, Any] = {}
        for retention_class in RETENTION_CLASSES_MEASURED:
            b = _probe_belief(content, retention_class)
            charged = _charged_tokens(b)
            emitted_chars = _emitted_chars(b)
            emitted = chars_to_tokens(emitted_chars)
            row[retention_class] = {
                "strategy": compress_for_retrieval(b, locked=False).strategy,
                "charged_tokens": charged,
                "emitted_chars": emitted_chars,
                "emitted_tokens": emitted,
                "ratio": round(emitted / charged, 1) if charged else None,
            }
        out[str(chars)] = row
    return out


def _pack_charge(lane: str, store: Any, budget: int, sub: int) -> dict[str, int]:
    """Charged against emitted, summed over one lane's pack's non-locked hits.

    Both halves are the lane's own. The hits come from `_lane_hits`, the same
    call the renderer makes, so `unlocked_hits` is a subset of the `n_items`
    published beside it. The charge comes from the lane's `belief_cost_fn`
    where it has one and from `_charged_tokens` where it does not, so the ratio
    is the lane's charge against the lane's emission and not a comparison of
    two lanes' accountings.

    That is what makes the ratio readable. On `ups` and `first_prompt` it is
    1.0 by #1551/#1552 — the cost function *is* the emitted line — and a
    producer that charged `_charged_tokens` there would publish a discount from
    a branch (`retrieval.py:4802`) the lane short-circuits past.

    Locks are excluded from both sums. L0 is never trimmed (#379), so a lock's
    charge is not what bought it a place in the pack; including them would
    measure that exemption rather than the compression discount, and on this
    corpus at 18,600 characters the locks are the *only* thing the control
    pack admits, so the ratio would be exactly 1.0 by selection.
    """
    from aelfrice.models import LOCK_USER
    from aelfrice.render_cost import chars_to_tokens

    hits = _lane_hits(lane, store, budget, sub, legacy=False)
    cost_fn = _lane_belief_cost(lane, legacy=False) or _charged_tokens
    unlocked = [h for h in hits if h.lock_level != LOCK_USER]
    charged = sum(cost_fn(h) for h in unlocked)
    emitted = sum(chars_to_tokens(_emitted_chars(h)) for h in unlocked)
    return {
        "hits": len(hits),
        "unlocked_hits": len(unlocked),
        "charged_tokens": charged,
        "emitted_tokens": emitted,
    }


def snapshot_arm(
    control: dict[int, Any],
    prose: dict[int, Any],
    snapshot: dict[int, Any],
    lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Each arm lane, run against the control corpus and the snapshot corpus.

    Three corpora, differing one step at a time. `control` is the corpus every
    `LANES` figure is measured on. `prose` adds the sentence boundaries the
    headline strategy needs and nothing else, so `control` → `prose` prices the
    text change on its own. `snapshot` adds `retention_class` on top of that,
    so `prose` → `snapshot` is the cost of the class and nothing else. Reading
    the class effect off `control` → `snapshot` would confound the two, which
    is why the middle column is here rather than an assertion that it does not
    matter.

    `_measure` is used on all three sides, so both budgets are varied on each
    and each reports which cap ended it, at `_probe_budget`'s bound rather than
    at a factor. At 18,600 content characters all three sides of all three
    lanes end on `token_budget`.

    `snapshot_pack_ratio` is that pack's emitted tokens over its charged
    tokens, non-locked hits only: the lane-level form of the
    `undercharge_table` cell, charged through the lane's own cost function
    (`_pack_charge`).

    **What the three lanes say is no longer the same thing, and that is the
    finding.** On `ups` and `first_prompt` the ratio is 1.0 at every length —
    those lanes charge `hook._ups_belief_line_cost`, which is the line they
    emit, so #1551/#1552 closed the charge-vs-emit gap on them and the arm now
    measures it closed rather than measuring it. `bytes_ratio` is 1.0 at every
    arm length on both: the retention class buys the pack nothing, because the
    charge no longer reads the class at all. The only column that still moves
    on them is `control` → `prose`, the cost of the text change alone — 16
    items to 17 at 300 characters — which is what that middle corpus is for.

    `agent_context` is where the gap survives, because it passes no
    `belief_cost_fn` and its pack therefore still charges the compressed form
    (see `SNAPSHOT_ARM_HEADLINE_LANE`). Its ratio is 1.3x at 300 characters, on
    4 non-locked hits charged 266 tokens against 352 emitted. At 7,170 and
    18,600 it is undefined: this lane's 600-token budget is spent by the six
    user locks — whose content is exempt from `BELIEF_CONTENT_CHAR_CAP` — before
    the pack loop reaches a candidate, so all three corpora return the same six
    locks and there is no non-locked hit to sum over. A ratio of `None` there
    is that starvation reported, not a suppressed cell; the bytes are printed
    either way.

    An earlier revision of this docstring said the control and prose sides end
    on `pool` at 18,600 and that the snapshot side admits "22 beliefs whose
    emitted text is 74,616 tokens against the 723 they were charged". Both
    halves are gone: `pool` was a probe too small to admit one belief of that
    size (`_probe_budget`), and the 22/723 pair was two different sets — 22 is
    `Arm.n_items` including six locks, 723 was summed over 16 non-locked hits.
    """
    sub = _subbudget()
    out: dict[str, Any] = {}
    for lane in SNAPSHOT_ARM_LANES:
        budget = shipped_budget(lane)
        rows: dict[str, Any] = {}
        for chars in lengths:
            c_arm, c_binds, c_probe = _measure(
                lane, control[chars], budget, chars=chars, legacy=False,
            )
            p_arm, p_binds, p_probe = _measure(
                lane, prose[chars], budget, chars=chars, legacy=False,
            )
            s_arm, s_binds, s_probe = _measure(
                lane, snapshot[chars], budget, chars=chars, legacy=False,
            )
            row: dict[str, Any] = {
                "control_items": c_arm.n_items,
                "control_bytes": c_arm.n_bytes,
                "control_binds_on": c_binds,
                "control_probe_budget": c_probe,
                "prose_items": p_arm.n_items,
                "prose_bytes": p_arm.n_bytes,
                "prose_binds_on": p_binds,
                "prose_probe_budget": p_probe,
                "snapshot_items": s_arm.n_items,
                "snapshot_bytes": s_arm.n_bytes,
                "snapshot_binds_on": s_binds,
                "snapshot_probe_budget": s_probe,
                "bytes_ratio": (
                    round(s_arm.n_bytes / p_arm.n_bytes, 2)
                    if p_arm.n_bytes
                    else None
                ),
            }
            for name, store in (("prose", prose), ("snapshot", snapshot)):
                charge = _pack_charge(lane, store[chars], budget, sub)
                row[f"{name}_charged_tokens"] = charge["charged_tokens"]
                row[f"{name}_emitted_tokens"] = charge["emitted_tokens"]
                row[f"{name}_unlocked_hits"] = charge["unlocked_hits"]
                row[f"{name}_pack_ratio"] = (
                    round(charge["emitted_tokens"] / charge["charged_tokens"], 1)
                    if charge["charged_tokens"]
                    else None
                )
            rows[str(chars)] = row
        out[lane] = rows
    return out


def dedupe_effect(store: Any, budget: int, sub: int) -> dict[str, Any]:
    """What #1547's AC2 envelope dedupe recovers, on this corpus.

    `_format_hits_with_session_start` (the shipped body, commit 570cb934)
    against the same composition with the `already_rendered` union removed —
    the body of that function on the commit before it. Transcribed here, the
    way `_legacy_accounting` transcribes the pre-#1526 cost functions, because
    the fix has no flag to turn off.

    This is the only lane in this module that can see AC2 at all: the dedupe
    only fires where a belief appears both in the session-start sub-block and
    in the per-turn pack, and a lane that renders one of those two halves in
    isolation has no repeat to find. That AC shipped with no producer able to
    measure it, which is the same gap as the snapshot arm's and is why both
    are closed in one change.
    """
    from aelfrice import hook

    hits = _lane_hits("first_prompt", store, budget, sub, legacy=False)
    block = _session_start_block(store)
    after = hook._format_hits_with_session_start(hits, block)

    belief_lines, manifest_lines = hook._split_belief_lines(hits)
    lines: list[str] = [hook.OPEN_TAG, hook._framing_header_for(hits)]
    if block:
        lines.append(block)
    lines.extend(belief_lines)
    lines.extend(hook._manifest_block_lines(manifest_lines))
    lines.append(hook.CLOSE_TAG)
    lines.append("")
    before = "\n".join(lines)

    block_ids = hook._ids_rendered_verbatim_in(block)
    repeated = sum(1 for h in hits if h.id in block_ids)
    return {
        "hits": len(hits),
        "session_start_chars": len(block),
        "repeated_ids": repeated,
        "bytes_before": len(before),
        "bytes_after": len(after),
        "pct": (
            round(100.0 * (len(after) - len(before)) / len(before), 1)
            if before
            else None
        ),
    }


# Content lengths the untuned rebuild block is reported at. Shorter than the
# full grid because this is a "did it move" figure, not a tuning.
REBUILD_LENGTHS: tuple[int, ...] = (92, 150, 300)


def _rebuild_block_bytes(stores: dict[int, Any]) -> dict[str, Any]:
    """The PreCompact rebuild block's emitted bytes, before and after.

    `context_rebuilder._estimate_belief_tokens` is left on the old currency by
    #1526 on purpose, because `DEFAULT_REBUILDER_TOKEN_BUDGET` is reachable
    from `.aelfrice.toml` as `[rebuilder] token_budget` and re-denominating it
    would silently reinterpret values users have already written.

    Measured anyway, because #1526 round 1 justified leaving it with "correcting
    it would move this block's bytes", and whether that is true is a question
    with an answer rather than a premise. The answer is: only where this
    block's own budget binds. `rebuild_v14` takes its non-locked candidates
    from `retrieve()`, whose pack cost #1526 changed, so the hits reaching this
    formatter *can* be a different set -- but at 92 and 150 content characters
    they are the same set and the emitted bytes are identical (9,011 -> 9,011
    and 12,178 -> 12,178), because `DEFAULT_REBUILDER_TOKEN_BUDGET` is nowhere
    near binding there. At 300 it binds and the bytes move (19,293 -> 16,421).

    So the round-1 justification is not so much wrong as unstated: it holds at
    one of the three lengths this function measures and not at the other two.
    That is the reason to produce these numbers rather than assert them, and
    it is why the byte figures quoted in `context_rebuilder._format_block`
    carry `derived:` markers back to the keys below. An earlier revision of
    this docstring said the bytes "are not" constant, flatly; that was written
    before the cost functions settled and is what this paragraph replaces.
    """
    from aelfrice.context_rebuilder import RecentTurn, rebuild_v14
    from aelfrice.rebuild_log import DEFAULT_REBUILDER_TOKEN_BUDGET

    turns = [RecentTurn(role="user", text=QUERY, session_id="s1")]
    out: dict[str, Any] = {}
    for chars in REBUILD_LENGTHS:
        store = stores[chars]
        with _legacy_accounting():
            before = len(
                rebuild_v14(
                    turns, store,
                    token_budget=DEFAULT_REBUILDER_TOKEN_BUDGET,
                    rebuild_log_enabled=False,
                )
            )
        after = len(
            rebuild_v14(
                turns, store,
                token_budget=DEFAULT_REBUILDER_TOKEN_BUDGET,
                rebuild_log_enabled=False,
            )
        )
        out[f"rebuild_bytes_before_{chars}"] = before
        out[f"rebuild_bytes_after_{chars}"] = after
    return out


def shipped_budget(lane: str) -> int:
    """The budget this lane is measured at, read off the module that holds it.

    Every entry but one is the lane's shipped constant. `session_start` has
    no shipped constant — the production lane passes no budget — so it is
    probed at this module's own `SESSION_START_PROBE_BUDGET`.
    """
    from aelfrice import hook, hook_agent_context, hook_search_tool, retrieval

    return {
        "ups": hook.DEFAULT_HOOK_TOKEN_BUDGET,
        # The composed envelope's per-turn half is the UPS pack, at the UPS
        # budget. The `<core>` half inside it carries its own budget and
        # `<locked>` carries none; this is the only one that can end this
        # lane's `retrieve()`.
        "first_prompt": hook.DEFAULT_HOOK_TOKEN_BUDGET,
        "core": hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET,
        "session_start": SESSION_START_PROBE_BUDGET,
        "search_tool": hook_search_tool.INJECTED_TOKEN_BUDGET,
        "search_tool_bash": hook_search_tool.BASH_INJECTED_TOKEN_BUDGET,
        "agent_context": hook_agent_context.INJECTED_TOKEN_BUDGET,
        "cli_search": retrieval.DEFAULT_TOKEN_BUDGET,
    }[lane]


def figures(*, lengths: tuple[int, ...] = LENGTH_GRID) -> dict[str, Any]:
    """Re-derive every #1526 figure this repo publishes.

    `lengths` narrows the content-length grid. The full grid is what the
    CHANGELOG publishes; a caller that only needs the headline cell passes a
    shorter one, which is how `tests/test_render_cost_1526.py` keeps its
    acceptance run inside the CI per-test timeout.
    """
    from aelfrice import hook
    from aelfrice.render_cost import BELIEF_LINE_WRAPPER_CHARS
    from aelfrice.retrieval import resolve_use_type_aware_compression

    values: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Hermetic: `retrieve()` resolves ~22 `[retrieval]` flags by walking
        # up from the process cwd, and this repo sits under a home directory
        # that has an `.aelfrice.toml`. A tempdir has no ancestor config, so
        # the run reads the shipped defaults and nobody's local file.
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            values["compression"] = resolve_use_type_aware_compression()
            values["belief_line_wrapper_chars"] = BELIEF_LINE_WRAPPER_CHARS
            values["framing_header_chars"] = len(hook._FRAMING_HEADER)
            values["store_beliefs"] = STORE_BELIEFS
            values["lengths"] = list(lengths)
            values["l25_token_subbudget"] = _subbudget()

            stores = {
                chars: _synthetic_store(tmp / f"s{chars}.db", chars)
                for chars in lengths
            }
            arm_lengths = tuple(c for c in SNAPSHOT_ARM_LENGTHS if c in lengths)
            prose_stores = {
                chars: _synthetic_store(
                    tmp / f"prose{chars}.db", chars, sentences=True,
                )
                for chars in arm_lengths
            }
            snap_stores = {
                chars: _synthetic_store(
                    tmp / f"snap{chars}.db",
                    chars,
                    snapshot_every=SNAPSHOT_EVERY,
                    sentences=True,
                )
                for chars in arm_lengths
            }
            try:
                values["corpus_shape"] = corpus_shape(stores[lengths[0]])
                values["snapshot_every"] = SNAPSHOT_EVERY
                values["sentence_chars"] = SENTENCE_CHARS
                values["snapshot_arm_lengths"] = list(arm_lengths)
                if arm_lengths:
                    # Read at the *top* of the arm grid, and the control beside
                    # them at the same length. `arm_lengths[0]` is 92, below
                    # `SENTENCE_CHARS`, where no belief closes a sentence: all
                    # three corpora carry `sentence_headline` 0 there and the
                    # middle column is indistinguishable from the control by
                    # construction. A shape read on that length can confirm the
                    # retention class was written but cannot confirm the text
                    # change was, which is half of what these corpora are for.
                    arm_top = arm_lengths[-1]
                    values["snapshot_corpus_shape"] = corpus_shape(
                        snap_stores[arm_top]
                    )
                    values["prose_corpus_shape"] = corpus_shape(
                        prose_stores[arm_top]
                    )
                    values["control_corpus_shape_at_arm_top"] = corpus_shape(
                        stores[arm_top]
                    )
                    values["snapshot_arm"] = snapshot_arm(
                        stores, prose_stores, snap_stores, arm_lengths,
                    )
                values["undercharge"] = undercharge_table(lengths)
                for lane in LANES:
                    values.update(_lane_figures(lane, stores, lengths))
                values["first_prompt_recent_work_chars"] = _recent_work_chars_in(
                    _session_start_block(
                        stores[values["first_prompt_headline_chars"]]
                    )
                )
                values["dedupe"] = dedupe_effect(
                    stores[values["first_prompt_headline_chars"]],
                    shipped_budget("first_prompt"),
                    _subbudget(),
                )
                values.update(_flat_1547_keys(values, lengths))
                if set(REBUILD_LENGTHS) <= set(lengths):
                    values.update(_rebuild_block_bytes(stores))
            finally:
                for group in (stores, prose_stores, snap_stores):
                    for store in group.values():
                        store.close()
        finally:
            os.chdir(cwd)
    return values


def _flat_1547_keys(
    values: dict[str, Any], lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Flatten the #1547 headline cells to scalar keys.

    `scripts/check_derived_figures.py` re-runs this module and compares one
    published number against one emitted key, so a figure that lives only
    inside a nested dict cannot be gated. These are the cells the CHANGELOG
    entry quotes, lifted to the top level so the gate can re-derive them.
    """
    top = str(max(lengths))
    row = values["undercharge"][top]
    out: dict[str, Any] = {
        "undercharge_top_chars": max(lengths),
        "undercharge_top_emitted_tokens": row["fact"]["emitted_tokens"],
        "undercharge_top_snapshot_charged_tokens": row["snapshot"][
            "charged_tokens"
        ],
        "undercharge_top_snapshot_ratio": row["snapshot"]["ratio"],
        "undercharge_top_transient_ratio": row["transient"]["ratio"],
    }
    # The arm corpus's shape, flattened. `SNAPSHOT_EVERY` is the one constant
    # in this module with no scalar key behind it, and it showed: mutating it
    # from 7 to 3 left `tests/test_render_cost_1526.py` green **and**
    # `scripts/check_derived_figures.py --mode all` at exit 0 while the
    # published corpus went from 43/42 snapshot beliefs to 100/98. The arm's
    # byte and item figures are insensitive to a 2.3x change in snapshot
    # density — the packs are ended by budgets and locks, not by how many
    # candidates carry the class — so they cannot stand in for the stride.
    # Read off `corpus_shape`, which counts the column back off the store, so
    # these keys also fail if the class stops being written.
    shape = values["snapshot_corpus_shape"]
    out["snapshot_corpus_snapshot"] = shape["snapshot"]
    out["snapshot_corpus_snapshot_unlocked"] = shape["snapshot_unlocked"]
    d = values["dedupe"]
    out["dedupe_repeated_ids"] = d["repeated_ids"]
    out["dedupe_bytes_before"] = d["bytes_before"]
    out["dedupe_bytes_after"] = d["bytes_after"]
    # The composed lane's own snapshot-arm cell, at the top of the grid. Lifted
    # here so the gate has a `first_prompt` arm figure to re-derive: every other
    # published arm number is the `ups` lane's, and a lane set that quietly lost
    # `first_prompt` would leave the CHANGELOG's composed-envelope paragraph
    # backed by nothing. Indexed rather than probed, so that loss is a crash.
    cell = values["snapshot_arm"]["first_prompt"][top]
    out["snapshot_arm_first_prompt_prose_bytes"] = cell["prose_bytes"]
    out["snapshot_arm_first_prompt_snapshot_bytes"] = cell["snapshot_bytes"]
    out["snapshot_arm_first_prompt_snapshot_items"] = cell["snapshot_items"]
    # The headline lane's cell, lifted whole. Its five keys travel together so
    # the denominators cannot drift apart in the prose: `items` is the whole
    # pack, `unlocked_hits` is the subset the charge and the emission are
    # summed over, and the ratio is those two sums. The CHANGELOG previously
    # read "admitting 22 beliefs charged 723 tokens" — 22 from `Arm.n_items`
    # including six locks, 723 from a sum over 16 non-locked hits, and no
    # single set of beliefs that was both.
    #
    # At the top of this grid the pack is all locks and the three sums are 0,
    # 0 and None; see `SNAPSHOT_ARM_HEADLINE_LANE` for why, and read the arm's
    # ratio off its 300-character row. The lane's *single-belief* ratio is
    # `undercharge_top_snapshot_ratio` above, which is this lane's number and
    # no other lane's now that `ups` and `first_prompt` charge what they emit.
    #
    # `ups` gets the same five, because the sentence they back is the one the
    # mixed denominator was found in and it quotes all of them at once.
    for lane in ("ups", SNAPSHOT_ARM_HEADLINE_LANE):
        out.update(_arm_cell_keys(values["snapshot_arm"][lane][top], lane))
    return out


def _arm_cell_keys(cell: dict[str, Any], lane: str) -> dict[str, Any]:
    """One snapshot-arm cell's five snapshot-side figures, as scalar keys.

    They are lifted as a set rather than one at a time because the CHANGELOG
    sentence they back reads all five in one breath, and the defect they were
    added for was a sentence that mixed two of them: "admitting 22 beliefs
    charged 723 tokens whose emitted text is 74,616 tokens, 103.2x" put
    `Arm.n_items` — 22, six of them locks — in the same clause as a charge
    `_pack_charge` summed over the 16 non-locked hits. No set of beliefs was
    both. Gated together, a sentence that says "admitting N of which the M
    non-locked ones are charged X against Y" cannot have N and M drift apart
    without the marker check failing.
    """
    return {
        f"snapshot_arm_{lane}_snapshot_items": cell["snapshot_items"],
        f"snapshot_arm_{lane}_snapshot_unlocked_hits": cell[
            "snapshot_unlocked_hits"
        ],
        f"snapshot_arm_{lane}_snapshot_charged_tokens": cell[
            "snapshot_charged_tokens"
        ],
        f"snapshot_arm_{lane}_snapshot_emitted_tokens": cell[
            "snapshot_emitted_tokens"
        ],
        f"snapshot_arm_{lane}_snapshot_pack_ratio": cell["snapshot_pack_ratio"],
    }


def _lane_figures(
    lane: str, stores: dict[int, Any], lengths: tuple[int, ...],
) -> dict[str, Any]:
    """One lane's full curve, plus the headline cell at the corpus median.

    The headline cell is the grid length nearest `CORPUS_MEDIAN_CHARS` that is
    present, chosen by proximity and not by how flattering the number is. There
    is no tuning here and therefore no tuning length: the budget is the same in
    both arms, so every cell is a measurement of the same change.
    """
    budget = shipped_budget(lane)
    curve = _curve(lane, stores, budget, lengths)
    headline_chars = min(lengths, key=lambda c: (abs(c - CORPUS_MEDIAN_CHARS), c))
    headline = curve[str(headline_chars)]
    return {
        f"{lane}_budget": budget,
        f"{lane}_headline_chars": headline_chars,
        f"{lane}_bytes_before": headline["before"],
        f"{lane}_bytes_after": headline["after"],
        f"{lane}_items_before": headline["before_items"],
        f"{lane}_items_after": headline["after_items"],
        f"{lane}_pct": headline["pct"],
        f"{lane}_pool_equality": headline["pool_equality"],
        f"{lane}_curve": curve,
    }


def _curve(
    lane: str, stores: dict[int, Any], budget: int, lengths: tuple[int, ...],
) -> dict[str, Any]:
    """Emitted bytes before and after, at every length in the grid.

    Every cell carries a byte count. `before_binds_on` / `after_binds_on`
    name which budget ended that arm's pack (see `_measure`); `pool_equality`
    is true only when both arms read `pool`, which is the one case where the
    two numbers say nothing about any budget because they are the same
    candidate pool rendered twice.

    `before_probe_budget` / `after_probe_budget` carry the budget those labels
    were established at, so a reader can check the label against the number
    that produced it instead of against the constant behind it. The old
    `SATURATION_PROBE_FACTOR` probe was published nowhere, which is part of why
    16 false `pool` labels survived a review that read the emitted figures.

    A byte count of **zero** is a measurement, not a suppressed cell, and the
    extended grid produces three — **all of them in the before arm**. The
    pre-#1526 charge `max(1, len(content) // 4)` is uncapped, so at 6,004,
    7,170 and 18,600 content characters one `<core>` line costs 1,501, 1,792
    and 4,650 tokens against `DEFAULT_SESSION_START_CORE_TOKEN_BUDGET = 1500`,
    and `_pack_core_candidates` — which skips an oversized belief rather than
    breaking — packs none of 300 candidates and the section emits nothing. The
    old grid stopped at 300 characters and never reached that.

    The **after** arm cannot empty at any length. `_core_belief_line` truncates
    the content at `hook.BELIEF_CONTENT_CHAR_CAP` (#1552), so
    `hook._core_belief_cost` — the function `_pack_core_candidates` actually
    charges with — plateaus at 320 tokens for any content past 1,200
    characters and 1,220 for the pathological case where every character is an
    angle bracket `_escape_for_hook_block` expands fourfold. Both are under the
    budget, so the shipped arm always packs at least one line; its `<core>`
    curve bottoms out at 5,120 bytes. That is the reverse of the relation this
    grid was extended to reach, and it is why the ninth length moved from
    5,950 to 6,004: the two accountings now cross the budget in one place
    only, and it is the pre-#1526 half that crosses.

    `pct` is None where the before arm is zero, because a percentage of nothing
    is not a number this module is willing to print.
    """
    rows: dict[str, Any] = {}
    for chars in lengths:
        store = stores[chars]
        row: dict[str, Any] = {}
        for name, legacy in (("before", True), ("after", False)):
            arm, binds_on, probe = _measure(
                lane, store, budget, chars=chars, legacy=legacy,
            )
            row[name] = arm.n_bytes
            row[f"{name}_items"] = arm.n_items
            row[f"{name}_binds_on"] = binds_on
            row[f"{name}_probe_budget"] = probe
        row["pool_equality"] = (
            row["before_binds_on"] == "pool" and row["after_binds_on"] == "pool"
        )
        b, a = row["before"], row["after"]
        row["pct"] = round(100.0 * (a - b) / b, 1) if b else None
        rows[str(chars)] = row
    return rows


def _budget_label(lane: str, budget: int) -> str:
    """How to read the number this lane was measured at.

    Every lane but `session_start` was measured at its shipped constant, held
    at the same value in both arms. `session_start` has no shipped constant to
    hold: the production lane passes no budget, so the number is this module's
    probe value and the reader must not take it for a setting.
    """
    if lane == "session_start":
        return f"probe budget {budget}; the lane itself passes none"
    return f"budget {budget} unchanged"


def _flag(row: dict[str, Any]) -> str:
    """The one-line legend for a cell, saying what is actually true of it."""
    if row["pool_equality"]:
        return (
            "  [both arms ended on the candidate pool: a pool equality, "
            "not budget evidence]"
        )
    marks = [
        f"{name} ended on {row[f'{name}_binds_on']}"
        for name in ("before", "after")
        if row[f"{name}_binds_on"] != "token_budget"
    ]
    return f"  [{'; '.join(marks)}]" if marks else ""


def _print_1547(values: dict[str, Any]) -> None:
    """The #1547 sections: the undercharge table, the arm, and the dedupe."""
    print()
    print("--- #1547 charged vs emitted, one belief, by retention class ---")
    print(
        f"  {'chars':>6}  {'class':<9} {'strategy':<9} {'charged':>8} "
        f"{'emitted':>8}  ratio"
    )
    for chars, row in values["undercharge"].items():
        for name in RETENTION_CLASSES_MEASURED:
            cell = row[name]
            ratio = "n/a" if cell["ratio"] is None else f"{cell['ratio']}x"
            print(
                f"  {chars:>6}  {name:<9} {cell['strategy']:<9} "
                f"{cell['charged_tokens']:>8} {cell['emitted_tokens']:>8}  "
                f"{ratio}"
            )
    if "snapshot_arm" in values:
        print()
        print(
            "--- #1547 snapshot arm: control -> prose (text only) -> snapshot "
            "(text + class) ---"
        )
        for lane, rows in values["snapshot_arm"].items():
            print(f"  {lane} (budget {values[f'{lane}_budget']}):")
            for chars, row in rows.items():
                ratio = (
                    "n/a" if row["bytes_ratio"] is None else f"{row['bytes_ratio']}x"
                )
                pack = (
                    "n/a"
                    if row["snapshot_pack_ratio"] is None
                    else f"{row['snapshot_pack_ratio']}x"
                )
                print(
                    f"    {chars:>6} chars: items {row['control_items']} -> "
                    f"{row['prose_items']} -> {row['snapshot_items']}, bytes "
                    f"{row['control_bytes']} -> {row['prose_bytes']} -> "
                    f"{row['snapshot_bytes']} ({ratio} over prose)  "
                    f"[ended on {row['control_binds_on']} / "
                    f"{row['prose_binds_on']} / {row['snapshot_binds_on']}]  "
                    f"snapshot pack charged {row['snapshot_charged_tokens']} "
                    f"tok, emitted {row['snapshot_emitted_tokens']} tok ({pack})"
                )
    d = values["dedupe"]
    print()
    print("--- #1547 AC2 envelope dedupe, measured on the composed lane ---")
    print(
        f"  {d['hits']} hits, {d['repeated_ids']} of them already rendered in a "
        f"{d['session_start_chars']}-char session-start sub-block: "
        f"{d['bytes_before']} -> {d['bytes_after']} bytes "
        f"({d['pct']:+.1f}%)"
    )
    print(
        f"  <recent-work> inside the sub-block this lane rendered: "
        f"{values['first_prompt_recent_work_chars']} chars"
    )


def main(argv: list[str] | None = None) -> int:
    """Print the figures, or emit them as JSON.

    `--lengths` is the same knob `figures()` already takes, reached from the
    command line. Without it the grid is `LENGTH_GRID` and the run is what
    every published figure is derived from; with it the grid is whatever was
    asked for, which is how a caller that only needs the *shape* of the
    output — the legends, the per-lane lines — pays for one store per length
    instead of nineteen. `values["lengths"]` carries the grid that ran, so a
    narrowed `--emit-figures` blob says so about itself.
    """
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--emit-figures",
        action="store_true",
        help="emit a flat JSON object of key -> value on stdout and nothing else",
    )
    ap.add_argument(
        "--curve",
        action="store_true",
        help="print the per-length curve for each lane as well as the summary",
    )
    ap.add_argument(
        "--lengths",
        metavar="N[,N...]",
        help=(
            "content lengths to measure, comma-separated; the default is the "
            "published grid and is what every published figure is read from"
        ),
    )
    args = ap.parse_args(argv)

    lengths = LENGTH_GRID
    if args.lengths:
        lengths = tuple(int(n) for n in args.lengths.split(","))
    values = figures(lengths=lengths)
    if args.emit_figures:
        json.dump(values, sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    flat = {
        k: v
        for k, v in values.items()
        if not k.endswith("_curve") and not isinstance(v, (dict, list))
    }
    width = max(len(k) for k in flat)
    for key in sorted(flat):
        print(f"{key:<{width}}  {flat[key]}")
    print(f"corpus_shape           {values['corpus_shape']}")
    if "snapshot_corpus_shape" in values:
        print(f"snapshot_corpus_shape  {values['snapshot_corpus_shape']}")
    print()
    for lane in LANES:
        before = values[f"{lane}_bytes_before"]
        after = values[f"{lane}_bytes_after"]
        chars = values[f"{lane}_headline_chars"]
        row = values[f"{lane}_curve"][str(chars)]
        print(
            f"{lane}: {_budget_label(lane, values[f'{lane}_budget'])}, "
            f"emitted bytes {before} -> {after} "
            f"({values[f'{lane}_pct']:+.1f}%) at {chars} "
            f"content chars{_flag(row)}"
        )
    _print_1547(values)
    if args.curve:
        print()
        for lane in LANES:
            print(f"--- {lane} ({_budget_label(lane, values[f'{lane}_budget'])}) ---")
            for chars, row in values[f"{lane}_curve"].items():
                pct = "" if row["pct"] is None else f"  ({row['pct']:+.1f}%)"
                print(
                    f"  {chars:>4} chars: {row['before']:>8} -> "
                    f"{row['after']:>8}{pct}"
                    f"  items {row['before_items']} -> {row['after_items']}"
                    f"{_flag(row)}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
