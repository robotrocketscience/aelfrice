# Injection-ledger kill experiment — re-spec (#1252)

Parent: #1177 proposal 11 (session injection ledger + turn-differential
lock rendering), funded by the operator ruling of 2026-07-31 on the
condition that its kill experiment be re-specced before any build.

This document is that re-spec, plus the recorded result of running it.
**No ledger code is proposed here.** The deliverable is a runnable
check and a verdict.

## 1. Why the original check could not be run

#1177 specified:

> run the free precondition check: count SessionStart rows with
> `source=='compact'` in `hook_audit.jsonl` against observed context
> resets.

#1252 blocked this on the right-hand side: "observed context resets" is
not a quantity the audit log records, so the check reduces to counting
compact events against themselves.

That diagnosis is correct and **understates the problem**. The
left-hand side is not computable either. Three independent facts about
`hook_audit.jsonl`, each read off `src/aelfrice/hook.py`:

1. **`source` is never written.** `_write_hook_audit_record` takes no
   `source` parameter and the `session_start` call site passes none.
   Its `session_start` rows cannot be separated into `startup` /
   `resume` / `clear` / `compact`. Confirmed on a live log: 0 of 266
   records carry a `source` key, across 40 `session_start` rows.
2. **The row is conditional on a non-empty baseline block.** The audit
   write is nested inside `if body:`. A `SessionStart` that fires
   against an empty locked set writes nothing, so the row count is not
   a count of firings even setting (1) aside.
3. **The rebuild block is not in the row.** `rendered_block` is the
   baseline `body`, built and recorded *before* the compact-only
   rebuild block is appended to stdout, so compact-ness cannot be
   recovered post-hoc from the stored block.

There is therefore no repair to the original check that keeps
`hook_audit.jsonl` as the substrate. It needs a different one.

## 2. The witness, and why it is independent

The host transcript records both sides, as two distinct record types,
and aelfrice emits neither:

| side | record | source |
|---|---|---|
| a context reset happened | `subtype == "compact_boundary"`, with `compactMetadata.trigger` (`manual` / `auto`) and pre/post token counts | host |
| the hook fired for it | a record carrying the `SessionStart:compact` hook-result marker | host |

This is the denominator #1252 says is unobtainable. It is obtainable —
it was simply not in the log that was being searched. The transcript is
not a bespoke channel either: it is the artifact the hook payload
already names (`transcript_path`, read at `hook.py` `_TRANSCRIPT_PATH_KEY`).

Because both records are host-emitted, neither side can be
self-confirming: aelfrice cannot manufacture a `compact_boundary`, and
cannot suppress one.

## 3. Pairing rule

Aggregate equality is not evidence — `n == n` can be two unrelated
populations. Each `compact_boundary` is paired with the **first**
`SessionStart:compact` marker that follows it in the same session file,
in file order; each marker satisfies at most one boundary, oldest
first. A burst of markers therefore cannot paper over a run of unfired
boundaries.

The **final** boundary of a session is reported as `trailing` and
excluded from the rate: the session may have ended between the reset
and the next fire, which is truncation rather than unreliability.
Counting it as a failure biases the rate down by about one per session.

## 4. Decision rule — fixed before the run

`fire_rate = fired / (fired + unfired)` over scoreable (non-trailing)
boundaries.

| condition | outcome |
|---|---|
| `fire_rate >= 98%` | **CLEARS** — precondition holds |
| `fire_rate < 90%` | **KILLS** — no dependable epoch; proposal 11 closes |
| `90% <= fire_rate < 98%` | **GREY** — kill *unless* a second, independent epoch trigger (the turn-count TTL backstop #1177 gestures at) is specified and measured first |
| fewer than 20 scoreable boundaries | **NO VERDICT** — underpowered; an underpowered pass is not a pass |

`manual` and `auto` are scored separately as well as pooled. **If the
two rates diverge by more than 10 percentage points, the pooled rate is
not the verdict.** A design that holds for `/compact` but not for
auto-compaction has not cleared: auto is the case the user cannot see
coming, and is the case the always-injected guarantee exists for.

Ordering disclosure: a scoping pass established that both record types
exist and that their aggregate counts matched on a 25-file sample. The
rule above was fixed before the per-boundary join was written or run.

## 5. Result

`scripts/epoch_precondition_check.py`, run over the full local
transcript corpus, no window:

```
files scanned        : 1460
sessions w/ boundary : 49
sessions w/o boundary: 1411
boundaries total     : 92
  scoreable          : 73
  trailing (excluded): 19
markers seen         : 76
markers unpaired     : 4

fired                : 72
unfired              : 1
fire_rate            : 98.6%

--- by trigger ---
auto     n=   1  fired=   0  rate=0.0%
manual   n=  72  fired=  72  rate=100.0%

trigger divergence   : 100.0pp (limit 10pp)
VERDICT: NO VERDICT (triggers diverge by 100.0pp > 10pp; the pooled rate is
not the verdict)
```

**The pooled 98.6% clears the bar and is not the verdict.** The
divergence guard fired, and it fired on something real: the corpus
contains exactly **one** auto-compaction boundary, and that one did not
pair with a fire. `n = 1` is not evidence that auto is broken. It is
evidence that **the auto path is unmeasured**, which under the
pre-registered rule is `NO VERDICT`, not a pass.

So the honest reading is:

- **`manual` clears decisively** — 72/72, well past the 98% bar and
  past the 20-boundary power floor.
- **`auto` has no verdict** — one observation, below the power floor by
  a factor of twenty.
- The precondition as a whole therefore **has not cleared**.

### The bigger number is not the fire rate

**1411 of 1460 sessions (96.6%) contain no compaction boundary at all.**

The epoch increments on compaction and on session creation, and nothing
else. In a session that never compacts, the ledger renders `full` once
and `manifest` for every turn thereafter, however reliable the compact
event is. That describes 96.6% of sessions in this corpus — the
always-injected guarantee degrading to never-reinjected is not an edge
case in the tail, it is the ordinary case.

This is not a precondition failure and the check does not score it as
one: there is no boundary in those sessions to score. It is a design
question, and it lands on #1177 rather than here. But it dwarfs the
fire rate in consequence, and proposal 11 should not be built against a
98.6% number while this one stands unaddressed.

## 6. Known limitation

The numerator matches a marker that the host writes when it records the
hook result. On this corpus 39 of 77 compact-marker lines carry no
`aelfrice-baseline` block, so the marker is present when aelfrice emits
nothing — it witnesses the firing, not merely the output. That is the
favourable direction, but it is an inference from the host's recording
behaviour rather than a documented contract, and a host change could
invalidate it silently. If the auto path is ever powered up, the
numerator should be re-derived from an aelfrice-side record of the
`source` field, which does not exist today (§1.1).

## 7. Recommendation to #1177

Proposal 11 stays **blocked**. Two things gate it, in order:

1. **Power up the auto path**, or state explicitly that the design
   accepts an unmeasured auto path and why. One observation is not a
   measurement. The cheapest route is a `--since`-windowed re-run once
   the corpus has accumulated auto-compaction boundaries.
2. **Answer the never-compacted session** — 96.6% of the corpus. Either
   the epoch gets the second increment trigger, or the design accepts
   the degradation and says so in the proposal rather than in a
   footnote.

The fifth acceptance item of #1252 asks whether the 20-turn scripted
compliance A/B is still the right second stage. **It is, and it does
not inherit the denominator problem** — it is a within-session
comparison of two rendering modes over a fixed scripted turn sequence,
so its denominator is the scripted turn count, which is fixed by
construction. It should not be run until (1) and (2) are resolved,
because it measures whether the ledger *helps*, not whether it *fires*,
and a ledger that never re-injects would still score well on a
20-turn session that compacts once.
