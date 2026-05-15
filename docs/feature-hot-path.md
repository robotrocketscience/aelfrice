# Hot-path touch state (v1)

**Status (v3.x):** storage substrate shipped, retrieval consumer **deferred-with-evidence** (post-R7c).

The hot-path touch state is a per-(belief, session) sidecar that
records the most-recent `fire_idx` at which each belief was injected
into the agent's context. v1 ships the write path and inspection
surface. The originally-planned rerank consumer that would read this
state is no longer scheduled — R7c (see ["Why no consumer in
v1"](#why-no-consumer-in-v1) below) found production posterior-touch
correlation above the pre-committed crossover on two corpora; the
synthetic-baseline signal that motivated the consumer mostly
evaporates at production correlation levels.

## What it is

`belief_touches` is a SQLite sidecar table next to `injection_events`
(#779). Where `injection_events` records every (turn × belief) inject
row for the close-the-loop relevance sweeper, `belief_touches` keeps
only the *last* touch per (belief, session) with a touch count and an
event-kind bitmask. The two tables answer different questions:

| Table | Read shape | Cardinality | Consumer |
|---|---|---|---|
| `injection_events` | "did the assistant reference this belief?" | one row per (turn × belief) | #779 sweeper |
| `belief_touches` | "was this belief recently in the prompt?" | one row per (belief × session) | (deferred-with-evidence — see #848) |

The intended consumer for `belief_touches` is a posterior-rerank
multiplier that boosts beliefs touched in the last K fires of the
current session. v1 writes the state but no production caller reads
it.

## Schema

```sql
CREATE TABLE belief_touches (
    belief_id           TEXT    NOT NULL,
    session_id          TEXT    NOT NULL,
    last_fire_idx       INTEGER NOT NULL,
    touch_count         INTEGER NOT NULL DEFAULT 0,
    event_kinds_bitmask INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (belief_id, session_id),
    FOREIGN KEY (belief_id) REFERENCES beliefs(id) ON DELETE CASCADE
);
CREATE INDEX idx_belief_touches_session_fire
    ON belief_touches(session_id, last_fire_idx DESC);
```

`event_kinds_bitmask` reserves four bits:

| Bit | Constant | Written in v1? |
|---|---|---|
| 0 | `TOUCH_EVENT_KIND_INJECTION` | yes (the only one) |
| 1 | `TOUCH_EVENT_KIND_RETRIEVE_HIT` | no — H4 REFUTED at R4/R4e/R5 |
| 2 | `TOUCH_EVENT_KIND_BFS_VISIT` | no |
| 3 | `TOUCH_EVENT_KIND_USER_ACTION` | no — H4a deferred |

## How it gets written

On every UserPromptSubmit hook fire that produces a rebuild block, the
hook:

1. Calls `_ring_append_ids(session_id, injected_ids, ...)` — the
   existing #744 JSON ring append. Returns the session's
   `next_fire_idx`.
2. Calls `_record_touches(session_id, injected_ids,
   fire_idx=next_fire_idx - 1, ...)`. The fire_idx the touches receive
   is the same one the ring just assigned, so the JSON ring and the
   sidecar share a counter.
3. Inside `_record_touches`, the supplied `injected_ids` are each
   recorded via `record_touch`. The hook is **forward-only** — it does
   NOT read the JSON ring or backfill pre-substrate entries. An earlier
   revision tried a one-shot per-session migration off the JSON ring,
   but `record_touch` uses `ON CONFLICT DO UPDATE` so the replay was
   non-idempotent: every UPS fire re-bumped `touch_count` on every ring
   entry. The migration is gone; ring entries that predate this table
   are simply not represented in `belief_touches`.

`record_touch` upserts: a new (belief, session) pair inserts with
`touch_count=1` and the supplied `event_kind` bit set; an existing
pair refreshes `last_fire_idx`, increments `touch_count` by one, and
OR-s the event_kind bit into the bitmask.

## Locked decisions honored

- **Determinism (#605, `c06f8d575fad71fb`).** `last_fire_idx` is a
  monotonic integer per session, not a wall-clock timestamp. Same
  query + same store + same fire_idx → same window contents across
  replays.
- **Federation (#661, `d0c5ecdebb3f0f4d`).** The composite primary key
  `(belief_id, session_id)` keeps foreign federated beliefs cold every
  read by construction. No touch row crosses the federation boundary.
- **PHILOSOPHY narrow surface (#605).** The decay shape is one
  integer comparison (`is_hot`); no embedding, no ML, no LLM. Pure
  stdlib.
- **Audit-immutable `beliefs` table.** Sidecar table preserves the
  per-turn audit invariant — touch-state updates don't mutate belief
  rows.

## Why no consumer in v1

The retrieval-consumer plan was a rerank-stage posterior multiplier
that boosts beliefs `is_hot(b, current_fire_idx, K)` returns True
for. DESIGN.md v1 (`experiments/hot-path/DESIGN.md` in the lab)
gated the consumer flip on two preconditions; both have now resolved:

1. **PR #782** (v3.1 `JUDGE_PROMPT_TEMPLATE` sharpening + hot_start
   fixture widening) — landed 2026-05-14.
2. **R7c production-posterior-temperature ρ measurement** — completed
   2026-05-15 against two real per-project DBs.

R7c result (cross-corpus):

| Corpus | Session events | Touched | ρ_mixed (load-bearing) | Band (per R7b) |
|---|---|---|---|---|
| aelfrice repo | 168 | 14 | **+0.8745** | `SHIP_H4_ONLY` |
| Independent project | 214 | 89 | **+0.7244** | `SHIP_H4_ONLY` |

Both ρ_mixed values are above the pre-committed 0.60 crossover
that R7b identified as the threshold above which R7's
91% top-K shift signal mostly evaporates. Cross-corpus agreement
across two independent project shapes strengthens the read.

**Decision: the consumer flip is not scheduled.** The originally-
modelled posterior-rerank touch-temperature multiplier is
deferred-with-evidence — not "we haven't measured yet," but "we
measured and the evidence doesn't justify the build." See #848 for
the tracker and re-opening conditions. The substrate writes shipped
here remain in place and remain useful for inspection (`aelf doctor
--hot-path`) and for any future consumer that wants a different
mechanism.

Caveat: each R7c corpus is a single-session measurement; N is
modest. Verdict is suggestive-not-decisive at this dispatch budget.
An extended sweep is what `scripts/probe_posterior_touch_correlation.py`
exists to enable — contributors with their own corpora can add to
the M=2 trial-equivalent verdict and either tighten the defer call
or re-open H3.

## What R4 measures and what it does NOT

The lab campaign's R4 series proved that adding `retrieve_hit` events
to the touch state changes the top-K rerank ordering on the corpus by
0% (Jaccard 1.000 across the standard cells). That's why bit 1 stays
unwritten: `retrieve_hit` adds zero observable surface to what
consumers actually see. The decision is robust to formula choice
(R4d/R4e), corpus dwell (R4b), event-mix frequency (R4c), and
multiplicative-vs-additive blend (R5).

R4 does **not** measure rebuilder continuation fidelity. That was H3's
job (R3 — load-bearing); R7c's cross-corpus measurement found the
synthetic R7 signal mostly evaporates at production correlation
levels, so R3 is not scheduled. The substrate shipped here remains
the foundation any future H3 mechanism would build on.

## Inspection

```
$ aelf doctor --hot-path
aelf doctor --hot-path: 2 session(s) with touch state.
session_id                                            rows    max_fire_idx
<session-A>                                             42              81
<session-B>                                             12              17
```

Empty (cold start, no UPS fires yet under this PR):

```
$ aelf doctor --hot-path
aelf doctor --hot-path: belief_touches is empty.
```

Read-only; always exits 0. The future consumer flip will add gate
semantics here.

## Window default

`DEFAULT_TOUCH_WINDOW_K = 50` lives as a module constant in
`src/aelfrice/hot_path.py`. The value is sourced from the lab
campaign's R2c canonical cell. Promotion to a
`meta:retrieval.hot_window_K` knob is an explicit follow-up if the
consumer flip lands and motivates tuning — DESIGN.md v1 locks the
constant per the "non-decisions" section: move it only by
re-measurement, not by config knob.

## File map

| Path | Role |
|---|---|
| `src/aelfrice/hot_path.py` | Pure helpers (`is_hot`), constants (`DEFAULT_TOUCH_WINDOW_K`, `TOUCH_EVENT_KIND_*`). |
| `src/aelfrice/store.py` | Schema DDL, `record_touch`, `read_touch_set_in_window`, `count_touches_for_session`, `list_touch_sessions`. |
| `src/aelfrice/hook.py` | `_record_touches` helper; UPS call site after `_ring_append_ids`. |
| `src/aelfrice/cli.py` | `aelf doctor --hot-path` surface. |
| `tests/test_hot_path_touch_state.py` | Schema + store + helper + hook integration tests (21). |

## Related issues

- [#748](https://github.com/robotrocketscience/aelfrice/issues/748) — R&D campaign tracker (closes once consumer ships and H3 reports).
- [#816](https://github.com/robotrocketscience/aelfrice/issues/816) — this storage substrate.
- [#779](https://github.com/robotrocketscience/aelfrice/issues/779) — `injection_events` sibling.
- [#744](https://github.com/robotrocketscience/aelfrice/issues/744) / [#740](https://github.com/robotrocketscience/aelfrice/issues/740) — JSON injection ring (predecessor; v1 shares its `fire_idx` counter but does NOT migrate ring entries — forward-only).
- [#605](https://github.com/robotrocketscience/aelfrice/issues/605) — locked PHILOSOPHY (determinism, narrow surface).
- [#661](https://github.com/robotrocketscience/aelfrice/issues/661) — locked federation decision.
