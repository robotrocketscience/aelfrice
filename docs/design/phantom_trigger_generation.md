# Trigger-driven phantom generation in normal turns

**Status:** RATIFIED 2026-06-23 (operator). Building. Decision points in
[§7](#7-ratified-decisions-2026-06-23) are resolved; this document is the
build contract.
**Tracking issue:** [#980](https://github.com/robotrocketscience/aelfrice/issues/980)
(recommended-work item 3 — "Spec + build trigger-driven phantom generation for
normal turns").
**Dependencies:** stdlib only. Reuses the existing UserPromptSubmit retrieval
pass, `session_ring` per-session state, and the `wonder_ingest` persist path.
No schema change.
**Risk:** low for the aelfrice side (deterministic, default-off, fail-soft,
additive injection only). The non-deterministic synthesis stays in the host
agent, exactly as `/aelf:wonder` already does.

## 1. Problem

Phantoms (`origin=speculative` beliefs) only appear when the user explicitly
types `/aelf:wonder`. The empirical audit on #980 found **74 speculative
beliefs (0.07%) across 101,383 total, 0 ever promoted, 0 ever GC'd** — the
feature is structurally under-exercised because nothing generates phantoms
during ordinary conversation. The operator's framing: there should be an
**automatic trigger during normal turns**, with `/aelf:wonder` remaining the
heavy explicit path.

## 2. The load-bearing constraint

**aelfrice's Python never calls an LLM or the network.** `/aelf:wonder` does
not synthesize content in aelfrice: the slash-command markdown
(`slash_commands/wonder.md` §process, steps 2–4) instructs the **host agent**
to fan out `Task` subagents that produce research documents; the
aelfrice CLI only runs deterministic gap-analysis (`wonder/dispatch.py` — "no
LLM calls, no randomness, no filesystem writes") and then persists the returned
documents via `wonder_ingest` (`wonder/lifecycle.py:95`). Every per-turn hook
(`hook_manifest.json`, 10 entries) is likewise strictly deterministic.

This boundary is non-negotiable and is what makes the design tractable:
**aelfrice can decide *when* a phantom-generation opportunity exists, but it
cannot perform the generation.** Generation is an LLM act and must run under
the host agent's credentials.

Aligns with the locked PHILOSOPHY decision (#605, ratified 2026-05-10): stay
deterministic, narrow surface; non-deterministic gates live in the consuming
agent, not aelfrice.

## 3. Architecture — detect / flag / synthesize / persist

The mechanism decomposes into four stages, split across the deterministic
aelfrice boundary and the host-agent boundary:

```
  ┌─────────────────────── aelfrice (deterministic) ───────────────────────┐
  │ 1. DETECT   UserPromptSubmit hook evaluates a cheap trigger predicate    │
  │             over the retrieval it ALREADY ran this turn.                 │
  │ 2. FLAG     If predicate fires ∧ opt-in flag on ∧ session budget left ∧  │
  │             gap not a recent dup → append an <aelfrice-phantom-          │
  │             opportunity> note to the injected context. Record the fire   │
  │             in session_ring state. NO generation happens here.           │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                      │ note surfaces in the agent's context
  ┌──────────────────────────────── host agent ─────────────────────────────┐
  │ 3. SYNTHESIZE  The agent SEES the note and MAY act: run the existing      │
  │                /aelf:wonder dispatch (Task subagents) on the gap topic,   │
  │                or surface it to the user. Under host credentials.         │
  │ 4. PERSIST     wonder_ingest writes the result as origin=speculative,     │
  │                (α,β)=(0.3,1.0), RELATES_TO edges, idempotent. Unchanged.  │
  └──────────────────────────────────────────────────────────────────────────┘
```

The only genuinely new aelfrice code is stages 1–2; stages 3–4 reuse the
shipped `/aelf:wonder` path verbatim.

### 3.1 Why a *note*, not auto-dispatch

aelfrice physically cannot dispatch the LLM work (§2). Its only channel to the
agent is the context it injects each turn. So the deterministic side surfaces
an **opportunity note**; the host agent is the actor. This yields two
independent opt-in gates — the flag must be on **and** the agent must choose to
act — which is the conservative posture for a feature that spends tokens.

### 3.2 Precedent: the cadence-checkpoint block

This is not a new injection shape. `hook.py:853–877` already writes a
**default-off, per-turn, fail-soft** `<cadence-checkpoint>` sub-block into the
UserPromptSubmit stdout ahead of the `<aelfrice-memory>` block (#870). The
phantom-opportunity note is the same shape: a small tagged block, gated on a
default-off flag, that never raises and never blocks the turn. We follow that
template exactly.

## 4. Trigger signals

The audit proposed three candidate triggers. **All three ship in v1**
(ratified 2026-06-23). They differ in how much new plumbing each needs; the
note shape, opt-in flag, and per-session bounds are shared across all three.

| Signal | Where it lives today | New plumbing for v1 |
|---|---|---|
| **(a) Retrieval gap** — the prompt seeded **zero** hits | `retrieve()` computes hits but the hook only reads the list. | Surface `len(hits)==0` to the predicate — read-only, value already on hand. |
| **(b) New entity, no edges** — an unrecognised entity with no graph neighbours | Entity extraction exists in the L2.5 retrieval lane (`extract_entities`, `store.lookup_entities`) but is **not** surfaced to the UPS hook. | Extract entities in the UPS hook (reuse `extract_entities`), look each up, fire when an entity resolves to **zero** beliefs / zero edges. |
| **(c) CONTRADICTS edge minted** — a new contradiction appeared since last turn | No edge-creation event exists, and the `edges` table has **no timestamp** (`src,dst,type,weight,anchor_text`). | **Poll + set-diff**: the UPS hook reads the CONTRADICTS pair-set (cheap `SELECT src,dst FROM edges WHERE type='CONTRADICTS'`) and diffs against a per-session snapshot in `session_ring`. No schema migration, no write-path event. Inert unless the #988 semantic-edge substrate (also default-off) is enabled to mint the edges. |

The three are independent predicates ORed together under one flag and one
budget: any firing predicate produces the same `<aelfrice-phantom-opportunity>`
note (with a `reason` attribute naming which signal fired), subject to the
shared per-session budget and dedup (§5).

### 4.1 Gap predicate (signal a)

After the hook's normal retrieval for the prompt, fire when `len(hits) == 0`
(zero-hits-only, §7 decision 3). The hook already has the hit list; the only
change is to test its length and feed the prompt topic into the note.

### 4.2 Entity-novelty predicate (signal b)

Extract entities from the prompt with the existing `extract_entities`
(the same call the L2.5 lane uses). For each, look it up via
`store.lookup_entities`. Fire when an extracted entity resolves to **zero**
matching beliefs (and thus no edges) — a genuinely new entity the store knows
nothing about. The note topic is the novel entity. Dedup is keyed on the
entity string so the same new entity is not re-flagged within a session.

**Named kinds only.** The loose `noun_phrase` entity kind is excluded — it
matches nearly any prompt, would make this signal fire on almost every turn,
and largely duplicates the gap signal. Signal (b) fires only on the *named*
kinds (`identifier`, `file_path`, `url`, `error_code`, `version`, `branch`):
a CamelCase symbol, path, URL, error code, version, or branch the store has
never seen is a strong "new entity" signal; an arbitrary noun phrase is not.

### 4.3 CONTRADICTS-edge predicate (signal c)

The `edges` table carries no timestamp and there is no edge-creation event, so
v1 uses **poll + set-diff** rather than write-path instrumentation. Each UPS
turn the predicate reads the current CONTRADICTS pair-set
(`SELECT src,dst FROM edges WHERE type='CONTRADICTS'`) and diffs it against the
per-session snapshot stored in `session_ring`. Any pair present now but absent
from the snapshot is a "new contradiction" → fire, with the contradicting pair
as the note topic; then the snapshot is updated. This needs no schema migration
and no change to the edge-write path. The CONTRADICTS pair-set is small in
practice (the audit found contradicts pairs are rare), so the poll is cheap.

**Coupling caveat.** CONTRADICTS edges are only minted by the #988 semantic-edge
substrate (`relationship_detector.write_semantic_edges`), which is itself
default-off (`is_auto_relationship_detection_enabled`). Signal (c) therefore
fires only when **both** `phantom_generation` and the #988 substrate are
enabled; on a default install the CONTRADICTS set is empty and (c) is inert.
This is acceptable layering (opt-in stacks on opt-in) but means (c) is the
least-exercised of the three until #988 ships default-on.

## 5. Bounds (determinism + cost)

Per the operator's constraint ("bounded so it does not spam the store") and the
opt-in posture (#606, ADR-0003 dec-4):

- **Default-off opt-in flag.** `should_trigger_phantom_generation(explicit, *, start)`
  resolving `AELFRICE_PHANTOM_GENERATION` env > kwarg > `[phantom_generation] enabled`
  TOML > `False`. Byte-for-byte the `is_bfs_enabled` resolver shape
  (`retrieval.py:2385`). A fresh install is unaffected.
- **Per-session fire budget.** At most `max_fires_per_session` opportunity notes
  per session (default **3**, §7 decision 4). Stored in `session_ring` state,
  which already carries per-session counters (`update_p3_velocity_state`,
  `read_ring_state`); we add a `phantom_gen` counter keyed by `session_id`.
- **Per-signal dedup.** Do not re-flag the same opportunity within a session.
  Dedup key depends on the signal: normalised prompt-topic hash (gap), entity
  string (entity-novelty), sorted belief-id pair (contradicts). Reuses the
  `session_ring` dedup pattern that already suppresses re-injection of the same
  belief IDs.
- **Cheap predicate.** The trigger reads a value retrieval already computed; it
  adds no new query, no LLM, no network. Worst case per turn is one extra
  dict read and a hash.

## 6. Config surface (proposed)

```toml
[phantom_generation]
enabled = false              # master opt-in (default off)
max_fires_per_session = 3    # per-session budget across all three signals
auto_dispatch = false        # passive-surface (default) vs. instruct-agent-to-run
```

Gap detection is zero-hits-only in v1 (no score floor knob yet — §7 decision 3).
Env overrides mirror the house pattern: `AELFRICE_PHANTOM_GENERATION` (master),
others as needed.

## 7. Ratified decisions (2026-06-23)

1. **Trigger scope.** Ship **all three** signals in v1 — gap (a),
   entity-novelty (b), and CONTRADICTS-edge (c) — ORed under one flag and one
   budget. (b) and (c) carry the new plumbing noted in §4.2/§4.3.
2. **Note posture.** **Passive surface** by default: the note states the
   opportunity ("a gap/new-entity/contradiction was detected on `<topic>`;
   consider `/aelf:wonder`") and the agent or user decides. An `auto_dispatch`
   sub-flag (default **off**) can later flip it to instruct the agent to run the
   dispatch automatically.
3. **Gap definition.** **Zero-hits-only** in v1 — fire only when retrieval
   returned nothing. No BM25 score floor yet; add one once calibratable against
   a bench.
4. **Per-session budget.** **3** opportunity notes per session (shared across
   all three signals).

## 8. Alternatives considered

- **aelfrice synthesizes phantoms directly in the hook.** Rejected: violates §2
  (no LLM in aelfrice Python) and the #605 determinism lock.
- **A new always-on background generator.** Rejected: violates the opt-in
  default posture (#606, ADR-0003 dec-4) and the operator's "bounded, no spam"
  constraint.
- **Promote the trigger to the Stop hook instead of UserPromptSubmit.** The Stop
  hook fires at end-of-turn and could batch gaps, but it cannot inject into the
  *next* turn's context as cleanly as the UPS block, and the gap signal is
  computed precisely at retrieval time (UPS). Keep it on UPS; revisit if
  end-of-turn batching proves better.

## 9. Out of scope

- GC auto-run (recommended-work item 2) — separate slice.
- Promotion wiring audit (item, separate).
- Any change to `wonder_ingest` semantics or the `/aelf:wonder` explicit path.

## 10. References

- #980 — phantom lifecycle umbrella + audit findings.
- #605 — PHILOSOPHY: deterministic, narrow surface (locked).
- #606 — sentiment-hook opt-in-default precedent (locked).
- ADR-0003 decision 4 — keep-opt-in default posture (locked).
- `slash_commands/wonder.md`, `wonder/lifecycle.py`, `wonder/dispatch.py` — the
  existing explicit phantom path and the LLM-dispatch boundary.
- `hook.py:853–877` — the `<cadence-checkpoint>` default-off per-turn injection
  precedent.
- `session_ring.py` — per-session counter + dedup state.
- `retrieval.py:2385` (`is_bfs_enabled`) — default-off flag-resolver house style.
