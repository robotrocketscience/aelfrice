# Open-issue priority + dependency DAG

**Snapshot:** 2026-04-29. Sources of truth are the GitHub issues + open
PRs; this doc is a stable index. Refresh whenever an issue closes or
a new design dependency surfaces.

## Reading guide

- **Wave** = ordering layer. All items in the same wave are
  independent and can land in parallel; later waves block on
  earlier ones.
- **Dep** = explicit prerequisite issue (must merge first).
- **Soft-dep** = informational dependency that won't block PRs but
  will cause rework if ignored.
- **Status:** `S` = spec-ready (memo merged or in-flight); `I` =
  implementation work; `R` = research / bench-gated; `B` = bug; `C`
  = closeable (already shipped or duplicate).

## Wave 0 — close out before starting anything new

Stale items. Walk these to zero so they stop showing up in `gh
issue list` triage.

| # | Title | Action |
|---|---|---|
| 223 | feedback_history orphan audit | **Close on PR #283 merge** — fix already in `store.py:514`, test exists. Comment posted. |
| 254 | T1 corroboration trigger gap | **Effectively closed by PR #283** (`UNIQUE(content_hash)` shipped). Operator close needed. |
| 281 | duplicate + off-topic beliefs | **Re-scope:** dedup half = PR #293 (fix-tests blocker). Floor half → #289. Comment-link both. |
| 286 | rebuild redesign parent | **Convert to tracker** — children #287→#291 carry the work. Parent stays open as roll-up. |
| 287 | rebuild eval harness (dup of #288) | **Close as duplicate** — comment posted on 2026-04-29. |
| 280 | hook hardening | **Spec PR #292 merged** — Phase-1 implementation PR #297 in flight. Issue auto-closes on #297 merge. |
| 288 | rebuild eval harness | **Spec PR #294 merged** — implementation issue stays open until layer-1 log code lands. |

## Wave 1 — unblocks the rest of the rebuild redesign

Land in any order; all four are spec memos awaiting ratification +
implementation.

| # | Title | Spec PR | Status | Soft-dep |
|---|---|---|---|---|
| 288 | Eval harness — phase-1 logs | #294 (merged) | S → I | none |
| 289 | Relevance floor + silent contract | #295 (open) | S | #288 (calibration of T values lands later) |
| 290 | Belief retention class + aging | #296 (open) | S | #289 (extends composite score), #288 (audit data), #196 (substrate) |
| 291 | Query understanding | #298 (open) | S | #289 (per-intent floor multipliers) |

**Rule of thumb:** ratify all four memos *first*, then land the four
implementations in numeric order (#288 first because the others
need its log to calibrate). Implementations that touch
`context_rebuilder.py` will conflict — sequence them, don't
parallelise.

**v2.0 substrate cross-cut:** #196 (multi-axis vs single-axis
posterior) is informational soft-dep for #290's `posterior_mean`
term. The retention-class memo's score formula is robust to either
substrate; ratify the substrate before *implementing* #290 so the
formula doesn't get rewritten.

## Wave 2 — phantom-prereqs (T1→T2→T3)

T1 already shipped (`belief_corroborations` table). T2 (PR #256) is
stuck on author rebase + post-#283 merge conflict; T3 hasn't started
because it depends on T2's `session_id` plumbing.

| # | Title | Status | Dep | Action |
|---|---|---|---|---|
| 189 | Tracker | meta | — | stays open until T3 ships |
| 191 | T2 deferred-feedback sweeper | I (PR #256 stale) | #190 (✅), #283 (✅) | author rebase → re-review |
| 192 | T3 session_id propagation | I (not started) | #191 (must merge first) | wait for #256 to clear |

**Soft-dep with Wave 1:** #290's promotion rule (snapshot → fact)
references `corroboration_count` from the same table T1 created.
No conflict, but #190's ledger is the signal source.

## Wave 3 — v2.0 substrate decision tree

These resolve "what shape is the posterior" and propagate
downstream. Order matters here.

| # | Title | Status | Dep | Notes |
|---|---|---|---|---|
| 196 | Multi-axis vs single-axis substrate | S (substrate decision in `docs/design/substrate_decision.md`; cascade landed) | — | **gate for everything below** |
| 151 | Posterior-weighted ranking (Beta-Bernoulli, log-additive) | S (PR #277 spec) | #196 | implementation after #196 final ratify |
| 228 | Wonder-consolidation strategy bake-off (RW/TC/STS) | R (spec-ready, bench-gated) | #196, #229 (consumes promotion trigger) | research lane |
| 229 | Phantom promotion-trigger rule | R (spec-ready, bench-gated) | #196, #190 (✅), #228 (informs trigger calibration) | research lane |
| 197 | Dedup module (v2.0 evaluation) | R (bench-gated) | #196, #283 (✅ — output-stage dedup distinct from store-stage) | evaluation lane |
| 198 | Multi-LLM consensus (multimodel) | R (bench-gated) | #196 | evaluation lane |
| 199 | Enforcement module (directives + audit) | R (bench-gated) | #280 (✅ on hook hardening), #196 | evaluation lane |
| 201 | Semantic contradiction detector | R (bench-gated) | #196, #190 (✅) | evaluation lane |
| 193 | Sentiment-from-prose feedback | S (spec-ready) | #196, privacy review | evaluation lane |

**Rule of thumb:** anything `bench-gated` doesn't ship until #288
harness is producing precision/recall numbers (Wave 1). The R-tag
research can run in parallel against synthetic stores; don't merge
anything bench-gated until the harness has a calibration fixture.

## Wave 4 — v2.x materialization (write-log-as-truth)

Sequential by design. Each step probes a fail-closed condition for
the next.

| # | Title | Status | Dep | Notes |
|---|---|---|---|---|
| 262 | replay_full_equality probe | S (spec in PR #267 docs/design/v2_replay.md) | #205 (✅ ingest_log), #271 (✅ legacy_unknown) | probe lands first |
| 264 | Derivation worker | S | #262 (probe must pass) | beliefs become materialized |
| 265 | View-flip — beliefs/edges as views over ingest_log | S | #264, #262 | terminal step of v2.x materialization |

Don't ship in parallel — #264 changes write-path ownership; #265
changes read-path; landing them out of order leaves the store in a
state where `replay_full_equality` doesn't hold and #262's probe is
useless.

## Wave 5 — long-tail retrieval and research

Lower priority; can pick up between waves whenever an operator has
appetite.

| # | Title | Status | Dep | Notes |
|---|---|---|---|---|
| 154 | Pipeline composition tracker (unified `retrieve()`) | S (v1.7 target) | #289 + #290 + #291 (all memos must land first — they all touch the unified path) | refactor only after #289-#291 implementations merge |
| 153 | uri_baki post-rank adjuster retest | R | #154, #288 (calibration data) | research lane, low-priority |

**Rule of thumb:** #154 is a refactor; doing it before #289–#291
ship guarantees rework. Hold.

## Cross-cutting hazards (write these on the wall)

These come up repeatedly and have caused rework or near-rework
already.

1. **`context_rebuilder.py` is the merge-conflict spot.** Any two
   PRs that touch `_query_for_recent_turns`, `rebuild_v14`, or
   `_format_block` will collide. Wave-1 implementations must
   sequence; do **not** open two of #288 / #289 / #290 / #291 PRs
   simultaneously.
2. **`store.py` schema list is the other merge-conflict spot.**
   Any two PRs that add a column or touch the migration helpers
   (`_maybe_*`) will collide. #283 → #284 is the current
   live conflict; #290 retention-class column adds a third surface.
3. **`UNIQUE(content_hash)` from #283 broke #293's tests.** Pattern
   to avoid: any PR that inserts duplicate `content_hash` rows
   directly via `store.insert_belief` will fail. Use raw `sqlite3`
   or refactor the dedup logic to take a `Belief` list at the
   pack-stage so tests can construct dups without the store.
4. **Calibration data is the bench-gate.** Multiple memos defer
   their *constants* (#289's T, #290's half-lives, #291's intent
   thresholds) to "after #288 logs land." If #288's log
   implementation slips, every Wave-1 calibration follow-up slips
   with it. Treat #288's implementation PR as a critical-path
   blocker, not a phase-1 nicety.
5. **Ratify the substrate (#196) before implementing #290.** The
   retention-aging multiplier consumes `posterior_mean`. The
   formula is robust to either substrate, but the implementation
   is not — a multi-axis posterior would mean computing
   per-aspect multipliers, which doubles test surface.

## Suggested ratification order (operator decision queue)

Each line is a yes/no the operator can stamp without code review:

1. **#296 retention-class memo** — accept the three live values +
   defaults table.
2. **#295 relevance-floor memo** — accept composite score shape +
   per-lane floors.
3. **#298 query-understanding memo** — accept three-bucket recency
   + role weighting + intent classes.
4. **#285 hibernation memo** — accept Candidate-A trigger
   + closed grammar + sweeper placement (after author sanitises
   the tier-leak flagged in 2026-04-29 review).
5. **#277 posterior-ranking memo** — accept log-additive weight
   = 0.5 (after #196 substrate ratification).

Once 1-3 are stamped, Wave-1 implementations unblock in sequence
(#288 first). Once 4 is stamped, #196 substrate work unblocks.
Once 5 is stamped, #151 implementation unblocks.

## What's *not* on this DAG

- Operator-facing UX work (`/aelf:*` slash commands, README, etc.)
  lives outside the rebuild / substrate / phantom-prereqs lanes.
- `aelfrice-lab` private experiments that haven't surfaced as
  public issues. Track those in the lab repo.
- v3 horizon items (multi-operator, federation). Out of scope until
  v2.0 ships.

## Index of in-flight specs and their status

| Spec PR | Issue | Status (2026-04-29) |
|---|---|---|
| #277 | #151 | open, awaiting ratification |
| #285 | #196 hibernation half | open, **leak flagged**, author needs sanitise |
| #292 | #280 | merged |
| #294 | #288 | merged |
| #295 | #289 | open, awaiting ratification |
| #296 | #290 | open, awaiting ratification |
| #297 | #280 phase-1 implementation | open, in review |
| #298 | #291 | open, awaiting ratification |

Implementation PRs:

| Impl PR | Issue | Status |
|---|---|---|
| #256 | #191 | DIRTY, awaiting author rebase |
| #284 | #254 | DIRTY (pytest 3.13), awaiting author |
| #293 | #281 partial | failing tests post-#283; needs test refactor |
| #297 | #280 phase-1 | in review |
