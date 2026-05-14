# Relevance floor + silent-vs-always contract (#289)

## Status

**Spec — proposing for ratification.** Doc-only. The actual floor
*value* T blocks on #288 calibration; this memo proposes the
*shape* of the contract change so implementation can land in v1.x
with a placeholder T tunable from `.aelfrice.toml`.

## Problem

`context_rebuilder.rebuild()` and `rebuild_v14()` always pack to
budget. There is no path that returns "I don't know — say nothing."
#281 caught the consequence: 10 duplicate "Graceful degradation"
beliefs and 4 duplicate "continuation fidelity" beliefs surfaced
for an operator session about PR review and `insert_belief`
content_hash dedup. None were related. The ranker found *something
above zero* and the rebuilder honored its always-pack contract.

The mechanical dedup fix (output-stage content_hash collapse) is
#281 / merged via #283 territory and lands separately. This memo is
about the gating question: **should `rebuild()` ever return empty?**

## Recommendation summary

- **Yes — add a silent path.** When no candidate clears the floor,
  `rebuild_v14()` returns an empty string. Locked (L0) beliefs
  bypass the floor and always inject when present.
- **Floor is composite, not raw BM25.** Floor compares against a
  normalized score that combines BM25 (signal-of-overlap) with
  posterior_mean (signal-of-not-stale). Raw BM25 alone over-rewards
  long content with incidental term-matches.
- **Three-tier contract per source lane.** L0 (locked) always packs
  if matched. Session-scoped (L2) packs if score > minimal floor
  (much weaker than L1 floor — operator opted in by talking about
  it this session). L1 (BM25) packs only above the calibrated floor.
- **No empty-marker tag.** `rebuild()` returning `""` is the
  contract; the consuming hooks already write nothing to stdout
  on empty (verified — `hook.py` `if hits:` guard at line 326),
  so no caller updates required. Adding a `<aelfrice-memory
  empty="true"/>` marker is rejected — surfaces noise without
  helping the model do anything different.
- **Floor value T ships as a config knob with a placeholder
  default; calibration via #288 sets the production value.**

## Detailed proposal

### 1. Per-lane floor contract

The current pack order in `rebuild_v14()` is:

```
1. L0 locked (full, never trimmed).
2. Session-scoped beliefs whose session_id matches.
3. L2.5 + L1 hits from retrieve(), in retrieve()'s native order.
```

Modified contract:

| Lane | Floor behavior | Reason |
|---|---|---|
| L0 locked | No floor. Always packs if `lock_level == LOCK_USER` and the row matches. Locks are user-asserted ground truth — operator wants them surfaced even when query is unrelated. | Operator intent dominates ranking. |
| L2 session-scoped | Soft floor `T_session = 0.10` (placeholder). Session-scoped beliefs come from this turn's session — high prior they're relevant. Reject only on near-zero scores. | Operator-introduced this session; high prior. |
| L1 BM25 / L2.5 entity | Hard floor `T_l1 = 0.40` (placeholder). Most candidates that fall below are off-topic. | Default ranker noise lives here. |

Floors apply to a **composite normalized score**:

```
final_score = bm25_normalized * (0.5 + 0.5 * posterior_mean)
```

where `bm25_normalized = bm25_raw / max(bm25_raw, 1.0)` clamped to
`[0, 1]`. Posterior-mean weighting prevents stale-but-high-BM25
beliefs (the `1bc8ab45a40351d9` "insert_belief does dedup" example
in #281) from dominating purely because they share tokens with the
query.

**Decision asks for the score shape:**

- [ ] **Composite score formula.** Confirm
  `bm25_normalized * (0.5 + 0.5 * posterior_mean)` as the floor's
  comparison value. Reject if posterior should not gate retrieval
  at all (and accept that stale beliefs with high token overlap
  pass the floor).
- [ ] **Three-tier per-lane floors.** Confirm L0 = no floor,
  L2 = soft floor, L1 / L2.5 = hard floor. Reject if a single
  global floor is preferred (and accept that locks will sometimes
  be filtered out).

### 2. Empty contract for `rebuild_v14()`

After applying the per-lane floors:

```python
def rebuild_v14(...) -> str:
    locked = store.list_locked_beliefs()
    session_hits = _retrieve_session_scoped(...)
    l1_hits = retrieve(store, query, ...)

    floored_session = [h for h in session_hits if score(h) >= T_session]
    floored_l1 = [h for h in l1_hits if score(h) >= T_l1]

    all_hits = locked + floored_session + floored_l1
    if not all_hits:
        return ""  # silent path
    return _format_block(recent_turns, all_hits, ...)
```

Empty inputs already returned `""` on the legacy `rebuild()` path
at line 547 / 551 / 828; this memo extends the same contract to the
*all-hits-floored-out* case.

**Caller verification:**

- `src/aelfrice/hook.py:326` — `if hits:` guard before
  `_format_hits`. Already empty-tolerant. The hook calls
  `search_for_prompt` not `rebuild_v14` directly, but the same
  pattern holds: empty in → empty out → no stdout write → no
  block injected.
- `src/aelfrice/hook.py:_pre_compact_rebuild_v14` (precompact path)
  — wraps the rebuild output in the harness's `additionalContext`
  envelope. An empty rebuild needs the envelope dropped, not
  emitted with empty body. Verify by inspection during
  implementation.
- `aelf rebuild` CLI — currently prints whatever `rebuild()`
  returns. Empty-on-empty is already correct.
- `/aelf:rebuild` slash command — same as CLI.

**Decision ask:**

- [ ] **Empty path returns `""`, no marker tag.** Confirm
  `rebuild_v14()` returns the empty string when all hits are
  floored out (and, transitively, when the store has no locks /
  session-scoped / above-floor matches). Reject if a sentinel
  marker (`<aelfrice-memory empty="true"/>`) is preferred — note
  what the model is supposed to do differently when it sees the
  marker.

### 3. `last_retrieved_at` stamping on empty queries

#222 / #266 stamp `beliefs.last_retrieved_at` on hook-driven
retrieval. Question: does an empty-result query bump the stamp?

**Recommendation: no.** `last_retrieved_at` is the "this belief was
returned" signal. A belief that didn't clear the floor was *not*
returned. Stamping on near-misses corrupts the staleness signal
that the hibernation lane (#196) relies on. Beliefs whose only
recent retrieval was a floored-out near-miss should look stale
exactly because they were stale.

Mechanically: stamping happens in `hook_search.py` after
`retrieve()` returns. Move the stamp to *after* the floor is
applied. Beliefs filtered out by the floor never reach the stamp
loop.

**Decision ask:**

- [ ] **No stamp on floored-out hits.** Confirm
  `last_retrieved_at` is bumped only for beliefs that survive the
  floor and get packed. Reject if the stamp should fire on every
  candidate the ranker considered (and accept that hibernation
  signal blurs).

### 4. Floor placeholder defaults and calibration plan

Ship with placeholder defaults driven from a new config block:

```toml
[rebuild_floor]
session = 0.10
l1 = 0.40
# composite = "bm25_normalized * (0.5 + 0.5 * posterior_mean)"
```

The `composite` field is documented but not user-tunable in v1
(formula change is a code change). `session` and `l1` are
operator-tunable.

**Calibration on #288 logs.** Once layer-1 logs are collected for
an operator-week:

1. Bucket all candidates by `final_score`.
2. Operator labels a sample of "should-have-been-packed" vs
   "should-not-have-been-packed" from the layer-1 audit script.
3. Pick `T_l1` at the `final_score` value where false-positive rate
   crosses some target (e.g. 10%).
4. Pick `T_session` at the value where the same FPR target holds
   on session-scoped only.

Calibration produces a number; the number lands in a follow-up PR
with a one-line config change. The contract shape doesn't move.

**Decision ask:**

- [ ] **Ship with placeholder defaults; calibrate post-#288.**
  Confirm v1.x lands with `T_session = 0.10`, `T_l1 = 0.40` as
  placeholders that operators can tune via `.aelfrice.toml`.
  Reject if implementation should block on calibration data
  (and accept the longer wait).

### Out of scope

- **Output-stage content_hash dedup.** Mechanical, already in
  flight (#281 / #283 family).
- **Reranking with richer signal.** Cross-encoder, learned
  reranker, etc. — separate question, can stack on top of the
  floor without changing the floor's contract.
- **Calibration of T values.** Lives in a follow-up PR after
  #288 logs are in.
- **The composite formula's coefficients.** The 0.5/0.5 split is
  a starting point; tuning the relative weight of BM25 vs
  posterior is a v2.x concern that wants the same #288 logs.
- **Multi-axis posterior** (#196). The posterior_mean used here is
  the scalar Beta-Bernoulli posterior; if #196 ratifies multi-axis,
  the formula updates accordingly without changing the floor's
  contract shape.

## Decision asks (consolidated)

- [ ] **Composite score formula** — `bm25_normalized * (0.5 + 0.5 *
  posterior_mean)` as the floor comparison value.
- [ ] **Three-tier per-lane floors** — L0 = no floor, L2 session
  = soft, L1 / L2.5 = hard.
- [ ] **Empty path returns `""`** with no sentinel marker tag.
- [ ] **No `last_retrieved_at` stamp on floored-out hits.**
- [ ] **Ship with placeholder T defaults**; calibrate from #288
  logs in a follow-up PR.

## Implementation tracker (post-ratification)

Once ratified, ~one PR.

1. **`context_rebuilder.rebuild_v14`** — composite score helper,
   per-lane floor application, empty-path return.
   `[rebuild_floor]` config loader (mirrors `[implicit_feedback]`
   pattern from `deferred_feedback.py` if it lands first; otherwise
   mirrors `[hook_audit]` from #280). Tests covering: L0 always
   packs, L2 soft floor, L1 hard floor, all-floored-out → `""`,
   composite formula determinism, config override.
2. **`hook.py` precompact-envelope drop on empty.** Small. Tests
   covering empty-rebuild → no `additionalContext` envelope written.
3. **`hook_search.py` stamp move** — `last_retrieved_at` writes only
   for survivors. Tests covering: floored-out belief → no stamp;
   above-floor belief → stamp fires.

~400 lines net incl. tests.

## Provenance

- Parent: #286.
- Blocking: #288 (eval harness — calibration of T values).
- Symptom evidence: #281.
- Adjacent: #196 (substrate decision — informs posterior shape),
  #197 (dedup module — distinct from output-stage dedup),
  #222 / #266 (`last_retrieved_at` stamp ownership).
- Code touchpoints:
  - `src/aelfrice/context_rebuilder.py` `rebuild_v14` (line 240).
  - `src/aelfrice/hook.py` `_pre_compact_rebuild_v14` (precompact
    envelope path).
  - `src/aelfrice/hook_search.py` (stamp call site).
