# Rebuild eval harness (#288 — phase-1 instrumentation for #286)

## Status

**Spec — proposing for ratification.** Doc-only. Ratification unblocks
phase-1 work under the #286 redesign tree (#289 floor, #290 typing,
#291 query understanding all block on this).

## Why this comes first

#286 names the failing assumption ("rebuilder must always inject
something — top-K-by-score regardless of relevance") but every fix
candidate — relevance floor, belief typing, query rewriting,
LLM-vs-deterministic — is unfalsifiable without per-rebuild
ground truth. The v1.4 continuation-fidelity harness measures
*downstream behavior* (does the agent continue similarly with vs
without the rebuild block?), not *retrieval precision* (did the
right beliefs come back?). #281 surfaced three failure modes in one
session — duplicates, off-topic results, stale-as-authoritative —
none of which the existing harness flags.

This memo proposes the two-layer harness: cheap diagnostic log first,
heavier fixed-corpus precision metric second. Both before any
ranker change.

## Recommendation summary

- **Layer 1 — per-rebuild diagnostic log.** New
  `<git-common-dir>/aelfrice/rebuild_logs/<session_id>.jsonl`. One
  record per `rebuild()` invocation. Captures input (recent-turns
  hash + extracted query), candidate set with all scores,
  pack/drop decision per candidate with reason. Default-on,
  size-bounded with single-slot rotation, opt-out env var. Lands in
  v1.x.
- **Layer 2 — fixed-corpus precision harness.** Captured corpus of
  ~75 (query, expected-belief-set) pairs across four operator
  categories (debugging, code review, research, design). Lab-only
  storage; public-repo entry point (`scripts/eval_rebuild.py`)
  reads the corpus from a configured path. Computes
  precision@K, recall@K, MRR. Lands in v2.0 after layer 1 has
  collected a week of operator data to inform corpus design.
- **Belief-correctness eval is explicitly out of scope.** A belief
  can be retrieved correctly and still be factually wrong (#281's
  `1bc8ab45a40351d9` insert_belief-dedup example). Correctness lives
  in feedback / locking / contradiction lanes.

## Detailed proposal

### Layer 1: per-rebuild diagnostic log

Path: `<git-common-dir>/aelfrice/rebuild_logs/<session_id>.jsonl`.

Per-session file rather than a single global JSONL because rebuild
fires constantly and a global file would interleave logs from many
sessions; a session-scoped file makes "what did the rebuilder see in
*this* session" answerable with a single `cat`.

One JSON object per `rebuild()` invocation that produces a non-empty
candidate set:

```json
{
  "ts": "2026-04-29T03:14:15Z",
  "session_id": "<from harness payload, optional>",
  "input": {
    "recent_turns_hash": "<sha256 of concatenated recent-turn
      content>",
    "n_recent_turns": 6,
    "extracted_query": "<the query passed to retrieve()>",
    "extracted_entities": ["<entity_1>", "<entity_2>"],
    "extracted_intent": "<intent label or null>"
  },
  "candidates": [
    {
      "belief_id": "abc123",
      "rank": 1,
      "scores": {
        "bm25": 8.42,
        "posterior_mean": 0.71,
        "reranker": 0.83,
        "final": 9.94
      },
      "lock_level": "user",
      "decision": "packed",
      "reason": null
    },
    {
      "belief_id": "def456",
      "rank": 2,
      "scores": { … },
      "lock_level": "none",
      "decision": "dropped",
      "reason": "below_floor:0.40"
    },
    {
      "belief_id": "abc123_dupe",
      "rank": 3,
      "scores": { … },
      "lock_level": "none",
      "decision": "dropped",
      "reason": "content_hash_collision_with:abc123"
    }
  ],
  "pack_summary": {
    "n_candidates": 12,
    "n_packed": 5,
    "n_dropped_by_floor": 4,
    "n_dropped_by_dedup": 2,
    "n_dropped_by_budget": 1,
    "total_chars_packed": 1342
  }
}
```

**Decision asks for the schema:**

- [ ] **Per-session vs global file.** Confirm per-session file with
  filename `<session_id>.jsonl`. Reject if a single global JSONL is
  preferred (and accept the 'grep across many sessions' tradeoff).
- [ ] **Recent-turns content vs hash.** Schema above stores a hash;
  the *content* is recoverable from the existing transcript log.
  Storing content here would double storage and add a privacy review
  surface (recent turns may contain user-sensitive prose). Confirm
  hash-only.

**Rotation and size limits.** One file per session; sessions are
short-lived enough that natural file churn handles bound-keeping in
practice. As a safety net, cap each session-file at 5 MB; on reaching
the cap, append a final `{"truncated": true, …}` row and stop
writing. No rotation slots — once a session-file is full, it's full.
Garbage collection: a follow-up cron-style cleanup (out of scope
here) can reap files older than 30 days.

**Opt-out.** `AELFRICE_REBUILD_LOG=0` env var or
`[rebuild_log] enabled = false` in `.aelfrice.toml`. Default-on. Same
fail-soft contract as `_write_telemetry` in `hook.py:358` —
any I/O error is logged to stderr and never breaks the rebuild.

**Where the write hook lives.** Two call sites:

1. Inside `rebuild_v14()` in `context_rebuilder.py`, immediately
   after the candidate list is finalized but before the rendered
   block is returned. Catches the PreCompact path and any future
   direct `rebuild_v14` caller.
2. Inside `user_prompt_submit()` in `hook.py`, after content-hash
   dedup, via the `record_user_prompt_submit_log` helper in
   `context_rebuilder.py`. Catches the high-frequency UPS retrieval
   path, which calls `search_for_prompt` directly and never reaches
   `rebuild_v14`.

Both call sites share the schema, the `_append_rebuild_log_record`
writer, the size cap, and the env / TOML opt-out. UPS records carry
a synthetic single-turn `RecentTurn` derived from the prompt; the
on-disk record shape is identical to the PreCompact one.

The original spec assumed all rebuild call sites went through
`rebuild_v14`, so phase-1a wired only that path. The UPS path
bypasses `rebuild_v14` entirely (`search_for_prompt` →
`retrieve()` returns the final hit list, no rebuild block built),
so a phase-1a-only ship produced empty logs under normal session
load — phase-1b was unreachable until UPS was wired.

### Layer 2: fixed-corpus precision harness

A captured corpus of (query, expected-belief-set) pairs. Each entry:

```json
{
  "id": "debug_001",
  "category": "debugging",
  "query": "store.py FK violation on belief insert",
  "context_turns": [
    "<turn 1 verbatim>",
    "<turn 2 verbatim>"
  ],
  "expected_belief_ids": ["a1b2c3", "d4e5f6"],
  "expected_belief_contents_hash": "<sha256 of sorted contents>",
  "labeler": "operator",
  "labeled_at": "2026-05-15T00:00:00Z",
  "notes": "FK constraint between beliefs and feedback_history; …"
}
```

**Corpus origin.** Hand-curated from operator sessions (the layer-1
log is the source feed). Synthetic queries are tempting but
underrepresent the "I'm in the middle of a multi-turn debugging
context" surface that operators actually hit — that's the precision
target, not the easy case. Mixed corpora (hand + synthetic) are
fine in principle; the hand-curated portion has to be
load-bearing for any "improved precision" claim.

**Labeling.** Operator post-hoc, with a second-pass LLM judge as a
*review* step (flag entries where judge disagrees with operator) but
not as the primary labeler. Reason: the LLM judge has its own bias
profile (will favor beliefs with content overlap to query, which
defeats the point of measuring whether the *ranker* picks the right
ones). Operator-as-primary-labeler is slow but defensible.

**Categories.** Four to start, each with ~15-20 queries:

| Category | Example query shape | Expected-set characteristic |
|---|---|---|
| debugging | "FK violation on belief insert" | Specific, recent, may include locked beliefs |
| code review | "ranking changes since v1.5" | Multi-belief; chronological |
| research | "phantom-prereqs T2 grace window" | Single dense thread |
| design | "should hibernation use multi-axis substrate" | Locked decisions; spec memos |

**Storage.** Lab-only (`~/projects/aelfrice-lab/eval/rebuild_corpus/`).
The corpus contains operator-session context, which falls under the
private side of the two-repo boundary. The public-repo entry point
(`scripts/eval_rebuild.py`) reads the corpus path from
`AELFRICE_EVAL_CORPUS_PATH` env var or a `--corpus` flag, with a
clear error when unset. Public CI does not run the harness; it's
operator-driven.

**Cadence.** Operator-run on every retrieval-touching PR before
merge. Not a CI gate (CI doesn't have access to the lab corpus);
the operator posts the precision/recall numbers as a PR comment.

**Decision asks for the harness:**

- [ ] **Operator-as-primary-labeler.** Confirm hand-labeling is the
  primary path with LLM as review. Reject if pure LLM labeling is
  preferred (and accept the bias-overlap tradeoff).
- [ ] **Lab-only corpus storage.** Confirm corpus lives in the
  private lab repo with a public-repo entry point that reads from a
  configured path. Reject if a synthetic public corpus is preferred
  (and accept the underrepresentation of multi-turn context).
- [ ] **Operator-run, not CI-gated.** Confirm precision is reported
  as a PR comment, not as a required status check. Reject if a
  CI-runnable synthetic harness is wanted (separate scope).

### What lands when

| Phase | Layer | Artifact | Milestone |
|---|---|---|---|
| 1a | 1 | `rebuild_log` write path + tests + opt-out config | v1.x patch |
| 1b | 1 | Operator-week of captured logs | v1.x operator workflow |
| 1c | 1 | Audit script (`scripts/audit_rebuild_log.py`) — distribution of dropped reasons, rank-of-packed by score percentile | v1.x patch |
| 2a | 2 | Lab corpus (~75 entries, 4 categories) | v2.0 |
| 2b | 2 | `scripts/eval_rebuild.py` (precision@K, recall@K, MRR over corpus) | v2.0 |
| 2c | 2 | LLM-judge review pass + disagreement-flagging | v2.0 |

Phase 1 is fully self-contained — it ships, it runs, it produces
data, even if phase 2 never lands. Phase 2 depends on phase-1 logs
to inform which queries are corpus-worthy.

### Out of scope

- **The actual ranker / floor / typing fixes.** Those are #289 /
  #290 / #291 — they consume this harness, they aren't this harness.
- **Continuation-fidelity replacement.** The v1.4 harness keeps
  measuring downstream behavior; rebuild precision is the missing
  *complementary* layer, not a substitute.
- **Belief-correctness eval.** Out of scope per #288 issue body and
  reaffirmed here. The phantom-prereqs / posterior / corroboration
  lanes own correctness.
- **Per-rebuild ranker telemetry into a dashboard / Grafana / etc.**
  JSONL on disk is enough; visualization is operator-driven via the
  audit script.
- **Cross-session corpus sharing.** Corpus is operator-personal in
  v2.0. Sharing across operators is a v2.x concern.

## Decision asks (consolidated)

- [ ] **Layer 1 schema** — per-session file, hash-only recent
  turns, single 5 MB cap with no rotation slot.
- [ ] **Layer 1 default-on** with env + TOML opt-out.
- [ ] **Layer 2 labeling** — operator-primary, LLM-review.
- [ ] **Layer 2 storage** — lab-only with public-repo entry point.
- [ ] **Layer 2 cadence** — operator-run, PR-comment reporting,
  not CI-gated.
- [ ] **Phase ordering** — phase 1 lands first standalone; phase 2
  blocks on a week of phase-1 data.

## Implementation tracker (post-ratification)

Phase 1: roughly one PR.

1. **`context_rebuilder` log hook** + write helper +
   `[rebuild_log]` config block + tests covering schema validity,
   size cap, opt-out. ~350 lines net incl. tests.
2. (Optional companion PR) **`scripts/audit_rebuild_log.py`** —
   read a session JSONL, summarise drop-reasons / score percentiles.
   ~80 lines.

Phase 2: separate work after phase-1 data is in hand. Spec memo for
the corpus shape gets its own issue at that point.

## Provenance

- Parent: #286 (rebuild redesign scoping).
- Sibling: #287 closed as duplicate of #288.
- Adjacent: v1.4 continuation-fidelity harness
  (`docs/v1_4_continuation_fidelity.md` if/when written).
- Telemetry pattern reference: `hook.py:_write_telemetry` and
  `_telemetry_path_for_db` (`src/aelfrice/hook.py:194-358`).
