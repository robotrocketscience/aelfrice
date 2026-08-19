# Architecture

This document describes how aelfrice fits together. The document maps directly to the source code under `src/aelfrice/`.

## Principles

1. **Determinism end to end.** Every retrieval result is bit-identical for the same write log and the same code. Every result traces to named beliefs and named rules. See [PHILOSOPHY § Determinism is the property](PHILOSOPHY.md#determinism-is-the-property). **The `edges` are the standing exception.** The code as shipped writes the edges outside the log. Two stores with identical `ingest_log` content can therefore hold different graphs. The L3 edge-walk lane can then return different results ([#1283](https://github.com/robotrocketscience/aelfrice/issues/1283)).
2. **SQLite plus a small numeric stack.** The hot path has no vector DB, no embeddings and no large language model (LLM). The required dependencies beyond the standard library are `numpy`, `scipy` and `snowballstemmer`. `numpy` and `scipy` (added v1.5.0, #148) serve the sparse matrix-vector (matvec) lane for Best Matching 25 (BM25). The lane for holographic reduced representations (HRR) and the spectral-graph lane now also use them. `snowballstemmer` (added v1.7.0, #154) does the Porter stemming. Three extras are optional:
    - `[onboard-llm]` — the software development kit (SDK) for the direct-API onboard classifier, for `aelf onboard --llm-classify`.
    - `[archive]` — cryptography, for `aelf uninstall --archive`.
    - `[benchmarks]` — the adapters on the development side.
3. **Confidence is Bayesian.** Confidence is `α / (α + β)`. Every update has a closed-form rule. At v1.3.0+ the code combines the posterior log-additively with BM25 on the L1 tier. See [LIMITATIONS](../user/LIMITATIONS.md) for what the partial ranking covers and does not cover.
4. **`apply_feedback` is the central endpoint.** It is the one writer of `(α, β)`. Each successful update writes one audit row.
5. **Locks are user-asserted ground truth.** A user-locked belief short-circuits decay. Decay is a mechanism that does not run at present. See the note on posterior decay below. To correct a lock, the user overwrites it with `aelf lock`. This correction is an explicit user act (PHILOSOPHY [#605](https://github.com/robotrocketscience/aelfrice/issues/605)). Issue [#814](https://github.com/robotrocketscience/aelfrice/issues/814) removed the automatic demotion that a contradiction drove.

### Enrichment-step boundary

The determinism contract applies to retrieval. Every read is reproducible from the inputs. Some operations on the write side include non-deterministic steps. Two examples are the LLM-driven sentence classification on the polymorphic onboard path, and the future capabilities on the research line. The boundary is explicit:

- The store records the inputs to enrichment. The inputs are the sentence and the source. The store also
  records the *outputs* of the classifier (`raw_meta["route_overrides"]` — belief type, origin, alpha, beta).
  **The store does not record the model id, the model version or the hash of the prompt template.** These
  three values live only on the transient router object and in the telemetry on stdout. The store therefore
  cannot answer the question "which model produced this belief's type and prior?". This limit bounds the
  exception less than the sentence above suggests. The exception localises the non-determinism to a recorded
  step. The exception does not make the step reproducible. The work to persist the three values into
  `raw_meta` is tracked separately.
- The store keeps the outputs as deterministic content with provenance. The outputs are the belief type, the prior and the derived edges. The *derived edges* carry provenance only in part. Until [#1354](https://github.com/robotrocketscience/aelfrice/issues/1354), all six `derive()` return paths emitted `edges=[]`. `ingest_log.derived_edge_ids` was therefore NULL on every row. No edge had a log row that pointed at its origin ([#1283](https://github.com/robotrocketscience/aelfrice/issues/1283)). `derive()` now emits `DERIVED_FROM` for the case inside one turn. The column populates forward-only, for at most 1.93% of the live edge set. The code still writes `TEMPORAL_NEXT` (88.3%) and the detector edges outside the log.
- All the retrieval mathematics and all the feedback mathematics after the enriched store are deterministic.

The contract is a *deterministic substrate plus a bounded, audited enrichment layer*. The contract is not "no model ever touches the data."

## Modules

The imports are *intended* to be one-directional. A module lower in the table imports from a module higher in
the table. This ordering is an aspiration, not an enforced invariant. A deferred import inside a function
breaks one known inversion. `classification.py` imports from `scanner`. The comment at the import site
names that import as circular. Duplication avoids a second inversion, rather than deferral. `store.py`
reimplements the constituent-key hash of `wonder.lifecycle` inline, and the comment there cites the cycle
(`store` ← `wonder.lifecycle`) as one of two reasons. A count of the deferred imports alone therefore
understates how often the code works around the ordering. The converse is also true: not every deferred
import is an inversion. `store.py` defers `federation` only for the cost of the import. The comment at that
site records that `federation` is a leaf module and imports nothing from `store`. That deferral keeps
`subprocess` and `json` out of every consumer of the store. The table is also a curated subset. It holds 32
modules against the 128 `.py` files under `src/aelfrice/`. The table is not an exhaustive map.

| Module | Responsibility |
|---|---|
| `models.py` | Holds the `Belief`, `Edge`, `FeedbackEvent` and `OnboardSession` dataclasses. Holds the constants for type, lock and origin. Does no input and no output. |
| `scoring.py` | Holds `posterior_mean`, `partial_bayesian_score` and the gamma and zeta posterior rerank scorers. Retrieval imports these functions. This module also defines `decay`, `type_half_life` and `TYPE_HALF_LIFE_SECONDS` (a lock-floor short-circuit, with the Jeffreys `(0.5, 0.5)` target). **No module under `src/` calls those three names.** The posterior decay is designed but not wired ([#1218](https://github.com/robotrocketscience/aelfrice/issues/1218)). The disposition is tracked under [#1162](https://github.com/robotrocketscience/aelfrice/issues/1162). |
| `store.py` | SQLite with write-ahead logging (WAL), full-text search version 5 (FTS5) and the create, read, update and delete (CRUD) operations. `propagate_valence` runs a breadth-first search (BFS) with attenuation by broker confidence. `apply_feedback` fires it on every direct feedback event. To disable it, set `AELFRICE_VALENCE_PROPAGATION=0`. |
| `retrieval.py` | `retrieve(store, query, token_budget=2400)`. The lanes are L0, L2.5, L1 and L3. L0 holds the locked beliefs. L2.5 is the entity index (v1.3+). L1 is the FTS5 lane with BM25 or BM25F, and BM25F is default-on since v1.7.0. L1 also applies Bayesian log-additive reranking (v1.3+). L3 is the BFS multi-hop lane (v1.3+, default-off) over the seed set of L0, L2.5 and L1. L0 is never trimmed. |
| `feedback.py` | `apply_feedback(store, belief_id, valence, source)`. This is the only path for a Bayesian update. It writes `feedback_history`. |
| `contradiction.py` | `resolve_contradiction` picks a winner by precedence. It inserts `SUPERSEDES` and writes an audit row. It backs `aelf resolve`. |
| `correction.py` | A heuristic detector for corrections. It uses no LLM. |
| `classification.py` | Type priors with a regex fallback. The polymorphic state machine for the onboard path. |
| `noise_filter.py` | `is_noise(text, config)` filters out markdown headings, checklist blocks, three-word fragments and license boilerplate. You tune it in `.aelfrice.toml`. See [CONFIG](../user/CONFIG.md). |
| `scanner.py` | `scan_repo` runs three extractors: the filesystem, the git log and the Python abstract syntax tree (AST). It is idempotent on `content_hash`. |
| `health.py` | The v1.0 regime classifier (`supersede`, `ignore`, `mixed`, `insufficient_data`). `aelf regime` surfaces it. |
| `auditor.py` | A structural auditor. It checks orphan threads, FTS5 synchronisation, locked contradictions and corpus volume. It backs `aelf health`. It is read-only. |
| `migrate.py` | A one-shot port from the legacy global DB into the per-project DB. It reads the source with SQLite `mode=ro`. It backs `aelf migrate`. |
| `doctor.py` | A linter for the settings. It walks every `command` in `settings.json` and verifies that the command resolves. It handles a `bash <script>` wrapper as a special case. It backs `aelf doctor`. |
| `lifecycle.py` | The update notifier (a background check against PyPI), the uninstall code and the archive encryption. |
| `transcript_logger.py` | The hook entry point for transcript capture (v1.2+). It writes one JSONL line per turn under `<git-common-dir>/aelfrice/transcripts/`. |
| `hook_commit_ingest.py` | The `PostToolUse:Bash` hook. It ingests a commit message after `git commit`. |
| `hook_search.py` | The retrieval helper for UserPromptSubmit. It records every hit as a `feedback_history` row tagged `source='hook'`. The row is audit-only since #1086 (v4.0). The row logs the exposure and the recurrence, but `record_retrieval` passes `update_posterior=False` by default. A surfacing therefore does **not** move α or β. Set `AELFRICE_EXPOSURE_UPDATES_POSTERIOR=1` to restore the legacy behaviour that promotes on exposure. |
| `triple_extractor.py` | Extraction of `(subject, relation, object)` triples with regex only, over six relation families. The commit-ingest lane and the transcript-ingest lane use it. |
| `context_rebuilder.py` | The rebuilder that runs after compaction. It surfaces the aelfrice retrieval again after the harness writes its summary. The block is delivered on `SessionStart(source="compact")` (#1031). |
| `benchmark.py` | A deterministic synthetic harness with 16 beliefs and 16 queries. The `BenchmarkReport` is frozen. |
| `cli.py` | The CLI with many subcommands, built on argparse. The entry point is `aelf`. `aelf --help` shows the everyday surface. `aelf --help --advanced` shows the full surface, which adds the diagnostic verbs, the hook verbs and the lifecycle verbs. |
| `federation.py` | (v3.0+) Read-only federation with a peer DB. `load_peer_deps()` parses `knowledge_deps.json`. `open_peer_connection(path)` opens a peer SQLite database in `mode=ro`. It honours the WAL of the peer, and it falls back to `immutable=1` for read-only media. `ForeignBeliefError` rejects a mutation against a foreign belief id at the API surface. See [LIMITATIONS § Sharing, sync, or federation](../user/LIMITATIONS.md). |
| `clamp_ghosts.py` | (v3.0+) The repair tool `clamp_ghost_alphas(store, target_alpha, dry_run)`. It clamps α on the belief rows that have inflated posteriors with no support in the audit trail. The tool targets only the artifacts from before the migration. The tool writes a negative-valence audit row inside the same transaction, so you can reverse the repair. It backs `aelf clamp-ghosts` (hidden). |
| `reason.py` | (v2.0+, expanded v3.0) Reasoning by graph walk over the graph of belief edges. v3.0 (#645, #658) adds the Verdict and ImpasseKind classifiers. It adds the `ConsequencePath` deriver, which forks on CONTRADICTS. It adds `dispatch_policy()`, which maps an impasse to the Verifier role, the Gap-filler role or the Fork-resolver role. It adds `suggested_updates()`, which derives the close-the-loop feedback rows. It backs `aelf reason` and `/aelf:reason`. |
| `wonder/` | (v2.0+, expanded v3.0) The wonder lifecycle. It holds the gap analysis (`dispatch.py`) and the generation of the research axes. It holds the phantom ingest and the garbage collection (`wonder_ingest`, `wonder_gc`). It holds the integration of subagents at the Skill layer (`skill_integration.py`, per #552). It holds the structured `WonderResult` dataclass (#656). |
| `sentiment_feedback.py` | (v2.0 module, v3.0 hook wired) A regex sentiment detector. v3.0 (#606) wires it into `UserPromptSubmit`, behind `[feedback] sentiment_from_prose = true`. |
| `auto_install.py` | (v3.0+, #623) The merger for the version-stamped manifest. The first `aelf <cmd>` after a wheel upgrade merges the new default-on hooks from `data/hook_manifest.json` into `~/.claude/settings.json`. The merge takes an `fcntl` lock. It honors `~/.aelfrice/opt-out-hooks.json`. |
| `working_state.py` | (v3.0+, #587) The projector that writes `<working-state>` after compaction. It reports the current branch, a bounded `git status`, the last HEAD log entries, the last K user prompts and the commits of the session. Each git invocation has a 1.5s timeout, and it returns empty on that timeout. |
| `setup.py` | An idempotent install and uninstall of all the hooks and of the statusline. It writes atomically with a tempfile and `os.replace`. |
| `hook.py` | `aelfrice.hook:main` — process Claude Code spawns on each prompt. Reads stdin, calls `retrieve()`, emits `<aelfrice-memory>` on stdout. Non-blocking. Entry: `aelf-hook`. |
| `slash_commands/` | One markdown file for each CLI subcommand that `/aelf:*` surfaces. |

## Data model

**Belief** — `id, content, content_hash, alpha, beta, type, lock_level, locked_at, origin, session_id, created_at, last_retrieved_at, corroboration_count, hibernation_score, activation_condition, retention_class, valid_to, scope, project_context` (v3.2+, #858)`, last_confirmed_at` (v3.5+, #936)`, lock_tier` (v3.7+, #1016).

- `type ∈ {factual, correction, preference, requirement, speculative}`. The v3.0 wonder lifecycle added `speculative` for phantom beliefs (#548).
- `retention_class ∈ {fact, snapshot, transient, unknown}`. This field drives the type-aware compression (#769).
- `lock_level ∈ {none, user}`
- `lock_tier ∈ {frozen, reference}` (v3.7+, #1016-B). This field is orthogonal to `lock_level`. It has a meaning only when `lock_level = user`. A `frozen` lock is always injected verbatim, and `frozen` is the default for every lock. A `reference` lock is bounded: it is injected as a one-line manifest entry. You read its full text on demand with `aelf locked` or `aelf search`. To demote a large lock, run `aelf lock <text> --reference`.
- `origin ∈ {user_stated, user_corrected, user_validated, user_transcript, agent_inferred, agent_remembered, document_recent, speculative, unknown}` (v1.2+). The v2.1 transcript-ingest lane added `user_transcript`. The v2.0 wonder substrate added `speculative` for phantom beliefs, and `wonder/lifecycle.py` now writes it.
- `scope ∈ {project, global, shared:<name>}` (v3.0+, #688). `project` is the default, and it stays local. `global` is surfaced to any peer DB that declares this DB in its `knowledge_deps.json`. `shared:<name>` is surfaced only to the peers that also list `shared:<name>` as a dep.

**Edge** — `src, dst, type, weight, anchor_text`. `EDGE_VALENCE` holds ten edge types:

| Type | Valence | |
|---|---|---|
| `SUPPORTS` | +1.0 | full positive |
| `IMPLEMENTS` | +0.65 | code-implements-spec link |
| `TESTS` | +0.55 | test-covers link |
| `CITES` | +0.5 | half positive |
| `DERIVED_FROM` | +0.5 | half positive (turn-to-turn provenance) |
| `RELATES_TO` | +0.3 | weak positive |
| `TEMPORAL_NEXT` | +0.2 | session-time successor |
| `SUPERSEDES` | 0.0 | structural; no propagation |
| `RESOLVES` | 0.0 | structural; closes a `CONTRADICTS` thread |
| `CONTRADICTS` | -0.5 | half negative |

A separate `POTENTIALLY_STALE` edge type exists as a producer-only signal from `aelf doctor` (#387). It is deliberately not in `EDGE_TYPES`, so it takes no part in valence propagation. The research line carried 17 edge types. The additional speculative and causal markers are `SPECULATES`, `DEPENDS_ON` and `HIBERNATED`. The additional structural extractors are `CALLS`, `CO_CHANGED`, `CONTAINS` and `COMMIT_TOUCHES`. Both groups stay parked until the extractors that produce them ship. The current set of ten types covers the v2.0 wonder lifecycle (`RESOLVES`, `SUPERSEDES`, `CONTRADICTS`) and the v1.x link between code and test (`IMPLEMENTS`, `TESTS`). See [ROADMAP § Recovery inventory](ROADMAP.md#recovery-inventory) for the deferred set.

**Core SQLite tables include:**

- `beliefs` (with the `scope` column since v3.0)
- `beliefs_fts` (virtual, porter unicode61)
- `edges`, with the primary key `(src, dst, type)`
- `feedback_history`
- `sessions`
- `onboard_sessions`
- `belief_corroborations` (sibling table, v1.5.1+)
- `ingest_log` (append-only, v1.6+)
- `belief_versions` and `edge_versions` (per-scope version vectors, v1.5+)
- `belief_entities` (the L2.5 entity index, v1.3+)
- `deferred_feedback_queue` (v1.6+, #191)
- `belief_documents` (the wonder research documents, v3.0)
- `injection_events` (the relevance signal, v3.x)
- `belief_touches` (the ring on the hot path, v3.x #748)
- `schema_meta`

The `scope` column has an `idx_beliefs_scope` index. The migration runner adds the column and the index idempotently.

## Bayesian update

`apply_feedback(store, belief_id, valence, source)`:

1. Load the belief. Reject a zero valence. Reject an empty source.
2. **Positive valence:** `α += valence`.
3. **Negative valence:** `β += |valence|`.
4. Persist the change atomically.
5. Append a `FeedbackEvent` row. This step always runs.

## Retrieval

L0 holds the locked beliefs. L0 is the **pool that is always injected**: every lock ships on every retrieval, with no scoring and no top-K cut. The lock count is the operator's control over the budget for the baseline context. The `frozen`-tier locks (the default) ship **in full**. The `reference`-tier locks (v3.7+, #1016-B) ship as a **one-line manifest entry**, and the budget counts them at that size. A large lock set therefore stays bounded. Demote a large lock with `aelf lock <text> --reference`. Read its full text on demand with `aelf locked` or `aelf search`. Only the pool that is not locked (L1, L2.5 and L3) gets relevance ranking and budget trim. When the locks alone approach the budget, a reserved relevance floor (#1015) keeps a part of the budget for the query-relevant results. In that condition, `aelf doctor` also gives a warning (#1016-D).

```
L0: store.list_locked()              always loaded; never trimmed
        ↓
L2.5: entity-index lookup (v1.3+)    NER-extracted entities → exact + stem match;
        ↓                             default-on; disable via [retrieval] entity_index_enabled = false
L1: FTS5 BM25 / BM25F                limit l1_limit, query escaped;
        ↓                             v1.3+: score = log(bm25) + 0.5*log(posterior_mean)
                                      v1.7+: BM25F anchor-augmented sparse matvec, default-on (#148/#154)
Temporal spine (#1064)               TEMPORAL_NEXT chain traversal from the top-5 L1 seeds,
        ↓                             appended after L1; default-ON since the #1107 Phase-2
                                      cutover, opt out via [retrieval] use_temporal_spine = false
L3: BFS multi-hop expansion (v1.3+)  edge-weighted graph walk from L0+L2.5+L1 seed set;
        ↓                             default-OFF; enable via [retrieval] bfs_enabled = true
Dedupe L1+L2.5+L3 against L0 ids
        ↓
Trim from tail until sum(estimated_tokens) ≤ token_budget
```

Two **rerank modifiers** refine the ranked tiers (L1 and L2.5). They add no lane. Since v4.0 the production `retrieve()` hook path is a thin adapter over `retrieve_v2` (the #1107 cutover). Both modifiers are therefore exposed on the live path. The entity-persistence demotion is **default-on and live on production**. The origin tie-break is exposed, but it is **held off** by default.

- **Entity-persistence demotion** (#1096, `[retrieval] use_entity_persist_demote` or `AELFRICE_ENTITY_PERSIST_DEMOTE`). This modifier is the sink for the junk-percolation problem of #1086. It is a log-additive **demotion** term `min(0, log(S1 + ε))`. The term reads `S1 = durable / (durable + transient + 1)` from the `belief_entities` index, in one batched query. The term down-weights the candidates that ground only to *transient* coordination tokens, relative to the *durable* entities. The transient tokens are bare pull-request numbers, bare issue numbers, version tags and branch tags. The durable entities are file paths, error codes and symbols. The term only demotes: a well-grounded candidate is neutral, and the term never boosts it. The term applies only to a candidate that carries an entity, so prose with no entity is untouched. The sink is **content-referential, not temporal**. A measurement found a sink on time or cold decay empirically inert here, because the junk is *recent*, not stale. This modifier is **default-ON and live on the production `retrieve()` path since v4.0**. It was flipped on once the G2 mixed-corpus eval of #1096 (#1103) cleared the no-regression gate. It then graduated onto the hook path with the #1107 cutover. To opt out, set a falsy value on the environment rung, on the keyword-argument rung or on the TOML rung.
- **Origin-priority tie-break** (#1089, `[retrieval] use_origin_tiebreak` or `AELFRICE_ORIGIN_TIEBREAK`). This is a tie-break inside a tier, not a rerank term. When two candidates tie on relevance, the *origin* with the higher trust wins. The tie-break sits between the relevance score and the id tie-break, so relevance always dominates. The result is byte-identical when the tie-break is off.

Both modifiers are deterministic (#605). Both give byte-identical results to the previous pipeline when their flags resolve falsy. The origin tie-break is default-off. The entity-persistence demotion is default-on, so opt out of it explicitly for that parity.

The token estimate is `(len(content) + 3) // 4`. An empty query returns L0 only. L0 always wins an overflow.

The specification documents are [entity_index.md](../design/entity_index.md) (L2.5), [bfs_multihop.md](../design/bfs_multihop.md) (L3) and [bayesian_ranking.md](../design/bayesian_ranking.md) (L1 Bayesian reranking).

**A caveat on the temporal coherence of BFS.** L3 resolves each hop to the globally latest serial of its target belief. For a recall query this behaviour is correct. An audit query asks what the agent believed at decision-time. For an audit query, a supersession that landed after the seed can appear in the middle of the chain. The fix for temporal coherence was first targeted at v2.0.0. It slipped, and it is not scheduled on a current milestone. See [LIMITATIONS § BFS multi-hop temporal coherence](../user/LIMITATIONS.md#bfs-multi-hop-temporal-coherence).

## Onboarding

`scan_repo(store, path)`:

1. **The filesystem walk** covers `*.md`, `*.rst`, `*.txt` and `*.adoc`. It produces `factual` and `requirement` candidates.
2. **The git log** produces `factual` candidates with the recency of the file. Since v1.1.0, `belief.created_at` is the file's most recent commit, so decay penalises old branches.
3. **The Python AST** gives the names of the functions and classes and their docstrings. These become `factual` candidates.

Classification uses the priors, with a regex fallback. The scan is idempotent on `content_hash`.

**The LLM onboard classifier (v1.3+, default-OFF).** `aelf onboard --llm-classify` routes each candidate through the vendor's small model instead of through the regex path. Four consent gates enforce the privacy boundary:

- the `[onboard-llm]` extra is installed;
- `ANTHROPIC_API_KEY` is present;
- the `--llm-classify` flag is set, or the `[onboard.llm].enabled` TOML key is set;
- a one-time interactive consent prompt is answered, and a sentinel file records the answer.

`--dry-run` previews the candidates and calls no API. The specification is [llm_classifier.md](../design/llm_classifier.md). This is the only path in aelfrice that transmits user content outbound. See [PRIVACY § Onboard-time outbound call](../user/PRIVACY.md#onboard-time-outbound-call).

## Claude Code hook

```
settings.json  hooks.UserPromptSubmit: [{command: "aelf-hook"}]
                          ↓ written by aelf setup
                  Claude Code spawns aelf-hook on each prompt
                          ↓ JSON payload on stdin
                  aelfrice.hook:main
                          ↓ retrieve(store, prompt)
                  .git/aelfrice/memory.db
                          ↓ <aelfrice-memory> block on stdout
                  Claude Code injects above your prompt
```

The non-blocking contract has two parts. Every failure path exits 0. A hook problem must never block your
prompt. Two qualifications apply. Both are deliberate, and both are reachable today.

- **One hook blocks by design.** `aelf-pre-issue-hook` exits `2` on a successful duplicate match
  (`pre_issue_create_hook.py`). See its row below. It never exits non-zero on an *error*. The blocking exit
  is the feature, not a failure path.
- **Partial stdout is possible.** The UserPromptSubmit lane writes the cadence-checkpoint block before the
  retrieval runs (`hook.py`). A failure in a later stage therefore exits 0 with that block already flushed.
  The guarantee is "exits 0", not "emits nothing".

## Default-on hooks (v2.1+ / v3.0+)

| Hook | Event | Purpose | Default since |
|---|---|---|---|
| `aelf-hook` | `UserPromptSubmit` | Injects the retrieval results. This is the core mechanism. | v1.0 |
| `aelf-transcript-logger` | `UserPromptSubmit`, `Stop`, `PreCompact`, `PostCompact` | Writes one JSONL line for each conversation turn. On PreCompact it rotates the file and ingests it again. | v2.1 (#529) |
| `aelf-commit-ingest` | `PostToolUse:Bash` | After a `git commit`, ingests the commit message with the triple extractor. | v2.1 (#529) |
| `aelf-session-start-hook` | `SessionStart` | Injects the locked beliefs as `<aelfrice-baseline>` once per session. Emits the `<recent-work>` sub-block (#887). | v2.1 (#529) |
| `aelf-stop-hook` | `Stop` | The lock prompt at the end of the session. It surfaces the unlocked beliefs of the session scope as `<aelfrice-session-end>`, with pre-filled `aelf lock` commands. Those beliefs are of the correction class (#582) and of the directive class (#1315). A directive command carries `--for` when a memory verb governs a stated window. This hook also hosts the cadence checkpoint dispatch, which is default-off (#749 / #871 / #876). | v3.0 |
| `aelf-search-tool-hook` | `PreToolUse:Grep|Glob` | Surfaces the relevant beliefs next to a tool-driven search (#674). | v3.0.1 (#738) |
| `aelf-search-tool-hook` | `PreToolUse:Bash` | The same entry point, with a separate matcher in settings.json, for a bash search invocation. | v3.0.1 (#738) |
| `aelf-pre-issue-hook` | `PreToolUse:Bash` | A guard that detects a duplicate before `gh issue create`. It blocks with exit 2 when the Jaccard overlap of the title is ≥ 0.5 against the open issues and the shipped commits (#941). | v3.5.0 |
| `aelf-claude-memory-mirror` | `PostToolUse:Write\|Edit\|MultiEdit` | A one-way mirror. It copies the writes to the host claude-memory fact files into the belief graph (#985). A consent gates it since v4.0 (#1089). It runs once the reconcile at first setup records the consent for the project. It also runs when `AELFRICE_MIRROR_CLAUDE_MEMORY` or `[memory] mirror_claude_memory` enables it explicitly. An explicit falsy value is the opt-out. | v3.7.0 |
| `aelf-agent-context-hook` | `PreToolUse:Agent\|Task` | Injects the context for a worker. A dispatched worker inherits the L0 locked beliefs and the task-relevant beliefs through the harness `updatedInput` channel. The passthrough fails open. The kill switch is `AELFRICE_AGENT_CONTEXT=0` (#1068). | v4.0.0 |
| `aelf-pre-compact-hook` | `PreCompact` | Does the bookkeeping for the trigger mode of the rebuilder, and nothing else. It never injects (#1031). Opt in with `--rebuilder`. The default trigger flipped from `manual` to `threshold` at v3.1 (#746). | opt-in |
| `aelf-session-start-hook` | `SessionStart(source="compact")` | Emits the rebuild block after compaction. This is the channel that the harness honors (#1031). | opt-in |

You opt out of each lane with `aelf setup --no-<lane>`. All the lanes exit 0 on failure, with the two
qualifications stated under the non-blocking contract above. `aelf-pre-issue-hook` exits `2` on a duplicate
match by design. Partial stdout is possible when an early stage has already emitted.

The `PostToolUseFailure:<tool_name>` namespace of event names inside
`~/.aelfrice/hook-activity.jsonl` is reserved for the raw observation
of a tool failure from a hook on the HOME side. That hook is tracked
separately. See
[hook_activity_schema](../design/hook_activity_schema.md) for the schema of the
fields. That document also gives the warning about dedupe by fingerprint on the
consumer side.

## Post-compaction rebuilder

> **Delivery channel (corrected by [#1031](https://github.com/robotrocketscience/aelfrice/issues/1031)).** The rebuild block ships on **SessionStart** with `source == "compact"`, *after* compaction. It does not ship on `PreCompact`. The harness rejects `additionalContext` emitted from a `PreCompact` hook, because `PreCompact` is absent from the events that support `additionalContext`. A block written there is discarded with a validation error. `pre_compact()` therefore emits nothing on stdout. It is retained only for parity of the trigger mode. `rebuild_v14` and the content of the block are unchanged.

Compaction is reached in two ways. The harness reaches its context limit, or the user compacts explicitly. Both ways take the same path. `PreCompact` fires. The harness compacts. `SessionStart` then fires with `source == "compact"`. The rebuilder does its bookkeeping on the first event and its injection on the last event. Do not read the flow below as a path for automatic compaction only. On a measured local corpus, 72 of 73 scoreable compactions were explicit rather than automatic ([#1252](https://github.com/robotrocketscience/aelfrice/issues/1252)). An operator who debugs the rebuilder is therefore most likely on the explicit route:

```
PreCompact fires
      ↓
aelf-pre-compact-hook resolves trigger_mode and surfaces the
dynamic-mode parked trace on stderr — no stdout, nothing injected
      ↓
the harness compacts the conversation
      ↓
SessionStart fires with source == "compact"
      ↓
aelf-session-start-hook reads the last N turns from turns.jsonl
      ↓
rebuild_v14(recent_turns, store, token_budget)
      → L0 locked beliefs (always first)
      → session-scoped beliefs matching recent content
      → BM25+posterior hits against the session tail
      packed to token_budget (default: [rebuilder].token_budget in .aelfrice.toml)
      ↓
emitted as additionalContext — the aelfrice block lands in the
new context alongside the harness's own summary (augment mode)
```

`aelf rebuild [--transcript PATH] [--n N] [--budget N]` runs the same codepath manually. It prints the block to stdout. Install it with `aelf setup --rebuilder`. The default `DEFAULT_TRIGGER_MODE` flipped from `"manual"` to `"threshold"` at v3.1 ([#746](https://github.com/robotrocketscience/aelfrice/issues/746)). The specification is [context_rebuilder.md](../design/context_rebuilder.md). The policy for the eval fixtures is [eval_fixture_policy.md](../design/eval_fixture_policy.md).

## Tests

| Layer | Marker | Coverage |
|---|---|---|
| Unit | default | One property per test. Pyright runs in strict mode. |
| Property | default | The pre-registered invariants: Bayesian inertia, decay-required, lock-floor sharpness, the token-budget invariant and broker attenuation. |
| Regression | `@pytest.mark.regression` | The scenarios that cross modules: the retrieval round-trip, the feedback loop, the onboarding, the setup→hook→unsetup sequence and `aelf bench` end to end. |

Run `uv run pytest`. The suite holds 7,300+ tests at v4.2.0.

## Out of scope through v1.x

These items stay parked until a benchmark, an experiment or a concrete failure mode justifies them:

- Sentence-transformer embeddings. The HRR primitives shipped at v1.7.0 as a structural lane, not as a learned-embedding lane. The PHILOSOPHY ratification at v3.0 (#605) keeps determinism as the property. No embedding lane is planned.
- Multi-writer federation and conflict-free replicated data type (CRDT) primitives. v3.0 ships *read-only* federation (#650 / #655 / #688). A peer opens a foreign DB read-only and takes the UNION of the FTS5 results. The multi-writer extension (the CRDT primitives of #651-#654) closed WONTFIX at the v3.0 cut, per the #661 ratification.
- The full composition tracker. It covers the 10-round uplift in mean reciprocal rank (MRR). It covers the calibration measured by the expected calibration error (ECE). It covers the composition eval of BM25F × heat-kernel × HRR-structural (#154). The heat-kernel default and the HRR-structural default flipped on at v2.1. The joint-composition bench gate as such was not run separately. But the #437 reproducibility-harness cleared 11/11 **at the v2.1.0 cut (2026-05-08)**, and it covers the
substrate as of that measurement. The README badge currently reads `partial (6/11 adapters)`. That badge
reports a later regression in the nightly canonical run. That reading is not a retraction of the v2.1.0 gate. Read the
11/11 as the evidence that stood at the time of the flip. Do not read it as the present state of the harness.

The following items were listed here in the past. They have since shipped:
- Posterior-aware retrieval ranking → **shipped v1.3.0** (partial; [bayesian_ranking.md](../design/bayesian_ranking.md))
- BFS multi-hop graph retrieval → **shipped v1.3.0** ([bfs_multihop.md](../design/bfs_multihop.md))
- Entity index and named-entity recognition (NER) → **shipped v1.3.0** ([entity_index.md](../design/entity_index.md))
- LLM in the hot path (the optional onboard classifier) → **shipped v1.3.0** ([llm_classifier.md](../design/llm_classifier.md))
- BM25F anchor-text retrieval → **shipped v1.7.0**, default-on (#148/#154). The uplift is +0.6650 in normalized discounted cumulative gain at k (NDCG@k) on the v0.1 retrieve_uplift fixture.
- HRR primitives and the structural lane → **shipped v1.7.0**, default-on as of v2.1 ([feature-hrr-integration.md](../design/feature-hrr-integration.md)). The source is at `src/aelfrice/hrr_index.py`. This closes three items: the vocabulary-gap-recovery claim, the #154 composition tracker, and the #437 reproducibility-harness at 11/11 (the v2.1.0 cut, 2026-05-08). See the note above on the current badge.
- Heat-kernel authority scorer → **shipped v1.6.0** (opt-in), default-on as of v2.1 (the #154 composition tracker)
- HRR persistence (save and load in the split format `.npy` and `.npz`, default-on) → **shipped v3.0** (#553)
- Wonder lifecycle (the graph walk, the axes dispatch, and the phantom promotion of Surfaces A and B) → **shipped v2.0/v3.0** ([#542](https://github.com/robotrocketscience/aelfrice/issues/542) umbrella)
- Read-only cross-project federation → **shipped v3.0** (#650 / #655 / #688)
- The LLM judge in the eval harness, with Cohen's-κ calibration → **shipped v3.0** (#592 / #600 / #687)
- Type-aware compression A2 bench gate → **shipped v3.0** (#434)
- `query_strategy` stack-r1-r3 default → **shipped v3.0** (#718), **reverted to legacy-bm25** (#1501)
