# Limitations

This page lists what aelfrice does not do yet. The project tracks these limitations openly.

## The big one: feedback doesn't drive ranking (lifted at v1.3.0, partially)

Through v1.2.x, `apply_feedback` updates `(α, β)`. `apply_feedback` also writes an audit row. Through v1.2.x, `aelf stats` and the Model Context Protocol (MCP) show the posterior mean. But L1 retrieval orders the hits by `bm25(beliefs_fts)` alone. L1 retrieval does not order the hits by the posterior. If you mark a belief `harmful`, `apply_feedback` lowers the posterior mean of that belief. The next retrieval that matches the keywords of that belief still surfaces it.

The benchmark harness shipped at v1.0 as the measurement instrument. The harness was not yet a proof of the feedback claim. Three changes closed that gap in steps:

- the v1.3 partial Bayesian re-rank,
- the v1.6 deferred-feedback sweeper,
- the v1.7 BM25F default-on flip (see below).

**At v1.3.0:** the ranking starts to use the posterior. The L1 score becomes `log(bm25) + 0.5 * log(posterior_mean)`. Locked beliefs (L0) bypass the scoring as before. The cache invalidates correctly through the existing store-mutation hook.

**At v1.6.0:** the evaluation harness for mean reciprocal rank (MRR) uplift and expected calibration error (ECE) ships at `benchmarks/posterior_ranking/`. Run that harness with `python -m benchmarks.posterior_ranking`. The heat-kernel composition wiring also lands, as a log-additive term in the ranking score (#151, #306, #310). Both of these ship default-OFF.

**At v2.1.0:** the #154 default-flip lands after the gate of the #437 reproducibility harness clears 11/11. `use_heat_kernel` and `use_hrr_structural` flip to default-on.

**At #1162:** `use_heat_kernel` returns to default-off. The lane needs a `GraphEigenbasisCache` that no production caller constructs. Therefore the on-default described a lane that could not fire. The flip does not change the ranking. `LaneTelemetry.heat_used` now reports the reachability at runtime.

See [`docs/design/bayesian_ranking.md`](../design/bayesian_ranking.md) for the v1.3 contract.

## No semantic similarity

Retrieval is Best Matching 25 (BM25) keyword search over full-text search version 5 (FTS5). FTS5 uses porter unicode61 stemming. Since v1.7.0, retrieval uses BM25F with anchor-text augmentation by default. The query "deploy" does not find "publish to prod" without a tokenizable substring overlap.

This is a deliberate scope choice. It is not a roadmap item. The addition of embeddings would break determinism along the full path — see [PHILOSOPHY § Determinism is the property](../concepts/PHILOSOPHY.md#determinism-is-the-property). For a fuzzy semantic recall query, use aelfrice together with a separate tool. Do not blend embeddings into the retrieval path.

## Onboarding scope

The CLI scanner reads three sources:

- prose files (`*.md`, `*.rst`, `*.txt`, `*.adoc`),
- `git log`,
- the Python abstract syntax tree (AST).

The scanner does not read the abstract syntax tree of JavaScript, TypeScript, Rust or Go yet.

The research line shipped a larger set of extractors:

- a scanner for citation references across markdown bodies,
- a linkage between a test and its implementation, taken from the filename patterns and the import patterns,
- a directive detector that captured imperative user statements as TODO beliefs.

These extractors stay deferred from the scope. The directive-detection path has consequences for the architecture. If that path lands, it lands together with the violation-detection tier of [PHILOSOPHY § What we can and can't guarantee](../concepts/PHILOSOPHY.md#what-we-can-and-cant-guarantee).

Classification on the CLI path uses regex-based priors by default. Three paths give classification of higher quality:
- **`/aelf:onboard <path>`** (v1.5.x, default-on) — a host-driven flow. The flow drives the four-class classifier through the Task tool of the host model. This path needs no API key. If the Task tool of the host is not available, the flow falls back to the regex classifier. See [llm_classifier.md](../design/llm_classifier.md).
- **`/aelf:onboard`** — the polymorphic flow. This flow routes through the host LLM.
- **`aelf onboard --llm-classify`** (v1.3+, default-off) — routes through Claude Haiku directly. Requires `ANTHROPIC_API_KEY`. Four consent gates enforce the privacy boundary. See [llm_classifier.md](../design/llm_classifier.md).

## BFS multi-hop temporal coherence

The v1.3.0 breadth-first search (BFS) multi-hop expansion resolves each hop independently. Each hop resolves to the globally latest serial of its target belief. See [bfs_multihop.md](../design/bfs_multihop.md).

An audit-shaped query asks "what did the agent believe at the time it made this decision?". For such a query, the expansion can show a chain whose intermediate hops are later than the session of the seed. This is a loss of fidelity against a corpus where supersessions accumulated after the write of the seed.

The default retrieval mode is recall, not audit. The latest-serial-per-hop rule serves that mode correctly. The plan targeted the temporal-coherence fix at v2.0.0, together with the reproducibility harness. The harness shipped (#437). The temporal fix did not ship at v2.0 or v2.1, and no current milestone schedules it. The section "Open question: temporal coherence" of the spec gives the reasons to defer the fix at v1.3.

## Sharp edges

- **Locks are durable.** A fresh lock is `(α, β) = (9.0, 0.5)`. Passive feedback does not move a lock. Passive feedback is `aelf feedback <id> harmful`, sentiment-derived turn valence, retrieval exposure and valence propagation. `apply_feedback` enforces this rule. The deferred sweeper enforces the same rule, since [#1168](https://github.com/robotrocketscience/aelfrice/issues/1168). `feedback_history` still records the event. The CLI tells you that it recorded the event but did not apply it. `aelf confirm` is the exception. `aelf confirm` is an explicit affirmation, not passive feedback, and it still moves a locked posterior. Before #1168 the floor was only in the dead `decay()` path, so passive feedback did move locks. v3.2.0 [#814](https://github.com/robotrocketscience/aelfrice/issues/814) removed the v2.x `demotion_pressure` machinery and the auto-demote machinery. v3.1.0 as released still contained that machinery, see #833. To change a lock, use `aelf unlock` or `aelf delete`, or lock the corrected statement again. `aelf demote <id>` drops the lock immediately.
- **Contradiction detection is partial.** Contradiction **resolution** ships. Resolution is `aelf resolve`, the tie-breaker that selects a winner for a `CONTRADICTS` edge. Contradiction **detection** is deferred. Detection is the detector for semantic divergence that runs after an insert and creates those edges automatically. Until detection ships, contradiction-flagging appears only on the edges that the triple extractor produces from its six explicit relation-family regexes. The `/aelf:reason` `VERDICT=CONTRADICTORY` path also emits structured impasses at query time, but only over persisted `CONTRADICTS` edges. The query-time surface that needs no edge is the slot-conflict flag `AELF_SHOW_CONFLICTS=1` on `aelf search` (#938, v3.5.0). You must opt in to that flag. The flag marks the hits whose value-slots collide with a locked belief.
- **aelfrice does not capture natural-language sentiment automatically.** You can write "ok good" or "no that's wrong" in the chat. That text does **not** strengthen the beliefs the agent just used. That text does **not** weaken them either. The explicit CLI command `aelf feedback <id> used|harmful` stays the channel with high confidence. Retrieval **exposure** is not a posterior signal by default. Since #1086 (v4.0), a retrieval that surfaces a belief is audit-only (`AELFRICE_EXPOSURE_UPDATES_POSTERIOR`, default off). aelfrice logs the exposure for the recurrence axis and leaves α/β untouched. A retrieval is an exposure, not an endorsement. No residual path treats exposure as evidence. The v1.6 deferred-feedback sweeper ([#191](https://github.com/robotrocketscience/aelfrice/issues/191)) applied a small alpha bump after a grace window. The default bump was +0.05, and the default window was 30 minutes. The sweeper did not apply the bump if an explicit signal arrived first. But `aelf sweep-feedback` is **audit-only since [#1162](https://github.com/robotrocketscience/aelfrice/issues/1162)**. The sweeper classifies every pending row exactly as the mutating sweeper did, and reports what it *would* have applied. The sweeper writes no `alpha` move, no `feedback_history` row and no change of the queue status. The enqueue side is default-off too (`AELFRICE_IMPLICIT_FEEDBACK_ENQUEUE`), so `retrieve()` banks no rows unless you opt in. aelfrice does not parse the affect words of the user either. You can opt in to the sentiment-from-prose detector (`[feedback] sentiment_from_prose = true` in `.aelfrice.toml`, default off, #193/#606). That detector uses regexes to match short affect phrases. The detector then distributes the signal across the beliefs retrieved in the previous turn.
- **A drop of the confidence below 0.5 raises no automatic flag.** aelfrice shows no warning for a belief whose posterior drifts under the prior. In v3.x, negative evidence drives no automatic change of state. [#814](https://github.com/robotrocketscience/aelfrice/issues/814) removed the v2.x demotion-pressure auto-demote. To find the beliefs that drift, use `aelf speculative --json`. That command emits α and β for each belief that is not locked. As an alternative, query the SQLite store directly, for example `SELECT id, alpha, beta FROM beliefs WHERE alpha / (alpha + beta) < 0.5`.
- **The organic sink is a rerank modifier, not a posterior decay. It demotes a belief, it does not delete a belief.** #1086 established that the store held no automatic force to lower the rank of low-value coordination junk, now that exposure is audit-only. Coordination junk is a bare pull-request number, a bare issue number, and version chatter or branch chatter. The answer that shipped is the entity-persistence demotion lane (#1096, `[retrieval] use_entity_persist_demote`). That lane is a deterministic rank down-weight. The lane ranks the candidates that ground only to transient tokens below the candidates that ground to durable entities. A durable entity is a file path, an error code or a symbol. The lane is content-referential, not time-based. An empirical measurement showed that a temporal sink or a cold-decay sink is inert, because the junk is *recent*, not stale. At **v4.0 the lane is default-on and live on the production `retrieve()` path**. The flip happened once #1103 cleared the G2 gate. The #1107 cutover graduated the lane onto the live hook path. The residual limitation is that the lane **ranks, it does not delete**. Junk still occupies rows in the store. Junk can appear again when the other candidates for a query are weak. There is no automatic posterior decay, and there is no hard garbage collection of stale content. To *find* junk for manual curation, use `aelf introspect --only-noise`. To soft-delete junk, use `aelf retire`.
- **Origin trust is only a default-off tie-break.** The origin-priority tie-break is #1089, `[retrieval] use_origin_tiebreak`, **default off**. The tie-break lets an origin of higher trust win a *tie* on relevance. The tie-break is not a rerank term, and it never overrides relevance. A measurement on LoCoMo refuted an origin *rerank lane* (#1013). On LoCoMo the real gap was BM25 recall, and reranking cannot correct that gap.
- **aelfrice does not track the decided-vs-floated lifecycle automatically.** `aelf introspect` reports a floated-vs-decided *status*. `aelf introspect` reads that status off the existing `RESOLVES` and `POTENTIALLY_STALE` edges. But no producer at ingest time detects that an option floated in conversation later becomes a decision, and no producer promotes that option. This is fix #4 of the #1086 scoring umbrella. A measurement showed the fix is not retrofittable, because no genuine float-to-decision pairs exist on real stores to seed from. The fix is also not applicable forward, because the #1081/#1083 hedge-drop discards float sentences at capture. Therefore the project removed the fix from the scope ([#1100](https://github.com/robotrocketscience/aelfrice/issues/1100), closed).
- **The decay target reads as 0.5.** Fresh beliefs start at priors that are adjusted for the type and for the source (0.375–0.95). Fresh beliefs do not start at 0.5. The Jeffreys prior `(0.5, 0.5)` is the decay *target*. A belief whose evidence decayed fully reads exactly `0.5`. That value means "no surviving evidence". That value does not mean that the belief is as likely to be true as false.
- **`aelf onboard` is not incremental on duplicates.** A repeated run is idempotent. aelfrice does not re-score or refresh an existing belief. Since v1.7.0 (#264, closed), every ingest entry point writes to `ingest_log`. The derivation worker then materialises `beliefs`. Therefore `aelf doctor --derive-pending` runs the worker once over any unstamped rows of the log. That command helps when an earlier batch crashed in the middle of the stamp operation. The view-flip that makes `beliefs` a materialized projection of the log shipped under #265 (closed 2026-05-08). That view-flip is gated default-off behind `AELFRICE_WRITE_LOG_AUTHORITATIVE`. **The log covers `edges` only to a small degree.** Until [#1354](https://github.com/robotrocketscience/aelfrice/issues/1354), aelfrice wrote every edge outside the log, and `ingest_log.derived_edge_ids` was NULL on every row. `derive()` now emits the intra-turn `DERIVED_FROM` edges, and the column populates forward-only on new rows. Those rows are at most 1.93% of the live edge set. aelfrice still writes the other 98% outside the log. That other 98% is `TEMPORAL_NEXT` plus the relationship detectors and the contradiction detectors. Therefore a replay still reconstructs the beliefs, but not the graph ([#1283](https://github.com/robotrocketscience/aelfrice/issues/1283)). The CLI still exposes no re-derivation of the rule set without a new onboard run.
- **Near-duplicates from different ingest paths persist.** A UNIQUE constraint on `content_hash`, the sha256 of the belief text, dedupes exact matches. That constraint does not dedupe paraphrases. For example, a locked belief "don't push to main" and a scanned belief "never push directly to main" both surface in retrieval. The v1.7.0 dedup audit pass and the `aelf doctor --dedup` CLI command (#197 R1, see CHANGELOG) list the duplicate clusters that Jaccard and Levenshtein confirm. That list is read-only. The SUPERSEDES hook on the write path (#197 full module) is bench-gated and still deferred.
- **There are no bulk operations.** aelfrice has no batch lock, no `delete <pattern>` and no merge.
- **There is no edit operation.** To correct a wrong belief, insert a new belief with a `SUPERSEDES` edge. The original belief stays in the store.
- **Graph visualization is available at the CLI only.** `aelf graph` ([#629](https://github.com/robotrocketscience/aelfrice/issues/629)) emits DOT subgraphs and JSON subgraphs. The edges carry color codes. The nodes carry shades for the lock state and for the posterior. A locked node is cyan. A high posterior is green, and a low posterior is red. For image output, pipe the result through Graphviz (`dot -Tpng`). There is no graphical user interface (GUI). For raw inspection, run `sqlite3 "$(python -c 'from aelfrice.cli import db_path; print(db_path())')"`.
- **The JSONL batch ingest has no scrubber for personally identifiable information (PII).** `aelf ingest-transcript --batch ~/.claude/projects/` pulls whatever you typed in the chat into the local belief graph. Review the material before you backfill it.
- **Hook framing relies on the model to honor the trust boundary.** The `UserPromptSubmit` hook ships three structural defenses: the framing tag, the escape of a tag substring at render time, and the per-turn audit log. See [hook_hardening.md](../design/hook_hardening.md) and [PHILOSOPHY § Trust boundary at the hook surface](../concepts/PHILOSOPHY.md#trust-boundary-at-the-hook-surface). The user-turn text next to the block can name session artifacts. The hook cannot force the model to verify those artifacts. The hook also cannot guarantee that the model treats `<belief>`-wrapped content as data rather than as instruction. A model that ignores the framing is a failure of the model layer, not a failure of the hook layer. The `hook_audit.jsonl` log is the recovery surface. That log records what aelfrice injected, so a human can review it after the fact.

- **The model honors locks, but nothing enforces them. Put a true prohibition in a hook.** aelfrice injects a locked belief as high-trust context. Since v3.7, the provenance-aware framing ([#1016-A](https://github.com/robotrocketscience/aelfrice/issues/1016)) presents user locks as standing instructions, and capable models follow locked rules in practice. But that compliance is a *behaviour of the model layer*, never a hard gate. The same caveat about the trust boundary above applies. Put a **must-never** constraint in a **`PreToolUse` enforcement hook**, not in a lock. Examples of such a constraint are "never push to prod", "never `rm -rf` the data dir" and "never commit secrets". The hook blocks the action deterministically (exit 2), whatever the model decides. Use a lock for a ground-truth fact and for a standing preference you want the agent to follow. Use a hook for an action that must be *impossible*. aelfrice ships one such gate as a pattern. `aelf-pre-issue-hook` (`PreToolUse:Bash`) blocks a duplicate `gh issue create` call. See [ARCHITECTURE § hooks](../concepts/ARCHITECTURE.md). This is also the reason that lock injection is bounded rather than unlimited (v3.7 #1016-B layered locks). Locks are advisory context with a budget, not an enforcement substrate.

## Out of scope

These are scope choices. They follow from the commitments of aelfrice. They are *not* roadmap items.

### Sharing, sync, or distributed-write federation

aelfrice ships no mechanism to sync the memory contents between users or machines. aelfrice also ships no mechanism to distribute those contents by write, or to replicate them in any other way. The brain graph stays on the machine that wrote it.

This is a choice about privacy and audit. A graph derived from real session activity contains these items:

- filesystem paths,
- hostnames,
- internal URLs,
- names from the git config,
- details of the project architecture,
- content that the agent inferred from the chat.

None of these items is suitable for cross-machine distribution by default. A per-belief allowlist is not reliable enough to make an automated export safe.

To bootstrap a new clone or a new collaborator, run `aelf onboard .`. The command re-extracts the graph from the publicly-visible repo content. To share rules, record them in CLAUDE.md, CONTRIBUTING.md or other repo-tracked prose. The onboard scanner then reads those rules.

**Read-only cross-project federation shipped at v3.0** (#650, #655, ratified as read-only under #661). A local DB can declare peer DBs through `knowledge_deps.json`. The local DB then surfaces the beliefs of those peers in retrieval. But the DB of each project is the sole writer for its own beliefs, and the mutation tools reject a foreign belief ID. That mechanism is read-only by construction. There is no cross-machine write replication, and there is no conflict-free replicated data type (CRDT) layer for distributed writes. Therefore the privacy and audit properties above stay preserved. Multi-writer federation is not on the roadmap.

### Multi-session aggregation

aelfrice is not optimised for a query such as "how many times did the user mention X across last quarter?". That query is a task for an LLM with retrieval-augmented generation (RAG) and a summary buffer. It is not a task of recall for a behavioural directive. On a benchmark such as LongMemEval multi-session, an embedding system will outperform aelfrice for that query category.

The principled response is to add aggregative-query routing at the structural-analysis layer. That layer uses SQL aggregations over `feedback_history`, scoped graph walks and time-bucketed COUNT queries. The principled response is not to add embeddings.

### Multi-project query

Only one DB writes at a time. The beliefs written in project A are not *written* into project B, because the two projects have different `.git/` directories. Use `AELFRICE_DB` to scope each project explicitly. The read-only federation of v3.0, above, is the path to *read* the beliefs of a peer project into a local query. That path does not merge the underlying stores.

## Compatibility

- Python 3.12 or 3.13.
- The project tests macOS and Linux routinely. The full suite runs on both platforms on every pull request.
- aelfrice supports Windows at a narrower level. This page states the difference
  plainly, because the page previously said "should work but is not exercised on
  every release". That statement was wrong. Until
  [#1329](https://github.com/robotrocketscience/aelfrice/issues/1329) aelfrice
  did not start on Windows at all. Every command died at import on the Unix-only
  `fcntl` module. The claude-memory directory encoder also produced a name that
  Windows cannot create. The project fixed both defects. A `windows-latest` job
  now asserts these properties:
  - `aelf --help` runs.
  - `aelf doctor` does not crash at import.
  - The portability suite passes.
  - The encoder produces a creatable directory name.

  That job does **not** run the full suite, because much of the suite assumes
  POSIX semantics. So aelfrice runs on Windows, and the project tests its
  portability surface. The project does not verify the individual subcommands
  there one by one yet.
- Advisory file locking uses `flock` on POSIX and `msvcrt.locking` on Windows.
  If the host or the backing filesystem supplies neither of them, the locking
  degrades to a no-op. Single-process use stays unaffected. But two concurrent
  `aelf` processes are **not** serialised. Two such processes can interleave a
  read-modify-write of the same settings file or the same ring file.
  `aelfrice.file_lock.HAVE_ADVISORY_LOCKS` reports which of the two cases you
  are in.
- Since v3.0.1, `uv tool install aelfrice` is the only supported install channel ([#730](https://github.com/robotrocketscience/aelfrice/issues/730)).

## Reporting

To add a limitation or to correct one, file an issue tagged `limitations`.
