<p align="center"><img src="docs/assets/01-hero-kulili.png" width="100%" alt="A figure of shimmering cloud rising from a dark sea, weaving threads of light into a constellation of beliefs"></p>

# aelfrice

> Your AI stops forgetting.
> Set up once. Stays out of the way.
>
> _Local SQLite. Fully auditable. No GPU, no network._

[![PyPI](https://img.shields.io/pypi/v/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![Python](https://img.shields.io/pypi/pyversions/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![License](https://img.shields.io/pypi/l/aelfrice.svg)](LICENSE)
[![CI](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml/badge.svg)](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml)
[![OSSInsight](https://img.shields.io/badge/OSSInsight-analytics-blue)](https://ossinsight.io/analyze/robotrocketscience/aelfrice)
<!-- bench-canonical-badge:start -->
[![Reproducibility](https://img.shields.io/badge/reproducibility-partial%20%286%2F11%20adapters%29-yellow)](docs/v2_reproducibility_harness.md)
<!-- bench-canonical-badge:end -->

You correct your agent. *"Got it,"* it says. Next session, same mistake.

aelfrice runs in the background and stops the amnesia and context drift. You write a rule once and it gets attached to every prompt thereafter — no cross-references for the agent to skip, no markdown files to maintain, nothing to remember to do.

```bash
pipx install aelfrice    # or: uv tool install aelfrice
aelf setup               # wire the hook into your agent
aelf onboard .           # scan the current project and ingest beliefs
```

Then add your first rule and restart your agent:

```bash
aelf lock "never push directly to main; use scripts/publish.sh"
```

That's it. Your next prompt that mentions "push" already has the rule attached. From here on out aelfrice is invisible — no command to remember to run, no file to keep updated.

> The `aelf lock` line above is an example — substitute your own rule. Skip it entirely if you'd rather start with onboarded beliefs only and add locks as you go.

---

## What makes aelfrice different

<p align="center"><img src="docs/assets/02-eterne-hrr.png" width="100%" alt="A layered retrieval pipeline: keyword matches at the surface, structural-marker queries below, locked beliefs threading through both"></p>

- **The agent can't skip the rule.** The `UserPromptSubmit` hook injects matched beliefs into your prompt *before* the model sees it. Not "the agent will check a file" — the file is already in the prompt.
- **Bit-level determinism.** No embeddings, no learned re-rankers, no LLM in the retrieval path. Same write log, same code → bit-for-bit identical results across runs and machines. ([PHILOSOPHY.md](docs/PHILOSOPHY.md))
- **Every belief has a confidence and a confidence-in-its-confidence.** A `(α, β)` Beta-Bernoulli posterior gives both: `α / (α+β)` says which way the belief leans, `α + β` says how sure we are of that lean. New beliefs sit at low evidence (high variance, retrievable but discounted); locked beliefs short-circuit decay and pin as ground truth.
- **One prompt, every layer responds.** Four parallel lookups happen at once, each catching what the others miss: locked rules pin to the front, keyword match for literal word overlap, anchor-text match that weighs labels and references heavier than body prose, and a graph lane that finds beliefs by how they connect rather than what they say. You don't have to pick. The build cost is paid once when you onboard the project — every prompt thereafter gets every layer for free.
- **Local-only.** SQLite at `<git-common-dir>/aelfrice/memory.db` (or `~/.aelfrice/memory.db` outside git). No network calls, no telemetry, no accounts. One brain per project, all on your machine. ([PRIVACY.md](docs/PRIVACY.md))
- **Auditable to the row.** Every belief has an `origin` column (`user_stated`, `user_corrected`, `commit_ingest`, …) tying it to the action that wrote it. Open the DB in any SQLite browser; nothing hidden. Query it with the tools; full traceability.
- **Reversible.** `aelf uninstall --archive backup.aenc` encrypts the DB and deletes the live copy. `--purge` wipes it. `--keep-db` leaves data untouched. No vendor lock-in by construction. You're the boss of your memories.
- **No GPU, no network, no inference cost.** Runtime deps are `numpy`, `scipy`, `snowballstemmer` — all CPU, all offline. Retrieval is a sparse-matrix query, not an LLM call. Operations are fast and local.

---

## What it does

When you submit a prompt in Claude Code, aelfrice's `UserPromptSubmit` hook fires before the model sees your message. It runs a two-layer search:

```
L0: locked beliefs   -> rules you marked permanent (always returned)
L1: FTS5 keyword     -> SQLite full-text search, BM25-ranked
```

The matching beliefs come back as an `<aelfrice-memory>` block prepended to your prompt. The agent reads it as part of the prompt — it doesn't have to remember to check a file.

```text
<aelfrice-memory>
[locked] never push directly to main; use scripts/publish.sh
[locked] commits must be SSH-signed with ~/.ssh/id_rrs
         the publish script runs gitleaks before tagging
</aelfrice-memory>

push the release
```

Default budget is 2,400 tokens per prompt. Locked beliefs are the always-injected pool — every lock ships on every prompt, in full, regardless of relevance score. **Lock count is your baseline-context budget knob:** if you've locked 200 things, every session opens with all 200, by your design. The non-locked pool (FTS/L1) is BM25-ranked and truncated to fit.

### Session-start enrichment vs per-turn retrieval

The first `UserPromptSubmit` of a new session carries extra context compared to subsequent prompts in the same session.

**Per-turn retrieval** (every prompt): BM25-ranked beliefs matching your current prompt, wrapped in `<aelfrice-memory>`.

**Session-start enrichment** (first prompt only): a `<session-start>` sub-block is embedded inside that same `<aelfrice-memory>` envelope. It contains two sections:

- `<locked>` — all user-locked beliefs (the full L0 pool, same as `aelf locked`).
- `<core>` — load-bearing unlocked beliefs: those with corroboration count ≥ 2, or posterior mean ≥ ⅔ with sufficient feedback mass (α+β ≥ 4). Same selection as `aelf core`.

Detection is session-scoped. aelfrice records the last-seen `session_id` in `<git-common-dir>/aelfrice/session_first_prompt.json`. When the id changes (or the file is absent), the current call is treated as the first prompt of a new session; subsequent calls with the same id skip the sub-block.

Cost: one additional `list_locked_beliefs()` + one belief-id walk per session, not per prompt.

---

## What it remembers

| You run | It stores |
|---|---|
| `aelf lock "never commit .env files"` | Permanent rule. Returned on every retrieval. |
| `aelf onboard .` | Walks the project — git log, README headings, code structure — and ingests structural facts. |
| `aelf feedback <id> used` | Bayesian feedback. Strengthens the belief's posterior. |
| `aelf feedback <id> harmful` | Weakens it. After enough independent harmfuls, locks auto-demote. |
| _(passive — no command)_ | Default-on auto-capture: every prompt/response turn is logged to per-project JSONL and ingested into the belief graph at compaction; successful `git commit` events are ingested too. See [INSTALL § hooks](docs/INSTALL.md). Opt out with `aelf setup --no-transcript-ingest --no-commit-ingest`. |

Each belief carries a `(α, β)` Beta-Bernoulli posterior. `α / (α+β)` is the confidence. Locks short-circuit decay; everything else fades over time so stale beliefs eventually drop out of retrieval.

```bash
aelf stats
# beliefs:    142   locked: 8   threads: 67
# feedback:   31    avg_confidence: 0.71
```

---

## Why files don't solve this

The standard workaround for "agent keeps forgetting" is more files: `STATE.md`, `DECISIONS.md`, a CLAUDE.md with cross-references to runbooks. Every cross-reference is a bet that the agent will read the file, find the right section, and follow what it says.

The failure modes are predictable. The agent reads the rule and runs `git push` anyway. Cross-references break silently after compaction. State files rot the moment you forget to update them. Each new failure mode begets another file.

aelfrice replaces the chain with a mechanism. The hook injects matched beliefs *as part of your prompt*, before the agent sees it. Nothing voluntary. Nothing the agent can skip.

| Manual approach | What breaks | aelfrice |
|---|---|---|
| Rules in CLAUDE.md | Agent reads them, doesn't follow them | Injected per-prompt, not per-session |
| Cross-references | Agent skips or reads the wrong section | Matched beliefs injected directly |
| Hand-maintained state files | One missed update breaks the chain | State is the SQLite DB; no manual sync |

---

## Why this is a memory system, not a key-value store

[Leonard Lin's review of agentic-memory implementations](https://github.com/lhl/agentic-memory/blob/main/ANALYSIS.md) frames the bar bluntly:

> The biggest differentiator is not "vector DB vs SQLite" — it's **write correctness and governance**: provenance / audit trail, write gates / confirmation, conflict handling, reversibility (inspect / edit / delete).

By that bar, "a vector store with a similarity query" is not a memory system — it is a search index. A memory system has to answer *who wrote this, when, via what ingress, what supersedes it, and how do I take it back*. Here is how aelfrice answers each.

| Lin's pillar | What it means | aelfrice mechanism |
|---|---|---|
| **Provenance / audit trail** | Every row traces back to the action that wrote it: who, when, via what ingress channel. | `origin` column on every belief — `user_stated`, `user_corrected`, `user_validated`, `agent_inferred`, `agent_remembered`, `document_recent`, `speculative` ([`src/aelfrice/models.py`](src/aelfrice/models.py)). Append-only [`ingest_log`](src/aelfrice/store.py) table records every raw input with its source kind, source path, and session id — tear the DB down and rebuild it from this log alone. `content_hash` binds each row to its content. `belief_versions` / `edge_versions` sidecar tables carry per-scope version vectors. Open the file in any SQLite browser; nothing is hidden. |
| **Write gates / confirmation** | Persistence is not unconditional. Some writes need explicit approval; external-origin claims cannot be laundered into ground truth. | Two-tier lock state: `lock_level ∈ {none, user}` with a `locked_at` timestamp and a `demotion_pressure` counter that blocks silent removal. `aelf lock` is the only path to L0 user-asserted ground truth — locked beliefs short-circuit decay and pin to every retrieval. `aelf confirm` only bumps a Beta-Bernoulli posterior — it cannot promote `origin` without an explicit `aelf promote`. The `(α, β)` posterior means feedback *accumulates* rather than overwrites; one harmful click does not erase a belief, it nudges the mean. |
| **Conflict handling** | Competing claims about the same thing are surfaced, not silently overwritten. | First-class edge types `CONTRADICTS` and `SUPERSEDES` in [`src/aelfrice/models.py`](src/aelfrice/models.py) — disagreement is a graph relation, not a vanished row. New facts about the same (entity, property) chain via `SUPERSEDES` rather than mutating in-place, leaving the prior claim queryable. Per-scope version vectors (#204 / #205) preserve causal ordering for concurrent edits across worktrees. |
| **Reversibility (inspect / edit / delete)** | Mutations remain auditable and partially undoable. The user is the boss of their memories. | `aelf delete <id>` writes an audit row *before* the cascade ([`cli.py:1363`](src/aelfrice/cli.py)). `aelf unlock` writes a `lock:unlock` audit row ([`promotion.py`](src/aelfrice/promotion.py)); `aelf promote` and its inverse leave `promotion:revert_to_agent_inferred` rows; `aelf feedback` lands rows in `feedback_history`. The `ingest_log` is append-only and replay-capable. At the top level: `aelf uninstall --archive backup.aenc` encrypts and removes the live DB, `--purge` wipes, `--keep-db` leaves data untouched. No vendor lock-in by construction. |

---

## What you get for free

Running in the background. No action required after `aelf setup`.

- **Passive capture.** Default-on transcript-ingest, commit-ingest, and session-start hooks (since v2.1). Session activity flows into the belief graph without you typing `aelf` at all; opt out per-hook via `aelf setup --no-transcript-ingest`, `--no-commit-ingest`, `--no-session-start`. See [INSTALL § default-on hooks](docs/INSTALL.md).
- **Determinism.** Stdlib + SQLite. No embeddings, no learned re-rankers, no LLM in the retrieval path. Every result traces to the action that wrote it.
- **Local-only.** SQLite at `<repo>/.git/aelfrice/memory.db`. No telemetry, no network calls, no accounts. Per-project isolation by construction. See [PRIVACY.md](docs/PRIVACY.md).
- **Removable.** `aelf uninstall --archive backup.aenc` encrypts the DB to a file, then deletes it. Or `--purge` for a full wipe.

Tradeoff: no fuzzy semantic recall. See [PHILOSOPHY.md](docs/PHILOSOPHY.md).

---

## Day-to-day surface

After `aelf setup` you should rarely type `aelf` again. The day-to-day commands are six:

```
aelf onboard .                      # once per project — scan and ingest
aelf lock "never push to main"      # add a permanent rule
aelf locked                          # see what rules are active
aelf search "push to main"           # check what the agent will see
aelf status                          # quick health summary
aelf setup / aelf doctor            # initial install + verification
```

Everything else (deeper diagnostics, archive/uninstall, migration tools, hook entry-points called by Claude Code itself) is callable but not something you reach for in normal use. `aelf --help` shows the everyday surface; `aelf --help --advanced` lists the rest. Full reference: [COMMANDS](docs/COMMANDS.md).

The same operations are also available as MCP tools and `/aelf:*` slash commands — same library underneath. See [MCP](docs/MCP.md) and [SLASH_COMMANDS](docs/SLASH_COMMANDS.md).

---

## Status

Latest stable: **v3.0.0** — wonder lifecycle complete (#542) + wonder/reason agentmemory parity (#645); HRR persistence default-ON with split-format migration (#553); type-aware compression (#434); federation read-only mechanics (#650 umbrella: scope #688, `aelf promote/demote --to-scope` #689, peer-aware `aelf reason` walk #690, transport #655); eval-judge κ calibration (#687). v3.0 design decisions ratified: NL-relatedness philosophy (#605), sentiment-feedback hook (#606), multimodel scope (#607, deferred), federation write model (#661, read-only).

Next: TBD. Milestone tracker for v3.0: [#608](https://github.com/robotrocketscience/aelfrice/issues/608).

Per-version detail: [docs/ROADMAP.md](docs/ROADMAP.md).
Open issues / known limits: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

---

## Documentation

- **Getting started:** [Install](docs/INSTALL.md) · [Quickstart](docs/QUICKSTART.md)
- **Reference:** [Commands](docs/COMMANDS.md) · [MCP](docs/MCP.md) · [Slash commands](docs/SLASH_COMMANDS.md) · [Config](docs/CONFIG.md)
- **Background:** [Architecture](docs/ARCHITECTURE.md) · [Philosophy](docs/PHILOSOPHY.md) · [Privacy](docs/PRIVACY.md) · [Limitations](docs/LIMITATIONS.md)
- **Development:** [Releasing](docs/RELEASING.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## Citation

```bibtex
@software{aelfrice2026,
  author = {robotrocketscience},
  title  = {aelfrice: deterministic Bayesian memory for AI coding agents},
  year   = {2026},
  url    = {https://github.com/robotrocketscience/aelfrice},
  license = {MIT}
}
```

[MIT](LICENSE)
