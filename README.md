<p align="center"><img src="docs/assets/01-hero-kulili.png" width="100%" alt="A figure of shimmering cloud rising from a dark sea, weaving threads of light into a constellation of beliefs"></p>

# aelfrice

> Your AI stops forgetting.
> Set up once. Stays out of the way.
>
> _No cloud. No account. No telemetry._

[![PyPI](https://img.shields.io/pypi/v/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![Python](https://img.shields.io/pypi/pyversions/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![License](https://img.shields.io/pypi/l/aelfrice.svg)](LICENSE)
[![CI](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml/badge.svg)](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml)
[![OSSInsight](https://img.shields.io/badge/OSSInsight-analytics-blue)](https://ossinsight.io/analyze/robotrocketscience/aelfrice)
<!-- bench-canonical-badge:start -->
[![Reproducibility](https://img.shields.io/badge/reproducibility-partial%20%286%2F11%20adapters%29-yellow)](docs/v2_reproducibility_harness.md)
<!-- bench-canonical-badge:end -->

You correct your agent. *"Got it,"* it says. Next session, same mistake.

aelfrice runs in the background and stops the amnesia. Write a rule once and every relevant prompt thereafter ships with that rule already attached — *before* the model sees your message. No `CLAUDE.md` chain to maintain, no cross-references for the agent to skip; the matched beliefs are in the prompt, not in a file the agent is supposed to consult.

## Install

```bash
uv tool install aelfrice    # requires uv — https://docs.astral.sh/uv/
aelf setup                  # wire the UserPromptSubmit hook into your agent
aelf onboard .              # one-shot project scan: filesystem, git log, code structure
aelf lock "never push directly to main; use scripts/publish.sh"
```

That's it. The next prompt that mentions "push" already has the rule. From here on out aelfrice is invisible — no command to remember to run, no file to keep updated.

---

## What it does for you

- **Stops the AI forgetting your rules.** Lock a rule once with `aelf lock "..."` — it comes back attached to every prompt after that, in every new session. You don't have to remind the AI; aelfrice does.
- **The AI can't skip it.** What aelfrice remembers is in the prompt *before* the model sees it — not stored in a file the AI is supposed to check on its own. The reminding happens for the AI, not by you.
- **Stays on your computer.** One file on your machine. No cloud account. No telemetry. If you stop trusting aelfrice, `aelf uninstall` removes it cleanly in one command.
- **Set up once. Forget it's there.** Three commands at install time. After that, you reach for `aelf` rarely. The background hooks do the rest.

---

## How it works

In plain English: four searches run in parallel over your stored rules, the best matches get prepended to your prompt, and the model reads the lot as one message. Below is the wiring for readers who want the receipts.

When you submit a prompt, aelfrice's `UserPromptSubmit` hook fires before the model sees your message. It runs four retrieval lanes in parallel, merges the result, and prepends the matched beliefs as an `<aelfrice-memory>` block:

```text
L0: locked beliefs   -> rules you marked permanent (always returned, never trimmed)
L1: FTS5 keyword     -> SQLite full-text search, BM25 + posterior-weighted rerank
L2: graph walk       -> typed-edge BFS from L1 seeds (SUPPORTS, CONTRADICTS, SUPERSEDES, DERIVED_FROM, ...)
L2.5: structural HRR -> Plate-FFT bind/probe against anchor text + structural markers
```

L0 always ships. L1, L2, and L2.5 are budget-trimmed against the merged candidate set in score-descending order; locked beliefs win every overflow. Default budget: 2,400 tokens per prompt. The default ranking stack is `stack-r1-r3` (entity expansion + per-store IDF clipping); bench evidence on the labelled query-strategy corpus measured **+0.2851 absolute NDCG@k (+94.8%)** versus the v1.4 raw-BM25 baseline at p99 latency 4.5 ms. Full lane wiring, composition, and federation peer DBs: [ARCHITECTURE § Retrieval](docs/ARCHITECTURE.md#retrieval).

The result is prepended to your prompt verbatim:

```text
<aelfrice-memory>
[locked] never push directly to main; use scripts/publish.sh
[locked] commits must be SSH-signed with ~/.ssh/id_rrs
         the publish script runs gitleaks before tagging
</aelfrice-memory>

push the release
```

**Lock count is the operator's baseline-context budget knob.** If you lock 200 things, every session opens with all 200, by design. Everything non-locked is BM25-ranked and budget-trimmed. The first prompt of a new session carries one extra block — a `<session-start>` sub-block listing all locks plus load-bearing unlocked beliefs (corroboration ≥ 2, or posterior mean ≥ ⅔ with α+β ≥ 4); subsequent prompts in the same session skip it. One extra block per new session, not per prompt.

---

## Day-to-day

After `aelf setup` you rarely type `aelf` again. The everyday surface is six commands:

```text
aelf onboard .                      # once per project — scan and ingest
aelf lock "never push to main"      # add a permanent rule
aelf locked                          # see what rules are active
aelf search "push to main"           # check what the agent will see
aelf status                          # quick health summary
aelf setup / aelf doctor            # initial install + verification
```

`aelf --help` shows the everyday surface; `aelf --help --advanced` lists the rest. Full reference: [COMMANDS](docs/COMMANDS.md). The same operations are exposed as MCP tools and `/aelf:*` slash commands — same library underneath. See [MCP](docs/MCP.md) and [SLASH_COMMANDS](docs/SLASH_COMMANDS.md).

---

## Reasoning surfaces (v3.0)

Two slash commands let the agent reach back into the belief graph mid-turn, beyond the auto-injected retrieval block. They pair: `/aelf:wonder` grows the graph by researching; `/aelf:reason` walks the enriched graph for structured verdicts.

**`/aelf:wonder <topic>`** — the research surface. Given a topic, aelfrice runs gap analysis on what the store already knows, generates 2–6 orthogonal research axes (always-on `domain_research` + `internal_gap_analysis`; conditional `contradiction_resolution` / `uncertainty_deep_dive` / `coverage_extension`), fans out one subagent per axis to research and write up findings, then persists the merged research as new speculative beliefs via `wonder_ingest`. Those phantoms sit in the graph at low evidence — discoverable by retrieval and by the next `/aelf:reason <topic>` — until you promote them with `aelf promote` (or lock the underlying statement, which auto-promotes a matching phantom per [#550](https://github.com/robotrocketscience/aelfrice/issues/550)). Agent-count shorthand `quick 2-agent` / `deep 4-agent` is recognised in the query string.

**`/aelf:reason <query>`** — the structured-walk surface. Walks the belief graph from BM25-seeded starting points and emits a typed reasoning trace: hops with edge-type breadcrumbs, a `VERDICT` (`SUFFICIENT` / `INCOMPLETE` / `CONTRADICTED` / `IMPASSE`), `IMPASSES` (typed gaps, ties, or constraint failures), and `SUGGESTED UPDATES` — `(belief_id, direction, note)` rows that map straight to `aelf feedback` so the conclusion closes the loop on the beliefs that fed it. Each impasse is dispatched to a role-tagged subagent (Verifier / Gap-filler / Fork-resolver). Peer hops in foreign federation scopes are annotated `[scope:<name>]`.

The pair-rhythm is the point: `/aelf:wonder` adds fresh thinking to the graph, then `/aelf:reason` draws conclusions across it. Both surfaces are deterministic in the aelfrice layer (verdict classification, impasse derivation, axis generation, suggested-update mapping). The only LLM calls happen when the host agent dispatches a subagent per impasse or research axis — and those calls run under the host's own credentials, not aelfrice's. Specs: [COMMANDS § `wonder`](docs/COMMANDS.md), [COMMANDS § `reason`](docs/COMMANDS.md), [v3.0 wonder+reason parity (#645)](https://github.com/robotrocketscience/aelfrice/issues/645).

---

## Memory model

Every belief carries a `(α, β)` Beta-Bernoulli posterior: `α / (α+β)` is the confidence; `α + β` is how much evidence backs that confidence. New beliefs sit at low evidence (high variance, retrievable but discounted); locked beliefs short-circuit decay and pin as ground truth.

| You run | It stores |
|---|---|
| `aelf lock "never commit .env files"` | Permanent rule. Returned on every retrieval. |
| `aelf onboard .` | Walks the project — git log, prose headings, code structure — and ingests structural facts as `agent_inferred` beliefs. |
| `aelf feedback <id> used` | `α += 1`. Strengthens the belief's posterior. |
| `aelf feedback <id> harmful` | `β += 1`. Weakens it. Five independent harmfuls through `CONTRADICTS` edges to a lock auto-demote it. |
| `aelf promote <id>` | Flips `origin` from `agent_inferred` to `user_validated`. With `--to-scope <SCOPE>`, also moves federation visibility (`project` / `global` / `shared:<name>`). |
| `/aelf:wonder <topic>` | Researches the topic and writes the findings as `speculative` phantoms; `/aelf:reason <topic>` can then walk them. |
| _(passive — no command)_ | Default-on auto-capture: every prompt/response turn is logged and ingested at compaction; successful `git commit` events are ingested too. Opt out via `aelf setup --no-transcript-ingest --no-commit-ingest`. |

Each belief has an `origin` column tying it to the action that wrote it — one of `user_stated`, `user_corrected`, `user_validated`, `agent_inferred`, `agent_remembered`, `document_recent`, `speculative`, `unknown`. The store is a single SQLite file; open it in any browser, nothing is hidden.

---

## Why files alone don't solve this

The standard workaround for "agent keeps forgetting" is more files: `STATE.md`, `DECISIONS.md`, a `CLAUDE.md` with cross-references to runbooks. Every cross-reference is a bet that the agent will read the file, find the right section, and follow what it says.

The failure modes are predictable: the agent reads the rule and runs `git push` anyway; cross-references break silently after compaction; state files rot the moment someone forgets to update them. Each new failure mode begets another file.

aelfrice replaces the chain with a mechanism. Matched beliefs are in the prompt, prepended by the hook before the model sees your message. Not voluntary; nothing the agent can skip.

| Manual approach | What breaks | aelfrice |
|---|---|---|
| Rules in `CLAUDE.md` | Agent reads them; doesn't follow them | Injected per-prompt, not per-session |
| Cross-references | Agent skips or reads the wrong section | Matched beliefs injected directly |
| Hand-maintained state files | One missed update breaks the chain | State is the SQLite DB; no manual sync |

---

## Why this is a memory system, not a key-value store

[Leonard Lin's review of agentic-memory implementations](https://github.com/lhl/agentic-memory/blob/main/ANALYSIS.md) frames the bar bluntly:

> The biggest differentiator is not "vector DB vs SQLite" — it's **write correctness and governance**: provenance / audit trail, write gates / confirmation, conflict handling, reversibility (inspect / edit / delete).

By that bar, "a vector store with a similarity query" is not a memory system — it is a search index. A memory system has to answer *who wrote this, when, via what ingress, what supersedes it, and how do I take it back*. Here is how aelfrice v3.0 answers each.

| Lin's pillar | What it means | aelfrice mechanism (v3.0) |
|---|---|---|
| **Provenance / audit trail** | Every row traces back to the action that wrote it: who, when, via what ingress channel. | `origin` column on every belief — eight tier values including `speculative` for `/aelf:wonder` phantoms ([`src/aelfrice/models.py`](src/aelfrice/models.py)). v3.0 adds a `scope` column (`project` / `global` / `shared:<name>`, [#688](https://github.com/robotrocketscience/aelfrice/issues/688)) tagging federation visibility. Append-only [`ingest_log`](src/aelfrice/store.py) records every raw input — tear the DB down and rebuild from this log alone. Open the file in any SQLite browser; nothing hidden. |
| **Write gates / confirmation** | Persistence is not unconditional. Some writes need explicit approval; external-origin claims cannot be laundered into ground truth. | `aelf lock` is the only path to user-asserted ground truth. `aelf confirm` bumps the `(α, β)` posterior but cannot flip `origin`. Phantom promotion has two explicit surfaces (v3.0, [#550](https://github.com/robotrocketscience/aelfrice/issues/550)): `aelf promote <id>` for the explicit path, and `aelf lock <text>` with Jaccard ≥ 0.9 substring match for the implicit auto-promote — both write audit rows. Feedback accumulates rather than overwrites: one harmful click nudges the mean, it doesn't erase a belief. |
| **Conflict handling** | Competing claims about the same thing are surfaced, not silently overwritten. | First-class edge types `CONTRADICTS`, `SUPERSEDES`, `RESOLVES` — disagreement is a graph relation, not a vanished row. v3.0's `/aelf:reason` emits a typed `VERDICT` (`SUFFICIENT` / `INCOMPLETE` / `CONTRADICTED` / `IMPASSE`) plus typed `IMPASSES` (`TIE` / `GAP` / `CONSTRAINT_FAILURE` / `NO_CHANGE`) so a downstream agent can act on the disagreement ([#645](https://github.com/robotrocketscience/aelfrice/issues/645) classifiers, [#658](https://github.com/robotrocketscience/aelfrice/issues/658) `ConsequencePath` fork-on-`CONTRADICTS`). Per-scope version vectors preserve causal ordering across worktrees and federation peers. |
| **Reversibility (inspect / edit / delete)** | Mutations remain auditable and partially undoable. The user is the boss of their memories. | `aelf delete`, `aelf unlock`, `aelf promote --to-scope`, and `aelf feedback` all write audit rows; the `ingest_log` is append-only and replay-capable. Read-only federation (v3.0, [#650](https://github.com/robotrocketscience/aelfrice/issues/650) / [#655](https://github.com/robotrocketscience/aelfrice/issues/655)) lets a project surface peer beliefs via `knowledge_deps.json` without taking ownership — foreign-id mutations raise `ForeignBeliefError` at the API surface. Top level: `aelf uninstall --archive backup.aenc` encrypts and removes; `--purge` wipes; `--keep-db` leaves data untouched. No vendor lock-in. |

---

## What you get for free

Running in the background. No action required after `aelf setup`.

- **Passive capture.** Default-on transcript-ingest, commit-ingest, and session-start hooks. Session activity flows into the belief graph without you typing `aelf` at all; opt out per-hook via `aelf setup --no-transcript-ingest`, `--no-commit-ingest`, `--no-session-start`. See [INSTALL § default-on hooks](docs/INSTALL.md).
- **Determinism.** Stdlib + SQLite. No embeddings, no learned re-rankers, no LLM in the retrieval path. Every result traces to the action that wrote it.
- **Local-only.** SQLite at `<git-common-dir>/aelfrice/memory.db`. aelfrice itself makes no network calls and emits no telemetry; no accounts. (Subagent LLM dispatches in `/aelf:wonder` / `/aelf:reason` flows do reach the network — under the host agent's credentials, not aelfrice's. The retrieval path stays local.) Per-project isolation by construction. v3.0 ships *read-only* cross-project federation via `knowledge_deps.json` — peer DBs are opened read-only, foreign-id mutations are rejected at the API surface, no multi-writer extension. See [PRIVACY.md](docs/PRIVACY.md).
- **Removable.** `aelf uninstall --archive backup.aenc` encrypts the DB to a file, then deletes it. Or `--purge` for a full wipe.

---

## Status

Latest stable: **v3.0.1** (2026-05-13). Per-entry detail in [CHANGELOG § 3.0.1](CHANGELOG.md). Per-version history: [docs/ROADMAP.md](docs/ROADMAP.md). Known limits: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

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
