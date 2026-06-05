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

You correct your agent. *"Got it,"* it says. Next session, same mistake.

aelfrice runs in the background and stops the amnesia. Write a rule once and every relevant prompt thereafter ships with that rule already attached — *before* the model sees your message. No `CLAUDE.md` chain to maintain, no cross-references for the agent to skip; the matched beliefs are in the prompt, not in a file the agent is supposed to consult.

**For developers** using AI coding agents — first-class via the `UserPromptSubmit` hook for hosts that expose it; any MCP host via the included stdio server. Local-only by design — embeddings, vector RAG, and cloud sync are explicitly out of scope. See [Philosophy](docs/concepts/PHILOSOPHY.md) for the trade-off.

## Install

```bash
uv tool install aelfrice    # requires uv — https://docs.astral.sh/uv/
aelf setup                  # wire the UserPromptSubmit hook into your agent
aelf onboard .              # one-shot project scan: filesystem, git log, code structure
aelf lock "never push directly to main; use scripts/publish.sh"
```

That's it. The next prompt that mentions "push" already has the rule. From here on out aelfrice is invisible — no command to remember to run, no file to keep updated.

---

## What you'll see

You type a message in your agent. aelfrice's hook fires before the model sees it and prepends matched beliefs as an `<aelfrice-memory>` block:

```text
<aelfrice-memory>
[locked] never push directly to main; use scripts/publish.sh
[locked] commits must be SSH-signed with ~/.ssh/id_ed25519
         the publish script runs gitleaks before tagging
</aelfrice-memory>

push the release
```

The model reads the whole thing as one message. Your rules arrive every relevant time, not when the agent decides to check a file.

---

## What it does for you

- **Stops the AI forgetting your rules.** Lock a rule once with `aelf lock "..."` — it comes back attached to every prompt after that, in every new session. You don't have to remind the AI; aelfrice does.
- **The AI can't skip it.** What aelfrice remembers is in the prompt *before* the model sees it — not stored in a file the AI is supposed to check on its own. The reminding happens for the AI, not by you.
- **Stays on your computer.** One file on your machine. No cloud account. No telemetry. If you stop trusting aelfrice, `aelf uninstall` removes it cleanly in one command.
- **Set up once. Forget it's there.** Three commands at install time. After that, you reach for `aelf` rarely. The background hooks do the rest.

---

## Day-to-day

After `aelf setup` you rarely type `aelf` again. The everyday surface:

```text
aelf onboard .                      # once per project — scan and ingest
aelf lock "never push to main"      # add a permanent rule
aelf locked                          # see what rules are active
aelf search "push to main"           # check what the agent will see
aelf status                          # quick health summary
aelf setup / aelf doctor            # initial install + verification
aelf feed                            # read the belief-write event log (v3.4+)
aelf stale --older-than 90 --cold-for 30   # surface forgotten beliefs (v3.4+)
aelf review --generate               # weekly keep / remove / lock checkpoint (v3.5+)
```

`aelf --help` shows the everyday surface; `aelf --help --advanced` lists the rest. Full reference: [COMMANDS](docs/user/COMMANDS.md). The same operations are exposed as MCP tools and `/aelf:*` slash commands — same library underneath. See [MCP](docs/user/MCP.md) and [SLASH_COMMANDS](docs/user/SLASH_COMMANDS.md).

---

## How it works

Four retrieval lanes run in parallel, the best matches get prepended to your prompt, and the model reads the lot as one message:

```text
L0: locked beliefs   -> rules you marked permanent (always returned, never trimmed)
L1: FTS5 keyword     -> SQLite full-text search, BM25 + posterior-weighted rerank
L2: graph walk       -> typed-edge BFS from L1 seeds (SUPPORTS, CONTRADICTS, SUPERSEDES, DERIVED_FROM, ...)
L2.5: structural HRR -> Plate-FFT bind/probe against anchor text + structural markers
```

L0 always ships. L1, L2, and L2.5 are budget-trimmed against the merged candidate set in score-descending order; locked beliefs win every overflow. Default budget: 2,400 tokens per prompt.

**Lock count is the operator's baseline-context budget knob.** If you lock 200 things, every session opens with all 200, by design. Everything non-locked is BM25-ranked and budget-trimmed. The first prompt of a new session carries one extra block — a `<session-start>` sub-block listing all locks plus load-bearing unlocked beliefs (corroboration ≥ 2, or posterior mean ≥ ⅔ with α+β ≥ 4); subsequent prompts in the same session skip it.

Bench evidence on the labelled query-strategy corpus measured **+0.2851 absolute NDCG@k (+94.8%)** versus the v1.4 raw-BM25 baseline at +0.96 ms p99 over legacy-bm25 (v3.0 30-row corpus, 2026-05-12, gate budget +5 ms delta; see [`tests/bench_gate/test_query_strategy.py`](tests/bench_gate/test_query_strategy.py)). Full lane wiring, composition, and federation peer DBs: [ARCHITECTURE § Retrieval](docs/concepts/ARCHITECTURE.md#retrieval).

---

## Reasoning surfaces

Two slash commands let the agent reach back into the belief graph mid-turn, beyond the auto-injected retrieval block. They pair: `/aelf:wonder` grows the graph by researching; `/aelf:reason` walks the enriched graph for structured verdicts.

**`/aelf:wonder <topic>`** — the research surface. Given a topic, aelfrice runs gap analysis on what the store already knows, generates 2–6 orthogonal research axes (always-on `domain_research` + `internal_gap_analysis`; conditional `contradiction_resolution` / `uncertainty_deep_dive` / `coverage_extension`), fans out one subagent per axis to research and write up findings, then persists the merged research as new speculative beliefs via `wonder_ingest`. Those phantoms sit in the graph at low evidence — discoverable by retrieval and by the next `/aelf:reason <topic>` — until you promote them with `aelf promote` (or lock the underlying statement, which auto-promotes a matching phantom). Agent-count shorthand `quick 2-agent` / `deep 4-agent` is recognised in the query string.

**`/aelf:reason <query>`** — the structured-walk surface. Walks the belief graph from BM25-seeded starting points and emits a typed reasoning trace: hops with edge-type breadcrumbs, a `VERDICT` (`SUFFICIENT` / `INCOMPLETE` / `CONTRADICTED` / `IMPASSE`), `IMPASSES` (typed gaps, ties, or constraint failures), and `SUGGESTED UPDATES` — `(belief_id, direction, note)` rows that map straight to `aelf feedback` so the conclusion closes the loop on the beliefs that fed it. Each impasse is dispatched to a role-tagged subagent (Verifier / Gap-filler / Fork-resolver). Peer hops in foreign federation scopes are annotated `[scope:<name>]`.

The pair-rhythm is the point: `/aelf:wonder` adds fresh thinking to the graph, then `/aelf:reason` draws conclusions across it. Both surfaces are deterministic in the aelfrice layer (verdict classification, impasse derivation, axis generation, suggested-update mapping). The only LLM calls happen when the host agent dispatches a subagent per impasse or research axis — and those calls run under the host's own credentials, not aelfrice's. Specs: [COMMANDS § `wonder`](docs/user/COMMANDS.md), [COMMANDS § `reason`](docs/user/COMMANDS.md).

---

## Memory model

Every belief carries a `(α, β)` Beta-Bernoulli posterior: `α / (α+β)` is the confidence; `α + β` is how much evidence backs that confidence. New beliefs sit at low evidence (high variance, retrievable but discounted); locked beliefs short-circuit decay and pin as ground truth.

| You run | It stores |
|---|---|
| `aelf lock "never commit .env files"` | Permanent rule. Returned on every retrieval. |
| `aelf onboard .` | Walks the project — git log, prose headings, code structure — and ingests structural facts as `agent_inferred` beliefs. |
| `aelf feedback <id> used` | `α += 1`. Strengthens the belief's posterior. |
| `aelf feedback <id> harmful` | `β += 1`. Weakens it. Locks resist passive feedback by design — change with `aelf unlock` / `aelf delete`. |
| `aelf promote <id>` | Flips `origin` from `agent_inferred` to `user_validated`. With `--to-scope <SCOPE>`, also moves federation visibility (`project` / `global` / `shared:<name>`). |
| `/aelf:wonder <topic>` | Researches the topic and writes the findings as `speculative` phantoms; `/aelf:reason <topic>` can then walk them. |
| _(passive — no command)_ | Default-on auto-capture: every prompt/response turn is logged and ingested at compaction; successful `git commit` events are ingested too. Opt out per-hook via `aelf setup --no-transcript-ingest`, `--no-commit-ingest`, `--no-session-start`, `--no-stop-hook`, `--no-sessionstart-recap`, `--no-search-tool`, `--no-search-tool-bash`, `--no-pre-issue-guard` — see [INSTALL § default-on hooks](docs/user/INSTALL.md). |

Each belief has an `origin` column tying it to the action that wrote it — one of `user_stated`, `user_corrected`, `user_validated`, `user_transcript`, `agent_inferred`, `agent_remembered`, `document_recent`, `speculative`, `unknown`. The store is a single SQLite file; open it in any browser, nothing is hidden.

---

## Obsidian export

If you already live in Obsidian, `aelf export-obsidian <vault-path>` emits the belief graph as one Markdown note per belief under `<vault>/aelfrice/`. Typed edges land in YAML front-matter for [Dataview](https://blacksmithgu.github.io/obsidian-dataview/); the same edges appear in the note body as wikilinks so the graph view has something to draw. The export is **one-way (DB → vault)**: SQLite stays the source of truth, and the `<vault>/aelfrice/` subdirectory is wiped and rewritten on each run.

Scopes: `--scope all` (everything, capped by `--max-notes`), `--scope recent` (newest first), `--scope query "<text>"` (BM25 seeds + N-hop neighbourhood). Default cap is 500 notes; the hard ceiling is 5000 unless `--force` is passed.

Two structural limits ship with the feature, not as later documentation drift:

> **Obsidian's built-in graph view does not scale.** Force-directed layout chokes around a few thousand nodes; every zoom or hover re-runs layout. Do not export your full belief store and expect the graph view to be usable. Use `--scope query` or `--max-notes` to bound the export, or use `aelf graph` for query-anchored visualization that works at any store size.
>
> **Obsidian's graph view is untyped.** Wikilinks are wikilinks; the aelfrice edge types are not distinguishable in the graph. Edge types are preserved in YAML front-matter and queryable via the [Dataview plugin](https://blacksmithgu.github.io/obsidian-dataview/), but graph view will not show them. This is an Obsidian limitation, not an export bug.

---

## What you get for free

Running in the background. No action required after `aelf setup`.

- **Passive capture.** Eight default-on hooks: `UserPromptSubmit` retrieval, four-event transcript-ingest, `PostToolUse:Bash` commit-ingest, `SessionStart` locked-belief injection, `SessionStart` belief-write recap (v3.5+), `Stop` lock-prompt, `PreToolUse:Grep|Glob` and `PreToolUse:Bash` memory-first search, and the `PreToolUse:Bash` issue-dup guard. Session activity flows into the belief graph without you typing `aelf` at all; opt out per-hook via `aelf setup --no-transcript-ingest`, `--no-commit-ingest`, `--no-session-start`, `--no-stop-hook`, `--no-sessionstart-recap`, `--no-search-tool`, `--no-search-tool-bash`, `--no-pre-issue-guard`. See [INSTALL § default-on hooks](docs/user/INSTALL.md).
- **Determinism.** Stdlib + SQLite. No embeddings, no learned re-rankers, no LLM in the retrieval path. Every result traces to the action that wrote it.
- **Local-only.** SQLite at `<git-common-dir>/aelfrice/memory.db`. aelfrice itself makes no network calls and emits no telemetry; no accounts. (Subagent LLM dispatches in `/aelf:wonder` / `/aelf:reason` flows do reach the network — under the host agent's credentials, not aelfrice's. The retrieval path stays local.) Per-project isolation by construction. Read-only cross-project federation via `knowledge_deps.json` — peer DBs are opened read-only, foreign-id mutations are rejected at the API surface. See [PRIVACY.md](docs/user/PRIVACY.md).
- **Removable.** `aelf uninstall --archive backup.aenc` encrypts the DB to a file, then deletes it. Or `--purge` for a full wipe.

---

## A memory system, not a key-value store

By [Leonard Lin's bar](https://github.com/lhl/agentic-memory/blob/main/ANALYSIS.md), "a vector store with a similarity query" is not a memory system — it is a search index. A memory system has to answer *who wrote this, when, via what ingress, what supersedes it, and how do I take it back.* aelfrice meets the four pillars (provenance, write gates, conflict handling, reversibility) directly. The full breakdown lives in [COMPARISON.md](docs/concepts/COMPARISON.md), including the side-by-side against `CLAUDE.md` and hand-maintained state files.

---

## Status

Latest stable: **v3.5.0** (2026-06-04). Per-entry detail in [CHANGELOG § 3.5.0](CHANGELOG.md). Per-version history: [docs/concepts/ROADMAP.md](docs/concepts/ROADMAP.md). Known limits: [docs/user/LIMITATIONS.md](docs/user/LIMITATIONS.md).

[![OSSInsight](https://img.shields.io/badge/OSSInsight-analytics-blue)](https://ossinsight.io/analyze/robotrocketscience/aelfrice)
<!-- bench-canonical-badge:start -->
[![Reproducibility](https://img.shields.io/badge/reproducibility-partial%20%286%2F11%20adapters%29-yellow)](docs/design/v2_reproducibility_harness.md)
<!-- bench-canonical-badge:end -->

---

## Documentation

- **Getting started:** [Install](docs/user/INSTALL.md) · [Quickstart](docs/user/QUICKSTART.md)
- **Reference:** [Commands](docs/user/COMMANDS.md) · [MCP](docs/user/MCP.md) · [Slash commands](docs/user/SLASH_COMMANDS.md) · [Config](docs/user/CONFIG.md)
- **Background:** [Architecture](docs/concepts/ARCHITECTURE.md) · [Philosophy](docs/concepts/PHILOSOPHY.md) · [Comparison](docs/concepts/COMPARISON.md) · [Privacy](docs/user/PRIVACY.md) · [Limitations](docs/user/LIMITATIONS.md)
- **Development:** [Releasing](docs/concepts/RELEASING.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

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
