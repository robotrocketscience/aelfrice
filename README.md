<p align="center"><img src="docs/assets/01-hero-kulili.png" width="100%" alt="A figure of shimmering cloud rising from a dark sea, weaving threads of light into a constellation of beliefs"></p>

# aelfrice

> Your AI agent stops forgetting.
> Set up aelfrice one time. aelfrice then does not interrupt your work.
>
> _No cloud. No account. No telemetry._

[![PyPI](https://img.shields.io/pypi/v/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![Python](https://img.shields.io/pypi/pyversions/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![License](https://img.shields.io/pypi/l/aelfrice.svg)](LICENSE)
[![CI](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml/badge.svg)](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml)

You correct your agent. *"Got it,"* the agent says. In the next session, the agent makes the same mistake.

aelfrice runs in the background. aelfrice stops this loss of memory. Write a rule one time. Each relevant prompt after that carries the rule. The rule is attached *before* the model reads your message. There is no rules file to maintain. There is nothing for the agent to skip, because the matched beliefs are in the prompt itself.

aelfrice is for developers who use AI coding agents. A host that supplies a `UserPromptSubmit` hook gets full support. A tool cannot put the correct beliefs in front of the model before the model reads your message, because the model decides whether to call the tool. The hook therefore makes the guarantee possible. aelfrice is local-only by design. Embeddings, vector retrieval-augmented generation (RAG) and cloud synchronization are outside the scope. [Philosophy](docs/concepts/PHILOSOPHY.md) explains why that trade-off is worth it.

## Install

```bash
uv tool install aelfrice    # requires uv — https://docs.astral.sh/uv/
aelf setup                  # wire the UserPromptSubmit hook into your agent
aelf onboard .              # deterministic project scan (regex classifier). For LLM-quality with no API key, run /aelf:onboard in your agent.
aelf lock "never push directly to main; use scripts/publish.sh"
```

The setup is complete. The next prompt that mentions "push" already carries the rule. After this, aelfrice does not ask for your attention. There is no command to remember. There is no file to keep current.

Do you use the Codex CLI? Run `aelf setup --host codex`. This command installs the same set of hooks into the `hooks.json` file of `$CODEX_HOME` or `~/.codex`. The command also installs the `/aelf:*` command bundle as `$aelf-*` agent skills (v4.1.0+). For more data, read [INSTALL § Codex host](docs/user/INSTALL.md).

## What you'll see

You type a message in your agent. The hook of aelfrice operates before the model reads the message. The hook injects the matched beliefs at the start of the message, in an `<aelfrice-memory>` block:

```text
<aelfrice-memory>
The following are retrieved beliefs from the local memory store. ...
<belief id="a1f3c2d0" lock="user">never push directly to main; use scripts/publish.sh</belief>
<belief id="91e02d3c" lock="user">commits must be SSH-signed with ~/.ssh/id_ed25519</belief>
<belief id="77c01b2a">the publish script runs the release checks before tagging</belief>
</aelfrice-memory>

push the release
```

The model reads all of this as one message. Your rules arrive at each relevant prompt. The rules do not arrive only when the agent decides to read a file.

## What it does for you

Lock a rule one time with `aelf lock "..."`. The rule then returns attached to each relevant prompt, in each later session. aelfrice does the reminding for you. The model cannot skip the rule, because the rule is already in the prompt when the model starts to read. The rule is not in a file that the model can decide not to read.

There is also nothing to maintain. Passive capture logs each turn. Passive capture then ingests each turn. Passive capture also ingests the message of each successful `git commit`. The memory thus grows while you work. You do not have to type `aelf` to make this happen.

All of the data stays on your computer. The data is one SQLite file. There is no cloud account and no telemetry. If you stop trusting aelfrice, `aelf uninstall` removes aelfrice with one command. The `--archive` option encrypts the database to a file first.

## Why not just a rules file?

A rules file is advice that the agent *may* read. aelfrice is context that the model *has already read*. By [Leonard Lin's standard](https://github.com/lhl/agentic-memory/blob/main/ANALYSIS.md), "a vector store with a similarity query" is also not a memory system. A memory system has to answer these questions: *who wrote this, when, via what ingress, what supersedes it, and how do I take it back.* aelfrice meets the four pillars directly. The four pillars are provenance, write gates, conflict handling and reversibility. [COMPARISON.md](docs/concepts/COMPARISON.md) gives the comparison against hand-maintained rules files and vector stores.

## Day-to-day

You rarely type `aelf` again after you run `aelf setup`. The commands for everyday use follow:

```text
aelf onboard .                      # once per project — deterministic scan (or /aelf:onboard for the no-key subagent flow)
aelf lock "never push to main"      # add a permanent rule
aelf locked                          # see what rules are active
aelf search "push to main"           # check what the agent will see
aelf status                          # quick health summary
aelf setup / aelf doctor            # initial install + verification
aelf feed                            # read the belief-write event log (v3.5+)
aelf stale --older-than 90 --cold-for 30   # surface forgotten beliefs (v3.5+)
aelf review --generate               # weekly keep / remove / lock checkpoint (v3.5+)
```

The `aelf --help` command shows the commands for everyday use. The `aelf --help --advanced` command lists the other commands. [COMMANDS](docs/user/COMMANDS.md) is the full reference. aelfrice supplies the same operations as `/aelf:*` slash commands. The slash commands use the same library. Read [SLASH_COMMANDS](docs/user/SLASH_COMMANDS.md).

## How it works

Three retrieval lanes run on each prompt. The fourth lane, the breadth-first search (BFS) graph expansion, runs only when you enable it. aelfrice injects the best matches at the start of your prompt. The model reads all of this as one message.

```text
L0: locked beliefs   -> rules you marked permanent (always returned, never trimmed)
L2.5: entity index   -> deterministic NER-extracted entity lookup, exact + stem match
L1: FTS5 keyword     -> SQLite full-text search, BM25 + posterior-weighted rerank
L3: graph walk       -> typed-edge BFS from the L0+L2.5+L1 seed set (DERIVED_FROM, CONTRADICTS,
                        SUPERSEDES, RELATES_TO, ...) — opt-in: [retrieval] bfs_enabled = true
```

<p align="center"><img src="docs/assets/retrieval-lanes.png" width="88%" alt="An illustrative schematic of the retrieval lanes of aelfrice over a belief graph. The L0 locked beliefs are pinned at the query. The L1 keyword seeds from full-text search version 5 (FTS5) and Best Matching 25 (BM25) spread outward. When you enable the L3 lane, the typed-edge graph walk of L3 moves outward one hop at a time. The bridges of the structural holographic reduced representation (HRR) connect to matches that share no vocabulary with the query."></p>

<p align="center"><sub><i>The figure is illustrative. The figure is not a trace of a real store. The L0 locked rules always return. A query on FTS5 and BM25 seeds L1. When you enable the L3 lane, the L3 graph walk steps along typed edges one hop at a time. The separate lane for the structural HRR (<code>retrieve_v2</code>) connects to matches that share no vocabulary with the query. The color gives the lane. The distance from the center gives the depth of the graph walk. The figure omits the L2.5 entity-index lane, to keep the figure legible. <a href="docs/assets/render_retrieval_lanes.py">render_retrieval_lanes.py</a> rendered the figure.</i></sub></p>

aelfrice always returns L0. When you enable L3, aelfrice trims L1, L2.5 and L3 to the budget. Otherwise aelfrice trims L1 and L2.5. The trim runs against the merged candidate set in order of descending score. The locked beliefs win each overflow. The default budget is 1,500 tokens for each prompt that the hook injects into. The default for `aelf search` and for the library function `retrieve()` is 2,400 tokens. A separate structural-HRR lane uses the Plate-FFT bind and probe operations. This lane receives the queries that parse as structural markers in the `retrieve_v2` API. Ordinary prompts never use this lane.

Your count of locks is also the budget for the baseline context. If you lock 200 statements, each session opens with all 200 statements, by design. aelfrice ranks each unlocked belief with BM25. aelfrice then trims the unlocked beliefs to the budget. The first prompt of a new session carries one extra block. That block is a `<session-start>` sub-block. The sub-block lists all of the locks. The sub-block also lists the load-bearing unlocked beliefs. A load-bearing unlocked belief has a corroboration ≥ 2, or a posterior mean ≥ ⅔ with α+β ≥ 4. The later prompts in the same session skip the sub-block.

The query that reaches BM25 is the raw prompt. A `stack-r1-r3` rewriter was the default from v3.0. That rewriter does entity expansion and per-store clipping of the inverse document frequency (IDF). The default came from a measurement of +0.2851 absolute normalized discounted cumulative gain at k (NDCG@k) on a labelled corpus. Issue #1177 replaced the conjunctive FTS5 match with a disjunction over the rarest tokens. That conjunctive match caused the drop in recall. The rewriter existed to compensate for that drop. Issue #1501 then reverted the rewriter, because the drop in recall was gone. On the same 30 rows the raw-query arm scored 0.9553, against 0.8229 for the rewriter. You can still select the rewriter with `[rebuilder] query_strategy`. **Both figures come from a labelled corpus that is not shipped in
this repository**, so you cannot reproduce either figure from a public clone. The gate in this repository is
[`tests/bench_gate/test_query_strategy.py`](tests/bench_gate/test_query_strategy.py). Without
`AELFRICE_CORPUS_ROOT` the gate skips. When the gate does run, it asserts that the shipped default is the winning arm. The gate does not check a
quoted number. For figures that you can reproduce on HEAD, read the scripts under [`benchmarks/`](benchmarks/). [ARCHITECTURE § Retrieval](docs/concepts/ARCHITECTURE.md#retrieval) gives the full wiring of the lanes, the composition and the federation peer databases.

## Memory model

Each belief carries a `(α, β)` Beta-Bernoulli posterior. The value `α / (α+β)` is the confidence. The value `α + β` is the quantity of evidence that backs that confidence. A new belief starts at low evidence and high variance. aelfrice can retrieve such a belief, but aelfrice discounts it. A locked belief does not decay. aelfrice pins a locked belief as ground truth.

| You run | aelfrice stores |
|---|---|
| `aelf lock "never commit .env files"` | A permanent rule. aelfrice returns it on every retrieval. |
| `aelf onboard .` | The `aelf onboard` command walks the project. The command reads the git log, the headings in the prose and the structure of the code. The command then ingests the structural facts as `agent_inferred` beliefs with the deterministic regular-expression classifier. |
| `/aelf:onboard` | The same scan, with a higher-quality classification. In-session subagents drive the classification. No API key and no billing are necessary. `/aelf:onboard` is the preferred path in an agent. The `aelf onboard` command alone is the deterministic fallback. |
| `aelf feedback <id> used` | `α += 1`. The command strengthens the posterior of the belief. |
| `aelf feedback <id> harmful` | `β += 1`. The command weakens the posterior of the belief. A lock resists passive feedback, by design. To change a lock, run `aelf unlock` or `aelf delete`. |
| `aelf promote <id>` | The `aelf promote` command flips `origin` from `agent_inferred` to `user_validated`. With `--to-scope <SCOPE>`, the command also moves the federation visibility to `project`, `global` or `shared:<name>`. |
| `/aelf:wonder <topic>` | The `/aelf:wonder` command researches the topic. The command writes the findings as `speculative` phantoms. The `/aelf:reason <topic>` command can then walk those phantoms. |
| _(passive — no command)_ | Passive capture is on by default. aelfrice logs each turn of prompt and response. aelfrice ingests each turn at compaction. aelfrice also ingests each successful `git commit` event. To opt out of one hook, use one of these options: `aelf setup --no-transcript-ingest`, `--no-commit-ingest`, `--no-session-start`, `--no-stop-hook`, `--no-sessionstart-recap`, `--no-search-tool`, `--no-search-tool-bash`, `--no-pre-issue-guard`, `--no-claude-memory-mirror`, `--no-agent-context`. For more data, read [INSTALL § default-on hooks](docs/user/INSTALL.md). |

Each belief has an `origin` column. That column ties the belief to the action that wrote it. The value is one of `user_stated`, `user_corrected`, `user_validated`, `user_transcript`, `agent_inferred`, `agent_remembered`, `document_recent`, `speculative` or `unknown`. The store is a single SQLite file. Open the file in any browser. Nothing is hidden.

## Reasoning surfaces

Two slash commands let the agent query the belief graph during a turn. The two commands go beyond the retrieval block that aelfrice injects automatically. The two commands operate together. `/aelf:wonder` grows the graph by researching. `/aelf:reason` walks the enriched graph for structured verdicts.

**`/aelf:wonder <topic>`** is the research surface. You give a topic. aelfrice then runs a gap analysis on what the store already knows. aelfrice generates 2–6 orthogonal research axes. The axes `domain_research` and `internal_gap_analysis` are always on. The axes `contradiction_resolution`, `uncertainty_deep_dive` and `coverage_extension` are conditional. The host agent then dispatches one research task for each axis. Each task researches the axis and writes up the findings. aelfrice then persists the merged research as new speculative beliefs with `wonder_ingest`. Those phantoms sit in the graph at low evidence. Retrieval can discover them. The next `/aelf:reason <topic>` command can also discover them. The phantoms stay speculative until you promote them with `aelf promote`. If you lock the statement behind a phantom, aelfrice promotes the matching phantom automatically. aelfrice also recognises an agent-count shorthand in the query string, for example `quick 2-agent` or `deep 4-agent`. The integer sets the agent count. The words `quick` and `deep` are optional qualifiers.

**`/aelf:reason <query>`** is the structured-walk surface. The command walks the belief graph from starting points that BM25 seeds. The command emits a typed reasoning trace. The trace holds the hops. Each hop carries its edge type. The trace holds a `VERDICT`. The verdict is `SUFFICIENT`, `PARTIAL`, `UNCERTAIN`, `INSUFFICIENT` or `CONTRADICTORY`. The trace holds `IMPASSES`, which are typed gaps, ties or constraint failures. The trace holds `SUGGESTED UPDATES`. Each suggested update is a `(belief_id, direction, note)` row. The fields map straight onto `aelf feedback`. The conclusion therefore closes the loop on the beliefs that fed it. The host agent dispatches each impasse to a role-tagged worker. The roles are Verifier, Gap-filler and Fork-resolver. aelfrice annotates a peer hop in a foreign federation scope with `[scope:<name>]`.

Use the two surfaces in that sequence. `/aelf:wonder` adds new research results to the graph. `/aelf:reason` then draws conclusions across the graph. Both surfaces are deterministic in the aelfrice layer. The deterministic parts are the verdict classification, the impasse derivation, the axis generation and the suggested-update mapping. The only calls to a large language model (LLM) happen when the host agent dispatches one worker for each impasse or research axis. Those calls run under the credentials of the host, not the credentials of aelfrice. The specifications are [COMMANDS § `wonder`](docs/user/COMMANDS.md) and [COMMANDS § `reason`](docs/user/COMMANDS.md).

## What you get for free

These functions run in the background. No action is required after `aelf setup`.

- **Passive capture.** Ten hooks are on by default. The hooks are:
  - the `UserPromptSubmit` retrieval,
  - the four-event transcript ingest,
  - the `PostToolUse:Bash` commit ingest,
  - the `SessionStart` injection of the locked beliefs (with a recap line for the belief writes, v3.5+),
  - the `Stop` lock prompt,
  - the `PreToolUse:Grep|Glob` memory-first search,
  - the `PreToolUse:Bash` memory-first search,
  - the `PreToolUse:Bash` pre-issue duplicate guard,
  - the `PostToolUse:Write|Edit|MultiEdit` claude-memory mirror (v3.7+),
  - the `PreToolUse:Agent` injection of the worker context.

  The claude-memory mirror operates as follows. Since v4.0, in #1089, the one-shot reconcile at the first `aelf setup` records the consent for each project. The mirror runs from then on. To opt out of the mirror at any time, set `AELFRICE_MIRROR_CLAUDE_MEMORY=0` or `[memory] mirror_claude_memory = false`. Both of these always win over the consent sentinel.

  The `PreToolUse:Agent` hook injects the worker context. A dispatched worker inherits the locked beliefs and the task-relevant beliefs. To opt out of the worker-context injection, use `--no-agent-context`. The session activity flows into the belief graph. You do not have to type `aelf` to make this happen. To opt out of one hook, read [INSTALL § default-on hooks](docs/user/INSTALL.md).
- **Determinism.** aelfrice uses SQLite and a deterministic numeric stack. That stack is numpy, scipy and snowballstemmer. It uses no GPU and no network. aelfrice uses no embeddings, no learned re-rankers and no LLM in the retrieval path. Every result traces to the action that wrote it.
- **Local-only.** aelfrice keeps the SQLite file at `<git-common-dir>/aelfrice/memory.db`. Two outbound calls are on by default. The first call is the update notifier. The notifier makes a read-only GET request to `https://pypi.org/pypi/aelfrice/json` under a time-to-live (TTL) gate. The notifier transmits nothing. To disable the notifier, set `AELF_NO_UPDATE_CHECK=1`. The second call is the pre-issue duplicate guard. That guard runs `gh issue list --search` with tokens from your issue title. The guard runs only when you run `gh issue create`. To disable the guard, set `AELFRICE_NO_PRE_ISSUE_GUARD=1` or run `aelf setup --no-pre-issue-guard`. There is no telemetry and there are no accounts. The path for memory and retrieval never touches the network. The LLM dispatches in the `/aelf:wonder` and `/aelf:reason` flows do reach the network. Those dispatches run under the credentials of the host agent, not the credentials of aelfrice. The retrieval path stays local. Each project is isolated by construction. Cross-project federation is read-only. Federation uses `knowledge_deps.json`. aelfrice opens a peer database read-only. aelfrice rejects a mutation of a foreign identifier at the API surface. Read [PRIVACY.md](docs/user/PRIVACY.md).
- **Removable.** The `aelf uninstall --archive backup.aenc` command encrypts the database to a file. The command then deletes the database. The `--purge` option removes all of the data.

## Obsidian export

If you already use Obsidian, run `aelf export-obsidian <vault-path>`. This command emits the belief graph as one Markdown note for each belief, under `<vault>/aelfrice/`. The command puts the typed edges into the YAML front matter for [Dataview](https://blacksmithgu.github.io/obsidian-dataview/). The command also writes the same edges into the body of the note as wikilinks, so the graph view has something to draw. The export is **one-way (DB → vault)**. SQLite stays the source of truth. The command deletes the `<vault>/aelfrice/` subdirectory. The command writes the subdirectory again on each run.

The command has three scopes. `--scope all` exports everything, with the cap that `--max-notes` sets. `--scope recent` exports the newest beliefs first. `--scope query "<text>"` exports the BM25 seeds and the neighbourhood at N hops. The default cap is 500 notes. The hard ceiling is 5000 notes, unless you pass `--force`.

> The feature ships with two structural limits. First, the built-in graph view of Obsidian does not scale. The graph view becomes too slow to use at approximately a few thousand nodes. Bound the export with `--scope query` or `--max-notes`. As an alternative, use `aelf graph` for a query-anchored visualization at any store size. Second, the graph view is untyped. aelfrice preserves the edge types in the YAML front matter. You can query the edge types with Dataview. The graph view will not show the edge types.

## Status

The latest stable version is **v4.3.0** (2026-08-13). [CHANGELOG § 4.3.0](CHANGELOG/v4.md) gives the detail for each entry. [docs/concepts/ROADMAP.md](docs/concepts/ROADMAP.md) gives the history for each version. [docs/user/LIMITATIONS.md](docs/user/LIMITATIONS.md) gives the known limits.

[![OSSInsight](https://img.shields.io/badge/OSSInsight-analytics-blue)](https://ossinsight.io/analyze/robotrocketscience/aelfrice)
<!-- bench-canonical-badge:start -->
[![Reproducibility](https://img.shields.io/badge/reproducibility-partial%20%286%2F11%20adapters%29-yellow)](docs/design/v2_reproducibility_harness.md)
<!-- bench-canonical-badge:end -->

## Documentation

- **Getting started:** [Install](docs/user/INSTALL.md) · [Quickstart](docs/user/QUICKSTART.md)
- **Reference:** [Commands](docs/user/COMMANDS.md) · [Slash commands](docs/user/SLASH_COMMANDS.md) · [Config](docs/user/CONFIG.md)
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
