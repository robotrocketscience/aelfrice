<p align="center"><img src="docs/assets/01-hero-kulili.png" width="100%" alt="A figure of shimmering cloud rising from a dark sea, weaving threads of light into a constellation of beliefs"></p>

# aelfrice

[![PyPI](https://img.shields.io/pypi/v/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![Python](https://img.shields.io/pypi/pyversions/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![License](https://img.shields.io/pypi/l/aelfrice.svg)](LICENSE)
[![CI](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml/badge.svg)](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml)
<!-- bench-canonical-badge:start -->
reproducibility: ⚠️ 6/11 ok · last run 2026-10-23
<!-- bench-canonical-badge:end -->

**aelfrice gives an AI coding agent a local memory: before the model reads each prompt, a hook adds your locked beliefs and the best-ranked matches to it.** An agent forgets your corrections between sessions, so you repeat them.

- aelfrice stores your turns and commits as beliefs; a belief becomes ground truth only when you lock it with `/aelf:lock`.
- Retrieval uses an entity index and BM25 over local SQLite and needs no account, telemetry, embeddings, or LLM.

By default, aelfrice checks PyPI for updates and runs your `claude` CLI in the background to label new core beliefs, and `/aelf:onboard` classifies with your agent's model. See the [privacy document](docs/user/PRIVACY.md).

## Beliefs

A belief is one statement with its source, type, and Bayesian confidence score. Every 12 logged lines (about six prompts and replies), aelfrice stores each sentence of your prompts as a fact, correction, preference, or requirement. It skips noise, questions, and the model's replies.

`/aelf:confirm` and `aelf feedback` change a belief's confidence, as does praise such as "perfect" for an answer it informed. Complaints such as "that's wrong" count only with [negative sentiment feedback](docs/user/CONFIG.md) on.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/how-it-works-dark.png">
    <img src="docs/assets/how-it-works-light.png" width="100%" alt="How aelfrice works. At session start the context window holds the host's system prompt and tool definitions, CLAUDE.md, and an aelfrice-baseline block of your locked rules. On each turn, three paths run: the prompt hook, and a search hook that fires whenever the model searches, query the store (locks always, the entity index, and BM25), ranks locks first, then entity matches, then BM25 hits weighted by confidence, and adds an aelfrice-memory block to what the model reads; a typed /aelf:lock writes a locked belief that is injected at every session start and every prompt; and after each reply the Stop hook logs the turn, and every 12 turns your sentences are ingested as beliefs, each linked to the one before by a TEMPORAL_NEXT edge. The ranking follows each prompt: two example prompts return different orders. All three paths read or write one local SQLite belief graph.">
  </picture>
</p>

## Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install aelfrice:

   ```bash
   uv tool install aelfrice
   ```

3. Connect aelfrice to your agent:

   ```bash
   aelf setup
   ```

   On Codex, run `aelf setup --host codex` instead, then trust the hooks with `/hooks`. Codex uses `$aelf-*` in place of `/aelf:`.

4. In your agent, scan your project from its directory:

   ```text
   /aelf:onboard .
   ```

5. Lock your first belief:

   ```text
   /aelf:lock never push directly to main
   ```

aelfrice then runs without further commands. `/aelf:doctor` checks the install (`aelf doctor --host codex` on Codex). The [installation guide](docs/user/INSTALL.md) covers every option.

<p align="center"><img src="docs/assets/retrieval-lanes.png" width="88%" alt="An illustrative schematic of the retrieval lanes of aelfrice over a belief graph: locked beliefs pinned at the query, keyword seeds from full-text search spreading outward, an optional typed-edge graph walk, and a structural lane for marker queries."></p>

## Commands

| Command | Does | Use when |
|---|---|---|
| `/aelf:lock <text>` | Locks a belief as ground truth for every prompt. | Must always hold. |
| `/aelf:search <query>` | Lists matches, locks first. A hook also runs it on the agent's Grep, Glob, WebSearch, WebFetch, and Bash `grep` or `rg` calls and appends the results. | Checking a topic. |
| `/aelf:locked` | Lists locked beliefs. | Reviewing locks. |
| `/aelf:onboard <path>` | Scans a project and stores what it learns. | New project. |
| `/aelf:status` | Shows counts of beliefs, locks, and feedback. | Health check. |
| `/aelf:doctor` | Checks hooks and store; exits nonzero on failure. | After setup, or no injection. |
| `/aelf:unlock <id>` | Removes a belief's lock. | Lock no longer applies. |
| `/aelf:confirm <id>` | Raises a belief's confidence without locking it. | Proved right, but not a rule to lock. |
| `/aelf:review` | Writes a keep, remove, or lock checklist, then applies your verdicts. | Weekly cleanup. |
| `/aelf:retire <id>` | Hides a belief from retrieval until `/aelf:restore`. | Wrong or outdated. |
| `/aelf:wonder <topic>` | Stores research on a topic as speculative beliefs. | The store has a gap. |
| `/aelf:reason <query>` | Walks the graph and reports a verdict and suggested updates. | A structured answer. |

<details>
<summary>More commands</summary>

| Command | Does | Use when |
|---|---|---|
| `/aelf:setup` | Installs the hooks into your agent. | Install, or broken hooks. |
| `/aelf:show <id>` | Prints one belief in full with its origin and confidence. | Inspecting one belief. |
| `/aelf:core` | Lists locked, recurring, and high-confidence beliefs. | Seeing what the store relies on. |
| `/aelf:feed` | Shows the belief-write log. | Auditing recent writes. |
| `/aelf:tail` | Streams hook injections per prompt and session start. | Checking injections. |
| `/aelf:stale` | Lists old beliefs not retrieved recently. | Pruning. |
| `/aelf:restore <id>` | Brings back a retired belief. | Mistaken retire. |
| `/aelf:delete <id>` | Deletes a belief permanently. | Retire is not enough. |
| `/aelf:promote <id>` | Marks an agent-inferred belief as user-validated. | The agent got it right. |
| `/aelf:speculative` | Lists unlocked beliefs, most-supported first. | Auditing unlocked beliefs. |
| `/aelf:scope-out <text>` | Hides matching beliefs for this session. | Off-task this session. |
| `/aelf:category` | Binds belief groups to trigger keywords; injection is off by default. | Task-specific beliefs. |
| `/aelf:graph <id>` | Emits the graph around a belief as DOT or JSON. | Visualizing links. |
| `/aelf:introspect` | Shows beliefs by session or project, with confidence and grounding. | Judging captures. |
| `/aelf:rebuild` | Prints the context block the rebuilder would restore. | Debugging context after compaction. |
| `/aelf:audit-claude-memory` | Compares locked beliefs with your agent's `MEMORY.md` index. | Keeping both consistent. |
| `/aelf:eval` | Runs the relevance-calibration harness on a `--corpus`. | Measuring retrieval quality. |
| `/aelf:upgrade` | Upgrades to the latest release. | A new version is out. |
| `/aelf:uninstall` | Removes the hooks and keeps, deletes, or archives the store. `uv tool uninstall aelfrice` removes the package. | Removing aelfrice. |

</details>

The [command reference](docs/user/COMMANDS.md) lists the terminal form of each command, such as `aelf lock "..."`.

## Belief types and edges

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/belief-graph-dark.png">
    <img src="docs/assets/belief-graph-light.png" width="100%" alt="An illustrative belief graph of eight beliefs, colored by type. A locked factual belief, never push directly to main, has no edges: a lock is injected whether or not anything links to it. A requirement, the release checks must include pyright, is DERIVED_FROM a factual belief, the publish script runs the release checks. A factual belief from a commit, publish.sh runs pytest and pyright, SUPPORTS the requirement and IMPLEMENTS the publish-script belief. A correction, do not deploy staging from main and use the release branch, is TEMPORAL_NEXT after the publish-script belief and SUPERSEDES an older factual belief, staging deploys from main. A preference, I prefer small atomic commits, is TEMPORAL_NEXT after the correction. A speculative belief, drawn as a hollow outline, cache the wheel build between releases, RELATES_TO the publish-script belief.">
  </picture>
</p>

Every belief has one type. A rule-based classifier types transcript text; `/aelf:onboard` uses your agent's model by default. Text from `/aelf:lock` or the commit hook is `factual`. Locking does not change a belief's type.

| Type | Holds | Assigned when |
|---|---|---|
| `factual` | A statement about the project or world. | No other rule matches. |
| `requirement` | A rule that has to hold. | Contains "must", "required", or similar. |
| `correction` | A fix to something said earlier. | The correction detector matches it. |
| `preference` | How you like things done. | Contains "I prefer", "always use", or similar. |
| `speculative` | A phantom: a guess that isn't trusted yet. | `/aelf:wonder` writes it. |

Edges link a source belief to a target belief. Today, only `TEMPORAL_NEXT` and `DERIVED_FROM` are written for every conversation. The commit hook reads the agent's `git commit` messages and matches only unambiguous phrasings: "is supported by" counts, and "supports" doesn't.

| Edge | Meaning | Written by | Default |
|---|---|---|---|
| `TEMPORAL_NEXT` | Stored right after the target in the same session. | Transcript ingest, for each new belief; commit hook ("comes after", "is after", "succeeds"). | Yes |
| `DERIVED_FROM` | Follows from the target: the previous turn, or an earlier clause in the same turn. | Transcript ingest; commit hook ("is derived from", "is based on"). | Yes |
| `RELATES_TO` | Same topic as the target. | `/aelf:wonder`; commit hook ("relates to", "is related to"). | When run, or on phrases |
| `SUPPORTS` | Evidence for the target. | Commit hook ("is supported by"). | Phrase only |
| `CITES` | Mentions the target. | Commit hook ("cites", "mentions"). | Phrases only |
| `IMPLEMENTS` | Code implementing the target. | Commit hook ("implements", "is an implementation of", "realizes", "fulfills"). | Phrases only |
| `TESTS` | Tests the target. | Commit hook ("is a test for", "is test of", "is tested by", "is covered by"). | Phrases only |
| `CONTRADICTS` | Conflicts with the target. | Ingest relationship detector; commit hook ("contradicts", "disagrees with"). | Detector: off unless `[relationship_detector] auto_detect = true`. Commit hook: phrases only |
| `SUPERSEDES` | Replaces the target. | `aelf resolve` (keeps each contradicting pair's winner); commit hook ("supersedes"). | When run, or on phrase |
| `POTENTIALLY_STALE` | Flags an older target as possibly stale. | `/aelf:doctor --detect-stale`. | When run |
| `RESOLVES` | A contradicting pair's winner resolves the loser. | `aelf resolve`, beside each `SUPERSEDES` edge ([issue #1658](https://github.com/robotrocketscience/aelfrice/issues/1658)). | When run |

[Issue #1653](https://github.com/robotrocketscience/aelfrice/issues/1653) tracks writing more edges automatically.

<p align="center"><img src="docs/assets/02-eterne-hrr.png" width="88%" alt="A pen-and-ink figure holding a sword, with ranks of armored figures branching above and behind him like a tree"></p>

## Documentation

- [Quickstart](docs/user/QUICKSTART.md)
- [Architecture](docs/concepts/ARCHITECTURE.md)
- [Configuration](docs/user/CONFIG.md)
- [Philosophy](docs/concepts/PHILOSOPHY.md)
- [Comparison](docs/concepts/COMPARISON.md)
- [Privacy](docs/user/PRIVACY.md)
- [Limitations](docs/user/LIMITATIONS.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)

[MIT license](LICENSE)
