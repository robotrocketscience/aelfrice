<p align="center"><img src="docs/assets/01-hero-kulili.png" width="100%" alt="A figure of shimmering cloud rising from a dark sea, weaving threads of light into a constellation of beliefs"></p>

# aelfrice

[![PyPI](https://img.shields.io/pypi/v/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![Python](https://img.shields.io/pypi/pyversions/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![License](https://img.shields.io/pypi/l/aelfrice.svg)](LICENSE)
[![CI](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml/badge.svg)](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml)
<!-- bench-canonical-badge:start -->
[![Reproducibility](https://img.shields.io/badge/reproducibility-partial%20%286%2F11%20adapters%29-yellow)](docs/design/v2_reproducibility_harness.md)
<!-- bench-canonical-badge:end -->

**aelfrice gives an AI coding agent a local memory: the rules you lock, and the beliefs that match what you ask, reach the model before it reads your message.**

- **Problem.** An agent forgets your corrections between sessions, so you repeat them.
- **Method.** Before the model reads each prompt, a hook adds every rule you locked and the beliefs that best match the prompt, from an entity index and BM25 full-text search over a local SQLite store.
- **Provenance.** Every belief records where it came from and carries a Bayesian confidence that your feedback moves; a lock pins it as ground truth.
- **Scope.** aelfrice captures your turns and commits as you work, and retrieval runs on your machine with no account, telemetry, embeddings, or LLM. By default it also checks PyPI for updates, and `/aelf:onboard` classifies with your agent's model ([privacy](docs/user/PRIVACY.md)).

**How a conversation becomes beliefs.** A belief is one statement stored with its source, a type, and a confidence score. Every 12 turns, aelfrice splits your prompts into sentences, drops noise and questions, and stores the rest as facts, corrections, preferences, or requirements. The model's replies to you are skipped. When a later prompt matches stored beliefs, the best matches are ranked and added to it. They're beliefs, not facts: `/aelf:confirm` and `aelf feedback` move each one's confidence, and if you turn on [sentiment feedback](docs/user/CONFIG.md), so do short replies such as "perfect" or "that's wrong". Only a lock makes a belief ground truth.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/how-it-works-dark.png">
    <img src="docs/assets/how-it-works-light.png" width="100%" alt="How aelfrice works. At session start the context window holds the host's system prompt and tool definitions, CLAUDE.md, and an aelfrice-baseline block of your locked rules. On each turn, three paths run: the prompt hook, and a search hook that fires whenever the model searches, query the store (locks always, the entity index, and BM25), ranks locks first, then entity matches, then BM25 hits weighted by confidence, and adds an aelfrice-memory block to what the model reads; a typed /aelf:lock writes a locked belief that is injected at every session start and every prompt; and after each reply the Stop hook logs the turn, and every 12 turns your sentences are ingested as beliefs, each linked to the one before by a TEMPORAL_NEXT edge. The ranking follows each prompt: two example prompts return different orders. All three paths read or write one local SQLite belief graph.">
  </picture>
</p>

## Install

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install aelfrice:

   ```bash
   uv tool install aelfrice
   ```

3. Connect it to your agent:

   ```bash
   aelf setup
   ```

   If your agent is the Codex CLI, run `aelf setup --host codex` instead, then run `/hooks` in Codex to trust the new hooks. Codex uses `$aelf-*` skills where this page says `/aelf:`.

4. In your agent, from your project directory, scan the project:

   ```text
   /aelf:onboard .
   ```

5. Lock your first rule:

   ```text
   /aelf:lock never push directly to main
   ```

From then on, aelfrice runs by itself. To check the install, run `/aelf:doctor` (`aelf doctor --host codex` on Codex). The [installation guide](docs/user/INSTALL.md) covers every option.

<p align="center"><img src="docs/assets/retrieval-lanes.png" width="88%" alt="An illustrative schematic of the retrieval lanes of aelfrice over a belief graph: locked beliefs pinned at the query, keyword seeds from full-text search spreading outward, an optional typed-edge graph walk, and a structural lane for marker queries."></p>

## Commands

| Command | What it does | When to use it |
|---|---|---|
| `/aelf:lock <text>` | Locks a statement as ground truth that every prompt carries. | You want a rule the agent never forgets. |
| `/aelf:search <query>` | Shows what the store returns for a query, locked rules first; a hook also runs it on the agent's own searches (Grep, Glob, WebSearch, WebFetch, and `grep` or `rg` in Bash) and returns the results next to the tool's output. | You want to check what memory holds on a topic. |
| `/aelf:locked` | Lists your locked rules. | You want to review the rules in force. |
| `/aelf:onboard <path>` | Scans a project and stores what it learns. | You start using aelfrice on a project. |
| `/aelf:status` | Shows counts of beliefs, locks, and feedback. | You want a quick health summary. |
| `/aelf:doctor` | Checks the hooks and the store, and exits nonzero on a failure. | You finish setup, or memory seems silent. |
| `/aelf:unlock <id>` | Removes the lock from a belief. | A rule no longer applies. |
| `/aelf:confirm <id>` | Raises a belief's confidence without locking it. | A belief proved right, but it isn't a rule. |
| `/aelf:review` | Writes a keep, remove, or lock checklist, then applies your verdicts. | You do a weekly cleanup. |
| `/aelf:retire <id>` | Removes a belief from retrieval, reversibly. | A belief is wrong or outdated. |
| `/aelf:wonder <topic>` | Researches a topic and stores the findings as speculative beliefs. | You want memory to fill a gap. |
| `/aelf:reason <query>` | Walks the belief graph and reports a verdict and suggested updates. | You want a structured answer from memory. |

<details>
<summary>The other 19 commands</summary>

| Command | What it does | When to use it |
|---|---|---|
| `/aelf:setup` | Installs the aelfrice hooks into your agent. | You install aelfrice, or repair its hooks. |
| `/aelf:show <id>` | Prints one belief in full with its origin and confidence. | You want the whole of one belief. |
| `/aelf:core` | Lists the load-bearing beliefs: locks, repeated ones, and high-confidence ones. | You want to see what anchors your memory. |
| `/aelf:feed` | Shows the log of belief writes. | You want to see what aelfrice recorded. |
| `/aelf:tail` | Streams what the hooks injected on each prompt and session start. | You want to see what the agent was actually told. |
| `/aelf:stale` | Lists old beliefs that haven't been retrieved recently. | You prune memory that went cold. |
| `/aelf:restore <id>` | Brings a retired belief back. | You retired something by mistake. |
| `/aelf:delete <id>` | Deletes a belief permanently. | A belief must be gone for good. |
| `/aelf:promote <id>` | Marks an agent-inferred belief as validated by you. | You confirm what the agent worked out. |
| `/aelf:speculative` | Lists unlocked beliefs, the most-supported first. | You want to audit what isn't locked. |
| `/aelf:scope-out <text>` | Hides matching beliefs for the rest of this session. | A belief is noise for the task at hand. |
| `/aelf:category` | Groups rules and binds them to trigger keywords; injection is off by default. | A set of rules applies to one kind of task. |
| `/aelf:graph <id>` | Emits the belief graph around a belief as DOT or JSON. | You want to visualize connections. |
| `/aelf:introspect` | Shows beliefs by session or project with their confidence and grounding. | You want to judge what was captured. |
| `/aelf:rebuild` | Prints the context block the rebuilder would restore. | You're debugging context after compaction. |
| `/aelf:audit-claude-memory` | Compares locked beliefs with the `MEMORY.md` index your agent keeps. | You keep both and want them consistent. |
| `/aelf:eval` | Runs the relevance-calibration harness on a corpus you pass with `--corpus`. | You're measuring retrieval quality. |
| `/aelf:upgrade` | Upgrades aelfrice to the latest release. | A new version is out. |
| `/aelf:uninstall` | Removes the hooks, and keeps, deletes, or archives the store as you choose; `uv tool uninstall aelfrice` then removes the package. | You stop using aelfrice. |

</details>

Each command has a terminal form, such as `aelf lock "..."`. The [command reference](docs/user/COMMANDS.md) documents all of them.

## Belief types and edges

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/belief-graph-dark.png">
    <img src="docs/assets/belief-graph-light.png" width="100%" alt="An illustrative belief graph of eight beliefs, colored by type. A locked factual belief, never push directly to main, has no edges: a lock is injected whether or not anything links to it. A requirement, the release checks must include pyright, is DERIVED_FROM a factual belief, the publish script runs the release checks. A factual belief from a commit, publish.sh runs pytest and pyright, SUPPORTS the requirement and IMPLEMENTS the publish-script belief. A correction, do not deploy staging from main and use the release branch, is TEMPORAL_NEXT after the publish-script belief and SUPERSEDES an older factual belief, staging deploys from main. A preference, I prefer small atomic commits, is TEMPORAL_NEXT after the correction. A speculative belief, drawn as a hollow outline, cache the wheel build between releases, RELATES_TO the publish-script belief.">
  </picture>
</p>

Every belief has one type. Transcript text gets its type from a rule-based classifier, and `/aelf:onboard` uses your agent's model by default. New text from `aelf lock` or the commit hook is stored as `factual`, and `/aelf:wonder` stores `speculative`. Locked is a flag on top of the type, not a type of its own.

| Type | What it holds | How a belief gets it |
|---|---|---|
| `factual` | A statement about the project or the world. | The default, when no other rule matches. |
| `requirement` | A rule that has to hold. | It contains a word such as "must" or "required". |
| `correction` | A fix to something said earlier. | The correction detector matches it. |
| `preference` | How you like things done. | It contains a phrase such as "I prefer" or "always use". |
| `speculative` | A guess that isn't trusted yet, called a phantom. | `/aelf:wonder` writes it. |

Edges link beliefs, from a source to a target. Only the first two are written for every conversation today. The others need a command, an opt-in setting, or a specific phrase in a commit message. The commit hook reads a message when the agent runs `git commit`, and it matches only unambiguous phrasings: "is supported by" counts, but "supports" doesn't.

| Edge | Meaning | Written by | On by default |
|---|---|---|---|
| `TEMPORAL_NEXT` | Stored right after the target, in the same session. | Transcript ingest, for each new belief; the commit hook ("comes after", "is after", "succeeds"). | Yes |
| `DERIVED_FROM` | Follows from the target: the previous turn, or an earlier clause in the same turn. | Transcript ingest; the commit hook ("is derived from", "is based on"). | Yes |
| `RELATES_TO` | About the same topic as the target. | `/aelf:wonder`; the commit hook ("relates to", "is related to"). | When you run it, or on those phrases |
| `SUPPORTS` | Evidence for the target. | The commit hook ("is supported by"). | On that phrase only |
| `CITES` | Mentions the target. | The commit hook ("cites", "mentions"). | On those phrases only |
| `IMPLEMENTS` | Code that implements the target. | The commit hook ("implements", "is an implementation of", "realizes", "fulfills"). | On those phrases only |
| `TESTS` | A test of the target. | The commit hook ("is a test for", "is test of", "is tested by", "is covered by"). | On those phrases only |
| `CONTRADICTS` | Conflicts with the target. | The relationship detector at ingest; the commit hook ("contradicts", "disagrees with"). | Detector: no, set `[relationship_detector] auto_detect = true`. Commit hook: on those phrases |
| `SUPERSEDES` | Replaces the target. | `aelf resolve`, which keeps the winner of each contradicting pair; the commit hook ("supersedes"). | When you run it, or on that phrase |
| `POTENTIALLY_STALE` | Marks the target, an older belief, as possibly out of date. | `aelf doctor --detect-stale`. | When you run it |
| `RESOLVES` | A phantom answers the target. | Nothing writes it yet ([#1658](https://github.com/robotrocketscience/aelfrice/issues/1658)). | No |

Work to write more of these edges automatically is tracked in [#1653](https://github.com/robotrocketscience/aelfrice/issues/1653).

<p align="center"><img src="docs/assets/02-eterne-hrr.png" width="88%" alt="A pen-and-ink figure holding a sword, with ranks of armored figures branching above and behind him like a tree"></p>

## Learn more

[Quickstart](docs/user/QUICKSTART.md) · [Architecture](docs/concepts/ARCHITECTURE.md) · [Configuration](docs/user/CONFIG.md) · [Philosophy](docs/concepts/PHILOSOPHY.md) · [Comparison](docs/concepts/COMPARISON.md) · [Privacy](docs/user/PRIVACY.md) · [Limitations](docs/user/LIMITATIONS.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

[MIT license](LICENSE)
