<p align="center"><img src="docs/assets/01-hero-kulili.png" width="100%" alt="A figure of shimmering cloud rising from a dark sea, weaving threads of light into a constellation of beliefs"></p>

# aelfrice

> Lock the rules your AI agent keeps forgetting. SQLite on your laptop. Auditable.

[![PyPI](https://img.shields.io/pypi/v/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![Python](https://img.shields.io/pypi/pyversions/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![License](https://img.shields.io/pypi/l/aelfrice.svg)](LICENSE)
[![CI](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml/badge.svg)](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml)

You correct your agent. *"Got it,"* it says. Next session, same mistake.

aelfrice gives the agent a memory you can lock down. You write the rule once. It gets injected into every prompt thereafter. No cross-references for the agent to skip, no markdown files to maintain.

```bash
pip install aelfrice
aelf onboard .
aelf lock "never push directly to main; use scripts/publish.sh"
aelf setup       # wire the UserPromptSubmit hook into Claude Code
```

Restart Claude Code. The next prompt that mentions "push" will already have your rule attached.

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

Default budget is 2,000 tokens per prompt. Locked beliefs always go first; the rest is BM25-ranked and truncated to fit.

---

## What it remembers

| You run | It stores |
|---|---|
| `aelf lock "never commit .env files"` | Permanent rule. Returned on every retrieval. |
| `aelf onboard .` | Walks the project — git log, README headings, code structure — and ingests structural facts. |
| `aelf feedback <id> used` | Bayesian feedback. Strengthens the belief's posterior. |
| `aelf feedback <id> harmful` | Weakens it. After enough independent harmfuls, locks auto-demote. |

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

## Determinism is the property

Every retrieval is reproducible bit-for-bit from the write log and the code. No embeddings, no learned re-rankers, no LLM in the retrieval path. *"Why did this rule surface for this query?"* has a finite answer that bottoms out in named beliefs created by named user actions at named timestamps.

That property is what makes aelfrice usable in regulated contexts. It's also what costs you fuzzy semantic recall — that's a deliberate trade. See [PHILOSOPHY.md](docs/PHILOSOPHY.md).

---

## Your data stays yours

- **100% local.** SQLite at `<repo>/.git/aelfrice/memory.db`. No network calls in the retrieval path.
- **No telemetry.** No accounts, no signup, no phone-home.
- **No GPU, no vector DB.** Stdlib + SQLite. The optional `[mcp]` extra adds `fastmcp`. That's it.
- **Per-project isolation.** Beliefs from project A cannot leak into project B (they live in different `.git/` directories).
- **Removable.** `aelf uninstall --archive backup.aenc` encrypts the DB to a file, then deletes it. Or `--purge` for a full wipe.

[docs/PRIVACY.md](docs/PRIVACY.md) for verifiable specifics.

---

## What's in the box

| Surface | Count | Where |
|---|---|---|
| CLI subcommands | 22 | `aelf <subcommand>` — see [COMMANDS](docs/COMMANDS.md) |
| MCP tools | 9 | called by the agent automatically — see [MCP](docs/MCP.md) |
| Slash commands | 22 | `/aelf:*` in Claude Code — see [SLASH_COMMANDS](docs/SLASH_COMMANDS.md) |
| Hook events | 4 | UserPromptSubmit, PreCompact, Stop, PostToolUse (opt-in) |

The CLI, MCP, and slash command surfaces are 1:1 wrappers over the same library. Anything you can do in one, you can do in the others.

---

## Roadmap

| Version | Status | Theme |
|---|---|---|
| v1.0.x | shipped | core memory, CLI, MCP, hook wiring, install routing |
| v1.1.0 | shipped | per-project DBs (`.git/aelfrice/`), `aelf migrate`, `edges`→`threads` rename, `aelf health` rewrite |
| v1.2.0 | shipped | auto-capture pipeline (transcript-ingest, commit-ingest, SessionStart), `agent_inferred → user_validated` promotion, triple extractor, `--batch` JSONL ingest, CLI consolidation, `INEDIBLE` per-file opt-out |
| v1.3 | planned | retrieval wave — entity index + BFS multi-hop + LLM classification |
| v2.0 | planned | feature parity with the original research line + benchmark reproducibility |

Per-version detail: [docs/ROADMAP.md](docs/ROADMAP.md). Open issues: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

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
