<p align="center"><img src="docs/assets/01-hero-kulili.png" width="100%" alt="A figure of shimmering cloud rising from a dark sea, weaving threads of light into a constellation of beliefs"></p>

# aelfrice

> Persistent memory for AI agents.
> Set up once. Stays out of your way.
>
> _Local SQLite. Auditable. No GPU, no network._

[![PyPI](https://img.shields.io/pypi/v/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![Python](https://img.shields.io/pypi/pyversions/aelfrice.svg)](https://pypi.org/project/aelfrice/)
[![License](https://img.shields.io/pypi/l/aelfrice.svg)](LICENSE)
[![CI](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml/badge.svg)](https://github.com/robotrocketscience/aelfrice/actions/workflows/ci.yml)

You correct your agent. *"Got it,"* it says. Next session, same mistake.

aelfrice runs in the background and stops the forgetting. You write a rule once and it gets attached to every prompt thereafter — no cross-references for the agent to skip, no markdown files to maintain, nothing to remember to do.

```bash
pip install aelfrice
aelf onboard .
aelf lock "never push directly to main; use scripts/publish.sh"
aelf setup       # wire the hook into Claude Code
```

That's it. Restart Claude Code and your next prompt that mentions "push" already has the rule attached. From here on out aelfrice is invisible — no command to remember to run, no file to keep updated.

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

## Determinism

Same store + same query gives the same beliefs. The retrieval path is stdlib + SQLite — no embeddings, no learned re-rankers, no LLM — so every result traces back to a specific belief and the user action that wrote it.

Tradeoff: no fuzzy semantic recall. See [PHILOSOPHY.md](docs/PHILOSOPHY.md).

---

## Your data stays yours

- **100% local.** SQLite at `<repo>/.git/aelfrice/memory.db`. No network calls in the retrieval path.
- **No telemetry.** No accounts, no signup, no phone-home.
- **No GPU, no vector DB.** Stdlib + SQLite. The optional `[mcp]` extra adds `fastmcp`. That's it.
- **Per-project isolation.** Beliefs from project A cannot leak into project B (they live in different `.git/` directories).
- **Removable.** `aelf uninstall --archive backup.aenc` encrypts the DB to a file, then deletes it. Or `--purge` for a full wipe.

[docs/PRIVACY.md](docs/PRIVACY.md) for verifiable specifics.

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

## Roadmap

| Version | Status | Theme |
|---|---|---|
| v1.0.x | shipped | core memory, CLI, MCP, hook wiring, install routing |
| v1.1.0 | shipped | per-project DBs (`.git/aelfrice/`), `aelf migrate`, `edges`→`threads` rename, `aelf health` rewrite |
| v1.2.0 | shipped | auto-capture pipeline (transcript-ingest, commit-ingest, SessionStart), `agent_inferred → user_validated` promotion, triple extractor, `--batch` JSONL ingest, CLI consolidation, `INEDIBLE` per-file opt-out |
| v1.2.x | planned | search-tool `PreToolUse` hook — memory-first context on Grep/Glob |
| v1.3 | shipped | retrieval wave — entity index (L2.5), BFS multi-hop (L3), LLM-Haiku onboard classifier (opt-in), partial Bayesian-weighted ranking |
| v1.4 | shipped | context rebuilder — PreCompact retrieval-curated continuation (augment mode); manual + threshold trigger; continuation-fidelity scorer (exact-match) |
| v2.0 | planned | feature parity with the original research line + benchmark reproducibility. v2.0's component issues (#148–#154) will land incrementally across v1.5+ minor versions; final v2.0 tag is the reproducibility cut. |

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
