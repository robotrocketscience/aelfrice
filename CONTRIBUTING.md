# Contributing

Thanks for considering a contribution. aelfrice is a one-author project — the bar for changes is "is the system better afterward, in a way that's defensible by tests."

## Status

v1.0 has shipped. Issues are welcome. PRs are evaluated on a case-by-case basis — the bar is "moves the system measurably forward, justifies the change with a test."

Best categories of PR:

- Bug fixes with a regression test that fails before and passes after.
- Doc fixes (typo, broken link, stale claim against current code).
- Closing one of the [known issues at v1.0](docs/LIMITATIONS.md#known-issues-at-v10) — but ping in an issue first to align on approach.

Hard to land without prior alignment:

- New CLI subcommands or MCP tools.
- Schema changes.
- Anything that adds a hard dependency.
- Reintroducing v2.0 features without a benchmark / experiment showing the impact.

## How to file a useful issue

Title format: a one-line description, in lowercase, ending without a period.

```
search: locked-belief order is unstable when budget is exhausted
```

Body should include:

- **What happened** — exact CLI invocation or MCP call, exact output.
- **What you expected** — one-line.
- **Environment** — OS, Python version (`python --version`), aelfrice version, MCP host (if relevant).
- **A minimal repro.** A directory you can `aelf onboard <here>` and reproduce, or the smallest sequence of CLI calls that triggers it.

Don't include `~/.aelfrice/memory.db` from a real project — it contains your private beliefs. Reproduce on a scratch DB (`AELFRICE_DB=/tmp/scratch.db`) and share that.

## What's likely to land where

The v1.x roadmap is bucketed. See [LIMITATIONS](docs/LIMITATIONS.md#known-issues-at-v10) for each issue's target version.

- **v1.0.1** — launch fix-up. Hook → `aelfrice.retrieval.retrieve()` rewrite + `feedback_history` recording (highest-impact gap). `aelf --version` flag. Onboard noise filters. CONTRADICTS auto-supersession. Onboard performance regression baseline.
- **v1.1.0** — project identity (`.git/aelfrice/memory.db`, `.aelfrice.toml`, orphan-DB cleanup, worktree concurrency). Onboard behavior (git-recency weighting, `agent_inferred` → `user_validated` promotion). Cosmetic surface (edges→threads, split `aelf status` from `aelf health`).
- **v1.2.0** — commit-ingest PostToolUse hook for automatic capture. Full hook performance audit. Seed files for git-tracked knowledge bootstrapping.
- **v1.3** — retrieval wave. HRR / vocabulary bridging. LLM-Haiku classification on the onboard path (~$0.005/session, opt-in). Cross-project knowledge federation. Posterior consumed in ranking. 6–9 weeks estimated.
- **v2.0** — full academic benchmark suite (LoCoMo, MAB, LongMemEval, StructMemEval, AmaBench) + the v2-line feature surface (`wonder`, `reason`, `core`, snapshot/timeline tools).

Highest-leverage contributions right now are tied to **v1.0.1** — the launch fix-up — because they unblock the "feedback-driven memory" claim end-to-end.

## What's not on the path

- Vector embeddings or ANN in retrieval (would require a hard dep on a vector library; defeats the local-stdlib design).
- Cloud sync, accounts, or any non-local data path.
- A web UI.
- Integration with chat platforms outside MCP.

## Development setup

Once contributions are open:

```bash
git clone https://github.com/robotrocketscience/aelfrice.git
cd aelfrice
uv sync --all-groups
uv run pytest tests/ -x -q
uv run pyright src/
```

Conventions:

- Conventional-commit prefixes: `feat:`, `fix:`, `perf:`, `refactor:`, `test:`, `docs:`, `build:`, `ci:`, `style:`, `revert:`, `chore:`.
- Atomic commits. Each commit moves the tree from one tested green state to another.
- Tests required for every behavioral change.
- `pyright --strict` must pass.
- Conventional commit messages plus the Co-Authored-By footer if pair-coded.

We use a pre-push hook (`scripts/check-commit-msg.py`) that enforces the prefix list. Don't `--no-verify`.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). The short version: be respectful, focus on the work, no harassment.

## Security

See [SECURITY.md](SECURITY.md). Privacy bugs are treated as security bugs.
