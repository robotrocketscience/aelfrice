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

## What's likely to land in v1.x

- **Posterior in retrieval ranking.** Today `apply_feedback` updates `(α, β)` but ranking is BM25-only. Wiring posterior into ordering is the precondition for using the bench harness to claim feedback drives accuracy.
- **Per-project DB defaults.** Path-keyed databases so per-project memory is automatic, surviving worktrees / clones / moves.
- **Onboard noise filtering.** Drop markdown headings, checklist items, license boilerplate, near-duplicates before insertion.
- **Benchmarks against LongMemEval.** Move beyond the synthetic 16-belief × 16-query harness.
- **Cross-project read-only mode.** Port from the legacy v2.0 codebase.
- **The `wonder` and `reason` commands.** Reintroduced with evidence.

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
