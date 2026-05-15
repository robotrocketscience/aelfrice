# Contributing

Thanks for considering a contribution. aelfrice is a one-author project — the bar for changes is "is the system better afterward, in a way that's defensible by tests."

## Status

v1.0 has shipped. Issues are welcome. PRs are evaluated on a case-by-case basis — the bar is "moves the system measurably forward, justifies the change with a test."

Best categories of PR:

- Bug fixes with a regression test that fails before and passes after.
- Doc fixes (typo, broken link, stale claim against current code).
- Closing one of the [known issues at v1.0](docs/user/LIMITATIONS.md#known-issues-at-v10) — but ping in an issue first to align on approach.

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

## Triage labels

The issue tracker uses a small label vocabulary that gh-driven tooling
(`aelf-scan.sh`, `aelf-claim.sh`; see `src/aelfrice/cli.py` and
`src/aelfrice/gate_list.py`) reads to decide whether an open issue is
ready to be claimed or should be hidden from the active queue. Apply
one or more of these at file-time when an issue isn't immediately
actionable:

- **`gate:operator`** — operator decision or operator-side data must
  arrive before the issue can move. Example: a tracker that opens once
  enough telemetry has accumulated to baseline against (#749, #488).
- **`gate:prereq`** — blocked on another tracked work item (sub-task,
  upstream dependency, framework landing first).
- **`gate:lab-corpus`** — blocked on lab-side corpus delivery; the
  public-tree work cannot exercise its acceptance criteria until the
  corpus is committed.
- **`gate:ratify`** — needs ratification of a design decision before
  implementation should begin.
- **`gate:umbrella`** — umbrella issue that coordinates sub-issues but
  has no implementation surface of its own; closes when its children
  close.
- **`bench-gated`** — implementation has shipped; the only outstanding
  work is a benchmark run whose result determines whether to flip a
  default, ship a tuning change, or revert (#769, #697, #491).
- **`attn:decisions-needed`** — operator must adjudicate something
  before the issue can move. Sets it apart from `gate:operator`: the
  operator has all the information, just hasn't picked.

Issues carrying any of these labels surface in scanner inventory
output but are excluded from the "next actionable" list. Adding the
right label at file-time prevents the issue from being re-evaluated
on every fresh scan.

## What's likely to land where

The v1.x roadmap is bucketed. See [LIMITATIONS](docs/user/LIMITATIONS.md#known-issues-at-v10) for each issue's target version.

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

- Conventional-commit prefixes: `feat:`, `fix:`, `perf:`, `refactor:`, `test:`, `docs:`, `build:`, `ci:`, `style:`, `revert:`, `exp:`, `chore:`, `release:`, `gate:`, `audit:`.
- Atomic commits. Each commit moves the tree from one tested green state to another.
- Tests required for every behavioral change.
- `pyright --strict` must pass.

### Commit-message prefix enforcement

`scripts/check-commit-msg.py` validates that every commit subject starts with
an allowed conventional-commit prefix (`feat:`, `fix:`, etc., with optional
scope and `!`).  Install the local hook once after cloning:

```bash
sh scripts/setup-hooks.sh
```

This sets `core.hooksPath = .githooks` so the `commit-msg` hook runs
automatically.  Do not use `--no-verify`.

**CI also enforces prefixes** — the `commit-msg-prefix` job in
`.github/workflows/staging-gate.yml` checks every commit in the PR range.
It fails if any commit has an invalid prefix.  `Merge ` and `Revert ` subjects
generated by git are exempt.

### Pre-push branch-freshness check

`.githooks/pre-push` aborts a push when the branch's merge-base with
`origin/main` is older than the freshness threshold (default 4 hours).
This catches the parallel-session drift pattern where a feature branch
sits long enough that PRs land against a stale baseline. The same
`scripts/setup-hooks.sh` step above wires the hook in.

The threshold is configurable two ways:

- `AELF_PRE_PUSH_FRESHNESS_HOURS=24 git push ...` — one-shot per invocation.
- `git config aelfrice.prepushFreshnessHours 24` — repo-local default.

To bypass for a one-off emergency push:

```bash
ALLOW_STALE_BRANCH_PUSH=1 git push ...
```

The override emits a warning to stderr so the divergence is visible in
the transcript. Pushes to `main` itself and branch deletions are always
allowed.

### PR body requirements

The `pr-body-issue-link` CI job warns (but never blocks) when a PR body
contains no GitHub auto-close keyword (`Closes #N`, `Fixes #N`,
`Resolves #N`, etc.).  The warning is advisory — it helps keep
issue↔PR traceability intact.

If your PR legitimately has no associated issue (a dependency bump, a
release commit, a refactor with no issue filed), add the opt-out marker
anywhere in the PR body:

```html
<!-- no-issue -->
```

This silences the warning without requiring a fake issue link.

### Merging — the `ready-to-merge` label

`main` is FF-only and signature-required. To get a PR onto `main`:

1. Open the PR and let CI run.
2. When CI is green and you (or a reviewer) are satisfied, add the
   `ready-to-merge` label to the PR.

The `merge-train` workflow (`.github/workflows/merge-train.yml`)
serializes merges: it picks up labeled PRs one at a time, verifies the
branch is fast-forward on current `main` and all commits are signed,
waits for required checks to complete, and FF-pushes to `main`.
Concurrency-1 — no two merges race.

If the bot rejects the push it removes the label and posts a comment
explaining why. The most common cause is "branch is not fast-forward"
(another PR merged while yours was queued). Rebase locally
(`git rebase github/main`), force-push, and re-add the label.

The bot has no signing key, so it cannot rebase on your behalf
(see `.github/workflows/flag-stale-open-prs.yml` for the original
"no auto-rebase" rationale, #341). Authors rebase; the bot only FFs.

The PR-size soft-cap (`.github/workflows/pr-size-soft-cap.yml`) posts
an advisory comment on PRs over 200 LOC or 3 files. Smaller PRs are
less likely to lose the FF race; apply `size:override` for legitimate
large diffs (refactors, removals, generated code).

Both workflows shipped as part of #602.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). The short version: be respectful, focus on the work, no harassment.

## Security

See [SECURITY.md](SECURITY.md). Privacy bugs are treated as security bugs.
