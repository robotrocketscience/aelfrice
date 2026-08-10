# Contributing

Thanks for considering a contribution. aelfrice is a one-author project — the bar for changes is "is the system better afterward, in a way that's defensible by tests."

## Status

Current line: see [CHANGELOG.md](CHANGELOG.md) for the latest release and [ROADMAP](docs/concepts/ROADMAP.md) for the landing record. Issues are welcome. PRs are evaluated on a case-by-case basis — the bar is "moves the system measurably forward, justifies the change with a test."

Best categories of PR:

- Bug fixes with a regression test that fails before and passes after.
- Doc fixes (typo, broken link, stale claim against current code).
- Closing one of the [known limitations](docs/user/LIMITATIONS.md) — but ping in an issue first to align on approach.

Hard to land without prior alignment:

- New CLI subcommands.
- Schema changes.
- Anything that adds a hard dependency.
- Reintroducing earlier research-line features without a benchmark / experiment showing the impact.

## How to file a useful issue

Title format: a one-line description, in lowercase, ending without a period.

```
search: locked-belief order is unstable when budget is exhausted
```

Body should include:

- **What happened** — exact CLI invocation, exact output.
- **What you expected** — one-line.
- **Environment** — OS, Python version (`python --version`), aelfrice version, host agent (if relevant).
- **A minimal repro.** A directory you can `aelf onboard <here>` and reproduce, or the smallest sequence of CLI calls that triggers it.

Don't include your real memory.db (at `<repo>/.git/aelfrice/memory.db`, or `~/.aelfrice/memory.db` for non-git directories) — it contains your private beliefs. Reproduce on a scratch DB (`AELFRICE_DB=/tmp/scratch.db`) and share that.

## Triage labels

The issue tracker uses a small label vocabulary. The in-repo `aelf gate
list` aggregator (`src/aelfrice/gate_list.py`) reads `gate:ratify` /
`gate:prereq` / `bench-gated` / `gate:license` via `gh`; the remaining
labels are read by operator-side scan tooling (not shipped in this
repo) to decide whether an open issue is ready to be claimed or should
be hidden from the active queue. Apply
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

## Where to look for work

[ROADMAP](docs/concepts/ROADMAP.md) carries the version-by-version landing record plus the current active line. [LIMITATIONS](docs/user/LIMITATIONS.md) lists known gaps against current HEAD. The issue tracker is the canonical source of in-flight work.

Highest-leverage contributions tend to land in three places:

- **Bench gates.** Several `bench-gated` issues are waiting on benchmark runs to decide whether to flip a default or revert (see `gh issue list --label bench-gated`).
- **Triage drift.** Issues labeled `Blocked` whose blockers have since closed; surface a status flip rather than implement.
- **Stale-doc fixes.** Anything where the docs lie about the code at current HEAD (a docs audit lives in [docs/audits/](docs/audits/) — the latest pass enumerates what's outstanding).

## What's not on the path

- Vector embeddings or ANN in retrieval (would require a hard dep on a vector library; defeats the local-stdlib design).
- Cloud sync, accounts, or any non-local data path.
- A web UI.
- Integration with chat platforms.

## Development setup

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

### The bench-gate tier does not run here

A green `pytest tests/` on this repository does **not** mean the quality gates
passed. It means they were skipped.

The retrieval, compression and clustering quality gates under
`tests/bench_gate/` are marked `bench_gated` and skip unless
`AELFRICE_CORPUS_ROOT` points at a labelled evaluation corpus. That corpus is
private and this repository is public, so on every public CI run the whole tier
skips. The run prints a `bench-gate tier` summary line saying how many, so the
skips cannot be mistaken for passes.

**This is a deliberate disposition (#1420 §3), not an oversight.** The
self-hosted-runner alternative was considered and rejected: `ci.yml` is
`on: pull_request` and runs `uv run pytest tests/`, so a fork PR executes its
own test files on whatever host runs them — a self-hosted runner on a public
repo is arbitrary code execution by any fork author. The mitigations (one-shot
non-privileged container, network isolation, no repo secrets, an approving
label gate on fork PRs) are all real, but they amount to a standing security
commitment taken on to move one quality signal earlier. Not worth it at this
repository's size.

If you are changing retrieval ranking, compression, or clustering behaviour,
say so in the PR body and expect the quality evidence to come from a
corpus-bearing run rather than from CI being green.

### Changelog entries — one file per entry

**Add a file under `CHANGELOG/unreleased/`. Do not edit the
`[Unreleased]` block of `CHANGELOG/v4.md`.**

```
CHANGELOG/unreleased/<issue>-<slug>.md
```

```markdown
### Fixed

- **One-line title ([#1475](https://github.com/robotrocketscience/aelfrice/issues/1475)).** Body prose.
```

Exactly one `### <Category>` heading and exactly one top-level `- `
bullet per file. Category is one of `Added`, `Changed`, `Deprecated`,
`Removed`, `Fixed`, `Security`, `Performance`, `Documentation`,
`Build`, `CI`, `Dependencies`, `Internal`, `Reverted`, `Notes` — the
`CATEGORIES` list in `scripts/collate_changelog.py`, which is exactly
the set the committed changelogs already use. Indented
continuation paragraphs under the bullet are preserved verbatim.
`scripts/collate_changelog.py` refuses a file that breaks either rule
rather than guessing.

The directory is flat and holds nothing else. A different suffix
(`.txt`, `.markdown`, an uppercase `.MD`), an extensionless file or a
subdirectory is an error naming the path — in collation, in
`scripts/check_changelog_dupes.py` and in `release-docs-check` alike.
All three refuse rather than skip: a file collation will not collect
is one the release would omit without a word.

Why (#1475): entries are 2,000-4,500-character single lines, and
thirteen of fourteen open PRs were inserting them into the same
eight-line region. Every merge then forced a hand resolution on every
remaining PR, and the resolution — two 4 KB lines with no intra-line
granularity — can drop an entry without leaving a trace in the diff.
Two branches adding files at distinct paths never conflict.

**Transition.** The `[Unreleased]` block is still valid; collation
emits it first and then the files. A PR already editing that block does
not need to be rebased onto this convention to merge. New entries
should be files.

**At release time**, `scripts/collate_changelog.py` folds both into the
dated section of `CHANGELOG/v<major>.md` and empties the directory —
see [docs/concepts/RELEASING.md](docs/concepts/RELEASING.md).
`release-docs-check` fails a release PR that leaves either the block or
the directory undrained, and `scripts/check_changelog_dupes.py` compares
entry files against each other and against the block, so two PRs
restating the same fix in two files are still caught.

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
waits for the gating checks to complete, and FF-pushes to `main`.
Concurrency-1 — no two merges race.

"Gating" is wider than the five contexts branch protection marks
*required*. `scripts/merge_train_gate.py` blocks on any failing
non-advisory check-run, and its presence floor — the checks that must
have *reported at all*, not merely not-failed — is the required set plus
every check emitted by a `pull_request` workflow with no `paths:` filter
(#1458). So a head that never ran `migration-policy-check`, `typos` or
`bench-smoke` will not merge, even though none of those is required.
Path-filtered workflows are deliberately outside the floor: a docs-only
PR never runs `e2e` or `CodeQL`, and flooring on them would block it
forever.

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

### When a check never reports — the manual re-run hatch

Occasionally a required check simply never appears on a PR: GitHub throttles
webhook delivery during an incident, a run is deleted before it can be re-run,
or an `on:` block stops matching. The PR is then unmergeable *and* unfixable by
pushing, because pushing is exactly what is not being delivered.

`ci.yml` and `staging-gate.yml` — which between them carry all five required
contexts (`pytest (3.12)`, `pytest (3.13)`, `secrets-scan`, `pattern-scan`,
`history-scan`) — accept `workflow_dispatch`. Dispatch goes over the REST API
rather than the webhook path, so it still works when delivery is degraded:

```sh
gh workflow run ci.yml          --repo robotrocketscience/aelfrice --ref <your-branch>
gh workflow run staging-gate.yml --repo robotrocketscience/aelfrice --ref <your-branch>
gh run list --repo robotrocketscience/aelfrice --workflow ci.yml --limit 3
```

Three properties worth knowing, and the third is a caveat, not a feature.

- **A dispatch cannot report against a commit it did not test.** There is
  deliberately no `ref` *input*. A run's check-runs attach to the head SHA of
  the ref it was dispatched on, and both branch protection and `merge-train`
  evaluate checks on the PR's head SHA. (`actions/checkout` in these two
  workflows must likewise never pin a `ref:`, for the same reason; a test
  enforces both.)
- **A dispatched `ci.yml` always runs the full suite.** The `dorny/paths-filter`
  short-circuit is `pull_request`-only, because a dispatch has no diff base and
  a job that skips must never report a pass that looks like a run (#1160).
  Relatedly, no job in either workflow may be guarded to `pull_request` only: a
  guarded job still emits a check-run with conclusion `skipped` and a *later*
  `started_at`, and `merge_train_gate.latest_per_name` keeps the newest row per
  name while `skipped` is not a failing conclusion — so it would overwrite an
  earlier real `failure` and clear a red gate. The two jobs that genuinely
  cannot run outside a pull request live in `pr-metadata.yml`, which has no
  `workflow_dispatch`.
- **⚠️ Dispatching these two does *not* mean the PR is safe to label.** They
  produce the five *required* contexts, and `merge-train`'s presence floor
  (`missing`, #1435) is computed over the required set only. Every other gating
  check is evaluated by *absence tests*, which an absent check satisfies. So a
  head carrying only the dispatched rows evaluates as green while those never
  ran. Before labelling, list what is actually on the head SHA:

  ```sh
  gh api repos/robotrocketscience/aelfrice/commits/<head-sha>/check-runs \
      --jq '[.check_runs[] | {n: .name, c: .conclusion}]'
  ```

  Compare against the *check-run* names, which are job names and not workflow
  names — the two differ for most of these, and a list of workflow names would
  never match anything:

  | Workflow | Check-run name(s) |
  | --- | --- |
  | `migration-policy-check.yml` | `migration-policy-check` |
  | `typos.yml` | `typos` |
  | `bench-smoke.yml` | `bench-smoke` |
  | `deadcode.yml` | `deptry`, `vulture` |
  | `e2e.yml` | `e2e (pipx)`, `e2e (uv-tool)`, `e2e (venv-pip)` |
  | `codeql.yml` | `analyze (actions)`, `analyze (python)` |
  | `eval-calibration.yml` | `calibration` |
  | `windows-smoke.yml` | `smoke` |
  | `zizmor.yml` | `zizmor` |
  | `ci.yml` | `pytest (3.12)`, `pytest (3.13)` |
  | `staging-gate.yml` | `secrets-scan`, `pattern-scan`, `history-scan`, `commit-msg-prefix`, `release-docs-check` |
  | `pr-metadata.yml` | `pr-title-prefix`, `pr-body-issue-link` |
  | `pr-size-soft-cap.yml` | `size-check` |
  | `replay-soak-gate.yml` | `consecutive-green` |
  | `e2e.yml` | also `surface-failure` |
  | `auto-add-to-board.yml` | `add-to-board` |
  | `merge-train.yml` | `merge` |

  **`pr-title-prefix` and `pr-body-issue-link` are the two rows to check first
  after a dispatch-only recovery.** They live in `pr-metadata.yml`, which
  deliberately has no `workflow_dispatch` — so they are exactly the gating
  checks a dispatch cannot produce, and their absence is evaluated by an
  absence test that an absent check satisfies. If they are missing, the head is
  not safe to label no matter how green the dispatched rows look.

  Not all of them belong on every head, so an absent row is not automatically a
  problem. `windows-smoke.yml`, `eval-calibration.yml`, `e2e.yml` and
  `zizmor.yml` carry workflow-level `paths:` filters and `deadcode.yml` and
  `codeql.yml` carry `paths-ignore:`, so those check-runs are legitimately
  missing on a head that touches none of their paths — `smoke`, for instance,
  only appears when the PR touches `src/**`, `tests/test_windows_portability_1329.py`,
  `pyproject.toml` or its own workflow file. Read the `on:` block before
  concluding a row should have been there. One extra name to expect:
  code scanning posts its own `CodeQL` check-run from the
  `github-advanced-security` app alongside the two `analyze (…)` jobs.

  Widening the presence floor to cover the whole gating set is tracked
  separately — this section is the interim instruction, not the fix.

Shipped as part of #1436.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). The short version: be respectful, focus on the work, no harassment.

## Security

See [SECURITY.md](SECURITY.md). Privacy bugs are treated as security bugs.
